from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.util import TensixInstructionDecoder
from tt_sim.perf.model import unit_cost_model
from tt_sim.util.bits import extract_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TensixSyncUnit(TensixBackendUnit, MemMapable):
    """
    The sync unit provides mutex and semaphore functionality to instruct the waitgate
    to block the issuing of instructions to the backend based on specific conditions.
    Also provides the STALLWAIT which is more generic.

    Based on descriptions and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SyncUnit.md
    """

    class TTSemaphore:
        def __init__(self):
            self.value = 0
            self.max = 0

    class TTMutex:
        def __init__(self):
            self.held_by = None

    OPCODE_TO_HANDLER = {
        "SEMINIT": "handle_seminit",
        "STALLWAIT": "handle_stallwait",
        "SEMWAIT": "handle_semwait",
        "SEMPOST": "handle_sempost",
        "SEMGET": "handle_semget",
        "ATGETM": "handle_atgetm",
        "ATRELM": "handle_atrelm",
    }

    #: Width of ``STALLWAIT``'s ``wait_res`` condition mask **on Blackhole**
    #: (raw bits 12:0, conditions C0-C12). Wormhole's is 15 bits (14:0) and is
    #: what the shared, Wormhole-shaped ``tensix_instructions.yaml`` already
    #: decodes, so only this one has to be re-read from the raw word -- without
    #: which bits 14:13 leak in and select spurious conditions. See
    #: :meth:`_read_wait_res` for why 13 rather than the 12 tt-sim used to read,
    #: and for the source conflict behind the difference.
    BLACKHOLE_WAIT_RES_BITS = 13

    def __init__(self, backend):
        super().__init__(backend, TensixSyncUnit.OPCODE_TO_HANDLER, "Sync")
        # Phase 5 of docs/plans/event-driven-pump.md. ``None`` unless
        # TT_SIM_COST_MODEL is set. Every Sync Unit instruction is one cycle in
        # both arches' tables, so this charges nothing new -- the interesting
        # costs of this unit are not occupancy at all: SEMWAIT / STALLWAIT
        # "consist purely of passing them over to the Wait Gate", so the wait
        # is unbounded and belongs to the gate, and ATGETM's stall on a held
        # mutex is likewise a gate cost. tt-sim already models both
        # functionally.
        self.cost_model = unit_cost_model(
            "SYNC", "blackhole" if backend.blackhole else "wormhole"
        )
        self.semaphores = [TensixSyncUnit.TTSemaphore() for i in range(8)]
        # 7 mutexes, but index 1 is ignored
        self.mutexes = [TensixSyncUnit.TTMutex() for i in range(8)]
        self.blocked_mutex = []

    def issueInstruction(self, instruction, from_thread):
        # An occupied IPC group takes nothing; see TensixBackendUnit.is_occupied.
        # No Sync Unit op is costed above one cycle, so this never fires today.
        # It also has no published IPC groups -- its page describes two
        # throughput classes in prose (mutex ops "up to three per cycle,
        # provided they refer to different mutexes" against the semaphore ops'
        # shared one) but tabulates no group column, so ``issue_group`` returns
        # None and the hold is whole-unit. Unobservable either way while every
        # entry costs one cycle.
        if self.busy_until is not None and self.is_occupied(
            self.issue_group(instruction)
        ):
            return self._refuse("unit_busy")
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        instruction_name = instruction_info["name"]
        if instruction_name == "ATGETM" or instruction_name == "ATRELM":
            # Per SyncUnit.md's throughput table, the mutex ops "issue up to
            # three per cycle, provided they refer to different mutexes",
            # whereas the semaphore ops issue "at most one of these per cycle"
            # (the else arm). This branch used to test for "ATGEM", which the
            # instruction table does not contain, so every ATGETM fell to the
            # else arm and took the queue exclusively -- both rules were dead
            # code.
            if len(self.next_instruction) < 3:
                index = instruction_info["instr_args"]["mutex_index"]
                for instr, _ in self.next_instruction:
                    instruction_n_info = TensixInstructionDecoder.getInstructionInfo(
                        instr
                    )
                    # Only the mutex ops name a mutex. The queue can also hold
                    # one op that does not (the else branch below admits one
                    # while the queue holds nothing else), and reading
                    # ``mutex_index`` off that raised KeyError -- found by the
                    # RISC-V cost model, which delays the drain enough for a
                    # mixed queue to occur on ``loopback``. Correcting the
                    # opcode name above made that path reachable with the cost
                    # model *off* too: two Blackhole guards issue an ATGETM
                    # while a SEMPOST / SEMWAIT is still queued. A queued
                    # semaphore op cannot conflict over a mutex it does not
                    # reference (the doc's throughput limits are per row, not
                    # shared across the unit), so
                    # skipping it is the correct answer as well as the safe
                    # one. Keyed off the presence of the field rather than off
                    # the opcode name so it stays right for any future queued
                    # op that names no mutex.
                    if "mutex_index" not in (
                        instruction_n_info.get("instr_args") or {}
                    ):
                        continue
                    if instruction_n_info["instr_args"]["mutex_index"] == index:
                        # Same mutex referenced, do not issue this cycle
                        return self._refuse("issue_slot_taken")
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                # Three or more already issued, do not issue this cycle
                return self._refuse("issue_slot_taken")
        else:
            # Only one of any other instruction allowed
            if not self.checkIfNextInstructionsContainAnyOtherOpcodes(
                "ATGETM", "ATRELM"
            ):
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                return self._refuse("issue_slot_taken")

    def is_clock_idle(self):
        # A queued mutex waiter is retried every cycle by the override below.
        return not self.next_instruction and not self.blocked_mutex

    def clock_tick(self, cycle_num):
        super().clock_tick(cycle_num)
        to_remove = []
        for idx, (issue_thread, index) in enumerate(self.blocked_mutex):
            if (
                self.mutexes[index].held_by is None
                or self.mutexes[index].held_by == issue_thread
            ):
                self.mutexes[index].held_by = issue_thread
                self.backend.getFrontendThread(
                    issue_thread
                ).wait_gate.informMutexAcquired()
                to_remove.append(idx)
        # Every granted waiter leaves the queue, not just the second and
        # subsequent ones. Deleting only when ``len(to_remove) > 1`` left a
        # lone grant queued forever, and a queued entry is re-granted on every
        # later tick -- so the mutex was pinned to that thread for the rest of
        # the device's life, silently re-acquired the cycle after each ATRELM.
        # Nothing failed while no other thread wanted it; the first ATGETM from
        # another thread then blocked for ever (a live spin, not a stall: the
        # queue also keeps the unit non-idle). That is what made a second
        # ``LaunchProgram`` in one process hang once the first had contended
        # for a mutex. Delete back-to-front so the surviving indices stay
        # valid.
        for idx in reversed(to_remove):
            del self.blocked_mutex[idx]

    def getSemaphore(self, idx):
        assert idx <= 7
        return self.semaphores[idx]

    def handle_atrelm(self, instruction_info, issue_thread, instr_args):
        index = instr_args["mutex_index"]
        if self.mutexes[index].held_by == issue_thread:
            # If I own the mutex then release it, otherwise ignore
            self.mutexes[index].held_by = None

    def handle_atgetm(self, instruction_info, issue_thread, instr_args):
        index = instr_args["mutex_index"]
        if (
            self.mutexes[index].held_by is not None
            and self.mutexes[index].held_by != issue_thread
        ):
            self.blocked_mutex.append((issue_thread, index))
        else:
            self.mutexes[index].held_by = issue_thread
            self.backend.getFrontendThread(issue_thread).wait_gate.informMutexAcquired()

    def handle_sempost(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        for i in range(8):
            if get_nth_bit(sem_sel, i) and self.semaphores[i].value < 15:
                self.semaphores[i].value += 1

    def handle_seminit(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        new_value = instr_args["init_value"]
        max_value = instr_args["max_value"]

        for i in range(8):
            if get_nth_bit(sem_sel, i):
                self.semaphores[i].value = new_value
                self.semaphores[i].max = max_value

    def handle_semget(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        for i in range(8):
            if get_nth_bit(sem_sel, i) and self.semaphores[i].value > 0:
                self.semaphores[i].value -= 1

    def _read_wait_res(self, instruction_info, instr_args):
        """``STALLWAIT``'s condition mask, at this architecture's field width.

        THE TWO SOURCES DISAGREE ABOUT BLACKHOLE, and this reads the ISA
        documentation rather than the vendor data file. Recorded here in full
        because the disagreement is the only reason the constant is not
        obvious, and because tt-sim followed the other side of it until
        2026-08-12.

        **13 bits (raw 12:0), which is what this returns.** The BlackholeA0
        ``STALLWAIT.md`` page says so three independent ways: its syntax line is
        ``TT_STALLWAIT(/* u9 */ BlockMask, /* u13 */ ConditionMask)``; its
        encoding diagram (``Diagrams/Out/Bits32_STALLWAIT_BH.svg``) gives
        ConditionMask bits 0-12, BlockMask 15-23 and the opcode 24-31; and its
        condition table has thirteen rows, C0 through C12, the last of which is
        "Any thread has an instruction in any stage of the Configuration Unit
        pipeline". A fourth source agrees and is not an ISA document at all:
        tt-metal's own Blackhole LLK header
        (``tt_llk_blackhole/common/inc/ckernel_instr_params.h``) defines
        ``p_stall::CFGEXU = 0x1000`` -- bit 12 -- alongside the twelve below it,
        which a 12-bit field could not hold.

        **12 bits (raw 11:0), which this used to return.** ttsim's
        ``data/bh/tensix_isa.json`` records ``STALLWAIT/args/wait_res`` as
        ``"11:0"``. That is a single statement, in a data file, and ttsim's own
        executor contradicts it: ``src/tensix.cpp``'s ``TT_ARCH_VERSION == 1``
        branch clears the bits it handles and then hands the residual --
        which still includes ``0x1000`` -- to
        ``TTSIM_VERIFY(!wait_res, UnimplementedFunctionality)``. Its execution
        path is written as though bit 12 could arrive; its decoder makes that
        impossible. An internal inconsistency on one side and four agreeing
        statements on the other is not a close call, and the repo's own
        provenance ladder ranks ``isa_doc`` above ``vendor_source`` for exactly
        this situation.

        **What changing it can and cannot do.** Bits 14:13 stay excluded, so the
        leak this reader was written to stop is still stopped -- only bit 12,
        which the page says is part of the field, comes back. No in-tree kernel
        sets it (nothing in tt-metal 0.74 references ``p_stall::CFGEXU``), so no
        captured trace moves. And the behaviour it replaces was not a shorter
        wait but the *wrong* wait: a ``STALLWAIT`` whose only condition was C12
        had its mask trimmed to zero and fell through to the ``0x7F`` "all
        resources" default, which on Blackhole means C0-C6 -- a different set of
        conditions, not a subset.

        **If a card ever shows bit 12 is inert**, this is one constant,
        :attr:`BLACKHOLE_WAIT_RES_BITS`; the citations above are what would have
        to be weighed against that measurement, and per the cost tables' "why
        there is no ``measured`` provenance" note a measurement would not
        silently win.
        """
        if not self.backend.blackhole:
            return instr_args["wait_res"]
        return extract_bits(
            instruction_info["raw_instruction"], self.BLACKHOLE_WAIT_RES_BITS, 0
        )

    def handle_stallwait(self, instruction_info, issue_thread, instr_args):
        cond_mask = self._read_wait_res(instruction_info, instr_args)
        block_mask = instr_args["stall_res"]

        self.backend.getFrontendThread(
            issue_thread
        ).wait_gate.setLatchedWaitInstruction(
            "STALLWAIT",
            cond_mask if cond_mask else 0x7F,
            block_mask if block_mask else 1 << 6,
        )

    def handle_semwait(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        cond_mask = instr_args["wait_sem_cond"]
        block_mask = instr_args["stall_res"]
        block_mask = block_mask if block_mask else 1 << 6

        if cond_mask:
            self.backend.getFrontendThread(
                issue_thread
            ).wait_gate.setLatchedWaitInstruction(
                "SEMWAIT", cond_mask, block_mask, sem_sel
            )
        else:
            self.backend.getFrontendThread(
                issue_thread
            ).wait_gate.setLatchedWaitInstruction("STALLWAIT", 0x7F, block_mask)

    def read(self, addr, size):
        # Accesses semaphore[i].value, where each
        # entry is 32 bit
        idx = int(addr / 4)
        assert idx < 8
        return conv_to_bytes(self.semaphores[idx].value)

    def write(self, addr, value, size=None):
        """
        This is taken from the functional model code at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SyncUnit.md#semaphores
        """
        idx = int(addr / 4)
        assert idx < 8
        if conv_to_uint32(value) & 1:
            # This is like a SEMGET instruction
            if self.semaphores[idx].value > 0:
                self.semaphores[idx].value -= 1
        else:
            # This is like a SEMPOST instruction
            if self.semaphores[idx].value < 15:
                self.semaphores[idx].value += 1

    def getSize(self):
        return 0xFFDF
