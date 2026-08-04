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
            return False
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
                        return False
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                # Three or more already issued, do not issue this cycle
                return False
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
                return False

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
        # STALLWAIT's ``wait_res`` condition mask is 12 bits on Blackhole (raw
        # bits 11:0, conditions C0-C11) but 15 bits on Wormhole (14:0). The
        # shared instruction table encodes the Wormhole width, so on Blackhole
        # bits 14:12 - which are not part of the field there - leak into the
        # mask and select spurious conditions. Read the true field from the raw
        # word on Blackhole (widths from ttsim's data/{bh,wh}/tensix_isa.json).
        # Blackhole's LLK does define one condition above the field - CFGEXU,
        # bit 12, which WaitGate models as C12 - but never emits it; such a
        # wait now falls back to the "wait for all resources" 0x7F mask below.
        # See docs/plans/blackhole-support.md (Phase 8).
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 12, 0)
        return instr_args["wait_res"]

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
