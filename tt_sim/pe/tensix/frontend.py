from abc import ABC

from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import MemoryStall
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.util import TensixInstructionDecoder
from tt_sim.trace import DispatchEvent, EventCategory, StallEvent, get_bus
from tt_sim.util.bits import extract_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_uint32

#: Instructions one thread may have in flight inside its frontend (MOP FIFO +
#: replay FIFO + wait-gate FIFO) before a *core* push is refused and the
#: issuing baby RISC-V core stalls on its ``.ttinsn`` store / ``sw`` to the
#: instruction push buffer.
#:
#: This is a **conservative, uncalibrated mechanism bound — not a silicon
#: calibration**. Blackhole silicon measures ~31–32 instructions absorbed at
#: one issuing thread (``docs/plans/riscv-front-end-benchmark.md``, phase Q's
#: untimed-drain estimator), and that figure is a *lower bound* on the true
#: per-thread depth (the backlog was still growing at the longest burst run).
#: 64 is chosen to sit safely above that lower bound so the model never
#: invents back-pressure the hardware does not have (the cost model's
#: charge-at-the-low-end direction), while still being finite — which is the
#: whole point: a permanently blocked backend instruction now wedges the
#: issuing core, exactly as on silicon, instead of the kernel running to
#: completion past a wedged unit (ROADMAP item 1, the ``UNPACR_NOP``
#: acquire-without-release case). Do not present this number as measured; when
#: the longer burst sweep runs on a card, calibrate it there.
#:
#: The count deliberately includes the MOP/replay expansions already inside
#: the frontend, not just words the core pushed: tt-sim's expanders emit a
#: whole template in one tick rather than one instruction per cycle, so the
#: expansion products are the only honest proxy for the work in flight ahead
#: of the next push. Internal pushes (expander output) are never refused —
#: only the core-facing write is.
CORE_PUSH_INFLIGHT_BOUND = 64


class TensixFrontend(MemMapable):
    """
    Tensix unit frontend, we have a frontend per thread and there are three threads.
    Each frontend contains a MOP expander, replay expander and wait gate.

    Based on description at
    https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor
    """

    def __init__(self, thread_id, backend, diags_settings, blackhole_conditions=False):
        self.thread_id = thread_id
        self.backend = backend
        self.mop_instruction_fifo = []
        self.replay_instruction_fifo = []
        self.wait_gate_instruction_fifo = []
        self.mop_expander = TensixMOPExpander(self)
        self.replay_expander = TensixReplayExpander(self)
        self.wait_gate = WaitGate(self, backend, blackhole_conditions)
        self.diags_settings = diags_settings
        self.unit_id: tuple | None = None

    def getClocks(self):
        return [self.mop_expander, self.replay_expander, self.wait_gate]

    def getDiagnosticSettings(self):
        return self.diags_settings

    def hasInflightInstructions(self):
        return (
            len(self.mop_instruction_fifo) > 0
            or len(self.replay_instruction_fifo)
            or len(self.wait_gate_instruction_fifo)
        )

    def inflight_count(self):
        """Instructions queued anywhere in this thread's frontend path."""
        return (
            len(self.mop_instruction_fifo)
            + len(self.replay_instruction_fifo)
            + len(self.wait_gate_instruction_fifo)
        )

    def MOPExpanderDoneCheck(self):
        return len(self.mop_instruction_fifo) > 0

    def pop_mop_instruction(self):
        if len(self.mop_instruction_fifo) > 0:
            return self.mop_instruction_fifo.pop(0)
        else:
            return None

    def push_mop_instruction(self, instruction):
        return self.mop_instruction_fifo.append(instruction)

    def pop_replay_instruction(self):
        if len(self.replay_instruction_fifo) > 0:
            return self.replay_instruction_fifo.pop(0)
        else:
            return None

    def push_replay_instruction(self, instruction):
        return self.replay_instruction_fifo.append(instruction)

    def pop_wait_gate_instruction(self):
        if len(self.wait_gate_instruction_fifo) > 0:
            return self.wait_gate_instruction_fifo.pop(0)
        else:
            return None

    def inspect_wait_gate_instruction(self):
        if len(self.wait_gate_instruction_fifo) > 0:
            return self.wait_gate_instruction_fifo[0]
        else:
            return None

    def push_wait_gate_instruction(self, instruction):
        return self.wait_gate_instruction_fifo.append(instruction)

    def getMOPExpander(self):
        return self.mop_expander

    def read(self, addr, size):
        raise NotImplementedError(
            f"Reading at {hex(addr)} from tensix thread {self.thread_id}"
        )

    def write(self, addr, value, size=None):
        instruction = conv_to_uint32(value)
        if TensixInstructionDecoder.isInstructionRecognised(instruction):
            # Bounded, so a backed-up path into the backend back-pressures the
            # issuing baby core: :class:`MemoryStall` propagates up through the
            # memory dispatch and both push paths — the ``.ttinsn`` extension
            # (``pe/rv/isa/tt_isa.py``) and a plain ``sw`` to the push buffer
            # (``handle_s_store``) — turn it into a ``PEStall``, so the core
            # retries the same store on the next cycle with the PC unmoved.
            # On hardware this is the instruction FIFO filling; see
            # :data:`CORE_PUSH_INFLIGHT_BOUND` for what the bound is and is not.
            if self.inflight_count() >= CORE_PUSH_INFLIGHT_BOUND:
                return MemoryStall
            self.push_mop_instruction(instruction)
        else:
            opcode = extract_bits(instruction, 8, 24)
            raise NotImplementedError(
                f"Unknown op code issued {hex(opcode)} on tensix thread {self.thread_id}"
            )

    def getSize(self):
        return 0xFFFF


class TensixFrontendUnit(Clockable, ABC):
    def __init__(self, frontend):
        self.frontend = frontend

    def getDiagnosticSettings(self):
        return self.frontend.getDiagnosticSettings()


class WaitGate(TensixFrontendUnit):
    """
    The wait gate will pause instructions based on SEMWAIT, mutex or semaphores
    as well as some other conditions such as math instructions not having
    srcA or srcB bank ready for them.

    This is based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/WaitGate.md
    """

    # The math instructions to check allowedclient for, first element is
    # list of applicable instructions that need to check srcA, second element
    # is list of applicable instructions that need to check srcB.
    #
    # Every name here must exist in ``tensix_instructions.yaml`` -- the check is
    # a string membership test against the decoded opcode, so a misspelling
    # silently removes an instruction from the gate rather than failing. That
    # happened: ``TRNSPSRCB`` was spelled ``TRANSPSRCB`` and so never waited for
    # SrcB, where ttsim's ``TENSIX_EXECUTE_TRNSPSRCB`` stalls until the FPU owns
    # the bank. ``waitgate_allowed_client_test.py`` now pins the whole list
    # against the instruction table.
    #
    # ``MOVDBGA2D`` is deliberately *absent* from the SrcA group, and the
    # absence is the documented behaviour rather than an oversight:
    # ``MOVDBGA2D.md`` opens "This instruction is identical to ``MOVA2D``,
    # except that it doesn't _automatically_ wait for
    # ``SrcA[MatrixUnit.SrcABank].AllowedClient == MatrixUnit``", and its
    # functional model says "See ``MOVA2D``'s functional model, ignoring the
    # paragraph about the Wait Gate" -- that paragraph being the
    # ``while (SrcA[...].AllowedClient != MatrixUnit) wait;`` loop this list
    # implements. Gating it charged a stall the hardware does not take, which
    # is the one direction the floor policy forbids. Unreachable today
    # (``matrix.py`` has no handler, so issuing one raises), so this moves no
    # number; ``test_movdbga2d_is_not_gated`` pins it either way.
    MATH_ALLOWED_CLIENT_INSTRUCTIONS = [
        (
            "MVMUL",
            "DOTPV",
            "GAPOOL",
            "GMPOOL",
            "ELWMUL",
            "ELWADD",
            "ELWSUB",
            "MOVA2D",
            "SHIFTXA",
        ),
        (
            "MVMUL",
            "DOTPV",
            "GAPOOL",
            "GMPOOL",
            "ELWMUL",
            "ELWADD",
            "ELWSUB",
            "MOVB2D",
            "MOVB2A",
            "SHIFTXB",
            "TRNSPSRCB",
        ),
    ]

    class LatchedInstruction:
        BLOCKED_INSTRUCTION_TYPES = [
            (
                "TDMA",
                "XMOV",
                "THCON",
                "PACK",
                "UNPACK",
            ),
            ("SYNC",),
            ("PACK",),
            ("UNPACK",),
            ("XMOV",),
            ("THCON",),
            ("MATH",),
            ("CFG",),
            ("SFPU",),
        ]

        def __init__(self, opcode, condition_mask, block_mask, semaphore_mask=None):
            self.opcode = opcode
            self.condition_mask = condition_mask
            self.block_mask = block_mask
            self.semaphore_mask = semaphore_mask

        def doesInstructionMatchBlockMask(self, instruction_info):
            if instruction_info["name"] == "STALLWAIT":
                # ``STALLWAIT`` is the one instruction *every* block bit
                # catches: its row in ``STALLWAIT.md``'s "exact set of
                # instructions blocked from starting by each bit" table is
                # ticked in all nine columns, on both architectures. It is
                # therefore held at the Wait Gate by any latched wait, however
                # narrow that wait's block mask, and cannot overwrite the
                # latch until the latch has been forgotten. The block mask is
                # never empty by the time it is latched -- both handlers
                # substitute ``1 << 6`` for a zero one, per the functional
                # models -- so any live latch blocks it.
                #
                # Modelling it is what keeps the *previous* wait honoured.
                # ``SEMWAIT.md`` warns that "a ``SEMWAIT`` instruction can
                # execute whilst there is still a latched wait instruction in
                # place, and latching of the new ``SEMWAIT`` will cause the
                # previous wait to be forgotten" -- true, and modelled, but it
                # is a warning *about* ``SEMWAIT`` (block bit B1 only)
                # precisely because ``STALLWAIT`` is immune to it.
                #
                # Reaching it through the per-unit table below instead is what
                # went wrong: ``STALLWAIT``'s ``ex_resource`` is ``SYNC``, so
                # only B1 caught it, and a ``STALLWAIT`` issued behind an
                # unsatisfied ``SEMWAIT`` whose block mask was B0 sailed
                # through and dropped that ``SEMWAIT``. That is exactly the
                # shape tt-metal's ``pack_untilize_dest_init`` has when it is
                # called *after* ``tile_regs_wait()``: the ``SEMWAIT`` on
                # ``MATH_PACK`` (B0) is still unsatisfied when
                # ``_llk_init_packer_dest_offset_registers_``'s
                # ``STALLWAIT(STALL_TDMA | STALL_THCON, PACK)`` arrives, so the
                # packer stopped waiting for the math and packed a Dst that was
                # still mid-accumulation.
                #
                # ``NOP`` is the table's other row that the per-unit walk
                # cannot express ("❌ if all bits of the block mask are set").
                # It is deliberately left alone: a ``NOP`` has no effect, and
                # an all-bits block mask blocks whatever follows it anyway, so
                # holding one changes cycle counts and nothing else.
                return True
            tgt_backend_unit = instruction_info["ex_resource"]
            for bit_idx in range(9):
                do_check = get_nth_bit(self.block_mask, bit_idx)
                if do_check:
                    if (
                        tgt_backend_unit
                        in WaitGate.LatchedInstruction.BLOCKED_INSTRUCTION_TYPES[
                            bit_idx
                        ]
                    ):
                        return True
            return False

        def getConditionCheck(self, idx):
            return get_nth_bit(self.condition_mask, idx)

        def isSemaphoreMode(self):
            if self.opcode == "SEMWAIT":
                assert self.semaphore_mask > 0
                return True
            return False  # is STALLWAIT

        def getSemaphoresToCheck(self):
            sem_check = []
            for i in range(8):
                if get_nth_bit(self.semaphore_mask, i):
                    sem_check.append(i)
            return sem_check

    #: Which backend unit each STALLWAIT condition bit is *about*, so a stall on
    #: it can name the unit responsible rather than a bare condition number.
    #: Indices follow :meth:`check_for_semwait_condition_match` (Wormhole) and
    #: :meth:`_check_blackhole_condition` (Blackhole) exactly -- the two layouts
    #: genuinely differ, which is why there are two maps. A condition absent
    #: from a map is one that names no single unit (Wormhole's C0/C13, the two
    #: "memory request outstanding" bits), and stalls on it carry no
    #: ``blocked_on``.
    STALLWAIT_COND_UNIT_WH = {
        1: "UNPACK",
        2: "UNPACK",
        3: "PACK",
        4: "PACK",
        5: "PACK",
        6: "PACK",
        7: "MATH",
        8: "UNPACK",
        9: "UNPACK",
        10: "MATH",
        11: "MATH",
        12: "XMOV",
        14: "SFPU",
    }
    STALLWAIT_COND_UNIT_BH = {
        0: "THCON",
        1: "UNPACK",
        2: "UNPACK",
        3: "PACK",
        4: "MATH",
        5: "UNPACK",
        6: "UNPACK",
        7: "MATH",
        8: "MATH",
        9: "XMOV",
        11: "SFPU",
        12: "CFG",
    }

    def __init__(self, frontend, backend, blackhole_conditions=False):
        super().__init__(frontend)
        self.mutex_stall = False
        self.backend_enforced_stall = False
        self.latchedWaitInstruction = None
        self.latch_wait = False
        self.backend = backend
        # Open stall episode, or None. See ``_note_stall``: a stall is published
        # once per contiguous span rather than once per cycle, so this holds the
        # span's first cycle, its (reason, blocked_on, opcode, semaphore) key
        # and the last cycle seen on it. All three stay None/-1 for ever unless
        # the STALL trace category is enabled, which is what keeps an untraced
        # run's cost to the single ``is not None`` test on the accept path.
        self._stall_since = None
        self._stall_key = None
        self._stall_last = -1
        # The tile's RISCV_DEBUG_REG_PERF_CNT_* block, set by
        # ``TensixCoProcessor.__init__``. Unlike the trace bus this is armed by
        # the *kernel* -- nothing accumulates until a start edge is written to
        # PERF_CNT_INSTRN_THREAD2 -- so an unprofiled run pays one attribute
        # read and one boolean test on paths that were already doing more.
        self.perf_counters = None
        # The STALLWAIT/SEMWAIT condition-mask bit assignments differ between
        # Wormhole and Blackhole (compare the two arch's STALLWAIT.md). When set,
        # ``check_for_semwait_condition_match`` uses the Blackhole layout.
        self.blackhole_conditions = blackhole_conditions

    def _note_stall(
        self, cycle_num, reason, blocked_on="", opcode="", semaphore=-1, src_bank=None
    ):
        """Record one stalled cycle into the open episode, opening one if needed.

        Called from every branch of :meth:`clock_tick` that declines to release
        the head instruction. The bus check comes first and is the only cost an
        untraced run pays here -- one global read and one dict lookup, on a
        branch that was already doing strictly more work than that (a semaphore
        comparison, a Src-bank ownership test, a refused issue). Nothing is
        allocated and no episode state is written when the category is off, so
        :attr:`_stall_since` stays ``None`` and the accept path's ``is not
        None`` test stays false for the whole run.

        A change of *any* key component closes the previous episode and opens a
        new one, so a thread that moves from waiting on the unpackers to waiting
        on a semaphore is two episodes, not one blurred span.
        """
        # The hardware perf counters come first and are independent of the
        # trace bus: ``THREAD_STALLS_n`` is a register a kernel reads, so it
        # must count whether or not anybody asked for a trace, and it must
        # count on a tile whose ``unit_id`` was never assigned.
        counters = self.perf_counters
        if counters is not None and counters.instrn_running:
            counters.note_stall(self.frontend.thread_id, reason, src_bank)
        unit_id = self.frontend.unit_id
        if unit_id is None:
            return
        bus = get_bus()
        if not bus.is_enabled(EventCategory.STALL):
            return
        key = (reason, blocked_on, opcode, semaphore)
        if self._stall_key != key:
            if self._stall_since is not None:
                self._flush_stall(bus)
            self._stall_since = cycle_num
            self._stall_key = key
        self._stall_last = cycle_num

    def _flush_stall(self, bus=None):
        """Publish the open episode, if any, and close it.

        Called when the gate makes progress (an instruction issued, or a latched
        wait's condition was met) and, via :meth:`_note_stall`, when the reason
        changes under it. A mutex or backend-enforced stall needs no explicit
        flush: the tick after it clears either accepts -- which flushes -- or
        stalls for a different reason, which is a key change. A run that ends
        mid-stall leaves its final episode unpublished; that is one span at the
        very end of a trace, and coalescing is worth more than chasing it.
        """
        since = self._stall_since
        if since is None:
            return
        reason, blocked_on, opcode, semaphore = self._stall_key
        self._stall_since = None
        self._stall_key = None
        if bus is None:
            bus = get_bus()
        bus.publish(
            StallEvent(
                cycle=since,
                unit_id=self.frontend.unit_id,
                reason=reason,
                blocked_on=blocked_on,
                cycles=self._stall_last - since + 1,
                opcode=opcode,
                thread_id=self.frontend.thread_id,
                semaphore=semaphore,
            )
        )

    def _note_latched_wait(self, cycle_num):
        """Attribute a cycle spent under a latched SEMWAIT / STALLWAIT.

        The reason is read out of the latched instruction's own masks, which is
        the whole reason this is expressible: the hardware's wait conditions
        *are* a named vocabulary (STALLWAIT.md tabulates one unit per condition
        bit), so the model does not have to guess what the thread is waiting
        for -- it is written in the instruction.
        """
        latched = self.latchedWaitInstruction
        if latched is None:
            return
        if latched.isSemaphoreMode():
            found = self._latched_semaphore_reason()
            if found is not None:
                reason, sem = found
                self._note_stall(cycle_num, reason, semaphore=sem)
            return
        cond_units = (
            WaitGate.STALLWAIT_COND_UNIT_BH
            if self.blackhole_conditions
            else WaitGate.STALLWAIT_COND_UNIT_WH
        )
        for idx in range(15):
            if latched.getConditionCheck(idx) and not (
                self.check_for_semwait_condition_match(idx)
            ):
                self._note_stall(
                    cycle_num, "resource_wait", blocked_on=cond_units.get(idx, "")
                )
                return

    def _latched_semaphore_reason(self):
        """The first unsatisfied semaphore condition of the latched wait.

        ``(reason, semaphore)``, or ``None`` when the latched wait is not in
        semaphore mode or every selected semaphore condition is satisfied.
        Shared by all three readers of those masks so they cannot drift:
        :meth:`check_for_wait_condition_met`, which turns ``None`` into a
        release; :meth:`_note_latched_wait`, which turns a reason into a
        stall; and :meth:`_tick_unheld_latched_wait`, which turns one into a
        live wait condition with nothing held.

        A ``None`` therefore carries the release decision as well as the
        absence of a reason, and the loop below is the whole quantifier: it
        runs to exhaustion over *every* selected semaphore, because
        ``SEMWAIT.md``'s condition mask says to keep waiting if **any**
        selected semaphore is zero (C0) or at its max (C1).

        First unsatisfied condition wins, which is the rule the whole gate
        already attributes by rather than a new one: a cycle on which C0 and C1
        are both unsatisfied, on different semaphores, is charged to the first.
        The hardware's two reason counters are independent signals and would
        both count -- the same simultaneity ``note_stall``'s docstring records
        for stalls, unchanged here and not closed by this method.
        """
        latched = self.latchedWaitInstruction
        if latched is None or not latched.isSemaphoreMode():
            return None
        for sem in latched.getSemaphoresToCheck():
            semaphore = self.backend.getSyncUnit().getSemaphore(sem)
            if latched.getConditionCheck(0) and semaphore.value == 0:
                return "semaphore_empty", sem
            if latched.getConditionCheck(1) and semaphore.value >= semaphore.max:
                return "semaphore_full", sem
        return None

    def _tick_unheld_latched_wait(self, cycle_num):
        """Re-evaluate a latched wait on a cycle with nothing held at the gate.

        ``SEMWAIT.md``: "The Wait Gate will then continuously re-evaluate the
        latched wait instruction until all of the selected conditions are
        simultaneously met, at which point the latched wait instruction will be
        forgotten". Re-evaluation is not conditional on anything being blocked
        -- "the issuing thread can continue execution until one of the blocked
        instructions is reached" -- so this is the evaluation the held path
        does, run on the cycles the held path never sees. Two things follow,
        and only the second is a counter:

        * the latch is **forgotten** the moment its condition is met, whether
          or not anything was ever blocked by it. tt-sim previously cleared a
          latch only on the held path, so a wait that nothing happened to block
          stayed armed for the rest of the run and could still block a later
          instruction the hardware had long since forgotten the wait for;
        * every cycle the condition is *not* met counts on the wait-condition
          counter and **not** as a stall -- nothing is held, so the thread lost
          no cycle. That is the counting rule
          :meth:`~tt_sim.misc.perf_counters.TensixPerfCounters.note_wait_condition`
          exists for, and the reason a hardware reason counter can outrun
          ``THREAD_STALLS_n``.

        Only the semaphore reasons are counted. A latched ``STALLWAIT``'s
        conditions are the Src / pipeline ones, which ``note_wait_condition``
        declines by design; see its docstring for why they are not latched
        conditions in the same sense. Those cycles are therefore left
        uncounted rather than attributed to something they are not.

        One cycle per held episode is still counted by neither path, and it
        predates this: on the tick the block mask matches, ``latch_wait`` goes
        true *after* the point the un-held branch would have counted, and the
        held branch does not run until the tick after. The instruction is held
        on that cycle, so the hardware counts both a stall and the reason for
        it. Closing it moves ``THREAD_STALLS_n`` on every workload, which is a
        separate change from this one and is not made here.
        """
        if self.check_for_wait_condition_met():
            self.latchedWaitInstruction = None
            return
        counters = self.perf_counters
        if counters is None or not counters.instrn_running:
            return
        found = self._latched_semaphore_reason()
        if found is not None:
            counters.note_wait_condition(self.frontend.thread_id, found[0])

    def setBackendEnforcedStall(self):
        self.backend_enforced_stall = True

    def clearBackendEnforcedStall(self):
        self.backend_enforced_stall = False

    def setLatchedWaitInstruction(
        self, opcode, condition_mask, block_mask, semaphore_mask=None
    ):
        self.latchedWaitInstruction = WaitGate.LatchedInstruction(
            opcode, condition_mask, block_mask, semaphore_mask
        )

    def check_for_wait_condition_met(self):
        """Is the latched wait's condition met, so the thread may be released?

        In semaphore mode this is exactly "no selected condition holds on any
        selected semaphore", which is
        :meth:`_latched_semaphore_reason` returning nothing. ``SEMWAIT.md``
        tabulates the condition mask as a pair of *keep waiting* rules --
        "**C0**: Any of the semaphores selected by ``SemaphoreMask`` have
        ``Value == 0``", "**C1**: ... have ``Value >= Max``" -- over an
        instruction that blocks "until **all** of the selected conditions are
        simultaneously met". Both quantifiers matter and they are the same
        rule from either end: the thread runs only when *every* selected
        (semaphore, condition) pair is satisfied, so *any* one unsatisfied
        pair holds it.

        This delegates rather than repeating the walk because repeating it is
        what went wrong: the walk used to be inlined here with its ``return
        True`` indented into the ``for`` body, so a multi-semaphore mask only
        ever tested the *first* selected semaphore and released the thread
        while the rest were still unsatisfied -- silently, with no fault. That
        is the third reader of these masks, and the drift
        :meth:`_latched_semaphore_reason` exists to prevent; there is now one
        walk and no way for the readers to disagree.
        """
        assert self.latchedWaitInstruction is not None
        if self.latchedWaitInstruction.isSemaphoreMode():
            return self._latched_semaphore_reason() is None
        else:
            for idx in range(15):
                if self.latchedWaitInstruction.getConditionCheck(idx):
                    if not self.check_for_semwait_condition_match(idx):
                        return False
            return True

    def check_for_semwait_condition_match(self, cond_idx):
        if self.blackhole_conditions:
            return self._check_blackhole_condition(cond_idx)
        # Some of these conditions do not apply, as per
        # https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/STALLWAIT.md
        match cond_idx:
            case 0:
                return True
            case 1:
                return not self.backend.unpacker_units[
                    0
                ].hasInflightInstructionsFromThread(self.frontend.thread_id)
            case 2:
                # Unpacker *1*. ``STALLWAIT.md``: C1 is "an instruction in any
                # stage of Unpacker 0's pipeline", C2 the same for "Unpacker
                # 1's". This asked unpacker 0 twice, so a thread waiting on C2
                # was told about the wrong unit -- and the Blackhole twin below
                # has always had it right, which is what makes it a
                # transcription slip rather than a modelling choice.
                return not self.backend.unpacker_units[
                    1
                ].hasInflightInstructionsFromThread(self.frontend.thread_id)
            case 3 | 4 | 5 | 6:
                return not self.backend.packer_unit.hasInflightInstructionsFromThread(
                    self.frontend.thread_id
                )
            case 7:
                return not self.backend.matrix_unit.hasInflightInstructionsFromThread(
                    self.frontend.thread_id
                )
            case 8:
                return (
                    self.backend.getSrcA(
                        self.backend.unpacker_units[0].srcBank
                    ).allowedClient
                    == SrcRegister.SrcClient.Unpackers
                )
            case 9:
                return (
                    self.backend.getSrcB(
                        self.backend.unpacker_units[1].srcBank
                    ).allowedClient
                    == SrcRegister.SrcClient.Unpackers
                )
            case 10:
                return (
                    self.backend.getSrcA(
                        self.backend.matrix_unit.srcABank
                    ).allowedClient
                    == SrcRegister.SrcClient.MatrixUnit
                )
            case 11:
                return (
                    self.backend.getSrcB(
                        self.backend.matrix_unit.srcBBank
                    ).allowedClient
                    == SrcRegister.SrcClient.MatrixUnit
                )
            case 12:
                return not self.backend.mover_unit.checkForOutstandingInstructions()
            case 13:
                return True
            case 14:
                return not self.backend.vector_unit.hasInflightInstructionsFromThread(
                    self.frontend.thread_id
                )
            case _:
                return True

    def _check_blackhole_condition(self, cond_idx):
        """Blackhole STALLWAIT condition bits (C0-C12), per
        BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md. Returns True when
        the (selected) condition is satisfied so the stall can clear. The bit
        assignments differ substantially from Wormhole's — e.g. bit 10 here is
        "the RISCV T core has an unprocessed memory request" (always satisfied in
        this functionally-timed sim), where Wormhole's bit 10 meant "SrcA owned
        by the matrix unit"; using the Wormhole layout deadlocks the unpacker."""
        be = self.backend
        thread = self.frontend.thread_id
        match cond_idx:
            case 0:  # C0: ThCon (scalar) has outstanding memory requests
                # Scalar-unit memory is synchronous here, so never outstanding.
                return True
            case 1:  # C1: instruction in Unpacker 0's pipeline for this thread
                return not be.unpacker_units[0].hasInflightInstructionsFromThread(
                    thread
                )
            case 2:  # C2: instruction in Unpacker 1's pipeline for this thread
                return not be.unpacker_units[1].hasInflightInstructionsFromThread(
                    thread
                )
            case 3:  # C3: instruction in the Packer pipeline for this thread
                return not be.packer_unit.hasInflightInstructionsFromThread(thread)
            case 4:  # C4: instruction in the Matrix Unit (FPU) pipeline
                return not be.matrix_unit.hasInflightInstructionsFromThread(thread)
            case 5:  # C5: SrcA not yet available for unpacker writes
                return (
                    be.getSrcA(be.unpacker_units[0].srcBank).allowedClient
                    == SrcRegister.SrcClient.Unpackers
                )
            case 6:  # C6: SrcB not yet available for unpacker writes
                return (
                    be.getSrcB(be.unpacker_units[1].srcBank).allowedClient
                    == SrcRegister.SrcClient.Unpackers
                )
            case 7:  # C7: SrcA not yet available for FPU reads
                return (
                    be.getSrcA(be.matrix_unit.srcABank).allowedClient
                    == SrcRegister.SrcClient.MatrixUnit
                )
            case 8:  # C8: SrcB not yet available for FPU reads
                return (
                    be.getSrcB(be.matrix_unit.srcBBank).allowedClient
                    == SrcRegister.SrcClient.MatrixUnit
                )
            case 9:  # C9: the mover has outstanding memory requests
                return not be.mover_unit.checkForOutstandingInstructions()
            case 10:  # C10: RISCV T core has an unprocessed memory request
                # RISCV<->L1 accesses are synchronous here, so never pending.
                return True
            case 11:  # C11: instruction in the Vector Unit (SFPU) pipeline
                return not be.vector_unit.hasInflightInstructionsFromThread(thread)
            case 12:  # C12: instruction in the Config Unit pipeline (any thread)
                # "*Any* thread", not the current one -- the only condition in
                # either arch's table that is not scoped to the issuing thread,
                # which its own note spells out: "This won't prevent other
                # threads from issuing new Configuration Unit instructions
                # though, and those new instructions will cause this thread to
                # continue to wait."
                #
                # Reachable since 2026-08-12, and it took two fixes. The config
                # unit reported every instruction as gone one cycle after
                # acceptance whatever its documented latency, so this was always
                # true (``TensixBackendConfigurationUnit.
                # hasInflightInstructionsFromThread``); and bit 12 was trimmed
                # off the Blackhole condition mask before it got here
                # (``TensixSyncUnit._read_wait_res``).
                return not any(
                    be.config_unit.hasInflightInstructionsFromThread(t)
                    for t in range(3)
                )
            case _:
                return True

    def is_clock_idle(self):
        """Idle with no wait latched and nothing waiting at the gate.

        A latched wait keeps the gate awake **whether or not anything is being
        held by it**, which is why this tests :attr:`latchedWaitInstruction`
        and not just :attr:`latch_wait`. Per ``SEMWAIT.md`` the Wait Gate
        "will then continuously re-evaluate the latched wait instruction until
        all of the selected conditions are simultaneously met, at which point
        the latched wait instruction will be forgotten" — re-evaluation is not
        conditional on anything being blocked, and both of its outcomes are
        observable: the latch is forgotten, and every unmet cycle counts on
        ``WAITING_FOR_NON{ZERO,FULL}_SEM_n`` (see
        :meth:`_tick_unheld_latched_wait`). A gate that called itself idle
        there would let the tile sleep through exactly the cycles the counter
        is defined over, and no placement of the increment could recover them.

        The cost is bounded by the same condition it counts: the tile is held
        awake only while a latched wait is *unsatisfied*, which is the wait's
        real duration and nothing more — the tick that finds it satisfied
        forgets the latch, and the dormancy decision is made against post-tick
        state. Measured over the Blackhole replay guards, the number of extra
        ticks this buys is zero: no tile in them ever slept with a live
        latched wait, so it changes what the model *can* count without
        changing what these workloads tick.

        The two stall flags are deliberately *not* treated as idle — they are
        cleared by the sync / ThCon units, which report themselves busy while
        that is pending, so the tile stays awake either way.
        """
        return (
            not self.latch_wait
            and self.latchedWaitInstruction is None
            and not self.frontend.wait_gate_instruction_fifo
        )

    def clock_tick(self, cycle_num):
        if self.mutex_stall:
            self._note_stall(cycle_num, "mutex_wait", blocked_on="SYNC")
        elif self.backend_enforced_stall:
            self._note_stall(cycle_num, "backend_enforced_stall")
        else:
            instruction = self.frontend.inspect_wait_gate_instruction()
            if not self.latch_wait and self.latchedWaitInstruction is not None:
                if instruction is not None:
                    instruction_info = TensixInstructionDecoder.getInstructionInfo(
                        instruction
                    )
                    self.latch_wait = (
                        self.latchedWaitInstruction.doesInstructionMatchBlockMask(
                            instruction_info
                        )
                    )
                if not self.latch_wait:
                    # Nothing became blocked, so nothing is being held -- but
                    # the wait is latched and the gate re-evaluates it anyway.
                    self._tick_unheld_latched_wait(cycle_num)
            elif self.latch_wait:
                condition_met = self.check_for_wait_condition_met()
                if condition_met:
                    self.latch_wait = False
                    self.latchedWaitInstruction = None
                    self._flush_stall()
                    return  # One cycle latency here
                self._note_latched_wait(cycle_num)

            if not self.latch_wait:
                if instruction is not None:
                    instruction_info = TensixInstructionDecoder.getInstructionInfo(
                        instruction
                    )
                    thread_id = self.frontend.thread_id
                    if self.backend.thread_issue_block[thread_id] > cycle_num:
                        # A documented whole-thread interlock is live: the
                        # Scalar Unit's, which holds the thread for the whole
                        # of its instruction's execution, or an unpacker's, for
                        # an ``UNPACR``'s address phase. *Nothing* from this
                        # thread may pass the gate while it is, whichever unit
                        # it is bound for, which is what separates this from
                        # the per-unit issue refusal below and why it is
                        # checked first. See
                        # ``TensixBackend.block_thread_issue`` for the
                        # sentences it comes from.
                        self._note_stall(
                            cycle_num,
                            "thread_issue_block",
                            blocked_on=self.backend.thread_issue_block_unit[thread_id],
                            opcode=instruction_info["name"],
                        )
                        return
                    if instruction_info["ex_resource"] == "MATH":
                        # For FPU instructions need to ensure that srcA and srcB
                        # being consumed has allowed client of MatrixUnit
                        blocked_bank = self.whichSrcBankBlocksFPUInstruction(
                            instruction_info["name"]
                        )
                        if blocked_bank is not None:
                            self._note_stall(
                                cycle_num,
                                "src_reserved_by_unpacker",
                                blocked_on="UNPACK",
                                opcode=instruction_info["name"],
                                src_bank=blocked_bank,
                            )
                            return
                    instruction_accepted = self.frontend.backend.issueInstruction(
                        instruction, self.frontend.thread_id
                    )
                    if not instruction_accepted:
                        # The unit left its reason behind in ``_refuse``; this is
                        # the only caller, and it reads it in the same statement
                        # that saw the refusal. The unit it named wins over the
                        # one we offered to, because a unit blocked on *another*
                        # unit knows which, and the gate does not.
                        backend = self.frontend.backend
                        self._note_stall(
                            cycle_num,
                            backend.last_refusal_reason,
                            blocked_on=(
                                backend.last_refusal_blocked_on
                                or instruction_info["ex_resource"]
                            ),
                            opcode=instruction_info["name"],
                            src_bank=backend.last_refusal_src_bank,
                        )
                    elif self._stall_since is not None:
                        # Progress: whatever the thread was blocked on has
                        # cleared. One attribute test on the accept path, and it
                        # is only ever true when the STALL category is enabled.
                        self._flush_stall()
                    if instruction_accepted:
                        # An accepted ATGETM stalls the gate until the sync unit
                        # grants the mutex (informMutexAcquired clears mutex_stall).
                        # Set the stall only once the instruction has actually
                        # reached the sync unit: if issueInstruction rejected it
                        # this cycle (its issue queue was busy), the gate must stay
                        # free to retry next cycle — otherwise mutex_stall latches
                        # forever with no ATGETM in flight to ever clear it.
                        if instruction_info["name"] == "ATGETM":
                            self.mutex_stall = True
                        # If the instruction was accepted then remove it,
                        # otherwise retry next cycle
                        if self.getDiagnosticSettings().reportIssuedInstructions():
                            print(
                                f"Wait gate: issued {instruction_info['name']} to {instruction_info['ex_resource']} "
                                f"from thread {self.frontend.thread_id}"
                            )
                        counters = self.perf_counters
                        if counters is not None and counters.instrn_running:
                            # THREAD_INSTRUCTIONS_n: the grant side of the same
                            # RTL instance whose req side is THREAD_STALLS_n.
                            counters.note_dispatch(self.frontend.thread_id)
                        bus = get_bus()
                        if self.frontend.unit_id is not None and bus.is_enabled(
                            EventCategory.DISPATCH
                        ):
                            bus.publish(
                                DispatchEvent(
                                    cycle=cycle_num,
                                    unit_id=self.frontend.unit_id,
                                    opcode=instruction_info["name"],
                                    target_unit=instruction_info["ex_resource"],
                                    thread_id=self.frontend.thread_id,
                                )
                            )
                        self.frontend.pop_wait_gate_instruction()

    def whichSrcBankBlocksFPUInstruction(self, opcode):
        """``"A"``, ``"B"`` or ``None`` — which Src bank holds this instruction.

        The bank identity is what separates ``WAITING_FOR_SRCA_VALID`` from
        ``WAITING_FOR_SRCB_VALID`` in the hardware performance counters, and A
        is tested before B here for the same reason the boolean predicate this
        replaces did: an instruction blocked on both is charged once.
        """
        if opcode in WaitGate.MATH_ALLOWED_CLIENT_INSTRUCTIONS[0]:
            if (
                self.backend.getMatrixUnit().getSrcA().allowedClient
                == SrcRegister.SrcClient.Unpackers
            ):
                return "A"

        if opcode in WaitGate.MATH_ALLOWED_CLIENT_INSTRUCTIONS[1]:
            if (
                self.backend.getMatrixUnit().getSrcB().allowedClient
                == SrcRegister.SrcClient.Unpackers
            ):
                return "B"

        return None

    def checkIfFPUInstructionShouldStall(self, opcode):
        """Boolean form of :meth:`whichSrcBankBlocksFPUInstruction`.

        Kept because the wait-gate guards call it by name.
        """
        return self.whichSrcBankBlocksFPUInstruction(opcode) is not None

    def informMutexAcquired(self):
        self.mutex_stall = False


class TensixReplayExpander(TensixFrontendUnit):
    """
    The replay expander front end unit to record or replay sequences
    of instructions.

    Based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/REPLAY.md
    """

    def __init__(self, frontend):
        self.replay_buffer = [0] * 32
        self.append_instruction_to_buffer = False
        self.exec_while_load = False
        self.replay_len = 0
        self.replay_start_idx = 0
        self.replay_idx = 0
        super().__init__(frontend)

    def _publish_replay_dispatch(self, cycle_num, instruction):
        bus = get_bus()
        if self.frontend.unit_id is None or not bus.is_enabled(EventCategory.DISPATCH):
            return
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        bus.publish(
            DispatchEvent(
                cycle=cycle_num,
                unit_id=self.frontend.unit_id,
                opcode=instruction_info["name"],
                target_unit=instruction_info["ex_resource"],
                thread_id=self.frontend.thread_id,
            )
        )

    def is_clock_idle(self):
        return not self.frontend.replay_instruction_fifo

    def clock_tick(self, cycle_num):
        instruction = self.frontend.pop_replay_instruction()
        if instruction is not None:
            self._publish_replay_dispatch(cycle_num, instruction)
            if self.append_instruction_to_buffer:
                if self.replay_idx < self.replay_len:
                    self.replay_buffer[
                        (self.replay_start_idx + self.replay_idx) % 32
                    ] = instruction
                    self.replay_idx += 1
                    if self.exec_while_load:
                        self.frontend.push_wait_gate_instruction(instruction)
                else:
                    self.append_instruction_to_buffer = False

            if not self.append_instruction_to_buffer:
                instruction_info = TensixInstructionDecoder.getInstructionInfo(
                    instruction
                )
                if instruction_info["name"] == "REPLAY":
                    instr_args = instruction_info["instr_args"]
                    if instr_args["load_mode"] == 0:
                        index = instr_args["start_idx"]
                        for i in range(instr_args["len"] or 64):
                            self.frontend.push_wait_gate_instruction(
                                self.replay_buffer[(index + i) % 32]
                            )
                    else:
                        self.replay_start_idx = instr_args["start_idx"]
                        self.exec_while_load = instr_args["execute_while_loading"]
                        self.replay_len = instr_args["len"] or 64
                        self.replay_idx = 0
                        self.append_instruction_to_buffer = True
                else:
                    self.frontend.push_wait_gate_instruction(instruction)


class TensixMOPExpander(TensixFrontendUnit, MemMapable):
    """
    The MOP expander allows templating of instructions, with a single MOP then
    issuing a chain of preset instructions. Often used for unpacker and matrix
    unit opertions, where multiple instructions are needed for one logical step.

    Based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MOPExpander.md
    """

    def __init__(self, frontend):
        self.mop_cfg = [0] * 9
        self.mask_hi = 0
        super().__init__(frontend)

    def read(self, address, size):
        raise NotImplementedError("Can not read from Tensix MOP expander configuration")

    def write(self, addr, value, size=None):
        idx = int(addr / 4)
        assert idx < 9

        self.mop_cfg[idx] = conv_to_uint32(value)

    def is_clock_idle(self):
        return not self.frontend.mop_instruction_fifo

    def clock_tick(self, cycle_num):
        instruction = self.frontend.pop_mop_instruction()
        if instruction is not None:
            bus = get_bus()
            if self.frontend.unit_id is not None and bus.is_enabled(
                EventCategory.DISPATCH
            ):
                mop_info = TensixInstructionDecoder.getInstructionInfo(instruction)
                bus.publish(
                    DispatchEvent(
                        cycle=cycle_num,
                        unit_id=self.frontend.unit_id,
                        opcode=mop_info["name"],
                        target_unit=mop_info["ex_resource"],
                        thread_id=self.frontend.thread_id,
                    )
                )
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            if instruction_info["name"] == "MOP":
                instr_args = instruction_info["instr_args"]
                if instr_args["mop_type"] == 0:
                    for new_instr in self.expand_template_zero(
                        (self.mask_hi << 16) + instr_args["zmask_lo16"],
                        instr_args["loop_count"],
                    ):
                        self.frontend.push_replay_instruction(new_instr)
                else:
                    for new_instr in self.expand_template_one():
                        self.frontend.push_replay_instruction(new_instr)
            elif instruction_info["name"] == "MOP_CFG":
                self.mask_hi = instruction_info["instr_args"]["zmask_hi16"]
            else:
                self.frontend.push_replay_instruction(instruction)

    def expand_template_zero(self, mask, count1):
        flags = self.mop_cfg[1]
        insnb = self.mop_cfg[2]
        insna0 = self.mop_cfg[3]
        insna1 = self.mop_cfg[4]
        insna2 = self.mop_cfg[5]
        insna3 = self.mop_cfg[6]
        skipa0 = self.mop_cfg[7]
        skipb = self.mop_cfg[8]
        hasb = flags & 1
        hasa123 = flags & 2
        for i in range(count1 + 1):
            if mask & 1 == 0:
                yield insna0
                if hasa123:
                    yield insna1
                    yield insna2
                    yield insna3
                if hasb:
                    yield insnb
            else:
                yield skipa0
                if hasb:
                    yield skipb
            mask >>= 1

    def expand_template_one(self):
        outercount = self.mop_cfg[0] & 127
        innercount = self.mop_cfg[1] & 127
        startop = self.mop_cfg[2]
        endop0 = self.mop_cfg[3]
        endop1 = self.mop_cfg[4]
        loopop = self.mop_cfg[5]
        loopop1 = self.mop_cfg[6]
        loop0last = self.mop_cfg[
            7
        ]  # overrides loopop or loopop1 on last iteration of inner loop, if also last iteration of outer loop
        loop1last = self.mop_cfg[
            8
        ]  # overrides loopop or loopop1 on last iteration of inner loop, if not last iteration of outer loop

        if self.isnop(loopop1):
            loopopflip = 0
        else:
            loopopflip = (
                loopop ^ loopop1
            )  # inner loop will alternate between the two instructions and will
            innercount *= (
                2  # execute for twice as many iterations. it is expressed like this
            )
            # because loop0last / loop1last override the last iteration.
        if (
            outercount == 1
            and self.isnop(startop)
            and innercount == 0
            and not self.isnop(endop0)
        ):
            outercount += 128  # hardware bug
        for j in range(outercount):
            if not self.isnop(startop):
                yield startop
            for i in range(innercount):
                if i != innercount - 1:
                    yield loopop
                elif j != outercount - 1:
                    yield loop1last
                else:
                    yield loop0last
                loopop ^= loopopflip
            if not self.isnop(endop0):
                yield endop0
                if not self.isnop(endop1):
                    yield endop1

    def isnop(self, instruction):
        # this only recognises plain nop (opcode 0x02), not dmanop nor sfpnop.
        opcode = extract_bits(instruction, 8, 24)
        return opcode == 0x02

    def getSize(self):
        return 0x23
