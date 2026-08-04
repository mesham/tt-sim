from abc import ABC
from enum import IntEnum

from tt_sim.device.clock import Clockable
from tt_sim.pe.tensix.util import TensixInstructionDecoder
from tt_sim.trace import ComputeEvent, EventCategory, get_bus


class DataFormat(IntEnum):
    FP32 = 0
    FP16 = 1
    BFP8 = 2
    BFP4 = 3
    BFP2 = 11
    # BF16 is the canonical Wormhole name for code 5 (per the ISA docs'
    # FormatConversion.md table); FP16_b is tt-metal's name for the same
    # format, kept here as a Python IntEnum alias.
    BF16 = 5
    FP16_b = 5
    BFP8_b = 6
    BFP4_b = 7
    BFP2_b = 15
    INT8 = 14
    UINT8 = 30
    UINT16 = 9
    INT32 = 8
    UINT32 = 24
    TF32 = 4

    def isBFPFormat(self):
        return self.value == 2 or self.value == 3 or self.value == 11


DATA_FORMAT_TO_BITS = {
    DataFormat.FP32: 32,
    DataFormat.FP16: 16,
    DataFormat.BFP8: 8,
    DataFormat.BFP4: 4,
    DataFormat.BFP2: 2,
    DataFormat.BF16: 16,
    DataFormat.BFP8_b: 8,
    DataFormat.BFP4_b: 4,
    DataFormat.BFP2_b: 2,
    DataFormat.INT8: 8,
    DataFormat.UINT8: 8,
    DataFormat.UINT16: 16,
    DataFormat.UINT32: 32,
    DataFormat.INT32: 32,
    DataFormat.TF32: 32,
}

DATA_FORMAT_TO_NAME = {
    DataFormat.FP32: "FP32",
    DataFormat.FP16: "FP16",
    DataFormat.BFP8: "BFP8",
    DataFormat.BFP4: "BFP4",
    DataFormat.BFP2: "BFP2",
    DataFormat.BF16: "BF16",
    DataFormat.BFP8_b: "BFP8_b",
    DataFormat.BFP4_b: "BFP4_b",
    DataFormat.BFP2_b: "BFP2_b",
    DataFormat.INT8: "INT8",
    DataFormat.UINT8: "UINT8",
    DataFormat.UINT16: "UINT16",
    DataFormat.UINT32: "UINT32",
    DataFormat.INT32: "INT32",
    DataFormat.TF32: "TF32",
}


class TensixBackendUnit(Clockable, ABC):
    def __init__(self, backend, opcode_to_method_map, unit_name):
        self.backend = backend
        self.next_instruction = []
        self.opcode_to_method_map = opcode_to_method_map
        self.unit_name = unit_name
        self.unit_id: tuple | None = None
        # Shadows the Clockable class attribute as an instance attribute so
        # clock_tick's guard is a single dict lookup rather than an MRO walk;
        # this is read once per unit per cycle on the hot path. See
        # ``occupy_for``.
        self.busy_until = None
        # A ``tt_sim.perf.model.UnitCostModel`` once a unit opts into the
        # cycle-cost tables *and* ``TT_SIM_COST_MODEL`` is set; ``None``
        # otherwise, which is the default and keeps every existing cycle count
        # byte-identical. See ``instruction_occupancy``.
        self.cost_model = None

    def issueInstruction(self, instruction, from_thread):
        # The default issuing of instructions here, which applies to most
        # units, is one instruction per cycle. Can override for specific
        # units with more complex behaviour
        if self.is_occupied():
            return False
        if len(self.next_instruction) == 0:
            self.next_instruction.append(
                (
                    instruction,
                    from_thread,
                )
            )
            return True
        else:
            return False

    def getDiagnosticSettings(self):
        return self.backend.getDiagnosticSettings()

    def hasInflightInstructionsFromThread(self, from_thread):
        if len(self.next_instruction) > 0:
            for _, thread_id in self.next_instruction:
                if thread_id == from_thread:
                    return True
        return False

    def is_clock_idle(self):
        """Idle with an empty issue queue.

        Complete for every unit that does not override ``clock_tick``: the
        base implementation only drains ``next_instruction``, and a handler
        has no way to defer work to a later cycle except through its own
        ``clock_tick``. Units that *do* override it (config, sync, thcon,
        mover, unpacker) extend this with the state their override reads.
        """
        return not self.next_instruction

    def occupy_for(self, cycle_num, cycles):
        """Declare this unit busy for ``cycles`` cycles starting at ``cycle_num``.

        The Phase 5 entry point (``docs/plans/event-driven-pump.md``): a
        per-unit cost table calls this once the cycle's issue batch has
        retired, with the longest cost in it, and the unit then refuses
        further work until ``cycle_num + cycles``. ``cycles <= 1`` is the
        status quo and is a no-op, so a table that has no entry for an opcode
        costs nothing.

        **Occupancy is back-pressure on the next instruction, never a delay of
        this one.** That is the distinction :meth:`clock_tick` turns on and it
        is what both architectures' Configuration Unit pages describe: the unit
        is a pipeline (Blackhole names the stages, -4..+1) whose throughput
        limits how fast instructions may *enter*, while each instruction's own
        write commits at its documented *latency* — 1 cycle for ``SETC16``,
        2 for ``WRCFG`` — regardless of what else the unit is doing. An
        instruction the unit has already accepted must therefore still take
        effect in the cycle it was accepted for; anything else reorders it
        behind instructions its own thread issued later, which is the bug this
        method's caller used to have. See the comment in ``config.py``.

        Two things fall out of setting :attr:`busy_until`, both handled here
        and in :meth:`next_wake_cycle`: ``clock_tick`` stops draining the
        issue queue until the deadline, which *is* the back-pressure the wait
        gates already query through
        :meth:`hasInflightInstructionsFromThread`; and the pump skips
        straight to the retire cycle rather than visiting the unit in between.

        What it gates is the *base* drain. The five units that override
        ``clock_tick`` (config, sync, thcon, mover, unpacker) all reach it
        through ``super()``, but their own pre-``super()`` work — the mover's
        TDMA queue, ThCon's ``FLUSHDMA`` polling — runs first and is
        deliberately not covered: that work is the unit servicing somebody
        else, not retiring the instruction it was issued.
        """
        if cycles > 1:
            self.busy_until = cycle_num + cycles

    def is_occupied(self):
        """True while a multi-cycle instruction still holds this unit.

        Always False with the cost model off, because nothing arms
        :attr:`busy_until` — so this costs one attribute read per issue and
        changes nothing.

        Why issue is refused rather than queued, which is the whole reason this
        exists: tt-sim's frontend treats an instruction as *issued* the moment
        a unit accepts it, and the thread moves on in the same cycle. Phase 4's
        ``occupy_for`` stopped an occupied unit *draining* its queue but left it
        *accepting* into one — so a parked instruction would retire after the
        thread's next instruction had already run in a different, idle unit.
        That is a reordering of a single thread's program, and it is not a
        theoretical hazard: it was found while charging the config unit's
        documented ">= 2" for ``RDCFG``, which delays five ``SETC16``s on the
        math thread and four ``WRCFG``s on the pack thread by a cycle each.

        Refusing is necessary and was not sufficient. It cannot reach an
        instruction the unit has *already accepted*, and that was the other
        half of the same failure: the config unit takes a whole batch in one
        cycle, so the ``SETC16`` beside the ``RDCFG`` had been accepted before
        there was anything to refuse. :meth:`clock_tick` is where that half is
        handled — the batch retires, and only then is the unit held.

        Refusing is also the closer reading of the ISA docs, which say of the
        Scalar Unit that the issuing thread "is unable to start any further
        instruction (in any unit)" until the current one completes. This models
        the weaker half of that — the thread cannot start another instruction
        *in this unit* — which is a floor on the stall, in the same direction
        as charging bounded costs at their low end.
        """
        return self.busy_until is not None

    def instruction_occupancy(self, instruction_name, issue_thread):
        """Cycles ``instruction_name`` occupies this unit, or ``None``.

        Only consulted when :attr:`cost_model` is set, which happens only for
        units that have been wired to the cycle-cost tables (Phase 5 of
        ``docs/plans/event-driven-pump.md``) *and* only when
        ``TT_SIM_COST_MODEL`` is truthy — so an unwired unit reads one ``None``
        attribute per instruction and nothing else.

        The default is the straight table lookup, which is the right answer for
        every unit whose published cost is a per-opcode constant (SFPU, ThCon,
        packer, sync). ``None`` means "no opinion" and leaves the same-cycle
        retire alone, which is deliberately what an untabulated opcode gets —
        see ``tt_sim/perf/model.py`` for why that choice is made once, there.
        Only a unit whose cost is a *function* of state needs an override; the
        matrix unit is the one such case, because its fidelity-scaled ops are
        costed against the phase they run at.

        Called *before* the handler runs, because occupancy is a property of
        issue and because a handler may advance the state the cost depends on
        (the matrix unit's ``ADDR_MOD`` step moves the fidelity phase on).
        """
        model = self.cost_model
        return None if model is None else model.occupancy(instruction_name)

    def next_wake_cycle(self, cycle_num):
        # Identical to Clockable's default; spelled out because this is the
        # unit the cost tables attach to and the derivation should be readable
        # next to clock_tick's matching guard.
        busy_until = self.busy_until
        if busy_until is not None and busy_until > cycle_num:
            return busy_until
        return None if self.is_clock_idle() else cycle_num + 1

    def clock_tick(self, cycle_num):
        """Retire this cycle's issue batch, then charge the unit for it.

        ``next_instruction`` holds exactly the instructions accepted *for this
        cycle* — ``issueInstruction`` refuses everything while
        :meth:`is_occupied`, and an unoccupied tick empties the queue — so the
        whole batch belongs to one cycle and every member of it retires in that
        cycle. A unit that accepts more than one per cycle (config: up to three
        ``SETC16`` plus one of the shared-IPC-group ops; sync: three mutex ops;
        misc: one per thread) is modelling parallel hardware paths, and a cost
        charged to one of them does not push the others out.

        Hence :attr:`busy_until` is armed *after* the drain, from the longest
        occupancy in the batch, rather than mid-loop. Arming it mid-loop and
        returning left the rest of the batch queued for a later cycle — and
        because the issuing threads had already been told those instructions
        were accepted, they had moved on and their *later* instructions
        retired first, in other units. That reordered a single thread's
        program: charging ``RDCFG`` two cycles pushed the math thread's
        ``SETC16`` of ``DEST_TARGET_REG_CFG_MATH_Offset`` behind two of its own
        ``MVMUL``s, which then accumulated into the wrong half of Dst and made
        ``matmul_block`` compute a wrong answer. See ``config.py`` and
        :meth:`occupy_for`.
        """
        busy_until = self.busy_until
        if busy_until is not None:
            if cycle_num < busy_until:
                # Occupied by a multi-cycle instruction; nothing new is
                # accepted and nothing is left over to retire.
                return
            self.busy_until = None
        # The longest occupancy charged across this cycle's batch. The unit is
        # held by whichever of its parallel paths is slowest, so ``max``; the
        # batch is nearly always one instruction, in which case this is just
        # that instruction's cost.
        batch_occupancy = 0
        # next_instruction is all instructions to process in this cycle,
        # is often one but for some units might be more
        while len(self.next_instruction) > 0:
            instruction, issue_thread = self.next_instruction.pop(0)
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            instruction_name = instruction_info["name"]
            if instruction_name in self.opcode_to_method_map:
                if "instr_args" in instruction_info:
                    instr_args = instruction_info["instr_args"]
                else:
                    instr_args = None
                occupancy = (
                    self.instruction_occupancy(instruction_name, issue_thread)
                    if self.cost_model is not None
                    else None
                )
                getattr(self, self.opcode_to_method_map[instruction_name])(
                    instruction_info, issue_thread, instr_args
                )
                bus = get_bus()
                if self.unit_id is not None and bus.is_enabled(EventCategory.COMPUTE):
                    bus.publish(
                        ComputeEvent(
                            cycle=cycle_num,
                            unit_id=self.unit_id,
                            op=instruction_name,
                            target_unit=self.unit_name,
                            thread_id=issue_thread,
                            # The occupancy the cost table charged, or 0 for
                            # "no claim" — an unwired unit, an uncosted opcode,
                            # or TT_SIM_COST_MODEL unset. A trace consumer must
                            # not read 0 as one cycle; see ComputeEvent.
                            duration=occupancy or 0,
                        )
                    )
                if occupancy is not None and occupancy > batch_occupancy:
                    batch_occupancy = occupancy
            else:
                raise NotImplementedError(
                    f"{self.unit_name} unit can not handle instruction '{instruction_info['name']}'"
                )
        if batch_occupancy:
            self.occupy_for(cycle_num, batch_occupancy)

    def getThreadConfigValue(self, issue_thread, key):
        return self.backend.getThreadConfigValue(issue_thread, key)

    def getConfigValue(self, state_id, key, words=1):
        return self.backend.getConfigValue(state_id, key, words)

    def getRWC(self, thread_id):
        return self.backend.getRWC(thread_id)

    def getDst(self):
        return self.backend.getDst()

    def checkIfNextInstructionsContainOpcodes(self, *instr_op):
        for instruction, _ in self.next_instruction:
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            instruction_name = instruction_info["name"]
            if instruction_name in list(instr_op):
                return True
        return False

    def checkIfNextInstructionsContainAnyOtherOpcodes(self, *allowed_instr_op):
        for instruction, _ in self.next_instruction:
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            instruction_name = instruction_info["name"]
            if instruction_name not in list(allowed_instr_op):
                return True
        return False
