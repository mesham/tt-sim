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
        #
        # The LATEST deadline across every occupied IPC group, or None when the
        # unit is entirely free. That makes it the cheap "is anything armed at
        # all?" test every hot path starts with, and it keeps meaning exactly
        # what it meant before groups existed for the units that have none.
        self.busy_until = None
        # Deadline per IPC group: ``{group_name: cycle}``, with ``None`` as the
        # group name for a unit whose published limit is whole-unit. The
        # authority :meth:`is_occupied` consults; :attr:`busy_until` is its max.
        self.busy_groups = {}
        # A ``tt_sim.perf.model.UnitCostModel`` once a unit opts into the
        # cycle-cost tables *and* ``TT_SIM_COST_MODEL`` is set; ``None``
        # otherwise, which is the default and keeps every existing cycle count
        # byte-identical. See ``instruction_occupancy``.
        self.cost_model = None
        # Cross-thread grant fairness for the single-slot issue queue below.
        # ``None`` until a refusal ever happens (the common case pays one falsy
        # check); thereafter the set of threads refused since their last grant.
        # ``_grant_lru`` orders the three threads least-recently-granted first.
        #
        # This exists because the front-end FIFO bound
        # (``tt_sim/pe/tensix/frontend.CORE_PUSH_INFLIGHT_BOUND``) lets a
        # refused thread stall its issuing baby core: the wait gates tick in a
        # fixed thread order, so without rotation a thread sustaining one
        # instruction per cycle at a shared unit would win the slot every
        # cycle and *starve* the other threads' cores for ever — a deadlock no
        # silicon has (hardware round-robins) and, less dramatically, the
        # reason three threads sharing a 1-IPC unit measure 3.0x each on
        # silicon rather than 1x/inf/inf. A thread that was refused keeps
        # re-offering every cycle (the wait gate retries its head instruction
        # until accepted), which is what makes the waiting set safe to trust
        # without timestamps.
        self._starved_threads = None
        self._grant_lru = [0, 1, 2]

    def _refuse(self, reason, blocked_on=""):
        """Record *why* this unit is refusing, then refuse.

        The wait gate publishes the :class:`~tt_sim.trace.events.StallEvent`,
        because it is the caller that holds ``cycle_num`` and the decoded
        opcode -- including the ``ex_resource`` name that becomes
        ``blocked_on``, so the unit's own identity does not need recording
        here. Written to the shared backend rather than to ``self`` so the gate
        can read it back without first working out which of the ten units
        ``backend.issueInstruction`` dispatched to.

        ``blocked_on`` overrides the gate's default of "the unit I offered to".
        Usually those are the same thing, but not always: an unpacker refusing
        because the Src bank it writes is still owned by the *matrix unit* is
        blocked on ``MATH``, and reporting ``UNPACK`` there would point a reader
        at the symptom instead of the cause.

        Costs one test and two attribute stores, on a branch that was already
        returning False, and nothing at all on the accept path. It is
        deliberately not conditional on the bus being enabled: a branch on a
        global would cost more here than the stores it guards.

        A unit built without a backend -- which the clock and cost-model unit
        tests do, exercising a single unit in isolation -- records nothing and
        still refuses, because the recording only exists to be read back by a
        wait gate that such a unit does not have.
        """
        backend = self.backend
        if backend is not None:
            backend.last_refusal_reason = reason
            backend.last_refusal_blocked_on = blocked_on
        return False

    def issueInstruction(self, instruction, from_thread):
        # The default issuing of instructions here, which applies to most
        # units, is one instruction per cycle. Can override for specific
        # units with more complex behaviour
        #
        # ``busy_until is not None`` first: with the cost model off (the
        # default) nothing is ever armed, so this is one attribute read and the
        # group is never even looked up.
        starved = self._starved_threads
        if self.busy_until is not None and self.is_occupied(
            self.issue_group(instruction)
        ):
            if starved is None:
                starved = self._starved_threads = set()
            starved.add(from_thread)
            return self._refuse("unit_busy")
        if self.next_instruction:
            if starved is None:
                starved = self._starved_threads = set()
            starved.add(from_thread)
            return self._refuse("issue_slot_taken")
        # The slot is free. Yield it when a thread granted less recently than
        # this one is also waiting: that thread's gate re-offers every cycle,
        # so the grant rotates rather than following wait-gate tick order.
        if starved and self._must_yield(from_thread, starved):
            starved.add(from_thread)
            return self._refuse("issue_yield_fairness")
        self.next_instruction.append(
            (
                instruction,
                from_thread,
            )
        )
        if starved:
            starved.discard(from_thread)
        lru = self._grant_lru
        if lru[-1] != from_thread:
            lru.remove(from_thread)
            lru.append(from_thread)
        return True

    def _must_yield(self, from_thread, starved):
        """True when a less-recently-granted thread is waiting for this unit."""
        lru = self._grant_lru
        for other in lru[: lru.index(from_thread)]:
            if other in starved:
                return True
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

    def occupy_for(self, cycle_num, cycles, group=None):
        """Declare ``group`` busy for ``cycles`` cycles starting at ``cycle_num``.

        The Phase 5 entry point (``docs/plans/event-driven-pump.md``): a
        per-unit cost table calls this once the cycle's issue batch has
        retired, with the longest cost charged to each group in it, and the
        unit then refuses further work *of that group* until
        ``cycle_num + cycles``. ``cycles <= 1`` is the status quo and is a
        no-op, so a table that has no entry for an opcode costs nothing.

        **The hold is per IPC group, not per unit**, because that is the shape
        the only source that publishes one gives it. Blackhole's Configuration
        Unit page tabulates an "IPC group" column: ``SETC16`` alone is
        ``ThreadConfig``, everything else is ``Config``, and "sustained
        throughput across the entire group is limited to one instruction per
        cycle (or half an instruction per cycle if ``CFGSHIFTMASK`` is used)".
        A whole-unit hold charged ``CFGSHIFTMASK``'s two cycles against the
        ``SETC16`` path as well, refusing an instruction the hardware issues
        from a different group in the same cycle. ``group=None`` — the default,
        and what every other unit passes — is the whole-unit hold unchanged;
        it is also what a *grouped* unit charges an opcode the table gives no
        group for, since refusing more is the safe direction.

        Only one unit on one architecture has groups today, and the check for
        whether any other needed it was made against the documents rather than
        assumed: the Matrix Unit and Scalar Unit tables have no group column
        (the Scalar Unit is explicitly "executing at most one instruction at a
        time, and has no internal pipelining", so whole-unit is not an
        approximation there but the stated behaviour), and while the Sync Unit
        does describe two throughput classes in prose — mutex ops "up to three
        per cycle, provided they refer to different mutexes" against the
        semaphore ops' shared one — every Sync Unit entry costs one cycle, so
        nothing ever arms and the distinction is unobservable.

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
            deadline = cycle_num + cycles
            if deadline > self.busy_groups.get(group, 0):
                self.busy_groups[group] = deadline
            if self.busy_until is None or deadline > self.busy_until:
                self.busy_until = deadline

    def _release_expired(self, cycle_num):
        """Drop every group whose deadline has arrived, and re-derive the max.

        Called once at the top of :meth:`clock_tick`, which is the only place
        occupancy is ever released — so a group stays occupied from the cycle
        it is armed in right up to its deadline tick, exactly as the single
        :attr:`busy_until` did.
        """
        groups = self.busy_groups
        for group in [g for g, deadline in groups.items() if cycle_num >= deadline]:
            del groups[group]
        self.busy_until = max(groups.values()) if groups else None

    def is_occupied(self, group=None):
        """True while a multi-cycle instruction still holds ``group``.

        Always False with the cost model off, because nothing arms
        :attr:`busy_until` — so this costs one attribute read per issue and
        changes nothing.

        ``group=None`` asks about the whole-unit hold, which for a unit with no
        published IPC groups is the only hold there is: everything it charges
        goes to the ``None`` key, and this reduces to the pre-group behaviour
        exactly. For a grouped unit it asks only about the named group, which
        is the point — Blackhole's ``CFGSHIFTMASK`` holds ``Config`` and leaves
        ``ThreadConfig`` free, so the ``SETC16`` the hardware would accept in
        the next cycle is accepted here too.

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
        return self.busy_until is not None and group in self.busy_groups

    def instruction_group(self, instruction_name):
        """The IPC group ``instruction_name`` contends for, or ``None``.

        ``None`` means "charge this against the whole unit" and is the answer
        for every instruction of every unit whose documentation publishes no
        group column, so this reduces to the pre-group mechanism wherever the
        sources do not say otherwise. See
        :meth:`tt_sim.perf.model.UnitCostModel.ipc_group`.
        """
        model = self.cost_model
        return None if model is None else model.ipc_group(instruction_name)

    def issue_group(self, instruction):
        """:meth:`instruction_group` for a still-encoded instruction word.

        Split out because the issue path holds the raw word rather than a
        decoded name, and decoding is not free. A unit with no IPC groups —
        which is all of them but Blackhole's config unit — answers from
        :attr:`~tt_sim.perf.model.UnitCostModel.has_ipc_groups` without
        decoding anything. A unit that overrides ``issueInstruction`` and has
        already decoded should call :meth:`instruction_group` directly instead.
        """
        model = self.cost_model
        if model is None or not model.has_ipc_groups:
            return None
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        return model.ipc_group(instruction_info["name"])

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
        # Clockable's default, extended for per-group occupancy; spelled out
        # because this is the unit the cost tables attach to and the derivation
        # should be readable next to clock_tick's matching guard.
        busy_until = self.busy_until
        if busy_until is not None and busy_until > cycle_num:
            # A held unit may still have accepted work into a *free* group, and
            # that work must retire in the cycle it was accepted for or it is
            # reordered behind its own thread's later instructions. So the
            # queue decides first; only an idle queue lets the pump stride to a
            # deadline, and then to the soonest one, not the latest.
            if self.next_instruction:
                return cycle_num + 1
            # ``_release_expired`` runs at the top of every tick, so after one
            # no live deadline is in the past and this is always > cycle_num;
            # ``max`` guards the case where a caller asks before the first
            # tick, where the answer must still be "come back", never "never".
            return max(min(self.busy_groups.values()), cycle_num + 1)
        return None if self.is_clock_idle() else cycle_num + 1

    def clock_tick(self, cycle_num):
        """Retire this cycle's issue batch, then charge the unit for it.

        ``next_instruction`` holds exactly the instructions accepted *for this
        cycle* — ``issueInstruction`` refuses everything whose group
        :meth:`is_occupied`, and every tick empties the queue — so the whole
        batch belongs to one cycle and every member of it retires in that
        cycle. A unit that accepts more than one per cycle (config: up to three
        ``SETC16`` plus one of the shared-IPC-group ops; sync: three mutex ops;
        misc: one per thread) is modelling parallel hardware paths, and a cost
        charged to one of them does not push the others out.

        **The drain is unconditional**, which is what per-group occupancy costs
        the mechanism and it is not optional. While the hold was whole-unit an
        occupied unit could not have accepted anything, so returning early was
        equivalent to draining an empty queue. With groups a held unit accepts
        into its *free* groups, and skipping the drain would leave those
        instructions queued for a later cycle after their threads had been told
        they were accepted — which is precisely the reordering this method was
        rewritten to stop.

        Hence occupancy is armed *after* the drain, from the longest cost
        charged to each group in the batch, rather than mid-loop. Arming it
        mid-loop and returning left the rest of the batch queued for a later
        cycle — and because the issuing threads had already been told those
        instructions were accepted, they had moved on and their *later*
        instructions retired first, in other units. That reordered a single
        thread's program: charging ``RDCFG`` two cycles pushed the math
        thread's ``SETC16`` of ``DEST_TARGET_REG_CFG_MATH_Offset`` behind two of
        its own ``MVMUL``s, which then accumulated into the wrong half of Dst
        and made ``matmul_block`` compute a wrong answer. See ``config.py`` and
        :meth:`occupy_for`.
        """
        if self.busy_until is not None:
            self._release_expired(cycle_num)
        # The longest occupancy charged across this cycle's batch, per IPC
        # group. Each group is held by whichever of its parallel paths is
        # slowest, so ``max`` within a group; a unit with no published groups
        # keys everything under ``None`` and this is the single whole-unit
        # deadline it always was. Left as ``None`` until something is actually
        # charged, which for every unit but Blackhole's config unit is never.
        batch = None
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
                # ``> 1`` rather than ``is not None``: occupy_for no-ops at one
                # cycle, so filtering here keeps the dict unallocated on every
                # path that charges nothing, which is every path today outside
                # Blackhole's CFGSHIFTMASK and ThCon's multi-cycle ops.
                if occupancy is not None and occupancy > 1:
                    group = self.instruction_group(instruction_name)
                    if batch is None:
                        batch = {group: occupancy}
                    elif occupancy > batch.get(group, 0):
                        batch[group] = occupancy
            else:
                raise NotImplementedError(
                    f"{self.unit_name} unit can not handle instruction '{instruction_info['name']}'"
                )
        if batch is not None:
            for group, cycles in batch.items():
                # The hold runs from *acceptance*, not from this retire tick.
                # Everything in this batch was accepted in the previous cycle
                # (backend units tick before the wait gates within a cycle, so
                # a gate's issue lands in the unit's next tick), and an
                # occupancy is the minimum interval between acceptances: a
                # 3-cycle ThCon op must let the next one in 3 cycles after
                # this one entered, not 3 cycles after it retired. Arming from
                # ``cycle_num`` made every multi-cycle occupancy cost one
                # extra cycle at the issuing thread — 4 for ThCon's
                # documented 3, where Blackhole silicon measures 2.97 — which
                # is over-charging, the direction the cost model's floor
                # policy forbids.
                self.occupy_for(cycle_num - 1, cycles, group)

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
