from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.util import TensixConfigurationConstants, TensixInstructionDecoder
from tt_sim.perf.model import unit_cost_model
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TensixBackendConfigurationUnit(TensixBackendUnit, MemMapable):
    """
    Backend configuration unit as per description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/ConfigurationUnit.md
    """

    OPCODE_TO_HANDLER = {
        "SETC16": "handle_setc16",
        "RMWCIB0": "handle_rmwcib0",
        "RMWCIB1": "handle_rmwcib1",
        "RMWCIB2": "handle_rmwcib2",
        "RMWCIB3": "handle_rmwcib3",
        "WRCFG": "handle_wrcfg",
        "RDCFG": "handle_rdcfg",
        # Blackhole only
        "CFGSHIFTMASK": "handle_cfgshiftmask",
    }
    # The only CFGSHIFTMASK mode the vendor reference simulator (ttsim
    # ``src/tensix.cpp``) models: no circular shift, a full-width mask, the
    # "old + scratch" operation and no masking of the old value. Every other
    # combination raises there too, so tt-sim has nothing to port and no oracle
    # to check an implementation against — fail loudly instead of guessing.
    # (right_cshift_amt, mask_width, operation, disable_mask_on_old_val)
    CFGSHIFTMASK_MODELLED_MODE = (0, 31, 3, 1)
    # Per-arch sizes of the backend config state (number of 128-bit config
    # registers) and the per-thread config state. Wormhole defaults; Blackhole
    # is larger (56 / 68 — ttsim ``sim.h`` TT_ARCH_VERSION 1), threaded in via
    # the arch profile.
    CFG_STATE_SIZE = 47
    THD_STATE_SIZE = 57

    def __init__(
        self, backend, gprs, cfg_state_size=None, thd_state_size=None, blackhole=False
    ):
        # Select the arch-specific config-register layout (name -> ADDR32 / SHAMT
        # / MASK) that every backend unit reads through TensixConfigurationConstants.
        TensixConfigurationConstants.use_blackhole(blackhole)
        self.CFG_STATE_SIZE = (
            cfg_state_size if cfg_state_size is not None else self.CFG_STATE_SIZE
        )
        self.THD_STATE_SIZE = (
            thd_state_size if thd_state_size is not None else self.THD_STATE_SIZE
        )
        self.config = [
            [0] * (self.CFG_STATE_SIZE * 4),
            [0] * (self.CFG_STATE_SIZE * 4),
        ]
        self.threadConfig = [
            [0] * self.THD_STATE_SIZE,
            [0] * self.THD_STATE_SIZE,
            [0] * self.THD_STATE_SIZE,
        ]
        self.gprs = gprs
        self.prev_cycle_setc16_or_wrcfg = False
        # Per-thread cycle at which that thread's most recent Configuration Unit
        # instruction leaves the pipeline, or 0 for "nothing of this thread's is
        # in a stage". Written only by ``_arm_residency``, which runs only with
        # the cost model on -- so with ``TT_SIM_COST_MODEL`` unset this stays
        # ``[0, 0, 0]`` for the life of the device and every predicate below
        # answers exactly what it answered before. See ``_arm_residency``.
        self._pipeline_exit = [0, 0, 0]
        # The config word holding Blackhole's DEST_ACCESS_CFG (ADDR32 220), or
        # None on Wormhole, where the register does not exist. Writes to it are
        # mirrored into the Dst register's row-remap gates; see
        # ``_apply_dest_access_cfg``.
        self.dest_access_cfg_idx = (
            TensixConfigurationConstants.get_addr32("DEST_ACCESS_CFG_remap_addrs")
            if TensixConfigurationConstants.exists("DEST_ACCESS_CFG_remap_addrs")
            else None
        )
        super().__init__(
            backend, TensixBackendConfigurationUnit.OPCODE_TO_HANDLER, "Config"
        )
        # Wired to the cycle-cost tables (Phase 5 of
        # docs/plans/event-driven-pump.md) on 2026-08-04, last of the six
        # units the plan named and the only one that had to be wired twice.
        # ``None`` unless TT_SIM_COST_MODEL is set.
        #
        # ON WORMHOLE THIS CHARGES NOTHING: every entry in that arch's table is
        # one cycle, and ``occupy_for`` no-ops at <= 1. ON BLACKHOLE IT DOES --
        # CFGSHIFTMASK is a 2-cycle occupancy (throughput_ipc 0.5, "requires two
        # cycles in stage 0") and the `untilize` guard executes it 32 times, so
        # this is a real timing change and it was gated as one:
        # driver/tests/cost_model_gate.py, RESULT: PASS, with every
        # budget-dependent guard needing exactly the poll-budget multiplier it
        # needed before.
        self.cost_model = unit_cost_model(
            "CFG", "blackhole" if blackhole else "wormhole"
        )
        #
        # THE ORDERING BUG THIS UNIT FOUND IS FIXED, and it is the reason the
        # wiring is safe rather than merely tested.
        #
        # The bug, for the record. Until 2026-08-04 RDCFG's *occupancy* in the
        # table read ">= 2": the Wormhole page's documented **latency**, copied
        # into the occupancy field because no separate throughput figure had
        # been found for it. Charging that 2 made `matmulblock` compute the
        # WRONG ANSWER (608.0 where the golden is 1120.0). The mechanism was in
        # TensixBackendUnit.clock_tick, not here: this unit accepts a whole
        # *batch* per cycle -- up to three SETC16, one per thread, alongside one
        # shared-IPC-group op -- and the old drain armed ``busy_until``
        # mid-batch and returned, leaving the rest of the batch queued for a
        # later cycle. The threads had already been told those instructions
        # were accepted and had moved on, so the math thread's SETC16 of
        # DEST_TARGET_REG_CFG_MATH_Offset landed two cycles late, behind two of
        # that same thread's MVMULs, which accumulated into the wrong half of
        # Dst. An already-accepted write being overtaken by its own thread's
        # later instructions is the missing ordering guarantee, and the fix is
        # to retire the batch and only then hold the unit -- occupancy is
        # throughput back-pressure on the *next* instruction, while each
        # accepted instruction commits at its own documented latency, which is
        # what both arches' Configuration Unit pages describe (Blackhole names
        # the pipeline stages, -4..+1) and what the vendor reference simulator
        # (ttsim, libttsim.cpp's per-pipe issue loop) enforces absolutely by
        # never letting a thread's next instruction start until this one's
        # side effects are committed. See backend_base.TensixBackendUnit.
        #
        # THE DIVERGENCE THE BUG EXPOSED IS NOT FIXED, only made unreachable
        # from these tables: nothing in tt-sim orders a config write against the
        # units that read it, because tt-sim's config writes land instantly and
        # nothing had ever made one late. With RDCFG corrected to 1 no table
        # entry delays a *config write* any more -- CFGSHIFTMASK does hold the
        # unit for two cycles, but only ever behind its own already-committed
        # write. A future entry that delayed a SETC16 or a WRCFG would meet the
        # same missing guarantee, and the failure would look like a wrong
        # answer, not a slow one. That is why this paragraph outlives the fix.
        #
        # THE OVER-CHARGE THIS WIRING USED TO CARRY IS GONE, and the shape of
        # the fix is worth keeping because it is the only per-group constraint
        # in the tree. The unit's constraints are cross-thread: SETC16 is one
        # per thread per cycle, RMWCIB needs neither RDCFG nor WRCFG issued in
        # the previous cycle by *any* thread, and Blackhole folds everything but
        # SETC16 into one shared IPC group. ``busy_until`` was a WHOLE-UNIT
        # hold, so a cycle in which CFGSHIFTMASK was charged 2 also refused the
        # next cycle's SETC16 -- which the Blackhole page places OUTSIDE that
        # IPC group and lets through. Occupancy is now charged PER IPC GROUP
        # (TensixBackendUnit.occupy_for / is_occupied, keyed by the ``ipc_group``
        # column of the Blackhole table, which the cost YAML transcribes), so
        # CFGSHIFTMASK holds ``Config`` and leaves ``ThreadConfig`` free.
        #
        # IT COST NOTHING TO CARRY AND NOTHING TO FIX. No in-tree guard ever
        # had an issue refused by the whole-unit hold -- ``untilize``, the only
        # workload in the tree that executes CFGSHIFTMASK at all, arms the hold
        # 32 times and refuses zero -- so both the over-charge and its removal
        # measure 12,189 cycles there. The fix is for the mechanism's
        # correctness, not for a number that moved.
        #
        # WORMHOLE HAS NO GROUPS HERE and deliberately gets none: its page
        # states the same partition as prose without a group column, and a
        # grouping inferred from prose is exactly the kind of guess that lets
        # an instruction through that the hardware would stall. It is inert
        # either way -- every Wormhole entry is one cycle, so nothing arms.
        #
        # tt-sim models the first two constraints in ``issueInstruction`` /
        # ``prev_cycle_setc16_or_wrcfg``; neither is expressible as a number
        # attached to an opcode.
        #
        # THE UNIT ALSO READS ITS **LATENCY** COLUMN, since 2026-08-12, and that
        # is a different column from everything above. Occupancy answers "when
        # may the next instruction enter"; latency answers "when has this one
        # left", and on this unit they are not the same number -- ``WRCFG`` is
        # latency 2, occupancy 1, so the unit takes the next instruction a cycle
        # before the previous one is out. Nothing consumed the latency column
        # until now, so every instruction here was reported as having left the
        # unit in the tick after it was accepted, and ``STALLWAIT``'s C12 ("any
        # thread has an instruction in any stage of the Configuration Unit
        # pipeline") could not be observed by another thread at all. See
        # ``hasInflightInstructionsFromThread`` and ``_arm_residency``: the
        # residency is reporting only, and the writes still commit exactly where
        # they always did.

    @property
    def pipeline_exit_cycles(self):
        """Per-thread cycle at which the pipeline empties, 0 for "not resident".

        Exposed read-only for tests and diagnostics; the residency itself is
        consumed through :meth:`hasInflightInstructionsFromThread`.
        """
        return tuple(self._pipeline_exit)

    def is_clock_idle(self):
        # The override also clears prev_cycle_setc16_or_wrcfg, which is
        # observable, so it must already be clear for the tick to be a no-op.
        # A live residency is a state change still owed (the pipeline exit), and
        # ``_expire_residency`` only runs from a tick, so the unit is not idle
        # while one is outstanding -- the same reason the unpacker is not idle
        # while it owes a deferred Src hand-over.
        return (
            not self.next_instruction
            and not self.prev_cycle_setc16_or_wrcfg
            and not (
                self._pipeline_exit[0]
                or self._pipeline_exit[1]
                or self._pipeline_exit[2]
            )
        )

    def hasInflightInstructionsFromThread(self, from_thread):
        """True while this thread's instruction is still in a pipeline stage.

        ``STALLWAIT.md`` (Blackhole) states condition **C12** as "Any thread has
        an instruction in any stage of the Configuration Unit pipeline", and the
        Configuration Unit page says how long a stage sequence lasts: it is the
        **Latency** column, 1 cycle for ``SETC16`` and ``RMWCIB``, 2 for
        ``WRCFG`` and ``CFGSHIFTMASK``, ≥ 2 for ``RDCFG``, ≥ 5 for
        ``STREAMWRCFG``. The base implementation answers from the issue queue
        alone, which this unit drains in the tick after acceptance -- so every
        instruction here was reported as having left the unit after one cycle,
        whatever its documented latency, and C12 could never be observed by
        another thread's Wait Gate at all.

        **Residency is not occupancy**, and this unit is the clearest case in
        the tree: ``WRCFG`` is *latency 2, occupancy 1*, i.e. the unit accepts
        the next instruction a cycle before this one has left. Reading
        ``busy_until`` here would therefore be wrong in both directions, and the
        unpacker's note on the same predicate says as much ("an instruction can
        be in a stage of the pipeline long after the unit will accept the next
        one ... Nothing here licenses reading another unit's ``busy_until`` as
        residency; that would need each unit's own latency"). This is that
        unit's own latency.
        """
        if self._pipeline_exit[from_thread]:
            return True
        return super().hasInflightInstructionsFromThread(from_thread)

    def clock_tick(self, cycle_num):
        self.prev_cycle_setc16_or_wrcfg = self.checkIfNextInstructionsContainOpcodes(
            "SETC16", "WRCFG"
        )
        model = self.cost_model
        # The batch has to be captured *before* the drain, because the drain
        # empties it -- and residency is a property of what was accepted, not of
        # what the handlers did.
        batch = list(self.next_instruction) if model is not None else ()
        if model is not None:
            self._expire_residency(cycle_num)
        super().clock_tick(cycle_num)
        if batch:
            self._arm_residency(cycle_num, batch)

    def _expire_residency(self, cycle_num):
        """Drop every thread whose pipeline exit has arrived.

        Run at the top of the tick, before anything is armed, so a deadline is
        released in exactly the cycle it names -- which is what makes the window
        a deadline rather than a tick-order artefact. Backend units tick before
        the Wait Gates, so a gate evaluating C12 in cycle ``c`` sees the state
        this call left, whichever thread's gate it is and in whatever order the
        units ticked.
        """
        exits = self._pipeline_exit
        for thread in range(3):
            if exits[thread] and cycle_num >= exits[thread]:
                exits[thread] = 0

    def _arm_residency(self, cycle_num, batch):
        """Hold each thread resident for its instruction's documented latency.

        **From acceptance, not from this retire tick.** Everything in ``batch``
        was accepted in the previous cycle -- backend units tick before the Wait
        Gates, so a gate's issue lands in the unit's *next* tick -- and the
        Latency column is measured from issue. This is the same anchor
        ``TensixBackendUnit.clock_tick`` uses for occupancy, and for the same
        reason: anchoring at ``cycle_num`` would charge every entry one cycle
        more than the document prints.

        **The side effects are untouched.** The handlers have already run, in
        the tick they always ran in; this only decides how long the instruction
        is *reported* as being in the unit. Delaying the config write itself is
        a different change and a dangerous one -- nothing in tt-sim orders a
        config write against the units that read it, so a late ``SETC16`` is
        overtaken by its own thread's later instructions and the failure is a
        wrong answer rather than a slow one. That is the bug recorded at length
        in ``__init__`` above, and this deliberately does not go near it.

        A latency of 1 arms nothing: the deadline lands on this very tick, and
        the instruction was already visible through ``next_instruction`` for the
        cycle between acceptance and now. So ``SETC16`` and ``RMWCIB`` keep the
        exact behaviour they have always had, and only the pipelined opcodes --
        ``WRCFG``, ``RDCFG``, ``CFGSHIFTMASK``, ``STREAMWRCFG`` -- extend it.

        Bounds are charged at their low end by the model's ``BOUND_POLICY``, so
        ``RDCFG``'s "≥ 2 cycles" arms 2. For a residency the low end is the
        *under*-reporting end, which is the safe one: a residency
        held too long makes a ``STALLWAIT`` on C12 wait for a stall the hardware
        does not have, and inventing back-pressure is the single direction this
        project's bounds policy forbids.
        """
        model = self.cost_model
        accepted = cycle_num - 1
        exits = self._pipeline_exit
        for instruction, thread in batch:
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            cycles = model.latency(instruction_info["name"])
            if cycles is None:
                # No published latency: no opinion, and the pre-existing
                # same-cycle report stands. Nothing in this unit's table is in
                # that position today.
                continue
            deadline = accepted + cycles
            if deadline > cycle_num and deadline > exits[thread]:
                exits[thread] = deadline

    def issueInstruction(self, instruction, from_thread):
        # An occupied IPC GROUP takes nothing, however many slots the per-opcode
        # rules below would otherwise allow. See TensixBackendUnit.is_occupied
        # -- this is the unit where deferring instead of refusing was shown to
        # reorder a thread's program.
        #
        # Per group rather than per unit because this is the one unit whose
        # documentation publishes the groups: Blackhole's page tabulates an
        # "IPC group" column, SETC16 alone in ThreadConfig and everything else
        # in Config. A CFGSHIFTMASK therefore holds Config for its two cycles
        # and leaves the next cycle's SETC16 free, which is what the hardware
        # does and what the whole-unit hold got wrong. instruction_group is
        # called rather than issue_group because the name is already decoded.
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        instruction_name = instruction_info["name"]
        if self.busy_until is not None and self.is_occupied(
            self.instruction_group(instruction_name)
        ):
            return self._refuse("unit_busy")
        if instruction_name == "SETC16":
            if self.next_instruction.count("SETC16") < 3:
                # Max one per thread per cycle allowed
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                return self._refuse("issue_slot_taken")
        elif instruction_name == "WRCFG":
            if self.next_instruction.count("WRCFG") == 0:
                # Max one in total per cycle allowed
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                return self._refuse("issue_slot_taken")
        else:
            if (
                not self.prev_cycle_setc16_or_wrcfg
                and not self.checkIfNextInstructionsContainAnyOtherOpcodes("SETC16")
            ):
                self.next_instruction.append(
                    (
                        instruction,
                        from_thread,
                    )
                )
                return True
            else:
                # The config unit's own throughput rule: a SETC16/WRCFG in the
                # previous cycle blocks this cycle's other-opcode issue.
                return self._refuse("unit_busy")

    def setConfig(self, stateID, cfgIndex, value, from_thread=None):
        if self.getDiagnosticSettings().reportConfigurationSet():
            frm_thread = f"from thread {from_thread}"
            print(
                f"Config: set config [{stateID}]{TensixConfigurationConstants.get_name(cfgIndex)} "
                f"value={hex(value)} {frm_thread if from_thread is not None else ''}"
            )
        self.config[stateID][cfgIndex] = value
        if cfgIndex == self.dest_access_cfg_idx:
            self._apply_dest_access_cfg(value)

    def _apply_dest_access_cfg(self, value):
        """Mirror a ``DEST_ACCESS_CFG`` write into the Dst row-remap gates.

        ``DstRegister.adj16`` / ``adj32`` implement BlackholeA0 Dst.md's row
        transforms gated on ``DEST_ACCESS_CFG_remap_addrs`` /
        ``DEST_ACCESS_CFG_swizzle_32b``; this is what sets those gates. Until
        the Blackhole ``pack_untilize`` path arrived nothing wrote the register,
        so the transforms sat inert (remap off is the identity, swizzle off is
        the shared 32-bit fold) — the LLK's ``_llk_math_reconfig_remap_`` is the
        first caller, setting both bits together for the packer's stride-of-16
        Dst read.

        Dst is one physical register file with one pair of gates, so the most
        recent write wins whichever config state carried it: the gates are not
        tracked per ``CFG_STATE_ID``. The LLK fences around every toggle
        (``tensix_sync`` plus a wait on the MATH_PACK semaphore, because
        "changing them mid-access would corrupt reads"), so no Dst access
        straddles a change and the distinction is unobservable in practice.
        """
        dst = self.backend.getDst()
        dst.dest_remap_addrs = bool(
            TensixConfigurationConstants.parse_raw_config_value(
                value, "DEST_ACCESS_CFG_remap_addrs"
            )
        )
        dst.dest_swizzle_32b = bool(
            TensixConfigurationConstants.parse_raw_config_value(
                value, "DEST_ACCESS_CFG_swizzle_32b"
            )
        )

    def setThreadConfig(self, thread_id, cfg_index, value):
        if self.getDiagnosticSettings().reportConfigurationSet():
            print(
                f"Config: set threadConfig [{thread_id}]{TensixConfigurationConstants.get_name(cfg_index)} "
                f"value={hex(value)}"
            )
        self.threadConfig[thread_id][cfg_index] = value

    def handle_rdcfg(self, instruction_info, issue_thread, instr_args):
        cfgIndex = instr_args["CfgReg"]
        assert cfgIndex < self.CFG_STATE_SIZE * 4

        resultReg = instr_args["GprAddress"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        self.gprs.getRegisters(issue_thread)[resultReg] = self.config[stateID][cfgIndex]

    def handle_wrcfg(self, instruction_info, issue_thread, instr_args):
        cfgIndex = instr_args["CfgReg"]
        assert cfgIndex < self.CFG_STATE_SIZE * 4

        inputReg = instr_args["GprAddress"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        if instr_args["wr128b"]:
            for i in range(4):
                self.setConfig(
                    stateID,
                    cfgIndex + i,
                    self.gprs.getRegisters(issue_thread)[inputReg + i],
                    issue_thread,
                )
        else:
            self.setConfig(
                stateID,
                cfgIndex,
                self.gprs.getRegisters(issue_thread)[inputReg],
                issue_thread,
            )

    def handle_setc16(self, instruction_info, issue_thread, instr_args):
        cfg_index = instr_args["setc16_reg"]
        new_value = instr_args["setc16_value"]

        assert cfg_index < self.THD_STATE_SIZE

        self.setThreadConfig(issue_thread, cfg_index, new_value)

    def handle_rmwcib0(self, instruction_info, issue_thread, instr_args):
        self.handle_rmwcib(instruction_info, issue_thread, instr_args, 0)

    def handle_rmwcib1(self, instruction_info, issue_thread, instr_args):
        self.handle_rmwcib(instruction_info, issue_thread, instr_args, 1)

    def handle_rmwcib2(self, instruction_info, issue_thread, instr_args):
        self.handle_rmwcib(instruction_info, issue_thread, instr_args, 2)

    def handle_rmwcib3(self, instruction_info, issue_thread, instr_args):
        self.handle_rmwcib(instruction_info, issue_thread, instr_args, 3)

    def handle_rmwcib(self, instruction_info, issue_thread, instr_args, index1):
        index4 = instr_args["CfgRegAddr"]
        new_value = instr_args["Data"] << (8 * index1)
        mask = instr_args["Mask"] << (8 * index1)

        assert index4 < self.CFG_STATE_SIZE * 4

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        existing_val = self.config[stateID][index4]
        new_value = (new_value & mask) | (existing_val & ~mask)

        self.setConfig(stateID, index4, new_value, issue_thread)

    def handle_cfgshiftmask(self, instruction_info, issue_thread, instr_args):
        """Blackhole CFGSHIFTMASK: combine a CFG scratch register with a CFG
        register, storing the result back in that CFG register.

        A verbatim port of ttsim's ``TENSIX_EXECUTE_CFGSHIFTMASK`` (which itself
        only models ``CFG[CfgReg] += SCRATCH[scratch_sel]``); the full shift/mask
        pseudocode is in this instruction's description in
        ``tensix_instructions.yaml`` for whenever a kernel needs another mode.
        """
        if not self.backend.blackhole:
            raise NotImplementedError(
                "CFGSHIFTMASK is a Blackhole-only instruction (it has no Wormhole "
                "encoding, and the Wormhole config layout has no CFG scratch registers)"
            )

        mode = (
            instr_args["right_cshift_amt"],
            instr_args["mask_width"],
            instr_args["operation"],
            instr_args["disable_mask_on_old_val"],
        )
        if mode != self.CFGSHIFTMASK_MODELLED_MODE:
            raise NotImplementedError(
                f"CFGSHIFTMASK (right_cshift_amt, mask_width, operation, "
                f"disable_mask_on_old_val)={mode} is not modelled in tt-sim; only "
                f"{self.CFGSHIFTMASK_MODELLED_MODE} (add the scratch register into the "
                f"CFG register) is, matching the vendor reference simulator"
            )

        # scratch_sel 0b11 means "select the scratch register of my thread".
        scratch_sel = instr_args["scratch_sel"]
        scratch_index = issue_thread if scratch_sel == 3 else scratch_sel
        assert scratch_index < 3

        cfg_index = instr_args["CfgReg"]
        assert cfg_index < self.CFG_STATE_SIZE * 4

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        scratch_value = self.getConfigValue(stateID, f"SCRATCH_SEC{scratch_index}_val")
        new_value = (self.config[stateID][cfg_index] + scratch_value) & 0xFFFFFFFF

        self.setConfig(stateID, cfg_index, new_value, issue_thread)

    def get_threadConfig_entry(self, thread, entry_idx):
        return self.threadConfig[thread][entry_idx]

    def get_config_entry(self, state, entry_idx):
        return self.config[state][entry_idx]

    def read(self, addr, size):
        threadConfigStart = self.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = self.CFG_STATE_SIZE * 4
            second_idx = 1 if idx > self.CFG_STATE_SIZE * 4 else 0
            first_idx = int(idx - (each_config_size * second_idx))
            return conv_to_bytes(self.config[second_idx][first_idx])
        else:
            idx = idx - threadConfigStart
            second_idx = idx / self.THD_STATE_SIZE
            return conv_to_bytes(
                self.threadConfig[second_idx][
                    idx - ((self.THD_STATE_SIZE) * second_idx)
                ]
            )

    def write(self, addr, value, size=None):
        threadConfigStart = self.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = self.CFG_STATE_SIZE * 4
            second_idx = 1 if idx > each_config_size else 0
            first_idx = int(idx - (each_config_size * second_idx))
            self.setConfig(second_idx, first_idx, conv_to_uint32(value))
        else:
            raise ValueError("Can not write to thread config from RISC-V core")

    def getSize(self):
        return 0xFFFF
