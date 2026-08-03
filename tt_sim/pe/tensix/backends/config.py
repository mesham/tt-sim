from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.util import TensixConfigurationConstants, TensixInstructionDecoder
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
        # NOT wired to the cycle-cost tables (Phase 5 of
        # docs/plans/event-driven-pump.md), and this is the one unit whose
        # omission is a finding rather than a scoping decision.
        #
        # Both arches publish this unit's table in full, and everything tt-sim
        # implements is one cycle except RDCFG, which the Wormhole page gives
        # ">= 2". Charging that documented 2 makes `matmulblock` compute the
        # wrong answer: the occupancy delays five SETC16s on the math thread
        # and four WRCFGs on the pack thread by one cycle each, and something
        # downstream reads config that has not landed yet. Refusing the issue
        # outright (TensixBackendUnit.is_occupied) rather than deferring it
        # does not fix it either, so this is not the frontend reordering one
        # thread's stream -- it is a missing ordering guarantee between a
        # config write and the units that read it, of the kind the ISA docs'
        # config-visibility rules and the LLK's "WRCFG takes 2 cycles" padding
        # both exist to express. That is worth fixing on its own terms, and it
        # is worth *not* hiding by charging RDCFG a 1 the docs do not give it.
        #
        # The other reason this unit is a poor fit for a per-opcode occupancy:
        # its real constraints are cross-thread. SETC16 is one per thread per
        # cycle, RMWCIB needs neither RDCFG nor WRCFG issued in the previous
        # cycle by *any* thread, and Blackhole folds everything but SETC16
        # into one shared IPC group. tt-sim models the first two in
        # ``issueInstruction`` / ``prev_cycle_setc16_or_wrcfg``; none of them
        # is expressible as a number attached to an opcode.

    def is_clock_idle(self):
        # The override also clears prev_cycle_setc16_or_wrcfg, which is
        # observable, so it must already be clear for the tick to be a no-op.
        return not self.next_instruction and not self.prev_cycle_setc16_or_wrcfg

    def clock_tick(self, cycle_num):
        self.prev_cycle_setc16_or_wrcfg = self.checkIfNextInstructionsContainOpcodes(
            "SETC16", "WRCFG"
        )
        super().clock_tick(cycle_num)

    def issueInstruction(self, instruction, from_thread):
        # An occupied unit takes nothing, however many slots the per-opcode
        # rules below would otherwise allow. See TensixBackendUnit.is_occupied
        # -- this is the unit where deferring instead of refusing was shown to
        # reorder a thread's program.
        if self.is_occupied():
            return False
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        instruction_name = instruction_info["name"]
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
                return False
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
                return False
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
                return False

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
