from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.util import TensixInstructionDecoder
from tt_sim.util.bits import replace_bits
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TensixBackendConfigurationUnit(TensixBackendUnit, MemMapable):
    OPCODE_TO_HANDLER = {
        "SETC16": "handle_setc16",
        "RMWCIB0": "handle_rmwcib0",
        "RMWCIB1": "handle_rmwcib1",
        "RMWCIB2": "handle_rmwcib2",
        "RMWCIB3": "handle_rmwcib3",
        "WRCFG": "handle_wrcfg",
    }
    CFG_STATE_SIZE = 47
    THD_STATE_SIZE = 57

    def __init__(self, backend, gprs):
        self.config = [[0] * TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4] * 2
        self.threadConfig = [[0] * TensixBackendConfigurationUnit.THD_STATE_SIZE] * 3
        self.gprs = gprs
        self.prev_cycle_setc16_or_wrcfg = False
        super().__init__(
            backend, TensixBackendConfigurationUnit.OPCODE_TO_HANDLER, "Config"
        )

    def clock_tick(self, cycle_num):
        self.prev_cycle_setc16_or_wrcfg = self.checkIfNextInstructionsContainOpcodes(
            "SETC16", "WRCFG"
        )
        super().clock_tick(cycle_num)

    def issueInstruction(self, instruction, from_thread):
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

    def handle_wrcfg(self, instruction_info, issue_thread, instr_args):
        cfgIndex = instr_args["CfgReg"]
        assert cfgIndex < TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4

        inputReg = instr_args["GprAddress"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        if instr_args["wr128b"]:
            for i in range(4):
                self.config[stateID][cfgIndex + i] = self.gprs.getRegisters(
                    issue_thread
                )[inputReg + i]
        else:
            self.config[stateID][cfgIndex] = self.gprs.getRegisters(issue_thread)[
                inputReg
            ]

    def handle_setc16(self, instruction_info, issue_thread, instr_args):
        cfg_index = instr_args["setc16_reg"]
        new_value = instr_args["setc16_value"]

        assert cfg_index < TensixBackendConfigurationUnit.THD_STATE_SIZE

        self.threadConfig[issue_thread][cfg_index] = new_value

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
        new_value = instr_args["Data"]
        mask = instr_args["Mask"]

        assert index4 < TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        existing_val = self.config[stateID][index4]

        new_value = new_value & mask
        replaced_value = replace_bits(existing_val, new_value, index1 * 8, 8)

        self.config[stateID][index4] = replaced_value

    def get_threadConfig_entry(self, thread, entry_idx):
        return self.threadConfig[thread][entry_idx]

    def get_config_entry(self, state, entry_idx):
        return self.config[state][entry_idx]

    def read(self, addr, size):
        threadConfigStart = TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4
            second_idx = (
                1 if idx > TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4 else 0
            )
            first_idx = int(idx - (each_config_size * second_idx))
            return conv_to_bytes(self.config[second_idx][first_idx])
        else:
            idx = idx - threadConfigStart
            second_idx = idx / TensixBackendConfigurationUnit.THD_STATE_SIZE
            return conv_to_bytes(
                self.threadConfig[second_idx][
                    idx - ((TensixBackendConfigurationUnit.THD_STATE_SIZE) * second_idx)
                ]
            )

    def write(self, addr, value, size=None):
        threadConfigStart = TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = TensixBackendConfigurationUnit.CFG_STATE_SIZE * 4
            second_idx = 1 if idx > each_config_size else 0
            first_idx = int(idx - (each_config_size * second_idx))
            self.config[second_idx][first_idx] = conv_to_uint32(value)
        else:
            idx = idx - threadConfigStart
            second_idx = int(idx / TensixBackendConfigurationUnit.THD_STATE_SIZE)
            self.threadConfig[second_idx][
                idx - ((TensixBackendConfigurationUnit.THD_STATE_SIZE) * second_idx)
            ] = conv_to_uint32(value)

    def getSize(self):
        return 0xFFFF
