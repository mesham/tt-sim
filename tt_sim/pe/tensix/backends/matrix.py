from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.util.bits import extract_bits


class MatrixUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        if len(self.instruction_buffer) > 0:
            instruction, issue_thread = self.instruction_buffer.pop(0)
            instruction_info = (
                self.backend.tensix_instruction_decoder.getInstructionInfo(instruction)
            )
            if instruction_info["name"] == "ZEROACC":
                self.handle_zeroacc(instruction_info, issue_thread)
            else:
                raise NotImplementedError(
                    f"Matrix unix can not handle instruction '{instruction_info['name']}'"
                )

    def handle_zeroacc(self, instruction_info, issue_thread):
        ZEROACC_MODE_ONE_ROW = 0
        ZEROACC_MODE_16_ROWS = 1
        ZEROACC_MODE_HALF_OF_DST = 2
        ZEROACC_MODE_ALL_OF_DST = 3

        instr_args = instruction_info["instr_args"]

        mode = extract_bits(instr_args["clear_mode"], 2, 0)
        useDst32b = extract_bits(instr_args["clear_mode"], 1, 2)
        imm10 = instr_args["dst"]

        if mode == ZEROACC_MODE_ONE_ROW:
            state_id = self.getThreadConfigValue(issue_thread, "CFG_STATE_ID_StateID")
            row = imm10
            row += self.getThreadConfigValue(
                issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
            )
            row += self.getRCW(issue_thread).Dst + self.getConfigValue(
                state_id, "DEST_REGW_BASE_Base"
            )

            if (
                self.getConfigValue(state_id, "ALU_ACC_CTRL_Fp32_enabled") == 1
                or self.getConfigValue(state_id, "ALU_ACC_CTRL_INT8_math_enabled") == 1
            ):
                self.getDst().setUndefinedRow(row, True)
            else:
                self.getDst().setUndefinedRow(row)
        else:
            if mode == ZEROACC_MODE_16_ROWS:
                imm10 &= 0xFF
                if useDst32b:
                    if imm10 < 32:
                        for i in range(16):
                            self.getDst().setUndefinedRow(imm10 * 16 + i, True)
                else:
                    if imm10 < 64:
                        for i in range(16):
                            self.getDst().setUndefinedRow(imm10 * 16 + i, False)
            elif mode == ZEROACC_MODE_HALF_OF_DST:
                if imm10 & 1:
                    # High half
                    for i in range(512, 1024):
                        self.getDst().setUndefinedRow(i)
                else:
                    # Low half
                    for i in range(512):
                        self.getDst().setUndefinedRow(i)
            elif mode == ZEROACC_MODE_ALL_OF_DST:
                #  Mode == ZEROACC_MODE_ALL_OF_DST
                for i in range(1024):
                    self.getDst().setUndefinedRow(i)
            else:
                raise ValueError(f"Unknown mode {hex(mode)}")

        if mode == ZEROACC_MODE_ONE_ROW or mode == ZEROACC_MODE_16_ROWS:
            addr_mod = extract_bits(instr_args["AddrMode"], 2, 0)
            self.getRCW(issue_thread).applyAddrMod(issue_thread, addr_mod)
