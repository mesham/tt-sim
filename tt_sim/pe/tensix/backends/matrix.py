from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.util.bits import extract_bits, get_nth_bit


class MatrixUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {
        "ZEROACC": "handle_zeroacc",
        "SETRWC": "handle_setrwc",
        "ELWADD": "handle_elwadd",
    }

    def __init__(self, backend):
        super().__init__(backend, MatrixUnit.OPCODE_TO_HANDLER, "Matrix")

    def handle_elwadd(self, instruction_info, issue_thread, instr_args):
        pass

    def handle_setrwc(self, instruction_info, issue_thread, instr_args):
        rcw = self.getRCW(issue_thread)

        srca = get_nth_bit(instr_args["BitMask"], 0)
        srcb = get_nth_bit(instr_args["BitMask"], 1)
        dst = get_nth_bit(instr_args["BitMask"], 2)
        fidelity = get_nth_bit(instr_args["BitMask"], 3)

        srca_cr = get_nth_bit(instr_args["rwc_cr"], 0)
        srcb_cr = get_nth_bit(instr_args["rwc_cr"], 1)
        dst_cr = get_nth_bit(instr_args["rwc_cr"], 2)
        dst_c_to_cr = get_nth_bit(instr_args["rwc_cr"], 3)

        flipsrca = get_nth_bit(instr_args["clear_ab_vld"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_ab_vld"], 1)

        SrcAVal = instr_args["rwc_a"]
        SrcBVal = instr_args["rwc_b"]
        DstVal = instr_args["rwc_d"]

        if srca:
            if srca_cr:
                SrcAVal += rcw.SrcA_Cr
            rcw.SrcA = SrcAVal
            rcw.SrcA_Cr = SrcAVal

        if srcb:
            if srcb_cr:
                SrcBVal += rcw.SrcB_Cr
            rcw.SrcB = SrcBVal
            rcw.SrcB_Cr = SrcBVal

        if dst or dst_c_to_cr:
            if dst_c_to_cr:
                DstVal += rcw.Dst
            elif dst_cr:
                DstVal += rcw.Dst_Cr
            rcw.Dst = DstVal
            rcw.Dst_Cr = DstVal

        if fidelity:
            rcw.FidelityPhase = 0

        if flipsrca:
            if not self.getThreadConfigValue(issue_thread, "CLR_DVALID_SrcA_Disable"):
                # TODO: SrcA[MatrixUnit.SrcABank].AllowedClient = SrcClient::Unpackers;
                # MatrixUnit.SrcABank ^= 1;
                pass

        if flipsrcb:
            if not self.getThreadConfigValue(issue_thread, "CLR_DVALID_SrcB_Disable"):
                # TODO: SrcB[MatrixUnit.SrcABank].AllowedClient = SrcClient::Unpackers;
                # MatrixUnit.SrcBBank ^= 1;
                pass

    def handle_zeroacc(self, instruction_info, issue_thread, instr_args):
        ZEROACC_MODE_ONE_ROW = 0
        ZEROACC_MODE_16_ROWS = 1
        ZEROACC_MODE_HALF_OF_DST = 2
        ZEROACC_MODE_ALL_OF_DST = 3

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
