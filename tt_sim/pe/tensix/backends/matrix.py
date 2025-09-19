from tt_sim.pe.tensix.backends.backend_base import DataFormat, TensixBackendUnit
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.util.bits import extract_bits, get_nth_bit


class MatrixUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {
        "ZEROACC": "handle_zeroacc",
        "SETRWC": "handle_setrwc",
        "ELWADD": "handle_elwadd",
        "ZEROSRC": "handle_zerosrc",
        "INCRWC": "handle_incrwc",
    }

    def __init__(self, backend):
        self.srcABank = 0
        self.srcBBank = 0
        super().__init__(backend, MatrixUnit.OPCODE_TO_HANDLER, "Matrix")

    def getSrcA(self):
        return self.backend.getSrcA(self.srcABank)

    def getSrcB(self):
        return self.backend.getSrcB(self.srcBBank)

    def handle_incrwc(self, instruction_info, issue_thread, instr_args):
        srcAInc = instr_args["rwc_a"]
        srcBInc = instr_args["rwc_b"]
        dstInc = instr_args["rwc_d"]
        rwc_CR = instr_args["rwc_cr"]
        srcACr = get_nth_bit(rwc_CR, 0)
        srcBCr = get_nth_bit(rwc_CR, 1)
        dstCr = get_nth_bit(rwc_CR, 2)

        rwc = self.backend.getRCW(issue_thread)

        if srcACr:
            rwc.SrcA_Cr += srcAInc
            rwc.SrcA = rwc.SrcA_Cr
        else:
            rwc.SrcA += srcAInc

        if srcBCr:
            rwc.SrcB_Cr += srcBInc
            rwc.SrcB = rwc.SrcB_Cr
        else:
            rwc.SrcB += srcBInc

        if dstCr:
            rwc.Dst_Cr += dstInc
            rwc.Dst = rwc.Dst_Cr
        else:
            rwc.Dst += dstInc

    def handle_zerosrc(self, instruction_info, issue_thread, instr_args):
        clearSrcABank = [False] * 2
        clearSrcBBank = [False] * 2

        clearSrcA = get_nth_bit(instr_args["src_mask"], 0)
        clearSrcB = get_nth_bit(instr_args["src_mask"], 0)
        bothBanks = instr_args["bank_mask"]
        singleBankMatrixUnit = instr_args["write_mode"]
        negativeInfSrcA = instr_args["zero_val"]

        if clearSrcA:
            if bothBanks:
                clearSrcABank[0] = True
                clearSrcABank[1] = True
            elif singleBankMatrixUnit:
                clearSrcABank[self.srcABank] = True
            else:
                pass
                # clearSrcABank[Unpackers[0].SrcBank] = True

        if clearSrcB:
            if bothBanks:
                clearSrcBBank[0] = True
                clearSrcBBank[1] = True
            elif singleBankMatrixUnit:
                clearSrcBBank[self.srcBBank] = True
            else:
                pass
                # clearSrcBBank[Unpackers[0].SrcBank] = True

        # Do the clearing
        for bank in range(2):
            for i in range(64):
                for j in range(16):
                    if clearSrcABank[bank]:
                        self.backend.getSrcA(bank)[i, j] = ~0 if negativeInfSrcA else 0
                    if clearSrcBBank[bank]:
                        self.backend.getSrcA(bank)[i, j] = 0

    def handle_elwadd(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rwc = self.getRCW(issue_thread)
        broadcastSrcBCol0 = get_nth_bit(instr_args["instr_mod19"], 0)
        broadcastSrcBRow = get_nth_bit(instr_args["instr_mod19"], 1)
        dstRow = instr_args["dst"]
        addDst = instr_args["dest_accum_en"]

        flipsrca = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_dvalid"], 1)

        # Determine data formats
        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            srcAStyle = DataFormat.FP16
            useDst32b = False
        elif self.getConfigValue(stateID, "ALU_ACC_CTRL_INT8_math_enabled"):
            srcAStyle = DataFormat.INT8
            useDst32b = True
        else:
            if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
            else:
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")
            if srcAFmt in [
                DataFormat.FP32,
                DataFormat.BF16,
                DataFormat.BFP8,
                DataFormat.BFP4,
                DataFormat.BFP2,
                DataFormat.INT32,
                DataFormat.UINT16,
            ]:
                srcAStyle = DataFormat.BF16
            elif srcAFmt in [
                DataFormat.FP16,
                DataFormat.BFP8_b,
                DataFormat.BFP4_b,
                DataFormat.BFP2_b,
                DataFormat.INT8,
            ]:
                srcAStyle = DataFormat.FP16
            else:
                # SrcAFmt == TF32
                srcAStyle = DataFormat.TF32

            useDst32b = self.getConfigValue(stateID, "ALU_ACC_CTRL_Fp32_enabled")

        # Determine the row range
        srcARow = rwc.SrcA & 0x38
        srcBRow = rwc.SrcB & (0x3F if broadcastSrcBRow else 0x38)
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        dstRow &= 0x3F8

        # Determine the fidelity phase.
        fidelityPhase = rwc.FidelityPhase
        fidelityPhase += self.getThreadConfigValue(issue_thread, "FIDELITY_BASE_Phase")
        fidelityPhase &= 3

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"Perform FPU compute, dst starts at {dstRow}, srcA starts at {srcARow} and srcB at {srcBRow}"
            )

        # Perform the element-wise computation
        for i in range(8):
            for j in range(16):
                srcAVal = self.backend.getSrcA(self.srcABank)[srcARow + i, j]
                srcBVal = self.backend.getSrcB(self.srcBBank)[
                    srcBRow + (0 if broadcastSrcBRow else i),
                    0 if broadcastSrcBCol0 else j,
                ]
                result = srcAVal + srcBVal
                if 1 == 1 or srcAFmt == DataFormat.INT8:
                    if addDst:
                        result += self.backend.getDst().getDst32b(dstRow + i, j)

                    self.backend.getDst().setDst32b(int(dstRow / 2) + i, j, result)
                else:
                    # These divisions are rarely desirable, so software
                    # is encouraged to ensure that FidelityPhase == 0
                    if fidelityPhase & 1:
                        result /= 32.0
                    elif fidelityPhase & 2:
                        result /= 128.0

                    if useDst32b:
                        # Dst is FP32, regardless of SrcAStyle
                        if addDst:
                            result += self.backend.getDst().getDst32b(dstRow + i, j)
                        self.backend.getDst().setDst32b(dstRow + i, j, result)
                    elif srcAStyle == DataFormat.FP16:
                        # Dst is FP16, just like SrcAStyle
                        if addDst:
                            result += self.backend.getDst().getDst16b(dstRow + i, j)
                        self.backend.getDst().setDst16b(dstRow + i, j, result)
                    else:
                        # Dst is BF16 (SrcAStyle is either BF16 or TF32)
                        if addDst:
                            result += self.backend.getDst().getDst16b(dstRow + i, j)
                        self.backend.getDst().setDst16b(dstRow + i, j, result)

        # Possibly flip source banks
        if flipsrca:
            if not self.getThreadConfigValue(issue_thread, "CLR_DVALID_SrcA_Disable"):
                self.backend.getSrcA(
                    self.srcABank
                ).allowedClient = SrcRegister.SrcClient.Unpackers
            self.srcABank ^= 1

        if flipsrcb:
            if not self.getThreadConfigValue(issue_thread, "CLR_DVALID_SrcB_Disable"):
                self.backend.getSrcB(
                    self.srcBBank
                ).allowedClient = SrcRegister.SrcClient.Unpackers
            self.srcBBank ^= 1

        # Advance the RWCs
        addr_mod = extract_bits(instr_args["addr_mode"], 2, 0)
        rwc.applyAddrMod(issue_thread, addr_mod)

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
                self.backend.getSrcA(
                    self.srcABank
                ).allowedClient = SrcRegister.SrcClient.Unpackers
            self.srcABank ^= 1

        if flipsrcb:
            if not self.getThreadConfigValue(issue_thread, "CLR_DVALID_SrcB_Disable"):
                self.backend.getSrcB(
                    self.srcBBank
                ).allowedClient = SrcRegister.SrcClient.Unpackers
            self.srcBBank ^= 1

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
