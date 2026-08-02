from tt_sim.pe.tensix.backends.backend_base import DataFormat, TensixBackendUnit
from tt_sim.pe.tensix.backends.vector import VectorUnit
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.util.bits import extract_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_float, conv_to_uint32


class MatrixUnit(TensixBackendUnit):
    """
    Performs operations on srcA and srcB, writing results to dst register. Most obvious
    is matrix multiplication, but other element wise operations supported too. This implementation
    performs all real number calculations in FP32, converting from the data format in srcA and srcB
    (e.g. BF16, TF32, FP16) into FP32, and then the result is converted back to the data format
    and then written to dst. This will give slightly different results than on the Tenstorrent
    hardware.

    Based on description and code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/MatrixUnit.md
    """

    OPCODE_TO_HANDLER = {
        "ZEROACC": "handle_zeroacc",
        "SETRWC": "handle_setrwc",
        "ELWADD": "handle_elwadd",
        "ELWSUB": "handle_elwsub",
        "ELWMUL": "handle_elwmul",
        "ZEROSRC": "handle_zerosrc",
        "INCRWC": "handle_incrwc",
        "CLEARDVALID": "handle_cleardvalid",
        "GATESRCRST": "handle_gatesrcrst",
        "TRNSPSRCB": "handle_trnspsrcb",
        "MOVB2A": "handle_movb2a",
        "MOVA2D": "handle_mova2d",
        "MOVD2B": "handle_movd2b",
        "MVMUL": "handle_mvmul",
        "DOTPV": "handle_dotpv",
        "GAPOOL": "handle_gapool",
    }

    def __init__(self, backend):
        self.srcABank = 0
        self.srcBBank = 0
        super().__init__(backend, MatrixUnit.OPCODE_TO_HANDLER, "Matrix")

    def getSrcA(self):
        return self.backend.getSrcA(self.srcABank)

    def getSrcB(self):
        return self.backend.getSrcB(self.srcBBank)

    def _read_addr_mode(self, instruction_info, instr_args):
        # The math ``addr_mode`` field is 3 bits on Blackhole (raw bits 16:14)
        # but only 2 bits on Wormhole (16:15) — Blackhole has 8 ADDR_MOD
        # sections, Wormhole 4. The shared instruction table encodes the
        # Wormhole start_bit (15), so on Blackhole its low bit (14) is dropped
        # and the value is shifted: ADDR_MOD_1/2/4/5 silently decode as
        # 0/1/2/2, losing the matmul MOP's dest-reset mods (ADDR_MOD_5) so the
        # second K-face never accumulates. Read the true field from the raw
        # word on Blackhole. See docs/plans/blackhole-support.md (`six`).
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 3, 14)
        return instr_args["addr_mode"]

    def _read_pool_addr_mode(self, instruction_info, instr_args):
        # GAPOOL and GMPOOL keep their addressing-mode field at bit 15 on
        # Blackhole (where it is renamed ``pool_addr_mode``), but widen it to 3
        # bits: raw 17:15 versus Wormhole's 16:15. That is a different position
        # to every other math instruction (which move to 16:14, see
        # ``_read_addr_mode``), so these two need their own reader — using the
        # generic one would pick up bit 14, which here is ``max_pool_index_en``.
        # See docs/plans/blackhole-support.md (Phase 8).
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 3, 15)
        return instr_args["addr_mode"]

    def handle_cleardvalid(self, instruction_info, issue_thread, instr_args):
        reset = instr_args["reset"] & 0x1
        keepReadingSameSrc = (instr_args["reset"] >> 1) & 0x1
        flipSrcA = instr_args["cleardvalid"] & 0x1
        flipSrcB = (instr_args["cleardvalid"] >> 1) & 0x1

        if reset:
            self.srcABank = 0
            self.srcBBank = 0
            self.backend.unpacker_units[0].srcBank = 0
            self.backend.unpacker_units[1].srcBank = 0
            self.backend.getSrcA(0).allowedClient = SrcRegister.SrcClient.Unpackers
            self.backend.getSrcA(1).allowedClient = SrcRegister.SrcClient.Unpackers
            self.backend.getSrcB(0).allowedClient = SrcRegister.SrcClient.Unpackers
            self.backend.getSrcB(1).allowedClient = SrcRegister.SrcClient.Unpackers
        else:
            if flipSrcA:
                self.backend.getSrcA(
                    self.srcABank
                ).allowedClient = SrcRegister.SrcClient.Unpackers
                if not keepReadingSameSrc:
                    self.srcABank ^= 1
            if flipSrcB:
                self.backend.getSrcB(
                    self.srcBBank
                ).allowedClient = SrcRegister.SrcClient.Unpackers
                if not keepReadingSameSrc:
                    self.srcBBank ^= 1

    def handle_gatesrcrst(self, instruction_info, issue_thread, instr_args):
        # GATESRCRST forcibly invalidates the one-slot operand cache sitting
        # between SrcB and the Matrix Unit and resets the SrcA/SrcB pipeline
        # gating to "don't gate". This functional model reads SrcA/SrcB
        # directly on every op and models neither the operand cache nor the
        # gating mechanism, so there is no state to reset here; it is a no-op.
        pass

    def handle_trnspsrcb(self, instruction_info, issue_thread, instr_args):
        # Transpose the 16x16 matrix held in SrcB rows [16, 32) in place. The
        # ISA wait-gate (FPU must own the SrcB bank before dispatch) is modelled
        # by the one-instruction-per-cycle issue behaviour, so only the swap is
        # performed here.
        srcB = self.getSrcB()
        rowBase = 16
        for i in range(16):
            for j in range(i):
                ij = srcB[rowBase + i, j]
                ji = srcB[rowBase + j, i]
                srcB[rowBase + i, j] = ji
                srcB[rowBase + j, i] = ij

    def handle_movb2a(self, instruction_info, issue_thread, instr_args):
        srcBRow = instr_args["srcb"]
        move4Rows = (instr_args["instr_mod"] >> 1) & 0x1
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcARow = instr_args["srca"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        flushDenormals = not self.getConfigValue(
            stateID, "ALU_ACC_CTRL_Zero_Flag_disabled_src"
        )

        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        srcARow += rwc.SrcA
        srcBRow += rwc.SrcB
        if move4Rows:
            numRows = 4
            srcARow &= 0x3C
            srcBRow &= 0x3C
        else:
            numRows = 1
            srcARow &= 0x3F
            srcBRow &= 0x3F

        # Now copy the row(s) from SrcB into SrcA
        srcA = self.getSrcA()
        srcB = self.getSrcB()
        for i in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue
                srcVal = srcB[srcBRow, j]
                if flushDenormals and not (srcVal & 0xFF):
                    srcVal = 0
                srcA[srcARow, j] = srcVal
            srcARow += 1
            srcBRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_mova2d(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        move8Rows = (instr_args["instr_mod"] >> 1) & 0x1
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcRow = instr_args["src"]
        useDst32bLo = instr_args["dest_32b_lo"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
            srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
        else:
            srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")

        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            use8bExponent = False
        elif srcAFmt in [
            DataFormat.FP32,
            DataFormat.TF32,
            DataFormat.BF16,
            DataFormat.BFP8,
            DataFormat.BFP4,
            DataFormat.BFP2,
            DataFormat.INT32,
            DataFormat.UINT16,
        ]:
            use8bExponent = True
        else:
            use8bExponent = False

        flushDenormals = not self.getConfigValue(
            stateID, "ALU_ACC_CTRL_Zero_Flag_disabled_src"
        )
        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        srcRow += rwc.SrcA

        if move8Rows:
            numRows = 8
            dstRow &= 0x3F8
            srcRow &= 0x38
        else:
            numRows = 1
            dstRow &= 0x3FF
            srcRow &= 0x3F

        # Now copy the row(s)
        for i in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue
                srcAVal = self.backend.getSrcA(self.srcABank)[srcRow, j]
                if flushDenormals and not (srcAVal & 0xFF):
                    srcAVal = 0
                val16b = (
                    DataFormatConversions.removeLowMantissa(srcAVal)
                    if use8bExponent
                    else DataFormatConversions.removeHighExponent(srcAVal)
                )
                if srcAFmt == DataFormat.TF32:
                    lowMantissa = ((srcAVal >> 8) & 7) << 13
                    if useDst32bLo:
                        lowMantissa |= val16b
                    # dst holds TF32 as sign,himan(7b),exp(8b),loman(3b),zeros(13b)
                    self.getDst().setDst32b(dstRow, j, (val16b << 16) | lowMantissa)
                elif useDst32bLo:
                    self.getDst().setDst32b(
                        dstRow,
                        j,
                        (self.getDst().getDst32b(dstRow, j) & 0xFFFF0000) | val16b,
                    )
                else:
                    self.getDst().setDst16b(dstRow, j, val16b)
            dstRow += 1
            srcRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_movd2b(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        move4Rows = (instr_args["instr_mod"] >> 1) & 0x1
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        srcRow = instr_args["src"]
        useDst32bLo = instr_args["dest_32b_lo"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        # Determine the data formats. Note that (as with MOVA2D) SrcAFmt is
        # deliberately used to select the SrcB style; this is not a typo.
        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            useDst32b = False
            srcBStyle = DataFormat.FP16
        else:
            if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
            else:
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")

            useDst32b = self.getConfigValue(
                stateID, "ALU_ACC_CTRL_Fp32_enabled"
            ) or self.getConfigValue(stateID, "ALU_ACC_CTRL_INT8_math_enabled")

            if srcAFmt in [
                DataFormat.FP32,
                DataFormat.BF16,
                DataFormat.BFP8,
                DataFormat.BFP4,
                DataFormat.BFP2,
                DataFormat.INT32,
                DataFormat.UINT16,
            ]:
                srcBStyle = DataFormat.BF16
            elif srcAFmt in [
                DataFormat.FP16,
                DataFormat.BFP8_b,
                DataFormat.BFP4_b,
                DataFormat.BFP2_b,
                DataFormat.INT8,
            ]:
                srcBStyle = DataFormat.FP16
            else:
                # SrcAFmt == TF32
                srcBStyle = DataFormat.TF32

        rwc = self.backend.getRWC(issue_thread)

        # Determine the row range
        dstRow += self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        srcRow += rwc.SrcB

        if move4Rows:
            numRows = 4
            dstRow &= 0x3FC
            srcRow &= 0x3C
        else:
            numRows = 1
            dstRow &= 0x3FF
            srcRow &= 0x3F

        # Now copy the row(s) from Dst into SrcB
        srcB = self.getSrcB()
        for i in range(numRows):
            for j in range(16):
                if get_nth_bit(
                    self.backend.vector_unit.laneConfigValue(
                        int(j / 2), VectorUnit.BLOCK_DEST_MOV
                    ),
                    j & 1,
                ):
                    continue

                if useDst32b:
                    # Read from Dst in 32-bit mode
                    dstVal = self.getDst().getDst32b(dstRow, j)
                    if useDst32bLo:
                        # Only useful if software has deliberately packed two
                        # bf16/fp16 values into 32 bits and written them to Dst32b
                        dstVal = (dstVal << 16) | (dstVal & 0xFFFF)

                    if srcBStyle == DataFormat.BF16:
                        # Treat dstVal as fp32 or tf32, truncate to bf16
                        srcBVal = DataFormatConversions.ShuffleBF16(dstVal >> 16)
                    elif srcBStyle == DataFormat.FP16:
                        srcBVal = DataFormatConversions.ShuffleFP16(dstVal >> 16)
                    elif not useDst32bLo:
                        # Treat dstVal as fp32 or tf32, truncate to tf32
                        srcBVal = DataFormatConversions.ShuffleTF32(dstVal >> 13)
                    else:
                        # The 13 bits discarded by the fp32 -> tf32 conversion
                        srcBVal = dstVal & 0x1FFF
                else:
                    # Read from Dst in 16-bit mode
                    dstVal = self.getDst().getDst16b(dstRow, j)
                    if srcBStyle == DataFormat.BF16:
                        # Treat dstVal as bf16
                        srcBVal = DataFormatConversions.ShuffleBF16(dstVal)
                    elif srcBStyle == DataFormat.FP16:
                        # Treat dstVal as fp16 (int8 is overlaid onto fp16 here)
                        srcBVal = DataFormatConversions.ShuffleFP16(dstVal)
                    else:
                        # dstVal isn't wide enough to hold fp32/tf32 data; the ISA
                        # documents this combination as undefined behaviour
                        srcBVal = 0

                srcB[srcRow, j] = srcBVal
            dstRow += 1
            srcRow += 1

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def handle_incrwc(self, instruction_info, issue_thread, instr_args):
        srcAInc = instr_args["rwc_a"]
        srcBInc = instr_args["rwc_b"]
        dstInc = instr_args["rwc_d"]
        rwc_CR = instr_args["rwc_cr"]
        srcACr = get_nth_bit(rwc_CR, 0)
        srcBCr = get_nth_bit(rwc_CR, 1)
        dstCr = get_nth_bit(rwc_CR, 2)

        rwc = self.backend.getRWC(issue_thread)

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"FPU: incrwc AInc={srcAInc} BInc={srcBInc} dstInc={dstInc} by thread {issue_thread}"
            )

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

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"FPU: perform zerosrc for srcA={clearSrcA}, srcB={clearSrcB} by thread {issue_thread}"
            )

        if clearSrcA:
            if bothBanks:
                clearSrcABank[0] = True
                clearSrcABank[1] = True
            elif singleBankMatrixUnit:
                clearSrcABank[self.srcABank] = True
            else:
                clearSrcABank[self.backend.unpacker_units[0].srcBank] = True

        if clearSrcB:
            if bothBanks:
                clearSrcBBank[0] = True
                clearSrcBBank[1] = True
            elif singleBankMatrixUnit:
                clearSrcBBank[self.srcBBank] = True
            else:
                clearSrcABank[self.backend.unpacker_units[1].srcBank] = True

        # Do the clearing
        for bank in range(2):
            for i in range(64):
                for j in range(16):
                    if clearSrcABank[bank]:
                        self.backend.getSrcA(bank)[i, j] = ~0 if negativeInfSrcA else 0
                    if clearSrcBBank[bank]:
                        self.backend.getSrcA(bank)[i, j] = 0

    def handle_gapool(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        addrMod = self._read_pool_addr_mode(instruction_info, instr_args)
        flipSrcA = instr_args["clear_dvalid"] & 0x1
        flipSrcB = instr_args["clear_dvalid"] & 0x2

        self.perform_mvmul(issue_thread, dstRow, addrMod, False, flipSrcA, flipSrcB, 4)

    def handle_dotpv(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        flipSrcA = instr_args["clear_dvalid"] & 0x1
        flipSrcB = instr_args["clear_dvalid"] & 0x2

        self.perform_mvmul(issue_thread, dstRow, addrMod, False, flipSrcA, flipSrcB, 4)

    def handle_mvmul(self, instruction_info, issue_thread, instr_args):
        dstRow = instr_args["dst"]
        addrMod = self._read_addr_mode(instruction_info, instr_args)
        broadcastSrcBRow = instr_args["instr_mod19"]
        flipSrcA = instr_args["clear_dvalid"] & 0x1
        flipSrcB = instr_args["clear_dvalid"] & 0x2

        self.perform_mvmul(
            issue_thread, dstRow, addrMod, broadcastSrcBRow, flipSrcA, flipSrcB, 8
        )

    def perform_mvmul(
        self,
        issue_thread,
        dstRow,
        addrMod,
        broadcastSrcBRow,
        flipSrcA,
        flipSrcB,
        numRows,
    ):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )
        rwc = self.getRWC(issue_thread)

        srcAStyle, useDst32b = self.get_dataformat_and_useDst(issue_thread, stateID)
        srcARow, srcBRow, dstRow = self.get_base_row_ranges(
            issue_thread, stateID, rwc, broadcastSrcBRow
        )
        if broadcastSrcBRow:
            numRows -= 1
        dstRow &= 0x400 - numRows

        fidelityPhase = self.determine_fidelity_phase(issue_thread, rwc)

        srcAMatrix = [[0 for _ in range(16)] for _ in range(16)]
        srcBMatrix = [[0 for _ in range(16)] for _ in range(numRows)]
        multipliedMatrix = [[0 for _ in range(16)] for _ in range(numRows)]

        for i in range(numRows):
            for j in range(16):
                srcBVal = self.backend.getSrcB(self.srcBBank)[
                    srcBRow + (0 if broadcastSrcBRow else i), j
                ]
                if srcAStyle == DataFormat.INT8:
                    srcBMatrix[i][j] = self.DataFormatConversions.Int8InSrcToInt8(
                        srcBVal & (0x40FFF if fidelityPhase & 2 else 0x7F0FF)
                    )
                else:
                    srcBValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcBVal)
                    srcBMatrix[i][j] = self.srcBFidelityBits(srcBValFP32, fidelityPhase)

        for i in range(16):
            for j in range(16):
                srcAVal = self.backend.getSrcA(self.srcABank)[srcARow + i, j]
                if srcAStyle == DataFormat.INT8:
                    srcAMatrix[i][j] = self.DataFormatConversions.Int8InSrcToInt8(
                        srcAVal & (0x41FFF if fidelityPhase & 2 else 0x4E0FF)
                    )
                else:
                    srcAValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcAVal)
                    srcAMatrix[i][j] = self.srcAFidelityBits(srcAValFP32, fidelityPhase)

        for i in range(numRows):
            for j in range(16):
                multipliedMatrix[i][j] = 0
                for k in range(16):
                    multipliedMatrix[i][j] += srcBMatrix[i][k] * srcAMatrix[k][j]

        for i in range(numRows):
            for j in range(16):
                x = multipliedMatrix[i][j]
                if srcAStyle == DataFormat.INT8:
                    x = multipliedMatrix[i][j]
                    x += self.backend.getDst().getDst32b(dstRow + i, j)
                    self.backend.getDst().setDst32b(
                        dstRow + i, j, DataFormatConversions.FP32ToDstFormatFP32(x)
                    )
                else:
                    self.store_elementwise_fp_result(
                        useDst32b, True, srcAStyle, dstRow, i, j, multipliedMatrix[i][j]
                    )
            if broadcastSrcBRow:
                i += 1

        self.optionally_flip_src_banks(issue_thread, flipSrcA, flipSrcB)

        # Advance the RWCs
        rwc.applyAddrMod(issue_thread, addrMod)

    def get_dataformat_and_useDst(self, issue_thread, stateID):
        # Determine data formats
        if self.getThreadConfigValue(issue_thread, "FP16A_FORCE_Enable"):
            srcAStyle = DataFormat.FP16
            useDst32b = False
            return srcAStyle, useDst32b
        elif self.getConfigValue(stateID, "ALU_ACC_CTRL_INT8_math_enabled"):
            srcAStyle = DataFormat.INT8
            useDst32b = True
            return srcAStyle, useDst32b
        else:
            if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_override"):
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcA_val")
            else:
                srcAFmt = self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG0_SrcA")
            srcAFmt = self.implied_srcA_format(issue_thread, srcAFmt)
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
            return srcAStyle, useDst32b

    def implied_srcA_format(self, issue_thread, configuredFmt):
        """The operand format Blackhole's matrix unit actually uses for SrcA.

        Wormhole always takes it from ALU_FORMAT_SPEC_REG0_SrcA. Blackhole
        instead *implies* it from the format the unpacker last wrote the bank
        in, latched when the bank was handed over; the configured register is
        only consulted when DISABLE_IMPLIED_SRCA_FMT_Base is set.

        This matters for the FP32 path. tt-metal leaves ALU_FORMAT_SPEC_REG0_SrcA
        at FP32 while the unpacker converts to TF32 on the way into Src, and
        SrcA holds TF32 in a 19-bit form. Reading the configured FP32 selects
        the BF16 operand style, which narrows the datum to 7 mantissa bits and
        loses ~2 bits of every copied value; the implied TF32 keeps all 10.
        """
        if not self.backend.blackhole:
            return configuredFmt
        if self.getThreadConfigValue(issue_thread, "DISABLE_IMPLIED_SRCA_FMT_Base"):
            return configuredFmt
        latched = self.backend.getSrcA(self.srcABank).getDataFormat()
        return configuredFmt if latched is None else latched

    def get_base_row_ranges(self, issue_thread, stateID, rwc, broadcastSrcBRow):
        # Determine the row range
        srcARow = rwc.SrcA & 0x38
        srcBRow = rwc.SrcB & (0x3F if broadcastSrcBRow else 0x38)
        dstRow = self.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        dstRow += rwc.Dst + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
        return srcARow, srcBRow, dstRow

    def determine_fidelity_phase(self, issue_thread, rwc):
        # Determine the fidelity phase.
        fidelityPhase = rwc.FidelityPhase
        fidelityPhase += self.getThreadConfigValue(issue_thread, "FIDELITY_BASE_Phase")
        fidelityPhase &= 3
        return fidelityPhase

    def optionally_flip_src_banks(self, issue_thread, flipsrca, flipsrcb):
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

    def handle_elementwise_op(
        self,
        stateID,
        issue_thread,
        rwc,
        broadcastSrcBCol0,
        broadcastSrcBRow,
        dstRow,
        addDst,
        flipsrca,
        flipsrcb,
        addrMode,
        op_handler,
        int8_handler,
        fp_handler,
    ):
        srcAStyle, useDst32b = self.get_dataformat_and_useDst(issue_thread, stateID)

        srcARow, srcBRow, dstRow = self.get_base_row_ranges(
            issue_thread, stateID, rwc, broadcastSrcBRow
        )
        dstRow &= 0x3F8

        fidelityPhase = self.determine_fidelity_phase(issue_thread, rwc)

        if self.getDiagnosticSettings().reportFPUCalculations():
            print(
                f"FPU: perform element wise op, dst starts at {dstRow}, "
                f"srcA starts at {srcARow} and srcB at {srcBRow} by thread {issue_thread}"
            )

        # Perform the element-wise computation
        for i in range(8):
            for j in range(16):
                srcAVal = self.backend.getSrcA(self.srcABank)[srcARow + i, j]
                srcBVal = self.backend.getSrcB(self.srcBBank)[
                    srcBRow + (0 if broadcastSrcBRow else i),
                    0 if broadcastSrcBCol0 else j,
                ]

                if srcAStyle == DataFormat.INT8:
                    int8_handler(
                        fidelityPhase,
                        addDst,
                        srcAVal,
                        srcBVal,
                        dstRow,
                        i,
                        j,
                        op_handler,
                    )
                else:
                    fp_handler(
                        fidelityPhase,
                        useDst32b,
                        addDst,
                        srcAStyle,
                        srcAVal,
                        srcBVal,
                        dstRow,
                        i,
                        j,
                        op_handler,
                    )

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
        rwc.applyAddrMod(issue_thread, addrMode)

    def elementwise_addsub_int8(
        self, fidelityPhase, addDst, srcA, srcB, dstRow, i, j, op_handler
    ):
        srcAInt = DataFormatConversions.Int8InSrcToInt8(srcA)
        srcBInt = DataFormatConversions.Int8InSrcToInt8(srcB)
        result = op_handler(srcAInt, srcBInt, None)
        if addDst:
            result += self.backend.getDst().getDst32b(dstRow + i, j)

        self.backend.getDst().setDst32b(
            dstRow + i, j, DataFormatConversions.FP32ToDstFormatFP32(result)
        )

    def elementwise_fp_other(
        self,
        fidelityPhase,
        useDst32b,
        addDst,
        srcAStyle,
        srcA,
        srcB,
        dstRow,
        i,
        j,
        op_handler,
    ):
        srcAValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcA)
        srcBValFP32 = self.get_elementwise_fp_src_type(srcAStyle, srcB)

        result = op_handler(srcAValFP32, srcBValFP32, fidelityPhase)

        self.store_elementwise_fp_result(
            useDst32b, addDst, srcAStyle, dstRow, i, j, result
        )

    def elementwise_mul_int8(
        self,
        fidelityPhase,
        addDst,
        srcA,
        srcB,
        dstRow,
        i,
        j,
        _,
    ):
        srcAValInt = self.DataFormatConversions.Int8InSrcToInt8(
            srcA & (0x41FFF if fidelityPhase & 1 else 0x4E0FF)
        )
        srcBValInt = self.DataFormatConversions.Int8InSrcToInt8(
            srcB & (0x40FFF if fidelityPhase & 2 else 0x7F0FF)
        )

        result = srcAValInt * srcBValInt
        result += self.backend.getDst().getDst32b(dstRow + i, j)

        self.backend.getDst().setDst32b(
            dstRow + i, j, DataFormatConversions.FP32ToDstFormatFP32(result)
        )

    def store_elementwise_fp_result(
        self, useDst32b, addDst, srcAStyle, dstRow, i, j, result
    ):
        if useDst32b:
            # Dst is FP32, regardless of SrcAStyle
            if addDst:
                fp32_bits = DataFormatConversions.FP32InDstToFP32(
                    self.backend.getDst().getDst32b(dstRow + i, j)
                )
                result += conv_to_float(fp32_bits)
            self.backend.getDst().setDst32b(
                dstRow + i,
                j,
                DataFormatConversions.FP32ToDstFormatFP32(conv_to_uint32(result)),
            )
        elif srcAStyle == DataFormat.FP16:
            # Dst is FP16, just like SrcAStyle
            if addDst:
                val = self.backend.getDst().getDst16b(dstRow + i, j)
                fp16_bits = DataFormatConversions.FP16InDstToFP16(val)
                result += conv_to_float(DataFormatConversions.FP16ToFP32(fp16_bits))
            self.backend.getDst().setDst16b(
                dstRow + i,
                j,
                DataFormatConversions.FP32ToDstFormatFP16(conv_to_uint32(result)),
            )
        else:
            # Dst is BF16 (SrcAStyle is either BF16 or TF32)
            if addDst:
                bf16_bits = DataFormatConversions.BF16InDstToBF16(
                    self.backend.getDst().getDst16b(dstRow + i, j)
                )
                result += conv_to_float(bf16_bits << 16)
            self.backend.getDst().setDst16b(
                dstRow + i,
                j,
                DataFormatConversions.FP32ToDstFormatBF16(conv_to_uint32(result)),
            )

    def get_elementwise_fp_src_type(self, srcStyle, src):
        match srcStyle:
            case DataFormat.BF16:
                return conv_to_float(DataFormatConversions.BF16InSrcToFP32(src))
            case DataFormat.FP16:
                return conv_to_float(DataFormatConversions.FP16InSrcToFP32(src))
            case DataFormat.TF32:
                return conv_to_float(DataFormatConversions.TF32InSrcToFP32(src))
            case _:
                raise NotImplementedError()

    def handle_elwmul(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rwc = self.getRWC(issue_thread)
        broadcastSrcBCol0 = get_nth_bit(instr_args["instr_mod19"], 0)
        broadcastSrcBRow = get_nth_bit(instr_args["instr_mod19"], 1)
        dstRow = instr_args["dst"]
        addDst = True  # Always add for ELWMUL
        addrMode = self._read_addr_mode(instruction_info, instr_args)

        flipsrca = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_dvalid"], 1)

        def mul_handler(srcAValFP32, srcBValFP32, fidelityPhase):
            return self.srcAFidelityBits(
                srcAValFP32, fidelityPhase
            ) * self.srcBFidelityBits(srcBValFP32, fidelityPhase)

        self.handle_elementwise_op(
            stateID,
            issue_thread,
            rwc,
            broadcastSrcBCol0,
            broadcastSrcBRow,
            dstRow,
            addDst,
            flipsrca,
            flipsrcb,
            addrMode,
            mul_handler,
            self.elementwise_mul_int8,
            self.elementwise_fp_other,
        )

    def handle_elwadd(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rwc = self.getRWC(issue_thread)
        broadcastSrcBCol0 = get_nth_bit(instr_args["instr_mod19"], 0)
        broadcastSrcBRow = get_nth_bit(instr_args["instr_mod19"], 1)
        dstRow = instr_args["dst"]
        addDst = instr_args["dest_accum_en"]
        addrMode = self._read_addr_mode(instruction_info, instr_args)

        flipsrca = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_dvalid"], 1)

        def add_handler(srcAVal, srcBVal, fidelityPhase=None):
            result = srcAVal + srcBVal

            if fidelityPhase is not None:
                # These divisions are rarely desirable, so software
                # is encouraged to ensure that FidelityPhase == 0
                if fidelityPhase & 1:
                    result /= 32.0
                elif fidelityPhase & 2:
                    result /= 128.0

            return result

        self.handle_elementwise_op(
            stateID,
            issue_thread,
            rwc,
            broadcastSrcBCol0,
            broadcastSrcBRow,
            dstRow,
            addDst,
            flipsrca,
            flipsrcb,
            addrMode,
            add_handler,
            self.elementwise_addsub_int8,
            self.elementwise_fp_other,
        )

    def handle_elwsub(self, instruction_info, issue_thread, instr_args):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rwc = self.getRWC(issue_thread)
        broadcastSrcBCol0 = get_nth_bit(instr_args["instr_mod19"], 0)
        broadcastSrcBRow = get_nth_bit(instr_args["instr_mod19"], 1)
        dstRow = instr_args["dst"]
        addDst = instr_args["dest_accum_en"]
        addrMode = self._read_addr_mode(instruction_info, instr_args)

        flipsrca = get_nth_bit(instr_args["clear_dvalid"], 0)
        flipsrcb = get_nth_bit(instr_args["clear_dvalid"], 1)

        def sub_handler(srcAVal, srcBVal, fidelityPhase=None):
            result = srcAVal - srcBVal

            if fidelityPhase is not None:
                # These divisions are rarely desirable, so software
                # is encouraged to ensure that FidelityPhase == 0
                if fidelityPhase & 1:
                    result /= 32.0
                elif fidelityPhase & 2:
                    result /= 128.0

            return result

        self.handle_elementwise_op(
            stateID,
            issue_thread,
            rwc,
            broadcastSrcBCol0,
            broadcastSrcBRow,
            dstRow,
            addDst,
            flipsrca,
            flipsrcb,
            addrMode,
            sub_handler,
            self.elementwise_addsub_int8,
            self.elementwise_fp_other,
        )

    def handle_setrwc(self, instruction_info, issue_thread, instr_args):
        rwc = self.getRWC(issue_thread)

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
                SrcAVal += rwc.SrcA_Cr
            rwc.SrcA = SrcAVal
            rwc.SrcA_Cr = SrcAVal

        if srcb:
            if srcb_cr:
                SrcBVal += rwc.SrcB_Cr
            rwc.SrcB = SrcBVal
            rwc.SrcB_Cr = SrcBVal

        if dst or dst_c_to_cr:
            if dst_c_to_cr:
                DstVal += rwc.Dst
            elif dst_cr:
                DstVal += rwc.Dst_Cr
            rwc.Dst = DstVal
            rwc.Dst_Cr = DstVal

        if fidelity:
            rwc.FidelityPhase = 0

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

        # Blackhole ZEROACC carries a `clear_zero_flags` bit (raw bit 17, absent
        # from the shared Wormhole-layout argument table). When set, the
        # instruction only re-asserts DEST zero-flags and must NOT clear the
        # data: it is the workaround `copy_tile` issues immediately after
        # unpack-to-dest (a Blackhole HW bug drops the zero-flag clear when
        # unpack-to-dest and a packer ZEROACC land in the same cycle), so the
        # rows it names hold the freshly-unpacked operands. tt-sim emulates
        # zero-flags by zeroing the data, so here it must skip the clear or it
        # wipes those operands (the SFPU then reads 0). The RWC still advances.
        # See ttsim TT_ARCH_VERSION==1 ZEROACC (clear_zero_flags -> valid=true).
        if self.backend.blackhole and get_nth_bit(
            instruction_info["raw_instruction"], 17
        ):
            if mode == ZEROACC_MODE_ONE_ROW or mode == ZEROACC_MODE_16_ROWS:
                addr_mod = extract_bits(instr_args["AddrMode"], 2, 0)
                self.getRWC(issue_thread).applyAddrMod(issue_thread, addr_mod)
            return

        if mode == ZEROACC_MODE_ONE_ROW:
            state_id = self.getThreadConfigValue(issue_thread, "CFG_STATE_ID_StateID")
            row = imm10
            row += self.getThreadConfigValue(
                issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
            )
            row += self.getRWC(issue_thread).Dst + self.getConfigValue(
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
            self.getRWC(issue_thread).applyAddrMod(issue_thread, addr_mod)

    def srcAFidelityBits(self, x, fidelityPhase):
        x = conv_to_uint32(x)
        if fidelityPhase & 1 == 0:
            # Sign, Exp, implicit 1 of Man, next four Man bits
            return conv_to_float(x & 0xFFF80000)
        else:
            # Isolate the next five Man bits not consumed by prior branch
            return conv_to_float(x - (x & 0xFFF83FFF))

    def srcBFidelityBits(self, x, fidelityPhase):
        x = conv_to_uint32(x)
        if fidelityPhase & 2 == 0:
            # Sign, Exp, implicit 1 of Man, next six Man bits
            return conv_to_float(x & 0xFFFE0000)
        else:
            # Isolate the next four Man bits not consumed by prior branch
            return conv_to_float(x - (x & 0xFFFE1FFF))
