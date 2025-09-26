from math import ceil, floor

from tt_sim.network.tt_noc import NoCOverlay
from tt_sim.pe.tensix.backends.backend_base import (
    DATA_FORMAT_TO_BITS,
    DATA_FORMAT_TO_NAME,
    DataFormat,
    TensixBackendUnit,
)
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.util.bits import get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class UnPackerUnit(TensixBackendUnit):
    """
    Unpacker unit, which unpacks from L1 into either srcA/srcB or dst.

    Based on the description and functional code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/UNPACR.md
    """

    OPCODE_TO_HANDLER = {"UNPACR": "handle_unpacr", "UNPACR_NOP": "handle_unpacr_nop"}

    def __init__(self, backend, unpacker_id):
        self.unpacker_id = unpacker_id
        self.context_counter = [0] * 3
        self.srcBank = 0
        self.srcRow = [0] * 3
        self.blocked = False
        self.repeat_instruction = None
        self.setRegBase = 0
        self.setRegAcc = 0
        super().__init__(backend, UnPackerUnit.OPCODE_TO_HANDLER, "Unpacker")

    def issueInstruction(self, instruction, from_thread):
        if self.blocked:
            return False
        else:
            return super().issueInstruction(instruction, from_thread)

    def clock_tick(self, cycle_num):
        if self.blocked:
            assert self.repeat_instruction is not None
            instruction_info, issue_thread = self.repeat_instruction
            assert instruction_info["name"] in UnPackerUnit.OPCODE_TO_HANDLER
            getattr(self, UnPackerUnit.OPCODE_TO_HANDLER[instruction_info["name"]])(
                instruction_info, issue_thread, instruction_info["instr_args"]
            )
        else:
            super().clock_tick(cycle_num)

    def handle_unpacr_nop(self, instruction_info, issue_thread, instr_args):
        args = instr_args["NoOp"]

        mode1 = args & 0x3
        mode2 = args & 0x7

        if mode1 == 0x1:
            # Set srcA or srcB to zero
            self.handle_set_src_to_zero(instruction_info, issue_thread, args)
        else:
            match mode2:
                case 0x2:
                    # Occupy Unpacker for one cycle
                    pass
                case 0x3 | 0x0:
                    # MMIO register write to Overlay STREAM_MSG_DATA_CLEAR_REG_INDEX
                    self.handle_write_stream_data_clear_reg_index(issue_thread, args)
                case 0x4:
                    # MMIO register write
                    self.handle_mmio_register_write(args)
                case 0x7:
                    # Give srcA or srcB banks to matrix unit
                    self.handle_give_src_to_fpu(issue_thread, args)
                case _:
                    raise NotImplementedError()

    def handle_give_src_to_fpu(self, issue_thread, args):
        if self.unpacker_id == 0:
            self.backend.getSrcA(self.srcBank).setAllowedClient(
                SrcRegister.SrcClient.MatrixUnit
            )
            self.srcBank ^= 1
            self.srcRow[issue_thread] = (
                self.backend.getThreadConfigValue(issue_thread, "SRCA_SET_Base") << 4
            )
        else:
            self.backend.getSrcB(self.srcBank).setAllowedClient(
                SrcRegister.SrcClient.MatrixUnit
            )
            self.srcBank ^= 1
            self.srcRow[issue_thread] = (
                self.backend.getThreadConfigValue(issue_thread, "SRCB_SET_Base") << 4
            )

    def handle_mmio_register_write(self, args):
        accumulate = get_nth_bit(args, 3)
        value11 = (args >> 4) & 0x3FF
        addrMid = (args >> 16) & 0x3F
        addrSel = get_nth_bit(args, 22)

        addr = 0xFFB00000 + self.setRegBase[addrSel] + (addrMid << 12)
        if accumulate:
            accValue = self.setRegAcc
            if value11 == 0:
                accValue = 0
            else:
                accValue = (accValue + value11) & 0x1FFFF
                self.backend.getAddressableMemory().write(addr, conv_to_bytes(accValue))
            self.setRegAcc = accValue
        else:
            self.backend.getAddressableMemory().write(addr, conv_to_bytes(value11))

    def handle_write_stream_data_clear_reg_index(self, issue_thread, args):
        clearCount = (args >> 4) & 0x3FF
        whichStream = (args >> 16) & 0x1F

        if clearCount != 0:
            streamId = whichStream
        else:
            streamId = self.backend.getThreadConfigValue(
                issue_thread, "NOC_OVERLAY_MSG_CLEAR_StreamId_" + str(self.unpacker_id)
            )

        overlay_addr = NoCOverlay.NOC_STREAM_REG_SPACE_SIZE * streamId + (
            NoCOverlay.STREAM_MSG_DATA_CLEAR_REG_INDEX << 2
        )
        self.backend.getAddressableMemory().write(
            0xFFB40000 + overlay_addr, conv_to_bytes(1)
        )

    def handle_set_src_to_zero(self, instruction_info, issue_thread, args):
        negativeInfSrcA = get_nth_bit(args, 2)
        bothBanks = get_nth_bit(args, 3)
        waitLikeUnpacr = get_nth_bit(args, 4)

        unpackBank = self.srcBank

        if self.unpacker_id == 0:
            if waitLikeUnpacr:
                srcBank = self.srcBank
            else:
                srcBank = self.backend.matrix_unit.srcABank
            if (
                self.backend.getSrcA(srcBank).getAllowedClient()
                != SrcRegister.SrcClient.Unpackers
            ):
                self.blocked = True
                self.repeat_instruction = (instruction_info, issue_thread)
                return
        else:
            if waitLikeUnpacr:
                srcBank = self.srcBank
            else:
                srcBank = self.backend.matrix_unit.srcBBank
            if (
                self.backend.getSrcB(srcBank).getAllowedClient()
                != SrcRegister.SrcClient.Unpackers
            ):
                self.blocked = True
                self.repeat_instruction = (instruction_info, issue_thread)
                return

        self.blocked = False
        self.repeat_instruction = None

        for bank in range(2):
            if bothBanks or bank == unpackBank:
                if self.unpacker_id == 0:
                    clearVal = ~0 if negativeInfSrcA else 0
                    for i in range(64):
                        for j in range(16):
                            self.backend.getSrcA(bank)[i, j] = clearVal
                else:
                    for i in range(64):
                        for j in range(16):
                            self.backend.getSrcB(bank)[i, j] = 0

    def handle_unpacr(self, instruction_info, issue_thread, instr_args):
        one_bit = instr_args["SearchCacheFlush"]
        thirteen_bit = instr_args["CfgContextCntInc"]
        if one_bit:
            self.handle_flush(instruction_info, issue_thread, instr_args)
        else:
            if thirteen_bit:
                self.handle_increment_context_counter(
                    instruction_info, issue_thread, instr_args
                )
            else:
                self.handle_regular(instruction_info, issue_thread, instr_args)

    def handle_flush(self, instruction_info, issue_thread, instr_args):
        pass

    def handle_increment_context_counter(
        self, instruction_info, issue_thread, instr_args
    ):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        increment_ctr = self.context_counter[issue_thread]
        thcon_context_count = self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Context_count"
        )
        if increment_ctr >= (1 << thcon_context_count):
            increment_ctr = 0
        self.context_counter[issue_thread] = increment_ctr

    def wrapAddr(self, stateID, addr):
        if addr is None:
            return None
        if (
            addr
            > self.getConfigValue(
                stateID,
                "THCON_SEC" + str(self.unpacker_id) + "_REG2_Unpack_limit_address",
            )
            * 16
        ):
            addr -= (
                self.getConfigValue(
                    stateID,
                    "THCON_SEC" + str(self.unpacker_id) + "_REG2_Unpack_fifo_size",
                )
                * 16
            )
        return addr

    def read_unpack_configuration(
        self,
        issue_thread,
        multiContextMode,
        useContextCounter,
        contextNumber,
        contextADC,
    ):
        if multiContextMode:
            if useContextCounter:
                whichContext = self.context_counter[issue_thread]
            else:
                whichContext = contextNumber
            whichContext += self.backend.getThreadConfigValue(
                issue_thread,
                "UNPACK_MISC_CFG_CfgContextOffset_" + str(self.unpacker_id),
            )

            whichADC = contextADC
            assert not (self.unpacker_id == 1 and whichContext >= 2)
            assert whichADC != 3
        else:
            whichContext = 0
            whichADC = issue_thread

        return whichContext, whichADC

    def get_isUncompressed(
        self, configDescriptor, stateID, multiContextMode, whichContext
    ):
        return True
        if multiContextMode:
            return self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG2_Disable_zero_compress_cntx"
                + str(whichContext),
            )
        else:
            return get_nth_bit(configDescriptor[0], 4)

    def get_xyzw_dim(self, configDescriptor, stateID, multiContextMode, whichContext):
        if multiContextMode and self.unpacker_id == 0:
            xdim = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG5_Tile_x_dim_cntx"
                + str(whichContext & 3),
            )
        else:
            xdim = get_bits(configDescriptor[0], 16, 31)

        ydim = get_bits(configDescriptor[1], 0, 15)
        zdim = get_bits(configDescriptor[1], 16, 31)
        if not zdim:
            zdim = 1
        wdim = get_bits(configDescriptor[2], 0, 15)
        if not wdim:
            wdim = 1

        return xdim, ydim, zdim, wdim

    def get_inaddr(self, configDescriptor, stateID, multiContextMode, whichContext):
        if multiContextMode and whichContext != 0:
            inAddr = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG3_Base_cntx"
                + str(whichContext)
                + "_address",
            ) + (
                self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG7_Offset_cntx"
                    + str(whichContext & 3)
                    + "_address",
                )
                & 0xFFFF
            )
        else:
            inAddr = self.getConfigValue(
                stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG3_Base_address"
            ) + (
                self.getConfigValue(
                    stateID,
                    "THCON_SEC" + str(self.unpacker_id) + "_REG7_Offset_address",
                )
                & 0xFFFF
            )
        inAddr = (inAddr + 1 + get_bits(configDescriptor[3], 24, 31)) * 16
        return inAddr

    def get_first_datum_and_inputNumDatums(
        self,
        configDescriptor,
        stateID,
        issue_thread,
        whichADC,
        whichContext,
        isUncompressed,
        rowSearch,
        multiContextMode,
        blobsPerXYPlane,
        xdim,
        ydim,
        zdim,
        wdim,
        inAddr_RowStart,
    ):
        adc_xy = self.backend.getADC(whichADC).Unpacker[self.unpacker_id].Channel[0]
        adc_zw = self.backend.getADC(issue_thread).Unpacker[self.unpacker_id].Channel[0]
        if isUncompressed:
            if not rowSearch:
                xpos = adc_xy.X
                ypos = adc_xy.Y
                xend = (
                    self.backend.getADC(whichADC)
                    .Unpacker[self.unpacker_id]
                    .Channel[1]
                    .X
                    + 1
                )
            elif blobsPerXYPlane:
                if multiContextMode and self.unpacker_id == 0:
                    blobsYStart = self.getConfigValue(
                        stateID,
                        "UNP0_BLOBS_Y_START_CNTX_"
                        + str(whichContext & 2)
                        + "_blobs_y_start",
                    )
                else:
                    blobsYStart = get_bits(configDescriptor[2], 16, 32)
                xpos = get_bits(blobsYStart, adc_xy.X & 7, (adc_xy.X & 7) + 4) << 4
                ypos = 0
                x71 = (adc_xy.X & 7) + 1
                if x71 == blobsPerXYPlane:
                    xend = xdim & 0x1F0
                else:
                    xend = get_bits(blobsYStart, x71, x71 + 4) << 4
            else:
                xpos = 0
                ypos = adc_xy.Y
                xend = (
                    self.backend.getADC(whichADC)
                    .Unpacker[self.unpacker_id]
                    .Channel[1]
                    .X
                )
            firstDatum = ((adc_zw.W * zdim + adc_zw.Z) * ydim + ypos) * xdim + xpos
            inputNumDatums = xend - xpos
        else:
            raise NotImplementedError()

        return firstDatum, inputNumDatums

    def generate_input_addresses_and_sizes(
        self, issue_thread, stateID, multiContextMode, whichContext, whichADC, rowSearch
    ):
        configDescriptor = self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG0_TileDescriptor", 4
        )

        isUncompressed = self.get_isUncompressed(
            configDescriptor, stateID, multiContextMode, whichContext
        )
        xdim, ydim, zdim, wdim = self.get_xyzw_dim(
            configDescriptor, stateID, multiContextMode, whichContext
        )

        if multiContextMode and self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Ovrd_data_format"
        ):
            inDataFormat = DataFormat(
                self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG7_Unpack_data_format_cntx"
                    + str(whichContext),
                )
            )
        else:
            inDataFormat = DataFormat(get_bits(configDescriptor[0], 0, 3))

        datumSizeBytes = int(DATA_FORMAT_TO_BITS[inDataFormat] / 8)

        inAddr = self.get_inaddr(
            configDescriptor, stateID, multiContextMode, whichContext
        )

        blobsPerXYPlane = get_bits(configDescriptor[3], 8, 11)
        if not isUncompressed:
            inAddr_RowStart = inAddr
            if blobsPerXYPlane:
                numBlobs = blobsPerXYPlane * zdim * wdim
                inAddr += ceil((numBlobs + 1) * 2 / 16) * 16
            else:
                numRows = ydim * zdim * wdim
                inAddr += ceil((numRows + 1) * 2 / 16) * 16
        else:
            inAddr_RowStart = 0

        inAddr_Exponents = 0
        if inDataFormat.isBFPFormat() and not self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Force_shared_exp"
        ):
            inAddr_Exponents = inAddr
            if inDataFormat == DataFormat.BFP8:
                # missing BFP8a and ConfigDescriptor.NoBFPExpSection
                numElements = xdim * ydim * zdim * wdim
                numExponents = ceil(numElements / 16)
                inAddr += ceil(numExponents / 16) * 16

        firstDatum, inputNumDatums = self.get_first_datum_and_inputNumDatums(
            configDescriptor,
            stateID,
            issue_thread,
            whichADC,
            whichContext,
            isUncompressed,
            rowSearch,
            multiContextMode,
            blobsPerXYPlane,
            xdim,
            ydim,
            zdim,
            wdim,
            inAddr_RowStart,
        )

        inAddr_Datums = inAddr
        inAddr_Exponents += int(firstDatum / 16)
        if isUncompressed:
            inAddr_Datums += firstDatum * datumSizeBytes
            inAddr_Deltas = None
        else:
            inAddr_Datums += int(firstDatum / 32) * int(32 * datumSizeBytes + 32 * 0.5)
            inAddr_Deltas = inAddr_Datums + 32 * datumSizeBytes
            inAddr_Datums += (firstDatum % 32) * datumSizeBytes
            inAddr_Deltas += int((firstDatum % 32) * 0.5)

        inAddr_Exponents = self.wrapAddr(stateID, inAddr_Exponents)
        inAddr_Datums = self.wrapAddr(stateID, inAddr_Datums)
        inAddr_Deltas = self.wrapAddr(stateID, inAddr_Deltas)

        discontiguousInputRows = self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Tileize_mode"
        )
        if discontiguousInputRows:
            # Note that each Shift_amount_cntx is a 4-bit field, so there's 12 bits of
            # precision here, and therefore the maximum RowStride is 65520 bytes.
            rowStride = (
                (
                    self.getConfigValue(
                        stateID,
                        "THCON_SEC"
                        + str(self.unpacker_id)
                        + "_REG2_Shift_amount_cntx0",
                    )
                    << 4
                )
                | (
                    self.getConfigValue(
                        stateID,
                        "THCON_SEC"
                        + str(self.unpacker_id)
                        + "_REG2_Shift_amount_cntx1",
                    )
                    << 4
                )
                | (
                    self.getConfigValue(
                        stateID,
                        "THCON_SEC"
                        + str(self.unpacker_id)
                        + "_REG2_Shift_amount_cntx2",
                    )
                    << 12
                )
            )
        else:
            rowStride = datumSizeBytes * 16

        return (
            inAddr_Datums,
            datumSizeBytes,
            inputNumDatums,
            inAddr_Deltas,
            inAddr_Exponents,
            rowStride,
            discontiguousInputRows,
            isUncompressed,
            inDataFormat,
        )

    def generate_output_address(
        self, stateID, issue_thread, multiContextMode, whichContext
    ):
        if multiContextMode and self.getConfigValue(
            stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Ovrd_data_format"
        ):
            outDataFormat = DataFormat(
                self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG7_Unpack_out_data_format_cntx"
                    + str(whichContext),
                )
            )
        else:
            outDataFormat = DataFormat(
                self.getConfigValue(
                    stateID,
                    "THCON_SEC" + str(self.unpacker_id) + "_REG2_Out_data_format",
                )
            )

        if self.unpacker_id == 0:
            if multiContextMode:
                unpackToDst = self.getConfigValue(
                    stateID,
                    "THCON_SEC"
                    + str(self.unpacker_id)
                    + "_REG2_Unpack_if_sel_cntx"
                    + str(whichContext),
                )
            else:
                unpackToDst = self.getConfigValue(
                    stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Unpack_If_Sel"
                )
            transpose = self.getConfigValue(
                stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Haloize_mode"
            )
        else:
            unpackToDst = False
            transpose = False

        adc_out = (
            self.backend.getADC(issue_thread).Unpacker[self.unpacker_id].Channel[1]
        )
        outAddr = (
            self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_BASE_REG_1_Base"
            )
            + adc_out.Y
            * self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_CTRL_XY_REG_1_Ystride"
            )
            + adc_out.Z
            * self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_CTRL_ZW_REG_1_Zstride"
            )
            + adc_out.W
            * self.getConfigValue(
                stateID, "UNP" + str(self.unpacker_id) + "_ADDR_CTRL_ZW_REG_1_Wstride"
            )
        )

        if (
            outDataFormat == DataFormat.FP32
            or outDataFormat == DataFormat.TF32
            or outDataFormat == DataFormat.INT32
        ):
            assert not outAddr & 3
            outAddr >>= 2
        elif (
            outDataFormat == DataFormat.FP16
            or outDataFormat == DataFormat.BF16
            or outDataFormat == DataFormat.UINT16
        ):
            assert not outAddr & 1
            outAddr >>= 1

        if multiContextMode and self.unpacker_id == 0:
            ctxOutAddr = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG5_Dest_cntx"
                + str(whichContext & 3)
                + "_address",
            )
            if unpackToDst or self.getConfigValue(
                stateID,
                "UNP"
                + str(self.unpacker_id)
                + "_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr",
            ):
                outAddr += ctxOutAddr
            else:
                outAddr = ctxOutAddr

        return outAddr, outDataFormat, unpackToDst, transpose

    def check_unpacker_settings(
        self,
        transpose,
        discontiguousInputRows,
        inAddr_Datums,
        upsampleZeroes,
        isUncompressed,
        unpackToDst,
        colShift,
    ):
        if transpose or discontiguousInputRows:
            # These modes require that InAddr_Datums start at an aligned 16 byte boundary.
            assert inAddr_Datums == floor(inAddr_Datums / 16) * 16

        assert not (
            discontiguousInputRows and (upsampleZeroes > 0 or not isUncompressed)
        )
        assert not (unpackToDst and (colShift or transpose))

    def perform_unpack(
        self,
        stateID,
        issue_thread,
        inputNumDatums,
        inAddr_Datums,
        outAddr,
        datumSizeBytes,
        rowStride,
        upsampleZeroes,
        upsampleInterleave,
        colShift,
        inDataFormat,
        outDataFormat,
        unpackToDst,
        transpose,
        allDatumsAreZero,
    ):
        start_row = int(outAddr / 16)
        if self.unpacker_id == 0:
            assert start_row >= 4
            start_row -= 4

        if self.getDiagnosticSettings().reportUnpacking():
            tgt = (
                "srcB" if self.unpacker_id == 1 else ("dst" if unpackToDst else "srcA")
            )
            print(
                f"Unpacker {self.unpacker_id}: start read at {hex(inAddr_Datums)} for "
                f"{inputNumDatums} datums of bytes size {datumSizeBytes} "
                f"starting write to {tgt} at row {start_row}, read data type "
                f"{DATA_FORMAT_TO_NAME[inDataFormat]} -> write data type {DATA_FORMAT_TO_NAME[outDataFormat]}"
            )

        for row in range(int(inputNumDatums / 16)):
            for col in range(16):
                assert datumSizeBytes <= 4
                raw_datum = conv_to_uint32(
                    self.backend.addressable_memory.read(inAddr_Datums, datumSizeBytes)
                )

                datum = self.formatConversion(
                    stateID, inDataFormat, outDataFormat, raw_datum, unpackToDst
                )

                if allDatumsAreZero:
                    datum = 0
                inAddr_Datums += datumSizeBytes

                if self.unpacker_id == 1:
                    # always srcB
                    row = (row + self.srcRow[issue_thread] + start_row) & 0x3F
                    self.backend.getSrcB(self.srcBank)[row, col] = datum
                else:
                    # Always srcA
                    if not unpackToDst:
                        col -= colShift
                        if self.backend.getThreadConfigValue(
                            issue_thread, "SRCA_SET_SetOvrdWithAddr"
                        ):
                            assert row < 64
                        else:
                            assert row < 16
                            row += self.srcRow[issue_thread] + start_row

                            if transpose:
                                rowLowBits = col
                                col = row & 0xF
                                row = (row & ~0xF) | rowLowBits
                        self.backend.getSrcA(self.srcBank)[row, col] = datum
                    else:
                        if self.backend.getThreadConfigValue(
                            issue_thread, "SRCA_SET_SetOvrdWithAddr"
                        ):
                            row &= 15
                        else:
                            row &= 0x3FF
                        if DATA_FORMAT_TO_BITS[outDataFormat] == 32:
                            self.backend.getDst().setDst32b(row + start_row, col, datum)
                        else:
                            self.backend.getDst().setDst16b(row + start_row, col, datum)
                outAddr += 1

    def increment_counter(
        self, stateID, issue_thread, whichContext, multiContextMode, useContextCounter
    ):
        if multiContextMode and useContextCounter:
            incrementedCounter = whichContext + 1
            if incrementedCounter >= (
                1
                << self.getConfigValue(
                    stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Context_count"
                )
            ):
                incrementedCounter = 0
            self.context_counter[issue_thread] = incrementedCounter

    def update_ADC(self, issue_thread, whichADC, ch0YInc, ch0ZInc, ch1YInc, ch1ZInc):
        for i in range(3):
            if i == issue_thread or i == whichADC:
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    0
                ].Y += ch0YInc
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    0
                ].Z += ch0ZInc
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    1
                ].Y += ch1YInc
                self.backend.getADC(i).Unpacker[self.unpacker_id].Channel[
                    1
                ].Z += ch1ZInc

    def flip_src_banks(self, flipSrc, issue_thread):
        srcRowBase = (
            self.backend.getThreadConfigValue(issue_thread, "SRCB_SET_Base")
            if self.unpacker_id
            else self.backend.getThreadConfigValue(issue_thread, "SRCA_SET_Base")
        ) << 4
        if flipSrc:
            if self.unpacker_id == 0:
                self.backend.getSrcA(self.srcBank).flipAllowedClient()
            else:
                self.backend.getSrcB(self.srcBank).flipAllowedClient()
            self.srcBank ^= 1
            self.srcRow[issue_thread] = srcRowBase
        else:
            self.srcRow[issue_thread] += 16 + srcRowBase

    def formatConversion(
        self, stateID, inDataFormat, outDataFormat, raw_datum, unpackToDst
    ):
        if inDataFormat == DataFormat.FP32:
            match outDataFormat:
                case DataFormat.FP32:
                    pass
                case DataFormat.TF32:
                    if unpackToDst:
                        # when unpacking to Dst TF32 means FP32
                        return DataFormatConversions.FP32ToDstFormatFP32(raw_datum)
                    else:
                        return DataFormatConversions.TF32ToSrcFormatTF32(
                            raw_datum >> 13
                        )
                case DataFormat.BF16:
                    if not raw_datum & 0x7F800000:
                        # Flush denormals to zero
                        raw_datum &= 0x80000000
                    raw_datum >>= 16
                    inDataFormat = DataFormat.BF16
                case DataFormat.FP16:
                    raw_datum = DataFormatConversions.FP32ToFP16(raw_datum)
                    inDataFormat = DataFormat.FP16
                case _:
                    raise NotImplementedError()
        else:
            assert inDataFormat == outDataFormat

            match inDataFormat:
                case DataFormat.INT8:
                    # INT8 is either uint8_t or 8 bit sign-magnitude, and becomes "Integer 8",
                    # which is then overlaid onto FP16
                    int8MeansUnsigned = (
                        self.getConfigValue(
                            stateID, "ALU_FORMAT_SPEC_REG0_SrcBUnsigned"
                        )
                        if self.unpacker_id == 1
                        else self.getConfigValue(
                            stateID, "ALU_FORMAT_SPEC_REG0_SrcAUnsigned"
                        )
                    )
                    sign = 0 if int8MeansUnsigned else raw_datum & 0x80
                    raw_datum -= sign
                    if raw_datum:
                        raw_datum |= 16 << 10
                    raw_datum |= sign << 8
                    inDataFormat = DataFormat.FP16
                case DataFormat.TF32:
                    if unpackToDst:
                        #  When unpacking to Dst, TF32 means FP32
                        return DataFormatConversions.FP32ToDstFormatFP32(raw_datum)
                    else:
                        # Otherwise, TF32 is not valid as InDataFormat, but software can instead
                        # specify InDataFormat == FP32 and OutDataFormat == TF32
                        raise ValueError()

        # Now rearrange bits to the format expected by Dst or by SrcA / SrcB
        match inDataFormat:
            case DataFormat.UINT16:
                if unpackToDst:
                    return (raw_datum & 0xFF00) << 3
                else:
                    return raw_datum & 0xFF
            case DataFormat.INT32 | DataFormat.FP32:
                assert unpackToDst
                return DataFormatConversions.FP32ToDstFormatFP32(raw_datum)
            case DataFormat.BF16:
                if unpackToDst:
                    return DataFormatConversions.BF16ToDstFormatBF16(raw_datum)
                else:
                    return DataFormatConversions.BF16ToSrcBF16(raw_datum)
            case DataFormat.FP16:
                if unpackToDst:
                    return DataFormatConversions.FP16ToDstFormatFP16(raw_datum)
                else:
                    return DataFormatConversions.FP16ToSrcFP16(raw_datum)
            case _:
                raise NotImplementedError()

    def handle_regular(self, instruction_info, issue_thread, instr_args):
        if self.unpacker_id == 0:
            if (
                self.backend.getSrcA(self.srcBank).getAllowedClient()
                != SrcRegister.SrcClient.Unpackers
            ):
                self.blocked = True
                self.repeat_instruction = (instruction_info, issue_thread)
                return
        elif self.unpacker_id == 1:
            if (
                self.backend.getSrcB(self.srcBank).getAllowedClient()
                != SrcRegister.SrcClient.Unpackers
            ):
                self.blocked = True
                self.repeat_instruction = (instruction_info, issue_thread)
                return

        self.blocked = False
        self.repeat_instruction = None

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        rowSearch = instr_args["RowSearch"]
        useContextCounter = instr_args["AutoIncContextID"]
        allDatumsAreZero = instr_args["ZeroWrite2"]
        flipSrc = instr_args["SetDatValid"]
        multiContextMode = instr_args["OvrdThreadId"]
        contextADC = instr_args["AddrCntContextId"]
        contextNumber = instr_args["CfgContextId"]

        addrMode = instr_args["AddrMode"]
        ch0ZInc = get_bits(addrMode, 0, 1)
        ch0YInc = get_bits(addrMode, 2, 3)
        ch1ZInc = get_bits(addrMode, 4, 5)
        ch1YInc = get_bits(addrMode, 6, 7)

        # Determine initial input address(es) and input datum count
        whichContext, whichADC = self.read_unpack_configuration(
            issue_thread, multiContextMode, useContextCounter, contextNumber, contextADC
        )

        # Determine initial output address
        (
            inAddr_Datums,
            datumSizeBytes,
            inputNumDatums,
            inAddr_Deltas,
            inAddr_Exponents,
            rowStride,
            discontiguousInputRows,
            isUncompressed,
            inDataFormat,
        ) = self.generate_input_addresses_and_sizes(
            issue_thread, stateID, multiContextMode, whichContext, whichADC, rowSearch
        )

        outAddr, outDataFormat, unpackToDst, transpose = self.generate_output_address(
            stateID, issue_thread, multiContextMode, whichContext
        )

        upsampleZeroes = (
            1
            << self.getConfigValue(
                stateID, "THCON_SEC" + str(self.unpacker_id) + "_REG2_Upsample_rate"
            )
        ) - 1
        upsampleInterleave = self.getConfigValue(
            stateID,
            "THCON_SEC" + str(self.unpacker_id) + "_REG2_Upsample_and_interleave",
        )
        if discontiguousInputRows or self.unpacker_id == 1:
            colShift = 0
        else:
            colShift = self.getConfigValue(
                stateID,
                "THCON_SEC"
                + str(self.unpacker_id)
                + "_REG2_Shift_amount_cntx"
                + str(whichContext & 3),
            )

        # Check that various settings are compatible with each other:
        self.check_unpacker_settings(
            transpose,
            discontiguousInputRows,
            inAddr_Datums,
            upsampleZeroes,
            isUncompressed,
            unpackToDst,
            colShift,
        )

        # Main unpack loop
        self.perform_unpack(
            stateID,
            issue_thread,
            inputNumDatums,
            inAddr_Datums,
            outAddr,
            datumSizeBytes,
            rowStride,
            upsampleZeroes,
            upsampleInterleave,
            colShift,
            inDataFormat,
            outDataFormat,
            unpackToDst,
            transpose,
            allDatumsAreZero,
        )

        # Increment the counter if applicable
        self.increment_counter(
            stateID, issue_thread, whichContext, multiContextMode, useContextCounter
        )

        # Update ADCs
        self.update_ADC(issue_thread, whichADC, ch0YInc, ch0ZInc, ch1YInc, ch1ZInc)

        # Flip src banks
        self.flip_src_banks(flipSrc, issue_thread)
