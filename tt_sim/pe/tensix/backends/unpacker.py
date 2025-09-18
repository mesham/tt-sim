from math import ceil, floor

from tt_sim.pe.tensix.backends.backend_base import (
    DATA_FORMAT_TO_BITS,
    DataFormat,
    TensixBackendUnit,
)
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.util.bits import get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_uint32


class UnPackerUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {"UNPACR": "handle_unpacr"}

    def __init__(self, backend, unpacker_id):
        self.unpacker_id = unpacker_id
        self.context_counter = [0] * 3
        self.srcBank = 0
        self.srcRow = [0] * 3
        self.blocked = False
        self.repeat_instruction = None
        super().__init__(backend, UnPackerUnit.OPCODE_TO_HANDLER, "Unpacker")

    def clock_tick(self, cycle_num):
        if self.blocked:
            assert self.repeat_instruction is not None
            instruction_info, issue_thread = self.repeat_instruction
            self.handle_regular(
                instruction_info, issue_thread, instruction_info["instr_args"]
            )
        else:
            super().clock_tick(cycle_num)

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

        if multiContextMode and self.unpacker_id != 0:
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
        outDataFormat,
        unpackToDst,
        transpose,
        allDatumsAreZero,
    ):
        if self.unpacker_id == 1:
            inAddr_Datums += 8
        else:
            inAddr_Datums += 4

        if self.getDiagnosticSettings().reportUnpacking():
            print(
                f"Starting unpacker {self.unpacker_id} read at {hex(inAddr_Datums)} for "
                f"{inputNumDatums} datums of bytes size {datumSizeBytes} storing to "
                f"{outAddr} starting at row {int(outAddr / 16)}"
            )
        for i in range(inputNumDatums):  # don't handle DecompressNumDatums for now
            val = conv_to_uint32(
                self.backend.addressable_memory.read(inAddr_Datums, datumSizeBytes)
            )

            if allDatumsAreZero:
                val = 0
            inAddr_Datums += datumSizeBytes
            if (i + 1) % 16 == 0:
                inAddr_Datums -= datumSizeBytes * 16
                inAddr_Datums += rowStride
                inAddr_Datums = self.wrapAddr(stateID, inAddr_Datums)

            for k in range(upsampleZeroes + 1):
                if upsampleInterleave and k == 0:
                    continue
                row = int(outAddr / 16)
                col = outAddr & 15

                if self.unpacker_id == 1:
                    # always srcB
                    row = (row + self.srcRow[issue_thread]) & 0x3F
                    self.backend.getSrcB(self.srcBank)[row, col] = val
                else:
                    # Always srcA
                    if not unpackToDst:
                        row = int(outAddr / 16)
                        col = outAddr & 15
                        col -= colShift
                        if self.backend.getThreadConfigValue(
                            issue_thread, "SRCA_SET_SetOvrdWithAddr"
                        ):
                            assert row < 64
                        else:
                            assert row < 16
                            row += self.srcRow[issue_thread]

                            if transpose:
                                rowLowBits = col
                                col = row & 0xF
                                row = (row & ~0xF) | rowLowBits
                        self.backend.getSrcA(self.srcBank)[row, col] = val
                    else:
                        if self.backend.getThreadConfigValue(
                            issue_thread, "SRCA_SET_SetOvrdWithAddr"
                        ):
                            row &= 15
                        else:
                            row &= 0x3FF
                            if DATA_FORMAT_TO_BITS[outDataFormat] == 32:
                                self.backend.getDst().setDst32b(row, col, val)
                            else:
                                self.backend.getDst().setDst16b(row, col, val)
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
