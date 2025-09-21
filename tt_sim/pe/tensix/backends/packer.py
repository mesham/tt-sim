from enum import IntEnum
from math import ceil

from tt_sim.pe.tensix.backends.backend_base import (
    DATA_FORMAT_TO_BITS,
    DataFormat,
    TensixBackendUnit,
)
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_bytes


class PackerUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {"PACR": "handle_pacr"}

    packer0InitialAddr = 0
    datastreamNeedsNewAddr = True
    byteAddress = 0

    class PackerI:
        def __init__(self):
            self.inputNumDatums = 0
            self.inputSource = 0
            self.inputSourceAddr = 0
            self.inputSourceStride = 0
            self.byteAddress = 0
            self.datastreamNeedsNewAddr = 0
            self.outBytes = 0

    class InputSource(IntEnum):
        L1 = 1
        DST = 2

    def __init__(self, backend):
        self.packerI = [PackerUnit.PackerI() for i in range(4)]
        super().__init__(backend, PackerUnit.OPCODE_TO_HANDLER, "Packer")

    def getIPackerConfig(self, i, s, one_id, two_id=0):
        if s == 2:
            match i:
                case 0:
                    return "THCON_SEC0_REG" + str(one_id) + "_"
                case 1:
                    return "THCON_SEC0_REG" + str(one_id) + "_"
                case 2:
                    return "THCON_SEC1_REG" + str(one_id) + "_"
                case 3:
                    return "THCON_SEC1_REG" + str(one_id) + "_"
                case _:
                    raise ValueError()
        elif s == 4:
            match i:
                case 0:
                    return "THCON_SEC0_REG" + str(one_id) + "_"
                case 1:
                    return "THCON_SEC0_REG" + str(two_id) + "_"
                case 2:
                    return "THCON_SEC1_REG" + str(one_id) + "_"
                case 3:
                    return "THCON_SEC1_REG" + str(two_id) + "_"
                case _:
                    raise ValueError()

    def generate_input_address(
        self, stateID, issue_thread, flush, zeroWrite, addrMod, packMask, ovrdThreadId
    ):
        ADCsToAdvance = [False, False, False]
        for i in range(4):
            if not (get_nth_bit(packMask, i) or (i == 0 and packMask == 0x0)):
                continue

            whichADC = issue_thread
            if ovrdThreadId:
                whichADC = self.getConfigValue(
                    stateID,
                    self.getConfigValue(
                        stateID, self.getIPackerConfig(i, 4, 1, 8) + "Addr_cnt_context"
                    ),
                )
                if whichADC == 3:
                    whichADC = 0

            adc = self.backend.getADC(whichADC).Packers.Channel[0]
            addr = (
                self.getConfigValue(stateID, "PCK0_ADDR_BASE_REG_0_Base")
                + adc.X
                * (self.getConfigValue(stateID, "PCK0_ADDR_BASE_REG_0_Base") & 0xF)
                + adc.Y
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_XY_REG_0_Ystride")
                + adc.Z
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_0_Zstride")
                + adc.W
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_0_Wstride")
            )

            ADCsToAdvance[whichADC] = True

            match (
                self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 4, 1, 8) + "In_data_format"
                )
                & 3
            ):
                case 0:
                    # FP32, TF32, I32
                    bytesPerDatum = 4
                    ADC_X_Mask = 0x3
                case 1:
                    # FP16, BF16, I16
                    bytesPerDatum = 2
                    ADC_X_Mask = 0x7
                case _:
                    # All other formats
                    bytesPerDatum = 1
                    ADC_X_Mask = 0xF

            if flush:
                self.packerI[i].inputNumDatums = 0
            else:
                self.packerI[i].inputNumDatums = (
                    self.backend.getADC(whichADC).Packers.Channel[1].X - adc.X + 1
                )

            if zeroWrite or flush:
                self.packerI[i].inputSource = None
                self.packerI[i].inputSourceAddr = 0
                self.packerI[i].inputSourceStride = 0
            elif (
                i == 0
                and self.getConfigValue(
                    stateID,
                    self.getIPackerConfig(i, 4, 1, 8) + "Source_interface_selection",
                )
                == 1
            ):
                # Only low 18 bits of Addr used; high bits of L1 address come from L1_source_addr
                addr = (
                    self.getConfigValue(
                        stateID, self.getIPackerConfig(i, 4, 1, 8) + "L1_source_addr"
                    )
                    << 18
                ) + (addr & 0x3FFFF)
                addr = (addr & ~0xF) + bytesPerDatum * (adc.X & ADC_X_Mask)
                self.packerI[i].inputSource = PackerUnit.InputSource.L1
                self.packerI[i].inputSourceAddr = (
                    addr & 0x1FFFFF
                )  # Byte address into L1
                self.packerI[
                    i
                ].inputSourceStride = bytesPerDatum  # L1 is addressed in bytes
            else:
                addr = (int(addr / bytesPerDatum) & ~ADC_X_Mask) + (adc.X & ADC_X_Mask)
                offset = (
                    self.getConfigValue(
                        stateID, "DEST_TARGET_REG_CFG_PACK_SEC" + str(i) + "_Offset"
                    )
                    << 4
                )
                if i == 0 and offset == 32:
                    # This is a bug fix, a strange issue where unpacker core writes 32 after unpack
                    # into GPR 4, this is expected to hold the offset but is overwritten by this
                    # The other way of looking at it is that for packer 0 we should align with a
                    # segment, therefore if do not (% 64) then round down
                    offset = 0
                addr += offset
                self.packerI[i].inputSource = PackerUnit.InputSource.DST
                self.packerI[i].inputSourceAddr = (
                    addr & 0x3FFF
                )  # Datum index into Dst; `>> 4` is row, `& 0xf` is column
                self.packerI[
                    i
                ].inputSourceStride = 1  # Dst is addressed in datums, not in bytes

        for i in range(3):
            if not ADCsToAdvance[i]:
                continue

            adc = self.backend.getADC(i).Packers.Channel[0]

            AM_KEY = "ADDR_MOD_PACK_SEC" + str(addrMod)

            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YsrcClear"):
                adc.Y = 0
                adc.Y_Cr = 0
            elif self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YsrcCR"):
                adc.Y_Cr += self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_YsrcIncr"
                )
                adc.Y = adc.Y_Cr
            else:
                adc.Y = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_YsrcIncr"
                )

            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_ZsrcClear"):
                adc.Z = 0
                adc.Z_Cr = 0
            else:
                adc.Z = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_ZsrcIncr"
                )

    def generate_output_address(
        self, stateID, issue_thread, packMask, flush, last, addrMod, ovrdThreadId
    ):
        ADCsToAdvance = [False, False, False]
        for i in range(4):
            # was ! on the second arg below
            addr = self.getConfigValue(
                stateID, self.getIPackerConfig(i, 4, 1, 8) + "L1_Dest_addr"
            ) + self.getConfigValue(
                stateID, self.getIPackerConfig(i, 4, 1, 8) + "Sub_l1_tile_header_size"
            )

            if i == 0:
                packer0InitialAddr = addr
            elif get_nth_bit(packer0InitialAddr, 31):
                addr += packer0InitialAddr

            if not (get_nth_bit(packMask, i) or (i == 0 and packMask == 0x0)):
                continue

            whichADC = issue_thread
            if ovrdThreadId:
                whichADC = self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 4, 1, 8) + "Addr_cnt_context"
                )
                if whichADC == 3:
                    whichADC = 0

            adc = self.backend.getADC(whichADC).Packers.Channel[1]
            yzw_addr = (
                self.getConfigValue(stateID, "PCK0_ADDR_BASE_REG_1_Base")
                + adc.Y
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_XY_REG_1_Ystride")
                + adc.Z
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_1_Zstride")
                + adc.W
                * self.getConfigValue(stateID, "PCK0_ADDR_CTRL_ZW_REG_1_Wstride")
            )
            addr += yzw_addr & ~0xF
            ADCsToAdvance[whichADC] = True

            if self.getConfigValue(
                stateID, self.getIPackerConfig(i, 4, 1, 8) + "Add_l1_dest_addr_offset"
            ):
                raise NotImplementedError()

            if (
                addr
                > self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 2, 2) + "Unpack_limit_address"
                )
                * 2
                + 1
            ):
                addr -= (
                    self.getConfigValue(
                        stateID, self.getIPackerConfig(i, 2, 2) + "Unpack_fifo_size"
                    )
                    * 2
                )

            performingCompression = not (
                get_nth_bit(
                    self.getConfigValue(
                        stateID,
                        self.getIPackerConfig(i, 2, 1)
                        + "All_pack_disable_zero_compress",
                    ),
                    i,
                )
                if self.getConfigValue(
                    stateID,
                    self.getIPackerConfig(i, 2, 1)
                    + "All_pack_disable_zero_compress_ovrd",
                )
                else self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 2, 1) + "Disable_zero_compress"
                )
            )
            if performingCompression:
                raise NotImplementedError()

            outputFormatLessThan16Bits = (
                self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 4, 1, 8) + "Out_data_format"
                )
                & 2
            )

            if outputFormatLessThan16Bits:
                raise NotImplementedError()

            output_data_format = DataFormat(
                self.getConfigValue(
                    stateID, self.getIPackerConfig(i, 4, 1, 8) + "Out_data_format"
                )
            )
            self.packerI[i].outBytes = ceil(DATA_FORMAT_TO_BITS[output_data_format] / 8)

            if self.packerI[i].datastreamNeedsNewAddr:
                self.packerI[i].byteAddress = (addr & 0x1FFFF) << 4
                self.packerI[i].datastreamNeedsNewAddr = False

            if last or flush:
                self.packerI[i].datastreamNeedsNewAddr = True

        for i in range(3):
            if not ADCsToAdvance[i]:
                continue

            adc = self.backend.getADC(i).Packers.Channel[1]

            AM_KEY = "ADDR_MOD_PACK_SEC" + str(addrMod)

            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YsrcClear"):
                adc.Y = 0
                adc.Y_Cr = 0
            elif self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_YsrcCR"):
                adc.Y_Cr += self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_YsrcIncr"
                )
                adc.Y = adc.Y_Cr
            else:
                adc.Y = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_YsrcIncr"
                )

            if self.backend.getThreadConfigValue(issue_thread, AM_KEY + "_ZsrcClear"):
                adc.Z = 0
                adc.Z_Cr = 0
            else:
                adc.Z = self.backend.getThreadConfigValue(
                    issue_thread, AM_KEY + "_ZsrcIncr"
                )

    def handle_pacr(self, instruction_info, issue_thread, instr_args):
        last = instr_args["Last"]
        flush = get_nth_bit(instr_args["Flush"], 0)
        ovrdThreadId = instr_args["OvrdThreadId"]
        packMask = instr_args["PackSel"]
        zeroWrite = get_nth_bit(instr_args["ZeroWrite"], 0)
        addrMod = instr_args["AddrMode"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        self.generate_input_address(
            stateID, issue_thread, flush, zeroWrite, addrMod, packMask, ovrdThreadId
        )
        self.generate_output_address(
            stateID, issue_thread, packMask, flush, last, addrMod, ovrdThreadId
        )

        for i in range(1):
            if not (get_nth_bit(packMask, i) or (i == 0 and packMask == 0x0)):
                continue

            addr = self.packerI[i].byteAddress

            row_start = int(self.packerI[i].inputSourceAddr / (16))
            rows = int(self.packerI[i].inputNumDatums / 16)

            # Are offset by this amount (packer does the same, but +1 is added
            # before the << 4
            addr += 0x10

            addr += 0x2
            if self.getDiagnosticSettings().reportPacking():
                print(
                    f"Packer {i}: Copy from {self.packerI[i].inputSourceAddr} (row start "
                    f"= {row_start}, num rows= {rows}) total size "
                    f"{self.packerI[i].inputNumDatums} to {hex(addr)}"
                )

            # For example four need an extra two for alignment also
            for j in range(self.packerI[i].inputNumDatums):
                idx = self.packerI[i].inputSourceAddr + j
                row = idx >> 4
                col = idx & 0xF
                val = self.backend.getDst().getDst32b(row, col)
                self.backend.addressable_memory.write(
                    addr, conv_to_bytes(val, self.packerI[i].outBytes)
                )
                addr += self.packerI[i].outBytes
