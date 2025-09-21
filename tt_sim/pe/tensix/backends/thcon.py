from enum import IntEnum

from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.util.bits import get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class ScalarUnit(TensixBackendUnit):
    class THConStallType(IntEnum):
        FLUSHDMA = 1
        SRC_UNPACKER = 2
        ATCAS = 3

    OPCODE_TO_HANDLER = {
        "SETDMAREG": "handle_setdmareg",
        "REG2FLOP": "handle_reg2flop",
        "STOREREG": "handle_storereg",
        "FLUSHDMA": "handle_flushdma",
        "ADDDMAREG": "handle_adddmareg",
        "SUBDMAREG": "handle_subdmareg",
        "MULDMAREG": "handle_muldmareg",
        "CMPDMAREG": "handle_cmpdmareg",
        "BITWOPDMAREG": "handle_bitwopdmareg",
        "SHIFTDMAREG": "handle_shiftdmareg",
        "STOREIND": "handle_storeind",
        "ATSWAP": "handle_atswap",
        "LOADIND": "handle_loadind",
        "LOADREG": "handle_loadreg",
        "ATINCGET": "handle_atincget",
        "ATCAS": "handle_atcas",
    }

    GLOBAL_CFGREG_BASE_ADDR32 = 152
    THCON_CFGREG_BASE_ADDR32 = 52

    CMPDMAREG_MODE_GT = 0
    CMPDMAREG_MODE_LT = 1
    CMPDMAREG_MODE_EQ = 2
    SHIFTDMAREG_MODE_LEFT = 0
    SHIFTDMAREG_MODE_RIGHT = 1
    BITWOPDMAREG_MODE_AND = 0
    BITWOPDMAREG_MODE_OR = 1
    BITWOPDMAREG_MODE_XOR = 2

    def __init__(self, backend, gprs):
        self.gprs = gprs
        self.stalled = False
        self.stalled_type = None
        self.stalled_condition = 0
        self.stalled_thread = 0
        super().__init__(backend, ScalarUnit.OPCODE_TO_HANDLER, "Scalar")

    def issueInstruction(self, instruction, from_thread):
        if self.stalled:
            return False

        return super().issueInstruction(instruction, from_thread)

    def clock_tick(self, cycle_num):
        if self.stalled:
            if self.stalled_type == ScalarUnit.THConStallType.FLUSHDMA:
                if not self.checkStalledCondition(self.stalled_condition):
                    self.stalled = False
                    self.backend.getFrontendThread(
                        self.stalled_thread
                    ).wait_gate.clearBackendEnforcedStall()
                    self.stalled_condition = self.stalled_thread = 0
            elif self.stalled_type == ScalarUnit.THConStallType.SRC_UNPACKER:
                self.process_srca_srcb_from_gpr(*self.stalled_condition)
            elif self.stalled_type == ScalarUnit.THConStallType.ATCAS:
                self.handle_atcas(*self.stalled_condition)
            else:
                raise NotImplementedError()
        else:
            super().clock_tick(cycle_num)

    def checkStalledCondition(self, stalled_condition):
        if get_nth_bit(self.stalled_condition, 1):
            if self.backend.unpacker_units[0].hasInflightInstructionsFromThread(
                self.stalled_thread
            ):
                return True
        if get_nth_bit(self.stalled_condition, 2):
            if self.backend.unpacker_units[1].hasInflightInstructionsFromThread(
                self.stalled_thread
            ):
                return True

        if get_nth_bit(self.stalled_condition, 3):
            if self.backend.packer_unit.hasInflightInstructionsFromThread(
                self.stalled_thread
            ):
                return True

        return False

    def handle_flushdma(self, instruction_info, issue_thread, instr_args):
        conditionMask = instr_args["FlushSpec"]

        if conditionMask == 0:
            conditionMask = 0xF

        if self.checkStalledCondition(conditionMask):
            self.stalled_thread = issue_thread
            self.stalled_condition = conditionMask
            self.stalled_type = ScalarUnit.THConStallType.FLUSHDMA
            self.stalled = True
            self.backend.getFrontendThread(
                issue_thread
            ).wait_gate.setBackendEnforcedStall()

    def handle_shiftdmareg(self, instruction_info, issue_thread, instr_args):
        leftReg = instr_args["OpARegIndex"]
        rightRegOrImm5 = instr_args["OpBRegIndex"]
        resultReg = instr_args["ResultRegIndex"]
        mode = instr_args["OpSel"]
        use_val = instr_args["OpBisConst"]

        leftVal = self.gprs.getRegisters(issue_thread)[leftReg]

        if use_val == 0:
            rightVal = self.gprs.getRegisters(issue_thread)[rightRegOrImm5] & 0x1F
        else:
            rightVal = rightRegOrImm5 & 0x1F

        match mode:
            case ScalarUnit.SHIFTDMAREG_MODE_LEFT:
                resultVal = leftVal << rightVal
            case ScalarUnit.SHIFTDMAREG_MODE_RIGHT:
                resultVal = leftVal >> rightVal
            case _:
                raise NotImplementedError()

        self.gprs.getRegisters(issue_thread)[resultReg] = resultVal

    def handle_bitwopdmareg(self, instruction_info, issue_thread, instr_args):
        leftReg = instr_args["OpARegIndex"]
        rightRegOrImm6 = instr_args["OpBRegIndex"]
        resultReg = instr_args["ResultRegIndex"]
        mode = instr_args["OpSel"]
        use_val = instr_args["OpBisConst"]

        leftVal = self.gprs.getRegisters(issue_thread)[leftReg]

        if use_val == 0:
            rightVal = self.gprs.getRegisters(issue_thread)[rightRegOrImm6]
        else:
            rightVal = rightRegOrImm6

        match mode:
            case ScalarUnit.BITWOPDMAREG_MODE_AND:
                resultVal = leftVal & rightVal
            case ScalarUnit.BITWOPDMAREG_MODE_OR:
                resultVal = leftVal | rightVal
            case ScalarUnit.BITWOPDMAREG_MODE_XOR:
                resultVal = leftVal ^ rightVal
            case _:
                raise NotImplementedError()

        self.gprs.getRegisters(issue_thread)[resultReg] = resultVal

    def handle_cmpdmareg(self, instruction_info, issue_thread, instr_args):
        leftReg = instr_args["OpARegIndex"]
        rightRegOrImm6 = instr_args["OpBRegIndex"]
        resultReg = instr_args["ResultRegIndex"]
        mode = instr_args["OpSel"]
        use_val = instr_args["OpBisConst"]

        leftVal = self.gprs.getRegisters(issue_thread)[leftReg]

        if use_val == 0:
            rightVal = self.gprs.getRegisters(issue_thread)[rightRegOrImm6]
        else:
            rightVal = rightRegOrImm6

        match mode:
            case ScalarUnit.CMPDMAREG_MODE_GT:
                resultval = leftVal > rightVal
            case ScalarUnit.CMPDMAREG_MODE_LT:
                resultval = leftVal < rightVal
            case ScalarUnit.CMPDMAREG_MODE_EQ:
                resultval = leftVal == rightVal
            case _:
                raise NotImplementedError()

        self.gprs.getRegisters(issue_thread)[resultReg] = 1 if resultval else 0

    def handle_muldmareg(self, instruction_info, issue_thread, instr_args):
        leftReg = instr_args["OpARegIndex"]
        rightRegOrImm6 = instr_args["OpBRegIndex"]
        resultReg = instr_args["ResultRegIndex"]
        mode = instr_args["OpBisConst"]

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"GPR[{resultReg}] = GPR[leftReg] * {'GPR'+str(rightRegOrImm6) if mode == 0 else str(hex(rightRegOrImm6))} "
                f"from thread {issue_thread}"
            )

        leftVal = self.gprs.getRegisters(issue_thread)[leftReg]

        if mode == 0:
            rightVal = self.gprs.getRegisters(issue_thread)[rightRegOrImm6]
        else:
            rightVal = rightRegOrImm6

        result = (leftVal & 0xFFFF) * (rightVal & 0xFFFF)
        self.gprs.getRegisters(issue_thread)[resultReg] = result

    def handle_subdmareg(self, instruction_info, issue_thread, instr_args):
        leftReg = instr_args["OpARegIndex"]
        rightRegOrImm6 = instr_args["OpBRegIndex"]
        resultReg = instr_args["ResultRegIndex"]
        mode = instr_args["OpBisConst"]

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"GPR[{resultReg}] = GPR[leftReg] - {'GPR'+str(rightRegOrImm6) if mode == 0 else str(hex(rightRegOrImm6))} "
                f"from thread {issue_thread}"
            )

        leftVal = self.gprs.getRegisters(issue_thread)[leftReg]

        if mode == 0:
            rightVal = self.gprs.getRegisters(issue_thread)[rightRegOrImm6]
        else:
            rightVal = rightRegOrImm6

        result = leftVal - rightVal
        self.gprs.getRegisters(issue_thread)[resultReg] = result

    def handle_adddmareg(self, instruction_info, issue_thread, instr_args):
        leftReg = instr_args["OpARegIndex"]
        rightRegOrImm6 = instr_args["OpBRegIndex"]
        resultReg = instr_args["ResultRegIndex"]
        mode = instr_args["OpBisConst"]

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"GPR[{resultReg}] = GPR[leftReg] + {'GPR'+str(rightRegOrImm6) if mode == 0 else str(hex(rightRegOrImm6))} "
                f"from thread {issue_thread}"
            )

        leftVal = self.gprs.getRegisters(issue_thread)[leftReg]

        if mode == 0:
            rightVal = self.gprs.getRegisters(issue_thread)[rightRegOrImm6]
        else:
            rightVal = rightRegOrImm6

        result = leftVal + rightVal
        self.gprs.getRegisters(issue_thread)[resultReg] = result

    def handle_reg2flop(self, instruction_info, issue_thread, instr_args):
        inputReg = instr_args["RegIndex"]
        thConCfgIndex = instr_args["FlopIndex"]
        targetSel = instr_args["TargetSel"]
        sizeSel = instr_args["SizeSel"]
        shift8 = instr_args["ByteOffset"]
        threadSel = instr_args["ContextId_2"]

        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        assert thConCfgIndex < (
            ScalarUnit.GLOBAL_CFGREG_BASE_ADDR32 - ScalarUnit.THCON_CFGREG_BASE_ADDR32
        )

        if targetSel == 0x0:
            # Move from GPRs to THCON configuration
            if sizeSel == 0:
                # 128 bit configuration write
                for i in range(4):
                    self.backend.getConfigUnit().setConfig(
                        stateID,
                        (thConCfgIndex + ScalarUnit.THCON_CFGREG_BASE_ADDR32) + i,
                        self.gprs.getRegisters(issue_thread)[inputReg + i],
                    )
            else:
                # 32 bit configuration write
                self.backend.getConfigUnit().setConfig(
                    stateID,
                    thConCfgIndex + ScalarUnit.THCON_CFGREG_BASE_ADDR32,
                    self.gprs.getRegisters(issue_thread)[inputReg],
                )
        else:
            # Move from GPRs to ADCs
            overrideThread = get_nth_bit(targetSel, 0)

            xyzw = get_bits(thConCfgIndex, 0, 1)
            cr = get_bits(thConCfgIndex, 2, 2)
            adcsel = get_bits(thConCfgIndex, 3, 4)
            channel = get_bits(thConCfgIndex, 5, 5)

            value = self.gprs.getRegisters(issue_thread)[inputReg]

            match sizeSel:
                case 0:
                    # 128 bit
                    value = 0
                case 1:
                    # 32 bit
                    if shift8 != 0:
                        value = 0
                case 2:
                    # 16 bit
                    if shift8 == 0:
                        value &= 0xFFFF
                    elif shift8 == 2:
                        value >>= 16
                    else:
                        value = 0
                case 3:
                    # 8 bit
                    value = (value >> (shift8 * 8)) & 0xFF

            if overrideThread:
                whichThread = threadSel
                if whichThread >= 3:
                    return
            else:
                whichThread = issue_thread

            match adcsel:
                case 0:
                    adc = self.backend.getADC(whichThread).Unpacker[0]
                case 1:
                    adc = self.backend.getADC(whichThread).Unpacker[1]
                case 2:
                    adc = self.backend.getADC(whichThread).Packers
                case _:
                    return

            tgt_channel = adc.Channel[channel]

            match xyzw:
                case 0:
                    if cr:
                        tgt_channel.X_Cr = value
                    else:
                        tgt_channel.X = value
                case 1:
                    if cr:
                        tgt_channel.Y_Cr = value
                    else:
                        tgt_channel.Y = value
                case 2:
                    if cr:
                        tgt_channel.Z_Cr = value
                    else:
                        tgt_channel.Z = value
                case 3:
                    if cr:
                        tgt_channel.W_Cr = value
                    else:
                        tgt_channel.W = value

    def handle_storereg(self, instruction_info, issue_thread, instr_args):
        addrLo = instr_args["RegAddr"]
        dataReg = instr_args["TdmaDataRegIndex"]

        addr = 0xFFB00000 + (addrLo << 2)

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"Write to MMIO at {hex(addr)} from reg {dataReg} "
                f"from thread {issue_thread}"
            )

        self.backend.getAddressableMemory().write(
            addr, conv_to_bytes(self.gprs.getRegisters(issue_thread)[dataReg])
        )

    def handle_setdmareg(self, instruction_info, issue_thread, instr_args):
        setSignalsMode = instr_args["SetSignalsMode"]
        if setSignalsMode == 0:
            # Set 16 bits of one GPR
            resultHalfReg = instr_args["RegIndex16b"]
            newValue1 = instr_args["Payload_SigSel"]
            newValue2 = instr_args["Payload_SigSelSize"]

            newValue = newValue1 | (newValue2 << 14)

            base_reg = int(resultHalfReg / 2)

            existing_val = self.gprs.getRegisters(issue_thread)[base_reg]
            if resultHalfReg & 0x1 != 0:
                nv = (existing_val & 0xFFFF) | (newValue << 16)
            else:
                nv = (existing_val & 0xFFFF0000) | newValue

            self.gprs.getRegisters(issue_thread)[base_reg] = nv
        else:
            raise NotImplementedError()

    def handle_atswap(self, instruction_info, issue_thread, instr_args):
        addrReg = instr_args["AddrRegIndex"]
        dataReg = instr_args["DataRegIndex"]
        mask = instr_args["SwapMask"]
        singleDataReg = instr_args["MemHierSel"]

        L1Address = self.gprs.getRegisters(issue_thread)[addrReg] * 16
        assert L1Address < (1464 * 1024)

        toWrite = [0] * 4
        if singleDataReg:
            for i in range(4):
                toWrite[i] = self.gprs.getRegisters(issue_thread)[dataReg + i]
        else:
            for i in range(4):
                toWrite[i] = self.gprs.getRegisters(issue_thread)[(dataReg & 0x3C) + i]

        for i in range(8):
            if mask & (1 << i):
                val = toWrite[int(i / 2)]
                if i % 2 == 0:
                    # Low part
                    val &= 0xFFFF
                else:
                    # High part
                    val >>= 16
                self.backend.getAddressableMemory().write(
                    L1Address + (i * 2), conv_to_bytes(val, 2)
                )

    def handle_atcas(self, instruction_info, issue_thread, instr_args):
        # L1 atomic compare and set
        addrReg = instr_args["AddrRegIndex"]
        Ofs = instr_args["Sel32b"]
        cmpVal = instr_args["CmpVal"]
        setVal = instr_args["SwapVal"] & 0xF

        L1Address = self.gprs.getRegisters(issue_thread)[addrReg] * 16 + Ofs * 4
        assert L1Address < (1464 * 1024)

        testValue = conv_to_uint32(
            self.backend.getAddressableMemory().read(L1Address, 4)
        )

        if testValue != cmpVal:
            self.stalled_condition = (instruction_info, issue_thread, instr_args)
            self.stalled_type = ScalarUnit.THConStallType.ATCAS
            self.stalled = True
        else:
            self.stalled = False
            self.stalled_condition = self.stalled_type = None
            self.backend.getAddressableMemory().write(L1Address, conv_to_bytes(setVal))

    def handle_atincget(self, instruction_info, issue_thread, instr_args):
        # Atomically increment an integer in L1
        addrReg = instr_args["AddrRegIndex"]
        inOutReg = instr_args["DataRegIndex"]
        Ofs = instr_args["Sel32b"]
        intWidth = instr_args["WrapVal"] & 0x1F

        L1Address = self.gprs.getRegisters(issue_thread)[addrReg] * 16 + Ofs * 4
        assert L1Address < (1464 * 1024)

        incrementBy = self.gprs.getRegisters(issue_thread)[inOutReg]
        intMask = (2 << intWidth) - 1

        originalValue = conv_to_uint32(
            self.backend.getAddressableMemory().read(L1Address, 4)
        )
        incremented = originalValue + incrementBy
        self.backend.getAddressableMemory().write(
            L1Address,
            conv_to_bytes((incremented & intMask) | (originalValue & ~intMask)),
        )

        self.gprs.getRegisters(issue_thread)[inOutReg] = originalValue

    def handle_loadreg(self, instruction_info, issue_thread, instr_args):
        # Read from MMIO into a register
        addrLo = instr_args["RegAddr"]
        resultReg = instr_args["TdmaDataRegIndex"]

        addr = 0xFFB00000 + (addrLo << 2)
        assert addr >= 0xFFB11000

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"Write to GPR {resultReg} from MMIO at {hex(addr)} "
                f"from thread {issue_thread}"
            )

        self.gprs.getRegisters(issue_thread)[resultReg] = conv_to_uint32(
            self.backend.getAddressableMemory().read(addr, 4)
        )

    def handle_loadind(self, instruction_info, issue_thread, instr_args):
        addrReg = instr_args["AddrRegIndex"]
        resultReg = instr_args["DataRegIndex"]
        offsetIncrement = instr_args["AutoIncSpec"]
        offsetHalfReg = instr_args["OffsetIndex"]
        sizeSel = instr_args["sizeSel"]

        gpr_reg_idx = resultReg & (0x3F if sizeSel else 0x3C)
        offset_val = self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2]
        L1Address = (self.gprs.getRegisters(issue_thread)[addrReg] * 16) + offset_val
        assert L1Address < (1464 * 1024)

        match offsetIncrement:
            case 0:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 0
            case 1:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 2
            case 2:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 4
            case 3:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 16

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"Read from L1 address {hex(L1Address)} to GPR {gpr_reg_idx}, number of bytes "
                f"{16 if sizeSel == 0 else 4 if sizeSel == 1 else 2 if sizeSel == 2 else 1} "
                f"from thread {issue_thread}"
            )

        match sizeSel:
            case 0:
                # Four consecutive GPRs
                L1Address &= ~15
                for i in range(4):
                    l1_val = conv_to_uint32(
                        self.backend.getAddressableMemory().read(L1Address, 4)
                    )
                    self.gprs.getRegisters(issue_thread)[gpr_reg_idx + i] = l1_val
                    L1Address += 4
            case 1:
                l1_val = conv_to_uint32(
                    self.backend.getAddressableMemory().read(L1Address & ~3, 4)
                )
                self.gprs.getRegisters(issue_thread)[gpr_reg_idx] = l1_val
            case 2:
                # Low 16 bits of GPR
                l1_val = conv_to_uint32(
                    self.backend.getAddressableMemory().read(L1Address & ~1, 2)
                )
                gpr_val = self.gprs.getRegisters(issue_thread)[gpr_reg_idx] & 0xFFFF0000
                self.gprs.getRegisters(issue_thread)[gpr_reg_idx] = gpr_val | l1_val
            case 3:
                # Low 8 bits of GPR
                l1_val = conv_to_uint32(
                    self.backend.getAddressableMemory().read(L1Address, 1)
                )
                gpr_val = self.gprs.getRegisters(issue_thread)[gpr_reg_idx] & 0xFFFFFF00
                self.gprs.getRegisters(issue_thread)[gpr_reg_idx] = gpr_val | l1_val

    def handle_storeind(self, instruction_info, issue_thread, instr_args):
        addrReg = instr_args["AddrRegIndex"]
        dataReg = instr_args["DataRegIndex"]
        offsetIncrement = instr_args["AutoIncSpec"]
        offsetHalfReg = instr_args["OffsetIndex"]
        memHierSel = instr_args["MemHierSel"]
        regSizeSel = instr_args["RegSizeSel"]
        sizeSel = instr_args["SizeSel"]

        if memHierSel == 1:
            self.process_storeind_l1_from_gpr(
                issue_thread,
                addrReg,
                dataReg,
                offsetIncrement,
                offsetHalfReg,
                memHierSel,
                regSizeSel,
                sizeSel,
            )
        else:
            if sizeSel == 1:
                self.process_mmio_from_gpr(
                    issue_thread, addrReg, dataReg, offsetIncrement, offsetHalfReg
                )
            else:
                self.process_srca_srcb_from_gpr(
                    issue_thread,
                    addrReg,
                    dataReg,
                    offsetIncrement,
                    offsetHalfReg,
                    regSizeSel,
                )

    def process_srca_srcb_from_gpr(
        self,
        issue_thread,
        addrReg,
        dataReg,
        offsetIncrement,
        offsetHalfReg,
        storeToSrcB,
    ):
        # SrcA/SrcB write 4xBF16 from 2xGPR
        datum = [0] * 4
        datum[0] = self.readBF16Lo(self.gprs.getRegisters(issue_thread)[dataReg & 0x3C])
        datum[1] = self.readBF16Hi(self.gprs.getRegisters(issue_thread)[dataReg & 0x3C])
        datum[2] = self.readBF16Lo(
            self.gprs.getRegisters(issue_thread)[(dataReg & 0x3C) + 1]
        )
        datum[3] = self.readBF16Hi(
            self.gprs.getRegisters(issue_thread)[(dataReg & 0x3C) + 1]
        )

        offset_val = self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2]
        addr = self.gprs.getRegisters(issue_thread)[addrReg] + (offset_val >> 4)

        match offsetIncrement:
            case 0:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 0
            case 1:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 2
            case 2:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 4
            case 3:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 16

        assert not (addr & 0xF0000)

        column0 = (addr & 3) * 4
        if storeToSrcB:
            bank = self.backend.unpackers[1].srcBank

            if (
                self.backend.getSrcB[bank].allowedClient
                != SrcRegister.SrcClient.Unpackers
            ):
                self.stalled_condition = (
                    issue_thread,
                    addrReg,
                    dataReg,
                    offsetIncrement,
                    offsetHalfReg,
                    storeToSrcB,
                )
                self.stalled_type = ScalarUnit.THConStallType.SRC_UNPACKER
                self.stalled = True
            else:
                self.stalled = False
                self.stalled_condition = self.stalled_type = None

            row = addr >> 2
            assert row < 16
            row += self.backend.unpackers[1].srcRow[
                issue_thread
            ]  # Will add 0 / 16 / 32 / 48
            for i in range(4):
                self.backend.getSrcB(bank)[row][column0 + i] = datum[i]
        else:
            bank = self.backend.unpackers[0].srcBank
            if (
                self.backend.getSrcA[bank].allowedClient
                != SrcRegister.SrcClient.Unpackers
            ):
                self.stalled_condition = (
                    issue_thread,
                    addrReg,
                    dataReg,
                    offsetIncrement,
                    offsetHalfReg,
                    storeToSrcB,
                )
                self.stalled_type = ScalarUnit.THConStallType.SRC_UNPACKER
                self.stalled = True
            else:
                self.stalled = False
                self.stalled_condition = self.stalled_type = None
            row = (addr >> 2) - 4
            if row >= 0:
                if self.getThreadConfigValue(issue_thread, "SRCA_SET_SetOvrdWithAddr"):
                    assert row < 64
                else:
                    assert row < 16
                    row += self.backend.unpackers[0].srcRow[issue_thread]
                for i in range(4):
                    self.backend.getSrcA(bank)[row][column0 + i] = datum[i]

    def process_mmio_from_gpr(
        self,
        issue_thread,
        addrReg,
        dataReg,
        offsetIncrement,
        offsetHalfReg,
    ):
        # MMIO register write from GPR
        offset_val = self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2]
        addr = self.gprs.getRegisters(issue_thread)[addrReg] + (offset_val >> 4)
        addr = 0xFFB00000 + (addr & 0x000FFFFC)

        assert addr >= 0xFFB11000

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"Write to MMIO at address {hex(addr)} from GPR {dataReg} from thread {issue_thread}"
            )

        gpr_val = self.gprs.getRegisters(issue_thread)[dataReg]
        self.backend.getAddressableMemory().write(addr, conv_to_bytes(gpr_val))

        match offsetIncrement:
            case 0:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 0
            case 1:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 2
            case 2:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 4
            case 3:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 16

    def process_storeind_l1_from_gpr(
        self,
        issue_thread,
        addrReg,
        dataReg,
        offsetIncrement,
        offsetHalfReg,
        memHierSel,
        regSizeSel,
        sizeSel,
    ):
        # L1 write from GPR
        size = ((sizeSel << 1) | regSizeSel) & 0x3
        gpr_reg_idx = dataReg & (0x3F if size else 0x3C)
        offset_val = self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2]
        L1Address = (self.gprs.getRegisters(issue_thread)[addrReg] * 16) + offset_val
        assert L1Address < (1464 * 1024)

        if self.getDiagnosticSettings().reportThCon():
            print(
                f"Write to L1 address {hex(L1Address)} from GPR {gpr_reg_idx}, number of bytes "
                f"{16 if size == 0 else 4 if size == 1 else 2 if size == 2 else 1} "
                f"from thread {issue_thread}"
            )

        match offsetIncrement:
            case 0:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 0
            case 1:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 2
            case 2:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 4
            case 3:
                self.gprs.getRegisters(issue_thread)[offsetHalfReg * 2] += 16

        match size:
            case 0:
                # Four consecutive GPRs
                L1Address &= ~15
                for i in range(4):
                    gpr_val = self.gprs.getRegisters(issue_thread)[gpr_reg_idx + i]
                    self.backend.getAddressableMemory().write(
                        L1Address, conv_to_bytes(gpr_val)
                    )
                    L1Address += 4
            case 1:
                gpr_val = self.gprs.getRegisters(issue_thread)[gpr_reg_idx]
                self.backend.getAddressableMemory().write(
                    L1Address & ~3, conv_to_bytes(gpr_val)
                )
            case 2:
                # Low 16 bits of GPR
                gpr_val = self.gprs.getRegisters(issue_thread)[gpr_reg_idx] & 0xFFFF
                self.backend.getAddressableMemory().write(
                    L1Address & ~1, conv_to_bytes(gpr_val, 2)
                )
            case 3:
                # Low 8 bits of GPR
                gpr_val = self.gprs.getRegisters(issue_thread)[gpr_reg_idx] & 0xFF
                self.backend.getAddressableMemory().write(
                    L1Address, conv_to_bytes(gpr_val, 1)
                )

    def readBF16Lo(self, x):
        sign = x & 0x8000
        man = x & 0x7F00
        exp = x & 0x00FF

        return (sign << 3) | (man << 3) | exp

    def readBF16Hi(self, x):
        sign = x >> 31
        man = (x >> 23) & 0xFF
        exp = (x >> 16) & 0x7F

        return (sign << 18) | (man << 11) | exp
