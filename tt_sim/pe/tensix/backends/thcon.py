from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.util.bits import get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_bytes


class ScalarUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {
        "SETDMAREG": "handle_setdmareg",
        "REG2FLOP": "handle_reg2flop",
        "STOREREG": "handle_storereg",
        "FLUSHDMA": "handle_flushdma",
    }

    GLOBAL_CFGREG_BASE_ADDR32 = 152
    THCON_CFGREG_BASE_ADDR32 = 52

    def __init__(self, backend, gprs):
        self.gprs = gprs
        self.stalled = False
        self.stalled_condition = 0
        self.stalled_thread = 0
        super().__init__(backend, ScalarUnit.OPCODE_TO_HANDLER, "Scalar")

    def issueInstruction(self, instruction, from_thread):
        if self.stalled:
            return False

        return super().issueInstruction(instruction, from_thread)

    def clock_tick(self, cycle_num):
        if self.stalled:
            if not self.checkStalledCondition(self.stalled_condition):
                self.stalled = False
                self.backend.getFrontendThread(
                    self.stalled_thread
                ).wait_gate.clearBackendEnforcedStall()
                self.stalled_condition = self.stalled_thread = 0
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
            self.stalled = True
            self.backend.getFrontendThread(
                issue_thread
            ).wait_gate.setBackendEnforcedStall()

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
