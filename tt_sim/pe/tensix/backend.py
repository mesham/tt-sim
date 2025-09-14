from enum import IntEnum

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.backends.matrix import MatrixUnit
from tt_sim.pe.tensix.backends.vector import VectorUnit
from tt_sim.pe.tensix.registers import DstRegister, SrcRegister
from tt_sim.pe.tensix.util import TensixConfigurationConstants, TensixInstructionDecoder
from tt_sim.util.bits import extract_bits, get_nth_bit, int_to_bin_list, replace_bits
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class ADCThread:
    class ADCUnit:
        class ADCChannel:
            def __init__(self):
                self.X = 0
                self.X_Cr = 0
                self.Y = 0
                self.Y_Cr = 0
                self.Z = 0
                self.Z_Cr = 0
                self.W = 0
                self.W_Cr = 0

        def __init__(self):
            self.Channel = [ADCThread.ADCUnit.ADCChannel()] * 2

    def __init__(self):
        self.Unpacker = [ADCThread.ADCUnit()] * 2
        self.Packers = ADCThread.ADCUnit()


class RCW:
    def __init__(self, backend):
        self.Dst = 0
        self.Dst_Cr = 0
        self.SrcA = 0
        self.SrcA_Cr = 0
        self.SrcB = 0
        self.SrcB_Cr = 0
        self.FidelityPhase = 0
        self.ExtraAddrModBit = 0
        self.backend = backend

    def applyAddrMod(self, thread_id, addrmod, updateFidelityPhase=True):
        if self.ExtraAddrModBit or self.backend.getThreadConfigValue(
            thread_id, "ADDR_MOD_SET_Base"
        ):
            addrmod += 4

        AB_key = "ADDR_MOD_AB_SEC" + str(addrmod)
        Dst_key = "ADDR_MOD_DST_SEC" + str(addrmod)
        Bias_key = "ADDR_MOD_BIAS_SEC" + str(addrmod)

        if self.backend.getThreadConfigValue(thread_id, AB_key + "_SrcAClear"):
            self.SrcA = 0
            self.SrcA_Cr = 0
        elif self.backend.getThreadConfigValue(thread_id, AB_key + "_SrcACR"):
            self.SrcA_Cr += self.backend.getThreadConfigValue(
                thread_id, AB_key + "_SrcAIncr"
            )
            self.SrcA = self.SrcA_Cr
        else:
            self.SrcA += self.backend.getThreadConfigValue(
                thread_id, AB_key + "_SrcAIncr"
            )

        if self.backend.getThreadConfigValue(thread_id, AB_key + "_SrcBClear"):
            self.SrcB = 0
            self.SrcB_Cr = 0
        elif self.backend.getThreadConfigValue(thread_id, AB_key + "_SrcBCR"):
            self.SrcB_Cr += self.backend.getThreadConfigValue(
                thread_id, AB_key + "_SrcBIncr"
            )
            self.SrcB = self.SrcB_Cr
        else:
            self.SrcB += self.backend.getThreadConfigValue(
                thread_id, AB_key + "_SrcBIncr"
            )

        if self.backend.getThreadConfigValue(thread_id, Dst_key + "_DestClear"):
            self.Dst = 0
            self.Dst_Cr = 0
        elif self.backend.getThreadConfigValue(thread_id, Dst_key + "_DestCToCR"):
            self.Dst += self.backend.getThreadConfigValue(
                thread_id, Dst_key + "_DestIncr"
            )
            self.Dst_Cr = self.Dst
        elif self.backend.getThreadConfigValue(thread_id, Dst_key + "_DestCR"):
            self.Dst_Cr += self.backend.getThreadConfigValue(
                thread_id, Dst_key + "_DestIncr"
            )
            self.Dst = self.Dst_Cr
        else:
            self.Dst += self.backend.getThreadConfigValue(
                thread_id, Dst_key + "_DestIncr"
            )

        if updateFidelityPhase:
            # SFPLOAD / SFPSTORE / SFPLOADMACRO do not update FidelityPhase, all other instructions do.
            if self.backend.getThreadConfigValue(thread_id, Dst_key + "_FidelityClear"):
                self.FidelityPhase = 0
            else:
                self.FidelityPhase += self.backend.getThreadConfigValue(
                    thread_id, Dst_key + "_FidelityIncr"
                )

        if self.backend.getThreadConfigValue(thread_id, Bias_key + "_BiasClear"):
            self.ExtraAddrModBit = 0
        elif self.backend.getThreadConfigValue(thread_id, Bias_key + "_BiasIncr") & 3:
            self.ExtraAddrModBit += 1

    def applyPartialAddrMod(self, thread_id, addrMod):
        self.applyAddrMod(thread_id, addrMod, False)


class TensixBackend:
    def __init__(self):
        self.gpr = TensixGPR()
        self.mover_unit = MoverUnit(self)
        self.sync_unit = TensixSyncUnit(self)
        self.matrix_unit = MatrixUnit(self)
        self.scalar_unit = ScalarUnit(self, self.gpr)
        self.vector_unit = VectorUnit(self)
        self.unpacker_units = [UnPackerUnit(self)] * 2
        self.packer_units = [PackerUnit(self)] * 4
        self.misc_unit = MiscellaneousUnit(self)
        self.config_unit = TensixBackendConfigurationUnit(self, self.gpr)
        self.dst = DstRegister()
        self.srcA = [SrcRegister(), SrcRegister()]
        self.srcB = [SrcRegister(), SrcRegister()]
        self.backend_units = {
            "MATH": self.matrix_unit,
            "SFPU": self.vector_unit,
            "THCON": self.scalar_unit,
            "SYNC": self.sync_unit,
            "XMOV": self.mover_unit,
            "TDMA": self.misc_unit,
            "CFG": self.config_unit,
        }
        self.rcw = [RCW(self)] * 3
        self.adc = [ADCThread()] * 3
        self.addressable_memory = None

    def getRCW(self, thread_id):
        assert thread_id <= 2
        return self.rcw[thread_id]

    def getADC(self, thread_id):
        assert thread_id <= 2
        return self.adc[thread_id]

    def getMoverUnit(self):
        return self.mover_unit

    def setFrontendThreads(self, frontend_threads):
        self.frontend_threads = frontend_threads

    def getFrontendThread(self, thread_id):
        assert thread_id < 3
        return self.frontend_threads[thread_id]

    def getSyncUnit(self):
        return self.sync_unit

    def getSrcA(self, idx):
        assert idx < 2
        return self.srcA[idx]

    def getSrcB(self, idx):
        assert idx < 2
        return self.srcB[idx]

    def getDst(self):
        return self.dst

    def getConfigUnit(self):
        return self.config_unit

    def setAddressableMemory(self, addressable_memory):
        self.addressable_memory = addressable_memory

    def getAddressableMemory(self):
        return self.addressable_memory

    def getGPR(self):
        return self.gpr

    def getClocks(self):
        unit_clocks = [
            self.matrix_unit,
            self.scalar_unit,
            self.vector_unit,
            self.misc_unit,
            self.sync_unit,
            self.mover_unit,
            self.config_unit,
        ]
        unit_clocks += self.unpacker_units
        unit_clocks += self.packer_units
        return unit_clocks

    def getThreadConfigValue(self, issue_thread, key):
        addr_idx = TensixConfigurationConstants.get_addr32(key)
        val = self.getConfigUnit().get_threadConfig_entry(issue_thread, addr_idx)
        return TensixConfigurationConstants.parse_raw_config_value(val, key)

    def getConfigValue(self, state_id, key):
        addr_idx = TensixConfigurationConstants.get_addr32(key)
        val = self.getConfigUnit().get_config_entry(state_id, addr_idx)
        return TensixConfigurationConstants.parse_raw_config_value(val, key)

    def hasInflightInstructionsFromThread(self, from_thread):
        for unit in self.backend_units.values():
            if unit.hasInflightInstructionsFromThread(from_thread):
                return True
        for unpacker in self.unpacker_units:
            if unpacker.hasInflightInstructionsFromThread(from_thread):
                return True
        for packer in self.packer_units:
            if packer.hasInflightInstructionsFromThread(from_thread):
                return True
        return False

    def issueInstruction(self, instruction, from_thread):
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        tgt_backend_unit = instruction_info["ex_resource"]
        print(
            f"Issue {instruction_info['name']} to {tgt_backend_unit} from thread {from_thread}"
        )
        if tgt_backend_unit != "NONE":
            if tgt_backend_unit == "UNPACK":
                unpacker = get_nth_bit(instruction, 23)
                self.unpacker_units[unpacker].issueInstruction(instruction, from_thread)
            elif tgt_backend_unit == "PACK":
                packers_int = extract_bits(instruction, 4, 8)
                if packers_int == 0x0:
                    self.packer_units[0].issueInstruction(instruction, from_thread)
                else:
                    packers = int_to_bin_list(packers_int, 4)
                    for idx, packer_bit in enumerate(packers):
                        if packer_bit:
                            # Working left to right, hence 3-idx as the first bit
                            # represents the highest number packer
                            self.packer_units[3 - idx].issueInstruction(
                                instruction, from_thread
                            )
            else:
                assert tgt_backend_unit in self.backend_units
                self.backend_units[tgt_backend_unit].issueInstruction(
                    instruction, from_thread
                )


class TensixGPR(MemMapable):
    class TensixGPRPerTRISCInMem(MemMapable):
        def __init__(self, thread_id, tensix_gpr):
            self.thread_id = thread_id
            self.tensix_gpr = tensix_gpr

        def read(self, addr, size):
            addr += self.thread_id * 64 * 4
            return self.tensix_gpr.read(addr, size)

        def write(self, addr, value, size=None):
            addr += self.thread_id * 64 * 4
            self.tensix_gpr.write(addr, value, size)

        def getSize(self):
            return 0xFFF

    def __init__(self):
        self.registers = [[0] * 64] * 3
        self.GPRPerTRISC = [TensixGPR.TensixGPRPerTRISCInMem(i, self) for i in range(3)]

    def getGPRPerTRISC(self, trisc_id):
        return self.GPRPerTRISC[trisc_id]

    def getRegisters(self, thread_id):
        return self.registers[thread_id]

    def read(self, addr, size):
        base_idx, element_idx = self.get_base_and_element_idx(addr)
        return conv_to_bytes(self.registers[base_idx][element_idx])

    def write(self, addr, value, size=None):
        if size is not None:
            assert size <= 4
        base_idx, element_idx = self.get_base_and_element_idx(addr)
        self.registers[base_idx][element_idx] = conv_to_uint32(value)

    def getSize(self):
        return 0xFFF

    def get_base_and_element_idx(self, addr):
        base_idx = int(addr / (64 * 4))
        element_idx = int((addr - (base_idx * 64 * 4)) / 4)
        return base_idx, element_idx


class ScalarUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {"SETDMAREG": "handle_setdmareg", "REG2FLOP": "handle_reg2flop"}

    def __init__(self, backend, gprs):
        self.gprs = gprs
        super().__init__(backend, ScalarUnit.OPCODE_TO_HANDLER, "Scalar")

    def handle_reg2flop(self, instruction_info, issue_thread, instr_args):
        # TODO
        pass

    def handle_setdmareg(self, instruction_info, issue_thread, instr_args):
        setSignalsMode = instr_args["SetSignalsMode"]
        if setSignalsMode == 0:
            # Set 16 bits of one GPR
            resultHalfReg = instr_args["RegIndex16b"]
            newValue1 = instr_args["Payload_SigSel"]
            newValue2 = instr_args["Payload_SigSelSize"]

            newValue = newValue1 | (newValue2 << 14)

            existing_val = self.gprs.getRegisters(issue_thread)[int(resultHalfReg / 2)]
            self.gprs.getRegisters(issue_thread)[int(resultHalfReg / 2)] = (
                existing_val & 0xFF00
            ) | newValue
        else:
            pass


class TensixSyncUnit(TensixBackendUnit, MemMapable):
    class TTSemaphore:
        def __init__(self):
            self.value = 0
            self.max = 0

    class TTMutex:
        def __init__(self):
            self.held_by = None

    OPCODE_TO_HANDLER = {
        "SEMINIT": "handle_seminit",
        "STALLWAIT": "handle_stallwait",
        "SEMWAIT": "handle_semwait",
        "SEMPOST": "handle_sempost",
        "ATGETM": "handle_atgetm",
        "ATRELM": "handle_atrelm",
    }

    def __init__(self, backend):
        super().__init__(backend, TensixSyncUnit.OPCODE_TO_HANDLER, "Sync")
        self.semaphores = [TensixSyncUnit.TTSemaphore()] * 8
        # 7 mutexes, but index 1 is ignored
        self.mutexes = [TensixSyncUnit.TTMutex()] * 8
        self.blocked_mutex = []

    def clock_tick(self, cycle_num):
        super().clock_tick(cycle_num)
        to_remove = []
        for idx, (issue_thread, index) in enumerate(self.blocked_mutex):
            if (
                self.mutexes[index].held_by is None
                or self.mutexes[index].held_by == issue_thread
            ):
                self.mutexes[index].held_by = issue_thread
                self.backend.getFrontendThread(
                    issue_thread
                ).wait_gate.informMutexAcquired()
                to_remove.append(idx)
        if len(to_remove) > 1:
            to_remove.reverse()
            for idx in to_remove:
                del self.blocked_mutex[idx]

    def handle_atrelm(self, instruction_info, issue_thread, instr_args):
        index = instr_args["mutex_index"]
        self.mutexes[index].held_by = None  # TODO: wait here

    def handle_atgetm(self, instruction_info, issue_thread, instr_args):
        index = instr_args["mutex_index"]
        if (
            self.mutexes[index].held_by is not None
            and self.mutexes[index].held_by != issue_thread
        ):
            self.blocked_mutex.append((issue_thread, index))
        else:
            self.mutexes[index].held_by = issue_thread
            self.backend.getFrontendThread(issue_thread).wait_gate.informMutexAcquired()

    def handle_sempost(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        for i in range(8):
            if get_nth_bit(sem_sel, i) and self.semaphores[i].value < 15:
                self.semaphores[i].value += 1

    def handle_seminit(self, instruction_info, issue_thread, instr_args):
        sem_sel = instr_args["sem_sel"]
        new_value = instr_args["init_value"]
        max_value = instr_args["max_value"]

        for i in range(8):
            if get_nth_bit(sem_sel, i):
                self.semaphores[i].value = new_value
                self.semaphores[i].max = max_value

    def handle_stallwait(self, instruction_info, issue_thread, instr_args):
        pass

    def handle_semwait(self, instruction_info, issue_thread, instr_args):
        pass

    def read(self, addr, size):
        # Accesses semaphore[i].value, where each
        # entry is 32 bit
        idx = int(addr / 4)
        assert idx < 8
        return conv_to_bytes(self.semaphores[idx].value)

    def write(self, addr, value, size=None):
        """
        This is taken from the functional model code at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SyncUnit.md#semaphores
        """
        idx = int(addr / 4)
        assert idx < 8
        if conv_to_uint32(value) & 1:
            # This is like a SEMGET instruction
            if self.semaphores[idx].value > 0:
                self.semaphores[idx].value = -1
        else:
            # This is like a SEMPOST instruction
            if self.semaphores[idx].value < 15:
                self.semaphores[idx].value += 1

    def getSize(self):
        return 0xFFDF


class MiscellaneousUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {
        "SETADCXY": "handle_setadcxy",
        "SETADCZW": "handle_setadczw",
        "SETADCXX": "handle_setadcxx",
        "DMANOP": "handle_dmanop",
        "SETADC": "handle_setadc",
    }

    def __init__(self, backend):
        super().__init__(backend, MiscellaneousUnit.OPCODE_TO_HANDLER, "Misc")

    def handle_dmanop(self, instruction_info, issue_thread, instr_args):
        # This is a nop (but in documentation says for the scalar unit, but it is
        # directed to the misc unit for some reason)
        pass

    def handle_setadcxx(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, X0Val, X1Val):
            adc_channel.Channel[0].X = X0Val
            adc_channel.Channel[0].X_Cr = X0Val
            adc_channel.Channel[1].X = X1Val
            adc_channel.Channel[1].X_Cr = X1Val

        X0Val = instr_args["x_start"]
        X1Val = instr_args["x_end2"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(self.backend.getADC(issue_thread).Unpacker[0], X0Val, X1Val)
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(self.backend.getADC(issue_thread).Unpacker[1], X0Val, X1Val)
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(self.backend.getADC(issue_thread).Packers, X0Val, X1Val)

    def handle_setadc(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, xyzw, newValue):
            match xyzw:
                case 0:
                    adc_channel.X = newValue
                    adc_channel.X_Cr = newValue
                case 1:
                    adc_channel.Y = newValue
                    adc_channel.Y_Cr = newValue
                case 2:
                    adc_channel.Z = newValue
                    adc_channel.Z_Cr = newValue
                case 3:
                    adc_channel.W = newValue
                    adc_channel.W_Cr = newValue

        newValue = instr_args["Value"]
        threadOverride = newValue >> 16
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        channelIndex = instr_args["ChannelIndex"]
        xyzw = instr_args["DimensionIndex"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0].Channel[channelIndex],
                xyzw,
                newValue,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1].Channel[channelIndex],
                xyzw,
                newValue,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers.Channel[channelIndex],
                xyzw,
                newValue,
            )

    def handle_setadczw(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, enables, X0Val, Y0Val, X1Val, Y1Val):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].Z = X0Val
                adc_channel.Channel[0].W_Cr = X0Val

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].Z = Y0Val
                adc_channel.Channel[0].W_Cr = Y0Val

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].Z = X1Val
                adc_channel.Channel[1].W_Cr = X1Val

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].Z = Y1Val
                adc_channel.Channel[1].W_Cr = Y1Val

        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        X0Val = instr_args["Ch0_X"]
        Y0Val = instr_args["Ch0_Y"]
        X1Val = instr_args["Ch1_X"]
        Y1Val = extract_bits(instr_args["Ch1_Y"], 3, 0)
        enables = instr_args["BitMask"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers,
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )

    def handle_setadcxy(self, instruction_info, issue_thread, instr_args):
        def apply_to(adc_channel, enables, X0Val, Y0Val, X1Val, Y1Val):
            if get_nth_bit(enables, 0):
                adc_channel.Channel[0].X = X0Val
                adc_channel.Channel[0].X_Cr = X0Val

            if get_nth_bit(enables, 1):
                adc_channel.Channel[0].Y = Y0Val
                adc_channel.Channel[0].Y_Cr = Y0Val

            if get_nth_bit(enables, 2):
                adc_channel.Channel[1].X = X1Val
                adc_channel.Channel[1].X_Cr = X1Val

            if get_nth_bit(enables, 3):
                adc_channel.Channel[1].Y = Y1Val
                adc_channel.Channel[1].Y_Cr = Y1Val

        threadOverride = extract_bits(instr_args["Ch1_Y"], 2, 3)
        whichThread = issue_thread if threadOverride == 0 else threadOverride - 1

        X0Val = instr_args["Ch0_X"]
        Y0Val = instr_args["Ch0_Y"]
        X1Val = instr_args["Ch1_X"]
        Y1Val = extract_bits(instr_args["Ch1_Y"], 3, 0)
        enables = instr_args["BitMask"]

        if get_nth_bit(instr_args["CntSetMask"], 0):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[0],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 1):
            apply_to(
                self.backend.getADC(whichThread).Unpacker[1],
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )
        if get_nth_bit(instr_args["CntSetMask"], 2):
            apply_to(
                self.backend.getADC(whichThread).Packers,
                enables,
                X0Val,
                Y0Val,
                X1Val,
                Y1Val,
            )


class UnPackerUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {}

    def __init__(self, backend):
        super().__init__(backend, UnPackerUnit.OPCODE_TO_HANDLER, "Unpacker")


class PackerUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {"PACR": "handle_pacr"}

    def __init__(self, backend):
        super().__init__(backend, PackerUnit.OPCODE_TO_HANDLER, "Packer")

    def handle_pacr(self, instruction_info, issue_thread, instr_args):
        pass


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
        super().__init__(
            backend, TensixBackendConfigurationUnit.OPCODE_TO_HANDLER, "Config"
        )

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


class MoverUnit(TensixBackendUnit):
    class XMOV_DIRECTION(IntEnum):
        XMOV_L0_TO_L1 = 0
        XMOV_L1_TO_L0 = 1
        XMOV_L0_TO_L0 = 2
        XMOV_L1_TO_L1 = 3

    OPCODE_TO_HANDLER = {}

    TENSIX_CFG_BASE = 0xFFEF0000
    MEM_NCRISC_IRAM_BASE = 0xFFC00000

    def __init__(self, backend):
        super().__init__(backend, MoverUnit.OPCODE_TO_HANDLER, "Mover")

    def move(self, dst, src, count, mode):
        """
        This is based on the functional model description at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/Mover.md
        """
        assert self.backend.getAddressableMemory() is not None
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L0_TO_L1
        ):
            # In the "_TO_L1" modes, dst must be an address in L1.
            assert dst < 1024 * 1464
        else:
            if dst <= 0xFFFF:
                dst += MoverUnit.TENSIX_CFG_BASE
            elif 0x40000 <= dst and dst <= 0x4FFFF:
                dst = (dst - 0x40000) + MoverUnit.MEM_NCRISC_IRAM_BASE
            else:
                dst = None  # Operation still happens, but the writes get discarded.

            if (dst & 0xFFFF) + count > 0x10000:
                raise NotImplementedError(
                    "Can not access more than one region at a time"
                )

        # Perform the operation.
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L0
        ):
            # In the "L1_TO_" modes, a memcpy is done, and src must be an address in L1.
            if src >= (1024 * 1464):
                raise NotImplementedError("")
            # print(f"Write to {hex(dst)} from {hex(src)} elements {hex(count)}")
            self.backend.getAddressableMemory().write(
                dst, self.backend.getAddressableMemory().read(src, count)
            )
        else:
            # In the "L0_TO_" modes, a memset is done.
            zero_val = conv_to_bytes(0, count)
            self.backend.getAddressableMemory().write(dst, zero_val)
