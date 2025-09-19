from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.tensix.backends.config import TensixBackendConfigurationUnit
from tt_sim.pe.tensix.backends.matrix import MatrixUnit
from tt_sim.pe.tensix.backends.misc import MiscellaneousUnit
from tt_sim.pe.tensix.backends.mover import MoverUnit
from tt_sim.pe.tensix.backends.packer import PackerUnit
from tt_sim.pe.tensix.backends.sync import TensixSyncUnit
from tt_sim.pe.tensix.backends.thcon import ScalarUnit
from tt_sim.pe.tensix.backends.unpacker import UnPackerUnit
from tt_sim.pe.tensix.backends.vector import VectorUnit
from tt_sim.pe.tensix.registers import DstRegister, SrcRegister
from tt_sim.pe.tensix.util import (
    DiagnosticsSettings,
    TensixConfigurationConstants,
    TensixInstructionDecoder,
)
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TensixBackend:
    def __init__(self, diags_settings=None):
        self.gpr = TensixGPR()
        self.mover_unit = MoverUnit(self)
        self.sync_unit = TensixSyncUnit(self)
        self.matrix_unit = MatrixUnit(self)
        self.scalar_unit = ScalarUnit(self, self.gpr)
        self.vector_unit = VectorUnit(self)
        self.unpacker_units = [UnPackerUnit(self, i) for i in range(2)]
        self.packer_unit = PackerUnit(self)
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
            "PACK": self.packer_unit,
        }
        self.rwc = [RWC(self) for i in range(3)]
        self.adc = [ADCThread() for i in range(3)]
        self.addressable_memory = None
        self.diags_settings = (
            diags_settings if diags_settings is not None else DiagnosticsSettings()
        )

    def getDiagnosticSettings(self):
        return self.diags_settings

    def getRWC(self, thread_id):
        assert thread_id <= 2
        return self.rwc[thread_id]

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

    def getMatrixUnit(self):
        return self.matrix_unit

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
            self.packer_unit,
        ]
        unit_clocks += self.unpacker_units
        return unit_clocks

    def getThreadConfigValue(self, issue_thread, key):
        addr_idx = TensixConfigurationConstants.get_addr32(key)
        val = self.getConfigUnit().get_threadConfig_entry(issue_thread, addr_idx)
        return TensixConfigurationConstants.parse_raw_config_value(val, key)

    def getConfigValue(self, state_id, key, words=1):
        addr_idx = TensixConfigurationConstants.get_addr32(key)
        if words == 1:
            val = self.getConfigUnit().get_config_entry(state_id, addr_idx)
            return TensixConfigurationConstants.parse_raw_config_value(val, key)
        else:
            results = []
            for word in range(words):
                val = self.getConfigUnit().get_config_entry(
                    state_id, addr_idx + (word * 4)
                )
                results.append(
                    TensixConfigurationConstants.parse_raw_config_value(val, key)
                )
            return results

    def hasInflightInstructionsFromThread(self, from_thread):
        for unit in self.backend_units.values():
            if unit.hasInflightInstructionsFromThread(from_thread):
                return True
        for unpacker in self.unpacker_units:
            if unpacker.hasInflightInstructionsFromThread(from_thread):
                return True
        if self.packer_unit.hasInflightInstructionsFromThread(from_thread):
            return True
        return False

    def issueInstruction(self, instruction, from_thread):
        instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
        tgt_backend_unit = instruction_info["ex_resource"]
        if tgt_backend_unit != "NONE":
            if tgt_backend_unit == "UNPACK":
                which_unpacker = get_nth_bit(instruction, 23)
                return self.unpacker_units[which_unpacker].issueInstruction(
                    instruction, from_thread
                )
            else:
                assert tgt_backend_unit in self.backend_units
                return self.backend_units[tgt_backend_unit].issueInstruction(
                    instruction, from_thread
                )
        else:
            # NOP is handled here, just ignore
            return True


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
            self.Channel = [
                ADCThread.ADCUnit.ADCChannel(),
                ADCThread.ADCUnit.ADCChannel(),
            ]

    def __init__(self):
        self.Unpacker = [ADCThread.ADCUnit(), ADCThread.ADCUnit()]
        self.Packers = ADCThread.ADCUnit()


class RWC:
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
