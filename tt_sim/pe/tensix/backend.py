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
    TensixConfigurationConstants,
    TensixInstructionDecoder,
)
from tt_sim.util.bits import get_nth_bit
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

#: Blackhole-only Tensix instructions that tt-sim decodes but does not model:
#: name -> what the hardware does, for the error message. The vendor reference
#: simulator (ttsim, ``data/bh/tensix_isa.json``) marks all four "unsupported"
#: and raises on them, so there is no behaviour to port and nothing to validate
#: an implementation against. Rather than silently no-op them (RESOURCEDECL in
#: particular has ``ex_resource: NONE``, which would otherwise be ignored like a
#: NOP) they are rejected loudly here — the same choice as the baby-RISC-V
#: guards in ``tt_sim/pe/rv/isa/guard_isa.py``. Decoding them at least means the
#: fields are named (see ``tensix_instructions.yaml``) for whoever implements one.
UNMODELLED_BLACKHOLE_INSTRUCTIONS = {
    "MOVDBGB2D": "moves SrcB into Dst in debug mode, bypassing the ready signals",
    "RESOURCEDECL": "declares the resources a class of Tensix instructions uses, "
    "for the Tensix-TRISC sync mechanism",
    "STREAMWAIT": "stalls resources until a NoC stream reaches a phase / message count",
    "STREAMWRCFG": "copies a NoC stream register into a config register",
}


class TensixBackend:
    """
    Tensix backend, containing the units that make this up, see
    https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor
    """

    def __init__(
        self,
        diags_settings,
        cfg_state_size=None,
        thd_state_size=None,
        blackhole=False,
    ):
        self.blackhole = blackhole
        self.gpr = TensixGPR()
        self.mover_unit = MoverUnit(self)
        self.sync_unit = TensixSyncUnit(self)
        self.matrix_unit = MatrixUnit(self)
        self.scalar_unit = ScalarUnit(self, self.gpr)
        self.vector_unit = VectorUnit(self)
        self.unpacker_units = [UnPackerUnit(self, i) for i in range(2)]
        self.packer_unit = PackerUnit(self)
        self.misc_unit = MiscellaneousUnit(self)
        self.config_unit = TensixBackendConfigurationUnit(
            self, self.gpr, cfg_state_size, thd_state_size, blackhole
        )
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
        self.diags_settings = diags_settings
        # Why the most recent ``issueInstruction`` refusal happened, written by
        # ``TensixBackendUnit._refuse`` and read back by the wait gate, which is
        # the caller that holds the cycle number and the decoded opcode needed
        # to publish a StallEvent. Only meaningful immediately after a call that
        # returned False; the gate reads it in that same statement.
        # ``last_refusal_blocked_on`` is empty unless the refusing unit wants to
        # point at a *different* unit as the cause -- see ``_refuse``.
        self.last_refusal_reason = ""
        self.last_refusal_blocked_on = ""
        # The documented *whole-thread* issue interlock, as a deadline per
        # Tensix thread: while ``cycle_num < thread_issue_block[i]``, nothing
        # from thread ``i`` may leave its Wait Gate, whichever unit it is bound
        # for. That is what the existing per-unit issue refusal cannot express
        # -- it can only refuse the unit an instruction was offered to, and a
        # held thread's next instruction is usually offered somewhere else.
        #
        # ``0`` means "not blocked", which is what every entry reads for a
        # whole run with ``TT_SIM_COST_MODEL`` unset: the deadline is armed
        # from a *modelled occupancy*, and nothing arms one with the model off.
        # The wait gate therefore pays one list index and one integer
        # comparison per tick in the default configuration.
        #
        # One unit arms it today, from ``TensixCoprocessor/ScalarUnit.md``:
        # "once a thread has started executing a Scalar Unit instruction, it
        # cannot start executing its next instruction until the Scalar Unit
        # instruction completes, regardless of which unit that next instruction
        # executes in." The unpackers publish an identically-shaped interlock
        # for an ``UNPACR``'s address phase and are deliberately NOT wired to
        # this: see the ``UNPACR`` entry's note in the Tensix cost table for
        # the mechanism that blocks it (tt-sim flips Src ``AllowedClient`` at
        # retire, hardware at the end of the transfer).
        #
        # ``thread_issue_block_unit`` carries the ``ex_resource`` name of
        # whichever unit imposed the live deadline, so the StallEvent the gate
        # publishes names the unit responsible rather than a bare reason.
        self.thread_issue_block = [0, 0, 0]
        self.thread_issue_block_unit = ["", "", ""]

    def block_thread_issue(self, thread_id, until_cycle, unit_name):
        """Hold thread ``thread_id`` at its Wait Gate until ``until_cycle``.

        Called by the unit whose documentation states the interlock, from the
        same anchor cycle its occupancy is charged from, so the block is
        exactly as long as the charge and inherits its "low end of the bound"
        floor. A later, longer deadline extends the block; a shorter one is
        ignored, because the thread is already held past it.
        """
        if until_cycle > self.thread_issue_block[thread_id]:
            self.thread_issue_block[thread_id] = until_cycle
            self.thread_issue_block_unit[thread_id] = unit_name

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
        # A thread-config register that doesn't exist on this architecture (e.g.
        # Wormhole's ADDR_MOD_SET_Base, which Blackhole replaces with the
        # ADDR_MOD_AB2_* register set) reads as unset — 0.
        if not TensixConfigurationConstants.exists(key):
            return 0
        addr_idx = TensixConfigurationConstants.get_addr32(key)
        val = self.getConfigUnit().get_threadConfig_entry(issue_thread, addr_idx)
        return TensixConfigurationConstants.parse_raw_config_value(val, key)

    def getConfigValue(self, state_id, key, words=1):
        addr_idx = TensixConfigurationConstants.get_addr32(key)
        if words == 1:
            val = self.getConfigUnit().get_config_entry(state_id, addr_idx)
            return TensixConfigurationConstants.parse_raw_config_value(val, key)
        else:
            # A multi-word register occupies *consecutive* config words: the
            # config array is indexed in 32-bit words, so word n of the register
            # named by ``key`` is at ``addr_idx + n``, not ``addr_idx + 4 * n``.
            # The only such register is the unpacker's tile descriptor, whose
            # fields the ISA docs place at bits 16-31 (XDim), 32-39 (YDim),
            # 48-55 (ZDim) and 64-71 (WDim) of one contiguous bit string -- i.e.
            # words 0, 1, 1 and 2. Striding by four read THCON_SEC0_REG1 / REG2 /
            # REG3 in place of descriptor words 1 / 2 / 3, which happened to hold
            # the right ZDim and a YDim of 0 (read as 1) for an ordinary tile,
            # and so went unnoticed until ``llk_unpack_untilize`` set YDim to 16.
            return [
                TensixConfigurationConstants.parse_raw_config_value(
                    self.getConfigUnit().get_config_entry(state_id, addr_idx + word),
                    key,
                )
                for word in range(words)
            ]

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
        instruction_name = instruction_info["name"]
        if instruction_name in UNMODELLED_BLACKHOLE_INSTRUCTIONS:
            raise NotImplementedError(
                f"Tensix instruction {instruction_name} ({hex(instruction)}) from thread "
                f"{from_thread} is not modelled in tt-sim: it "
                f"{UNMODELLED_BLACKHOLE_INSTRUCTIONS[instruction_name]}, and the vendor "
                f"reference simulator does not implement it either, so its behaviour "
                f"cannot be ported. Implement it here (tt_sim/pe/tensix/backend.py) when "
                f"a kernel needs it."
            )
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
            # Per ISA RWCs.md, ExtraAddrModBit is uint1_t — it wraps modulo 2.
            self.ExtraAddrModBit = (self.ExtraAddrModBit + 1) & 1

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
        self.registers = [[0] * 64 for _ in range(3)]
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
