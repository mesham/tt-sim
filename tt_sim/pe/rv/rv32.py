from tt_sim.memory.memory import VisibleMemory
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.register.register import Register, RegisterAccessMode
from tt_sim.pe.register.register_file import RegisterFile
from tt_sim.pe.rv.isa.i_isa import RV_I_ISA
from tt_sim.pe.rv.isa.m_isa import RV_M_ISA
from tt_sim.pe.rv.isa.tt_isa import RV_TT_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

REGISTER_NAME_MAPPING = {
    "x0": 0,
    "x1": 1,
    "x2": 2,
    "x3": 3,
    "x4": 4,
    "x5": 5,
    "x6": 6,
    "x7": 7,
    "x8": 8,
    "x9": 9,
    "x10": 10,
    "x11": 11,
    "x12": 12,
    "x13": 13,
    "x14": 14,
    "x15": 15,
    "x16": 16,
    "x17": 17,
    "x18": 18,
    "x19": 19,
    "x20": 20,
    "x21": 21,
    "x22": 22,
    "x23": 23,
    "x24": 24,
    "x25": 25,
    "x26": 26,
    "x27": 27,
    "x28": 28,
    "x29": 29,
    "x30": 30,
    "x31": 31,
    "zero": 0,
    "ra": 1,
    "sp": 2,
    "gp": 3,
    "tp": 4,
    "t0": 5,
    "t1": 6,
    "t2": 7,
    "s0": 8,
    "s1": 9,
    "a0": 10,
    "a1": 11,
    "a2": 12,
    "a3": 13,
    "a4": 14,
    "a5": 15,
    "a6": 16,
    "a7": 17,
    "s2": 18,
    "s3": 19,
    "s4": 20,
    "s5": 21,
    "s6": 22,
    "s7": 23,
    "s8": 24,
    "s9": 25,
    "s10": 26,
    "s11": 27,
    "t3": 28,
    "t4": 29,
    "t5": 30,
    "t6": 31,
    "fp": 8,
    "pc": 32,
    "nextpc": 33,
}


class RV32I(ProcessingElement):
    def __init__(
        self,
        start_address,
        device_memory=None,
        pe_memory=None,
        extensions=[],
        unknown_instr_is_error=False,
    ):
        self.isas = [RV_I_ISA] + extensions
        self.device_memory = device_memory
        self.pe_memory = pe_memory
        self.start_address = start_address
        self.active = False
        self.unknown_instructions = 0

        # 32 registers plus the PC
        registers = []
        # Register 0 is read only and hardcoded to zero
        registers.append(Register(4, conv_to_bytes(0), RegisterAccessMode.R))
        for i in range(33):
            registers.append(Register(4))

        self.register_file = RegisterFile(registers, REGISTER_NAME_MAPPING)
        self.unknown_instr_is_error = unknown_instr_is_error

        # Now determine the visible memory for the core, this will either be a combination of
        # global device memory and local PE memory, or one of these if the other is not supplied
        if self.device_memory is None and self.pe_memory is None:
            raise Exception(
                "An RV32 core must have access to either (or both) device and/or pe memory"
            )
        elif self.device_memory is not None and self.pe_memory is None:
            self.visible_memory = self.device_memory
        elif self.device_memory is None and self.pe_memory is not None:
            self.visible_memory = self.pe_memory
        else:
            self.visible_memory = VisibleMemory.merge(
                self.device_memory, self.pe_memory
            )

    def clock_tick(self):
        if not self.active:
            return

        pc = self.register_file["pc"]
        nextpc = self.register_file["nextpc"]
        pc_val = conv_to_uint32(pc.read())
        nextpc.write(conv_to_bytes(pc_val + 4))

        actioned = False
        for isa in self.isas:
            actioned = isa.run(self.register_file, self.visible_memory)
            if actioned:
                break

        if not actioned:
            self.unknown_instructions += 1
            if self.unknown_instr_is_error:
                raise Exception("Unknown instruction")

        pc.write(nextpc.read())

    def reset(self):
        self.stop()
        self.start()

    def getRegisterFile(self):
        return self.register_file

    def start(self):
        pc = self.register_file["pc"]
        pc.write(conv_to_bytes(self.start_address))

        self.unknown_instructions = 0

        self.active = True

    def stop(self):
        self.active = False


class RV32IM(RV32I):
    def __init__(
        self,
        start_address,
        device_memory=None,
        pe_memory=None,
        extensions=[],
        unknown_instr_is_error=False,
    ):
        if RV_M_ISA not in extensions:
            extensions.append(RV_M_ISA)
        super().__init__(
            start_address, device_memory, pe_memory, extensions, unknown_instr_is_error
        )


class RV32IM_TT(RV32IM):
    def __init__(
        self,
        start_address,
        device_memory=None,
        pe_memory=None,
        extensions=[],
        unknown_instr_is_error=False,
    ):
        if RV_TT_ISA not in extensions:
            extensions.append(RV_TT_ISA)
        super().__init__(
            start_address, device_memory, pe_memory, extensions, unknown_instr_is_error
        )
