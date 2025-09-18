from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit
from tt_sim.pe.tensix.registers import LReg
from tt_sim.util.bits import get_bits, get_nth_bit


class VectorUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {
        "SFPENCC": "handle_sfpencc",
        "SFPLOADI": "handle_sfploadi",
        "SFPCONFIG": "handle_sfpconfig",
    }

    MOD1_IMM16_IS_VALUE = 1
    MOD1_BITWISE_OR = 2
    MOD1_BITWISE_AND = 4
    MOD1_BITWISE_XOR = 6
    MOD1_IMM16_IS_LANE_MASK = 8

    SFPLOADI_MOD0_FLOATB = 0  # Immediate is BF16
    SFPLOADI_MOD0_FLOATA = 1  # Immediate is FP16 (ish)
    SFPLOADI_MOD0_USHORT = 2  # Immediate is UINT16
    SFPLOADI_MOD0_SHORT = 4  # Immediate is INT16
    SFPLOADI_MOD0_UPPER = 8  # Immediate overwrites upper 16 bits
    SFPLOADI_MOD0_LOWER = 10  # Immediate overwrites lower 16 bits

    SFPENCC_MOD1_EC = 1  # Invert UseLaneFlagsForLaneEnable
    SFPENCC_MOD1_EI = 2  # Set UseLaneFlagsForLaneEnable from SFPENCC_IMM2_E
    SFPENCC_MOD1_RI = 8  # Set LaneFlags from SFPENCC_IMM2_R
    SFPENCC_IMM2_E = 1  # Immediate bit for UseLaneFlagsForLaneEnable
    SFPENCC_IMM2_R = 2  # Immediate bit for LaneFlags

    class LoadMacroConfig:
        def __init__(self):
            self.storeMod0 = 0
            self.usesLoadMod0ForStore = False
            self.unitDelayKind = 0
            self.sequence = [0] * 4
            self.instructionTemplate = [0] * 4

        def misc(self, value, mode=0):
            self.storeMod0 = get_bits(value, 0, 3)
            self.usesLoadMod0ForStore = get_bits(value, 4, 7)
            self.unitDelayKind = get_bits(value, 8, 11)
            if mode == VectorUnit.MOD1_BITWISE_OR:
                self.storeMod0 |= self.storeMod0
                self.usesLoadMod0ForStore |= self.usesLoadMod0ForStore
                self.unitDelayKind |= self.unitDelayKind
            elif mode == VectorUnit.MOD1_BITWISE_AND:
                self.storeMod0 &= self.storeMod0
                self.usesLoadMod0ForStore &= self.usesLoadMod0ForStore
                self.unitDelayKind &= self.unitDelayKind
            elif mode == VectorUnit.MOD1_BITWISE_XOR:
                self.storeMod0 ^= self.storeMod0
                self.usesLoadMod0ForStore ^= self.usesLoadMod0ForStore
                self.unitDelayKind ^= self.unitDelayKind

    def __init__(self, backend):
        self.lregs = [LReg() for i in range(17)]
        self.lregs[8].setReadOnly(0.8373)
        self.lregs[9].setReadOnly(0)
        self.lregs[10].setReadOnly(1.0)
        for i in range(32):
            self.lregs[15][i] = i * 2
        self.lregs[15].setReadOnly()
        self.laneEnabled = [True] * 32
        self.laneFlags = [False] * 32
        self.useLaneFlagsForLaneEnable = [False] * 32
        self.flagStack = [0] * 32
        self.laneConfig = [0] * 32
        self.loadMacroConfig = [VectorUnit.LoadMacroConfig() for i in range(32)]
        super().__init__(backend, VectorUnit.OPCODE_TO_HANDLER, "Vector")

    def handle_sfpencc(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        imm2 = instr_args["imm12_math"]

        for lane in range(32):
            if lane < 12:  # TODO: || LaneConfig.DISABLE_BACKDOOR_LOAD
                if mod1 & VectorUnit.SFPENCC_MOD1_EI:
                    self.useLaneFlagsForLaneEnable[lane] = (
                        imm2 & VectorUnit.SFPENCC_IMM2_E
                    ) != 0
                elif mod1 & VectorUnit.SFPENCC_MOD1_EC:
                    self.useLaneFlagsForLaneEnable[
                        lane
                    ] = not self.useLaneFlagsForLaneEnable[lane]
                else:
                    # UseLaneFlagsForLaneEnable left as-is.
                    pass

                if mod1 & VectorUnit.SFPENCC_MOD1_RI:
                    self.laneFlags[lane] = (imm2 & VectorUnit.SFPENCC_IMM2_R) != 0
                else:
                    self.laneFlags[lane] = True

    def handle_sfploadi(self, instruction_info, issue_thread, instr_args):
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]
        imm16 = instr_args["imm16"]

        assert vd < 8
        for lane in range(32):
            if self.laneEnabled:
                match mod0:
                    case VectorUnit.SFPLOADI_MOD0_FLOATB:
                        self.lregs[vd][lane] = self.BF16toFP32(imm16)
                    case VectorUnit.SFPLOADI_MOD0_FLOATA:
                        self.lregs[vd][lane] = self.FP16toFP32(imm16)
                    case VectorUnit.SFPLOADI_MOD0_USHORT:
                        self.lregs[vd][lane] = imm16
                    case VectorUnit.SFPLOADI_MOD0_SHORT:
                        self.lregs[vd][lane] = imm16
                    case VectorUnit.SFPLOADI_MOD0_UPPER:
                        self.lregs[vd][lane] = (imm16 << 16) | (
                            self.lregs[vd][lane] & 0x0000FFFF
                        )
                    case VectorUnit.SFPLOADI_MOD0_LOWER:
                        self.lregs[vd][lane] = (
                            self.lregs[vd][lane] & 0xFFFF0000
                        ) | imm16
                    case _:
                        raise ValueError()

    def BF16toFP32(self, val):
        return val << 16

    def FP16toFP32(self, val):
        sign = val >> 15
        exp = (val >> 10) & 0x1F
        man = val & 0x3FF

        exp += 112  # Rebias 5b exponent to 8b
        return (sign << 31) | (exp << 23) | (man << 13)

    def handle_sfpconfig(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["config_dest"]
        imm16 = instr_args["imm16_math"]
        for lane in range(32):
            if mod1 & VectorUnit.MOD1_IMM16_IS_LANE_MASK:
                if not get_nth_bit(imm16, (lane & 7) * 2):
                    continue

            if self.useLaneFlagsForLaneEnable[lane & 7]:
                if not self.laneFlags[lane & 7]:
                    continue

            match vd:
                case 0 | 1 | 2 | 3:
                    # Write to LoadMacroConfig::InstructionTemplate.
                    self.loadMacroConfig[lane].instructionTemplate[vd] = self.lregs[0][
                        lane & 7
                    ]
                case 4 | 5 | 6 | 7:
                    # Write to LoadMacroConfig::Sequence
                    value = (
                        imm16
                        if (mod1 & VectorUnit.MOD1_IMM16_IS_VALUE)
                        else self.lregs[0][lane & 7]
                    )
                    self.loadMacroConfig[lane].sequence[vd - 4] = value
                case 8:
                    # Write or manipulate LoadMacroConfig::Misc
                    value = (
                        imm16
                        if (mod1 & VectorUnit.MOD1_IMM16_IS_VALUE)
                        else self.lregs[0][lane & 7]
                    )
                    self.loadMacroConfig[lane].misc(value, mod1 & 6)
                case 9 | 10:
                    # Does nothing
                    pass
                case 11 | 12 | 13 | 14:
                    if mod1 & VectorUnit.MOD1_IMM16_IS_VALUE:
                        match vd:
                            case 11:
                                value = -1.0
                            case 12:
                                value = -1 / 65536.0
                            case 13:
                                value = -0.67487759
                            case 14:
                                value = -0.34484843
                    else:
                        value = self.lregs[0][lane & 7]
                    self.lregs[vd][lane] = value
                case 15:
                    # Write or manipulate LaneConfig
                    original = self.laneConfig[lane]
                    value = (
                        imm16
                        if (mod1 & VectorUnit.MOD1_IMM16_IS_VALUE)
                        else self.lregs[0][lane & 7]
                    )
                    match mod1 & 6:
                        case 0:
                            self.laneConfig[lane] = value
                        case VectorUnit.MOD1_BITWISE_OR:
                            self.laneConfig[lane] |= value
                        case VectorUnit.MOD1_BITWISE_AND:
                            self.laneConfig[lane] &= value
                        case VectorUnit.MOD1_BITWISE_XOR:
                            self.laneConfig[lane] ^= value

                    if mod1 & VectorUnit.MOD1_IMM16_IS_VALUE:
                        self.laneConfig[lane] |= original & ~0xFFFF
