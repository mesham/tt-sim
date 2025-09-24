import importlib.resources as resources
from copy import copy

import yaml

from tt_sim.util.bits import extract_bits, get_bits


class DiagnosticsSettings:
    def __init__(
        self,
        issued_instructions=False,
        configurations_set=False,
        unpacking=False,
        packing=False,
        fpu_calculations=False,
        sfpu_calculations=False,
        thcon=False,
    ):
        self.issued_instructions = issued_instructions
        self.configurations_set = configurations_set
        self.unpacking = unpacking
        self.packing = packing
        self.fpu_calculations = fpu_calculations
        self.sfpu_calculations = sfpu_calculations
        self.thcon = thcon

    def reportThCon(self):
        return self.thcon

    def reportFPUCalculations(self):
        return self.fpu_calculations

    def reportSFPUCalculations(self):
        return self.sfpu_calculations

    def reportUnpacking(self):
        return self.unpacking

    def reportPacking(self):
        return self.packing

    def reportIssuedInstructions(self):
        return self.issued_instructions

    def reportConfigurationSet(self):
        return self.configurations_set


class TensixConfigurationConstants:
    @classmethod
    def init(cls):
        if not hasattr(cls, "config_constants"):
            with (
                resources.files("tt_sim.pe.tensix")
                .joinpath("tensix_backend_cfg.yaml")
                .open("r") as f
            ):
                cls.config_constants = yaml.safe_load(f)
                cls.ids = {}
                for k in cls.config_constants.keys():
                    addr32 = cls.get_addr32(k)
                    cls.ids[addr32] = k

    @classmethod
    def get_name(cls, id):
        cls.init()
        if id in cls.ids:
            return cls.ids[id]
        else:
            return "NONE"

    @classmethod
    def get_addr32(cls, key):
        cls.init()
        if key not in cls.config_constants:
            raise IndexError(f"'{key}' not in constants")
        return cls.config_constants[key]["ADDR32"]

    @classmethod
    def get_shamt(cls, key):
        cls.init()
        if key not in cls.config_constants:
            raise IndexError(f"'{key}' not in constants")
        return cls.config_constants[key]["SHAMT"]

    @classmethod
    def get_mask(cls, key):
        cls.init()
        if key not in cls.config_constants:
            raise IndexError(f"'{key}' not in constants")
        return cls.config_constants[key]["MASK"]

    @classmethod
    def parse_raw_config_value(cls, value, key):
        cls.init()
        mask = cls.get_mask(key)
        shamt = cls.get_shamt(key)
        return cls.tensix_be_config_parse_value(value, shamt, mask)

    @classmethod
    def tensix_be_config_parse_value(cls, value, shamt, mask):
        return (value & mask) >> shamt


class TensixInstructionDecoder:
    @classmethod
    def init(cls):
        if not hasattr(cls, "tensix_instructions") or not hasattr(cls, "opcodes"):
            with (
                resources.files("tt_sim.pe.tensix")
                .joinpath("tensix_instructions.yaml")
                .open("r") as f
            ):
                cls.tensix_instructions = yaml.safe_load(f)

            cls.opcodes = cls._generate_tensix_instructions_by_opcode()

    @classmethod
    def _generate_tensix_instructions_by_opcode(cls):
        by_opcode = {}
        for k, instruction in cls.tensix_instructions.items():
            by_opcode[instruction["op_binary"]] = instruction
            by_opcode[instruction["op_binary"]]["name"] = k
        return by_opcode

    @classmethod
    def isInstructionRecognised(cls, instruction):
        cls.init()
        opcode = extract_bits(instruction, 8, 24)
        return opcode in cls.opcodes

    @classmethod
    def getInstructionInfo(cls, instruction):
        cls.init()
        opcode = extract_bits(instruction, 8, 24)
        assert opcode in cls.opcodes
        # Create a copy of the top level object here as will insert the
        # instruction arguments. If we don't create a copy then the underlying
        # object is modified and the next same instruction would get those
        # arguments. This matters if an instruction is blocked. Only do a shallow
        # copy as only going to change the top level object (add instr_args)
        instruction_info = copy(cls.opcodes[opcode])
        instr_args = {}
        if "arguments" in instruction_info and isinstance(
            instruction_info["arguments"], list
        ):
            arg_ends = []  # end of each argument (inclusive)
            for arg in instruction_info["arguments"][1:]:
                arg_ends.append(arg["start_bit"] - 1)
            arg_ends.append(23)  # opcode is from 24 onwards

            for idx, arg in enumerate(instruction_info["arguments"]):
                instr_args[arg["name"]] = get_bits(
                    instruction, arg["start_bit"], arg_ends[idx]
                )

        instruction_info["instr_args"] = instr_args

        return instruction_info


class DataFormatConversions:
    """
    These are data format conversion utilities used throughout the Tensix coprocessor
    backend, typically for converting to and from dst, srcA and srcB data storage
    formats.

    This is heavily based on the functional implementations at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor
    """

    # Conversion to Dst register format routines

    @classmethod
    def FP16ToDstFormatFP16(cls, x):
        # Rearrange fields from Sign,Exp,Man to Sign,Man,Exp as Dst holds
        # FP16 data in this rearranged form
        sign = x & 0x8000
        exp = x & 0x7C00
        man = x & 0x03FF
        return sign | (man << 5) | (exp >> 10)

    @classmethod
    def BF16ToDstFormatBF16(cls, x):
        # Rearrange fields from Sign,Exp,Man to Sign,Man,Exp as dst holds
        # BF16 data in this rearranged form
        sign = x & 0x8000
        exp = x & 0x7F80
        mantissa = x & 0x007F
        return sign | (mantissa << 8) | (exp >> 7)

    @classmethod
    def FP32ToDstFormatFP32(cls, x):
        # Rearrange fields from Sign,Exp,Man to Sign,ManHi,Exp,ManLo
        # because dst holds FP32 in this rearranged form
        hi = x >> 16
        lo = x & 0xFFFF
        hi = DataFormatConversions.BF16ToDstFormatBF16(hi)

        return (hi << 16) | lo

    @classmethod
    def FP32ToDstFormatFP16(cls, x):
        return DataFormatConversions.FP16ToDstFormatFP16(
            DataFormatConversions.FP32ToFP16(x)
        )

    @classmethod
    def FP32ToDstFormatBF16(cls, x):
        return DataFormatConversions.BF16ToDstFormatBF16(
            DataFormatConversions.FP32ToBF16(x)
        )

    # Conversion from dst register format routines

    @classmethod
    def FP16InDstToFP32(cls, x, enable_fp16a_inf=False):
        # Dst contained Sign,Man(10b),Exp(5b)
        sign = x >> 15
        man = (x >> 5) & 0x3FF
        exp = x & 0x1F

        if enable_fp16a_inf and exp == 0x1F and man == 0x3FF:
            # Remap largest possible value to IEEE754 infinity
            exp = 255
            man = 0
        elif exp != 0:
            # Rebias from 5b Exp to 8b Exp
            exp += 112

        return (sign << 31) | (exp << 23) | (man << 13)

    @classmethod
    def FP32InDstToFP32(cls, x):
        # dst contains sign,manhi(7b),exp(8b),manlo(16b)
        # rearrange this to sign,exp(8b),manhi(7b),manlo(16b)

        hi = x >> 16
        lo = x & 0xFFFF
        hi = DataFormatConversions.BF16InDstToBF16(hi)
        return (hi << 16) | lo

    @classmethod
    def BF16InDstToBF16(cls, x):
        # dst contained sign,man(7b),exp(8b),
        # rearrange this to sign,exp(8b),man(7b)

        sign = x & 0x8000
        exp = x & 0x00FF
        man = x & 0x7F00
        return sign | (exp << 7) | (man >> 8)

    @classmethod
    def FP16InDstToFP16(cls, x):
        # dst ontained sign,man(10b),exp(5b)
        # rearrange this to sign,exp(5b),man(10b)

        sign = x & 0x8000
        exp = x & 0x001F
        man = x & 0x7FE0
        return sign | (exp << 10) | (man >> 5)

    # Conversion to src register format routines

    @classmethod
    def TF32ToSrcTF32(cls, x):
        # Rearrange fields from Sign,Exp,Man to Sign,Man,Exp as Src holds
        # TF32 data in this rearranged form
        sign = x & 0x40000
        exp = x & 0x3FC00
        man = x & 0x003FF
        return sign | (man << 8) | (exp >> 10)

    @classmethod
    def BF16ToSrcBF16(cls, x):
        return DataFormatConversions.TF32ToSrcTF32(x << 3)

    @classmethod
    def FP16ToSrcFP16(cls, x):
        return DataFormatConversions.TF32ToSrcTF32(((x & 0x8000) << 3) | (x & 0x7FFF))

    @classmethod
    def Int8InSrcToInt8(cls, x):
        # src holds INT8 as Sign,Mag(10b),Zero(3b),Exp(5b)
        sign = x >> 18
        mag = (x >> 8) & 0x3FF
        return -mag if sign else mag

    # Conversion from src register format

    @classmethod
    def TF32InSrcToTF32(cls, x):
        # Rearrange fields from Sign,Man,Exp in src to Sign,Exp,Man
        sign = x & 0x40000
        exp = x & 0xFF
        man = x & 0x3FF00

        return sign | (exp << 10) | (man >> 8)

    @classmethod
    def BF16InSrcToBF16(cls, x):
        return DataFormatConversions.TF32InSrcToTF32(x) >> 3

    @classmethod
    def FP16InSrcToFP16(cls, x):
        tf32 = DataFormatConversions.TF32InSrcToTF32(x)
        return ((tf32 & 0x40000) >> 3) | (tf32 & 0x7FFF)

    @classmethod
    def FP16InSrcToFP32(cls, srcv):
        return DataFormatConversions.FP16ToFP32(
            DataFormatConversions.FP16InSrcToFP16(srcv)
        )

    @classmethod
    def BF16InSrcToFP32(cls, srcv):
        return DataFormatConversions.BF16InSrcToBF16(srcv) << 16

    @classmethod
    def TF32InSrcToFP32(cls, srcv):
        return DataFormatConversions.TF32InSrcToTF32(srcv) << 13

    # General number format and precision conversion routines

    @classmethod
    def FP16ToFP32(cls, x):
        # Widens the exponent field from 5b to 8b and rebiases

        sign = x >> 15
        exp = (x >> 10) & 0x1F
        man = x & 0x3FF

        exp += 112  # Rebias 5b exponent to 8b
        return (sign << 31) | (exp << 23) | (man << 13)

    @classmethod
    def FP32ToFP16(cls, x):
        sign = x >> 31
        exponent = ((x >> 23) & 0xFF) - 112
        mantissa = x & 0x7FFFFF

        if exponent <= 0:
            # Flush underflow and denormals to signed zero
            exponent = 0
            mantissa = 0
        elif exponent > 31:
            # Saturate on overflow
            # As dst does not handle infinite, the number is a huge one
            exponent = 31
            mantissa = 0x7FFFFF

        # Truncate toward zero
        mantissa >>= 13
        return (sign << 15) | (exponent << 10) | mantissa

    @classmethod
    def TF32ToSrcFormatTF32(cls, x):
        # Rearrange fields from Sign,Exp,Man to Sign,Man,Exp as Src holds
        # TF32 data in this rearranged form.
        sign = x & 0x40000
        exp = x & 0x3FC00
        mantissa = x & 0x003FF
        return sign | (mantissa << 8) | (exp >> 10)

    @classmethod
    def FP32ToBF16(cls, x):
        # Flush denormals to signed zero, then truncate toward zero
        sign = x & 0x80000000
        exp = x & 0x7F800000
        man = x & 0x007FFFFF
        if exp == 0:
            man = 0

        return (sign | exp | man) >> 16

    # Integer manipulation routines

    @classmethod
    def signMagToTwosComp(cls, x):
        # Convert from sign and 31-bit magnitude to 32-bit two's complement
        sign = x & 0x80000000
        mag = x & 0x7FFFFFFF
        -mag if sign else mag

    @classmethod
    def signMag8ToSignMag32(cls, x):
        # dst contained sign,ignored(3b),mag(7b),ignored(5b)
        sign = x >> 15
        mag = (x >> 5) & 0x7F
        return (sign << 31) | mag

    @classmethod
    def signMag11ToSignMag32(cls, x):
        # dst contained sign,mag(10b),ignored(5b)
        sign = x >> 15
        mag = (x >> 5) & 0x3FF
        return (sign << 31) | mag

    @classmethod
    def signMag16ToSignMag32(cls, x):
        # dst contained sign,mag(15b)
        sign = x >> 15
        mag = x & 0x7FFF
        return (sign << 31) | mag

    @classmethod
    def toSignMag(cls, x):
        # Convert from 32-bit two's complement to sign and 31-bit magnitude
        sign = x & 0x80000000
        mag = -x if sign else x
        return sign | (mag & 0x7FFFFFFF)

    # FP manipulation routines

    @classmethod
    def signMag11ToFP16(cls, x):
        # Converts to the type dst/srca/srcb refers to as int8
        sign = (x >> 31) << 15
        exp = 16 << 10
        man = x & 0x3FF
        return sign | exp | man

    @classmethod
    def removeLowMantissa(cls, x):
        # input is sign,man(10b),exp(8b)
        # output is sign,man(7b),exp(8b) with man taken from high 7b of input man

        sign = x & (1 << 18)
        manhi = x & (0x7F << 11)
        exp = x & 0xFF
        return (sign >> 3) | (manhi >> 3) | exp

    @classmethod
    def removeHighExponent(cls, x):
        # input is sign,man(10b),exp(8b)
        # output is sign,man(10b),exp(5b) with exp taken from low 5b of input exp

        sign = x & (1 << 18)
        man = x & (0x3FF << 8)
        explo = x & 0x1F
        return (sign >> 3) | (man >> 3) | explo
