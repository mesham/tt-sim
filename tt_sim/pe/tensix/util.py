import importlib.resources as resources
from copy import copy

import numpy as np

from tt_sim.util.bits import extract_bits, get_bits
from tt_sim.util.yaml_cache import load_yaml_cached

# Lookup tables for the block conversions at the end of DataFormatConversions,
# built on first use and then frozen. Keyed by the name of the conversion they
# stand in for; see that section's comment for why they exist and why they
# cannot encode a different function.
_BLOCK_LUTS = {}


def _block_lut(key, bits, build, distributes=True):
    """The table of ``build`` over the whole ``bits``-wide input space.

    Building it has to be cheap, not just amortised: a workload with a couple of
    hundred MVMULs in it would otherwise pay more to build the table than it
    ever saves. Evaluating ``build`` over 2**19 directly is the expensive way --
    it walks a 4 MB intermediate once per operation in the expression, and at
    one page fault per 4 KB of freshly allocated memory that is ~58 ms.

    So the default builds it as an outer OR of two half-width tables, which
    touches the full-size array once (~8 ms). That is valid exactly when
    ``build`` is a *permutation of bits* -- every output bit comes from one input
    bit, with no carries, so the function distributes over an OR of disjoint
    fields -- which is what the storage-layout conversions are. ``distributes``
    is ``False`` for the one that is not (``FP32ToBF16``'s denormal flush reads
    the exponent to decide what to do with the mantissa), and
    ``conversion_batch_test`` walks every table against its arithmetic form over
    the whole input space, so getting this wrong is a loud test failure rather
    than a quietly approximate conversion.
    """
    lut = _BLOCK_LUTS.get(key)
    if lut is None:
        if distributes:
            split = bits // 2
            high = build(np.arange(1 << (bits - split), dtype=np.int64) << split)
            low = build(np.arange(1 << split, dtype=np.int64))
            lut = (high[:, np.newaxis] | low).reshape(-1)
        else:
            lut = build(np.arange(1 << bits, dtype=np.int64))
        lut.setflags(write=False)
        _BLOCK_LUTS[key] = lut
    return lut


class TensixCoprocessorDiagnostics:
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
    # The Tensix backend config-register layout (register name -> ADDR32 / SHAMT
    # / MASK) differs between architectures: Blackhole has a larger, differently
    # indexed register map (e.g. SRCA_SET_Base is thread-config index 3 on
    # Wormhole but 5 on Blackhole). ``use_blackhole`` selects which layout the
    # (process-global) accessors use; it is set when a device's config unit is
    # built. Both layouts are cached, so switching is cheap.
    _YAML_BY_ARCH = {
        False: "tensix_backend_cfg.yaml",
        True: "tensix_backend_cfg_blackhole.yaml",
    }
    _blackhole = False

    @classmethod
    def use_blackhole(cls, blackhole):
        if getattr(cls, "_loaded_arch", None) != blackhole:
            cls._blackhole = blackhole
            cls._load(blackhole)

    @classmethod
    def _load(cls, blackhole):
        yaml_name = cls._YAML_BY_ARCH[blackhole]
        cls.config_constants = load_yaml_cached(
            resources.files("tt_sim.pe.tensix").joinpath(yaml_name),
            yaml_name.removesuffix(".yaml"),
        )
        cls.ids = {}
        for k in cls.config_constants.keys():
            cls.ids[cls.config_constants[k]["ADDR32"]] = k
        cls._loaded_arch = blackhole

    @classmethod
    def init(cls):
        if not hasattr(cls, "config_constants"):
            cls._load(cls._blackhole)

    @classmethod
    def get_name(cls, id):
        cls.init()
        if id in cls.ids:
            return cls.ids[id]
        else:
            return "NONE"

    @classmethod
    def exists(cls, key):
        cls.init()
        return key in cls.config_constants

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
            cls.tensix_instructions = load_yaml_cached(
                resources.files("tt_sim.pe.tensix").joinpath(
                    "tensix_instructions.yaml"
                ),
                "tensix_instructions",
            )

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
        # Keep the raw 32-bit word so handlers can read fields the shared
        # (Wormhole-layout) argument table doesn't expose — e.g. Blackhole's
        # ZEROACC `clear_zero_flags` bit.
        instruction_info["raw_instruction"] = instruction

        return instruction_info


class DataFormatConversions:
    """
    These are data format conversion utilities used throughout the Tensix coprocessor
    backend, typically for converting to and from dst, srcA and srcB data storage
    formats.

    This is heavily based on the functional implementations at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor

    Most of these are pure bit arithmetic -- masks, shifts and ors -- which
    numpy applies elementwise, so passing an int64 array converts a whole block
    of Src/Dst in one call with no second implementation to keep in step. The
    matrix unit's batched MVMUL path does exactly that, as does the unpacker's
    batched datum loop (which converts *into* the Src/Dst layouts where the
    matrix unit converts out of them). ``conversion_batch_test`` proves it: it
    walks the entire 16- or 19-bit input space of each one, and every reachable
    class of the FP32 ones, asserting the array form equals the scalar loop.
    Keep new conversions branch-free where the branch can be written as
    arithmetic (see :meth:`FP32ToBF16`); the ones that cannot (
    :meth:`Int8InSrcToInt8`, :meth:`FP32ToFP16`) are scalar-only and are not
    used by a batched path.
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
    def FP32InDstToBF16(cls, x):
        """A 32-bit Dst datum narrowed to bf16, as the packer narrows it.

        The path `fp32_dest_acc_en` over bf16 circular buffers takes: Dst holds
        fp32 (in the rearranged Dst layout) while the pack source format is
        bf16, so PCK_DEST_RD_CTRL_Read_32b_data reads 32 bits and the packer
        narrows. Follows ttsim's `intermediate_format == 5` arm of the
        `PCK_DEST_RD_CTRL_Read_32b_data` pack path: NaN saturated to infinity,
        round-to-nearest into 16 bits, then a flush of the result's denormals
        to +0. Note the rounding -- unlike ``FP32ToBF16``, which is the Src/Dst
        write path and truncates.
        """
        x = DataFormatConversions.FP32InDstToFP32(x)
        if (x & 0x7FFFFFFF) > 0x7F800000:
            x = (x & 0x80000000) | 0x7F800000
        x = (x + 0x8000) >> 16
        return 0 if (x & 0x7FFF) < 0x80 else x

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
    def ShuffleBF16(cls, x):
        # Dst holds BF16 as Sign,Man(7b),Exp(8b)
        # Src holds BF16 as Sign,Man(10b),Exp(8b)
        return ((x & 0xFF00) << 3) | (x & 0xFF)

    @classmethod
    def ShuffleFP16(cls, x):
        # Dst holds FP16 as Sign,Man(10b),Exp(5b)
        # Src holds FP16 as Sign,Man(10b),Zero(3b),Exp(5b)
        return ((x & 0xFFE0) << 3) | (x & 0x1F)

    @classmethod
    def ShuffleTF32(cls, x):
        # Dst holds TF32 as Sign,HiMan(7b),Exp(8b),LoMan(3b)
        # Src holds TF32 as Sign,Man(10b),Exp(8b)
        signHiMan = x & 0x7F800
        exp = x & 0x007F8
        loMan = x & 0x00007
        return signHiMan | (loMan << 8) | (exp >> 3)

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
        # Flush denormals to signed zero, then truncate toward zero.
        # ``man * (exp != 0)`` is the flush written without a branch, so that
        # this converts a whole numpy block as readily as a single datum (see
        # the note at the top of the class); for a scalar it is the same int.
        sign = x & 0x80000000
        exp = x & 0x7F800000
        man = x & 0x007FFFFF

        return (sign | exp | man * (exp != 0)) >> 16

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

    # -- Table-driven block conversions ---------------------------------------
    #
    # The conversions above are bit permutations of a *narrow* datum: a Src
    # datum is 19 bits, a Dst16b datum 16, and the FP32 ones read nothing below
    # bit 16 that survives into the result. So each one's entire input-output
    # mapping fits in a table, and a whole block converts in one gather instead
    # of the six to nine numpy calls the arithmetic form costs.
    #
    # That matters because of *where* they are called from. An MVMUL rectangle
    # is at most 16x16 datums, so every one of those calls spends its time in
    # numpy's dispatch and in allocating a 2 KB temporary, not on arithmetic.
    # Measured on a 16x16 block: ``BF16InSrcToFP32`` 27.7 us -> 1.0 us,
    # ``FP32ToDstFormatBF16`` 31.0 us -> 2.7 us, ``FP32InDstToFP32`` 34.9 us ->
    # ~7 us.
    #
    # Two things keep them honest:
    #
    # - **The tables are built by the arithmetic form itself**, over its whole
    #   input space, so they cannot encode a different function -- and
    #   ``conversion_batch_test`` walks each block form against the scalar one
    #   over the same space regardless.
    # - **The masks are not defensive.** Each conversion reads only bits inside
    #   its stated width (``TF32InSrcToTF32``'s masks are 0x40000 / 0x3FF00 /
    #   0xFF, so ``x & 0x7FFFF`` is the identity as far as it is concerned), so
    #   masking the index discards exactly the bits the arithmetic already
    #   ignored.
    #
    # They are built lazily -- a 2**19 table is 4 MB, and a workload that never
    # issues a TF32 MVMUL should not pay for one -- and cheaply; see
    # :func:`_block_lut` for why the build is not simply the arithmetic form
    # over an ``arange``, and what a BF16 MVMUL's three tables end up costing.

    @classmethod
    def blockBF16InSrcToFP32(cls, x):
        return _block_lut("BF16InSrcToFP32", 19, cls.BF16InSrcToFP32)[x & 0x7FFFF]

    @classmethod
    def blockTF32InSrcToFP32(cls, x):
        return _block_lut("TF32InSrcToFP32", 19, cls.TF32InSrcToFP32)[x & 0x7FFFF]

    @classmethod
    def blockBF16InDstToFP32(cls, x):
        """``BF16InDstToBF16`` widened to FP32 -- the Dst16b MVMUL operand read.

        The widening shift is folded into the table because the matrix unit
        wants an FP32 bit pattern, never the BF16 one on its own.
        """
        lut = _block_lut("BF16InDstToFP32", 16, lambda a: cls.BF16InDstToBF16(a) << 16)
        return lut[x & 0xFFFF]

    @classmethod
    def blockFP32InDstToFP32(cls, x):
        """Only the high half is permuted, so only the high half is tabulated."""
        lut = _block_lut("BF16InDstToBF16", 16, cls.BF16InDstToBF16)
        return (lut[(x >> 16) & 0xFFFF] << 16) | (x & 0xFFFF)

    @classmethod
    def blockFP32ToDstFormatBF16(cls, x):
        """A function of the high half alone, so one gather does the whole thing.

        ``FP32ToBF16`` truncates toward zero and flushes denormals: the survivor
        is either ``x >> 16`` or just its sign bit, and *which* is decided by
        the exponent, which also lives in the high half. Nothing below bit 16
        can affect the result, so tabulating over ``x >> 16`` is exact rather
        than approximate.
        """
        lut = _block_lut(
            "FP32ToDstFormatBF16",
            16,
            lambda a: cls.FP32ToDstFormatBF16(a << 16),
            # The denormal flush reads the exponent to decide what to do with
            # the mantissa, so this one does not distribute over a bit split.
            distributes=False,
        )
        return lut[(x >> 16) & 0xFFFF]

    @classmethod
    def blockFP32ToDstFormatFP32(cls, x):
        """Only the high half is permuted, so only the high half is tabulated."""
        lut = _block_lut("BF16ToDstFormatBF16", 16, cls.BF16ToDstFormatBF16)
        return (lut[(x >> 16) & 0xFFFF] << 16) | (x & 0xFFFF)
