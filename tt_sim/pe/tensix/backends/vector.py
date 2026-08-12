from tt_sim.pe.tensix.backends.backend_base import DataFormat, TensixBackendUnit
from tt_sim.pe.tensix.registers import LReg
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.perf.model import unit_cost_model
from tt_sim.util.bits import extract_bits, get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_float, conv_to_int32, conv_to_uint32

_M64 = 0xFFFFFFFFFFFFFFFF
_M32 = 0xFFFFFFFF


def _semi_sticky_shift(var, amount, mask):
    """ttsim's ``semi_sticky_shift``: a right shift that ORs a sticky bit into
    the result when it discarded anything — but only if the result is non-zero
    (that is the "semi"). ``mask`` gives the C variable's width."""
    if amount >= mask.bit_length():
        return 0
    orig = var
    v = var >> amount
    if v:
        v |= 1 if (((v << amount) & mask) != orig) else 0
    return v


def fma_model_bh(x, y, z):
    """Blackhole SFPU FP32 fused multiply-add ``x*y + z`` on uint32 bit patterns.

    Verbatim port of ttsim's ``src/fma.cpp`` ``fma_model_bh`` — the exact
    hardware FMA the float ALU implements (denormal-flushing inputs/output,
    3 guard/round/sticky bits, round-to-nearest-even). Every SFPU float
    mul/add/mad on Blackhole rounds through this, so tt-sim must too (a plain
    double ``a*b+c`` diverges in the low mantissa bits and compounds through
    e.g. recip's Newton refinement).
    """

    def unpack(v):
        e = (v >> 23) & 255
        m = (v & 0x7FFFFF) ^ 0x800000
        if e == 0:  # flush denormals
            m = 0
        return e, m

    x_e, x_m = unpack(x)
    y_e, y_m = unpack(y)
    z_e, z_m = unpack(z)
    z_sign = z & 0x80000000

    p_sign = (x ^ y) & 0x80000000
    p_m = x_m * y_m
    p_e = x_e + y_e - 23 - 127

    p_m = (p_m << 3) & _M64
    z_m = (z_m << 3) & _M32
    p_m = (p_m >> 23) | (1 if (p_m & 0x7FFFFF) else 0)
    p_e += 23

    if x_e == 255 or y_e == 255 or p_e >= 255 or z_e == 255:
        if (
            (x_e == 255 and (x_m != 0x800000 or y_m == 0))
            or (y_e == 255 and (y_m != 0x800000 or x_m == 0))
            or (z_e == 255 and z_m != 0x4000000)
            or (z_e == 255 and (x_e == 255 or y_e == 255) and z_sign != p_sign)
        ):
            return 0x7FC00000  # NaN
        if z_e == 255:
            return z  # Inf
        return p_sign | 0x7F800000  # Inf

    if p_m == 0 or p_e < 0:
        return z if z_m else (z_sign & p_sign)

    r_e = p_e if p_e > z_e else z_e
    if p_e < r_e:
        p_m = _semi_sticky_shift(p_m, r_e - p_e, _M64)
    if z_e < r_e:
        z_m = _semi_sticky_shift(z_m, r_e - z_e, _M32)
    r_sign = p_sign if p_m >= z_m else z_sign
    if z_sign != r_sign:
        z_m = (~z_m) & _M32
    if p_sign != r_sign:
        p_m = (~p_m) & _M64
    r_m = (z_m + p_m + (1 if p_sign != z_sign else 0)) & _M32

    if r_m == 0:
        return z_sign & p_sign

    n = 5 - (32 - r_m.bit_length())  # 5 - clz(r_m)
    r_e += n
    if r_e >= 255:
        return r_sign | 0x7F800000  # Inf
    if r_e <= 0:  # denorm or zero
        n += 1
        r_e = 0
    if n <= 0:
        r_m = (r_m << (-n)) & _M32
    else:
        r_m = (r_m >> n) | (1 if (r_m & (n | 1)) else 0)

    r = ((r_e << 23) + ((r_m >> 3) & 0x7FFFFF)) & _M32
    r += 1 if (((r_m & 7) + (r & 1)) > 4) else 0  # round to nearest even
    if not (r >> 23):  # flush denormals (post-round, keep sign)
        r = 0
    return (r_sign | r) & _M32


def fma_model_wh(x, y, z):
    """Wormhole SFPU FP32 fused multiply-add ``x*y + z`` on uint32 bit patterns.

    Verbatim port of ttsim's ``src/fma.cpp`` ``fma_model_wh``. It is the same
    shape as ``fma_model_bh`` but is a *different* piece of silicon, and the
    differences are not cosmetic:

    - NaNs are not returned immediately. Wormhole assembles ``nan_result``
      (``0x7f800001``, sign-carrying) and keeps computing, so mantissa bits of
      the real result leak into the returned NaN.
    - An underflowing product returns ``+0`` rather than ``z_sign & p_sign``,
      and a denormal result is flushed *discarding* the sign (``r_e < 0``
      returns the NaN slot, i.e. ``0``) rather than being renormalised.
    - The final sticky is ``r_m & 1`` rather than Blackhole's
      ``(r_m & (n | 1)) != 0``, so a shifted-out bit below the low bit does not
      make it into the rounding decision.

    Fuzz-matched bit-for-bit against ttsim's C over 200k random triples
    (``tt_sim/pe/tensix/fma_model_test.py`` pins the vectors).
    """

    def unpack(v):
        e = (v >> 23) & 255
        m = (v & 0x7FFFFF) ^ 0x800000
        if e == 0:  # flush denormals
            m = 0
        return e, m

    x_e, x_m = unpack(x)
    y_e, y_m = unpack(y)
    z_e, z_m = unpack(z)
    z_sign = z & 0x80000000

    p_sign = (x ^ y) & 0x80000000
    p_m = x_m * y_m
    p_e = x_e + y_e - 23 - 127

    p_m = (p_m << 3) & _M64
    z_m = (z_m << 3) & _M32
    p_m = (p_m >> 23) | (1 if (p_m & 0x7FFFFF) else 0)
    p_e += 23

    nan_result = 0
    if x_e == 255 or y_e == 255 or p_e >= 255 or z_e == 255:
        if (
            (x_e == 255 and (x_m != 0x800000 or y_m == 0))
            or (y_e == 255 and (y_m != 0x800000 or x_m == 0))
            or (
                z_e == 255
                and z_m == 0x4000000
                and (x_e == 255 or y_e == 255 or p_e >= 255)
                and z_sign != p_sign
            )
        ):
            nan_result = p_sign | 0x7F800001
        elif z_e == 255 and z_m != 0x4000000:  # z NaN
            nan_result = z_sign | 0x7F800001
        elif z_e == 255:  # z Inf
            return z
        else:  # (x * y) Inf
            return p_sign | 0x7F800000
        if p_e > 255:
            p_e = 255

    if p_m == 0 or p_e < 0:
        if nan_result:
            p_m, p_e = 0, 0
        else:
            return z if z_m else 0

    r_e = p_e if p_e > z_e else z_e
    if p_e < r_e:
        p_m = _semi_sticky_shift(p_m, r_e - p_e, _M64)
    if z_e < r_e:
        z_m = _semi_sticky_shift(z_m, r_e - z_e, _M32)
    r_sign = p_sign if p_m >= z_m else z_sign
    if z_sign != r_sign:
        z_m = (~z_m) & _M32
    if p_sign != r_sign:
        p_m = (~p_m) & _M64
    r_m = (z_m + p_m + (1 if p_sign != z_sign else 0)) & _M32

    if r_m == 0:
        return nan_result

    n = 5 - (32 - r_m.bit_length())  # 5 - clz(r_m)
    r_e += n
    if r_e >= 255:
        return nan_result if nan_result else (r_sign | 0x7F800000)
    if r_e < 0:  # flush blatant denormals (before rounding, discarding sign)
        return nan_result
    if n <= 0:
        r_m = (r_m << (-n)) & _M32
    else:
        r_m = (r_m >> n) | (r_m & 1)

    r = ((r_e << 23) + ((r_m >> 3) & 0x7FFFFF)) & _M32
    r += 1 if (((r_m & 7) + (r & 1)) > 4) else 0  # round to nearest even
    if not (r >> 23):  # flush denormals (after rounding, discarding sign)
        return nan_result
    return ((nan_result if nan_result else r_sign) | r) & _M32


class VectorUnit(TensixBackendUnit):
    """
    SFPU vector unit, which has 32 lanes of 32 bit and 17 LRegs that can feed these lanes.

    This is based on the description and functional code snippets at
    https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/VectorUnit.md
    """

    OPCODE_TO_HANDLER = {
        "SFPENCC": "handle_sfpencc",
        "SFPLOADI": "handle_sfploadi",
        "SFPLOAD": "handle_sfpload",
        "SFPCONFIG": "handle_sfpconfig",
        "SFPIADD": "handle_sfpiadd",
        "SFPSTORE": "handle_sfpstore",
        "SFPMAD": "handle_mad",
        "SFPADD": "handle_add",
        "SFPMUL": "handle_mul",
        "SFPLUT": "handle_sfplut",
        "SFPLUTFP32": "handle_sfplutfp32",
        "SFPTRANSP": "handle_sfptransp",
        "SFPLOADMACRO": "handle_sfploadmacro",
        # Blackhole-only SFPU superset.
        "SFPGT": "handle_sfpgt",
        "SFPLE": "handle_sfple",
        "SFPMUL24": "handle_sfpmul24",
        "SFPARECIP": "handle_sfparecip",
        "SFPADDI": "handle_addi",
        "SFPMULI": "handle_muli",
        "SFPABS": "handle_sfpabs",
        "SFPMOV": "handle_sfpmov",
        "SFPSWAP": "handle_sfpswap",
        "SFPCAST": "handle_sfpcast",
        "SFP_STOCH_RND": "handle_sfp_stoch_rnd",
        "SFPDIVP2": "handle_sfpdivp2",
        "SFPEXEXP": "handle_sfpexexp",
        "SFPEXMAN": "handle_sfpexman",
        "SFPSETEXP": "handle_sfpsetexp",
        "SFPSETMAN": "handle_sfpsetman",
        "SFPSHFT": "handle_sfpshft",
        "SFPSHFT2": "handle_sfpshft2",
        "SFPSETSGN": "handle_sfpsetsgn",
        "SFPAND": "handle_sfpand",
        "SFPOR": "handle_sfpor",
        "SFPXOR": "handle_sfpxor",
        "SFPNOT": "handle_sfpnot",
        "SFPNOP": "handle_sfpnop",
        "SFPSETCC": "handle_sfpsetcc",
        "SFPPUSHC": "handle_sfppushc",
        "SFPPOPC": "handle_sfppopc",
        "SFPCOMPC": "handle_sfpcompc",
    }

    MOD1_IMM16_IS_VALUE = 1
    MOD1_BITWISE_OR = 2
    MOD1_BITWISE_AND = 4
    MOD1_BITWISE_XOR = 6
    MOD1_IMM16_IS_LANE_MASK = 8

    # Blackhole SFPGT / SFPLE ``instr_mod1``: where the comparison result goes.
    SFPCMP_MOD1_SET_LANE_FLAGS = 1
    SFPCMP_MOD1_SET_VD = 8

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

    SFPSETCC_MOD1_IMM_BIT0 = 1
    SFPSETCC_MOD1_CLEAR = 8

    SFPSETCC_MOD1_LREG_LT0 = 0
    SFPSETCC_MOD1_LREG_NE0 = 2
    SFPSETCC_MOD1_LREG_GTE0 = 4
    SFPSETCC_MOD1_LREG_EQ0 = 6

    MOD0_FMT_SRCB = 0
    MOD0_FMT_FP16 = 1
    MOD0_FMT_BF16 = 2
    MOD0_FMT_FP32 = 3
    MOD0_FMT_INT32 = 4
    MOD0_FMT_INT8 = 5
    MOD0_FMT_UINT16 = 6
    MOD0_FMT_HI16 = 7
    MOD0_FMT_INT16 = 8
    MOD0_FMT_LO16 = 9
    MOD0_FMT_INT32_ALL = 10
    MOD0_FMT_ZERO = 11
    MOD0_FMT_INT32_SM = 12
    MOD0_FMT_INT8_COMP = 13
    MOD0_FMT_LO16_ONLY = 14
    MOD0_FMT_HI16_ONLY = 15

    SFPIADD_MOD1_ARG_LREG_DST = 0
    SFPIADD_MOD1_ARG_IMM = 1
    SFPIADD_MOD1_ARG_2SCOMP_LREG_DST = 2
    SFPIADD_MOD1_CC_LT0 = 0
    SFPIADD_MOD1_CC_NONE = 4
    SFPIADD_MOD1_CC_GTE0 = 8
    SFPMAD_MOD1_INDIRECT_VA = 4
    SFPMAD_MOD1_INDIRECT_VD = 8
    SFPABS_MOD1_FLOAT = 1
    SFPMOV_MOD1_NEGATE = 1
    SFPMOV_MOD1_ALL_LANES_ENABLED = 2
    SFPMOV_MOD1_FROM_SPECIAL = 8

    # SFPSWAP instruction modifier. Mode 0 is an unconditional swap of VD/VC;
    # modes 1-8 are min/max sorts where VD receives the smaller value on the
    # rows listed here (and the larger value on the other rows), VC receiving
    # the opposite. Row = lane // 8, matching the SFPLOAD/SFPSTORE lane->row
    # mapping. Modes >= 9 are reserved and documented as "VD gets max on every
    # row", i.e. the empty min-row set. Per
    # WormholeB0/TensixTile/TensixCoprocessor/SFPSWAP.md.
    SFPSWAP_MOD1_UNCONDITIONAL = 0
    SFPSWAP_MOD1_MIN_ROWS = {
        1: frozenset({0, 1, 2, 3}),
        2: frozenset({0, 1}),
        3: frozenset({0, 2}),
        4: frozenset({0, 3}),
        5: frozenset({0}),
        6: frozenset({1}),
        7: frozenset({2}),
        8: frozenset({3}),
    }

    # SFP_STOCH_RND conversion modes (instr_mod1 bits [2:0]); bit 3 selects the
    # immediate descale over src_b. The round-to-nearest constant the ISA uses
    # for the deterministic path is 0x400000; tt-sim has no SFPU PRNG so the
    # stochastic (rnd_mode == 1) path reuses it. Per
    # WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_*.md.
    SFP_STOCH_RND_PRNG_RNE = 0x400000
    # rnd_mode values. Wormhole has a one-bit field (nearest-even / stochastic);
    # Blackhole widens it to two bits and adds round-toward-zero (sfpi's
    # RoundMode::Zero, guarded by __riscv_xtttensixbh).
    SFP_STOCH_RND_RND_NEAREST_EVEN = 0
    SFP_STOCH_RND_RND_STOCHASTIC = 1
    SFP_STOCH_RND_RND_ZERO = 2
    # Every rounding decision in the op is ``discarded >= constant`` where the
    # discarded part is always below 0x800000, so this constant never rounds up
    # — i.e. it truncates, which on these sign-magnitude data is exactly
    # round-toward-zero.
    SFP_STOCH_RND_PRNG_TRUNCATE = 0x800000
    SFP_STOCH_RND_FP32_TO_FP16A = 0
    SFP_STOCH_RND_FP32_TO_FP16B = 1
    # fp32 -> integer modes: (keep_sign, max_magnitude).
    SFP_STOCH_RND_FLOAT_TO_INT = {
        2: (False, 255),  # fp32 -> unsigned int8
        3: (True, 127),  # fp32 -> signed int8
        6: (False, 65535),  # fp32 -> unsigned int16
        7: (True, 32767),  # fp32 -> signed int16
    }
    SFP_STOCH_RND_INT32_TO_UINT8 = 4
    SFP_STOCH_RND_INT32_TO_INT8 = 5

    # SFPDIVP2: bit 0 set -> wrapping-add the 8-bit immediate to the exponent
    # field (multiply/divide by 2**imm); clear -> replace the exponent field.
    # Per WormholeB0/TensixTile/TensixCoprocessor/SFPDIVP2.md.
    SFPDIVP2_MOD1_ADD = 1
    SFPSETSGN_MOD1_ARG_IMM = 1
    SFPEXEXP_MOD1_NODEBIAS = 1
    SFPEXEXP_MOD1_SET_CC_SGN_EXP = 2
    SFPEXEXP_MOD1_SET_CC_COMP_EXP = 8
    SFPEXMAN_MOD1_PAD9 = 1
    SFPSETEXP_MOD1_ARG_IMM = 1
    SFPSETEXP_MOD1_ARG_EXPONENT = 2
    SFPSETMAN_MOD1_ARG_IMM = 1
    SFPSHFT_MOD1_ARG_IMM = 1
    # Blackhole-only SFPSHFT modifier bits (Wormhole reserves everything above
    # bit 0): bit 1 makes right shifts arithmetic, bit 2 takes the value being
    # shifted from VC instead of VD (sfpi's SFPSHFT_MOD1_{ARITHMETIC,SRC_LREGC}).
    SFPSHFT_MOD1_ARITHMETIC = 2
    SFPSHFT_MOD1_SRC_LREG_C = 4

    # SFPSHFT2 instruction modifiers (the subset the reference simulator models).
    SFPSHFT2_MOD1_SUBVEC_SHFLROR1 = 3
    SFPSHFT2_MOD1_SUBVEC_SHFLSHR1 = 4  # Blackhole only: broken in Wormhole silicon
    SFPSHFT2_MOD1_SHFT_LREG = 5

    # SFPLUT / SFPLUTFP32 instruction modifiers, per
    # {WormholeB0,BlackholeA0}/TensixTile/TensixCoprocessor/SFPLUT{,FP32}.md
    # (both architectures document identical behaviour). Both instructions
    # evaluate a piecewise-linear function of Abs(LReg[3]) as a single
    # multiply-add, picking the coefficients from a table selected by the range
    # the input falls in.
    SFPLUT_MOD0_SGN_RETAIN = 4  # Result takes LReg[3]'s sign bit
    SFPLUT_MOD0_INDIRECT_VD = 8  # Destination index comes from LReg[7][3:0]
    # SFPLUTFP32's table-selection modifiers are decoded as combinations rather
    # than as an enum: bit 1 selects the FP16-packed tables (clear = one FP32
    # coefficient per LReg), and within those (Mod1 & 10) == 10 selects the
    # 3-entry table while bit 0 moves the 6-entry table's last cut from 3.0 to
    # 4.0. Note FP16_3ENTRY_TABLE (10) has INDIRECT_VD (8) set as part of its
    # encoding — that is the hardware workaround the ISA docs call out, not a
    # typo, and the verbatim decode below reproduces it.
    SFPLUTFP32_MOD1_FP32_3ENTRY_TABLE = 0
    SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1 = 2
    SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2 = 3
    SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE = 10
    SFPLUTFP32_MOD1_SGN_RETAIN = 4
    SFPLUTFP32_MOD1_INDIRECT_VD = 8
    # FP32 bit patterns of the 6-entry table's range boundaries.
    SFPLUT_RANGE_1_0 = 0x3F800000
    SFPLUT_RANGE_2_0 = 0x40000000
    SFPLUT_RANGE_0_5 = 0x3F000000
    SFPLUT_RANGE_1_5 = 0x3FC00000
    SFPLUT_RANGE_3_0 = 0x40400000
    SFPLUT_RANGE_4_0 = 0x40800000

    # SFPCAST instruction modifiers. Modes 0 and 1 (int32 -> fp32, rounding to
    # nearest-even / stochastically) exist on both architectures; Blackhole adds
    # the sign-magnitude <-> two's-complement conversions.
    SFPCAST_MOD1_INT32_TO_FP32_RNE = 0
    SFPCAST_MOD1_INT32_TO_FP32_RNS = 1
    SFPCAST_MOD1_SM32_TO_INT32 = 2
    SFPCAST_MOD1_INT32_TO_SM32 = 3

    ENABLE_FP16A_INF = (0, 1)
    DISABLE_BACKDOOR_LOAD = (1, 1)
    ENABLE_DEST_INDEX = (2, 1)
    CAPTURE_DEFAULT_DEST_INDEX = (3, 1)
    BLOCK_DEST_WR_FROM_SFPU = (4, 1)
    BLOCK_SFPU_RD_FROM_DEST = (5, 1)
    DEST_RD_COL_EXCHANGE = (6, 1)
    DEST_WR_COL_EXCHANGE = (7, 1)
    EXCHANGE_SRCB_SRCC = (8, 1)
    BLOCK_DEST_MOV = (9, 2)
    ROW_MASK = (12, 4)

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
        # The float ALU is per-architecture silicon: every SFPU float
        # multiply/add rounds through the model for this chip (ttsim's
        # ``#define fma_model fma_model_{wh,bh}``).
        self.fma = fma_model_bh if backend.blackhole else fma_model_wh
        self.lregs = [LReg(blackhole=backend.blackhole) for i in range(17)]
        self.lregs[8].setReadOnly(0.8373)
        self.lregs[9].setReadOnly(0)
        self.lregs[10].setReadOnly(1.0)
        for i in range(32):
            self.lregs[15][i] = i * 2
        self.lregs[15].setReadOnly()

        self.laneFlags = [False] * 32
        self.useLaneFlagsForLaneEnable = [False] * 32
        self.flagStack = []
        self.laneConfig = [0] * 32
        self.loadMacroConfig = [VectorUnit.LoadMacroConfig() for i in range(32)]
        super().__init__(backend, VectorUnit.OPCODE_TO_HANDLER, "Vector")
        # Phase 5 of docs/plans/event-driven-pump.md. ``None`` unless
        # TT_SIM_COST_MODEL is set. The SFPU's table is the best-sourced in the
        # file -- the ISA docs publish a latency for all 42 opcodes -- and its
        # answer is that *occupancy* is 1 cycle for every one of them: the five
        # sub-units are pipelined, so the 2-cycle latency of the arithmetic and
        # LUT ops is time-to-result, not time-the-unit-is-held. The unit "can
        # only accept one instruction per cycle from the outside world", which
        # is exactly tt-sim's issue behaviour, so this charges nothing new.
        # See docs/plans/cost-model.md ("The first consumer").
        #
        # THE LATENCY COLUMN IS WHAT ``STALLWAIT``'s C14 NEEDS, and since
        # 2026-08-12 it is read: "the current thread has an instruction in any
        # stage of the Vector Unit (SFPU) pipeline" (Blackhole numbers the same
        # condition C11) is a residency question, and the sentence above is
        # precisely why occupancy cannot answer it -- 1-cycle occupancy with
        # 2-cycle latency is a pipelined unit, so the instruction is still in a
        # stage after the unit has taken the next one. The 2-cycle rows are the
        # arithmetic and LUT ops (``SFPADD``, ``SFPADDI``, ``SFPMAD``,
        # ``SFPMUL``, ``SFPMULI``, ``SFPLUT``, ``SFPLUTFP32``, ``SFPSWAP``,
        # Blackhole's ``SFPMUL24``) plus ``SFPCONFIG``'s and one ``SFPSHFT2``
        # form's "≤ 2 cycles", charged at the low end like every other bound;
        # every other opcode is 1 cycle and so arms nothing at all.
        # ``SFPLOADMACRO``'s latency is published as "Complex", which the table
        # records as no latency, so it keeps the same-cycle report it had.
        # tt-metal reaches C14 through ``p_stall::WAIT_SFPU``, issued by
        # ``_llk_math_dest_section_done_`` and ``_llk_math_eltwise_sfpu_done_``.
        self.cost_model = unit_cost_model(
            "SFPU", "blackhole" if backend.blackhole else "wormhole"
        )

    def laneConfigValue(self, lane, key):
        assert len(key) == 2
        return get_bits(self.laneConfig[lane], key[0], (key[0] + key[1]) - 1)

    def handle_sfpnot(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = ~ lreg[{vc}]")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = ~conv_to_uint32(self.lregs[vc][lane])

    def handle_sfpxor(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        # The SFPU MATHI binary logical ops take their first source from the
        # imm12 field: LREG[dest] = LREG[imm12] OP LREG[c]. sfpi emits a nonzero
        # imm12 (often != dest) when it reuses another lreg for the result — e.g.
        # bitwise_and/or_tile's last tile chunk writes into the mask's register,
        # so assuming vb==vd there drops the loaded data and stores mask OP mask.
        # imm12 == 0 encodes the in-place 2-operand form (vb == dest), which is
        # what tt-metal's xor path emits; fall back to vd so that stays correct
        # and every existing imm12==0 caller is byte-identical.
        vb = instr_args["imm12_math"] or vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vb}] ^ lreg[{vc}]")
        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = conv_to_uint32(
                        self.lregs[vb][lane]
                    ) ^ conv_to_uint32(self.lregs[vc][lane])

    def handle_sfpand(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        vb = (
            instr_args["imm12_math"] or vd
        )  # src1 = imm12 (0 => in-place vd); see SFPXOR
        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vb}] & lreg[{vc}]")
        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = conv_to_uint32(
                        self.lregs[vb][lane]
                    ) & conv_to_uint32(self.lregs[vc][lane])

    def handle_sfpor(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        vb = (
            instr_args["imm12_math"] or vd
        )  # src1 = imm12 (0 => in-place vd); see SFPXOR
        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vb}] | lreg[{vc}]")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    self.lregs[vd][lane] = conv_to_uint32(
                        self.lregs[vb][lane]
                    ) | conv_to_uint32(self.lregs[vc][lane])

    def handle_sfpsetsgn(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm1 = instr_args["imm12_math"]

        vb = vd
        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    c = conv_to_uint32(self.lregs[vc][lane])
                    exp = (c >> 23) & 0xFF
                    man = c & 0x7FFFFF
                    if mod1 & VectorUnit.SFPSETSGN_MOD1_ARG_IMM:
                        sign = imm1 & 0x1
                    else:
                        b = conv_to_uint32(self.lregs[vb][lane])
                        sign = b >> 31
                    self.lregs[vd][lane] = (sign << 31) | (exp << 23) | man

    def handle_sfpabs(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = abs(lreg[{vc}])")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    x = conv_to_uint32(self.lregs[vc][lane])
                    is_float = bool(mod1 & VectorUnit.SFPABS_MOD1_FLOAT)
                    if x >= 0x80000000:
                        # Sign bit is set, i.e. value is negative.
                        if is_float:
                            if x > 0xFF800000:
                                # Value is -NaN; leave it as -NaN
                                pass
                            else:
                                # Clear the sign bit, i.e. floating-point negation
                                x &= 0x7FFFFFFF
                        else:
                            # Two's complement integer negation, unless the input is
                            # -2147483648, in which case it remains as -2147483648
                            x = -x
                    else:
                        # Value is positive (or zero); leave it as-is
                        pass
                    self.lregs[vd][lane] = x

    def handle_sfpmov(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]

        if mod1 & VectorUnit.SFPMOV_MOD1_FROM_SPECIAL:
            raise NotImplementedError(
                "SFPMOV with FROM_SPECIAL (mod1 bit 3) is not yet supported"
            )

        if self.getDiagnosticSettings().reportSFPUCalculations():
            src = (
                f"-lreg[{vc}]"
                if mod1 & VectorUnit.SFPMOV_MOD1_NEGATE
                else f"lreg[{vc}]"
            )
            print(f"SFPU: lreg[{vd}] = {src}")

        bypass_mask = bool(mod1 & VectorUnit.SFPMOV_MOD1_ALL_LANES_ENABLED)
        if vd < 8 or vd == 16:
            for lane in range(32):
                if bypass_mask or self.isLaneEnabled(lane):
                    value = self.lregs[vc][lane]
                    if mod1 & VectorUnit.SFPMOV_MOD1_NEGATE:
                        value = conv_to_uint32(value) ^ 0x80000000
                    self.lregs[vd][lane] = value

    @staticmethod
    def _sign_mag_key(value):
        """Monotonic key implementing the SFPSWAP total order.

        Reinterprets the 32-bit lane as sign-magnitude and returns an
        unsigned key such that numeric ``key(a) < key(b)`` iff ``a`` is
        smaller under the order ``-NaN < -Inf < ... < -0 < +0 < ... <
        +Inf < +NaN`` (the ``SignMagIsSmaller`` semantics from the ISA
        docs). Negatives get all bits inverted; non-negatives get only
        the sign bit set.
        """
        u = conv_to_uint32(value) & 0xFFFFFFFF
        if u & 0x80000000:
            return ~u & 0xFFFFFFFF
        return u | 0x80000000

    def handle_sfpswap(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: swap(lreg[{vd}], lreg[{vc}]) mode {mod1}")

        vd_writable = vd < 8 or vd == 16
        vc_writable = vc < 8 or vc == 16
        if not (vd_writable or vc_writable):
            return

        # Modes >= 9 are reserved ("VD gets max on every row"); treat as the
        # empty min-row set so the else-branch (VD=max) applies everywhere.
        min_rows = VectorUnit.SFPSWAP_MOD1_MIN_ROWS.get(mod1, frozenset())

        for lane in range(32):
            if not self.isLaneEnabled(lane):
                continue
            d_val = self.lregs[vd][lane]
            c_val = self.lregs[vc][lane]

            if mod1 == VectorUnit.SFPSWAP_MOD1_UNCONDITIONAL:
                new_d, new_c = c_val, d_val
            else:
                # Preserve original bit patterns: pick the stored lane values,
                # ordering them by the sign-magnitude total order.
                if self._sign_mag_key(c_val) < self._sign_mag_key(d_val):
                    smaller, larger = c_val, d_val
                else:
                    smaller, larger = d_val, c_val
                if (lane // 8) in min_rows:
                    new_d, new_c = smaller, larger
                else:
                    new_d, new_c = larger, smaller

            if vd_writable:
                self.lregs[vd][lane] = new_d
            if vc_writable:
                self.lregs[vc][lane] = new_c

    @staticmethod
    def _cast_negate(c, sign):
        """Blackhole SFPCAST modes 2 and 3 (sign-magnitude <-> two's complement).

        Verbatim from ttsim's ``TENSIX_EXECUTE_SFPCAST`` Blackhole branch
        (``dst = sign | (sign ? -src : src)``): the whole word is negated when
        the sign bit is set and the sign bit is then forced back on. That single
        involution converts in both directions, which is why the two modes share
        it. It also reproduces the hardware quirk sfpi documents next to
        ``SFPCAST_MOD1_SM32_TO_INT32`` — sign-magnitude ``-0`` (0x80000000)
        negates to itself, so it comes out as the most negative int32 rather
        than as zero.
        """
        return (sign | ((-c) if sign else c)) & 0xFFFFFFFF

    def handle_sfpcast(self, instruction_info, issue_thread, instr_args):
        # Cast a sign-magnitude int32 in VC to FP32 in VD. Verbatim port of the
        # pseudocode in WormholeB0/TensixTile/TensixCoprocessor/SFPCAST.md.
        # mod1 & 1 selects stochastic rounding (seven PRNG bits) on hardware;
        # tt-sim does not model the SFPU PRNG, so both modes round to nearest
        # even here (the round-to-nearest branch below).
        #
        # Blackhole adds two more modes (2 and 3), which convert between
        # sign-magnitude and two's-complement integers rather than producing a
        # float; see ``_cast_negate`` for the shared body.
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        int_convert = self.backend.blackhole and mod1 in (
            VectorUnit.SFPCAST_MOD1_SM32_TO_INT32,
            VectorUnit.SFPCAST_MOD1_INT32_TO_SM32,
        )

        if self.getDiagnosticSettings().reportSFPUCalculations():
            if int_convert:
                kind = (
                    "sm32" if mod1 == VectorUnit.SFPCAST_MOD1_SM32_TO_INT32 else "i32"
                )
                print(f"SFPU: lreg[{vd}] = negate_if_signed({kind} lreg[{vc}])")
            else:
                mode = "stochastic~rne" if (mod1 & 1) else "rne"
                print(f"SFPU: lreg[{vd}] = fp32(int32 lreg[{vc}]) [{mode}]")

        if not (vd < 8 or vd == 16):
            return

        for lane in range(32):
            if not self.isLaneEnabled(lane):
                continue
            c = conv_to_uint32(self.lregs[vc][lane]) & 0xFFFFFFFF
            sign = c & 0x80000000
            if int_convert:
                self.lregs[vd][lane] = self._cast_negate(c, sign)
                continue
            mag = c & 0x7FFFFFFF
            # __builtin_clz of the 32-bit magnitude; the docs use 157 as the
            # sentinel for mag == 0 so the exponent field lands on zero.
            lz = (32 - mag.bit_length()) if mag else 157
            norm = (mag << (lz & 31)) & 0xFFFFFFFF
            # The implicit leading 1 in (norm >> 8) carries into the exponent
            # field, which is why (157 - lz) rather than (158 - lz) is used.
            d = (sign + ((157 - lz) << 23) + (norm >> 8)) & 0xFFFFFFFF
            # Round to nearest, ties to even: round up when the guard bit is
            # set and either the LSB or any sticky bit is set.
            if (norm & 0x80) and (norm & 0x17F):
                d = (d + 1) & 0xFFFFFFFF
            self.lregs[vd][lane] = d

    def _read_rnd_mode(self, instruction_info):
        # SFP_STOCH_RND's ``rnd_mode`` is one bit on Wormhole (raw bit 21) and
        # two bits on Blackhole (raw bits 22:21, the extra encoding being
        # round-toward-zero). The shared instruction table gives the field the
        # whole top of the argument word (23:21), so read the architecture's
        # real width straight out of the raw instruction — the same idiom as
        # ``_read_sfpu_addr_mode``.
        if instruction_info is None:
            return VectorUnit.SFP_STOCH_RND_RND_NEAREST_EVEN
        width = 2 if self.backend.blackhole else 1
        return extract_bits(instruction_info["raw_instruction"], width, 21)

    def handle_sfp_stoch_rnd(self, instruction_info, issue_thread, instr_args):
        # Round / narrow VC into VD. Verbatim port of the pseudocode in
        # WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_{FloatFloat,
        # FloatInt,IntInt}.md. mod bits [2:0] pick the conversion; bit 3
        # selects the immediate descale over src_b (int32->int8 only). rnd_mode
        # selects stochastic rounding on hardware, which needs the SFPU PRNG
        # tt-sim does not model, so that path uses the round-to-nearest
        # constant 0x400000. Blackhole additionally has rnd_mode 2 (round toward
        # zero), which is deterministic and therefore modelled exactly.
        mod = instr_args["instr_mod1"]
        mode = mod & 0x7
        use_imm = bool(mod & 0x8)
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        imm8 = instr_args["imm8_math"]
        rnd_mode = self._read_rnd_mode(instruction_info)

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = stochrnd(lreg[{vc}]) mode {mode} rnd {rnd_mode}")

        if rnd_mode == VectorUnit.SFP_STOCH_RND_RND_ZERO:
            prng = VectorUnit.SFP_STOCH_RND_PRNG_TRUNCATE
        else:
            prng = VectorUnit.SFP_STOCH_RND_PRNG_RNE

        for lane in range(32):
            gate = vd < 12 or self.laneConfigValue(
                lane, VectorUnit.DISABLE_BACKDOOR_LOAD
            )
            if not (gate and self.isLaneEnabled(lane)):
                continue

            if mode in (
                VectorUnit.SFP_STOCH_RND_FP32_TO_FP16A,
                VectorUnit.SFP_STOCH_RND_FP32_TO_FP16B,
            ):
                x = conv_to_uint32(self.lregs[vc][lane]) & 0xFFFFFFFF
                exp = (x >> 23) & 0xFF
                if exp == 0:
                    x = 0  # denormals / zero -> +0
                elif exp == 255:
                    x &= 0xFF800000  # NaN / Inf -> normalized Inf
                elif mode == VectorUnit.SFP_STOCH_RND_FP32_TO_FP16A:
                    discarded = x & 0x1FFF  # keep 10 mantissa bits
                    x -= discarded
                    if discarded >= (prng >> 10):
                        x += 0x2000
                else:
                    discarded = x & 0xFFFF  # keep 7 mantissa bits (bf16)
                    x -= discarded
                    if discarded >= (prng >> 7):
                        x += 0x10000
                result = x & 0xFFFFFFFF
            elif mode in VectorUnit.SFP_STOCH_RND_FLOAT_TO_INT:
                keep_sign, max_mag = VectorUnit.SFP_STOCH_RND_FLOAT_TO_INT[mode]
                c = conv_to_uint32(self.lregs[vc][lane]) & 0xFFFFFFFF
                sign = (c & 0x80000000) if keep_sign else 0
                exp = ((c >> 23) & 0xFF) - 127
                if exp < -1:
                    mag = 0  # |x| < 0.5 -> 0
                    sign = 0
                elif exp >= 16:
                    mag = max_mag  # |x| >= 2**16 (and NaN) -> saturate
                else:
                    mag = 0x800000 | (c & 0x7FFFFF)
                    mag = (mag << exp) if exp >= 0 else (mag >> -exp)
                    mag = (mag >> 23) + (1 if (mag & 0x7FFFFF) >= prng else 0)
                    if mag > max_mag:
                        mag = max_mag
                    if mag == 0:
                        sign = 0
                result = (sign + mag) & 0xFFFFFFFF  # sign-magnitude integer
            elif mode in (
                VectorUnit.SFP_STOCH_RND_INT32_TO_UINT8,
                VectorUnit.SFP_STOCH_RND_INT32_TO_INT8,
            ):
                c = conv_to_uint32(self.lregs[vc][lane]) & 0xFFFFFFFF
                sign = c & 0x80000000
                mag = c & 0x7FFFFFFF  # sign-magnitude source
                mag <<= 23
                descale = (
                    (imm8 & 0x1F)
                    if use_imm
                    else (conv_to_uint32(self.lregs[vb][lane]) & 0x1F)
                )
                mag >>= descale
                mag = (mag >> 23) + (1 if (mag & 0x7FFFFF) >= prng else 0)
                if mode == VectorUnit.SFP_STOCH_RND_INT32_TO_UINT8:
                    mag = min(mag, 255)
                    sign = 0
                else:
                    mag = min(mag, 127)
                    if mag == 0:
                        sign = 0
                result = (sign + mag) & 0xFFFFFFFF  # sign-magnitude integer
            else:
                raise NotImplementedError(f"SFP_STOCH_RND mode {mode} is reserved")

            if vd < 8 or vd == 16:
                self.lregs[vd][lane] = result

    def handle_sfpdivp2(self, instruction_info, issue_thread, instr_args):
        # Scale by a power of two by adjusting the FP32 exponent field.
        # Verbatim port of WormholeB0/TensixTile/TensixCoprocessor/SFPDIVP2.md.
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm8 = instr_args["imm12_math"] & 0xFF

        if self.getDiagnosticSettings().reportSFPUCalculations():
            op = (
                f"exp += {imm8}"
                if (mod1 & VectorUnit.SFPDIVP2_MOD1_ADD)
                else f"exp = {imm8}"
            )
            print(f"SFPU: lreg[{vd}] = scale2(lreg[{vc}]) [{op}]")

        if not (vd < 8 or vd == 16):
            return

        for lane in range(32):
            if not self.isLaneEnabled(lane):
                continue
            c = conv_to_uint32(self.lregs[vc][lane]) & 0xFFFFFFFF
            sign = c & 0x80000000
            exp = (c >> 23) & 0xFF
            man = c & 0x7FFFFF
            if mod1 & VectorUnit.SFPDIVP2_MOD1_ADD:
                if exp != 0xFF:  # leave Inf / NaN unchanged
                    exp = (exp + imm8) & 0xFF
            else:
                exp = imm8
            self.lregs[vd][lane] = sign | (exp << 23) | man

    def handle_sfpexexp(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]

        bias = 0 if (mod1 & VectorUnit.SFPEXEXP_MOD1_NODEBIAS) else 127

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = exponent(lreg[{vc}]) - {bias}")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    c = conv_to_uint32(self.lregs[vc][lane])
                    exp = (c >> 23) & 0xFF
                    result = exp - bias
                    self.lregs[vd][lane] = result
                    if vd < 8:
                        if mod1 & VectorUnit.SFPEXEXP_MOD1_SET_CC_SGN_EXP:
                            self.laneFlags[lane] = result < 0
                        if mod1 & VectorUnit.SFPEXEXP_MOD1_SET_CC_COMP_EXP:
                            self.laneFlags[lane] = not self.laneFlags[lane]

    def handle_sfpexman(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]

        hidden_bit = 0 if (mod1 & VectorUnit.SFPEXMAN_MOD1_PAD9) else (1 << 23)

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = mantissa(lreg[{vc}]) + {hex(hidden_bit)}")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    c = conv_to_uint32(self.lregs[vc][lane])
                    man = c & 0x7FFFFF
                    self.lregs[vd][lane] = hidden_bit + man

    def handle_sfpsetexp(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm = instr_args["imm12_math"] & 0xFF
        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = setexp(lreg[{vc}])")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    c = conv_to_uint32(self.lregs[vc][lane])
                    sign = c >> 31
                    man = c & 0x7FFFFF
                    if mod1 & VectorUnit.SFPSETEXP_MOD1_ARG_IMM:
                        exp = imm
                    else:
                        b = conv_to_uint32(self.lregs[vb][lane])
                        if mod1 & VectorUnit.SFPSETEXP_MOD1_ARG_EXPONENT:
                            exp = (b >> 23) & 0xFF
                        else:
                            exp = b & 0xFF
                    self.lregs[vd][lane] = (sign << 31) | (exp << 23) | man

    def handle_sfpsetman(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm12 = instr_args["imm12_math"] & 0xFFF
        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = setman(lreg[{vc}])")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    c = conv_to_uint32(self.lregs[vc][lane])
                    sign = c >> 31
                    exp = (c >> 23) & 0xFF
                    if mod1 & VectorUnit.SFPSETMAN_MOD1_ARG_IMM:
                        man = (imm12 << 11) & 0x7FFFFF
                    else:
                        b = conv_to_uint32(self.lregs[vb][lane])
                        man = b & 0x7FFFFF
                    self.lregs[vd][lane] = (sign << 31) | (exp << 23) | man

    def handle_sfpshft(self, instruction_info, issue_thread, instr_args):
        # Wormhole reserves every instr_mod1 bit above bit 0 (shift amount from
        # the immediate rather than from VC), so it is masked away there.
        # Blackhole adds bit 1 (right shifts become arithmetic) and bit 2 (the
        # value being shifted comes from VC rather than VD) — per ttsim's
        # ``TT_ARCH_VERSION >= 1`` branch of ``TENSIX_EXECUTE_SFPSHFT``.
        mod1 = instr_args["instr_mod1"] & (7 if self.backend.blackhole else 1)
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm12 = instr_args["imm12_math"] & 0xFFF
        if imm12 >= 0x800:
            imm12 -= 0x1000  # sign-extend 12-bit immediate
        vb = vc if (mod1 & VectorUnit.SFPSHFT_MOD1_SRC_LREG_C) else vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            amount = (
                imm12 if (mod1 & VectorUnit.SFPSHFT_MOD1_ARG_IMM) else f"lreg[{vc}]"
            )
            print(f"SFPU: lreg[{vd}] = lreg[{vb}] shift {amount}")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    if mod1 & VectorUnit.SFPSHFT_MOD1_ARG_IMM:
                        shift_amount = imm12
                    else:
                        u = conv_to_uint32(self.lregs[vc][lane])
                        shift_amount = u - 0x100000000 if u >= 0x80000000 else u
                    b = conv_to_uint32(self.lregs[vb][lane])
                    if shift_amount >= 0:
                        result = (b << (shift_amount & 31)) & 0xFFFFFFFF
                    elif mod1 & VectorUnit.SFPSHFT_MOD1_ARITHMETIC:
                        result = (
                            conv_to_int32(b) >> ((-shift_amount) & 31)
                        ) & 0xFFFFFFFF
                    else:
                        result = b >> ((-shift_amount) & 31)
                    self.lregs[vd][lane] = result

    def handle_sfpshft2(self, instruction_info, issue_thread, instr_args):
        """Cross-lane and register-indirect shifts (SFPSHFT2).

        Port of ttsim's ``TENSIX_EXECUTE_SFPSHFT2``, which models three of the
        seven modes: 3 rotates each 8-lane subvector up by one lane, 4 does the
        same but shifts a zero into lane 0 instead of wrapping, and 5 shifts
        ``LReg[imm12]`` by the signed amount held in VC. Mode 4 is Blackhole
        only — the Wormhole silicon has a bug in it, and ttsim rejects it there.
        The remaining modes (the LReg[3:0] block copies, and mode 6's immediate
        shift amount) are not modelled by the reference either.
        """
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        imm12 = instr_args["imm12_math"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = shft2(lreg[{vc}]) mode {mod1}")

        if (
            mod1 == VectorUnit.SFPSHFT2_MOD1_SUBVEC_SHFLSHR1
            and not self.backend.blackhole
        ):
            raise NotImplementedError(
                "SFPSHFT2 SUBVEC_SHFLSHR1 (instr_mod1 4) must not be used on "
                "Wormhole, where it is broken in hardware"
            )
        if mod1 not in (
            VectorUnit.SFPSHFT2_MOD1_SUBVEC_SHFLROR1,
            VectorUnit.SFPSHFT2_MOD1_SUBVEC_SHFLSHR1,
            VectorUnit.SFPSHFT2_MOD1_SHFT_LREG,
        ):
            raise NotImplementedError(f"SFPSHFT2 instr_mod1 {mod1} is not modelled")

        if not (vd < 8 or vd == 16):
            return

        if mod1 == VectorUnit.SFPSHFT2_MOD1_SHFT_LREG:
            # imm12 names the LReg holding the value to shift; VC holds a signed
            # per-lane shift amount (negative shifts right, logically).
            assert imm12 < 16, f"SFPSHFT2 source register {imm12} out of range"
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    amount = conv_to_int32(self.lregs[vc][lane])
                    b = conv_to_uint32(self.lregs[imm12][lane])
                    if amount >= 0:
                        b = (b << (amount & 31)) & 0xFFFFFFFF
                    else:
                        b = b >> ((-amount) & 31)
                    self.lregs[vd][lane] = b
            return

        # Snapshot VC before writing: VD and VC are often the same register, and
        # every lane reads its neighbour's pre-shift value (ttsim copies the
        # source register first for exactly this reason).
        src = [conv_to_uint32(self.lregs[vc][lane]) for lane in range(32)]
        for lane in range(32):
            if not self.isLaneEnabled(lane):
                continue
            if lane & 7:
                self.lregs[vd][lane] = src[lane - 1]
            elif mod1 == VectorUnit.SFPSHFT2_MOD1_SUBVEC_SHFLROR1:
                self.lregs[vd][lane] = src[lane + 7]  # rotate within the subvector
            else:
                self.lregs[vd][lane] = 0  # zero fill

    def handle_addi(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        imm16 = instr_args["imm16_math"]
        vc = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = {hex(imm16)} + lreg[{vc}]")
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    # ttsim SFPADDI: fma(imm<<16, 1.0, dst) = imm + dst.
                    d = self.fma(
                        imm16 << 16,
                        0x3F800000,
                        self._as_fp32_bits(self.lregs[vc][lane]),
                    )
                    if (mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VD) and vd != 16:
                        vd = self.lregs[7][lane] & 15
                    else:
                        vd = vd
                    if vd < 8 or vd == 16:
                        self.lregs[vd][lane] = d

    def handle_muli(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        imm16 = instr_args["imm16_math"]
        vc = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = {hex(imm16)} * lreg[{vc}]")

        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    # ttsim SFPMULI: fma(imm<<16, dst, 0) = imm * dst.
                    d = self.fma(
                        imm16 << 16, self._as_fp32_bits(self.lregs[vc][lane]), 0
                    )
                    if (mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VD) and vd != 16:
                        vd = self.lregs[7][lane] & 15
                    else:
                        vd = vd
                    if vd < 8 or vd == 16:
                        self.lregs[vd][lane] = d

    @staticmethod
    def _as_fp32(value):
        """Interpret an LReg lane as the FP32 *value* the float ALU sees.

        Lanes hold uint32 bit patterns, so this is a reinterpretation, not a
        conversion — using the pattern as an integer would turn ``1.4427`` into
        ``1069738555``. Only the comparison ops need the value; everything
        arithmetic goes through ``fma`` on the bits.
        """
        return conv_to_float(value & 0xFFFFFFFF)

    @staticmethod
    def _as_fp32_bits(value):
        """The FP32 bit-pattern (uint32) of an LReg lane, for the bit-exact
        float ALU (``fma``)."""
        return value & 0xFFFFFFFF

    def perform_mad(self, va, vb, vc, vd, mod1):
        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{va}] * lreg[{vb}] + lreg[{vc}]")
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    va = (
                        self.lregs[7][lane] & 15
                        if mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VA
                        else va
                    )
                    # Bit-exact hardware FMA. On Blackhole mod1 bit0 negates the
                    # multiply operand a and bit1 negates the addend c (SFPMAD;
                    # also gives SFPADD its -1.0 constant, and bit1 is never set
                    # for SFPMUL). Wormhole reserves both bits — ttsim rejects a
                    # non-zero instr_mod1 there — so the negation is gated.
                    a_bits = self._as_fp32_bits(self.lregs[va][lane])
                    b_bits = self._as_fp32_bits(self.lregs[vb][lane])
                    c_bits = self._as_fp32_bits(self.lregs[vc][lane])
                    if self.backend.blackhole:
                        if mod1 & 1:
                            a_bits ^= 0x80000000
                        if mod1 & 2:
                            c_bits ^= 0x80000000
                    d = self.fma(a_bits, b_bits, c_bits)
                    if (mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VD) and vd != 16:
                        vd = self.lregs[7][lane] & 15
                    else:
                        vd = vd
                    if vd < 8 or vd == 16:
                        self.lregs[vd][lane] = d

    def handle_mad(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        va = instr_args["lreg_src_a"]

        self.perform_mad(va, vb, vc, vd, mod1)

    def handle_add(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        va = instr_args["lreg_src_a"]

        va = 10  # hardcoded to be lanes containing 1.0
        self.perform_mad(va, vb, vc, vd, mod1)

    def handle_mul(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        va = instr_args["lreg_src_a"]

        vc = 9  # hardcoded to be lanes containing 0
        self.perform_mad(va, vb, vc, vd, mod1)

    @staticmethod
    def _lut8_to_fp32(x):
        """SFPLUT's ``Lut8ToFp32``: an 8-bit table entry as an FP32 bit pattern.

        Sign in bit 7, a 3-bit *negated* exponent offset in bits 6:4 and 4
        mantissa bits in bits 3:0, so the representable magnitudes run from
        2**-7 to just under 2. 0xFF is the reserved "zero" encoding. Verbatim
        from the ISA docs (and ttsim's ``lut8_to_fp32``).
        """
        if x == 0xFF:
            return 0
        return ((x >> 7) << 31) | ((127 - ((x >> 4) & 7)) << 23) | ((x & 0xF) << 19)

    @staticmethod
    def _lut16_to_fp32(x):
        """SFPLUTFP32's ``Lut16ToFp32``: an FP16 table entry widened to FP32.

        A plain FP16 -> FP32 widening except that the all-ones exponent (which
        would be Inf/NaN) maps to exponent zero, i.e. to +/-0 rather than to
        Inf. Verbatim from the ISA docs (and ttsim's ``lut16_to_fp32``).
        """
        exp = (x >> 10) & 0x1F
        return (
            ((x >> 15) << 31)
            | ((0 if exp == 0x1F else 112 + exp) << 23)
            | ((x & 0x3FF) << 13)
        )

    def _lut_fma(self, a, b, c, sign, sign_retain):
        """``a * b + c`` for the LUT ops, through this arch's hardware FMA.

        Every operand is an FP32 bit pattern and the result is returned as bits,
        exactly as ttsim does it. ``sign_retain`` replaces the result's sign bit
        with ``sign`` (LReg[3]'s sign bit), which is the docs' ``copysignf(d, l3)``.
        """
        d = self.fma(a, b, c)
        if sign_retain:
            d = (d & 0x7FFFFFFF) | sign
        return d

    def _lut_write_lane(self, vd, lane, indirect, value):
        """Store one LUT result, honouring the shared INDIRECT_VD modifier."""
        if indirect and vd != 16:
            vd = conv_to_uint32(self.lregs[7][lane]) & 15
        if vd < 8 or vd == 16:
            self.lregs[vd][lane] = value

    def _lut_lane_active(self, vd, lane):
        return (
            vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD)
        ) and self.isLaneEnabled(lane)

    def handle_sfplut(self, instruction_info, issue_thread, instr_args):
        """SFPLUT: piecewise-linear evaluation from three 8-bit coefficient pairs.

        Verbatim port of the functional model in
        ``{WormholeB0,BlackholeA0}/TensixTile/TensixCoprocessor/SFPLUT.md``
        (identical on both architectures, and matching ttsim's
        ``TENSIX_EXECUTE_SFPLUT``). ``LReg[0..2]`` each hold a packed
        (multiplier, addend) pair selected by which of |LReg[3]| < 1, < 2 or >=
        2 holds; the result is ``a * |LReg[3]| + c``.

        This is what the *approximate* ``tanh_tile`` / ``tanh_derivative`` /
        ``sigmoid_tile`` compile down to — sfpi's ``lut()`` / ``lut_sign()``
        emit it directly rather than via a ``TTI_SFPLUT`` macro, so it is easy
        to miss in the LLK sources. (Their default FP32 paths take a polynomial
        exp plus a Newton reciprocal instead and never reach a LUT.)
        """
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lut8(lreg[0:3], lreg[3]) mod {mod0}")

        # Bits 0 and 1 are reserved (the ISA docs define only SGN_RETAIN and
        # INDIRECT_VD, and ttsim rejects every modifier other than SGN_RETAIN),
        # so there is nothing to port for them.
        if mod0 & 3:
            raise NotImplementedError(
                f"SFPLUT instr_mod0 {mod0} sets a reserved bit (only "
                "SGN_RETAIN (4) and INDIRECT_VD (8) are defined)"
            )

        for lane in range(32):
            if not self._lut_lane_active(vd, lane):
                continue
            l3 = self._as_fp32_bits(self.lregs[3][lane])
            b = l3 & 0x7FFFFFFF  # absolute value
            if b < VectorUnit.SFPLUT_RANGE_1_0:
                coeffs = self._as_fp32_bits(self.lregs[0][lane])
            elif b < VectorUnit.SFPLUT_RANGE_2_0:
                coeffs = self._as_fp32_bits(self.lregs[1][lane])
            else:
                coeffs = self._as_fp32_bits(self.lregs[2][lane])
            a = self._lut8_to_fp32((coeffs >> 8) & 0xFF)
            c = self._lut8_to_fp32(coeffs & 0xFF)
            d = self._lut_fma(
                a, b, c, l3 & 0x80000000, mod0 & VectorUnit.SFPLUT_MOD0_SGN_RETAIN
            )
            self._lut_write_lane(vd, lane, mod0 & VectorUnit.SFPLUT_MOD0_INDIRECT_VD, d)

    def handle_sfplutfp32(self, instruction_info, issue_thread, instr_args):
        """SFPLUTFP32: the wider-precision / finer-grained sibling of SFPLUT.

        Verbatim port of the functional model in
        ``{WormholeB0,BlackholeA0}/TensixTile/TensixCoprocessor/SFPLUTFP32.md``
        (identical on both architectures). Same ``a * |LReg[3]| + c`` shape as
        SFPLUT, but the coefficients come from ``LReg[0..2]`` (multipliers) and
        ``LReg[4..6]`` (addends) at full FP32, or as FP16 halves giving a 3- or
        6-entry table. ttsim only models the 6-entry table with the 3.0 cut, but
        the ISA docs give the complete decode, so every modifier is ported here
        — tt-llk's ``_calculate_sigmoid_`` emits the 4.0-cut table with
        SGN_RETAIN (Mod1 = 7), which ttsim declines.
        """
        mod1 = instr_args["instr_mod1"]
        # SFPLUTFP32 declares only two fields, so the shared instruction table
        # hands ``lreg_dest`` everything above bit 3 rather than the real 7:4
        # (ttsim's data/{bh,wh} agree it is 4 bits wide).
        vd = instr_args["lreg_dest"] & 0xF

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lut32(lreg[0:3], lreg[4:7], lreg[3]) mod {mod1}")

        fp16_tables = mod1 & VectorUnit.SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1
        three_entry = (
            mod1 & VectorUnit.SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE
        ) == VectorUnit.SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE
        cut = (
            VectorUnit.SFPLUT_RANGE_4_0
            if (mod1 & VectorUnit.SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2)
            == VectorUnit.SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2
            else VectorUnit.SFPLUT_RANGE_3_0
        )

        for lane in range(32):
            if not self._lut_lane_active(vd, lane):
                continue
            l3 = self._as_fp32_bits(self.lregs[3][lane])
            b = l3 & 0x7FFFFFFF  # absolute value
            if b < VectorUnit.SFPLUT_RANGE_1_0:
                i = 0
            elif b < VectorUnit.SFPLUT_RANGE_2_0:
                i = 1
            else:
                i = 2

            if fp16_tables:
                if three_entry:
                    # One LReg holds both halves of the pair for this range.
                    entry = self._as_fp32_bits(self.lregs[i][lane])
                    a = self._lut16_to_fp32((entry >> 16) & 0xFFFF)
                    c = self._lut16_to_fp32(entry & 0xFFFF)
                else:
                    # Each range is split in two, the halfword selected by j.
                    if b < VectorUnit.SFPLUT_RANGE_0_5:
                        j = 0
                    elif b < VectorUnit.SFPLUT_RANGE_1_0:
                        j = 16
                    elif b < VectorUnit.SFPLUT_RANGE_1_5:
                        j = 0
                    elif b < VectorUnit.SFPLUT_RANGE_2_0:
                        j = 16
                    elif b < cut:
                        j = 0
                    else:
                        j = 16
                    a = self._lut16_to_fp32(
                        (self._as_fp32_bits(self.lregs[0 + i][lane]) >> j) & 0xFFFF
                    )
                    c = self._lut16_to_fp32(
                        (self._as_fp32_bits(self.lregs[4 + i][lane]) >> j) & 0xFFFF
                    )
            else:
                a = self._as_fp32_bits(self.lregs[0 + i][lane])
                c = self._as_fp32_bits(self.lregs[4 + i][lane])

            d = self._lut_fma(
                a, b, c, l3 & 0x80000000, mod1 & VectorUnit.SFPLUTFP32_MOD1_SGN_RETAIN
            )
            self._lut_write_lane(
                vd, lane, mod1 & VectorUnit.SFPLUTFP32_MOD1_INDIRECT_VD, d
            )

    def _transpose4(self, base):
        """Transpose the 4x4 matrix each of the 8 columns of ``LReg[base:base+4]``
        forms. Verbatim from SFPTRANSP.md's ``Transpose4`` (and ttsim's
        ``TENSIX_EXECUTE_SFPTRANSP``): the lane index is ``row * 8 + column``, so
        the movement is purely within a column — this is *not* a transpose of the
        4x8 grid. Both halves of each swap read the pre-swap values, and each
        write is gated on its own lane's enable."""
        for column in range(8):
            for i in range(4):
                for j in range(i):
                    ij = self.lregs[base + i][j * 8 + column]
                    ji = self.lregs[base + j][i * 8 + column]
                    if self.isLaneEnabled(j * 8 + column):
                        self.lregs[base + i][j * 8 + column] = ji
                    if self.isLaneEnabled(i * 8 + column):
                        self.lregs[base + j][i * 8 + column] = ij

    def handle_sfptransp(self, instruction_info, issue_thread, instr_args):
        # SFPTRANSP transposes LReg[0:4] and LReg[4:8] independently, used by the
        # topk / welfords / ema / binary-broadcast SFPU kernels to get at data
        # held in a different lane. Identical on both architectures.
        vd = instr_args["lreg_dest"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print("SFPU: transpose lreg[0:4] and lreg[4:8]")

        # The VD / backdoor-load gate is on the instruction as a whole rather
        # than per lane (SFPTRANSP.md wraps the two Transpose4 calls in it), so
        # it is the same "any lane permits it" test SFPPUSHC / SFPPOPC use.
        do_transpose = any(
            vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD)
            for lane in range(32)
        )
        if do_transpose:
            self._transpose4(0)
            self._transpose4(4)

    def handle_sfploadmacro(self, instruction_info, issue_thread, instr_args):
        # SFPLOADMACRO is an SFPLOAD that additionally schedules up to four
        # previously-configured vector instructions across the SFPU sub-units,
        # with per-sub-unit delays. tt-sim's backend has no notion of SFPU
        # sub-units or of deferred issue, and the reference simulator declines
        # to model the op at all ("explicitly out of scope"), so there is no
        # oracle to port and nothing here is invented. Fail loudly with the
        # workaround tt-metal itself ships. Note the op still *decodes*, and its
        # Blackhole ``sfpu_addr_mode`` shift is handled by _read_sfpu_addr_mode.
        raise NotImplementedError(
            "SFPLOADMACRO is not modelled (its macro-scheduling semantics are "
            "out of scope for the reference simulator too); set "
            "TT_METAL_DISABLE_SFPLOADMACRO=1 in the host environment to stop "
            "tt-metal emitting it"
        )

    def handle_sfpcompc(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]

        do_compc = False
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                do_compc = True
                break

        if do_compc:
            if len(self.flagStack) == 0:
                # With nothing pushed, SFPCOMPC is a plain inversion of the lane
                # flags (ttsim: `cc = ~cc` when cc_sp == 0). The per-lane loop
                # below indexes both halves of `top`, so they must be vectors --
                # scalars here raise TypeError the moment a kernel complements
                # an empty stack, which `tanh_tile` and `where_tile` both do.
                top = ([True] * 32, [True] * 32)
            else:
                top = self.flagStack[-1]

            # Note we are doing this on a lane by lane basis, whereas have implemented
            # popc and pushc across all lanes
            for lane in range(32):
                # Invert laneFlags, subject to top
                if top[1][lane] and self.useLaneFlagsForLaneEnable[lane]:
                    self.laneFlags[lane] = top[0][lane] and (not self.laneFlags[lane])
                else:
                    self.laneFlags[lane] = False

    def handle_sfppopc(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]

        do_pop = False
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                do_pop = True
                break

        if do_pop:
            if len(self.flagStack) == 0:
                # Both halves are indexed per lane, so they must be vectors.
                top = ([False] * 32, [False] * 32)
            else:
                top = self.flagStack[-1]

            if mod1 == 0:
                # Plain pop from stack
                assert len(self.flagStack) > 0
                self.flagStack.pop()
            elif len(self.flagStack) == 8:
                self.flagStack[7] = top

            if mod1 == 0:
                # Set LaneFlags and UseLaneFlagsForLaneEnable to Top
                self.laneFlags = list(top[0])
                self.useLaneFlagsForLaneEnable = list(top[1])
            elif mod1 <= 12:
                # Mutate LaneFlags and UseLaneFlagsForLaneEnable based on Top
                self.laneFlags = list(self.booleanOp(mod1, self.laneFlags, top[0]))
                self.useLaneFlagsForLaneEnable = list(top[1])
            elif mod1 == 13:
                # Just invert laneFlags
                self.laneFlags = [not v for v in self.laneFlags]
            elif mod1 == 14:
                # Set laneFlags and useLaneFlagsForLaneEnable to constants
                self.laneFlags = self.useLaneFlagsForLaneEnable = [
                    True for _ in range(32)
                ]
            elif mod1 == 15:
                # Set LaneFlags and UseLaneFlagsForLaneEnable to constants
                self.useLaneFlagsForLaneEnable = [True for _ in range(32)]
                self.laneFlags = [False for _ in range(32)]

    def booleanOp(self, mod1, A_list, B_list):
        for A, B in zip(A_list, B_list):
            match mod1:
                case 1:
                    yield B
                case 2:
                    yield not B
                case 3:
                    yield A and B
                case 4:
                    yield A or B
                case 5:
                    yield A and (not B)
                case 6:
                    yield A or (not B)
                case 7:
                    yield (not A) and B
                case 8:
                    yield (not A) or B
                case 9:
                    yield (not A) and (not B)
                case 10:
                    yield (not A) or (not B)
                case 11:
                    yield A != B
                case 12:
                    yield A == B

    def handle_sfppushc(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]
        do_push = False
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                do_push = True
                break

        if do_push:
            assert len(self.flagStack) < 8
            # Snapshot by value: SFPSETCC and friends mutate these lists in
            # place, which would otherwise rewrite the entry already pushed.
            # The stack is LIFO -- pushed at the end, read and popped from the
            # end (ttsim writes at cc_sp then increments, and pops the reverse).
            self.flagStack.append(
                (list(self.laneFlags), list(self.useLaneFlagsForLaneEnable))
            )

    def handle_sfpsetcc(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm1 = instr_args["imm12_math"] & 0x1

        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(
                    lane
                ):  # Is this correct? Seems strange that can not reenable
                    if not self.useLaneFlagsForLaneEnable[lane]:
                        self.laneFlags[lane] = False
                    elif mod1 & VectorUnit.SFPSETCC_MOD1_CLEAR:
                        self.laneFlags[lane] = False
                    elif mod1 & VectorUnit.SFPSETCC_MOD1_IMM_BIT0:
                        self.laneFlags[lane] = imm1 != 0
                    else:
                        # Lanes are raw uint32 bit patterns, and ttsim compares
                        # them as `int32_t src = LReg[c]` -- i.e. for a float
                        # lane the FP32 sign bit is what decides `< 0`. An
                        # unsigned Python int is never negative, so without this
                        # every `v_if(x < 0)` (the Newton-Raphson step in
                        # sfpu_reciprocal_iter, among others) silently disabled
                        # all 32 lanes.
                        c = conv_to_int32(self.lregs[vc][lane])
                        match mod1:
                            case VectorUnit.SFPSETCC_MOD1_LREG_LT0:
                                self.laneFlags[lane] = c < 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_NE0:
                                self.laneFlags[lane] = c != 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_GTE0:
                                self.laneFlags[lane] = c >= 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_EQ0:
                                self.laneFlags[lane] = c == 0

    @staticmethod
    def _sign_mag_total_order(value):
        """Map a lane's 32-bit sign-magnitude pattern onto a monotonic int32.

        The Blackhole comparisons order lanes as sign-magnitude integers rather
        than as IEEE floats: a negative pattern is flipped (``x ^ 0x7fffffff``)
        so that ordinary integer ``<``/``>`` reproduces the hardware's ordering,
        including ``-0 < +0``. Mirrors ttsim's ``sign_mag32_total_order``.
        """
        value &= 0xFFFFFFFF
        if value & 0x80000000:
            value ^= 0x7FFFFFFF
        return conv_to_int32(value)

    def _compare_lanes(self, name, mod1, vd, vc, compare):
        """Shared body of the Blackhole SFPGT / SFPLE comparisons.

        Per BlackholeA0/.../VectorUnit.md, ``instr_mod1`` picks where the result
        of comparing ``LReg[lreg_dest]`` (VD) against ``LReg[lreg_c]`` (VC)
        lands: mod1 1 updates ``LaneFlags``, mod1 8 writes an all-ones / all-zero
        mask into VD (leaving LaneFlags alone). The mask form is what every stock
        Blackhole ``exp_tile`` emits — it masks the integer part before SFPSETEXP
        instead of clamping with an SFPSWAP — so dropping it silently produced a
        garbage exponent. Any other modifier is undefined; ttsim rejects it, and
        so do we rather than guess.
        """
        if mod1 not in (
            VectorUnit.SFPCMP_MOD1_SET_LANE_FLAGS,
            VectorUnit.SFPCMP_MOD1_SET_VD,
        ):
            raise NotImplementedError(
                f"{name} with instr_mod1={mod1} is not modelled (want 1 or 8)"
            )
        set_vd = mod1 == VectorUnit.SFPCMP_MOD1_SET_VD
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    result = compare(
                        self._sign_mag_total_order(self.lregs[vd][lane]),
                        self._sign_mag_total_order(self.lregs[vc][lane]),
                    )
                    if set_vd:
                        if vd < 8 or vd == 16:
                            self.lregs[vd][lane] = 0xFFFFFFFF if result else 0
                    else:
                        self.laneFlags[lane] = result

    def handle_sfpgt(self, instruction_info, issue_thread, instr_args):
        # Blackhole SFPGT: (VD > VC) into LaneFlags (mod1 1) or VD (mod1 8).
        self._compare_lanes(
            "SFPGT",
            instr_args["instr_mod1"],
            instr_args["lreg_dest"],
            instr_args["lreg_c"],
            lambda d, c: d > c,
        )

    def handle_sfple(self, instruction_info, issue_thread, instr_args):
        # Blackhole SFPLE: (VD <= VC) into LaneFlags (mod1 1) or VD (mod1 8).
        self._compare_lanes(
            "SFPLE",
            instr_args["instr_mod1"],
            instr_args["lreg_dest"],
            instr_args["lreg_c"],
            lambda d, c: d <= c,
        )

    def handle_sfpmul24(self, instruction_info, issue_thread, instr_args):
        """Blackhole SFPMUL24: 24-bit integer multiply.

        Two's-complement mode, per BlackholeA0/.../VectorUnit.md:
        ``VD = (VA * VB) & 0x7fffff`` — the low 23 bits of the product of the low
        23 bits of each source. (The ``>> 23`` high-bits variant is a separate
        mode not modelled here.)
        """
        vd = instr_args["lreg_dest"]
        va = instr_args["lreg_src_a"]
        vb = instr_args["lreg_src_b"]
        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    a = conv_to_uint32(self.lregs[va][lane]) & 0x7FFFFF
                    b = conv_to_uint32(self.lregs[vb][lane]) & 0x7FFFFF
                    self.lregs[vd][lane] = (a * b) & 0x7FFFFF

    # Reciprocal-mantissa lookup table for SFPARECIP (128 entries), a verbatim
    # port of ttsim's data/bh reference (src/tensix.cpp approx_recip).
    ARECIP_LUT = (
        127,
        125,
        123,
        121,
        119,
        117,
        116,
        114,
        112,
        110,
        109,
        107,
        105,
        104,
        102,
        100,
        99,
        97,
        96,
        94,
        93,
        91,
        90,
        88,
        87,
        85,
        84,
        83,
        81,
        80,
        79,
        77,
        76,
        75,
        74,
        72,
        71,
        70,
        69,
        68,
        66,
        65,
        64,
        63,
        62,
        61,
        60,
        59,
        58,
        57,
        56,
        55,
        54,
        53,
        52,
        51,
        50,
        49,
        48,
        47,
        46,
        45,
        44,
        43,
        42,
        41,
        40,
        40,
        39,
        38,
        37,
        36,
        35,
        35,
        34,
        33,
        32,
        31,
        31,
        30,
        29,
        28,
        28,
        27,
        26,
        25,
        25,
        24,
        23,
        23,
        22,
        21,
        21,
        20,
        19,
        19,
        18,
        17,
        17,
        16,
        15,
        15,
        14,
        14,
        13,
        12,
        12,
        11,
        11,
        10,
        9,
        9,
        8,
        8,
        7,
        7,
        6,
        5,
        5,
        4,
        4,
        3,
        3,
        2,
        2,
        1,
        1,
        0,
    )

    @staticmethod
    def _approx_recip(x):
        # x is the FP32 magnitude (bits 30:0). Port of ttsim approx_recip:
        # exponent 253 - e reflects the ~1/x scaling, mantissa from the LUT.
        if x < 0x800000:  # zero / denormal -> +inf
            return 0x7F800000
        elif x < 0x7E800000:  # x < 2**126
            return ((253 - (x >> 23)) << 23) | (
                VectorUnit.ARECIP_LUT[(x >> 16) & 0x7F] << 16
            )
        else:
            return 0

    def handle_sfparecip(self, instruction_info, issue_thread, instr_args):
        # Blackhole approximate reciprocal: sign preserved, magnitude replaced by
        # the LUT-based 1/x approximation. Mirrors ttsim's SFPARECIP, which only
        # models the base variant (no instruction modifier / immediate).
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        assert instr_args["instr_mod1"] == 0, "SFPARECIP instr_mod1 not modelled"
        assert instr_args["imm12_math"] == 0, "SFPARECIP imm12_math not modelled"
        if vd < 8:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    x = conv_to_uint32(self.lregs[vc][lane])
                    self.lregs[vd][lane] = (x & 0x80000000) | self._approx_recip(
                        x & 0x7FFFFFFF
                    )

    def handle_sfpencc(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        imm2 = instr_args["imm12_math"]

        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
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

    def handle_sfpiadd(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        # The immediate is a 12-bit two's-complement value, so it must be sign
        # extended before use: sfpi's `sfpiadd_i(v, -255, ...)` (in the FP32 exp
        # kernel's overflow check) arrives as 0xF01 and adding it raw both
        # computes v+3841 and leaves the lane flag stuck at "not negative".
        imm12 = instr_args["imm12_math"] & 0xFFF
        if imm12 >= 0x800:
            imm12 -= 0x1000

        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vc}] + lreg[{vb}]")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    # Every lane is a raw uint32 bit pattern, so read the
                    # operands as such (ttsim's TENSIX_EXECUTE_SFPIADD).
                    c = self.lregs[vc][lane]
                    b = self.lregs[vb][lane]

                    if mod1 & VectorUnit.SFPIADD_MOD1_ARG_IMM:
                        result = c + imm12
                    elif mod1 & VectorUnit.SFPIADD_MOD1_ARG_2SCOMP_LREG_DST:
                        result = c - b
                    else:
                        result = c + b

                    # The add wraps, and "negative" is bit 31 of the wrapped
                    # result (ttsim's `src & 0x80000000`). Testing an unsigned
                    # Python int for `< 0` never fires.
                    result &= 0xFFFFFFFF
                    negative = bool(result & 0x80000000)
                    self.lregs[vd][lane] = result

                    if vd < 8:
                        # Mod1 bit 3 (CC_GTE0) wins over bit 2 (CC_NONE), which
                        # in turn leaves LaneFlags alone.
                        if mod1 & VectorUnit.SFPIADD_MOD1_CC_GTE0:
                            self.laneFlags[lane] = not negative
                        elif not (mod1 & VectorUnit.SFPIADD_MOD1_CC_NONE):
                            self.laneFlags[lane] = negative

    def _read_sfpu_addr_mode(self, instruction_info, instr_args):
        # SFPLOAD/SFPSTORE/SFPLOADMACRO ``sfpu_addr_mode`` is 3 bits on Blackhole
        # (raw bits 15:13) but only 2 bits on Wormhole (15:14). The shared
        # instruction table encodes the Wormhole field, so on Blackhole its low
        # bit (13) is dropped and the value shifts. Read the true 3-bit field
        # from the raw word on Blackhole, mirroring the matrix unit's addr_mode
        # handling. See docs/plans/blackhole-support.md.
        if self.backend.blackhole:
            return extract_bits(instruction_info["raw_instruction"], 3, 13)
        return instr_args["sfpu_addr_mode"]

    @staticmethod
    def _read_dest_reg_addr(instr_args):
        # SFPLOAD/SFPSTORE ``dest_reg_addr`` is the ISA's ``imm10`` — bits 9:0,
        # on *both* architectures. The shared instruction table only carries
        # each field's start bit and infers its width from the next field's, so
        # ``dest_reg_addr`` (start 0) is decoded as bits 13:0, swallowing the
        # three reserved bits 12:10 and, on Blackhole, bit 13 — which there
        # belongs to the 3-bit ``sfpu_addr_mode`` (15:13, see
        # ``_read_sfpu_addr_mode``). A Blackhole SFPU addr_mode of 4..7 therefore
        # leaked 0x2000 into the Dst row address, putting every stock bfloat16-Dst
        # ``init_sfpu`` kernel thousands of rows past the end of Dst. Wormhole's
        # addr_mode is 15:14, so nothing ever set those bits there. Mask to the
        # documented width (ttsim's ``tensix_isa.json`` says ``"9:0"`` for both
        # arches).
        return instr_args["dest_reg_addr"] & 0x3FF

    def get_dst_address(self, issue_thread, mod0, imm10):
        stateID = self.backend.getThreadConfigValue(
            issue_thread, "CFG_STATE_ID_StateID"
        )

        if mod0 == VectorUnit.MOD0_FMT_SRCB:
            if self.getConfigValue(stateID, "ALU_ACC_CTRL_SFPU_Fp32_enabled"):
                # Functionally identical to MOD0_FMT_INT32
                mod0 = VectorUnit.MOD0_FMT_FP32
            else:
                srcBFmt = (
                    self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcB_val")
                    if self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG_SrcB_override")
                    else self.getConfigValue(stateID, "ALU_FORMAT_SPEC_REG1_SrcB")
                )
                if srcBFmt in [
                    DataFormat.FP32,
                    DataFormat.TF32,
                    DataFormat.BF16,
                    DataFormat.BF16,
                    DataFormat.BFP4,
                    DataFormat.BFP2,
                    DataFormat.INT32,
                    DataFormat.UINT16,
                ]:
                    mod0 = VectorUnit.MOD0_FMT_BF16
                else:
                    mod0 = VectorUnit.MOD0_FMT_FP16

        # Apply various Dst address adjustments.
        # The top 8 bits of Addr end up selecting an aligned group of four rows of Dst, the
        # next bit selects between even and odd columns, and the low bit goes unused.

        addr = imm10 + self.backend.getThreadConfigValue(
            issue_thread, "DEST_TARGET_REG_CFG_MATH_Offset"
        )
        if mod0 == VectorUnit.MOD0_FMT_INT32_ALL:
            addr += (
                self.backend.getRWC(issue_thread).Dst
                + self.getConfigValue(stateID, "DEST_REGW_BASE_Base")
                & 3
            )
        else:
            addr += self.backend.getRWC(issue_thread).Dst + self.getConfigValue(
                stateID, "DEST_REGW_BASE_Base"
            )

        return addr, mod0

    def handle_sfpstore(self, instruction_info, issue_thread, instr_args):
        imm10 = self._read_dest_reg_addr(instr_args)
        addrmod = self._read_sfpu_addr_mode(instruction_info, instr_args)
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]

        addr, mod0 = self.get_dst_address(issue_thread, mod0, imm10)

        if self.getDiagnosticSettings().reportSFPUCalculations():
            if addr & 2:
                col_start = 1
            else:
                col_start = 0
            print(
                f"SFPU: store lreg[{vd}] into between dst[{(addr & ~3)}, {col_start}] and dst"
                f"[{(addr & ~3) + int(31 / 8)}, X] from thread{issue_thread}"
            )

        for lane in range(32):
            if self.laneConfigValue(lane, VectorUnit.BLOCK_SFPU_RD_FROM_DEST):
                continue
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane) or mod0 == VectorUnit.MOD0_FMT_INT32_ALL:
                    row = (addr & ~3) + int(lane / 8)
                    column = (lane & 7) * 2
                    if addr & 2 or self.laneConfigValue(
                        lane & 7, VectorUnit.DEST_RD_COL_EXCHANGE
                    ):
                        column += 1

                    datum = self.lregs[vd][lane]
                    match mod0:
                        case VectorUnit.MOD0_FMT_FP16:
                            write_val = DataFormatConversions.FP16ToDstFormatFP16(
                                DataFormatConversions.FP32ToFP16(conv_to_uint32(datum))
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_BF16:
                            write_val = DataFormatConversions.BF16ToDstFormatBF16(
                                DataFormatConversions.FP32ToBF16(conv_to_uint32(datum))
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_FP32:
                            self.getDst().setDst32b(
                                row,
                                column,
                                DataFormatConversions.FP32ToDstFormatFP32(
                                    conv_to_uint32(datum)
                                ),
                            )
                        case VectorUnit.MOD0_FMT_INT32 | VectorUnit.MOD0_FMT_INT32_ALL:
                            # INT32 stored verbatim (mirrors the SFPLOAD case) —
                            # no float-format rearrangement for integers.
                            self.getDst().setDst32b(row, column, conv_to_uint32(datum))
                        case VectorUnit.MOD0_FMT_INT32_SM:
                            write_val = DataFormatConversions.FP32ToDstFormatFP32(
                                DataFormatConversions.toSignMag(datum)
                            )
                            self.getDst().setDst32b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_INT8:
                            write_val = DataFormatConversions.FP16ToDstFormatFP16(
                                DataFormatConversions.signMag11ToFP16(datum)
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_INT8_COMP:
                            write_val = DataFormatConversions.FP16ToDstFormatFP16(
                                DataFormatConversions.signMag11ToFP16(
                                    DataFormatConversions.ToSignMag(datum)
                                )
                            )
                            self.getDst().setDst16b(row, column, write_val)
                        case VectorUnit.MOD0_FMT_LO16_ONLY | VectorUnit.MOD0_FMT_UINT16:
                            self.getDst().setDst16b(row, column, datum & 0xFFFF)
                        case VectorUnit.MOD0_FMT_HI16_ONLY:
                            self.getDst().setDst16b(row, column, datum >> 16)
                        case VectorUnit.MOD0_FMT_INT16:
                            self.getDst().setDst16b(
                                row, column, ((datum >> 31) << 15) | (datum & 0x7FFF)
                            )
                        case VectorUnit.MOD0_FMT_LO16:
                            self.getDst().setDst32b(
                                row, column, (datum << 16) | (datum >> 16)
                            )
                        case VectorUnit.MOD0_FMT_HI16:
                            self.getDst().setDst32b(row, column, datum)
                        case VectorUnit.MOD0_FMT_ZERO:
                            self.getDst().setDst16b(row, column, 0)
                        case _:
                            raise NotImplementedError()

        self.backend.getRWC(issue_thread).applyPartialAddrMod(issue_thread, addrmod)

    def handle_sfpload(self, instruction_info, issue_thread, instr_args):
        imm10 = self._read_dest_reg_addr(instr_args)
        addrmod = self._read_sfpu_addr_mode(instruction_info, instr_args)
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]

        addr, mod0 = self.get_dst_address(issue_thread, mod0, imm10)

        if self.getDiagnosticSettings().reportSFPUCalculations():
            if addr & 2:
                col_start = 1
            else:
                col_start = 0
            print(
                f"SFPU: load between dst[{(addr & ~3)}, {col_start}] and dst[{(addr & ~3) + int(31 / 8)}, "
                f"X]into lreg[{vd}] from thread{issue_thread}"
            )

        if vd < 8:
            for lane in range(32):
                if self.laneConfigValue(lane, VectorUnit.BLOCK_SFPU_RD_FROM_DEST):
                    continue
                if self.isLaneEnabled(lane) or mod0 == VectorUnit.MOD0_FMT_INT32_ALL:
                    row = (addr & ~3) + int(lane / 8)
                    column = (lane & 7) * 2
                    if addr & 2 or self.laneConfigValue(
                        lane & 7, VectorUnit.DEST_RD_COL_EXCHANGE
                    ):
                        column += 1

                    match mod0:
                        case VectorUnit.MOD0_FMT_FP16:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.FP16InDstToFP32(
                                rd,
                                self.laneConfigValue(lane, VectorUnit.ENABLE_FP16A_INF),
                            )
                        case VectorUnit.MOD0_FMT_BF16:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.BF16InDstToBF16(rd) << 16
                        case VectorUnit.MOD0_FMT_FP32:
                            rd = self.getDst().getDst32b(row, column)
                            datum = DataFormatConversions.FP32InDstToFP32(rd)
                        case VectorUnit.MOD0_FMT_INT32 | VectorUnit.MOD0_FMT_INT32_ALL:
                            # INT32 is stored verbatim in Dst, so load it raw. The
                            # FP32InDstToFP32 rearrangement is only for actual
                            # floats — applying it to an integer permutes its bits
                            # and corrupts every non-bit-symmetric op (add, sub,
                            # and, or), while XOR-with-a-halfword-mask survives.
                            datum = self.getDst().getDst32b(row, column)
                        case VectorUnit.MOD0_FMT_INT32_SM:
                            rd = self.getDst().getDst32b(row, column)
                            datum = DataFormatConversions.signMagToTwosComp(
                                DataFormatConversions.FP32InDstToFP32(rd)
                            )
                        case VectorUnit.MOD0_FMT_INT8:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.signMag8ToSignMag32(rd)
                        case VectorUnit.MOD0_FMT_INT8_COMP:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.signMagToTwosComp(
                                DataFormatConversions.signMag11ToSignMag32(rd)
                            )
                        case VectorUnit.MOD0_FMT_LO16_ONLY:
                            rd = self.getDst().getDst16b(row, column)
                            datum = (self.lregs[vd][lane] & 0xFFFF0000) | rd
                        case VectorUnit.MOD0_FMT_HI16_ONLY:
                            rd = self.getDst().getDst16b(row, column)
                            datum = (rd << 16) | (self.lregs[vd][lane] & 0xFFFF)
                        case VectorUnit.MOD0_FMT_HI16_ONLY:
                            rd = self.getDst().getDst16b(row, column)
                            datum = DataFormatConversions.signMag16ToSignMag32(rd)
                        case VectorUnit.MOD0_FMT_UINT16 | VectorUnit.MOD0_FMT_LO16:
                            datum = DataFormatConversions.signMag16ToSignMag32(rd)
                        case VectorUnit.MOD0_FMT_HI16:
                            datum = DataFormatConversions.signMag16ToSignMag32(rd) << 16
                        case VectorUnit.MOD0_FMT_ZERO:
                            datum = 0
                        case _:
                            raise NotImplementedError()

                    self.lregs[vd][lane] = datum
                    if (
                        (vd < 4)
                        and self.laneConfigValue(lane, VectorUnit.ENABLE_DEST_INDEX)
                        and self.laneConfigValue(
                            lane, VectorUnit.CAPTURE_DEFAULT_DEST_INDEX
                        )
                    ):
                        self.lregs[vd + 4][lane] = (row << 4) | column

        self.backend.getRWC(issue_thread).applyPartialAddrMod(issue_thread, addrmod)

    def handle_sfploadi(self, instruction_info, issue_thread, instr_args):
        mod0 = instr_args["instr_mod0"]
        vd = instr_args["lreg_ind"]
        imm16 = instr_args["imm16"]

        assert vd < 8
        for lane in range(32):
            if self.isLaneEnabled(lane):
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
                            conv_to_uint32(self.lregs[vd][lane]) & 0x0000FFFF
                        )
                    case VectorUnit.SFPLOADI_MOD0_LOWER:
                        self.lregs[vd][lane] = (
                            conv_to_uint32(self.lregs[vd][lane]) & 0xFFFF0000
                        ) | imm16
                    case _:
                        raise ValueError()

    def handle_sfpnop(self, instruction_info, issue_thread, instr_args):
        pass

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

    def isLaneEnabled(self, lane):
        if get_nth_bit(
            self.laneConfigValue(lane & 7, VectorUnit.ROW_MASK), int(lane / 8)
        ):
            return False
        elif self.useLaneFlagsForLaneEnable[lane]:
            return self.laneFlags[lane]
        else:
            return True

    def BF16toFP32(self, val):
        return val << 16

    def FP16toFP32(self, val):
        sign = val >> 15
        exp = (val >> 10) & 0x1F
        man = val & 0x3FF

        exp += 112  # Rebias 5b exponent to 8b
        return (sign << 31) | (exp << 23) | (man << 13)
