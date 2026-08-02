from tt_sim.pe.tensix.backends.backend_base import DataFormat, TensixBackendUnit
from tt_sim.pe.tensix.registers import LReg
from tt_sim.pe.tensix.util import DataFormatConversions
from tt_sim.util.bits import extract_bits, get_bits, get_nth_bit
from tt_sim.util.conversion import conv_to_float, conv_to_uint32

_M64 = 0xFFFFFFFFFFFFFFFF
_M32 = 0xFFFFFFFF


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

    def semi_sticky_shift(var, amount, mask):
        if amount >= mask.bit_length():
            return 0
        orig = var
        v = var >> amount
        if v:
            v |= 1 if (((v << amount) & mask) != orig) else 0
        return v

    r_e = p_e if p_e > z_e else z_e
    if p_e < r_e:
        p_m = semi_sticky_shift(p_m, r_e - p_e, _M64)
    if z_e < r_e:
        z_m = semi_sticky_shift(z_m, r_e - z_e, _M32)
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
                    self.lregs[vd][lane] = conv_to_float(
                        (sign << 31) | (exp << 23) | man
                    )

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
                    self.lregs[vd][lane] = conv_to_float(x) if is_float else x

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
                        x = conv_to_uint32(value) ^ 0x80000000
                        value = conv_to_float(x) if isinstance(value, float) else x
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

    def handle_sfpcast(self, instruction_info, issue_thread, instr_args):
        # Cast a sign-magnitude int32 in VC to FP32 in VD. Verbatim port of the
        # pseudocode in WormholeB0/TensixTile/TensixCoprocessor/SFPCAST.md.
        # mod1 & 1 selects stochastic rounding (seven PRNG bits) on hardware;
        # tt-sim does not model the SFPU PRNG, so both modes round to nearest
        # even here (the round-to-nearest branch below).
        mod1 = instr_args["instr_mod1"]
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            mode = "stochastic~rne" if (mod1 & 1) else "rne"
            print(f"SFPU: lreg[{vd}] = fp32(int32 lreg[{vc}]) [{mode}]")

        if not (vd < 8 or vd == 16):
            return

        for lane in range(32):
            if not self.isLaneEnabled(lane):
                continue
            c = conv_to_uint32(self.lregs[vc][lane]) & 0xFFFFFFFF
            sign = c & 0x80000000
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
            self.lregs[vd][lane] = conv_to_float(d)

    def handle_sfp_stoch_rnd(self, instruction_info, issue_thread, instr_args):
        # Round / narrow VC into VD. Verbatim port of the pseudocode in
        # WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_{FloatFloat,
        # FloatInt,IntInt}.md. mod bits [2:0] pick the conversion; bit 3
        # selects the immediate descale over src_b (int32->int8 only). rnd_mode
        # selects stochastic rounding on hardware, which needs the SFPU PRNG
        # tt-sim does not model, so both paths use the round-to-nearest
        # constant 0x400000.
        mod = instr_args["instr_mod1"]
        mode = mod & 0x7
        use_imm = bool(mod & 0x8)
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_src_c"]
        vb = instr_args["lreg_src_b"]
        imm8 = instr_args["imm8_math"]

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = stochrnd(lreg[{vc}]) mode {mode}")

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
                result = conv_to_float(x & 0xFFFFFFFF)
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
            self.lregs[vd][lane] = conv_to_float(sign | (exp << 23) | man)

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
                    self.lregs[vd][lane] = conv_to_float(
                        (sign << 31) | (exp << 23) | man
                    )

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
                    self.lregs[vd][lane] = conv_to_float(
                        (sign << 31) | (exp << 23) | man
                    )

    def handle_sfpshft(self, instruction_info, issue_thread, instr_args):
        mod1 = instr_args["instr_mod1"] & 1  # spec: high bits reserved, mask to 1
        vd = instr_args["lreg_dest"]
        vc = instr_args["lreg_c"]
        imm12 = instr_args["imm12_math"] & 0xFFF
        if imm12 >= 0x800:
            imm12 -= 0x1000  # sign-extend 12-bit immediate
        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(
                f"SFPU: lreg[{vd}] = lreg[{vb}] shift {imm12 if mod1 else f'lreg[{vc}]'}"
            )

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
                    else:
                        result = b >> ((-shift_amount) & 31)
                    self.lregs[vd][lane] = result

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
                    if self.backend.blackhole:
                        # ttsim SFPADDI: fma(imm<<16, 1.0, dst) = imm + dst.
                        d = fma_model_bh(
                            imm16 << 16,
                            0x3F800000,
                            self._as_fp32_bits(self.lregs[vc][lane]),
                        )
                    else:
                        c = self._as_fp32(self.lregs[vc][lane])
                        d = conv_to_float(self.BF16toFP32(imm16)) + c
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
                    if self.backend.blackhole:
                        # ttsim SFPMULI: fma(imm<<16, dst, 0) = imm * dst.
                        d = fma_model_bh(
                            imm16 << 16, self._as_fp32_bits(self.lregs[vc][lane]), 0
                        )
                    else:
                        c = self._as_fp32(self.lregs[vc][lane])
                        d = conv_to_float(self.BF16toFP32(imm16)) * c
                    if (mod1 & VectorUnit.SFPMAD_MOD1_INDIRECT_VD) and vd != 16:
                        vd = self.lregs[7][lane] & 15
                    else:
                        vd = vd
                    if vd < 8 or vd == 16:
                        self.lregs[vd][lane] = d

    @staticmethod
    def _as_fp32(value):
        """Interpret an LReg lane as the FP32 value the float ALU sees.

        LReg lanes hold either a Python float (already an FP32 value) or a
        raw 32-bit integer bit-pattern — e.g. a full-precision constant
        assembled by ``SFPLOADI`` UPPER/LOWER halves and copied in via
        ``SFPCONFIG``. The float MAD/ADD/MUL ops treat every operand as
        FP32, so an int-stored operand must be reinterpreted from its bits
        rather than used as an integer value (otherwise ``1.4427`` read back
        as ``1069738555`` blows the result up to infinity). Float operands
        pass through untouched, so all-float paths are unaffected.
        """
        if isinstance(value, float):
            return value
        return conv_to_float(value & 0xFFFFFFFF)

    @staticmethod
    def _as_fp32_bits(value):
        """The FP32 bit-pattern (uint32) of an LReg lane, for the bit-exact
        Blackhole float ALU (``fma_model_bh``). Floats are packed to FP32;
        raw int patterns pass through masked."""
        if isinstance(value, float):
            return conv_to_uint32(value)
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
                    if self.backend.blackhole:
                        # Bit-exact hardware FMA. mod1 bit0 negates the multiply
                        # operand a, bit1 negates the addend c (SFPMAD; also gives
                        # SFPADD its -1.0 constant and is unused by SFPMUL where
                        # bit1 is never set). Result stored as an FP32 bit pattern.
                        a_bits = self._as_fp32_bits(self.lregs[va][lane])
                        b_bits = self._as_fp32_bits(self.lregs[vb][lane])
                        c_bits = self._as_fp32_bits(self.lregs[vc][lane])
                        if mod1 & 1:
                            a_bits ^= 0x80000000
                        if mod1 & 2:
                            c_bits ^= 0x80000000
                        d = fma_model_bh(a_bits, b_bits, c_bits)
                    else:
                        a = self._as_fp32(self.lregs[va][lane])
                        b = self._as_fp32(self.lregs[vb][lane])
                        c = self._as_fp32(self.lregs[vc][lane])
                        d = a * b + c
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

    def handle_sfpcompc(self, instruction_info, issue_thread, instr_args):
        vd = instr_args["lreg_dest"]

        do_compc = False
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                do_compc = True
                break

        if do_compc:
            if len(self.flagStack) == 0:
                top = (True, True)
            else:
                top = self.flagStack[0]

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
                top = (False, False)
            else:
                top = self.flagStack[0]

            if mod1 == 0:
                # Plain pop from stack
                assert len(self.flagStack) > 0
                self.flagStack.pop(0)
            elif len(self.flagStack) == 8:
                self.flagStack[7] = top

            if mod1 == 0:
                # Set LaneFlags and UseLaneFlagsForLaneEnable to Top
                self.laneFlags = top[0]
                self.useLaneFlagsForLaneEnable = top[1]
            elif mod1 <= 12:
                # Mutate LaneFlags and UseLaneFlagsForLaneEnable based on Top
                self.laneFlags = list(self.booleanOp(mod1, self.laneFlags, top[0]))
                self.useLaneFlagsForLaneEnable = top[1]
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
            self.flagStack.append((self.laneFlags, self.useLaneFlagsForLaneEnable))

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
                        c = self.lregs[vc][lane]
                        match mod1:
                            case VectorUnit.SFPSETCC_MOD1_LREG_LT0:
                                self.laneFlags[lane] = c < 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_NE0:
                                self.laneFlags[lane] = c != 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_GTE0:
                                self.laneFlags[lane] = c >= 0
                            case VectorUnit.SFPSETCC_MOD1_LREG_EQ0:
                                self.laneFlags[lane] = c == 0

    def _set_lane_flags_cmp(self, vd, vc, compare):
        """Shared body of the Blackhole SFPGT / SFPLE comparisons.

        Per BlackholeA0/.../VectorUnit.md these set ``LaneFlags[Lane]`` from a
        comparison of ``LReg[lreg_dest]`` (VD) against ``LReg[lreg_c]`` (VC),
        under FP32 ordering. Only the FP32-mode flag update is modelled (the
        sign-magnitude-integer mode's extra ``VD = ... ? -(2^31-1) : +0`` write
        is not yet emitted by any path we run).
        """
        for lane in range(32):
            if vd < 12 or self.laneConfigValue(lane, VectorUnit.DISABLE_BACKDOOR_LOAD):
                if self.isLaneEnabled(lane):
                    self.laneFlags[lane] = compare(
                        self._as_fp32(self.lregs[vd][lane]),
                        self._as_fp32(self.lregs[vc][lane]),
                    )

    def handle_sfpgt(self, instruction_info, issue_thread, instr_args):
        # Blackhole SFPGT: LaneFlags[Lane] = (VD > VC).
        self._set_lane_flags_cmp(
            instr_args["lreg_dest"], instr_args["lreg_c"], lambda d, c: d > c
        )

    def handle_sfple(self, instruction_info, issue_thread, instr_args):
        # Blackhole SFPLE: LaneFlags[Lane] = (VD <= VC).
        self._set_lane_flags_cmp(
            instr_args["lreg_dest"], instr_args["lreg_c"], lambda d, c: d <= c
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
        imm12 = instr_args["imm12_math"] & 0xFFF

        vb = vd

        if self.getDiagnosticSettings().reportSFPUCalculations():
            print(f"SFPU: lreg[{vd}] = lreg[{vc}] + lreg[{vb}]")

        if vd < 8 or vd == 16:
            for lane in range(32):
                if self.isLaneEnabled(lane):
                    if mod1 & VectorUnit.SFPIADD_MOD1_ARG_IMM:
                        self.lregs[vd][lane] = self.lregs[vc][lane] + imm12
                    elif mod1 & VectorUnit.SFPIADD_MOD1_ARG_2SCOMP_LREG_DST:
                        self.lregs[vd][lane] = (
                            self.lregs[vc][lane] - self.lregs[vb][lane]
                        )
                    else:
                        self.lregs[vd][lane] = (
                            self.lregs[vc][lane] + self.lregs[vb][lane]
                        )

                    if vd < 8:
                        if mod1 & VectorUnit.SFPIADD_MOD1_CC_NONE:
                            # Leave LaneFlags as-is
                            pass
                        else:
                            self.laneFlags[lane] = self.lregs[vd][lane] < 0

                        if mod1 & VectorUnit.SFPIADD_MOD1_CC_GTE0:
                            self.laneFlags[lane] = not self.laneFlags[lane]

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
        imm10 = instr_args["dest_reg_addr"]
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
        imm10 = instr_args["dest_reg_addr"]
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
                            datum = conv_to_float(
                                DataFormatConversions.FP16InDstToFP32(
                                    rd,
                                    self.laneConfigValue(
                                        lane, VectorUnit.ENABLE_FP16A_INF
                                    ),
                                )
                            )
                        case VectorUnit.MOD0_FMT_BF16:
                            rd = self.getDst().getDst16b(row, column)
                            datum = conv_to_float(
                                DataFormatConversions.BF16InDstToBF16(rd) << 16
                            )
                        case VectorUnit.MOD0_FMT_FP32:
                            rd = self.getDst().getDst32b(row, column)
                            datum = conv_to_float(
                                DataFormatConversions.FP32InDstToFP32(rd)
                            )
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
                        self.lregs[vd][lane] = conv_to_float(self.BF16toFP32(imm16))
                    case VectorUnit.SFPLOADI_MOD0_FLOATA:
                        self.lregs[vd][lane] = conv_to_float(self.FP16toFP32(imm16))
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
