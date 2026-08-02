"""Tests for the Blackhole-enhanced modes of SFPU ops that exist on both arches.

Blackhole widens several existing SFPU instructions rather than adding new
opcodes: SFPCAST gains the sign-magnitude <-> two's-complement conversions,
SFP_STOCH_RND gains a wider ``rnd_mode`` field (with round-toward-zero),
SFPSHFT gains arithmetic shifts and a VC source, SFPSHFT2's zero-filling
subvector shift is usable (it is broken in Wormhole silicon), SFPMAD gains
operand negation, and SFPLOADMACRO's ``sfpu_addr_mode`` field moves down a bit.
Every case below is checked against ttsim (``src/tensix.cpp`` under
``TT_ARCH_VERSION >= 1``, and ``data/{bh,wh}/tensix_isa.json`` for the field
layouts), and each is paired with the Wormhole behaviour so the arch gating is
verified in both directions.

Runs standalone (``python3 -m tt_sim.pe.tensix.sfpu_blackhole_modes_test``) or
under pytest. Instructions are encoded exactly as tt-metal's ``ckernel_ops.h``
does and pushed through a real ``VectorUnit``, so the decoder and the handler
are both exercised.
"""

from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixInstructionDecoder
from tt_sim.util.conversion import conv_to_uint32


def _vector_unit(blackhole):
    return TensixCoProcessor(None, blackhole=blackhole).getBackend().vector_unit


def _run(vu, instruction):
    """Issue a raw instruction word and let the unit execute it."""
    assert TensixInstructionDecoder.isInstructionRecognised(instruction)
    assert vu.issueInstruction(instruction, 0)
    vu.clock_tick(0)


# Instruction encoders, mirroring tt-metal's ckernel_ops.h TT_OP_* macros.


def _op_sfpcast(lreg_src_c, lreg_dest, instr_mod1):
    return (0x90 << 24) | (lreg_src_c << 8) | (lreg_dest << 4) | instr_mod1


def _op_sfp_stoch_rnd(rnd_mode, imm8, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1):
    return (
        (0x8E << 24)
        | (rnd_mode << 21)
        | (imm8 << 16)
        | (lreg_src_b << 12)
        | (lreg_src_c << 8)
        | (lreg_dest << 4)
        | instr_mod1
    )


def _op_sfpshft(imm12, lreg_c, lreg_dest, instr_mod1):
    return (
        (0x7A << 24)
        | ((imm12 & 0xFFF) << 12)
        | (lreg_c << 8)
        | (lreg_dest << 4)
        | instr_mod1
    )


def _op_sfpshft2(imm12, lreg_src_c, lreg_dest, instr_mod1):
    return (
        (0x94 << 24)
        | ((imm12 & 0xFFF) << 12)
        | (lreg_src_c << 8)
        | (lreg_dest << 4)
        | instr_mod1
    )


def _op_sfpmad(lreg_src_a, lreg_src_b, lreg_src_c, lreg_dest, instr_mod1):
    return (
        (0x84 << 24)
        | (lreg_src_a << 16)
        | (lreg_src_b << 12)
        | (lreg_src_c << 8)
        | (lreg_dest << 4)
        | instr_mod1
    )


def _op_sfploadmacro(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr):
    # Blackhole field placement (sfpu_addr_mode at bits 15:13); on Wormhole the
    # same bits decode as a 2-bit field at 15:14.
    return (
        (0x93 << 24)
        | (lreg_ind << 20)
        | (instr_mod0 << 16)
        | (sfpu_addr_mode << 13)
        | dest_reg_addr
    )


# --------------------------------------------------------------------------
# SFPCAST: Blackhole modes 2 (sign-magnitude -> int32) and 3 (int32 -> sign-mag)
# --------------------------------------------------------------------------

_CAST_SM32_TO_INT32 = 2
_CAST_INT32_TO_SM32 = 3


def test_sfpcast_blackhole_int32_to_sign_magnitude():
    vu = _vector_unit(blackhole=True)
    values = [0xFFFFFFFB, 7, 0x80000000, 0]  # -5, +7, INT32_MIN, 0
    for lane, value in enumerate(values):
        vu.lregs[1][lane] = value
    _run(vu, _op_sfpcast(1, 0, _CAST_INT32_TO_SM32))
    # -5 -> sign-magnitude -5; positives pass through; INT32_MIN negates to
    # itself, so it stays 0x80000000 (sign-magnitude -0).
    assert [vu.lregs[0][i] for i in range(4)] == [0x80000005, 7, 0x80000000, 0]


def test_sfpcast_blackhole_sign_magnitude_to_int32():
    vu = _vector_unit(blackhole=True)
    values = [0x80000005, 5, 0x80000000, 0xFFFFFFFF]  # sm -5, +5, sm -0, sm -max
    for lane, value in enumerate(values):
        vu.lregs[1][lane] = value
    _run(vu, _op_sfpcast(1, 0, _CAST_SM32_TO_INT32))
    # sm -0 becoming INT32_MIN rather than 0 is the hardware quirk sfpi
    # documents alongside SFPCAST_MOD1_SM32_TO_INT32.
    assert [vu.lregs[0][i] for i in range(4)] == [
        0xFFFFFFFB,
        5,
        0x80000000,
        0x80000001,
    ]


def test_sfpcast_wormhole_keeps_int32_to_fp32_for_all_modes():
    # Wormhole has no modes 2/3: every modifier still performs the int32->fp32
    # cast (mod1 bit 0 only selects stochastic rounding, which is unmodelled).
    vu = _vector_unit(blackhole=False)
    vu.lregs[1][0] = 1
    _run(vu, _op_sfpcast(1, 0, _CAST_INT32_TO_SM32))
    assert conv_to_uint32(vu.lregs[0][0]) == 0x3F800000  # 1.0f


# --------------------------------------------------------------------------
# SFP_STOCH_RND: the rnd_mode field is 2 bits on Blackhole, 1 on Wormhole
# --------------------------------------------------------------------------

_FP32_TO_UINT8 = 2
_FP32_TO_FP16B = 1
_RND_NEAREST_EVEN = 0
_RND_ZERO = 2


def test_sfp_stoch_rnd_blackhole_round_toward_zero():
    vu = _vector_unit(blackhole=True)
    vu.lregs[1][0] = 3.7
    _run(vu, _op_sfp_stoch_rnd(_RND_NEAREST_EVEN, 0, 0, 1, 0, _FP32_TO_UINT8))
    assert vu.lregs[0][0] == 4

    vu.lregs[1][0] = 3.7
    _run(vu, _op_sfp_stoch_rnd(_RND_ZERO, 0, 0, 1, 0, _FP32_TO_UINT8))
    assert vu.lregs[0][0] == 3  # truncated rather than rounded up


def test_sfp_stoch_rnd_blackhole_round_toward_zero_narrowing():
    # fp32 -> bf16 keeps 7 mantissa bits; 0x3FCCCCCD (1.6f) rounds up to
    # 0x3FCD0000 but truncates to 0x3FCC0000.
    vu = _vector_unit(blackhole=True)
    vu.lregs[1][0] = 0x3FCCCCCD
    _run(vu, _op_sfp_stoch_rnd(_RND_NEAREST_EVEN, 0, 0, 1, 0, _FP32_TO_FP16B))
    assert vu.lregs[0][0] == 0x3FCD0000

    vu.lregs[1][0] = 0x3FCCCCCD
    _run(vu, _op_sfp_stoch_rnd(_RND_ZERO, 0, 0, 1, 0, _FP32_TO_FP16B))
    assert vu.lregs[0][0] == 0x3FCC0000


def test_sfp_stoch_rnd_wormhole_ignores_the_extra_rnd_mode_bit():
    # The same word: on Wormhole rnd_mode is only bit 21, so the round-to-zero
    # encoding's high bit is not part of the field and rounding is unchanged.
    vu = _vector_unit(blackhole=False)
    vu.lregs[1][0] = 3.7
    _run(vu, _op_sfp_stoch_rnd(_RND_ZERO, 0, 0, 1, 0, _FP32_TO_UINT8))
    assert vu.lregs[0][0] == 4


# --------------------------------------------------------------------------
# SFPSHFT: Blackhole adds arithmetic right shift (bit 1) and a VC source (bit 2)
# --------------------------------------------------------------------------

_SHFT_IMM = 1
_SHFT_ARITHMETIC = 2
_SHFT_SRC_LREG_C = 4


def test_sfpshft_blackhole_arithmetic_right_shift():
    vu = _vector_unit(blackhole=True)
    vu.lregs[0][0] = 0x80000000
    _run(vu, _op_sfpshft(-4, 1, 0, _SHFT_IMM | _SHFT_ARITHMETIC))
    assert vu.lregs[0][0] == 0xF8000000  # sign extended

    vu.lregs[0][0] = 0x80000000
    _run(vu, _op_sfpshft(-4, 1, 0, _SHFT_IMM))
    assert vu.lregs[0][0] == 0x08000000  # logical


def test_sfpshft_blackhole_value_from_vc():
    vu = _vector_unit(blackhole=True)
    vu.lregs[0][0] = 0xDEADBEEF  # would be the source without bit 2
    vu.lregs[1][0] = 0x1234
    _run(vu, _op_sfpshft(4, 1, 0, _SHFT_IMM | _SHFT_SRC_LREG_C))
    assert vu.lregs[0][0] == 0x12340


def test_sfpshft_wormhole_masks_off_the_blackhole_bits():
    # Identical words on Wormhole: bits 1 and 2 are reserved there, so the
    # source stays VD and right shifts stay logical.
    vu = _vector_unit(blackhole=False)
    vu.lregs[0][0] = 0x80000000
    _run(vu, _op_sfpshft(-4, 1, 0, _SHFT_IMM | _SHFT_ARITHMETIC))
    assert vu.lregs[0][0] == 0x08000000

    vu = _vector_unit(blackhole=False)
    vu.lregs[0][0] = 0x1234
    vu.lregs[1][0] = 0xDEADBEEF
    _run(vu, _op_sfpshft(4, 1, 0, _SHFT_IMM | _SHFT_SRC_LREG_C))
    assert vu.lregs[0][0] == 0x12340


# --------------------------------------------------------------------------
# SFPSHFT2
# --------------------------------------------------------------------------

_SHFT2_SUBVEC_SHFLROR1 = 3
_SHFT2_SUBVEC_SHFLSHR1 = 4
_SHFT2_SHFT_LREG = 5


def test_decoder_recognises_sfpshft2():
    info = TensixInstructionDecoder.getInstructionInfo(0x94 << 24)
    assert info["name"] == "SFPSHFT2"


def test_sfpshft2_subvector_rotate():
    vu = _vector_unit(blackhole=True)
    for lane in range(32):
        vu.lregs[1][lane] = 0x100 + lane
    _run(vu, _op_sfpshft2(0, 1, 0, _SHFT2_SUBVEC_SHFLROR1))
    expected = [0x100 + (lane - 1 if lane & 7 else lane + 7) for lane in range(32)]
    assert [vu.lregs[0][lane] for lane in range(32)] == expected


def test_sfpshft2_subvector_rotate_in_place():
    # VD == VC is the common encoding; every lane must see the pre-shift value.
    vu = _vector_unit(blackhole=True)
    for lane in range(32):
        vu.lregs[0][lane] = 0x100 + lane
    _run(vu, _op_sfpshft2(0, 0, 0, _SHFT2_SUBVEC_SHFLROR1))
    expected = [0x100 + (lane - 1 if lane & 7 else lane + 7) for lane in range(32)]
    assert [vu.lregs[0][lane] for lane in range(32)] == expected


def test_sfpshft2_subvector_shift_zero_fill_is_blackhole_only():
    vu = _vector_unit(blackhole=True)
    for lane in range(32):
        vu.lregs[1][lane] = 0x100 + lane
    _run(vu, _op_sfpshft2(0, 1, 0, _SHFT2_SUBVEC_SHFLSHR1))
    expected = [(0x100 + lane - 1) if lane & 7 else 0 for lane in range(32)]
    assert [vu.lregs[0][lane] for lane in range(32)] == expected

    wh = _vector_unit(blackhole=False)
    try:
        _run(wh, _op_sfpshft2(0, 1, 0, _SHFT2_SUBVEC_SHFLSHR1))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("SFPSHFT2 mode 4 must be rejected on Wormhole")


def test_sfpshft2_shift_by_lreg():
    vu = _vector_unit(blackhole=True)
    vu.lregs[2][0], vu.lregs[1][0] = 0x1234, 4  # left shift by 4
    vu.lregs[2][1], vu.lregs[1][1] = 0x1234, 0xFFFFFFFC  # right shift by 4
    _run(vu, _op_sfpshft2(2, 1, 0, _SHFT2_SHFT_LREG))
    assert vu.lregs[0][0] == 0x12340
    assert vu.lregs[0][1] == 0x123


def test_sfpshft2_unmodelled_modes_raise():
    vu = _vector_unit(blackhole=True)
    for mod1 in (0, 1, 2, 6):
        try:
            _run(vu, _op_sfpshft2(0, 1, 0, mod1))
        except NotImplementedError:
            continue
        raise AssertionError(f"SFPSHFT2 mode {mod1} should not be modelled")


# --------------------------------------------------------------------------
# SFPMAD operand negation (Blackhole instr_mod1 bits 0 and 1)
# --------------------------------------------------------------------------


def test_sfpmad_blackhole_negates_operands():
    # a=2.0 b=3.0 c=1.0 through the bit-exact Blackhole FMA.
    for mod1, expected in (
        (0, 0x40E00000),
        (1, 0xC0A00000),
        (2, 0x40A00000),
        (3, 0xC0E00000),
    ):
        vu = _vector_unit(blackhole=True)
        vu.lregs[1][0], vu.lregs[2][0], vu.lregs[3][0] = 2.0, 3.0, 1.0
        _run(vu, _op_sfpmad(1, 2, 3, 0, mod1))
        assert vu.lregs[0][0] == expected, f"mod1={mod1}"


def test_sfpmad_wormhole_ignores_negation_bits():
    # Wormhole reserves those bits (ttsim rejects a non-zero instr_mod1 there),
    # so the same word must still compute a*b+c = 7.0 through the Wormhole FMA.
    vu = _vector_unit(blackhole=False)
    vu.lregs[1][0], vu.lregs[2][0], vu.lregs[3][0] = 2.0, 3.0, 1.0
    _run(vu, _op_sfpmad(1, 2, 3, 0, 1))
    assert vu.lregs[0][0] == 0x40E00000


# --------------------------------------------------------------------------
# SFPLOADMACRO: no handler yet, but its sfpu_addr_mode field moved on Blackhole
# --------------------------------------------------------------------------


def test_sfploadmacro_addr_mode_field_shifts_on_blackhole():
    word = _op_sfploadmacro(lreg_ind=0, instr_mod0=3, sfpu_addr_mode=5, dest_reg_addr=8)
    info = TensixInstructionDecoder.getInstructionInfo(word)
    assert info["name"] == "SFPLOADMACRO"

    bh = _vector_unit(blackhole=True)
    assert bh._read_sfpu_addr_mode(info, info["instr_args"]) == 5  # raw bits 15:13
    wh = _vector_unit(blackhole=False)
    assert wh._read_sfpu_addr_mode(info, info["instr_args"]) == 5 >> 1  # bits 15:14


# --------------------------------------------------------------------------
# SFPPOPC is identical on both architectures (no ttsim arch guard); this pins
# that down so a future "Blackhole fixes SFPPOPC" claim has to be evidenced.
# --------------------------------------------------------------------------


def test_sfppopc_matches_across_architectures():
    results = []
    for blackhole in (False, True):
        vu = _vector_unit(blackhole=blackhole)
        vu.laneFlags = [lane % 2 == 0 for lane in range(32)]
        vu.useLaneFlagsForLaneEnable = [True] * 32
        _run(vu, 0x87 << 24)  # SFPPUSHC
        vu.laneFlags = [False] * 32
        _run(vu, 0x88 << 24)  # SFPPOPC, instr_mod1 = 0
        results.append((list(vu.laneFlags), list(vu.useLaneFlagsForLaneEnable)))
    assert results[0] == results[1]
    assert results[0][0] == [lane % 2 == 0 for lane in range(32)]


def test_sfppushc_is_lifo_and_snapshots_by_value():
    # ttsim writes cc_stack[cc_sp] then increments (and reverses on pop), so the
    # stack is LIFO, and it stores the flags by value -- a later SFPSETCC must
    # not rewrite an entry already pushed.
    vu = _vector_unit(blackhole=True)
    vu.useLaneFlagsForLaneEnable = [True] * 32
    outer = [lane % 2 == 0 for lane in range(32)]
    inner = [lane % 4 == 0 for lane in range(32)]

    vu.laneFlags = list(outer)
    _run(vu, 0x87 << 24)  # SFPPUSHC
    vu.laneFlags[:] = inner  # in place, as SFPSETCC mutates it
    _run(vu, 0x87 << 24)  # SFPPUSHC
    vu.laneFlags = [False] * 32

    _run(vu, 0x88 << 24)  # SFPPOPC -> the inner snapshot
    assert vu.laneFlags == inner
    _run(vu, 0x88 << 24)  # SFPPOPC -> the outer snapshot
    assert vu.laneFlags == outer


# --------------------------------------------------------------------------
# SFPIADD / SFPSETCC lane flags. Both are arch-identical in ttsim, but their
# operands are read as *signed 32-bit* there (`int32_t src = LReg[c]`), which
# on Blackhole means the raw uint32 lane's bit 31 -- never `< 0` if compared as
# a Python int. sfpi's `v_if` lowers onto exactly these two, so getting the
# sign wrong silently disables every lane inside a conditional block.
# --------------------------------------------------------------------------


def _op_sfpiadd(imm12_math, lreg_c, lreg_dest, instr_mod1):
    return (
        (0x79 << 24)
        | (imm12_math << 12)
        | (lreg_c << 8)
        | (lreg_dest << 4)
        | instr_mod1
    )


def _op_sfpsetcc(imm12_math, lreg_c, lreg_dest, instr_mod1):
    return (
        (0x7B << 24)
        | (imm12_math << 12)
        | (lreg_c << 8)
        | (lreg_dest << 4)
        | instr_mod1
    )


def test_sfpiadd_sign_extends_the_immediate_and_sets_the_flag_from_bit31():
    # The FP32 exp kernel's overflow guard is `sfpiadd_i(e, -255, CC_LT0)`, which
    # arrives as imm12_math = 0xF01.
    for blackhole in (True, False):
        vu = _vector_unit(blackhole=blackhole)
        vu.lregs[1][0] = 124  # e < 255 -> in range, flag set
        vu.lregs[1][1] = 300  # e > 255 -> overflow, flag clear
        _run(vu, _op_sfpiadd(-255 & 0xFFF, 1, 0, 1))  # ARG_IMM | CC_LT0
        assert conv_to_uint32(vu.lregs[0][0]) == 0xFFFFFF7D  # 124 - 255
        assert vu.lregs[0][1] == 45
        assert vu.laneFlags[0] is True
        assert vu.laneFlags[1] is False


def test_sfpiadd_cc_gte0_inverts_and_cc_none_leaves_the_flag_alone():
    vu = _vector_unit(blackhole=True)
    vu.lregs[1][0] = 124
    _run(vu, _op_sfpiadd(-255 & 0xFFF, 1, 0, 1 | 8))  # ARG_IMM | CC_GTE0
    assert vu.laneFlags[0] is False
    vu.laneFlags[0] = True
    vu.lregs[1][0] = 124
    _run(vu, _op_sfpiadd(-255 & 0xFFF, 1, 0, 1 | 4))  # ARG_IMM | CC_NONE
    assert vu.laneFlags[0] is True


def test_sfpsetcc_lt0_reads_the_fp32_sign_bit():
    # sfpi's `v_if(t < 0)` -- e.g. the Newton-Raphson step of
    # sfpu_reciprocal_iter. Blackhole holds -1.0 as the bit pattern 0xBF800000.
    bh = _vector_unit(blackhole=True)
    bh.useLaneFlagsForLaneEnable = [True] * 32
    bh.laneFlags = [True] * 32
    bh.lregs[1][0], bh.lregs[1][1] = 0xBF800000, 0x3F800000
    _run(bh, _op_sfpsetcc(0, 1, 0, 0))  # LREG_LT0
    assert bh.laneFlags[0] is True
    assert bh.laneFlags[1] is False

    # Wormhole uses the same uint32 lane model, and must agree.
    wh = _vector_unit(blackhole=False)
    wh.useLaneFlagsForLaneEnable = [True] * 32
    wh.laneFlags = [True] * 32
    wh.lregs[1][0], wh.lregs[1][1] = -1.0, 1.0
    _run(wh, _op_sfpsetcc(0, 1, 0, 0))
    assert wh.laneFlags[:2] == bh.laneFlags[:2]


def test_sfpsetcc_gte0_reads_the_fp32_sign_bit():
    vu = _vector_unit(blackhole=True)
    vu.useLaneFlagsForLaneEnable = [True] * 32
    vu.laneFlags = [True] * 32
    vu.lregs[1][0], vu.lregs[1][1] = 0xBF800000, 0x3F800000
    _run(vu, _op_sfpsetcc(0, 1, 0, 4))  # LREG_GTE0
    assert vu.laneFlags[0] is False
    assert vu.laneFlags[1] is True


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "sfpu_blackhole_modes_test OK: SFPCAST/SFP_STOCH_RND/SFPSHFT/SFPSHFT2/"
        "SFPMAD/SFPLOADMACRO/SFPPUSHC/SFPPOPC/SFPIADD/SFPSETCC arch behaviour "
        "verified"
    )


if __name__ == "__main__":
    main()
