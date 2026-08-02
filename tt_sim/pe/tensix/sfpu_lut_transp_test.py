"""Tests for the SFPU LUT / transpose ops: SFPLUT, SFPLUTFP32 and SFPTRANSP.

These three decode on both architectures but had no handler, so issuing one
raised ``NotImplementedError`` out of ``TensixBackendUnit.clock_tick``. They are
easy to miss in the LLK sources because they never appear as ``TTI_*`` macros:
sfpi's ``lut()`` / ``lut_sign()`` / ``lut2()`` helpers emit them directly. The
callers are the *approximate* kernels — ``tanh_tile<fast_and_approx=true>`` and
``sigmoid_tile<true>`` (SFPLUT), ``gelu_tile<true>`` and tt-llk's own
``_calculate_sigmoid_`` (SFPLUTFP32). The accurate FP32 paths those tiles take
by default do not reach a LUT at all: on ``fp32_dest_acc_en`` both tanh and
sigmoid evaluate ``1/(1 + exp(-x))`` with a polynomial exp and a Newton
reciprocal, which is why ``optests/sfpumath`` issues neither instruction.
SFPTRANSP is used by the topk / welfords / ema / binary-broadcast SFPU kernels.

All three are arch-identical (same opcode, same argument bit ranges in ttsim's
``data/bh`` and ``data/wh``, same functional model in the Wormhole and Blackhole
ISA docs), so every Blackhole case below is paired with the same instruction
word on a Wormhole unit. The only difference is the float ALU: Blackhole rounds
through the bit-exact ``fma_model_bh``, Wormhole keeps tt-sim's historical
double-precision ``a * b + c``.

Expected bit patterns come from ttsim's C (``lut8_to_fp32`` / ``lut16_to_fp32``
/ ``fma_model_bh`` from ``src/tensix.cpp`` and ``src/fma.cpp``), evaluated on the
coefficient tables tt-metal's own ``_init_tanh_`` / ``_init_sigmoid_`` load.

Runs standalone (``python3 -m tt_sim.pe.tensix.sfpu_lut_transp_test``) or under
pytest.
"""

from contextlib import contextmanager

import pytest

from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixConfigurationConstants, TensixInstructionDecoder
from tt_sim.util.conversion import conv_to_uint32

# The coefficient tables tt-metal loads, verbatim from tt-llk's
# ckernel_sfpu_tanh.h ``_init_tanh_`` and ckernel_sfpu_sigmoid.h
# ``_init_sigmoid_``. TANH packs (multiplier, addend) as two 8-bit LUT entries
# per LReg; SIGMOID packs six FP16 entries across LReg[0:3] (multipliers) and
# LReg[4:7] (addends).
TANH_TABLE = (0x1DFF, 0x481A, 0xFF00)
SIGMOID_MULTIPLIERS = (0x32F433D9, 0x300A318A, 0x7C002A35)
SIGMOID_ADDENDS = (0x23C89018, 0x30272BAA, 0x37FF34CC)

SFPLUT_MOD0_SGN_UPDATE = 0
SFPLUT_MOD0_SGN_RETAIN = 4
SFPLUT_MOD0_INDIRECT_VD = 8

SFPLUTFP32_MOD1_FP32_3ENTRY_TABLE = 0
SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1 = 2
SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2 = 3
SFPLUTFP32_MOD1_SGN_RETAIN = 4
SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE = 10


@contextmanager
def _vector_unit(blackhole):
    """A vector unit for the given architecture.

    The Tensix config-register layout is a process-global selection, so restore
    the Wormhole layout on the way out — other tests in the same session (and
    the Wormhole replay guards) expect it.
    """
    try:
        yield TensixCoProcessor(None, blackhole=blackhole).getBackend().vector_unit
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _run(vu, instruction):
    """Issue a raw instruction word and let the unit execute it."""
    assert TensixInstructionDecoder.isInstructionRecognised(instruction)
    assert vu.issueInstruction(instruction, 0)
    vu.clock_tick(0)


# Instruction encoders, mirroring tt-metal's ckernel_ops.h TT_OP_* macros.


def _op_sfplut(lreg_ind, instr_mod0, dest_reg_addr=0):
    return (0x73 << 24) | (lreg_ind << 20) | (instr_mod0 << 16) | dest_reg_addr


def _op_sfplutfp32(lreg_dest, instr_mod1):
    return (0x95 << 24) | (lreg_dest << 4) | instr_mod1


def _op_sfptransp(imm12_math=0, lreg_c=0, lreg_dest=0, instr_mod1=0):
    return (
        (0x8C << 24)
        | (imm12_math << 12)
        | (lreg_c << 8)
        | (lreg_dest << 4)
        | instr_mod1
    )


def _op_sfploadmacro(lreg_ind=0, instr_mod0=0, sfpu_addr_mode=0, dest_reg_addr=0):
    return (
        (0x93 << 24)
        | (lreg_ind << 20)
        | (instr_mod0 << 16)
        | (sfpu_addr_mode << 13)
        | dest_reg_addr
    )


def _load_lane0(vu, values):
    """Put ``values`` into lane 0 of the named LRegs."""
    for lreg, value in values.items():
        vu.lregs[lreg][0] = value


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------


def test_decoder_recognises_the_lut_and_transpose_ops():
    for opcode, name in ((0x73, "SFPLUT"), (0x95, "SFPLUTFP32"), (0x8C, "SFPTRANSP")):
        instruction = opcode << 24
        assert TensixInstructionDecoder.isInstructionRecognised(instruction)
        assert TensixInstructionDecoder.getInstructionInfo(instruction)["name"] == name


def test_argument_bit_layouts():
    # An all-ones payload pins each field's width against ttsim's
    # data/{bh,wh}/tensix_isa.json ranges (identical on both architectures):
    # SFPLUT instr_mod0 19:16 / lreg_ind 23:20, SFPLUTFP32 instr_mod1 3:0 /
    # lreg_dest 7:4, SFPTRANSP lreg_dest 7:4.
    lut = TensixInstructionDecoder.getInstructionInfo((0x73 << 24) | 0xFFFFFF)
    assert lut["instr_args"]["instr_mod0"] == 0xF
    assert lut["instr_args"]["lreg_ind"] == 0xF
    lutfp32 = TensixInstructionDecoder.getInstructionInfo((0x95 << 24) | 0xFFFFFF)
    assert lutfp32["instr_args"]["instr_mod1"] == 0xF
    # SFPLUTFP32 declares no field above lreg_dest, so the shared table lets it
    # run to the top of the word; the handler masks it back to 4 bits.
    assert lutfp32["instr_args"]["lreg_dest"] & 0xF == 0xF
    transp = TensixInstructionDecoder.getInstructionInfo((0x8C << 24) | 0xFFFFFF)
    assert transp["instr_args"]["lreg_dest"] == 0xF


# --------------------------------------------------------------------------
# SFPLUT — what approximate tanh_tile / tanh_derivative emit
# (sfpi lut() -> Mod0 = 4)
# --------------------------------------------------------------------------

_TANH_LREGS = {0: TANH_TABLE[0], 1: TANH_TABLE[1], 2: TANH_TABLE[2]}

# ttsim: lut8_to_fp32(0x1D)=0x3F680000 (0.90625), lut8_to_fp32(0xFF)=0 -> the
# first range is 0.90625*|x|; the 0xFF00 entry is 1.0*0 + 1.0 = 1.0 (saturation).
_TANH_INPUTS = (0x3F000000, 0xBF000000, 0x3FC00000, 0xBFC00000, 0x40400000, 0xC0400000)
_TANH_SGN_RETAIN = (
    0x3EE80000,
    0xBEE80000,
    0x3F740000,
    0xBF740000,
    0x3F800000,
    0xBF800000,
)
_TANH_SGN_UPDATE = (
    0x3EE80000,
    0x3EE80000,
    0x3F740000,
    0x3F740000,
    0x3F800000,
    0x3F800000,
)


def _run_lut(vu, table, value, mod0, vd=4):
    _load_lane0(vu, dict(table))
    vu.lregs[3][0] = value
    _run(vu, _op_sfplut(vd, mod0))
    return vu.lregs[vd][0]


def test_sfplut_tanh_table_sign_retain():
    # sfpi's lut(), i.e. approximate tanh_tile / tanh_derivative: the magnitude comes from
    # the table and the sign is copied back from the input.
    with _vector_unit(blackhole=True) as vu:
        got = [
            _run_lut(vu, _TANH_LREGS, v, SFPLUT_MOD0_SGN_RETAIN) for v in _TANH_INPUTS
        ]
    assert got == list(_TANH_SGN_RETAIN)


def test_sfplut_tanh_table_sign_update():
    # sfpi's lut_sign(): no sign copy, so the (always positive) LUT result keeps
    # its own sign — the only difference Mod0 bit 2 makes.
    with _vector_unit(blackhole=True) as vu:
        got = [
            _run_lut(vu, _TANH_LREGS, v, SFPLUT_MOD0_SGN_UPDATE) for v in _TANH_INPUTS
        ]
    assert got == list(_TANH_SGN_UPDATE)


def test_sfplut_zero_input_uses_the_first_range():
    with _vector_unit(blackhole=True) as vu:
        assert _run_lut(vu, _TANH_LREGS, 0, SFPLUT_MOD0_SGN_RETAIN) == 0


def test_sfplut_wormhole_matches_blackhole_on_the_tanh_table():
    # Same opcode, same fields, same functional model on both architectures;
    # these coefficients are exact in FP32 so the two float ALUs agree bit for
    # bit. Wormhole keeps its float-valued LReg model, hence the conversion.
    with _vector_unit(blackhole=False) as vu:
        got = [
            _run_lut(vu, _TANH_LREGS, v, SFPLUT_MOD0_SGN_RETAIN) for v in _TANH_INPUTS
        ]
    assert [conv_to_uint32(v) for v in got] == list(_TANH_SGN_RETAIN)


def test_sfplut_indirect_vd_takes_the_destination_from_lreg7():
    with _vector_unit(blackhole=True) as vu:
        for lane in (0, 1):
            for lreg, value in _TANH_LREGS.items():
                vu.lregs[lreg][lane] = value
            vu.lregs[3][lane] = 0x3F000000  # 0.5
        vu.lregs[7][0] = 5
        vu.lregs[7][1] = 6
        _run(vu, _op_sfplut(4, SFPLUT_MOD0_SGN_RETAIN | SFPLUT_MOD0_INDIRECT_VD))
        # The VD field (4) is ignored; each lane writes where LReg[7] says.
        assert vu.lregs[4][0] == 0
        assert vu.lregs[5][0] == 0x3EE80000
        assert vu.lregs[6][1] == 0x3EE80000


def test_sfplut_reserved_modifier_bits_are_rejected():
    # The ISA docs define only SGN_RETAIN and INDIRECT_VD; ttsim rejects
    # everything except SGN_RETAIN. Guessing at bits 0/1 is not an option.
    for mod0 in (1, 2, 3, 5):
        with _vector_unit(blackhole=True) as vu:
            with pytest.raises(NotImplementedError):
                _run(vu, _op_sfplut(4, mod0))


# --------------------------------------------------------------------------
# SFPLUTFP32 — what tt-llk's _calculate_sigmoid_ emits
# (sfpi lut2(..., mode=0) -> Mod1 = 7)
# --------------------------------------------------------------------------

_SIGMOID_LREGS = {
    0: SIGMOID_MULTIPLIERS[0],
    1: SIGMOID_MULTIPLIERS[1],
    2: SIGMOID_MULTIPLIERS[2],
    4: SIGMOID_ADDENDS[0],
    5: SIGMOID_ADDENDS[1],
    6: SIGMOID_ADDENDS[2],
}


def _run_lutfp32(vu, table, value, mod1, vd=7):
    _load_lane0(vu, dict(table))
    vu.lregs[3][0] = value
    _run(vu, _op_sfplutfp32(vd, mod1))
    return vu.lregs[vd][0]


_SIGMOID_INPUTS = (
    0x3E800000,
    0x3F400000,
    0x3FA00000,
    0x3FE00000,
    0x40200000,
    0x40A00000,
)
_SIGMOID_EXPECTED = (
    0x3D791400,
    0x3E367000,
    0x3E8D7000,
    0x3EB38800,
    0x3ED79200,
    0x3EFFE000,
)


def test_sfplutfp32_sigmoid_six_entry_table():
    # Mod1 = FP16_6ENTRY_TABLE2 | SGN_RETAIN, exactly what lut2(v, l0..l6, 0)
    # expands to; each input lands in a different one of the six ranges.
    mod1 = SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2 | SFPLUTFP32_MOD1_SGN_RETAIN
    with _vector_unit(blackhole=True) as vu:
        got = [_run_lutfp32(vu, _SIGMOID_LREGS, v, mod1) for v in _SIGMOID_INPUTS]
    assert got == list(_SIGMOID_EXPECTED)


def test_sfplutfp32_sign_retain_mirrors_negative_inputs():
    mod1 = SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2 | SFPLUTFP32_MOD1_SGN_RETAIN
    with _vector_unit(blackhole=True) as vu:
        assert _run_lutfp32(vu, _SIGMOID_LREGS, 0xBE800000, mod1) == 0xBD791400
        assert _run_lutfp32(vu, _SIGMOID_LREGS, 0xC0A00000, mod1) == 0xBEFFE000
        # Without SGN_RETAIN the table (all positive) result keeps its own sign.
        assert (
            _run_lutfp32(
                vu, _SIGMOID_LREGS, 0xC0200000, SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1
            )
            == 0x3ED79200
        )


def test_sfplutfp32_table1_and_table2_differ_only_in_the_last_cut():
    # TABLE1 splits the final range at 3.0, TABLE2 at 4.0. 3.5 is the only one
    # of these inputs that falls either side of that boundary.
    with _vector_unit(blackhole=True) as vu:
        table1 = SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE1
        table2 = SFPLUTFP32_MOD1_FP16_6ENTRY_TABLE2
        assert _run_lutfp32(vu, _SIGMOID_LREGS, 0x40200000, table1) == 0x3ED79200
        assert _run_lutfp32(vu, _SIGMOID_LREGS, 0x40200000, table2) == 0x3ED79200
        assert _run_lutfp32(vu, _SIGMOID_LREGS, 0x40600000, table1) == 0x3EFFE000
        assert _run_lutfp32(vu, _SIGMOID_LREGS, 0x40600000, table2) == 0x3EF06600
        assert _run_lutfp32(vu, _SIGMOID_LREGS, 0x40900000, table2) == 0x3EFFE000


# FP32 3-entry table: full-precision multipliers in LReg[0:3], addends in
# LReg[4:7] (no FP16 packing, so Mod1 bit 1 is clear).
_FP32_TABLE = {
    0: 0x40000000,  # 2.0
    1: 0x40400000,  # 3.0
    2: 0x40800000,  # 4.0
    4: 0x3F800000,  # 1.0
    5: 0x3F000000,  # 0.5
    6: 0x3E800000,  # 0.25
}


def test_sfplutfp32_fp32_three_entry_table():
    with _vector_unit(blackhole=True) as vu:
        mod1 = SFPLUTFP32_MOD1_FP32_3ENTRY_TABLE
        assert _run_lutfp32(vu, _FP32_TABLE, 0x3F000000, mod1) == 0x40000000  # 2*.5+1
        assert _run_lutfp32(vu, _FP32_TABLE, 0x3FC00000, mod1) == 0x40A00000  # 3*1.5+.5
        assert (
            _run_lutfp32(vu, _FP32_TABLE, 0x40200000, mod1) == 0x41240000
        )  # 4*2.5+.25
        # Only the magnitude feeds the multiply-add...
        assert _run_lutfp32(vu, _FP32_TABLE, 0xBF000000, mod1) == 0x40000000
        # ...unless SGN_RETAIN puts the input's sign back on the result.
        assert (
            _run_lutfp32(vu, _FP32_TABLE, 0xBF000000, SFPLUTFP32_MOD1_SGN_RETAIN)
            == 0xC0000000
        )


# FP16 3-entry table (Mod1 = 10): both halves of the pair come from LReg[i], and
# INDIRECT_VD is set as part of the encoding — the hardware workaround the ISA
# docs call out, so the destination really does come from LReg[7].
_FP16_3ENTRY_TABLE = {0: 0x3C004000, 1: 0x40003C00, 2: 0x34003800}


def test_sfplutfp32_fp16_three_entry_table_is_indirect_by_encoding():
    with _vector_unit(blackhole=True) as vu:
        for value, expected in (
            (0x3F000000, 0x40200000),  # 1.0*0.5 + 2.0
            (0x3FC00000, 0x40800000),  # 2.0*1.5 + 1.0
            (0x40200000, 0x3F900000),  # 0.5*2.5 + 0.25
        ):
            _load_lane0(vu, dict(_FP16_3ENTRY_TABLE))
            vu.lregs[3][0] = value
            vu.lregs[7][0] = 5  # destination index, not the VD field
            _run(vu, _op_sfplutfp32(0, SFPLUTFP32_MOD1_FP16_3ENTRY_TABLE))
            assert vu.lregs[5][0] == expected


def test_sfplutfp32_wormhole_matches_blackhole_on_exact_coefficients():
    # Arch-identical op; on coefficients that need no rounding the Wormhole
    # double-precision path lands on the same FP32 bits.
    with _vector_unit(blackhole=False) as vu:
        mod1 = SFPLUTFP32_MOD1_FP32_3ENTRY_TABLE
        assert (
            conv_to_uint32(_run_lutfp32(vu, _FP32_TABLE, 0x3F000000, mod1))
            == 0x40000000
        )
        assert (
            conv_to_uint32(_run_lutfp32(vu, _FP32_TABLE, 0x40200000, mod1))
            == 0x41240000
        )
        assert (
            conv_to_uint32(
                _run_lutfp32(vu, _FP32_TABLE, 0xBF000000, SFPLUTFP32_MOD1_SGN_RETAIN)
            )
            == 0xC0000000
        )


# --------------------------------------------------------------------------
# SFPTRANSP
# --------------------------------------------------------------------------


def _transposed(before):
    """Reference permutation: for each of the 8 columns, the 4x4 matrix formed by
    lanes ``row * 8 + column`` of four consecutive LRegs is transposed."""
    after = [list(row) for row in before]
    for base in (0, 4):
        for column in range(8):
            for i in range(4):
                for j in range(4):
                    after[base + i][j * 8 + column] = before[base + j][i * 8 + column]
    return after


def _fill_lregs(vu):
    before = [[(r << 8) | lane for lane in range(32)] for r in range(8)]
    for r in range(8):
        for lane in range(32):
            vu.lregs[r][lane] = before[r][lane]
    return before


def test_sfptransp_transposes_both_register_groups():
    with _vector_unit(blackhole=True) as vu:
        before = _fill_lregs(vu)
        _run(vu, _op_sfptransp())
        got = [[vu.lregs[r][lane] for lane in range(32)] for r in range(8)]
    assert got == _transposed(before)


def test_sfptransp_is_column_wise_not_a_4x8_transpose():
    # Data never crosses columns: lane 9 (row 1, column 1) can only ever be
    # exchanged with lanes 1, 17 and 25 (rows 0, 2, 3 of column 1).
    with _vector_unit(blackhole=True) as vu:
        before = _fill_lregs(vu)
        _run(vu, _op_sfptransp())
        assert vu.lregs[1][9] == before[1][9]  # diagonal element, unmoved
        assert vu.lregs[0][9] == before[1][1]
        assert vu.lregs[1][1] == before[0][9]


def test_sfptransp_matches_on_wormhole():
    with _vector_unit(blackhole=False) as vu:
        before = _fill_lregs(vu)
        _run(vu, _op_sfptransp())
        got = [[vu.lregs[r][lane] for lane in range(32)] for r in range(8)]
    assert got == _transposed(before)


def test_sfptransp_writes_only_to_enabled_lanes():
    # Each half of a swap is gated on its own lane, so a disabled lane keeps its
    # value while its partner still receives the pre-swap value from it.
    with _vector_unit(blackhole=True) as vu:
        before = _fill_lregs(vu)
        vu.useLaneFlagsForLaneEnable = [True] * 32
        vu.laneFlags = [lane != 8 for lane in range(32)]  # lane 8 disabled
        _run(vu, _op_sfptransp())
        assert vu.lregs[0][8] == before[0][8]  # disabled lane keeps its value
        assert vu.lregs[1][0] == before[0][8]  # its partner still gets that value


# --------------------------------------------------------------------------
# SFPLOADMACRO — decoded, but deliberately not modelled
# --------------------------------------------------------------------------


def test_sfploadmacro_fails_loudly_with_the_workaround():
    for blackhole in (False, True):
        with _vector_unit(blackhole=blackhole) as vu:
            with pytest.raises(
                NotImplementedError, match="TT_METAL_DISABLE_SFPLOADMACRO"
            ):
                _run(vu, _op_sfploadmacro())


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "sfpu_lut_transp_test OK: SFPLUT / SFPLUTFP32 / SFPTRANSP modelled on "
        "both architectures; SFPLOADMACRO guarded"
    )


if __name__ == "__main__":
    main()
