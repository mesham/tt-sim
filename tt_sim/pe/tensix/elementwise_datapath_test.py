"""ELWMUL / ELWADD / ELWSUB run on the exact FPU datapath, pinned to ttsim.

The compiler team's Gauss-Seidel stencils (``docs/plans/`` hand-off of
2026-09-11) failed their equality gate on tt-sim while the same binaries were
bit-exact on n300 and p150b: every wrong point low by a multiple of 1/16, and
the first wrong intermediate a HiFi4 ``mul_tiles`` of an integer in 129..177 by
0.25 -- 129 * 0.25 came out 32.0, not 32.25. Every operand needing its seventh
mantissa bit lost it, i.e. only fidelity phase 0 (implicit one and six SrcB
mantissa bits) ever contributed.

Two things were wrong, and both are pinned here:

- ``srcAFidelityBits`` / ``srcBFidelityBits`` isolated the odd phases' bits by
  subtracting the even phase's *bit pattern* from the operand's, where the ISA
  helpers subtract its value. The residue reinterpreted as FP32 is a denormal,
  so phases 1-3 multiplied by ~0.
- The element-wise ops evaluated in real numbers and rounded once into Dst,
  where ttsim's ``elwmul`` / ``elwadd`` are one-term cases of its ``mvmul``
  datapath: the fidelity-sliced product, or the 11-bit-mantissa sum, fed to
  ``fpu_accum_normalize_encode`` as a lone first term. On random operands the
  real-number add disagreed with ttsim on 701 of 1024 fp32 elements (the
  hand-off's data happened to fit in 11 bits, so only the multiply showed).
  All three now share :meth:`MatrixUnit._fpu_accumulate_batch` with MVMUL
  through :meth:`MatrixUnit._fpu_product_batch` / :meth:`_fpu_sum_batch`.

The vectors at the bottom come from ``optests/elwmul`` run on ttsim, the
vendor's reference simulator (``optests/diff.sh elwmul`` -- which passes on
both architectures, 18432 elements), so the datapath stays pinned without the
oracle to hand.

Runs standalone (``python3 -m tt_sim.pe.tensix.elementwise_datapath_test``)
or under pytest.
"""

from contextlib import contextmanager

import numpy as np
import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.backends.backend_base import DataFormat
from tt_sim.pe.tensix.backends.matrix import MatrixUnit
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import DataFormatConversions as DFC
from tt_sim.pe.tensix.util import TensixConfigurationConstants
from tt_sim.util.conversion import conv_to_float, conv_to_uint32

ELWMUL = 0x27
ELWADD = 0x28
ELWSUB = 0x30
# ``instr_mod19`` is (BroadcastSrcBRow << 1) | BroadcastSrcBCol0, tt-llk's
# ``p_elwise::SRCB_BCAST_COL`` / ``SRCB_BCAST_ROW``.
SRCB_BCAST_COL = 0x1
SRCB_BCAST_ROW = 0x2


@contextmanager
def _backend(blackhole=False):
    try:
        yield TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
            BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
            blackhole=blackhole,
        ).getBackend()
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


def _elw(opcode, instr_mod19=0, dest_accum_en=0, addr_mode=0, dst=0):
    return (
        (opcode << 24)
        | (dest_accum_en << 21)
        | (instr_mod19 << 19)
        | (addr_mode << 15)
        | dst
    )


def _src_tf32(bits):
    """An FP32 bit pattern narrowed into the Src TF32 layout, as the unpacker
    does it (Sign, Man(10b), Exp(8b); the low 13 mantissa bits dropped)."""
    return DFC.TF32ToSrcFormatTF32(bits >> 13)


def _src_bf16(bits):
    return DFC.BF16ToSrcBF16(bits >> 16)


def _configure(backend, fp32):
    """TF32 in Src and FP32 in Dst (tt-metal's fp32 CBs with fp32_dest_acc_en),
    or BF16 in Src and a 16-bit Dst."""
    fmt = DataFormat.TF32 if fp32 else DataFormat.BF16
    _set_config(backend, "ALU_FORMAT_SPEC_REG0_SrcA", fmt)
    _set_config(backend, "ALU_FORMAT_SPEC_REG1_SrcB", fmt)
    _set_config(backend, "ALU_ACC_CTRL_Fp32_enabled", int(fp32))


def _load_src(backend, srcA, srcB, fp32=True):
    """``srcA`` / ``srcB`` are 8x16 arrays of FP32 bit patterns (SrcB may be a
    single row, or a single column, for the broadcast modes)."""
    encode = _src_tf32 if fp32 else _src_bf16
    matrix = backend.matrix_unit
    regA = backend.getSrcA(matrix.srcABank)
    regB = backend.getSrcB(matrix.srcBBank)
    for i, row in enumerate(srcA):
        for j, bits in enumerate(row):
            regA[i, j] = encode(bits)
    for i, row in enumerate(srcB):
        for j, bits in enumerate(row):
            regB[i, j] = encode(bits)


def _issue(backend, instruction, phases=1):
    """Issue the instruction once per fidelity phase, as HiFi<phases> would."""
    rwc = backend.getRWC(0)
    for phase in range(phases):
        rwc.FidelityPhase = phase
        assert backend.issueInstruction(instruction, 0)
        backend.matrix_unit.clock_tick(phase)


def _read_dst(backend, fp32=True):
    """Dst rows 0..7 as FP32 (or BF16) bit patterns."""
    dst = backend.getDst()
    if fp32:
        return np.array(
            [
                [DFC.FP32InDstToFP32(dst.getDst32b(i, j)) for j in range(16)]
                for i in range(8)
            ]
        )
    return np.array(
        [
            [DFC.BF16InDstToBF16(dst.getDst16b(i, j)) for j in range(16)]
            for i in range(8)
        ]
    )


def _bits(values):
    return [[conv_to_uint32(float(v)) for v in row] for row in values]


def _floats(bits):
    return np.vectorize(lambda b: conv_to_float(int(b)))(bits)


def _run_hifi4(backend, srcA, srcB, instr_mod19=0):
    """Four ELWMULs into a zeroed FP32 Dst, TF32 in Src; Dst rows 0..7 as floats."""
    _configure(backend, True)
    _load_src(backend, _bits(srcA), _bits(srcB))
    _issue(backend, _elw(ELWMUL, instr_mod19), 4)
    return _floats(_read_dst(backend))


# --- the hand-off's failing product, end to end -------------------------------


def test_hifi4_keeps_every_tf32_bit_of_the_srcb_operand():
    """129 * 0.25 is 32.25 at HiFi4; it was 32.0 (phase 0 only) before."""
    srcA = [[0.25] * 16 for _ in range(8)]
    srcB = [[129.0 + 2 * j for j in range(16)] for _ in range(8)]
    with _backend() as backend:
        got = _run_hifi4(backend, srcA, srcB)
    assert np.array_equal(got, 0.25 * np.array(srcB))


def test_hifi4_matches_mvmul_against_a_unit_vector():
    """A 1x1 dot product on MVMUL's datapath and an ELWMUL are the same
    product on the same datapath, so they must round identically over every
    phase, on random TF32 operands."""
    rng = np.random.default_rng(20260913)
    values = (rng.random((2, 8, 16)) * 512 - 256).astype(np.float32)
    values = np.vectorize(lambda v: conv_to_float(conv_to_uint32(float(v)) & ~0x1FFF))(
        values
    )
    srcA, srcB = values
    with _backend() as backend:
        got = _run_hifi4(backend, srcA, srcB)
    aBits = np.array(_bits(srcA), dtype=np.int64)
    bBits = np.array(_bits(srcB), dtype=np.int64)
    dst = np.zeros((8, 16), dtype=np.int64)
    for phase in range(4):
        lanesA = np.zeros((16, 8, 16), dtype=np.int64)
        lanesB = np.zeros((16, 8, 16), dtype=np.int64)
        lanesA[0], lanesB[0] = aBits, bBits
        dst = MatrixUnit._fpu_accumulate_batch(
            MatrixUnit._fpu_group_sums_batch(lanesA, lanesB, phase, _adj(phase)),
            dst,
            True,
            True,
        )
    assert np.array_equal(got, _floats(dst))


def test_srcb_row_and_column_broadcasts():
    srcA = [[float(1 + i) for _ in range(16)] for i in range(8)]
    with _backend() as backend:
        got = _run_hifi4(backend, srcA, [[3.0] * 16], SRCB_BCAST_ROW)
    assert np.array_equal(got, 3.0 * np.array(srcA))
    with _backend() as backend:
        got = _run_hifi4(backend, srcA, [[5.0] + [7.0] * 15] * 8, SRCB_BCAST_COL)
    assert np.array_equal(got, 5.0 * np.array(srcA))


def test_zero_operands_leave_dst_zero():
    srcA = [[0.0] * 16 for _ in range(8)]
    srcB = [[float(j) for j in range(16)] for _ in range(8)]
    with _backend() as backend:
        got = _run_hifi4(backend, srcA, srcB)
    assert not got.any()


def test_elwadd_accumulates_onto_dst_only_when_asked():
    srcA = [[1.0] * 16 for _ in range(8)]
    srcB = [[2.0] * 16 for _ in range(8)]
    with _backend() as backend:
        _configure(backend, True)
        _load_src(backend, _bits(srcA), _bits(srcB))
        _issue(backend, _elw(ELWADD))
        _issue(backend, _elw(ELWADD, dest_accum_en=1))
        got = _floats(_read_dst(backend))
        assert np.array_equal(got, np.full((8, 16), 6.0))
        _issue(backend, _elw(ELWSUB))
        got = _floats(_read_dst(backend))
        assert np.array_equal(got, np.full((8, 16), -1.0))


# --- the term builders against the group-sum datapath -------------------------


def _adj(phase):
    adj = -127
    if phase & 1:
        adj -= 5
    if phase & 2:
        adj -= 7
    return adj


def _random_patterns(rng, shape):
    return (
        (rng.integers(0, 2, shape) << 31)
        | (rng.integers(0, 256, shape) * (rng.random(shape) > 0.1) << 23)
        | rng.integers(0, 1 << 23, shape)
    ).astype(np.int64)


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_product_batch_is_the_one_lane_group_sum(phase):
    """The lone product equals the group sums of a lane vector holding it in
    lane 0 and zeros elsewhere -- the form ttsim's ``elwmul`` reaches the
    shared accumulate in (``sop0`` the product, ``sop1`` empty)."""
    rng = np.random.default_rng(20260911 + phase)
    a = _random_patterns(rng, (3000,))
    b = _random_patterns(rng, (3000,))
    lanesA = np.zeros((16, 3000), dtype=np.int64)
    lanesB = np.zeros((16, 3000), dtype=np.int64)
    lanesA[0], lanesB[0] = a, b
    expected = MatrixUnit._fpu_group_sums_batch(lanesA, lanesB, phase, _adj(phase))
    got = MatrixUnit._fpu_product_batch(a, b, phase, _adj(phase))
    for e, g in zip(expected, got):
        assert np.array_equal(e, g)


def test_sum_batch_is_the_larger_operand_plus_the_aligned_smaller():
    """The sum term is non-negative, carries the larger operand's sign and
    exponent, and is exact whenever the operands' 11-bit mantissas align
    within eleven bits -- the case the hand-off's stencil lives in."""
    rng = np.random.default_rng(20260912)
    a = _random_patterns(rng, (3000,)) & ~0x1FFF
    b = _random_patterns(rng, (3000,)) & ~0x1FFF
    sign, exp, man = MatrixUnit._fpu_sum_batch(a, b)
    # A zero exponent is a zero operand, sign and all.
    a = np.where(a & 0x7F800000, a, 0)
    b = np.where(b & 0x7F800000, b, 0)
    for empty in (sign[1], exp[1], man[1]):
        assert not empty.any()
    assert (man[0] >= 0).all()
    larger = np.where((a & 0x7FFFFFFF) >= (b & 0x7FFFFFFF), a, b)
    assert np.array_equal(sign[0], larger >> 31)
    assert np.array_equal(exp[0], (larger >> 23) & 0xFF)
    # Where the exponents agree, the term is exactly |a| +/- |b| in units of
    # the shared exponent's ULP.
    same = (((a >> 23) & 0xFF) == ((b >> 23) & 0xFF)) & (a != 0) & (b != 0)
    manA = ((larger >> 13) & 0x3FF) | 0x400
    smaller = np.where(larger == a, b, a)
    manB = ((smaller >> 13) & 0x3FF) | 0x400
    exact = np.where((a >> 31) == (b >> 31), manA + manB, manA - manB) << 13
    assert np.array_equal(man[0][same], exact[same])


# --- the real-number helpers the FP16 path still uses -------------------------


def test_fidelity_bit_helpers_partition_the_mantissa():
    """Phases 0 and 1 of SrcA together are the operand less its unconsumed
    lowest TF32 bit; phases 0 and 2 of SrcB together are the whole operand."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        bits = int(rng.integers(0x00800000, 0x7F800000)) & ~0x1FFF
        x = conv_to_float(bits)
        a0, a1 = MatrixUnit.srcAFidelityBits(x, 0), MatrixUnit.srcAFidelityBits(x, 1)
        assert a0 + a1 == conv_to_float(bits & ~0x3FFF)
        b0, b2 = MatrixUnit.srcBFidelityBits(x, 0), MatrixUnit.srcBFidelityBits(x, 2)
        assert b0 + b2 == x
        # The odd phases are the *value* of the low bits, never a denormal.
        assert a1 == 0.0 or abs(a1) >= 2.0 ** (((bits >> 23) & 0xFF) - 127 - 9)
        assert b2 == 0.0 or abs(b2) >= 2.0 ** (((bits >> 23) & 0xFF) - 127 - 10)


# --- ttsim-pinned vectors -----------------------------------------------------
#
# (srcA, srcB, mul@LoFi, mul@HiFi2, mul@HiFi3, mul@HiFi4, add, sub): the
# operands as FP32 bit patterns and what ttsim packs for each op, read out of
# the ``optests/elwmul`` oracle dump. The fp32 arm is TF32 in Src (the
# unpacker drops the low 13 mantissa bits) with an FP32 Dst; the bf16 arm is
# the same operands truncated to BF16 with a 16-bit Dst, results as BF16.
TTSIM_FP32_VECTORS = [
    (
        0x42693C99,
        0x417F99FA,
        0x44663000,
        0x44672E00,
        0x44688A00,
        0x44688B80,
        0x42948000,
        0x42294000,
    ),
    (
        0xBD8F0FF9,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0xBD8F0000,
        0xBD8F0000,
    ),
    (
        0x3E773093,
        0xC06E871C,
        0xBF5F2000,
        0xBF65A200,
        0xBF661A00,
        0xBF661D80,
        0xC05F0000,
        0x407E0000,
    ),
    (
        0x3FB106AE,
        0xBECA7C03,
        0xBF0AE000,
        0xBF0BAA00,
        0xBF0BEC00,
        0xBF0BEC60,
        0x3F7CC000,
        0x3FE3A000,
    ),
    (
        0xBCF5BD50,
        0xBD5AF36D,
        0x3ACC6000,
        0x3AD10F00,
        0x3AD1E100,
        0x3AD1E5D0,
        0xBDAAE000,
        0x3CC00000,
    ),
    (
        0x41F4B556,
        0x3ED6224B,
        0x4148A000,
        0x414C6300,
        0x414C8100,
        0x414C8190,
        0x41F80000,
        0x41F14000,
    ),
    (
        0x3CFD85DB,
        0x4235E284,
        0x3FAE6000,
        0x3FB23E00,
        0x3FB40F00,
        0x3FB41950,
        0x42360000,
        0xC235C000,
    ),
    (
        0x3F303AC5,
        0x3DAA2256,
        0x3D69C000,
        0x3D69C000,
        0x3D69EC00,
        0x3D69EC00,
        0x3F456000,
        0x3F1AE000,
    ),
    (
        0xBFD7DD92,
        0x3EB04EF7,
        0xBF0F0000,
        0xBF145400,
        0xBF148800,
        0xBF1489F0,
        0xBFABA000,
        0xC001F000,
    ),
]
TTSIM_BF16_VECTORS = [
    (0x42690000, 0x417F0000, 0x4466, 0x4467, 0x4468, 0x4468, 0x4294, 0x4229),
    (0xBD8F0000, 0x00000000, 0x0000, 0x0000, 0x0000, 0x0000, 0xBD8F, 0xBD8F),
    (0x3E770000, 0xC06E0000, 0xBF5F, 0xBF66, 0xBF66, 0xBF66, 0xC05F, 0x407E),
    (0x3FB10000, 0xBECA0000, 0xBF0B, 0xBF0C, 0xBF0C, 0xBF0C, 0x3F7D, 0x3FE4),
    (0xBCF50000, 0xBD5A0000, 0x3ACC, 0x3AD0, 0x3AD0, 0x3AD0, 0xBDAA, 0x3CBF),
    (0x41F40000, 0x3ED60000, 0x4149, 0x414C, 0x414C, 0x414C, 0x41F7, 0x41F1),
    (0x3CFD0000, 0x42350000, 0x3FAE, 0x3FB2, 0x3FB3, 0x3FB3, 0x4235, 0xC235),
    (0x3F300000, 0x3DAA0000, 0x3D6A, 0x3D6A, 0x3D6A, 0x3D6A, 0x3F45, 0x3F1B),
    (0xBFD70000, 0x3EB00000, 0xBF0F, 0xBF14, 0xBF14, 0xBF14, 0xBFAB, 0xC002),
]


def _pinned_block(vectors):
    """The vectors spread over an 8x16 block (one per column, the rows alike),
    as (srcA bits, srcB bits, expected per op)."""
    srcA = [[v[0] for v in vectors] + [0] * (16 - len(vectors))] * 8
    srcB = [[v[1] for v in vectors] + [0] * (16 - len(vectors))] * 8
    expected = {
        (ELWMUL, phases): [v[1 + phases] for v in vectors] for phases in (1, 2, 3, 4)
    }
    expected[(ELWADD, 1)] = [v[6] for v in vectors]
    expected[(ELWSUB, 1)] = [v[7] for v in vectors]
    return srcA, srcB, expected


@pytest.mark.parametrize("blackhole", [False, True])
@pytest.mark.parametrize(
    "fp32, vectors", [(True, TTSIM_FP32_VECTORS), (False, TTSIM_BF16_VECTORS)]
)
def test_elementwise_ops_match_ttsim(fp32, vectors, blackhole):
    srcA, srcB, expected = _pinned_block(vectors)
    for (opcode, phases), want in expected.items():
        with _backend(blackhole) as backend:
            _configure(backend, fp32)
            _load_src(backend, srcA, srcB, fp32)
            _issue(backend, _elw(opcode), phases)
            got = _read_dst(backend, fp32)
        assert got[:, : len(want)].tolist() == [want] * 8, (opcode, phases)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
