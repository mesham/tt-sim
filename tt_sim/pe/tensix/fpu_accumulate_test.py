"""Pin the FPU accumulate datapath against ttsim, on both architectures.

``MatrixUnit._fpu_group_sums`` / ``_fpu_accumulate`` are a port of ttsim's
``mvmul`` + ``fpu_accum_normalize_encode`` (``src/tensix.cpp``): the hardware's
fixed-point datapath, where each product is truncated to ~12 bits, aligned
within its group of eight lanes, and rounded into Dst on *every* instruction.
What ``perform_mvmul_exact`` actually calls is the batched numpy pair
``_fpu_group_sums_batch`` / ``_fpu_accumulate_batch``; the scalar pair stays the
readable reference. The last two tests here fuzz the batched pair against the
scalar one so the two cannot drift, which is also the standing proof that
numpy's fixed-width int64 never wraps where the scalar code's Python ints would
have widened.

ttsim's model is architecture-independent bar one thing -- a
``#if TT_ARCH_VERSION == 0`` (Wormhole) fixup for renormalising a result whose
sign/magnitude form is exactly -1, where the hardware's leading-sign count
comes out 27 too large. Blackhole fixed it, so the same inputs land on
exponents 27 apart.

The vectors below were produced by compiling ttsim's C model standalone for
both ``TT_ARCH_VERSION`` values; the same harness fuzz-matched this port on
60k random cases (3351 of which exercised the Wormhole/Blackhole difference).
"""

import numpy as np
import pytest

from tt_sim.pe.tensix.backends.matrix import MatrixUnit

# (dstVal, useDst32b, groupSums, expected Wormhole, expected Blackhole).
# The first four straddle the -1 renormalisation quirk: the two arches disagree
# whenever the accumulation lands on -1 exactly, and agree otherwise (case 3
# differs only in the sign of the surviving term, and is the control).
ACCUMULATE_VECTORS = [
    (0x407734D7, True, [(1, 128, 16200920), (0, 0, 0)], 0xC2000000, 0xB4800000),
    (0x42309D6B, True, [(1, 132, 11574636), (0, 0, 0)], 0xC4000000, 0xB6800000),
    (0xC279CB9E, True, [(0, 132, 16370591), (0, 0, 0)], 0x36800000, 0x36800000),
    (0x45AFA914, True, [(1, 139, 11512085), (0, 0, 0)], 0xC7800000, 0xBA000000),
    (
        0x27122657,
        False,
        [(0, 138, 27167162), (1, 178, 21288702)],
        0xD9A20000,
        0xD9A20000,
    ),
    (
        0xFE32B55E,
        False,
        [(1, 116, 21941999), (1, 158, 21812762)],
        0xFE330000,
        0xFE330000,
    ),
    (0x826A7532, False, [(1, 17, 1998203), (0, 49, 29513303)], 0x19610000, 0x19610000),
    (0x8463990E, True, [(0, 84, 14781771), (1, 216, 6553897)], 0xEBC80252, 0xEBC80252),
    (0x5225A6C2, False, [(0, 2, 22228117), (0, 118, 21971497)], 0x52260000, 0x52260000),
    (0x477F72D2, True, [(1, 216, 2791845), (1, 66, 10578947)], 0xEB2A6694, 0xEB2A6694),
]

# (srcA lanes, srcB lanes, dstVal, useDst32b, fidelityPhase, expected). The lane
# values are whitespace-separated FP32 bit patterns, i.e. what the 8-bit-exponent
# Src formats widen to. The whole dot product runs through the shared path, so
# one expectation covers both arches.
MVMUL_VECTORS = [
    (
        "be722000 40a74000 be90a000 3f81c000 3c0d8000 00000000 3c418000 3ee2a000 "
        "00000000 bed42000 41ae6000 befbe000 bec94000 00000000 bfcb8000 c114e000",
        "40296000 bdca8000 00000000 00000000 00000000 be690000 3ef32000 3fc64000 "
        "3ccf4000 bf9e6000 41cbc000 00000000 3c072000 3c14c000 bc7ca000 bc44e000",
        0x418BE000,
        False,
        0,
        0x44090000,
    ),
    (
        "be3ec000 3a3ba000 c0f9a000 35472000 43476000 3535e000 40618000 bf682000 "
        "bf08e000 b9c5c000 3be2c000 c3868000 32d0c000 00000000 bf9fc000 b4ab2000",
        "b59e0000 c1664000 ba808000 36dfe000 32962000 bffa2000 00000000 45582000 "
        "c396a000 3a730000 33ee8000 32d62000 3b572000 3ba30000 37318000 00000000",
        0xC160C000,
        True,
        1,
        0xC152B000,
    ),
    (
        "40238000 3df24000 bd404000 bffea000 00000000 3c6dc000 00000000 3de18000 "
        "bf856000 3e19a000 40eb2000 41c12000 bea58000 c1d3c000 3cf4a000 bc116000",
        "41a5a000 be48e000 3c4d2000 3e83c000 00000000 00000000 3f7ee000 bdc44000 "
        "bf54e000 bddf4000 be3a6000 c1108000 be6a4000 3f2cc000 c1446000 3f360000",
        0x00000000,
        False,
        2,
        0xBEA70000,
    ),
    (
        "41694000 bf2ee000 3355c000 b7b30000 c1fbc000 00000000 bfe82000 c07d8000 "
        "bffd8000 00000000 3e208000 00000000 00000000 badc0000 b4602000 c0c56000",
        "b4362000 3b8f0000 c2cd8000 b86d0000 bba26000 41f4c000 c325c000 b29c2000 "
        "c3938000 00000000 35d92000 ba0e2000 b4ea2000 00000000 c17a8000 4221c000",
        0xBF7F0000,
        True,
        3,
        0xBF706000,
    ),
]


def _exp_prod_adj(fidelityPhase):
    adj = -127
    if fidelityPhase & 1:
        adj -= 5
    if fidelityPhase & 2:
        adj -= 7
    return adj


def _batch_group_sums(lanePairs, fidelityPhase):
    """Run a list of (srcAVals, srcBVals) lane sets through the batched path.

    The batched methods want lanes (then groups) on the leading axis, so the
    list of cases becomes the trailing axis here and the results are transposed
    back into the scalar version's per-case list of two triples.
    """
    srcA = np.array([a for a, _ in lanePairs], dtype=np.int64).T
    srcB = np.array([b for _, b in lanePairs], dtype=np.int64).T
    fields = np.stack(
        MatrixUnit._fpu_group_sums_batch(
            srcA, srcB, fidelityPhase, _exp_prod_adj(fidelityPhase)
        )
    )  # (3 fields, 2 groups, n)
    return [[tuple(g) for g in case] for case in fields.T.tolist()]


def _batch_accumulate(groupSums, dstVals, useDst32b, negOneRenormBug):
    """Run a list of group-sum pairs through the batched path."""
    fields = np.array(groupSums, dtype=np.int64).T  # (3 fields, 2 groups, n)
    return MatrixUnit._fpu_accumulate_batch(
        (fields[0], fields[1], fields[2]),
        np.array(dstVals, dtype=np.int64),
        useDst32b,
        negOneRenormBug,
    ).tolist()


@pytest.mark.parametrize(
    "dstVal,useDst32b,groupSums,wormhole,blackhole",
    ACCUMULATE_VECTORS,
    ids=[f"accum{i}" for i in range(len(ACCUMULATE_VECTORS))],
)
def test_accumulate_matches_ttsim(dstVal, useDst32b, groupSums, wormhole, blackhole):
    assert MatrixUnit._fpu_accumulate(groupSums, dstVal, useDst32b, True) == wormhole, (
        "Wormhole accumulate"
    )
    assert (
        MatrixUnit._fpu_accumulate(groupSums, dstVal, useDst32b, False) == blackhole
    ), "Blackhole accumulate"
    assert _batch_accumulate([groupSums], [dstVal], useDst32b, True) == [wormhole]
    assert _batch_accumulate([groupSums], [dstVal], useDst32b, False) == [blackhole]


def test_neg_one_renorm_quirk_is_wormhole_only():
    """The quirk is reachable and arch-specific, not dead code."""
    differing = [v for v in ACCUMULATE_VECTORS if v[3] != v[4]]
    assert differing, "no vector exercises the Wormhole -1 renormalisation"
    for _, _, _, wormhole, blackhole in differing:
        # Same sign and mantissa; the bogus leading-sign count shifts the
        # exponent by exactly 27.
        assert wormhole & 0x807FFFFF == blackhole & 0x807FFFFF
        assert ((wormhole >> 23) & 0xFF) - ((blackhole >> 23) & 0xFF) == 27


@pytest.mark.parametrize(
    "srcAVals,srcBVals,dstVal,useDst32b,fidelityPhase,expected",
    MVMUL_VECTORS,
    ids=[f"mvmul{i}" for i in range(len(MVMUL_VECTORS))],
)
def test_group_sums_matches_ttsim(
    srcAVals, srcBVals, dstVal, useDst32b, fidelityPhase, expected
):
    groupSums = MatrixUnit._fpu_group_sums(
        [int(v, 16) for v in srcAVals.split()],
        [int(v, 16) for v in srcBVals.split()],
        fidelityPhase,
        _exp_prod_adj(fidelityPhase),
    )
    for negOneRenormBug in (False, True):
        assert (
            MatrixUnit._fpu_accumulate(groupSums, dstVal, useDst32b, negOneRenormBug)
            == expected
        )
    assert _batch_group_sums(
        [
            (
                [int(v, 16) for v in srcAVals.split()],
                [int(v, 16) for v in srcBVals.split()],
            )
        ],
        fidelityPhase,
    ) == [groupSums]


@pytest.mark.parametrize("fidelityPhase", [0, 1, 2, 3])
def test_batch_group_sums_matches_scalar(fidelityPhase):
    """The batched datapath is bit-identical to the scalar reference.

    Operands are drawn as (sign, exponent, mantissa) rather than as uniform
    32-bit words so the exponent range -- which is what drives the alignment
    shifts, the clamp at 30 and the whole-group underflow -- is actually swept,
    and so the zero-exponent lanes the datapath drops are hit often.
    """
    rng = np.random.default_rng(20260802 + fidelityPhase)
    lanes = (
        (rng.integers(0, 2, (2000, 32)) << 31)
        | (rng.integers(0, 256, (2000, 32)) * (rng.random((2000, 32)) > 0.1) << 23)
        | (rng.integers(0, 1 << 23, (2000, 32)))
    ).astype(np.int64)
    cases = [(row[:16].tolist(), row[16:].tolist()) for row in lanes]

    assert _batch_group_sums(cases, fidelityPhase) == [
        MatrixUnit._fpu_group_sums(a, b, fidelityPhase, _exp_prod_adj(fidelityPhase))
        for a, b in cases
    ]


@pytest.mark.parametrize("useDst32b", [False, True])
@pytest.mark.parametrize("negOneRenormBug", [False, True])
def test_batch_accumulate_matches_scalar(useDst32b, negOneRenormBug):
    """As above, over both real and deliberately extreme group sums.

    Half the cases come out of the datapath above, which is the distribution
    that actually occurs; the other half are synthetic, biased toward tiny
    mantissas (a magnitude of exactly 1 is what trips the Wormhole -1
    renormalisation, and a uniform draw over the 28-bit range would never
    produce one) and toward exponents far enough apart to exercise the
    align-to-zero path at a shift of 31.
    """
    rng = np.random.default_rng(31415)
    n = 2000
    lanes = (
        (rng.integers(0, 2, (n, 32)) << 31)
        | (rng.integers(0, 256, (n, 32)) * (rng.random((n, 32)) > 0.1) << 23)
        | (rng.integers(0, 1 << 23, (n, 32)))
    ).astype(np.int64)
    real = _batch_group_sums([(r[:16].tolist(), r[16:].tolist()) for r in lanes], 0)

    mans = rng.integers(0, 1 << 28, (n, 2))
    mans = np.where(rng.random((n, 2)) < 0.25, rng.integers(0, 4, (n, 2)), mans)
    synthetic = np.stack(
        [rng.integers(0, 2, (n, 2)), rng.integers(0, 384, (n, 2)), mans], axis=-1
    ).tolist()

    groupSums = [[tuple(g) for g in gs] for gs in real + synthetic]
    dstVals = rng.integers(0, 1 << 32, 2 * n).tolist()

    assert _batch_accumulate(groupSums, dstVals, useDst32b, negOneRenormBug) == [
        MatrixUnit._fpu_accumulate(gs, dst, useDst32b, negOneRenormBug)
        for gs, dst in zip(groupSums, dstVals)
    ]


# --- the optional Numba path ------------------------------------------------
#
# ``fpu_jit`` fuses the two batched methods into one compiled scalar loop nest.
# It is an accelerator, never a second source of truth, so the contract it has
# to keep is exactly "bit-identical to the numpy pair, at every shape and flag
# combination perform_mvmul_exact can produce". These fuzz that; the test above
# pins the numpy pair to the scalar reference, so the chain reaches ttsim.


def _fused_or_skip():
    try:
        from tt_sim.pe.tensix.backends.fpu_jit_kernel import mvmul_fused
    except ImportError:
        pytest.skip("numba not installed; the numpy path is the whole simulator")
    return mvmul_fused


def _numpy_pair(srcAMat, srcBMat, dstVals, fidelityPhase, expProdAdj, useDst32b, bug):
    """Exactly what perform_mvmul_exact runs when the jit is unavailable."""
    return MatrixUnit._fpu_accumulate_batch(
        MatrixUnit._fpu_group_sums_batch(
            srcAMat[:, np.newaxis, :],
            srcBMat[:, :, np.newaxis],
            fidelityPhase,
            expProdAdj,
        ),
        dstVals,
        useDst32b,
        bug,
    )


def test_fused_jit_matches_numpy_pair():
    """Fuzz the compiled kernel against the numpy pair over every shape.

    Covers both fidelity-phase bits (which reslice the mantissas), both Dst
    widths (16-bit rounds every term before the add), both settings of the
    Wormhole -1 renormalisation quirk, a broadcast and a per-row SrcB, every
    row count 1..16, and a mix of dense exponents and ones with zeros in them
    (a zero exponent takes the nonZero branch out, which is the path that
    decides whether a whole group is live).
    """
    mvmul_fused = _fused_or_skip()
    rng = np.random.default_rng(20260807)

    def pat(shape, dense):
        sign = rng.integers(0, 2, shape, dtype=np.int64) << 31
        exp = rng.integers(100, 150, shape, dtype=np.int64) << 23
        if not dense:
            exp = exp * rng.integers(0, 2, shape, dtype=np.int64)
        return sign | exp | (rng.integers(0, 128, shape, dtype=np.int64) << 16)

    for trial in range(300):
        rows = int(rng.integers(1, 17))
        dense = bool(rng.integers(0, 2))
        srcAMat = pat((16, 16), dense)
        srcBMat = pat((16, 1 if rng.integers(0, 2) else rows), dense)
        fidelityPhase = int(rng.integers(0, 4))
        expProdAdj = _exp_prod_adj(fidelityPhase)
        useDst32b = bool(rng.integers(0, 2))
        bug = bool(rng.integers(0, 2))
        dstVals = pat((rows, 16), dense)
        if not useDst32b:
            dstVals = (dstVals >> 16) << 16

        expected = np.broadcast_to(
            _numpy_pair(
                srcAMat, srcBMat, dstVals, fidelityPhase, expProdAdj, useDst32b, bug
            ),
            (rows, 16),
        )
        got = mvmul_fused(
            srcAMat, srcBMat, dstVals, fidelityPhase, expProdAdj, useDst32b, bug
        )
        assert np.array_equal(expected, got), f"trial {trial}: rows={rows}"


def test_jit_is_optional_and_off_by_default_for_short_runs(monkeypatch):
    """The numpy path must be reachable, and must be what a short run takes.

    ``get_fused_mvmul`` returning None is not a degraded mode -- it is the
    documented default. Compiling costs seconds; a guard with a handful of
    MVMULs would never earn that back, so the switch waits for the threshold
    and ``TT_SIM_NUMBA=0`` pins it off for good.
    """
    from tt_sim.pe.tensix.backends import fpu_jit

    monkeypatch.setenv("TT_SIM_NUMBA", "0")
    fpu_jit.reset_for_test()
    assert fpu_jit.get_fused_mvmul() is None
    assert fpu_jit.get_fused_mvmul() is None  # decision is frozen, not re-read

    monkeypatch.delenv("TT_SIM_NUMBA")
    monkeypatch.setenv("TT_SIM_NUMBA_THRESHOLD", "3")
    fpu_jit.reset_for_test()
    for _ in range(3):
        assert fpu_jit.get_fused_mvmul() is None
    # The fourth call crosses the threshold and tries to import; whether it
    # succeeds depends on numba being installed, and either answer is correct.
    fpu_jit.get_fused_mvmul()
    assert fpu_jit._resolved

    fpu_jit.reset_for_test()
