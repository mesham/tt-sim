"""Pin the FPU accumulate datapath against ttsim, on both architectures.

``MatrixUnit._fpu_group_sums`` / ``_fpu_accumulate`` are a port of ttsim's
``mvmul`` + ``fpu_accum_normalize_encode`` (``src/tensix.cpp``): the hardware's
fixed-point datapath, where each product is truncated to ~12 bits, aligned
within its group of eight lanes, and rounded into Dst on *every* instruction.
ttsim's model is architecture-independent bar one thing -- a
``#if TT_ARCH_VERSION == 0`` (Wormhole) fixup for renormalising a result whose
sign/magnitude form is exactly -1, where the hardware's leading-sign count
comes out 27 too large. Blackhole fixed it, so the same inputs land on
exponents 27 apart.

The vectors below were produced by compiling ttsim's C model standalone for
both ``TT_ARCH_VERSION`` values; the same harness fuzz-matched this port on
60k random cases (3351 of which exercised the Wormhole/Blackhole difference).
"""

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
