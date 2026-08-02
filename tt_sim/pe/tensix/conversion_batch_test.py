"""Prove the block form of each Src/Dst conversion equals the scalar one.

``MatrixUnit.perform_mvmul_exact`` gathers a whole rectangle of SrcA/SrcB/Dst
per instruction and converts it in one call, by handing the *same*
``DataFormatConversions`` classmethod a numpy array instead of an int. There is
therefore no second implementation of the storage-layout bit permutations to
keep in step -- but numpy is not Python: a fixed-width int64 wraps where a
Python int widens, ``bool`` arithmetic and ``>>`` on a signed array have their
own rules, and a stray branch would silently take one arm for the whole block.

So each conversion the batched path can reach is checked over its **entire**
input space (or, for the FP32-width ones, over every equivalence class of it):
the array result must equal the scalar result value for value, not
approximately and not on a sample. 2**19 is small enough that exhaustive is
simply the cheapest option.

Runs standalone (``python3 -m tt_sim.pe.tensix.conversion_batch_test``) or under
pytest.
"""

import numpy as np
import pytest

from tt_sim.pe.tensix.util import DataFormatConversions as DFC

# A Src datum is 19 bits wide (Sign,Man(10b),Exp(8b), variously rearranged).
SRC_SPACE = 1 << 19
# A Dst16b datum is 16 bits.
DST16_SPACE = 1 << 16

# Everything the MVMUL gather applies to a SrcA/SrcB block, plus the
# intermediate conversions those are written in terms of.
IN_SRC_CONVERSIONS = [
    "TF32InSrcToTF32",
    "BF16InSrcToBF16",
    "FP16InSrcToFP16",
    "BF16InSrcToFP32",
    "TF32InSrcToFP32",
    "FP16InSrcToFP32",
]

# ... and to a Dst block, on the way in and on the way out.
IN_DST16_CONVERSIONS = ["BF16InDstToBF16", "FP16InDstToFP16"]
TO_DST16_CONVERSIONS = ["BF16ToDstFormatBF16", "FP16ToDstFormatFP16"]
FP32_CONVERSIONS = [
    "FP32ToBF16",
    "FP32ToDstFormatBF16",
    "FP32ToDstFormatFP32",
    "FP32InDstToFP32",
]


def _fp32_cases():
    """Every equivalence class of the 32-bit conversions, plus random words.

    The FP32 conversions read the sign, the 8-bit exponent, the top 7 mantissa
    bits (all that survives into BF16) and -- for the denormal flush -- only
    *whether* the low 16 bits are set. Sweeping all 2**16 (sign, exp, manHi)
    combinations against five low halves therefore covers every distinct case,
    and ``FP32InDstToFP32`` / ``FP32ToDstFormatFP32``, which pass the low half
    through untouched, are covered by the same sweep. The random words are
    belt-and-braces against that reasoning being wrong.
    """
    hi = np.arange(DST16_SPACE, dtype=np.int64) << 16
    lo = np.array([0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFF], dtype=np.int64)
    structured = (hi[:, np.newaxis] | lo).ravel()
    rng = np.random.default_rng(20260802)
    random = rng.integers(0, 1 << 32, 200_000, dtype=np.int64)
    return np.concatenate([structured, random])


def _assert_elementwise(fn, xs):
    """The array result equals the scalar result, value for value."""
    got = fn(xs)
    assert isinstance(got, np.ndarray), f"{fn.__name__} did not stay vectorised"
    assert got.tolist() == [fn(int(x)) for x in xs]


@pytest.mark.parametrize("name", IN_SRC_CONVERSIONS)
def test_in_src_conversions_over_the_whole_19_bit_space(name):
    _assert_elementwise(getattr(DFC, name), np.arange(SRC_SPACE, dtype=np.int64))


@pytest.mark.parametrize("name", IN_DST16_CONVERSIONS + TO_DST16_CONVERSIONS)
def test_dst16_conversions_over_the_whole_16_bit_space(name):
    _assert_elementwise(getattr(DFC, name), np.arange(DST16_SPACE, dtype=np.int64))


@pytest.mark.parametrize("name", FP32_CONVERSIONS)
def test_fp32_conversions_over_every_equivalence_class(name):
    _assert_elementwise(getattr(DFC, name), _fp32_cases())


def test_denormal_flush_is_the_only_case_the_branch_removal_touched():
    """``FP32ToBF16`` lost its ``if exp == 0`` so it could run over a block.

    The branch became ``man * (exp != 0)``; this pins the two arms directly,
    since the sweep above would pass just as happily if the flush were dropped
    altogether (the mantissa bits it clears are shifted out anyway -- what must
    survive is that a denormal keeps its sign and nothing else).
    """
    assert DFC.FP32ToBF16(0x007FFFFF) == 0x0000  # +denormal -> +0
    assert DFC.FP32ToBF16(0x807FFFFF) == 0x8000  # -denormal -> -0
    assert DFC.FP32ToBF16(0x00800000) == 0x0080  # smallest normal survives
    assert DFC.FP32ToBF16(0xBFC00000) == 0xBFC0


def main():
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        cases = [m.args[1] for m in marks if m.name == "parametrize"]
        for case in cases[0] if cases else [None]:
            fn(case) if case is not None else fn()
    print("conversion_batch_test OK: block conversions match the scalar ones")


if __name__ == "__main__":
    main()
