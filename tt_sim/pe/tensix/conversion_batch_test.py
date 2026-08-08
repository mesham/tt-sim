"""Prove the block form of each Src/Dst conversion equals the scalar one.

``MatrixUnit.perform_mvmul_exact`` gathers a whole rectangle of SrcA/SrcB/Dst
per instruction and converts it in one call, and ``UnPackerUnit._unpack_block``
converts a whole rectangle on the way *into* Src/Dst, both by handing the
*same* ``DataFormatConversions`` classmethod a numpy array instead of an int.
There is therefore no second implementation of the storage-layout bit
permutations to keep in step -- but numpy is not Python: a fixed-width int64
wraps where a Python int widens, ``bool`` arithmetic and ``>>`` on a signed
array have their own rules, and a stray branch would silently take one arm for
the whole block.

So each conversion the batched paths can reach is checked over its **entire**
input space (or, for the FP32-width ones, over every equivalence class of it):
the array result must equal the scalar result value for value, not
approximately and not on a sample. 2**19 is small enough that exhaustive is
simply the cheapest option. ``UnPackerUnit.formatConversion``, which chains
several of them per (input format, output format) pair and holds two branches
of its own, is then checked the same way end to end.

Runs standalone (``python3 -m tt_sim.pe.tensix.conversion_batch_test``) or under
pytest.
"""

from itertools import product

import numpy as np
import pytest

from tt_sim.pe.tensix.backends.backend_base import DataFormat
from tt_sim.pe.tensix.backends.unpacker import UnPackerUnit
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

# The unpack direction: 19-bit inputs (a datum already narrowed to Src width)
# and 16-bit ones (a whole BF16/FP16 datum).
TO_SRC_CONVERSIONS_19B = ["TF32ToSrcTF32", "TF32ToSrcFormatTF32"]
TO_SRC_CONVERSIONS_16B = ["BF16ToSrcBF16", "FP16ToSrcFP16"]

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


@pytest.mark.parametrize("name", IN_SRC_CONVERSIONS + TO_SRC_CONVERSIONS_19B)
def test_in_src_conversions_over_the_whole_19_bit_space(name):
    _assert_elementwise(getattr(DFC, name), np.arange(SRC_SPACE, dtype=np.int64))


@pytest.mark.parametrize("name", TO_SRC_CONVERSIONS_16B)
def test_to_src_conversions_over_the_whole_16_bit_space(name):
    _assert_elementwise(getattr(DFC, name), np.arange(DST16_SPACE, dtype=np.int64))


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


# -- The table-driven block forms -------------------------------------------
#
# ``perform_mvmul_exact`` does not call the arithmetic forms above; it calls the
# ``block*`` family, which gathers the answer out of a table of that same
# arithmetic form over its whole input space. Building a table from the function
# it replaces means it cannot encode a *different* function -- but it can be
# built over the wrong domain, indexed with the wrong bits, or fold in a
# widening shift the caller does not want, none of which the fpu tests would
# catch (they start from FP32 words, downstream of the conversion). So each one
# is swept against what it claims to compute, over the same exhaustive input
# space as the arithmetic forms above.

# (block classmethod, the expression it must equal).
BLOCK_SRC_EQUIVALENTS = [
    ("blockBF16InSrcToFP32", lambda x: DFC.BF16InSrcToFP32(x)),
    ("blockTF32InSrcToFP32", lambda x: DFC.TF32InSrcToFP32(x)),
]
BLOCK_DST16_EQUIVALENTS = [
    # The MVMUL Dst16b read wants an FP32 bit pattern, so the widening shift is
    # folded into the table.
    ("blockBF16InDstToFP32", lambda x: DFC.BF16InDstToBF16(x) << 16),
]
BLOCK_FP32_EQUIVALENTS = [
    ("blockFP32InDstToFP32", lambda x: DFC.FP32InDstToFP32(x)),
    ("blockFP32ToDstFormatBF16", lambda x: DFC.FP32ToDstFormatBF16(x)),
    ("blockFP32ToDstFormatFP32", lambda x: DFC.FP32ToDstFormatFP32(x)),
]


def _assert_block_matches(name, reference, xs):
    got = getattr(DFC, name)(xs)
    assert isinstance(got, np.ndarray), f"{name} did not stay vectorised"
    assert got.dtype == np.int64, f"{name} must stay int64 for the FPU datapath"
    assert np.array_equal(got, reference(xs)), name


@pytest.mark.parametrize("name,reference", BLOCK_SRC_EQUIVALENTS)
def test_block_src_conversions_over_the_whole_19_bit_space(name, reference):
    _assert_block_matches(name, reference, np.arange(SRC_SPACE, dtype=np.int64))


@pytest.mark.parametrize("name,reference", BLOCK_DST16_EQUIVALENTS)
def test_block_dst16_conversions_over_the_whole_16_bit_space(name, reference):
    _assert_block_matches(name, reference, np.arange(DST16_SPACE, dtype=np.int64))


@pytest.mark.parametrize("name,reference", BLOCK_FP32_EQUIVALENTS)
def test_block_fp32_conversions_over_every_equivalence_class(name, reference):
    _assert_block_matches(name, reference, _fp32_cases())


def test_block_src_conversions_ignore_bits_above_the_src_width():
    """Masking the index is not a behaviour change, and this is why.

    Every ``*InSrcTo*`` conversion reads only bits 0..18 (``TF32InSrcToTF32``'s
    masks are 0x40000 / 0x3FF00 / 0xFF), so a datum with rubbish above bit 18
    already converted as though those bits were absent. The table's ``& 0x7FFFF``
    keeps that true rather than raising an out-of-bounds index.
    """
    dirty = np.arange(SRC_SPACE, dtype=np.int64) | (1 << 30)
    clean = np.arange(SRC_SPACE, dtype=np.int64)
    assert np.array_equal(DFC.BF16InSrcToFP32(dirty), DFC.BF16InSrcToFP32(clean))
    assert np.array_equal(
        DFC.blockBF16InSrcToFP32(dirty), DFC.blockBF16InSrcToFP32(clean)
    )


def test_block_tables_are_shared_and_frozen():
    """One table per conversion, built once and not writable.

    A caller that mutated a returned block would otherwise corrupt every later
    conversion, since a gather's result is a fresh array but the table is not.
    """
    from tt_sim.pe.tensix.util import _BLOCK_LUTS, _block_lut

    first = _block_lut("BF16InDstToBF16", 16, DFC.BF16InDstToBF16)
    assert _block_lut("BF16InDstToBF16", 16, DFC.BF16InDstToBF16) is first
    assert not first.flags.writeable
    assert _BLOCK_LUTS["BF16InDstToBF16"] is first


# -- The unpacker's own conversion, end to end ------------------------------
#
# ``UnPackerUnit.formatConversion`` is what ``_unpack_block`` hands a whole
# rectangle of freshly-read datums. It chains the conversions above per
# (InDataFormat, OutDataFormat, unpackToDst) triple and holds two branches that
# had to become arithmetic to survive a block: INT8's "give a non-zero
# magnitude an exponent", and the FP32 -> BF16 denormal flush (now
# ``FP32ToBF16``). Every triple ``_unpack_block`` accepts is swept below; the
# one it declines, FP32 -> FP16, is listed too so the reason stays visible.

# (InDataFormat, OutDataFormat, unpackToDst) for the 8- and 16-bit input
# formats, whose whole input space fits in one sweep.
NARROW_FMT_CASES = [
    (DataFormat.BF16, DataFormat.BF16, False),
    (DataFormat.BF16, DataFormat.BF16, True),
    (DataFormat.FP16, DataFormat.FP16, False),
    (DataFormat.FP16, DataFormat.FP16, True),
    (DataFormat.UINT16, DataFormat.UINT16, False),
    (DataFormat.UINT16, DataFormat.UINT16, True),
    (DataFormat.INT8, DataFormat.INT8, False),
    (DataFormat.INT8, DataFormat.INT8, True),
]

# ... and for the 32-bit ones, swept by equivalence class.
WIDE_FMT_CASES = [
    (DataFormat.FP32, DataFormat.FP32, False),
    (DataFormat.FP32, DataFormat.FP32, True),
    (DataFormat.FP32, DataFormat.TF32, False),
    (DataFormat.FP32, DataFormat.TF32, True),
    (DataFormat.FP32, DataFormat.BF16, False),
    (DataFormat.FP32, DataFormat.BF16, True),
    (DataFormat.INT32, DataFormat.INT32, False),
    (DataFormat.INT32, DataFormat.INT32, True),
    (DataFormat.TF32, DataFormat.TF32, True),
]


def _fp32_unpack_cases():
    """32-bit datums for the unpack conversions, by equivalence class.

    No unpack conversion mixes the two halves of a 32-bit datum: each is a
    shift, a mask or a straight passthrough of the low half, and the only
    data-dependent choice (the FP32 -> BF16 denormal flush) reads the exponent,
    which lives entirely in the high half. So a full sweep of the 2**16 high
    halves against a rotating low half covers every class, in a sixth of the
    values the cross product of ``_fp32_cases`` would need. Random words guard
    the reasoning.
    """
    hi = np.arange(DST16_SPACE, dtype=np.int64) << 16
    lo = np.array([0x0000, 0x0001, 0x7FFF, 0x8000, 0xFFFF], dtype=np.int64)
    structured = hi | lo[np.arange(DST16_SPACE) % lo.size]
    rng = np.random.default_rng(20260802)
    return np.concatenate(
        [structured, rng.integers(0, 1 << 32, 50_000, dtype=np.int64)]
    )


class _FormatConverter:
    """``formatConversion`` with the two attributes it actually reads.

    It is the real method, not a copy -- only ``getConfigValue`` (the INT8
    signedness bit) and ``unpacker_id`` (which of the two SrcA/SrcB signedness
    registers that bit comes from) are stubbed.
    """

    formatConversion = UnPackerUnit.formatConversion

    def __init__(self, unpacker_id=0, int8_unsigned=0):
        self.unpacker_id = unpacker_id
        self._int8_unsigned = int8_unsigned

    def getConfigValue(self, stateID, key, num=1):
        assert key.endswith("Unsigned")
        return self._int8_unsigned


def _assert_format_conversion(unit, inFmt, outFmt, toDst, xs):
    def convert(x):
        return unit.formatConversion(0, inFmt, outFmt, x, toDst)

    convert.__name__ = f"formatConversion[{inFmt.name}->{outFmt.name}]"
    _assert_elementwise(convert, xs)


@pytest.mark.parametrize("inFmt,outFmt,toDst", NARROW_FMT_CASES)
@pytest.mark.parametrize("int8_unsigned", [0, 1])
def test_unpacker_narrow_format_conversion_over_the_whole_space(
    inFmt, outFmt, toDst, int8_unsigned
):
    width = 8 if inFmt == DataFormat.INT8 else 16
    _assert_format_conversion(
        _FormatConverter(int8_unsigned=int8_unsigned),
        inFmt,
        outFmt,
        toDst,
        np.arange(1 << width, dtype=np.int64),
    )


@pytest.mark.parametrize("inFmt,outFmt,toDst", WIDE_FMT_CASES)
def test_unpacker_wide_format_conversion_over_every_equivalence_class(
    inFmt, outFmt, toDst
):
    _assert_format_conversion(
        _FormatConverter(), inFmt, outFmt, toDst, _fp32_unpack_cases()
    )


def test_fp32_to_fp16_is_the_case_the_block_path_declines():
    """The one unpack conversion still written with per-datum branches.

    ``FP32ToFP16`` saturates on overflow and flushes on underflow with an
    if/elif, which numpy refuses to evaluate for a whole block. So
    ``_unpack_block`` declines this pair and the scalar loop takes it; this
    pins that the refusal is real rather than a precaution, i.e. that the pair
    cannot quietly start taking one arm for a whole block.
    """
    unit = _FormatConverter()
    # 1.0, a value that saturates, and one that flushes -- one input per arm.
    xs = np.array([0x3F800000, 0x7F000000, 0x00000001], dtype=np.int64)
    with pytest.raises(ValueError, match="truth value of an array"):
        unit.formatConversion(0, DataFormat.FP32, DataFormat.FP16, xs, False)
    # Scalar, i.e. one arm each, as SrcA FP16 words (Sign,Man(10b),Zero,Exp).
    assert [
        unit.formatConversion(0, DataFormat.FP32, DataFormat.FP16, int(x), False)
        for x in xs
    ] == [0x0000F, 0x3FF1F, 0x00000]


def main():
    """Run every test without pytest, expanding ``parametrize`` as pytest would."""
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        grids = []
        for mark in getattr(fn, "pytestmark", []):
            if mark.name != "parametrize":
                continue
            argnames = [n.strip() for n in mark.args[0].split(",")]
            grids.append(
                [
                    dict(zip(argnames, values if len(argnames) > 1 else (values,)))
                    for values in mark.args[1]
                ]
            )
        for combination in product(*grids):
            kwargs = {}
            for case in combination:
                kwargs.update(case)
            fn(**kwargs)
    print("conversion_batch_test OK: block conversions match the scalar ones")


if __name__ == "__main__":
    main()
