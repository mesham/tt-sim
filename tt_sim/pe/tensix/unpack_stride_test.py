"""Tests for the UNPACR input walk: the strided one, and the modes still declined.

``UNPACR_Regular.md`` lets the input rows be *discontiguous*: with
``Tileize_mode`` set, the input address advances by ``RowStride`` bytes every
``UnpackRowWidth`` datums instead of abutting -- input datum ``i`` comes from
``InAddr_Datums + DatumSizeBytes * (i % UnpackRowWidth) + RowStride * (i /
UnpackRowWidth)``. That walk is what tt-metal's ``tilize`` LLK uses to gather a
row-major block into faces, and it is modelled here. ``Upsample_rate`` (which
spreads each input datum over several output positions) and -- with
``Tileize_mode`` clear -- ``ColShift`` (which slides each datum towards column 0,
dropping the ones that fall off the near edge) still are not, and are rejected at
decode rather than silently ignored. These pin the boundary:

* the *contiguous* RowStride is ``DatumSizeBytes * UnpackRowWidth``, and
  ``UnpackRowWidth`` is 16 on Wormhole but 32 on Blackhole for anything wider
  than a byte per datum -- so the same Tileize_mode configuration reads a
  different L1 pattern on each architecture, and the walk has to know which;
* ``RowStride`` is assembled from three 4-bit ``Shift_amount_cntx`` fields at
  shifts 4, 8 and 12 (max 65520 bytes);
* the batched ``_unpack_block`` gathers the strided source with an index array,
  and must agree with the scalar loop datum for datum, overlapping strides
  included;
* ``Upsample_and_interleave`` on its own, at ``Upsample_rate == 0``, is a no-op
  -- the doc's inner loop runs exactly once -- so it is accepted, not rejected;
* the *same* ``Shift_amount_cntx0`` field is the ``RowStride`` when
  ``Tileize_mode`` is set and the ``ColShift`` when it is clear, so the
  ``ColShift`` rejection is told apart from a legitimate stride by
  ``Tileize_mode`` and nothing else.

Every case drives a real backend through ``read_unpack_state`` /
``perform_unpack_state``, so the config decode, the checks and (for the accepted
cases) the datum walk itself are all exercised.

Runs standalone (``python3 -m tt_sim.pe.tensix.unpack_stride_test``) or under
pytest.
"""

import struct
from contextlib import contextmanager
from itertools import product

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import DataFormatConversions, TensixConfigurationConstants

BF16 = 5  # DataFormat.BF16, as it appears in a tile descriptor / REG2

#: Where the tile lives in L1, and the ``REG3_Base_address`` that names it
#: (``InAddr = (Base + Offset + 1) * 16``).
IN_ADDR = 0x1000
BASE_ADDRESS = IN_ADDR // 16 - 1

#: Rows of 16 datums to unpack. 4 is enough for a stride to be visible and
#: leaves SrcA rows 4..63 untouched.
NUM_ROWS = 4

#: The UNPACR argument bits, all of them off: no row search, no context
#: counting, no zero-write, no dvalid flip, single-context mode.
UNPACR_ARGS = {
    "RowSearch": 0,
    "AutoIncContextID": 0,
    "ZeroWrite2": 0,
    "SetDatValid": 0,
    "OvrdThreadId": 0,
    "AddrCntContextId": 0,
    "CfgContextId": 0,
    "AddrMode": 0,
}


class _L1:
    """Just enough addressable memory for the unpacker to read a tile out of.

    Big enough for the largest encodable RowStride (65520 bytes) times the rows
    unpacked here, and poisoned rather than zeroed so that a datum read from the
    wrong address is a *wrong* value rather than a plausible zero.
    """

    POISON = 0xA5

    def __init__(self, size=1 << 19):
        self.data = bytearray([_L1.POISON]) * size

    def read(self, addr, size):
        return bytes(self.data[addr - IN_ADDR : addr - IN_ADDR + size])

    def write(self, addr, payload):
        self.data[addr - IN_ADDR : addr - IN_ADDR + len(payload)] = payload


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


def _set_thread_config(backend, key, value, thread=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    cfg = backend.config_unit.threadConfig[thread]
    cfg[addr32] = (cfg[addr32] & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)


@contextmanager
def _unpacker(
    blackhole=False, tileize=0, shifts=(0, 0, 0), upsample=(0, 0), data_format=BF16
):
    """A backend configured for a BF16 -> BF16 unpack of ``NUM_ROWS`` rows into SrcA.

    The config-register layout is a process-global selection, so restore the
    Wormhole layout on the way out -- other tests in the same session (and the
    Wormhole replay guards) expect it.
    """
    try:
        backend = TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
            BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
            blackhole=blackhole,
        ).getBackend()
        memory = _L1()
        backend.setAddressableMemory(memory)

        # Tile descriptor: BF16 datums, X=16, Y=Z=W=1, uncompressed.
        descriptor = TensixConfigurationConstants.get_addr32(
            "THCON_SEC0_REG0_TileDescriptor"
        )
        backend.config_unit.setConfig(0, descriptor + 0, (16 << 16) | data_format)
        backend.config_unit.setConfig(0, descriptor + 1, (1 << 16) | 1)
        backend.config_unit.setConfig(0, descriptor + 2, 1)
        backend.config_unit.setConfig(0, descriptor + 3, 0)

        _set_config(backend, "THCON_SEC0_REG3_Base_address", BASE_ADDRESS)
        _set_config(backend, "THCON_SEC0_REG2_Out_data_format", data_format)
        # Byte address 128 -> OutAddr 64 -> SrcA start row 0 (unpacker 0's OutAddr
        # counts from row 4).
        _set_config(backend, "UNP0_ADDR_BASE_REG_1_Base", 128)
        _set_thread_config(backend, "SRCA_SET_SetOvrdWithAddr", 1)

        _set_config(backend, "THCON_SEC0_REG2_Tileize_mode", tileize)
        for index, shift in enumerate(shifts):
            _set_config(backend, f"THCON_SEC0_REG2_Shift_amount_cntx{index}", shift)
        _set_config(backend, "THCON_SEC0_REG2_Upsample_rate", upsample[0])
        _set_config(backend, "THCON_SEC0_REG2_Upsample_and_interleave", upsample[1])

        # ADC channel 1 X is the last datum index to read (inclusive).
        backend.getADC(0).Unpacker[0].Channel[1].X = NUM_ROWS * 16 - 1
        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _bf16_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def _datum(row, col):
    """A distinct value per (row, column), so a misread address is visible."""
    return float(row * 16 + col + 1)


def _fill(memory, stride):
    for row in range(NUM_ROWS):
        for col in range(16):
            memory.write(
                IN_ADDR + row * stride + col * 2,
                struct.pack("<H", _bf16_bits(_datum(row, col))),
            )


def _row_width(blackhole):
    """``UnpackRowWidth`` for a 2-byte datum: 16 on Wormhole, 32 on Blackhole."""
    return 32 if blackhole else 16


def _walk_addr(index, stride, width):
    """Where the doc says input datum ``index`` is read from."""
    return IN_ADDR + 2 * (index % width) + stride * (index // width)


def _fill_walk(memory, stride, width):
    """Lay the tile out so that the ``(stride, width)`` walk sees ``_expected()``.

    Datum ``i`` of the walk carries the value the unpack should land in SrcA row
    ``i // 16``, column ``i % 16`` -- so a walk that reads a different address
    picks up the poison, or another row's datum, and the comparison fails.
    """
    for index in range(NUM_ROWS * 16):
        memory.write(
            _walk_addr(index, stride, width),
            struct.pack("<H", _bf16_bits(_datum(index // 16, index % 16))),
        )


def _walk_of(memory, stride, width):
    """What SrcA must hold if the unpack walked L1 with this stride and width."""
    return [
        [
            DataFormatConversions.BF16ToSrcBF16(
                struct.unpack(
                    "<H", memory.read(_walk_addr(row * 16 + col, stride, width), 2)
                )[0]
            )
            for col in range(16)
        ]
        for row in range(NUM_ROWS)
    ]


def _unpack(backend):
    unpacker = backend.unpacker_units[0]
    unpacker.perform_unpack_state(0, unpacker.read_unpack_state(0, UNPACR_ARGS))
    srcA = backend.getSrcA(0)
    return [[srcA[row, col] for col in range(16)] for row in range(NUM_ROWS)]


def _expected():
    return [
        [
            DataFormatConversions.BF16ToSrcBF16(_bf16_bits(_datum(row, col)))
            for col in range(16)
        ]
        for row in range(NUM_ROWS)
    ]


def _contiguous_stride(blackhole):
    # BF16 is 2 bytes, and UnpackRowWidth is 32 on Blackhole above 1 byte/datum.
    return 2 * (32 if blackhole else 16)


@pytest.mark.parametrize("blackhole", [False, True])
def test_contiguous_unpack_moves_the_datums(blackhole):
    """The baseline the checks are calibrated against: rows abutting in L1."""
    with _unpacker(blackhole) as (backend, memory):
        _fill(memory, stride=32)
        assert _unpack(backend) == _expected()


@pytest.mark.parametrize("blackhole", [False, True])
def test_tileize_with_a_contiguous_row_stride_is_accepted(blackhole):
    """Tileize_mode is fine as long as its RowStride is the abutting one.

    The check is "the stride equals the contiguous value", not "the stride is
    zero" -- and the contiguous value is arch-dependent, so the *same* stride
    that is accepted here would be rejected on the other architecture.
    """
    stride = _contiguous_stride(blackhole)
    with _unpacker(blackhole, tileize=1, shifts=(stride >> 4, 0, 0)) as (
        backend,
        memory,
    ):
        _fill(memory, stride=32)
        assert _unpack(backend) == _expected()


def _decline(*args, **kwargs):
    """Stand-in for ``_unpack_block`` that always refuses the batched path."""
    return False


def _shifts_for(stride):
    """The three 4-bit ``Shift_amount_cntx`` fields that encode ``stride``."""
    assert stride % 16 == 0
    assert stride <= 65520
    return ((stride >> 4) & 0xF, (stride >> 8) & 0xF, (stride >> 12) & 0xF)


@pytest.mark.parametrize("blackhole", [False, True])
@pytest.mark.parametrize("stride", [96, 128, 160, 4096, 65520])
def test_tileize_with_a_strided_row_stride_reads_the_strided_rows(blackhole, stride):
    """The point of the whole exercise: a non-abutting stride lands the right datums.

    ``_fill_walk`` writes each datum where the doc's walk says the unpacker
    should look for it, so the assertion is precisely "the walk read from
    ``InAddr + 2*(i % UnpackRowWidth) + RowStride*(i / UnpackRowWidth)``". Every
    stride here exceeds the contiguous one on both architectures, so the input
    rows are genuinely apart in L1 and everything between them is poison.
    """
    width = _row_width(blackhole)
    with _unpacker(blackhole, tileize=1, shifts=_shifts_for(stride)) as (
        backend,
        memory,
    ):
        _fill_walk(memory, stride, width)
        assert _unpack(backend) == _expected()


@pytest.mark.parametrize("blackhole", [False, True])
def test_the_strided_walk_uses_this_architectures_row_width(blackhole):
    """16 datums per input row on Wormhole, 32 on Blackhole -- from the same config.

    The L1 image is laid out for a *Wormhole* walk (16 datums then a jump), so
    Wormhole must reproduce ``_expected()`` and Blackhole -- which reads 32
    datums before jumping -- must reproduce the different, but equally
    well-defined, image a 32-wide walk sees. Hardcoding either width would fail
    on one architecture, and the two answers differ, so this cannot pass by
    accident.
    """
    stride = 256
    with _unpacker(blackhole, tileize=1, shifts=_shifts_for(stride)) as (
        backend,
        memory,
    ):
        _fill_walk(memory, stride, width=16)
        result = _unpack(backend)
        assert result == _walk_of(memory, stride, _row_width(blackhole))
        assert result != _walk_of(memory, stride, _row_width(not blackhole))
        if not blackhole:
            assert result == _expected()


@pytest.mark.parametrize("blackhole", [False, True])
@pytest.mark.parametrize("stride", [16, 32, 48, 64, 96, 4096])
def test_the_batched_strided_gather_matches_the_scalar_walk(blackhole, stride):
    """``_unpack_block``'s index array must agree with the datum loop exactly.

    The batched path reads one span of L1 and gathers it with an arithmetic
    index array; the scalar loop walks a byte cursor. They are two
    implementations of one walk, so they are diffed here datum for datum, over
    strides shorter than a row (input rows *overlap*, and a datum is read
    twice), equal to it, and longer.
    """
    results = []
    for batched in (True, False):
        with _unpacker(blackhole, tileize=1, shifts=_shifts_for(stride)) as (
            backend,
            memory,
        ):
            _fill_walk(memory, stride, _row_width(blackhole))
            if not batched:
                # Decline the block path, exactly as it declines a BFP width or
                # an FP32 -> FP16 conversion, so the scalar loop runs instead.
                backend.unpacker_units[0]._unpack_block = _decline
            results.append(_unpack(backend))
            image = _walk_of(memory, stride, _row_width(blackhole))
    assert results[0] == results[1]
    assert results[0] == image


@pytest.mark.parametrize(
    "shifts,stride",
    [
        ((2, 0, 0), 32),
        ((0, 1, 0), 256),
        ((0, 0, 1), 4096),
        ((0xF, 0xF, 0xF), 65520),
    ],
)
def test_row_stride_is_assembled_from_the_three_shift_fields(shifts, stride):
    """RowStride = cntx0 << 4 | cntx1 << 8 | cntx2 << 12 (UNPACR_Regular.md).

    Observed through the walk itself: the tile is laid out at the stride the
    three fields are meant to encode, so a decode that dropped ``cntx1`` into
    ``cntx0``'s bits (as it once did) reads the poison instead of the datums.
    """
    with _unpacker(tileize=1, shifts=shifts) as (backend, memory):
        _fill_walk(memory, stride, width=16)
        assert _unpack(backend) == _expected()


def test_tileize_with_a_sub_byte_datum_raises():
    """BFP4/BFP2 under Tileize_mode is UndefinedBehavior, and unaddressable here.

    ``DatumSizeBytes`` rounds to 0, so the walk cannot index L1 in datums at
    all; the doc says the combination is undefined on hardware anyway.
    """
    BFP4_B = 7
    with _unpacker(tileize=1, shifts=_shifts_for(256), data_format=BFP4_B) as (
        backend,
        memory,
    ):
        _fill(memory, stride=32)
        with pytest.raises(NotImplementedError, match="sub-byte"):
            _unpack(backend)


@pytest.mark.parametrize("rate", [1, 2, 3])
def test_upsample_rate_raises(rate):
    """Upsampling emits several output datums per input datum; the walk emits one."""
    with _unpacker(upsample=(rate, 0)) as (backend, memory):
        _fill(memory, stride=32)
        with pytest.raises(NotImplementedError) as excinfo:
            _unpack(backend)
        message = str(excinfo.value)
        assert f"Upsample_rate={rate}" in message
        assert f"UpsampleZeroes={(1 << rate) - 1}" in message


def test_upsample_interleave_alone_is_accepted():
    """At Upsample_rate 0 the doc's upsample loop runs once, so interleave is a no-op.

    Rejecting it would be over-strict: it changes nothing about where the datums
    land.
    """
    with _unpacker(upsample=(0, 1)) as (backend, memory):
        _fill(memory, stride=32)
        assert _unpack(backend) == _expected()


@pytest.mark.parametrize("blackhole", [False, True])
@pytest.mark.parametrize("shift", [1, 2, 15])
def test_col_shift_raises(blackhole, shift):
    """ColShift slides each datum left and drops what falls off; the walk does not.

    Without ``Tileize_mode``, ``Shift_amount_cntx0`` is the ColShift rather than
    the RowStride, and the doc's walk is ``if (Row < 4 || Col < ColShift)
    continue; Col -= ColShift;`` -- so the datums in the first ``ColShift``
    columns are *discarded* and the last ``ColShift`` columns of every SrcA row
    are left as they were.
    """
    with _unpacker(blackhole, shifts=(shift, 0, 0)) as (backend, memory):
        _fill(memory, stride=32)
        with pytest.raises(NotImplementedError) as excinfo:
            _unpack(backend)
        message = str(excinfo.value)
        assert f"ColShift={shift}" in message
        assert "drops" in message


@pytest.mark.parametrize("blackhole", [False, True])
def test_col_shift_raises_before_any_datum_moves(blackhole):
    """The regression guard for what the unguarded ``Col -= ColShift`` used to do.

    ``outCol = col - ColShift`` is a negative index for the datums the doc drops,
    and numpy wraps a negative column onto the *far* end of the row: with
    ColShift 2 the datums from columns 0 and 1 landed in columns 14 and 15,
    clobbering the two columns the doc says to leave untouched, on both the
    scalar walk and the batched ``_unpack_block``. The rejection is raised from
    ``read_unpack_state``, i.e. at decode and before either walk runs, so SrcA is
    still exactly as it was.
    """
    with _unpacker(blackhole, shifts=(2, 0, 0)) as (backend, memory):
        _fill(memory, stride=32)
        srcA = backend.getSrcA(0)
        with pytest.raises(NotImplementedError):
            backend.unpacker_units[0].read_unpack_state(0, UNPACR_ARGS)
        assert [[srcA[row, col] for col in range(16)] for row in range(NUM_ROWS)] == [
            [0] * 16 for _ in range(NUM_ROWS)
        ]


@pytest.mark.parametrize("blackhole", [False, True])
def test_col_shift_reads_only_the_selected_context(blackhole):
    """ColShift is one context's field, not the three assembled into a RowStride.

    ``Shift_amount_cntx1`` and ``cntx2`` are the upper bits of the RowStride when
    Tileize_mode is set, but with it clear they belong to contexts 1 and 2 -- so
    a single-context unpack reading context 0 must ignore them rather than
    rejecting on a stride-shaped assembly of all three.
    """
    with _unpacker(blackhole, shifts=(0, 0xF, 0xF)) as (backend, memory):
        _fill(memory, stride=32)
        assert _unpack(backend) == _expected()


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
    print("unpack_stride_test OK: unmodelled unpack walks raise, contiguous ones run")


if __name__ == "__main__":
    main()
