"""Tests for the UNPACR *output* address generator, and the untilize mode built on it.

``unpack_stride_test.py`` pins the input walk (where in L1 each datum is read
from). This is its output-side twin: where in SrcA each datum *lands*.
``UNPACR_Regular.md`` derives both destination coordinates from one running
counter --

    OutAddr = UNP_ADDR_BASE_REG_1_Base + ADC_Out.Y * ADDR_CTRL_XY_REG_1_Ystride
                                       + ADC_Out.Z * ADDR_CTRL_ZW_REG_1_Zstride
                                       + ADC_Out.W * ADDR_CTRL_ZW_REG_1_Wstride
    ...
    Row = OutAddr / 16;  Col = OutAddr & 15;   // OutAddr advances per datum

-- so the row an UNPACR *starts* at is part of every datum's row index. That is
the whole mechanism behind ``llk_unpack_untilize`` (the deprecated CB -> CB
``untilize_block``, still the tt-xftn compiler team's codegen target): it leaves
the input walk contiguous and instead

* rewrites ``UNP0_ADDR_CTRL_XY_REG_1_Ystride`` to one face row (16 datums),
* sets the tile descriptor's ``YDim`` and ``Tile_x_dim_cntx`` to 16, so the
  channel-0 Z stride ``XDim * YDim`` becomes a whole 16x16 face,
* and issues UNPACRs with ``AddrMode`` = ch0 Z += 1, ch1 Y += 1.

Face rows therefore arrive scattered down SrcA -- row ``i`` of SrcA takes face
``i % 2``'s row ``i / 2`` -- which *is* the row-major interleave of a tile's four
faces. Nothing about ``Tileize_mode``/``RowStride`` is involved.

The two defects these pin, both of which produced the identical wrong tile on
Wormhole and Blackhole:

* the tile descriptor was read with a stride of *four* config words instead of
  one, so ``YDim`` came from ``THCON_SEC0_REG1``. For an ordinary tile that
  register happens to yield the right ``ZDim`` and a ``YDim`` of 0 (read as 1),
  which is also the right answer -- so it only showed up once untilize set
  ``YDim`` to 16 and every UNPACR read the wrong face;
* the start row was dropped on the ``SRCA_SET_SetOvrdWithAddr`` path (the one
  every current LLK takes), collapsing all sixteen UNPACRs onto SrcA row 0.

Every case drives a real backend through ``read_unpack_state`` /
``perform_unpack_state`` on both architectures, and each asserts against the
*specific* wrong image the pre-change walk produced as well as the right one.

Runs standalone (``python3 -m tt_sim.pe.tensix.unpack_untilize_test``) or under
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

#: Where the tile lives in L1, and the ``REG3_Base_address`` naming it
#: (``InAddr = (Base + Offset + 1) * 16``).
IN_ADDR = 0x2000
BASE_ADDRESS = IN_ADDR // 16 - 1

FACE_DIM = 16  #: rows and columns per face
FACE_DATUMS = FACE_DIM * FACE_DIM
TILE_DATUMS = 4 * FACE_DATUMS

#: One untilize pass unpacks the sixteen rows of a tile's top (or bottom) two
#: faces into SrcA rows 0..15, one 16-datum row per UNPACR.
NUM_UNPACRS = 16

#: ``TT_SETADCXX(UNP_A, 15, 0)``: sixteen datums, i.e. one face row, per UNPACR.
DATUMS_PER_UNPACR = 16

#: ``UNP0_ADDR_CTRL_XY_REG_1_Ystride`` = ``FACE_R_DIM * datum_size`` bytes.
Y_STRIDE_BYTES = FACE_DIM * 2

#: ``THCON_SEC0_REG5_Dest_cntx0_address``. Unpacker 0's output address counts
#: from row 4, so 64 datums is "SrcA row 0" -- the LLK's usual value.
DEST_ADDRESS = 64

#: The MOP's ``TTI_UNPACR(SrcA, 0b01000001, ...)``: ch0 Z += 1, ch1 Y += 1.
ADDR_MODE_UNTILIZE = 0b01000001

#: ``OvrdThreadId`` (multi-context mode) is set; the rest are off, matching the
#: untilize MOP's UNPACR.
UNPACR_ARGS = {
    "RowSearch": 0,
    "AutoIncContextID": 0,
    "ZeroWrite2": 0,
    "SetDatValid": 0,
    "OvrdThreadId": 1,
    "AddrCntContextId": 0,
    "CfgContextId": 0,
    "AddrMode": ADDR_MODE_UNTILIZE,
}


class _L1:
    """Just enough addressable memory to hold one tile, poisoned rather than zeroed.

    A datum read from the wrong address is then a *wrong* value rather than a
    plausible zero.
    """

    POISON = 0xA5

    def __init__(self, size=1 << 16):
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


def _descriptor_addr32():
    return TensixConfigurationConstants.get_addr32("THCON_SEC0_REG0_TileDescriptor")


@contextmanager
def _backend(blackhole=False):
    """A backend wired exactly as ``_llk_unpack_untilize_init_`` leaves the unpacker.

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

        # Tile descriptor. Word 0 is BF16 + uncompressed (XDim there is ignored
        # in multi-context mode); word 1 is YDim = 16 (the value untilize
        # writes) in bits 0..7 and ZDim = 4 in bits 16..23; words 2 and 3 are
        # zero (WDim defaults to 1, no blobs, no digest).
        descriptor = _descriptor_addr32()
        backend.config_unit.setConfig(0, descriptor + 0, (1 << 4) | BF16)
        backend.config_unit.setConfig(0, descriptor + 1, (4 << 16) | FACE_DIM)
        backend.config_unit.setConfig(0, descriptor + 2, 0)
        backend.config_unit.setConfig(0, descriptor + 3, 0)

        _set_config(backend, "THCON_SEC0_REG3_Base_address", BASE_ADDRESS)
        _set_config(backend, "THCON_SEC0_REG2_Out_data_format", BF16)
        # Multi-context mode with context 0: XDim comes from the cntx0 register
        # (untilize sets it to 16) and the output address is the ADC-derived one
        # *plus* Dest_cntx0_address, since add_dest_addr_cntr is set.
        _set_config(backend, "THCON_SEC0_REG5_Tile_x_dim_cntx0", FACE_DIM)
        _set_config(backend, "THCON_SEC0_REG5_Dest_cntx0_address", DEST_ADDRESS)
        _set_config(backend, "UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr", 1)
        _set_config(backend, "UNP0_ADDR_CTRL_XY_REG_1_Ystride", Y_STRIDE_BYTES)
        _set_thread_config(backend, "SRCA_SET_SetOvrdWithAddr", 1)

        # TT_SETADCXX(UNP_A, 15, 0): one face row of datums per UNPACR.
        backend.getADC(0).Unpacker[0].Channel[0].X = 0
        backend.getADC(0).Unpacker[0].Channel[1].X = DATUMS_PER_UNPACR - 1
        yield backend, memory
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _bf16_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def _datum(index):
    """A distinct value per linear datum index, so a misread address is visible."""
    return float(index + 1)


def _fill_tile(memory):
    """Write the tile as the packer left it: four 16x16 faces, each row-major."""
    for index in range(TILE_DATUMS):
        memory.write(IN_ADDR + index * 2, struct.pack("<H", _bf16_bits(_datum(index))))


def _src(value):
    return DataFormatConversions.BF16ToSrcBF16(_bf16_bits(value))


def _read_srca(backend, rows=NUM_UNPACRS):
    srcA = backend.getSrcA(0)
    return [[srcA[row, col] for col in range(16)] for row in range(rows)]


def _run_untilize_pass(backend, first_pass=True):
    """Issue one pass of the untilize MOP, stepping the counters as it does.

    Per iteration the MOP runs two UNPACRs (each bumping ch0 Z and ch1 Y via
    ``AddrMode``), then ``ADDRCRZW`` rewinds ch0 Z and ``INCADCXY`` advances
    ch0 Y -- so ch0 walks (Y = face row, Z = which face) while ch1 Y counts
    output rows straight up.
    """
    unpacker = backend.unpacker_units[0]
    adc0 = backend.getADC(0).Unpacker[0].Channel[0]
    adc0.Y = 0
    adc0.Z = 0 if first_pass else 2
    backend.getADC(0).Unpacker[0].Channel[1].Y = 0
    for pair in range(NUM_UNPACRS // 2):
        for _ in range(2):
            unpacker.perform_unpack_state(0, unpacker.read_unpack_state(0, UNPACR_ARGS))
        adc0.Z = 0 if first_pass else 2  # ADDRCRZW
        adc0.Y = pair + 1  # INCADCXY
    return _read_srca(backend)


def _expected_untilize(first_pass=True):
    """SrcA row ``i`` is face ``i % 2``'s row ``i / 2`` -- the row-major interleave.

    FirstDatum is ``((W * ZDim + Z) * YDim + Y) * XDim + X`` with XDim = YDim =
    16, so ch0 (Y = i / 2, Z = i % 2 [+ 2 for the bottom faces]) selects exactly
    that, and OutAddr puts it at SrcA row ch1.Y = i.
    """
    base_face = 0 if first_pass else 2
    return [
        [
            _src(_datum((base_face + i % 2) * FACE_DATUMS + (i // 2) * FACE_DIM + col))
            for col in range(16)
        ]
        for i in range(NUM_UNPACRS)
    ]


def _image_without_start_row(first_pass=True):
    """What the walk produced before: every UNPACR onto SrcA row 0.

    The last UNPACR of the pass wins row 0; rows 1..15 are never written.
    """
    rows = [[0] * 16 for _ in range(NUM_UNPACRS)]
    rows[0] = _expected_untilize(first_pass)[NUM_UNPACRS - 1]
    return rows


def _image_with_ydim_one(first_pass=True):
    """What the walk produced when the descriptor was read with a stride of four.

    ``THCON_SEC0_REG1`` stood in for descriptor word 1, giving YDim = 1, so the
    channel-0 Z stride was one face *row* rather than a whole face and both
    UNPACRs of a pair read from face 0.
    """
    base_face = 0 if first_pass else 2
    return [
        [
            _src(
                _datum(
                    base_face * FACE_DATUMS
                    + (i % 2) * FACE_DIM
                    + (i // 2) * FACE_DIM
                    + col
                )
            )
            for col in range(16)
        ]
        for i in range(NUM_UNPACRS)
    ]


@pytest.mark.parametrize("blackhole", [False, True])
@pytest.mark.parametrize("first_pass", [True, False])
def test_untilize_scatters_face_rows_down_srca(blackhole, first_pass):
    """The mode itself: sixteen UNPACRs interleave two faces into sixteen SrcA rows.

    Both halves of the mechanism are load-bearing here and are checked against
    the exact image each defect produced, so neither can pass by accident: drop
    the start row and every UNPACR lands on row 0; read the descriptor with the
    old four-word stride and YDim is 1, so the pair reads two rows of the *same*
    face instead of the same row of two faces.
    """
    with _backend(blackhole) as (backend, memory):
        _fill_tile(memory)
        result = _run_untilize_pass(backend, first_pass)
        assert result == _expected_untilize(first_pass)
        assert result != _image_without_start_row(first_pass)
        assert result != _image_with_ydim_one(first_pass)


@pytest.mark.parametrize("blackhole", [False, True])
def test_the_tile_descriptor_is_four_consecutive_config_words(blackhole):
    """XDim/YDim/ZDim/WDim live at bits 16-31, 32-39, 48-55 and 64-71 of one string.

    That is config words n, n+1, n+1 and n+2 -- consecutive. Reading every
    fourth word instead landed on THCON_SEC0_REG1 / REG2 / REG3, which is why
    ``YDim`` silently came from a register holding an exponent-section size.
    """
    with _backend(blackhole) as (backend, _memory):
        descriptor = _descriptor_addr32()
        for word in range(4):
            backend.config_unit.setConfig(0, descriptor + word, 0xC0DE0000 | word)
        for word in range(1, 4):
            backend.config_unit.setConfig(0, descriptor + 4 * word, 0xBAD00000 | word)
        assert backend.getConfigValue(0, "THCON_SEC0_REG0_TileDescriptor", 4) == [
            0xC0DE0000 | word for word in range(4)
        ]


@pytest.mark.parametrize("blackhole", [False, True])
def test_the_ydim_the_descriptor_carries_is_the_one_used(blackhole):
    """YDim scales the channel-0 Z stride, so changing it moves every UNPACR's read.

    Independent of the untilize sequence: one UNPACR at ch0 Z = 1 reads
    ``XDim * YDim`` datums into the tile, which is a whole face at YDim = 16 and
    a single row at YDim = 1.
    """
    for ydim, expected_offset in ((FACE_DIM, FACE_DATUMS), (1, FACE_DIM)):
        with _backend(blackhole) as (backend, memory):
            _fill_tile(memory)
            descriptor = _descriptor_addr32()
            backend.config_unit.setConfig(0, descriptor + 1, (4 << 16) | ydim)
            backend.getADC(0).Unpacker[0].Channel[0].Z = 1
            unpacker = backend.unpacker_units[0]
            unpacker.perform_unpack_state(0, unpacker.read_unpack_state(0, UNPACR_ARGS))
            assert _read_srca(backend, rows=1) == [
                [_src(_datum(expected_offset + col)) for col in range(16)]
            ]


def _decline(*args, **kwargs):
    """Stand-in for ``_unpack_block`` that always refuses the batched path."""
    return False


@pytest.mark.parametrize("blackhole", [False, True])
def test_the_batched_and_scalar_walks_agree_on_the_untilize_pass(blackhole):
    """``_unpack_block``'s row index array must match the datum loop's arithmetic.

    The batched path took this mode (nothing here is a BFP width or an
    ``FP32 -> FP16`` conversion), so both paths have to place the start row the
    same way; the scalar loop is forced by declining the block, exactly as it
    declines those cases.
    """
    results = []
    for batched in (True, False):
        with _backend(blackhole) as (backend, memory):
            _fill_tile(memory)
            if not batched:
                backend.unpacker_units[0]._unpack_block = _decline
            results.append(_run_untilize_pass(backend))
    assert results[0] == results[1]
    assert results[0] == _expected_untilize()


@pytest.mark.parametrize("blackhole", [False, True])
def test_a_start_row_past_the_end_of_srca_wraps(blackhole):
    """SrcA is 64 rows and the row index is six bits, so row 64 is row 0.

    Blackhole's UNPACR_Regular.md says so outright; Wormhole's calls it
    UndefinedBehavior, but its own LLK depends on it -- the SrcA clear ahead of
    the int32/fp32 SFPU kernels arrives with ADC channel 1's Z accumulated to
    exactly 64 rows.
    """
    with _backend(blackhole) as (backend, memory):
        _fill_tile(memory)
        # 64 rows past the start: Zstride is in bytes, and OutAddr is halved for
        # a 16-bit output format.
        _set_config(backend, "UNP0_ADDR_CTRL_ZW_REG_1_Zstride", 64 * 16 * 2)
        backend.getADC(0).Unpacker[0].Channel[1].Z = 1
        unpacker = backend.unpacker_units[0]
        unpacker.perform_unpack_state(0, unpacker.read_unpack_state(0, UNPACR_ARGS))
        assert _read_srca(backend, rows=1) == [[_src(_datum(col)) for col in range(16)]]


@pytest.mark.parametrize("blackhole", [False, True])
@pytest.mark.parametrize("offset", [1, 8, 15])
def test_an_output_address_off_a_row_boundary_raises(blackhole, offset):
    """``Row``/``Col`` come from one counter, so a partial-row start is not a rectangle.

    UNPACR_Regular.md marks a misaligned OutAddr UnsupportedFunctionality and
    the reference simulator refuses it too; the walk here splits the counter
    into whole rows of 16, which would silently drop the column offset (and the
    row carry it produces). Rejected at decode, before any datum moves.
    """
    with _backend(blackhole) as (backend, memory):
        _fill_tile(memory)
        _set_config(
            backend, "THCON_SEC0_REG5_Dest_cntx0_address", DEST_ADDRESS + offset
        )
        with pytest.raises(NotImplementedError) as excinfo:
            backend.unpacker_units[0].read_unpack_state(0, UNPACR_ARGS)
        assert "16-datum row boundary" in str(excinfo.value)
        assert _read_srca(backend, rows=1) == [[0] * 16]


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
    print("unpack_untilize_test OK: face rows scatter down SrcA off the output address")


if __name__ == "__main__":
    main()
