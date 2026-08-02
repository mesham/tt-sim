"""Tests for Dst register row addressing (``DstRegister`` Adj16 / Adj32) and
per-row validity (the "zero flags").

Runs standalone (``python3 -m tt_sim.pe.tensix.registers_test``) or under pytest.
Pins three things: the shared, config-off addressing that both Wormhole and
Blackhole use today; the exact Blackhole ``Adj16`` / ``Adj32`` transforms
from BlackholeA0/.../Dst.md that a future compute path will enable via
``DEST_ACCESS_CFG``; and the zero-flag semantics ported from ttsim's
``dst_row_valid`` -- cleared by ZEROACC, re-asserted by any write, and read back
as all-ones (minus infinity) by GMPOOL alone.
"""

import numpy as np

from tt_sim.pe.tensix.registers import DstRegister, SrcRegister


def _shared_fold(r):
    """The 32-bit row fold both architectures apply (the config-off Adj32)."""
    return ((r & 0x1F8) << 1) | (r & 0x207)


def test_adj16_identity_when_remap_off():
    dst = DstRegister()
    assert all(dst.adj16(r) == r for r in range(1024))


def test_adj32_is_shared_fold_when_gates_off():
    dst = DstRegister()
    assert all(dst.adj32(r) == _shared_fold(r) for r in range(1024))


def test_adj16_remap_matches_isa_pseudocode():
    dst = DstRegister()
    dst.dest_remap_addrs = True
    for r in range(1024):
        want = (r & 0x3C7) ^ ((r & 0x030) >> 1) ^ ((r & 0x008) << 2)
        assert dst.adj16(r) == want, r


def test_adj32_swizzle_matches_isa_pseudocode():
    dst = DstRegister()
    dst.dest_swizzle_32b = True
    for r in range(1024):
        s = (r & 0x3F3) ^ ((r & 0x018) >> 1) ^ ((r & 0x004) << 1)
        assert dst.adj32(r) == _shared_fold(s), r


def test_dst16b_round_trips():
    dst = DstRegister()
    dst.setDst16b(5, 3, 0xBEEF)
    assert dst.getDst16b(5, 3) == 0xBEEF


def test_dst32b_round_trips_across_two_16b_rows():
    dst = DstRegister()
    dst.setDst32b(7, 2, 0xDEADBEEF)
    assert dst.getDst32b(7, 2) == 0xDEADBEEF
    # The 32-bit value is split high/low across two backing 16-bit rows.
    br = _shared_fold(7)
    assert int(dst.dstBits[br][2]) == 0xDEAD
    assert int(dst.dstBits[br + 8][2]) == 0xBEEF


def test_remap_changes_backing_row():
    """With remap on, a Dst16b write lands in the ISA-remapped backing row."""
    plain = DstRegister()
    remapped = DstRegister()
    remapped.dest_remap_addrs = True
    plain.setDst16b(0x30, 0, 0x1234)  # 0x30 has bits Adj16 permutes
    remapped.setDst16b(0x30, 0, 0x1234)
    assert plain.adj16(0x30) != remapped.adj16(0x30)
    assert int(remapped.dstBits[remapped.adj16(0x30)][0]) == 0x1234


def test_rows_start_valid():
    # ttsim tensix_init: "hardware resets zero flags to 0, not 1", i.e. every
    # row starts *valid*, so an untouched Dst never reads as minus infinity.
    dst = DstRegister()
    assert dst.dstRowValid.all()
    assert dst.getDst16b(3, 0, isGmpool=True) == 0
    assert dst.getDst32b(3, 0, isGmpool=True) == 0


def test_undefined_row_reads_as_zero_except_for_gmpool():
    dst = DstRegister()
    dst.setDst16b(5, 3, 0xBEEF)
    dst.setUndefinedRow(5)
    assert not dst.dstRowValid[5]
    assert dst.getDst16b(5, 3) == 0
    # GMPOOL alone substitutes all-ones -- the most negative datum in its
    # comparison order (ttsim read_dst16b<is_gmpool=true>).
    assert dst.getDst16b(5, 3, isGmpool=True) == 0xFFFF


def test_undefined_32b_row_clears_the_pair_getDst32b_reads():
    dst = DstRegister()
    dst.setDst32b(7, 2, 0xDEADBEEF)
    dst.setUndefinedRow(7, True)
    br = _shared_fold(7)
    assert not dst.dstRowValid[br]
    assert int(dst.dstBits[br][2]) == 0
    assert int(dst.dstBits[br + 8][2]) == 0
    assert dst.getDst32b(7, 2) == 0
    assert dst.getDst32b(7, 2, isGmpool=True) == 0xFFFFFFFF


def test_write_reasserts_the_flag():
    dst = DstRegister()
    dst.setUndefinedRow(5)
    dst.setDst16b(5, 0, 0x1234)
    assert dst.dstRowValid[5]
    assert dst.getDst16b(5, 7, isGmpool=True) == 0  # whole row is valid again


def test_valid_on_last_column_holds_the_flag_until_column_15():
    # GMPOOL's accumulating write: the row must keep reading as minus infinity
    # for the rest of the pass, or columns 1-15 would max against column 0's
    # result instead (ttsim's set_valid_on_last_column_only).
    dst = DstRegister()
    dst.setUndefinedRow(5)
    for col in range(15):
        dst.setDst16b(5, col, 0x1234, validOnLastColumn=True)
        assert not dst.dstRowValid[5]
        assert dst.getDst16b(5, col + 1, isGmpool=True) == 0xFFFF
    dst.setDst16b(5, 15, 0x1234, validOnLastColumn=True)
    assert dst.dstRowValid[5]


# -- Whole-row accessors ---------------------------------------------------
#
# The matrix unit reads and writes a whole Dst rectangle per MVMUL, and a whole
# Src block, through the block accessors rather than 16 scalar calls per row.
# These pin them as being exactly the scalar accessors applied row by row --
# including the row adjustment, the zero flags, and the 32-bit hi/lo split.


def _seeded(shape, seed):
    return np.random.default_rng(seed).integers(0, 1 << 16, shape, dtype=np.int64)


def _clear_flag(dst, row, isDst32=False):
    """Clear a row's zero flag *without* zeroing its data, so a read that
    ignored the flag would return something visibly different from zero."""
    dst.dstRowValid[dst.adj32(row) if isDst32 else dst.adj16(row)] = False


def test_dst16b_row_block_reads_match_the_scalar_accessor():
    for remap in (False, True):
        dst = DstRegister()
        dst.dest_remap_addrs = remap
        dst.dstBits[:] = _seeded(dst.dstBits.shape, 1)
        for row in (0, 5, 0x30):
            _clear_flag(dst, row)
        rows = list(range(0x38))
        block = dst.getDst16bRows(np.array(rows))
        assert block.tolist() == [
            [dst.getDst16b(r, c) for c in range(16)] for r in rows
        ]


def test_dst16b_row_block_writes_match_the_scalar_accessor():
    for remap in (False, True):
        block_written, scalar_written = DstRegister(), DstRegister()
        for dst in (block_written, scalar_written):
            dst.dest_remap_addrs = remap
            for row in (2, 9):
                _clear_flag(dst, row)
        rows = list(range(2, 10))
        values = _seeded((len(rows), 16), 2)

        block_written.setDst16bRows(np.array(rows), values)
        for i, r in enumerate(rows):
            for c in range(16):
                scalar_written.setDst16b(r, c, int(values[i][c]))

        assert np.array_equal(block_written.dstBits, scalar_written.dstBits)
        assert np.array_equal(block_written.dstRowValid, scalar_written.dstRowValid)


def test_dst32b_row_block_reads_match_the_scalar_accessor():
    for swizzle in (False, True):
        dst = DstRegister()
        dst.dest_swizzle_32b = swizzle
        dst.dstBits[:] = _seeded(dst.dstBits.shape, 3)
        for row in (1, 4):
            _clear_flag(dst, row, isDst32=True)
        rows = list(range(0x20))
        block = dst.getDst32bRows(np.array(rows))
        assert block.tolist() == [
            [dst.getDst32b(r, c) for c in range(16)] for r in rows
        ]


def test_dst32b_row_block_writes_match_the_scalar_accessor():
    for swizzle in (False, True):
        block_written, scalar_written = DstRegister(), DstRegister()
        for dst in (block_written, scalar_written):
            dst.dest_swizzle_32b = swizzle
            _clear_flag(dst, 3, isDst32=True)
        rows = list(range(3, 11))
        values = _seeded((len(rows), 16), 4) << 16 | _seeded((len(rows), 16), 5)

        block_written.setDst32bRows(np.array(rows), values)
        for i, r in enumerate(rows):
            for c in range(16):
                scalar_written.setDst32b(r, c, int(values[i][c]))

        assert np.array_equal(block_written.dstBits, scalar_written.dstBits)
        assert np.array_equal(block_written.dstRowValid, scalar_written.dstRowValid)


def test_src_read_rows_matches_the_scalar_accessor():
    src = SrcRegister()
    src.data[:] = np.random.default_rng(6).integers(0, 1 << 19, src.data.shape)
    block = src.readRows(8, 16)
    assert block.dtype == np.int64
    assert block.tolist() == [[src[r, c] for c in range(16)] for r in range(8, 24)]


def test_src_read_rows_past_the_end_of_the_bank_raises():
    src = SrcRegister()
    try:
        src.readRows(56, 16)
    except IndexError:
        return
    raise AssertionError("reading past row 63 should raise, as scalar indexing does")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("registers_test OK: Dst Adj16/Adj32 addressing and zero flags verified")


if __name__ == "__main__":
    main()
