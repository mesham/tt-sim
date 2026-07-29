"""Tests for Dst register row addressing (``DstRegister`` Adj16 / Adj32).

Runs standalone (``python3 -m tt_sim.pe.tensix.registers_test``) or under pytest.
Pins two things: the shared, config-off addressing that both Wormhole and
Blackhole use today, and the exact Blackhole ``Adj16`` / ``Adj32`` transforms
from BlackholeA0/.../Dst.md that a future compute path will enable via
``DEST_ACCESS_CFG``.
"""

from tt_sim.pe.tensix.registers import DstRegister


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


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("registers_test OK: Dst Adj16/Adj32 addressing verified")


if __name__ == "__main__":
    main()
