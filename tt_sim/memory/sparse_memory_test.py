"""Tests for the sparse DRAM backing and the per-arch DRAM channel size.

Runs standalone (``python3 -m tt_sim.memory.sparse_memory_test``) or under
pytest.

A DRAM channel is 2 GiB (Wormhole) / 0xFF00_0000 (Blackhole), 6 / 8 of them, so
a ``DRAMTile`` cannot hold a flat ``np.zeros`` — hence
``SparseAddressableMemory``, which allocates chunks on write and reads zeros
from the ones nobody has touched. The tests below pin both halves of that: the
chunking is transparent (including across chunk boundaries), and a tile really
does answer for the whole channel, which is what a top-down tt-metal allocation
(``DeviceLocalBufferConfig{.bottom_up = false}``, landing at 0x3FFF_FC00 on
Wormhole / 0xFEFF_FC00 on Blackhole) needs.
"""

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.arch.wormhole import WORMHOLE_PROFILE
from tt_sim.device.blackhole import BlackholeDRAMTile
from tt_sim.device.tiles import DRAMTile
from tt_sim.memory.memory import SparseAddressableMemory

# Addresses tt-metal's allocator picked for the `optests/dramtop` buffer on
# each arch, straight out of the reference simulator's run (it prints the
# buffer address). Both used to fall outside every registered range.
WORMHOLE_TOP_DOWN_ADDR = 0x3FFF_FC00
BLACKHOLE_TOP_DOWN_ADDR = 0xFEFF_FC00


def test_channel_sizes_match_the_soc_descriptors():
    """`dram_bank_size` from driver/<arch>/soc_descriptor.yaml."""
    assert WORMHOLE_PROFILE.dram_channel_size == 2147483648
    assert BLACKHOLE_PROFILE.dram_channel_size == 4278190080


def test_unwritten_memory_reads_as_zeros():
    mem = SparseAddressableMemory(1 << 30, chunk_size=4096)
    assert mem.read(0, 8) == bytes(8)
    assert mem.read((1 << 30) - 8, 8) == bytes(8)
    assert mem.chunks == {}


def test_write_then_read_back():
    mem = SparseAddressableMemory(1 << 30, chunk_size=4096)
    mem.write(0x1234, b"\xde\xad\xbe\xef")
    assert mem.read(0x1234, 4) == b"\xde\xad\xbe\xef"
    # Neighbouring bytes in the same chunk stay zero.
    assert mem.read(0x1230, 4) == bytes(4)
    assert mem.read(0x1238, 4) == bytes(4)


def test_access_spanning_chunks():
    """A transfer wider than a chunk is stitched back together byte-exactly."""
    mem = SparseAddressableMemory(1 << 20, chunk_size=256)
    payload = bytes(range(256)) * 3  # 768 B, i.e. 3 chunks' worth
    mem.write(200, payload)
    assert mem.read(200, len(payload)) == payload
    # Straddling a boundary with only the far side written still reads right.
    assert mem.read(0, 456) == bytes(200) + payload[:256]


def test_only_touched_chunks_are_allocated():
    mem = SparseAddressableMemory(1 << 40, chunk_size=4096)
    mem.write(0, b"\x01")
    mem.write((1 << 40) - 1, b"\x02")
    assert len(mem.chunks) == 2
    assert mem.read((1 << 40) - 1, 1) == b"\x02"


def test_out_of_range_access_raises():
    mem = SparseAddressableMemory(4096, chunk_size=1024)
    with pytest.raises(IndexError):
        mem.read(4094, 4)
    with pytest.raises(IndexError):
        mem.write(4094, b"\x00\x00\x00\x00")


@pytest.mark.parametrize(
    "tile_class,profile,addr",
    [
        (DRAMTile, WORMHOLE_PROFILE, WORMHOLE_TOP_DOWN_ADDR),
        (BlackholeDRAMTile, BLACKHOLE_PROFILE, BLACKHOLE_TOP_DOWN_ADDR),
    ],
    ids=["wormhole", "blackhole"],
)
def test_dram_tile_covers_a_top_down_allocation(tile_class, profile, addr):
    """The address a `.bottom_up = false` buffer lands on must be mapped."""
    coord = profile.dram_channel_unified_coords[0]
    tile = tile_class(coord[0], coord[1], coord[0], coord[1], profile=profile)
    tile.write(addr, b"\xa5\xa5\x00\x00")
    assert tile.read(addr, 4) == b"\xa5\xa5\x00\x00"
    # ... and the very last byte of the channel, too.
    last = profile.dram_channel_size - 1
    tile.write(last, b"\x5a")
    assert tile.read(last, 1) == b"\x5a"


def test_dram_tile_still_rejects_addresses_past_the_channel():
    coord = BLACKHOLE_PROFILE.dram_channel_unified_coords[0]
    tile = BlackholeDRAMTile(
        coord[0], coord[1], coord[0], coord[1], profile=BLACKHOLE_PROFILE
    )
    # The top 16 MiB of a Blackhole tile's NoC address space is the register
    # aperture, not DRAM — an access there stays loud rather than silently
    # landing in data.
    with pytest.raises(IndexError):
        tile.write(0xFFB20000, b"\x00\x00\x00\x00")


def main():
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        params = [m for m in marks if m.name == "parametrize"]
        if params:
            for case in params[0].args[1]:
                fn(*case)
        else:
            fn()
    print(
        "sparse_memory_test OK: per-arch DRAM channel sizes match the SoC "
        "descriptors, sparse chunks read as zeros until written, chunk-spanning "
        "accesses round-trip, and a DRAM tile covers a top-down allocation "
        "on both arches"
    )


if __name__ == "__main__":
    main()
