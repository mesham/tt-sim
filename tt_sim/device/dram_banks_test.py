"""DRAM banks: distinct storage, distinct coordinates, the declared count.

This is the substrate under every interleaved DRAM buffer. tt-metal splits such
a buffer's pages round-robin over the device's DRAM **banks** — 12 on Wormhole
(6 channels x two 1 GiB ``dram_views``), 8 on Blackhole (8 channels x one view)
— and the host's scatter and the kernel's gather compute the landing site
independently, from the page size, the bank count, and the per-bank coordinate
and base offset the host wrote into L1 at init. Nothing cross-checks the two.

So the failure this pins is not a fault, it is an *alias*. A simulator that
modelled DRAM as one flat store, or routed every bank endpoint to one tile,
would run any such program to completion with every address a legal address and
the numbers simply wrong — and would run the single-page examples (which all
hardcode bank 0) perfectly green. What has to hold for the corruption to be
visible at all is that the banks are really apart:

* every bank's range is backed by storage disjoint from every other bank's, so
  a write to one is invisible in the rest — on Wormhole that means *both*
  senses of apart, since two of its banks share one channel (and one
  ``SparseDRAM``) at ``address_offset`` 0 and 1 GiB, while banks in different
  channels are different tiles;
* every bank's SoC-descriptor NoC 0 coordinate is a distinct cell, is
  registered in the NoC directory, and resolves to *that bank's* channel tile —
  including the alias endpoints, since a Wormhole channel is one controller
  behind two worker-visible cells at opposite ends of its column; and
* the number of banks is the number the shipped SoC descriptor declares, which
  is where a consumer reads it, and every view's range lies inside the channel
  tt-sim actually models.

The banks here are derived from ``driver/<arch>/soc_descriptor.yaml`` exactly
as tt-metal derives them — ``dram_views[b].channel`` picks the row of ``dram``,
``worker_endpoint[0]`` picks the NoC 0 subchannel within it, ``address_offset``
is the base inside the channel — rather than from the profile the model is
built from, because reading both sides from ``tt_sim/arch/`` would assert the
model against itself. Bank *order* is deliberately not asserted: which page
lands in which bank is tt-metal's arithmetic on both sides and never tt-sim's.

Cheap by construction: two device builds, no kernel, no trace. The end-to-end
complement — a real interleaved buffer scattered by a real host and gathered by
a real kernel — is ``examples/banks`` and
``driver/blackhole/server/banks_replay_test.py``, which need a captured trace
and a tt-metal checkout and therefore cannot be what an external consumer's
gate is pinned to.

Published as ``dram-interleaved-bank-distinctness``; see ``tt_sim/behaviour.py``.

Runs standalone (``python3 -m tt_sim.device.dram_banks_test``) or under pytest.
"""

import collections
import pathlib

import pytest
import yaml

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.arch.wormhole import WORMHOLE_PROFILE
from tt_sim.behaviour import require
from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.tt_noc import resolved_nui

_ROOT = pathlib.Path(__file__).resolve().parents[2]

_ARCHES = {
    "wormhole": (Wormhole, WORMHOLE_PROFILE),
    "blackhole": (Blackhole, BLACKHOLE_PROFILE),
}

#: Banks tt-metal will interleave over, per architecture. Spelled out as well
#: as derived so that a descriptor edit which changes the count has to come
#: past this line: the count is a JIT define (``NUM_DRAM_BANKS``) on the kernel
#: side, so changing it changes where every page of every interleaved buffer
#: lands.
EXPECTED_BANK_COUNT = {"wormhole": 12, "blackhole": 8}

#: Where inside a bank the distinctness probes go: one a little way in, one on
#: the last word of the view. The second is the one that notices a channel
#: modelled shorter than the views banked into it.
PROBE_OFFSET = 0x2000


def _soc_descriptor(arch):
    path = _ROOT / "driver" / arch / "soc_descriptor.yaml"
    if not path.exists():  # pragma: no cover - a checkout without drivers
        pytest.skip(f"no {path}")
    return yaml.safe_load(path.read_text())


def _banks(soc):
    """``(channel, NoC 0 coord, address_offset)`` per bank, descriptor order.

    The same three lookups tt-metal does: a view names its channel, indexes
    that channel's DRAM row by ``worker_endpoint[0]`` for the coordinate a
    worker addresses it at on NoC 0, and carries the base offset inside the
    channel that ``bank_to_dram_offset`` ends up holding.
    """
    rows = [[tuple(int(v) for v in c.split("-")) for c in row] for row in soc["dram"]]
    return [
        (
            view["channel"],
            rows[view["channel"]][view["worker_endpoint"][0]],
            view["address_offset"],
        )
        for view in soc["dram_views"]
    ]


def _marker(index):
    return (0xBA5E0000 | index).to_bytes(4, "little")


@pytest.mark.parametrize("arch", sorted(_ARCHES), ids=sorted(_ARCHES))
def test_the_bank_count_is_the_one_the_descriptor_declares(arch):
    """...and every bank's range is inside a channel tt-sim models whole."""
    soc = _soc_descriptor(arch)
    profile = _ARCHES[arch][1]
    banks = _banks(soc)

    assert len(banks) == EXPECTED_BANK_COUNT[arch]
    assert len(soc["dram"]) == len(profile.dram_channel_unified_coords), (
        "tt-sim builds one DRAM tile per descriptor channel; a mismatch means "
        "some channel is not modelled at all and reads back zeros"
    )
    assert soc["dram_bank_size"] == profile.dram_channel_size

    view_size = soc["dram_view_size"]
    per_channel = collections.Counter(channel for channel, _, _ in banks)
    assert sorted(per_channel) == list(range(len(soc["dram"]))), (
        "every channel must carry at least one view, or its storage is "
        "unreachable through the bank tables"
    )
    for channel, count in per_channel.items():
        assert count * view_size == profile.dram_channel_size, (
            f"channel {channel} is banked into {count} x {view_size} B, which "
            f"is not the {profile.dram_channel_size} B tt-sim models for it"
        )
    for channel, _, offset in banks:
        assert 0 <= offset
        assert offset + view_size <= profile.dram_channel_size, (
            f"a bank of channel {channel} runs from {offset:#x} past the end "
            f"of the modelled channel"
        )


@pytest.mark.parametrize("arch", sorted(_ARCHES), ids=sorted(_ARCHES))
def test_every_bank_is_its_own_storage_at_its_own_coordinate(arch):
    soc = _soc_descriptor(arch)
    device_class, profile = _ARCHES[arch]
    banks = _banks(soc)
    view_size = soc["dram_view_size"]

    coords = [coord for _, coord, _ in banks]
    assert len(set(coords)) == len(banks), (
        f"{arch}: two banks share a NoC coordinate, so a kernel cannot address "
        f"them apart: {sorted(coords)}"
    )

    device = device_class()
    try:
        tiles = {}
        for channel, coord, _ in banks:
            assert coord in device.noc_0_directory, (
                f"{arch}: bank coordinate {coord} is not registered on NoC 0 — "
                f"a kernel addressing that bank reaches nothing"
            )
            tile = device._tile_of_nui[id(resolved_nui(device.noc_0_directory[coord]))]
            channel_tile = device.tile_directory[
                profile.dram_channel_unified_coords[channel]
            ]
            assert tile is channel_tile, (
                f"{arch}: bank coordinate {coord} resolves to {tile} rather "
                f"than the tile of its own channel {channel}"
            )
            tiles[coord] = tile
        assert len({id(tile) for tile in tiles.values()}) == len(soc["dram"]), (
            f"{arch}: the bank coordinates resolve to fewer tiles than there "
            f"are channels, so some channel's storage is aliased onto another"
        )

        # Distinct storage: one marker per bank at a common within-view offset
        # and one on the view's last word, all written before any is read back.
        # Two banks sharing a store — the flat-DRAM modelling this exists to
        # exclude — would leave the later marker in both places.
        for index, (_, coord, offset) in enumerate(banks):
            tiles[coord].write(offset + PROBE_OFFSET, _marker(index))
            tiles[coord].write(offset + view_size - 4, _marker(0xFF00 | index))
        for index, (channel, coord, offset) in enumerate(banks):
            got = tiles[coord].read(offset + PROBE_OFFSET, 4)
            assert got == _marker(index), (
                f"{arch}: bank {index} (channel {channel} at {coord}, offset "
                f"{offset:#x}) reads back {got.hex()} — it shares storage with "
                f"another bank, so an interleaved buffer would alias rather "
                f"than corrupt"
            )
            top = tiles[coord].read(offset + view_size - 4, 4)
            assert top == _marker(0xFF00 | index), (
                f"{arch}: the last word of bank {index} reads back {top.hex()}"
            )
    finally:
        device.shutdown()


def test_the_marker_is_published():
    require("dram-interleaved-bank-distinctness")


def main():
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        params = [m for m in marks if m.name == "parametrize"]
        if params:
            for case in params[0].args[1]:
                fn(case)
        else:
            fn()
    print(
        "dram_banks_test OK: 12 Wormhole / 8 Blackhole DRAM banks, each its "
        "own storage at its own NoC coordinate, counts matching the shipped "
        "SoC descriptors"
    )


if __name__ == "__main__":
    main()
