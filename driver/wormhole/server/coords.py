"""Translation tables derived from ``driver/wormhole/soc_descriptor.yaml``.

The wire bridge receives messages addressed to **physical NoC coordinates**
(the workers / dram / eth grid positions enumerated in the SoC descriptor).
tt-sim packs the same hardware into a **unified coordinate** band (16-25)
defined by ``Wormhole`` in ``tt_sim/device/tt_device.py``. This module pairs
the two by reading the descriptor — no hand-rolled tables.

How the pairing works:

- **DRAM.** Each ``dram_views[i]`` entry pins a sub-endpoint of a controller
  to the wire: ``dram[channel][worker_endpoint[0]]`` is the physical NoC
  coord; ``Wormhole.DRAM_CHANNEL_UNIFIED_COORDS[channel]`` is the
  ``DRAMTile`` that backs it. The two sub-endpoints serving the same
  controller alias to the same ``DRAMTile`` — the 1 GB ``address_offset``
  baked into the buffer address by the tt-metal allocator selects
  ``ddr_bank_0`` vs ``ddr_bank_1`` inside that tile.

- **Tensix.** ``Wormhole.TENSIX_UNIFIED_COORDS[i]`` is paired with
  ``functional_workers[i]``. With one Tensix tile today the map is just
  ``(1, 1) → (18, 18)``; extending ``TENSIX_UNIFIED_COORDS`` plus the
  matching ``TensixTile`` construction in ``Wormhole.__init__`` is all that
  multi-Tensix expansion (ROADMAP §A) needs from this file.

Anything else (workers, eth, pcie, arc, router-only) is intentionally left
out — the fabric falls back to ``NullCore`` (zero-fills reads, swallows
writes), which has proven sufficient for tt-metal device-init traffic.
"""

import pathlib

import yaml

from tt_sim.device.tt_device import Wormhole

_SOC_DESCRIPTOR_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "soc_descriptor.yaml"
)


def _parse_coord(spec):
    """Parse a SoC-descriptor coord, accepting either ``"x-y"`` or ``[x, y]``."""
    if isinstance(spec, (list, tuple)):
        return (int(spec[0]), int(spec[1]))
    x, y = spec.split("-")
    return (int(x), int(y))


def _load_soc_descriptor(path=_SOC_DESCRIPTOR_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_dram_map(soc):
    """Pair every ``dram_views`` sub-endpoint with its ``DRAMTile`` unified coord."""
    dram = [[_parse_coord(c) for c in ch] for ch in soc["dram"]]
    coord_map = {}
    for view in soc["dram_views"]:
        channel = view["channel"]
        n0, n1 = view["worker_endpoint"]
        # Both NoCs use the same sub-index inside a worker's dram view; the
        # descriptor would have to grow asymmetric routing for this to differ,
        # which Wormhole B0 does not today.
        assert n0 == n1, f"asymmetric worker_endpoint {view['worker_endpoint']}"
        physical = dram[channel][n0]
        coord_map[physical] = Wormhole.DRAM_CHANNEL_UNIFIED_COORDS[channel]
    return coord_map


def _build_tensix_map(soc):
    """Pair the first ``len(Wormhole.TENSIX_UNIFIED_COORDS)`` workers with their tiles."""
    workers = [_parse_coord(w) for w in soc["functional_workers"]]
    return {workers[i]: u for i, u in enumerate(Wormhole.TENSIX_UNIFIED_COORDS)}


_SOC = _load_soc_descriptor()
TENSIX_COORD_MAP = _build_tensix_map(_SOC)
DRAM_COORD_MAP = _build_dram_map(_SOC)

# Wormhole NoC grid dimensions, sourced from the descriptor.
NOC_GRID_X = int(_SOC["grid"]["x_size"])
NOC_GRID_Y = int(_SOC["grid"]["y_size"])


def noc1_mirror(noc0_coord):
    """NoC 1 origin is the bottom-right tile, so its coords are mirrored.

    See ``tt-isa-documentation/WormholeB0/NoC/Coordinates.md``.
    """
    x, y = noc0_coord
    return (NOC_GRID_X - 1 - x, NOC_GRID_Y - 1 - y)
