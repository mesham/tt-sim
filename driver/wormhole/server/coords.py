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

- **Tensix.** Every ``functional_workers[i]`` entry maps to a unified coord
  in the 8×10 worker band ``(18-25, 16-25)`` (DRAM occupies ``(16-17, 16-18)``
  so worker x starts at 18). The y-axis is offset by ``+2 mod 10`` so that
  physical ``(1, 1)`` lands on unified ``(18, 18)`` — preserving the
  default single-tile coord the existing examples and ``Wormhole.__init__``
  hardcode. The full 80-entry map is materialised eagerly so callers can
  resolve any worker coord; tt-sim ``TensixTile`` instances are only built
  on demand (see ``server/device.py:Device.ensure_tensix_tile``).

- **Ethernet.** Each ``eth[i]`` entry maps to a unified coord in the eth
  band ``y ∈ {14, 15}`` (below the worker/DRAM band so the historical
  (18, 18) Tensix default is preserved). Backed by an ``EthTile`` with
  L1 SRAM only — no ERisc baby cores yet. Single-chip kernels that
  hardcode an eth coord (``hello_world_datatypes_kernel`` reads ``(1, 0)``)
  now see deterministic memory-backed state instead of NullCore zeros.

Anything else (pcie, arc, router-only) is intentionally left out — the
fabric falls back to ``NullCore`` (zero-fills reads, swallows writes), which
has proven sufficient for tt-metal device-init traffic.
"""

import pathlib

import yaml

from tt_sim.device.wormhole import Wormhole

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
    """Pair every ``functional_workers`` entry with a unified coord.

    The mapping is derived by indexing each axis into the unified worker
    band: ``unified_x = 18 + x_idx`` (8 columns → 18..25) and
    ``unified_y = (y_idx + 2) % 10 + 16``. The y offset of 2 puts
    physical ``(1, 1)`` at unified ``(18, 18)``, which is the coord
    ``Wormhole.TENSIX_UNIFIED_COORDS`` defaults to and every existing
    single-tile example bakes in.
    """
    workers = [_parse_coord(w) for w in soc["functional_workers"]]
    xs = sorted({x for x, _ in workers})
    ys = sorted({y for _, y in workers})
    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}
    coord_map = {
        (x, y): (18 + x_to_idx[x], (y_to_idx[y] + 2) % len(ys) + 16)
        for (x, y) in workers
    }
    assert coord_map.get((1, 1)) == Wormhole.TENSIX_UNIFIED_COORDS[0], (
        f"physical (1, 1) maps to {coord_map.get((1, 1))!r}, "
        f"expected {Wormhole.TENSIX_UNIFIED_COORDS[0]!r}"
    )
    return coord_map


def _build_eth_map(soc):
    """Pair every ``eth`` entry with its ``EthTile`` unified coord."""
    eth = [_parse_coord(c) for c in soc["eth"]]
    return {
        physical: Wormhole.eth_unified_coord_from_physical(physical) for physical in eth
    }


_SOC = _load_soc_descriptor()
TENSIX_COORD_MAP = _build_tensix_map(_SOC)
DRAM_COORD_MAP = _build_dram_map(_SOC)
ETH_COORD_MAP = _build_eth_map(_SOC)

# Primary override-selectable worker block: physical x in 1..4, y in 1..5 (the
# 20-tile 4x5 grid that TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4 selects).
# Filled before the extended-grid workers when materialising by count.
_PRIMARY_MAX_X = 4
_PRIMARY_MAX_Y = 5


def default_tensix_coords(n):
    """Return the first ``n`` physical worker coords for ``TT_SIM_TENSIX_CORES=N``.

    Orders workers **column-major** — y fastest within a column: ``(1,1),
    (1,2), … ,(1,5),(2,1),…`` — because that's the order tt-metal actually
    drives program cores under the default grid override. Empirically both
    ``matmul_multi_core`` and ``noc_tile_transfer`` launch ``(1,1)`` then
    ``(1,2)`` and matmul fills the whole ``x=1`` column before ``x=2``. So a
    bare *core count* materialises the coords real programs use, without the
    user naming them. The primary 4x5 block is filled first, then any
    extended-grid workers. A program that launches on a coord outside this set
    still hits the clear go=GO error naming the exact coord to add.
    """
    if n < 1:
        raise ValueError(f"core count must be >= 1, got {n}")

    def order_key(coord):
        x, y = coord
        in_primary = x <= _PRIMARY_MAX_X and y <= _PRIMARY_MAX_Y
        # Primary block first; column-major (x outer, y inner) within each band.
        return (0 if in_primary else 1, x, y)

    ordered = sorted(TENSIX_COORD_MAP, key=order_key)
    if n > len(ordered):
        raise ValueError(
            f"core count {n} exceeds the {len(ordered)} functional workers "
            f"in soc_descriptor.yaml"
        )
    return ordered[:n]
