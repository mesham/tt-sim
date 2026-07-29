"""Translation tables for the Blackhole wire bridge.

Unlike Wormhole (which packs tiles into a 16-25 "unified" band), the tt-sim
Blackhole device keys its tiles by their **physical NoC 0 coordinate** — the
same coordinates the SoC descriptor enumerates and the wire carries — so the
maps here are essentially identity. See ``docs/plans/blackhole-support.md``.

Scope note: this is the minimal single-tile bring-up. ``TENSIX_COORD_MAP`` covers
every functional worker (so any is materialisable on demand), but only the DRAM
channel(s) the ``BLACKHOLE_PROFILE`` actually builds are mapped; everything else
(other DRAM, eth, pcie, arc, router-only) falls through to ``NullCore``, exactly
as unmapped coords do on Wormhole.
"""

import pathlib

import yaml

from tt_sim.arch import BLACKHOLE_PROFILE

_SOC_DESCRIPTOR_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "soc_descriptor.yaml"
)


def _parse_coord(spec):
    if isinstance(spec, (list, tuple)):
        return (int(spec[0]), int(spec[1]))
    x, y = spec.split("-")
    return (int(x), int(y))


def _load_soc_descriptor(path=_SOC_DESCRIPTOR_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def _build_tensix_map(soc):
    """Every functional worker maps to itself (physical == tile coord)."""
    return {_parse_coord(w): _parse_coord(w) for w in soc["functional_workers"]}


def _build_dram_map():
    """Map each DRAM tile the profile builds to itself.

    Blackhole names a DRAM tile by its worker-visible physical NoC 0 endpoint,
    which is also its tile coord, so this is identity over the profile's DRAM
    channels.
    """
    return {coord: coord for coord in BLACKHOLE_PROFILE.dram_channel_unified_coords}


_SOC = _load_soc_descriptor()
TENSIX_COORD_MAP = _build_tensix_map(_SOC)
DRAM_COORD_MAP = _build_dram_map()
# No eth tiles are modelled for Blackhole yet (2-core eth is Phase 6 work), so
# eth coords fall through to NullCore.
ETH_COORD_MAP = {}


def default_tensix_coords(n):
    """Return the first ``n`` functional-worker coords for ``TT_SIM_TENSIX_CORES=N``.

    Column-major (x outer, y inner), matching how tt-metal drives program cores.
    """
    if n < 1:
        raise ValueError(f"core count must be >= 1, got {n}")
    ordered = sorted(TENSIX_COORD_MAP, key=lambda c: (c[0], c[1]))
    if n > len(ordered):
        raise ValueError(
            f"core count {n} exceeds the {len(ordered)} functional workers "
            f"in soc_descriptor.yaml"
        )
    return ordered[:n]
