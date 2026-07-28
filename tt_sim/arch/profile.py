"""The architecture profile: per-arch hardware constants as plain data.

This is intentionally data-only. Per-architecture *behaviour* that cannot be
reduced to a constant (NoC address encoding, coordinate translation, the SFPU
instruction set) is modelled as strategies threaded in later phases of the
Blackhole port; see ``docs/plans/blackhole-support.md``. Fields are added here
as they are actually threaded through the device — an unused field is a promise
the code does not yet keep, so this grows phase by phase rather than all at once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchProfile:
    """Hardware constants that differ between Tenstorrent architectures."""

    #: Human-readable architecture name, e.g. ``"wormhole"``.
    name: str

    #: NoC torus dimensions (columns x rows) in SoC-physical NoC 0 space. NoC 1's
    #: origin is the opposite corner, so a tile's NoC 1 coord is the mirror
    #: ``(noc_grid_x - 1 - x, noc_grid_y - 1 - y)``.
    noc_grid_x: int
    noc_grid_y: int

    #: Largest single NoC transfer (L1<->L1 burst), in bytes. A request larger
    #: than this is split into this-sized flits by the NIU. 8 KiB on Wormhole,
    #: 16 KiB on Blackhole.
    noc_max_burst_size: int

    #: Tensix L1 SRAM size, in bytes.
    tensix_l1_size: int

    #: Unified coord assigned to each physical DRAM channel, one entry per
    #: channel, in the same order as the SoC descriptor's DRAM list.
    dram_channel_unified_coords: tuple[tuple[int, int], ...]

    #: Parallel to :attr:`dram_channel_unified_coords`. Each entry is the
    #: channel's worker-visible SoC-physical NoC 0 coords: element 0 is the
    #: primary (the NUI's id), the rest are aliases the NoC directory also
    #: registers so kernels addressing either sub-endpoint hit the same tile.
    dram_channel_physical_noc0_coords: tuple[tuple[tuple[int, int], ...], ...]

    #: Default Tensix worker tiles to instantiate, as unified coords.
    tensix_unified_coords: tuple[tuple[int, int], ...]
