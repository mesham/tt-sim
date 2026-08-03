"""The Blackhole A0 device and its tiles.

A minimal bring-up: one DRAM tile + one Tensix tile, mirroring Wormhole's current
extent. The tile classes are the Wormhole ones — their memory maps are shared
between the two architectures per the ISA docs — subclassed only to widen the
tile-coordinate band, since Blackhole addresses tiles by their physical NoC 0
coordinate (a 17x12 grid) rather than Wormhole's 16-25 "unified" band. All the
per-arch numbers (grid, L1 size, NoC burst) flow in through ``BLACKHOLE_PROFILE``.

Everything that is not a per-arch hardware fact — tracing, the progress
watchdog, diagnostics fan-out, clock/reset and NoC-directory registration —
belongs to ``TT_Device`` and arrives here by inheritance. Do not wire a
device-level facility in this constructor: doing that is what left Blackhole
without ``TT_SIM_TRACE_*`` and without ``[DEADLOCK]`` reports. See
``tt_sim/device/parity_test.py``.

Known not-yet-modelled for Blackhole (see ``docs/plans/blackhole-support.md``):
the doubled baby-core local memories and 2-core eth tiles (Phase 6, only matter
once kernels run), the NoC 64-bit address encoding and combined coordinate
translation (Phase 4 strategy work), and the DST swizzle. This is enough to
construct the device and move data DRAM<->Tensix over the NoC.
"""

from tt_sim.arch import BLACKHOLE_PROFILE
from tt_sim.device.tiles import DRAMTile, TensixTile
from tt_sim.device.tt_device import TT_Device


class _BlackholeTileCoords:
    """Blackhole tile-coordinate band: the full 17x12 physical NoC grid.

    Overrides ``TTDeviceTile``'s Wormhole 16-25 "unified" band. Placed first in
    the MRO so these win over the base class constants.
    """

    UNIFIED_COORD_X_MIN = 0
    UNIFIED_COORD_X_MAX = 16
    UNIFIED_COORD_Y_MIN = 0
    UNIFIED_COORD_Y_MAX = 11


class BlackholeTensixTile(_BlackholeTileCoords, TensixTile):
    pass


class BlackholeDRAMTile(_BlackholeTileCoords, DRAMTile):
    pass


class Blackhole(TT_Device):
    # Blackhole addresses tiles by their physical NoC 0 coord, so the tile
    # coord and the NUI's physical coord are the same — ``TT_Device``'s
    # identity ``_tensix_physical_coord`` is already right.
    tensix_tile_class = BlackholeTensixTile
    dram_tile_class = BlackholeDRAMTile

    def __init__(self, diagnostics=None, tensix_coords=None):
        # Shared pre-tile setup (profile, diagnostics, tracing bus) — see
        # ``TT_Device``. Must come first: tile construction reads the profile.
        tensix_coords = self._begin_construction(
            BLACKHOLE_PROFILE, diagnostics, tensix_coords
        )
        dram_tiles = self._build_dram_tiles()
        tensix_tiles = [self._build_tensix_tile(coord) for coord in tensix_coords]

        # No eth tiles: Blackhole's are 2-core and not yet modelled (Phase 6 of
        # ``docs/plans/blackhole-support.md``). The only genuine device-level
        # asymmetry with Wormhole.
        super().__init__(None, dram_tiles, tensix_tiles, profile=self.profile)
