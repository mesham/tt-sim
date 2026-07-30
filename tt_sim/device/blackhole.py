"""The Blackhole A0 device and its tiles.

A minimal bring-up: one DRAM tile + one Tensix tile, mirroring Wormhole's current
extent. The tile classes are the Wormhole ones — their memory maps are shared
between the two architectures per the ISA docs — subclassed only to widen the
tile-coordinate band, since Blackhole addresses tiles by their physical NoC 0
coordinate (a 17x12 grid) rather than Wormhole's 16-25 "unified" band. All the
per-arch numbers (grid, L1 size, NoC burst) flow in through ``BLACKHOLE_PROFILE``.

Known not-yet-modelled for Blackhole (see ``docs/plans/blackhole-support.md``):
the doubled baby-core local memories and 2-core eth tiles (Phase 6, only matter
once kernels run), the NoC 64-bit address encoding and combined coordinate
translation (Phase 4 strategy work), and the DST swizzle. This is enough to
construct the device and move data DRAM<->Tensix over the NoC.
"""

from tt_sim.arch import BLACKHOLE_PROFILE
from tt_sim.device.tiles import DRAMTile, TensixTile
from tt_sim.device.tt_device import DeviceTileDiagnostics, TT_Device


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
    def __init__(self, diagnostics=None, tensix_coords=None):
        # Per-core RISC-V / NoC / coprocessor diagnostic flags, wired into each
        # tile so TT_SIM_DIAG_* env vars work on Blackhole as on Wormhole.
        self.diagnostics = diagnostics or DeviceTileDiagnostics()
        self.profile = BLACKHOLE_PROFILE
        if tensix_coords is None:
            tensix_coords = self.profile.tensix_unified_coords

        dram_tiles = []
        noc1_coords = self.profile.dram_channel_physical_noc1_coords or (
            (None,) * len(self.profile.dram_channel_unified_coords)
        )
        for unified, physicals, noc1_coord in zip(
            self.profile.dram_channel_unified_coords,
            self.profile.dram_channel_physical_noc0_coords,
            noc1_coords,
        ):
            primary = physicals[0]
            aliases = physicals[1:]
            dram_tiles.append(
                BlackholeDRAMTile(
                    unified[0],
                    unified[1],
                    primary[0],
                    primary[1],
                    aliases,
                    profile=self.profile,
                    noc1_endpoint_coord=noc1_coord,
                )
            )

        tensix_tiles = [self._build_tensix_tile(coord) for coord in tensix_coords]

        super().__init__(None, dram_tiles, tensix_tiles, profile=self.profile)

    def _build_tensix_tile(self, coord):
        # Blackhole addresses tiles by physical NoC 0 coord, so the tile coord
        # and the NUI's physical coord are the same.
        x, y = coord
        d = self.diagnostics
        return BlackholeTensixTile(
            x,
            y,
            x,
            y,
            d.reportBRISC(),
            d.reportNCRISC(),
            d.reportTRISC0(),
            d.reportTRISC1(),
            d.reportTRISC2(),
            d.reportNoC0(),
            d.reportNoC1(),
            d.getTensixCoprocessorDiagnostics(),
            profile=self.profile,
        )

    def add_tensix_tile(self, coord):
        """Construct and register a Tensix tile at ``coord`` after __init__.

        The wire-bridge ``Device`` wrapper calls this to lazily materialise a
        worker on first access. Mirrors ``Wormhole.add_tensix_tile`` without the
        deadlock detector (not wired for Blackhole yet).
        """
        tile = self._build_tensix_tile(coord)
        self.register_tensix_tile(tile)
        return tile
