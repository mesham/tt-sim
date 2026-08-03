"""The Wormhole B0 device.

``Wormhole`` (a :class:`~tt_sim.device.tt_device.TT_Device`) assembles the
Wormhole netlist: six ``DRAMTile``s, the ethernet tiles, and one or more
``TensixTile``s, wired into both NoC directories. The per-arch constants come
from ``WORMHOLE_PROFILE``. The tile classes themselves are arch-agnostic and
live in ``tt_sim/device/tiles.py`` (shared with Blackhole); the base device
classes live in ``tt_sim/device/tt_device.py``. See
``docs/plans/blackhole-support.md``.
"""

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.device.tiles import DRAMTile, EthTile, TensixTile
from tt_sim.device.tt_device import TT_Device


class Wormhole(TT_Device):
    tensix_tile_class = TensixTile
    dram_tile_class = DRAMTile

    # The Wormhole hardware constants now live in ``WORMHOLE_PROFILE``. These
    # class attributes are compatibility aliases for external callers that still
    # reference them by their historical names (e.g. the wire bridge's
    # ``driver/wormhole/server/coords.py``). The profile is the source of truth.
    #
    #: Unified coord per physical DRAM channel; the two NoC sub-endpoints serving
    #: the same controller alias to the same unified coord.
    DRAM_CHANNEL_UNIFIED_COORDS = WORMHOLE_PROFILE.dram_channel_unified_coords
    #: Worker-side SoC-physical NoC 0 coords per channel (primary + aliases).
    DRAM_CHANNEL_PHYSICAL_NOC0_COORDS = (
        WORMHOLE_PROFILE.dram_channel_physical_noc0_coords
    )
    #: Unified coords of each Tensix tile instantiated by default. Overridable
    #: via the ``tensix_coords`` kwarg on ``__init__``.
    TENSIX_UNIFIED_COORDS = WORMHOLE_PROFILE.tensix_unified_coords

    # Ethernet tile layout. SoC physical eth coords (per soc_descriptor.yaml
    # ``eth:`` list) sit at y ∈ {0, 6} with x ∈ {1,2,3,4,6,7,8,9}. The unified
    # band reserves y ∈ {14, 15} below the worker/DRAM band so the historical
    # (18, 18) default for Tensix is untouched. Physical eth y=0 maps to
    # unified y=14, eth y=6 to unified y=15. Within each row physical x maps
    # in ascending order to unified x ∈ {16..23} (8 columns, gap-free).
    _ETH_PHYSICAL_X_VALUES = (1, 2, 3, 4, 6, 7, 8, 9)
    _ETH_PHYSICAL_Y_VALUES = (0, 6)
    ETH_UNIFIED_Y_FOR_PHYSICAL = {0: 14, 6: 15}

    @classmethod
    def eth_unified_coord_from_physical(cls, physical):
        """Map an SoC-physical eth NoC 0 coord (per ``soc_descriptor.yaml``)
        to its unified coord."""
        px, py = physical
        if px not in cls._ETH_PHYSICAL_X_VALUES:
            raise ValueError(f"eth physical x={px} not in soc descriptor")
        if py not in cls._ETH_PHYSICAL_Y_VALUES:
            raise ValueError(f"eth physical y={py} not in soc descriptor")
        ux = 16 + cls._ETH_PHYSICAL_X_VALUES.index(px)
        uy = cls.ETH_UNIFIED_Y_FOR_PHYSICAL[py]
        return (ux, uy)

    @classmethod
    def all_eth_physical_coords(cls):
        """Enumerate every SoC-physical eth coord (matches descriptor order
        is unimportant — only that the set is complete)."""
        return tuple(
            (px, py)
            for py in cls._ETH_PHYSICAL_Y_VALUES
            for px in cls._ETH_PHYSICAL_X_VALUES
        )

    # SoC descriptor `functional_workers` grid: x ∈ {1,2,3,4,6,7,8,9},
    # y ∈ {1,2,3,4,5,7,8,9,10,11}. Used to map unified worker coord
    # (18..25, 16..25) ↔ SoC-physical NoC 0 coord. The y offset of +2 (mod 10)
    # in the unified→physical formula puts physical (1, 1) at unified (18, 18)
    # — the historical default every single-tile example bakes in.
    _TENSIX_PHYSICAL_X_VALUES = (1, 2, 3, 4, 6, 7, 8, 9)
    _TENSIX_PHYSICAL_Y_VALUES = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)

    @classmethod
    def physical_noc0_coord_from_unified_worker(cls, unified):
        """Map a unified worker coord (18..25, 16..25) to SoC-physical NoC 0.

        The inverse of ``server/coords.py:_build_tensix_map`` — exposed in
        tt-sim core so ``NUI``s can be keyed by canonical NoC 0 coord without
        the Python-driver path having to round-trip through the wire bridge.
        """
        ux, uy = unified
        if not (18 <= ux <= 25):
            raise ValueError(
                f"unified worker x={ux} out of band (18..25); coord {unified!r}"
            )
        if not (16 <= uy <= 25):
            raise ValueError(
                f"unified worker y={uy} out of band (16..25); coord {unified!r}"
            )
        x_idx = ux - 18
        y_idx = (uy - 18) % 10
        return (
            cls._TENSIX_PHYSICAL_X_VALUES[x_idx],
            cls._TENSIX_PHYSICAL_Y_VALUES[y_idx],
        )

    def __init__(self, diagnostics=None, tensix_coords=None):
        # Shared pre-tile setup (profile, diagnostics, tracing bus) — see
        # ``TT_Device``. Must come first: tile construction reads the profile.
        tensix_coords = self._begin_construction(
            WORMHOLE_PROFILE, diagnostics, tensix_coords
        )
        dram_tiles = self._build_dram_tiles()
        eth_tiles = []
        for physical in Wormhole.all_eth_physical_coords():
            unified = Wormhole.eth_unified_coord_from_physical(physical)
            eth_tiles.append(
                EthTile(
                    unified[0],
                    unified[1],
                    physical[0],
                    physical[1],
                    profile=self.profile,
                )
            )
        tensix_tiles = [self._build_tensix_tile(coord) for coord in tensix_coords]

        # For now don't provide any memory, in future this will be the memory
        # map of the PCIe endpoing
        super().__init__(
            None, dram_tiles, tensix_tiles, eth_tiles=eth_tiles, profile=self.profile
        )

    def _tensix_physical_coord(self, coord):
        # Wormhole tile coords are the unified band, so the NUI's SoC-physical
        # NoC 0 coord has to be derived.
        return Wormhole.physical_noc0_coord_from_unified_worker(coord)
