from abc import ABC

from tt_sim.device.clock import MultiTileClock, TileClock
from tt_sim.device.device import Device, DeviceTile
from tt_sim.device.reset import Reset
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.util.bits import clear_bit, set_bit
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_uint32,
)


class TT_Device(Device):
    def __init__(
        self, device_memory, dram_tiles, tensix_tiles, eth_tiles=(), *, profile
    ):
        #: The architecture profile supplying every per-arch hardware constant.
        self.profile = profile
        self.dram_tiles = list(dram_tiles)
        self.eth_tiles = list(eth_tiles)
        self.tensix_tiles = []

        self.tile_directory = {}
        # NoC directories are shared by reference across every NUI on the
        # corresponding NoC. Storing them as instance attrs lets
        # ``add_tensix_tile`` extend them post-construction.
        self.noc_0_directory = {}
        self.noc_1_directory = {}

        self.clocks = [MultiTileClock()]
        self.resets = [Reset([])]

        for tile in self.dram_tiles:
            self._register_tile_internals(tile)
        for tile in self.eth_tiles:
            self._register_tile_internals(tile)
        for tile in tensix_tiles:
            self.tensix_tiles.append(tile)
            self._register_tile_internals(tile)

        super().__init__(device_memory, self.clocks, self.resets)

    def _register_tile_internals(self, tile):
        """Insert a tile into the directory, NoCs, clocks, and resets.

        Used by ``__init__`` for the initial fan-out and by
        ``add_tensix_tile`` for tiles materialised later. The NoC directories
        are shared by reference, so inserting once per NoC makes the new tile
        reachable from every existing NUI on the same NoC.

        Both NoC directories key by the tile's canonical (SoC-physical
        NoC 0) coord — kernels supply that coord on either NoC when
        translation is enabled (which is how tt-metal kernels under our
        ``driver/wormhole/<n>/`` tree address tiles). Real tt-metal's
        bank-to-noc table (consulted by ``TensorAccessor`` and friends in
        the canonical ``programming_examples/``) instead writes per-NoC
        *mirror* coords directly into NoC 1's half of the table — so on
        NoC 1 the kernel actually emits the noc1-mirror of the canonical
        coord. For DRAM (and Tensix) we register both forms as keys:
        canonical on both NoCs, plus the noc1-mirror as an additional
        alias on NoC 1 so the bank-table flow resolves the right tile.
        Extra sub-endpoint aliases (e.g. DRAM channels with two
        worker-visible endpoints) get the same canonical + noc1-mirror
        pair of keys.

        **NoC 1 precedence.** On NoC 1 a tile is *physically* reachable at its
        mirror coord ``(GRID-1-x, GRID-1-y)``; the canonical (primary) coord is
        only a convention-1 accommodation for the ``driver/wormhole/<n>/`` tree.
        The mirror formula maps coords across the DRAM / worker / eth bands, so
        the two conventions collide on some cells — e.g. with the truncated 4×5
        worker grid, Tensix ``(4, y)`` mirrors onto the DRAM column ``x = 5``
        (Tensix ``(4, 1)`` ↔ DRAM ``(5, 10)``), and DRAM ch0 ``(0, 11)`` mirrors
        onto eth ``(9, 0)``. When both a mirror and a primary want the same NoC 1
        cell the **mirror must win**, because the bank-to-noc table used by the
        canonical ``programming_examples/`` addresses DRAM over NoC 1 by its
        mirror; letting a Tensix/eth *primary* clobber a DRAM *mirror* routes a
        DRAM read into that tile's (smaller) L1 and overflows it. So:

        - mirror registrations are authoritative (plain assignment);
        - primary and alias registrations use ``setdefault`` — they fill only
          cells no mirror has claimed.

        Eth tiles additionally *skip* mirror registration entirely: an eth
        mirror would land on a DRAM canonical coord and steal that DRAM's own
        primary cell, and no bank-to-noc flow addresses eth by mirror anyway
        (single-chip kernels like ``hello_world_datatypes_kernel`` read eth by
        its canonical ``(1, 0)``).
        """
        coord = tile.get_coord_pair()
        assert coord not in self.tile_directory, f"tile already registered at {coord}"
        self.tile_directory[coord] = tile
        nui0 = tile.get_noc_nui(0)
        nui1 = tile.get_noc_nui(1)
        primary = nui0.get_id_pair()
        self.noc_0_directory[primary] = nui0
        register_mirror = getattr(tile, "register_noc1_mirror", True)
        # A tile whose NoC 1 endpoint is a different SoC-physical coord than its
        # NoC 0 endpoint (Blackhole DRAM: NoC 0 (0,11), NoC 1 (0,1)) mirrors that
        # NoC 1 coord instead of the primary — otherwise kernels addressing the
        # channel over NoC 1 route to an empty grid cell.
        noc1_source = getattr(tile, "noc1_endpoint_coord", None) or primary
        # Primary is non-authoritative on NoC 1: never clobber a mirror.
        self.noc_1_directory.setdefault(primary, nui1)
        if register_mirror:
            self.noc_1_directory[self._noc1_mirror(noc1_source)] = nui1
        for alias in getattr(tile, "noc_aliases", ()):
            self.noc_0_directory[alias] = nui0
            self.noc_1_directory.setdefault(alias, nui1)
            if register_mirror:
                self.noc_1_directory[self._noc1_mirror(alias)] = nui1
        nui0.set_noc_directory(self.noc_0_directory)
        nui1.set_noc_directory(self.noc_1_directory)
        # Tensix tiles host the bulk of per-cycle work (5 baby RV cores +
        # the coprocessor); DRAM and eth tiles are mostly idle NUI traffic
        # so they should not pull the composite into threaded mode on
        # their own. Only count Tensix toward the auto-engage threshold.
        gated, always = tile.get_clock_partition()
        tile_clock = TileClock(gated, always=always, quiescent=tile.clock_quiescent)
        tile._bind_clock(tile_clock)
        self.clocks[0].add_tile_clock(tile_clock, heavy=tile.is_tensix)
        self.resets[0].add_resetables(tile.get_resets())

    def _noc1_mirror(self, canonical):
        """Mirror a canonical (NoC 0 physical) coord to NoC 1's coord space.

        NoC 1's origin is the bottom-right tile of the grid, so the same
        physical tile lives at ``(GRID_X-1-x, GRID_Y-1-y)`` on NoC 1. The grid
        dims come from the architecture profile (10 × 12 for Wormhole).
        """
        return (
            self.profile.noc_grid_x - 1 - canonical[0],
            self.profile.noc_grid_y - 1 - canonical[1],
        )

    def register_tensix_tile(self, tile):
        """Register a TensixTile constructed after device __init__.

        Mirrors what ``__init__`` does for one tile: stitches the tile into
        the directory, both NoC directories, and the central Clock / Reset
        aggregators. Subclasses (e.g. ``Wormhole.add_tensix_tile``) layer a
        coord-based constructor on top.
        """
        self.tensix_tiles.append(tile)
        self._register_tile_internals(tile)

    def reset(self, reset_number=0):
        """Reset every registered component, waking every tile.

        A reset rewrites core state behind the pump's back, so no tile may
        stay dormant on the strength of a pre-reset quiescence decision.
        """
        for tile in self.tile_directory.values():
            tile.clock.awake = True
        super().reset(reset_number)

    def shutdown(self):
        """Join any worker threads spawned by the per-tile clock pump.

        Safe to call on a single-tile / ``TT_SIM_THREADED=0`` device — the
        composite clock no-ops if no workers were ever started. Idempotent.
        """
        for clock in self.clocks:
            shutdown = getattr(clock, "shutdown", None)
            if shutdown is not None:
                shutdown()

    def read(self, coordinate_pair, address, size):
        assert coordinate_pair in self.tile_directory
        tile = self.tile_directory[coordinate_pair]
        # A host-side access is one of the three stimuli that can act on a
        # dormant tile (see ``TileClock``). Reads are woken too: they are rare
        # relative to cycles, and it removes any need to reason about which
        # MMIO reads have side effects.
        tile.clock.awake = True
        return tile.read(address, size)

    def write(self, coordinate_pair, address, value, size=None):
        assert coordinate_pair in self.tile_directory
        tile = self.tile_directory[coordinate_pair]
        tile.clock.awake = True
        tile.write(address, value, size)

    def deassert_soft_reset(self, coordinate_pair=None, core_type=None):
        if coordinate_pair is None:
            for pair, value in self.tile_directory.items():
                if value.is_tensix:
                    self.perform_soft_reset_change(clear_bit, pair, core_type)
        else:
            self.perform_soft_reset_change(clear_bit, coordinate_pair, core_type)

    def assert_soft_reset(self, coordinate_pair=None, core_type=None):
        if coordinate_pair is None:
            for pair, value in self.tile_directory.items():
                if value.is_tensix:
                    self.perform_soft_reset_change(set_bit, pair, core_type)
        else:
            self.perform_soft_reset_change(set_bit, coordinate_pair, core_type)

    def reset_tile(self, coordinate_pair):
        """Reset only the baby cores on a single tile.

        ``wormhole.reset()`` (the generic Device.reset) resets every baby
        core on every tile, which is fine for single-Tensix flows but
        clobbers PCs on tiles already running firmware in multi-Tensix
        setups. Callers bringing up one tile at a time should use this
        scoped variant.
        """
        tile = self.tile_directory[coordinate_pair]
        tile.clock.awake = True
        for core in tile.get_resets():
            core.reset()

    def perform_soft_reset_change(
        self, bit_change_method, coordinate_pair, core_type=None
    ):
        if core_type is not None:
            if core_type == BabyRISCVCoreType.BRISC:
                bit = 11
            elif core_type == BabyRISCVCoreType.NCRISC:
                bit = 18
            elif core_type == BabyRISCVCoreType.TRISC0:
                bit = 12
            elif core_type == BabyRISCVCoreType.TRISC1:
                bit = 13
            elif core_type == BabyRISCVCoreType.TRISC2:
                bit = 14
            elif core_type == BabyRISCVCoreType.ERISC:
                # Per WormholeB0/EthernetTile/SoftReset.md — ERisc reuses
                # bit 11 in this tile's own RISCV_DEBUG_REG_SOFT_RESET_0.
                bit = 11
            else:
                raise NotImplementedError()
            existing_config = conv_to_uint32(self.read(coordinate_pair, 0xFFB121B0, 4))
            new_config = bit_change_method(existing_config, bit)
            if existing_config != new_config:
                self.write(coordinate_pair, 0xFFB121B0, conv_to_bytes(new_config))
        else:
            existing_config = conv_to_uint32(self.read(coordinate_pair, 0xFFB121B0, 4))
            new_config = bit_change_method(existing_config, 11)
            new_config = bit_change_method(new_config, 18)
            new_config = bit_change_method(new_config, 12)
            new_config = bit_change_method(new_config, 13)
            new_config = bit_change_method(new_config, 14)
            if existing_config != new_config:
                self.write(coordinate_pair, 0xFFB121B0, conv_to_bytes(new_config))


class DeviceTileDiagnostics:
    def __init__(
        self,
        brisc_diagnostics=False,
        ncrisc_diagnostics=False,
        trisc0_diagnostics=False,
        trisc1_diagnostics=False,
        trisc2_diagnostics=False,
        noc0_diagnostics=False,
        noc1_diagnostics=False,
        coprocessor_diagnostics=None,
    ):
        self.brisc_diagnostics = brisc_diagnostics
        self.ncrisc_diagnostics = ncrisc_diagnostics
        self.trisc0_diagnostics = trisc0_diagnostics
        self.trisc1_diagnostics = trisc1_diagnostics
        self.trisc2_diagnostics = trisc2_diagnostics
        self.noc0_diagnostics = noc0_diagnostics
        self.noc1_diagnostics = noc1_diagnostics
        self.coprocessor_diagnostics = coprocessor_diagnostics

    def reportBRISC(self):
        return self.brisc_diagnostics

    def reportNCRISC(self):
        return self.ncrisc_diagnostics

    def reportTRISC0(self):
        return self.trisc0_diagnostics

    def reportTRISC1(self):
        return self.trisc1_diagnostics

    def reportTRISC2(self):
        return self.trisc2_diagnostics

    def reportNoC0(self):
        return self.noc0_diagnostics

    def reportNoC1(self):
        return self.noc1_diagnostics

    def getTensixCoprocessorDiagnostics(self):
        return self.coprocessor_diagnostics


class TTDeviceTile(DeviceTile, ABC):
    # Unified-coord band layout:
    #   x ∈ {16..25}                       — DRAM (16-17, 16-18), Tensix workers (18-25, 16-25)
    #   y ∈ {14..15}                       — Ethernet tiles (16-23, 14-15)
    #   y ∈ {16..25}                       — DRAM + Tensix as above
    # Eth lives below the worker/DRAM band so the historical (18, 18) default
    # for Tensix is preserved verbatim.
    UNIFIED_COORD_X_MIN = 16
    UNIFIED_COORD_X_MAX = 25
    UNIFIED_COORD_Y_MIN = 14
    UNIFIED_COORD_Y_MAX = 25

    #: Whether this tile is a Tensix worker. The device treats Tensix tiles
    #: specially (they carry the heavy per-cycle clock load and are the only
    #: tiles that take part in soft reset). Overridden to ``True`` by
    #: ``TensixTile``; keeping the discriminator on the tile avoids the base
    #: device depending on the concrete Wormhole tile classes.
    is_tensix = False

    def get_clock_partition(self):
        """Split ``get_clocks()`` into ``(gated, always)``.

        ``gated`` components are skipped while the tile is dormant; ``always``
        components are ticked every cycle regardless (see
        :class:`~tt_sim.device.clock.TileClock`). The concatenation must equal
        ``get_clocks()`` in order, because ``always`` is ticked after
        ``gated`` — so only a trailing slice may be moved across.
        Default: everything is gated.
        """
        return list(self.get_clocks()), []

    def clock_quiescent(self):
        """True when ticking this tile's gated components changes nothing.

        The generic implementation asks every gated component (see
        :meth:`~tt_sim.device.clock.Clockable.is_clock_idle`), which fails
        safe: a component that has not opted in keeps the tile awake forever.
        Subclasses override to put the cheapest discriminator first — a
        Tensix tile with a core out of reset is never quiescent, and finding
        that out costs five attribute reads rather than twenty probe calls.
        """
        for probe in self._idle_probes:
            if not probe():
                return False
        return True

    def _bind_clock(self, tile_clock):
        """Attach the tile's :class:`TileClock` and precompute its probes.

        Called by ``TT_Device`` once the tile clock exists. The wake hooks go
        onto the components that can be acted on from outside while the tile
        is dormant: the two NIUs (a NoC message from another tile) and the
        tile-control block (a soft-reset write from any path).
        """
        self.clock = tile_clock
        gated, _ = self.get_clock_partition()
        self._idle_probes = tuple(item.is_clock_idle for item in gated)
        self.noc0_router.clock_owner = tile_clock
        self.noc1_router.clock_owner = tile_clock
        tile_ctrl = getattr(self, "tile_ctrl", None)
        if tile_ctrl is not None:
            tile_ctrl.clock_owner = tile_clock

    def __init__(self, coord_x, coord_y, noc0_router, noc1_router):
        # Reference the bounds via ``self`` so an architecture whose tile coords
        # occupy a different range (e.g. Blackhole's 17x12 physical grid) can
        # override the class constants in a subclass.
        if not (
            self.UNIFIED_COORD_X_MIN <= coord_x <= self.UNIFIED_COORD_X_MAX
            and self.UNIFIED_COORD_Y_MIN <= coord_y <= self.UNIFIED_COORD_Y_MAX
        ):
            raise Exception(
                f"Tile coordinate ({coord_x}, {coord_y}) is outside this "
                f"architecture's tile-coordinate range "
                f"(x in {self.UNIFIED_COORD_X_MIN}..{self.UNIFIED_COORD_X_MAX}, "
                f"y in {self.UNIFIED_COORD_Y_MIN}..{self.UNIFIED_COORD_Y_MAX})"
            )
        super().__init__(coord_x, coord_y, noc0_router, noc1_router)
        #: Set by ``TT_Device._register_tile_internals`` via ``_bind_clock``.
        self.clock = None
        self._idle_probes = ()
        # Extra SoC-physical NoC 0 coords that ``TT_Device`` registers as
        # noc-directory aliases on both NoCs (so kernels addressing either
        # sub-endpoint hit the same tile). Default empty; DRAMTile populates.
        self.noc_aliases: tuple[tuple[int, int], ...] = ()


# The concrete Wormhole device and its tiles moved to ``wormhole.py``. They are
# re-exported here so ``from tt_sim.device.tt_device import Wormhole`` (and the
# tile classes) keeps working. This is done lazily via module ``__getattr__``
# (PEP 562) rather than a top-level import, so it stays safe regardless of which
# module is imported first — a plain import would be circular, since
# ``wormhole`` imports the base classes from this module.
_REEXPORTED = ("Wormhole", "DRAMTile", "EthTile", "TensixTile")


def __getattr__(name):
    if name in _REEXPORTED:
        from tt_sim.device import wormhole

        return getattr(wormhole, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
