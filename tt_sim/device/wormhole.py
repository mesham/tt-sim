"""The Wormhole B0 device and its tiles.

``Wormhole`` (a :class:`~tt_sim.device.tt_device.TT_Device`) assembles the
Wormhole netlist: six ``DRAMTile``s, the ethernet tiles, and one or more
``TensixTile``s, wired into both NoC directories. The per-arch constants come
from ``WORMHOLE_PROFILE``; the arch-agnostic base classes live in
``tt_sim/device/tt_device.py``. See ``docs/plans/blackhole-support.md``.
"""

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.device.deadlock import DeadlockDetector, deadlock_config_from_env
from tt_sim.device.tt_device import (
    DeviceTileDiagnostics,
    TT_Device,
    TTDeviceTile,
)
from tt_sim.memory.memory import DRAM, TensixMemory, TileMemory
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.misc.mailbox import Mailbox
from tt_sim.misc.tile_ctrl import TensixTileControl
from tt_sim.misc.ttsync import TTSync
from tt_sim.network.tt_noc import NUI, NoCOverlay
from tt_sim.pe.pcbuf import PCBuf
from tt_sim.pe.pe import PEMemory
from tt_sim.pe.rv.babyriscv import BabyRISCV, BabyRISCVCoreType
from tt_sim.pe.tensix.tdma import TDMA
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.trace import Unit, enable_from_env, get_registry
from tt_sim.util.conversion import conv_to_bytes


class Wormhole(TT_Device):
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
        # Set before any tile is built: ``_build_tensix_tile`` reads
        # ``self.profile.tensix_l1_size``, and ``TT_Device.__init__`` (called
        # below) reads the profile for NoC-1 mirroring.
        self.profile = WORMHOLE_PROFILE
        if diagnostics is None:
            # All off by default if no diagnostics provided
            diagnostics = DeviceTileDiagnostics()
        # Saved so ``add_tensix_tile`` can construct lazily-materialised tiles
        # with the same diagnostic flags as the originals.
        self.diagnostics = diagnostics
        if tensix_coords is None:
            tensix_coords = self.profile.tensix_unified_coords
        # Opt-in structured tracing: if TT_SIM_TRACE*=<...> is set in the
        # environment, the bus is enabled and writers are registered
        # before any device-construction events are missed. The
        # state-dump writer specifically needs a device reference,
        # so we call enable_from_env again at the bottom of __init__
        # after tiles are constructed.
        enable_from_env()
        dram_tiles = []
        for unified, physicals in zip(
            self.profile.dram_channel_unified_coords,
            self.profile.dram_channel_physical_noc0_coords,
        ):
            primary = physicals[0]
            aliases = physicals[1:]
            dram_tiles.append(
                DRAMTile(
                    unified[0],
                    unified[1],
                    primary[0],
                    primary[1],
                    aliases,
                    profile=self.profile,
                )
            )
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

        enabled, threshold = deadlock_config_from_env()
        self.deadlock_detector = DeadlockDetector(
            threshold,
            enabled,
            tensix_tiles,
            dram_tiles,
        )
        self.clocks[0].on_tick = self.deadlock_detector.tick
        # Second call wires the state-dump writer now that we have tiles.
        enable_from_env(wormhole=self)

    def _build_tensix_tile(self, coord):
        x, y = coord
        physical_x, physical_y = Wormhole.physical_noc0_coord_from_unified_worker(coord)
        return TensixTile(
            x,
            y,
            physical_x,
            physical_y,
            self.diagnostics.reportBRISC(),
            self.diagnostics.reportNCRISC(),
            self.diagnostics.reportTRISC0(),
            self.diagnostics.reportTRISC1(),
            self.diagnostics.reportTRISC2(),
            self.diagnostics.reportNoC0(),
            self.diagnostics.reportNoC1(),
            self.diagnostics.getTensixCoprocessorDiagnostics(),
            profile=self.profile,
        )

    def add_tensix_tile(self, coord):
        """Construct and register a TensixTile at the given unified coord.

        Used by the wire bridge for lazy multi-Tensix materialisation: when
        tt-metal addresses a worker the simulator hasn't built yet, this
        method stands up the matching ``TensixTile`` and wires it into the
        directory, NoC topology, clock/reset aggregators, and deadlock
        detector. Returns the new tile.
        """
        tile = self._build_tensix_tile(coord)
        self.register_tensix_tile(tile)
        self.deadlock_detector.add_tensix_tile(tile)
        return tile


class DRAMTile(TTDeviceTile):
    def __init__(
        self,
        coord_x,
        coord_y,
        physical_x,
        physical_y,
        physical_aliases=(),
        safe=True,
        snoop_addresses=None,
        profile=WORMHOLE_PROFILE,
        noc1_endpoint_coord=None,
    ):
        dram_tile_mem_map = MemoryMap()

        self.ddr_bank_0 = DRAM(10 * 1024 * 1024)
        ddr_range = AddressRange(0x0, self.ddr_bank_0.getSize())
        dram_tile_mem_map[ddr_range] = self.ddr_bank_0

        self.ddr_bank_1 = DRAM(10 * 1024 * 1024)
        ddr_range = AddressRange(0x0_4000_0000, self.ddr_bank_1.getSize())
        dram_tile_mem_map[ddr_range] = self.ddr_bank_1

        self.dram_memory = TileMemory(dram_tile_mem_map, safe, snoop_addresses)

        r0 = NUI(0, physical_x, physical_y, self.dram_memory, **profile.noc_kwargs)
        r1 = NUI(1, physical_x, physical_y, self.dram_memory, **profile.noc_kwargs)

        # Register DRAM-tile NoC routers so their NoCEvents (request-phase
        # arrivals at the destination tile) appear in the trace.
        registry = get_registry()
        r0.unit_id = registry.register(0, coord_y, coord_x, Unit.NOC0).as_tuple()
        r1.unit_id = registry.register(0, coord_y, coord_x, Unit.NOC1).as_tuple()

        super().__init__(coord_x, coord_y, r0, r1)
        self.noc_aliases = tuple(physical_aliases)
        # SoC-physical NoC 1 endpoint of this DRAM channel, when it differs from
        # the NoC 0 endpoint (Blackhole). The device mirrors this — not the NoC 0
        # coord — into the NoC 1 routing table. See ``TT_Device._register``.
        self.noc1_endpoint_coord = noc1_endpoint_coord

    def get_clocks(self):
        return [self.noc0_router, self.noc1_router]

    def get_resets(self):
        return []

    def read(self, address, size):
        return self.dram_memory.read(address, size)

    def write(self, address, value, size=None):
        return self.dram_memory.write(address, value, size)

    def getSize(self):
        # Dummy value for now
        return 0xFFFF


class EthTile(TTDeviceTile):
    """Wormhole Ethernet tile — L1 SRAM + one ERisc baby core.

    Per the WormholeB0 EthernetTile ISA docs: 256 KiB L1, one RV32IM baby
    core (ERisc), two NoC connections. The ethernet MAC/PHY and chip-to-
    chip routing are *not* modelled (no second chip exists in tt-sim yet),
    but the L1 + ERisc are enough for single-chip kernels that hardcode an
    eth coord (e.g. ``hello_world_datatypes_kernel`` reading ``(1, 0)``) to
    see deterministic memory-backed state instead of the former
    ``NullEndpoint`` zero-fill, and for a driver script to run RV32IM code
    on the ERisc core.
    """

    # Eth tiles must NOT register noc1-mirror keys in the device's NoC 1
    # directory: the mirror formula maps eth canonicals to DRAM canonicals
    # (e.g. eth ``(9, 0)`` ↔ DRAM ch0 ``(0, 11)``), so mirror-registration
    # would overwrite DRAM primary entries and route DRAM-targeted NoC 1
    # traffic into eth L1. See ``TT_Device._register_tile_internals``.
    register_noc1_mirror = False

    # Per ``driver/wormhole/soc_descriptor.yaml`` (``eth_l1_size: 262144``).
    L1_SIZE = 0x40000
    # Per WormholeB0/EthernetTile/BabyRISCV/README.md.
    ERISC_LOCAL_MEM_BASE = 0xFFB00000
    ERISC_LOCAL_MEM_SIZE = 4 * 1024
    ERISC_IRAM_BASE = 0xFFC00000
    ERISC_IRAM_SIZE = 64 * 1024

    def __init__(
        self,
        coord_x,
        coord_y,
        physical_x,
        physical_y,
        erisc_snoop=False,
        safe=True,
        snoop_addresses=None,
        profile=WORMHOLE_PROFILE,
    ):
        eth_tile_mem_map = MemoryMap()

        self.L1_mem = DRAM(EthTile.L1_SIZE)
        l1_range = AddressRange(0x0, self.L1_mem.getSize())
        eth_tile_mem_map[l1_range] = self.L1_mem

        # NUIs handle inbound NoC requests by reading/writing L1 directly —
        # matches TensixTile's pattern of passing L1_mem (not the full tile
        # memory) as attached_memory. Outbound MMIO writes from the local
        # ERisc to the NUI's register file are dispatched via the tile
        # memory map below, which routes 0xFFB20000/0xFFB30000 to the NUI's
        # own ``write``/``read`` MMIO handlers.
        noc0_router = NUI(0, physical_x, physical_y, self.L1_mem, **profile.noc_kwargs)
        noc0_range = AddressRange(0xFFB20000, noc0_router.getSize())
        eth_tile_mem_map[noc0_range] = noc0_router

        noc1_router = NUI(1, physical_x, physical_y, self.L1_mem, **profile.noc_kwargs)
        noc1_range = AddressRange(0xFFB30000, noc1_router.getSize())
        eth_tile_mem_map[noc1_range] = noc1_router

        # Tile control owns RISCV_DEBUG_REG_SOFT_RESET_0 (offset 0x1B0 inside
        # the 0xFFB12000 region). ERisc reset bit is 11 per the ISA docs;
        # held in reset at power-on so a wormhole.reset() doesn't run the
        # core before firmware is loaded.
        self.tile_ctrl = TensixTileControl()
        self.tile_ctrl.RISCV_DEBUG_REG_SOFT_RESET_0 = conv_to_bytes(1 << 11)
        tile_ctrl_range = AddressRange(0xFFB12000, self.tile_ctrl.getSize())
        eth_tile_mem_map[tile_ctrl_range] = self.tile_ctrl

        self.eth_memory = TileMemory(eth_tile_mem_map, safe, snoop_addresses)

        registry = get_registry()
        noc0_router.unit_id = registry.register(
            0, coord_y, coord_x, Unit.NOC0
        ).as_tuple()
        noc1_router.unit_id = registry.register(
            0, coord_y, coord_x, Unit.NOC1
        ).as_tuple()

        # ERisc per-core address space: local data RAM (4 KiB) + IRAM (64
        # KiB). Local mem lives at the same architectural base (0xFFB00000)
        # as BRISC's, distinguished by being in this tile's address space.
        self.local_mem_erisc = DRAM(EthTile.ERISC_LOCAL_MEM_SIZE)
        local_mem_range = AddressRange(
            EthTile.ERISC_LOCAL_MEM_BASE, self.local_mem_erisc.getSize()
        )
        self.local_imem_erisc = DRAM(EthTile.ERISC_IRAM_SIZE)
        local_imem_range = AddressRange(
            EthTile.ERISC_IRAM_BASE, self.local_imem_erisc.getSize()
        )
        erisc_mem_map = MemoryMap()
        erisc_mem_map[local_mem_range] = self.local_mem_erisc
        erisc_mem_map[local_imem_range] = self.local_imem_erisc
        self.erisc_mem = PEMemory(erisc_mem_map)

        self.erisc = BabyRISCV(
            BabyRISCVCoreType.ERISC,
            [self.eth_memory, self.erisc_mem],
            snoop=erisc_snoop,
            isa_extensions=profile.baby_core_isa_extensions,
        )

        super().__init__(coord_x, coord_y, noc0_router, noc1_router)

    def get_clocks(self):
        return [self.noc0_router, self.noc1_router, self.erisc, self.tile_ctrl]

    def get_resets(self):
        return [self.erisc]

    def read(self, address, size):
        return self.eth_memory.read(address, size)

    def write(self, address, value, size=None):
        return self.eth_memory.write(address, value, size)

    def getSize(self):
        return 0xFFFF


class TensixTile(TTDeviceTile):
    is_tensix = True

    def __init__(
        self,
        coord_x,
        coord_y,
        physical_x,
        physical_y,
        brisc_snoop=False,
        ncrisc_snoop=False,
        trisc0_snoop=False,
        trisc1_snoop=False,
        trisc2_snoop=False,
        noc0_snoop=False,
        noc1_snoop=False,
        coprocessor_diagnostics=None,
        profile=WORMHOLE_PROFILE,
    ):
        self.tensix_coprocessor = TensixCoProcessor(coprocessor_diagnostics)

        self._mb_brisc = Mailbox(BabyRISCVCoreType.BRISC)
        self._mb_trisc0 = Mailbox(BabyRISCVCoreType.TRISC0)
        self._mb_trisc1 = Mailbox(BabyRISCVCoreType.TRISC1)
        self._mb_trisc2 = Mailbox(BabyRISCVCoreType.TRISC2)
        mb_brisc = self._mb_brisc
        mb_trisc0 = self._mb_trisc0
        mb_trisc1 = self._mb_trisc1
        mb_trisc2 = self._mb_trisc2

        mb_brisc.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])
        mb_trisc0.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])
        mb_trisc1.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])
        mb_trisc2.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])

        tensix_mem_map = MemoryMap()

        # Tensix L1 SRAM. Size comes from the architecture profile; per
        # WormholeB0/TensixTile/L1.md, Wormhole's is TENSIX_SRAM_SIZE =
        # 1464 * 1024 = 1,499,136 bytes.
        self.L1_mem = DRAM(profile.tensix_l1_size)
        l1_range = AddressRange(0x0, self.L1_mem.getSize())
        tensix_mem_map[l1_range] = self.L1_mem

        self.tensix_coprocessor_be_config = (
            self.tensix_coprocessor.getBackend().getConfigUnit()
        )
        tensix_config_range = AddressRange(
            0xFFEF0000, self.tensix_coprocessor_be_config.getSize()
        )
        tensix_mem_map[tensix_config_range] = self.tensix_coprocessor_be_config

        noc0_router = NUI(
            0, physical_x, physical_y, self.L1_mem, noc0_snoop, **profile.noc_kwargs
        )
        noc0_range = AddressRange(0xFFB20000, noc0_router.getSize())
        tensix_mem_map[noc0_range] = noc0_router

        noc1_router = NUI(
            1, physical_x, physical_y, self.L1_mem, noc1_snoop, **profile.noc_kwargs
        )
        noc1_range = AddressRange(0xFFB30000, noc1_router.getSize())
        tensix_mem_map[noc1_range] = noc1_router

        self.noc_overlay = NoCOverlay()
        noc_overlay_range = AddressRange(0xFFB40000, self.noc_overlay.getSize())
        tensix_mem_map[noc_overlay_range] = self.noc_overlay

        self.tdma = TDMA(self.tensix_coprocessor, self.L1_mem)
        tdma_range = AddressRange(0xFFB11000, self.tdma.getSize())
        tensix_mem_map[tdma_range] = self.tdma

        self.tile_ctrl = TensixTileControl()
        # All 5 baby cores (BRISC=11, TRISC0=12, TRISC1=13, TRISC2=14,
        # NCRISC=18) come out of power-on held in soft reset on real silicon.
        # Without this, multi-tile setups would have sibling tiles' cores
        # running from PC=0 while only the target tile's BRISC is brought up
        # by launch_firmware.
        initial_reset = (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14) | (1 << 18)
        self.tile_ctrl.RISCV_DEBUG_REG_SOFT_RESET_0 = conv_to_bytes(initial_reset)
        tile_ctrl_range = AddressRange(0xFFB12000, self.tile_ctrl.getSize())
        tensix_mem_map[tile_ctrl_range] = self.tile_ctrl

        self.pc_buf_0 = PCBuf(self.tile_ctrl, 0)
        self.pc_buf_1 = PCBuf(self.tile_ctrl, 1)
        self.pc_buf_2 = PCBuf(self.tile_ctrl, 2)

        self.ttsync_0 = TTSync(self.tile_ctrl, self.tensix_coprocessor, 0)
        self.ttsync_1 = TTSync(self.tile_ctrl, self.tensix_coprocessor, 1)
        self.ttsync_2 = TTSync(self.tile_ctrl, self.tensix_coprocessor, 2)

        self.tensix_mem = TensixMemory(tensix_mem_map)

        # Create brisc CPU
        self.local_mem_brisc = DRAM(profile.brisc_local_mem_size)
        local_mem_brisc_range = AddressRange(0xFFB00000, self.local_mem_brisc.getSize())
        brisc_pc_buf_0_range = AddressRange(0xFFE80000, self.pc_buf_0.getSize())
        brisc_pc_buf_1_range = AddressRange(0xFFE90000, self.pc_buf_1.getSize())
        brisc_pc_buf_2_range = AddressRange(0xFFEA0000, self.pc_buf_2.getSize())
        brisc_mb_range = AddressRange(0xFFEC0000, mb_brisc.getSize())
        tensix_cp_thread_0_range = AddressRange(0xFFE40000, 0xFFFF)
        tensix_cp_thread_1_range = AddressRange(0xFFE50000, 0xFFFF)
        tensix_cp_thread_2_range = AddressRange(0xFFE60000, 0xFFFF)
        brisc_tensix_gpr_range = AddressRange(
            0xFFE00000, self.tensix_coprocessor.getBackend().getGPR().getSize()
        )

        brisc0_mem_map = MemoryMap()
        brisc0_mem_map[local_mem_brisc_range] = self.local_mem_brisc
        brisc0_mem_map[brisc_pc_buf_0_range] = self.pc_buf_0
        brisc0_mem_map[brisc_pc_buf_1_range] = self.pc_buf_1
        brisc0_mem_map[brisc_pc_buf_2_range] = self.pc_buf_2
        brisc0_mem_map[tensix_cp_thread_0_range] = self.tensix_coprocessor.getThread(0)
        brisc0_mem_map[tensix_cp_thread_1_range] = self.tensix_coprocessor.getThread(1)
        brisc0_mem_map[tensix_cp_thread_2_range] = self.tensix_coprocessor.getThread(2)
        brisc0_mem_map[brisc_tensix_gpr_range] = (
            self.tensix_coprocessor.getBackend().getGPR()
        )
        brisc0_mem_map[brisc_mb_range] = mb_brisc

        self.brisc0_mem = PEMemory(brisc0_mem_map)

        self.brisc = BabyRISCV(
            BabyRISCVCoreType.BRISC,
            [self.tensix_mem, self.brisc0_mem],
            snoop=brisc_snoop,
            isa_extensions=profile.baby_core_isa_extensions,
        )

        # Create ncrisc CPU
        self.local_mem_ncrisc = DRAM(profile.ncrisc_local_mem_size)
        local_mem_ncrisc_range = AddressRange(
            0xFFB00000, self.local_mem_ncrisc.getSize()
        )
        # ncrisc also has 16KB of IRAM (we don't distinguish here but in reality
        # this can only be accessed by ncrisc frontend and not by instructions
        # when they are executed, but that is fine as instruction fetch is
        # frontend and this IRAM is just used for instructions
        self.local_imem_ncrisc = DRAM(16384)
        local_imem_ncrisc_range = AddressRange(
            0xFFC00000, self.local_imem_ncrisc.getSize()
        )
        ncrisc_mem_map = MemoryMap()
        ncrisc_mem_map[local_mem_ncrisc_range] = self.local_mem_ncrisc
        ncrisc_mem_map[local_imem_ncrisc_range] = self.local_imem_ncrisc
        self.ncrisc_mem = PEMemory(ncrisc_mem_map)

        self.ncrisc = BabyRISCV(
            BabyRISCVCoreType.NCRISC,
            [self.tensix_mem, self.ncrisc_mem],
            snoop=ncrisc_snoop,
            reset_pc_debug_regs=profile.baby_core_reset_pc_debug_regs,
            start_address=profile.ncrisc_firmware_base,
            isa_extensions=profile.baby_core_isa_extensions,
        )

        # Common addresses for TRISC cores
        trisc_pc_buf_range = AddressRange(0xFFE80000, 0x4)
        trisc_ttsync_range = AddressRange(0xFFE80004, 0x1B)
        trisc_mb_range = AddressRange(0xFFEC0000, mb_trisc0.getSize())
        trisc_semaphores_range = AddressRange(0xFFE80020, 0xFFDF)
        trisc_mop_expander_cfg_range = AddressRange(0xFFB80000, 0x23)
        trisc_cp_thread_range = AddressRange(0xFFE40000, 0xFFFF)
        trisc_tensix_gpr_range = AddressRange(
            0xFFE00000, self.tensix_coprocessor.getBackend().getGPR().getSize()
        )

        # Create trisc0 CPU
        self.local_mem_trisc0 = DRAM(profile.trisc_local_mem_size)
        local_mem_trisc0_range = AddressRange(
            0xFFB00000, self.local_mem_trisc0.getSize()
        )
        trisc0_mem_map = MemoryMap()
        trisc0_mem_map[local_mem_trisc0_range] = self.local_mem_trisc0
        trisc0_mem_map[trisc_pc_buf_range] = self.pc_buf_0
        trisc0_mem_map[trisc_ttsync_range] = self.ttsync_0
        trisc0_mem_map[trisc_semaphores_range] = (
            self.tensix_coprocessor.getBackend().getSyncUnit()
        )
        trisc0_mem_map[trisc_mop_expander_cfg_range] = (
            self.tensix_coprocessor.getThread(0).getMOPExpander()
        )
        trisc0_mem_map[trisc_cp_thread_range] = self.tensix_coprocessor.getThread(0)
        trisc0_mem_map[trisc_tensix_gpr_range] = (
            self.tensix_coprocessor.getBackend().getGPR().getGPRPerTRISC(0)
        )
        trisc0_mem_map[trisc_mb_range] = mb_trisc0
        self.trisc0_mem = PEMemory(trisc0_mem_map)

        self.trisc0 = BabyRISCV(
            BabyRISCVCoreType.TRISC0,
            [self.tensix_mem, self.trisc0_mem],
            snoop=trisc0_snoop,
            reset_pc_debug_regs=profile.baby_core_reset_pc_debug_regs,
            start_address=profile.trisc0_firmware_base,
            isa_extensions=profile.baby_core_isa_extensions,
        )

        # Create trisc1 CPU
        self.local_mem_trisc1 = DRAM(profile.trisc_local_mem_size)
        local_mem_trisc1_range = AddressRange(
            0xFFB00000, self.local_mem_trisc1.getSize()
        )
        trisc1_mem_map = MemoryMap()
        trisc1_mem_map[local_mem_trisc1_range] = self.local_mem_trisc1
        trisc1_mem_map[trisc_pc_buf_range] = self.pc_buf_1
        trisc1_mem_map[trisc_ttsync_range] = self.ttsync_1
        trisc1_mem_map[trisc_semaphores_range] = (
            self.tensix_coprocessor.getBackend().getSyncUnit()
        )
        trisc1_mem_map[trisc_mop_expander_cfg_range] = (
            self.tensix_coprocessor.getThread(1).getMOPExpander()
        )
        trisc1_mem_map[trisc_cp_thread_range] = self.tensix_coprocessor.getThread(1)
        trisc1_mem_map[trisc_tensix_gpr_range] = (
            self.tensix_coprocessor.getBackend().getGPR().getGPRPerTRISC(1)
        )
        trisc1_mem_map[trisc_mb_range] = mb_trisc1
        self.trisc1_mem = PEMemory(trisc1_mem_map)

        self.trisc1 = BabyRISCV(
            BabyRISCVCoreType.TRISC1,
            [self.tensix_mem, self.trisc1_mem],
            snoop=trisc1_snoop,
            reset_pc_debug_regs=profile.baby_core_reset_pc_debug_regs,
            start_address=profile.trisc1_firmware_base,
            isa_extensions=profile.baby_core_isa_extensions,
        )

        # Create trisc2 CPU
        self.local_mem_trisc2 = DRAM(profile.trisc_local_mem_size)
        local_mem_trisc2_range = AddressRange(
            0xFFB00000, self.local_mem_trisc2.getSize()
        )
        trisc2_mem_map = MemoryMap()
        trisc2_mem_map[local_mem_trisc2_range] = self.local_mem_trisc2
        trisc2_mem_map[trisc_pc_buf_range] = self.pc_buf_2
        trisc2_mem_map[trisc_ttsync_range] = self.ttsync_2
        trisc2_mem_map[trisc_semaphores_range] = (
            self.tensix_coprocessor.getBackend().getSyncUnit()
        )
        trisc2_mem_map[trisc_mop_expander_cfg_range] = (
            self.tensix_coprocessor.getThread(2).getMOPExpander()
        )
        trisc2_mem_map[trisc_cp_thread_range] = self.tensix_coprocessor.getThread(2)
        trisc2_mem_map[trisc_tensix_gpr_range] = (
            self.tensix_coprocessor.getBackend().getGPR().getGPRPerTRISC(2)
        )
        trisc2_mem_map[trisc_mb_range] = mb_trisc2
        self.trisc2_mem = PEMemory(trisc2_mem_map)

        self.trisc2 = BabyRISCV(
            BabyRISCVCoreType.TRISC2,
            [self.tensix_mem, self.trisc2_mem],
            snoop=trisc2_snoop,
            reset_pc_debug_regs=profile.baby_core_reset_pc_debug_regs,
            start_address=profile.trisc2_firmware_base,
            # TRISC2 additionally carries the V vector extension (guarded).
            isa_extensions=(
                profile.baby_core_isa_extensions + profile.trisc2_isa_extensions
            ),
        )

        # Set addressable memory for Tensix co-processor
        self.tensix_coprocessor.setAddressableMemory([self.tensix_mem, self.ncrisc_mem])

        self._register_trace_ids(coord_x, coord_y, noc0_router, noc1_router)

        super().__init__(coord_x, coord_y, noc0_router, noc1_router)

    def _register_trace_ids(self, coord_x, coord_y, noc0_router, noc1_router):
        registry = get_registry()
        chip_id = 0

        def assign(component, unit):
            uid = registry.register(chip_id, coord_y, coord_x, unit)
            component.unit_id = uid.as_tuple()
            return uid

        assign(self.brisc, Unit.BRISC)
        assign(self.ncrisc, Unit.NCRISC)
        trisc0_uid = assign(self.trisc0, Unit.TRISC0)
        trisc1_uid = assign(self.trisc1, Unit.TRISC1)
        trisc2_uid = assign(self.trisc2, Unit.TRISC2)
        assign(noc0_router, Unit.NOC0)
        assign(noc1_router, Unit.NOC1)

        # TensixCoprocessor threads issue from TRISC0/1/2 — wire so the
        # dispatch event's unit_id reflects the issuing TRISC.
        self.tensix_coprocessor.getThread(0).unit_id = trisc0_uid.as_tuple()
        self.tensix_coprocessor.getThread(1).unit_id = trisc1_uid.as_tuple()
        self.tensix_coprocessor.getThread(2).unit_id = trisc2_uid.as_tuple()

        # Tensix backend units — each gets the per-tile unit_id matching
        # its architectural block; the ComputeEvent base-class publish in
        # TensixBackendUnit.clock_tick picks this up.
        backend = self.tensix_coprocessor.getBackend()
        assign(backend.matrix_unit, Unit.MATRIX)
        assign(backend.vector_unit, Unit.SFPU)
        assign(backend.packer_unit, Unit.PACKER)
        for u in backend.unpacker_units:
            uid = registry.register(chip_id, coord_y, coord_x, Unit.UNPACKER)
            u.unit_id = uid.as_tuple()
        assign(backend.mover_unit, Unit.MOVER)
        assign(backend.scalar_unit, Unit.THCON)
        assign(backend.sync_unit, Unit.SYNC)
        assign(backend.misc_unit, Unit.TDMA)
        assign(backend.config_unit, Unit.CFG)

        # Sync infrastructure (mailboxes, ttsync registers) gets MAILBOX
        # / TTSYNC unit ids so SyncEvents are attributable to their
        # architectural origin within the tile.
        for mb in (self._mb_brisc, self._mb_trisc0, self._mb_trisc1, self._mb_trisc2):
            uid = registry.register(chip_id, coord_y, coord_x, Unit.MAILBOX)
            mb.unit_id = uid.as_tuple()
        for ts in (self.ttsync_0, self.ttsync_1, self.ttsync_2):
            uid = registry.register(chip_id, coord_y, coord_x, Unit.TTSYNC)
            ts.unit_id = uid.as_tuple()

    def get_clocks(self):
        return self.tensix_coprocessor.getClocks() + [
            self.tdma,
            self.brisc,
            self.ncrisc,
            self.trisc0,
            self.trisc1,
            self.trisc2,
            self.noc0_router,
            self.noc1_router,
            self.tile_ctrl,
        ]

    def get_tensix_memory(self):
        return self.tensix_mem

    def get_resets(self):
        return [self.brisc, self.ncrisc, self.trisc0, self.trisc1, self.trisc2]

    def read(self, address, size):
        return self.tensix_mem.read(address, size)

    def write(self, address, value, size=None):
        return self.tensix_mem.write(address, value, size)

    def getSize(self):
        # Dummy value for now
        return 0xFFFF
