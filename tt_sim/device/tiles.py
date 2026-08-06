"""Arch-agnostic device tiles: ``DRAMTile``, ``EthTile``, ``TensixTile``.

These build a single tile's memory, NoC routers, baby cores and (for Tensix)
the coprocessor from an :class:`~tt_sim.arch.profile.ArchProfile`, so the same
classes back both Wormhole and Blackhole — the arch differences arrive through
the required ``profile`` argument, never a hard-coded default. The per-arch
``Wormhole`` / ``Blackhole`` devices (``wormhole.py`` / ``blackhole.py``) just
assemble these into a netlist. See ``docs/plans/blackhole-support.md``.
"""

from tt_sim.device.tt_device import (
    TTDeviceTile,
)
from tt_sim.memory.memory import DRAM, SparseDRAM, TensixMemory, TileMemory
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
from tt_sim.perf.model import dram_cost_model
from tt_sim.trace import Unit, get_registry
from tt_sim.util.conversion import conv_to_bytes


class DRAMEndpointNUI(NUI):
    """A DRAM channel's NIU, which charges the device's own service time.

    The NoC hop model (``docs/plans/cost-model.md``) times a packet's *flight*
    and parks it until it lands. DRAM access latency is the different thing
    that happens next: the time the endpoint takes to service the request once
    it has arrived. It belongs here rather than in the flight, because a flight
    time is a property of the distance and this is a property of the device —
    and because keeping them apart is what lets the number be derived at all
    (``dram.access_latency``'s derivation subtracts one end-to-end measurement
    from another so that the NoC term cancels).

    Two terms, since 2026-08-04, and the second is a *rate* rather than a
    latency. A DRAM transfer crosses two queues in series at two different
    rates — the NoC link at ``noc.flit_bits / 8`` bytes per cycle, and the
    GDDR6 channel at ``dram.channel_serialisation.bytes_per_cycle`` — so it
    runs at the slower of them. The link's share is already charged, once, by
    ``NUI._bandwidth_delay`` when the packet carrying the bytes is injected;
    what is added here is the **excess** of the channel's time over the link's,
    so the round trip pays ``ceil(N / channel_rate)`` and not the sum. Charging
    the sum is the double-billing that (wrongly) kept ``dram.bandwidth``
    unconsumed until rung 2 measured a Wormhole DRAM read plateauing at 24.38
    B/cycle — the channel's published 24, not a fraction of the link's 32.

    Modelled by holding the *request* rather than delaying the response, which
    is both the more faithful ordering — the data really is not read until the
    device has got to it, so a write becomes visible late and its ACK later
    still — and free in machinery: :meth:`NUI.transmit` already parks a packet
    in ``delayed_arrivals`` until its cycle, and ``DRAMTile.next_wake_cycle``
    already reports ``next_arrival``, so the tile *sleeps* through the service
    window instead of being ticked 99 times for nothing.

    Only requests are charged. A DRAM tile has no request initiator of its own
    (its NIU registers are not mapped into any core's address space), so in
    practice nothing else arrives here — but naming the three request actions
    keeps that an assertion of intent rather than an accident.

    ``service_cycles`` is ``None`` with ``TT_SIM_COST_MODEL`` unset **and** on
    any architecture whose table sources no latency, so this class then costs
    one attribute read and one extra frame per inbound packet, and changes no
    cycle.
    """

    #: The actions that reach a DRAM endpoint from outside — everything a
    #: remote NIU can ask a memory to do. Responses (``RESPONSE_READ``,
    #: ``ACK``, ``RESPONSE_ATOMIC``) only ever travel the other way.
    _SERVICED_ACTIONS = frozenset(
        {
            NUI.NoCDataRequest.DataRequestAction.READ,
            NUI.NoCDataRequest.DataRequestAction.WRITE,
            NUI.NoCDataRequest.DataRequestAction.ATOMIC,
        }
    )

    def __init__(self, *args, service_cycles=None, dram_cost=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_cycles = service_cycles
        #: The :class:`~tt_sim.perf.model.DramCostModel`, or ``None``. Only its
        #: channel rate is read here; the service time is unpacked above so the
        #: hot path is one attribute read rather than a method call.
        self.dram_cost = dram_cost

    def _channel_excess(self, data_request):
        """Cycles the channel is slower than the link, for this request's bytes.

        ``data_length_bytes`` rather than what is on the wire: it is the
        transaction size on *both* legs of a read, and the array is read (or
        written) once whichever leg carries the payload. So a read's bytes are
        charged when its request lands and a write's when its data does, which
        is the same total either way and needs no second hook on the response.
        """
        model = self.dram_cost
        if model is None or model.channel_bytes_per_cycle is None:
            return 0
        noc = self.noc_latency
        link = (
            None
            if noc is None
            else noc.serialisation_cycles(data_request.data_length_bytes)
        )
        return model.channel_excess_cycles(data_request.data_length_bytes, link)

    def transmit(self, data_request, delay=None):
        service = self.service_cycles
        if service is not None and data_request.action in self._SERVICED_ACTIONS:
            service += self._channel_excess(data_request)
            delay = service if delay is None else delay + service
        NUI.transmit(self, data_request, delay)


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
        *,
        profile,
        noc1_endpoint_coord=None,
    ):
        dram_tile_mem_map = MemoryMap()

        # One flat range covering the whole channel (2 GiB Wormhole /
        # 0xFF00_0000 Blackhole, from the profile). tt-metal banks the channel
        # into views — Wormhole's two 1 GiB views put the second one at
        # 0x4000_0000 — but those are offsets *within* this space, not separate
        # memories, so registering the channel whole is both simpler and what
        # the hardware does. Modelling less than the channel leaves holes: a
        # top-down allocation (``DeviceLocalBufferConfig{.bottom_up = false}``)
        # starts at the very top of the bank and used to land outside every
        # registered range. Sparse because 6 x 2 GiB / 8 x ~4 GiB cannot be
        # allocated up front; untouched chunks read as zeros.
        self.ddr = SparseDRAM(profile.dram_channel_size)
        dram_tile_mem_map[AddressRange(0x0, self.ddr.getSize())] = self.ddr

        self.dram_memory = TileMemory(dram_tile_mem_map, safe, snoop_addresses)

        # The DRAM device's own service time, or None with the cost model off
        # (and on any arch whose table sources no latency — both shipped ones
        # now do; see ``dram.access_latency``). The same model carries the
        # channel rate the endpoint charges on top, which is Wormhole-only.
        latency = dram_cost_model(profile.name)
        service_cycles = None if latency is None else latency.service_cycles

        # tile_kind="D" so NoC reads sourced from this tile pick the DRAM
        # congruence rule (32 B Wormhole / 64 B Blackhole) rather than L1's 16 B.
        r0 = DRAMEndpointNUI(
            0,
            physical_x,
            physical_y,
            self.dram_memory,
            tile_kind="D",
            service_cycles=service_cycles,
            dram_cost=latency,
            **profile.noc_kwargs,
        )
        r1 = DRAMEndpointNUI(
            1,
            physical_x,
            physical_y,
            self.dram_memory,
            tile_kind="D",
            service_cycles=service_cycles,
            dram_cost=latency,
            **profile.noc_kwargs,
        )

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

    def next_wake_cycle(self, cycle_num):
        # A DRAM tile's only clockables are its two NIUs, and nothing runs on
        # it — so it needs the next cycle exactly when one of them has a
        # request queued or arriving, and otherwise nothing but an inbound
        # transmit() can ever wake it. Written out rather than looping the
        # generic probes because it is on the per-cycle path of every DRAM
        # tile in the device.
        n0 = self.noc0_router
        n1 = self.noc1_router
        if (
            n0.noc_requests_to_handle
            or n0.noc_new_requests_to_handle
            or n1.noc_requests_to_handle
            or n1.noc_new_requests_to_handle
        ):
            return cycle_num + 1
        # A packet still in flight under the NoC latency model (§I) names the
        # cycle it lands on, so the tile sleeps through the flight instead of
        # spinning. Both are None without TT_SIM_COST_MODEL, which is two
        # attribute reads and the same ``return None`` as before.
        a0 = n0.next_arrival
        a1 = n1.next_arrival
        if a0 is None and a1 is None:
            return None
        soonest = a1 if a0 is None else (a0 if a1 is None else min(a0, a1))
        return soonest if soonest > cycle_num else cycle_num + 1

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
        *,
        profile,
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
            arch=profile.name,
        )

        super().__init__(coord_x, coord_y, noc0_router, noc1_router)

    def get_clocks(self):
        # tile_ctrl is deliberately absent: since Phase 2 it latches nothing
        # (the wall clock is read from the tile's TileClock on demand), so it
        # has no per-cycle work and does not need registering.
        return [self.noc0_router, self.noc1_router, self.erisc]

    def next_wake_cycle(self, cycle_num):
        # A running core needs the next cycle — unless the firmware-loop
        # recogniser has it parked, in which case its own probe (in the
        # generic sweep below) decides.
        if self.erisc.soft_active and not self.erisc.spin_parked:
            return cycle_num + 1
        return TTDeviceTile.next_wake_cycle(self, cycle_num)

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
        *,
        profile,
    ):
        self.tensix_coprocessor = TensixCoProcessor(
            coprocessor_diagnostics,
            cfg_state_size=profile.tensix_cfg_state_size,
            thd_state_size=profile.tensix_thd_state_size,
            blackhole=profile.tensix_blackhole,
        )

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
            arch=profile.name,
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
            arch=profile.name,
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
            arch=profile.name,
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
            arch=profile.name,
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
            arch=profile.name,
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
        ]

    def next_wake_cycle(self, cycle_num):
        # Fast reject first: a tile with any baby core genuinely running needs
        # the very next cycle, and a few attribute reads settle that. A core
        # the firmware-loop recogniser has parked (``spin_parked`` — see
        # ``tt_sim/pe/rv/spin.py``) is exempt: its own ``next_wake_cycle``
        # probe in the generic sweep below verifies the watched memory and
        # decides. ``spin_parked`` is False for any running core, so the
        # common live-tile path still short-circuits on the first conjunct.
        brisc = self.brisc
        ncrisc = self.ncrisc
        trisc0 = self.trisc0
        trisc1 = self.trisc1
        trisc2 = self.trisc2
        if (
            (brisc.soft_active and not brisc.spin_parked)
            or (ncrisc.soft_active and not ncrisc.spin_parked)
            or (trisc0.soft_active and not trisc0.spin_parked)
            or (trisc1.soft_active and not trisc1.spin_parked)
            or (trisc2.soft_active and not trisc2.spin_parked)
        ):
            return cycle_num + 1
        return TTDeviceTile.next_wake_cycle(self, cycle_num)

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
