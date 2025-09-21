import itertools
from abc import ABC

from tt_sim.device.clock import Clock
from tt_sim.device.device import Device, DeviceTile
from tt_sim.device.reset import Reset
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
from tt_sim.pe.tensix.tensix import (
    TensixCoProcessor,
)
from tt_sim.pe.tensix.util import DiagnosticsSettings
from tt_sim.util.bits import clear_bit, set_bit
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_uint32,
)


class TT_Device(Device):
    def __init__(self, device_memory, dram_tiles, tensix_tiles):
        self.dram_tiles = dram_tiles
        self.tensix_tiles = tensix_tiles

        components_to_clock = []
        components_to_reset = []
        self.tile_directory = {}
        noc_0_directory = {}
        noc_1_directory = {}
        for tile in itertools.chain(dram_tiles, tensix_tiles):
            components_to_clock += tile.get_clocks()
            components_to_reset += tile.get_resets()
            self.tile_directory[tile.get_coord_pair()] = tile
            # Add entry to NoC directory
            noc_0_directory[tile.get_noc_nui(0).get_id_pair()] = tile.get_noc_nui(0)
            noc_1_directory[tile.get_noc_nui(1).get_id_pair()] = tile.get_noc_nui(1)

        # Set NoC directory for each tile
        for tile in itertools.chain(dram_tiles, tensix_tiles):
            tile.get_noc_nui(0).set_noc_directory(noc_0_directory)
            tile.get_noc_nui(1).set_noc_directory(noc_1_directory)

        self.clocks = [Clock(components_to_clock)]
        self.resets = [Reset(components_to_reset)]

        super().__init__(device_memory, self.clocks, self.resets)

    def read(self, coordinate_pair, address, size):
        assert coordinate_pair in self.tile_directory
        return self.tile_directory[coordinate_pair].read(address, size)

    def write(self, coordinate_pair, address, value, size=None):
        assert coordinate_pair in self.tile_directory
        self.tile_directory[coordinate_pair].write(address, value, size)

    def deassert_soft_reset(self, coordinate_pair=None, core_type=None):
        if coordinate_pair is None:
            for pair, value in self.tile_directory.items():
                if isinstance(value, TensixTile):
                    self.perform_soft_reset_change(clear_bit, pair, core_type)
        else:
            self.perform_soft_reset_change(clear_bit, coordinate_pair, core_type)

    def assert_soft_reset(self, coordinate_pair=None, core_type=None):
        if coordinate_pair is None:
            for pair, value in self.tile_directory.items():
                if isinstance(value, TensixTile):
                    self.perform_soft_reset_change(set_bit, pair, core_type)
        else:
            self.perform_soft_reset_change(set_bit, coordinate_pair, core_type)

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


class Wormhole(TT_Device):
    def __init__(self):
        dram_tile = DRAMTile(16, 16)
        tensix_tile = TensixTile(
            18,
            18,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            coprocessor_diagnostics=DiagnosticsSettings(
                unpacking=False,
                packing=False,
                configurations_set=False,
                issued_instructions=False,
                fpu_calculations=False,
                sfpu_calculations=True,
            ),
        )

        # For now don't provide any memory, in future this will be the memory
        # map of the PCIe endpoing
        super().__init__(None, [dram_tile], [tensix_tile])


class TTDeviceTile(DeviceTile, ABC):
    def __init__(self, coord_x, coord_y, noc0_router, noc1_router):
        if coord_x <= 15 or coord_y <= 15 or coord_x >= 26 or coord_y >= 26:
            raise Exception(
                f"Tensix tile coordinates should be the unified coordinate system "
                f"(16 to 25), whereas ({coord_x}, {coord_y}) provided"
            )
        super().__init__(coord_x, coord_y, noc0_router, noc1_router)


class DRAMTile(TTDeviceTile):
    def __init__(self, coord_x, coord_y, safe=True, snoop_addresses=None):
        dram_tile_mem_map = MemoryMap()

        self.ddr_bank_0 = DRAM(10 * 1024 * 1024)
        ddr_range = AddressRange(0x0, self.ddr_bank_0.getSize())
        dram_tile_mem_map[ddr_range] = self.ddr_bank_0

        self.ddr_bank_1 = DRAM(10 * 1024 * 1024)
        ddr_range = AddressRange(0x0_4000_0000, self.ddr_bank_1.getSize())
        dram_tile_mem_map[ddr_range] = self.ddr_bank_1

        self.dram_memory = TileMemory(dram_tile_mem_map, safe, snoop_addresses)

        r0 = NUI(0, coord_x, coord_y, self.dram_memory)
        r1 = NUI(1, coord_x, coord_y, self.dram_memory)

        super().__init__(coord_x, coord_y, r0, r1)

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


class TensixTile(TTDeviceTile):
    def __init__(
        self,
        coord_x,
        coord_y,
        brisc_snoop=False,
        ncrisc_snoop=False,
        trisc0_snoop=False,
        trisc1_snoop=False,
        trisc2_snoop=False,
        noc0_snoop=False,
        noc1_snoop=False,
        coprocessor_diagnostics=None,
    ):
        self.tensix_coprocessor = TensixCoProcessor(coprocessor_diagnostics)

        mb_brisc = Mailbox(BabyRISCVCoreType.BRISC)
        mb_trisc0 = Mailbox(BabyRISCVCoreType.TRISC0)
        mb_trisc1 = Mailbox(BabyRISCVCoreType.TRISC1)
        mb_trisc2 = Mailbox(BabyRISCVCoreType.TRISC2)

        mb_brisc.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])
        mb_trisc0.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])
        mb_trisc1.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])
        mb_trisc2.setOtherMBs([mb_brisc, mb_trisc0, mb_trisc1, mb_trisc2])

        tensix_mem_map = MemoryMap()

        # Create DRAM
        self.L1_mem = DRAM(1507327)
        l1_range = AddressRange(0x0, self.L1_mem.getSize())
        tensix_mem_map[l1_range] = self.L1_mem

        self.tensix_coprocessor_be_config = (
            self.tensix_coprocessor.getBackend().getConfigUnit()
        )
        tensix_config_range = AddressRange(
            0xFFEF0000, self.tensix_coprocessor_be_config.getSize()
        )
        tensix_mem_map[tensix_config_range] = self.tensix_coprocessor_be_config

        noc0_router = NUI(0, coord_x, coord_y, self.L1_mem, noc0_snoop)
        noc0_range = AddressRange(0xFFB20000, noc0_router.getSize())
        tensix_mem_map[noc0_range] = noc0_router

        noc1_router = NUI(1, coord_x, coord_y, self.L1_mem, noc1_snoop)
        noc1_range = AddressRange(0xFFB30000, noc1_router.getSize())
        tensix_mem_map[noc1_range] = noc1_router

        self.noc_overlay = NoCOverlay()
        noc_overlay_range = AddressRange(0xFFB40000, self.noc_overlay.getSize())
        tensix_mem_map[noc_overlay_range] = self.noc_overlay

        self.tdma = TDMA(self.tensix_coprocessor, self.L1_mem)
        tdma_range = AddressRange(0xFFB11000, self.tdma.getSize())
        tensix_mem_map[tdma_range] = self.tdma

        self.tile_ctrl = TensixTileControl()
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
        self.local_mem_brisc = DRAM(4096)
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
        )

        # Create ncrisc CPU
        self.local_mem_ncrisc = DRAM(4096)
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
        self.local_mem_trisc0 = DRAM(2048)
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
        )

        # Create trisc1 CPU
        self.local_mem_trisc1 = DRAM(2048)
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
        )

        # Create trisc2 CPU
        self.local_mem_trisc2 = DRAM(2048)
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
        )

        # Set addressable memory for Tensix co-processor
        self.tensix_coprocessor.setAddressableMemory([self.tensix_mem, self.ncrisc_mem])

        super().__init__(coord_x, coord_y, noc0_router, noc1_router)

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
