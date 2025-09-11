from tt_sim.device.clock import Clock
from tt_sim.device.device import Device
from tt_sim.device.reset import Reset
from tt_sim.memory.memory import DRAM, TensixMemory, TileMemory
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.misc.tile_ctrl import TensixTileControl
from tt_sim.network.tt_noc import NUI, NoCOverlay
from tt_sim.pe.pe import PEMemory
from tt_sim.pe.rv.babyriscv import BabyRISCV, BabyRISCVCoreType
from tt_sim.pe.tensix.tdma import TDMA
from tt_sim.pe.tensix.tensix import (
    TensixBackendConfiguration,
    TensixCoProcessor,
    TensixGPR,
)
from tt_sim.util.conversion import (
    clear_bit,
    conv_to_bytes,
    conv_to_int32,
)

"""
This is the first example of a Tensix core connected to DDR. Everything is
"hanging out" here to show how things are popped together, whereas in reality
we abstract much of this via "Tensix tile" and "DRAM tile" etc.

Here we create a tensix tile with the baby cores and all other aspects, launch
firmware and then launch the kernel to add pairs of numbers by BRISC. This involves
moving data via the NoC between the tensix and DDR tiles
"""

# Read in firmware binaries
with open("brisc_firmware_text.bin", "rb") as file:
    brisc_text = file.read()

with open("ncrisc_firmware_text.bin", "rb") as file:
    ncrisc_text = file.read()

with open("ncrisc_firmware_data.bin", "rb") as file:
    ncrisc_data = file.read()

with open("trisc0_firmware_text.bin", "rb") as file:
    trisc0_text = file.read()

with open("trisc1_firmware_text.bin", "rb") as file:
    trisc1_text = file.read()

with open("trisc2_firmware_text.bin", "rb") as file:
    trisc2_text = file.read()

dram_tile_mem_map = MemoryMap()

ddr_bank_0 = DRAM(10 * 1024 * 1024)
ddr_range = AddressRange(0x0, ddr_bank_0.getSize())
dram_tile_mem_map[ddr_range] = ddr_bank_0

dram_tile_mem = TileMemory(dram_tile_mem_map, "10M")

dram_noc0_router = NUI(0, 0, 0, dram_tile_mem)
dram_noc1_router = NUI(1, 9, 11, dram_tile_mem)

# Create tensix specific memory

tensix_mem_map = MemoryMap()

# Create DRAM
L1_mem = DRAM(1507327)
l1_range = AddressRange(0x0, L1_mem.getSize())
tensix_mem_map[l1_range] = L1_mem

tenxix_coprocessor = TensixCoProcessor()
tensix_range = AddressRange(0xFFE40000, tenxix_coprocessor.getSize())
tensix_mem_map[tensix_range] = tenxix_coprocessor

tenxix_coprocessor_be_config = TensixBackendConfiguration(tenxix_coprocessor)
tensix_config_range = AddressRange(0xFFEF0000, tenxix_coprocessor_be_config.getSize())
tensix_mem_map[tensix_config_range] = tenxix_coprocessor_be_config

tensix_gpr = TensixGPR(tenxix_coprocessor)
tensix_gpr_range = AddressRange(0xFFE00000, tensix_gpr.getSize())
tensix_mem_map[tensix_gpr_range] = tensix_gpr

noc0_router = NUI(0, 1, 1, L1_mem, True)
noc0_range = AddressRange(0xFFB20000, noc0_router.getSize())
tensix_mem_map[noc0_range] = noc0_router

noc1_router = NUI(1, 8, 10, L1_mem)
noc1_range = AddressRange(0xFFB30000, noc1_router.getSize())
tensix_mem_map[noc1_range] = noc1_router

noc_overlay = NoCOverlay()
noc_overlay_range = AddressRange(0xFFB40000, noc_overlay.getSize())
tensix_mem_map[noc_overlay_range] = noc_overlay

tdma = TDMA()
tdma_range = AddressRange(0xFFB11000, tdma.getSize())
tensix_mem_map[tdma_range] = tdma

tile_ctrl = TensixTileControl()
tile_ctrl_range = AddressRange(0xFFB12000, tile_ctrl.getSize())
tensix_mem_map[tile_ctrl_range] = tile_ctrl

# Create global PE memory space
tensix_mem = TensixMemory(tensix_mem_map, "10M")

# Create NoC directory
noc_0_directory = {
    dram_noc0_router.get_id_pair(): dram_noc0_router,
    noc0_router.get_id_pair(): noc0_router,
}
noc_1_directory = {
    dram_noc1_router.get_id_pair(): dram_noc1_router,
    noc1_router.get_id_pair(): noc1_router,
}

dram_noc0_router.set_noc_directory(noc_0_directory)
dram_noc1_router.set_noc_directory(noc_1_directory)
noc0_router.set_noc_directory(noc_0_directory)
noc1_router.set_noc_directory(noc_1_directory)

# Create brisc CPU
brisc0_mem_map = MemoryMap()
local_mem_brisc = DRAM(4096)
local_mem_brisc_range = AddressRange(0xFFB00000, local_mem_brisc.getSize())
brisc0_mem_map[local_mem_brisc_range] = local_mem_brisc
brisc0_mem = PEMemory(brisc0_mem_map, "1M")

brisc = BabyRISCV(BabyRISCVCoreType.BRISC, [tensix_mem, brisc0_mem], snoop=True)

# Create ncrisc CPU
ncrisc_mem_map = MemoryMap()
local_mem_ncrisc = DRAM(4096)
local_mem_ncrisc_range = AddressRange(0xFFB00000, local_mem_ncrisc.getSize())
ncrisc_mem_map[local_mem_ncrisc_range] = local_mem_ncrisc
ncrisc_mem = PEMemory(ncrisc_mem_map, "1M")

ncrisc = BabyRISCV(BabyRISCVCoreType.NCRISC, [tensix_mem, ncrisc_mem], snoop=False)

# Create trisc0 CPU
trisc0_mem_map = MemoryMap()
local_mem_trisc0 = DRAM(2048)
local_mem_trisc0_range = AddressRange(0xFFB00000, local_mem_trisc0.getSize())
trisc0_mem_map[local_mem_trisc0_range] = local_mem_trisc0
trisc0_mem = PEMemory(trisc0_mem_map, "1M")

trisc0 = BabyRISCV(BabyRISCVCoreType.TRISC0, [tensix_mem, trisc0_mem], snoop=False)

# Create trisc1 CPU
trisc1_mem_map = MemoryMap()
local_mem_trisc1 = DRAM(2048)
local_mem_trisc1_range = AddressRange(0xFFB00000, local_mem_trisc1.getSize())
trisc1_mem_map[local_mem_trisc1_range] = local_mem_trisc1
trisc1_mem = PEMemory(trisc1_mem_map, "1M")

trisc1 = BabyRISCV(BabyRISCVCoreType.TRISC1, [tensix_mem, trisc1_mem], snoop=False)

# Create trisc2 CPU
trisc2_mem_map = MemoryMap()
local_mem_trisc2 = DRAM(2048)
local_mem_trisc2_range = AddressRange(0xFFB00000, local_mem_trisc2.getSize())
trisc2_mem_map[local_mem_trisc2_range] = local_mem_trisc2
trisc2_mem = PEMemory(trisc2_mem_map, "1M")

trisc2 = BabyRISCV(BabyRISCVCoreType.TRISC2, [tensix_mem, trisc2_mem], snoop=False)

# Create a clock
clock = Clock(
    [
        brisc,
        ncrisc,
        trisc0,
        trisc1,
        trisc2,
        dram_noc0_router,
        dram_noc1_router,
        noc0_router,
        noc1_router,
    ]
)

# Create a reset which comprises the clock and CPUs
reset = Reset([clock, brisc, ncrisc, trisc0, trisc1, trisc2])

# Create a device
device = Device(None, [clock], [reset])

# Write executables into L1 memory
tensix_mem.write(0x3780, brisc_text)
tensix_mem.write(0x4D80, ncrisc_text)
tensix_mem.write(0x5580, trisc0_text)
tensix_mem.write(0x5B80, trisc1_text)
tensix_mem.write(0x6180, trisc2_text)

tensix_mem.write(
    0x8190, ncrisc_data
)  # Set the data area for ncrisc this is in L1 and then the core will grab this


# BRISC always starts at 0x0, as per Metalium, at 0x0 place instruction JAL 0x0, 0x3780 to jump
# to the firmware binary
tensix_mem.write(0x0, conv_to_bytes(0x7800306F))

soft_reset = clear_bit(0xFFFFFFFF, 11)
tensix_mem.write(0xFFB121B0, conv_to_bytes(soft_reset))

# Reset the device and run the clock for 5000 iterations
device.reset()
device.run(3100)

## Write mailbox configuration

# watcher_kernel_ids (uint16)
tensix_mem.write(0x20, conv_to_bytes(0x4, 2))
tensix_mem.write(0x22, conv_to_bytes(0x0, 2))
tensix_mem.write(0x24, conv_to_bytes(0x0, 2))
# ncrisc_kernel_size16 (uint16)
tensix_mem.write(0x26, conv_to_bytes(0x0, 2))
# kernel_config_base (uint32)
tensix_mem.write(0x28, conv_to_bytes(0x7190))
tensix_mem.write(0x2C, conv_to_bytes(0x3F520))
tensix_mem.write(0x30, conv_to_bytes(0x73D0))
# sem_offset (uint16)
tensix_mem.write(0x34, conv_to_bytes(0x20, 2))
tensix_mem.write(0x36, conv_to_bytes(0x0, 2))
tensix_mem.write(0x38, conv_to_bytes(0x0, 2))
# local_cb_offset (uint16)
tensix_mem.write(0x3A, conv_to_bytes(0x20, 2))
# remote_cb_offset (uint16)
tensix_mem.write(0x3C, conv_to_bytes(0x20, 2))
# rta_offset (3 by 2 x uint16)
tensix_mem.write(0x3E, conv_to_bytes(0x0, 2))
tensix_mem.write(0x40, conv_to_bytes(0x20, 2))
tensix_mem.write(0x42, conv_to_bytes(0x0, 2))
tensix_mem.write(0x44, conv_to_bytes(0x20, 2))
tensix_mem.write(0x46, conv_to_bytes(0x0, 2))
tensix_mem.write(0x48, conv_to_bytes(0x20, 2))
# mode (uint 8)
tensix_mem.write(0x4A, conv_to_bytes(0x1, 1))
# padding (uint 8) hence plus one
# kernel_text_offset (uint32)
tensix_mem.write(0x4C, conv_to_bytes(0x20))
tensix_mem.write(0x50, conv_to_bytes(0x0))
tensix_mem.write(0x54, conv_to_bytes(0x0))
tensix_mem.write(0x58, conv_to_bytes(0x0))
tensix_mem.write(0x5C, conv_to_bytes(0x0))
# local_cb_mask (uint32)
tensix_mem.write(0x60, conv_to_bytes(0x0))
# brisc_noc_id (uint8)
tensix_mem.write(0x64, conv_to_bytes(0x0, 1))
# brisc_noc_mode (uint8)
tensix_mem.write(0x65, conv_to_bytes(0x0, 1))
# min_remote_cb_start_index (uint8)
tensix_mem.write(0x66, conv_to_bytes(0x20, 1))
# exit_erisc_kernel (uint8)
tensix_mem.write(0x67, conv_to_bytes(0x0, 1))
# host_assigned_id (uint32)
tensix_mem.write(0x68, conv_to_bytes(0x0))
# sub_device_origin_x (uint8)
tensix_mem.write(0x6C, conv_to_bytes(0x0, 1))
# sub_device_origin_y (uint8)
tensix_mem.write(0x6D, conv_to_bytes(0x0, 1))
# enables (uint8)
tensix_mem.write(0x6E, conv_to_bytes(0x1, 1))
# preload (uint8)
tensix_mem.write(0x6F, conv_to_bytes(0x0, 1))

## Write go message configuration
tensix_mem.write(0x2A0, conv_to_bytes(0x50, 1))
tensix_mem.write(0x2A1, conv_to_bytes(0x73, 1))
tensix_mem.write(0x2A2, conv_to_bytes(0x2A, 1))
tensix_mem.write(0x2A3, conv_to_bytes(0x80, 1))

## Read in kernel binary and write this into L1
with open("brisc_kernel_text.bin", "rb") as file:
    brisc_kernel = file.read()
tensix_mem.write(0x71B0, brisc_kernel)

## Write runtime arguments
tensix_mem.write(0x7190, conv_to_bytes(0x20))
tensix_mem.write(0x7194, conv_to_bytes(0x1C0))
tensix_mem.write(0x7198, conv_to_bytes(0x360))
tensix_mem.write(0x719C, conv_to_bytes(0x16DE60))
tensix_mem.write(0x71A0, conv_to_bytes(0x16DCC0))
tensix_mem.write(0x71A4, conv_to_bytes(0x64))

## Write input data to DDR memory
list1 = list(range(100))
list2 = [100 - i for i in range(100)]
dram_tile_mem.write(0x20, conv_to_bytes(list1))
dram_tile_mem.write(0x1C0, conv_to_bytes(list2))

## Run 3000 cycles to process the kernel
device.run(3000)

## Check results in DDR memory are correct
for i in range(100):
    val = conv_to_int32(dram_tile_mem.read(0x360 + (i * 4), 4))
    assert val == list1[i] + list2[i]
