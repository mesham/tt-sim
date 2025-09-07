from tt_sim.device.clock import Clock
from tt_sim.device.device import Device
from tt_sim.memory.memory import DRAM
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.pe.pe import PEMemory
from tt_sim.pe.rv.rv32 import RV32IM_TT
from tt_sim.pe.tensix.tensix import TensixCoProcessor, TensixBackendConfiguration
from tt_sim.pe.tensix.tdma import TDMA
from tt_sim.network.tt_noc import NoCRouter

# First wormhole similar configuration, with single data mover core

# Read in binary executable (sets 10 to location 0x512)
with open("brisc_firmware.bin", "rb") as file:
    data = file.read()

mem_map = MemoryMap()

# Create DRAM
L1_mem = DRAM(1507327)
l1_range = AddressRange(0x0, L1_mem.getSize())
mem_map[l1_range] = L1_mem

local_mem = DRAM(4096)
local_range = AddressRange(0xFFB00000, local_mem.getSize())
mem_map[local_range] = local_mem

tenxix_coprocessor = TensixCoProcessor()
tensix_range = AddressRange(0xFFE40000, tenxix_coprocessor.getSize())
mem_map[tensix_range] = tenxix_coprocessor

tenxix_coprocessor_be_config = TensixBackendConfiguration(tenxix_coprocessor)
tensix_config_range = AddressRange(0xFFEF_0000, tenxix_coprocessor_be_config.getSize())
mem_map[tensix_config_range] = tenxix_coprocessor_be_config

noc1_router=NoCRouter(0, 0, 0)
noc2_range = AddressRange(0xFFB20000, noc1_router.getSize())
mem_map[noc2_range] = noc1_router

noc2_router=NoCRouter(1, 0, 0)
noc2_range = AddressRange(0xFFB30000, noc2_router.getSize())
mem_map[noc2_range] = noc2_router

tdma=TDMA()
tdma_range = AddressRange(0xFFB11000, tdma.getSize())
mem_map[tdma_range] = tdma

# Create PE memory space
pem = PEMemory(mem_map, "10M")

# Create CPU
cpu = RV32IM_TT(0x0, [pem], snoop=True)

# Create a clock
clock = Clock([cpu])

# Create a device
device = Device(pem, [clock], [cpu])

# Write executable into L1 memory
pem.write(0x0, data)

# Reset the device and run the clock for 5000 iterations
device.reset()
device.run(5000, print_cycle=True)
