from tt_sim.device.clock import Clock
from tt_sim.device.device import Device, DeviceMemory
from tt_sim.memory.memory import DRAM
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.pe.pe import PEMemory
from tt_sim.pe.rv.rv32 import RV32IM
from tt_sim.util.conversion import conv_to_int32

# This is very similar to ex4, but uses two separate memory spaces, a device memory space
# and PE local memory space for RAM. Therefore the PE will have a globally visible (device)
# space and a private local space

# Read in binary executable (sets 10 to location 0x512)
with open("main.bin", "rb") as file:
    data = file.read()

# Create DRAM
dram_rom = DRAM(16384)
dram_ram = DRAM(8196)

# Create memory map and set DRAM into here
device_mem_map = MemoryMap()
rom_range = AddressRange(0x0, 16384)
device_mem_map[rom_range] = dram_rom

# Create a PE memory map
pe_mem_map = MemoryMap()
ram_range = AddressRange(0x80000000, 8196)
pe_mem_map[ram_range] = dram_ram

# Create device memory and write executable into this
dm = DeviceMemory(device_mem_map, "2G")
dm.write(0x0, data)

# Create PE memory space
pem = PEMemory(pe_mem_map, "1M")

# Create CPU
cpu = RV32IM(0x0, [dm, pem])

# Create a clock
clock = Clock([cpu])

# Create a device
device = Device(dm, [clock], [cpu])

# Reset the device and run the clock for 5000 iterations
device.reset()
device.run(5000)

# Now check the results that have been stored
for i in range(10):
    rval = dram_ram.read(0x512 + (i * 4), 4)
    assert conv_to_int32(rval) == (i * 100) / 2
