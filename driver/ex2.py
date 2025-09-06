from tt_sim.memory.memory import DRAM
from tt_sim.device.memory_map import MemoryMap, AddressRange
from tt_sim.device.device_memory import DeviceMemory
from tt_sim.device.clock import Clock
from tt_sim.pe.rv.rv32 import RV32I
from tt_sim.util.conversion import conv_to_int32

# Read in binary executable (sets 10 to location 0x512)
with open("main.bin", 'rb') as file:
    data = file.read()    

# Create DRAM
dram=DRAM(4096)

# Create memory map and set DRAM into here
mem_map=MemoryMap()
dram_range=AddressRange(0x0, 4096)
mem_map[dram_range]=dram

# Create device memory and write executable into this
dm=DeviceMemory(mem_map, "1M")
dm.write(0x0, data)

# Create CPU
cpu=RV32I(dm, 0x0)

# Create the clock
clock=Clock([cpu])

# Run the clock for 100 iterations
clock.reset()
clock.run(100)

# Now check the result at location 0x512
rval=dram.read(0x512,4)
assert conv_to_int32(rval) == 10
