from tt_sim.device.device import DeviceMemory
from tt_sim.device.memory_map import AddressRange, MemoryMap
from tt_sim.memory.memory import DRAM
from tt_sim.util.conversion import conv_to_bytes, conv_to_int32

dram = DRAM(1024)

val = 25
by = conv_to_bytes(val)
dram.write(0, by)

rval = dram.read(0, 4)
assert conv_to_int32(rval) == 25

vals = [1, 2, 3, 4]
by = conv_to_bytes(vals)
dram.write(0x16, by)

for idx, in_val in enumerate(vals):
    rval = dram.read(0x16 + (idx * 4), 4)
    assert conv_to_int32(rval) == in_val

# Now we have tested simple memory, test as part of a memory map

dram2 = DRAM(1024)

mem_map = MemoryMap()

dram_range = AddressRange(0x8192, 1024)
mem_map[dram_range] = dram

dram_range = AddressRange(0x16384, 1024)
mem_map[dram_range] = dram2

dm = DeviceMemory(mem_map, "2G")

dm.write(0x8192, conv_to_bytes(100))
rv = dm.read(0x8192, 4)
assert conv_to_int32(rv) == 100
# Test the value is in dram directly
rv = dram.read(0, 4)
assert conv_to_int32(rv) == 100

# Test adding a lit of values to the second dram area
vals = [10, 20, 30, 40]
by = conv_to_bytes(vals)
dm.write(0x16392, by)

for idx, in_val in enumerate(vals):
    rval = dm.read(0x16392 + (idx * 4), 4)
    # is 0xe here for dram2 the address above is hex
    rval2 = dram2.read(0xE + (idx * 4), 4)
    assert conv_to_int32(rval) == in_val
    assert conv_to_int32(rval2) == in_val
