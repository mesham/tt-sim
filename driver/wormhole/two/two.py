from driver.wormhole.tt_metal import TT_Metal
from driver.wormhole.wormhole_driver import launch_firmware, run_kernel
from tt_sim.device.tt_device import Wormhole
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_int32,
)

wormhole = Wormhole()
tt_metal = TT_Metal("tt_metal_0.62.2.json")
launch_firmware(wormhole, tt_metal)

## Write input data to DDR memory
list1 = list(range(100))
list2 = [100 - i for i in range(100)]
wormhole.write((16, 16), 0x20, conv_to_bytes(list1))
wormhole.write((16, 16), 0x1C0, conv_to_bytes(list2))

run_kernel(wormhole, tt_metal, "two/parameters.json")

## Check results in DDR memory are correct
for i in range(100):
    val = conv_to_int32(wormhole.read((16, 16), 0x360 + (i * 4), 4))
    assert val == list1[i] + list2[i]
