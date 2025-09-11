from tt_metal import TT_Metal

from tt_sim.device.tt_device import Wormhole
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.util.conversion import (
    conv_to_bytes,
)

wormhole = Wormhole()

tt_metal = TT_Metal("tt_metal_0.62.2.json")
firmware_package = tt_metal.read_firmware("firmware")

# Write firmware binaries to correct locations in tensix
for firmware in firmware_package:
    wormhole.write((18, 18), firmware.get_text_addr(), firmware.get_text_bin())
    if firmware.get_data_addr() is not None and firmware.get_data_bin() is not None:
        wormhole.write((18, 18), firmware.get_data_addr(), firmware.get_data_bin())

# Set jal for BRISC to jump to its firmware from 0x0 start point
wormhole.write(
    (18, 18),
    tt_metal.get_config_value("l1_memory_map", "MEM_BOOT_CODE_BASE"),
    conv_to_bytes(0x7800306F),
)

wormhole.assert_soft_reset()
wormhole.deassert_soft_reset((18, 18), BabyRISCVCoreType.BRISC)

wormhole.reset()
wormhole.run(3100)
