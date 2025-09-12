from tt_metal import TT_Metal

from tt_sim.device.tt_device import Wormhole
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_int32,
    conv_to_uint32,
)

wormhole = Wormhole()

tt_metal = TT_Metal("tt_metal_0.62.2.json")
go_signal_start_addr, go_signal_byte_len = tt_metal.get_mailbox_config_details(
    "go_message", "signal"
)
run_msg_done = tt_metal.get_constant("RUN_MSG_DONE")
run_msg_go = tt_metal.get_constant("RUN_MSG_GO")
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

# Set the go signal and then loop round whilst this is not done, the firmware will
# set it to be done when it has finished and is ready for a kernel
wormhole.write(
    (18, 18), go_signal_start_addr, conv_to_bytes(run_msg_go, go_signal_byte_len)
)
while (
    conv_to_uint32(wormhole.read((18, 18), go_signal_start_addr, go_signal_byte_len))
    != run_msg_done
):
    wormhole.run(100)

tt_metal.load_kernel("one/parameters.json")

# Grab the data transfers needed to the device (includes the kernel binaries,
# mailbox values, go message, runtime arguments, cb configurations) and
# write them to the memory of the tensix core
data_transfers = tt_metal.generate_kernel_to_device_data_transfer_details()
for data_transfer in data_transfers:
    if not isinstance(data_transfer[2], bytes):
        d = conv_to_bytes(data_transfer[2], data_transfer[1])
    else:
        d = data_transfer[2]
    wormhole.write((18, 18), data_transfer[0], d, data_transfer[1])

## Write input data to DDR memory
list1 = list(range(100))
list2 = [100 - i for i in range(100)]
wormhole.write((16, 16), 0x20, conv_to_bytes(list1))
wormhole.write((16, 16), 0x1C0, conv_to_bytes(list2))

## Run cycles to process the kernel, continue until MSG_DONE is written
while (
    conv_to_uint32(wormhole.read((18, 18), go_signal_start_addr, go_signal_byte_len))
    != run_msg_done
):
    wormhole.run(100)

## Check results in DDR memory are correct
for i in range(100):
    val = conv_to_int32(wormhole.read((16, 16), 0x360 + (i * 4), 4))
    assert val == list1[i] + list2[i]
