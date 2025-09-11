from tt_metal import TT_Metal

from tt_sim.device.tt_device import Wormhole
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_int32,
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

tt_metal.load_kernel("one/parameters.json")

kernel_binaries = tt_metal.get_transfer_kernel_binary_details()
for kernel_bin in kernel_binaries:
    wormhole.write((18, 18), kernel_bin[0], kernel_bin[2], kernel_bin[1])

launch_mailbox = tt_metal.generate_transfer_mailbox_details("launch")
for mailbox_data in launch_mailbox:
    wormhole.write(
        (18, 18), mailbox_data[0], conv_to_bytes(mailbox_data[2], mailbox_data[1])
    )

go_message = tt_metal.generate_transfer_mailbox_details("go_message")
for go_message_component in go_message:
    wormhole.write(
        (18, 18),
        go_message_component[0],
        conv_to_bytes(go_message_component[2], go_message_component[1]),
    )

rt_args = tt_metal.generate_transfer_runtime_arguments_details()
for rt_arg in rt_args:
    wormhole.write((18, 18), rt_arg[0], conv_to_bytes(rt_arg[2], rt_arg[1]))

## Write input data to DDR memory
list1 = list(range(100))
list2 = [100 - i for i in range(100)]
wormhole.write((16, 16), 0x20, conv_to_bytes(list1))
wormhole.write((16, 16), 0x1C0, conv_to_bytes(list2))

## Run 3000 cycles to process the kernel
wormhole.run(3000)

## Check results in DDR memory are correct
for i in range(100):
    val = conv_to_int32(wormhole.read((16, 16), 0x360 + (i * 4), 4))
    assert val == list1[i] + list2[i]
