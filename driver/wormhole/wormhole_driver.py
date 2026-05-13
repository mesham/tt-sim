from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.trace import EventCategory, LifecycleEvent, Unit, get_bus, get_registry
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_uint32,
)

DEFAULT_TENSIX_COORD = (18, 18)

_HOST_UNIT_ID = (0, 0, 0, Unit.HOST.value)


def _emit_lifecycle(kind, detail=""):
    bus = get_bus()
    if not bus.is_enabled(EventCategory.LIFECYCLE):
        return
    # Register the host pseudo-unit lazily on first emit.
    registry = get_registry()
    registry.register(0, 0, 0, Unit.HOST)
    bus.publish(
        LifecycleEvent(
            cycle=0,
            unit_id=_HOST_UNIT_ID,
            kind=kind,
            detail=detail,
        )
    )


def launch_firmware(wormhole, tt_metal, coord=DEFAULT_TENSIX_COORD):
    print(f"--> Launching and running firmware on {coord}")
    _emit_lifecycle("firmware_launch_start", detail=str(coord))
    go_signal_start_addr, go_signal_byte_len = tt_metal.get_mailbox_config_details(
        "go_message", "signal"
    )
    run_msg_done = tt_metal.get_constant("RUN_MSG_DONE")
    run_msg_go = tt_metal.get_constant("RUN_MSG_GO")
    firmware_package = tt_metal.read_firmware("firmware")

    # TT Metal sets up a mapping table from DRAM bank to tile for each NoC, this is then
    # looked up when calling get_noc_addr_from_bank_id
    dram_to_noc_xy_transfers = tt_metal.generate_dram_noc_mapping_transfers()
    for data_transfer in dram_to_noc_xy_transfers:
        wormhole.write(
            coord,
            data_transfer[0],
            conv_to_bytes(data_transfer[2], data_transfer[1]),
            data_transfer[1],
        )

    # Write firmware binaries to correct locations in tensix
    for firmware in firmware_package:
        wormhole.write(coord, firmware.get_text_addr(), firmware.get_text_bin())
        if firmware.get_data_addr() is not None and firmware.get_data_bin() is not None:
            wormhole.write(coord, firmware.get_data_addr(), firmware.get_data_bin())

    # Set jal for BRISC to jump to its firmware from 0x0 start point
    wormhole.write(
        coord,
        tt_metal.get_config_value("l1_memory_map", "MEM_BOOT_CODE_BASE"),
        conv_to_bytes(0x7800306F),
    )

    # Scope the reset to the target tile so sibling tiles' running firmware is
    # left alone — important for multi-Tensix flows that bring up tiles serially.
    wormhole.assert_soft_reset(coord)
    wormhole.deassert_soft_reset(coord, BabyRISCVCoreType.BRISC)

    # Reset baby-core execution state on the target tile only. The generic
    # wormhole.reset() would clobber every tile's PCs, including siblings
    # already running firmware.
    wormhole.reset_tile(coord)

    # Set the go signal and then loop round whilst this is not done, the firmware will
    # set it to be done when it has finished and is ready for a kernel
    wormhole.write(
        coord, go_signal_start_addr, conv_to_bytes(run_msg_go, go_signal_byte_len)
    )
    while (
        conv_to_uint32(wormhole.read(coord, go_signal_start_addr, go_signal_byte_len))
        != run_msg_done
    ):
        wormhole.run(100)

    print("  --> Done, device is ready")
    _emit_lifecycle("firmware_launch_done", detail=str(coord))


def run_kernel(wormhole, tt_metal, parameters, coord=DEFAULT_TENSIX_COORD):
    print(f"--> Launching and running kernel on {coord}")
    _emit_lifecycle("kernel_start", detail=f"{coord} {parameters}")
    go_signal_start_addr, go_signal_byte_len = tt_metal.get_mailbox_config_details(
        "go_message", "signal"
    )
    run_msg_done = tt_metal.get_constant("RUN_MSG_DONE")

    tt_metal.load_kernel(parameters)

    # Grab the data transfers needed to the device (includes the kernel binaries,
    # mailbox values, go message, runtime arguments, cb configurations) and
    # write them to the memory of the tensix core
    data_transfers = tt_metal.generate_kernel_to_device_data_transfer_details()
    for data_transfer in data_transfers:
        if not isinstance(data_transfer[2], bytes):
            d = conv_to_bytes(data_transfer[2], data_transfer[1])
        else:
            d = data_transfer[2]
        wormhole.write(coord, data_transfer[0], d, data_transfer[1])

    ## Run cycles to process the kernel, continue until MSG_DONE is written
    while (
        conv_to_uint32(wormhole.read(coord, go_signal_start_addr, go_signal_byte_len))
        != run_msg_done
    ):
        wormhole.run(100)

    print("  --> Done")
    _emit_lifecycle("kernel_done", detail=str(coord))
