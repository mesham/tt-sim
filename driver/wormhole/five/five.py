from driver.wormhole.tt_metal import TT_Metal
from driver.wormhole.wormhole_driver import launch_firmware, run_kernel
from tt_sim.device.tt_device import DeviceTileDiagnostics, Wormhole
from tt_sim.pe.tensix.util import TensixCoprocessorDiagnostics
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_int32,
)

# These diagnostics are optional to the Wormhole (if omitted then all
# assumed to be off). Put here explicitly so can see how can turn specific
# reporting on and off

coprocessor_diagnostics = TensixCoprocessorDiagnostics(
    unpacking=False,
    packing=False,
    configurations_set=False,
    issued_instructions=False,
    fpu_calculations=False,
    sfpu_calculations=False,
    thcon=False,
)

tile_diags = DeviceTileDiagnostics(
    brisc_diagnostics=False,
    ncrisc_diagnostics=False,
    trisc0_diagnostics=False,
    trisc1_diagnostics=False,
    trisc2_diagnostics=False,
    noc0_diagnostics=False,
    noc1_diagnostics=False,
    coprocessor_diagnostics=coprocessor_diagnostics,
)


wormhole = Wormhole(tile_diags)
tt_metal = TT_Metal("tt_metal_0.62.2.json")
launch_firmware(wormhole, tt_metal)

list1 = [i for i in range(256)]
list2 = [(256 - i) for i in range(256)]
wormhole.write((16, 16), 0x20, conv_to_bytes(list1, 4))
wormhole.write((16, 16), 0x420, conv_to_bytes(list2, 4))

run_kernel(wormhole, tt_metal, "five/parameters.json")

## Check results in DDR memory are correct
for i in range(256):
    val = conv_to_int32(wormhole.read((16, 16), 0x820 + (i * 4), 4))
    assert val == list1[i] + list2[i]

print("Example five completed successfully")
