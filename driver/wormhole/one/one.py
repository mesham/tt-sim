import os

from driver.wormhole.tt_metal import TT_Metal
from driver.wormhole.wormhole_driver import launch_firmware, run_kernel
from tt_sim.device.tt_device import DeviceTileDiagnostics, Wormhole
from tt_sim.pe.tensix.util import TensixCoprocessorDiagnostics
from tt_sim.trace import JSONLLogger, get_bus, get_registry
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

# Optional structured tracing — set TT_SIM_TRACE=path/to/trace.jsonl to enable.
trace_path = os.environ.get("TT_SIM_TRACE")
trace_logger = None
if trace_path:
    bus = get_bus()
    bus.enabled = True
    trace_logger = JSONLLogger(trace_path)

tt_metal = TT_Metal("tt_metal_0.62.2.json")
launch_firmware(wormhole, tt_metal)

## Write input data to DDR memory
list1 = list(range(100))
list2 = [100 - i for i in range(100)]
wormhole.write((16, 16), 0x20, conv_to_bytes(list1))
wormhole.write((16, 16), 0x1C0, conv_to_bytes(list2))

run_kernel(wormhole, tt_metal, "one/parameters.json")

## Check results in DDR memory are correct
for i in range(100):
    val = conv_to_int32(wormhole.read((16, 16), 0x360 + (i * 4), 4))
    assert val == list1[i] + list2[i]

if trace_logger is not None:
    trace_logger.close()
    get_registry().dump(trace_path + ".ids.json")

print("Example one completed successfully")
