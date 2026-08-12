"""Architecture-agnostic tt-metal wire bridge.

UMD's "simulation" chip backend spawns a process that speaks tt-metal's wire
protocol over an nng socket; these modules are that process's machinery, factored
out of the Wormhole driver so both `driver/wormhole` and `driver/blackhole` share
one implementation. Nothing here is architecture-specific — a driver injects its
device factory + coordinate maps. See ``docs/plans/blackhole-support.md``.

Modules:
- ``protocol`` / ``_flatbuf`` — the tt-metal wire message format.
- ``transport`` — the nng pair1 socket + dispatch loop.
- ``fabric`` — routes a (coord, addr) op to the right ``cores`` wrapper.
- ``grid`` — tt-metal's compute grid and the order it fills workers in.
- ``cores`` — DRAM / eth / Tensix / null endpoints over the device.
- ``trace`` — record/replay of wire conversations.
- ``device`` — the cycle-pumping ``Device`` wrapper + diagnostics-from-env.
- ``hostlink`` — ending a host the simulator can no longer answer.
"""

from tt_sim.bridge.cores import DramCore, EthCore, NullCore, TensixCore
from tt_sim.bridge.device import (
    Device,
    diagnostics_from_env,
    enabled_diagnostic_names,
)
from tt_sim.bridge.fabric import Fabric, install_worker_guards
from tt_sim.bridge.grid import compute_grid, fill_order
from tt_sim.bridge.hostlink import find_wire_peer, host_not_stranded, stop_host
from tt_sim.bridge.trace import TraceWriter, parse_trace_line
from tt_sim.bridge.transport import Transport

__all__ = [
    "Device",
    "DramCore",
    "EthCore",
    "Fabric",
    "NullCore",
    "TensixCore",
    "TraceWriter",
    "Transport",
    "compute_grid",
    "diagnostics_from_env",
    "enabled_diagnostic_names",
    "fill_order",
    "find_wire_peer",
    "host_not_stranded",
    "install_worker_guards",
    "parse_trace_line",
    "stop_host",
]
