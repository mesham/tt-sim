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
- ``cores`` — DRAM / eth / Tensix / null endpoints over the device.
- ``trace`` — record/replay of wire conversations.
- ``device`` — the cycle-pumping ``Device`` wrapper + diagnostics-from-env.
"""

from tt_sim.bridge.cores import DramCore, EthCore, NullCore, TensixCore
from tt_sim.bridge.device import (
    Device,
    diagnostics_from_env,
    enabled_diagnostic_names,
)
from tt_sim.bridge.fabric import Fabric, install_worker_guards
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
    "diagnostics_from_env",
    "enabled_diagnostic_names",
    "install_worker_guards",
    "parse_trace_line",
]
