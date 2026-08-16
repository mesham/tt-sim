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
- ``cores`` — DRAM / eth / Tensix / deferred / null endpoints over the device.
- ``materialise`` — building exactly the workers a program launches on.
- ``trace`` — record/replay of wire conversations.
- ``device`` — the cycle-pumping ``Device`` wrapper + diagnostics-from-env.
- ``hostlink`` — ending a host the simulator can no longer answer.
"""

from tt_sim.bridge.cores import (
    DeferredTensixCore,
    DramCore,
    EthCore,
    NullCore,
    TensixCore,
)
from tt_sim.bridge.device import (
    Device,
    diagnostics_from_env,
    enabled_diagnostic_names,
    link_contention_summary,
    profiler_flush_summary,
)
from tt_sim.bridge.fabric import (
    Fabric,
    install_convention_guard,
    install_worker_guards,
)
from tt_sim.bridge.grid import compute_grid, fill_order
from tt_sim.bridge.hostlink import find_wire_peer, host_not_stranded, stop_host
from tt_sim.bridge.materialise import LazyTensixPool
from tt_sim.bridge.trace import TraceWriter, parse_trace_line
from tt_sim.bridge.transport import Transport

__all__ = [
    "DeferredTensixCore",
    "Device",
    "DramCore",
    "EthCore",
    "Fabric",
    "LazyTensixPool",
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
    "install_convention_guard",
    "install_worker_guards",
    "link_contention_summary",
    "parse_trace_line",
    "profiler_flush_summary",
    "stop_host",
]
