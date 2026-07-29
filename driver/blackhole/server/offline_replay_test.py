"""Socket-free replay of a captured tt-metal "one" wire trace on Blackhole.

Like the Wormhole ``offline_replay_test``, but drives the Blackhole device +
coord maps directly (no nng socket) so it runs in restricted environments. The
trace (``traces/one.trace``) was captured from a real tt-metal ``one`` program
run against ``driver/blackhole`` — it covers full device init, kernel/firmware
load, and multi-core baby-RISCV execution.

The trace is a **complete, successful** run: device init, kernel/firmware load,
multi-core baby-RISCV execution, the compute result read back, and clean shutdown
(it ends with the host's EXIT). Every READ carries the reply the live server
produced, so a clean replay means the whole program — including the computed
result — reproduces bit-for-bit.

Run:  python3 -m driver.blackhole.server.offline_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "one.trace"
# The "one" program launches a single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]


def _build_fabric():
    device = make_device()
    fabric = Fabric()
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    for physical in TENSIX_POOL:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    return device, fabric


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)  # never connects; only _handle is used

    n_msgs = n_reads = mismatches = 0
    with TRACE.open() as f:
        for lineno, line in enumerate(f, 1):
            parsed = parse_trace_line(line)
            if parsed is None:
                continue
            req = SimpleNamespace(
                cmd=parsed["cmd"],
                core=parsed["core"],
                address=parsed["address"],
                size=parsed["size"],
                data=parsed["data"],
            )
            reply = transport._handle(fabric, req)
            n_msgs += 1
            if parsed["cmd"] != proto.CMD_READ or parsed["reply"] is None:
                continue
            n_reads += 1
            if bytes(reply) != bytes(parsed["reply"]):
                mismatches += 1
                if mismatches <= 5:
                    print(
                        f"  line {lineno} READ {parsed['core']} "
                        f"@0x{parsed['address']:x}: expected "
                        f"{parsed['reply'].hex()} got {bytes(reply).hex()}",
                        file=sys.stderr,
                    )

    device.tt_device.shutdown()
    if mismatches:
        raise AssertionError(
            f"{mismatches}/{n_reads} READ replies mismatched replaying {TRACE.name}"
        )
    print(
        f"blackhole offline_replay test OK ({n_msgs} messages, {n_reads} READs "
        f"reproduced bit-for-bit)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
