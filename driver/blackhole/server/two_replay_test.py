"""Socket-free replay of a captured tt-metal "two" wire trace on Blackhole.

``two`` is the first *multi-processor* example: a reader on RISCV_0 (BRISC) adds
two DRAM vectors element-wise into a circular buffer, and a writer on RISCV_1
(NCRISC) drains that CB back to a third DRAM buffer over NoC 1. It exercises
three things ``one`` (BRISC-only) never did on Blackhole:

  * NCRISC booting from its Blackhole firmware base (0x5BE0, not the Wormhole
    0x12000) and running a real kernel;
  * the launch message's ``enables`` bitmask deciding which subordinate RISCs a
    wire DEASSERT releases (here BRISC + NCRISC);
  * a NoC 1 write to DRAM, which routes to the channel's *NoC 1* endpoint
    (physical (0, 1), mirror (16, 10)) — a different subchannel than the NoC 0
    endpoint (0, 11) the reader used.

Unlike ``one``'s bit-for-bit test, the captured trace here predates the fix, so
its recorded result read-back is the buggy all-zeros. We therefore replay for
side effects and assert the *computed* result directly: ``dst[i]`` must equal
``src0[i] + src1[i]`` = 100 for every element.

Run:  python3 -m driver.blackhole.server.two_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "two.trace"
# The "two" program launches a single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]
# Host runtime layout for this capture: the dst DRAM buffer (100 uint32 = 400
# bytes) lives at DRAM channel 0, offset 0x594200; every element must be 100.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594200
DATA_SIZE = 100
EXPECTED = 100


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

    n_msgs = 0
    with TRACE.open() as f:
        for line in f:
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
            transport._handle(fabric, req)
            n_msgs += 1

    result = device.read(DST_DRAM_COORD, DST_ADDR, DATA_SIZE * 4)
    values = [
        int.from_bytes(result[i : i + 4], "little") for i in range(0, len(result), 4)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v) for i, v in enumerate(values) if v != EXPECTED]
    if wrong:
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"(reader-add on BRISC + writer on NCRISC over NoC 1); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {EXPECTED}"
        )
    print(
        f"blackhole two_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"result elements == {EXPECTED}, computed by BRISC reader + NCRISC writer)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
