"""Socket-free replay of a captured tt-metal "eight" wire trace on Blackhole.

``eight`` is the same elementwise add as example one (``dst[i] = src0[i] +
src1[i]`` over 100 Int32s, so every result is ``i + (100 - i) == 100``), but it
runs **entirely on the BRISC reader** — there is no compute kernel or Tensix
coprocessor. Its purpose is the **NoC transaction-ID lifecycle**: the reader
issues the two source reads with *distinct* trids (1 and 2) and no intervening
barrier, then barriers on them **out of order** (``trid=2`` first, ``trid=1``
second). That exercises tt-sim's per-trid machinery in ``tt_sim/network/tt_noc.py``:
the per-trid FIFO of return addresses, the independent
``NIU_MST_REQS_OUTSTANDING_ID_<n>`` counters each ``noc_async_read_barrier_with_trid``
polls, and routing each response to the correct return buffer.

Pumps until the go-message flips to DONE, like the other replays.

Run:  python3 -m driver.blackhole.server.eight_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "eight.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# dst DRAM buffer: 100 int32 at channel (0, 11), offset 0x594200 (the third
# allocation, after the two 0x593e80/0x594080 source buffers). Every element is
# src0[i] + src1[i] = i + (100 - i) = 100.
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


def _go_signal(device, core):
    return device.tt_device.read(core, GO_MSG_ADDR, 4)[3]


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)

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
            if (
                parsed["cmd"] == proto.CMD_READ
                and parsed["core"] in TENSIX_POOL
                and parsed["address"] == GO_MSG_ADDR
                and _go_signal(device, parsed["core"]) == RUN_MSG_GO
            ):
                pumped = 0
                while (
                    _go_signal(device, parsed["core"]) != RUN_MSG_DONE
                    and pumped < PUMP_CAP
                ):
                    device.tt_device.run(PUMP_CHUNK)
                    pumped += PUMP_CHUNK

    result = device.read(DST_DRAM_COORD, DST_ADDR, DATA_SIZE * 4)
    values = [
        int.from_bytes(result[i : i + 4], "little") for i in range(0, len(result), 4)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v) for i, v in enumerate(values) if v != EXPECTED]
    if wrong:
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"(Int32 add on BRISC via out-of-order per-trid barriers); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {EXPECTED}"
        )
    print(
        f"blackhole eight_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"result elements == {EXPECTED}, out-of-order per-trid NoC barriers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
