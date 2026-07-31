"""Socket-free replay of a captured tt-metal "nine" wire trace on Blackhole.

``nine`` is the first **two-tile** example. It splits example four's int8
elementwise add across two Tensix workers connected by a NoC-bridged circular
buffer:

* Tile A ``(1, 2)`` runs the reader (BRISC), the compute (TRISC int8 add, packed
  to int32), and the sender (NCRISC). The sender ``noc_async_write``s each output
  chunk into Tile B's L1 receive buffer and ``noc_semaphore_inc``s a semaphore
  on Tile B.
* Tile B ``(2, 2)`` runs the writer (BRISC), which ``noc_semaphore_wait_min``s
  for chunk ``i`` then forwards it from its L1 receive buffer to DRAM.

So it exercises the multi-Tensix path end-to-end: two independent tiles (L1 /
coprocessor / mailbox), cross-tile NoC routing ``(1,2) -> (2,2)``, and a
cross-tile semaphore. Result: ``dst[i] = src0[i] + src1[i] = (i % 128) +
((256 - i) % 128)`` for all 256 elements.

Both tiles are launched, so we pump whichever tile's go-message is still GO
until both report DONE.

Run:  python3 -m driver.blackhole.server.nine_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "nine.trace"
# Producer (reader+compute+sender) and consumer (writer) tiles.
TENSIX_POOL = [(1, 2), (2, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# dst DRAM buffer: 256 int32 at channel (0, 11), offset 0x594080, written by the
# Tile B writer. Every element is src0[i] + src1[i] = (i % 128) + ((256-i) % 128).
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594080
DATA_SIZE = 256
EXPECTED = [(i % 128) + ((256 - i) % 128) for i in range(DATA_SIZE)]


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
            # Each launched tile flips its own go-message; pump the one being
            # polled until it reports DONE (bounded).
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

    wrong = [(i, v, e) for i, (v, e) in enumerate(zip(values, EXPECTED)) if v != e]
    if wrong:
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"(two-tile int8 add bridged over NoC); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {wrong[0][2]}"
        )
    print(
        f"blackhole nine_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"result elements == src0[i] + src1[i], two Tensix tiles bridged over NoC "
        f"with a cross-tile semaphore)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
