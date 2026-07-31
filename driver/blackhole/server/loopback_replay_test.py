"""Socket-free replay of a captured tt-metal "loopback" wire trace on Blackhole.

``loopback`` copies a 256-element Int32 vector DRAM -> DRAM through the full
Tensix pipeline: the reader (BRISC) streams 64-element chunks into a single-page
Int32 CB, the compute kernel ``copy_tile``s each chunk into **Dst register
segment 2** (via the SFPU unary path) and ``pack_tile``s it back out, and the
writer (NCRISC) stores each chunk to the output buffer. Four chunks of 64 cover
the 256 elements; the result must be an exact copy, ``dst[i] == i``.

It is the first example to pack from a *non-zero* Dst segment (segment 2), so it
guards that pack-source offset alongside the chunked SFPU copy path. The buffers
are single-page (one DRAM bank), so the interleaved-DRAM machinery `six` needs
is not exercised here.

Pumps until the go-message flips to DONE, like the other replays, so the kernel
runs to completion regardless of the captured cycle count.

Run:  python3 -m driver.blackhole.server.loopback_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "loopback.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# dst DRAM buffer: 256 int32 at channel (0, 11), offset 0x594280 (the second
# allocation, after the 0x593e80 source buffer). Result is an exact copy of the
# 0..255 source, so dst[i] == i.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594280
DATA_SIZE = 256
EXPECTED = list(range(DATA_SIZE))


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

    wrong = [(i, v) for i, (v, e) in enumerate(zip(values, EXPECTED)) if v != e]
    if wrong:
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"(Int32 DRAM->DRAM copy via Dst segment 2); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {wrong[0][0]}"
        )
    print(
        f"blackhole loopback_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"Int32 elements copied DRAM->DRAM, dst[i] == i)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
