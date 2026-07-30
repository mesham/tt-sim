"""Socket-free replay of a captured tt-metal "three" wire trace on Blackhole.

``three`` is like ``two`` (reader on BRISC + writer on NCRISC, adding two DRAM
vectors and writing the sum back over NoC 1) but chunked: it loops ``num_chunks``
times, moving one ``CHUNK_SIZE``-element tile per iteration through a single-page
circular buffer. It exercises the multi-chunk CB producer/consumer handshake
(via the NoC-overlay stream counters) that ``two``'s single push/pop never hits.

**Why this replay pumps until the go-message is DONE.** The captured trace comes
from a run whose (buggy, pre-Zbb) inner loop truncated to 16 of 64 elements and
so finished in far fewer cycles than the correct 64-element loop now takes. A
plain replay feeds that frozen, too-short poll sequence and reads the result
back before the (correct, slower) kernel has finished, yielding a partial result
— purely a replay-timing artifact, not a device bug (pumping extra cycles before
teardown yields the full 256/256). The live server never has this problem: its
``wait_until_cores_done`` polls the go-message and pumps until the firmware flips
it to DONE. This replay mirrors that: on each go-message poll it pumps until the
core signals DONE (capped), so the kernel runs to completion exactly as it would
live. With that, the reader adds all 256 elements and the writer stores all four
chunks: ``dst[i]`` must equal ``src0[i] + src1[i]`` = 256 for every element.

Run:  python3 -m driver.blackhole.server.three_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "three.trace"
TENSIX_POOL = [(1, 2)]

# go_msg_t at L1 0x4f0; signal is the top byte. The host writes GO to launch and
# spins until the firmware writes DONE (see tt_metal dev_msgs.h RUN_MSG_*).
GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
# Pump budget per poll while a core is running (chunk size × safety margin; the
# kernel finishes in ~20k cycles). Bounded so a genuine hang fails, not spins.
PUMP_CHUNK = 5000
PUMP_CAP = 400_000

# dst DRAM buffer: 256 uint32 at channel 0, offset 0x594680; every element = 256.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594680
DATA_SIZE = 256
EXPECTED = 256


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
            # Mirror the live host: while a launched core reports GO, keep
            # pumping until it flips the go-message to DONE (bounded).
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
            f"(chunked reader+writer through a 1-page CB); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {EXPECTED}"
        )
    print(
        f"blackhole three_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"result elements == {EXPECTED}, chunked via the circular buffer)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
