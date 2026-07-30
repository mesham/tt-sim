"""Socket-free replay of a captured tt-metal "four" wire trace on Blackhole.

``four`` is the first *compute* example: a reader (BRISC) streams two Int8 DRAM
vectors into circular buffers, the Tensix coprocessor adds them tile-by-tile
through the unpacker -> matrix (FPU) -> packer pipeline into an Int32 output CB,
and a writer (NCRISC) copies each result chunk back to DRAM. It exercises the
whole Tensix compute datapath that ``one``/``two``/``three`` (pure data-movement)
never touch: the STALLWAIT/SEMWAIT wait-gates, the Int8 elementwise add, and the
Blackhole pack path.

**Why this replay pumps until the go-message is DONE** — same reason as
``three``: the trace was captured from a run whose kernel finished on a different
(shorter) cycle count than the corrected one, so a plain replay reads the result
back before the compute has finished. On each go-message poll we pump until the
core flips the go-message to DONE (bounded), mirroring the live host's
``wait_until_cores_done``.

The Blackhole pack path needed three fixes for this to be bit-exact (all guarded
so Wormhole is unaffected): PACR ``read_intf_sel`` scaling the datum count so one
PACR spans four contiguous Dst rows, carrying the output address forward across
the PACRs of a tile, and initialising ``datastreamNeedsNewAddr`` so the very
first tile packs to the CB base rather than address 0. With those,
``dst[i] == src0[i] + src1[i]`` for all 256 elements.

Run:  python3 -m driver.blackhole.server.four_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "four.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 5000
PUMP_CAP = 400_000

# dst DRAM buffer: 256 int32 at channel (0, 11), offset 0x594080. The host fills
# src0[i] = i % 128 and src1[i] = (256 - i) % 128, so dst[i] = src0[i] + src1[i].
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594080
DATA_SIZE = 256
EXPECTED = [(i % 128) + ((DATA_SIZE - i) % 128) for i in range(DATA_SIZE)]


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

    wrong = [(i, v, EXPECTED[i]) for i, v in enumerate(values) if v != EXPECTED[i]]
    if wrong:
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"(Int8 elementwise add through the Tensix compute pipeline); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {wrong[0][2]}"
        )
    print(
        f"blackhole four_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"result elements == src0[i] + src1[i], Int8 add via the Tensix FPU)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
