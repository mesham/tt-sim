"""Socket-free replay of a captured tt-metal "five" wire trace on Blackhole.

``five`` is the first example on the **SFPU / vector engine** rather than the
matrix FPU: the compute kernel does `copy_tile` (×2, unpacking each Int32 input
tile straight into DEST via Blackhole's unpack-to-dest path) then
`add_int_tile<Int32>` (the SFPU integer add) then `pack_tile`. It exercises the
whole vector-unit datapath and, critically, the cross-thread hazard that
unpack-to-dest introduces.

The fix this regression guards: `copy_tile`'s MATH-side datacopy issues a
Blackhole HW-bug workaround — a `ZEROACC` with the `clear_zero_flags` bit set,
right after unpack-to-dest — which must only re-assert DEST zero-flags, **not**
clear the data. tt-sim emulates zero-flags by zeroing the data, so it was wiping
the freshly-unpacked operands and the SFPU added `0 + src1` (even chunks came out
as `src1` alone → 127/256 wrong). Honouring `clear_zero_flags` (raw bit 17) on
Blackhole fixes it. See `handle_zeroacc` and
`tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h`.

Like the other compute replays this pumps until the go-message is DONE. The host
fills `src0[i] = i` and `src1[i] = 256 - i`, so every output element is `256`.

Run:  python3 -m driver.blackhole.server.five_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "five.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 5000
PUMP_CAP = 400_000

# dst DRAM buffer: 256 int32 at channel (0, 11), offset 0x594680. The host fills
# src0[i] = i and src1[i] = 256 - i, so dst[i] = 256 for all i.
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
            f"(Int32 elementwise add through the SFPU / vector engine); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {EXPECTED}"
        )
    print(
        f"blackhole five_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"result elements == {EXPECTED}, Int32 add via the SFPU)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
