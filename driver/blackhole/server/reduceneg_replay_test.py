"""Socket-free replay of the captured tt-metal "reduceneg" wire trace on Blackhole.

``reduceneg`` is ``reduce`` with the sign of every input flipped (see
``optests/reduceneg``), which is what makes it the guard for **Dst row-validity
bits**. ``reduce_tile`` clears its destination with ZEROACC and then accumulates
into it with GMPOOL. ZEROACC does not zero that row's data -- it clears the
row's *zero flag* -- and GMPOOL is the one Dst consumer that reads a
flag-cleared row as all-ones, i.e. minus infinity. Read it as +0 instead and
every MAX reduction over this all-negative tile saturates at zero, which is
exactly what tt-sim did before the flags were modelled (``DstRegister``'s
``dstRowValid``). ``optests/reduce`` cannot see the bug: its inputs are all
non-negative, so +0 and minus infinity pick the same winner.

The op sequence is ``reduce``'s: MAX/{COL,ROW,SCALAR} and SUM/{COL,SCALAR} at
``MathFidelity::HiFi4``, one output tile each. As there, the golden is ttsim's
own dump (``traces/reduceneg.expected``) rather than a closed form -- the SUM
reductions' answers are whatever the fidelity phases and Dst rounding produce.
All 5120 bfloat16 elements must match bit-exactly.

Run:  python3 -m driver.blackhole.server.reduceneg_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "reduceneg.trace"
EXPECTED_DUMP = Path(__file__).resolve().parent / "traces" / "reduceneg.expected"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS bfloat16 at channel (0, 11), offset
# 0x594e80 (the third allocation, after the data and scaler tiles). Written by
# the BRISC writer, one 32x32 tile per reduction, contiguous.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594E80
TILE_ELEMS = 1024
NUM_OPS = 5
DATA_SIZE = NUM_OPS * TILE_ELEMS
OP_NAMES = ["MAX/COL", "MAX/ROW", "MAX/SCALAR", "SUM/COL", "SUM/SCALAR"]


def _load_expected():
    """ttsim's dump of the same program, as one contiguous hex string."""
    text = EXPECTED_DUMP.read_text().strip()
    assert len(text) == DATA_SIZE * 4, (
        f"{EXPECTED_DUMP.name} holds {len(text) // 4} elements, expected {DATA_SIZE}"
    )
    return [int(text[i : i + 4], 16) for i in range(0, len(text), 4)]


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

    expected = _load_expected()
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
            # Pump the worker until its go-message reports DONE (bounded).
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

    result = device.read(DST_DRAM_COORD, DST_ADDR, DATA_SIZE * 2)
    values = [
        int.from_bytes(result[i : i + 2], "little") for i in range(0, len(result), 2)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v, e) for i, (v, e) in enumerate(zip(values, expected)) if v != e]
    if wrong:
        i, got, exp = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements differ from the ttsim golden "
            f"replaying {TRACE.name} ({OP_NAMES[i // TILE_ELEMS]} datum "
            f"{i % TILE_ELEMS}): dst[{i}]={got:#06x} expected {exp:#06x}"
        )
    print(
        f"blackhole reduceneg_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"bfloat16 results across {NUM_OPS} reductions over an all-negative tile "
        f"({', '.join(OP_NAMES)}) bit-exact against the ttsim golden)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
