"""Socket-free replay of the captured tt-metal "reduce" wire trace on Wormhole.

The Wormhole twin of ``driver/blackhole/server/reduce_replay_test.py``. Like
``sfpumath``, and unlike the example-derived Wormhole traces (which replay their
recorded READ replies), this checks *values* against the vendor reference sim:
``optests/reduce`` exists precisely to be diffed against ttsim (see
``optests/diff.sh``, which takes ``TT_SIM_ARCH=wormhole``).

One bfloat16 data tile plus the standard reduce-scaler tile are run through five
(PoolType, ReduceDim) combinations at ``MathFidelity::HiFi4``, one output tile
each -- MAX/{COL,ROW,SCALAR} and SUM/{COL,SCALAR}. Between them they exercise
most of the reduction pipeline: GMPOOL and the multi-phase high-fidelity GAPOOL
MOP; the unpacker's haloize transpose (how a row reduce becomes a column-wise
pool); the MOVD2B / TRNSPSRCB / MOVB2A dance that turns a pooled row back into a
column through SrcB's rows 16-31 scratch; and the packer's reduce edge masks,
which are the only thing zeroing the Dst rows the reduction did not write.

There is no closed-form golden -- the answer is what the hardware's fidelity
phases and Dst rounding produce -- so the golden is ttsim-Wormhole's own dump,
frozen from a passing ``TT_SIM_ARCH=wormhole ./optests/diff.sh reduce`` run in
``traces/reduce.expected`` so it runs with no oracle present. All 5120 bfloat16
elements must match bit-exactly. (``diff.sh`` reports 2560: it assumes 8 hex
chars per element, which halves the count for a bfloat16 program.)

Run:  python3 -m driver.wormhole.server.reduce_replay_test
      (or under pytest, as ``test_reduce_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "reduce.trace"
EXPECTED_DUMP = Path(__file__).resolve().parent / "traces" / "reduce.expected"
# Single worker at Wormhole physical coord (1, 1) — logical (0, 0).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS bfloat16 written by the BRISC writer,
# one 32x32 tile per reduction, contiguous, at the third allocation in the
# channel behind physical (0, 11).
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D5C40
TILE_ELEMS = 1024
NUM_OPS = 5
DATA_SIZE = NUM_OPS * TILE_ELEMS
OP_NAMES = ["MAX/COL", "MAX/ROW", "MAX/SCALAR", "SUM/COL", "SUM/SCALAR"]


def _load_expected():
    """ttsim-Wormhole's dump of the same program, as one contiguous hex string."""
    text = EXPECTED_DUMP.read_text().strip()
    assert len(text) == DATA_SIZE * 4, (
        f"{EXPECTED_DUMP.name} holds {len(text) // 4} elements, expected {DATA_SIZE}"
    )
    return [int(text[i : i + 4], 16) for i in range(0, len(text), 4)]


def _build_fabric():
    device = make_device()
    fabric = Fabric()
    for translated, unified in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, unified))
    for translated, unified in ETH_COORD_MAP.items():
        fabric.register(translated, EthCore(device, unified))
    for physical in TENSIX_POOL:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    return device, fabric


def _go_signal(device, core):
    return device.tt_device.read(TENSIX_COORD_MAP[core], GO_MSG_ADDR, 4)[3]


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

    result = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, DATA_SIZE * 2)
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
        f"wormhole reduce_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"bfloat16 results across {NUM_OPS} reductions "
        f"({', '.join(OP_NAMES)}) bit-exact against the ttsim-Wormhole golden)"
    )
    return 0


def test_reduce_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
