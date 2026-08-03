"""Socket-free replay of the captured tt-metal "where" wire trace on Blackhole.

``where`` (see ``optests/where``) is the op test for ``where_tile`` -- the
ternary ``cond ? a : b`` select -- on Int32 tiles. Three input tiles stay
resident; each op copies the condition into DST[0], the two value tiles into
DST[1] and DST[2], selects into DST[2] and packs that out (the in-place,
output-over-idst2 form ttnn's own kernels use). The two ops swap the value
operands, so the two output tiles are elementwise complements of one another and
between them every lane takes both branches:

* op0  ``cond ? a : b``
* op1  ``cond ? b : a``

with ``cond[i] = 0`` for ``i % 3 == 0``, ``i + 1`` for ``i % 3 == 1`` and
``-(i + 1)`` otherwise (so the false branch, positive-true and negative-true all
occur inside every face), ``a[i] = 0x11110000 + i`` and ``b[i] = 0x22220000 + i``
-- distinct values, so a lane reading the wrong operand is unambiguous.

On Blackhole ``where_tile`` normally lowers to SFPLOADMACRO, which *both*
simulators decline as out of scope; ``optests/where/env`` sets
TT_METAL_DISABLE_SFPLOADMACRO=1 so what the captured run (and therefore this
guard) covers is the LLK's non-macro SFPU select sequence.

The select is exact, so the golden is **computed here** rather than frozen from
a dump -- verified equal to ttsim's own dump on all 2048 elements before being
written down (``./optests/diff.sh where`` passes), and stronger than a frozen
dump because it pins what the op means.

Run:  python3 -m driver.blackhole.server.where_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "where.trace"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS Int32 at channel (0, 11), offset
# 0x596e80 (the fourth allocation, after the cond / a / b input tiles). Written
# by the BRISC writer, one 32x32 tile per op, contiguous.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x596E80
TILE_ELEMS = 1024
NUM_OPS = 2
DATA_SIZE = NUM_OPS * TILE_ELEMS
OP_NAMES = ["cond ? a : b", "cond ? b : a"]

# Host-side inputs, verbatim from optests/where/src/where.cpp.
_COND = [
    0 if i % 3 == 0 else ((i + 1) if i % 3 == 1 else (-(i + 1)) & 0xFFFFFFFF)
    for i in range(TILE_ELEMS)
]
_A = [(0x11110000 + i) & 0xFFFFFFFF for i in range(TILE_ELEMS)]
_B = [(0x22220000 + i) & 0xFFFFFFFF for i in range(TILE_ELEMS)]
EXPECTED = [_A[i] if _COND[i] else _B[i] for i in range(TILE_ELEMS)] + [
    _B[i] if _COND[i] else _A[i] for i in range(TILE_ELEMS)
]


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

    result = device.read(DST_DRAM_COORD, DST_ADDR, DATA_SIZE * 4)
    values = [
        int.from_bytes(result[i : i + 4], "little") for i in range(0, len(result), 4)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v, e) for i, (v, e) in enumerate(zip(values, EXPECTED)) if v != e]
    if wrong:
        i, got, exp = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"(op{i // TILE_ELEMS} '{OP_NAMES[i // TILE_ELEMS]}' datum "
            f"{i % TILE_ELEMS}): dst[{i}]={got:#010x} expected {exp:#010x}"
        )
    print(
        f"blackhole where_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"Int32 results across {NUM_OPS} where_tile ops "
        f"({', '.join(OP_NAMES)}) match the computed golden)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
