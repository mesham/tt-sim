"""Socket-free replay of the captured tt-metal "optest" wire trace on Blackhole.

``optest`` is the op-coverage differential test (see ``optests/optest`` and
``optests/diff.sh``): one Int32 input tile is fed through a compute kernel that
applies a *sequence* of Tensix/SFPU ops, each emitting its own output tile, and
the whole dump is diffed against the ttsim reference sim. This guard bakes the
passing result in: it replays the captured run and checks every output element
against a golden computed here in Python (so it needs no oracle at run time).

The kernel runs six ops on ``in[i] = (i * 2654435761) mod 2**32``:

* op0  identity (copy)                -> in[i]
* op1  ``bitwise_and`` 0x0f0f0f0f     -> in[i] & 0x0f0f0f0f
* op2  ``bitwise_or``  0xf0f0f0f0     -> in[i] | 0xf0f0f0f0
* op3  ``bitwise_xor`` 0xffff0000     -> in[i] ^ 0xffff0000
* op4  ``add_unary``   12345 (int32)  -> (in[i] + 12345) mod 2**32
* op5  ``sub_unary``   1000  (int32)  -> (in[i] - 1000)  mod 2**32

Output tiles land contiguously in DRAM (one interleaved 6-tile page). Getting
all 6144 elements right exercises the full Int32 tile path end-to-end — unpack
-> Dst, the SFPU int32 load/store, the SFPU logical ops' imm12 first source, and
the packer — which the other examples don't cover (they use partial tiles /
non-Int32 formats). See ``docs/plans/blackhole-support.md``.

Run:  python3 -m driver.blackhole.server.optest_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "optest.trace"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS Int32 at channel (0, 11), offset
# 0x594e80 (the second allocation, right after the single input tile at
# 0x593e80). Written by the NCRISC writer, one 32x32 tile per op, contiguous.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594E80
TILE_ELEMS = 1024
NUM_OPS = 6
DATA_SIZE = NUM_OPS * TILE_ELEMS

_HASH = 2654435761  # Knuth multiplicative constant; in[i] = (i * _HASH) mod 2**32
_IN = [(i * _HASH) & 0xFFFFFFFF for i in range(TILE_ELEMS)]
EXPECTED = (
    [_IN[i] for i in range(TILE_ELEMS)]  # op0 identity
    + [_IN[i] & 0x0F0F0F0F for i in range(TILE_ELEMS)]  # op1 and
    + [_IN[i] | 0xF0F0F0F0 for i in range(TILE_ELEMS)]  # op2 or
    + [_IN[i] ^ 0xFFFF0000 for i in range(TILE_ELEMS)]  # op3 xor
    + [(_IN[i] + 12345) & 0xFFFFFFFF for i in range(TILE_ELEMS)]  # op4 add int32
    + [(_IN[i] - 1000) & 0xFFFFFFFF for i in range(TILE_ELEMS)]  # op5 sub int32
)


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
            f"(op{i // TILE_ELEMS} datum {i % TILE_ELEMS}): "
            f"dst[{i}]={got:#010x} expected {exp:#010x}"
        )
    print(
        f"blackhole optest_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"Int32 results across {NUM_OPS} ops (identity/and/or/xor/add/sub) match "
        f"the computed golden)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
