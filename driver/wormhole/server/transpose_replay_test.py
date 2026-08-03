"""Socket-free replay of the captured tt-metal "transpose" wire trace on Wormhole.

The Wormhole twin of ``driver/blackhole/server/transpose_replay_test.py``, and
like ``reduce`` / ``sfpumath`` a *value* check rather than a replay of recorded
READ replies (see ``optests/diff.sh``, which takes ``TT_SIM_ARCH=wormhole``).

``transpose`` (see ``optests/transpose``) is the op test for the matrix unit's
Dst<->Src move path. One Float32 input tile (``in[i] = 1.0 + (i+1)/256``, every
datum distinct in *both* 16-bit halves) is run through three ops, one output
tile each, with ``fp32_dest_acc_en`` and MathFidelity::HiFi4:

* op0  ``transpose_tile``   -- unpacker face transpose plus the in-Dst 4x 16x16
  face transpose (``transpose_of_faces=false``, ``is_32bit=true``): MOVD2B /
  TRNSPSRCB / MOVB2A / MOVB2D / MOVA2D under SrcA format switching
* op1  ``transpose_dest``   -- the full in-Dst 32x32 transpose
  (``transpose_of_faces=true``, ``is_32bit=true``)
* op2  ``binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA>`` -- MOVD2A moves a Dst
  face back into SrcA, giving ``out = in + in``

Float32 is deliberate: it selects the 32-bit transpose path, where each datum is
transposed as two 16-bit halves. Because none of the three ops rounds -- two are
pure permutations and the third doubles an exactly-representable value -- the
golden is **computed here** rather than frozen from a dump: op0 and op1 must both
be the exact 32x32 transpose of the input and op2 exactly ``2 * in``. That was
checked against ttsim-Wormhole's own dump before being written down
(``TT_SIM_ARCH=wormhole ./optests/diff.sh transpose`` passes on all 3072
elements, and the dump is byte-identical to Blackhole's -- no rounding to differ
over), and a computed golden is the stronger of the two: it pins what the ops
*mean*, not just what one reference run produced.

Run:  python3 -m driver.wormhole.server.transpose_replay_test
      (or under pytest, as ``test_transpose_replay``)
"""

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "transpose.trace"
# Single worker at Wormhole physical coord (1, 1) — logical (0, 0).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS Float32 written by the BRISC writer,
# one 32x32 tile per op, contiguous, at the second allocation (right after the
# single input tile) in the channel behind physical (0, 11).
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D5C40
TILE_ELEMS = 1024
NUM_OPS = 3
DATA_SIZE = NUM_OPS * TILE_ELEMS
OP_NAMES = ["transpose_tile", "transpose_dest", "ELWADD DEST_TO_SRCA"]


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _tile_index(row, col):
    """Linear index of element (row, col) in a 32x32 tile's 4x 16x16 face layout."""
    face = (row // 16) * 2 + (col // 16)
    return face * 256 + (row % 16) * 16 + (col % 16)


def _golden():
    """out[op0] = in^T, out[op1] = in^T, out[op2] = in + in -- all exact in FP32."""
    src = [1.0 + (i + 1) / 256.0 for i in range(TILE_ELEMS)]
    transposed = [0] * TILE_ELEMS
    for row in range(32):
        for col in range(32):
            transposed[_tile_index(row, col)] = _f32_bits(src[_tile_index(col, row)])
    doubled = [_f32_bits(2.0 * v) for v in src]
    return transposed + transposed + doubled


EXPECTED = _golden()


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

    result = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, DATA_SIZE * 4)
    values = [
        int.from_bytes(result[i : i + 4], "little") for i in range(0, len(result), 4)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v, e) for i, (v, e) in enumerate(zip(values, EXPECTED)) if v != e]
    if wrong:
        i, got, exp = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"({OP_NAMES[i // TILE_ELEMS]} datum {i % TILE_ELEMS}): "
            f"dst[{i}]={got:#010x} expected {exp:#010x}"
        )
    print(
        f"wormhole transpose_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"Float32 results across {NUM_OPS} ops ({', '.join(OP_NAMES)}) match the "
        f"computed golden)"
    )
    return 0


def test_transpose_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
