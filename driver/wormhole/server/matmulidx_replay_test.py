"""Socket-free replay of the captured tt-metal "matmulidx" wire trace on Wormhole.

The Wormhole twin of ``driver/blackhole/server/matmulidx_replay_test.py``, and
like ``reduce`` / ``transpose`` / ``sfpumath`` a *value* check rather than a
replay of recorded READ replies (see ``optests/diff.sh``, which takes
``TT_SIM_ARCH=wormhole``).

``matmulidx`` (see ``optests/matmulidx``) is the op test for ``matmul_tiles``
with *distinct* operand tile indices. ``examples/six`` — the only other matmul
in the tree — streams one tile at a time through a FIFO circular buffer and
always calls ``matmul_tiles(cb0, cb1, 0, 0, 0)``. The other first-class tt-metal
pattern keeps the whole operand block resident (``cb_wait_front(cb, n)``) and
indexes tiles directly, and that is what this covers: four matmuls over two
resident tiles per operand — two of them with ``in0_tile_index !=
in1_tile_index`` — plus a fifth output tile accumulating two distinct-index
matmuls into one DST, the shape a blocked GEMM's K loop has.

Both operand CBs hold constant tiles (in0: 1.0, 2.0; in1: 3.0, 5.0), so for a
32x32 tile every output element is ``32 * a * b`` — 96 / 320 / 192 / 160, and
352 for the accumulating one. All are exactly representable in bfloat16, so the
golden is **computed here** rather than frozen from a dump, and the comparison
is bit-exact rather than by PCC. ``six`` remains the guard for the matmul
arithmetic itself; this one is about which tiles it reads.

The bug it pins down: an ``UNPACR`` reads its configuration (context selection,
base addresses) up front and only *then* waits for its Src bank. tt-sim had that
order inverted, re-reading the configuration when a stalled unpack resumed. The
LLK's matmul unpack flips ``UNPACK_MISC_CFG_CfgContextOffset`` right after
issuing its UNPACRs, so from the third matmul on each operand unpacked through
the *next* call's context and read the wrong tile. It went unnoticed because
with equal tile indices both contexts hold the same address.

All 5120 bf16 elements must match. (``diff.sh`` reports 2560: it assumes 8 hex
chars per element, which halves the count for a bfloat16 program.)

Run:  python3 -m driver.wormhole.server.matmulidx_replay_test
      (or under pytest, as ``test_matmulidx_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "matmulidx.trace"
# Single worker at Wormhole physical coord (1, 1) — logical (0, 0).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: five 32x32 bf16 tiles, contiguous in one interleaved page
# at channel (0, 11). The two operand buffers (4 KiB each) are allocated first.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D6C40
TILE_ELEMS = 1024
TILE_BYTES = 2 * TILE_ELEMS
NUM_TILES = 5

# (in0_tile_index, in1_tile_index) per output tile, matching the compute kernel.
# The last tile accumulates two matmuls into one DST, as a blocked GEMM's K loop
# does, so it carries two index pairs.
IN0_VALUES = (1.0, 2.0)
IN1_VALUES = (3.0, 5.0)
INDICES = (((0, 0),), ((1, 1),), ((1, 0),), ((0, 1),), ((0, 1), (1, 0)))
EXPECTED = [
    sum(32.0 * IN0_VALUES[i0] * IN1_VALUES[i1] for i0, i1 in pairs) for pairs in INDICES
]


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


def _bf16(raw, offset):
    """Widen the little-endian bfloat16 at ``offset`` to a Python float."""
    bits = int.from_bytes(raw[offset : offset + 2], "little") << 16
    sign = -1.0 if bits >> 31 else 1.0
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0:
        return sign * mantissa * 2.0**-149
    return sign * (mantissa + (1 << 23)) * 2.0 ** (exponent - 150)


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

    raw = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, NUM_TILES * TILE_BYTES)
    device.tt_device.shutdown()

    for tile, (pairs, expected) in enumerate(zip(INDICES, EXPECTED)):
        for datum in range(TILE_ELEMS):
            got = _bf16(raw, tile * TILE_BYTES + datum * 2)
            if got != expected:
                calls = " + ".join(
                    f"matmul_tiles(in0={i0}, in1={i1})" for i0, i1 in pairs
                )
                distinct = any(i0 != i1 for i0, i1 in pairs)
                raise AssertionError(
                    f"replaying {TRACE.name}: {calls} datum {datum} is {got}, "
                    f"expected {expected}"
                    + ("  [distinct tile indices]" if distinct else "")
                )
    print(
        f"wormhole matmulidx_replay test OK ({n_msgs} messages; all "
        f"{NUM_TILES * TILE_ELEMS} bf16 elements match across the "
        f"matmul_tiles operand-index combinations {[list(p) for p in INDICES]})"
    )
    return 0


def test_matmulidx_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
