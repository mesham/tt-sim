"""Socket-free replay of the captured tt-metal "matmulblock" wire trace on Blackhole.

``matmulblock`` (see ``optests/matmulblock``) is the block twin of
``matmulidx``: where ``matmul_tiles`` unpacks one operand tile per Src bank,
``matmul_block`` walks several of them per call at addresses the unpacker's MOP
computes for itself, so it exercises far more of the operand addressing --
``_llk_unpack_AB_matmul_`` reprograms ``THCON_SEC{0,1}_REG3`` base addresses per
step while the replay buffer bumps them by a tile between unpacks, alternating
the two config contexts.

Both operand CBs hold a whole 2x2 tile block, resident at once (A: 1.0, 2.0,
4.0, 8.0; B: 3.0, 5.0, 16.0, 32.0), so for 32x32 constant tiles every element of
C = A @ B is ``32 * sum_k A[r][k] * B[k][c]`` -- 1120 / 2208 / 4480 / 8832, all
exactly representable in bfloat16, so this compares bit-exactly rather than by
PCC.

One ``matmul_block`` call is a *single* K step: ``kt_dim`` only supplies operand
A's row stride, so the kernel loops K itself. It emits the same C block three
times under the three interesting shapes, covering both sides of the LLK's
``reuse_a = ct_dim >= rt_dim`` split:

    ct=2 rt=1  reuse A, one output tile-row per call
    ct=2 rt=2  reuse A, the whole 2x2 block per call
    ct=1 rt=2  reuse B (the ``!reuse_a`` branch), one output tile-column per call

The values match ttsim (the vendor reference simulator) bit-exactly on both
Blackhole and Wormhole -- ``optests/diff.sh matmulblock``.

Run:  python3 -m driver.blackhole.server.matmulblock_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "matmulblock.trace"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: twelve 32x32 bf16 tiles, contiguous in one interleaved
# page at channel (0, 11). The two operand blocks (8 KiB each) are allocated
# first.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x597E80
TILE_ELEMS = 1024
TILE_BYTES = 2 * TILE_ELEMS

RT = 2
CT = 2
KT = 2
# Constant tile values, row-major: A is RT x KT, B is KT x CT.
A_VALUES = (1.0, 2.0, 4.0, 8.0)
B_VALUES = (3.0, 5.0, 16.0, 32.0)
# Per shape, the (r, c) of each packed output tile, in pack order.
SHAPES = (
    ("ct=2 rt=1 (reuse A, one output row per call)", ((0, 0), (0, 1), (1, 0), (1, 1))),
    ("ct=2 rt=2 (reuse A, whole block per call)", ((0, 0), (0, 1), (1, 0), (1, 1))),
    (
        "ct=1 rt=2 (reuse B, one output column per call)",
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ),
)
NUM_TILES = sum(len(tiles) for _, tiles in SHAPES)

GOLDEN = {
    (r, c): 32.0 * sum(A_VALUES[r * KT + k] * B_VALUES[k * CT + c] for k in range(KT))
    for r in range(RT)
    for c in range(CT)
}


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

    raw = device.read(DST_DRAM_COORD, DST_ADDR, NUM_TILES * TILE_BYTES)
    device.tt_device.shutdown()

    tile = 0
    for shape, positions in SHAPES:
        for r, c in positions:
            expected = GOLDEN[(r, c)]
            for datum in range(TILE_ELEMS):
                got = _bf16(raw, tile * TILE_BYTES + datum * 2)
                if got != expected:
                    raise AssertionError(
                        f"replaying {TRACE.name}: matmul_block {shape} output tile "
                        f"{tile} (C[{r}][{c}]) datum {datum} is {got}, "
                        f"expected {expected}"
                    )
            tile += 1
    print(
        f"blackhole matmulblock_replay test OK ({n_msgs} messages; all "
        f"{NUM_TILES * TILE_ELEMS} bf16 elements match across the three "
        f"matmul_block shapes {[shape for shape, _ in SHAPES]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
