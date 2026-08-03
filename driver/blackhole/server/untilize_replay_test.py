"""Socket-free replay of the captured tt-metal "untilize" wire trace on Blackhole.

``untilize`` (see ``optests/untilize`` and ``optests/diff.sh``) is the op test
for the packer's multi-face address generation. One bfloat16 tile is run through
three ops that share a phase-1 ``matmul_tiles`` -> DST -> tiled ``pack_tile``
into an intermediate CB, and differ only in how that intermediate tile reaches
the output CB:

* op0  tiled ``pack_tile``      -- the control: each face packs at Y = 0
* op1  ``untilize_block``       -- CB -> CB row-major untilize (unpack-side)
* op2  ``pack_untilize_dest``   -- DST -> CB row-major untilize (pack-side)

The matmul's second operand is the identity tile, so C = A and every stage is a
lossless permutation of an input chosen to be 1024 distinct bfloat16-exact
values. The golden is therefore **computed here** rather than frozen from a
dump; ttsim-Blackhole reproduces the ramp exactly on all three ops
(``./optests/diff.sh untilize``).

op2 is what makes this guard Blackhole-specific: ``pack_untilize_dest`` lowers
to a completely different instruction sequence here than on Wormhole. Its MOP
runs one packer interface *pair* in ``DST_ACCESS_STRIDED_MODE`` -- PACR's
Blackhole-only ``dst_access_mode`` (raw bit 17), paired with the
``DEST_ACCESS_CFG`` remap/swizzle the math thread enables -- so the second half
of each PACR's 32 datums comes from Dst row ``base + 16`` rather than
``base + 1``, and one PACR emits one 32-datum row spanning two faces. Two decode
gaps had to close for it: ``dst_access_mode`` did not exist in tt-sim's PACR
argument table at all, and the same bit was being swallowed by ``AddrMode``,
whose shared Wormhole encoding runs to bit 23 -- so ``ADDR_MOD_1`` decoded as a
section nothing programs and the pack Y counter never advanced. Wormhole's own
form of the op is checked in ``driver/wormhole/server/untilize_replay_test.py``.

op1 is the unpack-side untilize, and it is the reason this guard grew: its
row-major-ness comes from the *unpacker*'s output address generator rather than
the packer. ``llk_unpack_untilize`` widens the tile descriptor's ``YDim`` to 16
so channel 0's Z stride is a whole face, and steps ADC channel 1's Y once per
UNPACR against a 16-datum ``Ystride`` -- so each 16-datum face row lands at its
own SrcA row and the four faces interleave into output rows. Reading the
descriptor a word at a time and carrying that start row through the
``SRCA_SET_SetOvrdWithAddr`` path is what this op checks.

op0 is not busywork either: it is the control that pins the shared packer
address generator -- the AddrMod Ydst/Zdst output-counter advance and the output
byte address carrying forward across a tile's PACRs, both of which this
architecture now takes the same path through as Wormhole.

Run:  python3 -m driver.blackhole.server.untilize_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "untilize.trace"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS bfloat16 at channel (0, 11), offset
# 0x594e80 (the third allocation, after the two operand tiles). Written by the
# BRISC writer, one 32x32 tile per op, contiguous.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594E80
TILE_ELEMS = 1024
NUM_OPS = 3
DATA_SIZE = NUM_OPS * TILE_ELEMS

OP_NAMES = ["pack_tile (tiled control)", "untilize_block", "pack_untilize_dest"]
# op -> is the output tiled (rather than row-major)?
OP_IS_TILED = [True, False, False]
CHECKED_OPS = [0, 1, 2]


def _ramp_bits(j):
    """bfloat16 bits of ``2**(j // 128 - 4) * (1 + (j % 128) / 128)``.

    1024 distinct values, each exactly representable in bfloat16's seven
    explicit mantissa bits, matching the ``ramp()`` the host writes in
    ``optests/untilize/src/untilize.cpp``. Exponent 127 + (j // 128 - 4)
    sits in bits 14:7, and j % 128 *is* the mantissa field.
    """
    return ((123 + j // 128) << 7) | (j % 128)


def _tile_index(row, col):
    """Linear index of element (row, col) in a 32x32 tile's 4x 16x16 face layout."""
    face = (row // 16) * 2 + (col // 16)
    return face * 256 + (row % 16) * 16 + (col % 16)


def _golden():
    """Per-op expected tile: the ramp, tiled for op0 and row-major for op1/op2."""
    tiles = []
    for op in range(NUM_OPS):
        tile = [0] * TILE_ELEMS
        for row in range(32):
            for col in range(32):
                j = row * 32 + col
                tile[_tile_index(row, col) if OP_IS_TILED[op] else j] = _ramp_bits(j)
        tiles.append(tile)
    return tiles


EXPECTED = _golden()


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

    result = device.read(DST_DRAM_COORD, DST_ADDR, DATA_SIZE * 2)
    values = [
        int.from_bytes(result[i : i + 2], "little") for i in range(0, len(result), 2)
    ]
    device.tt_device.shutdown()

    for op in CHECKED_OPS:
        got = values[op * TILE_ELEMS : (op + 1) * TILE_ELEMS]
        wrong = [(i, v, e) for i, (v, e) in enumerate(zip(got, EXPECTED[op])) if v != e]
        if wrong:
            i, value, exp = wrong[0]
            raise AssertionError(
                f"{len(wrong)}/{TILE_ELEMS} result elements wrong replaying "
                f"{TRACE.name} (op{op} {OP_NAMES[op]}, datum {i}): "
                f"dst[{i}]={value:#06x} expected {exp:#06x}"
            )
    checked = ", ".join(OP_NAMES[op] for op in CHECKED_OPS)
    print(
        f"blackhole untilize_replay test OK ({n_msgs} messages; all "
        f"{len(CHECKED_OPS) * TILE_ELEMS} bfloat16 results across "
        f"{len(CHECKED_OPS)} ops ({checked}) match the computed golden)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
