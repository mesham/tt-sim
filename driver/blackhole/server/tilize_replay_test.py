"""Socket-free replay of the captured tt-metal "tilize" wire trace on Blackhole.

The Blackhole sibling of ``driver/wormhole/server/tilize_replay_test.py``, and
like it a *value* check rather than a replay of recorded READ replies (see
``optests/diff.sh``, which defaults to this architecture).

``tilize`` (see ``optests/tilize``) exercises the **unpacker's strided input
walk** and, on this architecture, the **packer's Dst read interfaces**:

* op0  ``copy_tile`` of an already-tiled tile -- the control, no tilize at all
* op1  ``tilize_block(block = 1)``  -- RowStride 64 bytes
* op2  ``tilize_block(block = 2)``  -- RowStride 128 bytes, two output tiles

Blackhole's ``UnpackRowWidth`` is 32 datums, so op1's 64-byte stride is exactly
contiguous here and only op2 is a genuinely strided unpack -- the mirror of
Wormhole, where the 16-datum row width makes both strided. What is specific to
this architecture is the *pack*: the tilize pack MOP issues its PACRs with
``read_intf_sel`` ``0b0101`` and ``0b1010``, i.e. the single Blackhole packer
gathering from Dst rows ``base``/``base + 2`` and then ``base + 1``/``base + 3``
to interleave two Dst halves into one tile. Reading that mask as Wormhole's
four-packer ``PackSel`` both packed twice and consulted ``THCON_SEC1_REG1``,
which tt-metal never writes -- whose zero ``Disable_zero_compress`` bit read as
"compressing" and refused the whole program with ``NotImplementedError``.

Every input value is a distinct bfloat16-exact ramp value and every stage
(unpack, datacopy, pack) is a lossless permutation, so the golden is **computed
here** rather than frozen from a dump: a datum fetched from the wrong L1 row or
the wrong Dst row is an unmistakable wrong value rather than a rounding
question. All three ops are checked, and match ttsim-Blackhole's own dump
(``./optests/diff.sh tilize`` passes, 2048 elements byte-identical).

Run:  python3 -m driver.blackhole.server.tilize_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "tilize.trace"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: four 32x32 bfloat16 tiles written by the BRISC writer,
# contiguous, at the fourth allocation (after the three input buffers) in the
# channel behind physical (0, 11).
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x595E80
TILE_ELEMS = 1024
OUT_TILES = 4
DATA_SIZE = OUT_TILES * TILE_ELEMS

OP_NAMES = ["copy_tile (tiled control)", "tilize_block x1", "tilize_block x2"]
# op -> (first output tile, number of output tiles), matching optests/tilize.
OP_TILES = [(0, 1), (1, 1), (2, 2)]
# Nothing is excluded: all three ops are modelled and pass.
CHECKED_OPS = [0, 1, 2]
# Columns in op2's row-major input block (two tiles side by side).
WIDE_COLS = 64


def _ramp_bits(j):
    """bfloat16 bits of ``2**(j // 128 - 8) * (1 + (j % 128) / 128)``.

    2048 distinct values, each exactly representable in bfloat16's seven
    explicit mantissa bits, matching the ``ramp()`` the host writes in
    ``optests/tilize/src/tilize.cpp``. Exponent 127 + (j // 128 - 8) sits in
    bits 14:7, and j % 128 *is* the mantissa field.
    """
    return ((119 + j // 128) << 7) | (j % 128)


def _tile_index(row, col):
    """Linear index of element (row, col) in a 32x32 tile's 4x 16x16 face layout."""
    face = (row // 16) * 2 + (col // 16)
    return face * 256 + (row % 16) * 16 + (col % 16)


def _golden():
    """The four expected output tiles, all in tiled layout (tilize's output).

    ops 0 and 1 carry the same logical 32x32 ramp -- one already tiled in L1,
    one row-major -- so they must produce identical tiles. op2's input is one
    row-major 32x64 block, whose left half becomes tile 2 and right half tile 3.
    """
    tiles = [[0] * TILE_ELEMS for _ in range(OUT_TILES)]
    for row in range(32):
        for col in range(32):
            value = _ramp_bits(row * 32 + col)
            tiles[0][_tile_index(row, col)] = value
            tiles[1][_tile_index(row, col)] = value
    for row in range(32):
        for col in range(WIDE_COLS):
            tiles[2 + col // 32][_tile_index(row, col % 32)] = _ramp_bits(
                row * WIDE_COLS + col
            )
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

    checked_tiles = 0
    for op in CHECKED_OPS:
        first, count = OP_TILES[op]
        for tile in range(first, first + count):
            got = values[tile * TILE_ELEMS : (tile + 1) * TILE_ELEMS]
            wrong = [
                (i, v, e) for i, (v, e) in enumerate(zip(got, EXPECTED[tile])) if v != e
            ]
            if wrong:
                i, value, exp = wrong[0]
                # Report the face-relative position too: a strided-walk fault
                # lands whole input rows in the wrong place, and a mis-selected
                # Dst read interface lands alternate rows of them.
                raise AssertionError(
                    f"{len(wrong)}/{TILE_ELEMS} result elements wrong replaying "
                    f"{TRACE.name} (op{op} {OP_NAMES[op]}, output tile {tile}, "
                    f"datum {i} = face {i // 256} row {(i % 256) // 16} col "
                    f"{i % 16}): dst[{i}]={value:#06x} expected {exp:#06x}"
                )
            checked_tiles += 1
    checked = ", ".join(OP_NAMES[op] for op in CHECKED_OPS)
    print(
        f"blackhole tilize_replay test OK ({n_msgs} messages; all "
        f"{checked_tiles * TILE_ELEMS} bfloat16 results across {checked_tiles} "
        f"tiles and {len(CHECKED_OPS)} ops ({checked}) match the computed "
        f"golden; RowStride 128 strided at Blackhole's 32-datum UnpackRowWidth, "
        f"packed through read_intf_sel 0b0101/0b1010)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
