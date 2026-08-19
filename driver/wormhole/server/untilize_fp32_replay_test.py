"""Socket-free replay of the "untilize" wire trace in its **fp32-DEST** arm (Wormhole).

The sibling of ``untilize_replay_test.py``, replaying the *same* op-test program
(``optests/untilize``) built the same way but launched with
``fp32_dest_acc_en = true`` while every circular buffer stays ``Float16_b``
(``UNTILIZE_FP32=1``, see the run-mode note at the top of
``optests/untilize/src/untilize.cpp``). That is a 32-bit DEST feeding a 16-bit
pack source format, and it is the ordinary GEMM configuration the tt-xftn
compiler team runs -- so it needs its own recorded conversation: the compute
kernel is the same source, but tt-metal JITs it differently (the pack config
carries ``PCK_DEST_RD_CTRL_Read_32b_data`` and the math thread runs the 32-bit
DEST path), so the default arm's trace cannot stand in for it.

What it guards. tt-sim used to take the packer's DEST read width from the *pack
source format* rather than from ``PCK_DEST_RD_CTRL_Read_32b_data``, so under a
16-bit format it read DEST 16 bits at a time even when DEST held fp32: 896, 960
and 896 of 1024 datums wrong on the three ops, with op 0 -- the *tiled control*,
which shares nothing with untilize but the pack -- landing only the top-left
8 rows x 16 columns. The unit-level pin is
``tt_sim/pe/tensix/pack_dest_rd_ctrl_test.py``; this guard is the integrated
one, over a real kernel, real config plumbing and a real pack.

Why 128 datums still came back right, and why that matters here. A 32-bit DEST
read of row ``r`` takes its *high* half from ``dstBits[Adj32(r)]`` and its low
half from ``dstBits[Adj32(r) + 8]``, and ``Adj32`` is the identity for
``r < 8``. So for DEST rows 0-7 a 16-bit read lands exactly on the fp32 datum's
high half -- which for bf16-exact data *is* the right answer -- and the fault is
invisible. It only shows from row 8 up, where ``Adj32(8) = 16`` and the 16-bit
read returns the low half of row 0 instead. **A 32x32 tile occupies DEST rows
0-63** (face ``f`` row ``r`` at DEST row ``16f + r``), so this guard covers rows
0-63 and the 896 datums per op that come from rows 8-63 are what makes it bite.
A guard confined to the first eight rows would pass against the broken packer.

The golden -- the ramp, tiled for op 0 and row-major for ops 1 and 2 -- is
imported from the default arm's guard rather than recomputed: bf16 operands
accumulated in fp32 and packed back to bf16 are still exact, so both arms must
produce byte-identical output, and importing it is what makes that a property of
the code rather than of two copies staying in step. tt-sim reproduces this arm
bit-for-bit against ttsim-Wormhole as well (``TT_SIM_ARCH=wormhole
UNTILIZE_FP32=1 ./optests/diff.sh untilize``), which is also how the trace was
captured, with ``TT_SIM_RECORD`` pointed at ``traces/untilize_fp32.trace``.

Run:  python3 -m driver.wormhole.server.untilize_fp32_replay_test
      (or under pytest, as ``test_untilize_fp32_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .untilize_replay_test import EXPECTED, NUM_OPS, OP_IS_TILED, OP_NAMES, TILE_ELEMS
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "untilize_fp32.trace"
# Single worker at Wormhole physical coord (1, 1) — logical (0, 0).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: the same shape and the same allocation as the default arm
# (NUM_OPS 32x32 bfloat16 tiles, contiguous, third allocation in the channel
# behind physical (0, 11)) — only the compute config differs between the arms.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D5C40
DATA_SIZE = NUM_OPS * TILE_ELEMS

CHECKED_OPS = [0, 1, 2]
# DEST rows 0-7 alias onto the fp32 datum's high half under a 16-bit read, so a
# mismatch there means something other than the read width went wrong.
ALIASING_DEST_ROWS = 8


def _tile_rc(op, i):
    """The tile (row, col) that output datum ``i`` of op ``op`` carries.

    Op 0 packs tiled, so its output index is face-major (four 16x16 faces);
    ops 1 and 2 pack row-major, where the index is already (row, col).
    """
    if OP_IS_TILED[op]:
        face, within = divmod(i, 256)
        return (face // 2) * 16 + within // 16, (face % 2) * 16 + within % 16
    return divmod(i, 32)


def _dest_row(op, i):
    """Which DEST row output datum ``i`` of op ``op`` was packed from.

    Face ``f`` row ``r`` of the tile lives at DEST row ``16f + r``, so a whole
    32x32 tile spans DEST rows 0-63 whichever way the op packs it.
    """
    row, col = _tile_rc(op, i)
    return ((row // 16) * 2 + (col // 16)) * 16 + row % 16


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

    result = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, DATA_SIZE * 2)
    values = [
        int.from_bytes(result[i : i + 2], "little") for i in range(0, len(result), 2)
    ]
    device.tt_device.shutdown()

    high_rows = 0
    for op in CHECKED_OPS:
        got = values[op * TILE_ELEMS : (op + 1) * TILE_ELEMS]
        wrong = [(i, v, e) for i, (v, e) in enumerate(zip(got, EXPECTED[op])) if v != e]
        high_rows += sum(
            1 for i in range(TILE_ELEMS) if _dest_row(op, i) >= ALIASING_DEST_ROWS
        )
        if wrong:
            i, value, exp = wrong[0]
            high = sum(1 for j, _, _ in wrong if _dest_row(op, j) >= ALIASING_DEST_ROWS)
            raise AssertionError(
                f"{len(wrong)}/{TILE_ELEMS} result elements wrong replaying "
                f"{TRACE.name} (op{op} {OP_NAMES[op]}, first at datum {i} = "
                f"tile row {_tile_rc(op, i)[0]} col {_tile_rc(op, i)[1]}, DEST "
                f"row {_dest_row(op, i)}): "
                f"dst[{i}]={value:#06x} expected {exp:#06x}; {high} of the wrong "
                f"datums come from DEST rows {ALIASING_DEST_ROWS}-63 and "
                f"{len(wrong) - high} from rows 0-{ALIASING_DEST_ROWS - 1}, which "
                "alias onto the fp32 datum's high half — see the module docstring"
            )
    checked = ", ".join(OP_NAMES[op] for op in CHECKED_OPS)
    print(
        f"wormhole untilize_fp32_replay test OK ({n_msgs} messages; all "
        f"{len(CHECKED_OPS) * TILE_ELEMS} bfloat16 results across "
        f"{len(CHECKED_OPS)} ops ({checked}) match the computed golden, "
        f"{high_rows} of them packed from DEST rows {ALIASING_DEST_ROWS}-63 "
        "where a 16-bit DEST read does not alias)"
    )
    return 0


def test_untilize_fp32_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
