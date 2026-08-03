"""Socket-free replay of the captured tt-metal "sfpuchain" wire trace on Blackhole.

``sfpuchain`` is the *default bfloat16 Dst* half of the SFPU op-coverage
differential test (see ``optests/sfpuchain`` and ``optests/diff.sh``). Its
sibling guard ``sfpumath_replay_test`` runs with ``fp32_dest_acc_en`` and so
takes the FP32 branch of ``VectorUnit.get_dst_address`` throughout; this one
runs the path every stock ``init_sfpu`` + ``*_tile`` kernel uses, which had no
coverage at all until two Blackhole-only bugs surfaced together on it:

* ``SFPLOAD`` / ``SFPSTORE`` decoded ``dest_reg_addr`` 14 bits wide, but the
  field is the ISA's ``imm10`` (bits 9:0). On Blackhole bit 13 belongs to the
  3-bit ``sfpu_addr_mode`` (15:13), so every SFPU access leaked ``0x2000`` into
  the Dst row address and ran off the end of Dst.
* ``SFPGT`` with ``instr_mod1 == 8`` writes an all-ones / all-zero mask into VD
  rather than setting the lane flags. Blackhole's ``exp_tile`` masks the
  integer part with that mask before ``SFPSETEXP`` instead of clamping with an
  ``SFPSWAP``, so ignoring the modifier left the exponent unmasked and produced
  ``2**-122``-scale garbage for every result below 2.0.

One fixed bfloat16 input tile (``x = (i % 256) / 256``) is re-read by five ops,
each packing its own output tile, so a mismatch's tile index names the op:
``copy``, ``exp``, ``log``, ``add_binary(x, 1)`` and upstream's whole softplus
chain ``log(exp(x) + 1)``.

As with ``sfpumath`` and ``reduce`` there is no closed-form golden -- the answer
is whatever the SFPU's approximations produce -- so the golden is ttsim's own
dump, captured from the vendor reference sim and stored beside the trace in
``traces/sfpuchain.expected``. All 5120 bfloat16 elements must match
bit-exactly, and the guard runs with no oracle present.

Run:  python3 -m driver.blackhole.server.sfpuchain_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "sfpuchain.trace"
EXPECTED_DUMP = Path(__file__).resolve().parent / "traces" / "sfpuchain.expected"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS bfloat16 at channel (0, 11), offset
# 0x594e80 (the third allocation, after the input and ones tiles). Written by
# the BRISC writer, one 32x32 tile per op, contiguous.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594E80
TILE_ELEMS = 1024
NUM_OPS = 5
DATA_SIZE = NUM_OPS * TILE_ELEMS
OP_NAMES = ["copy", "exp", "log", "add_binary", "softplus chain"]


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
        f"blackhole sfpuchain_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"bfloat16 results across {NUM_OPS} SFPU ops "
        f"({', '.join(OP_NAMES)}) bit-exact against the ttsim golden)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
