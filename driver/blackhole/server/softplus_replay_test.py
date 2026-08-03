"""Socket-free replay of the captured tt-metal "softplus" wire trace on Blackhole.

The Blackhole twin of ``driver/wormhole/server/softplus_replay_test.py``.
``optests/softplus`` is upstream's ``sfpu_eltwise_chain`` programming example
with its ``std::random_device`` input replaced by a fixed one — the three
kernels are byte-for-byte upstream's, so the same compiled code runs here and on
ttsim, and the output tile is dumped for ``optests/diff.sh``. Softplus is
``log(exp(x) + 1)``: ``exp_tile``, then ``add_binary_tile`` against a tile of
ones, then ``log_tile``, all chained inside one ``tile_regs_acquire`` on a
bfloat16 Dst — the default (non-``fp32_dest_acc_en``) Dst path that
``sfpumath_replay_test`` does not cover.

It also guards the *ones tile*, which the reader kernel builds on-device with
16-bit stores into L1 (``ptr[i] = fp32_to_bf16_truncate(1.0f)``) rather than
DMA-ing it in. RV32I ``sh`` used to write a single byte, so every element of
that tile came out 0x0080 instead of 0x3F80 — a denormal — and
``add_binary_tile`` silently added nothing, collapsing softplus to
``log(exp(x)) == x``.

There is no closed-form golden — the answer is whatever the SFPU's
approximations produce — so the expected dump is ttsim-Blackhole's own, frozen
from a passing ``./optests/diff.sh softplus`` run in ``traces/softplus.expected``.
All 1024 bfloat16 elements must match bit-exactly.

Run:  python3 -m driver.blackhole.server.softplus_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "softplus.trace"
EXPECTED_DUMP = Path(__file__).resolve().parent / "traces" / "softplus.expected"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: one 32x32 bfloat16 tile at channel (0, 11), the second
# allocation (right after the single input tile), written by the NCRISC writer.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594680
TILE_ELEMS = 1024


def _load_expected():
    """ttsim's dump of the same program, as one contiguous hex string."""
    text = EXPECTED_DUMP.read_text().strip()
    assert len(text) == TILE_ELEMS * 4, (
        f"{EXPECTED_DUMP.name} holds {len(text) // 4} elements, expected {TILE_ELEMS}"
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

    result = device.read(DST_DRAM_COORD, DST_ADDR, TILE_ELEMS * 2)
    values = [
        int.from_bytes(result[i : i + 2], "little") for i in range(0, len(result), 2)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v, e) for i, (v, e) in enumerate(zip(values, expected)) if v != e]
    if wrong:
        i, got, exp = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{TILE_ELEMS} result elements differ from the ttsim golden "
            f"replaying {TRACE.name}: dst[{i}]={got:#06x} expected {exp:#06x}"
        )
    print(
        f"blackhole softplus_replay test OK ({n_msgs} messages; all {TILE_ELEMS} "
        f"bfloat16 results of log(exp(x) + 1) bit-exact against the ttsim golden)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
