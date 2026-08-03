"""Socket-free replay of the captured tt-metal "softplus" wire trace on Wormhole.

``optests/softplus`` is upstream's ``sfpu_eltwise_chain`` programming example
with its ``std::random_device`` input replaced by a fixed one — the three
kernels are byte-for-byte upstream's, so the same compiled code runs here and
on ttsim, and the output tile is dumped for ``optests/diff.sh``. Softplus is
``log(exp(x) + 1)``: ``exp_tile``, then ``add_binary_tile`` against a tile of
ones, then ``log_tile``, all chained inside one ``tile_regs_acquire`` on a
bfloat16 Dst.

What it guards that ``sfpumath_replay_test`` does not is the *ones tile*: the
reader kernel builds it on-device with 16-bit stores into L1
(``ptr[i] = fp32_to_bf16_truncate(1.0f)``) rather than DMA-ing it in. RV32I
``sh`` used to write a single byte, so every element of that tile came out
0x0080 instead of 0x3F80 — a denormal — and ``add_binary_tile`` silently added
nothing, collapsing softplus to ``log(exp(x)) == x``. Upstream reads that only
as a marginal ``PCC 0.9986 < 0.999``, because softplus is near-linear over the
``[0, 1)`` range it samples; here all 1024 elements differ.

There is no closed-form golden — the answer is whatever the SFPU's
approximations produce — so the expected dump is ttsim-Wormhole's own, frozen
from a passing ``TT_SIM_ARCH=wormhole ./optests/diff.sh softplus`` run in
``traces/softplus.expected``. All 1024 bfloat16 elements must match bit-exactly.

Run:  python3 -m driver.wormhole.server.softplus_replay_test
      (or under pytest, as ``test_softplus_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "softplus.trace"
EXPECTED_DUMP = Path(__file__).resolve().parent / "traces" / "softplus.expected"
# Single worker at Wormhole physical coord (1, 1) — logical (0, 0).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: one 32x32 bfloat16 tile written by the NCRISC writer, at
# the second allocation in the channel behind physical (0, 11).
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D5440
TILE_ELEMS = 1024


def _load_expected():
    """ttsim-Wormhole's dump of the same program, as one contiguous hex string."""
    text = EXPECTED_DUMP.read_text().strip()
    assert len(text) == TILE_ELEMS * 4, (
        f"{EXPECTED_DUMP.name} holds {len(text) // 4} elements, expected {TILE_ELEMS}"
    )
    return [int(text[i : i + 4], 16) for i in range(0, len(text), 4)]


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

    result = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, TILE_ELEMS * 2)
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
        f"wormhole softplus_replay test OK ({n_msgs} messages; all {TILE_ELEMS} "
        f"bfloat16 results of log(exp(x) + 1) bit-exact against the "
        f"ttsim-Wormhole golden)"
    )
    return 0


def test_softplus_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
