"""Socket-free replay of the captured tt-metal "sfpumath" wire trace on Wormhole.

The Wormhole twin of ``driver/blackhole/server/sfpumath_replay_test.py``, and
the only Wormhole guard that checks *values* against the vendor reference
rather than replaying READ replies byte-for-byte: the other Wormhole traces
come from examples that validate themselves, whereas ``optests/sfpumath``
exists precisely to be diffed against ttsim (see ``optests/diff.sh``, which
takes ``TT_SIM_ARCH=wormhole``).

One Float32 input tile (``in[i] = 1.0 + i*0.01``) is copied into Dst and run
through ``recip_tile``, ``tanh_tile`` and ``sigmoid_tile`` with
``fp32_dest_acc_en`` and MathFidelity::HiFi4. On tt-metal 0.74 none of the
three reaches a LUT: they are polynomial exp plus Newton reciprocal, so this is
a whole FP32 SFPU pipeline in miniature — exponent extract, the lane-flag
machinery (SFPIADD immediate condition codes, SFPPUSHC / SFPSETCC / SFPPOPC)
and, on every multiply-add, the bit-exact Wormhole float ALU ``fma_model_wh``.

Getting it bit-exact needed two things, both recorded in
``docs/plans/blackhole-support.md``: the **uniform uint32 LReg model** (the
Wormhole SFPU used to hold float lanes as Python floats, which destroyed the
NaN payload the accurate exp builds ``2**n`` through, made SFPIADD do float
arithmetic, and carried double precision between ops), and ``fma_model_wh``
itself — a *different* model from Blackhole's, not a shared one.

As on Blackhole there is no closed-form golden: the answer is whatever the
hardware's approximations produce, so the expected dump is ttsim's own, frozen
from a passing ``TT_SIM_ARCH=wormhole ./optests/diff.sh sfpumath`` run in
``traces/sfpumath.expected`` so the differential check runs with no oracle
present. All 3072 elements must match bit-exactly.

Run:  python3 -m driver.wormhole.server.sfpumath_replay_test
      (or under pytest, as ``test_sfpumath_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "sfpumath.trace"
EXPECTED_DUMP = Path(__file__).resolve().parent / "traces" / "sfpumath.expected"
# Single worker at Wormhole physical coord (1, 1) — logical (0, 0).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS Float32 written by the NCRISC writer,
# one 32x32 tile per op, at the second allocation in the channel behind
# physical (0, 11).
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D5C40
TILE_ELEMS = 1024
NUM_OPS = 3
DATA_SIZE = NUM_OPS * TILE_ELEMS


def _load_expected():
    """ttsim-Wormhole's dump of the same program, as one contiguous hex string."""
    text = EXPECTED_DUMP.read_text().strip()
    assert len(text) == DATA_SIZE * 8, (
        f"{EXPECTED_DUMP.name} holds {len(text) // 8} elements, expected {DATA_SIZE}"
    )
    return [int(text[i : i + 8], 16) for i in range(0, len(text), 8)]


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

    result = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, DATA_SIZE * 4)
    values = [
        int.from_bytes(result[i : i + 4], "little") for i in range(0, len(result), 4)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v, e) for i, (v, e) in enumerate(zip(values, expected)) if v != e]
    if wrong:
        i, got, exp = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements differ from the ttsim golden "
            f"replaying {TRACE.name} (op{i // TILE_ELEMS} datum {i % TILE_ELEMS}): "
            f"dst[{i}]={got:#010x} expected {exp:#010x}"
        )
    print(
        f"wormhole sfpumath_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"FP32 results across {NUM_OPS} ops (recip/tanh/sigmoid) bit-exact "
        f"against the ttsim-Wormhole golden)"
    )
    return 0


def test_sfpumath_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
