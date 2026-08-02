"""Socket-free replay of the captured tt-metal "sfpumath" wire trace on Blackhole.

``sfpumath`` is the FP32 half of the op-coverage differential test (see
``optests/sfpumath`` and ``optests/diff.sh``): one Float32 input tile
(``in[i] = 1.0 + i*0.01``) is copied into Dst and run through an SFPU math op,
with ``fp32_dest_acc_en`` and MathFidelity::HiFi4. The op sequence is
``recip_tile``, ``tanh_tile`` and ``sigmoid_tile``: a whole FP32 pipeline in
miniature — the unpack-to-Src conversion, the matrix unit's copy through ELWADD,
the SFPU's exponent extract and Newton refinement (every multiply-add going
through the bit-exact ``fma_model_bh``), and the pack back to L1. tanh and
sigmoid additionally exercise the lane-flag machinery: on FP32 Dst both run
``1/(1+exp(-x))``, whose accurate exp guards its overflow branch with an
``SFPIADD`` immediate condition code and whose Newton reciprocal guards its
second iteration with ``v_if(t < 0)`` (SFPPUSHC / SFPSETCC / SFPPOPC).

Unlike the other guards there is no closed-form golden to compute here: the
answer is whatever the hardware's approximations produce. The golden is
therefore ttsim's own dump, captured from a run of the vendor reference sim and
stored beside the trace in ``traces/sfpumath.expected``. All 3072 elements must
match **bit-exactly** — this is the differential-testing gate the plan's Phase 7
asked for, frozen so it can run with no oracle present.

Getting it to bit-exact needed Blackhole's *implied SrcA format*: the matrix
unit takes the operand format from what the unpacker last wrote the bank in,
not from ALU_FORMAT_SPEC_REG0_SrcA. tt-metal leaves that register at FP32 while
unpacking to TF32, so reading it narrowed every copied datum to 7 mantissa bits
and cost ~2 bits of precision on 860 of the 1024 results. See
``docs/plans/blackhole-support.md``.

Run:  python3 -m driver.blackhole.server.sfpumath_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "sfpumath.trace"
EXPECTED_DUMP = Path(__file__).resolve().parent / "traces" / "sfpumath.expected"
# Single worker at Blackhole physical coord (1, 2).
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# Output DRAM buffer: NUM_OPS * TILE_ELEMS Float32 at channel (0, 11), offset
# 0x594e80 (the second allocation, right after the single input tile at
# 0x593e80). Written by the NCRISC writer, one 32x32 tile per op.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594E80
TILE_ELEMS = 1024
NUM_OPS = 3
DATA_SIZE = NUM_OPS * TILE_ELEMS


def _load_expected():
    """ttsim's dump of the same program, as one contiguous hex string."""
    text = EXPECTED_DUMP.read_text().strip()
    assert len(text) == DATA_SIZE * 8, (
        f"{EXPECTED_DUMP.name} holds {len(text) // 8} elements, expected {DATA_SIZE}"
    )
    return [int(text[i : i + 8], 16) for i in range(0, len(text), 8)]


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

    result = device.read(DST_DRAM_COORD, DST_ADDR, DATA_SIZE * 4)
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
        f"blackhole sfpumath_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"FP32 results across {NUM_OPS} ops (recip/tanh/sigmoid) bit-exact "
        f"against the ttsim golden)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
