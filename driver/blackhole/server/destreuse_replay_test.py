"""Socket-free replay of the compiler team's dest-reuse precision probe on Blackhole.

The probe (``ttsim-divergences-2026-09-18/2-dest-reuse-probe``) is one core
running 46 element-wise arms twice, at HiFi4 and HiFi3, under
``fp32_dest_acc_en`` over fp32 circular buffers: every ``mul/add/sub_tiles``
and every ``binary_dest_reuse_tiles<OP, DEST_TO_SRCA/B>`` combination over
hashed 24-bit data, 8/10/11/12/16-bit-exact tiles and tie rows. Each arm was
measured bit for bit on a p150b, and the measured register model reproduces
the card on all 92 arms -- so this guard replays the captured wire trace and
checks every element of every arm against that model, in Python, here.

It pins the three mechanisms the hand-off exposed:

* ``MOVD2B`` takes its style from the *implied* SrcB format (TF32 here), not
  the configured FP32, so a ``DEST_TO_SRCB`` operand keeps 10 mantissa bits.
* A 32-bit ``ZEROACC`` clears the face relative to the math half-bank, so the
  dest-reuse ``ELWMUL`` on an odd (high-half) tile replaces rather than
  accumulates onto the operand just moved out of DEST.
* The FPU datapath itself: SrcA at 10 significand bits, SrcB at 11, exact
  products (minus the low x low phase at HiFi3), and the adder's rounding on
  the larger operand's 11-bit grid with ties away on an effective addition
  and toward zero on an effective subtraction.

Run:  python3 -m driver.blackhole.server.destreuse_replay_test
"""

import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.fabric import install_convention_guard
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP, wire_conventions

TRACE = Path(__file__).resolve().parent / "traces" / "destreuse.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 5000
PUMP_CAP = 2_000_000

# The host allocates the 14 input pages then the 46 output pages, 4 KiB each,
# contiguously in DRAM channel 0 (translated wire coord (17, 14)), and launches
# once per fidelity, HiFi4 first.
DRAM_COORD = (0, 11)
IN_BASE = 0x593E80
OUT_BASE = 0x5A1E80
PAGE = 0x1000
TILE_ELEMS = 1024
NUM_INPUTS = 14
NUM_ARMS = 46
FIDELITIES = ["HiFi4", "HiFi3"]

MUL, ADD, SUB = "MUL", "ADD", "SUB"
# (op, SrcA input, SrcB input) per arm, from kernels/compute/compute.cpp: a
# plain op puts in0 in SrcA and in1 in SrcB; DEST_TO_SRCB puts the cb in SrcA
# and DST in SrcB; DEST_TO_SRCA the reverse. Inputs: 0 u24, 1 v24, 2 ones,
# 3 p10, 4 q11, 5 a11, 6 b11, 7 w12, 8 x16, 9 e8, 10 u24 (UnpackToDestFp32),
# 11 mask, 12 -a11, 13 -b11.
ARMS = [
    (MUL, 0, 1), (MUL, 1, 0), (MUL, 1, 10), (MUL, 10, 1),
    (MUL, 3, 4), (MUL, 4, 3), (MUL, 3, 4), (MUL, 4, 3), (MUL, 4, 3), (MUL, 3, 4),
    (MUL, 0, 2), (MUL, 2, 0), (MUL, 2, 10), (MUL, 10, 2),
    (MUL, 9, 2), (MUL, 2, 9), (MUL, 3, 2), (MUL, 2, 3), (MUL, 4, 2), (MUL, 2, 4),
    (MUL, 7, 2), (MUL, 2, 7), (MUL, 8, 2), (MUL, 2, 8),
    (MUL, 11, 10), (MUL, 10, 11),
    (ADD, 0, 1), (ADD, 1, 10), (ADD, 10, 1), (ADD, 5, 6), (ADD, 6, 5), (ADD, 5, 6),
    (SUB, 0, 1), (SUB, 1, 10), (SUB, 10, 1), (SUB, 5, 6), (SUB, 6, 5), (SUB, 5, 6),
    (SUB, 6, 5), (ADD, 5, 13), (SUB, 5, 13), (ADD, 12, 6), (SUB, 13, 5), (ADD, 12, 13),
    (ADD, 5, 13), (SUB, 5, 13),
]  # fmt: skip
assert len(ARMS) == NUM_ARMS

# --- the measured register model ------------------------------------------

SRC_BITS = 11  # what SrcA / SrcB hold of an fp32 operand
MUL_SRCA_BITS = 10  # what the multiplier keeps of SrcA


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _trunc(value, bits):
    """Drop the significand below ``bits`` toward zero (fp32 sign-magnitude)."""
    if not math.isfinite(value) or value == 0.0:
        return value
    return _f32(_bits(value) & ((0xFFFFFFFF << (24 - bits)) & 0xFFFFFFFF))


def _exponent(value):
    return math.frexp(value)[1] - 1


def _round_on_grid(value, ulp, ties_away):
    scaled = abs(value) / ulp
    below = math.floor(scaled)
    remainder = scaled - below
    if remainder > 0.5 or (remainder == 0.5 and ties_away):
        below += 1
    return math.copysign(below * ulp, value)


def _fpu_mul(a, b, hifi3):
    a = _trunc(a, MUL_SRCA_BITS)
    b = _trunc(b, SRC_BITS)
    product = a * b  # 10 x 11 significand bits: exact in a double
    if hifi3:
        # HiFi3 drops the low x low phase: SrcA is 5 + 5 bits, SrcB 7 + 4.
        a_lo = a - _trunc(a, 5)
        b_lo = b - _trunc(b, 7)
        product -= a_lo * b_lo
    return _f32(_bits(product))


def _fpu_add_sub(a, b, subtract):
    a = _trunc(a, SRC_BITS)
    b = _trunc(b, SRC_BITS)
    if subtract:
        b = -b
    total = a + b
    if not (math.isfinite(a) and math.isfinite(b)) or a == 0.0 or b == 0.0:
        return _f32(_bits(total))
    grid = 2.0 ** (max(_exponent(a), _exponent(b)) - (SRC_BITS - 1))
    return _round_on_grid(total, grid, ties_away=(a > 0.0) == (b > 0.0))


def _model(op, a, b, hifi3):
    if op == MUL:
        return _fpu_mul(a, b, hifi3)
    return _fpu_add_sub(a, b, op == SUB)


# --- replay -------------------------------------------------------------


def _build_fabric():
    # Captured with NoC translation on: DRAM arrives at its translated wire
    # coordinate, so register the alias the live server installs.
    device = make_device(noc_translation=True)
    fabric = Fabric()
    alias, _translated_only, untranslated_only = wire_conventions()
    install_convention_guard(
        fabric,
        translated=True,
        wire_alias=alias,
        foreign_coords=untranslated_only,
        reason="replay",
        descriptor_hint="",
        wire_addr="",
    )
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    for physical in TENSIX_POOL:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    return device, fabric


def _go_signal(device, core):
    return device.tt_device.read(core, GO_MSG_ADDR, 4)[3]


def _read_tile(device, address):
    raw = device.read(DRAM_COORD, address, TILE_ELEMS * 4)
    return [
        _f32(int.from_bytes(raw[i : i + 4], "little")) for i in range(0, len(raw), 4)
    ]


def _check_launch(device, fidelity):
    inputs = [_read_tile(device, IN_BASE + i * PAGE) for i in range(NUM_INPUTS)]
    hifi3 = fidelity == "HiFi3"
    wrong = []
    for k, (op, src_a, src_b) in enumerate(ARMS):
        got = _read_tile(device, OUT_BASE + k * PAGE)
        for e, (a, b, g) in enumerate(zip(inputs[src_a], inputs[src_b], got)):
            expected = _model(op, a, b, hifi3)
            if _bits(g) != _bits(expected):
                wrong.append((k, e, g, expected, a, b))
    return wrong


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)

    n_msgs = 0
    launches = 0
    wrong = []
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
                fidelity = FIDELITIES[launches]
                launches += 1
                wrong += [(fidelity, *w) for w in _check_launch(device, fidelity)]
    device.tt_device.shutdown()

    assert launches == len(FIDELITIES), (
        f"expected {len(FIDELITIES)} launches, saw {launches}"
    )
    if wrong:
        fidelity, k, e, got, expected, a, b = wrong[0]
        bad_arms = sorted({(fid, arm) for fid, arm, *_ in wrong})
        raise AssertionError(
            f"{len(wrong)} elements wrong across {len(bad_arms)} arms replaying "
            f"{TRACE.name}: first {fidelity} arm {k} ({ARMS[k][0]}) elem {e}: "
            f"got {_bits(got):#010x} ({got!r}) expected {_bits(expected):#010x} "
            f"({expected!r}) from SrcA {a!r} SrcB {b!r}; bad arms {bad_arms[:12]}"
        )
    print(
        f"blackhole destreuse_replay test OK ({n_msgs} messages; all "
        f"{len(FIDELITIES) * NUM_ARMS * TILE_ELEMS} fp32 results across {NUM_ARMS} "
        f"mul/add/sub and dest-reuse arms at HiFi4 and HiFi3 bit-exact against "
        f"the p150b-measured register model)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
