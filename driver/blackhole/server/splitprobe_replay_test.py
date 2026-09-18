"""Socket-free replay of the compiler team's fp32 split probe on Blackhole.

The probe (``ttsim-divergences-2026-09-18/1-split-probe``) is one core under
``fp32_dest_acc_en`` over fp32 circular buffers. Its arms o14..o17 form the
compiler's hi/lo split of an fp32 operand *on the device*: ``copy_tile`` of a
CB marked ``UnpackToDestFp32`` (o14, exact), ``bitwise_and_tile<Int32>(dst,
0xFFFFC000)`` on that fp32 tile (o15, the 10-bit ``hi``), ``sub_binary_tile``
(o16, the residual ``lo``) and ``typecast_tile<Float32, Float16_b>`` (o17,
round to nearest even). All four are bit-exact on a p150b; this guard replays
the captured wire trace and checks every element of the four against the
same closed-form references.

It pins two things the Int32 view of an fp32 datum depends on: ``SFPLOADI``'s
``SHORT`` mode sign-extends (sfpi emits it for the ``0xFFFFC000`` mask, which
zero-extended masks only the low half), and Integer "32" shares the FP32 Dst
layout, so an int32 load of an fp32 datum sees the datum's own bit pattern.

Run:  python3 -m driver.blackhole.server.splitprobe_replay_test
"""

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

TRACE = Path(__file__).resolve().parent / "traces" / "splitprobe.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 5000
PUMP_CAP = 2_000_000

# 18 input pages then 23 output pages, 4 KiB each, contiguous in DRAM channel
# 0 (translated wire coord (17, 14)). Input 0 is the hashed 24-bit u24 tile.
DRAM_COORD = (0, 11)
IN_BASE = 0x593E80
OUT_BASE = 0x5A5E80
PAGE = 0x1000
TILE_ELEMS = 1024
U24 = 0


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _rne_bf16(bits):
    """fp32 bit pattern -> fp32 pattern of its bf16 round-to-nearest-even."""
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) << 16


def _expected(u_bits):
    hi = u_bits & 0xFFFFC000
    lo = _bits(_f32(u_bits) - _f32(hi))  # exact: lo needs at most 14 bits
    return {
        14: u_bits,  # copy_tile of the UnpackToDestFp32 CB
        15: hi,  # bitwise_and_tile<Int32>(dst, 0xFFFFC000)
        16: lo,  # sub_binary_tile(u, hi)
        17: _rne_bf16(u_bits),  # typecast_tile<Float32, Float16_b>
    }


ARM_NAMES = {
    14: "copy_tile (exact)",
    15: "bitwise_and_tile<Int32> 0xFFFFC000",
    16: "sub_binary_tile(u, hi)",
    17: "typecast_tile<Float32,Float16_b>",
}


def _build_fabric():
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


def _read_tile_bits(device, address):
    raw = device.read(DRAM_COORD, address, TILE_ELEMS * 4)
    return [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]


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

    u24 = _read_tile_bits(device, IN_BASE + U24 * PAGE)
    outs = {k: _read_tile_bits(device, OUT_BASE + k * PAGE) for k in ARM_NAMES}
    device.tt_device.shutdown()

    wrong = []
    for e, u in enumerate(u24):
        for k, expected in _expected(u).items():
            if outs[k][e] != expected:
                wrong.append((k, e, outs[k][e], expected, u))
    if wrong:
        k, e, got, expected, u = wrong[0]
        raise AssertionError(
            f"{len(wrong)} elements wrong replaying {TRACE.name}: first arm o{k} "
            f"({ARM_NAMES[k]}) elem {e}: got {got:#010x} expected {expected:#010x} "
            f"from u24 {u:#010x}; bad arms {sorted({w[0] for w in wrong})}"
        )
    print(
        f"blackhole splitprobe_replay test OK ({n_msgs} messages; all "
        f"{len(ARM_NAMES) * TILE_ELEMS} fp32 results across the device hi/lo split "
        f"({', '.join(ARM_NAMES.values())}) bit-exact against the closed-form "
        f"references the p150b matches)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
