"""Socket-free replay of upstream ``pad_multi_core`` on Blackhole.

tt-metal's ``programming_examples/pad_multi_core`` pads a 64x32 bfloat16
row-major tensor out to 64x64 with the value 2.0, splitting the rows over four
workers. Each core runs the TTNN-shaped reader/writer pair
(``pad_reader_dims_rm_interleaved`` / ``pad_writer_dims_rm_interleaved``) over
three circular buffers — data, pad value, alignment scratch — against
**4-byte-paged interleaved DRAM buffers**, so a single row of the output is
stitched together from 32 separate pages spread round-robin over all eight DRAM
channels. That page-granular interleaved dataflow is what this guard covers;
no compute kernel is involved.

**It has no self-check** — the host just prints the padded matrix — so the
expected contents are pinned here. They are *computed*, not dumped: the host
fills the source with ``src_vec[i] = i + 1`` and pads with 2.0, so output row
``m`` is ``bfloat16(32*m + n + 1)`` for ``n < 32`` and ``bfloat16(2.0)`` for
the 32 padded columns. Values above 256 are not representable in bfloat16, so
the golden rounds to nearest even exactly as the host's ``bfloat16`` type does
when it builds the input.

The readback itself comes from the trace: the recorded host reads the whole
destination buffer back one 4-byte page at a time, and those are the only DRAM
READs in the conversation, so the guard replays the trace, re-reads the same
``(core, address)`` list off the device and concatenates it in page order.
That sidesteps having to model the interleaving here.

Run:  python3 -m driver.blackhole.server.pad_multi_core_replay_test
"""

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "pad_multi_core.trace"
# Upstream logical (0, 0..3) -> Blackhole physical (1, 2..5).
TENSIX_POOL = [(1, 2), (1, 3), (1, 4), (1, 5)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

SRC_M, SRC_N = 64, 32
DST_M, DST_N = 64, 64
PAD_VALUE = 2.0
PAGE_BYTES = 4  # one uint32 page == two bfloat16 values


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
    return device.tt_device.read(TENSIX_COORD_MAP[core], GO_MSG_ADDR, 4)[3]


def _to_bf16(value):
    """The bfloat16 bit pattern for ``value``, round-to-nearest-even."""
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    upper, lower = bits >> 16, bits & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return upper & 0xFFFF


def _expected():
    """The padded 64x64 tensor, as bfloat16 bit patterns in row-major order."""
    pad = _to_bf16(PAD_VALUE)
    out = []
    for m in range(DST_M):
        out += [_to_bf16(m * SRC_N + n + 1) for n in range(SRC_N)]
        out += [pad] * (DST_N - SRC_N)
    return out


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)

    dram_cores = set(DRAM_COORD_MAP)
    readback = []
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
            if parsed["cmd"] == proto.CMD_READ and parsed["core"] in dram_cores:
                readback.append((parsed["core"], parsed["address"], parsed["size"]))
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

    pages = DST_M * DST_N * 2 // PAGE_BYTES
    if len(readback) != pages or any(size != PAGE_BYTES for _, _, size in readback):
        raise AssertionError(
            f"replaying {TRACE.name}: expected {pages} {PAGE_BYTES}-byte DRAM "
            f"page reads (the destination-buffer readback), saw {len(readback)} "
            "— recapture the trace"
        )

    raw = b"".join(
        bytes(device.read(DRAM_COORD_MAP[core], addr, size))
        for core, addr, size in readback
    )
    device.tt_device.shutdown()

    got = [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]
    expected = _expected()
    wrong = [(i, g, e) for i, (g, e) in enumerate(zip(got, expected)) if g != e]
    if wrong:
        i, g, e = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{len(expected)} elements wrong replaying "
            f"{TRACE.name} (64x32 -> 64x64 row-major pad over 4 workers); "
            f"first: dst[{i // DST_N}][{i % DST_N}] is 0x{g:04x}, "
            f"expected 0x{e:04x}"
        )
    print(
        f"blackhole pad_multi_core_replay test OK ({n_msgs} messages; all "
        f"{len(expected)} bfloat16 elements of the padded {DST_M}x{DST_N} "
        f"tensor match, {pages} interleaved 4-byte DRAM pages stitched by "
        f"{len(TENSIX_POOL)} workers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
