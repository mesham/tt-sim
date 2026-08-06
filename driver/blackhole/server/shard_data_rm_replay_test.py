"""Socket-free replay of upstream ``shard_data_rm`` on Blackhole.

tt-metal's ``programming_examples/shard_data_rm`` shards a row-major bfloat16
vector ``{2, 4, 6, ..., 32}`` out of an interleaved DRAM buffer into the L1 of
four workers: each core's reader kernel walks its own *sticks* (a stick is one
4-byte page, i.e. two bfloat16 values) with ``TensorAccessor::get_noc_addr`` +
``noc_async_read``, landing each one ``padded_offset_bytes`` apart in circular
buffer ``c_0``. Core ``c`` gets values ``8c+2 .. 8c+8``.

**It has no self-check at all** — upstream verifies it by eye, through DPRINT
from the device (`Core (0,0): 2 4 6 8` and so on), so a live run's exit status
means nothing. That is exactly why it is worth freezing: the expected L1
contents are pinned here instead, computed from the host's own
``src_vec[i] = (i + 1) * 2`` rather than dumped from a run.

Layout constants, read off the recorded run:

* the four workers are logical ``(0, 0..3)`` -> Blackhole physical
  ``(1, 2..5)``;
* CB ``c_0`` sits at L1 ``0x1b300`` on every one of them;
* ``padded_offset_bytes`` is 64 — the host aligns the 4-byte page up to the
  allocator's DRAM alignment — so stick 1 lands 64 bytes after stick 0 with
  the gap between them untouched.

Run:  python3 -m driver.blackhole.server.shard_data_rm_replay_test
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

TRACE = Path(__file__).resolve().parent / "traces" / "shard_data_rm.trace"
# Upstream logical (0, 0..3) -> Blackhole physical (1, 2..5).
TENSIX_POOL = [(1, 2), (1, 3), (1, 4), (1, 5)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

CB0_ADDR = 0x1B300
PADDED_OFFSET_BYTES = 64
VALUES_PER_STICK = 2
STICKS_PER_CORE = 2


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


def _bf16(bits):
    """Widen a bfloat16 bit pattern to a Python float."""
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


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

    shards = {}
    for core in TENSIX_POOL:
        tile = TENSIX_COORD_MAP[core]
        values = []
        for stick in range(STICKS_PER_CORE):
            raw = bytes(
                device.tt_device.read(
                    tile, CB0_ADDR + stick * PADDED_OFFSET_BYTES, 2 * VALUES_PER_STICK
                )
            )
            values += [
                _bf16(int.from_bytes(raw[i : i + 2], "little"))
                for i in range(0, len(raw), 2)
            ]
        shards[core] = values
    device.tt_device.shutdown()

    per_core = VALUES_PER_STICK * STICKS_PER_CORE
    for index, core in enumerate(TENSIX_POOL):
        # src_vec[i] = (i + 1) * 2, sharded `per_core` values at a time.
        expected = [float((index * per_core + n + 1) * 2) for n in range(per_core)]
        if shards[core] != expected:
            raise AssertionError(
                f"replaying {TRACE.name}: worker {core} (logical (0, {index})) "
                f"holds {shards[core]} in CB c_0, expected {expected} — the "
                "row-major shard did not land"
            )

    print(
        f"blackhole shard_data_rm_replay test OK ({n_msgs} messages; all "
        f"{len(TENSIX_POOL) * per_core} bfloat16 values sharded row-major into "
        f"{len(TENSIX_POOL)} workers' L1: "
        + ", ".join(f"{c}={[int(v) for v in shards[c]]}" for c in TENSIX_POOL)
        + ")"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
