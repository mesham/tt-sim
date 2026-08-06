"""Socket-free replay of upstream ``NoC_tile_transfer`` on Blackhole.

The first guard built from tt-metal's **own** ``programming_examples/`` rather
than from the in-tree ``examples/`` or ``optests/`` trees (see
``docs/upstream-examples-status.md``), and the only multi-core upstream program
with a real self-check: the host prints ``Result = 14 : Expected = 14``.

What the program does, on two workers connected by a semaphore:

* core 0 (logical ``(0, 0)``, physical ``(1, 2)``) — ``reader0`` pulls the one
  ``uint16`` tile out of DRAM into CB ``c_0`` with ``noc_async_read_page``;
  ``writer0`` then ``noc_async_write``s that page straight into **core 1's** CB
  ``c_1`` and bumps core 1's semaphore.
* core 1 (logical ``(0, 1)``, physical ``(1, 3)``) — ``reader1`` waits on that
  semaphore, and ``writer1`` writes the page back out to the destination DRAM
  buffer.

So the tile makes the round trip DRAM -> L1 -> (NoC, core to core) -> L1 ->
DRAM, gated by a cross-core semaphore, with no compute kernel anywhere. Every
one of the 1024 ``uint16`` elements is the host's ``input_data = 14``, so the
golden is the program's own expectation rather than a frozen dump — this guard
asserts exactly the verdict the live host prints.

Both workers are launched, so pump whichever tile's go-message is still GO
until it reports DONE.

Run:  python3 -m driver.blackhole.server.noc_tile_transfer_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "noc_tile_transfer.trace"
# Upstream logical (0, 0) and (0, 1) -> Blackhole physical (1, 2) and (1, 3).
TENSIX_POOL = [(1, 2), (1, 3)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# The destination DRAM buffer: one 32x32 uint16 tile, single page, at the
# address the recorded host read back from.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x594680
TILE_ELEMS = 1024
INPUT_DATA = 14


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

    raw = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, 2 * TILE_ELEMS)
    device.tt_device.shutdown()

    values = [int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2)]
    wrong = [(i, v) for i, v in enumerate(values) if v != INPUT_DATA]
    if wrong:
        raise AssertionError(
            f"{len(wrong)}/{TILE_ELEMS} elements wrong replaying {TRACE.name} "
            f"(DRAM -> core (1,2) L1 -> NoC -> core (1,3) L1 -> DRAM); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {INPUT_DATA}  "
            "[the host's own check is Result = 14 : Expected = 14]"
        )
    print(
        f"blackhole noc_tile_transfer_replay test OK ({n_msgs} messages; all "
        f"{TILE_ELEMS} uint16 elements == {INPUT_DATA}, the tile round-tripped "
        "DRAM -> L1 -> NoC -> L1 -> DRAM across two semaphore-synchronised workers)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
