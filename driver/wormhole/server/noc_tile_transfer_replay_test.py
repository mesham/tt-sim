"""Socket-free replay of upstream ``NoC_tile_transfer`` on Wormhole.

The Wormhole twin of ``driver/blackhole/server/noc_tile_transfer_replay_test.py``
— see that module for what the program does. In short: tt-metal's own
``programming_examples/NoC_tile_transfer`` moves one ``uint16`` tile
DRAM -> core 0's L1 -> (NoC, core to core) -> core 1's L1 -> DRAM, gated by a
cross-core semaphore and with no compute kernel anywhere, and it is the only
multi-core upstream example carrying a real self-check
(``Result = 14 : Expected = 14``).

The two workers are logical ``(0, 0)`` and ``(0, 1)``, i.e. Wormhole physical
``(1, 1)`` and ``(1, 2)``. Every one of the 1024 elements of the destination
tile must be the host's ``input_data = 14``, so the golden is the program's own
expectation rather than a frozen dump.

Run:  python3 -m driver.wormhole.server.noc_tile_transfer_replay_test
      (or under pytest, as ``test_noc_tile_transfer_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "noc_tile_transfer.trace"
# Upstream logical (0, 0) and (0, 1) -> Wormhole physical (1, 1) and (1, 2).
TENSIX_POOL = [(1, 1), (1, 2)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 400_000

# The destination DRAM buffer: one 32x32 uint16 tile, single page, at the
# address the recorded host read back from.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D5440
TILE_ELEMS = 1024
INPUT_DATA = 14


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
            f"(DRAM -> core (1,1) L1 -> NoC -> core (1,2) L1 -> DRAM); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {INPUT_DATA}  "
            "[the host's own check is Result = 14 : Expected = 14]"
        )
    print(
        f"wormhole noc_tile_transfer_replay test OK ({n_msgs} messages; all "
        f"{TILE_ELEMS} uint16 elements == {INPUT_DATA}, the tile round-tripped "
        "DRAM -> L1 -> NoC -> L1 -> DRAM across two semaphore-synchronised workers)"
    )
    return 0


def test_noc_tile_transfer_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
