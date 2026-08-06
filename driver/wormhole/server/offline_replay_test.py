"""Socket-free replay of the captured tt-metal "one" wire trace.

Unlike ``one_replay_test`` (which spawns a server and dials it over nng), this
drives the fabric + ``Device`` wrapper *directly*, in one process, with no
socket — so it runs in restricted environments where IPC transport is
unavailable, and pins the wire path (translated-coord routing, the cycle pump,
reset handling, the tt-sim Wormhole itself) against real captured traffic.

The assertion strategy is the Blackhole guards' *poll-until-DONE* shape: the
recorded host->device traffic replays exactly as captured, but rather than
requiring the kernel to finish within the host's recorded poll count, the
device is pumped until the worker's go-message reaches ``RUN_MSG_DONE``
(bounded), and *then* every data READ reply — including the host's final
result-buffer read-back, which the trace places after the last go-poll — must
reproduce bit-for-bit. Only the go-message mailbox itself (whose intermediate
poll values are timing, not data) and Ethernet-core reads (the trace predates
``EthTile``) are exempt from the comparison, so the guard survives a
legitimate timing change (``TT_SIM_COST_MODEL``) while still failing on any
wrong value.

Run::  python3 -m driver.wormhole.server.offline_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import (
    DramCore,
    EthCore,
    Fabric,
    TensixCore,
    Transport,
    parse_trace_line,
)
from tt_sim.bridge import protocol as proto

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP, TENSIX_COORD_MAP
from .wh_device import make_device

REPO = Path(__file__).resolve().parents[3]
TRACE = REPO / "driver" / "wormhole" / "server" / "traces" / "one.trace"
# The "one" program launches a single worker at translated coord (1, 1).
TENSIX_POOL = [(1, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
PUMP_CAP = 600_000


def _build_fabric():
    device = make_device()
    fabric = Fabric()
    for translated, unified in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, unified))
    for translated, unified in ETH_COORD_MAP.items():
        fabric.register(translated, EthCore(device, unified))
    for physical in TENSIX_POOL:
        unified = TENSIX_COORD_MAP[physical]
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, unified))
    return device, fabric


def _go_signal(device, core):
    return device.tt_device.read(TENSIX_COORD_MAP[core], GO_MSG_ADDR, 4)[3]


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE.relative_to(REPO)} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)  # never connects; we only use _handle
    eth_coords = set(ETH_COORD_MAP)

    n_msgs = n_reads = verified = tolerated = mismatches = 0
    with TRACE.open() as f:
        for lineno, line in enumerate(f, 1):
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
            reply = transport._handle(fabric, req)
            n_msgs += 1
            if parsed["cmd"] != proto.CMD_READ or parsed["reply"] is None:
                continue
            # Pump the worker until its go-message reports DONE (bounded):
            # this is where the recorded poll budget stops mattering.
            if (
                parsed["address"] == GO_MSG_ADDR
                and parsed["core"] in TENSIX_POOL
                and _go_signal(device, parsed["core"]) == RUN_MSG_GO
            ):
                pumped = 0
                while (
                    _go_signal(device, parsed["core"]) != RUN_MSG_DONE
                    and pumped < PUMP_CAP
                ):
                    device.tt_device.run(PUMP_CHUNK)
                    pumped += PUMP_CHUNK
                if _go_signal(device, parsed["core"]) != RUN_MSG_DONE:
                    raise AssertionError(
                        f"worker {parsed['core']} go-message never reached "
                        f"RUN_MSG_DONE within {PUMP_CAP} pumped cycles replaying "
                        f"{TRACE.name}"
                    )
            n_reads += 1
            # Compare first: an exempt read that happens to match still counts
            # as verified (model off, everything reproduces). A mismatch is
            # tolerated only for the worker's go-message polls (timing, not
            # data) and eth reads (the trace predates EthTile); everything
            # else must reproduce bit-for-bit.
            if bytes(reply) == bytes(parsed["reply"]):
                verified += 1
                continue
            if parsed["core"] in eth_coords or (
                parsed["address"] == GO_MSG_ADDR and parsed["core"] in TENSIX_POOL
            ):
                tolerated += 1
                continue
            mismatches += 1
            if mismatches <= 5:
                print(
                    f"  line {lineno} READ {parsed['core']} "
                    f"@0x{parsed['address']:x}: expected "
                    f"{parsed['reply'].hex()} got {bytes(reply).hex()}",
                    file=sys.stderr,
                )

    device.tt_device.shutdown()
    if mismatches:
        raise AssertionError(
            f"{mismatches}/{n_reads} data READ replies mismatched after "
            f"pump-to-DONE replaying {TRACE.name}"
        )
    print(
        f"offline_replay test OK ({n_msgs} messages; pumped to DONE; "
        f"{verified}/{n_reads} data READs reproduced bit-for-bit, "
        f"{tolerated} go-message/eth reads exempt)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
