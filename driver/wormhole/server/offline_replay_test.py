"""Socket-free replay of the captured tt-metal "one" wire trace.

Unlike ``one_replay_test`` (which spawns a server and dials it over nng), this
drives the fabric + ``Device`` wrapper *directly*, in one process, with no
socket — so it runs in restricted environments where IPC transport is
unavailable, and pins the wire path (translated-coord routing, the cycle pump,
reset handling, the tt-sim Wormhole itself) against real captured traffic.

Every READ in the trace carries the reply the live server produced; a clean run
means every one reproduces bit-for-bit through ``Transport._handle``.

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

# The captured trace predates two things, so a handful of READ replies no longer
# reproduce byte-for-byte. Neither reflects the compute path — the DRAM result
# buffer still matches exactly — so they are reported but not failed:
#   * Ethernet coords: the trace was recorded before EthTile existed, when eth
#     coords were NullCore zero-fills; they now read real eth L1.
#   * Address 0x2a0: the go-message mailbox the host spin-polls waiting for
#     RUN_MSG_DONE. Its intermediate poll values depend on cycle-exact firmware
#     timing the functional sim does not reproduce across code revisions.
_TRANSIENT_ADDRS = {0x2A0}


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
            n_reads += 1
            if bytes(reply) == bytes(parsed["reply"]):
                verified += 1
                continue
            # A mismatch is tolerated only where the trace is known-stale.
            if parsed["core"] in eth_coords or parsed["address"] in _TRANSIENT_ADDRS:
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
            f"{mismatches}/{n_reads} READ replies mismatched (beyond known-stale "
            f"eth/mailbox reads) replaying {TRACE.name}"
        )
    print(
        f"offline_replay test OK ({n_msgs} messages; {verified}/{n_reads} READs "
        f"reproduced bit-for-bit, {tolerated} known-stale eth/mailbox reads tolerated)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
