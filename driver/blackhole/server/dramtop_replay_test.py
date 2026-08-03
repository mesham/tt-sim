"""Socket-free replay of the captured tt-metal "dramtop" wire trace on Blackhole.

``dramtop`` (see ``optests/dramtop``) is the reproducer for a *top-down* DRAM
allocation: one ``MeshBuffer`` created with ``DeviceLocalBufferConfig{.bottom_up
= false}``, one host write of ``0xA5A50000 + i``, one read back. No kernel, no
compute — the whole test is whether the simulator models enough DRAM for the
address tt-metal picks.

It did not. ``DRAMTile`` used to register two **10 MiB** banks, at 0x0 and
0x4000_0000, so everything in between was unmapped. tt-metal's default
bottom-up allocation stays inside the first 10 MiB, which is why every other
example passed; ``bottom_up = false`` walks down from the top of the bank and
landed in the hole, killing the server with ``IndexError: Provided address
'0xfefffc00' does not match any registered memory spaces`` (0x3fffd000 and
friends on Wormhole). It took out upstream's ``contributed/vecadd`` and
``contributed/multicast`` on both arches. A DRAM channel is now registered
whole — ``ArchProfile.dram_channel_size``, the SoC descriptor's
``dram_bank_size`` — over a sparse backing, since 8 x 0xFF00_0000 cannot be
allocated up front.

So this guard is about an *address*, not an op: the buffer lands at the very top
of channel 0 and every word must survive the round trip.

Run:  python3 -m driver.blackhole.server.dramtop_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge.trace import parse_trace_line

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "dramtop.trace"
TENSIX_POOL = [(1, 2)]

# The top-down allocation: 1 KiB at the very top of DRAM channel 0, i.e.
# `dram_bank_size` (0xFF00_0000) minus one page. The last address the modelled
# DRAM has to answer for, and the one that used to be unmapped.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0xFEFFFC00
NUM_WORDS = 256
EXPECTED = [0xA5A50000 + i for i in range(NUM_WORDS)]


def _build_fabric():
    device = make_device()
    fabric = Fabric()
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    for physical in TENSIX_POOL:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    return device, fabric


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
    # No kernel runs, so there is no go-message to pump on: the buffer is
    # written and read by the host over the wire.

    raw = device.read(DST_DRAM_COORD, DST_ADDR, NUM_WORDS * 4)
    values = [int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw), 4)]
    device.tt_device.shutdown()

    wrong = [(i, v) for i, (v, e) in enumerate(zip(values, EXPECTED)) if v != e]
    if wrong:
        index, got = wrong[0]
        raise AssertionError(
            f"{len(wrong)}/{NUM_WORDS} words wrong replaying {TRACE.name} "
            f"(top-down DRAM buffer at {hex(DST_ADDR)} on channel {DST_DRAM_COORD}); "
            f"first: word[{index}]={hex(got)} expected {hex(EXPECTED[index])}"
        )
    print(
        f"blackhole dramtop_replay test OK ({n_msgs} messages; all {NUM_WORDS} "
        f"words survive a top-down allocation at {hex(DST_ADDR)}, the top of "
        f"DRAM channel 0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
