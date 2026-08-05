"""Socket-free replay of the captured tt-metal "pipestall" wire trace on Wormhole.

``pipestall`` (see ``examples/pipestall``) is the tree's only workload in which a
Tensix backend unit waits on **another core**. Tile A ``(1, 1)`` runs
reader + compute + sender; tile B ``(2, 1)`` runs the writer and hands a credit
back over the NoC once it has drained a chunk to DRAM. With one credit and a
one-page output CB, tile A's pack thread parks in ``cb_reserve_back``, the math
thread cannot free Dst, and tile A's **unpackers block on a Src bank for as long
as tile B takes** — the cross-core case the per-unit stall detector's original
survey could not see, because no in-tree workload pipelined core-to-core.

So this guard checks two things, and the second is the reason it exists:

1. **Values.** Both tiles are launched, so we pump whichever tile's go-message is
   still GO until both report DONE, then read the DRAM result:
   ``dst[i] = src0[i] + src1[i] = (i % 128) + ((512 - i) % 128)`` for 512
   elements.
2. **The blocked run.** Every cycle of the replay, each unpacker's
   ``blocked_on()`` is sampled and the longest unbroken run recorded — the same
   quantity ``DeadlockDetector._watch_unit_stalls`` counts, measured exactly
   here rather than from the detector's (deliberately late-arming) watch. The
   run must be long enough to prove the cross-core stall really happened, and
   below ``DEFAULT_UNIT_STALL_THRESHOLD`` so a correct kernel stays quiet.

The frozen trace is the default configuration (``PIPESTALL_DELAY=200``,
``CREDITS=1``, ``OUT_DEPTH=1``), which blocks tile A's unpacker for ~1,056
cycles. That is not a ceiling — it is one point on a line. The consumer's cost
per chunk is a runtime argument, and the blocked run tracks it linearly
(~5 cycles per delay iteration), so ``PIPESTALL_DELAY=2000`` — a consumer doing
~10,000 cycles of perfectly ordinary downstream work per tile — pushes the same
*correct* kernel past the 10,000-cycle threshold. See ``tt_sim/device/deadlock.py``
for what that measurement means for the detector.

``OUT_DEPTH`` is a knob, though, and the *multi-page* case is where this example
found a bug in itself: its output CB pages were originally sized to the 256-byte
chunk the kernels move, while ``pack_tile`` writes a whole 4,096-byte tile, so at
``OUT_DEPTH=2`` page 1 lay inside page 0's pack footprint and the pack of chunk N
shredded the unread chunk N-1. The pages are tile-sized now; the live runners
cover the two-page shape as ``pipestall-2page``, and ``optests/packspill`` pins
the full-tile pack footprint against the vendor reference simulator.

Run:  python3 -m driver.wormhole.server.pipestall_replay_test
      (or under pytest, as ``test_pipestall_replay``)
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, EthCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line
from tt_sim.device.deadlock import DEFAULT_UNIT_STALL_THRESHOLD

from .coords import DRAM_COORD_MAP, ETH_COORD_MAP
from .wh_device import make_device

TRACE = Path(__file__).resolve().parent / "traces" / "pipestall.trace"
# Producer (reader + compute + sender) and consumer (writer) tiles.
TENSIX_POOL = [(1, 1), (2, 1)]

GO_MSG_ADDR = 0x4A0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 500
PUMP_CAP = 600_000

# dst DRAM buffer: 512 int32 at channel (0, 11), written by the tile B writer.
DST_DRAM_COORD = (0, 11)
DST_ADDR = 0x2D5040
DATA_SIZE = 512
EXPECTED = [(i % 128) + ((DATA_SIZE - i) % 128) for i in range(DATA_SIZE)]

# The cross-core stall this example exists to produce, at the frozen delay.
# Measured: 1,056 cycles (unpacker 0) / 1,055 (unpacker 1). The floor is well
# below that because the point of the assertion is "a multi-hundred-cycle
# cross-core stall happened at all", not the exact figure — timing changes (the
# cost model, issue-latency work) move it and should not fail this guard. The
# ceiling is the shipped threshold: a correct kernel must not be reported.
MIN_EXPECTED_BLOCKED_RUN = 200


class _BlockedRunTracker:
    """Longest unbroken run of blocked cycles per unit, sampled every cycle.

    Deliberately *not* the detector's own counter: that one arms at the
    watchdog's sample cadence and so under-reports the head of a run by up to
    one sample interval. This starts at the cycle the unit is first seen
    blocked, which is what the measurement wants.
    """

    def __init__(self):
        self.units = []
        self.open_runs = {}
        self.max_run = {}

    def register(self, coord, tile):
        for unpacker in tile.tensix_coprocessor.getBackend().unpacker_units:
            self.units.append((coord, f"Unpacker {unpacker.unpacker_id}", unpacker))

    def tick(self, cycle):
        for coord, name, unit in self.units:
            key = (coord, name)
            now = unit.blocked_on()
            cur = self.open_runs.get(key)
            if now is None:
                if cur is not None:
                    self._close(key, cycle, cur)
                    del self.open_runs[key]
            elif cur is None or cur[1] != now:
                if cur is not None:
                    self._close(key, cycle, cur)
                self.open_runs[key] = [cycle, now]

    def _close(self, key, cycle, cur):
        length = cycle - cur[0]
        if length > self.max_run.get(key, (0,))[0]:
            self.max_run[key] = (length, cur[1])

    def finish(self, cycle):
        for key, cur in list(self.open_runs.items()):
            self._close(key, cycle, cur)
        self.open_runs = {}

    def worst(self):
        if not self.max_run:
            return None
        key = max(self.max_run, key=lambda k: self.max_run[k][0])
        return key, self.max_run[key]


def _build_fabric(tracker):
    device = make_device()
    fabric = Fabric()
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    for translated, tile in ETH_COORD_MAP.items():
        fabric.register(translated, EthCore(device, tile))
    unified_of = {}
    for physical in TENSIX_POOL:
        unified = device.ensure_tensix_tile(physical)
        unified_of[physical] = unified
        fabric.register(physical, TensixCore(device, unified))
        tracker.register(physical, device.tt_device.tile_directory[unified])
    inner = device.tt_device.clocks[0].on_tick

    def on_tick(cycle):
        if inner is not None:
            inner(cycle)
        tracker.tick(cycle)

    device.tt_device.clocks[0].on_tick = on_tick
    return device, fabric, unified_of


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    tracker = _BlockedRunTracker()
    device, fabric, unified_of = _build_fabric(tracker)
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
            # Each launched tile flips its own go-message; pump the one being
            # polled until it reports DONE (bounded).
            if (
                parsed["cmd"] == proto.CMD_READ
                and parsed["core"] in unified_of
                and parsed["address"] == GO_MSG_ADDR
            ):
                core = unified_of[parsed["core"]]

                def go(core=core):
                    return device.tt_device.read(core, GO_MSG_ADDR, 4)[3]

                if go() == RUN_MSG_GO:
                    pumped = 0
                    while go() != RUN_MSG_DONE and pumped < PUMP_CAP:
                        device.tt_device.run(PUMP_CHUNK)
                        pumped += PUMP_CHUNK

    tracker.finish(device.tt_device.clocks[0].clock_tick_num)
    result = device.read(DRAM_COORD_MAP[DST_DRAM_COORD], DST_ADDR, DATA_SIZE * 4)
    values = [
        int.from_bytes(result[i : i + 4], "little") for i in range(0, len(result), 4)
    ]
    device.tt_device.shutdown()

    wrong = [(i, v, e) for i, (v, e) in enumerate(zip(values, EXPECTED)) if v != e]
    if wrong:
        raise AssertionError(
            f"{len(wrong)}/{DATA_SIZE} result elements wrong replaying {TRACE.name} "
            f"(two-tile credit-paced pipeline); "
            f"first: dst[{wrong[0][0]}]={wrong[0][1]} expected {wrong[0][2]}"
        )

    worst = tracker.worst()
    if worst is None:
        raise AssertionError(
            f"replaying {TRACE.name}: no unpacker ever blocked, so the "
            "cross-core stall this example exists to produce did not happen — "
            "the credit handshake or the CB depths have stopped back-pressuring "
            "the producer's Tensix pipeline"
        )
    (coord, name), (blocked, waiting) = worst
    if blocked < MIN_EXPECTED_BLOCKED_RUN:
        raise AssertionError(
            f"replaying {TRACE.name}: longest blocked run was only {blocked} "
            f"cycles ({coord} {name}), under the {MIN_EXPECTED_BLOCKED_RUN} this "
            "guard expects — the producer is no longer waiting on the consumer "
            "core, so the workload has stopped measuring what it is for"
        )
    if blocked >= DEFAULT_UNIT_STALL_THRESHOLD:
        raise AssertionError(
            f"replaying {TRACE.name}: {coord} {name} blocked for {blocked} "
            f"cycles, at or past the shipped TT_SIM_UNIT_STALL threshold of "
            f"{DEFAULT_UNIT_STALL_THRESHOLD} — this is a *correct* kernel, so "
            "the detector would now be reporting a false stall"
        )

    print(
        f"wormhole pipestall_replay test OK ({n_msgs} messages; all {DATA_SIZE} "
        f"result elements == src0[i] + src1[i] across two Tensix tiles paced by "
        f"a credit semaphore). Longest legitimate blocked run: {blocked} cycles "
        f"— {coord} {name} on {waiting[1]} waiting for {waiting[2]} bank "
        f"{waiting[3]}, i.e. waiting on the consumer core "
        f"(threshold {DEFAULT_UNIT_STALL_THRESHOLD})"
    )
    return 0


def test_pipestall_replay():
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
