"""Two ``detail::LaunchProgram``s in one host process, replayed socket-free.

A host program is free to launch more than one program against the same open
device, and tt-metal's slow-dispatch path does not reset the RISCs between
launches: the firmware parks in its go-message loop and the second launch reuses
whatever the first left behind. Every other guard in this directory launches
exactly once, so nothing covered the boundary.

It was broken. ``perfbench/tensixbench`` phase B launches one program per math
fidelity, and the *second* launch never completed -- the simulator span at 95 %
CPU indefinitely (25 min and 18 min on two attempts) while a single launch of
either fidelity finished in seconds. Because the wedge burned cycles rather than
going quiet, neither the dormancy path nor the deadlock watchdog caught it, and
it presented as "the simulator is slow".

The cause was in the Sync Unit's mutex queue, not in anything phase B does:
``TensixSyncUnit.clock_tick`` deleted granted waiters from ``blocked_mutex``
only when it granted more than one in a tick, so a lone grant stayed queued and
re-took the mutex on every later tick -- pinning it to that thread for the rest
of the device's life, one cycle after each ``ATRELM``. The first launch merely
had to contend for a mutex once; the second launch's first cross-thread
``ATGETM`` then blocked for ever. See ``tt_sim/pe/tensix/sync_mutex_queue_test``
for the unit-level pin.

The capture is phase B at ``--iters 1 --fidelities LoFi,HiFi2``: a
``matmul_tiles`` compute kernel on all three TRISCs fed by a BRISC dataflow
kernel over two circular buffers, launched twice with only the math fidelity
changed. It predates the fix -- the recorded replies for the second launch's
polls are all RUN_MSG_GO, because on the recording device it never finished --
so this guard replays for side effects and asserts what the device does:
**both** launches must reach RUN_MSG_DONE, and the second launch's results must
land in L1.

Run:  python3 -m driver.blackhole.server.twolaunch_replay_test
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from tt_sim.bridge import DramCore, Fabric, TensixCore, Transport
from tt_sim.bridge import protocol as proto
from tt_sim.bridge.trace import parse_trace_line
from tt_sim.util.conversion import conv_to_uint32

from .bh_device import make_device
from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP

TRACE = Path(__file__).resolve().parent / "traces" / "twolaunch.trace"
TENSIX_POOL = [(1, 2)]

GO_MSG_ADDR = 0x4F0
RUN_MSG_GO = 0x80
RUN_MSG_DONE = 0x00
PUMP_CHUNK = 2000
# Each launch of this shape completes in ~1000 cycles, so a launch that has not
# finished inside this has not merely been unlucky. Kept generous anyway: the
# gate re-runs every guard under the cost model, which stretches every cycle
# count.
PUMP_CAP = 200_000
# Both launches must complete. This is the assertion the bug broke: before the
# fix the count came out 1, whatever the pump cap.
EXPECTED_LAUNCHES = 2

# Host-side result layout (perfbench/tensixbench/src/kernels/compute/bench_layout.h).
RESULTS_ADDR = 0x17FC00
MAGIC = 0x7B10CE02
HDR_MAGIC, HDR_NUM_POINTS, HDR_ACTIVE_MASK = 0, 3, 5
HDR_WORDS = 8
NUM_PROBES, NUM_POINTS = 20, 4
# Threads 0 (unpack) and 1 (math) are timed; thread 2's copy of the inner loop
# is empty by construction and the kernel writes nothing for it.
TIMED_THREADS = 2


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
    return device.tt_device.read(core, GO_MSG_ADDR, 4)[3]


def _results(device, core, n_words):
    raw = device.tt_device.read(core, RESULTS_ADDR, n_words * 4)
    return [conv_to_uint32(raw[i * 4 : i * 4 + 4]) for i in range(n_words)]


def main():
    if not TRACE.exists():
        print(f"skipped: {TRACE} not present", file=sys.stderr)
        return 0

    device, fabric = _build_fabric()
    transport = Transport(addr=None)

    n_msgs = 0
    completed = 0
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
                if _go_signal(device, parsed["core"]) == RUN_MSG_DONE:
                    completed += 1

    core = TENSIX_POOL[0]
    words = _results(device, core, HDR_WORDS + TIMED_THREADS * NUM_PROBES * NUM_POINTS)
    sync = (
        device.tt_device.tile_directory[TENSIX_COORD_MAP[core]]
        .tensix_coprocessor.getBackend()
        .getSyncUnit()
    )
    held = [i for i, m in enumerate(sync.mutexes) if m.held_by is not None]
    queued = list(sync.blocked_mutex)
    device.tt_device.shutdown()

    if completed != EXPECTED_LAUNCHES:
        raise AssertionError(
            f"replaying {TRACE.name}: {completed} of {EXPECTED_LAUNCHES} launches "
            f"reached RUN_MSG_DONE within {PUMP_CAP} pumped cycles each. "
            f"Sync Unit at the end: mutexes held={held}, blocked_mutex={queued}"
        )
    if words[HDR_MAGIC] != MAGIC:
        raise AssertionError(
            f"replaying {TRACE.name}: result header magic 0x{words[HDR_MAGIC]:08X} "
            f"!= 0x{MAGIC:08X} -- the second launch's kernel did not write results"
        )
    if words[HDR_NUM_POINTS] != NUM_POINTS or words[HDR_ACTIVE_MASK] != 0x3:
        raise AssertionError(
            f"replaying {TRACE.name}: result header says num_points="
            f"{words[HDR_NUM_POINTS]} active_mask=0x{words[HDR_ACTIVE_MASK]:x}, "
            f"expected {NUM_POINTS} / 0x3"
        )
    # Point p times p+1 times the base iteration count, so each thread's four
    # measurements must be non-zero and must grow. The cycle counts themselves
    # are a timing model's business and are deliberately not pinned.
    for t in range(TIMED_THREADS):
        base = HDR_WORDS + t * NUM_PROBES * NUM_POINTS
        points = words[base : base + NUM_POINTS]
        if any(p == 0 for p in points):
            raise AssertionError(
                f"replaying {TRACE.name}: thread {t} timings {points} contain a "
                "zero -- the measured loop did not run at every iteration count"
            )
        if points[-1] <= points[0]:
            raise AssertionError(
                f"replaying {TRACE.name}: thread {t} timings {points} do not grow "
                "with the iteration count"
            )
    if held or queued:
        raise AssertionError(
            f"replaying {TRACE.name}: Sync Unit left mutexes held={held} "
            f"blocked_mutex={queued} after both launches finished"
        )
    print(
        f"blackhole twolaunch_replay test OK ({n_msgs} messages; both "
        f"LaunchPrograms reached RUN_MSG_DONE, second launch wrote "
        f"{TIMED_THREADS * NUM_POINTS} timings, no mutex left held)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
