"""Micro-benchmark for `EventBus.publish` overhead in three modes.

Run as a module:

    python3 -m tt_sim.trace.benchmark

Target (from ROADMAP §H Phase 1 design principles): <100 ns per call
on the no-subscriber fast path so hooks can stay compiled in.
"""

import time

from tt_sim.trace.bus import get_bus
from tt_sim.trace.events import EventCategory, InstrEvent

ITERATIONS = 1_000_000
SAMPLE_UNIT_ID = (0, 18, 18, "BRISC")


def _bench(label, fn):
    # Warm-up
    for _ in range(10000):
        fn()
    start = time.perf_counter_ns()
    for _ in range(ITERATIONS):
        fn()
    elapsed_ns = time.perf_counter_ns() - start
    per_call_ns = elapsed_ns / ITERATIONS
    print(f"  {label:<42s}  {per_call_ns:8.1f} ns/call")
    return per_call_ns


def bench_disabled_bus():
    """Master enable off — fast path is `is_enabled` returning False."""
    bus = get_bus()
    bus.reset()  # ensures no subscribers, disabled

    def call():
        if bus.is_enabled(EventCategory.INSTR):
            bus.publish(
                InstrEvent(
                    cycle=0,
                    unit_id=SAMPLE_UNIT_ID,
                    pc=0,
                    instruction=0,
                    stalled=False,
                )
            )

    return _bench("disabled bus (gate short-circuits)", call)


def bench_enabled_no_subscribers():
    """Master enabled, no subscribers — `publish` runs but loop is empty."""
    bus = get_bus()
    bus.reset()
    bus.enabled = True

    def call():
        if bus.is_enabled(EventCategory.INSTR):
            bus.publish(
                InstrEvent(
                    cycle=0,
                    unit_id=SAMPLE_UNIT_ID,
                    pc=0,
                    instruction=0,
                    stalled=False,
                )
            )

    return _bench("enabled, no subscribers (construct+publish)", call)


def bench_enabled_one_subscriber():
    """One no-op subscriber — measures the dispatch loop cost too."""
    bus = get_bus()
    bus.reset()
    bus.enabled = True
    sink = []
    bus.subscribe(EventCategory.INSTR, sink.append)

    def call():
        if bus.is_enabled(EventCategory.INSTR):
            bus.publish(
                InstrEvent(
                    cycle=0,
                    unit_id=SAMPLE_UNIT_ID,
                    pc=0,
                    instruction=0,
                    stalled=False,
                )
            )
            sink.clear()

    return _bench("enabled, 1 list.append subscriber", call)


def main():
    print(f"EventBus.publish micro-benchmark — {ITERATIONS:,} iterations each\n")
    bench_disabled_bus()
    bench_enabled_no_subscribers()
    bench_enabled_one_subscriber()
    print("\nTarget (Phase 1 design principle): disabled-bus path <100 ns/call.")


if __name__ == "__main__":
    main()
