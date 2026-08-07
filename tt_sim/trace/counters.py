"""Counter aggregator — derives running performance counters from the
event stream and emits :class:`CounterSnapshot` events at a configurable
cadence.

Subscribes to every category except :class:`EventCategory.COUNTER`
(would recurse). Only :class:`InstrEvent` drives flush decisions —
it's the highest-frequency event with a real ``cycle`` field; the
other categories either share the cycle field or carry no clock
context (``cycle == 0``).

Counters today are event-derived (instruction counts, dispatch
breakdowns, NoC throughput, mem op counts per region) plus, where the
cost model supplies the state, cycle-attributing ones:

- ``stall_cycles`` and ``stall_<reason>`` per baby RISC-V core, from
  ``InstrEvent.stall_cycles`` — the per-instruction half of
  ``tt_sim.pe.rv.cost.RiscvCostState.stall_by_reason``, so a run can
  say *where* its RV time went.
- ``busy_cycles`` per Tensix backend unit, from
  ``ComputeEvent.duration`` — the occupancy the cost tables charged,
  which against the run length is the unit's utilisation.
- ``noc_flight_cycles`` and ``noc_txns_timed`` per NIU, from
  ``NoCEvent.issue_cycle`` against the event's own cycle.
- ``tensix_stall_cycles`` per Tensix thread, split by
  ``tensix_stall_<reason>`` and ``tensix_stall_on_<unit>``, from
  ``StallEvent`` — where a thread's *lost* time went, against
  ``dispatch_total`` for where its spent time went.

Every one of those is **absent, not zero, with ``TT_SIM_COST_MODEL``
unset**: a counter is only emitted when something incremented it, so a
dataset from an un-modelled run simply has no ``stall_cycles`` rows
rather than rows asserting a stall-free machine. (``noc_flight_cycles``
is the exception and is emitted in both regimes, because a flight time
is measured rather than modelled — it is just always the one or two
cycles the two-list swap in ``NUI.clock_tick`` costs when the model is
off.)
"""

from collections import defaultdict

from tt_sim.trace.bus import get_bus
from tt_sim.trace.events import (
    ComputeEvent,
    CounterSnapshot,
    DispatchEvent,
    EventCategory,
    InstrEvent,
    LifecycleEvent,
    MemEvent,
    NoCEvent,
    StallEvent,
    SyncEvent,
)

DEFAULT_FLUSH_INTERVAL_CYCLES = 100


class CounterAggregator:
    def __init__(self, flush_interval_cycles: int = DEFAULT_FLUSH_INTERVAL_CYCLES):
        self._interval = max(1, flush_interval_cycles)
        self._last_flush_cycle = 0
        self._max_cycle = 0
        self._kernel_id = 0
        self._counters: dict[tuple[tuple, str], int] = defaultdict(int)
        bus = get_bus()
        bus.subscribe(EventCategory.INSTR, self._on_instr)
        bus.subscribe(EventCategory.DISPATCH, self._on_dispatch)
        bus.subscribe(EventCategory.COMPUTE, self._on_compute)
        bus.subscribe(EventCategory.NOC, self._on_noc)
        bus.subscribe(EventCategory.MEM, self._on_mem)
        bus.subscribe(EventCategory.SYNC, self._on_sync)
        bus.subscribe(EventCategory.STALL, self._on_stall)
        bus.subscribe(EventCategory.LIFECYCLE, self._on_lifecycle)

    def _on_instr(self, e: InstrEvent):
        self._counters[(e.unit_id, "instr_retired")] += 1
        if e.stalled:
            self._counters[(e.unit_id, "instr_stalled")] += 1
        if e.stall_cycles:
            self._counters[(e.unit_id, "stall_cycles")] += e.stall_cycles
            self._counters[(e.unit_id, f"stall_{e.stall_reason}")] += e.stall_cycles
        self._maybe_flush(e.cycle)

    def _on_dispatch(self, e: DispatchEvent):
        self._counters[(e.unit_id, "dispatch_total")] += 1
        self._counters[(e.unit_id, f"dispatch_to_{e.target_unit}")] += 1
        self._maybe_flush(e.cycle)

    def _on_compute(self, e: ComputeEvent):
        self._counters[(e.unit_id, "compute_ops")] += 1
        if e.duration:
            # Modelled occupancy only. An uncosted opcode contributes nothing
            # rather than a presumed 1, so ``busy_cycles`` is never inflated by
            # ops the tables have no opinion about.
            self._counters[(e.unit_id, "busy_cycles")] += e.duration
        self._maybe_flush(e.cycle)

    def _on_noc(self, e: NoCEvent):
        self._counters[(e.unit_id, f"noc_{e.phase}_{e.txn_type}")] += 1
        if e.phase == "response":
            self._counters[(e.unit_id, "noc_bytes_total")] += e.size_bytes
        if e.issue_cycle >= 0:
            self._counters[(e.unit_id, "noc_flight_cycles")] += max(
                0, e.cycle - e.issue_cycle
            )
            self._counters[(e.unit_id, "noc_txns_timed")] += 1
        self._maybe_flush(e.cycle)

    def _on_mem(self, e: MemEvent):
        # MemEvents carry cycle=0; they accumulate into whatever the
        # most-recent real-cycle bucket is.
        self._counters[(e.unit_id, f"mem_{e.op}_{e.region}")] += 1
        self._counters[(e.unit_id, f"mem_bytes_{e.op}")] += e.size

    def _on_sync(self, e: SyncEvent):
        self._counters[(e.unit_id, f"sync_{e.kind}")] += 1

    def _on_stall(self, e: StallEvent):
        # Namespaced ``tensix_`` because a Tensix thread's unit_id is the *same*
        # unit_id its baby RISC-V core publishes InstrEvents under -- an
        # unprefixed ``stall_cycles`` here would silently sum into the RV cost
        # model's counter of the same name and make both unreadable.
        self._counters[(e.unit_id, "tensix_stall_cycles")] += e.cycles
        self._counters[(e.unit_id, "tensix_stall_episodes")] += 1
        self._counters[(e.unit_id, f"tensix_stall_{e.reason}")] += e.cycles
        if e.blocked_on:
            self._counters[(e.unit_id, f"tensix_stall_on_{e.blocked_on}")] += e.cycles
        self._maybe_flush(e.cycle)

    def _on_lifecycle(self, e: LifecycleEvent):
        # Always flush at lifecycle boundaries; bump kernel_id at each
        # kernel_start so the next snapshots are attributed to the new
        # kernel.
        if e.kind == "kernel_start":
            self._flush(self._max_cycle)
            self._kernel_id += 1
        else:
            self._flush(self._max_cycle)

    def _maybe_flush(self, cycle: int):
        if cycle > self._max_cycle:
            self._max_cycle = cycle
        if cycle - self._last_flush_cycle >= self._interval:
            self._flush(cycle)

    def _flush(self, cycle: int):
        if not self._counters:
            return
        bus = get_bus()
        if not bus.is_enabled(EventCategory.COUNTER):
            # Still reset so we don't accumulate unboundedly when the
            # category is disabled.
            self._counters.clear()
            self._last_flush_cycle = cycle
            return
        for (unit_id, name), value in self._counters.items():
            bus.publish(
                CounterSnapshot(
                    cycle=cycle,
                    unit_id=unit_id,
                    counter_name=name,
                    value=value,
                    kernel_id=self._kernel_id,
                )
            )
        self._counters.clear()
        self._last_flush_cycle = cycle

    def flush(self):
        """Force a flush of any pending counters at the current max
        cycle. Called by the writer at close-time."""
        self._flush(self._max_cycle)
