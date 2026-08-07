"""Per-PC cycle attribution — the raw material for the ranked report.

Subscribes to :class:`~tt_sim.trace.events.InstrEvent` and accumulates,
per ``(unit, pc)``, how many instructions retired there and how many
cycles were spent stalled before them, split by reason. DWARF resolution
happens **once, at close**, over the few thousand distinct PCs a run
touches — never per event. A ``DwarfIndex`` lookup on the retirement
path would be a dict hit per simulated instruction on a tree whose
tracing overhead is already over budget; the whole point of keying on
the raw PC is that the expensive half is post-processing.

Cost on the hot path is one ``defaultdict[int]`` increment per retired
instruction, plus a second only when the instruction stalled.

**Stall reasons are read, never enumerated.** The reason string comes
straight off the event, so a reason added to
``tt_sim.pe.rv.cost.STALL_REASON_NAMES`` — or a new one surfaced by any
other work in flight — appears in the output with no change here and no
change in the report. The same is true of ``instr_stalled``, the Tensix
front-end back-pressure flag, which is counted separately because it is
a different mechanism from an RV scoreboard stall and exists in both
timing regimes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import EventCategory, InstrEvent

#: Label used when the cost model is off, so every retired instruction is
#: one cycle and no reason string exists.
ISSUE = "issue"


@dataclass
class Hotspot:
    """One resolved (unit, pc) row."""

    unit: str
    pc: int
    retired: int
    stall_cycles: int
    by_reason: dict[str, int]
    frontend_stalls: int = 0
    function: str = ""
    file: str = ""
    line: int = 0

    @property
    def cycles(self) -> int:
        """Cycles this PC is charged: one per retirement, plus everything
        the core waited before issuing it. A floor, like every number the
        cost model produces."""
        return self.retired + self.stall_cycles

    @property
    def resolved(self) -> bool:
        return bool(self.file)

    def location(self) -> str:
        if not self.file:
            return f"pc=0x{self.pc:x}"
        where = f"{self.file}:{self.line}"
        return f"{self.function} ({where})" if self.function else where


@dataclass
class HotspotTable:
    rows: list[Hotspot] = field(default_factory=list)
    #: Units that published instructions but had no ELF loaded for them.
    unattributed_units: list[str] = field(default_factory=list)

    def total_cycles(self) -> int:
        return sum(r.cycles for r in self.rows)

    def resolved_cycles(self) -> int:
        return sum(r.cycles for r in self.rows if r.resolved)

    def ranked(self, limit: int | None = None) -> list[Hotspot]:
        out = sorted(self.rows, key=lambda r: (-r.cycles, r.unit, r.pc))
        return out[:limit] if limit else out

    def by_function(self) -> list[Hotspot]:
        """Rows folded to one per (unit, function), for a report that wants
        to name a routine rather than an instruction."""
        merged: dict[tuple[str, str], Hotspot] = {}
        for row in self.rows:
            key = (row.unit, row.function or f"pc=0x{row.pc:x}")
            head = merged.get(key)
            if head is None:
                merged[key] = Hotspot(
                    unit=row.unit,
                    pc=row.pc,
                    retired=row.retired,
                    stall_cycles=row.stall_cycles,
                    by_reason=dict(row.by_reason),
                    frontend_stalls=row.frontend_stalls,
                    function=row.function,
                    file=row.file,
                    line=row.line,
                )
                continue
            head.retired += row.retired
            head.stall_cycles += row.stall_cycles
            head.frontend_stalls += row.frontend_stalls
            for reason, value in row.by_reason.items():
                head.by_reason[reason] = head.by_reason.get(reason, 0) + value
        return sorted(merged.values(), key=lambda r: (-r.cycles, r.unit))


class HotspotAggregator:
    def __init__(self, bus: EventBus | None = None):
        self._retired: dict[tuple[str, int], int] = defaultdict(int)
        self._stalls: dict[tuple[str, int, str], int] = defaultdict(int)
        self._frontend: dict[tuple[str, int], int] = defaultdict(int)
        self._bus = bus if bus is not None else get_bus()
        self._bus.subscribe(EventCategory.INSTR, self._on_instr)

    def _on_instr(self, event: InstrEvent):
        unit = event.unit_id[-1]
        self._retired[(unit, event.pc)] += 1
        if event.stall_cycles:
            self._stalls[(unit, event.pc, event.stall_reason or ISSUE)] += (
                event.stall_cycles
            )
        if event.stalled:
            self._frontend[(unit, event.pc)] += 1

    def resolve(self, index=None) -> HotspotTable:
        """Fold the raw counts into a table, attributing each PC through
        ``index`` (a :class:`~tt_sim.trace.dwarf.DwarfIndex`) if given."""
        reasons: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
        for (unit, pc, reason), value in self._stalls.items():
            reasons[(unit, pc)][reason] = value

        table = HotspotTable()
        units_with_elf = set(index.units()) if index is not None else set()
        seen_units: set[str] = set()
        for (unit, pc), retired in self._retired.items():
            seen_units.add(unit)
            by_reason = reasons.get((unit, pc), {})
            row = Hotspot(
                unit=unit,
                pc=pc,
                retired=retired,
                stall_cycles=sum(by_reason.values()),
                by_reason=by_reason,
                frontend_stalls=self._frontend.get((unit, pc), 0),
            )
            if index is not None:
                loc = index.nearest(pc, unit=unit)
                if loc is not None:
                    row.file = loc.file
                    row.line = loc.line
                    # Resolved against the executed PC rather than against the
                    # line-table row below it, so an inlined callee is named
                    # even when its entry shares a row with its caller. Once
                    # per distinct PC, not per retirement.
                    row.function = loc.function or index.function_at(pc, unit=unit)
            table.rows.append(row)
        table.unattributed_units = sorted(seen_units - units_with_elf)
        return table
