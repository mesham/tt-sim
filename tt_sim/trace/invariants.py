"""Architectural invariants over the simulator event stream.

An ``Invariant`` subscribes to one or more event categories and
checks a rule on every event. On violation it can either raise (strict
mode) — useful for halting runs at the first sign of a bug — or
record into a list for post-run reporting.

The seed invariants below cover rules that are checkable from today's
events without any simulator changes:

- ``PCAlignmentInvariant`` — RV32 instructions must be 4-byte aligned.
- ``MemAlignmentInvariant`` — power-of-two-sized memory accesses must
  align to their size.
- ``LifecycleOrderInvariant`` — ``firmware_launch_done`` must follow
  ``firmware_launch_start``; same for kernel_start/done.
- ``NoCRequestResponseInvariant`` — every ``response`` should pair
  with a prior ``request`` for the same (txn_id, src/dst, txn_type).

Phase 7 of [ROADMAP §H] — the catalogue is intentionally small to
start; the framework is the thing.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from tt_sim.trace.bus import get_bus
from tt_sim.trace.events import (
    Event,
    EventCategory,
    InstrEvent,
    LifecycleEvent,
    MemEvent,
    NoCEvent,
)


@dataclass
class Violation:
    invariant: str
    message: str
    event: Event


class Invariant:
    """Base class — subclasses register themselves on construction."""

    CATEGORIES: tuple[EventCategory, ...] = ()

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.violations: list[Violation] = []
        bus = get_bus()
        for cat in self.CATEGORIES:
            bus.subscribe(cat, self._on_event)

    def _on_event(self, event: Event):
        msg = self.check(event)
        if msg is not None:
            v = Violation(invariant=type(self).__name__, message=msg, event=event)
            self.violations.append(v)
            if self.strict:
                raise AssertionError(
                    f"Invariant {v.invariant} violated: {v.message}\n  event: {v.event}"
                )

    def check(self, event: Event) -> str | None:
        """Override in subclasses. Return a message on violation, None
        otherwise."""
        raise NotImplementedError


class PCAlignmentInvariant(Invariant):
    CATEGORIES = (EventCategory.INSTR,)

    def check(self, event: InstrEvent) -> str | None:
        if event.pc & 0x3:
            return f"PC 0x{event.pc:08x} not 4-byte aligned"
        return None


class MemAlignmentInvariant(Invariant):
    CATEGORIES = (EventCategory.MEM,)

    def check(self, event: MemEvent) -> str | None:
        size = event.size
        # Only check power-of-two sizes ≤ 8 — larger transfers are
        # legitimately not size-aligned (block copies, etc.).
        if size not in (1, 2, 4, 8):
            return None
        if event.address & (size - 1):
            return (
                f"{event.op} of {size}B at 0x{event.address:x} "
                f"({event.region}) not {size}-byte aligned"
            )
        return None


class LifecycleOrderInvariant(Invariant):
    CATEGORIES = (EventCategory.LIFECYCLE,)

    # Expected ordering rules. Each entry: ``done_kind`` must be
    # preceded by ``start_kind`` (most recent).
    PAIRS = {
        "firmware_launch_done": "firmware_launch_start",
        "kernel_done": "kernel_start",
    }

    def __init__(self, strict: bool = False):
        self._seen: list[str] = []
        super().__init__(strict=strict)

    def check(self, event: LifecycleEvent) -> str | None:
        msg = None
        expected_pred = self.PAIRS.get(event.kind)
        if expected_pred is not None:
            if expected_pred not in self._seen:
                msg = f"{event.kind} without prior {expected_pred}"
            elif self._seen[-1] != expected_pred:
                msg = (
                    f"{event.kind} after {self._seen[-1]} "
                    f"(expected {expected_pred} immediately prior)"
                )
        self._seen.append(event.kind)
        return msg


class NoCRequestResponseInvariant(Invariant):
    CATEGORIES = (EventCategory.NOC,)

    def __init__(self, strict: bool = False):
        # Key: (txn_id, src, dst, txn_type) — same shape as the
        # PerfettoWriter's flow-id map. A response should find a
        # matching pending request (with src/dst swapped, since the
        # request was destination-side and the response is source-side).
        self._pending: dict[tuple, int] = {}
        super().__init__(strict=strict)

    def check(self, event: NoCEvent) -> str | None:
        if event.phase == "request":
            key = (
                event.txn_id,
                tuple(event.src),
                tuple(event.dst),
                event.txn_type,
            )
            self._pending[key] = self._pending.get(key, 0) + 1
            return None
        if event.phase == "response":
            req_key = (
                event.txn_id,
                tuple(event.dst),
                tuple(event.src),
                event.txn_type,
            )
            outstanding = self._pending.get(req_key, 0)
            if outstanding <= 0:
                return (
                    f"NoC {event.txn_type} response for txn_id={event.txn_id} "
                    f"{event.src}->{event.dst} with no matching outstanding request"
                )
            self._pending[req_key] = outstanding - 1
        return None


DEFAULT_INVARIANTS: tuple[type[Invariant], ...] = (
    PCAlignmentInvariant,
    MemAlignmentInvariant,
    LifecycleOrderInvariant,
    NoCRequestResponseInvariant,
)


class InvariantRunner:
    """Wires up a set of invariants and exposes the aggregated
    violations. The atexit hook in :mod:`tt_sim.trace.auto` calls
    :meth:`report` to dump violations and (optionally) exit non-zero.
    """

    def __init__(
        self,
        invariants: Iterable[type[Invariant]] = DEFAULT_INVARIANTS,
        strict: bool = False,
    ):
        self._strict = strict
        self._invariants: list[Invariant] = [cls(strict=strict) for cls in invariants]

    def violations(self) -> list[Violation]:
        out: list[Violation] = []
        for inv in self._invariants:
            out.extend(inv.violations)
        return out

    def report(self, path) -> int:
        """Write violations to ``path`` as JSONL. Returns the count."""
        import json
        from pathlib import Path

        violations = self.violations()
        with Path(path).open("w") as f:
            for v in violations:
                f.write(
                    json.dumps(
                        {
                            "invariant": v.invariant,
                            "message": v.message,
                            "event_category": (
                                v.event.CATEGORY.value
                                if v.event.CATEGORY is not None
                                else None
                            ),
                            "event_cycle": v.event.cycle,
                            "event_unit_id": list(v.event.unit_id),
                        }
                    )
                    + "\n"
                )
        return len(violations)
