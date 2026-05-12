"""Perfetto / Chrome Trace Event Format writer.

Subscribes to ``instr``, ``dispatch``, ``compute``, ``sync``, ``noc``,
and ``lifecycle`` events from the bus and streams them as JSON in the
shape ``ui.perfetto.dev`` ingests directly. ``mem`` events are skipped
by default — their volume swamps the UI and they don't have natural
slice semantics.

Cycles map to microseconds in the trace (``ts = cycle``, ``dur = 1``).
The sim isn't cycle-accurate today, so this is a synthetic but
faithful-enough rendering; once §I cycle accuracy lands, durations
become real and the writer needs no schema change.

Events with ``cycle == 0`` (today: ``mem``, ``sync``, ``lifecycle``)
are stamped with the highest cycle seen so far, so they appear at the
"current time" of the run rather than collapsing onto t=0.

Output is gzip-compressed when the path ends in ``.gz`` — Perfetto
loads ``.json.gz`` natively and traces typically compress 10-20×.
"""

import gzip
import json
from pathlib import Path
from typing import IO

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import Event, EventCategory


class PerfettoWriter:
    """Streaming writer for Chrome Trace Event Format JSON."""

    def __init__(self, path: Path | str, bus: EventBus | None = None):
        self._path = Path(path)
        if str(self._path).endswith(".gz"):
            self._file: IO = gzip.open(self._path, "wt")
        else:
            self._file = self._path.open("w")
        self._first = True
        self._max_cycle = 0
        self._pid_for_tile: dict[tuple, int] = {}
        self._tid_for_unit: dict[tuple, int] = {}
        self._next_pid = 1
        self._next_tid = 1
        # NoC transaction IDs are reused across the run, so we mint a
        # unique per-emission flow id and pair (txn_id, src/dst) →
        # flow id when the response arrives.
        self._next_flow_id = 1
        self._pending_flows: dict[tuple, int] = {}
        self._file.write('{"displayTimeUnit":"ns","traceEvents":[')
        self._bus = bus if bus is not None else get_bus()
        for cat in (
            EventCategory.INSTR,
            EventCategory.DISPATCH,
            EventCategory.COMPUTE,
            EventCategory.SYNC,
            EventCategory.NOC,
            EventCategory.LIFECYCLE,
        ):
            self._bus.subscribe(cat, self._on_event)

    def _pid_tid(self, unit_id: tuple) -> tuple[int, int]:
        chip, y, x, unit = unit_id
        tile_key = (chip, y, x)
        pid = self._pid_for_tile.get(tile_key)
        if pid is None:
            pid = self._next_pid
            self._next_pid += 1
            self._pid_for_tile[tile_key] = pid
            self._emit_raw(
                {
                    "ph": "M",
                    "name": "process_name",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": f"Tile chip={chip} ({y},{x})"},
                }
            )
        tid = self._tid_for_unit.get(unit_id)
        if tid is None:
            tid = self._next_tid
            self._next_tid += 1
            self._tid_for_unit[unit_id] = tid
            self._emit_raw(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": unit},
                }
            )
        return pid, tid

    def _emit_raw(self, event_dict: dict):
        if not self._first:
            self._file.write(",")
        self._file.write(json.dumps(event_dict, separators=(",", ":")))
        self._first = False

    def _on_event(self, event: Event):
        cycle = event.cycle
        if cycle > self._max_cycle:
            self._max_cycle = cycle
        # cycle=0 means "no clock-tick context" (mem / sync / lifecycle);
        # stamp those at the most recent observed cycle so they don't all
        # pile up at t=0.
        ts = cycle if cycle > 0 else self._max_cycle

        pid, tid = self._pid_tid(event.unit_id)
        cat = event.CATEGORY
        if cat is EventCategory.INSTR:
            self._emit_raw(
                {
                    "ph": "X",
                    "name": "stall" if event.stalled else f"pc={hex(event.pc)}",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": 1,
                    "args": {
                        "pc": event.pc,
                        "instruction": event.instruction,
                        "stalled": event.stalled,
                    },
                }
            )
        elif cat is EventCategory.DISPATCH:
            self._emit_raw(
                {
                    "ph": "X",
                    "name": f"dispatch:{event.opcode}",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": 1,
                    "args": {
                        "opcode": event.opcode,
                        "target_unit": event.target_unit,
                        "thread_id": event.thread_id,
                    },
                }
            )
        elif cat is EventCategory.COMPUTE:
            self._emit_raw(
                {
                    "ph": "X",
                    "name": event.op,
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": 1,
                    "args": {
                        "target_unit": event.target_unit,
                        "thread_id": event.thread_id,
                    },
                }
            )
        elif cat is EventCategory.SYNC:
            self._emit_raw(
                {
                    "ph": "i",
                    "name": event.kind,
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "s": "t",
                    "args": {"detail": event.detail},
                }
            )
        elif cat is EventCategory.LIFECYCLE:
            self._emit_raw(
                {
                    "ph": "i",
                    "name": event.kind,
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "s": "g",
                    "args": {"detail": event.detail},
                }
            )
        elif cat is EventCategory.NOC:
            self._emit_raw(
                {
                    "ph": "X",
                    "name": f"noc:{event.phase}:{event.txn_type}",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": 1,
                    "args": {
                        "phase": event.phase,
                        "txn_type": event.txn_type,
                        "src": list(event.src),
                        "dst": list(event.dst),
                        "size_bytes": event.size_bytes,
                        "txn_id": event.txn_id,
                    },
                }
            )
            # Flow events link request→response. Key on
            # (txn_id, src, dst, txn_type) so the response can find
            # the matching request even when txn_id is reused.
            flow_key = (
                event.txn_id,
                tuple(event.src),
                tuple(event.dst),
                event.txn_type,
            )
            if event.phase == "request":
                flow_id = self._next_flow_id
                self._next_flow_id += 1
                self._pending_flows[flow_key] = flow_id
                self._emit_raw(
                    {
                        "ph": "s",
                        "name": f"noc_txn_{event.txn_id}",
                        "cat": "noc",
                        "id": flow_id,
                        "pid": pid,
                        "tid": tid,
                        "ts": ts,
                    }
                )
            elif event.phase == "response":
                # The response's src/dst are swapped relative to the
                # original request, so flip them for the lookup.
                req_key = (
                    event.txn_id,
                    tuple(event.dst),
                    tuple(event.src),
                    event.txn_type,
                )
                flow_id = self._pending_flows.pop(req_key, None)
                if flow_id is not None:
                    self._emit_raw(
                        {
                            "ph": "f",
                            "name": f"noc_txn_{event.txn_id}",
                            "cat": "noc",
                            "id": flow_id,
                            "pid": pid,
                            "tid": tid,
                            "ts": ts,
                            "bp": "e",
                        }
                    )

    def close(self):
        if not self._file.closed:
            self._file.write("]}")
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
