"""Perfetto / Chrome Trace Event Format writer.

Subscribes to ``instr``, ``dispatch``, ``compute``, ``sync``, ``noc``,
``stall`` and ``lifecycle`` events from the bus and streams them as JSON
in the shape ``ui.perfetto.dev`` ingests directly. ``mem`` events are
skipped by default — their volume swamps the UI and they don't have
natural slice semantics.

Cycles map to microseconds in the trace (``ts = cycle``).

**Durations, and which timing regime produced them.** A slice's width
is only ever a number the simulator actually holds:

* ``compute`` slices are as wide as the occupancy
  ``tensix_instruction_costs.yaml`` charged the op
  (``ComputeEvent.duration``).
* ``noc`` slices span issue → arrival, from ``NoCEvent.issue_cycle``
  and the event's own cycle. These are emitted as **async** slices
  (``ph`` ``b``/``e``) rather than ``X``, because two packets can be
  in flight on one NIU at once and partially overlapping ``X`` slices
  are not legally nestable. The request→response arrows ride on the
  slices as ``bind_id`` + ``flow_out``/``flow_in``: a standalone
  ``s``/``f`` flow event binds to the enclosing slice of a *thread*
  track, which an async slice is not, and Perfetto would report every
  one of them as ``flow_no_enclosing_slice``.
* ``instr`` slices are one cycle wide — a baby RISC-V core retires one
  instruction per cycle — preceded by a ``stall:<reason>`` slice as
  wide as the cycles the cost model held that instruction
  (``InstrEvent.stall_cycles``).
* ``dispatch`` slices are one cycle wide in both regimes: issuing to a
  backend unit is a single-cycle act.
* ``stall`` slices span a whole episode — the contiguous cycles a Tensix
  thread was held for one named reason (``StallEvent.cycles``) — and are
  drawn on a ``<thread> stall`` track of their own, beside the track
  carrying what that thread did run. The slice name is
  ``stall:<reason>-><unit>``, so the *cause* is legible without opening
  the args.

**With ``TT_SIM_COST_MODEL`` unset all of that collapses to one cycle**
(a NoC flight to the one or two cycles the two-list swap in
``NUI.clock_tick`` really costs), and that is the truthful rendering
rather than a degraded one: with the model off nothing stalls and every
unit retires in the tick it was issued, so one cycle *is* the width.
What the writer will not do is invent a plausible-looking number to
fill the gap. Which regime a trace was captured in is recorded in three
places so a reader can never be in doubt: the ``cost_model`` key of the
trailing ``otherData`` block, a ``process_labels`` metadata event on
every tile, and the ``timing_model`` argument on every slice that
carries an unmodelled width.

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

from tt_sim.perf.model import cost_model_enabled
from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import Event, EventCategory


class PerfettoWriter:
    """Streaming writer for Chrome Trace Event Format JSON."""

    def __init__(self, path: Path | str, bus: EventBus | None = None):
        self._path = Path(path)
        # Read once, at open: the cost model is a construction-time decision
        # (``tt_sim.perf.model.cost_model_enabled``), so a trace is captured
        # wholly in one regime or the other.
        self._cost_model = cost_model_enabled()
        self._regime = "on" if self._cost_model else "off"
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
        # unique per-emission bind id and pair (txn_id, src/dst) →
        # bind id when the response arrives.
        self._next_flow_id = 1
        self._pending_flows: dict[tuple, int] = {}
        # Async-slice ids, from a separate counter to the flow ids so the two
        # id spaces cannot collide inside Perfetto's legacy JSON importer.
        self._next_slice_id = 1
        self._file.write('{"displayTimeUnit":"ns","traceEvents":[')
        self._bus = bus if bus is not None else get_bus()
        for cat in (
            EventCategory.INSTR,
            EventCategory.DISPATCH,
            EventCategory.COMPUTE,
            EventCategory.SYNC,
            EventCategory.NOC,
            EventCategory.LIFECYCLE,
            EventCategory.STALL,
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
            # Regime label, per tile, so the answer to "are these durations
            # modelled?" is visible in the UI next to the process rather than
            # only in the file trailer.
            self._emit_raw(
                {
                    "ph": "M",
                    "name": "process_labels",
                    "pid": pid,
                    "tid": 0,
                    "args": {"labels": f"TT_SIM_COST_MODEL {self._regime}"},
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
            # The cycles the core was held before this instruction issued get
            # their own slice, immediately preceding it and exactly abutting
            # the previous instruction's — an in-order core stalled from the
            # cycle after its last retirement. Only ever non-zero with the
            # cost model on.
            args = {
                "pc": event.pc,
                "instruction": event.instruction,
                "stalled": event.stalled,
            }
            if event.stall_cycles > 0:
                self._emit_raw(
                    {
                        "ph": "X",
                        "name": f"stall:{event.stall_reason}",
                        "pid": pid,
                        "tid": tid,
                        "ts": ts - event.stall_cycles,
                        "dur": event.stall_cycles,
                        "args": {
                            "stall_reason": event.stall_reason,
                            "stall_cycles": event.stall_cycles,
                            "blocked_pc": event.pc,
                        },
                    }
                )
                # Only when there was one. An always-present ``"stall_cycles":0``
                # is ~16 bytes on the highest-volume event in the trace, and on
                # a run with the cost model off it would be 16 bytes of zero on
                # every one of them — measured at +9 MB on a 66 MB `four` trace
                # (+13.6 %), for nothing a reader can use.
                args["stall_cycles"] = event.stall_cycles
            self._emit_raw(
                {
                    "ph": "X",
                    "name": "stall" if event.stalled else f"pc={hex(event.pc)}",
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": 1,
                    "args": args,
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
            # ``duration == 0`` is the cost tables saying "no claim" — the
            # unit is unwired, the opcode uncosted, or the model off. Render
            # it at one cycle (what the simulator did) and say so in the args
            # rather than leaving the reader to guess which.
            modelled = event.duration > 0
            args = {
                "target_unit": event.target_unit,
                "thread_id": event.thread_id,
            }
            if modelled:
                args["occupancy_cycles"] = event.duration
            else:
                args["timing_model"] = f"unmodelled (TT_SIM_COST_MODEL {self._regime})"
            self._emit_raw(
                {
                    "ph": "X",
                    "name": event.op,
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "dur": event.duration if modelled else 1,
                    "args": args,
                }
            )
        elif cat is EventCategory.STALL:
            # Its own track, not the thread's instruction track. A stall spans
            # many cycles and the instruction slices beneath it are one cycle
            # each; two partially overlapping ``X`` slices on one track are not
            # legally nestable and Perfetto drops them. On a track of its own
            # the episodes cannot overlap each other either — the wait gate's
            # stall branches are mutually exclusive, so a thread is in at most
            # one episode at a time — and the result reads as a "why is this
            # thread not progressing" lane beside "what it ran".
            chip, y, x, unit = event.unit_id
            _, stall_tid = self._pid_tid((chip, y, x, f"{unit} stall"))
            name = (
                f"stall:{event.reason}->{event.blocked_on}"
                if event.blocked_on
                else f"stall:{event.reason}"
            )
            args = {
                "stall_reason": event.reason,
                "stall_cycles": event.cycles,
                "thread_id": event.thread_id,
                # Modelled floors, never calibrated absolutes. Repeated on the
                # slice because a stall width is the number a reader is most
                # tempted to quote out of context.
                "timing_model": (
                    "modelled floor from published bounds, "
                    f"not calibrated to silicon (TT_SIM_COST_MODEL {self._regime})"
                ),
            }
            if event.blocked_on:
                args["blocked_on"] = event.blocked_on
            if event.opcode:
                args["blocked_opcode"] = event.opcode
            if event.semaphore >= 0:
                args["semaphore"] = event.semaphore
            self._emit_raw(
                {
                    "ph": "X",
                    "name": name,
                    "pid": pid,
                    "tid": stall_tid,
                    "ts": ts,
                    "dur": event.cycles,
                    "args": args,
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
            # A NoC packet's slice spans its flight: issue at the sending NIU
            # to arrival at this one. Emitted as an *async* pair rather than a
            # single ``X`` because a NIU can have several packets in flight
            # simultaneously, and partially overlapping ``X`` slices on one
            # track are not legally nestable — Perfetto drops them.
            issue = event.issue_cycle
            if issue < 0 or issue > ts:
                # No owning tile clock at transmit time, so the flight is
                # untimed. One cycle, and labelled as such.
                start, flight, timed = ts, 1, False
            else:
                start, flight, timed = issue, max(1, ts - issue), True
            slice_id = self._next_slice_id
            self._next_slice_id += 1
            args = {
                "phase": event.phase,
                "txn_type": event.txn_type,
                "src": list(event.src),
                "dst": list(event.dst),
                "size_bytes": event.size_bytes,
                "txn_id": event.txn_id,
                "flight_cycles": flight,
                "issue_cycle": start,
                "arrival_cycle": ts,
            }
            if not timed:
                args["timing_model"] = "untimed flight (no clock at transmit)"
            begin = {
                "ph": "b",
                "name": f"noc:{event.phase}:{event.txn_type}",
                "cat": "noc",
                "id": slice_id,
                "pid": pid,
                "tid": tid,
                "ts": start,
                "args": args,
            }
            # Request→response arrows. Carried as ``bind_id`` +
            # ``flow_out``/``flow_in`` on the slices themselves rather than as
            # separate ``s``/``f`` events, because a standalone flow event
            # binds to the enclosing slice of a *thread* track and these
            # slices are async — Perfetto would report every one of them as
            # ``flow_no_enclosing_slice`` and draw no arrow at all. Keyed on
            # (txn_id, src, dst, txn_type) so a response finds its request
            # even though kernels reuse one txn_id for many transactions.
            if event.phase == "request":
                flow_id = self._next_flow_id
                self._next_flow_id += 1
                self._pending_flows[
                    (
                        event.txn_id,
                        tuple(event.src),
                        tuple(event.dst),
                        event.txn_type,
                    )
                ] = flow_id
                begin["bind_id"] = flow_id
                begin["flow_out"] = True
            elif event.phase == "response":
                # The response's src/dst are swapped relative to the
                # original request, so flip them for the lookup.
                flow_id = self._pending_flows.pop(
                    (
                        event.txn_id,
                        tuple(event.dst),
                        tuple(event.src),
                        event.txn_type,
                    ),
                    None,
                )
                if flow_id is not None:
                    begin["bind_id"] = flow_id
                    begin["flow_in"] = True
            self._emit_raw(begin)
            self._emit_raw(
                {
                    "ph": "e",
                    "name": f"noc:{event.phase}:{event.txn_type}",
                    "cat": "noc",
                    "id": slice_id,
                    "pid": pid,
                    "tid": tid,
                    "ts": start + flight,
                }
            )

    def close(self):
        if not self._file.closed:
            # ``otherData`` is the Chrome Trace Event Format's metadata slot;
            # Perfetto surfaces it under "Info and stats". It is what stops a
            # trace being ambiguous about where its widths came from once it
            # has been copied away from the run that produced it.
            trailer = {
                "cost_model": self._cost_model,
                "timing_note": (
                    "durations are modelled cycle counts"
                    if self._cost_model
                    else "TT_SIM_COST_MODEL unset: nothing stalls and every "
                    "unit retires in its issue tick, so every duration is a "
                    "real 1 cycle, not a modelled figure"
                ),
                "cycle_unit": "1 trace microsecond == 1 simulated cycle",
            }
            self._file.write(
                '],"otherData":' + json.dumps(trailer, separators=(",", ":")) + "}"
            )
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
