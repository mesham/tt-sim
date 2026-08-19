"""Rung 4's NoC-bound leg: tt-sim's NoC timing against a card's, by mechanism.

The sibling of :mod:`tt_sim.perf.stall_attribution`, and deliberately the same
shape. Where that leg reads Tensix hardware stall counters out of
``profile_log_device.csv``, this one reads tt-metal's **NoC event trace** --
the artefact ``TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`` produces -- and compares
the two sides' *decomposition* of a data-movement core's span, not just its
total.

What the artefact is
--------------------

``TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`` (``tt_metal/llrt/rtoptions.cpp:852``)
sets ``profiler_noc_events_enabled`` and force-enables the device profiler, and
the JIT build then injects ``-DPROFILE_NOC_EVENTS=1`` into every kernel compile
(``tt_metal/jit_build/build.cpp:188``). The device-side recorder
(``tt_metal/tools/profiler/noc_event_profiler.hpp``) writes an 8-byte
``KernelProfilerNocEventMetadata`` (``tt_metal/tools/profiler/event_metadata.hpp:13``)
into the ordinary per-RISC profiler L1 vector -- the *same* buffer the zone
profiler uses, not a separate one -- alongside a wall-clock timestamp read from
``RISCV_DEBUG_REG_WALL_CLOCK`` (``kernel_profiler.hpp:178``). Host-side,
``DeviceProfiler::convertNocTracePacketsToJson``
(``tt_metal/impl/profiler/profiler.cpp:756``) and ``dumpJsonNocTraces``
(``profiler.cpp:1090``) write one JSON array per (device, program execution) to
``.logs/noc_trace_dev<d>[_<op>]_ID<n>.json``.

**Every transaction event is stamped at issue**, before the command buffer is
even written (``dataflow_api.h:518`` for ``noc_async_read``, ``:835`` for
``noc_async_write``, and ~40 more call sites, all with the macro ahead of the
issue). There is **no completion timestamp per transaction**. The only
completion-side timing in the whole artefact is the pair
``READ_BARRIER_START``/``READ_BARRIER_END`` (``dataflow_api.h:1738``, ``:1751``)
and its write twin (``:1768``, ``:1781``), which bracket the
``ncrisc_noc_reads_flushed`` spin.

That fact shapes everything below, and it is worth stating plainly because the
roadmap's phrasing for this leg -- "agreement with ``noc_flight_cycles`` plus
queueing" -- presupposes a per-packet hardware flight time that **does not
exist**. See :func:`internal_attribution` for what is done about that instead.

Recording is BRISC/NCRISC only (``noc_event_profiler.hpp:7``) and firmware
traffic is excluded (``brisc.cc:10`` ``#undef PROFILE_NOC_EVENTS``), so the
streams are exactly the two data-movement kernels.

The partition
-------------

Per ``(core, RISC)``, the kernel zone is cut at every recorded event and each
interval is attributed by **the event that opened it**:

===============  =========================================================
bucket           the interval from ...
===============  =========================================================
``prologue``     ``ZONE_START`` to the first NoC event -- **not a setup cost**,
                 see below
``issue``        a transaction-issuing event to the next event
``read_wait``    ``READ_BARRIER_START`` to ``READ_BARRIER_END``
``write_wait``   ``WRITE_BARRIER_START`` to ``WRITE_BARRIER_END``
``other_wait``   a single-event blocking call (``FULL_BARRIER``,
                 ``SEMAPHORE_WAIT``, ``WRITE_FLUSH``, ...) to the next event
``local``        a barrier END to the next event -- the core's own work
``profiler``     a ``PROFILER-NOC-QUICK-PUSH`` ZONE_START to the next event
                 outside that push -- **the instrument, not the kernel**
===============  =========================================================

The buckets telescope to ``ZONE_END - ZONE_START`` **by construction**, which
is what makes both sides' denominators the same quantity and keeps the
``E_total <= E_int`` inequality (and therefore the compensation ratio) meaning
something. It also makes :func:`gate_partition_closes` a live test rather than
a formality: a negative interval means the timestamps are not monotonic, which
the 44-bit wall clock and its non-atomic high/low read pair can genuinely
produce.

One kernel zone, and the profiler's own zone nested inside it
------------------------------------------------------------

A stream's zone endpoints are **not** all launches. tt-metal wraps the kernel
in ``BRISC-KERNEL`` / ``NCRISC-KERNEL``, and *nested inside that* the device
profiler emits its own ``PROFILER-NOC-QUICK-PUSH`` pair every time the per-RISC
L1 marker buffer fills and is flushed to DRAM. ``recordNocEvent`` calls
``flush_to_dram_if_full`` **before** it records, so on a kernel that issues more
transactions than the buffer holds the push repeats through the run.

This is why this leg used to refuse real workloads. ``gate_single_window``
counted *every* ``ZONE_START`` and read "5 starts" as five program executions,
when the stream held one kernel window and four buffer flushes. Reported by the
compiler team on 2026-08-19: a 16-core Blackhole ``gemm_256_check`` pair, every
one of the 32 streams refused as ``single_window``; all 32 in fact carry exactly
one ``*-KERNEL`` pair, with the kernel ``ZONE_START`` the first record and the
``ZONE_END`` the last, no record outside it, and 0-4 strictly nested pushes.
The window was never in doubt; the gate was reading the instrument.

**The push is charged to its own bucket rather than absorbed.** A push costs
~190 cycles on that capture -- ~24 for the two markers and ~165 for the DMA and
its posted-write flush -- and it is instrumentation the capture added, not work
the kernel asked for. Folding it into ``issue`` (which is where it would land,
since a push is triggered from inside the issue path) would charge the *model*
for the difference between how tt-sim and a card execute the profiler's own
DRAM write. That is the mirror of the argument against a ``pre_first_event_wait``
bucket below: this one belongs in the partition precisely because **both** sides
can fill it, from the same firmware doing the same thing.

The bucket runs from the push's ``ZONE_START`` to the next event outside the
push, because the ``ZONE_END`` marker is written *before* the flush is issued
(``kernel_profiler.hpp:414`` marks the end; the ``profiler_noc_async_write_posted``
and ``profiler_noc_async_flush_posted_write`` are at ``:455``). It is therefore
an **upper** bound: on the ordinary path the next event is the very transaction
whose recording triggered the push, but a push from ``quick_push_if_linked``
could return into kernel work before the next recorded event. On the reference
capture the interval is 161-166 cycles in all 112 pushes, which is the flush and
nothing else.

Any *other* nested zone -- a ``DeviceZoneScopedN`` a consumer put in their own
dataflow kernel -- is passed through transparently: its endpoints open no
bucket, the intervals telescope across it, and the time stays in whichever
bucket the enclosing kernel work opened, which is where it belongs. The report
names any such zone in a note so it is not a silent pass-through.

``prologue`` is unattributed time, not setup time
------------------------------------------------

This is the one bucket whose name flatters it, and the correction belongs next
to the definition rather than in a footnote. ``prologue`` is defined purely
mechanically -- ``ZONE_START`` to the *first NoC event* -- and **anything at all
the core did in that interval lands in it**, including a blocking wait.

The reason is structural and cannot be fixed on this side. Every bucket except
this one is opened by a recorded event, and a wait that emits no event therefore
has no interval to attach to. ``cb_wait_front`` / ``cb_reserve_back`` emit
nothing: the artefact carries exactly ``ZONE_START``/``ZONE_END`` and the
``NocEventType`` records, and a circular-buffer wait is neither. So a kernel
whose body is "read four ``get_arg_val`` arguments, ``cb_wait_front`` for the
compute kernel's first output tile, then ``noc_async_write``" spends its
entire dependency wait inside ``prologue``, indistinguishable from address
arithmetic.

Reported by this leg's first external consumer on a 4-core ``gemm_128``,
2026-08-18: a writer kernel with ``prologue`` at **91.89 %** of a 6,932-cycle
span. No semaphore anywhere in the build; it was ``cb_wait_front(2, 1)`` waiting
on the compute pipeline.

**Reader/writer asymmetry: ``prologue`` is not comparable across cores.** Where
a core sits in the pipeline decides what its ``prologue`` contains. A reader at
the head has nothing upstream of it -- a few argument reads and a
``cb_reserve_back`` on an empty CB that cannot block -- so its ``prologue``
really is setup, tens of cycles. A writer at the tail cannot start until its
producer has produced, so its ``prologue`` is that producer's whole latency.
Ranking cores by ``prologue``, or summing it across a grid, therefore adds
quantities that are not the same quantity.

Read ``prologue`` as **"time before the first NoC event, cause unknown"**. It
is an *upper* bound on setup and nothing more; a large value is evidence of a
dependency, not of expensive initialisation. :data:`PROLOGUE_DOMINATES` is the
share above which the report says so out loud, per stream.

**No tt-metal flag recovers the split for these streams.** Checked against
tt-metal 0.74 rather than assumed. The dataflow ``cb_wait_front`` /
``cb_reserve_back`` (``tt_metal/hw/inc/api/dataflow/dataflow_api.h:473``,
``:403``) carry only a watcher ``WAYPOINT`` -- a four-character string stored
into a mailbox slot and overwritten by the next waypoint, with no timestamp and
no path into any artefact -- and ``NocEventType``
(``tools/profiler/event_metadata.hpp:15``) has no CB member at all. The
comparison worth drawing is ``noc_semaphore_wait`` (``dataflow_api.h:1929``),
which *does* record ``SEMAPHORE_WAIT``, which is why ``other_wait`` exists and
``prologue`` cannot be split the same way. The one opt-in that measures a CB
wait, ``TT_METAL_PROFILER_SUM=1``, instruments the **compute** side only
(``DeviceZoneScopedSumN1("CB-COMPUTE-WAIT-FRONT")`` in
``hw/ckernels/*/metal/llk_io/llk_io_unpack.h``), emits a single accumulated
cycle *total* per RISC per op into ``profile_log_device.csv`` with the op's
timestamp rather than the wait's, and is dropped outright by
``convertNocTracePacketsToJson`` (it is a ``ZONE_TOTAL``, which that filter does
not admit). So it neither reaches this artefact nor describes the interval in
question.

**Why there is no ``pre_first_event_wait`` bucket splitting the two.** Not for
want of asking -- the consumer suggested exactly that. It is refused because
the split is not recoverable *on the card*, and a bucket only the simulator can
fill would silently break the comparison this module exists for: the card's
share of it would be zero by construction and ``E_int`` would charge the model
for an artefact of instrumentation. tt-sim does know more here (a baby core
polling L1 is recognised by :mod:`tt_sim.pe.rv.spin`), but that knowledge has no
counterpart in ``noc_trace_*.json``, so it stays out of the partition.

The criterion is the same as the sibling leg's, with this leg's own stated
tolerance:

    ``E_total = |sum c_sim - sum c_hw| / sum c_hw``
    ``E_int   = sum |c_m,sim - c_m,hw| / sum c_hw``

Both must be within **25 %**. The roadmap names one number for this leg, so one
number is applied; inventing a tighter envelope bar here would be a bar with no
provenance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

#: This leg's stated tolerance, from ROADMAP §4 point 4. Applied to ``E_total``,
#: to ``E_int`` and to every transaction class's mean observed latency.
TOLERANCE = 0.25

#: Share of a stream's span above which ``prologue`` is called out as a probable
#: dependency wait rather than setup. A heuristic for the *warning* only -- the
#: caveat under the table is unconditional, because a small ``prologue`` is not
#: proof of setup either, only a smaller upper bound on it. A quarter of a
#: data-movement kernel's span spent before its first NoC transaction is not
#: plausibly address arithmetic, which is what makes it worth a line of its own.
PROLOGUE_DOMINATES = 0.25

#: What ``prologue`` actually contains, said wherever the number is shown. See
#: the module docstring for the full argument.
PROLOGUE_CAVEAT = (
    "prologue is ZONE_START to the first NoC event -- NOT a setup cost. A",
    "blocking wait that emits no event (cb_wait_front / cb_reserve_back on a",
    "producer's output) has no bucket of its own and lands here. A core at the",
    "END of a pipeline shows its upstream dependency as prologue, so prologue",
    "is not comparable across cores. Read it as time before the first NoC",
    "event, cause unknown.",
)

#: Buckets, in report order. See the module docstring for what opens each.
MECHANISMS = (
    "prologue",
    "issue",
    "read_wait",
    "write_wait",
    "other_wait",
    "local",
    "profiler",
)

#: Suffix of the zone tt-metal wraps a data-movement kernel in --
#: ``BRISC-KERNEL`` / ``NCRISC-KERNEL``. This, and not "any zone endpoint", is
#: what bounds the span: a stream can carry *nested* zones as well (see
#: :data:`PROFILER_PUSH_ZONE`), and counting those as launches is what made this
#: leg refuse every multi-core capture put to it.
KERNEL_ZONE_SUFFIX = "-KERNEL"

#: The device profiler's own zone, emitted by ``kernel_profiler::quick_push()``
#: (``tt_metal/tools/profiler/kernel_profiler.hpp:409``) when the per-RISC L1
#: marker buffer fills mid-kernel and has to be flushed to DRAM.
#:
#: **It is in the artefact by an upstream oversight, not by design.**
#: ``convertNocTracePacketsToJson`` (``impl/profiler/profiler.cpp:780``) filters
#: zone endpoints and explicitly excludes ``"PROFILER-NOC-QUICK-SEND"`` -- a name
#: **no device code emits**. It is the only occurrence of that string in the
#: tt-metal tree; the zone was renamed to ``PROFILER-NOC-QUICK-PUSH`` and the
#: filter did not follow. So the profiler's own flush zone leaks into every
#: ``noc_trace_*.json`` whose kernel fills the buffer, which is every capture
#: past roughly 120 transactions on a RISC (a 512-uint32 vector, wIndex
#: resetting to CUSTOM_MARKERS=12, DISPATCH_HEADROOM_SIZE=16, 4 uint32 per
#: record; our own capture's first flush is at record 122).
PROFILER_PUSH_ZONE = "PROFILER-NOC-QUICK-PUSH"

#: ``NocEventType`` names (``event_metadata.hpp:15``) that open a *read* wait
#: and close it. Only these two carry completion-side timing.
READ_BARRIER_START = "READ_BARRIER_START"
READ_BARRIER_END = "READ_BARRIER_END"
WRITE_BARRIER_START = "WRITE_BARRIER_START"
WRITE_BARRIER_END = "WRITE_BARRIER_END"

BARRIER_STARTS = {READ_BARRIER_START, WRITE_BARRIER_START}
BARRIER_ENDS = {READ_BARRIER_END, WRITE_BARRIER_END}

#: Single-event blocking calls: the macro is placed *before* the spin and there
#: is no END, so the interval they open is a wait whose end is only bounded by
#: the next recorded event. Kept out of ``issue`` because calling a blocking
#: wait "issue work" would quietly move real stall time into the wrong bucket.
BLOCKING_SINGLE = {
    "READ_BARRIER_WITH_TRID",
    "WRITE_BARRIER_WITH_TRID",
    "WRITE_FLUSH",
    "WRITE_FLUSH_WITH_TRID",
    "FULL_BARRIER",
    "ATOMIC_BARRIER",
    "SEMAPHORE_WAIT",
}

#: Issue events that a read barrier will later complete. ``*_SET_STATE`` and
#: ``*_SET_TRID`` are excluded on purpose -- they configure a command buffer and
#: put no packet on the wire, so pairing them with a barrier would invent a
#: transaction and drag the mean latency down.
READ_ISSUES = {
    "READ",
    "READ_WITH_STATE",
    "READ_WITH_STATE_AND_TRID",
    "READ_DRAM_SHARDED_WITH_STATE",
}
WRITE_ISSUES = {
    "WRITE_",
    "WRITE_WITH_TRID",
    "WRITE_INLINE",
    "WRITE_MULTICAST",
    "WRITE_WITH_STATE",
    "WRITE_WITH_TRID_WITH_STATE",
}

#: ``perfbench/nocevbench``'s experimental arms, as the NoC each RISC's
#: transactions must be seen on. Direction and NoC are fully confounded in the
#: original program -- the reader runs on NoC 1 and the writer on NoC 0, so
#: "writes cost +95 cycles" and "NoC 0 costs +95 cycles" are the same statement
#: -- and arm B swaps exactly those two enum values to break it.
#:
#: The table is duplicated in ``perfbench/nocevbench/check_arm.py``, which is
#: the collection-time check and has to run on a card box with no tt-sim on it.
#: Three rows, deliberately: the copy is what lets each side check the other
#: rather than agree with it by construction.
#:
#: **Arms A and C are indistinguishable by this table** -- C changes the target,
#: not the NoCs -- so ``--expect-arm C`` separates C from B but not from A. What
#: separates A from C is the transaction *destination*, which is why
#: ``--peer-noc`` exists below.
ARM_NOC_PAIRING = {
    "A": {("NCRISC", "read"): "NOC_1", ("BRISC", "write"): "NOC_0"},
    "B": {("NCRISC", "read"): "NOC_0", ("BRISC", "write"): "NOC_1"},
    "C": {("NCRISC", "read"): "NOC_1", ("BRISC", "write"): "NOC_0"},
}


# ---------------------------------------------------------------------------
# Reading the artefact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NocTraceEvent:
    """One record of a ``noc_trace_dev*.json`` array.

    ``kind`` is ``"zone"`` for a zone endpoint (which carries ``zone`` and
    ``zone_phase`` instead of ``type``) and ``"noc"`` otherwise.
    """

    core: tuple
    proc: str
    timestamp: int
    kind: str
    type: str
    noc: str
    num_bytes: int
    dst: tuple | None
    vc: int
    run_host_id: int
    zone: str
    zone_phase: str


def load_trace(path):
    """Parse one ``noc_trace_dev*_ID*.json`` into :class:`NocTraceEvent` records.

    Raises rather than guessing on a file that is not this artefact: a silently
    empty event list would read downstream as "this program moved no data".
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: expected a JSON array of NoC trace records, got {type(raw).__name__}"
        )
    events = []
    for rec in raw:
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: array element is not an object: {rec!r}")
        if "sx" not in rec or "sy" not in rec or "timestamp" not in rec:
            raise ValueError(
                f"{path}: record lacks sx/sy/timestamp, so it is not a NoC trace "
                f"record: {rec!r}"
            )
        is_zone = "zone_phase" in rec
        dst = (rec["dx"], rec["dy"]) if "dx" in rec and "dy" in rec else None
        events.append(
            NocTraceEvent(
                core=(int(rec["sx"]), int(rec["sy"])),
                proc=str(rec.get("proc", "")),
                timestamp=int(rec["timestamp"]),
                kind="zone" if is_zone else "noc",
                type=str(rec.get("type", "")),
                noc=str(rec.get("noc", "")),
                num_bytes=int(rec.get("num_bytes") or 0),
                dst=dst,
                vc=int(rec.get("vc", -1)),
                run_host_id=int(rec.get("run_host_id", -1)),
                zone=str(rec.get("zone", "")),
                zone_phase=str(rec.get("zone_phase", "")),
            )
        )
    return events


def load_traces(paths):
    """Every event from every path, concatenated.

    Concatenation is allowed at load time and refused at analysis time by
    :func:`gate_single_window` -- a caller who globs a directory should be told
    what they did, not have half their data silently dropped.
    """
    out = []
    for path in paths:
        out.extend(load_trace(path))
    return out


def is_kernel_zone(event):
    """Is ``event`` an endpoint of the *kernel* zone, the one that bounds the span?

    A zone record with no name at all is treated as the kernel zone: the only
    producers of one are this module's own tests and hand-built fixtures, and
    refusing them would be pedantry about a field the real artefact always sets.
    """
    return event.kind == "zone" and (
        event.zone.endswith(KERNEL_ZONE_SUFFIX) or not event.zone
    )


def is_profiler_push_zone(event):
    """Is ``event`` an endpoint of the profiler's own buffer-flush zone?"""
    return event.kind == "zone" and event.zone == PROFILER_PUSH_ZONE


def is_transparent_zone(event):
    """A nested zone that is neither the kernel's nor the profiler's.

    A consumer's own ``DeviceZoneScopedN`` inside a dataflow kernel. Its
    endpoints are dropped before the partition so the intervals telescope
    across them and the time stays in the enclosing bucket -- which is correct,
    because unlike a profiler flush that time *is* the kernel's work.
    """
    return (
        event.kind == "zone"
        and not is_kernel_zone(event)
        and not is_profiler_push_zone(event)
    )


@dataclass
class Stream:
    """Every event one RISC on one core recorded, in timestamp order."""

    core: tuple
    proc: str
    events: list = field(default_factory=list)

    @property
    def key(self):
        return (self.core, self.proc)

    @property
    def label(self):
        return f"{self.proc} @ {self.core[0]},{self.core[1]}"

    @property
    def runs_seen(self):
        return {e.run_host_id for e in self.events}

    @property
    def zone_starts(self):
        """``ZONE_START`` of the **kernel** zone only.

        Deliberately not "every zone endpoint": the profiler's own
        :data:`PROFILER_PUSH_ZONE` nests inside the kernel zone and counting it
        here is what refused every real workload. See the module docstring.
        """
        return [
            e for e in self.events if is_kernel_zone(e) and e.zone_phase == "ZONE_START"
        ]

    @property
    def zone_ends(self):
        return [
            e for e in self.events if is_kernel_zone(e) and e.zone_phase == "ZONE_END"
        ]

    @property
    def profiler_pushes(self):
        """How many times the profiler flushed its L1 buffer inside this stream."""
        return sum(
            1
            for e in self.events
            if is_profiler_push_zone(e) and e.zone_phase == "ZONE_START"
        )

    @property
    def nested_zone_names(self):
        """Names of any nested zone this module passes through transparently."""
        return sorted({e.zone for e in self.events if is_transparent_zone(e)})

    @property
    def noc_events(self):
        return [e for e in self.events if e.kind == "noc"]

    @property
    def span(self):
        """``ZONE_END - ZONE_START``, or ``0`` if the zone is not a clean pair."""
        starts, ends = self.zone_starts, self.zone_ends
        if len(starts) != 1 or len(ends) != 1:
            return 0
        return ends[0].timestamp - starts[0].timestamp


def group_by_stream(events):
    """``{(core, proc): Stream}``, each stream sorted by timestamp.

    Ties are broken by the original file order, which is the order the events
    were recorded in: two records can share a wall-clock value because the
    clock is read at a coarser granularity than the RISC retires at, and
    reordering them would manufacture a negative interval.
    """
    grouped = defaultdict(list)
    for index, event in enumerate(events):
        grouped[(event.core, event.proc)].append((event.timestamp, index, event))
    streams = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: (r[0], r[1]))
        streams[key] = Stream(core=key[0], proc=key[1], events=[r[2] for r in rows])
    return streams


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def _bucket_for(event):
    """Which bucket the interval *opened by* ``event`` belongs in.

    Only ever called on a kernel or profiler-push zone record: transparent
    nested zones are dropped by :func:`partition_events` first, because
    returning ``None`` for them would drop the interval they open and break
    closure rather than telescoping across them.
    """
    if event.kind == "zone":
        if is_profiler_push_zone(event):
            return "profiler"
        return "prologue" if event.zone_phase == "ZONE_START" else None
    if event.type == READ_BARRIER_START:
        return "read_wait"
    if event.type == WRITE_BARRIER_START:
        return "write_wait"
    if event.type in BARRIER_ENDS:
        return "local"
    if event.type in BLOCKING_SINGLE:
        return "other_wait"
    return "issue"


def partition_events(stream):
    """``stream``'s events with transparent nested zone endpoints removed.

    Removed rather than ignored: an ignored record would leave the interval it
    opens unattributed, and the buckets would then fail to close.
    """
    return [e for e in stream.events if not is_transparent_zone(e)]


def partition(stream):
    """Cut ``stream``'s kernel zone into :data:`MECHANISMS`.

    Every interval between consecutive events is attributed to the event that
    opened it, so the buckets telescope to the zone span exactly -- **but only
    if the kernel's ``ZONE_START`` really is the stream's first event and its
    ``ZONE_END`` the last**. No *NoC* event is filtered to make that true;
    the only records dropped are transparent nested zone endpoints, which
    telescoping passes straight through. Filtering out an event that
    fell outside the zone would quietly repair a non-monotonic stream, and
    catching exactly that is what :func:`gate_partition_closes` is for: with the
    events left where they are, a zone endpoint out of place makes the buckets
    sum to something other than the span, and the gate refuses.
    """
    part = dict.fromkeys(MECHANISMS, 0)
    if len(stream.zone_starts) != 1 or len(stream.zone_ends) != 1:
        return part
    events = partition_events(stream)
    for current, following in zip(events, events[1:]):
        bucket = _bucket_for(current)
        if bucket is None:
            continue
        part[bucket] += following.timestamp - current.timestamp
    return part


def prologue_note(part, span, side):
    """A note when ``prologue`` dominates ``side``'s span, else ``None``.

    Emitted per side rather than per report because the two sides are separate
    measurements: a card stream can be dependency-bound while the simulator's is
    not, and saying which one it was is the actionable half. It is a *note*, not
    a gate -- a dependency wait is a true property of the workload, not a defect
    in either side's timing, so refusing the stream over it would be wrong.
    """
    if span <= 0:
        return None
    share = part.get("prologue", 0) / span
    if share <= PROLOGUE_DOMINATES:
        return None
    return (
        f"{side}: prologue is {100.0 * share:.2f} % of the span. That is very "
        "unlikely to be setup -- a blocking dependency wait before the first NoC "
        "event (cb_wait_front on a producer's output) is recorded nowhere and "
        "falls into prologue. Do not read it as an initialisation cost."
    )


def profiler_note(stream, part, span, side):
    """A note when the device profiler flushed inside ``side``'s span, else ``None``.

    Always emitted when a push happened, at any size. A reader comparing two
    captures needs to know that one of them carries more of the instrument than
    the other, and "1.24 % of the span" is only small until it is the size of
    the effect being measured.
    """
    pushes = stream.profiler_pushes
    if not pushes:
        return None
    cycles = part.get("profiler", 0)
    share = f", {100.0 * cycles / span:.2f} % of the span" if span > 0 else ""
    return (
        f"{side}: the device profiler flushed its own L1 marker buffer to DRAM "
        f"{pushes} time(s) inside this span, costing {cycles} cycles{share}. That "
        "is the instrument, not the kernel -- it is in the `profiler` bucket so it "
        "does not inflate `issue`, and it is an upper bound (the flush's DMA runs "
        "after the zone's own END marker). Subtract it for the uninstrumented span."
    )


def nested_zone_note(stream, side):
    """A note naming any nested zone passed through transparently, else ``None``."""
    names = stream.nested_zone_names
    if not names:
        return None
    return (
        f"{side}: nested zone(s) {', '.join(names)} inside the kernel zone were "
        "passed through: their endpoints open no bucket and the time stays in "
        "whichever bucket the surrounding kernel work opened. If one of them "
        "brackets something that is not the kernel's own work, this partition "
        "attributes it to the kernel."
    )


def partition_closes(part, span):
    """``(ok, reasons)`` -- no negative bucket, and the buckets sum to ``span``."""
    reasons = []
    for name, value in part.items():
        if value < 0:
            reasons.append(f"{name} = {value} is negative (timestamps out of order)")
    total = sum(part.values())
    if total != span:
        reasons.append(f"buckets sum to {total}, zone span is {span}")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Observed transaction latency
# ---------------------------------------------------------------------------


@dataclass
class LatencyClass:
    """Observed issue-to-barrier-end latency for one class of transaction.

    The class key is ``(noc, type, num_bytes)``: latency is size-dependent, and
    pooling a 32-byte semaphore poke with an 8 KiB tile read would produce a
    mean that describes neither.

    **This is not a per-packet flight time and must not be quoted as one.** The
    artefact records no completion timestamp per transaction (see the module
    docstring), so what is measured is *this transaction's issue to the moment
    the barrier that covers it returned* -- an upper bound that includes the
    residual of every other transaction in the same batch. It is the same upper
    bound on both sides, computed the same way, which is what makes it
    comparable; it is not a flight time.
    """

    noc: str
    type: str
    num_bytes: int
    samples: list = field(default_factory=list)

    @property
    def key(self):
        return (self.noc, self.type, self.num_bytes)

    @property
    def count(self):
        return len(self.samples)

    @property
    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    def to_dict(self):
        return {
            "noc": self.noc,
            "type": self.type,
            "num_bytes": self.num_bytes,
            "count": self.count,
            "mean_cycles": self.mean,
            "min_cycles": min(self.samples) if self.samples else None,
            "max_cycles": max(self.samples) if self.samples else None,
        }


def latency_classes(stream):
    """``({key: LatencyClass}, unbarriered)`` for one stream.

    ``unbarriered`` counts transactions still outstanding at ``ZONE_END`` -- a
    kernel that issues without ever barriering, or whose last batch is drained
    by firmware after the zone closes. They are reported rather than dropped:
    silently discarding them would make a kernel that barriers rarely look
    faster than one that barriers often.
    """
    classes = {}
    pending = {"read": [], "write": []}
    for event in stream.noc_events:
        if event.type in READ_ISSUES:
            pending["read"].append(event)
        elif event.type in WRITE_ISSUES:
            pending["write"].append(event)
        elif event.type in (READ_BARRIER_END, WRITE_BARRIER_END):
            side = "read" if event.type == READ_BARRIER_END else "write"
            for issued in pending[side]:
                key = (issued.noc, issued.type, issued.num_bytes)
                cls = classes.get(key)
                if cls is None:
                    cls = LatencyClass(
                        noc=issued.noc, type=issued.type, num_bytes=issued.num_bytes
                    )
                    classes[key] = cls
                cls.samples.append(event.timestamp - issued.timestamp)
            pending[side] = []
    return classes, len(pending["read"]) + len(pending["write"])


def census(stream):
    """``{(noc, type, num_bytes): count}`` -- what the kernel actually issued.

    The quantity :func:`gate_census_matches` compares. Two sides that disagree
    here did not run the same work, and any timing comparison between them is a
    comparison of two different programs.
    """
    counts = defaultdict(int)
    for event in stream.noc_events:
        counts[(event.noc, event.type, event.num_bytes)] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# The criterion
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    unit: str
    mechanisms: tuple
    sim: dict
    hw: dict
    e_total: float
    e_int: float
    ratio: float | None
    denominator: int

    @property
    def passed(self):
        return self.e_int <= TOLERANCE and self.e_total <= TOLERANCE

    def to_dict(self):
        return {
            "unit": self.unit,
            "mechanisms": list(self.mechanisms),
            "sim": {k: self.sim[k] for k in self.mechanisms},
            "hw": {k: self.hw[k] for k in self.mechanisms},
            "e_total": self.e_total,
            "e_int": self.e_int,
            "compensation_ratio": self.ratio,
            "denominator_cycles": self.denominator,
            "passed": self.passed,
        }


def compare_partitions(unit, sim, hw):
    """``E_total``, ``E_int`` and the compensation ratio over :data:`MECHANISMS`."""
    denom = sum(hw[m] for m in MECHANISMS)
    if denom <= 0:
        raise ValueError(f"{unit}: card span is {denom}; nothing to divide by")
    e_total = abs(sum(sim[m] for m in MECHANISMS) - denom) / denom
    e_int = sum(abs(sim[m] - hw[m]) for m in MECHANISMS) / denom
    if e_total > 0:
        ratio = e_int / e_total
    else:
        ratio = None if e_int == 0 else float("inf")
    return Comparison(
        unit=unit,
        mechanisms=MECHANISMS,
        sim=dict(sim),
        hw=dict(hw),
        e_total=e_total,
        e_int=e_int,
        ratio=ratio,
        denominator=denom,
    )


@dataclass
class LatencyComparison:
    """One transaction class's mean observed latency, both sides."""

    key: tuple
    sim_mean: float
    hw_mean: float
    sim_count: int
    hw_count: int

    @property
    def error(self):
        return (
            abs(self.sim_mean - self.hw_mean) / self.hw_mean
            if self.hw_mean
            else float("inf")
        )

    @property
    def passed(self):
        return self.error <= TOLERANCE

    def to_dict(self):
        return {
            "noc": self.key[0],
            "type": self.key[1],
            "num_bytes": self.key[2],
            "sim_mean_cycles": self.sim_mean,
            "hw_mean_cycles": self.hw_mean,
            "sim_count": self.sim_count,
            "hw_count": self.hw_count,
            "relative_error": self.error,
            "passed": self.passed,
        }


def compare_latencies(sim_classes, hw_classes):
    """One :class:`LatencyComparison` per class present on both sides.

    Classes present on only one side are not compared here -- that is
    :func:`gate_census_matches`'s job, and it refuses before this runs.
    """
    return [
        LatencyComparison(
            key=key,
            sim_mean=sim_classes[key].mean,
            hw_mean=hw_classes[key].mean,
            sim_count=sim_classes[key].count,
            hw_count=hw_classes[key].count,
        )
        for key in sorted(set(sim_classes) & set(hw_classes))
    ]


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def line(self):
        return f"  [{'PASS' if self.passed else 'REFUSE'}] {self.name}: {self.detail}"


def _sides(sim, hw):
    return [("sim", sim)] + ([("card", hw)] if hw is not None else [])


def gate_events_present(sim, hw):
    """Both sides must carry a kernel zone *and* at least one NoC event.

    A zone with no NoC events is the fingerprint of the profiler running without
    ``-DPROFILE_NOC_EVENTS`` -- ``TT_METAL_DEVICE_PROFILER=1`` alone produces
    exactly that, and it decodes as "this kernel moved no data", which is a
    perfectly plausible and completely wrong reading.
    """
    problems = []
    for label, side in _sides(sim, hw):
        if not side.zone_starts or not side.zone_ends:
            problems.append(f"{label} {side.label} has no kernel zone")
        if not side.noc_events:
            problems.append(
                f"{label} {side.label} has zero NoC events -- was "
                "TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 set at *build* time?"
            )
    if problems:
        return GateResult("events_present", False, "; ".join(problems))
    detail = f"sim {len(sim.noc_events)} NoC events"
    if hw is not None:
        detail += f", card {len(hw.noc_events)}"
    return GateResult("events_present", True, detail)


def gate_single_window(sim, hw):
    """One **kernel launch** per stream, on both sides.

    The JSON is written per program-execution UID, so more than one
    ``run_host_id`` or more than one ``*-KERNEL`` zone means two runs were
    concatenated -- by globbing a directory, or by a caller passing two
    ``--sim`` paths. Averaging across them is averaging across two different
    device states, and it stays refused: each execution has its own artefact and
    its own span, and they are separate measurements, not two halves of one.

    **What this gate no longer refuses.** It used to count *every* zone
    endpoint, and the profiler's own ``PROFILER-NOC-QUICK-PUSH`` pairs nest
    inside the kernel zone (see :data:`PROFILER_PUSH_ZONE`), so any capture long
    enough to fill the L1 marker buffer was reported as "5 ZONE_START, expected
    one" and declined. That was a misreading of the instrument, not a property
    of the workload: those streams carry one window each. They now decompose,
    with the flushes charged to the ``profiler`` bucket.
    """
    problems = []
    for label, side in _sides(sim, hw):
        if len(side.runs_seen) > 1:
            problems.append(
                f"{label} {side.label} spans run host IDs "
                f"{sorted(side.runs_seen)} -- that is more than one program "
                "execution in one comparison"
            )
        starts, ends = side.zone_starts, side.zone_ends
        if len(starts) != 1 or len(ends) != 1:
            names = sorted({e.zone for e in starts + ends if e.zone}) or ["unnamed"]
            problems.append(
                f"{label} {side.label} has {len(starts)} kernel ZONE_START / "
                f"{len(ends)} ZONE_END ({', '.join(names)}), expected one of each. "
                f"Nested profiler flushes are not counted here and are not the "
                f"cause ({side.profiler_pushes} seen). Each program execution "
                "writes its own noc_trace_dev*_ID*.json: decompose them one at a "
                "time. They are separate spans and this leg will not blend them"
            )
    if problems:
        return GateResult("single_window", False, "; ".join(problems))
    detail = f"sim span {sim.span} cycles"
    if sim.profiler_pushes:
        detail += f" ({sim.profiler_pushes} profiler flush(es) nested inside it)"
    if hw is not None:
        detail += f", card span {hw.span} cycles"
        if hw.profiler_pushes:
            detail += f" ({hw.profiler_pushes} profiler flush(es))"
    return GateResult("single_window", True, detail)


def gate_per_core(sim, hw, mapped=False):
    """One core each side, the same core, and the same RISC.

    Same discipline and the same reason as the sibling leg: on the 2026-08-10
    part a whole physical column keeps a wall-clock epoch 1.5e13 cycles from the
    rest of the die, so a cross-core span is not a coarser measurement but a
    meaningless one. ``--map-core`` states a deliberate correspondence out loud,
    which the Wormhole ``(1,1)`` versus Blackhole ``(1,2)`` worker-origin
    difference makes necessary rather than pedantic.
    """
    if hw is None:
        return GateResult("per_core", True, f"single stream {sim.label}")
    problems = []
    if sim.core != hw.core and not mapped:
        problems.append(
            f"sim core {sim.core} compared against card core {hw.core}; "
            "pass --map-core SIMX,SIMY=CARDX,CARDY if that is intended"
        )
    if sim.proc != hw.proc:
        problems.append(f"sim RISC {sim.proc} compared against card RISC {hw.proc}")
    if problems:
        return GateResult("per_core", False, "; ".join(problems))
    detail = f"{sim.proc} on core {sim.core}"
    if sim.core != hw.core:
        detail = f"sim {sim.label} mapped to card {hw.label} explicitly"
    return GateResult("per_core", True, detail)


def gate_census_matches(sim, hw):
    """The two sides must have issued the *same* transactions.

    Not the same timing -- the same work: every ``(noc, type, bytes)`` class and
    its count. The event stream is emitted by the kernel binary, so an identical
    binary on an identical problem size produces an identical census; a
    difference means the two sides ran different programs (a different tile
    count is the obvious way, and it is the mistake the sibling leg's README
    records as the one to pin down before collecting anything).

    This is the gate that keeps the comparison a comparison of timing.
    """
    if hw is None:
        return GateResult("census_matches", True, "no card data to match against")
    sim_c, hw_c = census(sim), census(hw)
    problems = []
    for key in sorted(set(sim_c) | set(hw_c), key=repr):
        s, h = sim_c.get(key, 0), hw_c.get(key, 0)
        if s != h:
            problems.append(f"{key[1]} {key[2]}B on {key[0]}: sim {s}, card {h}")
    if problems:
        return GateResult(
            "census_matches",
            False,
            "the two sides issued different transactions -- " + "; ".join(problems),
        )
    total = sum(sim_c.values())
    return GateResult(
        "census_matches",
        True,
        f"{len(sim_c)} transaction classes, {total} events, identical on both sides",
    )


def gate_partition_closes(sim, hw):
    """No negative interval, and the buckets sum to the zone span.

    Closure is by construction, so this is really a monotonicity check on the
    timestamps -- and a real one. The device-side stamp reads the wall clock's
    high and low halves as two separate register accesses
    (``kernel_profiler.hpp:178-183``) and stores only 44 bits, so both a torn
    read and a wrap are physically possible. If it refuses on card data, that is
    a finding about the instrument and must be reported, not smoothed over.
    """
    problems = []
    for label, side in _sides(sim, hw):
        ok, reasons = partition_closes(partition(side), side.span)
        if not ok:
            problems.append(f"{label} {side.label}: {'; '.join(reasons)}")
    if problems:
        return GateResult("partition_closes", False, " | ".join(problems))
    return GateResult(
        "partition_closes", True, "every bucket >= 0 and sums to the zone span"
    )


def _arm_problems(label, stream, arm, peer):
    """What ``stream`` contradicts about running nocevbench arm ``arm``."""
    problems = []
    direction = (
        "read"
        if any(e.type in READ_ISSUES for e in stream.noc_events)
        else "write"
        if any(e.type in WRITE_ISSUES for e in stream.noc_events)
        else None
    )
    if direction is None:
        return [f"{label} {stream.label} issued no transactions at all"]
    want = ARM_NOC_PAIRING[arm].get((stream.proc, direction))
    if want is None:
        return [f"{label} {stream.label} issued {direction}s, which arm {arm} does not"]
    issues = READ_ISSUES if direction == "read" else WRITE_ISSUES
    seen = {e.noc for e in stream.noc_events if e.type in issues}
    if seen != {want}:
        problems.append(
            f"{label} {stream.label}'s {direction}s ran on {sorted(seen)}, arm {arm} "
            f"asks for {want} -- the NoC assignment did not take and this is a "
            "different arm than the one named"
        )
    if peer is not None:
        dests = {e.dst for e in stream.noc_events if e.type in issues and e.dst}
        if dests != {peer}:
            problems.append(
                f"{label} {stream.label} addressed {sorted(dests)} rather than the "
                f"peer {peer}"
            )
    return problems


def gate_arm_matches(sim, hw, arm, peer=None):
    """Both sides really ran ``perfbench/nocevbench``'s arm ``arm``.

    **Opt-in** (``--expect-arm``), and not one of the six standing gates: it is
    meaningful only for that program. It exists because an arm-B run that
    silently kept arm A's NoC pairing is *well-formed* -- it passes every other
    gate, decomposes cleanly, and reports a per-class latency table that reads
    as a result. It would say the ~95-cycle error "stayed on the write", which
    is exactly what the rival hypothesis predicts, so the session would produce
    a confident wrong conclusion rather than a refusal.

    The claim is therefore checked against the emitted trace -- every record
    carries the NoC its transaction used (``profiler.cpp:840-887``) -- and never
    against the command line that asked for it.

    ``census_matches`` already refuses an A-against-B comparison as a side
    effect, because the NoC is part of the class key; this gate additionally
    catches *both* sides having drifted onto the same wrong arm, which the
    census cannot see. It does **not** separate A from C, whose NoC pairings are
    identical by design -- pass ``--peer-noc`` for that, and see
    ``perfbench/nocevbench/check_arm.py``, which does the same check at
    collection time on a box with no tt-sim on it.
    """
    problems = []
    for label, side in _sides(sim, hw):
        problems.extend(_arm_problems(label, side, arm, peer))
    if problems:
        return GateResult("arm_matches", False, "; ".join(problems))
    pairing = ARM_NOC_PAIRING[arm]
    used = pairing.get((sim.proc, "read"), pairing.get((sim.proc, "write")))
    detail = f"arm {arm}: {sim.proc}'s transactions all on {used}"
    if peer is not None:
        detail += f", every transfer addressed the peer at {peer}"
    return GateResult("arm_matches", True, detail)


def gate_barriers_pair(sim, hw):
    """Every barrier START is closed by an END, on both sides.

    An unmatched START is a dropped event, and events *are* dropped silently
    when the profiler's L1 vector overflows and the DRAM buffer is full
    (``mark_dropped_timestamps``, ``kernel_profiler.hpp:193``). A truncated
    stream produces a partition that still closes and a latency mean over a
    biased subset, so the drop has to be caught here or not at all.
    """
    problems = []
    for label, side in _sides(sim, hw):
        depth = {"READ": 0, "WRITE": 0}
        for event in side.noc_events:
            if event.type in BARRIER_STARTS:
                depth[event.type.split("_")[0]] += 1
            elif event.type in BARRIER_ENDS:
                depth[event.type.split("_")[0]] -= 1
                if depth[event.type.split("_")[0]] < 0:
                    problems.append(
                        f"{label} {side.label}: {event.type} with no matching START"
                    )
                    break
        for kind, value in depth.items():
            if value > 0:
                problems.append(
                    f"{label} {side.label}: {value} {kind}_BARRIER_START with no END "
                    "-- events were dropped"
                )
    if problems:
        return GateResult("barriers_pair", False, "; ".join(problems))
    return GateResult("barriers_pair", True, "every barrier START has its END")


# ---------------------------------------------------------------------------
# The sim-side internal attribution
# ---------------------------------------------------------------------------


def internal_attribution(noc_parquet_dir, core=None):
    """tt-sim's own modelled per-transaction cycles, for the same run.

    **This is a diagnostic, never a gate, and it has no card counterpart.**

    The roadmap asks this leg to check ``noc_flight_cycles`` "plus queueing".
    Two things have to be said about that plainly rather than papered over:

    1. **There is no hardware number to check it against.** Every NoC event in
       tt-metal's trace is stamped at *issue*; the artefact contains no
       per-transaction completion time. The only completion timing on a card is
       the barrier interval, which is a property of a *batch*, of the NIU's
       counters, and of the RISC's poll loop -- not of one packet's flight.
    2. **"plus queueing" would double-count.** tt-sim's ``noc_flight_cycles`` is
       already ``cycle - issue_cycle`` where ``issue_cycle`` is stamped inside
       ``NUI.transmit`` *after* ``NUI.send_to`` has computed the delay, and that
       delay is ``flight + injection-port queueing + link-contention wait +
       serialisation - 1`` (``NUI._bandwidth_delay``). The queueing is inside
       the number, not beside it.

    So the counter is reported here as what it is: the model's own account of
    where the observed barrier interval came from, on the simulator side only.
    It is what turns "the card and the simulator disagree" into "they disagree
    and the model's hop term is the part that moved".

    Reads the ``TT_SIM_TRACE_NOC`` Parquet dataset. Returns ``None`` when
    pyarrow is unavailable or the dataset is empty, rather than raising -- an
    absent diagnostic must not take down a comparison that does not depend on
    it.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    files = sorted(Path(noc_parquet_dir).rglob("*.parquet"))
    if not files:
        return None
    table = pq.read_table(files).to_pydict()
    rows = len(table.get("cycle", []))
    agg = defaultdict(lambda: {"count": 0, "cycles": 0, "bytes": 0})
    for i in range(rows):
        if core is not None and (table["core_x"][i], table["core_y"][i]) != core:
            continue
        flight = table["flight_cycles"][i]
        if flight is None:
            continue
        entry = agg[(table["unit"][i], table["phase"][i], table["txn_type"][i])]
        entry["count"] += 1
        entry["cycles"] += int(flight)
        entry["bytes"] += int(table["size_bytes"][i] or 0)
    return {
        "/".join(k): {
            **v,
            "mean_cycles": v["cycles"] / v["count"] if v["count"] else 0.0,
        }
        for k, v in sorted(agg.items())
    }


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class StreamReport:
    core: tuple
    proc: str
    card_core: tuple | None = None
    gates: list = field(default_factory=list)
    refused: bool = False
    comparison: Comparison | None = None
    latencies: list = field(default_factory=list)
    sim_partition: dict = field(default_factory=dict)
    hw_partition: dict = field(default_factory=dict)
    sim_span: int = 0
    hw_span: int = 0
    sim_classes: dict = field(default_factory=dict)
    unbarriered: int = 0
    sim_pushes: int = 0
    hw_pushes: int = 0
    notes: list = field(default_factory=list)

    @property
    def label(self):
        return f"{self.proc} @ {self.core[0]},{self.core[1]}"

    @property
    def passed(self):
        if self.refused or self.comparison is None:
            return False
        return self.comparison.passed and all(
            latency.passed for latency in self.latencies
        )

    def to_dict(self):
        return {
            "core": list(self.core),
            "proc": self.proc,
            "card_core": list(self.card_core) if self.card_core else None,
            "refused": self.refused,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
            "sim_span": self.sim_span,
            "hw_span": self.hw_span,
            "sim_partition": self.sim_partition,
            "hw_partition": self.hw_partition,
            "partition_comparison": self.comparison.to_dict()
            if self.comparison
            else None,
            "latency_classes_sim": [c.to_dict() for c in self.sim_classes.values()],
            "latency_comparisons": [latency.to_dict() for latency in self.latencies],
            "unbarriered_transactions": self.unbarriered,
            "sim_profiler_pushes": self.sim_pushes,
            "hw_profiler_pushes": self.hw_pushes,
            "passed": self.passed,
            "notes": self.notes,
        }


def analyse_stream(sim, hw, mapped=False, expect_arm=None, peer=None):
    """Every gate, then the criterion, for one ``(core, RISC)`` stream."""
    report = StreamReport(
        core=sim.core, proc=sim.proc, card_core=hw.core if hw is not None else None
    )
    gates = [
        gate_events_present(sim, hw),
        gate_single_window(sim, hw),
        gate_per_core(sim, hw, mapped=mapped),
        gate_barriers_pair(sim, hw),
        gate_partition_closes(sim, hw),
        gate_census_matches(sim, hw),
    ]
    if expect_arm is not None:
        gates.append(gate_arm_matches(sim, hw, expect_arm, peer))
    for gate in gates:
        report.gates.append(gate)
        if not gate.passed:
            report.refused = True
    if report.refused:
        return report

    report.sim_span = sim.span
    report.sim_partition = partition(sim)
    report.sim_pushes = sim.profiler_pushes
    for note in (
        prologue_note(report.sim_partition, report.sim_span, "sim"),
        profiler_note(sim, report.sim_partition, report.sim_span, "sim"),
        nested_zone_note(sim, "sim"),
    ):
        if note:
            report.notes.append(note)
    report.sim_classes, report.unbarriered = latency_classes(sim)
    if report.unbarriered:
        report.notes.append(
            f"{report.unbarriered} transaction(s) were still outstanding at ZONE_END "
            "and contribute to no latency class"
        )
    if hw is None:
        report.notes.append(
            "decompose-only: no card data, so no E_total, E_int or compensation ratio"
        )
        return report

    report.hw_span = hw.span
    report.hw_partition = partition(hw)
    report.hw_pushes = hw.profiler_pushes
    for note in (
        prologue_note(report.hw_partition, report.hw_span, "card"),
        profiler_note(hw, report.hw_partition, report.hw_span, "card"),
        nested_zone_note(hw, "card"),
    ):
        if note:
            report.notes.append(note)
    if report.sim_pushes != report.hw_pushes:
        report.notes.append(
            f"sim flushed the profiler buffer {report.sim_pushes} time(s), card "
            f"{report.hw_pushes} -- the two spans carry different amounts of the "
            "instrument, and the `profiler` bucket is where that difference sits"
        )
    hw_classes, hw_unbarriered = latency_classes(hw)
    if hw_unbarriered != report.unbarriered:
        report.notes.append(
            f"sim left {report.unbarriered} transaction(s) unbarriered, card "
            f"{hw_unbarriered} -- the latency means are over different subsets"
        )
    report.comparison = compare_partitions(
        report.label, report.sim_partition, report.hw_partition
    )
    report.latencies = compare_latencies(report.sim_classes, hw_classes)
    return report


def analyse(sim_events, hw_events, core_map=None, expect_arm=None, peer=None):
    """One :class:`StreamReport` per ``(core, RISC)`` stream present in ``sim``."""
    core_map = dict(core_map or {})
    sim_streams = group_by_stream(sim_events)
    hw_streams = group_by_stream(hw_events) if hw_events is not None else None
    reports = []
    for key in sorted(sim_streams, key=repr):
        sim = sim_streams[key]
        if hw_streams is None:
            reports.append(analyse_stream(sim, None, expect_arm=expect_arm, peer=peer))
            continue
        target = (core_map.get(sim.core, sim.core), sim.proc)
        hw = hw_streams.get(target)
        if hw is None:
            report = StreamReport(
                core=sim.core, proc=sim.proc, card_core=target[0], refused=True
            )
            report.gates.append(
                GateResult(
                    "per_core",
                    False,
                    f"card trace has no {sim.proc} stream on core {target[0]} "
                    f"(has {sorted(hw_streams, key=repr)})",
                )
            )
            reports.append(report)
            continue
        reports.append(
            analyse_stream(
                sim,
                hw,
                mapped=sim.core in core_map,
                expect_arm=expect_arm,
                peer=peer,
            )
        )
    return reports


def _pct(x):
    return f"{100.0 * x:.2f} %"


def _label(name):
    """Bucket name as shown, starred where the number needs its caveat read."""
    return f"{name} *" if name == "prologue" else name


def _caveat_lines():
    """The standing ``prologue`` caveat, indented under a mechanism table."""
    return [
        f"  {'*' if i == 0 else ' '} {line}" for i, line in enumerate(PROLOGUE_CAVEAT)
    ]


def render(reports, decompose_only=False, internal=None):
    out = []
    out.append("tt-sim NoC timing vs a card's NoC event trace, by mechanism")
    out.append("=" * 74)
    out.append("")
    if not reports:
        out.append("REFUSED: no NoC trace events in the simulator artefact.")
        return "\n".join(out) + "\n"
    for report in reports:
        out.append(f"{report.label}")
        out.append("-" * 74)
        for gate in report.gates:
            out.append(gate.line())
        if report.refused:
            out.append("  REFUSED -- no comparison reported for this stream.")
            out.append("")
            continue
        out.append("")
        if decompose_only or not report.hw_partition:
            out.append(f"  kernel zone span = {report.sim_span} cycles")
            out.append(f"  {'mechanism':<16}{'sim cycles':>14}{'% of span':>12}")
            span = max(1, report.sim_span)
            for name in MECHANISMS:
                value = report.sim_partition[name]
                out.append(
                    f"  {_label(name):<16}{value:>14}{100.0 * value / span:>11.2f} %"
                )
            out.append(
                f"  {'TOTAL':<16}{sum(report.sim_partition.values()):>14}{100.0:>11.2f} %"
            )
            out.extend(_caveat_lines())
            out.append("")
            out.append(
                "  observed issue-to-barrier-end latency (NOT a per-packet flight time)"
            )
            out.append(
                f"  {'noc':<7}{'type':<18}{'bytes':>8}{'n':>6}{'mean':>10}{'min':>8}{'max':>8}"
            )
            for cls in sorted(report.sim_classes.values(), key=lambda c: repr(c.key)):
                out.append(
                    f"  {cls.noc:<7}{cls.type:<18}{cls.num_bytes:>8}{cls.count:>6}"
                    f"{cls.mean:>10.1f}{min(cls.samples):>8}{max(cls.samples):>8}"
                )
            out.append("")
            for note in report.notes:
                out.append(f"  note: {note}")
            out.append("")
            continue
        out.append(
            f"  sim span {report.sim_span} cycles, card span {report.hw_span} cycles"
        )
        for note in report.notes:
            out.append(f"  note: {note}")
        out.append("")
        out.append(
            f"  {'mechanism':<16}{'sim':>12}{'card':>12}{'|diff|':>12}{'% span':>10}"
        )
        denom = max(1, report.hw_span)
        for name in MECHANISMS:
            sim_value, hw_value = report.sim_partition[name], report.hw_partition[name]
            out.append(
                f"  {_label(name):<16}{sim_value:>12}{hw_value:>12}"
                f"{abs(sim_value - hw_value):>12}"
                f"{100.0 * abs(sim_value - hw_value) / denom:>9.2f} %"
            )
        out.extend(_caveat_lines())
        out.append("")
        comparison = report.comparison
        ratio = (
            "n/a (both exact)"
            if comparison.ratio is None
            else (
                "inf"
                if comparison.ratio == float("inf")
                else f"{comparison.ratio:.2f}x"
            )
        )
        out.append(
            f"  [{'PASS' if comparison.passed else 'FAIL'}] {comparison.unit} partition: "
            f"E_total = {_pct(comparison.e_total)}, E_int = {_pct(comparison.e_int)} "
            f"(limit {_pct(TOLERANCE)} each), compensation E_int/E_total = {ratio}"
        )
        out.append("")
        out.append("  observed issue-to-barrier-end latency, per transaction class")
        out.append(
            f"  {'noc':<7}{'type':<18}{'bytes':>8}{'n':>5}{'sim':>10}{'card':>10}{'err':>9}"
        )
        for latency in report.latencies:
            noc, kind, num_bytes = latency.key
            out.append(
                f"  {noc:<7}{kind:<18}{num_bytes:>8}{latency.hw_count:>5}"
                f"{latency.sim_mean:>10.1f}{latency.hw_mean:>10.1f}"
                f"{100.0 * latency.error:>8.1f}%"
                + ("" if latency.passed else "   FAIL")
            )
        out.append("")
    if internal:
        out.append(
            "tt-sim's own modelled NoC cycles for this run (DIAGNOSTIC, not a gate)"
        )
        out.append("-" * 74)
        out.append(
            "  noc_flight_cycles already contains the queueing: it is arrival minus"
        )
        out.append(
            "  issue, and the delay is hops + injection-port wait + link-contention"
        )
        out.append("  wait + serialisation. There is no card counterpart to it.")
        out.append(
            f"  {'unit/phase/type':<32}{'n':>6}{'cycles':>12}{'mean':>10}{'bytes':>12}"
        )
        for name, entry in internal.items():
            out.append(
                f"  {name:<32}{entry['count']:>6}{entry['cycles']:>12}"
                f"{entry['mean_cycles']:>10.1f}{entry['bytes']:>12}"
            )
        out.append("")
    if decompose_only:
        out.append("RESULT: DECOMPOSITION ONLY (no card data supplied)")
    else:
        verdict = all(r.passed for r in reports)
        out.append(f"RESULT: {'PASS' if verdict else 'FAIL'}")
    return "\n".join(out) + "\n"


def parse_core_map(values):
    """``["1,1=1,2"]`` -> ``{(1, 1): (1, 2)}``."""
    out = {}
    for value in values or []:
        try:
            left, right = value.split("=")
            sx, sy = (int(v) for v in left.split(","))
            cx, cy = (int(v) for v in right.split(","))
        except ValueError as exc:
            raise ValueError(
                f"--map-core expects SIMX,SIMY=CARDX,CARDY, got {value!r}"
            ) from exc
        out[(sx, sy)] = (cx, cy)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare tt-sim's NoC timing against a card's, from the NoC event "
            "trace TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 produces on both sides."
        )
    )
    parser.add_argument(
        "--sim",
        required=True,
        action="append",
        help="noc_trace_dev*_ID*.json from a run against tt-sim (repeatable)",
    )
    parser.add_argument(
        "--card",
        action="append",
        default=[],
        help="noc_trace_dev*_ID*.json from the same program on a card (repeatable)",
    )
    parser.add_argument(
        "--sim-internal",
        help=(
            "TT_SIM_TRACE_NOC Parquet directory from the same simulator run; "
            "prints tt-sim's own modelled per-transaction cycles as a diagnostic"
        ),
    )
    parser.add_argument(
        "--decompose-only",
        action="store_true",
        help="print the tt-sim decomposition alone; no card data, no criterion",
    )
    parser.add_argument(
        "--map-core",
        action="append",
        default=[],
        metavar="SIMX,SIMY=CARDX,CARDY",
        help="state a deliberate sim-core to card-core correspondence",
    )
    parser.add_argument(
        "--expect-arm",
        choices=sorted(ARM_NOC_PAIRING),
        help=(
            "refuse unless both traces really ran perfbench/nocevbench's arm "
            "A, B or C, judged by the NoC each RISC's transactions were "
            "recorded on rather than by the flag the run was given"
        ),
    )
    parser.add_argument(
        "--peer-noc",
        metavar="X,Y",
        help=(
            "arm C's peer core, as the program's nocevbench-config line reports "
            "it; with --expect-arm C, every transaction must address it. This is "
            "what separates arm C from arm A, whose NoC pairing is identical"
        ),
    )
    parser.add_argument("--report", help="write the text report here as well as stdout")
    parser.add_argument("--json", dest="json_path", help="write the report as JSON")
    args = parser.parse_args(argv)

    if not args.card and not args.decompose_only:
        parser.error("--card is required unless --decompose-only is given")
    if args.card and args.decompose_only:
        parser.error("--decompose-only and --card are mutually exclusive")

    if args.peer_noc and not args.expect_arm:
        parser.error("--peer-noc is only meaningful with --expect-arm")
    peer = None
    if args.peer_noc:
        try:
            px, py = (int(v) for v in args.peer_noc.split(","))
        except ValueError:
            parser.error(f"--peer-noc expects X,Y, got {args.peer_noc!r}")
        peer = (px, py)

    sim_events = load_traces(args.sim)
    hw_events = load_traces(args.card) if args.card else None
    reports = analyse(
        sim_events,
        hw_events,
        parse_core_map(args.map_core),
        expect_arm=args.expect_arm,
        peer=peer,
    )
    internal = internal_attribution(args.sim_internal) if args.sim_internal else None
    text = render(reports, decompose_only=hw_events is None, internal=internal)
    print(text, end="")
    if args.report:
        Path(args.report).write_text(text)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "tolerance": TOLERANCE,
                    "decompose_only": hw_events is None,
                    "streams": [r.to_dict() for r in reports],
                    "sim_internal": internal,
                },
                indent=2,
            )
            + "\n"
        )
    if hw_events is None:
        return 0 if reports and not any(r.refused for r in reports) else 1
    return 0 if reports and all(r.passed for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
