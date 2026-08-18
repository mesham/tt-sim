"""Compare tt-sim's *interior* cycle attribution against silicon's own counters.

This is the home-side half of rung 4's **mechanism-attribution leg**
(``ROADMAP.md`` §4, criterion 2 and 3). It consumes two device-profiler logs --
one produced by running a program against tt-sim, one produced by running the
*same binary* on a card -- and answers one question:

    **does tt-sim spend the core's cycles on the same mechanisms hardware
    does, or does it merely arrive at the same total?**

Why the interior, and not the total
-----------------------------------

Three silicon comparisons already exist in this repo: the component slopes
(~1 %), nekbone's per-core zones (± 10.2 %) and ``dramratebench``'s shape. All
three are **envelope** checks. A total can agree while the decomposition behind
it is wrong in compensating directions -- an over-charged unpacker paying for an
under-charged Matrix unit reads as a perfect span. The bottleneck report's claim
that nekbone spends 79.8 % of its span in NoC flight is exactly such an
unverified interior.

The criterion, stated so compensation cannot pass it
----------------------------------------------------

With mechanisms ``m`` partitioning the span, and an explicit **unattributed**
bucket on both sides so the two denominators are the same quantity::

    E_total = |Σ c_sim − Σ c_hw| / Σ c_hw
    E_int   = Σ |c_m,sim − c_m,hw| / Σ c_hw

Pass requires ``E_int <= 25 %`` **and** ``E_total <= 10 %``. The triangle
inequality gives ``E_total <= E_int`` always, so the ratio ``E_int / E_total``
is **the compensation, measured** -- the number a passing total cannot fake, and
the one this module prints most prominently.

Every criterion is **per core**. On the 2026-08-10 part the whole physical
column ``x = 11`` keeps a wall-clock epoch 1.5e13 cycles away from the rest, so
no cross-core span is admissible and the :func:`gate_per_core` refusal exists to
make that unrepresentable rather than merely discouraged.

The partition
-------------

Only four counter families map cleanly onto quantities tt-sim tracks (see
:mod:`tt_sim.misc.perf_counters`), and the partition is built from exactly
those. Per core, over ``3 x ref_cnt`` **thread-cycles** -- three Tensix threads,
each observed for the same window::

    per thread t:  issue_t        = THREAD_INSTRUCTIONS_t
                   sem_empty_t    = WAITING_FOR_NONZERO_SEM_t
                   sem_full_t     = WAITING_FOR_NONFULL_SEM_t
                   idle_t         = ref_cnt − THREAD_STALLS_t − THREAD_INSTRUCTIONS_t
    core level:    srca_valid     = WAITING_FOR_SRCA_VALID
                   srcb_valid     = WAITING_FOR_SRCB_VALID
                   srca_clear     = WAITING_FOR_SRCA_CLEAR
                   srcb_clear     = WAITING_FOR_SRCB_CLEAR
                   unattributed   = Σ THREAD_STALLS − Σ sem − Σ src

which sums to ``3 x ref_cnt`` identically, by construction of the last term.
The four Src conditions sit at core level rather than per thread because
**the hardware counts them that way** -- they are shared signals with one
counter instance each, not a per-thread block -- and inventing a per-thread
split for them would be fitting.

A second, purely per-thread partition of ``ref_cnt`` is also reported::

    issue_t, sem_empty_t, sem_full_t,
    other_stall_t = THREAD_STALLS_t − sem_empty_t − sem_full_t,
    idle_t

Here the Src waits are inside ``other_stall_t``, because at thread granularity
the hardware does not say which thread they belong to. Both partitions close
identically on both sides or the comparison is refused; see
:func:`gate_partition_closes`.

What can make the partition fail to close, and why that is a result
-------------------------------------------------------------------

``unattributed`` goes negative when the named stall reasons out-count
``THREAD_STALLS``. On hardware that is precisely the symptom the vendor tech
report describes for the ``WAITING_FOR_{UNIT}_IDLE`` family -- counters that
report *unit-busy* cycles rather than thread-stall cycles, producing "> 100 %"
values. Those are not used here, but the Src conditions come from the same RTL
generator, so the closure check is a live test of whether they behave as stall
counts. **If it fails on real card data, that is this leg's finding.** It must
be reported, not normalised away.

``idle_t`` goes negative when a thread's stalls plus issues exceed the window,
which means the counters were not all latched over one span.

It fired, 2026-08-17, and it was not the Src conditions
--------------------------------------------------------

The first real card session for this leg (a Wormhole part, both arms, all three
repeats) refuses. The instrument is healthy -- all 16 counters present, the bank
armed, one window, one core -- and the four Src conditions read **exactly zero**
on both arms. What out-counts ``THREAD_STALLS`` is the *semaphore* pair::

    elw  thread 2:  THREAD_STALLS_2 = 270   WAITING_FOR_NONZERO_SEM_2 = 3561
    mm   thread 2:  THREAD_STALLS_2 = 249   WAITING_FOR_NONZERO_SEM_2 = 2805

so ``other_stall_2`` lands at -3291 and ``unattributed_stall`` at -2895 on a
3 x 4317 thread-cycle span.

The partition assumed ``WAITING_FOR_{NONZERO,NONFULL}_SEM_t`` are disjoint
sub-counts of ``THREAD_STALLS_t``. **They are not, and the vendor's own
documentation says so in two places.** Those two counters are members of the
same nine-counter per-thread block as the ``WAITING_FOR_{UNIT}_IDLE_t`` family
this leg already declined -- one Verilog generate array in
``tt_instruction_thread.sv``, Wormhole selects 39-65 and Blackhole selects
31-57, nine reasons x three threads either way
(``tt_metal/tech_reports/PerfCounters/perf-counters.md:74``). Metric 12 there
normalises the semaphore counters against ``ref_cnt``, not against
``THREAD_STALLS_N`` (line 340), exactly as metric 21 does for the idle waits;
and metric 36, "Stall Cause Overlap Factor", is defined as
``sum(all 9 WAITING_FOR_*_N) / THREAD_STALLS_N`` and documents values above 1
as normal -- "*values >1.0 mean multiple stall conditions are active
simultaneously*" -- for **Wormhole and Blackhole alike** (line 871-883). A
family with a documented overlap factor is not a partition.

The ISA documentation says why, and it is the same on both architectures.
``SEMWAIT`` does not stall the thread: it *latches* a wait instruction and the
thread carries on until it reaches an instruction the block mask catches
(``WormholeB0/TensixTile/TensixCoprocessor/SEMWAIT.md``, "Functional model";
``BlackholeA0/.../SEMWAIT.md`` is identical bar one recommendation paragraph).
The Wait Gate then re-evaluates the latched condition *every cycle* until it is
met (``WormholeB0/TensixTile/TensixCoprocessor/WaitGate.md``). So the counter
runs for every cycle the condition is unsatisfied, including the cycles the
thread has nothing at the gate to hold -- cycles which are inside ``idle_t``,
not inside ``THREAD_STALLS_t``. On the card's thread 2 the 3561 semaphore cycles
sit comfortably inside ``idle_2 = 3911``. The partition was therefore
double-counting them: once in ``idle_t``, once subtracted from
``unattributed_stall``.

Metric 36 was tried as a correction, and it is not one
-------------------------------------------------------

The obvious move once the hardware *documents* an overlap is to stop assuming
disjointness and subtract a measured overlap instead. That was settled, and the
answer is no, on four independent grounds.

**It is not a counter.** There is no select for it. The INSTRN_THREAD bank's
complete select map is in ``tt_metal/hw/inc/internal/tt-1xx/wormhole/hw_counters.h``
lines 116-175 (59 counters; req side 0-65, grant side 256/264/272 and 283) and
``.../blackhole/hw_counters.h`` lines 158-219 (also 59); neither carries an
overlap entry, and no other bank does either. Metric 36 is computed on the
**host, in Python**, from nine already-read counters --
``tools/tracy/perf_counter_analysis.py:1332-1350`` -- and the sibling LLK
document files it under a heading that says so:
``tt_metal/tt-llk/docs/performance_counters/performance_counters.md:785`` is
``### Composite``, with the same metric as its number 26 at line 787. So the
question "can a value for it be collected in the same window as the partition's
counters" has no content: nothing collects it. What *is* collectable is its nine
inputs, and they are collectable for free -- all nine are per-thread selects in
the one bank the leg already enables (mask 32, ``INSTRN``,
``perf-counters.md:876``), on both architectures
(``perf-counters.md:875``), so no second pass is needed and the card session of
2026-08-17 already carries them.

**A scalar cannot say which buckets overlap.** The factor is one number per
thread over all nine reasons. A partition needs to know how much of
``WAITING_FOR_NONZERO_SEM_t`` in particular lies outside ``THREAD_STALLS_t``,
and pairwise intersections are not recoverable from a sum of nine counts.

**It cannot tell the two cases apart, and they need opposite corrections.**
Reason cycles can double up *inside* a stalled cycle (several conditions true at
once -- rescale the buckets) or fall *outside* ``THREAD_STALLS_t`` altogether
(a latched condition counting while the thread runs -- move the mass into
``idle_t``). Metric 36 produces the same number either way.

**The card falsifies its documented mechanism.** The tech report explains values
above 1 as "*multiple stall conditions are active simultaneously*"
(``perf-counters.md:871``, ``:883``). Simultaneity between reasons can never
push a *single* reason past the total, yet on the 2026-08-17 Wormhole part three
counters do exactly that: ``WAITING_FOR_NONZERO_SEM_2`` = 3561 against
``THREAD_STALLS_2`` = 270, and ``WAITING_FOR_MATH_IDLE_1`` =
``WAITING_FOR_SFPU_IDLE_1`` = 38 against ``THREAD_STALLS_1`` = 36. The excess is
therefore not simultaneity; it is the ``SEMWAIT``/Wait Gate behaviour above, and
metric 36 mis-describes what it measures. :func:`stall_reason_overlap` reports
``max_reason_excess`` so this is a printed number rather than a remembered
argument.

And even setting all of that aside, the arithmetic self-destructs.
:func:`metric36_as_correction` computes what dividing a bucket by the factor
would give: because the factor's numerator *contains* the bucket, the corrected
value tends to ``THREAD_STALLS_t`` as the counter grows, whatever the counter
says. On the card's thread 2 it returns 251 for the measured 3561, and 260 for a
hypothetical 7122 -- a 100 % change in the input moving the output by 3.7 %. A
correction that is nearly independent of the quantity it corrects is a fit.

Two further points, either of which would be disqualifying alone: seven of the
nine terms in the numerator are the ``WAITING_FOR_{UNIT}_IDLE`` family this leg
already declined as unit-busy counts (see :data:`INSTRN_DECLINED` and
``perf-counters.md:564``), so using the factor readmits them through the back
door; and the counter semantics remain ``vendor_source``, which is corroboration
and never provenance.

**Conclusion, stated so the roadmap can quote it as settled rather than open:
rung 4's mechanism leg cannot be built from the counters this hardware exposes.**
The bank has no counter for the overlap it documents, and without one the named
stall reasons cannot be turned into a partition of anything.

Why this is not fixed by a per-arch bucket, and what Blackhole's pass is worth
------------------------------------------------------------------------------

**Nothing here is Wormhole-specific.** The block, the vendor's overlap metric
and the ``SEMWAIT`` semantics that explain it are documented identically for
both parts. Wormhole is simply the first architecture on which this gate has
ever seen silicon.

**tt-sim cannot fail this gate, on either architecture.**
:meth:`tt_sim.misc.perf_counters.TensixPerfCounters.note_stall` increments
``thread_stalls[t]`` and at most one reason bucket in the *same call*, so
``sem_empty_t + sem_full_t <= thread_stalls_t`` holds by construction. Every
Blackhole result this leg has -- the two checked-in ``sim-*-blackhole.csv`` logs
and both synthetic card files hand-derived from them -- inherits that identity.
So Blackhole's ``partition_closes`` has never been a test of hardware; it closes
because the simulator's counters are built from one hook that makes it close.
That is worth saying plainly: **Blackhole does not close with margin, and it
does not close by luck -- it closes by construction, and is untested.**

A gate that cannot fail is not a gate, so the identity has been broken where it
belongs -- in the counter model, not in the criterion.
:meth:`tt_sim.misc.perf_counters.TensixPerfCounters.note_wait_condition` counts
a cycle in which a *latched* wait condition was unsatisfied without counting a
stall, which is what ``SEMWAIT.md`` says the hardware does, and the test suite
now drives ``TensixPerfCounters`` through its own hooks and its own MMIO
readback into an overlapping state and asserts this gate refuses it -- and,
in the other direction, passes a disjoint one built the same way. The refusal
path is therefore reachable from the machinery rather than only from a
hand-written CSV. Since 2026-08-18 the front end calls the hook as well:
``WaitGate._tick_unheld_latched_wait`` re-evaluates a latched wait on the
cycles nothing is held by it, so a simulated window can carry the overlap
rather than only a hand-built one. What that is **not** is a reproduction of
the card's magnitude -- the excess is however long the modelled thread ran on
past its ``SEMWAIT``, which on the ``mechbench`` arms is a handful of cycles
against the card's 13x, and nothing is scaled to close the difference.

The partition is therefore left exactly as it is, and the refusal is left in
place and made *specific*: :func:`sem_overlap_findings` names the counters, the
excess and the vendor's overlap factor. The three alternatives were all worse.
A per-arch bucket definition would fit the one card session in hand. Dropping
the semaphore counters would leave the Wormhole core partition with ``issue``,
``idle`` and ``unattributed`` and nothing else -- its Src counters are zero --
which is not a mechanism attribution at all. Refusing "Wormhole" by name would
mislabel a defect in the partition's model of the counters as a property of a
part, and would bury the fact that Blackhole is untested rather than passing.

Provenance
----------

``vendor_source``, and inert. The counter semantics come from a vendor tech
report and the RTL it cites, not from the ISA documentation -- there is no
``PerformanceCounters.md`` in either architecture tree. That is fine for
corroboration and **disqualifying as provenance**: no number this module reads,
computes or prints may become a cost, and nothing here writes to
``unit_costs.yaml`` or any other cost table. Silicon is corroboration, never
provenance; ttsim is not a cycle oracle.

Usage
-----

::

    python3 -m tt_sim.perf.stall_attribution \\
        --sim  sim-session/.logs/profile_log_device.csv \\
        --card card-session/instrn/profile_log_device.csv \\
        --report report.txt --json report.json

    # tt-sim side alone, to see the decomposition with no card data yet:
    python3 -m tt_sim.perf.stall_attribution --sim <log> --decompose-only
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: The device-profiler marker id that carries a performance-counter sample.
#: ``PERF_COUNTER_PROFILER_ID`` in ``tt_metal/tools/profiler/perf_counters.hpp``.
PERF_COUNTER_MARKER_ID = 9090

#: ``E_int <= 25 %``. Kept as a fraction so the report can print either.
#: 2.5x the envelope threshold: a decomposition has strictly more ways to be
#: wrong than a total does, the model is a floor with a deliberate unattributed
#: remainder, and it is still tighter than the one known open level error
#: (``dramratebench``, 26-35 %).
E_INT_LIMIT = 0.25

#: ``E_total <= 10 %`` -- what nekbone already achieved, kept as the control
#: against envelope regression rather than as the interesting number.
E_TOTAL_LIMIT = 0.10

#: The counters the partition needs, per thread. Suffixed ``_0/_1/_2``.
PER_THREAD_COUNTERS = (
    "THREAD_STALLS",
    "THREAD_INSTRUCTIONS",
    "WAITING_FOR_NONZERO_SEM",
    "WAITING_FOR_NONFULL_SEM",
)

#: The counters the partition needs at core level. The hardware gives these one
#: counter instance each, shared across threads.
CORE_COUNTERS = (
    "WAITING_FOR_SRCA_VALID",
    "WAITING_FOR_SRCB_VALID",
    "WAITING_FOR_SRCA_CLEAR",
    "WAITING_FOR_SRCB_CLEAR",
)

#: The ordered core-partition mechanism names. Fixed and append-only: a report
#: from an older run must stay readable against a newer one.
CORE_MECHANISMS = (
    "issue_0",
    "issue_1",
    "issue_2",
    "sem_empty_0",
    "sem_empty_1",
    "sem_empty_2",
    "sem_full_0",
    "sem_full_1",
    "sem_full_2",
    "srca_valid",
    "srcb_valid",
    "srca_clear",
    "srcb_clear",
    "unattributed_stall",
    "idle_0",
    "idle_1",
    "idle_2",
)

#: The ordered per-thread partition mechanism names.
THREAD_MECHANISMS = ("issue", "sem_empty", "sem_full", "other_stall", "idle")

#: The nine per-thread stall-reason counters the ``INSTRN_THREAD`` bank
#: generates as **one block** -- a single Verilog generate array in
#: ``tt_instruction_thread.sv``, Wormhole selects 39-65 and Blackhole selects
#: 31-57, nine reasons x three threads either way
#: (``tt_metal/tech_reports/PerfCounters/perf-counters.md:74``). The partition
#: consumes two of them; the other seven were declined as unit-busy counts. The
#: whole block is named here because the vendor's own overlap metric (36) is
#: defined over all nine, and because a diagnosis that quotes that metric has to
#: be able to compute it.
PER_THREAD_STALL_REASONS = (
    "WAITING_FOR_THCON_IDLE",
    "WAITING_FOR_UNPACK_IDLE",
    "WAITING_FOR_PACK_IDLE",
    "WAITING_FOR_MATH_IDLE",
    "WAITING_FOR_NONZERO_SEM",
    "WAITING_FOR_NONFULL_SEM",
    "WAITING_FOR_MOVE_IDLE",
    "WAITING_FOR_MMIO_IDLE",
    "WAITING_FOR_SFPU_IDLE",
)

#: The two members of that block the partition treats as disjoint sub-counts of
#: ``THREAD_STALLS_t``. This is the assumption :func:`sem_overlap_findings`
#: tests directly rather than leaving to be inferred from a negative bucket.
SEM_STALL_REASONS = ("WAITING_FOR_NONZERO_SEM", "WAITING_FOR_NONFULL_SEM")

#: Said once, appended to the refusal, so the reader does not have to rediscover
#: it from the arithmetic. Every claim in it is sourced in the module docstring.
SEM_OVERLAP_EXPLANATION = (
    "WAITING_FOR_{NONZERO,NONFULL}_SEM_t are members of the same nine-counter "
    "per-thread stall-reason block as the WAITING_FOR_{UNIT}_IDLE_t family this "
    "leg declined, and the vendor's own metric 36 defines an overlap factor "
    "sum(all 9 WAITING_FOR_*_N)/THREAD_STALLS_N whose documented values exceed "
    "1 on both Wormhole and Blackhole -- so they are NOT disjoint sub-counts of "
    "THREAD_STALLS_t. Per SEMWAIT.md and WaitGate.md a SEMWAIT latches a "
    "condition and lets the thread run on; the Wait Gate re-evaluates it every "
    "cycle until it is met, so the counter also runs through cycles the thread "
    "had nothing at the gate to hold -- cycles inside idle_t. This partition "
    "cannot close on them, and that is the finding: do not widen a tolerance, "
    "clamp the bucket, or make the buckets per-architecture to hide it. "
    "Metric 36 cannot repair it either: it is a host-side ratio computed in "
    "tools/tracy/perf_counter_analysis.py, not a counter any select exposes, "
    "it cannot say which pair of buckets overlaps, and dividing a bucket by it "
    "drives that bucket to THREAD_STALLS_t whatever the counter reads."
)


def required_counters():
    """Every counter name the partition consumes, as a sorted tuple."""
    names = [f"{base}_{t}" for base in PER_THREAD_COUNTERS for t in range(3)]
    names += list(CORE_COUNTERS)
    return tuple(sorted(names))


# ---------------------------------------------------------------------------
# Reading a device-profiler log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CounterSample:
    """One ``PerfCounter`` packet as the host post-processor wrote it."""

    core: tuple
    risc: str
    run_host_id: int
    counter: str
    value: int
    ref_cnt: int


#: The profiler writes ``meta data`` as JSON with every comma replaced by a
#: semicolon, so the field survives an unquoted CSV. Undo exactly that.
_META_KEY = re.compile(r'"([^"]+)"\s*:\s*("?[^;}"]*"?)')


def _parse_meta(text):
    """Decode one ``meta data`` cell into a dict, or ``None`` if it is not one.

    Deliberately not ``json.loads`` after a blanket ``;``->``,`` swap: a counter
    name never contains either character, but a future field might, and a
    silently mis-parsed row would enter the partition as a wrong number rather
    than as a missing one.
    """
    text = (text or "").strip()
    if not text.startswith("{"):
        return None
    out = {}
    for key, raw in _META_KEY.findall(text):
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            out[key] = raw[1:-1]
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                out[key] = raw
    return out or None


def load_counter_samples(path):
    """Read every performance-counter sample out of a ``profile_log_device.csv``.

    The file's first line is a free-form ``ARCH: ...`` banner and the second is
    the column header; everything after is one marker per row. Only rows whose
    ``timer_id`` is :data:`PERF_COUNTER_MARKER_ID` and whose ``meta data``
    decodes to a counter triple are returned -- zone markers from the same run
    are simply not this module's business.
    """
    path = Path(path)
    with open(path, newline="") as fh:
        lines = fh.read().splitlines()
    header_idx = None
    for idx, line in enumerate(lines[:5]):
        if line.lstrip().startswith("PCIe slot"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(
            f"{path}: not a tt-metal device profiler log "
            "(no 'PCIe slot, core_x, core_y, ...' header in the first five lines)"
        )
    reader = csv.DictReader(
        lines[header_idx:], skipinitialspace=True, restkey="_overflow"
    )
    samples = []
    for row in reader:
        if not row.get("timer_id"):
            continue
        try:
            marker_id = int(row["timer_id"])
        except (TypeError, ValueError):
            continue
        if marker_id != PERF_COUNTER_MARKER_ID:
            continue
        meta = row.get("meta data")
        if isinstance(row.get("_overflow"), list):
            # An unquoted meta-data cell can only overflow if it still holds a
            # comma, which the writer replaces. Rejoin rather than drop.
            meta = ",".join([meta or ""] + [str(x) for x in row["_overflow"]])
        parsed = _parse_meta(meta)
        if not parsed or "counter type" not in parsed:
            continue
        samples.append(
            CounterSample(
                core=(int(row["core_x"]), int(row["core_y"])),
                risc=(row.get("RISC processor type") or "").strip(),
                run_host_id=int(row.get("run host ID") or 0),
                counter=str(parsed["counter type"]),
                value=int(parsed.get("value", 0)),
                ref_cnt=int(parsed.get("ref cnt", 0)),
            )
        )
    return samples


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


@dataclass
class CoreCounters:
    """One core's counter set from one measured window."""

    core: tuple
    cores_seen: set = field(default_factory=set)
    runs_seen: set = field(default_factory=set)
    ref_cnts: set = field(default_factory=set)
    values: dict = field(default_factory=dict)

    @property
    def ref_cnt(self):
        return max(self.ref_cnts) if self.ref_cnts else 0

    def get(self, name):
        return self.values.get(name, 0)


def group_by_core(samples):
    """``{(x, y): CoreCounters}``, keeping the provenance the gates check.

    A repeated counter name (several launches in one log) keeps the **last**
    sample, matching what a re-armed hardware bank would report, and records the
    extra ``run host ID`` so :func:`gate_single_window` can refuse the ambiguity
    rather than average over it.
    """
    out = {}
    for s in samples:
        entry = out.setdefault(s.core, CoreCounters(core=s.core))
        entry.cores_seen.add(s.core)
        entry.runs_seen.add(s.run_host_id)
        entry.ref_cnts.add(s.ref_cnt)
        entry.values[s.counter] = s.value
    return out


def core_partition(counters):
    """The ``3 x ref_cnt`` thread-cycle partition, as an ordered dict.

    Buckets may come back negative. That is on purpose: a negative bucket is
    what :func:`gate_partition_closes` refuses on, and clamping it here would
    turn a finding into a silently plausible number.
    """
    ref = counters.ref_cnt
    stalls = [counters.get(f"THREAD_STALLS_{t}") for t in range(3)]
    issued = [counters.get(f"THREAD_INSTRUCTIONS_{t}") for t in range(3)]
    empty = [counters.get(f"WAITING_FOR_NONZERO_SEM_{t}") for t in range(3)]
    full = [counters.get(f"WAITING_FOR_NONFULL_SEM_{t}") for t in range(3)]
    src = {
        "srca_valid": counters.get("WAITING_FOR_SRCA_VALID"),
        "srcb_valid": counters.get("WAITING_FOR_SRCB_VALID"),
        "srca_clear": counters.get("WAITING_FOR_SRCA_CLEAR"),
        "srcb_clear": counters.get("WAITING_FOR_SRCB_CLEAR"),
    }
    part = {}
    for t in range(3):
        part[f"issue_{t}"] = issued[t]
    for t in range(3):
        part[f"sem_empty_{t}"] = empty[t]
    for t in range(3):
        part[f"sem_full_{t}"] = full[t]
    part.update(src)
    part["unattributed_stall"] = (
        sum(stalls) - sum(empty) - sum(full) - sum(src.values())
    )
    for t in range(3):
        part[f"idle_{t}"] = ref - stalls[t] - issued[t]
    return {name: part[name] for name in CORE_MECHANISMS}


def thread_partition(counters, thread):
    """One thread's ``ref_cnt`` partition. Src waits fall in ``other_stall``."""
    ref = counters.ref_cnt
    stalls = counters.get(f"THREAD_STALLS_{thread}")
    issued = counters.get(f"THREAD_INSTRUCTIONS_{thread}")
    empty = counters.get(f"WAITING_FOR_NONZERO_SEM_{thread}")
    full = counters.get(f"WAITING_FOR_NONFULL_SEM_{thread}")
    return {
        "issue": issued,
        "sem_empty": empty,
        "sem_full": full,
        "other_stall": stalls - empty - full,
        "idle": ref - stalls - issued,
    }


def stall_reason_overlap(counters, thread):
    """The vendor's stall-overlap metric for one thread, plus what breaks here.

    ``block_sum`` and ``block_factor`` are the tech report's metric 36
    (``sum(all 9 WAITING_FOR_*_N) / THREAD_STALLS_N``) and come back ``None``
    when the log does not carry all nine counters -- only two of the nine are in
    :func:`required_counters`, and a missing counter read as zero would
    *understate* the overlap, which is the one direction a diagnosis must never
    err in. ``sem_excess`` needs only counters the partition already requires,
    so it is always available.

    ``max_reason_excess`` is the one that decides whether metric 36 means what
    it says. Its documented reading -- "values >1.0 mean multiple stall
    conditions are active simultaneously" -- can only ever inflate the *sum* of
    the nine. A positive ``max_reason_excess`` says some single counter beat the
    thread's whole stall count on its own, which no simultaneity can do, so the
    excess is a counter running outside ``THREAD_STALLS_t`` and the factor is
    not a measure of overlap. See :func:`metric36_as_correction`.
    """
    stalls = counters.get(f"THREAD_STALLS_{thread}")
    sem = sum(counters.get(f"{name}_{thread}") for name in SEM_STALL_REASONS)
    present = [
        f"{name}_{thread}" in counters.values for name in PER_THREAD_STALL_REASONS
    ]
    block_sum = None
    if all(present):
        block_sum = sum(
            counters.get(f"{name}_{thread}") for name in PER_THREAD_STALL_REASONS
        )
    # The largest *single* reason, over whatever the log carries. Computed from
    # the present counters rather than withheld like ``block_sum``, because the
    # two err in opposite directions: a missing counter can only make this max
    # smaller, and a smaller max can only *weaken* the conclusion drawn from it
    # (see ``max_reason_excess`` below), so reporting it is safe where
    # understating the block sum would not be.
    max_reason, max_value = None, 0
    for name in PER_THREAD_STALL_REASONS:
        key = f"{name}_{thread}"
        if key in counters.values and counters.get(key) > max_value:
            max_reason, max_value = key, counters.get(key)
    return {
        "thread": thread,
        "thread_stalls": stalls,
        "sem": sem,
        "sem_excess": sem - stalls,
        "block_sum": block_sum,
        "block_factor": (block_sum / stalls)
        if (block_sum is not None and stalls)
        else None,
        "max_reason": max_reason,
        "max_reason_value": max_value,
        #: Positive here falsifies the vendor's own explanation of metric 36.
        #: "Multiple stall conditions active simultaneously" can make the *sum*
        #: of the nine exceed ``THREAD_STALLS_t``; it can never make any one of
        #: them exceed it, because a counter is not simultaneous with itself.
        #: So a positive value means at least one reason counter runs on cycles
        #: outside the thread's stalls entirely -- which is what ``SEMWAIT.md``
        #: describes and what the 2026-08-17 card measured.
        "max_reason_excess": max_value - stalls,
    }


def metric36_as_correction(counters, thread, hypothetical_sem=None):
    """What the vendor's overlap factor gives if used to deflate a bucket.

    Computed, rather than argued, because "subtract the documented overlap" is
    the first thing anyone will reach for once they learn metric 36 exists --
    and the arithmetic refutes itself in a way that is easy to state and easy to
    check.

    Metric 36 is ``block_sum / stalls`` where ``block_sum`` *contains* the
    counter being corrected, so::

        corrected(sem) = sem / (block_sum / stalls)
                       = sem * stalls / (sem + other)

    which tends to ``stalls`` as ``sem`` grows, for any ``other``. The
    "correction" therefore discards the very quantity it is applied to and
    returns ``THREAD_STALLS_t`` back: it cannot disagree with the partition's
    assumption, so it cannot test it. ``hypothetical_sem`` re-runs it against a
    made-up semaphore count so the insensitivity is a number and not a claim.

    Returns ``None`` when the log does not carry all nine counters, or when the
    thread never stalled -- there is nothing to divide by in either case.
    """
    overlap = stall_reason_overlap(counters, thread)
    stalls = overlap["thread_stalls"]
    if overlap["block_sum"] is None or not stalls:
        return None
    sem = overlap["sem"]
    other = overlap["block_sum"] - sem

    def corrected(value):
        return value * stalls / (value + other) if (value + other) else None

    out = {
        "thread": thread,
        "thread_stalls": stalls,
        "sem": sem,
        "factor": overlap["block_factor"],
        "corrected_sem": corrected(sem),
        #: What ``corrected_sem`` converges on for a large counter. It is the
        #: thread's own stall count -- i.e. the assumption the partition
        #: already makes, handed back as if it had been measured.
        "limit": float(stalls),
        "hypothetical_sem": hypothetical_sem,
        "corrected_hypothetical": (
            corrected(hypothetical_sem) if hypothetical_sem is not None else None
        ),
    }
    return out


def sem_overlap_findings(label, counters):
    """Name the counters that out-count ``THREAD_STALLS_t``, one line per thread.

    This exists so the refusal is a statement about *which* counters are not
    disjoint rather than a bare negative number. A negative
    ``unattributed_stall`` is the arithmetic; this is the reason.
    """
    findings = []
    for thread in range(3):
        overlap = stall_reason_overlap(counters, thread)
        if overlap["sem_excess"] <= 0:
            continue
        parts = [
            f"{name}_{thread} = {counters.get(f'{name}_{thread}')}"
            for name in SEM_STALL_REASONS
        ]
        detail = (
            f"{label} thread {thread}: {' + '.join(parts)} exceed "
            f"THREAD_STALLS_{thread} = {overlap['thread_stalls']} by "
            f"{overlap['sem_excess']} cycles"
        )
        if overlap["block_factor"] is not None:
            detail += f" (9-counter block overlap {overlap['block_factor']:.2f}x)"
        findings.append(detail)
    return findings


def partition_closes(part, total):
    """``(ok, reasons)`` -- every bucket non-negative and the sum exact."""
    reasons = []
    for name, value in part.items():
        if value < 0:
            reasons.append(f"{name} = {value} (negative)")
    got = sum(part.values())
    if got != total:
        reasons.append(f"buckets sum to {got}, span is {total}")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# The criterion
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """``E_total``, ``E_int`` and the compensation ratio for one partition."""

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
        return self.e_int <= E_INT_LIMIT and self.e_total <= E_TOTAL_LIMIT

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


def compare_partitions(unit, mechanisms, sim, hw):
    """Build a :class:`Comparison` from two partitions over the same buckets.

    ``E_total`` reduces to the span error, because both partitions sum to their
    own span by construction. That is the point of insisting on an explicit
    unattributed bucket: without it the two denominators would be different
    quantities and the inequality ``E_total <= E_int`` would stop meaning
    anything.
    """
    denom = sum(hw[m] for m in mechanisms)
    if denom <= 0:
        raise ValueError(f"{unit}: hardware span is {denom}; nothing to divide by")
    e_total = abs(sum(sim[m] for m in mechanisms) - denom) / denom
    e_int = sum(abs(sim[m] - hw[m]) for m in mechanisms) / denom
    if e_total > 0:
        ratio = e_int / e_total
    else:
        ratio = None if e_int == 0 else float("inf")
    return Comparison(
        unit=unit,
        mechanisms=tuple(mechanisms),
        sim=dict(sim),
        hw=dict(hw),
        e_total=e_total,
        e_int=e_int,
        ratio=ratio,
        denominator=denom,
    )


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def line(self):
        """The gate's verdict, one ``|``-separated clause per line.

        A refusal that names counters is long by design, and a 600-character
        single line is a refusal nobody reads to the end of.
        """
        head = f"  [{'PASS' if self.passed else 'REFUSE'}] {self.name}: "
        clauses = [c.strip() for c in self.detail.split(" | ")]
        pad = " " * len(head)
        return "\n".join(
            (head if i == 0 else pad) + clause for i, clause in enumerate(clauses)
        )


def gate_counters_present(sim, hw):
    """Every counter the partition consumes must be present on both sides."""
    need = set(required_counters())
    missing_sim = sorted(need - set(sim.values))
    missing_hw = sorted(need - set(hw.values)) if hw is not None else []
    if missing_sim or missing_hw:
        parts = []
        if missing_sim:
            parts.append(f"sim missing {', '.join(missing_sim)}")
        if missing_hw:
            parts.append(f"card missing {', '.join(missing_hw)}")
        return GateResult("counters_present", False, "; ".join(parts))
    return GateResult(
        "counters_present", True, f"all {len(need)} partition counters present"
    )


def gate_armed(sim, hw):
    """The bank must actually have counted something on both sides.

    An unarmed bank reads back as a full set of zeros, which decodes as a
    perfectly plausible "nothing ever stalled" -- the exact failure mode
    :mod:`tt_sim.misc.perf_counters` exists to remove, and the one a comparison
    would otherwise pass with flying colours.
    """
    sides = [("sim", sim)] + ([("card", hw)] if hw is not None else [])
    dead = []
    for label, side in sides:
        activity = sum(
            side.get(f"{base}_{t}")
            for base in ("THREAD_STALLS", "THREAD_INSTRUCTIONS")
            for t in range(3)
        )
        if side.ref_cnt <= 0:
            dead.append(f"{label} ref_cnt = {side.ref_cnt}")
        elif activity == 0:
            dead.append(f"{label} has zero stalls and zero instructions on all threads")
    if dead:
        return GateResult(
            "armed",
            False,
            "; ".join(dead) + " -- the kernel never armed the bank",
        )
    detail = f"sim ref_cnt = {sim.ref_cnt}"
    if hw is not None:
        detail += f", card ref_cnt = {hw.ref_cnt}"
    return GateResult("armed", True, detail)


def gate_single_window(sim, hw):
    """All of a core's counters must come from one span of one launch.

    The hardware latches ``ref_cnt`` at the stop edge and BRISC then reads all
    59 selects out of the same frozen bank, so a core whose samples disagree on
    ``ref cnt``, or carry more than one ``run host ID``, is not one window --
    it is two runs concatenated, or a bank that was still counting.
    """
    problems = []
    for label, side in [("sim", sim)] + ([("card", hw)] if hw is not None else []):
        if len(side.ref_cnts) > 1:
            problems.append(f"{label} has {len(side.ref_cnts)} distinct ref_cnt values")
        if len(side.runs_seen) > 1:
            problems.append(f"{label} spans run host IDs {sorted(side.runs_seen)}")
    if problems:
        return GateResult("single_window", False, "; ".join(problems))
    return GateResult("single_window", True, "one launch, one latched span per side")


def gate_per_core(sim, hw, mapped=False):
    """One core on each side, and the same core -- no cross-core span.

    On the 2026-08-10 part the whole physical column ``x = 11`` keeps a
    wall-clock epoch 1.5e13 cycles from the rest of the die, so pooling two
    cores' windows is not a coarser measurement, it is a meaningless one. This
    refuses both halves of that: a unit assembled from more than one coordinate,
    and a sim core silently compared against a differently-numbered card core
    (the Wormhole-vs-Blackhole worker convention makes ``(1,1)`` against
    ``(1,2)`` an easy and invisible mistake). ``--map-core`` states the
    correspondence out loud when it is genuinely intended.
    """
    problems = []
    for label, side in [("sim", sim)] + ([("card", hw)] if hw is not None else []):
        if len(side.cores_seen) > 1:
            problems.append(
                f"{label} unit pools cores {sorted(side.cores_seen)} into one span"
            )
    if hw is not None and sim.core != hw.core and not mapped:
        problems.append(
            f"sim core {sim.core} compared against card core {hw.core}; "
            "pass --map-core SIMX,SIMY=CARDX,CARDY if that is intended"
        )
    if problems:
        return GateResult("per_core", False, "; ".join(problems))
    detail = f"single core {sim.core}"
    if hw is not None and sim.core != hw.core:
        detail = f"sim {sim.core} mapped to card {hw.core} explicitly"
    return GateResult("per_core", True, detail)


def gate_partition_closes(sim, hw):
    """Both sides' partitions must be non-negative and sum to their own span.

    This is the gate that can turn into a finding. ``unattributed_stall`` goes
    negative exactly when the named stall reasons out-count ``THREAD_STALLS``,
    which is the symptom the vendor tech report reports for the
    ``WAITING_FOR_{UNIT}_IDLE`` family it contradicts itself about. The Src
    conditions used here come from the same RTL generator, so this is a live
    test of whether they behave as stall counts on silicon. If it refuses on
    real card data, report the refusal -- do not reshape the partition to make
    it close.

    It did refuse, on the 2026-08-17 Wormhole session, and not on the Src
    conditions: the semaphore pair out-counts ``THREAD_STALLS_2`` by 3291
    cycles. :func:`sem_overlap_findings` runs first so the refusal *names the
    counters* instead of reporting only the negative bucket they produce.
    """
    findings = []
    problems = []
    for label, side in [("sim", sim)] + ([("card", hw)] if hw is not None else []):
        findings.extend(sem_overlap_findings(label, side))
        ok, reasons = partition_closes(core_partition(side), 3 * side.ref_cnt)
        if not ok:
            problems.append(f"{label} core partition: {'; '.join(reasons)}")
        for t in range(3):
            ok_t, reasons_t = partition_closes(thread_partition(side, t), side.ref_cnt)
            if not ok_t:
                problems.append(f"{label} thread {t} partition: {'; '.join(reasons_t)}")
    if findings or problems:
        detail = " | ".join(findings + problems)
        if findings:
            detail += " | " + SEM_OVERLAP_EXPLANATION
        return GateResult("partition_closes", False, detail)
    return GateResult(
        "partition_closes",
        True,
        "core partition sums to 3 x ref_cnt and every thread partition to ref_cnt, "
        "no negative bucket",
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class CoreReport:
    core: tuple
    card_core: tuple | None
    gates: list = field(default_factory=list)
    refused: bool = False
    comparisons: list = field(default_factory=list)
    sim_core_partition: dict = field(default_factory=dict)
    hw_core_partition: dict = field(default_factory=dict)
    sim_thread_partitions: list = field(default_factory=list)
    hw_thread_partitions: list = field(default_factory=list)
    sim_ref_cnt: int = 0
    hw_ref_cnt: int = 0
    #: The vendor's stall-overlap metric per thread, per side. Filled **before**
    #: the gates run, so that a refused core still records the numbers that
    #: explain the refusal rather than an empty report.
    sim_stall_overlap: list = field(default_factory=list)
    hw_stall_overlap: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def passed(self):
        return (
            not self.refused
            and bool(self.comparisons)
            and all(c.passed for c in self.comparisons)
        )

    def to_dict(self):
        return {
            "core": list(self.core),
            "card_core": list(self.card_core) if self.card_core else None,
            "refused": self.refused,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
            "sim_ref_cnt": self.sim_ref_cnt,
            "hw_ref_cnt": self.hw_ref_cnt,
            "sim_stall_overlap": self.sim_stall_overlap,
            "hw_stall_overlap": self.hw_stall_overlap,
            "sim_core_partition": self.sim_core_partition,
            "hw_core_partition": self.hw_core_partition,
            "sim_thread_partitions": self.sim_thread_partitions,
            "hw_thread_partitions": self.hw_thread_partitions,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "passed": self.passed,
            "notes": self.notes,
        }


def regime_notes(sim_part, hw_part):
    """Say when a whole mechanism family is *absent* rather than measured zero.

    Deliberately a note and not a gate: nothing in the profiler log records
    whether ``TT_SIM_COST_MODEL`` was set, so this cannot be checked, only
    suspected. But the suspicion is worth printing, because a zero column reads
    exactly like "this never happens" and the two readings have opposite
    consequences for a decomposition.

    Src *ownership* is the fingerprint. With the cost model off no Tensix
    backend arms an occupancy, so the Matrix unit hands a Src bank back the
    cycle it took it and ``src_reserved_by_matrix`` -- the SrcA/SrcB CLEAR
    counters -- can essentially never fire, whatever the kernel does.
    """
    notes = []
    sim_clear = sim_part["srca_clear"] + sim_part["srcb_clear"]
    hw_clear = hw_part["srca_clear"] + hw_part["srcb_clear"]
    if sim_clear == 0 and hw_clear > 0:
        notes.append(
            "tt-sim reports exactly zero SrcA/SrcB CLEAR cycles while the card "
            f"reports {hw_clear}. With TT_SIM_COST_MODEL unset no backend arms "
            "an occupancy, so that column is ABSENT rather than measured zero "
            "-- re-run the simulator side with TT_SIM_COST_MODEL=1 before "
            "reading anything into this mechanism."
        )
    return notes


def analyse_core(sim, hw, mapped=False):
    """Run every gate, then the criterion, for one core.

    Gates run first and a refusal stops the analysis: an ``E_int`` computed over
    a partition that does not close is a number, but it is not a measurement of
    anything, and printing it invites it to be quoted.
    """
    report = CoreReport(core=sim.core, card_core=hw.core if hw is not None else None)
    report.sim_ref_cnt = sim.ref_cnt
    report.hw_ref_cnt = hw.ref_cnt if hw is not None else 0
    report.sim_stall_overlap = [stall_reason_overlap(sim, t) for t in range(3)]
    if hw is not None:
        report.hw_stall_overlap = [stall_reason_overlap(hw, t) for t in range(3)]
    for gate in (
        gate_counters_present(sim, hw),
        gate_armed(sim, hw),
        gate_single_window(sim, hw),
        gate_per_core(sim, hw, mapped=mapped),
        gate_partition_closes(sim, hw),
    ):
        report.gates.append(gate)
        if not gate.passed:
            report.refused = True
    if report.refused:
        return report

    report.sim_core_partition = core_partition(sim)
    report.sim_thread_partitions = [thread_partition(sim, t) for t in range(3)]
    if hw is None:
        report.hw_core_partition = {}
        report.notes.append(
            "decompose-only: no card data, so no E_total, E_int or compensation ratio"
        )
        return report

    report.hw_core_partition = core_partition(hw)
    report.hw_thread_partitions = [thread_partition(hw, t) for t in range(3)]
    report.notes.extend(
        regime_notes(report.sim_core_partition, report.hw_core_partition)
    )
    report.comparisons.append(
        compare_partitions(
            f"core {sim.core}",
            CORE_MECHANISMS,
            report.sim_core_partition,
            report.hw_core_partition,
        )
    )
    for t in range(3):
        report.comparisons.append(
            compare_partitions(
                f"core {sim.core} thread {t}",
                THREAD_MECHANISMS,
                thread_partition(sim, t),
                thread_partition(hw, t),
            )
        )
    return report


def analyse(sim_samples, hw_samples, core_map=None):
    """Per-core reports for every core present on both sides.

    ``core_map`` is ``{sim_core: card_core}``; a core absent from it is matched
    to the identically-numbered card core, and :func:`gate_per_core` refuses
    that pairing if the coordinates differ.
    """
    core_map = dict(core_map or {})
    sim_cores = group_by_core(sim_samples)
    hw_cores = group_by_core(hw_samples) if hw_samples is not None else None
    reports = []
    for core in sorted(sim_cores):
        sim = sim_cores[core]
        if hw_cores is None:
            reports.append(analyse_core(sim, None))
            continue
        target = core_map.get(core, core)
        hw = hw_cores.get(target)
        if hw is None:
            report = CoreReport(core=core, card_core=target, refused=True)
            report.gates.append(
                GateResult(
                    "per_core",
                    False,
                    f"card log has no core {target} (has {sorted(hw_cores)})",
                )
            )
            reports.append(report)
            continue
        reports.append(analyse_core(sim, hw, mapped=core in core_map))
    return reports


def _pct(x):
    return f"{100.0 * x:.2f} %"


def _overlap_lines(label, overlaps):
    """The assumption the partition rests on, printed as a number.

    Reported whether or not it is violated, because "the semaphore counters fit
    inside THREAD_STALLS on this run" is exactly the kind of thing that is
    silently true on every simulator log and false on the first card.
    """
    lines = [
        f"  {label} stall-reason overlap (the partition needs sem <= stalls per thread)"
    ]
    for overlap in overlaps:
        factor = overlap["block_factor"]
        if overlap["block_sum"] is None:
            block = "9-counter block n/a (log carries only 2 of its 9 counters)"
        elif factor is None:
            block = f"9-counter block {overlap['block_sum']}, no factor (no stalls)"
        else:
            block = f"9-counter block {overlap['block_sum']} = {factor:.2f}x"
        lines.append(
            f"    thread {overlap['thread']}: sem {overlap['sem']} vs "
            f"THREAD_STALLS {overlap['thread_stalls']} "
            f"(excess {overlap['sem_excess']:+d}); {block}"
        )
        # The single largest reason, because a *single* counter exceeding
        # THREAD_STALLS is the thing simultaneity cannot explain -- and so the
        # thing that decides whether the vendor's overlap factor describes what
        # it claims to. Printed on every run, violated or not, for the same
        # reason the line above is.
        if overlap["max_reason"] is not None:
            verdict = (
                " -- one counter alone out-counts the thread's stalls, which "
                "simultaneity between reasons cannot produce"
                if overlap["max_reason_excess"] > 0
                else ""
            )
            lines.append(
                f"      largest single reason {overlap['max_reason']} = "
                f"{overlap['max_reason_value']} "
                f"(excess {overlap['max_reason_excess']:+d}){verdict}"
            )
    return lines


def render(reports, decompose_only=False):
    out = []
    out.append("tt-sim interior cycle attribution vs Tensix hardware stall counters")
    out.append("=" * 74)
    out.append("")
    if not reports:
        out.append("REFUSED: no performance-counter samples in the simulator log.")
        return "\n".join(out) + "\n"
    for report in reports:
        out.append(f"Core {report.core[0]},{report.core[1]}")
        out.append("-" * 74)
        for gate in report.gates:
            out.append(gate.line())
        if report.refused:
            out.append("  REFUSED -- no comparison reported for this core.")
            out.append("")
            continue
        out.append("")
        if decompose_only or not report.hw_core_partition:
            out.append(
                f"  span (ref_cnt) = {report.sim_ref_cnt} cycles; "
                f"partition denominator = {3 * report.sim_ref_cnt} thread-cycles"
            )
            out.append(f"  {'mechanism':<22}{'sim cycles':>14}{'% of span':>12}")
            total = 3 * report.sim_ref_cnt
            for name in CORE_MECHANISMS:
                value = report.sim_core_partition[name]
                out.append(f"  {name:<22}{value:>14}{100.0 * value / total:>11.2f} %")
            out.append(
                f"  {'TOTAL':<22}{sum(report.sim_core_partition.values()):>14}"
                f"{100.0:>11.2f} %"
            )
            out.append("")
            # The per-thread view of the same window. The Src waits are inside
            # ``other_stall`` here, because the hardware's four Src counters are
            # shared across threads and do not say whose cycle each one was.
            out.append(
                f"  per thread, of {report.sim_ref_cnt} cycles each "
                "(Src waits fold into other_stall)"
            )
            head = "".join(f"{m:>13}" for m in THREAD_MECHANISMS)
            out.append(f"  {'thread':<8}{head}")
            for t, part in enumerate(report.sim_thread_partitions):
                row = "".join(f"{part[m]:>13}" for m in THREAD_MECHANISMS)
                out.append(f"  {t:<8}{row}")
            out.append("")
            out.extend(_overlap_lines("sim", report.sim_stall_overlap))
            out.append("")
            for note in report.notes:
                out.append(f"  note: {note}")
            out.append("")
            continue
        out.append(
            f"  sim span {report.sim_ref_cnt} cycles, card span {report.hw_ref_cnt} cycles"
        )
        for note in report.notes:
            out.append(f"  note: {note}")
        out.append("")
        out.append(
            f"  {'mechanism':<22}{'sim':>12}{'card':>12}{'|diff|':>12}{'% span':>10}"
        )
        denom = 3 * report.hw_ref_cnt
        for name in CORE_MECHANISMS:
            s = report.sim_core_partition[name]
            h = report.hw_core_partition[name]
            out.append(
                f"  {name:<22}{s:>12}{h:>12}{abs(s - h):>12}"
                f"{100.0 * abs(s - h) / denom:>9.2f} %"
            )
        out.append("")
        for comparison in report.comparisons:
            ratio = (
                "n/a (both exact)"
                if comparison.ratio is None
                else (
                    "inf"
                    if comparison.ratio == float("inf")
                    else f"{comparison.ratio:.2f}x"
                )
            )
            verdict = "PASS" if comparison.passed else "FAIL"
            out.append(
                f"  [{verdict}] {comparison.unit}: "
                f"E_total = {_pct(comparison.e_total)} (limit {_pct(E_TOTAL_LIMIT)}), "
                f"E_int = {_pct(comparison.e_int)} (limit {_pct(E_INT_LIMIT)}), "
                f"compensation E_int/E_total = {ratio}"
            )
        out.append("")
    verdict = all(r.passed for r in reports) and not any(r.refused for r in reports)
    if decompose_only:
        out.append("RESULT: DECOMPOSITION ONLY (no card data supplied)")
    else:
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
            "Compare tt-sim's per-mechanism cycle attribution against a card's "
            "Tensix hardware stall counters, per core."
        )
    )
    parser.add_argument(
        "--sim",
        required=True,
        help="profile_log_device.csv from a run against tt-sim",
    )
    parser.add_argument(
        "--card",
        help="profile_log_device.csv from the same program on a card",
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
    parser.add_argument("--report", help="write the text report here as well as stdout")
    parser.add_argument("--json", dest="json_path", help="write the report as JSON")
    args = parser.parse_args(argv)

    if not args.card and not args.decompose_only:
        parser.error("--card is required unless --decompose-only is given")
    if args.card and args.decompose_only:
        # Silently ignoring one of them would let a caller believe a criterion
        # was applied when it was not, or the reverse.
        parser.error("--decompose-only and --card are mutually exclusive")

    sim_samples = load_counter_samples(args.sim)
    hw_samples = load_counter_samples(args.card) if args.card else None
    reports = analyse(sim_samples, hw_samples, parse_core_map(args.map_core))
    text = render(reports, decompose_only=hw_samples is None)
    print(text, end="")
    if args.report:
        Path(args.report).write_text(text)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "e_int_limit": E_INT_LIMIT,
                    "e_total_limit": E_TOTAL_LIMIT,
                    "decompose_only": hw_samples is None,
                    "cores": [r.to_dict() for r in reports],
                },
                indent=2,
            )
            + "\n"
        )
    if hw_samples is None:
        return 0 if reports and not any(r.refused for r in reports) else 1
    return 0 if reports and all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
