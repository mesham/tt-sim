"""Rung 3 of the calibration ladder: measured Tensix instruction costs, swept.

``docs/plans/cost-model.md`` lists four rungs of validation below "captured
silicon traces". Rungs 1 and 2 are climbed, and both of them validated the
**NoC and memory** path only. The Tensix instruction costs in
``tt_sim/pe/tensix/tensix_instruction_costs.yaml`` have no external validation
of any kind -- they are well sourced, but provenance is not validation, and they
are the bulk of the cycles in a compute workload.

No public dataset closes that gap: tt-metal's ``1_compute_mm`` microbenchmark
ships no reference numbers, and the only goldens under
``perf_microbenchmark/`` are for dispatch, a path tt-sim does not implement. So
this module consumes a dataset that does not exist yet, produced by a benchmark
that does: ``perfbench/tensixbench``, one tt-metal program that runs unchanged
on silicon and against tt-sim. The methodology, and what each measurement can
and cannot establish, is in ``docs/plans/tensix-cost-benchmark.md``.

What the input is
-----------------

``perfbench/tensixbench`` writes a CSV of **raw points**, never a cost::

    phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles

* **phase A** -- ``n`` unrolled blocks of ``unroll`` back-to-back ``TTI_*``
  instructions, timestamped off ``RISCV_DEBUG_REG_WALL_CLOCK_L``, the same
  register tt-metal's device profiler reads. ``variant`` is ``t1``/``t2``/``t3``:
  how many TRISCs issued the identical burst at once.
* **phase B** -- ``n`` iterations of a ``matmul_tiles`` inner loop at one math
  fidelity (``variant`` is ``LoFi``/``HiFi2``/``HiFi4``), ``unroll`` = 1.

Every cost below is a **slope** over ``n``, so the fixed cost of the two clock
reads, the barrier and the surrounding call cancels exactly. Phase A then
subtracts the ``loop_overhead`` probe -- the identical loop with a body that
emits no instructions -- so the RISC-V loop counter and branch cancel too::

    cycles_per_instruction = (slope(probe) - slope(loop_overhead)) / unroll

Run it
------

::

    python3 -m tt_sim.perf.tensix_bench_sweep
    python3 -m tt_sim.perf.tensix_bench_sweep --measured hw.csv
    python3 -m tt_sim.perf.tensix_bench_sweep --measured hw.csv --reference sim.csv
    python3 -m tt_sim.perf.tensix_bench_sweep --measured sim.csv --arch blackhole
    python3 -m tt_sim.perf.tensix_bench_sweep --formats bf16.csv fp32.csv tf32.csv

With no ``--measured`` the sweep reads the **primary tracked reference
measurement** (:data:`PRIMARY_DATASET`) in ``tt_sim/perf/datasets/``, so the
rung-3 comparison reproduces with no arguments and no hardware. That directory
holds curated silicon datasets only; a local ``perfbench`` run writes next to
its own binary and is gitignored, and has to be passed explicitly. Each
dataset's ``#`` header carries its own provenance -- card, firmware, KMD,
flags -- because a measurement separated from those is not a measurement.

Not every tracked dataset is a result. ``tensixbench-blackhole-dvalid-per-thread
.csv`` is a **control**: the same binary run with the benchmark's original,
confounded dvalid setup, kept so that the artefact it produces can be pointed at
rather than described. Its header says so in capitals and the sweep will never
choose it for you.

With ``--reference`` the report additionally diffs two runs of the same binary
-- silicon against tt-sim -- which is the differential form ``optests/diff.sh``
established for values, applied to cycles.

With ``--formats`` it does something different again: it reads SEVERAL runs of
the same binary that differ only in the source data format the Matrix Unit
decoded, and reports the MATH probes side by side. That is experiment X2 of
``docs/plans/matrix-unit-thread-contention.md``, and it needs several files
because the format is a per-run configuration and not a column. See
:data:`FORMAT_EXPECTATION` for what it predicts and why.

If no dataset can be found the script prints where it looked and exits 0 -- the
same "degrade gracefully" contract ``tt_sim/perf/noc_dataset_sweep.py`` uses for
its dataset.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The dataset.
# ---------------------------------------------------------------------------

#: The control probe. Its slope is the RISC-V block-loop overhead, subtracted
#: from every other phase A slope. Not a measurement of anything Tensix.
CONTROL_PROBE = "loop_overhead"

#: Least-squares fit quality below which a slope is not a slope. The benchmark
#: reports the same number and refuses the run; this is the second gate, for a
#: CSV that arrives from somewhere else.
MIN_R2 = 0.99

#: Where tracked reference measurements live. Silicon only, each carrying its
#: provenance in its own ``#`` header.
DATASET_DIR = Path(__file__).resolve().parent / "datasets"

#: The PRIMARY tracked dataset, read when ``--measured`` is omitted. Named
#: rather than inferred because the directory now holds more than one file and
#: they are not peers: ``tensixbench-blackhole-dvalid-per-thread.csv`` is a
#: deliberate CONTROL reproducing a known-bad setup (one SETDVALID per active
#: thread, which confounds thread count with SrcA/SrcB bank state and at three
#: threads hands the Matrix Unit a bank it already owns). Its value is that it
#: demonstrates the artefact; sweeping it by accident and reading its MATH rows
#: as hardware behaviour is exactly the mistake it exists to prevent, so the
#: default has to be a choice made here rather than whatever sorts first.
PRIMARY_DATASET = "tensixbench-blackhole.csv"

#: Fraction of perfect N-fold scaling above which the unit is called ``shared``.
#: Below it the growth is real but sub-proportional, i.e. ``partial``.
SHARED_LOWER = 0.75

#: Fraction of perfect N-fold scaling ABOVE which "shared" stops being the right
#: word. A shared 1-IPC unit is a *ceiling*: T threads each get 1/T of it, per
#: thread cost grows T-fold and aggregate throughput is flat. It cannot grow
#: faster than that, so a ratio materially above T is not sharing -- it is
#: aggregate throughput FALLING as issuers are added, which no ISA document
#: describes and which is either a real microarchitectural effect or a broken
#: measurement. Either way it must not be printed under the same word as the
#: normal case. At three threads this band starts at 3.45x; the run that forced
#: it read 12.1x and was labelled ``shared (12.1x)``, which is how a
#: benchmark-setup artefact came within one word of reading as documented
#: behaviour. See docs/plans/matrix-unit-thread-contention.md.
SUPERLINEAR_UPPER = 1.15

#: The dvalid setup a format comparison requires, and the reason it is a
#: requirement rather than a preference. ``SETDVALID`` is
#: ``UnsupportedFunctionality`` on Blackhole and leaves ``ImpliedSrc{A,B}Fmt`` an
#: ``UnpredictableValue()`` -- and that field is exactly what the Matrix Unit
#: reads for the source format there, since no Blackhole LLK ever sets
#: ``DISABLE_IMPLIED_SRC{A,B}_FMT_Base``. So a run whose header says
#: ``dvalid_setup=once`` has no defined source format at all, whatever else its
#: header claims, and comparing two of them by format would be comparing two
#: unpredictable values.
FORMAT_SETUP = "unpacr-nop"

#: The Matrix Unit probes a format axis can move. Everything else in phase A is
#: format-blind by construction: the SFPU, ThCon, TDMA and config probes read
#: neither ``SrcA`` nor ``SrcB``, and ``SETRWC``/``INCRWC`` are matrix-unit
#: instructions that touch no Src rows.
FORMAT_PROBES = ("MVMUL", "ELWADD", "ELWMUL")

#: Which formats share a decoded ``SrcAStyle``, from the functional models in
#: ``MVMUL.md`` / ``ELWADD.md`` / ``ELWMUL.md``. Read from the CSV header's
#: ``src_style=`` token when present; this is the fallback and the pin.
FORMAT_STYLE = {
    "bf16": "BF16",
    "fp32": "BF16",
    "tf32": "TF32",
    "fp16": "FP16",
}

#: How many standard errors of the fitted slope count as "the fit cannot tell".
#: Two, i.e. ~95 %, which is the ordinary convention and is written here rather
#: than inlined so that widening it is a visible edit.
RESOLUTION_SIGMA = 2.0


def reference_datasets():
    """Every tracked reference measurement, sorted by path."""
    return sorted(DATASET_DIR.glob("tensixbench-*.csv"))


def default_measured_path(arch=None):
    """The tracked dataset to sweep when ``--measured`` is not given.

    With an ``--arch`` the name is determined by the architecture. Without one
    it is :data:`PRIMARY_DATASET`, which is a named choice rather than "the
    only file present": the directory also holds a control run whose MATH rows
    are a known artefact, and picking between them by glob order would be
    exactly the kind of silent choice this module exists not to make.
    """
    if arch is not None:
        candidate = DATASET_DIR / f"tensixbench-{arch}.csv"
        return candidate if candidate.exists() else None
    primary = DATASET_DIR / PRIMARY_DATASET
    return primary if primary.exists() else None


def read_csv(path):
    """``(rows, meta)``. ``meta`` carries the ``# arch=...`` comment line."""
    meta = {}
    lines = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            for token in line.lstrip("#").split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    meta[key] = value
            continue
        lines.append(line)
    rows = []
    for raw in csv.DictReader(lines):
        rows.append(
            {
                "phase": raw["phase"],
                "variant": raw["variant"],
                "probe_id": int(raw["probe_id"]),
                "probe": raw["probe"],
                "unit": raw["unit"],
                "active_threads": int(raw["active_threads"]),
                "thread": int(raw["thread"]),
                "n": int(raw["n"]),
                "unroll": int(raw["unroll"]),
                "cycles": int(raw["cycles"]),
            }
        )
    return rows, meta


# ---------------------------------------------------------------------------
# The exclusion criteria. DECLARED BEFORE ANY RESIDUAL WAS COMPUTED.
# ---------------------------------------------------------------------------
#
# Same discipline, and the same ordering, as ``noc_dataset_sweep._exclusions``:
# dropping a series because it *disagrees* is fitting, not validating. Every
# rule below names something the comparison is structurally unable to ask, and
# every one could have been written before the benchmark was ever run. The cost
# of each in series is reported so a reader can see how much of the measurement
# the tables are declining to be tested against.

RESIDUAL_EXPECTATION = """\
Two things are predicted, and only the second is a test of the tables.

1. A FLOOR, not an equality. Every occupancy in the tables is charged at the
   low end of its bound (`at_least` and `range` both take the minimum), so the
   modelled figure is deliberately a floor: measured >= table. A NEGATIVE
   residual is the interesting failure -- it means the table over-charges, and
   invents back-pressure the hardware does not have.

2. A CEILING ON WHAT ONE THREAD CAN SHOW. Each TTI_* macro is one `.ttinsn`
   word in the RISC-V instruction stream, and a baby RISC-V core issues at most
   one instruction per cycle. So a single-thread measurement is
   min(1 cycle, unit occupancy) from below: NO probe can measure less than 1.0
   cycles per instruction, whatever the unit could actually do.

   That splits the table into two classes, and the sweep reports them apart:

     * occupancy <= 1  -- NOT TESTABLE from above by this instrument. A
       measured 1.0 is consistent with any unit throughput of 1 IPC or better,
       and would be produced by an issue-limited front end just as readily.
     * occupancy > 1   -- TESTABLE. The front end can deliver an instruction
       every cycle, so anything above 1.0 is the unit refusing them, and the
       measured value is the unit's occupancy.

   The multi-thread variants exist to attack the first class from the other
   side: two or three TRISCs issuing the identical burst can offer 2-3
   instructions per cycle to one shared unit. If the unit is 1 IPC and shared,
   each thread's slope grows in proportion. If it does not grow, either the
   unit accepts more than one instruction per cycle or the issue path never
   back-pressures -- which are different findings and the report says which."""


FORMAT_EXPECTATION = """\
This is EXPLORATORY, not confirmatory. Say so first, because the difference
decides how a null result reads.

WHAT THE ISA DOCUMENTATION PREDICTS. `MatrixUnit.md`'s throughput table gives
MVMUL, ELWMUL, ELWADD one instruction per cycle with NO format qualification of
any kind; the one caveat it carries is about fidelity phases, which is a count
of instructions software issues rather than a per-instruction cost. And the
functional models in MVMUL.md / ELWADD.md / ELWMUL.md reduce every source format
to a three-way `SrcAStyle`:

    FP32, BF16, BFP8, BFP4, BFP2, INT32, INT16  ->  SrcAStyle = BF16
    FP16, FP8, BFP8a, BFP4a, BFP2a, INT8        ->  SrcAStyle = FP16
    TF32                                        ->  SrcAStyle = TF32

So the documented prediction has two strengths, and the report separates them:

  * SAME STYLE (bf16 vs fp32 is the case in point) -- predicted EXACTLY
    indistinguishable. The two codes take the same branch of the same decode.
    A difference here would contradict the functional model outright, and is
    the strongest thing this comparison can find.
  * DIFFERENT STYLE (anything vs tf32 or fp16) -- the datapath genuinely
    differs, but no document gives it a cost. There is no prediction to confirm
    or refute, only a number nothing has ever measured.

WHY THE TABLES CARE. `tensix_instruction_costs.yaml` gives the MATH occupancies
no format axis at all, and `docs/plans/tensix-cost-benchmark.md` lists data
format under "what is not measured, and why". A format-dependent cost would mean
the MATH entries need one, alongside the existing `scales_with: fidelity_phases`.
A format-independent one closes the question with a measurement instead of an
omission, which is worth about as much.

WHAT THIS CANNOT SEE. Phase A issues each op as an individual `.ttinsn` word, so
every number here is the Wait-Gate-bound regime (~6 cycles on Blackhole), not
the MOP-issued ~1 cycle the tables charge. A format effect visible here is
evidence about the Wait Gate and the operand decode; it is NOT directly a
measurement of the quantity the tables hold. See "Two regimes for one
instruction" in docs/plans/tensix-cost-benchmark.md."""


def _exclusions():
    """The retained-set predicate for the per-instruction sweep, as a ladder."""
    return [
        (
            "phase != A",
            "phase B times a matmul_tiles iteration, which is 16 MVMULs plus "
            "unpack, circular-buffer and semaphore work. It is not a "
            "per-instruction cost and is reported separately, as a fidelity "
            "DIFFERENCE, where all of that cancels.",
            lambda s: s["phase"] == "A",
        ),
        (
            "probe == loop_overhead",
            "the control series. Its slope is the subtrahend, not a datum.",
            lambda s: s["probe"] != CONTROL_PROBE,
        ),
        (
            "active_threads > 1",
            "a contended measurement is a different quantity from the table's "
            "per-instruction occupancy: it is the unit's shared throughput. "
            "Reported separately as the issue-limit discriminator.",
            lambda s: s["active_threads"] == 1,
        ),
        (
            f"R^2 < {MIN_R2:.2f}",
            "a series that is not linear in the instruction count has no "
            "slope to read. The benchmark refuses such a run outright; this "
            "catches a CSV that arrived from elsewhere.",
            lambda s: s["r2"] >= MIN_R2,
        ),
        (
            "no occupancy in the table",
            "an opcode with no `occupancy` field, or `provenance: unknown`, is "
            "one the tables have no opinion about -- `model.py` charges it "
            "nothing rather than a plausible-looking 1. There is no claim to "
            "test.",
            lambda s: s["table"] is not None,
        ),
    ]


def retained(series):
    """``(kept, [(rule, removed, remaining), ...])`` -- the ladder, in order."""
    kept, ladder = series, []
    for name, _reason, keep in _exclusions():
        nxt = [s for s in kept if keep(s)]
        ladder.append((name, len(kept) - len(nxt), len(nxt)))
        kept = nxt
    return kept, ladder


# ---------------------------------------------------------------------------
# Fitting.
# ---------------------------------------------------------------------------


def linear_fit(xs, ys):
    """``(intercept, slope, r2)``; ``(mean, 0, 1)`` for a degenerate x."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0, 1.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return mean_y, 0.0, 1.0
    slope = sxy / sxx
    r2 = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return mean_y - slope * mean_x, slope, r2


def slope_stderr(xs, ys, intercept, slope):
    """Standard error of the fitted slope; 0.0 when the fit is exact.

    R^2 does not answer the question this needs answering. Four points can sit
    on a line to R^2 = 0.9999 and still leave the slope uncertain in the third
    decimal, which is the size of the discrepancies this sweep is being asked
    to adjudicate. The standard error is the quantity that says how far the
    slope could move, so it is what the resolution below is built from.
    """
    n = len(xs)
    if n < 3:
        return 0.0
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return 0.0
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return (sse / ((n - 2) * sxx)) ** 0.5


#: Why a small negative residual is not evidence of anything.
FIT_RESOLUTION_NOTE = """\
A residual is only a finding if it is bigger than what the instrument can
resolve, and this instrument has two known one-sided biases, both of which push
the measured value DOWN. They are added per series and reported as `resol`:

  1. CONTROL OVER-SUBTRACTION, worth up to slope(loop_overhead)/unroll. The
     control's cycles are the RISC-V loop counter, compare and branch. Those
     are additive only while the Tensix unit is IDLE waiting for the issuing
     core; the moment the unit back-pressures, the loop's own instructions
     issue underneath it and cost nothing. The subtraction is therefore exactly
     right in the issue-limited regime and up to slope(control)/unroll too much
     in the unit-limited one -- and which regime a probe is in is the thing
     being measured, so the correction cannot be applied selectively.

  2. FIT UNCERTAINTY, {sigma:.0f} standard errors of the fitted slope (plus the
     control's, in quadrature), divided by unroll. Silicon's first burst is not
     like its later ones -- cold i-cache, an unfilled Tensix instruction FIFO,
     a DVFS transition -- and a warm-up offset on the smallest n tilts a
     four-point least-squares fit.

So: residual >= 0 confirms the floor; residual within `resol` of zero is BELOW
the table but INSIDE the instrument, and is reported as such rather than as an
over-charge; only a residual beyond `resol` is a claim about the hardware. The
price is that this instrument cannot detect an over-charge smaller than
`resol`, which is a fraction of a cycle."""


def series_of(rows):
    """Collapse raw points into one fitted series per measurement.

    A series is ``(phase, variant, probe, thread)``; its slope is cycles per
    ``n``, which is per block in phase A and per ``matmul_tiles`` iteration in
    phase B.
    """
    groups = {}
    for row in rows:
        key = (row["phase"], row["variant"], row["probe"], row["thread"])
        groups.setdefault(key, []).append(row)
    out = []
    for (phase, variant, probe, thread), points in groups.items():
        points.sort(key=lambda r: r["n"])
        xs = [p["n"] for p in points]
        ys = [float(p["cycles"]) for p in points]
        intercept, slope, r2 = linear_fit(xs, ys)
        out.append(
            {
                "phase": phase,
                "variant": variant,
                "probe": probe,
                "thread": thread,
                "unit": points[0]["unit"],
                "unroll": points[0]["unroll"],
                "active_threads": points[0]["active_threads"],
                "points": points,
                "intercept": intercept,
                "slope": slope,
                "r2": r2,
                "stderr": slope_stderr(xs, ys, intercept, slope),
            }
        )
    out.sort(key=lambda s: (s["phase"], s["variant"], s["probe"], s["thread"]))
    return out


def apply_control(series):
    """Subtract the control probe's slope and divide by the unroll factor.

    Adds ``measured``: cycles per instruction, for phase A only. The control is
    matched on ``(variant, thread)`` so a contended run is corrected by its own
    contended loop overhead.

    Also adds ``resolution``: the size below which a negative residual says
    nothing, built from the control subtraction's own one-sided bias and the
    fit's standard error. See :data:`FIT_RESOLUTION_NOTE`.
    """
    control = {
        (s["variant"], s["thread"]): s
        for s in series
        if s["phase"] == "A" and s["probe"] == CONTROL_PROBE
    }
    for s in series:
        if s["phase"] != "A":
            s["measured"] = s["slope"]
            s["control"] = None
            s["resolution"] = None
            continue
        base = control.get((s["variant"], s["thread"]))
        s["control"] = None if base is None else base["slope"]
        if base is None:
            s["measured"] = None
            s["resolution"] = None
            continue
        unroll = float(s["unroll"])
        s["measured"] = (s["slope"] - base["slope"]) / unroll
        se_diff = (s["stderr"] ** 2 + base["stderr"] ** 2) ** 0.5
        s["resolution"] = (
            base["slope"] + RESOLUTION_SIGMA * se_diff
        ) / unroll  # bias + noise
    return series


# ---------------------------------------------------------------------------
# The tables.
# ---------------------------------------------------------------------------


def attach_table(series, arch):
    """Look each probe's occupancy up in ``tensix_instruction_costs.yaml``.

    Reads the shipped table through the ordinary loader rather than restating a
    number here, so a table edit moves the comparison too. ``table`` is None
    when the tables have no opinion -- no entry, no ``occupancy`` field, or
    ``provenance: unknown`` -- which is an exclusion, not a disagreement.
    """
    from tt_sim.perf.costs import SOURCED_PROVENANCE, load_costs

    table = load_costs(arch)
    for s in series:
        s["table"] = None
        s["bound"] = None
        s["provenance"] = None
        s["table_max"] = None
        if s["phase"] != "A" or s["probe"] == CONTROL_PROBE:
            continue
        entry = table.find(s["probe"])
        if entry is None or entry.occupancy is None:
            continue
        if entry.provenance not in SOURCED_PROVENANCE:
            continue
        s["table"] = float(entry.occupancy.cycles)
        s["table_max"] = entry.occupancy.max_cycles
        s["bound"] = entry.occupancy.bound
        s["provenance"] = entry.provenance
    return series


def unwired_units(arch):
    """Units whose table entries exist but which tt-sim never charges.

    Read from the test that owns the list rather than restated, so the two
    cannot drift. A hardware measurement still tests the *table* for these; it
    just does not test tt-sim.
    """
    del arch
    try:
        from tt_sim.perf.costs_test import UNWIRED_UNITS

        return set(UNWIRED_UNITS)
    except Exception:  # pragma: no cover - the test module is not a hard dep
        return set()


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _summary(values):
    values = sorted(values)
    if not values:
        return None
    quartile = len(values) // 4
    return {
        "n": len(values),
        "min": values[0],
        "p25": values[quartile],
        "median": statistics.median(values),
        "p75": values[-1 - quartile],
        "max": values[-1],
    }


def _grouped(rows, key_fn):
    groups = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


def report(rows, arch, out=None, label="measured", reference=None, meta=None):
    """The whole sweep. Returns the retained per-instruction series."""
    out = sys.stdout if out is None else out

    def emit(line=""):
        print(line, file=out)

    series = attach_table(apply_control(series_of(rows)), arch)
    kept, ladder = retained(series)

    emit("=" * 78)
    emit(f"Rung 3: measured Tensix instruction costs vs the tables [{arch}]")
    emit("=" * 78)
    emit()
    emit(f"input: {len(rows)} raw points -> {len(series)} fitted series ({label})")
    # The MATH rows mean different things under different setups, and the
    # difference is not visible in any column -- it is a per-run configuration.
    # Printing it next to the input is the cheapest way to stop a reader
    # attributing one setup's numbers to another.
    for key in ("dvalid_setup", "src_format"):
        if meta and meta.get(key):
            emit(f"  {key} = {meta[key]}")
    emit()
    emit("Exclusion ladder (declared before any residual was computed):")
    total = len(series)
    for name, removed, remaining in ladder:
        emit(f"  - {name:<44} removes {removed:>4}  leaves {remaining:>4}")
    emit()
    emit(f"retained: {len(kept)} of {total} series")
    emit()
    emit(RESIDUAL_EXPECTATION)
    emit()

    unwired = unwired_units(arch)

    # -- the per-instruction table -----------------------------------------
    emit("-" * 78)
    emit("Per instruction: measured cycles/instruction vs the table's occupancy")
    emit("-" * 78)
    emit(
        f"{'probe':<14}{'unit':<7}{'table':>8}{'bound':>10}"
        f"{'measured':>10}{'residual':>10}{'resol':>8}  {'testable':<9}{'wired'}"
    )
    for s in sorted(kept, key=lambda s: (s["unit"], s["probe"])):
        s["residual"] = s["measured"] - s["table"]
        s["testable"] = s["table"] > 1.0
        s["resolved"] = s["residual"] < -(s["resolution"] or 0.0)
        emit(
            f"{s['probe']:<14}{s['unit']:<7}{s['table']:>8.2f}{s['bound'] or '':>10}"
            f"{s['measured']:>10.3f}{s['residual']:>10.3f}"
            f"{s['resolution'] or 0.0:>8.3f}  "
            f"{'yes' if s['testable'] else 'no':<9}"
            f"{'no' if s['unit'] in unwired else 'yes'}"
        )

    if not kept:
        emit("  nothing retained; no per-instruction cost to check.")
        _issue_limit_check(series, emit)
        _fidelity_check(series, arch, emit)
        return kept

    # -- is the model a floor? ---------------------------------------------
    emit()
    emit("-" * 78)
    emit("Is the table a floor?")
    emit("-" * 78)
    emit(FIT_RESOLUTION_NOTE.format(sigma=RESOLUTION_SIGMA))
    emit()
    at_or_above = [s for s in kept if s["residual"] >= -1e-9]
    within = [s for s in kept if -1e-9 > s["residual"] and not s["resolved"]]
    over = [s for s in kept if s["resolved"]]
    if at_or_above:
        emit(
            f"  {len(at_or_above)} series at or above the table: "
            f"{', '.join(sorted(s['probe'] for s in at_or_above))}"
        )
    if within:
        stats = _summary([s["residual"] for s in within])
        emit(
            f"  {len(within)} series below the table but INSIDE the "
            f"instrument's resolution\n"
            f"  (worst {stats['min']:+.3f} cycles/instruction). Not an "
            f"over-charge; not a finding:\n"
            f"    {', '.join(sorted(s['probe'] for s in within))}"
        )
    emit()
    if not over:
        emit(
            "  VERDICT: yes. No residual is below the table by more than the\n"
            "  fit can resolve, which is the direction every bound in these\n"
            "  tables is chosen to lean."
        )
    else:
        emit("  VERDICT: NO. The table OVER-CHARGES these, beyond the resolution:")
        for s in sorted(over, key=lambda s: s["residual"]):
            emit(
                f"    {s['probe']:<14} table {s['table']:.2f}  measured "
                f"{s['measured']:.3f}  ({s['residual']:+.3f}, "
                f"resolution {s['resolution']:.3f})"
            )

    testable = [s for s in kept if s["testable"]]
    emit()
    emit(
        f"  of {len(kept)} retained series, {len(testable)} carry an occupancy "
        f"above 1 cycle\n  and are therefore testable by a single-thread "
        f"measurement at all."
    )
    if testable:
        stats = _summary([s["residual"] for s in testable])
        emit(
            f"  their residuals: min {stats['min']:.2f}  median "
            f"{stats['median']:.2f}  max {stats['max']:.2f} cycles"
        )

    # -- residual by axis ---------------------------------------------------
    emit()
    emit("-" * 78)
    emit("Residual by axis -- where the floor stops being flat")
    emit("-" * 78)
    for axis_name, key_fn in (
        ("unit", lambda s: s["unit"]),
        ("bound", lambda s: s["bound"] or "-"),
        ("provenance", lambda s: s["provenance"] or "-"),
        (
            "wired into tt-sim",
            lambda s: "no" if s["unit"] in unwired else "yes",
        ),
        ("testable (table occupancy > 1)", lambda s: "yes" if s["testable"] else "no"),
        (
            "beyond the fit's resolution",
            lambda s: "yes" if s["resolved"] else "no",
        ),
    ):
        emit()
        emit(f"  {axis_name}")
        for name, group in sorted(_grouped(kept, key_fn).items()):
            stats = _summary([s["residual"] for s in group])
            emit(
                f"    {name:<22} n={stats['n']:<4} median {stats['median']:>7.3f}"
                f"   min {stats['min']:>7.3f}   max {stats['max']:>7.3f}"
            )

    _issue_limit_check(series, emit)
    _fidelity_check(series, arch, emit)
    if reference is not None:
        _differential(rows, reference, arch, emit)
    return kept


def _issue_limit_check(series, emit):
    """Does the measured rate change when a second thread issues the same burst?

    This is the discriminator the whole design rests on. A single thread cannot
    offer more than one instruction per cycle, so a measured 1.0 cycles per
    instruction is ambiguous between "the unit is 1 IPC" and "the front end was
    the limit". Two or three threads issuing at once can offer 2-3, so:

      * per-thread cost scales with the thread count  -> one shared unit, and
        the single-thread number was the unit's throughput after all;
      * per-thread cost is unchanged                  -> the unit is not the
        constraint at one instruction per cycle per thread, and nothing in the
        issue path back-pressures.
    """
    emit()
    emit("-" * 78)
    emit("Issue-limit discriminator: the same burst from 1, 2 and 3 TRISCs")
    emit("-" * 78)
    variants = sorted(
        {
            s["variant"]
            for s in series
            if s["phase"] == "A" and s["variant"].startswith("t")
        }
    )
    if len(variants) < 2:
        emit("  only one thread set in this run; nothing to compare.")
        return
    # How many issuers each thread set actually ran, read from the data rather
    # than from the number of columns: a run with only t1 and t3 has two
    # variants and a three-fold issuer step, and calling that "expected 2"
    # would make a perfectly shared unit read as faster-than-shared.
    issuers = {
        variant: max(
            (
                s["active_threads"]
                for s in series
                if s["phase"] == "A" and s["variant"] == variant
            ),
            default=1,
        )
        for variant in variants
    }
    expected = issuers[variants[-1]] / max(issuers[variants[0]], 1)
    emit(
        f"  {'probe':<14}{'unit':<7}"
        + "".join(f"{v:>10}" for v in variants)
        + "     verdict"
    )
    probes = sorted({s["probe"] for s in series if s["phase"] == "A"})
    for probe in probes:
        if probe == CONTROL_PROBE:
            continue
        cells, unit = [], ""
        for variant in variants:
            values = [
                s["measured"]
                for s in series
                if s["phase"] == "A"
                and s["variant"] == variant
                and s["probe"] == probe
                and s.get("measured") is not None
            ]
            unit = next((s["unit"] for s in series if s["probe"] == probe), unit)
            cells.append(statistics.fmean(values) if values else None)
        if any(c is None for c in cells) or not cells[0]:
            continue
        ratio = cells[-1] / cells[0]
        if ratio > SUPERLINEAR_UPPER * expected:
            # Faster than a shared unit can possibly degrade; see
            # SUPERLINEAR_UPPER. Named loudly rather than filed under "shared".
            verdict = f"SUPERLINEAR ({ratio:.1f}x) -- INVESTIGATE"
        elif ratio >= SHARED_LOWER * expected:
            verdict = f"shared ({ratio:.1f}x)"
        elif ratio <= 1.25:
            verdict = "no back-pressure"
        else:
            verdict = f"partial ({ratio:.1f}x)"
        emit(
            f"  {probe:<14}{unit:<7}"
            + "".join(f"{c:>10.3f}" for c in cells)
            + f"     {verdict}"
        )
    emit()
    emit(
        "  'shared' means the unit is the constraint and the single-thread\n"
        "  column really is its occupancy. 'no back-pressure' means adding\n"
        "  issuers cost nothing, so the unit either accepts one instruction\n"
        "  per thread per cycle or nothing in the issue path ever stalls --\n"
        "  and the single-thread column is an upper bound only.\n"
        "  'SUPERLINEAR' means per-thread cost grew by MORE than the thread\n"
        "  count, i.e. the three threads together got LESS work done than one\n"
        "  did. Sharing a 1-IPC unit cannot do that -- it is a ceiling, not a\n"
        "  penalty -- so this is either an undocumented microarchitectural\n"
        "  effect or, far more likely, a confounded probe setup. It is not a\n"
        "  cost-table input until something has separated thread count from\n"
        "  whatever else the thread sets changed."
    )


def _fidelity_check(series, arch, emit):
    """Phase B: does raising math fidelity cost what the table says?

    ``fidelity_phases.mvmuls_per_tile`` x one cycle per MVMUL predicts that a
    tile matmul costs 16 cycles more per fidelity phase. Everything else in the
    measured loop -- the circular buffers, the unpack, the semaphore handshake
    -- is identical across fidelities and cancels in the difference. The
    absolute slope is a confounded composite and is deliberately not compared to
    anything.
    """
    emit()
    emit("-" * 78)
    emit("Fidelity phases: the difference, which is the only clean part")
    emit("-" * 78)
    phase_b = [s for s in series if s["phase"] == "B"]
    if not phase_b:
        emit("  no phase B data in this run.")
        return

    from tt_sim.perf.costs import SOURCED_PROVENANCE, load_costs

    table = load_costs(arch)
    math_unit = table.units.get("MATH")
    extras = (math_unit.extras if math_unit else {}) or {}
    fidelity = extras.get("fidelity_phases") or {}
    mvmuls = fidelity.get("mvmuls_per_tile") or {}
    per_phase = mvmuls.get("count")
    provenance = mvmuls.get("provenance")
    if per_phase is None or provenance not in SOURCED_PROVENANCE:
        emit("  the tables carry no sourced mvmuls_per_tile; nothing to predict.")
        per_phase = None

    # The math thread is the one whose time the extra MVMULs land on.
    by_variant = {}
    for s in phase_b:
        if s["thread"] == 1:
            by_variant[s["variant"]] = s
    order = [v for v in ("LoFi", "HiFi2", "HiFi3", "HiFi4") if v in by_variant]
    if len(order) < 2:
        emit("  fewer than two fidelities measured; nothing to difference.")
        return
    phases = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}

    emit(f"  {'fidelity':<8}{'phases':>7}{'cycles/matmul_tiles':>22}")
    for variant in order:
        emit(f"  {variant:<8}{phases[variant]:>7}{by_variant[variant]['slope']:>22.2f}")
    emit()
    emit(f"  {'step':<16}{'measured':>10}{'predicted':>11}{'residual':>10}")
    for a, b in zip(order, order[1:]):
        measured = by_variant[b]["slope"] - by_variant[a]["slope"]
        steps = phases[b] - phases[a]
        predicted = None if per_phase is None else steps * per_phase
        if predicted is None:
            emit(f"  {a + ' -> ' + b:<16}{measured:>10.2f}{'-':>11}{'-':>10}")
        else:
            emit(
                f"  {a + ' -> ' + b:<16}{measured:>10.2f}{predicted:>11.2f}"
                f"{measured - predicted:>10.2f}"
            )
    emit()
    if per_phase is not None:
        emit(
            f"  predicted = {per_phase} MVMULs per fidelity phase x 1 cycle each\n"
            f"  ({provenance}, from the MATH unit's fidelity_phases block)."
        )
    spread = max(s["slope"] for s in by_variant.values()) - min(
        s["slope"] for s in by_variant.values()
    )
    if spread < 1.0:
        emit(
            "  WARNING: the three fidelity slopes do not separate at all. This\n"
            "  run is limited by the feeder or by the unpacker, not by the math\n"
            "  thread, and the difference above measures nothing. Raise the\n"
            "  operand buffer depth or lower the feeder cost and re-run."
        )


def format_report(datasets, arch, out=None):
    """Experiment X2: the MATH probes at several source data formats.

    ``datasets`` is a list of ``(label, rows, meta)``, one per run. The format is
    a per-run configuration rather than a column in the CSV -- it is programmed
    once, before the burst -- so the comparison is across files by construction,
    and each file's ``#`` header is what says which format it was.

    Returns ``{(probe, format): cycles_per_instruction}`` for the retained
    series, so a caller can assert on it without re-parsing the report.
    """
    out = sys.stdout if out is None else out

    def emit(line=""):
        print(line, file=out)

    emit("=" * 78)
    emit(f"Experiment X2: MATH cost against the source data format [{arch}]")
    emit("=" * 78)
    emit()
    emit(FORMAT_EXPECTATION)
    emit()

    # -- admission, declared before any number is read from any file ---------
    emit("-" * 78)
    emit("Which runs are admitted, and why the criteria are these")
    emit("-" * 78)
    emit(
        f"  1. dvalid_setup == {FORMAT_SETUP}. Any other setup uses a bare\n"
        f"     SETDVALID, which on Blackhole is UnsupportedFunctionality and\n"
        f"     leaves ImpliedSrc{{A,B}}Fmt an UnpredictableValue() -- the very\n"
        f"     field the Matrix Unit reads. Such a run has no source format,\n"
        f"     so it cannot be a point on a format axis.\n"
        f"  2. src_format names one of {', '.join(sorted(FORMAT_STYLE))}.\n"
        f"  3. one run per format. Two files claiming the same format are a\n"
        f"     repeat measurement, not an axis, and are refused rather than\n"
        f"     silently averaged.\n"
        f"  4. then the ordinary per-instruction exclusion ladder, unchanged,\n"
        f"     so a format point is admitted on exactly the terms every other\n"
        f"     measurement in this module is.\n"
    )

    admitted, refused = {}, []
    for label, rows, meta in datasets:
        setup = meta.get("dvalid_setup")
        fmt = meta.get("src_format")
        if setup != FORMAT_SETUP:
            refused.append((label, f"dvalid_setup={setup!r}, not {FORMAT_SETUP!r}"))
            continue
        if fmt not in FORMAT_STYLE:
            refused.append((label, f"src_format={fmt!r} is not a known format"))
            continue
        if fmt in admitted:
            refused.append((label, f"src_format={fmt} already supplied by another run"))
            continue
        admitted[fmt] = (label, rows, meta)
    for label, why in refused:
        emit(f"  REFUSED {label}: {why}")
    if refused:
        emit()
    if len(admitted) < 2:
        emit(
            f"  {len(admitted)} admissible run(s); a format axis needs at least\n"
            "  two. Nothing to compare."
        )
        return {}

    # -- the ladder, per run ------------------------------------------------
    per_format = {}
    for fmt, (label, rows, _meta) in sorted(admitted.items()):
        kept, _ladder = retained(attach_table(apply_control(series_of(rows)), arch))
        per_format[fmt] = {s["probe"]: s for s in kept if s["probe"] in FORMAT_PROBES}
        emit(
            f"  admitted {fmt:<5} ({FORMAT_STYLE[fmt]:<4}) from {label}: "
            f"{len(per_format[fmt])} of {len(FORMAT_PROBES)} MATH probes retained"
        )
    emit()

    formats = sorted(per_format, key=lambda f: (FORMAT_STYLE[f], f))
    probes = [p for p in FORMAT_PROBES if all(p in per_format[f] for f in formats)]
    if not probes:
        emit("  no MATH probe survives the ladder in every run; nothing to compare.")
        return {}

    emit("-" * 78)
    emit("Cycles per instruction, single thread, by source format")
    emit("-" * 78)
    emit(f"  {'probe':<10}" + "".join(f"{f:>10}" for f in formats) + f"{'spread':>10}")
    values = {}
    for probe in probes:
        cells = [per_format[f][probe]["measured"] for f in formats]
        for fmt, cell in zip(formats, cells):
            values[(probe, fmt)] = cell
        emit(
            f"  {probe:<10}"
            + "".join(f"{c:>10.3f}" for c in cells)
            + f"{max(cells) - min(cells):>10.3f}"
        )
    emit()
    emit("  style:    " + "".join(f"{FORMAT_STYLE[f]:>10}" for f in formats))
    emit()

    # -- the pairwise verdicts ---------------------------------------------
    emit("-" * 78)
    emit("Pairwise, against what the functional models predict")
    emit("-" * 78)
    emit(f"  {'pair':<14}{'probe':<10}{'delta':>9}{'resol':>8}  verdict")
    findings = []
    for i, a in enumerate(formats):
        for b in formats[i + 1 :]:
            same_style = FORMAT_STYLE[a] == FORMAT_STYLE[b]
            for probe in probes:
                sa, sb = per_format[a][probe], per_format[b][probe]
                delta = sb["measured"] - sa["measured"]
                # Both runs carry their own one-sided control bias and fit
                # noise; a difference can hide inside either, so take the
                # larger rather than pretending they cancel.
                resolution = max(sa["resolution"] or 0.0, sb["resolution"] or 0.0)
                resolved = abs(delta) > resolution
                if same_style:
                    verdict = (
                        "CONTRADICTS the model (same SrcAStyle)"
                        if resolved
                        else "as predicted: indistinguishable"
                    )
                else:
                    verdict = (
                        "format-dependent (undocumented)"
                        if resolved
                        else "no difference beyond the instrument"
                    )
                if resolved:
                    findings.append((a, b, probe, delta, same_style))
                emit(
                    f"  {a + ' vs ' + b:<14}{probe:<10}{delta:>9.3f}"
                    f"{resolution:>8.3f}  {verdict}"
                )
    emit()
    if not findings:
        emit(
            "  VERDICT: no format effect this instrument can resolve. On silicon\n"
            "  that says the MATH occupancies need no format axis on this\n"
            "  evidence -- a measured statement where there was an unexamined\n"
            "  one -- subject to its being one run per format, on one part, in\n"
            "  the Wait-Gate regime rather than the MOP-issued one the tables\n"
            "  charge.\n"
            "\n"
            "  AGAINST tt-sim this verdict is FORCED and means nothing about any\n"
            "  hardware: nothing back-pressures the issuing core there, so every\n"
            "  phase A probe reads exactly 1.000 whatever the unit or the format\n"
            "  (docs/plans/tensix-cost-benchmark.md, 'the cost model is invisible\n"
            "  to a device-side clock'). A null from a simulator run tests this\n"
            "  harness end to end and nothing else."
        )
    else:
        contradictions = [f for f in findings if f[4]]
        emit(
            f"  VERDICT: {len(findings)} pair(s) differ beyond the instrument's\n"
            f"  resolution, of which {len(contradictions)} are between formats the\n"
            "  functional models decode IDENTICALLY. A format axis on the MATH\n"
            "  occupancies is now motivated -- but see FORMAT_EXPECTATION: phase A\n"
            "  measures the Wait-Gate regime, so what this licenses is an\n"
            "  investigation, not an edit to a table charging the MOP-issued cost."
        )
    return values


def _differential(rows, reference_rows, arch, emit):
    """The same binary, two devices: silicon against tt-sim, per series.

    ``optests/diff.sh`` runs one compiled tt-metal program through tt-sim and
    through the vendor reference simulator and diffs the values. This is that,
    for cycles.
    """
    emit()
    emit("-" * 78)
    emit("Differential: the same binary on both devices")
    emit("-" * 78)
    measured = {
        (s["phase"], s["variant"], s["probe"], s["thread"]): s
        for s in attach_table(apply_control(series_of(rows)), arch)
    }
    other = {
        (s["phase"], s["variant"], s["probe"], s["thread"]): s
        for s in attach_table(apply_control(series_of(reference_rows)), arch)
    }
    shared = sorted(set(measured) & set(other))
    if not shared:
        emit("  no series in common.")
        return
    emit(
        f"  {'probe':<14}{'variant':<8}{'thr':>4}{'measured':>11}{'reference':>11}{'delta':>10}"
    )
    deltas = []
    for key in shared:
        a, b = measured[key], other[key]
        if a.get("measured") is None or b.get("measured") is None:
            continue
        if a["probe"] == CONTROL_PROBE:
            continue
        delta = a["measured"] - b["measured"]
        deltas.append(delta)
        emit(
            f"  {a['probe']:<14}{a['variant']:<8}{a['thread']:>4}"
            f"{a['measured']:>11.3f}{b['measured']:>11.3f}{delta:>10.3f}"
        )
    if deltas:
        stats = _summary(deltas)
        emit()
        emit(
            f"  delta: n {stats['n']}  min {stats['min']:.3f}  median "
            f"{stats['median']:.3f}  max {stats['max']:.3f} cycles/instruction"
        )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep a perfbench/tensixbench CSV against the Tensix cost tables."
    )
    parser.add_argument("--measured", help="CSV from a tensixbench run")
    parser.add_argument(
        "--reference",
        help="a second CSV of the same binary on another device, for a differential",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        metavar="CSV",
        help="two or more tensixbench CSVs differing only in --src-format, for "
        "experiment X2. Mutually exclusive with the ordinary report: the "
        "format is a per-run configuration, so the comparison is across files",
    )
    parser.add_argument(
        "--arch",
        choices=("wormhole", "blackhole"),
        help="architecture the tables are read for (default: the CSV's arch= comment)",
    )
    args = parser.parse_args(argv)

    if args.formats:
        datasets, archs = [], set()
        for name in args.formats:
            path = Path(name)
            if not path.exists():
                print(f"no CSV at {path}.")
                return 2
            rows, meta = read_csv(path)
            datasets.append((str(path), rows, meta))
            archs.add(args.arch or meta.get("arch"))
        if len(archs) != 1 or archs == {None}:
            print(
                "the format comparison needs every run to be from the same "
                f"architecture; found {sorted(str(a) for a in archs)}. Pass --arch."
            )
            return 2
        arch = archs.pop()
        if arch not in ("wormhole", "blackhole"):
            print(f"unknown architecture {arch!r}; pass --arch.")
            return 2
        format_report(datasets, arch)
        return 0

    if args.measured:
        path = Path(args.measured)
    else:
        path = default_measured_path(args.arch)
        if path is None:
            tracked = reference_datasets()
            print(
                "no --measured CSV given, and no single tracked reference to "
                "fall back on.\n"
                "\n"
                f"looked in: {DATASET_DIR}\n"
                f"found: {', '.join(p.name for p in tracked) or 'nothing'}\n"
                "\n"
                "Pass --arch to pick one, or produce a new dataset by running\n"
                "perfbench/tensixbench on silicon or against tt-sim. See\n"
                "perfbench/tensixbench/README.md."
            )
            return 0
    if not path.exists():
        print(f"no CSV at {path}; nothing to sweep.")
        return 0

    rows, meta = read_csv(path)
    arch = args.arch or meta.get("arch")
    if arch not in ("wormhole", "blackhole"):
        print(f"cannot tell which architecture {path} came from; pass --arch.")
        return 2

    reference_rows = None
    if args.reference:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            print(f"no reference CSV at {reference_path}.")
            return 2
        reference_rows, _ = read_csv(reference_path)

    report(rows, arch, label=str(path), reference=reference_rows, meta=meta)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
