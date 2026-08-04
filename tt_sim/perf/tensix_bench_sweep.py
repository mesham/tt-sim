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

    python3 -m tt_sim.perf.tensix_bench_sweep --measured hw.csv
    python3 -m tt_sim.perf.tensix_bench_sweep --measured hw.csv --reference sim.csv
    python3 -m tt_sim.perf.tensix_bench_sweep --measured sim.csv --arch blackhole

With ``--reference`` the report additionally diffs two runs of the same binary
-- silicon against tt-sim -- which is the differential form ``optests/diff.sh``
established for values, applied to cycles.

The CSV lives outside this repo, so with no file the script prints where it
looked and exits 0 -- the same "degrade gracefully" contract
``tt_sim/perf/noc_dataset_sweep.py`` uses for its dataset.
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
            }
        )
    out.sort(key=lambda s: (s["phase"], s["variant"], s["probe"], s["thread"]))
    return out


def apply_control(series):
    """Subtract the control probe's slope and divide by the unroll factor.

    Adds ``measured``: cycles per instruction, for phase A only. The control is
    matched on ``(variant, thread)`` so a contended run is corrected by its own
    contended loop overhead.
    """
    control = {
        (s["variant"], s["thread"]): s["slope"]
        for s in series
        if s["phase"] == "A" and s["probe"] == CONTROL_PROBE
    }
    for s in series:
        if s["phase"] != "A":
            s["measured"] = s["slope"]
            s["control"] = None
            continue
        base = control.get((s["variant"], s["thread"]))
        s["control"] = base
        s["measured"] = (
            None if base is None else (s["slope"] - base) / float(s["unroll"])
        )
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


def report(rows, arch, out=sys.stdout, label="measured", reference=None):
    """The whole sweep. Returns the retained per-instruction series."""

    def emit(line=""):
        print(line, file=out)

    series = attach_table(apply_control(series_of(rows)), arch)
    kept, ladder = retained(series)

    emit("=" * 78)
    emit(f"Rung 3: measured Tensix instruction costs vs the tables [{arch}]")
    emit("=" * 78)
    emit()
    emit(f"input: {len(rows)} raw points -> {len(series)} fitted series ({label})")
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
        f"{'probe':<14}{'unit':<7}{'table':>10}{'bound':>10}"
        f"{'measured':>10}{'residual':>10}  {'testable':<9}{'wired'}"
    )
    for s in sorted(kept, key=lambda s: (s["unit"], s["probe"])):
        s["residual"] = s["measured"] - s["table"]
        s["testable"] = s["table"] > 1.0
        emit(
            f"{s['probe']:<14}{s['unit']:<7}{s['table']:>10.2f}{s['bound'] or '':>10}"
            f"{s['measured']:>10.3f}{s['residual']:>10.3f}  "
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
    over = [s for s in kept if s["residual"] < -1e-9]
    if not over:
        emit(
            "  Yes: every residual is >= 0, which is the direction every bound\n"
            "  in these tables is chosen to lean."
        )
    else:
        emit("  NO. The table OVER-CHARGES these instructions:")
        for s in sorted(over, key=lambda s: s["residual"]):
            emit(
                f"    {s['probe']:<14} table {s['table']:.2f}  measured "
                f"{s['measured']:.3f}  ({s['residual']:+.3f})"
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
        expected = len(variants)  # t1 -> tN, so N-fold if the unit is shared
        if ratio >= 0.75 * expected:
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
        "  and the single-thread column is an upper bound only."
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
        "--arch",
        choices=("wormhole", "blackhole"),
        help="architecture the tables are read for (default: the CSV's arch= comment)",
    )
    args = parser.parse_args(argv)

    if not args.measured:
        print(
            "no --measured CSV given.\n"
            "\n"
            "This sweep consumes a dataset that does not ship with any vendor\n"
            "tree: it has to be produced by running perfbench/tensixbench, on\n"
            "silicon or against tt-sim. See perfbench/tensixbench/README.md."
        )
        return 0
    path = Path(args.measured)
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

    report(rows, arch, label=str(path), reference=reference_rows)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
