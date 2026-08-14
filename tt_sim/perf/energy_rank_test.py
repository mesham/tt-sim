"""Exercise of the ranking-energy analysis path, synthetic and real.

Most of what is here is constructed, and that is the point: a gate is only worth
anything if it has been shown to fire in **both** directions, and synthetic data
is the only way to build the failing case on demand. Every gate in
:mod:`tt_sim.perf.energy_rank` gets a passing session and a failing one, and the
failing one differs from the passing one in exactly the thing the gate is about.
Exclusions get the same treatment: a row that should be cut, one that should not,
and a case where cutting it costs the session.

The rest is the **one real card session** this analysis has ever seen, read from
``perfbench/card-sessions/2026-08-13-energybench/power.csv`` rather than
transcribed. Two of the module's changes exist because of properties of that
file, and a test that reads the file cannot quietly stop being about it.

Nothing here touches the simulator: the module under test reads two CSVs and
writes a report.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tt_sim.perf.energy_activity import ACTIVITY_TERMS, load_activity
from tt_sim.perf.energy_rank import (
    BASELINE_LABEL,
    CONTROL_SUFFIX,
    DESIGNED_ARM_TERMS,
    FITTED_PROVENANCE,
    MODEL_FORM,
    QUANTITY,
    analyse,
    check_destination,
    coefficients_document,
    design_condition,
    designed_terms,
    load_measured,
    nnls,
    rankdata,
    reduced_model_r2,
    render,
    select_terms,
    spearman,
)

# ---------------------------------------------------------------------------
# A synthetic board
# ---------------------------------------------------------------------------

#: The truth the synthetic session is generated from. Joules per unit of
#: activity, per launch. These are invented; that is what makes them safe to
#: put in a test and unsafe to put anywhere else.
#: ``matrix_arith_cycles`` and not ``matrix_busy_cycles``: the Matrix Unit's
#: occupancy includes the RWC bookkeeping every compute kernel pays, which the
#: ``sfpu`` arm pays 41 of per iteration against no matrix arithmetic at all.
#: The synthetic truth charges that bookkeeping **nothing**, which is a fixture
#: simplification and not a claim about silicon -- what it buys is a truth that
#: lives entirely on the designed terms, and so is exactly recoverable.
TRUE = {
    "launch": 2.0e-6,
    "instr_retired": 1.0e-9,
    "noc_bytes_total": 4.0e-11,
    "matrix_arith_cycles": 8.0e-10,
    "sfpu_busy_cycles": 5.0e-10,
}

BASELINE_W = 40.0

#: The four terms the five arms were **constructed** to separate -- one per
#: non-idle arm, straight off the arms table in
#: ``perfbench/energybench/README.md``. Written out here rather than imported from
#: :data:`tt_sim.perf.energy_rank.DESIGNED_ARM_TERMS`, so that editing that table
#: cannot silently move what these tests assert. :data:`TRUE` is a truth over
#: exactly these and nothing else, which is what makes it recoverable.
DESIGN_TERMS = [
    "instr_retired",
    "noc_bytes_total",
    "matrix_arith_cycles",
    "sfpu_busy_cycles",
]


#: The synthetic workload set has the same shape as the real schedule: five arms
#: at two inner counts each, minus ``idle`` which has no inner loop, so nine
#: workloads. That count is not decoration -- the fit spends one coefficient per
#: term plus one for the launch machinery and leave-one-out needs a spare, so
#: nine workloads is what makes a four-term truth recoverable at all. Five would
#: not be, and a test built on five would be testing a design that cannot work.
#:
#: The per-arm shapes are the ones the real arms produced against tt-sim on
#: 2026-08-13, extended linearly in ``inner``; only the ratios matter here.
#: The two matrix columns are deliberately different shapes, because that
#: difference is the whole reason ``matrix_arith_cycles`` exists: per iteration
#: the ``mm`` arm issues 64 ``MVMUL`` and 2 ``SETRWC`` (66 cycles of occupancy,
#: 64 of arithmetic) and the ``sfpu`` arm issues 32 ``INCRWC`` and 9 ``SETRWC``
#: and no arithmetic at all (41 cycles of occupancy, 0 of arithmetic). Both
#: numbers are measured, on Blackhole, at ``inner`` 2 and 6.
def _arm_activity(arm: str, inner: int) -> dict:
    base = {
        "instr_retired": 14_610,
        "noc_bytes_total": 0,
        "matrix_busy_cycles": 1,
        "matrix_arith_cycles": 1,
        "sfpu_busy_cycles": 3,
    }
    if arm == "idle":
        return base
    if arm == "rv":
        return {**base, "instr_retired": 14_610 + 16 * inner}
    if arm == "noc":
        return {
            **base,
            "instr_retired": 14_610 + 20 * inner,
            "noc_bytes_total": 2048 * inner,
        }
    if arm == "mm":
        return {
            **base,
            "instr_retired": 23_200,
            "noc_bytes_total": 6_144,
            "matrix_busy_cycles": 66 * inner,
            "matrix_arith_cycles": 64 * inner,
        }
    if arm == "sfpu":
        return {
            **base,
            "instr_retired": 26_300,
            "noc_bytes_total": 12_288,
            "matrix_busy_cycles": 1 + 41 * inner,
            "sfpu_busy_cycles": 128 * inner,
        }
    raise AssertionError(arm)


_SCHEDULE = [("idle", 0)] + [
    (arm, inner)
    for scale in (1, 4)
    for arm, inner in (
        ("rv", 200_000 * scale),
        ("noc", 4_096 * scale),
        ("mm", 4_096 * scale),
        ("sfpu", 4_096 * scale),
    )
]

ACTIVITY = {f"{arm}-{inner}": _arm_activity(arm, inner) for arm, inner in _SCHEDULE}

#: Launch rate falls as the arm's inner loop grows, which is what makes the
#: watts-to-joules-per-launch conversion do real work rather than being a
#: constant rescale.
RATES = {
    f"{arm}-{inner}": 1.0 / (50e-6 + 2.0e-9 * max(inner, 1)) for arm, inner in _SCHEDULE
}


def activity_rows(labels=None, extra=None):
    rows = []
    for label, terms in ACTIVITY.items():
        if labels is not None and label not in labels:
            continue
        row = {"label": label, "arm": label.split("-")[0], "inner": 0, "launches": 1}
        for term in ACTIVITY_TERMS:
            row[term] = float(terms.get(term, 0))
        if extra and label in extra:
            row.update(extra[label])
        rows.append(row)
    return rows


def true_energy(label: str) -> float:
    terms = ACTIVITY[label]
    return TRUE["launch"] + sum(
        TRUE[t] * terms.get(t, 0) for t in TRUE if t != "launch"
    )


#: A well-formed session row carries more than a power now. ``status``, the
#: sysfs clock record and ``therm_trip_delta`` are what the ``telemetry``,
#: ``clock`` and ``thermal`` gates read, and a row without them is refused --
#: which is the point, so the fixture builds complete rows and the failing cases
#: below take them away one at a time.
def _health(aiclk=1350.0, drift=0.2, trip=0.0, status="ok"):
    return {
        "status": status,
        "attempts": 40,
        "aiclk_mean": aiclk,
        "aiclk_drift_pct": drift,
        "therm_trip_delta": trip,
        "pre_idle_w": BASELINE_W,
        "tt_smi_version": "6.2.0",
    }


def measured_rows(
    labels=None,
    cycles=4,
    samples=40,
    noise=0.0,
    control="noc-4096",
    control_offset=0.0,
    include_baseline=True,
    scale=1.0,
    seed=7,
    health=None,
    flat_energy=None,
):
    """Build a card-session power CSV in memory.

    ``scale`` compresses every workload's *delta* from the baseline towards
    zero, which is how the ``spread`` gate's failing case is made: the same
    ordering, but inside the noise floor. ``health`` is a callable
    ``(cycle, label) -> dict`` overriding that row's health columns, which is how
    the telemetry, clock and thermal gates get their failing cases.

    ``flat_energy`` gives **every** workload the same joules per launch, so the
    board's power differs only in how often each one launched. That is the
    ``target_triviality`` gate's failing case, and it is deliberately a session
    that every other gate is happy with.
    """
    rng = np.random.default_rng(seed)
    labels = list(ACTIVITY) if labels is None else list(labels)
    rows = []

    def _energy(label):
        return true_energy(label) if flat_energy is None else flat_energy

    def _add(row):
        row.update(_health())
        if health is not None:
            row.update(health(row["cycle"], row["label"]) or {})
        rows.append(row)

    for cycle in range(cycles):
        if include_baseline:
            _add(
                {
                    "label": BASELINE_LABEL,
                    "cycle": cycle,
                    "slot": 0,
                    "power_w": BASELINE_W + rng.normal(0, noise),
                    "samples": samples,
                    "launches": 0,
                    "wall_s": 30,
                    "rate": 0.0,
                }
            )
        for slot, label in enumerate(labels, start=1):
            power = BASELINE_W + scale * RATES[label] * _energy(label)
            _add(
                {
                    "label": label,
                    "cycle": cycle,
                    "slot": slot,
                    "power_w": power + rng.normal(0, noise),
                    "samples": samples,
                    "launches": RATES[label] * 30,
                    "wall_s": 30.0,
                    "rate": RATES[label],
                }
            )
        if control is not None and control in labels:
            power = BASELINE_W + scale * RATES[control] * _energy(control)
            _add(
                {
                    "label": control + CONTROL_SUFFIX,
                    "cycle": cycle,
                    "slot": len(labels) + 1,
                    "power_w": power + control_offset + rng.normal(0, noise),
                    "samples": samples,
                    "launches": RATES[control] * 30,
                    "wall_s": 30.0,
                    "rate": RATES[control],
                }
            )
    return rows


def gate(report, name):
    for g in report.gates:
        if g.name == name:
            return g
    raise AssertionError(f"no gate named {name!r} in {[g.name for g in report.gates]}")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def test_rankdata_averages_ties():
    assert list(rankdata([10, 20, 20, 30])) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_is_one_for_a_monotone_map_and_minus_one_for_a_reversal():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_is_nan_when_one_side_is_constant():
    assert math.isnan(spearman([1, 2, 3], [5, 5, 5]))


def test_nnls_recovers_a_non_negative_solution_and_clamps_a_negative_one():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert nnls(A, A @ np.array([3.0, 5.0])) == pytest.approx([3.0, 5.0], abs=1e-9)
    # The unconstrained answer here wants a negative second coefficient.
    x = nnls(np.array([[1.0, 1.0], [1.0, 2.0]]), np.array([2.0, 1.0]))
    assert (x >= 0).all()


# ---------------------------------------------------------------------------
# The fit, on clean data
# ---------------------------------------------------------------------------


def test_a_clean_session_ranks_perfectly_and_recovers_the_ordering():
    report = analyse(activity_rows(), measured_rows())
    assert report.ok, render(report)
    assert report.spearman_loo == pytest.approx(1.0)
    measured_order = sorted(report.measured_energy, key=report.measured_energy.get)
    loo_order = sorted(report.loo_predicted, key=report.loo_predicted.get)
    assert measured_order == loo_order
    # Ratios, which is what a ranking claim really is.
    assert max(report.ratio_errors.values()) < 1e-6


def test_the_report_names_the_coefficients_as_fitted_not_sourced():
    report = analyse(activity_rows(), measured_rows())
    text = render(report)
    assert "FITTED, NOT SOURCED" in text
    assert "unit_costs.yaml" in text
    assert "leave-one-out" in text


def test_every_report_states_which_quantity_was_measured():
    """A refused session says it too, because a refusal is also a statement about
    a measurement and the reader still has to know which one."""
    assert "SUSTAINED LOAD" in QUANTITY
    assert QUANTITY in render(analyse(activity_rows(), measured_rows()))
    refused = analyse(activity_rows(), measured_rows(include_baseline=False))
    assert QUANTITY in render(refused)
    assert QUANTITY == analyse(activity_rows(), measured_rows()).to_dict()["quantity"]


def test_a_shuffled_measurement_does_not_rank():
    """The instrument must be able to say no. Re-labelling the measured powers
    breaks the correspondence without changing anything else, and the
    leave-one-out correlation has to fall away."""
    rows = measured_rows()
    permutation = {
        "idle-0": "mm-4096",
        "mm-4096": "idle-0",
        "rv-200000": "sfpu-4096",
        "sfpu-4096": "rv-200000",
    }
    for row in rows:
        row["label"] = permutation.get(row["label"], row["label"])
    report = analyse(activity_rows(), rows)
    assert report.ok, render(report)
    assert report.spearman_loo < 0.9


# ---------------------------------------------------------------------------
# The parameterisation: the floor is FITTED, not subtracted
# ---------------------------------------------------------------------------


def test_the_floor_is_fitted_from_the_arms_and_lands_on_the_truth():
    """The synthetic board's arms are generated as ``P = 40 + rate*E``, and
    nothing tells the fit what the 40 is: it comes back out of the arm rows
    alone, together with the launch constant it has to be told apart from."""
    report = analyse(activity_rows(), measured_rows())
    assert report.p_floor_w == pytest.approx(BASELINE_W, abs=1e-6)
    assert report.coefficients["launch"] == pytest.approx(TRUE["launch"], rel=1e-6)
    for term, value in TRUE.items():
        if term != "launch":
            assert report.coefficients[term] == pytest.approx(value, rel=1e-6)


def test_the_fitted_floor_and_not_the_measured_baseline_defines_the_energies():
    """A session whose baseline slot is 20 W below the busy floor -- which is
    what a DVFS drop looks like -- must produce the SAME per-launch energies as
    one whose baseline happens to sit at the busy floor. If the baseline still
    fed the arithmetic, every energy here would be 20 W/rate too large."""
    rows = measured_rows()
    dropped = [
        {**r, "power_w": r["power_w"] - 20.0} if r["label"] == BASELINE_LABEL else r
        for r in measured_rows()
    ]
    a, b = analyse(activity_rows(), rows), analyse(activity_rows(), dropped)
    assert b.baseline_w == pytest.approx(a.baseline_w - 20.0)
    assert b.p_floor_w == pytest.approx(a.p_floor_w)
    for label in a.measured_energy:
        assert b.measured_energy[label] == pytest.approx(a.measured_energy[label])
        assert a.measured_energy[label] == pytest.approx(true_energy(label), rel=1e-6)


def test_the_report_and_the_record_state_the_new_model_form(tmp_path):
    report = analyse(activity_rows(), measured_rows())
    text = render(report)
    assert MODEL_FORM in text
    assert "P_floor" in text
    assert "NOT subtracted" in text
    doc = coefficients_document(report, {"measured_csv": "power.csv"})
    assert "p_floor_w:" in doc
    assert "P_floor" in doc
    assert report.to_dict()["p_floor_w"] == pytest.approx(report.p_floor_w)


# ---------------------------------------------------------------------------
# Gate: baseline
# ---------------------------------------------------------------------------


def test_baseline_gate_passes_when_an_in_session_baseline_exists():
    assert gate(analyse(activity_rows(), measured_rows()), "baseline").passed


def test_baseline_gate_refuses_a_session_with_no_baseline():
    report = analyse(activity_rows(), measured_rows(include_baseline=False))
    assert not report.ok
    assert not gate(report, "baseline").passed
    assert "REFUSED" in render(report)


# ---------------------------------------------------------------------------
# Gate: telemetry -- the 2026-08-13 failure, in both directions
# ---------------------------------------------------------------------------


def test_telemetry_gate_passes_when_every_slot_was_actually_measured():
    assert gate(analyse(activity_rows(), measured_rows()), "telemetry").passed


def test_telemetry_gate_refuses_a_slot_that_recorded_zero_samples():
    """The whole reason this gate exists. On 2026-08-13 every arm slot of a
    three-cycle card session came back ``samples=0, power_w=0`` -- tt-smi 3.0.32
    panicked on a device held by tt-metal, the sampler swallowed the exception,
    and ``mean([]) -> 0.0`` turned a refusal into a reading. The session ran to
    completion and looked finished. It must now be impossible to mistake that for
    success."""
    rows = measured_rows()
    for row in rows:
        if row["label"] != BASELINE_LABEL:  # only the baselines got telemetry
            row["samples"] = 0
            row["power_w"] = 0.0
    report = analyse(activity_rows(), rows)
    assert not gate(report, "telemetry").passed
    assert "0 telemetry samples" in gate(report, "telemetry").detail
    assert not report.ok
    assert "REFUSED" in render(report)


def test_telemetry_gate_refuses_an_empty_power_cell_rather_than_reading_it_as_zero():
    """``aggregate_power.py`` writes an empty cell for a reading never taken;
    ``load_measured`` turns that into ``nan``. A ``nan`` must refuse, not
    propagate into a mean."""
    rows = measured_rows()
    rows[3]["power_w"] = float("nan")
    report = analyse(activity_rows(), rows)
    assert not gate(report, "telemetry").passed
    assert "no power reading" in gate(report, "telemetry").detail
    assert not report.ok


def test_telemetry_gate_refuses_a_status_the_runner_already_knew_was_bad():
    rows = measured_rows(
        health=lambda c, label: {"status": "run_failed"} if c == 1 else None
    )
    report = analyse(activity_rows(), rows)
    assert not gate(report, "telemetry").passed
    assert "run_failed" in gate(report, "telemetry").detail
    assert not report.ok


def test_an_empty_power_cell_loads_as_nan_and_not_as_zero(tmp_path):
    """The contract between the aggregator and the analysis, at the seam. If an
    empty cell parsed as 0.0 here, the gate above could never fire."""
    csv_path = tmp_path / "power.csv"
    csv_path.write_text(
        "label,cycle,slot,power_w,samples,launches,wall_s,launches_per_s\n"
        "baseline,0,0,,0,0,30,0\n"
        "idle-0,0,1,41.5,40,1000,30,33.3\n"
    )
    rows = load_measured(csv_path)
    assert math.isnan(rows[0]["power_w"])
    assert rows[1]["power_w"] == pytest.approx(41.5)


# ---------------------------------------------------------------------------
# Gate: schedule -- cycle 2's missing slot
# ---------------------------------------------------------------------------


def test_schedule_gate_passes_when_every_cycle_carries_every_label():
    assert gate(analyse(activity_rows(), measured_rows()), "schedule").passed


def test_schedule_gate_refuses_a_cycle_with_a_hole_in_it():
    """Cycle 2 of the 2026-08-13 session was missing ``idle-0`` outright: the CSV
    jumped slot 0 to slot 2, because the runner wrote a manifest row only for a
    slot that succeeded, so a crash left no trace in either machine-readable
    output. Nothing looked, and the gap was found by eye."""
    rows = [
        r for r in measured_rows() if not (r["cycle"] == 2 and r["label"] == "idle-0")
    ]
    report = analyse(activity_rows(), rows)
    g = gate(report, "schedule")
    assert not g.passed
    assert "cycle 2 is missing idle-0" in g.detail
    assert not report.ok


# ---------------------------------------------------------------------------
# Gate: thermal
# ---------------------------------------------------------------------------


def test_thermal_gate_passes_when_the_trip_counter_held_still():
    assert gate(analyse(activity_rows(), measured_rows()), "thermal").passed


def test_thermal_gate_refuses_a_session_where_the_part_throttled():
    rows = measured_rows(
        health=lambda c, label: (
            {"therm_trip_delta": 1.0} if c == 3 and label == "mm-16384" else None
        )
    )
    report = analyse(activity_rows(), rows)
    assert not gate(report, "thermal").passed
    assert "throttled" in gate(report, "thermal").detail
    assert not report.ok


def test_thermal_gate_refuses_a_session_with_no_thermal_record_at_all():
    rows = measured_rows(health=lambda c, label: {"therm_trip_delta": None})
    report = analyse(activity_rows(), rows)
    assert not gate(report, "thermal").passed
    assert "cannot show the part did not" in gate(report, "thermal").detail


# ---------------------------------------------------------------------------
# Gate: clock -- the 42% baseline swing
# ---------------------------------------------------------------------------


def test_clock_gate_passes_when_the_board_held_one_clock():
    assert gate(analyse(activity_rows(), measured_rows()), "clock").passed


def test_clock_gate_refuses_slots_measured_at_different_clocks():
    """The 2026-08-13 baselines: 61.7 W at 1350 MHz in cycle 0, ~39 W at 800 MHz
    in cycles 1 and 2. Each slot was steady; they were differenced against each
    other anyway, and 42% of the reference moved."""
    rows = measured_rows(
        health=lambda c, label: {"aiclk_mean": 800.0} if c >= 1 else None
    )
    report = analyse(activity_rows(), rows)
    g = gate(report, "clock")
    assert not g.passed
    assert "different clocks" in g.detail
    assert not report.ok


def test_a_clock_that_moved_inside_one_slot_excludes_that_row_not_the_session():
    """One contaminated slot must not discard the sound ones. A row whose clock
    moved *during* it is a mean over two DVFS states, so it is cut out of the fit
    and named -- and the session carries on, because three other cycles of that
    arm are untouched."""
    rows = measured_rows(
        cycles=4,
        health=lambda c, label: (
            {"aiclk_drift_pct": 41.4} if c == 1 and label == "rv-800000" else None
        ),
    )
    report = analyse(activity_rows(), rows)
    g = gate(report, "clock")
    assert g.passed, g.detail
    assert "EXCLUDED 1 row(s)" in g.detail
    assert len(report.excluded_rows) == 1
    assert "cycle 1 slot 6 rv-800000" in report.excluded_rows[0]
    assert "41.40%" in report.excluded_rows[0]
    assert report.ok, render(report)
    # ...and it is in the report, not merely absent from the fit.
    assert "Rows excluded from the fit" in render(report)
    assert report.to_dict()["excluded_rows"] == report.excluded_rows


def test_an_excluded_row_is_not_averaged_into_its_label():
    """The exclusion has to actually change the arithmetic, or it is decoration.
    A wild power on the drifted row moves the fitted energy if it is averaged in,
    and must not."""

    def health(c, label):
        if c == 1 and label == "rv-800000":
            return {"aiclk_drift_pct": 41.4}
        return None

    clean = analyse(activity_rows(), measured_rows(cycles=4))
    rows = measured_rows(cycles=4, health=health)
    for row in rows:
        if row["cycle"] == 1 and row["label"] == "rv-800000":
            row["power_w"] += 25.0
    contaminated = analyse(activity_rows(), rows)
    assert contaminated.measured_energy["rv-800000"] == pytest.approx(
        clean.measured_energy["rv-800000"], rel=1e-9
    )


def test_excluding_rows_can_still_cost_the_session_through_repeats():
    """The other side of the exclusion, and the one that keeps it honest: if
    cutting the contaminated rows out leaves too few, `repeats` refuses. That is
    the refusal an operator can act on -- run more cycles -- rather than one that
    says only that a clock moved."""
    rows = measured_rows(
        cycles=3,
        health=lambda c, label: (
            {"aiclk_drift_pct": 41.4} if c == 1 and label == "rv-800000" else None
        ),
    )
    report = analyse(activity_rows(), rows)
    assert gate(report, "clock").passed
    g = gate(report, "repeats")
    assert not g.passed
    assert "'rv-800000': 2" in g.detail
    assert "more" in g.detail
    assert "cycles" in g.detail
    assert not report.ok


def test_a_row_inside_the_drift_cap_is_not_excluded():
    """The threshold is a threshold: 4.9% stays in, 5.1% comes out."""

    def drift(pct):
        return measured_rows(
            cycles=4,
            health=lambda c, label: (
                {"aiclk_drift_pct": pct} if c == 1 and label == "rv-800000" else None
            ),
        )

    assert analyse(activity_rows(), drift(4.9)).excluded_rows == []
    assert len(analyse(activity_rows(), drift(5.1)).excluded_rows) == 1


def test_clock_gate_refuses_a_session_with_no_clock_record():
    rows = measured_rows(health=lambda c, label: {"aiclk_mean": None})
    report = analyse(activity_rows(), rows)
    assert not gate(report, "clock").passed
    assert "no clock record" in gate(report, "clock").detail


def test_clock_gate_tolerance_is_a_threshold_and_not_a_switch():
    """Both sides of the same knob: a 4% spread passes a 5% cap and refuses a 2%
    one. A gate whose threshold does nothing is not a threshold."""
    rows = measured_rows(
        health=lambda c, label: {"aiclk_mean": 1350.0 if c % 2 else 1296.0}
    )
    assert gate(analyse(activity_rows(), rows, max_clock_drift_pct=5.0), "clock").passed
    assert not gate(
        analyse(activity_rows(), rows, max_clock_drift_pct=2.0), "clock"
    ).passed


def test_clock_gate_ignores_the_baseline_because_that_confound_is_structural():
    """The across-session half runs over the ARM rows only. A baseline at the
    board's idle DVFS state is not drift and cannot be fixed by re-running, so
    including it made this gate unsatisfiable rather than strict -- the arms here
    all hold 1350 MHz and the clock gate has nothing to say, while
    ``baseline_clock`` says the specific thing that is true."""
    rows = measured_rows(
        health=lambda c, label: (
            {"aiclk_mean": 800.0} if label == BASELINE_LABEL else None
        )
    )
    report = analyse(activity_rows(), rows)
    g = gate(report, "clock")
    assert g.passed, g.detail
    assert "arm rows" in g.detail
    assert not gate(report, "baseline_clock").passed


# ---------------------------------------------------------------------------
# Gate: baseline_clock -- a finding, not a refusal
# ---------------------------------------------------------------------------


def test_baseline_clock_gate_passes_when_the_baseline_shared_the_arms_clock():
    g = gate(analyse(activity_rows(), measured_rows()), "baseline_clock")
    assert g.passed, g.detail
    assert g.advisory


def test_baseline_clock_gate_reports_a_dvfs_split_without_refusing_the_session():
    """The 2026-08-13 p150 session: every baseline at 800 MHz, every arm at 1350.
    That is two DVFS states rather than drift, tt-smi 6.2.0 has no clock pinning,
    and a refusal would therefore be a gate that can never pass. What it means is
    narrower and is what gets said: baseline subtraction is invalid here. The
    session still analyses, because the fit no longer subtracts the baseline."""
    rows = measured_rows(
        health=lambda c, label: (
            {"aiclk_mean": 800.0} if label == BASELINE_LABEL else None
        )
    )
    report = analyse(activity_rows(), rows)
    g = gate(report, "baseline_clock")
    assert not g.passed
    assert g.advisory
    assert "BASELINE SUBTRACTION IS INVALID" in g.detail
    # ...and the session is NOT refused on it, and still fits.
    assert report.ok, render(report)
    assert report.spearman_loo == pytest.approx(1.0)
    text = render(report)
    assert "[FINDING] baseline_clock" in text
    assert "REFUSED" not in text


def test_an_advisory_finding_is_still_reported_in_the_json_and_the_yaml():
    rows = measured_rows(
        health=lambda c, label: (
            {"aiclk_mean": 800.0} if label == BASELINE_LABEL else None
        )
    )
    report = analyse(activity_rows(), rows)
    payload = report.to_dict()
    entry = next(g for g in payload["gates"] if g["name"] == "baseline_clock")
    assert entry["advisory"] is True
    assert entry["passed"] is False
    assert payload["refused"] is False
    doc = coefficients_document(report, {"measured_csv": "power.csv"})
    assert "FINDING: baseline" in doc


def test_a_refusing_gate_is_not_advisory_and_still_refuses():
    """The other direction of the advisory flag itself: it is one check's
    property, not a global softening. A drifted control still throws the session
    away."""
    report = analyse(activity_rows(), measured_rows(noise=0.05, control_offset=6.0))
    g = gate(report, "control")
    assert not g.advisory
    assert g.refuses
    assert not report.ok


# ---------------------------------------------------------------------------
# Gate: repeats
# ---------------------------------------------------------------------------


def test_repeats_gate_passes_with_enough_interleave_cycles():
    assert gate(analyse(activity_rows(), measured_rows(cycles=4)), "repeats").passed


def test_repeats_gate_refuses_a_single_cycle():
    report = analyse(activity_rows(), measured_rows(cycles=1))
    assert not gate(report, "repeats").passed
    assert not report.ok


# ---------------------------------------------------------------------------
# Gate: samples
# ---------------------------------------------------------------------------


def test_samples_gate_passes_when_every_row_has_enough_telemetry():
    assert gate(analyse(activity_rows(), measured_rows(samples=40)), "samples").passed


def test_samples_gate_refuses_an_under_sampled_arm():
    rows = measured_rows(samples=40)
    for row in rows:
        if row["label"] == "mm-4096":
            row["samples"] = 3
    report = analyse(activity_rows(), rows)
    assert not gate(report, "samples").passed
    assert "mm-4096" in gate(report, "samples").detail
    assert not report.ok


# ---------------------------------------------------------------------------
# Gate: control (the verified zero)
# ---------------------------------------------------------------------------


def test_control_gate_passes_when_the_same_arm_measures_the_same_twice():
    report = analyse(activity_rows(), measured_rows(noise=0.05, control_offset=0.0))
    assert gate(report, "control").passed, gate(report, "control").detail


def test_control_gate_refuses_a_session_that_drifted():
    """The control arm is the SAME workload in a different slot, so its true
    delta is zero by construction. A non-zero one is drift, and drift makes
    every other difference in the session unsafe."""
    report = analyse(activity_rows(), measured_rows(noise=0.05, control_offset=6.0))
    assert not gate(report, "control").passed
    assert not report.ok


def test_control_gate_refuses_a_session_with_no_control_at_all():
    report = analyse(activity_rows(), measured_rows(control=None))
    assert not gate(report, "control").passed
    assert "verified-zero" in gate(report, "control").detail
    assert not report.ok


# ---------------------------------------------------------------------------
# The noise floor: the arms' repeatability, and not the baseline's
# ---------------------------------------------------------------------------


def _swing_the_baseline(rows, amplitude):
    """Make the baseline label alternate +/- ``amplitude`` between cycles.

    This is what a DVFS board does to an idle slot: the real 2026-08-13 session
    swung its three baselines over 4.3 W while no arm moved by more than 1.2 W.
    Only the baseline rows are touched, so any change downstream is attributable
    to the baseline alone.
    """
    for row in rows:
        if row["label"] == BASELINE_LABEL:
            row["power_w"] += amplitude * (1.0 if row["cycle"] % 2 else -1.0)
    return rows


def _contaminated_floor(rows):
    """The floor as it was computed before the fix: every label, baseline and
    all. Recomputed here rather than imported, so the test states the old
    arithmetic instead of trusting the module to still contain it."""
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row["power_w"])
    sds = [
        float(np.std(powers, ddof=1)) for powers in by_label.values() if len(powers) > 1
    ]
    return float(np.sqrt(np.mean(np.square(sds))))


def test_a_wildly_swinging_baseline_does_not_move_the_noise_floor():
    """The baseline is a diagnostic in a different DVFS state, so its spread is
    not the noise of the busy-state rows and must set the scale for nothing.
    Two sessions identical but for an 8 W alternation on the idle slot: the
    floor is unchanged to the last digit, and so is the threshold the `control`
    gate is measured against."""
    quiet_rows = measured_rows(noise=0.05, control_offset=0.1)
    wild_rows = _swing_the_baseline(measured_rows(noise=0.05, control_offset=0.1), 8.0)
    # The two sessions differ in the baseline and in nothing else...
    assert [r["power_w"] for r in quiet_rows if r["label"] != BASELINE_LABEL] == [
        r["power_w"] for r in wild_rows if r["label"] != BASELINE_LABEL
    ]
    # ...and the old arithmetic would have read that as 50x the noise.
    assert _contaminated_floor(wild_rows) > 50 * _contaminated_floor(quiet_rows)

    quiet = analyse(activity_rows(), quiet_rows)
    wild = analyse(activity_rows(), wild_rows)
    assert wild.baseline_w == pytest.approx(quiet.baseline_w, abs=1e-9)  # 4 cycles
    assert wild.noise_floor_w == pytest.approx(quiet.noise_floor_w, abs=1e-12)
    assert gate(wild, "control").detail == gate(quiet, "control").detail
    assert gate(wild, "control").passed == gate(quiet, "control").passed
    assert gate(wild, "spread").detail == gate(quiet, "spread").detail


def test_a_control_that_drifted_is_refused_even_when_a_wild_baseline_widens_it():
    """The failure the fix prevents, built end to end.

    The control is 0.5 W from its twin -- genuine drift, five times the arms'
    own repeatability -- and the baseline swings 8 W between cycles. Under the
    old arithmetic that swing dragged the floor to 2.8 W, the `control` gate's
    bar to 8.4 W, and the session sailed through every gate it has. Against the
    arms' own repeatability the bar is 0.12 W and the session is refused, which
    is the whole point of running a verified-zero control."""
    rows = _swing_the_baseline(measured_rows(noise=0.05, control_offset=0.5), 8.0)
    report = analyse(activity_rows(), rows)

    old_floor = _contaminated_floor(rows)
    assert old_floor == pytest.approx(2.79, abs=0.05)
    assert report.noise_floor_w == pytest.approx(0.040, abs=0.005)

    control = gate(report, "control")
    assert not control.passed
    assert not report.ok
    # The drift is real and unchanged; only the yardstick moved. It would have
    # been inside the old bar with room to spare.
    mean_of = lambda label: float(  # noqa: E731
        np.mean([r["power_w"] for r in rows if r["label"] == label])
    )
    delta = abs(mean_of("noc-4096" + CONTROL_SUFFIX) - mean_of("noc-4096"))
    assert delta == pytest.approx(0.5, abs=0.05)
    assert delta < 3.0 * old_floor
    assert delta > 3.0 * report.noise_floor_w
    assert f"{3.0 * report.noise_floor_w:.3f} W" in control.detail
    # ...and `control` is the ONLY thing wrong with this session, so under the
    # old floor it would have been fitted and reported.
    assert [g.name for g in report.gates if g.refuses] == ["control"]


# ---------------------------------------------------------------------------
# Gate: spread
# ---------------------------------------------------------------------------


def test_spread_gate_passes_when_the_arms_separate():
    report = analyse(activity_rows(), measured_rows(noise=0.05))
    assert gate(report, "spread").passed, gate(report, "spread").detail


def test_spread_gate_refuses_arms_inside_the_noise_floor():
    """Same ordering, same everything -- but the deltas are squeezed to a
    thousandth while the sample noise stays put. There is still an ordering in
    the numbers, and reporting it would be reporting a ranking of noise."""
    report = analyse(activity_rows(), measured_rows(noise=0.5, scale=1e-3))
    assert not gate(report, "spread").passed
    assert "nothing to rank" in gate(report, "spread").detail
    assert not report.ok


# ---------------------------------------------------------------------------
# Gate: identifiability
# ---------------------------------------------------------------------------


def test_identifiability_gate_passes_for_the_automatic_selection():
    report = analyse(activity_rows(), measured_rows())
    g = gate(report, "identifiability")
    assert g.passed, g.detail
    # terms + launch + floor, with a degree of freedom left for leave-one-out.
    assert len(report.terms) + 2 <= len(report.labels) - 1


def test_the_count_bound_fires_on_its_own_and_not_only_on_conditioning():
    """The two halves of this gate are separable, and the count half is the one
    the floor column changed: it now spends two coefficients before any activity
    term. Seven well-conditioned terms is nine coefficients against nine
    workloads, which leaves leave-one-out nothing to predict with; six is eight,
    and passes at the same conditioning."""
    rng = np.random.default_rng(3)
    extra = {
        label: {
            "noc_txns": 100 + int(rng.integers(1, 900)),
            "tensix_dispatch": 200 + int(rng.integers(1, 900)),
            "rv_stall_cycles": 300 + int(rng.integers(1, 900)),
        }
        for label in ACTIVITY
    }
    terms = [
        "instr_retired",
        "noc_bytes_total",
        "matrix_arith_cycles",
        "sfpu_busy_cycles",
        "noc_txns",
        "tensix_dispatch",
        "rv_stall_cycles",
    ]
    too_many = gate(
        analyse(activity_rows(extra=extra), measured_rows(), terms=terms),
        "identifiability",
    )
    assert not too_many.passed
    assert "9 coefficients" in too_many.detail
    assert "condition number 950" in too_many.detail  # ...and NOT on conditioning
    enough = gate(
        analyse(activity_rows(extra=extra), measured_rows(), terms=terms[:6]),
        "identifiability",
    )
    assert enough.passed, enough.detail
    assert "8 coefficients" in enough.detail


def test_identifiability_gate_refuses_more_coefficients_than_workloads_support():
    report = analyse(
        activity_rows(),
        measured_rows(),
        terms=[
            "instr_retired",
            "noc_bytes_total",
            "matrix_busy_cycles",
            "sfpu_busy_cycles",
            "tensix_dispatch",
        ],
    )
    assert not gate(report, "identifiability").passed
    assert not report.ok


def test_identifiability_gate_refuses_a_collinear_pair():
    """``noc_txns`` is made an exact multiple of ``noc_bytes_total`` plus a
    speck, so the pair is collinear without being an exact duplicate: the
    selector keeps it and the gate is what refuses it."""
    extra = {}
    for label, terms in ACTIVITY.items():
        extra[label] = {
            "noc_txns": terms["noc_bytes_total"] / 2048.0 + 1e-9,
        }
    report = analyse(
        activity_rows(extra=extra),
        measured_rows(),
        terms=["noc_bytes_total", "noc_txns"],
        max_cond=100.0,
    )
    g = gate(report, "identifiability")
    assert not g.passed, g.detail
    assert not report.ok


def test_the_selector_keeps_an_ill_conditioned_pair_for_the_gate_to_judge():
    """The other half of the same point: selection excludes only exact
    duplicates, so an ill-conditioned column reaches the gate. A selector that
    filtered on the gate's threshold would make the gate unable to fail."""
    matrix = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0 + 1e-7]])
    chosen = select_terms(matrix, ["a", "b"], budget=2)
    assert sorted(chosen) == [0, 1]
    assert design_condition(np.column_stack([np.ones(3), matrix])) > 100.0
    # An EXACT duplicate is dropped, because that is arithmetic, not judgement.
    dup = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert select_terms(dup, ["a", "b"], budget=2) == [0]


# ---------------------------------------------------------------------------
# The term set: the design's, not a search over the data
# ---------------------------------------------------------------------------


def test_the_designed_term_set_is_one_term_per_non_idle_arm():
    """The mapping is the README's arms table and nothing else, and the function
    that reads it is handed **arm names**. No matrix, no target, no condition
    number -- which is the structural reason restricting the fit to it is
    honouring the design rather than searching over the data."""
    assert DESIGNED_ARM_TERMS["idle"] is None  # moves nothing; that is the point
    assert designed_terms(["idle", "rv", "noc", "mm", "sfpu"]) == [
        "instr_retired",
        "noc_bytes_total",
        "sfpu_busy_cycles",
        "matrix_arith_cycles",
    ]
    # The `mm` arm is fitted against matrix ARITHMETIC, not matrix occupancy.
    # `matrix_busy_cycles` is a column BOTH compute arms move -- the sfpu arm
    # by 41 cycles of RWC bookkeeping per iteration -- so it is not a statement
    # of what one arm was built to isolate. See the 2026-08-13 entry in
    # perfbench/energybench/README.md.
    assert DESIGNED_ARM_TERMS["mm"] == "matrix_arith_cycles"
    assert "matrix_busy_cycles" not in DESIGNED_ARM_TERMS.values()
    # Schema order, whatever order the arms ran in, and duplicates collapse.
    assert designed_terms(["sfpu", "rv", "sfpu"]) == [
        "instr_retired",
        "sfpu_busy_cycles",
    ]
    # An arm the design never described is a hard error, not a quietly smaller
    # model than the one the report describes.
    with pytest.raises(ValueError, match="eth"):
        designed_terms(["rv", "eth"])


def test_an_unknown_arm_stops_the_analysis_rather_than_fitting_a_smaller_model():
    rows = activity_rows()
    for row in rows:
        if row["label"] == "mm-4096":
            row["arm"] = "tensor-core"
    with pytest.raises(ValueError, match="tensor-core"):
        analyse(rows, measured_rows())


def test_a_designed_term_with_no_spread_is_dropped_and_named():
    """The stated degradation. An occupancy column is *absent, not zero*, without
    ``TT_SIM_COST_MODEL=1``, so a cost-model-off matrix has a designed term the
    fit cannot use. It is dropped -- arithmetic, a constant column -- and said so
    in the report, because a model quietly smaller than the one described is the
    failure this module is built around."""
    flat = {label: {"sfpu_busy_cycles": 0.0} for label in ACTIVITY}
    report = analyse(activity_rows(extra=flat), measured_rows())
    assert "sfpu_busy_cycles" not in report.terms
    assert report.term_source == "designed"
    note = next(n for n in report.notes if "designed term(s) not in the fit" in n)
    assert "sfpu_busy_cycles" in note


def test_the_designed_set_can_still_fail_the_gate():
    """The requirement that keeps this from being a rewrite of the guard: a
    session whose *designed* columns are collinear is refused, exactly as before.
    Here the ``sfpu`` arm's column is made a near-multiple of the ``mm`` arm's, as
    it would be if the two arms stopped being separable in silicon, and the
    session is refused on conditioning even though every fitted term is one the
    arms table names."""
    extra = {
        # A near-multiple, not an exact one: an exact duplicate is dropped by the
        # selector as arithmetic, and what has to be shown here is that a merely
        # ill-conditioned designed pair reaches the gate and is refused by it.
        label: {
            "sfpu_busy_cycles": 3.0 * terms["matrix_arith_cycles"] * (1.0 + 1e-7 * k)
        }
        for k, (label, terms) in enumerate(ACTIVITY.items())
    }
    report = analyse(activity_rows(extra=extra), measured_rows())
    g = gate(report, "identifiability")
    assert report.term_source == "designed"
    assert sorted(report.terms) == sorted(DESIGN_TERMS)
    assert not g.passed, g.detail
    # Refused on CONDITIONING with all four designed terms present -- not on the
    # coefficient count, and not by having quietly dropped one.
    assert "6 coefficients (4 designed terms + launch + floor)" in g.detail
    assert "condition number 2.17e+07" in g.detail
    assert [x.name for x in report.gates if x.refuses] == ["identifiability"]
    assert not report.ok
    # ...and the same design with the collinear pair broken passes, so this is a
    # statement about that session and not about the designed set as such.
    assert gate(analyse(activity_rows(), measured_rows()), "identifiability").passed


def test_naming_terms_outside_the_design_is_stamped_operator_specified():
    """A term outside the designed set gets in exactly one way -- a human writing
    it down with ``--terms`` -- and every report of that fit says so, so nobody
    later reads an overridden model as the design's."""
    report = analyse(
        activity_rows(), measured_rows(), terms=["instr_retired", "noc_bytes_total"]
    )
    assert report.term_source == "operator-specified"
    assert report.terms == ["instr_retired", "noc_bytes_total"]
    text = render(report)
    assert "operator-specified" in text
    assert "pre-registered by the operator" in text
    assert "designed terms" not in gate(report, "identifiability").detail
    # The default path says the other thing, in the same places.
    default = render(analyse(activity_rows(), measured_rows()))
    assert "fitted terms (designed)" in default
    assert "fixed a priori by the energybench arms table" in default


# ---------------------------------------------------------------------------
# Gate: target_triviality
# ---------------------------------------------------------------------------


def test_target_triviality_gate_passes_when_the_arms_differ_by_more_than_rate():
    report = analyse(activity_rows(), measured_rows())
    g = gate(report, "target_triviality")
    assert g.passed, g.detail
    assert report.target_triviality_r2 < 0.5


def test_target_triviality_gate_refuses_a_session_that_is_only_a_launch_rate():
    """Every workload given the SAME joules per launch, so the board's power
    differs only in how often each one launched. There is still an ordering, a
    spread far outside the noise floor and a clean control -- and a fit would
    still report a Spearman -- but the ranking carries no information about
    activity, and this is the only check in the set that can see that."""
    report = analyse(activity_rows(), measured_rows(flat_energy=1.0e-3))
    g = gate(report, "target_triviality")
    assert not g.passed, g.detail
    assert report.target_triviality_r2 > 0.99
    assert "without knowing any activity" in g.detail
    assert not report.ok
    # The point of the gate is that NOTHING ELSE catches this session.
    assert [x.name for x in report.gates if x.refuses] == ["target_triviality"]


def test_target_triviality_threshold_is_a_threshold_and_not_a_switch():
    rows = measured_rows(flat_energy=1.0e-3)
    assert not gate(
        analyse(activity_rows(), rows, max_triviality_r2=0.95), "target_triviality"
    ).passed
    # Above 1.0 nothing can trip it, which is how a knob is shown to be a knob.
    assert gate(
        analyse(activity_rows(), rows, max_triviality_r2=1.01), "target_triviality"
    ).passed


# ---------------------------------------------------------------------------
# Gate: rankable
# ---------------------------------------------------------------------------


def test_rankable_gate_passes_with_five_workloads():
    assert gate(analyse(activity_rows(), measured_rows()), "rankable").passed


def test_rankable_gate_refuses_two_workloads():
    labels = ["idle-0", "mm-4096"]
    report = analyse(
        activity_rows(labels), measured_rows(labels=labels, control="mm-4096")
    )
    assert not gate(report, "rankable").passed
    assert "at least 3" in gate(report, "rankable").detail
    assert not report.ok


# ---------------------------------------------------------------------------
# Missing activity vectors
# ---------------------------------------------------------------------------


def test_a_measured_arm_with_no_activity_vector_is_dropped_and_said_so():
    rows = activity_rows([label for label in ACTIVITY if label != "sfpu-4096"])
    report = analyse(rows, measured_rows())
    assert "sfpu-4096" in " ".join(report.notes)
    assert "sfpu-4096" not in report.labels


# ---------------------------------------------------------------------------
# The real session: Blackhole p150, tt-smi 6.2.0, 2026-08-13
# ---------------------------------------------------------------------------

#: The first card session that collected. Preserved in the tree, verbatim, and
#: read here rather than retyped: `power.csv` is the file the card box wrote,
#: `raw/` is what makes any row of it diagnosable, and a fixture transcribed by
#: hand would be a fixture that can quietly stop matching the session it claims
#: to be. 33/33 slots `status=ok`, 29-37 telemetry samples each,
#: `attempts == samples`, zero `sample_failures`, `therm_trip_delta` 0
#: everywhere, launch rates reproducible across the three cycles to ~1%.
#:
#: Nothing here fits a coefficient to it. This repo's activity vectors were
#: collected at SMOKE inner counts and are not these workloads' vectors, so what
#: the session is used for is the properties of the MEASUREMENT that drove the
#: changes above -- the triviality of the old target, the two clock findings, and
#: what board power can and cannot resolve.
REAL_SESSION_CSV = (
    Path(__file__).resolve().parents[2]
    / "perfbench"
    / "card-sessions"
    / "2026-08-13-energybench"
    / "power.csv"
)

#: The simulator-side activity vectors collected at the **card's** inner counts,
#: which is what makes them joinable with the session above. ``...-scale2`` is the
#: intermediate scale (rv-400000, noc-8192, mm-8192, sfpu-8192) plus a duplicate
#: ``idle-0`` row; it exists to answer "would a third scale help?" and the answer
#: is pinned below.
CARD_ACTIVITY_CSV = (
    Path(__file__).resolve().parents[2]
    / "perfbench"
    / "energybench"
    / "activity-sim-blackhole-card.csv"
)
CARD_ACTIVITY_SCALE2_CSV = (
    CARD_ACTIVITY_CSV.parent / "activity-sim-blackhole-card-scale2.csv"
)

#: **Both card CSVs predate ``matrix_arith_cycles``** (added 2026-08-13, when the
#: Matrix Unit's RWC bookkeeping was found to be 100 % of the ``sfpu`` arm's
#: matrix column). They carry ``matrix_busy_cycles`` and read back with the newer
#: term at 0, which is why the designed fit now drops it and says so -- asserted
#: directly below, because that refusal is the fix working.
#:
#: Re-reducing them means re-running the arms at the card's inner counts: ~22M
#: simulator cycles at ~900 cycles/s, about seven hours, and it has not been
#: done. The two tests that pin how *that session* conditions are about its
#: launch rates and its column geometry, neither of which the split moves, so
#: they name the vintage's own term set explicitly. Naming it is what keeps the
#: numbers below comparable with the ones the session was refused on.
LEGACY_CARD_TERMS = [
    "instr_retired",
    "noc_bytes_total",
    "matrix_busy_cycles",
    "sfpu_busy_cycles",
]


def real_session_rows():
    assert REAL_SESSION_CSV.exists(), (
        f"the preserved 2026-08-13 card session is missing from {REAL_SESSION_CSV}. "
        "It is the only real measurement this analysis has ever seen; the "
        "regression tests below are about that session and cannot be run without "
        "it."
    )
    return load_measured(REAL_SESSION_CSV)


def real_means(exclude_drifted=True):
    """Per-label mean power, rate and repeat count, as the analysis would take
    them -- i.e. after the within-slot clock exclusion, unless asked otherwise."""
    out = {}
    for row in real_session_rows():
        if exclude_drifted and (row["aiclk_drift_pct"] or 0.0) > 5.0:
            continue
        out.setdefault(row["label"], []).append(row)
    return {
        label: (
            float(np.mean([r["power_w"] for r in rows])),
            float(np.mean([r["rate"] for r in rows])),
            len(rows),
        )
        for label, rows in out.items()
    }


def test_the_preserved_session_is_the_one_that_was_reported():
    """A fixture that does not match the session it names is worse than none, so
    the collection facts are asserted before anything is concluded from them."""
    rows = real_session_rows()
    assert len(rows) == 33
    assert {r["status"] for r in rows} == {"ok"}
    assert min(r["samples"] for r in rows) >= 29
    assert all(r["samples"] == r["attempts"] for r in rows)
    assert {r["therm_trip_delta"] for r in rows} == {0.0}
    assert {r["tt_smi_version"] for r in rows} == {"6.2.0"}


def test_the_old_parameterisation_was_almost_exactly_one_over_the_launch_rate():
    """Why the arithmetic changed, in one number, from the real session.

    ``(P - P_baseline)/rate`` with a baseline 30 W below every arm is a ~30 W
    DVFS step plus 2.9 W of workload, divided by a rate that varies 5.3x. Fitted
    against ``1/rate`` alone -- floor and launch term, no activity whatsoever --
    it comes back at R^2 = 0.9995. A model reproducing that target would have
    reported an excellent leave-one-out Spearman for the ranking "whichever
    workload launches slowest costs most", and it passed `spread`, passed
    `control`, and passed `identifiability`, which inspects the design matrix
    rather than the target."""
    means = real_means()
    arms = [
        label
        for label in means
        if label not in (BASELINE_LABEL, "noc-4096" + CONTROL_SUFFIX)
    ]
    power = np.array([means[label][0] for label in arms])
    rate = np.array([means[label][1] for label in arms])
    baseline = means[BASELINE_LABEL][0]
    assert baseline == pytest.approx(37.736, abs=0.001)

    old_target = (power - baseline) / rate
    old = reduced_model_r2(
        np.column_stack([np.ones(len(arms)), 1.0 / rate]), old_target
    )
    assert old > 0.99
    assert old == pytest.approx(0.9995, abs=5e-4)

    # The new target is measured power itself, against [1, rate]. It does not
    # inherit the triviality: the floor is a fitted coefficient rather than a
    # constant stamped onto every point, so what is left to explain is the
    # workload difference and not the DVFS step.
    new = reduced_model_r2(np.column_stack([np.ones(len(arms)), rate]), power)
    assert new < 0.5
    assert new == pytest.approx(0.360, abs=0.01)


def test_the_real_session_clock_spread_passes_over_the_arms_and_would_not_over_all():
    """800-1350 MHz over every row, baselines included, is 42.3% -- a cap of 5%
    could never be met, because a slot that launches nothing is in the idle DVFS
    state by definition. Over the arm rows it is 1.59%, and over the arm rows the
    analysis actually fits (one excluded for drifting within itself) 0.007%. The
    same 5% cap, three different questions."""
    rows = real_session_rows()
    everything = [r["aiclk_mean"] for r in rows]
    arms = [r["aiclk_mean"] for r in rows if r["label"] != BASELINE_LABEL]
    kept_arms = [
        r["aiclk_mean"]
        for r in rows
        if r["label"] != BASELINE_LABEL and (r["aiclk_drift_pct"] or 0.0) <= 5.0
    ]
    pct = lambda c: 100.0 * (max(c) - min(c)) / np.mean(c)  # noqa: E731
    assert pct(everything) == pytest.approx(42.33, abs=0.02)
    assert pct(arms) == pytest.approx(1.586, abs=0.01)
    assert pct(kept_arms) == pytest.approx(0.0074, abs=0.001)
    assert pct(arms) < 5.0 < pct(everything)

    report = analyse(activity_rows(), rows)
    assert gate(report, "clock").passed, gate(report, "clock").detail
    finding = gate(report, "baseline_clock")
    assert not finding.passed
    assert finding.advisory
    assert "BASELINE SUBTRACTION IS INVALID" in finding.detail


def test_the_real_sessions_one_drifted_slot_is_excluded_and_costs_it_the_repeats():
    """Cycle 1's ``rv-800000`` straddled a DVFS transition: 41.4% drift within
    the slot, ``sysfs_aiclk`` 800-1350, and it shows in every other column too
    (``power_sd_w`` 5.14 against 0.1-0.9 elsewhere, a 75.0 W pre-idle probe,
    ``delta_w`` -6.81). It is cut out of the fit and named -- and because that
    leaves the arm with two repeats, `repeats` refuses. That is the right
    refusal, reached through the right gate, and it tells the operator to run
    more cycles."""
    report = analyse(activity_rows(), real_session_rows())
    assert len(report.excluded_rows) == 1
    assert "cycle 1 slot 6 rv-800000" in report.excluded_rows[0]
    assert "41.40%" in report.excluded_rows[0]
    assert "Rows excluded from the fit" in render(report)

    repeats = gate(report, "repeats")
    assert not repeats.passed
    assert "'rv-800000': 2" in repeats.detail
    assert not report.ok
    # ...and it is the ONLY refusal: nothing else about the session is broken.
    assert [g.name for g in report.gates if g.refuses] == ["repeats"]


def test_the_real_session_noise_floor_is_the_arms_and_not_the_baselines():
    """The floor is the repeatability of the rows that enter the fit.

    This session's three baselines read 34.97, 39.28 and 38.97 W -- an
    across-cycle SD of 2.40 W, against 0.09-1.15 W for every arm -- because they
    sat at 800 MHz while every arm ran at 1350. That is the DVFS state moving,
    not the instrument's repeatability, and letting it into the RMS inflates the
    floor from 0.441 W to 0.838 W: a factor of 1.9 on the yardstick that
    `control` and `spread` are measured against, in the same session whose
    `baseline_clock` finding says baseline subtraction is invalid.

    The control label IS in the floor -- it is a genuine arm measurement -- and
    keeping it is not the permissive choice: it lowers the floor here.
    """
    rows = real_session_rows()
    kept = [r for r in rows if (r["aiclk_drift_pct"] or 0.0) <= 5.0]
    by_label = {}
    for row in kept:
        by_label.setdefault(row["label"], []).append(row["power_w"])
    sds = {
        label: float(np.std(powers, ddof=1))
        for label, powers in by_label.items()
        if len(powers) > 1
    }
    rms = lambda vals: float(np.sqrt(np.mean(np.square(list(vals)))))  # noqa: E731

    assert sds[BASELINE_LABEL] == pytest.approx(2.403, abs=0.002)
    assert max(v for k, v in sds.items() if k != BASELINE_LABEL) == pytest.approx(
        1.149, abs=0.002
    )

    report = analyse(activity_rows(), rows)
    arms_only = rms(v for k, v in sds.items() if k != BASELINE_LABEL)
    contaminated = rms(sds.values())
    assert report.noise_floor_w == pytest.approx(arms_only, abs=1e-9)
    assert report.noise_floor_w == pytest.approx(0.4409, abs=0.001)
    # ...and materially below the baseline-contaminated value it used to be.
    assert contaminated == pytest.approx(0.8376, abs=0.001)
    assert contaminated / report.noise_floor_w == pytest.approx(1.90, abs=0.02)

    # The control is a repeat like any other, and dropping it would RAISE the
    # floor here (0.463 W), i.e. make its own gate more permissive.
    without_control = rms(
        v
        for k, v in sds.items()
        if k != BASELINE_LABEL and not k.endswith(CONTROL_SUFFIX)
    )
    assert without_control == pytest.approx(0.4634, abs=0.001)
    assert report.noise_floor_w < without_control


def test_the_real_session_passes_the_gates_it_was_reported_to_pass():
    """The session's own arithmetic, checked rather than taken on trust: a noise
    floor of 0.441 W over the arm rows, arms spanning 2.924 W (6.63x the floor),
    and a control 0.513 W from its twin (1.16x, still inside the 3-sigma bar).
    The fit itself is not asserted -- this repo's activity vectors are
    smoke-size and are not these workloads'."""
    report = analyse(activity_rows(), real_session_rows())
    assert report.noise_floor_w == pytest.approx(0.4409, abs=0.001)
    for name in (
        "telemetry",
        "schedule",
        "thermal",
        "clock",
        "samples",
        "spread",
        "target_triviality",
        "identifiability",
        "rankable",
    ):
        assert gate(report, name).passed, gate(report, name).detail
    control = gate(report, "control")
    assert control.passed
    assert "0.513" in control.detail
    # Against the arm-only floor, not the 2.513 W the baseline-contaminated one
    # would have allowed. The verdict does not flip here; the yardstick does.
    assert "3*noise = 1.323 W" in control.detail
    assert "2.924" in gate(report, "spread").detail
    assert report.baseline_w == pytest.approx(37.736, abs=0.001)


def test_a_stale_activity_csv_drops_the_matrix_term_and_names_it():
    """The 2026-08-13 matrix split, from the consumer's end.

    A CSV reduced before ``matrix_arith_cycles` existed reads the column back as
    zero in every row, so the designed fit **drops it and names it**. That is the
    whole point of adding a term rather than redefining ``matrix_busy_cycles`` in
    place: a stale CSV cannot quietly reproduce the old fit, which fitted the
    ``mm`` arm against a column the ``sfpu`` arm moved by 41 cycles per iteration
    of dest bookkeeping.

    The stale vectors are **synthesised here**, deliberately, rather than read
    from a preserved file. This test used to read the real card CSV and assert it
    was of the older vintage -- which made a data file that must never be
    regenerated into a test fixture, and the test duly broke the moment that file
    was regenerated (as it had to be, to give the session its ``mm`` direction).
    A behaviour worth pinning should not depend on a large artefact staying
    stale."""
    activity = [dict(row) for row in load_activity(CARD_ACTIVITY_CSV)]
    for row in activity:
        row["matrix_arith_cycles"] = 0.0
    assert any(row["matrix_busy_cycles"] > 0.0 for row in activity)

    report = analyse(activity, real_session_rows())
    assert report.term_source == "designed"
    assert "matrix_arith_cycles" not in report.terms
    note = next(n for n in report.notes if "designed term(s) not in the fit" in n)
    assert "matrix_arith_cycles" in note


def test_the_regenerated_card_vectors_carry_the_matrix_direction():
    """The converse, and the reason the fixture was regenerated.

    Post-split vectors separate the two compute arms: ``mm`` moves
    ``matrix_arith_cycles`` and ``sfpu`` leaves it flat while moving
    ``sfpu_busy_cycles``. With that column present the fit carries all four
    designed terms."""
    activity = load_activity(CARD_ACTIVITY_CSV)
    by_label = {row["label"]: row for row in activity}
    mm_lo, mm_hi = by_label["mm-4096"], by_label["mm-16384"]
    sf_lo, sf_hi = by_label["sfpu-4096"], by_label["sfpu-16384"]

    # The mm arm's characteristic term scales with its inner count...
    assert mm_hi["matrix_arith_cycles"] > 3.5 * mm_lo["matrix_arith_cycles"]
    # ...while the sfpu arm's matrix arithmetic is flat and near zero, which is
    # what the bookkeeping split was for: its 41 ops per iteration are all
    # INCRWC/SETRWC and none of them are arithmetic.
    assert sf_lo["matrix_arith_cycles"] == sf_hi["matrix_arith_cycles"]
    assert sf_hi["matrix_arith_cycles"] < 0.001 * sf_hi["matrix_busy_cycles"]
    assert sf_hi["sfpu_busy_cycles"] > 3.5 * sf_lo["sfpu_busy_cycles"]

    report = analyse(activity, real_session_rows())
    assert report.term_source == "designed"
    assert "matrix_arith_cycles" in report.terms


def test_the_real_session_identifies_its_designed_terms_and_still_fails_repeats():
    """The measurement joined to its own activity vectors, which is the pairing
    the session was collected for.

    Fitted with the four terms the arms were **built** to separate,
    ``identifiability`` passes at a condition number of 616 against a 1e6 cap.
    Fitted with the six the old ``n_workloads - 3`` budget reached for, over
    eleven mutually-correlated counters, the same session conditions at 2.34e7
    and is refused. The refusal was arithmetically correct and its cause was the
    budget, not the board.

    And the session is **still refused overall**, for the reason it was always
    refused for: cycle 1's ``rv-800000`` straddled a DVFS transition, so that arm
    has two usable repeats against a floor of three. That refusal is right, it is
    the one an operator can act on, and nothing here may make it go away.

    The term set is named explicitly -- see :data:`LEGACY_CARD_TERMS` -- and
    that is now a deliberate isolation rather than a fact about the fixture. The
    vectors were regenerated on 2026-08-13 and do carry ``matrix_arith_cycles``;
    naming the older set here holds the DATA fixed while varying only the TERMS,
    so 616 is attributable to the term set and to nothing else. Conditioning is a
    property of the columns and the rates. (With the current designed set, which
    swaps in ``matrix_arith_cycles``, the same session conditions at 598 --
    close, because the two matrix columns differ only by the ``sfpu`` arm's
    bookkeeping, which is exactly what the split removed.)"""
    activity = load_activity(CARD_ACTIVITY_CSV)
    report = analyse(activity, real_session_rows(), terms=LEGACY_CARD_TERMS)

    assert report.term_source == "operator-specified"
    assert sorted(report.terms) == sorted(LEGACY_CARD_TERMS)
    ident = gate(report, "identifiability")
    assert ident.passed, ident.detail
    assert (
        "6 coefficients (4 operator-specified terms + launch + floor)" in ident.detail
    )
    assert "condition number 616" in ident.detail

    # The other direction, on the same session: the six terms the old budget
    # selected, named explicitly, are refused on conditioning.
    over_reached = analyse(
        activity,
        real_session_rows(),
        terms=[
            "noc_bytes_total",
            "noc_flight_cycles",
            "noc_txns",
            "tensix_stall_cycles",
            "thcon_busy_cycles",
            "rv_stall_cycles",
        ],
    )
    old = gate(over_reached, "identifiability")
    assert not old.passed
    assert "8 coefficients" in old.detail
    assert "2.34e+07" in old.detail

    # ...and the session's real refusal is untouched. `repeats`, and only that.
    assert [g.name for g in report.gates if g.refuses] == ["repeats"]
    assert "'rv-800000': 2" in gate(report, "repeats").detail
    assert not report.ok


def _rate_model(label_rates: dict) -> dict:
    """Launch rates for the intermediate scale, from the two the card measured.

    A launch period is a fixed overhead plus a per-inner-iteration cost, so
    ``1/rate`` is linear in ``inner`` and two measured points determine it. This
    is a **test fixture assumption**, stated here rather than buried: the card
    never ran the intermediate scale, and the question these rates serve is
    structural (how large does the term budget grow?) rather than numerical.
    """
    out = dict(label_rates)
    for arm, small, big, mid in (
        ("rv", 200_000, 800_000, 400_000),
        ("noc", 4_096, 16_384, 8_192),
        ("mm", 4_096, 16_384, 8_192),
        ("sfpu", 4_096, 16_384, 8_192),
    ):
        p_small = 1.0 / label_rates[f"{arm}-{small}"]
        p_big = 1.0 / label_rates[f"{arm}-{big}"]
        slope = (p_big - p_small) / (big - small)
        out[f"{arm}-{mid}"] = 1.0 / (p_small + slope * (mid - small))
    return out


def _synthetic_session(
    activity, rates, cycles=4, control="noc-4096", samples=40, terms=None
):
    """A clean synthetic board over an arbitrary activity set.

    The module-level ``measured_rows`` is tied to this file's own nine-workload
    schedule; this builds the same shape of session over whatever labels it is
    given, which is what the thirteen-workload case needs. The energies are
    :data:`TRUE`, invented, and no number from it is ever quoted as a result --
    the assertions it serves are about how many terms the fit reaches for.
    """
    # ``terms`` names the columns the truth is spent on, so an activity set of a
    # different vintage can be driven by the columns it actually carries.
    terms = DESIGN_TERMS if terms is None else terms
    energy_of = {
        t: TRUE.get(t, TRUE["matrix_arith_cycles"] if t.startswith("matrix") else 0.0)
        for t in terms
    }
    rows = []
    for cycle in range(cycles):
        rows.append(
            {
                "label": BASELINE_LABEL,
                "cycle": cycle,
                "slot": 0,
                "power_w": BASELINE_W,
                "samples": samples,
                "launches": 0,
                "wall_s": 30,
                "rate": 0.0,
                **_health(),
            }
        )
        for slot, row in enumerate(activity, start=1):
            label = row["label"]
            energy = TRUE["launch"] + sum(
                energy_of[t] * float(row.get(t, 0.0)) for t in terms
            )
            for suffix, extra_slot in (
                ("", slot),
                *((CONTROL_SUFFIX, len(activity) + 1),) * (label == control),
            ):
                rows.append(
                    {
                        "label": label + suffix,
                        "cycle": cycle,
                        "slot": extra_slot,
                        "power_w": BASELINE_W + rates[label] * energy,
                        "samples": samples,
                        "launches": rates[label] * 30,
                        "wall_s": 30.0,
                        "rate": rates[label],
                        **_health(),
                    }
                )
    return rows


def test_a_third_scale_adds_rows_but_no_directions_and_the_budget_knows_it():
    """The negative result, pinned so it does not get proposed again.

    Merging the intermediate-scale activity CSV into the card's gives thirteen
    workloads. Under the old ``n_workloads - 3`` budget that is **ten** terms out
    of eleven spread-carrying counters, and every one of the eleven possible
    ten-term subsets conditions past 3.2e16: an extra scale is a row that
    interpolates between rows already there, and rows are not activity
    directions. The number of directions is set by the number of arms, which the
    third scale does not change.

    Bounded by the design instead, the fit reaches for the same four terms it
    reached for at nine workloads and conditions at 693. More card time buys
    nothing here, and that is the point of this test."""
    activity = load_activity(CARD_ACTIVITY_CSV)
    have = {row["label"] for row in activity}
    activity += [
        row
        for row in load_activity(CARD_ACTIVITY_SCALE2_CSV)
        if row["label"] not in have
    ]
    assert len(activity) == 13
    assert sum(row["label"] == "idle-0" for row in activity) == 1

    rates = _rate_model({label: value[1] for label, value in real_means().items()})
    # The question here is how many activity DIRECTIONS thirteen rows carry, and
    # a row that interpolates between two others carries none.
    report = analyse(activity, _synthetic_session(activity, rates))

    assert len(report.labels) == 13
    assert report.ok, render(report)
    assert report.term_source == "designed"
    ident = gate(report, "identifiability")
    assert "6 coefficients (4 designed terms + launch + floor)" in ident.detail
    assert "condition number 693" in ident.detail
    # The degrees-of-freedom bound is still there and still the looser one: it
    # would have allowed ten. `min` of the two is what stops it.
    assert len(report.terms) < len(report.labels) - 3

    # The other direction: what ten terms would actually have cost. Every
    # ten-subset of the spread-carrying columns, on the same design.
    rate_col = np.array([rates[label] for label in report.labels])
    by_label = {row["label"]: row for row in activity}
    scaled = (
        np.array(
            [
                [float(by_label[label][t]) for t in ACTIVITY_TERMS]
                for label in report.labels
            ]
        )
        * rate_col[:, None]
    )
    base = np.column_stack([np.ones(len(report.labels)), rate_col])
    spread = [j for j in range(len(ACTIVITY_TERMS)) if np.ptp(scaled[:, j]) > 0]
    assert len(spread) == 12
    conds = [
        design_condition(np.column_stack([base] + [scaled[:, j] for j in combo]))
        for combo in itertools.combinations(spread, 10)
    ]
    assert len(conds) == 66
    assert min(conds) > 2.9e16


def test_the_real_session_within_arm_scalings_do_not_clear_the_noise_floor():
    """The honest half, and the claim the README has to stop making unqualified.

    The arms separate in the MEAN -- 2.924 W at 6.63 noise floors, ordered
    mm-16384 highest and idle-0 lowest. The within-arm 4x scalings, billed as
    "the sharpest ranking test in the set", do far less well: against the
    arm-only floor of 0.441 W, rv reaches 2.20 floors, mm 1.94 and noc 1.13,
    while sfpu (0.81) is INSIDE it -- and not one of the four reaches the 3
    floors that `spread` holds the set as a whole to. All four have the right
    sign."""
    report = analyse(activity_rows(), real_session_rows())
    floor = report.noise_floor_w
    means = real_means()
    pairs = {
        arm: means[f"{arm}-{big}"][0] - means[f"{arm}-{small}"][0]
        for arm, small, big in (
            ("rv", 200000, 800000),
            ("noc", 4096, 16384),
            ("mm", 4096, 16384),
            ("sfpu", 4096, 16384),
        )
    }
    assert all(delta > 0 for delta in pairs.values()), pairs
    ratios = {arm: delta / floor for arm, delta in pairs.items()}
    assert ratios["rv"] == pytest.approx(2.20, abs=0.02)
    assert ratios["mm"] == pytest.approx(1.94, abs=0.02)
    assert ratios["noc"] == pytest.approx(1.13, abs=0.02)
    assert ratios["sfpu"] == pytest.approx(0.81, abs=0.02)
    assert max(ratios.values()) < 3.0  # none reaches the bar `spread` holds the set to

    # ...while the set as a whole does clear it, which is why `spread` passes and
    # the ranking claim is about the set and not about the pairs.
    arms = [
        value[0]
        for label, value in means.items()
        if label not in (BASELINE_LABEL, "noc-4096" + CONTROL_SUFFIX)
    ]
    assert max(arms) - min(arms) == pytest.approx(2.924, abs=0.001)
    assert max(arms) - min(arms) > 3.0 * floor


# ---------------------------------------------------------------------------
# The pre-idle probe: a diagnostic, flagged rather than gated
# ---------------------------------------------------------------------------


def test_an_implausible_pre_idle_reading_is_flagged_in_the_report():
    """Cycle 1's slot 6 probe read 75.0 W -- above every arm in the session --
    because it caught the board still hot from the slot before, giving a negative
    delta_w; its slot 9 probe read 62.0 W against a 37.7 W idle baseline, which
    is below the arms but nowhere near idle. Both have to be flagged or the
    column misleads."""
    report = analyse(activity_rows(), real_session_rows())
    note = next(n for n in report.notes if "cannot be idle readings" in n)
    assert "2 pre-idle probe(s)" in note
    assert "cycle 1 slot 6" in note
    assert "above the session's arm floor" in note
    assert "delta_w = -6" in note
    assert "cycle 1 slot 9" in note
    assert "had not clocked down" in note
    assert note in render(report)


def test_the_pre_idle_probe_is_not_a_gate():
    """Deliberately: ``delta_w`` is a diagnostic column and no fit input reads
    it, so refusing a session over it would be theatre. The session with the bad
    probes must still be analysable, and no gate may carry its name."""
    report = analyse(activity_rows(), real_session_rows())
    assert not any("pre_idle" in g.name or "delta" in g.name for g in report.gates)
    # The session IS refused -- but by `repeats`, over the excluded drifted slot,
    # and not by anything that read the pre-idle column.
    assert [g.name for g in report.gates if g.refuses] == ["repeats"]


def test_plausible_pre_idle_readings_produce_no_note():
    """The other direction: the flag has to be about the reading, not about the
    column existing."""
    report = analyse(activity_rows(), measured_rows())
    assert "cannot be idle readings" not in " ".join(report.notes)


# ---------------------------------------------------------------------------
# The coefficient quarantine
# ---------------------------------------------------------------------------


def test_coefficients_may_not_be_written_into_the_cost_model_tree(tmp_path):
    with pytest.raises(ValueError, match="tt_sim/"):
        check_destination("tt_sim/perf/energy.yaml")
    with pytest.raises(ValueError, match="unit_costs"):
        check_destination(tmp_path / "unit_costs.yaml")
    # ...and a path outside it is fine.
    assert check_destination(tmp_path / "fitted_energy_coefficients.yaml")


def _write_session(tmp_path, **kwargs):
    """Spill the synthetic session to the two CSVs the CLI reads."""
    import csv as _csv

    from tt_sim.perf.energy_activity import CSV_COLUMNS, write_row

    activity = tmp_path / "activity.csv"
    for row in activity_rows():
        write_row(activity, {k: row.get(k, 0) for k in CSV_COLUMNS}, append=True)

    # The column names are `aggregate_power.py`'s, so this exercises the seam
    # between the card-box aggregator and `load_measured` rather than a private
    # shorthand: the health columns the new gates read have to survive the CSV.
    measured = tmp_path / "power.csv"
    rows = measured_rows(**kwargs)
    with open(measured, "w", newline="") as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "cycle",
                "slot",
                "status",
                "power_w",
                "samples",
                "attempts",
                "launches",
                "wall_s",
                "launches_per_s",
                "pre_idle_w",
                "sysfs_aiclk_mean",
                "sysfs_aiclk_drift_pct",
                "therm_trip_delta",
                "tt_smi_version",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "cycle": row["cycle"],
                    "slot": row["slot"],
                    "status": row["status"],
                    "power_w": row["power_w"],
                    "samples": row["samples"],
                    "attempts": row["attempts"],
                    "launches": row["launches"],
                    "wall_s": row["wall_s"],
                    "launches_per_s": row["rate"],
                    "pre_idle_w": row["pre_idle_w"],
                    "sysfs_aiclk_mean": row["aiclk_mean"],
                    "sysfs_aiclk_drift_pct": row["aiclk_drift_pct"],
                    "therm_trip_delta": row["therm_trip_delta"],
                    "tt_smi_version": row["tt_smi_version"],
                }
            )
    return activity, measured


def test_the_command_line_writes_a_report_and_quarantined_coefficients(tmp_path):
    from tt_sim.perf.energy_rank import main

    activity, measured = _write_session(tmp_path, noise=0.05)
    out = tmp_path / "coefficients.yaml"
    rc = main(
        [
            "--activity",
            str(activity),
            "--measured",
            str(measured),
            "--report",
            str(tmp_path / "report.txt"),
            "--json",
            str(tmp_path / "report.json"),
            "--write-coefficients",
            str(out),
        ]
    )
    assert rc == 0
    assert "Spearman (leave-one-out)" in (tmp_path / "report.txt").read_text()
    assert "provenance: fitted" in out.read_text()

    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["refused"] is False
    assert payload["spearman_loo"] == pytest.approx(1.0)


def test_the_command_line_refuses_and_writes_no_coefficients_when_a_gate_fires(
    tmp_path,
):
    """The other direction of the same path: a drifted session exits non-zero
    and leaves no coefficient file behind. A refusal that still writes the
    numbers is not a refusal."""
    from tt_sim.perf.energy_rank import main

    activity, measured = _write_session(tmp_path, noise=0.05, control_offset=6.0)
    out = tmp_path / "coefficients.yaml"
    rc = main(
        [
            "--activity",
            str(activity),
            "--measured",
            str(measured),
            "--write-coefficients",
            str(out),
        ]
    )
    assert rc == 1
    assert not out.exists()


def test_the_coefficient_document_stamps_fitted_and_warns_about_the_cost_tables():
    report = analyse(activity_rows(), measured_rows())
    doc = coefficients_document(report, {"measured_csv": "power.csv"})
    assert f"provenance: {FITTED_PROVENANCE}" in doc
    assert "NOT A COST TABLE" in doc
    assert "unit_costs.yaml" in doc
    assert "not_a_cost_table: true" in doc
    # The record has to say what it was fitted to, or it is not a record.
    assert "measured_csv" in doc
    assert "spearman_loo" in doc
