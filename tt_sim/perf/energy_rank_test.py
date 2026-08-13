"""Synthetic exercise of the ranking-energy analysis path.

There is no card data yet, so everything here is constructed. That is the point:
a gate is only worth anything if it has been shown to fire in **both**
directions, and synthetic data is the only way to build the failing case on
demand. Every gate in :mod:`tt_sim.perf.energy_rank` gets a passing session and
a failing one, and the failing one differs from the passing one in exactly the
thing the gate is about.

Nothing here touches the simulator: the module under test reads two CSVs and
writes a report.
"""

import json
import math

import numpy as np
import pytest

from tt_sim.perf.energy_activity import ACTIVITY_TERMS
from tt_sim.perf.energy_rank import (
    BASELINE_LABEL,
    CONTROL_SUFFIX,
    FITTED_PROVENANCE,
    analyse,
    check_destination,
    coefficients_document,
    design_condition,
    nnls,
    rankdata,
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
TRUE = {
    "launch": 2.0e-6,
    "instr_retired": 1.0e-9,
    "noc_bytes_total": 4.0e-11,
    "matrix_busy_cycles": 8.0e-10,
    "sfpu_busy_cycles": 5.0e-10,
}

BASELINE_W = 40.0


#: The synthetic workload set has the same shape as the real schedule: five arms
#: at two inner counts each, minus ``idle`` which has no inner loop, so nine
#: workloads. That count is not decoration -- the fit spends one coefficient per
#: term plus one for the launch machinery and leave-one-out needs a spare, so
#: nine workloads is what makes a four-term truth recoverable at all. Five would
#: not be, and a test built on five would be testing a design that cannot work.
#:
#: The per-arm shapes are the ones the real arms produced against tt-sim on
#: 2026-08-13, extended linearly in ``inner``; only the ratios matter here.
def _arm_activity(arm: str, inner: int) -> dict:
    base = {
        "instr_retired": 14_610,
        "noc_bytes_total": 0,
        "matrix_busy_cycles": 1,
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
        }
    if arm == "sfpu":
        return {
            **base,
            "instr_retired": 26_300,
            "noc_bytes_total": 12_288,
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
):
    """Build a card-session power CSV in memory.

    ``scale`` compresses every workload's *delta* from the baseline towards
    zero, which is how the ``spread`` gate's failing case is made: the same
    ordering, but inside the noise floor.
    """
    rng = np.random.default_rng(seed)
    labels = list(ACTIVITY) if labels is None else list(labels)
    rows = []
    for cycle in range(cycles):
        if include_baseline:
            rows.append(
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
            power = BASELINE_W + scale * RATES[label] * true_energy(label)
            rows.append(
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
            power = BASELINE_W + scale * RATES[control] * true_energy(control)
            rows.append(
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
    assert len(report.terms) + 1 <= len(report.labels) - 1


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

    measured = tmp_path / "power.csv"
    rows = measured_rows(**kwargs)
    with open(measured, "w", newline="") as fh:
        writer = _csv.DictWriter(
            fh,
            fieldnames=[
                "label",
                "cycle",
                "slot",
                "power_w",
                "samples",
                "launches",
                "wall_s",
                "launches_per_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "cycle": row["cycle"],
                    "slot": row["slot"],
                    "power_w": row["power_w"],
                    "samples": row["samples"],
                    "launches": row["launches"],
                    "wall_s": row["wall_s"],
                    "launches_per_s": row["rate"],
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
