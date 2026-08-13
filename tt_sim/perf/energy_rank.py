"""Fit energy coefficients to a card session and report **ranking quality**.

This is the home-side half of the ranking-level energy estimator (ROADMAP v2.0
item 5). It consumes two CSVs -- the activity vectors
(:mod:`tt_sim.perf.energy_activity`) and the measured board power from a card
session (``perfbench/energybench/run_card.sh``) -- and answers one question:

    **does the simulator put the workloads in the right order, and roughly by
    the right factor?**

It does NOT answer "how many joules". It cannot, and neither can any amount of
data from this instrument. ``tt-smi`` reports board power at ~1 Hz, including
DRAM, PHYs, ARC, PCIe and fans, so what is being fitted is a difference in
steady-state repeated-kernel board power against a measured idle baseline. That
is why the headline metric is a **Spearman correlation of predicted against
measured ordering** plus **ratio errors**, and emphatically not an R² on
absolute joules.

WHERE THE COEFFICIENTS LIVE, AND WHY NOT HERE
---------------------------------------------

Every coefficient this module produces is **FITTED**. Tenstorrent publishes no
per-event energy figure -- no pJ/op, no pJ/bit, nothing -- so there is no
document any of these numbers could ever be traced to.

``tt_sim/perf/unit_costs.yaml`` and ``tt_sim/pe/tensix/tensix_instruction_costs.yaml``
run a provenance ladder (``isa_doc > isa_doc_derived > vendor_source >
vendor_source_derived > estimated > unknown``) whose entire purpose is to keep
un-sourced numbers out of the cycle model; ``costs_test.py`` records that there
are currently **zero** ``estimated`` entries. A fitted energy coefficient is
weaker than ``estimated`` -- it is a regression coefficient from a ~1 Hz
board-level instrument against a nine-point design -- so it must never enter
those files, under any provenance.

They live in ``perfbench/energybench/fitted_energy_coefficients.yaml``, written
by ``--write-coefficients``, which:

* stamps ``provenance: fitted``, a token that is **not in**
  :data:`tt_sim.perf.costs.PROVENANCE_RANK`, so the cost loader raises
  ``KeyError`` on any file that carries it -- pasting one of these entries into
  a cost table breaks the loader rather than silently ranking the number;
* **refuses to write inside ``tt_sim/``** at all (see :func:`check_destination`);
* carries a ``not_a_cost_table`` banner and the full fitting record.

``tt_sim/perf/energy_quarantine_test.py`` asserts all three, in both directions.

THE GATES
---------

A fit is only meaningful if the measurement it is fitted to actually said
something. Seven gates run before any ranking is reported, and a failure is a
**refusal**, not a warning:

``repeats``
    Every label needs at least ``--min-repeats`` interleave cycles. One
    observation has no spread and so no noise floor.
``samples``
    Every measured row needs at least ``--min-samples`` telemetry samples. At
    ~1 Hz that is a floor on how long the arm ran.
``control``
    A designated workload is run **twice per interleave cycle** under two
    labels. The two must agree within the noise floor. This is the verified-zero
    control: if the same workload measured twice in the same session disagrees,
    the session drifted and every other difference in it is suspect.
``spread``
    The workloads' measured per-launch energies must span more than
    ``--sigma`` noise floors. Inside the noise floor there is nothing to rank,
    and a ranking reported anyway is a ranking of noise.
``identifiability``
    The number of fitted coefficients (activity terms plus the per-launch
    constant) must not exceed ``n_workloads - 1``, and the design matrix's
    condition number must be under ``--max-cond``. The first bound is what
    leaves a degree of freedom for leave-one-out; the second is what stops two
    collinear columns being reported as two independent findings.

Every one of those is proven to fire in **both** directions in
``tt_sim/perf/energy_rank_test.py`` -- a gate that cannot fail is as damaging as
one that cannot pass.

THE MODEL
---------

::

    E_launch(w)  =  c0  +  Σ_j c_j · a_j(w)                      [joules/launch]
    P_board(w)   =  P_baseline  +  rate(w) · E_launch(w)         [watts]

so the regression target is ``y(w) = (P(w) - P_baseline) / rate(w)``, per-launch
energy, and the design matrix is the activity vector with a leading column of
ones for the launch machinery. Coefficients are constrained non-negative
(Lawson-Hanson NNLS): a negative energy per instruction is not a finding, it is
a fit artefact, and allowing one lets the model buy accuracy with nonsense.

RANKING QUALITY
---------------

The headline number is **leave-one-out** Spearman: each workload is predicted by
a model refitted without it. In-sample Spearman is reported alongside, and the
gap between them is the honest measure of how much of the agreement is fitting
rather than predicting -- with nine workloads and up to seven coefficients an
in-sample fit will look excellent whatever the truth is.

Ratio errors are reported as ``|log(predicted ratio / measured ratio)|`` over
every workload pair, since a ranking claim is really a claim about ratios.

Usage
-----

::

    python3 -m tt_sim.perf.energy_rank --activity activity-sim.csv \\
        --measured card-session/power.csv --report report.txt
    python3 -m tt_sim.perf.energy_rank ... \\
        --write-coefficients perfbench/energybench/fitted_energy_coefficients.yaml
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tt_sim.perf.energy_activity import ACTIVITY_TERMS, load_activity

#: The label of the measured idle baseline: the board with the device open and
#: no kernel launching. It is P_static, not a workload, and is never fitted.
BASELINE_LABEL = "baseline"

#: A control row's label is the workload's with this suffix. Exactly one
#: workload must carry one.
CONTROL_SUFFIX = "__control"

#: The provenance token stamped on every fitted coefficient file. It is
#: deliberately absent from :data:`tt_sim.perf.costs.PROVENANCE_RANK` so that a
#: cost table carrying it fails to load rather than quietly ranking it.
FITTED_PROVENANCE = "fitted"

#: Refuse to write coefficients anywhere under this directory name.
QUARANTINE_FORBIDDEN_ROOT = "tt_sim"


# ---------------------------------------------------------------------------
# Small numeric helpers -- scipy is not a dependency of this repo
# ---------------------------------------------------------------------------


def rankdata(values) -> np.ndarray:
    """Ranks with ties averaged. ``scipy.stats.rankdata``'s default method."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    sorted_vals = arr[order]
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation. ``nan`` when either side is constant."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def nnls(A: np.ndarray, y: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Lawson-Hanson non-negative least squares.

    Kept in-tree rather than pulled from scipy because scipy is not a dependency
    and a fifty-line reference algorithm is cheaper than one. Returns ``x >= 0``
    minimising ``||Ax - y||``.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    n = A.shape[1]
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        w = A.T @ (y - A @ x)
        candidates = (~passive) & (w > 1e-12)
        if not candidates.any():
            break
        j = int(np.argmax(np.where(candidates, w, -np.inf)))
        passive[j] = True
        for _ in range(max_iter):
            idx = np.flatnonzero(passive)
            s = np.zeros(n)
            sol, *_ = np.linalg.lstsq(A[:, idx], y, rcond=None)
            s[idx] = sol
            if (s[idx] > 0).all():
                x = s
                break
            neg = idx[s[idx] <= 0]
            alpha = np.min(x[neg] / (x[neg] - s[neg]))
            x = x + alpha * (s - x)
            passive &= x > 1e-12
            if not passive.any():
                break
        else:  # pragma: no cover - the inner loop converges for well-posed A
            break
    return x


# ---------------------------------------------------------------------------
# Measured data
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"  [{'PASS' if self.passed else 'REFUSE'}] {self.name}: {self.detail}"


@dataclass
class RankReport:
    gates: list[GateResult] = field(default_factory=list)
    refused: bool = False
    labels: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    measured_energy: dict[str, float] = field(default_factory=dict)
    predicted_energy: dict[str, float] = field(default_factory=dict)
    loo_predicted: dict[str, float] = field(default_factory=dict)
    spearman_in_sample: float = float("nan")
    spearman_loo: float = float("nan")
    ratio_errors: dict[str, float] = field(default_factory=dict)
    noise_floor_w: float = 0.0
    baseline_w: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refused

    def to_dict(self) -> dict:
        return {
            "refused": self.refused,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
            "labels": self.labels,
            "terms": self.terms,
            "coefficients": self.coefficients,
            "measured_energy_j": self.measured_energy,
            "predicted_energy_j": self.predicted_energy,
            "loo_predicted_energy_j": self.loo_predicted,
            "spearman_in_sample": self.spearman_in_sample,
            "spearman_loo": self.spearman_loo,
            "ratio_errors": self.ratio_errors,
            "noise_floor_w": self.noise_floor_w,
            "baseline_w": self.baseline_w,
            "notes": self.notes,
        }


def load_measured(path: Path | str) -> list[dict]:
    """Read a card-session power CSV.

    Required columns: ``label``, ``cycle``, ``power_w``, ``samples``. Workload
    rows additionally need ``launches`` and ``wall_s``; the baseline does not
    (it launches nothing).
    """
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            if not raw.get("label"):
                continue
            row = {
                "label": raw["label"].strip(),
                "cycle": int(float(raw.get("cycle", 0) or 0)),
                "slot": int(float(raw.get("slot", 0) or 0)),
                "power_w": float(raw["power_w"]),
                "samples": int(float(raw.get("samples", 0) or 0)),
                "launches": float(raw.get("launches", 0) or 0),
                "wall_s": float(raw.get("wall_s", 0) or 0),
            }
            rate = float(raw.get("launches_per_s", 0) or 0)
            if rate <= 0 and row["wall_s"] > 0:
                rate = row["launches"] / row["wall_s"]
            row["rate"] = rate
            rows.append(row)
    return rows


def _by_label(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row["label"], []).append(row)
    return out


def noise_floor(grouped: dict[str, list[dict]]) -> float:
    """The session's own noise floor, in watts.

    Defined as the RMS of each label's across-cycle standard deviation. It is
    derived from this session's repeats rather than assumed, because a floor
    carried over from another session is exactly the drift the control exists to
    catch.
    """
    sds = []
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        sds.append(float(np.std([r["power_w"] for r in rows], ddof=1)))
    if not sds:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(sds)))))


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


#: A column that is an exact linear duplicate of the ones already chosen carries
#: no information at all, and dropping it is arithmetic rather than judgement.
#: This threshold is NOT the identifiability gate -- see :func:`select_terms`.
DEGENERATE_COND = 1e12


def design_condition(design: np.ndarray) -> float:
    """Condition number of a column-normalised design matrix.

    Normalisation first, because the condition number of a raw matrix mixing a
    column of ones, a byte count and a cycle count is dominated by the choice of
    units and says nothing about collinearity.
    """
    norms = np.linalg.norm(design, axis=0)
    norms[norms == 0] = 1.0
    return float(np.linalg.cond(design / norms))


def select_terms(matrix: np.ndarray, term_names: list[str], budget: int) -> list[int]:
    """Pick at most ``budget`` activity columns, greedily by relative spread.

    Two exclusions, and only two, both of them arithmetic:

    * a column with **no spread** across the workloads is a constant, which the
      launch column already covers;
    * a column that is an exact linear duplicate of those already chosen
      (condition number past :data:`DEGENERATE_COND`) adds a rank without adding
      a dimension.

    Everything else is selected. In particular a merely *ill-conditioned* set is
    selected and then **refused by the identifiability gate** rather than being
    quietly avoided here. That split is deliberate: a selector that filtered on
    the gate's own threshold would make the gate incapable of failing, which is
    as bad as one incapable of passing.
    """
    spreads = []
    for j, name in enumerate(term_names):
        col = matrix[:, j]
        rng = float(col.max() - col.min())
        if rng <= 0:
            continue
        # Scale-free ranking: a byte count and a cycle count are not comparable
        # in absolute spread.
        scale = float(np.abs(col).max()) or 1.0
        spreads.append((rng / scale, rng, j, name))
    spreads.sort(key=lambda t: (-t[0], t[3]))

    chosen: list[int] = []
    for _, _, j, _ in spreads:
        if len(chosen) >= budget:
            break
        trial = chosen + [j]
        design = np.column_stack(
            [np.ones(matrix.shape[0])] + [matrix[:, k] for k in trial]
        )
        if design_condition(design) > DEGENERATE_COND:
            continue
        chosen = trial
    return chosen


def _fit(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    return nnls(design, y)


def analyse(
    activity_rows: list[dict],
    measured_rows: list[dict],
    sigma: float = 3.0,
    min_repeats: int = 3,
    min_samples: int = 20,
    max_cond: float = 1e6,
    terms: list[str] | None = None,
) -> RankReport:
    report = RankReport()
    grouped = _by_label(measured_rows)

    # -- baseline ---------------------------------------------------------
    if BASELINE_LABEL not in grouped:
        report.refused = True
        report.gates.append(
            GateResult(
                "baseline",
                False,
                f"no {BASELINE_LABEL!r} rows: an idle baseline measured in the "
                "same session is what every delta is taken against",
            )
        )
        return report
    baseline_w = float(np.mean([r["power_w"] for r in grouped[BASELINE_LABEL]]))
    report.baseline_w = baseline_w
    report.gates.append(
        GateResult(
            "baseline",
            True,
            f"{baseline_w:.2f} W over {len(grouped[BASELINE_LABEL])} in-session repeats",
        )
    )

    floor = noise_floor(grouped)
    report.noise_floor_w = floor

    # -- G1 repeats -------------------------------------------------------
    thin = {
        label: len(rows) for label, rows in grouped.items() if len(rows) < min_repeats
    }
    report.gates.append(
        GateResult(
            "repeats",
            not thin,
            f"every label has >= {min_repeats} interleave cycles"
            if not thin
            else f"too few repeats: {thin} (need {min_repeats})",
        )
    )

    # -- G2 samples -------------------------------------------------------
    starved = sorted(
        {row["label"] for row in measured_rows if row["samples"] < min_samples}
    )
    report.gates.append(
        GateResult(
            "samples",
            not starved,
            f"every row has >= {min_samples} telemetry samples"
            if not starved
            else f"under-sampled rows in: {starved} (need {min_samples} each)",
        )
    )

    # -- G3 control (verified zero) ---------------------------------------
    controls = [label for label in grouped if label.endswith(CONTROL_SUFFIX)]
    if not controls:
        report.gates.append(
            GateResult(
                "control",
                False,
                f"no {CONTROL_SUFFIX!r} label: a session with no verified-zero "
                "control cannot show it did not drift",
            )
        )
    else:
        worst = 0.0
        detail = []
        ok = True
        for control in controls:
            base = control[: -len(CONTROL_SUFFIX)]
            if base not in grouped:
                ok = False
                detail.append(f"{control} has no matching {base!r} arm")
                continue
            delta = abs(
                float(np.mean([r["power_w"] for r in grouped[control]]))
                - float(np.mean([r["power_w"] for r in grouped[base]]))
            )
            worst = max(worst, delta)
            detail.append(f"{base}: |delta| = {delta:.3f} W")
        limit = sigma * floor
        ok = ok and worst <= limit
        report.gates.append(
            GateResult(
                "control",
                ok,
                "; ".join(detail) + f" against {sigma:g}*noise = {limit:.3f} W",
            )
        )

    # -- workloads --------------------------------------------------------
    activity_by_label = {row["label"]: row for row in activity_rows}
    labels = sorted(
        label
        for label in grouped
        if label != BASELINE_LABEL
        and not label.endswith(CONTROL_SUFFIX)
        and label in activity_by_label
    )
    missing = sorted(
        label
        for label in grouped
        if label != BASELINE_LABEL
        and not label.endswith(CONTROL_SUFFIX)
        and label not in activity_by_label
    )
    if missing:
        report.notes.append(
            f"measured but no activity vector, so not fitted: {', '.join(missing)}"
        )
    report.labels = labels

    energy = {}
    for label in labels:
        rows = grouped[label]
        rate = float(np.mean([r["rate"] for r in rows if r["rate"] > 0] or [0.0]))
        power = float(np.mean([r["power_w"] for r in rows]))
        if rate <= 0:
            report.notes.append(
                f"{label}: no launch rate, cannot convert W to J/launch"
            )
            continue
        energy[label] = (power - baseline_w) / rate
    report.measured_energy = energy

    # -- G4 spread --------------------------------------------------------
    # Only workloads that made it as far as a per-launch energy count towards the
    # spread: one whose launch rate never arrived is not a point on the axis the
    # gate is about.
    powers = [
        float(np.mean([r["power_w"] for r in grouped[label]]))
        for label in labels
        if label in energy
    ]
    spread = (max(powers) - min(powers)) if powers else 0.0
    limit = sigma * floor
    spread_ok = bool(powers) and spread > limit
    report.gates.append(
        GateResult(
            "spread",
            spread_ok,
            f"workload spread {spread:.3f} W against {sigma:g}*noise = {limit:.3f} W"
            + ("" if spread_ok else " -- inside the noise floor, nothing to rank"),
        )
    )

    # -- G5 identifiability -----------------------------------------------
    usable = [label for label in labels if label in energy]
    matrix = np.array(
        [
            [float(activity_by_label[label][t]) for t in ACTIVITY_TERMS]
            for label in usable
        ],
        dtype=float,
    ).reshape(len(usable), len(ACTIVITY_TERMS))
    if terms is not None:
        unknown = [t for t in terms if t not in ACTIVITY_TERMS]
        if unknown:
            raise ValueError(f"unknown activity terms: {unknown}")
        chosen = [ACTIVITY_TERMS.index(t) for t in terms]
    else:
        # One column for the launch constant, and one degree of freedom spared
        # so leave-one-out has something left to predict with.
        budget = max(0, len(usable) - 2)
        chosen = select_terms(matrix, list(ACTIVITY_TERMS), budget) if usable else []
    design_full = (
        np.column_stack([np.ones(len(usable))] + [matrix[:, j] for j in chosen])
        if usable
        else np.zeros((0, 1))
    )
    cond = design_condition(design_full) if usable else 0.0
    n_coeff = len(chosen) + 1
    ident_ok = bool(usable) and n_coeff <= max(1, len(usable) - 1) and cond <= max_cond
    report.gates.append(
        GateResult(
            "identifiability",
            ident_ok,
            f"{n_coeff} coefficients ({len(chosen)} terms + launch) against "
            f"{len(usable)} workloads, condition number {cond:.3g} "
            f"(cap {max_cond:.3g})",
        )
    )
    report.terms = [ACTIVITY_TERMS[j] for j in chosen]

    # -- G6 enough points to rank -----------------------------------------
    rank_ok = len(usable) >= 3
    report.gates.append(
        GateResult(
            "rankable",
            rank_ok,
            f"{len(usable)} workloads with both a vector and a rate"
            + ("" if rank_ok else " -- a rank correlation needs at least 3"),
        )
    )

    report.refused = any(not g.passed for g in report.gates)
    if report.refused:
        return report

    # -- fit ---------------------------------------------------------------
    y = np.array([energy[label] for label in usable], dtype=float)
    design = design_full
    coeff = _fit(design, y)
    names = ["launch"] + report.terms
    report.coefficients = {n: float(c) for n, c in zip(names, coeff)}
    pred = design @ coeff
    report.predicted_energy = {label: float(p) for label, p in zip(usable, pred)}

    # -- leave-one-out -----------------------------------------------------
    loo = {}
    for i, label in enumerate(usable):
        keep = [k for k in range(len(usable)) if k != i]
        c = _fit(design[keep], y[keep])
        loo[label] = float(design[i] @ c)
    report.loo_predicted = loo

    report.spearman_in_sample = spearman(y, pred)
    report.spearman_loo = spearman(y, [loo[label] for label in usable])

    # -- ratio errors ------------------------------------------------------
    errors = {}
    for i, a in enumerate(usable):
        for b in usable[i + 1 :]:
            ya, yb = y[i], y[usable.index(b)]
            pa, pb = loo[a], loo[b]
            if ya <= 0 or yb <= 0 or pa <= 0 or pb <= 0:
                continue
            errors[f"{a}/{b}"] = abs(math.log((pa / pb) / (ya / yb)))
    report.ratio_errors = errors
    return report


# ---------------------------------------------------------------------------
# Rendering and the coefficient quarantine
# ---------------------------------------------------------------------------


def render(report: RankReport) -> str:
    lines = ["# energybench ranking report", ""]
    lines.append(f"baseline (idle, in session): {report.baseline_w:.3f} W")
    lines.append(f"noise floor (RMS of per-label SDs): {report.noise_floor_w:.3f} W")
    lines.append("")
    lines.append("## Gates")
    lines.extend(g.line() for g in report.gates)
    lines.append("")
    if report.refused:
        lines.append("REFUSED: the measurement does not support a ranking claim.")
        lines.append(
            "No coefficients were fitted and none may be quoted from this session."
        )
        for note in report.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines) + "\n"

    lines.append(
        "## Fitted coefficients (J per unit of activity) -- FITTED, NOT SOURCED"
    )
    for name, value in report.coefficients.items():
        lines.append(f"  {name:<24} {value:.6g}")
    lines.append("")
    lines.append("## Ranking quality")
    lines.append(
        f"  Spearman (leave-one-out) : {report.spearman_loo:.4f}   <- the claim"
    )
    lines.append(
        f"  Spearman (in sample)     : {report.spearman_in_sample:.4f}   (fit, not prediction)"
    )
    if report.ratio_errors:
        vals = sorted(report.ratio_errors.values())
        median = vals[len(vals) // 2]
        lines.append(
            f"  |log ratio error| median : {median:.4f}  (~x{math.exp(median):.3f})"
        )
        lines.append(
            f"  |log ratio error| max    : {vals[-1]:.4f}  (~x{math.exp(vals[-1]):.3f})"
        )
    lines.append("")
    lines.append("## Per-workload energy (J/launch)")
    lines.append(f"  {'workload':<16} {'measured':>14} {'LOO predicted':>16}")
    for label in report.labels:
        if label not in report.measured_energy:
            continue
        lines.append(
            f"  {label:<16} {report.measured_energy[label]:>14.6g} "
            f"{report.loo_predicted.get(label, float('nan')):>16.6g}"
        )
    lines.append("")
    lines.append(
        "These coefficients are FITTED to board-level telemetry. They are not a "
        "cost-model provenance and must never be moved into tt_sim/perf/unit_costs.yaml."
    )
    for note in report.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines) + "\n"


def check_destination(path: Path | str) -> Path:
    """Refuse to write fitted coefficients into the cost-model tree.

    The structural half of the quarantine: a file cannot become a cost table by
    accident if it cannot be written next to one. ``provenance: fitted`` is the
    other half -- the loader rejects it -- but a rule enforced at write time is
    the one that stops the mistake being made at all.
    """
    path = Path(path)
    parts = set(path.resolve().parts)
    if QUARANTINE_FORBIDDEN_ROOT in parts:
        raise ValueError(
            f"refusing to write fitted energy coefficients to {path}: fitted "
            f"numbers must not live under {QUARANTINE_FORBIDDEN_ROOT}/, where the "
            "provenance-ranked cost tables are. See the module docstring."
        )
    if path.name in ("unit_costs.yaml", "tensix_instruction_costs.yaml"):
        raise ValueError(f"refusing to write fitted energy coefficients to {path.name}")
    return path


def coefficients_document(report: RankReport, sources: dict) -> str:
    """The YAML text of the quarantined coefficient file."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# FITTED ENERGY COEFFICIENTS -- NOT A COST TABLE.",
        "#",
        "# Every number below is a REGRESSION COEFFICIENT fitted to board-level",
        "# tt-smi telemetry. Tenstorrent publishes no per-event energy figure, so",
        "# none of these can ever be traced to a document. They are weaker than",
        "# the `estimated` provenance that tt_sim/perf/unit_costs.yaml forbids.",
        "#",
        "# DO NOT MOVE THESE INTO tt_sim/perf/unit_costs.yaml OR",
        "# tt_sim/pe/tensix/tensix_instruction_costs.yaml. The `provenance: fitted`",
        "# token below is not in tt_sim.perf.costs.PROVENANCE_RANK, so the cost",
        "# loader raises KeyError on any table that carries it -- that is",
        "# deliberate, and tt_sim/perf/energy_quarantine_test.py asserts it.",
        "#",
        "# What these predict is STEADY-STATE REPEATED-KERNEL BOARD POWER against a",
        "# measured in-session idle baseline, not the energy of a single launch,",
        "# and they are validated on ORDERING and RATIOS, never on absolute joules.",
        "",
        "not_a_cost_table: true",
        f"provenance: {FITTED_PROVENANCE}",
        "units: joules per unit of activity, per launch",
        f"fitted_on: {stamp}",
        "fitting_record:",
    ]
    for key, value in sources.items():
        lines.append(f"  {key}: {json.dumps(value)}")
    lines.append(f"  spearman_loo: {report.spearman_loo:.6f}")
    lines.append(f"  spearman_in_sample: {report.spearman_in_sample:.6f}")
    lines.append(f"  baseline_w: {report.baseline_w:.6f}")
    lines.append(f"  noise_floor_w: {report.noise_floor_w:.6f}")
    lines.append("  workloads:")
    for label in report.labels:
        lines.append(f"    - {label}")
    lines.append("  gates:")
    for gate in report.gates:
        lines.append(f"    {gate.name}: {json.dumps(gate.detail)}")
    lines.append("")
    lines.append("coefficients:")
    for name, value in report.coefficients.items():
        lines.append(f"  {name}: {value:.9g}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--activity", required=True, help="activity vector CSV")
    ap.add_argument("--measured", required=True, help="card session power CSV")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--min-repeats", type=int, default=3)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--max-cond", type=float, default=1e6)
    ap.add_argument(
        "--terms",
        help="comma-separated activity terms to fit, overriding the automatic "
        "selection. Naming more than the design supports is refused by the "
        "identifiability gate rather than fitted.",
    )
    ap.add_argument("--report", help="write the text report here as well as stdout")
    ap.add_argument("--json", help="write the machine-readable report here")
    ap.add_argument(
        "--write-coefficients",
        help="write the fitted coefficients here (refused inside tt_sim/)",
    )
    args = ap.parse_args(argv)

    activity = load_activity(args.activity)
    measured = load_measured(args.measured)
    report = analyse(
        activity,
        measured,
        sigma=args.sigma,
        min_repeats=args.min_repeats,
        min_samples=args.min_samples,
        max_cond=args.max_cond,
        terms=[t.strip() for t in args.terms.split(",") if t.strip()]
        if args.terms
        else None,
    )
    text = render(report)
    print(text, end="")
    if args.report:
        Path(args.report).write_text(text)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
    if args.write_coefficients:
        if report.refused:
            print(
                "not writing coefficients: the session was refused",
                file=sys.stderr,
            )
            return 1
        dest = check_destination(args.write_coefficients)
        dest.write_text(
            coefficients_document(
                report,
                {
                    "activity_csv": str(args.activity),
                    "measured_csv": str(args.measured),
                },
            )
        )
        print(f"fitted coefficients -> {dest}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
