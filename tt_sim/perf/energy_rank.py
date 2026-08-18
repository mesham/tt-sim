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

THE QUANTITY
------------

``power_w`` is **in-slot board power under sustained load** -- sampled while the
workload runs, which needs ``tt-smi`` >= 4.0.0 (the tt-umd backend; the older
Luwen backend panics on a device held by a tt-metal program). If a session was
run with ``run_card.sh --bracket`` it also carries ``power_bracket_w``, a
post-exit sample on a **decaying edge**. That is a different quantity and this
module never fits it.

WHAT A SESSION HAS TO CARRY
---------------------------

Two card sessions have been run, both on 2026-08-13, and both are why this module
looks the way it does.

The **first** recorded ``samples=0, power_w=0`` for every arm slot of a
three-cycle run and still looked like a finished measurement (``tt-smi`` 3.0.32
panicked on a device held by tt-metal and the sampler swallowed it). Three gates
refuse the shapes it had: a row with no telemetry (``telemetry``), a cycle
missing a slot (``schedule``), and a clock that moved under the measurement
(``clock``).

The **second**, after the tool upgrade, collected cleanly -- 33/33 slots ok, zero
sample failures, no thermal trips -- and is preserved at
``perfbench/card-sessions/2026-08-13-energybench/``. It is a sound measurement
that this module refuses, and it changed the arithmetic rather than only the
guards: see ``baseline_clock`` and ``target_triviality`` below, and THE MODEL.

Two more have been run since and **both fit**: a six-cycle Blackhole p150 at
``2026-08-13-energybench-2/`` and a six-cycle Wormhole n300 at
``2026-08-17-wh-energybench/``. The Wormhole box publishes no
``tt_therm_trip_count``, which is where ``thermal``'s temperature fallback comes
from, and it is the better-conditioned of the two on every axis that matters --
~107 samples a slot against ~31, a 15x launch-rate span against 5.3x, and a
design conditioning at 240 against 1.15e3. Read both against THE NULL MODEL.

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
something. Thirteen checks run before any ranking is reported. Twelve of them
are **refusals**, not warnings; the thirteenth (``baseline_clock``) is a
**reported finding** that changes what the arithmetic is allowed to do rather
than a refusal, because the thing it detects cannot be fixed by re-running:

``telemetry``
    Every row must carry at least one **successful** power sample and a finite
    power, and no row may carry a non-``ok`` status. A slot with zero samples is
    not a slot that measured zero watts, and this is what stops the two being
    confused: the aggregator writes an empty cell rather than ``0.0``, and an
    empty cell arrives here as ``nan`` and is refused.
``schedule``
    Every interleave cycle must carry every label. Cycle 2 of the 2026-08-13
    session was simply missing ``idle-0`` -- the CSV jumped slot 0 to slot 2 --
    because the runner recorded a manifest row only on success, so a failed slot
    left no trace anywhere machine-readable.
``thermal``
    ``tt_therm_trip_count`` must not move. A part that throttled was not running
    the workload the activity vector describes.
``clock``
    The session's **arm** slots must agree on their clock to within
    ``--max-clock-drift-pct``, and there must be a clock record at all. A slot
    whose clock moved *within itself* past the same cap is **excluded from the
    fit and named in the report** instead of refusing the session -- one
    contaminated slot must not discard 32 sound ones, and what decides whether
    the survivors are still a session is ``repeats``, counting surviving rows.
    The 2026-08-13 session has exactly one such row (cycle 1's ``rv-800000``,
    41.4% drift, 800-1350 MHz within the slot) and it takes that arm down to two
    repeats, so ``repeats`` refuses -- which tells the operator to run more
    cycles rather than merely that a clock moved. The across-session half
    deliberately excludes the baseline rows: see ``baseline_clock``.
``baseline_clock`` (a finding, not a refusal)
    The baseline slot launches nothing **by definition**, so a DVFS board drops
    to its idle clock state for it: the 2026-08-13 card session measured every
    baseline at 800 MHz and every arm at 1350 MHz. That is not drift and it is
    not fixable at the source -- ``tt-smi`` 6.2.0 offers no clock-pinning option
    -- so refusing the session on it would be a gate that can never pass. What it
    actually means is narrower and worth naming: **baseline subtraction is
    invalid for this session**, because the two rows are two DVFS states rather
    than one board with and without a workload. The fit therefore does not
    subtract it (see THE MODEL: ``P_floor`` is fitted from the arm rows), the
    baseline stays a session-completeness check and a recorded diagnostic, and
    this check reports the finding instead of refusing.
``repeats``
    Every label needs at least ``--min-repeats`` **surviving** interleave cycles.
    One observation has no spread and so no noise floor. Counting survivors is
    what keeps the ``clock`` exclusion honest: cutting a contaminated row out has
    to be able to cost the session, and this is where it does.
``samples``
    Every measured row needs at least ``--min-samples`` telemetry samples. At
    ~1 Hz that is a floor on how long the arm ran.
``control``
    A designated workload is run **twice per interleave cycle** under two
    labels. The two must agree within the noise floor. This is the verified-zero
    control: if the same workload measured twice in the same session disagrees,
    the session drifted and every other difference in it is suspect. The floor
    it is measured against is the **arm** rows' own repeatability and excludes
    the baseline (see :func:`noise_floor`) -- this gate passes when the
    disagreement is *below* the floor, so a floor inflated by an idle-state
    label's DVFS swing would make the session's drift detector permissive.
``spread``
    The workloads' measured per-launch energies must span more than
    ``--sigma`` noise floors. Inside the noise floor there is nothing to rank,
    and a ranking reported anyway is a ranking of noise.
``identifiability``
    The number of fitted coefficients (activity terms plus the per-launch
    constant plus the fitted floor) must not exceed ``n_workloads - 1``, and the
    design matrix's condition number must be under ``--max-cond``. The first
    bound is what leaves a degree of freedom for leave-one-out; the second is
    what stops two collinear columns being reported as two independent findings,
    and it is also what decides whether ``P_floor`` and ``c_launch`` are
    separable at all in a given session's spread of launch rates. What is
    *offered* to it is the design's own term set -- see THE TERM SET -- and that
    restriction is a statement about the experiment, never about this gate's
    verdict.
``target_triviality``
    The fit target must not be reproducible **without any activity term**. The
    same design with only the floor and the launch column is fitted to the
    target, and its R² must be under ``--max-triviality-r2``. A model that
    reaches the measured ordering knowing nothing but how often each workload
    launched is not evidence about activity, however good its Spearman looks.

Every one of those is proven to fire in **both** directions in
``tt_sim/perf/energy_rank_test.py`` -- a gate that cannot fail is as damaging as
one that cannot pass.

THE MODEL
---------

::

    E_launch(w)  =  c_launch  +  Σ_j c_j · a_j(w)                [joules/launch]
    P_board(w)   =  P_floor  +  rate(w) · E_launch(w)            [watts]

The regression target is **measured power itself**, ``y(w) = P(w)``, and the
design matrix is ``[1, rate(w), rate(w)·a_j(w)]`` -- a column of ones for
``P_floor``, a bare rate column for the launch machinery, and the activity terms
scaled by the rate they were paid at. ``P_floor`` is a **free, non-negative
fitted coefficient estimated from the arm rows alone**.

That is a change from the original parameterisation, which subtracted the
measured baseline and fitted ``y(w) = (P(w) - P_baseline)/rate(w)``. Two things
killed it, and the 2026-08-13 card session showed both:

* **The busy-state floor is not measurable on a DVFS board.** An idle board is
  not the busy board minus the kernel; it is in a different clock state (800 MHz
  against the arms' 1350 MHz, measured). There is no idle measurement of the busy
  state to subtract, so the floor has to be **extrapolated from the arms** --
  which is exactly what a free coefficient does.
* **Subtracting it made the target trivial.** With the baseline 30 W below every
  arm and the arms within 2.9 W of each other, ``(P - P_baseline)`` was ~91% a
  fixed DVFS step, and dividing that by a rate varying 5.3x left a target that is
  essentially the launch period. Regressed on ``1/rate`` alone, with no activity
  term at all, it scored **R² = 0.9995**. Fitting the floor removes the constant
  from the numerator instead of stamping it on every point; the
  ``target_triviality`` gate is what checks the result.

``rate(w)`` varies 5.3x across the arms, which is what makes ``P_floor`` and
``c_launch`` distinguishable *in principle*; whether they are distinguishable in
a given session is an empirical question about that session's design matrix, and
the ``identifiability`` gate is where it is answered. Nothing here special-cases
around it.

Coefficients are constrained non-negative (Lawson-Hanson NNLS): a negative energy
per instruction is not a finding, it is a fit artefact, and allowing one lets the
model buy accuracy with nonsense. A negative floor is likewise not a board.

THE TERM SET
------------

Which activity terms are fitted is decided by :data:`DESIGNED_ARM_TERMS` -- a
transcription of the arms table in ``perfbench/energybench/README.md``, one
non-idle arm to the one term that arm was **built** to move. It is a constant in
this file, and :func:`designed_terms` is handed **arm names and nothing else**:
no matrix, no condition number, no target, no fit. That is the point, and it is
structural rather than asserted. The term set is a property of the experiment,
fixed before any board was plugged in, and a reader in a year has to be able to
see that it could not have been chosen by looking at the data, because the
function that chooses it is not given any.

The alternative -- deriving each arm's term from whichever activity column that
arm's rows dominate -- was rejected. It would adapt automatically when the arms
change, which is its only advantage, and in exchange it would make the fitted
model a function of the collected matrix: a search over the data, differing from
the laundering this module exists to prevent only in which statistic it searched
on. A constant has to be edited by a person, in the same commit as the README's
arms table, which is exactly the friction wanted.

Two bounds apply, and the smaller wins::

    budget = min(len(designed terms present), n_workloads - 3)

The degrees-of-freedom bound (two coefficients spent before any activity term,
one spared for leave-one-out) still holds and is unchanged. What changed is that
it is **no longer the only bound**. It grows with the number of measured rows,
and rows are not directions: the 2026-08-13 session's five arms at two scales
give nine workloads and, at the old budget of six, the selector reached six deep
into eleven mutually-correlated counters and the design's condition number came
back at 2.34e7 against a 1e6 cap. Adding a third scale makes it worse, not
better -- thirteen workloads, a budget of ten, and **all eleven** possible
ten-term subsets land past 3.2e16 -- because an extra scale interpolates between
existing rows and adds no new activity direction. The same session fitted with
the four terms its arms were designed to separate conditions at 598 (713 with
the third scale added), and the 2026-08-17 Wormhole session -- same nine
workloads, a launch-rate span of 15x rather than 5.3x -- at 240.

Three rules keep this from becoming a back door:

* **A term outside the designed set is admitted only by ``--terms``**, which is a
  human writing the term set down before the run. The report and the coefficient
  file stamp such a fit ``operator-specified`` rather than ``designed``, so a
  reader can always tell which model was fitted. It is never admitted because it
  improved the conditioning, the fit or a gate's verdict -- an override still
  faces ``identifiability`` unchanged, and naming more terms than the design
  supports is refused there rather than fitted.
* **An arm the mapping does not know is a hard error.** :func:`designed_terms`
  raises ``ValueError`` naming the arm rather than silently fitting whatever
  terms it did recognise, because a model quietly smaller than the one reported
  is the failure this whole file is built around.
* **A designed term that cannot be fitted is named in the report.** If it has no
  spread across these workloads (a cost-model-off column of zeros), is an exact
  linear duplicate of another designed column, or falls outside the
  degrees-of-freedom bound, it is dropped and said so. When the DoF bound bites,
  the drop order is :func:`select_terms`' relative spread -- deliberately a
  criterion the ``identifiability`` gate does not use.

The gate stays able to fail, which was the requirement: on this session's rates
2 of the 330 four-term subsets still blow the cap, and a session that fails
``identifiability`` with the *designed* terms is reporting something real about
the measurement -- that the arms did not separate -- and must be refused.

RANKING QUALITY
---------------

The headline number is **leave-one-out** Spearman: each workload is predicted by
a model refitted without it. In-sample Spearman is reported alongside, and the
gap between them is the honest measure of how much of the agreement is fitting
rather than predicting -- with nine workloads and up to seven coefficients an
in-sample fit will look excellent whatever the truth is.

Ratio errors are reported as ``|log(predicted ratio / measured ratio)|`` over
every workload pair, since a ranking claim is really a claim about ratios. They
are computed from the **leave-one-out** predictions, not the in-sample ones, so
a ratio claim is out-of-sample in the same sense the Spearman is.

THE NULL MODEL
--------------

Every number in RANKING QUALITY is also reported for a model that has **no
energy content at all**: per-launch energy taken as proportional to the
simulator's own cycle count, with no coefficients, no fit and nothing from the
card. Both statistics are scale-free, so the null needs no constant and there is
no free parameter in it to tune.

It is here because the headline is easy to over-read. The fit target is board
power, but the *reported* ranking is per-launch energy -- ``(P - P_floor)/rate``
-- and across these arms the board power span is small against the floor while
the launch rate varies by more than an order of magnitude. So the measured
energy ordering is mostly an ordering by *how long the kernel took*, which the
cycle model already predicts and which needs no energy modelling whatever.

``target_triviality`` does not catch this and is not meant to: it asks whether
the **fit target** is reproducible from the floor and the launch rate, in power
space. The null asks whether the **reported ranking** is reproducible from the
cycle count, in energy space. The two are independent, and measured sessions
have passed the first comfortably (R² = 0.03) while the second matched the
fitted model's Spearman exactly.

It is reported rather than gated. A refusal needs a threshold, and there is no
principled one -- a model that ties the null on ordering may still be well ahead
on ratios, which is what has been observed. The honest treatment is to put both
in front of the reader every time, so a Spearman is never quoted without the
number it had to beat.

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
#: no kernel launching. It is a **session-completeness check and a recorded
#: diagnostic** -- the board's idle-state floor -- and is neither a workload nor
#: the fit's reference. See ``baseline_clock`` and THE MODEL: on a DVFS board the
#: idle state is not the busy state, so ``P_floor`` is fitted from the arms.
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

#: **The experiment's design, transcribed.** One entry per energybench arm, giving
#: the single activity term that arm was *constructed* to move -- the "what it
#: moves in the activity vector" column of the arms table in
#: ``perfbench/energybench/README.md``, which predates every measurement this
#: module has ever seen. ``idle`` maps to ``None``: it moves nothing by design,
#: which is what the launch column is for.
#:
#: This is a hand-written constant and not a derivation, on purpose. The number of
#: independent activity directions in a session is set by the number of distinct
#: **arms**, not by the number of rows, and the arms table is where that number
#: was decided -- a priori, on paper. Restricting the fit to these terms is
#: therefore honouring the design. Choosing terms by which set conditions best,
#: fits best or passes ``identifiability`` would be laundering, and the structural
#: difference is that :func:`designed_terms` is handed arm **names** and is never
#: given the design matrix, the target or any gate's threshold.
#:
#: Changing an arm means editing this table by hand, in the same commit as the
#: README's arms table. That friction is the feature.
DESIGNED_ARM_TERMS: dict[str, str | None] = {
    "idle": None,  # a kernel that returns immediately: the launch machinery alone
    "rv": "instr_retired",  # a dependent integer chain on BRISC
    "noc": "noc_bytes_total",  # barriered DRAM->L1 reads on BRISC
    # matmul_tiles on two RESIDENT tiles. **arith, not busy**: the Matrix Unit
    # also executes the RWC/dvalid bookkeeping every compute kernel pays, and
    # the `sfpu` arm pays 41 of those per iteration against zero matrix
    # arithmetic ops -- so `matrix_busy_cycles` is a column BOTH arms move and
    # is not the thing an arm was built to isolate. See the "Matrix arithmetic
    # is not matrix occupancy" section of tt_sim.perf.energy_activity.
    "mm": "matrix_arith_cycles",
    "sfpu": "sfpu_busy_cycles",  # Int32 tile add on the same two resident tiles
}


def arm_of(activity_row: dict) -> str:
    """Which arm an activity row belongs to.

    The ``arm`` column, written by :mod:`tt_sim.perf.energy_activity`. A row from
    an older CSV that lacks it falls back to the label prefix, since a label is
    ``<arm>-<inner>`` by the contract both scripts join on.
    """
    arm = str(activity_row.get("arm") or "").strip()
    if arm:
        return arm
    return str(activity_row.get("label", "")).split("-")[0]


def designed_terms(arms) -> list[str]:
    """The terms the arms that ran were **built** to separate, in schema order.

    Takes arm **names** -- metadata about which workloads were scheduled -- and
    nothing else. It is never handed the activity matrix, the measured power, a
    condition number or a gate threshold, and that is the whole argument for why
    restricting the fit to what it returns is not a search over the data. Every
    value it can return was written down in :data:`DESIGNED_ARM_TERMS` before any
    board was plugged in.

    An arm with no entry raises. It is tempting to skip it and fit the terms that
    *were* recognised, and that is precisely the failure this module is built
    against: the fit would then be of a smaller model than the report describes,
    silently. A new arm is a change to the design, so it needs a line in the
    table and a line in the README's arms table, by hand.
    """
    seen = list(dict.fromkeys(arms))
    unknown = [a for a in seen if a not in DESIGNED_ARM_TERMS]
    if unknown:
        raise ValueError(
            f"activity CSV carries arm(s) {unknown} that are not in "
            "DESIGNED_ARM_TERMS, so there is no a-priori statement of which term "
            "they were built to move. Add them to that table (and to the arms "
            "table in perfbench/energybench/README.md), or name the term set "
            "explicitly with --terms. Refusing to fit a model smaller than the "
            f"one being reported. Known arms: {sorted(DESIGNED_ARM_TERMS)}"
        )
    wanted = {DESIGNED_ARM_TERMS[a] for a in seen} - {None}
    off_schema = sorted(t for t in wanted if t not in ACTIVITY_TERMS)
    if off_schema:
        raise ValueError(
            f"DESIGNED_ARM_TERMS names term(s) {off_schema} that are not in "
            "tt_sim.perf.energy_activity.ACTIVITY_TERMS -- the design and the "
            "activity schema have drifted apart"
        )
    # Schema order, so the reported column order never depends on which arm was
    # listed first anywhere.
    return [t for t in ACTIVITY_TERMS if t in wanted]


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
    """One check's verdict.

    ``advisory`` marks a check whose failure is a **named finding rather than a
    refusal**: it is reported, it is recorded in the coefficient file, and it
    changes what the arithmetic is allowed to do, but it does not throw the
    session away. Exactly one check is advisory (``baseline_clock``) and the
    reason is written out at its definition -- refusing on a confound that no
    setting of the instrument can remove would be a gate that can never pass,
    which this module treats as being as bad as one that can never fail.
    """

    name: str
    passed: bool
    detail: str
    advisory: bool = False

    @property
    def refuses(self) -> bool:
        return not self.passed and not self.advisory

    def line(self) -> str:
        if self.passed:
            tag = "PASS"
        else:
            tag = "FINDING" if self.advisory else "REFUSE"
        return f"  [{tag}] {self.name}: {self.detail}"


@dataclass
class RankReport:
    gates: list[GateResult] = field(default_factory=list)
    refused: bool = False
    labels: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    #: Where :attr:`terms` came from: ``"designed"`` when it is the arms table's
    #: own term set (:data:`DESIGNED_ARM_TERMS`, fixed a priori), or
    #: ``"operator-specified"`` when a human pre-registered it with ``--terms``.
    #: It is reported, and stamped into the coefficient file, because which model
    #: was fitted is not something a later reader should have to reconstruct.
    term_source: str = "designed"
    coefficients: dict[str, float] = field(default_factory=dict)
    measured_energy: dict[str, float] = field(default_factory=dict)
    predicted_energy: dict[str, float] = field(default_factory=dict)
    loo_predicted: dict[str, float] = field(default_factory=dict)
    spearman_in_sample: float = float("nan")
    spearman_loo: float = float("nan")
    ratio_errors: dict[str, float] = field(default_factory=dict)
    #: The **null model**: per-launch energy taken as proportional to the
    #: simulator's own cycle count, with no energy fit, no coefficients and no
    #: card data behind it. See THE NULL MODEL. ``target_triviality`` already
    #: asks whether the *fit target* is reproducible without activity; this asks
    #: the harder question about the *reported ranking*, and the two do not
    #: answer each other -- a session has passed the first at R² = 0.03 while
    #: the second matched the fit exactly.
    spearman_null: float = float("nan")
    null_ratio_errors: dict[str, float] = field(default_factory=dict)
    #: The RMS of each label's across-cycle SD over the **arm** rows that
    #: entered the fit. The baseline is not in it: see :func:`noise_floor`.
    noise_floor_w: float = 0.0
    #: The board's idle-state floor as *measured* by the baseline slot. A
    #: diagnostic: on a DVFS board it is a different clock state from the arms
    #: and nothing is differenced against it.
    baseline_w: float = 0.0
    #: The busy-state floor as *fitted* from the arm rows -- the model's
    #: reference, and the number the per-launch energies are taken against.
    p_floor_w: float = float("nan")
    #: R² of the target against the floor-plus-launch model alone, i.e. how much
    #: of the ranking is reproducible without knowing any activity at all.
    target_triviality_r2: float = float("nan")
    #: Rows cut out of the fit, one line each, saying which row and why. A row
    #: that is not fitted must never be silently absent.
    excluded_rows: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refused

    def to_dict(self) -> dict:
        return {
            "quantity": QUANTITY,
            "refused": self.refused,
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "advisory": g.advisory,
                    "detail": g.detail,
                }
                for g in self.gates
            ],
            "model": MODEL_FORM,
            "labels": self.labels,
            "terms": self.terms,
            "term_source": self.term_source,
            "coefficients": self.coefficients,
            "measured_energy_j": self.measured_energy,
            "predicted_energy_j": self.predicted_energy,
            "loo_predicted_energy_j": self.loo_predicted,
            "spearman_in_sample": self.spearman_in_sample,
            "spearman_loo": self.spearman_loo,
            "spearman_null": self.spearman_null,
            "ratio_errors": self.ratio_errors,
            "null_ratio_errors": self.null_ratio_errors,
            "noise_floor_w": self.noise_floor_w,
            "baseline_w": self.baseline_w,
            "p_floor_w": self.p_floor_w,
            "target_triviality_r2": self.target_triviality_r2,
            "excluded_rows": self.excluded_rows,
            "notes": self.notes,
        }


def _optional_float(raw: dict, key: str) -> float | None:
    """A numeric column that may legitimately be absent or **empty**.

    Empty is not zero. ``aggregate_power.py`` writes an empty cell for a reading
    that was never taken, precisely so that it cannot be read as a measurement
    of zero, and this is the other end of that contract: an empty cell comes back
    as ``None`` and a gate refuses it.
    """
    value = raw.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_measured(path: Path | str) -> list[dict]:
    """Read a card-session power CSV.

    Required columns: ``label``, ``cycle``, ``power_w``, ``samples``. Workload
    rows additionally need ``launches`` and ``wall_s``; the baseline does not
    (it launches nothing). A session written by the current ``aggregate_power.py``
    also carries ``status``, the sysfs clock record and ``therm_trip_delta``,
    which the ``telemetry``, ``thermal`` and ``clock`` gates require.

    ``power_w`` is parsed leniently on purpose: an **empty** cell becomes ``nan``
    rather than raising, so the session reaches the gates and is refused there
    with a row-by-row explanation, instead of dying with a ``ValueError`` that
    says nothing about which slot failed.
    """
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            if not raw.get("label"):
                continue
            power = _optional_float(raw, "power_w")
            row = {
                "label": raw["label"].strip(),
                "cycle": int(float(raw.get("cycle", 0) or 0)),
                "slot": int(float(raw.get("slot", 0) or 0)),
                "power_w": float("nan") if power is None else power,
                "samples": int(float(raw.get("samples", 0) or 0)),
                "attempts": int(float(raw.get("attempts", 0) or 0)),
                "status": (raw.get("status") or "").strip(),
                "launches": float(raw.get("launches", 0) or 0),
                "wall_s": float(raw.get("wall_s", 0) or 0),
                "aiclk_mean": _optional_float(raw, "sysfs_aiclk_mean"),
                "aiclk_min": _optional_float(raw, "sysfs_aiclk_min"),
                "aiclk_max": _optional_float(raw, "sysfs_aiclk_max"),
                "aiclk_drift_pct": _optional_float(raw, "sysfs_aiclk_drift_pct"),
                "therm_trip_delta": _optional_float(raw, "therm_trip_delta"),
                "temp_c": _optional_float(raw, "temp_c"),
                "pre_idle_w": _optional_float(raw, "pre_idle_w"),
                "tt_smi_version": (raw.get("tt_smi_version") or "").strip(),
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
    """The session's own noise floor, in watts, over the **arm rows only**.

    Defined as the RMS of each label's across-cycle standard deviation, taken
    over the rows that actually enter the fit: the caller passes rows that have
    already survived the within-slot clock exclusion, and the baseline label is
    dropped here.

    It is derived from **this session's** repeats rather than assumed, because a
    floor carried over from another session is exactly the drift the control
    exists to catch. And it is derived from this session's **arm** repeats
    specifically, because an idle-state label's variance is not the noise of the
    busy-state measurements it would be used to judge. The baseline slot
    launches nothing by definition, so on a DVFS board it sits in the idle clock
    state (see ``_baseline_clock_gate``) and its across-cycle swing measures how
    that state wandered, not how repeatable the arms are. The 2026-08-13 session
    puts a size on the mistake: its three baselines read 34.97, 39.28 and
    38.97 W -- an SD of 2.40 W against 0.09-1.15 W for every arm -- and letting
    that one label in inflates the floor from 0.441 W to 0.838 W, a factor of
    1.9. The gates are then measured against the wrong yardstick, and
    ``control`` in particular passes when the twin disagreement is *below*
    ``sigma * floor``, so an inflated floor makes the session's own drift
    detector roughly twice as permissive as intended -- in the same session
    whose ``baseline_clock`` finding says baseline subtraction is invalid.

    The **control** label is counted, deliberately. It is a genuine arm
    measurement in the busy clock state and its across-cycle spread is as
    legitimate a repeat as any other label's. The circularity worry -- that the
    ``control`` gate would then be judged partly against its own variance --
    does not survive being looked at: that gate compares the control's *mean*
    against its twin's, while the floor is built from within-label across-cycle
    spread, and any session-level wander that inflated the control's spread
    inflated its twin's identically. The twin is in the floor whatever is done
    with the control, so dropping the control alone breaks no loop; it only
    discards one repeat out of ten. Nor is keeping it self-serving: on the
    2026-08-13 session including the control *lowers* the floor (0.441 W against
    0.463 W without it) and so makes its own gate stricter.
    """
    sds = []
    for label, rows in grouped.items():
        if label == BASELINE_LABEL or len(rows) < 2:
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


def select_terms(
    matrix: np.ndarray,
    term_names: list[str],
    budget: int,
    base: np.ndarray | None = None,
) -> list[int]:
    """Pick at most ``budget`` activity columns, greedily by relative spread.

    ``base`` is the part of the design that is there whatever is selected -- the
    floor and launch columns. It matters: a column that duplicates *the rate*
    is as degenerate as one that duplicates another activity term, and a
    duplicate check run against a bare column of ones would not see it.

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

    Note what this function is *given*, because that is where the restriction to
    the design lives. :func:`analyse` offers it only the columns in
    :func:`designed_terms` -- fixed a priori by the arms table, and computed from
    arm names alone. That narrowing happens **outside** here and for a reason
    that has nothing to do with conditioning. Inside, nothing has changed: no
    ``max_cond``, no target, no gate verdict is visible to this code, and the
    relative-spread ordering it ranks by is deliberately not the criterion the
    gate applies. It only decides which designed terms go first when the
    degrees-of-freedom bound cannot afford all of them.
    """
    if base is None:
        base = np.ones((matrix.shape[0], 1))
    base = np.asarray(base, dtype=float).reshape(matrix.shape[0], -1)
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
        design = np.column_stack([base] + [matrix[:, k] for k in trial])
        if design_condition(design) > DEGENERATE_COND:
            continue
        chosen = trial
    return chosen


def _fit(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NNLS on a column-normalised design, rescaled back.

    The design now mixes a column of ones, a launch rate in the hundreds and
    activity-times-rate products in the millions, and an unscaled least squares
    over that loses digits for no reason. Scaling columns by a positive number
    leaves the non-negative solution unchanged, so this is arithmetic hygiene and
    not a modelling choice -- the conditioning *question* is still asked, and
    answered by the ``identifiability`` gate on the same normalised basis.
    """
    design = np.asarray(design, dtype=float)
    norms = np.linalg.norm(design, axis=0)
    norms[norms == 0] = 1.0
    return nnls(design / norms, y) / norms


def reduced_model_r2(design: np.ndarray, y: np.ndarray) -> float:
    """R² of ``y`` against ``design``, by **unconstrained** least squares.

    Used by the ``target_triviality`` gate with the activity columns stripped
    off, so what comes back is "how much of the target the floor and the launch
    rate alone already explain". ``nan`` when ``y`` has no variance to explain.

    Deliberately OLS and not the model's NNLS. The question here is how much of
    the target a launch rate can reproduce, not how much it can reproduce while
    obeying a sign constraint that the trivial model has no need of: clamping a
    reduced fit at the non-negativity boundary would report R² = 0 for a target
    a rate explains perfectly with the opposite sign. Unconstrained is the
    strictly more conservative measurement, and a gate that errs should err
    towards refusing.
    """
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    if len(y) < 2:
        return float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coeff
    return float(1.0 - float(np.sum(resid**2)) / ss_tot)


def _describe(row: dict) -> str:
    return f"cycle {row.get('cycle')} slot {row.get('slot')} {row.get('label')}"


def _telemetry_gate(rows: list[dict]) -> GateResult:
    """Every row must actually have been measured.

    This is the gate the 2026-08-13 session needed and did not have. Every arm
    slot in it came back with zero samples and a power of ``0.0``, and nothing
    anywhere distinguished that from a board drawing no power -- the session ran
    to completion, wrote a full CSV and was analysed as if it meant something.

    Three separate things are refused here, because the failure had three faces:
    a status the runner already knew was bad, a sample count of zero, and a power
    cell that is empty (``nan``) rather than a number.
    """
    problems = []
    for row in rows:
        status = row.get("status", "")
        if status and status != "ok":
            problems.append(f"{_describe(row)}: status={status}")
        elif row.get("samples", 0) <= 0:
            problems.append(f"{_describe(row)}: 0 telemetry samples")
        elif not math.isfinite(row.get("power_w", float("nan"))):
            problems.append(f"{_describe(row)}: no power reading")
    if problems:
        return GateResult(
            "telemetry",
            False,
            f"{len(problems)} row(s) were never measured: "
            + "; ".join(problems[:6])
            + ("; ..." if len(problems) > 6 else "")
            + ". A slot with no samples did not measure zero watts",
        )
    return GateResult(
        "telemetry",
        True,
        f"all {len(rows)} rows carry a status of ok, at least one telemetry "
        "sample and a finite power",
    )


def _schedule_gate(rows: list[dict]) -> GateResult:
    """Every interleave cycle must carry every label.

    Cycle 2 of the 2026-08-13 session was missing ``idle-0`` outright and the CSV
    simply skipped from slot 0 to slot 2, because the runner wrote a manifest row
    only for a slot that succeeded. A hole was therefore indistinguishable from a
    schedule that never had that slot, and no gate looked.
    """
    by_cycle: dict[int, set] = {}
    for row in rows:
        by_cycle.setdefault(row.get("cycle", 0), set()).add(row["label"])
    if not by_cycle:
        return GateResult("schedule", False, "no rows at all")
    every = set().union(*by_cycle.values())
    holes = [
        f"cycle {cycle} is missing {label}"
        for cycle in sorted(by_cycle)
        for label in sorted(every - by_cycle[cycle])
    ]
    return GateResult(
        "schedule",
        not holes,
        f"all {len(by_cycle)} cycles carry all {len(every)} labels"
        if not holes
        else "; ".join(holes) + " -- a cycle with a hole in it is not an interleave",
    )


#: The ceiling and the drift a temperature record has to stay inside to stand in
#: for a trip counter. Both parts idle in the high 30s and these sessions run
#: 36-43 C, so 70 C is far above anything observed and far below where either
#: part throttles; 15 C is roughly twice the 7.0 C a clean 54-minute Wormhole
#: session drifted. Neither is a published throttle point -- they are "nothing
#: like a thermal event", which is all a fallback can honestly assert.
TEMP_FALLBACK_CEILING_C = 70.0
TEMP_FALLBACK_DRIFT_C = 15.0


def _thermal_gate(rows: list[dict]) -> GateResult:
    """``tt_therm_trip_count`` must not move anywhere in the session.

    A part that throttled was not running the workload the activity vector
    describes, and no amount of averaging recovers that. Read from sysfs, which
    costs nothing and needs no device handle.

    **The trip counter is preferred and is not always published.** The Wormhole
    box used on 2026-08-17 exposes only ``aiclk``/``arcclk``/``axiclk``, so
    ``therm_trip_delta`` is empty on every row of an otherwise clean session --
    66/66 ok, no sample failures, arm separation at ten times the noise floor.
    Refusing that is refusing a driver difference, not a measurement.

    So a session with no trip record falls back to ``temp_c``, which tt-smi
    reports either way, under its own falsifiable criterion: every slot below
    :data:`TEMP_FALLBACK_CEILING_C` and a session-wide spread under
    :data:`TEMP_FALLBACK_DRIFT_C`. **This is weaker evidence and the report says
    so**: a trip counter records that the part *did not* throttle, whereas a
    temperature merely fails to show it happening. It is not licence to skip the
    check -- a session that heats away is still refused, and a session with
    neither record is still refused, because "no thermal evidence at all" is the
    thing this gate was written for.
    """
    missing = [r for r in rows if r.get("therm_trip_delta") is None]
    if missing:
        temps = [r.get("temp_c") for r in rows]
        if any(t is None for t in temps):
            return GateResult(
                "thermal",
                False,
                f"{len(missing)} row(s) have no tt_therm_trip_count record and "
                f"{sum(t is None for t in temps)} have no temperature either "
                f"({_describe(missing[0])}...): a session that cannot show the "
                "part did not throttle cannot be differenced",
            )
        hot = [
            f"{_describe(r)}: {r['temp_c']:.1f} C"
            for r in rows
            if r["temp_c"] > TEMP_FALLBACK_CEILING_C
        ]
        lo, hi = min(temps), max(temps)
        drift = hi - lo
        if hot:
            return GateResult(
                "thermal",
                False,
                "no trip counter, and the temperature fallback refuses: "
                + "; ".join(hot[:4])
                + f" is above the {TEMP_FALLBACK_CEILING_C:g} C ceiling",
            )
        if drift > TEMP_FALLBACK_DRIFT_C:
            return GateResult(
                "thermal",
                False,
                f"no trip counter, and the temperature fallback refuses: the "
                f"session drifted {drift:.1f} C ({lo:.1f}-{hi:.1f}) against a "
                f"{TEMP_FALLBACK_DRIFT_C:g} C limit, so a thermal event cannot "
                "be ruled out",
            )
        return GateResult(
            "thermal",
            True,
            f"NO trip counter on this box, so this is the WEAKER temperature "
            f"fallback: {lo:.1f}-{hi:.1f} C, drift {drift:.1f} C, all under the "
            f"{TEMP_FALLBACK_CEILING_C:g} C ceiling. A trip count would record "
            "that the part did not throttle; a temperature only fails to show "
            "it, so read this as 'nothing like a thermal event' and not as "
            "'the part did not throttle'",
        )
    tripped = [
        f"{_describe(r)}: +{r['therm_trip_delta']:.0f}"
        for r in rows
        if r["therm_trip_delta"] != 0
    ]
    return GateResult(
        "thermal",
        not tripped,
        "tt_therm_trip_count held still across every slot"
        if not tripped
        else "the part throttled: " + "; ".join(tripped[:6]),
    )


def _spread_pct(clocks: list[float]) -> float:
    mean_clock = float(np.mean(clocks)) if clocks else 0.0
    return 100.0 * (max(clocks) - min(clocks)) / mean_clock if mean_clock else 0.0


def within_slot_drifted(rows: list[dict], max_drift_pct: float) -> list[dict]:
    """The rows whose AI clock moved **during** the slot, past the cap.

    Such a row's mean is a mean over two DVFS states and is not a measurement of
    either. The 2026-08-13 session has exactly one: cycle 1's ``rv-800000``
    straddled a transition -- ``sysfs_aiclk_min`` 800, ``sysfs_aiclk_max`` 1350,
    41.4% drift -- and it shows in every other column too (``power_sd_w`` 5.14
    against 0.1-0.9 everywhere else, a 75.0 W pre-idle probe, ``delta_w`` -6.81).
    Every other row in that session drifted negligibly.

    It is **excluded from the fit and named**, not made grounds for refusing the
    session: one contaminated slot must not discard 32 sound ones. What decides
    whether the survivors are still a session is the ``repeats`` gate, counting
    surviving rows only -- which is the refusal an operator can act on ("run more
    cycles") rather than one that just says the clock moved.
    """
    return [r for r in rows if (r.get("aiclk_drift_pct") or 0.0) > max_drift_pct]


def _clock_gate(
    rows: list[dict], kept: list[dict], excluded: list[dict], max_drift_pct: float
) -> GateResult:
    """The AI clock must hold still within a slot, and the **arms** must agree.

    The 2026-08-13 session read a baseline of 61.7 W at 1350 MHz in one cycle and
    ~39 W at 800 MHz in the next, and differenced them: a 42% swing in the
    reference, driven by a clock nothing was recording. Two checks, because there
    are two ways for it to bite -- a clock that moved *during* a slot makes that
    slot's mean meaningless, and slots at *different* clocks cannot be compared
    against each other however steady each one was.

    The across-session half runs over the **arm rows only**, and only over those
    that survived the within-slot exclusion. The baseline is left out not to be
    kind to it but because including it made this gate unsatisfiable: a slot that
    launches nothing sits in the board's idle DVFS state by definition, so
    baseline-versus-arm is always two clock states and never drift. That
    comparison is real and is made -- by ``_baseline_clock_gate``, which reports
    it as a finding and changes the arithmetic.

    What is left here can still fail, and that is the point: a session with no
    clock record at all, or one whose arm slots each held a *different* steady
    clock, is refused. Only the within-slot case became an exclusion, because
    there the damage is confined to one row and can be cut out.
    """
    missing = [_describe(r) for r in rows if r.get("aiclk_mean") is None]
    if missing:
        return GateResult(
            "clock",
            False,
            f"{len(missing)} row(s) have no clock record ({missing[0]}...): a "
            "session that cannot show its clock held still cannot be differenced",
        )
    arms = [r["aiclk_mean"] for r in kept if r["label"] != BASELINE_LABEL]
    if not arms:  # a session of nothing but baselines; the baseline gate owns it
        arms = [r["aiclk_mean"] for r in kept] or [r["aiclk_mean"] for r in rows]
    across = _spread_pct(arms)
    dropped = (
        "; ".join(
            f"{_describe(r)}: {r['aiclk_drift_pct']:.2f}% "
            f"({r.get('aiclk_min', float('nan')):.0f}-"
            f"{r.get('aiclk_max', float('nan')):.0f} MHz)"
            if r.get("aiclk_min") is not None
            else f"{_describe(r)}: {r['aiclk_drift_pct']:.2f}%"
            for r in excluded[:6]
        )
        if excluded
        else ""
    )
    detail = (
        (
            f"EXCLUDED {len(excluded)} row(s) whose clock moved inside the slot "
            f"past {max_drift_pct:g}% -- {dropped} -- their mean is a mean over "
            "two DVFS states; the survivors carry on and `repeats` decides "
            "whether enough is left. "
        )
        if excluded
        else f"within-slot drift <= {max_drift_pct:g}% on all {len(rows)} rows; "
    ) + (
        f"across-session spread over the {len(arms)} surviving arm rows "
        f"{across:.2f}% ({min(arms):.1f}-{max(arms):.1f} MHz)"
    )
    ok = across <= max_drift_pct
    if not ok:
        detail = (
            f"arm slots ran at different clocks: {min(arms):.1f}-{max(arms):.1f} MHz "
            f"is a {across:.2f}% spread against a {max_drift_pct:g}% cap, so these "
            "rows are not comparable"
        )
    return GateResult("clock", ok, detail)


def _baseline_clock_gate(rows: list[dict], max_drift_pct: float) -> GateResult:
    """Did the baseline run at the arms' clock? A **finding**, not a refusal.

    The baseline slot launches nothing by definition, so on a DVFS board it is
    measured in the idle clock state while every arm is in the busy one. The
    2026-08-13 Blackhole p150 session is the case in point: 800 MHz for every
    baseline, 1350 MHz for every arm, a 42% gap that is not drift and cannot be
    closed -- ``tt-smi`` 6.2.0 has no clock-pinning option (checked: only
    ``-l/-v/-s/-ls/-f/-c/--offline/-r/--snapshot_no_tty/-glx_*/--no_reinit/
    --use_luwen/--eth_train_skip``), so no way of running the session removes it.

    Refusing on it would therefore be a check that can never pass, which this
    module treats as being exactly as damaging as one that can never fail. So it
    reports, and what it reports is specific: **baseline subtraction is invalid
    for this session**. That is wired to the arithmetic rather than to a refusal
    -- ``P_floor`` is fitted from the arm rows (see THE MODEL) and the baseline is
    kept as a session-completeness check and a recorded diagnostic.
    """
    base = [
        r["aiclk_mean"]
        for r in rows
        if r["label"] == BASELINE_LABEL and r.get("aiclk_mean") is not None
    ]
    arms = [
        r["aiclk_mean"]
        for r in rows
        if r["label"] != BASELINE_LABEL and r.get("aiclk_mean") is not None
    ]
    if not base or not arms:
        return GateResult(
            "baseline_clock",
            False,
            "no clock record on the baseline or on the arms, so it cannot be "
            "shown that the baseline is even in the same DVFS state as the arms",
            advisory=True,
        )
    b, a = float(np.mean(base)), float(np.mean(arms))
    gap = _spread_pct([b, a])
    if gap <= max_drift_pct:
        return GateResult(
            "baseline_clock",
            True,
            f"baseline {b:.0f} MHz against arms {a:.0f} MHz ({gap:.2f}% apart, "
            f"cap {max_drift_pct:g}%): the same DVFS state, so the measured "
            "baseline is at least a like-for-like floor",
            advisory=True,
        )
    return GateResult(
        "baseline_clock",
        False,
        f"baseline {b:.0f} MHz against arms {a:.0f} MHz ({gap:.2f}% apart, cap "
        f"{max_drift_pct:g}%) -- two DVFS states, not drift: BASELINE SUBTRACTION "
        "IS INVALID for this session. Not a refusal and not fixable at the source "
        "(tt-smi offers no clock pinning); the fit does not subtract the baseline, "
        "it fits P_floor from the arm rows, and the baseline stays a "
        "session-completeness check and a recorded diagnostic",
        advisory=True,
    )


def _pre_idle_notes(
    rows: list[dict], baseline_w: float, arm_floor_w: float
) -> list[str]:
    """Flag pre-idle readings that cannot be what they claim to be.

    ``pre_idle_w`` is taken in the gap before a slot, with the device free, and
    there is **no guarantee the board has clocked down or cooled** by then. The
    2026-08-13 session shows what that looks like: cycle 1's slot 6 probe read
    75.0 W -- above every arm in the session -- for a ``delta_w`` of -6.8, and
    its slot 9 probe read 62.0 W against a measured idle baseline of 37.7 W.
    Neither is an idle board; both caught one still coming down from the slot
    before.

    Three shapes are named, in the language of what went wrong:

    * a probe **above every arm** cannot be an idle reading of anything;
    * a **negative** ``delta_w`` on a slot that actually launched says the slot
      drew less under load than its own "idle" reference, which is a statement
      about the reference. (Not applied to the baseline slot, whose ``delta_w``
      is idle minus idle and is negative as often as not: the 2026-08-13
      baselines sit at -0.03, -0.53 and -1.22 W and none of those is a fault.)
    * a probe **nearer the busy state than the measured idle floor** has not
      clocked down, even when it stays below the arms -- cycle 1's slot 9 read
      62.0 W against a 37.7 W idle baseline and a 66.0 W arm floor.

    ``delta_w`` is a diagnostic column and never a fit input, so this is
    deliberately **not** a gate: refusing a session over a column nothing is
    computed from would be theatre. But a reader looking down that column has to
    be told which cells are not idle readings, or the column misleads.
    """
    notes = []
    suspicious = []
    midpoint = (
        0.5 * (baseline_w + arm_floor_w)
        if math.isfinite(baseline_w) and math.isfinite(arm_floor_w)
        else float("inf")
    )
    for row in rows:
        pre = row.get("pre_idle_w")
        if pre is None:
            continue
        power = row.get("power_w", float("nan"))
        why = []
        if math.isfinite(arm_floor_w) and pre > arm_floor_w:
            why.append(
                f"{pre:.1f} W is above the session's arm floor {arm_floor_w:.1f} W"
            )
        elif pre > midpoint:
            why.append(
                f"{pre:.1f} W is nearer the busy state than the measured idle "
                f"floor {baseline_w:.1f} W, so the board had not clocked down"
            )
        if math.isfinite(power) and power - pre < 0 and row.get("rate", 0.0) > 0:
            why.append(f"delta_w = {power - pre:.2f} W is negative")
        if why:
            suspicious.append(f"{_describe(row)}: " + " and ".join(why))
    if suspicious:
        notes.append(
            f"{len(suspicious)} pre-idle probe(s) cannot be idle readings -- the "
            "probe has no way to make the board clock down or cool first, so "
            "delta_w is unusable on those rows (it is a diagnostic, not a fit "
            "input, and nothing here is computed from it): "
            + "; ".join(suspicious[:6])
            + ("; ..." if len(suspicious) > 6 else "")
        )
    return notes


def analyse(
    activity_rows: list[dict],
    measured_rows: list[dict],
    sigma: float = 3.0,
    min_repeats: int = 3,
    min_samples: int = 20,
    max_cond: float = 1e6,
    max_clock_drift_pct: float = 5.0,
    max_triviality_r2: float = 0.95,
    terms: list[str] | None = None,
) -> RankReport:
    report = RankReport()

    # -- G0 telemetry -----------------------------------------------------
    # First, and an early return: every other gate averages these numbers, and
    # averaging a row that was never measured is the failure being guarded
    # against rather than a degraded version of it.
    telemetry = _telemetry_gate(measured_rows)
    report.gates.append(telemetry)
    if not telemetry.passed:
        report.refused = True
        return report

    # -- row exclusion ----------------------------------------------------
    # A row whose clock moved *during* the slot is cut out here and nowhere else;
    # everything downstream that averages, differences or fits sees only the
    # survivors. `schedule`, `thermal`, `telemetry` and `baseline_clock` still
    # see every row, because those are statements about the session as it ran
    # rather than about the numbers being fitted.
    excluded = within_slot_drifted(measured_rows, max_clock_drift_pct)
    dropped_ids = {id(r) for r in excluded}
    kept = [r for r in measured_rows if id(r) not in dropped_ids]
    report.excluded_rows = [
        f"{_describe(r)}: AI clock drifted {r['aiclk_drift_pct']:.2f}% within the "
        f"slot (cap {max_clock_drift_pct:g}%), so its mean is a mean over two "
        "clock states -- excluded from the fit, not from the record"
        for r in excluded
    ]
    grouped = _by_label(kept)

    # -- baseline ---------------------------------------------------------
    # The baseline is no longer the fit's reference -- see THE MODEL and
    # ``_baseline_clock_gate``: on a DVFS board an idle slot is a different clock
    # state, not the busy state minus the kernel. It is still required, for two
    # reasons that are worth keeping apart from each other. A session without one
    # did not run the schedule it claims to have run (completeness), and the
    # measured idle floor is the diagnostic that shows how far the *fitted*
    # busy-state floor sits above it. Neither of those survives deleting the gate.
    if BASELINE_LABEL not in grouped:
        report.refused = True
        report.gates.append(
            GateResult(
                "baseline",
                False,
                f"no usable {BASELINE_LABEL!r} rows: a session that never measured its "
                "own idle floor did not run this schedule, and has no diagnostic "
                "to put the fitted busy-state floor against",
            )
        )
        return report
    baseline_w = float(np.mean([r["power_w"] for r in grouped[BASELINE_LABEL]]))
    report.baseline_w = baseline_w
    report.gates.append(
        GateResult(
            "baseline",
            True,
            f"idle floor {baseline_w:.2f} W over "
            f"{len(grouped[BASELINE_LABEL])} in-session repeats -- a completeness "
            "check and a recorded diagnostic, NOT the fit reference (P_floor is "
            "fitted from the arm rows)",
        )
    )

    # The scale the `control` and `spread` gates are measured against, from the
    # rows that enter the fit: `grouped` is already post-exclusion, and
    # `noise_floor` drops the baseline label itself -- an idle-state repeat is
    # not the noise of the busy-state rows it would be judging.
    floor = noise_floor(grouped)
    report.noise_floor_w = floor

    # -- G1 schedule, thermal, clock: was this a session at all ------------
    report.gates.append(_schedule_gate(measured_rows))
    report.gates.append(_thermal_gate(measured_rows))
    report.gates.append(_clock_gate(measured_rows, kept, excluded, max_clock_drift_pct))
    report.gates.append(_baseline_clock_gate(measured_rows, max_clock_drift_pct))

    # -- G2 repeats -------------------------------------------------------
    # Counted over SURVIVING rows, which is what makes exclusion honest: cutting
    # a contaminated row out has to be able to cost the session, and this is
    # where it does. A label whose every row was excluded still appears here,
    # with a count of zero, rather than vanishing into "no such label".
    scheduled = {row["label"] for row in measured_rows}
    thin = {
        label: len(grouped.get(label, []))
        for label in sorted(scheduled)
        if len(grouped.get(label, [])) < min_repeats
    }
    report.gates.append(
        GateResult(
            "repeats",
            not thin,
            f"every label has >= {min_repeats} usable interleave cycles"
            if not thin
            else f"too few usable repeats: {thin} (need {min_repeats})"
            + (
                " -- after the within-slot clock exclusions above; more "
                "interleave cycles is the fix"
                if excluded
                else ""
            ),
        )
    )

    # -- G3 samples -------------------------------------------------------
    starved = sorted({row["label"] for row in kept if row["samples"] < min_samples})
    report.gates.append(
        GateResult(
            "samples",
            not starved,
            f"every row has >= {min_samples} telemetry samples"
            if not starved
            else f"under-sampled rows in: {starved} (need {min_samples} each)",
        )
    )

    # -- G4 control (verified zero) ---------------------------------------
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

    # A workload is usable when it has a vector, a rate and a power: the fit is
    # in POWER space now (see THE MODEL), so the rate is a design column rather
    # than a divisor, but a workload with no rate still has no row to contribute.
    usable, rates, powers_by_label = [], {}, {}
    for label in labels:
        rows = grouped[label]
        rate = float(np.mean([r["rate"] for r in rows if r["rate"] > 0] or [0.0]))
        if rate <= 0:
            report.notes.append(
                f"{label}: no launch rate, so it has no place in a per-launch model"
            )
            continue
        usable.append(label)
        rates[label] = rate
        powers_by_label[label] = float(np.mean([r["power_w"] for r in rows]))

    # -- G5 spread --------------------------------------------------------
    powers = [powers_by_label[label] for label in usable]
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
    # Over EVERY row, not just the fitted ones: this is a note about the record a
    # reader is looking at, and an excluded row's columns are still in the CSV.
    report.notes.extend(
        _pre_idle_notes(
            measured_rows, baseline_w, min(powers) if powers else float("nan")
        )
    )

    # -- the design -------------------------------------------------------
    # [1, rate, rate*a_j]: a column of ones for the fitted busy-state floor, a
    # bare rate column for the launch machinery, and each activity term scaled by
    # the rate it was paid at.
    matrix = np.array(
        [
            [float(activity_by_label[label][t]) for t in ACTIVITY_TERMS]
            for label in usable
        ],
        dtype=float,
    ).reshape(len(usable), len(ACTIVITY_TERMS))
    rate_col = np.array([rates[label] for label in usable], dtype=float)
    base = np.column_stack([np.ones(len(usable)), rate_col])
    scaled = matrix * rate_col[:, None]
    if terms is not None:
        # The pre-registered override: a human named the term set, before the
        # run, and the report says so. It is not vetted against the design --
        # that is what makes it an override -- but it is stamped, and it still
        # faces `identifiability` unchanged.
        unknown = [t for t in terms if t not in ACTIVITY_TERMS]
        if unknown:
            raise ValueError(f"unknown activity terms: {unknown}")
        chosen = [ACTIVITY_TERMS.index(t) for t in terms]
        report.term_source = "operator-specified"
    elif usable:
        # The term set comes from THE DESIGN (see THE TERM SET): one term per
        # non-idle arm, fixed by the arms table before any measurement existed,
        # and computed here from arm NAMES only -- `designed_terms` is never
        # shown the matrix, the target or a condition number.
        design_set = designed_terms(
            arm_of(activity_by_label[label]) for label in usable
        )
        design_cols = [ACTIVITY_TERMS.index(t) for t in design_set]
        # Both bounds apply and the smaller wins. The degrees-of-freedom rule is
        # unchanged -- two coefficients before any activity term, one spared for
        # leave-one-out -- it is simply no longer the only bound. Rows are not
        # directions: more workloads at more scales buy neither.
        budget = min(len(design_set), max(0, len(usable) - 3))
        picked = select_terms(scaled[:, design_cols], design_set, budget, base=base)
        chosen = [design_cols[k] for k in picked]
        report.term_source = "designed"
        dropped = [t for t in design_set if ACTIVITY_TERMS.index(t) not in chosen]
        if dropped:
            report.notes.append(
                f"designed term(s) not in the fit: {', '.join(dropped)} -- either "
                "no spread across these workloads (an occupancy term needs "
                "TT_SIM_COST_MODEL=1), an exact linear duplicate of another "
                f"designed column, or beyond what {len(usable)} workloads can "
                f"afford ({budget} term(s)). The model reported is the model "
                "fitted; this is the arm whose direction it no longer separates"
            )
    else:
        chosen = []
    design_full = (
        np.column_stack([base] + [scaled[:, j] for j in chosen])
        if usable
        else np.zeros((0, 2))
    )
    report.terms = [ACTIVITY_TERMS[j] for j in chosen]
    y = np.array([powers_by_label[label] for label in usable], dtype=float)

    # -- G6 identifiability -----------------------------------------------
    # This is also where "are P_floor and c_launch separable in this session?"
    # gets answered. They are separable in principle when the launch rates
    # spread; whether they are separable HERE is a fact about this design matrix,
    # and it is checked rather than assumed or worked around.
    cond = design_condition(design_full) if usable else 0.0
    n_coeff = len(chosen) + 2
    ident_ok = bool(usable) and n_coeff <= max(1, len(usable) - 1) and cond <= max_cond
    report.gates.append(
        GateResult(
            "identifiability",
            ident_ok,
            f"{n_coeff} coefficients ({len(chosen)} {report.term_source} terms + "
            f"launch + floor) against {len(usable)} workloads, condition number "
            f"{cond:.3g} (cap {max_cond:.3g})"
            + (
                " -- the term set is the arms table's, fixed before the data "
                "existed, so this gate is judging the design and not a search "
                "over it"
                if report.term_source == "designed"
                else ""
            ),
        )
    )

    # -- G7 target triviality ---------------------------------------------
    # The 2026-08-13 session's OLD target -- (P - P_baseline)/rate, with a 30 W
    # DVFS step in the numerator and 2.9 W of actual workload difference --
    # scored R^2 = 0.9995 against 1/rate with no activity term at all. It passed
    # `spread`, it passed `control`, and `identifiability` looks at the design
    # matrix rather than the target, so nothing in the set could see it. A fit
    # would have reported an excellent leave-one-out Spearman for a ranking that
    # was 99.95% "whichever workload launches slowest costs most". Hence this.
    triviality = reduced_model_r2(design_full[:, :2], y) if usable else float("nan")
    report.target_triviality_r2 = triviality
    trivial_ok = bool(usable) and not (triviality > max_triviality_r2)
    report.gates.append(
        GateResult(
            "target_triviality",
            trivial_ok,
            f"the floor-and-launch-only model explains R^2 = {triviality:.4f} of "
            f"the target (cap {max_triviality_r2:g})"
            + (
                ""
                if trivial_ok
                else " -- the ranking is reproducible without knowing any "
                "activity at all, so this session is evidence about launch rates "
                "and not about activity"
            ),
        )
    )

    # -- G8 enough points to rank -----------------------------------------
    rank_ok = len(usable) >= 3
    report.gates.append(
        GateResult(
            "rankable",
            rank_ok,
            f"{len(usable)} workloads with both a vector and a rate"
            + ("" if rank_ok else " -- a rank correlation needs at least 3"),
        )
    )

    report.refused = any(g.refuses for g in report.gates)
    if report.refused:
        return report

    # -- fit ---------------------------------------------------------------
    coeff = _fit(design_full, y)
    p_floor = float(coeff[0])
    report.p_floor_w = p_floor
    # The reported coefficients are the ENERGY ones, in joules per unit of
    # activity per launch; P_floor is watts and is reported separately so the two
    # units cannot be read off the same list.
    names = ["launch"] + report.terms
    report.coefficients = {n: float(c) for n, c in zip(names, coeff[1:])}

    # Per-launch energies, recomputed consistently from the new parameterisation:
    # measured is what the board drew above the FITTED floor, per launch, and
    # predicted is the model's energy sum. The floor is a single session-level
    # constant on both sides, so the axis does not move under the point being
    # predicted.
    energy = {
        label: (powers_by_label[label] - p_floor) / rates[label] for label in usable
    }
    report.measured_energy = energy
    y_energy = np.array([energy[label] for label in usable], dtype=float)
    pred = (design_full @ coeff - p_floor) / rate_col
    report.predicted_energy = {label: float(p) for label, p in zip(usable, pred)}
    if any(v <= 0 for v in energy.values()):
        report.notes.append(
            "at least one workload measured below the fitted floor, so its "
            "per-launch energy came out non-positive and it is excluded from the "
            "ratio errors"
        )

    # -- leave-one-out -----------------------------------------------------
    # Each held-out workload is predicted by a model refitted without it --
    # including that model's own floor, because the floor is part of what the
    # reduced session claims to know.
    loo = {}
    for i, label in enumerate(usable):
        keep = [k for k in range(len(usable)) if k != i]
        c = _fit(design_full[keep], y[keep])
        loo[label] = float((design_full[i] @ c - c[0]) / rate_col[i])
    report.loo_predicted = loo

    report.spearman_in_sample = spearman(y_energy, pred)
    report.spearman_loo = spearman(y_energy, [loo[label] for label in usable])

    # -- ratio errors ------------------------------------------------------
    errors = {}
    for i, a in enumerate(usable):
        for b in usable[i + 1 :]:
            ya, yb = y_energy[i], y_energy[usable.index(b)]
            pa, pb = loo[a], loo[b]
            if ya <= 0 or yb <= 0 or pa <= 0 or pb <= 0:
                continue
            errors[f"{a}/{b}"] = abs(math.log((pa / pb) / (ya / yb)))
    report.ratio_errors = errors

    # -- the null model ----------------------------------------------------
    # Energy proportional to the simulator's cycle count: no coefficients, no
    # fit, nothing from the card. It needs no scale factor because both the
    # Spearman and the ratio errors are scale-free, which is the point -- there
    # is no free parameter here to make it look good or bad. Whatever the fitted
    # model reports has to be read against this, or a headline number is being
    # credited to an energy model that a cycle count already had.
    null = {
        label: float(activity_by_label[label].get("sim_cycles") or 0.0)
        / max(1.0, float(activity_by_label[label].get("launches") or 1))
        for label in usable
    }
    if all(v > 0 for v in null.values()):
        report.spearman_null = spearman(y_energy, [null[label] for label in usable])
        null_errors = {}
        for i, a in enumerate(usable):
            for b in usable[i + 1 :]:
                ya, yb = y_energy[i], y_energy[usable.index(b)]
                if ya <= 0 or yb <= 0:
                    continue
                null_errors[f"{a}/{b}"] = abs(math.log((null[a] / null[b]) / (ya / yb)))
        report.null_ratio_errors = null_errors
    else:
        # `sim_cycles` is a KEY column of the activity schema, so a real CSV
        # always carries it; a hand-made one may not, and `load_activity` reads
        # an absent column back as 0. Say so rather than reporting a null model
        # computed from zeros, which would look like a model the fit had beaten.
        report.notes.append(
            "no null model: the activity vectors carry no non-zero sim_cycles, so "
            "there is nothing to read the ranking numbers against"
        )
    return report


# ---------------------------------------------------------------------------
# Rendering and the coefficient quarantine
# ---------------------------------------------------------------------------


#: What ``power_w`` is, in one line, printed on every report and stamped into
#: every coefficient file. It is here rather than inline so the report, the JSON
#: and the YAML cannot drift apart on the one thing a reader must not get wrong.
QUANTITY = (
    "MEASURED QUANTITY: in-slot board power under SUSTAINED LOAD -- sampled while "
    "the kernel is launching, at ~1 Hz, board-wide (DRAM, PHYs, ARC, PCIe, fans "
    "included). Not the energy of one launch, and not a post-exit decaying edge."
)

#: The fitted model, in one line, for the same reason :data:`QUANTITY` is here:
#: the report, the JSON and the YAML must not be able to disagree about what was
#: fitted. The floor is FITTED from the arm rows, not subtracted from a measured
#: idle slot -- see THE MODEL.
MODEL_FORM = (
    "FITTED MODEL: P_board(w) = P_floor + rate(w) * (c_launch + sum_j c_j*a_j(w)). "
    "P_floor is a free non-negative coefficient estimated from the ARM rows: the "
    "busy-state floor is not measurable on a DVFS board, because an idle board is "
    "not in the busy state, so it is extrapolated rather than subtracted."
)


def render(report: RankReport) -> str:
    lines = ["# energybench ranking report", "", QUANTITY, "", MODEL_FORM, ""]
    lines.append(
        f"measured idle baseline (diagnostic, NOT subtracted): "
        f"{report.baseline_w:.3f} W"
    )
    if math.isfinite(report.p_floor_w):
        lines.append(f"fitted busy-state floor P_floor: {report.p_floor_w:.3f} W")
    lines.append(
        f"noise floor (RMS of per-label across-cycle SDs, ARM rows only): "
        f"{report.noise_floor_w:.3f} W"
    )
    if report.terms:
        origin = (
            "fixed a priori by the energybench arms table -- one term per "
            "non-idle arm, chosen before any measurement existed"
            if report.term_source == "designed"
            else "pre-registered by the operator with --terms"
        )
        lines.append(
            f"fitted terms ({report.term_source}): {', '.join(report.terms)}\n"
            f"  {origin}"
        )
    lines.append("")
    lines.append("## Gates")
    lines.extend(g.line() for g in report.gates)
    lines.append("")
    if report.excluded_rows:
        lines.append("## Rows excluded from the fit (still part of the record)")
        lines.extend(f"  - {line}" for line in report.excluded_rows)
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
    lines.append(f"  {'P_floor (W, not J)':<24} {report.p_floor_w:.6g}")
    lines.append("")
    lines.append("## Ranking quality")
    lines.append(
        f"  Spearman (leave-one-out) : {report.spearman_loo:.4f}   <- the claim"
    )
    lines.append(
        f"  Spearman (in sample)     : {report.spearman_in_sample:.4f}   (fit, not prediction)"
    )
    lines.append(
        f"  target triviality R^2    : {report.target_triviality_r2:.4f}   "
        "(floor + launch rate alone, no activity)"
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
    if not math.isnan(report.spearman_null):
        lines.append("## The null model: energy proportional to simulated cycles")
        lines.append(
            "  No coefficients, no fit, nothing from the card. Read every number "
            "above against these:"
        )
        lines.append(
            f"  Spearman (null)          : {report.spearman_null:.4f}   "
            "<- what the leave-one-out claim has to beat"
        )
        if report.null_ratio_errors:
            nvals = sorted(report.null_ratio_errors.values())
            nmedian = nvals[len(nvals) // 2]
            lines.append(
                f"  |log ratio error| median : {nmedian:.4f}  (~x{math.exp(nmedian):.3f})"
            )
            lines.append(
                f"  |log ratio error| max    : {nvals[-1]:.4f}  "
                f"(~x{math.exp(nvals[-1]):.3f})"
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
        "# What these predict is STEADY-STATE REPEATED-KERNEL BOARD POWER UNDER",
        "# SUSTAINED LOAD, sampled in slot -- not the energy of a single launch,",
        "# and not a post-exit decaying edge. They are validated on ORDERING and",
        "# RATIOS, never on absolute joules.",
        "#",
        "# The board's busy-state floor `p_floor_w` below is FITTED from the arm",
        "# rows and is in WATTS, not joules. It is not the measured idle baseline",
        "# and the two are not interchangeable: on a DVFS board the idle slot is a",
        "# different clock state, which is why nothing here subtracts it.",
        "#",
        "# `term_source` below says where the fitted term set came from.",
        "# `designed` means it is the energybench arms table's own -- one term per",
        "# non-idle arm, fixed before any board was plugged in. `operator-specified`",
        "# means a human named it with --terms. Neither is ever chosen by which set",
        "# conditions best or passes a gate.",
        "",
        "not_a_cost_table: true",
        f"quantity: {json.dumps(QUANTITY)}",
        f"model: {json.dumps(MODEL_FORM)}",
        f"provenance: {FITTED_PROVENANCE}",
        "units: joules per unit of activity, per launch (p_floor_w is watts)",
        f"fitted_on: {stamp}",
        "fitting_record:",
    ]
    for key, value in sources.items():
        lines.append(f"  {key}: {json.dumps(value)}")
    lines.append(f"  spearman_loo: {report.spearman_loo:.6f}")
    lines.append(f"  spearman_in_sample: {report.spearman_in_sample:.6f}")
    lines.append(
        f"  spearman_null: {report.spearman_null:.6f}  "
        "# cycles-proportional, no fit; what spearman_loo has to beat"
    )
    lines.append(f"  target_triviality_r2: {report.target_triviality_r2:.6f}")
    lines.append(f"  p_floor_w: {report.p_floor_w:.6f}")
    lines.append(f"  baseline_w: {report.baseline_w:.6f}  # measured idle, diagnostic")
    lines.append(
        f"  noise_floor_w: {report.noise_floor_w:.6f}  # arm rows only, post-exclusion"
    )
    lines.append(f"  term_source: {report.term_source}")
    lines.append("  terms:")
    for term in report.terms:
        lines.append(f"    - {term}")
    lines.append("  workloads:")
    for label in report.labels:
        lines.append(f"    - {label}")
    lines.append("  excluded_rows:")
    for line in report.excluded_rows:
        lines.append(f"    - {json.dumps(line)}")
    lines.append("  gates:")
    for gate in report.gates:
        verdict = "PASS" if gate.passed else ("FINDING" if gate.advisory else "REFUSE")
        lines.append(f"    {gate.name}: {json.dumps(verdict + ': ' + gate.detail)}")
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
        "--max-clock-drift-pct",
        type=float,
        default=5.0,
        help="how far the AI clock may move within a slot, and across the "
        "session's ARM slots, before the rows stop being comparable (default 5). "
        "The baseline-versus-arm gap is reported against the same cap as a "
        "finding rather than a refusal -- see the baseline_clock gate",
    )
    ap.add_argument(
        "--max-triviality-r2",
        type=float,
        default=0.95,
        help="refuse when the floor-and-launch-rate model alone, with no activity "
        "term, already explains this much of the fit target (default 0.95). A "
        "ranking a model can reproduce without knowing any activity is not "
        "evidence about activity",
    )
    ap.add_argument(
        "--terms",
        help="comma-separated activity terms to fit, overriding the DESIGNED set "
        "(one term per non-idle arm, from DESIGNED_ARM_TERMS). This is a "
        "pre-registration: name the terms before the run, not after seeing which "
        "set conditions best. The fit is stamped `operator-specified` wherever it "
        "is reported, and naming more terms than the design supports is refused by "
        "the identifiability gate rather than fitted.",
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
        max_clock_drift_pct=args.max_clock_drift_pct,
        max_triviality_r2=args.max_triviality_r2,
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
