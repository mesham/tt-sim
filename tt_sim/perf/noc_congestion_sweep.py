"""Read back a ``nocbench`` run and say what it measured.

The planning half is :mod:`tt_sim.perf.noc_congestion_plan`; the executing half
is ``perfbench/nocbench``. This is the third: it re-checks the plan's invariants
against what actually came back, decides whether the run is valid at all, and
only then reports a coefficient.

That order matters, and it mattered most while tt-sim's own answer was forced.
Until 2026-08-05 the model charged an NIU for its own injection port and
nothing whatever for a router-to-router link, so two flows sharing links could
not interact and every congestion experiment here read flat against the
simulator -- exactly as a harness that measured nothing would. It no longer
does (``NocLinkRegistry``, one watermark per link), and the same plan now reads
``SATURATING`` there at a saturating transaction size. The controls are what
made that difference legible rather than a coincidence, so they are still the
gate on any verdict:

* ``size`` must rise with transaction size (tt-sim models link serialisation);
* ``readport`` must rise when a second master reads from the same subordinate;
* every multi-flow point's timed regions must actually have overlapped.

If any of those fails, the verdict is ``INVALID`` and no coefficient is printed
-- because at that point the run does not distinguish "the hardware has no
per-link congestion" from "the two flows never met".

Run it
------

::

    python3 -m tt_sim.perf.noc_congestion_sweep --measured nocbench-blackhole.csv
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

#: What each experiment is allowed to move, per point. Anything else moving in
#: the measured file means the plan was edited by hand, or the executor ran a
#: different plan from the one analysed. Mirrors the declarations in
#: ``noc_congestion_plan``; ``noc_congestion_sweep_test`` pins the two together.
MAY_VARY = {
    "hops": {
        "sub_lx",
        "sub_ly",
        "sub_nx",
        "sub_ny",
        "sub_px",
        "sub_py",
        "fwd_hops",
        "rt_hops",
    },
    "size": {"tx_bytes"},
    "readport": {
        "n_flows",
        "mst_lx",
        "mst_ly",
        "mst_nx",
        "mst_ny",
        "mst_px",
        "mst_py",
        "flow",
        "shared_payload_links",
        "shared_other_links",
    },
    # RETRACTED, kept only so that files recorded before the retraction still
    # parse. `selfport` put two flows on one core's two data-movement RISCs;
    # `report_selfport` explains why its reading is not evidence either way and
    # the verdict below no longer consults it. `noc_congestion_plan` refuses to
    # emit one.
    "selfport": {
        "n_flows",
        "sub_lx",
        "sub_ly",
        "sub_nx",
        "sub_ny",
        "sub_px",
        "sub_py",
        "proc",
        "flow",
        "shared_payload_links",
        "shared_other_links",
    },
    "shared": {
        "mst_lx",
        "mst_ly",
        "mst_nx",
        "mst_ny",
        "mst_px",
        "mst_py",
        "sub_lx",
        "sub_ly",
        "sub_nx",
        "sub_ny",
        "sub_px",
        "sub_py",
        "proc",
        "flow",
        "fwd_hops",
        "tx_bytes",
        "shared_payload_links",
    },
    "contention": {
        "n_flows",
        "mst_lx",
        "mst_ly",
        "mst_nx",
        "mst_ny",
        "mst_px",
        "mst_py",
        "flow",
        "shared_payload_links",
        "shared_other_links",
    },
    # Two writers sharing one link, the SECOND one's VC swept. Flow A and flow
    # B differ in coordinates and hop count within every point, so those are
    # pooled across the group and have to be allowed; `vc` is the axis.
    "vc": {
        "vc",
        "flow",
        "mst_lx",
        "mst_ly",
        "mst_nx",
        "mst_ny",
        "mst_px",
        "mst_py",
        "sub_lx",
        "sub_ly",
        "sub_nx",
        "sub_ny",
        "sub_px",
        "sub_py",
        "fwd_hops",
    },
}

#: Columns never worth comparing: bookkeeping, or the measurement itself.
_IGNORED = {
    "run",
    "exp",
    "point",
    "measured",
    "cycles",
    "t0",
    "t1",
    "rendezvous_cycles",
    "kernel_node_x",
    "kernel_node_y",
    "kernel_noc",
}


def load_measured(path):
    """``(rows, comments)``; every numeric cell is an int, the rest a string."""
    rows, comments = [], []
    header = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        cells = line.split(",")
        if header is None:
            header = cells
            continue
        row = {}
        for name, value in zip(header, cells):
            try:
                row[name] = int(value)
            except ValueError:
                row[name] = value
        rows.append(row)
    if header is None:
        raise ValueError(f"{path} has no header row")
    return rows, comments


def _ols(xs, ys):
    """``(slope, intercept, r2)`` -- plain least squares, no dependencies."""
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope, intercept, r2


def by_experiment(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["exp"]].append(r)
    return out


def recheck_invariants(rows):
    """``[complaint]`` -- columns that moved which this experiment may not move.

    The plan asserted these at planning time. Re-asserting them here is not
    belt-and-braces: the file being analysed is the executor's output, so this
    is the check that the numbers being fitted came from the geometry that was
    planned rather than from an edited or stale plan.
    """
    complaints = []
    for exp, group in by_experiment(rows).items():
        allowed = MAY_VARY.get(exp)
        if allowed is None:
            complaints.append(
                f"{exp}: no MAY_VARY declaration; refusing to interpret it"
            )
            continue
        for column in group[0]:
            if column in _IGNORED or column in allowed:
                continue
            values = {r.get(column) for r in group}
            if len(values) > 1:
                complaints.append(
                    f"{exp}: column {column!r} took {sorted(str(v) for v in values)} but this "
                    f"experiment declares it fixed"
                )
    return complaints


#: The wall-clock stamps are the low 32 bits of ``RISCV_DEBUG_REG_WALL_CLOCK``,
#: so every difference between two of them is modular.
_WRAP = 1 << 32

#: Two cores in one run disagreeing about the time by more than this many
#: cycles is not a scheduling delay. The rendezvous releases every flow in a
#: run within a few hundred cycles of the others and the longest timed region
#: ever measured is ~35 000 cycles, so a disagreement three orders of magnitude
#: past that is either a genuine non-overlap or a difference of *epoch*.
_SKEW_FLOOR = 100_000

#: ...and this is what tells those two apart. A genuine non-overlap is a
#: property of one run; a difference of epoch is a property of the *core*, so
#: it reproduces, to the cycle, in every run that core appears in. Implied
#: offsets agreeing this closely across independent runs cannot be a delay.
_SKEW_TOLERANCE = 1_000

#: Below this many runs an outlying core's disagreement has not reproduced and
#: is left alone -- one large disagreement is exactly what a real non-overlap
#: looks like.
_SKEW_MIN_RUNS = 2


def _signed_wrap(delta):
    """``delta`` reduced into ``(-2**31, 2**31]``: the stamps are 32-bit."""
    delta %= _WRAP
    return delta - _WRAP if delta > _WRAP // 2 else delta


def _elapsed_span(stamps):
    """How long the session ran, from stamps that are only 32 bits wide.

    ``max - min`` is wrong here: the stamps are modular, so a session whose
    wall clock happens to cross ``2**32`` part-way through reports a span of
    nearly ``2**32`` -- and this span is the bar an epoch offset has to clear
    to be believed. A wrapping session therefore silently swallows every real
    offset, which is exactly what happened to two of the Blackhole runs.

    ``stamps`` arrive in file order, which is run order, and successive runs
    are milliseconds apart, so each step is unambiguous: reduce it into
    ``(-2**31, 2**31]`` and accumulate.
    """
    if not stamps:
        return 0
    walk = [0]
    for prev, cur in zip(stamps, stamps[1:]):
        walk.append(walk[-1] + _signed_wrap(cur - prev))
    return max(walk) - min(walk)


def _master(row):
    return (row.get("mst_px"), row.get("mst_py"))


def clock_skew_report(rows):
    """``{core: {"offset", "runs", "spread"}}`` -- per-core wall-clock epochs.

    ``RISCV_DEBUG_REG_WALL_CLOCK`` is a **per-tile** free-running counter with
    no defined epoch; nothing in tt-metal aligns the tiles to each other
    (``syncDeviceHost`` runs on one core and ``setShift`` applies its answer to
    the whole device). So two stamps from two tiles are only comparable if
    those tiles happen to share an epoch, and on the Blackhole part this was
    first seen on, one tile does not: see ``docs/bh_arch.md`` §4.4.

    Correcting for that would be circular if the correction were fitted per
    run -- it would assume the overlap it is used to check. It is not. An
    offset is only accepted when it **reproduces**: the same core, the same
    constant, in :data:`_SKEW_MIN_RUNS` or more independent runs, agreeing to
    within :data:`_SKEW_TOLERANCE` cycles. A flow that genuinely started a
    second late cannot start a second late by the same number of cycles twice,
    so the rule separates the two rather than papering over either.

    Reproducibility alone would still be fooled by a delay tied to a core's
    *role* rather than to the core -- a rendezvous that always releases flow 1
    late would reproduce too. So an offset must also be **impossible as a
    delay**: bigger than the whole session, measured as the span of the file's
    own stamps within the reference frame. A flow cannot start later than the
    program ran.

    The reference frame is the largest set of cores that agree with each other,
    which is the only frame derivable from the file. With fewer than three
    distinct cores there is no majority and nothing is corrected.
    """
    parent = {}

    def find(c):
        parent.setdefault(c, c)
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pairs = []  # (core_a, core_b, implied offset of b relative to a, run)
    for run, group in _by_run(rows).items():
        measured = [r for r in group if r.get("measured")]
        if len(measured) < 2:
            continue
        base = measured[0]
        for other in measured[1:]:
            a, b = _master(base), _master(other)
            if a == b:
                continue
            find(a), find(b)
            delta = _signed_wrap(other["t0"] - base["t0"])
            pairs.append((a, b, delta, run))
            if abs(delta) <= _SKEW_FLOOR:
                union(a, b)

    if len(parent) < 3:
        return {}
    sizes = defaultdict(int)
    for c in parent:
        sizes[find(c)] += 1
    biggest = max(sizes.values())
    if sum(1 for s in sizes.values() if s == biggest) > 1:
        return {}  # no majority frame; refuse to name one
    frame = max(sizes, key=lambda root: sizes[root])

    in_frame = [r for r in rows if r.get("measured") and find(_master(r)) == frame]
    session = _elapsed_span([r["t0"] for r in in_frame])

    implied = defaultdict(list)
    for a, b, delta, _run in pairs:
        if find(a) == frame and find(b) != frame:
            implied[b].append(delta)
        elif find(b) == frame and find(a) != frame:
            implied[a].append(-delta)

    out = {}
    for core, deltas in implied.items():
        if len(deltas) < _SKEW_MIN_RUNS:
            continue
        spread = max(deltas) - min(deltas)
        if spread > _SKEW_TOLERANCE:
            continue
        if abs(statistics.median(deltas)) <= session:
            continue
        out[core] = {
            "offset": int(statistics.median(deltas)),
            "runs": len(deltas),
            "spread": spread,
        }
    return out


def overlap_report(rows, skew=None):
    """Per multi-flow run: how much of the shorter flow ran while the other did.

    Concurrency is an assumption every congestion measurement rests on and none
    of them can check from its own number. The kernels stamp raw wall clock, so
    it is checkable here -- with one caveat stated rather than buried: it
    compares timestamps taken on different cores, which assumes the free-running
    counters are common to the device. Same-core durations need no such
    assumption and are what the coefficients are fitted to.

    That assumption is false on at least one part, so ``skew`` (from
    :func:`clock_skew_report`) is subtracted first when it is supplied. Each
    flow's *end* is taken as its own ``t0 + cycles`` rather than its ``t1``,
    which makes the arithmetic proof against the 32-bit stamp wrapping inside a
    timed region.
    """
    skew = skew or {}
    out = {}
    for run, group in _by_run(rows).items():
        if len(group) < 2 or not all(r.get("measured") for r in group):
            continue
        base = group[0]

        def rel(row, base=base):
            offset = skew.get(_master(row), {}).get("offset", 0)
            base_offset = skew.get(_master(base), {}).get("offset", 0)
            return _signed_wrap((row["t0"] - offset) - (base["t0"] - base_offset))

        starts = [rel(r) for r in group]
        ends = [s + r["cycles"] for s, r in zip(starts, group)]
        shortest = min(r["cycles"] for r in group)
        overlap = max(0, min(ends) - max(starts))
        out[run] = 0.0 if shortest == 0 else min(1.0, overlap / shortest)
    return out


def _coordinate_check(rows):
    """``([mismatched rows], note)`` from the kernels' own NIU self-report.

    A measurement whose coordinates are not the planned ones is the single
    failure that would silently reproduce the shipped dataset's problem, so the
    kernels read their own ``NOC_NODE_ID`` and the number is compared against
    the plan's PHYSICAL column -- the space the link arithmetic lives in.

    A device that answers 0 for that register reports every core as (0, 0),
    which is a missing register rather than every kernel running on one core.
    tt-sim is such a device. That case is reported as unavailable, not as a
    mismatch; a MIXED result is a genuine mismatch and is reported as one.
    """
    measured = [r for r in rows if r.get("measured") and r.get("noc") == 0]
    if not measured:
        return [], "no NoC 0 flows to check"
    reported = [
        r for r in measured if (r["kernel_node_x"], r["kernel_node_y"]) != (0, 0)
    ]
    if not reported:
        return [], (
            "coordinate self-report unavailable: every kernel's NOC_NODE_ID read 0. "
            "The host-side check against worker_core_from_logical_core still applied."
        )
    bad = [
        r
        for r in reported
        if (r["kernel_node_x"], r["kernel_node_y"]) != (r["mst_px"], r["mst_py"])
    ]
    note = None
    if len(reported) < len(measured):
        note = f"{len(measured) - len(reported)} flow(s) reported (0, 0) and were not checked"
    return bad, note


def _by_run(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["run"]].append(r)
    return out


def _per_tx(row):
    """Cycles per transaction, which is the quantity every fit uses.

    The timed region is N transactions and one closing barrier, so dividing by
    N removes the launch and the barrier's constant and leaves something
    comparable across points with different sizes."""
    return row["cycles"] / row["num_tx"] if row["num_tx"] else float("nan")


# ---------------------------------------------------------------------------
# The per-experiment reports.
# ---------------------------------------------------------------------------


def report_hops(rows, emit):
    """Experiment 1: the uncongested line, and the directional-torus check.

    This is the ONE report that fits RAW cycles rather than cycles per
    transaction, and the reason is worth stating because getting it wrong
    silently divides the answer by N. The timed region is N transactions
    pipelined behind a single closing barrier, so the round-trip *latency* is
    paid once for the whole region while bandwidth is paid per transaction.
    Dividing by N would therefore report a per-hop cost N times too small --
    which is exactly what a first simulator run did (2.24 instead of 9.0 at
    N = 4). Every other report here is bandwidth-shaped and per-transaction is
    the right normalisation for those.
    """
    families = defaultdict(list)
    for r in rows:
        families[r["point"].split("+")[0]].append(r)

    emit("  family        n   round-trip hops   cycles/region  (spread)")
    levels = {}
    for name in sorted(families):
        group = families[name]
        rt = {r["rt_hops"] for r in group}
        vals = [r["cycles"] for r in group]
        spread = max(vals) - min(vals)
        emit(
            f"  {name:<12} {len(group):>2}   {sorted(rt)!s:<15}   "
            f"{statistics.fmean(vals):>13.1f}  (+/-{spread / 2:.1f})"
        )
        if len(rt) == 1:
            levels[rt.pop()] = statistics.fmean(vals)

    # The torus test: within a family the round trip is predicted constant, so
    # any trend against the FORWARD hop count is the model being wrong about
    # the direction of travel, not noise.
    verdicts = []
    for name in sorted(families):
        group = families[name]
        fit = _ols([r["fwd_hops"] for r in group], [r["cycles"] for r in group])
        if fit is None:
            continue
        slope, _b, r2 = fit
        verdicts.append((name, slope, r2))
        emit(f"    {name}: cycles vs FORWARD hops -> slope {slope:+.2f}, r2 {r2:.2f}")
    emit(
        "    (a directional torus predicts slope 0 in every family: the reply travels "
        "the rest of the way round)"
    )

    if len(levels) >= 2:
        fit = _ols(sorted(levels), [levels[k] for k in sorted(levels)])
        slope, intercept, r2 = fit
        emit(
            f"  uncongested line over {len(levels)} round-trip levels: "
            f"{intercept:.1f} + {slope:.2f} * hops  (r2 {r2:.2f})"
        )
        emit(
            "    slope is the per-hop round-trip cost (ISA docs: ~9 cycles router to router); "
            "intercept is the issuing core's path plus the endpoint"
        )
        return {"per_hop": slope, "intercept": intercept, "torus": verdicts}
    emit("  fewer than two round-trip levels measured: no line to fit")
    return {"per_hop": None, "intercept": None, "torus": verdicts}


def report_size(rows, emit):
    """Positive control: bytes per transaction."""
    pts = sorted(((r["tx_bytes"], _per_tx(r)) for r in rows), key=lambda p: p[0])
    for size, cycles in pts:
        emit(f"  {size:>7} B  {cycles:>9.1f} cycles/tx")
    fit = _ols([p[0] for p in pts], [p[1] for p in pts])
    if fit is None:
        return {"ok": False, "reason": "fewer than two sizes"}
    slope, intercept, r2 = fit
    emit(
        f"  slope {slope * 1024:+.2f} cycles per KiB, intercept {intercept:.1f}, r2 {r2:.2f}"
    )
    rose = pts[-1][1] > pts[0][1] * 1.05
    emit(
        f"  CONTROL: {'PASS' if rose else 'FAIL'} -- larger transactions must cost more"
    )
    return {"ok": rose, "slope_per_kib": slope * 1024, "intercept": intercept}


def report_readport(rows, emit, noise=None):
    """Positive control: a second master reading from the same subordinate.

    Replaces ``report_selfport``. The difference that matters is not the
    geometry but the *bookkeeping*: one data-movement RISC per core, so each
    kernel's ``noc_async_read_barrier`` waits on its own core's responses and
    the timed region ends when that core's traffic has landed rather than when
    half of somebody else's has.
    """
    per_point = defaultdict(list)
    for r in rows:
        per_point[r["point"]].append(_per_tx(r))
    for name in sorted(per_point):
        emit(
            f"  {name:<8} {statistics.fmean(per_point[name]):>9.1f} cycles/tx  (n={len(per_point[name])})"
        )
    if "1flow" not in per_point or "2flows" not in per_point:
        return {"ok": False, "reason": "missing a point"}
    one = statistics.fmean(per_point["1flow"])
    two = statistics.fmean(per_point["2flows"])
    ratio = two / one if one else float("nan")
    emit(f"  ratio {ratio:.2f}x")
    # The bar is the harness's own resolution, not a round number: the second
    # flow must cost more than the row family's spread AND more than 2 % of the
    # single-flow cost, so a control that "passes" on noise cannot.
    floor = max(noise or 0.0, 0.02 * one)
    ok = (two - one) > floor
    emit(
        f"  (bar: +{floor:.1f} cycles/tx, from the noise floor and 2 % of the single-flow cost)"
    )
    emit(
        f"  CONTROL: {'PASS' if ok else 'FAIL'} -- two masters reading from one NIU must "
        f"contend for its injection port"
    )
    emit(
        "  (this proves the flows contend; it does not say WHERE. Two response streams out "
        "of one tile share the first link out of it as well as the port, and on silicon "
        "those are one reading. Attribution is experiment 2's job.)"
    )
    return {"ok": ok, "ratio": ratio}


def report_selfport(rows, emit, noise=None):
    """RETRACTED control: two flows on one core's two data-movement RISCs.

    Reported, because files recorded before the retraction still contain it,
    but it gates nothing. ``noc_async_*_barrier`` compares a per-NIU *hardware*
    counter against a per-RISC *software* one seeded from it at kernel init, so
    with two issuers on one NIU each RISC's ``==`` is satisfied by any N acks
    and both kernels stop at the halfway point. The ratio it prints is
    therefore about 1.0 whether the injection port serialises perfectly or does
    not exist -- which was established by deleting the mechanism in tt-sim and
    watching the ratio move by 0.04, in the wrong direction, while the absolute
    cost moved 35 %. See docs/plans/cost-model.md.
    """
    per_point = defaultdict(list)
    for r in rows:
        per_point[r["point"]].append(_per_tx(r))
    for name in sorted(per_point):
        emit(
            f"  {name:<8} {statistics.fmean(per_point[name]):>9.1f} cycles/tx  (n={len(per_point[name])})"
        )
    one = statistics.fmean(per_point.get("1flow") or [float("nan")])
    two = statistics.fmean(per_point.get("2flows") or [float("nan")])
    emit(f"  ratio {two / one if one else float('nan'):.2f}x")
    emit(
        "  RETRACTED -- this control is blind. Both kernels share one NIU's hardware ack "
        "counter, so each stops after any N acks and the region ends at the halfway point; "
        "the ratio reads ~1.0 whether the port serialises or not. It gates nothing. The "
        "replacement is `readport`, and the planner refuses to emit `selfport` at all "
        "(two flows on one master core can also hang a card)."
    )
    return {"ok": None, "retracted": True, "ratio": two / one if one else None}


def report_shared(rows, emit, noise=None):
    """Experiment 2: the per-shared-link coefficient, per transaction size."""
    by_size = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_size[r["tx_bytes"]][r["shared_payload_links"]].append(_per_tx(r))

    results = {}
    for size in sorted(by_size):
        emit(f"  {size} B:")
        shares = sorted(by_size[size])
        means = [statistics.fmean(by_size[size][s]) for s in shares]
        for s, m in zip(shares, means):
            emit(f"    {s:>2} shared links  {m:>9.1f} cycles/tx")
        fit = _ols(shares, means)
        if fit is None:
            emit("    fewer than two overlap values: nothing to fit")
            continue
        slope, intercept, r2 = fit
        span = max(means) - min(means)
        emit(
            f"    slope {slope:+.2f} cycles per shared link, intercept {intercept:.1f}, r2 {r2:.2f}"
        )
        if noise is not None:
            emit(f"    span {span:.1f} cycles against a noise floor of {noise:.1f}")
        flat = span <= (noise if noise is not None else 0.0) or abs(slope) < 1e-9
        # Linear vs saturating: does the step from 0 to 1 shared links account
        # for most of the whole span? If it does, the effect is a bandwidth
        # split at the first shared link rather than a per-hop adder.
        shape = "FLAT"
        if not flat:
            first = means[1] - means[0] if len(means) > 1 else 0.0
            shape = "SATURATING" if span > 0 and first / span > 0.7 else "LINEAR"
        emit(f"    SHAPE: {shape}")
        results[size] = {
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "span": span,
            "shape": shape,
            # The per-share means themselves. A SATURATING fit's slope is not a
            # coefficient -- it is a regression drawn through a step -- so
            # anything reading this result needs the points and not the line.
            "points": list(zip(shares, means)),
        }
    return results


def report_contention(rows, emit):
    """Experiment 3: the endpoint contention curve."""
    by_n = defaultdict(list)
    shared = {}
    for r in rows:
        by_n[r["n_flows"]].append(_per_tx(r))
        shared.setdefault(r["n_flows"], 0)
        shared[r["n_flows"]] += r["shared_payload_links"]
    emit("   N   cycles/tx   aggregate B/cycle   shared payload links (covariate)")
    base = None
    for n in sorted(by_n):
        mean = statistics.fmean(by_n[n])
        base = mean if base is None else base
        emit(f"  {n:>2}   {mean:>9.1f}   {'-':>17}   {shared[n]:>4}")
    emit(
        "  the shared-link column is NOT held fixed and cannot be: N flows into one "
        "endpoint must share the links next to it. See HYPOTHESES['contention']."
    )
    return {n: statistics.fmean(v) for n, v in by_n.items()}


def report_vc(rows, emit):
    """Experiment 4: virtual-channel arbitration, two writers on one link.

    Every point has flow A pinned at ``NOC_UNICAST_WRITE_VC`` (1) and flow B on
    the swept channel, so the point to read is ``vc1`` -- both writers on one
    channel -- against the other three. The reading is taken per point rather
    than per row, because the two flows in a point carry different ``vc``
    values by construction.
    """
    by_point = defaultdict(list)
    swept = {}
    for r in rows:
        by_point[r["point"]].append(_per_tx(r))
        # Flow 0 is the pinned one; flow 1 carries the swept channel.
        if r["flow"] == 1:
            swept[r["point"]] = r["vc"]
    means = {}
    for name in sorted(by_point):
        means[swept.get(name, name)] = statistics.fmean(by_point[name])
    for vc in sorted(means, key=str):
        same = " (both writers on this channel)" if vc == 1 else ""
        emit(f"  flow B on VC {vc}  {means[vc]:>9.1f} cycles/tx{same}")
    if len(means) < 2:
        return means
    span = max(means.values()) - min(means.values())
    emit(f"  span across VCs: {span:.1f} cycles")
    if 1 in means and len(means) > 1:
        others = [v for k, v in means.items() if k != 1]
        emit(
            f"  same VC / different VC: {means[1] / statistics.fmean(others):.2f}x -- above 1 "
            f"means the shared link is arbitrated per virtual channel; 1.00 means it is "
            f"occupancy and the channel number does not enter it"
        )
    return means


REPORTERS = {
    "hops": report_hops,
    "size": report_size,
    "readport": report_readport,
    "selfport": report_selfport,
    "shared": report_shared,
    "contention": report_contention,
    "vc": report_vc,
}


def sweep(rows, emit=print, min_overlap=0.5):
    """Run every report present in ``rows`` and return the verdict."""
    groups = by_experiment(rows)

    unmeasured = [r for r in rows if not r.get("measured")]
    coord_mismatch, coord_note = _coordinate_check(rows)
    complaints = recheck_invariants(rows)
    # Which NoC a kernel actually ran on. `DataMovementConfig.noc` is a request;
    # this is the kernel's own `noc_index`. Two flows the plan put on one NoC
    # that land on different ones do not share an injection port, which changes
    # what a self-port control means -- so it is checked rather than assumed.
    wrong_noc = [
        r
        for r in rows
        if r.get("measured") and "kernel_noc" in r and r["kernel_noc"] != r["noc"]
    ]
    complaints += [
        f"{r['exp']} run {r['run']} flow {r['flow']}: planned NoC {r['noc']} but the kernel "
        f"ran on NoC {r['kernel_noc']}"
        for r in wrong_noc
    ]
    skew = clock_skew_report(rows)
    overlaps = overlap_report(rows, skew)
    poor = {run: frac for run, frac in overlaps.items() if frac < min_overlap}

    emit("=" * 78)
    emit("VALIDITY")
    emit("=" * 78)
    emit(f"  flows in file           {len(rows)}")
    emit(f"  unmeasured (no stamp)   {len(unmeasured)}")
    emit(f"  coordinate mismatches   {len(coord_mismatch)}")
    if coord_note:
        emit(f"    {coord_note}")
    emit(f"  invariant complaints    {len(complaints)}")
    for c in complaints:
        emit(f"    {c}")
    if skew:
        emit(
            f"  clock-epoch skew        {len(skew)} core(s) whose wall clock keeps a "
            f"different epoch from the rest of the device"
        )
        for core in sorted(skew, key=lambda c: (c is None, c)):
            info = skew[core]
            emit(
                f"    core {core}: {info['offset']:+d} cycles, reproduced over "
                f"{info['runs']} runs to within {info['spread']} cycles -- subtracted "
                f"before the overlap below"
            )
        emit(
            "    (RISCV_DEBUG_REG_WALL_CLOCK is per tile and free-running; a constant "
            "that reproduces across independent runs cannot be a scheduling delay. "
            "Same-core durations, which every coefficient is fitted to, are unaffected.)"
        )
    if overlaps:
        emit(
            f"  multi-flow runs         {len(overlaps)}, median timed-region overlap "
            f"{statistics.median(overlaps.values()):.2f}, {len(poor)} below {min_overlap}"
        )
        for run in sorted(poor):
            emit(f"    run {run}: overlap {poor[run]:.2f} -- flows barely coincided")
    emit("")

    results = {}
    noise = None
    for name in ("hops", "size", "readport", "selfport", "shared", "contention", "vc"):
        if name not in groups:
            continue
        emit("=" * 78)
        emit(f"{name.upper()}")
        emit("=" * 78)
        if name == "shared":
            results[name] = report_shared(groups[name], emit, noise)
        elif name in ("readport", "selfport"):
            results[name] = REPORTERS[name](groups[name], emit, noise)
        else:
            results[name] = REPORTERS[name](groups[name], emit)
        if name == "hops":
            # The row family predicts one round-trip cost; its spread is this
            # harness's resolution, and every "flat" claim is read against it.
            row_family = [r for r in groups["hops"] if r["point"].startswith("row")]
            if len(row_family) > 1:
                vals = [_per_tx(r) for r in row_family]
                noise = max(vals) - min(vals)
                emit(
                    f"  NOISE FLOOR (row family spread): {noise:.1f} cycles/tx -- the row "
                    f"family is predicted constant, so its spread is this harness's "
                    f"resolution, and every 'flat' claim below is read against it"
                )
        emit("")

    # --- the verdict --------------------------------------------------------
    emit("=" * 78)
    emit("VERDICT")
    emit("=" * 78)
    controls_ok = True
    for control in ("size", "readport"):
        res = results.get(control)
        if res is None:
            emit(
                f"  {control}: NOT RUN -- without it a flat congestion reading means nothing"
            )
            controls_ok = False
        elif not res.get("ok"):
            emit(f"  {control}: FAILED -- {res.get('reason', 'did not move')}")
            controls_ok = False
        else:
            emit(f"  {control}: passed")
    if "selfport" in results:
        emit(
            "  selfport: RETRACTED and not consulted -- it reads ~1.0x whether or not the "
            "injection port serialises (see report above). A file containing it predates the "
            "retraction; re-plan to get `readport` instead."
        )
    if unmeasured or coord_mismatch or complaints:
        emit(
            "  RESULT: INVALID -- the file does not describe the experiment it claims to"
        )
        return {"verdict": "INVALID", "results": results}
    if not controls_ok:
        emit(
            "  RESULT: INVALID -- the controls did not move, so a flat congestion reading "
            "is indistinguishable from a harness that measured nothing"
        )
        return {"verdict": "INVALID", "results": results}
    if poor:
        emit(
            f"  RESULT: INVALID -- {len(poor)} multi-flow run(s) whose timed regions barely "
            f"overlapped; those flows did not contend"
        )
        return {"verdict": "INVALID", "results": results}

    shared = results.get("shared") or {}
    shapes = {v["shape"] for v in shared.values()}
    if not shared:
        emit("  RESULT: CONTROLS PASS, no shared-link experiment in this file")
        return {"verdict": "PARTIAL", "results": results}
    if shapes == {"FLAT"}:
        emit(
            "  RESULT: NO CONGESTION EFFECT. The controls moved and the flows overlapped, so "
            "this is a measurement and not a null harness. It used to be the FORCED answer "
            "on tt-sim; since 2026-08-05 the model charges link occupancy, so a flat reading "
            "there is now a reading like any other -- check the transaction size is above "
            "the issue loop before believing it."
        )
        return {"verdict": "FLAT", "results": results}
    emit(f"  RESULT: CONGESTION MEASURED -- shapes {sorted(shapes)}")
    for size, res in sorted(shared.items()):
        emit(
            f"    {size} B: {res['slope']:+.2f} cycles per shared link ({res['shape']}, r2 {res['r2']:.2f})"
        )
    emit(
        "    NOTE: a coefficient fitted here is `vendor_source`-grade at best -- it is a "
        "measurement on one part, not a published number. See docs/plans/cost-model.md's "
        "provenance convention before it goes anywhere near unit_costs.yaml."
    )
    return {"verdict": "MEASURED", "results": results}


#: The banked silicon run, so the analysis reproduces with no hardware. It is
#: the file with the controls in it; the ``-sizes`` companion beside it is a
#: separate run and carries none, which is why it is not the default.
DEFAULT_MEASURED = Path(__file__).with_name("datasets") / "nocbench-blackhole.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--measured",
        default=str(DEFAULT_MEASURED),
        help="a nocbench output CSV (default: the banked Blackhole silicon run)",
    )
    ap.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args(argv)
    rows, comments = load_measured(args.measured)
    for c in comments:
        print(c)
    out = sweep(rows, min_overlap=args.min_overlap)
    return 0 if out["verdict"] in ("FLAT", "MEASURED", "PARTIAL") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
