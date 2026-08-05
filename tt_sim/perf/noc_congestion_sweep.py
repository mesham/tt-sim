"""Read back a ``nocbench`` run and say what it measured.

The planning half is :mod:`tt_sim.perf.noc_congestion_plan`; the executing half
is ``perfbench/nocbench``. This is the third: it re-checks the plan's invariants
against what actually came back, decides whether the run is valid at all, and
only then reports a coefficient.

That order matters. Against tt-sim every congestion experiment here is FORCED
to read flat -- the model charges an NIU for its own injection port and nothing
whatever for a router-to-router link, so two flows sharing links cannot
interact. A harness that measured nothing would read flat too. So a verdict of
"no congestion effect" is only issued when the positive controls moved:

* ``size`` must rise with transaction size (tt-sim models link serialisation);
* ``selfport`` must rise when a second flow joins the same injection port;
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
    "vc": {"vc"},
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


def overlap_report(rows):
    """Per multi-flow run: how much of the shorter flow ran while the other did.

    Concurrency is an assumption every congestion measurement rests on and none
    of them can check from its own number. The kernels stamp raw wall clock, so
    it is checkable here -- with one caveat stated rather than buried: it
    compares timestamps taken on different cores, which assumes the free-running
    counters are common to the device. Same-core durations need no such
    assumption and are what the coefficients are fitted to.
    """
    out = {}
    for run, group in _by_run(rows).items():
        if len(group) < 2 or not all(r.get("measured") for r in group):
            continue
        start, end = max(r["t0"] for r in group), min(r["t1"] for r in group)
        shortest = min(r["cycles"] for r in group)
        overlap = max(0, end - start)
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


def report_selfport(rows, emit, noise=None):
    """Positive control: a second flow on the same injection port."""
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
        f"  CONTROL: {'PASS' if ok else 'FAIL'} -- two flows out of one NIU must contend "
        f"for its injection port"
    )
    return {"ok": ok, "ratio": ratio}


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
    """Experiment 4: virtual-channel arbitration."""
    by_vc = defaultdict(list)
    for r in rows:
        by_vc[r["vc"]].append(_per_tx(r))
    means = {}
    for vc in sorted(by_vc):
        means[vc] = statistics.fmean(by_vc[vc])
        emit(f"  write VC {vc}  {means[vc]:>9.1f} cycles/tx")
    if len(means) < 2:
        return means
    span = max(means.values()) - min(means.values())
    emit(f"  span across VCs: {span:.1f} cycles")
    return means


REPORTERS = {
    "hops": report_hops,
    "size": report_size,
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
    overlaps = overlap_report(rows)
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
    for name in ("hops", "size", "selfport", "shared", "contention", "vc"):
        if name not in groups:
            continue
        emit("=" * 78)
        emit(f"{name.upper()}")
        emit("=" * 78)
        if name == "shared":
            results[name] = report_shared(groups[name], emit, noise)
        elif name == "selfport":
            results[name] = report_selfport(groups[name], emit, noise)
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
    for control in ("size", "selfport"):
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
            "this is a measurement and not a null harness. On tt-sim this is the FORCED "
            "answer (no router-to-router link is modelled); on silicon it would be a real "
            "and reportable result."
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measured", required=True, help="a nocbench output CSV")
    ap.add_argument("--min-overlap", type=float, default=0.5)
    args = ap.parse_args(argv)
    rows, comments = load_measured(args.measured)
    for c in comments:
        print(c)
    out = sweep(rows, min_overlap=args.min_overlap)
    return 0 if out["verdict"] in ("FLAT", "MEASURED", "PARTIAL") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
