"""Tests for the NoC congestion sweep.

The property that matters most: a run with **no** congestion term must be
distinguishable from a run that measured nothing at all. Both read flat. The
sweep is supposed to tell them apart using the positive controls, and the tests
below drive it with synthetic measurements generated from a real plan under
three explicit hypotheses -- no congestion, a known per-link coefficient, and a
broken harness -- and check that the verdict differs in each.

The synthetic generator is deliberately a *different* model from anything in
the sweep: it is a closed form applied to the plan's own geometry columns, so
recovering its coefficient is a real round trip and not the sweep agreeing with
itself.
"""

import collections

import pytest

from tt_sim.perf import noc_congestion_plan as plan
from tt_sim.perf import noc_congestion_sweep as sweep


def _grid(tmp_path, arch="blackhole"):
    cols, rows = plan.WORKER_COLUMNS[arch], plan.WORKER_ROWS[arch]
    gx, gy = plan.SOC_GRID[arch]
    lines = [
        f"# nocbench-grid arch={arch} soc_grid_x={gx} soc_grid_y={gy}",
        "log_x,log_y,noc_x,noc_y",
    ]
    for ly, py in enumerate(rows):
        for lx, px in enumerate(cols):
            lines.append(f"{lx},{ly},{px},{py}")
    path = tmp_path / "grid.csv"
    path.write_text("\n".join(lines) + "\n")
    return plan.load_grid(path)


def _plan_rows(grid, names=("hops", "size", "selfport", "shared", "contention", "vc")):
    rows, run = [], 0
    for name in names:
        points = plan.EXPERIMENTS[name](grid)
        rows += plan.plan_rows(points, grid, first_run=run)
        run += len(points)
    return rows


def _synthesise(rows, *, congestion=0.0, port_contention=True, overlap=True):
    """Measurements under a stated hypothesis.

    ``cycles = 9 * round_trip_hops + N * (90 + ports * bytes/64 + k * shared)``.

    The shape matters as much as the numbers: the round-trip latency is paid
    ONCE for the whole pipelined region while the issue loop, the serialisation
    and any congestion are paid per transaction. That is why the hop report fits
    raw cycles and everything else fits cycles per transaction, and a generator
    that got it wrong would let a divided-by-N per-hop cost through.

    ``ports`` is how many flows share the issuing core's injection port. Setting
    ``port_contention=False`` models a harness whose flows never actually
    coincide, which must NOT be reported as "no congestion".
    """
    ports = collections.Counter()
    for r in rows:
        ports[(r["run"], r["mst_nx"], r["mst_ny"])] += 1
    out = []
    for r in rows:
        share = ports[(r["run"], r["mst_nx"], r["mst_ny"])] if port_contention else 1
        per = 90 + share * r["tx_bytes"] / 64.0 + congestion * r["shared_payload_links"]
        cycles = int(9 * r["rt_hops"] + per * r["num_tx"])
        m = dict(r)
        m.update(
            {
                "measured": 1,
                "cycles": cycles,
                "t0": 1000 if overlap else 1000 + 10_000_000 * r["flow"],
                "t1": (1000 if overlap else 1000 + 10_000_000 * r["flow"]) + cycles,
                "rendezvous_cycles": 50,
                "kernel_node_x": r["mst_px"],
                "kernel_node_y": r["mst_py"],
            }
        )
        out.append(m)
    return out


@pytest.fixture
def rows(tmp_path):
    return _plan_rows(_grid(tmp_path))


def _quiet(_line):
    pass


# --- the modules stay in step ----------------------------------------------


def test_may_vary_covers_exactly_the_planned_experiments():
    assert set(sweep.MAY_VARY) == set(plan.EXPERIMENTS)


def test_every_plan_column_is_either_ignored_or_classified():
    for exp, allowed in sweep.MAY_VARY.items():
        unknown = allowed - set(plan.PLAN_COLUMNS)
        assert not unknown, f"{exp} names columns no plan has: {unknown}"


# --- mechanics --------------------------------------------------------------


def test_ols_recovers_a_line():
    slope, intercept, r2 = sweep._ols([0, 1, 2, 3], [5, 8, 11, 14])
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(5.0)
    assert r2 == pytest.approx(1.0)


def test_ols_declines_a_single_x():
    assert sweep._ols([2, 2, 2], [1, 2, 3]) is None


def test_load_measured_round_trips(tmp_path, rows):
    measured = _synthesise(rows)
    header = list(measured[0])
    path = tmp_path / "m.csv"
    path.write_text(
        "# a comment\n"
        + ",".join(header)
        + "\n"
        + "\n".join(",".join(str(m[c]) for c in header) for m in measured)
        + "\n"
    )
    loaded, comments = sweep.load_measured(path)
    assert comments == ["# a comment"]
    assert len(loaded) == len(measured)
    assert loaded[0]["cycles"] == measured[0]["cycles"]
    assert isinstance(loaded[0]["exp"], str)


def test_recheck_invariants_is_silent_on_a_real_plan(rows):
    assert sweep.recheck_invariants(_synthesise(rows)) == []


def test_recheck_invariants_catches_a_column_that_should_not_have_moved(rows):
    measured = _synthesise(rows)
    for m in measured:
        if m["exp"] == "shared":
            m["num_tx"] += 1
            break
    complaints = sweep.recheck_invariants(measured)
    assert any("num_tx" in c for c in complaints)


def test_an_undeclared_experiment_is_refused(rows):
    measured = _synthesise(rows)
    measured[0]["exp"] = "something_new"
    assert any("refusing to interpret" in c for c in sweep.recheck_invariants(measured))


def test_overlap_report_spots_flows_that_never_coincided(rows):
    good = sweep.overlap_report(_synthesise(rows, overlap=True))
    bad = sweep.overlap_report(_synthesise(rows, overlap=False))
    assert good
    assert min(good.values()) > 0.9
    assert bad
    assert max(bad.values()) == 0.0


# --- the verdicts -----------------------------------------------------------


def test_no_congestion_reads_flat_when_the_controls_moved(rows):
    out = sweep.sweep(_synthesise(rows, congestion=0.0), emit=_quiet)
    assert out["verdict"] == "FLAT"
    assert {v["shape"] for v in out["results"]["shared"].values()} == {"FLAT"}


def test_a_known_coefficient_is_recovered(rows):
    out = sweep.sweep(_synthesise(rows, congestion=7.0), emit=_quiet)
    assert out["verdict"] == "MEASURED"
    for res in out["results"]["shared"].values():
        assert res["slope"] == pytest.approx(7.0, abs=0.01)
        assert res["shape"] == "LINEAR"


def test_a_saturating_effect_is_named_as_such(rows):
    measured = _synthesise(rows, congestion=0.0)
    for m in measured:
        if m["exp"] == "shared" and m["shared_payload_links"] > 0:
            m["cycles"] += 100 * m["num_tx"]
            m["t1"] += 100 * m["num_tx"]
    out = sweep.sweep(measured, emit=_quiet)
    assert out["verdict"] == "MEASURED"
    assert {v["shape"] for v in out["results"]["shared"].values()} == {"SATURATING"}


def test_a_harness_whose_flows_never_contend_is_invalid_not_flat(rows):
    """The failure mode the whole controls apparatus exists for."""
    out = sweep.sweep(
        _synthesise(rows, congestion=0.0, port_contention=False), emit=_quiet
    )
    assert out["verdict"] == "INVALID"
    assert out["results"]["selfport"]["ok"] is False


def test_flows_that_did_not_overlap_in_time_are_invalid(rows):
    out = sweep.sweep(_synthesise(rows, congestion=7.0, overlap=False), emit=_quiet)
    assert out["verdict"] == "INVALID"


def test_an_unstamped_flow_invalidates_the_run(rows):
    measured = _synthesise(rows)
    measured[0]["measured"] = 0
    assert sweep.sweep(measured, emit=_quiet)["verdict"] == "INVALID"


def test_a_kernel_that_ran_on_the_wrong_core_invalidates_the_run(rows):
    """The one failure that would silently reproduce the shipped dataset's
    problem: a measurement whose coordinates are not what was planned."""
    measured = _synthesise(rows)
    measured[0]["kernel_node_x"] += 1
    assert sweep.sweep(measured, emit=_quiet)["verdict"] == "INVALID"


def test_missing_controls_are_invalid_rather_than_flat(tmp_path):
    grid = _grid(tmp_path)
    rows = _plan_rows(grid, names=("shared",))
    assert sweep.sweep(_synthesise(rows), emit=_quiet)["verdict"] == "INVALID"


def test_a_run_with_controls_but_no_shared_experiment_is_partial(tmp_path):
    grid = _grid(tmp_path)
    rows = _plan_rows(grid, names=("hops", "size", "selfport"))
    assert sweep.sweep(_synthesise(rows), emit=_quiet)["verdict"] == "PARTIAL"


# --- the reports ------------------------------------------------------------


def test_the_hop_report_recovers_the_per_hop_cost_and_the_intercept(rows):
    out = sweep.sweep(_synthesise(rows), emit=_quiet)
    hops = out["results"]["hops"]
    assert hops["per_hop"] == pytest.approx(9.0, abs=0.01)
    # 90 (issue path) + one flow's own serialisation at the fixed size.
    assert hops["intercept"] > 90


def test_the_torus_check_reads_flat_within_a_family(rows):
    out = sweep.sweep(_synthesise(rows), emit=_quiet)
    for _family, slope, _r2 in out["results"]["hops"]["torus"]:
        assert slope == pytest.approx(0.0, abs=1e-9)


def test_a_shortest_path_noc_would_show_up_in_the_torus_check(rows):
    """A NoC that routed the short way round would make latency grow with the
    forward hop count inside a family. That must not read as noise."""
    measured = _synthesise(rows)
    for m in measured:
        if m["exp"] == "hops":
            m["cycles"] += 20 * m["fwd_hops"] * m["num_tx"]
    out = sweep.sweep(measured, emit=_quiet)
    assert all(abs(slope) > 1.0 for _f, slope, _r2 in out["results"]["hops"]["torus"])


def test_the_size_control_reports_its_slope(rows):
    out = sweep.sweep(_synthesise(rows), emit=_quiet)
    # 1 KiB is 16 flits of 64 B on Blackhole, one cycle each.
    assert out["results"]["size"]["slope_per_kib"] == pytest.approx(16.0, abs=0.5)


def test_the_contention_curve_is_reported_per_n(rows):
    out = sweep.sweep(_synthesise(rows), emit=_quiet)
    curve = out["results"]["contention"]
    assert sorted(curve) == sorted(
        {r["n_flows"] for r in rows if r["exp"] == "contention"}
    )


def test_emit_produces_a_report_a_human_can_read(rows):
    lines = []
    sweep.sweep(_synthesise(rows), emit=lines.append)
    text = "\n".join(lines)
    for section in (
        "VALIDITY",
        "HOPS",
        "SIZE",
        "SELFPORT",
        "SHARED",
        "CONTENTION",
        "VC",
        "VERDICT",
    ):
        assert section in text
    assert "NOISE FLOOR" in text


def test_main_exits_zero_on_a_valid_run(tmp_path, rows, capsys):
    measured = _synthesise(rows)
    header = list(measured[0])
    path = tmp_path / "m.csv"
    path.write_text(
        ",".join(header)
        + "\n"
        + "\n".join(",".join(str(m[c]) for c in header) for m in measured)
        + "\n"
    )
    assert sweep.main(["--measured", str(path)]) == 0
    assert "VERDICT" in capsys.readouterr().out
