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


def _plan_rows(grid, names=("hops", "size", "readport", "shared", "contention", "vc")):
    rows, run = [], 0
    for name in names:
        points = plan.EXPERIMENTS[name](grid)
        rows += plan.plan_rows(points, grid, first_run=run)
        run += len(points)
    return rows


def _payload_niu(row):
    """The NIU whose injection port carries this flow's payload.

    A write pushes its payload out of the master; a read's payload comes back
    out of the subordinate. Two flows contend for a port when this is equal.
    """
    if row["direction"] == plan.DIR_READ:
        return (row["sub_nx"], row["sub_ny"])
    return (row["mst_nx"], row["mst_ny"])


def _skew(measured, core, offset):
    """Give one core's wall clock a different epoch, 32-bit, as silicon did.

    Models the Blackhole finding in ``docs/bh_arch.md`` §4.4: the stamps are the
    low 32 bits of a per-tile free-running counter, so the offset both shifts
    and wraps. Same-core durations are untouched, which is the whole point.
    """
    out = []
    for r in measured:
        m = dict(r)
        if (m["mst_px"], m["mst_py"]) == core:
            m["t0"] = (m["t0"] + offset) % (1 << 32)
            m["t1"] = (m["t0"] + m["cycles"]) % (1 << 32)
        out.append(m)
    return out


def _synthesise(rows, *, congestion=0.0, port_contention=True, overlap=True):
    """Measurements under a stated hypothesis.

    ``cycles = 9 * round_trip_hops + N * (90 + ports * bytes/64 + k * shared)``.

    The shape matters as much as the numbers: the round-trip latency is paid
    ONCE for the whole pipelined region while the issue loop, the serialisation
    and any congestion are paid per transaction. That is why the hop report fits
    raw cycles and everything else fits cycles per transaction, and a generator
    that got it wrong would let a divided-by-N per-hop cost through.

    ``ports`` is how many flows share the injection port that carries their
    payload -- which is the MASTER's for a write and the SUBORDINATE's for a
    read, because a read's payload rides the return leg. That asymmetry is the
    whole design of the `readport` control, so a generator that ignored it
    would let a control which cannot see the port through. Setting
    ``port_contention=False`` models a harness whose flows never actually
    coincide, which must NOT be reported as "no congestion".
    """
    ports = collections.Counter()
    for r in rows:
        ports[(r["run"], _payload_niu(r))] += 1
    out = []
    for r in rows:
        share = ports[(r["run"], _payload_niu(r))] if port_contention else 1
        per = 90 + share * r["tx_bytes"] / 64.0 + congestion * r["shared_payload_links"]
        cycles = int(9 * r["rt_hops"] + per * r["num_tx"])
        m = dict(r)
        # Runs happen one after the other, so the file's own stamps bound how
        # long the session was -- which is what tells a clock-epoch offset from
        # a delay. Flows within a run share a base: the rendezvous released
        # them together.
        base = 1000 + 100_000 * int(r["run"])
        m.update(
            {
                "measured": 1,
                "cycles": cycles,
                "t0": base if overlap else base + 10_000_000 * r["flow"],
                "t1": (base if overlap else base + 10_000_000 * r["flow"]) + cycles,
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


def test_may_vary_covers_every_planned_experiment():
    assert set(plan.EXPERIMENTS) <= set(sweep.MAY_VARY)
    # `selfport` is retracted and the planner no longer emits it, but files
    # recorded before the retraction still contain it and must still parse.
    assert set(sweep.MAY_VARY) - set(plan.EXPERIMENTS) == {"selfport"}


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
    assert out["results"]["readport"]["ok"] is False


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
    rows = _plan_rows(grid, names=("hops", "size", "readport"))
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


def test_a_retracted_selfport_reading_gates_nothing(tmp_path):
    """A file recorded before the retraction must still analyse, and not vote.

    The planner cannot emit `selfport` any more, so the rows are built by hand
    from the shape those files have: two flows on one master core, which is
    exactly what `check_invariants` now refuses.
    """
    grid = _grid(tmp_path)
    rows = _synthesise(_plan_rows(grid, names=("hops", "size", "readport", "shared")))
    m = rows[0]["mst_nx"], rows[0]["mst_ny"]
    legacy = []
    for i, point in enumerate(("1flow", "2flows", "2flows")):
        r = dict(rows[0])
        r.update(
            {
                "run": 900,
                "exp": "selfport",
                "point": point,
                "flow": 0 if i != 2 else 1,
                "n_flows": 1 if i == 0 else 2,
                "proc": 0 if i != 2 else 1,
                "mst_nx": m[0],
                "mst_ny": m[1],
                "cycles": 17234,
                "num_tx": 64,
            }
        )
        legacy.append(r)
    out = sweep.sweep(rows + legacy, emit=_quiet)
    assert out["results"]["selfport"]["retracted"] is True
    assert out["results"]["selfport"]["ok"] is None
    # It read ~1.0x -- and the verdict is still driven by `readport`.
    assert out["verdict"] in ("FLAT", "MEASURED")


def test_the_vc_report_names_the_same_channel_point(rows):
    lines = []
    measured = _synthesise(rows)
    sweep.report_vc([r for r in measured if r["exp"] == "vc"], lines.append)
    text = "\n".join(lines)
    assert "both writers on this channel" in text
    assert "same VC / different VC" in text


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
        "READPORT",
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


# --- a per-tile wall clock is not a device clock ----------------------------


def test_a_reproducible_epoch_offset_is_named_and_corrected(rows):
    """One tile's clock keeping its own epoch must not read as "did not overlap".

    This is the Blackhole finding of ``docs/bh_arch.md`` §4.4 in miniature: the
    flows contended perfectly, and the only thing wrong is that two stamps came
    off two free-running counters that were never aligned to each other.
    """
    measured = _synthesise(rows, congestion=0.0)
    core = next(
        (r["mst_px"], r["mst_py"])
        for r in measured
        if r["exp"] == "shared" and r["flow"] == 1
    )
    skewed = _skew(measured, core, 1_143_914_613)

    found = sweep.clock_skew_report(skewed)
    assert set(found) == {core}
    assert found[core]["offset"] == 1_143_914_613
    assert found[core]["spread"] == 0

    # Uncorrected the affected runs read as no overlap at all; corrected they
    # read as the full overlap they really had, and the verdict comes back.
    assert min(sweep.overlap_report(skewed).values()) == 0.0
    assert min(sweep.overlap_report(skewed, found).values()) > 0.9
    assert sweep.sweep(skewed, emit=_quiet)["verdict"] == "FLAT"


def test_one_run_is_not_enough_to_call_a_disagreement_an_epoch(rows):
    """A single large disagreement is exactly what a real non-overlap looks like."""
    measured = _synthesise(rows, congestion=0.0)
    one_run = max(r["run"] for r in measured if r["exp"] == "shared")
    late = [
        dict(r, t0=r["t0"] + 1_143_914_613, t1=r["t1"] + 1_143_914_613)
        if r["run"] == one_run and r["flow"] == 1
        else r
        for r in measured
    ]
    assert sweep.clock_skew_report(late) == {}
    assert sweep.sweep(late, emit=_quiet)["verdict"] == "INVALID"


def test_a_delay_that_fits_inside_the_session_is_not_an_epoch(rows):
    """Reproducibility alone is not enough, and this is why.

    A rendezvous that always released one core late would reproduce to the
    cycle too. What separates it is that an epoch offset is *impossible as a
    delay*: it exceeds the whole session. This one does not, so it stays a
    failure.
    """
    measured = _synthesise(rows, congestion=0.0)
    core = next(
        (r["mst_px"], r["mst_py"])
        for r in measured
        if r["exp"] == "shared" and r["flow"] == 1
    )
    session = max(r["t0"] for r in measured) - min(r["t0"] for r in measured)
    late = _skew(measured, core, session // 2)
    assert session // 2 > sweep._SKEW_FLOOR
    assert sweep.clock_skew_report(late) == {}
    assert sweep.sweep(late, emit=_quiet)["verdict"] == "INVALID"


def test_a_session_that_crosses_the_stamp_wrap_still_finds_its_epoch():
    """The bar an epoch must clear is the session length, which is modular too.

    Computing it as ``max - min`` over 32-bit stamps reads nearly ``2**32``
    whenever the wall clock happens to wrap mid-run, and no real offset can
    clear that -- so a wrapping session silently swallows every epoch it has.
    Both 2026-08-09 Blackhole ``noc-epoch`` runs were graded INVALID for
    exactly this reason, and their flows had overlapped perfectly.
    """
    span = 400_000_000
    stamps = [(1 << 32) - span // 2 + step for step in range(0, span, span // 40)]
    assert max(stamps) >= 1 << 32  # this session really does cross the wrap
    wrapped = [s % (1 << 32) for s in stamps]

    assert max(wrapped) - min(wrapped) > (1 << 31)  # what the old span read
    assert sweep._elapsed_span(wrapped) == pytest.approx(span, rel=0.05)


def test_the_banked_wrapping_run_reads_its_epoch_and_its_congestion():
    """The 22:24 ``noc-epoch`` file, kept as the fixture for the bug above."""
    path = sweep.DEFAULT_MEASURED.with_name(
        "nocbench-blackhole-2026-08-09-2224-epoch.csv"
    )
    rows, _ = sweep.load_measured(path)
    skew = sweep.clock_skew_report(rows)
    assert skew[(11, 2)]["offset"] == -1_760_493_889
    assert skew[(11, 2)]["runs"] == 5
    overlaps = sweep.overlap_report(rows, skew)
    assert min(overlaps.values()) > 0.9
    assert sweep.sweep(rows, emit=_quiet)["verdict"] != "INVALID"


def test_the_stamp_wrapping_does_not_break_the_overlap(rows):
    """The stamps are 32 bits; a run that straddles the wrap is still a run."""
    measured = _synthesise(rows, congestion=0.0)
    wrapped = [dict(r) for r in measured]
    for r in wrapped:
        r["t0"] = (r["t0"] - 1000 + (1 << 32) - 8) % (1 << 32)
        r["t1"] = (r["t0"] + r["cycles"]) % (1 << 32)
    assert min(sweep.overlap_report(wrapped).values()) > 0.9


# --- the banked silicon run -------------------------------------------------


def test_the_banked_blackhole_run_still_reads_the_way_it_was_banked():
    """The numbers in ``docs/bh_arch.md`` section 4, re-derived from the file.

    Pinning them here is what keeps the document and the data from drifting
    apart: an edit to either that changed the reading would fail rather than be
    noticed by nobody.
    """
    rows, _comments = sweep.load_measured(sweep.DEFAULT_MEASURED)
    out = sweep.sweep(rows, emit=_quiet)
    assert out["verdict"] == "MEASURED"

    shared = out["results"]["shared"]
    assert shared[64]["shape"] == "FLAT"
    at = {n: cycles for n, cycles in shared[16384]["points"]}
    assert at[0] == pytest.approx(268.8, abs=0.2)
    assert at[1] == pytest.approx(518.3, abs=0.2)
    # ...and then it stops moving. Not to the cycle -- 1 through 7 span 15.8
    # cycles -- but against the 250-cycle step that is under 7 %, which is the
    # claim: the whole effect is at the FIRST shared link.
    rest = [at[n] for n in range(1, 8)]
    assert (max(rest) - min(rest)) < 0.1 * (at[1] - at[0])

    # The one tile whose wall clock keeps its own epoch. Two runs, spread <= 1.
    skew = sweep.clock_skew_report(rows)
    assert set(skew) == {(11, 2)}
    assert skew[(11, 2)]["offset"] == pytest.approx(1_143_914_613, abs=8)


def test_the_size_sweep_agrees_with_the_main_run_where_they_overlap():
    """Two separate runs, never averaged; the two cells they share must agree."""
    sizes = sweep.DEFAULT_MEASURED.with_name("nocbench-blackhole-sizes.csv")
    main_rows, _ = sweep.load_measured(sweep.DEFAULT_MEASURED)
    size_rows, _ = sweep.load_measured(sizes)

    def cell(rows, tx_bytes, links):
        vals = [
            sweep._per_tx(r)
            for r in rows
            if r["exp"] == "shared"
            and r["tx_bytes"] == tx_bytes
            and r["shared_payload_links"] == links
        ]
        return sum(vals) / len(vals)

    for tx_bytes in (64, 16384):
        for links in (0, 1):
            assert cell(main_rows, tx_bytes, links) == pytest.approx(
                cell(size_rows, tx_bytes, links), abs=1.0
            )


def test_the_delta_at_one_shared_link_is_one_transactions_occupancy():
    """The banked claim, stated as arithmetic rather than as a table.

    Above the regime boundary the cost of a second flow on a shared link is the
    first flow's own link occupancy -- ``tx_bytes / 64`` on Blackhole -- which
    is the whole reason a term of this shape is expressible at all.
    """
    sizes = sweep.DEFAULT_MEASURED.with_name("nocbench-blackhole-sizes.csv")
    rows, _ = sweep.load_measured(sizes)
    by = collections.defaultdict(list)
    for r in rows:
        if r["exp"] == "shared":
            by[(r["tx_bytes"], r["shared_payload_links"])].append(sweep._per_tx(r))

    def mean(key):
        return sum(by[key]) / len(by[key])

    for tx_bytes in (4096, 8192, 16384):
        delta = mean((tx_bytes, 1)) - mean((tx_bytes, 0))
        assert delta / (tx_bytes / 64) == pytest.approx(1.0, abs=0.05)
    # ...and below it, nothing happens at all: a 512 B packet holds the link
    # for 8 cycles against a ~40-cycle issue loop.
    for tx_bytes in (64, 512):
        assert abs(mean((tx_bytes, 1)) - mean((tx_bytes, 0))) < 0.5
