"""The rung-3 sweep harness: what it excludes, and what it derives.

Everything here is pure arithmetic on a synthetic CSV, so none of it needs
hardware, tt-metal, or a benchmark run. That is deliberate: the parts of this
harness that can silently go wrong are the ones that turn raw points into a
cost -- the control subtraction, the unroll divisor, the exclusion ladder and
the fidelity differencing -- and every one of them is checkable against a CSV
whose right answer is known by construction.

The one thing these tests cannot check is whether the *benchmark* measures what
it claims. That is what ``perfbench/tensixbench``'s own validity gate
(monotonicity, R^2, fidelity separation) is for, and it runs on the device.

Run:  python3 -m pytest tt_sim/perf/tensix_bench_sweep_test.py
"""

import io

import pytest

from tt_sim.perf import tensix_bench_sweep as sweep

HEADER = (
    "# tensixbench raw points\n"
    "# arch=blackhole magic=0x7B10CE02 unroll=64\n"
    "phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles\n"
)


def _csv(tmp_path, rows, header=HEADER):
    path = tmp_path / "bench.csv"
    path.write_text(header + "".join(rows))
    return path


def _phase_a(probe, unit, per_block, variant="t1", thread=1, threads=1, intercept=17):
    """Four points whose slope is exactly ``per_block`` cycles per block."""
    return [
        f"A,{variant},1,{probe},{unit},{threads},{thread},{n},64,"
        f"{intercept + per_block * n}\n"
        for n in (4, 8, 12, 16)
    ]


def _control(per_block=2, variant="t1", thread=1, threads=1):
    return [
        f"A,{variant},0,loop_overhead,-,{threads},{thread},{n},64,{5 + per_block * n}\n"
        for n in (4, 8, 12, 16)
    ]


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------


def test_arch_comes_from_the_comment_line(tmp_path):
    _, meta = sweep.read_csv(_csv(tmp_path, _control()))
    assert meta["arch"] == "blackhole"


def test_comment_lines_are_not_data(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, _control()))
    assert len(rows) == 4
    assert all(r["probe"] == "loop_overhead" for r in rows)


# ---------------------------------------------------------------------------
# The fit, and the two cancellations the whole method rests on.
# ---------------------------------------------------------------------------


def test_slope_ignores_the_intercept():
    """The fixed cost -- clock reads, barrier, call -- must cancel exactly."""
    xs = [4, 8, 12, 16]
    for intercept in (0, 17, 100_000):
        _, slope, r2 = sweep.linear_fit(xs, [intercept + 66 * x for x in xs])
        assert slope == pytest.approx(66.0)
        assert r2 == pytest.approx(1.0)


def test_r2_falls_when_the_series_is_not_linear():
    _, _, r2 = sweep.linear_fit([1, 2, 3, 4], [1, 2, 3, 40])
    assert r2 < sweep.MIN_R2


def test_control_subtraction_and_unroll_divisor(tmp_path):
    """(slope - loop overhead) / unroll is the per-instruction cost."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control(per_block=2) + _phase_a("ADDDMAREG", "THCON", 194))
    )
    series = sweep.apply_control(sweep.series_of(rows))
    entry = next(s for s in series if s["probe"] == "ADDDMAREG")
    # (194 - 2) / 64 = 3.0
    assert entry["measured"] == pytest.approx(3.0)


def test_the_control_is_matched_per_variant_and_thread(tmp_path):
    """A contended run must be corrected by its own contended loop overhead."""
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(per_block=2)
            + _control(per_block=10, variant="t3", thread=1, threads=3)
            + _phase_a("NOP", "NONE", 66)
            + _phase_a("NOP", "NONE", 74, variant="t3", threads=3),
        )
    )
    series = sweep.apply_control(sweep.series_of(rows))
    solo = next(s for s in series if s["probe"] == "NOP" and s["variant"] == "t1")
    contended = next(s for s in series if s["probe"] == "NOP" and s["variant"] == "t3")
    assert solo["measured"] == pytest.approx(1.0)
    assert contended["measured"] == pytest.approx(1.0)  # (74 - 10) / 64


# ---------------------------------------------------------------------------
# The fit's own resolution, which is what stops a rounding error being a
# finding.
# ---------------------------------------------------------------------------


def test_a_perfect_fit_has_no_slope_uncertainty():
    xs = [4, 8, 12, 16]
    ys = [17 + 66 * x for x in xs]
    intercept, slope, _ = sweep.linear_fit(xs, ys)
    assert sweep.slope_stderr(xs, ys, intercept, slope) == pytest.approx(0.0)


def test_one_off_scatter_shows_up_as_slope_uncertainty():
    xs = [4, 8, 12, 16]
    clean = [17 + 66 * x for x in xs]
    dirty = list(clean)
    dirty[0] += 15  # a cold-cache first burst, the shape silicon actually shows
    dirty_i, dirty_slope, r2 = sweep.linear_fit(xs, dirty)
    assert r2 > sweep.MIN_R2, "R^2 stays high, which is why it cannot be the gate"
    assert dirty_slope < 66.0, "and the slope is pulled off the true value"
    assert sweep.slope_stderr(xs, dirty, dirty_i, dirty_slope) > 0.0
    assert sweep.slope_stderr(xs, clean, *sweep.linear_fit(xs, clean)[:2]) == 0.0


def test_the_resolution_is_at_least_the_control_over_subtraction(tmp_path):
    """The loop overhead is subtracted unconditionally but overlaps whenever
    the unit back-pressures, so it can only ever make `measured` too small --
    by up to slope(control)/unroll, which is the floor of the resolution."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control(per_block=2) + _phase_a("ADDDMAREG", "THCON", 194))
    )
    series = sweep.apply_control(sweep.series_of(rows))
    entry = next(s for s in series if s["probe"] == "ADDDMAREG")
    assert entry["resolution"] == pytest.approx(2.0 / 64)


def test_a_residual_inside_the_resolution_is_not_reported_as_an_over_charge(tmp_path):
    """A table of 3.0 measured at 2.99 is the instrument, not the hardware."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control(per_block=2) + _phase_a("ADDDMAREG", "THCON", 193))
    )
    out = io.StringIO()
    sweep.report(rows, "blackhole", out=out)
    text = out.getvalue()
    assert "INSIDE the instrument" in text
    assert "VERDICT: yes" in text


def test_a_residual_beyond_the_resolution_is_reported_as_an_over_charge(tmp_path):
    """A whole cycle is not a rounding error, and must still be called."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control(per_block=2) + _phase_a("ADDDMAREG", "THCON", 130))
    )
    out = io.StringIO()
    sweep.report(rows, "blackhole", out=out)
    text = out.getvalue()
    assert "VERDICT: NO" in text
    assert "OVER-CHARGES" in text


def test_the_resolution_note_names_both_of_its_terms():
    note = sweep.FIT_RESOLUTION_NOTE.format(sigma=sweep.RESOLUTION_SIGMA)
    assert "CONTROL OVER-SUBTRACTION" in note
    assert "FIT UNCERTAINTY" in note
    # And admits what it costs, rather than only what it buys.
    assert "cannot detect an over-charge smaller" in note


# ---------------------------------------------------------------------------
# The exclusion ladder. Declared before any residual; asserted here.
# ---------------------------------------------------------------------------


def test_every_exclusion_rule_carries_a_reason():
    for name, reason, predicate in sweep._exclusions():
        assert name
        assert callable(predicate)
        assert len(reason) > 40, f"{name} needs a real reason, not a label"


def test_the_control_probe_is_excluded(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, _control() + _phase_a("NOP", "NONE", 66)))
    series = sweep.attach_table(sweep.apply_control(sweep.series_of(rows)), "blackhole")
    kept, _ = sweep.retained(series)
    assert [s["probe"] for s in kept] == ["NOP"]


def test_contended_series_are_excluded_from_the_per_instruction_sweep(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control()
            + _control(per_block=2, variant="t3", thread=0, threads=3)
            + _phase_a("NOP", "NONE", 66)
            + _phase_a("NOP", "NONE", 66, variant="t3", thread=0, threads=3),
        )
    )
    series = sweep.attach_table(sweep.apply_control(sweep.series_of(rows)), "blackhole")
    kept, ladder = sweep.retained(series)
    assert all(s["active_threads"] == 1 for s in kept)
    assert dict((name, removed) for name, removed, _ in ladder)["active_threads > 1"]


def test_an_opcode_the_tables_have_no_opinion_about_is_excluded(tmp_path):
    """`MOVD2B` is in the instruction set and `provenance: unknown` in the table."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control() + _phase_a("MOVD2B", "MATH", 66))
    )
    series = sweep.attach_table(sweep.apply_control(sweep.series_of(rows)), "blackhole")
    kept, _ = sweep.retained(series)
    assert kept == []


def test_a_nonlinear_series_is_excluded(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control()
            + [
                f"A,t1,1,NOP,NONE,1,1,{n},64,{c}\n"
                for n, c in ((4, 10), (8, 900), (12, 12), (16, 3000))
            ],
        )
    )
    series = sweep.attach_table(sweep.apply_control(sweep.series_of(rows)), "blackhole")
    kept, _ = sweep.retained(series)
    assert kept == []


# ---------------------------------------------------------------------------
# The tables.
# ---------------------------------------------------------------------------


def test_the_occupancy_comes_from_the_shipped_table_not_a_constant(tmp_path):
    """A table edit must move the comparison, so nothing is restated here."""
    from tt_sim.perf.costs import load_costs

    expected = load_costs("blackhole").find("ADDDMAREG").occupancy.cycles
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control() + _phase_a("ADDDMAREG", "THCON", 194))
    )
    series = sweep.attach_table(sweep.apply_control(sweep.series_of(rows)), "blackhole")
    entry = next(s for s in series if s["probe"] == "ADDDMAREG")
    assert entry["table"] == float(expected)


def test_bounds_are_charged_at_their_low_end_like_the_model(tmp_path):
    """`range` and `at_least` take the minimum, matching `perf.model`'s policy."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control() + _phase_a("ADDDMAREG", "THCON", 194))
    )
    series = sweep.attach_table(sweep.apply_control(sweep.series_of(rows)), "blackhole")
    entry = next(s for s in series if s["probe"] == "ADDDMAREG")
    assert entry["bound"] == "range"
    assert entry["table_max"] is not None
    assert entry["table_max"] > entry["table"]


def test_unwired_units_are_read_from_the_list_that_owns_them():
    from tt_sim.perf.costs_test import UNWIRED_UNITS

    assert sweep.unwired_units("blackhole") == set(UNWIRED_UNITS)


# ---------------------------------------------------------------------------
# Phase B: the fidelity difference.
# ---------------------------------------------------------------------------


def _phase_b(variant, per_iter):
    return [
        f"B,{variant},0,matmul_tiles,MATH,3,1,{n},1,{40 + per_iter * n}\n"
        for n in (8, 16, 24, 32)
    ]


def test_the_fidelity_difference_is_what_is_compared_not_the_absolute(tmp_path):
    """Everything the loop does besides the extra MVMULs must cancel."""
    from tt_sim.perf.costs import load_costs

    per_phase = (
        load_costs("blackhole")
        .units["MATH"]
        .extras["fidelity_phases"]["mvmuls_per_tile"]["count"]
    )
    # An absolute cost with a large, fidelity-independent overhead.
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _phase_b("LoFi", 400 + per_phase)
            + _phase_b("HiFi2", 400 + 2 * per_phase)
            + _phase_b("HiFi4", 400 + 4 * per_phase),
        )
    )
    out = io.StringIO()
    sweep.report(rows, "blackhole", out=out)
    text = out.getvalue()
    assert "Fidelity phases" in text
    # LoFi -> HiFi2 is one phase; HiFi2 -> HiFi4 is two.
    assert f"{float(per_phase):.2f}" in text
    assert f"{float(2 * per_phase):.2f}" in text
    assert "residual" in text


def test_a_feeder_limited_phase_b_is_called_out_rather_than_reported(tmp_path):
    """Three equal fidelity slopes measure nothing, and must say so."""
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _phase_b("LoFi", 500) + _phase_b("HiFi2", 500) + _phase_b("HiFi4", 500),
        )
    )
    out = io.StringIO()
    sweep.report(rows, "blackhole", out=out)
    assert "do not separate" in out.getvalue()


# ---------------------------------------------------------------------------
# The issue-limit discriminator and the differential.
# ---------------------------------------------------------------------------


def test_a_shared_unit_is_distinguished_from_no_back_pressure(tmp_path):
    """Three threads, one 1-IPC unit: each thread's cost must triple."""
    shared = (
        _control()
        + _control(variant="t3", thread=0, threads=3)
        + _phase_a("ADDDMAREG", "THCON", 194)
        + _phase_a("ADDDMAREG", "THCON", 2 + 3 * 192, variant="t3", thread=0, threads=3)
        + _phase_a("NOP", "NONE", 66)
        + _phase_a("NOP", "NONE", 66, variant="t3", thread=0, threads=3)
    )
    rows, _ = sweep.read_csv(_csv(tmp_path, shared))
    out = io.StringIO()
    sweep.report(rows, "blackhole", out=out)
    text = out.getvalue()
    adddma = next(
        line for line in text.splitlines() if "ADDDMAREG" in line and "x)" in line
    )
    assert "shared" in adddma
    nop = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("NOP") and "back-pressure" in line
    )
    assert "no back-pressure" in nop


def test_the_differential_reports_the_per_instruction_delta(tmp_path):
    hardware = _control() + _phase_a("ADDDMAREG", "THCON", 194)  # 3.0 cycles
    simulated = _control() + _phase_a("ADDDMAREG", "THCON", 66)  # 1.0 cycles
    rows, _ = sweep.read_csv(_csv(tmp_path, hardware))
    reference, _ = sweep.read_csv(_csv(tmp_path, simulated))
    out = io.StringIO()
    sweep.report(rows, "blackhole", out=out, reference=reference)
    text = out.getvalue()
    assert "Differential" in text
    assert "2.000" in text  # 3.0 measured - 1.0 reference


# ---------------------------------------------------------------------------
# The entry point degrades gracefully, like the rung-2 sweep.
# ---------------------------------------------------------------------------


def test_main_without_a_csv_falls_back_to_the_tracked_reference(capsys):
    """The rung-3 comparison must reproduce with no arguments and no hardware."""
    assert sweep.main([]) == 0
    assert "Rung 3" in capsys.readouterr().out


def test_main_with_a_missing_csv_is_not_an_error(capsys, tmp_path):
    assert sweep.main(["--measured", str(tmp_path / "nope.csv")]) == 0
    assert "nothing to sweep" in capsys.readouterr().out


def test_no_tracked_dataset_for_an_arch_says_where_it_looked(capsys):
    assert sweep.main(["--arch", "wormhole"]) == 0
    out = capsys.readouterr().out
    assert "looked in" in out
    assert "perfbench/tensixbench" in out


# ---------------------------------------------------------------------------
# The tracked reference measurement.
# ---------------------------------------------------------------------------
#
# The datasets in ``tt_sim/perf/datasets/`` are the only things in this
# repository that came off silicon. These guard the two ways they could quietly
# stop being what they say they are: the provenance being separated from the
# numbers, and the numbers changing.
#
# There are two of them and they are NOT peers. The primary is the X1
# configuration (one legal SETDVALID, hoisted); the other is a deliberate
# CONTROL that reproduces the confounded per-thread setup so the artefact can be
# pointed at. Which one the sweep reads by default is therefore load-bearing,
# and is tested.


# ---------------------------------------------------------------------------
# Experiment X2: the source data format axis.
# ---------------------------------------------------------------------------


def _format_csv(tmp_path, name, fmt, costs, setup="unpacr-nop"):
    """One run's CSV at a named source format. ``costs`` is probe -> cyc/instr."""
    header = (
        "# tensixbench raw points\n"
        f"# arch=blackhole unroll=64 dvalid_setup={setup} src_format={fmt} "
        f"src_style={sweep.FORMAT_STYLE.get(fmt, 'undefined')}\n"
        "phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles\n"
    )
    rows = list(_control(per_block=0))
    for probe, cost in costs.items():
        rows += _phase_a(probe, "MATH", int(round(cost * 64)))
    path = tmp_path / name
    path.write_text(header + "".join(rows))
    return path


def _format_dataset(path):
    rows, meta = sweep.read_csv(path)
    return (str(path), rows, meta)


def test_a_format_comparison_needs_the_sanctioned_dvalid_setup(tmp_path, capsys):
    """A ``SETDVALID`` run has no source format, whatever its header says.

    This is the whole reason X2 exists as a separate setup rather than a flag on
    the old one: on Blackhole the format the Matrix Unit decodes after a bare
    ``SETDVALID`` is ``UnpredictableValue()``, so labelling such a run "fp32"
    would be a fiction, and comparing two of them would compare two fictions.
    """
    good = _format_csv(tmp_path, "a.csv", "bf16", {"MVMUL": 6.0})
    bad = _format_csv(tmp_path, "b.csv", "fp32", {"MVMUL": 6.0}, setup="once")
    values = sweep.format_report(
        [_format_dataset(good), _format_dataset(bad)], "blackhole"
    )
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "dvalid_setup" in out
    assert values == {}, "one admissible run is not an axis"


def test_two_runs_claiming_the_same_format_are_refused_not_averaged(tmp_path, capsys):
    a = _format_csv(tmp_path, "a.csv", "bf16", {"MVMUL": 6.0})
    b = _format_csv(tmp_path, "b.csv", "bf16", {"MVMUL": 9.0})
    sweep.format_report([_format_dataset(a), _format_dataset(b)], "blackhole")
    out = capsys.readouterr().out
    assert "already supplied by another run" in out


def test_two_formats_sharing_a_srcastyle_are_predicted_identical(tmp_path, capsys):
    """bf16 and fp32 are the same branch of the same decode, so a difference
    between them contradicts the functional model rather than merely being
    undocumented. The report has to say which of those two it is seeing."""
    assert sweep.FORMAT_STYLE["bf16"] == sweep.FORMAT_STYLE["fp32"]
    assert sweep.FORMAT_STYLE["tf32"] != sweep.FORMAT_STYLE["bf16"]

    a = _format_csv(tmp_path, "a.csv", "bf16", {"MVMUL": 6.0})
    b = _format_csv(tmp_path, "b.csv", "fp32", {"MVMUL": 9.0})
    sweep.format_report([_format_dataset(a), _format_dataset(b)], "blackhole")
    out = capsys.readouterr().out
    assert "CONTRADICTS the model (same SrcAStyle)" in out

    c = _format_csv(tmp_path, "c.csv", "tf32", {"MVMUL": 9.0})
    sweep.format_report([_format_dataset(a), _format_dataset(c)], "blackhole")
    out = capsys.readouterr().out
    assert "format-dependent (undocumented)" in out
    assert "CONTRADICTS" not in out


def test_a_format_difference_inside_the_resolution_is_not_a_finding(tmp_path, capsys):
    """The same one-sided-bias discipline as the per-instruction sweep. A
    difference smaller than the control over-subtraction is not evidence, and
    calling it one here would be the mistake `FIT_RESOLUTION_NOTE` exists to
    prevent, transplanted to a new axis."""
    a = _format_csv(tmp_path, "a.csv", "bf16", {"MVMUL": 6.0})
    b = _format_csv(tmp_path, "b.csv", "tf32", {"MVMUL": 6.0})
    values = sweep.format_report([_format_dataset(a), _format_dataset(b)], "blackhole")
    out = capsys.readouterr().out
    assert "no format effect this instrument can resolve" in out
    assert values[("MVMUL", "bf16")] == pytest.approx(6.0, abs=1e-6)
    assert values[("MVMUL", "tf32")] == pytest.approx(6.0, abs=1e-6)


def test_the_format_expectation_is_declared_as_exploratory(tmp_path):
    """A pre-declaration is only worth having if it says what it is. The docs
    make no per-format cost claim, so this experiment cannot confirm one; the
    text has to say so before any number is printed."""
    assert "EXPLORATORY" in sweep.FORMAT_EXPECTATION
    assert "SrcAStyle" in sweep.FORMAT_EXPECTATION
    # And it must warn that phase A is the wrong regime for the tables' number.
    assert "Wait-Gate" in sweep.FORMAT_EXPECTATION


def test_a_null_against_the_simulator_is_labelled_as_forced(tmp_path, capsys):
    """tt-sim retires one instruction per cycle whatever the format, so a null
    there is produced by the simulator's missing FIFO back-pressure and not by
    the hardware. Reading it as "no format effect" is exactly the mistake."""
    a = _format_csv(tmp_path, "a.csv", "bf16", {"MVMUL": 1.0})
    b = _format_csv(tmp_path, "b.csv", "tf32", {"MVMUL": 1.0})
    sweep.format_report([_format_dataset(a), _format_dataset(b)], "blackhole")
    out = capsys.readouterr().out
    assert "FORCED" in out
    assert "tt-sim" in out


def test_the_tracked_dataset_carries_its_own_provenance():
    """A measurement without its card, firmware and flags is not a measurement.

    They live in the CSV's own ``#`` header rather than in a sidecar file
    precisely so that they cannot be separated from it by a copy.
    """
    datasets = sweep.reference_datasets()
    assert datasets, "the tracked reference measurement has gone missing"
    for path in datasets:
        _, meta = sweep.read_csv(path)
        assert meta["arch"] in ("wormhole", "blackhole")
        assert meta["device"].endswith("-silicon"), "this directory is silicon only"
        assert meta["simulator"] == "no"
        assert meta["firmware_bundle"]
        assert meta["kmd"]
        assert meta["valid"] == "yes"
        header = path.read_text().split("phase,variant")[0]
        assert "--blocks" in header, "the run's flags are part of its identity"
        assert "ONE RUN" in header.upper()
        assert meta["dvalid_setup"] in ("once", "per-thread", sweep.FORMAT_SETUP), (
            "the dvalid setup is the difference between a result and an "
            "artefact; a dataset that does not name it cannot be read"
        )
        if meta["dvalid_setup"] == sweep.FORMAT_SETUP:
            # The UNPACR_NOP setup exists precisely to make the source format
            # defined, so a dataset taken under it that does not name the
            # format has thrown away the only thing it was for.
            assert meta.get("src_format") in sweep.FORMAT_STYLE


def test_the_default_dataset_is_the_deconfounded_one():
    """The control must never be swept by accident.

    Its MATH rows are a known artefact of an illegal per-thread ``SETDVALID``,
    and they look like a clean, repeatable, three-significant-figure
    measurement. Picking the default by glob order -- which is what the sweep
    used to do when there was only one file -- would hand a reader those
    numbers under the same heading as the real ones.
    """
    assert sweep.default_measured_path().name == sweep.PRIMARY_DATASET
    assert sweep.default_measured_path("blackhole").name == sweep.PRIMARY_DATASET
    _, meta = sweep.read_csv(sweep.default_measured_path())
    assert meta["dvalid_setup"] == "once"


def test_the_control_dataset_is_labelled_as_a_control():
    """It is tracked for what it disproves, and its header has to say so."""
    path = sweep.DATASET_DIR / "tensixbench-blackhole-dvalid-per-thread.csv"
    header = path.read_text().split("phase,variant")[0]
    assert "CONTROL" in header
    _, meta = sweep.read_csv(path)
    assert meta["dvalid_setup"] == "per-thread"


def test_the_blackhole_reference_still_says_what_it_said():
    """The headline single-thread numbers, pinned.

    Not a test of the tables -- ``attach_table`` reads those live, so a table
    edit is supposed to move the residual. This pins the *measurement*, so that
    a re-export, a re-fit or a change to the control subtraction cannot alter
    the dataset's meaning without failing here.
    """
    rows, _ = sweep.read_csv(sweep.default_measured_path("blackhole"))
    series = sweep.attach_table(sweep.apply_control(sweep.series_of(rows)), "blackhole")
    solo = {
        s["probe"]: s["measured"]
        for s in series
        if s["variant"] == "t1" and s["phase"] == "A"
    }
    assert solo["loop_overhead"] == pytest.approx(0.0)
    for probe in ("NOP", "SFPADD", "SETDMAREG", "RDCFG", "SETRWC", "INCRWC"):
        assert solo[probe] == pytest.approx(0.998, abs=5e-4), probe
    for probe in ("ADDDMAREG", "MULDMAREG", "SHIFTDMAREG", "CMPDMAREG"):
        assert solo[probe] == pytest.approx(2.973, abs=5e-4), probe
    # The Wait-Gate regime: an individually issued .ttinsn matrix op waits on
    # SrcA/SrcB AllowedClient per instruction and costs ~6, which is the
    # documented latency of 5 plus one. NOT the table's occupancy -- see
    # test_the_two_matrix_unit_regimes_are_six_fold_apart below.
    for probe in ("MVMUL", "ELWADD", "ELWMUL"):
        assert solo[probe] == pytest.approx(5.98, abs=0.02), probe


def test_the_two_matrix_unit_regimes_are_six_fold_apart():
    """The headline of this dataset, and the reason MVMUL's occupancy is still 1.

    Phase A can only issue matrix ops one ``.ttinsn`` word at a time, so it
    necessarily measures the Wait-Gate-bound regime (~6 cycles). Phase B times a
    real ``matmul_tiles``, whose MVMULs come out of a ttreplay buffer driven by
    the MOP expander, and differencing the fidelities gives the back-to-back
    marginal cost (~1.07). Both are real; the tables charge the second, because
    every non-experimental LLK path goes through the MOP.
    """
    rows, _ = sweep.read_csv(sweep.default_measured_path("blackhole"))
    series = sweep.apply_control(sweep.series_of(rows))

    wait_gate = [
        s["measured"]
        for s in series
        if s["phase"] == "A" and s["variant"] == "t1" and s["probe"] == "MVMUL"
    ]
    assert wait_gate
    assert sum(wait_gate) / len(wait_gate) == pytest.approx(5.99, abs=0.02)

    # Phase B, math thread. 16 MVMULs per fidelity phase, so LoFi/HiFi2/HiFi4
    # are 16/32/64 MVMULs of otherwise identical work.
    slopes = {
        s["variant"]: s["slope"]
        for s in series
        if s["phase"] == "B" and s["thread"] == 1
    }
    marginal = (slopes["HiFi4"] - slopes["LoFi"]) / (64 - 16)
    assert marginal == pytest.approx(1.07, abs=0.03)
    assert 5.5 < wait_gate[0] / marginal < 6.5


def test_the_fidelity_steps_land_on_the_tables_arithmetic():
    """``mvmuls_per_tile`` x 1 cycle predicts 16 and 32 cycles per step."""
    rows, _ = sweep.read_csv(sweep.default_measured_path("blackhole"))
    series = sweep.series_of(rows)
    slopes = {
        s["variant"]: s["slope"]
        for s in series
        if s["phase"] == "B" and s["thread"] == 1
    }
    assert slopes["HiFi2"] - slopes["LoFi"] == pytest.approx(17.55, abs=0.05)
    assert slopes["HiFi4"] - slopes["HiFi2"] == pytest.approx(33.65, abs=0.05)
    # The guard the design named in advance: a feeder-limited phase B has all
    # three slopes equal. These separate.
    assert slopes["HiFi4"] - slopes["LoFi"] > 40


def test_the_control_reproduces_the_artefact_it_is_kept_for():
    """The per-thread SETDVALID run, and what makes it a control rather than a
    second sample: every non-MATH probe agrees with the primary dataset, and
    the MATH probes do not -- in *both* the single- and multi-thread columns.
    """
    path = sweep.DATASET_DIR / "tensixbench-blackhole-dvalid-per-thread.csv"
    rows, _ = sweep.read_csv(path)
    series = sweep.apply_control(sweep.series_of(rows))
    by = {}
    for s in series:
        if s["phase"] == "A":
            by.setdefault((s["probe"], s["variant"]), []).append(s["measured"])
    mean = {k: sum(v) / len(v) for k, v in by.items()}

    # Unaffected: the config unit, the SFPU and the ThCon family read what the
    # primary dataset reads.
    assert mean[("RDCFG", "t1")] == pytest.approx(0.998, abs=5e-4)
    assert mean[("RDCFG", "t3")] == pytest.approx(2.950, abs=5e-3)
    assert mean[("SFPADD", "t3")] == pytest.approx(2.972, abs=5e-3)
    assert mean[("ADDDMAREG", "t1")] == pytest.approx(2.973, abs=5e-4)

    # The artefact. Note t1 too: the single-thread column moved as well, so no
    # column of this run survives for these three probes.
    for probe in ("MVMUL", "ELWADD", "ELWMUL"):
        assert mean[(probe, "t1")] == pytest.approx(0.998, abs=5e-4), probe
        assert mean[(probe, "t3")] > 12.0, probe


def test_a_superlinear_thread_scaling_is_not_called_shared():
    """The control's own verdict line, which is the point of tracking it.

    ``shared (12.1x)`` is what this used to print -- three threads getting less
    done than one, filed under the word for normal behaviour.
    """
    path = sweep.DATASET_DIR / "tensixbench-blackhole-dvalid-per-thread.csv"
    rows, _ = sweep.read_csv(path)
    out = io.StringIO()
    sweep.report(rows, "blackhole", out=out)
    line = next(
        ln
        for ln in out.getvalue().splitlines()
        if ln.strip().startswith("MVMUL") and "x)" in ln
    )
    assert "SUPERLINEAR" in line
    assert "shared" not in line


def test_the_blackhole_reference_scales_rdcfg_threefold_across_threads():
    """The measurement that reconciles RDCFG: a shared 1-IPC unit.

    ``RDCFG``'s table entry is 1-cycle *occupancy* against a ">= 2" *latency*,
    and the single-thread column cannot tell a 1-IPC unit from an issue-limited
    front end. The 1 -> 2 -> 3 thread scaling can, and does.
    """
    rows, _ = sweep.read_csv(sweep.default_measured_path("blackhole"))
    series = sweep.apply_control(sweep.series_of(rows))
    per_variant = {}
    for variant in ("t1", "t2", "t3"):
        values = [
            s["measured"]
            for s in series
            if s["probe"] == "RDCFG" and s["variant"] == variant
        ]
        per_variant[variant] = sum(values) / len(values)
    assert per_variant["t3"] / per_variant["t1"] == pytest.approx(3.0, abs=0.1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
