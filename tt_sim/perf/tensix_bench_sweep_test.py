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


def test_main_without_a_csv_explains_where_one_comes_from(capsys):
    assert sweep.main([]) == 0
    assert "perfbench/tensixbench" in capsys.readouterr().out


def test_main_with_a_missing_csv_is_not_an_error(capsys, tmp_path):
    assert sweep.main(["--measured", str(tmp_path / "nope.csv")]) == 0
    assert "nothing to sweep" in capsys.readouterr().out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
