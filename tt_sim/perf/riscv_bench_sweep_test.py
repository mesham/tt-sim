"""The RISC-V front-end sweep harness: what it predicts, and what it excludes.

Everything here is pure arithmetic on a synthetic CSV, so none of it needs
hardware, tt-metal, or a benchmark run. That is deliberate, and it is the same
split ``tensix_bench_sweep_test`` draws: the parts of this harness that can
silently go wrong are the ones turning raw points into a cost -- the control
subtraction, the per-probe unroll divisor, the exclusion ladder and the
predictions read out of the YAML -- and every one of them is checkable against a
CSV whose right answer is known by construction.

The one thing these tests cannot check is whether the *benchmark* measures what
it claims. That is what ``perfbench/riscvbench``'s own per-phase validity gate
(monotonicity, R^2, the "is the instrument live?" control) is for, and it runs
on the device.

Run:  python3 -m pytest tt_sim/perf/riscv_bench_sweep_test.py
"""

import io

import pytest

from tt_sim.perf import riscv_bench_sweep as sweep

HEADER = (
    "# riscvbench raw points\n"
    "# arch=blackhole magic=0x7B10CF01 phases=rtcqf variants=t1 base_blocks=4 "
    "pad=16 stack_addr=0xFFB00F98\n"
    "phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles\n"
)


def _csv(tmp_path, rows, header=HEADER):
    path = tmp_path / "bench.csv"
    path.write_text(header + "".join(rows))
    return path


def _slope_rows(
    probe,
    unit,
    per_block,
    phase="r",
    variant="t1",
    thread=1,
    threads=1,
    unroll=64,
    intercept=17,
):
    """Four points whose slope is exactly ``per_block`` cycles per block."""
    return [
        f"{phase},{variant},1,{probe},{unit},{threads},{thread},{n},{unroll},"
        f"{intercept + per_block * n}\n"
        for n in (4, 8, 12, 16)
    ]


def _control(per_block=2, phase="r", variant="t1", thread=1, threads=1):
    return [
        f"{phase},{variant},0,loop_overhead,-,{threads},{thread},{n},64,"
        f"{5 + per_block * n}\n"
        for n in (4, 8, 12, 16)
    ]


def _queue_rows(probe, per_instr, control=0):
    return [
        f"q,t1,27,{probe},TTQUEUE,1,1,{n},1,{control + per_instr * n}\n"
        for n in (1, 2, 4, 8, 16, 32, 64, 128)
    ]


# ---------------------------------------------------------------------------
# Reading.
# ---------------------------------------------------------------------------


def test_arch_and_stack_address_come_from_the_comment_line(tmp_path):
    _, meta = sweep.read_csv(_csv(tmp_path, _control()))
    assert meta["arch"] == "blackhole"
    # The stack's region is a tt-metal placement decision, so the benchmark
    # reports the address rather than the benchmark or the sweep assuming one.
    assert meta["stack_addr"] == "0xFFB00F98"


def test_comment_lines_are_not_data(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, _control()))
    assert len(rows) == 4
    assert all(r["probe"] == "loop_overhead" for r in rows)


# ---------------------------------------------------------------------------
# The two cancellations the whole method rests on.
# ---------------------------------------------------------------------------


def test_slope_ignores_the_intercept():
    """Kernel launch, clock reads and the barrier must cancel exactly."""
    xs = [4, 8, 12, 16]
    for intercept in (0, 17, 100_000):
        _, slope, r2 = sweep.linear_fit(xs, [intercept + 66 * x for x in xs])
        assert slope == pytest.approx(66.0)
        assert r2 == pytest.approx(1.0)


def test_control_subtraction_and_unroll_divisor(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control(per_block=2) + _slope_rows("rv_addi_dep", "RV", 66))
    )
    series = {s["probe"]: s for s in sweep.apply_control(sweep.series_of(rows))}
    # (66 - 2) / 64 == 1.000, which is the documented one instruction per cycle.
    assert series["rv_addi_dep"]["measured"] == pytest.approx(1.0)


def test_each_probe_uses_its_own_unroll(tmp_path):
    """Phase T's groups and phase F's footprints do not share the divisor.

    A single global unroll would report a 16-instruction group as costing one
    cycle, which is the kind of error the CSV carrying `unroll` per row exists
    to make impossible.
    """
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(per_block=2, phase="t")
            + _slope_rows("tt_pad", "TTINSN", 258, phase="t", unroll=16)
            + _slope_rows("f_512", "RV_FETCH", 514, phase="f", unroll=512)
            + _control(per_block=2, phase="f"),
        )
    )
    series = {s["probe"]: s for s in sweep.apply_control(sweep.series_of(rows))}
    assert series["tt_pad"]["measured"] == pytest.approx(16.0)  # (258-2)/16
    assert series["f_512"]["measured"] == pytest.approx(1.0)  # (514-2)/512


def test_the_control_is_matched_per_phase(tmp_path):
    """Every phase is its own kernel build, so it gets its own loop overhead.

    Correcting a phase F probe with a phase R control would be correcting a
    measurement with a number from a different binary.
    """
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(per_block=2, phase="r")
            + _control(per_block=9, phase="f")
            + _slope_rows("f_64", "RV_FETCH", 73, phase="f", unroll=64),
        )
    )
    series = {
        (s["phase"], s["probe"]): s for s in sweep.apply_control(sweep.series_of(rows))
    }
    assert series[("f", "f_64")]["measured"] == pytest.approx(1.0)  # (73-9)/64


# ---------------------------------------------------------------------------
# The predictions, which are READ from the YAML rather than restated here.
# ---------------------------------------------------------------------------


def test_predictions_come_from_the_tables_not_from_this_module():
    """Doubling a table field must move the prediction."""
    from tt_sim.perf import costs

    base = sweep.predictions("wormhole")["rv_div"].cycles

    real = costs.load_costs

    def doubled(arch="wormhole"):
        table = real(arch)
        riscv = table.section("riscv")
        entry = dict(riscv["integer_unit"]["divide_general"])
        entry["cycles"] = entry["cycles"] * 2
        riscv["integer_unit"] = dict(riscv["integer_unit"], divide_general=entry)
        table.sections["riscv"] = dict(riscv)
        return table

    costs.load_costs = doubled
    try:
        assert sweep.predictions("wormhole")["rv_div"].cycles == pytest.approx(base * 2)
    finally:
        costs.load_costs = real


def test_wormhole_and_blackhole_disagree_about_multiply():
    """The one probe whose prediction is genuinely per-arch.

    Wormhole's multiply blocks the integer unit for two cycles; Blackhole's
    "spend exactly one cycle in EX1, and then exactly one cycle in EX2", so it
    pipelines. Independent multiplies are therefore predicted 2 and 1, and a
    DEPENDENT chain 2 on both -- by different routes, which the derivation says.
    """
    wh = sweep.predictions("wormhole")
    bh = sweep.predictions("blackhole")
    assert wh["rv_mul_indep"].cycles == 2
    assert bh["rv_mul_indep"].cycles == 1
    assert wh["rv_mul_dep"].cycles == 2
    assert bh["rv_mul_dep"].cycles == 2
    assert bh["rv_mul_dep"].kind == "derived"
    assert "EX2" in bh["rv_mul_dep"].derivation


def test_blackhole_l1_load_uses_the_dcache_hit_row():
    """And it uses model.py's mapping, not a second copy of it."""
    from tt_sim.perf.model import _LOAD_LATENCY_KEYS, RV_REGION_L1

    pred = sweep.predictions("blackhole")["rv_load_chase"]
    assert pred.path.endswith(_LOAD_LATENCY_KEYS["blackhole"][RV_REGION_L1])
    assert pred.cycles == 2
    assert sweep.predictions("wormhole")["rv_load_chase"].cycles == 8


def test_sustained_load_throughput_is_derived_from_the_formula():
    """Wormhole: latency 8 >= 5, so four loads every 7 cycles = 1.75 each."""
    pred = sweep.predictions("wormhole")["rv_load_indep"]
    assert pred.kind == "derived"
    assert pred.cycles == pytest.approx(1.75)
    # Blackhole's d-cache hit is 2 cycles, under the threshold, so the docs'
    # "one per cycle" branch applies instead.
    assert sweep.predictions("blackhole")["rv_load_indep"].cycles == pytest.approx(1.0)


def test_the_store_pair_is_a_cross_architecture_discriminator():
    """Predicted identical on Wormhole and 5x apart on Blackhole."""
    wh = sweep.predictions("wormhole")
    bh = sweep.predictions("blackhole")
    assert wh["rv_store_spread"].cycles == wh["rv_store_coalesce"].cycles == 5
    assert bh["rv_store_spread"].cycles == 5
    assert bh["rv_store_coalesce"].cycles == 1


def test_the_fusion_prediction_is_what_makes_phase_t_a_test():
    """fuse4 and spread4 carry the same 20 instructions and differ by 3.

    That difference is the whole experiment: `riscv.ttinsn_fusion.max_fused`
    says four adjacent `.ttinsn` words become one issue slot, so the two probes
    are predicted 17 and 20. Without fusion both would be 20.
    """
    pred = sweep.predictions("blackhole", {"pad": "16"})
    assert pred["tt_pad"].cycles == 16
    assert pred["tt_fuse4"].cycles == 17
    assert pred["tt_fuse2"].cycles == 17
    assert pred["tt_spread4"].cycles == 20
    assert pred["tt_spread4"].cycles - pred["tt_fuse4"].cycles == 3


def test_the_group_prediction_follows_the_headers_pad():
    """The group size is a benchmark constant, so it travels in the CSV."""
    assert sweep.predictions("blackhole", {"pad": "8"})["tt_pad"].cycles == 8


def test_stack_probe_is_classified_by_the_simulators_own_classifier():
    """0xFFB00F98 is core-local data RAM, latency 2 on both architectures."""
    pred = sweep.predictions("wormhole", {"stack_addr": "0xFFB00F98"})["rv_load_stack"]
    assert pred.cycles == 2
    assert pred.path.endswith("core_local_data_ram")
    # An L1 stack would be a completely different row, and on Wormhole four
    # times the cost -- which is why the benchmark reports the address.
    l1 = sweep.predictions("wormhole", {"stack_addr": "0x00010000"})["rv_load_stack"]
    assert l1.cycles == 8


def test_the_undocumented_probes_are_marked_exploratory_and_carry_no_number():
    """The discipline that keeps a null from reading as a confirmation."""
    pred = sweep.predictions("blackhole")
    for probe in ("c_t", "c_xor_t", "c_xor_alt", "c_jal", "f_64", "f_2048"):
        assert pred[probe].kind == "exploratory", probe
        assert pred[probe].cycles is None, probe
        assert pred[probe].note


def test_no_prediction_is_ever_charged_from_an_unsourced_entry():
    """Every prediction traces to a table path or is exploratory."""
    for arch in ("wormhole", "blackhole"):
        for probe, pred in sweep.predictions(
            arch, {"stack_addr": "0xFFB00F98"}
        ).items():
            if pred.cycles is None:
                assert pred.kind == "exploratory", probe
            else:
                assert pred.path, probe


# ---------------------------------------------------------------------------
# The exclusion ladder. DECLARED BEFORE ANY RESIDUAL.
# ---------------------------------------------------------------------------


def test_the_ladder_drops_controls_first():
    names = [name for name, _reason, _keep in sweep._exclusions()]
    assert names[0] == "probe is a control"


def test_every_exclusion_carries_a_reason():
    for name, reason, _keep in sweep._exclusions():
        assert len(reason) > 40, name


def test_phase_q_is_excluded_from_the_slope_ladder(tmp_path):
    """It sweeps burst length looking for a knee; a slope through it is noise."""
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control()
            + _queue_rows("q_ctrl", 0, control=3)
            + _queue_rows("q_nop", 1, control=3),
        )
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    kept, ladder = sweep.retained(series)
    assert all(s["phase"] != "q" for s in kept)
    # One, not two: `q_ctrl` was already dropped a rung earlier as a control,
    # and the ladder is ordered so the cheapest, least contestable exclusion
    # goes first.
    assert dict((name, removed) for name, removed, _ in ladder)["phase == Q"] == 1


def test_contended_series_are_excluded_and_reported_separately(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control()
            + _slope_rows("rv_div", "RV", 66)
            + _control(variant="t3", threads=3)
            + _slope_rows("rv_div", "RV", 200, variant="t3", threads=3),
        )
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    kept, _ = sweep.retained(series)
    assert {s["variant"] for s in kept} == {"t1"}
    out = io.StringIO()
    sweep._issue_limit_check(series, lambda line="": print(line, file=out))
    assert "rv_div" in out.getvalue()


def test_exploratory_probes_are_excluded_from_residuals_but_still_reported(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path, _control(phase="c") + _slope_rows("c_jal", "RV_BR", 66, phase="c")
        )
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    kept, _ = sweep.retained(series)
    assert not kept
    out = io.StringIO()
    sweep._exploratory_readout(series, lambda line="": print(line, file=out))
    assert "c_jal" in out.getvalue()


# ---------------------------------------------------------------------------
# The resolution term.
# ---------------------------------------------------------------------------


def test_resolution_covers_the_control_over_subtraction(tmp_path):
    """A probe that stalls issues the loop's instructions inside the stall.

    So the unconditional control subtraction over-corrects by up to
    slope(control)/unroll, and a residual smaller than that is inside the
    instrument rather than a finding. This is not hypothetical: against tt-sim
    with the cost model on, `rv_store_spread`'s raw slope is exactly 320 per
    block of 64 stores -- 5.000 each with no room for the loop -- and reports
    4.969.
    """
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(per_block=2) + _slope_rows("rv_store_spread", "RV_LSU", 320),
        )
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    store = next(s for s in series if s["probe"] == "rv_store_spread")
    assert store["measured"] == pytest.approx(4.96875)
    assert store["resolution"] == pytest.approx(2 / 64)
    # The shortfall is EXACTLY the resolution, which is the point: the whole
    # 0.031 is the control subtraction and none of it is a claim about the
    # device, so the series must not be marked as an over-charge.
    store["residual"] = store["measured"] - store["predicted"]
    assert not store["residual"] < -(store["resolution"])


def test_a_real_over_charge_is_beyond_the_resolution(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(per_block=2) + _slope_rows("rv_store_spread", "RV_LSU", 66),
        )
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    kept, _ = sweep.retained(series)
    for s in kept:
        s["residual"] = s["measured"] - s["predicted"]
        s["resolved"] = s["residual"] < -(s["resolution"] or 0.0)
    assert all(s["resolved"] for s in kept)
    out = io.StringIO()
    sweep._floor_verdict(kept, lambda line="": print(line, file=out))
    assert "VERDICT: NO" in out.getvalue()


# ---------------------------------------------------------------------------
# The phase read-outs, each of which is a difference rather than a level.
# ---------------------------------------------------------------------------


def _fusion_series(tmp_path, fuse4_per_group, spread4_per_group):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(per_block=2, phase="t")
            + _slope_rows("tt_pad", "TTINSN", 2 + 16 * 16, phase="t", unroll=16)
            + _slope_rows(
                "tt_fuse4", "TTINSN", 2 + 16 * fuse4_per_group, phase="t", unroll=16
            )
            + _slope_rows(
                "tt_spread4", "TTINSN", 2 + 16 * spread4_per_group, phase="t", unroll=16
            ),
        )
    )
    return sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )


def test_fusion_verdict_when_the_cache_fuses(tmp_path):
    out = io.StringIO()
    sweep._fusion_check(
        _fusion_series(tmp_path, 17, 20),
        lambda line="": print(line, file=out),
        {"pad": "16"},
    )
    text = out.getvalue()
    assert "+3.000" in text
    assert "FUSES" in text


def test_fusion_verdict_when_it_does_not(tmp_path):
    out = io.StringIO()
    sweep._fusion_check(
        _fusion_series(tmp_path, 20, 20),
        lambda line="": print(line, file=out),
        {"pad": "16"},
    )
    text = out.getvalue()
    assert "no fusion" in text
    # And it must say that a simulator null is forced rather than informative,
    # which is the mistake tensixbench's first reading made.
    assert "FORCED" in text


def test_branch_check_reports_the_direction_delta(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(phase="c")
            + _slope_rows("c_nt", "RV_BR", 66, phase="c")
            + _slope_rows("c_t", "RV_BR", 66 + 64 * 4, phase="c"),
        )
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    out = io.StringIO()
    sweep._branch_check(series, "blackhole", lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "+4.000" in text
    assert "branch direction costs cycles" in text


def test_branch_check_calls_a_simulator_null_forced(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _control(phase="c")
            + _slope_rows("c_nt", "RV_BR", 66, phase="c")
            + _slope_rows("c_t", "RV_BR", 66, phase="c"),
        )
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    out = io.StringIO()
    sweep._branch_check(series, "blackhole", lambda line="": print(line, file=out))
    assert "forced" in out.getvalue()


# ---------------------------------------------------------------------------
# Phase Q. The read-out these pin is the SECOND one: the point-by-point
# `q_ctrl` subtraction was retracted on 2026-08-05 after the first silicon run
# (see `sweep._queue_check`), so what a test may assume changed with it. The
# helpers below build bursts whose RAW cycles carry a fixed cost, because that
# is what the new construction is built to cancel and the old one was not.
# ---------------------------------------------------------------------------


def _burst(probe, per_instr, fixed=0, threads=1, thread=1, probe_id=30):
    """A burst whose raw cycles are ``fixed + per_instr * n``."""
    return [
        f"q,t{threads},{probe_id},{probe},TTQUEUE,{threads},{thread},{n},1,"
        f"{fixed + per_instr * n}\n"
        for n in (1, 2, 4, 8, 16, 32, 64, 128)
    ]


def _wobbly_control(values, threads=1, thread=1):
    """`q_ctrl`, deliberately non-monotone -- which is what silicon gave."""
    return [
        f"q,t{threads},27,q_ctrl,TTQUEUE,{threads},{thread},{n},1,{v}\n"
        for n, v in zip((1, 2, 4, 8, 16, 32, 64, 128), values)
    ]


def test_the_queue_rate_is_immune_to_a_fixed_cost(tmp_path):
    """The property the whole retraction turns on.

    The cascade evaluates all seven of its `if` tests at every burst length, so
    its cost is CONSTANT in n. A rate taken as a difference of two raw points
    therefore cannot see it -- which is why subtracting a noisy `q_ctrl` was
    strictly worse than subtracting nothing.
    """
    for fixed in (0, 19, 100_000):
        rows, _ = sweep.read_csv(
            _csv(
                tmp_path,
                _wobbly_control([6, 11, 20, 13, 21, 23, 14, 19])
                + _burst("q_adddmareg", 3, fixed=fixed),
            )
        )
        points = {r["n"]: r["cycles"] for r in rows if r["probe"] == "q_adddmareg"}
        assert sweep._wide_rate(points)[0] == pytest.approx(3.0)


def test_the_queue_read_out_never_subtracts_the_control(tmp_path):
    """`q_ctrl` is printed as a noise floor and is not arithmetic on anything.

    The regression this pins is the one the silicon run found: subtracting a
    6-to-23-cycle control from a burst whose whole signal is 1-4 cycles at small
    `n` produced NEGATIVE costs, which is not a queue depth, it is an artefact.
    """
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _wobbly_control([6, 11, 20, 13, 21, 23, 14, 19])
            + _burst("q_adddmareg", 1, fixed=6)
            + _burst("q_adddmareg_sync", 3, fixed=20, probe_id=31),
        )
    )
    out = io.StringIO()
    sweep._queue_check(rows, lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "NOISE FLOOR" in text
    assert "6-23 cycles" in text
    # The control's own row is reproduced verbatim, labelled, and unused.
    assert "(noise)" in text
    assert "-" not in text.split("q_ctrl")[1].split("\n")[0]


def test_queue_check_finds_the_knee_against_the_measured_saturated_rate(tmp_path):
    """Cheap below the depth, the backend's occupancy above it.

    The saturated rate comes from `q_adddmareg_sync` -- the same burst with the
    drain inside the timed region -- rather than from the documented occupancy,
    so the knee is located by measurement on both sides.
    """
    plain = {1: 1, 2: 2, 4: 4, 8: 8, 16: 16, 32: 32, 64: 64 + 96, 128: 128 + 288}
    rows = _wobbly_control([3] * 8)
    rows += [
        f"q,t1,30,q_adddmareg,TTQUEUE,1,1,{n},1,{5 + v}\n" for n, v in plain.items()
    ]
    rows += _burst("q_adddmareg_sync", 3, fixed=20, probe_id=31)
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._queue_check(parsed, lambda line="": print(line, file=out))
    assert "KNEE between n=32 and n=64" in out.getvalue()


def test_queue_check_says_so_when_the_core_never_stops_running_ahead(tmp_path):
    """One cycle per instruction out to the longest burst, against a backend
    doing three -- so the queue absorbed everything and the sweep is too short
    to have found its depth. That is a result and must not print as a depth."""
    rows = _wobbly_control([3] * 8)
    rows += _burst("q_adddmareg", 1, fixed=5)
    rows += _burst("q_adddmareg_sync", 3, fixed=20, probe_id=31)
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._queue_check(parsed, lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "NO KNEE up to n=128" in text
    assert "beyond 128" in text


def test_queue_check_ignores_the_slope_phases_control(tmp_path):
    """`loop_overhead` is emitted alongside every phase, including Q.

    It sweeps block count, not burst length, and has nothing to do with a
    queue; reporting it under a knee heading would be nonsense with a number
    attached.
    """
    rows = _wobbly_control([3] * 8) + _burst("q_nop", 1) + _control(phase="q")
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._queue_check(parsed, lambda line="": print(line, file=out))
    assert "loop_overhead" not in out.getvalue()


def test_queue_check_hunts_a_knee_only_where_one_could_exist(tmp_path):
    """`q_nop` and `q_setdmareg` have a backend occupancy at or below the issue
    rate, so "issue-limited" and "back-pressured" are the same number for them
    and no knee exists to find. The first read-out announced one for `q_nop`
    off two noisy points; this pins that it cannot again."""
    rows = _wobbly_control([6, 11, 20, 13, 21, 23, 14, 19])
    rows += _burst("q_nop", 1, fixed=6)
    rows += _burst("q_setdmareg", 1, fixed=6, probe_id=29)
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._queue_check(parsed, lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "q_nop" in text  # printed...
    assert "KNEE between" not in text  # ...but never adjudicated
    assert "NO KNEE up to" not in text
    assert "no knee exists to find" in text


def test_queue_check_reads_the_in_flight_work_off_the_sync_pair(tmp_path):
    """The queue's occupancy, read directly rather than inferred from a knee.

    `tensix_sync()`'s own cost is removed by taking the pair's difference at the
    smallest burst, so what is left is work rather than the call.
    """
    rows = _wobbly_control([3] * 8)
    rows += _burst("q_adddmareg", 1, fixed=0)
    rows += _burst("q_adddmareg_sync", 4, fixed=9, probe_id=31)
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._queue_check(parsed, lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "in flight" in text
    # (9 + 4*128) - 128 = 393 cycles the burst did not wait for, less the
    # 12-cycle `tensix_sync()` baseline the n=1 pair measures.
    assert "n=128 +381" in text


def test_queue_check_reports_every_thread_count(tmp_path):
    """The thread count is where phase Q's answer actually lives: whether the
    core outruns the backend depends on how many cores are feeding it, so a
    read-out that showed only t1 would hide the transition."""
    rows = _wobbly_control([3] * 8)
    rows += _burst("q_adddmareg", 1, fixed=5)
    rows += _burst("q_adddmareg_sync", 3, fixed=20, probe_id=31)
    rows += _wobbly_control([3] * 8, threads=3, thread=2)
    rows += _burst("q_adddmareg", 9, fixed=5, threads=3, thread=2)
    rows += _burst("q_adddmareg_sync", 9, fixed=20, threads=3, thread=2, probe_id=31)
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._queue_check(parsed, lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "1 issuing thread(s), thread 1" in text
    assert "3 issuing thread(s), thread 2" in text


def test_fetch_check_reports_flatness_and_a_cliff(tmp_path):
    def series(costs):
        rows = _control(phase="f")
        for k, c in costs.items():
            rows += _slope_rows(
                f"f_{k}", "RV_FETCH", 2 + int(k * c), phase="f", unroll=k
            )
        parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
        return sweep.attach_predictions(
            sweep.apply_control(sweep.series_of(parsed)), "blackhole"
        )

    out = io.StringIO()
    sweep._fetch_check(series({64: 1, 2048: 1}), lambda line="": print(line, file=out))
    assert "flat" in out.getvalue()

    out = io.StringIO()
    sweep._fetch_check(series({64: 1, 2048: 3}), lambda line="": print(line, file=out))
    assert "NOT flat" in out.getvalue()


# ---------------------------------------------------------------------------
# The live check -- the thing that tells a null from an un-instrumented run.
# ---------------------------------------------------------------------------


def _live_series(tmp_path, values):
    rows = _control(per_block=2)
    for probe, per in values.items():
        rows += _slope_rows(probe, "RV", 2 + 64 * per)
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    return sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(parsed)), "blackhole"
    )


def test_live_check_fails_a_run_where_everything_reads_one(tmp_path):
    out = io.StringIO()
    sweep._live_check(
        _live_series(
            tmp_path,
            dict.fromkeys(
                ("rv_mul_dep", "rv_div", "rv_load_chase", "rv_store_spread"), 1
            ),
        ),
        lambda line="": print(line, file=out),
    )
    text = out.getvalue()
    assert "measured nothing" in text
    assert "unsafe until this one passes" in text


def test_live_check_passes_when_the_instrumented_probes_move(tmp_path):
    out = io.StringIO()
    sweep._live_check(
        _live_series(
            tmp_path,
            {"rv_mul_dep": 1, "rv_div": 6, "rv_load_chase": 2, "rv_store_spread": 5},
        ),
        lambda line="": print(line, file=out),
    )
    assert "3 of 4 read above 1.0" in out.getvalue()


# ---------------------------------------------------------------------------
# End to end.
# ---------------------------------------------------------------------------


def test_report_runs_and_names_its_exclusions(tmp_path):
    rows, meta = sweep.read_csv(
        _csv(
            tmp_path,
            _control(per_block=2)
            + _slope_rows("rv_addi_dep", "RV", 66)
            + _slope_rows("rv_div", "RV", 2 + 64 * 6)
            + _control(per_block=2, phase="c")
            + _slope_rows("c_jal", "RV_BR", 66, phase="c"),
        )
    )
    out = io.StringIO()
    kept = sweep.report(rows, "blackhole", out=out, meta=meta)
    text = out.getvalue()
    assert "Exclusion ladder (declared before any residual was computed)" in text
    assert "no prediction in the tables" in text
    assert {s["probe"] for s in kept} == {"rv_addi_dep", "rv_div"}


def test_main_reads_the_primary_tracked_dataset(capsys):
    """No arguments must sweep the silicon run, so the rung reproduces without
    hardware. Until 2026-08-05 there was no dataset at all and this test pinned
    the graceful degradation instead; the file arriving is what changed."""
    assert sweep.main([]) == 0
    out = capsys.readouterr().out
    assert sweep.PRIMARY_DATASET in out
    assert "Is the instrument live?" in out


def test_the_failed_run_is_never_chosen_for_you():
    """`MIN_BLOCKS_DATASET` is a run every phase of which the benchmark refused.

    It is tracked because that refusal is the measurement -- it puts a floor
    under `--blocks` -- and it must never be swept as if it were a result. Both
    files exist, so "whatever sorts first" would pick the wrong one.
    """
    tracked = {p.name for p in sweep.reference_datasets()}
    assert {sweep.PRIMARY_DATASET, sweep.MIN_BLOCKS_DATASET} <= tracked
    assert sweep.default_measured_path().name == sweep.PRIMARY_DATASET


def test_the_tracked_datasets_carry_their_own_provenance():
    """A measurement separated from its card, firmware and flags is not one."""
    for path in sweep.reference_datasets():
        head = path.read_text()[:6000]
        assert "device=blackhole-silicon" in head, path.name
        assert "firmware_bundle=" in head and "kmd=" in head, path.name
        assert "ONE RUN, ON ONE CARD" in head, path.name
        assert "valid:" in head, path.name


def test_differential_diffs_two_runs_of_the_same_binary(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _ = sweep.read_csv(
        _csv(
            tmp_path / "a",
            _control() + _slope_rows("rv_load_chase", "RV_LSU", 2 + 64 * 8),
        )
    )
    b, _ = sweep.read_csv(
        _csv(
            tmp_path / "b",
            _control() + _slope_rows("rv_load_chase", "RV_LSU", 2 + 64 * 1),
        )
    )
    out = io.StringIO()
    sweep._differential(a, b, "blackhole", lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "rv_load_chase" in text
    assert "7.000" in text
