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
        # `doubled` mutates the CACHED table in place, so restoring the
        # function is not enough -- every later test in the process would read
        # a doubled divide. Drop the entry and let it reload from the YAML.
        costs._CACHE.pop("wormhole", None)


def test_the_divide_floor_is_kept_and_says_what_the_range_denotes():
    """Reviewed on 2026-08-05 against silicon's 33.001 and deliberately kept.

    6-33 is a DATA dependence -- "dependent upon the magnitude of the dividend"
    -- not an uncertainty band, and the function relating the two is published
    nowhere. So the low end stands, as it does for every other bound in these
    files, and the note carries what the range means so that the 5.5x residual
    is read as the benchmark's choice of dividend rather than as a table that
    is wrong. If someone charges the max, or interpolates, this test is where
    they have to argue for it.
    """
    for arch in ("wormhole", "blackhole"):
        pred = sweep.predictions(arch)["rv_div"]
        assert pred.cycles == 6
        assert pred.bound == "range"
        assert "magnitude of the dividend" in pred.note


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


def test_a_pointer_chase_is_predicted_against_the_dcache_MISS_row():
    """The defect the silicon run exposed, and the rule that replaced it.

    Until 2026-08-05 this probe was predicted off ``_LOAD_LATENCY_KEYS``, which
    on Blackhole names ``l1_dcache_hit`` -- so a pointer chase over a 1 KiB ring
    was compared against a 2-cycle hit and read out as a 6-cycle discrepancy
    that was entirely the sweep's own. The chase's working set is sixteen times
    the L0 data cache's published 64-byte capacity, so the miss row is the row
    it reaches under any organisation of the cache. That is a documentary fact
    (``bh_riscv#l0-data-cache``), which is why it may move a prediction.
    """
    from tt_sim.perf.model import _LOAD_LATENCY_KEYS, RV_REGION_L1

    pred = sweep.predictions("blackhole")["rv_load_chase"]
    assert pred.path == "riscv.load_latency.l1_dcache_miss"
    assert pred.cycles == 8
    assert pred.bound == "at_least"
    # ...and it is NOT the row the simulator charges, which is the whole point:
    # the two answer different questions.
    assert _LOAD_LATENCY_KEYS["blackhole"][RV_REGION_L1] == "l1_dcache_hit"
    # Wormhole publishes no L0 cache and so has a single L1 row; nothing about
    # the working set can change which row applies there.
    wh = sweep.predictions("wormhole")["rv_load_chase"]
    assert wh.path == "riscv.load_latency.l1"
    assert wh.cycles == 8


def test_the_row_is_chosen_by_working_set_against_the_published_capacity():
    """Not hardcoded per probe: the capacity comes out of the YAML.

    Below the capacity the table's default row stands; above it, the miss row.
    Both directions are exercised so that a future probe with a small working
    set is predicted against the row it would actually reach.
    """
    from tt_sim.perf.costs import load_costs

    riscv = load_costs("blackhole").section("riscv")
    capacity = riscv["l0_data_cache"]["capacity_bytes"]
    assert capacity == 64
    assert sweep._l1_load_row(riscv, "blackhole", capacity // 2)[0] == "l1_dcache_hit"
    assert sweep._l1_load_row(riscv, "blackhole", capacity * 16)[0] == "l1_dcache_miss"
    # No working set at all -- an unknown access pattern -- keeps the default.
    assert sweep._l1_load_row(riscv, "blackhole", None)[0] == "l1_dcache_hit"
    # And the probes' declared working sets are the kernel's, not invented.
    assert sweep.L1_WORKING_SET_BYTES["rv_load_chase"] == 64 * 16
    assert sweep.L1_WORKING_SET_BYTES["rv_load_indep"] == 4 * 16


def test_a_working_set_exactly_at_the_capacity_is_not_moved():
    """The boundary the documentation does not settle, and must not be fitted.

    ``rv_load_indep`` touches exactly four 16-byte lines -- exactly the
    published capacity. Silicon reads the MISS row's sustained rate (1.742
    against the formula's 1.750 at N = 8), but the page publishes no
    associativity, no replacement policy and a ~0.8 % periodic flush, so
    nothing in it says a working set at the capacity misses. Moving the
    prediction to match the measurement would be fitting the table to the
    reading. The residual is left standing and the derivation says why.
    """
    pred = sweep.predictions("blackhole")["rv_load_indep"]
    assert pred.cycles == pytest.approx(1.0)
    assert "EXACTLY the" in pred.derivation
    assert "l1_dcache_hit" in pred.derivation


def test_sustained_load_throughput_is_derived_from_the_formula():
    """Wormhole: latency 8 >= 5, so four loads every 7 cycles = 1.75 each."""
    pred = sweep.predictions("wormhole")["rv_load_indep"]
    assert pred.kind == "derived"
    assert pred.cycles == pytest.approx(1.75)


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


# ---------------------------------------------------------------------------
# Phase Q's LOOP form, and the refusal that governs it. The read-out may print
# a depth in entries only when the backlog has both cleared the control's own
# measured spread and stopped growing; the 2026-08-05 silicon run announced
# "~1 instruction in flight" from an 8-cycle backlog and three NEGATIVE ones,
# which is arithmetic run where there is no signal.
# ---------------------------------------------------------------------------

_LOOP_NS = (16, 32, 64, 128, 256, 512, 1024)


def _loop_burst(probe, cycles, threads=1, thread=1, probe_id=39):
    return [
        f"q,t{threads},{probe_id},{probe},TTQUEUE,{threads},{thread},{n},1,{cycles[n]}\n"
        for n in _LOOP_NS
    ]


def _loop_slot(backlog, rate=3, control=(6, 11, 20, 13, 21, 22, 13, 17)):
    """A loop-form slot whose plain burst lags its drained one by `backlog(n)`.

    The drained probe costs `rate` cycles an instruction throughout; the plain
    one returns `backlog(n)` cycles early, which is the work still in flight.
    The read-out subtracts the pair's difference at the smallest burst -- that
    is `tensix_sync()`'s own cost -- so `backlog(16)` is the zero of the scale.
    """
    plain = {n: 100 + rate * n - backlog(n) for n in _LOOP_NS}
    synced = {n: 100 + rate * n + 40 for n in _LOOP_NS}
    return (
        _wobbly_control(list(control))
        + _loop_burst("q_loop_adddmareg", plain)
        + _loop_burst("q_loop_adddmareg_sync", synced, probe_id=40)
    )


def _settling(n):
    """A backlog that grows with the burst and then stops -- a real asymptote."""
    return 0 if n == 16 else min(n, 150)


def _forty(n):
    """Settled at 40 cycles: above a quiet run's noise floor, under a noisy
    one's. Which of the two it is, is the thing under test."""
    return 0 if n == 16 else 40


def _loop_text(tmp_path, rows):
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._queue_check(parsed, lambda line="": print(line, file=out))
    return out.getvalue()


def test_loop_form_reports_a_depth_once_the_backlog_settles(tmp_path):
    """A backlog that stops growing well clear of the noise floor is the one
    case a depth in entries may be divided out of. 150 cycles at 3.0 each."""
    text = _loop_text(tmp_path, _loop_slot(_settling))
    assert "BACKLOG FLATTENED at ~150 cycles" in text
    assert "~50 INSTRUCTIONS in flight" in text


def test_loop_form_refuses_a_negative_backlog(tmp_path):
    """Work in flight cannot be negative. Silicon produced -50, -10 and -45 at
    three issuing threads and the read-out divided one of them by a rate."""
    text = _loop_text(tmp_path, _loop_slot(lambda n: 0 if n == 16 else -20))
    assert "BACKLOG NEGATIVE" in text
    assert "INSTRUCTIONS in flight" not in text
    assert "NO DEPTH IN ENTRIES IS REPORTED" in text


def test_loop_form_refuses_a_backlog_inside_the_noise_floor(tmp_path):
    """The failure this was built for: +8 cycles, printed as "~1 instruction".

    The backlog is a difference of two differences of raw single-shot points,
    so two `q_ctrl` spreads is the least it has to clear -- and the floor is
    taken from the run's own measured spread, not from a constant.
    """
    text = _loop_text(tmp_path, _loop_slot(lambda n: 0 if n == 16 else 8))
    assert "INSIDE THE NOISE FLOOR (32 = two" in text
    assert "INSTRUCTIONS in flight" not in text


def test_the_noise_floor_scales_with_the_measured_control(tmp_path):
    """Same backlog, two runs whose controls scatter differently: the quiet one
    resolves it and the noisy one refuses. A constant threshold could not."""
    quiet = _loop_text(tmp_path, _loop_slot(_forty, control=(6, 7, 8, 9, 10, 11, 8, 7)))
    noisy = _loop_text(
        tmp_path, _loop_slot(_forty, control=(6, 40, 20, 13, 21, 22, 13, 9))
    )
    assert "~13 INSTRUCTIONS in flight" in quiet
    assert "INSIDE THE NOISE FLOOR" in noisy


def test_loop_form_refuses_a_backlog_that_is_still_growing(tmp_path):
    """An unbounded queue absorbs the whole of every doubling, so its backlog
    doubles too -- which is what tt-sim does and what a depth must not be read
    off. It clears the noise floor and is still not an asymptote."""
    text = _loop_text(tmp_path, _loop_slot(lambda n: n - 16))
    assert "BACKLOG STILL GROWING" in text
    assert "INSTRUCTIONS in flight" not in text


def test_loop_form_measures_whether_the_burst_form_changed_the_quantity(tmp_path):
    """The cascade and the loop overlap at n = 16..128, and that overlap is
    what licenses reading the loop's longer bursts as the same quantity."""
    rows = _loop_slot(_settling)
    rows += _burst("q_adddmareg", 3, fixed=7)
    text = _loop_text(tmp_path, rows)
    assert "form check n=16..128" in text
    assert "cascade 3.000" in text


# ---------------------------------------------------------------------------
# Phase S -- is the queue phase Q measured SHARED between the TRISCs or private
# to each? Everything below is synthetic, and the point of the synthesis is
# that the right answer is known: a slot is built to hold `depth` entries at
# saturation, and the read-out has to recover that number.
#
# THE ARITHMETIC BEING PINNED, because it is the one thing here that could be
# quietly wrong. A core pushing at 1/p instructions per cycle against a backend
# draining one every S leaves `n * (1 - p/S)` outstanding, so the reference
# burst is NOT empty and the depth is
#
#     D = backlog / S + n_ref * (1 - p/S)
#
# Phase Q's read-out drops that second term, which is why it reads a lower
# bound. Making n_ref small (4, not 16) is what makes the term small.
# ---------------------------------------------------------------------------

_SHARE_NS = (4, 8, 16, 32, 64, 128, 256, 512)
_SHARE_SLOTS = (
    ("s_loop_addi", 41),
    ("s_co_plain", 42),
    ("s_co_repeat", 43),
    ("s_co_sync", 44),
    ("s_solo_plain", 45),
    ("s_solo_sync", 46),
)


def _share_slot(
    depth,
    service,
    issue=1.5,
    threads=1,
    thread=1,
    solo_depth=None,
    noise=0,
    variant=None,
):
    """A phase-S slot whose queue holds `depth` entries at `service` cyc/instr."""
    variant = variant or f"t{threads}"
    solo_depth = depth if solo_depth is None else solo_depth

    def outstanding(d, n):
        return min(d, int(n * (1 - issue / service)))

    values = {
        "s_loop_addi": {n: 500 + int(n * issue) for n in _SHARE_NS},
        "s_co_sync": {n: 1000 + service * n for n in _SHARE_NS},
        "s_solo_sync": {n: 900 + service * n for n in _SHARE_NS},
    }
    values["s_co_plain"] = {
        n: values["s_co_sync"][n] - service * outstanding(depth, n) for n in _SHARE_NS
    }
    values["s_co_repeat"] = {
        n: values["s_co_plain"][n] + (noise if n == _SHARE_NS[-1] else 0)
        for n in _SHARE_NS
    }
    values["s_solo_plain"] = {
        n: values["s_solo_sync"][n] - service * outstanding(solo_depth, n)
        for n in _SHARE_NS
    }
    rows = []
    for probe, probe_id in _SHARE_SLOTS:
        rows += [
            f"s,{variant},{probe_id},{probe},TTQUEUE,{threads},{thread},{n},1,"
            f"{values[probe][n]}\n"
            for n in _SHARE_NS
        ]
    return rows


def _share_text(tmp_path, rows):
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._sharing_check(parsed, lambda line="": print(line, file=out))
    return out.getvalue()


def test_phase_s_is_excluded_from_the_slope_ladder(tmp_path):
    """It sweeps burst length, and its answer is a ratio rather than a level."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _control() + _share_slot(depth=24, service=3))
    )
    series = sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(rows)), "blackhole"
    )
    kept, ladder = sweep.retained(series)
    assert all(s["phase"] != "s" for s in kept)
    assert dict((name, removed) for name, removed, _ in ladder)["phase == S"] == 6


def test_the_depth_includes_the_reference_bursts_own_occupancy(tmp_path):
    """The correction phase Q's read-out drops, pinned against a known answer.

    The slot below holds 24 entries by construction. Its backlog at the longest
    burst is (24 - 2) * 3 = 66 cycles, and 66/3 alone would report 22. The
    missing 2 are what the n=4 reference burst is itself holding.
    """
    text = _share_text(tmp_path, _share_slot(depth=24, service=3))
    assert "backlog +66 cycles" in text
    assert "DEPTH ~24 ENTRIES" in text


def test_a_queue_that_does_not_shrink_with_more_issuers_is_per_thread(tmp_path):
    text = _share_text(
        tmp_path,
        _share_slot(depth=24, service=3, threads=1)
        + _share_slot(depth=24, service=6, threads=2),
    )
    assert "1.00x (per-thread predicts 1.00x" in text
    assert "PER-THREAD -- each core has its own queue" in text


def test_a_queue_that_halves_with_two_issuers_is_shared(tmp_path):
    """The discriminator. Both slots are saturated and both service rates are
    measured in their own slot, so the only thing left to explain a halved
    depth is that the second issuer is holding half the entries."""
    text = _share_text(
        tmp_path,
        _share_slot(depth=24, service=3, threads=1)
        + _share_slot(depth=12, service=6, threads=2),
    )
    assert "t2: 12 entries against t1's 24 = 0.50x" in text
    assert "shared predicts 0.50x" in text
    assert "SHARED -- one queue, split between the issuers" in text


def test_the_spinning_thread_is_a_control_and_is_reported_as_one(tmp_path):
    """A spinning thread pushes nothing, so it holds no entry under EITHER
    hypothesis and its depth must not move. That is why it cannot be the
    discriminator, and why it is worth running: a departure here is instruction
    fetch or something else, not queue sharing."""
    text = _share_text(
        tmp_path,
        _share_slot(depth=24, service=3, threads=1)
        + _share_slot(depth=12, service=6, threads=2, solo_depth=24),
    )
    assert "SHARED" in text
    assert "control: with the others only SPINNING" in text
    assert "Both hypotheses predict 1.00x here" in text


def test_no_verdict_without_a_second_thread_count(tmp_path):
    """One thread count is a level, and a level is not this phase's answer."""
    text = _share_text(tmp_path, _share_slot(depth=24, service=3))
    assert "NO VERDICT: only one thread count" in text
    assert "PER-THREAD" not in text.split("NO VERDICT")[1]


def test_an_unbounded_queue_resolves_nothing_and_the_read_out_says_which(tmp_path):
    """tt-sim's forced answer: `push_mop_instruction` is a list append, so the
    backlog doubles with every doubling of the burst and never settles. No
    depth may be divided out of that, at any thread count."""
    growing = _share_slot(depth=10**6, service=3, threads=1)
    text = _share_text(tmp_path, growing)
    assert "still growing at the longest burst" in text
    assert "DEPTH ~" not in text
    assert "NO VERDICT: the single-thread slot resolved no depth" in text
    assert "unbounded list append" in text


def test_the_noise_floor_is_measured_by_a_byte_identical_repeat(tmp_path):
    """`s_co_repeat` runs `s_co_plain` a second time, so the disagreement
    between them is what ONE raw point can be wrong by -- measured in the run
    rather than inherited from a control that runs a different body."""
    quiet = _share_text(tmp_path, _share_slot(depth=24, service=3, noise=1))
    noisy = _share_text(tmp_path, _share_slot(depth=24, service=3, noise=40))
    assert "DEPTH ~24 ENTRIES" in quiet
    assert "inside 2x this slot's measured repeatability" in noisy
    assert "DEPTH ~" not in noisy


def _footprint_series(tmp_path, costs):
    """`costs` maps a probe name (`f_64`, `g_1280`, ...) to cycles/instruction."""
    rows = []
    for phase in sorted({nm[0] for nm in costs}):
        rows += _control(phase=phase)
    for nm, c in costs.items():
        k = int(nm.split("_")[1])
        rows += _slope_rows(nm, "RV_FETCH", 2 + int(k * c), phase=nm[0], unroll=k)
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    return sweep.attach_predictions(
        sweep.apply_control(sweep.series_of(parsed)), "blackhole"
    )


def test_fetch_check_reports_flatness_and_a_cliff(tmp_path):
    out = io.StringIO()
    sweep._fetch_check(
        _footprint_series(tmp_path, {"f_64": 1, "f_2048": 1}),
        lambda line="": print(line, file=out),
    )
    assert "flat" in out.getvalue()

    out = io.StringIO()
    sweep._fetch_check(
        _footprint_series(tmp_path, {"f_64": 1, "f_2048": 3}),
        lambda line="": print(line, file=out),
    )
    assert "NOT flat" in out.getvalue()


# ---------------------------------------------------------------------------
# Reconciling the two burst forms. Phase Q and phase S both resolve a Tensix
# instruction queue depth at one issuing thread, from different reference
# bursts and different loop blocks, and the numbers they PRINT are therefore
# not comparable: phase Q's read-out drops the reference burst's own occupancy
# and phase S's carries it. The reconciliation recomputes both under one
# estimator and says what is left -- and on 2026-08-05 silicon what was left
# was ~5 entries that the correction does not explain.
# ---------------------------------------------------------------------------

_RECONCILE_FIXED = 12  # the timed region's own cost, in both forms


def _one_form(names, ns, n_ref, depth, service, issue):
    """One burst form whose queue holds exactly `depth` entries.

    Both probes carry the same fixed cost as the issue-limited baseline, which
    is what silicon does -- they are the same macro with a different body -- and
    is what lets the run-ahead estimator read the depth back undivided.
    """
    plain_name, sync_name, base_name = names
    outstanding = {n: min(depth, n * (1 - issue / service)) for n in ns}
    values = {
        base_name: {n: _RECONCILE_FIXED + issue * n for n in ns},
        sync_name: {n: _RECONCILE_FIXED + service * n + 9 for n in ns},
    }
    values[plain_name] = {
        n: _RECONCILE_FIXED + service * (n - outstanding[n]) for n in ns
    }
    phase = "q" if n_ref == sweep.QUEUE_MIN_N else "s"
    return [
        f"{phase},t1,{50 + i},{probe},TTQUEUE,1,1,{n},1,{values[probe][n]:.0f}\n"
        for i, probe in enumerate(values)
        for n in ns
    ]


def _reconcile_text(tmp_path, q_depth, s_depth, service=3):
    rows = _one_form(
        (sweep.QUEUE_LOOP_PLAIN, sweep.QUEUE_LOOP_SYNC, sweep.QUEUE_LOOP_BASELINE),
        _LOOP_NS,
        sweep.QUEUE_MIN_N,
        q_depth,
        service,
        issue=1.125,
    ) + _one_form(
        (sweep.SHARE_CO_PLAIN, sweep.SHARE_CO_SYNC, sweep.SHARE_BASELINE),
        _SHARE_NS,
        sweep.SHARE_MIN_N,
        s_depth,
        service,
        issue=1.5,
    )
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._depth_reconcile(parsed, lambda line="": print(line, file=out))
    return out.getvalue()


def test_one_form_alone_produces_no_reconciliation(tmp_path):
    """The comparison is the whole output, so half of it is not half an answer."""
    rows = _one_form(
        (sweep.QUEUE_LOOP_PLAIN, sweep.QUEUE_LOOP_SYNC, sweep.QUEUE_LOOP_BASELINE),
        _LOOP_NS,
        sweep.QUEUE_MIN_N,
        24,
        3,
        issue=1.125,
    )
    parsed, _ = sweep.read_csv(_csv(tmp_path, rows))
    out = io.StringIO()
    sweep._depth_reconcile(parsed, lambda line="": print(line, file=out))
    assert out.getvalue() == ""


def test_the_same_queue_seen_by_both_forms_reconciles(tmp_path):
    """One device, two forms: the printed numbers differ by the reference-burst
    term and by nothing else, and the read-out has to say so rather than
    reporting a disagreement it manufactured itself."""
    text = _reconcile_text(tmp_path, q_depth=24, s_depth=24)
    assert "RECONCILED" in text
    assert "LOWER BOUND by exactly the term it drops" in text
    assert "UNEXPLAINED" not in text


def test_the_bare_and_levelled_columns_are_the_two_read_outs(tmp_path):
    """`bare` is what phase Q prints and `levelled` is what phase S prints, so
    the gap between the two documents is a column subtraction and not a
    difference between devices. At depth 24, n_ref=16 and p/S = 1.125/3 the
    dropped term is 16 * (1 - 0.375) = 10 entries."""
    text = _reconcile_text(tmp_path, q_depth=24, s_depth=24)
    assert "terms differ by 8.0 entries (10.0 at n_ref = 16 against 2.0 at" in text


def test_a_form_dependent_depth_is_reported_as_unexplained(tmp_path):
    """What 2026-08-05 silicon actually produced. The reference-burst correction
    is applied and the two forms STILL disagree, so neither number may be quoted
    alone -- and the run-ahead estimator, which shares no term with the backlog
    arithmetic, has to be shown disagreeing too or the residual could be blamed
    on the correction."""
    text = _reconcile_text(tmp_path, q_depth=21, s_depth=28)
    assert "ENTRIES UNEXPLAINED" in text
    assert "run-ahead estimator" in text
    assert "SO THE DEPTH DEPENDS ON WHICH FORM MEASURES IT" in text
    assert "RECONCILED" not in text


def test_the_run_ahead_estimator_recovers_the_depth_it_was_built_from(tmp_path):
    """It uses no `_sync` probe and no reference burst: only how far the plain
    burst returns ahead of `n * S`, with the timed region's own fixed cost added
    back from the issue-limited probe's intercept. Dropping that `c` would bias
    every run-ahead down by 4 entries at a 3-cycle service rate."""
    text = _reconcile_text(tmp_path, q_depth=24, s_depth=24)
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("phase ")]
    assert [ln.split()[-1] for ln in rows] == ["24.0", "24.0"]


# ---------------------------------------------------------------------------
# Phase G -- the footprints between phase F's 1024 and 2048. It is a separate
# kernel BUILD because phase F's is already at tt-metal's kernel config buffer
# limit, and its probes therefore arrive under their own phase letter with
# their own control. The read-out has to fold them into one footprint table
# ordered by size, and it must not promote "a boundary in loop-body size" into
# "a cache capacity" on the strength of a narrower bracket.
# ---------------------------------------------------------------------------


def test_phase_g_footprints_join_the_same_table_in_size_order(tmp_path):
    out = io.StringIO()
    sweep._fetch_check(
        _footprint_series(
            tmp_path,
            {"f_64": 1, "f_1024": 1, "g_1024": 1, "g_1280": 1, "f_2048": 1.25},
        ),
        lambda line="": print(line, file=out),
    )
    table = [
        line.split()[0]
        for line in out.getvalue().splitlines()
        if line.startswith(("  f_", "  g_"))
    ]
    assert table == ["f_64", "f_1024", "g_1024", "g_1280", "f_2048"]
    assert "5120" in out.getvalue()  # g_1280's footprint in BYTES, which is the claim


def test_the_bracket_narrows_with_a_phase_g_point_and_stays_a_loop_body_size(tmp_path):
    """The whole point of phase G, and the whole point of the wording.

    A stepped 1792 and a flat 1536 move the bracket from 4096-8192 to
    6144-7168. What must NOT move is the noun: no document gives an
    instruction cache size or a miss cost, so a narrower bracket is a narrower
    bracket and nothing else.
    """
    out = io.StringIO()
    sweep._fetch_check(
        _footprint_series(
            tmp_path,
            {"f_64": 1, "f_1024": 1, "g_1536": 1, "g_1792": 1.25, "f_2048": 1.25},
        ),
        lambda line="": print(line, file=out),
    )
    text = out.getvalue()
    assert "flat through a 6144-byte loop body, stepped by 7168 bytes" in text
    assert "it is not a cache size" in text


def test_the_additions_are_reported_present_so_a_forced_null_is_not_an_absence(
    tmp_path,
):
    """Against tt-sim both new phases read their null BY CONSTRUCTION -- no
    instruction cache is modelled and the Tensix queue is a list append -- so a
    reader has to be able to tell that from a probe that never ran."""
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            _share_slot(depth=24, service=3)
            + _slope_rows("g_1280", "RV_FETCH", 2, phase="g"),
        )
    )
    out = io.StringIO()
    sweep._additions_present(rows, lambda line="": print(line, file=out))
    text = out.getvalue()
    assert "phase S: " in text
    assert "s_co_plain 8 points" in text
    assert "phase G: " in text
    assert "g_1280" in text
    assert "g_1536" not in text.split("phase G:")[1].split("Exactly ONE")[0]


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


def test_the_single_phase_gset_runs_are_tracked_and_never_chosen():
    """Phase G could not be one kernel, so `g_1536` and `g_1792` exist only in
    twelve-row `--gset` runs. They are tracked because they are the only
    evidence for those two footprints -- and they can never be the primary,
    because a single-phase single-thread run cannot run the live-instrument
    check that makes any of its numbers readable."""
    tracked = {p.name for p in sweep.reference_datasets()}
    assert set(sweep.FOOTPRINT_DATASETS) <= tracked
    assert sweep.PRIMARY_DATASET not in sweep.FOOTPRINT_DATASETS
    assert sweep.default_measured_path().name == sweep.PRIMARY_DATASET
    for name in sweep.FOOTPRINT_DATASETS:
        rows, _ = sweep.read_csv(sweep.DATASET_DIR / name)
        assert {r["phase"] for r in rows} == {"g"}
        assert {r["variant"] for r in rows} == {"t1"}


def test_the_tracked_datasets_carry_their_own_provenance():
    """A measurement separated from its card, firmware and flags is not one."""
    for path in sweep.reference_datasets():
        head = path.read_text()[:6000]
        assert "device=blackhole-silicon" in head, path.name
        assert "firmware_bundle=" in head, path.name
        assert "kmd=" in head, path.name
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
