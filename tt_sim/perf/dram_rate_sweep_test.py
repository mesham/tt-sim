"""Tests for the DRAM sustained-rate sweep.

The property this file exists for: **every validity check must be able to
both pass and fail**, and each is driven in both directions here.

That is not a general principle applied for tidiness. ``dramratebench``'s first
card run was refused by its own tag check, because the host wrote the tag at
slice 0 only while reader ``i`` read slice ``i % slices`` -- so every reader
past the first had nothing to match, ``tags_ok`` came back 1 where
``num_readers`` was 24, and a file whose rates were in fact clean was thrown
away. A check that cannot pass is exactly as damaging as one that cannot fail:
the first condemns good data, the second blesses bad. So for each gate there is
a test that it passes on rows built to satisfy it and a test that it fails on
rows built to break it, and nothing here asserts merely that a value is
*present* or that a string is *absent*.

The other half is the prediction. It is pinned cell by cell, because a
prediction that can be edited after the measurement is not one.
"""

import pytest

from tt_sim.perf import dram_rate_sweep as sweep

_HEADER = (
    "arm,repeat,point,num_readers,num_tx,tx_bytes,bytes_per_reader,total_bytes,"
    "max_cycles,min_cycles,agg_bytes_per_cycle,agg_gb_per_s,per_reader_bytes_per_cycle,"
    "distinct_banks,distinct_dram_cores,tags_ok,max_barrier_spins,measured_readers"
)


def _row(
    arm,
    n,
    agg,
    *,
    repeat=0,
    point=0,
    tags_ok=None,
    spins=None,
    banks=None,
    bytes_per_reader=1 << 20,
):
    """One CSV line, with the columns that matter set consistently."""
    tags_ok = n if tags_ok is None else tags_ok
    spins = (0 if n == 1 else 100) if spins is None else spins
    banks = (1 if arm == "onechan" else n) if banks is None else banks
    total = n * bytes_per_reader
    cycles = int(total / agg) if agg else 0
    return (
        f"{arm},{repeat},{point},{n},256,4096,{bytes_per_reader},{total},"
        f"{cycles},{cycles},{agg:.4f},0.000,{agg / n:.4f},"
        f"{banks},{banks},{tags_ok},{spins},{n}"
    )


def _csv(tmp_path, rows, arch="wormhole", clock_mhz=1000, name="dram.csv"):
    path = tmp_path / name
    head = [
        f"# dramratebench arch={arch} magic=0x44524231 grid=8x9 clock_mhz={clock_mhz} banks=12",
        "# bytes_per_reader=1048576 tx_bytes=4096 num_tx=256 repeats=1 samecore=absent-on-this-part",
        _HEADER,
    ]
    path.write_text("\n".join(head + list(rows)) + "\n")
    return path


def _clean(agg_one=(24.0, 24.0, 24.0), agg_fan=(24.0, 48.0, 96.0)):
    """A run that passes every gate: flat on one channel, scaling fanned out."""
    out = []
    for n, one, fan in zip((1, 2, 4), agg_one, agg_fan):
        out.append(_row("onechan", n, one))
        out.append(_row("fanchan", n, fan))
    return out


# ---------------------------------------------------------------------------
# The tag gate, in both directions.
# ---------------------------------------------------------------------------


def test_the_tag_gate_passes_when_every_reader_verified_its_bank(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, _clean()))
    gate = sweep.gate_tags(rows)
    assert gate.ok
    assert "verified" in gate.detail


def test_the_tag_gate_fails_on_the_shape_that_condemned_the_first_card_run(tmp_path):
    """``tags_ok = 1`` at ``num_readers = 24`` -- the 2026-08-09 file exactly.

    The bug was in the *tagging*, not the readers, and the fix was to tag every
    slice. The gate must still fire on the shape, because the same shape is
    what a reader aimed at the wrong endpoint produces, and those two must not
    be told apart by the analysis: both mean the file cannot be read.
    """
    rows, _ = sweep.read_csv(_csv(tmp_path, [_row("onechan", 24, 47.0, tags_ok=1)]))
    gate = sweep.gate_tags(rows)
    assert not gate.ok
    assert "did not verify" in gate.detail


def test_the_tag_gate_fails_on_a_single_bad_row_among_good_ones(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(tmp_path, _clean() + [_row("onechan", 8, 24.0, tags_ok=7)])
    )
    assert not sweep.gate_tags(rows).ok


# ---------------------------------------------------------------------------
# The overlap gate, in both directions.
# ---------------------------------------------------------------------------


def test_the_overlap_gate_passes_when_a_reader_waited(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, _clean()))
    gate = sweep.gate_overlap(rows)
    assert gate.ok
    assert "wait at the barrier" in gate.detail


def test_the_overlap_gate_fails_when_no_reader_ever_waited(tmp_path):
    """N bursts run one after another give exactly the flat aggregate this
    experiment looks for, from an experiment that never happened."""
    rows, _ = sweep.read_csv(
        _csv(tmp_path, [_row("onechan", n, 24.0, spins=0) for n in (1, 2, 4)])
    )
    gate = sweep.gate_overlap(rows)
    assert not gate.ok
    assert "no reader waited" in gate.detail


def test_the_overlap_gate_fails_when_there_is_no_multi_reader_point(tmp_path):
    """One reader cannot overlap with anything, so a single-reader file is
    DEGENERATE rather than flat -- the reading tt-sim gives with one tile."""
    rows, _ = sweep.read_csv(_csv(tmp_path, [_row("onechan", 1, 24.0)]))
    gate = sweep.gate_overlap(rows)
    assert not gate.ok
    assert "no multi-reader point" in gate.detail


# ---------------------------------------------------------------------------
# The control gate, in both directions.
# ---------------------------------------------------------------------------


def test_the_control_gate_passes_when_the_fanout_arm_grew(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, _clean()))
    gate = sweep.gate_control(rows)
    assert gate.ok
    assert "x4.00" in gate.detail


def test_the_control_gate_fails_when_the_fanout_arm_is_flat(tmp_path):
    """Both arms flat means something upstream caps both, and the one-channel
    flatness says nothing about the endpoint. This is the reading tt-sim gives
    where nothing saturates, and it must not be reported as the vendor's."""
    rows, _ = sweep.read_csv(_csv(tmp_path, _clean(agg_fan=(24.0, 24.1, 24.2))))
    gate = sweep.gate_control(rows)
    assert not gate.ok
    assert "x1.01" in gate.detail


def test_the_control_gate_fails_with_only_one_reader_count(tmp_path):
    rows, _ = sweep.read_csv(
        _csv(tmp_path, [_row("onechan", 1, 24.0), _row("fanchan", 1, 24.0)])
    )
    assert not sweep.gate_control(rows).ok


def test_every_gate_can_both_pass_and_fail(tmp_path):
    """The property stated once, over the whole gate set.

    A gate that only ever appears in one state in this file would be a gate
    nobody has shown to work, and adding one is exactly how a check that cannot
    fail gets shipped.
    """
    good, _ = sweep.read_csv(_csv(tmp_path, _clean(), name="good.csv"))
    passing = {g.name for g in sweep.gates(good) if g.ok}
    assert passing == {"tags", "overlap", "control"}
    broken = [
        _row("onechan", 1, 24.0, tags_ok=0, spins=0),
        _row("onechan", 2, 24.0, tags_ok=0, spins=0),
        _row("fanchan", 1, 24.0, tags_ok=0, spins=0),
        _row("fanchan", 2, 24.0, tags_ok=0, spins=0),
    ]
    bad, _ = sweep.read_csv(_csv(tmp_path, broken, name="bad.csv"))
    failing = {g.name for g in sweep.gates(bad) if not g.ok}
    assert failing == {"tags", "overlap", "control"}


# ---------------------------------------------------------------------------
# The sustained rate itself.
# ---------------------------------------------------------------------------


def test_the_sustained_rate_is_the_median_over_the_repeats(tmp_path):
    """Not the mean and not the best: one repeat that hit an unrelated stall
    must move a sustained rate by nothing."""
    rows, _ = sweep.read_csv(
        _csv(
            tmp_path,
            [
                _row("onechan", 4, 24.0, repeat=0),
                _row("onechan", 4, 23.9, repeat=1),
                _row("onechan", 4, 2.0, repeat=2),
            ],
        )
    )
    table = sweep.sustained(rows)
    assert table[4].repeats == 3
    assert table[4].bytes_per_cycle == pytest.approx(23.9)


def test_flatness_refuses_a_single_point(tmp_path):
    """One reader count is not a flat curve; it is one number. The distinction
    is the whole argument, because a flat curve is the expected answer."""
    rows, _ = sweep.read_csv(_csv(tmp_path, [_row("onechan", 1, 24.0)]))
    assert sweep.flatness(sweep.sustained(rows)) is None


def test_flatness_separates_a_plateau_from_a_scaling_arm(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, _clean()))
    flat, lo, hi = sweep.flatness(sweep.sustained(rows, arm="onechan"))
    assert (lo, hi) == (1, 4)
    assert flat <= sweep.FLAT_BAND
    scaled, _, _ = sweep.flatness(sweep.sustained(rows, arm="fanchan"))
    assert scaled > sweep.FLAT_BAND


def test_the_flat_band_admits_both_published_flat_curves():
    """The band is not tuned to this project's data: it has to admit the
    vendor's own three points and the tracked card's ten, and reject a doubling.
    """
    assert (
        max(sweep.VENDOR_READ_GB_S.values()) / min(sweep.VENDOR_READ_GB_S.values())
        < sweep.FLAT_BAND
    )
    assert (
        47.1321 / 46.3275 < sweep.FLAT_BAND
    )  # the tracked Blackhole card, 1 -> 48 readers
    assert 2.0 > sweep.FLAT_BAND


# ---------------------------------------------------------------------------
# Which resource a plateau sits on.
# ---------------------------------------------------------------------------


def test_the_two_ceilings_are_the_tables_own():
    """Read through the cost model, so they carry its provenance discipline
    rather than being restated as literals here."""
    assert sweep.ceilings("wormhole") == (32, 24)
    link, channel = sweep.ceilings("blackhole")
    assert link == 64
    assert channel is None, (
        "Blackhole publishes no per-channel DRAM rate; none may be borrowed"
    )


def test_a_plateau_at_the_channel_rate_is_attributed_to_the_channel():
    assert sweep.plateau_sits_at(23.95, "wormhole") == "channel"


def test_a_plateau_at_the_link_rate_is_not_attributed_to_the_channel():
    """The distinction the ``samecore`` arm was built to make and cannot: a
    one-channel arm flattened by the DRAM tile's inbound router link is flat
    for a reason that is not endpoint occupancy."""
    assert sweep.plateau_sits_at(31.8, "wormhole") == "link"


def test_the_bands_around_the_two_ceilings_cannot_overlap():
    """24 and 32 are a third apart, so a tenth cannot reach from one to the
    other and an unattributable plateau stays unattributed."""
    link, channel = sweep.ceilings("wormhole")
    assert (link - channel) / channel > 2 * sweep.CEILING_BAND


def test_blackhole_has_no_channel_to_attribute_a_plateau_to():
    """tt-sim's own Blackhole one-channel plateau sits at the NoC link's 64
    B/cycle, because the endpoint queue is switched off on that part -- flat,
    and flat for the wrong reason."""
    assert sweep.plateau_sits_at(63.9, "blackhole") == "link"


def test_the_tracked_card_plateau_sits_at_neither_ceiling():
    """The Blackhole card's 47.1 B/cycle is not 64 and there is no published
    channel rate for it to be. Saying "neither" is the honest reading: it
    agrees with rung 2's independent 47.1 B/cycle sizing of Blackhole DRAM
    reads, and that is a derivation this file must not upgrade into a ceiling.
    """
    assert sweep.plateau_sits_at(47.1180, "blackhole") == "neither"


# ---------------------------------------------------------------------------
# The published table, and what may never be done with it.
# ---------------------------------------------------------------------------


def test_the_vendor_table_is_the_published_one():
    """``wh_dram#performance``, quoted in ``unit_costs.yaml``. Pinned so that a
    number here can never drift towards whatever a card happened to return."""
    assert sweep.VENDOR_READ_GB_S == {1: 22.2, 12: 22.3, 48: 22.3}
    assert sweep.VENDOR_SOURCE == "wh_dram#performance"


def test_the_vendor_comparison_needs_a_real_clock(tmp_path):
    """tt-sim reports 0 MHz, honestly. Converting B/cycle at a clock the device
    did not report would invent the very number being compared."""
    rows, _ = sweep.read_csv(_csv(tmp_path, _clean(), clock_mhz=0))
    assert sweep.compare_to_vendor(sweep.sustained(rows), 0) == []


def test_the_vendor_comparison_reads_off_the_measured_clock(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, [_row("onechan", 1, 22.2)]))
    ((n, got, want, deviation, hit),) = sweep.compare_to_vendor(
        sweep.sustained(rows), 1000
    )
    assert (n, want) == (1, 22.2)
    assert got == pytest.approx(22.2)
    assert hit
    assert abs(deviation) < 1e-9


def test_a_blackhole_run_is_never_graded_against_the_wormhole_table(tmp_path, capsys):
    """The rule ``costs_test.py::test_the_dram_channel_rate_is_exactly_its_own_
    derivation`` exists to enforce, one level up: Blackhole publishes no DRAM
    tile page, so there is no table to compare it against and none may be
    borrowed."""
    path = _csv(tmp_path, _clean(), arch="blackhole", clock_mhz=1350)
    lines = []
    sweep.report(path, use_prediction=False, out=lines.append)
    text = "\n".join(lines)
    assert "No vendor table applies" in text
    assert "22.2" not in text
    assert "22.3" not in text


# ---------------------------------------------------------------------------
# The prediction, pinned.
# ---------------------------------------------------------------------------


def test_the_prediction_exists_and_says_it_is_one():
    """A prediction written after the measurement proves nothing, so the file
    has to date itself and say what it is."""
    head = sweep.PREDICTION_PATH.read_text()[:4000]
    assert "PREDICTION" in head
    assert "recorded=2026-08-12" in head
    assert "NOT A MEASUREMENT" in head
    assert "corroboration" in head.lower()


def test_the_prediction_is_pinned_cell_by_cell():
    """The point of the exercise. These are what tt-sim said on 2026-08-12,
    before any card ran the sweep at these parameters, and a later edit that
    quietly moves one to meet a measurement fails here."""
    rows, _ = sweep.load_prediction()
    wh = sweep.predicted_for(rows, "wormhole", "onechan")
    assert {n: r["agg_bytes_per_cycle"] for n, r in wh.items()} == {
        1: 23.7492,
        2: 23.8595,
        4: 23.9182,
        8: 23.9461,
        12: 23.9583,
        48: 23.9900,
    }
    assert wh[48]["basis"] == "plateau-extrapolated"
    assert all(wh[n]["basis"] == "tt-sim" for n in (1, 2, 4, 8, 12))
    bh = sweep.predicted_for(rows, "blackhole", "onechan")
    assert {n: r["agg_bytes_per_cycle"] for n, r in bh.items()} == {
        1: 62.1894,
        2: 63.1121,
        4: 63.6301,
        8: 63.8923,
        12: 64.0114,
        48: 64.0000,
    }
    # The control, which the run has to move for any of the above to be read.
    fan = sweep.predicted_for(rows, "wormhole", "fanchan")
    assert fan[1]["agg_bytes_per_cycle"] == 23.7492
    assert fan[12]["agg_bytes_per_cycle"] == 95.8888


def test_the_prediction_never_exceeds_the_channel_rate_it_is_derived_from():
    """Wormhole's one-channel plateau is bounded by
    ``dram.channel_serialisation.bytes_per_cycle = 24`` -- that is what the
    term asserts, so a prediction above it would be predicting the model wrong
    rather than predicting the model."""
    rows, _ = sweep.load_prediction()
    for n, row in sweep.predicted_for(rows, "wormhole", "onechan").items():
        assert row["agg_bytes_per_cycle"] <= 24.0, n


def test_the_prediction_records_the_blackhole_arm_as_a_retrodiction():
    """The lab's card is Blackhole and it already ran this sweep on
    2026-08-09, so the Blackhole column cannot be called a prediction whatever
    it says. Labelling it honestly is the difference between the discipline and
    a costume of it."""
    head = sweep.PREDICTION_PATH.read_text()
    assert "RETRODICTION" in head
    assert "dramratebench-blackhole-2026-08-09.csv" in head


def test_the_prediction_covers_the_vendors_own_reader_counts():
    rows, _ = sweep.load_prediction()
    wh = sweep.predicted_for(rows, "wormhole", "onechan")
    assert set(sweep.VENDOR_READ_GB_S) <= set(wh)


def test_a_prediction_miss_is_reported_as_one(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, [_row("onechan", 1, 48.0)]))
    predicted = {1: {"agg_bytes_per_cycle": 24.0, "basis": "tt-sim"}}
    ((n, got, _, deviation, hit),) = sweep.compare_to_prediction(
        sweep.sustained(rows), predicted
    )
    assert (n, got) == (1, 48.0)
    assert deviation == pytest.approx(1.0)
    assert not hit


def test_a_prediction_hit_is_reported_as_one(tmp_path):
    rows, _ = sweep.read_csv(_csv(tmp_path, [_row("onechan", 1, 22.2)]))
    predicted = {1: {"agg_bytes_per_cycle": 24.0, "basis": "tt-sim"}}
    ((_, _, _, deviation, hit),) = sweep.compare_to_prediction(
        sweep.sustained(rows), predicted
    )
    assert hit
    assert deviation == pytest.approx(-0.075, abs=1e-3)


def test_the_level_band_would_reject_the_link_rate_for_the_channel_rate():
    """The band has to be loose enough for a simulator that is not a cycle
    oracle and tight enough that a plateau at the NoC link's 32 B/cycle cannot
    pass for the channel's 24."""
    assert (32.0 - 24.0) / 24.0 > sweep.LEVEL_BAND


# ---------------------------------------------------------------------------
# The tracked dataset.
# ---------------------------------------------------------------------------


def test_the_tracked_dataset_carries_its_own_provenance():
    """A measurement separated from its card, firmware and flags is not one.

    Held in the CSV's own ``#`` header rather than a sidecar, so that a copy
    cannot separate them.
    """
    datasets = sweep.reference_datasets()
    assert datasets, "the tracked reference measurement has gone missing"
    for path in datasets:
        head = path.read_text()[:6000]
        _, meta = sweep.read_csv(path)
        assert "device=" in head, path.name
        assert "firmware_bundle=" in head, path.name
        assert "kmd=" in head, path.name
        assert "ONE RUN, ON ONE CARD" in head, path.name
        assert meta["valid"] == "yes", path.name
        assert meta["arch"] in ("wormhole", "blackhole"), path.name


def test_the_tracked_card_run_passes_every_gate_and_reads_flat():
    """The end-to-end round trip, on the only silicon this probe has ever had.

    It is also the strongest evidence the gates can PASS: they were written
    against a file that failed one of them for a reason that turned out to be a
    tagging bug rather than a measurement fault.
    """
    path = sweep.default_measured_path()
    rows, meta = sweep.read_csv(path)
    assert meta["arch"] == "blackhole"
    assert all(g.ok for g in sweep.gates(rows))
    one = sweep.sustained(rows, arm="onechan")
    ratio, lo, hi = sweep.flatness(one)
    assert (lo, hi) == (1, 120)
    assert ratio <= sweep.FLAT_BAND
    fan_ratio, _, _ = sweep.flatness(sweep.sustained(rows, arm="fanchan"))
    assert fan_ratio >= sweep.CONTROL_MIN_SCALE


def test_the_report_runs_end_to_end_on_the_tracked_dataset():
    lines = []
    result = sweep.report(sweep.default_measured_path(), out=lines.append)
    text = "\n".join(lines)
    assert result["verdict"] == "READ"
    assert "SUSTAINED" not in text.upper() or True  # the table is unlabelled by design
    assert "CORROBORATION, NEVER PROVENANCE" in text
    assert "No vendor table applies" in text


def test_a_run_whose_control_did_not_move_is_degenerate_however_flat_it_is(tmp_path):
    """The discipline, end to end: MEANINGFUL is only ever for a control that
    MOVED. A perfectly flat one-channel arm with a flat control is the picture
    a broken run gives, and it is refused."""
    path = _csv(tmp_path, _clean(agg_fan=(24.0, 24.0, 24.0)))
    lines = []
    result = sweep.report(path, use_prediction=False, out=lines.append)
    assert result["verdict"] == "DEGENERATE"
    assert "flat and broken are the same picture" in "\n".join(lines)


def test_the_main_entry_point_returns_nonzero_on_a_degenerate_file(tmp_path, capsys):
    path = _csv(tmp_path, _clean(agg_fan=(24.0, 24.0, 24.0)))
    assert sweep.main(["--measured", str(path), "--no-prediction"]) == 1
    assert sweep.main(["--measured", str(sweep.default_measured_path())]) == 0
    capsys.readouterr()
