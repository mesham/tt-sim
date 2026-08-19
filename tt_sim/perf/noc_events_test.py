"""Guards for :mod:`tt_sim.perf.noc_events`.

A guard that cannot fail is as damaging as one that cannot pass, so every gate
has a passing case *and* a refusing case built from an input a real session
could plausibly produce: two runs concatenated, a Wormhole-numbered core against
a Blackhole-numbered one, a profiler run with the NoC recorder not compiled in,
a dropped barrier END, a non-monotonic timestamp, and two sides that ran
different transfer sizes.

The two checked-in synthetic card traces are exercised end to end here, in both
directions -- an agreeing card that must PASS and a compensating one that must
FAIL. The compensating one is the leg's whole argument: same span to within the
envelope limit, **every per-class latency correct**, and the disagreement
visible only in the partition.
"""

import json
from pathlib import Path

import pytest

from tt_sim.perf.noc_events import (
    ARM_NOC_PAIRING,
    MECHANISMS,
    PROFILER_PUSH_ZONE,
    PROLOGUE_CAVEAT,
    PROLOGUE_DOMINATES,
    TOLERANCE,
    analyse,
    census,
    gate_arm_matches,
    gate_barriers_pair,
    gate_census_matches,
    gate_events_present,
    gate_partition_closes,
    gate_per_core,
    gate_single_window,
    group_by_stream,
    latency_classes,
    load_trace,
    load_traces,
    main,
    parse_core_map,
    partition,
    prologue_note,
    render,
)

TESTDATA = Path(__file__).resolve().parents[2] / "perfbench" / "nocevbench" / "testdata"
SIM = TESTDATA / "sim-blackhole-4096.json"
GEMM256 = TESTDATA / "sim-gemm256-16core-brisc-EXTRACT.json"
CARD_AGREE = TESTDATA / "card-agreeing-SYNTHETIC-NOT-A-MEASUREMENT.json"
CARD_COMP = TESTDATA / "card-compensating-SYNTHETIC-NOT-A-MEASUREMENT.json"


# ---------------------------------------------------------------------------
# Builders -- small synthetic streams, so a gate's refusing case is readable
# ---------------------------------------------------------------------------


def rec(ts, **kw):
    out = {"sx": 1, "sy": 2, "proc": "NCRISC", "timestamp": ts, "run_host_id": 0}
    out.update(kw)
    return out


def zone(ts, phase, proc="NCRISC", name=None):
    return rec(ts, proc=proc, zone=name or f"{proc}-KERNEL", zone_phase=phase)


def push(ts, phase, proc="NCRISC"):
    """One endpoint of the device profiler's own L1-buffer flush zone."""
    return zone(ts, phase, proc, name=PROFILER_PUSH_ZONE)


def noc(ts, kind, proc="NCRISC", num_bytes=0, nocname="NOC_1"):
    return rec(ts, proc=proc, type=kind, noc=nocname, num_bytes=num_bytes, vc=-1)


def simple_stream(scale=1, proc="NCRISC"):
    """ZONE_START, one read, its barrier, ZONE_END -- the minimum valid stream."""
    return [
        zone(0, "ZONE_START", proc),
        noc(10 * scale, "READ", proc, 4096),
        noc(20 * scale, "READ_BARRIER_START", proc),
        noc(120 * scale, "READ_BARRIER_END", proc),
        zone(140 * scale, "ZONE_END", proc),
    ]


def stream_of(records):
    return group_by_stream(_load(records))[((1, 2), records[0]["proc"])]


def _load(records, tmp_path=None):
    """Round-trip through the real loader, so the tests exercise the parser."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(records, fh)
        path = fh.name
    return load_trace(path)


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def test_loads_the_real_simulator_artefact():
    events = load_trace(SIM)
    assert len(events) == 54
    streams = group_by_stream(events)
    assert sorted(streams, key=repr) == [((1, 2), "BRISC"), ((1, 2), "NCRISC")]


def test_loader_refuses_a_file_that_is_not_this_artefact(tmp_path):
    path = tmp_path / "not-a-trace.json"
    path.write_text(json.dumps([{"hello": "world"}]))
    with pytest.raises(ValueError, match="not a NoC trace record"):
        load_trace(path)

    path.write_text(json.dumps({"records": []}))
    with pytest.raises(ValueError, match="expected a JSON array"):
        load_trace(path)


def test_ties_keep_file_order_rather_than_being_reordered():
    """Two events can share a wall-clock value; reordering them would make a
    negative interval out of nothing."""
    records = [
        zone(0, "ZONE_START"),
        noc(10, "READ", num_bytes=64),
        noc(10, "READ_BARRIER_START"),
        noc(50, "READ_BARRIER_END"),
        zone(60, "ZONE_END"),
    ]
    part = partition(stream_of(records))
    assert min(part.values()) >= 0
    assert sum(part.values()) == 60


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_partition_telescopes_to_the_zone_span():
    for stream in group_by_stream(load_trace(SIM)).values():
        part = partition(stream)
        assert sum(part.values()) == stream.span
        assert min(part.values()) >= 0


def test_every_bucket_is_opened_by_the_event_it_names():
    part = partition(stream_of(simple_stream()))
    assert part == {
        "prologue": 10,  # ZONE_START -> READ
        "issue": 10,  # READ -> READ_BARRIER_START
        "read_wait": 100,  # the barrier spin
        "write_wait": 0,
        "other_wait": 0,
        "local": 20,  # READ_BARRIER_END -> ZONE_END
        "profiler": 0,
    }


def test_a_blocking_single_event_call_is_not_counted_as_issue_work():
    records = [
        zone(0, "ZONE_START"),
        noc(10, "SEMAPHORE_WAIT"),
        noc(500, "READ", num_bytes=64),
        noc(510, "READ_BARRIER_START"),
        noc(600, "READ_BARRIER_END"),
        zone(610, "ZONE_END"),
    ]
    part = partition(stream_of(records))
    assert part["other_wait"] == 490
    assert part["issue"] == 10


# ---------------------------------------------------------------------------
# Latency classes and the census
# ---------------------------------------------------------------------------


def test_latency_is_issue_to_the_barrier_end_that_covers_it():
    classes, unbarriered = latency_classes(stream_of(simple_stream()))
    assert unbarriered == 0
    assert classes[("NOC_1", "READ", 4096)].samples == [110]


def test_set_state_records_issue_no_packet_and_are_not_latency_paired():
    records = [
        zone(0, "ZONE_START"),
        noc(10, "READ_SET_STATE"),
        noc(20, "READ", num_bytes=128),
        noc(30, "READ_BARRIER_START"),
        noc(200, "READ_BARRIER_END"),
        zone(210, "ZONE_END"),
    ]
    classes, _ = latency_classes(stream_of(records))
    assert list(classes) == [("NOC_1", "READ", 128)]
    # ... but the configuring event is still in the census, so a side that
    # skipped it would be caught.
    assert ("NOC_1", "READ_SET_STATE", 0) in census(stream_of(records))


def test_transactions_never_barriered_are_reported_not_dropped():
    records = [
        zone(0, "ZONE_START"),
        noc(10, "READ", num_bytes=64),
        noc(20, "READ", num_bytes=64),
        zone(30, "ZONE_END"),
    ]
    classes, unbarriered = latency_classes(stream_of(records))
    assert classes == {}
    assert unbarriered == 2


# ---------------------------------------------------------------------------
# The gates -- a passing case and a refusing case for each
# ---------------------------------------------------------------------------


def test_gate_events_present_passes_on_a_real_stream():
    sim = group_by_stream(load_trace(SIM))[((1, 2), "NCRISC")]
    assert gate_events_present(sim, None).passed


def test_gate_events_present_refuses_a_zone_with_no_noc_events():
    """TT_METAL_DEVICE_PROFILER=1 without the NoC recorder produces exactly
    this, and it decodes as 'this kernel moved no data'."""
    records = [zone(0, "ZONE_START"), zone(900, "ZONE_END")]
    gate = gate_events_present(stream_of(records), None)
    assert not gate.passed
    assert "zero NoC events" in gate.detail


def test_gate_single_window_passes_on_one_launch():
    assert gate_single_window(stream_of(simple_stream()), None).passed


def test_gate_single_window_refuses_two_concatenated_runs():
    records = simple_stream() + [
        zone(1000, "ZONE_START"),
        noc(1010, "READ", num_bytes=4096),
        noc(1020, "READ_BARRIER_START"),
        noc(1120, "READ_BARRIER_END"),
        zone(1140, "ZONE_END"),
    ]
    for index, record in enumerate(records):
        if index >= 5:
            record["run_host_id"] = 1
    gate = gate_single_window(stream_of(records), None)
    assert not gate.passed
    assert "run host IDs" in gate.detail


def test_gate_per_core_refuses_a_silent_wormhole_blackhole_mismatch():
    sim = stream_of(simple_stream())
    card_records = [dict(r, sx=1, sy=1) for r in simple_stream()]
    card = group_by_stream(_load(card_records))[((1, 1), "NCRISC")]
    gate = gate_per_core(sim, card)
    assert not gate.passed
    assert "--map-core" in gate.detail
    assert gate_per_core(sim, card, mapped=True).passed


def test_gate_per_core_refuses_comparing_two_different_riscs():
    sim = stream_of(simple_stream(proc="NCRISC"))
    card = stream_of(simple_stream(proc="BRISC"))
    gate = gate_per_core(sim, card)
    assert not gate.passed
    assert "RISC" in gate.detail


def test_gate_barriers_pair_refuses_a_dropped_end():
    """The profiler drops events silently when its L1 vector overflows and the
    DRAM buffer is full, and a truncated stream still partitions cleanly."""
    records = [
        zone(0, "ZONE_START"),
        noc(10, "READ", num_bytes=64),
        noc(20, "READ_BARRIER_START"),
        zone(300, "ZONE_END"),
    ]
    gate = gate_barriers_pair(stream_of(records), None)
    assert not gate.passed
    assert "dropped" in gate.detail


def test_gate_barriers_pair_refuses_an_end_with_no_start():
    records = [
        zone(0, "ZONE_START"),
        noc(10, "READ_BARRIER_END"),
        zone(300, "ZONE_END"),
    ]
    gate = gate_barriers_pair(stream_of(records), None)
    assert not gate.passed
    assert "no matching START" in gate.detail


def test_gate_partition_closes_refuses_a_non_monotonic_timestamp():
    """The wall clock is stored in 44 bits and read as two separate register
    accesses, so a torn read or a wrap is physically possible."""
    stream = stream_of(simple_stream())
    # Move ZONE_END before the last NoC event: the final interval goes negative.
    events = list(stream.events)
    events[-1] = type(events[-1])(**{**events[-1].__dict__, "timestamp": 5})
    stream.events = events
    gate = gate_partition_closes(stream, None)
    assert not gate.passed
    assert "negative" in gate.detail or "sum" in gate.detail


def test_gate_census_matches_refuses_two_different_transfer_sizes():
    """The mistake the sibling leg's README records: a simulator run at one
    problem size compared against a card run at another."""
    sim = stream_of(simple_stream())
    card_records = [dict(r) for r in simple_stream()]
    for record in card_records:
        if record.get("num_bytes"):
            record["num_bytes"] = 8160
    card = group_by_stream(_load(card_records))[((1, 2), "NCRISC")]
    gate = gate_census_matches(sim, card)
    assert not gate.passed
    assert "different transactions" in gate.detail


def test_gate_census_matches_passes_when_the_work_is_identical():
    sim = stream_of(simple_stream())
    card = stream_of(simple_stream(scale=3))
    assert gate_census_matches(sim, card).passed


# ---------------------------------------------------------------------------
# The criterion, on the checked-in synthetic pair
# ---------------------------------------------------------------------------


def test_the_agreeing_card_passes():
    reports = analyse(load_trace(SIM), load_trace(CARD_AGREE))
    assert len(reports) == 2
    for report in reports:
        assert not report.refused, report.gates
        assert report.passed
        assert report.comparison.e_int <= TOLERANCE
        assert report.comparison.e_total <= TOLERANCE


def test_the_compensating_card_fails_on_the_interior_alone():
    """The leg's whole argument, in one assertion block.

    The BRISC stream's span is inside the envelope limit and **every per-class
    latency is inside the limit too** -- the shift was constructed to leave every
    inter-event distance among the NoC events untouched. Only the partition sees
    it.
    """
    reports = {r.proc: r for r in analyse(load_trace(SIM), load_trace(CARD_COMP))}
    brisc = reports["BRISC"]
    assert not brisc.refused
    assert brisc.comparison.e_total <= TOLERANCE  # the envelope PASSES
    assert brisc.comparison.e_int > TOLERANCE  # the interior does not
    assert brisc.comparison.ratio > 5.0  # compensation, measured
    assert all(latency.passed for latency in brisc.latencies)  # latency PASSES
    assert not brisc.passed

    # The other stream is untouched, so a failure is localised rather than
    # condemning the whole run.
    assert reports["NCRISC"].passed


def test_the_compensating_card_is_the_same_work_as_the_simulator_run():
    """If the census differed, the FAIL above would be uninteresting -- it would
    just mean the two sides ran different programs."""
    reports = analyse(load_trace(SIM), load_trace(CARD_COMP))
    for report in reports:
        gate = next(g for g in report.gates if g.name == "census_matches")
        assert gate.passed


# ---------------------------------------------------------------------------
# Reporting and the CLI
# ---------------------------------------------------------------------------


def test_render_names_the_compensation_ratio_on_every_comparison():
    text = render(analyse(load_trace(SIM), load_trace(CARD_COMP)))
    assert "compensation E_int/E_total" in text
    assert "RESULT: FAIL" in text


def test_render_says_decomposition_only_without_card_data():
    text = render(analyse(load_trace(SIM), None), decompose_only=True)
    assert "RESULT: DECOMPOSITION ONLY" in text
    assert "NOT a per-packet flight time" in text


def test_cli_exit_codes_and_json(tmp_path):
    out = tmp_path / "report.json"
    assert main(["--sim", str(SIM), "--card", str(CARD_AGREE), "--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["tolerance"] == TOLERANCE
    assert len(payload["streams"]) == 2

    assert main(["--sim", str(SIM), "--card", str(CARD_COMP)]) == 1
    assert main(["--sim", str(SIM), "--decompose-only"]) == 0


def test_cli_refuses_a_meaningless_argument_combination():
    with pytest.raises(SystemExit):
        main(["--sim", str(SIM)])
    with pytest.raises(SystemExit):
        main(["--sim", str(SIM), "--card", str(CARD_AGREE), "--decompose-only"])


def test_load_traces_concatenates_and_the_gate_then_refuses():
    """Globbing a directory of traces is allowed at load time and refused at
    analysis time, so the caller is told what they did."""
    events = load_traces([SIM, CARD_AGREE])
    assert len(events) == 108
    reports = analyse(events, None)
    assert all(r.refused for r in reports)


# ---------------------------------------------------------------------------
# The opt-in arm gate
# ---------------------------------------------------------------------------


def test_the_checked_in_sim_run_is_arm_a_and_the_gate_says_so():
    """The recorded run really is the control arm: NCRISC reads on NoC 1 and
    BRISC writes on NoC 0. That pairing is what makes direction and NoC the same
    statement, which is the confound arm B exists to break."""
    reports = analyse(load_trace(SIM), load_trace(CARD_AGREE), expect_arm="A")
    assert len(reports) == 2
    for report in reports:
        assert not report.refused, report.gates
        assert any(g.name == "arm_matches" and g.passed for g in report.gates)


def test_the_arm_gate_refuses_a_run_that_kept_the_other_arm_s_nocs():
    """The failure the gate exists for: a run labelled B that silently ran A's
    configuration. It is well-formed, every other gate passes it, and it would
    read as "the error stayed on the write" -- which is what the *rival*
    hypothesis predicts."""
    reports = analyse(load_trace(SIM), load_trace(CARD_AGREE), expect_arm="B")
    assert all(r.refused for r in reports)
    details = " ".join(
        g.detail for r in reports for g in r.gates if g.name == "arm_matches"
    )
    assert "did not take" in details


def test_the_arm_gate_is_absent_unless_asked_for():
    """Six standing gates, and the seventh only when a caller names an arm."""
    reports = analyse(load_trace(SIM), load_trace(CARD_AGREE))
    for report in reports:
        assert [g.name for g in report.gates] == [
            "events_present",
            "single_window",
            "per_core",
            "barriers_pair",
            "partition_closes",
            "census_matches",
        ]


def test_arms_a_and_c_share_a_pairing_so_the_peer_is_what_separates_them():
    """Stated as a test rather than a comment, because it is the reason
    ``--peer-noc`` exists: arm C changes the target, not the NoCs, so the NoC
    table alone cannot tell a peer-L1 run from a DRAM one."""
    assert ARM_NOC_PAIRING["A"] == ARM_NOC_PAIRING["C"]
    sim = stream_of(simple_stream())
    assert gate_arm_matches(sim, None, "C").passed
    refused = gate_arm_matches(sim, None, "C", peer=(2, 3))
    assert not refused.passed
    assert "peer" in refused.detail


def test_the_arm_gate_accepts_a_run_that_addressed_the_peer():
    records = [dict(r) for r in simple_stream()]
    for record in records:
        if record.get("type") == "READ":
            record["dx"], record["dy"] = 2, 3
    stream = group_by_stream(_load(records))[((1, 2), "NCRISC")]
    assert gate_arm_matches(stream, None, "C", peer=(2, 3)).passed


def test_cli_refuses_a_peer_without_an_arm():
    with pytest.raises(SystemExit):
        main(["--sim", str(SIM), "--decompose-only", "--peer-noc", "2,3"])


def test_parse_core_map():
    assert parse_core_map(["1,1=1,2"]) == {(1, 1): (1, 2)}
    with pytest.raises(ValueError, match="SIMX,SIMY=CARDX,CARDY"):
        parse_core_map(["1,1"])


# ---------------------------------------------------------------------------
# perfbench/nocevbench/check_arm.py -- the card-box copy of the arm check.
#
# It is standalone by design (a card box has only perfbench/nocevbench/ on it
# and no tt-sim), so it is loaded by path rather than imported. Guarded here
# because its arm-C peer check is the one that a Wormhole session breaks, in two
# opposite ways, and both were found against tt-sim rather than at a card:
# under NoC coordinate translation the config's peer coord and the trace's are
# in different spaces and a naive equality REFUSES a good run; untranslated, the
# NoC 1 destination is the peer's grid mirror and a naive equality refuses it
# with a message that names neither cause.
# ---------------------------------------------------------------------------
import importlib.util  # noqa: E402

_CHECK_ARM = (
    Path(__file__).resolve().parents[2] / "perfbench" / "nocevbench" / "check_arm.py"
)


def _check_arm():
    spec = importlib.util.spec_from_file_location("check_arm", _CHECK_ARM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_c_trace(read_dest, write_dest, source=(1, 1)):
    """One NCRISC read on NOC_1 and one BRISC write on NOC_0, as arm C emits."""
    return [
        {
            "proc": "NCRISC",
            "noc": "NOC_1",
            "type": "READ",
            "num_bytes": 256,
            "timestamp": 10,
            "sx": source[0],
            "sy": source[1],
            "dx": read_dest[0],
            "dy": read_dest[1],
        },
        {
            "proc": "BRISC",
            "noc": "NOC_0",
            "type": "WRITE_",
            "num_bytes": 256,
            "timestamp": 20,
            "sx": source[0],
            "sy": source[1],
            "dx": write_dest[0],
            "dy": write_dest[1],
        },
    ]


def test_check_arm_accepts_arm_c_across_two_coordinate_spaces():
    """A translated Wormhole run: the config names the peer (19, 19) and this
    core (18, 18) while the trace numbers them (2, 2) and (1, 1). Both are
    right; only the spaces differ, and the disagreement on the *initiating*
    core is what says so."""
    module = _check_arm()
    config = {
        "arm": "C",
        "reader": "NCRISC:NOC_1",
        "writer": "BRISC:NOC_0",
        "target": "l1",
        "self_noc": "18,18",
        "peer_noc": "19,19",
        "noc_grid": "10,12",
    }
    problems, notes = module.check("C", _arm_c_trace((2, 2), (2, 2)), config)
    assert problems == []
    assert any("different coordinate spaces" in note for note in notes)


def test_check_arm_names_the_untranslated_noc1_mirror():
    """An untranslated Wormhole run: the NoC 1 read went to (7, 9), the grid
    mirror of the peer (2, 2) on 10x12. Refused -- the two sides are not
    comparable -- but refused by name, with the fix in the message."""
    module = _check_arm()
    config = {
        "arm": "C",
        "reader": "NCRISC:NOC_1",
        "writer": "BRISC:NOC_0",
        "target": "l1",
        "self_noc": "1,1",
        "peer_noc": "2,2",
        "noc_grid": "10,12",
    }
    problems, _ = module.check("C", _arm_c_trace((7, 9), (2, 2)), config)
    assert len(problems) == 1
    assert "grid mirror" in problems[0]
    assert "TT_METAL_MOCK_CLUSTER_DESC_PATH" in problems[0]


def test_check_arm_still_refuses_a_peer_it_never_addressed():
    """The check the other two must not have weakened."""
    module = _check_arm()
    config = {
        "arm": "C",
        "reader": "NCRISC:NOC_1",
        "writer": "BRISC:NOC_0",
        "target": "l1",
        "self_noc": "1,1",
        "peer_noc": "2,2",
        "noc_grid": "10,12",
    }
    problems, _ = module.check("C", _arm_c_trace((4, 4), (4, 4)), config)
    assert any("did not go to the core the arm names" in p for p in problems)


# ---------------------------------------------------------------------------
# ``prologue`` is unattributed time, not setup time
#
# The defect these guard is a reading error, not an arithmetic one: a writer
# kernel that blocks in ``cb_wait_front`` before its first NoC transaction shows
# 92 % ``prologue``, and nothing in the artefact says so. The caveat therefore
# has to be unconditional, and the loud line has to fire on exactly that shape.
# ---------------------------------------------------------------------------


def writer_stream(prologue=6370, proc="BRISC"):
    """The gemm_128 writer shape: a long blocking wait, then a short burst.

    ``cb_wait_front`` emits nothing, so the whole wait is one interval from
    ``ZONE_START`` to the first ``WRITE_``.
    """
    return [
        zone(0, "ZONE_START", proc),
        noc(prologue, "WRITE_", proc, 4096, "NOC_0"),
        noc(prologue + 196, "WRITE_BARRIER_START", proc, nocname="NOC_0"),
        noc(prologue + 308, "WRITE_BARRIER_END", proc, nocname="NOC_0"),
        zone(prologue + 562, "ZONE_END", proc),
    ]


def test_a_blocking_dependency_wait_lands_entirely_in_prologue():
    """The finding itself, pinned: the partition still closes, and the wait is
    indistinguishable from setup by arithmetic alone -- which is why the report
    has to say it in words."""
    stream = stream_of(writer_stream())
    part = partition(stream)
    assert stream.span == 6932
    assert part["prologue"] == 6370
    assert sum(part.values()) == stream.span
    assert 0.9189 == pytest.approx(part["prologue"] / stream.span, abs=5e-5)


def test_prologue_dominating_raises_a_note_naming_the_side():
    report = analyse(_load(writer_stream()), None)[0]
    assert not report.refused
    note = next(n for n in report.notes if "prologue" in n)
    assert note.startswith("sim: ")
    assert "91.89 %" in note
    assert "cb_wait_front" in note
    assert "not read it as an initialisation cost" in note


def test_a_small_prologue_raises_no_note():
    """The other direction: the checked-in artefact's prologue is 0.34 % and
    1.91 %, so a note there would be noise that trains a reader to ignore it."""
    for report in analyse(load_trace(SIM), None):
        assert not any("prologue is" in n for n in report.notes), report.notes


def test_the_note_threshold_is_the_documented_constant():
    """Straddle :data:`PROLOGUE_DOMINATES` rather than trusting a magic 25 %."""
    span = 1000
    under = int(span * PROLOGUE_DOMINATES) - 1
    over = int(span * PROLOGUE_DOMINATES) + 1
    # span is prologue + 562 of tail in the builder, so size the wait directly.
    assert prologue_note({"prologue": under}, span, "sim") is None
    assert prologue_note({"prologue": over}, span, "sim") is not None
    assert prologue_note({"prologue": 100}, 0, "sim") is None


def test_the_card_side_gets_its_own_note():
    """A dependency-bound *card* stream must be named as the card's, not the
    simulator's -- which side stalled is the actionable half."""
    sim = writer_stream(prologue=100, proc="NCRISC")
    card = writer_stream(prologue=6370, proc="NCRISC")
    report = analyse(_load(sim), _load(card))[0]
    assert not report.refused, report.gates
    assert any(n.startswith("card: prologue") for n in report.notes)
    assert not any(n.startswith("sim: prologue") for n in report.notes)


def test_the_prologue_caveat_is_unconditional_in_both_render_modes():
    """It appears even when the number is small: a small prologue is a smaller
    upper bound on setup, not proof that it *is* setup."""
    head = PROLOGUE_CAVEAT[0]
    decompose = render(analyse(load_trace(SIM), None), decompose_only=True)
    comparison = render(analyse(load_trace(SIM), load_trace(CARD_AGREE)))
    for text in (decompose, comparison):
        assert head in text
        assert "not comparable across cores" in text
        assert "prologue *" in text  # the row itself carries the marker


def test_the_partition_stays_the_compared_seven_buckets():
    """The refusal, pinned. A ``pre_first_event_wait`` bucket is recoverable
    only simulator-side, so adding it to *this* tuple would put a bucket the
    card cannot fill into E_int. If a future change adds one, it must move this
    guard deliberately.

    ``profiler`` was added deliberately, on 2026-08-19, and passes that test:
    the profiler's own ``PROFILER-NOC-QUICK-PUSH`` flush is recorded by the same
    firmware in the same artefact on **both** sides, so E_int compares like with
    like. It is here rather than inside ``issue`` because a push is triggered
    from the issue path, and folding it in would charge the model for the
    difference between how a card and tt-sim execute the *instrument's* own DRAM
    write."""
    assert MECHANISMS == (
        "prologue",
        "issue",
        "read_wait",
        "write_wait",
        "other_wait",
        "local",
        "profiler",
    )
    for report in analyse(load_trace(SIM), None):
        assert set(report.sim_partition) == set(MECHANISMS)
        assert sum(report.sim_partition.values()) == report.sim_span


# ---------------------------------------------------------------------------
# The profiler's own zone, nested inside the kernel zone
# ---------------------------------------------------------------------------
#
# ``kernel_profiler::quick_push()`` flushes the per-RISC L1 marker buffer to
# DRAM when it fills mid-kernel, and brackets itself in a
# ``PROFILER-NOC-QUICK-PUSH`` zone that reaches the artefact because tt-metal's
# own filter still names the zone's old spelling. Counting those endpoints as
# launches is what made this leg decline every real workload put to it: the
# 16-core ``gemm_256_check`` pair of 2026-08-19 refused all 32 streams on both
# halves, all of which carry exactly one kernel window.


def push_stream(pushes=1, proc="NCRISC"):
    """A read kernel whose profiler flushed ``pushes`` times mid-kernel.

    Shaped like the real capture: the push sits between two issue events,
    because ``recordNocEvent`` flushes *before* it records.
    """
    records = [zone(0, "ZONE_START", proc), noc(100, "READ", proc, 4096)]
    ts = 100
    for _ in range(pushes):
        ts += 110  # workload's own inter-issue interval
        records.append(push(ts, "ZONE_START", proc))
        records.append(push(ts + 24, "ZONE_END", proc))  # the two markers
        ts += 190  # the markers plus the DMA and its posted-write flush
        records.append(noc(ts, "READ", proc, 4096))
    records.append(noc(ts + 10, "READ_BARRIER_START", proc))
    records.append(noc(ts + 110, "READ_BARRIER_END", proc))
    records.append(zone(ts + 130, "ZONE_END", proc))
    return records


def test_the_profiler_s_own_flush_is_not_a_second_launch():
    """The refusal this leg used to give a real 16-core capture. Five zone
    starts on one stream were one kernel window and four buffer flushes."""
    stream = stream_of(push_stream(pushes=4))
    assert stream.profiler_pushes == 4
    assert len(stream.zone_starts) == 1
    gate = gate_single_window(stream, None)
    assert gate.passed, gate.detail
    assert "4 profiler flush(es)" in gate.detail


def test_the_flush_is_bucketed_apart_from_issue_not_folded_into_it():
    """A push is triggered from inside the issue path, so absorbing it would
    charge the model for how the *instrument* executes its own DRAM write."""
    part = partition(stream_of(push_stream(pushes=2)))
    # the push's own ZONE_START to the next event outside it: 24 cycles of
    # markers, then 166 for the DMA the ZONE_END marker precedes
    assert part["profiler"] == 2 * 190
    # the workload's inter-issue intervals, and nothing of the push
    assert part["issue"] == 110 + 110 + 10
    assert sum(part.values()) == stream_of(push_stream(pushes=2)).span


def test_a_stream_with_flushes_still_closes_and_never_goes_negative():
    for pushes in (0, 1, 4):
        stream = stream_of(push_stream(pushes=pushes))
        part = partition(stream)
        assert sum(part.values()) == stream.span, pushes
        assert min(part.values()) >= 0
        assert gate_partition_closes(stream, None).passed


def test_the_flush_is_noted_at_any_size():
    """Not gated on a share threshold: a reader comparing two captures has to
    know one of them carries more of the instrument than the other."""
    report = analyse(_load(push_stream(pushes=3)), None)[0]
    assert report.sim_pushes == 3
    note = next(n for n in report.notes if "flushed its own L1 marker buffer" in n)
    assert "3 time(s)" in note
    assert "upper bound" in note


def test_two_real_kernel_launches_are_still_refused_and_told_what_to_do():
    """The gate that matters is unchanged: two *kernel* zones in one file are
    two program executions and will not be blended, flushes or no flushes."""
    records = push_stream(pushes=1) + [
        zone(5000, "ZONE_START"),
        noc(5010, "READ", num_bytes=4096),
        noc(5020, "READ_BARRIER_START"),
        noc(5120, "READ_BARRIER_END"),
        zone(5140, "ZONE_END"),
    ]
    gate = gate_single_window(stream_of(records), None)
    assert not gate.passed
    assert "2 kernel ZONE_START" in gate.detail
    assert "NCRISC-KERNEL" in gate.detail
    assert "one at a time" in gate.detail
    assert "1 seen" in gate.detail  # the flush is named as NOT the cause


def test_a_consumers_own_nested_zone_is_passed_through_and_named():
    """Unlike a profiler flush, a ``DeviceZoneScopedN`` brackets the kernel's
    own work, so its time stays in the enclosing bucket rather than moving to a
    bucket of its own -- but the pass-through is never silent."""
    records = [
        zone(0, "ZONE_START"),
        noc(10, "READ", num_bytes=4096),
        zone(20, "ZONE_START", name="MY-INNER-ZONE"),
        zone(60, "ZONE_END", name="MY-INNER-ZONE"),
        noc(110, "READ_BARRIER_START"),
        noc(210, "READ_BARRIER_END"),
        zone(230, "ZONE_END"),
    ]
    stream = stream_of(records)
    assert stream.nested_zone_names == ["MY-INNER-ZONE"]
    part = partition(stream)
    assert sum(part.values()) == stream.span == 230
    assert part["issue"] == 100  # READ -> READ_BARRIER_START, straight through
    assert part["profiler"] == 0
    report = analyse(_load(records), None)[0]
    assert any("MY-INNER-ZONE" in n for n in report.notes)


# ---------------------------------------------------------------------------
# The reduced real extract
# ---------------------------------------------------------------------------


def test_the_real_16_core_capture_decomposes_rather_than_refusing():
    """One BRISC stream of the 2026-08-19 ``gemm_256_check`` pair, unmodified.
    Every one of that capture's 32 streams was declined as ``single_window``;
    this is the shape of the input that did it."""
    reports = analyse(load_trace(GEMM256), None)
    assert len(reports) == 1
    report = reports[0]
    assert not report.refused, [g.detail for g in report.gates if not g.passed]
    assert report.sim_span == 60754
    assert report.sim_pushes == 1
    assert sum(report.sim_partition.values()) == report.sim_span
    assert report.sim_partition["profiler"] == 183
    assert report.sim_partition["issue"] == 9202


def test_the_real_extract_carries_a_genuine_profiler_flush():
    """Guards the extract itself: if it were re-cut without a flush in it, the
    test above would pass for the wrong reason."""
    events = load_trace(GEMM256)
    names = {e.zone for e in events if e.kind == "zone"}
    assert names == {"BRISC-KERNEL", PROFILER_PUSH_ZONE}
