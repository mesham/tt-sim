"""Guards for :mod:`tt_sim.perf.retire_attribution`.

A guard that cannot fail is as damaging as one that cannot pass, so every gate
has a passing case *and* a refusing case built from an input a real session
could plausibly produce: a Wormhole artefact, two sides run at different
``--scale``, a Blackhole-numbered core against a Wormhole-numbered one, a
simulator run with ``TT_SIM_COST_MODEL`` unset, a zone too short for the
instrument, a counter that went backwards, and two sides whose retired counts
disagree.

The three checked-in artefacts are exercised end to end here, in both
directions -- an agreeing card that must PASS and a compensating one that must
FAIL. **The compensating one is the leg's whole argument**: the same total to
within 0.91 %, which every envelope check in this repo would wave through, and
the disagreement visible only in the partition.

Two invariants are asserted directly, because if either slips the compensation
ratio stops being a measurement and becomes decoration: the buckets telescope to
the outer window on both sides, and ``E_total <= E_int`` always.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tt_sim.perf.retire_attribution import (
    CALIBRATION_ZONE,
    E_INT_LIMIT,
    E_TOTAL_LIMIT,
    MAX_ZONES,
    MIN_ZONE_CYCLES,
    UNATTRIBUTED,
    Zone,
    analyse,
    compare_cpi,
    compare_partitions,
    gate_arch_supported,
    gate_cost_model_engaged,
    gate_counters_advanced,
    gate_partition_closes,
    gate_per_core,
    gate_retire_census_matches,
    gate_zone_budget,
    gate_zone_table_matches,
    load_run,
    main,
    mechanisms,
    parse_core_map,
    partition,
    render,
    retire_partition,
    zone_cpi,
)

TESTDATA = (
    Path(__file__).resolve().parents[2] / "perfbench" / "retirebench" / "testdata"
)
SIM = TESTDATA / "sim-blackhole-scale1.json"
CARD_AGREE = TESTDATA / "card-agreeing-SYNTHETIC-NOT-A-MEASUREMENT.json"
CARD_COMP = TESTDATA / "card-compensating-SYNTHETIC-NOT-A-MEASUREMENT.json"


# ---------------------------------------------------------------------------
# Builders -- small synthetic runs, so a gate's refusing case is readable
# ---------------------------------------------------------------------------


def artefact(zones=None, **kw):
    """A minimal well-formed artefact dict.

    Two measured zones and a calibration zone: the smallest thing that is a
    partition of a span by a mechanism rather than by the instrument.
    """
    zones = (
        zones
        if zones is not None
        else [
            {
                "index": 0,
                "name": CALIBRATION_ZONE,
                "mechanism": "calibration",
                "reps": 0,
                "cycles": 4,
                "retired": 1,
            },
            {
                "index": 1,
                "name": "alu_dep",
                "mechanism": "dependent integer chain",
                "reps": 94,
                "cycles": 6000,
                "retired": 6000,
            },
            {
                "index": 2,
                "name": "load_dep",
                "mechanism": "L1 load-use interlock",
                "reps": 12,
                "cycles": 6000,
                "retired": 780,
            },
        ]
    )
    body = {
        "retirebench": 1,
        "arch": "blackhole",
        "label": "unit-test",
        "risc": "BRISC",
        "core": [1, 2],
        "logical_core": [0, 0],
        "scale": 1,
        "window": {
            "cycles": sum(z["cycles"] for z in zones) + 10,
            "retired": sum(z["retired"] for z in zones) + 8,
        },
        "zones": zones,
    }
    body.update(kw)
    return body


def run_of(tmp_path, name="a.json", **kw):
    """Round-trip through the real loader, so the tests exercise the parser."""
    path = tmp_path / name
    path.write_text(json.dumps(artefact(**kw)))
    return load_run(path)


def edited(run, **zone_cycles):
    """A copy of ``run`` with named zones' cycles replaced, window kept closed."""
    zones = [replace(z, cycles=zone_cycles.get(z.name, z.cycles)) for z in run.zones]
    unattributed = run.window_cycles - sum(z.cycles for z in run.zones)
    new = replace(run, zones=zones)
    new.window_cycles = sum(z.cycles for z in zones) + unattributed
    return new


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def test_loads_the_real_simulator_artefact():
    run = load_run(SIM)
    assert run.arch == "blackhole"
    assert run.risc == "BRISC"
    assert run.core == (1, 2)
    assert len(run.zones) == 12
    assert run.zones[0].name == CALIBRATION_ZONE
    assert run.window_cycles == 60359
    assert run.window_retired == 40915


def test_loader_refuses_a_file_that_is_not_this_artefact(tmp_path):
    path = tmp_path / "not-a-run.json"
    path.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError, match="retirebench"):
        load_run(path)

    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="retirebench"):
        load_run(path)


def test_loader_refuses_an_artefact_with_a_zone_missing_its_counts(tmp_path):
    """A zone with no ``cycles`` would default to 0 and read as a free zone."""
    path = tmp_path / "truncated.json"
    body = artefact()
    del body["zones"][1]["cycles"]
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="lacks 'cycles'"):
        load_run(path)


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_partition_telescopes_to_the_outer_window():
    """The property that makes both sides' denominators the same quantity."""
    for path in (SIM, CARD_AGREE, CARD_COMP):
        run = load_run(path)
        assert sum(partition(run).values()) == run.window_cycles
        assert sum(retire_partition(run).values()) == run.window_retired


def test_unattributed_is_whatever_no_zone_claimed(tmp_path):
    run = run_of(tmp_path)
    assert partition(run)[UNATTRIBUTED] == 10
    assert retire_partition(run)[UNATTRIBUTED] == 8


def test_mechanisms_are_the_zones_then_unattributed(tmp_path):
    run = run_of(tmp_path)
    assert mechanisms(run) == (CALIBRATION_ZONE, "alu_dep", "load_dep", UNATTRIBUTED)


def test_zone_cpi_skips_a_zone_that_retired_nothing(tmp_path):
    run = run_of(tmp_path)
    run.zones[1] = replace(run.zones[1], retired=0)
    assert "alu_dep" not in zone_cpi(run)
    assert "load_dep" in zone_cpi(run)


def test_e_total_never_exceeds_e_int():
    """The triangle inequality, which is what makes the ratio >= 1 and therefore
    a measurement of compensation rather than a decoration."""
    sim = load_run(SIM)
    for path in (CARD_AGREE, CARD_COMP):
        hw = load_run(path)
        c = compare_partitions("t", mechanisms(sim), partition(sim), partition(hw))
        assert c.e_total <= c.e_int
        assert c.ratio >= 1.0


def test_perfect_agreement_has_no_compensation_ratio():
    sim = load_run(SIM)
    c = compare_partitions("t", mechanisms(sim), partition(sim), partition(sim))
    assert c.e_total == 0.0
    assert c.e_int == 0.0
    assert c.ratio is None
    assert c.passed


# ---------------------------------------------------------------------------
# The gates -- each in both directions
# ---------------------------------------------------------------------------


def test_gate_arch_supported_passes_two_blackhole_artefacts(tmp_path):
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json")
    assert gate_arch_supported(sim, hw).passed


def test_gate_arch_supported_refuses_a_wormhole_artefact(tmp_path):
    """THE refusal this leg is required to make, rather than degrading to an
    elapsed-only envelope check."""
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json", arch="wormhole")
    gate = gate_arch_supported(sim, hw)
    assert not gate.passed
    assert "wormhole" in gate.detail
    # The refusal has to say WHY, or the next reader patches it out.
    assert "minstret" in gate.detail
    assert "envelope" in gate.detail
    # ... and it refuses on the simulator side too, not only the card's.
    assert not gate_arch_supported(hw, None).passed


def test_gate_zone_table_matches_passes_the_same_program(tmp_path):
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json")
    assert gate_zone_table_matches(sim, hw).passed


def test_gate_zone_table_matches_refuses_two_different_scales(tmp_path):
    """A scale mismatch is the easiest way to compare two different programs and
    the hardest to see: every zone is simply longer on one side."""
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json", scale=2)
    gate = gate_zone_table_matches(sim, hw)
    assert not gate.passed
    assert "--scale" in gate.detail


def test_gate_zone_table_matches_refuses_different_zones_or_reps(tmp_path):
    sim = run_of(tmp_path, "s.json")
    other = artefact()["zones"]
    other[1]["name"] = "alu_ind"
    hw = run_of(tmp_path, "h.json", zones=other)
    assert not gate_zone_table_matches(sim, hw).passed

    reps = artefact()["zones"]
    reps[1]["reps"] = 47
    hw2 = run_of(tmp_path, "h2.json", zones=reps)
    gate = gate_zone_table_matches(sim, hw2)
    assert not gate.passed
    assert "reps" in gate.detail


def test_gate_counters_advanced_passes_a_real_run():
    assert gate_counters_advanced(load_run(SIM), load_run(CARD_AGREE)).passed


def test_gate_counters_advanced_refuses_a_run_whose_counters_stood_still(tmp_path):
    """An unread counter reads back as zero, which decodes as a perfectly
    plausible partition of a zero-cycle span."""
    dead = artefact()
    for z in dead["zones"]:
        z["cycles"] = 0
        z["retired"] = 0
    dead["window"] = {"cycles": 0, "retired": 0}
    sim = run_of(tmp_path, "s.json", zones=dead["zones"], window=dead["window"])
    gate = gate_counters_advanced(sim, None)
    assert not gate.passed
    assert "did not run" in gate.detail


def test_gate_counters_advanced_exempts_only_the_calibration_zone_s_cycles(tmp_path):
    """A zero-cycle instrument is a claim about the instrument, not a dead run."""
    zones = artefact()["zones"]
    zones[0]["cycles"] = 0
    sim = run_of(tmp_path, "s.json", zones=zones)
    assert gate_counters_advanced(sim, None).passed


def test_gate_per_core_refuses_a_silent_wormhole_blackhole_mismatch(tmp_path):
    """Worker (0,0) is physical (1,1) on Wormhole and (1,2) on Blackhole, which
    makes this an easy and invisible mistake."""
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json", core=[1, 1])
    gate = gate_per_core(sim, hw)
    assert not gate.passed
    assert "--map-core" in gate.detail
    assert gate_per_core(sim, hw, mapped=True).passed


def test_gate_per_core_refuses_comparing_two_different_riscs(tmp_path):
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json", risc="NCRISC")
    gate = gate_per_core(sim, hw, mapped=True)
    assert not gate.passed
    assert "NCRISC" in gate.detail


def test_gate_zone_budget_passes_the_real_artefact():
    gate = gate_zone_budget(load_run(SIM), load_run(CARD_AGREE))
    assert gate.passed
    # The smallest measured zone and the measured marker cost are printed, not
    # assumed -- that is the whole reason the calibration zone exists.
    assert "two-marker cost 3" in gate.detail


def test_gate_zone_budget_refuses_a_zone_under_the_floor(tmp_path):
    sim = run_of(tmp_path, "s.json")
    short = edited(sim, alu_dep=MIN_ZONE_CYCLES - 1)
    gate = gate_zone_budget(short, None)
    assert not gate.passed
    assert f"under the {MIN_ZONE_CYCLES}-cycle floor" in gate.detail
    # ... and passes at exactly the floor, so the bound is the stated one.
    assert gate_zone_budget(edited(sim, alu_dep=MIN_ZONE_CYCLES), None).passed


def test_gate_zone_budget_refuses_an_instrument_that_costs_too_much(tmp_path):
    """The marker cost is checked against the MEASURED calibration zone, so an
    instrument that grew cannot hide behind the roadmap's ~30-cycle estimate."""
    zones = artefact()["zones"]
    zones[0]["cycles"] = 400  # 6.7 % of a 6000-cycle zone, over the 6 % budget
    sim = run_of(tmp_path, "s.json", zones=zones)
    gate = gate_zone_budget(sim, None)
    assert not gate.passed
    assert "over the 6 % budget" in gate.detail


def test_gate_zone_budget_refuses_an_artefact_with_no_calibration_zone(tmp_path):
    zones = artefact()["zones"][1:]
    sim = run_of(tmp_path, "s.json", zones=zones)
    gate = gate_zone_budget(sim, None)
    assert not gate.passed
    assert CALIBRATION_ZONE in gate.detail


def test_gate_zone_budget_refuses_more_zones_than_the_ceiling(tmp_path):
    zones = [artefact()["zones"][0]]
    for i in range(MAX_ZONES + 1):
        zones.append(
            {
                "index": i + 1,
                "name": f"z{i}",
                "mechanism": "m",
                "reps": 1,
                "cycles": 2000,
                "retired": 2000,
            }
        )
    sim = run_of(tmp_path, "s.json", zones=zones)
    gate = gate_zone_budget(sim, None)
    assert not gate.passed
    assert f"over the {MAX_ZONES} ceiling" in gate.detail


def test_gate_zone_budget_refuses_a_partition_with_no_measured_zone(tmp_path):
    sim = run_of(tmp_path, "s.json", zones=[artefact()["zones"][0]])
    gate = gate_zone_budget(sim, None)
    assert not gate.passed
    assert "decomposes nothing" in gate.detail


def test_gate_cost_model_engaged_passes_the_real_simulator_run():
    gate = gate_cost_model_engaged(load_run(SIM), None)
    assert gate.passed
    assert "7.7" in gate.detail  # load_dep/alu_dep on the real run


def test_gate_cost_model_engaged_refuses_a_run_with_the_cost_model_off(tmp_path):
    """With TT_SIM_COST_MODEL unset every load costs one cycle, so the pointer
    chase collapses onto the addi chain -- and the run is otherwise well-formed,
    which is exactly why this has to be a gate and not an eyeball check."""
    zones = artefact()["zones"]
    zones[2]["retired"] = 6000  # load_dep at 1.0 cycles/instruction
    sim = run_of(tmp_path, "s.json", zones=zones)
    gate = gate_cost_model_engaged(sim, None)
    assert not gate.passed
    assert "TT_SIM_COST_MODEL" in gate.detail


def test_gate_partition_closes_passes_the_real_artefacts():
    assert gate_partition_closes(load_run(SIM), load_run(CARD_AGREE)).passed


def test_gate_partition_closes_refuses_zones_overflowing_their_window(tmp_path):
    """A counter that went backwards comes back as a huge unsigned delta, which
    drives the unattributed bucket negative rather than looking wrong."""
    body = artefact()
    body["window"]["cycles"] = 100  # less than the zones inside it
    path = tmp_path / "s.json"
    path.write_text(json.dumps(body))
    gate = gate_partition_closes(load_run(path), None)
    assert not gate.passed
    assert "negative" in gate.detail


def test_gate_partition_closes_also_checks_the_retire_partition(tmp_path):
    body = artefact()
    body["window"]["retired"] = 3
    path = tmp_path / "s.json"
    path.write_text(json.dumps(body))
    gate = gate_partition_closes(load_run(path), None)
    assert not gate.passed
    assert "retire partition" in gate.detail


def test_gate_retire_census_matches_passes_when_the_work_is_identical():
    gate = gate_retire_census_matches(load_run(SIM), load_run(CARD_AGREE))
    assert gate.passed


def test_gate_retire_census_matches_refuses_one_extra_retired_instruction(tmp_path):
    """Exact equality, not a tolerance: a difference is a statement about WHAT
    ran, and there is no size of it that would be acceptable."""
    sim = run_of(tmp_path, "s.json")
    other = artefact()["zones"]
    other[1]["retired"] += 1
    hw = run_of(tmp_path, "h.json", zones=other)
    gate = gate_retire_census_matches(sim, hw)
    assert not gate.passed
    assert "alu_dep" in gate.detail
    assert "(-1)" in gate.detail  # signed: the card retired the extra one


# ---------------------------------------------------------------------------
# The checked-in artefacts, end to end, in both directions
# ---------------------------------------------------------------------------


def test_the_agreeing_card_passes():
    (report,) = analyse(load_run(SIM), load_run(CARD_AGREE))
    assert not report.refused, [g.detail for g in report.gates if not g.passed]
    assert report.passed
    assert report.comparison.e_total <= E_TOTAL_LIMIT
    assert report.comparison.e_int <= E_INT_LIMIT


def test_the_compensating_card_fails_on_the_interior_alone():
    """The leg's whole argument, in one assertion block.

    The total agrees to well inside the envelope limit -- every envelope check
    in this repo would wave it through -- and the decomposition behind it is
    wrong by 45 %, in compensating directions. The ratio is what a passing total
    cannot fake.
    """
    (report,) = analyse(load_run(SIM), load_run(CARD_COMP))
    assert not report.refused, [g.detail for g in report.gates if not g.passed]
    comparison = report.comparison
    assert comparison.e_total <= E_TOTAL_LIMIT  # the envelope check PASSES
    assert comparison.e_total < 0.02  # and not marginally
    assert comparison.e_int > E_INT_LIMIT  # the interior does not
    assert comparison.ratio > 20.0  # compensation, measured
    assert not comparison.passed
    assert not report.passed


def test_the_compensating_card_is_the_same_work_as_the_simulator_run():
    """It has to fail for the right reason: same program, same instructions
    retired, only the cycles moved. Otherwise it would be caught by a gate and
    would demonstrate nothing about the criterion."""
    sim, hw = load_run(SIM), load_run(CARD_COMP)
    assert gate_zone_table_matches(sim, hw).passed
    assert gate_retire_census_matches(sim, hw).passed
    assert retire_partition(sim) == retire_partition(hw)
    assert sum(partition(sim).values()) != sum(partition(hw).values())


def test_the_calibration_zone_is_in_the_partition_but_not_graded(tmp_path):
    """It measures the instrument, and the two sides' marker costs are EXPECTED
    to differ, because tt-sim charges a CSR instruction nothing. Grading it
    would fail every honest run."""
    sim, hw = load_run(SIM), load_run(CARD_AGREE)
    assert CALIBRATION_ZONE in partition(sim)
    assert sim.marker_cycles == 3
    assert hw.marker_cycles == 30
    assert CALIBRATION_ZONE not in {c.zone for c in compare_cpi(sim, hw)}
    # ... and if it WERE graded, it would fail, which is why the exclusion is
    # load-bearing rather than cosmetic.
    assert abs(hw.marker_cycles - sim.marker_cycles) / hw.marker_cycles > 0.25


def test_the_marker_cost_difference_is_reported_as_a_note():
    (report,) = analyse(load_run(SIM), load_run(CARD_AGREE))
    assert any("two-marker cost differs" in n for n in report.notes)
    assert any("DisCsrSync" in n for n in report.notes)


# ---------------------------------------------------------------------------
# The report and the CLI
# ---------------------------------------------------------------------------


def test_render_names_the_compensation_ratio_and_the_structural_caveat():
    text = render(analyse(load_run(SIM), load_run(CARD_COMP)))
    assert "compensation E_int/E_total" in text
    # A reader must not be able to mistake a zone label for a hardware counter.
    assert "STRUCTURAL" in text
    assert "minstret" in text
    assert "NOT an independent check" in text


def test_render_says_decomposition_only_without_card_data():
    text = render(analyse(load_run(SIM), None), decompose_only=True)
    assert "DECOMPOSITION ONLY" in text
    assert "E_total" not in text.split("RESULT")[-1]


def test_render_reports_a_refusal_without_a_comparison(tmp_path):
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json", arch="wormhole")
    text = render(analyse(sim, hw))
    assert "REFUSE" in text
    assert "REFUSED -- no comparison reported." in text
    assert "compensation" not in text


def test_a_refused_run_exports_no_comparison_at_all(tmp_path):
    """Gates run first and a refusal stops the analysis.

    An E_int computed over a partition that did not close is a number, but it is
    not a measurement of anything -- and once it is in the JSON somebody quotes
    it. So the refusal has to leave the comparison ABSENT rather than merely
    unprinted.
    """
    sim = run_of(tmp_path, "s.json")
    hw = run_of(tmp_path, "h.json", arch="wormhole")
    (report,) = analyse(sim, hw)
    assert report.refused
    assert report.comparison is None
    assert report.cpis == []
    assert report.sim_partition == {}
    assert not report.passed
    exported = report.to_dict()
    assert exported["partition_comparison"] is None
    assert exported["cpi_comparisons"] == []


def test_cli_exit_codes_and_json(tmp_path):
    out = tmp_path / "report.json"
    assert main(["--sim", str(SIM), "--card", str(CARD_AGREE), "--json", str(out)]) == 0
    assert main(["--sim", str(SIM), "--card", str(CARD_COMP)]) == 1
    assert main(["--sim", str(SIM), "--decompose-only"]) == 0

    data = json.loads(out.read_text())
    assert data["e_total_limit"] == E_TOTAL_LIMIT
    assert data["e_int_limit"] == E_INT_LIMIT
    headline = data["runs"][0]["partition_comparison"]
    assert headline["passed"]
    assert headline["compensation_ratio"] >= 1.0
    assert data["runs"][0]["mechanism_labels"]["load_dep"].startswith("L1 load-use")


def test_cli_refuses_a_wormhole_artefact_rather_than_reporting_a_weaker_claim(tmp_path):
    body = json.loads(SIM.read_text())
    body["arch"] = "wormhole"
    path = tmp_path / "wh.json"
    path.write_text(json.dumps(body))
    out = tmp_path / "r.json"
    assert main(["--sim", str(path), "--decompose-only", "--json", str(out)]) == 1
    data = json.loads(out.read_text())
    gate = next(g for g in data["runs"][0]["gates"] if g["name"] == "arch_supported")
    assert not gate["passed"]
    assert "WormholeB0" in gate["detail"]


def test_cli_refuses_a_meaningless_argument_combination():
    with pytest.raises(SystemExit):
        main(["--sim", str(SIM)])
    with pytest.raises(SystemExit):
        main(["--sim", str(SIM), "--card", str(CARD_AGREE), "--decompose-only"])


def test_cli_refuses_a_core_map_that_does_not_describe_the_two_artefacts():
    with pytest.raises(SystemExit):
        main(["--sim", str(SIM), "--card", str(CARD_AGREE), "--map-core", "9,9=1,1"])


def test_parse_core_map():
    assert parse_core_map("1,2=1,1") == ((1, 2), (1, 1))
    assert parse_core_map(None) is None
    with pytest.raises(ValueError, match="SIMX,SIMY=CARDX,CARDY"):
        parse_core_map("1,2")


# ---------------------------------------------------------------------------
# The artefacts are what they say they are
# ---------------------------------------------------------------------------


def test_the_synthetic_card_files_are_labelled_as_synthetic():
    """They are hand-derived from the simulator run and are NOT measurements.
    The filename is the only thing that travels with a file someone copies out
    of the tree, so it has to carry the warning."""
    for path in (CARD_AGREE, CARD_COMP):
        assert "SYNTHETIC-NOT-A-MEASUREMENT" in path.name
        assert "SYNTHETIC" in json.loads(path.read_text())["label"]
    assert "SYNTHETIC" not in json.loads(SIM.read_text())["label"]


def test_the_checked_in_simulator_run_is_a_real_blackhole_run():
    run = load_run(SIM)
    assert run.arch == "blackhole"
    assert run.scale == 1
    # Every mechanism zone really is a different mechanism: if the bodies had
    # been folded together the cycles-per-instruction column would be flat.
    cpi = zone_cpi(run)
    measured = {k: v for k, v in cpi.items() if k != CALIBRATION_ZONE}
    assert len(set(round(v, 1) for v in measured.values())) >= 5


def test_every_zone_carries_a_structural_mechanism_label():
    """The label is what a reader has instead of a hardware counter, so a zone
    without one is a bucket nobody can interpret."""
    for zone in load_run(SIM).zones:
        assert zone.mechanism
        assert isinstance(zone, Zone)
