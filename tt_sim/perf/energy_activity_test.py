"""The counters-to-activity-vector reduction.

Small and boring on purpose: this module is the only thing standing between the
simulator's counter dataset and a regression, so what it must not do is quietly
mis-attribute a counter. Every assertion here is about a name, a unit, or a
divisor.
"""

import pytest

from tt_sim.perf.energy_activity import (
    ACTIVITY_TERMS,
    CSV_COLUMNS,
    KEY_COLUMNS,
    load_activity,
    per_launch,
    reduce_counters,
    write_row,
)


def test_the_schema_is_ordered_and_has_no_duplicates():
    assert len(set(ACTIVITY_TERMS)) == len(ACTIVITY_TERMS)
    assert CSV_COLUMNS[: len(KEY_COLUMNS)] == KEY_COLUMNS
    assert set(CSV_COLUMNS) == set(KEY_COLUMNS) | set(ACTIVITY_TERMS)


def test_counters_land_in_the_right_terms():
    totals = {
        ("2,1 BRISC", "instr_retired"): 100,
        ("2,1 NCRISC", "instr_retired"): 50,
        ("2,1 BRISC", "stall_cycles"): 7,
        ("2,1 TRISC1", "dispatch_total"): 9,
        ("2,1 MATRIX", "busy_cycles"): 500,
        ("2,1 SFPU", "busy_cycles"): 200,
        ("2,1 PACKER", "busy_cycles"): 16,
        ("2,1 NOC0", "noc_bytes_total"): 4096,
        ("2,1 NOC0", "noc_txns_timed"): 4,
        ("2,1 NOC0", "noc_flight_cycles"): 300,
        ("2,1 TRISC0", "tensix_stall_cycles"): 42,
    }
    v = reduce_counters(totals)
    assert v["instr_retired"] == 150
    assert v["rv_stall_cycles"] == 7
    assert v["tensix_dispatch"] == 9
    assert v["matrix_busy_cycles"] == 500
    assert v["sfpu_busy_cycles"] == 200
    assert v["packer_busy_cycles"] == 16
    assert v["noc_bytes_total"] == 4096
    assert v["noc_txns"] == 4
    assert v["noc_flight_cycles"] == 300
    assert v["tensix_stall_cycles"] == 42
    assert v["mover_busy_cycles"] == 0


def test_a_counter_the_schema_has_no_opinion_about_is_dropped_not_guessed():
    v = reduce_counters({("2,1 BRISC", "mem_read_l1"): 1234})
    assert set(v) == set(ACTIVITY_TERMS)
    assert all(value == 0 for value in v.values())


def test_a_backend_units_instr_retired_is_not_counted_as_risc_v():
    """``instr_retired`` is published per baby core. A Tensix backend unit that
    ever grew one must not silently double the RISC-V column."""
    v = reduce_counters({("2,1 MATRIX", "instr_retired"): 999})
    assert v["instr_retired"] == 0


def test_stall_cycles_and_tensix_stall_cycles_do_not_mix():
    """They share a unit_id on purpose (``tt_sim/trace/counters.py`` says so),
    and summing them would make both unreadable."""
    v = reduce_counters(
        {("2,1 TRISC0", "stall_cycles"): 5, ("2,1 TRISC0", "tensix_stall_cycles"): 11}
    )
    assert v["rv_stall_cycles"] == 5
    assert v["tensix_stall_cycles"] == 11


def test_per_launch_divides_and_refuses_a_zero_launch_count():
    assert per_launch({"instr_retired": 300}, 3) == {"instr_retired": 100.0}
    with pytest.raises(ValueError, match="launches must be positive"):
        per_launch({"instr_retired": 300}, 0)


def test_a_written_row_reads_back_with_its_numbers(tmp_path):
    path = tmp_path / "activity.csv"
    row = {
        "label": "mm-8",
        "arch": "blackhole",
        "arm": "mm",
        "inner": 8,
        "launches": 2,
        "cost_model": 1,
        "sim_cycles": 10499,
    }
    row.update(dict.fromkeys(ACTIVITY_TERMS, 0.0))
    row["matrix_busy_cycles"] = 533.0
    write_row(path, row, append=True)
    row2 = dict(row, label="sfpu-8", matrix_busy_cycles=342.0)
    write_row(path, row2, append=True)

    back = load_activity(path)
    assert [r["label"] for r in back] == ["mm-8", "sfpu-8"]
    assert back[0]["matrix_busy_cycles"] == 533.0
    assert back[0]["sim_cycles"] == 10499
    assert back[1]["matrix_busy_cycles"] == 342.0
