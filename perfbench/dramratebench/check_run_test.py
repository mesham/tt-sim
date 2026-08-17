#!/usr/bin/env python3
"""Every gate in ``check_run``, driven in BOTH directions.

Rows built to satisfy each gate and rows built to break it. This benchmark has
been bitten from both sides: its first card run was refused by its own tag check
because the host had tagged one slice, so the check could not pass and clean
data was thrown away; and the reason the write direction re-lays POISON before
every point is that the obvious alternative is a check that cannot FAIL, which
certifies bad data instead. A gate that has only ever been seen passing is not
known to be a gate.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import shutil

import check_run
import pytest

COLUMNS = (
    "arm,repeat,point,num_readers,num_tx,tx_bytes,bytes_per_reader,total_bytes,"
    "max_cycles,min_cycles,agg_bytes_per_cycle,agg_gb_per_s,per_reader_bytes_per_cycle,"
    "distinct_banks,distinct_dram_cores,tags_ok,max_barrier_spins,measured_readers,"
    "direction,witness_ok,stray_writes"
).split(",")


def _row(arm, n, agg, direction="read", **over):
    row = {
        "arm": arm,
        "repeat": 0,
        "point": 0,
        "num_readers": n,
        "num_tx": 16,
        "tx_bytes": 4096,
        "bytes_per_reader": 65536,
        "total_bytes": n * 65536,
        "max_cycles": 1000,
        "min_cycles": 1000,
        "agg_bytes_per_cycle": f"{agg:.4f}",
        "agg_gb_per_s": 0,
        "per_reader_bytes_per_cycle": f"{agg / n:.4f}",
        "distinct_banks": 1,
        "distinct_dram_cores": 1,
        "tags_ok": n,
        "max_barrier_spins": 7 if n > 1 else 0,
        "measured_readers": n,
        "direction": direction,
        "witness_ok": n if direction == "write" else -1,
        "stray_writes": 0 if direction == "write" else -1,
    }
    row.update(over)
    return row


def _csv(tmp_path, rows, arch="wormhole", name="run.csv"):
    path = tmp_path / name
    lines = [f"# dramratebench arch={arch} magic=0x44524232 clock_mhz=1000 banks=12"]
    lines.append(",".join(COLUMNS))
    for row in rows:
        lines.append(",".join(str(row[c]) for c in COLUMNS))
    path.write_text("\n".join(lines) + "\n")
    return path


# A run that satisfies everything: the control fans out, the concentrated arm
# does not, and every addressing check is clean.
def _good_write_rows():
    return [
        _row("onechan", 1, 21.0),
        _row("onechan", 4, 23.4),
        _row("fanchan", 1, 21.0),
        _row("fanchan", 4, 58.5),
        _row("onechan-write", 1, 21.0, "write"),
        _row("onechan-write", 4, 23.3, "write"),
        _row("fanchan-write", 1, 21.0, "write"),
        _row("fanchan-write", 4, 60.0, "write"),
    ]


def _failures(checks):
    return [line for ok, line in checks if ok is False]


# --------------------------------------------------------------------------
# The read arm's control. The one thing the write direction must not move.
# --------------------------------------------------------------------------
def test_read_control_reproduces_the_wormhole_card():
    assert check_run.READ_CONTROL_CSV.exists(), check_run.READ_CONTROL_CSV
    assert _failures(check_run.check_read_control()) == []


def test_read_control_notices_a_moved_onechan_point(tmp_path):
    """The pinned figures are checked against the file, not asserted."""
    doctored = tmp_path / "doctored.csv"
    text = check_run.READ_CONTROL_CSV.read_text()
    # 48 readers, every repeat, pushed 10% up: within a plausible band, far
    # outside the 2% this re-derivation allows.
    out = []
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) > 10 and parts[0] == "onechan" and parts[3] == "48":
            parts[10] = f"{float(parts[10]) * 1.1:.4f}"
            line = ",".join(parts)
        out.append(line)
    doctored.write_text("\n".join(out) + "\n")
    assert _failures(check_run.check_read_control(doctored))


def test_read_control_notices_a_flattened_control_arm(tmp_path):
    """A fanchan arm that stopped scaling must not pass as the control."""
    doctored = tmp_path / "flat.csv"
    out = []
    for line in check_run.READ_CONTROL_CSV.read_text().splitlines():
        parts = line.split(",")
        if len(parts) > 10 and parts[0] == "fanchan":
            parts[10] = "22.2000"
            line = ",".join(parts)
        out.append(line)
    doctored.write_text("\n".join(out) + "\n")
    assert _failures(check_run.check_read_control(doctored))


def test_read_control_needs_the_file(tmp_path):
    assert _failures(check_run.check_read_control(tmp_path / "nope.csv"))


# --------------------------------------------------------------------------
# The schema. Appending columns is safe; renaming or reordering is not.
# --------------------------------------------------------------------------
def test_schema_accepts_an_appended_column(tmp_path):
    good = _csv(tmp_path, _good_write_rows())
    assert _failures(check_run.check_schema(good)) == []


def test_schema_rejects_a_renamed_read_arm(tmp_path):
    rows = _good_write_rows()
    rows[0]["arm"] = "onechan-read"
    bad = _csv(tmp_path, rows)
    assert _failures(check_run.check_schema(bad))


def test_schema_rejects_a_reordered_prefix(tmp_path):
    good = _csv(tmp_path, _good_write_rows())
    lines = good.read_text().splitlines()
    header = lines[1].split(",")
    header[0], header[1] = header[1], header[0]
    lines[1] = ",".join(header)
    (tmp_path / "swapped.csv").write_text("\n".join(lines) + "\n")
    assert _failures(check_run.check_schema(tmp_path / "swapped.csv"))


def test_schema_rejects_a_read_row_graded_as_a_write(tmp_path):
    """A read row whose witness columns were filled is a check that cannot fail."""
    rows = _good_write_rows()
    rows[0]["witness_ok"] = rows[0]["num_readers"]
    bad = _csv(tmp_path, rows)
    assert _failures(check_run.check_schema(bad))


# --------------------------------------------------------------------------
# The write gates, each broken on its own.
# --------------------------------------------------------------------------
def test_write_gates_pass_on_a_clean_run(tmp_path):
    assert _failures(check_run.check_write(_csv(tmp_path, _good_write_rows()))) == []


def test_write_needs_write_rows(tmp_path):
    reads = [r for r in _good_write_rows() if r["direction"] == "read"]
    assert _failures(check_run.check_write(_csv(tmp_path, reads)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("measured_readers", 3),  # a writer never stamped a result
        ("tags_ok", 3),  # a write was dropped or mis-congruent
        ("witness_ok", 3),  # the bytes are not in the block the plan named
        ("stray_writes", 1),  # bytes landed on a block it did not name
    ],
)
def test_each_addressing_gate_can_fail(tmp_path, field, value):
    rows = _good_write_rows()
    target = next(
        r for r in rows if r["arm"] == "onechan-write" and r["num_readers"] == 4
    )
    target[field] = value
    assert _failures(check_run.check_write(_csv(tmp_path, rows)))


def test_barrier_gate_can_fail(tmp_path):
    """N writers that never overlapped give exactly the flat curve we look for."""
    rows = _good_write_rows()
    for row in rows:
        if row["direction"] == "write":
            row["max_barrier_spins"] = 0
    assert any(
        "waited at the barrier" in line
        for line in _failures(check_run.check_write(_csv(tmp_path, rows)))
    )


def test_fanout_control_gate_can_fail(tmp_path):
    """A control that did not move makes the concentrated arm unreadable."""
    rows = _good_write_rows()
    next(r for r in rows if r["arm"] == "fanchan-write" and r["num_readers"] == 4)[
        "agg_bytes_per_cycle"
    ] = "23.5000"
    assert any(
        "fan-out CONTROL moved" in line
        for line in _failures(check_run.check_write(_csv(tmp_path, rows)))
    )


def test_a_single_writer_count_is_degenerate(tmp_path):
    rows = [r for r in _good_write_rows() if r["num_readers"] == 4]
    assert any(
        "no low/high pair" in line
        for line in _failures(check_run.check_write(_csv(tmp_path, rows)))
    )


def test_endpoint_bound_is_read_as_a_ratio_not_a_threshold(tmp_path):
    """The concentrated arm may scale a long way and still be bound.

    tt-sim's own Wormhole numbers are the case: onechan x2.21 against fanchan
    x3.97 is the endpoint costing 44% of the available scaling, and an absolute
    "onechan must stay under x1.5" rule calls that a refutation.
    """
    rows = _good_write_rows()
    for arm, lo, hi in (("onechan-write", 21.0, 46.4), ("fanchan-write", 21.0, 83.4)):
        next(r for r in rows if r["arm"] == arm and r["num_readers"] == 1)[
            "agg_bytes_per_cycle"
        ] = f"{lo:.4f}"
        next(r for r in rows if r["arm"] == arm and r["num_readers"] == 4)[
            "agg_bytes_per_cycle"
        ] = f"{hi:.4f}"
    lines = [line for _, line in check_run.check_write(_csv(tmp_path, rows))]
    assert any("-> ENDPOINT BOUND" in line for line in lines), lines


def test_blackhole_write_ceiling_is_reported_as_unsourced(tmp_path):
    """No source gives a Blackhole DRAM write rate, and the run must say so."""
    good = _csv(tmp_path, _good_write_rows(), arch="blackhole", name="bh.csv")
    lines = [line for _, line in check_run.check_write(good)]
    assert any("unsourced" in line for line in lines), lines


# --------------------------------------------------------------------------
# The whole entry point, since the exit code is what a card operator sees.
# --------------------------------------------------------------------------
def test_main_returns_nonzero_on_a_bad_run(tmp_path, capsys):
    rows = _good_write_rows()
    next(r for r in rows if r["arm"] == "onechan-write" and r["num_readers"] == 4)[
        "stray_writes"
    ] = 2
    assert check_run.main(["--measured", str(_csv(tmp_path, rows))]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_passes_a_clean_run(tmp_path, capsys):
    path = _csv(tmp_path, _good_write_rows())
    shutil.copy(path, tmp_path / "ref.csv")
    assert (
        check_run.main(
            ["--measured", str(path), "--schema-against", str(tmp_path / "ref.csv")]
        )
        == 0
    )
    assert "PASS" in capsys.readouterr().out
