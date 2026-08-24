#!/usr/bin/env python3
"""The DRAM arm's gates, driven in BOTH directions.

Every gate here exists because its failure mode *looks like a result*. A
``--dram`` run that in fact read a worker's L1 reproduces the registered
prediction perfectly — the L1 arm is what the prediction says the DRAM arm
costs below the crossover — so "the card agrees with tt-sim" is exactly what a
broken run prints. A run taken under tt-metal's NoC instrumentation returns
numbers that are 10–19 % larger *and unequally so between the arms*, which is
the shape of the effect being measured rather than of noise.

So each gate is fed a row set built to satisfy it and a row set built to break
it. A gate that has only ever been seen passing is not known to be a gate.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import check_mode

COLUMNS = (
    "experiment,repeat,point,mst_x,mst_y,mst_node_x,mst_node_y,num_src,src0_x,src0_y,"
    "hops,num_tx,tx_bytes,dst_stride,src_stride,cycles,cycles_per_tx,"
    "outstanding_max,outstanding_end,samples,cmdbuf_avail_rest,cmdbuf_avail_busy,"
    "outstanding_rest,outstanding_delta,inflight_max,inflight_rest,trid,"
    "cmdbuf_avail_max,cmdbuf_ovfl_rest,cmdbuf_ovfl_end,"
    "mode,probe_word,sig_src,sig_witness,"
    "src_kind,src_base,dst_base,landed,landed_ok"
).split(",")

WORKER_SIG = "0x5A5A0901"
DRAM_SIG = "0xD4A5000B"
WITNESS_SIG = "0x5A5A0202"


def _row(experiment, num_tx, tx_bytes, cycles, kind="l1", **over):
    sig = DRAM_SIG if kind == "dram" else WORKER_SIG
    row = dict.fromkeys(COLUMNS, "0")
    row.update(
        {
            "experiment": experiment,
            "point": "7",
            "num_tx": str(num_tx),
            "tx_bytes": str(tx_bytes),
            "cycles": str(int(cycles)),
            "cycles_per_tx": f"{cycles / num_tx:.3f}",
            "mode": "stateless",
            "probe_word": sig,
            "sig_src": sig,
            "sig_witness": WITNESS_SIG,
            "src_kind": kind,
            "landed": sig,
            "landed_ok": "1",
        }
    )
    row.update(over)
    return row


def _sweep(experiment, tx_bytes, marginal, kind, constant=150):
    """Four burst points whose differenced marginal is exactly ``marginal``.

    Integral cycle counts, because the kernel stamps a ``uint32`` and a checker
    that accepted floats would accept a file no run can produce.
    """
    return [
        _row(experiment, n, tx_bytes, round(constant + marginal * n), kind=kind)
        for n in (4, 16, 64, 128)
    ]


def _wormhole_plan(dram_2k=86.0, l1_2k=64.0, dram_512=44.0, l1_512=44.0):
    return (
        _sweep("dramburst", 2048, dram_2k, "dram")
        + _sweep("bigburst", 2048, l1_2k, "l1")
        + _sweep("dramsize", 512, dram_512, "dram")
        + _sweep("bigburst", 512, l1_512, "l1")
    )


PLAIN = {"arch": "wormhole", "sim": "0", "noc_events": "0"}


# ---------------------------------------------------------------------------
# The marginal is a difference, and it drops the shortest burst.
# ---------------------------------------------------------------------------
def test_the_marginal_differences_the_burst_axis_and_ignores_the_constant():
    """Two plans with the same per-transaction cost and different constants must
    read the same. That is the whole reason the arm is graded on marginals: the
    launch, the prologue and the closing barrier all live in the constant."""
    cheap = check_mode.dram_marginals(_sweep("dramburst", 2048, 86.0, "dram", 10))
    dear = check_mode.dram_marginals(_sweep("dramburst", 2048, 86.0, "dram", 9000))
    assert cheap == dear
    assert cheap[("dram", 2048)] == 86.0


def test_the_marginal_keys_on_the_source_kind_as_well_as_the_size():
    """A DRAM 2 KB point and an L1 2 KB point are the comparison; folding them
    together would average away the very difference being measured."""
    marg = check_mode.dram_marginals(_wormhole_plan())
    assert marg[("dram", 2048)] == 86.0
    assert marg[("l1", 2048)] == 64.0


# ---------------------------------------------------------------------------
# The source witness. This is the gate whose failure looks like agreement.
# ---------------------------------------------------------------------------
def test_a_dram_row_that_landed_a_workers_signature_is_refused():
    rows = _sweep("dramburst", 2048, 86.0, "dram")
    rows[2]["landed"] = WORKER_SIG
    problems = check_mode.check_source_arm(rows, [])
    assert problems
    assert "SOURCE ARM DID NOT TAKE" in problems[0]


def test_a_dram_row_whose_source_is_stamped_as_a_worker_is_refused():
    """The signature *half* is the arm. A point aimed at a worker cannot become
    a DRAM measurement by landing successfully."""
    rows = _sweep("dramburst", 2048, 86.0, "dram")
    for row in rows:
        row["sig_src"] = WORKER_SIG
        row["landed"] = WORKER_SIG
    problems = check_mode.check_source_arm(rows, [])
    assert problems
    assert "other arm's half" in " ".join(problems)


def test_a_row_that_was_not_looked_at_is_not_counted_as_verified():
    """``landed_ok`` of -1 means the landing address was walked or the sources
    cycled, so the word names no tile. It must neither fail nor be counted."""
    rows = _sweep("dramburst", 2048, 86.0, "dram")
    for row in rows:
        row["landed_ok"] = "-1"
        row["landed"] = "0xDEADBEEF"
    notes = []
    assert check_mode.check_source_arm(rows, notes) == []
    assert notes == []


def test_a_good_dram_sweep_passes_the_source_witness():
    notes = []
    assert check_mode.check_source_arm(_wormhole_plan(), notes) == []
    assert "8 of them DRAM" in notes[0]


def test_a_csv_without_the_landed_column_cannot_claim_dram():
    rows = [
        {k: v for k, v in row.items() if k not in ("landed", "landed_ok")}
        for row in _sweep("dramburst", 2048, 86.0, "dram")
    ]
    problems = check_mode.check_source_arm(rows, [])
    assert problems
    assert "no `landed` column" in problems[0]


# ---------------------------------------------------------------------------
# Plain instrumentation, enforced from the artefact.
# ---------------------------------------------------------------------------
def test_the_verdict_refuses_a_file_that_cannot_show_it_was_uninstrumented(capsys):
    header = {"arch": "wormhole", "sim": "0"}  # no noc_events field at all
    assert not check_mode.dram_verdict("wormhole", {("dram", 2048): 86.0}, header)
    assert "no noc_events field" in capsys.readouterr().out


def test_an_instrumented_run_is_refused_by_report_one(tmp_path):
    path = tmp_path / "noc.csv"
    body = ["# nocreadbench arch=wormhole sim=0 dram=1 noc_events=1", ",".join(COLUMNS)]
    body += [",".join(row[c] for c in COLUMNS) for row in _wormhole_plan()]
    path.write_text("\n".join(body) + "\n")
    ok, _, _, _ = check_mode.report_one(str(path), None, quiet=True)
    assert not ok


# ---------------------------------------------------------------------------
# The three registered outcomes, and the fact that one of them is a loss.
# ---------------------------------------------------------------------------
def test_the_model_holding_is_prediction_one(capsys):
    marg = check_mode.dram_marginals(_wormhole_plan())
    assert check_mode.dram_verdict("wormhole", marg, PLAIN)
    out = capsys.readouterr().out
    assert "PREDICTION 1 -- BANDWIDTH AND NOTHING ELSE" in out
    assert "No gap" in out


def test_a_dram_excess_over_a_reproducing_l1_arm_is_prediction_two(capsys):
    """The outcome the consuming team predicts: DRAM above the figure, L1 on it."""
    marg = check_mode.dram_marginals(_wormhole_plan(dram_2k=130.0, dram_512=60.0))
    assert not check_mode.dram_verdict("wormhole", marg, PLAIN)
    out = capsys.readouterr().out
    assert "PREDICTION 2 -- THERE IS A DRAM-SIDE COST" in out


def test_an_l1_arm_that_missed_its_own_figure_stops_the_comparison(capsys):
    """If the instruction-level model both arms rest on is wrong on this part,
    the pair is not readable and no mechanism may be named."""
    marg = check_mode.dram_marginals(_wormhole_plan(l1_2k=110.0, l1_512=80.0))
    assert not check_mode.dram_verdict("wormhole", marg, PLAIN)
    out = capsys.readouterr().out
    assert "CONTROL MOVED" in out
    assert "PREDICTION 2" not in out


def test_a_gap_below_the_crossover_is_called_out_even_when_the_totals_agree(capsys):
    """The sharpest falsifier. Below the crossover tt-sim says both arms are
    issue-loop bound and therefore equal; a gap there cannot be absorbed by any
    bandwidth term, so it must be reported even though both marginals are
    inside tolerance of their own figures."""
    marg = check_mode.dram_marginals(_wormhole_plan(dram_512=48.0, l1_512=44.0))
    check_mode.dram_verdict("wormhole", marg, PLAIN)
    out = capsys.readouterr().out
    assert "A REAL GAP" in out
    assert "charged at exactly zero" in out


def test_blackhole_crosses_over_later_than_wormhole():
    """The per-architecture crossover is itself a registered claim: at 2048 B
    Wormhole is transfer-bound and Blackhole is still issue-loop bound."""
    assert (
        check_mode.DRAM_CROSSOVER["blackhole"] > check_mode.DRAM_CROSSOVER["wormhole"]
    )
    assert check_mode.SIM_DRAM_PREDICTION[("wormhole", "dram", 2048)] == 86.00
    assert check_mode.SIM_DRAM_PREDICTION[("blackhole", "dram", 2048)] == 47.00


def test_every_registered_size_has_both_arms_on_both_architectures():
    """A prediction for one arm and not the other cannot be graded: the verdict
    turns on the DIFFERENCE as much as on the absolute figures."""
    for arch in ("wormhole", "blackhole"):
        sizes = {s for a, _, s in check_mode.SIM_DRAM_PREDICTION if a == arch}
        for size in sizes:
            assert (arch, "dram", size) in check_mode.SIM_DRAM_PREDICTION
            assert (arch, "l1", size) in check_mode.SIM_DRAM_PREDICTION


# ---------------------------------------------------------------------------
# The checked-in simulator runs ARE the registered prediction. If they and the
# table ever disagree, the pre-registration has silently drifted.
# ---------------------------------------------------------------------------
def test_the_checked_in_sim_runs_reproduce_the_registered_table():
    import pathlib

    here = pathlib.Path(__file__).parent / "src"
    for arch in ("wormhole", "blackhole"):
        path = here / f"nocreadbench-{arch}-sim-dram.csv"
        if not path.exists():  # pragma: no cover - the files are checked in
            continue
        header, rows = check_mode.read_csv(str(path))
        assert header.get("noc_events") == "0"
        assert check_mode.check_source_arm(rows, []) == []
        for key, got in check_mode.dram_marginals(rows).items():
            want = check_mode.SIM_DRAM_PREDICTION[(arch, *key)]
            assert abs(got - want) < 0.01, (arch, key, got, want)


if __name__ == "__main__":  # pragma: no cover
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
