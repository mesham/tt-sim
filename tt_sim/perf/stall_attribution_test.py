"""Tests for the mechanism-attribution comparison.

Two things are being defended here.

**The arithmetic**: the partition closes identically on both sides, ``E_total``
really is the span error, and the triangle inequality ``E_total <= E_int`` holds
so that the compensation ratio is always at least 1. If any of those slips, the
ratio stops being a measurement of compensation and becomes decoration.

**The gates, in both directions**. A guard that cannot fail is as damaging as
one that cannot pass, so every gate gets a case that passes it and a case that
refuses it, and the refusing case is constructed from an input a real session
could plausibly produce -- a concatenated log, a Wormhole-numbered core compared
against a Blackhole-numbered one, an unarmed bank, a partition that does not
close.

Two of the gate's refusals are now reachable from tt-sim's own machinery rather
than only from a hand-written CSV: :func:`_drive_perf_counters` runs a window on
a real :class:`~tt_sim.misc.perf_counters.TensixPerfCounters` and reads it back
through the MMIO path a profiler would use, in both directions.

Two kinds of card data appear here. ``CARD_WH_ELW`` and ``CARD_WH_ELW_BLOCK``
are **real**, transcribed from
``perfbench/card-sessions/2026-08-17-wh-mechbench-wormhole/`` -- corroboration,
never provenance, and no number in them may become a cost. Everything else is
**synthetic**: every file this module writes, and the two it reads out of
``perfbench/mechbench/testdata/`` whose names say so, are stamped
NOT-A-MEASUREMENT in the directory's README and must never be quoted as one.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tt_sim.perf import stall_attribution as sa

REPO = Path(__file__).resolve().parents[2]
TESTDATA = REPO / "perfbench" / "mechbench" / "testdata"

#: The counters tt-sim actually produced for ``mechbench elw 8`` on Blackhole,
#: 2026-08-13. Used as the shape every synthetic case is built from, so that a
#: test which passes is a test against a realistic decomposition rather than
#: against a hand-chosen one.
ELW_REAL = {
    "ref_cnt": 631,
    "THREAD_STALLS_0": 0,
    "THREAD_STALLS_1": 401,
    "THREAD_STALLS_2": 168,
    "THREAD_INSTRUCTIONS_0": 110,
    "THREAD_INSTRUCTIONS_1": 124,
    "THREAD_INSTRUCTIONS_2": 195,
    "WAITING_FOR_NONZERO_SEM_0": 0,
    "WAITING_FOR_NONZERO_SEM_1": 0,
    "WAITING_FOR_NONZERO_SEM_2": 158,
    "WAITING_FOR_NONFULL_SEM_0": 0,
    "WAITING_FOR_NONFULL_SEM_1": 0,
    "WAITING_FOR_NONFULL_SEM_2": 0,
    "WAITING_FOR_SRCA_VALID": 394,
    "WAITING_FOR_SRCB_VALID": 5,
    "WAITING_FOR_SRCA_CLEAR": 0,
    "WAITING_FOR_SRCB_CLEAR": 0,
}


# ---------------------------------------------------------------------------
# Building synthetic device-profiler logs
# ---------------------------------------------------------------------------


def write_log(path, cores, arch="blackhole"):
    """Write a device-profiler log carrying the given counters.

    ``cores`` is ``{(x, y): {"ref_cnt": N, "run": id, <counter>: value, ...}}``,
    or a list of such dicts per core when a case needs two windows in one file.
    The format is byte-compatible with what ``tt_metal/impl/profiler/profiler.cpp``
    emits, including the semicolons it substitutes for the commas inside the
    JSON meta-data cell, because the whole point of this comparison is that both
    sides are read by one parser.
    """
    lines = [
        f"ARCH: {arch}, CHIP_FREQ[MHz]: 0, Max Compute Cores: 140",
        "PCIe slot, core_x, core_y, RISC processor type, timer_id, "
        "time[cycles since reset], data, run host ID, trace id, trace id counter, "
        "zone name, type, source line, source file, meta data",
    ]
    stamp = 1000
    for core, windows in sorted(cores.items()):
        if isinstance(windows, dict):
            windows = [windows]
        for window in windows:
            ref = window["ref_cnt"]
            run = window.get("run", 0)
            # A real log carries the firmware's own zone markers too; include
            # one so the loader is exercised on a file it must skip rows in.
            lines.append(
                f"0,{core[0]},{core[1]},BRISC,45671,{stamp},0,{run},,,BRISC-FW,ZONE_START,433,brisc.cc,"
            )
            stamp += 1
            for name, value in window.items():
                if name in ("ref_cnt", "run"):
                    continue
                lines.append(
                    f"0,{core[0]},{core[1]},BRISC,9090,{stamp},0,{run},,,,TS_DATA,0,,"
                    f'{{"counter type":"{name}";"ref cnt":{ref};"value":{value}}}'
                )
                stamp += 1
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)


def counters(core=(1, 2), **overrides):
    """One core's counter dict, ELW_REAL with fields replaced."""
    values = dict(ELW_REAL)
    values.update(overrides)
    return {core: values}


def load(tmp_path, name, cores):
    return sa.load_counter_samples(write_log(tmp_path / name, cores))


def one_core(tmp_path, name, cores):
    grouped = sa.group_by_core(load(tmp_path, name, cores))
    assert len(grouped) == 1
    return next(iter(grouped.values()))


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def test_loads_only_counter_markers(tmp_path):
    samples = load(tmp_path, "log.csv", counters())
    assert len(samples) == len(ELW_REAL) - 1  # ref_cnt is not a counter
    assert all(s.core == (1, 2) for s in samples)
    assert all(s.ref_cnt == 631 for s in samples)
    by_name = {s.counter: s.value for s in samples}
    assert by_name["THREAD_STALLS_1"] == 401
    assert by_name["WAITING_FOR_SRCA_VALID"] == 394


def test_rejects_a_file_that_is_not_a_profiler_log(tmp_path):
    path = tmp_path / "nope.csv"
    path.write_text("label,cycle,power_w\nbaseline,0,12.0\n")
    with pytest.raises(ValueError, match="not a tt-metal device profiler log"):
        sa.load_counter_samples(path)


def test_reads_the_real_simulator_log():
    """The checked-in tt-sim log parses and closes -- no synthetic shortcut."""
    samples = sa.load_counter_samples(TESTDATA / "sim-elw-blackhole.csv")
    grouped = sa.group_by_core(samples)
    assert list(grouped) == [(1, 2)]
    core = grouped[(1, 2)]
    assert core.ref_cnt == ELW_REAL["ref_cnt"]
    for name in sa.required_counters():
        assert core.get(name) == ELW_REAL[name], name


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def test_core_partition_closes_on_real_simulator_counters(tmp_path):
    core = one_core(tmp_path, "log.csv", counters())
    part = sa.core_partition(core)
    ok, reasons = sa.partition_closes(part, 3 * core.ref_cnt)
    assert ok, reasons
    assert sum(part.values()) == 3 * 631
    assert set(part) == set(sa.CORE_MECHANISMS)


def test_thread_partitions_close_on_real_simulator_counters(tmp_path):
    core = one_core(tmp_path, "log.csv", counters())
    for t in range(3):
        part = sa.thread_partition(core, t)
        ok, reasons = sa.partition_closes(part, core.ref_cnt)
        assert ok, (t, reasons)
        assert sum(part.values()) == 631


def test_partition_closure_is_not_vacuous(tmp_path):
    """The closure check must reject a partition that does not close."""
    core = one_core(tmp_path, "log.csv", counters(WAITING_FOR_SRCA_VALID=10_000))
    ok, reasons = sa.partition_closes(sa.core_partition(core), 3 * core.ref_cnt)
    assert not ok
    assert any("unattributed_stall" in r for r in reasons)


# ---------------------------------------------------------------------------
# The criterion
# ---------------------------------------------------------------------------


def test_e_total_is_the_span_error_and_e_int_bounds_it(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    hw = one_core(
        tmp_path,
        "hw.csv",
        counters(
            ref_cnt=660,
            THREAD_STALLS_1=430,
            WAITING_FOR_SRCA_VALID=410,
            WAITING_FOR_SRCB_VALID=8,
        ),
    )
    cmp_ = sa.compare_partitions(
        "core", sa.CORE_MECHANISMS, sa.core_partition(sim), sa.core_partition(hw)
    )
    assert cmp_.e_total == pytest.approx(abs(631 - 660) / 660)
    # Triangle inequality: the interior error can never be smaller than the
    # total, so the compensation ratio is never below 1.
    assert cmp_.e_int >= cmp_.e_total
    assert cmp_.ratio >= 1.0


def test_perfect_agreement_has_no_compensation_ratio(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    hw = one_core(tmp_path, "hw.csv", counters())
    cmp_ = sa.compare_partitions(
        "core", sa.CORE_MECHANISMS, sa.core_partition(sim), sa.core_partition(hw)
    )
    assert cmp_.e_total == 0.0
    assert cmp_.e_int == 0.0
    assert cmp_.ratio is None
    assert cmp_.passed


def test_a_matching_total_with_a_wrong_interior_fails(tmp_path):
    """The case the whole leg exists for.

    Same span, so ``E_total`` is exactly zero and every envelope check in this
    repo would pass. The mass has simply moved from ``srca_valid`` into the
    unattributed bucket, which is precisely the compensating error an envelope
    cannot see.
    """
    sim = one_core(tmp_path, "sim.csv", counters())
    hw = one_core(
        tmp_path,
        "hw.csv",
        counters(WAITING_FOR_SRCA_VALID=0, WAITING_FOR_SRCB_VALID=0),
    )
    cmp_ = sa.compare_partitions(
        "core", sa.CORE_MECHANISMS, sa.core_partition(sim), sa.core_partition(hw)
    )
    assert cmp_.e_total == 0.0
    assert cmp_.e_int > sa.E_INT_LIMIT
    assert cmp_.ratio == float("inf")
    assert not cmp_.passed


# ---------------------------------------------------------------------------
# The gates -- each one proven to fire in both directions
# ---------------------------------------------------------------------------


def test_gate_counters_present_passes(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    assert sa.gate_counters_present(sim, sim).passed


def test_gate_counters_present_refuses_a_missing_counter(tmp_path):
    short = dict(ELW_REAL)
    del short["WAITING_FOR_SRCB_CLEAR"]
    sim = one_core(tmp_path, "sim.csv", {(1, 2): short})
    good = one_core(tmp_path, "hw.csv", counters())
    gate = sa.gate_counters_present(sim, good)
    assert not gate.passed
    assert "WAITING_FOR_SRCB_CLEAR" in gate.detail


def test_gate_armed_passes(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    assert sa.gate_armed(sim, sim).passed


def test_gate_armed_refuses_an_all_zero_bank(tmp_path):
    zeros = {k: 0 for k in ELW_REAL}
    zeros["ref_cnt"] = 4096  # the bank ran; nothing ever incremented
    dead = one_core(tmp_path, "sim.csv", {(1, 2): zeros})
    good = one_core(tmp_path, "hw.csv", counters())
    gate = sa.gate_armed(dead, good)
    assert not gate.passed
    assert "never armed" in gate.detail


def test_gate_armed_refuses_a_zero_reference_count(tmp_path):
    dead = one_core(tmp_path, "sim.csv", counters(ref_cnt=0))
    good = one_core(tmp_path, "hw.csv", counters())
    gate = sa.gate_armed(dead, good)
    assert not gate.passed
    assert "ref_cnt = 0" in gate.detail


def test_gate_single_window_passes(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    assert sa.gate_single_window(sim, sim).passed


def test_gate_single_window_refuses_two_concatenated_runs(tmp_path):
    two = {
        (1, 2): [
            dict(ELW_REAL, run=0),
            dict(ELW_REAL, ref_cnt=700, run=1),
        ]
    }
    both = one_core(tmp_path, "sim.csv", two)
    good = one_core(tmp_path, "hw.csv", counters())
    gate = sa.gate_single_window(both, good)
    assert not gate.passed
    assert "ref_cnt" in gate.detail or "run host ID" in gate.detail


def test_gate_per_core_passes(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    assert sa.gate_per_core(sim, sim).passed


def test_gate_per_core_refuses_a_pooled_unit(tmp_path):
    """A unit assembled from two coordinates is not a span at all."""
    sim = one_core(tmp_path, "sim.csv", counters())
    sim.cores_seen.add((2, 2))
    good = one_core(tmp_path, "hw.csv", counters())
    gate = sa.gate_per_core(sim, good)
    assert not gate.passed
    assert "pools cores" in gate.detail


def test_gate_per_core_refuses_a_silent_coordinate_mismatch(tmp_path):
    """Wormhole worker (1,1) against Blackhole worker (1,2) -- the easy mistake."""
    sim = one_core(tmp_path, "sim.csv", counters(core=(1, 1)))
    hw = one_core(tmp_path, "hw.csv", counters(core=(1, 2)))
    gate = sa.gate_per_core(sim, hw)
    assert not gate.passed
    assert "--map-core" in gate.detail
    # ... and passes once the correspondence is stated out loud.
    assert sa.gate_per_core(sim, hw, mapped=True).passed


def test_gate_partition_closes_passes(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    assert sa.gate_partition_closes(sim, sim).passed


def test_gate_partition_closes_refuses_a_negative_unattributed_bucket(tmp_path):
    """Named stall reasons out-counting THREAD_STALLS -- this leg's finding."""
    sim = one_core(tmp_path, "sim.csv", counters())
    bad = one_core(tmp_path, "hw.csv", counters(WAITING_FOR_SRCA_VALID=10_000))
    gate = sa.gate_partition_closes(sim, bad)
    assert not gate.passed
    assert "unattributed_stall" in gate.detail


def test_gate_partition_closes_refuses_a_negative_idle_bucket(tmp_path):
    sim = one_core(tmp_path, "sim.csv", counters())
    bad = one_core(tmp_path, "hw.csv", counters(THREAD_STALLS_0=10_000))
    gate = sa.gate_partition_closes(sim, bad)
    assert not gate.passed
    assert "idle_0" in gate.detail


# ---------------------------------------------------------------------------
# The semaphore-counter overlap -- the 2026-08-17 Wormhole finding
#
# The partition treats WAITING_FOR_{NONZERO,NONFULL}_SEM_t as disjoint
# sub-counts of THREAD_STALLS_t. On silicon they are not: they belong to the
# nine-counter per-thread stall-reason block, and per SEMWAIT.md/WaitGate.md a
# latched wait condition keeps counting while the thread has nothing at the gate
# to hold. These tests defend the refusal in both directions: it must fire, it
# must name the counters, and it must NOT fire on a partition that does close.
# ---------------------------------------------------------------------------


#: The real Wormhole card counters for ``mechbench elw 8``, core (1,1),
#: 2026-08-17 (``perfbench/card-sessions/2026-08-17-wh-mechbench-wormhole/``,
#: run ``instrn-elw-1``). Corroboration, never provenance: no number here may
#: become a cost. Only the fields the diagnosis reads are transcribed.
CARD_WH_ELW = {
    "ref_cnt": 4317,
    "THREAD_STALLS_0": 367,
    "THREAD_STALLS_1": 36,
    "THREAD_STALLS_2": 270,
    "THREAD_INSTRUCTIONS_0": 126,
    "THREAD_INSTRUCTIONS_1": 143,
    "THREAD_INSTRUCTIONS_2": 136,
    "WAITING_FOR_NONZERO_SEM_0": 0,
    "WAITING_FOR_NONZERO_SEM_1": 0,
    "WAITING_FOR_NONZERO_SEM_2": 3561,
    "WAITING_FOR_NONFULL_SEM_0": 0,
    "WAITING_FOR_NONFULL_SEM_1": 7,
    "WAITING_FOR_NONFULL_SEM_2": 0,
    "WAITING_FOR_SRCA_VALID": 0,
    "WAITING_FOR_SRCB_VALID": 0,
    "WAITING_FOR_SRCA_CLEAR": 0,
    "WAITING_FOR_SRCB_CLEAR": 0,
}

#: The other seven counters of the same block, same run. Present so the
#: diagnosis can quote the vendor's own overlap factor rather than only the
#: semaphore excess.
CARD_WH_ELW_BLOCK = {
    "WAITING_FOR_THCON_IDLE_0": 0,
    "WAITING_FOR_UNPACK_IDLE_0": 367,
    "WAITING_FOR_PACK_IDLE_0": 0,
    "WAITING_FOR_MATH_IDLE_0": 0,
    "WAITING_FOR_MOVE_IDLE_0": 0,
    "WAITING_FOR_MMIO_IDLE_0": 6,
    "WAITING_FOR_SFPU_IDLE_0": 0,
    "WAITING_FOR_THCON_IDLE_1": 0,
    "WAITING_FOR_UNPACK_IDLE_1": 0,
    "WAITING_FOR_PACK_IDLE_1": 0,
    "WAITING_FOR_MATH_IDLE_1": 38,
    "WAITING_FOR_MOVE_IDLE_1": 0,
    "WAITING_FOR_MMIO_IDLE_1": 0,
    "WAITING_FOR_SFPU_IDLE_1": 38,
    "WAITING_FOR_THCON_IDLE_2": 0,
    "WAITING_FOR_UNPACK_IDLE_2": 0,
    "WAITING_FOR_PACK_IDLE_2": 270,
    "WAITING_FOR_MATH_IDLE_2": 0,
    "WAITING_FOR_MOVE_IDLE_2": 0,
    "WAITING_FOR_MMIO_IDLE_2": 0,
    "WAITING_FOR_SFPU_IDLE_2": 0,
}


def card_wh(core=(1, 1), block=True, **overrides):
    values = dict(CARD_WH_ELW)
    if block:
        values.update(CARD_WH_ELW_BLOCK)
    values.update(overrides)
    return {core: values}


def test_stall_reason_overlap_reproduces_the_vendor_metric(tmp_path):
    """Metric 36 -- ``sum(all 9 WAITING_FOR_*_N) / THREAD_STALLS_N``."""
    card = one_core(tmp_path, "card.csv", card_wh())
    overlap = sa.stall_reason_overlap(card, 2)
    assert overlap["thread_stalls"] == 270
    assert overlap["sem"] == 3561
    assert overlap["sem_excess"] == 3291
    assert overlap["block_sum"] == 3561 + 270
    assert overlap["block_factor"] == pytest.approx(3831 / 270)


def test_stall_reason_overlap_declines_the_factor_on_a_partial_log(tmp_path):
    """A missing counter read as zero would *understate* the overlap.

    That is the one direction a diagnosis must never err in, so the factor is
    withheld rather than computed from the two counters that are required.
    """
    card = one_core(tmp_path, "card.csv", card_wh(block=False))
    overlap = sa.stall_reason_overlap(card, 2)
    assert overlap["block_sum"] is None
    assert overlap["block_factor"] is None
    # ...but the part built from required counters is still reported.
    assert overlap["sem_excess"] == 3291


def test_sem_overlap_findings_are_silent_when_the_counters_are_disjoint(tmp_path):
    """The direction that matters most: this must not fire on a closing side."""
    sim = one_core(tmp_path, "sim.csv", counters())
    assert sa.sem_overlap_findings("sim", sim) == []


def test_sem_overlap_findings_name_the_counters_on_real_card_data(tmp_path):
    card = one_core(tmp_path, "card.csv", card_wh())
    findings = sa.sem_overlap_findings("card", card)
    assert len(findings) == 1
    assert "WAITING_FOR_NONZERO_SEM_2 = 3561" in findings[0]
    assert "WAITING_FOR_NONFULL_SEM_2 = 0" in findings[0]
    assert "THREAD_STALLS_2 = 270" in findings[0]
    assert "3291 cycles" in findings[0]
    assert "14.19x" in findings[0]


def test_gate_partition_closes_names_the_semaphore_counters(tmp_path):
    """The refusal must be specific, not a bare negative bucket.

    A generic "unattributed_stall = -2895 (negative)" is the arithmetic; a
    reader six months from now needs the reason, which is that two named
    counters are not the quantity the partition took them for.
    """
    sim = one_core(tmp_path, "sim.csv", counters(core=(1, 1)))
    card = one_core(tmp_path, "card.csv", card_wh())
    gate = sa.gate_partition_closes(sim, card)
    assert not gate.passed
    assert "WAITING_FOR_NONZERO_SEM_2" in gate.detail
    assert "THREAD_STALLS_2" in gate.detail
    # the arithmetic is kept as the evidence, not replaced by the explanation
    assert "unattributed_stall = -2895" in gate.detail
    assert "other_stall = -3291" in gate.detail
    assert "SEMWAIT.md" in gate.detail
    # and the line is broken up rather than emitted as one unreadable run
    assert gate.line().count("\n") >= 3


def test_the_refusal_is_not_widened_away(tmp_path):
    """No tolerance, no clamp: a negative bucket stays negative and refuses."""
    card = one_core(tmp_path, "card.csv", card_wh())
    part = sa.core_partition(card)
    assert part["unattributed_stall"] == -2895
    ok, reasons = sa.partition_closes(part, 3 * card.ref_cnt)
    assert not ok
    assert reasons
    thread = sa.thread_partition(card, 2)
    assert thread["other_stall"] == -3291


def test_no_real_simulator_log_has_ever_failed_this_gate(tmp_path):
    """Why Blackhole's pass is not evidence about Blackhole.

    ``TensixPerfCounters.note_stall`` increments ``thread_stalls[t]`` and at
    most one reason bucket in the same call, and it is tt-sim's *only* stall
    hook that the Tensix front end calls, so ``sem_empty_t + sem_full_t <=
    thread_stalls_t`` holds on every log tt-sim has ever produced, on either
    architecture. Both checked-in simulator logs are pinned here to make that
    concrete: the gate closing on them says nothing about silicon.

    This is a statement about the front end's call sites, not about the counter
    model -- which can now represent the hardware's behaviour, and is driven
    into it two tests below.
    """
    for name in ("sim-elw-blackhole.csv", "sim-mm-blackhole.csv"):
        core = next(
            iter(sa.group_by_core(sa.load_counter_samples(TESTDATA / name)).values())
        )
        for t in range(3):
            overlap = sa.stall_reason_overlap(core, t)
            assert overlap["sem_excess"] <= 0, (name, t, overlap)
        assert sa.sem_overlap_findings("sim", core) == []


def test_note_stall_makes_the_partition_close_by_construction():
    """The invariant above, straight from the counter model rather than a log."""
    from tt_sim.misc.perf_counters import TensixPerfCounters

    counters_ = TensixPerfCounters(blackhole=True)
    for reason, bank in (
        ("semaphore_empty", None),
        ("semaphore_full", None),
        ("src_reserved_by_unpacker", "A"),
        ("src_reserved_by_matrix", "B"),
        ("something_else", None),
    ):
        for _ in range(7):
            counters_.note_stall(2, reason, src_bank=bank)
    assert counters_.thread_stalls[2] == 35
    named = (
        counters_.sem_empty[2]
        + counters_.sem_full[2]
        + sum(counters_.src_valid.values())
        + sum(counters_.src_clear.values())
    )
    assert named <= counters_.thread_stalls[2]


# ---------------------------------------------------------------------------
# Metric 36, tried as a correction and rejected
#
# The hardware documents the overlap (tech report metric 36, "Stall Cause
# Overlap Factor"), so the tempting move is to stop assuming disjointness and
# subtract a measured overlap. These pin the two reasons that cannot be argued
# around: the factor's own documented mechanism is falsified by the card, and
# using it as a divisor hands back the assumption it was meant to test.
# ---------------------------------------------------------------------------


def test_a_single_reason_alone_out_counts_the_thread_stalls_on_card_data(tmp_path):
    """The measurement that falsifies metric 36's documented explanation.

    "Values >1.0 mean multiple stall conditions are active simultaneously" can
    inflate the *sum* of the nine reasons above ``THREAD_STALLS_t``. It can
    never lift any one of them above it -- a counter is not simultaneous with
    itself. On the 2026-08-17 Wormhole part, three do.
    """
    card = one_core(tmp_path, "card.csv", card_wh())
    t2 = sa.stall_reason_overlap(card, 2)
    assert t2["max_reason"] == "WAITING_FOR_NONZERO_SEM_2"
    assert t2["max_reason_value"] == 3561
    assert t2["max_reason_excess"] == 3561 - 270
    # thread 1's pair of unit-idle counters do it too, on much smaller numbers,
    # so this is not one anomalous counter on one thread.
    t1 = sa.stall_reason_overlap(card, 1)
    assert t1["max_reason_value"] == 38
    assert t1["max_reason_excess"] == 38 - 36
    # ...and the direction that matters: it must not fire where it should not.
    assert sa.stall_reason_overlap(card, 0)["max_reason_excess"] == 0


def test_metric36_as_a_correction_hands_back_the_assumption_it_should_test(tmp_path):
    """Dividing a bucket by a factor that contains it is not a correction.

    ``corrected(sem) = sem * stalls / (sem + other)`` tends to ``stalls`` as
    ``sem`` grows. So the "corrected" semaphore bucket converges on exactly the
    number the partition already assumed it could be at most, and it gets there
    almost immediately: doubling the measured counter moves the answer by under
    4 %.
    """
    card = one_core(tmp_path, "card.csv", card_wh())
    fit = sa.metric36_as_correction(card, 2, hypothetical_sem=2 * 3561)
    assert fit["factor"] == pytest.approx(3831 / 270)
    assert fit["corrected_sem"] == pytest.approx(3561 * 270 / 3831)
    assert fit["corrected_sem"] == pytest.approx(250.97, abs=0.01)
    assert fit["corrected_hypothetical"] == pytest.approx(260.14, abs=0.01)
    # a 100 % change in the input for under 4 % in the output
    moved = abs(fit["corrected_hypothetical"] - fit["corrected_sem"])
    assert moved / fit["corrected_sem"] < 0.04
    # and the limit is the thread's own stall count, i.e. the assumption itself
    assert fit["limit"] == 270
    assert sa.metric36_as_correction(card, 2, hypothetical_sem=10**9)[
        "corrected_hypothetical"
    ] == pytest.approx(270, abs=0.01)


def test_metric36_as_a_correction_is_withheld_without_the_whole_block(tmp_path):
    """No factor, no correction -- rather than one computed from two of nine."""
    partial = one_core(tmp_path, "card.csv", card_wh(block=False))
    assert sa.metric36_as_correction(partial, 2) is None


def test_metric36_as_a_correction_is_withheld_when_a_thread_never_stalled(tmp_path):
    card = one_core(tmp_path, "card.csv", card_wh(THREAD_STALLS_0=0))
    assert sa.metric36_as_correction(card, 0) is None


# ---------------------------------------------------------------------------
# The gate must be reachable from tt-sim's own machinery, not only from a CSV
#
# ``partition_closes`` refused on silicon and has never refused on a simulator
# log, because the front end's one stall hook couples the reason buckets to the
# total. ``note_wait_condition`` is the counting rule ``SEMWAIT.md`` actually
# describes -- a latched condition is re-evaluated every cycle whether or not
# anything is held at the gate -- and these drive ``TensixPerfCounters``
# through its own hooks and its own MMIO readback, both directions.
# ---------------------------------------------------------------------------


def _drive_perf_counters(overlapping):
    """Run a window on a real ``TensixPerfCounters`` and read it back by MMIO.

    Wormhole selects, to mirror the part the refusal came from. Everything goes
    through the register interface a profiler would use -- start edge, counting
    hooks, stop edge, then one ``mode``/``OUT_H`` pair per counter -- so what
    comes back has been through the same decode path as a real readback rather
    than lifted off the attributes.
    """
    from tt_sim.misc import perf_counters as pc

    counters_ = pc.TensixPerfCounters(blackhole=False)
    base, out_l, out_h = pc.BANK_REGISTERS["INSTRN_THREAD"]
    counters_.write(base + 8, 1, 0)  # start edge: begins the window and clears
    assert counters_.instrn_running

    for _ in range(10):
        counters_.note_stall(0, "src_reserved_by_unpacker", src_bank="A")
    for _ in range(5):
        counters_.note_stall(1, "semaphore_full")
    for _ in range(30):
        counters_.note_stall(2, "semaphore_empty")
    if overlapping:
        # The cycles hardware counts and tt-sim's front end never visits: the
        # latched condition is unsatisfied but nothing is held at the gate.
        for _ in range(250):
            counters_.note_wait_condition(2, "semaphore_empty")
    for thread in range(3):
        for _ in range(20):
            counters_.note_dispatch(thread)

    counters_.write(base + 8, 2, 400)  # stop edge
    assert not counters_.instrn_running

    selects = {name: key for key, name in pc._instrn_selects(False).items()}
    values = {"ref_cnt": counters_.read(out_l, 400)}
    for name in sa.required_counters():
        select, grant = selects[name]
        counters_.write(base + 4, (select << 8) | (0x10000 if grant else 0), 400)
        values[name] = counters_.read(out_h, 400)
    return values


def test_the_counter_model_can_represent_the_hardware_overlap():
    """``note_wait_condition`` counts a reason without counting a stall.

    That is the whole of the fix at this level: the class no longer enforces an
    invariant silicon breaks. It also refuses the four Src conditions, which are
    per-instruction ownership tests and cannot run with nothing at the gate.
    """
    from tt_sim.misc.perf_counters import TensixPerfCounters

    counters_ = TensixPerfCounters(blackhole=False)
    counters_.note_stall(2, "semaphore_empty")
    for _ in range(9):
        counters_.note_wait_condition(2, "semaphore_empty")
    counters_.note_wait_condition(1, "semaphore_full")
    assert counters_.thread_stalls == [0, 0, 1]
    assert counters_.sem_empty[2] == 10
    assert counters_.sem_full[1] == 1
    with pytest.raises(ValueError, match="not a latched wait condition"):
        counters_.note_wait_condition(0, "src_reserved_by_unpacker")


def test_the_gate_refuses_a_window_the_counter_model_itself_produced(tmp_path):
    """The refusal path, reached from tt-sim's machinery rather than a CSV.

    A gate that cannot fail is not a gate. Before ``note_wait_condition`` no
    sequence of calls on ``TensixPerfCounters`` could produce a set this gate
    rejects, so its passing said nothing. Now one can, and it does.
    """
    values = _drive_perf_counters(overlapping=True)
    assert values["ref_cnt"] == 400
    assert values["WAITING_FOR_NONZERO_SEM_2"] == 280
    assert values["THREAD_STALLS_2"] == 30

    core = one_core(tmp_path, "sim.csv", {(1, 1): values})
    gate = sa.gate_partition_closes(core, None)
    assert not gate.passed
    assert "WAITING_FOR_NONZERO_SEM_2 = 280" in gate.detail
    assert "THREAD_STALLS_2 = 30" in gate.detail
    assert "unattributed_stall = -250" in gate.detail
    assert "SEMWAIT.md" in gate.detail

    # and end to end, through the same entry point the CLI uses
    reports = sa.analyse(sa.load_counter_samples(tmp_path / "sim.csv"), None)
    assert len(reports) == 1
    assert reports[0].refused


def test_the_gate_passes_a_window_the_counter_model_itself_produced(tmp_path):
    """The other direction, from the same machinery: disjoint counters close.

    Without the un-held cycles this is exactly what tt-sim produces today, and
    it must keep passing -- otherwise the test above would be demonstrating a
    broken gate rather than a reachable refusal.
    """
    values = _drive_perf_counters(overlapping=False)
    assert values["WAITING_FOR_NONZERO_SEM_2"] == 30

    core = one_core(tmp_path, "sim.csv", {(1, 1): values})
    gate = sa.gate_partition_closes(core, None)
    assert gate.passed, gate.detail
    assert sa.sem_overlap_findings("sim", core) == []
    assert sa.core_partition(core)["unattributed_stall"] == 0

    reports = sa.analyse(sa.load_counter_samples(tmp_path / "sim.csv"), None)
    assert len(reports) == 1
    assert not reports[0].refused


# ---------------------------------------------------------------------------
# analyse() and the report
# ---------------------------------------------------------------------------


def test_analyse_refuses_a_card_log_missing_the_core(tmp_path):
    sim = load(tmp_path, "sim.csv", counters(core=(1, 2)))
    hw = load(tmp_path, "hw.csv", counters(core=(4, 4)))
    reports = sa.analyse(sim, hw)
    assert len(reports) == 1
    assert reports[0].refused
    assert "has no core" in reports[0].gates[-1].detail


def test_analyse_reports_core_and_per_thread_comparisons(tmp_path):
    sim = load(tmp_path, "sim.csv", counters())
    hw = load(tmp_path, "hw.csv", counters())
    reports = sa.analyse(sim, hw)
    assert len(reports) == 1
    units = [c.unit for c in reports[0].comparisons]
    assert units == [
        "core (1, 2)",
        "core (1, 2) thread 0",
        "core (1, 2) thread 1",
        "core (1, 2) thread 2",
    ]
    assert reports[0].passed


def test_render_names_the_compensation_ratio(tmp_path):
    sim = load(tmp_path, "sim.csv", counters())
    hw = load(tmp_path, "hw.csv", counters(ref_cnt=650, THREAD_STALLS_1=420))
    text = sa.render(sa.analyse(sim, hw))
    assert "compensation E_int/E_total" in text
    assert "E_int" in text
    assert "E_total" in text


def test_parse_core_map():
    assert sa.parse_core_map(["1,1=1,2"]) == {(1, 1): (1, 2)}
    with pytest.raises(ValueError, match="--map-core expects"):
        sa.parse_core_map(["1-1:1-2"])


# ---------------------------------------------------------------------------
# End to end, over the checked-in artefacts
# ---------------------------------------------------------------------------


def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "tt_sim.perf.stall_attribution", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_cli_decompose_only_on_the_real_simulator_log():
    result = _cli("--decompose-only", "--sim", str(TESTDATA / "sim-elw-blackhole.csv"))
    assert result.returncode == 0, result.stderr
    assert "DECOMPOSITION ONLY" in result.stdout
    assert "srca_valid" in result.stdout


def test_cli_passes_against_agreeing_synthetic_card_data(tmp_path):
    out = tmp_path / "report.json"
    result = _cli(
        "--sim",
        str(TESTDATA / "sim-elw-blackhole.csv"),
        "--card",
        str(TESTDATA / "card-elw-agreeing-SYNTHETIC-NOT-A-MEASUREMENT.csv"),
        "--json",
        str(out),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout
    payload = json.loads(out.read_text())
    core = payload["cores"][0]
    assert core["passed"]
    headline = core["comparisons"][0]
    assert headline["e_total"] <= sa.E_TOTAL_LIMIT
    assert headline["e_int"] <= sa.E_INT_LIMIT
    assert headline["compensation_ratio"] >= 1.0


def test_cli_fails_against_compensating_synthetic_card_data(tmp_path):
    """Same total, wrong interior: the failure an envelope check cannot see."""
    out = tmp_path / "report.json"
    result = _cli(
        "--sim",
        str(TESTDATA / "sim-elw-blackhole.csv"),
        "--card",
        str(TESTDATA / "card-elw-compensating-SYNTHETIC-NOT-A-MEASUREMENT.csv"),
        "--json",
        str(out),
    )
    assert result.returncode == 1
    assert "RESULT: FAIL" in result.stdout
    payload = json.loads(out.read_text())
    headline = payload["cores"][0]["comparisons"][0]
    assert headline["e_total"] <= sa.E_TOTAL_LIMIT, "the envelope must still agree"
    assert headline["e_int"] > sa.E_INT_LIMIT
    assert headline["compensation_ratio"] > 2.5


def test_cli_refuses_without_card_data_unless_asked():
    result = _cli("--sim", str(TESTDATA / "sim-elw-blackhole.csv"))
    assert result.returncode == 2
    assert "--decompose-only" in result.stderr


def test_cli_refuses_to_be_told_both_things():
    """No mode where a caller could believe a criterion was applied and it wasn't."""
    result = _cli(
        "--sim",
        str(TESTDATA / "sim-elw-blackhole.csv"),
        "--card",
        str(TESTDATA / "card-elw-agreeing-SYNTHETIC-NOT-A-MEASUREMENT.csv"),
        "--decompose-only",
    )
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


# ---------------------------------------------------------------------------
# The provenance rule
# ---------------------------------------------------------------------------


def _executable_strings(tree):
    """Every string literal in a module that is not a docstring.

    Prose may name a cost table -- the module's own docstring says at length
    that it must never write to one. Code may not.
    """
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_counter_value_can_become_a_cost():
    """Nothing in this leg may reach a cost table, at any provenance.

    The counter semantics come from a vendor tech report and RTL, not the ISA
    documentation -- ``vendor_source``, which is fine for corroboration and
    disqualifying as provenance for a cycle charge. The cheapest durable defence
    is that the module's *code* cannot name a cost table or the loader that
    ranks one; the prose is free to explain why.
    """
    tree = ast.parse((REPO / "tt_sim" / "perf" / "stall_attribution.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "tt_sim.perf.costs" not in imported
    assert not any(name.endswith(".costs") for name in imported), imported
    for text in _executable_strings(tree):
        for forbidden in ("unit_costs", "tensix_instruction_costs"):
            assert forbidden not in text, (forbidden, text)


def test_the_cost_table_guard_would_notice():
    """...and the guard is not vacuous: it fires on a module that does offend."""
    tree = ast.parse(
        "'''prose may mention unit_costs.yaml'''\n"
        "from tt_sim.perf.costs import load\n"
        "PATH = 'tt_sim/perf/unit_costs.yaml'\n"
    )
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "tt_sim.perf.costs" in imported
    assert any("unit_costs" in text for text in _executable_strings(tree))
