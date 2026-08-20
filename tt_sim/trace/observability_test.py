"""What the writers do with each of the two timing regimes.

The point of these tests is not that a number is emitted — it is that a
number is only emitted when the simulator actually holds it, and that a
reader of the output can tell which regime produced it. ROADMAP section H
listed real Perfetto durations, NoC issue/arrival cycles and stall
attribution as "gated on section I"; section I supplies the state, and the
failure mode this file guards against is the writers filling the gap with
something plausible instead.
"""

import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tt_sim.trace import hotspots as hotspotsmod
from tt_sim.trace import report as reportmod
from tt_sim.trace.bus import EventBus
from tt_sim.trace.counters import CounterAggregator
from tt_sim.trace.events import (
    STALL_REASONS,
    ComputeEvent,
    CounterSnapshot,
    Event,
    EventCategory,
    InstrEvent,
    NoCEvent,
    StallEvent,
)
from tt_sim.trace.writers.noc_parquet import NoCParquetWriter
from tt_sim.trace.writers.perfetto import PerfettoWriter

CORE = (0, 18, 18, "TRISC1")
UNIT = (0, 18, 18, "THCON")
MATRIX = (0, 18, 18, "MATRIX")
NIU = (0, 18, 18, "NOC0")


def _write(tmp_path, events, cost_model, monkeypatch):
    """Run ``events`` through a PerfettoWriter and hand back the parsed JSON."""
    monkeypatch.setenv("TT_SIM_COST_MODEL", "1" if cost_model else "0")
    bus = EventBus()
    bus.enabled = True
    path = tmp_path / "trace.json"
    writer = PerfettoWriter(path, bus=bus)
    for event in events:
        bus.publish(event)
    writer.close()
    return json.loads(path.read_text())


def _slices(trace, phase="X"):
    return [e for e in trace["traceEvents"] if e["ph"] == phase]


# ---------------------------------------------------------------------------
# Perfetto durations
# ---------------------------------------------------------------------------


def test_a_modelled_occupancy_becomes_the_slice_width(tmp_path, monkeypatch):
    """``ComputeEvent.duration`` is the cycles the cost table charged, so it is
    the width of the slice — the whole point of the section H item."""
    trace = _write(
        tmp_path,
        [
            ComputeEvent(
                cycle=100, unit_id=UNIT, op="ATCAS", target_unit="THCON", duration=15
            )
        ],
        cost_model=True,
        monkeypatch=monkeypatch,
    )
    (compute,) = [s for s in _slices(trace) if s["name"] == "ATCAS"]
    assert compute["ts"] == 100
    assert compute["dur"] == 15
    assert compute["args"]["occupancy_cycles"] == 15
    # A modelled slice does not carry the disclaimer; its absence is what
    # distinguishes it from the case below.
    assert "timing_model" not in compute["args"]


def test_an_unmodelled_op_gets_one_cycle_and_says_so(tmp_path, monkeypatch):
    """With the cost model off nothing occupies anything, so a slice one cycle
    wide is the truth. What must not happen is a fabricated width, and what
    must not happen either is a width of 1 that is indistinguishable from a
    modelled 1."""
    trace = _write(
        tmp_path,
        [ComputeEvent(cycle=100, unit_id=UNIT, op="ATCAS", target_unit="THCON")],
        cost_model=False,
        monkeypatch=monkeypatch,
    )
    (compute,) = [s for s in _slices(trace) if s["name"] == "ATCAS"]
    assert compute["dur"] == 1
    assert "occupancy_cycles" not in compute["args"]
    assert "TT_SIM_COST_MODEL off" in compute["args"]["timing_model"]
    assert trace["otherData"]["cost_model"] is False
    assert "not a modelled figure" in trace["otherData"]["timing_note"]


def test_the_regime_is_declared_three_ways(tmp_path, monkeypatch):
    """A trace copied away from the run that produced it must still say which
    regime it came from: the file trailer, a per-tile process label, and the
    per-slice argument above."""
    trace = _write(
        tmp_path,
        [
            ComputeEvent(
                cycle=1, unit_id=UNIT, op="ATCAS", target_unit="THCON", duration=15
            )
        ],
        cost_model=True,
        monkeypatch=monkeypatch,
    )
    assert trace["otherData"]["cost_model"] is True
    labels = [e for e in trace["traceEvents"] if e.get("name") == "process_labels"]
    assert len(labels) == 1
    assert labels[0]["args"]["labels"] == "TT_SIM_COST_MODEL on"


def test_a_stall_is_its_own_slice_abutting_the_instruction(tmp_path, monkeypatch):
    """The RV load-use interlock's cycles are attributed to the instruction
    they held, and rendered as a slice ending exactly where that instruction
    begins — so a Perfetto track reads as stall-then-issue rather than as a
    gap."""
    trace = _write(
        tmp_path,
        [
            InstrEvent(
                cycle=50,
                unit_id=CORE,
                pc=0x1000,
                instruction=0x13,
                stall_cycles=4,
                stall_reason="load_use",
            )
        ],
        cost_model=True,
        monkeypatch=monkeypatch,
    )
    stall, instr = _slices(trace)
    assert stall["name"] == "stall:load_use"
    assert (stall["ts"], stall["dur"]) == (46, 4)
    assert stall["ts"] + stall["dur"] == instr["ts"]
    assert instr["dur"] == 1
    assert instr["args"]["stall_cycles"] == 4


def test_no_stall_slice_and_no_stall_arg_when_nothing_stalled(tmp_path, monkeypatch):
    """The `instr` slice is the highest-volume event in the trace, so a
    constant ``"stall_cycles":0`` on every one of them is ~16 bytes of nothing
    — measured at +9 MB on a 66 MB `four` trace with the model off."""
    trace = _write(
        tmp_path,
        [InstrEvent(cycle=50, unit_id=CORE, pc=0x1000, instruction=0x13)],
        cost_model=False,
        monkeypatch=monkeypatch,
    )
    (instr,) = _slices(trace)
    assert instr["name"] == "pc=0x1000"
    assert "stall_cycles" not in instr["args"]


def test_a_noc_flight_is_an_async_slice_spanning_issue_to_arrival(
    tmp_path, monkeypatch
):
    """Async (``b``/``e``) rather than ``X``: two packets can be in flight on
    one NIU at once, and partially overlapping ``X`` slices are not legally
    nestable — Perfetto drops them and reports an error."""
    trace = _write(
        tmp_path,
        [
            NoCEvent(
                cycle=281,
                unit_id=NIU,
                phase="request",
                txn_type="read",
                src=(1, 1),
                dst=(18, 18),
                size_bytes=2048,
                txn_id=3,
                issue_cycle=46,
            )
        ],
        cost_model=True,
        monkeypatch=monkeypatch,
    )
    assert not _slices(trace)  # nothing on a thread track to overlap
    (begin,) = _slices(trace, "b")
    (end,) = _slices(trace, "e")
    assert begin["ts"] == 46
    assert end["ts"] == 281
    assert begin["args"]["flight_cycles"] == 235
    assert begin["args"]["arrival_cycle"] == 281


def test_an_untimed_noc_flight_is_labelled_not_guessed(tmp_path, monkeypatch):
    """``issue_cycle == -1`` is a NIU with no owning tile clock — the unit
    tests and ``driver/simple``. There is no flight time to report, so the
    slice is one cycle and says why."""
    trace = _write(
        tmp_path,
        [
            NoCEvent(
                cycle=281,
                unit_id=NIU,
                phase="request",
                txn_type="read",
                src=(1, 1),
                dst=(18, 18),
            )
        ],
        cost_model=True,
        monkeypatch=monkeypatch,
    )
    (begin,) = _slices(trace, "b")
    (end,) = _slices(trace, "e")
    assert (begin["ts"], end["ts"]) == (281, 282)
    assert "untimed" in begin["args"]["timing_model"]


def test_request_and_response_are_still_linked(tmp_path, monkeypatch):
    """The arrows survived the move to async slices, as ``bind_id`` +
    ``flow_out``/``flow_in`` on the slices themselves. A standalone ``s``/``f``
    flow event binds to the enclosing slice of a *thread* track, which an
    async slice is not, so Perfetto would have drawn no arrow at all."""
    common = dict(unit_id=NIU, txn_type="read", size_bytes=32, txn_id=7)
    trace = _write(
        tmp_path,
        [
            NoCEvent(
                cycle=10,
                phase="request",
                src=(1, 1),
                dst=(18, 18),
                issue_cycle=5,
                **common,
            ),
            NoCEvent(
                cycle=20,
                phase="response",
                src=(18, 18),
                dst=(1, 1),
                issue_cycle=15,
                **common,
            ),
        ],
        cost_model=True,
        monkeypatch=monkeypatch,
    )
    out, incoming = _slices(trace, "b")
    assert out["flow_out"] is True
    assert incoming["flow_in"] is True
    assert out["bind_id"] == incoming["bind_id"]


# ---------------------------------------------------------------------------
# NoC Parquet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cost_model", [True, False])
def test_noc_parquet_carries_the_flight_columns(tmp_path, monkeypatch, cost_model):
    """The three columns section H asked for, in a dataset a query engine can
    read. ``cost_model`` rides along because a ``flight_cycles`` of 1 means
    something different in each regime."""
    monkeypatch.setenv("TT_SIM_COST_MODEL", "1" if cost_model else "0")
    bus = EventBus()
    bus.enabled = True
    writer = NoCParquetWriter(tmp_path / "noc", bus=bus)
    bus.publish(
        NoCEvent(
            cycle=300,
            unit_id=NIU,
            phase="response",
            txn_type="read",
            src=(1, 1),
            dst=(18, 18),
            size_bytes=64,
            txn_id=1,
            issue_cycle=65,
        )
    )
    writer.close()
    rows = pq.read_table(tmp_path / "noc").to_pylist()
    assert len(rows) == 1
    (row,) = rows
    assert row["issue_cycle"] == 65
    assert row["arrival_cycle"] == 300
    assert row["flight_cycles"] == 235
    assert row["cost_model"] is cost_model


def test_an_untimed_flight_is_zero_not_one(tmp_path, monkeypatch):
    """``issue_cycle == -1`` must not be arithmetic'd into a plausible flight.
    Zero here means "no measurement", and the ``issue_cycle`` column keeps the
    sentinel so a query can exclude those rows."""
    bus = EventBus()
    bus.enabled = True
    writer = NoCParquetWriter(tmp_path / "noc", bus=bus)
    bus.publish(
        NoCEvent(
            cycle=300,
            unit_id=NIU,
            phase="request",
            txn_type="read",
            src=(1, 1),
            dst=(2, 2),
        )
    )
    writer.close()
    (row,) = pq.read_table(tmp_path / "noc").to_pylist()
    assert row["issue_cycle"] == -1
    assert row["flight_cycles"] == 0


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------


def _counters(events):
    """Totals the aggregator would flush for ``events``."""
    from tt_sim.trace import bus as bus_module

    previous = bus_module._BUS
    bus_module._BUS = EventBus()
    bus_module._BUS.enabled = True
    collected: dict[tuple, int] = {}
    bus_module._BUS.subscribe(
        EventCategory.COUNTER,
        lambda e: collected.__setitem__((e.unit_id, e.counter_name), e.value),
    )
    try:
        aggregator = CounterAggregator()
        for event in events:
            bus_module._BUS.publish(event)
        aggregator.flush()
    finally:
        bus_module._BUS = previous
    return collected


def test_rv_stalls_are_counted_by_reason():
    counters = _counters(
        [
            InstrEvent(
                cycle=10,
                unit_id=CORE,
                pc=0,
                instruction=0x13,
                stall_cycles=4,
                stall_reason="load_use",
            ),
            InstrEvent(
                cycle=20,
                unit_id=CORE,
                pc=4,
                instruction=0x13,
                stall_cycles=2,
                stall_reason="store_rate",
            ),
        ]
    )
    assert counters[(CORE, "stall_cycles")] == 6
    assert counters[(CORE, "stall_load_use")] == 4
    assert counters[(CORE, "stall_store_rate")] == 2


def test_an_unmodelled_run_has_no_stall_rows_rather_than_zeroed_ones():
    """Absence is the honest encoding: a dataset from a run with the cost model
    off should not contain a row asserting the machine never stalled, because
    the machine was never able to."""
    counters = _counters([InstrEvent(cycle=10, unit_id=CORE, pc=0, instruction=0x13)])
    assert (CORE, "stall_cycles") not in counters
    assert counters[(CORE, "instr_retired")] == 1


def test_backend_unit_busy_cycles_come_only_from_modelled_ops():
    counters = _counters(
        [
            ComputeEvent(
                cycle=1, unit_id=UNIT, op="ATCAS", target_unit="THCON", duration=15
            ),
            ComputeEvent(cycle=2, unit_id=UNIT, op="NOP", target_unit="THCON"),
        ]
    )
    assert counters[(UNIT, "compute_ops")] == 2
    assert counters[(UNIT, "busy_cycles")] == 15


# ---------------------------------------------------------------------------
# Matrix Unit occupancy, cut by whether any operand data moved
# ---------------------------------------------------------------------------


def test_matrix_bookkeeping_cycles_are_a_subset_of_busy_cycles():
    """``INCRWC`` occupies the Matrix Unit and moves no operand data; ``MVMUL``
    does both. The bookkeeping counter re-cuts the same cycles rather than
    adding any, which is why ``tt_sim.trace.report`` ranks it as redundant."""
    counters = _counters(
        [
            ComputeEvent(
                cycle=1, unit_id=MATRIX, op="MVMUL", target_unit="Matrix", duration=1
            ),
            ComputeEvent(
                cycle=2, unit_id=MATRIX, op="INCRWC", target_unit="Matrix", duration=1
            ),
            ComputeEvent(
                cycle=3, unit_id=MATRIX, op="SETRWC", target_unit="Matrix", duration=1
            ),
        ]
    )
    assert counters[(MATRIX, "busy_cycles")] == 3
    assert counters[(MATRIX, "bookkeeping_cycles")] == 2


def test_a_matrix_run_with_no_bookkeeping_emits_no_bookkeeping_row():
    """Absent, not zero — the same rule every other modelled counter follows."""
    counters = _counters(
        [
            ComputeEvent(
                cycle=1, unit_id=MATRIX, op="MVMUL", target_unit="Matrix", duration=1
            )
        ]
    )
    assert (MATRIX, "busy_cycles") in counters
    assert (MATRIX, "bookkeeping_cycles") not in counters


def test_bookkeeping_cycles_are_absent_without_the_cost_model():
    """``duration=0`` is "no claim", so neither counter appears — a bookkeeping
    row of 0 alongside no ``busy_cycles`` row would read as "no bookkeeping"."""
    counters = _counters(
        [ComputeEvent(cycle=1, unit_id=MATRIX, op="INCRWC", target_unit="Matrix")]
    )
    assert (MATRIX, "busy_cycles") not in counters
    assert (MATRIX, "bookkeeping_cycles") not in counters


def test_bookkeeping_cycles_are_redundant_against_busy_cycles():
    """The attribution report partitions cycles. Ranking a subset beside the set
    it came from double-counts every cycle in it."""
    from tt_sim.trace import report

    assert report.is_redundant("bookkeeping_cycles")
    assert not report.is_redundant("busy_cycles")


def test_every_matrix_unit_opcode_is_classified_as_bookkeeping_or_datapath():
    """The completeness guard. ``TensixBackendUnit`` raises
    ``NotImplementedError`` for a Matrix opcode it has no handler for, so
    ``MatrixUnit.OPCODE_TO_HANDLER`` is exactly the set that can reach a
    ``ComputeEvent`` — and every member of it has to have been called either
    work or bookkeeping by somebody, rather than defaulting to work because
    nobody looked."""
    from tt_sim.pe.tensix.backends.matrix import MatrixUnit
    from tt_sim.trace.events import MATRIX_BOOKKEEPING_OPS, MATRIX_DATAPATH_OPS

    handled = set(MatrixUnit.OPCODE_TO_HANDLER)
    assert not (MATRIX_BOOKKEEPING_OPS & MATRIX_DATAPATH_OPS)
    assert MATRIX_BOOKKEEPING_OPS | MATRIX_DATAPATH_OPS == handled, (
        "a Matrix Unit opcode is unclassified (or classified but unhandled). "
        "Say whether it moves operand data, in tt_sim/trace/events.py."
    )
    # The two that must never swap sides, named so the guard reads as a claim
    # about the machine rather than about set arithmetic.
    assert "MVMUL" in MATRIX_DATAPATH_OPS
    assert "INCRWC" in MATRIX_BOOKKEEPING_OPS


def test_noc_flight_cycles_are_counted_when_timed():
    counters = _counters(
        [
            NoCEvent(
                cycle=300,
                unit_id=NIU,
                phase="request",
                txn_type="read",
                src=(1, 1),
                dst=(2, 2),
                issue_cycle=65,
            ),
            NoCEvent(
                cycle=400,
                unit_id=NIU,
                phase="request",
                txn_type="read",
                src=(1, 1),
                dst=(2, 2),
            ),
        ]
    )
    assert counters[(NIU, "noc_flight_cycles")] == 235
    assert counters[(NIU, "noc_txns_timed")] == 1


def test_counter_snapshots_are_still_the_documented_shape():
    """Guard on the long-format contract the Parquet writer and the canned
    DuckDB queries both depend on."""
    counters = _counters([InstrEvent(cycle=1, unit_id=CORE, pc=0, instruction=0x13)])
    assert all(isinstance(value, int) for value in counters.values())
    assert CounterSnapshot.CATEGORY is EventCategory.COUNTER


# ---------------------------------------------------------------------------
# Tensix stall reasons
# ---------------------------------------------------------------------------


def test_the_schema_version_covers_the_stall_event():
    """A new event kind is a schema change. ``StallEvent`` was added at 4, so a
    consumer that pinned 3 and sees 4 knows to re-read the taxonomy — even
    though the change is additive and its own categories are untouched."""
    assert Event.SCHEMA_VERSION == 4
    assert StallEvent.CATEGORY is EventCategory.STALL
    assert EventCategory.STALL.value == "stall"


def test_every_stall_reason_is_in_the_frozen_vocabulary():
    """The reasons the simulator can actually emit, pinned against the exported
    set. A reason that is not in ``STALL_REASONS`` is a typo, and a code
    generator switching on the set would silently drop it."""
    import ast
    import inspect

    from tt_sim.pe.tensix import frontend as frontend_mod
    from tt_sim.pe.tensix.backends import backend_base, config, misc, sync, thcon
    from tt_sim.pe.tensix.backends import unpacker as unpacker_mod

    emitted = set()
    for module in (
        backend_base,
        config,
        misc,
        sync,
        thcon,
        unpacker_mod,
        frontend_mod,
    ):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None)
            if name not in ("_refuse", "_note_stall"):
                continue
            # ``_refuse(reason, ...)`` / ``_note_stall(cycle, reason, ...)``.
            idx = 0 if name == "_refuse" else 1
            if len(node.args) > idx and isinstance(node.args[idx], ast.Constant):
                emitted.add(node.args[idx].value)
    assert emitted, "found no stall emission sites — did the helpers get renamed?"
    assert emitted <= STALL_REASONS, sorted(emitted - STALL_REASONS)


def test_a_stall_episode_is_one_wide_slice_on_its_own_track(tmp_path, monkeypatch):
    """A stall spans many cycles while the instruction slices under it span one.
    They must not share a track: two partially overlapping ``X`` slices are not
    legally nestable and Perfetto drops them."""
    trace = _write(
        tmp_path,
        [
            StallEvent(
                cycle=100,
                unit_id=CORE,
                reason="src_reserved_by_unpacker",
                blocked_on="UNPACK",
                cycles=37,
                opcode="MVMUL",
                thread_id=1,
            ),
            InstrEvent(cycle=100, unit_id=CORE, pc=0x100, instruction=0x13),
        ],
        cost_model=True,
        monkeypatch=monkeypatch,
    )
    stall = [s for s in _slices(trace) if s["name"].startswith("stall:")]
    assert len(stall) == 1
    assert stall[0]["dur"] == 37
    assert stall[0]["ts"] == 100
    # Names the cause in the slice title, so the UI is readable without
    # opening the args.
    assert stall[0]["name"] == "stall:src_reserved_by_unpacker->UNPACK"
    assert stall[0]["args"]["blocked_on"] == "UNPACK"
    assert stall[0]["args"]["blocked_opcode"] == "MVMUL"
    instr = [s for s in _slices(trace) if s["name"] == "pc=0x100"]
    assert len(instr) == 1
    assert instr[0]["tid"] != stall[0]["tid"], "stall shares the instruction track"


def test_a_stall_slice_never_reads_as_a_calibrated_absolute(tmp_path, monkeypatch):
    """Modelled cycles are floors built from published bounds. The slice a
    reader is most tempted to quote out of context has to say so on itself."""
    for cost_model in (True, False):
        trace = _write(
            tmp_path,
            [
                StallEvent(
                    cycle=5,
                    unit_id=CORE,
                    reason="semaphore_empty",
                    cycles=9,
                    semaphore=3,
                )
            ],
            cost_model=cost_model,
            monkeypatch=monkeypatch,
        )
        (stall,) = [s for s in _slices(trace) if s["name"].startswith("stall:")]
        assert "not calibrated to silicon" in stall["args"]["timing_model"]
        # No unit is named for a semaphore wait: it is about a sync object.
        assert stall["name"] == "stall:semaphore_empty"
        assert "blocked_on" not in stall["args"]
        assert stall["args"]["semaphore"] == 3


def test_stall_cycles_are_counted_by_reason_and_by_blamed_unit():
    counters = _counters(
        [
            StallEvent(
                cycle=10,
                unit_id=CORE,
                reason="src_reserved_by_unpacker",
                blocked_on="UNPACK",
                cycles=700,
            ),
            StallEvent(
                cycle=20,
                unit_id=CORE,
                reason="unit_busy",
                blocked_on="UNPACK",
                cycles=74,
            ),
        ]
    )
    assert counters[(CORE, "tensix_stall_cycles")] == 774
    assert counters[(CORE, "tensix_stall_episodes")] == 2
    assert counters[(CORE, "tensix_stall_src_reserved_by_unpacker")] == 700
    assert counters[(CORE, "tensix_stall_unit_busy")] == 74
    # Both reasons blame the same unit, so the per-unit roll-up is their sum —
    # which is the number "how much is the unpacker costing me?" wants.
    assert counters[(CORE, "tensix_stall_on_UNPACK")] == 774


def test_tensix_stall_counters_do_not_collide_with_the_rv_ones():
    """A Tensix thread publishes StallEvents under the *same* unit_id its baby
    RISC-V core publishes InstrEvents under. An unprefixed ``stall_cycles``
    would sum two unrelated mechanisms into one unreadable number."""
    counters = _counters(
        [
            InstrEvent(
                cycle=10,
                unit_id=CORE,
                pc=0,
                instruction=0x13,
                stall_cycles=5,
                stall_reason="load_use",
            ),
            StallEvent(cycle=11, unit_id=CORE, reason="mutex_wait", cycles=40),
        ]
    )
    assert counters[(CORE, "stall_cycles")] == 5
    assert counters[(CORE, "stall_load_use")] == 5
    assert counters[(CORE, "tensix_stall_cycles")] == 40
    assert counters[(CORE, "tensix_stall_mutex_wait")] == 40


def test_a_run_with_no_stalls_has_no_stall_rows_rather_than_zeroed_ones():
    """Same contract as the RV stall counters: absent, not zero. A dataset must
    not assert a stall-free machine just because nothing was measured."""
    counters = _counters([InstrEvent(cycle=1, unit_id=CORE, pc=0, instruction=0x13)])
    assert not [name for (_, name) in counters if name.startswith("tensix_stall")]


def test_the_bus_stays_free_when_the_stall_category_is_off():
    """The emission guard is ``is_enabled``, so a disabled category must cost a
    subscriber nothing — the untraced path is the one that has to stay free."""
    bus = EventBus()
    bus.enabled = True
    bus.set_category_enabled(EventCategory.STALL, False)
    seen = []
    bus.subscribe(EventCategory.STALL, seen.append)
    bus.publish(StallEvent(cycle=1, unit_id=CORE, reason="mutex_wait"))
    assert seen == []
    assert not bus.is_enabled(EventCategory.STALL)


# ---------------------------------------------------------------------------
# The published schema contract (docs/trace-schema.md)
# ---------------------------------------------------------------------------
#
# These pin what ``docs/trace-schema.md`` promises external consumers. The
# compiler team builds tooling against those names, so from v4 onwards a
# rename or a removal is a breaking change and must come with a
# ``SCHEMA_VERSION`` bump. A test that fails when a documented field is
# renamed is worth more than a paragraph asking people to be careful.

#: Field names per event class, exactly as documented. Adding a field with a
#: default is additive and safe -- list it here in the same commit. Renaming or
#: removing one is breaking: bump ``Event.SCHEMA_VERSION`` and say so in
#: ``docs/trace-schema.md``.
DOCUMENTED_FIELDS = {
    "InstrEvent": {
        "cycle",
        "unit_id",
        "pc",
        "instruction",
        "stalled",
        "reg_write_idx",
        "reg_write_value",
        "stall_cycles",
        "stall_reason",
    },
    "DispatchEvent": {"cycle", "unit_id", "opcode", "target_unit", "thread_id"},
    "NoCEvent": {
        "cycle",
        "unit_id",
        "phase",
        "txn_type",
        "src",
        "dst",
        "size_bytes",
        "txn_id",
        "issue_cycle",
    },
    "LifecycleEvent": {"cycle", "unit_id", "kind", "detail"},
    "MemEvent": {"cycle", "unit_id", "op", "address", "size", "region", "pc"},
    "ComputeEvent": {
        "cycle",
        "unit_id",
        "op",
        "target_unit",
        "thread_id",
        "detail",
        "duration",
    },
    "SyncEvent": {"cycle", "unit_id", "kind", "detail"},
    "CounterSnapshot": {"cycle", "unit_id", "counter_name", "value", "kernel_id"},
    "StallEvent": {
        "cycle",
        "unit_id",
        "reason",
        "blocked_on",
        "cycles",
        "opcode",
        "thread_id",
        "semaphore",
    },
}


def test_every_documented_event_field_is_still_there():
    """Field *names* are the frozen half of the contract -- cycle *values* are
    not, and move with each cost-model instalment by design. A rename here
    breaks every consumer silently, because a missing key reads as a null."""
    import dataclasses

    from tt_sim.trace import events as events_mod

    for name, expected in DOCUMENTED_FIELDS.items():
        cls = getattr(events_mod, name)
        actual = {f.name for f in dataclasses.fields(cls)}
        assert actual == expected, (
            f"{name}: documented {sorted(expected)}, found {sorted(actual)}. "
            "Adding a field is additive — list it above. Renaming or removing "
            "one is breaking: bump SCHEMA_VERSION and update "
            "docs/trace-schema.md."
        )


def test_every_event_class_is_documented():
    """A new event kind that nobody wrote down is the failure this catches."""
    from tt_sim.trace import events as events_mod

    published = {
        cls.__name__
        for cls in vars(events_mod).values()
        if isinstance(cls, type) and issubclass(cls, Event) and cls is not Event
    }
    assert published == set(DOCUMENTED_FIELDS)


def test_the_parquet_counter_columns_are_frozen(tmp_path):
    """The counter dataset's column names are what every canned query and every
    downstream notebook is written against."""
    import glob

    from tt_sim.trace.writers.parquet import ParquetCounterWriter

    bus = EventBus()
    bus.enabled = True
    writer = ParquetCounterWriter(tmp_path, bus=bus)
    bus.publish(
        CounterSnapshot(cycle=100, unit_id=CORE, counter_name="instr_retired", value=7)
    )
    writer.close()
    files = glob.glob(str(tmp_path / "**" / "*.parquet"), recursive=True)
    row = pq.ParquetDataset(files).read().to_pylist()[0]
    assert set(row) == {
        "cycle",
        "chip",
        "kernel_id",
        "core_y",
        "core_x",
        "unit",
        "counter_name",
        "value",
    }


def test_the_three_unit_vocabularies_stay_in_sync():
    """``unit_id[3]``, ``ComputeEvent.target_unit`` and
    ``DispatchEvent.target_unit`` name the same backend three different ways,
    and a consumer joining on the wrong one gets an empty result rather than an
    error. The alias table is the documented translation, so it has to track
    the code that produces the other two spellings."""
    import ast
    import inspect
    from pathlib import Path

    import yaml

    from tt_sim.pe.tensix.backends import (
        config,
        matrix,
        misc,
        mover,
        packer,
        sync,
        thcon,
        unpacker,
        vector,
    )
    from tt_sim.trace.events import BACKEND_UNIT_ALIASES, Unit

    # Left column: every key is a real ``Unit`` enum value.
    for unit in BACKEND_UNIT_ALIASES:
        assert unit in {u.value for u in Unit}, unit

    # Middle column: the backend classes' own ``unit_name`` strings, read out
    # of their ``super().__init__(..., "<name>")`` calls.
    names = set()
    for module in (config, matrix, misc, mover, packer, sync, thcon, unpacker, vector):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "__init__"
                and node.args
                and isinstance(node.args[-1], ast.Constant)
                and isinstance(node.args[-1].value, str)
            ):
                names.add(node.args[-1].value)
    documented = {alias for alias, _ in BACKEND_UNIT_ALIASES.values()}
    assert names == documented, (
        f"backend unit_name strings {sorted(names)} != documented "
        f"{sorted(documented)} — update BACKEND_UNIT_ALIASES and "
        "docs/trace-schema.md"
    )

    # Right column: every alias is a real ``ex_resource`` in the ISA tables.
    spec = yaml.safe_load(
        (
            Path(inspect.getfile(matrix)).parent.parent / "tensix_instructions.yaml"
        ).read_text()
    )
    entries = spec.values() if isinstance(spec, dict) else spec
    resources = {
        entry["ex_resource"]
        for entry in entries
        if isinstance(entry, dict) and "ex_resource" in entry
    }
    for unit, (_, ex_resource) in BACKEND_UNIT_ALIASES.items():
        assert ex_resource in resources, (
            f"{unit} -> ex_resource {ex_resource!r} is not in tensix_instructions.yaml"
        )


# ---------------------------------------------------------------------------
# The profile artefacts' schema contract (docs/trace-schema.md §9)
# ---------------------------------------------------------------------------
#
# ``report.json``, ``hotspots.json`` and ``profile.json`` are documented field
# by field in §9 and carry their own version, ``tt_sim.trace.report.
# SCHEMA_VERSION`` — deliberately not the event ``SCHEMA_VERSION`` above,
# which is scoped to event shape and would move for reasons a report consumer
# cannot act on.
#
# The field sets are **parsed out of the document** rather than copied here.
# The failure this contract exists to prevent is the document and the artefact
# drifting apart, and a list duplicated into this file drifts with them: it
# would still pass while §9 described a field nothing writes.

_SCHEMA_DOC = Path(__file__).resolve().parents[2] / "docs" / "trace-schema.md"


def _documented_fields(heading: str) -> set[str]:
    """Field names in the first Markdown table under ``heading`` in §9.

    Reads the left-hand column, takes every backticked token in it (one row
    documents two fields), and strips the ``[]`` / ``{}`` shape suffix the
    document uses to hint at a list or a map.
    """
    lines = _SCHEMA_DOC.read_text().splitlines()
    assert heading in lines, f"{heading} is gone from {_SCHEMA_DOC.name}"
    fields: set[str] = set()
    in_table = False
    for line in lines[lines.index(heading) + 1 :]:
        if not line.startswith("|"):
            if in_table:
                break  # first blank line after the table ends it
            if line.startswith("#"):
                break  # next section, and no table found
            continue
        in_table = True
        for name in re.findall(r"`([^`]+)`", line.split("|")[1]):
            fields.add(name.rstrip("[]{}"))
    # A silently-empty parse would make every assertion below vacuous, so the
    # table's shape is itself asserted.
    assert len(fields) >= 5, f"parsed only {sorted(fields)} from {heading}"
    return fields


def _report_with_everything() -> "reportmod.Report":
    table = hotspotsmod.HotspotTable(
        rows=[
            hotspotsmod.Hotspot(
                unit="TRISC0",
                pc=0x100,
                retired=3,
                stall_cycles=2,
                by_reason={"load_use": 2},
                function="mm",
            )
        ],
        unattributed_units=["NCRISC"],
    )
    return reportmod.build(
        None,
        hotspots=reportmod.hotspots_to_dict(table),
        elfs=[{"unit": "TRISC0", "role": "kernel", "path": "/k.elf", "how": "recent"}],
        cost_model=True,
        label="doc-contract",
        notes=["a note"],
    )


def test_the_report_carries_its_own_version_not_the_event_one(tmp_path):
    """The version a report consumer reads has to move when a *report* field
    moves, and stay put when an event field does. Sharing the event counter
    gets both wrong, which is why this is a constant of its own."""
    reportmod.write(_report_with_everything(), tmp_path)
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["schema_version"] == reportmod.SCHEMA_VERSION
    assert isinstance(reportmod.SCHEMA_VERSION, int)
    # The embedded hotspot block is governed by the same number and says so,
    # so a consumer never has to ask which file it came from.
    assert payload["hotspots"]["schema_version"] == reportmod.SCHEMA_VERSION


def test_report_json_is_exactly_the_field_set_section_9_documents(tmp_path):
    """Documentation and artefact, compared directly. A field written but not
    documented is an undocumented promise; a field documented but not written
    reads as a null to every consumer that trusts the table."""
    reportmod.write(_report_with_everything(), tmp_path)
    payload = json.loads((tmp_path / "report.json").read_text())
    documented = _documented_fields("### `report.json`")
    assert set(payload) == documented, (
        f"report.json has {sorted(set(payload) - documented)} undocumented and "
        f"is missing {sorted(documented - set(payload))}. Adding a field is "
        "additive, renaming or removing one is breaking: either way bump "
        "tt_sim.trace.report.SCHEMA_VERSION and update docs/trace-schema.md §9."
    )


def test_each_contribution_row_is_the_documented_shape():
    """``contributions[]`` is the list every consumer iterates, and §9 spells
    its keys inline. Parsed from that same cell."""
    import dataclasses

    line = next(
        line
        for line in _SCHEMA_DOC.read_text().splitlines()
        if line.startswith("| `contributions[]`")
    )
    documented = {
        name.strip() for name in re.search(r"\{([^}]*)\}", line)[1].split(",")
    }
    actual = {f.name for f in dataclasses.fields(reportmod.Contribution)}
    assert actual == documented, (
        f"contributions[] rows are {sorted(actual)}, §9 documents {sorted(documented)}"
    )


def test_hotspots_json_is_exactly_the_field_set_section_9_documents():
    """The same check for the sibling artefact, which is also the block
    embedded in ``report.json`` — one structure, one contract."""
    table = hotspotsmod.HotspotTable(rows=[], unattributed_units=[])
    produced = set(reportmod.hotspots_to_dict(table))
    # ``elfs`` and ``cost_model`` are attached by the profile writer rather
    # than the serialiser (auto.py), because provenance is a property of the
    # run, not of the table.
    produced |= {"elfs", "cost_model"}
    documented = _documented_fields("### `hotspots.json`")
    assert produced == documented, (
        f"hotspots.json has {sorted(produced - documented)} undocumented and "
        f"is missing {sorted(documented - produced)}."
    )


def test_profile_json_carries_the_same_version(tmp_path, monkeypatch):
    """One number across the three artefacts of §9: they are written by one
    run and read together."""
    from tt_sim.trace import auto, elfdisc

    # ELF discovery walks the real tt-metal cache; the version is what is
    # under test, not what is on this machine's disk.
    monkeypatch.setattr(auto, "discover", lambda **kwargs: elfdisc.Discovery())
    auto.write_profile_report({"dir": str(tmp_path)}, None)
    meta = json.loads((tmp_path / "profile.json").read_text())
    assert meta["schema_version"] == reportmod.SCHEMA_VERSION
