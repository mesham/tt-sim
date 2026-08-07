"""Source-level attribution and the ranked report.

Three things are being defended here, and they are the three that would
make the feature quietly useless rather than loudly broken:

1. **A real ELF parses.** tt-metal's kernel ELFs keep ``.rela.debug_*``
   sections that pyelftools tries, and fails, to apply on RISC-V. The
   index must read DWARF without relocating it, or every real kernel
   raises. There is no RISC-V toolchain to rely on in CI, but DWARF is
   machine-independent for the parts used here, so the fixture is a
   host ``gcc -g`` object — it exercises the same line program and the
   same DIE walk.
2. **Attribution uses a floor lookup.** A line program emits a row only
   where the source position changes, so exact-PC lookup silently drops
   most of a run. That failure mode looks like "the tool works, my
   kernel is just cold".
3. **The report generalises over counters it has never heard of.** New
   stall reasons and new cycle-bearing counters are landing continuously;
   a report that ranked a hardcoded list would be stale on merge and,
   worse, would look complete while omitting them.
"""

import json
import shutil
import subprocess
import textwrap

import pytest

from tt_sim.trace import elfdisc, report
from tt_sim.trace.bus import EventBus
from tt_sim.trace.dwarf import DwarfIndex, SourceLoc, _lookup_function
from tt_sim.trace.events import InstrEvent
from tt_sim.trace.hotspots import HotspotAggregator

CORE = (0, 18, 18, "TRISC1")
OTHER = (0, 18, 18, "BRISC")


# ---------------------------------------------------------------------------
# 1. A real, DWARF-carrying ELF
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dwarf_elf(tmp_path_factory):
    """A small ELF with DWARF, inlining and a nested call — the shape the
    kernel ELFs have after LTO folds everything into one subprogram."""
    if shutil.which("gcc") is None:
        pytest.skip("no gcc to build a DWARF fixture")
    tmp = tmp_path_factory.mktemp("dwarf")
    src = tmp / "fixture.c"
    src.write_text(
        textwrap.dedent(
            """
            static inline int inner(int x) { return x * 3 + 1; }
            static int middle(int x) { return inner(x) + inner(x + 1); }
            int outer(int x) { return middle(x) + middle(x + 2); }
            int main(void) { return outer(7) & 1; }
            """
        )
    )
    out = tmp / "fixture.elf"
    proc = subprocess.run(
        ["gcc", "-g3", "-O2", "-o", str(out), str(src)],
        capture_output=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"gcc could not build the fixture: {proc.stderr.decode()[:200]}")
    return out


def test_a_real_elf_loads_and_names_functions(dwarf_elf):
    index = DwarfIndex()
    added = index.load(dwarf_elf, unit="TRISC1")
    assert added > 0, "the fixture carries DWARF, so rows must be added"
    names = {name for _, _, name in index._funcs[None]}
    # ``inner`` is inlined into ``middle``; the whole point of walking
    # DW_TAG_inlined_subroutine is that it is still nameable.
    assert "outer" in names
    assert "inner" in names, "inlined callees must be attributable"


def test_the_index_is_scoped_per_unit(dwarf_elf):
    """Two baby cores run different ELFs at overlapping addresses. A lookup
    for a unit that loaded nothing must fall back rather than lie."""
    index = DwarfIndex()
    index.load(dwarf_elf, unit="TRISC1")
    assert index.units() == ["TRISC1"]
    pc = min(index._by_unit["TRISC1"])
    assert index.lookup(pc, unit="TRISC1") is not None
    # BRISC loaded nothing; the merged index answers rather than nothing at all.
    assert index.lookup(pc, unit="BRISC") is not None


def test_nearest_resolves_a_pc_the_line_program_never_names(dwarf_elf):
    """The failure this guards: exact lookup returns None for most PCs, so a
    profile built on it reports a nearly-cold kernel."""
    index = DwarfIndex()
    index.load(dwarf_elf)
    pcs = sorted(index._by_unit[None])
    # A PC one byte past a recorded row is, on every real ISA, still inside
    # the same source line.
    probe = pcs[len(pcs) // 2] + 1
    assert index.lookup(probe) is None, "fixture assumption: no exact row here"
    loc = index.nearest(probe)
    assert isinstance(loc, SourceLoc)
    assert loc.line > 0


def test_nearest_refuses_to_answer_outside_the_line_programs_coverage(dwarf_elf):
    """The expensive failure mode: an index built from the *wrong* ELF
    answers every query, and the report claims ~100 % attribution to a
    kernel that never ran. A PC outside coverage must be unresolved."""
    index = DwarfIndex()
    index.load(dwarf_elf)
    highest = max(index._by_unit[None])
    assert index.nearest(highest + 0x100000) is None
    assert index.covers(highest + 0x100000) is False
    assert index.nearest(0) is None


def test_a_stripped_elf_is_a_no_op_not_an_error(dwarf_elf, tmp_path):
    if shutil.which("strip") is None:
        pytest.skip("no strip")
    stripped = tmp_path / "stripped.elf"
    shutil.copy(dwarf_elf, stripped)
    subprocess.run(["strip", "-g", str(stripped)], check=True)
    index = DwarfIndex()
    assert index.load(stripped) == 0
    assert index.lookup(0x1000) is None


def test_innermost_range_wins():
    """A PC inside an inlined callee must name the callee, not the caller
    it disappeared into — that distinction is the whole value on LTO'd
    tt-metal kernels, where one subprogram spans the entire kernel."""
    funcs = sorted(
        [(0x1000, 0x2000, "kernel_main"), (0x1400, 0x1410, "cb_wait_front")],
        key=lambda f: (f[0], f[1] - f[0]),
    )
    assert _lookup_function(funcs, 0x1404) == "cb_wait_front"
    assert _lookup_function(funcs, 0x1800) == "kernel_main"
    assert _lookup_function(funcs, 0x9000) == ""


# ---------------------------------------------------------------------------
# 2. Hotspots
# ---------------------------------------------------------------------------


def _instr(unit, pc, cycle=1, stall_cycles=0, reason="", stalled=False):
    return InstrEvent(
        cycle=cycle,
        unit_id=unit,
        pc=pc,
        instruction=0x13,
        stalled=stalled,
        stall_cycles=stall_cycles,
        stall_reason=reason,
    )


def _aggregate(events):
    bus = EventBus()
    bus.enabled = True
    agg = HotspotAggregator(bus=bus)
    for event in events:
        bus.publish(event)
    return agg


def test_a_pcs_cycles_are_issue_plus_everything_it_waited_for():
    agg = _aggregate(
        [
            _instr(CORE, 0x100, stall_cycles=9, reason="load_use"),
            _instr(CORE, 0x100, stall_cycles=1, reason="load_use"),
            _instr(CORE, 0x104),
        ]
    )
    rows = {r.pc: r for r in agg.resolve().rows}
    assert rows[0x100].retired == 2
    assert rows[0x100].stall_cycles == 10
    assert rows[0x100].cycles == 12
    assert rows[0x104].cycles == 1


def test_an_unknown_stall_reason_is_carried_through_verbatim():
    """The concurrency contract: new reasons appear in the output with no
    change here. Nothing enumerates the reason set."""
    agg = _aggregate(
        [_instr(CORE, 0x200, stall_cycles=5, reason="packer_backpressure")]
    )
    row = agg.resolve().rows[0]
    assert row.by_reason == {"packer_backpressure": 5}


def test_the_same_pc_on_two_cores_stays_two_rows():
    """BRISC and TRISC1 kernel text overlaps in L1; folding them would
    attribute one core's cycles to the other's source."""
    agg = _aggregate([_instr(CORE, 0x300), _instr(OTHER, 0x300)])
    assert len({(r.unit, r.pc) for r in agg.resolve().rows}) == 2


def test_units_without_an_elf_are_named_not_silently_dropped(dwarf_elf):
    index = DwarfIndex()
    index.load(dwarf_elf, unit="TRISC1")
    agg = _aggregate([_instr(CORE, 0x400), _instr(OTHER, 0x400)])
    table = agg.resolve(index)
    assert table.unattributed_units == ["BRISC"]


def test_resolution_names_the_function_containing_the_executed_pc(dwarf_elf):
    """Function names are resolved at lookup, against the PC itself. Doing it
    at load, per line-table row, is both slower and less accurate."""
    index = DwarfIndex()
    index.load(dwarf_elf, unit="TRISC1")
    inner = next(f for f in index._funcs["TRISC1"] if f[2] == "inner")
    pc = inner[0]
    agg = _aggregate([_instr(CORE, pc)])
    row = agg.resolve(index).rows[0]
    assert row.function == "inner"
    assert row.file
    assert "inner (" in row.location()


def test_by_function_folds_pcs_and_keeps_the_reason_split():
    agg = _aggregate(
        [
            _instr(CORE, 0x500, stall_cycles=4, reason="load_use"),
            _instr(CORE, 0x504, stall_cycles=6, reason="store_rate"),
        ]
    )
    table = agg.resolve()
    for row in table.rows:
        row.function = "cb_wait_front"
        row.file = "cb.h"
        row.line = 12
    folded = table.by_function()
    assert len(folded) == 1
    assert folded[0].cycles == 12
    assert folded[0].by_reason == {"load_use": 4, "store_rate": 6}


def test_the_lcov_writer_still_emits_records_off_a_real_elf(dwarf_elf, tmp_path):
    """The only pre-existing `DwarfIndex` consumer. It moved from exact to
    floor lookup and from tuples to `SourceLoc`; this is the guard that the
    move did not silently empty its output."""
    from tt_sim.trace.writers.lcov import LCOVWriter

    index = DwarfIndex()
    index.load(dwarf_elf, unit="TRISC1")
    bus = EventBus()
    bus.enabled = True
    out = tmp_path / "run.lcov"
    writer = LCOVWriter(out, index, bus=bus)
    pc = sorted(index._by_unit["TRISC1"])[2]
    for _ in range(5):
        bus.publish(_instr(CORE, pc + 1))
    writer.close()
    text = out.read_text()
    assert "SF:" in text
    assert "DA:" in text
    assert ",5" in text, "five retirements at one line must be five hits"


# ---------------------------------------------------------------------------
# 3. The report generalises
# ---------------------------------------------------------------------------


def test_an_unknown_cycle_counter_is_ranked_and_flagged():
    """A counter this file has never heard of must still be ranked. It is
    marked ``discovered`` so a reader can tell prose is missing, which is a
    very different thing from the counter being missing."""
    contributions, volumes = report.classify(
        {
            ("2,1 TRISC0", "stall_load_use"): 100,
            ("2,1 PACKER", "backpressure_cycles"): 250,
            ("2,1 UNPACKER", "stall_unpacker_idle"): 70,
            ("2,1 NOC0", "noc_bytes_total"): 4096,
        }
    )
    ranked = {c.counter: c for c in contributions}
    assert ranked["backpressure_cycles"].cycles == 250
    assert ranked["backpressure_cycles"].discovered is True
    assert ranked["stall_unpacker_idle"].discovered is True
    assert ranked["stall_load_use"].discovered is False
    assert contributions[0].counter == "backpressure_cycles", "ranked by cycles"
    assert volumes == {"noc_bytes_total": 4096}, "byte counters are not cycles"


def test_stall_cycles_is_not_double_counted():
    """``stall_cycles`` is the sum of the per-reason rows; ranking both
    would double every RV stall in the headline table."""
    contributions, _ = report.classify(
        {
            ("2,1 BRISC", "stall_cycles"): 100,
            ("2,1 BRISC", "stall_load_use"): 100,
        }
    )
    assert [c.counter for c in contributions] == ["stall_load_use"]


def test_tensix_stall_reasons_are_cycles_not_volumes():
    """Measured on the Blackhole ``sfpumath`` guard: ``tensix_stall_*`` rows
    carry cycles. They were being filed under a table headed "Volume counters
    (not cycles)", which is wrong in the most expensive direction — a reader
    ranking bottlenecks would never see the largest one."""
    contributions, volumes = report.classify(
        {
            ("2,1 TRISC2", "tensix_stall_semaphore_empty"): 6072,
            ("2,1 TRISC0", "tensix_stall_src_reserved_by_matrix"): 3913,
        }
    )
    assert volumes == {}
    ranked = {c.counter: c for c in contributions}
    assert ranked["tensix_stall_semaphore_empty"].cycles == 6072
    assert ranked["tensix_stall_semaphore_empty"].kind == "stall"
    # Described structurally, from the shape of the name, so a reason invented
    # tomorrow is explained rather than left blank. The vocabulary is open.
    assert "semaphore_empty" in ranked["tensix_stall_semaphore_empty"].described


def test_the_two_redundant_recuts_of_a_tensix_stall_are_not_ranked():
    """One partition of a thread's lost cycles, three ways of writing it down.
    ``tensix_stall_<reason>`` is the partition; ``tensix_stall_cycles`` is its
    total and ``tensix_stall_on_<unit>`` the same cycles re-cut by blame.
    Ranking all three triples the stall."""
    contributions, volumes = report.classify(
        {
            ("2,1 TRISC0", "tensix_stall_cycles"): 3913,
            ("2,1 TRISC0", "tensix_stall_src_reserved_by_matrix"): 3913,
            ("2,1 TRISC0", "tensix_stall_on_MATH"): 3913,
            ("2,1 TRISC0", "tensix_stall_episodes"): 4,
        }
    )
    assert [c.counter for c in contributions] == ["tensix_stall_src_reserved_by_matrix"]
    assert sum(c.cycles for c in contributions) == 3913
    # An episode count is a count, not a duration, however it is spelled.
    assert volumes == {"tensix_stall_episodes": 4}


def test_the_redundancy_rule_is_exported_for_consumers():
    """The single easiest way to get a wrong total out of the counter dataset
    is to sum a total alongside the parts it restates, so the rule is a
    predicate a consumer can call rather than prose they must re-derive."""
    assert report.is_redundant("stall_cycles")
    assert report.is_redundant("tensix_stall_cycles")
    assert report.is_redundant("tensix_stall_on_MATH")
    assert not report.is_redundant("stall_load_use")
    assert not report.is_redundant("tensix_stall_semaphore_empty")
    assert not report.is_redundant("busy_cycles")

    # Cycle-bearing is matched by pattern, never by an enumeration: a stall
    # reason added anywhere in the tree must rank with no change here.
    assert report.is_cycle_bearing("stall_a_reason_invented_tomorrow")
    assert report.is_cycle_bearing("tensix_stall_a_reason_invented_tomorrow")
    assert report.is_cycle_bearing("some_new_occupancy_cycles")
    assert report.is_cycle_bearing("instr_retired")
    assert not report.is_cycle_bearing("noc_bytes_total")
    assert not report.is_cycle_bearing("tensix_stall_episodes")


def test_the_report_states_its_own_framing_and_shows_the_remainder():
    rendered = report.render(
        report.Report(
            span=1000,
            cost_model=True,
            contributions=[report.Contribution("2,1 BRISC", "stall_load_use", 400)],
            hotspots={"total_cycles": 10, "resolved_cycles": 4, "functions": []},
        )
    )
    assert "floors" in rendered
    assert "not calibrated" in rendered
    assert "relative" in rendered
    # span - named, visible rather than dropped
    assert "600" in rendered
    assert "unattributed" in rendered
    assert "40.0 %" in rendered, "unresolved PC share must be stated"


def test_a_run_without_the_cost_model_says_so_rather_than_reporting_zero():
    rendered = report.render(report.Report(span=100, cost_model=False))
    assert "cost model is off" in rendered
    assert "absent rather than zero" in rendered


def test_the_report_round_trips_through_disk(tmp_path):
    built = report.Report(
        span=10,
        cost_model=True,
        contributions=[report.Contribution("2,1 BRISC", "busy_cycles", 4)],
    )
    report.write(built, tmp_path)
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["attributed_cycles"] == 4
    assert (tmp_path / "report.md").read_text().startswith("# tt-sim bottleneck report")
    assert report.main([str(tmp_path)]) == 0


# ---------------------------------------------------------------------------
# 4. ELF discovery
# ---------------------------------------------------------------------------


def _fake_cache(root, kernel, contents=b"\x00"):
    for risc in elfdisc.RISC_TO_UNIT:
        directory = root / "hash" / "kernels" / kernel / "cfg" / risc
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{risc}.elf").write_bytes(contents)
    return root


def test_discovery_finds_one_elf_per_core(tmp_path):
    found = elfdisc.discover(env={}, roots=[_fake_cache(tmp_path, "matmul")])
    assert {d.unit for d in found.elfs} == set(elfdisc.RISC_TO_UNIT.values())
    assert all(d.how == "recent" for d in found.elfs)


def test_firmware_is_discovered_alongside_the_kernel(tmp_path):
    """A data-movement core spends most of a run inside firmware's launch
    and barrier loops; kernel-only discovery leaves the biggest rows bare."""
    _fake_cache(tmp_path, "matmul")
    for risc in elfdisc.RISC_TO_UNIT:
        directory = tmp_path / "hash" / "firmware" / risc
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{risc}.elf").write_bytes(b"\x00")
    found = elfdisc.discover(env={}, roots=[tmp_path])
    assert {d.role for d in found.elfs} == {"kernel", "firmware"}
    assert len(found.elfs) == 2 * len(elfdisc.RISC_TO_UNIT)


def test_discovery_prefers_the_newest_build(tmp_path):
    import os
    import time

    _fake_cache(tmp_path, "old")
    time.sleep(0.01)
    _fake_cache(tmp_path, "new")
    now = time.time()
    for path in (tmp_path / "hash" / "kernels" / "new").rglob("*.elf"):
        os.utime(path, (now, now))
    found = elfdisc.discover(env={}, roots=[tmp_path])
    assert all("/new/" in str(d.path) for d in found.elfs), found.as_pairs()


def test_explicit_elfs_win_and_are_labelled(tmp_path):
    elf = tmp_path / "brisc.elf"
    elf.write_bytes(b"\x00")
    found = elfdisc.discover(env={"TT_SIM_PROFILE_ELFS": f"BRISC:{elf}"})
    assert found.as_pairs() == [("BRISC", str(elf))]
    assert found.elfs[0].how == "explicit"


def test_a_byte_verified_pick_beats_a_newer_unverified_one(tmp_path):
    """Recency is a guess; bytes are proof. When the device can answer, the
    ELF whose segments are actually resident wins even if it is older."""
    _fake_cache(tmp_path, "wrong")
    right = tmp_path / "hash" / "kernels" / "right" / "cfg" / "brisc"
    right.mkdir(parents=True)
    target = right / "brisc.elf"
    target.write_bytes(b"\x00")

    def verifier(unit, vaddr, size):
        return b"\xaa" * size

    def fake_segments(path):
        return (
            [(0x1000, b"\xaa" * 8)] if "right" in str(path) else [(0x1000, b"\xbb" * 8)]
        )

    original = elfdisc.segments_of
    elfdisc.segments_of = fake_segments
    try:
        found = elfdisc.discover(env={}, roots=[tmp_path], verifier=verifier)
    finally:
        elfdisc.segments_of = original
    brisc = next(d for d in found.elfs if d.unit == "BRISC")
    assert brisc.how == "verified"
    assert "right" in str(brisc.path)


def test_a_relocated_kernel_is_found_by_searching_memory(tmp_path):
    """tt-metal places a kernel at a runtime L1 base, not its link address.
    Without recovering that bias the kernel half of every run is
    unattributable — the segment check at ``p_vaddr`` always misses."""
    _fake_cache(tmp_path, "only")
    text = bytes(range(256)) * 2
    original = elfdisc.segments_of
    elfdisc.segments_of = lambda path: [(0x49A0, text)]
    try:
        found = elfdisc.discover(
            env={},
            roots=[tmp_path],
            # Reads the link address and contradicts it — that, not an
            # abstention, is what licenses the search.
            verifier=lambda unit, vaddr, size: b"\x00" * size,
            searcher=lambda blob: 0x8B00 if blob == text else None,
        )
    finally:
        elfdisc.segments_of = original
    brisc = next(d for d in found.elfs if d.unit == "BRISC" and d.role == "kernel")
    assert brisc.bias == 0x8B00 - 0x49A0
    assert brisc.how.startswith("relocated +0x")


def test_an_unreadable_link_address_does_not_trigger_a_search(tmp_path):
    """NCRISC's kernel links into its private IRAM, which a tile cannot read
    back. Searching L1 anyway finds the staging copy the host DMAs from and
    yields a bias for an address the core never executes — the run's whole
    NCRISC attribution then resolves to nothing."""
    _fake_cache(tmp_path, "only")
    text = bytes(range(256))
    original = elfdisc.segments_of
    elfdisc.segments_of = lambda path: [(0xFFC00000, text)]
    searched = []

    def searcher(blob):
        searched.append(blob)
        return 0x8D20

    try:
        found = elfdisc.discover(
            env={},
            roots=[tmp_path],
            verifier=lambda unit, vaddr, size: None,  # abstain
            searcher=searcher,
        )
    finally:
        elfdisc.segments_of = original
    assert searched == [], "an abstention must not license a relocation search"
    ncrisc = next(d for d in found.elfs if d.unit == "NCRISC" and d.role == "kernel")
    assert ncrisc.bias == 0
    assert ncrisc.how == "recent"


def test_a_bias_shifts_every_indexed_address(dwarf_elf):
    """The bias must move the line table, the function ranges *and* the
    coverage bound together, or a relocated ELF resolves to nothing."""
    plain = DwarfIndex()
    plain.load(dwarf_elf)
    shifted = DwarfIndex()
    shifted.load(dwarf_elf, bias=0x10000)
    pc = sorted(plain._by_unit[None])[len(plain._by_unit[None]) // 2]
    assert shifted.lookup(pc) is None
    assert shifted.lookup(pc + 0x10000) == plain.lookup(pc)
    assert shifted.covers(pc + 0x10000) is True


def test_a_unit_whose_code_matches_nothing_is_rejected_not_guessed(tmp_path):
    """When the device *can* be read and no candidate is what ran, naming
    that unit from the newest ELF would be a confidently wrong answer."""
    _fake_cache(tmp_path, "wrong")

    def verifier(unit, vaddr, size):
        return b"\xaa" * size

    original = elfdisc.segments_of
    elfdisc.segments_of = lambda path: [(0x1000, b"\xbb" * 8)]
    try:
        found = elfdisc.discover(env={}, roots=[tmp_path], verifier=verifier)
    finally:
        elfdisc.segments_of = original
    assert {d.how for d in found.elfs} == {elfdisc.REJECTED}


def test_an_unreadable_address_abstains_rather_than_rejecting(tmp_path):
    """NCRISC runs out of a private IRAM window a tile may not read back.
    Abstaining must fall back to recency, not refuse the core outright."""
    _fake_cache(tmp_path, "only")
    original = elfdisc.segments_of
    elfdisc.segments_of = lambda path: [(0xFFC00000, b"\xbb" * 8)]
    try:
        found = elfdisc.discover(
            env={}, roots=[tmp_path], verifier=lambda unit, vaddr, size: None
        )
    finally:
        elfdisc.segments_of = original
    assert {d.how for d in found.elfs} == {"recent"}


def test_no_cache_is_a_stated_note_not_a_crash():
    found = elfdisc.discover(env={"TT_METAL_CACHE": "/nonexistent-cache"}, roots=None)
    if not found.elfs:
        assert found.note, "a run that could not attribute must say why"
