"""Tests for the per-unit cycle-cost tables and their loader.

Runs standalone (``python3 -m tt_sim.perf.costs_test``) or under pytest.

Three things are pinned here, in decreasing order of how much they matter:

1. **Provenance integrity.** Every entry declares where its numbers came from,
   every sourced entry names a document that the file actually declares, and
   every unsourced entry explains itself in prose and carries no numbers. This
   is the property the whole exercise exists for -- see ROADMAP.md
   "Positioning": tt-sim is a cycle-*approximate* estimator, so a table that
   could not distinguish a documented constant from a placeholder would be
   unimprovable.
2. **The loader returns what the file says.** Spot checks against the ISA docs'
   own tables, plus the arch-resolution rules (overrides deep-merge, one-arch
   instructions are dropped for the other arch).
3. **Who consumes it.** A test pins the exact set of modules outside this
   package that name the tables — one unit so far, the Tensix matrix unit
   (Phase 5 of ``docs/plans/event-driven-pump.md``) — so wiring a cost into
   the execution path has to be a deliberate act rather than a drive-by
   import.
"""

import pathlib
import re

import yaml

from tt_sim.perf.costs import (
    ARCHITECTURES,
    BOUNDS,
    ENTRY_KEYS,
    PROVENANCE_RANK,
    PROVENANCE_REQUIRING_DERIVATION,
    PROVENANCE_REQUIRING_NOTE,
    PROVENANCE_REQUIRING_SOURCE,
    CostTable,
    CycleCost,
    load_costs,
    raw_tables,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TENSIX_YAML = _REPO_ROOT / "tt_sim" / "pe" / "tensix" / "tensix_instruction_costs.yaml"
_UNIT_YAML = _REPO_ROOT / "tt_sim" / "perf" / "unit_costs.yaml"
_INSTRUCTIONS_YAML = (
    _REPO_ROOT / "tt_sim" / "pe" / "tensix" / "tensix_instructions.yaml"
)


def _load_raw():
    return (
        yaml.safe_load(_TENSIX_YAML.read_text()),
        yaml.safe_load(_UNIT_YAML.read_text()),
    )


def _declared_documents(raw):
    docs = dict(raw.get("documents") or {})
    docs.update(raw.get("vendor_documents") or {})
    return docs


def _walk_provenanced(node, path=()):
    """Yield ``(path, mapping)`` for every mapping carrying a ``provenance``."""
    if isinstance(node, dict):
        if "provenance" in node:
            yield path, node
        for key, value in node.items():
            yield from _walk_provenanced(value, path + (str(key),))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_provenanced(value, path + (str(i),))


# ---------------------------------------------------------------------------
# 1. The files parse, and their shape is what the schema says.
# ---------------------------------------------------------------------------


def test_both_yaml_files_parse():
    tensix, units = _load_raw()
    assert tensix["schema_version"] == 1
    assert units["schema_version"] == 1
    assert set(tensix["units"]) == {
        "MATH",
        "SFPU",
        "THCON",
        "SYNC",
        "TDMA",
        "CFG",
        "PACK",
        "UNPACK",
        "XMOV",
        "NONE",
    }
    assert {"clock", "noc", "dram", "riscv", "mover", "l1"} <= set(units)


def test_unit_keys_are_the_ex_resource_names_the_instruction_table_uses():
    tensix, _ = _load_raw()
    instructions = yaml.safe_load(_INSTRUCTIONS_YAML.read_text())
    ex_resources = {
        str(v.get("ex_resource")).split("#")[0].strip() for v in instructions.values()
    }
    assert set(tensix["units"]) == ex_resources


def test_arch_overrides_declare_every_architecture():
    for raw in _load_raw():
        assert set(raw["arch_overrides"]) == set(ARCHITECTURES)


def test_entries_carry_no_unrecognised_keys():
    tensix, _ = _load_raw()
    for unit_name, unit in tensix["units"].items():
        for name, entry in (unit.get("instructions") or {}).items():
            stray = set(entry) - ENTRY_KEYS
            assert not stray, f"{unit_name}.{name} has unknown keys {sorted(stray)}"


# ---------------------------------------------------------------------------
# 2. Provenance integrity -- the point of the exercise.
# ---------------------------------------------------------------------------


def test_every_tensix_instruction_declares_a_provenance():
    tensix, _ = _load_raw()
    for unit_name, unit in tensix["units"].items():
        for name, entry in (unit.get("instructions") or {}).items():
            assert "provenance" in entry, f"{unit_name}.{name} has no provenance"
            assert entry["provenance"] in PROVENANCE_RANK, (
                f"{unit_name}.{name} has unknown provenance {entry['provenance']!r}"
            )


def test_every_provenanced_mapping_anywhere_uses_the_vocabulary():
    """Unit-level and section-level blocks carry provenance too, not just
    instruction entries -- the NoC hop table, the DRAM bandwidth block, the
    Mover transfer rates. None of them may opt out."""
    for raw in _load_raw():
        for path, node in _walk_provenanced(raw):
            assert node["provenance"] in PROVENANCE_RANK, (
                f"{'.'.join(path)} has unknown provenance {node['provenance']!r}"
            )


def test_sourced_entries_name_a_declared_document():
    for raw in _load_raw():
        declared = _declared_documents(raw)
        for path, node in _walk_provenanced(raw):
            if node["provenance"] not in PROVENANCE_REQUIRING_SOURCE:
                continue
            where = ".".join(path) or "<root>"
            assert "source" in node, f"{where} is sourced but names no document"
            doc_id = re.split(r"[#:]", str(node["source"]))[0]
            assert doc_id in declared, f"{where} cites undeclared document {doc_id!r}"


def test_unsourced_entries_explain_themselves():
    for raw in _load_raw():
        for path, node in _walk_provenanced(raw):
            if node["provenance"] not in PROVENANCE_REQUIRING_NOTE:
                continue
            where = ".".join(path) or "<root>"
            note = (node.get("note") or "").strip()
            assert note, f"{where} is unsourced but has no note"


def test_unknown_entries_carry_no_numbers():
    """A gap must read as a gap. An entry that says 'unknown' and then supplies
    a cycle count is the exact failure this convention is meant to prevent."""
    for raw in _load_raw():
        for path, node in _walk_provenanced(raw):
            if node["provenance"] != "unknown":
                continue
            numeric = {
                k: v
                for k, v in node.items()
                if k not in {"provenance", "note", "source", "arch"}
            }
            assert not numeric, (
                f"{'.'.join(path)} claims unknown provenance but carries {sorted(numeric)}"
            )


def test_derived_entries_show_their_working():
    """Both derived ranks, everywhere they appear. A derived number is only
    reviewable if the arithmetic is written down — which is the whole
    difference between ``isa_doc_derived`` / ``vendor_source_derived`` and a
    number somebody arrived at privately."""
    for raw in _load_raw():
        for path, node in _walk_provenanced(raw):
            if node["provenance"] not in PROVENANCE_REQUIRING_DERIVATION:
                continue
            where = ".".join(path) or "<root>"
            assert node.get("derivation"), (
                f"{where} is derived but does not say from what"
            )


def test_vendor_derived_entries_are_exactly_the_ones_we_expect():
    """``vendor_source_derived`` is arithmetic on vendor numbers — below a
    published figure, above a guess. It exists for exactly two entries — the
    same subtraction once per arch — and adding a third must be as deliberate
    as adding an ``estimated`` one, so the list is here rather than inferred.
    Every one must also show its working, which
    :func:`test_derived_entries_show_their_working` enforces."""
    expected = {
        "arch_overrides.wormhole.dram.access_latency",
        "arch_overrides.blackhole.dram.access_latency",
    }
    found = set()
    for raw in _load_raw():
        for path, node in _walk_provenanced(raw):
            if node["provenance"] == "vendor_source_derived":
                found.add(".".join(path))
    assert found == expected


def test_the_dram_latency_is_exactly_its_own_derivation():
    """The one number in these files that is neither published nor guessed, so
    the arithmetic is checked rather than described: DRAM access latency is the
    measured DRAM end-to-end figure minus the measured L1-remote-read one, both
    from the same vendor table, which is why the NoC round trip and the issuing
    core's path cancel out of it. If either reference figure is corrected, this
    fails instead of drifting.

    Both arches now, and the *same* arithmetic on the *same* four constants,
    which is the point: Blackhole's was a gap until 2026-08-03 and closing it
    with a different source or a different subtraction would have made the two
    figures incommensurable."""
    _, units = _load_raw()
    reference = units["dram"]["end_to_end_reference"]
    for arch in ("wormhole", "blackhole"):
        derived = units["arch_overrides"][arch]["dram"]["access_latency"]
        assert (
            derived["cycles"]
            == reference["dram_cycles"][arch] - reference["l1_remote_read_cycles"][arch]
        )
        # A floor, not an equals sign: the L1 endpoint's own service time is
        # folded out by the subtraction and is not negative.
        assert derived["bound"] == "at_least"
    # The base entry stays ``unknown``, so a third architecture inherits a gap
    # rather than one of these two numbers.
    assert units["dram"]["access_latency"]["provenance"] == "unknown"


def test_the_dram_channel_rate_is_exactly_its_own_derivation():
    """The second derived DRAM number, and a much cheaper one than the first:
    the channel's bytes per cycle is the published per-channel GB/s at the
    published clock, which is a unit conversion and nothing else. Both inputs
    are ``isa_doc``, so the result is ``isa_doc_derived`` — not
    ``vendor_source_derived``, and emphatically not fitted: 24 GB/s was in this
    file, unconsumed, long before rung 2 measured a DRAM read plateauing at
    24.38 B/cycle.

    The Blackhole half is the one worth having a test for. Its override says
    ``unknown``, but the overrides *deep-merge*, so Wormhole's 24 is still
    physically present under it — and a consumer that read the number without
    checking the provenance beside it would launder one architecture's
    published figure into another's gap. That is the exact failure the
    convention exists to prevent, and it is invisible from the raw YAML."""
    _, units = _load_raw()
    gb_per_s = units["dram"]["bandwidth"]["per_channel_gb_per_s"]
    ghz = units["clock"]["wormhole"]["frequency_mhz"] / 1000
    assert units["dram"]["channel_serialisation"]["bytes_per_cycle"] == gb_per_s / ghz
    # Blackhole: the override carries no number of its own...
    override = units["arch_overrides"]["blackhole"]["dram"]["channel_serialisation"]
    assert override["provenance"] == "unknown"
    assert "bytes_per_cycle" not in override
    # ...but the merged view still shows Wormhole's, so provenance is the only
    # thing standing between a consumer and the wrong architecture's number.
    merged = load_costs("blackhole").sections["dram"]["channel_serialisation"]
    assert merged["bytes_per_cycle"] == 24
    assert merged["provenance"] == "unknown"


def test_no_entry_is_an_uncalibrated_guess_without_saying_so():
    """There should be no ``estimated`` entries at all right now: everything in
    the tables is either published or explicitly absent. If one is added, this
    test does not stop it -- it makes it visible, since adding an estimate
    means editing this list."""
    expected_estimates = set()
    found = set()
    for raw in _load_raw():
        for path, node in _walk_provenanced(raw):
            if node["provenance"] == "estimated":
                found.add(".".join(path))
    assert found == expected_estimates


def test_bounds_are_from_the_vocabulary():
    for raw in _load_raw():
        for path, node in _walk_provenanced(raw):
            for key in ("latency", "occupancy"):
                value = node.get(key)
                if isinstance(value, dict):
                    assert value.get("bound", "exact") in BOUNDS, (
                        f"{'.'.join(path)}.{key} has unknown bound"
                    )


# ---------------------------------------------------------------------------
# 3. The loader returns what the file says.
# ---------------------------------------------------------------------------


def test_load_costs_returns_a_cost_table_for_each_arch():
    for arch in ARCHITECTURES:
        table = load_costs(arch)
        assert isinstance(table, CostTable)
        assert table.arch == arch


def test_matrix_unit_matches_the_isa_doc_latency_table():
    """WormholeB0/.../MatrixUnit.md "Instruction latency and throughput"."""
    math = load_costs("wormhole").unit("MATH")
    for op in ("MVMUL", "DOTPV", "GAPOOL", "ELWMUL", "GMPOOL", "ELWADD", "ELWSUB"):
        assert math[op].latency.cycles == 5, op
        assert math[op].throughput_ipc == 1, op
        assert math[op].occupancy.cycles == 1, op
    for op in ("SETRWC", "INCRWC", "CLEARDVALID", "SHIFTXA", "ZEROACC", "TRNSPSRCB"):
        assert math[op].latency.cycles == 1, op
    assert math["SHIFTXB"].throughput_ipc == 0.5
    assert math["SHIFTXB"].occupancy.cycles == 2
    assert math["MOVD2A"].latency.cycles == 2
    for op in ("MOVA2D", "MOVDBGA2D", "MOVB2D", "MOVB2A"):
        assert math[op].latency.cycles == 4, op


def test_fidelity_phases_are_recorded_but_not_folded_into_mvmul():
    """An MVMUL's cost depends on the fidelity phase count, but the ISA docs
    give no cycles-per-phase figure -- only "one instruction per phase". So the
    dependency is a marker on the instruction and the numbers live at unit
    level, tagged vendor_source because they come from tt-metal, not the ISA
    docs."""
    math = load_costs("wormhole").unit("MATH")
    assert math["MVMUL"].scales_with == "fidelity_phases"
    assert math["GMPOOL"].scales_with is None
    phases = math.extras["fidelity_phases"]
    assert phases["provenance"] == "vendor_source"
    assert phases["cycles_per_tile"] == {
        "LoFi": 16,
        "HiFi2": 32,
        "HiFi3": 48,
        "HiFi4": 64,
    }


def test_scalar_unit_occupancies_and_their_bounds():
    """ScalarUnit.md's column is literally "Number of cycles required for
    execution", so these are occupancies rather than latencies, and the ">="
    and "3 or 4" of the source survive into the data."""
    thcon = load_costs("wormhole").unit("THCON")
    assert thcon["DMANOP"].occupancy == CycleCost(1, "exact")
    assert thcon["REG2FLOP"].occupancy == CycleCost(2, "at_least")
    assert thcon["ADDDMAREG"].occupancy == CycleCost(3, "range", 4)
    assert thcon["ATCAS"].occupancy == CycleCost(15, "at_least")
    assert thcon["LOADIND"].occupancy.bound == "at_least"
    assert thcon["RSTDMA"].provenance == "unknown"
    assert not thcon["RSTDMA"].has_costs


def test_vector_unit_two_cycle_ops_are_the_arithmetic_ones():
    sfpu = load_costs("wormhole").unit("SFPU")
    two_cycle = {
        name
        for name, entry in sfpu.instructions.items()
        if entry.latency is not None and entry.latency.cycles == 2
    }
    assert two_cycle == {
        "SFPADDI",
        "SFPADD",
        "SFPMAD",
        "SFPMUL",
        "SFPMULI",
        "SFPLUT",
        "SFPLUTFP32",
        "SFPSWAP",
        "SFPSHFT2",
        "SFPCONFIG",
    }
    assert sfpu["SFPCONFIG"].latency.bound == "at_most"
    assert sfpu["SFPLOADMACRO"].latency is None


def test_unpacker_records_both_halves_of_its_cost():
    unpack = load_costs("wormhole").unit("UNPACK")
    assert unpack["UNPACR"].occupancy == CycleCost(2, "at_least")
    modes = unpack.extras["l1_bandwidth"]["throttle_modes"]
    assert [modes[k]["bytes_per_cycle"] for k in ("x1", "x2", "x4")] == [16, 32, 64]
    assert unpack.extras["joint_bandwidth"]["bytes_per_cycle"] == 80


def test_noc_hop_latencies_are_the_published_ones():
    hops = load_costs("wormhole").section("noc")["hops"]
    assert hops["router_to_router"]["latency"] == 9
    assert hops["niu_to_router"]["latency"] == {"cycles": 5, "bound": "approximate"}
    assert CycleCost.parse(hops["niu_to_router"]["latency"]).bound == "approximate"


def test_riscv_pipeline_costs_are_the_published_ones():
    rv = load_costs("wormhole").section("riscv")
    assert rv["issue"]["instructions_per_cycle"] == 1
    assert rv["integer_unit"]["multiply"] == 2
    assert CycleCost.parse(rv["integer_unit"]["divide_general"]) == CycleCost(
        6, "range", 33
    )
    assert rv["load_latency"]["core_local_data_ram"] == 2
    assert CycleCost.parse(rv["load_latency"]["l1"]) == CycleCost(8, "at_least")
    assert rv["store_throughput"]["l1_period_cycles"] == 5


def test_dram_latency_is_derived_on_both_arches_and_says_so():
    """This entry was ``unknown`` on both arches, then derived on Wormhole
    alone, and is now derived on both. The rank matters as much as the number:
    a reader must be able to tell at a glance that nobody published 99 or 126.

    Blackhole's was blocked on two things and neither turned out to be about
    Blackhole's DRAM — a clock conflict that does not touch a measured cycle
    count, and a consistency failure confined to the one end-to-end row the
    subtraction does not use. See the entry's ``derivation``."""
    wormhole = load_costs("wormhole").section("dram")
    assert wormhole["access_latency"]["provenance"] == "vendor_source_derived"
    assert wormhole["access_latency"]["cycles"] == 99
    assert wormhole["access_latency"]["derivation"]

    blackhole = load_costs("blackhole").section("dram")
    assert blackhole["access_latency"]["provenance"] == "vendor_source_derived"
    assert blackhole["access_latency"]["cycles"] == 126
    assert blackhole["access_latency"]["derivation"]

    # The end-to-end measurements are kept under their own key, still not to be
    # plugged in as a DRAM latency: 358 folds in the NoC round trip and the
    # issuing core's path, and it is the *difference* of two of these rows —
    # not any single one of them — that the derivation above uses.
    assert wormhole["end_to_end_reference"]["provenance"] == "vendor_source"
    assert wormhole["end_to_end_reference"]["dram_cycles"]["wormhole"] == 358


def test_mover_records_ideal_and_contended_rates():
    mover = load_costs("wormhole").section("mover")
    assert mover["issue"]["occupancy"] == 1
    l1 = mover["transfer"]["l1_to_l1"]
    assert l1["ideal"]["bits_per_cycle"] == 93.1
    assert l1["contended"]["bits_per_cycle"] == 32


# ---------------------------------------------------------------------------
# 4. Architecture resolution.
# ---------------------------------------------------------------------------


def test_blackhole_only_instructions_are_absent_on_wormhole():
    wh, bh = load_costs("wormhole"), load_costs("blackhole")
    for unit, op in (
        ("SFPU", "SFPMUL24"),
        ("SFPU", "SFPARECIP"),
        ("SFPU", "SFPGT"),
        ("SFPU", "SFPLE"),
        ("CFG", "CFGSHIFTMASK"),
        ("CFG", "STREAMWRCFG"),
        ("SYNC", "STREAMWAIT"),
        ("MATH", "MOVDBGB2D"),
        ("NONE", "RESOURCEDECL"),
    ):
        assert op not in wh.unit(unit), f"{op} should not exist on Wormhole"
        assert op in bh.unit(unit), f"{op} should exist on Blackhole"


def test_shared_instructions_are_identical_across_arches():
    wh, bh = load_costs("wormhole"), load_costs("blackhole")
    for op in ("MVMUL", "SFPMAD", "ATCAS", "UNPACR", "XMOV"):
        assert wh.find(op).latency == bh.find(op).latency, op
        assert wh.find(op).occupancy == bh.find(op).occupancy, op


def test_arch_overrides_deep_merge_rather_than_replace():
    """Blackhole overrides SETC16 alone; the rest of the CFG unit survives."""
    wh, bh = load_costs("wormhole"), load_costs("blackhole")
    assert wh.unit("CFG")["SETC16"].throughput_ipc is None
    assert bh.unit("CFG")["SETC16"].throughput_ipc == 3
    assert bh.unit("CFG")["WRCFG"].latency.cycles == 2
    assert bh.unit("CFG")["RDCFG"].latency.bound == "at_least"


def test_arch_differences_in_the_non_tensix_units():
    wh, bh = load_costs("wormhole"), load_costs("blackhole")
    assert wh.section("clock")["wormhole"]["frequency_mhz"] == 1000
    assert bh.section("clock")["blackhole"]["frequency_mhz"] == 1350
    assert wh.section("noc")["flit_bits"] == 256
    assert bh.section("noc")["flit_bits"] == 512
    # Hop latency is unchanged between the arches; only the flit width moved.
    assert wh.section("noc")["hops"]["router_to_router"]["latency"] == 9
    assert bh.section("noc")["hops"]["router_to_router"]["latency"] == 9
    # Blackhole's link bandwidth in GB/s is deliberately not carried over: the
    # override marks it ``unknown`` rather than scaling Wormhole's 32.
    assert bh.section("noc")["link_bandwidth_gb_per_s"]["provenance"] == "unknown"
    assert wh.section("riscv")["integer_unit"]["branch_mispredict_bubble"] == 2
    assert bh.section("riscv")["integer_unit"]["branch_mispredict_bubble"] == 4
    assert wh.section("riscv")["integer_unit"]["multiply"] == 2
    assert bh.section("riscv")["integer_unit"]["multiply"] == 1
    assert bh.section("riscv")["pipeline"]["stages"][1] == "ex1"


def test_the_nocs_two_bandwidth_figures_are_one_fact_and_agree():
    """``flit_bits`` at one flit per cycle and ``link_bandwidth_gb_per_s`` are
    the same number written two ways, and the file records both without either
    citing the other: 256 bits per cycle at the ``clock`` section's 1 GHz is
    exactly 32 GB/s. Cheap, and it is the check that says the flit rate is safe
    to spend as bandwidth (``tt_sim/perf/model.py``'s ``NocCostModel``) rather
    than being a packet-format detail that happens to be in the same block."""
    wh = load_costs("wormhole")
    bytes_per_cycle = wh.section("noc")["flit_bits"] / 8
    assert wh.section("noc")["hops"]["niu_to_router"]["throughput_flits_per_cycle"] == 1
    ghz = wh.section("clock")["wormhole"]["frequency_mhz"] / 1000
    assert bytes_per_cycle * ghz == wh.section("noc")["link_bandwidth_gb_per_s"]


def test_loading_one_arch_does_not_perturb_the_other():
    """The YAML uses anchors, so aliased entries are the *same* dict object
    after parsing; a merge that mutated in place would corrupt every alias."""
    before = load_costs("wormhole").unit("SFPU")["SFPADD"].latency.cycles
    load_costs("blackhole")
    assert load_costs("wormhole").unit("SFPU")["SFPADD"].latency.cycles == before
    assert load_costs("wormhole").unit("SFPU")["SFPABS"].latency.cycles == 1


def test_unknown_arch_is_rejected():
    try:
        load_costs("grayskull")
    except ValueError:
        return
    raise AssertionError("an unknown architecture should raise")


# ---------------------------------------------------------------------------
# 5. Coverage against the instruction table.
# ---------------------------------------------------------------------------


def test_every_tensix_instruction_is_costed_or_named_as_undocumented():
    """The strongest guarantee available without silicon: no opcode in
    ``tensix_instructions.yaml`` is silently missing from the cost table. It is
    either costed, or explicitly listed as absent from the ISA docs."""
    instructions = yaml.safe_load(_INSTRUCTIONS_YAML.read_text())
    table = load_costs("blackhole")  # the superset arch
    covered = set()
    for unit in table.units.values():
        covered |= set(unit.instructions)
        covered |= set(unit.not_documented)
    missing = set(instructions) - covered
    assert not missing, f"uncosted and unlisted: {sorted(missing)}"


def test_cost_table_invents_no_instructions():
    instructions = set(yaml.safe_load(_INSTRUCTIONS_YAML.read_text()))
    table = load_costs("blackhole")
    for unit in table.units.values():
        for name in unit.instructions:
            assert name in instructions, f"{name} is not in tensix_instructions.yaml"
        for name in unit.not_documented:
            assert name in instructions, f"{name} is not in tensix_instructions.yaml"


def test_a_unit_disagreeing_with_ex_resource_is_explained_not_silent():
    """DMANOP is ``ex_resource: TDMA`` in ``tensix_instructions.yaml``, but the
    ISA docs put it in the Scalar Unit (ThCon) and give it a cost there. Rather
    than pick a side silently, both units carry an entry: ThCon has the number,
    TDMA has a note pointing at it. Any such disagreement must be written down,
    and an instruction may only be costed once."""
    instructions = yaml.safe_load(_INSTRUCTIONS_YAML.read_text())
    table = load_costs("blackhole")
    costed_in = {}
    for unit_name, unit in table.units.items():
        for name, entry in unit.instructions.items():
            declared = str(instructions[name].get("ex_resource")).split("#")[0].strip()
            if declared != unit_name:
                assert entry.note, (
                    f"{unit_name}.{name} contradicts ex_resource {declared} "
                    "without explaining why"
                )
            if entry.has_costs:
                assert name not in costed_in, (
                    f"{name} is costed twice: {costed_in[name]} and {unit_name}"
                )
                costed_in[name] = unit_name
    assert costed_in["DMANOP"] == "THCON"


def test_unsourced_lists_exactly_the_entries_without_numbers_we_expect():
    table = load_costs("blackhole")
    assert {e.name for e in table.unsourced()} == {
        "MOVD2B",
        "TRNSPSRCA",
        "MOVDBGB2D",
        "RSTDMA",
        "STREAMWAIT",
        "DMANOP",  # the TDMA cross-reference; costed under THCON
        "TBUFCMD",
        "RESOURCEDECL",
    }


# ---------------------------------------------------------------------------
# 6. Who consumes the tables, and nobody else.
# ---------------------------------------------------------------------------

#: Every module outside ``tt_sim/perf/`` that may name the cost tables. This
#: list started empty — the tables were dead data by design until Phase 5 of
#: ``docs/plans/event-driven-pump.md`` — and it grows one unit at a time,
#: deliberately. The matrix unit was the first, and everything any of them
#: needs lives in ``tt_sim/perf/model.py``, so the next unit adds itself here
#: rather than growing its own reading of the YAML.
EXPECTED_CONSUMERS = {
    # The FPU: MVMUL and friends charged the table's occupancy, fidelity-phase
    # aware. See MatrixUnit.instruction_occupancy.
    "tt_sim/pe/tensix/backends/matrix.py",
    # The SFPU. Best-sourced unit in the file (a published latency for all 42
    # opcodes) and every one of them is a 1-cycle *occupancy*: the sub-units
    # are pipelined, so latency 2 is time-to-result, not time-held.
    "tt_sim/pe/tensix/backends/vector.py",
    # ThCon. The only unit whose ISA-doc table is already an occupancy table,
    # and the first place in tt-sim where an op costs more than one cycle.
    "tt_sim/pe/tensix/backends/thcon.py",
    # The packers: PACR's documented one-cycle *issue* cost only. The drain has
    # no published figure and is charged nothing.
    "tt_sim/pe/tensix/backends/packer.py",
    # The sync unit: one cycle throughout; the real costs are wait-gate stalls,
    # which are not occupancy.
    "tt_sim/pe/tensix/backends/sync.py",
    # Only mentions the model in prose: the ``cost_model`` attribute and the
    # ``instruction_occupancy`` hook every unit inherits, whose default is now
    # the straight table lookup the five constant-cost units above rely on.
    "tt_sim/pe/tensix/backends/backend_base.py",
    "tt_sim/pe/tensix/matrix_cost_model_test.py",
    "tt_sim/pe/tensix/backend_cost_model_test.py",
    # The baby RISC-V cores' load/store path — the first consumer outside the
    # Tensix coprocessor, and the first to read the tables as something other
    # than a per-opcode occupancy: ``riscv.load_latency`` is a latency table
    # driving a load-use scoreboard, not a table of cycles a unit is held.
    # ``rv32.py``, ``babyriscv.py`` and ``cost_test.py`` deliberately do not
    # appear here; they reach the model only through ``tt_sim/pe/rv/cost.py``,
    # which is the single point where the RV side names the tables at all.
    "tt_sim/pe/rv/cost.py",
    # The NoC, and the first consumer that is not a *unit* at all: the cost is
    # a function of the distance between two endpoints, so ``NUI`` reads the
    # ``noc.hops`` block and spends it as flight time on a packet rather than
    # as occupancy on anything. Hop counting is topology and stays here, in
    # ``noc_hop_count``; the table only supplies the two constants.
    "tt_sim/network/tt_noc.py",
    "tt_sim/network/noc_cost_model_test.py",
    # DRAM, and the first cost charged at an *endpoint* rather than to a unit
    # or to a flight: ``DRAMEndpointNUI`` holds an arriving request for the
    # device's own service time before the channel answers it. Separate from
    # the NoC consumer above on purpose — a flight time is a property of the
    # distance, this is a property of the device, and keeping them apart is
    # what lets the number be derived without double-counting the hops.
    "tt_sim/device/tiles.py",
    "tt_sim/device/dram_cost_model_test.py",
    # The trace writers, and the only consumers that read the tables to
    # *describe* a run rather than to charge it: nothing here can move a cycle.
    # They ask ``cost_model_enabled()`` one question — which timing regime
    # produced this trace — so that a duration can say whether it is a modelled
    # figure or the flat 1 an un-modelled unit really did retire in. That
    # distinction is the whole reason they appear here: a writer that guessed
    # would fabricate performance data, which is worse than emitting none.
    "tt_sim/trace/writers/perfetto.py",
    "tt_sim/trace/writers/noc_parquet.py",
    # Prose only — ``ComputeEvent.duration`` documents which table its cycles
    # came from. No import; the event is a plain dataclass field populated by
    # ``TensixBackendUnit.clock_tick``, which is already on this list.
    "tt_sim/trace/events.py",
}

#: The Tensix backend units that are *not* wired to the tables, and why. Kept
#: next to the allow-list so "which units are costed" is one thing to read.
UNWIRED_UNITS = {
    # Owned by concurrent work on its row-stride path at the time this landed;
    # its cost is also the one genuinely non-constant entry in the file (a
    # ">= 2" address phase plus a throttle-mode-dependent data phase), so it
    # wants more than the flat lookup the other units use.
    "UNPACK": "tt_sim/pe/tensix/backends/unpacker.py",
    # The one omission that is a finding rather than a scoping decision.
    # RDCFG's documented ">= 2" is the only cost in this unit's table above one
    # cycle, and charging it makes `matmulblock` compute a wrong answer -- a
    # config write lands one cycle late and something reads it stale. See the
    # comment in config.py; the divergence is tt-sim's, not the table's, and
    # charging RDCFG a 1 the docs do not give it would only hide it.
    "CFG": "tt_sim/pe/tensix/backends/config.py",
    # XMOV's 1-cycle entry is the issue cost only; the transfer duration is
    # bandwidth-derived and lives in tt_sim/perf/unit_costs.yaml under
    # ``mover``, which nothing consumes yet.
    "XMOV": "tt_sim/pe/tensix/backends/mover.py",
    # Every Miscellaneous Unit op is one cycle by one blanket sentence, so
    # wiring it would charge nothing; left out to keep the allow-list honest
    # about which units the model has actually been reasoned about for.
    "TDMA": "tt_sim/pe/tensix/backends/misc.py",
}


def test_the_cost_tables_have_exactly_the_consumers_we_expect():
    """Reading a cost into the execution path must be a deliberate change that
    trips this test, not a drive-by import. Timing is validated here by
    byte-identical replay and a pinned matmul PCC, so a new consumer is a
    change to the thing those guards measure and should be reviewed as one."""
    consumers = set()
    for path in sorted(_REPO_ROOT.rglob("*.py")):
        if "tt_sim/perf/" in path.as_posix():
            continue
        if any(part in {".git", "build", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(errors="ignore")
        if "tt_sim.perf" in text or "tensix_instruction_costs" in text:
            consumers.add(path.relative_to(_REPO_ROOT).as_posix())
    assert consumers == EXPECTED_CONSUMERS


def test_the_consumers_only_reach_the_tables_through_the_model():
    """``tt_sim/perf/model.py`` is where the three judgement calls live (no
    entry means no opinion, a bound is not an equals sign, derived is not
    measured). A unit that loaded ``load_costs`` itself would be making them
    again, differently."""
    for relative in EXPECTED_CONSUMERS:
        text = (_REPO_ROOT / relative).read_text()
        assert "tt_sim.perf.costs" not in text, relative


def test_every_unit_with_a_backend_is_either_wired_or_named_as_unwired():
    """No unit gets to be neither. The table names a ``backend`` file for every
    unit that has one, so "which units read their own costs" is answerable from
    the data rather than from memory, and dropping a unit off both lists — the
    way a partial rollout goes quietly stale — fails here."""
    unwired = set(UNWIRED_UNITS.values())
    assert not (unwired & EXPECTED_CONSUMERS)
    for name, unit in load_costs("blackhole").units.items():
        if unit.backend is None:
            assert name == "NONE", name  # the only unit with no backend at all
            continue
        assert (_REPO_ROOT / unit.backend).exists(), unit.backend
        assert unit.backend in EXPECTED_CONSUMERS or unit.backend in unwired, name


def test_raw_tables_exposes_the_unresolved_yaml():
    tensix, units = raw_tables()
    assert "arch_overrides" in tensix
    assert "arch_overrides" in units


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    table = load_costs("wormhole")
    costed = sum(1 for e in table.entries() if e.has_costs)
    print(
        f"costs_test OK: {costed} costed Tensix instructions on wormhole, "
        f"{len(table.unsourced())} honest gaps, "
        f"{len(load_costs('blackhole').units)} units, provenance verified"
    )


if __name__ == "__main__":
    main()
