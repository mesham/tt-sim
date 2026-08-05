"""Tests for the NoC congestion experiment planner.

The planner's whole job is to refuse a confounded experiment, so most of what is
tested here is that it *does* refuse: a plan in which two of {flow count, path
length, shared links} move together is the same unidentifiable thing tt-metal's
shipped dataset is, and shipping one would be worse than shipping nothing.

The routing model is cross-checked against ``tt_sim.network.tt_noc`` rather than
re-derived, so a planner that drifted from the simulator's own hop counting
fails here instead of quietly reporting agreement with itself.
"""

import pathlib

import pytest
import yaml

from tt_sim.network.tt_noc import noc_hop_count
from tt_sim.perf import noc_congestion_plan as plan


def _grid_text(arch, translated=False):
    cols, rows = plan.WORKER_COLUMNS[arch], plan.WORKER_ROWS[arch]
    gx, gy = plan.SOC_GRID[arch]
    lines = [
        f"# nocbench-grid arch={arch} soc_grid_x={gx} soc_grid_y={gy} "
        f"logical_grid_x={len(cols)} logical_grid_y={len(rows)}",
        "log_x,log_y,noc_x,noc_y",
    ]
    for ly, py in enumerate(rows):
        for lx, px in enumerate(cols):
            if translated:
                lines.append(f"{lx},{ly},{18 + lx},{18 + ly}")
            else:
                lines.append(f"{lx},{ly},{px},{py}")
    return "\n".join(lines) + "\n"


@pytest.fixture(params=["wormhole", "blackhole"])
def grid(request, tmp_path):
    path = tmp_path / "grid.csv"
    path.write_text(_grid_text(request.param))
    return plan.load_grid(path)


# --- routing ---------------------------------------------------------------


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_route_length_is_the_simulators_own_hop_count(arch):
    """The planner must not have its own opinion about how far anything is."""
    gx, gy = plan.SOC_GRID[arch]
    for src in ((0, 0), (3, 5), (gx - 1, gy - 1)):
        for dst in ((0, 0), (1, 2), (gx - 2, 4), (2, gy - 1)):
            assert len(plan.route_links(src, dst, gx, gy)) == noc_hop_count(
                src, dst, gx, gy
            )


def test_routing_is_dimension_ordered_x_then_y():
    links = plan.route_links((0, 0), (2, 2), 8, 8)
    assert links == (("X", 0, 0), ("X", 0, 1), ("Y", 2, 0), ("Y", 2, 1))


def test_routing_wraps_rather_than_turning_round():
    # 6 -> 1 on a ring of 8 goes forwards through 7 and 0, not backwards.
    links = plan.route_links((6, 0), (1, 0), 8, 8)
    assert [x for _axis, _row, x in links] == [6, 7, 0]


def test_noc1_is_the_same_formula_in_a_mirrored_space():
    gx, gy = plan.SOC_GRID["blackhole"]
    a, b = (2, 3), (11, 7)
    n1a = plan.to_noc_space(a, 1, gx, gy)
    n1b = plan.to_noc_space(b, 1, gx, gy)
    assert noc_hop_count(n1a, n1b, gx, gy) == noc_hop_count(b, a, gx, gy)


def test_round_trip_takes_only_three_values(grid):
    """The structural claim experiment 1 exists to test, asserted here so that
    a plan which could not express it fails loudly."""
    gx, gy = grid.grid_x, grid.grid_y
    seen = set()
    for m in sorted(grid.phys.values())[:20]:
        for s in sorted(grid.phys.values())[:20]:
            if m == s:
                continue
            seen.add(plan.Flow(m, s).rt_hops(gx, gy))
    assert seen <= {gx, gy, gx + gy}


# --- the grid dump ---------------------------------------------------------


def test_worker_layout_matches_the_soc_descriptors():
    """These tables are transcribed; this is what stops them drifting."""
    root = pathlib.Path(__file__).resolve().parents[2]
    for arch in ("wormhole", "blackhole"):
        path = root / "driver" / arch / "soc_descriptor.yaml"
        if not path.exists():  # pragma: no cover - a checkout without drivers
            pytest.skip(f"no {path}")
        soc = yaml.safe_load(path.read_text())
        workers = [
            tuple(int(v) for v in w.split("-")) for w in soc["functional_workers"]
        ]
        assert tuple(sorted({w[0] for w in workers})) == plan.WORKER_COLUMNS[arch]
        assert tuple(sorted({w[1] for w in workers})) == plan.WORKER_ROWS[arch]
        assert (soc["grid"]["x_size"], soc["grid"]["y_size"]) == plan.SOC_GRID[arch]


def test_translated_coordinates_are_remapped_to_physical(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text(_grid_text("blackhole", translated=True))
    g = plan.load_grid(path)
    assert g.coord_space == "translated"
    assert g.noc[(0, 0)] == (18, 18)
    assert g.phys[(0, 0)] == (
        plan.WORKER_COLUMNS["blackhole"][0],
        plan.WORKER_ROWS["blackhole"][0],
    )
    # The plan must carry the ADDRESSABLE coordinate to the executor and the
    # PHYSICAL one to the link arithmetic; conflating them is the bug this
    # whole two-column scheme exists to prevent.
    assert g.phys != g.noc


def test_a_harvested_translated_dump_is_refused(tmp_path):
    text = _grid_text("blackhole", translated=True).splitlines()
    kept = [ln for ln in text if not ln.startswith("13,")]  # drop a column
    path = tmp_path / "g.csv"
    path.write_text("\n".join(kept) + "\n")
    with pytest.raises(plan.PlanError, match="Harvesting renumbers"):
        plan.load_grid(path)


def test_an_unknown_arch_is_refused(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text("# nocbench-grid arch=quasar\nlog_x,log_y,noc_x,noc_y\n0,0,1,1\n")
    with pytest.raises(plan.PlanError, match="expected one of"):
        plan.load_grid(path)


# --- the invariant checker -------------------------------------------------


def test_check_invariants_rejects_a_second_moving_quantity(grid):
    m = sorted(grid.phys.values())[0]
    a, b = sorted(grid.phys.values())[1], sorted(grid.phys.values())[2]
    points = [
        plan.Point("x", "p0", [plan.Flow(m, a, tx_bytes=64)]),
        plan.Point("x", "p1", [plan.Flow(m, b, tx_bytes=128)]),
    ]
    with pytest.raises(plan.PlanError, match="also moved"):
        plan.check_invariants(points, grid, {"subs", "fwd_hops", "rt_hops"})


def test_check_invariants_rejects_an_axis_that_does_not_move(grid):
    m = sorted(grid.phys.values())[0]
    a = sorted(grid.phys.values())[1]
    points = [plan.Point("x", f"p{i}", [plan.Flow(m, a)]) for i in range(2)]
    with pytest.raises(plan.PlanError, match="no axis to"):
        plan.check_invariants(points, grid, {"tx_bytes"})


def test_a_second_flow_with_the_same_shape_does_not_read_as_a_moved_size(grid):
    """Per-flow scalars compare as sets, so N=1 -> N=2 moves n_flows only."""
    m = sorted(grid.phys.values())[0]
    a, b = sorted(grid.phys.values())[1], sorted(grid.phys.values())[2]
    sigs = [
        plan._fixed_signature(plan.Point("x", "1", [plan.Flow(m, a)]), grid),
        plan._fixed_signature(
            plan.Point("x", "2", [plan.Flow(m, a), plan.Flow(m, b)]), grid
        ),
    ]
    assert sigs[0]["tx_bytes"] == sigs[1]["tx_bytes"]
    assert sigs[0]["num_tx"] == sigs[1]["num_tx"]


# --- the experiments -------------------------------------------------------


@pytest.mark.parametrize("name", sorted(plan.EXPERIMENTS))
def test_every_experiment_plans_on_both_architectures(grid, name):
    points = plan.EXPERIMENTS[name](grid)
    assert len(points) >= 2
    rows = plan.plan_rows(points, grid)
    assert rows
    for row in rows:
        assert set(row) == set(plan.PLAN_COLUMNS)
        for value in row.values():
            assert "," not in str(value), "a CSV cell may not contain a comma"


def test_hops_reaches_all_three_round_trip_levels(grid):
    points = plan.plan_hops(grid)
    gx, gy = grid.grid_x, grid.grid_y
    assert {p.flows[0].rt_hops(gx, gy) for p in points} == {gx, gy, gx + gy}
    assert len({p.flows[0].master for p in points}) == 1


def test_shared_varies_only_the_payload_overlap(grid):
    """The decisive experiment: everything but one link count is frozen."""
    points = plan.plan_shared(grid)
    gx, gy = grid.grid_x, grid.grid_y
    by_size = {}
    for p in points:
        by_size.setdefault(p.flows[0].tx_bytes, []).append(p)
    for size, group in by_size.items():
        a = {(p.flows[0].master, p.flows[0].sub) for p in group}
        assert len(a) == 1, f"flow A moved at {size} B"
        assert len({p.flows[1].fwd_hops(gx, gy) for p in group}) == 1
        assert len({p.flows[1].rt_hops(gx, gy) for p in group}) == 1
        assert all(len(p.flows) == 2 for p in group)
        # Every leg-pair overlap except payload-with-payload is zero, at every
        # point. This is the claim the geometry was searched to satisfy.
        assert {plan.other_overlap(p.flows[0], p.flows[1], gx, gy) for p in group} == {
            0
        }
        overlaps = sorted(
            plan.payload_overlap(p.flows[0], p.flows[1], gx, gy) for p in group
        )
        assert overlaps[0] == 0, "the zero-overlap baseline is missing"
        assert len(set(overlaps)) == len(overlaps), "two points share an overlap value"
        assert len(overlaps) >= 3


def test_shared_points_use_four_distinct_cores(grid):
    for p in plan.plan_shared(grid):
        cores = {p.flows[0].master, p.flows[0].sub, p.flows[1].master, p.flows[1].sub}
        assert len(cores) == 4


def test_contention_holds_every_masters_distance_equal(grid):
    points = plan.plan_contention(grid)
    gx, gy = grid.grid_x, grid.grid_y
    subs = {f.sub for p in points for f in p.flows}
    assert len(subs) == 1
    assert len({f.fwd_hops(gx, gy) for p in points for f in p.flows}) == 1
    assert len({f.rt_hops(gx, gy) for p in points for f in p.flows}) == 1
    assert [len(p.flows) for p in points] == sorted(len(p.flows) for p in points)


def test_contention_shared_links_grow_with_n_and_that_is_declared(grid):
    """The honest limit: N flows into one endpoint MUST share links near it."""
    points = plan.plan_contention(grid)
    gx, gy = grid.grid_x, grid.grid_y
    shared = [plan._shared_payload_total(p.flows, gx, gy) for p in points]
    assert shared == sorted(shared)
    assert shared[0] == 0
    assert shared[-1] > 0


def test_selfport_puts_both_flows_on_one_core_at_equal_distance(grid):
    points = plan.plan_selfport(grid)
    gx, gy = grid.grid_x, grid.grid_y
    two = points[-1].flows
    assert len(two) == 2
    assert two[0].master == two[1].master
    assert two[0].proc != two[1].proc
    assert two[0].fwd_hops(gx, gy) == two[1].fwd_hops(gx, gy)
    assert two[0].rt_hops(gx, gy) == two[1].rt_hops(gx, gy) == gx + gy


def test_vc_moves_nothing_but_the_virtual_channel(grid):
    points = plan.plan_vc(grid)
    assert [p.flows[0].vc for p in points] == [0, 1, 2, 3]
    assert (
        len({(p.flows[0].master, p.flows[0].sub, p.flows[0].tx_bytes) for p in points})
        == 1
    )
    assert all(p.flows[0].direction == plan.DIR_BIDIR for p in points)


def test_a_read_puts_its_payload_on_the_return_leg():
    gx, gy = plan.SOC_GRID["blackhole"]
    w = plan.Flow((1, 2), (3, 4), plan.DIR_WRITE)
    r = plan.Flow((1, 2), (3, 4), plan.DIR_READ)
    assert w.payload_links(gx, gy) == set(plan.route_links((1, 2), (3, 4), gx, gy))
    assert r.payload_links(gx, gy) == set(plan.route_links((3, 4), (1, 2), gx, gy))
    assert w.payload_links(gx, gy) != r.payload_links(gx, gy)


# --- emitting ---------------------------------------------------------------


def test_tensix_coords_lists_every_core_the_plan_touches(grid):
    rows = plan.plan_rows(plan.plan_shared(grid), grid)
    coords = plan.tensix_coords(rows)
    for row in rows:
        assert f"{row['mst_nx']}-{row['mst_ny']}" in coords.split(",")
        assert f"{row['sub_nx']}-{row['sub_ny']}" in coords.split(",")


def test_written_plan_round_trips(grid, tmp_path):
    rows = plan.plan_rows(plan.plan_hops(grid), grid)
    out = tmp_path / "plan.csv"
    plan.write_plan(out, rows, grid, ["note"])
    text = out.read_text().splitlines()
    header = [ln for ln in text if not ln.startswith("#")][0]
    assert header.split(",") == list(plan.PLAN_COLUMNS)
    assert any("tt_sim_tensix_coords=" in ln for ln in text)
    assert len([ln for ln in text if not ln.startswith("#")]) == len(rows) + 1


def test_hypotheses_cover_every_experiment_and_name_the_null():
    assert set(plan.HYPOTHESES) == set(plan.EXPERIMENTS)
    for name, text in plan.HYPOTHESES.items():
        assert "Held fixed" in text, name
        assert "Varying" in text or "swept" in text, name
    assert "NO CONGESTION EFFECT" in plan.HYPOTHESES["shared"]
    assert set(plan.MINIMUM) <= set(plan.EXPERIMENTS)


def test_main_writes_a_plan(grid, tmp_path, capsys):
    src = tmp_path / "grid.csv"
    src.write_text(_grid_text(grid.arch))
    out = tmp_path / "plan.csv"
    rc = plan.main(
        ["--grid", str(src), "--out", str(out), "--experiments", "hops,size"]
    )
    assert rc == 0
    assert out.exists()
    assert "TT_SIM_TENSIX_COORDS=" in capsys.readouterr().out


def test_main_rejects_an_unknown_experiment(grid, tmp_path):
    src = tmp_path / "grid.csv"
    src.write_text(_grid_text(grid.arch))
    assert (
        plan.main(
            [
                "--grid",
                str(src),
                "--out",
                str(tmp_path / "p.csv"),
                "--experiments",
                "nope",
            ]
        )
        == 2
    )
