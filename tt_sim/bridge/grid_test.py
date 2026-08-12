"""``TT_SIM_TENSIX_CORES=N`` must name the workers tt-metal launches on.

These are pure table tests — no device, no socket, no tt-metal — so they cost
milliseconds and can live in the ordinary suite. What they pin is the one thing
a count-based knob can get wrong silently: materialising N *plausible* workers
that are not the N a program will actually use, which shows up much later as a
kernel launch on an un-materialised tile (best case) or zeros (worst).

The expected coords below are derived from tt-metal's
``num_cores_to_corerangeset`` (``tt_metal/common/work_split.cpp``): fill the
compute grid column-major, y fastest.
"""

import pytest

from tt_sim.bridge.grid import compute_grid, fill_order, logical_to_physical

# The Wormhole worker grid from driver/wormhole/soc_descriptor.yaml: 8 columns
# (physical x skips 0 and 5, the DRAM/router columns) by 10 rows (physical y
# skips 0 and 6, the ethernet rows).
WH_XS = (1, 2, 3, 4, 6, 7, 8, 9)
WH_YS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
WH_WORKERS = [(x, y) for x in WH_XS for y in WH_YS]
WH_DEFAULT_GRID = (8, 9)


def test_logical_axes_are_index_lookups_not_offsets():
    xs, ys = logical_to_physical(WH_WORKERS)
    assert xs == [1, 2, 3, 4, 6, 7, 8, 9]
    assert ys == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    # The gap is the whole point: logical column 4 is physical 6, not 5.
    assert xs[4] == 6
    assert ys[5] == 7


def test_compute_grid_defaults_to_the_arch_constant():
    assert compute_grid(WH_DEFAULT_GRID, env={}) == (8, 9)


def test_metal_override_names_an_inclusive_maximum():
    # TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4 selects a 4x5 grid.
    env = {"TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": "3,4"}
    assert compute_grid(WH_DEFAULT_GRID, env=env) == (4, 5)
    assert compute_grid(
        WH_DEFAULT_GRID, env={"TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": "[1, 1]"}
    ) == (2, 2)


def test_tt_sim_compute_grid_wins_over_the_metal_override():
    env = {
        "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": "3,4",
        "TT_SIM_COMPUTE_GRID": "2x3",
    }
    assert compute_grid(WH_DEFAULT_GRID, env=env) == (2, 3)


def test_blank_env_values_are_ignored():
    env = {"TT_SIM_COMPUTE_GRID": "  ", "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": ""}
    assert compute_grid(WH_DEFAULT_GRID, env=env) == (8, 9)


def test_fill_order_is_column_major_over_the_default_grid():
    order = fill_order(WH_WORKERS, WH_DEFAULT_GRID, env={})
    # A column is 9 tall (the compute grid), not 10 (the SoC grid) and not 5.
    assert order[:9] == [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
    ]
    assert order[9] == (2, 1)
    # Physical row 11 is outside the compute grid, so it is never in the first
    # 72 entries however many cores are asked for.
    assert not any(y == 11 for _, y in order[:72])


def test_the_sixth_core_moves_with_the_grid_override():
    """The regression this module exists for.

    With no override the 6th core tt-metal uses is physical ``1-7`` (logical
    ``(0,5)``, because the column is 9 tall); under the 4x5 override it is
    ``2-1``. A fixed column height is wrong in one regime or the other.
    """
    assert fill_order(WH_WORKERS, WH_DEFAULT_GRID, env={})[5] == (1, 7)
    override = {"TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": "3,4"}
    assert fill_order(WH_WORKERS, WH_DEFAULT_GRID, env=override)[5] == (2, 1)


def test_workers_outside_the_compute_grid_come_last_and_are_not_lost():
    order = fill_order(WH_WORKERS, WH_DEFAULT_GRID, env={})
    assert len(order) == len(WH_WORKERS)
    assert len(set(order)) == len(WH_WORKERS)
    assert set(order[72:]) == {(x, 11) for x in WH_XS}


def test_a_tiny_override_still_orders_the_whole_grid():
    env = {"TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": "0,0"}
    order = fill_order(WH_WORKERS, WH_DEFAULT_GRID, env=env)
    assert order[0] == (1, 1)
    assert len(order) == len(WH_WORKERS)


def test_bad_grid_values_raise_rather_than_silently_defaulting():
    with pytest.raises(ValueError, match="separated by 'x'"):
        compute_grid(WH_DEFAULT_GRID, env={"TT_SIM_COMPUTE_GRID": "8"})
    with pytest.raises(ValueError, match="at least 1x1"):
        compute_grid(WH_DEFAULT_GRID, env={"TT_SIM_COMPUTE_GRID": "0x9"})
    with pytest.raises(ValueError, match="separated by ','"):
        compute_grid(
            WH_DEFAULT_GRID, env={"TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": "3"}
        )


def test_wormhole_driver_agrees_with_the_table():
    from driver.wormhole.server.coords import (
        DEFAULT_COMPUTE_GRID,
        TENSIX_COORD_MAP,
        default_tensix_coords,
    )

    assert DEFAULT_COMPUTE_GRID == WH_DEFAULT_GRID
    assert sorted(TENSIX_COORD_MAP) == sorted(WH_WORKERS)
    assert default_tensix_coords(1, env={}) == [(1, 1)]
    assert default_tensix_coords(2, env={}) == [(1, 1), (1, 2)]
    assert default_tensix_coords(6, env={})[-1] == (1, 7)
    override = {"TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE": "3,4"}
    assert default_tensix_coords(20, env=override) == [
        (x, y) for x in (1, 2, 3, 4) for y in (1, 2, 3, 4, 5)
    ]
    with pytest.raises(ValueError, match="must be >= 1"):
        default_tensix_coords(0, env={})
    with pytest.raises(ValueError, match="exceeds the 80 functional workers"):
        default_tensix_coords(len(WH_WORKERS) + 1, env={})


def test_blackhole_driver_skips_the_missing_worker_columns():
    from driver.blackhole.server.coords import (
        DEFAULT_COMPUTE_GRID,
        default_tensix_coords,
    )

    assert DEFAULT_COMPUTE_GRID == (13, 10)
    coords = default_tensix_coords(130, env={})
    assert len(coords) == 130
    assert coords[0] == (1, 2)
    # Blackhole worker columns skip physical 8 and 9, so logical column 7 (the
    # eighth) is physical 10 — the slip that has already cost a wrong run.
    assert coords[70] == (10, 2)
    assert not any(x in (8, 9) for x, _ in coords)
