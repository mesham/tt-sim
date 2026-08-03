"""The rung-2 sweep harness: what it excludes, and what it predicts.

Two halves, and the split is the point.

1. **Everything that does not need tt-metal.** The exclusion ladder, the
   dataset schema constants, and -- the substantive one -- that
   :func:`~tt_sim.perf.noc_dataset_sweep.predict_cycles`, which drives a *real*
   device rather than evaluating a formula, agrees with the closed-form
   composition of the published terms. That is the guard that keeps the sweep
   honest: a harness that silently stopped exercising the DRAM endpoint, or
   quietly picked the wrong geometry, would still print a tidy table.

2. **The sweep itself**, which needs the dataset and is skipped without it --
   the same contract ``examples/examples_test.py`` uses for its tt-metal
   dependency. Set ``TT_SIM_NOC_LATENCIES`` or ``TT_METAL_RUNTIME_ROOT`` to run
   it.

Run:  python3 -m tt_sim.perf.noc_dataset_sweep_test   (or under pytest)
"""

import math
import os
from contextlib import contextmanager

import pytest

from tt_sim.arch import BLACKHOLE_PROFILE, WORMHOLE_PROFILE
from tt_sim.network.tt_noc import noc_hop_count
from tt_sim.perf import noc_dataset_sweep as sweep
from tt_sim.perf.model import dram_cost_model, noc_cost_model


@contextmanager
def _cost_model_on():
    previous = os.environ.get("TT_SIM_COST_MODEL")
    os.environ["TT_SIM_COST_MODEL"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


_PROFILES = {"wormhole": WORMHOLE_PROFILE, "blackhole": BLACKHOLE_PROFILE}


def _dataset_path():
    path = sweep.default_dataset_path()
    return path if path and path.exists() else None


_needs_dataset = pytest.mark.skipif(
    _dataset_path() is None,
    reason="noc_latencies.yaml not found; set TT_SIM_NOC_LATENCIES or "
    "TT_METAL_RUNTIME_ROOT",
)


# ---------------------------------------------------------------------------
# 1. The exclusion ladder. Declared before any residual existed, and the
#    property that matters is that each rule names a term tt-sim does not
#    model -- not that the retained set is large.
# ---------------------------------------------------------------------------


def test_every_exclusion_rule_carries_a_modelling_reason():
    """Dropping entries because they *disagree* is fitting, not validating.
    The defence is that every rule is justified by something absent from the
    model, in prose, in the file, next to the predicate."""
    for name, reason, predicate in sweep._exclusions(sweep.ARCH_IDS["wormhole"]):
        assert name
        assert callable(predicate)
        assert len(reason) > 20, name


def test_the_ladder_is_ordered_and_reports_its_own_cost():
    entries = [
        ({**sweep.KEY_DEFAULTS, "arch": 2}, [1.0]),
        ({**sweep.KEY_DEFAULTS, "arch": 3}, [1.0]),
        ({**sweep.KEY_DEFAULTS, "arch": 2, "mechanism": 1}, [1.0]),
        ({**sweep.KEY_DEFAULTS, "arch": 2, "stateful": True}, [1.0]),
    ]
    kept, ladder = sweep.retained(entries, sweep.ARCH_IDS["wormhole"])
    assert len(kept) == 1
    assert [name for name, _, _ in ladder][:2] == ["arch", "mechanism != UNICAST"]
    assert ladder[0][1] == 1  # one Blackhole entry removed
    assert sum(removed for _, removed, _ in ladder) == len(entries) - len(kept)


def test_pattern_zero_is_a_read_because_the_csv_carries_the_name():
    """The one thing about this dataset that is genuinely a trap. The test
    that produced it numbers ``ONE_TO_ONE = 0``; the library that consumes it
    numbers ``ONE_FROM_ONE = 0``. They are reconciled by the CSV carrying the
    pattern's *name*, which ``csv_reader.cpp`` maps to the library's value --
    so the YAML follows the library, and 0 is a read."""
    assert sweep.PATTERN_READ == 0
    assert sweep.PATTERN_WRITE == 1
    assert sweep.KEY_DEFAULTS["pattern"] == sweep.PATTERN_WRITE


def test_the_key_defaults_are_the_loaders_defaults():
    """The YAML writes only non-default fields, so an entry cannot be read at
    all without these. They are ``types.hpp``'s ``DEFAULT_*`` constants."""
    assert sweep.KEY_DEFAULTS == {
        "mechanism": 0,
        "pattern": 1,
        "memory": 0,
        "arch": 2,
        "num_transactions": 1,
        "num_subordinates": 1,
        "same_axis": False,
        "stateful": False,
        "loopback": False,
        "noc_index": 0,
    }


def test_only_dram_rows_above_the_sweep_cap_are_treated_as_unmeasured():
    """``dram_accessor_sweep`` issues at most 256 pages of 32 bytes, and the
    offline extractor pads every standard size above the largest measured one
    by repeating the last value. A fact about how the file was generated, not
    about whether the model agrees with it."""
    l1 = {**sweep.KEY_DEFAULTS, "memory": sweep.MEMORY_L1}
    dram = {**sweep.KEY_DEFAULTS, "memory": sweep.MEMORY_DRAM_SHARDED}
    for size in (64, 8192, 65536):
        assert sweep.point_is_measured(l1, size)
    assert sweep.point_is_measured(dram, 8192)
    assert not sweep.point_is_measured(dram, 16384)


# ---------------------------------------------------------------------------
# 2. The predictor. It drives a real device; this pins what that device does
#    to the published constants, so the sweep cannot quietly stop exercising a
#    term and still print a tidy table.
# ---------------------------------------------------------------------------


def _closed_form(arch, memory, same_axis, size):
    """What the assembled model *should* cost, composed from the tables.

    ``flight(there) + flight(back) + dram service + serialisation``, where the
    serialisation is the injection time of every burst chunk but the last plus
    the last one's tail -- the packet's own size paid once, because the NoC is
    wormhole-routed. The trailing ``+ 1`` is the polling loop's off-by-one (it
    pumps before it looks), not a cost.
    """
    profile = _PROFILES[arch]
    grid = (profile.noc_grid_x, profile.noc_grid_y)
    geometry = sweep.GEOMETRY[arch]
    with _cost_model_on():
        device = sweep._build_device(arch, [geometry["master"]])
        noc = noc_cost_model(arch)
        dram = dram_cost_model(arch)
    local = device.tensix_tiles[0].noc0_router.id_pair
    if memory == sweep.MEMORY_L1:
        remote = sweep._physical_of(
            arch, geometry["same_axis" if same_axis else "diff_axis"]
        )
        service = 0
    else:
        remote = device.dram_tiles[0].noc0_router.id_pair
        service = 0 if dram is None else dram.service_cycles

    flight = noc.flight_cycles(noc_hop_count(local, remote, *grid)) + noc.flight_cycles(
        noc_hop_count(remote, local, *grid)
    )
    chunks = math.ceil(size / profile.noc_max_burst_size)
    full = profile.noc_max_burst_size if chunks > 1 else size
    last = size - (chunks - 1) * full
    serialisation = (chunks - 1) * noc.serialisation_cycles(full) + noc.tail_cycles(
        last
    )
    return flight + service + serialisation + 1


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
@pytest.mark.parametrize("same_axis", [True, False])
@pytest.mark.parametrize("size", [64, 8192, 65536])
def test_a_real_l1_round_trip_costs_what_the_tables_compose_to(arch, same_axis, size):
    with _cost_model_on():
        measured = sweep.predict_cycles(arch, sweep.MEMORY_L1, True, same_axis, size)
    assert measured == _closed_form(arch, sweep.MEMORY_L1, same_axis, size)


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
@pytest.mark.parametrize("size", [64, 8192])
def test_a_real_dram_round_trip_adds_the_service_window(arch, size):
    """A DRAM row costs the same round trip as a different-axis L1 row (both
    are ``grid_x + grid_y`` hops) plus the endpoint's own service time, and
    nothing else. If an arch sourced no ``access_latency`` the difference would
    be zero, which is what an unsourced arch is *meant* to look like."""
    with _cost_model_on():
        dram = sweep.predict_cycles(arch, sweep.MEMORY_DRAM_SHARDED, True, False, size)
        l1 = sweep.predict_cycles(arch, sweep.MEMORY_L1, True, False, size)
        model = dram_cost_model(arch)
    assert dram - l1 == (0 if model is None else model.service_cycles)


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_same_axis_is_the_shorter_round_trip_and_by_a_whole_ring(arch):
    """The geometry the dataset does not record and the model needs. On a
    directional torus a round trip between two tiles differing on both axes is
    exactly ``grid_x + grid_y`` hops whatever the distance, and one sharing an
    axis is exactly the other dimension -- so ``same_axis`` alone fixes the hop
    count, on either NoC, with no core-placement assumption doing any work."""
    profile = _PROFILES[arch]
    grid = (profile.noc_grid_x, profile.noc_grid_y)
    geometry = sweep.GEOMETRY[arch]
    master = sweep._physical_of(arch, geometry["master"])
    for name, expected in (
        ("same_axis", profile.noc_grid_y),
        ("diff_axis", profile.noc_grid_x + profile.noc_grid_y),
    ):
        sub = sweep._physical_of(arch, geometry[name])
        round_trip = noc_hop_count(master, sub, *grid) + noc_hop_count(
            sub, master, *grid
        )
        assert round_trip == expected, name


# ---------------------------------------------------------------------------
# 3. The sweep. Needs the dataset; skipped without it.
# ---------------------------------------------------------------------------


def test_the_script_says_where_it_looked_when_the_dataset_is_absent(capsys, tmp_path):
    assert sweep.main(["--dataset", str(tmp_path / "nope.yaml")]) == 0
    assert "not found" in capsys.readouterr().out


@_needs_dataset
def test_the_dataset_is_the_shape_this_module_documents():
    entries, sizes = sweep.load_dataset(_dataset_path())
    assert len(entries) == 740
    assert sizes == [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    assert all(len(latencies) == len(sizes) for _, latencies in entries)
    # ``noc_index`` carries no information: the kernels log the column as
    # "NoC Index" and ``csv_reader.cpp`` looks for "NOC index", so every point
    # takes the default. Recorded as a fact about the data, because a reader
    # would otherwise take the field at face value.
    assert {key["noc_index"] for key, _ in entries} == {0}


#: The one place the assembled model is known to over-charge, found by this
#: sweep and pinned here so it cannot quietly spread. A Blackhole DRAM *write*
#: is answered when the endpoint accepts it, not when the array has been
#: written, so the flat ``access_latency`` derived from the *read* figures is
#: too large for it -- by 12 to 28 cycles. The entry's own note in
#: ``unit_costs.yaml`` names the asymmetry; this is what it costs.
KNOWN_OVER_CHARGED = {("blackhole", "DRAM write diff-axis")}


@_needs_dataset
@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_the_model_is_a_floor_everywhere_but_one_named_row(arch):
    """The headline of rung 2, as an assertion rather than a paragraph. The
    measurement includes an issuing-core path the model does not charge for,
    so every residual should be positive; a negative one means the model
    over-charges, inventing back-pressure the hardware does not have, which is
    the direction every bound in these tables is chosen to avoid. Exactly one
    row does, it is named in :data:`KNOWN_OVER_CHARGED`, and a second appearing
    is a change somebody has to make deliberately."""
    entries, sizes = sweep.load_dataset(_dataset_path())
    with _cost_model_on():
        kept, _ = sweep.retained(entries, sweep.ARCH_IDS[arch])
        rows = sweep.sweep(kept, sizes, arch)
    assert rows
    over = {(arch, row.label) for row in rows if row.residual <= 0}
    assert over <= KNOWN_OVER_CHARGED, sorted(over - KNOWN_OVER_CHARGED)
    assert all(
        row.residual > 0 for row in rows if (arch, row.label) not in KNOWN_OVER_CHARGED
    )


@_needs_dataset
@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_the_hop_term_explains_the_geometry_difference_out_of_sample(arch):
    """The rung-2 result the ladder was built to get. Nothing in the hop model
    was fitted to this dataset, and the measured same-axis-vs-different-axis
    difference is a pure hop-count difference -- ``9 * grid_x`` cycles, 90 on
    Wormhole and 153 on Blackhole. If the two geometries' residuals agree, the
    hop term reproduced all of it.

    L1 rows only, because that is where the dataset samples both geometries;
    including the DRAM rows (which are different-axis only) would confound the
    geometry axis with the memory-type one."""
    entries, sizes = sweep.load_dataset(_dataset_path())
    with _cost_model_on():
        kept, _ = sweep.retained(entries, sweep.ARCH_IDS[arch])
        rows = [
            r
            for r in sweep.sweep(kept, sizes, arch)
            if r.size <= 512 and r.key["memory"] == sweep.MEMORY_L1
        ]
    same = [r.residual for r in rows if r.key["same_axis"]]
    diff = [r.residual for r in rows if not r.key["same_axis"]]
    assert same
    assert diff
    # Ten cycles against a 90-to-153-cycle effect the model had to get right.
    assert abs(sum(same) / len(same) - sum(diff) / len(diff)) < 10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
