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

    A DRAM row picks up one more term, the channel's excess over the link
    rate for the same bytes. It is written for the single-chunk case only,
    which is every DRAM row there is: the sweep drops DRAM sizes above 8 KiB
    as unmeasured fill, and 8 KiB is one burst chunk on both arches.
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
    if service and chunks == 1:
        service += dram.channel_excess_cycles(size, noc.serialisation_cycles(size))
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
    are ``grid_x + grid_y`` hops) plus the endpoint's own two terms, and
    nothing else: the flat service time, and the channel's *excess* over the
    NoC link for these bytes. If an arch sourced neither the difference would
    be zero, which is what an unsourced arch is *meant* to look like. Both
    shipped arches now source both terms for a READ, which is the direction
    this test predicts — by two different routes (a published GB/s converted
    on Wormhole, a secant on the measured dataset on Blackhole), and the
    closed form has to compose the same either way."""
    with _cost_model_on():
        dram = sweep.predict_cycles(arch, sweep.MEMORY_DRAM_SHARDED, True, False, size)
        l1 = sweep.predict_cycles(arch, sweep.MEMORY_L1, True, False, size)
        model = dram_cost_model(arch)
    if model is None:
        assert dram - l1 == 0
        return
    with _cost_model_on():
        link = noc_cost_model(arch).serialisation_cycles(size)
    assert dram - l1 == model.service_cycles + model.channel_excess_cycles(size, link)


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


#: Rows the assembled model is known to over-charge, found by this sweep and
#: pinned here so a new one cannot appear quietly.
#:
#: **EMPTY since 2026-08-17, and the empty set is the result.** The single
#: entry was ``("blackhole", "DRAM write diff-axis")``: a Blackhole DRAM write
#: is answered when the endpoint accepts it, not when the array has been
#: written, so the flat ``dram.access_latency`` derived from the *read* rows
#: was too large for it -- by 37 to 52 cycles on this sweep's eight measured
#: sizes, net of the issuing-core path the harness now runs. (The figure this
#: comment used to carry, "12 to 28", predates that harness change and was not
#: re-measured with it; the residuals were -52 / -44 / -52 / -49 / -52 / -42 /
#: -42 / -37 over 64 B..8 KiB immediately before the split landed.)
#: ``unit_costs.yaml`` now derives that direction's own figure from the same
#: vendor campaign (22 cycles against 126), so every prediction on that row
#: falls by exactly 104 and its residual becomes +52 / +60 / +52 / +55 / +52 /
#: +62 / +62 / +67. No other row on either arch moves by a cycle. The
#: assertion below is deliberately left in place so a future over-charge is
#: still somebody's decision rather than a drift.
KNOWN_OVER_CHARGED = set()


@_needs_dataset
@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_the_model_is_a_floor_everywhere_but_one_named_row(arch):
    """The headline of rung 2, as an assertion rather than a paragraph. The
    measurement includes an issuing-core path the model does not charge for,
    so every residual should be positive; a negative one means the model
    over-charges, inventing back-pressure the hardware does not have, which is
    the direction every bound in these tables is chosen to avoid. No row does
    today -- :data:`KNOWN_OVER_CHARGED` is empty -- and one appearing is a
    change somebody has to make deliberately."""
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


# ---------------------------------------------------------------------------
# 3. What the sweep says about the rows it drops. The ladder reports a count
#    per rule; these pin the claims the report makes ON TOP of that count,
#    because "removes 150" read as "150 rows waiting on one missing term" is
#    exactly the misreading the extra section exists to prevent.
# ---------------------------------------------------------------------------


def test_every_rule_says_what_closing_it_would_take():
    """A rule that names no missing term is a rule nobody can act on. Pinned
    against the ladder itself, so adding a rule without answering the question
    fails here rather than printing a ``?`` in the report."""
    for arch_id in sweep.ARCH_IDS.values():
        names = {name for name, _reason, _keep in sweep._exclusions(arch_id)}
        assert names == set(sweep.MISSING_TERM)
    assert all(len(v) > 20 for v in sweep.MISSING_TERM.values())


def test_sole_cause_counts_are_a_subset_of_the_ladder_counts():
    """The two ways of counting an exclusion, and the relationship between
    them. ``sole_cause`` can never exceed what the ladder removes -- an entry
    only one rule excludes is an entry that rule removes wherever it sits in
    the order -- and the point of reporting it is that it is usually far
    smaller."""
    entries = [
        ({**sweep.KEY_DEFAULTS, "arch": 2}, [1.0]),  # retained
        ({**sweep.KEY_DEFAULTS, "arch": 2, "pattern": 4}, [1.0]),  # pattern only
        # pattern AND num_transactions: no single term unlocks it
        ({**sweep.KEY_DEFAULTS, "arch": 2, "pattern": 4, "num_transactions": 8}, [1.0]),
        ({**sweep.KEY_DEFAULTS, "arch": 3, "pattern": 4}, [1.0]),  # other arch
    ]
    by_count, sole = sweep.exclusion_multiplicity(entries, 2)
    assert by_count == {0: 1, 1: 1, 2: 1}
    assert sole["pattern not in {ONE_FROM_ONE, ONE_TO_ONE}"] == 1
    assert sole["num_transactions per barrier != 1, outside the L1 write path"] == 0
    _kept, ladder = sweep.retained(entries, 2)
    removed = dict((name, count) for name, count, _left in ladder)
    for name, count in sole.items():
        assert count <= removed[name]


def test_the_concurrency_series_is_one_transaction_per_pair():
    """The series the report calls "the only shape where the flow count is the
    only thing that changes". That is only true of the rows whose transactions
    per barrier equal their subordinate count -- one per (master, subordinate)
    pair -- so a row with a longer burst must not be in it."""
    sizes = [64]
    entries = [
        (
            {
                **sweep.KEY_DEFAULTS,
                "arch": 2,
                "pattern": 4,
                "num_subordinates": 4,
                "num_transactions": 4,
            },
            [100.0],
        ),
        # same grid, four transactions each: a burst, not more concurrency
        (
            {
                **sweep.KEY_DEFAULTS,
                "arch": 2,
                "pattern": 4,
                "num_subordinates": 4,
                "num_transactions": 16,
            },
            [400.0],
        ),
    ]
    series = sweep.concurrency_series(entries, sizes, 2, 64)
    assert [row[0] for row in series] == [4]
    cores, cycles, aggregate, per_core = series[0]
    assert cycles == 100.0
    # 4 masters x 4 subordinates x 64 B in 100 cycles.
    assert aggregate == pytest.approx(4 * 4 * 64 / 100)
    assert per_core == pytest.approx(aggregate / cores)


@_needs_dataset
@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_closing_congestion_alone_would_unlock_almost_nothing(arch):
    """The finding that makes the whole "report what you drop" section worth
    printing, as an assertion. The pattern rule removes ~150 entries, but
    almost every one of them ALSO carries a multi-transaction burst or a
    multicast, so a perfect congestion model on its own would move the retained
    set by a couple of entries. If that ever stops being true -- because the
    dataset changed, or another rule was retired -- the report's headline claim
    needs rewriting, and this is what says so."""
    entries, _sizes = sweep.load_dataset(_dataset_path())
    arch_id = sweep.ARCH_IDS[arch]
    _by_count, sole = sweep.exclusion_multiplicity(entries, arch_id)
    _kept, ladder = sweep.retained(entries, arch_id)
    removed = dict((name, count) for name, count, _left in ladder)
    rule = "pattern not in {ONE_FROM_ONE, ONE_TO_ONE}"
    assert removed[rule] >= 100
    assert sole[rule] <= 5


@_needs_dataset
@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_the_dataset_bounds_congestion_even_though_it_cannot_model_it(arch):
    """The useful half of the null result: per-core bandwidth falls as the
    grid grows, monotonically, on both architectures. That curve is what any
    future congestion term has to reproduce, and it is the only quantitative
    statement about congestion this dataset supports."""
    entries, sizes = sweep.load_dataset(_dataset_path())
    series = sweep.concurrency_series(entries, sizes, sweep.ARCH_IDS[arch], 65536)
    assert len(series) >= 4
    per_core = [row[3] for row in series]
    assert per_core == sorted(per_core, reverse=True)
    # More cores always buys SOME aggregate bandwidth, and never proportionally.
    aggregate = [row[2] for row in series]
    assert aggregate == sorted(aggregate)
    assert aggregate[-1] / aggregate[0] < series[-1][0] / series[0][0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
