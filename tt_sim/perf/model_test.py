"""Tests for the occupancy model that consumes the cycle-cost tables.

Runs standalone (``python3 -m tt_sim.perf.model_test``) or under pytest.

Four things are pinned, in decreasing order of how much they matter:

1. **Off by default, and off means off.** ``TT_SIM_COST_MODEL`` unset gives
   every unit a ``None`` cost model, so no table is loaded and no cycle count
   moves. The replay guards and ``six``'s matmul PCC are how this project
   catches timing regressions; a cost model that shifted them silently would
   have broken the instrument before proving anything.
2. **Bounds survive.** ``>= 15`` and "3 or 4" are charged at their low end and
   report themselves as inexact, rather than being flattened into an integer
   that claims a precision nobody published.
3. **A gap stays a gap.** An opcode with no published cost is charged nothing
   at all, not a plausible-looking 1.
4. **The fidelity-phase cost is computed from the table**, not hardcoded — a
   test doubles the table's numbers and watches the answer follow.
"""

import os
from contextlib import contextmanager

from tt_sim.perf.costs import CycleCost, load_costs
from tt_sim.perf.model import (
    BOUND_POLICY,
    UnitCostModel,
    cost_model_enabled,
    modelled_occupancy,
    unit_cost_model,
)


@contextmanager
def _env(value):
    """Set (or clear) TT_SIM_COST_MODEL for the duration of the block."""
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if value is None:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    else:
        os.environ["TT_SIM_COST_MODEL"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def _math_model(arch="wormhole"):
    return UnitCostModel(load_costs(arch).unit("MATH"), arch)


# ---------------------------------------------------------------------------
# 1. Opt-in.
# ---------------------------------------------------------------------------


def test_the_model_is_off_unless_the_env_var_says_otherwise():
    with _env(None):
        assert not cost_model_enabled()
        assert unit_cost_model("MATH", "wormhole") is None


def test_env_var_truthiness_matches_the_repo_convention():
    for truthy in ("1", "true", "TRUE", "Yes", "on"):
        assert cost_model_enabled({"TT_SIM_COST_MODEL": truthy}), truthy
    for falsy in ("0", "false", "off", "no", "", "  "):
        assert not cost_model_enabled({"TT_SIM_COST_MODEL": falsy}), falsy
    assert not cost_model_enabled({})


def test_enabling_it_yields_a_cached_model_per_unit_and_arch():
    with _env("1"):
        wh = unit_cost_model("MATH", "wormhole")
        assert wh is not None
        assert unit_cost_model("MATH", "wormhole") is wh
        assert unit_cost_model("MATH", "blackhole") is not wh


# ---------------------------------------------------------------------------
# 2. Bounds are not equals signs.
# ---------------------------------------------------------------------------


def test_every_bound_in_the_vocabulary_has_a_policy():
    from tt_sim.perf.costs import BOUNDS

    assert set(BOUND_POLICY) == set(BOUNDS)


def test_bounded_costs_are_charged_at_their_low_end():
    assert modelled_occupancy(None) is None
    assert modelled_occupancy(CycleCost(1)) == 1
    assert modelled_occupancy(CycleCost(15, "at_least")) == 15
    assert modelled_occupancy(CycleCost(3, "range", 4)) == 3
    assert modelled_occupancy(CycleCost(5, "approximate")) == 5
    assert modelled_occupancy(CycleCost(2, "at_most")) == 2
    # A unit cannot be busy for two thirds of a cycle.
    assert modelled_occupancy(CycleCost(1.5)) == 2


def test_a_charged_number_knows_whether_it_was_exact():
    thcon = UnitCostModel(load_costs("wormhole").unit("THCON"), "wormhole")
    assert thcon.occupancy("ATCAS") == 15  # the doc writes ">= 15"
    assert not thcon.is_exact("ATCAS")
    assert thcon.occupancy("ADDDMAREG") == 3  # the doc writes "3 or 4"
    assert not thcon.is_exact("ADDDMAREG")
    assert thcon.occupancy("DMANOP") == 1
    assert thcon.is_exact("DMANOP")


# ---------------------------------------------------------------------------
# 3. A gap stays a gap.
# ---------------------------------------------------------------------------


def test_untabulated_and_unknown_opcodes_are_charged_nothing():
    math = _math_model()
    # In the table, but explicitly ``provenance: unknown`` with no numbers.
    assert load_costs("wormhole").unit("MATH")["MOVD2B"].provenance == "unknown"
    assert math.occupancy("MOVD2B") is None
    # Not in the cost table at all (listed under ``not_documented``).
    assert math.occupancy("SETASHRMH") is None
    # Not an instruction at all.
    assert math.occupancy("NOT_AN_OPCODE") is None


# ---------------------------------------------------------------------------
# 4. Fidelity phases -- the one cost this unit has to compute.
# ---------------------------------------------------------------------------


def test_the_fidelity_scaled_ops_are_the_ones_the_table_marks():
    math = _math_model()
    for op in ("MVMUL", "DOTPV", "GAPOOL", "ELWMUL"):
        assert math.scales_with_fidelity(op), op
    for op in ("GMPOOL", "ELWADD", "ELWSUB", "SETRWC"):
        assert not math.scales_with_fidelity(op), op


def test_fidelity_occupancy_is_one_cycle_at_every_phase():
    """The substantive result of wiring this unit up, and not a degenerate one.

    tt-metal's whole-tile figures (LoFi 16, HiFi2 32, HiFi3 48, HiFi4 64
    cycles) do scale with the phase count, but so does the *instruction
    stream*: a HiFi4 matmul issues four MVMULs where a LoFi one issues a
    single instruction. Dividing the tile cost by the instructions the tile
    takes therefore gives 1 cycle per MVMUL at every phase — and charging the
    phase multiplier per instruction as well would bill a HiFi4 matmul 2.5x
    what the hardware costs it (1+2+3+4 phases against 4).

    That the vendor per-tile numbers land exactly on the ISA docs'
    ``throughput_ipc: 1`` is an independent corroboration between two sources
    that do not cite each other — rung 1 of ``docs/plans/cost-model.md``'s
    calibration ladder, and the cheapest check available without silicon.
    """
    math = _math_model()
    for phase in range(4):
        assert math.fidelity_occupancy(phase) == 1, phase
    assert math.occupancy("MVMUL") == 1


def test_fidelity_occupancy_follows_the_table_rather_than_a_constant():
    """Double the published per-tile costs and the charge doubles with them."""
    math = _math_model()
    doubled = dict(math._fidelity)
    doubled["cycles_per_tile"] = {
        name: cycles * 2 for name, cycles in doubled["cycles_per_tile"].items()
    }
    math._fidelity = doubled
    assert [math.fidelity_occupancy(p) for p in range(4)] == [2, 2, 2, 2]


def test_a_unit_without_a_fidelity_block_has_no_fidelity_opinion():
    sfpu = UnitCostModel(load_costs("wormhole").unit("SFPU"), "wormhole")
    assert sfpu.fidelity_occupancy(0) is None
    assert not sfpu.scales_with_fidelity("SFPMAD")


def test_the_matrix_unit_charges_the_isa_doc_table_elsewhere():
    math = _math_model()
    assert math.occupancy("SHIFTXB") == 2  # 0.5 IPC
    for op in ("SETRWC", "INCRWC", "ZEROACC", "ZEROSRC", "CLEARDVALID", "MOVA2D"):
        assert math.occupancy(op) == 1, op
    assert math.provenance_of("MVMUL") == "isa_doc_derived"


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "model_test OK: cost model off by default, bounds preserved, "
        "MVMUL costed at 1 cycle per fidelity phase"
    )


if __name__ == "__main__":
    main()
