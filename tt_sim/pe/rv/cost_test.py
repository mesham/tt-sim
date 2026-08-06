"""The baby RISC-V load/store cost model, on and off.

Runs standalone (``python3 -m tt_sim.pe.rv.cost_test``) or under pytest. The
same split as the Tensix cost-model tests: ``tt_sim/perf/model_test.py`` pins
what the tables *say*, this pins what the RV interpreter *does* with what they
say.

Five claims:

1. With ``TT_SIM_COST_MODEL`` unset a core has no cost state at all and
   retires one instruction per cycle exactly as it always has. That is the
   property every replay guard depends on.
2. With it set, a load is charged the latency its address region is documented
   at, and that latency is spent as a **load-use interlock** rather than as
   occupancy: independent instructions after the load run for free, and only a
   dependent read stalls.
3. On Blackhole, *which* L1 latency a load is charged — the L0 d-cache hit
   row or the miss row — is decided per load by the four-line-tag model,
   including the documented store/fence/atomic flushes; Wormhole, which
   publishes no L0, is untouched by it. Likewise Blackhole's pipelined
   multiply spends its 2-cycle result latency as a scoreboard entry where
   Wormhole's blocking multiply keeps its occupancy charge.
4. The bound / provenance policy is honoured — regions the table does not name
   are charged nothing, the ``>=`` rows are charged at their low end, and the
   divide stays at its band's floor at any dividend magnitude.
5. Stores to L1 are rate-limited at the documented one-every-five-cycles, and
   stores anywhere else are not.
"""

import os
from contextlib import contextmanager

from tt_sim.device.device import DeviceMemory
from tt_sim.memory.memory import DRAM
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.pe.rv.cost import (
    RV_REGION_L1,
    RV_REGION_LOCAL_DATA_RAM,
    RV_REGION_MAILBOX_GROUP,
    RV_REGION_TDMA,
    RV_REGION_TENSIX_GPR_CFG,
    RV_REGION_TILECTRL_PIC_OVERLAY,
    RV_REGION_UNNAMED,
    classify_address,
    make_cost_state,
)
from tt_sim.pe.rv.rv32 import RV32IM
from tt_sim.util.conversion import conv_to_bytes

LOCAL_RAM_BASE = 0xFFB00000


@contextmanager
def _env(enabled):
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if enabled:
        os.environ["TT_SIM_COST_MODEL"] = "1"
    else:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


# -- instruction encoders, so the tests read as assembly --------------------


def _lw(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (0x2 << 12) | (rd << 7) | 0x03


def _sw(rs2, rs1, imm):
    imm &= 0xFFF
    return (
        ((imm >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (0x2 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
    )


def _addi(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (rd << 7) | 0x13


def _lui(rd, imm20):
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | 0x37


def _mul(rd, rs1, rs2):
    return (0x01 << 25) | (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33


def _div(rd, rs1, rs2):
    return (0x01 << 25) | (rs2 << 20) | (rs1 << 15) | (0x4 << 12) | (rd << 7) | 0x33


def _core(program, arch="wormhole", base=0x100):
    """An RV32IM core with L1 at 0 and a core-local data RAM at 0xFFB00000.

    Deliberately built by hand rather than through ``TensixTile``: this is a
    test of the interlock, and a whole tile would drag in five cores, the
    coprocessor and the NoC to observe one stall.
    """
    l1 = DRAM(0x2000)
    local = DRAM(0x1000)
    mem_map = MemoryMap()
    mem_map[AddressRange(0x0, l1.getSize())] = l1
    mem_map[AddressRange(LOCAL_RAM_BASE, local.getSize())] = local
    memory = DeviceMemory(mem_map)
    for i, word in enumerate(program):
        memory.write(base + i * 4, conv_to_bytes(word))
    cpu = RV32IM(base, [memory])
    cpu.rv_cost = make_cost_state(arch)
    cpu.start()
    return cpu, memory


def _run(cpu, cycles):
    for cycle in range(cycles):
        cpu.clock_tick(cycle)


# ---------------------------------------------------------------------------
# 1. Off by default.
# ---------------------------------------------------------------------------


def test_without_the_env_var_a_core_has_no_cost_state():
    with _env(False):
        assert make_cost_state("wormhole") is None
        assert make_cost_state("blackhole") is None


def test_a_core_built_without_an_arch_is_never_charged():
    """``driver/simple`` and the ISA unit tests build cores with no
    architecture; they must not start paying for a model they cannot name."""
    with _env(True):
        assert make_cost_state(None) is None


def test_off_means_a_dependent_load_use_pair_still_costs_two_cycles():
    with _env(False):
        cpu, _ = _core([_lw(15, 0, 0x40), _addi(15, 15, 1)])
        assert cpu.rv_cost is None
        _run(cpu, 2)
        assert cpu.register_file["pc"].read_uint() == 0x108


# ---------------------------------------------------------------------------
# 2. On: the load-use interlock.
# ---------------------------------------------------------------------------


def test_on_an_l1_load_is_charged_the_documented_wormhole_latency():
    with _env(True):
        state = make_cost_state("wormhole")
        assert state.load_latency[RV_REGION_L1] == 8
        assert state.load_latency[RV_REGION_LOCAL_DATA_RAM] == 2
        assert state.load_latency[RV_REGION_MAILBOX_GROUP] == 3
        assert state.load_latency[RV_REGION_TENSIX_GPR_CFG] == 4
        assert state.load_latency[RV_REGION_TILECTRL_PIC_OVERLAY] == 7


def test_a_dependent_read_stalls_for_latency_minus_one_cycles():
    """The docs state the table as "N - 1 independent instructions need to
    follow the load if the latency is to be entirely hidden", so a consumer in
    the very next slot waits exactly N - 1 cycles."""
    with _env(True):
        cpu, _ = _core([_lw(15, 0, 0x40), _addi(14, 15, 1)])
        _run(cpu, 8)
        # The load retires on cycle 0; the dependent addi cannot issue until
        # cycle 8 (latency 8 from L1), so 7 cycles are stalled and the addi is
        # still the pending instruction at the end of cycle 7.
        assert cpu.register_file["pc"].read_uint() == 0x104
        assert cpu.rv_cost.stall_cycles == 7
        cpu.clock_tick(8)
        assert cpu.register_file["pc"].read_uint() == 0x108


def test_independent_instructions_after_a_load_are_free():
    """The whole point of modelling a latency as a scoreboard rather than as
    occupancy: a load does not hold the core, it only makes one register
    late."""
    with _env(True):
        cpu, _ = _core([_lw(15, 0, 0x40)] + [_addi(14, 14, 1)] * 7)
        _run(cpu, 8)
        assert cpu.rv_cost.stall_cycles == 0
        assert cpu.register_file["pc"].read_uint() == 0x120


def test_a_load_from_core_local_ram_costs_one_stall_not_seven():
    with _env(True):
        cpu, _ = _core(
            [_lui(10, LOCAL_RAM_BASE >> 12), _lw(15, 10, 0), _addi(14, 15, 1)]
        )
        _run(cpu, 4)
        assert cpu.rv_cost.stall_cycles == 1
        assert cpu.rv_cost.loads_by_region[RV_REGION_LOCAL_DATA_RAM] == 1


def test_overwriting_a_pending_load_register_retires_the_hazard():
    """A later writer of the same register means an in-order core has nothing
    to wait for; without this the model would stall on a value the program
    already threw away."""
    with _env(True):
        cpu, _ = _core([_lw(15, 0, 0x40), _lui(15, 1), _addi(14, 15, 1)])
        _run(cpu, 3)
        assert cpu.rv_cost.stall_cycles == 0


def test_blackhole_charges_the_l0_hit_row_only_for_a_resident_line():
    """Blackhole's table gives L1 two latencies (2 on an L0 d-cache hit, >= 8
    on a miss), and since 2026-08-06 which one a load pays is decided per load
    by the four-line-tag model: the hit row is the *default* row (the low end,
    what an unknown access pattern is charged) and the miss row is reached
    when the loaded line is provably not resident. Wormhole publishes no L0 at
    all, so its single L1 row stands and the line model never engages."""
    with _env(True):
        wormhole = make_cost_state("wormhole")
        blackhole = make_cost_state("blackhole")
        assert wormhole.load_latency[RV_REGION_L1] == 8
        assert blackhole.load_latency[RV_REGION_L1] == 2
        assert blackhole.l1_miss_latency == 8
        assert blackhole._l0_enabled
        assert not wormhole._l0_enabled
        assert wormhole.l1_miss_latency is None
        # Blackhole moves the TDMA registers from the ">= 7" group to ">= 4".
        assert wormhole.load_latency[RV_REGION_TDMA] == 7
        assert blackhole.load_latency[RV_REGION_TDMA] == 4


# ---------------------------------------------------------------------------
# 2b. The Blackhole L0 data-cache line model.
# ---------------------------------------------------------------------------


def test_blackhole_a_cold_l1_load_misses_and_a_reload_of_the_line_hits():
    """The first touch of a line charges the miss row's >= 8; touching the
    same 16-byte line again charges the hit row's 2. The stalls say so: the
    first dependent read waits 7 cycles, the second waits 1."""
    with _env(True):
        cpu, _ = _core(
            [
                _lw(15, 0, 0x40),
                _addi(14, 15, 1),
                _lw(15, 0, 0x44),  # same 16-byte line, different word
                _addi(13, 15, 1),
            ],
            arch="blackhole",
        )
        _run(cpu, 12)
        assert cpu.rv_cost.l0_misses == 1
        assert cpu.rv_cost.l0_hits == 1
        assert cpu.rv_cost.stall_cycles == 7 + 1
        assert cpu.register_file["pc"].read_uint() == 0x110


def test_blackhole_a_chase_beyond_the_l0_capacity_misses_every_time():
    """A cyclic walk over five distinct lines against four tags misses on
    every load however the (unpublished) replacement works — the one-sided
    capacity argument, and the regime ``rv_load_chase`` measures at 8.098."""
    with _env(True):
        lines = [0x40, 0x50, 0x60, 0x70, 0x80]
        cpu, _ = _core([_lw(15, 0, a) for a in lines + lines], arch="blackhole")
        _run(cpu, 10)
        assert cpu.rv_cost.l0_misses == 10
        assert cpu.rv_cost.l0_hits == 0


def test_blackhole_a_hot_loop_within_four_lines_hits_after_warm_up():
    """Four distinct lines is what the cache can hold, and under the model's
    least-recently-loaded replacement a working set that fits never misses
    again once warm."""
    with _env(True):
        lines = [0x40, 0x50, 0x60, 0x70]
        cpu, _ = _core([_lw(15, 0, a) for a in lines + lines + lines], arch="blackhole")
        _run(cpu, 12)
        assert cpu.rv_cost.l0_misses == 4
        assert cpu.rv_cost.l0_hits == 8


def test_blackhole_an_l1_store_flushes_the_containing_line():
    """ "Stores to L1 flush the containing line, so the cache is never dirty"
    — a reload of a stored-to line is a documented miss, not a hit."""
    with _env(True):
        cpu, _ = _core(
            [_lw(15, 0, 0x40), _sw(0, 0, 0x44), _lw(14, 0, 0x40)],
            arch="blackhole",
        )
        _run(cpu, 3)
        assert cpu.rv_cost.l0_misses == 2
        assert cpu.rv_cost.l0_hits == 0


def test_blackhole_a_fence_flushes_the_whole_l0():
    """ "The entire L0 data cache will be flushed by any fence or atomic
    instruction." ``fence`` executes as a nop on these cores; its flush is
    still architectural."""
    with _env(True):
        fence = 0x0000000F
        cpu, _ = _core([_lw(15, 0, 0x40), fence, _lw(14, 0, 0x40)], arch="blackhole")
        _run(cpu, 3)
        assert cpu.rv_cost.l0_misses == 2
        assert cpu.rv_cost.l0_hits == 0


def test_wormhole_l1_loads_are_untouched_by_the_line_model():
    """Wormhole's L1 row is a single >= 8 with no L0 in front of it; the same
    program that hits on Blackhole charges 8 both times on Wormhole."""
    with _env(True):
        cpu, _ = _core(
            [
                _lw(15, 0, 0x40),
                _addi(14, 15, 1),
                _lw(15, 0, 0x44),
                _addi(13, 15, 1),
            ],
            arch="wormhole",
        )
        _run(cpu, 18)
        assert cpu.rv_cost.l0_misses == 0
        assert cpu.rv_cost.l0_hits == 0
        assert cpu.rv_cost.stall_cycles == 7 + 7


# ---------------------------------------------------------------------------
# 3. The policy: unnamed regions and bounded rows.
# ---------------------------------------------------------------------------


def test_the_regions_the_table_does_not_name_are_charged_nothing():
    """The NIU register block is the one that matters — every
    ``noc_async_*_barrier`` in every dataflow kernel polls it — and the ">= 7"
    row names the NoC *overlay*, a different block. Charging the overlay's
    number to the NIUs would be a guess with a citation stapled to it."""
    with _env(True):
        state = make_cost_state("wormhole")
        assert classify_address(0xFFB20000 + 0x200) == RV_REGION_UNNAMED
        assert classify_address(0xFFB30000 + 0x200) == RV_REGION_UNNAMED
        assert classify_address(0xFFC00000) == RV_REGION_UNNAMED
        assert state.load_latency[RV_REGION_UNNAMED] is None


def test_an_uncosted_region_leaves_the_timing_alone():
    with _env(True):
        cpu, _ = _core([_lui(10, 0xFFB20), _lw(15, 10, 0), _addi(14, 15, 1)])
        # No memory is mapped at the NIU base in this harness, so the issue
        # path is driven directly: the load is classified, charged nothing,
        # and the dependent instruction does not stall.
        cpu.register_file[10].write(conv_to_bytes(0xFFB20000))
        state = cpu.rv_cost
        assert state.can_issue(_lw(15, 10, 0), 0, cpu.register_file)
        assert state.load_latency[RV_REGION_UNNAMED] is None
        assert state.can_issue(_addi(14, 15, 1), 1, cpu.register_file)
        assert state.stall_cycles == 0


def test_every_named_region_is_classified_where_the_memory_map_puts_it():
    assert classify_address(0x0) == RV_REGION_L1
    assert classify_address(0x16E000) == RV_REGION_L1
    assert classify_address(LOCAL_RAM_BASE + 0x100) == RV_REGION_LOCAL_DATA_RAM
    assert classify_address(0xFFB11000) == RV_REGION_TDMA
    assert classify_address(0xFFB121B0) == RV_REGION_TILECTRL_PIC_OVERLAY
    assert classify_address(0xFFB40000) == RV_REGION_TILECTRL_PIC_OVERLAY
    assert classify_address(0xFFE00000) == RV_REGION_TENSIX_GPR_CFG
    assert classify_address(0xFFEF0000) == RV_REGION_TENSIX_GPR_CFG
    assert classify_address(0xFFE80000) == RV_REGION_MAILBOX_GROUP
    assert classify_address(0xFFEC0000) == RV_REGION_MAILBOX_GROUP


def test_the_bounded_rows_are_charged_at_their_low_end():
    with _env(True):
        state = make_cost_state("wormhole")
        assert state.model.load_latency_bound(RV_REGION_L1) == "at_least"
        assert state.model.load_latency_bound(RV_REGION_LOCAL_DATA_RAM) == "exact"
        # ">= 8" charged as 8, not 9 and not some invented average.
        assert state.load_latency[RV_REGION_L1] == 8


def test_the_branch_mispredict_penalty_is_sourced_and_deliberately_uncharged():
    """It is a cost per *mispredicted* branch, and nothing in the ISA docs or
    in tt-sim describes the predictor, so the count is unknowable. Kept
    reachable so a report can name the predictor as the gap."""
    with _env(True):
        assert make_cost_state("wormhole").model.branch_mispredict_observed == 3
        assert make_cost_state("blackhole").model.branch_mispredict_observed == 5


# ---------------------------------------------------------------------------
# 4. Stores, multiply and divide.
# ---------------------------------------------------------------------------


def test_sustained_l1_stores_are_held_to_one_every_five_cycles():
    with _env(True):
        cpu, _ = _core([_sw(0, 0, 0x40 + i * 64) for i in range(3)])
        _run(cpu, 11)
        assert cpu.rv_cost.l1_stores == 3
        # Cycle 0, 5, 10: two runs of four stalled cycles.
        assert cpu.rv_cost.stall_cycles == 8


def test_stores_outside_l1_are_one_per_cycle():
    with _env(True):
        cpu, _ = _core(
            [_lui(10, LOCAL_RAM_BASE >> 12)] + [_sw(0, 10, i * 4) for i in range(3)]
        )
        _run(cpu, 4)
        assert cpu.rv_cost.stall_cycles == 0
        assert cpu.rv_cost.l1_stores == 0


def test_blackhole_coalesces_neighbouring_l1_stores_and_wormhole_does_not():
    """Blackhole's store queue coalesces stores to the same 16-byte aligned
    region of L1 whose start addresses are within +/-4, and the constituent
    stores of a coalesced group go at one per cycle. Charging the flat five to
    those would over-charge by 5x against a document that says the opposite."""
    with _env(True):
        program = [_sw(0, 0, 0x40), _sw(0, 0, 0x44)]
        wormhole, _ = _core(program, arch="wormhole")
        _run(wormhole, 6)
        assert wormhole.rv_cost.stall_cycles == 4

        blackhole, _ = _core(program, arch="blackhole")
        _run(blackhole, 6)
        assert blackhole.rv_cost.stall_cycles == 0

        # The predicate has teeth: 0x40 and 0x50 are in different 16-byte
        # blocks, so Blackhole pays the same five cycles Wormhole does.
        apart, _ = _core([_sw(0, 0, 0x40), _sw(0, 0, 0x50)], arch="blackhole")
        _run(apart, 6)
        assert apart.rv_cost.stall_cycles == 4


def test_a_multiply_occupies_the_integer_unit_for_the_documented_cycles():
    """The one place the docs state blocking outright: "multiply instructions
    occupy the Integer Unit for two cycles, and the next instruction cannot
    enter the unit until the multiply instruction has finished". Blackhole
    pipelines it into EX1 + EX2, so an *independent* successor costs nothing
    there — only a dependent read pays (the next test)."""
    with _env(True):
        wormhole, _ = _core([_mul(10, 0, 0), _addi(11, 0, 1)], arch="wormhole")
        _run(wormhole, 3)
        assert wormhole.rv_cost.stall_cycles == 1

        blackhole, _ = _core([_mul(10, 0, 0), _addi(11, 0, 1)], arch="blackhole")
        _run(blackhole, 3)
        assert blackhole.rv_cost.stall_cycles == 0


def test_blackhole_a_dependent_multiply_chain_pays_the_ex2_latency():
    """ "Exactly one cycle in EX1, and then exactly one cycle in EX2": on
    Blackhole a multiply's occupancy is 1 but its result latency is 2, spent
    as a scoreboard entry like a load's. A dependent chain therefore runs at
    2 cycles per multiply — silicon's ``rv_mul_dep`` reads 1.985 where tt-sim
    read 1.000 before the entry existed. Wormhole publishes no multiply
    latency (its multiply *blocks*, charged as occupancy already), so its
    dependent chain is unchanged at the same 2 it always cost."""
    with _env(True):
        assert make_cost_state("blackhole").multiply_latency == 2
        assert make_cost_state("wormhole").multiply_latency is None

        # A reader in the very next slot waits exactly one cycle.
        pair, _ = _core([_mul(10, 12, 12), _addi(11, 10, 1)], arch="blackhole")
        _run(pair, 3)
        assert pair.rv_cost.stall_cycles == 1

        # A chain of three dependent multiplies retires one every two cycles.
        chain, _ = _core([_mul(10, 10, 10)] * 3, arch="blackhole")
        _run(chain, 5)
        assert chain.rv_cost.stall_cycles == 2
        assert chain.register_file["pc"].read_uint() == 0x100 + 3 * 4

        # Wormhole's chain: same total, all of it integer-unit occupancy.
        wh_chain, _ = _core([_mul(10, 10, 10)] * 3, arch="wormhole")
        _run(wh_chain, 5)
        assert wh_chain.rv_cost.stall_cycles == 2
        assert wh_chain.register_file["pc"].read_uint() == 0x100 + 3 * 4


def test_a_divide_by_one_costs_the_documented_special_case_not_the_general_one():
    with _env(True):
        state = make_cost_state("wormhole")
        assert state.divide_trivial == 2
        # "between six and 33 cycles", charged at the low end of the range.
        assert state.divide_general == 6
        # x12 = 1, so this is the two-cycle special case, not the six-cycle
        # general one: one stall rather than five.
        cpu, _ = _core([_addi(12, 0, 1), _div(10, 12, 12), _addi(11, 0, 1)])
        _run(cpu, 4)
        assert cpu.rv_cost.stall_cycles == 1
        # A general divide by something other than 0, 1 or -1 costs five.
        general, _ = _core([_addi(12, 0, 7), _div(10, 12, 12), _addi(11, 0, 1)])
        _run(general, 8)
        assert general.rv_cost.stall_cycles == 5


def test_the_divide_charge_is_the_floor_at_any_dividend_magnitude():
    """The benchmark's own operands — ``0x12345678 / 3``, a 29-significant-bit
    dividend — cost 33.001 cycles on Blackhole silicon and are still charged
    the documented band's floor of 6 here, deliberately: "between six and 33
    cycles ... dependent upon the magnitude of the dividend" is a data
    dependence whose *function* neither BabyRISCV page publishes (re-searched
    2026-08-06 — no iteration rule, no bits-per-cycle, no formula), so any
    per-magnitude charge would be a curve fitted to one silicon point. The
    module docstring records the decision; this pins that the charge really
    is operand-independent above the documented special cases."""
    with _env(True):
        # lui+addi materialise 0x12345678: lui 0x12345, addi 0x678 (< 0x800,
        # so no sign-extension correction is needed).
        program = [
            _lui(12, 0x12345),
            _addi(12, 12, 0x678),
            _addi(13, 0, 3),
            _div(10, 12, 13),
            _addi(11, 0, 1),
        ]
        for arch in ("wormhole", "blackhole"):
            cpu, _ = _core(program, arch=arch)
            _run(cpu, 10)
            # Three setup cycles, the divide, five stalled cycles (the 6-cycle
            # floor), then the successor: 27 fewer stalls than the silicon
            # reading at this operand, and recorded as such.
            assert cpu.rv_cost.stall_cycles == 5
            assert cpu.register_file["pc"].read_uint() == 0x100 + 5 * 4


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "rv cost_test OK: no cost state by default, load latency spent as a "
        "load-use interlock, unnamed regions charged nothing, L1 stores rate "
        "limited"
    )


if __name__ == "__main__":
    main()
