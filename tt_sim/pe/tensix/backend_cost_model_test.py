"""The SFPU, ThCon, packer, config and sync units driven with the cycle costs.

Runs standalone (``python3 -m tt_sim.pe.tensix.backend_cost_model_test``) or
under pytest. Companion to ``matrix_cost_model_test.py``, which covers the FPU
— the first unit wired — and to ``tt_sim/perf/model_test.py``, which pins what
:mod:`tt_sim.perf.model` *says* rather than what a unit *does* with what it
says.

What these five units add to the story the matrix unit started:

1. **Off is still off.** None of them has a cost model unless
   ``TT_SIM_COST_MODEL`` is truthy, and with it unset every one of them retires
   in the tick it was issued exactly as before. That is the property the replay
   guards and ``six``'s pinned PCC depend on, and it is asserted per unit
   rather than once, because a single unit arming ``busy_until`` by accident
   would move every cycle count downstream of it.
2. **The SFPU is a throughput check, not a new cost.** Its table is the
   best-sourced in the file — the ISA docs publish a latency for all 42 opcodes
   — and every one of them comes out at a 1-cycle *occupancy*, because the
   unit's five sub-units are pipelined and "can only accept one instruction per
   cycle from the outside world". So the 2-cycle latency of ``SFPMAD`` and
   friends is time-to-result, not time-the-unit-is-held, and tt-sim's
   one-op-per-cycle issue was already reproducing the documented throughput.
   Same shape of answer as the matrix unit's, from a different direction.
3. **ThCon is where a Tensix instruction first costs more than a cycle.** Its
   table is the one the ISA docs publish as an occupancy table outright
   ("Number of cycles required for execution"), and it is mostly *not* ones:
   3 for the GPR arithmetic ops, 3 for the loads and stores, 15 for ``ATCAS``.
   Those are charged at the low end of their bounds, so the modelled cost is a
   floor, and they really do hold the unit — which is the first time any of
   this machinery changes a simulated cycle count.
"""

import os
from contextlib import contextmanager

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor

#: ``ADDDMAREG GPR[2] = GPR[0] + 1`` with ``OpBisConst``: the cheapest ThCon op
#: with a visible side effect, so a test can tell a retired instruction from a
#: stalled one. Fields per ``tensix_instructions.yaml``: OpARegIndex at bit 0,
#: OpBRegIndex at 6, ResultRegIndex at 12, OpBisConst at 23.
ADDDMAREG_G2_EQ_G0_PLUS_1 = (88 << 24) | (1 << 23) | (2 << 12) | (1 << 6) | 0

#: ``SFPNOP``, which takes no arguments at all.
SFPNOP = 143 << 24

#: SFPU opcodes the backend implements but ``load_costs("wormhole")`` drops,
#: because the cost table marks them ``arch: blackhole``. The handler map is
#: shared across arches (Blackhole's Tensix ISA is a strict superset); the cost
#: table is not. Covered on the Blackhole backend instead.
BLACKHOLE_ONLY_SFPU = ("SFPGT", "SFPLE", "SFPMUL24", "SFPARECIP")


@contextmanager
def _backend(cost_model):
    """A fresh Tensix backend with the cost model forced on or off."""
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if cost_model:
        os.environ["TT_SIM_COST_MODEL"] = "1"
    else:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    try:
        yield TensixCoProcessor(
            None,
            WORMHOLE_PROFILE.tensix_cfg_state_size,
            WORMHOLE_PROFILE.tensix_thd_state_size,
        ).getBackend()
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


#: The units this test covers, by their ``ex_resource`` name — which is also
#: their key in both ``tensix_instructions.yaml`` and the cost table.
WIRED = ("SFPU", "THCON", "PACK", "SYNC")

#: ``MATH`` is wired too, in ``matrix_cost_model_test.py``. ``UNPACK``,
#: ``XMOV`` and ``CFG`` are not: ``UNPACK``'s cost is the one genuinely
#: non-constant entry in the file (a ">= 2" address phase plus a
#: throttle-mode-dependent data phase), ``XMOV``'s published 1 is issue cost
#: only, and ``CFG`` is the finding — see
#: ``test_the_config_unit_is_left_uncosted_and_that_is_a_finding``.


# ---------------------------------------------------------------------------
# 1. Off by default, per unit.
# ---------------------------------------------------------------------------


def test_without_the_env_var_no_unit_has_a_cost_model():
    with _backend(False) as backend:
        units = [backend.backend_units[name] for name in WIRED]
        units += [backend.backend_units[name] for name in ("MATH", "XMOV", "CFG")]
        units += backend.unpacker_units
        for unit in units:
            assert unit.cost_model is None, unit.unit_name
            assert unit.instruction_occupancy("ATCAS", 0) is None, unit.unit_name


def test_off_means_thcon_retires_one_instruction_per_cycle_as_before():
    """ThCon is the unit whose timing the model actually moves, so this is the
    one that has to be shown unmoved with the switch off."""
    with _backend(False) as backend:
        thcon = backend.backend_units["THCON"]
        gprs = backend.gpr.getRegisters(0)
        for cycle in range(4):
            assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
            thcon.clock_tick(cycle)
            assert thcon.busy_until is None
            assert gprs[2] == 1
            assert not thcon.hasInflightInstructionsFromThread(0)


def test_off_means_the_sfpu_retires_one_instruction_per_cycle_as_before():
    with _backend(False) as backend:
        sfpu = backend.backend_units["SFPU"]
        for cycle in range(3):
            assert sfpu.issueInstruction(SFPNOP, 0)
            sfpu.clock_tick(cycle)
            assert sfpu.busy_until is None
            assert not sfpu.hasInflightInstructionsFromThread(0)


# ---------------------------------------------------------------------------
# 2. The SFPU: 42 documented latencies, every one a one-cycle occupancy.
# ---------------------------------------------------------------------------


def test_every_sfpu_opcode_the_backend_implements_costs_exactly_one_cycle():
    """The unit's throughput sanity check, and it is the substantive result.

    The ISA docs give the SFPU a per-instruction *latency* — 2 cycles for the
    arithmetic and LUT ops, 1 for the rest — and separately state that the unit
    "can only accept one instruction per cycle from the outside world". Those
    are two different numbers, and the one occupancy wants is the second: the
    five sub-units are pipelined, so a 2-cycle ``SFPMAD`` is 2 cycles until its
    result is readable, not 2 cycles during which nothing else may issue.
    Charging the latency instead would have doubled the modelled cost of every
    SFPU-heavy kernel against a document that never claimed it.
    """
    with _backend(True) as backend:
        sfpu = backend.backend_units["SFPU"]
        assert sfpu.cost_model is not None
        costed = 0
        shared = [
            name
            for name in sfpu.opcode_to_method_map
            if name not in BLACKHOLE_ONLY_SFPU
        ]
        for name in shared:
            occupancy = sfpu.instruction_occupancy(name, 0)
            if occupancy is None:
                # SFPLOADMACRO alone: the doc's latency column reads "Complex"
                # because it expands to up to four more instructions, so it is
                # deliberately left uncosted rather than given a made-up number.
                assert name == "SFPLOADMACRO", name
                continue
            assert occupancy == 1, name
            costed += 1
        assert costed == len(shared) - 1


def test_the_two_cycle_sfpu_ops_are_two_cycles_of_latency_not_occupancy():
    with _backend(True) as backend:
        model = backend.backend_units["SFPU"].cost_model
        for name in ("SFPMAD", "SFPMUL", "SFPADD", "SFPLUT", "SFPLUTFP32"):
            assert model.occupancy(name) == 1, name
        # ...and the model charges occupancy, so an SFPU-only run cannot move.
        sfpu = backend.backend_units["SFPU"]
        for cycle in range(3):
            assert sfpu.issueInstruction(SFPNOP, 0)
            sfpu.clock_tick(cycle)
            assert sfpu.busy_until is None


def test_the_blackhole_sfpu_superset_is_costed_from_the_blackhole_page():
    """Blackhole adds four SFPU opcodes and changes none of the shared ones."""
    previous = os.environ.get("TT_SIM_COST_MODEL")
    os.environ["TT_SIM_COST_MODEL"] = "1"
    try:
        backend = TensixCoProcessor(None, blackhole=True).getBackend()
        model = backend.backend_units["SFPU"].cost_model
        for name in ("SFPGT", "SFPLE", "SFPMUL24", "SFPARECIP"):
            assert model.occupancy(name) == 1, name
        assert model.arch == "blackhole"
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


# ---------------------------------------------------------------------------
# 3. ThCon: the first Tensix instruction to cost more than a cycle.
# ---------------------------------------------------------------------------


def test_thcon_charges_the_isa_docs_own_occupancy_column():
    with _backend(True) as backend:
        thcon = backend.backend_units["THCON"]
        assert thcon.instruction_occupancy("SETDMAREG", 0) == 1
        assert thcon.instruction_occupancy("REG2FLOP", 0) == 2  # ">= 2"
        assert thcon.instruction_occupancy("FLUSHDMA", 0) == 2  # ">= 2"
        for name in ("ADDDMAREG", "SUBDMAREG", "MULDMAREG", "CMPDMAREG"):
            assert thcon.instruction_occupancy(name, 0) == 3, name  # "3 or 4"
        for name in ("STOREIND", "STOREREG", "LOADIND", "LOADREG", "ATSWAP"):
            assert thcon.instruction_occupancy(name, 0) == 3, name  # ">= 3"
        assert thcon.instruction_occupancy("ATCAS", 0) == 15  # ">= 15"
        # Every one of those but SETDMAREG came from a bound, so the modelled
        # count is a floor and the model says so.
        assert thcon.cost_model.is_exact("SETDMAREG")
        for name in ("REG2FLOP", "ADDDMAREG", "LOADREG", "ATCAS"):
            assert not thcon.cost_model.is_exact(name), name


def test_a_three_cycle_thcon_op_holds_the_unit_and_stalls_the_thread():
    """The first real multi-cycle instruction in the tree, driven end to end
    with the shipped table rather than a stub.

    An occupied unit **refuses** the next instruction rather than parking it in
    its queue, which is both the closer reading of the docs (the issuing thread
    "is unable to start any further instruction") and the thing that keeps a
    thread's program in order — see ``TensixBackendUnit.is_occupied``.
    """
    with _backend(True) as backend:
        thcon = backend.backend_units["THCON"]
        gprs = backend.gpr.getRegisters(0)

        assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(0)
        # The instruction itself retires in the cycle it issued; what the cost
        # buys is that the *unit* is unavailable for two more.
        assert gprs[2] == 1
        assert thcon.busy_until == 3

        gprs[0] = 10
        for cycle in (1, 2):
            assert not thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0), cycle
            thcon.clock_tick(cycle)
            assert gprs[2] == 1, cycle

        # The deadline cycle: backend units tick before the frontend issues, so
        # the deadline tick clears the occupancy and the retry lands the same
        # cycle rather than one late.
        thcon.clock_tick(3)
        assert thcon.busy_until is None
        assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(4)
        assert gprs[2] == 11


def test_an_occupied_thcon_tells_the_pump_when_to_come_back():
    with _backend(True) as backend:
        thcon = backend.backend_units["THCON"]
        assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(0)
        assert thcon.next_wake_cycle(0) == 3
        assert thcon.next_wake_cycle(2) == 3
        thcon.clock_tick(3)
        assert thcon.next_wake_cycle(3) is None


def test_a_busy_unit_refuses_every_thread_not_just_the_issuing_one():
    """ThCon has no internal pipelining and executes one instruction at a time,
    so an occupancy is a whole-unit property. Back-pressure reaches the
    frontend the way it always has — ``issueInstruction`` returning False — so
    no new path had to be built for it."""
    with _backend(True) as backend:
        thcon = backend.backend_units["THCON"]
        assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(0)
        for thread in (0, 1, 2):
            assert not thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, thread)


def test_with_the_model_off_no_unit_is_ever_occupied():
    """``is_occupied`` is on the issue path of every Tensix instruction, so the
    off case has to be one attribute read that is always False."""
    with _backend(False) as backend:
        for unit in list(backend.backend_units.values()) + backend.unpacker_units:
            assert not unit.is_occupied(), unit.unit_name
        thcon = backend.backend_units["THCON"]
        for cycle in range(3):
            assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
            thcon.clock_tick(cycle)
            assert not thcon.is_occupied()


# ---------------------------------------------------------------------------
# 4. The policy: a gap stays a gap, and a bound stays a bound.
# ---------------------------------------------------------------------------


def test_the_uncosted_opcodes_of_each_unit_are_charged_nothing():
    """``provenance: unknown`` entries and opcodes absent from the table alike
    keep the same-cycle retire. Charging them a plausible-looking 1 would be
    indistinguishable, in a report, from a documented 1."""
    with _backend(True) as backend:
        # In the table, explicitly unknown, and carrying no numbers.
        assert backend.backend_units["THCON"].instruction_occupancy("RSTDMA", 0) is None
        assert backend.backend_units["PACK"].instruction_occupancy("TBUFCMD", 0) is None
        assert backend.backend_units["SYNC"].instruction_occupancy("MOP", 0) is None
        # Not in this unit's table at all.
        assert backend.backend_units["SFPU"].instruction_occupancy("MVMUL", 0) is None
        assert backend.backend_units["PACK"].instruction_occupancy("NOPE", 0) is None


def test_the_units_left_unwired_still_have_no_opinion():
    """Wiring is per unit, so "the model is on" must not quietly mean "every
    unit is costed". The unpackers and the mover keep the same-cycle retire
    even with ``TT_SIM_COST_MODEL`` set."""
    with _backend(True) as backend:
        units = [backend.backend_units[n] for n in ("XMOV", "CFG", "TDMA")]
        units += list(backend.unpacker_units)
        for unit in units:
            assert unit.cost_model is None, unit.unit_name
            assert unit.instruction_occupancy("UNPACR", 0) is None, unit.unit_name
            assert unit.instruction_occupancy("XMOV", 0) is None, unit.unit_name


# ---------------------------------------------------------------------------
# 5. Packer, config and sync: documented, and documented as one cycle.
# ---------------------------------------------------------------------------


def test_the_packer_charges_the_issue_cost_and_not_a_guessed_drain():
    with _backend(True) as backend:
        packer = backend.backend_units["PACK"]
        # "At most one of these instructions can be started per cycle."
        assert packer.instruction_occupancy("PACR", 0) == 1
        assert packer.cost_model.is_exact("PACR")
        # The drain — how long the packers take to write the tile out — has no
        # published per-op figure anywhere, and is deliberately not invented.
        assert packer.cost_model.occupancy("TBUFCMD") is None


def test_the_config_unit_is_left_uncosted_and_that_is_a_finding():
    """The one unit whose table is fully published and still not wired.

    Everything tt-sim implements in it is one cycle except ``RDCFG``, which the
    Wormhole page gives ">= 2" — and charging that documented 2 makes the
    ``matmulblock`` Blackhole guard compute a **wrong answer**: five ``SETC16``
    on the math thread and four ``WRCFG`` on the pack thread land one cycle
    late, and something downstream reads config that has not arrived. Refusing
    the issue rather than deferring it does not fix it, so this is not the
    frontend reordering one thread's stream; it is a missing ordering
    guarantee between a config write and its readers.

    The temptation is to charge ``RDCFG`` a 1 and move on. That would make the
    guard pass by asserting a number the ISA docs do not give, which is exactly
    the failure mode the provenance convention exists to prevent — so the unit
    stays uncosted and the divergence stays visible.
    """
    with _backend(True) as backend:
        config = backend.backend_units["CFG"]
        assert config.cost_model is None
        assert config.instruction_occupancy("RDCFG", 0) is None
        assert config.instruction_occupancy("SETC16", 0) is None


def test_the_sync_unit_charges_one_cycle_throughout():
    """Every Sync Unit op is a single cycle in both arches' tables. What this
    unit costs a kernel is wait-gate time, which is not occupancy at all — the
    page is explicit that ``SEMWAIT`` execution "consists purely of passing
    them over to the Wait Gate" — and tt-sim already models that separately."""
    with _backend(True) as backend:
        sync = backend.backend_units["SYNC"]
        for name in sync.opcode_to_method_map:
            assert sync.instruction_occupancy(name, 0) == 1, name


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "backend_cost_model_test OK: five more units wired, SFPU at one cycle "
        "per op, ThCon multi-cycle and back-pressuring, off by default"
    )


if __name__ == "__main__":
    main()
