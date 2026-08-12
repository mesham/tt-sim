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
from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.registers import SrcRegister
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixConfigurationConstants

#: ``ADDDMAREG GPR[2] = GPR[0] + 1`` with ``OpBisConst``: the cheapest ThCon op
#: with a visible side effect, so a test can tell a retired instruction from a
#: stalled one. Fields per ``tensix_instructions.yaml``: OpARegIndex at bit 0,
#: OpBRegIndex at 6, ResultRegIndex at 12, OpBisConst at 23.
ADDDMAREG_G2_EQ_G0_PLUS_1 = (88 << 24) | (1 << 23) | (2 << 12) | (1 << 6) | 0

#: ``SFPNOP``, which takes no arguments at all.
SFPNOP = 143 << 24

#: ``RDCFG GPR[5] = CFG[12]``. Opcode 177; ``CfgReg`` at bit 0, ``GprAddress``
#: at bit 16.
RDCFG_G5_FROM_CFG12 = (177 << 24) | (5 << 16) | 12

#: The config register ``RDCFG_G5_FROM_CFG12`` reads, and a recognisable value
#: to put in it.
RDCFG_SOURCE_INDEX = 12
RDCFG_SOURCE_VALUE = 0xABCD1234

#: ``CFGSHIFTMASK CFG[41] += SCRATCH[0]`` — the Blackhole-only op, in the only
#: mode tt-sim models (no circular shift, full-width mask, "old + scratch", no
#: masking of the old value: see ``config.CFGSHIFTMASK_MODELLED_MODE``). Opcode
#: 0xB8; fields per ``tensix_instructions.yaml``, matching ``blackhole_ops_test``.
#: It is the only opcode in either arch's cost table that costs more than one
#: cycle on a unit with IPC groups.
CFGSHIFTMASK_CFG41_SCRATCH0 = (
    (0xB8 << 24) | (1 << 23) | (3 << 20) | (31 << 15) | (0 << 10) | (0 << 8) | 41
)


def setc16_math_offset(value):
    """``SETC16 DEST_TARGET_REG_CFG_MATH_Offset = value``.

    Thread-config register 1 on Wormhole, and the write at the centre of the
    ordering bug: the matrix unit reads it to place every ``MVMUL``'s Dst
    accumulation, and the math thread flips it between the two halves of Dst
    (0 / 0x200) between blocks. Opcode 178; value at bit 0, register at 16.
    """
    return (178 << 24) | (1 << 16) | value


class _ForcedOccupancy:
    """A stand-in for a :class:`~tt_sim.perf.model.UnitCostModel`.

    Charges what the ``dict`` says and one cycle otherwise, so a test can put a
    multi-cycle cost on the config unit **without touching the cost tables** —
    which matters, because ``RDCFG``'s 1-cycle occupancy there is a silicon
    measurement corroborated by two hardware runs and must not be edited to
    make a test reach a code path.
    """

    def __init__(self, charges, groups=None, latencies=None):
        self.charges = charges
        self.groups = groups or {}
        self.latencies = latencies or {}

    def occupancy(self, instruction_name):
        return self.charges.get(instruction_name, 1)

    def latency(self, instruction_name):
        """Residency, defaulting to one cycle — i.e. to no residency at all.

        A separate dict from ``charges`` on purpose: the config unit's whole
        point is that its Latency and IPC columns are different numbers, so a
        stand-in that derived one from the other would hide the distinction the
        tests using it are about.
        """
        return self.latencies.get(instruction_name, 1)

    def is_exact(self, instruction_name):
        return True

    @property
    def has_ipc_groups(self):
        return bool(self.groups)

    def ipc_group(self, instruction_name):
        """No groups unless a test asks for them, which is the Wormhole answer.

        Wormhole's Configuration Unit page publishes no "IPC group" column, so
        the real model returns ``None`` here too and the hold is whole-unit.
        """
        return self.groups.get(instruction_name)


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
WIRED = ("SFPU", "THCON", "PACK", "SYNC", "CFG")

#: ``MATH`` is wired too, in ``matrix_cost_model_test.py``, and so are
#: ``UNPACK`` and ``XMOV`` since 2026-08-06 — in ``unpacker_cost_model_test.py``
#: and ``mover_cost_model_test.py``, because neither is a flat table lookup:
#: the unpacker's charge is a function of the transfer size and the throttle
#: config, and the mover's is a function of the transfer size and the
#: bandwidth table. ``TDMA`` is the only unit left unwired.


# ---------------------------------------------------------------------------
# 1. Off by default, per unit.
# ---------------------------------------------------------------------------


def test_without_the_env_var_no_unit_has_a_cost_model():
    with _backend(False) as backend:
        units = [backend.backend_units[name] for name in WIRED]
        units += [backend.backend_units[name] for name in ("MATH", "XMOV")]
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
        # buys is that the *unit* is unavailable afterwards. The hold runs
        # from the cycle the instruction was *accepted* — one before this
        # retire tick, since backend units tick before the wait gates — so a
        # 3-cycle occupancy means "the next instruction enters 3 cycles after
        # this one did", deadline 2. Anchoring at retire made every ThCon op
        # cost 4 cycles at the issuing thread where silicon measures 2.97.
        assert gprs[2] == 1
        assert thcon.busy_until == 2

        gprs[0] = 10
        assert not thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(1)
        assert gprs[2] == 1
        assert not thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)

        # The deadline cycle: backend units tick before the frontend issues, so
        # the deadline tick clears the occupancy and the retry lands the same
        # cycle rather than one late.
        thcon.clock_tick(2)
        assert thcon.busy_until is None
        assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(3)
        assert gprs[2] == 11


def test_an_occupied_thcon_tells_the_pump_when_to_come_back():
    with _backend(True) as backend:
        thcon = backend.backend_units["THCON"]
        assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(0)
        # Deadline 2, not 3: the hold runs from the acceptance cycle.
        assert thcon.next_wake_cycle(0) == 2
        assert thcon.next_wake_cycle(1) == 2
        thcon.clock_tick(2)
        assert thcon.next_wake_cycle(2) is None


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
    unit is costed".

    Down to one unit since 2026-08-06: the Miscellaneous Unit (``TDMA``), whose
    every op is one cycle by one blanket sentence, so charging it would move
    nothing and the allow-list is more useful meaning "a unit somebody reasoned
    about". The unpackers and the mover are now wired — see
    ``unpacker_cost_model_test`` and ``mover_cost_model_test``.
    """
    with _backend(True) as backend:
        misc = backend.backend_units["TDMA"]
        assert misc.cost_model is None, misc.unit_name
        assert misc.instruction_occupancy("SETADCXY", 0) is None


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


def test_the_config_unit_charges_nothing_on_wormhole():
    """Wired 2026-08-04, last of the six — and on Wormhole it is a true no-op.

    Every entry in that arch's table is one cycle, and ``occupy_for`` no-ops at
    <= 1, so the unit can never hold itself. That half is worth its own test
    because it is what makes the wiring a one-arch timing change: no Wormhole
    guard can move, whatever the Blackhole ones do.

    It got here the interesting way. Until 2026-08-04 ``RDCFG``'s occupancy read
    ">= 2" — the Wormhole page's documented *latency*, copied into the occupancy
    field — and charging that made the ``matmulblock`` guards compute a **wrong
    answer**. Blackhole silicon then measured ``RDCFG`` at 1.0 cycles of
    occupancy, and the ordering bug that made a late config write corrupt a
    matmul was found and fixed in ``TensixBackendUnit.clock_tick`` (see
    ``test_a_multi_cycle_config_op_does_not_delay_the_batch_beside_it``).
    """
    with _backend(True) as backend:
        config = backend.backend_units["CFG"]
        assert config.cost_model is not None
        assert config.cost_model.arch == "wormhole"
        for name in ("RDCFG", "SETC16", "WRCFG", "RMWCIB0"):
            assert config.instruction_occupancy(name, 0) == 1, name
        # Nothing in the Wormhole table can arm the hold.
        config.setConfig(0, RDCFG_SOURCE_INDEX, RDCFG_SOURCE_VALUE)
        for cycle in range(3):
            assert config.issueInstruction(RDCFG_G5_FROM_CFG12, 0)
            config.clock_tick(cycle)
            assert config.busy_until is None


def test_the_config_unit_charges_blackholes_two_cycle_shift_mask():
    """``CFGSHIFTMASK`` is the only entry above one cycle in either arch's
    config table, and the whole reason wiring this unit was a timing change
    rather than a formality: ``throughput_ipc: 0.5``, "requires two cycles in
    stage 0", and the Blackhole ``untilize`` guard really executes it."""
    previous = os.environ.get("TT_SIM_COST_MODEL")
    os.environ["TT_SIM_COST_MODEL"] = "1"
    try:
        backend = TensixCoProcessor(None, blackhole=True).getBackend()
        config = backend.backend_units["CFG"]
        assert config.cost_model.arch == "blackhole"
        assert config.instruction_occupancy("CFGSHIFTMASK", 0) == 2
        # Every other opcode this unit implements is still one cycle, on this
        # arch too — so the hold is reachable from exactly one instruction.
        for name in ("RDCFG", "SETC16", "WRCFG", "STREAMWRCFG"):
            assert config.instruction_occupancy(name, 0) == 1, name
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def test_a_multi_cycle_config_op_does_not_delay_the_batch_beside_it():
    """The regression test for the ordering bug, at the cycle it happens in.

    The config unit accepts a whole *batch* per cycle — up to three ``SETC16``,
    one per thread, alongside one of the shared-IPC-group ops — because those
    are parallel paths in the hardware, and the issuing threads are told the
    instructions are accepted and move on in the same cycle. Charging the first
    member of that batch a multi-cycle occupancy used to leave the rest of it
    queued for a later cycle, which is a *reordering of an already-accepted
    write behind instructions its own thread issued afterwards*: with ``RDCFG``
    at 2 cycles the math thread's ``SETC16`` of
    ``DEST_TARGET_REG_CFG_MATH_Offset`` landed two cycles late, after two of the
    same thread's ``MVMUL``s had already read the stale offset and accumulated
    into the wrong half of Dst, and ``matmul_block`` came out wrong.

    What the hardware provides, and what this pins: occupancy is throughput
    back-pressure on the *next* instruction to enter the unit, while each
    accepted instruction commits at its own documented latency. So the batch
    retires, and only then is the unit held.

    Note what is deliberately *not* done here: the table's ``RDCFG`` occupancy
    is not edited. That 1 is a silicon measurement corroborated by two hardware
    runs; the multi-cycle cost comes from a local stand-in cost model instead.
    """
    with _backend(True) as backend:
        config = backend.backend_units["CFG"]
        config.cost_model = _ForcedOccupancy({"RDCFG": 2})
        config.setConfig(0, RDCFG_SOURCE_INDEX, RDCFG_SOURCE_VALUE)

        # One cycle's batch, in the order the matmulblock trace issues it: the
        # unpack thread's RDCFG, then the math thread's SETC16.
        assert config.issueInstruction(RDCFG_G5_FROM_CFG12, 0)
        assert config.issueInstruction(setc16_math_offset(0x200), 1)
        config.clock_tick(0)

        # Both members of the batch took effect in the cycle they were accepted
        # for. The SETC16 is the one that used to be left behind.
        assert backend.gpr.getRegisters(0)[5] == RDCFG_SOURCE_VALUE
        assert (
            backend.getThreadConfigValue(1, "DEST_TARGET_REG_CFG_MATH_Offset") == 0x200
        )
        assert not config.next_instruction

        # The cost is still charged: the unit is held (from the acceptance
        # cycle, one before the retire tick — deadline 1, not 2), and the
        # back-pressure lands where it belongs — on the *next* instruction,
        # which the wait gate then retries, keeping each thread in order.
        assert config.busy_until == 1
        assert not config.issueInstruction(setc16_math_offset(0), 1)
        config.clock_tick(1)
        assert (
            backend.getThreadConfigValue(1, "DEST_TARGET_REG_CFG_MATH_Offset") == 0x200
        )
        assert config.busy_until is None
        assert config.issueInstruction(setc16_math_offset(0), 1)
        config.clock_tick(2)
        assert backend.getThreadConfigValue(1, "DEST_TARGET_REG_CFG_MATH_Offset") == 0


def test_a_held_ipc_group_still_lets_a_different_group_through():
    """The regression test for per-group occupancy, on the one unit that has it.

    Blackhole's Configuration Unit page is the only Tensix page that publishes
    an **"IPC group" column**: ``SETC16`` alone is ``ThreadConfig``, and
    ``STREAMWRCFG`` / ``WRCFG`` / ``CFGSHIFTMASK`` / ``RMWCIB`` / ``RDCFG`` are
    all ``Config``, whose "sustained throughput across the entire group is
    limited to one instruction per cycle (or half an instruction per cycle if
    ``CFGSHIFTMASK`` is used)". So a ``CFGSHIFTMASK``'s two cycles are two
    cycles of the ``Config`` group and say nothing at all about ``SETC16``.

    A whole-unit ``busy_until`` could not express that, and this test is the
    difference: with the old mechanism the ``SETC16`` below is refused in the
    cycle after the ``CFGSHIFTMASK``, which the hardware issues. The ``RDCFG``
    beside it is the control — same cycle, same unit, and correctly refused,
    because it *is* in the held group.

    Uses the real Blackhole cost table rather than a stand-in: ``CFGSHIFTMASK``
    is the only opcode in either arch's file whose occupancy exceeds one cycle
    on a unit with groups, so it is the whole reason the mechanism exists and
    there is nothing to be gained by faking it.
    """
    previous = os.environ.get("TT_SIM_COST_MODEL")
    os.environ["TT_SIM_COST_MODEL"] = "1"
    try:
        backend = TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size,
            BLACKHOLE_PROFILE.tensix_thd_state_size,
            blackhole=True,
        ).getBackend()
        config = backend.backend_units["CFG"]

        # The groups, straight out of the table's column.
        assert config.instruction_group("CFGSHIFTMASK") == "Config"
        assert config.instruction_group("RDCFG") == "Config"
        assert config.instruction_group("SETC16") == "ThreadConfig"

        config.setConfig(0, RDCFG_SOURCE_INDEX, RDCFG_SOURCE_VALUE)
        assert config.issueInstruction(CFGSHIFTMASK_CFG41_SCRATCH0, 0)
        config.clock_tick(0)
        # Two cycles, charged to Config and to Config only — counted from the
        # acceptance cycle (one before the retire tick), hence deadline 1.
        assert config.busy_groups == {"Config": 1}
        assert config.busy_until == 1

        # While held: the Config group refuses...
        assert not config.issueInstruction(RDCFG_G5_FROM_CFG12, 0)
        # ...and ThreadConfig does not. THIS is the assertion the whole-unit
        # hold failed: SETC16 is outside the held group and the hardware takes
        # it.
        assert config.issueInstruction(setc16_math_offset(0x200), 1)
        config.clock_tick(1)
        assert config.get_threadConfig_entry(1, 1) == 0x200
        # Retired in the cycle it was accepted for, and the hold released on
        # schedule at its deadline tick.
        assert not config.next_instruction
        assert config.busy_groups == {}
        assert config.busy_until is None
        # The RDCFG is still refused for one more cycle — the unit's own
        # SETC16-retired-last-cycle rule, nothing to do with the occupancy —
        # and then goes through.
        assert not config.issueInstruction(RDCFG_G5_FROM_CFG12, 0)
        config.clock_tick(2)
        assert config.issueInstruction(RDCFG_G5_FROM_CFG12, 0)
        config.clock_tick(3)
        assert backend.gpr.getRegisters(0)[5] == RDCFG_SOURCE_VALUE
    finally:
        TensixConfigurationConstants.use_blackhole(False)
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def test_an_ungrouped_unit_still_holds_the_whole_unit():
    """The other half: no published groups means no groups invented.

    Every unit but Blackhole's config unit keys its occupancy under ``None``,
    so ``is_occupied()`` is the whole-unit question it always was. ThCon is the
    one to check it on — its page says outright that the unit "is executing at
    most one instruction at a time, and has no internal pipelining", so
    whole-unit is not an approximation there but the stated behaviour.
    """
    with _backend(True) as backend:
        thcon = backend.backend_units["THCON"]
        assert thcon.cost_model is not None
        assert not thcon.cost_model.has_ipc_groups
        assert thcon.instruction_group("ATCAS") is None

        assert thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, 0)
        thcon.clock_tick(0)
        assert thcon.busy_groups == {None: 2}
        # Nothing at all gets in, from any thread, for the whole hold.
        for thread in range(3):
            assert not thcon.issueInstruction(ADDDMAREG_G2_EQ_G0_PLUS_1, thread)


def test_a_batch_is_held_for_the_longest_cost_in_it():
    """A unit with parallel paths is busy until the slowest of them is done, so
    the deadline comes from the whole batch rather than from whichever member
    happened to retire last."""
    with _backend(True) as backend:
        config = backend.backend_units["CFG"]
        config.cost_model = _ForcedOccupancy({"RDCFG": 4})
        assert config.issueInstruction(RDCFG_G5_FROM_CFG12, 0)
        assert config.issueInstruction(setc16_math_offset(0x200), 1)
        assert config.issueInstruction(setc16_math_offset(0x200), 2)
        config.clock_tick(0)
        # The 1-cycle SETC16s do not shorten the RDCFG's hold (4 cycles from
        # the batch's acceptance cycle, i.e. deadline 3)...
        assert config.busy_until == 3
        # ...and none of them was pushed past it.
        assert not config.next_instruction
        for thread in (1, 2):
            assert (
                backend.getThreadConfigValue(thread, "DEST_TARGET_REG_CFG_MATH_Offset")
                == 0x200
            )


def test_the_sync_unit_charges_one_cycle_throughout():
    """Every Sync Unit op is a single cycle in both arches' tables. What this
    unit costs a kernel is wait-gate time, which is not occupancy at all — the
    page is explicit that ``SEMWAIT`` execution "consists purely of passing
    them over to the Wait Gate" — and tt-sim already models that separately."""
    with _backend(True) as backend:
        sync = backend.backend_units["SYNC"]
        for name in sync.opcode_to_method_map:
            assert sync.instruction_occupancy(name, 0) == 1, name


# ---------------------------------------------------------------------------
# 6. The config unit's *other* column: residency, and STALLWAIT's C12.
# ---------------------------------------------------------------------------
#
# Everything above is about what an instruction *costs* — how soon the unit will
# take the next one. This section is about how long it is still *in* the unit,
# which on this unit is a different number:
#
#   ConfigurationUnit.md (BlackholeA0) tabulates SETC16 at latency 1 / IPC 3,
#   STREAMWRCFG at latency >= 5 / IPC 1, WRCFG at latency 2 / IPC 1,
#   CFGSHIFTMASK at latency 2 / IPC 1/2, RMWCIB at latency 1 / IPC 1, and RDCFG
#   at latency >= 2 / IPC 1. Wormhole's page prints the same latencies against
#   prose throughputs.
#
# So WRCFG is latency 2 at one instruction per cycle: the unit accepts the next
# instruction a cycle *before* the previous one has left. Nothing read the
# Latency column until now, and the consequence was that
#
#   STALLWAIT.md (BlackholeA0), condition C12: "Any thread has an instruction in
#   any stage of the Configuration Unit pipeline."
#
# was satisfied the moment the issue queue drained — and, worse, that whether it
# saw anything at all came down to whether the *issuing* thread's Wait Gate
# happened to tick before the *waiting* thread's within the cycle.
# ``test_off_means_c12_was_decided_by_wait_gate_tick_order`` measures exactly
# that, and it is the argument for modelling the residency as a deadline.
#
# Bounds are charged at their low end (``BOUND_POLICY``), so RDCFG's ">= 2
# cycles" arms 2. For a residency the low end is the *under*-reporting end and
# that is the safe one: a residency held longer than the hardware's makes a
# STALLWAIT on C12 wait for a stall that does not exist, and inventing
# back-pressure is the one direction the bounds policy forbids.

#: ``WRCFG CFG[12] = GPR[5]`` — latency 2, occupancy 1, i.e. the shape that
#: makes residency a separate question. ``CfgReg`` at bit 0, ``GprAddress`` at
#: 16; opcode 176.
WRCFG_CFG12_FROM_G5 = (176 << 24) | (5 << 16) | 12

#: ``STALLWAIT`` on condition **C12** (bit 12 — ``p_stall::CFGEXU`` in
#: tt-metal's Blackhole LLK header), blocking the Vector Unit (block bit B8,
#: ``stall_res`` at bit 15). Opcode 0xA2. Paired with an ``SFPNOP``, which B8
#: catches and which does nothing else, so the cycle the waiting thread's Wait
#: Gate FIFO drains is exactly the cycle C12 was met plus ``STALLWAIT.md``'s own
#: "one cycle lag between the condition(s) being met and the block mask being
#: removed".
STALLWAIT_C12_BLOCK_SFPU = (0xA2 << 24) | (0x100 << 15) | 0x1000


@contextmanager
def _coprocessor(blackhole=True, cost_model=True):
    """A whole Tensix coprocessor — Wait Gates included — model on or off."""
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if cost_model:
        os.environ["TT_SIM_COST_MODEL"] = "1"
    else:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    profile = BLACKHOLE_PROFILE if blackhole else WORMHOLE_PROFILE
    try:
        yield TensixCoProcessor(
            None,
            profile.tensix_cfg_state_size,
            profile.tensix_thd_state_size,
            blackhole,
        )
    finally:
        TensixConfigurationConstants.use_blackhole(False)
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def _c12_drain_cycle(
    coprocessor, config_burst, cycles=80, reverse_units=False, reverse_gates=False
):
    """Cycle at which the *waiting* thread's Wait Gate FIFO empties, or ``None``.

    Thread 0 issues ``config_burst`` back-to-back ``WRCFG``s; thread 1 issues a
    ``STALLWAIT`` on C12 and then the ``SFPNOP`` that wait blocks. That is the
    smallest program in which C12 means anything, because the condition names
    *any* thread's instruction — so the waiter has to be a different thread from
    the issuer, which is what makes C12 unlike every other condition in the
    table.

    ``reverse_units`` reverses the order the backend units tick within a cycle
    and ``reverse_gates`` the order the three Wait Gates do. Neither ordering is
    a fact about the hardware; both are artefacts of the list ``getClocks``
    returns.
    """
    backend = coprocessor.getBackend()
    issuer, waiter = coprocessor.getThread(0), coprocessor.getThread(1)
    for _ in range(config_burst):
        issuer.push_wait_gate_instruction(WRCFG_CFG12_FROM_G5)
    waiter.push_wait_gate_instruction(STALLWAIT_C12_BLOCK_SFPU)
    waiter.push_wait_gate_instruction(SFPNOP)

    units = backend.getClocks()
    clocks = list(reversed(units)) if reverse_units else list(units)
    threads = list(coprocessor.threads)
    if reverse_gates:
        threads = list(reversed(threads))
    for thread in threads:
        clocks = clocks + thread.getClocks()

    for cycle in range(cycles):
        for clock in clocks:
            clock.clock_tick(cycle)
        if not waiter.wait_gate_instruction_fifo:
            return cycle
    return None


def test_the_config_residency_is_the_latency_column_not_the_occupancy_column():
    """Both columns are in the table, and for this unit they differ.

    Reading ``busy_until`` as residency would report 1 cycle for ``WRCFG`` and
    ``RDCFG``, which is the throughput answer and not the pipeline one. This is
    the assertion that keeps the two apart — and it is read off the unit's own
    cost model, so it is the table talking, not a constant repeated here.
    """
    for blackhole in (False, True):
        with _coprocessor(blackhole=blackhole) as coprocessor:
            model = coprocessor.getBackend().backend_units["CFG"].cost_model
            assert (model.latency("WRCFG"), model.occupancy("WRCFG")) == (2, 1)
            assert (model.latency("RDCFG"), model.occupancy("RDCFG")) == (2, 1)
            assert (model.latency("SETC16"), model.occupancy("SETC16")) == (1, 1)
            for name in ("RMWCIB0", "RMWCIB1", "RMWCIB2", "RMWCIB3"):
                assert (model.latency(name), model.occupancy(name)) == (1, 1), name
            if blackhole:
                # The two Blackhole-only rows. CFGSHIFTMASK is the one opcode
                # whose columns agree ("requires two cycles in stage 0");
                # STREAMWRCFG's ">= 5" is the pipeline depth, -4 through 0, at
                # one instruction per cycle.
                assert model.latency("CFGSHIFTMASK") == 2
                assert model.occupancy("CFGSHIFTMASK") == 2
                assert model.latency("STREAMWRCFG") == 5
                assert model.occupancy("STREAMWRCFG") == 1


def test_a_wrcfg_is_reported_in_the_pipeline_for_its_documented_latency():
    """Accepted in cycle 0, retired in cycle 1, still in a stage until cycle 2.

    The retire is unmoved — ``handle_wrcfg`` runs in the tick it always ran in,
    so no config value lands anywhere new — and only the *report* extends. That
    separation is deliberate: delaying the write itself is the ordering bug this
    unit already found once, where an accepted config write was overtaken by its
    own thread's later instructions.
    """
    with _coprocessor() as coprocessor:
        config = coprocessor.getBackend().backend_units["CFG"]
        assert config.issueInstruction(WRCFG_CFG12_FROM_G5, 0)
        config.clock_tick(1)
        assert not config.next_instruction, "the write must still retire here"
        assert config.pipeline_exit_cycles == (2, 0, 0)
        assert config.hasInflightInstructionsFromThread(0)
        assert not config.is_clock_idle()
        config.clock_tick(2)
        assert not config.hasInflightInstructionsFromThread(0)
        assert config.pipeline_exit_cycles == (0, 0, 0)


def test_a_setc16_keeps_the_same_cycle_report_it_always_had():
    """Latency 1 arms nothing, by construction rather than by special case.

    The deadline lands on the retire tick itself, and the instruction was
    already visible through the issue queue for the cycle between acceptance and
    that tick — so one documented cycle is exactly what the unit reported before
    any of this existed.
    """
    with _coprocessor() as coprocessor:
        config = coprocessor.getBackend().backend_units["CFG"]
        assert config.issueInstruction(setc16_math_offset(0x200), 0)
        assert config.hasInflightInstructionsFromThread(0), "queued, not yet retired"
        config.clock_tick(1)
        assert config.pipeline_exit_cycles == (0, 0, 0)
        assert not config.hasInflightInstructionsFromThread(0)


def test_the_config_residency_is_the_issuing_threads_alone():
    """C12 ORs over the three threads at the Wait Gate; the unit answers per
    thread, and must not smear one thread's instruction over the others."""
    with _coprocessor() as coprocessor:
        config = coprocessor.getBackend().backend_units["CFG"]
        assert config.issueInstruction(RDCFG_G5_FROM_CFG12, 2)
        config.clock_tick(1)
        assert config.pipeline_exit_cycles == (0, 0, 2)
        assert config.hasInflightInstructionsFromThread(2)
        assert not config.hasInflightInstructionsFromThread(0)
        assert not config.hasInflightInstructionsFromThread(1)


def test_off_means_the_config_unit_reports_only_its_issue_queue():
    """The invariance half, at the unit: with no cost model ``_arm_residency``
    never runs and the predicate is the base class's issue-queue scan, which is
    what every replay guard was recorded against."""
    with _coprocessor(cost_model=False) as coprocessor:
        config = coprocessor.getBackend().backend_units["CFG"]
        assert config.cost_model is None
        assert config.issueInstruction(WRCFG_CFG12_FROM_G5, 0)
        config.clock_tick(1)
        assert config.pipeline_exit_cycles == (0, 0, 0)
        assert not config.hasInflightInstructionsFromThread(0)


def test_stallwait_on_c12_waits_for_another_threads_config_burst():
    """C12 through the instruction that consumes it, now that it is reachable.

    Six ``WRCFG``s are accepted one per cycle (0-5); the last is in a stage until
    cycle 7, the Wait Gate sees C12 met there, and ``STALLWAIT.md``'s documented
    one-cycle lag puts the ``SFPNOP`` at cycle 8.
    """
    with _coprocessor() as coprocessor:
        assert _c12_drain_cycle(coprocessor, config_burst=6) == 8


def test_the_c12_wait_is_a_deadline_and_not_a_tick_order():
    """Perturbation: reverse the backend units, reverse the Wait Gates, or both.

    The cycle a documented condition clears at must not depend on either
    ordering. It does not — the answer is the burst length plus the trailing
    pipeline cycle plus the documented lag, in all four orderings and at three
    burst lengths.
    """
    for burst in (3, 6, 10):
        for reverse_units in (False, True):
            for reverse_gates in (False, True):
                with _coprocessor() as coprocessor:
                    drained = _c12_drain_cycle(
                        coprocessor,
                        config_burst=burst,
                        reverse_units=reverse_units,
                        reverse_gates=reverse_gates,
                    )
                assert drained == burst + 2, (burst, reverse_units, reverse_gates)


def test_off_means_c12_was_decided_by_wait_gate_tick_order():
    """What it was before, measured, because it is the argument for the change.

    With no residency the only thing that can make C12 unmet is the issue queue,
    which the unit drains at the top of its own tick. So the condition sees
    anything only if the *issuing* thread's Wait Gate happened to run before the
    *waiting* thread's in the same cycle. In the order ``getClocks`` returns it
    does, and the wait looks nearly right; reverse the two gates and it collapses
    to the same three cycles however many ``WRCFG``s are in flight — ten of them
    as invisible as none.

    Not "C12 waits too little", then, but "C12's answer is not a property of the
    machine". This test is kept as the record of that, and it is also the
    flag-off invariance check at the Wait Gate: with the model off, nothing here
    moved.
    """
    for burst in (3, 6, 10):
        with _coprocessor(cost_model=False) as coprocessor:
            in_order = _c12_drain_cycle(coprocessor, config_burst=burst)
        with _coprocessor(cost_model=False) as coprocessor:
            gates_reversed = _c12_drain_cycle(
                coprocessor, config_burst=burst, reverse_gates=True
            )
        assert in_order == burst + 1, burst
        assert gates_reversed == 3, burst


# ---------------------------------------------------------------------------
# 7. The same question for the Matrix Unit (C7) and the SFPU (C14).
# ---------------------------------------------------------------------------
#
# STALLWAIT.md (WormholeB0) states two more conditions the same way, and both
# name a unit whose Latency column is longer than its throughput column:
#
#   C7  "The current thread has an instruction in any stage of the Matrix Unit
#        (FPU) pipeline."
#   C14 "The current thread has an instruction in any stage of the Vector Unit
#        (SFPU) pipeline."
#
# (Blackhole numbers the same two conditions C4 and C11.) Unlike C12 these are
# scoped to the *issuing* thread, so the smallest program that means anything is
# one thread's burst followed by its own STALLWAIT.
#
# MatrixUnit.md's "Instruction latency and throughput" table:
#
#   MVMUL / DOTPV / GAPOOL / ELWMUL       IPC 1     latency 5
#   GMPOOL / ELWADD / ELWSUB              IPC 1     latency 5
#   SETRWC / INCRWC / CLEARDVALID /
#     CLREXPHIST / GATESRCRST             IPC 1     latency 1
#   SHIFTXA / ZEROACC / ZEROSRC /
#     TRNSPSRCB                           IPC 1     latency 1
#   SHIFTXB                               IPC 0.5   latency 2
#   MOVD2A                                IPC 1     latency 2
#   MOVA2D / MOVDBGA2D / MOVB2D / MOVB2A  IPC 1     latency 4
#
# VectorUnit.md's per-opcode tables give IPC 1 throughout and latency 2 for the
# arithmetic and LUT rows (SFPADD, SFPADDI, SFPMAD, SFPMUL, SFPMULI, SFPLUT,
# SFPLUTFP32, SFPSWAP; SFPCONFIG and one SFPSHFT2 form are "<= 2 cycles"),
# latency 1 for everything else and "Complex" for SFPLOADMACRO.
#
# So the Matrix Unit is a *five*-deep pipeline running at one instruction per
# cycle, which is the widest gap between the two columns anywhere in the tree,
# and the SFPU's is two. tt-metal reaches both conditions constantly:
# ``p_stall::MATH`` and ``p_stall::WAIT_SFPU`` are issued together by
# ``_llk_math_dest_section_done_`` (every tile of every compute kernel) and
# ``p_stall::MATH`` alone by ``_llk_math_eltwise_sfpu_start_``, which is how an
# SFPU kernel waits for the FPU to drain before it starts.

#: ``MOVB2A`` — SrcB to SrcA, latency 4 and occupancy 1, i.e. the shape that
#: makes residency a separate question, and the *longest* published latency
#: reachable through a Wait Gate without first arranging valid Src data (MVMUL
#: and friends are held at the gate until their Src banks are dvalid; this one
#: only needs the bank's ``AllowedClient``, which ``_residency_drain_cycle``
#: hands over). Opcode 0x0B, all fields zero.
MOVB2A = 0x0B << 24
#: ``MOVD2A`` — Dst to SrcA, latency 2. Not gated on ``AllowedClient`` at all.
MOVD2A = 0x08 << 24
#: ``ZEROACC`` — latency 1, the control: an op whose residency must not extend.
ZEROACC = 0x10 << 24
#: ``MVMUL``, all fields zero. Latency 5, and the reason any of this matters.
MVMUL = 0x26 << 24
#: ``SFPADDI`` — latency 2, occupancy 1. ``SFPNOP`` (latency 1) is the control.
SFPADDI = 0x75 << 24


def _stallwait(condition_mask, block_mask):
    """``STALLWAIT`` with an explicit condition and block mask. Opcode 0xA2."""
    return (0xA2 << 24) | (block_mask << 15) | condition_mask


def _residency_drain_cycle(
    coprocessor,
    unit,
    op,
    burst,
    cycles=90,
    reverse_units=False,
    reverse_gates=False,
):
    """Cycle at which the thread's own Wait Gate FIFO empties, or ``None``.

    Thread 0 issues ``burst`` back-to-back copies of ``op``, then a
    ``STALLWAIT`` whose condition is that unit's residency bit and whose block
    mask catches the single instruction after it. The blocked instruction is
    chosen from the *other* unit so that the block mask cannot itself be what
    holds it: the Matrix Unit's wait blocks an ``SFPNOP`` with B8, the SFPU's
    blocks a ``ZEROACC`` with B6.

    The condition bit is the architecture's, not a shared constant: Wormhole
    numbers these C7 and C14, Blackhole C4 and C11.
    """
    backend = coprocessor.getBackend()
    # Both Src banks handed to the Matrix Unit, which is where a kernel's
    # unpack + ``SETDVALID`` leaves them and what ``MOVB2A`` waits for at the
    # gate. Nothing here reads the data; the banks are zero either way.
    for bank in (0, 1):
        backend.getSrcA(bank).allowedClient = SrcRegister.SrcClient.MatrixUnit
        backend.getSrcB(bank).allowedClient = SrcRegister.SrcClient.MatrixUnit
    blackhole = backend.blackhole
    if unit == "MATH":
        condition = 0x10 if blackhole else 0x80
        block, blocked = 0x100, SFPNOP
    else:
        condition = 0x800 if blackhole else 0x4000
        block, blocked = 0x40, ZEROACC

    thread = coprocessor.getThread(0)
    for _ in range(burst):
        thread.push_wait_gate_instruction(op)
    thread.push_wait_gate_instruction(_stallwait(condition, block))
    thread.push_wait_gate_instruction(blocked)

    units = backend.getClocks()
    clocks = list(reversed(units)) if reverse_units else list(units)
    threads = list(coprocessor.threads)
    if reverse_gates:
        threads = list(reversed(threads))
    for each in threads:
        clocks = clocks + each.getClocks()

    for cycle in range(cycles):
        for clock in clocks:
            clock.clock_tick(cycle)
        if not thread.wait_gate_instruction_fifo:
            return cycle
    return None


def test_the_matrix_and_sfpu_residencies_are_the_latency_column():
    """The two columns, read off the units' own cost models.

    This is the table talking rather than a constant repeated here, and it is
    the assertion that keeps the columns apart: every one of these opcodes has
    occupancy 1, so anything reading ``busy_until`` as residency would report
    one cycle for all of them.
    """
    for blackhole in (False, True):
        with _coprocessor(blackhole=blackhole) as coprocessor:
            backend = coprocessor.getBackend()
            math = backend.backend_units["MATH"].cost_model
            sfpu = backend.backend_units["SFPU"].cost_model
            for name in ("MVMUL", "DOTPV", "GAPOOL", "GMPOOL", "ELWADD", "ELWSUB"):
                assert (math.latency(name), math.occupancy(name)) == (5, 1), name
            for name in ("MOVA2D", "MOVB2D", "MOVB2A", "MOVDBGA2D"):
                assert (math.latency(name), math.occupancy(name)) == (4, 1), name
            assert (math.latency("MOVD2A"), math.occupancy("MOVD2A")) == (2, 1)
            for name in ("SETRWC", "INCRWC", "ZEROACC", "ZEROSRC", "CLEARDVALID"):
                assert (math.latency(name), math.occupancy(name)) == (1, 1), name
            # ``unknown`` in the table: no latency, so no residency opinion.
            assert math.latency("MOVD2B") is None
            for name in ("SFPADD", "SFPADDI", "SFPMAD", "SFPMUL", "SFPLUT", "SFPSWAP"):
                assert (sfpu.latency(name), sfpu.occupancy(name)) == (2, 1), name
            for name in ("SFPNOP", "SFPMOV", "SFPLOAD", "SFPSTORE", "SFPIADD"):
                assert (sfpu.latency(name), sfpu.occupancy(name)) == (1, 1), name
            # "<= 2 cycles", charged at the low end like every other bound.
            assert sfpu.latency("SFPCONFIG") == 2
            # "Complex" — the one SFPU row with no number to charge.
            assert sfpu.latency("SFPLOADMACRO") is None


def test_an_mvmul_is_reported_in_the_pipeline_for_its_documented_latency():
    """Five cycles from acceptance, at the unit, for the op that matters most.

    Offered straight to the unit rather than through a Wait Gate, because
    ``MVMUL`` is held at the gate until its Src banks are dvalid and this is
    about the residency rather than about that interlock. The retire is
    unmoved: the handler still runs in the tick it always ran in.
    """
    with _coprocessor() as coprocessor:
        matrix = coprocessor.getBackend().backend_units["MATH"]
        assert matrix.issueInstruction(MVMUL, 0)
        matrix.clock_tick(1)
        assert not matrix.next_instruction, "the MVMUL must still retire here"
        # Accepted in cycle 0, so it leaves in cycle 0 + 5.
        assert matrix.pipeline_exit_cycles == (5, 0, 0)
        assert matrix.hasInflightInstructionsFromThread(0)
        assert not matrix.is_clock_idle()
        for cycle in (2, 3, 4):
            matrix.clock_tick(cycle)
            assert matrix.hasInflightInstructionsFromThread(0), cycle
        matrix.clock_tick(5)
        assert matrix.pipeline_exit_cycles == (0, 0, 0)
        assert not matrix.hasInflightInstructionsFromThread(0)
        assert matrix.is_clock_idle()


def test_the_matrix_residency_is_the_issuing_threads_alone():
    """C7 is "the *current* thread", unlike C12's "any thread"."""
    with _coprocessor() as coprocessor:
        matrix = coprocessor.getBackend().backend_units["MATH"]
        assert matrix.issueInstruction(MVMUL, 1)
        matrix.clock_tick(1)
        assert matrix.pipeline_exit_cycles == (0, 5, 0)
        assert matrix.hasInflightInstructionsFromThread(1)
        assert not matrix.hasInflightInstructionsFromThread(0)
        assert not matrix.hasInflightInstructionsFromThread(2)


def test_a_one_cycle_matrix_op_keeps_the_report_it_always_had():
    """Latency 1 arms nothing, by construction rather than by special case."""
    with _coprocessor() as coprocessor:
        matrix = coprocessor.getBackend().backend_units["MATH"]
        assert matrix.issueInstruction(ZEROACC, 0)
        assert matrix.hasInflightInstructionsFromThread(0), "queued, not yet retired"
        matrix.clock_tick(1)
        assert matrix.pipeline_exit_cycles == (0, 0, 0)
        assert not matrix.hasInflightInstructionsFromThread(0)


def test_an_sfpu_op_is_reported_in_the_pipeline_for_two_cycles():
    """The SFPU half of the same mechanism, at the unit.

    Held for the documented 2 cycles even though the Wait Gate below cannot see
    it — the mechanism is the unit's, and what a *particular* consumer can
    observe is a separate question. ``SFPNOP`` is the control at latency 1.
    """
    with _coprocessor() as coprocessor:
        sfpu = coprocessor.getBackend().backend_units["SFPU"]
        assert sfpu.issueInstruction(SFPADDI, 0)
        sfpu.clock_tick(1)
        assert sfpu.pipeline_exit_cycles == (2, 0, 0)
        assert sfpu.hasInflightInstructionsFromThread(0)
        sfpu.clock_tick(2)
        assert sfpu.pipeline_exit_cycles == (0, 0, 0)
        assert not sfpu.hasInflightInstructionsFromThread(0)
    with _coprocessor() as coprocessor:
        sfpu = coprocessor.getBackend().backend_units["SFPU"]
        assert sfpu.issueInstruction(SFPNOP, 0)
        sfpu.clock_tick(1)
        assert sfpu.pipeline_exit_cycles == (0, 0, 0)


def test_off_means_the_matrix_and_sfpu_units_report_only_their_issue_queues():
    """The invariance half, at the units: no cost model, no residency."""
    with _coprocessor(cost_model=False) as coprocessor:
        backend = coprocessor.getBackend()
        for key, op in (("MATH", MVMUL), ("SFPU", SFPADDI)):
            unit = backend.backend_units[key]
            assert unit.cost_model is None, key
            assert unit.issueInstruction(op, 0), key
            unit.clock_tick(1)
            assert unit.pipeline_exit_cycles == (0, 0, 0), key
            assert not unit.hasInflightInstructionsFromThread(0), key
            assert unit.is_clock_idle(), key


def test_stallwait_on_c7_waits_for_the_matrix_pipeline_to_empty():
    """C7 through the instruction that consumes it.

    A ``burst``-long run of ``MOVB2A`` is accepted one per cycle (0 through
    ``burst - 1``); the last is in a stage until cycle ``burst + 3``, the Wait
    Gate sees C7 met there, and ``STALLWAIT.md``'s documented one-cycle lag puts
    the ``SFPNOP`` at ``burst + 4``.
    """
    for blackhole in (False, True):
        with _coprocessor(blackhole=blackhole) as coprocessor:
            assert _residency_drain_cycle(coprocessor, "MATH", MOVB2A, burst=6) == 10, (
                blackhole
            )


def test_the_c7_wait_is_a_deadline_and_not_a_tick_order():
    """Perturbation: reverse the backend units, reverse the Wait Gates, or both.

    The cycle a documented condition clears at must not depend on either
    ordering, and it must be a function of the *instruction*. Both hold: the
    answer is ``burst + 4`` for a latency-4 op in all four orderings, on both
    architectures, at four burst lengths — where the same sweep with the model
    off gives ``burst + 3`` whatever the op is.
    """
    for blackhole in (False, True):
        for burst in (1, 3, 6, 10):
            for reverse_units in (False, True):
                for reverse_gates in (False, True):
                    with _coprocessor(blackhole=blackhole) as coprocessor:
                        drained = _residency_drain_cycle(
                            coprocessor,
                            "MATH",
                            MOVB2A,
                            burst=burst,
                            reverse_units=reverse_units,
                            reverse_gates=reverse_gates,
                        )
                    assert drained == burst + 4, (
                        blackhole,
                        burst,
                        reverse_units,
                        reverse_gates,
                    )


def test_off_means_c7_could_not_tell_a_four_cycle_op_from_a_one_cycle_one():
    """What it was before, measured, because it is the argument for the change.

    With no residency the only thing that can make C7 unmet is the issue queue,
    which the unit drains at the top of its own tick — so a ``ZEROACC`` (latency
    1), a ``MOVD2A`` (2) and a ``MOVB2A`` (4) all clear the condition at exactly
    the same cycle. Not "C7 waits too little", then, but "C7's answer is not a
    property of the instruction".

    This is also the flag-off invariance check at the Wait Gate: with the model
    off, every one of these numbers is what it was before the residency existed.
    """
    for burst in (1, 3, 6, 10):
        for op in (ZEROACC, MOVD2A, MOVB2A):
            with _coprocessor(cost_model=False) as coprocessor:
                drained = _residency_drain_cycle(coprocessor, "MATH", op, burst=burst)
            assert drained == burst + 3, (burst, hex(op))


def test_c14_is_inert_at_the_gate_because_the_sfpu_is_only_two_deep():
    """The SFPU half, and the honest answer for it: no number moves.

    The Wait Gate costs three cycles between the burst's last acceptance and the
    blocked instruction — latching the ``STALLWAIT``, evaluating it, and
    ``STALLWAIT.md``'s one-cycle lag — so a residency has to reach past cycle 3
    to be visible here at all. The SFPU's deepest published latency is 2, so it
    never does: a 2-cycle ``SFPADDI`` and a 1-cycle ``SFPNOP`` drain at the same
    cycle, with the model on and with it off.

    That is a property of the *table*, not of this test, and it is why C14 needs
    no separate treatment: the mechanism is in place and correct at the unit
    (``test_an_sfpu_op_is_reported_in_the_pipeline_for_two_cycles``), and a
    future SFPU row deeper than 3 cycles would start to show here without
    anything being rewritten.
    """
    for cost_model in (True, False):
        for burst in (1, 3, 6, 10):
            for op in (SFPNOP, SFPADDI):
                with _coprocessor(cost_model=cost_model) as coprocessor:
                    drained = _residency_drain_cycle(
                        coprocessor, "SFPU", op, burst=burst
                    )
                assert drained == burst + 3, (cost_model, burst, hex(op))


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "backend_cost_model_test OK: five more units wired, SFPU at one cycle "
        "per op, ThCon multi-cycle and back-pressuring, the config unit's "
        "residency and C12, the Matrix Unit's and SFPU's and C7/C14, off by "
        "default"
    )


if __name__ == "__main__":
    main()
