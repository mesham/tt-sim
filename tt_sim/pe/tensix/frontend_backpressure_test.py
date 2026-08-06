"""Tensix front-end back-pressure: a full FIFO stalls the issuing baby core.

ROADMAP item 1. ``TensixFrontend.push_mop_instruction`` used to be an unbounded
list append and a ``.ttinsn`` store returned the same tick, so nothing ever
back-pressured the issuing baby RISC-V core: a permanently blocked backend
instruction could not wedge the core (the ``UNPACR_NOP``
acquire-without-release case ran to completion where Blackhole silicon
deadlocks), and no cost-table occupancy above one cycle could ever reach a
device-side cycle count (``perfbench/tensixbench`` read a forced 1.000 for
every probe of every unit).

The mechanism under test: a core-facing push into a thread's frontend is
refused (``MemoryStall``) once that thread has
:data:`~tt_sim.pe.tensix.frontend.CORE_PUSH_INFLIGHT_BOUND` instructions in
flight, and both push paths — the ``.ttinsn`` extension and a plain ``sw`` to
the push buffer — turn the refusal into a ``PEStall`` so the core retries with
the PC unmoved. The bound is a **conservative uncalibrated mechanism
parameter** (see its comment), not a silicon calibration.

With ``TT_SIM_COST_MODEL`` set, the two licensed terms become observable at
the core exactly as silicon measures them
(``docs/plans/riscv-front-end-benchmark.md``):

* the ``.ttinsn`` push itself costs one cycle per core, and no more;
* a backend unit back-pressures the issuing core at its documented occupancy —
  a ThCon ``ADDDMAREG`` burst approaches 3 cycles/instruction at the core, and
  three threads sharing a 1-IPC unit approach 3x each (which also needs the
  grant rotation in ``TensixBackendUnit.issueInstruction``: without it the
  first wait gate in tick order would win the slot every cycle and starve the
  other cores for ever).

Run standalone (``python3 -m tt_sim.pe.tensix.frontend_backpressure_test``) or
under pytest.
"""

import os
from contextlib import contextmanager

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.memory.memory import MemoryStall
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.rv.isa.tt_isa import RV_TT_ISA
from tt_sim.pe.tensix.frontend import CORE_PUSH_INFLIGHT_BOUND
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixConfigurationConstants
from tt_sim.util.conversion import conv_to_bytes

#: ``TTI_NOP`` — backend resource NONE, so it never queues in a backend unit.
NOP = 0x02 << 24

#: ``ADDDMAREG GPR[2] = GPR[0] + 1`` (OpBisConst) — the ThCon op both silicon
#: benchmarks measured at ~2.97 cycles, table occupancy 3 (range 3-4, charged
#: at the low end). Same encoding as ``backend_cost_model_test``.
ADDDMAREG = (88 << 24) | (1 << 23) | (2 << 12) | (1 << 6) | 0

#: ``SFPNOP`` — a 1-cycle op on the shared SFPU, for the 3-thread scaling case.
SFPNOP = 143 << 24

#: ``UNPACR_NOP`` with ``set_dvalid`` in its Blackhole ZEROSRC form — the
#: acquire half of the unpacker handshake (see ``setdvalid_srcrow_test``).
UNPACR_NOP_SETDVALID_BH = (0x43 << 24) | (0 << 23) | (1 << 8) | 1


@contextmanager
def _coprocessor(cost_model=False, blackhole=False):
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
            blackhole=blackhole,
        )
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous
        if blackhole:
            TensixConfigurationConstants.use_blackhole(False)


def _push(frontend, word):
    """One core-facing push, as the memory dispatch delivers it."""
    return frontend.write(0, conv_to_bytes(word))


def _run(cp, cycles, pushes, start_cycle=0):
    """Tick the whole coprocessor for ``cycles``, attempting one core push per
    thread per cycle from ``pushes`` (``{thread_id: word}``). Returns the count
    of *accepted* pushes per thread — what an issuing core would have retired,
    since a refused push stalls the core on the same instruction."""
    clocks = cp.getClocks()
    accepted = dict.fromkeys(pushes, 0)
    for cycle in range(start_cycle, start_cycle + cycles):
        for thread_id, word in pushes.items():
            if _push(cp.getThread(thread_id), word) is not MemoryStall:
                accepted[thread_id] += 1
        for clock in clocks:
            clock.clock_tick(cycle)
    return accepted


# ---------------------------------------------------------------------------
# The bound itself.
# ---------------------------------------------------------------------------


def test_core_push_is_refused_at_the_inflight_bound():
    with _coprocessor() as cp:
        frontend = cp.getThread(0)
        for _ in range(CORE_PUSH_INFLIGHT_BOUND):
            assert _push(frontend, NOP) is not MemoryStall
        # No clock has ticked, so nothing drained: the bound is reached and
        # the next core push must refuse rather than append.
        assert frontend.inflight_count() == CORE_PUSH_INFLIGHT_BOUND
        assert _push(frontend, NOP) is MemoryStall
        assert frontend.inflight_count() == CORE_PUSH_INFLIGHT_BOUND


def test_internal_expansion_pushes_are_never_refused():
    """Only the core-facing write is bounded. tt-sim's MOP/replay expanders
    emit a whole template in one tick, so their output legitimately exceeds
    the bound; what that inflated count then does is refuse the *next core
    push*, which is the honest proxy for the expansion work in flight."""
    with _coprocessor() as cp:
        frontend = cp.getThread(0)
        for _ in range(2 * CORE_PUSH_INFLIGHT_BOUND):
            frontend.push_replay_instruction(NOP)
        assert frontend.inflight_count() == 2 * CORE_PUSH_INFLIGHT_BOUND
        assert _push(frontend, NOP) is MemoryStall


def test_below_the_bound_pushes_drain_at_one_per_cycle_as_before():
    """Model off, path clear: the status quo. Every push accepted, every
    instruction drained — the bound is unobservable on a healthy stream."""
    with _coprocessor() as cp:
        accepted = _run(cp, 200, {0: NOP})
        assert accepted[0] == 200
        assert cp.getThread(0).inflight_count() <= 4


# ---------------------------------------------------------------------------
# The stalled core, at the ISA level.
# ---------------------------------------------------------------------------


class _FullFrontendMemory:
    """Stands in for the core's memory space: the push buffer is full."""

    def __init__(self):
        self.writes = 0

    def write(self, addr, value, size=None):
        self.writes += 1
        return MemoryStall


class _AcceptingMemory:
    def write(self, addr, value, size=None):
        return None


def test_a_ttinsn_store_into_a_full_fifo_stalls_the_core():
    #: ``TTI_NOP`` as the .ttinsn encoding: the constant rotated *left* two.
    ttinsn = ((NOP << 2) | (NOP >> 30)) & 0xFFFFFFFF
    assert ttinsn & 0x3 != 0x3
    memory = _FullFrontendMemory()
    result = RV_TT_ISA.run(None, memory, False, ttinsn)
    # PEStall: the caller must not advance the PC, so the same .ttinsn is
    # retried next cycle — the core is stalled, exactly as hardware stalls on
    # a store to a full instruction push buffer.
    assert result is ProcessingElement.PEStall
    assert memory.writes == 1
    assert RV_TT_ISA.run(None, _AcceptingMemory(), False, ttinsn) is True


# ---------------------------------------------------------------------------
# The correctness half: a permanently blocked backend wedges the core.
# ---------------------------------------------------------------------------


def test_second_dvalid_acquire_with_no_release_blocks_the_issuing_core():
    """The ROADMAP-named test: issue the dvalid setup twice with no intervening
    clear, and the second must block the core.

    ``setdvalid_srcrow_test.test_unpacr_nop_setdvalid_twice_deadlocks_the_unpacker``
    pins the single-unit half (the second acquire waits for ever). This is the
    other half: the blocked unpacker holds its instruction, the thread's wait
    gate cannot issue past it, the frontend FIFO backs up, and within the
    bound's worth of further pushes the core's next push is refused — where it
    previously issued for ever and the kernel ran to completion behind a
    wedged unit.
    """
    with _coprocessor(blackhole=True) as cp:
        frontend = cp.getThread(1)
        clocks = cp.getClocks()
        cycle = 0

        def tick():
            nonlocal cycle
            for clock in clocks:
                clock.clock_tick(cycle)
            cycle += 1

        # First acquire: finds the bank free, hands it to the Matrix Unit.
        assert _push(frontend, UNPACR_NOP_SETDVALID_BH) is not MemoryStall
        for _ in range(4):
            tick()
        unpacker = cp.getBackend().unpacker_units[0]
        assert not unpacker.blocked

        # Second acquire, nothing having cleared dvalid: blocks for ever.
        assert _push(frontend, UNPACR_NOP_SETDVALID_BH) is not MemoryStall
        for _ in range(4):
            tick()
        assert unpacker.blocked

        # The thread keeps issuing unpacker work — the next launch's unpack
        # stream, in the silicon reproduction. The blocked unpacker refuses
        # every further instruction, the wait gate's head sticks, the FIFO
        # backs up, and within the bound's worth of pushes the core's next
        # push is refused — for ever, which is what hands the report to the
        # deadlock watchdog, the silicon-matching behaviour. (An instruction
        # for a *different* unit still dispatches past the blocked one, so a
        # kernel with no further unpacker work can still finish; that case is
        # the terminal ``[UNIT WEDGED]`` raise's.)
        stalled_at = None
        for n in range(2 * CORE_PUSH_INFLIGHT_BOUND):
            if _push(frontend, UNPACR_NOP_SETDVALID_BH) is MemoryStall:
                stalled_at = n
                break
            tick()
        assert stalled_at is not None, "the core was never back-pressured"
        assert stalled_at <= CORE_PUSH_INFLIGHT_BOUND
        # And it stays stalled: nothing can ever release the bank.
        for _ in range(50):
            tick()
        assert _push(frontend, UNPACR_NOP_SETDVALID_BH) is MemoryStall
        assert unpacker.blocked


# ---------------------------------------------------------------------------
# The cost-model half (TT_SIM_COST_MODEL=1): the two licensed terms.
# ---------------------------------------------------------------------------


def test_a_ttinsn_push_costs_one_cycle_at_the_core_with_the_model_on():
    """Term (i): the push itself is 1 cycle per core, and no more.

    Silicon: ``riscvbench`` phase T measures ``tt_nop`` at 0.996-1.029
    cycles/instruction at one, two and three issuing threads — the push path
    is per-core. A NOP's backend resource is NONE, so nothing downstream can
    refuse: every push must be accepted, at one per cycle.
    """
    with _coprocessor(cost_model=True) as cp:
        accepted = _run(cp, 300, {0: NOP})
        assert accepted[0] == 300


def test_a_thcon_burst_approaches_three_cycles_per_instruction_at_the_core():
    """Term (ii): a unit back-pressures the core at its documented occupancy.

    Silicon: ``ADDDMAREG`` at 2.97 cycles/instruction from one thread (both
    benchmarks). With the model on, ThCon holds itself for the table's 3
    cycles, the wait gate retries, the FIFO fills to the bound, and from then
    on the core can only push at the drain rate: 3 cycles per instruction.
    """
    with _coprocessor(cost_model=True) as cp:
        # Warm-up: fill the FIFO to the bound (net +2/3 per cycle).
        _run(cp, 200, {0: ADDDMAREG})
        accepted = _run(cp, 300, {0: ADDDMAREG}, start_cycle=200)
        rate = 300 / accepted[0]
        assert 2.9 <= rate <= 3.1, f"steady-state cost {rate:.2f} cyc/instr"


def test_thcon_burst_stays_at_one_cycle_per_instruction_with_the_model_off():
    """The A/B control: with the model off nothing is occupied, the FIFO
    drains at one per cycle, and the bound never bites."""
    with _coprocessor(cost_model=False) as cp:
        _run(cp, 200, {0: ADDDMAREG})
        accepted = _run(cp, 300, {0: ADDDMAREG}, start_cycle=200)
        assert accepted[0] == 300


def test_three_threads_sharing_a_one_ipc_unit_approach_three_x_each():
    """Silicon: ``tt_sfpnop`` scales 0.998 / 1.968 / 2.973 across one, two,
    three issuing threads — one shared unit accepting one instruction per
    cycle. Three cores each pushing SFPNOP must each converge on ~3
    cycles/instruction, which requires the unit's grant rotation: with
    fixed-priority acceptance the first thread would read 1.0 and the other
    two would starve for ever."""
    with _coprocessor(cost_model=True) as cp:
        pushes = {0: SFPNOP, 1: SFPNOP, 2: SFPNOP}
        _run(cp, 250, pushes)
        accepted = _run(cp, 300, pushes, start_cycle=250)
        for thread_id, n in accepted.items():
            rate = 300 / n
            assert 2.8 <= rate <= 3.2, (
                f"thread {thread_id}: {rate:.2f} cyc/instr (accepted {n}/300)"
            )


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
