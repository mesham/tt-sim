"""The matrix unit driven with and without the modelled cycle costs.

Runs standalone (``python3 -m tt_sim.pe.tensix.matrix_cost_model_test``) or
under pytest. This is the unit half of Phase 5 of
``docs/plans/event-driven-pump.md``: ``tt_sim/perf/model_test.py`` pins what
the tables *say*, and this pins what the FPU *does* with what they say.

Three claims:

1. With ``TT_SIM_COST_MODEL`` unset the unit has no cost model at all, never
   arms ``busy_until``, and retires one instruction per cycle exactly as it
   always has. That is the property the replay guards depend on.
2. With it set, ``MVMUL`` is charged one cycle at every one of the four
   fidelity phases (see
   :meth:`tt_sim.perf.model.UnitCostModel.fidelity_occupancy` for why that is
   the right answer rather than a missing multiplier).
3. A cost above one cycle really does occupy the unit: the next instruction
   waits for the deadline, and ``hasInflightInstructionsFromThread`` — which
   is what the wait gates already query — reports the unit busy in between.
   No opcode in the *matrix* unit's table costs more than two cycles and none
   of the two-cycle ones (``SHIFTXB``) is modelled by this backend, so the
   multi-cycle case is driven with a stub table rather than a real op.
"""

import os
from contextlib import contextmanager

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor

#: INCRWC with ``rwc_a = 1``: the cheapest matrix-unit op with a visible
#: side effect, so a test can tell a retired instruction from a stalled one.
INCRWC_A1 = (0x38 << 24) | (1 << 6)


@contextmanager
def _cost_model(enabled):
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if enabled:
        os.environ["TT_SIM_COST_MODEL"] = "1"
    else:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    try:
        yield _matrix_unit()
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


def _matrix_unit():
    backend = TensixCoProcessor(
        None,
        WORMHOLE_PROFILE.tensix_cfg_state_size,
        WORMHOLE_PROFILE.tensix_thd_state_size,
    ).getBackend()
    return backend.getMatrixUnit()


class _FixedCost:
    """A stand-in cost table that charges every op the same number of cycles."""

    def __init__(self, cycles):
        self.cycles = cycles

    def scales_with_fidelity(self, name):
        return False

    def occupancy(self, name):
        return self.cycles

    #: The Matrix Unit publishes no "IPC group" column — its table has
    #: Throughput and Latency and nothing else — so its occupancy is a
    #: whole-unit hold and the stand-in says so explicitly rather than by
    #: omission. See ``UnitCostModel.ipc_group``.
    has_ipc_groups = False

    def ipc_group(self, name):
        return None


# ---------------------------------------------------------------------------
# 1. Off by default.
# ---------------------------------------------------------------------------


def test_without_the_env_var_the_unit_has_no_cost_model():
    with _cost_model(False) as matrix:
        assert matrix.cost_model is None
        assert matrix.instruction_occupancy("MVMUL", 0) is None


def test_off_means_one_instruction_retires_per_cycle_as_before():
    with _cost_model(False) as matrix:
        rwc = matrix.backend.getRWC(0)
        for cycle in range(3):
            assert matrix.issueInstruction(INCRWC_A1, 0)
            matrix.clock_tick(cycle)
            assert matrix.busy_until is None
            assert rwc.SrcA == cycle + 1
            assert not matrix.hasInflightInstructionsFromThread(0)


# ---------------------------------------------------------------------------
# 2. On, with the real table.
# ---------------------------------------------------------------------------


def test_on_the_fpu_charges_one_cycle_per_mvmul_at_every_fidelity_phase():
    with _cost_model(True) as matrix:
        assert matrix.cost_model is not None
        rwc = matrix.backend.getRWC(0)
        for phase in range(4):
            rwc.FidelityPhase = phase
            assert matrix.instruction_occupancy("MVMUL", 0) == 1, phase


def test_on_the_table_still_retires_one_instruction_per_cycle():
    """Every matrix op the table costs is a one-cycle occupancy, so switching
    the model on must not move a single cycle of an FPU-bound run. Measured on
    the ``six`` replay (a 128^3 bf16 matmul at HiFi4): 27,500 simulated cycles
    with the model on and off alike, and the matmul PCC unmoved."""
    with _cost_model(True) as matrix:
        rwc = matrix.backend.getRWC(0)
        for cycle in range(3):
            assert matrix.issueInstruction(INCRWC_A1, 0)
            matrix.clock_tick(cycle)
            assert matrix.busy_until is None
            assert rwc.SrcA == cycle + 1


def test_an_uncosted_op_leaves_the_timing_alone():
    with _cost_model(True) as matrix:
        assert matrix.cost_model.occupancy("MOVD2B") is None
        assert matrix.instruction_occupancy("MOVD2B", 0) is None


# ---------------------------------------------------------------------------
# 3. A multi-cycle cost really occupies the unit.
# ---------------------------------------------------------------------------


def test_a_multi_cycle_cost_holds_the_unit_and_back_pressures_the_thread():
    """An occupied unit refuses the next instruction rather than queueing it.

    Queueing was the original shape and it is wrong for a reason worth keeping
    written down: tt-sim's frontend treats an instruction as issued the moment
    a unit accepts it, so an instruction parked in an occupied unit's queue
    retires *after* the thread's next instruction has already run in a
    different unit — a reordering of one thread's program. See
    ``TensixBackendUnit.is_occupied``, which found it by making the config
    unit's ``RDCFG`` cost two cycles and watching ``matmulblock`` go wrong.
    """
    with _cost_model(True) as matrix:
        matrix.cost_model = _FixedCost(3)
        rwc = matrix.backend.getRWC(0)

        assert matrix.issueInstruction(INCRWC_A1, 0)
        matrix.clock_tick(0)
        assert rwc.SrcA == 1  # the first one retires in the cycle it issued
        # The hold runs from *acceptance* (the cycle before this retire tick,
        # since backend units tick before the wait gates): a 3-cycle occupancy
        # admits the next instruction 3 cycles after this one entered, so the
        # deadline is 2, not 3 — anchoring it at retire charged 4 cycles per
        # instruction where silicon measures 2.97. See clock_tick's arming.
        assert matrix.busy_until == 2
        # The wait gates read this; it is the back-pressure signal.
        assert not matrix.issueInstruction(INCRWC_A1, 0)

        matrix.clock_tick(1)
        assert rwc.SrcA == 1
        assert not matrix.issueInstruction(INCRWC_A1, 0)

        # Backend units tick before the frontend issues, so the deadline tick
        # frees the unit and the retry lands on the deadline cycle itself.
        matrix.clock_tick(2)
        assert matrix.busy_until is None
        assert matrix.issueInstruction(INCRWC_A1, 0)
        matrix.clock_tick(3)
        assert rwc.SrcA == 2
        assert matrix.busy_until == 5


def test_an_occupied_unit_tells_the_pump_when_to_come_back():
    """``next_wake_cycle`` is what lets the pump skip the occupied cycles; note
    a *live* Tensix tile does not stride today (it fast-rejects on
    ``soft_active``), so this currently buys correct back-pressure rather than
    skipped work — see event-driven-pump.md Phase 4."""
    with _cost_model(True) as matrix:
        matrix.cost_model = _FixedCost(4)
        assert matrix.issueInstruction(INCRWC_A1, 0)
        matrix.clock_tick(0)
        # Deadline 3, not 4: the 4-cycle hold runs from the acceptance cycle
        # (one before the retire tick), so the pump comes back one earlier.
        assert matrix.next_wake_cycle(0) == 3
        assert matrix.next_wake_cycle(2) == 3
        matrix.clock_tick(3)
        assert matrix.next_wake_cycle(3) is None


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "matrix_cost_model_test OK: no cost model by default, MVMUL at 1 cycle "
        "per fidelity phase, multi-cycle occupancy back-pressures the thread"
    )


if __name__ == "__main__":
    main()
