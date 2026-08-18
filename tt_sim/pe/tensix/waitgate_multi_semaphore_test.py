"""A ``SEMWAIT`` over several semaphores holds until *every* one is satisfied.

``SEMWAIT.md``'s summary: "One or more semaphore conditions are selected using
a bitmask ... execution of the thread is paused until **all** of the selected
conditions are simultaneously met", and its condition-mask table states the
same rule from the waiting end:

    ||Keep on waiting and blocking if...|
    |**C0**|*Any* of the semaphores selected by ``SemaphoreMask`` have
      ``Value == 0``.|
    |**C1**|*Any* of the semaphores selected by ``SemaphoreMask`` have
      ``Value >= Max``.|

Both quantifiers are here, and they are one rule: the thread runs only when
every selected (semaphore, condition) pair is satisfied, so any single
unsatisfied pair holds it. The Blackhole page carries the identical table, so
there is nothing per-arch to model.

``check_for_wait_condition_met`` used to walk the selected semaphores with its
``return True`` indented into the ``for`` body, so the loop ran exactly one
iteration: **only the first selected semaphore was ever tested**, and a thread
waiting on two was released as soon as the lower-numbered one was satisfied.
Silently -- no fault, no diagnostic, just a thread let past its barrier. Every
in-tree kernel uses a one-hot ``sem_sel``, which is why nothing caught it and
why fixing it moves no existing number.

Run standalone (``python3 -m tt_sim.pe.tensix.waitgate_multi_semaphore_test``)
or under pytest.
"""

import pytest

from tt_sim.arch import BLACKHOLE_PROFILE, WORMHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.util.conversion import conv_to_bytes

#: Condition-mask bits, as ``SEMWAIT.md`` names them.
C0_WAIT_WHILE_ZERO = 0b01
C1_WAIT_WHILE_MAX = 0b10

#: Block mask bit B1, "block thread from starting new Sync Unit instructions"
#: -- the bit ``SEMWAIT.md`` recommends every wait carry.
BLOCK_SYNC = 1 << 1

#: ``SEMWAIT`` / ``SEMPOST`` from ``tensix_instructions.yaml``: opcode in bits
#: 24-31, ``sem_sel`` (one bit per semaphore) at bit 2, ``wait_sem_cond`` at
#: bit 0, ``stall_res`` (the block mask) at bit 15.
SEMWAIT_OP = 0xA6 << 24
SEMPOST_OP = 0xA4 << 24


def _semwait(sem_mask, cond=C0_WAIT_WHILE_ZERO, block=BLOCK_SYNC):
    return SEMWAIT_OP | (block << 15) | (sem_mask << 2) | cond


def _sempost(sem_mask):
    return SEMPOST_OP | (sem_mask << 2)


def _coprocessor(profile=WORMHOLE_PROFILE):
    return TensixCoProcessor(
        None,
        profile.tensix_cfg_state_size,
        profile.tensix_thd_state_size,
        blackhole=profile is BLACKHOLE_PROFILE,
    )


def _gate_and_semaphores(profile=WORMHOLE_PROFILE, thread_id=1):
    cop = _coprocessor(profile)
    sync = cop.getBackend().getSyncUnit()
    return cop, cop.getThread(thread_id).wait_gate, sync


def test_a_two_semaphore_wait_is_not_met_by_the_first_alone():
    """The reported defect, at its smallest.

    ``sem_sel = 0b11`` under C0 with ``sem0 = 5`` and ``sem1 = 0``: the first
    selected semaphore satisfies the condition and the second does not, so
    "any of the semaphores ... have ``Value == 0``" holds and the thread keeps
    waiting. The pre-fix gate returned ``True`` here.
    """
    _cop, gate, sync = _gate_and_semaphores()
    gate.setLatchedWaitInstruction("SEMWAIT", C0_WAIT_WHILE_ZERO, BLOCK_SYNC, 0b11)
    sync.getSemaphore(0).value = 5
    sync.getSemaphore(1).value = 0

    assert not gate.check_for_wait_condition_met()

    # ... and the reason names the semaphore that is actually holding it, not
    # the one that is already satisfied.
    assert gate._latched_semaphore_reason() == ("semaphore_empty", 1)

    sync.getSemaphore(1).value = 1
    assert gate.check_for_wait_condition_met()


@pytest.mark.parametrize("unsatisfied", range(8))
def test_any_one_of_eight_selected_semaphores_holds_the_thread(unsatisfied):
    """``semaphore_mask`` is 8 bits wide, so all eight positions must bite.

    A loop that ran one iteration passed the ``unsatisfied == 0`` case and
    failed the other seven, which is the shape of a truncated walk rather than
    of a wrong condition.
    """
    _cop, gate, sync = _gate_and_semaphores()
    gate.setLatchedWaitInstruction("SEMWAIT", C0_WAIT_WHILE_ZERO, BLOCK_SYNC, 0xFF)
    for i in range(8):
        sync.getSemaphore(i).value = 0 if i == unsatisfied else 3

    assert not gate.check_for_wait_condition_met()
    assert gate._latched_semaphore_reason() == ("semaphore_empty", unsatisfied)

    sync.getSemaphore(unsatisfied).value = 3
    assert gate.check_for_wait_condition_met()


def test_both_conditions_apply_to_every_selected_semaphore():
    """C0 and C1 together: the quantifier is over pairs, not over semaphores.

    With both condition bits set, a selected semaphore is satisfied only when
    it is neither empty nor full, and *every* selected semaphore must be. Here
    ``sem0`` is comfortably mid-range and ``sem1`` is at its max, so the C1 arm
    of the second semaphore is the one unsatisfied pair -- a case that needs
    both the loop and both condition tests to be reached at all.
    """
    _cop, gate, sync = _gate_and_semaphores()
    gate.setLatchedWaitInstruction(
        "SEMWAIT", C0_WAIT_WHILE_ZERO | C1_WAIT_WHILE_MAX, BLOCK_SYNC, 0b11
    )
    for i in (0, 1):
        sync.getSemaphore(i).max = 4
    sync.getSemaphore(0).value = 2
    sync.getSemaphore(1).value = 4

    assert not gate.check_for_wait_condition_met()
    assert gate._latched_semaphore_reason() == ("semaphore_full", 1)

    # Draining the full one satisfies the pair, and nothing else was pending.
    sync.getSemaphore(1).value = 3
    assert gate.check_for_wait_condition_met()

    # Emptying the other re-arms the wait through the *first* condition, on
    # the semaphore the truncated walk would have been looking at anyway --
    # included so the test cannot be passed by simply inverting the return.
    sync.getSemaphore(0).value = 0
    assert not gate.check_for_wait_condition_met()
    assert gate._latched_semaphore_reason() == ("semaphore_empty", 0)


def test_a_single_semaphore_mask_is_unchanged():
    """The shape every in-tree kernel uses, pinned so the fix is a no-op on it.

    This is why nothing else in the tree moves: with a one-hot ``sem_sel`` the
    truncated walk and the full walk visit the same single semaphore and agree
    on every input.
    """
    _cop, gate, sync = _gate_and_semaphores()
    gate.setLatchedWaitInstruction("SEMWAIT", C0_WAIT_WHILE_ZERO, BLOCK_SYNC, 0b1)
    assert not gate.check_for_wait_condition_met()
    sync.getSemaphore(0).value = 1
    assert gate.check_for_wait_condition_met()
    # A *different* semaphore going empty is not selected and must not matter.
    sync.getSemaphore(3).value = 0
    assert gate.check_for_wait_condition_met()


@pytest.mark.parametrize(
    "profile", [WORMHOLE_PROFILE, BLACKHOLE_PROFILE], ids=["wormhole", "blackhole"]
)
def test_the_rule_is_the_same_on_both_architectures(profile):
    """Both arch's ``SEMWAIT.md`` carry the identical condition-mask table.

    The Wormhole/Blackhole split in this gate is over ``STALLWAIT``'s condition
    numbering (``blackhole_conditions``); the semaphore arm has no per-arch
    behaviour and this pins that it stays that way.
    """
    _cop, gate, sync = _gate_and_semaphores(profile)
    gate.setLatchedWaitInstruction("SEMWAIT", C0_WAIT_WHILE_ZERO, BLOCK_SYNC, 0b101)
    sync.getSemaphore(0).value = 1
    assert not gate.check_for_wait_condition_met()
    sync.getSemaphore(2).value = 1
    assert gate.check_for_wait_condition_met()


def test_a_real_semwait_holds_a_real_blocked_instruction():
    """End to end through the frontend: instruction words in, gate state out.

    A thread issues a genuine ``SEMWAIT`` word with a two-bit ``sem_sel``,
    then a ``SEMPOST`` -- a Sync Unit instruction, which the wait's block mask
    names. With one selected semaphore satisfied and the other empty, the
    ``SEMPOST`` must stay held at the gate and its side effect must not
    happen. The pre-fix gate let it through, and semaphore 7 moved.
    """
    cop, gate, sync = _gate_and_semaphores()
    thread = cop.getThread(1)
    clocks = cop.getClocks()

    def tick(cycle, count=1):
        for _ in range(count):
            for clockable in clocks:
                clockable.clock_tick(cycle)
            cycle += 1
        return cycle

    sync.getSemaphore(0).value = 5  # satisfied ...
    sync.getSemaphore(1).value = 0  # ... and not.

    cycle = 0
    thread.write(0, conv_to_bytes(_semwait(0b11), 4))
    cycle = tick(cycle, 10)
    # The gate re-evaluates a latched wait every cycle and forgets it the
    # moment it is met, so a wait that is still latched here is a wait that is
    # still unmet. The pre-fix gate met it immediately off ``sem0`` and this
    # is where it dropped the latch.
    assert gate.latchedWaitInstruction is not None, (
        "the wait was forgotten while semaphore 1 was still empty"
    )

    # The blocked instruction arrives and must not execute.
    thread.write(0, conv_to_bytes(_sempost(1 << 7), 4))
    cycle = tick(cycle, 20)
    assert gate.latch_wait, "a two-semaphore wait released on the first alone"
    assert sync.getSemaphore(7).value == 0

    # Satisfying the second selected semaphore releases it, and only then.
    sync.getSemaphore(1).value = 1
    cycle = tick(cycle, 20)
    assert not gate.latch_wait
    assert gate.latchedWaitInstruction is None
    assert sync.getSemaphore(7).value == 1


def main():
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        cases = [
            case
            for mark in marks
            if mark.name == "parametrize"
            for case in mark.args[1]
        ]
        for case in cases or [None]:
            fn() if case is None else fn(case)
    print(
        "waitgate_multi_semaphore_test OK: a SEMWAIT over several semaphores "
        "holds until every selected condition is met"
    )


if __name__ == "__main__":
    main()
