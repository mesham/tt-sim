"""A granted mutex waiter must leave the Sync Unit's queue.

``TensixSyncUnit.clock_tick`` retries every entry in ``blocked_mutex`` — the
threads whose ``ATGETM`` found the mutex held by somebody else — and grants the
mutex to any whose turn has come. It then deleted the granted entries only
``if len(to_remove) > 1``, so a *lone* grant (the overwhelmingly common case:
one mutex, one waiter) stayed queued for ever.

A stale entry is not inert. Every later ``clock_tick`` re-runs the grant for it
and re-writes ``held_by``, so the mutex was silently re-acquired by that thread
on the cycle after each ``ATRELM``. Nothing failed while no other thread wanted
it — the owner's own ``ATGETM``/``ATRELM`` pairs still behaved — but the next
``ATGETM`` from a *different* thread blocked for ever behind an owner that had
already released. It also kept :meth:`is_clock_idle` false, so the tile never
went dormant: the wedge burned cycles at full speed instead of stalling, which
is why neither the dormancy path nor the deadlock watchdog caught it.

What it broke in practice: a host program calling ``detail::LaunchProgram``
twice in one process. The first launch contended for a mutex somewhere, leaving
the stale entry behind; the second launch's first cross-thread ``ATGETM`` then
never completed. Found by ``perfbench/tensixbench`` phase B (one launch per math
fidelity), which hung indefinitely on the second launch at 95 % CPU. The
end-to-end guard is ``driver/blackhole/server/twolaunch_replay_test.py``; these
tests pin the behaviour at the unit where it lives.

Run standalone (``python3 -m tt_sim.pe.tensix.sync_mutex_queue_test``) or under
pytest.
"""

from tt_sim.arch import BLACKHOLE_PROFILE, WORMHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor

#: Opcode words, from ``tensix_instructions.yaml``: the opcode is bits 24-31 and
#: ``mutex_index`` starts at bit 0. Mutex index 1 is not a valid mutex.
ATGETM = 0xA0 << 24
ATRELM = 0xA1 << 24
MUTEX = 2

# The Sync Unit is shared infrastructure, so both arches must behave the same.
PROFILES = (WORMHOLE_PROFILE, BLACKHOLE_PROFILE)


def _sync_unit(profile=WORMHOLE_PROFILE):
    return (
        TensixCoProcessor(
            None,
            profile.tensix_cfg_state_size,
            profile.tensix_thd_state_size,
            blackhole=profile.name == "blackhole",
        )
        .getBackend()
        .getSyncUnit()
    )


def _run(sync, instruction, thread, cycle):
    """Issue one instruction and retire it, returning the next cycle number."""
    assert sync.issueInstruction(instruction, thread)
    sync.clock_tick(cycle)
    return cycle + 1


def _idle(sync, cycle, ticks=4):
    for _ in range(ticks):
        sync.clock_tick(cycle)
        cycle += 1
    return cycle


def test_a_lone_granted_waiter_leaves_the_queue():
    for profile in PROFILES:
        sync = _sync_unit(profile)
        cycle = _run(sync, ATGETM | MUTEX, 0, 0)
        cycle = _run(sync, ATGETM | MUTEX, 1, cycle)
        # Thread 1 lost the race and is queued behind thread 0.
        assert sync.blocked_mutex == [(1, MUTEX)]
        assert sync.mutexes[MUTEX].held_by == 0

        cycle = _run(sync, ATRELM | MUTEX, 0, cycle)
        # The release and the grant land in the same tick.
        assert sync.mutexes[MUTEX].held_by == 1
        assert sync.blocked_mutex == []


def test_a_release_after_a_contended_grant_really_frees_the_mutex():
    """The two-``LaunchProgram`` regression, in miniature.

    Thread 1 acquires the mutex the contended way, then releases it. A later
    ``ATGETM`` from a different thread must win it. With the granted entry left
    in ``blocked_mutex`` the next tick handed the mutex straight back to thread
    1 and thread 0 waited for ever.
    """
    for profile in PROFILES:
        sync = _sync_unit(profile)
        cycle = _run(sync, ATGETM | MUTEX, 0, 0)
        cycle = _run(sync, ATGETM | MUTEX, 1, cycle)
        cycle = _run(sync, ATRELM | MUTEX, 0, cycle)
        assert sync.mutexes[MUTEX].held_by == 1

        cycle = _run(sync, ATRELM | MUTEX, 1, cycle)
        cycle = _idle(sync, cycle)
        assert sync.mutexes[MUTEX].held_by is None, (
            "a stale queue entry re-acquired the mutex after its owner released it"
        )

        cycle = _run(sync, ATGETM | MUTEX, 0, cycle)
        cycle = _idle(sync, cycle)
        assert sync.mutexes[MUTEX].held_by == 0
        assert sync.blocked_mutex == []


def test_two_waiters_are_granted_one_at_a_time_and_both_dequeue():
    sync = _sync_unit()
    cycle = _run(sync, ATGETM | MUTEX, 0, 0)
    cycle = _run(sync, ATGETM | MUTEX, 1, cycle)
    cycle = _run(sync, ATGETM | MUTEX, 2, cycle)
    assert sync.blocked_mutex == [(1, MUTEX), (2, MUTEX)]

    cycle = _run(sync, ATRELM | MUTEX, 0, cycle)
    assert sync.mutexes[MUTEX].held_by == 1
    assert sync.blocked_mutex == [(2, MUTEX)]

    cycle = _run(sync, ATRELM | MUTEX, 1, cycle)
    assert sync.mutexes[MUTEX].held_by == 2
    assert sync.blocked_mutex == []


def test_the_unit_goes_idle_again_once_the_waiter_is_granted():
    """The live-spin half of the bug: a queued waiter keeps the unit awake."""
    sync = _sync_unit()
    cycle = _run(sync, ATGETM | MUTEX, 0, 0)
    cycle = _run(sync, ATGETM | MUTEX, 1, cycle)
    assert not sync.is_clock_idle()
    cycle = _run(sync, ATRELM | MUTEX, 0, cycle)
    assert sync.is_clock_idle()


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("sync_mutex_queue_test OK: granted mutex waiters leave the queue")


if __name__ == "__main__":
    main()
