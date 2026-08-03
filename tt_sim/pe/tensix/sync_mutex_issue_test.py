"""The Sync Unit's mutex ops issue up to three per cycle, on distinct mutexes.

``SyncUnit.md``'s throughput table gives ``ATGETM`` / ``ATRELM`` "issue up to
three per cycle, provided they refer to different mutexes", against "issue at
most one of these per cycle" for the semaphore ops. ``issueInstruction`` had
that rule, but its branch tested for ``"ATGEM"`` -- a name the instruction table
does not contain -- so every ``ATGETM`` fell to the semaphore arm and took the
queue exclusively. Both halves of the rule (the three-per-cycle allowance and
the same-mutex conflict check) were therefore dead code for ``ATGETM``.

These pin the corrected behaviour at the unit's issue interface, which is where
the rule lives; nothing above it can observe the difference except as a
scheduling shift.

Run standalone (``python3 -m tt_sim.pe.tensix.sync_mutex_issue_test``) or under
pytest.
"""

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor

#: Opcode words, from ``tensix_instructions.yaml``: the opcode occupies bits
#: 24-31 and ``mutex_index`` starts at bit 0. Mutex 1 is not a valid index, so
#: the distinct mutexes used here are 2 (unpack0), 3 (unpack1) and 4 (pack0).
ATGETM = 0xA0 << 24
ATRELM = 0xA1 << 24
SEMWAIT = 0xA6 << 24


def _sync_unit():
    return (
        TensixCoProcessor(
            None,
            WORMHOLE_PROFILE.tensix_cfg_state_size,
            WORMHOLE_PROFILE.tensix_thd_state_size,
        )
        .getBackend()
        .getSyncUnit()
    )


def test_two_atgetms_on_different_mutexes_issue_in_the_same_cycle():
    sync = _sync_unit()
    assert sync.issueInstruction(ATGETM | 2, 0)
    assert sync.issueInstruction(ATGETM | 3, 1)
    assert len(sync.next_instruction) == 2


def test_two_atgetms_on_the_same_mutex_do_not_share_a_cycle():
    sync = _sync_unit()
    assert sync.issueInstruction(ATGETM | 2, 0)
    assert not sync.issueInstruction(ATGETM | 2, 1)
    assert len(sync.next_instruction) == 1


def test_an_atgetm_conflicts_with_an_atrelm_on_the_same_mutex():
    sync = _sync_unit()
    assert sync.issueInstruction(ATRELM | 4, 0)
    assert not sync.issueInstruction(ATGETM | 4, 1)
    assert sync.issueInstruction(ATGETM | 2, 1)


def test_at_most_three_mutex_ops_issue_in_one_cycle():
    sync = _sync_unit()
    assert sync.issueInstruction(ATGETM | 2, 0)
    assert sync.issueInstruction(ATGETM | 3, 1)
    assert sync.issueInstruction(ATGETM | 4, 2)
    # A fourth distinct mutex is still refused: three per cycle is the limit.
    assert not sync.issueInstruction(ATGETM | 5, 0)
    assert len(sync.next_instruction) == 3


def test_a_semaphore_op_still_shares_a_cycle_with_a_queued_atgetm():
    # The mutex and semaphore limits are independent, and the conflict loop
    # must skip the queued op that names no mutex (a SEMWAIT) rather than
    # raising KeyError on it -- see sync_mixed_queue_test.
    sync = _sync_unit()
    assert sync.issueInstruction(ATGETM | 2, 0)
    assert sync.issueInstruction(SEMWAIT, 1)
    assert not sync.issueInstruction(SEMWAIT, 2)
    assert sync.issueInstruction(ATGETM | 3, 2)
    assert len(sync.next_instruction) == 3


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "sync_mutex_issue_test OK: mutex ops issue three per cycle on distinct mutexes"
    )


if __name__ == "__main__":
    main()
