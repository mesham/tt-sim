"""The Sync Unit's issue queue may hold a mutex op next to a non-mutex one.

A regression test for a crash the RISC-V cycle-cost model found on the Wormhole
``loopback`` replay. ``TensixSyncUnit.issueInstruction`` admits one non-mutex op
into a queue that already holds a mutex op, and then, when the *next* mutex op
arrives, walked the whole queue reading ``instr_args["mutex_index"]`` off every
entry -- including the one that has no such field. That raised ``KeyError``.

It is unreachable while every unit drains in the tick it was issued, which is
why it survived until something delayed the drain; it is the same family as the
config unit's ordering bug (see ``docs/plans/cost-model.md``), and it is a
functional bug rather than a cost-model one, so it is tested here rather than
alongside the cost wiring.

Run standalone (``python3 -m tt_sim.pe.tensix.sync_mixed_queue_test``) or under
pytest.
"""

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor


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


#: Opcode words, from ``tensix_instructions.yaml``: the opcode occupies bits
#: 24-31 and ``mutex_index`` starts at bit 0.
ATRELM = 0xA1 << 24
SEMWAIT = 0xA6 << 24


def test_a_non_mutex_op_queued_beside_a_mutex_op_does_not_crash_the_next_issue():
    sync = _sync_unit()
    atrelm = ATRELM | 2
    semwait = SEMWAIT

    assert sync.issueInstruction(atrelm, 0)
    # The else branch admits this alongside the mutex op already queued.
    assert sync.issueInstruction(semwait, 1)
    assert len(sync.next_instruction) == 2

    # Before the fix this raised KeyError('mutex_index') reading the SEMWAIT.
    sync.issueInstruction(ATRELM | 3, 2)


def test_two_mutex_ops_on_the_same_mutex_still_refuse_to_share_a_cycle():
    sync = _sync_unit()
    assert sync.issueInstruction(ATRELM | 2, 0)
    assert not sync.issueInstruction(ATRELM | 2, 1)
    assert sync.issueInstruction(ATRELM | 3, 1)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("sync_mixed_queue_test OK: a mixed sync queue no longer crashes issue")


if __name__ == "__main__":
    main()
