"""The tripwire under the cost-model gate's guard classification.

The gate (``driver/tests/cost_model_gate.py``) is slow — it runs 28 kernels. The
part of it that can rot silently is not the running but the *classification*:
whether each ``*_replay_test.py`` guard asserts a value (timing-agnostic, so it
can validate a timing model) or asserts recorded READ replies (a timing pin,
which by construction fails under any timing change).

These tests are pure source inspection and run in milliseconds, so they ride in
the normal ``pytest driver`` gate. They fail when somebody adds a guard, or
changes an existing guard's shape, without the classification following.
"""

import pytest

from driver.tests import cost_model_gate as gate


@pytest.fixture(scope="module")
def guards():
    return gate.discover_guards()


def test_every_guard_is_classified_and_agrees_with_the_baseline(guards):
    """Discovery drives the gate; BASELINE is the hand-checked record.

    A new guard, a deleted guard, or a guard that changed shape all land here
    rather than quietly changing which family the gate runs.
    """
    assert gate.classification_problems(guards) == []


def test_guards_were_actually_discovered(guards):
    """A glob that matches nothing would make every other assertion vacuous."""
    arches = {g.arch for g in guards}
    assert {"wormhole", "blackhole"} <= arches
    assert len(guards) >= 30


def test_every_guard_has_exactly_one_role(guards):
    """Budget-dependent guards get the prover; the rest are run under the model."""
    for g in guards:
        assert (g.kind in gate.POLL_BUDGET_DEPENDENT) ^ (
            g.kind in gate.BUDGET_INDEPENDENT
        ), f"{g.key}: {g.kind!r} is on both role lists or neither"


def test_a_budget_dependent_guard_is_proved_or_excused(guards):
    """No budget-dependent guard may be simply run and believed."""
    for g in guards:
        if g.kind in gate.POLL_BUDGET_DEPENDENT:
            assert (
                g.key in gate.PROVERS
                or g.key in gate.UNPROVABLE
                or (g.kind == gate.VALUE_POLL_BUDGET)
            ), f"{g.key} is budget-dependent with no prover and no recorded reason"


def test_the_prover_never_pumps_after_the_replay(guards):
    """The blind spot that produced a false 'not a timing artefact' verdict.

    Every captured trace ends with RESET_ASSERT to all cores then EXIT, so a
    device pumped after an exhausted replay can never advance: pump-after-replay
    reports a benign timing shift as a permanently wrong value. The proof must
    re-run the replay at a larger poll budget instead.
    """
    src = (gate.REPO / "driver/tests/cost_model_gate.py").read_text()
    assert "POLL_BUDGET_LADDER" in src
    assert "tt_device.run(" not in src, (
        "the prover pumps the device directly again — that is the invalid proof"
    )


def test_every_trace_really_does_end_in_reset_then_exit(guards):
    """The premise the poll-budget proof rests on, checked rather than assumed."""
    from tt_sim.bridge import protocol as proto
    from tt_sim.bridge.trace import parse_trace_line

    traces = sorted((gate.REPO / "driver/blackhole/server/traces").glob("*.trace"))
    assert traces, "no traces to check"
    for trace in traces[:4]:
        cmds = [
            p["cmd"]
            for p in (parse_trace_line(ln) for ln in trace.open())
            if p is not None
        ]
        assert cmds[-1] == proto.CMD_EXIT, f"{trace.name} does not end with EXIT"
        assert proto.CMD_RESET_ASSERT in cmds[-30:], (
            f"{trace.name} does not assert reset before exiting"
        )


def test_an_exclusion_carries_its_reason(guards):
    """An exclusion without a reason is how an exclusion list rots."""
    for key, reason in gate.UNPROVABLE.items():
        assert len(reason) > 80, f"{key}: give the reason, not a label"


def test_value_guards_are_runnable_as_modules(guards):
    """The gate invokes each value guard as ``python3 -m <module>``."""
    for g in guards:
        if g.kind in gate.BUDGET_INDEPENDENT:
            src = g.path.read_text()
            assert "def main(" in src, f"{g.key} has no main() for -m to reach"
            assert "__main__" in src, f"{g.key} is not runnable as a module"


@pytest.mark.parametrize(
    "source,expected",
    [
        ('if bytes(reply) == bytes(p["reply"]):', gate.TIMING_PINNED),
        ('REPLAY = REPO / "wormhole" / "replay.py"', gate.TIMING_PINNED),
        ("while go != RUN_MSG_DONE: device.tt_device.run(2000)", gate.VALUE_PUMPED),
        ("result = device.read(COORD, ADDR, 400)", gate.VALUE_POLL_BUDGET),
    ],
)
def test_the_classifier_rules(source, expected):
    """The rules in prose, as executable examples — see ``gate.classify``."""
    assert gate.classify(source) == expected


def test_a_pinned_guard_cannot_be_reclassified_by_deleting_its_comparison():
    """The heuristic keys off the assertion, not off a name or a docstring."""
    assert gate.classify('"""byte-identical replay"""\nassert x == y') != (
        gate.TIMING_PINNED
    )
    assert gate.classify('assert bytes(r) == bytes(parsed["reply"])') == (
        gate.TIMING_PINNED
    )
