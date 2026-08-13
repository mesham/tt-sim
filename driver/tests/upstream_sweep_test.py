"""Unit tests for the upstream-example gate's own logic.

The gate itself needs a built tt-metal to say anything, so everything in it that
can be wrong *without* tt-metal is checked here instead: the program table's
shape, and the one real value check it performs (``contributed/vecadd`` has no
upstream self-check, so the gate scores it from its own printout — a check that
never fails is worse than no check at all).
"""

from __future__ import annotations

from driver.tests import upstream_sweep as sweep


def _printout(rows):
    head = "Partial results: (note we are running under BFP16.)\n"
    return head + "".join(f"  {a} + {b} = {c}\n" for a, b, c in rows)


def test_value_check_accepts_correct_sums():
    rows = [(0.25 * i, 0.5, 0.25 * i + 0.5) for i in range(10)]
    assert sweep._check_contributed_vecadd(_printout(rows)) is None


def test_value_check_rejects_a_wrong_sum():
    """The point of the check: one wrong element must turn the row red."""
    rows = [(0.25 * i, 0.5, 0.25 * i + 0.5) for i in range(10)]
    rows[4] = (1.0, 0.5, 2.0)
    why = sweep._check_contributed_vecadd(_printout(rows))
    assert why is not None
    assert "want ~1.5" in why


def test_value_check_rejects_a_truncated_printout():
    """A crashed run that printed three of its ten lines is not a pass."""
    rows = [(0.25 * i, 0.5, 0.25 * i + 0.5) for i in range(3)]
    why = sweep._check_contributed_vecadd(_printout(rows))
    assert why is not None
    assert "parsed 3" in why


def test_value_check_tolerates_bfloat16_rounding():
    """bfloat16 keeps 8 mantissa bits, so the sum is not the exact float sum."""
    rows = [(0.339844, 0.578125, 0.917969)] * 10
    assert sweep._check_contributed_vecadd(_printout(rows)) is None


def test_program_table_is_well_formed():
    names = [p.name for p in sweep.PROGRAMS]
    assert len(names) == len(set(names)), "duplicate program name"
    assert not set(names) & set(sweep.EXCLUDED), "a program is both run and excluded"
    for prog in sweep.PROGRAMS:
        assert prog.success, f"{prog.name} has no success line"
        assert prog.tier in ("fast", "full")
        assert prog.check in ("self", "value", "completion")
        # A row that claims to check a value must have something doing the
        # checking, and a row that does not must not silently carry one.
        assert (prog.value_check is not None) == (prog.check == "value")


def test_expected_record_names_real_programs():
    names = {p.name for p in sweep.PROGRAMS}
    for arch, name in sweep.EXPECTED:
        assert arch in sweep.ARCHES, f"unknown arch {arch!r} in EXPECTED"
        assert name in names, f"EXPECTED names {name!r}, which the gate never runs"


def test_fast_tier_is_a_subset_of_full():
    fast = {p.name for p in sweep.select("fast", [])}
    full = {p.name for p in sweep.select("full", [])}
    assert fast < full
