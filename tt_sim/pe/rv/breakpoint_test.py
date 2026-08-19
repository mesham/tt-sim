"""``ebreak`` is a device-side assertion, and must not be executed through.

The regression these lock in is the *silent* one: tt-sim used to decode
``ebreak`` and report "handled", so a kernel that had asserted on itself ran on
to a clean exit. Every tt-metal device-side ``ASSERT()`` and every tripped LLK
sanitizer check lowers to this instruction in the assert-only build, so
ignoring it turned "the kernel says it is wrong" into "the program passed" --
the exact sim-versus-silicon divergence tt-sim exists to close.

``ecall`` is deliberately *not* trapped, and that asymmetry is tested here too:
nothing on the baby-core path issues one, and unlike ``ebreak`` it is not what
a failed assertion compiles to.
"""

import pytest

from tt_sim.behaviour import require
from tt_sim.pe.rv import breakpoint as breakpoint_trap
from tt_sim.pe.rv.breakpoint import RiscvBreakpoint
from tt_sim.pe.rv.isa.i_isa import RV_I_ISA

EBREAK = 0x00100073
ECALL = 0x00000073


class _MemorySpace:
    """Just enough of a memory space to carry ``caller_context``."""

    def __init__(self, ctx=None):
        self.caller_context = ctx


def test_ebreak_raises():
    with pytest.raises(RiscvBreakpoint):
        RV_I_ISA.handle_i_misc(EBREAK, None, _MemorySpace(), False)


def test_ebreak_names_the_core_and_pc():
    """The report has to say *which* core asserted and *where*.

    ``RV32.clock_tick`` stamps ``(unit_id, core_label, pc)`` on the memory
    space every cycle, and ``unit_id`` is the trace registry's
    ``(chip, core_y, core_x, unit)`` tuple -- so the tile coordinates come out
    of it in (x, y) order, not the (y, x) order they are stored in.
    """
    ctx = ((0, 2, 1, "TRISC1"), "TRISC1", 0xD194)
    with pytest.raises(RiscvBreakpoint) as excinfo:
        RV_I_ISA.handle_i_misc(EBREAK, None, _MemorySpace(ctx), False)
    message = str(excinfo.value)
    assert "TRISC1" in message
    assert "x=1, y=2" in message
    assert "0xd194" in message
    assert excinfo.value.pc == 0xD194
    assert excinfo.value.core_label == "TRISC1"


def test_ebreak_without_context_still_raises():
    """A core with no trace registration (unit tests, ex1..ex5) must still trap."""
    with pytest.raises(RiscvBreakpoint) as excinfo:
        RV_I_ISA.handle_i_misc(EBREAK, None, _MemorySpace(None), False)
    assert excinfo.value.pc is None
    assert "executed `ebreak`" in str(excinfo.value)


def test_ecall_is_still_ignored():
    assert RV_I_ISA.handle_i_misc(ECALL, None, _MemorySpace(), False) is True


def test_env_var_restores_the_old_behaviour(monkeypatch):
    monkeypatch.setenv(breakpoint_trap.IGNORE_ENV_VAR, "1")
    assert breakpoint_trap.refresh_from_env() is False
    try:
        assert RV_I_ISA.handle_i_misc(EBREAK, None, _MemorySpace(), False) is True
    finally:
        monkeypatch.delenv(breakpoint_trap.IGNORE_ENV_VAR)
        assert breakpoint_trap.refresh_from_env() is True


def test_env_var_truthiness(monkeypatch):
    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv(breakpoint_trap.IGNORE_ENV_VAR, truthy)
        assert breakpoint_trap.refresh_from_env() is False
    for falsy in ("0", "false", "no", "", "maybe"):
        monkeypatch.setenv(breakpoint_trap.IGNORE_ENV_VAR, falsy)
        assert breakpoint_trap.refresh_from_env() is True
    monkeypatch.delenv(breakpoint_trap.IGNORE_ENV_VAR)
    assert breakpoint_trap.refresh_from_env() is True


def test_the_behaviour_marker_for_this_guard_is_published():
    """The guard and the name external suites assert on live and die together.

    ``tt_sim.behaviour`` publishes ``riscv-ebreak-halts`` so a consumer's suite can refuse
    to run against a tt-sim that lacks this check rather than collect another
    set of green results that exercised nothing. Deleting the registry entry
    therefore has to turn *this* suite red — a marker quietly withdrawn is
    exactly the failure the marker exists to prevent.
    """
    require("riscv-ebreak-halts")
