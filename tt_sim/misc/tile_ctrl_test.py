"""The wall-clock registers, which are how anything on the device tells time.

``RISCV_DEBUG_REG_WALL_CLOCK_L`` (0xFFB121F0) is the only timestamp source on a
Tensix tile. tt-metal's device profiler reads it for every ``DeviceZoneScopedN``
(``tt_metal/tools/profiler/kernel_profiler.hpp``), and so does
``perfbench/tensixbench``, which is why the two produce the same quantity on
silicon and on tt-sim -- see ``docs/plans/tensix-cost-benchmark.md``.

There are three registers and the reading order matters: a read of ``_L``
(0x1F0) latches the high half into 0x1F4, and ``_H`` (0x1F8) returns whatever
was last latched. The profiler reads 0x1F0 and 0x1F4; ``realtime_profiler.hpp``
reads 0x1F8 directly. That second path used to raise ``AttributeError`` on a
fresh tile, which in a memory read kills the server.

Run standalone (``python3 -m tt_sim.misc.tile_ctrl_test``) or under pytest.
"""

import pytest

from tt_sim.misc.tile_ctrl import (
    PERMISSIVE_ENV,
    TensixTileControl,
    UnmodelledTileRegisterError,
)
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

WALL_CLOCK_L = 0x1F0
WALL_CLOCK_LATCHED_H = 0x1F4
WALL_CLOCK_H = 0x1F8

#: RISCV_DEBUG_REG_DBG_ARRAY_RD_DATA -- a status register tt-sim does not
#: model, whose zero would read as "the debug array read back all-zero".
DBG_ARRAY_RD_DATA = 0x06C
#: RISCV_DEBUG_REG_TRISC_RESET_PC_OVERRIDE -- an unwritten override register
#: whose zero genuinely means "no override", and which Blackhole's boot path
#: depends on reading.
TRISC_RESET_PC_OVERRIDE = 0x234


class _Clock:
    def __init__(self, cycle):
        self.current_cycle = cycle
        self.awake = False


def _ctrl(cycle=0):
    ctrl = TensixTileControl()
    ctrl.clock_owner = _Clock(cycle)
    return ctrl


def test_wall_clock_high_reads_before_any_low_read():
    """0x1F8 must answer on a fresh tile rather than raising."""
    assert conv_to_uint32(_ctrl(1234).read(WALL_CLOCK_H, 4)) == 0


def test_wall_clock_low_is_the_previous_cycle():
    # Deliberately cycle - 1: tile_ctrl used to be ticked after the tile's
    # other components, so a core reading during cycle c saw the end of c - 1.
    assert conv_to_uint32(_ctrl(500).read(WALL_CLOCK_L, 4)) == 499


def test_reading_low_latches_high():
    ctrl = _ctrl((3 << 32) + 7)
    assert conv_to_uint32(ctrl.read(WALL_CLOCK_H, 4)) == 0
    ctrl.read(WALL_CLOCK_L, 4)
    assert conv_to_uint32(ctrl.read(WALL_CLOCK_H, 4)) == 3
    assert conv_to_uint32(ctrl.read(WALL_CLOCK_LATCHED_H, 4)) == 3


def test_unmodelled_status_register_read_is_loud():
    """The §6 hazard: a status register tt-sim does not model must not read 0.

    ``RISCV_DEBUG_REG_DBG_ARRAY_RD_DATA`` is a readback register. Before the
    allowlist, ``TensixTileControl.read`` answered it -- and every other
    unimplemented offset in the 4 KB window -- from ``self.regs.get(addr, 0)``,
    so a caller got a value that is indistinguishable from a real all-zero
    readback. **This test fails on the unfixed tree**, where the read returns 0
    instead of raising.
    """
    with pytest.raises(UnmodelledTileRegisterError) as excinfo:
        _ctrl().read(DBG_ARRAY_RD_DATA, 4)
    message = str(excinfo.value)
    assert "DBG_ARRAY_RD_DATA" in message
    assert PERMISSIVE_ENV in message


def test_unknown_offset_read_is_loud():
    """An offset that is not a register at all is louder still, not zero."""
    with pytest.raises(UnmodelledTileRegisterError):
        _ctrl().read(0x3F0, 4)


def test_permissive_env_restores_the_old_zero(monkeypatch):
    """The escape hatch, because a raise in a memory read kills the server."""
    monkeypatch.setenv(PERMISSIVE_ENV, "1")
    assert conv_to_uint32(_ctrl().read(DBG_ARRAY_RD_DATA, 4)) == 0


def test_reset_pc_override_still_reads_zero_unwritten():
    """The one in-tree dependency on a silent zero, kept working.

    ``BabyRISCV._blackhole_reset_pc`` reads 0x234 / 0x23C to decide whether a
    core boots from an overridden PC, and every Blackhole core boots because
    those registers read zero before anything writes them. That is a *correct*
    read of a real register at its reset value, not a fabricated status, so it
    stays on the store allowlist and stays silent.
    """
    ctrl = _ctrl()
    assert conv_to_uint32(ctrl.read(TRISC_RESET_PC_OVERRIDE, 4)) == 0
    ctrl.write(TRISC_RESET_PC_OVERRIDE, conv_to_bytes(0b101, 4))
    assert conv_to_uint32(ctrl.read(TRISC_RESET_PC_OVERRIDE, 4)) == 0b101


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
