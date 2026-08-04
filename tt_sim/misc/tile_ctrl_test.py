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

from tt_sim.misc.tile_ctrl import TensixTileControl
from tt_sim.util.conversion import conv_to_uint32

WALL_CLOCK_L = 0x1F0
WALL_CLOCK_LATCHED_H = 0x1F4
WALL_CLOCK_H = 0x1F8


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


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
