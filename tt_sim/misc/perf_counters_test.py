"""The Tensix performance counters, at the register interface a kernel uses.

The point of these guards is the one the roadmap names: a kernel reading
``RISCV_DEBUG_REG_PERF_CNT_*`` used to get zero from a permissive generic
store, and zero decodes as "nothing ever stalled". Two things therefore have
to hold — a counter tt-sim *does* source must report the quantity tt-sim
tracks, and a counter it does *not* source must say so rather than pass for a
measurement.

Run standalone (``python3 -m tt_sim.misc.perf_counters_test``) or under pytest.
"""

import pytest

from tt_sim.misc.perf_counters import (
    BANK_REGISTERS,
    PERF_CNT_MUX_CTRL,
    TensixPerfCounters,
)
from tt_sim.misc.tile_ctrl import TensixTileControl
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

INSTRN_BASE, INSTRN_OUT_L, INSTRN_OUT_H = BANK_REGISTERS["INSTRN_THREAD"]
FPU_BASE, FPU_OUT_L, FPU_OUT_H = BANK_REGISTERS["FPU"]

#: ``counter_sel`` values from tt-metal's ``blackhole/hw_counters.h``.
SEL_THREAD_STALLS_1 = 25
SEL_WAITING_FOR_NONZERO_SEM_0 = 35
SEL_WAITING_FOR_SRCA_VALID_BH = 29
SEL_WAITING_FOR_SRCA_CLEAR_BH = 27
SEL_WAITING_FOR_MATH_IDLE_1 = 43
SEL_THREAD_INSTRUCTIONS_1 = 264


class _Clock:
    def __init__(self, cycle=0):
        self.current_cycle = cycle


def _ctrl(blackhole=True, cycle=0):
    counters = TensixPerfCounters(blackhole)
    ctrl = TensixTileControl(perf_counters=counters)
    ctrl.clock_owner = _Clock(cycle)
    return ctrl, counters


def _start(ctrl):
    """The exact sequence ``start_single_group`` writes on hardware."""
    ctrl.write(INSTRN_BASE, conv_to_bytes(0xFFFFFFFF, 4))
    ctrl.write(INSTRN_BASE + 4, conv_to_bytes(0, 4))
    ctrl.write(INSTRN_BASE + 8, conv_to_bytes(0, 4))
    ctrl.write(INSTRN_BASE + 8, conv_to_bytes(1, 4))


def _stop(ctrl):
    ctrl.write(INSTRN_BASE + 8, conv_to_bytes(0, 4))
    ctrl.write(INSTRN_BASE + 8, conv_to_bytes(2, 4))


def _read(ctrl, sel):
    """The exact sequence ``read_single_group`` performs per counter."""
    ctrl.write(INSTRN_BASE + 4, conv_to_bytes(sel << 8, 4))
    assert conv_to_uint32(ctrl.read(INSTRN_BASE + 4, 4)) == sel << 8
    ref_cnt = conv_to_uint32(ctrl.read(INSTRN_OUT_L, 4))
    value = conv_to_uint32(ctrl.read(INSTRN_OUT_H, 4))
    return ref_cnt, value


def test_counter_read_returns_the_quantity_tt_sim_tracks():
    """THREAD_STALLS_1 reports the stalled cycles the wait gate recorded.

    **This test fails on the unfixed tree**, where ``TensixTileControl.read``
    falls through to ``self.regs.get(addr, 0)`` and every select answers 0 —
    which reads as a thread that never stalled.
    """
    ctrl, counters = _ctrl()
    _start(ctrl)
    for _ in range(7):
        counters.note_stall(1, "semaphore_empty")
    for _ in range(3):
        counters.note_stall(1, "backend_enforced_stall")
    counters.note_stall(0, "semaphore_empty")
    _stop(ctrl)

    assert _read(ctrl, SEL_THREAD_STALLS_1)[1] == 10
    assert _read(ctrl, SEL_WAITING_FOR_NONZERO_SEM_0)[1] == 1


def test_src_valid_and_clear_are_opposite_directions():
    """``_VALID`` is math starved by unpack; ``_CLEAR`` is unpack held by math.

    The tech report's own wording: ``WAITING_FOR_SRCA_VALID`` is "cycles
    waiting for source register data to become valid (unpacker hasn't filled
    it yet)" — tt-sim's ``src_reserved_by_unpacker`` — while
    ``WAITING_FOR_SRCA_CLEAR`` is "cycles waiting for source register to be
    cleared (math is still using the previous data)" — tt-sim's
    ``src_reserved_by_matrix``.
    """
    ctrl, counters = _ctrl()
    _start(ctrl)
    for _ in range(4):
        counters.note_stall(1, "src_reserved_by_unpacker", src_bank="A")
    for _ in range(9):
        counters.note_stall(0, "src_reserved_by_matrix", src_bank="A")
    _stop(ctrl)

    assert _read(ctrl, SEL_WAITING_FOR_SRCA_VALID_BH)[1] == 4
    assert _read(ctrl, SEL_WAITING_FOR_SRCA_CLEAR_BH)[1] == 9


def test_thread_instructions_is_the_grant_side():
    ctrl, counters = _ctrl()
    _start(ctrl)
    for _ in range(5):
        counters.note_dispatch(1)
    counters.note_dispatch(2)
    _stop(ctrl)
    assert _read(ctrl, SEL_THREAD_INSTRUCTIONS_1)[1] == 5


def test_nothing_counts_before_the_start_edge():
    """Armed by the kernel's own start bit, exactly as the hardware is."""
    ctrl, counters = _ctrl()
    assert not counters.instrn_running
    _start(ctrl)
    assert counters.instrn_running
    _stop(ctrl)
    assert not counters.instrn_running


def test_restart_clears_the_counters():
    """ "0->1 transition also clears the counters" -- the register reference."""
    ctrl, counters = _ctrl()
    _start(ctrl)
    counters.note_stall(1, "mutex_wait")
    _start(ctrl)
    _stop(ctrl)
    assert _read(ctrl, SEL_THREAD_STALLS_1)[1] == 0


def test_ref_cnt_is_the_elapsed_window():
    ctrl, counters = _ctrl(cycle=1001)
    _start(ctrl)
    ctrl.clock_owner.current_cycle = 3001
    _stop(ctrl)
    # ``cycle_num`` is deliberately ``current_cycle - 1`` throughout tile_ctrl.
    assert _read(ctrl, SEL_THREAD_STALLS_1)[0] == 2000


def test_declined_counter_is_loud_not_zero(capsys):
    """``WAITING_FOR_*_IDLE_*`` is declined, and says so on first read."""
    ctrl, _ = _ctrl()
    _start(ctrl)
    _stop(ctrl)
    assert _read(ctrl, SEL_WAITING_FOR_MATH_IDLE_1)[1] == 0
    warning = capsys.readouterr().err
    assert "WAITING_FOR_MATH_IDLE_1" in warning
    assert "NOT a measurement of zero" in warning


def test_declined_counter_raises_in_strict_mode(monkeypatch):
    monkeypatch.setenv(TensixPerfCounters.STRICT_ENV, "1")
    ctrl, _ = _ctrl()
    _start(ctrl)
    _stop(ctrl)
    ctrl.write(INSTRN_BASE + 4, conv_to_bytes(SEL_WAITING_FOR_MATH_IDLE_1 << 8, 4))
    with pytest.raises(NotImplementedError, match="WAITING_FOR_MATH_IDLE_1"):
        ctrl.read(INSTRN_OUT_H, 4)


def test_unsourced_bank_is_loud(capsys):
    """No FPU-bank counter is sourced; reading one must not pass for zero."""
    ctrl, _ = _ctrl()
    ctrl.write(FPU_BASE + 4, conv_to_bytes(0, 4))
    assert conv_to_uint32(ctrl.read(FPU_OUT_H, 4)) == 0
    assert "FPU bank" in capsys.readouterr().err


def test_wormhole_and_blackhole_selects_differ():
    """The two architectures place the Src conditions differently.

    Blackhole gives each one select (27-30); Wormhole replicates each across
    three (27-38) and starts the per-thread stall-reason block at 39 rather
    than 31. A criterion that can only be evaluated on one part is not a
    criterion, so both layouts are modelled.
    """
    wh_ctrl, wh = _ctrl(blackhole=False)
    bh_ctrl, bh = _ctrl(blackhole=True)
    for ctrl, counters in ((wh_ctrl, wh), (bh_ctrl, bh)):
        _start(ctrl)
        counters.note_stall(1, "semaphore_empty")
        counters.note_stall(1, "semaphore_empty")
        _stop(ctrl)
    # WAITING_FOR_NONZERO_SEM_1: sel 44 on Blackhole (31 + 9 + 4), 52 on
    # Wormhole (39 + 9 + 4) -- the values in tt-metal's two hw_counters.h.
    assert _read(bh_ctrl, 44)[1] == 2
    assert _read(wh_ctrl, 52)[1] == 2


def test_mux_ctrl_round_trips():
    ctrl, _ = _ctrl()
    ctrl.write(PERF_CNT_MUX_CTRL, conv_to_bytes(3 << 4, 4))
    assert conv_to_uint32(ctrl.read(PERF_CNT_MUX_CTRL, 4)) == 3 << 4


def test_counters_are_live_on_a_real_tensix_tile():
    """End to end through the tile's own memory map, at the real addresses."""
    from tt_sim.arch.wormhole import WORMHOLE_PROFILE
    from tt_sim.device.tiles import TensixTile

    tile = TensixTile(18, 18, 1, 1, profile=WORMHOLE_PROFILE)
    base = 0xFFB12000
    tile.tensix_mem.write(base + INSTRN_BASE + 8, conv_to_bytes(0, 4))
    tile.tensix_mem.write(base + INSTRN_BASE + 8, conv_to_bytes(1, 4))
    gate = tile.tensix_coprocessor.getThread(1).wait_gate
    assert gate.perf_counters is tile.tile_ctrl.perf_counters
    gate._note_stall(0, "semaphore_full")
    gate._note_stall(1, "semaphore_full")
    tile.tensix_mem.write(
        base + INSTRN_BASE + 4, conv_to_bytes(SEL_THREAD_STALLS_1 << 8, 4)
    )
    value = conv_to_uint32(tile.tensix_mem.read(base + INSTRN_OUT_H, 4))
    assert value == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
