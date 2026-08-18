"""A latched wait is live from the ``SEMWAIT``, not from the first block.

``SEMWAIT.md``: the issuing thread "can continue execution until one of the
blocked instructions is reached", and the Wait Gate "will then continuously
re-evaluate the latched wait instruction until all of the selected conditions
are simultaneously met, at which point the latched wait instruction will be
forgotten". Both halves of that sentence are tested here, because tt-sim
previously did neither on the cycles with nothing held at the gate:

* the condition is **re-evaluated** with nothing blocked, so the latch is
  forgotten when it is met rather than surviving to block a later instruction;
* every cycle it is **not** met counts on ``WAITING_FOR_NON{ZERO,FULL}_SEM_n``
  and **not** on ``THREAD_STALLS_n`` -- nothing is held, so no cycle is lost.

The last test is the point of the whole exercise: it drives a real Tensix tile
through its own instruction path and MMIO counter registers and reads back a
window in which a reason counter *exceeds* the thread's stall count. That
window is what
``tt_sim/perf/stall_attribution_test.py::test_stall_reason_overlap_refuses...``
had to hand-build a counter bank for; the simulator can now produce one.

Run standalone (``python3 -m tt_sim.pe.tensix.waitgate_latched_wait_test``) or
under pytest.
"""

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.misc.perf_counters import BANK_REGISTERS
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

INSTRN_BASE, _INSTRN_OUT_L, INSTRN_OUT_H = BANK_REGISTERS["INSTRN_THREAD"]

#: ``counter_sel`` values from tt-metal's ``wormhole/hw_counters.h``: the
#: per-thread stall-reason block starts at 39 on Wormhole, so
#: ``WAITING_FOR_NONZERO_SEM_1`` is 39 + 9 + 4.
SEL_THREAD_STALLS_1 = 25
SEL_WAITING_FOR_NONZERO_SEM_1_WH = 52

#: ``SEMWAIT``, from ``tensix_instructions.yaml``: opcode in bits 24-31,
#: ``wait_sem_cond`` at bit 0 (1 = wait while the semaphore is zero),
#: ``sem_sel`` at bit 2 (one-hot), ``stall_res`` at bit 15 (block mask; bit 1
#: is "block thread from starting new Sync Unit instructions", the bit
#: ``SEMWAIT.md`` recommends every wait carry).
SEMWAIT_OP = 0xA6 << 24
BLOCK_SYNC = 1 << 1


def _semwait(sem=0, cond=1, block=BLOCK_SYNC):
    return SEMWAIT_OP | (block << 15) | ((1 << sem) << 2) | cond


def _coprocessor():
    return TensixCoProcessor(
        None,
        WORMHOLE_PROFILE.tensix_cfg_state_size,
        WORMHOLE_PROFILE.tensix_thd_state_size,
    )


def _armed_gate(thread_id=1):
    """A wait gate whose tile's counter block a kernel has just started."""
    cop = _coprocessor()
    counters = cop.perf_counters
    counters.write(INSTRN_BASE + 8, 0, 0)
    counters.write(INSTRN_BASE + 8, 1, 0)
    assert counters.instrn_running
    return cop, cop.getThread(thread_id).wait_gate, counters


def test_a_live_latched_wait_counts_a_reason_but_not_a_stall():
    """The counting rule the held path cannot express."""
    _cop, gate, counters = _armed_gate()
    gate.setLatchedWaitInstruction("SEMWAIT", 0b01, BLOCK_SYNC, 0b1)

    for cycle in range(20):
        gate.clock_tick(cycle)

    # Nothing was ever held: the gate has no instruction and ``latch_wait``
    # never went true.
    assert not gate.latch_wait
    assert counters.sem_empty[1] == 20
    assert counters.thread_stalls[1] == 0
    # ... which is precisely the inequality the hardware breaks and the old
    # model could not.
    assert counters.sem_empty[1] > counters.thread_stalls[1]


def test_a_met_condition_forgets_the_latch_with_nothing_held():
    _cop, gate, counters = _armed_gate()
    sync = _cop.getBackend().getSyncUnit()
    gate.setLatchedWaitInstruction("SEMWAIT", 0b01, BLOCK_SYNC, 0b1)

    gate.clock_tick(0)
    # Unmet: the latch survives and the cycle is counted.
    assert gate.latchedWaitInstruction is not None
    assert counters.sem_empty[1] == 1

    sync.getSemaphore(0).value = 1
    gate.clock_tick(1)
    # Met: "the latched wait instruction will be forgotten", and the cycle it
    # was met on is not a cycle spent waiting.
    assert gate.latchedWaitInstruction is None
    assert counters.sem_empty[1] == 1

    # A latch that is gone cannot come back when the semaphore drains again --
    # the pre-fix model kept it for ever and would have blocked on it here.
    sync.getSemaphore(0).value = 0
    gate.clock_tick(2)
    assert gate.latchedWaitInstruction is None
    assert counters.sem_empty[1] == 1


def test_a_latched_stallwait_counts_nothing_on_the_unheld_path():
    """A ``STALLWAIT``'s conditions are not the two the counter accepts.

    ``note_wait_condition`` declines everything but the semaphore reasons and
    raises if offered one, so this also proves the gate does not offer one.
    """
    _cop, gate, counters = _armed_gate()
    # C10 -- "SrcA bank the Matrix Unit is pointed at is not yet its to read"
    # -- is genuinely *unmet* out of reset, so the latch survives and the
    # un-held path runs on it every cycle. A condition that were satisfied
    # would prove nothing here: the latch would be forgotten on the first tick
    # and there would be no cycles to miscount.
    gate.setLatchedWaitInstruction("STALLWAIT", 1 << 10, BLOCK_SYNC)
    assert not gate.check_for_wait_condition_met()

    reasons = []
    counters.note_wait_condition = lambda thread_id, reason: reasons.append(reason)
    for cycle in range(5):
        gate.clock_tick(cycle)

    assert gate.latchedWaitInstruction is not None  # still live, still unmet
    assert reasons == []  # nothing was offered, not even a declined reason
    assert counters.sem_empty == [0, 0, 0]
    assert counters.sem_full == [0, 0, 0]
    assert counters.thread_stalls == [0, 0, 0]


def test_nothing_counts_until_the_kernel_arms_the_block():
    """The hot-path guard: an unprofiled run must not accumulate."""
    cop = _coprocessor()
    counters = cop.perf_counters
    gate = cop.getThread(1).wait_gate
    assert not counters.instrn_running
    gate.setLatchedWaitInstruction("SEMWAIT", 0b01, BLOCK_SYNC, 0b1)

    for cycle in range(10):
        gate.clock_tick(cycle)
    assert counters.sem_empty[1] == 0

    counters.write(INSTRN_BASE + 8, 0, 10)
    counters.write(INSTRN_BASE + 8, 1, 10)
    for cycle in range(10, 20):
        gate.clock_tick(cycle)
    assert counters.sem_empty[1] == 10


def test_the_gate_stays_awake_while_a_wait_is_latched():
    """Dormancy would skip the very cycles the counter is defined over."""
    cop = _coprocessor()
    gate = cop.getThread(1).wait_gate
    assert gate.is_clock_idle()

    gate.setLatchedWaitInstruction("SEMWAIT", 0b01, BLOCK_SYNC, 0b1)
    assert not gate.latch_wait  # nothing is held ...
    assert not gate.is_clock_idle()  # ... and the gate is still not idle.

    cop.getBackend().getSyncUnit().getSemaphore(0).value = 1
    gate.clock_tick(0)
    assert gate.latchedWaitInstruction is None
    assert gate.is_clock_idle()


def test_a_simulated_run_puts_a_reason_counter_above_thread_stalls():
    """End to end on a real tile: instructions in, counter registers out.

    A thread issues a real ``SEMWAIT`` word into its frontend, runs on for a
    while without reaching a blocked instruction (the un-held span the
    hardware still counts), then reaches one and is held (the span that is
    also a stall). Read back through ``RISCV_DEBUG_REG_PERF_CNT_*`` at the
    addresses a kernel uses, ``WAITING_FOR_NONZERO_SEM_1`` exceeds
    ``THREAD_STALLS_1`` -- the state the mechanism-attribution gate refuses,
    produced by the simulator rather than assembled by a test.
    """
    from tt_sim.device.tiles import TensixTile

    tile = TensixTile(18, 18, 1, 1, profile=WORMHOLE_PROFILE)
    base = 0xFFB12000
    cop = tile.tensix_coprocessor
    thread = cop.getThread(1)
    gate = thread.wait_gate
    clocks = cop.getClocks()

    def tick(cycle, count=1):
        for _ in range(count):
            for clockable in clocks:
                clockable.clock_tick(cycle)
            cycle += 1
        return cycle

    tile.tensix_mem.write(base + INSTRN_BASE + 8, conv_to_bytes(0, 4))
    tile.tensix_mem.write(base + INSTRN_BASE + 8, conv_to_bytes(1, 4))

    cycle = 0
    thread.write(0, conv_to_bytes(_semwait(), 4))
    while gate.latchedWaitInstruction is None:
        cycle = tick(cycle)
        assert cycle < 20, "the SEMWAIT never reached the Sync Unit"
    latched_at = cycle

    # The thread continues: nothing it issues in this span is blocked, so
    # nothing is held at the gate. Every cycle of it is a cycle the condition
    # was live and unmet, and none of them is a stall.
    cycle = tick(cycle, 40)
    assert not gate.latch_wait
    unheld = gate.perf_counters.sem_empty[1]
    assert unheld >= cycle - latched_at
    assert gate.perf_counters.thread_stalls[1] == 0

    # Now it reaches a blocked instruction -- another Sync Unit op, which the
    # latched wait's block mask names -- and is held. Those cycles are stalls
    # and count on both.
    thread.write(0, conv_to_bytes(_semwait(sem=1), 4))
    cycle = tick(cycle, 20)
    assert gate.latch_wait
    held = gate.perf_counters.thread_stalls[1]
    assert 0 < held < unheld

    tile.tensix_mem.write(base + INSTRN_BASE + 8, conv_to_bytes(0, 4))
    tile.tensix_mem.write(base + INSTRN_BASE + 8, conv_to_bytes(2, 4))

    def read(sel):
        tile.tensix_mem.write(base + INSTRN_BASE + 4, conv_to_bytes(sel << 8, 4))
        return conv_to_uint32(tile.tensix_mem.read(base + INSTRN_OUT_H, 4))

    stalls = read(SEL_THREAD_STALLS_1)
    sem_empty = read(SEL_WAITING_FOR_NONZERO_SEM_1_WH)
    assert stalls == held
    # Each cycle counted once: the un-held span on the reason counter alone,
    # the held span on both.
    assert sem_empty == unheld + held
    assert sem_empty > stalls


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "waitgate_latched_wait_test OK: a live latched wait is re-evaluated, "
        "forgotten when met, and counted while it is not"
    )


if __name__ == "__main__":
    main()
