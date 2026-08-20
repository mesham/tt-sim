"""``STALLWAIT`` is held by any latched wait, so it cannot forget one.

``STALLWAIT.md``'s "exact set of instructions blocked from starting by each
bit" table gives ``STALLWAIT`` a tick in **all nine** block-mask columns -- on
Wormhole and on Blackhole alike -- and it is the only instruction that has
one. Every other row names the units the bit is about, which is what the Wait
Gate's per-unit table models; ``STALLWAIT``'s row does not, and modelling it
through ``ex_resource`` (``SYNC``, so bit B1 alone) let a ``STALLWAIT`` walk
past an unsatisfied ``SEMWAIT`` whose block mask was B0 and overwrite the
latch, which is a *dropped* wait rather than a late one.

That is not an abstract hazard: it is what tt-metal's
``pack_untilize_dest_init`` does when a kernel calls it after
``tile_regs_wait()`` rather than before ``matmul_init``. ``tile_regs_wait``
is ``SEMWAIT(B0, MATH_PACK, wait-while-zero)``, and the first thing
``_llk_init_packer_dest_offset_registers_`` issues is
``STALLWAIT(STALL_TDMA | STALL_THCON, PACK)``. With the ``SEMWAIT`` forgotten,
the ``PACR``s that follow no longer waited for the math thread, so the packer
read a Dst still mid-accumulation and the kernel's output was wrong -- on a
K >= 2 matmul, against both silicon and ttsim. ``optests/packuntilizeinit`` is
the end-to-end reproduction; this is the mechanism, pinned without it.

Run standalone (``python3 -m tt_sim.pe.tensix.waitgate_stallwait_blocked_test``)
or under pytest.
"""

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.pe.tensix.frontend import WaitGate
from tt_sim.pe.tensix.util import TensixInstructionDecoder
from tt_sim.util.conversion import conv_to_bytes

#: Block-mask bits, ``STALLWAIT.md``'s B0-B8.
B0_TDMA = 1 << 0  # Misc / Mover / ThCon / Packer / Unpacker
B1_SYNC = 1 << 1
B5_THCON = 1 << 5

#: Condition-mask bits. C0 ("ThCon has memory requests outstanding") is always
#: satisfied in tt-sim; C10 ("SrcA[MatrixUnit.SrcABank].AllowedClient !=
#: MatrixUnit") is genuinely unmet out of reset, which is what makes a latched
#: ``STALLWAIT`` observable rather than forgotten on the tick it lands.
C10_SRCA_NOT_MATH = 1 << 10

SEMWAIT_OP = 0xA6 << 24
STALLWAIT_OP = 0xA2 << 24


def _semwait(sem=0, cond=1, block=B0_TDMA):
    return SEMWAIT_OP | (block << 15) | ((1 << sem) << 2) | cond


def _stallwait(cond=C10_SRCA_NOT_MATH, block=B0_TDMA | B5_THCON):
    return STALLWAIT_OP | (block << 15) | cond


def _info(instruction):
    return TensixInstructionDecoder.getInstructionInfo(instruction)


def test_every_block_bit_catches_a_stallwait():
    """The table row, bit by bit, against the instructions around it."""
    stallwait = _info(_stallwait())
    semwait = _info(_semwait())
    for bit in range(9):
        latched = WaitGate.LatchedInstruction("SEMWAIT", 0b01, 1 << bit, 0b1)
        assert latched.doesInstructionMatchBlockMask(stallwait), (
            f"STALLWAIT should be blocked by B{bit}"
        )
        # Its neighbour in the Sync Unit is *not*: ``SEMWAIT``'s row is ticked
        # in B1 only, which is the asymmetry SEMWAIT.md's "highly recommended
        # ... include bit B1" note exists for.
        assert latched.doesInstructionMatchBlockMask(semwait) == (bit == 1)


def _tile():
    from tt_sim.device.tiles import TensixTile

    return TensixTile(18, 18, 1, 1, profile=WORMHOLE_PROFILE)


def test_a_stallwait_does_not_forget_an_unsatisfied_semwait():
    """End to end on a tile: the ``pack_untilize_dest_init`` ordering.

    A thread issues ``SEMWAIT`` on an empty semaphore with a B0 block mask --
    ``tile_regs_wait()`` exactly -- and then the ``STALLWAIT`` that opens
    ``_llk_init_packer_dest_offset_registers_``. The ``STALLWAIT`` must wait
    at the gate: while it does, the latched wait is still the ``SEMWAIT``, and
    the semaphore is still what releases the thread.
    """
    tile = _tile()
    cop = tile.tensix_coprocessor
    thread = cop.getThread(1)
    gate = thread.wait_gate
    sync = cop.getBackend().getSyncUnit()
    clocks = cop.getClocks()

    def tick(cycle, count=1):
        for _ in range(count):
            for clockable in clocks:
                clockable.clock_tick(cycle)
            cycle += 1
        return cycle

    cycle = 0
    thread.write(0, conv_to_bytes(_semwait(), 4))
    while gate.latchedWaitInstruction is None:
        cycle = tick(cycle)
        assert cycle < 20, "the SEMWAIT never reached the Sync Unit"
    assert gate.latchedWaitInstruction.opcode == "SEMWAIT"
    assert sync.getSemaphore(0).value == 0

    thread.write(0, conv_to_bytes(_stallwait(), 4))
    cycle = tick(cycle, 40)

    # The STALLWAIT is held at the gate, and the wait that holds it is still
    # the SEMWAIT -- semaphore mode, on semaphore 0.
    assert gate.latch_wait
    assert gate.latchedWaitInstruction is not None
    assert gate.latchedWaitInstruction.opcode == "SEMWAIT"
    assert gate.latchedWaitInstruction.isSemaphoreMode()
    assert gate.latchedWaitInstruction.getSemaphoresToCheck() == [0]

    # Posting the semaphore is what releases it, and only then does the
    # STALLWAIT execute -- proving it was queued behind the wait rather than
    # dropped. Its own condition (C10) is unmet out of reset, so the latch it
    # leaves behind is visible.
    sync.getSemaphore(0).value = 1
    for _ in range(40):
        cycle = tick(cycle)
        latched = gate.latchedWaitInstruction
        if latched is not None and latched.opcode == "STALLWAIT":
            break
    else:
        raise AssertionError("the STALLWAIT never executed after the semaphore")
    assert gate.latchedWaitInstruction.condition_mask == C10_SRCA_NOT_MATH


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "waitgate_stallwait_blocked_test OK: a STALLWAIT waits at the gate "
        "behind any latched wait instead of overwriting it"
    )


if __name__ == "__main__":
    main()
