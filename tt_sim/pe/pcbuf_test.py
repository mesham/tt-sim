"""The PC-buffer read-side handshake, against the pseudocode in the ISA docs.

``TensixTile/BabyRISCV/PCBufs.md`` (identical for Wormhole and Blackhole) gives
each access behaviour as pseudocode; the tests below are one-to-one with its
clauses, because **nothing in the tree exercises a PC buffer**. tt-metal
launches TRISCs through mailboxes, so a regression here would not be caught by
any replay guard, any example or any end-to-end run -- these tests are the only
thing standing between the implementation and the document.

The two behaviours under test:

*RISCV B read* -- a rendezvous that completes only when all three of

1. ``PCBuf[i].FIFO.empty``,
2. RISCV Ti blocked on a read of ``PC_BUF_BASE`` because the FIFO is empty,
3. Tensix Ti idle

hold at once, short-circuited by ``OverrideEn and OverrideBusy``.

*RISCV Ti read* -- pops the FIFO, waits while it is empty, and answers with
``OverrideValue`` instead if ``OverrideEn`` is set (``OverrideBusy`` is *not*
consulted on this side).

Run standalone (``python3 -m tt_sim.pe.pcbuf_test``) or under pytest.
"""

from tt_sim.memory.memory import AddressableMemory, MemorySpace, MemoryStall
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.misc.tile_ctrl import TensixTileControl
from tt_sim.pe.pcbuf import PCBuf
from tt_sim.pe.rv.babyriscv import BabyRISCV, BabyRISCVCoreType
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

BRISC = BabyRISCVCoreType.BRISC
TRISC = (
    BabyRISCVCoreType.TRISC0,
    BabyRISCVCoreType.TRISC1,
    BabyRISCVCoreType.TRISC2,
)

PC_BUF_BASE = 0xFFE80000
TILE_CTRL_BASE = 0xFFB12000
#: ``lw x1, 0(x2)``
LW_X1_0_X2 = 0x00012083


class _Coprocessor:
    """Just the ``CoprocessorDoneCheck`` the PCBuf needs.

    It returns True while the thread is still *busy* (see
    :class:`~tt_sim.misc.ttsync.TTSync`), so ``busy_threads`` is the set of
    threads with in-flight Tensix instructions.
    """

    def __init__(self, *busy_threads):
        self.busy_threads = set(busy_threads)

    def CoprocessorDoneCheck(self, thread_id):  # noqa: N802 - hardware name
        return thread_id in self.busy_threads


def _buf(buf_id=0, busy_threads=(), ctrl=None):
    return PCBuf(
        TensixTileControl() if ctrl is None else ctrl,
        buf_id,
        _Coprocessor(*busy_threads),
    )


def _set_override(buf, enable=False, busy=False, value=0):
    """Write ``RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE`` for one PCBuf.

    Per the doc's table the three fields sit at bit ``10*i`` (``OverrideEn``),
    ``10*i + 1`` (``OverrideBusy``) and ``10*i + 2`` (``uint8``
    ``OverrideValue``).
    """
    base = buf.buf_id * 10
    reg = conv_to_uint32(buf.tile_ctrl.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE)
    reg &= ~(0x3FF << base)
    reg |= (int(bool(enable)) | (int(bool(busy)) << 1) | ((value & 0xFF) << 2)) << base
    buf.tile_ctrl.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE = conv_to_bytes(reg)


def _park_trisc(buf):
    """Drive the TRISC-side read once so it blocks on the empty FIFO.

    This is condition 2, and the only way to establish it is the real one: the
    consumer actually attempting the read.
    """
    assert buf.read_from(TRISC[buf.buf_id]) is MemoryStall
    return buf


# --------------------------------------------------------------------------
# RISCV Ti read:  while FIFO.empty: if OverrideEn: return OverrideValue; wait
#                 return FIFO.pop()
# --------------------------------------------------------------------------


def test_trisc_read_of_empty_fifo_waits():
    """The empty case must wait, not pop.

    This used to be ``self.fifo.pop(0)`` unconditionally, so the first TRISC
    read to beat its BRISC producer killed the core with an ``IndexError``
    out of the memory subsystem -- the failure mode is a simulator crash, not
    a wrong answer, so it would be reported as a tt-sim bug rather than a
    kernel one.
    """
    assert _buf().read_from(TRISC[0]) is MemoryStall


def test_trisc_read_pops_in_fifo_order():
    buf = _buf()
    for value in (11, 22, 33):
        buf.write_from(BRISC, 0, conv_to_bytes(value))
    assert [conv_to_uint32(buf.read_from(TRISC[0])) for _ in range(3)] == [11, 22, 33]
    assert buf.read_from(TRISC[0]) is MemoryStall


def test_trisc_read_releases_when_brisc_pushes():
    """The wait is a wait, not a permanent refusal."""
    buf = _park_trisc(_buf())
    buf.write_from(BRISC, 0, conv_to_bytes(0xABCD))
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 0xABCD


def test_trisc_read_of_empty_fifo_returns_override_value():
    buf = _buf()
    _set_override(buf, enable=True, value=0xA5)
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 0xA5
    # Not a pop: the override answers every read while the FIFO stays empty.
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 0xA5


def test_trisc_read_ignores_override_busy():
    """The TRISC side consults ``OverrideEn`` alone.

    The BRISC side needs ``OverrideEn and OverrideBusy``; copying that pair
    over to this side would make an ``OverrideEn``-only configuration hang the
    consumer, and setting ``OverrideBusy`` alone would fabricate a value.
    """
    buf = _buf()
    _set_override(buf, enable=False, busy=True, value=0xA5)
    assert buf.read_from(TRISC[0]) is MemoryStall
    _set_override(buf, enable=True, busy=False, value=0xA5)
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 0xA5


def test_trisc_override_only_applies_to_an_empty_fifo():
    """``while FIFO.empty`` guards the override, so a queued value wins."""
    buf = _buf()
    _set_override(buf, enable=True, value=0xA5)
    buf.write_from(BRISC, 0, conv_to_bytes(7))
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 7
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 0xA5


def test_trisc_override_value_is_the_eight_bits_at_bit_two():
    """A misplaced shift on ``OverrideValue`` returns a plausible wrong number.

    Bits 2..9 of the buf's field, so 0xFF is the largest value and the next
    PCBuf's ``OverrideEn`` at bit 10 must not bleed into it.
    """
    buf = _buf()
    _set_override(buf, enable=True, value=0xFF)
    _set_override(PCBuf(buf.tile_ctrl, 1), enable=True, busy=True, value=0xFF)
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 0xFF


# --------------------------------------------------------------------------
# RISCV B read:  the three-condition rendezvous
# --------------------------------------------------------------------------


def test_brisc_read_waits_on_a_fresh_pcbuf():
    """Nothing has happened yet, so condition 2 cannot hold.

    This is the whole point of the change: the old implementation returned 0
    here, which makes ``while True: ... wait`` unconditionally fall through and
    lets RISCV B run past a TRISC that has not even started.
    """
    assert _buf().read_from(BRISC) is MemoryStall


def test_brisc_read_waits_while_the_fifo_is_not_empty():
    """Condition 1 alone, with 2 and 3 satisfied."""
    buf = _park_trisc(_buf())
    buf.write_from(BRISC, 0, conv_to_bytes(1))
    assert buf.read_from(BRISC) is MemoryStall


def test_brisc_read_waits_until_the_trisc_is_blocked_on_the_read():
    """Condition 2 alone, with 1 and 3 satisfied, then released."""
    buf = _buf()
    assert buf.read_from(BRISC) is MemoryStall
    _park_trisc(buf)
    assert conv_to_uint32(buf.read_from(BRISC)) == 0


def test_brisc_read_waits_while_the_tensix_thread_is_busy():
    """Condition 3 alone, with 1 and 2 satisfied, then released.

    Without this the handshake would let RISCV B proceed while thread ``i``
    still has Tensix instructions in flight -- the exact race
    ``CoprocessorDoneCheck`` exists to close.
    """
    buf = _park_trisc(_buf(busy_threads=(0,)))
    assert buf.read_from(BRISC) is MemoryStall
    buf.tensix_coprocessor.busy_threads.clear()
    assert conv_to_uint32(buf.read_from(BRISC)) == 0


def test_brisc_read_returns_zero_when_all_three_hold():
    buf = _park_trisc(_buf())
    assert conv_to_uint32(buf.read_from(BRISC)) == 0


def test_brisc_read_checks_its_own_tensix_thread():
    """PCBuf ``i`` waits on Tensix thread ``i``, not on thread 0."""
    buf = _park_trisc(_buf(buf_id=2, busy_threads=(0, 1)))
    assert conv_to_uint32(buf.read_from(BRISC)) == 0
    busy = _park_trisc(_buf(buf_id=2, busy_threads=(2,)))
    assert busy.read_from(BRISC) is MemoryStall


def test_brisc_read_override_short_circuits_the_rendezvous():
    """``OverrideEn and OverrideBusy`` returns 0 before anything else is read."""
    buf = _buf(busy_threads=(0,))
    buf.write_from(BRISC, 0, conv_to_bytes(1))
    assert buf.read_from(BRISC) is MemoryStall
    _set_override(buf, enable=True, busy=True)
    assert conv_to_uint32(buf.read_from(BRISC)) == 0


def test_brisc_read_override_needs_both_bits():
    """Either bit alone leaves the rendezvous in force.

    ``OverrideEn`` on its own reaching the ``return 0`` would silently disable
    every PC-buffer handshake in a tile that only meant to arm the override.
    """
    buf = _buf(busy_threads=(0,))
    _set_override(buf, enable=True, busy=False)
    assert buf.read_from(BRISC) is MemoryStall
    _set_override(buf, enable=False, busy=True)
    assert buf.read_from(BRISC) is MemoryStall


def test_brisc_read_uses_its_own_override_bits():
    """Ten bits apart: PCBuf 2's fields are at bits 20/21, not 0/1."""
    ctrl = TensixTileControl()
    buf0 = _buf(buf_id=0, ctrl=ctrl)
    buf2 = _buf(buf_id=2, ctrl=ctrl)
    _set_override(buf0, enable=True, busy=True)
    assert conv_to_uint32(buf0.read_from(BRISC)) == 0
    assert buf2.read_from(BRISC) is MemoryStall
    _set_override(buf2, enable=True, busy=True)
    assert conv_to_uint32(buf2.read_from(BRISC)) == 0


def test_the_blocked_flag_clears_once_the_trisc_is_served():
    """Condition 2 must be re-established for each rendezvous.

    A sticky flag is the subtle failure: the first handshake looks right and
    every later one completes early, against a TRISC that is off doing work.
    """
    buf = _park_trisc(_buf())
    buf.write_from(BRISC, 0, conv_to_bytes(1))
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 1
    assert buf.read_from(BRISC) is MemoryStall
    _park_trisc(buf)
    assert conv_to_uint32(buf.read_from(BRISC)) == 0


def test_the_blocked_flag_clears_when_the_override_serves_the_trisc():
    """The override answers the TRISC, so it is no longer blocked on the read."""
    buf = _park_trisc(_buf())
    _set_override(buf, enable=True, value=3)
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 3
    _set_override(buf, enable=False)
    assert buf.read_from(BRISC) is MemoryStall


def test_the_documented_handshake_end_to_end():
    """Producer pushes work, consumer drains it, producer rejoins."""
    buf = _park_trisc(_buf(busy_threads=(0,)))
    for value in (1, 2):
        buf.write_from(BRISC, 0, conv_to_bytes(value))
    # FIFO non-empty: RISCV B waits however long the consumer takes.
    assert buf.read_from(BRISC) is MemoryStall
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 1
    assert buf.read_from(BRISC) is MemoryStall
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 2
    # Drained, but the consumer is off issuing Tensix work.
    assert buf.read_from(BRISC) is MemoryStall
    _park_trisc(buf)
    assert buf.read_from(BRISC) is MemoryStall
    buf.tensix_coprocessor.busy_threads.clear()
    assert conv_to_uint32(buf.read_from(BRISC)) == 0


# --------------------------------------------------------------------------
# Writes -- unchanged behaviour, pinned because the read side now depends on it
# --------------------------------------------------------------------------


def test_trisc_writes_are_discarded():
    buf = _buf()
    buf.write_from(TRISC[0], 0, conv_to_bytes(1))
    assert buf.fifo == []


def test_the_fifo_is_unbounded_on_purpose():
    """``FIFO.full`` is never true here, and that is a deliberate choice.

    The doc publishes the depth (16) but overflows into "shared buffers within
    the RISCV B memory subsystem" whose capacity it does not publish. Bounding
    at 16 would invent back-pressure the hardware does not have, which for a
    queue depth is the over-charging direction. See ``ROADMAP.md`` §3 and the
    module docstring of ``tt_sim/pe/pcbuf.py``; if this test is ever changed,
    that decision is what is being reversed.
    """
    buf = _buf()
    for value in range(64):
        buf.write_from(BRISC, 0, conv_to_bytes(value))
    assert len(buf.fifo) == 64
    assert conv_to_uint32(buf.read_from(TRISC[0])) == 0


# --------------------------------------------------------------------------
# Through the real memory path: a baby core executing a real ``lw``
# --------------------------------------------------------------------------


class _CoreMemory(MemorySpace):
    pass


def _core(core_type, buf, ctrl):
    """A baby core whose ``lw x1, 0(x2)`` targets ``PC_BUF_BASE``."""
    memory_map = MemoryMap()
    l1 = AddressableMemory(0x8000)
    memory_map[AddressRange(0x0, l1.getSize())] = l1
    memory_map[AddressRange(TILE_CTRL_BASE, ctrl.getSize())] = ctrl
    memory_map[AddressRange(PC_BUF_BASE, buf.getSize())] = buf
    # safe=False so the reset-PC override read at 0xFFEF0284 answers 0 rather
    # than needing a whole Tensix backend behind it.
    core = BabyRISCV(core_type, [_CoreMemory(memory_map, safe=False)])
    core.start()
    l1.write(core.get_start_address(), conv_to_bytes(LW_X1_0_X2))
    core.register_file[2].write(conv_to_bytes(PC_BUF_BASE))
    return core


def test_a_stalled_load_does_not_retire_and_is_re_attempted():
    """The end-to-end claim: ``MemoryStall`` really parks the issuing core.

    ``PCBuf.read`` identifies its caller by walking the stack for a
    :class:`BabyRISCV`, so this is also the only test that covers the BRISC /
    TRISC discrimination through the real ``MemorySpace`` path rather than
    calling ``read_from`` directly.
    """
    ctrl = TensixTileControl()
    buf = _buf(ctrl=ctrl)
    brisc = _core(BRISC, buf, ctrl)
    trisc = _core(TRISC[0], buf, ctrl)
    brisc_pc = brisc.pc_register.read_uint()
    trisc_pc = trisc.pc_register.read_uint()

    # Both sides wait: the FIFO is empty and nobody is parked on it yet.
    brisc.clock_tick(1)
    assert brisc.pc_register.read_uint() == brisc_pc
    trisc.clock_tick(1)
    assert trisc.pc_register.read_uint() == trisc_pc

    # The TRISC's stalled read is condition 2, so the BRISC's load now retires.
    brisc.clock_tick(2)
    assert brisc.pc_register.read_uint() == brisc_pc + 4
    assert conv_to_uint32(brisc.register_file[1].read()) == 0

    # And a value pushed by the BRISC releases the TRISC's load.
    buf.write_from(BRISC, 0, conv_to_bytes(0x2A))
    trisc.clock_tick(3)
    assert trisc.pc_register.read_uint() == trisc_pc + 4
    assert conv_to_uint32(trisc.register_file[1].read()) == 0x2A


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
