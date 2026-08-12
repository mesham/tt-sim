"""The three BRISC -> TRISC PC buffers of a Tensix tile.

Implements ``TensixTile/BabyRISCV/PCBufs.md`` from
https://github.com/tenstorrent/tt-isa-documentation. The page is the same for
Wormhole and Blackhole (Blackhole's carries an explicit "PCBufs in Blackhole
are the same as those in Wormhole" tip), so there is nothing per-arch here.

Each PCBuf is a FIFO of 32-bit values from RISCV B to one of RISCV T0/T1/T2,
plus ten configuration bits in ``RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE``.
NCRISC, the NoC and the Tensix coprocessor cannot reach them.

The document gives all four access behaviours as pseudocode, quoted verbatim.
RISCV B write::

    while PCBuf[i].FIFO.full:
      wait
    PCBuf[i].FIFO.push(write_value)

RISCV B read::

    while True:
      if PCBuf[i].OverrideEn:
        if PCBuf[i].OverrideBusy:
          return 0
      if PCBuf[i].FIFO.empty:
        if RISCV Ti is waiting to read from PC_BUF_BASE:
          if Tensix Ti is idle:
            return 0
      wait

RISCV Ti write is discarded, and RISCV Ti read::

    while PCBuf[i].FIFO.empty:
      if PCBuf[i].OverrideEn:
        return PCBuf[i].OverrideValue
      wait
    return PCBuf[i].FIFO.pop()

The page also spells the RISCV B read's three-way wait out in prose: it waits
for all three of

1. ``PCBuf[i].FIFO.empty``, i.e. for RISCV Ti to have popped all the values
   pushed by RISCV B;
2. RISCV Ti to be performing a read against ``PC_BUF_BASE`` and be blocked
   waiting due to ``PCBuf[i].FIFO`` being empty;
3. Tensix Ti to be idle, i.e. for the Tensix coprocessor to not have any
   in-flight instructions from Tensix thread Ti -- the same condition
   ``CoprocessorDoneCheck`` of TTSync exposes to RISCV Ti.

So a RISCV B read is a *rendezvous*: it completes only once the FIFO has
drained, the consumer is parked on the empty FIFO, and that thread's Tensix
instructions have all retired. Returning 0 immediately -- what this used to do
-- makes the rendezvous vacuous and lets RISCV B run ahead of a TRISC that has
not finished.

``wait`` here is modelled by returning :class:`MemoryStall`, the same mechanism
:class:`~tt_sim.misc.mailbox.Mailbox` and :class:`~tt_sim.misc.ttsync.TTSync`
use: the memory request is not serviced and the issuing core re-attempts it on
the next cycle, which matches the document's note that the waits "happen within
the memory subsystem" and hold up that memory region rather than the core's
whole pipeline.

**The FIFO is deliberately unbounded**, so ``PCBuf[i].FIFO.full`` is never true
and a RISCV B write never waits. The document publishes the queue depth exactly
(16 32-bit values) but the very next sentence puts the overflow in "shared
buffers within the RISCV B memory subsystem", whose capacity is published
nowhere: stalling at 16 would invent back-pressure the hardware does not have.
See ``ROADMAP.md`` §3.
"""

import sys

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import MemoryStall
from tt_sim.pe.rv.babyriscv import BabyRISCV, BabyRISCVCoreType
from tt_sim.util.bits import get_bits
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class PCBuf(MemMapable):
    #: Bits per PCBuf in ``RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE``: PCBuf[i]'s
    #: ``OverrideEn`` is at bit ``10*i``, ``OverrideBusy`` at ``10*i + 1``, and
    #: the ``uint8`` ``OverrideValue`` at ``10*i + 2``.
    OVERRIDE_BITS_PER_BUF = 10

    def __init__(self, tile_ctrl, buf_id, tensix_coprocessor=None):
        # See the module docstring: unbounded on purpose, so a RISCV B write
        # never has to wait for space.
        self.fifo = []
        self.buf_id = buf_id
        self.tile_ctrl = tile_ctrl
        # Needed for condition 3 of the RISCV B read. None only in isolated
        # unit tests that never exercise that condition.
        self.tensix_coprocessor = tensix_coprocessor
        # Condition 2 of the RISCV B read: True while RISCV Ti is parked on a
        # read of PC_BUF_BASE that found the FIFO empty. Set when such a read
        # stalls, cleared as soon as one completes (by a pop or by the
        # override). Nothing else can observe it, because RISCV Ti is blocked
        # in the memory subsystem until its read is serviced.
        self.trisc_blocked_on_read = False

    # -- configuration bits ------------------------------------------------

    def _override_config(self):
        """``(OverrideEn, OverrideBusy, OverrideValue)`` for this PCBuf."""
        reg = conv_to_uint32(self.tile_ctrl.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE)
        base = self.buf_id * PCBuf.OVERRIDE_BITS_PER_BUF
        override_en = bool(get_bits(reg, base, base))
        override_busy = bool(get_bits(reg, base + 1, base + 1))
        override_value = get_bits(reg, base + 2, base + 9)
        return override_en, override_busy, override_value

    def _tensix_thread_idle(self):
        """Condition 3: no in-flight Tensix instructions from thread ``i``.

        ``CoprocessorDoneCheck`` answers "is the thread still busy", so idle is
        its negation -- the same reading :class:`~tt_sim.misc.ttsync.TTSync`
        takes of it.
        """
        if self.tensix_coprocessor is None:
            return True
        return not self.tensix_coprocessor.CoprocessorDoneCheck(self.buf_id)

    # -- the four documented behaviours ------------------------------------

    def read_from(self, core_type, address=0, size=4):
        """A PCBuf read issued by ``core_type``. See the module docstring."""
        if core_type == BabyRISCVCoreType.BRISC:
            return self._read_brisc()
        return self._read_trisc()

    def _read_brisc(self):
        override_en, override_busy, _ = self._override_config()
        if override_en and override_busy:
            # The override short-circuits the whole rendezvous, exactly as it
            # does for TTSync's CoprocessorDoneCheck.
            return conv_to_bytes(0)
        if (
            not self.fifo  # 1. FIFO drained
            and self.trisc_blocked_on_read  # 2. RISCV Ti parked on the read
            and self._tensix_thread_idle()  # 3. Tensix Ti idle
        ):
            return conv_to_bytes(0)  # an undefined value
        return MemoryStall

    def _read_trisc(self):
        if not self.fifo:
            override_en, _, override_value = self._override_config()
            if override_en:
                # Note the asymmetry with the RISCV B read, which is in the
                # document: the TRISC side consults OverrideEn alone, not
                # OverrideBusy.
                self.trisc_blocked_on_read = False
                return conv_to_bytes(override_value)
            self.trisc_blocked_on_read = True
            return MemoryStall
        self.trisc_blocked_on_read = False
        return self.fifo.pop(0)

    def write_from(self, core_type, address, value, size=None):
        """A PCBuf write issued by ``core_type``. See the module docstring."""
        if core_type == BabyRISCVCoreType.BRISC:
            self.fifo.append(value)
        # Anything else (RISCV Ti) is discarded.

    # -- MemMapable --------------------------------------------------------

    def read(self, address, size):
        caller = self.find_caller_instance(BabyRISCV)
        assert caller is not None
        return self.read_from(caller.core_type, address, size)

    def write(self, address, value, size=None):
        caller = self.find_caller_instance(BabyRISCV)
        assert caller is not None
        self.write_from(caller.core_type, address, value, size)

    def find_caller_instance(self, cls_type):
        # A plain frame walk rather than inspect.stack(): the latter builds a
        # FrameInfo (and reads source lines) for every frame on the stack, and
        # a RISCV B read now re-attempts once per cycle for as long as the
        # rendezvous takes.
        frame = sys._getframe(1)
        while frame is not None:
            local_self = frame.f_locals.get("self")
            if isinstance(local_self, cls_type):
                return local_self
            frame = frame.f_back
        return None

    def getSize(self):
        # This is for BRISC
        return 0xFFFF
