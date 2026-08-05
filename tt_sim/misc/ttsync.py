from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import MemoryStall
from tt_sim.trace import EventCategory, SyncEvent, get_bus
from tt_sim.util.bits import get_bits
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_uint32,
)


class TTSync(MemMapable):
    """The blocking words of a TRISC's PC buffer.

    A TRISC's PC buffer starts at ``0xFFE80000``; word 0 is the plain
    BRISC->TRISC FIFO (:class:`~tt_sim.pe.pcbuf.PCBuf`) and this object is
    mapped over the words above it, so **the offsets here are one word lower
    than the PC-buffer word index**: offset ``0x0`` is ``pc_buf_base[1]``,
    ``0x4`` is ``pc_buf_base[2]``.

    Which word does what is fixed by the two things that read them:

    * ``ckernel::tensix_sync()`` is ``store_blocking(&pc_buf_base[1], 0)`` and
      ``ckernel::mop_sync()`` is ``store_blocking(&pc_buf_base[2], 0)``
      (tt-metal ``tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/common/inc/
      ckernel.h``; ``store_blocking`` is a store followed by a load of the same
      address, and it is the *load* that blocks). Semaphores start at
      ``pc_buf_base[8]``, which is mapped elsewhere.
    * the vendor reference simulator agrees on the split
      (``ttsim`` ``src/tile.cpp``, ``tensix_pc_buf_rd32``): its
      ``TENSIX_PC_BUF_TENSIX_SYNC`` case stalls
      ``while inst_rd_ptr[pipe] != inst_wr_ptr[pipe]`` -- the thread's Tensix
      instruction FIFO -- and its ``TENSIX_PC_BUF_MOP_SYNC`` case returns
      immediately (that simulator expands MOPs instantly).

    Historically tt-sim had both checks one word too high: the FIFO-drain check
    sat on ``pc_buf_base[2]`` (so it answered ``mop_sync()``), the MOP-expander
    check on the unused ``pc_buf_base[3]``, and ``pc_buf_base[1]`` -- the word
    every ``tensix_sync()`` in the LLKs and in the TRISC firmware reads -- was
    an unconditional immediate return. ``tensix_sync()`` was therefore a no-op
    against tt-sim, and any kernel using it purely for ordering was unverifiable
    here. See ``docs/plans/matrix-unit-thread-contention.md``.

    Based on descriptions at
    https://github.com/tenstorrent/tt-isa-documentation (``TensixTile/BabyRISCV``).
    """

    #: ``pc_buf_base[1]`` -- ``ckernel::tensix_sync()``. Blocks until every
    #: Tensix instruction this thread has issued has retired.
    TENSIX_SYNC = 0x0
    #: ``pc_buf_base[2]`` -- ``ckernel::mop_sync()``. Blocks until the MOP
    #: expander has finished expanding this thread's outstanding MOPs.
    MOP_SYNC = 0x4

    def __init__(self, tile_ctrl, tensix_coprocessor, thread_id):
        self.tile_ctrl = tile_ctrl
        self.thread_id = thread_id
        self.tensix_coprocessor = tensix_coprocessor
        self.unit_id: tuple | None = None

    def _publish_sync(self, kind: str, detail: str = ""):
        bus = get_bus()
        if self.unit_id is None or not bus.is_enabled(EventCategory.SYNC):
            return
        bus.publish(
            SyncEvent(
                cycle=0,
                unit_id=self.unit_id,
                kind=kind,
                detail=detail,
            )
        )

    def _override_forces_ready(self):
        """``RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE`` short-circuiting the wait.

        Two bits per thread, ten bits apart: enable, then the value the busy
        signal is overridden with.
        """
        overrideEn = get_bits(
            conv_to_uint32(self.tile_ctrl.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE),
            self.thread_id * 10,
            self.thread_id * 10,
        )
        overrideBusy = get_bits(
            conv_to_uint32(self.tile_ctrl.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE),
            (self.thread_id * 10) + 1,
            (self.thread_id * 10) + 1,
        )
        return bool(overrideEn and overrideBusy)

    def read(self, address, size):
        # ``*DoneCheck`` return True while the thread is still BUSY, so the read
        # completes on the ``not`` arm and stalls (is retried next cycle) on the
        # other.
        if address == TTSync.TENSIX_SYNC:
            busy = self.tensix_coprocessor.CoprocessorDoneCheck(self.thread_id)
            kind = "ttsync_wait_coproc_done"
        elif address == TTSync.MOP_SYNC:
            busy = self.tensix_coprocessor.MOPExpanderDoneCheck(self.thread_id)
            kind = "ttsync_wait_mop_done"
        else:
            return conv_to_bytes(0)

        if not busy or self._override_forces_ready():
            self._publish_sync(kind, detail=f"thread={self.thread_id}")
            return conv_to_bytes(0)  # an undefined value
        return MemoryStall

    def write(self, address, value, size):
        # Writes are discarded
        pass

    def getSize(self):
        return 0x1B
