from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.conversion import conv_to_bytes


class TensixTileControl(MemMapable, Clockable):
    def __init__(self):
        self.RISCV_DEBUG_REG_SOFT_RESET_0 = conv_to_bytes(0)
        self.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE = conv_to_bytes(0)
        self.RISCV_DEBUG_REG_DBG_FEATURE_DISABLE = conv_to_bytes(0)
        # Backing store for RISCV_DEBUG_REG_* registers without dedicated
        # semantics — clock-gating control (DEST_CG_CTRL/CG_CTRL_EN/CG_KICK at
        # 0x240+), scratch/postcode registers, etc. The functional sim does not
        # model clock gating, so these behave as plain read-what-you-wrote
        # registers, which is what the firmware's init sequence expects.
        self.regs = {}
        # The high half of the wall clock, latched by a read of
        # RISCV_DEBUG_REG_WALL_CLOCK_L (0x1F0) and returned by ..._H (0x1F8).
        # Initialised here because 0x1F8 may be read first: tt-metal's
        # kernel_profiler reads 0x1F0 then 0x1F4 and never trips it, but
        # `realtime_profiler.hpp` reads 0x1F8 directly, and an AttributeError
        # in a memory read kills the server rather than raising anywhere a
        # driver could see it.
        self.counter_high_at = 0
        # Owning tile's TileClock, set by TTDeviceTile._bind_clock. A write to
        # RISCV_DEBUG_REG_SOFT_RESET_0 can bring a core out of reset, so it
        # must wake the tile whatever path the write arrived by.
        self.clock_owner = None

    @property
    def cycle_num(self):
        """The wall clock behind ``RISCV_DEBUG_REG_WALL_CLOCK_*``, sampled lazily.

        Phase 2 of ``docs/plans/event-driven-pump.md``. This used to be
        latched by ``clock_tick`` on every cycle of every tile — the one thing
        that still cost a dormant tile anything, and impossible to keep
        correct once Phase 4 lets the pump skip cycles outright. It is a pure
        function of the cycle number, so it is cheaper and strictly more
        robust to compute it when somebody actually reads it.

        The value is deliberately ``current_cycle - 1``, which reproduces the
        latched semantics exactly: ``tile_ctrl`` was ticked *after* the tile's
        other components, so a core reading the register during cycle ``c``
        saw the value stored at the end of cycle ``c - 1``, and a host reading
        it between ``run`` calls saw the last cycle executed.
        """
        owner = self.clock_owner
        if owner is None:
            return 0
        cycle = owner.current_cycle
        return cycle - 1 if cycle > 0 else 0

    def clock_tick(self, cycle_num):
        # Nothing to latch since Phase 2 — see ``cycle_num`` above. Kept so the
        # class stays a Clockable for drivers that register it by hand; the
        # in-tree tiles no longer list it in ``get_clocks()``.
        pass

    def is_clock_idle(self):
        # clock_tick is a no-op, so ticking this can never change anything.
        return True

    def read(self, addr, size):
        if addr == 0x1B0:
            return self.RISCV_DEBUG_REG_SOFT_RESET_0
        elif addr == 0x1F0:
            # RISCV_DEBUG_REG_WALL_CLOCK_L
            self.counter_high_at = self.cycle_num >> 32
            return conv_to_bytes(self.cycle_num & 0xFFFFFFFF, 4)
        elif addr == 0x1F4:
            # RISCV_DEBUG_REG_WALL_CLOCK_L+4
            return conv_to_bytes(self.cycle_num >> 32, 4)
        elif addr == 0x1F8:
            # RISCV_DEBUG_REG_WALL_CLOCK_H
            return conv_to_bytes(self.counter_high_at, 4)
        elif addr == 0x090:
            return self.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE
        elif addr == 0x68:
            return self.RISCV_DEBUG_REG_DBG_FEATURE_DISABLE
        else:
            return self.regs.get(addr, conv_to_bytes(0))

    def write(self, addr, value, size=None):
        if addr == 0x1B0:
            self.RISCV_DEBUG_REG_SOFT_RESET_0 = value
            if self.clock_owner is not None:
                self.clock_owner.awake = True
        elif addr == 0x1F0:
            # RISCV_DEBUG_REG_WALL_CLOCK_L
            self.counter_high_at = self.cycle_num >> 32
        elif addr == 0x1F4 or addr == 0x1F8:
            # nop
            pass
        elif addr == 0x090:
            self.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE = value
        elif addr == 0x68:
            self.RISCV_DEBUG_REG_DBG_FEATURE_DISABLE = value
        else:
            self.regs[addr] = value

    def getSize(self):
        return 0xFFF
