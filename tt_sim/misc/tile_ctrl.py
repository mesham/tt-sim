import os
import sys

from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.misc.perf_counters import PERF_CNT_OFFSETS
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

#: ``RISCV_DEBUG_REG_*`` offsets that behave as plain read-what-you-wrote
#: stores, with their names from tt-metal's ``tensix.h`` (the union of the
#: Wormhole and Blackhole headers; where the two disagree on a name both are
#: given). Every one of these is either pure configuration whose reset value of
#: zero genuinely means "off"/"no override", or a register tt-sim models
#: elsewhere by reading this store back — the reset-PC block at 0x228-0x23C is
#: read by ``BabyRISCV._blackhole_reset_pc``, and a Blackhole core booting from
#: its default PC *depends* on an unwritten override register reading zero.
#: That dependency is why the fallthrough below is an allowlist rather than a
#: blanket raise.
STORE_REGISTERS = {
    0x048: "DBG_L1_MEM_REG0",
    0x04C: "DBG_L1_MEM_REG1",
    0x050: "DBG_L1_MEM_REG2",
    0x054: "DBG_BUS_CTRL / DBG_BUS_CNTL_REG",
    0x058: "TENSIX_CREG_READ / CFGREG_RD_CNTL",
    0x060: "DBG_ARRAY_RD_EN",
    0x064: "DBG_ARRAY_RD_CMD",
    0x070: "CG_CTRL_HYST0",
    0x074: "CG_CTRL_HYST1",
    0x07C: "CG_CTRL_HYST2",
    0x080: "RISC_DBG_CNTL_0",
    0x084: "RISC_DBG_CNTL_1",
    0x0A0: "DBG_INSTRN_BUF_CTRL0",
    0x0A4: "DBG_INSTRN_BUF_CTRL1",
    0x0AC: "STOCH_RND_MASK0",
    0x0B0: "STOCH_RND_MASK1",
    0x0B8: "ETH_RISC_PREFECTH_CTRL",
    0x0BC: "ETH_RISC_PREFECTH_PC",
    0x1C0: "BREAKPOINT_CTRL",
    0x1C8: "BREAKPOINT_DATA",
    0x1D0: "ECC_CTRL",
    0x1E0: "WATCHDOG_TIMER / WDT",
    0x1E4: "WDT_CNTL",
    0x1FC: "TIMESTAMP_DUMP_CMD / TIMESTAMP",
    0x200: "TIMESTAMP_DUMP_CNTL",
    0x208: "TIMESTAMP_DUMP_BUF0_START_ADDR",
    0x20C: "TIMESTAMP_DUMP_BUF0_END_ADDR",
    0x210: "TIMESTAMP_DUMP_BUF1_START_ADDR",
    0x214: "TIMESTAMP_DUMP_BUF1_END_ADDR",
    0x21C: "DBG_L1_READBACK_OFFSET",
    0x220: "LFSR_HIT_MASK",
    0x224: "DISABLE_RESET",
    0x228: "TRISC0_RESET_PC",
    0x22C: "TRISC1_RESET_PC",
    0x230: "TRISC2_RESET_PC",
    0x234: "TRISC_RESET_PC_OVERRIDE",
    0x238: "NCRISC_RESET_PC",
    0x23C: "NCRISC_RESET_PC_OVERRIDE",
    0x240: "DEST_CG_CTRL",
    0x244: "CG_CTRL_EN",
    0x248: "CG_KICK",
}

#: Named ``RISCV_DEBUG_REG_*`` offsets that report *hardware state* rather than
#: storing software's own writes. tt-sim models none of them, and returning the
#: zero they were returning before is a fabricated all-clear: an unmodelled
#: status register is precisely the "silently-plausible generic store" the
#: roadmap flags. Reads of these are loud.
STATUS_REGISTERS = {
    0x05C: "DBG_RD_DATA / THREAD1_CREG_READ",
    0x06C: "DBG_ARRAY_RD_DATA",
    0x078: "TENSIX_CREG_RDDATA / THREAD0_CREG_RDDATA / CFGREG_RDDATA",
    0x088: "RISC_DBG_STATUS_0",
    0x08C: "RISC_DBG_STATUS_1",
    0x094: "DBG_INVALID_INSTRN",
    0x0A8: "DBG_INSTRN_BUF_STATUS",
    0x0B4: "FPU_STICKY_BITS",
    0x1C4: "BREAKPOINT_STATUS",
    0x1D4: "ECC_STATUS",
    0x1E8: "WDT_STATUS",
    0x204: "TIMESTAMP_DUMP_STATUS / TIMESTAMP_STATUS",
}

#: Set truthy to downgrade an unmodelled-register read from a raise to a
#: one-shot warning. The mirror of ``TT_SIM_DISABLE_ALIGNMENT_CHECKS``: the
#: check defaults on, and the escape hatch exists because a raise inside a
#: memory read takes the whole simulator down rather than surfacing anywhere a
#: driver could catch it.
PERMISSIVE_ENV = "TT_SIM_PERMISSIVE_TILE_CTRL"

_TRUTHY = ("1", "true", "yes", "on")


def _permissive():
    return os.environ.get(PERMISSIVE_ENV, "").lower() in _TRUTHY


class UnmodelledTileRegisterError(LookupError):
    """A ``RISCV_DEBUG_REG_*`` read tt-sim cannot answer truthfully.

    Raised rather than answered with zero, because zero is a *plausible* value
    for every register in this window — a clear status, an idle unit, an
    unstalled thread — and a plausible wrong answer is worse than a loud one.
    """


class TensixTileControl(MemMapable, Clockable):
    def __init__(self, perf_counters=None):
        self.RISCV_DEBUG_REG_SOFT_RESET_0 = conv_to_bytes(0)
        self.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE = conv_to_bytes(0)
        self.RISCV_DEBUG_REG_DBG_FEATURE_DISABLE = conv_to_bytes(0)
        # Backing store for the :data:`STORE_REGISTERS` set — clock-gating
        # control (DEST_CG_CTRL/CG_CTRL_EN/CG_KICK at 0x240+), the reset-PC
        # override block, scratch/postcode registers. The functional sim does
        # not model clock gating, so these behave as plain read-what-you-wrote
        # registers, which is what the firmware's init sequence expects.
        self.regs = {}
        # The performance-counter block (RISCV_DEBUG_REG_PERF_CNT_*), which
        # lives inside this same register window. ``None`` on a tile with no
        # Tensix coprocessor behind it (the Ethernet tile), where every counter
        # read is declined rather than answered with a fabricated zero.
        self.perf_counters = perf_counters
        # Addresses already complained about, so a profiler sweeping the window
        # on every core warns once rather than once per read.
        self._warned = set()
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
        elif addr in PERF_CNT_OFFSETS:
            counters = self.perf_counters
            if counters is None:
                self._complain(
                    addr,
                    "RISCV_DEBUG_REG_PERF_CNT_* on a tile with no Tensix "
                    "coprocessor behind it",
                )
                return conv_to_bytes(0, 4)
            return conv_to_bytes(counters.read(addr, self.cycle_num), 4)
        elif addr in STORE_REGISTERS:
            return self.regs.get(addr, conv_to_bytes(0))
        else:
            self._complain(
                addr,
                STATUS_REGISTERS.get(addr, "no such register in tt-metal's tensix.h"),
            )
            return self.regs.get(addr, conv_to_bytes(0))

    def write(self, addr, value, size=None):
        if addr == 0x1B0:
            self.RISCV_DEBUG_REG_SOFT_RESET_0 = value
            if self.clock_owner is not None:
                self.clock_owner.wake()
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
        elif addr in PERF_CNT_OFFSETS:
            counters = self.perf_counters
            if counters is not None:
                counters.write(addr, conv_to_uint32(value), self.cycle_num)
        elif addr in STORE_REGISTERS or addr in STATUS_REGISTERS:
            self.regs[addr] = value
        else:
            # A write cannot return a wrong value to the kernel the way a read
            # can, so this warns rather than raising even in strict mode: the
            # hazard is a *side effect* going unmodelled, which is worth
            # surfacing but not worth aborting a run over.
            self._warn(addr, "no such register in tt-metal's tensix.h", "written to")
            self.regs[addr] = value

    def _complain(self, addr, why):
        """Refuse to answer an unmodelled register read with a plausible zero."""
        name = (
            STATUS_REGISTERS.get(addr)
            or STORE_REGISTERS.get(addr)
            or ("PERF_CNT_*" if addr in PERF_CNT_OFFSETS else "unnamed")
        )
        if not _permissive():
            raise UnmodelledTileRegisterError(
                f"RISCV_DEBUG_REG at offset {addr:#05x} ({name}) "
                f"is not modelled in tt-sim: {why}. Returning 0 would read as a "
                f"clear status / an idle unit / an unstalled thread, which is a "
                f"measurement tt-sim has not made. Implement it in "
                f"tt_sim/misc/tile_ctrl.py, or set {PERMISSIVE_ENV}=1 to "
                f"downgrade this to a warning and get the old zero back."
            )
        self._warn(addr, why, "read")

    def _warn(self, addr, why, verb):
        if addr in self._warned:
            return
        self._warned.add(addr)
        print(
            f"tt-sim WARNING: RISCV_DEBUG_REG at offset {addr:#05x} is not "
            f"modelled and was {verb}: {why}.",
            file=sys.stderr,
        )

    def getSize(self):
        return 0xFFF
