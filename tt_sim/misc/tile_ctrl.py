from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.conversion import conv_to_bytes


class TensixTileControl(MemMapable, Clockable):
    def __init__(self):
        self.RISCV_DEBUG_REG_SOFT_RESET_0 = conv_to_bytes(0)
        self.RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE = conv_to_bytes(0)
        self.RISCV_DEBUG_REG_DBG_FEATURE_DISABLE = conv_to_bytes(0)
        self.cycle_num = 0
        # Backing store for RISCV_DEBUG_REG_* registers without dedicated
        # semantics — clock-gating control (DEST_CG_CTRL/CG_CTRL_EN/CG_KICK at
        # 0x240+), scratch/postcode registers, etc. The functional sim does not
        # model clock gating, so these behave as plain read-what-you-wrote
        # registers, which is what the firmware's init sequence expects.
        self.regs = {}

    def clock_tick(self, cycle_num):
        self.cycle_num = cycle_num

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
