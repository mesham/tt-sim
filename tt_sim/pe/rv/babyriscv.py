from enum import Enum

from tt_sim.pe.rv.rv32 import RV32IM_TT
from tt_sim.util.conversion import conv_to_uint32, get_nth_bit


class BabyRISCVCoreType(Enum):
    BRISC = 0
    NCRISC = 1
    TRISC0 = 2
    TRISC1 = 3
    TRISC2 = 4


class BabyRISCV(RV32IM_TT):
    CORE_TYPE_TO_SOFT_RESET_BIT = {
        BabyRISCVCoreType.BRISC: 11,
        BabyRISCVCoreType.NCRISC: 18,
        BabyRISCVCoreType.TRISC0: 12,
        BabyRISCVCoreType.TRISC1: 13,
        BabyRISCVCoreType.TRISC2: 14,
    }

    def __init__(self, core_type, memory_spaces, snoop=False):
        self.core_type = core_type
        self.soft_active = False
        if core_type == BabyRISCVCoreType.BRISC:
            core_id = 0
            start_addr = 0x3780  # 0x0
        elif core_type == BabyRISCVCoreType.NCRISC:
            core_id = 4
            start_addr = 0x4D80  # 0x12000
        elif core_type == BabyRISCVCoreType.TRISC0:
            core_id = 1
            start_addr = 0x5580  # 0x06000
        elif core_type == BabyRISCVCoreType.TRISC1:
            core_id = 2
            start_addr = 0x5B80  # 0x0A000
        elif core_type == BabyRISCVCoreType.TRISC2:
            core_id = 3
            start_addr = 0x6180  # 0x0E000
        super().__init__(start_addr, memory_spaces, [], snoop=snoop, core_id=core_id)

    def clock_tick(self, cycle_num):
        # These cores have a soft reset that they need to check
        reset_val = conv_to_uint32(self.visible_memory.read(0xFFB121B0, 4))
        is_in_reset = (
            get_nth_bit(
                reset_val, BabyRISCV.CORE_TYPE_TO_SOFT_RESET_BIT[self.core_type]
            )
            == 1
        )
        if not is_in_reset:
            if not self.soft_active:
                # About to be restarted, move into the resetted state
                self.initialise_core()
                self.soft_active = True
            super().clock_tick(cycle_num)
        else:
            if self.soft_active:
                self.soft_active = False
