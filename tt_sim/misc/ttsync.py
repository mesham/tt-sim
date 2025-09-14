from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import MemoryStall
from tt_sim.util.bits import get_bits
from tt_sim.util.conversion import (
    conv_to_bytes,
    conv_to_uint32,
)


class TTSync(MemMapable):
    def __init__(self, tile_ctrl, tensix_coprocessor, thread_id):
        self.tile_ctrl = tile_ctrl
        self.thread_id = thread_id
        self.tensix_coprocessor = tensix_coprocessor

    def read(self, address, size):
        if address == 0x0:
            return conv_to_bytes(0)
        else:
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
            if address == 0x4:
                if not self.tensix_coprocessor.CoprocessorDoneCheck(self.thread_id) or (
                    overrideEn and overrideBusy
                ):
                    return conv_to_bytes(0)  # an undefined value
                else:
                    return MemoryStall
            elif address == 0x8:
                if not self.tensix_coprocessor.MOPExpanderDoneCheck(self.thread_id) or (
                    overrideEn and overrideBusy
                ):
                    return conv_to_bytes(0)  # an undefined value
                else:
                    return MemoryStall

    def write(self, address, value, size):
        # Writes are discarded
        pass

    def getSize(self):
        return 0x1B
