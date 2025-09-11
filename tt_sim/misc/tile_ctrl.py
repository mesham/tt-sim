from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.conversion import conv_to_bytes


class TensixTileControl(MemMapable):
    def __init__(self):
        self.RISCV_DEBUG_REG_SOFT_RESET_0 = conv_to_bytes(0)

    def read(self, addr, size):
        if addr == 0x1B0:
            return self.RISCV_DEBUG_REG_SOFT_RESET_0
        return conv_to_bytes(0)
        # raise NotImplementedError(
        #        (
        #        f"Reading from address {hex(addr)} not yet supported by tensix "
        #        f"co-processor backend configuration"
        #
        #        )
        #    )

    def write(self, addr, value, size=None):
        if addr == 0x1B0:
            self.RISCV_DEBUG_REG_SOFT_RESET_0 = value
        # raise NotImplementedError(
        #        (
        #        f"Writing to address {hex(addr)} not yet supported by tensix "
        #        f"co-processor backend configuration"
        #
        #        )
        #    )

    def getSize(self):
        return 0xFFF
