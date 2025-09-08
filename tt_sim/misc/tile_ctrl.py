from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.conversion import conv_to_bytes


class TensixTileControl(MemMapable):
    def __init__(self):
        pass

    def read(self, addr, size):
        return conv_to_bytes(0)
        # raise NotImplementedError(
        #        (
        #        f"Reading from address {hex(addr)} not yet supported by tensix "
        #        f"co-processor backend configuration"
        #
        #        )
        #    )

    def write(self, addr, value, size=None):
        return
        # raise NotImplementedError(
        #        (
        #        f"Writing to address {hex(addr)} not yet supported by tensix "
        #        f"co-processor backend configuration"
        #
        #        )
        #    )

    def getSize(self):
        return 0xFFF
