from tt_sim.memory.mem_mapable import MemMapable


class TDMA(MemMapable):
    def __init__(self):
        pass

    def read(self, addr, size):
        raise NotImplementedError()

    def write(self, addr, value, size=None):
        if addr == 0x24:
            pass
        else:
            raise NotImplementedError(
                f"Writing to tdma address {hex(addr)} not supported"
            )

    def getSize(self):
        return 0xFFF
