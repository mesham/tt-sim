from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.pe import ProcessingElement
from tt_sim.util.conversion import conv_to_bytes


class TensixGPR(MemMapable):
    def __init__(self, tensix_cp):
        self.tensix_cp = tensix_cp

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


class TensixBackendConfiguration(MemMapable):
    def __init__(self, tensix_cp):
        self.tensix_cp = tensix_cp

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
        return 0xFFFF


class TensixCoProcessor(ProcessingElement, MemMapable):
    def read(self, addr, size):
        return conv_to_bytes(0)
        # raise NotImplementedError()

    def write(self, addr, value, size=None):
        return
        # raise NotImplementedError()

    def getSize(self):
        return 0x2FFFF

    def getRegisterFile(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def clock_tick(self):
        pass

    def reset(self):
        pass
