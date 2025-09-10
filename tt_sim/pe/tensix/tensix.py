from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.pe.pe import ProcessingElement
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


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
    CFG_STATE_SIZE = 47
    THD_STATE_SIZE = 57

    def __init__(self, tensix_cp):
        self.tensix_cp = tensix_cp
        self.config = [[0] * TensixBackendConfiguration.CFG_STATE_SIZE * 4] * 2
        self.threadConfig = [[0] * TensixBackendConfiguration.THD_STATE_SIZE] * 3

    def read(self, addr, size):
        threadConfigStart = TensixBackendConfiguration.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = TensixBackendConfiguration.CFG_STATE_SIZE * 4
            second_idx = 1 if idx > TensixBackendConfiguration.CFG_STATE_SIZE * 4 else 0
            first_idx = int(idx - (each_config_size * second_idx))
            print(f"Read tensix {second_idx}, {first_idx}")
            return conv_to_bytes(self.config[second_idx][first_idx])
        else:
            idx = idx - threadConfigStart
            second_idx = idx / TensixBackendConfiguration.THD_STATE_SIZE
            return conv_to_bytes(
                self.threadConfig[second_idx][
                    idx - ((TensixBackendConfiguration.THD_STATE_SIZE) * second_idx)
                ]
            )

    def write(self, addr, value, size=None):
        threadConfigStart = TensixBackendConfiguration.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = TensixBackendConfiguration.CFG_STATE_SIZE * 4
            second_idx = 1 if idx > each_config_size else 0
            first_idx = int(idx - (each_config_size * second_idx))
            self.config[second_idx][first_idx] = conv_to_uint32(value)
        else:
            idx = idx - threadConfigStart
            second_idx = int(idx / TensixBackendConfiguration.THD_STATE_SIZE)
            self.threadConfig[second_idx][
                idx - ((TensixBackendConfiguration.THD_STATE_SIZE) * second_idx)
            ] = conv_to_uint32(value)

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
