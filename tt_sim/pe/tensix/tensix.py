from abc import ABC
from enum import IntEnum

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import VisibleMemory
from tt_sim.pe.pe import ProcessingElement
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class XMOV_DIRECTION(IntEnum):
    XMOV_L0_TO_L1 = 0
    XMOV_L1_TO_L0 = 1
    XMOV_L0_TO_L0 = 2
    XMOV_L1_TO_L1 = 3


TENSIX_CFG_BASE = 0xFFEF0000
MEM_NCRISC_IRAM_BASE = 0xFFC00000


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
    def __init__(self):
        mover = Mover(self)
        self.backend = TensixBackend(mover)
        self.addressable_memory = None

    def setAddressableMemory(self, addressable_memory):
        if len(addressable_memory) == 1:
            self.addressable_memory = addressable_memory[0]
        else:
            self.addressable_memory = VisibleMemory.merge(*addressable_memory)

    def getBackend(self):
        return self.backend

    def read(self, addr, size):
        return conv_to_bytes(0)

    def write(self, addr, value, size=None):
        return

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


class TensixBackend:
    def __init__(self, mover):
        self.mover = mover

    def getMover(self):
        return self.mover

    def issue_instruction(self, instruction):
        pass


class TensixBackendUnit(ABC):
    def __init__(self, tensix_coprocessor):
        self.tensix_coprocessor = tensix_coprocessor


class Mover(TensixBackendUnit):
    def __init__(self, tensix_coprocessor):
        super().__init__(tensix_coprocessor)

    def move(self, dst, src, count, mode):
        """
        This is based on the functional model description at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/Mover.md
        """
        assert self.tensix_coprocessor.addressable_memory is not None
        if mode == XMOV_DIRECTION.XMOV_L1_TO_L1 or mode == XMOV_DIRECTION.XMOV_L0_TO_L1:
            # In the "_TO_L1" modes, dst must be an address in L1.
            assert dst < 1024 * 1464
        else:
            if dst <= 0xFFFF:
                dst += TENSIX_CFG_BASE
            elif 0x40000 <= dst and dst <= 0x4FFFF:
                dst = (dst - 0x40000) + MEM_NCRISC_IRAM_BASE
            else:
                dst = None  # Operation still happens, but the writes get discarded.

            if (dst & 0xFFFF) + count > 0x10000:
                raise NotImplementedError(
                    "Can not access more than one region at a time"
                )

        # Perform the operation.
        if mode == XMOV_DIRECTION.XMOV_L1_TO_L1 or mode == XMOV_DIRECTION.XMOV_L1_TO_L0:
            # In the "L1_TO_" modes, a memcpy is done, and src must be an address in L1.
            if src >= (1024 * 1464):
                raise NotImplementedError("")
            # print(f"Write to {hex(dst)} from {hex(src)} elements {hex(count)}")
            self.tensix_coprocessor.addressable_memory.write(
                dst, self.tensix_coprocessor.addressable_memory.read(src, count)
            )
        else:
            # In the "L0_TO_" modes, a memset is done.
            zero_val = conv_to_bytes(0, count)
            self.tensix_coprocessor.addressable_memory.write(dst, zero_val)
