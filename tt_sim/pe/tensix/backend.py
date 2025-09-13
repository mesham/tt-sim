from abc import ABC
from enum import IntEnum

from tt_sim.device.clock import Clockable
from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.util.bits import extract_bits, get_nth_bit, int_to_bin_list
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TensixBackend:
    def __init__(self, tensix_instruction_decoder):
        self.tensix_instruction_decoder = tensix_instruction_decoder
        self.mover_unit = MoverUnit(self)
        self.sync_unit = TensixSyncUnit(self)
        self.matrix_unit = MatrixUnit(self)
        self.scalar_unit = ScalarUnit(self)
        self.vector_unit = VectorUnit(self)
        self.unpacker_units = [UnPackerUnit(self)] * 2
        self.packer_units = [PackerUnit(self)] * 4
        self.misc_unit = MiscellaneousUnit(self)
        self.backend_units = {
            "MATH": self.matrix_unit,
            "SFPU": self.vector_unit,
            "THCON": self.scalar_unit,
            "SYNC": self.sync_unit,
            "XMOV": self.mover_unit,
            "TDMA": self.misc_unit,
        }
        self.addressable_memory = None

    def getMoverUnit(self):
        return self.mover_unit

    def getSyncUnit(self):
        return self.sync_unit

    def setAddressableMemory(self, addressable_memory):
        self.addressable_memory = addressable_memory

    def getAddressableMemory(self):
        return self.addressable_memory

    def getClocks(self):
        return []

    def issueInstruction(self, instruction):
        instruction_info = self.tensix_instruction_decoder.getInstructionInfo(
            instruction
        )
        tgt_backend_unit = instruction_info["ex_resource"]
        # For now ignore, need to add this
        if tgt_backend_unit == "CFG":
            return
        if tgt_backend_unit != "NONE":
            if tgt_backend_unit == "UNPACK":
                unpacker = get_nth_bit(instruction, 23)
                self.unpacker_units[unpacker].issueInstruction(instruction)
            elif tgt_backend_unit == "PACK":
                packers_int = extract_bits(instruction, 4, 8)
                if packers_int == 0x0:
                    self.packer_units[0].issueInstruction(instruction)
                else:
                    packers = int_to_bin_list(packers_int, 4)
                    for idx, packer_bit in enumerate(packers):
                        if packer_bit:
                            # Working left to right, hence 3-idx as the first bit
                            # represents the highest number packer
                            self.packer_units[3 - idx].issueInstruction(instruction)
            else:
                assert tgt_backend_unit in self.backend_units
                print(f"{instruction_info['name']}: {tgt_backend_unit}")
                exit(0)
                self.backend_units[tgt_backend_unit].issueInstruction(instruction)


class TensixBackendUnit(Clockable, ABC):
    def __init__(self, backend):
        self.backend = backend
        self.instruction_buffer = []

    def issueInstruction(self, instruction):
        self.instruction_buffer.append(instruction)


class VectorUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass


class MatrixUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass


class ScalarUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass


class TensixSyncUnit(TensixBackendUnit, MemMapable):
    class TTSemaphore:
        def __init__(self):
            self.value = 0
            self.max = 0

    def __init__(self, backend):
        super().__init__(backend)
        self.semaphores = [TensixSyncUnit.TTSemaphore()] * 8

    def clock_tick(self, cycle_num):
        pass

    def read(self, addr, size):
        # Accesses semaphore[i].value, where each
        # entry is 32 bit
        idx = int(addr / 4)
        assert idx < 8
        return conv_to_bytes(self.semaphores[idx].value)

    def write(self, addr, value, size=None):
        """
        This is taken from the functional model code at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SyncUnit.md#semaphores
        """
        idx = int(addr / 4)
        assert idx < 8
        if conv_to_uint32(value) & 1:
            # This is like a SEMGET instruction
            if self.semaphores[idx].value > 0:
                self.semaphores[idx].value = -1
        else:
            # This is like a SEMPOST instruction
            if self.semaphores[idx].value < 15:
                self.semaphores[idx].value += 1

    def getSize(self):
        return 0xFFDF


class MiscellaneousUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass


class UnPackerUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass


class PackerUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass


class MoverUnit(TensixBackendUnit):
    class XMOV_DIRECTION(IntEnum):
        XMOV_L0_TO_L1 = 0
        XMOV_L1_TO_L0 = 1
        XMOV_L0_TO_L0 = 2
        XMOV_L1_TO_L1 = 3

    TENSIX_CFG_BASE = 0xFFEF0000
    MEM_NCRISC_IRAM_BASE = 0xFFC00000

    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass

    def move(self, dst, src, count, mode):
        """
        This is based on the functional model description at
        https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/Mover.md
        """
        assert self.backend.getAddressableMemory() is not None
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L0_TO_L1
        ):
            # In the "_TO_L1" modes, dst must be an address in L1.
            assert dst < 1024 * 1464
        else:
            if dst <= 0xFFFF:
                dst += MoverUnit.TENSIX_CFG_BASE
            elif 0x40000 <= dst and dst <= 0x4FFFF:
                dst = (dst - 0x40000) + MoverUnit.MEM_NCRISC_IRAM_BASE
            else:
                dst = None  # Operation still happens, but the writes get discarded.

            if (dst & 0xFFFF) + count > 0x10000:
                raise NotImplementedError(
                    "Can not access more than one region at a time"
                )

        # Perform the operation.
        if (
            mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L1
            or mode == MoverUnit.XMOV_DIRECTION.XMOV_L1_TO_L0
        ):
            # In the "L1_TO_" modes, a memcpy is done, and src must be an address in L1.
            if src >= (1024 * 1464):
                raise NotImplementedError("")
            # print(f"Write to {hex(dst)} from {hex(src)} elements {hex(count)}")
            self.backend.getAddressableMemory().write(
                dst, self.backend.getAddressableMemory().read(src, count)
            )
        else:
            # In the "L0_TO_" modes, a memset is done.
            zero_val = conv_to_bytes(0, count)
            self.backend.getAddressableMemory().write(dst, zero_val)
