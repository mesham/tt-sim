import numpy as np

from tt_sim.memory.mem_mapable import MemMapable


class AddressableMemory(MemMapable):
    def __init__(self, size, alignment=None):
        self.memory = np.empty(size, dtype=np.uint8)
        self.size = size
        self.alignment = alignment

    def read(self, addr, size):
        if addr > self.size:
            raise IndexError(
                f"Start address '{addr}' overflows memory size '{self.size}'"
            )
        if addr + size > self.size:
            raise IndexError(
                f"End address '{addr + size}' overflows memory size '{self.size}'"
            )
        return self.memory[addr : addr + size].tobytes()

    def write(self, addr, value, size=None):
        assert isinstance(value, bytes)

        if size is None:
            size = len(value)

        if addr > self.size:
            raise IndexError(
                f"Start address '{addr}' overflows memory size '{self.size}'"
            )
        if addr + size > self.size:
            raise IndexError(
                f"End address '{addr + size}' overflows memory size '{self.size}'"
            )

        if self.alignment is not None:
            if addr % self.alignment != 0:
                raise IndexError(
                    f"Start address must be aligned to '{self.alignment}' whereas '{addr}' is not"
                )

        byte_buffer = np.frombuffer(value, dtype=np.uint8)
        self.memory[addr : addr + size] = byte_buffer[:size]

    def getSize(self):
        return self.size


class DRAM(AddressableMemory):
    def __init__(self, size):
        super().__init__(size, None)


class L1(AddressableMemory):
    def __init__(self, size):
        super().__init__(size, None)
