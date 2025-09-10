from abc import ABC

import numpy as np

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory_map import MemoryMap
from tt_sim.util.conversion import conv_to_uint32


class MemorySpace(MemMapable, ABC):
    def __init__(self, memory_map, size, safe=True, snoop_addresses=None):
        self.memory_map = memory_map
        if isinstance(size, str):
            size = MemorySpace.parse_size_str(size)

        assert isinstance(size, int)
        self.size = size
        self.safe = safe
        self.snoop_addresses = [] if snoop_addresses is None else snoop_addresses

    def add_snoop(self, snoop_addr_low, snoop_addr_high):
        self.snoop_addresses.append((snoop_addr_low, snoop_addr_high))
        return len(self.snoop_addresses)

    def _locate_memory_space(self, addr):
        for addr_range, memory_space in self.memory_map.items():
            if addr_range.check_match(addr):
                return addr_range, memory_space

        if self.safe:
            raise IndexError(
                f"Provided address '{hex(addr)}' does not match any registered memory spaces"
            )
        else:
            return None, None

    def convert_addr_to_target_range(self, addr_range, addr):
        return addr - addr_range.low

    def check_is_snoop(self, addr):
        for snoop in self.snoop_addresses:
            if addr >= snoop[0] and addr <= snoop[1]:
                return True
        return False

    def read(self, addr, size):
        addr_range, memory_space = self._locate_memory_space(addr)
        if addr_range is not None and memory_space is not None:
            target_addr = self.convert_addr_to_target_range(addr_range, addr)
            if len(self.snoop_addresses) > 0 and self.check_is_snoop(addr):
                print(
                    f"->>>>>> Read value {hex(conv_to_uint32(memory_space.read(target_addr, size)))} at address {hex(addr)}"
                )
            return memory_space.read(target_addr, size)
        else:
            if len(self.snoop_addresses) > 0 and self.check_is_snoop(addr):
                print(
                    f"->>>>>> Attempt to read at address {hex(addr)} but no memory registered"
                )
            return 0

    def write(self, addr, value, size=None):
        if len(self.snoop_addresses) > 0 and self.check_is_snoop(addr):
            print(
                f"->>>>>> Write value {hex(conv_to_uint32(value))} at address {hex(addr)}"
            )
        addr_range, memory_space = self._locate_memory_space(addr)
        if addr_range is not None and memory_space is not None:
            target_addr = self.convert_addr_to_target_range(addr_range, addr)
            memory_space.write(target_addr, value, size)

    def getSize(self):
        return self.size

    @classmethod
    def parse_size_str(cls, size_str):
        if "K" in size_str:
            sval = int(size_str.split("M")[0])
            return sval * 1024
        elif "M" in size_str:
            sval = int(size_str.split("M")[0])
            return sval * 1024 * 1024
        elif "G" in size_str:
            sval = int(size_str.split("G")[0])
            return sval * 1024 * 1024 * 1024
        else:
            raise Exception(f"Unable to parse memory size string '{size_str}'")

    @classmethod
    def merge(cls, *memory_spaces):
        max_size = 0
        memory_maps = []
        safe = False
        snoops = []
        for memory_space in memory_spaces:
            mem_size = memory_space.size
            if isinstance(mem_size, str):
                mem_size = MemorySpace.parse_size_str(mem_size)
            if mem_size > max_size:
                max_size = mem_size
            memory_maps.append(memory_space.memory_map)
            snoops += memory_space.snoop_addresses
            # Any memory space with safety turned on turns it on for all
            if memory_space.safe:
                safe = True

        new_mm = MemoryMap.merge(*memory_maps)
        return cls(new_mm, max_size, safe, snoops)


class VisibleMemory(MemorySpace):
    def __init__(self, memory_map, size, safe=True, snoop_addresses=None):
        super().__init__(memory_map, size, safe, snoop_addresses)


class TensixMemory(MemorySpace):
    def __init__(self, memory_map, size, safe=True, snoop_addresses=None):
        super().__init__(memory_map, size, safe, snoop_addresses)


class TileMemory(MemorySpace):
    def __init__(self, memory_map, size, safe=True, snoop_addresses=None):
        super().__init__(memory_map, size, safe, snoop_addresses)


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
