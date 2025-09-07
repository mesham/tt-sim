from tt_sim.memory.mem_mapable import MemMapable


class DeviceMemory(MemMapable):
    def __init__(self, memory_map, size, safe=True):
        self.memory_map = memory_map
        if isinstance(size, str):
            if "K" in size:
                sval = int(size.split("M")[0])
                size = sval * 1024
            elif "M" in size:
                sval = int(size.split("M")[0])
                size = sval * 1024 * 1024
            elif "G" in size:
                sval = int(size.split("G")[0])
                size = sval * 1024 * 1024 * 1024
            else:
                raise Exception(f"Unable to parse memory size string '{size}'")

        assert isinstance(size, int)
        self.size = size
        self.safe = safe

    def _locate_memory_space(self, addr):
        for addr_range, memory_space in self.memory_map.items():
            if addr_range.check_match(addr):
                return addr_range, memory_space

        if self.safe:
            raise IndexError(
                f"Provided address '{addr}' does not match any registered memory spaces"
            )
        else:
            return None, None

    def convert_addr_to_target_range(self, addr_range, addr):
        return addr - addr_range.low

    def read(self, addr, size):
        addr_range, memory_space = self._locate_memory_space(addr)
        if addr_range is not None and memory_space is not None:
            target_addr = self.convert_addr_to_target_range(addr_range, addr)
            return memory_space.read(target_addr, size)
        else:
            return 0

    def write(self, addr, value, size=None):
        addr_range, memory_space = self._locate_memory_space(addr)
        if addr_range is not None and memory_space is not None:
            target_addr = self.convert_addr_to_target_range(addr_range, addr)
            memory_space.write(target_addr, value, size)

    def getSize(self):
        return self.size


class Device:
    def __init__(self, device_memory, clocks, resets):
        self.device_memory = device_memory
        self.clocks = clocks
        self.resets = resets

    def run(self, num_iterations, clock_number=0):
        self.clocks[clock_number].run(num_iterations)

    def reset(self, reset_number=0):
        self.resets[reset_number].reset()
