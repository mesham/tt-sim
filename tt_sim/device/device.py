from abc import ABC, abstractmethod

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import MemorySpace


class DeviceMemory(MemorySpace):
    def __init__(self, memory_map, size, safe=True, snoop_addresses=None):
        super().__init__(memory_map, size, safe, snoop_addresses)


class Device:
    def __init__(self, device_memory, clocks, resets):
        self.device_memory = device_memory
        self.clocks = clocks
        self.resets = resets

    def run(self, num_iterations, clock_number=0):
        self.clocks[clock_number].run(num_iterations)

    def reset(self, reset_number=0):
        self.resets[reset_number].reset()


class DeviceTile(MemMapable, ABC):
    def __init__(self, coord_x, coord_y, noc0_router, noc1_router):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.noc0_router = noc0_router
        self.noc1_router = noc1_router

    def get_coord_x(self):
        return self.coord_x

    def get_coord_y(self):
        return self.coord_y

    def get_coord_pair(self):
        return (self.coord_x, self.coord_y)

    def get_noc_nui(self, idx):
        assert idx < 2
        if idx == 0:
            return self.noc0_router
        elif idx == 1:
            return self.noc1_router

    @abstractmethod
    def get_clocks(self):
        raise NotImplementedError()

    @abstractmethod
    def get_resets(self):
        raise NotImplementedError()
