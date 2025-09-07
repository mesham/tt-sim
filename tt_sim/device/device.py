from tt_sim.memory.memory import MemorySpace


class DeviceMemory(MemorySpace):
    def __init__(self, memory_map, size, safe=True):
        super().__init__(memory_map, size, safe)


class Device:
    def __init__(self, device_memory, clocks, resets):
        self.device_memory = device_memory
        self.clocks = clocks
        self.resets = resets

    def run(self, num_iterations, clock_number=0, print_cycle=False):
        self.clocks[clock_number].run(num_iterations, print_cycle)

    def reset(self, reset_number=0):
        self.resets[reset_number].reset()
