from abc import ABC, abstractmethod

from tt_sim.device.clock import Clockable, Resetable
from tt_sim.memory.memory import MemorySpace


class PEMemory(MemorySpace):
    def __init__(self, memory_map, size, safe=True):
        super().__init__(memory_map, size, safe)


class ProcessingElement(Clockable, Resetable, ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        raise NotImplementedError()

    @abstractmethod
    def getRegisterFile(self):
        raise NotImplementedError()
