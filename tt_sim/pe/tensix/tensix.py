from tt_sim.memory.memory import VisibleMemory
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.tensix.backend import TensixBackend
from tt_sim.pe.tensix.frontend import TensixFrontend


class TensixCoProcessor(ProcessingElement):
    def __init__(self):
        self.backend = TensixBackend()
        self.threads = [TensixFrontend(i, self.backend) for i in range(3)]

    def getThread(self, idx):
        assert idx < 3
        return self.threads[idx]

    def getClocks(self):
        clocks = self.backend.getClocks()
        for thread in self.threads:
            clocks += thread.getClocks()

        return clocks

    def setAddressableMemory(self, addressable_memory):
        if len(addressable_memory) == 1:
            self.backend.setAddressableMemory(addressable_memory[0])
        else:
            self.backend.setAddressableMemory(VisibleMemory.merge(*addressable_memory))

    def getBackend(self):
        return self.backend

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
