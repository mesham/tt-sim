from tt_sim.memory.memory import VisibleMemory
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.tensix.backend import TensixBackend
from tt_sim.pe.tensix.frontend import TensixFrontend
from tt_sim.pe.tensix.util import TensixCoprocessorDiagnostics


class TensixCoProcessor(ProcessingElement):
    """
    Entry point into the Tensix coprocessor, this is really just a wrapper for
    the frontend and backend parts
    """

    def __init__(self, diags_settings=None):
        if diags_settings is None:
            diags_settings = TensixCoprocessorDiagnostics()
        self.backend = TensixBackend(diags_settings)
        self.threads = [
            TensixFrontend(i, self.backend, diags_settings) for i in range(3)
        ]
        self.backend.setFrontendThreads(self.threads)

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

    def CoprocessorDoneCheck(self, thread_id):
        if self.threads[thread_id].hasInflightInstructions():
            return True
        return self.backend.hasInflightInstructionsFromThread(thread_id)

    def MOPExpanderDoneCheck(self, thread_id):
        return self.threads[thread_id].MOPExpanderDoneCheck()

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
