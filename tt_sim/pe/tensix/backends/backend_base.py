from abc import ABC

from tt_sim.device.clock import Clockable


class TensixBackendUnit(Clockable, ABC):
    def __init__(self, backend):
        self.backend = backend
        self.instruction_buffer = []

    def issueInstruction(self, instruction, from_thread):
        self.instruction_buffer.append(
            (
                instruction,
                from_thread,
            )
        )

    def getThreadConfigValue(self, issue_thread, key):
        return self.backend.getThreadConfigValue(issue_thread, key)

    def getConfigValue(self, state_id, key):
        return self.backend.getConfigValue(state_id, key)

    def getRCW(self, thread_id):
        return self.backend.getRCW(thread_id)

    def getDst(self):
        return self.backend.getDst()
