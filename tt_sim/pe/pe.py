from abc import ABC, abstractmethod

from tt_sim.device.clock import Clockable, Resetable


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
