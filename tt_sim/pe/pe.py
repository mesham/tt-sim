from abc import ABC, abstractmethod

from tt_sim.device.clockable import Clockable


class ProcessingElement(Clockable, ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def stop(self):
        raise NotImplementedError()
