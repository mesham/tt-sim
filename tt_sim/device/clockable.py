from abc import ABC, abstractmethod


class Clockable(ABC):
    @abstractmethod
    def clock_tick(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
