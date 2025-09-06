from abc import ABC, abstractmethod


class MemMapable(ABC):
    @abstractmethod
    def getSize(self):
        raise NotImplementedError()

    @abstractmethod
    def read(self, addr, size):
        raise NotImplementedError()

    @abstractmethod
    def write(self, addr, value, size):
        raise NotImplementedError()
