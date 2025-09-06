from abc import ABC, abstractmethod


class Resetable(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError()
