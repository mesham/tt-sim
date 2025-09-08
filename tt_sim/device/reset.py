from abc import ABC, abstractmethod


class Resetable(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class Reset:
    def __init__(self, resetables):
        self.reset_items = resetables

    def reset(self):
        for item in self.reset_items:
            item.reset()
