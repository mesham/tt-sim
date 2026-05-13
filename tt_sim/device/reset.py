from abc import ABC, abstractmethod


class Resetable(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class Reset:
    def __init__(self, resetables):
        self.reset_items = list(resetables)

    def add_resetable(self, resetable):
        self.reset_items.append(resetable)

    def add_resetables(self, resetables):
        self.reset_items.extend(resetables)

    def reset(self):
        for item in self.reset_items:
            item.reset()
