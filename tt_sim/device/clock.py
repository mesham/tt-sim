from abc import ABC, abstractmethod


class Resetable(ABC):
    @abstractmethod
    def reset(self):
        raise NotImplementedError()


class Clockable(ABC):
    @abstractmethod
    def clock_tick(self):
        raise NotImplementedError()


class Clock:
    def __init__(self, clockables):
        self.clock_items = clockables

    def clock_tick(self):
        for item in self.clock_items:
            item.clock_tick()

    def run(self, num_iterations):
        for i in range(num_iterations):
            self.clock_tick()
