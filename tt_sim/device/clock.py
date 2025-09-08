from abc import ABC, abstractmethod


class Clockable(ABC):
    @abstractmethod
    def clock_tick(self, cycle_num):
        raise NotImplementedError()


class Clock:
    def __init__(self, clockables):
        self.clock_items = clockables

    def clock_tick(self, cycle):
        for item in self.clock_items:
            item.clock_tick(cycle)

    def run(self, num_iterations):
        for i in range(num_iterations):
            self.clock_tick(i)
