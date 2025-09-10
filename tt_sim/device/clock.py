from abc import ABC, abstractmethod

from tt_sim.device.reset import Resetable


class Clockable(ABC):
    @abstractmethod
    def clock_tick(self, cycle_num):
        raise NotImplementedError()


class Clock(Resetable):
    def __init__(self, clockables):
        self.clock_items = clockables
        self.clock_tick_num = 0

    def clock_tick(self, cycle):
        for item in self.clock_items:
            item.clock_tick(cycle)

    def reset(self):
        self.clock_tick_num = 0

    def run(self, num_iterations):
        for i in range(num_iterations):
            self.clock_tick(i + self.clock_tick_num)
        self.clock_tick_num += num_iterations
