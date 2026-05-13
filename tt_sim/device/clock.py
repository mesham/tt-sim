from abc import ABC, abstractmethod

from tt_sim.device.reset import Resetable


class Clockable(ABC):
    @abstractmethod
    def clock_tick(self, cycle_num):
        raise NotImplementedError()


class Clock(Resetable):
    def __init__(self, clockables, on_tick=None):
        self.clock_items = list(clockables)
        self.clock_tick_num = 0
        self.on_tick = on_tick

    def add_clockable(self, clockable):
        self.clock_items.append(clockable)

    def add_clockables(self, clockables):
        self.clock_items.extend(clockables)

    def clock_tick(self, cycle):
        for item in self.clock_items:
            item.clock_tick(cycle)
        if self.on_tick is not None:
            self.on_tick(cycle)

    def reset(self):
        self.clock_tick_num = 0

    def run(self, num_iterations):
        for i in range(num_iterations):
            self.clock_tick(i + self.clock_tick_num)
        self.clock_tick_num += num_iterations
