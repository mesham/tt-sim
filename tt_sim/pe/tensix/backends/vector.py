from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit


class VectorUnit(TensixBackendUnit):
    def __init__(self, backend):
        super().__init__(backend)

    def clock_tick(self, cycle_num):
        pass
