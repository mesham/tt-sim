from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit


class UnPackerUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {}

    def __init__(self, backend):
        super().__init__(backend, UnPackerUnit.OPCODE_TO_HANDLER, "Unpacker")
