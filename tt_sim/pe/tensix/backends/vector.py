from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit


class VectorUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {
        "SFPENCC": "handle_sfpencc",
        "SFPLOADI": "handle_sfploadi",
        "SFPCONFIG": "handle_sfpconfig",
    }

    def __init__(self, backend):
        super().__init__(backend, VectorUnit.OPCODE_TO_HANDLER, "Vector")

    def handle_sfpencc(self, instruction_info, issue_thread, instr_args):
        pass

    def handle_sfploadi(self, instruction_info, issue_thread, instr_args):
        pass

    def handle_sfpconfig(self, instruction_info, issue_thread, instr_args):
        pass
