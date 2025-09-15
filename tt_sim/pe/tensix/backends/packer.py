from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit


class PackerUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {"PACR": "handle_pacr"}

    def __init__(self, backend):
        super().__init__(backend, PackerUnit.OPCODE_TO_HANDLER, "Packer")

    def handle_pacr(self, instruction_info, issue_thread, instr_args):
        pass
