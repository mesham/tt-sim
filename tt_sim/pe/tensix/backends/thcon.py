from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit


class ScalarUnit(TensixBackendUnit):
    OPCODE_TO_HANDLER = {
        "SETDMAREG": "handle_setdmareg",
        "REG2FLOP": "handle_reg2flop",
        "STOREREG": "handle_storereg",
    }

    def __init__(self, backend, gprs):
        self.gprs = gprs
        super().__init__(backend, ScalarUnit.OPCODE_TO_HANDLER, "Scalar")

    def handle_reg2flop(self, instruction_info, issue_thread, instr_args):
        # TODO
        pass

    def handle_storereg(self, instruction_info, issue_thread, instr_args):
        # TODO
        pass

    def handle_setdmareg(self, instruction_info, issue_thread, instr_args):
        setSignalsMode = instr_args["SetSignalsMode"]
        if setSignalsMode == 0:
            # Set 16 bits of one GPR
            resultHalfReg = instr_args["RegIndex16b"]
            newValue1 = instr_args["Payload_SigSel"]
            newValue2 = instr_args["Payload_SigSelSize"]

            newValue = newValue1 | (newValue2 << 14)

            existing_val = self.gprs.getRegisters(issue_thread)[int(resultHalfReg / 2)]
            self.gprs.getRegisters(issue_thread)[int(resultHalfReg / 2)] = (
                existing_val & 0xFF00
            ) | newValue
        else:
            pass
