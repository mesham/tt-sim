from abc import ABC
from enum import IntEnum

from tt_sim.device.clock import Clockable
from tt_sim.pe.tensix.util import TensixInstructionDecoder


class DataFormat(IntEnum):
    FP32 = 0
    FP16 = 1
    BFP8 = 2
    BFP4 = 3
    BFP2 = 11
    FP16_b = 5
    BFP8_b = 6
    BFP4_b = 7
    BFP2_b = 15
    INT8 = 14
    UINT8 = 30
    UINT16 = 9
    INT32 = 8
    UINT32 = 24
    TF32 = 4
    BF16 = 10


class TensixBackendUnit(Clockable, ABC):
    def __init__(self, backend, opcode_to_method_map, unit_name):
        self.backend = backend
        self.instruction_buffer = []
        self.opcode_to_method_map = opcode_to_method_map
        self.unit_name = unit_name

    def issueInstruction(self, instruction, from_thread):
        self.instruction_buffer.append(
            (
                instruction,
                from_thread,
            )
        )

    def clock_tick(self, cycle_num):
        if len(self.instruction_buffer) > 0:
            instruction, issue_thread = self.instruction_buffer.pop(0)
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            instruction_name = instruction_info["name"]
            if instruction_name in self.opcode_to_method_map:
                if "instr_args" in instruction_info:
                    instr_args = instruction_info["instr_args"]
                else:
                    instr_args = None
                getattr(self, self.opcode_to_method_map[instruction_name])(
                    instruction_info, issue_thread, instr_args
                )
            else:
                raise NotImplementedError(
                    f"{self.unit_name} unit can not handle instruction '{instruction_info['name']}'"
                )

    def getThreadConfigValue(self, issue_thread, key):
        return self.backend.getThreadConfigValue(issue_thread, key)

    def getConfigValue(self, state_id, key):
        return self.backend.getConfigValue(state_id, key)

    def getRCW(self, thread_id):
        return self.backend.getRCW(thread_id)

    def getDst(self):
        return self.backend.getDst()
