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

    def isBFPFormat(self):
        return self.value == 2 or self.value == 3 or self.value == 11


DATA_FORMAT_TO_BITS = {
    DataFormat.FP32: 32,
    DataFormat.FP16: 16,
    DataFormat.BFP8: 8,
    DataFormat.BFP4: 4,
    DataFormat.BFP2: 2,
    DataFormat.FP16_b: 16,
    DataFormat.BFP8_b: 8,
    DataFormat.BFP4_b: 4,
    DataFormat.BFP2_b: 2,
    DataFormat.INT8: 8,
    DataFormat.UINT8: 8,
    DataFormat.UINT16: 16,
    DataFormat.UINT32: 32,
    DataFormat.INT32: 32,
    DataFormat.TF32: 32,
    DataFormat.BF16: 16,
}

DATA_FORMAT_TO_NAME = {
    DataFormat.FP32: "FP32",
    DataFormat.FP16: "FP16",
    DataFormat.BFP8: "BFP8",
    DataFormat.BFP4: "BFP4",
    DataFormat.BFP2: "BFP2",
    DataFormat.FP16_b: "FP16_b",
    DataFormat.BFP8_b: "BFP8_b",
    DataFormat.BFP4_b: "BFP4_b",
    DataFormat.BFP2_b: "BFP2_b",
    DataFormat.INT8: "INT8",
    DataFormat.UINT8: "UINT8",
    DataFormat.UINT16: "UINT16",
    DataFormat.UINT32: "UINT32",
    DataFormat.INT32: "INT32",
    DataFormat.TF32: "TF32",
    DataFormat.BF16: "BF16",
}


class TensixBackendUnit(Clockable, ABC):
    def __init__(self, backend, opcode_to_method_map, unit_name):
        self.backend = backend
        self.next_instruction = []
        self.opcode_to_method_map = opcode_to_method_map
        self.unit_name = unit_name

    def issueInstruction(self, instruction, from_thread):
        # The default issuing of instructions here, which applies to most
        # units, is one instruction per cycle. Can override for specific
        # units with more complex behaviour
        if len(self.next_instruction) == 0:
            self.next_instruction.append(
                (
                    instruction,
                    from_thread,
                )
            )
            return True
        else:
            return False

    def getDiagnosticSettings(self):
        return self.backend.getDiagnosticSettings()

    def hasInflightInstructionsFromThread(self, from_thread):
        if len(self.next_instruction) > 0:
            for _, thread_id in self.next_instruction:
                if thread_id == from_thread:
                    return True
        return False

    def clock_tick(self, cycle_num):
        # next_instruction is all instructions to process in this cycle,
        # is often one but for some units might be more
        while len(self.next_instruction) > 0:
            instruction, issue_thread = self.next_instruction.pop(0)
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

    def getConfigValue(self, state_id, key, words=1):
        return self.backend.getConfigValue(state_id, key, words)

    def getRWC(self, thread_id):
        return self.backend.getRWC(thread_id)

    def getDst(self):
        return self.backend.getDst()

    def checkIfNextInstructionsContainOpcodes(self, *instr_op):
        for instruction, _ in self.next_instruction:
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            instruction_name = instruction_info["name"]
            if instruction_name in list(instr_op):
                return True
        return False

    def checkIfNextInstructionsContainAnyOtherOpcodes(self, *allowed_instr_op):
        for instruction, _ in self.next_instruction:
            instruction_info = TensixInstructionDecoder.getInstructionInfo(instruction)
            instruction_name = instruction_info["name"]
            if instruction_name not in list(allowed_instr_op):
                return True
        return False
