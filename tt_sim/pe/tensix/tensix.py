import importlib.resources as resources

import yaml

from tt_sim.memory.memory import VisibleMemory
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.tensix.backend import TensixBackend
from tt_sim.pe.tensix.frontend import TensixFrontend
from tt_sim.util.bits import extract_bits, get_bits


class TensixConfigurationConstants:
    def __init__(self):
        with (
            resources.files("tt_sim.pe.tensix")
            .joinpath("tensix_backend_cfg.yaml")
            .open("r") as f
        ):
            self.config_constants = yaml.safe_load(f)

    def get_addr32(self, value):
        assert value in self.config_constants
        return self.config_constants[value]["ADDR32"]

    def get_shamt(self, value):
        assert value in self.config_constants
        return self.config_constants[value]["SHAMT"]

    def get_mask(self, value):
        assert value in self.config_constants
        return self.config_constants[value]["MASK"]

    def parse_raw_config_value(self, value, key):
        mask = self.get_mask(value)
        shamt = self.get_shamt(value)
        return self.tensix_be_config_parse_value(value, shamt, mask)

    def tensix_be_config_parse_value(self, value, shamt, mask):
        return (value & mask) >> shamt


class TensixInstructionDecoder:
    def __init__(self):
        with (
            resources.files("tt_sim.pe.tensix")
            .joinpath("tensix_instructions.yaml")
            .open("r") as f
        ):
            self.tensix_instructions = yaml.safe_load(f)

        self.opcodes = self.generate_tensix_instructions_by_opcode()

    def generate_tensix_instructions_by_opcode(self):
        by_opcode = {}
        for k, instruction in self.tensix_instructions.items():
            by_opcode[instruction["op_binary"]] = instruction
            by_opcode[instruction["op_binary"]]["name"] = k
        return by_opcode

    def isInstructionRecognised(self, instruction):
        opcode = extract_bits(instruction, 8, 24)
        return opcode in self.opcodes

    def getInstructionInfo(self, instruction):
        opcode = extract_bits(instruction, 8, 24)
        assert opcode in self.opcodes
        instruction_info = self.opcodes[opcode]
        instr_args = {}
        if "arguments" in instruction_info and isinstance(
            instruction_info["arguments"], list
        ):
            arg_ends = []  # end of each argument (inclusive)
            for arg in instruction_info["arguments"][1:]:
                arg_ends.append(arg["start_bit"] - 1)
            arg_ends.append(23)  # opcode is from 24 onwards

            for idx, arg in enumerate(instruction_info["arguments"]):
                instr_args[arg["name"]] = get_bits(
                    instruction, arg["start_bit"], arg_ends[idx]
                )

        instruction_info["instr_args"] = instr_args

        return instruction_info


class TensixCoProcessor(ProcessingElement):
    def __init__(self):
        self.tensix_instruction_decoder = TensixInstructionDecoder()
        self.configuration_constants = TensixConfigurationConstants()
        self.backend = TensixBackend(
            self.tensix_instruction_decoder, self.configuration_constants
        )
        self.threads = [
            TensixFrontend(i, self.tensix_instruction_decoder, self.backend)
            for i in range(3)
        ]

    def getThread(self, idx):
        assert idx < 3
        return self.threads[idx]

    def getClocks(self):
        clocks = self.backend.getClocks()
        for thread in self.threads:
            clocks += thread.getClocks()

        return clocks

    def setAddressableMemory(self, addressable_memory):
        if len(addressable_memory) == 1:
            self.backend.setAddressableMemory(addressable_memory[0])
        else:
            self.backend.setAddressableMemory(VisibleMemory.merge(*addressable_memory))

    def getBackend(self):
        return self.backend

    def getRegisterFile(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def clock_tick(self):
        pass

    def reset(self):
        pass
