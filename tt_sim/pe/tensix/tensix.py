import importlib.resources as resources

import yaml

from tt_sim.memory.mem_mapable import MemMapable
from tt_sim.memory.memory import VisibleMemory
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.tensix.backend import TensixBackend
from tt_sim.pe.tensix.frontend import TensixFrontend
from tt_sim.util.bits import extract_bits, get_bits
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32


class TensixGPR(MemMapable):
    def __init__(self, tensix_cp):
        self.tensix_cp = tensix_cp

    def read(self, addr, size):
        return conv_to_bytes(0)
        # raise NotImplementedError(
        #        (
        #        f"Reading from address {hex(addr)} not yet supported by tensix "
        #        f"co-processor backend configuration"
        #
        #        )
        #    )

    def write(self, addr, value, size=None):
        return
        # raise NotImplementedError(
        #        (
        #        f"Writing to address {hex(addr)} not yet supported by tensix "
        #        f"co-processor backend configuration"
        #
        #        )
        #    )

    def getSize(self):
        return 0xFFF


class TensixBackendConfiguration(MemMapable):
    CFG_STATE_SIZE = 47
    THD_STATE_SIZE = 57

    def __init__(self, tensix_cp):
        self.tensix_cp = tensix_cp
        self.config = [[0] * TensixBackendConfiguration.CFG_STATE_SIZE * 4] * 2
        self.threadConfig = [[0] * TensixBackendConfiguration.THD_STATE_SIZE] * 3

    def read(self, addr, size):
        threadConfigStart = TensixBackendConfiguration.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = TensixBackendConfiguration.CFG_STATE_SIZE * 4
            second_idx = 1 if idx > TensixBackendConfiguration.CFG_STATE_SIZE * 4 else 0
            first_idx = int(idx - (each_config_size * second_idx))
            return conv_to_bytes(self.config[second_idx][first_idx])
        else:
            idx = idx - threadConfigStart
            second_idx = idx / TensixBackendConfiguration.THD_STATE_SIZE
            return conv_to_bytes(
                self.threadConfig[second_idx][
                    idx - ((TensixBackendConfiguration.THD_STATE_SIZE) * second_idx)
                ]
            )

    def write(self, addr, value, size=None):
        threadConfigStart = TensixBackendConfiguration.CFG_STATE_SIZE * 4 * 2
        idx = addr / 4
        if idx < threadConfigStart:
            each_config_size = TensixBackendConfiguration.CFG_STATE_SIZE * 4
            second_idx = 1 if idx > each_config_size else 0
            first_idx = int(idx - (each_config_size * second_idx))
            self.config[second_idx][first_idx] = conv_to_uint32(value)
        else:
            idx = idx - threadConfigStart
            second_idx = int(idx / TensixBackendConfiguration.THD_STATE_SIZE)
            self.threadConfig[second_idx][
                idx - ((TensixBackendConfiguration.THD_STATE_SIZE) * second_idx)
            ] = conv_to_uint32(value)

    def getSize(self):
        return 0xFFFF


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
        self.backend = TensixBackend(self.tensix_instruction_decoder)
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
