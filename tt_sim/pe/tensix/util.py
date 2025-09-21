import importlib.resources as resources

import yaml

from tt_sim.util.bits import extract_bits, get_bits


class DiagnosticsSettings:
    def __init__(
        self,
        issued_instructions=False,
        configurations_set=False,
        unpacking=False,
        packing=False,
        fpu_calculations=False,
        sfpu_calculations=False,
    ):
        self.issued_instructions = issued_instructions
        self.configurations_set = configurations_set
        self.unpacking = unpacking
        self.packing = packing
        self.fpu_calculations = fpu_calculations
        self.sfpu_calculations = sfpu_calculations

    def reportFPUCalculations(self):
        return self.fpu_calculations

    def reportSFPUCalculations(self):
        return self.sfpu_calculations

    def reportUnpacking(self):
        return self.unpacking

    def reportPacking(self):
        return self.packing

    def reportIssuedInstructions(self):
        return self.issued_instructions

    def reportConfigurationSet(self):
        return self.configurations_set


class TensixConfigurationConstants:
    @classmethod
    def init(cls):
        if not hasattr(cls, "config_constants"):
            with (
                resources.files("tt_sim.pe.tensix")
                .joinpath("tensix_backend_cfg.yaml")
                .open("r") as f
            ):
                cls.config_constants = yaml.safe_load(f)
                cls.ids = {}
                for k in cls.config_constants.keys():
                    addr32 = cls.get_addr32(k)
                    cls.ids[addr32] = k

    @classmethod
    def get_name(cls, id):
        cls.init()
        if id in cls.ids:
            return cls.ids[id]
        else:
            return "NONE"

    @classmethod
    def get_addr32(cls, key):
        cls.init()
        if key not in cls.config_constants:
            raise IndexError(f"'{key}' not in constants")
        return cls.config_constants[key]["ADDR32"]

    @classmethod
    def get_shamt(cls, key):
        cls.init()
        if key not in cls.config_constants:
            raise IndexError(f"'{key}' not in constants")
        return cls.config_constants[key]["SHAMT"]

    @classmethod
    def get_mask(cls, key):
        cls.init()
        if key not in cls.config_constants:
            raise IndexError(f"'{key}' not in constants")
        return cls.config_constants[key]["MASK"]

    @classmethod
    def parse_raw_config_value(cls, value, key):
        cls.init()
        mask = cls.get_mask(key)
        shamt = cls.get_shamt(key)
        return cls.tensix_be_config_parse_value(value, shamt, mask)

    @classmethod
    def tensix_be_config_parse_value(cls, value, shamt, mask):
        return (value & mask) >> shamt


class TensixInstructionDecoder:
    @classmethod
    def init(cls):
        if not hasattr(cls, "tensix_instructions") or not hasattr(cls, "opcodes"):
            with (
                resources.files("tt_sim.pe.tensix")
                .joinpath("tensix_instructions.yaml")
                .open("r") as f
            ):
                cls.tensix_instructions = yaml.safe_load(f)

            cls.opcodes = cls._generate_tensix_instructions_by_opcode()

    @classmethod
    def _generate_tensix_instructions_by_opcode(cls):
        by_opcode = {}
        for k, instruction in cls.tensix_instructions.items():
            by_opcode[instruction["op_binary"]] = instruction
            by_opcode[instruction["op_binary"]]["name"] = k
        return by_opcode

    @classmethod
    def isInstructionRecognised(cls, instruction):
        cls.init()
        opcode = extract_bits(instruction, 8, 24)
        return opcode in cls.opcodes

    @classmethod
    def getInstructionInfo(cls, instruction):
        cls.init()
        opcode = extract_bits(instruction, 8, 24)
        assert opcode in cls.opcodes
        instruction_info = cls.opcodes[opcode]
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
