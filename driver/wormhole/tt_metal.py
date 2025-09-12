import json
import os
import re
from abc import ABC


class BaseType(ABC):
    def __init__(self, start_addr, end_addr):
        self.start_addr = start_addr
        self.end_addr = end_addr

    def getBytes(self):
        return self.end_addr - self.start_addr

    @classmethod
    def process(cls, base_start_addr, entries_config, config):
        contents = {}
        current_end_addr = base_start_addr
        for key, value in entries_config.items():
            type_name = value[1]
            num_elements = value[2]
            start_addr = value[0] + base_start_addr
            if type_name in ElementType.type_widths:
                type = ElementType(type_name, start_addr)
            else:
                type = cls.process(start_addr, config[type_name], config)

                StructType(start_addr, config[type_name], config)
            if num_elements == 1:
                contents[key] = type
            else:
                contents[key] = ArrayType(type, start_addr, num_elements)

            current_end_addr += contents[key].getBytes()

        return StructType(base_start_addr, current_end_addr, contents)


class ElementType(BaseType):
    type_widths = {"uint8_t": 1, "uint16_t": 2, "uint32_t": 4}

    def __init__(self, type_name, start_addr):
        self.type_name = type_name
        self.type_width = ElementType.type_widths[type_name]
        super().__init__(start_addr, start_addr + self.type_width)

    def lookup(self, keys):
        return (self.start_addr, self.type_width)


class ArrayType(BaseType):
    def __init__(self, type, start_addr, num_entries):
        self.type = type
        self.num_entries = num_entries
        super().__init__(start_addr, start_addr + (type.getBytes() * num_entries))

    def lookup(self, keys):
        idx = keys[0]
        start_addr, type_width = self.type.lookup(keys[1:])
        return (
            start_addr + (int(self.getBytes() / self.num_entries) * idx),
            type_width,
        )


class StructType(BaseType):
    def __init__(self, start_addr, end_addr, contents):
        self.contents = contents
        super().__init__(start_addr, end_addr)

    def lookup(self, keys):
        k = keys[0]
        assert k in self.contents
        v = self.contents[k]
        return v.lookup(keys[1:])


class TT_Metal:
    class Firmware:
        def __init__(self, text_addr, text_bin, data_addr=None, data_bin=None):
            self.text_addr = text_addr
            self.text_bin = text_bin
            self.data_addr = data_addr
            self.data_bin = data_bin

        def get_text_addr(self):
            return self.text_addr

        def get_text_bin(self):
            return self.text_bin

        def get_data_addr(self):
            return self.data_addr

        def get_data_bin(self):
            return self.data_bin

    def __init__(self, config_file):
        with open(config_file) as json_file:
            self.config = json.load(json_file)
        self.config = TT_Metal.parse_json_hex(self.config)
        self.mailbox_config = StructType.process(
            self.config["l1_memory_map"]["MEM_MAILBOX_BASE"],
            self.config["mailboxes_t"],
            self.config,
        )
        self.parameter_file = None

    def load_kernel(self, parameter_file):
        with open(parameter_file) as json_file:
            self.parameters = json.load(json_file)
        self.parameters = TT_Metal.parse_json_hex(self.parameters)

    def get_mailbox_config_details(self, *paths):
        return self.mailbox_config.lookup(paths)

    def get_constant(self, constant):
        assert constant in self.config["constants"]
        return self.config["constants"][constant]

    def generate_kernel_to_device_data_transfer_details(self):
        transfer_details = self.get_transfer_kernel_binary_details()
        transfer_details += self.generate_transfer_mailbox_details("launch")
        transfer_details += self.generate_transfer_mailbox_details("go_message")
        transfer_details += self.generate_transfer_runtime_arguments_details()
        transfer_details += self.generate_transfer_cb_config_details()
        return transfer_details

    def generate_transfer_mailbox_details(self, key):
        assert self.parameters is not None
        return TT_Metal.flatten(
            TT_Metal.build_transfer_data(
                self.parameters[key], [key], self.mailbox_config
            )
        )

    def generate_transfer_runtime_arguments_details(self):
        assert self.parameters is not None
        contents = []

        if (
            "runtime_args_base_addr" in self.parameters
            and "runtime_args" in self.parameters
        ):
            rt_base_addr = self.parameters["runtime_args_base_addr"]

            for idx, rt_arg in enumerate(self.parameters["runtime_args"]):
                contents.append(((idx * 4) + rt_base_addr, 4, rt_arg))
        return contents

    def generate_transfer_cb_config_details(self):
        assert self.parameters is not None
        contents = []

        if "cb_config_base_addr" in self.parameters and "cb_config" in self.parameters:
            cb_config_base_addr = self.parameters["cb_config_base_addr"]

            for idx, cb_config in enumerate(self.parameters["cb_config"]):
                contents.append(((idx * 4) + cb_config_base_addr, 4, cb_config))
        return contents

    def get_risc_core_kernel_binary(self, filename, address_key):
        if filename in self.parameters and address_key in self.parameters:
            bin_file = self.read_binary_file(self.parameters[filename])
            return (self.parameters[address_key], len(bin_file), bin_file)
        else:
            return None

    def get_transfer_kernel_binary_details(self):
        assert self.parameters is not None
        contents = []
        brisc_details = self.get_risc_core_kernel_binary(
            "brisc_binary_file", "brisc_binary_text_addr"
        )
        if brisc_details is not None:
            contents.append(brisc_details)
        ncrisc_details = self.get_risc_core_kernel_binary(
            "ncrisc_binary_file", "ncrisc_binary_text_addr"
        )
        if ncrisc_details is not None:
            contents.append(ncrisc_details)
        trisc0_details = self.get_risc_core_kernel_binary(
            "trisc0_binary_file", "trisc0_binary_text_addr"
        )
        if trisc0_details is not None:
            contents.append(trisc0_details)
        trisc1_details = self.get_risc_core_kernel_binary(
            "trisc1_binary_file", "trisc1_binary_text_addr"
        )
        if trisc1_details is not None:
            contents.append(trisc1_details)
        trisc2_details = self.get_risc_core_kernel_binary(
            "trisc2_binary_file", "trisc2_binary_text_addr"
        )
        if trisc2_details is not None:
            contents.append(trisc2_details)
        return contents

    @classmethod
    def flatten(cls, nested_list):
        flat_list = []

        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(cls.flatten(item))
            else:
                flat_list.append(item)

        return flat_list

    @classmethod
    def build_transfer_data(cls, data, paths, mailbox_config):
        if isinstance(data, dict):
            contents = []
            for k, v in data.items():
                contents.append(cls.build_transfer_data(v, paths + [k], mailbox_config))
        elif isinstance(data, list):
            contents = []
            for idx, v in enumerate(data):
                contents.append(
                    cls.build_transfer_data(v, paths + [idx], mailbox_config)
                )
        else:
            start_addr, type_width = mailbox_config.lookup(paths)
            contents = [(start_addr, type_width, data)]
        return contents

    @classmethod
    def parse_json_hex(cls, json):
        """
        JSON doesn't permit hex as a literal, hence encode as a string and now
        convert to integer representation
        """
        hex_pattern = re.compile(r"^0x[0-9a-fA-F]+$")

        if isinstance(json, list):
            return [cls.parse_json_hex(item) for item in json]
        elif isinstance(json, dict):
            return {k: cls.parse_json_hex(v) for k, v in json.items()}
        elif isinstance(json, str) and hex_pattern.match(json):
            try:
                return int(json, 16)
            except ValueError:
                # If conversion fails, return original string
                return json
        else:
            return json

    def get_config_value(self, *keys):
        current_dict = self.config

        for key in keys[:-1]:
            current_dict = current_dict[key]
        return current_dict[keys[-1]]

    def read_binary_file(self, filename):
        with open(filename, "rb") as file:
            return file.read()

    def read_firmware(self, firmware_directory):
        # Read in firmware binaries
        with open(
            os.path.join(firmware_directory, "brisc_firmware_text.bin"), "rb"
        ) as file:
            brisc_text = file.read()

        with open(
            os.path.join(firmware_directory, "ncrisc_firmware_text.bin"), "rb"
        ) as file:
            ncrisc_text = file.read()

        with open(
            os.path.join(firmware_directory, "ncrisc_firmware_data.bin"), "rb"
        ) as file:
            ncrisc_data = file.read()

        with open(
            os.path.join(firmware_directory, "trisc0_firmware_text.bin"), "rb"
        ) as file:
            trisc0_text = file.read()

        with open(
            os.path.join(firmware_directory, "trisc1_firmware_text.bin"), "rb"
        ) as file:
            trisc1_text = file.read()

        with open(
            os.path.join(firmware_directory, "trisc2_firmware_text.bin"), "rb"
        ) as file:
            trisc2_text = file.read()

        memory_map_config = self.config["l1_memory_map"]

        return (
            TT_Metal.Firmware(
                memory_map_config["MEM_BRISC_FIRMWARE_BASE"],
                brisc_text,
            ),
            TT_Metal.Firmware(
                memory_map_config["MEM_NCRISC_FIRMWARE_BASE"],
                ncrisc_text,
                memory_map_config["MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH"],
                ncrisc_data,
            ),
            TT_Metal.Firmware(
                memory_map_config["MEM_TRISC0_FIRMWARE_BASE"],
                trisc0_text,
            ),
            TT_Metal.Firmware(
                memory_map_config["MEM_TRISC1_FIRMWARE_BASE"],
                trisc1_text,
            ),
            TT_Metal.Firmware(
                memory_map_config["MEM_TRISC2_FIRMWARE_BASE"],
                trisc2_text,
            ),
        )
