import json
import os
import re


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
                memory_map_config["MEM_NCRISC_FIRMWARE_BASE"],
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
