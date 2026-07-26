import builtins
from abc import ABC, abstractmethod


class RV_ISA(ABC):
    REGISTER_ID_TO_NAME = [
        "zero",
        "ra",
        "sp",
        "gp",
        "tp",
        "t0",
        "t1",
        "t2",
        "s0",
        "s1",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "t3",
        "t4",
        "t5",
        "t6",
    ]

    @classmethod
    def get_bits(cls, bytes, start, end):
        # Little-endian bit order: bit i is bit i of the integer formed from
        # the byte sequence read low-address-first. End is inclusive, so
        # start=0, end=6 returns 7 values (LSB-first, matching indices).
        word = int.from_bytes(builtins.bytes(bytes), "little")
        return [(word >> i) & 1 for i in range(start, end + 1)]

    @classmethod
    def get_int(cls, bytes, start, end):
        # Value of the inclusive bit field [start, end] where start is the
        # LSB position — equivalent to get_bits + reverse + bits_to_int.
        word = int.from_bytes(builtins.bytes(bytes), "little")
        return (word >> start) & ((1 << (end - start + 1)) - 1)

    @classmethod
    def get_reg_name(cls, reg_idx):
        assert reg_idx < len(RV_ISA.REGISTER_ID_TO_NAME)
        return RV_ISA.REGISTER_ID_TO_NAME[reg_idx]

    @classmethod
    def bits_to_int(cls, bits):
        binary_str = "".join(str(bit) for bit in bits)
        return int(binary_str, 2)

    @classmethod
    def print_snoop(cls, snoop, message, info_msg=None):
        if snoop:
            print(message, end="")
            if info_msg is not None:
                assert isinstance(info_msg, str)
                print("    # " + info_msg, end="")

    @classmethod
    @abstractmethod
    def run(cls, register_file, device_memory):
        raise NotImplementedError()
