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
        all_bits = []
        for b in bytes:
            for i in range(8):
                all_bits.append((b >> i) & 1)

        # End is inclusive, so if start=0 and end=6, get 7 values back
        return all_bits[start : end + 1]

    @classmethod
    def get_int(cls, bytes, start, end):
        val_bin = RV_ISA.get_bits(bytes, start, end)
        val_bin.reverse()
        return RV_ISA.bits_to_int(val_bin)

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
