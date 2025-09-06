from abc import ABC, abstractmethod


class RV_ISA(ABC):
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
    def bits_to_int(cls, bits):
        binary_str = "".join(str(bit) for bit in bits)
        return int(binary_str, 2)

    @classmethod
    @abstractmethod
    def run(cls, register_file, device_memory):
        raise NotImplementedError()
