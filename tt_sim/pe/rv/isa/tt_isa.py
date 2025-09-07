from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_int32, conv_to_bytes


class RV_TT_ISA(RV_ISA):
    @classmethod
    def run(cls, register_file, device_memory):
        pc = register_file["pc"]
        addr = conv_to_int32(pc.read())
        instr = device_memory.read(addr, 4)

        opcode_bin = RV_ISA.get_bits(instr, 0, 6)
        opcode_bin.reverse()

        if opcode_bin[5] != 1 or opcode_bin[6] != 1:
            # ttinsn
            """
            This is an encoding of the .ttinst which copies a constant into INSTRN_BUF_BASE (0xFFE40000) to send to the
            Tensix unit. As the constant is rotated left by two bits and is a maximum value of 0xC0000000u, it will
            always end up as not being 0b11 in the two LSB, which is unique when not including the C extension
            (which TT does not). Therefore it simply needs to be rotated right by two bits and then copied to the address.

            https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/PushTensixInstruction.md#ttinsn-instruction-set-extension
            """

            constant = RV_TT_ISA.rotate_right(RV_ISA.get_int(instr, 0, 31), 2)
            device_memory.write(0xFFE40000, conv_to_bytes(constant))
            return True
        else:
            return False

    @classmethod
    def rotate_right(cls, val, n, bit_width=32):
        return ((val >> n) | (val << (bit_width - n))) & ((1 << bit_width) - 1)
