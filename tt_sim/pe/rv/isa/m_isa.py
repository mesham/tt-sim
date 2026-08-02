from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_bytes


class RV_M_ISA(RV_ISA):
    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)

        if instr & 0x7F != 0x33:
            return False

        # The m variant of r has a one at location 25
        m_variant = RV_ISA.get_int(instr, 25, 25) == 1
        if not m_variant:
            return False

        type_val = RV_ISA.get_int(instr, 12, 14)

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs2 = RV_ISA.get_int(instr, 20, 24)
        rd = RV_ISA.get_int(instr, 7, 11)

        signed = False
        snoop_str = None
        snoop_op = None
        match type_val:
            case 0x0:
                # mul
                rs1_val = register_file[rs1].read_uint()
                rs2_val = register_file[rs2].read_uint()
                result = (rs1_val * rs2_val) % (1 << 32)  # Overflow is ignored
                snoop_str, snoop_op = "mul", "*"
            case 0x1:
                # mulh
                rs1_val = register_file[rs1].read_int()
                rs2_val = register_file[rs2].read_int()
                result = (rs1_val * rs2_val) >> 32
                signed = True
                snoop_str, snoop_op = "mulh", "*"
            case 0x2:
                # mulhsu
                rs1_val = register_file[rs1].read_int()
                rs2_val = register_file[rs2].read_uint()
                result = (rs1_val * rs2_val) >> 32
                signed = True
                snoop_str, snoop_op = "mulhsu", "*"
            case 0x3:
                # mulhu
                rs1_val = register_file[rs1].read_uint()
                rs2_val = register_file[rs2].read_uint()
                result = (rs1_val * rs2_val) >> 32
                snoop_str, snoop_op = "mulhu", "*"
            case 0x4:
                # div
                rs1_val = register_file[rs1].read_int()
                rs2_val = register_file[rs2].read_int()
                result = int(rs1_val / rs2_val)
                signed = True
                snoop_str, snoop_op = "div", "/"
            case 0x5:
                # divu
                rs1_val = register_file[rs1].read_uint()
                rs2_val = register_file[rs2].read_uint()
                result = int(rs1_val / rs2_val)
                snoop_str, snoop_op = "divu", "/"
            case 0x6:
                # rem
                rs1_val = register_file[rs1].read_int()
                rs2_val = register_file[rs2].read_int()
                result = rs1_val % rs2_val
                signed = True
                snoop_str, snoop_op = "rem", "%"
            case 0x7:
                # remu
                rs1_val = register_file[rs1].read_uint()
                rs2_val = register_file[rs2].read_uint()
                result = rs1_val % rs2_val
                snoop_str, snoop_op = "remu", "%"
            case _:
                return False

        register_file[rd].write(conv_to_bytes(result, signed=signed))
        assert snoop_str is not None
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{snoop_str} x{cls.get_reg_name(rd)}, x{cls.get_reg_name(rs1)}, x{cls.get_reg_name(rs2)}",
                f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} {snoop_op} {cls.get_reg_name(rs2)}",
            )
        return True
