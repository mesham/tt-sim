from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_int32, conv_to_uint32


class RV_M_ISA(RV_ISA):
    @classmethod
    def run(cls, register_file, memory_space, snoop):
        pc = register_file["pc"]
        addr = conv_to_uint32(pc.read())
        instr = memory_space.read(addr, 4)

        opcode_bin = RV_ISA.get_bits(instr, 0, 6)
        opcode_bin.reverse()
        opcode = RV_ISA.bits_to_int(opcode_bin)

        if opcode != 0x33:
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
        info_msg = None
        match type_val:
            case 0x0:
                # mul
                rs1_val = conv_to_uint32(register_file[rs1].read())
                rs2_val = conv_to_uint32(register_file[rs2].read())
                result = (rs1_val * rs2_val) % (1 << 32)  # Overflow is ignored
                snoop_str = "mul"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} * {cls.get_reg_name(rs2)}"
            case 0x1:
                # mulh
                rs1_val = conv_to_int32(register_file[rs1].read())
                rs2_val = conv_to_int32(register_file[rs2].read())
                result = (rs1_val * rs2_val) >> 16
                signed = True
                snoop_str = "mulh"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} * {cls.get_reg_name(rs2)}"
            case 0x2:
                # mulhsu
                rs1_val = conv_to_int32(register_file[rs1].read())
                rs2_val = conv_to_uint32(register_file[rs2].read())
                result = (rs1_val * rs2_val) >> 16
                signed = True
                snoop_str = "mulhsu"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} * {cls.get_reg_name(rs2)}"
            case 0x3:
                # mulhu
                rs1_val = conv_to_uint32(register_file[rs1].read())
                rs2_val = conv_to_uint32(register_file[rs2].read())
                result = (rs1_val * rs2_val) >> 16
                snoop_str = "mulhu"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} * {cls.get_reg_name(rs2)}"
            case 0x4:
                # div
                rs1_val = conv_to_int32(register_file[rs1].read())
                rs2_val = conv_to_int32(register_file[rs2].read())
                result = int(rs1_val / rs2_val)
                signed = True
                snoop_str = "div"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} / {cls.get_reg_name(rs2)}"
            case 0x5:
                # divu
                rs1_val = conv_to_uint32(register_file[rs1].read())
                rs2_val = conv_to_uint32(register_file[rs2].read())
                result = int(rs1_val / rs2_val)
                snoop_str = "divu"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} / {cls.get_reg_name(rs2)}"
            case 0x6:
                # rem
                rs1_val = conv_to_int32(register_file[rs1].read())
                rs2_val = conv_to_int32(register_file[rs2].read())
                result = rs1_val % rs2_val
                signed = True
                snoop_str = "rem"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} % {cls.get_reg_name(rs2)}"
            case 0x7:
                # remu
                rs1_val = conv_to_uint32(register_file[rs1].read())
                rs2_val = conv_to_uint32(register_file[rs2].read())
                result = rs1_val % rs2_val
                snoop_str = "remu"
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} % {cls.get_reg_name(rs2)}"
            case _:
                return False

        register_file[rd].write(conv_to_bytes(result, signed=signed))
        assert snoop_str is not None
        RV_ISA.print_snoop(
            snoop,
            f"{snoop_str} x{cls.get_reg_name(rd)}, x{cls.get_reg_name(rs1)}, x{cls.get_reg_name(rs2)}",
            info_msg,
        )
        return True
