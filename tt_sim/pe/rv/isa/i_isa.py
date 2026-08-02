from tt_sim.memory.memory import MemoryStall
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_int32, conv_to_uint32


class RV_I_ISA(RV_ISA):
    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        # ``instr`` is the 32-bit instruction word (see RV_ISA.fetch); the
        # handlers below index bit fields out of it directly.
        if instr is None:
            instr = cls.fetch(register_file, memory_space)

        match instr & 0x7F:
            case 0x37:
                return RV_I_ISA.handle_u_lui(instr, register_file, memory_space, snoop)
            case 0x17:
                return RV_I_ISA.handle_u_auipc(
                    instr, register_file, memory_space, snoop
                )
            case 0x6F:
                return RV_I_ISA.handle_j_jal(instr, register_file, memory_space, snoop)
            case 0x67:
                return RV_I_ISA.handle_i_jalr(instr, register_file, memory_space, snoop)
            case 0x63:
                return RV_I_ISA.handle_b_branch(
                    instr, register_file, memory_space, snoop
                )
            case 0x3:
                return RV_I_ISA.handle_i_load(instr, register_file, memory_space, snoop)
            case 0x23:
                return RV_I_ISA.handle_s_store(
                    instr, register_file, memory_space, snoop
                )
            case 0x13:
                return RV_I_ISA.handle_i_arith(
                    instr, register_file, memory_space, snoop
                )
            case 0x33:
                return RV_I_ISA.handle_r_arith(
                    instr, register_file, memory_space, snoop
                )
            case 0x0F:
                return RV_I_ISA.handle_i_fence(
                    instr, register_file, memory_space, snoop
                )
            case 0x73:
                return RV_I_ISA.handle_i_misc(instr, register_file, memory_space, snoop)
            case _:
                return False

    @classmethod
    def handle_u_lui(cls, instr, register_file, memory_space, snoop):
        rd = RV_ISA.get_int(instr, 7, 11)
        immediate = RV_I_ISA.extract_immediate(instr, "U")
        register_file[rd].write(conv_to_bytes(immediate))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"lui {cls.get_reg_name(rd)}, {hex(immediate)}",
                f"{cls.get_reg_name(rd)} = {hex(immediate)}",
            )
        return True

    @classmethod
    def handle_u_auipc(cls, instr, register_file, memory_space, snoop):
        rd = RV_ISA.get_int(instr, 7, 11)
        immediate = RV_I_ISA.extract_immediate(instr, "U")
        pc = register_file["pc"]
        pc_val = pc.read_uint()
        register_file[rd].write(conv_to_bytes(immediate + pc_val))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"auipc {cls.get_reg_name(rd)}, {hex(immediate)}",
                f"{cls.get_reg_name(rd)} = pc + {hex(immediate)}",
            )
        return True

    @classmethod
    def handle_j_jal(cls, instr, register_file, memory_space, snoop):
        rd = RV_ISA.get_int(instr, 7, 11)
        pc = register_file["pc"]
        pc_val = pc.read_uint()
        if rd > 0:
            # If provided register is x0 then don't store
            register_file[rd].write(
                conv_to_bytes(pc_val + 4)
            )  # Address of the next instruction

        offset = RV_I_ISA.extract_immediate(instr, "J")
        new_pc_val = pc_val + offset

        nextpc = register_file["nextpc"]
        nextpc.write(conv_to_bytes(new_pc_val))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"jal {cls.get_reg_name(rd)}, {hex(offset)}",
                f"jump to {hex(new_pc_val)}",
            )
        return True

    @classmethod
    def handle_i_jalr(cls, instr, register_file, memory_space, snoop):
        rd = RV_ISA.get_int(instr, 7, 11)
        pc = register_file["pc"]
        pc_val = pc.read_uint()
        if rd > 0:
            # If provided register is x0 then don't store
            register_file[rd].write(
                conv_to_bytes(pc_val + 4)
            )  # Address of the next instruction

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs1_val = register_file[rs1].read_uint()
        offset = RV_I_ISA.extract_immediate(instr, "I")

        new_pc_val = (rs1_val + offset) & ~1

        nextpc = register_file["nextpc"]
        nextpc.write(conv_to_bytes(new_pc_val))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"jalr {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                f"jump to {hex(new_pc_val)}",
            )
        return True

    @classmethod
    def handle_b_branch(cls, instr, register_file, memory_space, snoop):
        type_val = RV_ISA.get_int(instr, 12, 14)

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs1_val = register_file[rs1].read_uint()
        rs2 = RV_ISA.get_int(instr, 20, 24)
        rs2_val = register_file[rs2].read_uint()

        offset = RV_I_ISA.extract_immediate(instr, "B")

        pc = register_file["pc"]
        pc_val = pc.read_uint()
        new_pc_val = pc_val + offset

        nextpc = register_file["nextpc"]

        if type_val == 0x0:
            # beq
            info_msg = None
            if rs1_val == rs2_val:
                nextpc.write(conv_to_bytes(new_pc_val))
                info_msg = f"taken to {hex(new_pc_val)}" if snoop else None
            else:
                info_msg = "false"
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"beq {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}, {hex(offset)}",
                    info_msg,
                )
            return True
        elif type_val == 0x1:
            # bne
            info_msg = None
            if rs1_val != rs2_val:
                nextpc.write(conv_to_bytes(new_pc_val))
                info_msg = f"taken to {hex(new_pc_val)}" if snoop else None
            else:
                info_msg = "false"
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"bne {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}, {hex(offset)}",
                    info_msg,
                )
            return True
        elif type_val == 0x4 or type_val == 0x6:
            # blt (funct3 0x4, signed) or bltu (funct3 0x6, unsigned)
            instr_str = None
            if type_val == 0x4:
                rs1_val = register_file[rs1].read_int()
                rs2_val = register_file[rs2].read_int()
                instr_str = "blt"
            else:
                instr_str = "bltu"

            info_msg = None
            if rs1_val < rs2_val:
                nextpc.write(conv_to_bytes(new_pc_val))
                info_msg = f"taken to {hex(new_pc_val)}" if snoop else None
            else:
                info_msg = "false"
            assert instr_str is not None
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"{instr_str} {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}, {hex(offset)}",
                    info_msg,
                )
            return True
        elif type_val == 0x5 or type_val == 0x7:
            # bge (funct3 0x5, signed) or bgeu (funct3 0x7, unsigned)
            instr_str = None
            if type_val == 0x5:
                rs1_val = register_file[rs1].read_int()
                rs2_val = register_file[rs2].read_int()
                instr_str = "bge"
            else:
                instr_str = "bgeu"

            info_msg = None
            if rs1_val >= rs2_val:
                nextpc.write(conv_to_bytes(new_pc_val))
                info_msg = f"taken to {hex(new_pc_val)}" if snoop else None
            else:
                info_msg = "false"
            assert instr_str is not None
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"{instr_str} {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}, {hex(offset)}",
                    info_msg,
                )
            return True
        else:
            return False

    @classmethod
    def handle_i_load(cls, instr, register_file, memory_space, snoop):
        type_val = RV_ISA.get_int(instr, 12, 14)

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs1_val = register_file[rs1].read_uint()
        rd = RV_ISA.get_int(instr, 7, 11)
        offset = RV_I_ISA.extract_immediate(instr, "I")

        tgt_mem_address = rs1_val + offset

        write_result = True
        if type_val == 0x0:
            # lb
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"lb {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"{cls.get_reg_name(rd)} = mem[{hex(tgt_mem_address)}]",
                )
            byte_val = memory_space.read(tgt_mem_address, 1)
            if byte_val != MemoryStall:
                result = conv_to_bytes(
                    RV_I_ISA.sign_extend(conv_to_int32(byte_val), 8), signed=True
                )

        elif type_val == 0x4:
            # lu
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"lu {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"{cls.get_reg_name(rd)} = mem[{hex(tgt_mem_address)}]",
                )
            byte_val = memory_space.read(tgt_mem_address, 1)
            if byte_val != MemoryStall:
                result = conv_to_bytes(
                    RV_I_ISA.zero_extend(conv_to_uint32(byte_val), 8)
                )

        elif type_val == 0x1:
            # lh
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"lh {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"{cls.get_reg_name(rd)} = mem[{hex(tgt_mem_address)}]",
                )
            byte_val = memory_space.read(tgt_mem_address, 2)
            if byte_val != MemoryStall:
                result = conv_to_bytes(
                    RV_I_ISA.sign_extend(conv_to_int32(byte_val), 16), signed=True
                )

        elif type_val == 0x5:
            # lhu
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"lhu {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"{cls.get_reg_name(rd)} = mem[{hex(tgt_mem_address)}]",
                )
            byte_val = memory_space.read(tgt_mem_address, 2)
            if byte_val != MemoryStall:
                result = conv_to_bytes(
                    RV_I_ISA.zero_extend(conv_to_uint32(byte_val), 16)
                )

        elif type_val == 0x2:
            # lw
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"lw {cls.get_reg_name(rd)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"{cls.get_reg_name(rd)} = mem[{hex(tgt_mem_address)}]",
                )
            result = memory_space.read(tgt_mem_address, 4)
        else:
            write_result = False

        if write_result and result != MemoryStall:
            register_file[rd].write(result)
            return True
        elif result == MemoryStall:
            return ProcessingElement.PEStall
        else:
            return False

    @classmethod
    def handle_s_store(cls, instr, register_file, memory_space, snoop):
        type_val = RV_ISA.get_int(instr, 12, 14)

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs1_val = register_file[rs1].read_uint()
        rs2 = RV_ISA.get_int(instr, 20, 24)

        offset = RV_I_ISA.extract_immediate(instr, "S")
        tgt_mem_address = rs1_val + offset

        rs2_val = register_file[rs2].read()

        ret_val = None
        if type_val == 0x0:
            # sb
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"sb {cls.get_reg_name(rs2)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"mem[{hex(tgt_mem_address)}] = {cls.get_reg_name(rs2)}",
                )
            ret_val = memory_space.write(tgt_mem_address, conv_to_bytes(rs2_val[0], 1))
        elif type_val == 0x1:
            # sh
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"sh {cls.get_reg_name(rs2)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"mem[{hex(tgt_mem_address)}] = {cls.get_reg_name(rs2)}",
                )
            ret_val = memory_space.write(
                tgt_mem_address, conv_to_bytes(rs2_val[0:1], 2)
            )
        elif type_val == 0x2:
            # sw
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"sw {cls.get_reg_name(rs2)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                    f"mem[{hex(tgt_mem_address)}] = {cls.get_reg_name(rs2)}",
                )
            ret_val = memory_space.write(tgt_mem_address, rs2_val)
        else:
            return False

        if ret_val == MemoryStall:
            return ProcessingElement.PEStall
        else:
            return True

    @classmethod
    def handle_i_arith(cls, instr, register_file, memory_space, snoop):
        type_val = RV_ISA.get_int(instr, 12, 14)

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs1_val = register_file[rs1].read_uint()
        rd1 = RV_ISA.get_int(instr, 7, 11)

        signed_op = False
        write_result = True
        snoop_str = None
        info_msg = None
        if (
            type_val == 0x0
            or type_val == 0x2
            or type_val == 0x3
            or type_val == 0x4
            or type_val == 0x6
            or type_val == 0x7
        ):
            immediate = RV_I_ISA.extract_immediate(instr, "I")
            immediate_unsigned = immediate & 0xFFFFFFFF

            if type_val == 0x0:
                # addi
                result = (rs1_val + immediate) % (1 << 32)  # Overflow is ignored
                snoop_str = "addi"
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} + {hex(immediate)}"
            elif type_val == 0x2 or type_val == 0x3:
                # slti and sltiu
                if type_val == 0x2:
                    rs1_val = register_file[rs1].read_int()
                    signed_op = True
                    snoop_str = "slti"
                    result = 1 if rs1_val < immediate else 0
                else:
                    snoop_str = "sltiu"
                    result = 1 if rs1_val < immediate_unsigned else 0

                if snoop:
                    info_msg = (
                        f"{cls.get_reg_name(rd1)} = 1 if {cls.get_reg_name(rs1)} < "
                        f"{hex(immediate if signed_op else immediate_unsigned)} else 0 : "
                        f"{'TRUE' if result == 1 else 'FALSE'}"
                    )
            elif type_val == 0x4:
                # xori
                result = rs1_val ^ immediate
                snoop_str = "xori"
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} ^ {hex(immediate)}"
            elif type_val == 0x6:
                # ori
                result = rs1_val | immediate
                snoop_str = "ori"
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} | {hex(immediate)}"
            elif type_val == 0x7:
                # andi
                result = rs1_val & immediate
                snoop_str = "andi"
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} & {hex(immediate)}"
            else:
                write_result = False
            if write_result:
                assert snoop_str is not None
                if snoop:
                    RV_ISA.print_snoop(
                        snoop,
                        f"{snoop_str} {cls.get_reg_name(rd1)}, {cls.get_reg_name(rs1)}, {hex(immediate)}",
                        info_msg,
                    )
        elif type_val == 0x1 or type_val == 0x5:
            # Base shift-immediate uses shamt in bits [20:24] with a fixed high
            # field: 0x00 for slli/srli, 0x20 for srai. Any other high field is
            # a Zbb single-bit-manip op (clz/ctz/cpop/sext.*/rori/rev8/orc.b),
            # which must fall through to that extension rather than being run as
            # a shift.
            high = RV_ISA.get_int(instr, 25, 31)
            if type_val == 0x1 and high != 0x00:
                return False
            if type_val == 0x5 and high not in (0x00, 0x20):
                return False
            bit_pos = RV_ISA.get_int(instr, 20, 24)
            if type_val == 0x1:
                # slli
                result = (rs1_val << bit_pos) % (1 << 32)  # Overflow is ignored
                snoop_str = "slli"
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} << {hex(bit_pos)}"
            elif type_val == 0x5:
                # srli or srai
                arithmetic_variant = RV_ISA.get_int(instr, 30, 30) == 1
                result = rs1_val >> bit_pos
                if arithmetic_variant:
                    result = RV_I_ISA.sign_extend_shift(rs1_val, result, bit_pos)
                    snoop_str = "srai"
                else:
                    snoop_str = "srli"
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} >> {hex(bit_pos)}"
            else:
                write_result = False
            if write_result:
                assert snoop_str is not None
                if snoop:
                    RV_ISA.print_snoop(
                        snoop,
                        f"{snoop_str} {cls.get_reg_name(rd1)}, {cls.get_reg_name(rs1)}, {hex(bit_pos)}",
                        info_msg,
                    )

        if write_result:
            register_file[rd1].write(conv_to_bytes(result, signed=signed_op))
            return True
        else:
            return False

    @classmethod
    def handle_r_arith(cls, instr, register_file, memory_space, snoop):
        type_val = RV_ISA.get_int(instr, 12, 14)

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs1_val = register_file[rs1].read_uint()
        rs2 = RV_ISA.get_int(instr, 20, 24)
        rs2_val = register_file[rs2].read_uint()
        rd = RV_ISA.get_int(instr, 7, 11)

        # Base RV32I R-type uses funct7 == 0x00 for every funct3, plus 0x20 for
        # the two "alternate" ops sub (funct3 0) and sra (funct3 5). Any other
        # funct7 belongs to an extension (M's 0x01, Zba's 0x10, Zbb's
        # 0x04/0x05/0x30, ...) and must be left for the next ISA to decode —
        # otherwise, e.g., zext.h (funct7 0x04, funct3 4) is silently executed
        # as xor and andn (funct7 0x20, funct3 7) as and.
        funct7 = RV_ISA.get_int(instr, 25, 31)
        if funct7 == 0x20:
            if type_val not in (0x0, 0x5):
                return False
        elif funct7 != 0x00:
            return False

        signed_op = False
        write_result = True
        snoop_str = None
        info_msg = None
        if type_val == 0x0:
            # add and sub
            is_sub = RV_ISA.get_int(instr, 30, 30) == 1
            if is_sub:
                snoop_str = "sub"
                result = (rs1_val - rs2_val) % (1 << 32)  # Overflow is ignored
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} - {cls.get_reg_name(rs2)}"
            else:
                snoop_str = "add"
                result = (rs1_val + rs2_val) % (1 << 32)  # Overflow is ignored
                if snoop:
                    info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} + {cls.get_reg_name(rs2)}"
        elif type_val == 0x1:
            # sll
            shift_bits = rs2_val & 0x1F  # Least significant 5 bits for RV32I
            result = (rs1_val << shift_bits) % (1 << 32)  # Overflow is ignored
            snoop_str = "sll"
            if snoop:
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} << {hex(shift_bits)}"
        elif type_val == 0x2 or type_val == 0x3:
            # slt or sltu
            if type_val == 0x2:
                rs1_val = register_file[rs1].read_int()
                rs2_val = register_file[rs2].read_int()
                signed_op = True
                snoop_str = "slt"
            else:
                snoop_str = "sltu"
            result = 1 if rs1_val < rs2_val else 0
            if snoop:
                info_msg = (
                    f"{cls.get_reg_name(rd)} = 1 if {cls.get_reg_name(rs1)} < "
                    f"{cls.get_reg_name(rs2)} else 0 : {'TRUE' if result == 1 else 'FALSE'}"
                )
        elif type_val == 0x4:
            # xor
            result = rs1_val ^ rs2_val
            snoop_str = "xor"
            if snoop:
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} ^ {cls.get_reg_name(rs2)}"
        elif type_val == 0x5:
            # srl or sra
            arithmetic_variant = RV_ISA.get_int(instr, 30, 30) == 1
            shift_bits = rs2_val & 0x1F  # Least significant 5 bits for RV32I
            result = rs1_val >> shift_bits
            if arithmetic_variant:
                result = RV_I_ISA.sign_extend_shift(rs1_val, result, shift_bits)
                snoop_str = "sra"
            else:
                snoop_str = "srl"
            if snoop:
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} >> {hex(shift_bits)}"
        elif type_val == 0x6:
            # or
            result = rs1_val | rs2_val
            snoop_str = "or"
            if snoop:
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} | {cls.get_reg_name(rs2)}"
        elif type_val == 0x7:
            # and
            result = rs1_val & rs2_val
            snoop_str = "and"
            if snoop:
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} & {cls.get_reg_name(rs2)}"
        else:
            write_result = False

        if write_result:
            assert snoop_str is not None
            if snoop:
                RV_ISA.print_snoop(
                    snoop,
                    f"{snoop_str} {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}",
                    info_msg,
                )
            register_file[rd].write(conv_to_bytes(result, signed=signed_op))
            return True
        else:
            return False

    @classmethod
    def handle_i_fence(cls, instr, register_file, memory_space, snoop):
        if snoop:
            i_variant = RV_ISA.get_int(instr, 12, 12) == 1
            if i_variant:
                RV_ISA.print_snoop(snoop, "fence.i", "ignored")
            else:
                RV_ISA.print_snoop(snoop, "fence", "ignored")
        return True

    @classmethod
    def handle_i_misc(cls, instr, register_file, memory_space, snoop):
        if snoop:
            type_val = RV_ISA.get_int(instr, 12, 14)
            if type_val == 0x0:
                # ecall or ebreak
                is_ebreak = RV_ISA.get_int(instr, 20, 20) == 1
                if is_ebreak:
                    RV_ISA.print_snoop(snoop, "ebreak", "ignored")
                else:
                    RV_ISA.print_snoop(snoop, "ecall", "ignored")
            elif type_val == 0x1:
                pass
        return True

    @classmethod
    def sign_extend(cls, value, bit_width):
        """Sign-extend a value from bit_width to 32 bits."""
        sign_bit = (value >> (bit_width - 1)) & 1
        mask = (1 << bit_width) - 1
        if sign_bit:
            return value | (~0 << bit_width)  # Set upper bits to 1
        else:
            return value & mask  # Clear upper bits

    @classmethod
    def sign_extend_shift(cls, operand, shifted, shift_bits):
        """Fill in the bits an arithmetic right shift vacates.

        ``shifted`` is ``operand >> shift_bits`` over the unsigned 32-bit
        operand, so Python has already shifted zeroes in at the top. sra/srai
        replicate the operand's sign bit instead, so set the top
        ``shift_bits`` bits when the operand is negative.
        """
        if not (operand >> 31) & 1:
            return shifted
        return (shifted | (0xFFFFFFFF << (32 - shift_bits))) & 0xFFFFFFFF

    @classmethod
    def zero_extend(cls, value, bit_width):
        """Zero-extend a value from bit_width to 32 bits."""
        if bit_width > 32 or bit_width < 1:
            raise ValueError(f"Bit width must be between 1 and 32, got {bit_width}")

        # Mask to keep only the lower 'bit_width' bits
        mask = (1 << bit_width) - 1
        return value & mask

    @classmethod
    def extract_immediate(cls, instruction, inst_type):
        """
        Extract and sign-extend the immediate value from a RISC-V instruction.

        Parameters:
        - instruction: 32-bit instruction as an integer
        - inst_type: String indicating the instruction type ('I', 'S', 'B', 'U', 'J')

        Returns:
        - 32-bit sign-extended immediate value as an integer
        """

        if inst_type == "I":
            # I-type: imm[11:0] = inst[31:20]
            imm = (instruction >> 20) & 0xFFF  # Extract bits 31:20 (12 bits)
            return RV_I_ISA.sign_extend(imm, 12)

        elif inst_type == "S":
            # S-type: imm[11:5] = inst[31:25], imm[4:0] = inst[11:7]
            imm = ((instruction >> 25) & 0x7F) << 5  # Bits 31:25 -> imm[11:5]
            imm |= (instruction >> 7) & 0x1F  # Bits 11:7 -> imm[4:0]
            return RV_I_ISA.sign_extend(imm, 12)

        elif inst_type == "B":
            # B-type: imm[12|10:5|4:1|11] = inst[31|30:25|11:8|7]
            imm = ((instruction >> 31) & 0x1) << 12  # Bit 31 -> imm[12]
            imm |= ((instruction >> 7) & 0x1) << 11  # Bit 7 -> imm[11]
            imm |= ((instruction >> 25) & 0x3F) << 5  # Bits 30:25 -> imm[10:5]
            imm |= ((instruction >> 8) & 0xF) << 1  # Bits 11:8 -> imm[4:1]
            imm &= ~1  # Bit 0 is always 0 for B-type
            return RV_I_ISA.sign_extend(imm, 13)

        elif inst_type == "U":
            # U-type: imm[31:12] = inst[31:12]
            imm = instruction & 0xFFFFF000  # Bits 31:12, shifted already
            return imm  # No sign extension needed, upper 20 bits

        elif inst_type == "J":
            # J-type: imm[20|10:1|11|19:12] = inst[31|30:21|20|19:12]
            imm = ((instruction >> 31) & 0x1) << 20  # Bit 31 -> imm[20]
            imm |= ((instruction >> 12) & 0xFF) << 12  # Bits 19:12 -> imm[19:12]
            imm |= ((instruction >> 20) & 0x1) << 11  # Bit 20 -> imm[11]
            imm |= ((instruction >> 21) & 0x3FF) << 1  # Bits 30:21 -> imm[10:1]
            imm &= ~1  # Bit 0 is always 0 for J-type
            return RV_I_ISA.sign_extend(imm, 21)

        else:
            raise ValueError(f"Unsupported instruction type: {inst_type}")
