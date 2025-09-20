from tt_sim.memory.memory import MemoryStall
from tt_sim.pe.pe import ProcessingElement
from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_int32, conv_to_uint32


class RV_I_ISA(RV_ISA):
    @classmethod
    def run(cls, register_file, memory_space, snoop):
        pc = register_file["pc"]
        addr = conv_to_uint32(pc.read())
        instr = memory_space.read(addr, 4)

        opcode_bin = RV_ISA.get_bits(instr, 0, 6)
        opcode_bin.reverse()
        opcode = RV_ISA.bits_to_int(opcode_bin)

        match opcode:
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
        immediate = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "U")
        register_file[rd].write(conv_to_bytes(immediate))
        RV_ISA.print_snoop(
            snoop,
            f"lui {cls.get_reg_name(rd)}, {hex(immediate)}",
            f"{cls.get_reg_name(rd)} = {hex(immediate)}",
        )
        return True

    @classmethod
    def handle_u_auipc(cls, instr, register_file, memory_space, snoop):
        rd = RV_ISA.get_int(instr, 7, 11)
        immediate = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "U")
        pc = register_file["pc"]
        pc_val = conv_to_uint32(pc.read())
        register_file[rd].write(conv_to_bytes(immediate + pc_val))
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
        pc_val = conv_to_uint32(pc.read())
        if rd > 0:
            # If provided register is x0 then don't store
            register_file[rd].write(
                conv_to_bytes(pc_val + 4)
            )  # Address of the next instruction

        offset = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "J")
        new_pc_val = pc_val + offset

        nextpc = register_file["nextpc"]
        nextpc.write(conv_to_bytes(new_pc_val))
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
        pc_val = conv_to_uint32(pc.read())
        if rd > 0:
            # If provided register is x0 then don't store
            register_file[rd].write(
                conv_to_bytes(pc_val + 4)
            )  # Address of the next instruction

        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs1_val = conv_to_uint32(register_file[rs1].read())
        offset = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "I")

        new_pc_val = (rs1_val + offset) & ~1

        nextpc = register_file["nextpc"]
        nextpc.write(conv_to_bytes(new_pc_val))
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
        rs1_val = conv_to_uint32(register_file[rs1].read())
        rs2 = RV_ISA.get_int(instr, 20, 24)
        rs2_val = conv_to_uint32(register_file[rs2].read())

        offset = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "B")

        pc = register_file["pc"]
        pc_val = conv_to_uint32(pc.read())
        new_pc_val = pc_val + offset

        nextpc = register_file["nextpc"]

        if type_val == 0x0:
            # beq
            info_msg = None
            if rs1_val == rs2_val:
                nextpc.write(conv_to_bytes(new_pc_val))
                info_msg = f"taken to {hex(new_pc_val)}"
            else:
                info_msg = "false"
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
                info_msg = f"taken to {hex(new_pc_val)}"
            else:
                info_msg = "false"
            RV_ISA.print_snoop(
                snoop,
                f"bne {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}, {hex(offset)}",
                info_msg,
            )
            return True
        elif type_val == 0x4 or type_val == 0x6:
            # blt or blu
            instr_str = None
            if type_val == 0x6:
                rs1_val = conv_to_int32(register_file[rs1].read())
                rs2_val = conv_to_int32(register_file[rs2].read())
                instr_str = "blu"
            else:
                instr_str = "blt"

            info_msg = None
            if rs1_val < rs2_val:
                nextpc.write(conv_to_bytes(new_pc_val))
                info_msg = f"taken to {hex(new_pc_val)}"
            else:
                info_msg = "false"
            assert instr_str is not None
            RV_ISA.print_snoop(
                snoop,
                f"{instr_str} {cls.get_reg_name(rs1)}, {cls.get_reg_name(rs2)}, {hex(offset)}",
                info_msg,
            )
            return True
        elif type_val == 0x5 or type_val == 0x7:
            # bge or bgeu
            instr_str = None
            if type_val == 0x7:
                rs1_val = conv_to_int32(register_file[rs1].read())
                rs2_val = conv_to_int32(register_file[rs2].read())
                instr_str = "bgeu"
            else:
                instr_str = "bge"

            info_msg = None
            if rs1_val >= rs2_val:
                nextpc.write(conv_to_bytes(new_pc_val))
                info_msg = f"taken to {hex(new_pc_val)}"
            else:
                info_msg = "false"
            assert instr_str is not None
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
        rs1_val = conv_to_uint32(register_file[rs1].read())
        rd = RV_ISA.get_int(instr, 7, 11)
        offset = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "I")

        tgt_mem_address = rs1_val + offset

        write_result = True
        if type_val == 0x0:
            # lb
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
        rs1_val = conv_to_uint32(register_file[rs1].read())
        rs2 = RV_ISA.get_int(instr, 20, 24)

        offset = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "S")
        tgt_mem_address = rs1_val + offset

        rs2_val = register_file[rs2].read()

        ret_val = None
        if type_val == 0x0:
            # sb
            RV_ISA.print_snoop(
                snoop,
                f"sb {cls.get_reg_name(rs2)}, {hex(offset)}({cls.get_reg_name(rs1)})",
                f"mem[{hex(tgt_mem_address)}] = {cls.get_reg_name(rs2)}",
            )
            ret_val = memory_space.write(tgt_mem_address, conv_to_bytes(rs2_val[0], 1))
        elif type_val == 0x1:
            # sh
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
        rs1_val = conv_to_uint32(register_file[rs1].read())
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
            immediate = RV_I_ISA.extract_immediate(RV_ISA.get_int(instr, 0, 31), "I")
            immediate_unsigned = immediate & 0xFFFFFFFF

            if type_val == 0x0:
                # addi
                result = (rs1_val + immediate) % (1 << 32)  # Overflow is ignored
                snoop_str = "addi"
                info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} + {hex(immediate)}"
            elif type_val == 0x2 or type_val == 0x3:
                # slti and sltiu
                if type_val == 0x2:
                    rs1_val = conv_to_int32(register_file[rs1].read())
                    signed_op = True
                    snoop_str = "slti"
                    result = 1 if rs1_val < immediate else 0
                else:
                    snoop_str = "sltiu"
                    result = 1 if rs1_val < immediate_unsigned else 0

                info_msg = (
                    f"{cls.get_reg_name(rd1)} = 1 if {cls.get_reg_name(rs1)} < "
                    f"{hex(immediate if signed_op else immediate_unsigned)} else 0 : "
                    f"{'TRUE' if result == 1 else 'FALSE'}"
                )
            elif type_val == 0x4:
                # xori
                result = rs1_val ^ immediate
                snoop_str = "xori"
                info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} ^ {hex(immediate)}"
            elif type_val == 0x6:
                # ori
                result = rs1_val | immediate
                snoop_str = "ori"
                info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} | {hex(immediate)}"
            elif type_val == 0x7:
                # andi
                result = rs1_val & immediate
                snoop_str = "andi"
                info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} & {hex(immediate)}"
            else:
                write_result = False
            if write_result:
                assert snoop_str is not None
                RV_ISA.print_snoop(
                    snoop,
                    f"{snoop_str} {cls.get_reg_name(rd1)}, {cls.get_reg_name(rs1)}, {hex(immediate)}",
                    info_msg,
                )
        elif type_val == 0x1 or type_val == 0x5:
            bit_pos = RV_ISA.get_int(instr, 20, 25)
            if type_val == 0x1:
                # slli
                result = (rs1_val << bit_pos) % (1 << 32)  # Overflow is ignored
                snoop_str = "slli"
                info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} << {hex(bit_pos)}"
            elif type_val == 0x5:
                # srli or srai
                arithmetic_variant = RV_ISA.get_int(instr, 30, 30) == 1
                if arithmetic_variant:
                    msb = (rs1 >> 31) & 0x1
                    mask = (1 << bit_pos) - 1
                    result = rs1_val >> bit_pos
                    if msb:
                        result = result | (~0 << bit_pos)  # Set upper bits to 1
                    else:
                        result = result & mask  # Clear upper bits
                    snoop_str = "srai"
                else:
                    result = rs1_val >> bit_pos
                    snoop_str = "srli"
                info_msg = f"{cls.get_reg_name(rd1)} = {cls.get_reg_name(rs1)} >> {hex(bit_pos)}"
            else:
                write_result = False
            if write_result:
                assert snoop_str is not None
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
        rs1_val = conv_to_uint32(register_file[rs1].read())
        rs2 = RV_ISA.get_int(instr, 20, 24)
        rs2_val = conv_to_uint32(register_file[rs2].read())
        rd = RV_ISA.get_int(instr, 7, 11)

        # The i variant of r has a zero at location 25
        i_variant = RV_ISA.get_int(instr, 25, 25) == 0
        if not i_variant:
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
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} - {cls.get_reg_name(rs2)}"
            else:
                snoop_str = "add"
                result = (rs1_val + rs2_val) % (1 << 32)  # Overflow is ignored
                info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} + {cls.get_reg_name(rs2)}"
        elif type_val == 0x1:
            # sll
            shift_bits = rs2_val & 0x1F  # Least significant 5 bits for RV32I
            result = (rs1_val << shift_bits) % (1 << 32)  # Overflow is ignored
            snoop_str = "sll"
            info_msg = (
                f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} << {hex(shift_bits)}"
            )
        elif type_val == 0x2 or type_val == 0x3:
            # slt or sltu
            if type_val == 0x2:
                rs1_val = conv_to_int32(register_file[rs1].read())
                rs2_val = conv_to_int32(register_file[rs2].read())
                signed_op = True
                snoop_str = "slt"
            else:
                snoop_str = "sltu"
            result = 1 if rs1_val < rs2_val else 0
            info_msg = (
                f"{cls.get_reg_name(rd)} = 1 if {cls.get_reg_name(rs1)} < "
                f"{cls.get_reg_name(rs2)} else 0 : {'TRUE' if result == 1 else 'FALSE'}"
            )
        elif type_val == 0x4:
            # xor
            result = rs1_val ^ rs2_val
            snoop_str = "xor"
            info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} ^ {cls.get_reg_name(rs2)}"
        elif type_val == 0x5:
            # srl or sra
            arithmetic_variant = RV_ISA.get_int(instr, 30, 30) == 1
            shift_bits = rs2_val & 0x1F  # Least significant 5 bits for RV32I
            if arithmetic_variant:
                msb = (rs1 >> 31) & 0x1
                mask = (1 << shift_bits) - 1
                result = rs1_val >> shift_bits
                if msb:
                    result = result | (~0 << shift_bits)  # Set upper bits to 1
                else:
                    result = result & mask  # Clear upper bits
                snoop_str = "sra"
            else:
                result = rs1_val >> shift_bits
                snoop_str = "srl"
            info_msg = (
                f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} >> {hex(shift_bits)}"
            )
        elif type_val == 0x6:
            # or
            result = rs1_val | rs2_val
            snoop_str = "or"
            info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} | {cls.get_reg_name(rs2)}"
        elif type_val == 0x7:
            # and
            result = rs1_val & rs2_val
            snoop_str = "and"
            info_msg = f"{cls.get_reg_name(rd)} = {cls.get_reg_name(rs1)} & {cls.get_reg_name(rs2)}"
        else:
            write_result = False

        if write_result:
            assert snoop_str is not None
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
