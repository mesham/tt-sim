"""Bit-manipulation extensions (Zba address-generation, Zbb basic bit-manip).

Per the Blackhole baby-RISC-V instruction set
(``BlackholeA0/TensixTile/BabyRISCV/InstructionSet.md``) the cores implement:

- **Zba**: ``sh1add``, ``sh2add``, ``sh3add``
- **Zbb**: ``andn``, ``clz``, ``cpop``, ``ctz``, ``max``, ``maxu``, ``min``,
  ``minu``, ``orc.b``, ``orn``, ``rev8``, ``rol``, ``ror``, ``rori``,
  ``sext.b``, ``sext.h``, ``xnor``, ``zext.h``

Wormhole baby cores are RV32IM-only and do **not** implement these, so these
ISAs are attached to Blackhole baby cores alone (see ``BabyRISCV``).

Each class follows the ``RV_ISA.run`` contract: decode the instruction at the
PC; if it belongs to this extension, execute it and return ``True``; otherwise
return ``False`` so the next ISA in the core's list gets a chance. The base
``RV_I_ISA`` is strict about ``funct7`` / shift-immediate high bits, so these
encodings fall through to here rather than being mis-decoded as base ops.
"""

from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_bytes

_MASK32 = 0xFFFFFFFF


def _decode_r(instr):
    return (
        RV_ISA.get_int(instr, 12, 14),  # funct3
        RV_ISA.get_int(instr, 15, 19),  # rs1
        RV_ISA.get_int(instr, 20, 24),  # rs2
        RV_ISA.get_int(instr, 7, 11),  # rd
        RV_ISA.get_int(instr, 25, 31),  # funct7
    )


class RV_ZBA_ISA(RV_ISA):
    """Zba address-generation: ``shNadd rd, rs1, rs2 = (rs1 << N) + rs2``."""

    # funct3 -> shift amount for the shNadd family (funct7 == 0x10).
    _SHADD = {0x2: 1, 0x4: 2, 0x6: 3}

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        if instr & 0x7F != 0x33:
            return False
        funct3, rs1, rs2, rd, funct7 = _decode_r(instr)
        if funct7 != 0x10 or funct3 not in cls._SHADD:
            return False
        shift = cls._SHADD[funct3]
        rs1_val = register_file[rs1].read_uint()
        rs2_val = register_file[rs2].read_uint()
        result = ((rs1_val << shift) + rs2_val) & _MASK32
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"sh{shift}add {cls.get_reg_name(rd)}, {cls.get_reg_name(rs1)}, "
                f"{cls.get_reg_name(rs2)}",
                f"{cls.get_reg_name(rd)} = ({cls.get_reg_name(rs1)} << {shift}) + "
                f"{cls.get_reg_name(rs2)}",
            )
        register_file[rd].write(conv_to_bytes(result))
        return True


def _clz(val):
    return 32 if val == 0 else 32 - val.bit_length()


def _ctz(val):
    return 32 if val == 0 else (val & -val).bit_length() - 1


def _rotate_right(val, amount):
    amount &= 0x1F
    if amount == 0:
        return val & _MASK32
    return ((val >> amount) | (val << (32 - amount))) & _MASK32


def _sign_extend(val, bits):
    sign = 1 << (bits - 1)
    return ((val & ((1 << bits) - 1)) ^ sign) - sign & _MASK32


def _orc_b(val):
    out = 0
    for i in range(4):
        if (val >> (i * 8)) & 0xFF:
            out |= 0xFF << (i * 8)
    return out


def _rev8(val):
    return (
        ((val & 0x000000FF) << 24)
        | ((val & 0x0000FF00) << 8)
        | ((val & 0x00FF0000) >> 8)
        | ((val & 0xFF000000) >> 24)
    )


class RV_ZBB_ISA(RV_ISA):
    """Zbb basic bit-manipulation (logic-with-negate, min/max, count, rotate,
    byte/half sign/zero-extend, byte reverse, or-combine)."""

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        if opcode == 0x33:
            return cls._run_r(instr, register_file, snoop)
        if opcode == 0x13:
            return cls._run_i(instr, register_file, snoop)
        return False

    @classmethod
    def _run_r(cls, instr, register_file, snoop):
        funct3, rs1, rs2, rd, funct7 = _decode_r(instr)
        a = register_file[rs1].read_uint()
        b = register_file[rs2].read_uint()

        name = None
        result = 0
        if funct7 == 0x20:  # logic-with-negate (funct7 shared with sub/sra)
            if funct3 == 0x7:
                name, result = "andn", a & (~b & _MASK32)
            elif funct3 == 0x6:
                name, result = "orn", a | (~b & _MASK32)
            elif funct3 == 0x4:
                name, result = "xnor", ~(a ^ b) & _MASK32
        elif funct7 == 0x05:  # min / max
            sa = a - (1 << 32) if a & 0x80000000 else a
            sb = b - (1 << 32) if b & 0x80000000 else b
            if funct3 == 0x4:
                name, result = "min", (a if sa < sb else b)
            elif funct3 == 0x5:
                name, result = "minu", (a if a < b else b)
            elif funct3 == 0x6:
                name, result = "max", (a if sa > sb else b)
            elif funct3 == 0x7:
                name, result = "maxu", (a if a > b else b)
        elif funct7 == 0x30:  # rotate
            if funct3 == 0x1:
                name, result = "rol", _rotate_right(a, 32 - (b & 0x1F))
            elif funct3 == 0x5:
                name, result = "ror", _rotate_right(a, b)
        elif funct7 == 0x04 and funct3 == 0x4 and rs2 == 0x0:  # zext.h (RV32)
            name, result = "zext.h", a & 0xFFFF

        if name is None:
            return False
        if snoop:
            RV_ISA.print_snoop(snoop, f"{name} {cls.get_reg_name(rd)}", None)
        register_file[rd].write(conv_to_bytes(result & _MASK32))
        return True

    @classmethod
    def _run_i(cls, instr, register_file, snoop):
        funct3 = RV_ISA.get_int(instr, 12, 14)
        rs1 = RV_ISA.get_int(instr, 15, 19)
        rd = RV_ISA.get_int(instr, 7, 11)
        upper = RV_ISA.get_int(instr, 25, 31)  # funct7-like high immediate
        sel = RV_ISA.get_int(instr, 20, 24)  # unary op selector / shamt
        imm12 = RV_ISA.get_int(instr, 20, 31)
        a = register_file[rs1].read_uint()

        name = None
        result = 0
        if funct3 == 0x1 and upper == 0x30:
            # Count / sign-extend unary ops (rs2 field selects the operation).
            if sel == 0x0:
                name, result = "clz", _clz(a)
            elif sel == 0x1:
                name, result = "ctz", _ctz(a)
            elif sel == 0x2:
                name, result = "cpop", bin(a).count("1")
            elif sel == 0x4:
                name, result = "sext.b", _sign_extend(a, 8)
            elif sel == 0x5:
                name, result = "sext.h", _sign_extend(a, 16)
        elif funct3 == 0x5:
            if upper == 0x30:
                name, result = "rori", _rotate_right(a, sel)
            elif imm12 == 0x698:
                name, result = "rev8", _rev8(a)
            elif imm12 == 0x287:
                name, result = "orc.b", _orc_b(a)

        if name is None:
            return False
        if snoop:
            RV_ISA.print_snoop(snoop, f"{name} {cls.get_reg_name(rd)}", None)
        register_file[rd].write(conv_to_bytes(result & _MASK32))
        return True
