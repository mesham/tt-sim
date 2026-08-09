"""Unit tests for the RV32I cases where signedness decides the answer.

The signed/unsigned branch pairs and the arithmetic right shifts only differ
from their siblings when an operand has its top bit set, so a mix-up survives
any workload that compares small positive numbers. GCC's local-.data copy loop
(emitted for any kernel with initialised locals) is one that does not: it
walks a negative word count with `blt` and derives it with `srai`.
"""

from tt_sim.pe.rv.isa.i_isa import RV_I_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

M = 0xFFFFFFFF
PC = 0x1000
RS1, RS2, RD = 15, 20, 1


class _Reg:
    def __init__(self, v=0):
        self.v = v & M

    def read(self):
        return conv_to_bytes(self.v)

    def read_uint(self):
        return int.from_bytes(self.read(), "little")

    def read_int(self):
        return int.from_bytes(self.read(), "little", signed=True)

    def write(self, b):
        self.v = conv_to_uint32(b)


class _RF:
    def __init__(self):
        self.r = {i: _Reg() for i in range(34)}
        self.r["pc"] = _Reg(PC)
        self.r["nextpc"] = _Reg(PC + 4)

    def __getitem__(self, k):
        return self.r[k]


class _Mem:
    def __init__(self, instr):
        self.instr = instr

    def read(self, addr, size):
        return conv_to_bytes(self.instr)


def _b(imm, funct3):
    """Encode a B-type branch."""
    return (
        (((imm >> 12) & 1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | (RS2 << 20)
        | (RS1 << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 1) << 7)
        | 0x63
    )


def _taken(funct3, rs1v, rs2v, offset=8):
    rf = _RF()
    rf.r[RS1].v = rs1v & M
    rf.r[RS2].v = rs2v & M
    assert RV_I_ISA.run(rf, _Mem(_b(offset, funct3)), False) is True
    return rf.r["nextpc"].v == PC + offset


def _shift_imm(shamt, funct7):
    return (funct7 << 25) | (shamt << 20) | (RS1 << 15) | (5 << 12) | (RD << 7) | 0x13


def _shift_reg(funct7):
    return (funct7 << 25) | (RS2 << 20) | (RS1 << 15) | (5 << 12) | (RD << 7) | 0x33


def _shift(instr, rs1v, rs2v=0):
    rf = _RF()
    rf.r[RS1].v = rs1v & M
    rf.r[RS2].v = rs2v & M
    assert RV_I_ISA.run(rf, _Mem(instr), False) is True
    return rf.r[RD].v


class _StoreMem:
    """Instruction fetch plus a byte-addressed store log."""

    def __init__(self, instr):
        self.instr = instr
        self.written = None

    def read(self, addr, size):
        return conv_to_bytes(self.instr)

    def write(self, addr, data):
        self.written = (addr, bytes(data))


def _s(funct3, offset):
    """Encode an S-type store."""
    return (
        (((offset >> 5) & 0x7F) << 25)
        | (RS2 << 20)
        | (RS1 << 15)
        | (funct3 << 12)
        | ((offset & 0x1F) << 7)
        | 0x23
    )


def _store(funct3, value, base=0x1000, offset=4):
    rf = _RF()
    rf.r[RS1].v = base
    rf.r[RS2].v = value & M
    mem = _StoreMem(_s(funct3, offset))
    assert RV_I_ISA.run(rf, mem, False) is True
    return mem.written


def test_stores_write_their_full_width():
    """`sh` must write two bytes, not one.

    A single-byte `sh` is invisible to anything that only stores words, but it
    silently drops the upper half of every 16-bit store: a data-movement kernel
    filling an L1 tile with bfloat16 constants (``ptr[i] = 0x3F80``) ends up
    with 0x0080 in every element.
    """
    assert _store(0x0, 0x3F80) == (0x1004, b"\x80")
    assert _store(0x1, 0x3F80) == (0x1004, b"\x80\x3f")
    assert _store(0x2, 0xDEADBEEF) == (0x1004, b"\xef\xbe\xad\xde")


def test_blt_is_signed():
    assert _taken(0x4, -3, 2)
    assert not _taken(0x4, 2, -3)
    assert _taken(0x4, 1, 2)


def test_bltu_is_unsigned():
    assert _taken(0x6, 2, 0xFFFFFFFD)
    assert not _taken(0x6, 0xFFFFFFFD, 2)


def test_bge_is_signed():
    assert _taken(0x5, 2, -3)
    assert not _taken(0x5, -3, 2)
    assert _taken(0x5, 2, 2)


def test_bgeu_is_unsigned():
    assert _taken(0x7, 0xFFFFFFFD, 2)
    assert not _taken(0x7, 2, 0xFFFFFFFD)


def test_srai_shifts_arithmetically():
    assert _shift(_shift_imm(2, 0x20), 0x20) == 8
    assert _shift(_shift_imm(1, 0x20), 0xFFFFFFFD) == 0xFFFFFFFE
    assert _shift(_shift_imm(8, 0x20), 0xFFFFFF00) == 0xFFFFFFFF
    assert _shift(_shift_imm(31, 0x20), 0x80000000) == 0xFFFFFFFF
    assert _shift(_shift_imm(31, 0x20), 0x7FFFFFFF) == 0
    assert _shift(_shift_imm(0, 0x20), 0xFFFFFFFD) == 0xFFFFFFFD


def test_srli_shifts_logically():
    assert _shift(_shift_imm(1, 0x00), 0xFFFFFFFD) == 0x7FFFFFFE
    assert _shift(_shift_imm(2, 0x00), 0x20) == 8


def test_sra_and_srl_register_forms():
    assert _shift(_shift_reg(0x20), 0xFFFFFF00, 8) == 0xFFFFFFFF
    assert _shift(_shift_reg(0x20), 0x20, 2) == 8
    assert _shift(_shift_reg(0x00), 0xFFFFFF00, 8) == 0x00FFFFFF


def _i_arith(funct3, imm):
    """Encode an I-type ALU op (`opcode 0x13`) with a 12-bit signed immediate."""
    return ((imm & 0xFFF) << 20) | (RS1 << 15) | (funct3 << 12) | (RD << 7) | 0x13


def _alu_imm(funct3, rs1v, imm):
    rf = _RF()
    rf.r[RS1].v = rs1v & M
    assert RV_I_ISA.run(rf, _Mem(_i_arith(funct3, imm)), False) is True
    return rf.r[RD].v


def test_bitwise_immediates_are_32_bit_not_arbitrary_precision():
    """`xori rd, rs, -1` is the RV32I `not`, and it must not raise.

    The immediate is sign-extended, so in Python a negative one is an
    arbitrary-precision negative integer. Applying a bitwise operator to it
    keeps it negative -- `x ^ -1` is `-(x + 1)` -- and the result then reaches
    `conv_to_bytes(..., signed=False)`, which raises `OverflowError: can't
    convert negative int to unsigned`. That is a CRASH rather than a wrong
    answer, and every `~x` in a C kernel compiles to exactly this instruction:
    `dramratebench`'s reader kernel hit it on its first run against tt-sim.
    `ori` with any negative immediate is the same bug.
    """
    assert _alu_imm(0x4, 0x44524231, -1) == 0xBBADBDCE  # xori: the NOT idiom
    assert _alu_imm(0x4, 0x00000000, -1) == 0xFFFFFFFF
    assert _alu_imm(0x6, 0x0000000F, -16) == 0xFFFFFFFF  # ori with a negative
    assert _alu_imm(0x7, 0xFFFFFFFF, -16) == 0xFFFFFFF0  # andi, unchanged
    # The non-negative immediates every existing workload uses are untouched.
    assert _alu_imm(0x4, 0xF0F0F0F0, 0x0FF) == 0xF0F0F00F
    assert _alu_imm(0x6, 0xF0F0F0F0, 0x00F) == 0xF0F0F0FF
    assert _alu_imm(0x7, 0xF0F0F0F0, 0x0FF) == 0x000000F0
