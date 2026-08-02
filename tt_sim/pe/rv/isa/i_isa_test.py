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
