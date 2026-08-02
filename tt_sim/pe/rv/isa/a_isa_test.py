"""Unit tests for the Zaamo atomics and the F/V guard ISAs on Blackhole."""

import pytest

from tt_sim.pe.rv.isa.a_isa import RV_ZAAMO_ISA
from tt_sim.pe.rv.isa.guard_isa import RV_F_GUARD_ISA, RV_V_GUARD_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

M = 0xFFFFFFFF


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
        self.r["pc"] = _Reg(0)

    def __getitem__(self, k):
        return self.r[k]


class _Mem:
    """Backs the instruction at PC 0 and a tiny data word for AMO tests."""

    def __init__(self, instr, data_addr=0x100, data=0):
        self.instr = instr
        self.words = {data_addr: data & M}

    def read(self, addr, size):
        if addr == 0:
            return conv_to_bytes(self.instr)
        return conv_to_bytes(self.words.get(addr, 0))

    def write(self, addr, value, size=None):
        self.words[addr] = conv_to_uint32(value)


def _amo(funct5, rd=1, rs1=15, rs2=20):
    return (funct5 << 27) | (rs2 << 20) | (rs1 << 15) | (0x2 << 12) | (rd << 7) | 0x2F


def _run_amo(funct5, mem_old, src):
    rf = _RF()
    rf.r[15].v = 0x100  # rs1 = address
    rf.r[20].v = src & M  # rs2 = source operand
    mem = _Mem(_amo(funct5), 0x100, mem_old)
    assert RV_ZAAMO_ISA.run(rf, mem, False) is True
    return rf.r[1].v, mem.words[0x100]  # (rd = old value, new memory value)


def test_amoadd():
    assert _run_amo(0x00, 10, 5) == (10, 15)


def test_amoswap():
    assert _run_amo(0x01, 10, 5) == (10, 5)


def test_amo_logic():
    assert _run_amo(0x04, 0xF0, 0x0F) == (0xF0, 0xFF)  # amoxor
    assert _run_amo(0x08, 0xF0, 0x0F) == (0xF0, 0xFF)  # amoor
    assert _run_amo(0x0C, 0xF0, 0x0F) == (0xF0, 0x00)  # amoand


def test_amo_minmax():
    assert _run_amo(0x10, 0xFFFFFFFF, 5) == (0xFFFFFFFF, 0xFFFFFFFF)  # amomin signed
    assert _run_amo(0x18, 0xFFFFFFFF, 5) == (0xFFFFFFFF, 5)  # amominu
    assert _run_amo(0x14, 0xFFFFFFFF, 5) == (0xFFFFFFFF, 5)  # amomax signed
    assert _run_amo(0x1C, 0xFFFFFFFF, 5) == (0xFFFFFFFF, 0xFFFFFFFF)  # amomaxu


def test_lr_sc_not_claimed():
    # lr.w (funct5 0x02) / sc.w (0x03) are Zalrsc — unsupported on hardware.
    rf = _RF()
    assert RV_ZAAMO_ISA.run(rf, _Mem(_amo(0x02)), False) is False
    assert RV_ZAAMO_ISA.run(rf, _Mem(_amo(0x03)), False) is False


def test_f_guard_raises_on_fp_opcode():
    rf = _RF()
    fadd_s = (0x00 << 25) | (0x53)  # OP-FP opcode
    with pytest.raises(NotImplementedError):
        RV_F_GUARD_ISA.run(rf, _Mem(fadd_s), False)


def test_v_guard_raises_on_vector_opcode():
    rf = _RF()
    with pytest.raises(NotImplementedError):
        RV_V_GUARD_ISA.run(rf, _Mem(0x57), False)


def test_guards_ignore_other_opcodes():
    rf = _RF()
    add = (15 << 15) | (1 << 7) | 0x33
    assert RV_F_GUARD_ISA.run(rf, _Mem(add), False) is False
    assert RV_V_GUARD_ISA.run(rf, _Mem(add), False) is False
