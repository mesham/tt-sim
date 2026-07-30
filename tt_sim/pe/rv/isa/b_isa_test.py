"""Unit tests for the Zba/Zbb bit-manipulation ISAs used by Blackhole baby cores.

Each case encodes one instruction, runs it through the ISA class with crafted
register inputs, and checks the destination against a reference value. Encodings
follow the RISC-V Zba/Zbb spec (the same the Blackhole ISA doc references).
"""

from tt_sim.pe.rv.isa.b_isa import RV_ZBA_ISA, RV_ZBB_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

M = 0xFFFFFFFF


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
        self.r["pc"] = _Reg(0)

    def __getitem__(self, k):
        return self.r[k]


class _Mem:
    def __init__(self, instr):
        self.instr = instr

    def read(self, addr, size):
        return conv_to_bytes(self.instr)


def _run(isa, instr, rs1v=0, rs2v=0):
    rf = _RF()
    rf.r[15].v = rs1v & M
    rf.r[20].v = rs2v & M
    assert isa.run(rf, _Mem(instr), False) is True
    return rf.r[1].v


def _r(f7, rs2, f3, rd=1, rs1=15):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x33


def _i(imm12, f3, rd=1, rs1=15):
    return (imm12 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | 0x13


def test_zba_shadd():
    assert _run(RV_ZBA_ISA, _r(0x10, 20, 2), 5, 0x100) == (5 << 1) + 0x100
    assert _run(RV_ZBA_ISA, _r(0x10, 20, 4), 5, 0x100) == (5 << 2) + 0x100
    assert _run(RV_ZBA_ISA, _r(0x10, 20, 6), 5, 0x100) == (5 << 3) + 0x100


def test_zba_rejects_non_shadd():
    rf = _RF()
    # funct7 0x10 with funct3 0 is not a shNadd — must not be claimed.
    assert RV_ZBA_ISA.run(rf, _Mem(_r(0x10, 20, 0)), False) is False


def test_zbb_logic_with_negate():
    assert _run(RV_ZBB_ISA, _r(0x20, 20, 7), 0xFF, 0x0F) == 0xF0  # andn
    assert _run(RV_ZBB_ISA, _r(0x20, 20, 6), 0xF0, 0x0F) == (0xF0 | ~0x0F & M)  # orn
    assert _run(RV_ZBB_ISA, _r(0x20, 20, 4), 0xAA, 0x55) == (~(0xAA ^ 0x55) & M)  # xnor


def test_zbb_minmax():
    assert (
        _run(RV_ZBB_ISA, _r(0x05, 20, 4), 0xFFFFFFFF, 5) == 0xFFFFFFFF
    )  # min (signed)
    assert _run(RV_ZBB_ISA, _r(0x05, 20, 5), 0xFFFFFFFF, 5) == 5  # minu
    assert _run(RV_ZBB_ISA, _r(0x05, 20, 6), 0xFFFFFFFF, 5) == 5  # max (signed)
    assert _run(RV_ZBB_ISA, _r(0x05, 20, 7), 0xFFFFFFFF, 5) == 0xFFFFFFFF  # maxu


def test_zbb_rotate():
    assert _run(RV_ZBB_ISA, _r(0x30, 20, 1), 0x1, 1) == 0x2  # rol by 1
    assert _run(RV_ZBB_ISA, _r(0x30, 20, 5), 0x1, 1) == 0x80000000  # ror by 1


def test_zbb_zext_h():
    assert _run(RV_ZBB_ISA, _r(0x04, 0, 4), 0x1234ABCD) == 0xABCD


def test_zbb_count_and_extend():
    assert _run(RV_ZBB_ISA, _i(0x600, 1), 0x0000FFFF) == 16  # clz
    assert _run(RV_ZBB_ISA, _i(0x601, 1), 0x40) == 6  # ctz
    assert _run(RV_ZBB_ISA, _i(0x602, 1), 0xFF) == 8  # cpop
    assert _run(RV_ZBB_ISA, _i(0x604, 1), 0x80) == 0xFFFFFF80  # sext.b
    assert _run(RV_ZBB_ISA, _i(0x605, 1), 0x8000) == 0xFFFF8000  # sext.h


def test_zbb_rori_rev8_orcb():
    assert _run(RV_ZBB_ISA, _i((0x30 << 5) | 1, 5), 0x1) == 0x80000000  # rori 1
    assert _run(RV_ZBB_ISA, _i(0x698, 5), 0x11223344) == 0x44332211  # rev8
    assert _run(RV_ZBB_ISA, _i(0x287, 5), 0x01000200) == 0xFF00FF00  # orc.b


def test_non_extension_returns_false():
    rf = _RF()
    add = (0x00 << 25) | (20 << 20) | (15 << 15) | (0 << 12) | (1 << 7) | 0x33
    assert RV_ZBA_ISA.run(rf, _Mem(add), False) is False
    assert RV_ZBB_ISA.run(rf, _Mem(add), False) is False
