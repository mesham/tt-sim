"""Unit tests for the Zfh half-precision floating-point ISA.

Reference half values come from Python's ``struct`` ``'e'`` (IEEE binary16), the
same format the ISA uses, so these check encoding/decoding and control flow
rather than re-deriving IEEE arithmetic.
"""

import struct

import pytest

from tt_sim.pe.rv.isa.zfh_isa import RV_ZFH_ISA
from tt_sim.pe.rv.rv32 import FP_REGISTER_BASE
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

M = 0xFFFFFFFF


def h(x):
    """Python float -> NaN-boxed half register value."""
    bits = struct.unpack("<H", struct.pack("<e", x))[0]
    return 0xFFFF0000 | bits


def half_of(regval):
    """NaN-boxed half register value -> Python float."""
    return struct.unpack("<e", struct.pack("<H", regval & 0xFFFF))[0]


class _Reg:
    def __init__(self, v=0):
        self.v = v & M

    def read(self):
        return conv_to_bytes(self.v)

    def write(self, b):
        self.v = conv_to_uint32(b)


class _RF:
    def __init__(self):
        self.r = {i: _Reg() for i in range(FP_REGISTER_BASE + 33)}
        self.r["pc"] = _Reg(0)

    def __getitem__(self, k):
        return self.r[k]

    def setf(self, idx, regval):
        self.r[FP_REGISTER_BASE + idx].v = regval & M

    def getf(self, idx):
        return self.r[FP_REGISTER_BASE + idx].v

    def setx(self, idx, v):
        self.r[idx].v = v & M

    def getx(self, idx):
        return self.r[idx].v


class _Mem:
    def __init__(self, instr):
        self.instr = instr
        self.data = {}

    def read(self, addr, size):
        if addr == 0:
            return conv_to_bytes(self.instr)
        return conv_to_bytes(self.data.get(addr, 0), size)

    def write(self, addr, value, size=None):
        self.data[addr] = conv_to_uint32(value)


def op_fp(funct5, funct3=0, rd=1, rs1=2, rs2=0):
    return (
        (funct5 << 27)
        | (2 << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (rd << 7)
        | 0x53
    )


def run(rf, instr):
    return RV_ZFH_ISA.run(rf, _Mem(instr), False)


def _run_binary(funct5, a, b):
    rf = _RF()
    rf.setf(2, h(a))
    rf.setf(3, h(b))
    assert run(rf, op_fp(funct5, rd=1, rs1=2, rs2=3)) is True
    return half_of(rf.getf(1))


def test_arith():
    assert _run_binary(0x00, 1.5, 2.25) == 3.75  # fadd.h
    assert _run_binary(0x01, 5.0, 1.5) == 3.5  # fsub.h
    assert _run_binary(0x02, 3.0, 0.5) == 1.5  # fmul.h


def test_fdiv_fsqrt_unsupported():
    rf = _RF()
    rf.setf(2, h(1.0))
    rf.setf(3, h(1.0))
    with pytest.raises(NotImplementedError):
        run(rf, op_fp(0x03, rd=1, rs1=2, rs2=3))  # fdiv.h
    with pytest.raises(NotImplementedError):
        run(rf, op_fp(0x0B, rd=1, rs1=2))  # fsqrt.h


def test_fma():
    # fmadd.h: rs1*rs2 + rs3
    rf = _RF()
    rf.setf(2, h(2.0))
    rf.setf(3, h(3.0))
    rf.setf(4, h(1.0))
    instr = (4 << 27) | (2 << 25) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | 0x43
    assert run(rf, instr) is True
    assert half_of(rf.getf(1)) == 7.0
    # fnmsub.h (0x4B): -(rs1*rs2) + rs3
    rf2 = _RF()
    rf2.setf(2, h(2.0))
    rf2.setf(3, h(3.0))
    rf2.setf(4, h(1.0))
    instr2 = (4 << 27) | (2 << 25) | (3 << 20) | (2 << 15) | (0 << 12) | (1 << 7) | 0x4B
    assert run(rf2, instr2) is True
    assert half_of(rf2.getf(1)) == -5.0


def test_minmax():
    assert _run_binary(0x05, 2.0, 7.0) == 2.0  # fmin.h (funct3 0 default)
    rf = _RF()
    rf.setf(2, h(2.0))
    rf.setf(3, h(7.0))
    assert run(rf, op_fp(0x05, funct3=1, rd=1, rs1=2, rs2=3)) is True  # fmax.h
    assert half_of(rf.getf(1)) == 7.0


def test_min_with_nan():
    rf = _RF()
    rf.setf(2, 0xFFFF0000 | 0x7E00)  # canonical NaN
    rf.setf(3, h(4.0))
    assert run(rf, op_fp(0x05, funct3=0, rd=1, rs1=2, rs2=3)) is True
    assert half_of(rf.getf(1)) == 4.0  # NaN ignored


def test_fsgnj():
    rf = _RF()
    rf.setf(2, h(3.0))
    rf.setf(3, h(-1.0))  # sign source
    assert run(rf, op_fp(0x04, funct3=0, rd=1, rs1=2, rs2=3)) is True  # fsgnj
    assert half_of(rf.getf(1)) == -3.0


def test_compare():
    rf = _RF()
    rf.setf(2, h(1.0))
    rf.setf(3, h(2.0))
    assert run(rf, op_fp(0x14, funct3=1, rd=5, rs1=2, rs2=3)) is True  # flt.h
    assert rf.getx(5) == 1
    assert run(rf, op_fp(0x14, funct3=2, rd=5, rs1=2, rs2=3)) is True  # feq.h
    assert rf.getx(5) == 0


def test_fclass():
    rf = _RF()
    rf.setf(2, h(-0.0))
    assert run(rf, op_fp(0x1C, funct3=1, rd=5, rs1=2)) is True  # fclass.h
    assert rf.getx(5) == (1 << 3)  # negative zero


def test_fmv():
    rf = _RF()
    rf.setf(2, h(-1.0))
    assert run(rf, op_fp(0x1C, funct3=0, rd=5, rs1=2)) is True  # fmv.x.h
    bits = struct.unpack("<H", struct.pack("<e", -1.0))[0]
    assert rf.getx(5) == (bits - 0x10000) & M  # sign-extended
    rf.setx(6, 0x1234)
    assert run(rf, op_fp(0x1E, funct3=0, rd=1, rs1=6)) is True  # fmv.h.x
    assert rf.getf(1) == (0xFFFF0000 | 0x1234)


def test_fcvt_int():
    rf = _RF()
    rf.setf(2, h(6.0))
    assert run(rf, op_fp(0x18, rd=5, rs1=2, rs2=0)) is True  # fcvt.w.h
    assert rf.getx(5) == 6
    rf.setx(6, 10)
    assert run(rf, op_fp(0x1A, rd=1, rs1=6, rs2=0)) is True  # fcvt.h.w
    assert half_of(rf.getf(1)) == 10.0


def test_fcvt_single():
    rf = _RF()
    rf.setf(2, h(2.5))
    assert run(rf, op_fp(0x08, rd=1, rs1=2, rs2=2)) is True  # fcvt.s.h
    assert struct.unpack("<f", rf.r[FP_REGISTER_BASE + 1].read())[0] == 2.5
    # back: fcvt.h.s
    rf.r[FP_REGISTER_BASE + 3].v = conv_to_uint32(struct.pack("<f", 3.5))
    assert run(rf, op_fp(0x08, rd=1, rs1=3, rs2=0)) is True
    assert half_of(rf.getf(1)) == 3.5


def test_load_store():
    rf = _RF()
    rf.setx(2, 0x100)  # base address
    bits = struct.unpack("<H", struct.pack("<e", 1.25))[0]
    mem = _Mem((0 << 20) | (2 << 15) | (1 << 12) | (1 << 7) | 0x07)  # flh f1, 0(x2)
    mem.data[0x100] = bits
    assert RV_ZFH_ISA.run(rf, mem, False) is True
    assert half_of(rf.getf(1)) == 1.25
    # fsh f1, 0(x2)
    smem = _Mem((0 << 25) | (1 << 20) | (2 << 15) | (1 << 12) | (0 << 7) | 0x27)
    assert RV_ZFH_ISA.run(rf, smem, False) is True
    assert (smem.data[0x100] & 0xFFFF) == bits


def test_denormal_flush():
    # Smallest normal half is 2**-14; a value an order smaller is denormal and
    # must flush to zero on output.
    rf = _RF()
    rf.setf(2, h(2.0**-14))
    rf.setf(3, h(2.0**-6))  # product ~2**-20 -> denormal -> flushed to 0
    assert run(rf, op_fp(0x02, rd=1, rs1=2, rs2=3)) is True
    assert half_of(rf.getf(1)) == 0.0


def test_nan_boxed_read():
    # A register that is not NaN-boxed reads back as canonical NaN.
    rf = _RF()
    rf.setf(2, 0x00003C00)  # low bits = 1.0 but upper 16 bits not all ones
    rf.setf(3, h(1.0))
    assert run(rf, op_fp(0x00, rd=1, rs1=2, rs2=3)) is True  # fadd.h
    # NaN + 1.0 = NaN
    assert half_of(rf.getf(1)) != half_of(rf.getf(1))


def test_ignores_single_precision():
    # fmt=00 (.s) OP-FP must not be claimed by the half ISA.
    rf = _RF()
    single_fadd = (0 << 27) | (0 << 25) | (3 << 20) | (2 << 15) | (1 << 7) | 0x53
    assert RV_ZFH_ISA.run(rf, _Mem(single_fadd), False) is False
