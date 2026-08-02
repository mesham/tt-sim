"""Zfh half-precision (IEEE-754 binary16) floating-point extension.

Per ``BlackholeA0/TensixTile/BabyRISCV/InstructionSet.md`` the Blackhole baby
cores implement Zfh with the same caveats as their partial F extension:

- rounding mode bits are ignored — rounding is always RNE;
- denormal inputs are treated as zero and denormal results are flushed to zero;
- ``fdiv.h`` and ``fsqrt.h`` are **not** implemented;
- ``fmadd.h`` & friends "execute" but are not strictly IEEE fused;
- ``fcsr`` exists (``fflags`` / ``frm`` do not) — exception flags are not tracked.

This covers the ``.h`` opcode space (OP-FP / fp load-store / fused multiply-add
with the half format field); single-precision ``.s`` instructions fall through
to the F guard until F itself is modelled. Half values live NaN-boxed in the
32-bit FP registers (upper 16 bits all ones); a non-boxed register reads back as
a canonical NaN, per the RISC-V NaN-boxing rule.

A custom CSR bit (address unspecified in the docs) switches ``.h`` ops to operate
on BF16 rather than FP16; only FP16 is modelled here.
"""

import struct

from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.pe.rv.rv32 import FP_REGISTER_BASE
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

_MASK32 = 0xFFFFFFFF
_CANONICAL_QNAN_H = 0x7E00  # canonical half quiet NaN
_NAN_BOX_HI = 0xFFFF0000


def _is_denormal_h(bits):
    return (bits >> 10) & 0x1F == 0 and (bits & 0x3FF) != 0


def _half_bits_to_float(bits):
    """Half bit pattern -> Python float, flushing denormal *inputs* to zero."""
    if _is_denormal_h(bits):
        return -0.0 if bits & 0x8000 else 0.0
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def _float_to_half_bits(value):
    """Python float -> half bit pattern (RNE), flushing denormal *outputs* to
    zero and saturating overflow to signed infinity."""
    try:
        bits = struct.unpack("<H", struct.pack("<e", value))[0]
    except (OverflowError, struct.error):
        # Too large for half — RISC-V rounds to +/-inf under RNE.
        sign = 0x8000 if (value < 0) else 0x0000
        return sign | 0x7C00
    if _is_denormal_h(bits):
        return bits & 0x8000  # signed zero
    return bits


class RV_ZFH_ISA(RV_ISA):
    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        opcode = instr & 0x7F
        fmt = (instr >> 25) & 0x3  # OP-FP / FMA format field: 2 = half
        width = (instr >> 12) & 0x7  # load/store width field: 1 = half

        if opcode == 0x53:
            return cls._op_fp(instr, register_file, snoop) if fmt == 2 else False
        if opcode in (0x43, 0x47, 0x4B, 0x4F):
            return cls._fma(instr, register_file, snoop) if fmt == 2 else False
        if opcode == 0x07:
            return cls._flh(instr, register_file, memory_space) if width == 1 else False
        if opcode == 0x27:
            return cls._fsh(instr, register_file, memory_space) if width == 1 else False
        return False

    # -- register helpers ---------------------------------------------------
    @staticmethod
    def _rd_h(rf, idx):
        raw = rf[FP_REGISTER_BASE + idx].read_uint()
        # NaN-boxing: a register that does not hold a boxed half reads as NaN.
        return raw & 0xFFFF if (raw >> 16) == 0xFFFF else _CANONICAL_QNAN_H

    @staticmethod
    def _wr_h(rf, idx, bits16):
        rf[FP_REGISTER_BASE + idx].write(conv_to_bytes(_NAN_BOX_HI | (bits16 & 0xFFFF)))

    @staticmethod
    def _wr_x(rf, idx, value):
        if idx != 0:
            rf[idx].write(conv_to_bytes(value & _MASK32))

    # -- OP-FP (fmt = half) -------------------------------------------------
    @classmethod
    def _op_fp(cls, instr, rf, snoop):
        funct5 = (instr >> 27) & 0x1F
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rd = (instr >> 7) & 0x1F

        if funct5 in (0x00, 0x01, 0x02):  # fadd / fsub / fmul
            a = _half_bits_to_float(cls._rd_h(rf, rs1))
            b = _half_bits_to_float(cls._rd_h(rf, rs2))
            res = {0x00: a + b, 0x01: a - b, 0x02: a * b}[funct5]
            cls._wr_h(rf, rd, _float_to_half_bits(res))
            name = {0x00: "fadd", 0x01: "fsub", 0x02: "fmul"}[funct5]
            if snoop:
                RV_ISA.print_snoop(snoop, f"{name}.h f{rd}, f{rs1}, f{rs2}", None)
            return True
        if funct5 in (0x03, 0x0B):  # fdiv / fsqrt — unsupported on hardware
            raise NotImplementedError(
                f"{'fdiv.h' if funct5 == 3 else 'fsqrt.h'} is not implemented on "
                "the Blackhole baby cores (per the ISA doc)."
            )
        if funct5 == 0x04:  # fsgnj / fsgnjn / fsgnjx
            a = cls._rd_h(rf, rs1)
            b = cls._rd_h(rf, rs2)
            sign = {0: b & 0x8000, 1: (~b) & 0x8000, 2: (a ^ b) & 0x8000}[funct3]
            cls._wr_h(rf, rd, (a & 0x7FFF) | sign)
            return True
        if funct5 == 0x05:  # fmin / fmax
            cls._wr_h(
                rf, rd, cls._minmax(cls._rd_h(rf, rs1), cls._rd_h(rf, rs2), funct3)
            )
            return True
        if funct5 == 0x14:  # comparisons -> integer register
            a = _half_bits_to_float(cls._rd_h(rf, rs1))
            b = _half_bits_to_float(cls._rd_h(rf, rs2))
            res = {2: a == b, 1: a < b, 0: a <= b}[funct3]
            cls._wr_x(rf, rd, 1 if res else 0)
            return True
        if funct5 == 0x1C and funct3 == 0:  # fmv.x.h (bits, sign-extended)
            bits = cls._rd_h(rf, rs1)
            cls._wr_x(rf, rd, bits - 0x10000 if bits & 0x8000 else bits)
            return True
        if funct5 == 0x1C and funct3 == 1:  # fclass.h
            cls._wr_x(rf, rd, cls._fclass(cls._rd_h(rf, rs1)))
            return True
        if funct5 == 0x1E:  # fmv.h.x (int bits -> boxed half)
            cls._wr_h(rf, rd, rf[rs1].read_uint() & 0xFFFF)
            return True
        if funct5 == 0x18:  # fcvt.w.h / fcvt.wu.h (half -> int)
            cls._wr_x(rf, rd, cls._to_int(_half_bits_to_float(cls._rd_h(rf, rs1)), rs2))
            return True
        if funct5 == 0x1A:  # fcvt.h.w / fcvt.h.wu (int -> half)
            iv = rf[rs1].read_uint()
            fv = float(iv - (1 << 32) if (rs2 == 0 and iv & 0x80000000) else iv)
            cls._wr_h(rf, rd, _float_to_half_bits(fv))
            return True
        if funct5 == 0x08 and rs2 == 0:  # fcvt.h.s (single -> half)
            single = struct.unpack("<f", rf[FP_REGISTER_BASE + rs1].read())[0]
            cls._wr_h(rf, rd, _float_to_half_bits(single))
            return True
        if funct5 == 0x08 and rs2 == 2:  # fcvt.s.h (half -> single)
            val = _half_bits_to_float(cls._rd_h(rf, rs1))
            rf[FP_REGISTER_BASE + rd].write(struct.pack("<f", val))
            return True
        return False

    @classmethod
    def _fma(cls, instr, rf, snoop):
        opcode = instr & 0x7F
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        rs3 = (instr >> 27) & 0x1F
        rd = (instr >> 7) & 0x1F
        a = _half_bits_to_float(cls._rd_h(rf, rs1))
        b = _half_bits_to_float(cls._rd_h(rf, rs2))
        c = _half_bits_to_float(cls._rd_h(rf, rs3))
        prod = a * b
        res = {
            0x43: prod + c,  # fmadd
            0x47: prod - c,  # fmsub
            0x4B: -prod + c,  # fnmsub
            0x4F: -prod - c,  # fnmadd
        }[opcode]
        cls._wr_h(rf, rd, _float_to_half_bits(res))
        return True

    # -- helpers ------------------------------------------------------------
    @classmethod
    def _minmax(cls, a_bits, b_bits, funct3):
        a = _half_bits_to_float(a_bits)
        b = _half_bits_to_float(b_bits)
        a_nan, b_nan = a != a, b != b
        if a_nan and b_nan:
            return _CANONICAL_QNAN_H
        if a_nan:
            return b_bits
        if b_nan:
            return a_bits
        is_min = funct3 == 0
        if a == b:
            # Only +0 vs -0 differ here: fmin picks -0, fmax picks +0.
            a_neg = bool(a_bits & 0x8000)
            return a_bits if a_neg == is_min else b_bits
        return a_bits if ((a < b) == is_min) else b_bits

    @staticmethod
    def _to_int(value, unsigned_sel):
        unsigned = unsigned_sel == 1
        hi = 0xFFFFFFFF if unsigned else 0x7FFFFFFF
        lo = 0 if unsigned else -(1 << 31)
        if value != value:  # NaN -> saturate high (RISC-V)
            return hi
        if value in (float("inf"), float("-inf")):
            return hi if value > 0 else (lo & _MASK32)
        rounded = round(value)  # rounding mode is always RNE
        return max(lo, min(hi if unsigned else (1 << 31) - 1, rounded)) & _MASK32

    @staticmethod
    def _fclass(bits):
        sign = bits & 0x8000
        exp = (bits >> 10) & 0x1F
        mant = bits & 0x3FF
        if exp == 0x1F:
            if mant == 0:
                return 1 << (0 if sign else 7)  # -inf / +inf
            return 1 << (8 if (mant & 0x200) == 0 else 9)  # sNaN / qNaN
        if exp == 0 and mant == 0:
            return 1 << (3 if sign else 4)  # -0 / +0
        if exp == 0:
            return 1 << (2 if sign else 5)  # -/+ subnormal
        return 1 << (1 if sign else 6)  # -/+ normal

    # -- load / store -------------------------------------------------------
    @classmethod
    def _flh(cls, instr, rf, mem):
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F
        imm = instr >> 20
        imm -= 1 << 12 if imm & 0x800 else 0
        base = rf[rs1].read_uint()
        bits = conv_to_uint32(mem.read((base + imm) & _MASK32, 2)) & 0xFFFF
        cls._wr_h(rf, rd, bits)
        return True

    @classmethod
    def _fsh(cls, instr, rf, mem):
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
        imm -= 1 << 12 if imm & 0x800 else 0
        base = rf[rs1].read_uint()
        mem.write((base + imm) & _MASK32, conv_to_bytes(cls._rd_h(rf, rs2) & 0xFFFF, 2))
        return True
