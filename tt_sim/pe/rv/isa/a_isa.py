"""Zaamo atomic-memory-operation extension (``amo*.w``).

Per the Blackhole baby-RISC-V ISA doc the cores implement the Zaamo subset of
the A extension: ``amoadd.w``, ``amoswap.w``, ``amoxor.w``, ``amoor.w``,
``amoand.w``, ``amomin.w``, ``amomax.w``, ``amominu.w``, ``amomaxu.w`` — "limited
to local L1; cannot target MMIO or remote tile address spaces". The Zalrsc
subset (``lr.w`` / ``sc.w``) is **not** implemented on hardware, so it is not
decoded here (it falls through to UndefinedBehavior like any other unknown op).

Each ``amo`` atomically loads the word at ``rs1``, writes ``op(old, rs2)`` back,
and returns the old value in ``rd``. The simulator is single-stepped, so the
read-modify-write is naturally atomic.

Wormhole baby cores have no A extension at all, so this ISA is attached to
Blackhole baby cores only (see ``BabyRISCV``).
"""

from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

_MASK32 = 0xFFFFFFFF


def _s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


class RV_ZAAMO_ISA(RV_ISA):
    # funct5 (instr bits 27..31) -> (name, binary op on (old_unsigned, src)).
    _OPS = {
        0x00: ("amoadd", lambda o, s: (o + s) & _MASK32),
        0x01: ("amoswap", lambda o, s: s),
        0x04: ("amoxor", lambda o, s: o ^ s),
        0x08: ("amoor", lambda o, s: o | s),
        0x0C: ("amoand", lambda o, s: o & s),
        0x10: ("amomin", lambda o, s: o if _s32(o) < _s32(s) else s),
        0x14: ("amomax", lambda o, s: o if _s32(o) > _s32(s) else s),
        0x18: ("amominu", lambda o, s: o if o < s else s),
        0x1C: ("amomaxu", lambda o, s: o if o > s else s),
    }

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        if instr & 0x7F != 0x2F:
            return False
        if RV_ISA.get_int(instr, 12, 14) != 0x2:  # funct3 010 = word width
            return False
        funct5 = RV_ISA.get_int(instr, 27, 31)
        op = cls._OPS.get(funct5)
        if op is None:  # e.g. lr.w / sc.w (Zalrsc) — unsupported on hardware
            return False
        name, fn = op
        rs1 = RV_ISA.get_int(instr, 15, 19)
        rs2 = RV_ISA.get_int(instr, 20, 24)
        rd = RV_ISA.get_int(instr, 7, 11)
        mem_addr = register_file[rs1].read_uint()
        old = conv_to_uint32(memory_space.read(mem_addr, 4))
        src = register_file[rs2].read_uint()
        memory_space.write(mem_addr, conv_to_bytes(fn(old, src) & _MASK32))
        if snoop:
            RV_ISA.print_snoop(
                snoop,
                f"{name}.w {cls.get_reg_name(rd)}, {cls.get_reg_name(rs2)}, "
                f"({cls.get_reg_name(rs1)})",
                f"{cls.get_reg_name(rd)} = [{hex(mem_addr)}] (={hex(old)}); "
                f"[{hex(mem_addr)}] = {name}(old, {cls.get_reg_name(rs2)})",
            )
        register_file[rd].write(conv_to_bytes(old))
        return True
