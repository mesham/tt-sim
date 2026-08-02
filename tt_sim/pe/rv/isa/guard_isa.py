"""Guards for Blackhole baby-RISC-V extensions that are documented but not yet
modelled in tt-sim: the **F single-precision** floating-point extension and the
V vector extension. (Half-precision **Zfh** *is* modelled — see ``zfh_isa.py``;
its ISA is attached ahead of this guard, so ``.h`` instructions are handled and
only single-precision ``.s`` reaches the F guard.)

These extensions are only *partially* implemented in the hardware itself (per
``BlackholeA0/TensixTile/BabyRISCV/InstructionSet.md``: F omits ``fdiv.s`` /
``fsqrt.s``; V is RISC-V T2 only and omits the divide/sqrt/reciprocal family),
and no tt-metal kernel exercised so far uses them from the baby cores — scalar
and vector math there runs on the Tensix coprocessor, not the RISC-V F/V units.

Rather than silently mis-execute (the base ISA would otherwise treat an unknown
op as UndefinedBehavior — often a no-op — which hides real gaps), these ISAs
claim the relevant opcodes and raise ``NotImplementedError`` so that the first
kernel to actually use F/V fails loudly and points here. Implementing the
supported subset (single-precision add/sub/mul/fma reusing the FP register file
Zfh already added; the vector unit for T2) is future work; wire the real ISA in
place of the guard then.

Attachment is arch- and core-specific (see ``BabyRISCV``): the F guard goes on
every Blackhole baby core; the V guard goes on TRISC2 alone, matching the doc's
"RISC-V T2 only" note.
"""

from tt_sim.pe.rv.isa.rv_isa import RV_ISA


class _OpcodeGuard(RV_ISA):
    #: Opcodes this guard claims. Subclasses set it.
    OPCODES: frozenset = frozenset()
    #: Human name of the guarded extension, for the error message.
    EXTENSION = "?"

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        if (instr & 0x7F) not in cls.OPCODES:
            return False
        addr = register_file["pc"].read_uint()
        raise NotImplementedError(
            f"{cls.EXTENSION} extension instruction {hex(instr)} at PC {hex(addr)} "
            f"is not yet modelled in tt-sim. It is (partially) supported by the "
            f"Blackhole baby cores; implement the supported subset and replace "
            f"this guard (see tt_sim/pe/rv/isa/guard_isa.py)."
        )


class RV_F_GUARD_ISA(_OpcodeGuard):
    """F single-precision floating-point: LOAD-FP, STORE-FP, OP-FP and the fused
    multiply-add family. Half-precision (Zfh) forms are handled by RV_ZFH_ISA
    ahead of this guard, so in practice only ``.s`` (and ``.d``/``.q``) reach
    here."""

    OPCODES = frozenset({0x07, 0x27, 0x53, 0x43, 0x47, 0x4B, 0x4F})
    EXTENSION = "F single-precision floating-point"


class RV_V_GUARD_ISA(_OpcodeGuard):
    """V vector extension (OP-V: vector arithmetic and ``vsetvl*``)."""

    OPCODES = frozenset({0x57})
    EXTENSION = "V vector"
