"""Zicsr — CSR instructions — and the Blackhole baby-RISC-V CSR file.

Source: ``BlackholeA0/TensixTile/BabyRISCV/CSRs.md`` (plus its ``InstructionSet.md``
and ``README.md`` siblings). That table is the **only** source, and every value
here is traceable to it. **Blackhole only, by construction**: the string ``csr``
appears nowhere in the ``WormholeB0`` doc tree, so a Wormhole baby core has no
CSRs to model. A CSR instruction on a core without a :class:`CSRFile` raises
:class:`NoCSRsError` rather than executing — see "the defect this fixes" below.

What is modelled
----------------

* All six Zicsr instructions (``csrrw`` / ``csrrs`` / ``csrrc`` and their
  ``i`` forms), with the spec's side-effect-suppression rules: ``csrrw`` with
  ``rd == x0`` performs **no read**, and ``csrrs`` / ``csrrc`` with a zero
  ``rs1`` / ``uimm`` perform **no write**. Both matter here — the first because
  reading an unmodelled CSR is loud, the second because a naive read-modify-write
  would clobber the counters on what the assembler spells ``csrr``.
* ``mcycle`` / ``mcycleh``, off the **same clock the tile's
  ``RISCV_DEBUG_REG_WALL_CLOCK_*`` MMIO reads** (the owning ``TileClock``), not a
  private counter that could drift from it. A core whose CSR file has no clock
  bound refuses the read rather than inventing a cycle number.
* ``minstret`` / ``minstreth``, incremented by ``RV32I.clock_tick`` on each tick
  that **retired** an instruction: an instruction the cost model held (stalled)
  retires nothing, nor does a ``PEStall``, nor does an unknown instruction (which
  the hardware would trap on and tt-sim treats as UndefinedBehavior). Cycles the
  pump skipped while the firmware-loop recogniser had the core parked are added
  back exactly (``tt_sim/pe/rv/spin.py``), so parking cannot silently deflate
  the retire count.
* Both counters are 64-bit and **writable** through either name, per the doc.
  Writes are modelled as an offset from the underlying source, so a counter
  written to N keeps counting from N rather than freezing.
* The ``0xc00`` / ``0xc02`` / ``0xc80`` / ``0xc82`` Zicntr shadows as **read/write
  aliases** of the ``m*`` registers — a documented non-conformance (the spec
  makes them read-only aliases). Modelled by folding the address, so a write
  through either name is visible through the other.
* Read-only ``mstatus`` (``0x80006600``), ``misa`` (``0x40201123``), ``mhartid``
  (0) and ``vstart`` (0). Writes to these are **ignored, not trapped**: the doc
  describes them as read-only registers, and the hardware has no trap to take.
* ``cfg0`` and the other configuration/scratch CSRs as plain storage, with
  ``cfg0`` resetting to ``StMergeTimer = 16`` (bits 13..17) and everything else
  clear, and ``vlenb`` resetting to 16.
* ``fcsr`` (``0x003``) **aliased onto the existing FP register file** entry
  (``rv32.FCSR_INDEX``), not duplicated — the F/Zfh ISAs already own that word.

What is refused, and why
------------------------

Each of these raises rather than returning a plausible value, the same trade the
tile-control block makes for its unmodelled status registers: zero is a
*believable* answer for every one of them, and a believable wrong answer is
worse than a loud one.

* ``tt_cfg_qstatus`` (``0xbc0``) / ``tt_cfg_bstatus`` (``0xbc1``) report live
  Tensix frontend/backend occupancy as 11-bit per-instruction-type bitmasks.
  tt-sim models the frontend and the backend units, so these are *reachable* in
  principle, but the mapping from its queues to the doc's bitmask (and the
  thread-specific versus or-reduced halves, and the SFPU lane-enable bit) is a
  piece of work with its own correctness question. Until it is done, a wrong
  busy-status would be read as "the coprocessor is idle" by a spin loop.
* ``tt_cfg_sstatus0..7`` (``0xbc2``-``0xbc9``) are plain scratch **on RISCV B and
  RISCV NC only**. Elsewhere they read NoC-overlay stream registers — and the
  doc says "of *some* NoC Overlay stream" without naming which, so there is
  nothing to wire even if the overlay were modelled. Scratch on B/NC, refused on
  the TRISCs (and on eth cores, which this Tensix-tile doc does not cover).
* ``mhpmcounter3`` / ``mhpmcounter4`` (and their ``h`` halves) exist, but the
  encodings for their ``mhpmevent3`` / ``mhpmevent4`` selectors are **unpublished**,
  so no event can be given a meaning and no count can be honest. The selectors
  are storage; the counters read as 0 **while no event is selected** — which is
  the true count of "nothing is being counted" — and refuse once software has
  selected an event, because at that point 0 is a claim about an event tt-sim
  cannot identify.
* ``intp_restore_pc`` (``0xbca``) is a copy of the ``pc`` an ``mret`` would
  return to. tt-sim models no interrupts, so before software writes it there is
  no such pc; reads refuse. After a write it reads back what was written, which
  is exactly the hardware's behaviour for a register whose writes the doc says
  ``mret`` ignores.
* Any address not in the doc's table (``mepc``, ``mcause``, ``mtvec``, ``mie``,
  ``time``, ...) raises. The spec's answer is an illegal-instruction trap, which
  tt-sim has no machinery for; the doc explicitly records that ``time`` is *not*
  implemented.

Deliberately not modelled (and not refused)
-------------------------------------------

* **``cfg0`` bit 8 ``PmcClrOnRd`` and bit 7 ``DisPmcWrapArnd``** describe HPM
  counter behaviour — clear-on-read, and stopping at 2^64. Both are about
  ``mhpmcounter3/4``, which by the paragraph above can never leave zero, so
  ``PmcClrOnRd`` would clear a zero to zero and ``DisPmcWrapArnd`` guards a wrap
  that a stationary counter cannot reach. Implementing either would be theatre:
  code that looks like a feature and can never change an observed value. They
  stay as stored bits, and if the event encodings are ever published they must
  be implemented at the same time as the counters they qualify.
* **``cfg0`` bit 10 ``DisCsrSync``** (and bit 9 ``SyncAllOps``): when clear, a
  CSR instruction serialises the frontend until it retires. That is a *cycle*
  cost, and the doc gives **no number** for it — no cycle count for the
  serialisation, none for a CSR instruction at all. Per this repo's provenance
  rules an unsourceable number may not be invented, so Zicsr is charged nothing
  in ``tt_sim/pe/rv/cost.py`` and the omission is recorded there.
* **``pmacfg0`` / ``pmacfg1``** (strong ordering for all loads and stores) are
  storage. tt-sim executes every load and store to completion in program order
  already, so the ordering they request is what it does regardless.
* **``mcountinhibit``** is storage: the doc records that it cannot inhibit
  ``mcycle`` or ``minstret`` (a non-conformance), and the only other counters it
  could inhibit are the HPM pair that never count. So the register is observable
  but inert, which is the hardware's behaviour for every bit tt-sim can reach.
* **``vstart`` / ``vl`` / ``vtype`` / ``vlenb`` / ``vxsat`` / ``vxrm``** are
  present on every baby core because the doc's table is not qualified per core,
  even though the V *instructions* are RISCV T2 only (and guarded — see
  ``guard_isa.py``). They are storage; nothing executes vector instructions, so
  nothing consults them.

The defect this fixes
---------------------

Before this module, ``RV_I_ISA.handle_i_misc`` returned True for the whole
SYSTEM opcode, including funct3 1-7. A CSR read was therefore a **silent no-op
that left ``rd`` holding its previous value** — not UndefinedBehavior, not a
loud failure, just a wrong register. ``i_isa`` now declines funct3 1-7 so this
ISA sees them, and refuses loudly when the core has no CSR file at all.
"""

from tt_sim.pe.rv.isa.rv_isa import RV_ISA
from tt_sim.util.conversion import conv_to_bytes

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF

# -- addresses ------------------------------------------------------------

CSR_FCSR = 0x003
CSR_MCYCLE = 0xB00
CSR_MINSTRET = 0xB02
CSR_MCYCLEH = 0xB80
CSR_MINSTRETH = 0xB82
CSR_MHPMEVENT3 = 0x323
CSR_MHPMEVENT4 = 0x324
CSR_QSTATUS = 0xBC0
CSR_BSTATUS = 0xBC1
CSR_INTP_RESTORE_PC = 0xBCA
CSR_CFG0 = 0x7C0

#: Every CSR the doc's table lists, by address. Also the "does this exist"
#: predicate: an address absent from here is not recognised by the hardware.
CSR_NAMES = {
    0x003: "fcsr",
    0x008: "vstart",
    0x009: "vxsat",
    0x00A: "vxrm",
    0x300: "mstatus",
    0x301: "misa",
    0x320: "mcountinhibit",
    0x323: "mhpmevent3",
    0x324: "mhpmevent4",
    0x7C0: "cfg0",
    0x7C1: "pmacfg0",
    0x7C2: "pmacfg1",
    0x7C3: "cfg1",
    0x7C4: "hwa_mask",
    0x7C5: "hwa_cfg",
    0x7C6: "vgsrc",
    0xB00: "mcycle",
    0xB02: "minstret",
    0xB03: "mhpmcounter3",
    0xB04: "mhpmcounter4",
    0xB80: "mcycleh",
    0xB82: "minstreth",
    0xB83: "mhpmcounter3h",
    0xB84: "mhpmcounter4h",
    0xBC0: "tt_cfg_qstatus",
    0xBC1: "tt_cfg_bstatus",
    0xBC2: "tt_cfg_sstatus0",
    0xBC3: "tt_cfg_sstatus1",
    0xBC4: "tt_cfg_sstatus2",
    0xBC5: "tt_cfg_sstatus3",
    0xBC6: "tt_cfg_sstatus4",
    0xBC7: "tt_cfg_sstatus5",
    0xBC8: "tt_cfg_sstatus6",
    0xBC9: "tt_cfg_sstatus7",
    0xBCA: "intp_restore_pc",
    0xC00: "cycle",
    0xC02: "instret",
    0xC20: "vl",
    0xC21: "vtype",
    0xC22: "vlenb",
    0xC80: "cycleh",
    0xC82: "instreth",
    0xF14: "mhartid",
}

#: The Zicntr shadows are **read/write** aliases of the machine counters (a
#: documented non-conformance: the spec makes them read-only aliases). Folding
#: the address here is what makes a write through either name visible through
#: the other, because there is then only one piece of state.
CSR_ALIASES = {
    0xC00: CSR_MCYCLE,
    0xC02: CSR_MINSTRET,
    0xC80: CSR_MCYCLEH,
    0xC82: CSR_MINSTRETH,
}

#: Read-only registers and the constant each reads as. Writes are ignored.
CSR_READ_ONLY = {
    0x008: 0x00000000,  # vstart: always zero (should be writable per spec)
    0x300: 0x80006600,  # mstatus: FP + vector register state permanently dirty
    0x301: 0x40201123,  # misa: RV32IMABFV
    0xF14: 0x00000000,  # mhartid: always zero (should be unique per hart)
}

#: Reset values for the plain read/write registers; anything storage-backed and
#: absent from here resets to zero.
CSR_RESET_VALUES = {
    # cfg0: "At reset, all fields are initialized to zero/clear, except for
    # StMergeTimer" (bits 13..17), "which is initialized to 16".
    CSR_CFG0: 16 << 13,
    # vlenb: "Value is initially 16" (16-byte vector registers).
    0xC22: 16,
}

#: Plain read-what-you-wrote storage, present on every baby core.
_STORAGE = frozenset(
    {
        0x009,  # vxsat
        0x00A,  # vxrm
        0x320,  # mcountinhibit
        0x323,  # mhpmevent3
        0x324,  # mhpmevent4
        0x7C0,  # cfg0
        0x7C1,  # pmacfg0
        0x7C2,  # pmacfg1
        0x7C3,  # cfg1
        0x7C4,  # hwa_mask
        0x7C5,  # hwa_cfg
        0x7C6,  # vgsrc
        0xC20,  # vl
        0xC21,  # vtype
        0xC22,  # vlenb
    }
)

#: ``tt_cfg_sstatus0..7`` — scratch on RISCV B / RISCV NC, NoC-overlay stream
#: registers everywhere else.
_SSTATUS = frozenset(range(0xBC2, 0xBCA))

#: The two HPM counter halves, mapped to the event selector that qualifies them.
_HPM_COUNTERS = {
    0xB03: CSR_MHPMEVENT3,
    0xB83: CSR_MHPMEVENT3,
    0xB04: CSR_MHPMEVENT4,
    0xB84: CSR_MHPMEVENT4,
}

#: Which half of a 64-bit counter an address names.
_COUNTER_HIGH = frozenset({CSR_MCYCLEH, CSR_MINSTRETH, 0xB83, 0xB84})


def csr_name(addr):
    """``"mcycle"``-style name for ``addr``, or its hex form when unrecognised."""
    return CSR_NAMES.get(addr, hex(addr))


class CSRError(Exception):
    """Base for every refusal this module makes."""


class NoCSRsError(CSRError):
    """A CSR instruction on a core that has no CSR file.

    Wormhole baby cores are the case that matters: the WormholeB0 docs describe
    no CSRs at all, so there is nothing to read and no honest value to return.
    """


class UnknownCSRError(CSRError):
    """An address the hardware does not recognise (absent from :data:`CSR_NAMES`)."""


class UnmodelledCSRError(CSRError):
    """A documented CSR whose value tt-sim cannot produce truthfully."""


def _at(pc):
    return "" if pc is None else f" at PC {hex(pc)}"


class CSRFile:
    """One baby core's CSRs.

    Held by the core *and* by its :class:`~tt_sim.pe.register.register_file.RegisterFile`
    (one object, two references): the register file is what the ISA executors are
    handed, and the core is where the retire counter is bumped from.

    ``clock_owner`` is the owning tile's ``TileClock`` — the same object
    ``TensixTileControl`` reads ``RISCV_DEBUG_REG_WALL_CLOCK_*`` from — bound by
    ``TTDeviceTile._bind_clock``. It stays ``None`` for a core built outside a
    device (the ISA unit tests, ``driver/simple``), and ``mcycle`` then refuses
    rather than inventing a clock of its own.
    """

    def __init__(
        self, register_file, core_label="?", scratch_sstatus=False, clock_owner=None
    ):
        self.register_file = register_file
        self.core_label = core_label
        #: True on RISCV B and RISCV NC, where ``tt_cfg_sstatus*`` are scratch.
        self.scratch_sstatus = scratch_sstatus
        self.clock_owner = clock_owner
        self.store = dict(CSR_RESET_VALUES)
        #: Instructions this core has retired. Bumped by ``RV32I.clock_tick``
        #: (and by ``spin.py`` for the ticks a parked span skipped); the
        #: architectural ``minstret`` is this plus :attr:`_instret_base`.
        self.retired = 0
        self._cycle_base = 0
        self._instret_base = 0
        self._intp_restore_pc = None

    # -- counters ---------------------------------------------------------

    def _clock_cycles(self, addr, pc):
        owner = self.clock_owner
        if owner is None:
            raise UnmodelledCSRError(
                f"{csr_name(addr)} ({hex(addr)}) read on {self.core_label}"
                f"{_at(pc)}, but this core's CSR file has no clock bound. "
                f"tt-sim's only cycle counter is the owning tile's TileClock "
                f"(the one RISCV_DEBUG_REG_WALL_CLOCK_* reads); a core built "
                f"outside a device has no access to it, and a private counter "
                f"here would be free to disagree with every other cycle number "
                f"in the simulator. Bind a clock (TTDeviceTile._bind_clock) or "
                f"do not read this CSR."
            )
        return owner.current_cycle

    def _counter64(self, addr, pc):
        """The full 64-bit value of the counter ``addr`` names half of."""
        if addr in (CSR_MCYCLE, CSR_MCYCLEH):
            return (self._clock_cycles(addr, pc) + self._cycle_base) & _MASK64
        return (self.retired + self._instret_base) & _MASK64

    def _write_counter(self, addr, value, pc):
        """Write one half of a counter, leaving the other half alone.

        Stored as an offset from the underlying source rather than as a value,
        so a counter written to N carries on counting from N — which is what
        the hardware does, and what makes "read, run, read again" a measurement
        rather than a constant.
        """
        current = self._counter64(addr, pc)
        if addr in _COUNTER_HIGH:
            wanted = ((value & _MASK32) << 32) | (current & _MASK32)
        else:
            wanted = (current & (_MASK32 << 32)) | (value & _MASK32)
        if addr in (CSR_MCYCLE, CSR_MCYCLEH):
            self._cycle_base = (wanted - self._clock_cycles(addr, pc)) & _MASK64
        else:
            self._instret_base = (wanted - self.retired) & _MASK64

    # -- access -----------------------------------------------------------

    def _check_known(self, addr, pc, verb):
        if addr not in CSR_NAMES:
            raise UnknownCSRError(
                f"CSR {hex(addr)} {verb} on {self.core_label}{_at(pc)} is not one "
                f"the Blackhole baby cores recognise (see BlackholeA0/TensixTile/"
                f"BabyRISCV/CSRs.md for the complete list). The RISC-V spec's "
                f"answer is an illegal-instruction trap, which tt-sim does not "
                f"model."
            )

    def _refuse_tensix_status(self, addr, pc):
        raise UnmodelledCSRError(
            f"{csr_name(addr)} ({hex(addr)}) read on {self.core_label}{_at(pc)} "
            f"reports live Tensix frontend/backend occupancy as an 11-bit "
            f"per-instruction-type bitmask. tt-sim models those units but does "
            f"not yet map them onto this bitmask, and answering 0 would tell a "
            f"spin loop the coprocessor is idle when it may not be. See "
            f"tt_sim/pe/rv/isa/zicsr_isa.py."
        )

    def _refuse_sstatus(self, addr, pc, verb):
        raise UnmodelledCSRError(
            f"{csr_name(addr)} ({hex(addr)}) {verb} on {self.core_label}{_at(pc)}: "
            f"this CSR is plain scratch only on RISCV B and RISCV NC. On the "
            f"TRISCs it reads a NoC Overlay stream register, and the ISA doc "
            f"names no particular stream, so there is nothing to model."
        )

    def read(self, addr, pc=None):
        """Read a CSR, refusing anything tt-sim cannot answer truthfully."""
        self._check_known(addr, pc, "read")
        addr = CSR_ALIASES.get(addr, addr)

        if addr in CSR_READ_ONLY:
            return CSR_READ_ONLY[addr]
        if addr == CSR_FCSR:
            return self._fcsr_register(pc).read_uint()
        if addr in (CSR_MCYCLE, CSR_MCYCLEH, CSR_MINSTRET, CSR_MINSTRETH):
            value = self._counter64(addr, pc)
            return (value >> 32) & _MASK32 if addr in _COUNTER_HIGH else value & _MASK32
        if addr in _HPM_COUNTERS:
            event = self.store.get(_HPM_COUNTERS[addr], 0)
            if event != 0:
                raise UnmodelledCSRError(
                    f"{csr_name(addr)} ({hex(addr)}) read on {self.core_label}"
                    f"{_at(pc)} with {csr_name(_HPM_COUNTERS[addr])} = {hex(event)}. "
                    f"The event encodings for mhpmevent3/mhpmevent4 are "
                    f"unpublished, so tt-sim cannot know what was selected and "
                    f"cannot count it; 0 would be a claim about an event it "
                    f"cannot identify. With no event selected the counter reads "
                    f"0, which is true."
                )
            return 0
        if addr == CSR_INTP_RESTORE_PC:
            if self._intp_restore_pc is None:
                raise UnmodelledCSRError(
                    f"intp_restore_pc ({hex(addr)}) read on {self.core_label}"
                    f"{_at(pc)} before anything wrote it. It holds a copy of the "
                    f"pc an mret would return to, and tt-sim models no "
                    f"interrupts, so there is no such pc to report."
                )
            return self._intp_restore_pc
        if addr in (CSR_QSTATUS, CSR_BSTATUS):
            self._refuse_tensix_status(addr, pc)
        if addr in _SSTATUS:
            if not self.scratch_sstatus:
                self._refuse_sstatus(addr, pc, "read")
            return self.store.get(addr, 0)
        if addr in _STORAGE:
            return self.store.get(addr, 0)
        raise UnmodelledCSRError(  # pragma: no cover - every listed addr is handled
            f"{csr_name(addr)} ({hex(addr)}) is documented but has no behaviour "
            f"in tt-sim's CSR file."
        )

    def write(self, addr, value, pc=None):
        """Write a CSR. Writes to read-only registers are ignored, per the doc."""
        self._check_known(addr, pc, "written")
        addr = CSR_ALIASES.get(addr, addr)
        value &= _MASK32

        if addr in CSR_READ_ONLY:
            # Documented read-only. The spec would trap on a write to one; this
            # hardware simply has nothing to write, and its own firmware would
            # be the first casualty of a raise here.
            return
        if addr == CSR_FCSR:
            self._fcsr_register(pc).write(conv_to_bytes(value))
            return
        if addr in (CSR_MCYCLE, CSR_MCYCLEH, CSR_MINSTRET, CSR_MINSTRETH):
            self._write_counter(addr, value, pc)
            return
        if addr in _HPM_COUNTERS:
            # Storage, and deliberately inert: nothing increments these (see the
            # module docstring), so a write is only ever observed by the read
            # path's "an event is selected" refusal.
            self.store[addr] = value
            return
        if addr == CSR_INTP_RESTORE_PC:
            self._intp_restore_pc = value
            return
        if addr in (CSR_QSTATUS, CSR_BSTATUS):
            raise UnmodelledCSRError(
                f"{csr_name(addr)} ({hex(addr)}) written on {self.core_label}"
                f"{_at(pc)}. Software can overwrite these status registers on "
                f"hardware, but tt-sim does not model the state they otherwise "
                f"report (see the read path), so storing the write would make a "
                f"later read look answerable when it is not."
            )
        if addr in _SSTATUS:
            if not self.scratch_sstatus:
                self._refuse_sstatus(addr, pc, "written")
            self.store[addr] = value
            return
        self.store[addr] = value

    def _fcsr_register(self, pc):
        """``fcsr`` lives in the FP register file, where the F/Zfh ISAs put it.

        Aliased rather than copied: a second copy could disagree with the one
        those ISAs address.
        """
        try:
            return self.register_file["fcsr"]
        except (IndexError, KeyError):
            raise UnmodelledCSRError(
                f"fcsr ({hex(CSR_FCSR)}) accessed on {self.core_label}{_at(pc)}, "
                f"which was built without a floating-point register file. fcsr "
                f"is that file's last entry (rv32.FCSR_INDEX); a core with no "
                f"FP unit has nowhere to keep it."
            ) from None

    def reset(self):
        """Back to reset values. Called when the core is (re)started."""
        self.store = dict(CSR_RESET_VALUES)
        self.retired = 0
        self._cycle_base = 0
        self._instret_base = 0
        self._intp_restore_pc = None


_OP_NAMES = {1: "csrrw", 2: "csrrs", 3: "csrrc", 5: "csrrwi", 6: "csrrsi", 7: "csrrci"}


class RV_ZICSR_ISA(RV_ISA):
    """The six Zicsr instructions, executed against the core's :class:`CSRFile`.

    Claims SYSTEM (``0x73``) with funct3 1-3 and 5-7. funct3 0 (``ecall`` /
    ``ebreak`` / ``mret``) stays with ``RV_I_ISA``; funct3 4 is not a Zicsr
    encoding, so it is declined and falls through to tt-sim's unknown-instruction
    path, which is the doc's UndefinedBehavior for an invalid instruction.
    """

    @classmethod
    def run(cls, register_file, memory_space, snoop, instr=None):
        if instr is None:
            instr = cls.fetch(register_file, memory_space)
        if instr & 0x7F != 0x73:
            return False
        funct3 = (instr >> 12) & 0x7
        if funct3 not in _OP_NAMES:
            return False

        csrs = getattr(register_file, "csrs", None)
        if csrs is None:
            raise NoCSRsError(
                f"CSR instruction {hex(instr)} at PC "
                f"{hex(register_file['pc'].read_uint())} on a core with no CSR "
                f"file. Only Blackhole baby cores have CSRs in tt-sim; see "
                f"tt_sim/pe/rv/isa/zicsr_isa.py."
            )

        addr = (instr >> 20) & 0xFFF
        rd = (instr >> 7) & 0x1F
        rs1 = (instr >> 15) & 0x1F  # a uimm[4:0] in the immediate forms
        immediate = funct3 & 0x4
        operand = rs1 if immediate else register_file[rs1].read_uint()
        pc = register_file["pc"].read_uint()

        old = None
        if funct3 & 0x3 == 1:  # csrrw / csrrwi
            # "If rd=x0, then the instruction shall not read the CSR and shall
            # not cause any of the side effects that might occur on a CSR read."
            # Load-bearing here: the read side of an unmodelled CSR is loud, so
            # a plain `csrw` to one must not go looking.
            if rd != 0:
                old = csrs.read(addr, pc)
            csrs.write(addr, operand, pc)
        else:  # csrrs / csrrc / csrrsi / csrrci
            old = csrs.read(addr, pc)
            # "If rs1=x0, then the instruction will not write to the CSR at
            # all" — which is what keeps the assembler's `csrr rd, csr`
            # (a csrrs with rs1=x0) from writing back over a counter, and what
            # keeps it legal against a read-only CSR.
            if rs1 != 0:
                mask = operand
                new = (old | mask) if (funct3 & 0x3) == 2 else (old & ~mask)
                csrs.write(addr, new & _MASK32, pc)

        if rd != 0:
            register_file[rd].write(conv_to_bytes(old & _MASK32))

        if snoop:
            name = _OP_NAMES[funct3]
            source = str(rs1) if immediate else cls.get_reg_name(rs1)
            RV_ISA.print_snoop(
                snoop,
                f"{name} {cls.get_reg_name(rd)}, {csr_name(addr)}, {source}",
                None if old is None else f"{csr_name(addr)} was {hex(old)}",
            )
        return True
