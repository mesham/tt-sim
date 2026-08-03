"""Charging a baby RISC-V core for its load/store path.

The consuming half of :class:`tt_sim.perf.model.RiscvCostModel`, and the first
thing outside the Tensix coprocessor to read the cycle-cost tables at all
(ROADMAP.md section I, "RV pipeline modelling": *"fetch/decode/issue/retire
stages with memory-stall back-pressure on L1 / NoC reads"*).

**Opt-in, and free when off.** :class:`RiscvCostState` is only constructed when
``TT_SIM_COST_MODEL`` is truthy; otherwise ``RV32I.rv_cost`` is ``None`` and the
interpreter pays one attribute read and one branch per instruction. That matters
more here than anywhere else in the tree: the RV interpreter is the hottest path
in the simulator and was optimised ~2.3x by removing exactly this kind of
per-instruction overhead (see ``driver/wormhole/docs/profiling.md``).

**What is modelled, and what each thing is.** The three costs come from three
different kinds of statement in the ISA docs, and conflating them would be the
error the SFPU wiring already caught once:

1. **Load-use interlock.** The load-latency table is a *latency* table. The doc
   says so itself — "a latency of N cycles means that N - 1 independent
   instructions need to follow the load if the latency is to be entirely
   hidden" — so a load does not occupy the core for N cycles. It writes its
   destination register N cycles later, and an in-order single-issue core
   stalls only when the *next reader of that register* arrives too early.
   That is a scoreboard, and it is what this class mostly is.
2. **L1 store throughput.** "Throughput of sustained stores to L1 is at most
   one store every five cycles" *is* an occupancy of the load/store unit, and
   is charged as one. Blackhole's coalescing store queue is modelled with the
   docs' own predicate (same 16-byte aligned region of L1, start addresses
   within +/-4), because charging its 5 to a coalescable run would over-charge
   by 5x against a document that says the opposite.
3. **Integer unit multiply/divide.** The one place the docs state blocking
   outright: "multiply instructions occupy the Integer Unit for two cycles, and
   the next instruction cannot enter the unit until the multiply instruction
   has finished".

**What is deliberately not modelled** — each an honest gap rather than an
omission:

* **Branch mispredicts.** Sourced (a 2-cycle bubble on Wormhole, 4 on
  Blackhole) but uncountable: neither the docs nor tt-sim describe the
  predictor, so the number of mispredictions is unknowable and charging every
  taken branch would be a fabrication.
* **Instruction fetch / i-cache misses.** The table gives the fetch *period*
  (one 128-bit L1 read per four instructions) but no miss cost.
* **Sustained-load throughput** (four loads per N-1 cycles) and
  ``max_loads_in_flight``. Both are queue-depth statements about loads already
  in flight, and the interlock above already stalls on the dependent read,
  which is the first-order effect.
* **Regions the table does not name** — see
  :data:`tt_sim.perf.model.RV_UNNAMED_REGIONS`, which includes the NoC NIU
  register block that every ``noc_async_*_barrier`` polls.

Every one of those under-charges, which is the direction the cost model's
stated policy asks for: a modelled cycle count is a floor.
"""

from __future__ import annotations

from tt_sim.perf.model import (
    RV_REGION_COUNT,
    RV_REGION_L1,
    RV_REGION_LOCAL_DATA_RAM,
    RV_REGION_MAILBOX_GROUP,
    RV_REGION_TDMA,
    RV_REGION_TENSIX_GPR_CFG,
    RV_REGION_TILECTRL_PIC_OVERLAY,
    RV_REGION_UNNAMED,
    riscv_cost_model,
)

# MMIO block bases, read straight off the memory map ``TensixTile.__init__``
# builds in ``tt_sim/device/tiles.py``. Anything below ``_MMIO_BASE`` is L1 —
# the same discriminator ``MemorySpace._classify_region`` already uses.
_MMIO_BASE = 0xFFB00000
_LOCAL_DATA_RAM_END = 0xFFB10000  # local data RAM is <= 8 KiB at 0xFFB00000
_TDMA_BASE = 0xFFB11000
_TILE_CTRL_BASE = 0xFFB12000
_TILE_CTRL_END = 0xFFB13000
_NOC_OVERLAY_BASE = 0xFFB40000
_NOC_OVERLAY_END = 0xFFB50000
_TENSIX_GPR_BASE = 0xFFE00000
_TENSIX_GPR_END = 0xFFE10000
_PCBUF_TTSYNC_SEM_BASE = 0xFFE80000
_PCBUF_TTSYNC_SEM_END = 0xFFEB0000
_MAILBOX_BASE = 0xFFEC0000
_MAILBOX_END = 0xFFED0000
_TENSIX_CFG_BASE = 0xFFEF0000


def classify_address(addr):
    """Which load-latency row of the ISA docs' table ``addr`` falls under.

    Ordered cheapest-discriminator-first: L1 is both the largest region and the
    common case for kernel data, so it costs one comparison.
    """
    if addr < _MMIO_BASE:
        return RV_REGION_L1
    if addr < _LOCAL_DATA_RAM_END:
        return RV_REGION_LOCAL_DATA_RAM
    if _TDMA_BASE <= addr < _TILE_CTRL_BASE:
        return RV_REGION_TDMA
    if _TILE_CTRL_BASE <= addr < _TILE_CTRL_END:
        return RV_REGION_TILECTRL_PIC_OVERLAY
    if _NOC_OVERLAY_BASE <= addr < _NOC_OVERLAY_END:
        return RV_REGION_TILECTRL_PIC_OVERLAY
    if _TENSIX_GPR_BASE <= addr < _TENSIX_GPR_END:
        return RV_REGION_TENSIX_GPR_CFG
    if addr >= _TENSIX_CFG_BASE:
        return RV_REGION_TENSIX_GPR_CFG
    if _PCBUF_TTSYNC_SEM_BASE <= addr < _PCBUF_TTSYNC_SEM_END:
        return RV_REGION_MAILBOX_GROUP
    if _MAILBOX_BASE <= addr < _MAILBOX_END:
        return RV_REGION_MAILBOX_GROUP
    return RV_REGION_UNNAMED


# Which register fields an opcode actually reads, indexed by ``instr & 0x7F``.
# A generic "check bits 15-19 and 20-24" would be wrong for U- and J-format,
# whose immediates occupy those bits, and a false stall is exactly the kind of
# invented back-pressure the cost model's policy forbids. An opcode absent from
# this table reads nothing and writes nothing as far as the interlock is
# concerned, which under-charges rather than over-charges.
_F_RS1 = 1
_F_RS2 = 2
_F_RD = 4

_OPCODE_FLAGS = [0] * 128
for _op in (0x37, 0x17, 0x6F):  # LUI / AUIPC / JAL: immediate only
    _OPCODE_FLAGS[_op] = _F_RD
for _op in (0x03, 0x13, 0x67):  # LOAD / OP-IMM / JALR
    _OPCODE_FLAGS[_op] = _F_RS1 | _F_RD
for _op in (0x33, 0x2F):  # OP (incl. M, Zba, Zbb) / AMO
    _OPCODE_FLAGS[_op] = _F_RS1 | _F_RS2 | _F_RD
for _op in (0x23, 0x63):  # STORE / BRANCH
    _OPCODE_FLAGS[_op] = _F_RS1 | _F_RS2
for _op in (0x07, 0x27):  # FLH / FSH: the address register is a GPR
    _OPCODE_FLAGS[_op] = _F_RS1
# SYSTEM (0x73) is left out on purpose: ``csrrwi`` and friends put a uimm in
# the rs1 field, so treating it as a register read would stall on a register
# the instruction never touches.

#: Why a core was not able to issue this cycle. Counted per reason so a run can
#: say where its RV time went rather than only how much there was.
STALL_LOAD_USE = 0
STALL_STORE_RATE = 1
STALL_INTEGER_UNIT = 2
STALL_REASON_COUNT = 3
STALL_REASON_NAMES = ("load_use", "store_rate", "integer_unit")

_LOAD_OPCODE = 0x03
_STORE_OPCODE = 0x23
_OP_OPCODE = 0x33
_MULDIV_FUNCT7 = 0x01
_INT_MIN = 0x80000000


class RiscvCostState:
    """One core's cost-model state: a scoreboard, a store queue and a counter.

    Held by :class:`~tt_sim.pe.rv.rv32.RV32I` as ``rv_cost``, and consulted
    once per instruction through :meth:`can_issue`, which does the whole job —
    hazard check, rate limit and bookkeeping — in one call so the interpreter's
    inner loop grows one branch rather than three.
    """

    __slots__ = (
        "model",
        "load_latency",
        "l1_store_period",
        "coalesced_stores",
        "multiply",
        "divide_general",
        "divide_trivial",
        "divide_int_min",
        "_ready",
        "_stall_until",
        "_stall_reason",
        "_store_ready",
        "_group_block",
        "_group_lo",
        "_group_hi",
        "stall_cycles",
        "stall_by_reason",
        "loads_by_region",
        "l1_stores",
        "pending_stall",
    )

    def __init__(self, model):
        self.model = model
        self.load_latency = model.load_latency
        self.l1_store_period = model.l1_store_period
        # Blackhole publishes a coalesced store rate; Wormhole does not have
        # the queue at all, so its absence is the discriminator rather than an
        # arch name string compared on the hot path.
        self.coalesced_stores = model.l1_coalesced_store_period is not None
        self.multiply = model.multiply
        self.divide_general = model.divide_general
        self.divide_trivial = model.divide_trivial
        self.divide_int_min = model.divide_int_min
        # Cycle at which each GPR's value becomes readable. 0 = ready now, and
        # x0 is never written so index 0 stays 0 forever.
        self._ready = [0] * 32
        self._stall_until = 0
        self._stall_reason = STALL_LOAD_USE
        self._store_ready = 0
        self._group_block = -1
        self._group_lo = 0
        self._group_hi = 0
        self.stall_cycles = 0
        self.stall_by_reason = [0] * STALL_REASON_COUNT
        self.loads_by_region = [0] * RV_REGION_COUNT
        self.l1_stores = 0
        # Stalls accumulated since the last instruction retired, drained by
        # :meth:`take_pending_stall`. Only the trace path reads it, so it is
        # written on the stall path (which is already the slow path) and never
        # on the issue path.
        self.pending_stall = 0

    def reset(self):
        """Drop every in-flight hazard. Called when a core is (re)started, so
        a scoreboard entry from a previous launch cannot stall the next one."""
        ready = self._ready
        for i in range(32):
            ready[i] = 0
        self._stall_until = 0
        self._store_ready = 0
        self._group_block = -1
        self.pending_stall = 0

    # -- reporting ---------------------------------------------------------
    def take_pending_stall(self):
        """``(cycles, reason)`` this core was held before the instruction that
        is about to retire, clearing the accumulator.

        The per-instruction half of :attr:`stall_by_reason`: the same stalls,
        attributed to the instruction that eventually issued rather than only
        totalled, which is what lets a Perfetto slice have a real width. Called
        from ``RV32I.clock_tick`` **only when an ``InstrEvent`` is actually
        being published**, so the interpreter's non-tracing path is untouched.
        The reason is the last one recorded, which for a run of stalled cycles
        is the reason that ended them.
        """
        n = self.pending_stall
        if not n:
            return 0, ""
        self.pending_stall = 0
        return n, STALL_REASON_NAMES[self._stall_reason]

    def summary(self):
        """A plain dict of what this core was charged, for the §I reports."""
        return {
            "stall_cycles": self.stall_cycles,
            "stalls": {
                name: self.stall_by_reason[i]
                for i, name in enumerate(STALL_REASON_NAMES)
            },
            "loads_by_region": list(self.loads_by_region),
            "l1_stores": self.l1_stores,
        }

    # -- the hot path ------------------------------------------------------
    def _stall(self, until, reason):
        self._stall_until = until
        self._stall_reason = reason
        self.stall_cycles += 1
        self.stall_by_reason[reason] += 1
        self.pending_stall += 1
        return False

    def can_issue(self, instr, cycle_num, register_file):
        """True when the core may execute ``instr`` this cycle.

        False means the core is stalled: the caller returns without executing
        and without advancing the PC, so the same instruction is retried on a
        later cycle. Every state change this method makes happens *after* the
        last point at which it can answer False, so a stalled cycle leaves the
        model exactly as it found it.
        """
        if cycle_num < self._stall_until:
            self.stall_cycles += 1
            self.stall_by_reason[self._stall_reason] += 1
            self.pending_stall += 1
            return False
        if instr & 0x3 != 0x3:
            # A .ttinsn — a rotated Tensix instruction word with no GPR
            # operands, pushed to the instruction buffer at 0xFFE40000, which
            # is not a region the load-latency table names.
            return True

        opcode = instr & 0x7F
        flags = _OPCODE_FLAGS[opcode]
        ready = self._ready
        if flags & _F_RS1:
            at = ready[(instr >> 15) & 0x1F]
            if at > cycle_num:
                return self._stall(at, STALL_LOAD_USE)
        if flags & _F_RS2:
            at = ready[(instr >> 20) & 0x1F]
            if at > cycle_num:
                return self._stall(at, STALL_LOAD_USE)

        if opcode == _LOAD_OPCODE:
            imm = instr >> 20
            if imm & 0x800:
                imm -= 0x1000
            addr = (register_file[(instr >> 15) & 0x1F].read_uint() + imm) & 0xFFFFFFFF
            region = classify_address(addr)
            self.loads_by_region[region] += 1
            latency = self.load_latency[region]
            rd = (instr >> 7) & 0x1F
            if rd:
                # The value is written architecturally in this tick (tt-sim's
                # loads are functionally instantaneous); the scoreboard is what
                # makes it *unreadable* until the latency has elapsed. Setting
                # it here rather than after the handler is deliberate: the
                # handler may write rd, and rd may be rs1.
                ready[rd] = 0 if latency is None else cycle_num + latency
            return True

        if opcode == _STORE_OPCODE:
            if self.l1_store_period is None:
                return True
            imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
            if imm & 0x800:
                imm -= 0x1000
            addr = (register_file[(instr >> 15) & 0x1F].read_uint() + imm) & 0xFFFFFFFF
            if addr >= _MMIO_BASE:
                # "Other memory regions can achieve a throughput of one store
                # every cycle", which is what the simulator already does.
                return True
            return self._issue_l1_store(addr, cycle_num)

        if flags & _F_RD:
            rd = (instr >> 7) & 0x1F
            if rd:
                # Overwriting a register that a load is still in flight for
                # retires the hazard: an in-order core would not stall a later
                # reader of a value the load no longer supplies.
                ready[rd] = 0
        if opcode == _OP_OPCODE and (instr >> 25) == _MULDIV_FUNCT7:
            occupancy = self._muldiv_occupancy(instr, register_file)
            if occupancy is not None and occupancy > 1:
                # The instruction runs *this* cycle and holds the integer unit
                # for the rest, so the next issue is at cycle + occupancy.
                self._stall_until = cycle_num + occupancy
                self._stall_reason = STALL_INTEGER_UNIT
        return True

    def _issue_l1_store(self, addr, cycle_num):
        if self.coalesced_stores:
            block = addr >> 4
            # A store can only join a group that is still in the queue, i.e.
            # one whose five-cycle drain has not completed.
            if block == self._group_block and cycle_num < self._store_ready:
                lo = addr if addr < self._group_lo else self._group_lo
                hi = addr if addr > self._group_hi else self._group_hi
                if hi - lo <= 4:
                    # Coalesces into the open group: "the constituent stores
                    # have a throughput of one store every cycle".
                    self._group_lo, self._group_hi = lo, hi
                    self.l1_stores += 1
                    return True
            if cycle_num < self._store_ready:
                return self._stall(self._store_ready, STALL_STORE_RATE)
            self._group_block = block
            self._group_lo = self._group_hi = addr
        elif cycle_num < self._store_ready:
            return self._stall(self._store_ready, STALL_STORE_RATE)
        self._store_ready = cycle_num + self.l1_store_period
        self.l1_stores += 1
        return True

    def _muldiv_occupancy(self, instr, register_file):
        if instr & 0x4000:  # funct3 >= 4: div / divu / rem / remu
            divisor = register_file[(instr >> 20) & 0x1F].read_uint()
            if divisor == 0 or divisor == 1:
                return self.divide_trivial
            if divisor == 0xFFFFFFFF:
                dividend = register_file[(instr >> 15) & 0x1F].read_uint()
                if dividend == _INT_MIN:
                    return self.divide_int_min
            return self.divide_general
        return self.multiply


def make_cost_state(arch):
    """A :class:`RiscvCostState` for ``arch``, or ``None`` when the model is off.

    ``arch`` is ``None`` for cores built outside a device (the ``driver/simple``
    examples, the ISA unit tests), which never opt in.
    """
    model = riscv_cost_model(arch)
    return None if model is None else RiscvCostState(model)
