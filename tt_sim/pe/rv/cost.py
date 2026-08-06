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

**What is modelled, and what each thing is.** The four costs come from four
different kinds of statement in the ISA docs — a latency, a queue's throughput,
a unit's occupancy and a unit's blocking — and conflating them would be the
error the SFPU wiring already caught once:

1. **Load-use interlock.** The load-latency table is a *latency* table. The doc
   says so itself — "a latency of N cycles means that N - 1 independent
   instructions need to follow the load if the latency is to be entirely
   hidden" — so a load does not occupy the core for N cycles. It writes its
   destination register N cycles later, and an in-order single-issue core
   stalls only when the *next reader of that register* arrives too early.
   That is a scoreboard, and it is what this class mostly is.

   On Blackhole the L1 row is a **pair** — 2 on an L0 d-cache hit, >= 8 on a
   miss — and which of the two a given load pays is decided by a minimal
   per-core L0 line model: the ``riscv.l0_data_cache`` block publishes the
   geometry ("a mere 64 bytes: 4 lines of 16 bytes each", ``isa_doc``), so the
   model keeps four line *tags* per core (no data — tt-sim's loads stay
   functionally instantaneous) and charges the hit row when the loaded line's
   tag is resident, the miss row when it is not. The published flushes are
   honoured too, because they are documented and skipping them would
   mis-charge in both directions: an L1 store invalidates its containing line
   ("stores to L1 flush the containing line"), and a ``fence`` or atomic
   flushes all four ("the entire L0 data cache will be flushed by any fence
   or atomic instruction"). What is *not* published is the organisation —
   associativity, indexing, replacement, and a "~0.8 % chance of flushing the
   entire cache" on a hit — so the model takes the generous side of each:
   fully associative, least-recently-loaded replacement, no random flush.
   Every one of those choices under-charges relative to any stricter
   organisation the hardware might have, which keeps the modelled count a
   floor; what the capacity *does* settle — one-sidedly, and it is the whole
   licence — is that a working set larger than 64 bytes cannot be resident,
   so a chase over it charges the miss row however the real cache is
   organised. Silicon: ``rv_load_chase`` (a 1 KiB ring) reads 8.098 where
   this model charges 8; before the line model every L1 load charged the hit
   row's 2. Wormhole publishes no L0 at all (its single L1 row is >= 8), so
   the model never engages there.
2. **Sustained load throughput.** The load table's other half, and the one
   term that costs a *stream* of loads rather than a dependent pair: "the
   throughput of sustained loads is one per cycle if the load latency is less
   than five cycles. Otherwise, when the load latency is ``N`` cycles, the
   throughput of sustained loads is four such loads every ``N - 1`` cycles."
   Four slots, each held ``N - 1`` cycles; a load that finds no free slot
   stalls until one frees. Rows below the five-cycle threshold (core-local
   RAM, an L0 d-cache hit) take no slot, because one per cycle is what a
   single-issue core does anyway.

   **This is the same hazard as ``max_loads_in_flight`` and is charged once.**
   The table's "Maximum loads in flight" column says 4 in aggregate across
   every region but core-local RAM, and four loads in flight sustaining four
   loads per ``N - 1`` cycles is a residency of exactly ``N - 1`` — Little's
   law on two numbers printed a paragraph apart. So the queue depth and the
   throughput formula are one mechanism written down twice, and charging both
   would bill it twice; :attr:`~tt_sim.perf.model.RiscvCostModel.load_slots`
   reads the formula's own "four", not the in-flight column, and the column
   stays unconsumed. The other half of that column — 8 for core-local data
   RAM — is inert under either reading: at a latency of 2 the formula gives
   one per cycle, which is the issue width. Silicon agrees with the charged
   rate to eight thousandths of a cycle: ``rv_load_indep`` (four rotating
   destination registers, so nothing is ever read early) reads **1.742**
   cycles per load where four slots of 7 cycles give 1.750.
3. **L1 store throughput.** "Throughput of sustained stores to L1 is at most
   one store every five cycles" *is* an occupancy of the load/store unit, and
   is charged as one. Blackhole's coalescing store queue is modelled with the
   docs' own predicate (same 16-byte aligned region of L1, start addresses
   within +/-4), because charging its 5 to a coalescable run would over-charge
   by 5x against a document that says the opposite.
4. **Integer unit multiply/divide.** The one place the docs state blocking
   outright: "multiply instructions occupy the Integer Unit for two cycles, and
   the next instruction cannot enter the unit until the multiply instruction
   has finished". That is Wormhole. Blackhole's multiply *pipelines* instead —
   "exactly one cycle in EX1, and then exactly one cycle in EX2" — which is an
   occupancy of 1 and a result **latency** of 2, so on Blackhole a multiply
   writes a scoreboard entry (:attr:`RiscvCostState.multiply_latency`) exactly
   like a load does, and only a dependent read pays the second cycle. Silicon
   confirms the split to a hundredth: ``rv_mul_indep`` 0.999, ``rv_mul_dep``
   1.985 (tt-sim read 1.000 for both before the scoreboard entry existed).
   Wormhole publishes no multiply latency — only the blocking occupancy, which
   is already charged — so its ``multiply_latency`` is ``None`` and nothing
   there moved.

   **The divide is charged 6, which is the low end of a 6-33 band, and that
   was re-decided rather than inherited on 2026-08-05.** The band is a *data*
   dependence — "between six and 33 cycles are required, dependent upon the
   magnitude of the dividend" — so 6 and 33 are two operands, not two guesses
   at one number, and :meth:`RiscvCostState._muldiv_occupancy` already has the
   operands in hand (it reads the divisor, and the dividend for the ``INT_MIN``
   case). What it does *not* have is the function: no document in either tree
   relates a dividend to a cycle count anywhere between the endpoints, so any
   per-magnitude charge would be an invented curve wearing a citation, which is
   the failure the "WHY THERE IS NO ``measured`` PROVENANCE" block in
   ``tensix_instruction_costs.yaml`` exists to prevent. Silicon reads **33.001**
   for ``divu 0x12345678, 3`` — a 29-significant-bit dividend, at the top of the
   magnitude range — so the floor is loose by 5.5x *at that operand*. The
   exposure is small and was measured rather than assumed: across the in-tree
   Blackhole replay guards a whole kernel launch executes **0-2 divides in
   40,000-80,000 instructions**, with 9-to-12-bit dividends, three orders of
   magnitude below the benchmark's. 27 under-charged cycles twice per launch is
   under 0.15 % of a launch even at the worst-case operand, and these operands
   are not that.

   **Re-checked on 2026-08-06, when the other two silicon-backed fixes landed:
   the decision stands because the function is still unpublished.** Both
   BabyRISCV pages were searched whole for an iteration rule — bits per cycle,
   a radix, a leading-zeros term, any formula or table relating an operand to
   a count — and neither carries one; ttsim is functional-only and has no
   timing to borrow. The single silicon point does not even pin the obvious
   one-bit-per-cycle family: ``cycles = significant_bits + 4`` fits 33.001 at
   29 bits but gives 36 at a full 32-bit dividend, past the documented cap of
   33 — so the simplest candidate curve contradicts the band it would be
   fitted inside, and choosing a different family is exactly the free
   parameter one point cannot pin. The floor stays, and the point stays
   recorded as corroboration on the YAML entry rather than becoming a charge.

**What is deliberately not modelled** — each an honest gap rather than an
omission:

* **Branch mispredicts.** Sourced (a 2-cycle bubble on Wormhole, 4 on
  Blackhole) but uncountable: neither the docs nor tt-sim describe the
  predictor, so the number of mispredictions is unknowable and charging every
  taken branch would be a fabrication.
* **Instruction fetch / i-cache misses.** The table gives the fetch *period*
  (one 128-bit L1 read per four instructions) but no miss cost.
* **Per-region request throughput.** "Each memory region can process at most
  one request per cycle" for every region but L1
  (``BabyRISCV/MemoryOrdering.md``, both arches). It is the one published term
  that would make *two* cores hammering one NIU cost more than one core doing
  it, and it is not charged: the state it needs is per tile and per region,
  shared by five cores, where every other RV cost in this module is per core.
  Measured before it was declined — see ``docs/plans/cost-model.md``.
* **The instruction queue in the Load/Store Unit** (Wormhole: "up to eight
  instructions ... the oldest non-retired load, plus (up to) the next seven";
  Blackhole states the same 8 as a retire-order queue). A third view of the
  same 8 that the load-use interlock's ``N - 1`` already spends on every row
  the tables publish — at ``N <= 8`` a load leaves before the queue behind it
  can fill — so it can only bite on the ``>= 12`` atomic row, which no
  in-tree kernel reaches.
* **Regions the table does not name** — see
  :data:`tt_sim.perf.model.RV_UNNAMED_REGIONS`. The NoC NIU register block
  used to head that list and no longer does; see below.

Every one of those under-charges, which is the direction the cost model's
stated policy asks for: a modelled cycle count is a floor.

**The NIU register block is charged, and the gap that said otherwise was a
misreading of the table, not a missing number** (2026-08-06). The ROADMAP item
"cost the NIU register block" recorded the busiest MMIO load in the tree —
every ``noc_async_*_barrier`` polls a NIU counter at ``0xFFB20000`` /
``0xFFB30000`` — as blocked on
provenance, on the grounds that the ">= 7" row "covers the NoC *overlay* at
``0xFFB40000``, a different block". The row does cover the overlay. It also
covers the NIUs, by name, on both architectures, in the same cell:

    TDMA-RISC configuration and command / Tile control / debug / status /
    PIC configuration and status / **NoC 0 configuration and command** /
    **NoC 1 configuration and command** / NoC overlay configuration and
    command  ->  ">= 7 (more in the case of access conflicts)"

(Wormhole ``BabyRISCV/README.md``; Blackhole's is identical bar TDMA, which
moves to its ">= 4" row and which :data:`_LOAD_LATENCY_KEYS` already splits.)
Each of "NoC 0 configuration and command" and "NoC 1 configuration and
command" is its own line of that cell and both link to ``NoC/MemoryMap.md``,
the NIU register block, while the overlay's line links to
``NoC/Overlay/README.md``. So nothing had to be sourced, derived or measured:
the number was already in the table this module reads, under a *key name* —
``tdma_tilectrl_pic_noc_overlay`` — whose "noc_overlay" was read as one word.
The keys are now ``..._noc0_noc1_overlay`` on both arches so the same
misreading cannot recur.

Two address ranges were reclassified, both to the region that row supplies:

* ``0xFFB20000-0xFFB3FFFF``, the NoC 0 and NoC 1 NIU register blocks.
* ``0xFFB50000-0xFFB7FFFF``, the upper three quarters of the NoC overlay.
  ``NOC_OVERLAY_START_ADDR`` is ``0xFFB4_0000`` to ``0xFFB7_FFFF`` (64 stream
  register spaces of 4 KiB, which is also exactly what ``NoCOverlay`` models
  and what ``TensixTile`` maps), and this module's range ended at
  ``0xFFB50000`` — one stream space — so overlay streams 16-63 were counted
  as unnamed and charged nothing. That was a plain arithmetic slip in the
  same classifier, found by the same census, and it is the larger of the two
  on a matmul.
"""

from __future__ import annotations

from tt_sim.perf.model import (
    RV_REGION_COUNT,
    RV_REGION_L1,
    RV_REGION_LOCAL_DATA_RAM,
    RV_REGION_MAILBOX_GROUP,
    RV_REGION_TDMA,
    RV_REGION_TENSIX_GPR_CFG,
    RV_REGION_TILECTRL_PIC_NOC,
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
# Tile control / debug / status is 0xFFB1_2000-0xFFB1_2FFF and the PIC
# configuration and status registers are 0xFFB1_3000-0xFFB1_3FFF. The two are
# one row of the load-latency table, so they are one region here; tt-sim maps
# no PIC today, but classifying by the published map rather than by what
# happens to be modelled is what stops the next block from being missed.
_TILE_CTRL_END = 0xFFB14000
# NoC 0 (0xFFB2_0000), NoC 1 (0xFFB3_0000) and the NoC overlay
# (0xFFB4_0000-0xFFB7_FFFF) are three *consecutive* entries of the memory map
# and all three are named by the same ">= 7" row, so one range covers them.
_NOC_REGS_BASE = 0xFFB20000
_NOC_OVERLAY_END = 0xFFB80000
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
        return RV_REGION_TILECTRL_PIC_NOC
    if _NOC_REGS_BASE <= addr < _NOC_OVERLAY_END:
        return RV_REGION_TILECTRL_PIC_NOC
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
#: No free slot in the load queue — the sustained-load rate, which is a
#: property of the *stream* rather than of any one load's dependants.
STALL_LOAD_RATE = 3
STALL_REASON_COUNT = 4
STALL_REASON_NAMES = ("load_use", "store_rate", "integer_unit", "load_rate")

_LOAD_OPCODE = 0x03
_STORE_OPCODE = 0x23
_OP_OPCODE = 0x33
_FENCE_OPCODE = 0x0F  # MISC-MEM: fence flushes the whole L0 data cache
_AMO_OPCODE = 0x2F  # Zaamo atomics: "any fence or atomic instruction"
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
        "multiply_latency",
        "divide_general",
        "divide_trivial",
        "divide_int_min",
        "l1_miss_latency",
        "load_slot_cycles",
        "l1_miss_slot_cycles",
        "_load_slots",
        "_l0_enabled",
        "_l0_shift",
        "_l0_tags",
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
        "l0_hits",
        "l0_misses",
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
        # ``None`` on Wormhole, whose multiply blocks rather than pipelines;
        # 2 on Blackhole ("exactly one cycle in EX1, and then exactly one
        # cycle in EX2"), spent as a scoreboard entry on the result register.
        self.multiply_latency = model.multiply_latency
        self.divide_general = model.divide_general
        self.divide_trivial = model.divide_trivial
        self.divide_int_min = model.divide_int_min
        # The sustained-load rate: ``load_slots`` slots, each held for the
        # region's ``N - 1``. A ``None`` window means the row is below the
        # docs' five-cycle threshold (one load per cycle, which the core
        # cannot beat) or is a region the table does not name; an empty slot
        # list means the entry is not sourced and nothing is charged.
        self.load_slot_cycles = model.load_slot_cycles
        self.l1_miss_slot_cycles = model.l1_miss_slot_cycles
        # Sorted ascending, so slot 0 is always the next to free and the
        # check is one comparison.
        self._load_slots = [0] * (model.load_slots or 0)
        # The L0 data-cache line model — Blackhole only, by data: the tables
        # publish the geometry (4 lines of 16 bytes) and an L1 hit/miss
        # latency pair there and nothing of the kind on Wormhole. Tags only,
        # no data; -1 is "invalid" (no L1 byte address shifts to it).
        self.l1_miss_latency = model.l1_load_miss_latency
        self._l0_enabled = bool(
            self.l1_miss_latency is not None
            and model.l0_lines
            and model.l0_line_bytes
            and model.l0_line_bytes & (model.l0_line_bytes - 1) == 0
        )
        self._l0_shift = model.l0_line_bytes.bit_length() - 1 if self._l0_enabled else 0
        self._l0_tags = [-1] * model.l0_lines if self._l0_enabled else []
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
        self.l0_hits = 0
        self.l0_misses = 0
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
        slots = self._load_slots
        for i in range(len(slots)):
            slots[i] = 0
        tags = self._l0_tags
        for i in range(len(tags)):
            tags[i] = -1

    # -- firmware-loop parking ---------------------------------------------
    #
    # Three methods that let ``tt_sim/pe/rv/spin.py`` treat this scoreboard as
    # part of the state its fixed-point proof covers, without knowing which of
    # these fields are cycle numbers. That knowledge belongs here, next to the
    # fields, which is the whole reason the split is drawn this way.
    #
    # The normal form is **cycle-relative**: every absolute-cycle field is
    # reported as ``max(field - cycle, 0)``, i.e. "how many cycles into the
    # future". Two facts make that lossless for anything the model can go on to
    # do. First, every one of those fields is only ever read through a strict
    # "is it still in the future" comparison against the current cycle
    # (``cycle_num < self._stall_until``, ``at > cycle_num``,
    # ``cycle_num < self._store_ready``), so any two values at or below the
    # current cycle are indistinguishable from here on. Second, ``cycle_num``
    # never decreases within a run — ``RiscvCostState.reset`` is the only thing
    # that rewinds, and it zeroes everything. So a state restored from a
    # signature drives the interpreter identically to the state that produced
    # it, one cycle-shift later.

    def spin_signature(self, cycle_num):
        """This scoreboard's cycle-relative normal form at ``cycle_num``.

        Everything ``can_issue`` will consult, and nothing else: the per-GPR
        ready times, the issue and store-rate deadlines, the open coalescing
        group (only while its drain is still live — once ``_store_ready`` is in
        the past the group is unreachable), the load-queue slots (cycle-valued
        like the rest, and order-preserving under the clamp because a sorted
        list stays sorted when its past entries collapse onto 0), and the L0
        line tags, which are not cycle-valued but are state a loop's charges
        depend on.
        """
        ready = self._ready
        rel_ready = tuple(r - cycle_num if r > cycle_num else 0 for r in ready)
        stall = self._stall_until - cycle_num if self._stall_until > cycle_num else 0
        store = self._store_ready - cycle_num if self._store_ready > cycle_num else 0
        group = (self._group_block, self._group_lo, self._group_hi) if store else None
        return (
            rel_ready,
            stall,
            self._stall_reason,
            store,
            group,
            tuple(self._l0_tags),
            tuple(s - cycle_num if s > cycle_num else 0 for s in self._load_slots),
        )

    def spin_restore(self, signature, cycle_num):
        """Re-base ``signature`` onto ``cycle_num`` — the time translation.

        Called on the first tick after the pump skipped a span while the core
        was parked, with the signature recorded for the trajectory phase the
        unparked run would have reached.
        """
        rel_ready, stall, reason, store, group, tags, slots = signature
        ready = self._ready
        for i in range(32):
            rel = rel_ready[i]
            ready[i] = cycle_num + rel if rel else 0
        self._stall_until = cycle_num + stall if stall else 0
        self._stall_reason = reason
        self._store_ready = cycle_num + store if store else 0
        if group is None:
            self._group_block = -1
        else:
            self._group_block, self._group_lo, self._group_hi = group
        self._l0_tags[:] = tags
        self._load_slots[:] = [cycle_num + s if s else 0 for s in slots]

    def spin_counters(self):
        """The accumulators, flat, in the layout :meth:`spin_add_counters`
        expects. Not part of the signature — a counter cannot change what the
        model does — but a parked span still has to be *charged* for the
        iterations it skipped, or the §I report would under-count them."""
        return (
            self.stall_cycles,
            self.l1_stores,
            self.l0_hits,
            self.l0_misses,
            self.pending_stall,
            *self.stall_by_reason,
            *self.loads_by_region,
        )

    def spin_add_counters(self, delta):
        """Add a :meth:`spin_counters` difference — the charges of the loop
        iterations the pump skipped."""
        self.stall_cycles += delta[0]
        self.l1_stores += delta[1]
        self.l0_hits += delta[2]
        self.l0_misses += delta[3]
        self.pending_stall += delta[4]
        by_reason = self.stall_by_reason
        for i in range(STALL_REASON_COUNT):
            by_reason[i] += delta[5 + i]
        by_region = self.loads_by_region
        base = 5 + STALL_REASON_COUNT
        for i in range(RV_REGION_COUNT):
            by_region[i] += delta[base + i]

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
            "l0_hits": self.l0_hits,
            "l0_misses": self.l0_misses,
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
            tag = -1
            line = -2  # "no L0 lookup was made", as against -1 for a miss
            if region == RV_REGION_L1 and self._l0_enabled:
                tag = addr >> self._l0_shift
                line = self._l0_probe(tag)
                if line < 0:
                    latency = self.l1_miss_latency
                    window = self.l1_miss_slot_cycles
                else:
                    latency = self.load_latency[RV_REGION_L1]
                    window = self.load_slot_cycles[RV_REGION_L1]
            else:
                latency = self.load_latency[region]
                window = self.load_slot_cycles[region]
            if window is not None:
                # The sustained-load rate. Slot 0 is the next to free, so a
                # full queue is one comparison; taking a slot is a write and a
                # four-element sort. Nothing above this point has mutated
                # anything, which is what lets the stall be re-offered.
                slots = self._load_slots
                free = slots[0]
                if free > cycle_num:
                    return self._stall(free, STALL_LOAD_RATE)
                slots[0] = cycle_num + window
                slots.sort()
            self.loads_by_region[region] += 1
            if line != -2:
                self._l0_commit(tag, line)
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

        if (opcode == _FENCE_OPCODE or opcode == _AMO_OPCODE) and self._l0_enabled:
            # "The entire L0 data cache will be flushed by any fence or atomic
            # instruction." A documented flush, so the misses it causes are a
            # sourced charge rather than an invented one.
            tags = self._l0_tags
            for i in range(len(tags)):
                tags[i] = -1
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
            elif self.multiply_latency is not None and not (instr & 0x4000):
                # A pipelined multiply (Blackhole: one cycle in EX1, one in
                # EX2) does not block the unit; it makes its *result* late,
                # exactly like a load. The scoreboard entry is what a
                # dependent chain pays — silicon's rv_mul_dep at 1.985 —
                # while independent instructions after it stay free.
                rd = (instr >> 7) & 0x1F
                if rd:
                    ready[rd] = cycle_num + self.multiply_latency
        return True

    def _l0_probe(self, tag):
        """Which line ``tag`` is resident in, or ``-1`` for a miss. Read-only.

        The tag array is ordered most-recently-loaded first, so index 0 is the
        streak case (successive loads off one line) and costs one comparison.
        Split from :meth:`_l0_commit` because a load can still be refused
        after its latency is known — the sustained-load rate is checked
        against that latency — and ``can_issue``'s contract is that a stalled
        cycle leaves the model exactly as it found it. A probe that evicted a
        line and was then re-offered next cycle would evict a second one.
        """
        tags = self._l0_tags
        if tag == tags[0]:
            return 0
        return tags.index(tag) if tag in tags else -1

    def _l0_commit(self, tag, line):
        """Install ``tag`` at the head, given :meth:`_l0_probe`'s answer.

        Replacement is least-recently-loaded because the page publishes no
        policy and this is the generous choice: with 4 lines it never misses
        on a working set that fits and always misses on a cyclic walk over
        more than 4 lines — the two regimes the silicon probes pin — and any
        stricter organisation could only miss more, which keeps the modelled
        count a floor. The tag is the *start* address's line; a load that
        straddles two lines is charged as one, the cheaper reading.
        """
        if line == 0:
            self.l0_hits += 1
            return
        tags = self._l0_tags
        if line > 0:
            del tags[line]
            self.l0_hits += 1
        else:
            tags.pop()
            self.l0_misses += 1
        tags.insert(0, tag)

    def _l0_store_invalidate(self, addr):
        """ "Stores to L1 flush the containing line, so the cache is never
        dirty" — the next load of a stored-to line is a documented miss.
        Called only once the store actually issues, so a stalled cycle leaves
        the tags exactly as it found them (``can_issue``'s contract)."""
        if self._l0_enabled:
            tag = addr >> self._l0_shift
            tags = self._l0_tags
            if tag in tags:
                tags[tags.index(tag)] = -1

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
                    self._l0_store_invalidate(addr)
                    return True
            if cycle_num < self._store_ready:
                return self._stall(self._store_ready, STALL_STORE_RATE)
            self._group_block = block
            self._group_lo = self._group_hi = addr
        elif cycle_num < self._store_ready:
            return self._stall(self._store_ready, STALL_STORE_RATE)
        self._store_ready = cycle_num + self.l1_store_period
        self.l1_stores += 1
        self._l0_store_invalidate(addr)
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
