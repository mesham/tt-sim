"""Typed events published on the simulator's event bus.

Every event subclass fixes its CATEGORY as a ClassVar so the bus can
route purely by ``type(event)``, no per-instance discriminator field.
SCHEMA_VERSION is bumped on any breaking change to event shape; writers
should refuse to emit unknown versions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class EventCategory(Enum):
    INSTR = "instr"
    MEM = "mem"
    NOC = "noc"
    COMPUTE = "compute"
    SYNC = "sync"
    DISPATCH = "dispatch"
    LIFECYCLE = "lifecycle"
    COUNTER = "counter"
    STALL = "stall"


class Unit(Enum):
    BRISC = "BRISC"
    NCRISC = "NCRISC"
    TRISC0 = "TRISC0"
    TRISC1 = "TRISC1"
    TRISC2 = "TRISC2"
    FPU = "FPU"
    SFPU = "SFPU"
    PACKER = "PACKER"
    UNPACKER = "UNPACKER"
    MATRIX = "MATRIX"
    MOVER = "MOVER"
    THCON = "THCON"
    SYNC = "SYNC"
    TDMA = "TDMA"
    CFG = "CFG"
    MISC = "MISC"
    MAILBOX = "MAILBOX"
    TTSYNC = "TTSYNC"
    NOC0 = "NOC0"
    NOC1 = "NOC1"
    HOST = "HOST"
    UNKNOWN = "UNKNOWN"


#: The same Tensix backend unit is named three different ways in the trace,
#: and the three vocabularies are **not** interchangeable -- ``SFPU`` is
#: ``Vector`` is ``SFPU``, but ``MATRIX`` is ``Matrix`` is ``MATH``. Joining
#: two datasets on the wrong one silently returns nothing.
#:
#: ``{Unit enum value: (ComputeEvent.target_unit, DispatchEvent.target_unit)}``
#: where the second element is also :attr:`StallEvent.blocked_on`. The keys
#: are what appears in ``unit_id[3]`` and in the Parquet ``unit`` column, and
#: are the canonical join key; the aliases exist because the middle column is
#: the backend class's own ``unit_name`` and the right-hand one is the ISA's
#: ``ex_resource``, which the instruction tables are keyed by.
#:
#: ``ex_resource`` has one value with no unit behind it, ``NONE`` -- an
#: instruction the wait gate accounts for but dispatches to no backend.
BACKEND_UNIT_ALIASES = {
    "MATRIX": ("Matrix", "MATH"),
    "SFPU": ("Vector", "SFPU"),
    "PACKER": ("Packer", "PACK"),
    "UNPACKER": ("Unpacker", "UNPACK"),
    "THCON": ("Scalar", "THCON"),
    "CFG": ("Config", "CFG"),
    "SYNC": ("Sync", "SYNC"),
    "TDMA": ("Misc", "TDMA"),
    "MOVER": ("Mover", "XMOV"),
}


#: Every reason a :class:`StallEvent` may carry, frozen so a consumer can
#: switch on the set exhaustively and a typo cannot silently mint a new one.
#:
#: Each name is a cause the model **actually knows** at the moment it declines
#: to make progress -- there is deliberately no generic ``"stalled"``. A code
#: generator acts on the reason, so a reason that does not name a mechanism is
#: worse than no reason at all.
#:
#: The three ``issue_*`` reasons are back-pressure from the target backend unit
#: (:meth:`tt_sim.pe.tensix.backends.backend_base.TensixBackendUnit.issueInstruction`);
#: the rest are the Tensix wait gate declining to release its head instruction.
STALL_REASONS = frozenset(
    {
        # -- wait gate, latched SEMWAIT --------------------------------------
        #: A selected semaphore is still zero: the producer has not posted yet.
        "semaphore_empty",
        #: A selected semaphore is at its maximum: consumer back-pressure, the
        #: producer cannot post again until something drains.
        "semaphore_full",
        # -- wait gate, latched STALLWAIT ------------------------------------
        #: A STALLWAIT condition is unmet. ``blocked_on`` names the unit the
        #: condition is about, which is the actionable half.
        "resource_wait",
        # -- wait gate, other ------------------------------------------------
        #: An ATGETM has been accepted but the sync unit has not granted the
        #: mutex yet.
        "mutex_wait",
        #: A backend unit asserted the gate's stall directly.
        "backend_enforced_stall",
        #: A Matrix-unit instruction cannot start because the SrcA/SrcB bank it
        #: reads is still owned by the unpackers. ``blocked_on`` is ``UNPACK``.
        "src_reserved_by_unpacker",
        #: The mirror image: an unpacker (or a ThCon GPR-to-Src write) cannot
        #: start because the Src bank it *writes* is still owned by the matrix
        #: unit. ``blocked_on`` is ``MATH``. Together with the reason above,
        #: these two are the Src ping-pong a code generator is trying to
        #: overlap, and which way round it is stuck is the actionable part.
        "src_reserved_by_matrix",
        # -- backend unit servicing something else ----------------------------
        #: The scalar unit is mid-``FLUSHDMA``, waiting for the DMA units its
        #: condition mask names to drain.
        "flush_pending",
        #: The scalar unit is retrying an atomic (``ATCAS``) that has not yet
        #: observed its expected value.
        "atomic_pending",
        # -- backend issue refusal -------------------------------------------
        #: The target unit (or its IPC group) is still occupied and cannot take
        #: another instruction. Almost always cost-model state -- an occupancy
        #: is what ``busy_until`` arms, and nothing arms one with
        #: TT_SIM_COST_MODEL unset. The one exception is real and measured: the
        #: config unit's own single-issue throughput rule (a SETC16/WRCFG in
        #: the previous cycle blocks this cycle's other-opcode issue,
        #: ``backends/config.py``) is structural, not an occupancy, and refuses
        #: under this reason in **both** regimes. So a small
        #: ``tensix_stall_unit_busy`` on an un-modelled run is correct, and it
        #: is always blamed on ``CFG``.
        "unit_busy",
        #: The target unit's issue slot was already taken this cycle, by another
        #: thread or by this one's own earlier instruction.
        "issue_slot_taken",
        #: The slot was free but yielded to a thread granted less recently, so
        #: the grant rotates rather than following wait-gate tick order.
        "issue_yield_fairness",
        # -- whole-thread issue interlock -------------------------------------
        #: A documented interlock is holding *everything* this thread has,
        #: whichever unit the held instruction is bound for -- as opposed to
        #: ``unit_busy``, which only refuses the unit the instruction was
        #: offered to. Two units publish one: the Scalar Unit, for the whole of
        #: an instruction's execution ("it cannot start executing its next
        #: instruction ... regardless of which unit that next instruction
        #: executes in"), and an unpacker, for an ``UNPACR``'s address phase
        #: ("For the duration of these cycles, the issuing thread cannot start
        #: its next instruction") -- so ``blocked_on`` is ``THCON`` or
        #: ``UNPACK``. Cost-model state: the deadline is armed from a modelled
        #: occupancy, so this never fires with ``TT_SIM_COST_MODEL`` unset.
        "thread_issue_block",
    }
)


@dataclass(frozen=True, slots=True)
class Event:
    SCHEMA_VERSION: ClassVar[int] = 4
    CATEGORY: ClassVar[EventCategory | None] = None
    cycle: int
    unit_id: tuple


@dataclass(frozen=True, slots=True)
class InstrEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.INSTR
    pc: int
    instruction: int
    stalled: bool = False
    # Destination register write captured at retirement.
    # reg_write_idx == -1 means no architectural register was written
    # (stores, branches, jumps without link). x0 writes are not
    # recorded (Spike's commitlog format excludes them).
    reg_write_idx: int = -1
    reg_write_value: int = 0
    # How many cycles this core was held before this instruction could
    # issue, and which of ``tt_sim.pe.rv.cost.STALL_REASON_NAMES`` held
    # it. Both are cost-model state: without ``TT_SIM_COST_MODEL`` no
    # RV instruction can stall at all, so ``0`` / ``""`` is the truthful
    # reading of an un-modelled run, not a missing measurement. Distinct
    # from ``stalled``, which is the Tensix-instruction-buffer back-
    # pressure that exists in both regimes.
    stall_cycles: int = 0
    stall_reason: str = ""


@dataclass(frozen=True, slots=True)
class DispatchEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.DISPATCH
    opcode: str
    target_unit: str
    thread_id: int


@dataclass(frozen=True, slots=True)
class NoCEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.NOC
    phase: str
    txn_type: str
    src: tuple
    dst: tuple
    size_bytes: int = 0
    txn_id: int = 0
    #: Cycle the sending NIU put this packet on the wire; ``cycle`` is the
    #: cycle the receiving NIU serviced it, so ``cycle - issue_cycle`` is the
    #: flight time. Both are real measurements in either regime — with the
    #: cost model off the flight is simply always the one cycle the two-list
    #: swap in ``NUI.clock_tick`` costs. ``-1`` means the flight could not be
    #: timed: a NIU with no owning tile clock (unit tests, ``driver/simple``).
    issue_cycle: int = -1


@dataclass(frozen=True, slots=True)
class LifecycleEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.LIFECYCLE
    kind: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class MemEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.MEM
    op: str  # "read" or "write"
    address: int
    size: int
    region: str = ""  # "L1", "DRAM", "MMIO", etc. — coarse classifier
    # PC of the instruction that triggered the access, if known via the
    # MemorySpace caller_context. 0 when unattributed (NoC-driven or
    # internal accesses without a backing RV instruction).
    pc: int = 0


@dataclass(frozen=True, slots=True)
class ComputeEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.COMPUTE
    op: str  # opcode name (ELWADD, SFPCONFIG, PACK, UNPACK, ...)
    target_unit: str  # FPU / SFPU / PACKER / UNPACKER / MATRIX / MOVER / THCON
    thread_id: int = -1  # -1 when not thread-attributed
    detail: str = ""
    #: Cycles this op occupies its backend unit, straight from
    #: ``tensix_instruction_costs.yaml`` via
    #: ``TensixBackendUnit.instruction_occupancy``. ``0`` means *not modelled*
    #: — either ``TT_SIM_COST_MODEL`` is unset (nothing occupies anything) or
    #: the unit/opcode is one of the ones the tables deliberately leave
    #: uncosted. A consumer must not read 0 as "one cycle"; it is "no claim".
    duration: int = 0


@dataclass(frozen=True, slots=True)
class SyncEvent(Event):
    CATEGORY: ClassVar[EventCategory] = EventCategory.SYNC
    kind: str  # "mailbox_send" / "mailbox_recv" / "ttsync_signal" / "ttsync_wait"
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CounterSnapshot(Event):
    """Periodic performance-counter sample, emitted by CounterAggregator.

    Long-format row — one CounterSnapshot per (unit, counter_name) at
    each flush boundary. Parquet writer materialises columns from
    unit_id (chip/core_y/core_x/unit) for SQL convenience.
    """

    CATEGORY: ClassVar[EventCategory] = EventCategory.COUNTER
    counter_name: str
    value: int
    kernel_id: int = 0


@dataclass(frozen=True, slots=True)
class StallEvent(Event):
    """One contiguous span during which a Tensix thread made no progress.

    The complement of :class:`DispatchEvent`: that event says an instruction
    issued, this one says why the next one could not. ``unit_id`` is the
    **thread that suffered** the stall (TRISC0/1/2), ``blocked_on`` names the
    unit **responsible** for it, and ``reason`` says which mechanism held it.
    That split is the whole point -- a code generator needs "the math thread
    spent 61 k cycles waiting on the unpackers", not "something stalled".

    **Coalesced into episodes, not emitted per cycle.** The wait gate re-offers
    its head instruction every cycle while blocked, so a per-cycle event would
    put 64 k rows on one guard for one wait. Instead an episode accumulates
    while ``(reason, blocked_on, opcode)`` is unchanged and is published once,
    when it ends: ``cycle`` is the first cycle of the span and :attr:`cycles`
    its length. A thread is in at most one episode at a time (the gate's stall
    branches are mutually exclusive), so episodes on one thread never overlap.

    **These are modelled floors, not calibrated cycle counts.** The span is as
    long as the simulator held the thread, and the simulator charges every
    published bound at its low end (``docs/plans/cost-model.md``); it is
    corroborated against silicon but never calibrated to it. Read the
    *attribution* -- which reason dominates, and which unit -- rather than the
    absolute figure. Every reason is structural and occurs in both timing
    regimes; ``unit_busy`` is the one that mostly does not, because an
    occupancy is what the cost model arms -- but see its note in
    :data:`STALL_REASONS` for the config unit's structural exception, which
    does fire with ``TT_SIM_COST_MODEL`` unset.
    """

    CATEGORY: ClassVar[EventCategory] = EventCategory.STALL
    #: One of :data:`STALL_REASONS`.
    reason: str
    #: The backend unit responsible, as an ``ex_resource`` name (``MATH``,
    #: ``SFPU``, ``PACK``, ``UNPACK``, ``XMOV``, ``THCON``, ``CFG``, ``SYNC``).
    #: Empty when the mechanism names no single unit -- a semaphore or mutex
    #: wait is about a synchronisation object, not about a unit.
    blocked_on: str = ""
    #: Cycles in this episode. Always >= 1.
    cycles: int = 1
    #: The instruction held at the head of the wait gate, when the stall is
    #: attributable to one. Empty for a latched wait, which blocks a *class* of
    #: instructions rather than the one at the head.
    opcode: str = ""
    #: Issuing Tensix thread (0-2), mirroring :class:`DispatchEvent`.
    thread_id: int = -1
    #: For a semaphore wait, the semaphore index the thread is blocked on;
    #: ``-1`` for every other reason.
    semaphore: int = -1
