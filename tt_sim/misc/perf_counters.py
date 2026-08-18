"""The Tensix hardware performance counters (``RISCV_DEBUG_REG_PERF_CNT_*``).

Why this exists
---------------

A kernel built with ``TT_METAL_PROFILE_PERF_COUNTERS=<bitmask>`` programs these
registers, runs, and reads the counters back. Before this module tt-sim mapped
the whole ``RISCV_DEBUG_REG`` window onto a permissive generic store, so every
one of those reads returned **zero** — which decodes as a perfectly plausible
"nothing ever stalled". That is the worst failure mode available: wrong, and
quiet. The registers now either report a quantity tt-sim genuinely tracks, or
say out loud that they do not.

What the hardware is
--------------------

Five banks — ``FPU``, ``TDMA_UNPACK``, ``TDMA_PACK``, ``INSTRN_THREAD``, ``L1``
— each built from a reusable RTL module (``tt_perf_cnt``) which, per event,
provides ``req_cnt`` (cycles the event signal was high), ``grant_cnt`` (cycles
the grant/ready signal was high) and ``ref_cnt`` (total elapsed cycles). Each
bank is programmed through three consecutive debug registers and read back
through one ``_OUT_L``/``_OUT_H`` pair:

===============================  ==========================================
``PERF_CNT_<X>0``                reference period in cycles
``PERF_CNT_<X>1`` bits [7:0]     mode: 0 continuous, 1 count-to-refclk, 2
                                 continuous without refclk maintenance
``PERF_CNT_<X>1`` bits [15:8]    counter select within the bank
``PERF_CNT_<X>1`` bit  [16]      output format: 0 = ``req_cnt``, 1 =
                                 ``grant_cnt`` on ``_OUT_H``
``PERF_CNT_<X>2`` bit  [0]       start (rising edge; 0->1 also clears)
``PERF_CNT_<X>2`` bit  [1]       stop (rising edge)
``PERF_CNT_OUT_L_<X>``           ``ref_cnt``
``PERF_CNT_OUT_H_<X>``           ``req_cnt`` or ``grant_cnt`` per bit [16]
===============================  ==========================================

(The vendor tech report writes the select field as bits [12:8]; the driver in
``tt_metal/tools/profiler/perf_counters.hpp`` writes ``counter_sel << 8`` with
selects up to 57, and encodes the grant side as ``sel >= 256`` so that the
shift lands on bit 16. Eight bits is the only decode consistent with the code
that actually drives it, so that is what is modelled.)

Provenance
----------

**``vendor_source``, and deliberately inert.** The counter semantics come from
``tech_reports/PerfCounters/perf-counters.md`` plus the RTL it cites, not from
the ISA documentation — there is no ``PerformanceCounters.md`` in either
architecture tree. That is fine for a functional register model and
**disqualifying as provenance for a cycle charge**: nothing in this module
feeds the cost model, arms an occupancy or delays an instruction. It reads
state the simulator already keeps for its own reasons. No counter value may
ever become a cost.

What is sourced, and what is not
--------------------------------

Only the ``INSTRN_THREAD`` bank is sourced, because that is the bank whose
quantities tt-sim already tracks (``tt_sim/trace/events.py``'s
``STALL_REASONS`` vocabulary, materialised per cycle by the wait gate). Every
other bank's counters are *registered by name* and read back as zero with a
one-shot warning naming the counter — the point being that an unmodelled
counter must announce itself rather than pass for a measurement.

The mapping, and the three families that were declined rather than forced, are
in :data:`INSTRN_SOURCED` and :data:`INSTRN_DECLINED`.
"""

import os
import sys

#: ``(control-register-triplet base, OUT_L offset, OUT_H offset)`` per bank,
#: as offsets within the ``RISCV_DEBUG_REGS_START_ADDR`` (``0xFFB12000``)
#: window. Identical on Wormhole and Blackhole.
BANK_REGISTERS = {
    "INSTRN_THREAD": (0x000, 0x100, 0x104),
    "TDMA_UNPACK": (0x00C, 0x108, 0x10C),
    "FPU": (0x018, 0x120, 0x124),
    "L1": (0x030, 0x118, 0x11C),
    "TDMA_PACK": (0x0F0, 0x110, 0x114),
}

#: ``RISCV_DEBUG_REG_PERF_CNT_ALL`` — a broadcast start/stop across banks.
PERF_CNT_ALL = 0x03C
#: ``RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL`` — selects which L1 bank (0-4 on
#: Blackhole, 0-1 on Wormhole) the single L1 counter bank observes.
PERF_CNT_MUX_CTRL = 0x218

#: Every offset this block owns, for :mod:`tt_sim.misc.tile_ctrl` to delegate.
PERF_CNT_OFFSETS = frozenset(
    [PERF_CNT_ALL, PERF_CNT_MUX_CTRL]
    + [base + 4 * i for base, _, _ in BANK_REGISTERS.values() for i in range(3)]
    + [out_l for _, out_l, _ in BANK_REGISTERS.values()]
    + [out_h for _, _, out_h in BANK_REGISTERS.values()]
)

_START_BIT = 0x1
_STOP_BIT = 0x2

#: The nine per-thread stall-reason counters, in the order the RTL generates
#: them; a thread's block starts at ``_INSTRN_PER_THREAD_BASE`` and is
#: thread-major (``base + 9 * thread + index``).
_PER_THREAD_REASONS = (
    "WAITING_FOR_THCON_IDLE",
    "WAITING_FOR_UNPACK_IDLE",
    "WAITING_FOR_PACK_IDLE",
    "WAITING_FOR_MATH_IDLE",
    "WAITING_FOR_NONZERO_SEM",
    "WAITING_FOR_NONFULL_SEM",
    "WAITING_FOR_MOVE_IDLE",
    "WAITING_FOR_MMIO_IDLE",
    "WAITING_FOR_SFPU_IDLE",
)

#: The four Src-bank conditions, which are shared across threads rather than
#: replicated per thread. Blackhole gives them one select each (27-30);
#: Wormhole replicates each across three selects (27-38).
_SRC_CONDITIONS = (
    "WAITING_FOR_SRCA_CLEAR",
    "WAITING_FOR_SRCB_CLEAR",
    "WAITING_FOR_SRCA_VALID",
    "WAITING_FOR_SRCB_VALID",
)

#: The instruction-type availability counters at selects 0-23, three threads
#: each. Selects 9-11 are a documented gap (the XSEARCH kick is tied to 0).
_AVAILABILITY_UNITS = (
    "CFG",
    "SYNC",
    "THCON",
    "XSEARCH",
    "MOVE",
    "FPU",
    "UNPACK",
    "PACK",
)

#: Counter names the INSTRN bank reports from a quantity tt-sim tracks, with
#: the tech report's own description of each. Keep this table and
#: :meth:`TensixPerfCounters._instrn_value` in step.
INSTRN_SOURCED = {
    "THREAD_STALLS": (
        "cycles this Tensix thread was held at the wait gate, every reason "
        "summed -- the report's \"fraction of cycles each instruction thread "
        'was stalled"'
    ),
    "WAITING_FOR_NONZERO_SEM": (
        "cycles held on a selected semaphore still reading zero -- the "
        'report\'s "waiting for a producer to signal (semaphore is 0)", '
        "tt-sim's ``semaphore_empty``"
    ),
    "WAITING_FOR_NONFULL_SEM": (
        "cycles held on a selected semaphore at its maximum -- the report's "
        '"waiting for a consumer to drain (semaphore is at max)", tt-sim\'s '
        "``semaphore_full``"
    ),
    "WAITING_FOR_SRCA_VALID": (
        "cycles a Matrix Unit instruction was held because SrcA is still "
        "owned by the unpackers -- the report's \"waiting for source register "
        "data to become valid (unpacker hasn't filled it yet)\", tt-sim's "
        "``src_reserved_by_unpacker``"
    ),
    "WAITING_FOR_SRCB_VALID": ("as WAITING_FOR_SRCA_VALID, for the SrcB bank"),
    "WAITING_FOR_SRCA_CLEAR": (
        "cycles an unpacker (or a ThCon GPR-to-Src write) was held because "
        "the SrcA bank it writes is still owned by the Matrix Unit -- the "
        "report's \"waiting for source register to be cleared (math is still "
        "using the previous data)\", tt-sim's ``src_reserved_by_matrix``"
    ),
    "WAITING_FOR_SRCB_CLEAR": ("as WAITING_FOR_SRCA_CLEAR, for the SrcB bank"),
    "THREAD_INSTRUCTIONS": (
        "instructions this thread issued past the wait gate -- the report's "
        '"average instructions issued per cycle, per thread" numerator, and '
        "the grant side of the same RTL instance (``ibuffer_rden[th]``)"
    ),
}

#: INSTRN counters deliberately **not** mapped, and why. Reported rather than
#: forced, per the rule that a counter which does not map cleanly is a finding.
INSTRN_DECLINED = {
    "WAITING_FOR_*_IDLE_*": (
        "The obvious reading -- tt-sim's ``tensix_stall_on_<unit>`` -- is "
        "contradicted by the tech report itself. Metric 11 calls these "
        '"cycles each thread waits for its primary hardware unit to become '
        'idle", but the Stall Breakdown section then removes the metrics '
        'built on that reading: "``WAITING_FOR_X`` counts cycles a hardware '
        "unit was busy -- not cycles the thread was stalled by that unit. "
        "When thread stalls are low (e.g. concat, tilize), ``WAITING_FOR_X >> "
        'THREAD_STALLS``, producing meaningless >100% values." A counter '
        "that outruns the thread's own total stall count is not a stall "
        "count, so the per-thread stall attribution tt-sim has is the wrong "
        "quantity. Unit-busy cycles are cost-model occupancy, which is absent "
        "with TT_SIM_COST_MODEL unset and would read zero in the default "
        "regime -- exactly the failure this module exists to remove."
    ),
    "ANY_THREAD_STALL": (
        "Select 283 is grant-side index 27, and the report says only that "
        'the grant lines collapse aliases: "the RTL wires grant as '
        "``{9{inst_stall_thread[th]}}`` per per-thread stall-reason instance, "
        "so ... the per-thread stall-reason grants reproduce "
        '``THREAD_STALLS_{th}``". Which instance index 27 lands on is not '
        "stated, so whether this is a union across threads or one thread's "
        "stall count is a guess. Not guessed."
    ),
    "*_INSTRN_AVAILABLE_*": (
        "Selects 0-23 count cycles an instruction of a given type was "
        "*available* at the thread's instruction buffer, which is a property "
        "of the buffer's occupancy rather than of issue. tt-sim's frontend "
        "FIFOs are unbounded queues filled by the pushing RISC-V core, so "
        "their occupancy is a modelling artefact and not the hardware's."
    ),
}


def _instrn_selects(blackhole: bool):
    """Build ``{(index, is_grant): name}`` for the INSTRN_THREAD bank.

    The two architectures genuinely differ, which is why this is a function of
    ``blackhole`` rather than a constant: Blackhole gives the four shared
    Src conditions one select each (27-30) and starts the per-thread
    stall-reason block at 31, while Wormhole replicates each Src condition
    across three selects (27-38) and starts the per-thread block at 39.
    Wormhole's ``hw_counters.h`` reads only the first of each triple (27, 30,
    33, 36), but the other two answer as well on hardware, so they are
    modelled rather than left to fall off the end of the table.
    """
    selects = {}
    for unit_idx, unit in enumerate(_AVAILABILITY_UNITS):
        for thread in range(3):
            sel = unit_idx * 3 + thread
            if unit == "XSEARCH":
                # Selects 9-11: the XSEARCH kick is tied to zero in both RTLs.
                continue
            selects[(sel, False)] = f"{unit}_INSTRN_AVAILABLE_{thread}"
    for thread in range(3):
        selects[(24 + thread, False)] = f"THREAD_STALLS_{thread}"
    if blackhole:
        for idx, name in enumerate(_SRC_CONDITIONS):
            selects[(27 + idx, False)] = name
        per_thread_base = 31
    else:
        for idx, name in enumerate(_SRC_CONDITIONS):
            for replica in range(3):
                selects[(27 + idx * 3 + replica, False)] = name
        per_thread_base = 39
    for thread in range(3):
        for idx, reason in enumerate(_PER_THREAD_REASONS):
            selects[(per_thread_base + thread * 9 + idx, False)] = f"{reason}_{thread}"
    # Grant side: one distinct value per instance.
    for thread in range(3):
        selects[(thread * 8, True)] = f"THREAD_INSTRUCTIONS_{thread}"
    selects[(27, True)] = "ANY_THREAD_STALL"
    return selects


class PerfCounterBank:
    """One ``tt_perf_cnt`` instance: start/stop edges, ``ref_cnt``, select."""

    def __init__(self, name):
        self.name = name
        self.ref_period = 0
        self.mode = 0
        self.control = 0
        self.running = False
        self.start_cycle = 0
        self.stop_cycle = 0

    @property
    def select(self):
        return (self.mode >> 8) & 0xFF

    @property
    def grant_side(self):
        return bool((self.mode >> 16) & 1)

    def write_control(self, value, cycle_num):
        """Apply ``PERF_CNT_<X>2``. Both bits are rising-edge triggered."""
        rising = value & ~self.control
        self.control = value
        started = False
        if rising & _START_BIT:
            # "Start (rising edge only; 0->1 transition also clears the
            # counters)" -- the clear is what makes a re-start a fresh window.
            self.running = True
            self.start_cycle = cycle_num
            self.stop_cycle = cycle_num
            started = True
        if rising & _STOP_BIT:
            self.running = False
            self.stop_cycle = cycle_num
        return started

    def ref_cnt(self, cycle_num):
        end = cycle_num if self.running else self.stop_cycle
        return max(0, end - self.start_cycle) & 0xFFFFFFFF


class TensixPerfCounters:
    """The five perf-counter banks of one Tensix tile.

    Counting is **armed by the hardware's own start bit**, not by a tt-sim
    switch: nothing accumulates until a kernel writes the rising edge to
    ``PERF_CNT_INSTRN_THREAD2``, which is exactly when real silicon starts
    counting. That is also what keeps an unprofiled run free of the cost —
    every hot-path hook is guarded by :attr:`instrn_running`, which is False
    for the whole of a run that never programs the block.
    """

    #: Set truthy to make an unmodelled counter read raise instead of warning.
    STRICT_ENV = "TT_SIM_STRICT_PERF_COUNTERS"

    def __init__(self, blackhole=False):
        self.blackhole = blackhole
        self.banks = {name: PerfCounterBank(name) for name in BANK_REGISTERS}
        self.mux_ctrl = 0
        self.all_control = 0
        self._instrn_selects = _instrn_selects(blackhole)
        self._warned = set()
        #: Hot-path guard. ``True`` only between the start and stop edges on
        #: the INSTRN bank, which is the only bank tt-sim sources.
        self.instrn_running = False
        self._reset_instrn()

    def _reset_instrn(self):
        self.thread_stalls = [0, 0, 0]
        self.thread_instructions = [0, 0, 0]
        self.sem_empty = [0, 0, 0]
        self.sem_full = [0, 0, 0]
        self.src_valid = {"A": 0, "B": 0}
        self.src_clear = {"A": 0, "B": 0}

    # -- the counting hooks ------------------------------------------------
    #
    # All three are called from the Tensix wait gate, which is the one place
    # that sees every stalled cycle, every accepted instruction and every cycle
    # a latched wait went unmet. None allocates, none consults the trace bus,
    # and none can move a cycle: they are pure increments on a branch the gate
    # had already taken.

    def note_stall(self, thread_id, reason, src_bank=None):
        """One cycle in which ``thread_id`` made no progress at the gate.

        Increments ``thread_stalls[thread_id]`` **and** at most one reason
        bucket, in the same call. That coupling is why
        ``sem_empty_t + sem_full_t <= thread_stalls_t`` holds on every log this
        class produces today, and why the mechanism-attribution leg's
        ``partition_closes`` gate has never been able to fail in simulation.
        The hardware does not couple them; see :meth:`note_wait_condition`.
        """
        self.thread_stalls[thread_id] += 1
        if reason == "semaphore_empty":
            self.sem_empty[thread_id] += 1
        elif reason == "semaphore_full":
            self.sem_full[thread_id] += 1
        elif reason == "src_reserved_by_unpacker":
            if src_bank is not None:
                self.src_valid[src_bank] += 1
        elif reason == "src_reserved_by_matrix":
            if src_bank is not None:
                self.src_clear[src_bank] += 1

    def note_wait_condition(self, thread_id, reason):
        """One cycle in which a *latched* wait condition was unsatisfied.

        Deliberately separate from :meth:`note_stall`, and deliberately not a
        stall: this is the counting rule the hardware uses and the one tt-sim's
        model could not previously express.

        ``SEMWAIT`` does not pause the thread. Per
        ``WormholeB0/TensixTile/TensixCoprocessor/SEMWAIT.md`` -- "the issuing
        thread can continue execution until one of the blocked instructions is
        reached, at which point execution of the thread is paused until all of
        the selected conditions are simultaneously met", and "the Wait Gate
        will then **continuously re-evaluate** the latched wait instruction
        until all of the selected conditions are simultaneously met" -- the
        latched condition is live from the ``SEMWAIT`` onwards, whether or not
        anything is currently being held at the gate. A counter wired to that
        condition therefore also runs through cycles the thread was *idle*.
        ``BlackholeA0/.../SEMWAIT.md`` is identical bar one recommendation
        paragraph, so this is not architecture-specific.

        Measured on silicon, 2026-08-17: on a Wormhole part running
        ``mechbench elw``, ``WAITING_FOR_NONZERO_SEM_2`` reads 3561 against a
        ``THREAD_STALLS_2`` of 270 -- a **single** reason counter outrunning the
        thread's whole stall count by 13x, which no amount of simultaneity
        between reasons can explain.

        The caller is ``WaitGate._tick_unheld_latched_wait``
        (``tt_sim/pe/tensix/frontend.py``), on the branch where a wait is
        latched and nothing is held by it -- the cycles the held-path hook
        ``_note_latched_wait`` never sees. Every one of them is counted here
        and *not* as a stall, so a simulated window can now carry
        ``sem_empty_t > thread_stalls_t`` exactly as silicon's does. Nothing is
        scaled or calibrated towards the reading above: how far a reason
        counter runs ahead is however long the modelled thread ran past its
        ``SEMWAIT``, and on the in-tree ``mechbench`` arms that is a handful of
        cycles rather than 13x.

        Only the two semaphore reasons are accepted. The four Src conditions are
        per-instruction ownership tests evaluated on whatever is at the gate,
        not latched conditions that outlive it, so a Src counter cannot run on a
        cycle with nothing held -- and a caller reaching for one here has
        mistaken which quantity it is counting.
        """
        if reason == "semaphore_empty":
            self.sem_empty[thread_id] += 1
        elif reason == "semaphore_full":
            self.sem_full[thread_id] += 1
        else:
            raise ValueError(
                f"note_wait_condition: {reason!r} is not a latched wait condition; "
                "only semaphore_empty and semaphore_full can count while the "
                "thread is not held at the gate"
            )

    def note_dispatch(self, thread_id):
        """One instruction accepted past the gate from ``thread_id``."""
        self.thread_instructions[thread_id] += 1

    # -- the register interface -------------------------------------------

    def read(self, addr, cycle_num):
        for name, (base, out_l, out_h) in BANK_REGISTERS.items():
            bank = self.banks[name]
            if addr == base:
                return bank.ref_period
            if addr == base + 4:
                return bank.mode
            if addr == base + 8:
                return bank.control
            if addr == out_l:
                return bank.ref_cnt(cycle_num)
            if addr == out_h:
                return self._counter_value(bank)
        if addr == PERF_CNT_MUX_CTRL:
            return self.mux_ctrl
        if addr == PERF_CNT_ALL:
            return self.all_control
        raise KeyError(addr)

    def write(self, addr, value, cycle_num):
        for name, (base, _, _unused) in BANK_REGISTERS.items():
            bank = self.banks[name]
            if addr == base:
                bank.ref_period = value
                return
            if addr == base + 4:
                bank.mode = value
                return
            if addr == base + 8:
                started = bank.write_control(value, cycle_num)
                if name == "INSTRN_THREAD":
                    if started:
                        self._reset_instrn()
                    self.instrn_running = bank.running
                return
        if addr == PERF_CNT_MUX_CTRL:
            self.mux_ctrl = value
            return
        if addr == PERF_CNT_ALL:
            # A plain store, deliberately. ``RISCV_DEBUG_REG_PERF_CNT_ALL``
            # exists in both arches' ``tensix.h``, but the tech report's
            # control-register table documents only the per-bank ``<X>0..2``
            # triplets and nothing in tt-metal ever writes it. "Broadcast
            # start/stop to every bank" is the obvious guess and it is still a
            # guess, so it is not made: a write is remembered and nothing is
            # started by it.
            self.all_control = value
            return
        # Data registers are read-only; a write is discarded exactly as the
        # hardware discards it.
        for _, out_l, out_h in BANK_REGISTERS.values():
            if addr in (out_l, out_h):
                return
        raise KeyError(addr)

    def _counter_value(self, bank):
        if bank.name != "INSTRN_THREAD":
            self._decline(
                f"{bank.name} bank, select {bank.select}"
                f"{' (grant)' if bank.grant_side else ''}",
                f"tt-sim models no counter in the {bank.name} bank",
            )
            return 0
        name = self._instrn_selects.get((bank.select, bank.grant_side))
        if name is None:
            self._decline(
                f"INSTRN_THREAD select {bank.select}"
                f"{' (grant)' if bank.grant_side else ''}",
                "no counter is defined at that select on "
                f"{'Blackhole' if self.blackhole else 'Wormhole'}",
            )
            return 0
        value = self._instrn_value(name)
        if value is None:
            self._decline(
                name, INSTRN_DECLINED.get(_declined_key(name), "not modelled")
            )
            return 0
        return value & 0xFFFFFFFF

    def _instrn_value(self, name):
        """The tt-sim quantity behind ``name``, or ``None`` if unsourced."""
        if name.startswith("THREAD_STALLS_"):
            return self.thread_stalls[int(name[-1])]
        if name.startswith("THREAD_INSTRUCTIONS_"):
            return self.thread_instructions[int(name[-1])]
        if name.startswith("WAITING_FOR_NONZERO_SEM_"):
            return self.sem_empty[int(name[-1])]
        if name.startswith("WAITING_FOR_NONFULL_SEM_"):
            return self.sem_full[int(name[-1])]
        if name == "WAITING_FOR_SRCA_VALID":
            return self.src_valid["A"]
        if name == "WAITING_FOR_SRCB_VALID":
            return self.src_valid["B"]
        if name == "WAITING_FOR_SRCA_CLEAR":
            return self.src_clear["A"]
        if name == "WAITING_FOR_SRCB_CLEAR":
            return self.src_clear["B"]
        return None

    def _decline(self, what, why):
        """Say out loud that a counter read is not a measurement.

        Once per distinct counter, so a profiler sweeping 59 selects on every
        core does not drown the log, but the first read of each is unmissable.
        The value still comes back as zero, because a 32-bit MMIO read has no
        encoding for "unknown" and raising here would abort a real
        ``TT_METAL_PROFILE_PERF_COUNTERS`` run mid-sweep; set
        ``TT_SIM_STRICT_PERF_COUNTERS`` to raise instead.
        """
        if os.environ.get(self.STRICT_ENV, "").lower() in ("1", "true", "yes", "on"):
            raise NotImplementedError(
                f"Tensix performance counter {what} is not modelled in tt-sim: {why}"
            )
        if what in self._warned:
            return
        self._warned.add(what)
        print(
            f"tt-sim WARNING: performance counter {what} is not modelled; "
            f"reading it returns 0, which is NOT a measurement of zero. {why}",
            file=sys.stderr,
        )


def _declined_key(name):
    if name.endswith(("_IDLE_0", "_IDLE_1", "_IDLE_2")):
        return "WAITING_FOR_*_IDLE_*"
    if "_INSTRN_AVAILABLE_" in name:
        return "*_INSTRN_AVAILABLE_*"
    return name
