"""Rung 4's RV-bound leg: tt-sim's baby-RISC-V cycles against a card's, by zone.

The third sibling of :mod:`tt_sim.perf.stall_attribution` (the Tensix leg, which
reads hardware stall counters) and :mod:`tt_sim.perf.noc_events` (the NoC leg,
which reads tt-metal's NoC event trace), and deliberately the same shape. This
one reads ``perfbench/retirebench``'s artefact -- a JSON table of per-zone
``mcycle`` and ``minstret`` deltas written by the host program on whichever side
produced it -- and answers the same question for the scalar core:

    **does tt-sim spend a baby RISC-V's cycles on the same mechanisms hardware
    does, or does it merely arrive at the same total?**

The criterion, stated so compensation cannot pass it
----------------------------------------------------

With mechanisms ``m`` partitioning the span, and an explicit **unattributed**
bucket on both sides so the two denominators are the same quantity::

    E_total = |Σ c_sim − Σ c_hw| / Σ c_hw
    E_int   = Σ |c_m,sim − c_m,hw| / Σ c_hw

Pass requires ``E_total <= 10 %`` **and** ``E_int <= 25 %``. The triangle
inequality gives ``E_total <= E_int`` always, so ``E_int / E_total`` is **the
compensation, measured** -- the number a passing total cannot fake, and the one
this module prints most prominently.

Every criterion is **per core**. On the 2026-08-10 part the whole physical
column ``x = 11`` keeps a wall-clock epoch 1.5e13 cycles from the rest of the
die, so no cross-core span is admissible and :func:`gate_per_core` exists to
make that unrepresentable rather than merely discouraged. One artefact is one
core and one RISC by construction, which is the same discipline arrived at from
the other end.

The instrument, and the constraint that shaped the program
-----------------------------------------------------------

``mcycle`` (0xb00) and ``minstret`` (0xb02) are Blackhole baby-RISC-V CSRs
(``BlackholeA0/TensixTile/BabyRISCV/CSRs.md``), modelled in
:mod:`tt_sim.pe.rv.isa.zicsr_isa`. ``mcycle`` reads the same tile clock that
``RISCV_DEBUG_REG_WALL_CLOCK_*`` samples, so a cycle here is the same cycle
every other perfbench program counts in.

``cfg0`` bit 10 ``DisCsrSync`` is documented: **while it is clear**, once a
``csrr*`` instruction leaves the front end the next instruction does not leave
until the previous has retired. It is clear at reset and clear in tt-metal's own
init -- the observed ``cfg0 = 0x60008`` sets only ``DisLowCash``,
``DisTriscCache`` and ``StMergeTimer``. So on a real part **a CSR read is a
retirement barrier**, and the program reads the counters **around windows,
never per instruction**: two reads bracketing thousands of instructions
serialise only at the boundaries and dilate nothing in between. Per-instruction
bracketing is retired as unreachable in ROADMAP §4 for exactly this reason, and
nothing here has a fidelity that depends on a CSR read being free. The
``marker_null`` zone is two marker pairs with nothing between them, so the
instrument's own cost is a **measured** number in every artefact rather than an
assumption in a comment, and :func:`gate_zone_budget` divides the smallest zone
by it instead of by a remembered "~30 cycles".

Which buckets are hardware-attributed and which are structural
---------------------------------------------------------------

**This is the one thing a reader of this leg must not get wrong**, and it is
the respect in which it is honestly weaker than its two siblings.

* Each bucket's **magnitude is hardware-attributed**: it is an ``mcycle``
  delta, read by the core whose cycles it counts, on whichever side produced
  the artefact. Nothing here is inferred, modelled or apportioned.
* Each bucket's **mechanism label is structural**: it comes from the way
  ``perfbench/retirebench``'s kernel is built -- a zone of dependent ``addi``,
  a zone of L1 pointer chase, a zone of ``divu`` -- and **not** from any
  hardware counter that says "these cycles were a load-use interlock". The
  hardware has no such counter for a baby RISC-V.
* What makes the structural label **checkable rather than merely asserted** is
  ``minstret``. The same binary must retire the same instructions in the same
  zone on both sides; :func:`gate_retire_census_matches` requires exact
  equality and refuses otherwise. A zone whose two sides retired different
  instruction counts did not run the same zone, whatever it is called.

The sibling legs are stronger here: ``stall_attribution``'s buckets are named by
the hardware's own counter selects, and ``noc_events``'s are named by the event
tt-metal's own recorder emitted. This leg's are named by the program. Say so.

Why the decomposition cannot be finer, and what was refused
------------------------------------------------------------

The hardware gives, per window, **elapsed cycles and retired instructions**.
That is close to all of it.

* ``mhpmcounter3`` / ``mhpmcounter4`` exist, but the encodings of their
  ``mhpmevent3`` / ``mhpmevent4`` selectors are **unpublished**, so no event
  can be given a meaning and no count can be honest.
  :mod:`tt_sim.pe.rv.isa.zicsr_isa` already refuses those counters once
  software has selected an event. **No event mapping was invented to
  manufacture more buckets**, and none may be.
* There is **no PC sampler and no instruction-trace buffer** in tt-metal 0.74,
  in UMD, or in the public ISA docs. The debug daisychain is documented "at
  least five cycles stale" and every consumer of it is commented out.

So the mechanism split has to come from the program's own structure, which
ROADMAP §4's phrasing -- "an interior match by mechanism **and by zone**" --
admits. A coarse decomposition that is honest beats a fine one that is invented.

**The per-zone cycles-per-instruction table is not an independent second view.**
:func:`zone_cpi` divides a bucket by its retired count, and
:func:`gate_retire_census_matches` has already required the two sides' retired
counts to be *equal*; so once that gate passes, the CPI table is the partition
divided by a common constant per zone and carries exactly the same information.
It is reported because cycles-per-instruction is the unit a RISC-V mechanism is
naturally quoted in and is directly comparable to a ``perfbench/riscvbench``
slope -- not because it is a second chance to catch anything.
``noc_events``'s per-class latency table *is* independent of its partition;
this one is not, and the difference is real.

Blackhole only, and it refuses rather than degrades
-----------------------------------------------------

The string ``csr`` appears **zero times** in the whole ``WormholeB0`` doc tree,
so a Wormhole baby core has no CSRs to model: tt-sim raises ``NoCSRsError`` on a
CSR instruction there, and there is no ``minstret`` on the part either. Without
retired counts the structural labels stop being checkable and the leg degrades
to an elapsed-only envelope check -- which is precisely what rung 4 exists to
distrust, and three of which this repo already has. So
:func:`gate_arch_supported` **refuses a non-Blackhole artefact with a message
saying why**, and ``perfbench/retirebench``'s host program refuses a
non-Blackhole part before it builds its kernel or launches anything.

Provenance
----------

``isa_doc`` for the instrument (the CSR table), and **inert**. No number this
module reads, computes or prints may become a cost; nothing here writes to
``unit_costs.yaml`` or any other cost table. This leg *fits nothing* -- a
disagreement is the result, never a reason to tune the simulator. Silicon is
corroboration, never provenance.

Usage
-----

::

    python3 -m tt_sim.perf.retire_attribution \\
        --sim  sim-session/retirebench-blackhole-sim.json \\
        --card card-session/runs/1/retirebench-blackhole-card-1.json \\
        --report report.txt --json report.json

    # tt-sim side alone, to see the decomposition with no card data yet:
    python3 -m tt_sim.perf.retire_attribution --sim <artefact> --decompose-only
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

#: ``E_total <= 10 %`` -- the envelope control, the level nekbone's per-core
#: zones already reach. Kept as a fraction so the report can print either.
E_TOTAL_LIMIT = 0.10

#: ``E_int <= 25 %``. 2.5x the envelope threshold, the same allowance the
#: Tensix leg makes and for the same reason: a decomposition has strictly more
#: ways to be wrong than a total does.
E_INT_LIMIT = 0.25

#: Per-zone cycles-per-instruction tolerance. The same 25 %, because the CPI
#: table is the partition divided by a per-zone constant (see the module
#: docstring) and a second, different bar would imply a second, independent
#: measurement.
CPI_TOLERANCE = 0.25

#: ROADMAP §4: at most 60 zones per RISC. Half the profiler's hard 125-marker
#: budget, so that even an artefact routed through the device profiler could not
#: silently drop one. retirebench writes its own L1 table and is not bound by
#: that budget, but the ceiling is kept because the criterion states it and
#: because a partition with 60 buckets is already past what a reader can hold.
MAX_ZONES = 60

#: ROADMAP §4: every zone at least 1000 cycles, so the two-marker cost is a
#: small fraction of it. Checked against :data:`MARKER_BUDGET` as well, using
#: the marker cost the calibration zone **measured** on that side.
MIN_ZONE_CYCLES = 1000

#: The measured two-marker cost may be at most this fraction of the smallest
#: measured zone. ROADMAP §4 states 6 % against a ~28-36 cycle marker pair.
MARKER_BUDGET = 0.06

#: The zone that measures the instrument instead of a mechanism: two marker
#: pairs with nothing between them. Exempt from :data:`MIN_ZONE_CYCLES` -- it is
#: supposed to be short, and a 1000-cycle calibration zone would mean the
#: instrument cost more than the mechanisms.
CALIBRATION_ZONE = "marker_null"

#: The bucket holding everything inside the outer window that no zone claimed:
#: the harness's own result stores, the inter-zone register setup, and the outer
#: marker pair. Defined as ``window − Σ zones``, so the partition telescopes to
#: the window **by construction** -- which is what keeps both sides'
#: denominators the same quantity and the ``E_total <= E_int`` inequality (and
#: therefore the compensation ratio) meaning something.
UNATTRIBUTED = "unattributed"

#: The only architecture whose ISA documentation describes these CSRs.
SUPPORTED_ARCH = "blackhole"

#: Said once, appended to the architecture refusal, so a reader does not have to
#: reconstruct the argument from the roadmap.
WRONG_ARCH_EXPLANATION = (
    "This leg's instrument is the mcycle (0xb00) and minstret (0xb02) CSRs, "
    "documented in BlackholeA0/TensixTile/BabyRISCV/CSRs.md. The string 'csr' "
    "appears ZERO times in the whole WormholeB0 doc tree, so a Wormhole baby "
    "core has no CSRs to read: tt-sim raises NoCSRsError on a CSR instruction "
    "there. Without minstret there is no retired-instruction count, the zone "
    "labels stop being checkable against anything, and what is left is an "
    "elapsed-only envelope check. This repo already has three of those, and "
    "rung 4 exists because an envelope cannot see a compensating interior -- so "
    "emitting a weaker claim under this leg's name would be worse than emitting "
    "nothing. Do not add a Wormhole fallback."
)

#: Appended to the cost-model refusal. The claim it rests on is one-sided and
#: documented, which is what makes it a gate rather than a note.
COST_MODEL_EXPLANATION = (
    "An L1 load cannot cost what an addi costs on any part these docs "
    "describe: BabyRISCV/README.md's load-latency table gives L1 a latency of "
    ">= 8 on Wormhole and a 2-or->=8 pair on Blackhole, against an integer op "
    "documented at latency 1 and throughput 1. A dependent pointer chase whose "
    "cycles-per-instruction matches a dependent addi chain's therefore did not "
    "run against a modelled load path. On the simulator side the reachable "
    "cause is TT_SIM_COST_MODEL being unset, which leaves RV32I.rv_cost as None "
    "and every load, store, multiply and divide at one cycle; re-run with "
    "TT_SIM_COST_MODEL=1 (perfbench/retirebench/run_sim.sh sets it) before "
    "reading anything into this comparison."
)

#: The two zones the cost-model gate compares, and the minimum ratio between
#: their cycles-per-instruction. Both are dependent chains of a single
#: instruction, so the only thing that separates them is the load path.
COST_MODEL_PROBE = ("load_dep", "alu_dep")
COST_MODEL_MIN_RATIO = 1.25


# ---------------------------------------------------------------------------
# Reading the artefact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Zone:
    """One zone as ``perfbench/retirebench``'s host program wrote it.

    ``mechanism`` is the program's own description of what the zone was built to
    be dominated by. It is carried through to the report verbatim so that a
    reader of the JSON alone cannot mistake it for a hardware attribution.
    """

    index: int
    name: str
    mechanism: str
    reps: int
    cycles: int
    retired: int


@dataclass
class Run:
    """One artefact: one core, one RISC, one launch."""

    path: str
    version: int
    arch: str
    label: str
    risc: str
    core: tuple
    logical_core: tuple
    scale: int
    window_cycles: int
    window_retired: int
    zones: list = field(default_factory=list)

    @property
    def stream(self):
        return f"{self.risc} @ {self.core[0]},{self.core[1]}"

    @property
    def zone_names(self):
        return tuple(z.name for z in self.zones)

    @property
    def by_name(self):
        return {z.name: z for z in self.zones}

    @property
    def marker_cycles(self):
        """The two-marker cost this run measured, or ``None`` if it has no
        calibration zone."""
        zone = self.by_name.get(CALIBRATION_ZONE)
        return None if zone is None else zone.cycles


def _require(obj, key, path):
    if key not in obj:
        raise ValueError(
            f"{path}: record lacks '{key}', so it is not a retirebench artefact"
        )
    return obj[key]


def load_run(path):
    """Parse one ``retirebench-*.json``.

    Raises rather than guessing on a file that is not this artefact: a silently
    empty zone list would read downstream as "this core did no work", which is a
    perfectly plausible and completely wrong reading.
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or "retirebench" not in raw:
        raise ValueError(
            f"{path}: expected a retirebench artefact object with a "
            f"'retirebench' version field, got {type(raw).__name__}"
        )
    window = _require(raw, "window", path)
    zones = []
    for index, rec in enumerate(_require(raw, "zones", path)):
        zones.append(
            Zone(
                index=int(rec.get("index", index)),
                name=str(_require(rec, "name", path)),
                mechanism=str(rec.get("mechanism", "")),
                reps=int(rec.get("reps", 0)),
                cycles=int(_require(rec, "cycles", path)),
                retired=int(_require(rec, "retired", path)),
            )
        )
    core = _require(raw, "core", path)
    logical = raw.get("logical_core", core)
    return Run(
        path=str(path),
        version=int(raw["retirebench"]),
        arch=str(_require(raw, "arch", path)),
        label=str(raw.get("label", "")),
        risc=str(raw.get("risc", "")),
        core=(int(core[0]), int(core[1])),
        logical_core=(int(logical[0]), int(logical[1])),
        scale=int(raw.get("scale", 1)),
        window_cycles=int(_require(window, "cycles", path)),
        window_retired=int(_require(window, "retired", path)),
        zones=zones,
    )


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


def mechanisms(run):
    """The bucket names, in report order: every zone, then ``unattributed``."""
    return run.zone_names + (UNATTRIBUTED,)


def partition(run):
    """Cut ``run``'s outer window into :func:`mechanisms`, in cycles.

    ``unattributed`` is ``window − Σ zones`` and is therefore whatever is left:
    the harness's own result stores between zones, the register setup, and the
    outer marker pair. It may come back **negative**, and it is not clamped --
    a negative bucket is what :func:`gate_partition_closes` refuses on, and
    clamping it here would turn a finding into a plausible number.
    """
    part = {z.name: z.cycles for z in run.zones}
    part[UNATTRIBUTED] = run.window_cycles - sum(z.cycles for z in run.zones)
    return {name: part[name] for name in mechanisms(run)}


def retire_partition(run):
    """The same partition in retired instructions rather than cycles.

    This is the quantity :func:`gate_retire_census_matches` compares, and it is
    what makes the structural zone labels checkable: the same binary must retire
    the same instructions in the same zone on both sides.
    """
    part = {z.name: z.retired for z in run.zones}
    part[UNATTRIBUTED] = run.window_retired - sum(z.retired for z in run.zones)
    return {name: part[name] for name in mechanisms(run)}


def zone_cpi(run):
    """``{zone: cycles per retired instruction}``.

    Not an independent view of the partition -- see the module docstring. A zone
    that retired nothing is absent rather than infinite.
    """
    return {z.name: z.cycles / z.retired for z in run.zones if z.retired}


def partition_closes(part, span):
    """``(ok, reasons)`` -- every bucket non-negative and the sum exact."""
    reasons = []
    for name, value in part.items():
        if value < 0:
            reasons.append(f"{name} = {value} (negative)")
    total = sum(part.values())
    if total != span:
        reasons.append(f"buckets sum to {total}, window span is {span}")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# The criterion
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """``E_total``, ``E_int`` and the compensation ratio for one partition."""

    unit: str
    mechanisms: tuple
    sim: dict
    hw: dict
    e_total: float
    e_int: float
    ratio: float | None
    denominator: int

    @property
    def passed(self):
        return self.e_int <= E_INT_LIMIT and self.e_total <= E_TOTAL_LIMIT

    def to_dict(self):
        return {
            "unit": self.unit,
            "mechanisms": list(self.mechanisms),
            "sim": {k: self.sim[k] for k in self.mechanisms},
            "hw": {k: self.hw[k] for k in self.mechanisms},
            "e_total": self.e_total,
            "e_int": self.e_int,
            "compensation_ratio": self.ratio,
            "denominator_cycles": self.denominator,
            "passed": self.passed,
        }


def compare_partitions(unit, names, sim, hw):
    """Build a :class:`Comparison` from two partitions over the same buckets.

    ``E_total`` reduces to the span error, because both partitions sum to their
    own span by construction. That is the point of insisting on an explicit
    unattributed bucket: without it the two denominators would be different
    quantities and the inequality ``E_total <= E_int`` would stop meaning
    anything.
    """
    names = tuple(names)
    denom = sum(hw[m] for m in names)
    if denom <= 0:
        raise ValueError(f"{unit}: card span is {denom}; nothing to divide by")
    e_total = abs(sum(sim[m] for m in names) - denom) / denom
    e_int = sum(abs(sim[m] - hw[m]) for m in names) / denom
    if e_total > 0:
        ratio = e_int / e_total
    else:
        ratio = None if e_int == 0 else float("inf")
    return Comparison(
        unit=unit,
        mechanisms=names,
        sim=dict(sim),
        hw=dict(hw),
        e_total=e_total,
        e_int=e_int,
        ratio=ratio,
        denominator=denom,
    )


@dataclass
class CpiComparison:
    """One zone's cycles per retired instruction, both sides."""

    zone: str
    mechanism: str
    sim_cpi: float
    hw_cpi: float
    retired: int

    @property
    def error(self):
        return (
            abs(self.sim_cpi - self.hw_cpi) / self.hw_cpi
            if self.hw_cpi
            else float("inf")
        )

    @property
    def passed(self):
        return self.error <= CPI_TOLERANCE

    def to_dict(self):
        return {
            "zone": self.zone,
            "mechanism": self.mechanism,
            "sim_cycles_per_instruction": self.sim_cpi,
            "hw_cycles_per_instruction": self.hw_cpi,
            "retired": self.retired,
            "relative_error": self.error,
            "passed": self.passed,
        }


def compare_cpi(sim, hw):
    """One :class:`CpiComparison` per **measured** zone present on both sides.

    The calibration zone is left out, and that is a criterion decision rather
    than tidiness: it measures the instrument, and the two sides' marker costs
    are *expected* to differ, because tt-sim charges a CSR instruction nothing
    at all -- ``DisCsrSync``'s serialisation has no published cycle count and
    ``tt_sim/pe/rv/cost.py`` declines to invent one. Grading a zone against a
    number the model deliberately does not have would fail every honest run for
    a reason already recorded as an open gap. It stays in the *partition*, where
    its handful of cycles are counted like any other, and :func:`marker_notes`
    states the difference as a number.
    """
    sim_cpi, hw_cpi = zone_cpi(sim), zone_cpi(hw)
    hw_zones = hw.by_name
    out = []
    for zone in sim.zones:
        if zone.name == CALIBRATION_ZONE:
            continue
        if zone.name not in sim_cpi or zone.name not in hw_cpi:
            continue
        out.append(
            CpiComparison(
                zone=zone.name,
                mechanism=zone.mechanism,
                sim_cpi=sim_cpi[zone.name],
                hw_cpi=hw_cpi[zone.name],
                retired=hw_zones[zone.name].retired,
            )
        )
    return out


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def line(self):
        """The gate's verdict, one ``|``-separated clause per line.

        A refusal that explains itself is long by design, and a 600-character
        single line is a refusal nobody reads to the end of.
        """
        head = f"  [{'PASS' if self.passed else 'REFUSE'}] {self.name}: "
        clauses = [c.strip() for c in self.detail.split(" | ")]
        pad = " " * len(head)
        return "\n".join(
            (head if i == 0 else pad) + clause for i, clause in enumerate(clauses)
        )


def _sides(sim, hw):
    return [("sim", sim)] + ([("card", hw)] if hw is not None else [])


def gate_arch_supported(sim, hw):
    """Every side must be a Blackhole artefact. **The Wormhole refusal.**

    Not a warning and not a degraded mode: see :data:`WRONG_ARCH_EXPLANATION`.
    """
    wrong = [
        f"{label} artefact reports arch '{side.arch}'"
        for label, side in _sides(sim, hw)
        if side.arch != SUPPORTED_ARCH
    ]
    if wrong:
        return GateResult(
            "arch_supported", False, "; ".join(wrong) + " | " + WRONG_ARCH_EXPLANATION
        )
    return GateResult(
        "arch_supported", True, f"both sides are {SUPPORTED_ARCH} artefacts"
    )


def gate_zone_table_matches(sim, hw):
    """The two sides must carry the same zones, in the same order, at the same
    size.

    Not the same timing -- the same *program*. The zone table and the
    repetition counts are compiled into the binary, so an identical binary at an
    identical ``--scale`` produces an identical table; a difference means the
    two sides ran different work, and any timing comparison between them is a
    comparison of two programs.
    """
    if hw is None:
        return GateResult(
            "zone_table_matches", True, f"{len(sim.zones)} zones, no card data"
        )
    problems = []
    if sim.zone_names != hw.zone_names:
        problems.append(
            f"sim zones {list(sim.zone_names)} against card zones {list(hw.zone_names)}"
        )
    else:
        for s, h in zip(sim.zones, hw.zones):
            if s.reps != h.reps:
                problems.append(f"{s.name}: sim reps {s.reps}, card reps {h.reps}")
            if s.mechanism != h.mechanism:
                problems.append(
                    f"{s.name}: sim calls it '{s.mechanism}', card '{h.mechanism}'"
                )
    if sim.scale != hw.scale:
        problems.append(f"sim --scale {sim.scale}, card --scale {hw.scale}")
    if sim.version != hw.version:
        problems.append(
            f"sim artefact version {sim.version}, card {hw.version}; the zone "
            "table's meaning is not guaranteed across versions"
        )
    if problems:
        return GateResult(
            "zone_table_matches",
            False,
            "the two sides ran different programs -- " + "; ".join(problems),
        )
    return GateResult(
        "zone_table_matches",
        True,
        f"{len(sim.zones)} zones, identical names, mechanisms and reps at scale {sim.scale}",
    )


def gate_counters_advanced(sim, hw):
    """Both counters must have moved, on both sides, in every measured zone.

    A core whose counters stood still reads back as a full set of zeros, which
    decodes as a perfectly plausible partition of a zero-cycle span -- the exact
    failure mode a comparison would otherwise pass with flying colours. The
    calibration zone is exempt from the *cycle* requirement only: an instrument
    that costs zero cycles is a claim about the instrument, and
    :func:`gate_zone_budget` reports it rather than refusing it.
    """
    problems = []
    for label, side in _sides(sim, hw):
        if side.window_cycles <= 0:
            problems.append(
                f"{label} outer window advanced {side.window_cycles} cycles"
            )
        if side.window_retired <= 0:
            problems.append(
                f"{label} outer window retired {side.window_retired} instructions"
            )
        for zone in side.zones:
            if zone.name == CALIBRATION_ZONE:
                continue
            if zone.cycles <= 0:
                problems.append(
                    f"{label} zone {zone.name} advanced {zone.cycles} cycles"
                )
            if zone.retired <= 0:
                problems.append(
                    f"{label} zone {zone.name} retired {zone.retired} instructions"
                )
    if problems:
        return GateResult(
            "counters_advanced",
            False,
            "; ".join(problems) + " -- the counters did not run",
        )
    detail = f"sim window {sim.window_cycles} cycles / {sim.window_retired} retired"
    if hw is not None:
        detail += f", card {hw.window_cycles} / {hw.window_retired}"
    return GateResult("counters_advanced", True, detail)


def gate_per_core(sim, hw, mapped=False):
    """One core on each side, the same core, and the same RISC.

    On the 2026-08-10 part the whole physical column ``x = 11`` keeps a
    wall-clock epoch 1.5e13 cycles from the rest of the die, so pooling two
    cores' windows is not a coarser measurement, it is a meaningless one. An
    artefact is one core by construction, so what is left to refuse is a sim
    core silently compared against a differently-numbered card core -- and the
    Wormhole ``(1,1)`` versus Blackhole ``(1,2)`` worker-origin difference makes
    that an easy and invisible mistake. ``--map-core`` states the correspondence
    out loud when it is genuinely intended.
    """
    if hw is None:
        return GateResult("per_core", True, f"single stream {sim.stream}")
    problems = []
    if sim.core != hw.core and not mapped:
        problems.append(
            f"sim core {sim.core} compared against card core {hw.core}; "
            "pass --map-core SIMX,SIMY=CARDX,CARDY if that is intended"
        )
    if sim.risc != hw.risc:
        problems.append(f"sim RISC {sim.risc} compared against card RISC {hw.risc}")
    if problems:
        return GateResult("per_core", False, "; ".join(problems))
    detail = f"{sim.risc} on core {sim.core}"
    if sim.core != hw.core:
        detail = f"sim {sim.stream} mapped to card {hw.stream} explicitly"
    return GateResult("per_core", True, detail)


def gate_zone_budget(sim, hw):
    """ROADMAP §4's zone discipline, checked against measured numbers.

    At most :data:`MAX_ZONES` zones; every measured zone at least
    :data:`MIN_ZONE_CYCLES`; and the **measured** two-marker cost at most
    :data:`MARKER_BUDGET` of the smallest measured zone. The last is the reason
    the calibration zone exists: the roadmap's "~28-36 cycle two-marker cost"
    is a Blackhole estimate, tt-sim charges a CSR instruction nothing at all
    (``tt_sim/pe/rv/cost.py`` declines it for want of a published number), and
    the two sides' marker costs are therefore *not* the same. Reading it off the
    artefact makes that a printed number instead of an assumption, on each side
    separately.
    """
    problems = []
    for label, side in _sides(sim, hw):
        if len(side.zones) > MAX_ZONES:
            problems.append(
                f"{label} has {len(side.zones)} zones, over the {MAX_ZONES} ceiling"
            )
        measured = [z for z in side.zones if z.name != CALIBRATION_ZONE]
        if not measured:
            problems.append(
                f"{label} has no measured zone at all -- a partition whose only "
                f"bucket is the instrument's own calibration decomposes nothing"
            )
        short = [z for z in measured if z.cycles < MIN_ZONE_CYCLES]
        for zone in short:
            problems.append(
                f"{label} zone {zone.name} is {zone.cycles} cycles, under the "
                f"{MIN_ZONE_CYCLES}-cycle floor"
            )
        marker = side.marker_cycles
        if marker is None:
            problems.append(
                f"{label} has no '{CALIBRATION_ZONE}' zone, so the two-marker "
                "cost is unmeasured and the floor above rests on an assumption"
            )
        elif measured:
            smallest = min(measured, key=lambda z: z.cycles)
            if smallest.cycles > 0 and marker / smallest.cycles > MARKER_BUDGET:
                problems.append(
                    f"{label} two-marker cost {marker} is "
                    f"{100.0 * marker / smallest.cycles:.1f} % of the smallest "
                    f"zone ({smallest.name}, {smallest.cycles} cycles), over the "
                    f"{100.0 * MARKER_BUDGET:.0f} % budget"
                )
    if problems:
        return GateResult("zone_budget", False, "; ".join(problems))
    detail = []
    for label, side in _sides(sim, hw):
        measured = [z for z in side.zones if z.name != CALIBRATION_ZONE]
        smallest = min(measured, key=lambda z: z.cycles)
        detail.append(
            f"{label}: {len(side.zones)} zones, smallest measured {smallest.name} "
            f"{smallest.cycles} cycles, two-marker cost {side.marker_cycles} "
            f"({100.0 * side.marker_cycles / smallest.cycles:.2f} %)"
        )
    return GateResult("zone_budget", True, "; ".join(detail))


def gate_cost_model_engaged(sim, hw):
    """A dependent L1 load must cost more than a dependent ``addi``.

    The one regime error this leg can detect from the artefact alone, and it is
    detectable because the claim is one-sided and documented -- see
    :data:`COST_MODEL_EXPLANATION`. With ``TT_SIM_COST_MODEL`` unset,
    ``RV32I.rv_cost`` is ``None`` and every load, store, multiply and divide
    costs one cycle, so nine of the eleven measured zones collapse onto each
    other and the run produces a *well-formed* partition that would be compared
    against a card as if it meant something.

    Checked on both sides rather than only the simulator's, because a card that
    reported it would be saying something about its own instrument, and that is
    a finding rather than an assumption to be coded around.
    """
    chase, alu = COST_MODEL_PROBE
    problems = []
    for label, side in _sides(sim, hw):
        cpi = zone_cpi(side)
        if chase not in cpi or alu not in cpi:
            continue
        if cpi[alu] <= 0:
            continue
        ratio = cpi[chase] / cpi[alu]
        if ratio < COST_MODEL_MIN_RATIO:
            problems.append(
                f"{label} {chase} is {cpi[chase]:.3f} cycles/instruction against "
                f"{alu} at {cpi[alu]:.3f} ({ratio:.2f}x, under the "
                f"{COST_MODEL_MIN_RATIO}x floor)"
            )
    if problems:
        return GateResult(
            "cost_model_engaged",
            False,
            "; ".join(problems) + " | " + COST_MODEL_EXPLANATION,
        )
    detail = []
    for label, side in _sides(sim, hw):
        cpi = zone_cpi(side)
        if chase in cpi and alu in cpi and cpi[alu]:
            detail.append(f"{label} {chase}/{alu} = {cpi[chase] / cpi[alu]:.2f}x")
    return GateResult(
        "cost_model_engaged",
        True,
        "; ".join(detail) if detail else "no load/alu pair to compare",
    )


def gate_partition_closes(sim, hw):
    """No negative bucket, and the buckets sum to the outer window.

    Closure is by construction, so this is really a monotonicity and
    containment check -- and a real one. ``unattributed`` goes negative exactly
    when the zones claim more cycles than the window that contains them, which
    a torn or wrapped counter read produces (the kernel stores the low 32 bits
    of each counter, and an unsigned difference that went backwards comes back
    as a value near 2^32). If it refuses on card data, that is a finding about
    the instrument and must be reported, not smoothed over.
    """
    problems = []
    for label, side in _sides(sim, hw):
        ok, reasons = partition_closes(partition(side), side.window_cycles)
        if not ok:
            problems.append(f"{label} cycle partition: {'; '.join(reasons)}")
        ok_r, reasons_r = partition_closes(retire_partition(side), side.window_retired)
        if not ok_r:
            problems.append(f"{label} retire partition: {'; '.join(reasons_r)}")
    if problems:
        return GateResult("partition_closes", False, " | ".join(problems))
    return GateResult(
        "partition_closes",
        True,
        "every bucket >= 0 and both partitions sum to the outer window",
    )


def gate_retire_census_matches(sim, hw):
    """The two sides must have **retired the same instructions** in every zone.

    This is the gate that turns a structural zone label into a checkable claim,
    and it is the closest this leg comes to a hardware attribution: the same
    binary at the same ``--scale`` executes a deterministic instruction sequence
    with no interrupts, no traps and no data-dependent control flow, so the two
    sides' ``minstret`` deltas are the same number or the two sides did not run
    the same zone. Exact equality is required rather than a tolerance, for the
    same reason ``noc_events`` requires an exact transaction census: a
    difference is a statement about *what ran*, not about how fast it ran, and
    there is no size of difference that would be acceptable.

    If it refuses on card data, that is this leg's finding and it must be
    reported. Two things it would mean, in order of likelihood: the two sides
    were not built from the same source or at the same ``--scale`` (which
    :func:`gate_zone_table_matches` catches first when the table itself
    differs), or tt-sim's retire accounting does not match the hardware's --
    which would be a defect in :mod:`tt_sim.pe.rv.isa.zicsr_isa`'s counter,
    worth fixing, and not a reason to widen anything here.
    """
    if hw is None:
        return GateResult(
            "retire_census_matches", True, "no card data to match against"
        )
    sim_r, hw_r = retire_partition(sim), retire_partition(hw)
    problems = []
    for name in mechanisms(sim):
        s, h = sim_r.get(name, 0), hw_r.get(name, 0)
        if s != h:
            problems.append(f"{name}: sim retired {s}, card {h} ({s - h:+d})")
    if problems:
        return GateResult(
            "retire_census_matches",
            False,
            "the two sides did not retire the same instructions -- "
            + "; ".join(problems),
        )
    return GateResult(
        "retire_census_matches",
        True,
        f"{sim.window_retired} instructions retired, identical in every bucket",
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class RunReport:
    core: tuple
    risc: str
    card_core: tuple | None = None
    gates: list = field(default_factory=list)
    refused: bool = False
    comparison: Comparison | None = None
    cpis: list = field(default_factory=list)
    sim_partition: dict = field(default_factory=dict)
    hw_partition: dict = field(default_factory=dict)
    sim_retire: dict = field(default_factory=dict)
    sim_cpi: dict = field(default_factory=dict)
    mechanism_labels: dict = field(default_factory=dict)
    sim_window: int = 0
    hw_window: int = 0
    sim_marker: int | None = None
    hw_marker: int | None = None
    notes: list = field(default_factory=list)

    @property
    def label(self):
        return f"{self.risc} @ {self.core[0]},{self.core[1]}"

    @property
    def passed(self):
        if self.refused or self.comparison is None:
            return False
        return self.comparison.passed and all(c.passed for c in self.cpis)

    def to_dict(self):
        return {
            "core": list(self.core),
            "risc": self.risc,
            "card_core": list(self.card_core) if self.card_core else None,
            "refused": self.refused,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
            "sim_window_cycles": self.sim_window,
            "hw_window_cycles": self.hw_window,
            "sim_marker_cycles": self.sim_marker,
            "hw_marker_cycles": self.hw_marker,
            "mechanism_labels": self.mechanism_labels,
            "sim_partition": self.sim_partition,
            "hw_partition": self.hw_partition,
            "sim_retire_partition": self.sim_retire,
            "sim_cycles_per_instruction": self.sim_cpi,
            "partition_comparison": self.comparison.to_dict()
            if self.comparison
            else None,
            "cpi_comparisons": [c.to_dict() for c in self.cpis],
            "passed": self.passed,
            "notes": self.notes,
        }


def marker_notes(sim, hw):
    """Say what the two sides' measured marker costs were, and what differs.

    Deliberately a note and not a gate. The two sides are *expected* to disagree
    here: tt-sim charges a CSR instruction nothing, because ``DisCsrSync``'s
    serialisation has no published cycle count and this repo's provenance rules
    forbid inventing one (``tt_sim/pe/rv/cost.py`` records the omission). So the
    simulator's marker pair is a handful of cycles and a card's is the
    serialisation. It lands in every bucket, on the card side only, at
    ``2 x marker`` per zone -- one pair opening it and one closing it are inside
    the neighbouring buckets -- and it is small, bounded, and better stated than
    silently absorbed.
    """
    notes = []
    if sim.marker_cycles is None or hw is None or hw.marker_cycles is None:
        return notes
    if sim.marker_cycles == hw.marker_cycles:
        return notes
    zones = max(1, len(hw.zones))
    drift = (hw.marker_cycles - sim.marker_cycles) * zones
    notes.append(
        f"the two-marker cost differs: sim {sim.marker_cycles} cycles, card "
        f"{hw.marker_cycles}. tt-sim charges a CSR instruction nothing -- "
        f"cfg0's DisCsrSync serialisation has no published cycle count and "
        f"tt_sim/pe/rv/cost.py declines to invent one -- so about "
        f"{drift:+d} cycles of the span difference "
        f"({100.0 * abs(drift) / max(1, hw.window_cycles):.2f} % of the card "
        f"window) is the instrument rather than a mechanism."
    )
    return notes


def analyse(sim, hw, mapped=False):
    """Run every gate, then the criterion. One artefact is one core and one RISC.

    Returns a one-element list, so the shape matches the sibling legs' reports
    and ``render`` does not need to know which leg it is printing.

    Gates run first and a refusal stops the analysis: an ``E_int`` computed over
    a partition that does not close is a number, but it is not a measurement of
    anything, and printing it invites it to be quoted.
    """
    report = RunReport(
        core=sim.core, risc=sim.risc, card_core=hw.core if hw is not None else None
    )
    report.sim_window = sim.window_cycles
    report.hw_window = hw.window_cycles if hw is not None else 0
    report.sim_marker = sim.marker_cycles
    report.hw_marker = hw.marker_cycles if hw is not None else None
    report.mechanism_labels = {z.name: z.mechanism for z in sim.zones}
    for gate in (
        gate_arch_supported(sim, hw),
        gate_zone_table_matches(sim, hw),
        gate_counters_advanced(sim, hw),
        gate_per_core(sim, hw, mapped=mapped),
        gate_zone_budget(sim, hw),
        gate_cost_model_engaged(sim, hw),
        gate_partition_closes(sim, hw),
        gate_retire_census_matches(sim, hw),
    ):
        report.gates.append(gate)
        if not gate.passed:
            report.refused = True
    if report.refused:
        return [report]

    report.sim_partition = partition(sim)
    report.sim_retire = retire_partition(sim)
    report.sim_cpi = zone_cpi(sim)
    if hw is None:
        report.notes.append(
            "decompose-only: no card data, so no E_total, E_int or compensation ratio"
        )
        return [report]

    report.hw_partition = partition(hw)
    report.notes.extend(marker_notes(sim, hw))
    report.comparison = compare_partitions(
        report.label, mechanisms(sim), report.sim_partition, report.hw_partition
    )
    report.cpis = compare_cpi(sim, hw)
    return [report]


def _pct(x):
    return f"{100.0 * x:.2f} %"


def render(reports, decompose_only=False):
    out = []
    out.append("tt-sim baby RISC-V cycle attribution vs a card's, by zone")
    out.append("=" * 74)
    out.append("")
    out.append(
        "Every bucket's MAGNITUDE is hardware-attributed (an mcycle delta read by"
    )
    out.append("the core whose cycles it counts). Every bucket's MECHANISM LABEL is")
    out.append(
        "STRUCTURAL -- it comes from how retirebench's kernel is built, not from a"
    )
    out.append("hardware counter. What makes the label checkable is minstret: the same")
    out.append("binary must retire the same instructions in the same zone on both")
    out.append("sides, and retire_census_matches refuses it otherwise.")
    out.append("")
    if not reports:
        out.append("REFUSED: no retirebench artefact to analyse.")
        return "\n".join(out) + "\n"
    for report in reports:
        out.append(f"{report.label}")
        out.append("-" * 74)
        for gate in report.gates:
            out.append(gate.line())
        if report.refused:
            out.append("  REFUSED -- no comparison reported.")
            out.append("")
            continue
        out.append("")
        if decompose_only or not report.hw_partition:
            out.append(
                f"  outer window = {report.sim_window} cycles; two-marker cost "
                f"{report.sim_marker}"
            )
            out.append(
                f"  {'zone':<15}{'cycles':>10}{'% span':>9}{'retired':>10}"
                f"{'cyc/instr':>11}  mechanism (structural)"
            )
            span = max(1, report.sim_window)
            for name in report.sim_partition:
                value = report.sim_partition[name]
                retired = report.sim_retire.get(name, 0)
                cpi = report.sim_cpi.get(name)
                # Precomputed rather than nested inside the f-string below: a
                # nested same-quote f-string needs Python 3.12 and this project
                # targets 3.10.
                cpi_cell = "          -" if cpi is None else f"{cpi:>11.3f}"
                out.append(
                    f"  {name:<15}{value:>10}{100.0 * value / span:>8.2f} %"
                    f"{retired:>10}{cpi_cell}"
                    f"  {report.mechanism_labels.get(name, '')}"
                )
            out.append(
                f"  {'TOTAL':<15}{sum(report.sim_partition.values()):>10}{100.0:>8.2f} %"
                f"{sum(report.sim_retire.values()):>10}"
            )
            out.append("")
            for note in report.notes:
                out.append(f"  note: {note}")
            out.append("")
            continue
        out.append(
            f"  sim window {report.sim_window} cycles, card window {report.hw_window} cycles"
        )
        for note in report.notes:
            out.append(f"  note: {note}")
        out.append("")
        out.append(
            f"  {'zone':<15}{'sim':>11}{'card':>11}{'|diff|':>11}{'% span':>9}"
            f"  mechanism (structural)"
        )
        denom = max(1, report.hw_window)
        for name in report.comparison.mechanisms:
            s = report.sim_partition[name]
            h = report.hw_partition[name]
            out.append(
                f"  {name:<15}{s:>11}{h:>11}{abs(s - h):>11}"
                f"{100.0 * abs(s - h) / denom:>8.2f} %"
                f"  {report.mechanism_labels.get(name, '')}"
            )
        out.append("")
        comparison = report.comparison
        ratio = (
            "n/a (both exact)"
            if comparison.ratio is None
            else (
                "inf"
                if comparison.ratio == float("inf")
                else f"{comparison.ratio:.2f}x"
            )
        )
        out.append(
            f"  [{'PASS' if comparison.passed else 'FAIL'}] {comparison.unit} partition: "
            f"E_total = {_pct(comparison.e_total)} (limit {_pct(E_TOTAL_LIMIT)}), "
            f"E_int = {_pct(comparison.e_int)} (limit {_pct(E_INT_LIMIT)}), "
            f"compensation E_int/E_total = {ratio}"
        )
        out.append("")
        out.append(
            "  cycles per retired instruction, per zone. NOT an independent check:"
        )
        out.append(
            "  retire_census_matches has already made the two sides' retired counts"
        )
        out.append("  equal, so this is the partition divided by a per-zone constant.")
        out.append(f"  {'zone':<15}{'retired':>10}{'sim':>10}{'card':>10}{'err':>9}")
        for cpi in report.cpis:
            out.append(
                f"  {cpi.zone:<15}{cpi.retired:>10}{cpi.sim_cpi:>10.3f}"
                f"{cpi.hw_cpi:>10.3f}{100.0 * cpi.error:>8.1f}%"
                + ("" if cpi.passed else "   FAIL")
            )
        out.append("")
    if decompose_only:
        out.append("RESULT: DECOMPOSITION ONLY (no card data supplied)")
    else:
        verdict = all(r.passed for r in reports)
        out.append(f"RESULT: {'PASS' if verdict else 'FAIL'}")
    return "\n".join(out) + "\n"


def parse_core_map(value):
    """``"1,2=1,1"`` -> ``((1, 2), (1, 1))``, or ``None``."""
    if not value:
        return None
    try:
        left, right = value.split("=")
        sx, sy = (int(v) for v in left.split(","))
        cx, cy = (int(v) for v in right.split(","))
    except ValueError as exc:
        raise ValueError(
            f"--map-core expects SIMX,SIMY=CARDX,CARDY, got {value!r}"
        ) from exc
    return ((sx, sy), (cx, cy))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare tt-sim's per-zone baby RISC-V cycle attribution against a "
            "card's, from the retirebench artefact both sides emit."
        )
    )
    parser.add_argument(
        "--sim", required=True, help="retirebench-*.json from a run against tt-sim"
    )
    parser.add_argument(
        "--card", help="retirebench-*.json from the same binary on a card"
    )
    parser.add_argument(
        "--decompose-only",
        action="store_true",
        help="print the tt-sim decomposition alone; no card data, no criterion",
    )
    parser.add_argument(
        "--map-core",
        metavar="SIMX,SIMY=CARDX,CARDY",
        help="state a deliberate sim-core to card-core correspondence",
    )
    parser.add_argument("--report", help="write the text report here as well as stdout")
    parser.add_argument("--json", dest="json_path", help="write the report as JSON")
    args = parser.parse_args(argv)

    if not args.card and not args.decompose_only:
        parser.error("--card is required unless --decompose-only is given")
    if args.card and args.decompose_only:
        # Silently ignoring one of them would let a caller believe a criterion
        # was applied when it was not, or the reverse.
        parser.error("--decompose-only and --card are mutually exclusive")

    sim = load_run(args.sim)
    hw = load_run(args.card) if args.card else None
    mapping = parse_core_map(args.map_core)
    mapped = False
    if mapping is not None and hw is not None:
        if mapping[0] != sim.core or mapping[1] != hw.core:
            parser.error(
                f"--map-core {args.map_core} does not describe these two artefacts "
                f"(sim core {sim.core}, card core {hw.core})"
            )
        mapped = True
    reports = analyse(sim, hw, mapped=mapped)
    text = render(reports, decompose_only=hw is None)
    print(text, end="")
    if args.report:
        Path(args.report).write_text(text)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "e_total_limit": E_TOTAL_LIMIT,
                    "e_int_limit": E_INT_LIMIT,
                    "cpi_tolerance": CPI_TOLERANCE,
                    "decompose_only": hw is None,
                    "runs": [r.to_dict() for r in reports],
                },
                indent=2,
            )
            + "\n"
        )
    if hw is None:
        return 0 if reports and not any(r.refused for r in reports) else 1
    return 0 if reports and all(r.passed for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
