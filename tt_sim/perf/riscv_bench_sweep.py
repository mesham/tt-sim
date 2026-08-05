"""The baby RISC-V front end, measured, swept against the tables.

``docs/plans/cost-model.md`` wires six of the nine Tensix backend units, and
**every single one moved zero simulated cycles**. The diagnosis is the same each
time and the file states it outright: *"the constraint is the un-modelled RISC-V
front end"*. ``perfbench/tensixbench`` said the same thing from the measurement
side -- against tt-sim every phase A probe of every unit at every data format
reads exactly ``1.000`` cycles per instruction, because nothing back-pressures
the issuing core.

So this module consumes the dataset that would settle it: ``perfbench/
riscvbench``, one tt-metal program that runs unchanged on silicon and against
tt-sim, timing the front end itself. The methodology, and what each measurement
can and cannot establish, is in ``docs/plans/riscv-front-end-benchmark.md``.

What the input is
-----------------

``perfbench/riscvbench`` writes a CSV of **raw points**, never a cost::

    phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles

* **phases R / T / C / F** -- ``n`` unrolled blocks of ``unroll`` instructions,
  timestamped off ``RISCV_DEBUG_REG_WALL_CLOCK_L``, the same register tt-metal's
  device profiler reads. ``variant`` is ``t1``/``t2``/``t3``: how many TRISCs ran
  the identical probes at once.
* **phase Q** -- ``n`` is a *burst length* (1, 2, 4 ... 128) rather than a block
  count, and ``unroll`` is 1. It is looking for a **knee**, not a slope, and is
  reported and gated separately for that reason: a straight line there would be
  the null result rather than the healthy one. Its read-out is a *rate over a
  wide baseline* rather than a point-by-point control subtraction; the silicon
  run that forced that change is described at :func:`_queue_check`.

Every cost below is a **slope** over ``n``, so the fixed cost of the two clock
reads, the barrier and the surrounding call cancels exactly. The
``loop_overhead`` probe -- the identical loop with a body that emits no
instructions -- is then subtracted so the RISC-V loop counter and branch cancel
too::

    cycles_per_instruction = (slope(probe) - slope(loop_overhead)) / unroll

Run it
------

::

    python3 -m tt_sim.perf.riscv_bench_sweep
    python3 -m tt_sim.perf.riscv_bench_sweep --measured hw.csv
    python3 -m tt_sim.perf.riscv_bench_sweep --measured hw.csv --reference sim.csv

With no ``--measured`` the sweep reads the **primary tracked reference
measurement** (:data:`PRIMARY_DATASET`) in ``tt_sim/perf/datasets/``, so the
comparison reproduces with no hardware and no arguments. Each dataset's ``#``
header carries its own provenance -- card, firmware, KMD, flags, row count,
per-phase validity -- because a measurement separated from those is not one.

Not every tracked dataset is a peer of the primary, and one of them is not a
result about the *device* at all. :data:`MIN_BLOCKS_DATASET` is the same binary
at ``--blocks 8`` instead of 32, and it is kept precisely **because every one of
its phases failed the benchmark's own validity gate**: it establishes the
instrument's minimum usable block count, which is a fact about the instrument
and is invisible from a run that passed. The sweep will never choose it for you.
If neither can be found the script prints where it looked and exits 0 -- the
same "degrade gracefully" contract ``noc_dataset_sweep`` and
``tensix_bench_sweep`` use.

With ``--reference`` it additionally diffs two runs of the same binary --
silicon against tt-sim -- which is the differential form ``optests/diff.sh``
established for values, applied to cycles.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# The dataset.
# ---------------------------------------------------------------------------

#: The control probe. Its slope is the RISC-V block-loop overhead, subtracted
#: from every other slope. Not a measurement of anything.
CONTROL_PROBE = "loop_overhead"

#: Phase Q's own control: the burst cascade's compare-and-branch cost. It is
#: **no longer a subtrahend** -- see :func:`_queue_check` for the silicon run
#: that retired the subtraction -- but it is still not a datum, so the exclusion
#: ladder drops it exactly as it drops ``loop_overhead``. What it is now is the
#: phase's declared **noise floor**: its spread across the eight burst lengths
#: is how much a single phase-Q point can be wrong by.
QUEUE_CONTROL_PROBE = "q_ctrl"

#: The smallest phase-Q burst whose reading is interpretable. Below it the
#: cascade's own fixed cost -- two clock reads, a barrier, seven
#: compare-and-branch pairs -- is comparable to the entire measurement, and
#: :data:`QUEUE_CONTROL_PROBE` measured that fixed cost varying by up to ~17
#: cycles between burst lengths on silicon. A rate computed off a point below
#: this is noise with a decimal point on it.
QUEUE_MIN_N = 16

#: Least-squares fit quality below which a slope is not a slope. The benchmark
#: reports the same number and refuses the run; this is the second gate, for a
#: CSV that arrives from somewhere else.
MIN_R2 = 0.99

#: Where tracked reference measurements live. Silicon only, each carrying its
#: provenance in its own ``#`` header.
DATASET_DIR = Path(__file__).resolve().parent / "datasets"

#: The PRIMARY tracked dataset, read when ``--measured`` is omitted. Named
#: rather than inferred, for the same reason ``tensix_bench_sweep`` names its
#: own: the directory holds more than one ``riscvbench-*.csv`` and they are not
#: peers. See :data:`MIN_BLOCKS_DATASET`.
PRIMARY_DATASET = "riscvbench-blackhole.csv"

#: The ``--blocks 8`` companion run. It is tracked **because it failed**: every
#: one of its five phases was refused by the benchmark's own validity gate,
#: almost all of it because the ``loop_overhead`` CONTROL itself fits to
#: R^2 < 0.99 at four points spanning n = 8..32. That is not a statement about
#: the device -- its per-instruction numbers agree with the primary run to a few
#: thousandths of a cycle wherever both are readable -- it is the measurement
#: that puts a floor under ``--blocks``. Sweeping it as if it were a result is
#: exactly the mistake it exists to prevent, so the default is a choice made
#: here rather than whatever sorts first.
MIN_BLOCKS_DATASET = "riscvbench-blackhole-blocks8.csv"

#: How many standard errors of the fitted slope count as "the fit cannot tell".
#: Two, i.e. ~95 %, which is the ordinary convention and is written here rather
#: than inlined so that widening it is a visible edit.
RESOLUTION_SIGMA = 2.0

#: Which probes ``tt_sim/pe/rv/cost.py`` actually charges today. Its module
#: docstring is the authority and lists three things it models -- the load-use
#: interlock, the L1 store rate and the integer unit's multiply/divide -- and
#: four it deliberately does not: branch mispredicts ("neither the docs nor
#: tt-sim describe the predictor"), instruction fetch and i-cache misses,
#: sustained-load throughput, and regions the load-latency table does not name.
#: Nothing anywhere charges the ``.ttinsn`` push.
#:
#: This is what makes the sweep's "wired" axis mean something: a probe can carry
#: a perfectly good table prediction and still be a measurement of tt-sim's
#: silence rather than of its arithmetic.
TT_SIM_CHARGES = frozenset(
    {
        "rv_mul_indep",
        "rv_mul_dep",
        "rv_div",
        "rv_load_chase",
        "rv_load_stack",
        "rv_store_spread",
        "rv_store_coalesce",
        "rv_store_stack",
    }
)

#: MMIO base, from ``tt_sim/pe/rv/cost.py``. Used only to say, in the report,
#: which load-latency row the stack probe landed in.
_MMIO_BASE = 0xFFB00000


def reference_datasets():
    """Every tracked reference measurement, sorted by path."""
    if not DATASET_DIR.is_dir():
        return []
    return sorted(DATASET_DIR.glob("riscvbench-*.csv"))


def default_measured_path(arch=None):
    """The tracked dataset to sweep when ``--measured`` is not given.

    Never :data:`MIN_BLOCKS_DATASET`, whatever ``arch`` asks for: that file is a
    deliberately-invalid run kept as evidence about the instrument.
    """
    if arch is not None:
        candidate = DATASET_DIR / f"riscvbench-{arch}.csv"
        return candidate if candidate.exists() else None
    primary = DATASET_DIR / PRIMARY_DATASET
    if primary.exists():
        return primary
    found = [p for p in reference_datasets() if p.name != MIN_BLOCKS_DATASET]
    return found[0] if len(found) == 1 else None


def read_csv(path):
    """``(rows, meta)``. ``meta`` carries the ``# arch=...`` comment line."""
    meta = {}
    lines = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            for token in line.lstrip("#").split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    meta[key] = value
            continue
        lines.append(line)
    rows = []
    for raw in csv.DictReader(lines):
        rows.append(
            {
                "phase": raw["phase"],
                "variant": raw["variant"],
                "probe_id": int(raw["probe_id"]),
                "probe": raw["probe"],
                "unit": raw["unit"],
                "active_threads": int(raw["active_threads"]),
                "thread": int(raw["thread"]),
                "n": int(raw["n"]),
                "unroll": int(raw["unroll"]),
                "cycles": int(raw["cycles"]),
            }
        )
    return rows, meta


# ---------------------------------------------------------------------------
# What is predicted, and on what basis. DECLARED BEFORE ANY RESIDUAL.
# ---------------------------------------------------------------------------

RESIDUAL_EXPECTATION = """\
Three things are predicted, and they are not equally strong. The report keeps
them apart because conflating them is how an exploratory null gets read as a
confirmation.

1. A FLOOR OF ONE CYCLE, structurally. A baby RISC-V core is "in-order
   single-issue ... intended to execute one RV32IM instruction per cycle"
   (riscv.issue, isa_doc). Every probe here is one instruction per issue slot,
   so NO probe can legitimately read below 1.0 cycles per instruction. A value
   like 0.4 means the timer, not the instruction, is being measured.

2. WHERE THE TABLES MAKE A CLAIM, the claim is a floor and not an equality.
   Bounded entries (`at_least`, `range`) are read at their low end, exactly as
   `tt_sim/perf/model.py` charges them, so measured >= predicted. A NEGATIVE
   residual beyond the instrument's resolution is the interesting failure: it
   means the table over-charges, and invents back-pressure the hardware does not
   have.

3. WHERE THEY DO NOT, the probe is EXPLORATORY and is excluded from the residual
   ladder rather than compared to a plausible-looking 1. Four groups are in this
   class and each is here for a stated reason:

     * the Tensix instruction queue's DEPTH (phase Q). No document in either
       tree gives one. The knee is the measurement.
     * the branch PREDICTOR (phase C). The tables give the mispredict BUBBLE --
       2 cycles on Wormhole, 4 on Blackhole -- which is how much one costs. How
       OFTEN one happens is undescribed, which is exactly why
       `tt_sim/pe/rv/cost.py` declines to charge it.
     * the instruction CACHE (phase F). The docs give the fetch period and no
       cache size and no miss cost.
     * unconditional jumps (`c_jal`). No entry of any kind.

AND ONE THING THAT IS NOT A PREDICTION AT ALL, but is the axis that matters
most for tt-sim. Only eight of these probes are consumed by
`tt_sim/pe/rv/cost.py`; the rest measure a term the simulator does not model.
Against tt-sim those eight are the instrument's own calibration -- they are the
reason a run of 1.000s elsewhere is a FINDING rather than the signature of a
benchmark that measured nothing. The `wired` column says which is which."""


class Prediction:
    """What the tables say a probe should cost, and how firmly.

    ``kind`` is ``predicted`` when a table field says it, ``derived`` when it is
    arithmetic on table fields (with the arithmetic in ``derivation``), and
    ``exploratory`` when nothing in either tree gives a number -- in which case
    ``cycles`` is None and the probe is excluded from the residual ladder.
    """

    __slots__ = ("cycles", "bound", "path", "derivation", "kind", "note")

    def __init__(
        self,
        cycles=None,
        bound=None,
        path=None,
        derivation=None,
        kind="predicted",
        note=None,
    ):
        self.cycles = cycles
        self.bound = bound
        self.path = path
        self.derivation = derivation
        self.kind = kind
        self.note = note


def _cycles_of(entry):
    """``(cycles, bound)`` for a table value that may be an int or a mapping."""
    if entry is None:
        return None, None
    if isinstance(entry, dict):
        return entry.get("cycles"), entry.get("bound")
    return entry, "exact"


def _latency_key(arch, region):
    """Which ``riscv.load_latency`` key ``arch`` uses for a canonical region.

    Read out of ``tt_sim/perf/model.py``'s own private mapping rather than
    restated, and deliberately so: Blackhole's table is not a superset of
    Wormhole's -- it renames rows, splits L1 into a d-cache hit and a miss, and
    moves TDMA between groups -- so a second copy of that mapping here would be
    a place for the prediction and the simulator to drift apart silently. The
    same reasoning ``tensix_bench_sweep.unwired_units`` uses for reading its
    list out of the test that owns it.
    """
    from tt_sim.perf.model import _LOAD_LATENCY_KEYS

    return _LOAD_LATENCY_KEYS[arch].get(region)


def _load_row(riscv, arch):
    """``(key, raw entry)`` for an L1 access on ``arch``."""
    from tt_sim.perf.model import RV_REGION_L1

    key = _latency_key(arch, RV_REGION_L1)
    return key, (riscv.get("load_latency") or {}).get(key)


def predictions(arch, meta=None):
    """``{probe: Prediction}``, read out of ``unit_costs.yaml`` for ``arch``.

    Everything numeric here comes from the shipped tables through the ordinary
    loader. Nothing is hardcoded: doubling a field in the YAML moves the
    prediction, which is what a test asserts.
    """
    from tt_sim.perf.costs import load_costs

    table = load_costs(arch)
    riscv = table.section("riscv")
    integer = riscv.get("integer_unit") or {}
    stores = riscv.get("store_throughput") or {}
    loads = riscv.get("load_throughput") or {}
    fusion = riscv.get("ttinsn_fusion") or {}
    issue = (riscv.get("issue") or {}).get("instructions_per_cycle")
    pad = int((meta or {}).get("pad", 16))

    out = {}

    def add(probe, cycles, bound, path, kind="predicted", derivation=None, note=None):
        out[probe] = Prediction(cycles, bound, path, derivation, kind, note)

    # -- phase R -----------------------------------------------------------
    if issue is not None:
        add(
            "rv_addi_indep", float(issue), "exact", "riscv.issue.instructions_per_cycle"
        )
        add(
            "rv_addi_dep",
            float(issue),
            "exact",
            "riscv.issue.instructions_per_cycle + riscv.integer_unit.alu_forwarding",
            note="the forwarding path is what makes a dependent chain cost the same "
            "as an independent one; without it this probe would be the ALU latency",
        )
    mul, mul_bound = _cycles_of(integer.get("multiply"))
    if mul is not None:
        add("rv_mul_indep", float(mul), mul_bound, "riscv.integer_unit.multiply")
        ex2, _ = _cycles_of(integer.get("multiply_ex2"))
        if ex2 is None:
            add(
                "rv_mul_dep",
                float(mul),
                mul_bound,
                "riscv.integer_unit.multiply",
                note="Wormhole's multiply BLOCKS the integer unit, so a dependent "
                "chain costs the same as an independent one",
            )
        else:
            add(
                "rv_mul_dep",
                float(mul) + float(ex2),
                "at_least",
                "riscv.integer_unit.multiply + riscv.integer_unit.multiply_ex2",
                kind="derived",
                derivation=f"{mul} (EX1) + {ex2} (EX2) = {mul + ex2}. Blackhole's "
                "multiply pipelines across two stages, so its OCCUPANCY is one "
                "cycle but its LATENCY is two, and a dependent chain pays the "
                "latency. tt-sim charges the occupancy only.",
            )
    div, div_bound = _cycles_of(integer.get("divide_general"))
    if div is not None:
        add(
            "rv_div",
            float(div),
            div_bound,
            "riscv.integer_unit.divide_general",
            note="charged at the low end of the documented 6-33 range, as "
            "model.py charges every bound; the benchmark's dividend and divisor "
            "are in the CSV header",
        )
    row_key, row = _load_row(riscv, arch)
    lat, lat_bound = _cycles_of(row)
    if lat is not None:
        add("rv_load_chase", float(lat), lat_bound, f"riscv.load_latency.{row_key}")
        under = loads.get("one_per_cycle_if_latency_under")
        per_window = loads.get("else_loads_per_window")
        offset = loads.get("else_window_cycles_offset")
        if under is not None and per_window and offset is not None:
            if lat < under:
                add(
                    "rv_load_indep",
                    1.0,
                    "exact",
                    "riscv.load_throughput.one_per_cycle_if_latency_under",
                    kind="derived",
                    derivation=f"load latency {lat} < {under}, so the docs' "
                    f'"throughput of sustained loads is one per cycle" applies.',
                )
            else:
                value = (lat + offset) / float(per_window)
                add(
                    "rv_load_indep",
                    value,
                    "at_least",
                    "riscv.load_throughput",
                    kind="derived",
                    derivation=f"latency {lat} >= {under}, so the docs give "
                    f"{per_window} loads every {lat} - 1 cycles, i.e. "
                    f"({lat} + {offset}) / {per_window} = {value:.3f} cycles/load.",
                )
    l1_store, l1_store_bound = _cycles_of(stores.get("l1_period_cycles"))
    if l1_store is not None:
        add(
            "rv_store_spread",
            float(l1_store),
            l1_store_bound,
            "riscv.store_throughput.l1_period_cycles",
        )
        coalesced, coalesced_bound = _cycles_of(
            stores.get("l1_coalesced_period_cycles")
        )
        if coalesced is None:
            add(
                "rv_store_coalesce",
                float(l1_store),
                l1_store_bound,
                "riscv.store_throughput.l1_period_cycles",
                note="this architecture publishes no coalescing store queue, so "
                "the two store probes are predicted IDENTICAL. On the one that "
                "does they are predicted 5x apart, which makes the pair a "
                "cross-architecture discriminator rather than two readings.",
            )
        else:
            add(
                "rv_store_coalesce",
                float(coalesced),
                coalesced_bound,
                "riscv.store_throughput.l1_coalesced_period_cycles",
                note="the probe's stores are four bytes apart inside one "
                "16-byte-aligned block, which is exactly the documented "
                "coalescing predicate",
            )
    # The stack probes' region is a tt-metal placement decision, read from the
    # header rather than assumed. Classified with the simulator's own
    # classifier so the two cannot disagree.
    stack_addr = (meta or {}).get("stack_addr")
    if stack_addr is not None:
        try:
            addr = int(stack_addr, 0)
        except ValueError:
            addr = None
        if addr is not None:
            from tt_sim.pe.rv.cost import classify_address
            from tt_sim.perf.model import RV_REGION_NAMES

            region = classify_address(addr)
            key = _latency_key(arch, region)
            cycles, bound = _cycles_of((riscv.get("load_latency") or {}).get(key))
            if cycles is not None:
                add(
                    "rv_load_stack",
                    float(cycles),
                    bound,
                    f"riscv.load_latency.{key}",
                    note=f"the stack landed at {stack_addr}, which "
                    f"tt_sim/pe/rv/cost.classify_address puts in "
                    f"{RV_REGION_NAMES[region]}",
                )
            other, other_bound = _cycles_of(stores.get("other_regions_period_cycles"))
            if addr >= _MMIO_BASE and other is not None:
                add(
                    "rv_store_stack",
                    float(other),
                    other_bound,
                    "riscv.store_throughput.other_regions_period_cycles",
                )
            elif l1_store is not None:
                add(
                    "rv_store_stack",
                    float(l1_store),
                    l1_store_bound,
                    "riscv.store_throughput.l1_period_cycles",
                )

    # -- phase T -----------------------------------------------------------
    dequeue = fusion.get("dequeue_per_thread_per_cycle")
    max_fused = fusion.get("max_fused")
    if dequeue:
        for probe in ("tt_nop", "tt_sfpnop", "tt_setdmareg"):
            add(
                probe,
                1.0 / float(dequeue),
                "at_least",
                "riscv.ttinsn_fusion.dequeue_per_thread_per_cycle",
                kind="derived",
                derivation=f"the queue drains at most {dequeue} instruction per "
                "thread per cycle, so a SUSTAINED burst cannot go faster than "
                f"{1.0 / float(dequeue):.3f} cycles each whatever the core's push "
                "rate. This probe therefore cannot see fusion; the group probes "
                "are what can.",
            )
    thcon = table.instruction("THCON", "ADDDMAREG")
    if thcon is not None and thcon.occupancy is not None:
        add(
            "tt_adddmareg",
            float(
                max(thcon.occupancy.cycles, 1.0 / float(dequeue) if dequeue else 1.0)
            ),
            thcon.occupancy.bound,
            "tensix_instruction_costs THCON.ADDDMAREG.occupancy",
            kind="derived",
            derivation="a sustained burst runs at the slower of the queue's drain "
            f"rate and the unit's occupancy ({thcon.occupancy.cycles}); this "
            "probe is the one place a Tensix occupancy is visible to a "
            "device-side clock at all.",
        )
    if issue is not None:
        add(
            "tt_pad",
            float(pad) / float(issue),
            "exact",
            "riscv.issue.instructions_per_cycle",
            kind="derived",
            derivation=f"{pad} plain RV instructions at {issue} per cycle. This is "
            "the group probes' own baseline: the other three carry the same "
            f"{pad} and differ only in the `.ttinsn` words among them.",
        )
        if max_fused:
            for probe, count in (("tt_fuse2", 2), ("tt_fuse4", 4)):
                fused = math.ceil(count / float(max_fused))
                add(
                    probe,
                    float(pad) / float(issue) + fused,
                    "at_least",
                    "riscv.ttinsn_fusion.max_fused",
                    kind="derived",
                    derivation=f"{count} ADJACENT `.ttinsn` words fuse into "
                    f"ceil({count} / {max_fused}) = {fused} issue slot(s), on top "
                    f"of the group's {pad} plain instructions. WITHOUT fusion the "
                    f"same group would cost {pad + count}, so the residual against "
                    "this prediction IS the fusion question.",
                )
            add(
                "tt_spread4",
                float(pad) / float(issue) + 4.0,
                "at_least",
                "riscv.ttinsn_fusion.max_fused",
                kind="derived",
                derivation="four `.ttinsn` words with none adjacent, so there is "
                f"nothing to fuse and each takes its own issue slot: {pad} + 4. "
                "This probe is predicted the SAME with or without fusion, which is "
                "what makes it the control for `tt_fuse4`.",
            )

    # -- phase C -----------------------------------------------------------
    branch, branch_bound = _cycles_of(integer.get("branch"))
    if branch is not None and issue is not None:
        add(
            "c_ctrl_xor",
            float(issue),
            "exact",
            "riscv.issue.instructions_per_cycle",
        )
        add(
            "c_nt",
            float(branch),
            branch_bound,
            "riscv.integer_unit.branch",
            note="a NOT-taken branch. The tables' single `branch: 1` is the "
            "correctly-predicted cost; whether not-taken is the prediction is "
            "what this probe and `c_t` between them say.",
        )
        add(
            "c_xor_nt",
            float(issue) + float(branch),
            branch_bound,
            "riscv.issue + riscv.integer_unit.branch",
            kind="derived",
            derivation=f"one `xori` at {issue} cycle plus one correctly-predicted "
            f"branch at {branch}.",
        )
    observed, _ = _cycles_of(integer.get("branch_mispredict_observed"))
    bubble, _ = _cycles_of(integer.get("branch_mispredict_bubble"))
    for probe in ("c_t", "c_xor_t", "c_xor_alt"):
        add(
            probe,
            None,
            None,
            "riscv.integer_unit.branch_mispredict_bubble",
            kind="exploratory",
            note=f"the tables give the SIZE of a mispredict ({bubble}-cycle bubble, "
            f"looking like {observed} cycles from outside) and nothing about the "
            "predictor, so how often this probe mispredicts is undescribed. The "
            f"prediction is only that it lies between {branch} and {observed} "
            "cycles per branch, and where in that interval is the measurement.",
        )
    add(
        "c_jal",
        None,
        None,
        None,
        kind="exploratory",
        note="no table entry covers an unconditional jump. It is here because a "
        "jump is taken by definition and needs no prediction, so it separates "
        '"taken costs extra" from "a wrong prediction costs extra".',
    )

    # -- phase F -----------------------------------------------------------
    fetch = riscv.get("instruction_fetch") or {}
    period = fetch.get("expected_period_cycles")
    for probe in ("f_64", "f_128", "f_256", "f_512", "f_1024", "f_2048"):
        add(
            probe,
            None,
            None,
            "riscv.instruction_fetch",
            kind="exploratory",
            note="the docs give the fetch PERIOD (one 128-bit L1 read per four "
            f"instructions, expected at most once every {period} cycles) and its "
            'own note says "instruction cache miss cost is not published". So a '
            "flat row across footprints is consistent with the documentation and a "
            "cliff is a number nothing has ever printed. Both are results.",
        )
    return out


# ---------------------------------------------------------------------------
# The exclusion criteria. DECLARED BEFORE ANY RESIDUAL WAS COMPUTED.
# ---------------------------------------------------------------------------
#
# Same discipline, and the same ordering, as ``noc_dataset_sweep._exclusions``
# and ``tensix_bench_sweep._exclusions``: dropping a series because it
# *disagrees* is fitting, not validating. Every rule below names something the
# comparison is structurally unable to ask, and every one could have been
# written before the benchmark was ever run.


def _exclusions():
    """The retained-set predicate for the per-instruction sweep, as a ladder."""
    return [
        (
            "probe is a control",
            "`loop_overhead` and `q_ctrl` are subtrahends, not data. Their "
            "slopes are what everything else is reported relative to.",
            lambda s: s["probe"] not in (CONTROL_PROBE, QUEUE_CONTROL_PROBE),
        ),
        (
            "phase == Q",
            "phase Q sweeps BURST LENGTH looking for a knee, not block count "
            "looking for a slope. A least-squares line through it would be "
            "meaningless by construction. Reported separately.",
            lambda s: s["phase"] != "q",
        ),
        (
            "active_threads > 1",
            "a contended measurement is a different quantity from a "
            "per-instruction cost: it is the shared resource's throughput. "
            "Reported separately as the issue-limit discriminator.",
            lambda s: s["active_threads"] == 1,
        ),
        (
            f"R^2 < {MIN_R2:.2f}",
            "a series that is not linear in the block count has no slope to "
            "read. The benchmark refuses such a run outright; this catches a "
            "CSV that arrived from elsewhere.",
            lambda s: s["r2"] >= MIN_R2,
        ),
        (
            "no prediction in the tables",
            "the branch predictor, the Tensix queue depth, the instruction "
            "cache and unconditional jumps have no published number of any "
            "kind. Those probes are EXPLORATORY and are read out on their own "
            "terms below rather than compared against a plausible-looking 1.",
            lambda s: s["predicted"] is not None,
        ),
    ]


def retained(series):
    """``(kept, [(rule, removed, remaining), ...])`` -- the ladder, in order."""
    kept, ladder = series, []
    for name, _reason, keep in _exclusions():
        nxt = [s for s in kept if keep(s)]
        ladder.append((name, len(kept) - len(nxt), len(nxt)))
        kept = nxt
    return kept, ladder


# ---------------------------------------------------------------------------
# Fitting.
# ---------------------------------------------------------------------------


def linear_fit(xs, ys):
    """``(intercept, slope, r2)``; ``(mean, 0, 1)`` for a degenerate x."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0), 0.0, 1.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    syy = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sxx == 0:
        return mean_y, 0.0, 1.0
    slope = sxy / sxx
    r2 = 1.0 if syy == 0 else (sxy * sxy) / (sxx * syy)
    return mean_y - slope * mean_x, slope, r2


def slope_stderr(xs, ys, intercept, slope):
    """Standard error of the fitted slope; 0.0 when the fit is exact.

    R^2 does not answer the question this needs answering. Four points can sit
    on a line to R^2 = 0.9999 and still leave the slope uncertain in the third
    decimal, which is the size of the discrepancies this sweep adjudicates.
    """
    n = len(xs)
    if n < 3:
        return 0.0
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return 0.0
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return (sse / ((n - 2) * sxx)) ** 0.5


#: Why a small negative residual is not evidence of anything.
FIT_RESOLUTION_NOTE = """\
A residual is only a finding if it is bigger than what the instrument can
resolve, and this instrument has two known one-sided biases, both of which push
the measured value DOWN. They are added per series and reported as `resol`:

  1. CONTROL OVER-SUBTRACTION, worth up to slope(loop_overhead)/unroll. The
     control's cycles are the RISC-V loop counter, compare and branch. Those are
     additive only while the core is ISSUING; the moment something stalls it --
     an L1 store's five-cycle period, a load-use interlock, a full Tensix queue
     -- the loop's own instructions issue inside the stall window and cost
     nothing. The subtraction is therefore exactly right for a probe that runs
     at one instruction per cycle and up to slope(control)/unroll too much for
     one that stalls, and which of those a probe does is the thing being
     measured, so the correction cannot be applied selectively.

     This is not a theoretical worry. Against tt-sim with the cost model on, the
     `rv_store_spread` probe's raw slope is EXACTLY 320 cycles per block of 64
     stores -- 5.000 each, with no room in it for the loop's two cycles -- and
     the unconditional subtraction reports 4.969. Same arithmetic, same size,
     same direction as the one `tensix_bench_sweep` documents for its
     unit-limited probes.

  2. FIT UNCERTAINTY, {sigma:.0f} standard errors of the fitted slope (plus the
     control's, in quadrature), divided by unroll. Silicon's first burst is not
     like its later ones -- cold i-cache, an unfilled Tensix queue, a DVFS
     transition -- and a warm-up offset on the smallest n tilts a four-point
     least-squares fit.

So: residual >= 0 confirms the floor; residual within `resol` of zero is BELOW
the prediction but INSIDE the instrument, and is reported as such rather than as
an over-charge; only a residual beyond `resol` is a claim about the hardware.
The price is that this instrument cannot detect an over-charge smaller than
`resol`, which is a fraction of a cycle."""


def series_of(rows):
    """Collapse raw points into one fitted series per measurement.

    A series is ``(phase, variant, probe, thread)``; its slope is cycles per
    ``n``, which is per block everywhere except phase Q, where ``n`` is a burst
    length and the slope is not the quantity of interest.
    """
    groups = {}
    for row in rows:
        key = (row["phase"], row["variant"], row["probe"], row["thread"])
        groups.setdefault(key, []).append(row)
    out = []
    for (phase, variant, probe, thread), points in groups.items():
        points.sort(key=lambda r: r["n"])
        xs = [p["n"] for p in points]
        ys = [float(p["cycles"]) for p in points]
        intercept, slope, r2 = linear_fit(xs, ys)
        out.append(
            {
                "phase": phase,
                "variant": variant,
                "probe": probe,
                "thread": thread,
                "unit": points[0]["unit"],
                "unroll": points[0]["unroll"],
                "active_threads": points[0]["active_threads"],
                "points": points,
                "intercept": intercept,
                "slope": slope,
                "r2": r2,
                "stderr": slope_stderr(xs, ys, intercept, slope),
            }
        )
    out.sort(key=lambda s: (s["phase"], s["variant"], s["probe"], s["thread"]))
    return out


def apply_control(series):
    """Subtract the control probe's slope and divide by the probe's unroll.

    The control is matched on ``(phase, variant, thread)``: every phase is its
    own program launch with its own compiled kernel, so a phase's loop overhead
    is measured in the same binary as the probes it corrects.
    """
    control = {
        (s["phase"], s["variant"], s["thread"]): s
        for s in series
        if s["probe"] == CONTROL_PROBE
    }
    for s in series:
        base = control.get((s["phase"], s["variant"], s["thread"]))
        s["control"] = None if base is None else base["slope"]
        if base is None or s["phase"] == "q":
            s["measured"] = None
            s["resolution"] = None
            continue
        unroll = float(s["unroll"])
        s["measured"] = (s["slope"] - base["slope"]) / unroll
        se_diff = (s["stderr"] ** 2 + base["stderr"] ** 2) ** 0.5
        s["resolution"] = (base["slope"] + RESOLUTION_SIGMA * se_diff) / unroll
    return series


def attach_predictions(series, arch, meta=None):
    """Attach each probe's table prediction, or mark it exploratory."""
    table = predictions(arch, meta)
    for s in series:
        pred = table.get(s["probe"])
        s["prediction"] = pred
        s["predicted"] = None if pred is None else pred.cycles
        s["bound"] = None if pred is None else pred.bound
        s["kind"] = "none" if pred is None else pred.kind
        s["wired"] = s["probe"] in TT_SIM_CHARGES
    return series


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _summary(values):
    values = sorted(values)
    if not values:
        return None
    quartile = len(values) // 4
    return {
        "n": len(values),
        "min": values[0],
        "p25": values[quartile],
        "median": statistics.median(values),
        "p75": values[-1 - quartile],
        "max": values[-1],
    }


def _grouped(rows, key_fn):
    groups = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


def report(rows, arch, out=None, label="measured", reference=None, meta=None):
    """The whole sweep. Returns the retained per-instruction series."""
    out = sys.stdout if out is None else out

    def emit(line=""):
        print(line, file=out)

    series = attach_predictions(apply_control(series_of(rows)), arch, meta)
    kept, ladder = retained(series)

    emit("=" * 78)
    emit(f"The RISC-V front end: measured against the tables [{arch}]")
    emit("=" * 78)
    emit()
    emit(f"input: {len(rows)} raw points -> {len(series)} fitted series ({label})")
    for key in ("phases", "variants", "base_blocks", "stack_addr"):
        if meta and meta.get(key):
            emit(f"  {key} = {meta[key]}")
    emit()
    emit("Exclusion ladder (declared before any residual was computed):")
    total = len(series)
    for name, removed, remaining in ladder:
        emit(f"  - {name:<44} removes {removed:>4}  leaves {remaining:>4}")
    emit()
    emit(f"retained: {len(kept)} of {total} series")
    emit()
    emit(RESIDUAL_EXPECTATION)
    emit()

    # -- the per-instruction table -----------------------------------------
    emit("-" * 78)
    emit("Per probe: measured cycles vs what the tables predict, single thread")
    emit("-" * 78)
    emit(
        f"{'probe':<18}{'ph':<3}{'unit':<9}{'pred':>7}{'bound':>10}"
        f"{'measured':>10}{'residual':>10}{'resol':>8}  {'kind':<12}{'wired'}"
    )
    for s in sorted(kept, key=lambda s: (s["phase"], s["unit"], s["probe"])):
        s["residual"] = s["measured"] - s["predicted"]
        s["resolved"] = s["residual"] < -(s["resolution"] or 0.0)
        emit(
            f"{s['probe']:<18}{s['phase']:<3}{s['unit']:<9}{s['predicted']:>7.2f}"
            f"{s['bound'] or '':>10}{s['measured']:>10.3f}{s['residual']:>10.3f}"
            f"{s['resolution'] or 0.0:>8.3f}  {s['kind']:<12}"
            f"{'yes' if s['wired'] else 'no'}"
        )

    if not kept:
        emit("  nothing retained; no per-instruction cost to check.")
    else:
        _floor_verdict(kept, emit)
        _by_axis(kept, emit)

    _exploratory_readout(series, emit)
    _fusion_check(series, emit, meta)
    _branch_check(series, arch, emit)
    _queue_check(rows, emit)
    _fetch_check(series, emit)
    _issue_limit_check(series, emit)
    _live_check(series, emit)
    if reference is not None:
        _differential(rows, reference, arch, emit, meta)
    return kept


def _floor_verdict(kept, emit):
    emit()
    emit("-" * 78)
    emit("Is the prediction a floor?")
    emit("-" * 78)
    emit(FIT_RESOLUTION_NOTE.format(sigma=RESOLUTION_SIGMA))
    emit()
    at_or_above = [s for s in kept if s["residual"] >= -1e-9]
    within = [s for s in kept if -1e-9 > s["residual"] and not s["resolved"]]
    over = [s for s in kept if s["resolved"]]
    if at_or_above:
        emit(
            f"  {len(at_or_above)} series at or above the prediction: "
            f"{', '.join(sorted(s['probe'] for s in at_or_above))}"
        )
    if within:
        stats = _summary([s["residual"] for s in within])
        emit(
            f"  {len(within)} series below the prediction but INSIDE the "
            f"instrument's resolution\n"
            f"  (worst {stats['min']:+.3f} cycles). Not an over-charge; not a "
            f"finding:\n    {', '.join(sorted(s['probe'] for s in within))}"
        )
    emit()
    if not over:
        emit(
            "  VERDICT: yes. No residual is below the prediction by more than the\n"
            "  fit can resolve, which is the direction every bound in these tables\n"
            "  is chosen to lean."
        )
    else:
        emit("  VERDICT: NO. The tables OVER-PREDICT these, beyond the resolution:")
        emit(
            "  -- but read 'Is the instrument live?' at the end of this report FIRST.\n"
            "  A device on which nothing back-pressures the issuing core makes every\n"
            "  probe read 1.000 and therefore makes every prediction above 1 look like\n"
            "  an over-charge. That is what tt-sim looks like with TT_SIM_COST_MODEL\n"
            "  unset, and it is a statement about the simulator rather than about the\n"
            "  tables."
        )
        for s in sorted(over, key=lambda s: s["residual"]):
            emit(
                f"    {s['probe']:<18} predicted {s['predicted']:.2f}  measured "
                f"{s['measured']:.3f}  ({s['residual']:+.3f}, resolution "
                f"{s['resolution']:.3f})"
            )
            if s["prediction"] is not None and s["prediction"].path:
                emit(f"      from {s['prediction'].path}")


def _by_axis(kept, emit):
    emit()
    emit("-" * 78)
    emit("Residual by axis -- where the floor stops being flat")
    emit("-" * 78)
    for axis_name, key_fn in (
        ("phase", lambda s: s["phase"]),
        ("unit", lambda s: s["unit"]),
        ("bound", lambda s: s["bound"] or "-"),
        ("prediction kind", lambda s: s["kind"]),
        ("charged by tt_sim/pe/rv/cost.py", lambda s: "yes" if s["wired"] else "no"),
        ("beyond the fit's resolution", lambda s: "yes" if s["resolved"] else "no"),
    ):
        emit()
        emit(f"  {axis_name}")
        for name, group in sorted(_grouped(kept, key_fn).items()):
            stats = _summary([s["residual"] for s in group])
            emit(
                f"    {name:<22} n={stats['n']:<4} median {stats['median']:>7.3f}"
                f"   min {stats['min']:>7.3f}   max {stats['max']:>7.3f}"
            )


def _exploratory_readout(series, emit):
    """The probes the ladder dropped for having nothing to be compared to.

    Dropping them from the residual table is right -- there is no claim to test
    -- but dropping them from the REPORT would throw away the measurements this
    benchmark exists to make. They are printed with their reason attached.
    """
    rows = [
        s
        for s in series
        if s["kind"] == "exploratory"
        and s["active_threads"] == 1
        and s.get("measured") is not None
    ]
    if not rows:
        return
    emit()
    emit("-" * 78)
    emit("Exploratory: measured, with nothing to measure it against")
    emit("-" * 78)
    emit(
        "  Excluded from the residual ladder above because no document in either\n"
        "  tree gives a number. That makes these numbers the RESULT rather than a\n"
        "  check on one, and a boring value here is a real finding.\n"
    )
    emit(f"  {'probe':<18}{'ph':<3}{'unit':<9}{'measured':>10}")
    for s in sorted(rows, key=lambda s: (s["phase"], s["probe"])):
        emit(f"  {s['probe']:<18}{s['phase']:<3}{s['unit']:<9}{s['measured']:>10.3f}")


def _measured(series, probe, variant="t1", thread=1):
    for s in series:
        if s["probe"] == probe and s["variant"] == variant and s["thread"] == thread:
            return s.get("measured")
    return None


def _fusion_check(series, emit, meta=None):
    """Phase T: does the instruction cache fuse adjacent ``.ttinsn`` words?

    ``riscv.ttinsn_fusion`` is ``isa_doc`` -- "sequences of up to four adjacent
    .ttinsn instructions can be fused together by the RISCV T0 / T1 / T2
    instruction caches and executed in a single cycle" -- and is consumed by
    nothing. It is also the only claim in the whole ``riscv`` section that a
    sustained-throughput measurement structurally CANNOT see, because the same
    page caps the dequeue rate at one per thread per cycle.

    The discriminator is two probes with the identical twenty instructions,
    differing only in whether any two ``.ttinsn`` words are adjacent.
    """
    pad = float((meta or {}).get("pad", 16))
    fuse4 = _measured(series, "tt_fuse4")
    spread4 = _measured(series, "tt_spread4")
    if fuse4 is None or spread4 is None:
        return
    emit()
    emit("-" * 78)
    emit("Phase T: does the instruction cache fuse adjacent .ttinsn words?")
    emit("-" * 78)
    pad_m = _measured(series, "tt_pad")
    fuse2 = _measured(series, "tt_fuse2")
    emit(f"  {'probe':<12}{'cyc/group':>11}{'over tt_pad':>13}")
    for name, value in (
        ("tt_pad", pad_m),
        ("tt_fuse2", fuse2),
        ("tt_fuse4", fuse4),
        ("tt_spread4", spread4),
    ):
        if value is None:
            continue
        over = "" if pad_m is None or name == "tt_pad" else f"{value - pad_m:>13.3f}"
        emit(f"  {name:<12}{value:>11.3f}{over}")
    delta = spread4 - fuse4
    emit()
    emit(f"  spread4 - fuse4 = {delta:+.3f} cycles/group   (documented: +3.000)")
    emit()
    if delta > 1.5:
        emit(
            "  VERDICT: the instruction cache FUSES adjacent `.ttinsn` words, as\n"
            "  BlackholeA0/.../PushTensixInstruction.md describes. "
            "`riscv.ttinsn_fusion`\n"
            "  is a claim about something a kernel can actually reach, and the RISC-V\n"
            "  push cost is BELOW one cycle per Tensix instruction in a burst short\n"
            "  enough for the queue to drain -- which is a rate no sustained\n"
            "  measurement, this one or tensixbench's, could ever have shown."
        )
    elif abs(delta) <= 0.5:
        emit(
            "  VERDICT: no fusion this instrument can see. Every `.ttinsn` word\n"
            "  costs its own issue slot whether or not its neighbours are also\n"
            "  `.ttinsn`, so `riscv.ttinsn_fusion.max_fused` describes something a\n"
            "  compiler-generated instruction stream does not reach -- or does not\n"
            "  exist on this device at all.\n"
            "\n"
            "  AGAINST tt-sim this verdict is FORCED and says nothing about any\n"
            "  hardware: tt-sim has no instruction cache, decodes one `.ttinsn` per\n"
            "  cycle in RV_TT_ISA.run, and appends to an unbounded queue. A null\n"
            "  here from a simulator run tests this harness end to end and nothing\n"
            "  else."
        )
    else:
        emit(
            f"  VERDICT: partial. {delta:+.3f} is neither the documented +3 nor a\n"
            "  clean null, so either fewer than four words fuse or the fused push\n"
            "  costs more than one slot. Report the raw numbers, not a conclusion."
        )
    del pad


def _branch_check(series, arch, emit):
    """Phase C: is there a predictor, and what does a mispredict cost?"""
    nt = _measured(series, "c_nt")
    taken = _measured(series, "c_t")
    if nt is None or taken is None:
        return
    from tt_sim.perf.costs import load_costs

    integer = (load_costs(arch).section("riscv").get("integer_unit")) or {}
    bubble, _ = _cycles_of(integer.get("branch_mispredict_bubble"))
    observed, _ = _cycles_of(integer.get("branch_mispredict_observed"))

    emit()
    emit("-" * 78)
    emit("Phase C: is there a branch predictor?")
    emit("-" * 78)
    emit(
        "  `c_nt` and `c_t` execute the IDENTICAL dynamic instruction sequence -- a\n"
        "  branch whose target is the address the not-taken path falls through to --\n"
        "  and differ in exactly one bit: whether it was taken. The `c_xor_*` trio\n"
        "  repeats the comparison with an `xori` in every arm, so the alternating\n"
        "  pattern has a value to alternate on and the mix stays matched.\n"
    )
    emit(f"  {'pattern':<12}{'bare':>10}{'with xori':>12}")
    xnt = _measured(series, "c_xor_nt")
    xt = _measured(series, "c_xor_t")
    xalt = _measured(series, "c_xor_alt")
    emit(
        f"  {'not taken':<12}{nt:>10.3f}{(xnt if xnt is not None else float('nan')):>12.3f}"
    )
    emit(
        f"  {'taken':<12}{taken:>10.3f}{(xt if xt is not None else float('nan')):>12.3f}"
    )
    emit(
        f"  {'alternating':<12}{'-':>10}{(xalt if xalt is not None else float('nan')):>12.3f}"
    )
    jal = _measured(series, "c_jal")
    if jal is not None:
        emit(f"  {'jal (uncond)':<12}{jal:>10.3f}")
    emit()
    emit(f"  taken - not taken   = {taken - nt:+.3f} cycles/branch")
    if xnt is not None and xalt is not None:
        emit(f"  alternating - nt    = {xalt - xnt:+.3f} cycles/branch")
    emit(
        f"\n  The tables give the mispredict BUBBLE as {bubble} cycles on this\n"
        f"  architecture, looking like {observed} cycles of occupancy from outside\n"
        "  (`riscv.integer_unit.branch_mispredict_bubble`, isa_doc). That is the\n"
        "  SIZE of one mispredict. How OFTEN one happens is what these deltas say,\n"
        "  and it is the reason tt_sim/pe/rv/cost.py charges nothing here:\n"
        '  "neither the docs nor tt-sim describe the predictor, so the number of\n'
        "  mispredictions is unknowable and charging every taken branch would be a\n"
        '  fabrication."\n'
    )
    biggest = max(
        abs(taken - nt),
        abs((xalt - xnt) if xalt is not None and xnt is not None else 0.0),
    )
    if biggest <= 0.25:
        emit(
            "  VERDICT: no direction-dependent cost this instrument can resolve.\n"
            "  Either there is no predictor and no penalty, or every pattern here is\n"
            "  predicted correctly. `cost.py`'s refusal to charge a mispredict costs\n"
            "  nothing on evidence like this -- but note that AGAINST tt-sim the\n"
            "  verdict is forced: it has no predictor and no bubble, so a null is\n"
            "  what its own construction guarantees."
        )
    else:
        emit(
            "  VERDICT: branch direction costs cycles. Compare the delta against the\n"
            f"  {bubble}-cycle bubble above: a delta at the bubble means every branch\n"
            "  of that pattern mispredicts, and a delta below it means some fraction\n"
            "  does. Either way `riscv.integer_unit.branch_mispredict_bubble` now has\n"
            "  a measured mispredict RATE to go with its published size, which is the\n"
            "  missing half of a consumable cost."
        )


def _wide_rate(series, min_n=None):
    """``(rate, lo, hi)`` -- cycles per instruction over the widest usable span.

    A *difference of two raw measurements*, deliberately, rather than a slope
    through all eight or a point-by-point control subtraction. Every phase-Q
    point carries the same fixed cost -- two clock reads, a barrier, and the
    cascade's seven compare-and-branch pairs, all of which execute at every
    burst length -- so subtracting one point from another cancels it exactly,
    and taking the two furthest apart divides whatever is left of it by the
    largest possible denominator.
    """
    min_n = QUEUE_MIN_N if min_n is None else min_n
    usable = sorted(n for n in series if n >= min_n)
    if len(usable) < 2:
        return None
    lo, hi = usable[0], usable[-1]
    return (series[hi] - series[lo]) / float(hi - lo), lo, hi


#: How close to the saturated rate an adjacent-pair marginal has to get before
#: the burst is called back-pressured. Not 1.0: the marginal is a difference of
#: two single-shot measurements and carries the phase's whole noise floor.
QUEUE_KNEE_FRACTION = 0.9


def _queue_check(rows, emit):
    """Phase Q: does the issuing core run ahead of the Tensix backend?

    THE READ-OUT CHANGED ON 2026-08-05, on the strength of the first silicon
    run, and the change is a retraction rather than a tuning. It used to
    subtract ``q_ctrl`` **point by point** on the design's stated grounds that
    "the cascade that emits 2^p copies of a body executes p + 1
    compare-and-branch pairs", i.e. that the control grows with the burst index
    and so cannot be fitted away. **That premise is false.** Read the kernel:
    all seven ``if ((P) >= k)`` tests are evaluated at every burst length, so
    the cascade's instruction count is the SAME at n=1 and n=128. What actually
    varies with the burst index is which of those branches are taken and which
    blocks are entered for the first time -- worth 6 to 23 cycles on silicon,
    reproducibly (the ``--blocks 32`` and ``--blocks 8`` runs agree point for
    point) and NON-monotonically.

    Subtracting that from a probe whose entire signal is 1-4 cycles at n <= 4
    produced NEGATIVE net costs -- ``net -4.0``, ``marg -3.000`` -- and, worse,
    fed up to +-17 cycles of structured error into the marginals it then read a
    queue depth off. So ``q_ctrl`` is no longer a subtrahend. It is the phase's
    declared NOISE FLOOR: its spread is how wrong one phase-Q point can be, and
    everything below is a difference of raw points over a span wide enough for
    that spread to be small.

    THE SECOND DEFECT was hunting a knee in all four probes. A knee exists only
    where the backend's occupancy EXCEEDS the core's issue rate; ``q_nop``'s
    unit is inert and ``q_setdmareg``'s occupancy is 1, so for those two
    "issue-limited" and "back-pressured" are the same number and no knee can
    exist to find. The old code duly announced one for ``q_nop`` at n=32, off
    two noisy points. Only ``q_adddmareg`` -- whose unit ``tensixbench``
    measured at 3.0 cycles -- can answer this phase's question, and only against
    ``q_adddmareg_sync``, which supplies the saturated rate by measurement
    instead of by assumption.
    """
    slots = {}
    for row in rows:
        if row["phase"] != "q":
            continue
        # `loop_overhead` is emitted alongside every phase because it is the
        # slope phases' control; it sweeps block count, not burst length, and
        # has nothing to do with a queue.
        if row["probe"] == CONTROL_PROBE:
            continue
        key = (row["active_threads"], row["thread"])
        slots.setdefault(key, {}).setdefault(row["probe"], {})[row["n"]] = row["cycles"]
    slots = {k: v for k, v in slots.items() if QUEUE_CONTROL_PROBE in v}
    if not slots:
        return

    emit()
    emit("-" * 78)
    emit("Phase Q: does the core run ahead of the Tensix backend, and how far?")
    emit("-" * 78)
    emit(
        "  EXPLORATORY. Nothing in the ISA documentation or either vendor tree gives\n"
        "  a Tensix instruction queue depth, so there is no prediction here to\n"
        "  confirm or refute -- only a number nothing has measured.\n"
        "\n"
        "  CYCLES ARE RAW. `q_ctrl` is NOT subtracted point by point, because the\n"
        "  premise of that subtraction turned out to be wrong: the cascade evaluates\n"
        "  all seven of its `if (p >= k)` tests at EVERY burst length, so its cost is\n"
        "  constant in n rather than growing with it, and what does vary -- branch\n"
        "  directions, first-touch of the deeper blocks -- is non-monotonic. Every\n"
        "  number below is a DIFFERENCE of raw points, in which that constant cancels\n"
        f"  exactly, taken over a span starting at n = {QUEUE_MIN_N} so that what does not\n"
        "  cancel is divided by something large."
    )
    control_values = [
        c for slot in slots.values() for c in slot[QUEUE_CONTROL_PROBE].values()
    ]
    spread = max(control_values) - min(control_values)
    emit()
    emit(
        f"  NOISE FLOOR: `q_ctrl` spans {min(control_values)}-{max(control_values)} "
        f"cycles across the burst lengths in this\n"
        f"  run ({spread} cycles). It executes the identical cascade with an empty body, so\n"
        "  that spread is what a single phase-Q point can be wrong by, and any rate\n"
        f"  below carries {spread}/(span) cycles per instruction of it."
    )

    for threads, thread in sorted(slots):
        probes = slots[(threads, thread)]
        ns = sorted({n for series in probes.values() for n in series})
        emit()
        emit(f"  {threads} issuing thread(s), thread {thread}")
        emit(
            "      "
            + f"{'probe':<20}"
            + "".join(f"{n:>6}" for n in ns)
            + f"{'rate':>9}"
        )
        for probe in [QUEUE_CONTROL_PROBE] + sorted(
            p for p in probes if p != QUEUE_CONTROL_PROBE
        ):
            series = probes[probe]
            rate = _wide_rate(series)
            cell = "  (noise)" if probe == QUEUE_CONTROL_PROBE else f"{rate[0]:>9.3f}"
            emit(
                "      "
                + f"{probe:<20}"
                + "".join(f"{series[n]:>6}" if n in series else f"{'-':>6}" for n in ns)
                + cell
            )
        first = _wide_rate(probes[next(iter(probes))])
        if first is not None:
            emit(
                f"      rate = (cycles[n={first[2]}] - cycles[n={first[1]}]) / "
                f"{first[2] - first[1]}, cycles per instruction"
            )
        _queue_slot_readout(probes, emit)

    emit()
    emit(
        "  WHAT A KNEE WOULD BE, and why only one probe here can have one. Below the\n"
        "  queue's depth the core is not back-pressured and one more instruction costs\n"
        "  one cycle; above it, one more costs the backend unit's occupancy. That is\n"
        "  only two different numbers when the occupancy EXCEEDS the issue rate.\n"
        "  `q_nop`'s unit is inert and `q_setdmareg`'s occupancy is 1, so for those two\n"
        "  the two regimes are the same number and no knee exists to find; they are\n"
        "  printed as controls on the instrument, not as candidates. `q_adddmareg` is\n"
        "  the probe, and `q_adddmareg_sync` -- the identical burst with\n"
        "  `tensix_sync()` inside the timed region, so it cannot return until the pipe\n"
        "  has drained -- supplies the saturated rate by MEASUREMENT rather than by\n"
        "  assuming the documented occupancy.\n"
        "\n"
        "  AN ABSOLUTE PHASE-Q RATE IS NOT A COST, and the two inert probes are how\n"
        "  you can see that without leaving this table. `q_nop` and `q_setdmareg` must\n"
        "  structurally be at or above 1.000 cycles per instruction and cannot be far\n"
        "  above it -- and on silicon they came out BRACKETING it, at 1.75 and 1.22\n"
        "  from one thread. Each burst runs exactly ONCE, cold, so its own body's first\n"
        "  instruction fetch is inside the timed region and no repetition averages it\n"
        "  out; that is noise this phase cannot remove and `q_ctrl` cannot see, since\n"
        "  `q_ctrl` has no body to fetch. It is also why only the plain/sync PAIR is\n"
        "  read as a result: those two execute the identical burst, so the fetch, the\n"
        "  cascade and the clock reads all cancel between them. That pair is stable to\n"
        "  the third decimal across six thread slots and two runs.\n"
        "\n"
        "  AGAINST tt-sim every answer here is FORCED: `TensixFrontend.\n"
        "  push_mop_instruction` is an unbounded list append, so the core runs ahead\n"
        "  at every burst length by construction and a null says nothing about any\n"
        "  hardware."
    )


def _queue_slot_readout(probes, emit):
    """The plain/sync pair for one thread slot: knee, in-flight work, verdict."""
    plain = probes.get("q_adddmareg")
    synced = probes.get("q_adddmareg_sync")
    if not plain or not synced:
        return
    plain_rate = _wide_rate(plain)
    sync_rate = _wide_rate(synced)
    if plain_rate is None or sync_rate is None:
        return

    # The knee: the first adjacent interval above QUEUE_MIN_N whose marginal
    # reaches the saturated rate. Adjacent differences of raw points, so the
    # cascade's constant cancels here too.
    usable = sorted(n for n in plain if n >= QUEUE_MIN_N)
    margins = [
        (lo, hi, (plain[hi] - plain[lo]) / float(hi - lo))
        for lo, hi in zip(usable, usable[1:])
    ]
    if margins:
        emit(
            "      q_adddmareg marginal, adjacent burst lengths: "
            + "  ".join(f"{lo}->{hi} {m:.2f}" for lo, hi, m in margins)
        )
    threshold = QUEUE_KNEE_FRACTION * sync_rate[0]
    knee = next((m for m in margins if m[2] >= threshold), None)

    shared = sorted(set(plain) & set(synced))
    if shared:
        base = synced[shared[0]] - plain[shared[0]]
        emit(
            f"      in flight (cycles of ThCon work the burst did not wait for, "
            f"less {base}\n      cycles for tensix_sync()'s own cost at n={shared[0]}): "
            + "  ".join(
                f"n={n} {synced[n] - plain[n] - base:+.0f}" for n in shared if n >= 8
            )
        )
    emit(
        f"      -> issuing core {plain_rate[0]:.3f} cyc/instr against the work's own "
        f"{sync_rate[0]:.3f}"
    )
    if knee is None:
        emit(
            f"         NO KNEE up to n={plain_rate[2]}: the core never stops running "
            "ahead within\n         this sweep. The burst length that would settle it "
            "is beyond 128."
        )
    else:
        emit(
            f"         KNEE between n={knee[0]} and n={knee[1]}: the marginal reaches "
            f"{knee[2]:.2f} against a\n         saturated {sync_rate[0]:.3f}, so the "
            "core is back-pressured from there on."
        )


def _fetch_check(series, emit):
    """Phase F: does a bigger loop body cost more per instruction?"""
    names = ("f_64", "f_128", "f_256", "f_512", "f_1024", "f_2048")
    values = [(nm, _measured(series, nm)) for nm in names]
    values = [(nm, v) for nm, v in values if v is not None]
    if len(values) < 2:
        return
    emit()
    emit("-" * 78)
    emit("Phase F: instruction footprint")
    emit("-" * 78)
    emit(
        "  EXPLORATORY. `riscv.instruction_fetch` gives the fetch PERIOD -- one\n"
        '  128-bit L1 read per four instructions -- and its own note says "instruction\n'
        '  cache miss cost is not published". No cache SIZE is published either. So a\n'
        "  flat row is consistent with the documentation and a cliff is a number\n"
        "  nothing has ever printed; both are results.\n"
    )
    emit(f"  {'footprint':<12}{'bytes':>8}{'cyc/instr':>12}")
    for nm, v in values:
        k = int(nm.split("_")[1])
        emit(f"  {nm:<12}{k * 4:>8}{v:>12.3f}")
    spread = max(v for _, v in values) - min(v for _, v in values)
    emit()
    if spread < 0.05:
        emit(
            f"  VERDICT: flat to {spread:.3f} cycles/instruction across "
            f"{values[0][0].split('_')[1]}-{values[-1][0].split('_')[1]} instructions.\n"
            "  Instruction fetch is not the limit anywhere in this range, which is\n"
            "  what the documented fetch period predicts while the loop is resident.\n"
            "  It does NOT locate a cache size -- it says the cliff, if there is one,\n"
            "  is beyond the largest footprint measured. Widening the sweep is the\n"
            "  way to find it, not reading harder into a flat row."
        )
    else:
        emit(
            f"  VERDICT: NOT flat -- {spread:.3f} cycles/instruction between the\n"
            "  cheapest and dearest footprint. The step locates a capacity nothing\n"
            "  publishes, and it is the first number this benchmark produces that no\n"
            "  document could have given."
        )


def _issue_limit_check(series, emit):
    """Does the measured rate change when more TRISCs run the same probes?

    Three TRISCs share one L1, one instruction-fetch path and one Tensix
    coprocessor. A probe that costs the same at t1 and t3 is using a resource
    that is genuinely per-core; one that scales with the thread count is
    contending for something shared, and WHICH probes do that is the useful
    output -- an `addi` scaling would mean instruction fetch is shared, where a
    `.ttinsn` scaling would mean the Tensix queue is.
    """
    emit()
    emit("-" * 78)
    emit("Contention: the same probes from 1, 2 and 3 TRISCs")
    emit("-" * 78)
    variants = sorted({s["variant"] for s in series if s["variant"].startswith("t")})
    if len(variants) < 2:
        emit("  only one thread set in this run; nothing to compare.")
        return
    issuers = {
        v: max((s["active_threads"] for s in series if s["variant"] == v), default=1)
        for v in variants
    }
    expected = issuers[variants[-1]] / max(issuers[variants[0]], 1)
    emit(
        f"  {'probe':<18}{'unit':<9}"
        + "".join(f"{v:>10}" for v in variants)
        + "     verdict"
    )
    probes = sorted({s["probe"] for s in series if s["phase"] != "q"})
    for probe in probes:
        if probe == CONTROL_PROBE:
            continue
        cells, unit = [], ""
        for variant in variants:
            values = [
                s["measured"]
                for s in series
                if s["variant"] == variant
                and s["probe"] == probe
                and s.get("measured") is not None
            ]
            unit = next((s["unit"] for s in series if s["probe"] == probe), unit)
            cells.append(statistics.fmean(values) if values else None)
        if any(c is None for c in cells) or not cells[0]:
            continue
        ratio = cells[-1] / cells[0]
        if ratio >= 0.75 * expected:
            verdict = f"shared ({ratio:.1f}x)"
        elif ratio <= 1.25:
            verdict = "per-core"
        else:
            verdict = f"partial ({ratio:.1f}x)"
        emit(
            f"  {probe:<18}{unit:<9}"
            + "".join(f"{c:>10.3f}" for c in cells)
            + f"     {verdict}"
        )
    emit()
    emit(
        "  'per-core' means adding issuers cost nothing, so the resource is\n"
        "  replicated per baby core. 'shared' means per-thread cost grew with the\n"
        "  thread count, i.e. the three cores are queueing for one thing. For the\n"
        "  `tt_*` probes that thing would be the Tensix coprocessor; for `rv_*` and\n"
        "  `f_*` it would be L1 -- which all three TRISCs fetch instructions from."
    )


def _live_check(series, emit):
    """Is the instrument resolving anything at all?

    The trap `tensixbench` fell into: a run of 1.000s is simultaneously the
    expected simulator output and the signature of a benchmark that measured
    nothing. These four probes have a documented cost above one cycle on at
    least one architecture and are consumed by ``tt_sim/pe/rv/cost.py``, so they
    are the control that tells the two apart.
    """
    emit()
    emit("-" * 78)
    emit("Is the instrument live?")
    emit("-" * 78)
    checks = ("rv_mul_dep", "rv_div", "rv_load_chase", "rv_store_spread")
    seen, above = 0, 0
    for probe in checks:
        value = _measured(series, probe)
        if value is None:
            continue
        seen += 1
        if value > 1.5:
            above += 1
        emit(f"  {probe:<18}{value:>10.3f} cycles/instruction")
    if seen == 0:
        emit("  phase R was not run at t1; cannot check.")
        return
    emit()
    if above == 0:
        emit(
            "  ALL of them read ~1.0. Every one has a documented cost above one cycle\n"
            "  on at least one architecture, so this is the signature of a run that\n"
            "  measured nothing -- or of a device on which NOTHING back-pressures the\n"
            "  issuing core, which is what tt-sim looks like with TT_SIM_COST_MODEL\n"
            "  unset. On silicon, treat it as a broken run.\n"
            "\n"
            "  Every other verdict in this report is unsafe until this one passes."
        )
    else:
        emit(
            f"  {above} of {seen} read above 1.0, so the timer resolves a real\n"
            "  per-instruction cost and a 1.000 elsewhere in this run is a finding\n"
            "  rather than a floor."
        )


def _differential(rows, reference_rows, arch, emit, meta=None):
    """The same binary, two devices: silicon against tt-sim, per series."""
    emit()
    emit("-" * 78)
    emit("Differential: the same binary on both devices")
    emit("-" * 78)
    measured = {
        (s["phase"], s["variant"], s["probe"], s["thread"]): s
        for s in attach_predictions(apply_control(series_of(rows)), arch, meta)
    }
    other = {
        (s["phase"], s["variant"], s["probe"], s["thread"]): s
        for s in attach_predictions(
            apply_control(series_of(reference_rows)), arch, meta
        )
    }
    shared = sorted(set(measured) & set(other))
    if not shared:
        emit("  no series in common.")
        return
    emit(
        f"  {'probe':<18}{'ph':<3}{'variant':<8}{'thr':>4}{'measured':>11}{'reference':>11}{'delta':>10}"
    )
    deltas = []
    for key in shared:
        a, b = measured[key], other[key]
        if a.get("measured") is None or b.get("measured") is None:
            continue
        if a["probe"] == CONTROL_PROBE:
            continue
        delta = a["measured"] - b["measured"]
        deltas.append(delta)
        emit(
            f"  {a['probe']:<18}{a['phase']:<3}{a['variant']:<8}{a['thread']:>4}"
            f"{a['measured']:>11.3f}{b['measured']:>11.3f}{delta:>10.3f}"
        )
    if deltas:
        stats = _summary(deltas)
        emit()
        emit(
            f"  delta: n {stats['n']}  min {stats['min']:.3f}  median "
            f"{stats['median']:.3f}  max {stats['max']:.3f} cycles/instruction"
        )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sweep a perfbench/riscvbench CSV against the RISC-V cost tables."
    )
    parser.add_argument("--measured", help="CSV from a riscvbench run")
    parser.add_argument(
        "--reference",
        help="a second CSV of the same binary on another device, for a differential",
    )
    parser.add_argument(
        "--arch",
        choices=("wormhole", "blackhole"),
        help="architecture the tables are read for (default: the CSV's arch= comment)",
    )
    args = parser.parse_args(argv)

    if args.measured:
        path = Path(args.measured)
    else:
        path = default_measured_path(args.arch)
        if path is None:
            tracked = reference_datasets()
            print(
                "no --measured CSV given, and no tracked reference to fall back on.\n"
                "\n"
                f"looked in: {DATASET_DIR}\n"
                f"looked for: {PRIMARY_DATASET}\n"
                f"found: {', '.join(p.name for p in tracked) or 'nothing'}\n"
                "\n"
                "Produce one by running perfbench/riscvbench on a card -- see\n"
                "perfbench/riscvbench/README.md -- or point --measured at a\n"
                "simulator run, which tests this harness and nothing else."
            )
            return 0
    if not path.exists():
        print(f"no CSV at {path}; nothing to sweep.")
        return 0

    rows, meta = read_csv(path)
    arch = args.arch or meta.get("arch")
    if arch not in ("wormhole", "blackhole"):
        print(f"cannot tell which architecture {path} came from; pass --arch.")
        return 2

    reference_rows = None
    if args.reference:
        reference_path = Path(args.reference)
        if not reference_path.exists():
            print(f"no reference CSV at {reference_path}.")
            return 2
        reference_rows, _ = read_csv(reference_path)

    report(rows, arch, label=str(path), reference=reference_rows, meta=meta)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
