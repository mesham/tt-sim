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

* **phases R / T / C / F / G** -- ``n`` unrolled blocks of ``unroll``
  instructions, timestamped off ``RISCV_DEBUG_REG_WALL_CLOCK_L``, the same
  register tt-metal's device profiler reads. ``variant`` is ``t1``/``t2``/``t3``:
  how many TRISCs ran the identical probes at once. Phase G is phase F's
  question at the footprints between 1024 and 2048 instructions, in its own
  kernel build because phase F's is already at tt-metal's size limit; see
  :data:`FOOTPRINT_PROBES`.
* **phase Q** -- ``n`` is a *burst length* (1, 2, 4 ... 128) rather than a block
  count, and ``unroll`` is 1. It is looking for a **knee**, not a slope, and is
  reported and gated separately for that reason: a straight line there would be
  the null result rather than the healthy one. Its read-out is a *rate over a
  wide baseline* rather than a point-by-point control subtraction; the silicon
  run that forced that change is described at :func:`_queue_check`.
* **phase S** -- ``n`` is a burst length too (4, 8 ... 512), and the question is
  whether the queue phase Q measured is **shared between the TRISCs or private
  to each**. Its answer is a *ratio between thread counts* and never a level.
  Why a spinning second thread cannot answer it, and what can, is at
  :func:`_sharing_check`.

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

#: The ``--blocks 8`` companion run. It is tracked **because it failed**: four
#: of its five phases were refused by the benchmark's own validity gate, mostly
#: because the ``loop_overhead`` CONTROL itself fits to R^2 < 0.99 at four
#: points spanning n = 8..32. That is not a statement about the device -- its
#: per-instruction numbers agree with the primary run to a few thousandths of a
#: cycle wherever both are readable -- it is the measurement that puts a floor
#: under ``--blocks``. Sweeping its SLOPE phases as if they were a result is
#: exactly the mistake it exists to prevent, so the default is a choice made
#: here rather than whatever sorts first.
#:
#: Its PHASE Q is a different matter and is evidence rather than a warning.
#: Phase Q takes its burst length from the burst index alone -- the kernel's
#: ``QPROBE``/``QLOOPPROBE`` macros never read ``base_blocks``, and the host
#: labels a phase-Q point ``n0 << k`` where a slope probe gets
#: ``base_blocks * (k + 1)`` -- so it is independent of ``--blocks`` by
#: construction, and this file is that independence measured: every loop-form
#: point in it is within two cycles of the primary run's at a quarter the block
#: count.
MIN_BLOCKS_DATASET = "riscvbench-blackhole-blocks8.csv"

#: The two single-phase ``--gset`` runs. Phase G could not be one kernel --
#: phase F's bodies already sit within a few hundred bytes of tt-metal's kernel
#: config buffer and the three intermediates do not fit even alone -- so each
#: intermediate is paired with a ``g_1024`` anchor in its own build. Set 0
#: (``g_1280``) rides along in the two full runs; sets 1 and 2 are twelve rows
#: each and are tracked because they are the only evidence for ``g_1536`` and
#: ``g_1792``, which is what turned phase F's octave into a bracket and a
#: plateau. They are never a candidate for :data:`PRIMARY_DATASET`: a
#: single-phase, single-thread run cannot run the live-instrument check, and a
#: sweep of one is a footprint table with nothing to calibrate it.
FOOTPRINT_DATASETS = (
    "riscvbench-blackhole-gset1.csv",
    "riscvbench-blackhole-gset2.csv",
)

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

#: Every instruction-footprint probe, phase F's and phase G's together, in the
#: order the read-out prints them. The two phases are separate kernel BUILDS and
#: that is not an accident: phase F's six bodies already sit within a few
#: hundred bytes of tt-metal's kernel config buffer -- adding phase G's
#: intermediates to it aborts the launch with ``Program size (125040) too large
#: for kernel config buffer (70656)``, measured rather than predicted -- so the
#: footprints between 1024 and 2048 had to become a phase of their own, and one
#: split further into compile-time sets (``--gset``) at that. ``g_1024`` is
#: every set's in-build flat anchor, which is what makes a phase-G reading
#: interpretable without comparing across builds.
FOOTPRINT_PROBES = (
    "f_64",
    "f_128",
    "f_256",
    "f_512",
    "f_1024",
    "f_2048",
    "g_1024",
    "g_1280",
    "g_1536",
    "g_1792",
)


def reference_datasets():
    """Every tracked reference measurement, sorted by path."""
    if not DATASET_DIR.is_dir():
        return []
    return sorted(DATASET_DIR.glob("riscvbench-*.csv"))


def default_measured_path(arch=None):
    """The tracked dataset to sweep when ``--measured`` is not given.

    Never :data:`MIN_BLOCKS_DATASET`, whatever ``arch`` asks for: that file is a
    deliberately-invalid run kept as evidence about the instrument. Never a
    :data:`FOOTPRINT_DATASETS` entry either: those are twelve rows of one phase
    at one thread, so nothing in them could fail the live-instrument check.
    """
    if arch is not None:
        candidate = DATASET_DIR / f"riscvbench-{arch}.csv"
        return candidate if candidate.exists() else None
    primary = DATASET_DIR / PRIMARY_DATASET
    if primary.exists():
        return primary
    never = {MIN_BLOCKS_DATASET, *FOOTPRINT_DATASETS}
    found = [p for p in reference_datasets() if p.name not in never]
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


#: Each L1 load probe's DATA working set, in bytes, read off the kernel body in
#: ``perfbench/riscvbench/src/kernels/compute/rv_probes.cpp``:
#:
#: * ``rv_load_chase`` walks a ring of ``RVBENCH_UNROLL`` = 64 nodes placed 16
#:   bytes apart, so it touches 1024 bytes.
#: * ``rv_load_indep`` loads from ``0/16/32/48(%scratch)`` in rotation, so it
#:   touches 64 bytes.
#:
#: These are properties of the probe, not of the run, so they are declared here
#: rather than read from the CSV -- there is no column that could carry them
#: (``unroll`` is the instruction count, which is 64 for BOTH of these).
L1_WORKING_SET_BYTES = {
    "rv_load_chase": 64 * 16,
    "rv_load_indep": 4 * 16,
}


def _l1_load_row(riscv, arch, working_set_bytes=None):
    """``(key, raw entry)`` for an L1 load whose working set is known.

    THIS USED TO READ ``_LOAD_LATENCY_KEYS`` AND NOTHING ELSE, which on
    Blackhole names ``l1_dcache_hit`` -- and that was wrong for every probe
    here. The mapping in ``model.py`` answers "what should the simulator charge
    an arbitrary L1 load whose hit rate nobody publishes?", and its answer is
    the low end of the two-ended pair, which is the same floor policy as every
    ``at_least`` in the file. This function answers a different question:
    **which row does a probe of known access pattern actually reach?** Choosing
    between two distinct rows is not a bound-resolution problem, and resolving
    it "conservatively" does not make it conservative -- it makes the residual
    measure the wrong row. ``rv_load_chase`` is a pointer chase over 1 KiB
    against a 64-byte L0 data cache; the hit row does not apply to it at all,
    and predicting 2 where the documentation gives >= 8 read out as a 6-cycle
    discrepancy that was entirely the sweep's own.

    The discriminator is the cache's published CAPACITY and only that, because
    capacity is the only property of the L0 that ``bh_riscv#l0-data-cache``
    publishes. The test is STRICT: a working set *larger* than the capacity
    cannot be resident under any organisation, so it reaches the miss row and
    the document settles it. A working set that fits -- including one sitting
    exactly ON the capacity, which is where ``rv_load_indep`` sits -- is left on
    the table's default row. That is not a claim that it hits; it is a refusal
    to claim either way from a page that publishes no associativity, no
    replacement policy and a ~0.8 % periodic flush. ``rv_load_indep`` measures
    the miss row's sustained rate on silicon, and moving its prediction on the
    strength of that would be fitting the table to the measurement, which is the
    one thing this whole apparatus exists to not do.
    """
    from tt_sim.perf.model import RV_REGION_L1, l1_dcache_miss_key

    table = riscv.get("load_latency") or {}
    key = _latency_key(arch, RV_REGION_L1)
    miss_key = l1_dcache_miss_key(arch)
    capacity = (riscv.get("l0_data_cache") or {}).get("capacity_bytes")
    if (
        miss_key is not None
        and capacity is not None
        and working_set_bytes is not None
        and working_set_bytes > capacity
    ):
        key = miss_key
    return key, table.get(key)


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
            "model.py charges every bound. THE RANGE IS DATA-DEPENDENT, not a "
            "confidence interval: divide occupies EX1 for a number of cycles "
            '"dependent upon the magnitude of the dividend", so 6 and 33 are '
            "two operands rather than two guesses at one. The benchmark's "
            "dividend is 0x12345678 (29 significant bits, in the CSV header) "
            "and reads 33 -- the top of the band, from an operand near the top "
            "of the magnitude range. A residual here sizes the BENCHMARK's "
            "choice of dividend, not the table.",
        )
    capacity = (riscv.get("l0_data_cache") or {}).get("capacity_bytes")
    chase_set = L1_WORKING_SET_BYTES["rv_load_chase"]
    row_key, row = _l1_load_row(riscv, arch, chase_set)
    lat, lat_bound = _cycles_of(row)
    if lat is not None:
        add(
            "rv_load_chase",
            float(lat),
            lat_bound,
            f"riscv.load_latency.{row_key}",
            note=None
            if capacity is None
            else f"the chase's ring is {chase_set} bytes against an L0 data cache "
            f"of {capacity} (riscv.l0_data_cache.capacity_bytes), so it cannot be "
            "resident and this is the row it reaches whatever the cache's "
            "unpublished organisation. The hit row is not a conservative reading "
            "of this probe -- it is a different row. NOTE that tt-sim charges "
            "the HIT row for every L1 load, so against a tt-sim dataset this "
            "probe reads ~2 against a prediction of 8: an under-charge by the "
            "simulator of exactly the kind `rv_mul_dep` records, not an "
            "over-prediction by the table.",
        )
    indep_set = L1_WORKING_SET_BYTES["rv_load_indep"]
    indep_key, indep_row = _l1_load_row(riscv, arch, indep_set)
    lat, lat_bound = _cycles_of(indep_row)
    if lat is not None:
        under = loads.get("one_per_cycle_if_latency_under")
        per_window = loads.get("else_loads_per_window")
        offset = loads.get("else_window_cycles_offset")
        boundary = (
            None
            if capacity is None or indep_set != capacity
            else f" NOTE: this probe's {indep_set}-byte working set is EXACTLY the "
            f"L0 data cache's capacity, which the documentation does not settle "
            f"either way -- so it is left on `{indep_key}`, the table's default "
            "row, rather than moved to the miss row that the silicon reading "
            "matches. A residual here is the boundary, not an over-charge."
        )
        if under is not None and per_window and offset is not None:
            if lat < under:
                add(
                    "rv_load_indep",
                    1.0,
                    "exact",
                    "riscv.load_throughput.one_per_cycle_if_latency_under",
                    kind="derived",
                    derivation=f"load latency {lat} < {under} on "
                    f"`{indep_key}`, so the docs' "
                    f'"throughput of sustained loads is one per cycle" applies.'
                    + (boundary or ""),
                )
            else:
                value = (lat + offset) / float(per_window)
                add(
                    "rv_load_indep",
                    value,
                    "at_least",
                    "riscv.load_throughput",
                    kind="derived",
                    derivation=f"latency {lat} >= {under} on `{indep_key}`, so the "
                    f"docs give {per_window} loads every {lat} - 1 cycles, i.e. "
                    f"({lat} + {offset}) / {per_window} = {value:.3f} cycles/load."
                    + (boundary or ""),
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
    for probe in FOOTPRINT_PROBES:
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
            "phase == S",
            "phase S sweeps burst length too, and its answer is a RATIO between "
            "thread counts rather than a level: whether the queue phase Q "
            "measured is shared or per-thread. A slope through it, and a "
            "residual against a table entry that does not exist, would both be "
            "meaningless. Reported separately.",
            lambda s: s["phase"] != "s",
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
    _sharing_check(rows, emit)
    _depth_reconcile(rows, emit)
    _fetch_check(series, emit)
    _issue_limit_check(series, emit)
    _live_check(series, emit)
    _additions_present(rows, emit)
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

#: The loop-form burst probes, added so that phase Q could reach n = 1024
#: without its instruction stream growing with the burst. See
#: :func:`_queue_loop_readout`.
QUEUE_LOOP_PLAIN = "q_loop_adddmareg"
QUEUE_LOOP_SYNC = "q_loop_adddmareg_sync"
QUEUE_LOOP_BASELINE = "q_loop_addi"

#: How much of its own level the backlog's last doubling may add before the
#: series counts as an asymptote rather than a still-growing quantity. An
#: unbounded queue absorbs the whole of every doubling, so its backlog doubles
#: too and the last step is ~50 % of the level; a saturated one adds nothing.
QUEUE_FLATTEN_FRACTION = 0.25

#: How many ``q_ctrl`` spreads the backlog must clear before a depth in entries
#: is reported at all. The backlog is
#: ``(sync[n] - plain[n]) - (sync[lo] - plain[lo])`` -- a difference of two
#: differences, i.e. four raw single-shot points -- and ``q_ctrl``'s spread is
#: what ONE such point can be wrong by, so two of them is the floor. Taken from
#: the run's own measured spread rather than fixed in cycles, because the floor
#: is a property of the run and not of the instrument.
QUEUE_NOISE_MULTIPLE = 2.0


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
        f"  run ({spread} cycles, over every thread slot). It executes the identical\n"
        "  cascade with an empty body, so that spread is what a single phase-Q point can\n"
        f"  be wrong by, and any rate below carries {spread}/(span) cycles per instruction\n"
        "  of it. Each slot's refusal below uses ITS OWN spread rather than this one,\n"
        "  because the floor is a statement about that slot's run and not about the\n"
        "  noisiest slot in the file."
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
        _queue_loop_readout(probes, emit)

    emit()
    emit(
        "  TWO BURST FORMS, and `--blocks` enters NEITHER of them. The cascade emits\n"
        "  2^p copies of the body straight-line and runs n = 1..128; the loop form runs\n"
        "  n = 16..1024 as n/16 iterations of one 16-instruction block, so its\n"
        "  instruction footprint is 64 bytes at every burst length and no difference\n"
        "  between two of its points can be instruction fetch. Both take their burst\n"
        "  length from the burst index alone -- `riscvbench.cpp` labels a phase-Q point\n"
        "  `n0 << k` where the slope phases get `base_blocks * (k + 1)`, and the kernel's\n"
        "  QPROBE/QLOOPPROBE macros never read `base_blocks` at all -- so a phase-Q\n"
        "  reading is independent of `--blocks` by construction, and two runs an octave\n"
        "  apart in `--blocks` reproduce it point for point.\n"
        "\n"
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
            + "\n      (raw differences, UNADJUDICATED. They can come out negative, "
            "which work in\n      flight cannot be, because each is four single-shot "
            "points against a control\n      that spans ~16 cycles. The depth in entries "
            f"is read off the `{QUEUE_LOOP_PLAIN}`\n      pair below, which alone has a "
            "burst long enough for the backlog to settle.)"
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


def _queue_loop_readout(probes, emit):
    """The loop-form burst out to n = 1024: a depth in entries, or a refusal.

    THE REFUSAL IS THE POINT, and it was added on 2026-08-05 after a run in
    which three thread slots reported a NEGATIVE backlog and two more divided a
    single-digit one by a service rate to announce "~1 instruction in flight".
    Work in flight cannot be negative. That is not a noisy estimate, it is
    arithmetic run where there is no signal, and printing a number from it is
    the same class of error as the ``q_ctrl`` subtraction this phase already
    retracted once.

    So a depth is reported only when the backlog clears
    :data:`QUEUE_NOISE_MULTIPLE` times the control's own measured spread AND has
    stopped growing. The floor comes from ``q_ctrl`` rather than from a constant
    because it is the run's own statement about how wrong one of its points can
    be. It is what disqualifies every multi-thread slot on silicon: the drained
    rate there is three times higher, so the same queue holds a backlog three
    times smaller *in cycles*, and it lands underneath the floor.
    """
    plain = probes.get(QUEUE_LOOP_PLAIN)
    synced = probes.get(QUEUE_LOOP_SYNC)
    if not plain or not synced:
        return
    plain_rate = _wide_rate(plain)
    sync_rate = _wide_rate(synced)
    if plain_rate is None or sync_rate is None:
        return

    # Does changing the burst form change the quantity? Measured, over the span
    # where the cascade also ran, rather than argued.
    cascade = probes.get("q_adddmareg") or {}
    shared = sorted(n for n in set(cascade) & set(plain) if n >= QUEUE_MIN_N)
    if len(shared) >= 2:
        lo, hi = shared[0], shared[-1]
        casc = (cascade[hi] - cascade[lo]) / float(hi - lo)
        loop = (plain[hi] - plain[lo]) / float(hi - lo)
        emit(
            f"      form check n={lo}..{hi}: cascade {casc:.3f}, loop {loop:.3f}, "
            f"difference {loop - casc:+.3f} cyc/instr\n"
            "        (the loop's own back edge is inside that difference; "
            f"`{QUEUE_LOOP_BASELINE}` measures it)"
        )

    baseline = _wide_rate(probes.get(QUEUE_LOOP_BASELINE) or {})
    emit(
        f"      loop form to n={plain_rate[2]}: issuing core {plain_rate[0]:.3f} "
        f"| drained {sync_rate[0]:.3f}"
        + (
            f" | issue-limited baseline {baseline[0]:.3f}"
            if baseline is not None
            else " | issue-limited baseline not measured"
        )
    )

    ns = sorted(n for n in plain if n in synced)
    if len(ns) < 3:
        return
    base = synced[ns[0]] - plain[ns[0]]
    backlog = [(n, synced[n] - plain[n] - base) for n in ns]
    emit(
        f"      backlog in flight (cycles of ThCon work, less {base} for "
        f"tensix_sync()'s\n      own cost at n={ns[0]}): "
        + "  ".join(f"n={n} {v:+.0f}" for n, v in backlog)
    )
    margins = [(a, b, (plain[b] - plain[a]) / float(b - a)) for a, b in zip(ns, ns[1:])]
    emit(
        "      marginal, adjacent burst lengths: "
        + "  ".join(f"{a}->{b} {m:.2f}" for a, b, m in margins)
    )
    knee = next(
        (m for m in margins if m[2] >= QUEUE_KNEE_FRACTION * sync_rate[0]), None
    )
    if knee is None:
        emit(
            f"         NO KNEE up to n={plain_rate[2]}: the core never stops running "
            "ahead within this sweep."
        )
    else:
        emit(
            f"         KNEE between n={knee[0]} and n={knee[1]}: the marginal reaches "
            f"{knee[2]:.2f} against a\n         drained {sync_rate[0]:.3f}, so the core "
            "is back-pressured from there on."
        )

    ctrl = probes.get(QUEUE_CONTROL_PROBE) or {}
    spread = (max(ctrl.values()) - min(ctrl.values())) if ctrl else 0
    floor = QUEUE_NOISE_MULTIPLE * spread
    last = backlog[-1][1]
    last_step = backlog[-1][1] - backlog[-2][1]
    prev_step = backlog[-2][1] - backlog[-3][1]
    if last <= 0:
        emit(
            f"         BACKLOG NEGATIVE ({last:+.0f} cycles at n={ns[-1]}): work in "
            "flight cannot be\n         negative, so this slot has NO SIGNAL -- the true "
            f"backlog is smaller than\n         the {spread}-cycle spread `q_ctrl` "
            "measures for one raw point and the\n         subtraction has gone through "
            "zero. NO DEPTH IN ENTRIES IS REPORTED."
        )
    elif last <= floor:
        emit(
            f"         BACKLOG {last:+.0f} cycles at n={ns[-1]}, INSIDE THE NOISE FLOOR "
            f"({floor:.0f} = two\n         `q_ctrl` spreads of {spread}; the backlog is a "
            "difference of two differences\n         of raw single-shot points). A depth "
            "divided out of that would carry less\n         signal than the control's own "
            "scatter. NO DEPTH IN ENTRIES IS REPORTED."
        )
    elif last_step > QUEUE_FLATTEN_FRACTION * last:
        emit(
            "         BACKLOG STILL GROWING at the largest burst (last two doublings "
            f"added\n         {prev_step:+.0f} then {last_step:+.0f} cycles), so it is not "
            "an asymptote and no depth in\n         entries is resolvable. Either the queue "
            f"is deeper than n={ns[-1]}, or nothing\n         back-pressures this core at "
            "all -- which is what tt-sim is by construction."
        )
    else:
        emit(
            f"         BACKLOG FLATTENED at ~{last:.0f} cycles (last two doublings added "
            f"{prev_step:+.0f}\n         then {last_step:+.0f}), clearing a {floor:.0f}-cycle "
            f"noise floor, and at the drained rate\n         of {sync_rate[0]:.3f} cycles "
            f"per instruction that is ~{last / sync_rate[0]:.0f} INSTRUCTIONS in flight\n"
            "         -- a measurement of the Tensix instruction queue's depth in entries."
        )


#: Phase S: the reference burst, the issuing thread, and the probe names.
#:
#: THE REFERENCE BURST IS THE DESIGN. Phase Q's backlog subtracts its value at
#: the smallest burst it runs, n = 16, as ``tensix_sync()``'s own cost -- which
#: is only true where the queue is EMPTY at that burst. It is not: a core
#: pushing at 1/p instructions per cycle against a backend draining one every S
#: leaves ``n * (1 - p/S)`` outstanding, ~10 entries at n = 16 and one issuing
#: thread. At two and three threads a shared queue's per-thread share may be
#: *smaller* than that, which makes phase Q's multi-thread backlogs
#: structurally zero rather than merely small -- and zero is exactly what its
#: two banked runs read there. Phase S references n = 4 instead, where the same
#: arithmetic gives ~2 entries, so the correction is small and measured rather
#: than assumed away.
SHARE_MIN_N = 4
SHARE_ISSUER = 1
SHARE_BASELINE = "s_loop_addi"
SHARE_CO_PLAIN = "s_co_plain"
SHARE_CO_REPEAT = "s_co_repeat"
SHARE_CO_SYNC = "s_co_sync"
SHARE_SOLO_PLAIN = "s_solo_plain"
SHARE_SOLO_SYNC = "s_solo_sync"

#: How close to 1.0 the ratio of depths must be to read as a per-thread queue,
#: and how close to 1/k to read as a shared one. Neither is a fitted number:
#: the two hypotheses predict 1.00x and 1/k, which are a factor of k apart, and
#: anything landing between the bands is reported as ambiguous rather than
#: rounded to whichever is nearer.
SHARE_PRIVATE_RATIO = 0.75
SHARE_SHARED_SLACK = 1.25

#: Phase G's intermediates, one per ``--gset``. Named here so the presence
#: check can say that seeing one of them is a complete run.
SHARE_G_INTERMEDIATES = ("g_1280", "g_1536", "g_1792")


def _sharing_check(rows, emit):
    """Phase S: is the queue phase Q measured shared between the TRISCs?

    THE CONSTRUCTION, because the obvious one does not work. A shared queue of
    depth D and a per-thread queue of depth D are the same device seen from one
    thread, so the discriminator has to be a second thread that OCCUPIES queue
    entries. Two candidates do not:

    * a second thread that only **spins** pushes nothing, so it holds no entry
      under either hypothesis. It is run anyway, as ``s_solo_*``, because it is
      the control for "another core is awake" -- competing for instruction
      fetch out of the same L1 -- as against "another core is issuing".
    * a second thread issuing at a deliberately **low** rate holds ~0 entries
      too, and for a reason that is not about this benchmark: occupancy is
      arrival rate times residence time, so a thread served faster than it
      arrives is never queued however long it runs.

    Only a **saturated** second thread holds entries, and the price of
    saturation is that it takes backend bandwidth from the thread being
    measured. That is not a confound here because the drained rate is measured
    in the same slot by ``s_co_sync`` and divided back out.

    So the depth in entries is::

        D = backlog / S + SHARE_MIN_N * (1 - p/S)

    and the answer is ``D`` at k issuing threads against ``D`` at one:
    **equal means per-thread, 1/k means shared**. A level is never the answer.
    """
    slots = {}
    for row in rows:
        if row["phase"] != "s" or row["probe"] == CONTROL_PROBE:
            continue
        key = (row["variant"], row["active_threads"], row["thread"])
        slots.setdefault(key, {}).setdefault(row["probe"], {})[row["n"]] = row["cycles"]
    if not slots:
        return

    emit()
    emit("-" * 78)
    emit("Phase S: is the Tensix instruction queue shared, or one per thread?")
    emit("-" * 78)
    emit(
        "  EXPLORATORY, and the question phase Q left open. Nothing in either vendor\n"
        "  tree gives a Tensix instruction queue depth, let alone whether it is\n"
        "  replicated per thread.\n"
        "\n"
        "  ONLY A SATURATED SECOND THREAD OCCUPIES QUEUE ENTRIES. A spinning one holds\n"
        "  none under either hypothesis (it is the `s_solo_*` CONTROL below, not the\n"
        "  discriminator), and one issuing at a low rate holds ~0 by Little's law. The\n"
        "  backend bandwidth a saturated one takes is measured in the same slot by\n"
        f"  `{SHARE_CO_SYNC}` and divided back out.\n"
        "\n"
        f"      D = backlog / S + {SHARE_MIN_N} * (1 - p/S)\n"
        "\n"
        f"  with S from the `_sync` probe and p from `{SHARE_BASELINE}`. The second\n"
        f"  term is the n = {SHARE_MIN_N} reference burst's own occupancy, which phase Q's\n"
        "  read-out drops entirely -- so a phase-S depth at one thread should land\n"
        "  ABOVE phase Q's by roughly the difference, and if it does, phase Q's figure\n"
        "  is a lower bound.\n"
        "\n"
        "  EQUAL DEPTHS AT ONE AND k THREADS => PER-THREAD. 1/k => SHARED."
    )

    depths = {}
    for key in sorted(slots):
        variant, threads, thread = key
        probes = slots[key]
        ns = sorted({n for series in probes.values() for n in series})
        emit()
        emit(f"  [{variant} thread {thread}]  {threads} issuing thread(s)")
        emit(
            "      "
            + f"{'probe':<16}"
            + "".join(f"{n:>7}" for n in ns)
            + f"{'rate':>9}"
        )
        for probe in (
            SHARE_BASELINE,
            SHARE_CO_PLAIN,
            SHARE_CO_REPEAT,
            SHARE_CO_SYNC,
            SHARE_SOLO_PLAIN,
            SHARE_SOLO_SYNC,
        ):
            series = probes.get(probe)
            if not series:
                continue
            rate = _wide_rate(series, min_n=SHARE_MIN_N)
            emit(
                "      "
                + f"{probe:<16}"
                + "".join(f"{series[n]:>7}" if n in series else f"{'-':>7}" for n in ns)
                + (f"{rate[0]:>9.3f}" if rate else f"{'-':>9}")
            )
        noise = _sharing_noise(probes)
        emit(
            f"      repeatability: |{SHARE_CO_PLAIN} - {SHARE_CO_REPEAT}| <= {noise:.0f} "
            "cycles. Two identical\n        executions of one burst, so it is what ONE raw "
            "point can be wrong by."
        )
        co = _sharing_depth(probes, SHARE_CO_PLAIN, SHARE_CO_SYNC, noise)
        solo = _sharing_depth(probes, SHARE_SOLO_PLAIN, SHARE_SOLO_SYNC, noise)
        for label, depth in (("co-issuing", co), ("solo (others spin)", solo)):
            if depth is None:
                continue
            if depth["entries"] is None:
                emit(
                    f"      {label:<19} backlog {depth['backlog']:+.0f} cycles: "
                    f"{depth['refusal']}.\n        NO DEPTH IN ENTRIES IS REPORTED for this "
                    "slot."
                )
            else:
                emit(
                    f"      {label:<19} backlog {depth['backlog']:+.0f} cycles (last "
                    f"doubling {depth['step']:+.0f}),\n        drained {depth['rate']:.3f} | "
                    f"issue-limited {depth['issue']:.3f}  ->  DEPTH "
                    f"~{depth['entries']:.0f} ENTRIES"
                )
        if thread == SHARE_ISSUER:
            depths[threads] = (variant, co, solo)
    _sharing_verdict(depths, emit)


def _sharing_noise(probes):
    """This slot's single-shot repeatability, from two identical executions."""
    plain = probes.get(SHARE_CO_PLAIN) or {}
    repeat = probes.get(SHARE_CO_REPEAT) or {}
    shared = set(plain) & set(repeat)
    return max((abs(plain[n] - repeat[n]) for n in shared), default=0.0)


def _sharing_depth(probes, plain_name, sync_name, noise):
    """``{entries, backlog, step, rate, issue, refusal}`` for one plain/sync pair.

    The refusals are phase Q's, for phase Q's reasons: work in flight cannot be
    negative, a backlog inside twice one point's own scatter is the subtraction
    rather than the queue, and a backlog still growing at the longest burst has
    not reached the number it is growing towards. What differs is that the
    scatter is *measured here* -- by a byte-identical repeat of the probe --
    rather than inherited from a control that runs a different body.
    """
    plain = probes.get(plain_name)
    synced = probes.get(sync_name)
    if not plain or not synced:
        return None
    rate = _wide_rate(synced, min_n=SHARE_MIN_N)
    issue = _wide_rate(probes.get(SHARE_BASELINE) or {}, min_n=SHARE_MIN_N)
    ns = sorted(n for n in plain if n in synced and n >= SHARE_MIN_N)
    if rate is None or len(ns) < 3:
        return None
    base = synced[ns[0]] - plain[ns[0]]
    backlog = [synced[n] - plain[n] - base for n in ns]
    out = {
        "entries": None,
        "backlog": backlog[-1],
        "step": backlog[-1] - backlog[-2],
        "rate": rate[0],
        "issue": issue[0] if issue else 0.0,
        "refusal": "",
    }
    if backlog[-1] <= 0:
        out["refusal"] = "not positive, and work in flight cannot be"
    elif backlog[-1] <= QUEUE_NOISE_MULTIPLE * noise:
        out["refusal"] = (
            f"inside {QUEUE_NOISE_MULTIPLE:.0f}x this slot's measured repeatability"
        )
    elif out["step"] > QUEUE_FLATTEN_FRACTION * backlog[-1]:
        out["refusal"] = "still growing at the longest burst, so not an asymptote"
    elif rate[0] > 0:
        out["entries"] = backlog[-1] / rate[0] + SHARE_MIN_N * (
            1.0 - (out["issue"] / rate[0] if out["issue"] else 0.0)
        )
    return out


def _sharing_verdict(depths, emit):
    """The ratio, and only the ratio."""
    emit()
    base = depths.get(1)
    if base is None or base[1] is None or base[1]["entries"] is None:
        emit(
            "  NO VERDICT: the single-thread slot resolved no depth, so there is no\n"
            "  baseline to compare against. Against tt-sim this is forced --\n"
            "  `TensixFrontend.push_mop_instruction` is an unbounded list append, so the\n"
            "  backlog is still growing at every burst length in every slot and no depth\n"
            "  is resolvable anywhere. That is a fact about the simulator and says\n"
            "  nothing about any card."
        )
        return
    one = base[1]["entries"]
    compared = 0
    for threads in sorted(depths):
        if threads <= 1:
            continue
        variant, co, solo = depths[threads]
        if co is None or co["entries"] is None:
            emit(
                f"  {variant}: no depth resolved with {threads} issuing; nothing to compare."
            )
            continue
        compared += 1
        ratio = co["entries"] / one
        shared = 1.0 / threads
        if ratio >= SHARE_PRIVATE_RATIO:
            verdict = "PER-THREAD -- each core has its own queue"
        elif ratio <= SHARE_SHARED_SLACK * shared:
            verdict = "SHARED -- one queue, split between the issuers"
        else:
            verdict = "AMBIGUOUS -- between the two predictions"
        emit(
            f"  {variant}: {co['entries']:.0f} entries against t1's {one:.0f} = "
            f"{ratio:.2f}x (per-thread predicts 1.00x,\n    shared predicts "
            f"{shared:.2f}x)  ->  {verdict}"
        )
        if solo is not None and solo["entries"] is not None:
            emit(
                f"    control: with the others only SPINNING, {solo['entries']:.0f} entries "
                f"({solo['entries'] / one:.2f}x).\n    Both hypotheses predict 1.00x here; a "
                "departure is something other than queue\n    sharing, instruction fetch out "
                "of the shared L1 being the likeliest."
            )
    if compared == 0:
        emit(
            "  NO VERDICT: only one thread count is in this file, and the answer is a\n"
            "  comparison between thread counts. Run `--phase s --variants t1,t2,t3`."
        )


#: The two burst forms that have each resolved a Tensix instruction queue depth,
#: as ``(label, plain, sync, issue-limited baseline, reference burst)``. They are
#: the same experiment run twice with different constants, so their answers have
#: to be reconciled rather than quoted side by side -- which is what
#: :func:`_depth_reconcile` does.
RECONCILE_FORMS = (
    (
        "phase Q (16-instruction block)",
        QUEUE_LOOP_PLAIN,
        QUEUE_LOOP_SYNC,
        QUEUE_LOOP_BASELINE,
        QUEUE_MIN_N,
    ),
    (
        "phase S (4-instruction block)",
        SHARE_CO_PLAIN,
        SHARE_CO_SYNC,
        SHARE_BASELINE,
        SHARE_MIN_N,
    ),
)


def _depth_reconcile(rows, emit):
    """Do phase Q's and phase S's queue depths agree once the arithmetic is levelled?

    THEY ARE THE SAME MEASUREMENT WITH DIFFERENT CONSTANTS, and two things
    differ between them: the reference burst (n = 16 against n = 4) and the
    loop block (16 instructions against 4). Phase Q's read-out publishes
    ``backlog / S`` and drops the reference burst's own occupancy entirely;
    phase S carries it. So the first thing to do with two depths from one card
    is to recompute both under one estimator and see what is left.

    THE THIRD ESTIMATOR IS WHY THIS IS WORTH PRINTING. Once the core is
    back-pressured, ``plain[n] = n*S - runahead`` and the run-ahead is the queue
    read WITHOUT the ``_sync`` probe, without the reference burst and without
    ``tensix_sync()``'s own cost -- three of the four terms the other estimator
    depends on. If the forms disagree there too, the disagreement is not the
    correction arithmetic; it is the forms.
    """
    got = {}
    for row in rows:
        if row["variant"] != "t1" or row["thread"] != SHARE_ISSUER:
            continue
        got.setdefault(row["probe"], {})[row["n"]] = row["cycles"]
    read = [(label, _reconcile_form(got, *rest)) for label, *rest in RECONCILE_FORMS]
    read = [(label, d) for label, d in read if d is not None]
    if len(read) < 2:
        return

    emit()
    emit("-" * 78)
    emit("Do the two burst forms agree about the depth?")
    emit("-" * 78)
    emit(
        "  Phase Q and phase S measure ONE quantity twice, at one issuing thread, in\n"
        "  the same launch. They differ in the reference burst they subtract and in\n"
        "  the loop block they push from, so the two printed numbers are not\n"
        "  comparable until both are recomputed under one estimator.\n"
        "\n"
        "    bare        backlog / S                    -- phase Q's read-out prints this\n"
        "    levelled    bare + n_ref * (1 - p/S)       -- phase S's read-out prints this\n"
        "    run-ahead   (n*S + c - plain[n]) / S       -- neither `_sync` nor a reference\n"
        "\n"
        "  with `c` the timed region's fixed cost, taken from the issue-limited probe's\n"
        "  intercept. The two estimators share no term but `S`, so agreement between\n"
        "  them inside one form is a check and disagreement between forms is not."
    )
    emit()
    emit(
        f"  {'form':<32}{'n_ref':>6}{'p':>7}{'S':>7}{'backlog':>9}"
        f"{'bare':>8}{'levelled':>10}{'run-ahead':>11}"
    )
    for label, d in read:
        emit(
            f"  {label:<32}{d['n_ref']:>6}{d['issue']:>7.3f}{d['rate']:>7.3f}"
            f"{d['backlog']:>9.0f}{d['bare']:>8.1f}{d['levelled']:>10.1f}"
            f"{d['runahead']:>11.1f}"
        )
    _reconcile_verdict(read, emit)


def _reconcile_form(got, plain_name, sync_name, baseline_name, n_ref):
    """One form's three depth estimates, or ``None`` if it is not in this run."""
    plain, synced = got.get(plain_name), got.get(sync_name)
    if not plain or not synced:
        return None
    rate = _wide_rate(synced, min_n=n_ref)
    issue = _wide_rate(got.get(baseline_name) or {}, min_n=n_ref)
    ns = sorted(n for n in plain if n in synced and n >= n_ref)
    if rate is None or issue is None or rate[0] <= 0 or len(ns) < 3:
        return None
    backlog = (synced[ns[-1]] - plain[ns[-1]]) - (synced[ns[0]] - plain[ns[0]])
    # The marginal over the last doubling is the saturated service rate as the
    # PLAIN probe sees it, so the run-ahead is read off that probe alone. What
    # it needs from elsewhere is the timed region's own fixed cost -- the two
    # clock reads and the loop entry -- which the issue-limited baseline gives
    # as its intercept. Leaving it out biases the run-ahead DOWN by that many
    # cycles in both forms, so it cancels in a comparison but not in a level.
    marginal = (plain[ns[-1]] - plain[ns[-2]]) / float(ns[-1] - ns[-2])
    fixed = _timed_region_fixed_cost(got.get(baseline_name) or {}, n_ref)
    if marginal <= 0 or fixed is None:
        return None
    return {
        "n_ref": n_ref,
        "issue": issue[0],
        "rate": rate[0],
        "backlog": backlog,
        "bare": backlog / rate[0],
        "levelled": backlog / rate[0] + n_ref * (1.0 - issue[0] / rate[0]),
        "fixed": fixed,
        "runahead": (marginal * ns[-1] + fixed - plain[ns[-1]]) / marginal,
    }


def _timed_region_fixed_cost(baseline, n_ref):
    """The clock reads and loop entry, as the issue-limited probe's intercept.

    Read off the baseline's own last doubling rather than a fit through all its
    points, because its smallest bursts carry a cold instruction fetch that a
    fit would tilt the whole line to accommodate.
    """
    ns = sorted(n for n in baseline if n >= n_ref)
    if len(ns) < 2:
        return None
    slope = (baseline[ns[-1]] - baseline[ns[-2]]) / float(ns[-1] - ns[-2])
    return baseline[ns[-1]] - slope * ns[-1]


#: How far apart two forms' levelled depths may land and still be one answer.
#: One entry, because a depth in entries is an integer and the two estimators
#: below already differ from each other by a fraction of one.
RECONCILE_TOLERANCE = 1.0


def _reconcile_verdict(read, emit):
    """What the reference burst explains, and what is left over."""
    (lo_label, lo), (hi_label, hi) = sorted(read, key=lambda pair: pair[1]["levelled"])
    printed = hi["levelled"] - lo["bare"]
    correction = hi["levelled"] - hi["bare"] - (lo["levelled"] - lo["bare"])
    left = hi["levelled"] - lo["levelled"]
    emit()
    emit(
        f"  The two read-outs print {hi['levelled']:.0f} and {lo['bare']:.0f} entries, a gap of "
        f"{printed:+.1f}. The reference-burst\n"
        f"  terms differ by {-correction:.1f} entries ({lo['levelled'] - lo['bare']:.1f} at "
        f"n_ref = {lo['n_ref']} against {hi['levelled'] - hi['bare']:.1f} at\n"
        f"  n_ref = {hi['n_ref']}), which is what phase S was built to carry and phase Q's\n"
        "  read-out drops. Adding it back leaves"
    )
    if abs(left) <= RECONCILE_TOLERANCE:
        emit(
            f"      RECONCILED: {abs(left):.1f} entries between the two forms. Phase Q's figure\n"
            "      is a LOWER BOUND by exactly the term it drops, and the two\n"
            "      constructions measure the same queue."
        )
        return
    emit(
        f"      {left:+.1f} ENTRIES UNEXPLAINED, `{hi_label}` over\n"
        f"      `{lo_label}`. That is not the correction arithmetic: the\n"
        "      run-ahead estimator, which uses neither `_sync` probe nor either\n"
        f"      reference burst, puts the same two forms {hi['runahead'] - lo['runahead']:+.1f} entries apart.\n"
        "\n"
        "      SO THE DEPTH DEPENDS ON WHICH FORM MEASURES IT, and neither number may be\n"
        "      quoted alone. What the forms bracket is the finding; what separates them\n"
        "      is a property of the instrument that this run does not resolve."
    )


def _fetch_check(series, emit):
    """Phases F and G: does a bigger loop body cost more per instruction?

    The two phases are one measurement in two kernel BUILDS, and the split is
    forced rather than chosen -- see :data:`FOOTPRINT_PROBES`. Phase G's
    intermediates are each compiled against a ``g_1024`` anchor in their own
    build, so a phase-G step is read against that rather than across the build
    boundary; the table below prints both and the verdict says which it used.
    """
    values = [(nm, _measured(series, nm)) for nm in FOOTPRINT_PROBES]
    values = [(nm, v) for nm, v in values if v is not None]
    if len(values) < 2:
        return
    values.sort(key=lambda pair: (int(pair[0].split("_")[1]), pair[0]))
    emit()
    emit("-" * 78)
    emit("Phases F and G: instruction footprint")
    emit("-" * 78)
    emit(
        "  EXPLORATORY. `riscv.instruction_fetch` gives the fetch PERIOD -- one\n"
        '  128-bit L1 read per four instructions -- and its own note says "instruction\n'
        '  cache miss cost is not published". No cache SIZE is published either. So a\n'
        "  flat row is consistent with the documentation and a cliff is a number\n"
        "  nothing has ever printed; both are results.\n"
        "\n"
        "  WHAT A STEP LOCATES IS A BOUNDARY IN LOOP-BODY SIZE, and narrowing it does\n"
        "  not turn it into a cache capacity. A prefetch window, a TLB-like structure\n"
        "  or an L1 access pattern would all produce this column, and no document\n"
        "  distinguishes them. The `g_*` rows are phase G: the footprints between\n"
        "  1024 and 2048, one per `--gset`, each measured against a `g_1024` body\n"
        "  compiled into the SAME kernel -- phase F's six bodies already sit within a\n"
        "  few hundred bytes of tt-metal's kernel config buffer, so they could not\n"
        "  simply be added to it.\n"
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
            "  cheapest and dearest footprint. The step locates a boundary in\n"
            "  loop-body size that nothing publishes, and it is the first number this\n"
            "  benchmark produces that no document could have given."
        )
        _fetch_bracket(values, emit)


def _fetch_bracket(values, emit):
    """Where the step falls, as a bracket in bytes -- never as a cache size."""
    ordered = [(int(nm.split("_")[1]) * 4, nm, v) for nm, v in values]
    low = min(v for _, _, v in ordered)
    # "Stepped" is a per-instruction cost half way to the dearest reading, which
    # is well outside any resolution this instrument claims and needs no
    # threshold argument of its own.
    cut = low + 0.5 * (max(v for _, _, v in ordered) - low)
    flat = [b for b, _, v in ordered if v < cut]
    stepped = [b for b, _, v in ordered if v >= cut]
    if not flat or not stepped:
        return
    emit(
        f"  BRACKET: flat through a {max(flat)}-byte loop body, stepped by "
        f"{min(stepped)} bytes.\n"
        "  That is the whole claim. It is a boundary in loop-body size, measured\n"
        "  between two footprints that were run; it is not a cache size, and the\n"
        "  cost of crossing it is an AMORTISED figure over a body that is either\n"
        "  entirely resident or entirely not."
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
    # Phases Q and S sweep BURST LENGTH, so a least-squares slope through them
    # against the block-count control is not a per-instruction cost -- it comes
    # out negative, and a "per-core" verdict computed from two negative numbers
    # is worse than no verdict. Both phases have their own read-out and the
    # contention question is answered there, on their own terms.
    probes = sorted({s["probe"] for s in series if s["phase"] not in ("q", "s")})
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


def _additions_present(rows, emit):
    """Did phases S and G run at all? A null is not an absence.

    The same trap :func:`_live_check` exists for, in the form the two newer
    phases take it. Against tt-sim BOTH are forced: it models no instruction
    cache, so every phase-G footprint reads the same; and its Tensix queue is a
    list append, so no phase-S slot resolves a depth. A reader who cannot tell
    that from "the probe never ran" has learnt nothing from either, so this
    prints which probes produced points before any verdict is read.
    """
    present = {}
    for row in rows:
        if row["phase"] in ("s", "g") and row["probe"] != CONTROL_PROBE:
            present.setdefault(row["probe"], 0)
            present[row["probe"]] += 1 if row["cycles"] else 0
    if not present:
        return
    emit()
    emit("-" * 78)
    emit("Did the new phases run? (a forced null is not an absence)")
    emit("-" * 78)
    share = [p for p in present if p.startswith("s_")]
    fetch = [p for p in present if p.startswith("g_")]
    if share:
        emit(
            "  phase S: "
            + ", ".join(f"{p} {present[p]} points" for p in sorted(share))
            + "\n    Structural check: `s_co_sync` must exceed `s_co_plain` at every burst\n"
            "    length -- a drain cannot be free. If they are equal the phase measured\n"
            "    nothing, whatever the verdict above said. Against tt-sim the backlog\n"
            "    grows without bound and every slot refuses a depth: forced, and a fact\n"
            "    about the simulator."
        )
    if fetch:
        emit(
            "  phase G: "
            + ", ".join(f"{p} {present[p]} points" for p in sorted(fetch))
            + f"\n    Exactly ONE of {SHARE_G_INTERMEDIATES} is compiled per `--gset`, so a\n"
            "    file holding one of them is a complete run and not a truncated one.\n"
            "    Against tt-sim all footprints read alike: no instruction cache is\n"
            "    modelled, so the flat row is forced and says nothing about hardware."
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
