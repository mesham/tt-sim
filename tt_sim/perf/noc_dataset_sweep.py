"""Rung 2 of the calibration ladder: tt-metal's measured NoC dataset, swept.

``docs/plans/cost-model.md`` lists four rungs of validation below "captured
silicon traces". Rung 1 (internal consistency against tt-metal's four
hardcoded end-to-end figures) is climbed. This module is rung 2:

    tt-metal's measured NoC dataset. Entries of empirically measured
    end-to-end latency keyed by transaction size, access pattern, memory type
    and subordinate count, shipped in tt-metal's ``noc_estimator``. Too coarse
    to *derive* a per-hop congestion model from -- it folds everything into one
    number -- but exactly the right thing to *validate* one against.

The distinction that makes this worth doing: **not one term of the assembled
model was fitted to this dataset.** The hop latencies and the flit width are
``isa_doc``; the DRAM access latency is ``vendor_source_derived`` from two
*different* vendor numbers (``tm_dm_common``'s hardcoded 358 and 259). So every
comparison below is out of sample.

What the dataset actually is
----------------------------

``tt_metal/impl/experimental/noc_estimator/latencies/noc_latencies.yaml``:
740 entries x 11 transaction sizes (64 B .. 64 KiB) = 8,140 measured points,
each a **device cycle count**, not a time. The measurement is the duration of
a ``DeviceZoneScopedN`` region in the data-movement suite's estimator kernels
(``noc_estimator_tests/kernels/{reader,writer,dram_reader,dram_writer}.cpp``)
that wraps *the whole issue loop plus the closing barrier*: N ``noc_async_*``
calls followed by one ``noc_async_*_barrier``. So it includes the issuing
RISC-V core's register writes and its barrier polling. Since 2026-08-08 the
prediction includes them too -- :func:`predict_timed_region` runs that program
on a simulated BRISC rather than poking the registers from Python, which is
what :func:`predict_cycles` did and still does. See
:data:`RESIDUAL_EXPECTATION` and :mod:`tt_sim.perf.noc_issue_loop`.

Three things about the keying are worth stating because they are easy to get
wrong and none of them is documented:

* ``num_transactions`` in a YAML key is **transactions per barrier**, not the
  total: ``noc_estimator.cpp`` builds its lookup key with
  ``.num_transactions = params.num_transactions_per_barrier`` and then
  multiplies the looked-up latency by ``ceil(total / per_barrier)``.
* The pattern enum in ``types.hpp`` (``ONE_FROM_ONE = 0``) is **not** the one
  in the test that produced the data (``ONE_TO_ONE = 0``). They are reconciled
  by the CSV carrying the pattern's *name*, which ``csv_reader.cpp`` maps to
  the ``types.hpp`` value -- so YAML ``pattern: 0`` is a read and ``pattern: 1``
  is a write.
* ``noc_index`` is 0 on all 740 entries and carries no information: the kernels
  log the column as ``"NoC Index"`` and ``csv_reader.cpp`` looks for
  ``"NOC index"``, so it never matches and every point takes the default. Rows
  measured on NoC 0 and NoC 1 therefore collide on one key. This does not
  affect anything here, because a *round trip* on a directional torus costs the
  same number of hops on either NoC.

What it says about congestion, which is the largest ``unknown`` left
--------------------------------------------------------------------

Fourteen entries per architecture are retained and ~330 are dropped, most of
them because they contain multi-party traffic that ``noc.congestion``
(``provenance: unknown``) does not model. The obvious question is whether the
dropped rows could *supply* that model. They cannot, and :data:`CONGESTION_VERDICT`
sets out why with the arithmetic: the file never separates the number of
concurrent flows from the geometry (both are changed by resizing one grid), it
records no core coordinates, and its one scalar per key already contains the
issuing core's loop and the endpoint's L1 ports. A coefficient fitted through
that would be ``provenance: estimated``, which this cost model does not have.

So the sweep now *reports* the dropped rows rather than only counting them --
what is in them, how many rules each one fails, how much closing any single gap
would actually unlock, and the measured size of the term that is missing. The
last of those is the useful half of a null result: the dataset bounds congestion
even though it cannot model it, which is exactly the "validate, not derive" role
``docs/plans/cost-model.md`` gave rung 2.

Run it
------

::

    python3 -m tt_sim.perf.noc_dataset_sweep                  # auto-locate
    python3 -m tt_sim.perf.noc_dataset_sweep --dataset PATH
    python3 -m tt_sim.perf.noc_dataset_sweep --arch blackhole

The dataset lives outside this repo, so with no tt-metal checkout the script
prints where it looked and exits 0 -- the same "degrade gracefully" contract
``examples/examples_test.py`` uses for its tt-metal dependency.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# The dataset.
# ---------------------------------------------------------------------------

#: Path under a tt-metal checkout.
DATASET_SUFFIX = Path(
    "tt_metal/impl/experimental/noc_estimator/latencies/noc_latencies.yaml"
)

#: Every key field and the value the loader assumes when the entry omits it,
#: from ``types.hpp``'s ``DEFAULT_*`` constants. The YAML only writes
#: non-default fields, so an entry cannot be read without these.
KEY_DEFAULTS = {
    "mechanism": 0,  # UNICAST
    "pattern": 1,  # ONE_TO_ONE  (a write)
    "memory": 0,  # L1
    "arch": 2,  # WORMHOLE_B0
    "num_transactions": 1,  # ... per barrier
    "num_subordinates": 1,
    "same_axis": False,
    "stateful": False,
    "loopback": False,
    "noc_index": 0,
}

ARCH_IDS = {"wormhole": 2, "blackhole": 3}
PATTERN_READ, PATTERN_WRITE = 0, 1
PATTERN_ALL_TO_ALL = 4
MEMORY_L1, MEMORY_DRAM_INTERLEAVED, MEMORY_DRAM_SHARDED = 0, 1, 2

#: ``types.hpp``'s ``NocPattern`` and ``NocMechanism``, in declaration order,
#: so the report can name what it drops instead of printing an enum value.
#: Note ``ONE_FROM_*`` is a read and ``ONE_TO_*`` a write.
PATTERN_NAMES = (
    "ONE_FROM_ONE",
    "ONE_TO_ONE",
    "ONE_FROM_ALL",
    "ONE_TO_ALL",
    "ALL_TO_ALL",
    "ALL_FROM_ALL",
    "ONE_TO_ROW",
    "ROW_TO_ROW",
    "ONE_TO_COLUMN",
    "COLUMN_TO_COLUMN",
)
MECHANISM_NAMES = ("UNICAST", "MULTICAST", "MULTICAST_LINKED")

#: Largest transaction the DRAM sweep actually issues. ``dram_accessor_sweep``
#: caps at ``max_pages_per_txn = 256`` pages of ``obtain_page_size_bytes`` = 32
#: bytes. The offline ``data_extractor`` then pads every standard size above
#: the largest measured one by repeating the last value, so a DRAM row's
#: 16 KiB / 32 KiB / 64 KiB entries are fill, not measurements.
DRAM_SWEEP_MAX_BYTES = 8192


def default_dataset_path(env=None):
    """Where to look for the dataset, or ``None``.

    ``TT_SIM_NOC_LATENCIES`` wins if set; otherwise the same
    ``TT_METAL_RUNTIME_ROOT`` / ``TT_METAL_HOME`` pair the examples runner
    uses.
    """
    env = os.environ if env is None else env
    explicit = env.get("TT_SIM_NOC_LATENCIES")
    if explicit:
        return Path(explicit)
    root = env.get("TT_METAL_RUNTIME_ROOT") or env.get("TT_METAL_HOME")
    return Path(root) / DATASET_SUFFIX if root else None


def load_dataset(path):
    """``[(key_dict, [latency per standard size]), ...]`` plus the size list."""
    raw = yaml.safe_load(Path(path).read_text())
    sizes = list(raw["transaction_sizes"])
    entries = []
    for entry in raw["entries"]:
        key = dict(KEY_DEFAULTS)
        key.update(entry["key"])
        entries.append((key, [float(v) for v in entry["latencies"]]))
    return entries, sizes


# ---------------------------------------------------------------------------
# The exclusion criteria. DECLARED BEFORE ANY RESIDUAL WAS COMPUTED.
# ---------------------------------------------------------------------------
#
# This ordering matters more than the criteria do. Dropping entries because
# they *disagree* is fitting, not validating, and it would make the whole
# exercise worthless -- so each rule below names a term tt-sim does not model,
# and every one of them could have been written without the dataset in hand.
# The cost of each in entries is reported by the sweep so a reader can see how
# much of the dataset the model is declining to be tested against.

RESIDUAL_EXPECTATION = """\
The measurement is a DeviceZoneScopedN region wrapping N ``noc_async_*`` calls
and the closing barrier, so it contains the issuing RISC-V core's own
instruction stream as well as the network round trip and the endpoint's
service time.

Until 2026-08-08 the prediction did not: it drove the initiator's NoC
registers from Python, and the sweep recorded the difference as a constant
77-94 cycle residual described as "one unmodelled issuing-core path". THAT
DESCRIPTION WAS WRONG. The path was never unmodelled -- tt-sim runs baby
RISC-V cores against a published pipeline and a published load-latency table,
and a kernel issuing a NoC transaction pays for its own stores and polls like
any other code. What was missing is that the HARNESS never ran them. It does
now (`predict_timed_region`, `tt_sim.perf.noc_issue_loop`), and this adds no
cost-table entry and moves no simulated cycle outside this harness.

So the residual is still NOT expected to be zero, but what is left is
narrower and of a different kind: the profiler's own instrumentation (the
DeviceZone timestamp reads and their L1 stores) and the barrier's discovery
granularity -- one poll iteration between the last acknowledgement landing and
the core noticing. Neither is a property of the hardware and neither may
become a cost-table entry.

It is still expected to be a CONSTANT: the same in every row, independent of
transaction size, of geometry, of memory type, of direction AND NOW of the
number of transactions per barrier. That last axis is the new one, and it is
the sharpest of the five: it is the only axis along which the issuing core's
cost accumulates linearly, so structure in it points at the issue path
specifically rather than at any endpoint term."""


CONGESTION_VERDICT = """\
CAN A CONGESTION MODEL BE DERIVED FROM THIS DATASET? No -- and the reason is
identifiability rather than coarseness, which is a sharper claim than the one
`docs/plans/cost-model.md` recorded ("too coarse ... it folds everything into
one number"). Three specifics, all checkable from the table printed above:

1. CONCURRENCY AND GEOMETRY ARE NEVER SEPARATED. Every multi-party row varies
   the number of concurrent flows by changing the GRID SIZE -- the sweep runs
   2x2, 3x3, 5x5, 8x8 and the full device grid, all anchored at logical (0, 0)
   (`test_noc_estimator.cpp`, `run_all_to_all`). So the flow count, the path
   lengths and the number of links each path shares all move together, in four
   or five steps, and no two rows hold any one of them fixed. A per-hop
   congestion coefficient is a slope against "flows sharing this link"; there
   is no such axis in the file.

2. THE FILE RECORDS NO COORDINATES. `num_subordinates` is the only geometric
   field on a multi-party row. Per-flow distances are recoverable only by
   re-deriving the grid from the test source, and even then the file holds ONE
   aggregate cycle count per key -- so a fitted term would have to explain a
   whole grid's barrier through a single scalar.

3. THAT SCALAR ALREADY CONTAINS TWO OTHER UNMODELLED TERMS. The retained rows
   leave a residual of order 100 cycles, which is the issuing core's own path,
   and a multi-party row pays that once per transaction issued. The per-
   transaction line printed above is the check: at the smallest transaction
   size the all-to-all rows cost a small fraction of one round trip each, so
   they are pipelined and what sets them is the ISSUE LOOP, not the network.
   Endpoint L1 port arbitration (`noc.l1_ports`: two 128-bit read and two
   128-bit write connections, shared by every flow arriving at a core) is
   folded into the same number. One equation, three unknowns, and every extra
   configuration in the file moves all three at once.

Fitting a coefficient anyway would produce a number no document prints and no
arithmetic on vendor numbers yields -- `provenance: estimated`, which this cost
model does not have and will not gain here. `vendor_source_derived` is not
available either: it requires arithmetic on published vendor numbers (as
`dram.access_latency` does), and there is no arithmetic that isolates
congestion from the two terms above.

WHAT THE DATASET *CAN* DO, and it is worth having: it bounds the size of the
missing term, and it is a validation target. The all-to-all series above is a
measured saturation curve -- aggregate bandwidth rising far slower than the
core count while the per-core share collapses -- and any congestion model tt-sim
ever gains must reproduce it. That is exactly the role rung 2 was climbed for.

WHAT WOULD DERIVE ONE, on a card, from tt-metal's own microbenchmarks in
`tests/tt_metal/tt_metal/data_movement/` (which expose the coordinates the
shipped YAML drops, and write per-core profiler CSVs):

  a. `one_to_one` / `one_from_one` (test IDs 4, 5, 50, 51), sweeping
     `master_core_coord` x `subordinate_core_coord` across the grid. Gives
     latency against a KNOWN hop count -- the dataset only has the same-axis /
     different-axis pair -- and pins the uncongested line everything else is
     differenced against.
  b. `all_to_all` / `all_from_all` (300-308, 310-318) with `sub_grid_size`
     PINNED to 1x1 and `mst_grid_size` swept 1x1, 2x1, 2x2, 4x2, 4x4 at a
     fixed `mst_logical_start_coord`. N flows into one endpoint at a fixed
     distance: the endpoint-contention curve with geometry held still, which
     is the axis (1) says the shipped dataset lacks.
  c. The same pair with N FIXED at 2 and the two masters slid so their routes
     share 0, 1, 2 ... k links (same row -> maximal overlap, different rows ->
     none). The slope of latency against shared links IS the per-hop
     congestion coefficient, measured with everything else constant.
  d. `core_bidirectional` (140-148) with `write_vc` swept 0-3, which isolates
     virtual-channel arbitration -- the one congestion mechanism the ISA docs
     name ("if the two packets have the same virtual circuit number, then one
     packet will wait for the other") -- from distance and from endpoint ports.

(a) and (c) together are the minimum: an intercept and a slope, both measured
with the confounds held fixed rather than fitted out. Neither needs anything
tt-sim can supply, and neither can be substituted by more of this file.

THAT LIST IS NOW A HARNESS, and one correction to it is worth stating here
because it changes what (a) can deliver. The data_movement suite CANNOT be
parameterised: every coordinate, grid size and virtual channel in it is a
compile-time literal in a gtest body, and the one test whose name promises
otherwise (`TensixDataMovementOneToOneCustom`) is GTEST_SKIPped. So the harness
is a tt-metal program of this repository's own -- `perfbench/nocbench`, planned
by `tt_sim.perf.noc_congestion_plan` and read back by
`tt_sim.perf.noc_congestion_sweep`. It also corrects (a): sweeping master x
subordinate does NOT give a fine-grained hop sweep, because on a directional
torus a round trip costs grid_x, grid_y or grid_x + grid_y hops and nothing
else, whatever the coordinates. What (a) actually yields is a three-level line
plus a flatness test that would catch the routing model being wrong."""


def _exclusions(arch_id):
    """The retained-set predicate, as an ordered ladder of named rules."""
    return [
        (
            "arch",
            "swept one architecture at a time; see --arch",
            lambda k: k["arch"] == arch_id,
        ),
        (
            "mechanism != UNICAST",
            "tt-sim models a multicast as N unicast deliveries sharing one "
            "injection port. The hardware's router-level fan-out, and the "
            "arbitration between the copies, is not modelled at all.",
            lambda k: k["mechanism"] == 0,
        ),
        (
            "pattern not in {ONE_FROM_ONE, ONE_TO_ONE}",
            "every other pattern has >= 2 concurrent initiators or targets "
            "sharing links, which is congestion -- `noc.congestion` is "
            "`provenance: unknown` and nothing in tt-sim models it. The "
            "*_ALL patterns also sweep grids whose per-core distances the "
            "dataset does not record, so there is no hop count to predict.",
            lambda k: k["pattern"] in (PATTERN_READ, PATTERN_WRITE),
        ),
        (
            "num_transactions per barrier != 1, outside the L1 write path",
            "N > 1 is a pipelined burst, and what governs it is the issuing "
            "core's per-transaction cost. That is now modelled and validated "
            "for ONE ISSUE PATH -- the single-target unicast L1 write, which "
            "`tt_sim.perf.noc_issue_loop` reconstructs from the vendor headers "
            "and which the model reproduces to 0.0 cycles per transaction on "
            "both architectures. The other three are still declined, each for "
            "its own reason. (a) A READ burst is additionally limited by the "
            "initiator's outstanding-request credit: the measured marginal is "
            "27 cycles/transaction on Wormhole and 35 on Blackhole against an "
            "issue loop costing 18 and 19, and tt-sim's NIU imposes no such "
            "limit. The ISA docs state the protocol -- software must not write "
            "NOC_CMD_CTRL again until it reverts to 0 -- but publish no timing "
            "for it, so the term would be `provenance: unknown` and may carry "
            "no number. (b) The DRAM rows issue through `TensorAccessor`, a "
            "different instruction sequence (~3 cycles/transaction longer) "
            "that is not reconstructed. (c) Multi-party rows run the nested "
            "`WRITER_MODE_UNICAST_MULTI` loop, which reloads a destination "
            "coordinate per subordinate; also not reconstructed, and those "
            "rows are congestion rows anyway.",
            lambda k: (
                k["num_transactions"] == 1
                or (
                    k["pattern"] == PATTERN_WRITE
                    and k["memory"] == MEMORY_L1
                    and k["mechanism"] == 0
                )
            ),
        ),
        (
            "stateful",
            "stateful mode is an issue-side optimisation (`set_state` + "
            "`with_state` reuses the configured NoC command registers). "
            "tt-sim charges no per-transaction NoC register configuration "
            "cost, so it has no term that could distinguish the two and "
            "would predict them identical by construction.",
            lambda k: not k["stateful"],
        ),
        (
            "loopback",
            "a multicast-linked feature; nothing in tt-sim expresses it.",
            lambda k: not k["loopback"],
        ),
        (
            "memory == DRAM_INTERLEAVED",
            "interleaving spreads pages round-robin over 12 channels at "
            "different distances. tt-sim instantiates one DRAM tile and has "
            "no interleaving model.",
            lambda k: k["memory"] != MEMORY_DRAM_INTERLEAVED,
        ),
    ]


def retained(entries, arch_id):
    """``(kept, [(rule, removed, remaining), ...])`` -- the ladder, in order."""
    kept, ladder = entries, []
    for name, _reason, keep in _exclusions(arch_id):
        nxt = [e for e in kept if keep(e[0])]
        ladder.append((name, len(kept) - len(nxt), len(nxt)))
        kept = nxt
    return kept, ladder


#: What tt-sim would have to gain before a rule could be retired, one line per
#: ladder rule. The rules' own ``reason`` strings say why an entry is dropped;
#: this says what closing it would take, which is the question a reader of the
#: ladder actually has. Keyed by rule name and pinned to the ladder by a test,
#: so a new rule cannot be added without answering the question.
MISSING_TERM = {
    "arch": "nothing -- run the sweep again with --arch",
    "mechanism != UNICAST": (
        "router-level multicast fan-out: one packet replicated by the routers "
        "rather than N packets injected, and an arbitration policy between the "
        "copies"
    ),
    "pattern not in {ONE_FROM_ONE, ONE_TO_ONE}": (
        "a congestion term (`noc.congestion`, provenance: unknown) AND the "
        "per-core geometry of the grid, which the dataset does not record"
    ),
    "num_transactions per barrier != 1, outside the L1 write path": (
        "for reads, the initiator's outstanding-read-request credit limit -- "
        "worth 9 cycles per transaction on Wormhole and 16 on Blackhole, "
        "measured, and unpublished, so it cannot be charged at any provenance "
        "rank this cost model has; for the rest, reconstructions of the "
        "`TensorAccessor` and multi-subordinate issue paths alongside the "
        "plain one in `tt_sim.perf.noc_issue_loop`"
    ),
    "stateful": "a per-transaction NoC command-register configuration cost",
    "loopback": "the multicast fan-out above, of which this is a flag",
    "memory == DRAM_INTERLEAVED": (
        "more than one DRAM tile, and the interleaver's page-to-channel map"
    ),
}


def exclusion_multiplicity(entries, arch_id):
    """``(by_count, sole_cause)`` for the non-arch rules.

    ``by_count[k]`` is how many of this architecture's entries are excluded by
    exactly ``k`` rules; ``sole_cause[rule]`` is how many are excluded by that
    rule **and no other**.

    The second one is the number the ladder cannot show and that a reader
    invariably wants. A rule reported as "removes 150" reads like 150 entries
    waiting on one missing term; ``sole_cause`` is how many would actually
    become answerable if that term arrived and nothing else did. The two are
    wildly different here, and pretending otherwise would over-sell every gap
    in the model at once.
    """
    rules = _exclusions(arch_id)[1:]  # the arch rule is a selection, not a gap
    by_count, sole_cause = {}, {name: 0 for name, _r, _k in rules}
    for key, _latencies in entries:
        if key["arch"] != arch_id:
            continue
        failed = [name for name, _r, keep in rules if not keep(key)]
        by_count[len(failed)] = by_count.get(len(failed), 0) + 1
        if len(failed) == 1:
            sole_cause[failed[0]] += 1
    return by_count, sole_cause


def dropped_by_shape(entries, arch_id):
    """``{(mechanism, pattern): count}`` over the entries the ladder drops."""
    rules = _exclusions(arch_id)[1:]
    shape = {}
    for key, _latencies in entries:
        if key["arch"] != arch_id:
            continue
        if all(keep(key) for _n, _r, keep in rules):
            continue
        name = (MECHANISM_NAMES[key["mechanism"]], PATTERN_NAMES[key["pattern"]])
        shape[name] = shape.get(name, 0) + 1
    return shape


def concurrency_series(entries, sizes, arch_id, size):
    """The dataset's own answer to "what does concurrency cost?", per arch.

    ``[(cores, cycles, aggregate B/cycle, per-core B/cycle), ...]`` over the
    all-to-all rows carrying exactly **one transaction per (master, subordinate)
    pair per barrier** -- the only shape in the file where the number of
    concurrent flows is the only thing that changes between rows of the same
    pattern. ``num_transactions`` equals ``num_subordinates`` there, which is
    how those rows are found.

    This is the most this dataset can say about congestion, and saying it is
    not the same as modelling it: see :data:`CONGESTION_VERDICT`.
    """
    if size not in sizes:
        return []
    column = sizes.index(size)
    rows = []
    for key, latencies in entries:
        if key["arch"] != arch_id or key["memory"] != MEMORY_L1:
            continue
        if key["pattern"] != PATTERN_ALL_TO_ALL or key["mechanism"] != 0:
            continue
        cores = key["num_subordinates"]
        if cores < 2 or key["num_transactions"] != cores:
            continue
        cycles = latencies[column]
        if cycles <= 0:
            continue
        # N masters x N subordinates, one transaction each per barrier.
        moved = cores * cores * size
        rows.append((cores, cycles, moved / cycles, moved / cycles / cores))
    return sorted(rows)


def single_transaction_baseline(entries, sizes, arch_id, size, same_axis=False):
    """One ``size``-byte write, one transaction, no contention -- or ``None``.

    The row every multi-party number is worth comparing against, and the same
    row the primary sweep retains, so the comparison is between a measurement
    the model predicts and one it does not.
    """
    if size not in sizes:
        return None
    column = sizes.index(size)
    for key, latencies in entries:
        if key["arch"] != arch_id or key["memory"] != MEMORY_L1:
            continue
        if key["pattern"] != PATTERN_WRITE or key["mechanism"] != 0:
            continue
        if key["num_transactions"] != 1 or key["stateful"]:
            continue
        if bool(key["same_axis"]) != same_axis:
            continue
        return latencies[column]
    return None


def point_is_measured(key, size):
    """False for a size the offline extractor filled in rather than measured.

    The only case in the retained set is a DRAM row above
    :data:`DRAM_SWEEP_MAX_BYTES`: ``dram_accessor_sweep`` never issues one, and
    ``data_extractor.cpp`` pads the standard sizes above the largest measured
    one by repeating the last value. A structural fact about how the file was
    generated, visible in the raw data as a flat tail, and nothing to do with
    whether the model agrees.
    """
    return key["memory"] == MEMORY_L1 or size <= DRAM_SWEEP_MAX_BYTES


# ---------------------------------------------------------------------------
# Geometry: the one thing the dataset does not record and the model needs.
# ---------------------------------------------------------------------------
#
# ``run_single_core`` in ``test_noc_estimator.cpp`` fixes it:
#
#     CoreCoord master_coord = {0, 0};
#     CoreCoord sub_coord = cfg.same_axis ? CoreCoord{0, 1} : CoreCoord{1, 1};
#
# so ``same_axis`` means the two cores share their logical x, i.e. they are in
# the same column and differ only in y. The DRAM rows use logical core {0, 0}
# and DRAM bank 0.
#
# That is enough, and it is enough for a reason worth stating: on a directional
# torus a round trip between two tiles differing on both axes is exactly
# ``grid_x + grid_y`` hops *whatever* the distance between them, and one
# sharing an axis is exactly the other dimension. So the hop count is fixed by
# ``same_axis`` alone -- no core-placement assumption is doing any work, and
# the answer is the same on NoC 0 and NoC 1.

#: Logical -> tt-sim tile coord for the two cores the single-core patterns use.
#: Wormhole tiles are keyed by unified coord (18, 18) == SoC-physical (1, 1);
#: Blackhole tiles are keyed by physical NoC coord directly.
GEOMETRY = {
    "wormhole": {"master": (18, 18), "same_axis": (18, 19), "diff_axis": (19, 19)},
    "blackhole": {"master": (1, 2), "same_axis": (1, 3), "diff_axis": (2, 3)},
}

_L1_ADDR = 0x30000
_DRAM_ADDR = 0x1000


def _physical_of(arch, tile_coord):
    """SoC-physical NoC 0 coord of a tt-sim tile coord.

    Wormhole keys its tiles by *unified* coord (the 16-25 band) and Blackhole
    by the physical NoC coord directly, so this is the one place the two
    conventions have to be reconciled. The hop counts in this module are all
    computed in physical NoC space.
    """
    if arch == "wormhole":
        from tt_sim.device.wormhole import Wormhole

        return Wormhole.physical_noc0_coord_from_unified_worker(tile_coord)
    return tuple(tile_coord)


def _build_device(arch, tensix_coords):
    if arch == "wormhole":
        from tt_sim.device.wormhole import Wormhole

        return Wormhole(tensix_coords=list(tensix_coords))
    from tt_sim.device.blackhole import Blackhole

    return Blackhole(tensix_coords=list(tensix_coords))


def _set_target_coord(initiator, which, coord):
    from tt_sim.network.noc_coords import WormholeNocCoords

    x, y = coord
    if isinstance(initiator.nui.noc_coord_strategy, WormholeNocCoords):
        setattr(initiator, f"{which}_addr_mid", (x << 4) | (y << 10))
    else:
        setattr(initiator, f"{which}_addr_hi", (y << 6) | x)


def predict_cycles(arch, memory, is_read, same_axis, size, budget=500_000):
    """Cycles the **assembled model** takes for one transaction, measured.

    Not a closed form: this builds a real device with the cost model on, drives
    one NoC transaction through the initiator's registers exactly as tt-metal
    firmware does, and pumps until ``NIU_MST_REQS_OUTSTANDING`` returns to zero
    -- which is precisely what ``noc_async_read_barrier`` /
    ``noc_async_write_barrier`` wait for, and therefore precisely where the
    measured DeviceZone ends. Every term is exercised through its real
    consumer: the hop model and the injection port in ``tt_sim/network/``,
    burst splitting at ``noc_max_burst_size``, and the DRAM endpoint's service
    window in ``tt_sim/device/tiles.py``.
    """
    from tt_sim.network.tt_noc import NUI

    outstanding = NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
    geometry = GEOMETRY[arch]
    master = geometry["master"]
    sub = geometry["same_axis" if same_axis else "diff_axis"]

    if memory == MEMORY_L1:
        device = _build_device(arch, [master, sub])
        local, remote = device.tensix_tiles[0], device.tensix_tiles[1]
        remote_addr = _L1_ADDR
    else:
        device = _build_device(arch, [master])
        local, remote = device.tensix_tiles[0], device.dram_tiles[0]
        remote_addr = _DRAM_ADDR

    payload = bytes((i * 7) & 0xFF for i in range(size))
    source = remote if is_read else local
    device.write(source.get_coord_pair(), remote_addr if is_read else _L1_ADDR, payload)

    router = local.noc0_router
    initiator = router.request_initiators[0]
    initiator.packet_tag = 0
    initiator.at_len_be = size
    if is_read:
        _set_target_coord(initiator, "target", remote.noc0_router.id_pair)
        initiator.target_addr_low = remote_addr
        initiator.ret_addr_low = _L1_ADDR
        initiator.ctrl = 0
    else:
        initiator.target_addr_low = _L1_ADDR
        _set_target_coord(initiator, "ret", remote.noc0_router.id_pair)
        initiator.ret_addr_low = remote_addr
        # mode 2 = write, bit 4 = NOC_CMD_RESP_MARKED (non-posted: an ACK is
        # returned and OUTSTANDING tracks it, which is what a write barrier
        # waits on).
        initiator.ctrl = 2 | (1 << 4)
    initiator.cmd_ctrl = 1
    initiator.initiate()

    for cycle in range(1, budget + 1):
        device.run(1)
        if router.nui_counters[outstanding] == 0:
            return cycle
    raise RuntimeError(f"no completion within {budget} cycles for {arch} {size} B")


#: Memo for :func:`predict_timed_region`. A 256 x 64 KiB burst is half a
#: million simulated cycles, and the report and its tests ask for the same
#: point repeatedly; the answer is a pure function of the key.
_TIMED_REGION_CACHE = {}


def predict_timed_region(
    arch, memory, is_read, same_axis, size, num_transactions=1, budget=4_000_000
):
    """Cycles the model takes for the region the measurement actually times.

    :func:`predict_cycles` drives the initiator's registers from Python, so it
    predicts the network and the endpoint and *nothing the issuing core does*.
    The dataset's number is a ``DeviceZoneScopedN`` region wrapping N
    ``noc_async_*`` calls and the closing barrier, so it contains a RISC-V
    instruction stream as well.

    This runs that instruction stream: it loads
    :func:`~tt_sim.perf.noc_issue_loop.issue_loop_program` onto the initiator
    tile's BRISC, releases it, and times the core's own ``START -> DONE``
    markers -- the loop, the barrier's drain, and nothing else. Every cycle it
    costs is charged by cost-table entries that already existed; this adds no
    term and moves nothing outside the harness. See
    :mod:`tt_sim.perf.noc_issue_loop` for what the program is and why it is a
    reconstruction rather than a constant.
    """
    from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
    from tt_sim.perf import noc_issue_loop as loop
    from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

    # The cost model is a process-wide switch, so it is part of the key.
    memo = (
        arch,
        memory,
        is_read,
        same_axis,
        size,
        num_transactions,
        os.environ.get("TT_SIM_COST_MODEL"),
    )
    if memo in _TIMED_REGION_CACHE:
        return _TIMED_REGION_CACHE[memo]

    geometry = GEOMETRY[arch]
    master = geometry["master"]
    sub = geometry["same_axis" if same_axis else "diff_axis"]

    if memory == MEMORY_L1:
        device = _build_device(arch, [master, sub])
        local, remote = device.tensix_tiles[0], device.tensix_tiles[1]
        remote_addr = _L1_ADDR
    else:
        device = _build_device(arch, [master])
        local, remote = device.tensix_tiles[0], device.dram_tiles[0]
        remote_addr = _DRAM_ADDR

    coord = _coord_word(arch, remote)
    payload = bytes((i * 7) & 0xFF for i in range(size))
    source = remote if is_read else local
    coord_pair = local.get_coord_pair()
    device.write(source.get_coord_pair(), remote_addr if is_read else _L1_ADDR, payload)

    for index, word in enumerate(loop.issue_loop_program(arch, is_read)):
        device.write(coord_pair, loop.PROGRAM_ADDR + 4 * index, conv_to_bytes(word))
    # mode 0 = read; mode 2 = write, bit 4 = NOC_CMD_RESP_MARKED (non-posted,
    # so an ACK returns and OUTSTANDING tracks it -- what a barrier waits on).
    params = (
        coord,
        remote_addr if is_read else _L1_ADDR,  # NOC_TARG_ADDR_LO
        _L1_ADDR if is_read else remote_addr,  # NOC_RET_ADDR_LO
        size,
        num_transactions,
        0 if is_read else (2 | (1 << 4)),
    )
    for index, value in enumerate(params):
        device.write(coord_pair, loop.PARAMS_ADDR + 4 * index, conv_to_bytes(value))
    for marker in (loop.START_ADDR, loop.DONE_ADDR):
        device.write(coord_pair, marker, conv_to_bytes(0))

    device.reset_tile(coord_pair)
    device.deassert_soft_reset(coord_pair, BabyRISCVCoreType.BRISC)

    started = None
    for cycle in range(1, budget + 1):
        device.run(1)
        if started is None:
            if conv_to_uint32(bytes(device.read(coord_pair, loop.START_ADDR, 4))):
                started = cycle
            continue
        if conv_to_uint32(bytes(device.read(coord_pair, loop.DONE_ADDR, 4))):
            device.shutdown()
            _TIMED_REGION_CACHE[memo] = cycle - started
            return cycle - started
    device.shutdown()
    raise RuntimeError(
        f"no completion within {budget} cycles for {arch} {size} B x {num_transactions}"
    )


def _coord_word(arch, remote):
    """The value the kernel stores into ``NOC_*_ADDR_COORDINATE``."""
    from tt_sim.network.noc_coords import WormholeNocCoords

    x, y = remote.noc0_router.id_pair
    if isinstance(remote.noc0_router.noc_coord_strategy, WormholeNocCoords):
        return (x << 4) | (y << 10)
    return (y << 6) | x


# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------


class Residual:
    """One measured point and what the model said about it."""

    __slots__ = ("key", "size", "measured", "predicted")

    def __init__(self, key, size, measured, predicted):
        self.key = key
        self.size = size
        self.measured = measured
        self.predicted = predicted

    @property
    def residual(self):
        return self.measured - self.predicted

    @property
    def relative(self):
        return self.residual / self.measured

    @property
    def label(self):
        memory = {MEMORY_L1: "L1", MEMORY_DRAM_SHARDED: "DRAM"}[self.key["memory"]]
        direction = "read" if self.key["pattern"] == PATTERN_READ else "write"
        axis = "same-axis" if self.key["same_axis"] else "diff-axis"
        n = self.key["num_transactions"]
        burst = "" if n == 1 else f" x{n}"
        return f"{memory} {direction} {axis}{burst}"


def sweep(entries, sizes, arch, drop_unmeasured=True, issue_loop=True):
    """Predict every retained point. Returns ``[Residual, ...]``.

    ``issue_loop`` selects which predictor answers: the default runs the
    issuing core's real instruction stream (:func:`predict_timed_region`), and
    ``False`` falls back to the network-and-endpoint-only
    :func:`predict_cycles` this sweep used before 2026-08-08. The report prints
    both so the difference is visible rather than asserted.
    """
    cache = {}
    rows = []
    for key, latencies in entries:
        for size, measured in zip(sizes, latencies):
            if drop_unmeasured and not point_is_measured(key, size):
                continue
            signature = (
                key["memory"],
                key["pattern"] == PATTERN_READ,
                key["same_axis"],
                size,
            )
            burst = key["num_transactions"]
            if (signature, burst) not in cache:
                cache[(signature, burst)] = (
                    predict_timed_region(arch, *signature, num_transactions=burst)
                    if issue_loop
                    else predict_cycles(arch, *signature)
                )
            rows.append(Residual(key, size, measured, cache[(signature, burst)]))
    return rows


def _linear_fit(xs, ys):
    """Least-squares ``(intercept, slope)``; ``(mean, 0)`` for a degenerate x."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var
    return mean_y - slope * mean_x, slope


def _summary(values):
    values = sorted(values)
    quartile = len(values) // 4
    return {
        "n": len(values),
        "min": values[0],
        "p25": values[quartile],
        "median": statistics.median(values),
        "p75": values[-1 - quartile],
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def _grouped(rows, key_fn):
    groups = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def report(entries, sizes, arch, out=sys.stdout):
    arch_id = ARCH_IDS[arch]
    kept, ladder = retained(entries, arch_id)

    def emit(line=""):
        print(line, file=out)

    emit("=" * 78)
    emit(f"Rung 2: tt-metal measured NoC dataset vs the assembled model [{arch}]")
    emit("=" * 78)
    emit()
    emit(
        f"dataset: {len(entries)} entries x {len(sizes)} sizes "
        f"= {len(entries) * len(sizes)} measured points, in device cycles"
    )
    emit()
    emit("Exclusion ladder (declared before any residual was computed):")
    total = len(entries)
    for name, removed, remaining in ladder:
        emit(f"  - {name:<44} removes {removed:>4}  leaves {remaining:>4}")
    dropped_points = sum(
        1 for key, _ in kept for size in sizes if not point_is_measured(key, size)
    )
    emit(
        f"  - {'sizes the DRAM sweep never issued':<44} removes "
        f"{dropped_points:>4} points (not entries)"
    )
    emit()
    emit(
        f"retained: {len(kept)} of {total} entries, "
        f"{len(kept) * len(sizes) - dropped_points} points"
    )
    emit()
    emit(RESIDUAL_EXPECTATION)
    emit()

    if not kept:
        emit("nothing retained; nothing to check.")
        return []

    rows = sweep(kept, sizes, arch)

    emit("-" * 78)
    emit("Per-point: measured (tt-metal) vs predicted (tt-sim), cycles")
    emit("-" * 78)
    emit(f"{'':<32}" + "".join(f"{s:>8}" for s in sizes))
    for group_label, group in sorted(_grouped(rows, lambda r: r.label).items()):
        emit()
        by_size = {r.size: r for r in group}
        for name, attr in (
            ("measured", "measured"),
            ("predicted", "predicted"),
            ("residual", "residual"),
        ):
            cells = "".join(
                f"{getattr(by_size[s], attr):>8.0f}" if s in by_size else f"{'-':>8}"
                for s in sizes
            )
            label = group_label if attr == "measured" else ""
            emit(f"{label:<22}{name:<10}{cells}")

    emit()
    emit("-" * 78)
    emit("Residual distribution")
    emit("-" * 78)
    absolute = _summary([r.residual for r in rows])
    relative = _summary([100 * r.relative for r in rows])
    emit(f"  n = {absolute['n']}")
    emit(
        f"  cycles : min {absolute['min']:>7.0f}  p25 {absolute['p25']:>7.0f}  "
        f"median {absolute['median']:>7.0f}  p75 {absolute['p75']:>7.0f}  "
        f"max {absolute['max']:>7.0f}"
    )
    emit(
        f"  percent: min {relative['min']:>7.1f}  p25 {relative['p25']:>7.1f}  "
        f"median {relative['median']:>7.1f}  p75 {relative['p75']:>7.1f}  "
        f"max {relative['max']:>7.1f}"
    )
    emit()
    over = sorted({r.label for r in rows if r.residual < 0})
    if not over:
        emit(
            "  The model is a FLOOR everywhere: every residual is positive, "
            "which is the\n  direction every bound in these tables is chosen "
            "to lean."
        )
    else:
        emit(
            "  The model is NOT a floor everywhere. It OVER-CHARGES these "
            "rows, i.e. it\n  invents back-pressure the hardware does not "
            "have:"
        )
        for label in over:
            group = [r for r in rows if r.label == label and r.residual < 0]
            worst = min(r.residual for r in group)
            emit(f"    {label:<22} {len(group)} points, worst {worst:.0f} cycles")

    emit()
    emit("-" * 78)
    emit("Residual by axis -- where the constant stops being constant")
    emit("-" * 78)
    for axis_name, key_fn in (
        (
            "geometry (tests the hop term)",
            lambda r: "same-axis" if r.key["same_axis"] else "diff-axis",
        ),
        (
            "memory (tests the DRAM access-latency term)",
            lambda r: {MEMORY_L1: "L1", MEMORY_DRAM_SHARDED: "DRAM"}[r.key["memory"]],
        ),
        (
            "direction",
            lambda r: "read" if r.key["pattern"] == PATTERN_READ else "write",
        ),
        (
            "transactions per barrier (tests the issue loop)",
            lambda r: f"N={r.key['num_transactions']}",
        ),
    ):
        emit()
        emit(f"  {axis_name}")
        for name, group in sorted(_grouped(rows, key_fn).items()):
            small = [r for r in group if r.size <= 512]
            stats = _summary([r.residual for r in group])
            emit(
                f"    {name:<12} n={stats['n']:<4} median {stats['median']:>7.0f}"
                f"   median at <= 512 B {statistics.median([r.residual for r in small]):>7.0f}"
            )

    emit()
    emit("  size (tests the bandwidth term)")
    xs = [r.size for r in rows]
    ys = [float(r.residual) for r in rows]
    intercept, slope = _linear_fit(xs, ys)
    emit(
        f"    least squares over all points: residual = {intercept:.0f}"
        f" + {slope * 1024:.2f} cycles/KiB"
    )
    for name, group in sorted(_grouped(rows, lambda r: r.label).items()):
        gx = [r.size for r in group]
        gy = [float(r.residual) for r in group]
        gi, gs = _linear_fit(gx, gy)
        emit(f"    {name:<22} residual = {gi:>6.0f} + {gs * 1024:>6.2f} cycles/KiB")

    emit()
    emit("  implied sustained bandwidth, from the large-transfer slope")
    emit("  (N = 1 rows only: a burst row's slope is N transfers, not one)")
    sourced = _sourced_bandwidths(arch)
    single = [r for r in rows if r.key["num_transactions"] == 1]
    for name, group in sorted(_grouped(single, lambda r: r.label).items()):
        big = sorted([r for r in group if r.size >= 4096], key=lambda r: r.size)
        if len(big) < 2:
            continue
        d_bytes = big[-1].size - big[0].size
        measured_rate = d_bytes / (big[-1].measured - big[0].measured)
        model_rate = d_bytes / (big[-1].predicted - big[0].predicted)
        emit(
            f"    {name:<22} measured {measured_rate:>6.2f} B/cycle"
            f"   model {model_rate:>6.2f} B/cycle"
            f"   ({100 * measured_rate / model_rate:.0f} % of modelled peak)"
        )
    emit()
    emit("    for reference, the bandwidth figures the tables already hold:")
    for label, value in sourced.items():
        emit(f"      {label:<40} {value}")

    _issue_loop_readout(kept, sizes, arch, rows, emit)
    _bandwidth_ceiling_check(entries, sizes, arch_id, emit)
    _dropped_readout(entries, sizes, arch_id, emit)
    return rows


def _issue_loop_readout(kept, sizes, arch, rows, emit):
    """What running the issuing core's own program is worth, measured.

    The previous predictor is still here, so the two can be run side by side on
    the same points rather than compared against a number written down last
    week. The difference is the whole of this section's claim.
    """
    before = sweep(kept, sizes, arch, issue_loop=False)
    emit()
    emit("-" * 78)
    emit("The issuing core's own program, on and off")
    emit("-" * 78)
    emit(
        "  `predict_cycles` drives the NoC registers from Python and predicts the\n"
        "  network alone; `predict_timed_region` runs the loop the kernel runs. Same\n"
        "  points, same cost tables, no new term -- the only difference is whether the\n"
        "  issuing core's instructions are executed."
    )
    emit()
    emit(
        f"    {'series':<26}{'residual, network only':>24}{'+ issue loop':>16}"
        f"{'closed':>9}"
    )
    by_label_before = _grouped(before, lambda r: r.label)
    for label, group in sorted(_grouped(rows, lambda r: r.label).items()):
        old = by_label_before.get(label)
        if not old:
            continue
        gi, _ = _linear_fit([r.size for r in old], [float(r.residual) for r in old])
        ni, _ = _linear_fit([r.size for r in group], [float(r.residual) for r in group])
        emit(f"    {label:<26}{gi:>24.0f}{ni:>16.0f}{gi - ni:>9.0f}")

    def _single(rowset):
        return [r for r in rowset if r.key["num_transactions"] == 1]

    for scope, old, new in (
        ("all retained points", before, rows),
        (
            "N = 1 only, i.e. exactly the points this sweep retained before",
            _single(before),
            _single(rows),
        ),
        (
            "N = 1, <= 512 B, where no bandwidth term is in play",
            [r for r in _single(before) if r.size <= 512],
            [r for r in _single(rows) if r.size <= 512],
        ),
    ):
        emit()
        emit(f"    {scope}")
        for name, group in (("network only", old), ("+ issue loop", new)):
            stats = _summary([r.residual for r in group])
            emit(
                f"      {name:<14} n={stats['n']:<4} min {stats['min']:>6.0f}  "
                f"median {stats['median']:>6.0f}  max {stats['max']:>6.0f}  "
                f"mean {stats['mean']:>6.0f}"
            )
    emit(
        "\n  Read the three scopes together. The first widens in absolute spread and\n"
        "  that is honest: the new rows are burst rows, so they multiply the L1\n"
        "  bandwidth shortfall this file already records by up to 256. The second\n"
        "  and third are like-for-like, and there the residual only tightens.\n"
        "\n  What is left is the profiler's own instrumentation and the barrier's\n"
        "  discovery granularity. Neither is a hardware property, so neither is a\n"
        "  candidate for the cost tables."
    )


def _dropped_readout(entries, sizes, arch_id, emit):
    """What is in the entries the ladder drops, and what would consume them.

    The ladder says "removes 150" and nothing about what the 150 are, which
    makes every gap in the model look the same size. It is not: on both
    architectures the *sole* cause of exclusion for all but a handful of
    entries is more than one missing term at once, so closing any single one
    unlocks almost nothing. That is a fact about the model's distance from this
    dataset and it belongs in the report rather than in a reader's head.
    """
    by_count, sole_cause = exclusion_multiplicity(entries, arch_id)
    shape = dropped_by_shape(entries, arch_id)
    if not shape:
        return
    emit()
    emit("-" * 78)
    emit("What is in the rows the ladder drops")
    emit("-" * 78)
    emit(
        "  The ladder above counts entries as it removes them, in order, so a rule's\n"
        "  number depends on where it sits. This section does not: every count below\n"
        "  is over ALL of this architecture's entries, against every rule at once."
    )
    emit()
    emit("  Entries by how many rules exclude them:")
    for count in sorted(by_count):
        label = "retained" if count == 0 else f"{count} rule{'s' if count > 1 else ''}"
        emit(f"    {label:<12} {by_count[count]:>4}")
    emit()
    emit(
        "  SOLE CAUSE -- entries that ONE rule alone keeps out, i.e. what would become\n"
        "  answerable if that term arrived and nothing else did:"
    )
    for name, count in sorted(sole_cause.items(), key=lambda kv: (-kv[1], kv[0])):
        emit(f"    {count:>4}  {name}")
        emit(f"          needs: {MISSING_TERM.get(name, '?')}")
    emit()
    emit(
        "  Read that column before reading the ladder as a to-do list. The rule that\n"
        "  removes the most entries is not the one that would unlock the most: a\n"
        "  multi-party row is typically ALSO a multi-transaction row, and a multicast\n"
        "  row is typically both. Congestion is never the only thing in the way."
    )
    emit()
    emit("  The dropped set, by mechanism and pattern:")
    emit(f"    {'mechanism':<18}{'pattern':<16}{'entries':>8}")
    for (mechanism, pattern), count in sorted(shape.items()):
        emit(f"    {mechanism:<18}{pattern:<16}{count:>8}")

    biggest = max(sizes)
    series = concurrency_series(entries, sizes, arch_id, biggest)
    if series:
        emit()
        emit(
            f"  The size of the missing term, measured. All-to-all over an N-core grid,\n"
            f"  one {biggest} B transaction per (master, subordinate) pair per barrier -- the\n"
            "  only shape in the file where the flow count is the only thing that changes:"
        )
        emit(
            f"    {'cores':>6}{'cycles':>12}{'aggregate B/cyc':>18}{'per core B/cyc':>16}"
        )
        for cores, cycles, aggregate, per_core in series:
            emit(f"    {cores:>6}{cycles:>12.0f}{aggregate:>18.1f}{per_core:>16.2f}")
        first, last = series[0], series[-1]
        core_ratio = last[0] / first[0]
        emit(
            f"    -> {core_ratio:.0f}x the cores buys {last[2] / first[2]:.1f}x the aggregate "
            f"bandwidth, so the\n       per-core share falls {first[3] / last[3]:.1f}x. "
            "Whatever that is -- link queueing, L1\n       port arbitration, the issue loop -- "
            "tt-sim charges none of it, and this is\n       the scale of what it is not charging."
        )
    # The same rows at the SMALLEST size, which is where the third claim of
    # CONGESTION_VERDICT is checkable rather than asserted: divide the barrier
    # by the transactions each master issued into it and compare against one
    # transaction's whole round trip.
    smallest = min(sizes)
    small = concurrency_series(entries, sizes, arch_id, smallest)
    baseline = single_transaction_baseline(entries, sizes, arch_id, smallest)
    if small and baseline:
        emit()
        emit(
            f"  The same rows at {smallest} B, per transaction ISSUED (barrier cycles over the\n"
            "  transactions one master put into it):"
        )
        emit(
            "    "
            + "  ".join(
                f"N={cores} {cycles / cores:.0f}" for cores, cycles, _a, _p in small
            )
        )
        emit(
            f"    against {baseline:.0f} cycles for ONE {smallest} B round trip on its own "
            "(ONE_TO_ONE,\n    different axis). A transaction inside the grid costs a fraction "
            "of a round trip,\n    so these rows are pipelined and what governs them is the "
            "per-transaction ISSUE\n    cost -- which is the same term the retained rows leave "
            "in the residual, not\n    congestion. That is claim 3 below, in numbers."
        )
    emit()
    emit(CONGESTION_VERDICT)


def _sourced_bandwidths(arch):
    """The bandwidth constants already in the tables, for the report to quote.

    Read out of the YAML rather than restated here, so a table edit moves the
    reference number too.
    """
    from tt_sim.perf.costs import SOURCED_PROVENANCE, load_costs

    sections = load_costs(arch).sections
    noc = sections.get("noc") or {}
    dram = (sections.get("dram") or {}).get("bandwidth") or {}
    out = {}
    # Provenance is checked, not assumed. Blackhole's ``dram.bandwidth``
    # override is ``provenance: unknown``, and because the arch overrides are
    # deep-merged the Wormhole per-channel figure is still *present* under it.
    # Quoting it for Blackhole would be exactly the laundering the convention
    # exists to stop.
    if noc.get("flit_bits") and noc.get("provenance") in SOURCED_PROVENANCE:
        out["noc.flit_bits / 8 (B per cycle per link)"] = noc["flit_bits"] // 8
    if (
        dram.get("per_channel_gb_per_s")
        and dram.get("provenance") in SOURCED_PROVENANCE
    ):
        out["dram.bandwidth.per_channel_gb_per_s"] = dram["per_channel_gb_per_s"]
    # The DRAM figure the model actually spends, which on Blackhole is the only
    # one there is: that arch has no per-channel GB/s but does have a derived
    # bytes-per-cycle. Same provenance check, per direction, and a direction
    # the table declines simply does not appear.
    channel = (sections.get("dram") or {}).get("channel_serialisation") or {}
    if channel.get("provenance") in SOURCED_PROVENANCE:
        read = channel.get("bytes_per_cycle_read")
        write = channel.get("bytes_per_cycle_write")
        if read is None and write is None:
            out["dram.channel_serialisation (B/cycle, both ways)"] = channel.get(
                "bytes_per_cycle"
            )
        else:
            # A directional key replaces the deep-merged flat one, exactly as
            # ``DramCostModel`` reads it, so the other arch's figure cannot
            # appear here either.
            if read is not None:
                out["dram.channel_serialisation (B/cycle, read)"] = read
            if write is not None:
                out["dram.channel_serialisation (B/cycle, write)"] = write
        # The occupancy axis, where a direction the latency axis declines may
        # still be charged: this sweep's rows are all one transaction per
        # barrier, so nothing here can see it, and leaving it out of the table
        # would read as "not charged at all".
        occupancy = channel.get("write_occupancy") or {}
        if occupancy.get("provenance") in SOURCED_PROVENANCE:
            out["dram.channel_serialisation (B/cycle, write occupancy only)"] = (
                occupancy.get("bytes_per_cycle")
            )
    return out or {"(none sourced for this architecture)": ""}


def _bandwidth_ceiling_check(entries, sizes, arch_id, emit):
    """Secondary, dataset-only: does any measurement beat the modelled link?

    The primary sweep excludes every entry with more than one transaction per
    barrier, because their *latency* folds in an issue loop tt-sim does not
    model. Their **bandwidth** is a different question and a one-sided one: the
    model says one NIU's injection link carries ``flit_bits / 8`` bytes per
    cycle and no more, so a measured single-initiator transfer that exceeded it
    would falsify the constant outright, whatever the issue path costs. That
    check needs no prediction and no simulation, so those 40 entries are worth
    something after all -- just not what the primary sweep wanted from them.
    """
    from tt_sim.perf.costs import load_costs

    arch = {v: k for k, v in ARCH_IDS.items()}[arch_id]
    flit_bits = (load_costs(arch).sections.get("noc") or {}).get("flit_bits")
    if not flit_bits:
        return
    ceiling = flit_bits / 8

    best = {}
    for key, latencies in entries:
        if key["arch"] != arch_id or key["mechanism"] != 0:
            continue
        if key["pattern"] not in (PATTERN_READ, PATTERN_WRITE):
            continue
        if key["memory"] != MEMORY_L1:
            continue
        direction = "read" if key["pattern"] == PATTERN_READ else "write"
        for size, latency in zip(sizes, latencies):
            rate = size * key["num_transactions"] / latency
            record = best.setdefault(direction, (0.0, None))
            if rate > record[0]:
                best[direction] = (rate, (key["num_transactions"], size))

    emit()
    emit("-" * 78)
    emit("Secondary check (dataset only, no simulation): the bandwidth ceiling")
    emit("-" * 78)
    emit(
        f"  the model's per-NIU injection link: {ceiling:.0f} B/cycle "
        f"(noc.flit_bits = {flit_bits}, isa_doc)"
    )
    emit("  fastest single-initiator L1 transfer anywhere in the dataset,")
    emit("  including the entries the primary sweep excluded:")
    for direction, (rate, where) in sorted(best.items()):
        transactions, size = where
        emit(
            f"    {direction:<6} {rate:>6.2f} B/cycle "
            f"({100 * rate / ceiling:>5.1f} % of the ceiling)"
            f"   at {transactions} x {size} B"
        )
    exceeded = [d for d, (rate, _) in best.items() if rate > ceiling]
    emit(
        "  -> nothing exceeds it, so the constant is not falsified."
        if not exceeded
        else f"  -> EXCEEDED by {exceeded}: the flit-width constant is wrong."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--arch", default="wormhole", choices=sorted(ARCH_IDS))
    args = parser.parse_args(argv)

    path = args.dataset or default_dataset_path()
    if path is None or not path.exists():
        print(
            "noc_latencies.yaml not found"
            + (f" at {path}" if path else "")
            + ".\nSet TT_SIM_NOC_LATENCIES, or TT_METAL_RUNTIME_ROOT to a "
            "tt-metal checkout,\nor pass --dataset. Skipping (the dataset "
            "lives outside this repo)."
        )
        return 0

    os.environ["TT_SIM_COST_MODEL"] = "1"
    entries, sizes = load_dataset(path)
    report(entries, sizes, args.arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
