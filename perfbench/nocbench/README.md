# nocbench — measuring NoC congestion on a card

You have a Tenstorrent card. This asks it four questions the public
documentation does not answer, and one that nobody appears to have checked.
It takes **a few minutes** and produces one CSV to send back.

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal   # built with ./build_metal.sh
cd perfbench/nocbench/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
../run_card.sh                                      # ~2–5 minutes
# -> nocbench-<arch>.csv   <- this is the file to send back
```

`run_card.sh` does three things: dumps your card's core map, plans the
experiments against *that* map, and runs them. It prints a verdict at the end.
If the verdict is anything other than `NO CONGESTION EFFECT` or
`CONGESTION MEASURED`, see "Telling a good run from a degenerate one" below —
the CSV is still worth sending, with the console output.

Nothing here writes to a card's flash, changes any clock, or needs root. It
allocates L1 scratch, runs a data-movement kernel on a handful of Tensix cores,
and reads back cycle counts.

## Why

`tt_sim/perf/unit_costs.yaml` records the cost of a NoC hop, a flit, a DRAM
access and a dozen other things, each with a citation. One entry has no number
at all:

```yaml
congestion:
  provenance: unknown
  note: the docs describe the router buffers and say congestion "can negatively
        impact latency" without quantifying it
```

tt-metal ships 740 rows of measured NoC latency, and
[`tt_sim/perf/noc_dataset_sweep.py`](../../tt_sim/perf/noc_dataset_sweep.py)
established that a congestion model **cannot be derived from them** — not
because they are coarse but because they are *unidentifiable*. Every
multi-party row varies the number of concurrent flows by resizing a grid, so
flow count, path length and link sharing all move together; no core coordinates
are recorded at all; and the single number per row already contains the issuing
core's software loop and the destination's L1 port arbitration. One equation,
three unknowns.

This harness is the answer to that: the same measurement with **the coordinates
recorded** and **exactly one thing varying per experiment**.

## What it measures

| | experiment | held fixed | varying |
| --- | --- | --- | --- |
| 1 | `hops` | one flow; the master core; size; count; direction; NoC | the subordinate's coordinate |
| C | `size` | one flow; both coordinates; count | bytes per transaction |
| C | `selfport` | the master core; both subordinates' distance; sizes | whether a second flow shares the master's injection port |
| 2 | `shared` | **two** flows; flow A entirely; flow B's hop counts; sizes; counts; **every leg-pair link overlap except one, at zero** | how many links the two payloads share |
| 3 | `contention` | the subordinate; every master's hop count; sizes | the number of flows |
| 4 | `vc` | both coordinates; sizes; counts | the write's virtual channel (0–3) |

Rows marked **C** are controls, not results, and they are not optional — see
below. Experiment 2 is the decisive one: **the slope of latency against shared
links is the per-hop congestion coefficient**, and it is the measurement the
shipped dataset structurally cannot provide.

The full pre-declared predictions — what each experiment shows under each
hypothesis, *including what "no effect" looks like* — are printed by:

```bash
python3 -m tt_sim.perf.noc_congestion_plan --hypotheses
```

They were written before anything was run, and they are in the source rather
than in a report so that they cannot be edited after the fact without it
showing in the diff.

## Telling a good run from a degenerate one

A congestion measurement can fail in a way that looks exactly like a result:
if the two flows never actually run at the same time, the latency does not
change, and "no congestion" and "measured nothing" are the same reading. So the
analysis refuses to call a flat result a result unless three things hold, and it
tells you which one failed:

1. **`size` must rise.** Bigger transactions must cost more. If they do not, the
   timing is not connected to the traffic at all.
2. **`selfport` must rise.** A second flow out of the *same* core's NIU must
   cost more than one, because they serialise on that core's injection link.
   This is what proves the two-flow machinery — the rendezvous, the concurrency
   — actually makes flows contend.
3. **The flows must have overlapped in time.** Each kernel stamps its raw start
   and end wall clock, so the analysis computes what fraction of the shorter
   flow ran while the other was also running. Below 50 % and the run is
   rejected.

Plus two structural checks that catch the run having measured a *different*
experiment from the one planned: every kernel reports its own NoC coordinates
(they must match the plan) and every plan column that the experiment declared
fixed is re-checked in the returned file.

The possible verdicts:

| verdict | means |
| --- | --- |
| `CONGESTION MEASURED` | a coefficient came out; the report gives it per transaction size, with a shape (`LINEAR` or `SATURATING`) |
| `NO CONGESTION EFFECT` | flat **and** the controls moved **and** the flows overlapped. A real, reportable result |
| `PARTIAL` | controls passed but the file has no shared-link experiment |
| `INVALID` | one of the checks above failed; the report names which |

Against the **simulator** a flat shared-link reading is forced: tt-sim charges
an NIU for its own injection port and nothing whatever for a router-to-router
link, so two flows sharing links cannot interact. Running it there exercises
everything except the effect being looked for. In practice the simulator run
comes back `INVALID` rather than `NO CONGESTION EFFECT`, because the self-port
control does not move either — see "What has been verified" at the end. That is
the machinery working as designed: on a device whose only contention mechanism
is inert, a flat reading is not evidence of anything.

## What the upstream suite cannot express

tt-metal's own `tests/tt_metal/tt_metal/data_movement/` suite is the right
*shape* for this work — it exposes coordinates, unlike the shipped YAML, and it
writes per-core profiler CSVs. It is not usable as it stands, and the reason is
worth stating precisely rather than worked around:

**Every parameter is a compile-time literal inside a gtest body.** There is no
environment variable, no gtest flag and no argument that selects
`master_core_coord`, `subordinate_core_coord`, `mst_grid_size`,
`sub_grid_size`, `mst_logical_start_coord` or `write_vc`. They are C++ default
arguments on helper functions called from `TEST_F` bodies with literal
`CoreCoord(0, 0)` / `CoreCoord(0, 1)`. The one test whose name promises
otherwise, `TensixDataMovementOneToOneCustom`, begins with `GTEST_SKIP()` and
takes its coordinates from those same defaults. A grep for `getenv` across the
whole suite finds five hits, all of them `TT_METAL_SIMULATOR` in the Quasar
tests.

So `one_to_one` swept over master × subordinate is not expressible; placing two
masters at a chosen link overlap is not expressible; `all_to_all` with
`sub_grid_size` pinned to 1×1 is not expressible. What *would* make the suite
usable, as an upstream change: read the existing config structs' fields from
environment variables (or a small JSON), the way the estimator sweeps already
read their own parameters — `OneToOneConfig`, `AllToAllConfig` and
`CoreBidirectionalConfig` already have exactly the right fields, and only the
call sites hard-code them. That is a small change and it is a change to
tt-metal, so this harness does not make it: it is a standalone tt-metal program
in this repository instead, built the same way the examples are.

Two further things the suite could not have given even if parameterised, both
of which this harness supplies:

* a **rendezvous**, so two flows measurably overlap in time rather than merely
  being launched together;
* **positive controls**, so that a flat reading is distinguishable from a dead
  one.

## What experiment 1 can and cannot give

Asking for "latency against a known hop count, swept over the grid" sounds like
it should give a fine-grained line. It cannot, and this is a property of the
hardware rather than of the harness.

Both NoCs are **directional** tori: a packet only ever travels in the increasing
direction of that NoC's coordinates, wrapping past the edge rather than turning
round. So a request and its reply go the *same* way round the ring, and a round
trip costs

* `grid_x` hops between any two cores in one row,
* `grid_y` hops between any two cores in one column,
* `grid_x + grid_y` hops between any pair differing on both axes,

**whatever the coordinates**. Three values on Blackhole: 17, 12, 29. So the
sweep yields a three-level line, not a continuum — and the flatness *within* a
level is itself the interesting measurement, twice over. It is a falsifiable
test of the routing model (if latency rises with `|dx|` along a row, the NoC is
not the directional ring every hop count in the cost model assumes), and its
spread is a direct measurement of this harness's noise floor, which every "flat"
claim elsewhere is then read against.

## Running the pieces separately

```bash
# 1. what cores does this card have, and what coordinate does a kernel use?
./build/nocbench --dump-grid                      # -> nocbench-grid-<arch>.csv

# 2. plan the experiments against THAT map (this is where the invariants are
#    asserted; it refuses to emit a confounded plan)
python3 -m tt_sim.perf.noc_congestion_plan \
    --grid nocbench-grid-blackhole.csv --out plan.csv --experiments all

# 3. run it
./build/nocbench --plan plan.csv -v               # -> nocbench-<arch>.csv

# 4. read it
python3 -m tt_sim.perf.noc_congestion_sweep --measured nocbench-blackhole.csv
```

Useful flags: `--experiments hops,size,selfport,shared` (the minimum),
`--num-tx N` (transactions per flow; more is slower and quieter),
`--tx-bytes N`, `--noc 1`.

Step 2 will **refuse** rather than emit a plan whose experiments are confounded,
and it will refuse a card whose coordinate dump it cannot map onto physical NoC
coordinates — a harvested part being the case it names explicitly. A refusal
here is the harness working: a shared-link count computed for the wrong
geometry would be indistinguishable from a measurement.

## Against the simulator

Same binary, different environment; `perfbench/run.sh` sets it up. tt-sim only
materialises the worker tiles it is told about, and a multi-core plan touches
many, so the plan file records the exact value to use in a comment
(`# tt_sim_tensix_coords=...`):

```bash
COORDS=$(grep -o 'tt_sim_tensix_coords=[^ ]*' plan.csv | cut -d= -f2)
TT_METAL_HOME=/path/to/tt-metal TT_SIM_TENSIX_COORDS="$COORDS" TT_SIM_COST_MODEL=1 \
    ../../run.sh nocbench -- --plan plan.csv
```

Keep it small: the simulator runs tens of thousands of cycles per second where
the card runs a billion, so use `--num-tx 4 --tx-bytes 512` and a couple of
experiments rather than the full plan.

## What to send back

`nocbench-<arch>.csv` and the console output of step 4. The CSV carries the plan
it came from (including the per-flow hop counts and shared-link counts) in the
same rows as the measurements, so it is self-describing — nothing else is
needed to interpret it.

If a coefficient does come out of it, note what it is and is not: a measurement
on one part, which under
[`docs/plans/cost-model.md`](../../docs/plans/cost-model.md)'s provenance
convention is `vendor_source` at best and certainly not `isa_doc`. It would be
the first number in either cost table measured rather than read.

## What has been verified, and what has not

**Nobody has run this on a card.** Everything below is against tt-sim, which
models no router-to-router congestion at all, so the experiment this harness
exists for reads flat there by construction. What a simulator run *can* show is
that the machinery works, and it showed two things worth having and one worth
fixing.

The archived run is `src/nocbench-blackhole-sim.csv` (13 runs, 16 flows) with
its analysis in `src/nocbench-blackhole-sim.report.txt`; the core map it was
planned against is `src/nocbench-grid-blackhole-sim.csv`. Reproduce with

```bash
TT_METAL_HOME=... TT_SIM_TENSIX_COORDS="$(grep -o 'tt_sim_tensix_coords=[^ ]*' plan.csv | cut -d= -f2)" \
    TT_SIM_COST_MODEL=1 ../../run.sh nocbench -- --plan plan.csv
```

**1. The hop line comes out right, end to end.** Against tt-sim with the cost
model on, the three round-trip levels give

```
cycles = 188.3 + 8.95 * round_trip_hops     (r2 1.00)
```

8.95 against the ISA docs' 9-cycle router-to-router hop, recovered through a
real tt-metal program, a real kernel, the wire bridge and the whole simulated
NoC — not by asking the cost table what it thinks. The three levels are exactly
17 / 12 / 29, the directional-torus prediction for Blackhole, and the flatness
check inside each family reads slope 0.00 as it should.

**2. The controls caught a real bug in this harness.** The first simulator run
reported timed-region overlaps of 0.02–0.05: the flows were not concurrent. The
kernel had been passed a semaphore *id* and used it as an *address*, so
`noc_semaphore_wait_min` read L1 offset 0, found something non-zero and never
blocked. Every congestion number from that run would have been meaningless and
would have looked completely normal. The overlap check is the only reason it
was noticed; it now reads 0.98.

**3. tt-sim's one contention mechanism does not bite, and the run is therefore
`INVALID`.** The `selfport` control puts two flows on one core's two
data-movement RISCs, both on NoC 0, both 4 × 8 KiB. They start on the same
cycle and overlap fully (1.00), and each takes the same time it takes alone —
868 cycles solo, 868 and 798 together, a ratio of 0.96 where serialisation
predicts about 2. tt-sim's `NUI._tx_free_cycle` is per tile per NoC and is
meant to hold the injection link for one flit per cycle, so two cores sharing
it should queue; on this path they do not. That is a gap in the simulator's
contention model rather than in the harness, and the harness's response is the
right one: it refuses to report the flat shared-link reading as "no congestion
effect", because on a simulator whose only contention mechanism is inert a flat
reading proves nothing.

So the state of things: the plumbing, the geometry, the rendezvous, the
coordinate checks and the hop model are all verified. **The congestion
coefficient is not measured and cannot be measured here.** It needs a card.

One incidental finding for anyone porting this: tt-sim answers 0 for
`NOC_CFG(NOC_ID_LOGICAL)`, which is what tt-metal's firmware fills `my_x[]` /
`my_y[]` from, so those read (0, 0) on the simulator. The kernel therefore reads
`NOC_NODE_ID` instead, and an all-zero self-report is treated as "this device
does not answer that register" rather than as "every kernel ran on core (0, 0)".
