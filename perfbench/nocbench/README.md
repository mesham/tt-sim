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
2. **`readport` must rise.** A second master reading from the *same*
   subordinate must cost more than one, because both response streams leave
   that subordinate's single NIU and queue on its one injection port. This is
   what proves the two-flow machinery — the rendezvous, the concurrency —
   actually makes flows contend.
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

`CONGESTION MEASURED` with shape `SATURATING` is the answer the card gave, and
it is worth reading carefully: a saturating shape means the fitted slope is
**not** a coefficient. It is a step at the first shared link with a regression
drawn through it. The report prints the per-shared-link means so the step is
visible; do not lift the slope out of it.

Against the **simulator** a flat shared-link reading is forced: tt-sim charges
an NIU for its own injection port and nothing whatever for a router-to-router
link, so two flows sharing links cannot interact. Running it there exercises
everything except the effect being looked for, and the `readport` control does
pass there, which is what makes that flat reading a null rather than a dead
harness.

### Why the control is a *read*, and why the previous one was withdrawn

The control used to be `selfport`: two flows out of one core's two
data-movement RISCs, contending for that core's injection port. **It measured
nothing, and that is not a judgement call.** `noc_async_write_barrier` is

```c
return (NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]);
```

— a per-NIU *hardware* counter against a per-RISC *software* one, seeded from
it at kernel init. With two issuers on one NIU each RISC waits for the shared
counter to advance by *its own* N, so **any** N acks release it and both
kernels stop when half the traffic has landed. The ratio it prints is therefore
about 1.0 whether the port serialises perfectly or does not exist. That was
shown rather than argued: deleting `claim_injection_port` from tt-sim outright
moved the control by 0.04, in the wrong direction, while moving the absolute
cost 35 %. On the card it read exactly **1.00** — 17234 cycles alone, 17151 and
17221 together — which is precisely what "the port serialises perfectly and
each kernel stops at the halfway point" predicts, and also precisely what "the
port does nothing" predicts.

The configuration is also one a card should not be given: `BRISC_WR_CMD_BUF`
and `NCRISC_WR_CMD_BUF` are *both* command buffer 0, so two kernels issuing on
one NoC race on one set of `NOC_TARG_ADDR` / `NOC_PACKET_TAG` registers, and an
`==` against a monotonic counter that two issuers are advancing can be
overshot between polls. `noc_congestion_plan.check_invariants` now refuses any
point with two flows on one master core, and names that reason.

`readport` reaches the same shared resource by the route tt-metal actually
supports. A read's payload rides the *return* leg, so two masters reading from
one subordinate put both response streams through that subordinate's single
NIU — with one DM RISC per core, so every barrier counts only its own core's
responses and the rendezvous, the overlap check and the timed region all mean
what they say. It is sized so the port rather than the fixed round trip
dominates: the planner refuses a point carrying under 64 KiB per flow, which is
exactly how the old control came to be mis-sized at 4 x 8 KiB.

**It is not blind, and that was tested the same way the old one was convicted.**
Against tt-sim with `claim_injection_port` intact the control reads **1.48x**
and PASSes; with that function stubbed to return 0 and never advance
`_tx_free_cycle` — the mechanism under test deleted outright — it reads
**1.00x and FAILs**. A control that cannot detect the deletion of the effect it
tests is not a control; this one can.

Two honesty notes on that number. The prediction in
`HYPOTHESES["readport"]`, written before the run, was "roughly doubles"; 1.48x
is what the minimum permitted size gives, because at 8 x 8192 B the fixed round
trip is still ~25 % of the timed region and only the rest is port. Taking the
round trip out, the port-attributable part is 1.63x. At the hardware default
(64 x 8192 B) the round trip is under a tenth of the region and the ratio should
approach 2. And this is a **simulator** result: it establishes that the control
can see the mechanism, which is what a control has to establish. Whether the
mechanism is there on silicon is what running it on a card answers, and it has
not been run on one.

What `readport` does *not* do is attribute the rise to one mechanism: two
response streams leaving one tile share the first router-to-router link out of
it as well as the port, and on silicon those are one reading. That is fine for
a positive control, whose only job is to show the flows contend at all;
attribution is experiment 2's job.

## What the upstream suite cannot express

tt-metal's own `tests/tt_metal/tt_metal/data_movement/` suite is the right
*shape* for this work — it exposes coordinates, unlike the shipped YAML, and it
writes per-core profiler CSVs. It is not usable as it stands, and the reason is
worth stating precisely rather than worked around. (One correction to an earlier
version of this section, which listed `core_bidirectional` (140-148) as a usable
virtual-channel sweep: it is not. Tests 140-145 and 148 — the whole
directed-ideal family, same-kernel and different-kernel, write-VC sweep included
— begin with `GTEST_SKIP() << "Skipping test"; // Timeout issue (#36428)`.
Only the small `packet_sizes` variants still run. Nobody upstream is currently
running a bidirectional VC sweep either.)

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
# 1. what cores does this card have, what coordinate does a kernel use, and
#    where is each of them PHYSICALLY? (a probe kernel on the first logical row
#    and column reads NOC_NODE_ID; the host API cannot answer that one)
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

Useful flags: `--experiments hops,size,readport,shared` (the minimum),
`--num-tx N` (transactions per flow; more is slower and quieter),
`--tx-bytes N`, `--noc 1`.

Step 2 will **refuse** rather than emit a plan whose experiments are confounded,
and it will refuse a dump it cannot map onto physical NoC coordinates. That
second refusal is not hypothetical: the first card this ran on was harvested,
its dump looked physical, and it was not — see item 4 under "What has been
measured on a card". A dump that is short of the full unharvested worker grid
and carries no `phys_x` / `phys_y` column is now rejected rather than guessed
at, which is why step 1 launches a kernel. A refusal here is the harness
working: a shared-link count computed for the wrong geometry is
indistinguishable from a measurement.

Step 2 also refuses, unconditionally, any point with **two flows on one master
core** or any **bidirectional** flow. Both are hardware refusals rather than
statistical ones: the first races two kernels on one command buffer, the second
hung a card. They are explained where they are enforced, in
`noc_congestion_plan._refuse_unrunnable`.

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
the card runs a billion, so use `--num-tx 8 --max-points 2 --tx-bytes 512` and a
couple of experiments rather than the full plan. `readport` is the exception
that cannot be shrunk below 64 KiB per flow, because below that the fixed round
trip rather than the shared port dominates its timed region and it stops being
a control; `--num-tx 8 --readport-bytes 8192` is the floor.

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

## What has been measured on a card

Two campaigns on one **Blackhole** part, a day apart. The second is the one to
read; the first is kept below because three of its four problems are what the
harness's current refusals exist for.

### Campaign 2, on the corrected geometry — the one that certified

Two runs, **zero coordinate mismatches and zero invariant complaints in both**,
`RESULT: CONGESTION MEASURED`. Banked in
[`docs/bh_arch.md`](../../docs/bh_arch.md) §4, with the datasets in
`tt_sim/perf/datasets/` and the running record in
[`docs/plans/cost-model.md`](../../docs/plans/cost-model.md). Reproduce the
whole analysis with no hardware:

```bash
python3 -m tt_sim.perf.noc_congestion_sweep                      # the main run
python3 -m tt_sim.perf.noc_congestion_sweep \
    --measured tt_sim/perf/datasets/nocbench-blackhole-sizes.csv # the size sweep
```

**The step is one transaction's link occupancy, and the size sweep is what
shows it.** Two flows sharing one router-to-router link, six transaction sizes:

| transaction | occupancy = bytes ÷ 64 | 0 shared | 1 shared | delta | delta ÷ occupancy |
| --- | --- | --- | --- | --- | --- |
| 64 B | 1 | 39.9 | 39.8 | −0.1 | — |
| 512 B | 8 | 39.8 | 39.8 | 0.0 | — |
| 2048 B | 32 | 40.2 | 68.3 | 28.1 | 0.88 |
| 4096 B | 64 | 72.2 | 135.4 | 63.1 | 0.99 |
| 8192 B | 128 | 137.8 | 262.3 | 124.5 | 0.97 |
| 16384 B | 256 | 268.4 | 518.8 | 250.4 | 0.98 |

Flat from 2 through 7 shared links at every size. The regime boundary is
between 512 B and 2048 B — where occupancy reaches the ~40-cycle issue loop —
and the ratio then climbs 1.70 → 1.87 → 1.90 → 1.93 toward perfect halving as
the fixed cost amortises. **This is a separate run from the main one**; the two
share two of twelve cells and agree to 0.8 cycles/tx.

**`readport` reads 1.92× and PASSES on silicon**, which is the redesigned
control (below) working on hardware where the one it replaced read 1.00 and was
blind.

**The `vc` experiment ran, and different channels do not help.** Two writers on
one shared link at 16 KiB, flow A pinned at VC 1, flow B swept: 520.2 / 530.4 /
519.9 / 519.9. All four are ~1.94× the uncontended 268.4, so the split is
**link occupancy and not VC arbitration** — sharing a channel costs a further
2.0 % on top, which is real (three controls agreeing to 0.3 cycles against a
0.5-cycle floor) and small.

**One tile's wall clock keeps its own epoch**, +1,143,914,613 cycles,
reproducing across eight runs to a spread of 7. It is why two runs first read
"flows barely coincided"; `clock_skew_report` now detects and subtracts it, and
`docs/bh_arch.md` §4.4 is the entry. Same-core durations — everything any
coefficient is fitted to — were never affected.

### Campaign 1, superseded

79 flows over 55 runs, median timed-region overlap 0.99, zero invariant
complaints, the `size` control passing. Two results and two problems came out
of it, and its geometry was wrong for eight rows — see item 4.

**1. The uncongested line, on silicon.**

```
cycles = 4373.7 + 8.38 * round_trip_hops     (r2 1.00)
```

8.38 cycles per round-trip hop against the ISA docs' ~9 cycles router to
router, over the three predicted levels (12 / 17 / 29 on Blackhole) and with
r2 1.00. The flatness check inside each family holds: the row family, predicted
constant, spans 0.6 cycles/tx, and that spread is the noise floor every "flat"
claim below is read against.

**2. Congestion is a step at the first shared link, not a per-link slope.**
At a saturating transaction size, two flows whose payloads share **one**
router-to-router link each cost close to twice what they cost sharing none, and
sharing more links after that changes almost nothing:

| shared links | 64 B | 16384 B |
| --- | --- | --- |
| 0 | 39.9 | **270.1** |
| 1 | 39.8 | **517.9** |
| 2 | 39.8 | 519.1 |
| 3 | 39.8 | 517.2 |

At 64 B the whole series is flat — slope -0.00, span 0.1 cycles against a
0.6-cycle noise floor — which is the negative control the design predicted: a
64-byte packet occupies a link for one cycle against a ~40-cycle issue loop, so
the links are ~2 % busy and no sharing effect can show. At 16 KiB it is
**bandwidth splitting**: 16384 B / 64 B per cycle is 256 cycles of link
occupancy, one flow costs 270 cycles/tx, and two flows sharing one link cost
518 — two occupancies plus the same 6-14 cycles of overhead. This is not a
latency adder per hop; it is one link's bandwidth divided by the number of
saturating flows crossing it. **The harness was designed to measure a slope and
the answer is that there is no slope to measure.**

The `contention` curve says the same thing at an endpoint. Six flows of 4096 B
into one subordinate come back at 137, 203, 268, 333, 399, 399 cycles/tx —
evenly spaced by ~65.5, and 4096 / 64 = 64. N flows into one endpoint serialise
almost exactly at one transaction's link occupancy each, in a stable rank
order.

**3. The `selfport` control was not a measurement, and has been replaced.**
See "Why the control is a *read*" above. Its silicon reading of 1.00 is
consistent with perfect serialisation and with no serialisation at once. The
run's verdict is `INVALID` because of it, and the verdict is correct while the
cause given for it was not.

**4. The card was harvested, and the plan was built on the wrong geometry.**
This is the one that would have been missed. The card dumps addressed worker
columns `{1..7, 10..14}`, which is a legal *subset* of an unharvested
Blackhole's physical columns `{1..7, 10..16}` — so the planner concluded the
dump was already physical. The kernels' own `NOC_NODE_ID` says otherwise: the
true physical columns are `{1..7, 12..16}`, and addressed x >= 10 sits two
columns to the right of where the plan put it. Re-deriving the geometry from
the self-reports:

* the eight `shared` points are at **0, 1, 2, 3, 6, 7, 8, 9** shared links, not
  0 through 7;
* flow B's forward hop count, which the experiment declares fixed, is 8 at the
  first four points and 12 at the last four;
* every other invariant survives — `other_overlap` is 0 at all eight points on
  the corrected geometry too, flow A is identical throughout, and every
  round-trip count is still 29.

So the **step at the first shared link stands**, measured over the four points
(0, 1, 2, 3) where flow B's hop count really is constant. The 6-9 group is a
different experiment and its ~11-cycle offset is confounded with those four
extra hops. The executor caught 8 of the 79 rows as coordinate mismatches, which
is why any of this is knowable; `--dump-grid` now runs a probe kernel on the
first logical row and column and writes each core's own physical coordinate, and
`noc_congestion_plan` refuses a dump that is short of the full worker grid and
carries no such column.

**5. The `vc` experiment hung the card, and is now unidirectional.** The first
and only `direction=BIDIR` point never returned; all 79 unidirectional flows in
the same session completed. It is not the virtual channel: every one of those 79
flows issued its writes on VC 0. It is not the kernel: tt-sim runs the identical
binary and the identical plan (64 x 4096 B, bidirectional, VC 0-3) to completion
in 4958 cycles. tt-metal's own `core_bidirectional` suite disables its entire
directed-ideal family — same-kernel and different-kernel, write-VC sweep
included — with `GTEST_SKIP() << "Skipping test"; // Timeout issue (#36428)`.
The cause is not established and cannot be established without the card, so
`check_invariants` **refuses `DIR_BIDIR` outright**: no plan this repository
emits can contain one.

The virtual-channel question survives without it, and in a better form: two
writers whose payloads share exactly one link, the first pinned on tt-metal's
own `NOC_UNICAST_WRITE_VC` (1) and the second swept 0-3. The `vc1` point is
"both writers on one channel" and the other three are its control, so the
reading answers the question the banked result actually raises — *is the
halving at one shared link virtual-channel arbitration, or is it occupancy?* If
it is arbitration, `vc1` costs about twice the others; if it is occupancy, all
four read alike and equal to the shared-link-1 point of experiment 2.

## What the simulator run shows

tt-sim models no router-to-router congestion, so experiment 2 is forced flat
there; the point of running it is that everything else is exercised. The
archived run is `src/nocbench-blackhole-sim.csv` with its analysis in
`src/nocbench-blackhole-sim.report.txt`, planned against
`src/nocbench-grid-blackhole-sim.csv`. Reproduce with

```bash
TT_METAL_HOME=... TT_SIM_TENSIX_COORDS="$(grep -o 'tt_sim_tensix_coords=[^ ]*' plan.csv | cut -d= -f2)" \
    TT_SIM_COST_MODEL=1 ../../run.sh nocbench -- --plan plan.csv
```

* The hop line comes out end to end: the three round-trip levels give
  `cycles = a + ~9 * round_trip_hops` at r2 1.00, recovered through a real
  tt-metal program, a real kernel, the wire bridge and the whole simulated NoC
  rather than by asking the cost table what it thinks.
* The concurrency check caught a real bug in this harness. The first simulator
  run reported timed-region overlaps of 0.02-0.05: the kernel had been passed a
  semaphore *id* and used it as an *address*, so `noc_semaphore_wait_min` read
  L1 offset 0, found something non-zero and never blocked. Every congestion
  number from that run would have been meaningless and would have looked
  entirely normal.
* The `readport` control passes (1.48x) and fails when the mechanism it tests
  is deleted (1.00x). That is the ablation described above, and it is what
  makes a flat shared-link reading on the simulator a null rather than a
  non-reading.

One incidental finding for anyone porting this: tt-sim answers 0 for
`NOC_CFG(NOC_ID_LOGICAL)`, which is what tt-metal's firmware fills `my_x[]` /
`my_y[]` from, so those read (0, 0) on the simulator. The kernel therefore reads
`NOC_NODE_ID` instead, and an all-zero self-report is treated as "this device
does not answer that register" rather than as "every kernel ran on core (0, 0)".
