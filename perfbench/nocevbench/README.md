# nocevbench — checking tt-sim's NoC timing against a card's, by mechanism

Rung 4's **NoC-bound leg**, and the sibling of
[`perfbench/mechbench`](../mechbench/README.md). Where that leg looks inside a
Tensix core's span with the hardware stall counters, this one looks inside a
data-movement core's span with tt-metal's own **NoC event trace**.

It is the leg that would retire the bottleneck report's **unverified 79.8 % NoC
split on nekbone**. That figure is unverified today and nothing here has yet
checked it; do not quote it as measured.

It fits nothing. **A disagreement is the result**, not a bug to be closed by
tuning the simulator.

One card session exists (Blackhole p150, 2026-08-17), and it produced exactly
one disagreement. [The experimental arms](#the-experimental-arms-and-the-confound-they-break)
are what a second session would run to say whether that disagreement is about a
*direction* or about a *NoC*, which the first one structurally cannot.

## The headline: tt-sim already emits the artefact

The pivotal question for this leg was whether tt-sim could produce the artefact
at all, and the answer, verified end to end on 2026-08-16, is **yes, with no
bridge work**:

```
TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 ./build/nocevbench 4096 8
  -> .logs/noc_trace_dev0_ID0.json      54 records, both RISCs, both NoCs
  -> .logs/topology.json, cluster_coordinates.json
```

on **both** Blackhole and Wormhole. Nothing in `tt_sim/bridge/` needed changing:
the profiler readback path fixed on 2026-08-13 (`Device.settle_profiler_flush`)
already covers it, because NoC events are written into the *same* per-RISC L1
profiler vector as the zone markers — there is no separate buffer and no
separate readback.

That is worth stating loudly because it was not the expected answer. Every
readback hazard this project has been bitten by was already paid for.

## What the artefact actually is

Established by reading tt-metal 0.74, not by inference. Citations are
`file:line` into `$TT_METAL_HOME`.

| question | answer |
| --- | --- |
| what the env var gates | `rtoptions.cpp:852` — sets `profiler_noc_events_enabled` **and** force-enables the device profiler, so `TT_METAL_DEVICE_PROFILER=1` is *not* additionally required |
| how it reaches the kernel | `jit_build/build.cpp:188` injects `-DPROFILE_NOC_EVENTS=1` (and `-DPROFILE_KERNEL=1` if needed) into every kernel compile. It is part of the JIT build hash, so flipping it forces a rebuild |
| what records the events | `tt_metal/tools/profiler/noc_event_profiler.hpp`, compiled **only** for BRISC and NCRISC (`:7`); `brisc.cc:10` `#undef`s it so *firmware* traffic is excluded |
| the record | `KernelProfilerNocEventMetadata`, `event_metadata.hpp:13`, exactly 8 bytes: event type, `dst_x/dst_y`, multicast end coords, NoC id (4 bits), VC (signed 4 bits), `payload_chunks` (bytes/32, **saturating at 255 → 8160 B**), `posted` |
| where it lives on device | the ordinary per-RISC profiler L1 vector (`kernel_profiler.hpp:79-84`), 2048 B per RISC, 250 optional markers — **the same buffer the zone profiler uses** |
| what a timestamp means | `RISCV_DEBUG_REG_WALL_CLOCK`, read **at issue**, before the command buffer is written (`kernel_profiler.hpp:178`; call sites `dataflow_api.h:518` for `noc_async_read`, `:835` for `noc_async_write`, ~40 more) |
| the only completion timing | `READ_BARRIER_START`/`END` (`dataflow_api.h:1738`, `:1751`) and the write pair (`:1768`, `:1781`), bracketing the `*_flushed` spin |
| the host artefact | `.logs/noc_trace_dev<d>[_<op>]_ID<n>.json` — one JSON array per (device, program execution), written by `dumpJsonNocTraces` (`profiler.cpp:1090`) |
| its fields | `run_host_id, op_name, proc, noc, vc, src_device_id, sx, sy, num_bytes, type, timestamp`, plus `dx, dy` for a unicast destination and the multicast rectangle for `WRITE_MULTICAST` (`profiler.cpp:840-887`). Zone endpoints are interleaved into the same array with `zone` / `zone_phase` (`:825`) |
| what it does **not** carry | source and destination **addresses**. They need `NON_DROPPING`, which needs `TT_METAL_NOC_DEBUG_DUMP`, and in that mode `writeDeviceResultsToFiles` returns early (`profiler.cpp:2530`) and writes no JSON at all. The addresses are unreachable in the file-dumping path |
| the hard prerequisite | a **Tracy-enabled** tt-metal build. Every dump path is inside `#if defined(TRACY_ENABLE)`, and unlike `TT_METAL_DEVICE_PROFILER=1` — which `TT_FATAL`s without it (`rtoptions.cpp:815`) — this flag has no such guard and silently produces nothing |
| arch gating | none of the feature. Host-side coordinate translation does differ (`profiler.cpp:512-545`) but tt-metal applies it before the JSON is written, so the harness never sees raw encodings |
| cost | tt-metal warns of **1–15 % cycle overhead** (`profiler.cpp:2546`). It is paid on both sides, which is why the flag must be identical on both |

### The one fact that reshapes the leg

**Every transaction event is stamped at issue. There is no per-transaction
completion timestamp anywhere in the artefact.**

The roadmap's phrasing for this leg — "agreement with `noc_flight_cycles` plus
queueing to ± 25 %" — presupposes a hardware per-packet flight time. There isn't
one. So the correspondence is not forced, and two things are done instead:

1. **The comparison is between two instances of the same artefact.** tt-sim
   produces `noc_trace_dev*.json` exactly as a card does, so one parser reads
   both and there is no translation step in which a units mistake could hide.
   That is the same discipline as `mechbench`, for the same reason.
2. **`noc_flight_cycles` is reported on the simulator side only, as a
   diagnostic, and never as a gate** (`--sim-internal`). It is also **not** a
   quantity to which queueing should be added: tt-sim's `noc_flight_cycles` is
   `arrival - issue_cycle`, and `issue_cycle` is stamped inside `NUI.transmit`
   *after* `NUI.send_to` has computed a delay of
   `flight + injection-port queueing + link-contention wait + serialisation - 1`
   (`NUI._bandwidth_delay`). **The queueing is already inside the number.**
   Adding it again would double-count.

   One caveat on that diagnostic, stated because it would otherwise mislead: the
   `TT_SIM_TRACE_NOC` dataset counts **all** traffic through a NIU, including
   firmware and the profiler's own DRAM pushes, while the JSON is kernel-only.
   The two event counts are not meant to match.

## The partition

Per `(core, RISC)`, the kernel zone is cut at every recorded event and each
interval is attributed to **the event that opened it**:

| bucket | the interval from … |
| --- | --- |
| `prologue` | `ZONE_START` to the first NoC event |
| `issue` | a transaction-issuing event to the next event |
| `read_wait` | `READ_BARRIER_START` to `READ_BARRIER_END` |
| `write_wait` | `WRITE_BARRIER_START` to `WRITE_BARRIER_END` |
| `other_wait` | a single-event blocking call (`SEMAPHORE_WAIT`, `FULL_BARRIER`, `WRITE_FLUSH`, …) to the next event |
| `local` | a barrier END to the next event — the core's own work |

Three decisions in that table are load-bearing.

**The buckets telescope to `ZONE_END − ZONE_START` by construction**, and
nothing is filtered to make that true. That is what keeps both sides'
denominators the same quantity and keeps the `E_total ≤ E_int` inequality — and
therefore the compensation ratio — meaning something. It also makes
`partition_closes` a live test rather than a formality: the wall clock is stored
in 44 bits and read as two separate register accesses
(`kernel_profiler.hpp:178-183`), so a torn read or a wrap is physically
possible, and a zone endpoint out of place makes the sum miss the span.

**`other_wait` is separate from `issue`.** The blocking calls that record only a
START (`SEMAPHORE_WAIT` and friends) would otherwise put a potentially enormous
wait into a bucket named for issue work.

**`*_SET_STATE` and `*_SET_TRID` are in `issue` but are never latency-paired.**
They configure a command buffer and put no packet on the wire; pairing them with
a barrier would invent a transaction and drag the mean latency down.

## The criterion

```
E_total = |Σ c_sim − Σ c_hw| / Σ c_hw
E_int   = Σ |c_m,sim − c_m,hw| / Σ c_hw
```

Pass requires **both ≤ 25 %**, plus every transaction class's mean observed
latency within 25 %. One number, because the roadmap names one number for this
leg; a tighter envelope bar here would be a bar with no provenance. The
compensation `E_int / E_total` is printed on every comparison.

**Every criterion is per core and per RISC.** On the 2026-08-10 part a whole
physical column keeps a wall-clock epoch 1.5e13 cycles from the rest of the die,
so a cross-core span is not a coarser measurement but a meaningless one.

### What "observed latency" is, and is not

Per class `(noc, type, bytes)`: the mean of `BARRIER_END − issue_timestamp` over
the transactions that barrier covered.

**It is not a per-packet flight time and must not be quoted as one.** It is an
upper bound that includes the residual of every other transaction in the same
batch, the NIU's own issue overhead and the RISC's poll granularity. It is the
same upper bound on both sides, computed the same way, which is what makes it
comparable — and nothing more than that.

### The two detectors are not independent, and that is stated rather than hidden

`read_wait` / `write_wait` and the per-class latencies are **two readings of the
same clock intervals**. The genuinely independent part of the partition is
`prologue`, `issue`, `other_wait` and `local`. A leg that claimed two
independent checks here would be over-claiming, so it does not.

## The program

`src/nocevbench.cpp` — a normal tt-metal program with **no compute kernel at
all**. One core, two data-movement kernels, a DRAM round trip, and a host-side
check that is **exact**: every word of the pattern is distinct, and the bytes
that come back must be the bytes that went out, compared word for word.

* NCRISC pulls `chunks` chunks of `bytes` from DRAM on **NoC 1**, barriering
  after each one, then sets a semaphore.
* BRISC waits on the semaphore, then pushes the same bytes back to DRAM on
  **NoC 0**, barriering after each one.

That is arm A, the default and the control. `--arm B` swaps the two NoCs and
`--arm C` replaces DRAM with a peer core's L1; both are below.

**One barrier per chunk is the point.** A kernel that issues a hundred reads and
barriers once yields exactly one completion timestamp and one enormous latency
sample; barriering per chunk gives one clean issue→completion pair per
transaction.

**The two phases are serialised by a semaphore, not pipelined.** Pipelining
would make each RISC's span mostly a wait on the other one, and a stream whose
interior is dominated by the peer measures the peer, not the NoC.

**The size arms are a pair, and the pair is the point.** A per-hop latency that
is too large can be hidden by a per-byte rate that is too fast at *one* transfer
size; it cannot be hidden at two sizes an order of magnitude apart, because the
two terms scale differently. Each run also measures two different routes at once
(NCRISC/NoC 1 read, BRISC/NoC 0 write).

---

## The experimental arms, and the confound they break

The first Blackhole p150 session (2026-08-17,
`perfbench/card-sessions/2026-08-17-nocevbench/`) passed every instrument gate on
all six runs. Exactly one quantity failed: the DRAM **write** class at 256 B, at
25.6 % against the 25.0 % bar, while the read class agreed with silicon at
3–6.5 %. `unit_costs.yaml` already documents the suspect — `dram.access_latency`
is derived from *read* rows and charged to `{READ, WRITE, ATOMIC}` alike — and
`tt_sim/perf/noc_dataset_sweep_test.py` already pins it as its single
`KNOWN_OVER_CHARGED` row.

**That session cannot convict it, because direction and NoC are fully
confounded.** The reader runs on NoC 1 and the writer on NoC 0, so "writes cost
+95 cycles" and "NoC 0 costs +95 cycles" are *the same statement*. `--arm` is
the axis that separates them, and until it has been run nothing else can be
concluded.

| arm | reader | writer | target | what it buys |
| --- | --- | --- | --- | --- |
| **A** | NCRISC / NoC 1 | BRISC / NoC 0 | DRAM | **the control.** Byte-for-byte the 2026-08-17 configuration; it must reproduce it, or nothing else in a session is comparable to that one |
| **B** | NCRISC / **NoC 0** | BRISC / **NoC 1** | DRAM | **the discriminator.** Two enum values in the host program; no kernel change, no second core, no new address |
| **C** | NCRISC / NoC 1 | BRISC / NoC 0 | a **peer core's L1** | removes the DRAM endpoint service term from the path entirely, which is what makes an attribution *to the endpoint* rather than to the route |

Arm A's kernels (`reader.cpp`, `writer.cpp`) are **untouched** by B and C: B
changes only the `.noc` field of two `DataMovementConfig`s
(`kernel_types.hpp:33-38`), and C uses its own pair (`l1_reader.cpp`,
`l1_writer.cpp`) whose only difference is `get_noc_addr(peer_x, peer_y, addr)`
in place of `get_noc_addr_from_bank_id<true>(0, addr)`. A control whose kernel
grew a branch is no longer the control.

Arm C's peer runs **no kernel at all**. Its two regions are one interleaved L1
buffer — an address the allocator guarantees free on every bank — seeded and
read back by the host with `WriteToDeviceL1`/`ReadFromDeviceL1`, the same trick
`perfbench/nocbench` uses. So the check stays exact (the destination is poisoned
with the complement of the pattern before launch, so a run that moved nothing
cannot pass) and nothing on the peer side can contend with the measured
transactions.

### The pre-registered predictions

**Written before any of these runs existed, and reproduced verbatim.** Reading
an arm against a prediction adjusted after the fact would destroy the only thing
the arms are for.

| if the true cause is … | arm B predicts | arm C predicts |
| --- | --- | --- |
| **DRAM write service** (the hypothesis) | the +~95 **follows the write**, now on NoC 1; the read, now on NoC 0, stays within ~5 % | **both directions within ~10 %**; no +95 anywhere |
| a NoC 0 error | the +~95 **follows NoC 0**, i.e. now on the *read* | the +~95 persists on NoC 0 |
| a per-hop error | +~95 stays on the write (29 hops either way) | +~95 persists |
| a response-path error | the +~95 follows the direction | persists |
| profiler overhead | nothing systematic survives | nothing |

and the absolute prediction for arm C, from the model as it stands (no DRAM
term, 29-hop round trip, ~65-cycle core residual): **~349 cycles at 256 B and
~409 at 4096 B, both directions.**

### What tt-sim says, before any card time is spent

Blackhole, cost model on, `chunks 8`, worker `(1,2)`, arm C's peer logical
`(1,1)` = NoC `(2,3)`. Observed issue-to-barrier-END mean, per class:

| arm | class | 256 B | 4096 B |
| --- | --- | --- | --- |
| A | `NOC_1 READ` (DRAM) | 475 | 563 |
| A | `NOC_0 WRITE_` (DRAM) | 476 | 540 |
| B | `NOC_0 READ` (DRAM) | **475** | **563** |
| B | `NOC_1 WRITE_` (DRAM) | **476** | **540** |
| C | `NOC_1 READ` (peer L1) | **351** | **415** |
| C | `NOC_0 WRITE_` (peer L1) | **351** | **415** |

Two things fall out, and both matter before the card runs.

**tt-sim's arm B is numerically identical to its arm A, class for class.** The
swap changes the model's internals completely — the read's request leg goes from
156 to 363 cycles at 256 B and the write's from 365 to 158, because both the
NoC's direction and the cell of DRAM channel 0 it addresses change — and the
observed totals do not move by one cycle. That is the directional-torus identity: a round
trip between two tiles differing on both axes costs `grid_x + grid_y` hops
whichever way round it goes. So the simulator makes **no prediction that
distinguishes arm B from arm A**, and the card's arm B is therefore a clean
discriminator against an unchanged simulator baseline: whichever of the write or
NoC 0 the ~95-cycle error follows, it follows it against the same two numbers.

**Arm C lands on its pre-registered absolute prediction**: 351 against ~349 at
256 B, and 415 against ~409 at 4096 B — 2 and 6 cycles out, on numbers written
down before the arm was built. Removing the DRAM endpoint takes **125** cycles
off the write at both sizes (476 → 351, 540 → 415) and 124 / 148 off the read —
that is the 126-cycle `dram.access_latency`, plus the 23-cycle channel excess
the read alone pays at 4096 B, to within a cycle. The term arm C removes is
exactly the term it was designed to remove, which is what makes it a
confirmation of the diagnosis rather than a second guess at it.

### One thing the destination coordinates say, recorded and not acted on

Building `check_arm.py` meant reading the `dx`/`dy` fields for the first time,
and they do **not** agree between the two sides on the NoC 1 leg. tt-metal
writes them through the same host-side `translateNocCoordinatesToNoc0`
(`profiler.cpp:840-887`) in both cases, so they are directly comparable, and for
the same arm-A program at 256 B:

| leg | tt-sim | 2026-08-17 card |
| --- | --- | --- |
| `NOC_0 WRITE_` destination | `(0, 11)` | `(0, 11)` |
| `NOC_1 READ` destination | `(16, 10)` | `(0, 1)` |

`(16, 10)` is the exact grid mirror of `(0, 1)` on a 17×12 die
(`17-1-0, 12-1-1`), which is the signature of one side applying the NoC-1 mirror
where the other does not. The two sides' kernels are therefore putting
*different bits* into the NoC 1 command buffer for the same source-level
`noc_async_read` — the write leg, which never mirrors, agrees exactly.

**The round trip is 29 hops either way**, so no modelled latency moves, which is
consistent with the read class agreeing with silicon at 3–6.5 %. But the split
does move: on the coordinates each side recorded, tt-sim's NoC 1 read is 2 hops
out and 27 back, and the card's is **27 out and 2 back**. Recorded here because
it is the only *falsifiable* thing the destination fields say, and because it
bears directly on the one route this leg has ever checked against hardware.
Nothing in this leg's timing depends on it, one observation is not a licence to
change a routing table, and chasing it is a separate piece of work.

It is also why `check_arm.py` cannot separate arm A from arm C by comparing a
DRAM destination between the two sides — it separates them by whether one core
is addressed in both directions, which is convention-independent.

### Wormhole: arm C needs translation, and the peer check needed fixing

Established 2026-08-17 by running the simulator side at `--arch wormhole`, and
it would otherwise have been found at a card. `check_arm.py`'s arm-C peer check
was a plain equality between the `peer_noc` on the config line and the trace's
`dx`/`dy`. On Blackhole a worker's translated coordinate **is** its NOC 0
coordinate, so the two agree and the equality is right. On Wormhole they
disagree, in two opposite directions:

| the run | trace destination | config `peer_noc` | what the old check said |
| --- | --- | --- | --- |
| translated (what a **card** does) | `(2, 2)` | `(19, 19)` | FAIL — and the run was correct |
| untranslated (tt-sim's default) | `(2, 2)` and `(7, 9)` | `(2, 2)` | FAIL, naming no cause |

The first is a **coordinate-space** difference and nothing else:
`worker_core_from_logical_core` returns the *translated* coord while
tt-metal's host writes every trace coordinate in NOC 0 space. Every real
Wormhole card runs translated, so this would have refused the session's own
arm C.

The second is a **real incomparability**. `(7, 9)` is the grid mirror of
`(2, 2)` on Wormhole's 10x12 grid: untranslated, the kernel puts an *unmirrored*
worker coordinate on NoC 1, which is not the convention a card is in. So **the
Wormhole simulator side of arm C must be run translated**:

```bash
TT_METAL_MOCK_CLUSTER_DESC_PATH=~/tt-sim/driver/wormhole/cluster_descriptor.yaml \
  ./run_sim.sh --arch wormhole --arm C --peer 1,1 --out ~/nocev-sim-wh-C
```

Arms A and B target DRAM, which Wormhole never translates, and are unaffected.
`check_arm.py` now tells the two cases apart: the translated run passes with
both coordinate spaces named, and the untranslated one is refused **by name**
with the fix in the message. The program prints `noc_grid=X,Y` on its config
line (from `device->grid_size()`) so the mirror can be computed without knowing
the part, and `arch=` on its `--describe` line — `run_card.sh` used to record
`arch : unset` in every session's `env.txt`. Guarded in
`tt_sim/perf/noc_events_test.py`.

**The 375-382 / 480-510 band below is BLACKHOLE's.** On a Wormhole part arm A
*establishes* the control rather than reproducing one; `run_card.sh` says so at
the card. tt-sim's Wormhole arm A reads `WRITE_ 256 = 390`, `READ 256 = 386`,
`WRITE_ 4096 = 550`, `READ 4096 = 546`.

The Wormhole programme this belongs to is
[`docs/plans/wormhole-session.md`](../../docs/plans/wormhole-session.md).

### What is still deliberately not here: the distance arms

Varying the peer's *distance* (same column, same row, and a second both-axes
peer further away) is what would bear on `noc.hops` directly, and arm C's
`--peer` already reaches it — the harness would need only more values, not more
code. It is left for a later session because none of it is needed to act on the
write term, and because the two arms above are what one card session should
spend its time on.

---

## The card protocol

Copy-pasteable, for someone who is not the person who wrote it. Nothing from
this repo is needed on the card box beyond `perfbench/nocevbench/` and a
**Tracy-enabled** tt-metal. No Tracy front end, no `tt-exalens`, no board reset,
no root.

```bash
# 0. Get the tree onto the card box. Exclude build/ -- a CMakeCache.txt records
#    the absolute path it was generated in and cmake refuses a foreign one.
rsync -av --exclude 'build/' perfbench/nocevbench/ <card-box>:~/nocevbench/

# 1. On the card box.
export TT_METAL_HOME=/path/to/your/built/tt-metal     # must export TT-Metalium
cd ~/nocevbench

# 2. Pre-flight. THIS IS THE STEP THAT COSTS ALMOST NOTHING AND SAVES THE
#    SESSION. The free checks catch a non-Tracy build, which would otherwise run
#    to completion, report PASS and write no trace at all. After them it builds
#    (not card time) and then opens and closes the device once per arm to check
#    the two things that are properties of the device and cannot be known from
#    the host: that arm C's peer core exists in the compute grid, and that it
#    really differs on both axes. Getting that second one wrong does not fail --
#    it silently measures a different geometry.
./run_card.sh --preflight --arms "A B C" --peer 1,1
echo "preflight exit=$?"          # non-zero => fix it, no measurement taken

# 3. Look at the schedule and the wall estimate before committing to it.
./run_card.sh --list --arms "A B C"

# 4. Run it. Arms interleaved, three repeats each, two sizes -- 18 runs at ~20 s
#    is ~6 minutes after the first build (~2 min). ARM A IS THE CONTROL: it is
#    the 2026-08-17 session's configuration and must reproduce it, or the box has
#    changed and nothing else in the session is comparable to that one.
./run_card.sh --arms "A B C" --bytes "256 4096" --chunks 8 --repeats 3 \
              --peer 1,1 --out ~/tt_traces/nocfixed-session
echo "exit=$?"

# 5. Check ON THE SPOT, by eye, before you stop.
cat ~/tt_traces/nocfixed-session/summary.txt
#   (a) every line says PASS, and every line says "arm=X confirmed" -- that is
#       check_arm.py having read the NoC each RISC actually used back out of the
#       trace. A run without it measured something other than what it says;
#   (b) the `lat` line under each run carries that run's observed per-class
#       latencies, computed on the card. Arm A's WRITE 256 mean should be
#       375-382 and its READ 256 mean 480-510 -- that is the 2026-08-17 session
#       reproducing. If it is not, SAY SO and send it anyway; a control that
#       moved is the most important result in the directory;
#   (c) the three repeats of an arm agree. A big spread means something else was
#       on the card. Do NOT re-run and keep the better session -- say so in the
#       handover instead;
#   (d) arm B's traces really swapped the NoCs. run_card.sh already refused any
#       run where they did not, but look anyway:
for t in ~/tt_traces/nocfixed-session/runs/B-*/.logs/noc_trace_dev0_ID0.json; do
  ./check_arm.py --arm B --trace "$t"
done
#       Expect READ on NOC_0 and WRITE_ on NOC_1 -- the opposite of arm A.

# 6. Send the WHOLE directory home. stdout logs included -- they carry the
#    nocevbench-config line that says what each run was configured as, and
#    profile_log_device.csv is free.
rsync -av ~/tt_traces/nocfixed-session/ <home>:~/nocfixed-session/
```

### What the pre-flight checks, and why each one

| check | what it catches | why before card time |
| --- | --- | --- |
| Tracy in `CMakeCache.txt` | a tt-metal built without Tracy | every dump path is inside `#if defined(TRACY_ENABLE)`, and this flag has **no** `TT_FATAL` guard: the run passes and writes nothing |
| `bytes % 64 == 0` | a transfer the NoC congruence rule makes **undefined behaviour**, not slow | the data would come back wrong or skewed with no fault raised |
| `bytes <= 8160` | the event record's `payload_chunks` saturating | the trace would carry a size the kernel did not use |
| L1 ≤ 64 KiB (192 KiB for arm C) | a buffer that will not fit | fails at device open, but noisily and after the build |
| events per RISC < 100 | the profiler's 250-word / 125-marker budget | an overflow drops events **silently**; `barriers_pair` is the only thing downstream that would notice |
| `check_arm.py` present | a rsync that dropped it | without it, an arm that did not take is indistinguishable from one that did |
| **peer inside the compute grid** (on device) | an arm C whose peer does not exist | it would fail at device open, after the build and after the session had started |
| **peer's distance class** (on device) | a peer that is not on a both-axes route | it does **not** fail — it measures a different geometry, and the geometry is the whole question |

And one check that is deliberately **not** a pre-flight, because it cannot be:
**that each RISC really got the NoC its arm asked for.** That is a property of
the emitted trace, not of the command line, so `check_arm.py` reads it back out
of every trace *after* the run and `run_card.sh` fails the run if it disagrees.
This is the failure mode the whole arm-B session hinges on: a run that silently
kept arm A's pairing is well-formed, passes every analysis gate, and would say
the ~95-cycle error "stayed on the write" — which is exactly what the *rival*
hypothesis predicts.

### The schedule, and why it is repeated

| slot | arm | bytes | chunks | runs |
| --- | --- | --- | --- | --- |
| 1 | A | 256 | 8 | 3 |
| 2 | A | 4096 | 8 | 3 |
| 3 | B | 256 | 8 | 3 |
| 4 | B | 4096 | 8 | 3 |
| 5 | C | 256 | 8 | 3 |
| 6 | C | 4096 | 8 | 3 |

Three runs per arm, each into **its own output directory**. Repeats are not
averaged and must not be concatenated: the analysis refuses a trace carrying
more than one `run_host_id` or more than one zone episode, because that is two
windows and not one. The three exist so the operator can see the instrument's
own repeatability before sending anything, and so the home side can pick a
representative run and *say which*.

Everything is interleaved (`A 256`, `A 4096`, `B 256`, …, then the next repeat)
rather than blocked. Cycle counts are far less exposed to thermal drift than a
power reading, but a blocked schedule turning drift into a fake result has
bitten this project before, interleaving is free, and it matters more here
because the arms are now compared to *each other* and not only to the simulator.

**Expected wall time**: the first build is ~2 minutes. Each run is device open
plus one launch — budget **20 s**. So the six runs of a single arm are **~2
minutes** and all three arms are **~6**. `./run_card.sh --list` recomputes it.

### What lands in the output directory

```
summary.txt           per-run PASS/FAIL, event count, barrier-END count, arm
runs/<arm>-<bytes>-<n>/   one directory per run
  stdout.log          everything the program printed, including the
                      nocevbench-config line that records the arm, the NoCs
                      and (arm C) the peer's NoC coordinate
  .logs/noc_trace_dev0_ID*.json   <- the artefact the analysis consumes
  .logs/topology.json, cluster_coordinates.json
  .logs/profile_log_device.csv    the zone log, same run, free
env.txt               the exact environment and tt-metal commit
```

Send **all** of it. `stdout.log` is what makes a suspicious trace diagnosable
later, and `profile_log_device.csv` costs nothing extra.

### Blackhole versus Wormhole

Target **Blackhole**; that is the card that exists. The instrument is not
arch-gated and both sides of the harness run on either.

What a **Wormhole** session would add that a Blackhole one cannot:

* Wormhole's flit is half Blackhole's (256 vs 512 bit), so the per-byte term is
  twice the size relative to the per-hop term. The size arms separate the two
  terms much more sharply there, and Wormhole's contention row in the v2.0 list
  is currently *prediction with nothing to check it against*.
* Wormhole's DRAM read congruence is 32 bytes against Blackhole's 64, and its
  DRAM never takes translated coordinates — a different path through the
  endpoint-aliasing code that `73ad018` changed.
* Any two-arch claim for rung 4 needs it, per the v2.0 list's item 3.

---

## The simulator side

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/nocevbench/run_sim.sh \
    --arch blackhole --out /tmp/nocevbench-sim                        # arm A
TT_METAL_HOME=/path/to/tt-metal ./perfbench/nocevbench/run_sim.sh \
    --arch blackhole --arm C --peer 1,1 --out /tmp/nocevbench-sim-C   # arm C
```

**Run every arm here before it gets card time.** Arm C addresses a second core's
L1, and an arm that has never run in the simulator is a bring-up risk paid in
card minutes; `run_sim.sh` names both workers in `TT_SIM_TENSIX_COORDS` and then
checks the peer the device reports against the one it pre-built, because a
logical-to-physical map that is wrong for an arch fails silently by measuring a
different core. Each run is checked against its arm from the trace, exactly as
on the card.

**Keep the cost model on** — `run_sim.sh` exports `TT_SIM_COST_MODEL=1` and
`--no-cost-model` exists only to demonstrate what goes wrong. With it unset
every NoC flight collapses to the one cycle the two-list swap in
`NUI.clock_tick` costs, and the comparison measures nothing.

**Pin the two sides together before collecting anything you intend to quote.**
`bytes` and `chunks` must be identical on both, and `census_matches` refuses the
pair if they are not. Unlike the sibling leg, this is cheap to satisfy: the
program is microseconds of device time either way, so the card can simply run at
the simulator's parameters.

What it produced on 2026-08-16, Blackhole, cost model on, `4096 8`:

| stream | span | `prologue` | `issue` | `read_wait` | `write_wait` | `other_wait` | `local` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NCRISC | 5 066 | 97 | 486 | **4 048** | 0 | 0 | 435 |
| BRISC | 9 830 | 33 | 488 | 0 | **3 832** | 5 046 | 431 |

and the observed latencies, against the model's own account of them
(`--sim-internal`, restricted to the kernel's own transactions by payload size —
the Parquet dataset also carries firmware and profiler traffic):

| class | observed mean | modelled request | modelled response | sum | residual |
| --- | --- | --- | --- | --- | --- |
| `NOC_1 READ 256` | 475 | 156 | 256 | 412 | **63** |
| `NOC_1 READ 4096` | 563 | 177 | 316 | 493 | **70** |
| `NOC_0 WRITE_ 256` | 476 | 364 | 46 | 410 | **66** |
| `NOC_0 WRITE_ 4096` | 540 | 424 | 46 | 470 | **70** |

The residual is **63–70 cycles across both directions and both sizes** — flat,
which is what a fixed NIU issue cost plus the RISC's poll granularity looks like
and not what a mis-scaled bandwidth term would look like. It is the part of the
core-observed latency the flight model does not claim to cover. Arms B and C
leave it exactly where it was. Taking the read class, where the attribution has
no firmware traffic mixed into it (n = 8 on the nose): arm B's residual is
**63** at 256 B and **70** at 4096 B — identical to arm A's, on a route whose
request leg went from 156 cycles to 363 — and arm C's is **67** and **71** with
no DRAM in the path at all. Four routes, one residual band. That is the
strongest evidence so far that it belongs to the core and not to the path.

Wormhole, same command with `--arch wormhole --bytes 4096`, runs identically and
produces the same residual:

| class | observed mean | modelled request | modelled response | sum | residual |
| --- | --- | --- | --- | --- | --- |
| `NOC_1 READ 4096` | 546 | 179 | 308 | 487 | **59** |
| `NOC_0 WRITE_ 4096` | 550 | 450 | 37 | 487 | **63** |

Two things a card would arbitrate, both stated as open: whether that ~60–70
cycle residual is real, and whether Wormhole's write request really costs 450
cycles against Blackhole's 424 for the same 4096 bytes — the flit-width
difference the size arms exist to separate.

**It is reported, not explained away, and it is not a number to tune anything
against.** It is one of the things a card session would put a value on for the
first time.

## The analysis

```bash
python3 -m tt_sim.perf.noc_events \
    --sim  /tmp/nocevbench-sim/4096/.logs/noc_trace_dev0_ID0.json \
    --card ~/nocfixed-session/runs/A-4096-1/.logs/noc_trace_dev0_ID0.json \
    --expect-arm A \
    --report report.txt --json report.json
```

One arm and one size at a time — each arm is a different program by census, and
the analysis refuses a trace with two windows. Add `--sim-internal
/tmp/nocevbench-sim/4096-internal` for tt-sim's own modelled per-transaction
cycles, and `--decompose-only` to print the simulator side alone.

**Read the per-class latency table against the pre-registered predictions above
before looking at anything else.**

`--expect-arm` refuses unless both traces really ran that arm, judged by the NoC
each RISC's transactions were recorded on. `census_matches` already refuses an
A-against-B comparison as a side effect, because the NoC is part of the class
key; `--expect-arm` additionally catches *both* sides having drifted onto the
same wrong arm. It does **not** separate A from C, whose NoC pairing is
identical by design — add `--peer-noc 2,3` (the value the run's
`nocevbench-config` line reports) and every transaction must address that core.

### The gates

Six gates run before any comparison is reported, and a failure is a **refusal**,
not a warning. `--expect-arm` adds a seventh, opt-in and meaningful only for
this program.

| gate | passes when | refuses when |
| --- | --- | --- |
| `events_present` | a kernel zone **and** at least one NoC event, both sides | a zone with no NoC events — the fingerprint of `TT_METAL_DEVICE_PROFILER=1` without the recorder compiled in, which decodes as "this kernel moved no data" |
| `single_window` | one `run_host_id`, one `ZONE_START`/`ZONE_END` pair per side | two runs concatenated, or a trace globbed from a directory |
| `per_core` | one core each side, the same core, the same RISC | a silent `(1,1)`-vs-`(1,2)` Wormhole/Blackhole mismatch, or BRISC compared against NCRISC. `--map-core` states an intended correspondence out loud |
| `barriers_pair` | every barrier START is closed by an END | a dropped event — the profiler drops silently on overflow, and a truncated stream still partitions cleanly |
| `partition_closes` | every bucket ≥ 0 and the buckets sum to the zone span | a non-monotonic timestamp: a torn 44-bit wall-clock read, a wrap, or a zone endpoint out of place |
| `census_matches` | every `(noc, type, bytes)` class and its count identical on both sides | the two sides ran **different work** — a different transfer size or chunk count. This is what keeps the comparison a comparison of timing |
| `arm_matches` (opt-in) | each RISC's transactions are on the NoC the named arm asks for, on both sides | an arm-B run that silently kept arm A's pairing — well-formed, passes everything else, and supports the opposite conclusion |

**A guard that cannot fail is as damaging as one that cannot pass.**
`tt_sim/perf/noc_events_test.py` builds a passing case and a refusing case for
every one of them, from inputs a real session could plausibly produce.

### Synthetic card data, both directions

No card session existed when these were built, so `testdata/` carries two
synthetic card traces derived from the real simulator run. They are stamped
**NOT-A-MEASUREMENT** in their filenames and in `testdata/README.md`, both carry
an identical transaction census to the simulator run, and both are guarded by
the test suite:

* `card-agreeing-…` — every interval 5 % longer. **PASSES.**
* `card-compensating-…` — the same trace with 2 500 cycles moved out of BRISC's
  `SEMAPHORE_WAIT` interval and into its trailing `local` interval, in a way
  that leaves **every inter-event distance among the NoC events untouched**.
  `E_total` **4.78 %** — well inside the limit — against `E_int` **48.33 %**,
  compensation **10.12×**, and **every per-class latency passes at 4.8 %**.
  **FAILS, on the partition alone.**

The second is the whole argument for this leg in one file: every envelope check
in this repo would pass it, and so would a per-transaction latency check. Only
the decomposition sees it.

## What this will and will not license

**Will**, once a card session exists and the gates pass:

* that tt-sim's NoC timing has been checked against silicon **through the same
  instrument and the same artefact**, decomposed by mechanism, per core and per
  RISC, at a stated `E_int` — with the compensation `E_int / E_total` quoted, so
  a reader can see how much of a matching total was luck;
* the **first** check of any kind against hardware for the flight cycles the
  five NoC-coordinate commits of 2026-08-13/16 changed. Those were validated
  against tt-sim's own endpoint-consistency invariant, which is a
  self-consistency argument and cannot detect an error both sides of it share;
* a value for the residual between observed latency and modelled flight — the
  DRAM service, NIU issue overhead and poll granularity the flight model does
  not claim to cover;
* an evidenced **negative**: that the two disagree. That is a real result, it is
  the most valuable thing this leg can produce, and it must be reported rather
  than fixed in the same change.

**Will not**, ever, on this instrument:

* **a per-packet flight time on hardware.** There isn't one. Every transaction
  event is stamped at issue and the only completion timestamp is a barrier's.
  Nothing here may be quoted as silicon's NoC latency for a single packet;
* **anything about `noc_flight_cycles` directly.** It has no card counterpart,
  it is reported as a simulator-side diagnostic only, and it must never become a
  gate. "Plus queueing" is already inside it;
* **two independent detectors.** The `read_wait`/`write_wait` buckets and the
  per-class latencies are two readings of the same intervals;
* **anything about addresses, or about NoC virtual channels.** The addresses are
  unreachable in the file-dumping path; `vc` is `-1` for every read;
* **anything about multicast NoC behaviour**, which no arm exercises, or about
  congestion, which needs concurrent flows — `perfbench/nocbench` is the harness
  for that. Arm C reaches a *second core's L1*, but it is still one flow issued
  by one core: the peer runs nothing, so nothing contends;
* **provenance for a cycle cost.** Nothing measured here may enter
  `unit_costs.yaml`. A profiler observation is corroboration, not a documented
  source;
* anything about the other two rung-4 programs. Tensix-bound is
  `perfbench/mechbench`; the RV-bound leg has no instrument on Wormhole at all.

One more, worth stating because it is the most tempting mistake: a `PASS` here
is a statement about **this program's** NoC transactions on **this core**, at
**these two sizes**, on **this arm's route**. It says nothing about a route, a
size or a transaction type these arms do not reach.

And one that the arms add: **a simulator-side number is not a result.** Every
figure in "What tt-sim says" above is the model predicting itself. Its only
jobs are to bring the arms up before card time is spent, and to be the thing the
card is compared *against* — including where it says the two arms should be
indistinguishable, which is a prediction the card can falsify.
