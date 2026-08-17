# mechbench — checking tt-sim's *interior* against silicon's stall counters

Rung 4's **mechanism-attribution leg**. Three silicon comparisons already exist
in this repo — component slopes to ~1 %, nekbone's per-core zones within
± 10.2 %, `dramratebench`'s shape — and **all three are envelope checks**. None
inspects the interior, and a total can agree while the decomposition behind it
is wrong in compensating directions. This is the leg that looks inside.

It fits nothing. **A disagreement is the result**, not a bug to be closed by
tuning the simulator.

## What makes this possible, and what makes it worth doing

`TT_METAL_PROFILE_PERF_COUNTERS` programs the Tensix hardware performance
counters, and **needs no in-kernel instrumentation at all**: the firmware starts
and stops the bank on TRISC1 around the compute kernel
(`RecordPerfCounters()` in `tt_metal/hw/firmware/src/tt-1xx/trisck.cc`) and
reads it back on BRISC after the TRISCs finish
(`ReadPerfCounters()` in `brisc.cc`). **The measured window contains no
markers.** That is why this is worth more than a zone match: a zone
decomposition pays ≥ 28 device cycles per marker and can only cut the span where
a marker was placed, whereas these counters resolve *why* a thread was not
issuing, at cycle resolution, without touching the program.

tt-sim models the same registers (`tt_sim/misc/perf_counters.py`, landed
2026-08-13) and sources the `INSTRN_THREAD` bank from quantities it already
tracks. So the *same binary* under the *same environment* produces the *same
artefact* on both sides — a `profile_log_device.csv` full of `PerfCounter`
markers — and one parser reads both.

## The criterion

For one core, with mechanisms `m` partitioning the span and **an explicit
unattributed bucket on both sides so the denominators match**:

```
E_total = |Σ c_sim − Σ c_hw| / Σ c_hw
E_int   = Σ |c_m,sim − c_m,hw| / Σ c_hw
```

Pass requires **`E_int ≤ 25 %`** *and* `E_total ≤ 10 %`. The triangle inequality
gives `E_total ≤ E_int` always, so **`E_int / E_total` is the compensation,
measured** — the number a passing total cannot fake. `tt_sim.perf.stall_attribution`
prints it on every comparison line.

**Every criterion is per core.** On the 2026-08-10 part the whole physical
column `x = 11` keeps a wall-clock epoch 1.5e13 cycles from the rest of the die,
so no cross-core span is admissible; the `per_core` gate refuses one rather than
leaving it to discipline.

## The partition, and why it is shaped this way

Only four counter families map onto quantities tt-sim tracks. The partition uses
exactly those and nothing else. Per core, over **`3 × ref_cnt` thread-cycles** —
three Tensix threads each observed for the same window:

| bucket | from |
| --- | --- |
| `issue_t` | `THREAD_INSTRUCTIONS_t` |
| `sem_empty_t` | `WAITING_FOR_NONZERO_SEM_t` |
| `sem_full_t` | `WAITING_FOR_NONFULL_SEM_t` |
| `idle_t` | `ref_cnt − THREAD_STALLS_t − THREAD_INSTRUCTIONS_t` |
| `srca_valid` / `srcb_valid` | `WAITING_FOR_SRC{A,B}_VALID` |
| `srca_clear` / `srcb_clear` | `WAITING_FOR_SRC{A,B}_CLEAR` |
| `unattributed_stall` | `Σ THREAD_STALLS − Σ sem − Σ src` |

Three decisions in that table are load-bearing.

**The four Src conditions sit at core level, not per thread.** The hardware
counts them that way — one shared counter instance each, not a per-thread block
— and splitting them across threads would be fitting. A **second, purely
per-thread** partition of `ref_cnt` is reported alongside
(`issue`, `sem_empty`, `sem_full`, `other_stall`, `idle`), and there the Src
waits are inside `other_stall`, because at thread granularity hardware does not
say whose they are.

**`unattributed_stall` is explicit and is allowed to be large.** It is what
makes both sides' denominators the same quantity. Hiding it — normalising the
named reasons to 100 % — would make `E_int` a comparison of *shares* and destroy
the `E_total ≤ E_int` inequality the compensation ratio depends on.

**`WAITING_FOR_{UNIT}_IDLE_{n}` is not used, deliberately.** The vendor tech
report contradicts itself across two sections and the later one is right: those
count cycles a *unit was busy*, not cycles a thread was stalled by it, and
produce "> 100 %" values. `ANY_THREAD_STALL` and the `*_INSTRN_AVAILABLE_*`
family were likewise declined as unsourced. In tt-sim they read back zero with a
one-shot warning; `TT_SIM_STRICT_PERF_COUNTERS=1` raises instead.

### The closure check is a live test, not a formality

`unattributed_stall` goes negative exactly when the named stall reasons
out-count `THREAD_STALLS` — which is the symptom the tech report describes for
the family it contradicts itself about. The Src conditions come from the same
RTL generator. **If the partition fails to close on real card data, that is this
leg's finding and it must be reported, not reshaped away.**

## The program

`src/mechbench.cpp` — a normal tt-metal program. One core, three kernels, a DRAM
round trip, a host-side check of the arithmetic that is **exact** (bf16 with
small integers, so a mismatch is a mismatch and not a tolerance argument). Two
arms differing in exactly one instruction:

| arm | inner op | mechanism it is meant to load |
| --- | --- | --- |
| `elw` | `add_tiles` | cheap Matrix work per unpack pair → the **math** thread waits on the unpackers: `SrcA/SrcB VALID` |
| `mm` | `matmul_tiles` | ~an order of magnitude more Matrix occupancy for byte-identical unpacker work → the **unpackers** wait on the Matrix unit: `SrcA/SrcB CLEAR` |

Everything else is identical: tile count, tile format, circular-buffer depth (2
on every CB, so the unpacker can genuinely run ahead), NoC traffic, and both
data-movement kernels. **The pair is the point.** A model wrong in compensating
directions can match one arm's interior; it cannot match both, because the two
arms load the same mechanisms in opposite directions. The tile stream is
*streaming*, not resident — a resident-tile inner loop never re-unpacks and so
can never produce a `SrcA CLEAR` stall at all.

`examples/four` is the obvious starting point and is where the counter path was
first demonstrated, but it is a four-iteration ELWADD whose interior is one
mechanism: on tt-sim it reports `THREAD_STALLS_1 = 102` with
`WAITING_FOR_SRCA_VALID = 101`. A criterion evaluated on that is a criterion
evaluated on a single number.

---

## The card protocol

Copy-pasteable, for someone who is not the person who wrote it. Nothing from
this repo is needed on the card box beyond `perfbench/mechbench/` and a built
tt-metal. No Tracy, no `tt-exalens`, no board reset, no root.

```bash
# 0. Get the tree onto the card box. Exclude build/ -- a CMakeCache.txt records
#    the absolute path it was generated in and cmake refuses a foreign one.
rsync -av --exclude 'build/' perfbench/mechbench/ <card-box>:~/mechbench/

# 1. On the card box.
export TT_METAL_HOME=/path/to/your/built/tt-metal     # must export TT-Metalium
cd ~/mechbench

# 2. Look at the schedule and the wall estimate before committing to it.
./run_card.sh --list

# 3. Run it. ~2 minutes of runs after the first build (~2 min).
./run_card.sh --out ~/tt_traces/mechbench-session

# 4. Check on the spot: every run must say PASS, and the three repeats of an arm
#    must agree. If they do not, say so in the handover -- do not re-run and
#    keep the better session.
cat ~/tt_traces/mechbench-session/summary.txt

# 5. Send the WHOLE directory home.
rsync -av ~/tt_traces/mechbench-session/ <home>:~/mechbench-session/
```

### Which banks to enable, and in what combination

`TT_METAL_PROFILE_PERF_COUNTERS` is a **bitfield**, and the bits do not all
coexist:

| bit | value | bank | conflicts |
| --- | --- | --- | --- |
| 0 | 1 | `FPU` | none |
| 1 | 2 | `TDMA_PACK` | none |
| 2 | 4 | `TDMA_UNPACK` | none |
| 5 | 32 | **`INSTRN_THREAD`** | none |
| 3 | 8 | `L1_0` | one L1 bank per run |
| 4 | 16 | `L1_1` | one L1 bank per run |
| 6 | 64 | `L1_2` | one L1 bank per run |
| 7 | 128 | `L1_3` | one L1 bank per run |
| 8 | 256 | `L1_4` | one L1 bank per run |

**The five L1 banks share one hardware mux** (`RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL`),
so at most one L1 bit may be set per run — tt-metal `TT_THROW`s on two, it does
not silently pick one. `INSTRN`, `FPU`, `PACK` and `UNPACK` are separate
`tt_perf_cnt` instances and **do not conflict**: they can all be enabled in the
same run.

So the passes are:

| pass | mask | why |
| --- | --- | --- |
| **A — required** | `32` | `INSTRN_THREAD` alone. This is the leg. Every counter the partition consumes is in this bank. |
| B — corroboration | `39` (`1\|2\|4\|32`) | adds `FPU`, `PACK`, `UNPACK`. tt-sim models **none** of these and reads them back as zero with a warning, so they are **card-only context** and may not enter the criterion. Free to collect: same run, same window. |
| C1–C5 — optional | `8`, `16`, `64`, `128`, `256` | the five L1 banks, **one run each** because of the shared mux. tt-sim models none of them. Collect only if someone has asked for L1 port pressure. |

`run_card.sh` runs pass A by default and pass B with `--corroborate`; the L1
passes are `--l1`, and it runs them as five separate invocations because it has
to.

**The mask must be identical on both sides.** Enabling the profiler at all adds
firmware and a DRAM push per marker; that overhead cancels between simulator and
card only if both ran the same mask. `stall_attribution` prints the counter
families it found on each side so a mismatch is visible, but it cannot know what
you meant.

### The schedule, and why it is repeated

| slot | arm | tiles | mask | runs |
| --- | --- | --- | --- | --- |
| 1 | `elw` | 256 | 32 | 3 |
| 2 | `mm` | 256 | 32 | 3 |

Three runs per arm, each into **its own output directory**. Repeats are not
averaged and must not be concatenated: the analysis refuses a log carrying more
than one `run host ID` or more than one latched `ref_cnt`, because that is two
windows and not one. The three exist so the operator can see the counters'
own repeatability before sending anything, and so the home side can pick a
representative run and *say which*.

Arms are interleaved (`elw`, `mm`, `elw`, `mm`, …) rather than blocked. This
project has been bitten by thermal drift turning a blocked schedule into a fake
result; the counters are cycle counts and far less exposed than a power reading,
but interleaving is free and the discipline is established.

**Expected wall time**: the first build is ~2 minutes (tt-metal supplies every
flag; there is nothing to configure). Each run is device-open plus one launch —
budget **20 s**, dominated by device open and the JIT cache check, not by the
kernel, which is microseconds. So the six required runs are **~2 minutes**, and
the whole thing including the build is **under 5**. `--corroborate` adds six
runs (+2 min); `--l1` adds thirty (+10 min), because the five banks share one
mux and each needs its own run. `./run_card.sh --list` recomputes all of it.

### What lands in the output directory

```
summary.txt           per-run PASS/FAIL, span, and the headline counters
runs/<arm>-<n>/       one directory per run
  stdout.log          everything the program printed
  .logs/profile_log_device.csv     <- the artefact the analysis consumes
env.txt               the exact environment, tt-metal commit and mask used
```

Send **all** of it. `stdout.log` is what makes a suspicious counter row
diagnosable later.

---

## The simulator side

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/mechbench/run_sim.sh \
    --arch blackhole --tiles 8 --out /tmp/mechbench-sim
```

**Keep the cost model on** — `run_sim.sh` exports `TT_SIM_COST_MODEL=1` and
`--no-cost-model` exists only to demonstrate what goes wrong. With it unset no
Tensix backend arms an occupancy, so the Matrix unit hands a Src bank back the
cycle it takes it and the `srca_clear` / `srcb_clear` columns are **absent, not
zero** — a column of zeros that reads exactly like "this never happens".
`stall_attribution` prints a note when it sees that fingerprint.

### The readback blocker, and what closing it did and did not settle

**The profiler readback used to come back empty, and no longer does.** The
device profiler's `profile_log_device.csv` arrived containing nothing but its
header — no counter samples *and no zone markers either* — at `elw 32` with the
cost model off and at **every** tile count tried with `TT_SIM_COST_MODEL=1`. It
was not a loss in transit. tt-metal's BRISC writes `RUN_MSG_DONE` before
`finish_profiler()` publishes the run, the host stops driving the clock at
`DONE`, and its very next wire message is the control-vector read — so the
firmware got the bridge's 100-cycle poll budget to do a job that measures at
~1 400 cycles, and `readRiscProfilerResults` early-returned on a zero
`HOST_BUFFER_END_INDEX`. The bridge now waits for that publish, and for the
pushes it issues to land in DRAM, at the one read that consumes it
(`Device.settle_profiler_flush`; `tt_sim/bridge/profiler_readback_test.py`
pins it). `TT_SIM_CYCLES_PER_POLL=5000` is no longer needed, and the
cost-model-on regime this leg needs is now the default one.

That workaround was also hiding how narrow the old margin was: at 100 cycles
with the cost model *off* the publish loop got through exactly one iteration,
so the runs that looked fine carried **only BRISC's markers** and silently
dropped NCRISC's and all three TRISCs'. Any decomposition collected before
2026-08-13 is suspect for that reason and should be recollected.

**What that did not settle: the `mm` arm still does not reverse the Src
direction.** Collected in the intended regime (`--tiles 8`,
`TT_SIM_COST_MODEL=1`, default poll budget) both arms report `srca_clear = 0`
and `srcb_clear = 0`:

| arm | span (`ref_cnt`) | `srca_valid` | `srcb_valid` | `srca_clear` | `srcb_clear` |
| --- | --- | --- | --- | --- | --- |
| `elw` | 3 275 | 2 989 (30.4 %) | 40 | 0 | 0 |
| `mm` | 3 887 | 3 572 (30.6 %) | 0 | 0 | 0 |

This was previously attributed to the readback failure — "the regime that could
show it cannot be collected". That explanation is now spent: the regime *is*
collected and the reversal is still absent. **This is a finding about tt-sim's
Src-ownership modelling, and it is exactly the kind of interior disagreement
this leg exists to surface.** It is not a reason to change the criterion, and
it must not be tuned away before card data exists to check it against.

The simulator runs a few tens of thousands of cycles per second where hardware
runs a billion, so **the tile counts will not match between the sides** unless
someone is willing to spend a night on it. That is a real limit on this leg and
it is stated rather than papered over: a comparison between `--tiles 8` on the
simulator and `--tiles 256` on the card is a comparison of two different
programs. Either run the card at the simulator's tile count (cheap — add
`--tiles 8` to `run_card.sh`) or run the simulator overnight at the card's. The
protocol above uses 256 on the card because that is the sensible default for
hardware; **pin the two together before collecting anything you intend to
quote.**

## The analysis

```bash
python3 -m tt_sim.perf.stall_attribution \
    --sim  /tmp/mechbench-sim/elw/.logs/profile_log_device.csv \
    --card ~/mechbench-session/runs/elw-1/.logs/profile_log_device.csv \
    --report report.txt --json report.json
```

### The gates

Five gates run before any comparison is reported, and a failure is a
**refusal**, not a warning.

| gate | passes when | refuses when |
| --- | --- | --- |
| `counters_present` | all 16 partition counters are in both logs | any is missing — the wrong bank was enabled |
| `armed` | `ref_cnt > 0` and something incremented, on both sides | an all-zero bank: the kernel never armed it, and zero reads as "nothing ever stalled" |
| `single_window` | one `run host ID`, one latched `ref_cnt` per side | two runs concatenated, or a bank left counting |
| `per_core` | one core each side, and the same core | a unit pooled from two coordinates, or a silent (1,1)-vs-(1,2) mismatch. `--map-core` states an intended correspondence out loud |
| `partition_closes` | every bucket ≥ 0 and both partitions sum to their own span | a negative `unattributed_stall` (named reasons out-counting `THREAD_STALLS`) or a negative `idle_t` |

**A guard that cannot fail is as damaging as one that cannot pass.**
`tt_sim/perf/stall_attribution_test.py` builds a passing case and a refusing
case for every one of them, from inputs a real session could plausibly produce —
a concatenated log, a Wormhole-numbered core against a Blackhole-numbered one,
an unarmed bank, a partition that does not close.

### Synthetic card data, both directions

No card session exists for this leg yet, so `testdata/` carries two synthetic
card logs derived from the real simulator decomposition. They are stamped
**NOT-A-MEASUREMENT** in their filenames and in `testdata/README.md`, and both
are guarded by the test suite:

* `card-elw-agreeing-…` — a plausible ± 5 % card. **PASSES.**
* `card-elw-compensating-…` — **the same total** (`E_total = 2.9 %`, well inside
  the envelope limit) with the `srca_valid` mass moved into
  `unattributed_stall`. `E_int = 39 %`, compensation ratio **13.5×**. **FAILS.**

The second is the whole argument for this leg in one file: every envelope check
in this repo would pass it. Note also that its three *per-thread* comparisons
pass — the compensation is only visible in the core-level partition, because
that is the only place the Src conditions are decomposed.

## What this will and will not license

**Will**, once a card session exists and the gates pass:

* that tt-sim's cycle attribution has been checked against silicon's own
  hardware stall counters, **mechanism by mechanism**, per core, at a stated
  `E_int` — and, crucially, with the compensation `E_int / E_total` quoted, so
  the reader can see how much of a matching total was luck;
* the first check that has ever touched the wired Tensix backends at all —
  rungs 1 and 2 validate them *not at all*;
* an evidenced **negative**: that the interior does not agree. That is a real
  result, it is the most valuable thing this leg can produce, and it must be
  reported rather than fixed in the same change.

**Will not**, ever, on this instrument:

* anything per-instruction or about ordering. These are counts over a window;
  they say how many cycles went where, never in what sequence;
* anything about `WAITING_FOR_{UNIT}_IDLE`, `ANY_THREAD_STALL` or the
  `*_INSTRN_AVAILABLE_*` family — declined as unsourced or self-contradicting,
  and not to be resurrected because a gap needs filling;
* anything about the `FPU`, `PACK`, `UNPACK` or `L1` banks, which tt-sim does
  not model and which read back zero;
* provenance for a cycle cost. The counter semantics come from a **vendor tech
  report and RTL**, not the ISA docs — `vendor_source`, fine for corroboration,
  **disqualifying as provenance**. No counter value may become a cost, and
  nothing measured here may enter `unit_costs.yaml`;
* anything about the other two rung-4 programs. **RV-bound and NoC-bound are out
  of scope here.** The RV-bound leg has no instrument on Wormhole at all (no
  Zicsr, so no `mcycle`/`minstret`). The NoC-bound leg is built — a different
  artefact and a different parser, `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` read
  by [`perfbench/nocevbench`](../nocevbench/README.md) — and it corrected this
  file's own description of it: hardware records **no** per-transaction
  completion time, so there is nothing to compare `noc_flight_cycles` against
  directly.

One more, worth stating because it is the most tempting mistake: a `PASS` here
is a statement about **this program's** mechanisms on **this core**. It says
nothing about a backend these two arms do not exercise, and the SFPU, the mover
and the packer's own internals are all in that category.
