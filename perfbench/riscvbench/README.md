# riscvbench — running it on real hardware

**You have a Tenstorrent card. This page is everything you need; you do not need
to know anything about tt-sim.** It asks you to build one program, run it seven
times, and send back seven CSV files. Budget **20 minutes**, most of it the
build: the two main runs are a minute or two each and the two extra phase G runs
and the phase-Q drain run are seconds.

**`./run_card.sh` does all of it** — build, all seven documented runs plus a
focused sixth for the store-coalescing, multiply and divide probes, the
analysis, and a list of what to check before sending. The rest of this page is
what it does and why, and is worth reading once. If you are running the whole
roadmap block rather than this one program, use
[`../run_card_session.sh`](../README.md) instead.

If you want to know *why* the benchmark is shaped the way it is, and what its
numbers can and cannot prove, read
[`docs/plans/riscv-front-end-benchmark.md`](../../docs/plans/riscv-front-end-benchmark.md)
instead. This page is the runbook.

---

## What it does, in four sentences

It times short bursts of ordinary RISC-V instructions — `addi`, `mul`, `divu`,
loads, stores, branches — running on one of a Tensix tile's baby RISC-V cores,
using the device's own cycle counter. It also times bursts of `.ttinsn` words,
which is how a baby core pushes an instruction to the Tensix coprocessor, and
that is the number the whole exercise is for. Everything is run at four
different burst lengths and reported as the **slope**, so kernel launch, timer
overhead and loop setup cancel out exactly and never appear in a result. It
finishes by asking four questions nothing has ever measured: how deep the
Tensix instruction queue is, **whether that queue is shared between the three
baby cores or private to each**, whether a taken branch costs more than a
not-taken one, and whether a loop stops running at one instruction per cycle
once it gets big.

It is a **normal tt-metal program**. It creates a device, launches programs on
one core, reads a small buffer back and exits. It does not need Tracy, the
device profiler, `tt-exalens`, or a board reset.

**It leaves the card exactly as it found it.** The only Tensix instructions it
issues are `NOP`, `SFPNOP`, `SETDMAREG` and `ADDDMAREG`. It sets no data-valid
bit, acquires no SrcA/SrcB bank, starts no unpack or pack, and writes no backend
configuration — so unlike `tensixbench`, there is nothing for a successful run
to leave behind and nothing the next run has to wait for.

---

## What you need

| | |
| --- | --- |
| Hardware | One Wormhole **or** Blackhole card. The program detects which and adapts; you do not choose. |
| tt-metal | A built checkout that exports the `TT-Metalium` CMake package, i.e. `<tt-metal>/build/lib/cmake/tt-metalium/tt-metalium-config.cmake` exists. |
| Tools | `cmake` ≥ 3.22 and a C++ compiler. tt-metal supplies every flag and include path. |
| Time | ~15 min to build, ~1 min per run, two runs. |

The program writes to one core (logical `(0,0)`) and three small L1 buffers, all
obtained through the tt-metal allocator. It does not touch DRAM and cannot
corrupt anything that outlives the process.

---

## Build

```bash
export TT_METAL_HOME=/path/to/tt-metal          # your built checkout
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"

cd perfbench/riscvbench/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# -> build/riscvbench
```

## Run

Run it **from `src/`** — the program refers to its kernel by the relative path
`kernels/compute/rv_probes.cpp`.

```bash
cd perfbench/riscvbench/src
export TT_METAL_SLOW_DISPATCH_MODE=1     # see note below
unset TT_METAL_SIMULATOR                 # make sure you are on the real card

# Run 1 -- the main one. All seven phases, one/two/three issuing TRISCs.
./build/riscvbench --blocks 32

# Run 2 -- the same thing at a different burst size. Two minutes, and it is
# worth it: see "Why the second run".
./build/riscvbench --blocks 8 --out riscvbench-$(uname -n)-blocks8.csv

# Runs 3 and 4 -- phase G's other two footprint sets, seconds each. Run 1
# already covered set 0. Phase G is a SEPARATE PHASE because phase F's kernel
# is at tt-metal's size limit, and it is split into three compile-time sets
# because its three bodies do not fit in one kernel either.
./build/riscvbench --phase g --variants t1 --blocks 32 --gset 1 --out riscvbench-g1.csv
./build/riscvbench --phase g --variants t1 --blocks 32 --gset 2 --out riscvbench-g2.csv
./build/riscvbench --phase g --variants t1 --blocks 32 --gset 3 --out riscvbench-g3.csv
./build/riscvbench --phase g --variants t1 --blocks 32 --gset 4 --out riscvbench-g4.csv

# Run 5 -- seconds, and it is the one open question this benchmark has a
# written prediction for. See "Run 5" under "What the interesting answers look
# like" for the numbers that confirm it and the numbers that refute it.
./build/riscvbench --phase qs --variants t1 --blocks 32 --out riscvbench-qdrain.csv
```

Runs 1 and 2 include **phase S**, which is new and is the one that needs all
three thread sets: its answer is a *ratio between thread counts* and a single
variant produces no verdict at all. `--variants t1,t2,t3` is the default, so
nothing extra is needed — but if you cut the run down, keep at least `t1,t2`.

`--blocks 32` is the **hardware setting** — bigger than the default of 4, which
is sized for the simulator. Larger bursts push the fixed costs further into the
noise; on silicon they cost milliseconds.

> **Do not go below 32 for the primary run.** The 2026-08-05 Blackhole runs
> collected both, and at `--blocks 8` **six of the seven phases were refused by
> the validity gate** — five of the seven R² failures being the `loop_overhead`
> control itself fitting to R² < 0.99, and a control that does not fit is a
> phase that cannot be read whatever the probes did. Its numbers agreed with the
> `--blocks 32` run's to a few thousandths of a cycle anyway, which is a useful
> reminder that agreement is not validity. Both are tracked in
> `tt_sim/perf/datasets/` and the second one's header explains why it is kept.
> The minimum usable block count is somewhere above 8 and at or below 32.
>
> **Phase Q is the exception and it matters.** `--blocks` sets the four fitted
> points of the *slope* phases and nothing else: a phase-Q point is labelled
> `n0 << k` (the burst index) where a slope point is `base_blocks * (k + 1)`,
> and the kernel's `QPROBE`/`QLOOPPROBE` macros never read `base_blocks`. So a
> phase-Q reading is independent of `--blocks`, and the two banked runs
> reproduce every loop-form point to within two cycles at a quarter the block
> count. A run whose slope phases are refused can still be read for phase Q —
> and one of them is where the queue depth in §Q comes from.
>
> **This is not a licence to use `--blocks 8` as a cross-check, and the card
> session used to.** A cross-check establishes *repeatability*, which means the
> same experiment twice; a shorter run is a different experiment whose slope
> phases are expected to fail. On 2026-08-09 the `--blocks 8` cross-check was
> refused on R, C, Q, F **and** G while the `--blocks 32` primary was refused on
> phase Q alone, and because both runs wrote into one `.out` file the session
> greped, the short run's failures condemned the long one. `run_card_session.sh`
> now runs the cross-check at the same `--blocks 32`, into its own `.out`, and
> grades the two separately.

### Phase Q's small-burst failures are expected on silicon

The 2026-08-09 primary run failed `TTRVBENCH_VALID_Q` on 15 monotonicity checks
and **every one of them was at n ≤ 16** — `n=1 -> 14 cycles, n=2 -> 13 cycles`
and the like. That is the same 6–23 cycle scatter of a cold, once-only burst
that the phase-S measured tolerance exists to absorb, and this page has
documented it since phase S was added. So the session's verdict logic
(`perfbench/card_session_verdicts.sh`) treats a **phase-Q-only** failure whose
complaints are all at n ≤ 16 as expected, and still reports `SUSPECT` for a
phase-Q complaint at any larger n, or for any other phase failing. The
threshold is this page's own, not one picked to make a particular run pass;
`card_session_verdicts_test.sh` asserts a synthetic complaint at n = 64 is still
`SUSPECT`.

The first run writes `riscvbench-<arch>.csv` next to the binary. It is rewritten
after **every** program launch, so a run you have to kill part-way still leaves
the phases that did finish on disk.

> **`TT_METAL_SLOW_DISPATCH_MODE=1`** makes `LaunchProgram` the launch path.
> The benchmark works either way on hardware; setting it keeps the hardware run
> and the simulator run on the same path, which is the entire point of the
> exercise. If your tt-metal build refuses slow dispatch, run without it and
> **say so when you send the results** — it is a difference worth recording, not
> a reason to abandon the run.

### Why the second run

Every number here is a slope over four burst lengths, which cancels the fixed
cost exactly *provided the four points really are on a line*. The one thing that
breaks that is a **warm-up offset on the smallest point** — a cold instruction
cache, an empty Tensix queue, a DVFS transition — and it does not announce
itself: it tilts the line, leaves R² at 1.0000, and shifts the answer in the
third decimal. `tensixbench`'s first Blackhole run had exactly this, worth
0.2 %, and it was only found by looking at the raw points.

Running at two burst sizes is the cheapest possible check. If `--blocks 32` and
`--blocks 8` agree, the fits are clean. If they do not, the smaller one is
contaminated and the larger is the one to believe.

### If something goes wrong

There is no hang to expect here — no probe waits on any Tensix state — but if
one happens anyway:

- **Note which phase it hung in** (the program prints `phase <letter> [tN]:
  done` after each launch, so the hang is in the one after the last line
  printed) and say so. A hang is itself a result.
- `--phase r` / `t` / `c` / `q` / `f` / `s` / `g` runs one phase, and
  `--variants t1` runs only the single-thread launches. Between them any phase
  can be skipped. **Phase S is the exception to `--variants t1`**: its answer is
  a comparison between thread counts, so cutting it to one variant produces no
  verdict rather than a weaker one, and the program says so.
- `--probes 0xMASK` enables probe `i` with bit `i`; the summary table prints
  the probes in slot order. **Probe 0 is the empty-loop control and every
  slope phase needs it** — clearing bit 0 makes those phases unreadable.

Phase F builds a kernel with 8 KiB of instruction text in one loop body, and it
is **within a few hundred bytes of tt-metal's kernel config buffer** on the
release this was written against. If yours refuses to build or place it, run
`--phase rtcqsg` and say so; the other phases are unaffected. That ceiling is
also why the footprints *between* 1024 and 2048 are phase G rather than three
more phase F probes, and why phase G itself takes five runs — putting them all
in one kernel aborts the launch with `Program size (125040) too large for kernel
config buffer (70656)`, which is measured rather than predicted.

---

## What to send back

**All seven CSVs, plus the terminal output of each run if you have it** — the
summary tables, the per-phase read-outs and the validity verdicts are in the
terminal output and not in the CSV. The phase S verdict and the phase G step are
*only* in the terminal output.

Each CSV has one row per raw measurement and no derived numbers at all:

```
phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles
```

with the run's configuration in a `#` header line (`arch=`, `magic=`,
`probe_mask=`, `phases=`, `variants=`, `base_blocks=`, `gset=`, `stack_addr=`,
`scratch_addr=`, `div_dividend=`, `div_divisor=`). `gset=` matters for a phase G
run and for nothing else: it says which intermediate footprint was compiled in. `stack_addr=` matters more than it looks:
tt-metal chooses where a TRISC's stack lives, and the two candidate regions have
load latencies six cycles apart on Wormhole, so the analysis classifies the
address rather than assuming one.

If you ran with a reduced `--phase` or `--probes`, or without slow dispatch, or
the program exited non-zero — **say which**. All of those change how the numbers
are read.

---

## How to tell a good run from a silently bad one

The program checks itself and prints a verdict **per phase**:

```
TTRVBENCH_VALID_R: yes
TTRVBENCH_VALID_T: yes
TTRVBENCH_VALID_C: yes
TTRVBENCH_VALID_Q: yes
TTRVBENCH_VALID_F: yes
TTRVBENCH_VALID_S: yes
TTRVBENCH_VALID_G: yes
TTRVBENCH_VALID: yes
Completed successfully on the device
```

A failing phase names its failing checks, and the exit status is a bit mask
(1 = R, 2 = T, 4 = C, 8 = Q, 16 = F, 32 = S, 64 = G, 0 = clean).

**Send the CSV either way.** The verdict is per phase precisely because the
phases measure unrelated things: a phase F that measures nothing does not make
the phase T rows in the same file any less good.

The checks are:

- **Linearity.** Every slope probe is fitted against its four burst lengths and
  must reach R² ≥ 0.99. A probe that does not is marked `<-- NONLINEAR`. This
  catches a counter that wrapped, a burst short enough to be swamped by its own
  start-up, or a thread that was descheduled mid-measurement.
- **Monotonicity.** More instructions must take more cycles.

**Phase S carries a measured tolerance on the monotonicity check, and only phase
S.** Its reference burst is four instructions, so the step from n = 4 to n = 8 is
worth ~12 cycles at one issuing thread — comparable to what one cold, once-only
burst scatters by, which is exactly why phase Q's cascade failed 17 of these
checks at n ≤ 16 on silicon. Rather than exempt the phase or pick a constant,
`s_co_repeat` runs `s_co_plain` a **second time** and the largest disagreement
between the two is what one raw point can be wrong by. A phase-S pair is flagged
only when it falls by more than that. Every other phase keeps a tolerance of
zero, unchanged.

**Phase Q is deliberately not gated on linearity**, and that is not laziness. It
sweeps *burst length* looking for a knee; a straight line through it would be
the null result, not the healthy one, so requiring one would score the
interesting outcome as a failure. Its own control probe `q_ctrl` is exempt from
the monotonicity check, and only that one: it runs the cascade with an **empty
body**, so it has no burst to be monotone in, and on silicon its cost wobbles
non-monotonically by 6–23 cycles with which branches happen to be taken. Every
other phase-Q probe, including the three loop-form probes that carry the sweep
out to n = 1024, must grow with the burst like everything else.

### The one check that matters most

The program prints a section titled **"Is the instrument live?"**. Read it
before anything else.

The trap this benchmark is designed around is that **a run where everything
reads exactly `1.000` is simultaneously the expected answer and the signature of
a benchmark that measured nothing.** A baby RISC-V core issues at most one
instruction per cycle, so 1.000 is the floor for every probe here, and a broken
timer produces the same reading as a perfectly working core that never stalls.
`tensixbench` hit this: against the simulator every one of its probes reads
1.000, for a reason that has nothing to do with whether the benchmark works.

So four probes are singled out as the control. `rv_mul_dep`, `rv_div`,
`rv_load_chase` and `rv_store_spread` each have a **documented cost above one
cycle** on at least one architecture — a multiply's two cycles, a divide's six,
an L1 load-use latency, the five-cycle L1 store period. On silicon they must
read above 1.0. If they do, the timer resolves real per-instruction costs and a
1.000 anywhere else in the run is a *finding*. **If they all read 1.0 as well,
nothing in the run means anything**, and the program says so in those words.

Two further sanity checks you can make by eye, which the program does not
enforce because a violation is a finding rather than a fault:

1. **`loop_overhead` should be small and positive** — a couple of cycles per
   block. It is the empty control loop. If it reads a flat constant regardless
   of burst length, the compiler deleted the loop and every other number is
   inflated by whatever the loop really cost.
2. **Nothing should read below 1.00.** A baby RISC-V core cannot issue faster
   than that. A value like 0.4 means the timer, not the instruction, is being
   measured.

---

## What the interesting answers look like

Four numbers in this run are worth looking at the moment you have them, because
each of them is a question no document answers and no measurement has settled.

**1. `spread4 - fuse4`, in the phase T read-out.** Blackhole's
`PushTensixInstruction.md` says up to four *adjacent* `.ttinsn` words are fused
by the instruction cache and issued in one cycle. Two probes carry the identical
twenty instructions and differ only in whether any two `.ttinsn` words are
adjacent, so:

```
spread4 - fuse4 = +3.000   ->  the cache fuses, exactly as documented
spread4 - fuse4 = +0.000   ->  it does not, and the documented fusion is
                               something a compiled kernel cannot reach
```

**+3 would be the more surprising and more useful answer**: it would mean the
RISC-V push cost is *below* one cycle per Tensix instruction, which no sustained
measurement — this one's, or `tensixbench`'s — could ever have shown, because
the same page caps the queue's drain rate at one per thread per cycle.

**2. The phase Q knee, now swept to a burst of 1024.** Below the Tensix
instruction queue's depth the core runs ahead and each extra `ADDDMAREG` costs
one cycle; above it the core is back-pressured and each costs the unit's
occupancy, which `tensixbench` measured at 3.0 on Blackhole. **The burst length
at which the marginal cost steps up is the queue depth, and nothing in either
the ISA documentation or the vendor trees publishes one.**

The 2026-08-05 run swept to 128 and that was not far enough: at **one** issuing
thread the core had still not stopped running ahead at the longest burst, so the
backlog it had absorbed was still growing where it needed to have levelled off,
and a depth *in entries* was not resolvable. The sweep now goes to **1024**, in
two burst forms, and the read-out prints both:

```
cascade  n = 1..128     2^p `.ttinsn` words emitted straight-line, unchanged
loop     n = 16..1024   n/16 iterations of one 16-instruction block
```

**Why two forms, which is the one thing to understand before reading this
phase.** A 1024-word straight-line burst is 4 KiB of instruction text, and phase
F of this same benchmark found a fetch cliff in exactly that octave. Every
phase-Q burst runs *once*, cold, with its own instruction fetch inside the timed
region — so a longer straight-line burst would have folded a fetch cost that
*grows with n* into the one measurement whose entire question is whether cost per
instruction grows with n. It would have looked like a queue knee. The loop form
has a **64-byte body at every burst length**, so nothing in its column can be
fetch, and `q_loop_addi` — the identical loop with `addi` bodies — measures what
the form itself costs rather than assuming it. The two forms overlap at
n = 16…128 and the read-out prints the comparison, so "did changing the form
change the quantity" is measured too.

What to look for, in the read-out's own words:

- **`KNEE between n=A and n=B`** — the marginal cost of one more instruction has
  reached the drained rate, so the core is back-pressured from there on.
- **`the backlog FLATTENED at ~X cycles`** — the real prize. It means the burst
  stopped absorbing, and `X / (drained rate)` is the queue's depth **in
  entries**. **It is a LOWER BOUND**, because this line drops the reference
  burst's own occupancy and phase S's read-out does not. On Blackhole at one
  issuing thread it printed ~14–16 pre-drain and ~22 post-drain, against a
  settled depth of **~31–32** once the reference-burst term is added back and
  both forms are drained. Read the phase-S block and the reconciliation below
  it, not this line alone.
- **`BACKLOG STILL GROWING` / `NO KNEE up to n=1024`** — also a result, and an
  honest one: the queue is deeper than this sweep reaches. Do not read a depth
  off a backlog that is still growing; the program refuses to, and so should
  you.
- **`BACKLOG NEGATIVE` / `INSIDE THE NOISE FLOOR`** — the slot has no signal and
  no depth is printed. The backlog is a difference of two differences of raw
  single-shot points, so it has to clear **twice the `q_ctrl` spread this run
  measured** before it can be divided by anything; below that the subtraction
  can pass through zero, and work in flight cannot be negative. Expect this at
  two and three issuing threads, where the drained rate is 2–3× higher and the
  same queue's backlog is that many times smaller in cycles. It is a statement
  about the instrument, not about the queue.

**3. `taken - not taken`, in the phase C read-out.** Two probes execute the
identical dynamic instruction sequence — a branch whose target is the address
the not-taken path falls through to — and differ in exactly one bit. The tables
record a mispredict as a 2-cycle bubble on Wormhole and 4 on Blackhole, which is
how much one *costs*; how *often* one happens is undescribed, which is why
tt-sim charges nothing for branches at all. A non-zero delta here would supply
the missing half.

**4. The phase F row, and phase G's three narrowing points.** Six loop bodies
from 256 bytes to 8 KiB of instruction text, same instruction throughout. A flat
row says instruction fetch is not the limit anywhere in that range. A step
locates **a boundary in loop-body size** that nothing publishes — and the 2026-08-05
Blackhole run found one, flat at 0.998 through a 4 KiB body and 1.251 at 8 KiB.

Phase G narrows that octave. Each `--gset` compiles one intermediate — 1152,
1280, 1408, 1536 or 1792 instructions, i.e. 4.5, 5, 5.5, 6 and 7 KiB — against a
1024-instruction body **in the same kernel**, and prints the step between them.
Sets 3 (4608 B) and 4 (5632 B) were added on 2026-08-09 and are the two that
land INSIDE the (4096, 5120] bracket the ramp's onset had been narrowed to;
4608 says whether the rise has begun by then and 5632 whether it is linear in
footprint or a second step. Read the five runs
together: they bracket the boundary between two footprints that were actually
run, and that bracket is the whole claim.

> **It is a boundary in loop-body size and narrowing it does not make it a cache
> capacity.** No document in the ISA documentation or either vendor tree gives an
> instruction-cache size or a miss cost. A prefetch window, a TLB-like structure
> or an L1 access pattern would produce the same column, and nothing here
> separates them. The step's ~0.25 cycles/instruction is also an *amortised*
> figure over a body that is either entirely resident or entirely not, so it is
> not a miss cost either.

**5. The phase S verdict.** Phase Q resolved a queue depth at one issuing thread
and could not resolve one at any other, which left open whether that queue is
**shared between the three TRISCs or private to each** — the two are the same
device seen from one thread. Phase S answers it as a ratio, and on 2026-08-05
Blackhole silicon the answer was **PER-THREAD**, at 0.97×/0.95× with two issuers
and 1.06×/1.07× with three across two runs:

```
D at k issuing threads / D at one  ==  1.00   ->  PER-THREAD
                                   ==  1/k    ->  SHARED
```

The construction, and the two that look like they would work and do not, are
worth knowing before reading the number:

- **A second thread that only spins does not discriminate.** It pushes nothing,
  so it holds no queue entry under either hypothesis. It is run anyway, as
  `s_solo_plain`/`s_solo_sync`, because it is the control that separates
  "another core is *awake*" — competing for instruction fetch out of the same L1
  — from "another core is *issuing*". Both hypotheses predict it reads the same
  depth at t1, t2 and t3, so a departure there is not queue sharing.
- **A second thread issuing at a deliberately low rate does not either**, and
  for a reason that has nothing to do with this benchmark: queue occupancy is
  arrival rate times residence time, so a thread served faster than it arrives
  holds ~0 entries however long it runs.
- **Only a saturated second thread holds entries.** The price is that it also
  takes backend bandwidth — which is why phase Q's multi-thread slots looked
  hopeless — but that bandwidth is *measured in the same slot* by `s_co_sync`
  and divided back out.

The other half is the **reference burst**, which is `n = 4` here and `n = 16` in
phase Q. The backlog subtracts its value at the smallest burst as
`tensix_sync()`'s own cost, which is only true if the queue is empty there. It
is not: a core pushing at 1/p instructions per cycle against a backend draining
one every S leaves `n · (1 − p/S)` outstanding — ~10 entries at n = 16, and at
two or three issuing threads a *shared* queue's per-thread share may be smaller
than that, which makes phase Q's multi-thread backlogs structurally zero rather
than merely small. At n = 4 the same term is ~2, so phase S reports

```
D = backlog / S  +  4 · (1 − p/S)
```

with every term measured in the slot. **A consequence that was checked when the
numbers arrived, and it half-held:** phase Q's read-out drops that second term
entirely, so at one thread phase S should read a depth *above* phase Q's by
roughly the difference. It read 31 against phase Q's 16 — a lower bound
confirmed in direction, but a gap of 15.3 entries where 8.0 was declared. The
8.0 is exactly the reference-burst term; the remaining ~5–7 entries are a
difference between the two burst *forms* that is reproducible to the cycle in
four runs and was **not explained** at the time. The sweep prints the whole
reconciliation under `Do the two burst forms agree about the depth?`. **Run 5
below explained it**, so §1.10 now banks a number rather than a range.

### Run 5 — the untimed drain, pre-declared and confirmed

**`QLOOPPROBE` carries an untimed `ckernel::tensix_sync()` immediately before
`t0`**, which phase S has always had and phase Q never did. That one line is a
*test* of the leading explanation for the residual above — a saturated backend
never idles, so phase Q's `plain` was carrying the previous burst point's
residue forever — and **the prediction was written down before the line was**.
It is in
[`docs/plans/riscv-front-end-benchmark.md`](../../docs/plans/riscv-front-end-benchmark.md),
"The untimed drain, pre-declared before it was written", and the section after
it scores what came back.

**It has run** (2026-08-05, Blackhole, banked as
`tt_sim/perf/datasets/riscvbench-qdrain.csv`), so what follows is the recipe for
reproducing it rather than a request. Seconds on the card, one launch of phase Q
and one of phase S:

```bash
cd perfbench/riscvbench/src
./build/riscvbench --phase qs --variants t1 --blocks 32 --out riscvbench-qdrain.csv
```

Then, back in this repo:

```bash
python3 -m tt_sim.perf.riscv_bench_sweep --measured riscvbench-qdrain.csv
```

and read the raw t1 rows plus `Do the two burst forms agree about the depth?`.
**What confirmed it**, against the pre-drain run banked in
`tt_sim/perf/datasets/riscvbench-blackhole.csv`:

| probe (t1) | n = 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `q_loop_addi` — must not move | 32 | 48 | 84 | 156 | 300 | 588 | 1164 |
| `q_loop_addi` — **read** | 28 | 48 | 84 | 156 | 300 | 588 | 1164 |
| `q_loop_adddmareg` — was | 31 | 48 | 107 | 320 | 704 | 1472 | 3008 |
| `q_loop_adddmareg` — **predicted** | 31 | 48 | 107 | **299** | **683** | **1451** | **2987** |
| `q_loop_adddmareg` — **read** | 28 | 48 | 107 | **299** | **683** | **1451** | **2987** |
| `q_loop_adddmareg_sync` — must not move, and did not | 63 | 111 | 207 | 399 | 783 | 1551 | 3087 |

i.e. a **21-cycle fall at n ≥ 128 and nowhere else**, exactly as predicted,
which puts phase Q's run-ahead at 32.3 against phase S's 32.3 — the two forms
**0.0 entries apart** under the sync-free estimator and `RECONCILED: 0.7` under
the levelled one.

**The one thing that moved and should not have**: `q_loop_addi` and
`q_loop_adddmareg` each came in 3–4 cycles low at n = 16 and identical to the
cycle everywhere else. That is inside the 17-cycle `q_ctrl` spread for one raw
point and the two banked runs already disagree by a cycle there, but n = 16 is
the backlog's reference point, so it moved the levelled reading by one entry —
it is reported in `docs/bh_arch.md` §1.10 and scored in the plan doc rather than
absorbed.

**What would have refuted it:** `q_loop_adddmareg` not falling by 21 ± 14 cycles
at n ≥ 128; or `q_loop_addi` or `q_loop_adddmareg_sync` moving, neither of which
is edited and either of which would have meant the change perturbed instruction
placement rather than the queue. None fired, and nothing in the sweep was
adjusted to make the numbers meet.

`q_loop_addi` and `q_loop_adddmareg` in the two tracked full-run datasets are a
record of the *pre-drain* binary — superseded for the depth, still the "before"
arm of this comparison, and still surrounded by 1500-odd rows that nothing here
touches. Their headers say exactly that.

**What this run does NOT say.** It is `--variants t1`. It produces **no**
shared-versus-per-thread verdict — that is a ratio between thread counts and
comes from runs 3 and 4 — and it carries no phase R, so the live-instrument
check cannot run against it.

---

## Analysing the results

Back on a machine with this repo:

```bash
export PYTHONPATH=/path/to/tt-sim
python3 -m tt_sim.perf.riscv_bench_sweep --measured riscvbench-blackhole.csv

# With no --measured it sweeps the tracked 2026-08-05 Blackhole run instead,
# which is how to see what a good run looks like before comparing your own.
python3 -m tt_sim.perf.riscv_bench_sweep
```

and, to diff hardware against the simulator running the same binary:

```bash
python3 -m tt_sim.perf.riscv_bench_sweep \
    --measured riscvbench-blackhole.csv \
    --reference sim-blackhole.csv
```

The sweep prints its expectations before any number, declares its exclusion
criteria before any residual, reports a per-series resolution below which a
negative residual is the instrument rather than a finding, and breaks the
residuals down by phase, unit, bound, prediction kind and whether tt-sim charges
the term at all.

## Running it against the simulator instead

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh riscvbench -- --blocks 2
```

`TT_SIM_ARCH=wormhole` picks the other simulator; `TT_SIM_COST_MODEL=1` turns
the cycle cost model on — and for this benchmark that flag is the whole point,
because with it **off** tt-sim charges nothing anywhere and every probe reads
exactly 1.000, including the four that are supposed to be the instrument's own
control. Keep `--blocks` small: the simulator runs a few tens of thousands of
cycles per second.

A full run is twenty-one program launches (seven phases × three thread sets) and
takes about twenty minutes against the simulator.
`--phase r --variants t1` is the cheap check that the plumbing works, and
`--phase rt` is the pair that matters. The two newest phases are cheap:
`--phase s --variants t1,t2` is about a minute and `--phase g --variants t1
--blocks 1` a couple.

Against tt-sim the phase T, C, Q, F, S and G verdicts are **forced** and mean
nothing about any hardware: tt-sim has no instruction cache, no branch
predictor, and an unbounded Tensix instruction queue, so a null in each of those
is guaranteed by its own construction. Specifically, and verified:

- **Phase G** reads 1.001 cycles/instruction at 1024, 1152, 1280, 1408, 1536 and
  1792 alike, step `-0.000` in every `--gset`. Nothing models an instruction cache, so a
  flat row is the only row available.
- **Phase S** refuses a depth in every slot — the backlog is still growing at
  n = 512 because `TensixFrontend.push_mop_instruction` is a list append — and
  therefore prints **NO VERDICT** on the sharing question rather than a
  plausible-looking "per-thread". A shared-versus-private answer from tt-sim
  would be a bug in the read-out, not a finding.

The "Did the new probes run at all?" section exists for exactly this: a probe
that reads its forced null and a probe that never executed look identical in a
summary table, and it names which is which before any verdict is read. Phase Q at `--phase q --variants t1` is the cheapest
check that the n = 1024 extension is plumbed (one launch, about a minute): the
simulator gives `NO KNEE up to n=1024` with the backlog doubling at every
doubling of the burst — 690, 1426, 2898 cycles — which is the shape an infinite
queue has to have, and `q_loop_addi` reads exactly the same 1.125 cycles per
instruction as the `ADDDMAREG` burst does, i.e. nothing back-pressures anything.
The point of running it is that the plumbing is exercised, not that the answer
means anything. What the simulator run establishes is that the plumbing
works end to end — the kernel builds, the probes run, the CSV and the sweep
agree — and, with `TT_SIM_COST_MODEL=1`, that the instrument resolves a real
per-instruction cost when there is one to resolve. The sweep says so in its own
verdicts.
