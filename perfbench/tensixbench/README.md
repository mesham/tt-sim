# tensixbench — running it on real hardware

**You have a Tenstorrent card. This page is everything you need; you do not need
to know anything about tt-sim.** It asks you to build one program, run it a
handful of times, and send back the CSV files. Budget **20 minutes**, most of it
the build.

**`./run_card.sh` does all of it** — build, the six documented runs, the
analysis, and a list of what to check before sending. The rest of this page is
what it does and why, and is worth reading once. If you are running the whole
roadmap block rather than this one program, use
[`../run_card_session.sh`](../README.md) instead.

If you want to know *why* the benchmark is shaped the way it is, and what its
numbers can and cannot prove, read
[`docs/plans/tensix-cost-benchmark.md`](../../docs/plans/tensix-cost-benchmark.md)
instead. This page is the runbook.

---

## What it does, in three sentences

It times short bursts of individual Tensix coprocessor instructions —
`SFPADD`, `ADDDMAREG`, `MVMUL` and a dozen others — using the device's own cycle
counter, the same register tt-metal's profiler reads. It runs each burst at four
different lengths and reports the **slope**, so kernel launch, timer overhead
and loop setup cancel out exactly and never appear in a result. It then repeats
the whole thing with one, two and three Tensix threads issuing at once, and
finishes by timing a `matmul_tiles` loop at three math fidelities.

It is a **normal tt-metal program**. It creates a device, launches a program on
one core, reads a buffer back and exits. It does not need Tracy, the device
profiler, `tt-exalens`, or a board reset.

The two halves are checked and reported **separately** — one can be perfect
while the other measures nothing, which is what happened last time — so nothing
you send is thrown away because of a phase you did not care about.

---

## What you need

| | |
| --- | --- |
| Hardware | One Wormhole **or** Blackhole card. The program detects which and adapts; you do not choose. |
| tt-metal | A built checkout that exports the `TT-Metalium` CMake package, i.e. `<tt-metal>/build/lib/cmake/tt-metalium/tt-metalium-config.cmake` exists. |
| Tools | `cmake` ≥ 3.22 and a C++ compiler. tt-metal supplies every flag and include path. |
| Time | ~15 min to build, ~1 min per run, six runs. |

The program writes to one core (logical `(0,0)`) and one small L1 buffer. It
does not touch DRAM, does not allocate large buffers, and cannot corrupt
anything that outlives the process.

---

## Build

```bash
export TT_METAL_HOME=/path/to/tt-metal          # your built checkout
export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"

cd perfbench/tensixbench/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# -> build/tensixbench
```

## Run

Run it **from `src/`** — the program refers to its kernels by the relative path
`kernels/...`.

```bash
cd perfbench/tensixbench/src
export TT_METAL_SLOW_DISPATCH_MODE=1     # see note below
unset TT_METAL_SIMULATOR                 # make sure you are on the real card

# Run 1 -- the main one.
./build/tensixbench --blocks 32 --iters 64

# Run 2 -- the A/B for the matrix-unit question. Same binary, one flag.
./build/tensixbench --blocks 32 --iters 64 --dvalid-per-thread --phase a

# Runs 3-6 -- the source data format sweep (experiment X2). Four separate
# invocations, because the format is programmed once per run, not per probe.
# Each writes its own file. Seconds each. No board reset between them.
#
# `--variants t1` runs the single-thread launch ONLY. X2 is a single-thread
# comparison across formats, so this costs the experiment nothing; it is no
# longer a workaround for anything (see "If it hangs" below).
for f in bf16 fp32 tf32 fp16; do
  ./build/tensixbench --blocks 32 --phase a --variants t1 --dvalid-unpacr-nop --src-format $f
done
```

`--blocks 32` and `--iters 64` are the **hardware settings** — bigger than the
defaults, which are sized for the simulator. Larger bursts push the fixed costs
further into the noise; on silicon they cost milliseconds.

Every run writes its **own file** next to the binary, named after its
configuration, so none of them can overwrite another:

```
tensixbench-<arch>.csv                    run 1
tensixbench-<arch>-dvalid-per-thread.csv  run 2
tensixbench-<arch>-unpacr-nop-bf16.csv    runs 3-6, one per format
tensixbench-<arch>-unpacr-nop-fp32.csv
tensixbench-<arch>-unpacr-nop-tf32.csv
tensixbench-<arch>-unpacr-nop-fp16.csv
```

Each prints its own summary table.

### Why the second run

The previous hardware run measured `MVMUL`, `ELWADD` and `ELWMUL` at **6.1× and
12.1× per-thread cost** at two and three Tensix threads, where every other unit
came out at a flat 2×/3×. Three threads together retired one matrix instruction
per ~4 cycles; one thread alone retired one per cycle. If that is real it
contradicts the ISA documentation, which says the matrix unit accepts one
instruction per cycle *"regardless of how many threads are trying to use it"*.

It is probably not real. Those three probes need the SrcA/SrcB data-valid bits,
and the old kernel set them **inside `kernel_main`** — which every thread runs —
so the number of `SETDVALID`s executed was exactly the thread count. The thread
count and the Src bank state moved together and nothing in the data separates
them. Worse, on Blackhole `SETDVALID` is `UnsupportedFunctionality`, and the
third thread's copy hands the matrix unit a bank it already owns.

Run 1 now issues **exactly one** `SETDVALID`, from one thread, with a barrier
after it, so only the thread count varies. Run 2 (`--dvalid-per-thread`)
reproduces the old setup exactly. Comparing the `MVMUL`/`ELWADD`/`ELWMUL` rows
of the two settles it:

- **`t2` ≈ 2 and `t3` ≈ 3 in run 1** → the collapse was an artefact of the
  benchmark and is retracted. This is the expected outcome.
- **`t2` ≈ 6 and `t3` ≈ 12 in run 1 as well** → it survives the obvious
  confound, the effect is a genuine property of concurrent matrix-unit use, and
  it becomes worth chasing.
- **One moves and the other does not** → the effect tracks the number of valid
  Src banks, not the thread count. Still an artefact, and a sharper one.

Run 2 only needs `--phase a`; phase B has nothing to do with this.
The reasoning is `docs/plans/matrix-unit-thread-contention.md`, experiment X1.

### Why runs 3–6 — the source data format

**This one is exploratory. It is not confirming anything, and a boring result is
a real result.** Say so up front because it changes how the numbers read.

Runs 1 and 2 both set the SrcA/SrcB valid bits with a bare `SETDVALID`. On
Blackhole that instruction is `UnsupportedFunctionality` and its own page says it
leaves `ImpliedSrc{A,B}Fmt` an `UnpredictableValue()` — and *that field is what
Blackhole's matrix unit reads for the source data format*, since no Blackhole LLK
ever sets `DISABLE_IMPLIED_SRC{A,B}_FMT_Base`. So in runs 1 and 2 the format the
FPU decoded is simply undefined. Not wrong, not bf16: undefined.

`--dvalid-unpacr-nop` swaps it for the sanctioned replacement, `UNPACR_NOP`
carrying `set_dvalid` (`UNPACR_NOP_SETDVALID.md`), issued once per unpacker from
one thread with a barrier after. That instruction copies
`THCON_SEC{0,1}_REG2_Out_data_format` into the bank it hands over, so the format
becomes **defined** — and, being config, becomes something a run can *choose*.
`--src-format` chooses it. Getting a defined format is the precondition; the
sweep is what it buys.

**What the ISA documentation predicts.** `MatrixUnit.md` gives `MVMUL`,
`ELWMUL` and `ELWADD` one instruction per cycle with **no format qualification
at all**. And the functional models collapse every format code to one of three
`SrcAStyle`s:

| source format | decoded `SrcAStyle` |
|---|---|
| **bf16**, **fp32**, bfp8/4/2, int32, int16 | `BF16` |
| **fp16**, fp8, bfp8a/4a/2a, int8 | `FP16` |
| **tf32** | `TF32` |

So the prediction has two strengths, and they are worth keeping apart:

- **bf16 vs fp32 is predicted to be *exactly* nothing.** They are the same
  branch of the same decode. A difference there contradicts the functional
  model outright — that is the strongest thing this sweep can find, and it is
  why fp32 is in the list even though `SrcA`/`SrcB` cannot hold FP32 (19 bits
  is TF32 at most).
- **tf32 and fp16 genuinely change the datapath**, and no document gives that a
  cost. There is nothing to confirm or refute, only a number nothing has ever
  measured.

**What each outcome would mean for the cost tables**
(`tt_sim/pe/tensix/tensix_instruction_costs.yaml`, whose MATH entries carry no
format axis today, and `docs/plans/tensix-cost-benchmark.md`, which lists data
format under "what is not measured, and why"):

| result | what it means |
|---|---|
| **All four formats identical** (expected) | The MATH occupancies are right to have no format axis, and that becomes a *measured* statement instead of an unexamined omission. The benchmark's "data format is unmeasured" limitation is closed. Nothing in the tables changes. |
| **Costs differ between `SrcAStyle`s** (e.g. tf32 ≠ bf16) but bf16 = fp32 | The tables need a **format axis on the MATH occupancies**, alongside the existing `scales_with: fidelity_phases` — the first well-founded new axis this benchmark has motivated. The docs are silent rather than wrong. |
| **bf16 ≠ fp32** | The functional models are wrong about the decode, not just silent about the cost. That is a documentation finding, and it goes upstream before it goes into a table. |

**One caveat that belongs with the numbers, not in a footnote.** Phase A issues
each op as an individual `.ttinsn` word, so every figure here is the
**Wait-Gate-bound** regime (~6 cycles/instruction on Blackhole), not the
MOP-issued ~1 cycle the tables actually charge. A format effect visible here is
evidence about the operand decode; it is not directly a measurement of the
quantity in the table. See "Two regimes for one instruction" in
`docs/plans/tensix-cost-benchmark.md`.

`--src-format` **requires** `--dvalid-unpacr-nop` and the program refuses the
combination otherwise, on purpose: a CSV labelled `src_format=fp32` whose FPU
actually decoded an unpredictable value would be worse than no CSV.

> **`TT_METAL_SLOW_DISPATCH_MODE=1`** makes `LaunchProgram` the launch path.
> The benchmark works either way on hardware; setting it keeps the hardware run
> and the simulator run on the same path, which is the entire point of the
> exercise. If your tt-metal build refuses slow dispatch, run without it and
> **say so when you send the results** — it is a difference worth recording, not
> a reason to abandon the run.

### If it hangs

**Fixed: `--dvalid-unpacr-nop` used to poison the card, and no longer should.
No `tt-smi -r 0` is needed between any of the runs above.**

The old failure was worth understanding, because it is why the fix looks the way
it does. `UNPACR_NOP` with `UNP_ZEROSRC` is the *acquire* half of a handshake: it
waits until the bank `MatrixUnit.SrcABank` points at is no longer valid before it
may zero and re-hand it over. This benchmark holds the valid bits for the whole
burst **by design** (`clear_dvalid = 0` on every probe is what makes the burst
measurable), so nothing performed the *release* half and a **successful** run
left the banks owned by the Matrix Unit. The next execution of the setup then
waited for ever — the `t2` launch in the same process, or the first launch of the
next process on the same card. Two experiments on a card confirmed it: `t2` alone
on a reset card completed, and `t1` twice with no reset in between hung the
second time. Thread count was never the variable; leftover state was.

The kernel now issues the counterpart — one `CLEARDVALID` from one thread, after
the last probe has written its last result word, outside every timed region — so
a completed run hands the banks back and the next run finds them free. Phase A's
numbers are unchanged: the before/after data rows are byte-identical.
`--dvalid-once` and `--dvalid-per-thread` are untouched, because `SETDVALID` has
no wait half and never wedged.

**The one-line check, if you want to confirm the card is clean.** Run the same
single-thread command twice with **no reset between**. Before the fix the second
invocation hung; now both must print `TTBENCH_VALID_A: yes` and exit 0:

```bash
./build/tensixbench --blocks 32 --phase a --variants t1 --dvalid-unpacr-nop --src-format bf16 && \
./build/tensixbench --blocks 32 --phase a --variants t1 --dvalid-unpacr-nop --src-format bf16
echo "exit=$?"    # 0 means the card was left clean by the first run
```

`--variants t1` is still what X2 wants — it is a single-thread comparison across
formats, and the analysis (`tensix_bench_sweep --formats`) drops every
multi-thread series before it computes anything — but it is no longer avoiding
anything. If you have the time, `--variants t1,t2,t3` on this setup is now worth
one run purely as a check that the multi-thread launches survive too; **it has
never been run on silicon with the release in place**, so if it hangs, that is a
result and we want it.

The other probes and the general case:

Three probes (`MVMUL`, `ELWADD`, `ELWMUL`) need the matrix unit's SrcA/SrcB
data-valid bits, which the benchmark sets with a bare `SETDVALID` in runs 1–2
and with `UNPACR_NOP` in runs 3–6. That is the one thing here that depends on
Tensix state rather than being self-contained, and it is the likeliest place for
a hang. If the program hangs, kill it, **note which run it was**, and run:

```bash
./build/tensixbench --blocks 32 --iters 64 --no-dvalid-probes
```

**and tell us it hung.** A hang there is itself a result. Everything else still
gets measured.

To bisect further, `--probes 0xMASK` enables probe `i` with bit `i`; the summary
table prints the probes in slot order. `--phase a` or `--phase b` runs one half.

---

## What to send back

**All six CSVs.** Plus the terminal output of each if you have it, because the
summary tables and the validity verdicts are in there and not in the CSV.

```
tensixbench-<arch>.csv
tensixbench-<arch>-dvalid-per-thread.csv
tensixbench-<arch>-unpacr-nop-bf16.csv
tensixbench-<arch>-unpacr-nop-fp32.csv
tensixbench-<arch>-unpacr-nop-tf32.csv
tensixbench-<arch>-unpacr-nop-fp16.csv
```

Each has one row per raw measurement and no derived numbers at all:

```
phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles
```

with the run's configuration in a `#` header line (`arch=`, `probe_mask=`,
`dvalid_setup=`, `src_format=`, `src_style=`, `mm_block=`, `variants=`,
`fidelities=`), which is how the runs tell themselves apart. `variants=` is what
distinguishes a deliberately single-thread run from a full run that stopped
early.

If you ran with `--no-dvalid-probes`, or without slow dispatch, or the program
exited non-zero — say which. All three change how the numbers are read.

**If runs 3–6 hang or fail and runs 1–2 do not, say so and send runs 1–2
anyway.** The `UNPACR_NOP` setup has already produced exactly one hardware-only
failure mode this way — it left the card wedged for the *next* run — and its
release is the newest code here. Only the simulator has exercised that release,
and nothing there back-pressures the issuing core, so a second hardware-only
failure is entirely possible. That is a result about the setup, not a wasted run.

---

## How to tell a good run from a silently bad one

The program checks itself and prints a verdict **per phase**:

```
TTBENCH_VALID_A: yes
TTBENCH_VALID_B: yes
TTBENCH_VALID: yes
Completed successfully on the device
```

A failing phase names its failing checks, and the exit status is a bit mask
(1 = phase A failed, 2 = phase B failed, 3 = both, 0 = clean).

**Send the CSV either way.** The verdict is per phase precisely because the two
phases measure unrelated things: a phase B that measures nothing does not make
the phase A rows in the same file any less good, and an earlier run was thrown
away for exactly that reason. `TTBENCH_VALID_A: no` is the one that means the
main result is unusable.

The checks are:

- **Linearity.** Every probe is fitted against its four burst lengths and must
  reach R² ≥ 0.99. A probe that does not is marked `<-- NONLINEAR`. This is the
  one that catches a counter that wrapped, a burst short enough to be swamped by
  its own start-up, or a thread that was descheduled mid-measurement.
- **Monotonicity.** More instructions must take more cycles. A violation means
  the timestamps are not what we think they are.

Two further sanity checks you can make by eye, which the program does *not*
enforce because a violation is a finding rather than a fault:

1. **`loop_overhead` should be small and positive** — a few cycles per block.
   It is the empty control loop. If it reads a flat constant regardless of
   burst length, the compiler deleted the loop and every other number is
   inflated by whatever the loop really cost.
2. **`cyc/instr` should be ≥ 1.00 for everything.** A baby RISC-V core issues at
   most one instruction per cycle, so nothing can legitimately come out below
   1.0. A value like 0.4 means the timer, not the instruction, is being
   measured.

A run where every single probe reads exactly `1.000` is *possible* and is itself
informative (it means nothing back-pressures the issuing core) — but on silicon
we expect `ADDDMAREG`, `MULDMAREG`, `SHIFTDMAREG` and `CMPDMAREG` to come out
near **3** because that is what the ISA documentation says. If they all read 1.0
on hardware, that is the most interesting possible result and we want to hear
about it immediately.

`RDCFG` is the one to be careful with, and slot 14 is not the probe that answers
it **on Blackhole**. The ISA doc's `>= 2` for `RDCFG` is a **latency**; every
probe in phase A measures **occupancy**, and `BlackholeA0/.../RDCFG.md` says
"The issuing thread is not blocked, so it can potentially start its next
instructions (of any kind) during `RDCFG`'s subsequent cycles" — so slot 14
reading exactly 1.000 on Blackhole silicon is not a contradiction, and charging
the doc's 2 as an occupancy is what made tt-sim's `matmulblock` guard compute
the wrong answer.

**On Wormhole it is the other way round.** `WormholeB0/.../RDCFG.md` says "The
issuing thread is blocked for the entire duration", so there the `>= 2` *is* an
occupancy and slot 14 reaches it with no extra machinery. There is no Wormhole
`tensixbench` dataset yet; taking one closes this on that part.

### Reaching `RDCFG`'s `>= 2`: what it is, and what it is not

```
# the measurement
./build/tensixbench --phase a --blocks 32 --iters 64 --variants t1 \
                    --probes 0x3FF04601 --vis-reps 64
# the C12 liveness control, in its own run (it needs a t3 launch)
./build/tensixbench --phase a --blocks 32 --iters 64 --variants t1,t3 \
                    --probes 0x30000001
```

**`>= 2` IS A LATENCY, AND THE DOCUMENTS SAY SO TWICE.**
`BlackholeA0/.../ConfigurationUnit.md` tabulates the Configuration Unit's
instructions under columns headed **Latency** and **IPC**, and gives `RDCFG`
"≥ 2 cycles" at **IPC 1**. `RDCFG.md` says it in prose:

> This instruction requires at least two cycles to execute, and then additional
> cycles if there is contention for GPR writes. Assuming no contention, it is
> fully pipelined, so an `RDCFG` instruction can be started every cycle. The
> issuing thread is not blocked, so it can potentially start its next
> instructions (of any kind) during `RDCFG`'s subsequent cycles.

So the quantity is a **latency to the destination GPR**, the throughput is one
per cycle, and a Blackhole card measured exactly that: **0.998 cycles/instr**
bare (slot 14). Nothing about the `>= 2` is visible in issue cost.

**AND IT IS NOT VISIBLE TO A BUSY-CONDITION EITHER.** Slots 22–25 stall on C12,
*"Any thread has an instruction in any stage of the Configuration Unit
pipeline"*, which is the only condition on either architecture that names the
unit — and a card ran them on 2026-08-12:

```
TTBENCH_CFGLAT_COND:  C12 CFGEXU 0x1000
TTBENCH_CFGLAT_OCC:   0.9978 0.9979 0.9979
TTBENCH_CFGLAT_PAIRS: 2.9682 2.9682 2.9683
TTBENCH_CFGLAT_DIFF:  0.0000 -0.0001
```

All three arms identical to four decimal places. `RDCFG` leaves at most one
cycle of post-issue residency (stage +1, the GPR write), and `STALLWAIT`'s own
floor is at least that wide — *"There is a one cycle lag between the
condition(s) being met and the block mask being removed"* — so the quantity
completes inside the measuring apparatus.

**WHERE IT IS VISIBLE.** `RDCFG.md`'s "Instruction scheduling" section:

> Software must ensure that the instruction(s) immediately after `RDCFG` are not
> trying to consume the GPR written by the `RDCFG` instruction. In *most* cases,
> this applies to the one instruction after `RDCFG`, but it can apply to more
> than one instruction if there is contention for the GPR write.

An obligation on **software** is the documented absence of an interlock: a
consumer placed too close does not wait, it reads the **stale** value. The
latency is therefore a **distance**, not a duration, and that is what the
benchmark measures.

#### The visibility sweep (`--vis-reps N`)

Not a probe slot and not a slope — it carries no cycle count, so there is no
launch or timer overhead for an intercept to absorb. For each separation
`d = 1..4`, `N` times:

```
regfile[60] = seed              a RISC-V write, read back to order it
regfile[63] = 0xDEADBEEF
tensix_sync()
TTI_RDCFG(60, 0)
TTI_NOP  x (d - 1)              one `.ttinsn`, one issue slot, one cycle each
TTI_ADDDMAREG(1, 63, 0, 60)     the consumer: GPR63 = GPR60 + 0
tensix_sync()
observe regfile[63]
```

and the observation is one of exactly four things: the seed (**stale**), the
value `RDCFG` read (**fresh**), `0xDEADBEEF` (the consumer never ran), or
something unexplained. The reading, `TTBENCH_VIS_DMIN`, is the smallest `d` at
which **every** repetition is fresh.

It is a **lower bound**: if `ADDDMAREG` reads its operand some cycles into its
own execution, the true latency is larger. That is the direction the charging
policy takes bounds in, and `d_min = 2` corroborates the documented `>= 2`
exactly while `d_min = 1` leaves it *unreached* rather than refuted.

The sequence must be literal `.ttinsn` immediates — the runtime `TT_*` forms
compute their instruction word first and would insert RISC-V cycles into the gap
being measured — so `d` is a template parameter and the fillers are emitted by
recursion.

**The controls, and every one of them can only pass by MOVING:**

| control | passes only if | what it rules out |
| --- | --- | --- |
| no `RDCFG`, two different seeds | it reads back **two different** values, its own seeds | a readout stuck on a constant; a seed that never landed. Proves a *stale* observation is representable |
| `RDCFG` at `d = 8`, two different seeds | it reads back **one** value twice, and it is neither seed nor the marker | a readout that echoes its input; a result that never reaches the GPR. Proves a *fresh* observation is representable |
| the marker count | zero | a consumer that never ran |
| per-`d` counts | 0 or `N`, never in between | a mixture, which means the RISC-V front end did not deliver the sequence at one instruction per cycle and the separation is not what it says |

**It is free of the C12 problem.** It issues no `STALLWAIT` and consults no
condition bit, so it is untouched by tt-sim reading Blackhole's condition mask as
12 bits where the ISA page gives 13. It also runs on either architecture.

#### The dependence pair, slots 26/27

The house method — `riscvbench` measures "dependent multiply" at 1.985 cycles
this way — transplanted to Tensix GPRs. Both arms are the same two opcodes and
differ in one operand field:

| slot | body | reads |
| --- | --- | --- |
| 26 | `TTI_RDCFG(60,0)` ; `TTI_ADDDMAREG(1,63,0,60)` | the `RDCFG` destination |
| 27 | `TTI_RDCFG(60,0)` ; `TTI_ADDDMAREG(1,63,0,59)` | a GPR `RDCFG` never wrote |

so issue cost, unit occupancy and loop overhead all cancel and what is left is
the read-after-write. **The prediction is zero**, from "Software must ensure",
and the arm exists because it is falsifiable: if Blackhole *does* interlock,
`26 − 27` is the latency in cycles directly and is the better measurement. Its
own control is that the pair must cost probe 14 + probe 10, both measured in the
same run — under that sum, an arm is not issuing what it claims and a null
difference would mean nothing.

#### The C12 liveness control, slots 28/29

Two explanations survive the 0.0000 above, and the visibility sweep cannot
choose between them because it never consults a condition bit:

1. C12 works, and `RDCFG`'s one cycle of stage-+1 residency is no wider than the
   stall's own documented lag — structurally invisible, not absent;
2. C12 does not behave as documented on this part.

Only a C12 signal much **wider** than one cycle separates them, and there is one
way the documents support. C12 is *"ANY thread"*, and its note says *"This won't
prevent other threads from issuing new Configuration Unit instructions though,
and those new instructions will cause this thread to continue to wait."* So the
busy-ness has to come from another thread, where it is not coupled to the
waiting thread's own issue slots. These are the only thread-dependent bodies in
the benchmark:

| slot | thread 1 (graded) | threads 0 and 2 |
| --- | --- | --- |
| 28 | `STALLWAIT(STALL_THREAD, CFGEXU)` ; `NOP` | `RMWCIB0(0,0,0)` ; `NOP` |
| 29 | `NOP` ; `NOP` | `RMWCIB0(0,0,0)` ; `NOP` |

`d(v) = pair(28, v) − pair(29, v)`, and the reading is `d(t3) − d(t1)`. Slot 29
carries the cross-thread issue interference that grows with the thread count
anyway, and subtracts it. A large positive difference says C12 is live and
reading 1 holds; a zero says C12 is inert and slots 22–25 say nothing about
`RDCFG` at all. Both are reachable.

It runs **separately** because it needs a t3 launch, t3 series are contended by
construction, and phase A's validity gate is per-PHASE: one nonlinear contended
series would flip `TTBENCH_VALID_A` for the visibility measurement too. It
cannot hang — the hammering threads issue a bounded burst and then wait at the
barrier, after which the unit drains.

#### The two documented negatives, slots 20–25

Kept, never renumbered, and still printed:

| slots | condition | reading | why kept |
| --- | --- | --- | --- |
| 20/21 | `TRISC_CFG` (C10 on Blackhole) | 0.0000 on a card, 2026-08-09 | that bit is about the RISCV core's outstanding *memory requests*, not the unit's pipeline. The falsification control for 22–25 |
| 22/23/24/25 | `CFGEXU` (C12) | 0.0000 on a card, 2026-08-12 | the right condition, and the evidence that a busy-condition cannot see a latency |

Slot 24 − slot 9 is still the movement control for the stall itself (`~1.97
cycles/pair` on a card), and slot 25 next to slot 14 still shows the two config
ops cost the same to issue. Both config ops stay state-free: `RDCFG` only reads,
and `RMWCIB0` with `Mask = 0` is the identity by its own functional model
(`*CfgAddress = (NewValue & Mask) | (OldValue & ~Mask)`).

To reproduce a run comparable with the tracked datasets in
`tt_sim/perf/datasets/`, pass `--probes 0xFFFFF`: those were collected before
slots 20–29 existed, and all ten default ON.

---

## Reading the summary table

```
probe          variant unit   thr     cyc/block  cyc/instr      R^2
ADDDMAREG      t1      THCON  1          194.00      3.000   1.0000
```

- **`cyc/block`** — the fitted slope: cycles per 64-instruction block.
- **`cyc/instr`** — `(cyc/block − loop_overhead) / 64`. This is the number that
  matters. It is a *throughput* (issue occupancy), not a latency.
- **`variant`** — `t1`/`t2`/`t3`: how many Tensix threads issued the identical
  burst simultaneously. If a unit really is one-instruction-at-a-time and
  shared, `t3` should be about 3× `t1`. If `t3` equals `t1`, the unit is not the
  constraint.
- **phase B rows** (`matmul_tiles`, variants `LoFi`/`HiFi2`/`HiFi4`) report
  cycles per `matmul_tiles` call, for the **unpack thread (0)** and the **math
  thread (1)** only — the pack thread has no per-iteration work in a matmul
  inner loop and is not timed. The **absolute** value is a composite and means
  little on its own; the **difference** between fidelities is the result, and
  the program now prints it directly:

  ```
  phase B: fidelity deltas (math thread), and what bounded the loop
    LoFi   -> HiFi2   measured   +16.02   predicted  +16.00   residual   +0.02
    HiFi2  -> HiFi4   measured   +31.94   predicted  +32.00   residual   -0.06

    fidelity   math (thr 1) unpack (thr 0)
    LoFi              21.03          19.88
  ```

  If the deltas come out near zero the program says so and explains what to
  look at; **that is not scored as a failure**, and it does not touch phase A's
  verdict. See below.

### If the fidelity deltas are zero

The first Blackhole run reported LoFi/HiFi2/HiFi4 within 0.2 cycles of each
other where +16 and +32 were predicted. Before re-running, the generated code
was disassembled, and the fidelity setting demonstrably *does* reach the math
thread: the unpack and pack ELFs are byte-identical across all three fidelities,
the math ELF differs in exactly the MOP inner-loop count (1 / 2 / 4) and the
fidelity-increment address mod, and the recorded MVMUL sequence it expands is
exactly sixteen instructions long. So a tile matmul really is 16 / 32 / 64
MVMULs at LoFi / HiFi2 / HiFi4, and a null delta cannot mean "the setting was
ignored".

What it can mean is that the loop was never limited by the math unit. Those
MVMULs are emitted by the Tensix MOP expander, not by the RISC-V core, so they
only cost the math thread wall-clock time if the coprocessor backs up into the
instruction FIFO. The old inner loop paid a circular-buffer wait and pop per
operand per matmul — about 81 cycles, more than even HiFi4's 64 MVMULs — so it
never did. The loop is now **blocked**: one wait and one pop per operand covers
eight matmuls, which is the only way the Tensix side can become the limit.

If the deltas are still zero after that, compare the two columns the program
prints. An unpack-thread slope at or above the math thread's, flat across
fidelities, means a bf16 tile matmul is **unpack-bound at every fidelity the
hardware offers** — a real result, and one worth reporting, but it leaves the
fidelity arithmetic untested rather than refuted.

---

## Optional, and worth it if you have an afternoon

tt-metal ships a much heavier per-LLK-op performance harness at
`tt_metal/tt-llk/tests/` that reads the Tensix **hardware performance counters**
directly (`MATH_INSTRN_STARTED`, `MATH_FIDELITY_STALL`, `UNPACK0_BUSY_*`, …) via
`ttexalens`. It measures LLK ops rather than individual instructions, needs
`tt-smi -r 0` board resets between slices, and **cannot be pointed at a
simulator** — which is why it is not what this benchmark is. But it commits no
results, so its output does not exist anywhere either. If you can run

```bash
cd tt_metal/tt-llk/tests && ./run_llk_perf_blackhole.sh 0 1   # or _wormhole
```

then `tt_metal/tt-llk/tests/python_tests/perf_data/*.csv` is a second,
independent dataset covering the composite ops this benchmark deliberately does
not, and `MATH_INSTRN_HF_1_CYCLE`/`_2_CYCLE`/`_4_CYCLE` is a hardware counter
for the exact fidelity-phase question phase B has to infer.

---

## Analysing the results

Back on a machine with this repo:

```bash
export PYTHONPATH=/path/to/tt-sim
python3 -m tt_sim.perf.tensix_bench_sweep --measured tensixbench-blackhole.csv
```

and, to diff hardware against the simulator running the same binary:

```bash
python3 -m tt_sim.perf.tensix_bench_sweep \
    --measured tensixbench-blackhole.csv \
    --reference sim-blackhole.csv
```

The X1 pair is the same tool twice — read the `MVMUL`, `ELWADD` and `ELWMUL`
rows of *"Issue-limit discriminator: the same burst from 1, 2 and 3 TRISCs"* in
each and put them side by side:

```bash
python3 -m tt_sim.perf.tensix_bench_sweep --measured tensixbench-blackhole.csv
python3 -m tt_sim.perf.tensix_bench_sweep --measured tensixbench-blackhole-dvalid-per-thread.csv
```

The X2 format sweep has its own mode, because the format is a per-run
configuration rather than a column and the comparison is therefore across files:

```bash
python3 -m tt_sim.perf.tensix_bench_sweep --formats \
    tensixbench-blackhole-unpacr-nop-bf16.csv \
    tensixbench-blackhole-unpacr-nop-fp32.csv \
    tensixbench-blackhole-unpacr-nop-tf32.csv \
    tensixbench-blackhole-unpacr-nop-fp16.csv
```

It prints its expectation before any number, refuses any run whose header does
not say `dvalid_setup=unpacr-nop` (such a run has no defined source format to be
a point on the axis), applies the ordinary exclusion ladder unchanged, and then
reports each pair against what the functional models predict — distinguishing
"two formats that share a `SrcAStyle` differ", which contradicts the model, from
"two formats that do not share one differ", which is merely undocumented.

## Running it against the simulator instead

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh tensixbench -- --blocks 2 --iters 4
```

`TT_SIM_ARCH=wormhole` picks the other simulator; `TT_SIM_COST_MODEL=1` turns
the cycle cost model on. Keep `--blocks` small: the simulator runs a few tens of
thousands of cycles per second.

Phase B is the slow half. `--fidelities LoFi` (or `HiFi2`, or `HiFi4`) runs one
fidelity per process, which is what makes it checkable at all against tt-sim:
any single fidelity at `--iters 1` finishes in a couple of minutes, and the
three are indistinguishable there anyway — the math thread issues one `MOP`
whatever the fidelity, and tt-sim's coprocessor never back-pressures the issuing
core, so all three slopes are identical by construction. On hardware always run
all three; the difference is the whole point and it needs at least two in one
CSV.

> Two phase B launches **in the same process** stall against tt-sim: LoFi
> completes in ~2 minutes and a second `LaunchProgram` then runs for 25+ minutes
> without finishing, while the same fidelity launched on its own finishes in
> ~2 minutes. Phase A does three launches per process and is unaffected. That is
> a simulator-side limitation, not a benchmark one, and it does not arise on
> hardware.

`--phase a --blocks 2` is the cheap half — about five minutes for all three
thread sets — and covers everything the X1 question needs.

The X2 format runs are cheaper still against the simulator if you narrow the
probe mask to the control plus the three MATH probes, which is all a format axis
can move:

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh tensixbench -- \
    --phase a --blocks 1 --probes 0xE0001 --dvalid-unpacr-nop --src-format bf16
```

Against tt-sim every format reads exactly `1.000`, and that is **forced** rather
than informative: nothing back-pressures the issuing core there, so no phase-A
probe of any unit at any format can read anything else. The simulator run proves
the plumbing — that the kernel builds, that the `UNPACR_NOP` word is accepted,
that the config write lands, that the CSV header and the sweep agree — and
nothing about the hardware. The sweep says so in its own verdict.
