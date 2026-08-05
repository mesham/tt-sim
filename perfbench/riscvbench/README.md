# riscvbench — running it on real hardware

**You have a Tenstorrent card. This page is everything you need; you do not need
to know anything about tt-sim.** It asks you to build one program, run it twice,
and send back two CSV files. Budget **20 minutes**, most of it the build.

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
finishes by asking three questions nothing has ever measured: how deep the
Tensix instruction queue is, whether a taken branch costs more than a
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

# Run 1 -- the main one. All five phases, one/two/three issuing TRISCs.
./build/riscvbench --blocks 32

# Run 2 -- the same thing at a different burst size. Two minutes, and it is
# worth it: see "Why the second run".
./build/riscvbench --blocks 8 --out riscvbench-$(uname -n)-blocks8.csv
```

`--blocks 32` is the **hardware setting** — bigger than the default of 4, which
is sized for the simulator. Larger bursts push the fixed costs further into the
noise; on silicon they cost milliseconds.

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
- `--phase r` / `t` / `c` / `q` / `f` runs one phase, and `--variants t1` runs
  only the single-thread launches. Between them any phase can be skipped.
- `--probes 0xMASK` enables probe `i` with bit `i`; the summary table prints
  the probes in slot order. **Probe 0 is the empty-loop control and every
  slope phase needs it** — clearing bit 0 makes those phases unreadable.

Phase F builds a kernel with 8 KiB of instruction text in one loop body. If your
tt-metal refuses to build or place it, run `--phase rtcq` and say so; the other
four phases are unaffected.

---

## What to send back

**Both CSVs, plus the terminal output of each run if you have it** — the summary
tables, the per-phase read-outs and the validity verdicts are in the terminal
output and not in the CSV.

Each CSV has one row per raw measurement and no derived numbers at all:

```
phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles
```

with the run's configuration in a `#` header line (`arch=`, `probe_mask=`,
`phases=`, `variants=`, `base_blocks=`, `stack_addr=`, `scratch_addr=`,
`div_dividend=`, `div_divisor=`). `stack_addr=` matters more than it looks:
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
TTRVBENCH_VALID: yes
Completed successfully on the device
```

A failing phase names its failing checks, and the exit status is a bit mask
(1 = R, 2 = T, 4 = C, 8 = Q, 16 = F, 0 = clean).

**Send the CSV either way.** The verdict is per phase precisely because the five
phases measure unrelated things: a phase F that measures nothing does not make
the phase T rows in the same file any less good.

The checks are:

- **Linearity.** Every slope probe is fitted against its four burst lengths and
  must reach R² ≥ 0.99. A probe that does not is marked `<-- NONLINEAR`. This
  catches a counter that wrapped, a burst short enough to be swamped by its own
  start-up, or a thread that was descheduled mid-measurement.
- **Monotonicity.** More instructions must take more cycles.

**Phase Q is deliberately not gated on linearity**, and that is not laziness. It
sweeps *burst length* looking for a knee; a straight line through it would be
the null result, not the healthy one, so requiring one would score the
interesting outcome as a failure. Its own control probe `q_ctrl` is also exempt
from the monotonicity check, because its cost is a step function of the burst
*index* rather than of the burst length.

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

**2. The phase Q knee.** Bursts of 1, 2, 4 … 128 `ADDDMAREG`s. Below the Tensix
instruction queue's depth the core runs ahead and each extra instruction costs
one cycle; above it the core is back-pressured and each costs the unit's
occupancy, which `tensixbench` measured at 3.0 on Blackhole. **The burst length
at which the marginal cost steps up is the queue depth, and nothing in either
the ISA documentation or the vendor trees publishes one.** A flat 1.0 all the
way to 128 is also a result: it means a kernel can queue at least 128 Tensix
instructions before it is ever slowed down.

**3. `taken - not taken`, in the phase C read-out.** Two probes execute the
identical dynamic instruction sequence — a branch whose target is the address
the not-taken path falls through to — and differ in exactly one bit. The tables
record a mispredict as a 2-cycle bubble on Wormhole and 4 on Blackhole, which is
how much one *costs*; how *often* one happens is undescribed, which is why
tt-sim charges nothing for branches at all. A non-zero delta here would supply
the missing half.

**4. The phase F row.** Six loop bodies from 256 bytes to 8 KiB of instruction
text, same instruction throughout. A flat row says instruction fetch is not the
limit anywhere in that range. A cliff locates an instruction-cache capacity
nothing publishes.

---

## Analysing the results

Back on a machine with this repo:

```bash
export PYTHONPATH=/path/to/tt-sim
python3 -m tt_sim.perf.riscv_bench_sweep --measured riscvbench-blackhole.csv
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

A full run is fifteen program launches (five phases × three thread sets) and
takes about fifteen minutes against the simulator. `--phase r --variants t1` is
the cheap check that the plumbing works, and `--phase rt` is the pair that
matters.

Against tt-sim the phase T, C, Q and F verdicts are **forced** and mean nothing
about any hardware: tt-sim has no instruction cache, no branch predictor, and an
unbounded Tensix instruction queue, so a null in each of those is guaranteed by
its own construction. What the simulator run establishes is that the plumbing
works end to end — the kernel builds, the probes run, the CSV and the sweep
agree — and, with `TT_SIM_COST_MODEL=1`, that the instrument resolves a real
per-instruction cost when there is one to resolve. The sweep says so in its own
verdicts.
