# tensixbench — running it on real hardware

**You have a Tenstorrent card. This page is everything you need; you do not need
to know anything about tt-sim.** It asks you to build one program, run it
**twice**, and send back two CSV files. Budget **20 minutes**, most of it the
build.

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
| Time | ~15 min to build, ~1 min to run. |

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
```

`--blocks 32` and `--iters 64` are the **hardware settings** — bigger than the
defaults, which are sized for the simulator. Larger bursts push the fixed costs
further into the noise; on silicon they cost milliseconds.

The two runs write **two different files** — `tensixbench-<arch>.csv` and
`tensixbench-<arch>-dvalid-per-thread.csv` — next to the binary, so they cannot
overwrite each other. Each prints its own summary table.

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

> **`TT_METAL_SLOW_DISPATCH_MODE=1`** makes `LaunchProgram` the launch path.
> The benchmark works either way on hardware; setting it keeps the hardware run
> and the simulator run on the same path, which is the entire point of the
> exercise. If your tt-metal build refuses slow dispatch, run without it and
> **say so when you send the results** — it is a difference worth recording, not
> a reason to abandon the run.

### If it hangs

Three probes (`MVMUL`, `ELWADD`, `ELWMUL`) need the matrix unit's SrcA/SrcB
data-valid bits, which the benchmark sets with a bare `SETDVALID`. That is the
one thing here that depends on Tensix state rather than being self-contained.
If the program hangs, kill it and run:

```bash
./build/tensixbench --blocks 32 --iters 64 --no-dvalid-probes
```

**and tell us it hung.** A hang there is itself a result. Everything else still
gets measured.

To bisect further, `--probes 0xMASK` enables probe `i` with bit `i`; the summary
table prints the probes in slot order. `--phase a` or `--phase b` runs one half.

---

## What to send back

**Both CSVs.** Plus the terminal output of each if you have it, because the
summary tables and the validity verdicts are in there and not in the CSV.

```
tensixbench-<arch>.csv
tensixbench-<arch>-dvalid-per-thread.csv
```

Each has one row per raw measurement and no derived numbers at all:

```
phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles
```

with the run's configuration in a `#` header line (`arch=`, `probe_mask=`,
`dvalid_setup=`, `mm_block=`), which is how the two runs tell themselves apart.

If you ran with `--no-dvalid-probes`, or without slow dispatch, or the program
exited non-zero — say which. All three change how the numbers are read.

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
near **3**, and `RDCFG` near **2**, because that is what the ISA documentation
says. If they all read 1.0 on hardware, that is the most interesting possible
result and we want to hear about it immediately.

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
