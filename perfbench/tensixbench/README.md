# tensixbench — running it on real hardware

**You have a Tenstorrent card. This page is everything you need; you do not need
to know anything about tt-sim.** It asks you to build one program, run it once,
and send back one CSV file. Budget **20 minutes**, most of it the build.

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

./build/tensixbench --blocks 32 --iters 64
```

`--blocks 32` and `--iters 64` are the **hardware settings** — bigger than the
defaults, which are sized for the simulator. Larger bursts push the fixed costs
further into the noise; on silicon they cost milliseconds.

The run prints a summary table and writes `tensixbench-wormhole.csv` or
`tensixbench-blackhole.csv` next to itself.

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

**One file: the CSV.** Plus the terminal output if you have it, because the
summary table and the validity verdict are in there and not in the CSV.

```
tensixbench-<arch>.csv
```

It has one row per raw measurement and no derived numbers at all:

```
phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles
```

If you ran with `--no-dvalid-probes`, or without slow dispatch, or the program
exited non-zero — say which. All three change how the numbers are read.

---

## How to tell a good run from a silently bad one

The program checks itself and prints a verdict on the last line:

```
TTBENCH_VALID: yes
Completed successfully on the device
```

`TTBENCH_VALID: no` means **do not send that run** — it prints which check
failed. The checks are:

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
  cycles per `matmul_tiles` call. The **absolute** value is a composite of math,
  unpack and circular-buffer work and means little on its own. The
  **difference** between fidelities is the result: `HiFi2 − LoFi` should be
  ~16 cycles and `HiFi4 − HiFi2` ~32.

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

## Running it against the simulator instead

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh tensixbench -- --blocks 2 --iters 4
```

`TT_SIM_ARCH=wormhole` picks the other simulator; `TT_SIM_COST_MODEL=1` turns
the cycle cost model on. Keep `--blocks` small: the simulator runs a few tens of
thousands of cycles per second.
