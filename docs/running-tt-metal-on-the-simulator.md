# Running tt-metal programs on the simulator

This is a runbook for executing **real tt-metal programs** (e.g. the upstream
`programming_examples/`) against tt-sim instead of physical hardware. It is
written to be followed both by a human and by an automated agent driving a
test cycle.

The mechanism: tt-metal's UMD has a "simulation" chip backend. When
`TT_METAL_SIMULATOR` points at `driver/wormhole/`, UMD spawns
`driver/wormhole/run.sh`, which starts the tt-sim wire-bridge server
(`python -m driver.wormhole.server`). The tt-metal host binary then talks to
the simulator over an NNG IPC socket exactly as it would talk to real silicon.
Only the **slow-dispatch** launch path is supported (see Limitations).

---

## 1. Environment

### 1.1 Base setup (provided by the venv)

Activating the project venv exports the four base variables:

```bash
source /home/nick/projects/riscv/venv/bin/activate
```

sets:

| Variable | Value (this environment) | Meaning |
| --- | --- | --- |
| `TT_METAL_RUNTIME_ROOT` | `…/tt-metal-0.74/tt-metal` | tt-metal checkout the host binaries live in |
| `TT_METAL_SIMULATOR` | `…/tt-sim/driver/wormhole` | dir UMD launches (`run.sh`) as the sim device |
| `TT_METAL_SLOW_DISPATCH_MODE` | `1` | forces `EnqueueProgram` to fall back to `detail::LaunchProgram` |
| `LD_LIBRARY_PATH` | `…/tt-metal/build_Release/lib:…` | tt-metal shared libs |

If you are not using the venv, export those four yourself.

### 1.2 Compute-grid override (optional)

**Nothing extra is required on 0.74** — it works out of the box.
`driver/wormhole/soc_descriptor.yaml` declares the full 8×10 Wormhole worker
grid, which *contains* tt-metal's default compute-grid range, so the host
allocates cleanly with no override.

**The default compute grid is 8×9 = 72 cores, not 8×10 = 80.** The simulated
chip reports zero harvested rows, so tt-metal resolves the `galaxy` product's
row-dispatch entry in `tt_metal/core_descriptors/wormhole_b0_80_arch.yaml`,
whose `compute_with_storage_grid_range` ends at logical `[7, 8]`. Physical
worker row 11 is therefore never a compute core, and materialising it is wasted
money. Measure it rather than trusting this paragraph across releases — the
override's own bounds check prints the answer in about a second:

```bash
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=7,9 ./metal_example_add_2_integers_in_compute
# -> TT_FATAL ... compute_with_storage_end[1]= 8 should be >= ...override[1]= 9
```

Blackhole's is **13×10 = 130** (`blackhole_140_arch.yaml`, `unharvested`/`col`,
ending `[12, 9]`), against 14×10 = 140 declared workers. Both numbers live in
`DEFAULT_COMPUTE_GRID` in each driver's `server/coords.py`; each server prints
the grid it resolved in its `ready` line, and `TT_SIM_COMPUTE_GRID=WxH`
overrides it if a release moves the default.

```bash
# OPTIONAL — restricts tt-metal to the gap-free 4x5 sub-block:
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4
```

Set this only for **old-version backwards compatibility** (releases whose UMD
still hit the `SimulationChip::noc_multicast_write` gap-cell bug) or to make a
run cheaper by shrinking the worker grid. The 4×5 block is a subset of the
declared grid, so the override still resolves. If you ever revert the
descriptor to a truncated grid, this override becomes mandatory again (without
it the host aborts in `L1BankingAllocator::generate_config` asking for logical
workers like `(0, 5)`).

### 1.3 Multiple Tensix tiles

**Nothing to set.** tt-sim materialises exactly the workers a program uses, as
it discovers them, so single-core and 72-core programs both just run:

```bash
./metal_example_add_2_integers_in_compute   # 1 worker,  ~7 s
./metal_example_vecadd_multi_core           # 72 workers, ~74 s
./metal_example_matmul_multi_core           # 72 workers, ~390 s, PCC 0.9999
```

The server's `ready` line says `(on demand)`, and its shutdown line reports what
the program actually needed:

```
[server] tt-sim Wormhole ready (tensix=[(1, 1)] (on demand), … compute_grid=8x9, …)
[server] shutdown after 5547 messages, 72 tensix materialised (71 on demand)
```

How it decides, in `tt_sim/bridge/materialise.py`: a worker coordinate starts as
a journalling stand-in that is wire-identical to the old `NullCore` (writes
swallowed, reads zero — which is what lets the grid-wide `go=INIT` handshake
complete), and a real tile is built at the first of three signals — a host write
to a core already released from reset (i.e. its kernel binaries), a `go=GO`, or
**a NoC request from a peer**. The third is the one that has to exist: a worker
launched early runs its kernel thousands of cycles before the last worker is
mentioned, so it can address a peer that does not exist yet, and dropping that
packet would be a silently wrong answer rather than a hang.

**Why not simply materialise the whole grid?** Because a worker a program never
launches on is not free — tt-metal's init handshake releases BRISC on *every*
declared worker, so it runs base firmware all run. Measured, A/B/A:
`add_2_integers_in_compute` is **7.0 s** on demand, **28.5 s** with
`TT_SIM_TENSIX_CORES=72`, **7.3 s** on demand again.

#### Pinning the set by hand

Both env vars still work, and both now mean *exactly these workers and no
others* — nothing is materialised on demand when either is set. Use them for
reproducibility (the replay guards and the cost-model gate do) or to
deliberately starve a program of cores:

```bash
export TT_SIM_TENSIX_COORDS=1-1,1-2      # comma-separated PHYSICAL x-y coords
export TT_SIM_TENSIX_CORES=2             # ...or a bare count: 2 workers, 1-1 1-2
```

`TT_SIM_TENSIX_CORES=N` pins **the N workers tt-metal will actually launch on**:
its compute grid, filled column-major, which is what `split_work_to_cores` →
`num_cores_to_corerangeset` does. The column is as tall as *that* grid, so the
answer moves with §1.2 — with no override the 6th core is `1-7` (a 9-tall
column, skipping the ethernet row), under `…OVERRIDE_TODEPRECATE=3,4` it is
`2-1` (a 5-tall one). tt-sim reads the override out of the environment for you,
so a plain count is correct in both regimes; the resolved grid is printed in the
server's `ready` line. If a program places cores some other way you get the exact
go=GO error (below) naming what to add. The two vars are **mutually exclusive** —
set one, not both.

> **Interaction with §1.2.** A `*_multi_core` program distributes its work
> across tt-metal's *compute grid*, so a pinned run has to cover it: with no
> override that grid is 8×9 and correct results need **72** workers
> (`TT_SIM_TENSIX_CORES=72`). `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4`
> shrinks it to the 4×5 block, so **20** suffice. On demand this is simply not
> a decision you have to make. **Beware either way:** some examples (e.g.
> `vecadd_multi_core`, `vecadd_sharding`) print per-element mismatches but still
> `exit 0`, so exit code alone is not a correctness signal for them — grep the
> log for `Mismatch`.

- Coords are **physical NoC** coords, not logical, and the mapping is an
  **index into the sorted worker axes, not `+1`**: logical `(i, j)` is physical
  `(xs[i], ys[j])` where `xs = 1,2,3,4,6,7,8,9` and `ys = 1,2,3,4,5,7,8,9,10,11`
  on Wormhole. Inside the gap-free 4×5 corner that happens to look like
  `(col+1, row+1)` — logical `(0,0)`→`1-1`, `(0,1)`→`1-2` — but logical row 5 is
  physical `7`, not `6`, and on Blackhole logical column 7 is physical `10`, not
  `8`. Assuming the offset rule has already cost this project a wrong run.
- The 4×5 sub-block (what `…OVERRIDE_TODEPRECATE=3,4` selects) is 20 tiles
  (physical `x∈{1,2,3,4}`, `y∈{1,2,3,4,5}`):
  ```
  1-1,2-1,3-1,4-1,1-2,2-2,3-2,4-2,1-3,2-3,3-3,4-3,1-4,2-4,3-4,4-4,1-5,2-5,3-5,4-5
  ```
  The full declared grid additionally has `x∈{6,7,8,9}` and `y∈{7,8,9,10,11}`.
- Invalid coords fail fast at server start. If a program addresses a worker you
  did **not** list (pinned mode only — on demand there is no such thing), the
  server prints
  `WARNING: wire traffic to functional worker X-Y … not in TT_SIM_TENSIX_COORDS`
  and that traffic is silently NullCore-swallowed — a common cause of "result is
  zeros / low PCC".
- **A kernel *launch* on an unlisted worker is a hard error, not a warning —
  in pinned mode.** The grid-wide init handshake (`go=INIT`) touches every
  worker and is harmless, but a `go=GO` only ever targets cores a program
  actually runs on. When one reaches an un-materialized worker the server prints
  `ERROR: kernel launch (go=GO) sent to functional worker X-Y … which tt-sim did
  not materialise …` (naming the exact coord to add). This is the unambiguous
  "start tt-sim with more cores" signal: add the named `X-Y` to
  `TT_SIM_TENSIX_COORDS` and re-run.
  **The run ends there — the server stops the host as well as itself.** tt-metal
  has no "simulator died" path (UMD blocks in `recv_from_device` with no
  timeout), so a server that merely exited used to leave the host waiting for
  ever; that hang *was* this diagnostic, and it cost two debugging sessions. The
  server now identifies the host as the process holding the listening socket at
  `$NNG_SOCKET_ADDR` and `SIGTERM`s it, printing
  `stopping the tt-metal host (pid N): it is waiting for …`. The host exits with
  signal 15 within a couple of seconds instead of hanging. If the host cannot be
  identified the server says so and *keeps serving*, so the program reaches its
  own (meaningless) end rather than never ending. Set `TT_SIM_NO_HOST_STOP=1` to
  suppress the stop and get the old interrupt-by-hand behaviour.
- **A `go=GO` aimed at something that is not a functional worker at all**
  (off-grid, a DRAM endpoint, an eth core on an architecture whose eth tiles
  are unmodelled) is reported in both modes — nothing could ever materialise it
  — but is *not* fatal: a NullCore reads back `RUN_MSG_DONE` immediately so the
  host cannot hang on it, and the detection is a 4-byte-write heuristic that a
  false positive would turn into a false kill.
- Cost: each materialized Tensix tile is heavy (5 RISC-V cores + coprocessor,
  pumped every cycle). Wall clock is close to flat in the count of workers a
  program *uses* (§1.3.2), but workers it never launches on are pure loss —
  which is what on-demand materialisation exists to avoid.

#### 1.3.1 Sizing the grid override: the L1 bank count must be a power of 2

`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` is **not** a core count — it is the
*inclusive maximum logical `x,y`* of the compute-with-storage grid, so it
selects a grid of `(x+1) × (y+1)` cores (`core_descriptor.cpp`,
`compute_with_storage_end` override). tt-metal then gives **one L1 bank per
compute-and-storage core** (`AllocatorImpl::init_compute_and_storage_l1_bank_manager`)
and `validate_num_banks` rejects any count that is neither a power of 2 nor one
of its hardcoded special cases (`7, 12, 20, 48, 56, 63, 70, …`):

```
Invalid number of memory banks 15 for L1 (must be power of 2 or have a
dedicated modulo implementation)
```

So **choose the override so that `(x+1) × (y+1)` is a power of 2** (or one of
the listed exceptions — which is why the 4×5 = 20 default override works).
`3,4` → 20 ✓ (exception), `1,3` → 8 ✓, `3,1` → 8 ✓, `1,1` → 4 ✓, `3,3` → 16 ✓,
but `2,4` → **15 ✗** and `2,2` → **9 ✗**.

**Worked example — an 8-core 2×4 grid.** A program that asks for 8 cores out of
`compute_with_storage_grid_size()` (e.g. via `num_cores_to_corerangeset(8, …)`)
wants a compute grid of exactly 8, i.e. override `1,3` (logical `x∈{0,1}`,
`y∈{0,1,2,3}`). That grid is inside the gap-free corner, so logical→physical is
`+1` on each axis here:

```bash
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,3     # 2 x 4 compute grid -> 8 L1 banks
export TT_SIM_TENSIX_CORES=8                           # or name them:
#      TT_SIM_TENSIX_COORDS=1-1,1-2,1-3,1-4,2-1,2-2,2-3,2-4
```

Verify from the JIT compile line in the log — it carries the bank count the
host actually derived:

```
-DNUM_L1_BANKS=8 -DLOG_BASE_2_OF_NUM_L1_BANKS=3
```

`TT_SIM_TENSIX_CORES=8` **is** a substitute here, as long as the override is
exported in the same environment: the count knob reads it and fills that grid's
own 4-tall columns, giving exactly the eight coords above. (It did not always —
it used to fill a fixed 4×5 block first, producing
`1-1,1-2,1-3,1-4,1-5,2-1,2-2,2-3`, three of which the program never launches on
while three it does launch on were missing. That is the class of bug the
go=GO error below exists to catch, and `tt_sim/bridge/grid_test.py` now pins the
order.)

#### 1.3.2 What a bigger grid actually costs (measured 2026-08-12)

A grid sweep of `metal_example_vecadd_multi_core` (640 tiles of bfloat16 vector
add, spread across the whole compute grid) on Wormhole, tt-metal 0.74, one
worker per compute core. Every run computed the right answer — the example's own
`All results match expected values within tolerance`, checked for absence of
`Mismatch` as well:

| workers | grid override | compute grid | wall | verdict |
| --- | --- | --- | --- | --- |
| 1 | `0,0` | 1×1 | 49.5 s | correct |
| 4 | `1,1` | 2×2 | 44.4 s | correct |
| 12 | `3,2` | 4×3 | 65.0 s | correct |
| 20 | `3,4` | 4×5 | 68.0 s | correct |
| 32 | `7,3` | 8×4 | 79.0 s | correct |
| 64 | `7,7` | 8×8 | 87.7 s | correct |
| 80 | none | 8×9 (+8 idle) | 89.4 s | correct |
| 72 | none, `TT_SIM_TENSIX_CORES=72` | 8×9 | 81.5 s | correct |

**Wall clock is roughly flat in the worker count for a fixed problem.** That is
not a paradox: the total simulated work is the same 640 tiles however they are
split, tt-sim executes it sequentially, and firmware-loop parking means a worker
waiting on its go message costs almost nothing. The ~1.8× spread from 1 to 80 is
the per-tile fixed cost — construction, the launch handshake, the pump visiting
each tile — not the compute. **A wide grid is not the expensive thing a
single-tile intuition suggests.**

`metal_example_matmul_multi_core` (640³, PCC-checked against a host golden) on
the same day: 4 workers 352.7 s at PCC 0.99991, 20 workers 359.4 s at PCC
0.99989, 72 workers 492.3 s at PCC 0.99992 — all three inside the vendor
simulator's own PCC band, and 1.4× wall for 18× the workers. The smaller
multi-core examples: `noc_tile_transfer` 2 workers 7.9 s, `shard_data_rm` 4
workers 8.8 s, `contributed/multicast` 4 workers 10.2 s, `vecadd_sharding` 8
workers 11.3 s, `pad_multi_core` 4 workers 26.8 s. On Blackhole,
`TT_SIM_TENSIX_CORES=2` runs `noc_tile_transfer` in 8.6 s and
`TT_SIM_TENSIX_CORES=20` with `…OVERRIDE_TODEPRECATE=3,4` runs
`vecadd_multi_core` on `1-2 … 4-6` in 56.1 s.

**What is still expensive is a materialised worker a program never launches on.**
`add_2_integers_in_compute` is a one-core program: 7.6 s with one worker
materialised, **28.9 s with 72** (and 7.6 s again on the repeat, so that is not
drift). Every declared worker is released from soft reset by the grid-wide init
handshake and runs firmware whether or not a kernel lands on it. That 3.8× is
why raising the default was never the answer — the whole example ladder, every
replay guard and the cost-model gate are single-core, and they would all have
paid it.

**Superseded by on-demand materialisation (§1.3).** The table above is what a
*hand-pinned* grid costs, and the numbers it should be read against are what the
same programs now cost with **no grid environment variable set at all**, on the
same machine:

| program | pinned (survey) | on demand | workers built |
| --- | --- | --- | --- |
| `add_2_integers_in_compute` | 7.6 s @ 1, 28.9 s @ 72 | **7.0 / 7.3 s** | 1 |
| `vecadd_multi_core` | 81.5 s @ 72, 89.4 s @ 80 | **73.9 s** | 72 |
| `matmul_multi_core` | 492.3 s @ 72, PCC 0.99992 | **388.4 s**, PCC 0.99991 | 72 |

The on-demand runs are *faster* than the pinned ones that computed the same
answer, because the workers appear as the program reaches them rather than all
being live from device init. Ask for nothing; you get the workers your program
uses, no more and no fewer.

Timings are single runs on a machine with other work on it — read them as
shape, not as a benchmark.

---

## 2. Running a program

The upstream examples are pre-built binaries under
`$TT_METAL_RUNTIME_ROOT/build/programming_examples/` named
`metal_example_<name>`. The tt-sim examples in `examples/<name>/src/` are the
same kind of program — build one with `cmake -B build -S . && cmake --build build` and run
`./build/<name>` from its `src/` dir (see `driver/wormhole/README.md`), or run the whole
set via `python3 -m examples.examples_test`.

```bash
source /home/nick/projects/riscv/venv/bin/activate
cd "$TT_METAL_RUNTIME_ROOT/build/programming_examples"

./metal_example_add_2_integers_in_compute        # single-tile: just works
./metal_example_noc_tile_transfer               # 2 workers: also just works
./metal_example_vecadd_multi_core               # 72 workers: also just works
```

**Set no grid variable.** Workers materialise on demand (§1.3), so a multi-core
program needs nothing that a single-core one does not — and setting
`TT_SIM_TENSIX_COORDS` at all switches materialisation *off*, which is a way to
get a wrong answer rather than a way to help.

**Always run under `timeout`** in an automated cycle — a wedged kernel or a
crashed server manifests as a silent hang:

```bash
timeout 240 ./metal_example_<name>
```

---

## 3. Interpreting the result (pass/fail convention)

Decide pass/fail from **exit code + stdout**, in this order:

| Signal | Meaning |
| --- | --- |
| exit `0` **and** a success line | **PASS** |
| exit `124` | **TIMEOUT** — hung, or just too slow (see below) |
| exit `134` / non-zero with `TT_FATAL` / `mismatch` / `PCC not high enough` | **FAIL** (correctness) |
| stdout contains `can not handle instruction 'X'` | **FAIL** — unimplemented Tensix/SFPU op; the sim server raises and dies. It now takes the host down with it (`stopping the tt-metal host …`), so this shows as a prompt signal-15 exit rather than a timeout |

Success lines vary by example; match any of:
`Success`, `Test Passed`, `matches expected value`,
`Result = N : Expected = N`, `completed successfully`.

Distinguishing **hung** vs **slow** on a timeout: the simulator runs a progress
watchdog. If it prints a `[DEADLOCK cycle=…]` block, it is genuinely wedged. If
no deadlock fires and the server message count / DRAM upload is still advancing,
it is merely slow (multi-tile matmul is the usual culprit — raise the timeout or
reduce the tile set). See `TT_SIM_DEADLOCK*` below.

Machine-readable recipe: **use the gate.**
[`driver/tests/upstream_sweep.py`](../driver/tests/upstream_sweep.py) already
encodes each upstream program's binary name, success line, timeout and expected
verdict, runs them on both architectures with no grid variable set, and ends in
`RESULT: PASS` / `RESULT: FAIL`:

```bash
python3 -m driver.tests.upstream_sweep              # 17 programs x 2 arches, ~8 min
python3 -m driver.tests.upstream_sweep --tier full  # + the four heavy programs
python3 -m driver.tests.upstream_sweep --list       # what it runs, and what it excludes
```

Add a program there rather than hand-rolling a loop; the results and the triage
behind them are in
[`docs/upstream-examples-status.md`](upstream-examples-status.md).

### 3.1 Clean up between runs

When a run is killed or times out, the UMD-spawned simulator server can be
orphaned. The test scripts and the gate handle their own: each stamps a per-run
tag into the server's command line (`TT_SIM_RUN_TAG` → `--run-tag`, see
`driver/sim_procs.sh`) and kills only servers carrying its tag, plus servers
tagged by a run whose owner has since died. **A concurrent run in another
terminal is never disturbed.**

A *manual* run carries no tag, so nothing reaps it. Clear those by hand:

```bash
pkill -9 -f 'driver.wormhole.server'
pkill -9 -f 'metal_example_'
```

Do **not** put that `pkill` in a script that may run beside a live one — that is
exactly the mistake the run tags exist to prevent. The scripted opt-in for "kill
every simulator on the machine" is `TT_SIM_KILL_ALL_SERVERS=1`.

---

## 4. Diagnostics & tracing (opt-in)

All are read from the environment and work in this tt-metal-driven flow.

### 4.1 Server / protocol

| Variable | Effect |
| --- | --- |
| `TT_SIM_LOG_PROTOCOL=1` | print every wire message (READ/WRITE/RESET) to stderr |
| `TT_SIM_RECORD=<file>` | record every wire message **and READ reply data** to `<file>` (text) |
| `TT_SIM_CYCLES_PER_POLL=N` | sim cycles to run after each wire message (default 100) — leave it alone, including when profiling; see below |
| `TT_SIM_MOCK_TENSIX=1` | skip building the Wormhole; every core is a NullCore (fast, for wire-level debugging only) |
| `TT_SIM_PUMP_STRIDE=0` | disable the pump's time-skipping (on by default) — see below |
| `TT_SIM_COST_MODEL=1` | charge each op the cycle cost the ISA-doc tables give it (off by default) — see below |
| `TT_SIM_DISABLE_ALIGNMENT_CHECKS=1` | accept NoC transfers whose source and destination addresses are not congruent, which hardware treats as undefined behaviour |
| `TT_SIM_NOC1_SHADOW=warn\|error\|off` | what to do when a live Tensix worker is unreachable on NoC 1 because a mirror registration took its canonical coord (default `warn`, one line per coordinate on stderr) — see below |
| `TT_SIM_NUMBA=0` / `=1` | never / always use the optional compiled FPU kernel, overriding the call threshold — see below |
| `TT_SIM_NUMBA_THRESHOLD=N` | MVMULs to run before compiling the FPU kernel (default 512) |

`TT_SIM_CYCLES_PER_POLL=N` is how many simulated cycles the device advances
after each host message — the simulator's stand-in for "the host waited a
while before polling again". **The default of 100 is the right value at every
grid width; do not lower it.** Older notes here and in
[`docs/upstream-examples-status.md`](upstream-examples-status.md) told you to
set `=10` on wide grids, because the pump used to cost real time per message
per materialised tile even when nothing on the device could advance. Firmware
loop parking (a worker spinning on a go-message poll is recognised and parked)
and the pump's quiescent-window skip have between them removed that cost: at
the full Wormhole 8×10 grid, `programming_examples/vecadd_multi_core` now
passes in ~80 s at the default and measures the same at `=10`. Lowering it
only buys the device less time per message, which at the small end starts
breaking runs outright — the simulator misses the window a kernel needed and
the host reads a half-written buffer. Raise it if you want; that is only ever
"the host waited longer".

**You no longer need to raise it to profile.** The tt-metal device profiler
used to be the one thing that did: BRISC writes `RUN_MSG_DONE` *before*
`finish_profiler()` publishes the run, so the host's control-vector read — the
very next wire message — landed mid-publish and `readRiscProfilerResults`
returned early on a zero `HOST_BUFFER_END_INDEX`, giving a
`profile_log_device.csv` with nothing in it but its header. The workaround was
`TT_SIM_CYCLES_PER_POLL=5000`, which paid for it on every message of the run.
The bridge now recognises the profiler's control vector, arms on a launch, and
runs cycles at that one read until the firmware sets `PROFILER_DONE` *and* the
pushes it issued have landed in DRAM (`Device.settle_profiler_flush`); a run
with the profiler off never writes a control vector, so nothing is armed and
not one extra cycle is run. The server prints what the wait cost on its
shutdown line (`profiler flush: N settles, M extra cycles`) — measured at
1 300–1 400 cycles per launch on `mechbench` and `examples/four`.

`TT_SIM_PUMP_STRIDE=0` turns off the event-driven pump's ability to jump
straight to the next cycle any tile actually needs, making it tick every cycle
as it did before. `run(N)` advances exactly N cycles either way and
`TT_SIM_CYCLES_PER_POLL` is unaffected — a stride can never overshoot a poll
window — so this is a debugging switch: if a result differs with it set, the
difference is a pump bug and worth reporting. See
[`docs/plans/event-driven-pump.md`](plans/event-driven-pump.md).

`TT_SIM_NUMBA` controls the one optional accelerator in the tree. If
[numba](https://numba.pydata.org/) happens to be installed, the exact FPU
datapath's inner kernel — the thing an MVMUL, GAPOOL or DOTPV spends its time
in — can be run as one compiled loop nest instead of ~50 small numpy passes,
which is **13x** on that kernel and about **−24 %** on the pump time of a
matmul workload. It is bit-identical either way (`fpu_accumulate_test.py`
fuzzes the two against each other), and numba is **not a dependency**: without
it the numpy path runs exactly as before, which is why the default is the
pure-Python tree that CLAUDE.md promises.

Getting to a callable kernel costs ~3.4 s the first time on a machine and
**~800 ms per process** afterwards. Only ~30 ms of that second figure is
reading the kernel back out of numba's on-disk cache under `__pycache__`; the
rest is numba itself — ~450 ms to import the package and ~350 ms for the lazy
target-context initialisation it defers to the first compile *or cache load*.
A one-line `njit` kernel measures the same, so there is nothing to win by
making this kernel smaller, and backgrounding the warm-up on a thread is
measurably worse (the compile thread and the simulator's numpy calls convoy on
the GIL). Treat it as a floor.

That is pure loss on a workload with a handful of MVMULs, so the compile is
deferred until the first 512 have run, and a run that never issues an MVMUL
never even imports numba. **512 is a measured compromise, not a placeholder** —
engaging repays only after ~1580 further MVMULs, so lowering it to 128 costs
`matmulidx` (384 MVMULs) 789 ms for nothing, while raising it to 1024 costs
`matmulblock` 260 ms; see `tt_sim/pe/tensix/backends/fpu_jit.py`. Set
`TT_SIM_NUMBA=1` to compile on the first MVMUL, or `=0` to stay on numpy for
good.

`TT_SIM_COST_MODEL=1` (truthy = `1/true/yes/on`) turns on the per-unit
cycle-cost model: instead of every op retiring in the tick it was issued, a unit
is occupied for the number of cycles
[`tensix_instruction_costs.yaml`](../tt_sim/pe/tensix/tensix_instruction_costs.yaml)
gives its opcode, which back-pressures the thread that issued it. Only the
Tensix **matrix unit (FPU)** is wired up so far, so today the switch changes
nothing on any in-tree workload — every matrix op the ISA docs cost is a
one-cycle occupancy, including `MVMUL` at all four fidelity phases (the fidelity
multiplier is carried by the instruction *count*, not by a longer instruction).
An opcode the tables do not cost is charged nothing rather than a made-up
constant, and a cost the docs wrote as `≥ N` is charged at N, so a modelled
cycle count is a floor. See [`docs/plans/cost-model.md`](plans/cost-model.md)
and [`docs/plans/event-driven-pump.md`](plans/event-driven-pump.md) Phase 5.

`TT_SIM_LOG_PROTOCOL` is the first tool to reach for on a hang — it shows which
core/address the host is polling. `TT_SIM_RECORD` additionally captures the
data bytes so you can see the actual values (e.g. a go-message that never flips
to `RUN_MSG_DONE`).

### 4.2 Per-component instruction/transaction traces

`TT_SIM_DIAG_*` (truthy = `1/true/yes/on`) turn on human-readable stderr traces:

- Per baby-RISC-V core: `TT_SIM_DIAG_BRISC`, `_NCRISC`, `_TRISC0`, `_TRISC1`, `_TRISC2`
- Per NoC: `TT_SIM_DIAG_NOC0`, `_NOC1`
- Coprocessor: `TT_SIM_DIAG_CO_ISSUED`, `_CO_CONFIG`, `_CO_UNPACK`, `_CO_PACK`,
  `_CO_FPU`, `_CO_SFPU`, `_CO_THCON`
- Aggregates: `TT_SIM_DIAG_TRISC`, `_NOC`, `_CO`, `_ALL` (an individual var
  overrides an aggregate, so `TT_SIM_DIAG_ALL=1 TT_SIM_DIAG_NCRISC=0` = "all but
  NCRISC")

`TT_SIM_DIAG_CO_SFPU=1` in particular prints each SFPU op — invaluable when a
compute result is wrong. Full table: `driver/wormhole/README.md`.

### 4.3 Structured traces (JSONL / Perfetto / etc.)

Nine `TT_SIM_TRACE_*` writers produce machine-readable output for downstream
tooling. See **`driver/wormhole/docs/profiling.md`** for the full walkthrough.

### 4.4 Deadlock watchdog

On by default; warns (does not stop) after N cycles of no observable progress.

| Variable | Effect |
| --- | --- |
| `TT_SIM_DEADLOCK` | set falsy (`0/false/no/off`) to disable |
| `TT_SIM_DEADLOCK_THRESHOLD` | cycles of no progress before a `[DEADLOCK …]` warning (default `50000`); lower it to surface stalls sooner |

### 4.5 Wedged Tensix backend units

A wedged **Tensix backend unit** does not show up in the watchdog above, because
the rest of the device carries on: nothing back-pressures the baby RISC-V cores
on Tensix instruction issue, so the kernel behind a blocked unpacker runs to the
end and reports done. Two further checks cover that, and **they are not
equivalent** — one is a proof, the other is a hint. Reach for the proof first.

| Variable | Effect |
| --- | --- |
| `TT_SIM_UNIT_STALL` | set falsy to disable **both** checks below |
| `TT_SIM_UNIT_STALL_THRESHOLD` | consecutive cycles one unit may stay blocked on a single instruction before a `[UNIT STALL]` hint (default `10000`). Has no effect on `[UNIT WEDGED]` |

#### `[UNIT WEDGED]` — the authoritative one

```
[UNIT WEDGED cycle=41678] Unpacker 0 is still blocked with every baby core on the tile in soft reset
  UNPACR_NOP from thread 1: waiting for SrcA bank 0 to be given back by the Matrix Unit (dvalid was set and never cleared)
  ...
```

If you see this, you have a bug. It is not a heuristic and **has no threshold** —
there is nothing to tune, and `TT_SIM_UNIT_STALL_THRESHOLD` cannot silence it.
The reasoning is short enough to check: a Src bank is handed back to the unpacker
by an *instruction*, and no thread can issue an instruction from soft reset. So a
unit still blocked once every baby core on its tile is in reset is waiting for
something that can never happen, whatever the cycle count says. (A short grace
period covers a backend instruction that was still in flight when the reset
landed; those retire in single-digit cycles.)

Two practical consequences:

- It fires on a **short** reproduction — one that finishes long before any cycle
  threshold could elapse, which is exactly when someone is watching. The minimal
  `UNPACR_NOP` deadlock reproduction runs ~18,300 cycles end to end and is caught.
- It fires **after** the launch it belongs to has reported done. That pairing is
  correct, not a contradiction: on this model the issuing core is never
  back-pressured, so the launch completes and on silicon it is the *next* launch
  that hangs. See ROADMAP.md, "Unpacker dvalid deadlock".

No in-tree workload has ever produced one: not the 41 surveyed replay guards,
examples and differential op tests, and not any `examples/pipestall`
configuration. It is a warning today only because it is young; it is the check
intended to be promoted to a hard error.

#### `[UNIT STALL]` — a hint, and only a hint

Fires when one unit stays blocked on the same latched instruction for
`TT_SIM_UNIT_STALL_THRESHOLD` consecutive cycles, naming the unit, thread,
opcode and Src bank. It is genuinely useful — on a broken eight-core GEMM it
named every affected worker and none on the corrected form — but it **can fire
on correct code**, and no threshold avoids that.

The reason is architectural, not a tuning miss. A unit legitimately waits on the
whole downstream chain (math → Dst → output CB → packer → a consumer that may be
on another core), so the longest legitimate blocked run is linear in the
downstream consumer's cost with no ceiling. `examples/pipestall` measures it at
`5 · PIPESTALL_DELAY + 56` cycles (r = 1.000), so a consumer doing ~10,000 cycles
of ordinary work per tile trips the default on a kernel that computes the right
answer. Deeper buffering does not help: four credits or a four-page output CB
each bought under 8%, because a rate-limited producer stalls one consumer
turnaround per tile however deep the pipe. Raising the threshold to cover that
would defeat the other end — the wedge reproduction is only ~18,300 cycles long,
so anything above ~16,000 misses it. The interval between "high enough for a
correct pipeline" and "low enough for a short wedge" is empty.

So the message does not claim a verdict. It prints both readings and names the
line that settles it:

- **`[UNIT STALL CLEARED]`** for the same unit ⇒ it recovered. Deep pipeline;
  nothing is wrong, and the block's true length is in that line.
- **`[UNIT WEDGED]`** for the same unit ⇒ it never can. Act on that one.

Each unit reports **once per waited-on instruction** for the whole run, not once
per detection window, so the number of reports tells you how many units are
affected rather than how long you left the run going.

The default of 10,000 is measured, not guessed: every replay guard, example and
differential op test in the tree was instrumented for the longest legitimate
blocked run, with and without `TT_SIM_COST_MODEL`, and the worst single-core case
is 3,528 cycles (Wormhole `sfpumath`, unpacker 1 waiting for the SFPU to hand
SrcB back). Raise it if you are running a deliberately deep cross-core pipeline
and do not want the hint; the count is of *consecutive* blocked cycles, so a unit
that waits repeatedly but briefly never accumulates.

### 4.6 `[NoC1-SHADOW]` — a worker that is unreachable on NoC 1

NoC 1's destination directory is keyed in **two coordinate conventions at
once**, and this is not a modelling choice tt-sim is free to make: a real
kernel emits both in the same run. Verified on the captured
`noc_tile_transfer` trace, replayed offline:

```
noc1 from (1, 2) -> (9, 0) [DRAM]     # DRAM channel 0 is at (0, 11); (9, 0) is its grid mirror
noc1 from (1, 1) -> (1, 2) [Tensix]   # a peer worker, addressed by its canonical coord
```

tt-metal's `dram_bank_to_noc_xy` mirrors its NoC 1 half (`hal_.noc_coordinate`,
`risc_firmware_initializer.cpp`), while `get_noc_addr` does not mirror at all
(`NOC_0_X` is the identity on both architectures). So the directory has to hold
mirror keys for DRAM and canonical keys for workers, the two spaces overlap on
a 10×12 (Wormhole) or 17×12 (Blackhole) grid, and mirror registrations win
contested cells.

Where a mirror takes a cell a live Tensix worker owns canonically, **that
worker is unreachable on NoC 1**. The impostor accepts the write and ACKs it,
so `noc_async_write_barrier()` returns normally: a program that synchronises on
delivery hangs somewhere else entirely, and one that does not computes on stale
data.

**Tensix mirror keys are per-architecture, and the two arches genuinely
differ.** The worker half of the collision comes from `l1_bank_to_noc_xy`,
built through `RiscFirmwareInitializer::virtual_noc0_coordinate`, which
early-outs on `|| cluster_.arch() == ARCH::BLACKHOLE` — unconditionally, and
regardless of translation. So Wormhole's NoC 1 half of that table is
`mirror(NoC 0)` and Blackhole's is byte-identical to NoC 0: Blackhole has never
emitted a mirrored worker coord at all. tt-sim therefore registers Tensix
mirror aliases on Wormhole only (`ArchProfile.noc1_tensix_mirror_aliases`), and
the census with every functional worker built is **56 of Wormhole's 80** worker
coords (48 behind another worker, 8 behind DRAM) and **6 of Blackhole's 140**
(all behind DRAM, whose bank table *is* mirrored on both arches). Which ones a
given run hits depends on the workers it materialises.

Do not "simplify" this by giving both architectures the same answer: dropping
Wormhole's Tensix mirrors breaks every L1-sharded-buffer program there, and
`tt_sim/network/noc_routing_test.py` asserts both values so that change fails
loudly.

Every such coordinate is now named as it is created:

```
[NoC1-SHADOW] (4, 2) is the canonical coord of TensixTile(4, 2), but NoC 1
resolves it to DRAMTile(5, 2), which claimed the cell as its mirror.
```

| Env var | Effect |
| --- | --- |
| `TT_SIM_NOC1_SHADOW` | `warn` (default) — one line per coordinate on stderr; `error` — raise `NoC1ShadowError` at registration instead; `off` — silent. |

Warn is the default because a shadowed cell is only *wrong* once something
addresses it, and configurations that are correct today do carry an untouched
one: `programming_examples/vecadd_multi_core` under
`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4` passes its own value check while
shadowing `(4, 2)`, `(4, 3)` and `(4, 4)`, because it never writes to those
workers over NoC 1. Erroring at *resolve* time instead of registration is no
safer — the same key is the mirror by which the DRAM bank table legitimately
reaches DRAM. Use `error` when a run must not silently misdeliver, or to
bisect a hang. `tt_sim/network/noc_routing_test.py` pins the affected sets on
both architectures.

**The same collision has a self-address face.** A core reads its own coordinate
out of `NOC_CFG(NOC_ID_LOGICAL)`; tt-metal's firmware fills `my_x[]` / `my_y[]`
from it (`risc_init`), and kernels then both compare it against that NoC's bank
table (`is_local_bank`) and *emit* it as a destination — the single-argument
`get_noc_addr(addr)`. So the self-coordinate must follow the same per-arch
convention as the mirror aliases, and it does:
`ArchProfile.noc_id_logical_mirrored_on_noc1` is `True` on Wormhole (whose L1
bank table and NoC 1 directory are both mirrored) and `False` on Blackhole
(whose are not). The register's `NOC_CFG` index is architecture-specific too —
`0xE` on Wormhole, `0x12` on Blackhole, because Blackhole has six
ID-translation-table entries per axis rather than four.

With that, every tile on both architectures addresses itself back to itself,
except **Wormhole's 16 eth cores on NoC 1**: eth alone skips mirror
registration (an eth mirror would steal a DRAM tile's own canonical cell), so
its mirrored self-coordinate names a worker instead. Nothing hits it — the
slow-dispatch flow tt-sim supports launches no eth kernel — and the fix is the
translation port, not more aliases. On Blackhole the self-coordinate census is *exactly* the six
DRAM-shadowed workers above: same six cells, seen from the sending core.

---

## 5. Limitations to expect

- **Slow dispatch only.** Only `detail::LaunchProgram` is modelled; the
  command-queue/fast-dispatch path is not. `TT_METAL_SLOW_DISPATCH_MODE=1`
  (set by the venv) makes `EnqueueProgram` fall back to it.
- **Not a limitation any more: the tile count.** Workers materialise on demand
  (§1.3), so multi-core programs need no environment variable. `TT_SIM_TENSIX_*`
  is now a *pin* for reproducibility, not a requirement.
- **Throughput.** The simulator is functional, not fast. Full-tile matmuls and
  other heavy multi-tile compute can exceed a practical timeout even when
  correct. Prefer small inputs / minimal tile sets for a CI-style cycle.
- **Instruction coverage.** The Tensix coprocessor is incomplete. A
  `can not handle instruction 'X'` error means op `X` (usually an SFPU
  instruction) is not yet implemented — implement it in
  `tt_sim/pe/tensix/backends/` (see the `handle_*` methods in `vector.py` and
  the ISA docs referenced in their comments).
- **Release-specific host layout.** The L1 memory map / message structs are
  release-specific, but the tt-metal-driven flow gets the layout from the host
  binary over the wire, so it just needs a build whose release the sim tracks —
  nothing here is pinned to a tt-metal version.

---

## 6. Minimal test-cycle example

```bash
#!/usr/bin/env bash
set -u
source /home/nick/projects/riscv/venv/bin/activate
cd ~/tt-sim

# Both architectures, every quick upstream program, no grid variable, ~8 min.
python3 -m driver.tests.upstream_sweep || exit 1

# One program, one arch, while iterating on a fix:
python3 -m driver.tests.upstream_sweep --arch wormhole eltwise_sfpu
```

The gate sets `TT_METAL_SIMULATOR` itself and *removes* any inherited
`TT_SIM_TENSIX_COORDS` / `TT_SIM_TENSIX_CORES` /
`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE`, so its verdict does not depend on the
shell it was launched from. It cleans up only the simulator servers it started
(§3.1).
