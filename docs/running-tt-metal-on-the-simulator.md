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
grid, matching tt-metal's default compute-grid range, so the host allocates
cleanly with no override.

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

By **default only one worker tile, physical `(1, 1)`, is materialized** — every
other worker coordinate is a `NullCore` (zeroes reads, swallows writes). This
keeps the per-cycle pump cheap. Single-core programs work out of the box;
**multi-core programs must list every worker coord they use**:

```bash
export TT_SIM_TENSIX_COORDS=1-1,1-2      # comma-separated PHYSICAL x-y coords
```

**Or specify a bare count** and let the simulator pick sensible default coords:

```bash
export TT_SIM_TENSIX_CORES=2             # materialise 2 workers at 1-1, 1-2
```

`TT_SIM_TENSIX_CORES=N` materialises N workers **column-major from `1-1`**
(`1-1,1-2,1-3,1-4,1-5,2-1,…`). This matches the order tt-metal actually drives
program cores under the default grid override — both `matmul_multi_core` and
`noc_tile_transfer` launch `1-1` then `1-2`, and matmul fills the whole `x=1`
column first — so a plain count covers the coords typical programs use without
your having to know them. If a program launches on a coord outside the chosen
set you still get the exact go=GO error (§1.3, below) naming what to add; switch
to explicit `TT_SIM_TENSIX_COORDS` for off-origin placements. The two vars are
**mutually exclusive** — set one, not both.

> **Interaction with §1.2.** A `*_multi_core` program distributes its work
> across tt-metal's *compute grid*. With no override that grid is the full 8×10,
> so correct results would need **all 80** worker tiles materialized
> (impractically slow). Exporting `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4`
> shrinks the compute grid to the 4×5 block, so you only need those **20** tiles.
> Either way, cores you don't materialize return zeros. **Beware:** some
> examples (e.g. `vecadd_multi_core`, `vecadd_sharding`) print per-element
> mismatches but still `exit 0`, so exit code alone is not a correctness signal
> for them — grep the log for `Mismatch`.

- Coords are **physical NoC** coords, not logical. Logical→physical within the
  4×5 block: logical `(col, row)` → physical `(col+1, row+1)`. So logical
  `(0,0)`→`1-1`, logical `(0,1)`→`1-2`.
- The 4×5 sub-block (what `…OVERRIDE_TODEPRECATE=3,4` selects) is 20 tiles
  (physical `x∈{1,2,3,4}`, `y∈{1,2,3,4,5}`):
  ```
  1-1,2-1,3-1,4-1,1-2,2-2,3-2,4-2,1-3,2-3,3-3,4-3,1-4,2-4,3-4,4-4,1-5,2-5,3-5,4-5
  ```
  The full declared grid additionally has `x∈{6,7,8,9}` and `y∈{7,8,9,10,11}`.
- Invalid coords fail fast at server start. If a program addresses a worker you
  did **not** list, the server prints
  `WARNING: wire traffic to functional worker X-Y … not in TT_SIM_TENSIX_COORDS`
  and that traffic is silently NullCore-swallowed — a common cause of "result is
  zeros / low PCC".
- **A kernel *launch* on an unlisted worker is a hard error, not a warning.**
  The grid-wide init handshake (`go=INIT`) touches every worker and is harmless,
  but a `go=GO` only ever targets cores a program actually runs on. When one
  reaches an un-materialized worker the server prints
  `ERROR: kernel launch (go=GO) sent to functional worker X-Y … which tt-sim did
  not materialise …` (naming the exact coord to add) and exits. This is the
  unambiguous "start tt-sim with more cores" signal: add the named `X-Y` to
  `TT_SIM_TENSIX_COORDS`. Note the host (tt-metal) has no "simulator died" path,
  so it will still hang on its next poll — but the ERROR line prints the instant
  the launch is attempted, so the reason is on screen; interrupt and re-run with
  the coord added.
- Cost: each materialized Tensix tile is heavy (5 RISC-V cores + coprocessor,
  pumped every cycle via the threaded clock). More tiles = slower wall-clock.
  Reach for the minimal set a program actually uses.

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
`y∈{0,1,2,3}`). Translating logical→physical (`+1` on each axis) gives the
coords to materialise:

```bash
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,3     # 2 x 4 compute grid -> 8 L1 banks
export TT_SIM_TENSIX_COORDS=1-1,1-2,1-3,1-4,2-1,2-2,2-3,2-4
```

Verify from the JIT compile line in the log — it carries the bank count the
host actually derived:

```
-DNUM_L1_BANKS=8 -DLOG_BASE_2_OF_NUM_L1_BANKS=3
```

`TT_SIM_TENSIX_CORES=8` is **not** a substitute here: it fills column-major to
the full column height, giving `1-1,1-2,1-3,1-4,1-5,2-1,2-2,2-3`, which is not
the 2×4 block. Whenever the override's grid is not 5 rows tall, name the coords
explicitly.

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
```

Multi-core example:

```bash
TT_SIM_TENSIX_COORDS=1-1,1-2 ./metal_example_noc_tile_transfer
```

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
| stdout contains `can not handle instruction 'X'` | **FAIL** — unimplemented Tensix/SFPU op; the sim server raised and died, so the host then hangs → also shows as timeout |

Success lines vary by example; match any of:
`Success`, `Test Passed`, `matches expected value`,
`Result = N : Expected = N`, `completed successfully`.

Distinguishing **hung** vs **slow** on a timeout: the simulator runs a progress
watchdog. If it prints a `[DEADLOCK cycle=…]` block, it is genuinely wedged. If
no deadlock fires and the server message count / DRAM upload is still advancing,
it is merely slow (multi-tile matmul is the usual culprit — raise the timeout or
reduce the tile set). See `TT_SIM_DEADLOCK*` below.

Machine-readable recipe:

```bash
run_example() {           # usage: run_example <name> [coords] [timeout_s]
  local name="$1" coords="${2:-1-1}" tmo="${3:-240}"
  local log; log="$(mktemp)"
  TT_SIM_TENSIX_COORDS="$coords" timeout "$tmo" \
    "./metal_example_$name" >"$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && grep -qaE 'Success|Test Passed|matches expected|Result *=.*Expected|completed successfully' "$log"; then
    echo "PASS  $name"
  elif [ $rc -eq 124 ]; then
    echo "TIMEOUT  $name"
  else
    echo "FAIL  $name :: $(grep -aoE "can not handle instruction '[^']*'|PCC not high enough[^)]*|Result mismatch[^\"]*" "$log" | head -1)"
    echo "  (log: $log)"
  fi
}
```

### 3.1 Clean up between runs

When a run is killed or times out, the UMD-spawned simulator server can be
orphaned. Reap stragglers before the next run so they don't accumulate and
consume memory:

```bash
pkill -9 -f 'driver.wormhole.server'
pkill -9 -f 'metal_example_'
```

---

## 4. Diagnostics & tracing (opt-in)

All are read from the environment and work in this tt-metal-driven flow.

### 4.1 Server / protocol

| Variable | Effect |
| --- | --- |
| `TT_SIM_LOG_PROTOCOL=1` | print every wire message (READ/WRITE/RESET) to stderr |
| `TT_SIM_RECORD=<file>` | record every wire message **and READ reply data** to `<file>` (text) |
| `TT_SIM_CYCLES_PER_POLL=N` | sim cycles to run after each wire message (default 100) |
| `TT_SIM_MOCK_TENSIX=1` | skip building the Wormhole; every core is a NullCore (fast, for wire-level debugging only) |
| `TT_SIM_PUMP_STRIDE=0` | disable the pump's time-skipping (on by default) — see below |
| `TT_SIM_COST_MODEL=1` | charge each op the cycle cost the ISA-doc tables give it (off by default) — see below |
| `TT_SIM_DISABLE_ALIGNMENT_CHECKS=1` | accept NoC transfers whose source and destination addresses are not congruent, which hardware treats as undefined behaviour |

`TT_SIM_PUMP_STRIDE=0` turns off the event-driven pump's ability to jump
straight to the next cycle any tile actually needs, making it tick every cycle
as it did before. `run(N)` advances exactly N cycles either way and
`TT_SIM_CYCLES_PER_POLL` is unaffected — a stride can never overshoot a poll
window — so this is a debugging switch: if a result differs with it set, the
difference is a pump bug and worth reporting. See
[`docs/plans/event-driven-pump.md`](plans/event-driven-pump.md).

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

---

## 5. Limitations to expect

- **Slow dispatch only.** Only `detail::LaunchProgram` is modelled; the
  command-queue/fast-dispatch path is not. `TT_METAL_SLOW_DISPATCH_MODE=1`
  (set by the venv) makes `EnqueueProgram` fall back to it.
- **One tile by default** — see §1.3. Multi-core programs need
  `TT_SIM_TENSIX_COORDS`.
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
# TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE is NOT needed on 0.74 (see §1.2).
cd "$TT_METAL_RUNTIME_ROOT/build/programming_examples"

# name                              coords          timeout
run_example add_2_integers_in_compute  1-1            240
run_example noc_tile_transfer          1-1,1-2        240
run_example eltwise_sfpu               1-1            240

pkill -9 -f 'driver.wormhole.server' 2>/dev/null || true
```

(with `run_example` from §3.)
