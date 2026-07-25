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

### 1.2 Required for tt-metal ≥ 0.70

```bash
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4
```

**This is mandatory on 0.70+ (including 0.74).** It truncates tt-metal's
compute grid to the gap-free 4×5 worker block declared in
`driver/wormhole/soc_descriptor.yaml`. Without it the host asks for logical
workers that don't exist in the truncated SoC and aborts in
`L1BankingAllocator::generate_config`. (Background: works around a latent UMD
bug in `SimulationChip::noc_multicast_write`; can be dropped once that's fixed
upstream.)

### 1.3 Multiple Tensix tiles

By **default only one worker tile, physical `(1, 1)`, is materialized** — every
other worker coordinate is a `NullCore` (zeroes reads, swallows writes). This
keeps the per-cycle pump cheap. Single-core programs work out of the box;
**multi-core programs must list every worker coord they use**:

```bash
export TT_SIM_TENSIX_COORDS=1-1,1-2      # comma-separated PHYSICAL x-y coords
```

- Coords are **physical NoC** coords, not logical. Logical→physical for the
  truncated grid: logical `(col, row)` → physical `(col+1, row+1)`. So logical
  `(0,0)`→`1-1`, logical `(0,1)`→`1-2`.
- The full truncated worker grid is 20 tiles (physical `x∈{1,2,3,4}`,
  `y∈{1,2,3,4,5}`):
  ```
  1-1,2-1,3-1,4-1,1-2,2-2,3-2,4-2,1-3,2-3,3-3,4-3,1-4,2-4,3-4,4-4,1-5,2-5,3-5,4-5
  ```
- Invalid coords fail fast at server start. If a program addresses a worker you
  did **not** list, the server prints
  `WARNING: wire traffic to functional worker X-Y … not in TT_SIM_TENSIX_COORDS`
  and that traffic is silently NullCore-swallowed — a common cause of "result is
  zeros / low PCC".
- Cost: each materialized Tensix tile is heavy (5 RISC-V cores + coprocessor,
  pumped every cycle via the threaded clock). More tiles = slower wall-clock.
  Reach for the minimal set a program actually uses.

---

## 2. Running a program

The upstream examples are pre-built binaries under
`$TT_METAL_RUNTIME_ROOT/build/programming_examples/` named
`metal_example_<name>`.

```bash
source /home/nick/projects/riscv/venv/bin/activate
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4
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
  release-specific. The standalone (`python3 one/one.py`) flow reads
  `driver/wormhole/tt_metal_<ver>.json`; the tt-metal-driven flow gets the
  layout from the host binary over the wire, so it just needs a build whose
  release the sim tracks.

---

## 6. Minimal test-cycle example

```bash
#!/usr/bin/env bash
set -u
source /home/nick/projects/riscv/venv/bin/activate
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4
cd "$TT_METAL_RUNTIME_ROOT/build/programming_examples"

# name                              coords          timeout
run_example add_2_integers_in_compute  1-1            240
run_example noc_tile_transfer          1-1,1-2        240
run_example eltwise_sfpu               1-1            240

pkill -9 -f 'driver.wormhole.server' 2>/dev/null || true
```

(with `run_example` from §3.)
