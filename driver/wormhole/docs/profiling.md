# Profiling and analysing tt-sim runs

This is a walkthrough of every profiling and tracing output the
simulator can produce. The intended user is someone running a real
tt-metal program against tt-sim — i.e. a C++ binary built against
`libtt_metal` that, instead of touching silicon, is transparently
backed by the simulator. By the end you'll know what each output
gives you, how to enable it, and which downstream tool to feed it
into.

The flow assumed throughout:

```
your tt-metal binary  ──▶  UMD's tt_SimulationDevice
                                   │
                                   │ spawns (when TT_METAL_SIMULATOR is set)
                                   ▼
                          driver/wormhole/run.sh
                                   │
                                   ▼
                       python3 -m driver.wormhole.server
                                   │
                                   ▼
                            Wormhole device
                            (reads TT_SIM_TRACE_* env vars)
```

Every profiling control is a `TT_SIM_TRACE_*` environment variable.
Set it in the shell **before** invoking your tt-metal binary — UMD
inherits the parent environment, run.sh inherits UMD's environment,
the simulator inherits run.sh's environment, and `Wormhole.__init__`
reads the variables and wires up the right writers automatically.
You don't change any code; you don't change any tt-metal source; the
runtime opts in based on environment alone.

The concrete worked example throughout is
`driver/wormhole/one/src/one` — a tiny tt-metal program that has
BRISC read two vectors from DRAM, add them elementwise, and write
the result back. Build it once with `make`; everything else is just
env vars and shell.

## Prerequisites

You need three things in place once before profiling any run:

1. **`tt-metal` built and exported as `TT_METAL_RUNTIME_ROOT`** — the C++
   example links against `libtt_metal`. See the mesham fork of
   tt-metal for build instructions.

2. **`tt-sim` installed.** From the repo root:

   ```bash
   pip install -e .       # pulls numpy, pyyaml, pynng, flatbuffers,
                          # pyarrow, pyelftools
   ```

   The tracing infrastructure is always compiled in. With no env
   vars set, the bus disables itself and each publish call costs
   ~88 ns (a single attribute check) — leaving tracing off has
   essentially zero runtime impact.

3. **The `./one` binary built:**

   ```bash
   cd driver/wormhole/one/src
   make
   # produces ./one
   ```

For the downstream tools that consume each output, you'll also want
(only the ones you'll actually use):

- `duckdb` — SQL over Parquet outputs.
- `kcachegrind` (or `qcachegrind`) — memory hotspot viewer.
- `genhtml` (from the `lcov` package) — source-coverage HTML report.
- `spike` — only if you want to do RISC-V differential testing.

## Pointing the binary at the simulator

UMD inside tt-metal decides between real silicon and a simulator
based on the `TT_METAL_SIMULATOR` env var. If set to a directory
containing a `run.sh` (the tt-sim repo provides one at
`driver/wormhole/run.sh`), UMD spawns it and connects over an `nng`
IPC socket instead of opening a PCIe device. tt-metal then drives
the simulator with exactly the same API calls it would use against
silicon.

So the always-set-once-per-shell prep is:

```bash
export TT_METAL_RUNTIME_ROOT=$HOME/tt-metal     # wherever you built tt-metal
export TT_METAL_SIMULATOR=$HOME/tt-sim/driver/wormhole
```

The first tells the example where the tt-metal headers/libraries
live (used at build time); the second tells UMD to spawn tt-sim at
run time. After this, every run of `./one` is simulator-backed.

## A 30-second smoke test

Enable everything at once and run the example:

```bash
cd driver/wormhole/one/src

TT_SIM_TRACE=/tmp/out/events.jsonl \
TT_SIM_TRACE_PERFETTO=/tmp/out/timeline.json.gz \
TT_SIM_TRACE_COMMITLOG=/tmp/out/commitlog/ \
TT_SIM_TRACE_COUNTERS=/tmp/out/counters/ \
TT_SIM_TRACE_NOC=/tmp/out/noc/ \
TT_SIM_TRACE_MEMORY=/tmp/out/memory.callgrind \
TT_SIM_TRACE_INVARIANTS=/tmp/out/invariants.jsonl \
TT_SIM_TRACE_STATE_DUMP=/tmp/out/states/ \
./one
```

After the kernel finishes, `ls /tmp/out` shows eight outputs (a
JSONL file, a gzipped Perfetto trace, a commitlog directory, three
Parquet datasets, a Callgrind text file, an invariants log, and a
state-dump directory). The rest of this tutorial unpacks what each
one is for.

The same env vars apply to **any** tt-metal binary — `./one` here is
just convenient because it's small and predictable. Substitute your
own binary at the end of the env-var block and everything else stays
the same.

---

## 1. JSONL event log — `TT_SIM_TRACE`

**What:** Every architectural event the simulator emits, one
JSON object per line. Categories: `instr` (RV instruction
retirement), `dispatch` (Tensix instruction issue), `compute` (per-op
backend completion), `noc` (request/response), `mem` (L1/MMIO
access), `sync` (mailbox/ttsync), `lifecycle` (kernel boundaries).

**Why:** The most flexible output — every other writer is derived
from this same stream. Best for ad-hoc queries with `jq` /
`duckdb` / pandas / Polars when nothing else fits.

**Enable:**

```bash
TT_SIM_TRACE=/tmp/out/events.jsonl ./one
```

Produces two files:

- `/tmp/out/events.jsonl` — the event stream.
- `/tmp/out/events.jsonl.ids.json` — sidecar table mapping every
  `unit_id` tuple in the events to its `(chip_id, core_y, core_x,
  unit)` decomposition. Same scheme is reused by every other
  writer.

**Quick analysis:**

```bash
# How many events per category?
python3 -c "
import json
from collections import Counter
print(Counter(json.loads(l)['category'] for l in open('/tmp/out/events.jsonl')))
"
# Counter({'mem': 55242, 'instr': 14572, 'dispatch': 24,
#          'compute': 7, 'lifecycle': 4, 'noc': 3})

# Which PCs retired most on BRISC?
python3 -c "
import json
from collections import Counter
pcs = (json.loads(l) for l in open('/tmp/out/events.jsonl'))
brisc = [hex(e['pc']) for e in pcs
         if e.get('category') == 'instr' and e['unit_id'][3] == 'BRISC']
print(Counter(brisc).most_common(5))
"
```

The JSONL file scales linearly with the run length — a typical
single-kernel run is ~10 MB.

---

## 2. Visual timeline — `TT_SIM_TRACE_PERFETTO`

**What:** A zoomable timeline rendered in
[ui.perfetto.dev](https://ui.perfetto.dev). One row per Tensix tile
(plus host + DRAM tiles), each split into lanes for BRISC / NCRISC /
TRISC0–2 / MATRIX / SFPU / PACKER / UNPACKER / MOVER / THCON /
NOC0 / NOC1. NoC transactions render as arrows between requesting
and responding NUIs. Lifecycle events show as global markers.

**Why:** The single most useful output for getting a feel for what
the kernel actually does over time. "Why is SFPU idle for so long?"
or "did the unpacker finish before MATH started?" answers itself
visually.

**Enable:**

```bash
TT_SIM_TRACE_PERFETTO=/tmp/out/timeline.json.gz ./one
```

The `.gz` extension turns on gzip compression automatically; Perfetto
loads `.json.gz` natively.

**Analyse:**

1. Open <https://ui.perfetto.dev> in any browser.
2. Drag `/tmp/out/timeline.json.gz` onto the page (everything stays
   local — no upload).
3. Use the **Query (SQL)** tab in the left sidebar for SQL-style
   analysis. Three canned queries in
   `tt_sim/trace/queries/README.md`:
   - Top slices by duration.
   - Per-unit event count.
   - NoC roundtrip latency.

   The Perfetto schema is documented at
   <https://perfetto.dev/docs/analysis/sql-tables>.

**Caveat:** durations are synthetic (`dur = 1` cycle) until the
simulator gains cycle-accurate timing — useful for spatial reasoning
("what's happening on the SFPU around cycle 3000?"), not yet for
absolute performance numbers. Events that don't carry a real cycle
(memory accesses, sync points, lifecycle markers) are stamped with
the highest cycle seen so far so they appear at "current time"
rather than collapsing onto t=0.

---

## 3. Per-core RISC-V commitlog — `TT_SIM_TRACE_COMMITLOG`

**What:** One file per baby core (`brisc.commitlog`,
`ncrisc.commitlog`, `trisc0.commitlog`, …) in the byte-exact format
that the official Spike simulator's `--log-commits` flag produces:

```
core   0: 3 0x00003780 (0xffb001b7) x 3 0xffb00000
core   0: 3 0x00003784 (0x7f018193) x 3 0xffb007f0
core   0: 3 0x00003788 (0xffb01137) x 2 0xffb01000
```

Each line: `core <hart>: <priv> <PC> (<instr>) [x<rd> <value>]`. The
trailing register write is omitted for instructions that don't write
a register (stores, branches, jumps without link, writes to `x0`).

**Why:** Differential testing against Spike — the standard tool for
catching divergences between two RISC-V implementations. If a small
pure-RV ELF runs differently on tt-sim vs. Spike, one of them has a
bug; both outcomes are valuable to surface.

**Enable:**

```bash
TT_SIM_TRACE_COMMITLOG=/tmp/out/commitlog/ ./one
```

**Analyse:**

```bash
# Run the same ELF through Spike
spike --log-commits ./test_kernel.elf > /tmp/spike.log

# Diff with the bundled helper, which reports the first divergence
# with five lines of context
python3 -m tt_sim.trace.diff_spike \
    /tmp/out/commitlog/brisc.commitlog \
    /tmp/spike.log
# files match (4300 lines)
```

**Caveat:** Spike has no concept of Tensix or NoC MMIO, so the
diff is only meaningful for pure-RV32IM ELFs that run identically
under both simulators. Useful for catching RISC-V instruction-level
bugs (especially once cycle/pipeline modelling lands), not for
end-to-end kernel correctness.

---

## 4. Performance counters as time-series — `TT_SIM_TRACE_COUNTERS`

**What:** A Hive-partitioned Parquet dataset
(`chip=<N>/kernel_id=<N>/*.parquet`) of per-unit running counters,
flushed every N cycles and at every kernel boundary. Columns:

```
cycle, chip, kernel_id, core_y, core_x, unit, counter_name, value
```

Counters surfaced today include `instr_retired`, `instr_stalled`,
`dispatch_total`, `dispatch_to_<UNIT>` (one per target backend),
`compute_ops`, `noc_<phase>_<txn_type>`, `noc_bytes_total`,
`mem_<op>_<region>`, `mem_bytes_<op>`, `sync_<kind>`.

**Why:** Per-unit utilisation, per-kernel deltas, hotspot detection.
Long-format Parquet is the lingua franca for SQL/pandas/Polars
performance analysis — DuckDB reads it directly without any ETL.

**Enable:**

```bash
TT_SIM_TRACE_COUNTERS=/tmp/out/counters/ ./one

# Optional: tighten the flush cadence (default 100 cycles)
TT_SIM_TRACE_COUNTERS_INTERVAL=50 ...
```

The aggregator increments `kernel_id` at every `kernel_start`
lifecycle event, so the firmware setup and your kernel show up under
separate partitions.

**Analyse:**

```bash
duckdb -c "
  SELECT counter_name, COUNT(*) AS samples, SUM(value) AS total
  FROM read_parquet('/tmp/out/counters/**/*.parquet',
                    hive_partitioning=true)
  GROUP BY counter_name
  ORDER BY total DESC
  LIMIT 10
"
# Reports e.g. instr_retired=14572, mem_read_L1=31155, etc.
```

Or load straight into pandas:

```python
import pandas as pd
df = pd.read_parquet('/tmp/out/counters/')
df.groupby('counter_name')['value'].sum().sort_values(ascending=False)
```

Four canned DuckDB queries (top counters, per-unit retirement,
kernel-to-kernel diff, NoC hotspot) live in
[`tt_sim/trace/queries/counters.sql`](../../../tt_sim/trace/queries/counters.sql).

**Caveat:** counters that depend on cycle-accurate state (FPU stall
reasons, packer back-pressure, L1 bank conflicts, NoC VC occupancy)
aren't yet populated — they'll appear in the same schema once the
simulator's cycle accuracy work lands.

---

## 5. NoC traffic as a queryable table — `TT_SIM_TRACE_NOC`

**What:** A Hive-partitioned Parquet dataset (`chip=<N>/*.parquet`)
with one row per NoC event emission. Columns:

```
cycle, chip, core_y, core_x, unit,
phase ('request' / 'response'),
txn_type ('read' / 'write'),
src_x, src_y, dst_x, dst_y, size_bytes, txn_id
```

**Why:** "Which link carried the most bytes?" "Which transaction
took longest?" "How many bytes per kernel?" — all are one-liner SQL
queries.

**Enable:**

```bash
TT_SIM_TRACE_NOC=/tmp/out/noc/ ./one
```

**Analyse:**

```bash
# Bytes per (src, dst) pair
duckdb -c "
  SELECT src_y, src_x, dst_y, dst_x, SUM(size_bytes) AS bytes
  FROM read_parquet('/tmp/out/noc/**/*.parquet',
                    hive_partitioning=true)
  WHERE phase = 'response'
  GROUP BY src_y, src_x, dst_y, dst_x
  ORDER BY bytes DESC
"
```

The NoC dataset and the counter dataset (§4) share their unit_id
scheme with the JSONL sidecar (§1), so joins across all three are
straightforward.

---

## 6. Memory hotspots in KCachegrind — `TT_SIM_TRACE_MEMORY`

**What:** A single Callgrind text file consumable directly by
`kcachegrind` / `qcachegrind` / `callgrind_annotate`. Each unique
`(region, pc, address)` is bucketed into a `Dr` (data read) /
`Dw` (data write) count, with PC-keyed functions so KCachegrind's
call-cost tree gives one collapsible entry per executing
instruction.

**Why:** The call-cost-tree UI is the fastest way to spot which
instructions touch memory hottest. Accesses without a PC
(NoC-driven, internal engine) group under a synthetic
`<region>_no_pc` function so they're visible but separated.

**Enable:**

```bash
TT_SIM_TRACE_MEMORY=/tmp/out/memory.callgrind ./one
```

**Analyse:**

```bash
sudo apt install kcachegrind     # or qcachegrind on Mac
kcachegrind /tmp/out/memory.callgrind
```

In the KCachegrind UI, the left pane lists "functions" (here:
`<region>_pc_<hex>`) sorted by `Dr` + `Dw` cost; clicking one shows
the addresses it touched in the right pane.

**Caveat:** today's accesses are bucketed by PC and region, not by
source line — that comes from the LCOV writer (§7). When source-
attribution lands more broadly, the function names here can be
replaced with real symbols without any change to the writer's
contract.

---

## 7. Source-level coverage — `TT_SIM_TRACE_LCOV`

**What:** An LCOV-format coverage file mapping retired-instruction
counts back to kernel source lines, joined via DWARF info from the
ELFs you point at. Consumable by `genhtml`, GitHub Codecov, the VS
Code Coverage Gutters extension, and most CI coverage reporters.

**Why:** This is the headline feature for kernel optimisation —
when the closed simulator can't structurally offer it, tt-sim
shows you which kernel source lines burned the most cycles. Hot
lines stand out as a heatmap on your kernel source view.

**Enable:** (requires kernels built with `-g`):

```bash
TT_SIM_TRACE_LCOV=/tmp/out/coverage.lcov \
TT_SIM_TRACE_LCOV_ELFS=build/brisc_kernel.elf,build/trisc0_compute.elf \
./one
```

`TT_SIM_TRACE_LCOV_ELFS` is a comma-separated list of ELFs whose
DWARF info to load. Last-load-wins on overlapping PC ranges — list
kernel ELFs after firmware ELFs if you want kernel attribution to
take priority on collisions.

**Analyse:**

```bash
# HTML report
sudo apt install lcov
genhtml /tmp/out/coverage.lcov -o /tmp/out/cov_html/
xdg-open /tmp/out/cov_html/index.html
```

Or install [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters)
in VS Code, open your kernel source folder, and it renders the
heatmap inline next to each line of code.

"Coverage" semantics: the per-line count is **cycles spent at this
line**, not "executed at least once". Tools render it identically to
standard coverage but hot lines stand out instead of just covered
lines.

**Caveats:**

- Source files need to live at the paths DWARF embedded into the
  ELFs (typically absolute paths from the build host); copy or
  symlink the source tree to match, or post-process the LCOV file
  with `sed`.
- Stripped ELFs (no `-g`) load cleanly but contribute zero
  attribution. Build kernels you care about with debug info — the
  tt-metal Makefile defaults already include `-g` for kernel ELFs.

---

## 8. Architectural invariants — `TT_SIM_TRACE_INVARIANTS`

**What:** A small catalogue of architectural rules checked over the
event stream; violations are written to a JSONL log. Today's seed
catalogue: PC alignment, memory access alignment, kernel-lifecycle
ordering, NoC request/response pairing.

**Why:** Catches "the kernel produced the right output but did
something nominally illegal along the way" — broken alignment,
orphan NoC responses, etc. A non-empty violations log on a
known-good workload is a real regression.

**Enable:**

```bash
TT_SIM_TRACE_INVARIANTS=/tmp/out/invariants.jsonl ./one
```

For CI use, set `TT_SIM_TRACE_INVARIANTS_STRICT=1` to additionally
raise on the first violation (the simulator will exit non-zero,
which causes UMD to surface the error back to your tt-metal
program):

```bash
TT_SIM_TRACE_INVARIANTS=/tmp/out/invariants.jsonl \
TT_SIM_TRACE_INVARIANTS_STRICT=1 \
./one
```

**Analyse:**

```bash
# Just look — JSONL is human-friendly
cat /tmp/out/invariants.jsonl
# or count by invariant
python3 -c "
import json
from collections import Counter
v = (json.loads(l) for l in open('/tmp/out/invariants.jsonl'))
print(Counter(e['invariant'] for e in v))
"
```

Each violation line includes the invariant name, a message, the
event category and cycle that triggered it, and the offending
`unit_id`.

---

## 9. Cross-run / cross-sim state diffing — `TT_SIM_TRACE_STATE_DUMP`

**What:** A JSON snapshot of the device's relevant state — per
baby-core register files (32 GPRs + PC) and per-NUI counter sets —
captured at every kernel/firmware lifecycle boundary. Schema is
versioned (`schema_version: 1`).

**Why:** Two flavours of use:

1. **Cross-run regression checks** — capture a state dump on a
   known-good commit, capture one on a candidate commit, diff. Any
   divergence is either a regression or a real semantic change.
2. **Cross-simulator differential testing** — the long-term goal is
   to drive both tt-sim and `libttsim.so` (the official closed
   simulator) on the same tt-metal binary and diff state dumps.
   tt-sim ships the comparison primitive today; you provide the
   orchestration (swap `TT_METAL_SIMULATOR` accordingly).

**Enable:**

```bash
TT_SIM_TRACE_STATE_DUMP=/tmp/out/states/ ./one

ls /tmp/out/states/
# firmware_launch_start_0000.json
# firmware_launch_done_0001.json
# kernel_start_0002.json
# kernel_done_0003.json
```

**Analyse:**

```bash
# Diff two state dumps with the bundled helper. Walks recursively
# and pinpoints the first divergence with a readable path.
python3 -m tt_sim.trace.diff_state \
    /tmp/dumps_before/kernel_done_0003.json \
    /tmp/dumps_after/kernel_done_0003.json
# state matches
# OR:
# first divergence:
#   cores.18_18_BRISC.gpr[4]: 3357555520 != 43112
```

**Caveat on determinism:** State at `kernel_done` is non-deterministic
between runs today because the host polling loop on the simulator
side runs `wormhole.run(N)` then checks the go-message mailbox — so
`kernel_done` fires within an N-cycle window after the kernel
actually completes, and BRISC may execute slightly different
firmware-loop bytes in that window. For byte-exact regression
comparison either dump at a deterministic sub-checkpoint or tighten
the polling loop (`TT_SIM_CYCLES_PER_POLL`). The diff tool itself is
exact; the noise is in *when* the snapshot is taken.

---

## Combining outputs

All nine writers are independent subscribers to the same event bus,
so any combination is fine — including all nine at once (see the
30-second smoke test at the top). Per-writer cost is small (~ns per
publish call) and turning a writer off has effectively zero
overhead.

A useful "everything on" wrapper script for routine use:

```bash
#!/usr/bin/env bash
# profile-run.sh — run a tt-metal binary under tt-sim with every
# profiling writer enabled.
set -euo pipefail
OUT="${1:-/tmp/tt-sim-out}"
shift || true   # remaining args are the binary to run
mkdir -p "$OUT"

# Required: where tt-sim's run.sh lives, so UMD spawns the simulator.
: "${TT_METAL_SIMULATOR:?TT_METAL_SIMULATOR must point at driver/wormhole}"

export TT_SIM_TRACE="$OUT/events.jsonl"
export TT_SIM_TRACE_PERFETTO="$OUT/timeline.json.gz"
export TT_SIM_TRACE_COMMITLOG="$OUT/commitlog/"
export TT_SIM_TRACE_COUNTERS="$OUT/counters/"
export TT_SIM_TRACE_NOC="$OUT/noc/"
export TT_SIM_TRACE_MEMORY="$OUT/memory.callgrind"
export TT_SIM_TRACE_INVARIANTS="$OUT/invariants.jsonl"
export TT_SIM_TRACE_STATE_DUMP="$OUT/states/"
# Uncomment if you have ELFs with DWARF:
# export TT_SIM_TRACE_LCOV="$OUT/coverage.lcov"
# export TT_SIM_TRACE_LCOV_ELFS="$OUT/brisc_kernel.elf,..."

echo "profiling artefacts → $OUT"
exec "$@"
```

Use it as e.g. `./profile-run.sh /tmp/myrun ./one` or
`./profile-run.sh /tmp/myrun ./build/programming_examples/loopback`.

## How this works under the covers

`Wormhole.__init__` (in `tt_sim/device/tt_device.py`) is the only
place that reads any of the `TT_SIM_TRACE_*` environment variables,
via the `enable_from_env()` helper in `tt_sim/trace/auto.py`. That
constructor runs once when UMD spawns `run.sh` and `run.sh` execs
the server, well before your tt-metal binary issues its first wire
message. By the time the kernel starts retiring instructions, every
writer you asked for is already subscribed to the bus.

The diagnostic flags from
[`driver/wormhole/README.md#enabling-diagnostics-in-the-tt-metal-flow`](../README.md#enabling-diagnostics-in-the-tt-metal-flow)
(`TT_SIM_DIAG_BRISC`, `TT_SIM_DIAG_NOC0`, the `_CO_*` series for
Tensix coprocessor diagnostics, etc.) compose freely with the
`TT_SIM_TRACE_*` env vars — they're orthogonal: diagnostics print
human-readable text to stderr, trace writers produce machine-readable
artefacts on disk.

A few server-level env vars also apply (defined in
`driver/wormhole/run.sh`):

| Env var | Purpose |
| --- | --- |
| `TT_SIM_RECORD=<path>` | Record every wire message to a file for later replay. |
| `TT_SIM_LOG_PROTOCOL=1` | Print every wire message to stderr. |
| `TT_SIM_CYCLES_PER_POLL=N` | Cycles to run after each wire message (default 100). Tighten for more deterministic state dumps. Lowering it no longer buys wall clock at any grid width — see the last round of this document. |

These compose with the trace env vars too.

## Picking the right output

A rough mapping from question to writer:

| Question | Output |
|----------|--------|
| "What did each core do, in what order, when?" | Perfetto (§2) |
| "Which kernel source line burned the most cycles?" | LCOV (§7) |
| "Are the FPU and packer overlapping?" | Perfetto (§2) |
| "How many bytes did the NoC move?" | NoC Parquet (§5) |
| "Which instruction reads L1 most?" | Cachegrind (§6) |
| "Is the RV32 implementation right?" | Commitlog + Spike diff (§3) |
| "How many of each Tensix op fired?" | Counters Parquet (§4) |
| "Did the kernel violate any alignment rules?" | Invariants (§8) |
| "Does this commit produce the same final state?" | State dumps + diff (§9) |
| "Something weirder than the above" | JSONL (§1) |

## Further reading

- [`tt_sim/trace/README.md`](../../../tt_sim/trace/README.md) —
  developer-side documentation: the event schema, how to add a
  publish call, how to add a new writer.
- [`ROADMAP.md §H`](../../../ROADMAP.md) — the full design /
  history; covers what isn't yet modelled and why (cycle-accuracy-
  gated counters, multi-chip identity, etc.).
- [`tt_sim/trace/queries/`](../../../tt_sim/trace/queries/) — canned
  SQL queries for both the Perfetto and Parquet outputs.
- [`driver/wormhole/server/README.md`](../server/README.md) — how the
  wire-bridge server, UMD hand-off, and `TT_METAL_SIMULATOR`
  spawning fit together.

---

# Appendix: where tt-sim's own wall clock goes

Everything above is about profiling *your kernel*. This appendix is
about profiling *the simulator*, which is what [`ROADMAP.md`
§L](../../../ROADMAP.md) asks for before any JIT / rewrite decision is
taken: "no 'use Numba' or 'rewrite in Cython' decision without trace
data showing where the cycles go". §I (cycle-approximate performance
modelling) is the headline goal and it adds per-cycle work on top of
what is measured here, so the numbers below are the budget that work
has to fit into.

**Measured 2026-08-02** against `0b7e1c6` plus an unrelated dirty tree
(`git diff --stat`: 14 files, +413/-107, concurrent work in
`tensix/backends/{matrix,packer,unpacker}.py`, `tensix/registers.py`,
`network/tt_noc.py`, `arch/`, `optests/`). Machine: 12th Gen Core
i7-12700H, 16 threads, CPython 3.12.13, `TT_SIM_THREADED` unset
(sequential pump). Re-measure before quoting these against a later
tree.

## Method

Workloads are the **Blackhole offline replay guards**
(`driver/blackhole/server/*_replay_test.py`). Each replays a captured
tt-metal wire trace socket-free, in one process, with no tt-metal, no
UMD and no IPC in the measurement — so the wall clock is the
simulator and nothing else. That is confirmed by the numbers: with
module import excluded, 97–99 % of each run is inside
`MultiTileClock.run`.

Three instruments, all driven from throw-away scripts that monkeypatch
rather than edit the tree:

- **`cProfile`/`pstats`** for exact call counts. Note its distortion on
  this workload is large — `six` goes 35.8 s → 86.4 s (2.4×) — because
  the hot code is call-dense, so cProfile *over*-weights the leafiest
  code. Call counts are trusted; time shares are not.
- **A stack-sampling profiler** (2 ms interval, background thread +
  `sys._current_frames`) for undistorted time shares — overhead is
  inside run-to-run noise (`six`: 34.0 s sampled vs 35.8 s not).
  `py-spy` is **not installed** in this environment and nothing was
  installed system-wide to get it; the hand-rolled sampler stands in.
  Samples are attributed twice: to the leaf frame, and to the
  outermost tt-sim subsystem on the stack ("who owns this cycle").
- **Microbenchmarks** (`timeit`) isolating individual per-instruction
  costs, so the writeup can quote nanoseconds and not only percentages.

Simulated cycles are counted by wrapping `MultiTileClock.run`. Note
the replay guards pump in chunks driven by the captured host polling
(mostly `cycles_per_poll` = 100), so cycle counts are the real number
of device cycles the kernel needed.

## Headline: cycles per wall-clock second

| workload | what it is | device cycles | pump wall | **cycles/s** | RV instrs retired |
|---|---|---|---|---|---|
| `two` | NoC/L1 smoke | 8,200 | 2.10 s | 3,900 | — |
| `reduce` | 5 reduction variants | 10,200 | 2.97 s | 3,439 | — |
| `nine` | multi-CB dataflow | 11,500 | 5.87 s | 1,958 | — |
| `sfpumath` | SFPU transcendentals | 19,300 | 7.23 s | 2,671 | — |
| `four` | Int8 add via FPU | 107,400 | 29.51 s | 3,639 | 524,956 |
| `six` | **128³ bf16 matmul** | 27,500 | 35.68 s | **771** | 125,456 |

**tt-sim currently runs at roughly 1–4 k simulated cycles per second**,
dropping to ~770 cycles/s when the Tensix matrix unit is busy. Per
simulated RV instruction that is ~50 µs (`four`: 524,956 instructions
in the 84 % of 29.5 s owned by the RV cores ⇒ ~21 k instr/s), and
~84 M Python function calls for 525 k instructions is **~120 Python
calls per simulated RV instruction**.

## The wall-clock partition

Sampled shares, attributed to the outermost owning subsystem. Two
clearly distinct regimes:

| subsystem | `six` (matmul) | `four` | `nine` | `reduce` | `sfpumath` |
|---|---|---|---|---|---|
| Tensix matrix / FPU | **70.3 %** | 0.1 % | 0.4 % | 2.1 % | 0.8 % |
| RV32IM baby cores | 20.5 % | **84.2 %** | **77.4 %** | **69.7 %** | **65.0 %** |
| Tensix vector / SFPU | — | — | — | — | 11.2 % |
| Tensix unpacker | 2.7 % | 0.5 % | 2.0 % | 3.3 % | 1.5 % |
| Tensix packer | 0.7 % | 0.2 % | 0.4 % | 1.5 % | 0.4 % |
| Tensix frontend (decode) | 1.0 % | 0.9 % | 1.5 % | 3.9 % | 4.5 % |
| NoC (NIU ticks) | 1.8 % | 6.1 % | 5.3 % | 7.2 % | 4.3 % |
| bridge (trace replay, host msgs) | 1.0 % | 5.1 % | 6.6 % | 5.6 % | 4.3 % |
| clock/pump dispatch itself | 0.8 % | 0.4 % | 1.5 % | 0.3 % | 3.9 % |
| trace/event bus (disabled) | 0.1 % | — | — | — | 0.4 % |
| everything else | 1.0 % | 2.5 % | 5.0 % | 6.4 % | 3.7 % |

Read across the row: **outside matmul, the RV32IM interpreter is
two-thirds to five-sixths of the entire simulator.** Inside matmul, one
function — `matrix.py:_fpu_group_sums` — is 42 % of the whole process.

Top leaf frames, `six`:

```
  42.21%  tensix/backends/matrix.py:_fpu_group_sums
   5.31%  tensix/backends/matrix.py:_fpu_accumulate
   3.91%  tensix/registers.py:__getitem__
   3.89%  tensix/backends/matrix.py:<genexpr>
   3.10%  tensix/util.py:FP32ToDstFormatBF16
   2.67%  tensix/registers.py:getDst16b
   2.42%  pe/register/register.py:write
```

Top leaf frames, `four` (representative of the RV-bound majority):

```
  11.17%  pe/register/register.py:write
   8.51%  pe/register/register.py:read
   6.45%  network/tt_noc.py:clock_tick
   5.87%  util/conversion.py:conv_to_int32
   5.63%  pe/rv/isa/rv_isa.py:get_int
   5.08%  pe/rv/isa/rv_isa.py:get_bits
   4.74%  device/clock.py:clock_tick
   3.77%  pe/register/register_file.py:clear_write_record
   3.72%  trace/bus.py:get_bus
   3.48%  memory/memory.py:read
   3.46%  pe/rv/isa/rv_isa.py:<genexpr>
```

## The idle-pump floor, and how it scales

A device with every baby core held in soft reset still has to be
ticked. Running the `driver/blackhole` device (8 DRAM tiles + N Tensix
tiles) with nothing happening:

| Tensix tiles | registered clockables | idle cycles/s | µs/cycle |
|---|---|---|---|
| 1 | 44 | 24,850 | 40.2 |
| 2 | 72 | 15,095 | 66.2 |
| 4 | 128 | 9,230 | 108.3 |
| 8 | 240 | 4,428 | 225.9 |

Dead linear in component count at **~0.94 µs per component-tick**. Two
consequences:

1. **The pump is not today's bottleneck.** On `six` it is 0.8 % of the
   wall clock; on the single-Tensix workloads the 24.8 k cycles/s floor
   is 7–30× above the 0.8–3.9 k cycles/s actually achieved.
2. **It is a hard ceiling, and it moves the wrong way.** Even with all
   modelled work made free, one Tensix tile caps at ~25 k cycles/s, and
   a realistic 8-Tensix device at ~4.4 k cycles/s — *below* what a
   single tile achieves doing real work today.

Where the idle 40 µs/cycle goes (sampled, 1 Tensix tile):

```
  38.5%  network/tt_noc.py:clock_tick     (18 NIUs x empty request list,
                                           each taking self._inbox_lock and
                                           allocating a fresh list)
  13.6%  device/clock.py:clock_tick       (the dispatch loop itself)
 ~28%    the per-core soft-reset poll     (conv_to_int32 7.5%, memory
                                           convert_addr_to_target_range 6.8%,
                                           memory_map.locate 3.4%,
                                           get_nth_bit 2.7%, ...)
```

That last one is worth calling out on its own: `BabyRISCV.clock_tick`
reads `SOFT_RESET` (`0xFFB121B0`) through the **full memory map** —
interval lookup, polymorphic `mem_mapable` dispatch, byte→int
conversion — for all five cores on every cycle, whether or not the core
is running. That is 5 memory-map traversals per tile per cycle of pure
overhead, and it is ~28 % of the idle floor.

## Anatomy of one simulated RV instruction

~50 µs each, ~120 Python calls each. Microbenchmarks of the individual
pieces (200 k iterations, ns per call):

| operation | current | obvious alternative |
|---|---|---|
| `Register.read()` (4-byte numpy `uint8` array → `bytes`) | 511 ns | 30 ns (plain int) |
| `Register.write(bytes)` (`np.frombuffer` + slice assign) | 2,234 ns | 16 ns |
| opcode decode: `get_bits(instr,0,6)` + `.reverse()` + `bits_to_int()` | 4,303 ns | 327 ns (`int.from_bytes & 0x7F`) |
| the two disassembly f-strings passed to `print_snoop` | 1,477 ns | 0 ns (not built) |
| `get_bus().is_enabled(...)` with tracing off | 125 ns | ~0 ns (hoisted flag) |

Four structural findings behind those numbers, all confirmed by call
counts in the cProfile run of `four`:

- **Every GPR is a 4-element numpy array.** 2.41 M `Register.read` and
  1.41 M `Register.write` calls for 525 k instructions; `numpy.frombuffer`
  is called 1.42 M times. numpy's per-call overhead is being paid on
  4-byte scalars, where it is pure loss.
- **The opcode is decoded by building a string.** `RV_I_ISA.run` calls
  `get_bits(instr, 0, 6)` (a 7-element list comprehension), reverses it,
  then `bits_to_int` does `"".join(str(bit) ...)` and `int(s, 2)` —
  530,625 `str.join` calls, one per instruction.
- **Disassembly text is built even when diagnostics are off.** The
  handlers call `RV_ISA.print_snoop(snoop, f"lb {cls.get_reg_name(rd)},
  {hex(offset)}(...)", f"...")` — Python evaluates the f-strings *before*
  the call, so the strings and the `get_reg_name` lookups happen
  unconditionally and are then discarded. `get_reg_name` shows up at
  3–4 % of total wall clock on the RV-bound workloads.
- **Tracing bookkeeping runs with tracing off.** `RegisterFile` installs
  a Python closure over every `Register.write` (1.41 M extra calls), and
  `rv32.clock_tick` calls `clear_write_record()` (3.8 % of wall clock on
  `four`) plus `get_bus()` (3.7 %) on every instruction.

## Anatomy of one MVMUL

`six` issues **4,096 MVMULs** and spends ~70 % of 35.7 s in them ⇒
**~6.1 ms per MVMUL**. Each one calls `_fpu_group_sums` 128 times
(524,288 calls total, ~29 µs each) and `_fpu_accumulate` 128 times.

`_fpu_group_sums` is a faithful model of the hardware's fixed-point
datapath: 16 fidelity-masked mantissa products, then two 8-lane
exponent-aligned integer reductions. It is written with Python lists,
`range`, `max`/`min` over generators (2.1 M `max` and 8.4 M `min` calls
in the profile) and arbitrary-precision ints.

**It is not numpy.** §L's Numba bullet assumes the Tensix numeric
inner loops are "already numpy" and predicts a 2–3× win on top; that
assumption does not hold for the matrix path. The values involved all
fit comfortably in `int64` (mantissa products ≤ 2²², exponents 8-bit),
so the loop *is* `@njit`-able with explicit typing — but the expected
win is one to two orders of magnitude, not 2–3×, because the baseline
is interpreted scalar Python, not vectorised numpy.

## Tracing overhead

Measured on the same workloads by enabling the writers and re-running
(pump-only wall clock, so import and trace-file parsing are excluded):

| workload | off | `TT_SIM_TRACE_COUNTERS` (interval 100) | interval 1000 | `TT_SIM_TRACE` (JSONL) |
|---|---|---|---|---|
| `reduce` | 2.97 s | 5.98 s (**2.0×**) | 5.47 s (1.8×) | 11.05 s (**3.7×**) |
| `nine` | 5.87 s | 12.03 s (**2.0×**) | 12.08 s (2.1×) | 23.32 s (**4.0×**) |
| `six` | 35.68 s | 45.15 s (**1.27×**) | 45.38 s (1.27×) | 57.48 s (1.6×) |

Perfetto (`TT_SIM_TRACE_PERFETTO`) measures the same as counters
(1.9–2.4×).

- Counter tracing costs **~2× on RV-bound workloads**, and only ~1.27×
  on `six` — because `six`'s time is inside one Tensix op that publishes
  a handful of events, while the RV-bound runs publish per instruction
  and per memory access.
- **`TT_SIM_TRACE_COUNTERS_INTERVAL` barely matters** (interval 1000 is
  within noise of interval 100, in both directions across repeats). The
  cost is `EventBus.enabled = True` turning on publication everywhere,
  not the flush cadence. Tuning the interval is not a lever.
- Output size is the real differentiator: the counter Parquet dataset
  for `nine` is **40 KB**; the JSONL for the same run is **61 MB** and
  the Perfetto JSON **11 MB**. Counters are the only writer that is
  plausibly usable at kernel scale — a 2× slowdown on a run that already
  takes an hour is a different proposition, but the dataset stays small.
- **Caveat: `TT_SIM_TRACE_*` is Wormhole-only.**
  `tt_sim.trace.auto.enable_from_env` is called from
  `tt_sim/device/wormhole.py` and nowhere else, so a `Blackhole` device
  silently ignores every trace env var. The measurements above were
  obtained by calling `enable_from_env()` explicitly from the harness.
  Wiring it into `Blackhole.__init__` is a one-line fix and is a
  prerequisite for using any of §H's observability on Blackhole.
  (Since fixed — `enable_from_env` is called from `TT_Device.__init__`,
  guarded by `tt_sim/device/parity_test.py`, so the numbers below were
  taken on Blackhole with no harness patching.)

### What §H's cycle-attributing fields cost (2026-08-03)

Measured when ROADMAP §H's "gated on §I" observability landed (real
Perfetto durations, NoC `issue_cycle`/`arrival_cycle`, RV stall
attribution, per-unit `busy_cycles`). Method as elsewhere in this
document: **two frozen full-tree copies** — the change, and the same
tree with only the change reverted — alternating order per round,
**minimum of 5 rounds**, end-to-end wall clock of
`driver.blackhole.server.four_replay_test`. The Perfetto row is a
separate 3-round A/B re-run after the output-size fix below, so that it
measures the writer that shipped.

| config | before | after | ratio |
|---|---|---|---|
| tracing off | 11.53 s | 10.92 s | **0.95** |
| `TT_SIM_TRACE_COUNTERS` | 25.42 s | 28.81 s | 1.13 |
| `TT_SIM_TRACE_PERFETTO` | 31.60 s | 30.41 s | **0.96** |
| `TT_SIM_TRACE_COUNTERS` + `TT_SIM_COST_MODEL=1` | 22.49 s | 24.12 s | 1.07 |

- **Nothing measurable off the tracing path.** The after tree measured
  *faster* with tracing off, which is the only honest reading of "no
  cost": the added work is inside `if trace_instr:` and inside the
  writers. The one exception is a `self._current_cycle()` per NoC
  `transmit` (two attribute reads, 24 times on this workload).
- **Single-digit-percent on the tracing path**, and the 1.13 is the
  weakest of the four numbers: the `before`-minimum for that config came
  from the only round that ran on a quiet machine. Restricted to
  rounds 1–4, where both trees saw the same contention, the same config
  is **1.07**. Two other agents were running gates on this machine
  throughout, which the interleave-and-take-the-minimum protocol bounds
  but does not remove.
- **Perfetto output size is unchanged**: 65,828,313 B after vs
  65,805,799 B before (+0.03 %). It was +9 MB (+13.6 %) in a first cut
  that put `"stall_cycles": 0` in the args of every instruction slice —
  ~16 bytes of zero on the highest-volume event in the trace, on a run
  where nothing can stall. The arg is now emitted only when there was a
  stall, which is also the only time it says anything. Before that fix
  the same A/B measured 1.09; after it, 0.96 — the writer's extra work
  was never the cost, the extra bytes were.
- The headline from the previous section stands: **the cost is
  `EventBus.enabled`, not what the writers do with the events.** These
  fields ride along inside a 2–4× that was already being paid.

## How far is a 640³ matmul?

`six` is 128³ bf16 = 4×4×4 tiles = 64 tile-matmuls = 4,096 MVMULs in
35.7 s. `matmul_single_core` at 640³ is 20×20×20 = 8,000 tile-matmuls =
**512,000 MVMULs**, 125× more work on the same single-core code path.
Scaling the measured 6.1 ms/MVMUL:

- MVMUL alone: 512,000 × 6.1 ms ≈ **3,100 s ≈ 52 minutes**
- with the RV dataflow that feeds it scaling similarly: **~1.5–2 hours**
- cross-check via cycles: ~125 × 27,500 ≈ 3.4 M device cycles at 771
  cycles/s ≈ 4,400 s. Same order.

So §L's "already times out" is not marginal — it is off by two orders
of magnitude from interactive. To run 640³ in **one minute** needs
~57,000 cycles/s, a **~75× speedup**; that is above even the *idle*
single-tile pump ceiling of 24.8 k cycles/s, so no amount of making the
modelled work cheaper gets there on its own. A "tolerable" ten-minute
run needs ~10×, which is reachable without touching the pump.

## Ranked optimisation targets

Ranked by (measured share × confidence that the fix is mechanical).
This section was written as the measurement half of §L's "profile
first, optimise second"; **targets 1 and 3 have since been done** — see
"[What landed: targets 1 and 3](#what-landed-targets-1-and-3)" at the
end of this document for the changes and the measured result.

1. **RV32IM instruction interpreter — GPR representation and decode.**
   65–84 % of wall clock on every non-matmul workload. Four independent
   sub-targets, each backed by a microbenchmark above: numpy-backed
   4-byte GPRs (0.5 µs read / 2.2 µs write vs ~20 ns for ints); the
   string-join opcode decode (4.3 µs vs 0.33 µs); unconditionally-built
   disassembly f-strings (1.5 µs, and 3–4 % of total wall clock in
   `get_reg_name` alone); and the always-on tracing bookkeeping
   (`clear_write_record` 3.8 %, `get_bus` 3.7 %, the `wrapped_write`
   closure 1.4 M calls). These are ~12 µs of the ~50 µs per instruction
   and none of them requires a JIT — they are pure-Python wins, which
   makes them the cheapest thing on this list.
2. **`matrix.py:_fpu_group_sums` / `_fpu_accumulate`.** 47.5 % of the
   matmul workload in two static methods with no I/O, no state, and
   int64-representable arithmetic. Best Numba target in the codebase by
   a distance; also the one place where a numpy rewrite (16 products as
   a vector op) is plausible.
3. **The per-cycle soft-reset poll in `BabyRISCV.clock_tick`.** ~28 % of
   the idle floor, five full memory-map traversals per tile per cycle,
   for a register that changes a handful of times per run. Caching it,
   or having the reset write invalidate a flag, is a contained change.
4. **`NUI.clock_tick` when idle.** 38 % of the idle floor and 4–7 % of
   real workloads: 18 NIUs per tile-pair each take a `threading.Lock`
   and allocate a list every cycle, in a mode (`TT_SIM_THREADED`) that
   is off by default. Early-out when both queues are empty.
5. **`MemoryMap` interval lookup / `MemorySpace.read`.**
   `convert_addr_to_target_range` + `locate` + `_publish_mem_event` are
   3–7 % across workloads and 1.27 M calls on `four`. §L correctly flags
   this as Numba-hostile; the win here is caching the last-hit range,
   not JIT.
6. **The clock pump itself.** 0.3–1.5 % today, so *not* worth touching
   for its own sake — but it is a ceiling that degrades linearly with
   tile count (§I's multi-tile ambitions and §A's threading both push
   straight into it). Fix it when §I lands, not before.

## What this says about §L's own hypotheses

- **"The tick-every-component pump is the wrong shape"** — *supported
  as a ceiling, contradicted as today's bottleneck.* The pump is
  0.3–1.5 % of current wall clock; rewriting it now would be invisible.
  But it caps a single-Tensix device at ~25 k cycles/s and an
  8-Tensix device at ~4.4 k cycles/s, which is below the target for any
  kernel-scale workload. §L's sequencing ("land the event-driven pump
  *before* JIT'ing") is right for a different reason than stated: not
  because the pump is slow, but because the JIT targets in the current
  shape (per-component `clock_tick`) are not the JIT targets in the
  event-driven shape.
- **"Numba on the Tensix numeric inner loops; already numpy so expect
  2–3×"** — *target supported, rationale contradicted.*
  `_fpu_group_sums` is scalar interpreted Python over lists and
  arbitrary-precision ints, not numpy. The upside is therefore much
  larger than 2–3× — but it also means the cheaper move (rewrite the
  16 products as a numpy/int64 vector op, no JIT, no new dependency)
  should be tried first, and Numba evaluated against *that* baseline.
- **"RV32IM execute — only if rewritten table-driven over typed
  arrays; significant refactor"** — *supported, and under-prioritised.*
  RV32IM is the single largest consumer of wall clock across the suite
  (65–84 % on four of the five workloads measured), well ahead of the
  Tensix backends outside matmul. §L lists it second behind the Tensix
  loops; the data says it should be first. It also does not need the
  full table-driven-over-typed-arrays refactor to pay: the four
  sub-targets in item 1 above are local edits worth an estimated ~25 %
  of RV time on their own.
- **"MemoryMap lookup is the most-called function in the sim"** —
  *contradicted on call count, supported on cost class.* On `four` the
  most-called tt-sim functions are `Register.read` (2.41 M) and
  `RegisterFile.get` (2.83 M); `memory_map.locate` is 1.27 M. It is
  still 3–7 % of wall clock and still Numba-hostile, so the conclusion
  drawn from it (don't JIT it) stands.
- **`nogil=True` to revive §A threading** — *no new evidence either
  way, but note the arithmetic.* On the matmul workload 70 % of the
  time is in one Tensix unit on one tile, so releasing the GIL there
  helps only multi-tile runs; and the barrier those runs must cross is
  the ~0.94 µs/component-tick pump, which threading does not remove.

## Reproducing

The harnesses used here are deliberately throw-away (they monkeypatch;
they do not modify the simulator). To repeat the baseline:

```bash
export PYTHONPATH=~/tt-sim
# wall clock + simulated cycles for any replay guard
time python3 -m driver.blackhole.server.six_replay_test

# exact call counts
python3 -c "
import cProfile, pstats, importlib
m = importlib.import_module('driver.blackhole.server.six_replay_test')
cProfile.run('m.main()', '/tmp/six.prof')
pstats.Stats('/tmp/six.prof').sort_stats('tottime').print_stats(30)"

# tracing overhead: same run, writers on (Wormhole; on Blackhole call
# tt_sim.trace.enable_from_env() by hand first — see the caveat above)
TT_SIM_TRACE_COUNTERS=/tmp/counters \
  python3 -m driver.wormhole.server.offline_replay_test
```

For the undistorted time shares, a sampling profiler is required —
`cProfile` inflates this workload 2.4× and skews toward call-dense
code. `py-spy record -o out.svg -- python3 -m
driver.blackhole.server.six_replay_test` is the tool of choice if it is
available; it was not installed here.

# What landed: target 2, batching the FPU accumulate datapath

**Measured 2026-08-02**, same machine as the appendix above (12th Gen
Core i7-12700H, CPython 3.12.13, `TT_SIM_THREADED` unset). The tree was
busy with other concurrent work, so every number here is an A/B between
two git worktrees — `HEAD` and `HEAD` plus this change only — run
**interleaved** (base, then after, per workload) and reported as the
**minimum of 3–8 repeats**, because contention can only add time. An
earlier non-interleaved sweep showed every workload "improving",
including ones that never enter this code; that pass is discarded.

## What changed

`MatrixUnit._fpu_group_sums` / `_fpu_accumulate` are unchanged and are
now the *reference*: they stay the readable port of ttsim's C and the
oracle the tests fuzz against. `perform_mvmul_exact` instead calls two
new batched methods, `_fpu_group_sums_batch` / `_fpu_accumulate_batch`,
which are the same arithmetic in numpy over a whole MVMUL at once
(16 lanes × up to 16 SrcB rows × 16 Dst columns). This is legal because
nothing in an MVMUL sequences: each of the 256 (row, column)
accumulations reads and writes its own Dst element.

No JIT, no new dependency. §L's premise that this code was "already
numpy" was wrong (see above); a plain numpy/int64 rewrite was tried
first, as the appendix recommended, and it is enough.

## Result

| workload | uses this path | base | after | factor |
|---|---|---|---|---|
| `six` (128³ bf16 matmul) | 70 % of wall clock | 30.87 s | **16.78 s** | **1.84×** |
| `matmulidx` | matmul | 5.53 s | **3.93 s** | **1.41×** |
| `reduce` (GAPOOL) | 2 % | 2.83 s | 2.84 s | 1.00× |
| `four`, `nine`, `two`, `eight`, `three`, `five`, `sfpumath`, … | no | — | — | within ±7 % noise, no direction |

`six` is now ~1,640 device cycles/s, up from ~890 on this machine.

The datapath itself, timed in one process against the scalar pair on
identical inputs (min of 7 × 100 iterations, one MVMUL = 8 rows × 16
columns × 16 lanes = 128 group-sum calls):

| implementation | µs per MVMUL | µs per group-sum call |
|---|---|---|
| scalar (the reference) | 4,991 | 39.0 |
| batched, first cut | 798 | 6.2 |
| batched, final | **516** | **4.0** |

**9.7× on the two functions**, which is why a 47.5 % share turns into a
1.84× workload win rather than a 1.9× one — Amdahl, plus the operand
gather that is now the larger half of an MVMUL (below).

Two numpy details were worth 1.55× between the first cut and the final
one, and are worth knowing before optimising any other backend:

- **Reduce over a *leading* axis, never a trailing one.** The lane axis
  is what gets reduced, so the arrays are laid out lane-first.
  `y.max(axis=1)` on `(2, 8, rows, cols)` is **4.2 µs**; the identical
  reduction written as `y.max(axis=-1)` on `(rows, cols, 2, 8)` is
  **21.7 µs**. numpy reduces a leading axis as one vectorised pass per
  index over contiguous blocks; a trailing one becomes an interpreted
  loop over every outer element.
- **`np.clip` is ~3× `np.minimum(np.maximum(...))`** (9.5 µs vs 3.6 µs
  on these shapes), and with only two groups, combining them by index
  (`np.maximum(g[0], g[1])`, 2.8 µs) beats any reduction call.

Broadcasting is the remaining tax: the one op that broadcasts SrcA
against SrcB costs 16.0 µs where the same shape contiguous costs 6.1 µs.
At these array sizes (2,048 and 256 elements) numpy is overhead-bound at
roughly 3–10 µs per operation, so the ~55 operations are close to the
floor for this approach. Getting materially below 516 µs/MVMUL would
need a JIT — and that is now a ~30 % of `six` target, not a 47.5 % one,
so it should be re-argued against the new baseline rather than assumed.

## Bit-identity

Required, not approximated: this code feeds a bit-exact differential
against the vendor simulator.

- `tt_sim/pe/tensix/fpu_accumulate_test.py` keeps pinning the **scalar**
  pair against the vectors generated from ttsim's C model for both
  architectures, and gains two fuzz tests that assert the batched pair is
  **equal**, not close, to the scalar one: 8,000 lane sets across all
  four fidelity phases, and 16,000 accumulations (half from the real
  datapath, half synthetic and biased toward tiny mantissas, since a
  magnitude of exactly 1 is what trips the Wormhole −1 renormalisation
  and a uniform draw never produces one). 203 tests pass.
- `driver.wormhole.server.offline_replay_test` reproduces **126/126**
  host READs bit-for-bit, which is the end-to-end statement.
- All 16 Blackhole replay guards pass, as do the 11 Wormhole example
  replays.
- The live differential against the vendor simulator still matches on
  both FPU programs: `./optests/diff.sh reduce` and
  `./optests/diff.sh matmulidx`, 2,560 elements each, PASS.

The overflow argument, which is the thing a numpy rewrite can get wrong
silently where Python bigints would not: the fidelity slices cap manA at
31 and manB at 127 (checked exhaustively over all 1,024 mantissa
patterns × 4 phases), so a product is < 2¹²; the alignment shift is
clamped to 30, so `(man << 1) + (1 << shift)` < 2³¹ and the shifted-down
term never exceeds the product; eight of those sum to < 2¹⁵ and
`<< 13` to < 2²⁸; three aligned terms plus a ≤ 2²⁴ Dst mantissa stay
< 2³⁰. Everything is int64 with ~33 bits of headroom, and < 2³⁰ is also
what makes `np.frexp` on the float64 view an exact `bit_length`.

## What did not pay off

- **Vectorising each `_fpu_group_sums` call on its own**, i.e. 16 lanes
  per numpy call. At ~1.5 µs of ufunc overhead × ~25 operations that is
  roughly the 29 µs it replaces. The win comes entirely from batching the
  whole instruction; anything narrower is a wash.
- **Keeping the natural `(row, column, lane)` layout.** It reads better
  and costs 1.55× (see above).
- **`np.where(cond, x, 0)` → `x * cond`** and `np.int64(1) << shift` →
  a power-of-two lookup table: both measured 20–30 % faster on the
  individual operation, which is under 1 % of the MVMUL, and both make
  the correspondence with ttsim's C harder to see. Not taken.

## The new shape of an MVMUL, and the next target

With the datapath at 516 µs, the **operand gather is now the larger half
of an MVMUL**. Per MVMUL `perform_mvmul_exact` still makes ~384 scalar
`SrcRegister.__getitem__` calls and ~1,150 `DataFormatConversions` calls
(`BF16InSrcToFP32` → `BF16InSrcToBF16` → `TF32InSrcToTF32`, three Python
frames per lane), plus 256 `getDst16b`/`setDst16b` pairs. In the
post-change cProfile of `six` those are the top tt-sim leaves after the
batch methods themselves.

That is a contained follow-up — `SrcRegister` already stores a numpy
array, so the gather can be a slice — but it duplicates the bit
permutations that live in `util.py`, so it wants an exhaustive
equivalence test (the Src word is only 19 bits) rather than a copied
expression. It was left out of this pass deliberately: it is a different
target from the one the ranking above named. (It has since been done — see
"[What landed: the Tensix operand gather](#what-landed-the-tensix-operand-gather)"
at the end of this document.)

Revised ranking after this change: item 1 (the RV32IM interpreter) is
now the largest consumer on *every* workload including `six`, item 2 is
retired, and the Src/Dst gather above enters roughly where item 2 was.

# What landed: targets 1 and 3

**Measured 2026-08-02**, same machine and method as the appendix above
(12th Gen Core i7-12700H, CPython 3.12.13, `TT_SIM_THREADED` unset).
The working tree had moved on since the baseline measurements — concurrent
work in `tensix/backends/{matrix,vector}.py` had already roughly halved
`six` — so every number here comes from an **interleaved A/B**: the same
tree, with only `tt_sim/pe/rv/` + `tt_sim/pe/register/` swapped between
the pre-change and post-change versions, alternating variants round by
round and taking the minimum over rounds. Absolute "before" figures
therefore differ from the tables above; the ratios are the point.

## Result

| workload | before (pump) | after (pump) | speedup | before cyc/s | after cyc/s |
|---|---|---|---|---|---|
| `two` | 1.43 s | 0.64 s | **2.23×** | 5,750 | 12,809 |
| `reduce` | 2.25 s | 1.06 s | **2.12×** | 4,534 | 9,614 |
| `nine` | 4.31 s | 1.85 s | **2.33×** | 2,669 | 6,232 |
| `sfpumath` | 5.51 s | 2.82 s | **1.96×** | 3,503 | 6,852 |
| `four` | 27.26 s | 10.35 s | **2.64×** | 3,939 | 10,381 |
| `six` (matmul) | 17.33 s | 12.80 s | **1.35×** | 1,587 | 2,148 |

A second, independent A/B run on a more loaded machine reproduced
1.90–2.87× on the five non-matmul workloads. `six` gains least because
only ~20 % of it was ever RV; that ~20 % is now ~5 %.

**The RV-bound workloads went from 2.7–5.8 k to 6.2–12.8 k simulated
cycles/s.** For reference, the idle-pump ceiling for one Tensix tile
measured above is 24.8 k cycles/s — the interpreter is now within 2–4×
of the pump floor rather than 7–30× below it, which moves targets 4
(`NUI.clock_tick`) and 6 (the pump) up the list.

## The changes

Six edits, all in `tt_sim/pe/rv/` and `tt_sim/pe/register/`. Per-change
figures are single-run, taken in sequence during development (so they
carry the machine's ±15 % run-to-run noise), on `four` / `nine`:

1. **GPRs are `bytes`, not 4-element numpy arrays**
   (`pe/register/register.py`). `read()` returns the stored object with
   no copy; `write()` is an attribute store. `four` 31.3 → 24.2 s,
   `nine` 5.4 → 3.9 s.
2. **Fetch once per cycle; ISAs decode an integer word**
   (`rv32.clock_tick` + every `isa/*.py`). `run()` gained an `instr`
   parameter carrying the 32-bit word (defaulting to `RV_ISA.fetch()`
   so direct callers and unit tests are unaffected). This kills the
   `get_bits` → `reverse` → `bits_to_int` string-join opcode decode
   *and* the repeated `int.from_bytes` inside every `get_int`, *and*
   the duplicate instruction fetch each ISA in the chain used to do.
3. **`RegisterFile` cheapened**: `get()` indexes the list directly and
   falls back to the name map on `TypeError`; `__getitem__ = get`
   removes a call per access; the tracing write-hook closure is now
   installed only while an `INSTR` subscriber is listening, and
   `clear_write_record()` / `get_bus()` no longer run per instruction
   (the bus is held on the core). (2)+(3) together: `four` 24.2 →
   19.3 s, `nine` 3.9 → 3.4 s.
4. **The soft-reset poll resolves its component once**
   (`babyriscv._read_soft_reset`). `0xFFB121B0` is looked up through
   the memory map on the first tick and read directly thereafter, and
   the bit test is a precomputed mask instead of `get_nth_bit`. This
   was target 3. `four` 19.3 → 14.4 s, `nine` 3.4 → 2.9 s — much larger
   than the "~28 % of the idle floor" estimate suggested, because the
   poll was **half of every `MemorySpace.read` in the simulator**
   (on `nine`: 115,000 of 241,307 calls).
5. **Disassembly is built only when snooping.** Every `print_snoop`
   call site is wrapped in `if snoop:`, and the `info_msg` f-strings
   are assigned inside `if snoop:` blocks. `print_snoop` still takes
   and re-checks the flag; its docstring now explains why the callers
   guard anyway (Python evaluates the f-string before the call).
   `four` 14.4 → 11.8 s, `nine` 2.9 → 1.8 s — **the single biggest
   win on the list**, and the one that cost the least.
6. **`Register.read_uint()` / `read_int()`** replace
   `conv_to_uint32(reg.read())` at 50 call sites. Worth 1.10–1.21×,
   which needed six interleaved rounds to resolve — see "noise floor"
   below.

Microbenchmarks, 200 k iterations (compare with the table in "Anatomy
of one simulated RV instruction"):

| operation | before | after |
|---|---|---|
| `Register.read()` | 294 ns | 72 ns |
| `Register.write(bytes)` | 2,457 ns | 151 ns |
| opcode decode | 4,980 ns | ~0 (a mask on the fetched word) |
| `get_int` on one field | 1,049 ns | 370 ns |
| the two disassembly f-strings | 1,722 ns | 0 ns (not built) |

Call counts on `nine` (cProfile, so inflated but comparable):
total calls **15.10 M → 7.69 M**; `MemorySpace.read` 241,307 →
119,944; `RegisterFile.get` 485,186 → 207,607 (plus 485,186
`__getitem__` frames gone); `str.join` 90,995 → 0; `wrapped_write`
240,358 → 0; `clear_write_record` and `get_bus` → 0.

## Behaviour deltas (all deliberate, all verified)

Gates: `ruff` clean; `pytest tt_sim driver` 207 passed; 16/16 Blackhole
replay guards; `driver.wormhole.server.offline_replay_test` **126/126
byte-identical**; `examples_replay_test` 11 passed. Snoop output
(`driver/simple/ex2`, `ex5` with `snoop=True`, covering
add/addi/andi/auipc/beq/bge/ebreak/fence/fence.i/jal/jalr/lui/lw/**mul**/slli/sw)
diffs clean against the old code.

Three things do change, none of them architectural state:

- **Uninitialised GPRs now read as zero.** The old `Register` used
  `np.empty`, so a register read before its first write returned
  whatever was in the freed numpy page. That was *non-deterministic*:
  running the Wormhole replay twice under `TT_SIM_TRACE_COMMITLOG`
  produced different values for `x22`/`x24`/`x25` in one BRISC function
  epilogue (`0x0003633b` vs `0xc021ace0`, `0x2ae1ace0` vs
  `0xc021ace0` — leaked host-process pointers). Those three lines are
  the *only* commitlog difference across all five cores, and they are
  now stably zero.
- **The mem-event stream loses the soft-reset polls and the duplicate
  instruction fetches.** Both were simulator artefacts that no hardware
  bus transaction corresponds to; on `nine` they were half of all
  memory-space reads, so `TT_SIM_TRACE`/counter output gets both
  smaller and more faithful.
- **The register protocol grew `read_uint`/`read_int`.** Anything
  standing in for a `Register` must implement them — in-tree that is
  the four `_Reg` fakes in `isa/*_test.py`, which were updated.

## What did not pay off, and other notes

- **`__slots__` on `Register`** looked free and is not compatible with
  the tracing write hook, which installs a per-instance `write`
  attribute shadowing the method. Dropped rather than restructure the
  hook; plain attribute access was not measurably slower.
- **Speeding up `util/conversion.py` itself** was rejected: the
  helpers are used by every subsystem, so it is a wide blast radius
  for a win that change 6 gets locally on the path that actually cares.
  `conv_to_uint32(b)` is 435 ns against 153 ns for a bare
  `int.from_bytes` — two thirds of that is the wrapper and the nested
  `conv_to_int32` call. Worth revisiting as its own change.
- **The noise floor matters at this point.** With two other agents
  working on the same machine, single runs of `nine` varied 1.8–2.7 s
  (±20 %). Change 6 (1.1×) is *below* that; it took six interleaved
  A/B rounds with min-of-rounds to show it is real. Any further RV
  micro-optimisation needs the same discipline or it is unfalsifiable.
- **Readability tension, declared.** Two of the six changes trade
  some directness for speed: `if snoop:` guards around calls to a
  function that already checks `snoop` (documented in
  `RV_ISA.print_snoop`'s docstring), and `run()` taking a
  pre-fetched word instead of reading the PC itself. The second is
  arguably a clarity *gain* ("the core fetches, the ISA decodes") and
  the `instr=None` default keeps every ISA runnable standalone. What
  was **not** done: no table-driven dispatch, no inlined bit-field
  arithmetic replacing `RV_ISA.get_int`, no hand-unrolled handlers.
  Those are the next ~10–15 %, and they are where the ISA modules
  would stop reading like the RISC-V spec.
- **Still on the table inside the interpreter:** `MemorySpace.read` is
  now the largest single non-pump cost on the RV path (119,944 calls on
  `nine`, with `_publish_mem_event` and `memory_map.locate` under it) —
  that is target 5, unchanged. `conv_to_bytes` on the register-write
  path (150,180 calls on `nine`) is the mirror image of change 6 and
  cannot use the same trick without bypassing the tracing write hook.

## Reproducing these numbers

```bash
export PYTHONPATH=~/tt-sim
# per-workload wall clock; wrap MultiTileClock.run to get pump-only time
time python3 -m driver.blackhole.server.four_replay_test

# the A/B: snapshot tt_sim/pe/rv + tt_sim/pe/register at the two commits,
# swap them into the same tree, alternate variants, take min over rounds.
# Do not compare across sessions — the rest of the tree moves.
```

# What landed: target 6, an event-driven cycle pump (phase 1)

**Measured 2026-08-02**, same machine as the appendix above (12th Gen Core
i7-12700H, CPython 3.12.13, `TT_SIM_THREADED` unset). Design and the remaining
phases: [`docs/plans/event-driven-pump.md`](../../../docs/plans/event-driven-pump.md).

This is ranked target **6** ("the clock pump itself"), which the appendix
explicitly said was *not* worth touching for its own sake — 0.3–1.5 % of wall
clock — but was a ceiling that degrades linearly with tile count. Both halves
of that judgement held up: the pump was not a bottleneck, and removing the
ceiling is nonetheless worth 13–32× on the floor and 1.05–1.45× on real
workloads. Target **4** (`NUI.clock_tick` when idle) is folded in here rather
than fixed separately, as the appendix suggested.

## What changed

The pump stops ticking a tile that has nothing to do. `clock_tick(cycle_num)`
is untouched everywhere, and so is tick *order* — the only change is **who
gets ticked and when**.

- `Clockable` gains an optional `is_clock_idle()`, defaulting to `False`
  (i.e. "not idle"), so a component that has not opted in keeps its tile awake
  forever. Every override is the negation of a guard already present in that
  component's own `clock_tick`.
- `TileClock` (a `Clock` subclass, so `MultiTileClock.run` is unchanged) skips
  its whole component list while dormant. It falls asleep when the tile's
  `clock_quiescent()` probe passes, evaluated *after* the tick; it wakes on
  the three things that can act on a dormant tile from outside — `NUI.transmit`,
  a host read/write through `TT_Device`, and a reset.
- Fifteen components opted in: the NIUs, the baby cores, TDMA, the Tensix
  backend units (base predicate plus five that carry extra state), and the
  three frontend units. `matrix.py` / `vector.py` / `packer.py` / `misc.py`
  needed nothing — they do not override `clock_tick`, so the base predicate is
  complete for them.
- Independently, `NUI.clock_tick` early-outs when both queues are empty
  (target 4): **511 ns → 162 ns**, and it still matters after the tile gate,
  because a *live* Tensix tile's two NIUs are idle on most cycles.

## Result: the idle floor

A `driver/blackhole` device (8 DRAM tiles + N Tensix tiles) with every baby
core held in soft reset — the same benchmark as "The idle-pump floor, and how
it scales" above. Best of 4 interleaved rounds over two **frozen tree
snapshots**, alternating which variant runs first each round (see "Method
note" below — this matters).

| Tensix tiles | clockables | base cycles/s | after cycles/s | factor | base µs/cyc | after µs/cyc |
|---|---|---|---|---|---|---|
| 1 | 44 | 43,371 | **575,749** | **13.3×** | 23.06 | 1.74 |
| 2 | 72 | 30,933 | **529,901** | **17.1×** | 32.33 | 1.89 |
| 4 | 128 | 18,304 | **427,742** | **23.4×** | 54.63 | 2.34 |
| 8 | 240 | 10,413 | **335,141** | **32.2×** | 96.03 | 2.98 |

(The base column is faster than the original appendix table because targets 1
and 3 landed in between; 0.40 µs per component-tick now, not 0.94.)

**The scaling shape is the result, more than the factor.** The floor was dead
linear in *component* count. It is now flat at **0.19 µs per tile-clock** —
9 tile-clocks cost 1.74 µs/cycle, 16 cost 2.98 µs/cycle — i.e. linear in
*tiles* and independent of what is on them. A realistic 8-Tensix device now
idles at 335 k cycles/s where it used to idle at 10 k, which is above the
~57 k cycles/s a one-minute 640³ matmul would need rather than an order of
magnitude below it.

## Result: real workloads

Blackhole offline replay guards, pump-only wall clock. Six interleaved rounds
over the frozen snapshots with alternating order; both min-of-rounds and
median-of-rounds are given because the noise floor on this machine is ±15 %
and the two statistics disagree by more than the effect on `nine`.

| workload | µs/cycle (after) | base | after | min-factor | median-factor |
|---|---|---|---|---|---|
| `two` | 60 | 0.59 s | **0.41 s** | **1.45×** | 1.29× |
| `reduce` | 88 | 0.98 s | **0.78 s** | **1.26×** | 1.31× |
| `four` | 80 | 8.58 s | **6.76 s** | **1.27×** | 1.16× |
| `nine` (2 tiles) | 150 | 1.81 s | **1.50 s** | 1.20× | 1.07× |
| `sfpumath` | 140 | 2.87 s | **2.62 s** | 1.09× | 1.15× |
| `six` (matmul) | 254 | 7.31 s | **7.00 s** | 1.04× | 1.05× |

In cycles/s: `two` 13,898 → 20,197; `reduce` 10,387 → 13,077; `four` 12,513 →
15,888; `nine` 6,364 → 7,662; `sfpumath` 6,734 → 7,352; `six` 3,760 → 3,926.

**The spread across workloads is Amdahl, not variance in the saving.** The
saving is a constant ~13 µs of pump overhead per simulated cycle, and it
divides into whatever that workload's own cycle costs. The accounting closes:

```
  16 DRAM-tile NIU ticks       16 x 511 ns  =  8.2 us   (tiles now dormant)
   8 DRAM tile-clock frames     8 x ~190 ns =  1.5 us
   2 live-tile NIU early-outs   2 x 349 ns  =  0.7 us
  ------------------------------------------------------
                                              ~10.4 us modelled
                                               ~13 us measured (four)
```

Predicted factors from that alone: `two` 1.28×, `sfpumath` 1.10×, `six` 1.05×
— which is what the table says. `six` gains least for the same reason it
gained least from target 1: its cycle is 254 µs of matmul, so 13 µs is 5 %.

Where the dormancy actually happens, instrumented over a whole run
(`TileClock.dormant_cycles` / total tile-cycles):

| workload | DRAM tiles dormant | Tensix tiles dormant |
|---|---|---|
| `two` | 65,583 / 65,600 = **100.0 %** | 0 / 8,200 = 0.0 % |
| `four` | 859,165 / 859,200 = **100.0 %** | 0 / 107,400 = 0.0 % |
| `six` | 219,656 / 220,000 = **99.8 %** | 0 / 27,500 = 0.0 % |
| `nine` | 91,965 / 92,000 = **100.0 %** | 99 / 23,000 = 0.4 % |

That table is the honest limit of phase 1: **every DRAM tile sleeps through
essentially the entire run, and no Tensix tile ever sleeps.** BRISC spins in
the firmware loop from launch to teardown, so the tile is legitimately busy by
the predicate. Getting the Tensix side needs per-*component* gating inside a
live tile (phase 3 of the plan), which is a strictly larger correctness claim
because it has to be right about components that observe each other's state
mid-cycle.

## Bit-identity

The pump determines execution *order*, so a bug here surfaces as subtly wrong
numbers rather than a crash. Gates, all on the final tree:

- `driver.wormhole.server.offline_replay_test` — **126/126 host READs
  reproduced bit-for-bit**, unchanged.
- All **17** Blackhole replay guards pass; `six_replay_test` still reports
  **PCC = 0.9982**, to the digit. Any movement in that number would mean
  execution order changed.
- `pytest tt_sim driver` — 236 passed (including a new
  `tt_sim/device/clock_test.py`: dormancy engages, the always-list and
  `on_tick` still see every cycle, and each of the three wake stimuli works).
- `examples_replay_test` — 11 passed. `ruff` clean.

The argument behind the gates, in one line each: a skipped tick is only ever
skipped when every component on the tile has said its own `clock_tick` guard
would fail; the decision is made *after* the tick, so a component owing
exactly one more tick gets it; and within an awake tile nothing changes at all.

## Method note: interleaving is not enough on a shared machine

The first sweep of this change was interleaved base-then-after per workload,
per round — the discipline the rest of this document uses — and it reported
`four` at 1.30× and `two` at 1.68×. Both were inflated. Two confounds that
interleaving alone does not remove:

1. **The tree moves.** Another agent was editing `tensix/backends/matrix.py`
   and `tensix/registers.py` during the sweep. Only the changed files were
   being swapped, so the *shared* code differed between early and late runs.
2. **Order within a round is fixed.** With `base` always first and `after`
   always second, any monotone drift — the other agent's code getting faster,
   the machine getting quieter — lands entirely on `after`.

The numbers above come from **two frozen full-tree snapshots** (`tt_sim/` +
`driver/` copied wholesale, with only the 13 changed files reverted in one of
them) and **alternating the order every round**. Under that protocol `four`
came back at 1.16–1.27× and `two` at 1.29–1.45×. Anything measured on this
machine at under ~1.3× needs both.

`six` is the cautionary case: a first six-round interleaved-but-unfrozen A/B
gave min-ratio 1.027 and median-ratio 0.990 — i.e. "faster and slower" — with
base ranging 7.2–9.3 s on identical code. The frozen protocol resolves it to a
consistent 1.04–1.05×, which is exactly what the per-cycle accounting predicts.

## Reproducing

```bash
export PYTHONPATH=~/tt-sim
# idle floor: build a Blackhole device, materialise N Tensix tiles, run it
# with every core in soft reset and time MultiTileClock.run.
# real workloads: wrap MultiTileClock.run for pump-only time, e.g.
time python3 -m driver.blackhole.server.four_replay_test

# dormancy share: read TileClock.dormant_cycles off every tile after a run
python3 -c "
from driver.blackhole.server.bh_device import make_device
d = make_device().tt_device
d.run(1000)
print([(c, t.clock.dormant_cycles) for c, t in d.tile_directory.items()])"
```

Do **not** A/B across sessions, and on a shared machine do not A/B without
freezing both trees and alternating the order — see the method note above.

# What landed: the Tensix operand gather

**Measured 2026-08-02**, same machine as the appendix (12th Gen Core i7-12700H,
CPython 3.12.13, `TT_SIM_THREADED` unset). This is the follow-up named in "The
new shape of an MVMUL, and the next target" above: with the FPU datapath
batched, the operand gather was the larger half of an MVMUL.

Two other agents were working in the tree, so every number is an **interleaved
A/B between two frozen git worktrees** — `HEAD` and `HEAD` plus this change
only — alternating base/after per workload and reported as the **minimum of 5
rounds**. All correctness gates were run in the frozen worktree too, not in the
shared tree (a run in the shared tree failed 11 examples on someone else's
half-saved edit, which is exactly the trap this method exists to avoid).

## What changed

`perform_mvmul_exact` used to read its operands one datum at a time: ~384
scalar `SrcRegister.__getitem__` calls, ~1,150 `DataFormatConversions` frames
(`BF16InSrcToFP32` → `BF16InSrcToBF16` → `TF32InSrcToTF32` per lane), and
132 `getDst16b` + 128 `setDst16b`/`FP32ToDstFormatBF16` pairs, per instruction.
It now gathers a rectangle at a time:

- **`SrcRegister.readRows(row, n)`** returns the `n x 16` block as int64 (a
  slice of the numpy array it already stored), and **`DstRegister`** gained
  `getDst16bRows` / `setDst16bRows` / `getDst32bRows` / `setDst32bRows`, the
  per-datum accessors applied to whole rows — same row adjustment, same zero
  flags, same 32-bit hi/lo split.
- **The conversions are not reimplemented.** They are masks, shifts and ors, so
  handing the *same* classmethod an int64 array converts the whole block. The
  one exception was `FP32ToBF16`, whose denormal flush was an `if exp == 0`;
  it is now `man * (exp != 0)`, which is the same value for a scalar.

So there is no second copy of the Src/Dst bit permutations to keep in step —
the batched path calls the documented ISA port itself.

## Result

Whole-process wall clock, min of 5 interleaved rounds:

| workload | MVMUL/GAPOOLs | base | after | factor |
|---|---|---|---|---|
| `six` (128³ bf16 matmul) | 4,096 | 12.51 s | **7.81 s** | **1.60×** |
| `matmulblock` | 1,536 | 5.30 s | **4.05 s** | **1.31×** |
| `matmulidx` | 384 | 2.51 s | **2.24 s** | **1.12×** |
| `reduce` (GAPOOL) | 36 | 1.78 s | 1.68 s | 1.06× |
| `four`, `nine`, `two`, `sfpumath` | **0** | — | — | 0.93–1.06×, no direction |

The last row is the control, and it is a *verified* zero rather than an assumed
one: a wrapper counting `perform_mvmul_exact` calls reports exactly 0 for those
four workloads, so any difference there is noise (`nine`'s 0.93× included).

Per instruction, timed in-run by wrapping `perform_mvmul_exact` in both trees:

| workload | base µs/MVMUL | after µs/MVMUL | share of run, base → after |
|---|---|---|---|
| `six` | 1,829 | **753** | 57.6 % → 36.4 % |
| `matmulidx` | 1,624 | **637** | 33.7 % → 17.4 % |
| `matmulblock` | 1,584 | **662** | 53.7 % → 33.0 % |
| `reduce` (4 rows, not 8) | 1,106 | **634** | 3.3 % → 1.9 % |

And the gather itself, isolated (min of 7 × 300 iterations; one MVMUL = a 16×16
SrcA block, an 8×16 SrcB block and an 8×16 Dst rectangle):

| | base | after | factor |
|---|---|---|---|
| operand gather (SrcA + SrcB + Dst read) | 666 µs | **56.5 µs** | 11.8× |
| result store (convert + 128 `setDst16b`) | 222 µs | **28.1 µs** | 7.9× |
| together | 889 µs | **85 µs** | **10.5×** |

Where the remaining 85 µs goes: SrcA convert 17.1, SrcB transpose + convert
14.4, Dst read + convert 20.6, result convert 19.7, store 5.7. That is ~35 numpy
calls at the 1–3 µs/op floor the earlier section measured, so this is close to
done without a JIT — and the datapath is once again the whole of what is left
(~660 µs of the 753).

One numpy lesson, and it is the same one as last time in a new disguise:
**convert the transposed SrcB block through `np.ascontiguousarray`**. A ufunc
over a transposed view returns an F-ordered result, and every downstream op then
pays: `_fpu_group_sums_batch` is 155 µs on a contiguous SrcB operand and 180 µs
on the view-derived one, while the copy itself is free (15.5 vs 15.6 µs).

## Bit-identity

Required, not approximated — this feeds the differential against the vendor
simulator.

- `tt_sim/pe/tensix/conversion_batch_test.py` (new) proves the array form of
  every conversion the batched path can reach equals the scalar form **over the
  whole input space**, not on a sample: all 2¹⁹ Src words for the six
  `*InSrcTo*` conversions, all 2¹⁶ Dst16b words for the four Dst ones, and for
  the FP32-width ones every equivalence class (all 2¹⁶ sign/exponent/high-
  mantissa combinations × five low halves, plus 200 k random words). 15 tests,
  6.9 s. `registers_test.py` gains six tests pinning each block accessor against
  the scalar accessor it replaces, including flag-cleared rows and both
  Blackhole row-remap gates.
- All **17** Blackhole replay guards pass, `driver.wormhole.server.offline_replay_test`
  reproduces **126/126** host READs bit-for-bit, `examples_replay_test` 11
  passed, and `pytest tt_sim driver` is **228 passed** (207 + 21 new).
- The live differential against the vendor simulator passes on **both**
  architectures: `./optests/diff.sh matmulidx` and the same under
  `TT_SIM_ARCH=wormhole`, 2,560 elements each, PASS.
- `six` still reports `PCC(golden, device) = 0.9982`, unchanged to the last
  digit.

## What did not pay off

- **A 2¹⁹-entry lookup table per conversion.** It is the fastest option —
  `LUT[block]` is 3.5 µs against 16.1 µs for the ufunc chain — but it is 4.2 MB
  per conversion (two are live), it saves ~25 µs of a 753 µs MVMUL (3 %), and it
  replaces a call to the documented ISA port with a table. Not taken.
- **Hand-fusing the conversion expressions**, e.g. `BF16InSrcToFP32` as one
  shifted mask instead of nine ufunc calls. Worth ~5 µs of 85, and it is exactly
  the correspondence with the ISA pseudocode that makes these functions
  reviewable. Not taken.
- **Batching the element-wise FP path** (`elementwise_fp_other`, the
  ELWADD/ELWSUB/ELWMUL operand gather). Measured first: it is called 4,096 times
  in `four_fp` for 0.04 s of a 1.53 s run (2.6 %), 512 times in `reduce`, and
  **zero** times in the other 15 guards. It also computes in Python floats
  through caller-supplied closures, so vectorising it risks float64/float32
  rounding differences for ~1 % of one workload. Not taken; re-measure if an
  element-wise-heavy kernel ever shows up.

## The next target inside the matmul

With the gather at 85 µs, `six` is ~36 % `perform_mvmul_exact` and essentially
all of that is `_fpu_group_sums_batch` / `_fpu_accumulate_batch` again — now at
~660 µs/MVMUL of numpy at the 1–3 µs/op overhead floor. That is the JIT
argument's proper baseline, and it should be re-argued against it rather than
against the pre-batch scalar code. The next non-JIT item visible in `six`'s
profile is the **unpacker**, which still writes Src one datum at a time
(131,072 `BF16ToSrcBF16` + `SrcRegister.__setitem__` calls per run) — the same
shape of change as this one, in `backends/unpacker.py`.

## Reproducing

```bash
export PYTHONPATH=~/tt-sim
# per-workload A/B: two frozen worktrees, alternating, min of N rounds
git worktree add --detach /tmp/gb_base HEAD    # and /tmp/gb_after + your diff
time python3 -m driver.blackhole.server.six_replay_test

# per-MVMUL cost: wrap MatrixUnit.perform_mvmul_exact with perf_counter and
# run any guard; it also tells you whether a workload uses the path at all.
```

# What landed: the unpacker's Src/Dst writes

**Measured 2026-08-02**, same machine as the appendix (12th Gen Core i7-12700H,
CPython 3.12.13, `TT_SIM_THREADED` unset). This is the target named at the end
of the operand-gather section: with the MVMUL read side batched, the unpacker
was the largest remaining non-JIT hotspot, moving **131,072 datums one at a
time** per `six` run.

Other agents were active in the tree throughout (one of them mid-save in
`device/clock.py`, which produced a spurious `AttributeError` in an early
measurement), so every number below comes from an **interleaved A/B between two
frozen git worktrees** — `HEAD` and `HEAD` plus this change only — alternating
which variant runs first each round. All correctness gates were run in the
frozen worktree too.

## Measure first: what the loop actually sees

Before touching anything, `perform_unpack` was wrapped with a counter recording
its parameter combination, over all 17 Blackhole guards and the 11 Wormhole
example replays. Only nine combinations occur:

| in -> out | destination | where | datums |
|---|---|---|---|
| BF16 -> BF16 | SrcA + SrcB | `six`, `matmulblock`, `matmulidx`, `reduce` | 131,072 in `six` alone |
| INT8 -> INT8 | SrcA + SrcB | `four`, `nine` | 8,192 each |
| FP32 -> BF16 | SrcA + SrcB | `four_fp` | 8,192 |
| FP32 -> TF32 | SrcA | `sfpumath` | 3,072 |
| INT32 -> INT32 | Dst | `five`, `loopback`, `optest` | 8,192 / 4,096 / 6,144 |
| FP32 -> FP32 | Dst | `five_fp` | 8,192 |
| INT32 / FP32, `ZeroWrite2` | SrcA | WH `nine`, `sfpumath` | 16 per call |
| BF16 -> BF16, haloize | SrcA | `reduce`, `reduceneg` | 1,024 of 5,440 |
| *(none)* | — | `two`, `three`, `eight` | **0 calls** |

Three facts fell out of that census and shaped the change:

1. **The loop is already a clean rectangle.** The awkward modes the ISA doc
   describes — `rowStride`, `upsampleZeroes`, `upsampleInterleave` — were
   *passed to* `perform_unpack` and never read: the input walk is
   `inAddr_Datums += datumSizeBytes`, flat. So there was no discontiguity to
   work around, and the batched path is exactly as (in)complete as the scalar
   one it replaces. Nothing was skipped by batching that the scalar loop
   handled. (Since then those three no longer reach `perform_unpack` at all:
   `check_modelled_settings` rejects a non-abutting `RowStride` or a non-zero
   `Upsample_rate` at decode, so the rectangle is now a precondition rather
   than a lucky property of the corpus.)
2. **Every observed call has `colShift == 0` and `numRows <= 64`**, and the
   only transpose is `reduce`'s 4 calls.
3. **`two`, `three` and `eight` never enter the function at all** — the
   verified control group, not an assumed one.

## What changed

`_unpack_block` in `backends/unpacker.py` does what the datum loop did, a
rectangle at a time, and returns False for the cases it cannot:

- **One L1 read.** `AddressableMemory.read(addr, numRows * 16 * size)` and
  `np.frombuffer` with a `uint8`/`uint16`/`uint32` view. Datums are
  little-endian and so is the host, so the view is the same value per datum as
  `conv_to_uint32` of its bytes.
- **One conversion.** `formatConversion` is the *same* method, handed an int64
  array instead of an int — the technique the operand gather established. Two
  branches inside it had to become arithmetic to survive a block, and both are
  the same value for a scalar: INT8's `if raw_datum: raw_datum |= 16 << 10`
  became `raw_datum | (16 << 10) * (raw_datum != 0)`, and the FP32 -> BF16
  denormal flush turned out to be exactly `DataFormatConversions.FP32ToBF16`,
  which the gather had already made branch-free. No conversion is
  reimplemented.
- **One indexed write.** `SrcRegister.writeDatums(rows, columns, values)` is
  the new write counterpart to `readRows` — literally `self.data[rows, cols] =
  values`, with the scalar setter's indexing rules (a negative column wraps, an
  out-of-range row raises). Dst reuses the gather's `setDst16bRows` /
  `setDst32bRows`.

The destination index arithmetic is transcribed from the scalar loop rather
than re-derived: `rows` is the `row` loop variable as an array, `cols` is
`np.arange(16) - colShift`, and haloize is the same two assignments done at
once (`outRows, outCols = (outRows & ~0xF) | outCols, outRows & 0xF` — the
tuple's right-hand side evaluates first, which is what the scalar version's
`rowLowBits` temporary is for). Because they are index arrays rather than
slices, the **column shift and the haloize transpose came along for free**;
there was no reason to leave them on the scalar path.

### Deliberately left scalar

- **Datum widths that are not 1, 2 or 4 bytes** — the BFP formats with their
  shared exponents. `DATA_FORMAT_TO_BITS` gives BFP4/BFP2 a size of *zero*
  bytes, so the scalar loop does not handle them either; declining keeps the
  two paths equally (in)correct rather than inventing behaviour.
- **`FP32 -> FP16` out.** `FP32ToFP16` saturates and flushes with an `if/elif`.
  Zero guards reach it, and handing it a block raises rather than silently
  taking one arm — pinned by a test, so it cannot quietly start "working".
- **Row counts large enough for the destination row map to alias** (`> 64` into
  a Src bank, `> 16` into Dst under `SetOvrdWithAddr`). An indexed assignment
  would then depend on numpy's ordering where the scalar loop's last-write-wins
  is explicit. No guard reaches it; the check is 4 lines.

## Result

Per-`UNPACR` cost, timed in-run by wrapping `perform_unpack` in both frozen
worktrees, min of 5 alternating rounds:

| workload | UNPACRs | datums | base | after | factor | share of base run |
|---|---|---|---|---|---|---|
| `six` | 128 | 131,072 | 0.908 s | **0.021 s** | **43x** | 10.9 % -> 0.3 % |
| `matmulblock` | 36 | 36,864 | 0.285 s | **0.005 s** | **57x** | 8.4 % -> 0.2 % |
| `five` | 32 | 8,192 | 0.072 s | **0.004 s** | 18x | 4.6 % |
| `four` | 32 | 8,192 | 0.085 s | **0.005 s** | 17x | 0.9 % |
| `four_fp` | 32 | 8,192 | 0.052 s | **0.005 s** | 10x | 4.2 % |
| `sfpumath` | 12 | 3,072 | 0.021 s | **0.002 s** | 10x | 0.7 % |
| `reduce` | 40 | 5,440 | 0.044 s | **0.005 s** | 9x | 4.5 % |

Per datum that is **6.9 us -> 0.16 us** on `six`'s 1024-datum unpacks. After the
change the cost is nearly flat in the datum count (112–165 us per call whatever
the size), i.e. it is numpy's per-call overhead — about 20 array ops — and no
longer the data.

Whole-process wall clock and CPU time, min and median of 9 interleaved
alternating rounds:

| workload | wall base | wall after | min ratio | median ratio | CPU min ratio |
|---|---|---|---|---|---|
| `six` | 9.20 s | **8.07 s** | **1.14x** | 1.09x | 1.12x |
| `matmulblock` | 4.12 s | 3.98 s | 1.04x | 1.12x | 1.02x |
| `four_fp` | 2.11 s | 2.09 s | 1.01x | 1.04x | 0.96x |
| `four`, `reduce`, `five` | — | — | 0.94–1.00x | 1.01–1.02x | 0.96–1.04x |
| `two`, `eight` (**0 unpacks**) | — | — | 1.06–1.08x | 1.01–1.05x | 1.02–1.05x |

The control row is the honest reading of the rest of the table: on this machine,
under this much contention, whole-run A/B noise is about **±6 %** even frozen
and alternated. Only `six` clears it, and it clears it in all four statistics at
the value the per-call measurement predicts (-0.89 s on an 8–9 s run). Everything
else is a real but sub-noise 1–5 %, which is exactly what the share column
above says it should be. The per-call table, not the whole-run table, is the
measurement of this change.

## Bit-identity

Required, not approximated.

- `tt_sim/pe/tensix/conversion_batch_test.py` gains the unpack direction: the
  four to-Src conversions over their whole input space (2^19 for
  `TF32ToSrcTF32` / `TF32ToSrcFormatTF32`, 2^16 for `BF16ToSrcBF16` /
  `FP16ToSrcFP16`), **and `UnPackerUnit.formatConversion` itself end to end** —
  every (in, out, `unpackToDst`) triple `_unpack_block` accepts, array against
  scalar, over the whole 8- or 16-bit datum space or every equivalence class of
  the 32-bit one. That is the test that actually covers the two branch
  removals. 45 tests, 11 s.
- `registers_test.py` gains `writeDatums` against the scalar setter over the
  four index maps the unpacker builds (rectangle, wrapped SrcB rows, shifted
  column, haloize transpose).
- `pytest tt_sim driver` is **267 passed** (236 + 31), all **17** Blackhole
  replay guards pass, `driver.wormhole.server.offline_replay_test` reproduces
  **126/126** host READs bit-for-bit, `examples_replay_test` 11 passed.
- `six` still reports `PCC(golden, device) = 0.9982`, unchanged to the last
  digit.
- The live differential against the vendor simulator passes on **both**
  architectures: `./optests/diff.sh matmulidx` and the same under
  `TT_SIM_ARCH=wormhole`, 2,560 elements each, PASS.

## What did not pay off, and what is next

- **Batching the Src *clear* loops** (`handle_set_src_to_zero` and Blackhole's
  `_handle_unpacr_nop_blackhole`, each a 64x16 scalar write). Measured before
  writing anything: 0.041 s of `sfpumath`'s 3.1 s run and 0.006 s of `reduce`'s,
  **zero** in every other guard — and most of even that is the blocked-and-retry
  early return, not the clear itself. Not taken.
- **A separate fast path for the common rectangle** (contiguous rows, no column
  shift, no transpose) as a slice assignment instead of a 2-D indexed one. It
  is worth a few microseconds of a 150 us call and it would mean two
  destination-index derivations to keep in step with the ISA pseudocode instead
  of one. Not taken.
- **The remaining ~150 us per UNPACR is now per-call numpy overhead**, not the
  datums, so the next win in this file would have to come from doing fewer,
  larger unpacks — which the hardware model does not permit. With the unpacker
  at 0.3 % of `six`, the matmul datapath (`_fpu_group_sums_batch` /
  `_fpu_accumulate_batch`, ~660 us of a 753 us MVMUL) is once again the whole
  of what is left.

## Reproducing

```bash
export PYTHONPATH=~/tt-sim
# which formats a guard actually unpacks, and whether it unpacks at all:
# wrap UnPackerUnit.perform_unpack and count its arguments.
# per-UNPACR cost: wrap the same method with perf_counter.
git worktree add --detach /tmp/ub_base HEAD   # and /tmp/ub_after + your diff
time python3 -m driver.blackhole.server.six_replay_test
```

Freeze both trees, alternate the order, and read the control workloads before
believing any whole-run ratio under ~1.1x — see the method note in the
event-driven-pump section, which this change's numbers confirm again.

# Investigated and not real: the `six` "regression" after the optests commits

**Measured 2026-08-03**, same machine as the appendix (12th Gen Core i7-12700H,
CPython 3.12.13, `TT_SIM_THREADED` unset). A single run of `six_replay_test`
timed 6.99 s before commits `469899b` ("Pumping of instructions") and `0ac4b39`
(the optests differential fixes) landed, and 8.11 s after — an apparent 16 %
give-back on the day five optimisations took `six` from 35.68 s to single
digits. **It does not reproduce. HEAD is faster than the pre-change tree on
every statistic**, and this section is the evidence, kept because a documented
non-regression is worth as much as a fix.

## Result: three interleaved A/Bs, three frozen worktrees

`pre` = `fe0d279` (before both commits), `mid` = `469899b`, `head` = `0ac4b39`.
Full-tree `git worktree` snapshots, order rotated every round, whole-process
wall clock **and** child CPU time (`RUSAGE_CHILDREN`), min and median of each.

| sweep | rounds | machine | pre | mid | head | head vs pre |
|---|---|---|---|---|---|---|
| 1 | 9 | loaded (load 8–18) | 7.552 s | 6.889 s | **6.789 s** | **0.90×** |
| 2 | 11 | loaded → quiet | 7.691 s | — | **6.977 s** | **0.91×** |
| 3 | 6 | quiet (load 1.4) | 7.815 s | 7.298 s | **6.094 s** | **0.78×** |

(wall-clock **minimum** of the rounds — the statistic this document uses,
because contention can only add time. The medians agree in direction in all
three sweeps: 0.90×, 0.89×, 0.91×. CPU-time min agrees too: 0.92×, 0.94×,
0.84×.) `head` is not slower than `pre`; it is **~1.1× faster**, which is the
unpacker batching in `469899b` showing up where it should. The quiet-machine
`head` minimum of **6.09 s** is *below* the 6.99 s that was reported as the
"before".

## Why the original numbers disagreed

The machine was not quiet. During the first sweep an unrelated build was
running (`ftn-opt` / `tt-opt` / `tt-xftn`, load average climbing 1.4 → 18.7),
and a single `head` run in that window came back at 10.65 s against 7.47 s for
the same code twelve minutes later. Two single runs taken minutes apart cannot
separate a 16 % effect from that. This is the third time this document has had
to say it: **freeze both trees, alternate the order, take the minimum of
several rounds, and say which statistic you used.**

## Why it could not have been real, mechanically

The four changes in the bisection space were each instrumented in the frozen
`head` worktree rather than argued about. Call counts are the measurement:

| suspect | reached in `six`? | cost |
|---|---|---|
| **Sparse DRAM backing** (`SparseAddressableMemory`) | 192 reads + 48 writes, 480 KiB | **0.004 s of 6.0 s = 0.07 %** |
| **Cost-model hook** (`busy_until` / `cost_model` guards in `TensixBackendUnit.clock_tick`) | 275,000 unit-ticks | ≤ **0.023 s = 0.3 %** (85 ns/call upper bound, which includes the call frame the guard does not pay for) |
| **Blackhole SFPU decode** (`_read_dest_reg_addr`, `_compare_lanes`) | **0 calls** — verified, not assumed | 0 |
| **RV32I `sh` fix** (`rs2_val[0:1]` → `[0:2]`) | a slice bound | 0 |

Sum: **under 0.03 s of a 7 s run**, against the 1.12 s that was being looked
for. The suspicion that sparse DRAM would bite because "`six` streams tiles
from DRAM continuously" is the interesting one to correct: `six` moves its
128³ operands through *L1* once the reader kernel has fetched them, so the
sparse path sees 240 calls in the whole run, not one per tile row.

Sparse-DRAM census across all 22 Blackhole guards, so the point is not
specific to `six`:

| guard | sparse calls | bytes | share of run |
|---|---|---|---|
| `six` | 240 | 491,520 | 0.07 % |
| `matmulblock` | 18 | 106,496 | 0.03 % |
| `optest` | 10 | 81,920 | 0.06 % |
| every other guard | 3–16 | ≤ 49,152 | **≤ 0.05 %** |

The worst case in the tree is 0.07 %. A last-hit chunk cache, a bigger chunk
size, or a flat-array fast path would all be measurable in a microbenchmark and
invisible in every workload, so none was written — the change is already at the
point where the honest answer is "leave it alone".

## Gates (nothing changed, so these are a pin, not a check)

`ruff check` / `ruff format` clean; `pytest tt_sim driver` **349 passed**;
**22/22** Blackhole replay guards; `driver.wormhole.server.offline_replay_test`
**126/126 byte-identical**; `pytest driver/wormhole/server/` **18 passed**;
`six` reports `PCC(golden, device) = 0.9982`, unmoved.

## Reproducing

```bash
export PYTHONPATH=~/tt-sim
git worktree add --detach /tmp/wt_pre  fe0d279
git worktree add --detach /tmp/wt_head 0ac4b39
# alternate the order every round, take the min, and check `uptime` while you
# do it — a background build is worth more than the effect you are chasing.

# whether a change is even reached: wrap the suspect method with a counter and
# run the guard, e.g. SparseAddressableMemory.read / VectorUnit._compare_lanes.
```

---

# What landed: the RISC-V cost model, and what it costs when it is off

The §I cycle-cost model reached the baby RISC-V cores' load/store path
(`tt_sim/pe/rv/cost.py`; write-up in
[`docs/plans/cost-model.md`](../../../docs/plans/cost-model.md#the-risc-v-cores-where-the-cycles-finally-moved)).
It is not an optimisation, so the only question this document has to answer is
the one the RV interpreter's history makes urgent: **the interpreter is the
hottest path in the simulator and was made ~2.3× faster by deleting
per-instruction overhead, so what does adding a per-instruction cost hook cost
when the model is switched off?**

The hook is one instance-attribute read and one predicted branch, before the
ISA dispatch:

```python
cost = self.rv_cost                       # None unless TT_SIM_COST_MODEL
if cost is not None and not cost.can_issue(instr, cycle_num, register_file):
    return
```

`rv_cost` is an instance attribute (not the class attribute it shadows), for
the same reason `TensixBackendUnit.busy_until` is: a class-attribute read on
this path is an MRO walk.

## Result: no measurable cost, with a control group that says so

Three **frozen full-tree copies** — `new` (the working tree), `base` (the same
tree with *only* this change reverted, so the concurrently-uncommitted Tensix
cost wiring is in both arms), and `ctrl` (a byte-identical copy of `base`,
verified with `diff -rq`). Twelve rounds, order alternated each round, `four`
timed once per round and `two` five times per round (it is only ~1.5 s, so a
single run is startup-dominated). `TT_SIM_COST_MODEL` unset throughout.

| guard | statistic | base | ctrl | new |
| --- | --- | --- | --- | --- |
| `four` | median of 12 | 11.515 s | 11.502 s (**−0.1 %**) | 11.617 s (**+0.9 %**) |
| `four` | min of 12 | 10.508 s | 10.965 s (**+4.4 %**) | 10.696 s (**+1.8 %**) |
| `two` | median of 12×5 | 1.520 s | 1.531 s (**+0.7 %**) | 1.512 s (**−0.6 %**) |
| `two` | min of 12×5 | 1.398 s | 1.395 s (**−0.2 %**) | 1.396 s (**−0.2 %**) |

The control group is the whole point, and it is why both statistics are
reported. `ctrl` is the *same code as `base`*, and it lands anywhere from
−0.1 % to +4.4 % depending on which statistic you take. `new` is inside that
band on every line, and on `two` — the shorter, more RV-dominated run, where a
per-instruction overhead should show up most cleanly — it is indistinguishable
from the control to within 0.2 %. **No cost distinguishable from a control
group of identical code**, with an upper bound around 1 % rather than a claim
of exactly zero. (This is the discipline the "Investigated and not real"
section above exists to enforce — single-run timings on this machine have
already sent one agent chasing a regression that did not exist. An earlier pass
of this same A/B, against a `base` that was a plain HEAD worktree and therefore
*missing* the concurrent Tensix work, produced +0.1 % / −3.0 % with a control
at +4.4 % / −0.8 %: a different set of numbers, the same conclusion, and a
reminder to check what is actually in each arm.)

With the model **on**, the same `four` run is ~8 % slower in wall clock (11.55 s
→ 12.43 s, median of three alternating runs — only just outside the noise floor)
and 0.18 % longer in simulated cycles. The scoreboard walk is cheap because it
is a list index and two integer compares; nobody should read the on-state as a
performance path regardless.

## Gates

`ruff check` / `ruff format` clean; `pytest tt_sim driver` **401 passed** (380
before, +19 `tt_sim/pe/rv/cost_test.py`, +2
`tt_sim/pe/tensix/sync_mixed_queue_test.py`); **22/22** Blackhole replay
guards; `driver.wormhole.server.offline_replay_test` **126/126
byte-identical**; `pytest driver/wormhole/server/` **18 passed**; `six` reports
`PCC(golden, device) = 0.9982`, unmoved. All with `TT_SIM_COST_MODEL` unset,
which is the contract.

## Reproducing

```bash
export PYTHONPATH=~/tt-sim
git worktree add --detach /tmp/bench/base HEAD
cp -a /tmp/bench/base /tmp/bench/ctrl          # the control: identical code
rsync -a --exclude .git ~/tt-sim/ /tmp/bench/new/
diff -rq /tmp/bench/base/tt_sim /tmp/bench/ctrl/tt_sim   # verify the control
# then: 6 rounds, order alternated, median per variant. Report the control's
# own deviation next to the change's — a change smaller than the control is a
# non-result, not a win.

# simulated-cycle A/B at one-cycle resolution (the guards poll every 2000-5000
# cycles, which hides anything smaller): monkeypatch the guard module's
# PUMP_CHUNK to 1, wrap _build_fabric to keep the device, and read
# device.tt_device.clocks[0].clock_tick_num after main().

# where the RV stalls went: core.rv_cost.summary() on each baby core after a
# run — total stall cycles, split by load-use / store-rate / integer-unit, and
# loads counted per address region.
```

---

# What landed: the deadlock watchdog, sampled instead of polled

The progress watchdog (`tt_sim/device/deadlock.py`, wired from
`Wormhole.__init__` as the pump's `on_tick`) took its whole progress signature
**every cycle** — every tile, every baby core, every NIU, every Tensix thread —
to find a condition whose window is 50,000 cycles and which fires a handful of
times in a run, if ever. On the full-worker-grid feasibility study
([`docs/upstream-examples-status.md`](../../../docs/upstream-examples-status.md#what-the-full-grid-actually-costs))
that made it **92 % of an 80-worker idle pump**. It is now sampled once per
`threshold // 8` cycles, with a confirmation pass before it prints.

## Measure first: where the watchdog's time actually went

`cProfile` on an 8-Tensix Wormhole idle pump, striding off, 4,000 cycles:
`DeadlockDetector.tick` is **0.430 s of the 0.531 s** the pump takes (81 %),
and inside it the call counts are all one thing — the per-core soft-reset poll:
160,000 `deque.clear`, 160,000 `get_nth_bit`, 32,000 `conv_to_uint32` for
8 tiles × 5 cores × 4,000 cycles. Nothing else in the tick is reached, because
with every core in reset the signature is never built.

That is the *cheap* half. `timeit` on one observation (min of 5×200), with
cores in reset and then with BRISC out of reset on every tile:

| Tensix tiles | one scan, all cores in reset | per tile | one scan, cores live | per tile |
| --- | --- | --- | --- | --- |
| 1 | 4.2 µs | 4.2 | 25.9 µs | 25.9 |
| 8 | 17.6 µs | 2.2 | 137.8 µs | 17.2 |
| 20 | 38.7 µs | 1.9 | 357.9 µs | 17.9 |
| 80 | 218.5 µs | 2.7 | 1,881.9 µs | 23.5 |

**A live core makes the scan ~10× more expensive**, because it is what turns on
the other two thirds of the signature: a 61-entry tuple copy per NIU (176 of
them on an 80-worker Wormhole) and `CoprocessorDoneCheck` per Tensix thread,
which walks every backend unit. One live tile is enough — the signature is
built if *any* core anywhere is out of reset. Since one live tile also forces
the Phase 4 stride off, an 80-worker Wormhole with a single running kernel was
paying ~1.9 ms of watchdog per simulated cycle.

(For the record, the "it stops being the dominant term once tiles are awake"
line in the feasibility study was measured on **Blackhole**, which at the time
never wired the detector at all — both arms of that A/B had no watchdog. On
Wormhole the opposite holds, and the live-workload table below quantifies it.
Blackhole wires the watchdog now — it is built in `TT_Device.__init__`, so
every architecture gets it — which both fixes the missing `[DEADLOCK]`
diagnostic there and makes that A/B worth re-running.)

## What changed

Three things, all in `deadlock.py` plus one line of the device base class
(`tt_device.py`; it was `wormhole.py` when this was written):

1. **Sampled, not polled.** `tick` is now one integer compare against
   `_next_sample_cycle` — 91 ns, measured — and the scan happens once per
   `max(1, threshold // 8)` cycles (6,250 by default).
2. **Confirmed before reported.** Sampling alone can alias: a loop whose period
   matches the sample interval looks frozen at every sample point. A stall that
   clears the threshold is therefore re-checked on 64 consecutive ticks before
   anything is printed. A wedged device passes trivially; a core that is
   actually retiring instructions moves its PC and aborts the confirmation. The
   confirmation also refills the recent-PC window with consecutive-cycle PCs,
   so the report reads exactly as it did before.
3. **Disabled means unwired.** `TT_SIM_DEADLOCK=0` no longer installs the
   `on_tick` hook, so it costs nothing rather than a call that returns.

The window is now counted in simulated cycles rather than pump ticks. The two
are the same thing except when the pump strides, which only happens when every
tile is dormant.

## Result: the idle floor

Two frozen full-tree copies (`before` = HEAD, `after` = HEAD + this change),
`TT_SIM_PUMP_STRIDE=0` (what one live tile forces anyway), 4 rounds × 5
repeats, order alternated per round, **min of 20** per cell — µs per cycle:

| Tensix tiles | before, watchdog on | before, off | after, watchdog on | after, off | watchdog cost: before → after |
| --- | --- | --- | --- | --- | --- |
| 1 | 6.94 | 4.75 | 3.74 | 4.29 | 2.2 → below noise |
| 8 | 27.14 | 5.66 | 5.47 | 5.39 | 21.5 → 0.08 |
| 20 | 62.81 | 7.82 | 7.84 | 7.61 | 55.0 → 0.23 |
| 80 | 207.3 | 18.08 | 18.59 | 17.49 | 189 → 1.1 |

**The idle pump at 80 workers goes 207 → 18.6 µs/cycle (11×)**, and the
watchdog stops being visible in it: `after`-with-watchdog is within the
control's own spread of `after`-without.

**Control group.** A third tree, `diff -rq`-verified byte-identical to
`before`, run against `before` in the same interleaved shape (4 rounds): the
two identical arms differ by up to **11.5 %** on the min statistic (n=20,
watchdog on: 58.2 vs 51.5 µs/cycle) and by up to 9 % between separate runs of
the same arm on different occasions (n=80 watchdog on: 207.3 then 226.6). Every
number claimed above is a 5–200× effect, i.e. one to two orders of magnitude
outside that band.

An in-process A/B (same device object, `on_tick` wired and unwired
alternately, 6 rounds) puts the residual below the pump's own variance at every
tile count — the watchdog-on arm measured *faster* than watchdog-off in all
four, which is the honest way of saying "not measurable". The composed upper
bound from the microbenchmarks is 0.091 µs/cycle for the gate plus
218.5 µs / 6,250 = 0.035 µs/cycle for the amortised scan at 80 idle workers
(0.30 µs/cycle when they are live).

## Result: a real workload

The `one` offline replay guard with N workers materialised and one launched —
the regime an upstream full-grid run is in, since tt-metal releases BRISC on
every declared worker. Min of 3 alternating rounds; every run reproduced
**126/126 READs bit-for-bit** in both arms:

| Tensix tiles | before | after | speedup | replay wall clock |
| --- | --- | --- | --- | --- |
| 1 | 152.7 µs/cycle | 79.1 | **1.9×** | 0.86 s → 0.44 s |
| 8 | 279.3 | 81.2 | **3.4×** | 1.77 s → 0.46 s |
| 20 | 477.8 | 111.5 | **4.3×** | 2.69 s → 0.64 s |
| 80 | 1,920.2 | 119.9 | **16.0×** | 10.75 s → 0.67 s |

The `after` column is flat in tile count to within 50 % across an 80× range of
grid sizes; the `before` column is linear in it. That is the whole point: the
per-cycle cost of a grid you are not using is now the pump's ~0.24 µs per
dormant tile clock and nothing else.

## What it costs in detection latency

A stall is reported between `threshold` and
`threshold + threshold//8 + 64` cycles after the last observable change —
**50,000 to 56,314 cycles** with the defaults, against exactly 50,000 before.
Lowering `TT_SIM_DEADLOCK_THRESHOLD` shrinks the interval with it (it is a
fraction of the threshold, floored at 1), so a user who wants prompt detection
still gets it; at a threshold of 8 or less the detector polls every cycle again.

Directly measured on the same wedged device (BRISC executing `j .`, threshold
forced to 800): `before` prints at `cycle=800`, `after` at `cycle=864`, with
byte-identical report text. Nothing that used to be detected stops being
detected — the signature, the reset gating and the report are unchanged; only
*when it is looked at* has changed.

`tt_sim/device/deadlock_test.py` pins all of this down: fires on a wedged
device, latency inside the stated bound, silent while every core is in reset,
silent on a loop whose period is exactly the sample interval (the aliasing case
the confirmation pass exists for), and one scan per interval rather than one
per cycle.

### Follow-up: the bound holds on a *dormant* device too

Sampling made the watchdog depend on the Phase 4 pump actually visiting a
cycle, and it does not visit one that every tile clock has declined — a fully
dormant device is strided over in a single jump per `run()` call. Measured:
one sample in a 1,000,000-cycle `run`, and a wedged-but-dormant BRISC produced
**zero** `[DEADLOCK]` lines over 400,000 cycles. That was previously argued
unobservable, because dormancy implies every baby core is in soft reset, which
is the one state the watchdog ignores by design — a true statement resting on
an invariant in `tt_sim/device/tiles.py` rather than in the detector.

`DeadlockDetector.next_sample_cycle` is now handed to `MultiTileClock` as
`on_tick_wake` and joins the stride computation alongside every tile clock's
`next_event_cycle`, so a scheduled sample can never be jumped. The **stated
latency does not change** — `threshold` to `threshold + threshold//8 + 64`,
and the wedged-dormant device reports at `cycle=864` for a threshold of 800,
byte-identical to the awake case.

What it costs, on the same dormant-pump-floor benchmark (Blackhole, min of 3,
1,000,000 cycles):

| Tensix tiles | dormant floor, probe unwired | probe wired | delta |
| --- | --- | --- | --- |
| 1 | 0.0002 µs/cycle | 0.0021 | +0.002 |
| 8 | 0.0013 | 0.0051 | +0.004 |
| 80 | 0.0084 | 0.0410 | +0.033 |

The 80-worker delta is the amortised signature scan (218.5 µs / 6,250 =
0.035 µs/cycle) that the table above already charges the watchdog on an
*awake* idle grid — it is now paid on a dormant one as well, against a ~120
µs/cycle live floor. A **live** workload pays nothing: the probe is consulted
only when the pump is about to stride, which requires no Tensix tile to want
the next cycle, and BRISC spins in the firmware loop from launch to teardown.
The `one` offline replay measures 1.27 s wired against 1.30 s unwired (min of
3, 126/126 byte-identical in both arms) — i.e. inside noise, in the wrong
direction. `TT_SIM_DEADLOCK=0` leaves `on_tick_wake` unwired along with
`on_tick`.

## Should it still be on by default?

Yes, more clearly than before. The residual is ~0.1 µs/cycle plus an amortised
scan — under 0.4 µs/cycle on an 80-worker grid whose live floor is ~120
µs/cycle, i.e. **~0.3 %**, and unmeasurable against run-to-run noise. The
argument for `TT_SIM_DEADLOCK=0` in the practical-guidance sections of
[`docs/running-tt-metal-on-the-simulator.md`](../../../docs/running-tt-metal-on-the-simulator.md)
and the upstream-examples status doc no longer has numbers behind it.

## Gates

`ruff check` / `ruff format` clean; `pytest tt_sim driver` **408 passed** (401
before, +7 `tt_sim/device/deadlock_test.py`); **22/22** Blackhole replay guards;
`driver.wormhole.server.offline_replay_test` **126/126 byte-identical** (and
again at 8, 20 and 80 materialised workers); `pytest driver/wormhole/server/`
**18 passed**; `six` reports `PCC(golden, device) = 0.9982`, unmoved.

## Reproducing

```bash
export PYTHONPATH=~/tt-sim
# where the watchdog's time goes (idle, 8 tiles, striding off)
TT_SIM_PUMP_STRIDE=0 python3 -c "
import cProfile, pstats
from driver.wormhole.server.coords import TENSIX_COORD_MAP, default_tensix_coords
from tt_sim.device.wormhole import Wormhole
d = Wormhole(tensix_coords=[TENSIX_COORD_MAP[p] for p in default_tensix_coords(8)])
cProfile.run('d.clocks[0].run(4000)', '/tmp/idle8.prof')
pstats.Stats('/tmp/idle8.prof').sort_stats('tottime').print_stats(8)"

# idle floor per tile count: build the device, time clocks[0].run(N), divide.
# Alternate TT_SIM_DEADLOCK=1/0 and the two trees every round; min of >=20.

# live workload at N workers: wrap offline_replay_test._build_fabric, call
# device.ensure_tensix_tile(p) for default_tensix_coords(N), time main() minus
# the build, and divide by device.tt_device.clocks[0].clock_tick_num.

# cost of one observation, and of the per-cycle gate:
#   timeit DeadlockDetector._sample (after) / .tick (before), and
#   timeit .tick with _next_sample_cycle set past the horizon.
```

# What landed: target 5, a last-hit range cache on `MemoryMap`

**Measured 2026-08-06** against `24403ae`, machine as elsewhere in this
document (12th Gen Core i7-12700H, 16 threads, CPython 3.12, `TT_SIM_THREADED`
unset). This is ranked target 5 — "`MemoryMap` interval lookup /
`MemorySpace.read`", quoted at 3–7 % across workloads — and §L's note that the
win there is **caching the last-hit range, not a JIT** (the polymorphic
`mem_mapable` dispatch is Numba-hostile).

## The change

Ten lines in `tt_sim/memory/memory_map.py`. `MemoryMap.locate` keeps a
`_last_hit` tuple `(low, high, addr_range, value)` from the most recent
successful lookup and answers from it when `low <= addr <= high`, before
touching the `bisect_right` index. `_invalidate_index` clears it alongside
`_index`, so the one mutation path (`__setitem__` / `__delitem__`) drops the
cache too.

Correctness rests on three facts, all checked rather than assumed:

- **Only a successful full lookup populates the cache.** A miss leaves it
  untouched, so the out-of-mapped-range `IndexError` path in
  `MemorySpace._locate_memory_space` is bit-for-bit the code it was.
- **Ranges cannot shadow one another.** `MemoryMap.verify()` rejects any
  overlap, so the cached range is the *only* range that can cover its own
  addresses — a hit can never be a stale answer for a range that has since
  been shadowed.
- **Nothing remaps after construction.** Instrumenting `__setitem__` /
  `__delitem__` and splitting by phase: every workload measured shows
  **50 mutations during device construction and 0 during the pumped run**
  (`four`, `six`, `two` alike). The invalidation hook is belt-and-braces, not
  load-bearing.

Thread safety is by construction: each tile builds its own `MemoryMap` in
`TensixTile.__init__` (and `DRAMTile` / `EthTile`), so a per-instance cache is
per-tile, and `TT_SIM_THREADED=1` gives one worker per *tile*. No `MemoryMap`
is shared across worker threads, so no lock is needed and none was added.

## Does the cache actually hit?

Deterministic, machine-load independent — classify every `locate` call:

| workload | `locate` calls | cache hit | full lookup | miss |
|---|---|---|---|---|
| `four` | 671,485 | **98.1 %** | 13,073 | 0 |
| `nine` | 125,229 | 81.9 % | 22,687 | 0 |
| `two` | 39,897 | 74.4 % | 10,205 | 0 |
| `six` | 187,962 | 74.0 % | 48,883 | 0 |
| `softplus` | 112,852 | 69.3 % | 34,622 | 0 |

The premise holds — consecutive accesses do run in the same range — and **no
workload ever misses**, which is the other half of "exactly transparent".

`locate` itself, replaying each workload's real captured `(map, addr)` stream
through the real device maps (8 distinct maps, 1–16 ranges each), interleaved
5 rounds:

| | base | change |
|---|---|---|
| real stream, real maps | 488 ns/locate (median) | 400 ns/locate (median) |

Change wins 5/5 rounds, ≈ **−18 % on `locate`**, ~110 ns/call. A one-shot
un-interleaved version of this same benchmark measured the *opposite* sign;
it is recorded here as a reminder that nothing on this machine is measurable
without interleaving.

## The A/B

Two frozen `git archive` exports of `24403ae` differing **only** in
`tt_sim/memory/memory_map.py` (`diff -r` verified), alternating order every
round, paired within-round. `%` is the paired median of
`(change − base) / base`; the CI is on the paired mean.

| workload | rounds | base median (s) | change median (s) | paired median | 95 % CI on mean | change wins |
|---|---|---|---|---|---|---|
| `four` (RV-bound, 98 % hit) | 18 | 12.94 / 12.43 | 11.92 / 12.23 | **−6.66 %** | **−5.16 % ± 3.51** | 16/18 |
| `six` (matmul) | 10 | 8.56 | 8.53 | −2.46 % | −0.71 % ± 5.38 | 6/10 |
| `two` (NoC/L1 smoke) | 15 | 1.57 | 1.66 | −0.70 % | +0.34 % ± 9.51 | 7/15 |
| `softplus` (SFPU) | 15 | 3.27 | 3.29 | +3.05 % | +1.99 % ± 6.06 | 6/15 |
| `idle` control (0 `locate` calls) | 15 | 2.14 | 2.21 | −2.07 % | +0.35 % ± 5.20 | 7/15 |
| **null control** (base vs *identical copy*) | 9 | 13.70 | 13.58 | −0.88 % | +0.69 % ± 3.28 | 5/9 |

**Verdict: a real but modest win on the one workload where `locate` is hot
(`four`, ~5 %), and an honest null everywhere else.** Only `four` has a CI
that excludes zero. The null control — base against a byte-identical copy of
itself — lands on 5/9 wins and a CI straddling zero, which is what licenses
reading `four`'s 16/18 as signal rather than harness bias.

Two caveats stated as measured, not smoothed over:

- **The `four` point estimate is soft.** Its two independent 9-round runs gave
  −8.31 % and −3.48 %; pooling them is the honest summary and the CI
  (−1.7 % to −8.7 %) is wide. Quote "a few percent on RV-bound workloads", not
  −5.2 %.
- **The magnitude exceeds what the mechanism explains, and that is
  unresolved.** 671,485 calls × ~110 ns saved ≈ 0.07 s, which is ~0.6 % of
  `four`'s ~12.9 s — an order of magnitude below the measured −5 %. Either the
  per-call saving is larger in situ than the replay-loop microbenchmark
  captures, or some of the −5 % is machine drift the null control did not
  happen to sample. Do not treat the end-to-end figure as mechanism-verified.
- The `idle` control was *verified* zero, not assumed: instrumenting `locate`
  during its timed window counts **0 calls**. (It needs
  `TT_SIM_PUMP_STRIDE=0`; with striding on, the pump skips idle cycles and
  300 k of them finish in 1.5 ms.)

The ROADMAP's 3–7 % was always the ceiling for
`convert_addr_to_target_range` + `locate` + `_publish_mem_event` *together*;
this change addresses only the middle term, and only its hit path. The other
two are untouched and remain available.

## Gates

`pytest tt_sim/ driver/` **983 passed**; all **26/26** Blackhole replay guards
pass standalone (each validates its own DRAM result, so identical output is
proven, not asserted by eye); `ruff check` / `ruff format` clean.

## Reproducing

```bash
# two frozen trees differing only in memory_map.py
git archive HEAD | tar -x -C /tmp/ab/base
git archive HEAD | tar -x -C /tmp/ab/change && cp memory_map.py /tmp/ab/change/tt_sim/memory/

# interleave: odd rounds base-first, even rounds change-first, pair within round
for r in $(seq 1 9); do ...; done   # >=9 rounds; report the paired median

# always run the null control too — base against a copy of base
cp -a /tmp/ab/base /tmp/ab/base2    # any signal here is harness bias

# hit rate (deterministic, load-independent): wrap MemoryMap.locate and
# classify each call as cached / full-lookup / miss.
```

# What landed: the `TT_SIM_CYCLES_PER_POLL` term — a pump that costs nothing when nothing can happen

**Measured 2026-08-06** against `fd4e806`, machine as elsewhere in this
document (12th Gen Core i7-12700H, 16 threads, CPython 3.12, `TT_SIM_THREADED`
unset). This is the ROADMAP's "smarter `TT_SIM_CYCLES_PER_POLL` default" item.
**The default did not change, and the reason is the first result below.**

## Measure first: the premise was stale

The item's evidence was `programming_examples/vecadd_multi_core` on Wormhole at
the full 8x10 grid — "17 minutes on the kernel and still in the DRAM readback
60 minutes later" at the default poll of 100 — and the standing advice
everywhere was to set `TT_SIM_CYCLES_PER_POLL=10` by hand. That observation
predates firmware-loop parking. Re-run, same program, same 80 workers, same
machine, no grid override:

| | wall | result |
|---|---|---|
| default poll (100) | **105.6 s** | PASS, "all results match expected values within tolerance" |
| `TT_SIM_CYCLES_PER_POLL=10` | 82.7 s | PASS |

which looks like a 22 % gap until the two are *instrumented*, at which point
the gap disappears (81.0 s at 100, 79.7 s at 10 — and instrumentation only
ever makes a run slower). The instrumentation says why:

| | poll=100 | poll=10 |
|---|---|---|
| pump calls | 2,961 | 3,349 |
| ...of which fully dormant | 2,545 (**0.48 s total**) | 2,509 (0.44 s) |
| ...of which live | 416 (70.9 s) | 840 (68.5 s) |
| tile ticks | 3,179,034 | 1,112,208 |
| **live tile ticks** | **483,351** | **482,320** |

The simulator does the *same real work* either way, to within 0.2 %. The whole
of the poll knob's effect is how many *dormant* tile visits it buys, and those
are 0.5 s of an 81 s run. 105.6 vs 82.7 was machine drift, which is the third
time this document has recorded that lesson. **The hour-long readback was
killed by firmware-loop parking** (`tt_sim/pe/rv/spin.py`), exactly as the
2026-08-03 prediction in `docs/upstream-examples-status.md` said it would be:
a worker spinning on a go-message poll now parks, so the post-kernel grid is
genuinely dormant and the readback pumps into a device that strides.

What is left at 8x10 is 483 k live tile ticks at ~147 µs each. That is the
cost of a live Tensix tile, not of the pump, and it is now ROADMAP item 4.

## What changed anyway: the per-*call* cost

Phase 4 striding removed the pump's per-*cycle* cost on a sleeping grid; it
left the per-*call* one, which is what the wire bridge pays. `run()` ticks the
window's opening cycle over every registered tile clock and then probes every
tile for a deadline, so a host DMA readback — thousands of messages into a
parked grid — cost O(tiles) per message whatever the poll budget. Two changes
in `tt_sim/device/clock.py`, both cycle-neutral by construction:

- **`MultiTileClock.quiescent_until`** — the stride already computes "earliest
  cycle anything needs attention"; it is now kept rather than discarded, and a
  later `run` that ends before it skips the window whole: no ticks, no probes,
  same simulated time, same `dormant_cycles` credit. Nothing ticked means
  nothing changed, so the horizon survives its own use.
- **The dormant tick is inlined.** The tick pass still walks every tile in
  registration order — order is observable, since a tile ticked early can wake
  one ticked later, which then runs in the same cycle — but a tile with
  nothing awake and no due deadline does exactly one thing inside
  `clock_tick`, credit a dormant cycle, so the pump does that itself instead
  of paying for the call.

`TileClock.wake()` invalidates the horizon, so the six sites that assigned
`clock.awake = True` directly now call it, and `clock_test.py` scans the tree
for any other assignment — turning the plan doc's "wake-hook completeness"
risk from a claim into a test.

## Cycle-neutrality, checked rather than argued

Every guard on both architectures, run under a shim that records the pump's
final `clock_tick_num`, the number of *live* tile ticks, `stride_skipped_cycles`
and summed `dormant_cycles` — with the watchdog at its default **and** with
`TT_SIM_DEADLOCK=0`, since the watchdog's sampling cadence is what bounds a
stride:

```
diff base.txt head.txt  ->  no differences, 4 runs of 44 guards
```

**No guard's simulated cycle total moved, and neither did its tick
accounting.** That is stronger than "the values still match": the two trees
tick the same components on the same cycles. It also proves something about
the first change — `stride_skipped_cycles` is identical, so
`quiescent_until`'s fast path fires **zero times on any in-tree guard**. It
cannot: every wire message addresses a tile and therefore wakes it, and a woken
tile genuinely needs the cycle. It is kept because it makes "the pump does
nothing when nothing can advance" true by construction rather than by
measurement, and because it is what makes *raising* the knob free; it is not
where the guard-visible saving comes from.

## The A/B

Two frozen trees: `git archive fd4e806` as base, this worktree as head,
alternating order every round, paired within round.

Pump microbenchmarks — per `run(100)` call on Blackhole, `TT_SIM_DEADLOCK=0`,
7 rounds, medians (µs):

| tiles | shape | base | head | factor | base spread / head spread |
|---|---|---|---|---|---|
| 1 | quiescent | 4.7 | 1.9 | 2.5× | [2.4–7.6] / [1.1–2.2] |
| 20 | quiescent | 13.0 | 2.5 | 5.2× | [11.6–26.9] / [2.2–4.4] |
| 80 | quiescent | 58.7 | **6.1** | **9.6×** | [33.2–66.8] / [5.8–11.6] |
| 1 | readback | 8.8 | 6.4 | 1.4× | [5.2–13.3] / [4.1–6.9] |
| 20 | readback | 20.4 | 12.9 | 1.6× | [15.6–30.4] / [10.5–26.6] |
| 80 | readback | 40.8 | **30.5** | 1.3× | [38.1–78.1] / [28.1–56.1] |
| 1 | kernel | 1774 | 1249 | 1.4× | [1298–2097] / [981–2296] |
| 20 | kernel | 2185 | 1802 | 1.2× | [1832–3048] / [1438–2502] |
| 80 | kernel | 4382 | **2398** | **1.8×** | [3417–4657] / [2203–3314] |

*quiescent* = nothing poked; *readback* = one DRAM tile woken then pumped (a
host DMA page read); *kernel* = one permanently-live Tensix tile among N
(the host polling a go message). The shape is the point: the quiescent cost is
now nearly flat in grid width where it was linear, and the marginal cost of a
materialised-but-sleeping tile per pumped cycle falls ~3×.

End to end, 7 rounds, medians (s):

| workload | base | head | delta | base spread / head spread |
|---|---|---|---|---|
| **control** (`pytest tt_sim/util tt_sim/memory`, no pump at all) | 1.14 | 1.19 | +4.4 % | [0.99–1.29] / [1.06–1.40] |
| `blackhole/pad_multi_core` (2048 interleaved DRAM page reads) | 18.59 | 18.22 | −2.0 % | [16.94–20.67] / [15.08–19.03] |
| `blackhole/six` (matmul) | 7.01 | 6.94 | −1.0 % | [6.35–9.37] / [6.15–9.97] |
| `wormhole/examples` | 14.29 | 14.53 | +1.7 % | [13.53–23.38] / [14.15–20.04] |
| `blackhole/vecadd_sharding` (4 workers) | 3.71 | 3.58 | −3.4 % | [3.14–5.05] / [3.32–4.39] |

**Honest reading: nothing here is signal.** The verified-zero control moved
+4.4 %, which is larger than three of the four real deltas, so the noise floor
on this machine swallows the whole effect. That is expected and is the point
of the first section: at 4–8 materialised tiles the per-message-per-tile term
is single-digit microseconds against seconds of real work. The change pays at
grid width, which the microbenchmarks isolate and which only a live wide-grid
run exercises:

| live workload | rounds | base median (s) | head median (s) | delta | head wins |
|---|---|---|---|---|---|
| `vecadd_multi_core`, Wormhole **8x10 = 80 workers**, default poll | 6 | 81.75 | 80.28 | −1.8 % | 3/6 |
| control, same rounds | 6 | 1.19 | 1.24 | +4.5 % | 2/6 |

**Also a null, and that is the honest headline.** Each arm threw exactly one
105 s outlier out of six — which is the same drift that produced the 105.6 vs
82.7 "22 % gap" at the top of this round, now caught symmetrically by
interleaving. At 80 workers the pump is ~1 s of an 80 s run whichever tree you
use; the other 71 s is 483 k ticks of live Tensix tiles.

So: the change is a **structural** fix, not a wall-clock one. It removes the
last term that scaled with grid width × poll budget — provably, in the
microbenchmarks, and with every guard's cycle accounting byte-identical — and
what it exposes underneath is that the pump was no longer the problem.

## Gates

`pytest tt_sim/ driver/` **1074 passed** (1071 + 3 new) with the model off and
again with `TT_SIM_COST_MODEL=1`; **30/30** Blackhole replay guards standalone;
`driver/tests/cost_model_gate.py --jobs 4` **PASS** over 44 discovered guards
with **no poll-budget multiplier change** (`dramtop` 1×, `two` 2×, `offline`
4×, identical to base); `ruff check` / `ruff format` clean.

## Reproducing

```bash
export PYTHONPATH=~/tt-sim
git archive fd4e806 | tar -x -C /tmp/ab/base     # frozen base

# the premise check — the exact configuration the roadmap item cited
cd "$TT_METAL_RUNTIME_ROOT/build/programming_examples"
WH80=$(python3 -c "print(','.join(f'{x}-{y}' for y in [1,2,3,4,5,7,8,9,10,11] \
                             for x in [1,2,3,4,6,7,8,9]))")
TT_METAL_SIMULATOR=~/tt-sim/driver/wormhole TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_SIM_TENSIX_COORDS=$WH80 ./metal_example_vecadd_multi_core   # no grid override

# where its time goes: a sitecustomize.py on PYTHONPATH that wraps
# MultiTileClock.run (timing each call, bucketing by whether the window strided
# to its end) and TileClock.clock_tick (counting live vs dormant visits).

# cycle neutrality: same shim, dump clock_tick_num / live ticks /
# stride_skipped_cycles / summed dormant_cycles per guard, diff base vs head.
# Run it twice, with the watchdog on and with TT_SIM_DEADLOCK=0.

# pump microbenchmark: build a Blackhole with N Tensix tiles, run(300) to
# settle, then time run(100) in three shapes — untouched, with one DRAM tile
# read first, and with one tile's wake_probe pinned to cycle+1. Alternate
# trees every round. Poke tiles with clock.wake(), never clock.awake = True,
# or the benchmark measures the fast path instead of the workload.
```
