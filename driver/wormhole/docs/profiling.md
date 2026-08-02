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
| `TT_SIM_CYCLES_PER_POLL=N` | Cycles to run after each wire message (default 100). Tighten for more deterministic state dumps; loosen for faster runs. |

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
target from the one the ranking above named.

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
