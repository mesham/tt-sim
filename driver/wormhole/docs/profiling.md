# Profiling and analysing tt-sim runs

This is a walkthrough of every profiling and tracing output the
simulator can produce, from the point of view of a user running a
tt-metal program against tt-sim. By the end you'll know what each
output gives you, how to enable it, and which downstream tool to feed
it into.

The flow assumed throughout:

```
tt-metal CLI ──spawns──▶ UMD ──spawns──▶ driver/wormhole/run.sh
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

Throughout this tutorial we'll use the bundled standalone example
`driver/wormhole/one/one.py` (BRISC reads two vectors from DRAM, adds
them, writes the result back) as the concrete subject. **The same env
vars apply identically when your driver is a real tt-metal program
invoked via UMD** — the simulator's entry point is the same in both
cases.

## Prerequisites

```bash
cd ~/tt-sim
pip install -e .                   # picks up numpy, yaml, pynng,
                                   # flatbuffers, pyarrow, pyelftools
```

The tracing infrastructure is always compiled in. With no env vars
set, the bus disables itself and each publish call costs ~88 ns (a
single attribute check) — leaving tracing off has essentially zero
runtime impact.

For some downstream tools you'll also want, in addition:

- `duckdb` — for SQL-querying the Parquet outputs.
- `kcachegrind` (or `qcachegrind`) — for the memory hotspot viewer.
- `genhtml` (from the `lcov` package) — for the source-coverage HTML
  report.
- `spike` — the official Spike RISC-V simulator, only if you want to
  do RISC-V differential testing.

The simulator works fine with none of them installed; you only need
each one for the matching output format.

## A 30-second smoke test

Enable everything at once and run the example:

```bash
cd driver/wormhole

TT_SIM_TRACE=/tmp/out/events.jsonl \
TT_SIM_TRACE_PERFETTO=/tmp/out/timeline.json.gz \
TT_SIM_TRACE_COMMITLOG=/tmp/out/commitlog/ \
TT_SIM_TRACE_COUNTERS=/tmp/out/counters/ \
TT_SIM_TRACE_NOC=/tmp/out/noc/ \
TT_SIM_TRACE_MEMORY=/tmp/out/memory.callgrind \
TT_SIM_TRACE_INVARIANTS=/tmp/out/invariants.jsonl \
TT_SIM_TRACE_STATE_DUMP=/tmp/out/states/ \
python3 one/one.py
```

After ~5 seconds the example finishes; `ls /tmp/out` shows eight
outputs (a JSONL file, a gzipped Perfetto trace, a commitlog
directory, three Parquet datasets, a Callgrind text file, an
invariants log, and a state-dump directory). The rest of this
tutorial unpacks what each one is for.

In the **tt-metal flow**, the same `export` commands precede whatever
tt-metal binary spawns the simulator — exactly the same env vars
flow through UMD into run.sh into the simulator.

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
TT_SIM_TRACE=/tmp/out/events.jsonl python3 one/one.py
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
single-kernel example is ~10 MB.

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
TT_SIM_TRACE_PERFETTO=/tmp/out/timeline.json.gz python3 one/one.py
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
TT_SIM_TRACE_COMMITLOG=/tmp/out/commitlog/ python3 one/one.py
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
TT_SIM_TRACE_COUNTERS=/tmp/out/counters/ python3 one/one.py

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
TT_SIM_TRACE_NOC=/tmp/out/noc/ python3 one/one.py
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
TT_SIM_TRACE_MEMORY=/tmp/out/memory.callgrind python3 one/one.py
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
python3 one/one.py
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
  attribution. Build kernels you care about with debug info.
- The firmware bundled in `driver/wormhole/firmware/` is stripped,
  so to use this for tt-metal flows you'll want to rebuild
  firmware/kernels with `-g` from the tt-metal fork.

---

## 8. Architectural invariants — `TT_SIM_TRACE_INVARIANTS`

**What:** A small catalogue of architectural rules checked over the
event stream; violations are written to a JSONL log. Today's seed
catalogue: PC alignment, memory access alignment, kernel-lifecycle
ordering, NoC request/response pairing.

**Why:** Catches "the kernel produced the right output but did
something nominally illegal along the way" — broken alignment,
orphan NoC responses, etc. All 8 bundled examples run cleanly with
zero violations, so a non-empty violations log on those workloads is
a real regression.

**Enable:**

```bash
TT_SIM_TRACE_INVARIANTS=/tmp/out/invariants.jsonl python3 one/one.py
```

For CI use, set `TT_SIM_TRACE_INVARIANTS_STRICT=1` to additionally
raise on the first violation (the simulator will exit non-zero):

```bash
TT_SIM_TRACE_INVARIANTS=/tmp/out/invariants.jsonl \
TT_SIM_TRACE_INVARIANTS_STRICT=1 \
python3 one/one.py
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
   simulator) on the same kernel and diff state dumps. tt-sim ships
   the comparison primitive today; you provide the orchestration.

**Enable:**

```bash
TT_SIM_TRACE_STATE_DUMP=/tmp/out/states/ python3 one/one.py

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
between runs today because the host polling loop in
`wormhole_driver.py` runs `wormhole.run(100)` then checks the
go-message mailbox — so `kernel_done` fires within a 100-cycle
window after the kernel actually completes, and BRISC may execute
slightly different firmware-loop bytes in that window. For
byte-exact regression comparison either dump at a deterministic
sub-checkpoint or tighten the polling loop. The diff tool itself is
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
# profile-run.sh — run a tt-metal binary (or example) under tt-sim
# with every profiling writer enabled.
set -euo pipefail
OUT="${1:-/tmp/tt-sim-out}"
shift || true   # remaining args are the binary to run
mkdir -p "$OUT"

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

Used as e.g. `./profile-run.sh /tmp/myrun python3 one/one.py` or
`./profile-run.sh /tmp/myrun ./build/programming_examples/loopback`.

## How this works under tt-metal

`Wormhole.__init__` (in `tt_sim/device/tt_device.py`) is the only
place that reads any of these environment variables, via the
`enable_from_env()` helper from `tt_sim/trace/auto.py`. That helper
runs whether the simulator is constructed from a standalone driver
script (`python3 one/one.py`) or from the wire-bridge server
(`driver/wormhole/server/__main__.py`) that UMD spawns for tt-metal.
So **the env vars apply identically** in both flows:

```
# Standalone:
TT_SIM_TRACE_PERFETTO=/tmp/timeline.json.gz python3 one/one.py

# tt-metal:
TT_SIM_TRACE_PERFETTO=/tmp/timeline.json.gz ./build/programming_examples/loopback
```

The tt-metal binary spawns UMD, UMD reads env then spawns
`driver/wormhole/run.sh`, run.sh execs the server, the server
constructs Wormhole, and `enable_from_env` picks up the same
variables. No code changes are needed in tt-metal, no changes to your
program — just env vars.

The diagnostic flags from
[`driver/wormhole/README.md#enabling-diagnostics-in-the-tt-metal-flow`](../README.md#enabling-diagnostics-in-the-tt-metal-flow)
(`TT_SIM_DIAG_BRISC`, `TT_SIM_DIAG_NOC0`, the `_CO_*` series for
Tensix coprocessor diagnostics, etc.) compose freely with the
`TT_SIM_TRACE_*` env vars — they're orthogonal: diagnostics print
human-readable text to stderr, trace writers produce machine-readable
artefacts on disk.

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
