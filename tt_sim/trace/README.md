# Structured tracing (`tt_sim/trace/`)

A typed pub/sub event bus. The simulator publishes architectural events
(instruction retirement, Tensix dispatch, NoC traffic, kernel lifecycle);
external consumers subscribe and turn the stream into whatever they
need. Today: JSONL for ad-hoc analysis and Perfetto / Chrome Trace
Event Format for visual timelines on `ui.perfetto.dev`. Later phases
add Spike commitlog, Parquet counters, Cachegrind, and LCOV writers
as additional consumers of the same bus. tt-sim does not build viewers.

> **Consuming a trace rather than extending the simulator?** Read
> [`docs/trace-schema.md`](../../docs/trace-schema.md) instead. That is
> the stable, versioned contract — every field's meaning, unit and
> stability, plus the traps — written so you never have to open this
> directory. This file is the implementer's guide.

All seven phases of [ROADMAP §H](../../ROADMAP.md) are complete (event
bus, Perfetto, Spike-compatible commitlog, Parquet/DuckDB counters, NoC
Parquet + Cachegrind memory trace, LCOV source-level attribution,
invariants + state-dump diff testing); read that for the full plan.

## Quick start

```bash
export PYTHONPATH=~/tt-sim:$PYTHONPATH
cd driver/wormhole
TT_SIM_TRACE=/tmp/one.jsonl python3 one/one.py                 # JSONL for ad-hoc
TT_SIM_TRACE_PERFETTO=/tmp/one.json.gz python3 one/one.py      # visual timeline
TT_SIM_TRACE_COMMITLOG=/tmp/one/ python3 one/one.py            # Spike-compat per-core
TT_SIM_TRACE_COUNTERS=/tmp/one-counters/ python3 one/one.py    # Parquet counters / DuckDB
TT_SIM_TRACE_NOC=/tmp/one-noc/ python3 one/one.py              # Parquet NoC transactions
TT_SIM_TRACE_MEMORY=/tmp/one-mem.callgrind python3 one/one.py  # KCachegrind memory hotspots
TT_SIM_TRACE_LCOV=/tmp/one.lcov \
  TT_SIM_TRACE_LCOV_ELFS=path/to/kernel.elf,path/to/firmware.elf \
  python3 one/one.py                                            # Source-level coverage
# All seven can be enabled at once — writers subscribe independently:
TT_SIM_TRACE=/tmp/one.jsonl \
TT_SIM_TRACE_PERFETTO=/tmp/one.json.gz \
TT_SIM_TRACE_COMMITLOG=/tmp/one/ \
TT_SIM_TRACE_COUNTERS=/tmp/one-counters/ \
TT_SIM_TRACE_NOC=/tmp/one-noc/ \
TT_SIM_TRACE_MEMORY=/tmp/one-mem.callgrind \
    python3 one/one.py
```

### JSONL output

`TT_SIM_TRACE=<path>` produces two files:

- `<path>` — one JSON object per line, every event the simulator
  published.
- `<path>.ids.json` — sidecar mapping every `unit_id` tuple in the
  trace to its `(chip_id, core_y, core_x, unit)` decomposition.

Inspect with `jq` / `duckdb` / `pandas`. Example: count events per
category:

```bash
python3 -c "
import json
from collections import Counter
print(Counter(json.loads(l)['category'] for l in open('/tmp/one.jsonl')))
"
```

A `four/four.py` run reports roughly
`{mem: 65k, instr: 13.5k, dispatch: 864, compute: 309, noc: 12,
lifecycle: 4, sync: 4}` — all seven categories populated.

### Perfetto / Chrome trace output

`TT_SIM_TRACE_PERFETTO=<path>` writes Chrome Trace Event Format JSON.
A `.gz` extension enables gzip compression (Perfetto loads it
natively; traces compress 10-20×).

1. Visit <https://ui.perfetto.dev>.
2. Drag-and-drop your `*.json.gz` file onto the page.
3. The timeline renders one row per Tensix tile, with per-unit lanes
   (BRISC / NCRISC / TRISCn / MATRIX / SFPU / NOC0 / NOC1 / etc.).
   NoC transactions render as **arrows** from the requesting NUI to
   the responding NUI (flow events). Lifecycle events (firmware /
   kernel start / done) render as global instant markers.

Canned SQL queries that run in Perfetto's **Query (SQL)** tab live in
[`queries/README.md`](queries/README.md) — top slices by duration,
per-unit event counts, and NoC roundtrip latency. The schema is
documented at <https://perfetto.dev/docs/analysis/sql-tables>.

**Cycle-as-time mapping:** events are stamped with `ts = cycle`, one
trace microsecond per simulated cycle. Events with `cycle == 0` (mem /
sync / lifecycle) are stamped with the highest cycle seen so far, so
they appear at the "current time" rather than collapsing onto t=0.

**Durations are real, and say which regime produced them.** A slice is
only ever as wide as a number the simulator actually holds:

| slice | width | source |
|---|---|---|
| `compute` (`ELWADD`, `ATCAS`, …) | modelled occupancy | `ComputeEvent.duration` ← `tensix_instruction_costs.yaml` |
| `noc:<phase>:<type>` | issue → arrival | `NoCEvent.issue_cycle` vs. the event's cycle |
| `stall:<reason>` | cycles the core was held | `InstrEvent.stall_cycles` ← the RV load-use interlock |
| `pc=0x…`, `dispatch:…` | 1 cycle | a core retires one instruction per cycle; issue is single-cycle |

With **`TT_SIM_COST_MODEL` unset every one of those is 1**, and that is
the truth rather than a placeholder: with the model off nothing stalls
and every unit retires in the tick it was issued. The writer will not
invent a plausible number to fill the gap. Which regime a trace came
from is stated three ways so a file copied away from its run is never
ambiguous:

- `otherData.cost_model` (plus a one-line `timing_note`) in the file
  trailer — Perfetto shows it under **Info and stats**;
- a `process_labels` metadata event per tile, reading
  `TT_SIM_COST_MODEL on` / `off`;
- a `timing_model` argument on any individual slice whose width is not
  a modelled figure.

NoC transactions are emitted as **async** slices (`ph` `b`/`e`) rather
than `X`, because a NIU can have several packets in flight at once and
partially overlapping `X` slices are not legally nestable — Perfetto
drops them. The request→response arrows ride on the slices themselves
as `bind_id` + `flow_out`/`flow_in`; a standalone `s`/`f` flow event
binds to the enclosing slice of a *thread* track, which an async slice
is not, and would be reported as `flow_no_enclosing_slice`.

Verified by loading the output into Perfetto's own `trace_processor`
(the engine behind `ui.perfetto.dev`): a `four` run with the cost model
on imports with zero `error`/`data_loss` stats, 77,413 `stall:load_use`
slices totalling 80,570 cycles, and NoC flights up to 235 cycles.

**`mem` events are skipped** from the Perfetto stream — their volume
(~50k+ per kernel) would swamp the UI without adding slice-level
signal. They remain in the JSONL output.

### Spike-compatible commitlog (RISC-V)

`TT_SIM_TRACE_COMMITLOG=<dir>` writes one file per baby core
(`brisc.commitlog`, `ncrisc.commitlog`, `trisc0.commitlog`, …) in the
exact format `spike --log-commits` produces:

```
core   0: 3 0x00003780 (0xffb001b7) x 3 0xffb00000
core   0: 3 0x00003784 (0x7f018193) x 3 0xffb007f0
core   0: 3 0x00003788 (0xffb01137) x 2 0xffb01000
```

The trailing `x<N> 0x<value>` is omitted on instructions that don't
write to an architectural register (stores, branches, jumps without
link, writes to `x0`). All files report `core   0:` and machine-mode
privilege `3` — tt-sim's per-unit commitlog files are drop-in
comparable against a single-hart Spike run via plain `diff`.

**Differential testing.** A small helper compares two commitlog files
and reports the first divergence with five lines of context:

```bash
spike --log-commits ./test.elf > /tmp/spike.commitlog
TT_SIM_TRACE_COMMITLOG=/tmp/ttsim/ python3 your_driver.py
python3 -m tt_sim.trace.diff_spike /tmp/ttsim/brisc.commitlog /tmp/spike.commitlog
```

Caveat: the diff workflow is only meaningful for pure-RV32IM ELFs that
run identically under both simulators. tt-metal firmware kernels touch
Tensix / NoC MMIO that Spike has no concept of, so a diff there would
diverge on the first such access — useful for catching specific RV
mistakes (e.g. when the §B RV pipeline modelling lands), not for
end-to-end kernel correctness.

### Performance counters (Parquet)

`TT_SIM_TRACE_COUNTERS=<dir>` writes a Hive-partitioned Parquet
dataset (`chip=N/kernel_id=N/*.parquet`) of running performance
counters derived from the existing event stream — per-unit instruction
counts, per-target dispatch breakdowns, NoC bytes, memory op counts
per region (L1 / MMIO), and sync-event tallies. Flush cadence is
configurable via `TT_SIM_TRACE_COUNTERS_INTERVAL=N` (default: every
100 cycles). Kernel boundaries always force a flush and bump
`kernel_id` so per-kernel deltas are easy to query.

DuckDB reads the dataset directly:

```bash
duckdb -c "
  SELECT counter_name, SUM(value) AS total
  FROM read_parquet('/tmp/counters/**/*.parquet', hive_partitioning=true)
  GROUP BY counter_name
  ORDER BY total DESC
  LIMIT 10
"
```

Canned queries (top counters by total, per-unit instruction counts,
kernel-to-kernel diff, NoC hotspot detection) live in
[`queries/counters.sql`](queries/counters.sql).

Where §I supplies the state, the dataset also carries cycle
attribution — same long format, no consumer changes:

| counter | meaning |
|---|---|
| `stall_cycles`, `stall_load_use`, `stall_store_rate`, `stall_integer_unit` | per baby core, cycles the RV cost model held an instruction, split by reason |
| `busy_cycles` | per Tensix backend unit, the occupancy the cost tables charged |
| `bookkeeping_cycles` | a **subset** of the same cycles: Matrix Unit opcodes that move no operand data (RWC counters, dvalid flags, SrcB operand cache). `busy_cycles - bookkeeping_cycles` is datapath work, which is what an energy model wants and an occupancy reader does not |
| `noc_flight_cycles`, `noc_txns_timed` | per NIU, issue→arrival summed over timed transactions |

The stall and busy counters are **absent, not zero**, with
`TT_SIM_COST_MODEL` unset: a counter row only exists once something
incremented it, so an un-modelled run's dataset does not assert a
stall-free machine. `noc_flight_cycles` is emitted in both regimes
because a flight time is measured rather than modelled — it is just
always 1 with the model off. Measured on the Blackhole `four` guard:

```
                     model off      model on
stall_cycles                 -        80,695
stall_load_use               -        80,570
stall_store_rate             -           100
stall_integer_unit           -            25
busy_cycles                  -           267
noc_flight_cycles           36         3,384
```

Still gated on §I, with nothing to read yet: packer back-pressure and
unpacker idle cycles (neither unit is wired to the tables — the packer
charges its `PACR` issue cost only, the unpacker is uncosted), and L1
bank conflicts (tt-sim models no banks).

### NoC transactions (Parquet)

`TT_SIM_TRACE_NOC=<dir>` writes a Hive-partitioned Parquet dataset
(`chip=N/*.parquet`) — one row per `NoCEvent` emission with columns
`cycle, chip, core_y, core_x, unit, phase, txn_type, src_x, src_y,
dst_x, dst_y, size_bytes, txn_id, issue_cycle, arrival_cycle,
flight_cycles, cost_model`. Suitable for SQL queries about data
movement:

```bash
duckdb -c "
  SELECT src_y, src_x, dst_y, dst_x, SUM(size_bytes) AS bytes
  FROM read_parquet('/tmp/noc/**/*.parquet', hive_partitioning=true)
  WHERE phase = 'response'
  GROUP BY src_y, src_x, dst_y, dst_x
  ORDER BY bytes DESC
"
```

`issue_cycle` is when the *sending* NIU put the packet on the wire and
`arrival_cycle` (== `cycle`) when the receiving NIU serviced it, so
`flight_cycles` is their difference. Both regimes measure it, but a
`flight_cycles` of 1 means different things in each, which is what the
`cost_model` column is for: with `TT_SIM_COST_MODEL` unset a packet is
delivered on the next cycle however far it travelled. `issue_cycle` is
`-1` (and `flight_cycles` `0`) for the one case neither regime can
time — a NIU with no owning tile clock, i.e. the unit tests and
`driver/simple`.

```bash
duckdb -c "
  SELECT phase, txn_type, avg(flight_cycles), max(flight_cycles), any_value(cost_model)
  FROM read_parquet('/tmp/noc/**/*.parquet', hive_partitioning=true)
  WHERE issue_cycle >= 0 GROUP BY 1, 2
"
# model off:  request/read flight 2, response/read 1
# model on:   request/read flight 235, response/read 46  (Blackhole `four`)
```

**VC occupancy remains gated on §I** — tt-sim models no virtual
channels, so there is no `vc` column rather than a column of zeroes.

### Memory accesses (Callgrind / KCachegrind)

`TT_SIM_TRACE_MEMORY=<file>` writes a single Callgrind text file that
`kcachegrind` / `qcachegrind` / `callgrind_annotate` consume directly.
Each unique `(region, pc, address)` is bucketed into a `Dr` / `Dw`
count; functions group by `(region, pc)` so the call-cost tree
collapses cleanly per instruction:

```
events: Dr Dw
positions: instr
ob=tt-sim
fl=memory
fn=L1_pc_0x00003780
0x20 1 0
0x24 1 0
```

Open with `kcachegrind /tmp/run.callgrind` and the source-attribution
tree shows hottest PCs, with addresses underneath. Accesses without a
PC (NoC-driven or internal-engine traffic; ~10% of the typical trace)
group under a synthetic `<region>_no_pc` function.

<!-- BEGIN: ranked bottleneck report (tt_sim/trace/report.py) -->
### Ranked bottleneck report (`TT_SIM_PROFILE`)

The one-shot entry point. `TT_SIM_PROFILE=<dir>` enables the counter
aggregator and a per-PC hotspot aggregator, auto-discovers the ELFs
tt-metal built, and writes a ranked `report.md` (plus `report.json`,
`hotspots.json`, `profile.json` and the raw `counters/` dataset) at
exit. Set `TT_SIM_COST_MODEL=1` alongside it or nothing can stall.

```bash
TT_SIM_COST_MODEL=1 TT_SIM_PROFILE=/tmp/myrun ./build/six
python3 -m tt_sim.trace.report /tmp/myrun --stdout     # re-render
```

Three design points that matter to anyone extending this:

- **DWARF resolution is post-processing.** `HotspotAggregator` keys on
  the raw PC (one dict increment per retirement) and resolves through
  `DwarfIndex` once, at close, over the few thousand distinct PCs a run
  touches. A DWARF lookup per `InstrEvent` would be a real cost on a
  tree whose tracing overhead is already over budget.
- **The report generalises over counter names.** `stall_<reason>` is an
  RV stall with that reason; anything else ending in `_cycles` is a
  cycle-bearing counter and is ranked too, marked *(discovered)*. A new
  stall reason or a new occupancy counter appears in the output with no
  change to `report.py`.
- **Attribution refuses to guess.** `DwarfIndex.nearest` answers only
  inside the address ranges the line program actually covers, and
  discovery leaves a core unattributed when its resident code matches no
  candidate ELF. An index built from the wrong ELF would otherwise
  answer every query and report ~100 % coverage of a kernel that never
  ran.

The user-facing walkthrough is
[`driver/wormhole/docs/profiling.md` §0](../../driver/wormhole/docs/profiling.md).
<!-- END: ranked bottleneck report -->

### Source-level attribution (LCOV)

`TT_SIM_TRACE_LCOV=<file>` plus `TT_SIM_TRACE_LCOV_ELFS=<elf1,elf2>`
writes an LCOV-format coverage file mapping retired-instruction counts
back to source lines via DWARF info parsed from the supplied ELFs.
Output is consumable by:

- `genhtml file.lcov -o report/` → static HTML report with hot-line
  heatmap.
- The VS Code [Coverage Gutters extension](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters)
  renders inline decorations next to source.
- GitHub Codecov and most CI coverage reporters accept LCOV directly.

```bash
# Build kernels with -g, then:
TT_SIM_TRACE_LCOV=/tmp/run.lcov \
TT_SIM_TRACE_LCOV_ELFS=build/brisc_kernel.elf,build/trisc0_compute.elf \
  python3 four/four.py
genhtml /tmp/run.lcov -o /tmp/cov/
xdg-open /tmp/cov/index.html
```

"Coverage" semantics: the `DA:<line>,<count>` value is the number of
retirements at any PC mapping to that line — i.e. cycles spent at that
line, not "executed at least once". Tools render this identically to
standard coverage (heatmap on the source view), but hot lines stand
out instead of just covered lines.

Caveats:

- Source files need to live at the paths DWARF embedded into the
  ELFs (typically absolute paths from the build host). Copy or
  symlink the source tree to match, or post-process the LCOV file
  with `sed`.
- Stripped ELFs (no `-g`) load cleanly but contribute zero
  attribution. Build the kernel/firmware you care about with debug
  info.
- ELFs are loaded once at sim init; lookup is a bisect against the
  loaded line table per `InstrEvent`. It is a *floor* lookup, because a
  line program records a row only where the source position changes —
  exact-PC matching drops most of a run.
- PC ranges from multiple ELFs may overlap (firmware + kernel share
  L1 space). Load each ELF against the unit that runs it
  (`DwarfIndex.load(path, unit="TRISC2")`) to keep them apart; the
  unscoped index still uses last-load-wins, so list kernel ELFs after
  firmware ELFs in `TT_SIM_TRACE_LCOV_ELFS` if you want kernel
  attribution to take priority on collisions.
- `TT_SIM_TRACE_LCOV_ELFS` takes bare paths and does no auto-discovery
  or relocation. For a tt-metal kernel, which tt-metal places at a
  runtime base different from its link address, use `TT_SIM_PROFILE`
  above instead — it finds the ELFs and recovers the load bias.

### Architectural invariants

`TT_SIM_TRACE_INVARIANTS=<file>` enables a small set of seed
invariants that check architectural rules over the event stream and
write violations to a JSONL file. Each invariant subscribes to one
or more event categories and is implemented as a subclass of
`Invariant`. Today's catalogue:

- `PCAlignmentInvariant` — RV32 instructions must be 4-byte aligned.
- `MemAlignmentInvariant` — power-of-two-sized memory accesses align
  to their size.
- `LifecycleOrderInvariant` — `firmware_launch_done` must follow
  `firmware_launch_start`; same for `kernel_start`/`kernel_done`.
- `NoCRequestResponseInvariant` — every NoC response pairs with a
  prior outstanding request for the same `(txn_id, src, dst,
  txn_type)`.

By default violations are collected and logged at exit. Set
`TT_SIM_TRACE_INVARIANTS_STRICT=1` to additionally raise on the first
violation (useful in CI where you want a hard fail). All 8 wormhole
examples run cleanly today (zero violations), so the seed catalogue
also serves as a regression guard — any new violations on these
workloads indicate a real bug.

Adding a new invariant: subclass `Invariant`, declare which
`EventCategory` lanes to subscribe to in `CATEGORIES`, implement
`check(event) -> str | None`, return a message on violation. Add to
`DEFAULT_INVARIANTS` to have `enable_from_env` pick it up.

### State dumps and diff testing

`TT_SIM_TRACE_STATE_DUMP=<dir>` captures a JSON snapshot of relevant
device state at every lifecycle boundary (firmware/kernel
start/done). Today the snapshot includes per-baby-core register
files (32 GPRs + PC) and per-NUI counter sets. Schema is versioned
(`schema_version: 1`).

Compare two dumps with the included tool:

```bash
python3 -m tt_sim.trace.diff_state \
    /tmp/before/kernel_done_0003.json \
    /tmp/after/kernel_done_0003.json
```

Clean runs report `state matches`; divergences pinpoint the first
field that differs (e.g.
`cores.18_18_BRISC.gpr[4]: 0xc8000000 != 0xa868`) and exit non-zero.
The typical workflow: capture a dump on a known-good run, capture
another on a candidate commit/branch, diff. Catches regressions that
perturb counter trajectories or L1 contents without changing visible
kernel results.

Cross-simulator diffing (vs `libttsim.so`) is documented in §H Phase
7 as the strategic goal; the orchestration to drive both simulators
on the same kernel isn't wired here — generate dumps via your own
script and feed both files to `diff_state`.

**Caveat on determinism:** state at "kernel_done" is non-deterministic
between runs today because the host-side polling loop in
`wormhole_driver.py` checks every 100 cycles, so `kernel_done` fires
somewhere in a 100-cycle window after the kernel actually completes.
For byte-exact regression checks, dump at a deterministic
sub-checkpoint instead (or tighten the polling loop). The diff tool
itself is exact; the noise is in *when* the snapshot is taken.

## Programmatic use

The env-driven path above is the easy mode. For finer control,
module-level singletons are exposed directly:

```python
from tt_sim.trace import (
    JSONLLogger,
    get_bus,
    get_registry,
    EventCategory,
)

bus = get_bus()
bus.enabled = True                 # master enable; default off
bus.set_category_enabled(EventCategory.MEM, False)   # opt out of high-volume MEM

logger = JSONLLogger("/tmp/run.jsonl")
# ... run sim ...
logger.close()
get_registry().dump("/tmp/run.ids.json")
```

The bus is **off by default**. With it disabled, each publish site
costs ~88 ns (a single attribute check returning False — measured by
`python3 -m tt_sim.trace.benchmark`), well under the 100 ns target so
hooks can stay compiled in.

## Event schema

Events are frozen `@dataclass(slots=True)` instances. Each subclass
fixes its `CATEGORY` and `SCHEMA_VERSION` as `ClassVar`, so the bus
routes by `type(event)` with no per-instance discriminator overhead.

Common fields on every event:

| Field            | Type    | Meaning                                          |
|------------------|---------|--------------------------------------------------|
| `cycle`          | `int`   | Device-global cycle number at the time of emit.  |
| `unit_id`        | `tuple` | `(chip_id, core_y, core_x, unit)` of the source. |
| `schema_version` | `int`   | Bumps on any breaking change to the event shape. |
| `category`       | `str`   | Routing key — `instr`, `dispatch`, `noc`, etc.   |

`SCHEMA_VERSION` is **4**. Version 4 added the `stall` category and
`StallEvent`; it is additive, so a consumer of versions 1–3 sees the
categories it already knew, unchanged, and simply does not subscribe to
the new one.

Per-type fields published today:

### `InstrEvent` (category `instr`)

Emitted on each RV instruction retirement on the five baby cores
(BRISC, NCRISC, TRISC0–2).

| Field          | Type   |                                                   |
|----------------|--------|---------------------------------------------------|
| `pc`           | `int`  | PC at the retiring instruction.                   |
| `instruction`  | `int`  | Raw 32-bit instruction word.                      |
| `stalled`      | `bool` | True if the core stalled this cycle (no retire).  |
| `stall_cycles` | `int`  | Cycles the cost model held this instruction before it could issue. |
| `stall_reason` | `str`  | `load_use` / `store_rate` / `integer_unit`, or `""`. |

`stall_cycles` / `stall_reason` are cost-model state: `0` / `""`
whenever `TT_SIM_COST_MODEL` is unset, because no RV instruction can
stall then. They are distinct from `stalled`, which is the
Tensix-instruction-buffer back-pressure that exists in both regimes.

Disassembly is **not** included — kept out of the hot path. Decoding is
the consumer's job (every event carries the raw 32-bit word).

### `DispatchEvent` (category `dispatch`)

Emitted when the Tensix wait-gate issues an instruction to a backend
unit. Source `unit_id` is the issuing TRISC (TRISC0/1/2).

| Field         | Type  |                                                       |
|---------------|-------|-------------------------------------------------------|
| `opcode`      | `str` | Tensix instruction name (`ELWADD`, `SFPCONFIG`, ...). |
| `target_unit` | `str` | Backend unit (`MATH`, `SFPU`, `PACK`, `UNPACK`, ...). |
| `thread_id`   | `int` | 0 / 1 / 2 — redundant with `unit_id` for convenience. |

The MOP expander, replay expander, and any direct
`backend.issueInstruction` paths not gated by the wait-gate do **not**
currently publish — see [ROADMAP §H Phase 1 deferred items](../../ROADMAP.md).

### `NoCEvent` (category `noc`)

Emitted at the four `NUI.clock_tick` snoop sites.

| Field        | Type    |                                                   |
|--------------|---------|---------------------------------------------------|
| `phase`      | `str`   | `request` or `response`.                          |
| `txn_type`   | `str`   | `read` or `write`.                                |
| `src`        | `tuple` | Source NoC coord.                                 |
| `dst`        | `tuple` | Destination NoC coord.                            |
| `size_bytes` | `int`   | Transfer size.                                    |
| `txn_id`     | `int`   | NoC transaction ID (reused on issue + response).  |
| `issue_cycle`| `int`   | Cycle the sending NIU put the packet on the wire; `-1` if untimed. |

`cycle` is the *arrival* — this NIU servicing the packet — so
`cycle - issue_cycle` is the flight time. Pair a `request` with its
`response` on `(txn_id, src, dst, txn_type)` for the round trip.

### `LifecycleEvent` (category `lifecycle`)

Emitted from the host-side driver at firmware/kernel boundaries.
`unit_id` is the `(0, 0, 0, HOST)` pseudo-unit.

| Field    | Type  |                                                                                  |
|----------|-------|----------------------------------------------------------------------------------|
| `kind`   | `str` | `firmware_launch_start` / `firmware_launch_done` / `kernel_start` / `kernel_done`. |
| `detail` | `str` | Free-form (e.g. the kernel parameters path).                                     |

`cycle` is 0 for lifecycle events today because they are emitted from
the host side, outside the simulator's clock — treat them as anchors,
not measurements.

### `MemEvent` (category `mem`)

Emitted on every read/write through `MemorySpace` (`tt_sim/memory/memory.py`).
Covers L1, DRAM, and the MMIO range across every tile. Source `unit_id`
is derived from `caller_context` (set by the RV core whose load/store
triggered the access); when unavailable, falls back to
`(0, 0, 0, UNKNOWN)`.

| Field     | Type  |                                                                  |
|-----------|-------|------------------------------------------------------------------|
| `op`      | `str` | `read` or `write`.                                               |
| `address` | `int` | Address of the access.                                           |
| `size`    | `int` | Bytes transferred.                                               |
| `region`  | `str` | `L1` / `MMIO` (coarse, address-derived).                         |

`cycle` is 0 for `MemEvent` today — most accesses originate from
non-clocked dispatch paths (`MemorySpace` is shared, not per-cycle).
Pair with the most recent `InstrEvent` from the same `unit_id` to
recover the cycle of an RV-driven access.

**Volume note:** a typical kernel emits ~50k+ `MemEvent`s. Disable the
category via `bus.set_category_enabled(EventCategory.MEM, False)` if
you don't need it; the `instr / dispatch / compute / sync` events
together are an order of magnitude smaller.

Register-file accesses are deliberately **not** emitted — multiple per
instruction, low marginal signal for the trace volume cost. The
`InstrEvent` already names the retiring instruction; consumers can
decode register operands from the raw `instruction` field if needed.

### `ComputeEvent` (category `compute`)

Emitted as each Tensix backend unit completes an instruction —
single hook in `TensixBackendUnit.clock_tick`
(`tt_sim/pe/tensix/backends/backend_base.py`) covers every
unit uniformly. Source `unit_id` is the per-tile backend unit
(`MATRIX`, `SFPU`, `PACKER`, `UNPACKER`, `MOVER`, `THCON`, `SYNC`,
`TDMA`, `CFG`).

| Field         | Type  |                                                       |
|---------------|-------|-------------------------------------------------------|
| `op`          | `str` | Instruction name (e.g. `ELWADD`, `SFPCONFIG`, `PACK`). |
| `target_unit` | `str` | Unit name string (matches `TensixBackendUnit.unit_name`).|
| `thread_id`   | `int` | Issuing TRISC thread (0/1/2), or -1 if unattributed.   |
| `detail`      | `str` | Free-form per-handler payload (empty today).           |
| `duration`    | `int` | Modelled occupancy in cycles; `0` means *no claim*.    |

`duration` is the cycles `tensix_instruction_costs.yaml` charges the
op. `0` is "the tables have no opinion" — the model is off, the unit is
unwired (unpacker, config, mover, misc), or the opcode is uncosted. A
consumer must not read `0` as one cycle.

`unit_id` and `target_unit` name the same architectural unit **in two
different vocabularies, and they are not interchangeable.** `unit_id[3]`
is the `Unit` enum (`MATRIX`, `SFPU`, `PACKER`, …); `target_unit` here is
the backend class's own `unit_name` (`Matrix`, `Vector`, `Packer`, …) —
and `DispatchEvent.target_unit` / `StallEvent.blocked_on` are a *third*,
the ISA's `ex_resource` (`MATH`, `SFPU`, `PACK`, …). `SFPU` is `Vector`
is `SFPU`; `MATRIX` is `Matrix` is `MATH`. Join on `unit_id`, which is
the canonical key, and translate with
`tt_sim.trace.BACKEND_UNIT_ALIASES` — a join on the wrong one returns
nothing rather than failing. Full table in
[`docs/trace-schema.md`](../../docs/trace-schema.md).

### `SyncEvent` (category `sync`)

Emitted from cross-thread / cross-core synchronisation points.

- Mailbox traffic (`misc/mailbox.py`): `mailbox_send` and `mailbox_recv`.
- Tensix `ttsync` wait points (`misc/ttsync.py`):
  `ttsync_wait_coproc_done` and `ttsync_wait_mop_done`.

| Field    | Type  |                                                            |
|----------|-------|------------------------------------------------------------|
| `kind`   | `str` | One of the strings listed above.                           |
| `detail` | `str` | Free-form (e.g. `thread=N`, `from_idx=N`).                 |

`cycle` is 0 — sync points happen inside memory writes / register
reads which don't carry the device cycle directly. Pair with adjacent
`MemEvent` / `InstrEvent` for timing.

<!-- BEGIN stall-reasons (ROADMAP Tier 1 item 1) -->
### `StallEvent` (category `stall`)

The complement of `DispatchEvent`: that event says an instruction
issued, this one says why the next one could not. Emitted from the
Tensix wait gate (`pe/tensix/frontend.py`), one event per **episode** —
the contiguous run of cycles a thread was held for one reason — not one
per cycle. On `sfpumath` that is 32 events covering 10,793 stalled
cycles.

`unit_id` is the **thread that suffered** the stall (`TRISC0/1/2`);
`blocked_on` names the unit **responsible**. Those are deliberately
different: an unpacker that cannot start because the matrix unit still
owns the Src bank is reported as blocked on `MATH`, not on `UNPACK`.

| Field        | Type  |                                                    |
|--------------|-------|----------------------------------------------------|
| `reason`     | `str` | One of `STALL_REASONS` (below).                    |
| `blocked_on` | `str` | `ex_resource` name of the unit at fault, or `""`.  |
| `cycles`     | `int` | Length of the episode, ≥ 1.                        |
| `opcode`     | `str` | Instruction held at the gate, or `""`.             |
| `thread_id`  | `int` | Issuing Tensix thread, 0–2.                        |
| `semaphore`  | `int` | Semaphore index for a semaphore wait, else `-1`.   |

`reason` is drawn from a frozen vocabulary exported as
`tt_sim.trace.STALL_REASONS`, so a consumer can switch on it
exhaustively. There is deliberately no generic `"stalled"` — every
name is a mechanism the model actually knows:

| Reason | Meaning |
|---|---|
| `semaphore_empty` | SEMWAIT: selected semaphore still zero — the producer has not posted. |
| `semaphore_full` | SEMWAIT: semaphore at max — consumer back-pressure. |
| `resource_wait` | STALLWAIT condition unmet; `blocked_on` names the unit it is about. |
| `mutex_wait` | `ATGETM` accepted, mutex not yet granted. |
| `backend_enforced_stall` | A backend unit asserted the gate's stall directly. |
| `src_reserved_by_unpacker` | Matrix-unit op waiting for a Src bank the unpackers still own. |
| `src_reserved_by_matrix` | Unpacker / ThCon waiting for a Src bank the matrix unit still owns. |
| `unit_busy` | Target unit (or its IPC group) still occupied. **Mostly cost-model state** — see below. |
| `issue_slot_taken` | Target unit's issue slot already taken this cycle. |
| `issue_yield_fairness` | Slot yielded to a less-recently-granted thread. |
| `flush_pending` | Scalar unit mid-`FLUSHDMA`, waiting for the DMA units to drain. |
| `atomic_pending` | Scalar unit retrying an `ATCAS`. |

**Regime.** Every reason is structural and occurs in both regimes.
`unit_busy` is *nearly* the exception — an occupancy is what the cost
model arms, so with `TT_SIM_COST_MODEL` unset almost nothing can report
it. Not quite nothing: the config unit's own single-issue throughput
rule (a `SETC16`/`WRCFG` in the previous cycle blocks this cycle's
other-opcode issue, `backends/config.py`) is a structural refusal that
also carries this reason. Measured on the Blackhole `sfpumath` guard,
an un-modelled run reports `tensix_stall_unit_busy` = 6 cycles, all
blamed on `CFG`; the modelled run reports 84 across `CFG` and the rest.
Do not assert that this counter is absent without the cost model.

**These are modelled floors, not calibrated cycle counts.** The
simulator charges every published bound at its low end
(`docs/plans/cost-model.md`); the result is corroborated against
silicon but never calibrated to it. Read the *attribution* — which
reason dominates and which unit is named — rather than the absolute
figure. The Perfetto writer repeats this on every stall slice's
`timing_model` argument, because a stall width is the number most
likely to be quoted out of context.

Counters derived by `CounterAggregator`: `tensix_stall_cycles`,
`tensix_stall_episodes`, `tensix_stall_<reason>` and
`tensix_stall_on_<unit>`. The `tensix_` prefix is load-bearing — a
Tensix thread publishes under the *same* `unit_id` its baby RISC-V core
uses for `InstrEvent`, so an unprefixed `stall_cycles` would sum two
unrelated mechanisms.

Not surfaced, with reasons: **unpacker idle cycles** are derivable from
the existing `busy_cycles` counter against the run's cycle span, and
counting them directly would need the pump to visit units on cycles it
deliberately skips. **L1 bank conflicts** (L1 is flat memory) and NoC
`vc` occupancy (no VCs modelled) remain genuinely un-modelled.
<!-- END stall-reasons -->

## ID scheme

`unit_id` is a 4-tuple: `(chip_id, core_y, core_x, unit)`.

- `chip_id` — `0` today (single-chip). Multi-chip is gated on
  ROADMAP §A multi-Tensix expansion and §F ERISC.
- `core_y`, `core_x` — unified tile coordinates (16–25 range), matching
  the WHB0 SoC descriptor. `(18, 18)` is the single Tensix tile that
  `Wormhole.__init__` instantiates today.
- `unit` — string from the `Unit` enum:
  `BRISC, NCRISC, TRISC0, TRISC1, TRISC2, FPU, SFPU, PACKER, UNPACKER,
  MATRIX, MOVER, THCON, SYNC, TDMA, CFG, MAILBOX, TTSYNC, NOC0, NOC1,
  HOST, UNKNOWN`. (`UNKNOWN` is the fallback for `MemEvent`s whose
  caller context wasn't set.)

The sidecar `*.ids.json` enumerates every unit the run touched, in the
order they were registered. Use it as the joining table between traces
collected across runs.

## Adding a publish call

The pattern (gate, construct, publish — and **only when the gate
allows**, so construction cost stays off the hot path):

```python
from tt_sim.trace import EventCategory, InstrEvent, get_bus

bus = get_bus()
if self.unit_id is not None and bus.is_enabled(EventCategory.INSTR):
    bus.publish(
        InstrEvent(
            cycle=cycle_num,
            unit_id=self.unit_id,
            pc=pc_val,
            instruction=conv_to_uint32(instr_raw),
            stalled=pe_stall,
        )
    )
```

`is_enabled(category)` is a single attribute lookup; expensive payload
construction (disassembly, hex formatting, deep tuple copies) must live
inside the `if`. The `unit_id is not None` guard handles the brief
window during device construction before
`TensixTile._register_trace_ids` runs.

Components publishing events need a `unit_id: tuple | None` attribute
set externally during device construction. See
`tt_sim/device/tt_device.py:TensixTile._register_trace_ids` for the
assignment pattern.

## Adding a new event type

1. Add to `EventCategory` in `tt_sim/trace/events.py` if introducing a
   new category.
2. Add to the `Unit` enum if the source isn't covered.
3. Define the dataclass with `CATEGORY` and `SCHEMA_VERSION` `ClassVar`s.
4. Export from `tt_sim/trace/__init__.py`.
5. Bump `SCHEMA_VERSION` on the relevant event (or globally in
   `events.py`) if existing field semantics change. Writers should
   refuse unknown versions rather than silently mis-render.

Frozen dataclasses are deliberate — events should be immutable once
published. If a consumer wants to mutate, copy first.

## Adding a new writer

Implement a class that subscribes to the categories it cares about and
holds onto whatever output handle it needs. Pattern from
`writers/jsonl.py`:

```python
from tt_sim.trace import EventCategory, get_bus

class MyWriter:
    def __init__(self):
        bus = get_bus()
        for cat in (EventCategory.INSTR, EventCategory.DISPATCH):
            bus.subscribe(cat, self._on_event)

    def _on_event(self, event):
        ...  # serialise / aggregate / forward
```

Constraints from the design doc (ROADMAP §H "Design principles"):

- **Stream, don't accumulate.** Long sim runs produce millions of
  events; never hold the whole trace in memory.
- **Refuse unknown schema versions** rather than silently mis-format.
- **Reuse `unit_id`** verbatim — that's the join key for any
  cross-tool correlation (Perfetto pid/tid, Cachegrind function names,
  LCOV file paths).

Phase 2 onwards plans writers for Perfetto / Chrome trace
(`ui.perfetto.dev`), Spike `--log-commits` (Spike-compatible diff
tooling), Parquet (DuckDB / pandas), Cachegrind (KCachegrind), and
LCOV (genhtml / Codecov / VS Code).

## What's not modelled yet

All seven phases are complete. The major items deliberately deferred
out of scope across phases:

- **Register-file accesses** as `MemEvent` — multiple per instruction
  with low marginal signal vs. trace volume cost.
- **Multi-Tensix / multi-chip identity.** `chip_id` is hard-coded to
  `0`; `core_y/core_x` are the unified tile coords for the single
  Tensix at `(18, 18)`. Falls out of ROADMAP §A multi-Tensix.
- **Cycle-accurate fields.** Landed: Perfetto durations, NoC
  `issue_cycle` / `arrival_cycle` / `flight_cycles`, per-core RV
  stall cycles with reason, per-unit `busy_cycles`. Still gated on
  ROADMAP §I because the simulator holds no such state: NoC `vc` and
  VC occupancy (no virtual channels are modelled), packer
  back-pressure and unpacker idle cycles (neither unit is wired to the
  cost tables), and L1 bank conflicts (no banks are modelled).
- **Inline-print migration.** Events are additive today — existing
  `if self.snoop: print(...)` sites still run alongside bus publish.
- **VS Code coverage extension** and **end-to-end `libttsim.so`
  differential testing** — both ROADMAP-flagged as out of scope for
  the initial Phase 6/7 cuts.
- **Property tests with `hypothesis`** — would need a kernel
  generator to drive useful workloads, which is a separate project.
