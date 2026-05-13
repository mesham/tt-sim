# Structured tracing (`tt_sim/trace/`)

A typed pub/sub event bus. The simulator publishes architectural events
(instruction retirement, Tensix dispatch, NoC traffic, kernel lifecycle);
external consumers subscribe and turn the stream into whatever they
need. Today: JSONL for ad-hoc analysis and Perfetto / Chrome Trace
Event Format for visual timelines on `ui.perfetto.dev`. Later phases
add Spike commitlog, Parquet counters, Cachegrind, and LCOV writers
as additional consumers of the same bus. tt-sim does not build viewers.

Phases 1–5 of [ROADMAP §H](../../ROADMAP.md) are complete (event bus,
Perfetto, Spike-compatible commitlog, Parquet/DuckDB counters, NoC
Parquet + Cachegrind memory trace); read that for the full plan.

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
# All six can be enabled at once — writers subscribe independently:
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

**Cycle-as-time mapping:** events are stamped with `ts = cycle` and
`dur = 1`. The simulator isn't cycle-accurate today, so durations are
synthetic placeholders — useful for spatial reasoning ("what happened
on the SFPU around cycle 3000?") but not for absolute timing. Once §I
cycle accuracy lands, durations become meaningful and the writer
needs no schema change. Events with `cycle == 0` (mem / sync /
lifecycle) are stamped with the highest cycle seen so far, so they
appear at the "current time" rather than collapsing onto t=0.

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

As §I cycle accuracy lands, more counters (FPU stall reasons, packer
back-pressure, L1 bank conflicts) drop into the same long-format
schema with no consumer changes needed.

### NoC transactions (Parquet)

`TT_SIM_TRACE_NOC=<dir>` writes a Hive-partitioned Parquet dataset
(`chip=N/*.parquet`) — one row per `NoCEvent` emission with columns
`cycle, chip, core_y, core_x, unit, phase, txn_type, src_x, src_y,
dst_x, dst_y, size_bytes, txn_id`. Suitable for SQL queries about data
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

VC occupancy, issue/arrival cycle, and other cycle-accurate fields
are gated on §I and will land as additional columns once the
simulator carries that state.

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
group under a synthetic `<region>_no_pc` function. When §H Phase 6
(DWARF / LCOV source-level attribution) lands, the `<region>_pc_…`
function names are replaced with real source/function coordinates
with no change to the writer's contract.

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

Per-type fields published today:

### `InstrEvent` (category `instr`)

Emitted on each RV instruction retirement on the five baby cores
(BRISC, NCRISC, TRISC0–2).

| Field         | Type   |                                                   |
|---------------|--------|---------------------------------------------------|
| `pc`          | `int`  | PC at the retiring instruction.                   |
| `instruction` | `int`  | Raw 32-bit instruction word.                      |
| `stalled`     | `bool` | True if the core stalled this cycle (no retire).  |

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

Pair an `(issue, txn_id)` with its `(response, txn_id)` to recover
round-trip latency once cycle-accuracy lands (ROADMAP §I).

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

Both `unit_id` and `target_unit` carry the same architectural unit
information — `unit_id` is the canonical join key, `target_unit` is the
human-readable form. Consumers can use either.

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

Phase 1 is complete (ROADMAP §H Phase 1). Items deliberately deferred
out of scope:

- **Register-file accesses** as `MemEvent` — multiple per instruction
  with low marginal signal vs. trace volume cost. The `InstrEvent`'s
  raw `instruction` field already lets consumers decode register
  operands.
- **Multi-Tensix / multi-chip identity.** `chip_id` is hard-coded to
  `0`; `core_y/core_x` are the unified tile coords for the single
  Tensix at `(18, 18)`. Falls out of ROADMAP §A multi-Tensix.
- **Inline-print migration.** Events are additive today — existing
  `if self.snoop: print(...)` sites still run alongside bus publish.
  Routing `DeviceTileDiagnostics` flags through bus subscribers is a
  separate refactor, queued until Phase 2's Perfetto writer reveals
  what data consumers actually need.
