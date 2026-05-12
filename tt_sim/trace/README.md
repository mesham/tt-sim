# Structured tracing (`tt_sim/trace/`)

A typed pub/sub event bus. The simulator publishes architectural events
(instruction retirement, Tensix dispatch, NoC traffic, kernel lifecycle);
external consumers subscribe and turn the stream into whatever they
need — JSONL today, Perfetto / Spike commitlog / Parquet / LCOV in later
phases. tt-sim does not build viewers.

This is Phase 1 of the broader tracing plan in [ROADMAP §H](../../ROADMAP.md);
read that for the full picture (Perfetto, DuckDB, source-level
attribution, differential testing).

## Quick start

```bash
export PYTHONPATH=~/tt-sim:$PYTHONPATH
cd driver/wormhole
TT_SIM_TRACE=/tmp/one.trace.jsonl python3 one/one.py
```

Produces two files:

- `/tmp/one.trace.jsonl` — one JSON object per line, every event the
  simulator published during the run.
- `/tmp/one.trace.jsonl.ids.json` — sidecar mapping every `unit_id`
  tuple in the trace to its `(chip_id, core_y, core_x, unit)`
  decomposition. Same scheme will be reused by all later writers.

Inspect with `jq` / `duckdb` / `pandas`. Example: count events per
category:

```bash
python3 -c "
import json
from collections import Counter
print(Counter(json.loads(l)['category'] for l in open('/tmp/one.trace.jsonl')))
"
```

A `four/four.py` run reports roughly
`{mem: 65k, instr: 13.5k, dispatch: 864, compute: 309, noc: 12,
lifecycle: 4, sync: 4}` — all seven categories populated.

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
