# The tt-sim trace schema

**Status: stable as of `SCHEMA_VERSION` 4.** From this document onward,
a change to any field name or meaning described here is a breaking
change and comes with a version bump.

This is the contract between tt-sim and the tools you build on top of
it. It is written for someone who will never read tt-sim's source: it
says what each field means, what unit it is in, and whether you can rely
on it. If you find yourself opening `tt_sim/trace/` to answer a question
about an output format, that is a bug in this document — please report
it.

- **Producing the data**: [`driver/wormhole/docs/profiling.md`](../driver/wormhole/docs/profiling.md)
  is the walkthrough — which environment variable turns on which output.
- **Extending the simulator**: [`tt_sim/trace/README.md`](../tt_sim/trace/README.md)
  is the implementer's guide (the event bus, adding a writer). You do not
  need it to consume any of the formats below.
- **What a charged cycle means**: `docs/plans/cost-model.md`.

---

## 0. Read this first: what these numbers are

Every cycle count in every output below is **modelled**. It is charged
from a bound published in Tenstorrent's ISA documentation, always at the
**low end** of the published range. The results are **floors**. They
have been corroborated against silicon measurements; they have never
been *calibrated* against silicon, and no total has ever been fitted to
an end-to-end hardware run.

The supported use is **relative attribution**: which unit dominates,
which stall reason dominates, and how those change when you change the
kernel. That is the question the model is built to answer and the one it
answers well.

Three things follow, and tooling that ignores them will mislead its
users:

1. **Do not present an absolute cycle count as a prediction of
   hardware.** Do not diff a tt-sim total against a silicon measurement
   and report the difference as model error; the model has never claimed
   that difference was zero.
2. **Do not build a pie chart out of cycle counters** without reading
   §4.3 first. Units run concurrently and several counter families
   legitimately sum past 100 % of the run. On the workload measured for
   this document, one core's named cycles came to 113 % of the run's
   span. That is correct, not a bug — and a pie chart of it is a lie.
3. **Cycle *values* are not frozen** and will move as the cost model
   grows (see §2). Compare two runs of the same tree, never a run
   against a number written down last month.

The simulator is not cycle-accurate. A term nothing charges contributes
**nothing** rather than a guess, so an unattributed remainder is real
missing coverage — the ranked report surfaces it as its own column
rather than folding it away.

---

## 1. The outputs

Every output is off by default and enabled by one environment variable
set before your tt-metal binary runs. Enabling several is fine — they
subscribe independently.

| Output | Variable | Form | Best for |
|---|---|---|---|
| **Ranked report** | `TT_SIM_PROFILE=<dir>` | Markdown + JSON | "Where did my kernel's cycles go?" — start here |
| **Counters** | `TT_SIM_TRACE_COUNTERS=<dir>` | Parquet (Hive-partitioned) | SQL / pandas; the main machine-readable dataset |
| **NoC transactions** | `TT_SIM_TRACE_NOC=<dir>` | Parquet (Hive-partitioned) | Data-movement analysis, per packet |
| **Timeline** | `TT_SIM_TRACE_PERFETTO=<path>` | Chrome Trace JSON | Visual inspection at `ui.perfetto.dev` |
| **Raw events** | `TT_SIM_TRACE=<path>` | JSONL | Anything the above cannot express |
| Commitlog | `TT_SIM_TRACE_COMMITLOG=<dir>` | Spike `--log-commits` | RISC-V differential testing |
| Memory hotspots | `TT_SIM_TRACE_MEMORY=<path>` | Callgrind | KCachegrind |
| Source coverage | `TT_SIM_TRACE_LCOV=<path>` | LCOV | `genhtml`, Codecov |

`TT_SIM_PROFILE` is a preset over the counters dataset plus per-PC
attribution; it is the recommended entry point and produces
`report.md`, `report.json`, `hotspots.json`, `profile.json` and a
`counters/` dataset.

**Set `TT_SIM_COST_MODEL=1` as well, or nothing can stall.** See §3.4.

---

## 2. What is frozen, and what is not

### Frozen — you may assert on these

- **Field and column names**, exactly as spelled in this document.
- **Field meanings and units**, as described here.
- **Types**, as given in each table.
- **`unit_id` structure**: a 4-tuple `(chip_id, core_y, core_x, unit)`.
- **The `EventCategory` values** (`instr`, `mem`, `noc`, `compute`,
  `sync`, `dispatch`, `lifecycle`, `counter`, `stall`).
- **The `STALL_REASONS` vocabulary** (§6.2). Exported as
  `tt_sim.trace.STALL_REASONS` — switch on it exhaustively.
- **Counter-name *patterns*** (§4.2). Not the enumeration — see below.

Renaming or removing any of these bumps `SCHEMA_VERSION` and is
announced as breaking.

### Not frozen — do not assert on these

- **Any cycle value.** Every cost-model instalment moves these by
  design. Assert on relative ordering between two runs of the same tree,
  never on an absolute.
- **The set of counter names.** New counters — especially new stall
  reasons — appear without a code change. See §4.2.
- **Row ordering** in any dataset. Sort explicitly. (In particular
  `StallEvent`s are *not* ordered by `cycle`; see §7.4.)
- **File and partition layout within a dataset directory**, beyond the
  documented Hive partition keys. Read with a `**/*.parquet` glob.
- **The Markdown layout of `report.md`.** It is for humans. Parse
  `report.json` instead.

### Versioning

`SCHEMA_VERSION` is a single integer carried on every event and written
into every JSONL row as `schema_version`. It is currently **4**.

| Version | Change |
|---|---|
| 4 | Added the `stall` category and `StallEvent`. **Additive** — a consumer written against 1–3 sees its own categories unchanged and simply does not subscribe to the new one. |

The rule going forward: **additive changes** (a new event kind, a new
category, a new field with a default, a new counter name) bump the
version but do not break a consumer that ignores what it does not know.
**Breaking changes** (a rename, a removal, a change of meaning or unit)
bump the version and are called out as breaking. Writers refuse to emit
a version they do not recognise rather than silently mis-rendering.

Pin the version you were built against and warn on a mismatch; do not
hard-fail on a bump you have not read yet, because most will be
additive.

---

## 3. Concepts that apply to every output

### 3.1 `unit_id` — the join key

Every event and every row is attributed to one architectural unit:

```
(chip_id, core_y, core_x, unit)
```

| Part | Type | Meaning | Stability |
|---|---|---|---|
| `chip_id` | `int` | Always `0` today (single-chip). | Frozen; values will grow with multi-chip. |
| `core_y`, `core_x` | `int` | Tile coordinate. **Note y precedes x.** | Frozen shape; see §3.2 for values. |
| `unit` | `str` | One of the `Unit` vocabulary below. | Frozen. |

In the Parquet datasets these appear as four separate columns: `chip`,
`core_y`, `core_x`, `unit`.

`unit` values: `BRISC`, `NCRISC`, `TRISC0`, `TRISC1`, `TRISC2`, `FPU`,
`SFPU`, `PACKER`, `UNPACKER`, `MATRIX`, `MOVER`, `THCON`, `SYNC`,
`TDMA`, `CFG`, `MISC`, `MAILBOX`, `TTSYNC`, `NOC0`, `NOC1`, `HOST`,
`UNKNOWN`.

**`UNKNOWN` is a real value, not an error.** It is the attribution for a
memory access with no originating RISC-V instruction — NoC-driven
traffic, engine-internal accesses, and every access on a DRAM tile. On
the workload measured here, 3,388 of 124,590 `MemEvent`s (2.7 %) carried
it. Do not drop those rows as corrupt; they are real traffic that the
model cannot attribute to a PC.

`HOST` (`(0, 0, 0, HOST)`) is the pseudo-unit for lifecycle events,
which come from the host driver rather than from the device.

### 3.2 Coordinates are architecture-specific

`core_y` / `core_x` are **tile coordinates in the architecture's own
system**, and the two architectures do not agree:

| Architecture | System | Example |
|---|---|---|
| Wormhole | Unified coordinates, 16–25 | Tensix at `(18, 18)` |
| Blackhole | Physical NoC coordinates on the 17×12 grid | Tensix at `(2, 1)`, DRAM at `(11, 0)` |

**Do not join a Wormhole dataset to a Blackhole dataset on `unit_id`.**
The same logical core has different coordinates, and a coordinate valid
on one is a different tile — or no tile — on the other. If you correlate
across architectures, join on `unit` alone, or carry the architecture as
an explicit column of your own.

Neither is the *logical* Tensix coordinate that tt-metal's host API
uses. There is no logical-coordinate column in any dataset today.

### 3.3 The same unit has three different names

This is the single most common way to get an empty result. A Tensix
backend unit is spelled three ways, and they are **not** interchangeable
— they are not even consistently case-variants of each other:

| `unit_id[3]` (canonical) | `ComputeEvent.target_unit` | `DispatchEvent.target_unit`, `StallEvent.blocked_on` |
|---|---|---|
| `MATRIX` | `Matrix` | `MATH` |
| `SFPU` | `Vector` | `SFPU` |
| `PACKER` | `Packer` | `PACK` |
| `UNPACKER` | `Unpacker` | `UNPACK` |
| `THCON` | `Scalar` | `THCON` |
| `CFG` | `Config` | `CFG` |
| `SYNC` | `Sync` | `SYNC` |
| `TDMA` | `Misc` | `TDMA` |
| `MOVER` | `Mover` | `XMOV` |

The middle column is the backend class's own name; the right-hand column
is the ISA's `ex_resource`, which the instruction tables are keyed by.
`DispatchEvent.target_unit` additionally takes the value `NONE`, for an
instruction the wait gate accounts for but dispatches to no backend.

**Join on `unit_id`.** The table is exported as
`tt_sim.trace.BACKEND_UNIT_ALIASES` — `{unit: (unit_name, ex_resource)}`
— so you do not have to hand-copy it, and a test fails if the code
drifts from it.

The counter names `dispatch_to_<X>` and `tensix_stall_on_<X>` use the
**right-hand** column.

### 3.4 The two timing regimes

`TT_SIM_COST_MODEL` decides whether the simulator charges modelled
latencies at all. It is a construction-time decision, so a whole run is
in one regime.

| | `TT_SIM_COST_MODEL` unset | `TT_SIM_COST_MODEL=1` |
|---|---|---|
| RV stalls (`stall_*`) | never occur; counters **absent** | modelled |
| Backend occupancy (`busy_cycles`) | nothing is occupied; counters **absent** | modelled |
| Tensix thread stalls (`tensix_stall_*`) | **present** — these are structural | present, and larger |
| NoC flight (`noc_flight_cycles`) | **present** — 1–2 cycles, a real measurement of an un-modelled NoC | modelled per-hop latency |
| Event counts, byte counts | identical mechanism, present in both | present in both |

Measured on the Blackhole `sfpumath` guard, same workload both ways:

| counter | model off | model on |
|---|---|---|
| `stall_cycles` | *absent* | 41,163 |
| `busy_cycles` | *absent* | 5,754 |
| `noc_flight_cycles` | 12 | 1,880 |
| `tensix_stall_cycles` | 9,659 | 10,793 |

Always record which regime produced a dataset. Two outputs state it for
you: the NoC Parquet dataset has a `cost_model` boolean column on every
row, and the Perfetto trace states it three ways (§5). **The counter
Parquet dataset does not carry the regime** — this is a known gap; if
you archive counter datasets, record `TT_SIM_COST_MODEL` alongside them
yourself, or capture `profile.json`, which has a `cost_model` field.

### 3.5 Absent is not zero

**This is a contract, not an implementation detail.** A counter row
exists only once something incremented it. A dataset from a run where
nothing stalled has *no* `stall_cycles` rows — it does not assert a
stall-free machine, it makes no claim at all.

The distinction survives into SQL, where it is easy to destroy:

```sql
SELECT SUM(value) FILTER (WHERE counter_name = 'stall_load_use') AS load_use
FROM read_parquet('counters/**/*.parquet', hive_partitioning=true);
-- returns NULL when the counter is absent, not 0
```

`COALESCE(..., 0)` on that turns "the model makes no claim" into "the
model measured zero stalls". They are completely different statements
and only one of them is true. Keep the NULL, and render it as `—` or
"not modelled".

The same rule applies to individual fields: `ComputeEvent.duration == 0`
means *the cost tables have no opinion about this opcode* — because the
model is off, or the unit is unwired, or the opcode is uncosted. It does
not mean one cycle.

---

## 4. Performance counters (Parquet) — the main dataset

`TT_SIM_TRACE_COUNTERS=<dir>` (or `TT_SIM_PROFILE`, which owns
`<dir>/counters`).

Hive-partitioned: `chip=<N>/kernel_id=<N>/*.parquet`. Long format, one
row per `(cycle, unit, counter_name)` sample.

```sql
SELECT * FROM read_parquet('counters/**/*.parquet', hive_partitioning=true);
```

### 4.1 Columns

| Column | Type | Unit | Meaning | Stability |
|---|---|---|---|---|
| `cycle` | `int64` | simulated cycles | The **flush boundary** this sample was written at — not the time of any individual event. See §7.5. | Frozen name; values move with the model. |
| `chip` | `int32` | — | `chip_id`. Hive partition key. | Frozen. |
| `kernel_id` | `int32` | — | Increments at each `kernel_start`. `0` before the first kernel (firmware setup). Hive partition key. | Frozen. |
| `core_y`, `core_x` | `int32` | — | Tile coordinate; arch-specific (§3.2). | Frozen. |
| `unit` | `string` | — | `Unit` enum value (§3.1). | Frozen. |
| `counter_name` | `string` | — | See §4.2. **Open set.** | Patterns frozen; the set is not. |
| `value` | `int64` | depends on the counter — see §4.2 | The increment **since the previous flush**, not a running total. | Frozen. |

`value` is a **delta**. To get a run total, `SUM(value)` over the rows;
do not take `MAX(value)`.

### 4.2 Counter families

**The set of counter names is open by design.** New stall reasons and
new occupancy counters appear in this dataset with no code change
anywhere, because names are constructed from the mechanism that fired.
Match on the **pattern**, never on an enumeration — a query naming
reasons one by one silently drops the newest one, which is usually the
one you wanted.

| Pattern | Unit of `value` | Emitted per | Meaning |
|---|---|---|---|
| `instr_retired` | cycles | baby RV core | RV instructions retired; one cycle floor each. |
| `instr_stalled` | count | baby RV core | Retirements flagged with Tensix instruction-buffer back-pressure. Both regimes. |
| `stall_<reason>` | cycles | baby RV core | Cycles the RV cost model held an instruction, by reason. **Cost-model only.** |
| `stall_cycles` | cycles | baby RV core | **Redundant** — the sum of `stall_<reason>`. See §4.4. |
| `busy_cycles` | cycles | Tensix backend unit | Modelled occupancy charged by the cost tables. **Cost-model only.** |
| `compute_ops` | count | Tensix backend unit | Instructions completed by that unit. |
| `dispatch_total` | count | Tensix thread | Instructions issued to any backend. |
| `dispatch_to_<ex_resource>` | count | Tensix thread | …split by target (§3.3 right column, plus `NONE`). |
| `tensix_stall_<reason>` | cycles | Tensix thread | Cycles the thread made no progress, by mechanism. Both regimes. |
| `tensix_stall_cycles` | cycles | Tensix thread | **Redundant** — the sum of `tensix_stall_<reason>`. |
| `tensix_stall_on_<unit>` | cycles | Tensix thread | **Redundant re-cut** — the same cycles by the unit to blame. *Partial.* |
| `tensix_stall_episodes` | count | Tensix thread | Number of stall episodes. A count, despite the prefix. |
| `noc_flight_cycles` | cycles | NIU | Issue→arrival summed over timed transactions. Both regimes. |
| `noc_txns_timed` | count | NIU | Transactions contributing to the above. |
| `noc_bytes_total` | bytes | NIU | Bytes in `response`-phase transactions. |
| `noc_<phase>_<txn_type>` | count | NIU | e.g. `noc_request_read`, `noc_response_write`. |
| `mem_<op>_<region>` | count | accessing unit, or `UNKNOWN` | e.g. `mem_read_L1`, `mem_write_MMIO`. |
| `mem_bytes_<op>` | bytes | accessing unit, or `UNKNOWN` | Bytes read / written. |
| `sync_<kind>` | count | `TTSYNC` / `MAILBOX` | Synchronisation events by kind. |

The `tensix_` prefix is load-bearing. A Tensix thread publishes under
the **same** `unit_id` as the baby RISC-V core that feeds it, so an
unprefixed `stall_cycles` would sum two unrelated mechanisms into one
unreadable number.

Two rules for classifying a name you have never seen, and the tt-sim
report applies exactly these (`tt_sim.trace.report.is_cycle_bearing`,
`is_redundant` — importable, so you need not re-derive them):

- Cycle-bearing if it ends in `_cycles`, starts with `stall_` or
  `tensix_stall_`, or is `instr_retired` — **except** the count-shaped
  `tensix_stall_episodes`.
- Everything else is a **volume** (a count or a byte total) and must
  never be added to a cycle figure.

### 4.3 Is it safe to aggregate across units?

The first thing most people build from this dataset is a chart summing a
counter over units. For half the families that is meaningful and for the
other half it is nonsense. This column is the difference:

| Family | Sum across units? | Why |
|---|---|---|
| `noc_flight_cycles` | **Yes** | Shared-resource occupancy. Two NIUs servicing packets at once really are two occupied links. |
| `busy_cycles` | **Yes** | Shared-resource occupancy. Two backends busy at once really are two occupied pipes. |
| `noc_bytes_total`, `mem_bytes_*` | **Yes** | Byte volumes are additive. |
| all `count` families | **Yes** | Event counts are additive. |
| `instr_retired` | **No, not as time** | Per core. Five cores retiring concurrently sum to five times the wall clock. Safe as a *volume* ("instructions executed"), never as a share of the run. |
| `stall_<reason>` | **No** | Per core. A core can be charged far more stall than the run is long. |
| `tensix_stall_<reason>` | **No** | Per thread; three threads stall concurrently. |

Measured on the workload used throughout this document (span 21,299
cycles): TRISC2's named cycles came to 24,070 — **113 % of the run**.
TRISC0's came to 103 %. Both are correct. Charged is not delivered: the
run's length is set by when cores meet each other, not by a sum of
charges.

So:

- For "what fraction of the machine was busy", sum **only**
  `noc_flight_cycles` and `busy_cycles`, and say that is what you
  summed. `report.json`'s `shared_resource_cycles` is exactly this
  roll-up.
- For per-core work, present each core's share of the span **as its own
  bar**, not as a slice of a pie.
- Never mix the two in one total.

### 4.4 Do not sum a total with its parts

Three counters restate cycles that another row already carries. Ranking
or summing them alongside the partition they restate double-counts:

| Counter | Restates | Verified |
|---|---|---|
| `stall_cycles` | `Σ stall_<reason>` | 41,163 = 41,050 + 99 + 10 + 4 — exact |
| `tensix_stall_cycles` | `Σ tensix_stall_<reason>` | 10,793 = 6,072 + 3,913 + 703 + 84 + 16 + 3 + 2 — exact |
| `tensix_stall_on_<unit>` | the same cycles, re-cut by blame | 4,721 ≤ 10,793 — **partial** |

`tensix_stall_on_<unit>` is a genuinely useful second view — it answers
"which unit held this thread up", which is the actionable half — but it
is *partial*: a semaphore or mutex wait blames no unit, so these rows
sum to **less** than `tensix_stall_cycles` rather than equalling it. Do
not treat the shortfall as missing data, and do not add it to the
per-reason rows.

Canned queries that get all of this right live in
[`tt_sim/trace/queries/counters.sql`](../tt_sim/trace/queries/counters.sql).
They are the closest thing to a worked reference consumer.

---

## 5. NoC transactions (Parquet)

`TT_SIM_TRACE_NOC=<dir>`. Hive-partitioned by `chip` only — NoC traffic
is not naturally kernel-bound, since firmware setup generates it too.
One row per NoC event emission.

| Column | Type | Unit | Meaning | Stability |
|---|---|---|---|---|
| `cycle` | `int64` | cycles | Arrival: the cycle **this** NIU serviced the packet. Same as `arrival_cycle`. | Frozen. |
| `chip`, `core_y`, `core_x` | `int` | — | The **observing** NIU's tile. Partition key on `chip`. | Frozen. |
| `unit` | `string` | — | `NOC0` or `NOC1`. | Frozen. |
| `phase` | `string` | — | `request` or `response`. | Frozen. |
| `txn_type` | `string` | — | `read` or `write`. | Frozen. |
| `src_x`, `src_y` | `int32` | — | Source NoC coordinate. **Note x precedes y here** — the opposite order to `core_y`/`core_x`. | Frozen. |
| `dst_x`, `dst_y` | `int32` | — | Destination NoC coordinate, same ordering. | Frozen. |
| `size_bytes` | `int64` | bytes | Transfer size. | Frozen. |
| `txn_id` | `int64` | — | NoC transaction ID. **Reused** across a run — not a unique key. | Frozen. |
| `issue_cycle` | `int64` | cycles | When the *sending* NIU put the packet on the wire. `-1` when untimed. | Frozen. |
| `arrival_cycle` | `int64` | cycles | `== cycle`; duplicated for readability at the query site. | Frozen. |
| `flight_cycles` | `int64` | cycles | `arrival_cycle - issue_cycle`, floored at 0. `0` when `issue_cycle` is `-1`. | Frozen name; values move with the model. |
| `cost_model` | `bool` | — | Which regime produced this row (§3.4). | Frozen. |

**Coordinate ordering differs between the two coordinate pairs in this
table.** `core_y, core_x` (the observing NIU) is y-then-x, matching
`unit_id`; `src_x, src_y` / `dst_x, dst_y` are x-then-y. This is a wart,
frozen because renaming the columns would break existing queries. Read
the column names, not the position.

**Pairing a request with its response**: match on
`(txn_id, src, dst, txn_type)`, flipping src and dst — a response's
source and destination are swapped relative to its request. `txn_id`
alone is not unique.

`issue_cycle == -1` means the flight could not be timed at all: a NIU
with no owning tile clock, which happens only in unit tests and the
`driver/simple` examples. It is distinct from a flight of 0.

`flight_cycles == 1` means different things in the two regimes, which is
what `cost_model` is for. With the model off, a packet is delivered on
the next cycle however far it travelled — a real observation about an
un-modelled NoC, not an estimate of a hop count.

**No virtual-channel column exists.** tt-sim models no VCs, so there is
no `vc` column rather than a column of zeroes. Same for VC occupancy.

---

## 6. The event stream

The Parquet datasets are derived from a typed event stream. You can read
it directly with `TT_SIM_TRACE=<path>` (JSONL, one object per line),
which also writes `<path>.ids.json` — a sidecar listing every `unit_id`
the run touched as `{chip_id, core_y, core_x, unit}` objects.

Every JSONL row carries `category` and `schema_version` in addition to
the event's own fields. Volume is high: the workload used here produced
202,701 events, 61 % of them `mem`.

### 6.1 Per-event fields

All events carry `cycle` (`int`, simulated cycles — but read §7.5) and
`unit_id` (4-element array).

#### `instr` — `InstrEvent`

One per RV instruction retirement on the five baby cores.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `pc` | `int` | address | PC of the retiring instruction. **Runtime address** — see §7.2. |
| `instruction` | `int` | — | Raw 32-bit instruction word. Not disassembled; decoding is yours. |
| `stalled` | `bool` | — | Tensix instruction-buffer back-pressure. Exists in **both** regimes. |
| `reg_write_idx` | `int` | — | Destination register, or `-1` for no architectural write. `x0` writes are not recorded. |
| `reg_write_value` | `int` | — | Value written; meaningless when `reg_write_idx == -1`. |
| `stall_cycles` | `int` | cycles | Cycles the core was held before this instruction could issue. **Cost-model only**; `0` otherwise. |
| `stall_reason` | `str` | — | Which mechanism held it, or `""`. Cost-model only. |

`stalled` and `stall_cycles` are **different mechanisms** with
confusingly similar names: the first is Tensix front-end back-pressure
and is structural; the second is the RV cost model's scoreboard.

#### `stall` — `StallEvent`

Why a Tensix thread made no progress. The complement of `dispatch`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `reason` | `str` | — | One of `STALL_REASONS` (§6.2). |
| `blocked_on` | `str` | — | `ex_resource` name of the unit at fault, or `""`. |
| `cycles` | `int` | cycles | Length of the episode. Always ≥ 1. |
| `opcode` | `str` | — | Instruction held at the gate, or `""` for a latched wait that blocks a *class* of instructions. |
| `thread_id` | `int` | — | 0–2. |
| `semaphore` | `int` | — | Semaphore index for a semaphore wait; `-1` otherwise. |

Two properties matter more than the fields:

- **`unit_id` is the thread that *suffered*; `blocked_on` is the unit
  *responsible*.** They are deliberately different. An unpacker that
  cannot start because the matrix unit still owns the Src bank is
  reported with `unit_id` `TRISC0` and `blocked_on` `MATH`. That split
  is the actionable part for a code generator.
- **Episodes are coalesced, not per-cycle.** The wait gate re-offers its
  head instruction every cycle while blocked; one event per cycle would
  be 64k rows for one wait. Instead an episode accumulates while
  `(reason, blocked_on, opcode)` is unchanged and publishes once, when
  it ends. `cycle` is the **first** cycle of the span and `cycles` its
  length. On the workload here, 32 events covered 10,793 stalled cycles.

A thread is in at most one episode at a time, so episodes on one thread
never overlap. Episodes on *different* threads do.

#### `dispatch` — `DispatchEvent`

| Field | Type | Meaning |
|---|---|---|
| `opcode` | `str` | Tensix instruction name. |
| `target_unit` | `str` | `ex_resource` (§3.3), including `NONE`. |
| `thread_id` | `int` | 0–2; redundant with `unit_id`. |

The MOP expander, replay expander, and direct `issueInstruction` paths
not gated by the wait gate do **not** publish. Dispatch counts are a
lower bound on issued instructions.

#### `compute` — `ComputeEvent`

Emitted as a backend unit completes an instruction.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `op` | `str` | — | Opcode name. |
| `target_unit` | `str` | — | Backend `unit_name` (§3.3 middle column). |
| `thread_id` | `int` | — | Issuing thread, or `-1`. |
| `detail` | `str` | — | Free-form; empty today. |
| `duration` | `int` | cycles | Modelled occupancy. **`0` means "no claim", not one cycle** (§3.5). |

#### `noc` — `NoCEvent`

Fields as in §5, before the writer's flattening: `phase`, `txn_type`,
`src` (2-tuple, x-then-y), `dst`, `size_bytes`, `txn_id`, `issue_cycle`.

#### `mem` — `MemEvent`

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `op` | `str` | — | `read` or `write`. |
| `address` | `int` | address | Address accessed. |
| `size` | `int` | bytes | Bytes transferred. |
| `region` | `str` | — | Coarse address-derived classifier: `L1`, `DRAM`, `MMIO`. |
| `pc` | `int` | address | PC that triggered it; `0` when unattributed. |

**`cycle` is always 0** on these — see §7.5. Volume is very high
(~125k on a small workload); they are excluded from the Perfetto output
for that reason. Register-file accesses are deliberately not emitted.

#### `sync` — `SyncEvent`

`kind` (`mailbox_send`, `mailbox_recv`, `ttsync_wait_coproc_done`,
`ttsync_wait_mop_done`) and free-form `detail`. `cycle` is 0.

#### `lifecycle` — `LifecycleEvent`

`kind` (`firmware_launch_start`, `firmware_launch_done`, `kernel_start`,
`kernel_done`) and `detail`. `unit_id` is `(0, 0, 0, HOST)`; `cycle` is
0 because these come from the host, outside the simulator's clock.
Treat them as **anchors, not measurements**. They force a counter flush,
and `kernel_start` increments `kernel_id`.

#### `counter` — `CounterSnapshot`

`counter_name`, `value`, `kernel_id`. This is the event behind §4.

### 6.2 The stall-reason vocabulary

Frozen and exported as `tt_sim.trace.STALL_REASONS`, so you can switch
on it exhaustively. There is deliberately **no generic `"stalled"`** —
every name is a mechanism the model actually knows, because a code
generator acts on the reason, and a reason that does not name a
mechanism is worse than no reason at all.

| Reason | `blocked_on` | Meaning |
|---|---|---|
| `semaphore_empty` | `""` | `SEMWAIT`: selected semaphore still zero — the producer has not posted. |
| `semaphore_full` | `""` | `SEMWAIT`: semaphore at max — consumer back-pressure. |
| `resource_wait` | a unit | `STALLWAIT` condition unmet. |
| `mutex_wait` | `SYNC` | `ATGETM` accepted, mutex not yet granted. |
| `backend_enforced_stall` | a unit | A backend asserted the gate's stall directly. |
| `src_reserved_by_unpacker` | `UNPACK` | Matrix op waiting for a Src bank the unpackers still own. |
| `src_reserved_by_matrix` | `MATH` | Unpacker / ThCon waiting for a Src bank the matrix unit still owns. |
| `unit_busy` | a unit | Target unit or its IPC group still occupied. Mostly cost-model state — but see below. |
| `issue_slot_taken` | a unit | The unit's issue slot was already taken this cycle. |
| `issue_yield_fairness` | a unit | Slot yielded to a thread granted less recently. |
| `flush_pending` | `THCON` | Scalar unit mid-`FLUSHDMA`, waiting for DMA units to drain. |
| `atomic_pending` | `THCON` | Scalar unit retrying an `ATCAS`. |
| `thread_issue_block` | `THCON`, `UNPACK` | A documented whole-thread interlock: the Scalar Unit is mid-instruction for this thread, or an unpacker is mid address-phase for it, so nothing from that thread passes the gate, whichever unit it is bound for. Cost-model only. |

`thread_issue_block` is the one reason that is about the **thread**
rather than about the unit the held instruction wanted. `unit_busy`
says "the unit you asked for is busy, try another"; this one says "you
may not start anything at all yet". Both sentences it models are
quoted in `TensixBackend.block_thread_issue`.

The last two `src_reserved_*` reasons are the Src ping-pong a code
generator is trying to overlap, and **which way round it is stuck** is
the actionable part.

**`unit_busy` is not purely cost-model state**, despite what an earlier
version of the tt-sim docs claimed. An occupancy is what the cost model
arms, so almost nothing reports it with `TT_SIM_COST_MODEL` unset — but
the config unit's own single-issue throughput rule (a `SETC16`/`WRCFG`
in the previous cycle blocks this cycle's other-opcode issue) is a
structural refusal that carries the same reason. Measured: an
un-modelled run of `sfpumath` reports `tensix_stall_unit_busy` = 6
cycles, all blamed on `CFG`; the modelled run reports 84. **Do not
assert this counter is absent without the cost model.**

---

## 7. Traps

Each of these has been hit by someone building on this data. They are
listed in rough order of how expensive the mistake is.

### 7.1 A PC attribution table looks identical whether it is right or wrong

This is the worst one. `hotspots.json` is a ranked table of function
names and cycle counts. It has exactly the same shape whether every name
on it was proved correct or guessed from an unrelated kernel's build
directory — and the wrong version looks entirely plausible.

Discovery works by finding the ELFs tt-metal built and matching them to
what actually ran. **Always read the `elfs` block** in `hotspots.json`
(or `profile.json`) before believing a single row:

| `how` | What it proves |
|---|---|
| `verified` | The ELF's loadable segments were compared **byte for byte** against simulated memory. Proof. |
| `relocated +0x…` | Same proof, for a kernel placed away from its link address; the bias was recovered by finding the ELF's own text in L1. Proof. |
| `explicit` | Came from `TT_SIM_PROFILE_ELFS`. You asserted it. |
| `recent` | The newest matching ELF in the build cache, **not** confirmed against the device. **A guess.** |
| `no match (…)` | Memory was readable and no candidate is what ran. That unit is deliberately left unattributed rather than mislabelled. |

Treat `recent` as a strong hint and label it as such in your UI. It is
common and often correct — but a run replaying a captured trace, which
rebuilds nothing, can get `recent` for every core and attribute the
whole run to whichever kernel was compiled most recently. Producing
exactly that table is what motivated this section.

`hotspots.json` carries the `elfs` block and a `cost_model` flag so it
is self-describing; you do not need `profile.json` to know how much to
trust it.

If attribution matters, pin it: `TT_SIM_PROFILE_ELFS=BRISC:/path.elf,…`
(a trailing `@0x1234` sets an explicit bias) and you get `explicit`.

### 7.2 Kernel PCs are relocated

tt-metal places a kernel at a runtime base different from its link
address. Read off a real build cache: the BRISC kernel ELF links at
`0x49a0` and the TRISC2 kernel at `0x6f70`, while the same run retired
instructions at PCs above `0x7bcc` on TRISC2 and `0xa394` on NCRISC —
outside every link range in the cache. Firmware is *not* relocated; it
links to a fixed L1 address and is resident there.

**Every `pc` in the event stream and in `hotspots.json` is a runtime
address.** The ELF's symbols are at link addresses. `profile.json` and
`hotspots.json` record the `bias` per ELF as an integer, so:

```
link_address = runtime_pc - bias
```

Never mix the two, and never present one labelled as the other — a
downstream consumer cannot tell them apart and the error is
unrecoverable. `bias` is `0` for firmware and for any ELF chosen by
recency (in which case you do not know the bias, and the attribution is
a guess anyway — see §7.1).

One case to expect: the NCRISC kernel links into private IRAM
(`0xffc00000`), which cannot be read back through the tile, so discovery
abstains rather than rejecting, and falls back to recency with a zero
bias.

### 7.3 Counter names are an open set

Covered in §4.2, repeated here because it is the most common cause of a
quietly-wrong dashboard. A query that enumerates stall reasons will
silently omit any reason added after it was written, and new reasons
appear **without a code change** — that is the design. Match on
`counter_name LIKE 'stall\_%'`, not on an `IN` list.

### 7.4 Nothing is sorted

Row order is not part of the contract. `StallEvent`s in particular are
**not** in cycle order: an episode is published when it *ends* but
stamped with the cycle it *began*, and different threads end episodes at
different times. The workload measured here had 4 backward cycle steps
in 32 stall events. Sort explicitly.

### 7.5 `cycle` is a flush boundary, not an event time

On a `CounterSnapshot` — and therefore in the Parquet counter dataset —
`cycle` is **the cycle at which the aggregator flushed**, not the cycle
any individual event happened. Flushes happen every
`TT_SIM_TRACE_COUNTERS_INTERVAL` cycles (default 100) and at every
lifecycle boundary. Do not read a counter row's `cycle` as "this
happened at cycle N"; read it as "this accumulated in the window ending
at N".

Worse, **`MemEvent` carries `cycle = 0`** — memory accesses mostly
originate from non-clocked dispatch paths — as do `SyncEvent` and
`LifecycleEvent`. Those events do not advance the flush clock, so every
memory counter lands in whatever bucket happens to be current. Memory
counters are therefore reliable as **run totals** and unreliable as a
**time series**. (The Perfetto writer stamps such events at the highest
cycle seen so far, so they appear at "now" rather than piling onto t=0.
That is a rendering choice, not a measurement.)

`instr`, `dispatch`, `compute`, `noc` and `stall` all carry real
cycles.

### 7.6 Coordinates and architectures

Covered in §3.2. Do not join Wormhole and Blackhole datasets on
`unit_id`. `UNKNOWN` is a real value, not corrupt data.

---

## 8. Perfetto timeline

`TT_SIM_TRACE_PERFETTO=<path>` writes Chrome Trace Event Format JSON;
a `.gz` extension enables gzip (Perfetto loads it natively; traces
compress 10–20×). Drag the file onto <https://ui.perfetto.dev>.

`ts = cycle`, one trace **microsecond per simulated cycle**. The
`displayTimeUnit` is `ns`, so what the UI labels "µs" is a cycle.

Layout: one process per tile, one thread track per unit, plus a
`<thread> stall` track beside each Tensix thread's own track.

| Slice | Width | Source |
|---|---|---|
| `pc=0x…` | 1 cycle | one retirement per cycle |
| `stall:<reason>` (RV) | `InstrEvent.stall_cycles` | RV cost model; abuts the instruction it precedes |
| `dispatch:<op>` | 1 cycle | issue is single-cycle |
| `<opcode>` (compute) | `ComputeEvent.duration`, or 1 if unmodelled | cost tables |
| `stall:<reason>-><unit>` | `StallEvent.cycles` | a whole Tensix stall episode, on its own track |
| `noc:<phase>:<type>` | issue → arrival | async slice pair (`b`/`e`) |

With `TT_SIM_COST_MODEL` unset every width collapses to 1, and **that is
the truth rather than a placeholder**: nothing stalls and every unit
retires in its issue tick. The writer will not invent a plausible number
to fill the gap. Which regime produced a trace is stated three ways, so
a file copied away from its run is never ambiguous:

- `otherData.cost_model` and `otherData.timing_note` in the trailer
  (Perfetto shows these under **Info and stats**);
- a `process_labels` metadata event per tile reading
  `TT_SIM_COST_MODEL on` / `off`;
- a `timing_model` argument on any slice whose width is not a modelled
  figure — and on **every** stall slice, because a stall width is the
  number most likely to be quoted out of context.

NoC transactions are async slices with request→response arrows carried
as `bind_id` + `flow_out`/`flow_in` on the slices themselves.
`mem` events are omitted entirely (volume).

Canned SQL for Perfetto's **Query (SQL)** tab:
[`tt_sim/trace/queries/README.md`](../tt_sim/trace/queries/README.md).

---

## 9. The ranked report and its JSON

`TT_SIM_PROFILE=<dir>` writes, at process exit:

| File | Contents |
|---|---|
| `report.md` | Human-readable ranked bottleneck report. **Layout not frozen** — do not parse. |
| `report.json` | The same data, machine-readable. |
| `hotspots.json` | Per-PC and per-function attribution, plus its own provenance. |
| `profile.json` | Run metadata: cost-model regime, ELF choices with `how` and `bias`, counter directory. |
| `counters/` | The Parquet dataset of §4. |

Re-render at any time without re-running the simulator:

```bash
python3 -m tt_sim.trace.report <profile-dir> --stdout
```

### `report.json`

| Field | Type | Meaning |
|---|---|---|
| `span` | `int` | Highest cycle observed. The denominator for every share. |
| `cost_model` | `bool \| null` | Regime; `null` if unknown. |
| `contributions[]` | list | `{unit, counter, cycles, described, discovered}` — cycle-bearing counters, ranked. `unit` is `"<core_y>,<core_x> <UNIT>"`. |
| `volumes{}` | map | Non-cycle counters, totalled. |
| `attributed_cycles` | `int` | Sum of `contributions` — **not** a partition of the run (§4.3). |
| `shared_resource_cycles{}` | map | The only sum here meaningful across units: `noc_flight_cycles` and `busy_cycles` only. |
| `hotspots{}` | map | Embedded copy of `hotspots.json`. |
| `elfs[]` | list | Provenance (§7.1). |
| `notes[]`, `label` | | Free-form. |

`discovered: true` means the report has no hand-written prose for that
counter, not that anything is wrong — it is how a counter added
yesterday still gets ranked.

### `hotspots.json`

| Field | Meaning |
|---|---|
| `total_cycles` | Sum over all PCs of `retired + stall_cycles`. |
| `resolved_cycles` | The share of that with a source location. |
| `unattributed_units[]` | Units that executed code but had no ELF loaded. |
| `functions[]` | Folded per `(unit, function)`, so an inlined callee is named rather than the `kernel_main` it vanished into. |
| `pcs[]` | Per `(unit, pc)`. `pc` is a **runtime** address (§7.2). |
| `elfs[]`, `cost_model` | Provenance — **read this first** (§7.1). |

A row's `cycles` is `retired + stall_cycles`: one cycle per instruction
retired there, plus every cycle the core was held before issuing it.
`by_reason` splits the stall half.

Attribution **refuses to guess**: the DWARF index answers only inside
the address ranges the line program actually covers, and a core whose
resident code matches no candidate ELF is left unattributed rather than
labelled from the wrong kernel. Coverage below 100 % is honest reporting,
not a failure.

---

## 10. Reporting a problem with this document

A field whose meaning or unit is not stated here, or is stated wrongly,
is a defect — not something to work around. The tests in
`tt_sim/trace/observability_test.py` and
`tt_sim/trace/attribution_test.py` enforce the frozen half of this
contract (field names, event classes, Parquet columns, the unit-alias
table, the redundancy rules); if you find a promise here that no test
protects, that is worth raising too.
