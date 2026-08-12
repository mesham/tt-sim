# The cycle-cost tables

Status: **data landed; eight of the nine Tensix backend units wired, the baby
RISC-V load/store path, the NoC latency and bandwidth models, and both arches'
DRAM access latency plus Wormhole's DRAM channel bandwidth.** Validated
externally on the **NoC and memory path only** — the Tensix instruction costs
rest on provenance alone, and
["There is no rung-3 dataset for Tensix compute"](#there-is-no-rung-3-dataset-for-tensix-compute-and-that-changes-what-rung-2-climbed-means)
says why that is not going to change without silicon. Two YAML files and
a tested loader, plus `tt_sim/perf/model.py`, which turns a table entry into
the occupancy a unit is charged. The Tensix **matrix, vector (SFPU), scalar
(ThCon), packer, sync, config, unpacker and mover** units read it, so do the
**five baby RISC-V cores**, and so does the **NoC**, behind the
`TT_SIM_COST_MODEL` opt-in; with the variable unset nothing loads and no timing
changes. The last two are the ones whose cost is not a per-opcode constant at
all — see ["Unpacker and mover
occupancy"](#unpacker-and-mover-occupancy-the-last-two-units-and-the-first-cost-that-is-a-function-of-the-transfer),
which is also where a modelled charge first scales with a workload's *data*
rather than with its instruction mix, and where the 80 B/cycle joint unpacker
ceiling is refused for want of a sourced sharing rule. See ["The first
consumer"](#the-first-consumer) below for what that cost in numbers — including
the result that matters most: with all six Tensix units costed **not a single
simulated cycle moves**. The config unit is the one that took two attempts:
charging its then-documented multi-cycle `RDCFG` made a matmul come out wrong,
which turned out to be a real ordering bug in the mechanism rather than a
property of the unit, and it was wired on
[the second pass](#the-config-unit-on-the-second-pass) once that was fixed and
gated. It is also the only unit whose documentation publishes **IPC groups**,
so occupancy is charged
[per group rather than per unit](#occupancy-per-ipc-group-the-only-per-group-constraint-anyone-publishes)
— a grouping transcribed from a column, never inferred, and one that turns out
to cost nothing either way because nothing in the tree contends for that unit.
["The RISC-V cores"](#the-risc-v-cores-where-the-cycles-finally-moved) is where
a cycle count finally does move — by 0–7.8 % depending on the workload — and
where a second timing-perturbation bug turned up.
["The NoC hop model"](#the-noc-hop-model-where-a-total-changes-shape) is where
one changes *shape*: `six` goes from 27,168 cycles to **63,425**, 61 % of it
NoC flight time, and the model's hop term independently reproduces tt-metal's
measured Wormhole local-vs-remote latency difference to within 5 cycles.
["The NoC bandwidth
model"](#the-noc-bandwidth-model-where-a-packets-size-starts-to-matter) is
where two workloads the hop model priced identically finally come apart — a
tile read costs **26× per packet** what a semaphore poke costs — and where the
per-trid response-ordering hazard that section recorded is closed by
construction rather than watched.
["DRAM access
latency"](#dram-access-latency-where-the-number-had-to-be-derived-rather-than-read)
is where a number first has to be **derived rather than read** — 99 cycles on
Wormhole, the difference of two vendor measurements chosen so that the NoC
cancels — and where the provenance convention gains a rank
(`vendor_source_derived`) to say so out loud. Blackhole deliberately got
nothing — until ["Blackhole's DRAM latency, and two blockers that were not
about Blackhole"](#blackholes-dram-latency-and-two-blockers-that-were-not-about-blackhole)
went and read the sources, found that **one blocker was a stale constant on a
path that never touches a latency and the other was failing on a row the
subtraction does not use**, and shipped 126 — worth **+24.1 %** on `six`. On
the way it produced the first external check of the NoC hop model on
Blackhole: tt-metal's 740-row measured dataset reproduces the ISA docs'
9-cycle hop on *both* architectures, to within 4 %, from a key
(`same_axis`) that never mentions hops.
["The gate"](#the-gate) is how any of this is accepted: since a timing model
cannot be validated by byte-identical replay, `driver/tests/cost_model_gate.py`
runs the guards that *are* timing-agnostic with the model on, and **proves**
rather than excuses the timing-pinned ones' mismatches.
["Two queues, not one"](#two-queues-not-one-what-rung-2-handed-over-and-what-it-did-not)
is where rung 2's findings get spent, and it is the first instalment that lands
because **an earlier section's reasoning was shown to be wrong**: the argument
that size dependence "belongs in one place, once" confused two pipelined stages
for one queue, and wiring the already-sourced `dram.bandwidth` at the DRAM
endpoint takes the DRAM residual's slope from **+10.03 cycles/KiB to −0.65**,
landing its intercept on the same 77–80 the untouched L1 rows sit at. The same
instalment answers rung 2's other question with a **no** — the ~10 % L1 read
shortfall is *not* sourceable; the NoC's documented one-flit packet header
explains 3–6 % of it and none of its read-versus-write asymmetry — and records
the thing that most changes how this file should be read: **the Tensix
instruction costs have no external check of any kind**, because no such dataset
exists to check them against.
This is ROADMAP [**§I**](../../ROADMAP.md)'s "Per-unit cycle-cost tables"
bullet, which names the deliverable exactly:

> Tables should live next to the YAML they parallel
> (`tensix_instructions.yaml` → `tensix_instruction_costs.yaml`).

Phase 5 of [`event-driven-pump.md`](event-driven-pump.md) is where the data
becomes load-bearing. Standalone rather than a section of that plan because the
pump plan is about *when things get ticked* and this is about *what a tick
costs*: the two land independently, in either order, and a reader who wants to
add a number to the table should not have to read a pump design to do it.

## Why this is data and not code

Today every Tensix op, RV instruction, NoC request and Mover transfer completes
in the tick it was issued. Giving an instruction a duration needs a queue keyed
by integer cycles, which is Phase 4 of the pump plan. Landing the *table* first
is deliberate:

- The schema and the provenance discipline can be reviewed on their own,
  against the ISA docs, without a reviewer also having to hold a scheduler in
  their head.
- Authoring the table is the part that is bounded by *reading*, not by
  engineering. It is done once; consuming it will be revised repeatedly.
- It makes the size of the honest gap visible before anybody builds machinery
  that assumes the gap is small. It is not small — see "What is missing".

## Where things are

| File | Holds |
| --- | --- |
| `tt_sim/pe/tensix/tensix_instruction_costs.yaml` | Per-instruction costs for the ten Tensix backend units |
| `tt_sim/perf/unit_costs.yaml` | NoC, DRAM, baby RISC-V cores, Mover, L1 (only `l1` has no consumer now) |
| `tt_sim/perf/costs.py` | The loader, `load_costs(arch)` |
| `tt_sim/perf/model.py` | The consumer-side policy: table entry → cycles to charge |
| `tt_sim/perf/costs_test.py` | 48 tests: parse, provenance integrity, loader fidelity, coverage, "exactly these modules consume this", and the `corroboration` field's discipline |
| `tt_sim/perf/model_test.py` | 12 tests: off by default, bound policy, fidelity phases |
| `tt_sim/pe/tensix/matrix_cost_model_test.py` | 6 tests: the FPU driven with the model on and off |
| `tt_sim/pe/tensix/backend_cost_model_test.py` | 21 tests: the SFPU, ThCon, packer, sync and config units, same treatment — including the config unit charging nothing on Wormhole, Blackhole's `CFGSHIFTMASK` 2, and its 2-cycle hold on the `Config` IPC group leaving `SETC16` free |
| `tt_sim/pe/tensix/unpacker_cost_model_test.py` | 27 tests: the address phase at 2, the data phase at every throttle rate on both arches, the tileize-forces-x4 and Blackhole default-mode paths, a blocked unpacker charged nothing for waiting, and the two deliberate under-charges (the joint ceiling, the cross-unpacker hold) pinned by name |
| `tt_sim/pe/tensix/mover_cost_model_test.py` | 13 tests: the transfer duration per kind against the doc's own arithmetic, the contended column recorded and unspent, the XMOV and TDMA paths both charged, and a transfer in flight reading as outstanding work |
| `tt_sim/pe/tensix/frontend_backpressure_test.py` | 9 tests: the bounded front-end FIFO, the stalled `.ttinsn` store, the dvalid-twice wedge reaching the core, and the two licensed terms' arithmetic (push at 1.000, ThCon at 3.0, 3 threads at ~3× each) |
| `tt_sim/pe/rv/cost.py` | The baby RISC-V consumer: address-region classifier, load-use scoreboard, L1 store rate limiter |
| `tt_sim/pe/rv/cost_test.py` | 19 tests: off by default, the interlock, the unnamed regions, stores, multiply/divide |
| `tt_sim/network/tt_noc.py` | The NoC consumer: `noc_hop_count`, `NUI.send_to`, and the in-flight packet queue |
| `tt_sim/network/noc_cost_model_test.py` | 42 tests: opt-in, torus hop counting on both arches and both NoCs, flight time, an end-to-end DRAM read landing on the predicted cycle, the bandwidth terms, and a forced out-of-order response landing where it belongs |
| `tt_sim/device/tiles.py` | The DRAM consumer: `DRAMEndpointNUI` holds an arriving request for the channel's own service time, plus the channel bandwidth's excess over the NoC link's |
| `tt_sim/device/dram_cost_model_test.py` | 18 tests: opt-in, both arches' derivations and their provenance rank, the channel rate and its Blackhole refusal, the named gaps, and a DRAM read costing flight + service + channel + flight on a real device |
| `tt_sim/perf/noc_dataset_sweep.py` | **Rung 2.** Sweeps tt-metal's 8,140-point measured NoC dataset against the whole assembled model; declares its exclusion criteria up front and reports residuals by axis |
| `tt_sim/perf/noc_dataset_sweep_test.py` | 29 tests: the exclusion ladder, the dataset's undocumented keying, the predictor against the closed form, and the sweep itself (skipped without tt-metal) |
| `driver/tests/cost_model_gate.py` | **The gate.** Runs the timing-agnostic guards with the model on, and proves the timing-pinned ones' mismatches benign |
| `driver/tests/guard_classification_test.py` | 10 tests: the millisecond tripwire that keeps the guard classification from rotting |

The Tensix file sits next to `tensix_instructions.yaml` because §I says so, and
because the two are meant to be read side by side: the same `ex_resource` keys
(`MATH`, `SFPU`, `THCON`, `SYNC`, `TDMA`, `CFG`, `PACK`, `UNPACK`, `XMOV`,
`NONE`), the same instruction names. One says what an opcode *is*, the other
says what it *costs*. A test asserts the unit keys are exactly the set of
`ex_resource` values the instruction table uses, so the two cannot drift apart
silently.

The non-Tensix units have no YAML to sit next to. Their nearest equivalent is
`tt_sim/arch/profile.py`, which already owns arch-level hardware constants, so
they live in a new `tt_sim/perf/` package alongside the loader instead of being
crammed into a file named after the Tensix coprocessor.

## The schema

### Cost fields

```yaml
MVMUL:
  latency: 5              # cycles until the result is observable
  throughput_ipc: 1       # instructions per cycle, as the doc states it
  occupancy: 1            # cycles the unit cannot begin another instruction
  scales_with: fidelity_phases
  provenance: isa_doc_derived
  source: wh_matrix_unit#instruction-latency-and-throughput
  derivation: occupancy = 1 / throughput_ipc
```

§I asks for "an issue cost and an occupancy". The ISA docs, though, publish
**latency and throughput** — those are the columns of every per-unit table they
have. So the file records latency and throughput as stated, and `occupancy`
(the number Phase 5 actually wants: "busy until cycle `c + N`") is a third
field that is only present where it can be justified. Sometimes that is direct
— the Scalar Unit's table is literally headed "Number of cycles required for
execution", which *is* occupancy — and sometimes it is `1 / throughput_ipc`, in
which case the entry says so in `derivation` and its provenance drops a rank.

Where a document gives one of the pair and not the other, only the one it gives
is recorded. There is no attempt to complete a row.

### Cost values carry their bound

A cost is either a plain integer, meaning exactly N cycles, or a mapping:

```yaml
occupancy: { cycles: 2, bound: at_least }        # the doc wrote ">= 2 cycles"
latency:   { cycles: 5, bound: approximate }     # the doc wrote "~5 cycles"
latency:   { cycles: 2, bound: at_most }         # the doc wrote "<= 2 cycles"
occupancy: { cycles: 3, max: 4, bound: range }   # the doc wrote "3 or 4"
```

This exists because the ISA docs genuinely write all four forms, and often. Of
the Scalar Unit's eleven rows, six are `≥` and four are "3 or 4"; exactly one
is an unqualified integer. Flattening those to bare numbers would manufacture
precision that was never published, and the qualifier is the *useful* part when
the consumer is an estimator rather than a validator: `≥ 15` for `ATCAS` says
"this is where to look when a kernel is slow" in a way that `15` does not.

The loader normalises all four to `CycleCost(cycles, bound, max_cycles)`.

### Per-arch differences

One file per arch (the `tensix_backend_cfg.yaml` /
`tensix_backend_cfg_blackhole.yaml` pattern) would duplicate ~95% of the
content, because the two architectures' published cycle counts are largely
identical. So the arch pattern here follows `ArchProfile` instead — one set of
fields, per-arch values — via a deep-merged `arch_overrides` block:

```yaml
arch_overrides:
  wormhole: {}
  blackhole:
    units:
      CFG:
        instructions:
          SETC16: { throughput_ipc: 3, ... }
```

Instructions that exist on only one architecture are marked in place with
`arch: blackhole`, matching how `tensix_instructions.yaml` keeps Blackhole's
superset opcodes in the shared table; `load_costs("wormhole")` drops them.

The real per-arch differences turned out to be few and worth naming:

| | Wormhole | Blackhole |
| --- | --- | --- |
| Clock | 1 GHz | 1.35 GHz busy / 0.8 GHz idle |
| NoC flit | 256 bits | 512 bits (hop latencies unchanged: ~5 / 9 / ~5) |
| RV multiply | 2 cycles blocking the integer unit | 1 cycle EX1 + 1 cycle EX2 (pipelined) |
| RV branch mispredict | 2-cycle bubble (looks like 3) | 4-cycle bubble (looks like 5) |
| RV L1 load | ≥ 8 | 2 on L0 d-cache hit, ≥ 8 on miss, ≥ 12 for atomics |
| Config unit | prose throughput limits | explicit IPC groups (an `IPC group` column: `ThreadConfig` / `Config`) + a −4…+1 pipeline; adds `CFGSHIFTMASK`, `STREAMWRCFG` |
| SFPU | — | adds `SFPARECIP` / `SFPGT` / `SFPLE` / `SFPMUL24`; **every shared op has identical latency** |

That last row is the useful negative result: the Blackhole Vector Unit page
lists the same IPC and latency as Wormhole for every instruction the two share,
so the SFPU needs no per-arch cost override at all.

## The provenance convention

This is the part that matters. ROADMAP "Positioning" is explicit that tt-sim is
a *first-order performance estimator*, not cycle-accurate, because calibrating
against silicon needs RTL or captured traces that are not publicly available. A
table of unattributed constants under those conditions is worse than no table:
it cannot be improved, because nobody can tell which numbers came from a
document and which were plausible-looking guesses. So every entry — including
the unit-level and section-level blocks, not just instruction entries —
declares one of:

| `provenance` | Means | Requires |
| --- | --- | --- |
| `isa_doc` | Stated verbatim in the public Tenstorrent ISA documentation | `source` |
| `isa_doc_derived` | Arithmetic on an `isa_doc` number | `source` + `derivation` |
| `vendor_source` | From vendor source or tech reports (tt-metal, ttsim), not the ISA contract | `source` |
| `vendor_source_derived` | Arithmetic on vendor numbers | `source` + `derivation` |
| `estimated` | Uncalibrated. A guess, however educated | `note` saying what would replace it |
| `unknown` | No source gives one; deliberately left blank | `note`, and **no numbers at all** |

`vendor_source_derived` is the newest rank and was added, deliberately, for one
entry: [DRAM access
latency](#dram-access-latency-where-the-number-had-to-be-derived-rather-than-read),
which is a subtraction of two vendor measurements and therefore neither
`isa_doc_derived` (no ISA doc is involved), nor `vendor_source` (nobody printed
the answer), nor `estimated` (nothing about it is a judgement call). It is to
`vendor_source` exactly what `isa_doc_derived` is to `isa_doc`, one tier down,
and carries the same obligation to show its working. A rank is a serious thing
to add — the whole convention is that a reader can tell published from derived
at a glance — so the bar is that the new rank makes a *real* distinction the
existing five could only misreport, and that it ranks the entry **lower** than
anything a document says.

Three rules make this checkable rather than decorative, and all are enforced by
tests:

1. **An entry's provenance is the weakest provenance of any number in it.** An
   entry whose latency is documented but whose occupancy is derived reads
   `isa_doc_derived`. This understates what is sourced, which is the safe
   direction.
2. **An `unknown` entry carries no cycle counts.** A gap must read as a gap. An
   entry that claims `unknown` and then supplies a number is the exact failure
   the convention exists to prevent, so `test_unknown_entries_carry_no_numbers`
   rejects it.
3. **A derived entry shows its working.** Both derived ranks must carry a
   `derivation` complete enough for a reader to redo the arithmetic
   (`test_derived_entries_show_their_working`), and the entries at the weaker
   of the two are pinned to an explicit list
   (`test_vendor_derived_entries_are_exactly_the_ones_we_expect`) — the same
   device that keeps `estimated` at zero entries. Adding a derived number is
   then an edit to a list plus a paragraph of arithmetic, which is the intended
   friction.

`source` is a short document id (`wh_matrix_unit`, `bh_config_unit`,
`tm_gemm_flops`) resolved against a `documents:` / `vendor_documents:` map at
the top of each file; a test asserts every citation resolves. Vendor sources
are kept in a separate map and a separate rank because they are second-tier
authority and because **they disagree with each other** — its LLK
performance-counter doc weights fidelity phases 1/2/4 where its C++ model uses
1/2/3/4. Those conflicts are recorded in the notes rather than resolved.

One that *was* resolved is worth the exception, because it had a gap resting on
it: the file used to record tt-metal as internally inconsistent about the
Blackhole clock, 1.2 GHz against 1.35. It is not. **1.2 appears once in the
tree, in a bandwidth-to-cycles conversion, and never touches a latency**;
everything else that names a Blackhole frequency says 1350, UMD defines the
DVFS pair as 1350 busy / 800 idle, and tt-metal *reads* AICLK off the device
rather than assuming it. Chased in full
[below](#blocker-1-the-clock-which-was-one-stale-constant). Recording a
conflict is the right default; checking one is better when a number depends on
it.

## What is sourced

Far more than expected. The ISA documentation turns out to carry real
latency/throughput tables for most of the Tensix units, and a genuinely good
pipeline description for the baby RISC-V cores.

**Fully sourced from the ISA docs (`isa_doc` or `isa_doc_derived`):**

- **Matrix unit** — the complete latency/throughput table. `MVMUL` / `DOTPV` /
  `GAPOOL` / `ELWMUL` / `GMPOOL` / `ELWADD` / `ELWSUB` at 5 cycles, 1 IPC; the
  RWC and zero/shift ops at 1; `SHIFTXB` at 2 and 0.5 IPC; the `MOV*` family at
  2 or 4. 22 of 39 `MATH` opcodes.
- **Vector unit** — per-instruction latency for all 42 SFPU opcodes. Arithmetic
  and LUT ops 2 cycles, everything else 1, with the mode-dependent ones
  (`SFPSHFT2`, `SFPCONFIG`, `SFPSWAP`) recorded at their worst case with
  `bound: at_most`.
- **Scalar unit (ThCon)** — the only table that is *already* an occupancy
  table. 1 cycle for `DMANOP` / `SETDMAREG`, ≥ 2 for `REG2FLOP` / `FLUSHDMA`,
  3-or-4 for the arithmetic GPR ops, ≥ 3 for the load/store ops, ≥ 15 for
  `ATCAS` / `ATINCGETPTR`.
- **Config unit and sync unit** — both arches' tables, in full. Blackhole's
  `CFGSHIFTMASK` at 2 cycles is the only entry in either that is not 1, and it
  is the one that made wiring the config unit a timing change.
- **Miscellaneous unit** — one blanket sentence covers all nine ADC ops.
- **Unpacker** — both halves, and both now charged: the ≥ 2-cycle
  address-calculation phase (exactly 2 uncompressed) during which no thread may
  start an `UNPACR`, and the 16/32/64 B-per-cycle throttle modes for the data
  phase, plus Blackhole's own x8 / x4-"2x" / default-mode rules stated in the
  shared page's `TTArchitecture` conditionals.
- **NoC** — per-hop latency (~5 NIU→router, 9 router→router, ~5 router→NIU) and
  one flit per cycle per axis. This is exactly the shape Phase 5 wants.
- **DRAM bandwidth** — 24 GB/s per channel, 12 channels, 288 GB/s aggregate,
  ~92% achievable, plus the docs' own measured tables. The per-channel figure
  is now *consumed*, as 24 B/cycle at the docs' own 1 GHz
  (`dram.channel_serialisation`, `isa_doc_derived`); the ~92% achievable
  fraction is not, because it is a statement about sustained software behaviour
  rather than about one transfer.
- **NoC packet framing** — exactly one header flit per packet, up to 256 data
  flits, and single-flit read requests / write acks / atomics. Recorded and
  deliberately unconsumed; see "What is missing".
- **Baby RISC-V cores** — the full §I ask. Fetch/issue/retire stages with
  minimum occupancies, multiply 2 cycles, divide 6–33 (with the two-cycle
  special cases), branch mispredict penalty, the complete load-latency table by
  address region, the sustained-load throughput formula, and store throughput
  (one L1 store every five cycles).
- **Mover** — issue is 1 cycle; transfer rates are published as *measured*, with
  both an ideal and a contended column (93.1 vs 32 bits/cycle for L1→L1). The
  ideal column is charged as the mover unit's occupancy; the contended one is
  recorded and not, because nothing sources when contention applies.
- **L1** — 16 banks, 16 ports, 128 bits/bank/cycle, and the 5-cycle
  read-modify-write penalty that explains why RISC-V stores to L1 cost 5.

**From vendor source (`vendor_source`), where the ISA docs are silent:**

- **Fidelity-phase cost.** The ISA docs say only that each phase needs one more
  instruction. tt-metal's GEMM_FLOPS report gives cycles per *tile*: LoFi 16,
  HiFi2 32, HiFi3 48, HiFi4 64, backed by the 4096-muladds-per-cycle figure.
  Recorded at unit level rather than folded into `MVMUL`, because the
  conversion to a per-instruction cost needs a second number: how many `MVMUL`s
  a tile takes. That turned out to be a hardware property after all rather than
  a kernel one — `MVMUL`'s eight-row shape fixes a 32×32×32 tile matmul at 16
  instructions per fidelity phase — so it now sits alongside as
  `fidelity_phases.mvmuls_per_tile` (`isa_doc_derived`), and the division is
  what the matrix unit charges. See
  ["The first consumer"](#the-first-consumer).
- **Joint unpacker ceiling** of 80 B/cycle. The ISA docs' L1 page supplies the
  "five 128-bit reads per cycle" half; only the conversion to bytes is vendor.
  Sourced and **deliberately unconsumed**: it is a limit *shared* between two
  simultaneously streaming unpackers, and no source states how it divides per
  transfer, so each unpacker is charged its own uncontended rate instead. See
  ["Unpacker and mover
  occupancy"](#unpacker-and-mover-occupancy-the-last-two-units-and-the-first-cost-that-is-a-function-of-the-transfer).
- **Chip-level DRAM bandwidth**, 258 GB/s Wormhole and 512 GB/s Blackhole —
  the only DRAM bandwidth figure of any kind that exists for Blackhole, whose
  ISA-doc tree has no DRAM tile directory at all.
- **End-to-end transaction latencies** (DRAM 358 cycles Wormhole / 529
  Blackhole, L1 remote read 259 / 403, L1 local 56 / 88). Filed under
  `dram.end_to_end_reference`, explicitly *not* as a DRAM latency: any single
  row folds in the NoC round trip and the issuing core's path. The
  *difference* of two of them does not, which is where **both** arches' DRAM
  access latency came from — at its own, weaker, provenance rank. All eight are
  measured *cycle* counts: the function that returns them applies a device
  frequency to its bandwidth result and not to these, which is what makes the
  subtraction clock-free.

## What is missing

There are no `estimated` entries. Everything in the tables is either published,
derived from something published with the arithmetic written out, or explicitly
absent, and a test asserts each of those — adding a guess means editing a list,
which is the point. The `unknown` entries are:

- ~~**DRAM access latency, on Blackhole.**~~ **Closed 2026-08-03** at 126
  cycles, `vendor_source_derived`, by the same subtraction as Wormhole's 99 —
  see ["Blackhole's DRAM latency, and two blockers that were not about
  Blackhole"](#blackholes-dram-latency-and-two-blockers-that-were-not-about-blackhole).
  Kept in this list as a struck-through entry rather than deleted, because how
  it closed is the useful part: both recorded reasons for the gap survived
  several instalments and **neither was about Blackhole's DRAM**. One was a
  1.2-vs-1.35 GHz clock conflict that turns out never to touch a measured
  cycle count; the other was a consistency failure that turns out to be
  confined to the one end-to-end row the subtraction does not read. A gap
  whose stated reasons are precise is a gap somebody can go and check.
- **`end_to_end_reference.l1_local_cycles` on Blackhole**, 88, which is the
  residue of that second reason. It sits 54 cycles below where Blackhole's own
  three other rows put it under a hop model that is now independently
  confirmed on that arch. It is a defect in a *calibration target*, not in a
  cost, and nothing consumes it — but rung 1 of the ladder cannot fully pass on
  Blackhole until somebody explains it.
- **DRAM bank conflicts and refresh windows**, which §I names directly. No
  source quantifies either, and tt-sim has no DRAM bank model at all, so the
  device term above is a flat per-request latency and says so. Endpoint
  occupancy is likewise unmodelled: a second request is not queued behind the
  first.
- **Packer completion cost.** The issue side is documented (at most one `PACR`
  started per cycle; the thread blocks only until the packers *accept*) and the
  L1 write bandwidth is documented, but there is no datums-per-cycle or
  cycles-per-tile figure anywhere — not in the ISA docs, not in tt-metal, not
  in the LLK, whose performance-counter doc describes how to *measure*
  packer-busy cycles without saying what they are.
- **NoC congestion.** Per-hop latency and per-link bandwidth are sourced; head-
  of-line blocking and router arbitration are not. The docs describe the router
  buffers (2 KiB per inbound port, 32 B guaranteed per VC, 480 B claimable from
  the shared pool) and say congestion "can negatively impact latency" without
  quantifying it. This is now also where the **~10 % L1 read bandwidth
  shortfall** ends up: rung 2 measured it, and the search for a published
  per-packet explanation came back with a one-flit packet header worth 3–6 % of
  it and nothing else — see ["Is the ~10 % L1 read shortfall
  sourceable?"](#is-the-10--l1-read-shortfall-sourceable).
- **The NoC packet header, which is sourced and deliberately unspent.** Not a
  gap in the *sources* — `noc.packet_framing` records the docs' "exactly one
  header flit" at `isa_doc` — but a gap in what the model charges, and the only
  entry in either file whose reason for being unconsumed is proportion rather
  than provenance: it doubles a semaphore poke's modelled link occupancy to buy
  0.39 % on a 64 KiB transfer, so it wants its own change. The same applies to
  `noc.request_packet_flits`.
- **Seven individual opcodes** with no published timing at all: `MOVD2B`,
  `TRNSPSRCA`, `MOVDBGB2D`, `RSTDMA`, `STREAMWAIT`, `TBUFCMD`, `RESOURCEDECL`.
  A further three are costed but not by a constant: `SFPLOADMACRO`, whose
  latency the docs call "Complex" because it expands to up to four more
  instructions, and `REPLAY` / `MOP`, whose cost is the length of their
  expansion — an instruction field, not a hardware number.
- **`DMANOP`'s unit.** `tensix_instructions.yaml` marks it `ex_resource: TDMA`
  (so tt-sim dispatches it to the Miscellaneous Unit) while the ISA docs put it
  in the Scalar Unit. Both units give it 1 cycle, so the disagreement is about
  which unit is occupied, not about the cost. It is costed under `THCON` and
  cross-referenced from `TDMA`; a test asserts no instruction is costed twice
  and that any unit disagreeing with `ex_resource` says why.
- **Fourteen `MATH` opcodes** listed under `not_documented`. Seven are the
  legacy `CONV3S*` / `APOOL3S*` / `MPOOL3S*` family the docs describe as
  neutered and mark `NonContractualBehavior` — nothing worth costing. The
  `SETASHR*` / `SETIBRWC` / `SETPKEDGOF` / `RAREB` group simply has no
  published timing.
- **Blackhole L1 bank geometry.** Blackhole's L1 is 1536 KiB rather than 1464,
  so the bank size must differ, but no bank count or size is published. 96 KiB
  is the obvious inference and is deliberately not recorded as a fact.

A test enumerates the gaps (`test_unsourced_lists_exactly_the_entries_...`), so
closing one is a visible change rather than a quiet edit.

## What calibration would take

ROADMAP §I's "Calibration against silicon traces" bullet sets the bar at "match
a captured cycle trace within X%". That data does not exist publicly, and this
work does not change that. What it does change is that there is now something
concrete to calibrate, and a ladder of cheaper checks below the silicon one:

1. **Internal consistency.** The end-to-end numbers in
   `dram.end_to_end_reference` are a free check available today: a per-hop NoC
   model plus a DRAM term plus the RISC-V issue path should reproduce
   tt-metal's measured 358 cycles for a Wormhole DRAM transaction. If it does
   not, the hop model is wrong before any silicon is involved. **Passes on
   Wormhole** (residuals 36 / 38 / 41; see the hop-model section). On Blackhole
   three of the four rows agree and the fourth, `l1_local_cycles`, does not —
   the one open defect left by the DRAM instalment below. (The clock caveat
   this rung used to carry is withdrawn: those figures are measured cycle
   counts and no clock converts them.)
2. **tt-metal's measured NoC dataset.** 740 entries of empirically measured
   end-to-end latency keyed by transaction size, access pattern, memory type
   and subordinate count, shipped in tt-metal's `noc_estimator`. Too coarse to
   *derive* a per-hop congestion model from — it folds everything into one
   number — but exactly the right thing to *validate* one against, across a
   wide sweep, without hardware. **First partial climb, 2026-08-03**: the
   dataset's `same_axis` key turns out to isolate a pure hop-count difference,
   and it reproduces the ISA docs' 9-cycle hop on *both* architectures to
   within 4 % — the first external check of the hop model on Blackhole, and the
   thing that unblocked its DRAM term. **Swept in full, 2026-08-03**: see
   ["Rung 2, swept"](#rung-2-swept-8140-measured-points-60-the-model-is-allowed-to-predict).
   The verdict splits — **climbed for latency, failed for bandwidth** — and the
   harness is `tt_sim/perf/noc_dataset_sweep.py`. The bandwidth half is closed
   for DRAM by ["Two queues, not
   one"](#the-contention-dram-is-channel-limited-not-link-limited) and is left
   open, with the arithmetic, for L1.
3. **The §I driver assertions.** Re-run `four/` and `five/` and check reported
   cycle counts against ranges derived from these tables. Cheap, and it catches
   the class of error where a cost is plumbed to the wrong unit.
4. **Captured traces**, if they ever become available. Until then the honest
   claim is order-of-magnitude correctness on stalls, back-pressure and
   contention — not silicon-matching cycle counts.

**Read rungs 1 and 2 for what they are: checks on the NoC and memory path.**
Both are built entirely out of `dram.end_to_end_reference` and
`tm_noc_latencies`, neither of which contains a Tensix instruction. So the five
wired Tensix backend units — which are 21 % of `six`'s cycles and the largest
*sourced* block in either YAML file — have been validated **not at all**, and
there is no rung between 2 and 4 that would change that: tt-metal ships no
measured Tensix compute dataset, only a dispatch one, on a path tt-sim does not
implement. The search is written out in ["There is no rung-3 dataset for Tensix
compute"](#there-is-no-rung-3-dataset-for-tensix-compute-and-that-changes-what-rung-2-climbed-means),
including the one near miss (tt-llk's per-op perf harness, which generates the
right numbers on silicon and commits none of them). "Rung 2 climbed" must not
be read as "the model has been tested".

## The first consumer

The Tensix matrix unit, wired 2026-08-03 as Phase 5 of
[`event-driven-pump.md`](event-driven-pump.md), scoped to one unit on purpose:
to find out what the machinery costs and what the data is worth before either
spreads.

**How a cost reaches a cycle.** `MatrixUnit.__init__` asks
`tt_sim.perf.model.unit_cost_model("MATH", arch)` for a model, which is `None`
unless `TT_SIM_COST_MODEL` is truthy — so with the switch off no YAML is
parsed, `TensixBackendUnit.clock_tick` reads one `None` attribute per
instruction, and nothing else changes. With it on, `clock_tick` asks
`instruction_occupancy(...)` *before* running the handler (the handler's
`ADDR_MOD` step can advance the fidelity phase, and the cost belongs to the
phase the instruction ran at) and hands the answer to Phase 4's
`occupy_for(cycle, cycles)` after. `occupy_for` no-ops at ≤ 1 cycle, so a
one-cycle op costs a dict lookup and nothing else.

**The three judgement calls** the "Using it" section below asks a consumer to
make, made once in `model.py` rather than per call site:

- *No entry means no opinion.* An opcode with no `occupancy`, or with
  `provenance: unknown`, is charged nothing at all — not a plausible-looking 1.
- *A bound is not an equals sign.* `at_least` and `range` are charged at their
  **low end**, so a modelled count is a floor. Over-charging invents
  back-pressure the hardware does not have; under-charging only fails to model
  a stall that was not modelled before. `UnitCostModel.is_exact` keeps the
  distinction reachable for reporting.
- *Derived is not measured.* Nearly every Tensix occupancy is `1 /
  throughput_ipc` rather than a published occupancy column. Those are charged —
  arithmetic on a documented number is the best available — but
  `provenance_of` keeps the rank, and `estimated` / `unknown` are never
  charged.

**Fidelity phases: the number that had to be computed.** This is the one entry
in the MATH table a flat integer cannot express, and getting it right is most
of what this phase was for. The naive reading of the vendor block — LoFi 16,
HiFi2 32, HiFi3 48, HiFi4 64 cycles per tile — is that an `MVMUL` at HiFi4
costs four times one at LoFi. It does not, and charging that would be a
**2.5× over-count**:

- one `MVMUL` is eight rows of SrcB by a 16×16 SrcA, so a 32×32×32 tile matmul
  takes **16 instructions per fidelity phase** (recorded as
  `fidelity_phases.mvmuls_per_tile`, `isa_doc_derived`);
- the fidelity multiplier is carried by the instruction **stream**, not by a
  longer instruction: at HiFi4 the kernel issues one `MVMUL` per phase, four
  per K-face;
- so per-instruction occupancy is `cycles_per_tile / (phases × 16)` = **1
  cycle at every one of the four phases**. Charging the phase index on top
  would bill a HiFi4 tile 1+2+3+4 = 10 phases' worth against the real 4.

That is computed from the YAML rather than hardcoded (a test doubles the
table's numbers and watches the answer follow), and it lands exactly on the ISA
docs' own `throughput_ipc: 1` — two sources that do not cite each other,
agreeing to the cycle. That is rung 1 of the calibration ladder below, and the
first thing this table has actually been *used* to check.

**Measured, on `six`** (Blackhole replay guard: 128³ bf16 matmul at HiFi4,
4096 `MVMUL`s):

| | cost model off | cost model on |
| --- | --- | --- |
| simulated cycles | 27,500 | **27,500** |
| matmul PCC | 0.9982 | **0.9982** |
| FPU cycles charged | — | 4,244 (4096 `MVMUL` + 129 `SETRWC` + 18 `ZEROACC` + 1 `ZEROSRC`, all at 1) |

The headline is that **nothing moved**, and that is the correct answer rather
than a broken one: every matrix op this workload issues is a one-cycle
occupancy, and tt-sim's one-instruction-per-cycle issue behaviour was already
reproducing the ISA docs' 1 IPC by construction. The model's contribution here
is that this is now *asserted from a document* instead of being an accident of
the implementation, and that it is measured: 4,096 `MVMUL` cycles is exactly
the 64 tile-matmuls × 64 cycles/tile that tt-metal's GEMM_FLOPS report predicts
for a HiFi4 128³ matmul.

**Is 27,500 plausible?** The compute-bound floor for this matmul is those 4,096
cycles, so the simulated run is **6.7× off peak** and the FPU is busy 15 % of
it. For one Tensix core streaming every tile from DRAM through a
`matmul_tiles` loop that is a believable order of magnitude — but it should not
be read as a prediction, because the other 85 % is not modelled at all: the
baby RISC-V cores still retire one instruction per cycle with no memory stalls,
the NoC has no per-hop latency, and DRAM answers instantly. The 27,500 is a
structural artefact of the simulator, not a number the cost tables produced.
Closing that gap is what wiring the *next* units is for, and the honest reading
of this phase is that it proved the mechanism and the fidelity arithmetic while
confirming the FPU was never where the missing time was.

**What did not land.** No stride. `TensixTile.next_wake_cycle` fast-rejects on
`soft_active`, so a live tile is ticked every cycle whatever a backend unit's
`busy_until` says; multi-cycle occupancy therefore buys correct back-pressure
today and skipped cycles only once a baby core can name its own next-wake
cycle, which is the deferred Phase 3. Nothing in the matrix unit's table
exercises the multi-cycle path on an in-tree workload either — `SHIFTXB` at 2
cycles is the only `MATH` op above 1 and this backend does not implement it —
so the occupancy path is covered by a stub table in
`matrix_cost_model_test.py` rather than by a replay.

## The next four: SFPU, ThCon, packer, sync — and the config unit that would not go

Wired 2026-08-03, in the order this file ranked the data: the **vector unit
(SFPU)** first, because the ISA docs publish a latency for all 42 of its
opcodes; then **ThCon**, whose table is the only one that is *already* an
occupancy table; then the **packer** and **sync** units, which are fully
tabulated and cheap. The **config unit** was meant to be the fifth and is not
— see ["The unit that would not go"](#the-unit-that-would-not-go). The
**unpacker** and the **mover** were deliberately out of scope from the start.

Wiring cost one line per unit, because the base
`TensixBackendUnit.instruction_occupancy` now *is* the flat table lookup and
the matrix unit's override is the exception rather than the pattern. That is
the right split: a per-opcode constant is the shape of every one of these
tables, and only the FPU's fidelity-scaled ops are a function of unit state.

One thing did have to change in the mechanism. Phase 4's `occupy_for` made an
occupied unit stop *draining* its queue, but left it still *accepting* into it.
That is a reordering bug waiting to happen, because tt-sim's frontend treats an
instruction as issued the moment a unit accepts it: an instruction parked in an
occupied unit retires after the thread's *next* instruction has already run in
a different, idle unit. `TensixBackendUnit.is_occupied` now refuses the issue
instead, which is also the closer reading of the docs ("the issuing thread is
unable to start any further instruction"). It is a no-op with the model off,
because nothing arms `busy_until`. It did not fix the config unit — see below
— but it removes a hazard that would otherwise have been waiting for whichever
unit was wired next.

### The throughput check, per unit

The matrix unit's result was that the fidelity multiplier is carried by the
instruction *count*, so `MVMUL` costs 1 cycle at every phase — landing exactly
on the docs' independent `throughput_ipc: 1`. The same question, asked of each
new unit:

| Unit | Documented throughput | Modelled occupancy | Consistent? |
| --- | --- | --- | --- |
| SFPU | "can only accept one instruction per cycle from the outside world" | 1 cycle, all 42 opcodes | yes — and the docs' 2-cycle *latency* for the arithmetic and LUT ops is **not** occupancy |
| ThCon | "executing at most one instruction at a time, and has no internal pipelining" | 1 / 2 / 3 / 15 by opcode, from the docs' own occupancy column | yes — no throughput figure to contradict, because the table *is* the throughput |
| Packer | "at most one of these instructions can be started per cycle" | 1 cycle for `PACR` | yes, for *issue*; the drain is unmodelled and uncosted |
| Sync | `SEMINIT`/`SEMPOST`/`SEMGET`/`SEMWAIT`/`STALLWAIT` share a 1 IPC budget | 1 cycle throughout | yes; the interesting cost of this unit is wait-gate time, which is not occupancy |
| Config | SETC16 one per thread per cycle (BH: IPC 3); everything else 1 IPC | 1 cycle, except Blackhole's `CFGSHIFTMASK` at 2 | yes — but only after the mechanism was fixed; see ["The unit that would not go"](#the-unit-that-would-not-go) and ["on the second pass"](#the-config-unit-on-the-second-pass) |

The SFPU row is the substantive one, and it is the same *kind* of answer the
matrix unit gave from the opposite direction. The docs give the SFPU two
numbers — a per-instruction latency (2 cycles for `SFPADD` / `SFPMAD` /
`SFPMUL` / `SFPLUT` / `SFPLUTFP32` / `SFPADDI` / `SFPMULI` / `SFPSWAP`, 1 for
the rest) and a unit-level "one instruction per cycle from the outside world"
— and only the second is occupancy. The five sub-units (load, simple, MAD,
round, store) are pipelined, so a 2-cycle `SFPMAD` is 2 cycles until its result
is readable, not 2 cycles during which nothing may issue. Charging the latency
column instead would have roughly doubled the modelled cost of every SFPU-heavy
kernel against a document that never claimed it. `SFPLOADMACRO` is the one
opcode left uncosted: its latency column reads "Complex" because it expands to
up to four more instructions, and it is also the only way to have more than one
sub-unit busy at once, so a single number would misrepresent the unit.

### The first multi-cycle instruction in tt-sim

ThCon is where an instruction finally costs more than a cycle. Its published
occupancies are 1 (`SETDMAREG`), ≥ 2 (`REG2FLOP`, `FLUSHDMA`), "3 or 4" (the
GPR arithmetic ops), ≥ 3 (the loads and stores) and ≥ 15 (`ATCAS`,
`ATINCGETPTR`) — charged at the low end of every bound, so the modelled figure
is a floor. The `busy_until` path Phase 4 installed and Phase 5 left untested
against a real table is now driven by the shipped one: on `six`, ThCon spends
**416 ticks occupied** (144 `STOREREG` and 64 `ADDDMAREG`, two extra cycles
each).

Two scoping notes, both deliberate:

- **It is the base drain that `busy_until` gates.** A ThCon stalled on
  `FLUSHDMA` / `ATCAS` / a Src bank re-evaluates its stall every cycle through
  its own `clock_tick` override, which never reaches `super()`. That polling is
  the unit *waiting on somebody else*, not retiring its own instruction, and
  the ISA docs' ">= 2" for `FLUSHDMA` is explicitly a floor under exactly that
  wait — tt-sim already models the wait functionally.
- **The issuing thread is under-charged relative to the docs**, which say the
  thread cannot start any further instruction *in any unit* until a ThCon
  instruction completes. The model holds the unit, not the thread. That is the
  same "a modelled count is a floor" direction as the bounds policy, and
  tightening it means reaching into the wait gate rather than the cost table.

### The unit that would not go

> **Update, 2026-08-04 (wired).** The unit is now costed — see ["The config
> unit, on the second pass"](#the-config-unit-on-the-second-pass). Both of the
> objections below have been answered: the ordering bug was in
> `TensixBackendUnit.clock_tick` and is fixed, and "every entry is one cycle"
> was never true of Blackhole. This section is kept because the *reasoning* it
> records — never assert a number the docs do not give, however conveniently a
> test then passes — is what made the second pass possible on honest terms.
>
> **Update, 2026-08-04 (rung 3).** `RDCFG`'s occupancy is now **1**, and the
> section below is kept as the record of how the divergence was found rather
> than as a description of the current tables. The ">= 2" turned out to be the
> ISA doc's *latency* copied into an occupancy field: the page does give a
> throughput figure — `RDCFG` shares `RMWCIB`'s "issue at most one of these per
> cycle" — and Blackhole states the same constraint as an explicit IPC group.
> Blackhole silicon measures 0.998 cycles per instruction and 3.0× thread
> scaling, agreeing. So the "charge `RDCFG` 1 cycle" option rejected below was
> rejected on a premise that was false, though the *reasoning* — never assert a
> number the docs do not give, however conveniently a test then passes — was
> right, and would have been right had the premise held. **The divergence is
> not fixed. It is merely no longer reachable from these tables**, which is the
> easiest kind of bug to lose; `config.py`, `costs_test.py` and
> [`tensix-cost-benchmark.md`](tensix-cost-benchmark.md#rdcfg-two-true-facts-about-different-quantities)
> say so explicitly for that reason.

The config unit was supposed to be a five-minute wire-up: every opcode tt-sim
implements is one cycle except `RDCFG`, which the Wormhole page gives ">= 2".
Charging that documented 2 makes the `matmulblock` Blackhole guard **compute
the wrong answer** — 608.0 where the computed golden says 1120.0, on the first
datum of the first output tile.

What the occupancy actually does, instrumented: it delays **five `SETC16`s on
the math thread and four `WRCFG`s on the pack thread by one cycle each**, and
nothing else. The first hypothesis was that this was the frontend reordering a
thread's program — an instruction accepted into an occupied unit's queue
retiring after the thread's next instruction had already run elsewhere — so
`is_occupied` was added to refuse the issue instead. That is a real
improvement and it is kept, but it **does not fix this**: with the issue
refused and the math thread genuinely stalled, `matmulblock` still comes out
wrong.

So what is left is a missing ordering guarantee between a config write and the
units that read it. Both the hardware and the vendor LLK have machinery for
exactly this — the ISA docs' config-write visibility rules, and the LLK's
comment that it pads "to ensure WRCFG instruction has finished, since it takes
2 cycles" — and tt-sim's config writes land instantaneously, so nothing in
tt-sim needs that machinery until something makes a config write late. Nothing
ever had, until now.

Two responses were available and only one of them is honest:

- charge `RDCFG` 1 cycle. The guard passes, the wiring lands, and the file now
  asserts a number the ISA docs do not give — which is the precise failure the
  provenance convention exists to prevent, buried under a passing test;
- leave the unit uncosted, record why, and keep the divergence visible.

The second. `UNWIRED_UNITS` in `tt_sim/perf/costs_test.py` names it, a test in
`backend_cost_model_test.py` pins that the unit is deliberately uncosted and
explains the reasoning, and the comment sits in `config.py` where the next
person to reach for `unit_cost_model("CFG", ...)` will read it first.

Worth stating plainly, because it cuts both ways: this is the cost model
**finding a bug in the simulator** on its second outing, which is an argument
for the exercise. It is also a warning that every multi-cycle occupancy is a
timing perturbation, and tt-sim's functional correctness has never had to
survive one before. ThCon's multi-cycle costs pass all 22 Blackhole guards, but
that is evidence, not a proof, and the next unit wired should expect to have to
make the same argument.

### Measured, with the model on and off

Six Blackhole replay guards, cost model off versus on. `six` is the 128³ bf16
matmul; `sfpumath` and `sfpuchain` are the SFPU-heavy op-coverage guards;
`five` and `optest` mix SFPU with dataflow; `four` is dataflow-bound. Cycle
counts are the guards' own totals, and were re-measured at **one-cycle poll
resolution** (the guards poll the go-signal every 2000 cycles by default, which
would hide a change smaller than that):

| Guard | cycles off | cycles on | exact, off | exact, on | Tensix cycles charged | of the run |
| --- | --- | --- | --- | --- | --- | --- |
| `six` | 27,500 | **27,500** | 27,168 | **27,168** | 5,812 | 21.1 % |
| `sfpumath` | 19,300 | **19,300** | 19,129 | **19,129** | 5,665 | 29.4 % |
| `sfpuchain` | 15,400 | **15,400** | 15,220 | **15,220** | 3,627 | 23.6 % |
| `five` | 14,200 | **14,200** | 10,715 | **10,715** | 1,018 | 7.2 % |
| `optest` | 11,000 | **11,000** | 10,529 | **10,529** | 1,135 | 10.3 % |
| `four` | 107,400 | **107,400** | — | — | 267 | 0.25 % |

Every guard still passes: `six`'s matmul PCC is unmoved at 0.9982,
`sfpumath` / `sfpuchain` are still bit-exact against the ttsim golden, and all
22 Blackhole guards pass with the model on as well as off.

Where the charged cycles go, by unit:

| | `six` | `sfpumath` | `sfpuchain` | `five` | `optest` | `four` |
| --- | --- | --- | --- | --- | --- | --- |
| Matrix (FPU) | 4,244 | 172 | 327 | 205 | 244 | 60 |
| Vector (SFPU) | 3 | **5,354** | **3,075** | 516 | 588 | 3 |
| Scalar (ThCon) | 829 | 37 | 55 | 80 | 64 | 72 |
| Sync | 480 | 54 | 90 | 153 | 143 | 68 |
| Packer | 256 | 48 | 80 | 64 | 96 | 64 |

### What this actually tells us, which is less than it looks

**Not one cycle moved**, on any workload, at one-cycle resolution — including
the one unit with genuine multi-cycle occupancy. That is not a null result from
a broken hook; the occupancy is demonstrably armed: ThCon spends **416 ticks
occupied on `six`**, and on `sfpuchain` and `four` it refuses five issue
attempts outright. It is a statement about where the time in these runs
actually is, and the instrumentation says it plainly:

> Of ThCon's 416 occupied ticks on `six`, **zero had an instruction waiting**
> and **zero issue attempts were refused**.

The unit is not merely under-subscribed, it is never contended at all.
`sfpuchain` and `four` are the only guards where back-pressure happens — five
refused issues each, out of runs of 15 k and 107 k cycles — and even there the
stall is absorbed by slack elsewhere and the total is unchanged to the cycle.
Adding four units takes the modelled fraction of `six` from 15.4 % (the FPU
alone) to 21.1 %, and the SFPU-heavy guards from ~1 % to 24–29 %, so the gap is
narrower. It is not closed, and the remaining 71–99 % is the same three things
it was before: **baby RISC-V cores retiring one instruction per cycle with no
memory stalls, a NoC with no per-hop latency, and DRAM that answers
instantly.** `four` is the clearest evidence — 0.25 % of a 107,400-cycle run is
Tensix work, so its cycle count is a statement about tt-sim's dataflow
modelling and nothing else.

The honest reading: what these five units now provide is a *defensible
attribution* of the Tensix half of a run to a published number, and a
mechanism that will produce correct back-pressure the moment something else in
the model creates enough pressure to need it. A total cycle count is still not
a prediction, and the next unit that would change that is not a Tensix backend
unit at all — it is the RISC-V load/store path, whose table
(`tt_sim/perf/unit_costs.yaml`) is already the most completely sourced section
in the file and has no consumer.

### Which units are not wired, and why

- ~~**Config (`CFG`).**~~ **Wired 2026-08-04** — see ["The config unit, on the
  second pass"](#the-config-unit-on-the-second-pass). It stayed on this list
  through two separate reasons, both of which were eventually answered rather
  than argued away, which is the only reason it is kept here struck through.
- ~~**Unpacker (`UNPACK`).**~~ **Wired 2026-08-06** — see ["Unpacker and mover
  occupancy"](#unpacker-and-mover-occupancy-the-last-two-units-and-the-first-cost-that-is-a-function-of-the-transfer).
  The reason it stayed here was right and is what the wiring had to answer: a
  ≥ 2-cycle address phase (exactly 2 uncompressed, more when compressed) during
  which no thread may start an `UNPACR`, then a data phase whose length is the
  byte count divided by a configured throttle mode (16 / 32 / 64 B per cycle)
  and shared against an 80 B/cycle joint ceiling. The flat lookup the units
  above use is indeed the wrong shape, so the charge is computed at decode from
  `THCON_SEC[n].Throttle_mode` and the transfer size; the joint ceiling is
  still **not** charged, for want of a sourced sharing rule.
- ~~**Mover (`XMOV`).**~~ **Wired 2026-08-06**, in the same instalment. Its
  1-cycle entry is the *issue* cost once the mover is free; the transfer
  duration is bandwidth-derived and lives in `tt_sim/perf/unit_costs.yaml`
  under `mover`, alongside a measured ideal and contended rate. Both halves are
  now charged — the ideal rate only, since nothing sources when contention
  applies.
- **Miscellaneous unit (`TDMA`) and `NONE`.** Every entry is one cycle by one
  blanket sentence in the docs. There is nothing to learn from charging it, and
  the allow-list in `costs_test.py` is more useful if it means "a unit somebody
  reasoned about" than "a unit somebody imported".

`UNWIRED_UNITS` in `tt_sim/perf/costs_test.py` names the one that remains, and
a test asserts every unit in the table with a `backend:` file is on exactly one
of the two lists — so a unit cannot fall off both.

### The config unit, on the second pass

Wired 2026-08-04, and it is the only unit in this file that had to be wired
twice. Both of the reasons it was refused the first time are now answered, and
they were answered in opposite ways — one was a real bug that got fixed, the
other was a claim that turned out to be **false**:

- **"Charging it computes a wrong answer."** True, and it was not the unit's
  fault. `TensixBackendUnit.clock_tick` armed `busy_until` *mid-batch* and
  returned, leaving the rest of a cycle's already-accepted instructions queued
  for later — so a `SETC16` the issuing thread had been told was accepted was
  overtaken by that same thread's later `MVMUL`s. The drain now completes the
  batch and only then holds the unit, which is what both arches' Configuration
  Unit pages describe (occupancy is back-pressure on the *next* instruction;
  each accepted instruction commits at its own latency). The fix belongs to the
  mechanism, not to this unit, and it protects every future multi-cycle entry.
- **"Wiring it would charge nothing anyway."** False on Blackhole.
  `CFGSHIFTMASK` is a 2-cycle occupancy there (`throughput_ipc: 0.5`, "requires
  two cycles in stage 0", `isa_doc_derived`) and the `untilize` guard executes
  it **32 times**. Wormhole's table really is all ones, so this is the one unit
  whose wiring is a genuine timing change on one architecture and a strict
  no-op on the other.

**What the gate said.** `driver/tests/cost_model_gate.py` → **`RESULT: PASS`**,
and — the part worth stating rather than assuming — *every* budget-dependent
guard needed **exactly the poll-budget multiplier it needed before the change**:
`blackhole/dramtop` 1×, `blackhole/offline` 2×, `blackhole/two` 2×,
`wormhole/offline` 4×, and all eleven `wormhole/examples` traces unchanged
(2×/4×, with `six` still at 8×). Nothing moved onto a higher rung of the ladder,
which is a stronger statement than "it passed": the ladder is the gate's only
measure of *how much* slower a run got, and it did not register the change at
all.

**Measured, at one-cycle resolution.** The one guard that executes the one
multi-cycle opcode:

| Blackhole guard | config unit unwired | wired | `CFGSHIFTMASK` executions | occupancy armed | issues refused |
| --- | --- | --- | --- | --- | --- |
| `untilize` | 12,189 | **12,189** | 32 | 32 | **0** |
| `tilize` | 71,929 | **71,929** | 0 | 0 | 0 |

So the hook is demonstrably armed — 32 times — and **not one cycle moves**,
which is the sixth consecutive unit to give that answer and for the same
reason: the unit is never contended. Nothing else in the run is trying to issue
into the config unit in the cycle after a `CFGSHIFTMASK`.

**One test did have to change, and it is worth naming.** The back-pressure is
real, and `tt_sim/pe/tensix/blackhole_ops_test.py` found it: its `_issue`
helper called `clock_tick(0)` for every instruction, so the cycle number never
advanced and a `busy_until` of 2 could never expire — the second
`CFGSHIFTMASK` in a test was refused for ever. That is a defect in a helper
that pretended time did not pass, not a wrong assertion, and the fix is for it
to advance the cycle and retry, which is what the frontend's wait gate does on
a real run. Nothing the test asserts changed. It is also the only evidence in
the tree that this unit's occupancy can refuse an issue at all: on every replay
guard it refuses none.

**What this wiring over-charges**, recorded because it leans the wrong way for
once. `busy_until` is a whole-unit hold, but the config unit's real constraints
are per-IPC-group: Blackhole folds everything except `SETC16` into one shared
group, so a cycle in which `CFGSHIFTMASK` is charged 2 also refuses the next
cycle's `SETC16`, which the hardware would let through. It is bounded —
`CFGSHIFTMASK` is the only entry above one cycle in either arch's table, so it
can only bite in the cycle after one, and only on Blackhole — and fixing it
properly means per-group occupancy, a change to the mechanism rather than to
the table. `config.py` says so at the point of use.

> **Closed, same day** — see ["Occupancy per IPC
> group"](#occupancy-per-ipc-group-the-only-per-group-constraint-anyone-publishes).
> The mechanism now holds one deadline per group, sourced from the "IPC group"
> column of Blackhole's own table. The over-charge cost **nothing**: it never
> refused an issue on any in-tree workload, so `untilize` reads 12,189 cycles
> both before and after.

**The divergence the original bug exposed is still not fixed**, only
unreachable, and that has not changed: nothing in tt-sim orders a config write
against the units that read it, because tt-sim's config writes land instantly
and no table entry now makes one late. `CFGSHIFTMASK` holds the unit only
*behind* its own already-committed write. A future entry that delayed a
`SETC16` or a `WRCFG` would meet the same missing guarantee, and it would look
like a wrong answer rather than a slow one.

### Occupancy per IPC group: the only per-group constraint anyone publishes

Landed 2026-08-04, closing the one thing the wiring above knowingly got wrong.
`busy_until` was a *whole-unit* hold; the config unit's constraints are
*per-group*; the cycle after a `CFGSHIFTMASK` therefore refused a `SETC16` the
hardware issues. This instalment replaces the single deadline with one deadline
per IPC group.

**The grouping is transcribed, not inferred, and that is the whole
methodological point.** A grouping is back-pressure the simulator *stops
applying* — get it wrong and an instruction goes through that the hardware
would have stalled, which is a silent under-charge in a file whose bounds
policy exists to keep every error in the other direction. So the bar was a
published column, and there is exactly one:

> BlackholeA0/TensixTile/TensixCoprocessor/**ConfigurationUnit.md**, whose
> instruction table carries an **`IPC group`** column beside `Latency` and
> `IPC`:
>
> | Instruction | Latency | IPC | IPC group |
> | --- | --- | --- | --- |
> | `SETC16` | 1 cycle | 3 | `ThreadConfig` |
> | `STREAMWRCFG` | ≥ 5 cycles | 1 | `Config` |
> | `WRCFG` | 2 cycles | 1 | `Config` |
> | `CFGSHIFTMASK` | 2 cycles | ½ | `Config` |
> | `RMWCIB` | 1 cycle | 1 | `Config` |
> | `RDCFG` | ≥ 2 cycles | 1 | `Config` |
>
> and states it again in prose: "Everything other than `SETC16` is part of the
> same IPC group, and sustained throughput across the entire group is limited
> to one instruction per cycle (or half an instruction per cycle if
> `CFGSHIFTMASK` is used)."

Two group names, `ThreadConfig` and `Config`, copied verbatim into
`ipc_group:` on each entry — the doc's names, not this file's. The ISA
documentation was not on this machine and is not vendored anywhere under
`/home/nick/projects` (checked); the page was fetched from the upstream
repository, and the quoted sentence was already carried in the table's own
`note` from the original wiring, so the two agree.

**Every other unit was checked and none of them needs this.** The mechanism can
only mis-charge where a unit has *both* a group structure *and* an entry above
one cycle, since `occupy_for` no-ops at 1:

| Unit | Groups published? | Any occupancy > 1? | Verdict |
| --- | --- | --- | --- |
| Config (Blackhole) | **yes** — an `IPC group` column | `CFGSHIFTMASK` 2 | **the one case** |
| Config (Wormhole) | no — prose only, no column | no, all ones | inert; no groups assigned |
| Scalar (ThCon) | no — "executing at most one instruction at a time, and has no internal pipelining" | many (3, 3–4, ≥ 15) | whole-unit is the *stated* behaviour, not an approximation |
| Matrix (FPU) | no — table is Throughput and Latency only | `SHIFTXB` 2 | nothing to group by |
| Sync | no column, but **two throughput classes in prose** — mutex ops "up to three per cycle, provided they refer to different mutexes" against the semaphore ops' shared one | no, all ones | would need groups if anything there ever cost 2; today unobservable |
| SFPU, packer | no | no | nothing to do |

The Sync Unit row is the interesting one: it is the only other unit whose
throughput limit is genuinely not whole-unit, and it is left alone precisely
because assigning it a grouping today would buy nothing and could only be
wrong. The mechanism is ready for it if a number ever lands there.

**Wormhole's config unit deliberately gets no groups**, even though its prose
partitions the unit identically (`SETC16` "issue one per thread per cycle"
against `RMWCIB`/`RDCFG` "issue at most one of these per cycle"). Reading a
grouping out of prose is the inference this section refuses to make, and the
cost of refusing is zero: every Wormhole entry is one cycle, so nothing ever
holds that unit.

**The mechanism**, in `backend_base.py` and shared by every unit:

- `busy_groups: {group: deadline}` replaces the scalar as the authority, and
  `busy_until` becomes the *max* over it — which keeps it meaning exactly what
  it meant for the units that have no groups, and keeps it as the one-attribute
  "is anything armed at all?" fast path that every hot path starts with. With
  the model off nothing is armed and `issueInstruction` is now *cheaper* than
  before: one attribute read, short-circuit, no call.
- `occupy_for(cycle, cycles, group=None)` and `is_occupied(group=None)` take a
  group; `None` is the whole-unit hold unchanged, and is also what a *grouped*
  unit charges an opcode the table gives no group for. Unknown group ⇒ refuse
  more, never less.
- `clock_tick` charges the longest cost **per group** in the batch instead of
  one max over it.

**The batch-drain invariant survives, and paid for it.** The drain is now
**unconditional** — the old "occupied ⇒ return before draining" guard is gone.
That guard was only ever safe because a whole-unit hold made its premise true:
an occupied unit could not have accepted anything, so returning early was
draining an empty queue by another name. With groups a held unit *does* accept
into its free groups, and skipping the drain would leave that work queued for a
later cycle after its thread had been told it was accepted — which is precisely
the reordering that made `matmulblock` print 608.0. Retiring everything the
unit accepted, in the cycle it accepted it for, is the invariant; per-group
occupancy strengthens the case for it rather than weakening it.
`next_wake_cycle` follows: a non-empty queue wins over any deadline, and an
idle one strides to the *soonest* group deadline rather than the latest.

**Measured on `blackhole/untilize`**, the only workload in the tree that
executes `CFGSHIFTMASK` at all, at one-cycle resolution (the guard's own
`PUMP_CHUNK` of 2000 quantises the total, so it is forced to 1):

| `blackhole/untilize` | cycles | `CFGSHIFTMASK` executed | occupancy armed | **issues refused by the hold** |
| --- | --- | --- | --- | --- |
| model off | 11,299 | 32 | 0 | 0 |
| whole-unit occupancy | 12,189 | 32 | 32 | **0** |
| per-group occupancy | **12,189** | 32 | 32 | **0** |

**Nothing measurable, and it is worth saying plainly why rather than filing it
with the other five.** The previous five units each reported "hook armed,
nothing moved" because their opcodes cost one cycle. This one is different: the
hook really does fire, 32 times, and holds a unit for two cycles each time. It
still costs nothing because **the whole-unit hold never once refused an issue**
— a sweep of all 24 Blackhole replay guards under the model finds zero
occupancy refusals in the config unit anywhere, and `untilize` is the only
guard that reaches `CFGSHIFTMASK` at all. There was no over-charge to recover
because nothing was contending for the unit to be over-charged. The 890 cycles
between 11,299 and 12,189 are the occupancy itself, and they are identical
before and after, because both mechanisms hold `Config` for the same two
cycles; all per-group changes is *who else* may enter meanwhile, and on this
workload nobody tries.

So the fix is for the mechanism's correctness, not for a number. That is a
weaker claim than "it made something faster" and a stronger one than the five
before it: the over-charge was real, bounded, and demonstrably free.

**The regression test is the difference, not the behaviour.**
`backend_cost_model_test.test_a_held_ipc_group_still_lets_a_different_group_through`
issues a real `CFGSHIFTMASK` on a real Blackhole backend with the real cost
table, then in the next cycle asserts the `RDCFG` is refused (same group, the
control) and the `SETC16` is **accepted** (different group). Reverted to a
whole-unit `is_occupied`, it fails on exactly that line —
`assert config.issueInstruction(setc16_math_offset(0x200), 1)` — which was
checked by making the reversion and watching it fail there rather than
somewhere incidental. `test_an_ungrouped_unit_still_holds_the_whole_unit` pins
the other half on ThCon, and `model_test` pins by exhaustion that Blackhole's
`CFG` is the *only* `(arch, unit)` pair in either table with groups at all, so
a grouping added anywhere else is a reviewed change.

**What the gate said.** `driver/tests/cost_model_gate.py` → **`RESULT: PASS`**,
38 guards discovered, 32 run under the model, 5 proven, 1 excluded, zero dirty
data READs. And, the number that matters for a mechanism change rather than a
table change, **every poll-budget multiplier is exactly the one the previous
instalment recorded**: `blackhole/dramtop` 1×, `blackhole/offline` 2×,
`blackhole/two` 2×, `wormhole/offline` 4×, and all eleven `wormhole/examples`
traces at 2×/4× with `six` still at 8×. The ladder is the gate's only measure
of *how much* slower a run got, and it did not register this change at all —
consistent with `untilize` reading the same 12,189 cycles either way.

> The gate's classification tripwire fired first, on the `pipestall` guards a
> concurrent workstream is mid-flight on — first a guard file with no `BASELINE`
> entry, then, minutes later, a `blackhole/pipestall` entry with no guard file.
> That is the tripwire working, and those entries are that workstream's to
> settle; the runs above reconciled **only** the `*/pipestall` keys in-process
> rather than editing the gate, leaving discovery, every other guard's
> classification, both stages and the ladder untouched. `pipestall` is inside
> the 38 and passes (its trace is not yet captured, so it skips). The result
> above was reproduced twice, before and after that churn.

**One existing test changed shape**, and it is the same kind of finding as the
`blackhole_ops_test` helper the previous instalment named.
`clock_test.test_backend_unit_occupancy_defers_retire_and_arms_the_pump`
hand-placed an instruction in an occupied unit's queue and asserted it stayed
there — pinning the drain guard rather than any behaviour a unit can reach,
since a unit refuses at issue exactly what it would otherwise have to defer. It
is now
`test_backend_unit_occupancy_refuses_issue_and_arms_the_pump` and asserts the
refusal directly. Nothing it was testing about the pump changed.

## The RISC-V cores, where the cycles finally moved

Wired 2026-08-03, following the sentence the section above ends on. The whole
`riscv` block of `tt_sim/perf/unit_costs.yaml` is `isa_doc`; the consumer is
`tt_sim/pe/rv/cost.py` (the only file outside `tt_sim/perf/` on the RV side
that names the tables at all), reached from `RV32I.clock_tick` through one
attribute read.

### The load-latency table is not an occupancy table

This is the whole design, and it is the same lesson the SFPU taught, arriving
from the opposite direction. The ISA docs' load-latency table is a *latency*
table, and the page says so in the sentence directly under it:

> A latency of N cycles means that N − 1 independent instructions need to
> follow the load if the latency is to be entirely hidden.

So an L1 load does **not** occupy the core for eight cycles. The core is
single-issue and in-order, it keeps issuing after the load, and it stalls only
when something *reads the loaded register* too early. Charging the latency
column as occupancy would have made every Wormhole L1 load eight times as
expensive as the document allows. What the model does instead is a **load-use
interlock**: a scoreboard entry per GPR, `ready[rd] = cycle + latency(region)`,
and an instruction whose `rs1`/`rs2` is not ready yet does not issue. That is
`§I`'s "memory-stall back-pressure on L1 / NoC reads", exactly.

Three things *are* occupancy and are charged as such:

| | Sourced as | Charged |
| --- | --- | --- |
| Sustained stores to L1 | "at most one store every five cycles" | a 5-cycle issue rate limit on L1 stores; Blackhole's coalescing queue modelled with the docs' own predicate (same 16-byte aligned region, start addresses within ±4) |
| Multiply | "occupy the Integer Unit for two cycles, and the next instruction cannot enter the unit until the multiply has finished" | 2 cycles on Wormhole; **0** on Blackhole, where it pipelines into EX1 + EX2 |
| Divide | 6–33 cycles, with two-cycle special cases | 6 at the low end of the range; 2 for ÷0, ÷1 and INT_MIN ÷ −1, which needs the divisor read at issue |

### What is charged nothing, and why each is a gap not an omission

- **Branch mispredicts.** Sourced (a 2-cycle bubble on Wormhole, 4 on
  Blackhole) and *uncountable*: nothing in the ISA docs or in tt-sim describes
  the predictor, so the number of mispredictions is unknowable. Charging every
  taken branch would be a fabrication. `RiscvCostModel.branch_mispredict_observed`
  keeps the number reachable so a report can name the predictor as the gap.
- **The NoC NIU register block** (`0xFFB20000` / `0xFFB30000`). The ">= 7" row
  covers "TDMA / tile control / PIC / NoC **overlay**" — `0xFFB40000`, a
  different block. The NIU registers are what every `noc_async_*_barrier`
  polls, so this is the busiest MMIO load in the tree and it is uncosted;
  charging it the overlay's number would be a guess with a citation stapled to
  it. `RV_UNNAMED_REGIONS` in `tt_sim/perf/model.py` lists this and the three
  other unnamed blocks (MOP expander config, instruction RAM, the Tensix
  instruction push buffers).
  **— Retracted 2026-08-06, and this whole bullet was wrong.** That row's cell
  names "NoC 0 configuration and command" and "NoC 1 configuration and command"
  as their own entries alongside the overlay's, on both architectures. The
  block is charged the row's 7. See ["The NIU register
  block"](#the-niu-register-block-the-number-was-in-the-table-under-the-wrong-key).
- **Blackhole's L1 miss.** Blackhole's table gives L1 two latencies — 2 on an
  L0 d-cache hit, ≥ 8 on a miss — and tt-sim models no d-cache and no hit rate
  is published anywhere. The pair is charged at its **low end**, like every
  other two-ended cost in these files. See the sensitivity check below, which
  is why that choice turns out not to matter much.
- **Sustained-load throughput** (four loads every N − 1 cycles) and
  `max_loads_in_flight`. Both are statements about loads *already in flight*,
  and the interlock already stalls on the dependent read, which is the
  first-order effect of the same hardware.
- **Instruction fetch.** The table gives the fetch period (one 128-bit L1 read
  per four instructions) but no i-cache miss cost.
- **The store queue's depth.** The 5-cycle sustained rate is sourced; the queue
  depth that would let a short burst run ahead of it is not, so the model is
  depth-1 and over-charges a burst of two or three L1 stores relative to
  hardware. The only cost in this section that leans that way, and it is
  visible in `softplus`, the one workload with 1,183 L1 stores.

Every other gap under-charges, which is the direction the file's policy asks
for: a modelled cycle count is a floor.

### Measured: the first simulated cycles this project's cost model has moved

At **one-cycle poll resolution** (the guards poll the go-signal every 2,000 or
5,000 cycles by default, which would hide most of these):

| Blackhole guard | off | on | Δ |
| --- | --- | --- | --- |
| `six` (128³ bf16 matmul) | 27,168 | **29,291** | **+7.81 %** |
| `optest` | 10,529 | 10,652 | +1.17 % |
| `three` | 10,675 | 10,788 | +1.06 % |
| `five` | 10,715 | 10,820 | +0.98 % |
| `sfpumath` | 19,129 | 19,217 | +0.46 % |
| `sfpuchain` | 15,220 | 15,254 | +0.22 % |
| `four` | 103,145 | 103,330 | +0.18 % |
| `two` | 8,200 | 8,200 | 0.00 % |

| Wormhole guard | off | on | Δ |
| --- | --- | --- | --- |
| `softplus` | 17,845 | **19,071** | **+6.87 %** |
| `transpose` | 6,812 | 7,007 | +2.86 % |
| `matmulidx` | 7,113 | 7,307 | +2.73 % |
| `reduce` / `reduceneg` | 7,109 | 7,270 | +2.26 % |
| `matmulblock` | 9,781 | 9,992 | +2.16 % |
| `sfpumath` | 19,290 | 19,492 | +1.05 % |

`six`'s matmul PCC is unmoved at **0.9982**, and all 21 Blackhole *value*
guards pass with the model on.

Wormhole moves more than Blackhole per workload, and the reason is the one
real arch difference in the table: a Wormhole L1 load is ≥ 8 cycles, a
Blackhole one is 2 on an L0 hit.

### The stalls are real; the totals barely care

The instrumentation is blunt about how little of the charged time reaches the
bottom line. On `four`, the five baby cores are charged **76,615 stall cycles**
between them — TRISC1 alone stalls 25,120 of 103,145 cycles, a quarter of its
life — and the run gets **185 cycles longer**. 107,249 of the loads are from
L1.

The obvious suspicion is that the hook is not really armed, so here is the
control: charge Blackhole's L1 loads the **miss** latency (≥ 8) instead of the
hit (2) — a 4× increase on the single most frequent cost in the run — and
measure again.

| | model off | L1 = 2 (hit, shipped) | L1 = 8 (miss) |
| --- | --- | --- | --- |
| `six` | 27,168 | 29,291 | 29,411 |
| `four` | 103,145 | 103,330 | 103,462 |

Quadrupling the dominant latency moves `six` by a further 0.4 % and `four` by
0.13 %. These runs are **not issue-limited**. Their length is set by
cross-core synchronisation — semaphore and CB handshakes between BRISC, NCRISC
and the three TRISCs — and a core made 24 % slower simply spends fewer
iterations spinning in the loop it was already going to wait in. That is also
the reassuring reading of the Blackhole d-cache choice: the number that could
not be sourced turns out to be worth 0.4 %.

`six` is the exception that proves it. It is the only in-tree workload with a
long *dependent* compute phase rather than a handshake-dominated one, and it is
the one that moves 7.8 %.

### What this makes attributable, and what it still does not

Now attributable to a published number, on `six`: 5,812 Tensix cycles (21.1 %,
from the five backend units) plus 21,834 RV stall cycles across five cores, of
which 2,123 reach the total. Still not modelled at all, and still the reason a
total is not a prediction:

- **the NoC has no per-hop latency** — the ~5 / 9 / ~5 model is fully sourced
  and unconsumed, and it is now the largest sourced gap in the file (closed
  later the same day; see ["The NoC hop
  model"](#the-noc-hop-model-where-a-total-changes-shape), which is where the
  21.1 % becomes ~73 %);
- **DRAM answers instantly**, and its access latency is `unknown` in the table
  rather than merely unwired;
- **the NIU register block is uncosted**, which is where a dataflow kernel's
  polling actually lands (closed 2026-08-06, and it was never blocked on a
  number — see ["The NIU register
  block"](#the-niu-register-block-the-number-was-in-the-table-under-the-wrong-key));
- **there is no predictor**, so no mispredict cost.

The honest headline is narrower than "cycle counts moved": what moved is the
part of a run that is *issue-limited*, and almost none of these workloads are.
The RV interlock is the first mechanism in tt-sim that can express a memory
stall at all, and the measurement says the next thing worth building is the
NoC hop model — not because the RV model is wrong, but because the time these
runs spend is spent waiting for other cores, and what those cores are waiting
on is a NoC and a DRAM that answer in zero cycles.

### The second bug a timing perturbation found

The config unit's ordering bug has company. Charging the RV cores turned the
Wormhole `loopback` replay from a mismatch into a **crash**:
`KeyError: 'mutex_index'` in `TensixSyncUnit.issueInstruction`.

The sync unit admits one non-mutex op into a queue that already holds a mutex
op, and then, when the *next* mutex op arrives, walked the whole queue reading
`instr_args["mutex_index"]` off every entry — including the `SEMWAIT` that has
no such field. It is unreachable while every unit retires in the tick it was
issued; delaying a TRISC by a few cycles is enough to produce the mixed queue.
Fixed by skipping entries that do not carry the field (a queued `SEMWAIT`
cannot conflict over a mutex it does not name), with a regression test in
`tt_sim/pe/tensix/sync_mixed_queue_test.py`. The fix cannot change any run that
did not previously crash, and the model-off gates are unmoved.

The same read turned up a **second, separate bug that is deliberately not
fixed here**: that branch tests for `"ATGEM"` while the instruction table calls
the opcode `ATGETM`, so `ATGETM` has never taken the up-to-three-mutex-ops
path. Correcting the name changes issue behaviour with the cost model *off*, so
it needs its own change and its own guard run rather than riding along with
this one.

### Byte-identical replay is a timing pin, and it fires

With the model **on**, 11 of the Wormhole `examples_replay_test` cases and the
Blackhole `offline_replay_test` fail. Every one of them is the same thing, and
it is worth stating precisely rather than tolerating quietly.

Those guards replay a captured wire conversation *verbatim*, including the
number of times the host polled the go-message before giving up and reading the
result back. A slower device has not finished by that poll, so the host reads a
partially written result buffer. The Blackhole per-example *value* guards do
not have this problem because they pump until the go-message flips to DONE,
mirroring tt-metal's own `wait_until_cores_done` — and all 21 of them pass.

Demonstrated rather than asserted: re-running each Wormhole example replay and,
on a mismatch, pumping the device and retrying the *same* read gives

> **11 of 11 examples: exactly one mismatched READ each, every one resolved
> within 2,000 further cycles, zero still wrong.**

So no computed value goes wrong under the model. Nothing was weakened to make
these pass: with `TT_SIM_COST_MODEL` unset the Wormhole replay is still
**126/126 byte-identical** and all 22 Blackhole guards pass. The finding to
carry forward is that **byte-identical replay cannot be the acceptance gate for
a timing model**, and a cost model that ships on by default would need the
Wormhole guards converted to the Blackhole pump-until-done shape first.

*(That conversion has since landed: `examples_replay_test`,
`offline_replay_test` and `one_replay_test`/`replay.py` now pump to
`RUN_MSG_DONE` — or, over the socket, re-poll the spin-polled go-message to its
final recorded value — before asserting every data reply bit-for-bit, so all
three pass with the model on and the only remaining timing-pinned guard is
the Blackhole `offline_replay_test`.)*

That finding is what ["The gate"](#the-gate) below turns into something
runnable, and in doing so **supersedes the re-run-and-pump argument recorded
above**. Pumping after a replay is only valid while the replay is still going;
these traces end by asserting reset on every core, so a top-up applied at the
end proves nothing, and a fixed 2,000-cycle one was in any case calibrated
against this instalment's perturbation alone. The gate re-runs the whole replay
at a larger `cycles_per_poll` instead, which scales with the perturbation — and
gets a stronger answer: **not "the mismatches resolve" but "the mismatches are
not there"**.

### The cost when the model is off

The constraint that mattered most: the RV interpreter is the hottest path in
the simulator and was optimised ~2.3× by removing per-instruction overhead. The
whole model is behind one instance-attribute read and one predicted branch in
`RV32I.clock_tick` (`cost = self.rv_cost; if cost is not None and not
cost.can_issue(...)`), with `rv_cost` set to `None` unless the core was built
with an architecture *and* `TT_SIM_COST_MODEL` is truthy. Cores built outside a
device — `driver/simple`, the ISA unit tests — never opt in at all.

Measured with three frozen full-tree copies (this change, the same tree with
only this change reverted, and a byte-identical copy of that as a control),
interleaved with the order alternated, twelve rounds each. On `four` the change
measures **+0.9 %** by median against a control of *identical code* at
**−0.1 %**, and **+1.8 %** by minimum against the same control at **+4.4 %**;
on `two` — the shorter, more RV-dominated run — it is within **0.2 %** of the
control on both statistics. So: **no cost distinguishable from a control group
of identical code**, upper-bounded around 1 %, which is the only form in which
"no measurable cost" is a claim rather than a hope. Full method and numbers in
[`driver/wormhole/docs/profiling.md`](../../driver/wormhole/docs/profiling.md).
With the model *on*, `four` costs ~8 % more wall clock.

## The NoC hop model, where a total changes shape

Wired 2026-08-03, following the sentence the section above ends on, and it is
the first term in this file that is **not a unit cost at all**: the answer is a
function of the *distance between two endpoints*, not of an opcode. The
consumer is `tt_sim/network/tt_noc.py`; the table supplies two constants and
nothing else.

The whole model:

```
flight_cycles = endpoint_cycles + per_hop_cycles * hops
              = (~5 + ~5)       + 9              * hops
```

`unit_costs.yaml`'s own note on the section said what to do almost verbatim —
"NoC requests gain per-hop latency by scheduling the destination's request
event at `c + hops * latency` instead of `c + 1`" — and that is what happens: a
packet handed to `NUI.transmit` with a delay is parked in `delayed_arrivals`
until its cycle, instead of going into the one-cycle two-list swap. The two
architectures share both constants (Blackhole's NoC page changes the flit width
from 256 to 512 bits and leaves the latencies alone), so **the only per-arch
difference that reaches a flight time is the size of the torus**, and that
comes from `ArchProfile`.

### Hop count is a directional distance, and that is the whole subtlety

Each NoC is a torus routed in one direction only, so the hop count along an
axis is a *forward modular* distance, `(dst - src) % grid`, not `|dst - src|`.
Three consequences, all pinned in `noc_cost_model_test.py`:

- **It wraps rather than turning round.** Getting from `x = 9` to `x = 0` on
  Wormhole is one hop, not nine.
- **It is not symmetric.** A request and its response cover different distances
  on the same NoC. For two tiles differing on both axes the two legs sum to
  exactly `grid_x + grid_y` — 22 hops on Wormhole, 29 on Blackhole — *whatever*
  the distance between them. That constant is what makes the calibration check
  below well defined.
- **NoC 1 is the same formula in the mirrored space.** tt-sim already gives an
  `NUI` on NoC 1 the coord `(grid-1-x, grid-1-y)`; mirroring both endpoints
  negates `dx` and `dy`, which *is* the reversal of routing direction. So
  `hops_noc1(a, b) == hops_noc0(b, a)` falls out with no special case, and the
  same worker reading the same DRAM channel genuinely costs different amounts
  on the two NoCs.

The coordinates fed in are each endpoint's **per-NoC** coord (`NUI.x_coord` /
`y_coord`), never `id_pair` — which is the canonical NoC 0 coord on *both*
NoCs. This is the same distinction the response-routing fix in
`tt_sim/network/tt_noc.py` turns on, and it is why the latency lives in
`NUI.send_to`: the *sender* times the flight, from two endpoint objects, so no
coordinate lookup was reintroduced and nothing had to be carried across
coordinate spaces. A `NullEndpoint` (an unmodelled destination) has no clock of
its own, so both legs are charged to its response — same total, no queue.

The pump cooperates. `NUI.next_wake_cycle` names the arrival cycle of the
earliest packet in flight, so a DRAM tile with a request 130 cycles out
*sleeps* through it and Phase 4's stride jumps the gap, rather than the tile
being ticked 130 times for nothing.

### Measured, at one-cycle poll resolution

`only=noc` is the hop model alone (the RV and Tensix consumers stubbed out);
`full` is every consumer. Both against the model-off baseline.

| Blackhole guard | off | NoC only | full model | Δ NoC alone | Δ full |
| --- | --- | --- | --- | --- | --- |
| `six` (128³ bf16 matmul) | 27,168 | **61,299** | **63,425** | **+125.6 %** | **+133.5 %** |
| `three` | 10,675 | 12,067 | 12,180 | +13.0 % | +14.1 % |
| `five` | 10,715 | 11,865 | 12,068 | +10.7 % | +12.6 % |
| `optest` | 10,529 | 11,491 | 11,660 | +9.1 % | +10.7 % |
| `sfpuchain` | 15,220 | 15,727 | 15,791 | +3.3 % | +3.8 % |
| `sfpumath` | 19,129 | 19,683 | 19,765 | +2.9 % | +3.3 % |
| `four` | 103,145 | 104,527 | 104,712 | +1.3 % | +1.5 % |

| Wormhole guard | off | NoC only | full model | Δ NoC alone | Δ full |
| --- | --- | --- | --- | --- | --- |
| `matmulblock` | 9,781 | 11,577 | 11,784 | +18.4 % | +20.5 % |
| `matmulidx` | 7,113 | 8,169 | 8,375 | +14.8 % | +17.7 % |
| `reduce` / `reduceneg` | 7,109 | 8,117 | 8,327 | +14.2 % | +17.1 % |
| `transpose` | 6,812 | 7,400 | 7,571 | +8.6 % | +11.1 % |
| `softplus` | 17,845 | 18,277 | 19,507 | +2.4 % | +9.3 % |
| `sfpumath` | 19,290 | 19,714 | 19,916 | +2.2 % | +3.2 % |

`six`'s matmul PCC is unmoved at **0.9982**, and every value guard on both
architectures passes with the model on.

The striking number is how *little* traffic it takes. `six` issues **144 NoC
transactions in the whole run** — 128 tile reads and 16 tile writes — and they
carry 38,520 cycles of flight time, of which **34,131 land on the total**.
Nearly all of it is on the critical path, because the dataflow kernel is a
literal `noc_async_read` + `noc_async_read_barrier` per tile: a round trip that
used to cost 2 cycles now costs ~281, and there is nothing to overlap it with.
The three sections above spent five Tensix units and a load-use interlock to
move `six` by 7.8 %; 144 packets moved it by 126 %.

### Is a total plausible now? Partly, and for the first time the question is answerable

For `six`, of 63,425 cycles:

| | cycles | of the run |
| --- | --- | --- |
| NoC flight, charged | 38,520 | 60.7 % |
| Tensix occupancy, charged | 5,812 | 9.2 % |
| RV stalls reaching the total | ~2,100 | 3.3 % |
| **attributed to a published number** | **~46,400** | **~73 %** |

That is up from 21 % (five Tensix units) and ~24 % (plus the RV path). The
compute-bound floor is still 4,096 FPU cycles, so the run is now **15.5× off
peak** rather than 6.6× — further from peak, and that is the *correct*
direction: the earlier number was optimistic because the interconnect was free.
For a single Tensix core streaming 288 KiB of tiles from DRAM through a
serialised read-barrier loop, 144 × ~281 cycles of unavoidable round-trip
latency is a believable dominant term.

Still not a prediction, and the gaps are now few enough to name exactly:

- **DRAM answers instantly.** This is now the single largest missing term and
  the ranked next one. It is `unknown` in the table rather than unwired — see
  below for the first real constraint on it.
- **No congestion, and no bandwidth.** Only `noc.hops` is consumed. A 32-byte
  semaphore poke and an 8 KiB tile read cost the same flight time, and a
  saturated link costs the same as an empty one. The ISA docs quantify neither;
  `noc.congestion` is `provenance: unknown` for exactly this reason.
  (Bandwidth closed later the same day; see ["The NoC bandwidth
  model"](#the-noc-bandwidth-model-where-a-packets-size-starts-to-matter).
  Congestion stays open, and for the same reason it always did.)
- **The NIU register block is still uncosted** on the RV side, which is where
  a dataflow kernel's barrier polling lands. (Closed 2026-08-06; see ["The NIU
  register
  block"](#the-niu-register-block-the-number-was-in-the-table-under-the-wrong-key).)
- **No branch predictor**, so no mispredict cost.

### Rung 1 of the calibration ladder, which this is the first change able to climb

["What calibration would take"](#what-calibration-would-take) lists internal
consistency as the cheapest check: *"a per-hop NoC model plus a DRAM term plus
the RISC-V issue path should reproduce tt-metal's measured 358 cycles for a
Wormhole DRAM transaction. If it does not, the hop model is wrong before any
silicon is involved."* That check needed a hop model. Subtracting the modelled
NoC round trip from each of tt-metal's four measured end-to-end figures:

| Wormhole, measured (vendor) | cycles | − modelled NoC | residual |
| --- | --- | --- | --- |
| L1 local (0 hops) | 56 | 20 | **36** |
| L1 remote write (22 hops) | 256 | 218 | **38** |
| L1 remote read (22 hops) | 259 | 218 | **41** |
| DRAM (22 hops) | 358 | 218 | 140 |

The residual should be the issuing core's own path, which is the *same* in all
four rows — and it is: **36, 38, 41**. The hop model explains the entire
local-versus-remote difference to within 5 cycles, from an ISA-doc per-hop
constant and a vendor end-to-end measurement that do not cite each other. That
is rung 1 passing, and it is the second time these tables have corroborated
themselves across two independent sources (the first was the fidelity-phase
arithmetic landing on the docs' `throughput_ipc: 1`).

The DRAM row is the payoff: 140 − ~38 leaves **roughly 100 cycles of DRAM
device latency on Wormhole**, the first number of any kind for the term the
table calls `unknown`. Deliberately *not* recorded as a cost — it is a vendor
measurement minus a model, which is a calibration target, not a source — but it
is now the obvious thing to test the next consumer against.

**Blackhole does not reproduce it**, and that is worth stating rather than
burying: residuals 68 (local) against 122 / 123 (remote), a 54-cycle
disagreement where Wormhole's is 5. Its round trip is 29 hops rather than 22
and its end-to-end figures were measured against tt-metal's 1.2 GHz assumption,
which the ISA docs contradict at 1.35 GHz (see `clock` in the table). Not
resolved here; recorded.

> **Both halves of that last sentence were wrong, and the correction is
> [two sections
> down](#blocker-2-the-consistency-check-which-was-failing-on-a-row-the-derivation-does-not-use).**
> The clock never entered the measurement. And the 29-hop round trip is right —
> confirmed independently on Blackhole by a measured dataset — so the 54 cycles
> are not in the hop model at all. They are in one row, `l1_local_cycles`,
> which no derivation reads.

### What broke, and what it was

Three things failed under the model. **None of them is a wrong value**, and
proving that took more work than the change itself.

**1. Two unit tests with hardcoded cycle budgets.**
`noc_routing_test.py` pumped `device.run(16)` after issuing a NoC round trip
and `driver/blackhole/bringup.py` pumped `run(30)` — both fine at 2 cycles per
round trip and far too few at ~200. Both now wait on the thing they actually
mean (an empty outstanding-request FIFO; the data arriving) with a generous
cap, so they are routing checks again rather than timing pins.

**2. `blackhole/two`, reported by the gate as "still wrong after 200,000
further cycles — NOT a timing artefact".** It is a timing artefact, and the
reason the proof failed is worth recording because it generalises to every
*trace-pumped* guard. `two` has no pump-until-`DONE` loop: it replays the
captured conversation and lets the bridge pump `cycles_per_poll` per host READ.
The captured tail is `CMD_RESET_ASSERT` **to every core**, then `CMD_EXIT` — so
once the trace is exhausted every core is held in reset and no amount of
further pumping can ever finish the kernel. Pumping *after* the replay is
therefore not a valid benign-mismatch proof for this guard shape. The valid
control is to give the device the cycles a live host would have: at
`cycles_per_poll=200` (2× the recorded 100) `two` computes all 100 elements
correctly under the model. The device needs ~16,400 cycles where the untimed
one needed 8,200; the trace budgets 8,200.

**3. Wormhole `examples:three` and `examples:six`**, reported the same way by
the gate's 2,000-cycle prover. Same cause, same control: at
`cycles_per_poll=400`, **all eleven Wormhole example replays are byte-identical
under the cost model — zero mismatches, not even a late one.** That is a
stronger result than the previous section's "one late READ each, all resolve",
and it settles the question: the cost model changes no computed value anywhere.
`six` alone needs 63 k cycles, so a 2,000-cycle top-up was never going to
resolve it.

**No new simulator bug this time**, unlike the config-write ordering gap and
the sync-unit `KeyError` the previous two perturbations found. One hazard was
looked for specifically and is *not* present: the per-trid outstanding-request
FIFO assumes responses return in issue order, which variable per-destination
latency could violate if a trid were reused across two tiles at different hop
counts. Instrumented across `six`, `nine`, `four`, `eight` and `loopback`:
**zero out-of-order read responses**. It remains a live hazard for a future
multi-destination kernel, not a current bug. (Closed later the same day, by
removing the assumption rather than by watching it hold — see ["The FIFO that
was one multi-destination kernel away from being
wrong"](#the-fifo-that-was-one-multi-destination-kernel-away-from-being-wrong).)

## DRAM access latency, where the number had to be derived rather than read

Wired 2026-08-03, the step the section above ranks next, and the first entry in
these files that **no document publishes**. Every number charged before this
one could be pointed at in a source; this one is arithmetic, and most of the
work was deciding what that makes it. The consumer is `tt_sim/device/tiles.py`;
the table supplies one integer.

### Where it goes, and why that is not the NoC

The hop model times a packet's *flight* and parks it in `delayed_arrivals`
until it lands. DRAM access latency is the different thing that happens next:
what the device on the far end costs **after** the packet has arrived. So it
belongs at the endpoint, and `DRAMEndpointNUI` — a DRAM channel's NIU — holds
an arriving *request* for `service_cycles` before the channel answers it:

```
round trip = flight(there) + dram_service + flight(back)
           = (10 + 9·hops) + 99          + (10 + 9·hops')
```

Holding the request rather than delaying the response is deliberate on both
counts. It is the more faithful ordering — the data really is not read until
the device has got to it, so a write becomes visible late and its ACK later
still — and it is free in machinery: `NUI.transmit` already parks a packet
until its cycle and `DRAMTile.next_wake_cycle` already reports `next_arrival`,
so the tile **sleeps** through the service window rather than being ticked 99
times for nothing. Only the three request actions are charged; a response
arriving at a DRAM NIU would be charged nothing, and a test says so.

Keeping the two terms apart is not tidiness. It is what makes the number
derivable at all, as the next section shows: fold the device time into the
flight and the derivation double-counts the hops it was constructed to cancel.

### The derivation, which is the whole of it

`unit_costs.yaml` called this `unknown` and filed tt-metal's measured
end-to-end figures under `end_to_end_reference` as a *calibration target*,
explicitly not a cost, because 358 cycles folds in the NoC round trip and the
issuing core's path. The hop model then produced the first constraint on it —
358 − 218 (modelled 22 hops) − ~38 (the core path the three L1 rows agree on)
≈ **102**. That is a vendor measurement minus a model, and this file was right
that it is a calibration target rather than a source.

But there is a second route to the same term that never touches the model, and
it is the one shipped:

```
dram.access_latency = end_to_end.dram − end_to_end.l1_remote_read
                    = 358 − 259 = 99 cycles          (Wormhole)
```

Both are tt-metal's measured "initial latency" for the **same transaction
shape** issued by the **same agent**: a remote read from a Tensix core.
Everything the two have in common cancels — the issuing core's path, the NoC
round trip, the response handling — and what is left is the difference between
what a DRAM endpoint costs to service and what an L1 endpoint costs to service.
No modelled quantity appears anywhere in it.

The one assumption is that the two figures were measured at the same NoC
geometry, and on Wormhole that is nearly free: a round trip between two tiles
differing on both axes is exactly `grid_x + grid_y` = 22 hops *whatever* the
distance between them, so two remote measurements cannot differ by hop count
unless one shares an axis with its initiator. The torus constant that made rung
1 of the calibration ladder well defined makes this subtraction well defined
too.

And the two routes agree: **99 against ~102**, from derivations that share only
their source data. The 3-cycle gap is what an L1 endpoint's own service time
should be, which is the term the subtraction folds out — and it is small, as
the L1 bank model implies.

### What it honestly is: a new provenance rank

`isa_doc` > `isa_doc_derived` > `vendor_source` > `estimated` > `unknown` had
nowhere to put this. It is not `isa_doc_derived` (nothing in the ISA docs is
involved), not `vendor_source` (no document prints 99), and calling it
`estimated` — "uncalibrated, a guess however educated" — would be *understating*
it in a way that is its own dishonesty: nothing here is a judgement call, every
input is a published figure and the arithmetic is one subtraction.

So the convention gains one rank:

| `provenance` | Means | Requires |
| --- | --- | --- |
| `vendor_source_derived` | Arithmetic on vendor numbers | `source` + `derivation` |

placed **below** `vendor_source` and **above** `estimated`. It is exactly what
`isa_doc_derived` is to `isa_doc`, one tier down, and it inherits the same
obligation for the same reason: a derived number is only reviewable if the
arithmetic is written out, so `derivation` is mandatory and
`test_derived_entries_show_their_working` now enforces it for both derived
ranks across both files. The full derivation — both routes, the geometry
assumption, the cross-check — is *in the YAML*, not only here.

Two guards keep this from becoming a laundry chute. `PROVENANCE_REQUIRING_
DERIVATION` makes the working mandatory, and
`test_vendor_derived_entries_are_exactly_the_ones_we_expect` pins the list of
entries at the new rank to exactly one, so a second is an edit to a list rather
than a quiet addition — the same discipline that keeps `estimated` at zero
entries.

### Blackhole gets nothing, on purpose

> **Superseded 2026-08-03.** Blackhole now gets 126, and this section is kept
> rather than rewritten because *both* of the two reasons below were chased to
> the source and both dissolved — one into a single stale constant, the other
> into a single bad row that the subtraction does not read. See ["Blackhole's
> DRAM latency, and two blockers that were not about
> Blackhole"](#blackholes-dram-latency-and-two-blockers-that-were-not-about-blackhole).
> The reasoning below was right to withhold the number on the evidence it had;
> what it was wrong about is *where the fault was*, and it said so precisely
> enough that somebody could go and find out. That is the argument for writing
> gaps this way.

The same subtraction on Blackhole's rows gives 529 − 403 = 126, and it is not
recorded. Two independent reasons, either of which would be enough:

- **Its end-to-end set fails the check Wormhole's passes.** Subtracting the
  modelled round trip from all four Wormhole rows leaves 36 / 38 / 41 — the
  same issuing-core path to within 5 cycles. The same subtraction on Blackhole
  leaves **68 for the local row against 122 / 123 for the remote ones**, a
  54-cycle disagreement. Something about that row set is not understood, and
  subtracting two numbers out of a set we have shown we cannot explain is
  precisely the laundering the convention exists to stop. (The *difference*
  derivation is formally robust to a hop-model error, since the NoC cancels —
  but "robust to the error I know about" is not evidence about the one I do
  not.)
- **The clock is unresolved.** tt-metal measured against 1.2 GHz; the ISA docs
  say 1.35. A cycle count converted from a measured time scales with the clock,
  so the Blackhole term is 126 cycles at tt-metal's own assumption and 142 at
  the docs'. This file records conflicts rather than resolving them, and
  recording a number here would mean silently picking one.

The consequence is visible and should be stated plainly rather than
discovered — and it held for exactly one instalment: **every Blackhole guard's
cycle count is unmoved by this change, and that is not because Blackhole's DRAM
is fast.** `six` — the workload this
file has watched throughout, 60.7 % NoC flight and every operand streamed from
DRAM — is a Blackhole guard, and it charges zero DRAM cycles. Either fix closes
the gap: an internally consistent Blackhole end-to-end set, or a resolved
clock.

As a *sensitivity control only*, charging Blackhole the unshipped 126 anyway:

| `blackhole/six` | cycles | |
| --- | --- | --- |
| model off | 27,168 | |
| model on (shipped: no DRAM term) | 67,550 | |
| model on + hypothetical 126 | **83,804** | **+24.1 %** |

144 requests × 126 = 18,144 cycles charged, of which **16,254 reach the total**
— 90 %, the same critical-path story the Wormhole guards tell below. So the
provenance decision is not costing a rounding error: it is costing the largest
single term on the workload this file has watched throughout. That is the
strongest available argument for going and resolving the Blackhole clock, and
the reason the number is recorded here as a control rather than left as an
abstraction. It is *not* an argument for shipping it: the size of a term is not
evidence for its value. (What eventually shipped it was neither the size nor a
change of mind about the evidence, but going and reading two more vendor
sources.)

### Latency, and not the other four things

Named rather than implied, because "DRAM cost model" sounds like it covers
them and ROADMAP §I asks for two of them by name:

- **Bank conflicts.** Not modelled. tt-sim has no DRAM bank model at all and
  the ISA docs publish no bank geometry or conflict cost for the DRAM tile.
- **Refresh windows.** Not modelled, unpublished — and not even this shape,
  being periodic rather than per-request.
- **Occupancy.** A second request arriving during the first's service window is
  *not* queued behind it. The endpoint adds latency and no contention, which is
  the under-charging direction every bound in these files leans.
  (**Half superseded 2026-08-09.** The *channel* is now a queue, at the same
  `ceil(N / 24)` the bandwidth bullet below eventually got consumed as; the
  *device* behind it still is not, because nothing publishes its re-issue
  interval. See ["Endpoint occupancy: the queue was missing, the number was
  already there"](#endpoint-occupancy-the-queue-was-missing-the-number-was-already-there).)
- **Bandwidth.** A 32-byte poke and an 8 KiB burst are charged the same 99.
  DRAM bandwidth is well sourced (24 GB/s per channel, `isa_doc`) and
  deliberately still unconsumed: turning a byte count into cycles is the *same
  physical serialisation* the NoC's per-link bandwidth term describes, so it
  belongs in one place, once, rather than in two that add up.

`DramCostModel` carries `bank_conflicts_modelled` / `refresh_modelled` /
`occupancy_modelled`, all `False`, so a report can name the gaps instead of
letting a reader assume a "DRAM latency" covered them. (Since 2026-08-09
`occupancy_modelled` is `True` on Wormhole and `False` on Blackhole — a per-arch
consequence of whether a channel rate is published — and
`device_occupancy_modelled` is the flag for the half that is still a gap.)

### Measured, at one-cycle poll resolution

Wormhole value guards, model off; the whole model with the DRAM term stubbed
out; and the whole model. The middle column isolates this instalment from
everything else switched on at the time, which includes the NoC bandwidth model
of the section below — landed the same day, so "off" versus "full" would
otherwise conflate the two.

| Wormhole guard | off | model, no DRAM term | full | Δ DRAM alone | DRAM cycles charged |
| --- | --- | --- | --- | --- | --- |
| `matmulblock` | 9,781 | 13,044 | **14,320** | **+9.8 %** | 1,386 (14 requests) |
| `matmulidx` | 7,113 | 8,939 | **9,529** | **+6.6 %** | 693 (7) |
| `reduce` / `reduceneg` | 7,109 | 8,759 | **9,349** | **+6.7 %** | 693 (7) |
| `tilize` | 7,008 | 8,557 | **9,049** | **+5.7 %** | 693 (7) |
| `transpose` | 6,812 | 8,083 | **8,477** | **+4.9 %** | 396 (4) |
| `softplus` | 17,845 | 19,629 | **19,829** | **+1.0 %** | 198 (2) |
| `sfpumath` | 19,290 | 20,179 | **20,367** | **+0.9 %** | 396 (4) |
| `untilize` | 57,806 | 58,613 | **58,847** | **+0.4 %** | 495 (5) |

Every guard still passes with the model on: all eight compute their expected
values, bit-exact where they are bit-exact.

The striking number is not the size of the movement but **how much of the
charged time reaches the total**:

| | charged | reaching the total |
| --- | --- | --- |
| `softplus` | 198 | 200 (101 %) |
| `transpose` | 396 | 394 (99 %) |
| `matmulblock` | 1,386 | 1,276 (92 %) |
| `matmulidx` / `reduce` | 693 | 590 (85 %) |
| `tilize` | 693 | 492 (71 %) |
| `sfpumath` | 396 | 188 (47 %) |
| `untilize` | 495 | 234 (47 %) |

Compare the two instalments before it: of the five Tensix units' charged
cycles, **none** reached any total; of the RISC-V path's 76,615 stall cycles on
`four`, **185** did. Here it is 47–100 %. The reason is the same one the hop
model found: a tt-metal dataflow kernel is `noc_async_read` followed
immediately by `noc_async_read_barrier`, so a DRAM round trip is on the
critical path by construction and there is nothing to overlap it with. (The
101 % on `softplus` is knock-on, not miscounting: a core that arrives late at a
handshake can miss a window and wait longer than the delay that made it late.)

### More plausible, or merely larger?

Both, and it is worth separating them.

**Larger, defensibly.** For `matmulblock`, 14 DRAM requests now cost 1,386
cycles of a 14,320-cycle run — 9.7 %, against 22 hops' worth of flight time
that the hop model already charged them. The proportion is the check that
matters: DRAM device time is now a bit under half of what the NoC round trip
costs for the same transaction (99 against 218), which is the ratio tt-metal's
own measurements assert (358 against 259) and not something this model chose.

**More plausible, narrowly.** What has improved is *attribution*, not
prediction. Before this, a DRAM read and an L1 read cost the same in tt-sim
once they were the same distance away, which is a statement no source supports
and one that two vendor measurements directly contradict. Now the difference
between them is the measured difference. That is a real gain and it is the
whole of the gain.

It is emphatically **not** a validated total, and this instalment weakens the
usual claim rather than strengthening it: every term before this one was
traceable to something a vendor or the ISA docs *printed*. This one is not, and
`vendor_source_derived` exists so that a reader can see that at a glance rather
than having to reconstruct it. Adding a term nobody published does not make a
total a prediction. What would: the same subtraction reproduced on Blackhole
out of a self-consistent measurement set, tt-metal's 740-entry measured NoC
dataset swept against the whole model (rung 2 of the ladder, still unclimbed),
or silicon traces.

### What broke, and what it revealed

Less than the previous three instalments, and the one thing that did is worth
recording because of *where* it was.

**`noc_cost_model_test`'s end-to-end assertion**, which pinned a DRAM read to
`flight(there) + flight(back) + 1`. It is a DRAM read, so it now costs 99 more
— and that is the test doing its job: it is the only assertion in the tree that
pins a full round trip to a formula, and a new term in that formula *should*
break it. Updated to name both models' contributions separately rather than
folding the service time into a leg, which keeps the file honest about which
model owns which cycles.

**No new simulator bug.** Three specific hazards were considered, given that
the two previous perturbations each found a real ordering bug. Two are ruled
out by construction: the delay is constant per endpoint, so packets cannot
overtake each other within a NIU's `delayed_arrivals` (the per-trid
outstanding-request FIFO's ordering assumption is untouched), and the host's
own writes to DRAM go through `device.write` rather than the NoC, so nothing
the bridge does is delayed. The third — a write becoming visible 99 cycles
later than its issuer expects — is real and is *why* the request is held rather
than the response: the ACK is emitted after the write lands, so a kernel that
waits for its write barrier still sees its own data.

**The gate still passes**, and one number in it moved: `examples:six` now needs
an **8×** poll budget where ["The gate"](#the-gate)'s table recorded 4×.
That is the ladder doing exactly what it was built for — it scales with the
perturbation instead of being a fixed cycle top-up that has to be re-tuned per
instalment — and it is a *lower bound at ladder resolution*, not a measurement.
Every other guard's multiplier is unchanged, and every data READ on every trace
is still bit-for-bit.

**Nothing was lowered to get green.** With `TT_SIM_COST_MODEL` unset every
guard is byte-identical: the Wormhole offline replay is 126/126, all 24
Blackhole guards pass, and every model-off cycle count above is exactly the one
this file recorded before the change (9,781 / 7,113 / 7,109 / 6,812 / 19,290 /
17,845 / 27,168).

## The NoC bandwidth model, where a packet's size starts to matter

Wired 2026-08-03, and it closes the gap the hop-model section named first: *"a
32-byte semaphore poke and an 8 KiB tile read cost the same flight time"*. The
consumer is the same file, `tt_sim/network/tt_noc.py`; the table supplies one
more constant, and the interesting part is that the constant is not a latency
and cannot be spent as one.

### The shape: an occupancy of a link, not a cost of a packet

Everything charged before this was a duration attached to a *thing* — an opcode
occupies a unit, a load takes N cycles to be readable, a packet is N cycles in
flight. Bandwidth is not that. A link carries so many bytes per cycle, so what
a large transfer does is **hold the link**, and the two consequences of that
are different from each other:

```
flits      = ceil(payload_bytes / flit_bytes)      # 32 B Wormhole, 64 B Blackhole
arrival    = departure + flight_cycles + flits - 1 # the tail follows the head
port_free  = departure + flits                     # the next packet cannot start
```

- **The tail.** The last flit arrives `flits - 1` cycles after the first. Once,
  **not once per hop** — the NoC is *wormhole*-routed (the architecture is
  named after it), so the head flit propagates and the tail follows a cycle
  behind rather than each router receiving a whole packet before forwarding it.
  Charging the serialisation at every hop would have multiplied an 8 KiB
  transfer's cost by 22 against a document that says the opposite.
- **The port.** The injecting NIU's outbound link is held for the whole
  `flits` cycles, so the *next* packet that NIU sends departs no earlier. This
  is the serialisation half, and it is where a tile read really differs from a
  semaphore poke: not by arriving 255 cycles later, but by making everything
  behind it wait.

Per-**NIU** occupancy, then, and deliberately not per-**link**. The
router-to-router links a packet crosses are also 1 flit/cycle and are also
sourced, but a packet only ever *waits* on one of those behind some other
tile's traffic — which needs an arbitration policy between two senders, which
is the congestion term the ISA docs decline to quantify (`noc.congestion`,
`provenance: unknown`, "can negatively impact latency", no number). The
injection port needs no such policy: a NIU's own packets are already ordered by
the NIU that issued them. So the line drawn here is not "how much work is it"
but **"how far can we go without inventing an arbiter"**, and that is exactly
as far as one endpoint's own bytes.

Two smaller judgement calls, both in the under-charging direction the file's
policy asks for:

- **A header flit is not counted.** A packet is `max(1, ceil(bytes / flit))`
  flits, so a zero-payload packet — a read *request*, a write ACK, an atomic
  whose operand rides in the header — is one flit and no more. The docs' flit
  accounting does not say how many flits a header takes, and a made-up 1 would
  be a number with a citation stapled to it.
- **The payload is what is on the wire, not the transaction size.** A read
  request names 8 KiB and carries none of it; the bytes travel on the response.
  Charging `data_length_bytes` on both legs would have doubled every read.

And one place the model could easily have over-charged and does not: **a
multicast write is injected once**. The hardware fans a single packet out in
the routers; tt-sim models it as N unicasts, so claiming the injection port N
times would invent serialisation the hardware does not have. The rectangle
claims the port once and every copy shares the wait
(`NUI.claim_injection_port`).

### The bandwidth figure was already in the table, twice, and the two agree

`unit_costs.yaml`'s `noc` block records `flit_bits: 256` (512 on Blackhole)
with `throughput_flits_per_cycle: 1`, and separately
`link_bandwidth_gb_per_s: 32`. Those are the same fact written two ways, and
neither cites the other: 256 bits/cycle at the `clock` section's 1 GHz is
exactly 32 GB/s. That agreement is what says the flit rate is safe to spend as
bandwidth rather than being a packet-format detail that happens to live in the
same block, and it is pinned by
`test_the_nocs_two_bandwidth_figures_are_one_fact_and_agree`. It is also the
third time these tables have corroborated themselves across two independent
numbers, after the fidelity arithmetic and the hop model's residuals.

Blackhole's `link_bandwidth_gb_per_s` is `provenance: unknown` and deliberately
*not* derived: 512 bits at 1.35 GHz would be 86.4 GB/s, and nobody publishes
it. Only the flit width is overridden, which is all the model needs.

### Measured, at one-cycle poll resolution

Every run below has the DRAM service model neutralised, so the columns isolate
this change rather than folding in the section above: `hop` is the tree as the
hop-model section left it, `hop + bw` adds bandwidth and nothing else.

| Blackhole guard | off | hop | hop + bw | Δ bw | bytes on the NoC | packets |
| --- | --- | --- | --- | --- | --- | --- |
| `six` (128³ bf16 matmul) | 27,168 | 63,425 | **67,550** | **+6.51 %** | 294,912 | 288 |
| `optest` | 10,529 | 11,660 | **12,097** | **+3.75 %** | 28,672 | 14 |
| `sfpumath` | 19,129 | 19,765 | 19,887 | +0.62 % | 16,384 | 8 |
| `sfpuchain` | 15,220 | 15,791 | 15,866 | +0.47 % | 14,336 | 14 |
| `five` | 10,715 | 12,068 | 12,092 | +0.20 % | 3,072 | 24 |
| `eight` | 8,677 | 9,243 | 9,255 | +0.13 % | 1,200 | 6 |
| `three` | 10,675 | 12,180 | 12,188 | +0.07 % | 3,072 | 24 |
| `nine` (semaphore-heavy) | 10,245 | 12,060 | **12,064** | **+0.03 %** | 2,576 | 40 |
| `four` | 103,145 | 104,712 | 104,717 | +0.005 % | 1,536 | 24 |
| `loopback` | 8,837 | 10,386 | 10,383 | **−0.03 %** | 2,048 | 16 |

`six`'s matmul PCC is unmoved at **0.9982**; all 24 Blackhole guards pass.

### Does it differentiate? That was the whole point, and yes, by 26×

The question this change exists to answer is not "does a total get bigger" but
"do two workloads the hop model treats alike now come apart". Put `six` and
`nine` side by side — a tile-streaming matmul against a semaphore-and-handshake
kernel:

| | `six` | `nine` |
| --- | --- | --- |
| NoC packets | 288 | 40 |
| bytes carried | 294,912 | 2,576 |
| mean payload | 1,024 B | 64 B |
| hop-model flight charged | 38,520 | 4,756 |
| **bandwidth charged** | **4,464** | **24** |
| bandwidth per packet | **15.5 cycles** | **0.6 cycles** |
| effect on the total | +6.51 % | +0.03 % |

Under the hop model alone those two rows were, per packet, *the same number*.
They now differ by **26× per packet** and by **200× at the level of the run**.
`nine` moves 40 packets and is charged 24 cycles for all of them together,
because almost every one is a semaphore poke that fits in a single flit; `six`
moves 288 and is charged 4,464, because 128 of them are 2 KiB tiles that take
32 flits each to push onto a 64-byte Blackhole link. That is the differentiation
the change was for, and it is visible without squinting.

`loopback` is the honest oddity: it comes out **three cycles shorter**. A
timing perturbation moves a cross-core handshake across a poll boundary, and
sometimes that lands the other way. It is a reminder that these totals are set
by when cores meet each other, not by a sum of charged cycles — the same lesson
`four` taught with 76,615 stall cycles buying 185.

### Is the total more plausible, or merely larger?

More plausible, and this is the first term where "plausible" can be checked
against arithmetic anyone can do by hand. `six` moves 288 KiB over links that
carry 64 bytes a cycle, so the transfer time of its data — bytes ÷ link width,
no model needed — is **4,608 cycles**. The model charges **4,464**, a shade
under because request packets are headers with no tail. So the new term is
essentially the DMA time of the traffic the kernel actually issues, arrived at
from a flit width rather than from that division.

What it is *not* is the dominant term, and saying so is the useful half of the
result:

| `six`, of 67,550 cycles | cycles | of the run |
| --- | --- | --- |
| NoC flight (distance) | 38,520 | 57.0 % |
| NoC bandwidth (size) | 4,464 | 6.6 % |
| Tensix occupancy | 5,812 | 8.6 % |
| RV stalls reaching the total | ~2,100 | 3.1 % |
| **attributed to a published number** | **~50,900** | **~75 %** |

A 2 KiB tile read costs ~240 cycles of round-trip latency and 32 cycles of
serialisation. **These kernels are latency-bound by roughly 7×, not
bandwidth-bound**, and the measurement says so plainly: adding bandwidth moves
`six` by 6.5 % where adding the hop model moved it by 126 %. That is a result
about the workloads, not a shortcoming of the term — a kernel that issued its
reads without a barrier between them would spend the same bytes with the
latency overlapped, and *then* the injection port would be what it ran into.
The model can now express that difference; the in-tree kernels do not exercise
it.

So: `six` is now **16.5× off its 4,096-cycle compute floor** rather than
15.5×, and the direction is right for the same reason it was right last time —
the earlier number was optimistic because a byte moved for free. Still not a
prediction. What remains uncharged on the NoC is congestion, and congestion is
the one thing here that no amount of care can source.

### The FIFO that was one multi-destination kernel away from being wrong

The hop-model section recorded a hazard rather than a bug: the per-trid
outstanding-request store was a **FIFO popped from the front**, which is
correct exactly while responses come back in issue order, and it was
instrumented across five guards to find **zero** violations. Bandwidth makes
the assumption weaker, not stronger — a large transfer to one tile now occupies
its link while a small one to another does not — so the choice was between
detecting the violation and removing it.

**Removed.** Every request now carries `seq`, its issuing NIU's own monotonic
number, which whatever answers it echoes back unchanged; the outstanding store
is `{trid: {seq: state}}` and a response looks up the state of *its own*
request. The transaction ID cannot do this job, because kernels reuse one trid
for many in-flight requests deliberately — that is why the store is per-trid in
the first place.

Why removal rather than a check, given that a loud failure is this project's
established answer for a case it cannot handle: because this is a case it
*can* handle, and cheaply. The failure mode being guarded against is the worst
kind — a read response handed the *other* request's L1 address writes the right
bytes to the wrong place, with no exception and no assertion, and the first
symptom is a wrong number in a tensor. A detector converts that into a crash;
matching on the issue number means there is nothing to detect. What *is* raised
is a response no outstanding request accounts for (`NoCResponseError`), because
for that there genuinely is nothing sensible to return.

Three things make this more than an assertion:

- `test_a_response_that_overtakes_an_older_one_still_lands_where_it_belongs`
  **forces** the reordering rather than waiting for it: one worker issues two
  reads under one trid, one to DRAM (19 hops out) and one to its own L1 (0
  hops), and the second is answered while the first is still in flight. Both
  payloads land at their own address. Under the FIFO they did not.
- `NUI.out_of_order_responses` counts it in every run, so "this cannot happen
  today" is a number rather than a belief. Across `six`, `nine`, `four`,
  `eight`, `loopback`, `three`, `five`, `optest`, `sfpumath` and `sfpuchain`,
  with bandwidth on: still **zero**. The hazard is closed by construction, not
  because the workloads changed.
- The injection-port occupancy is itself part of the answer. Without it, a
  one-flit packet issued after a 256-flit one **to the same destination** would
  overtake it — reordering that no NoC performs and that would have been an
  artefact of the model rather than a property of the hardware. With it,
  departures from a NIU are monotonic and so are arrivals at any one
  destination. Only genuinely different destinations can reorder, which is the
  real thing.

The response-routing fix is untouched. Responses still carry `reply_to`, the
issuing endpoint *object*, and still funnel through the single
`NUI.send_response`; the only coordinate lookup left is `resolve_destination`,
consuming a coord that arrived from the kernel in that NoC's own space. `seq`
identifies *which request* a response answers, never *where it goes* — carrying
a coordinate across the NoC-0/NoC-1 spaces is the bug that made 56 of 80
Wormhole workers unroutable, and nothing here reintroduces one.

### What broke

Nothing, and that is worth one line rather than a section. Two unit tests
changed shape because the outstanding-request store is a dict of dicts rather
than a dict of lists (`noc_routing_test`'s `fifo == []` is `not pending`), and
no guard's value moved. With `TT_SIM_COST_MODEL` unset every model-off cycle
count above is exactly the one this file recorded before the change, the
Wormhole offline replay is 126/126 byte-identical, and all 24 Blackhole guards
pass.

## Blackhole's DRAM latency, and two blockers that were not about Blackhole

Landed 2026-08-03, and it is the only instalment so far whose work was
entirely *reading someone else's measurements* — no new mechanism, one number.
The DRAM instalment left the largest named gap in these files —
["Blackhole gets nothing, on purpose"](#blackhole-gets-nothing-on-purpose) —
with a price on it: charging Blackhole the unshipped 126 moved `six` by
**+24.1 %**, the single biggest term on the workload this file has watched
throughout. Two reasons were recorded for not charging it. **Neither survived
contact with the sources, and neither turned out to be about Blackhole's DRAM.**

The number now shipped is `529 − 403 = 126`, at `vendor_source_derived`, from
the same four constants and by the same subtraction as Wormhole's 99.

### Blocker 1: the clock, which was one stale constant

The recorded conflict was that tt-metal assumes 1.2 GHz for Blackhole while
the ISA docs say 1.35 — a 12.5 % swing on a quantity of the same order as the
one being derived. The question that dissolves it is the one the gap was never
asked: **are these measurements in cycles, or in time?**

They are in cycles, and the clock never touches them. In
`get_cycles_for_transaction_size` (`tm_dm_common`) the frequency appears in
exactly one expression:

```cpp
float device_frequency_hz = (arch == tt::ARCH::WORMHOLE_B0) ? 1e9 : 1.2e9;
uint32_t cycles = ceil(num_transactions * transaction_size * device_frequency_hz
                       / (transaction_bw * 1e9));
return {cycles, latency_cyles};
```

`latency_cyles` — which *is* `dram.end_to_end_reference`, the 358/529/56/88/
256/404/259/403 — is assigned from a table of constants above and returned
**unscaled**. The frequency multiplies a *bandwidth* into cycles and nothing
else. A term derived by subtracting two of those constants is therefore
clock-free, and the 12.5 % swing does not exist.

That is enough to unblock the derivation, but the underlying question is worth
settling too, because it recurs:

- **UMD, the layer that talks to the chip**, defines Blackhole
  `AICLK_BUSY_VAL = 1350` and `AICLK_IDLE_VAL = 800`, against Wormhole's
  `1000 / 500` (`umd_bh_impl`). AICLK is DVFS-managed: 1.35 GHz is the busy
  point, 0.8 GHz the idle one. So "the Blackhole clock" is 1.35 GHz under
  load, and the pair is the more useful fact than either number alone.
- **tt-metal does not assume a clock where it matters — it reads one.**
  `Cluster::get_device_aiclk` issues UMD's `GET_AICLK` ARC message
  (`tm_cluster:739`), `Device::get_clock_rate_mhz` is that call, and the
  device profiler takes its `device_core_frequency` from it. The
  data-movement suite that produced every measurement this file cites reports
  a column literally headed **"Latency (cycles)"**, from the RISCV wall clock,
  and converts to GB/s using the *device-logged* frequency — falling back to a
  hardcoded **1.35** for Blackhole when a run did not log one
  (`tm_dm_constants:15-19`, `tm_dm_stats:227-230`).
- **1.2 appears once in the whole tree.** Everywhere else that names a
  Blackhole frequency says 1350: the fabric test infra, the ttnn benchmark's
  `device_freq` (with a comment saying so), the distributed socket analysis'
  `CYCLES_PER_US`.

So: not a vendor contradiction, and not something to live with. One stale
constant, in one function, on a path that never touches a latency. Recorded in
the `clock` block so nobody re-litigates it — including the DVFS pair, because
`800` is the number a reader will otherwise hit and misread.

### Blocker 2: the consistency check, which was failing on a row the derivation does not use

The stronger objection was that Blackhole's end-to-end rows fail the check
Wormhole's pass. Subtracting the modelled NoC round trip:

| | measured | − modelled NoC | residual |
| --- | --- | --- | --- |
| Wormhole local / write / read | 56 / 256 / 259 | 20 / 218 / 218 | **36 / 38 / 41** |
| Blackhole local / write / read | 88 / 404 / 403 | 20 / 281 / 281 | **68 / 123 / 122** |

The residual should be the issuing core's own path and therefore the same in
every row. Wormhole's agree to 5 cycles; Blackhole's disagree by 54. The
previous instalment's reading was that *something about that row set is not
understood*, and that subtracting two numbers out of a set we cannot explain
is the laundering the provenance convention exists to stop. That reading was
right to be cautious and wrong about where the fault was.

**The hop model is not what is wrong, and there is now a measurement that says
so on both architectures.** tt-metal ships `tm_noc_latencies`, 740 rows of
measured end-to-end latency in cycles, and every row is keyed on `same_axis`.
The harness (`tm_noc_estimator_test:191-193`) picks the subordinate core as
logical `{0,1}` when `same_axis` is set and `{1,1}` when it is not — so the
same-axis pair shares a NoC column and the different-axis pair does not. On a
directional torus their *round trips* are exactly `grid_y` and
`grid_x + grid_y` hops, **whatever the actual coordinates**, so the measured
difference between them is `grid_x` hops of pure router-to-router latency with
the endpoints, the DRAM and the issuing core's path all cancelled out:

| | grid_x | measured Δ (64 B, 1 txn, read) | implied cycles/hop |
| --- | --- | --- | --- |
| Wormhole | 10 | 293 − 205 = **88** | **8.80** |
| Blackhole | 17 | 373 − 226 = **147** | **8.65** |

Across all 11 transaction sizes and the read / write / stateful variants, the
eight series give 8.3–9.4, mean **9.0**. The ISA docs' 9-cycle hop is
reproduced on **both** architectures, to within 4 %, from a dataset that never
mentions hops and does not cite the docs. That is the fourth time these tables
have corroborated themselves across two independent sources, it is the first
external check of the hop model on Blackhole, and it is a partial climb of
rung 2 of the calibration ladder — which had never been climbed at all.

With the 29-hop round trip vindicated, the 54 cycles have to be somewhere
else, and the same dataset localises them. `tm_noc_latencies` is a different
campaign by the same suite and has **no local row**; taking its remote L1 read
instead, the residual after the modelled round trip is **75 on Wormhole and 92
on Blackhole** — a 17-cycle arch difference for what is the same kernel code
on a differently-pipelined core, where `tm_dm_common`'s rows imply 81. So:

> The anomaly is confined to a single constant, that constant is
> `l1_local_cycles` on Blackhole, and the derivation subtracts
> `dram_cycles − l1_remote_read_cycles` — **neither of which is it.**

Why 88 is 88 is still not understood, and it is now recorded against
`end_to_end_reference`, which is the entry that actually has the problem,
rather than against `access_latency`, which does not. A calibration target
with one bad row is a calibration target with one bad row; it is not a reason
to refuse a subtraction that never reads that row.

### The derivation, and its cross-check

Identical in shape to Wormhole's, deliberately — a Blackhole figure arrived at
some other way would not be commensurable with 99:

```
dram.access_latency = end_to_end.dram − end_to_end.l1_remote_read
                    = 529 − 403 = 126 cycles         (Blackhole)
                    = 358 − 259 =  99 cycles         (Wormhole, unchanged)
```

Both inputs are the same transaction shape from the same agent, so the core
path, the NoC round trip and the response handling cancel and what is left is
(DRAM endpoint service) − (L1 endpoint service). The geometry assumption is as
cheap as Wormhole's for the same torus reason: a different-axis round trip is
`17 + 12 = 29` hops whatever the distance.

The cross-check is the second dataset, which carries L1 **and** `DRAM_SHARDED`
rows for the same pattern, transaction count and axis relationship, per
transaction size — so the same subtraction can be done again from data that
shares only a test suite with `tm_dm_common`:

| | 64 B | over 64 B – 2 KiB | mean | shipped |
| --- | --- | --- | --- | --- |
| Wormhole | 397 − 293 = **104** | 104 / 104 / 104 / 112 / 104 / 112 | 107 | 99 (+8 %) |
| Blackhole | 481 − 373 = **108** | 108 / 88 / 128 / 127 / 152 / 124 | 121 | 126 (−4 %) |

Two campaigns agree on **both** arches at the same level, which is the standard
the Wormhole entry's two routes were held to. Only the six smallest sizes are
usable: above 2 KiB the DRAM rows saturate and the comparison stops being
latency-dominated, and at more than one transaction in flight the two memory
types have different bandwidth ceilings and the difference goes *negative*.

The honest caveat, which is in the entry: **Blackhole's number is coarser than
Wormhole's.** Wormhole's cross-check series is flat to 4 cycles; Blackhole's
scatters over 88–152, and its L1 row is non-monotonic in transaction size by
21 cycles where Wormhole's is flat. 126 and 99 are the same arithmetic, not the
same confidence.

One thing deliberately **not** modelled, and it is new information the
cross-check turned up: Blackhole's rows say a DRAM *write* costs far less over
L1 than a read does — ~20–30 cycles against ~108–152 at one transaction. That
is physically unsurprising (a write is answered when accepted, a read when the
array has been read), and it means this flat term over-charges a Blackhole DRAM
write. Splitting the cost by request action would rest on one arch's data —
Wormhole's DRAM-write row has a visible outlier in its 64 B column — so it is
named rather than done.

### Measured

Every Blackhole guard below at **one-cycle poll resolution**. The middle
column is the whole model with this term stubbed out, so the Δ isolates this
instalment from the four before it:

| Blackhole guard | off | model, no DRAM term | full | Δ DRAM alone |
| --- | --- | --- | --- | --- |
| `six` (128³ bf16 matmul) | 27,168 | 67,550 | **83,804** | **+24.06 %** |
| `optest` | 10,529 | 12,097 | **12,978** | **+7.28 %** |
| `loopback` | 8,837 | 10,383 | **11,017** | +6.11 % |
| `nine` | 10,245 | 12,064 | **12,692** | +5.21 % |
| `three` | 10,675 | 12,188 | **12,817** | +5.16 % |
| `five` | 10,715 | 12,092 | **12,716** | +5.16 % |
| `eight` | 8,677 | 9,255 | **9,507** | +2.72 % |
| `sfpuchain` | 15,220 | 15,866 | **16,118** | +1.59 % |
| `sfpumath` | 19,129 | 19,887 | **20,138** | +1.26 % |
| `four` | 103,145 | 104,717 | **105,346** | +0.60 % |

`six`'s matmul PCC is unmoved at **0.9982** and every guard still computes its
expected value. On `six`, 144 requests × 126 = 18,144 cycles charged of which
**16,254 reach the total** — 90 %, the same critical-path story the Wormhole
guards told, and for the same reason: a tt-metal dataflow kernel is
`noc_async_read` followed immediately by `noc_async_read_barrier`.

Two of those numbers are worth a second look. `six` at **83,804** is the figure
the previous instalment predicted as a *sensitivity control*, to the cycle — the
control stubbed a value in, this is the shipped table doing it, so the two
agreeing is the plumbing checking out rather than a new fact. And `four`, the
guard that has been the file's standing reminder that these runs are set by
cross-core handshakes rather than by charged cycles, moves **0.60 %** on a term
worth 24 % elsewhere. Nothing about that has changed.

Wormhole is untouched: it already had its number, and every Wormhole cycle
count in this file is exactly what it was.

**The gate passes and no multiplier moved.** `python3 -m
driver.tests.cost_model_gate` is `RESULT: PASS` — 36 guards discovered, 30 run
under the model, 6 proven, 614 unit tests under the model, **zero dirty data
READs on any trace**. The three Blackhole budget-dependent guards need exactly
the budgets they needed before this term existed (`dramtop` 1×, `two` 2×,
`offline` 2×), which is the useful negative result: a term worth 24 % on `six`
is worth nothing at all on the guards whose kernels finish inside the recorded
poll budget either way. And `--model-off` is `RESULT: PASS` with everything
bit-for-bit at 1×: the Wormhole offline replay is **126/126**, the Blackhole
one 220/220, and every model-off cycle count in this file is unchanged.

## Rung 2, swept: 8,140 measured points, 60 the model is allowed to predict

Landed 2026-08-03, immediately after the instalment above, and it is the first
change in this file that **adds no cost, consumes no table and moves no
cycle**. It adds one thing: a harness that asks the assembled model to predict
somebody else's measurements and prints how badly it does.
`tt_sim/perf/noc_dataset_sweep.py` (runnable, `--arch`, degrades gracefully
without tt-metal) and `tt_sim/perf/noc_dataset_sweep_test.py` (29 tests, the
ones needing the dataset skipped without it).

The section above climbed rung 2 *partially*, by **differencing**: the
dataset's `same_axis` key isolates a pure hop-count difference, so subtracting
one row from another cancels the endpoints, the DRAM, the flit serialisation
and the issuing core's path, and what is left is the 9-cycle hop. That is a
strong check of one term and it is deliberately blind to every other. This
instalment does the complementary thing and keeps them all: predict the
**absolute** latency of a transaction, with every term in, and see what is
left over.

### What the dataset actually is, which is not quite what this file said

Checked rather than trusted, because the description here was written from a
quick look and three of its details were wrong or missing:

- **8,140 points, not 740.** 740 entries × 11 transaction sizes (64 B – 64 KiB),
  each a **device cycle count** taken by the on-device profiler — the duration
  of a `DeviceZoneScopedN` region that wraps *the whole issue loop plus the
  closing barrier* in the data-movement suite's estimator kernels. So a row
  contains the issuing RISC-V core's NIU register stores and its barrier
  polling as well as the transaction.
- **`num_transactions` in a key is transactions *per barrier*, not the total.**
  `noc_estimator.cpp` builds its lookup key with
  `.num_transactions = params.num_transactions_per_barrier` and multiplies the
  result by `ceil(total / per_barrier)` afterwards. Reading it as a total
  inverts the meaning of every N > 1 row.
- **The pattern enum in the consumer is not the one in the producer.**
  `types.hpp` numbers `ONE_FROM_ONE = 0`; the test that generated the data
  numbers `ONE_TO_ONE = 0`. They are reconciled only because the CSV carries
  the pattern's *name* and `csv_reader.cpp` maps the string. So YAML
  `pattern: 0` is a **read**. Nothing says so anywhere.
- **`noc_index` is 0 on all 740 entries and means nothing.** The kernels log
  the column as `"NoC Index"`; `csv_reader.cpp` looks for `"NOC index"`. The
  lookup never matches, so every point takes the default and NoC 0 and NoC 1
  rows collide on one key. Harmless here — a *round trip* on a directional
  torus is the same number of hops on either NoC — but it is a parsing bug in
  the vendor tree and a reader should not take the field at face value.

The clock question the previous instalment settled applies here too and more
simply: these are cycles from the device's own counter, so no frequency
converts them.

### The exclusion criteria, written down before any residual existed

This is the part that decides whether the exercise is worth anything. Dropping
rows because they *disagree* is fitting; dropping them because the model has no
term for them is validating, and the only defence against confusing the two is
to write the rules down first and report their cost. They are in
`_exclusions()`, each with its reason in prose next to its predicate, and a
test asserts every rule carries one.

| Rule | Why the model may not be asked | Removes | Leaves |
| --- | --- | --- | --- |
| `arch` | swept one at a time | 390 / 350 | 350 / 390 |
| `mechanism != UNICAST` | tt-sim models a multicast as N unicasts sharing one injection port; router fan-out and the arbitration between copies are not modelled | 150 / 170 | 200 / 220 |
| `pattern ∉ {ONE_FROM_ONE, ONE_TO_ONE}` | every other pattern has ≥ 2 concurrent initiators or targets sharing links, i.e. **congestion** (`noc.congestion`, `provenance: unknown`), and sweeps grids whose per-core distances the dataset does not record | 150 / 170 | 50 / 50 |
| `num_transactions per barrier ≠ 1` | N > 1 is a pipelined burst, set by the initiator's outstanding-transaction credits and the kernel loop's per-transaction issue cost. Neither is modelled | 40 | 10 |
| `stateful` | an issue-side optimisation; tt-sim charges no NoC register configuration cost, so it would predict both identical by construction | 4 | 6 |
| `loopback` | a multicast-linked feature; nothing expresses it | 0 | 6 |
| `memory == DRAM_INTERLEAVED` | pages round-robin over 12 channels at different distances; tt-sim has one DRAM tile and no interleaving model | 0 | 6 |

Then one **data-provenance** exclusion, which is about the file rather than the
model: a DRAM row's 16 / 32 / 64 KiB columns are not measurements.
`dram_accessor_sweep` caps at 256 pages of 32 bytes = 8 KiB, and
`data_extractor.cpp` pads every standard size above the largest measured one by
repeating the last value — visible in the raw data as a flat tail (`733, 733,
733`). Six points per arch.

**Retained: 6 entries, 60 points, on each architecture.** That is **0.7 %** of
the dataset, and the number is worth staring at rather than hurrying past: it
is the honest price of a model with no congestion term, no multicast fan-out
and no issue-loop model. 99.3 % of the most detailed public NoC measurement
that exists is asking questions tt-sim cannot be asked yet.

### What the residual is supposed to be, stated before looking

The model predicts the NoC round trip and the endpoint's service time. The
measurement additionally contains the issuing core's own path, which tt-sim
does not charge for at all — the NIU register block was on
`RV_UNNAMED_REGIONS` at the time, and this harness drives the initiator's
registers directly rather than running a kernel. (The block is charged from
2026-08-06, which moves nothing in this sweep: it drives the registers
directly and so has no core path to charge either way.) So:

> The residual is **not** expected to be zero. It is expected to be a
> **constant**: one issuing-core path, the same in every row, independent of
> transaction size, of geometry, of memory type and of direction. Its *value*
> is not predicted; its *constancy* is exactly what the assembled model claims.
> Structure along any of those four axes is model error, and names which term.

The prediction is not a formula. `predict_cycles` builds a real device with
`TT_SIM_COST_MODEL=1`, drives one transaction through the initiator's NoC
command registers the way tt-metal firmware does, and pumps until
`NIU_MST_REQS_OUTSTANDING` returns to zero — which is precisely what
`noc_async_read_barrier` waits for and therefore precisely where the measured
DeviceZone ends. Hop model, injection port, burst splitting at
`noc_max_burst_size` and the DRAM endpoint's service window are all exercised
through their real consumers. A test pins the answer to the closed-form
composition of the published terms, so the harness cannot quietly stop
exercising one and still print a tidy table.

### Wormhole: measured, predicted, left over

| row | 64 | 256 | 1 K | 4 K | 8 K | 16 K | 32 K | 64 K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **L1 read diff-axis** measured | 293 | 301 | 333 | 437 | 581 | 880 | 1453 | 2589 |
| predicted | 220 | 226 | 250 | 346 | 474 | 730 | 1242 | 2266 |
| residual | **73** | **75** | **83** | 91 | 107 | 150 | 211 | 323 |
| **L1 read same-axis** measured | 205 | 213 | 245 | 357 | 493 | 792 | 1381 | 2525 |
| predicted | 130 | 136 | 160 | 256 | 384 | 640 | 1152 | 2176 |
| residual | **75** | **77** | **85** | 101 | 109 | 152 | 229 | 349 |
| **L1 write diff-axis** residual | 73 | 83 | 84 | 118 | 139 | 175 | 211 | 235 |
| **L1 write same-axis** residual | 80 | 79 | 88 | 95 | 141 | 165 | 187 | 255 |
| **DRAM read** measured | 397 | 405 | 437 | 565 | 733 | — | — | — |
| predicted | 319 | 325 | 349 | 445 | 573 | — | — | — |
| residual | **78** | **80** | **88** | 120 | 160 | — | — | — |
| **DRAM write** residual | 202 | 111 | 124 | 156 | 195 | — | — | — |

Distribution over all 60 points: min **73**, p25 83, median 94, p75 150, max
349 cycles; 9.4 % – 38.8 % of the measurement. **Every residual is positive**,
which is the first thing to check and the property the bounds policy has
claimed all along: a modelled cycle count is a floor, never an over-charge.

Three of the four axes are flat, and they are flat where it matters.

**Geometry — the hop term.** Median residual at ≤ 512 B: **82** different-axis,
**78** same-axis. The two geometries differ by 90 cycles of modelled flight
(218 against 128, a whole `grid_x` ring on a directional torus) and by 88
cycles of measurement, and the model absorbs the difference to within **4
cycles**. Same check on Blackhole, where the effect is 153 cycles: 76 against
94, within 18. This is the third independent corroboration of the 9-cycle hop
and the first one that keeps the endpoint terms in rather than cancelling them.

**Memory type — the DRAM access-latency term.** At ≤ 512 B the *read* rows give
DRAM 82 against L1 77 — a **5-cycle** agreement. Put the other way: the dataset
says a Wormhole DRAM read costs 397 − 293 = **104** cycles more than an L1
read, and the table charges **99**, derived from `tm_dm_common`'s completely
separate 358 − 259. Two vendor campaigns, one number, 5 % apart. That is the
strongest out-of-sample result in this instalment, and it is a *validation* of
the file's only `vendor_source_derived` rank rather than another derivation
from it.

**Direction.** 78 read against 83 write at small sizes, and all of the
difference is one row — the DRAM-write series, whose 64 B column (521, higher
than its own 128 B column of 425) is a visible outlier in a monotone sweep. It
is reported, not excluded: excluding a point because it is inconvenient is the
thing this section is organised to avoid.

### Where it fails: the bandwidth term is optimistic, by about 10 %

The size axis is not flat, and it is the whole of the failure. Least squares on
the residual:

| row | intercept | slope |
| --- | --- | --- |
| L1 read diff-axis | 77 | **+3.94 cycles/KiB** |
| L1 read same-axis | 80 | **+4.30** |
| L1 write diff-axis | 94 | +2.65 |
| L1 write same-axis | 89 | +2.83 |
| DRAM read | 79 | **+10.03** |
| DRAM write | 129 | +7.47 |

A constant intercept — 77 to 94, the issuing-core path, exactly as predicted —
with a slope on top of it. A slope against transaction size is a bandwidth
error by construction, and reading it as a rate:

| Wormhole, large transfers | measured | modelled | |
| --- | --- | --- | --- |
| L1 read | 28.5 B/cycle | 32 | **89 %** |
| L1 write | 29.5 – 30.2 | 32 | 92 – 94 % |
| DRAM read / write | 24.4 / 24.5 | 32 | 76 / 77 % |

The 40 entries the primary sweep excluded turn out to be worth something after
all — just not what it wanted from them. Their *latency* cannot be predicted
(the issue loop is unmodelled), but a **bandwidth ceiling is one-sided**: the
model says one NIU's injection link carries `flit_bits / 8` bytes per cycle and
no more, and a measured single-initiator transfer that beat it would falsify
the constant outright whatever the issue path costs. So the harness runs that
check over every unicast one-to-one / one-from-one entry, dataset only, no
simulation. At 256 transactions × 64 KiB the measured rate has asymptoted:

| | Wormhole (32 B/cycle) | Blackhole (64 B/cycle) |
| --- | --- | --- |
| sustained read | 28.70 (**89.7 %**) | 61.61 (**96.3 %**) |
| sustained write | 31.18 (**97.4 %**) | 62.18 (**97.2 %**) |

**Nothing anywhere in 8,140 points exceeds the ceiling**, so `flit_bits` is a
true bound and the model stays a floor. But a Wormhole *read* plateaus at 90 %
of it, and that 10 % is real: it is the L1 read ports on one side and the write
ports on the other, both of which the ISA docs give as exactly 256 bits per
cycle per NoC — the same 32 B/cycle as the link, with no headroom, so any
contention at all comes straight off the top. That is the congestion term, and
it is `provenance: unknown` for the same reason it always was.

**Nothing is being tuned to close this.** A 0.897 efficiency factor would be an
`estimated` entry with a measurement stapled to it, in a file whose central
claim is that it has none — and it would be fitted to precisely the 60 points
it was then validated against, which is the failure mode this whole section is
arranged to prevent. The finding is recorded and the term stays as published.

**Chased 2026-08-04 and it is not sourceable.** The NoC's own framing overhead
is documented, and it is one header flit per packet — 0.125 cycles/KiB, i.e.
**3 % of this read gap and 6 % of the write gap**, and direction-symmetric, so
it explains none of the asymmetry that made it the prime suspect. See ["Is the
~10 % L1 read shortfall
sourceable?"](#is-the-10--l1-read-shortfall-sourceable) for the arithmetic and
for the three other candidates the docs positively rule out.

### The DRAM slope is a different animal, and the fix is already in the file

DRAM's 10 cycles/KiB is 2.5× L1's, and its sustained rate is not 90 % of the
NoC link — it is 24.4 B/cycle. That number is not a fraction of anything. It is
`dram.bandwidth.per_channel_gb_per_s: **24**`, `isa_doc`, straight off the
Wormhole DRAM tile page, at the `clock` section's 1 GHz. The measurement lands
on it to within **2 %**.

So the DRAM rows are not NoC-limited at all: they are **channel**-limited, by a
figure that has been sitting in `unit_costs.yaml` at the strongest provenance
rank in the file since the tables were written, marked *deliberately
unconsumed* on the argument that size dependence is "the same physical
serialisation the NoC's per-link bandwidth term describes, so it belongs in one
place, once". This dataset says that argument is wrong: they are two different
queues at two different rates, 32 B/cycle on the link and 24 B/cycle at the
channel, and a DRAM transfer is limited by the slower one. Wiring
`dram.bandwidth` as a second serialisation at the DRAM endpoint is now the
best-evidenced next term in this file, and it needs no new number.

**Landed 2026-08-04**, as the excess of the channel's serialisation over the
link's, charged once per request at the DRAM endpoint — see ["Two queues, not
one"](#two-queues-not-one-what-rung-2-handed-over-and-what-it-did-not), where
the contention above is evaluated rather than assumed and this slope goes to
**−0.65 cycles/KiB**.

### Blackhole, and the first place the model over-charges anything

Same sweep, same 60 points, and the shape is the same but tighter — the flit is
64 B, so serialisation is half of Wormhole's and the size axis nearly vanishes
(0.70 – 0.90 cycles/KiB against 2.65 – 4.30). L1 residuals sit at 76 – 152 with
an intercept of 80 – 99, and the geometry check holds across a 153-cycle
effect.

The DRAM rows do not, and this is the finding:

| Blackhole DRAM write | 64 | 128 | 256 | 512 | 1 K | 2 K | 4 K | 8 K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measured | 380 | 388 | 388 | 391 | 396 | 422 | 454 | 523 |
| predicted | 408 | 409 | 411 | 415 | 423 | 439 | 471 | 535 |
| residual | **−28** | −21 | −23 | −24 | −27 | −17 | −17 | **−12** |

**Eight negative residuals — the first anywhere in this file.** The model
over-charges a Blackhole DRAM write by 12 to 28 cycles, which means it invents
back-pressure the hardware does not have, the one direction every bound in
these tables is chosen to avoid.

It is not a surprise and it is not a new fact: the instalment immediately above
found the same asymmetry in the same dataset and wrote it into the entry's own
note — a write is answered when the endpoint accepts it, a read when the array
has been read, so a flat term derived from the *read* figures is too large for
a write. What this sweep adds is that the over-charge is now **measured, signed
and bounded** rather than named, and pinned in `KNOWN_OVER_CHARGED` in the test
so that a second one appearing is a change somebody has to make deliberately.
Splitting `access_latency` by request action still rests on one arch's data
(Wormhole's DRAM-write row has that 64 B outlier), so it is still not done.

### The verdict: climbed for latency, failed for bandwidth

Both halves are worth stating plainly, because a ladder rung that could not
fail would not have been worth building.

**Climbed.** On the 60 points the model is allowed to predict, three of the
four axes are flat and the residual is the constant it was predicted to be:
77 – 94 cycles of unmodelled issuing-core path. The hop term explains an
88-cycle geometry difference to 4 cycles on Wormhole and a 147-cycle one to 18
on Blackhole. The DRAM access latency, derived from a *different* vendor
dataset, reproduces this one's DRAM-vs-L1 difference to 5 cycles. Nothing was
fitted to any of it.

**Failed.** The size axis is not flat. The bandwidth term is ~10 % optimistic
on Wormhole L1 reads and ~24 % optimistic on DRAM, and the second of those is a
missing term rather than a wrong one. Rung 2 has done exactly what it was for:
it found something, it named which term, and the term it named is fixable from
a number already in the file at `isa_doc`.

**And 99.3 % of the dataset was not touched at all.** Every one of the seven
exclusions is a real gap — congestion first among them — and closing any of
them makes more of this dataset answerable. That is the most useful thing this
instalment produces: a measure of how much of a public dataset the model is
currently entitled to be tested against, which was previously unknown and is
now 0.7 %.

### Addendum, 2026-08-05: what is actually in the 99.3 %, and why congestion cannot come out of it

The sentence above is true and it is also the most over-readable sentence in
this file. "Closing any of them makes more of this dataset answerable" invites
the reading that the pattern rule's 150 entries are 150 rows waiting on a
congestion model. **They are not.** The sweep now counts exclusions the other
way round — over every rule at once rather than in ladder order — and reports,
per rule, how many entries that rule *alone* keeps out:

| rule | removes (in ladder order) | **sole cause** |
| --- | --- | --- |
| `mechanism != UNICAST` | 150 / 170 | **0** |
| `pattern ∉ {ONE_FROM_ONE, ONE_TO_ONE}` | 150 / 170 | **2** |
| `num_transactions per barrier ≠ 1` | 40 | **24** |
| `stateful` | 4 | **4** |
| `loopback`, `memory == DRAM_INTERLEAVED` | 0 | 0 |

So a perfect congestion model, arriving on its own, would take the retained set
from 6 entries to 8 — and both of the two are degenerate `num_subordinates: 0`
DRAM rows whose large sizes are extractor fill. A multi-party row is almost
always *also* a multi-transaction row, and a multicast row is almost always
both. **Congestion is never the only thing in the way**, and the honest
priority order that falls out of this is the opposite of the intuitive one: the
issue-loop term (24 entries) is worth more of this dataset than congestion is.

**Can a congestion model be derived from the dataset?** No, and the reason is
identifiability rather than coarseness — a sharper claim than the "too coarse …
folds everything into one number" this file recorded, and one that can be
checked rather than asserted:

1. Every multi-party row changes the flow count by **resizing a grid** (2×2,
   3×3, 5×5, 8×8, the whole device, all anchored at logical (0,0), from
   `test_noc_estimator.cpp`). Flow count, path length and link sharing move
   together, in four or five steps, and no two rows hold any one of them fixed.
   A per-hop congestion coefficient is a slope against "flows sharing this
   link"; that axis does not exist in the file.
2. **No core coordinates are recorded**, and there is one aggregate scalar per
   key, so even with the geometry re-derived from the test source a fit would
   have to explain a whole grid's barrier through a single number.
3. That scalar already contains two other unmodelled terms. The retained rows
   put the issuing core's path at a median residual of 90 (Wormhole) / 92
   (Blackhole) cycles, and the all-to-all rows at 64 B cost **90, 63, 47, 49 cycles
   per transaction issued** at N = 4, 9, 25, 56 against a **293-cycle**
   single-transaction round trip. They are pipelined, and what governs them is
   the issue loop, not the network. L1 port arbitration is in there too.

One equation, three unknowns, every configuration moving all three. A
coefficient fitted through that would be `provenance: estimated`, and
`vendor_source_derived` is not available either — it needs arithmetic on
published vendor numbers, and no arithmetic here separates congestion from the
issue loop and the L1 ports. **Nothing was wired**, and that is the result.

**What the dataset does give is a bound**, which is the useful half of the null.
All-to-all over an N-core grid at 64 KiB, one transaction per (master,
subordinate) pair per barrier — the only shape in the file where the flow count
is the only thing that changes:

| cores | Wormhole aggregate B/cycle | per core | Blackhole aggregate | per core |
| --- | --- | --- | --- | --- |
| 4 | 59.0 | 14.74 | 97.7 | 24.42 |
| 9 | 77.3 | 8.59 | 174.5 | 19.38 |
| 25 | 123.3 | 4.93 | 246.2 | 9.85 |
| 56 / 64 | 200.1 | 3.57 | 433.9 | 6.78 |
| 110 | — | — | 587.2 | 5.34 |

14× the cores buys 3.4× the aggregate bandwidth on Wormhole and 28× buys 6.0×
on Blackhole, so the per-core share falls ~4× on both. **Any congestion model
tt-sim ever gains has to reproduce that curve**, and that is precisely the
"validate, not derive" role this file gave rung 2 in the first place.

**What would derive one**, on a card, from tt-metal's own
`tests/tt_metal/tt_metal/data_movement/` microbenchmarks — which expose the
coordinates the shipped YAML drops and write per-core profiler CSVs:

- `one_to_one` / `one_from_one` (IDs 4, 5, 50, 51) swept over
  `master_core_coord` × `subordinate_core_coord`: latency against a **known**
  hop count, which pins the uncongested line. The shipped dataset has only the
  same-axis/different-axis pair.
- `all_to_all` / `all_from_all` (300–308, 310–318) with `sub_grid_size` **pinned
  to 1×1** and `mst_grid_size` swept: N flows into one endpoint at a fixed
  distance — the endpoint-contention curve with geometry held still.
- The same pair with **N fixed at 2** and the masters slid so their routes share
  0, 1, 2 … k links. The slope of latency against shared links *is* the per-hop
  congestion coefficient, measured with everything else constant.
- `core_bidirectional` (140–148) with `write_vc` swept 0–3, isolating
  virtual-channel arbitration — the one congestion mechanism the ISA docs name.

The first and third together are the minimum: an intercept and a slope, both
measured with the confounds held fixed rather than fitted out. The sweep prints
this list, so it is where a reader of the null result finds the next step.

### Addendum, 2026-08-05: the list above is now a harness, and one item of it was wrong

The four microbenchmarks the addendum lists as "what would derive one" are now
`perfbench/nocbench`, planned by `tt_sim/perf/noc_congestion_plan.py` and read
back by `tt_sim/perf/noc_congestion_sweep.py`. Three things came out of building
it, and two of them are corrections to the list rather than confirmations of it.

**It could not be tt-metal's data_movement suite, and the reason is exact.**
Every coordinate, grid size and virtual channel in that suite is a compile-time
literal inside a gtest body — C++ default arguments on helper functions, called
from `TEST_F` with `CoreCoord(0, 0)` / `CoreCoord(0, 1)`. No environment
variable, no gtest flag, no argument selects them; `getenv` appears five times in
the whole suite and every one is `TT_METAL_SIMULATOR` in the Quasar tests. The
one test whose name promises otherwise, `TensixDataMovementOneToOneCustom`,
begins with `GTEST_SKIP()`. The upstream change that would fix it is small —
read `OneToOneConfig` / `AllToAllConfig` / `CoreBidirectionalConfig` from the
environment, since those structs already have exactly the right fields and only
the call sites hard-code them — but it is a change to tt-metal, so the harness is
a standalone tt-metal program here instead.

**Item (a) cannot give what it promised, and that is a hardware fact.** Sweeping
`master_core_coord` x `subordinate_core_coord` does not produce a fine-grained
hop sweep. Both NoCs are *directional* tori, so a request and its reply travel
the same way round the ring and a round trip costs `grid_x` hops for any two
cores in one row, `grid_y` for any two in one column and `grid_x + grid_y` for
any pair differing on both axes — **three values, whatever the coordinates**.
What the sweep actually yields is a three-level line plus a flatness test, and
the flatness test is the more interesting half: latency rising with `|dx|` along
a row would mean the NoC is not the directional ring every hop count in these
tables assumes. Nobody appears to have checked that.

**Item (c) survives, and it is still the decisive one.** Two flows, N fixed at
2, both masters in one row and each writing to a *different* row, so the forward
paths overlap in the masters' row by an amount set by the masters' separation
while the acknowledgement paths (which return along the destinations' rows) do
not overlap at all. Sliding the second master moves one number. The placement is
searched rather than written down, because the constraint that matters — every
leg-pair link overlap except the payload one stays at zero — is easier to check
than to solve, and `check_invariants` refuses a plan in which anything else
moved. On Blackhole the search finds eight points, shared links 0 through 7,
with flow A identical at every one of them.

**What a simulator run showed.** tt-sim models no router-to-router congestion,
so the experiment is forced flat there; the point of running it is that
everything *else* is exercised. Two results and one gap:

* the three round-trip levels give `cycles = 188.3 + 8.95 * hops` at r2 1.00 —
  the ISA docs' 9-cycle hop, recovered end to end through a real tt-metal
  program, the wire bridge and the simulated NoC rather than by asking the cost
  table what it thinks;
* the concurrency control caught a real bug in the harness. The kernel had been
  handed a semaphore *id* and used it as an *address*, so the rendezvous never
  blocked and the flows overlapped by 0.02. Every congestion number from that
  run would have been meaningless and would have looked entirely normal;
* the run's verdict is `INVALID`, and correctly. The self-port control — two
  flows on one core's two data-movement RISCs, same NoC, starting on the same
  cycle and overlapping fully — costs 868 cycles alone and 868 / 798 together.
  **tt-sim's `NUI._tx_free_cycle` does not serialise two baby cores on one
  injection port**, so the one contention mechanism the model has is inert on
  this path. That is a gap in the simulator, not in the harness, and the harness
  refusing to call its flat shared-link reading "no congestion effect" under
  those conditions is the whole design working.

  **The bolded sentence above is wrong, and the retraction is below** — see
  "Retraction, 2026-08-05: the self-port control is not a measurement". The
  verdict of `INVALID` stands; its stated cause does not.

Nothing was wired into either cost table and nothing here is `estimated`. A
coefficient, if a card ever produces one, would be `vendor_source` at best: a
measurement on one part, not a published number.

### Retraction, 2026-08-05: the self-port control is not a measurement

The claim above — that `NUI._tx_free_cycle` fails to serialise two baby cores
sharing an injection port — is **false**, and the reading that produced it is
not evidence about any hardware. Nothing in `tt_sim/` changed as a result;
this section exists so that the next reader does not "fix" a working model.

**What the simulator actually does.** Instrumenting `claim_injection_port` and
`transmit` through the very run the finding came from: with two flows the one
NIU at `(1, 2)` on NoC 0 claimed its port eight times, `_tx_free_cycle` walking
`9956 → 10084 → 10212 → 10340 → 10468 → 10596 → 10724 → 10852` in exact
128-cycle steps (8 KiB at Blackhole's 64 B/cycle), and the eight writes landed
at their destinations `9992, 10120, 10248, 10376, 10504, 10632, 10760, 10888`
— evenly spaced, interleaved between the two subordinates, one packet's
injection apart. That is textbook serialisation of two RISCs on one port. The
model was never inert.

**Why the kernels reported otherwise.** Both kernels stamped `t1 = 10625`. The
acknowledgements arrived at `10236, 10364, 10492, 10620, 10748, 10876, 11004,
11132`; the **fourth** is at 10620. `noc_async_write_barrier` is

```c
// tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h
return (NOC_STATUS_READ_REG(noc, NIU_MST_WR_ACK_RECEIVED) == noc_nonposted_writes_acked[noc]);
```

— a **per-NIU hardware** counter compared against a **per-RISC software**
counter. Each RISC issued four writes and so waits for the shared counter to
advance by four; with two issuers it does that after any four acks. Both
kernels stop at the fourth. The timed region ends before more than half the
traffic has landed, and it does so by construction.

**The control is blind to the thing it exists to detect, and that was measured
rather than argued.** `claim_injection_port` was surgically stubbed to return 0
and never advance `_tx_free_cycle` — the mechanism under test removed outright
— and the same two points were re-run:

| | 1flow | 2flows | ratio | verdict |
| --- | --- | --- | --- | --- |
| injection port serialising | 217.0 cycles/tx | 208.5 | **0.96** | FAIL |
| injection port **removed** | 142.0 cycles/tx | 130.0 | **0.92** | FAIL |

Deleting the only contention mechanism in the model moves the control's ratio
by 0.04, in the wrong direction, and leaves its verdict unchanged — while
moving the *absolute* cost by 35 % (75 cycles/tx off the single flow). The
harness resolves the injection port perfectly well; what it cannot resolve is
that port through a 1-vs-2 ratio whose timed region ends on a shared ack
counter. A control that reads the same whether or not the effect exists is not
evidence either way, and the card's **1.00** is the same non-reading as
tt-sim's 0.96.

The reason the ratio barely moves is worth stating, because it is a design
lesson for the replacement: at four transactions the region is dominated by the
fixed round trip (~440 cycles of the 568 at 29 hops), so the per-transaction
term — the only part the port touches — is a small share of what the barrier is
timing, whichever flow's acks release it. A control of this shape needs the
port to be the *bottleneck*, not merely present.

**The premise cannot be built at all.** A Tensix's two data-movement RISCs
cannot share a NoC. `BRISC_WR_CMD_BUF` and `NCRISC_WR_CMD_BUF` are *both*
command buffer 0 ("for large writes", same header, both architectures): the two
RISCs are separated by *NoC*, never by command buffer, so two kernels on one
NoC race on one set of `NOC_TARG_ADDR` / `NOC_PACKET_TAG` / `NOC_CTRL`
registers. tt-metal forbids the configuration structurally —
`tt_metal/impl/program/program.cpp` sets `kernel_config.brisc_noc_id() = 1 -
arg.noc` for a `RISCV_1` kernel, so requesting NoC 0 on both RISCs still leaves
the firmware driving them as a pair on opposite NoCs. Running it anyway is not
just unmeasurable: the barrier's `==` against a shared monotonic counter can be
overshot, so two issuers with unequal transaction counts can **hang**.

**How many injection ports a Tensix has, from the sources.** One per NoC, four
command buffers each, shared by all five RISCs — the tile's NoC register blocks
are at `0xFFB20000` and `0xFFB30000` in the address space every baby core sees.
Tenstorrent's own reference simulator holds exactly that shape:
`src/sim.h` declares `noc_targ_addr_lo[NUM_NOCS][NUM_CMD_BUFS]` and
`noc_packet_tag[NUM_NOCS][NUM_CMD_BUFS]` — indexed by NoC and command buffer,
never by RISC — and `niu_mst_wr_ack_received[NUM_NOCS]`, one master ack counter
per NoC per tile. tt-sim's one `NUI` per NoC per tile, with one
`_tx_free_cycle`, is the same structure. **No serialisation rule was invented
and none was needed.**

What the harness needs, and it is a change to `perfbench/nocbench` rather than
to the simulator: the control must stop asking for two flows out of one core.
Two masters **reading** from one subordinate reaches the same shared resource
by the supported route — a read's payload rides the return leg, so both
response streams leave the subordinate's single NIU and queue on its one
injection port — with one data-movement RISC per core, so the rendezvous, the
overlap check and the barriers all mean what they say. A plan-time refusal of
any point with two flows on one core would have caught this before the run, and
would keep a card from being handed a configuration that can hang it.

**Both of those were then done, and the run they were done for produced a
result** — see "Banked, 2026-08-05" immediately below for the congestion
measurement, and "The replacement control" at the end of it for the ablation
that holds `readport` to the standard this retraction set.

### Banked, 2026-08-05: congestion on Blackhole is a step at the first shared link

The Blackhole run stands, and it answers a different question from the one the
harness was built to ask. 79 flows over 55 runs, median timed-region overlap
0.99, zero invariant complaints, the `size` control passing. Nothing below goes
into either cost table and `noc.congestion` stays `provenance: unknown`; the
reason is the substance of the finding rather than caution about it.

**The uncongested line, measured.** `cycles = 4373.7 + 8.38 * round_trip_hops`
at r2 1.00 over the three predicted round-trip levels (12 / 17 / 29). That is
the ISA docs' ~9-cycle router-to-router hop, on silicon, recovered through a
real tt-metal program rather than by asking the cost table what it thinks. The
directional-torus prediction the whole hop model rests on is confirmed: within
each family the round trip is constant whatever the coordinates, and the row
family — predicted constant — spans 0.6 cycles/tx, which is the noise floor
every flatness claim below is read against.

**The congestion result.** Two flows, everything frozen but the number of
router-to-router links their payloads share:

| shared links | 64 B | 16384 B |
| --- | --- | --- |
| 0 | 39.9 | **270.1** |
| 1 | 39.8 | **517.9** |
| 2 | 39.8 | 519.1 |
| 3 | 39.8 | 517.2 |

At 64 B: slope −0.00, span 0.1 cycles against the 0.6-cycle floor. Flat, and
predicted flat — a 64-byte packet holds a link for one cycle against a
~40-cycle issue loop, so the links are ~2 % busy and nothing can show. That is
the negative control working. At 16 KiB the cost nearly doubles at the **first**
shared link and then stops moving.

**It is bandwidth splitting, and the arithmetic closes.** 16384 B at Blackhole's
64 B per cycle is 256 cycles of link occupancy. One flow costs 270.1 cycles per
transaction: 256 of occupancy and ~14 of everything else. Two flows sharing one
link cost 517.9: 2 × 256 plus the same ~6. The link is saturated and its
bandwidth is divided by the number of saturating flows crossing it. The
endpoint curve says the same thing from the other end — six 4096 B flows into
one subordinate come back at 137 / 203 / 268 / 333 / 399 / 399 cycles per
transaction, evenly spaced by ~65.5 where 4096 / 64 is 64, in a stable rank
order.

**So there is no coefficient, and that is the answer.** The experiment was
designed to fit a slope in cycles per shared link per concurrent flow, because
that is the shape a per-hop congestion adder would have. The measurement says
congestion on this part is not a per-hop adder at all: it is an occupancy on
the *busiest* link of a route, and the second, third and fourth shared links
cost nothing extra because they are the same two flows on the same schedule.
Fitting +22.9 cycles per shared link through the eight points — which is what a
naive regression gives, at r2 0.39 — would describe a machine that does not
exist.

**Why no number goes into the tables.** The cost model charges a hop count and
an injection port. A term of this shape needs a quantity nothing in `tt_sim`
computes: for each flow, the maximum over the links on its route of the number
of concurrent saturating flows crossing that link — a per-link flow census,
maintained over time, which is a scheduler and not a coefficient. Writing 248
cycles into a per-shared-link slot would not be a conservative approximation of
the measurement, it would be false to it: it would charge two shared links
twice what it charges one, and the card charges the same. `noc.congestion`
therefore stays `unknown` with the measurement recorded in its `note`, which is
the honest state: *the quantity is measured, the model has nowhere to put it.*

**What is still open, and needs a card.** Two things, and the harness now asks
both:

* Is the halving virtual-channel arbitration or plain occupancy? The ISA docs
  name the one mechanism — "if the two packets have the same virtual circuit
  number, then one packet will wait for the other" — and every flow in the
  measured run was on one channel. `plan_vc` now puts two writers on one shared
  link with the first pinned at `NOC_UNICAST_WRITE_VC` and the second swept
  0–3, so `vc1` against the other three separates them. If different channels
  avoid the halving, the model term is per-VC-per-link; if they do not, it is
  per-link and the channel does not enter it.
* Where the 64 B and 16 KiB regimes meet. The `--shared-sizes` axis takes
  intermediate sizes; nothing has been run between them.

### Two problems in the run itself, both now fixed in the harness

**The card was harvested, and four of the eight shared-link counts were
therefore wrong.** The dump reports addressed worker columns `{1..7, 10..14}`.
That is a legal *subset* of an unharvested Blackhole's physical columns
`{1..7, 10..16}`, so `_resolve_grid` concluded the dump was already physical.
It was not: the kernels' own `NOC_NODE_ID` puts them at `{1..7, 12..16}`, two
columns right of where the plan placed anything with addressed x ≥ 10.
Re-deriving from the self-reports, the eight `shared` points sit at 0, 1, 2, 3,
**6, 7, 8, 9** shared links, and flow B's forward hop count — which the
experiment declares fixed — is 8 at the first four and 12 at the last four.
Everything else survives the correction: `other_overlap` is zero at all eight
points on the true geometry as well, flow A is identical throughout, and every
round trip is still 29. **The banked step therefore rests on the four points
(0, 1, 2, 3) where flow B really is held still**, which is where the step is;
the 6–9 group is a different experiment and its ~11-cycle offset is confounded
with those four extra hops.

The executor flagged 8 of the 79 rows as coordinate mismatches, which is the
only reason any of this is knowable, and it flagged them because the kernels
report where they are rather than being asked. Two changes followed.
`--dump-grid` now launches a probe kernel on the first logical row and column
and writes each core's own `NOC_NODE_ID` as `phys_x` / `phys_y` — the host API
has no way to ask for that coordinate, and the addressed one cannot be inverted
without it. And `_resolve_grid` refuses a dump that is short of the full worker
grid and carries no such column, instead of inferring: "these look physical"
was true of this card and wrong.

**The `vc` experiment hung the card, and no plan can emit one again.** The first
and only `direction=BIDIR` point never returned; all 79 unidirectional flows in
the same session completed. Three things are established about it and one is
not:

* it is **not** the virtual channel — every one of those 79 completed flows
  issued its writes on VC 0, which is the channel the hung point was on;
* it is **not** the kernel — tt-sim runs the identical binary and the identical
  plan (64 × 4096 B, bidirectional, VC 0–3) to completion in 4958 cycles, so
  there is no logic error to find;
* it is not unknown territory upstream: tt-metal's own `core_bidirectional`
  suite disables its whole directed-ideal family — same-kernel *and*
  different-kernel, write-VC sweep included, tests 140–145 and 148 — with
  `GTEST_SKIP() << "Skipping test"; // Timeout issue (#36428)`. Only the small
  `packet_sizes` variants still run;
* **the cause is not established.** The `==`-against-a-monotonic-counter
  overshoot the retraction above identifies needs *two* issuers on one NIU, and
  the `vc` point has one, so that specific mechanism is the `selfport` hazard
  and not this one. tt-sim models no virtual channels, no NoC buffer
  back-pressure and no request/response buffer coupling, which is the family a
  bidirectional deadlock would belong to — so the simulator is blind to it by
  construction and the diagnosis needs the card.

`check_invariants` therefore **refuses `DIR_BIDIR` outright**. Refusing is not a
diagnosis; it is the only responsible thing to emit while there is not one. The
virtual-channel question survives in a better form, described above, that needs
no bidirectional flow.

### The replacement control, and the proof it is not blind

The retraction above ends by proposing two masters reading from one
subordinate. That is now `plan_readport`, and the standard the retraction set —
*ablate the mechanism and show the control moves* — has been applied to it.

`claim_injection_port` was stubbed to return 0 and never advance
`_tx_free_cycle`, exactly as it was for the control it replaces, and the same
two points re-run:

| | 1 flow | 2 flows | ratio | verdict |
| --- | --- | --- | --- | --- |
| `selfport`, port serialising | 217.0 cycles/tx | 208.5 | 0.96 | FAIL |
| `selfport`, port **removed** | 142.0 | 130.0 | 0.92 | FAIL |
| `readport`, port serialising | 170.8 cycles/tx | 251.9 | **1.48** | **PASS** |
| `readport`, port **removed** | 80.5 | 80.5 | **1.00** | FAIL |

Deleting the mechanism moves the old control by 0.04 in the wrong direction and
leaves its verdict unchanged; it moves the new one by 0.48 and flips its
verdict. The difference is not the geometry, it is the bookkeeping: one
data-movement RISC per core means each kernel's `noc_async_read_barrier` waits
on its own core's `NIU_MST_RD_RESP_RECEIVED`, so the timed region ends when
*that* core's traffic has landed rather than when half of somebody else's has.
The sizing rule follows from the retraction's own diagnosis and is enforced:
`plan_readport` refuses a point carrying under 64 KiB per flow, because below
that the fixed round trip rather than the shared port dominates the region —
which is precisely how the old control came to be sized at 4 × 8 KiB.

One property of the new control is stated rather than glossed: it proves the
flows contend, it does not say *where*. Two response streams leaving one tile
share the first router-to-router link out of it as well as the port, and on
silicon those are one reading. That is all a positive control has to do;
attribution is experiment 2's job, and experiment 2 is what produced the result
banked above. On tt-sim the two are separable, because tt-sim charges nothing
whatever for a router-to-router link — which is what makes the ablation above a
clean test.

With `readport` in place the simulator run's verdict is `NO CONGESTION EFFECT`
rather than `INVALID`: the controls move, the flows overlap, and the flat
shared-link reading is the forced null it was always supposed to be, now
distinguishable from a harness that measured nothing.

### What broke

Nothing. The sweep consumes no table and changes no cycle: with
`TT_SIM_COST_MODEL` unset it does not run at all, and with it set it builds
throwaway devices in its own process. The whole suite is **659 passed** (the
24 Blackhole guards, the Wormhole guard family, the 11 live tt-metal examples
and this instalment's 29 new tests), the Wormhole offline replay is still
**126/126 byte-identical**, and `driver/tests/cost_model_gate.py` reports
**`RESULT: PASS`** with every example bit-for-bit at its recorded poll
multiplier.

Two things had to be got right and both are pinned by tests rather than by
prose:

1. **The geometry**, which the dataset does not record. It does not have to:
   `run_single_core` picks the subordinate as logical `{0,1}` for `same_axis`
   and `{1,1}` otherwise, and on a directional torus those round trips are
   exactly `grid_y` and `grid_x + grid_y` hops **whatever the coordinates**, on
   either NoC. So no core-placement assumption is doing any work, and
   `test_same_axis_is_the_shorter_round_trip_and_by_a_whole_ring` says so on
   both architectures.
2. **That the predictor is the model**, not a re-derivation of it.
   `test_a_real_l1_round_trip_costs_what_the_tables_compose_to` runs the real
   device against the closed form for both arches, both geometries and three
   sizes including the 8-chunk 64 KiB case, so a harness that drifted from
   `tt_sim/network/tt_noc.py` would fail rather than quietly report agreement
   with itself.

## The gate

Everything above is a **timing** change, and the section before it is the
finding that this project's strongest correctness guard cannot validate one:
byte-identical replay pins the host's poll count, so any change to timing makes
it fail whether or not a value went wrong. What survives is the family that
pumps until the go-message flips to `DONE` and *then* checks the result — and,
as it turned out, that family is smaller than the obvious reading of "value
guard" suggests. So the gate existed in fact and not in name — nothing ran it, nothing wrote down which
guards were in it, and nothing stopped the next reader from hitting a failing
`examples_replay_test`, concluding "the cost model breaks the guards", and
either reverting good work or weakening a guard to make it green.

`driver/tests/cost_model_gate.py` is that gate, named:

```bash
python3 -m driver.tests.cost_model_gate             # the whole thing
python3 -m driver.tests.cost_model_gate --list      # just the classification
python3 -m driver.tests.cost_model_gate --stage value six softplus
python3 -m driver.tests.cost_model_gate --model-off # the same set, model off
```

Three stages, all with `TT_SIM_COST_MODEL=1` in the environment of every
subprocess it spawns (the model is read once, when a unit is constructed, so
process isolation is the only honest way to set it):

1. **`pytest tt_sim -q`** — the simulator's own unit tests under the model.
   `tt_sim` and not `tt_sim driver`, because the driver tree is where the
   timing-pinned guards live and stage 3 handles those.
2. **Every budget-independent guard** (28), each as its own `python3 -m …`
   process.
3. **Every budget-dependent guard** (6), under the poll-budget prover below.

**Where it lives, and why.** Not next to `driver/blackhole/tests/run_examples.sh`,
because the gate spans both architectures — 20 of the budget-independent guards
are Blackhole's and 8 are Wormhole's, and a gate that covered one arch would
miss the arch where the model moves cycles most. Not a shell script either,
because proving a budget-dependent guard benign means *importing* the guard
module and rebinding its `make_device`, which a wrapper around `python3 -m`
cannot do. `driver/tests/`
is the first cross-arch, driver-level test directory; it holds the runner and a
millisecond-scale pytest tripwire (`guard_classification_test.py`) that rides in
the normal `pytest driver` run.

### Every guard, classified

This is the part a future agent actually needs: hitting a failing
`examples_replay_test` under the cost model, is it a bug or an artefact? The
first cut is a mechanical rule, *executable* (`gate.classify`) rather than a
list somebody maintains by hand:

> A guard is **timing-pinned** exactly when it compares a replayed reply against
> the reply recorded in the trace — in this tree, when it reads the parsed trace
> line's `["reply"]` field, or delegates to `replay.py`, which does the same in a
> subprocess. Everything else asserts a value it computed for itself.

But what a guard *asserts* is not what decides whether it can validate a timing
model. That is decided by a second, blunter property:

> **Does the guard's verdict depend on the recorded poll budget?** A captured
> trace contains a fixed number of go-message polls, each worth
> `cycles_per_poll` (100) simulated cycles. A guard whose kernel must finish
> inside that budget is pinned to it, whatever it goes on to check.

The two cuts do not coincide, and getting that wrong is what the first version
of this gate got wrong:

| Class | Guards | Budget? | What the gate does |
| --- | --- | --- | --- |
| **timing-pinned** (4) | `wormhole/examples` (11 traces), `wormhole/offline`, `wormhole/one`, `blackhole/offline` | dependent | poll-budget proof (`wormhole/one` excluded, below) |
| **value / poll-budget** (2) | `blackhole/two`, `blackhole/dramtop` | dependent | poll-budget proof |
| **value / pump-to-done** (30) | Blackhole `eight` `five` `five_fp` `four` `four_fp` `loopback` `matmulblock` `matmulidx` `nine` `optest` `reduce` `reduceneg` `sfpuchain` `sfpumath` `six` `softplus` `three` `tilize` `transpose` `untilize` `where`; Wormhole `matmulblock` `matmulidx` `reduce` `reduceneg` `sfpumath` `softplus` `tilize` `transpose` `untilize` | independent | run under the model |

`two` and `dramtop` check a computed value — so what they *assert* is
timing-agnostic — but they take the device's cycles from the trace's own polls
rather than pumping until `DONE`, so whether they get far enough to assert
anything is not. They are in the *same* gate role as the byte-identical family,
and calling them "value guards" and running them as-is, as the first version of
this gate did, is a category error: it reads a poll-budget truncation as a wrong
answer. The class name says the consequence now, not the mechanism.

The one guard on the Wormhole side worth naming is `wormhole/examples`: it is
*eleven* traces behind a single module, which is why "11 of the Wormhole
`examples_replay_test` cases fail" reads as eleven failures from one file.

### The poll-budget proof

An exclusion list would have been enough to make the gate green, and it would
have been the weaker answer: it records that a guard was not run, not that it
was not run *for a good reason*. So the gate **demonstrates** the artefact
instead — by re-running the guard whole at a larger `cycles_per_poll`:

> `cycles_per_poll` is the one knob that says *the host waited longer*, which is
> what a live host does and what a recorded poll count cannot express. The
> replay is re-run at 1×, 2×, 4× and 8× the recorded 100 cycles per poll, and
> the guard must come out **completely clean** at one of them — byte-identical
> for the timing-pinned family, value-correct for the poll-budget family. The
> multiplier it needed is reported. A guard still dirty at 8× is not a
> poll-budget artefact and the gate fails on it.

Two things make this trustworthy rather than a way of turning failures green.

**It is not a pump-after-replay, and that distinction is the whole lesson of
this section.** The obvious cheap proof — replay, and on a mismatch pump the
device and re-issue the same READ — is what the RISC-V instalment used by hand
and what the first version of this gate automated. It is valid *mid*-replay,
and invalid the moment the replay is exhausted: **every captured trace ends with
`RESET_ASSERT` to every core followed by `EXIT`**, so a device pumped after the
last trace line has all five baby cores held in reset and cannot advance the
kernel by a single instruction, however many cycles it is given. A guard of the
poll-budget shape is therefore guaranteed to report "still wrong after N further
cycles" no matter how benign the perturbation. That is the most expensive wrong
answer a gate can give, because it is precisely the signal meant to stop
everyone and start an investigation — and it gave it, on `blackhole/two`, in
this file's first draft. `test_the_prover_never_pumps_after_the_replay` now
fails if the technique comes back, and
`test_every_trace_really_does_end_in_reset_then_exit` checks the premise rather
than assuming it.

**The go-message mailbox is excluded, and identifies itself.** One class of READ
cannot reproduce under *any* timing change, in either direction: the mailbox the
host spin-polls. Its recorded value is literally "how far had the kernel got when
the host looked", so a slower device misses a `DONE` and a device given a longer
poll budget reports `DONE` where the recording still said `RUNNING`. Those are
counted and reported separately rather than failed on — and they are found *from
the trace*, as an address one core reads many times over, rather than from a
hardcoded offset. The separation is not marginal: in the captured traces the
polled address is read 30–101 times and every other address exactly once. So the
rule works on either architecture and survives a tt-metal offset bump, and
**every other READ, the result buffer included, must still be byte-identical**.

### Measured

Model on, working tree of 2026-08-03 (RV + five Tensix units + the NoC per-hop
latency model, all behind the same switch). Every data READ byte-identical on
every trace; the only reads that move are the go-message polls, which cannot do
anything else.

| Trace | data READs, all bit-for-bit | poll budget needed | go-message polls |
| --- | --- | --- | --- |
| `wh examples:one` | 104 / 126 | 4× | 22 |
| `wh examples:two` | 113 / 128 | 2× | 15 |
| `wh examples:three` | 128 / 143 | 2× | 15 |
| `wh examples:four` | 104 / 125 | 4× | 21 |
| `wh examples:four-fp` | 110 / 125 | 2× | 15 |
| `wh examples:five` | 117 / 133 | 2× | 16 |
| `wh examples:five-fp` | 117 / 135 | 2× | 18 |
| `wh examples:six` | 193 / 212 | 4× | 19 |
| `wh examples:eight` | 113 / 127 | 2× | 14 |
| `wh examples:nine` | 106 / 125 | 4× | 19 |
| `wh examples:loopback` | 110 / 125 | 2× | 15 |
| `wh offline` | 104 / 126 | 4× | 22 |
| `bh offline` | 191 / 220 | 2× | 29 |
| `bh two` (value) | 100 / 100 elements | 2× | — |
| `bh dramtop` (value) | 256 / 256 words | 1× | — |

**Zero dirty data READs, on every trace, at 4× or better.** With the model
*off*, all 1,750 READs across the 13 traces reproduce at 1× with nothing
excluded at all — go-message polls included — which is the control that says the
poll exclusion is not quietly covering a standing mismatch.

The multipliers in that table are from the tree that landed the hop model, and
two instalments have moved some of them since: **as of 2026-08-03 with the whole
model on**, `wh examples:six` needs 8× and `three` / `four` / `four-fp` /
`nine` / `loopback` / `one` need 4× where the table above records 2× or 4×.
Still zero dirty data READs anywhere, and the Blackhole rows (`offline` 2×,
`two` 2×, `dramtop` 1×) have not moved at all — including across the Blackhole
DRAM term, which is worth 24 % on `six` and 0 poll budgets here. The
multiplier is a ladder rung, not a measurement, and a table of them ages with
every timing change by design; what does not age is the "zero dirty" column.

That is a materially stronger claim than the one the RISC-V instalment recorded
("11 of 11, one mismatched READ each, resolved within 2,000 further cycles").
The right characterisation is not *mismatches that resolve* but **mismatches
that disappear entirely once the replay is given a realistic poll budget** — and
the 2,000-cycle top-up that instalment used was calibrated against a much weaker
perturbation. `six` alone now needs tens of thousands of cycles more under the
NoC model, so a fixed top-up was never going to survive the next unit wired.
This is why the ladder is expressed as a multiple of the host's poll budget
rather than as a constant number of cycles: it scales with the perturbation
instead of having to be re-tuned for each one.

### What it found, and one thing it got wrong

**Retracted: `blackhole/two` is a timing artefact after all.** An earlier
version of this section reported that the in-flight NoC latency model made
`blackhole/two` compute a wrong value, "still wrong after 200,000 further
cycles", and flagged it as a genuine regression. **That verdict was wrong, and
the fault was in the prover, not the model.** `two` is a `value/poll-budget`
guard, and the proof used was pump-after-replay — which cannot work for that
shape, because `two.trace` ends with `RESET_ASSERT` to every core: the 200,000
cycles were spent on a device with every baby core held in reset. Re-run
properly, at 2× the recorded poll budget, `two` computes all 100 elements
correctly under the model. It needs ~16,400 cycles where the untimed device
needed 8,200, and the trace budgets 8,200.

The episode is kept rather than quietly deleted because it is the sharpest
argument for the shape of this gate. A false "NOT a timing artefact" is more
expensive than a missed one: it is exactly the signal meant to stop everyone and
start an investigation, and it cost one. Both halves of it are now pinned by
tests — the premise (`test_every_trace_really_does_end_in_reset_then_exit`) and
the technique (`test_the_prover_never_pumps_after_the_replay`).

**Stands: a timing pin outside the replay guards.**
`tt_sim/network/noc_routing_test.py::test_noc1_responses_return_to_the_issuing_worker`
failed under the model while the NoC hop model was landing, and that one *is* an
artefact of the same family: the test drives a DRAM round trip and pumps
`device.run(16)`, a hardcoded cycle budget that a per-hop latency model of
~5 / 9 / ~5 no longer fits inside. Demonstrated by widening it — a copy of the
file with `run(400)` passes all five cases with the model on. Worth naming
because it extends the point past the replay guards: **a unit test that pumps a
fixed number of cycles is a poll-budget pin as surely as a byte-identical replay
is**, and `gate.classify` will not find those, because they are not guards and
have no trace.

### What the gate proves, and what it does not

Being precise here matters more than the gate does. Overclaiming would be worse
than having no gate at all.

**It establishes** that the cost model changes **no computed value** on any
in-tree workload: every budget-independent guard's result buffer is what it is
with the model off, and every budget-dependent guard is completely clean once
given a realistic poll budget. It also establishes that the model does not
crash, deadlock or reorder anything into a wrong answer — which is not idle,
since a timing perturbation has already found two real bugs (the config unit's
missing config-write ordering, and the sync unit's mixed-queue `KeyError`).

**It establishes nothing about whether the modelled cycle counts are right.**
The gate never compares a cycle count to anything, and could not: this file is
explicit that a total is not yet a prediction. DRAM answers instantly, the NIU
register block a dataflow kernel actually polls is uncosted, and there is no
predictor to charge a mispredict against. A green gate means *the model is
value-safe*, never *the model is calibrated*. The ladder that would go after
calibration is ["What calibration would take"](#what-calibration-would-take),
and none of its rungs are this gate.

The poll-budget multiplier is *not* a cycle-count measurement either, and should
not be read as one. It is a lower bound at ladder resolution: "2×" means the
guard was clean at twice the recorded budget and not at once it, which brackets
the slowdown very coarsely and says nothing about where the extra cycles went.

It also says nothing about workloads with no in-tree guard, and — because the
guard set is discovered — its coverage is exactly whatever the tree happens to
contain on the day it runs.

**Validated against** commit `1ab9e3b` plus the working tree that landed the NoC
per-hop latency model and the `untilize` guards — 34 guards discovered, 28 run
under the model, 6 proven, exit 0 on both. (The `tilize` guards landed after
that check: 36 discovered, 30 run under the model, the same 6 proven.) `BASELINE_TREE` in the gate records
the commit, so a future reader can tell how stale the hand-checked half of the
classification is. The guard *list* is not stale, because it is discovered —
only the hand-checked classification of each guard can be, and the pytest
tripwire fires the moment discovery and the record disagree.

## Two queues, not one: what rung 2 handed over, and what it did not

Landed 2026-08-04, and it is the first change in this file that exists because a
**previous section's reasoning was shown to be wrong**. Rung 2 finished with two
findings and one correction to how the whole model is described. All three are
here, and only one of them turned into code.

### The contention: DRAM is channel-limited, not link-limited

Rung 2 measured a Wormhole DRAM read sustaining **24.38 B/cycle** and a write
**24.53**, against a modelled 32. The instalment above noticed that 24.4 is not
a fraction of 32 — it is `dram.bandwidth.per_channel_gb_per_s: **24**`,
`isa_doc`, at the `clock` section's 1 GHz, agreeing to **2 %** — and argued that
the model's 76–77 % "link efficiency" for DRAM was an artefact of charging the
wrong queue's rate.

Standing against that was an argument written into `unit_costs.yaml` when the
DRAM latency landed, and it was not a throwaway line. It said: size dependence
is a bandwidth term, it is the *same physical serialisation* `noc.flit_bits`
already charges a packet on the wire, so charging it at the DRAM endpoint too
would bill one queue twice — and `dram.bandwidth`, well sourced and sitting
right there, was therefore **deliberately** unconsumed.

That argument was **wrong, and its error is worth naming precisely** because it
is a tempting one. The NoC link and the GDDR6 channel are not one queue seen
twice. They are two pieces of hardware, at two rates, in series: the wire moves
256 bits per cycle, the channel moves 24 GB/s, and the numbers are not equal.
A byte stream through two pipelined stages runs at the **bottleneck** stage's
rate, not at the sum of the two stages' times. So:

- charging `link_time + channel_time` really would double-bill, and that is what
  the old argument was defending against;
- charging `max(link_time, channel_time)` does not, and is the standard result.

The faster stage contributes only its pipeline *fill* — a constant, independent
of transfer size, and already inside the flat `access_latency`. Nothing in the
old argument distinguished the two cases; it treated "there is already a
size-dependent term" as sufficient reason not to have a second one.

Two further checks before touching anything, because agreeing with a measurement
is exactly the moment to be suspicious:

1. **Is 24 fitted?** No. It was in this file, at `isa_doc`, from the Wormhole
   DRAM tile page, before rung 2 existed and before anything was swept. The only
   thing this change adds is a unit conversion — 24 GB/s ÷ 1 GHz = 24 B/cycle —
   recorded as `dram.channel_serialisation`, `isa_doc_derived`, with the
   arithmetic written out and pinned by a test. Consuming a number that already
   sits in the table at the strongest rank, because a measurement independently
   landed on it, is the *opposite* of fitting.
2. **Does the direction hold?** Both. Reads and writes plateau at 24.38 and
   24.53 — 0.6 % apart — which is what one shared channel rate looks like and
   is why the term is charged once per request rather than split by action.

### How it is charged: the excess, once, at the endpoint

`DRAMEndpointNUI` already holds an arriving request for `access_latency`. It now
also holds it for

```
max(0, ceil(N / 24) - ceil(N / link_bytes_per_cycle))
```

which is the channel's serialisation **minus what the link already charged**.
The link's share is spent by `NUI._bandwidth_delay` when the packet carrying the
bytes is injected — the read's response leg, the write's request leg — so
subtracting it leaves the round trip paying `ceil(N / 24)` exactly once. `max(0,
…)` rather than a signed difference: an architecture whose NoC were the slower
of the two gets nothing from here, never a refund.

`data_length_bytes` rather than what is on the wire, because that field is the
transaction size on *both* legs of a read. So a read's bytes are billed when its
request lands and a write's when its data does, which is the same total either
way and needs no second hook on the response path.

**Blackhole gets nothing**, and the reason is a trap worth recording. Its
`dram.bandwidth` is `provenance: unknown` — BlackholeA0 has no DRAM tile
directory in the ISA docs at all — but the arch overrides are **deep-merged**,
so Wormhole's 24 is physically present under Blackhole's `unknown` node and a
consumer that read the number without looking at the provenance beside it would
launder one architecture's published figure into another's gap. `DramCostModel`
checks; `test_the_dram_channel_rate_is_exactly_its_own_derivation` asserts both
that the merge really does carry the number through and that the model declines
it. This is the same guard `noc_dataset_sweep._sourced_bandwidths` already
applies, for the same reason, and it is now the second place it has mattered.

That is not a gap wanting a scaled Wormhole number either. Rung 2 puts
Blackhole's DRAM at **47.1 B/cycle reading and 59.4 writing** against a 64
B/cycle link — 74 % and 93 %. Wormhole's two directions agree to 0.6 %, which is
what one channel rate looks like; Blackhole's differ by 26 %, which is not.

### What the DRAM residual did

The whole point, and it is the cleanest result in this file.

| Wormhole DRAM read | 64 | 128 | 256 | 512 | 1 K | 2 K | 4 K | 8 K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| measured | 397 | 405 | 405 | 421 | 437 | 478 | 565 | 733 |
| predicted (before) | 319 | 321 | 325 | 333 | 349 | 381 | 445 | 573 |
| predicted (after) | 320 | 323 | 328 | 339 | 360 | 403 | 488 | 659 |
| residual (before) | 78 | 84 | 80 | 88 | 88 | 97 | 120 | 160 |
| residual (after) | **77** | **82** | **77** | **82** | **77** | **75** | **77** | **74** |

| least-squares residual | before | after |
| --- | --- | --- |
| DRAM read | 79 + **10.03** cycles/KiB | 79 + **−0.65** |
| DRAM write | 129 + **7.47** | 129 + **−3.21** (111 + **0.02** without its known 64 B outlier) |
| implied model rate | 32.00 B/cycle | **23.95** against a measured 24.38 / 24.53 |

The size axis is **gone**: +10.03 cycles/KiB becomes −0.65, and the read
residual is flat at 74–82 cycles across a 128× range of transfer size. The
DRAM-write row's least-squares slope reads −3.21 only because of its 64 B
outlier (521 against its own 128 B column of 425, a visible break in an
otherwise monotone sweep, reported here rather than excluded for the same reason
rung 2 reported it); over 128 B – 8 KiB it is **+0.02 cycles/KiB**, which is as
flat as arithmetic gets.

The residual it settles on is the interesting part. **79 cycles** — and the L1
rows, which this change does not touch, sit at **77 and 80**. Rung 2 predicted
before looking that the residual would be a *constant* equal to one unmodelled
issuing-core path, the same in every row whatever the memory type. Before this
change DRAM disagreed with L1 by a slope; now four independent series agree on
one intercept to within 3 cycles. That is rung 2's own stated success criterion,
met on the axis where it had failed.

Two honesty notes. The model now very slightly **over**-charges the size term —
24 B/cycle modelled against 24.38 measured, so about 0.4 cycles/KiB — but the
+77 intercept swamps it and every Wormhole residual stays positive, so the model
is still a floor everywhere on that arch. And Blackhole is untouched by
construction, so its eight negative DRAM-write residuals (−12 to −28, recorded
in the instalment above) are exactly as they were.

### Measured, at one-cycle poll resolution

Only the Wormhole guards can move, and they move a little:

| Wormhole guard | model off | model on, before | model on, after | Δ from the channel term |
| --- | --- | --- | --- | --- |
| `matmulblock` | 9,781 | 14,320 | **14,672** | +352 (+2.46 %) |
| `transpose` | 6,812 | 8,477 | **8,649** | +172 (+2.03 %) |
| `matmulidx` | 7,113 | 9,529 | **9,683** | +154 (+1.62 %) |
| `reduce` | 7,109 | 9,349 | **9,479** | +130 (+1.39 %) |
| `sfpumath` | 19,290 | 20,367 | **20,456** | +89 (+0.44 %) |
| `softplus` | 17,845 | 19,829 | **19,871** | +42 (+0.21 %) |

| Blackhole guard | model on, before | model on, after |
| --- | --- | --- |
| `six` | 83,804 | **83,804** |
| `four` | 105,346 | **105,346** |

Blackhole identical **to the cycle**, which is what a provenance-guarded term is
supposed to look like from the outside. The Wormhole spread is the shape you
would expect: `matmulblock` and `transpose` stream the most DRAM per cycle of
compute, `softplus` the least.

### Is the ~10 % L1 read shortfall sourceable?

Rung 2's other finding, and the answer is **no** — with a real derivation
attempted and shown not to work, which is worth more than the finding would have
been.

The shortfall: Wormhole L1 reads sustain 28.5 B/cycle against a modelled 32
(89 %), writes 29.5–30.2 (92–94 %). Rung 2 refused to close this with a 0.897
efficiency factor, on the ground that it would be an `estimated` entry fitted to
the same 60 points that then validated it. **That refusal stands and is not
reversed here.** But a shortfall with a *documented* cause would be
`isa_doc_derived` rather than a fudge, so the question was whether the NoC's own
per-packet overheads account for it. The asymmetry is the test: reads lose ~11 %
and writes 6–8 %, so if framing is the cause the arithmetic should predict
roughly that split.

**What the docs turn out to say, and it is more than this file believed.** The
NoC page's first sentence is explicit:

> Each NoC transaction consists of one or more packets. Each packet consists of
> one or more flits (**exactly one header flit**, followed by up to 256 data
> flits). Each flit consists of exactly 256 bits (32 bytes).

and the same page's performance section adds the sentence that makes it a cost:

> The amount of *useful* throughput depends on the ratio of header flits to data
> flits; very short packets can use just a single header flit to transport 4
> bytes of data, whereas very long packets use one header flit and 256 data
> flits to transport 8192 bytes of data.

The request and response shapes are documented too, in the `NOC_CMD_CTRL` table:
a read is "a read request packet consisting of a single flit" plus a multi-flit
response; a write's acknowledgement is likewise "a write acknowledgement packet
consisting of a single flit"; atomics are single-flit both ways.

**So there is a per-packet overhead, it is sourced, and it is one flit. It is
also nowhere near big enough.** A Wormhole packet carries at most 8192 bytes =
256 data flits, so:

| | header cost | as a rate |
| --- | --- | --- |
| per packet | 1 flit | 1 cycle per 8 KiB |
| as a slope | | **0.125 cycles/KiB** |
| measured L1 read shortfall | | **3.94 cycles/KiB** (diff-axis), 4.30 (same-axis) |
| measured L1 write shortfall | | **2.65**, 2.83 |
| fraction explained | | **3 % of the read gap, 6 % of the write gap** |

At 64 KiB that is 8 header flits against 2048 data flits — **0.39 %** where 10.3
% is needed, a factor of 26 short. And the request leg contributes nothing at
all: a read's 8 single-flit requests and a write's 8 single-flit acks travel on
the *opposite* link from the one carrying the bytes, so they do not lengthen the
transfer. Worse for the hypothesis, **header overhead is direction-symmetric**
— 0.39 % each way, to two decimal places — so it explains none of the
read-versus-write asymmetry that was the reason to suspect it.

The docs also positively **rule out** three of the other candidates:

- **Credit traffic.** There is none described. Neither NoC tree contains the
  word "credit" or "backpressure" anywhere; the documented mechanism is VC
  arbitration ("if the two packets have the same virtual circuit number, then
  one packet will wait for the other"), which is congestion, which is
  `provenance: unknown`. The overlay has its own flow-control packets, on a
  path these measurements do not use, and their size is unpublished.
- **An efficiency caveat.** The DRAM page has one ("well-written software can
  expect approximately 92 %"); the NoC pages have none, and give no GB/s or
  B/cycle figure at all. Every bytes-per-cycle number in this file is derived
  from the flit width and the clock. There is no published 89 % to point at.
- **An L1 port asymmetry.** "Each NoC has four 128-bit connections to L1: two
  for reading and two for writing … This theoretically allows each NoC to read
  256 bits per cycle and write 256 bits per cycle." Symmetric, and matched
  *exactly* to the 256-bit router link with no headroom — which is suggestive
  of where the loss is, but it is the same on both sides and so predicts no
  asymmetry. The one remaining documented mechanism is **bank conflicts**,
  which the same page says the NoC's L1 connections can still suffer ("a bank
  conflict will occur if multiple ports try to access the same bank on the same
  cycle"), and which is unquantified, direction-blind, and needs the arbitration
  model that `noc.congestion` has always been `unknown` for.

**Verdict: a documented gap, not a fudge.** The header flit is real and is now
recorded — `noc.packet_framing`, `isa_doc`, alongside `noc.request_packet_flits`
— and it is *deliberately unconsumed*, which is a different statement from the
one `NocCostModel.packet_flits` used to make. That docstring said the docs "do
not say how many flits a header takes"; they do, and it was wrong. What stops it
being wired here is proportion: charging it takes a 32-byte semaphore poke's
modelled link occupancy from 1 cycle to 2 — a doubling, on the most common
packet in the tree — to close 3–6 % of the gap it was hoped to explain. That is
a whole-tree timing perturbation of its own and wants its own change and its own
gate run, not a ride-along on a DRAM instalment. The remaining 94–97 % has no
published explanation, and inventing one is the thing this file exists not to
do.

### There is no rung-3 dataset for Tensix compute, and that changes what "rung 2 climbed" means

The third finding, and the one that costs no code and matters most for how this
work should be described.

Both rungs climbed so far validate the **NoC and memory** path and nothing else.
Rung 1 is `dram.end_to_end_reference` against a hop model plus a DRAM term; rung
2 is 60 points of tt-metal's measured NoC dataset. Neither touches a Tensix
instruction. **The Tensix instruction costs — the matrix, vector, scalar, packer
and sync tables, which are the bulk of `six`'s charged cycles — have no external
check whatsoever.** They rest on provenance alone: the ISA docs' own
latency/throughput tables, read carefully, cross-checked against each other and
against the fidelity arithmetic, and never once compared to a measurement.

That is not a suspicion, it is a search. tt-metal 0.74 ships **no checked-in
dataset of measured Tensix compute cycle counts**, and the tree was gone through
looking for one:

- `tests/tt_metal/tt_metal/perf_microbenchmark/` contains exactly five
  checked-in data files, **all under `dispatch/`**:
  `pgm_dispatch_golden.json` (472 entries) and its Blackhole twin (460), plus
  three `benchmark_rw_buffer_*_golden.json` (56 / 44 / 180 entries). The first
  pair times `EnqueueProgram` with a deliberately empty kernel — it is pure
  command-queue overhead, a path tt-sim does not implement at all, since only
  direct `LaunchProgram` is supported. The second three are host↔device buffer
  `bytes_per_second`.
- `1_compute_mm/test_compute_mm.cpp` ships no reference cycles. Its ~20
  `golden_vec` references are **numerical-correctness** goldens (`pass &=
  (golden_vec == result_vec)`), and its only performance yardstick is an
  *analytic* roofline (`WH_FPU_BFP8_TFLOPS_PER_TENSIX = 2.05`) it compares a
  measured TFLOPS against at runtime. `1_compute_conv/` is an empty directory.
- Every other checked-in measured dataset in the tree is interconnect or host:
  fabric latency/bandwidth CSVs in ns, Ethernet EDM bandwidth in B/cycle, and
  the data-movement suite's `test_bounds.yaml`. The one structured
  measurement set of `noc_latencies.yaml`'s kind is `noc_latencies.yaml`.
- The **nearest miss** is worth naming because it is where a rung 3 would come
  from: `tt_metal/tt-llk/tests/` has a real per-LLK-op performance harness —
  `matmul_perf.cpp`, `eltwise_binary_fpu_perf.cpp`, `eltwise_unary_sfpu_perf.cpp`
  and friends, driven by ~20 `perf_*.py` scripts, reading hardware performance
  counters and the RISC-V wall clock, with run types like `L1_TO_L1` documented
  as "end-to-end pipeline cycles, unpack → math → pack". It *generates* exactly
  the numbers this file needs. It writes them to an untracked `perf_data/`
  directory, and **no snapshot is committed**; a grep for
  `golden|expected|baseline` across its Python tests returns nothing.
- The ttnn/model perf targets (`decoder_perf_targets_*.json`,
  `targets_test_perf_*.json`) are whole-op **durations in nanoseconds** over a
  full core grid with NoC traffic and dispatch folded in — not decomposable into
  per-instruction issue costs, and not cycles.

So ["What calibration would take"](#what-calibration-would-take)'s rung 2 should
be read as **"tt-metal's measured NoC dataset"** and not as "external
validation", and this file's status line should be read the same way. The
Tensix half of the model is sourced, internally consistent, and **untested**.
Producing the missing dataset means running the tt-llk harness on silicon, which
is hardware work and is being designed separately; nothing here should be taken
to have done it.

### What broke

Nothing, and the DRAM change is the second timing perturbation in this file to
cost **zero poll budgets**.

- Full suite **654 passed** (649 before, plus five: the channel rate's
  derivation, its Blackhole refusal, the excess-not-sum shape, an end-to-end
  large read that grows at 24 B/cycle on a real device, and the raw-table pin).
- `driver/tests/cost_model_gate.py` → **`RESULT: PASS`**, 36 guards discovered,
  30 run under the model, 6 proven, and **every poll-budget multiplier
  unchanged** from the instalment above: `six` 8×, `one`/`three`/`four`/
  `four-fp`/`nine`/`loopback` and `wh offline` 4×, the rest 2×, `bh dramtop` 1×.
  Zero dirty data READs anywhere.
- Wormhole offline replay **126/126 byte-identical** with the model off; `ruff
  check` and `ruff format` clean; **zero `estimated` entries**, still.

One test had to change shape rather than value, and it is worth naming because
it is a small piece of scope creep the term causes:
`noc_cost_model_test.test_a_semaphore_poke_no_longer_costs_what_a_tile_read_costs`
measures its 32 B-versus-2 KiB difference over a **DRAM** path, so it now
observes the NoC term *and* the channel term. It asserts them as two named
addends rather than one pinned number, which keeps it readable as a statement
about the NoC — but a future reader should know that the cleanest test of the
NoC bandwidth term alone would drive an L1→L1 path, and this one does not.

## The front end, measured: what may now be modelled and what may not

Landed 2026-08-05, and it is the first instalment in this file that changes
**nothing in the simulator and nothing in the tables' numbers**. What it changes
is the licence: for two years' worth of sections the front end has been the term
that made every other term unobservable, and it was un-measured. It is now
measured. This section says exactly what that buys.

The measurement is `perfbench/riscvbench` on a Blackhole card, 2026-08-05, two
runs, banked in `tt_sim/perf/datasets/`. The design, the method and the full
running record are in
[`riscv-front-end-benchmark.md`](riscv-front-end-benchmark.md); the findings
that are about the *chip* rather than about this model are collected in
[`docs/bh_arch.md`](../bh_arch.md). Only the consequences for the cost model are
here.

### The problem this was supposed to unblock, restated

Six of the nine Tensix backend units are wired. **Every one of them moved zero
simulated cycles.** The diagnosis has been the same every time and it is written
into this file in those words: *"the constraint is the un-modelled RISC-V front
end"*. `TensixFrontend.push_mop_instruction` is an unbounded list append and
`RV_TT_ISA.run` writes the rotated instruction word and returns in the same
tick, so **no Tensix unit can back-pressure the core that fed it**, whatever
occupancy the tables charge it. `tensixbench` says the same thing from the
measurement side: against tt-sim, every phase A probe of every unit at every
data format reads exactly 1.000 cycles per instruction, a number its own harness
calls *forced* rather than informative.

Two numbers now bound that gap quantitatively rather than rhetorically.

### The two numbers

**1. Issuing a `.ttinsn` costs the RISC-V core exactly one cycle, and it does
not depend on the target unit or on who else is issuing.**

```
probe          unit      t1      t2      t3
tt_nop         NONE   0.996   1.029   0.997
tt_sfpnop      SFPU   0.998   1.968   2.973
tt_setdmareg   THCON  0.996   1.969   2.972
tt_adddmareg   THCON  2.972   5.987   8.995
```

`TTI_NOP`'s backend unit is `NONE` and it is the only row flat in the thread
count. So what grows is the *shared backend*, not the per-thread push path.
The push is one cycle per core, full stop.

**2. At one thread the core outruns the backend by ~92 cycles over a
128-instruction burst; at three threads it does not outrun it at all.**
Phase Q times the same `ADDDMAREG` burst twice, once plain and once with
`ckernel::tensix_sync()` inside the timed region so it cannot return until the
pipe has drained. Cycles per instruction over n = 16…128:

| issuing threads | plain (core's view) | drained (work's rate) | back-pressured from |
| --- | --- | --- | --- |
| 1 | 2.22 | 3.000 | never, within n ≤ 128 |
| 2 | 5.66 / 5.78 | 5.920 | n between 64 and 128 |
| 3 | 8.99 / 8.95 / 9.22 | 9.000 | n between 32 and 64 |

**Both of these are the gap made quantitative.** The backend *is* doing the
work — ThCon's documented 3-cycle occupancy is measured three separate ways —
and the only reason no simulated cycle count moves is that no core is ever made
to wait.

### What may now be modelled

- **A one-cycle occupancy on the `.ttinsn` push itself.** It is measured, on
  three backend units, at three thread counts, in two runs, and it is
  independently what `riscv.ttinsn_fusion.dequeue_per_thread_per_cycle`
  documents. This is the least contestable term available.
- **Back-pressure from a shared backend unit onto the issuing core, at the
  unit's documented occupancy.** The 3.0× thread scaling on ThCon and SFPU is
  the ordinary shared-unit signature and it reaches the issuing core: a thread's
  own view of an `ADDDMAREG` is 2.972 cycles at one thread and 8.995 at three.
  A model in which a Tensix unit's occupancy stalls the pushing core would
  reproduce that; today's model, in which it cannot, reproduces 1.000 at every
  thread count.
- **The load/store and integer terms `cost.py` already charges, with one
  correction.** Silicon agrees with the table on the L1 store period (5.217
  against 5), on coalescing (0.999 against 1, and 5.2× against the un-coalesced
  probe), on core-local load latency (1.985 against 2), on `alu_forwarding`, and
  on the issue rate (0.999 against 1). Two rows are read differently than the
  model reads them, and both are *under*-charges:
  - `rv_load_chase` reads **8.098**, which is `l1_dcache_miss: >= 8` and not the
    `l1_dcache_hit: 2` the model applies to an L1 access on Blackhole.
    `rv_load_indep` at 1.742 confirms it from the docs' own throughput formula,
    which gives 1.750 at latency 8 and 1.0 at latency 2.
  - `rv_div` reads **33.001** on `0x12345678 / 3`, the `max` of the documented
    6–33 range, where `model.py` charges the low end of every bound.
  Neither is a table error — both table entries are correct and bounded — they
  are places where "charge every bound at its low end" is a long way from what
  this part does with these operands.

### What remains unknown, and must not be modelled as if it were not

- **The Tensix instruction queue's depth in entries.** Phase Q does not give
  one. At one thread the core had not stopped running ahead by the longest burst
  the benchmark emits, so the absorbed backlog is still growing where it would
  need to be an asymptote. The in-flight work divided by the per-thread service
  rate gives 31–34 instructions at one thread and 12–14 per thread at three —
  the spread is between two estimators, which is itself a measure of what this
  resolves. *Consistent with* a shared resource of roughly 30–40 entries and
  equally consistent with several other readings. **A model that hardcodes a queue depth
  today is fitting to one run's arithmetic**, and it should wait for the longer
  burst sweep described in the benchmark document.
- **Any sub-one-cycle push.** `riscv.ttinsn_fusion` documents four-way fusion
  and the measurement says it does not happen; the entry now carries a
  `contradiction` field with the "different quantities" escapes checked. No
  front-end model may assume the core can enqueue faster than one instruction
  per cycle.
- **The branch mispredict rate.** `cost.py` charges nothing for branches, on the
  stated grounds that the number of mispredictions is unknowable. That is now
  *supported* — `taken − not taken` is −0.047 — rather than merely argued, and
  it should stay unchanged. Note carefully that this says nothing about
  `branch_mispredict_bubble: 4`, which is the *size* of a mispredict and remains
  untested.
- **Instruction fetch.** There is a real cliff (~25 %, between a 4 KiB and an
  8 KiB loop body) and no published cache size or miss cost to hang a model on.
  It is recorded in [`docs/bh_arch.md`](../bh_arch.md), not in the tables. See
  below.
- **BRISC and NCRISC.** Every probe ran on TRISCs. Nothing here says what
  happens when a data-movement kernel and a compute kernel contend.
- **Wormhole.** Nothing. Multiply, the mispredict bubble, the L0 d-cache and the
  coalescing store queue all differ between the two architectures, so not one
  figure transfers.

### Is the ~377-cycles-in-flight figure solid enough to build on?

It is the number the previous instalment quoted as this gap's headline, and it
came from **tt-sim, not silicon** — the simulator, with the cost model on, at a
128-instruction burst: 128 cycles seen by the core against 505 for the work.
Silicon's equivalent is **~92 cycles**, about 31 instructions, and the two are
not comparable in the way the phrasing invites:

- tt-sim's 377 is what an **unbounded** queue does. It is a lower bound on
  nothing and an upper bound on nothing; it grows linearly with the burst
  because the list has no end, and it would have been 3,000 at a 1,024-burst.
- Silicon's 92 is what a **real** queue does at the longest burst this benchmark
  reaches, and it is still growing at that point — so it is a **lower bound on
  the hardware's capacity to absorb** and nothing more.

**So: build on the direction, not on the figure.** What is solid is that the
in-flight work is *large and grows with the burst* on both, i.e. the gap is real
and is not a rounding artefact; and that on silicon it *stops* growing once
enough threads are issuing, which tt-sim can never reproduce. What is not solid
is any specific capacity. A model calibrated to 377 would be calibrated to the
simulator's own unboundedness, which is the thing being fixed.

### The discipline problem this surfaced, stated and not solved

The instruction-fetch cliff is the first number either benchmark has produced
that **no document anywhere gives**. `riscv.instruction_fetch` publishes a fetch
period and its own note says the cache miss cost is not published; no cache size
is published either, in the ISA docs or in ttsim or tt-metal.

The provenance convention above is unambiguous about what to do with a
measurement: it enters as `corroboration`, it never becomes an entry's
authority, and there is no `measured` rank — for three reasons this file and
`tensix_instruction_costs.yaml` both spell out, all of which still hold. What
neither had had to face is the case where **there is no entry to corroborate**.
A measured-only quantity has, under the current rules, *nowhere to go*.

Three moves were available. Inventing a table entry so the number has a home
would make a measured value look sourced, which is precisely what "WHY THERE IS
NO `measured` PROVENANCE" exists to prevent. Leaving it in a YAML comment makes
it neither findable nor reviewable. Writing it somewhere that is explicitly not
a cost table keeps both the number and the discipline.

The third was taken: [`docs/bh_arch.md`](../bh_arch.md). It loads into no code,
it competes with no table, it says at the top what one card and one run are
worth, and where a table entry *does* exist it points at that entry's
`corroboration` rather than restating the number as if it were independent.

**The tension is real and this is a first answer, not the answer.** The model
can now measure things it has no way to record. One number and a new document
are proportionate. If it becomes ten — and the phase F sweep alone could produce
three more — the question of whether the tables need a rank, or a companion file
with a schema and its own tests, comes back. It should be answered deliberately
then rather than by drift now. Nothing in this instalment pre-empts it.

### What changed in the repository

- Two datasets tracked in `tt_sim/perf/datasets/`, each with card, firmware, KMD
  (**2.10.0**, against 2.9.0 on the `tensixbench` datasets — the driver moved
  between campaigns), flags, row count and **per-phase validity** in its own `#`
  header. `riscvbench-blackhole-blocks8.csv` is tracked *because every one of
  its phases failed the benchmark's validity gate*: that is what puts a floor
  under `--blocks`, and it is invisible from a run that passed.
- Seven `corroboration` fields, all prose, all under `arch_overrides.blackhole`
  because a Blackhole measurement is not evidence about Wormhole. Plus one on
  `THCON.ADDDMAREG`, now the only entry in either cost file measured by two
  different benchmarks.
- One `contradiction` field, on `riscv.ttinsn_fusion` — a new key, for the case
  the provenance convention describes as "record both and say so" and had no
  vehicle for. Pinned by `costs_test.py` to exactly one entry, held to the same
  runs-and-parts bar as a corroboration, and required to leave the contradicted
  entry's numbers, provenance and source intact.
- Phase Q's read-out rebuilt in `riscv_bench_sweep.py` after silicon showed its
  control subtraction was resting on a false premise. The benchmark's validity
  gate was **not** touched.
- **Zero changes to any cycle count, to `PROVENANCE_RANK`, or to the
  simulator.** Zero `estimated` entries, still.

## The congestion step, sized: what a second campaign changed and what it did not

Two more Blackhole runs, on the corrected geometry the previous instalment's
`--dump-grid` change produced. **Zero coordinate mismatches and zero invariant
complaints in both**, which is what the previous campaign — eight mismatched
rows out of 79, four of its eight shared-link counts wrong — could not say. The
measurements are banked in `docs/bh_arch.md` §4 and the datasets in
`tt_sim/perf/datasets/`; this section is the running record of what they are
allowed to change here, and the answer is **one verdict and no cycles**.

The two runs are two *different plans* and are never averaged:

| | flows | experiments | own verdict |
| --- | --- | --- | --- |
| `nocbench-blackhole.csv` | 87 / 55 runs | all six | **`CONGESTION MEASURED`** |
| `nocbench-blackhole-sizes.csv` | 96 / 48 runs | `shared` only, six sizes | `INVALID` — no controls, by construction |

Where they overlap — 64 B and 16384 B at 0 and 1 shared links — they agree to
**0.8 cycles/tx**, and that is quoted as an agreement between two runs rather
than folded into one number.

### The delta is the occupancy, and that is a different kind of fact from a step

The previous instalment banked "a step at the first shared link, not a slope"
and stopped there, on the grounds that a step is not a term a cost model can
consume. The size sweep says what the step *is*:

| transaction | occupancy = bytes ÷ 64 | 0 shared | 1 shared | delta | delta ÷ occupancy |
| --- | --- | --- | --- | --- | --- |
| 64 B | 1 | 39.9 | 39.8 | −0.1 | — |
| 512 B | 8 | 39.8 | 39.8 | 0.0 | — |
| 2048 B | 32 | 40.2 | 68.3 | 28.1 | 0.88 |
| 4096 B | 64 | 72.2 | 135.4 | 63.1 | **0.99** |
| 8192 B | 128 | 137.8 | 262.3 | 124.5 | **0.97** |
| 16384 B | 256 | 268.4 | 518.8 | 250.4 | **0.98** |

**Above the issue loop, a second flow on a shared link costs exactly one
transaction's link occupancy; below it, nothing.** The boundary sits between
512 B and 2048 B, which is where `bytes ÷ 64` reaches the ~39.8-cycle interval
at which the issuing RISC-V can produce transactions — so the shape is not a
threshold anybody has to choose, it is `max(occupancy, issue_interval)` falling
out of a queue. And the same arithmetic predicts the *contended* column:
`max(2 × occupancy, 39.8)` gives 39.8 / 39.8 / 64 / 128 / 256 / 512 against
39.8 / 39.8 / 68.3 / 135.4 / 262.3 / 518.8 measured — every point within
~7 cycles, a residual that does not grow with size and is the same per-packet
overhead the uncontended column carries.

That is the whole of it, and it is worth being explicit about how little is
left over: there is no fitted parameter anywhere in the paragraph above. The
64 comes from `flit_bits: 512` at `throughput_flits_per_cycle: 1`, both
`isa_doc`; the 39.8 is measured but is a property of the *benchmark's* issue
loop, not of the hardware term.

### So: can it be wired, and at what provenance?

**Yes to expressible, and the provenance is better than anyone expected.**
Taking the two questions in order, because they have different answers.

**Is there a term?** Yes, and it is not `noc.congestion`. The quantity is
`noc.hops.router_to_router.throughput_flits_per_cycle: 1` — "one flit (256
bits) per cycle per axis", `provenance: isa_doc`, recorded in this table since
the first instalment and **spent nowhere**. The model already charges exactly
this occupancy once, at the injecting NIU, in `NUI.claim_injection_port`. The
term the measurement asks for is the *same* charge on each router-to-router
link a packet crosses:

```
for each link on the packet's route:
    wait = max(0, link.free_cycle - now)
    link.free_cycle = now + wait + serialisation_cycles(payload_bytes)
```

**This is not the per-link flow census the previous instalment said was
needed.** That reading — "the maximum over the links on a route of the number
of concurrent saturating flows crossing it" — described the *answer* rather
than the mechanism, and the answer needs a scheduler. The mechanism is one
watermark per link, which is the same object `_tx_free_cycle` already is, and
"saturating" is not a predicate anyone has to evaluate: a flow that under-uses
a link never finds it busy, which is precisely why 64 B and 512 B read zero.
The previous instalment's obstacle was overstated and this corrects it.

**At what provenance?** `isa_doc_derived`. **No new number enters the tables**,
so nothing here is `vendor_source` and nothing is `estimated` — the silicon
goes in as a `corroboration` on `arch_overrides.blackhole.noc`, which is now
one of eight in `unit_costs.yaml` and is pinned by `costs_test.py` like the
rest. That is a materially different position from the previous instalment's,
which expected any congestion term to be `vendor_source` at best.

**Is it wired? No.** Not because a number is missing but because three
structural things in `tt_sim/network/tt_noc.py` are, and they are named here so
that the next instalment is a build rather than a rediscovery:

1. **There is no per-link state, anywhere.** `send_to` computes a flight time
   from a hop *count* and hands the packet to the destination; no router, no
   link and no shared timeline exists between two NIUs. The registry has to
   live on the device, not on an NIU, because the whole point is that two NIUs
   contend on it.
2. **The network layer knows a hop count but has never committed to a hop
   order.** A link's identity needs one. Dimension-ordered X-then-Y on a
   directional torus is what the docs describe and what
   `noc_congestion_plan.route_links` already implements and tests — but it is
   implemented in `tt_sim/perf/`, for planning an experiment, and promoting it
   into the network layer makes the routing order load-bearing for every
   simulated cycle for the first time.
3. **Multicast would over-charge, and it is the tree's most common packet.**
   tt-sim models a multicast write as N unicasts; claiming link occupancy once
   per destination would invent serialisation the hardware does not have, on
   the launch-message path every tt-metal program uses. That is the exact
   over-charge `claim_injection_port` was given its odd signature to avoid, and
   it has to be solved before a per-link charge can be turned on rather than
   after.

None of the three is a research question and none of them is small. Wiring the
term is a whole-tree timing change — it is inert for any single flow, because
the injection port already rate-limits at exactly the link rate, but it is not
inert for `six` or for anything multi-core — and under this file's own rule a
whole-tree timing change wants its own instalment and its own
`driver/tests/cost_model_gate.py` run rather than a ride-along on a
measurement instalment. **The honest state is: the quantity is measured, the
number is already sourced, and the model now has somewhere to put it and does
not yet have the plumbing to get it there.** That is a strictly better position
than "the model has nowhere to put it", and it is still not a cycle changed.

### Virtual channels: the question the banked result could not answer

The previous instalment left this open, and its bidirectional form had hung a
card. The unidirectional redesign — two writers on one shared link, flow A
pinned at `NOC_UNICAST_WRITE_VC` (1), flow B swept 0–3 — ran, and the answer is
clean:

| flow B's VC | cycles/tx at 16 KiB |
| --- | --- |
| 0 | 520.2 |
| **1 (same as A)** | **530.4** |
| 2 | 519.9 |
| 3 | 519.9 |

**Different virtual channels do not avoid the halving.** All four points are
~1.94× the 268.4 that the same transaction costs with no shared link. Sharing a
channel costs a further **2.0 %** — 10.5 cycles, against three different-channel
points that agree to 0.3 and a 0.5-cycle noise floor, so it is real and not
noise.

That settles the shape of the term, which is why the experiment was worth
running before proposing one: **it is per-link, not per-VC-per-link.** It also
narrows what `noc.congestion` is still `unknown` *for*. The docs' one named
mechanism — "if the two packets have the same virtual circuit number, then one
packet will wait for the other" — is confirmed to exist and measured at one
fiftieth of the effect. The other 98 % was never congestion in the docs' sense
at all; it is the published link rate, arriving somewhere the model was not
spending it.

### The two overlap failures were one tile's clock, and the harness now says so

Two of the main run's 25 multi-flow runs and six of the size sweep's 48 read
overlap **0.00**, which is the harness refusing to call a flat reading a
result. All eight are the same point and all eight have physical **(11, 2)** as
flow 1's master, out of 19 master tiles used. It is not placement, not the
rendezvous and not the harvested grid: that tile's
`RISCV_DEBUG_REG_WALL_CLOCK` keeps a different **epoch**, +1,143,914,613
cycles, reproducing across all eight runs to a spread of 7 and across a device
re-open to ±4. `docs/bh_arch.md` §4.4 is the entry; the mechanism is not
established and needs the card.

The fix in `noc_congestion_sweep.py` is `clock_skew_report`, and its design is
the point rather than its effect. Correcting a cross-core timestamp per **run**
would assume the overlap it is used to check — perfectly circular. It corrects
per **core**, and only when two independent conditions hold:

* the offset **reproduces** across runs, to within 1000 cycles. A flow that
  genuinely started late cannot start late by the same number of cycles twice;
* the offset is **impossible as a delay** — larger than the span of the file's
  own stamps. A flow cannot start later than the program ran.

The second condition exists because the first alone is not enough: a rendezvous
that always released flow 1 late would reproduce too, and
`test_a_delay_that_fits_inside_the_session_is_not_an_epoch` is that failure
mode held down. With the correction the main run's median overlap is 1.00, none
is below 0.5, and its verdict is `CONGESTION MEASURED` rather than `INVALID`.
The simulator run in `perfbench/nocbench/src/` still reads `NO CONGESTION
EFFECT`, unchanged, which is the forced null it has always been.

### What changed in the repository

- Three datasets tracked in `tt_sim/perf/datasets/` — the two silicon runs and
  the card's own core map — each with its harvesting, its campaign and the
  (11, 2) clock offset in its own `#` header.
  `python3 -m tt_sim.perf.noc_congestion_sweep` now defaults to the main run,
  so the analysis reproduces with no hardware.
- One `corroboration`, on `arch_overrides.blackhole.noc`: 64 B/cycle per link,
  measured twice off two different resources (the NIU's own port and a
  router-to-router link), in units the table never sees.
- `noc.congestion`'s note rewritten, because its stated reason for staying
  `unknown` was superseded rather than merely extended. It now says which part
  is expressible, at what provenance, what the three blockers are, and that VC
  arbitration is what the `unknown` is still *for*.
- `clock_skew_report` and a 32-bit-safe `overlap_report` in
  `noc_congestion_sweep.py`, plus `points` on the `shared` result so a
  `SATURATING` fit's consumers get the steps rather than the line through them.
- **Zero changes to any cycle count, to `PROVENANCE_RANK`, or to the
  simulator.** Zero `estimated` entries, still.

## The congestion step, wired: one number, spent a third time

The previous instalment ended with "the quantity is measured, the number is
already sourced, and the model now has somewhere to put it and does not yet
have the plumbing to get it there", and named three build items. They are
built. **No number entered any table**, `noc.congestion` is still
`provenance: unknown`, and `PROVENANCE_RANK` is untouched — what changed is
where `noc.hops.router_to_router.throughput_flits_per_cycle: 1` (`isa_doc`,
in the table since the first instalment) is *spent*. It was charged once, at
the injecting NIU. It is now charged on each router-to-router link a packet
crosses, which is the same occupancy on the resource the silicon measurement
resolves.

### The three blockers, and what each turned into

**1. There was no per-link state anywhere, and it could not go on an NIU.**
`NocLinkRegistry` is one free-cycle watermark per link — structurally the same
object `NUI._tx_free_cycle` already was, differing only in *ownership*. It
lives on `TT_Device`, one instance per NoC, created before any tile is
registered and handed to each NUI by `_register_tile_internals`. That is the
single fan-out point, so a worker the wire bridge materialises later
(`add_tensix_tile`) joins the same registry rather than quietly getting a
private one and ceasing to contend with anything — which has its own test,
because it is the failure mode that would look like the term simply not
working.

Two registries, not one: NoC 0 and NoC 1 are separate networks whose links are
named by the same tuples, and a shared registry would have them arbitrating
against each other.

**2. The hop *order* is now the network layer's, and there is exactly one of
it.** `noc_route_links` sits next to `noc_hop_count` — dimension-ordered X then
Y on a directional torus, a link named by the router it leaves and the axis it
leaves on. The planner's `route_links` **is** that function, by assignment, and
`test_the_planner_routes_with_the_simulators_own_function` asserts the
identity rather than the agreement. Two implementations of "dimension-ordered"
would pass every test anyone thought to write and still be free to disagree
about which of two links a packet crossed first; a `shared_payload_links`
column counted one way against a model that contends the other way is a
measurement of a different machine.

This does make routing order load-bearing for every simulated cycle for the
first time, and the honest statement about that is that the *count* has been
load-bearing since the hop model landed and the order adds no new degree of
freedom: `len(route_links(a, b)) == noc_hop_count(a, b)` on both architectures,
pinned, and the flight time a packet is charged is computed from the same pair
of coords in the same NoC's own space as its route.

**3. The multicast claims a tree, once.** `handle_multicast_write_transfer`
already claimed the injection port once for the whole fan-out; it now
de-duplicates the routes to every destination into the union — which for
dimension-ordered routing *is* the multicast tree — and claims that once, with
first-appearance order preserved so the walk goes outwards from the NIU as the
packet does. For the 4×3 rectangle in the test that is 12 links where twelve
unicasts would have claimed 42, and the launch-message path every tt-metal
program uses is the reason it had to be solved before the charge was turned on
rather than after.

### Inert for a single flow, which is why it may be turned on at all

The property that stops this being a second charge for the same bytes: a
flow's packets leave its injection port exactly one occupancy apart, so they
reach every link on their route one occupancy apart and never queue behind
themselves. Eight 2 KiB packets over a ten-hop route make eighty link claims
and wait zero cycles. Only *another tile's* traffic costs anything.

The wait is cumulative along the route — a packet held at one link reaches the
next one later — because the delay has to reach the packet's *arrival*, not
just the link. Propagation between links is deliberately not added to that
walk: the per-hop latency is already charged once by `NocCostModel`, and adding
it here would double-count a published number to no effect, since the
steady-state answer is a throughput and not a phase.

### Measured: the card's shape, twice, and no fitted parameter

**At the NIU registers**, with a 40-cycle issue loop standing in for the
benchmark's 39.8, two flows on Blackhole whose payload routes share one row-2
link:

| transaction | occupancy | 0 shared | 1 shared | delta | delta ÷ occupancy | card's ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 64 B | 1 | 50.0 | 50.0 | 0.0 | — | — |
| 512 B | 8 | 50.0 | 50.0 | 0.0 | — | — |
| 2048 B | 32 | 52.5 | 75.0 | 22.5 | 0.70 | 0.88 |
| 4096 B | 64 | 75.0 | 140.0 | 65.0 | **1.02** | 0.99 |
| 8192 B | 128 | 140.0 | 267.5 | 127.5 | **1.00** | 0.97 |

Same shape as `docs/bh_arch.md` §4.2: a step at the first shared link, sized
like one transaction's occupancy, **exactly nothing below the issue loop**, and
no threshold anybody chose — a flow that under-uses a link never finds it busy.

**And end to end**, through `perfbench/nocbench` — a real tt-metal program, a
real kernel, the wire bridge and the whole simulated NoC — against the archived
run of the *same plan* from before the term existed:

| | before | after |
| --- | --- | --- |
| hop line | `608.3 + 8.95 × hops` (r² 1.00) | identical |
| `size` control | 69.2 / 70.0 / 108.5 cycles/tx | identical |
| `readport` control | 1.48× | identical |
| `shared`, 512 B | 70.0 → 70.0, `FLAT` | identical |
| `shared`, 4096 B | 108.5 → 108.5, `FLAT` | 108.5 → **167.9**, `SATURATING` |
| verdict | `NO CONGESTION EFFECT` | **`CONGESTION MEASURED`** |

+59.4 cycles per transaction against an occupancy of 64, and **every
uncontended figure in the file byte-identical** — which is the inertness claim
above, measured rather than argued. `readport` staying at 1.48× is the sharpest
of those: two masters reading one subordinate contend on that tile's injection
port *and* on the first link out of it, and the model charges only the port,
because packets leaving one NIU are already spaced by it. On silicon those two
resources are one reading; here they are separable, and this says which one
tt-sim is spending.

A second run at 32 transactions per flow sweeps the shared-link count:
**75.1 / 132.1 / 137.8 / 138.0** cycles/tx at 0 / 1 / 2 / 3 shared links. The
1-to-3 span is 6 cycles against a 57-cycle step — 10 %, converging as the ramp
amortises (it was 22 at 8 transactions per flow) and the same order as the
card's own 6 % residual over its seven points. **A step, not a slope.**

The arithmetic check from the previous instalment survives contact:
`max(2 × occupancy, floor)` predicts the contended column at 128 against 140
measured at 4096 B and 256 against 267.5 at 8192 B, with the same per-packet
overhead the uncontended column carries. Nothing was fitted; the 64 B/cycle is
`flit_bits: 512` at `throughput_flits_per_cycle: 1`, both `isa_doc`.

### The gate: PASS, and byte-identical, because the term never fires in-tree

`driver/tests/cost_model_gate.py` → **`RESULT: PASS`**, 39 guards discovered,
33 budget-independent guards run under the model, 6 budget-dependent ones
proven, 1 excluded. Every proven guard came out clean at **the same poll-budget
rung, with the same data-READ counts, as the unmodified tree** — checked by
running the proof stage in a `git worktree` at `HEAD` and diffing, rather than
against the table earlier in this file, which is two instalments old.

`six` needs 8×, `one`/`three`/`four`/`four-fp`/`nine`/`loopback`/`wh offline`
4×, the rest 2×, `bh dramtop` 1×. Not one of them moved.

**The reason they did not is worth stating rather than being pleased about**:
the term does not fire on any in-tree workload. Instrumenting the registries
over `blackhole/six` — the heaviest multi-core guard in the tree — gives 3,960
link claims and **zero waits**. Same for `matmulblock` on both architectures.
These workloads' concurrent traffic is small packets spread thin; nothing in
the tree puts two saturating flows across one link, which is exactly the
configuration `nocbench` had to be written to create. So the gate's verdict
here is "the term is value-safe and costs nothing where nothing is contended",
and the evidence that it is not dead code is the nocbench run above, not the
gate.

With `TT_SIM_COST_MODEL` unset, **exactly nothing happens**: `send_to` takes
the same `model is None` branch it always did, `route_links_to` returns `()`,
and the registry is constructed and never touched. There is a test that asserts
all three counters stay at zero.

### What broke

Three pieces of prose that were true when written and became false the moment
the term was wired, all of them saying "against tt-sim a shared-link reading
MUST be flat":

- `noc_congestion_sweep`'s module docstring and its `NO CONGESTION EFFECT`
  verdict text, both rewritten — a flat simulator reading is now a reading like
  any other, and an operator being told it is forced would mis-file a real one.
- `HYPOTHESES["shared"]` in `noc_congestion_plan`, which is a **pre-declared
  prediction** and is therefore *not* edited. It gained a dated addendum
  underneath saying which sentence was superseded and that the prediction about
  the card is untouched. The card's answer was `SATURATING` before this model
  was built to match it, and the ordering is the whole reason the pre-declaration
  is kept in the source.
- `perfbench/nocbench/README.md`'s two "forced flat" passages and its "What the
  simulator run shows" section.

Nothing else broke. 983 tests pass (967 before, 16 added), ruff is clean, and
`unsourced()` still returns the same entries — `noc.congestion` among them.

### What changed in the repository

- `NocLinkRegistry` and `noc_route_links` in `tt_sim/network/tt_noc.py`;
  `NUI.route_links_to` / `claim_route_links`; `send_to` and `_bandwidth_delay`
  gained a `link_wait`; the multicast path claims its tree once.
- `TT_Device.noc_link_registries`, handed out in `_register_tile_internals`.
- `tt_sim/perf/noc_congestion_plan.route_links` is now the simulator's
  function rather than a second copy of it.
- `tt_sim/network/noc_link_congestion_test.py` — 16 tests: one per blocker,
  the inertness property, the opt-in, the registry's own arithmetic, and the
  three measurement shapes (step, absence below the issue loop, flatness
  beyond the first link).
- Two archived simulator runs in `perfbench/nocbench/src/`, both self-describing
  and both reproducible with no hardware: `nocbench-blackhole-sim-links.csv`
  (the certified plan, with controls) and `-flat.csv` (the shared-link sweep,
  controls-free and `INVALID` by construction, like the card's own size sweep).
- **Zero new numbers, zero `estimated` entries, `noc.congestion` still
  `unknown`** — and it is still `unknown` *for* virtual-channel arbitration,
  which §4.3 measured at one fiftieth of this effect and which nothing here
  models.

## The front end, bounded: the first mechanism the whole Tensix half was waiting on

Landed 2026-08-06 — ROADMAP item 1, and the change ["The front end,
measured"](#the-front-end-measured-what-may-now-be-modelled-and-what-may-not)
licensed. `TensixFrontend` was an unbounded list append and a `.ttinsn` store
returned the same tick, so no Tensix unit could ever back-pressure the core
that fed it. Now a thread's frontend path is **bounded**: a core-facing push
into a frontend holding `CORE_PUSH_INFLIGHT_BOUND` instructions (MOP + replay
+ wait-gate FIFOs) returns `MemoryStall`, and both push paths — the `.ttinsn`
extension and a plain `sw` to the push buffer, which already had the
`MemoryStall → PEStall` plumbing from the mailboxes — stall the core on the
same instruction with the PC unmoved. The mechanism is **active regardless of
`TT_SIM_COST_MODEL`**, because it is a correctness property first: on silicon
a thread's FIFO fills behind a permanently blocked backend instruction and
the core wedges; tt-sim used to run the kernel to completion past the wedged
unit (the `UNPACR_NOP` acquire-without-release case).

**The bound is a mechanism parameter, not a calibration, and its comment says
so at length.** Blackhole silicon absorbs ~31–32 instructions at one issuing
thread — a *lower bound*, still growing at the longest burst run
(`riscv-front-end-benchmark.md`, the untimed-drain estimator) — so the bound
is 64: safely above the measured floor, in the model's charge-at-the-low-end
direction (a too-small bound would invent back-pressure; a too-large one only
fails to model some), and finite, which is the whole point. When the longer
burst sweep runs on a card, calibrate it there; until then no depth in
entries is claimed anywhere.

### Exactly two cost terms, both already licensed, no new table entries

With `TT_SIM_COST_MODEL=1` the two terms the measurement instalment licensed
become observable at the core, and nothing else was added:

1. **The `.ttinsn` push costs one cycle per core.** Structural rather than
   charged: the core is single-issue and the frontend dequeues one per thread
   per cycle, so an uncontended push stream retires at exactly 1.000 —
   `test_a_ttinsn_push_costs_one_cycle_at_the_core_with_the_model_on` pins it
   against silicon's `tt_nop` at 0.996–1.029 across thread counts.
2. **A backend unit back-pressures the issuing core at its documented
   occupancy.** The occupancy machinery already existed (`occupy_for`, issue
   refusal, wait-gate retry); the bound is what lets it reach the core. A
   sustained `ADDDMAREG` burst now costs the issuing thread **3.000
   cycles/instruction** in the steady state where silicon measures 2.972/2.973
   on two instruments — and read 1.000 before, the number `tensixbench`'s
   harness printed as *forced*. Three threads sharing the 1-IPC SFPU each
   converge on ~3 cycles/instruction, silicon's 0.998 / 1.968 / 2.973 shape.

No cost-table entry was added or changed; the YAML is untouched.

### Two mechanism fixes the arithmetic forced

- **The occupancy hold now runs from acceptance, not from retire.** Backend
  units tick before the wait gates within a cycle, so an accepted instruction
  retires one tick later — and arming `occupy_for` at the retire tick made
  every multi-cycle occupancy cost one extra cycle at the issuing thread: 4
  for ThCon's documented 3, where silicon reads 2.97. That was invisible while
  nothing could observe a unit's rate from the core, and it is over-charging,
  the direction the floor policy forbids. `clock_tick` now arms from
  `cycle_num - 1` (the batch's acceptance cycle); the pinned windows in
  `backend_cost_model_test` / `matrix_cost_model_test` moved by exactly one
  cycle each and say why.
- **Single-slot units grant round-robin under contention.** The wait gates
  tick in fixed thread order, so once a refused thread can stall its core, a
  thread sustaining one instruction per cycle at a shared unit would win the
  slot every cycle and *starve* the other threads' cores for ever — a deadlock
  no silicon has, and the difference between per-thread 3.0× and 1×/∞/∞.
  `TensixBackendUnit.issueInstruction` now yields a free slot to a
  less-recently-granted waiting thread (safe without timestamps because a
  refused wait gate re-offers every cycle). The multi-slot units (config,
  sync, misc) keep their own published acceptance rules unchanged.

Folded in from the same ROADMAP item: **`[UNIT WEDGED]` is now a raise**
(`UnitWedgedError`), after printing its full report. The survey stands — 41
workloads plus every `pipestall` configuration, no unit ever ends a launch
blocked — and with the core now able to stall for ever, most wedges never
reach it: the *global* watchdog reports the deadlock, which is the
silicon-matching behaviour. The terminal raise remains the path for a kernel
with no further work behind the blocked instruction.

### Measured, at one-cycle poll resolution: the totals did not move, and why that is the right answer

`six` (the 128³ bf16 matmul), pump-to-DONE re-measured at one-cycle
resolution, before and after this change:

| | model off | model on |
| --- | --- | --- |
| before (2026-08-06 tree) | 17,968 | 74,604 (64,304 pumped) |
| after | **17,968** | **74,604 (64,304 pumped)** |

PCC unmoved at 0.9982. Zero movement, model off *and on*, and both zeros are
informative. Model off: the bound never bites on a healthy stream — every
Wormhole byte-identical replay passes unmodified, so no guard needed
converting and ROADMAP item 3 is untouched. Model on: a launch's completion
was already gated on the backend draining (`CoprocessorDoneCheck` holds the
end-of-thread sync until the coprocessor is done), so the *total* was already
charged; what the bound changes is when the **core** retires its
instructions, i.e. what a kernel's own timed region sees. That is precisely
the quantity `tensixbench` phase A measures — the row that read a forced
1.000 against tt-sim now reads the unit's occupancy, which was the entire
point of ROADMAP item 1. The instalment that makes a *total* move is item 5
(unpacker and mover occupancy), which this unblocks: those units' costs land
on the dataflow path, not behind a drain the total already waits for.

### The gate

Run whole, model on: **PASS** — 954 unit tests under the model, all 33
budget-independent value guards (`six` PCC 0.9982, both `pipestall`s with the
stall hint behaving, `twolaunch` clean), and every budget-dependent guard
proven completely clean on the poll-budget ladder: `dramtop` at 1×,
`blackhole/two` and both offlines at 2–4×, the eleven Wormhole example traces
at 2–8× (`six` the 8×, as the heaviest compute), `wormhole/one` excluded as
always (its trace is proven via `wormhole/offline`). Model off: all 954 unit
tests, all 38 driver pytest guards (including every byte-identical Wormhole
example replay, unmodified), all 26 Blackhole value guards.

### What changed in the repository

- `tt_sim/pe/tensix/frontend.py` — `CORE_PUSH_INFLIGHT_BOUND` (the documented
  uncalibrated bound) and the refusing `TensixFrontend.write`.
- `tt_sim/pe/rv/isa/tt_isa.py` — a `.ttinsn` store whose write returns
  `MemoryStall` returns `PEStall`; the `sw` path already did.
- `tt_sim/pe/tensix/backends/backend_base.py` — acceptance-anchored occupancy
  arming; round-robin grant under contention.
- `tt_sim/device/deadlock.py` — `UnitWedgedError`, raised by the terminal
  wedge check after its report.
- `tt_sim/pe/tensix/frontend_backpressure_test.py` — 9 tests: the bound, the
  never-refused internal pushes, the PEStall translation, the ROADMAP-named
  dvalid-twice-blocks-the-core case, and the licensed arithmetic (push at
  1.000, ThCon at 3.0, model-off control at 1.0, three threads at ~3× each).
- **Zero new cost-table entries, zero `estimated` provenance, no queue depth
  claimed in entries anywhere.**

## The three silicon-backed RV cost fixes: two charges move, one deliberately does not

Landed 2026-08-06 — ROADMAP item "Silicon-backed RV cost fixes", the three
under-charges `perfbench/riscvbench` found on Blackhole silicon
(`docs/plans/riscv-front-end-benchmark.md`). Each is small, each is in
`tt_sim/pe/rv/cost.py` territory, and they share one gate run because they
share one instalment; nothing else moved. All three are Blackhole-scoped by
*data* rather than by an arch string: each mechanism engages only where the
tables publish what it needs, and Wormhole publishes none of it — its
guards' modelled cycles are the control, and they moved zero.

### 1. The Blackhole multiply latency, spent as a scoreboard entry

`integer_unit.multiply: 1` + `multiply_ex2: 1` ("exactly one cycle in EX1,
and then exactly one cycle in EX2", `isa_doc`) is an occupancy of 1 and a
result **latency** of 2 — and `cost.py` charged the occupancy with no
scoreboard entry for the result, so a dependent multiply chain read 1.000
where silicon reads **1.985**. `RiscvCostModel.multiply_latency` now carries
`multiply + multiply_ex2` (`None` wherever `multiply_ex2` is unpublished),
and a pipelined multiply writes `ready[rd] = cycle + 2` exactly as a load
does: a dependent chain reads 2.000, independent successors stay free.
Wormhole publishes no multiply latency — its multiply *blocks* the integer
unit for two cycles, already charged as occupancy — so `multiply_latency` is
`None` there and its chain still costs the same 2 it always did, by the
mechanism it always did.

### 2. Dependent-chain loads: a minimal L0 d-cache line model

The model charged every Blackhole L1 load the `l1_dcache_hit` row's 2;
silicon's `rv_load_chase` — a 1 KiB pointer ring — reads **8.098**, the
`l1_dcache_miss` row, with `rv_load_indep` (1.742) corroborating latency 8
through the docs' own sustained-throughput formula. The row pair is not a
bound to be resolved at its low end — it is two rows, and which one a load
reaches is decidable per load from published facts. `cost.py` now keeps a
per-core line-tag model built entirely from `riscv.l0_data_cache`
(`isa_doc`: "a mere 64 bytes: 4 lines of 16 bytes each"): four tags, no
data, hit row when the loaded line's tag is resident, miss row when it is
not. The documented flushes are honoured — an L1 store invalidates its
containing line, a `fence` or atomic flushes all four — because skipping
them would mis-charge with a citation available. Every property the page
does *not* publish is resolved in the generous direction, and the choice is
recorded rather than implied: fully associative, least-recently-loaded
replacement, no modelling of the "~0.8 % chance of flushing the entire
cache" on a hit. With four lines that organisation never misses on a
working set that fits and always misses on a cyclic walk over more than
four lines — the two regimes the silicon probes pin — and any stricter
organisation could only miss more, so the modelled count remains a floor.
The charge itself needs no new provenance: `l1_dcache_miss: >= 8` was
already in the table at `isa_doc`, charged at its low end; what changed is
*which loads reach it*. Wormhole publishes no L0 (its single L1 row is
`>= 8` and was already charged), so the model never engages there.

### 3. Divide: still at the floor, and here is the search that kept it there

Silicon reads **33.001** for `divu 0x12345678, 3` — one point, at the top
of the documented 6–33 band — and the roadmap's open half was the
magnitude-dependent *curve*, the charge-the-floor policy having been
reviewed and kept on 2026-08-05. The curve was investigated and **does not
exist to transcribe**: both BabyRISCV pages (`WormholeB0` and
`BlackholeA0`) were searched whole for an iteration rule — bits per cycle,
a radix, a leading-zeros term, any formula or table relating an operand to
a cycle count — and neither publishes anything beyond "between six and 33
cycles are required, dependent upon the magnitude of the dividend"; ttsim
is functional-only and has no timing to borrow. Nor does the one silicon
point pin even the obvious one-bit-per-cycle family:
`cycles = significant_bits + 4` fits 33.001 at 29 bits but gives 36 at a
full 32-bit dividend, past the documented cap — the simplest candidate
contradicts the band it would be fitted inside. So the divide stays at the
documented floor of 6, the silicon point stays a `corroboration` on the
YAML entry, and `cost_test.py` now pins that the charge is
operand-independent above the documented special cases — at the benchmark's
own operands, with the 27-cycle under-charge stated in the test. The
exposure measurement stands: in-tree kernels divide 9–12-bit values 0–2
times per 40,000–80,000-instruction launch.

### The policy question, answered deliberately

Does "charge every bound at its low end" survive these numbers? **Yes —
because none of the three was ever a bound-resolution problem, and the two
that moved are the two where a published fact decides the case per
operand.** The rule resolves *one* two-ended cost (`>= 8`, "3 or 4", 6–33)
to the end that cannot invent back-pressure, and every `at_least` in the
RV section came out of the silicon run respected. What the multiply and the
load fix have in common is that the docs publish **two separate quantities**
(an occupancy and a latency; a hit row and a miss row) plus the mechanism
that selects between them (the EX1/EX2 split; the line geometry) — charging
only the smaller quantity was not floor policy, it was leaving a documented
mechanism unmodelled. Divide is the counter-case that proves the rule: there
the selector (the magnitude function) is genuinely unpublished, so the floor
stands. The rule applied throughout: **model a published mechanism; never
fit an unpublished one; where only a band remains, charge its low end.**

### Measured: the guard totals, and the control

Pump-to-DONE cycles at shutdown (`PUMP_CHUNK` forced to 10 for resolution),
model on, before → after all three fixes:

| guard | before | after | movement |
| --- | --- | --- | --- |
| `blackhole/six` (128³ bf16 matmul) | 83,810 | 83,890 | **+80** |
| `blackhole/softplus` (SFPU chain) | 18,330 | 18,410 | **+80** |
| `blackhole/nine` (two-tile NoC dataflow) | 12,700 | 12,800 | **+100** |
| `wormhole/softplus` (the control) | 19,880 | 19,880 | **0** |

Tens of cycles per launch, not thousands, and that smallness is itself the
measured claim from the exposure analysis: kernel time on these cores lands
on MMIO polls (the NIU block, charged nothing by name) and the Tensix drain,
not on L1 data loads in dependent chains, multiplies, or divides. The
Wormhole zero is load-bearing: all three mechanisms are engaged by published
Blackhole table entries, so nothing on Wormhole may move, and nothing did.
Model off is byte-identical everywhere by construction (`rv_cost` is never
built) — re-verified: all 39 driver guards and the full unit suite pass
unmodified.

Against the silicon rows the three probes now read, in simulation:
`rv_mul_dep`-shaped chains 2.000 (silicon 1.985, was 1.000);
`rv_load_chase`-shaped chases 8 per load (silicon 8.098, was ~2); `rv_div`
still 6 at any magnitude (silicon 33.001 at the benchmark's operand,
deliberately). `riscv_bench_sweep`'s predictions are unchanged by
construction — they read the YAML, which gained no number — and its two
notes claiming tt-sim under-charges the first two rows are updated to say
the simulator now agrees.

### The gate

Run whole, model on: **PASS** — the unit stage under the model (963 tests),
all 36 budget-independent value guards (`six` PCC 0.9982 unmoved, both
`pipestall`s, `twolaunch` clean), and all three budget-dependent guards
proven clean on the poll-budget ladder (`dramtop` at 1×, `two` at 2×,
`blackhole/offline` at 4× — the same multiples as the previous instalment).
Model off: the full `pytest tt_sim/ driver/` run is untouched — 39 driver
guards, every byte-identical Wormhole replay unmodified. One repair to the
gate's own instrument, found because this run tripped it:
`test_the_cost_tables_have_exactly_the_consumers_we_expect` filtered agent
worktrees out of its scan on the *absolute* path, so run from inside one it
excluded every file and compared an empty scan against the allow-list; the
filter now runs on the repo-relative path, which is what the exclusion
always meant.

### What changed in the repository

- `tt_sim/perf/model.py` — `RiscvCostModel.multiply_latency`,
  `l0_lines` / `l0_line_bytes` / `l1_load_miss_latency` (each `None` where
  unpublished); the `_LOAD_LATENCY_KEYS` / `_L1_DCACHE_MISS_KEYS` comments
  now describe the residency split.
- `tt_sim/pe/rv/cost.py` — the four-tag L0 line model (`_l0_load`, store
  invalidation, fence/atomic flush, `l0_hits`/`l0_misses` in `summary()`),
  the multiply scoreboard entry, and the re-search of the divide formula
  recorded in the module docstring.
- `tt_sim/pe/rv/cost_test.py` / `tt_sim/perf/model_test.py` — pins for all
  three: chain at 2 on BH and unchanged on WH; cold miss / warm hit / 5-line
  chase always-miss / 4-line loop always-hit / store flush / fence flush /
  WH untouched; divide at the floor at the benchmark's own 29-bit operand.
- `tt_sim/perf/unit_costs.yaml` — prose only (the `l0_data_cache` note's
  "NOTHING CHARGES THIS" is no longer true and now names both consumers; the
  multiply and L1-row corroborations record what the simulator now reads
  against them). **No cycle count, bound or provenance changed.**
- `tt_sim/perf/riscv_bench_sweep.py` prose, `docs/bh_arch.md` §1.6 — the
  under-charge notes updated to "the simulator now agrees".
- `tt_sim/perf/costs_test.py` — the worktree-path repair to the
  consumer-pinning scan described under "The gate".

## Unpacker and mover occupancy: the last two units, and the first cost that is a function of the transfer

Landed 2026-08-06 — ROADMAP item "Unpacker and mover occupancy", the two
Tensix backend units the cost tables have carried data for since the
beginning and deliberately not charged. `UNWIRED_UNITS` in
`tt_sim/perf/costs_test.py` is now **one entry** (`TDMA`, all-1-cycle by one
blanket sentence, out on purpose), and eight of the nine units read their own
costs. Both wirings needed something the six before them did not: neither
unit's cost is a per-opcode constant, so neither could use the flat lookup
`TensixBackendUnit.instruction_occupancy` hands every other unit.

### The unpacker: an address phase plus bytes over a configured rate

`UNPACR_Regular.md`'s Performance section publishes both halves in two
sentences, and the wiring charges both, serially:

> "An `UNPACR` instruction spends at least two cycles calculating the initial
> input address: uncompressed data requires exactly two cycles, whereas
> compressed data requires more. For the duration of these cycles, the issuing
> thread cannot start its next instruction, nor any can other thread start an
> `UNPACR` instruction. Once these cycles are complete, execution proceeds in a
> pipelined fashion, with the primary bottleneck being the fetching of bytes
> from L1."

- **Address phase: 2 cycles.** The table's existing `occupancy: { cycles: 2,
  bound: at_least }` (`isa_doc`), charged at its low end like every other
  bound. Here the low end is not merely a floor: 2 is the *exact* published
  figure for uncompressed data, and uncompressed is the only kind tt-sim
  unpacks (`get_isUncompressed` returns `True` unconditionally and the
  compressed walk raises).
- **Data phase: `ceil(transfer_bytes / rate)`.** `rate` is the throttle mode in
  effect — "x1 speed: Up to 16 bytes per cycle from L1", x2 32, x4 64, selected
  by `ConfigState.THCON_SEC[WhichUnpacker].Throttle_mode` where "`0` means x1,
  `1` means x2, and `2` means x4" (all `isa_doc`, the table's
  `l1_bandwidth.throttle_modes`, which was recorded and unconsumed until now).

That is the whole reason this unit needed its own shape. The charge is computed
in `read_unpack_state` — where the transfer size (`InputNumDatums` ×
`DatumSizeBytes`) and the throttle config have just been decoded — and armed by
`clock_tick`. The base pre-handler hook is *declined* for `UNPACR` on purpose,
and for a second reason as well: `UNPACR` is three instruction forms behind one
opcode, and only the datum-moving one has a Performance section, so the
increment-context-counter and flush-cache forms are charged **nothing** rather
than inheriting the regular form's two cycles.

**Two forced modes and one arch difference, all transcribed rather than
inferred.** The doc constrains the throttle in five cases; four of them force
modes of unpacks tt-sim rejects before moving a datum (compressed data,
`UpsampleZeroes`, BFP2), so exactly one is reachable and it is charged:
"tileize always runs at x4, regardless of `Throttle_mode`". Blackhole's
differences come from the same file — the BlackholeA0 tree holds a stub saying
architecture differences are "conditionalized inline using `TTArchitecture`" —
and are three quoted pseudocode lines, now in `arch_overrides.blackhole` so
that Wormhole can never read them:

| Blackhole fact | The doc's line |
| --- | --- |
| mode 3 = x8 = 128 B/cycle | "x8 ThrottleMode is illegal on Wormhole" (so it is legal here), with `ThrottleBytes = 16u << ThrottleMode` |
| x4 becomes 128 B/cycle for datums ≥ 2 bytes | "`ThrottleBytes = 128; // upgrade to x4 '2x'`" |
| the configured mode is ignored unless `REG1_ovrd_default_throttle_mode` is set | "`ThrottleMode = (DatumSizeBytes == 1) ? 3 : 2; // 8-bit modes use x8, others use x4`" |

The reference simulator (ttsim's `tensix.cpp`) carries the identical logic under
`TT_ARCH_VERSION == 1`, which is a second reading of the same source rather
than a second source. Not charging these would bill Blackhole's data phase at
up to **twice** its documented rate — over-charging, which the bounds policy
forbids as firmly as it forbids invention. They are not hypothetical, either:
instrumenting `blackhole/six` shows every one of its `UNPACR`s asking for
2,048 bytes (a whole 32×32 bf16 tile) with `Throttle_mode` 2 and
`ovrd_default_throttle_mode` **clear** — the default path — so it is charged
2,048 / 128 = 16 cycles rather than the 32 the shared x4 rate alone would
have billed. Wormhole's `softplus`, by contrast, unpacks 512 bytes with the
override bit set, at the shared x4: 512 / 64 = 8.

### The 80 B/cycle joint ceiling is still not charged, and that is the honest answer

The design constraint this instalment was given was explicit: model the shared
ceiling only if the sharing rule can be sourced. It cannot, so it is not.

What exists is a *number* (`joint_bandwidth`, 80 B/cycle, `vendor_source` from
tt-metal, with the ISA docs' L1 page supplying the "five 128-bit reads per
cycle" half) and a *qualitative* 3×3 table of what each unpacker gets when both
are streaming at once. Neither is an arbitration rule: the 3×3 table gives
sustained rates for two simultaneously-streaming units, not a per-transfer
division, and tt-sim charges each transfer once at issue with no notion of two
overlapping streams. Inventing the division would be exactly the failure the
provenance convention exists to prevent. So **each unpacker is charged its own
uncontended rate**, the ceiling is recorded in the table as unconsumed *with
the reason*, and a test pins that it stays that way. The same applies to the
cross-unpacker half of the address-phase sentence ("nor can any other thread
start an `UNPACR`"): the hold is per unit, so unpacker 1 issues while unpacker 0
is held. Both omissions are under-charges — the floor direction — and both are
pinned by name in `unpacker_cost_model_test.py` so they cannot be mistaken for
oversights.

### Blocking: a blocked unit is waiting, not busy

The unpacker is the one unit that legitimately blocks for thousands of cycles
(worst measured in-tree: 3,528, from the `[UNIT STALL]` survey) waiting for the
Matrix Unit to hand a `Src` bank back. Occupancy had to compose with that
without double-counting it, and the doc's own ordering says how: the address
phase happens *before* the wait ("spends at least two cycles calculating the
initial input address", *then* the L1 fetch), so

- the 2-cycle address phase is charged **once**, at the cycle the unpack was
  accepted, even when the unpack then blocks;
- every blocked re-run charges **nothing** — that is the unit waiting on
  somebody else, which tt-sim already models functionally;
- the data phase is charged from the cycle the transfer actually starts, i.e.
  when the bank comes back.

One mechanism fix fell out: the blocked path never reaches the base drain,
which is the only place holds are released, so an address-phase hold armed just
before a block would have outlived its deadline for the length of the wait.
`UnPackerUnit.clock_tick` now releases expired holds on that path too. Nothing
about `blocked_on()`, `hasInflightInstructionsFromThread` or the
`[UNIT STALL]` / `[UNIT WEDGED]` machinery changed, and every replay guard
passing is the proof that the composition is sound.

### The mover: the table's 1 was never the interesting number

`XMOV.md` splits the cost into two explicitly different quantities in one
sentence — "The thread issuing an `XMOV` instruction will be automatically
stalled until the mover is able to _start_ work, at which point `XMOV` will
execute in a single cycle - the mover proceeds with the task in the background"
— and the table has always held both halves in different files. The 1 is issue;
the duration is `unit_costs.yaml`'s `mover.transfer`, whose rates the ISA doc
publishes as *measured*, with an ideal and a contended column per transfer
kind. `MoverCostModel` now spends them:

| Transfer kind | XMOV modes | Ideal rate, as the doc states it | Charge |
| --- | --- | --- | --- |
| `l1_to_l1` | `XMOV_L1_TO_L1`, `XMOV_L1_TO_L0` | "eight 128b reads and eight 128b writes every 11 cycles i.e. 93.1 bits copied per cycle" | `ceil(bits × 11 / 1024)` — 1 KiB = 88 cycles |
| `l1_memset` | `XMOV_L0_TO_L1` | one 128b write per cycle | `ceil(bytes / 16)` |
| `non_l1_memset` | `XMOV_L0_TO_L0` | one 128b write per cycle | `ceil(bytes / 16)` |

Charged as **unit occupancy**, because "stalled until the mover is able to
start work" is precisely what the existing issue-refusal machinery models — no
new mechanism was needed for the XMOV path, and the TDMA command queue (which
bypasses the instruction path entirely) is charged where it runs. The mover is
also the one unit where the transfer is visible to
`checkForOutstandingInstructions`, and that asymmetry with the unpacker is
deliberate: the doc frames the mover's transfer as a *background* task the
thread explicitly waits for (`STALLWAIT` C12 on Wormhole, C9 on Blackhole),
where it frames the unpacker's data phase as pipelined throughput.

**The ideal column is charged and the contended one is not.** The page gives no
rule for when its L1-port contention applies, so the ideal rate is the floor —
the contended 32 bits/cycle would be ~2.9× the charge, and charging it on
faith would be over-charging with a citation attached. Two paths that stay at
the entry's 1: a transfer of no bytes, and a mode the table prices no rate for.

### Measured, at 10-cycle poll resolution: the totals move, and by tens

Pump-to-DONE cycles at shutdown (`PUMP_CHUNK` forced to 10), model on, before →
after. The instrumented columns are this change's own: cycles charged by the
two newly wired units, and issue attempts they refused.

| guard | before | after | movement | unpacker cycles charged | issues refused | PCC / value |
| --- | --- | --- | --- | --- | --- | --- |
| `blackhole/six` (128³ bf16 matmul) | 83,890 | **83,890** | **0** | 2,304 (128 UNPACRs × 18 = 2 + 2048 B / 128) | 0 | PCC 0.9982, unmoved |
| `blackhole/sfpumath` | 20,250 | **20,280** | **+30** | 120 | 74 | bit-exact, unmoved |
| `blackhole/nine` (two-tile NoC dataflow) | 12,800 | **12,820** | **+20** | 128 | 24 | all 256 elements, unmoved |
| `blackhole/tilize` (the tileize-x4 path) | 72,000 | **72,010** | **+10** | 78 | 12 | all 4,096 results, unmoved |
| `wormhole/softplus` | 19,880 | **19,910** | **+30** | 80 (+ 57 mover) | 43 | bit-exact, unmoved |

Model off: `blackhole/six` reads 27,170 before and after, and the whole
`pytest tt_sim/ driver/` run is untouched.

**This is less movement than the item predicted, and the instrumentation says
why.** The ROADMAP expected this to be the first change to move a total
substantially, on the reasoning that the front-end bound now lets a unit's
occupancy reach the issuing core. The bound does reach it — the refusal counts
above are real, 74 on `sfpumath` where the previous six units managed five
across the whole tree — but a refused issue is only a *slower* run if nothing
else was going to stall anyway, and on these workloads the dataflow cores have
slack. `six` is the extreme case and the informative one: **2,304 charged
cycles, zero refused issues**. Nothing in that kernel tries to start a second
unpack within 18 cycles of the first, because the matmul's own unpack/math
handshake is slower than the modelled fetch. The charge is armed, correct and
invisible, which is the seventh time this file has had to report that answer
and the first time it has had the refusal counts to say *how close* it came.

What *did* change qualitatively: this is the first instalment where a Tensix
unit's occupancy is a function of the workload's data rather than of its
instruction mix. Doubling a kernel's tile size now doubles its unpacker charge,
which no previous unit's table could express.

### The gate

Run whole, model on: **RESULT: PASS** — the unit stage under the model (1,003
tests), all 36 budget-independent value guards (`six` PCC 0.9982 unmoved, both
`pipestall`s, `twolaunch` clean, both `tilize`s and both `untilize`s — the
tileize-x4 path this change added), and all three budget-dependent guards
proven clean on the poll-budget ladder at **exactly the multiples they needed
before it**: `blackhole/dramtop` 1×, `blackhole/two` 2×, `blackhole/offline`
4×. Nothing moved onto a higher rung, which is the stronger statement: the
ladder is the gate's only measure of *how much* slower a run got, and it did
not register this change at all.

Run outside the gate, each Blackhole guard as its own standalone main under the
model, 26 of them: 24 pass; `blackhole/two` and `blackhole/offline` fail at the
recorded 1× poll budget, which is what being budget-dependent *means* and which
was verified to be true of the unchanged tree as well (both fail at 1× with the
model on before this change, and the gate passes both at their usual multiple
after it).

Model off: the full `pytest tt_sim/ driver/` run is unchanged — 1,003 unit
tests, all 39 driver guards, every byte-identical Wormhole replay unmodified —
and `blackhole/six` reads the same 27,170 cycles before and after.

### What changed in the repository

- `tt_sim/pe/tensix/tensix_instruction_costs.yaml` — `UNPACK`'s
  `l1_bandwidth` gains `tileize_forced_mode` and, under
  `arch_overrides.blackhole`, the `blackhole_throttle` block (x8, the x4 "2x"
  upgrade, the default-mode pair), each with its own quoted line;
  `joint_bandwidth` gains the note saying it is unconsumed and why; the
  `UNPACR` and `XMOV` entries record what is now charged and what is
  deliberately not.
- `tt_sim/perf/unit_costs.yaml` — prose only: `mover` now names its consumer
  and says the ideal column is charged and the contended one is not. **No
  cycle count, bound or provenance changed anywhere in either file.**
- `tt_sim/perf/model.py` — `UnitCostModel.unpack_data_phase_cycles` (the
  throttle-rate selection, transcribed in the pseudocode's own order) and the
  new `MoverCostModel` / `mover_cost_model`.
- `tt_sim/pe/tensix/backends/unpacker.py` — the cost model, the declined
  pre-handler hook, the charge computed at decode and armed at retire, the
  once-only address phase across a block, and the expired-hold release on the
  blocked path.
- `tt_sim/pe/tensix/backends/mover.py` — the two models, the mode→transfer-kind
  map, the occupancy on both the XMOV and TDMA paths, and the in-flight
  transfer reading as outstanding work.
- `tt_sim/pe/tensix/unpacker_cost_model_test.py` (27 tests) and
  `tt_sim/pe/tensix/mover_cost_model_test.py` (13 tests) — the charge
  arithmetic at every throttle rate on both arches, the forced modes, the
  blocking composition, the back-pressure, the TDMA queue, and the two
  deliberate under-charges pinned by name.
- `tt_sim/perf/costs_test.py` — `UNWIRED_UNITS` down to `TDMA`; four new
  entries on the consumer allow-list.

## The scoreboard learns to be read as a schedule, so parking reaches the model

Landed 2026-08-06 — ROADMAP item 1, "firmware-loop parking under the cost
model". **No charge changed and no cycle count moved**; this instalment exists
because the RV scoreboard grew three methods and because the house rule is that
anything touching whole-tree timing gets its own gate run. It got one:
`driver/tests/cost_model_gate.py` **PASS**, every stage, and the device cycle
count at DONE on all 26 Blackhole guards is identical with parking on and off,
guard for guard.

### What the previous version got wrong about its own reason

Firmware-loop parking (`tt_sim/pe/rv/spin.py`, `docs/plans/event-driven-pump.md`)
was switched off whenever `TT_SIM_COST_MODEL` was set, and the recorded reason
was that `RiscvCostState` carries **absolute cycle numbers** — `ready[rd] =
cycle + latency`, `_stall_until`, `_store_ready` — so restoring a recorded
trajectory across a skipped span needed a time-translation argument nobody had
made. That is a real problem, and it is not the first one. Switching detection
on and instrumenting it found something worse and much simpler:

> A **stalled tick** is a perfect one-tick fixed point. `can_issue` returns
> False, `RV32I.clock_tick` returns before executing anything, the PC does not
> advance and no register changes — which is exactly the predicate SEEK,
> RECORD and VERIFY were built to look for.

On the `six` Blackhole guard with detection naively enabled, BRISC and the
TRISCs parked every few hundred cycles in "pure poll loops of **1 ticks**",
each one an L1-store-rate stall (`l1_store_period` 5 — the park/unpark pairs
are five cycles apart, which is how the diagnosis was made). On those guards it
happened to be harmless, because five cores share a tile and the other four
kept it awake, so no stride ever began: the 26 cycle counts came out identical
anyway. Put a core **alone** on a tile and it is a silent wrong answer — a
BRISC in a store-rate loop executed **41 iterations in 4,000 cycles against the
399** an unparked run executes, having slept through the rest of its own stall.
That is the failure mode `event-driven-pump.md`'s risk section names: a
too-eager idle predicate does not crash, it just returns different numbers.

**Is the *already-landed* model-off parking exposed to the same thing? No, and
the argument is short enough to check.** With the model off a tick can still
retire nothing, via `ProcessingElement.PEStall`, and there are exactly three
sources of it. Two are writes — a stalling `sw`, and a `.ttinsn` push whose
front-end FIFO is full — and both reach `_ObservedMemory.write`, which aborts
the attempt before delegating, so RECORD never completes. The third is a `lw`
whose `read` returns `MemoryStall`, and the only components that do that are
`Mailbox` and `TTSync`; neither is an `AddressableMemory` /
`SparseAddressableMemory` leaf, so `_build_watch` refuses the span and the
attempt aborts there instead. So the landed behaviour is safe, and this
instalment fixes nothing that is live — but it is safe *incidentally*, by two
unrelated rejections, which is why the reasoning is now written into
`FirmwareSpin._advance` with the standing condition attached: if a plain-RAM
read is ever given a stalling path, the fresh-arrival gate has to learn to see
it with the model off too.

### The fix extends the proof rather than repairing after it

The scoreboard is not restored *after* a park; it is **part of what has to be
proved periodic before one**. Three methods on `RiscvCostState`, placed there
because which fields are cycle numbers is knowledge that belongs next to the
fields:

- `spin_signature(cycle)` — the **cycle-relative normal form**: every absolute
  deadline as `max(field - cycle, 0)` ("how many cycles from now"), plus the L0
  line tags and the store-coalescing group while its drain is still live. It is
  lossless for anything the model can go on to do, and for exactly two reasons.
  Every one of those fields is read only through a strict *is it still in the
  future* test against the current cycle, so two values at or below it are
  indistinguishable from here on; and the cycle number never decreases within a
  run (`reset()` is the only rewind, and it zeroes everything).
- `spin_restore(signature, cycle)` — the time translation, re-basing a
  signature onto the wake cycle.
- `spin_counters()` / `spin_add_counters(delta)` — the accumulators, so a
  skipped span is *charged* rather than lost.

`spin.py` records the signature alongside the register snapshot at every tick,
requires it to match at the second anchor, and requires VERIFY to reproduce it
tick for tick together with the exact per-tick charges. **A countdown fails
that by construction**: a fixed deadline shrinks by one every cycle while a
periodic schedule does not move. What survives is precisely the class of loops
whose *schedule* repeats, not merely whose registers do — and for those, an
unpark re-bases the phase's normal form onto the wake cycle and adds `laps`
whole iterations' worth of charges plus the partial one. So the §I report comes
out the same as an unparked run's, not only the cycle count;
`spin_test.py::test_cost_model_parked_skips_are_cycle_exact_and_charge_the_same`
asserts `rv_cost.summary()` equality directly, and the device-level test does
the same across a real Blackhole tile.

One half of the mechanism is not about the scoreboard at all. A stall also
means the *next* tick is the same instruction being re-attempted rather than a
fresh arrival at that PC, and anchoring on — or closing an iteration at — a
re-attempt finds the wrong period. `spin.py` now gates both on "the previous
tick retired something", read for free off `stall_cycles`. Without it the
detector still never parks wrongly (the signature check catches it) but it
mostly fails to park at all: on Wormhole, where the L1 load latency is `>= 8`
and seven of every nine ticks of a poll loop are stalls, **zero** parks in 3,000
cycles. With it, the same loop parks as the 9-tick period it is.

### What the model does to a loop, measured

The model makes the same firmware go-wait several times longer in ticks, which
is a scope statement worth writing down because the recogniser has a cap:

| loop | model off | model on |
| --- | --- | --- |
| Blackhole BRISC go-wait (18 instructions) | 18 ticks | 46 |
| Blackhole NCRISC idle | 8 | 15 |
| Blackhole TRISC go-waits | 3–7 | 10–13 |
| Wormhole BRISC go-wait | 18 | **49** |

49 against a 64-*tick* recognition budget is not enough headroom — a firmware
revision with a slightly longer go-wait would silently stop parking under the
model while still parking with it off, which is the quiet kind of regression.
Scaling the budget under the model was the first attempt and it was wrong in
the other direction: `cost_model_gate.py` caught it, because a 100-instruction
NOP loop that the model-off budget rejects became recognisable under the model,
and the deadlock watchdog started reporting a wedge in one configuration and
not the other (`deadlock_test.py::test_quiet_on_a_loop_whose_period_is_the_
sample_interval`). The budget is now counted in **retired instructions**
(`MAX_LOOP_INSTRUCTIONS`), with a tick backstop that only a pathologically
stalled attempt can reach. That makes the two configurations agree on *which
loops are recognisable* in both directions: an 18-instruction go-wait is in
budget whether it takes 18 ticks or 49, and a 100-instruction loop is out of
budget either way.

### The gate, and what it does and does not establish

`driver/tests/cost_model_gate.py` **PASS** on every stage. `pytest tt_sim
driver` 1,060 passed with the model off and again with `TT_SIM_COST_MODEL=1`.
All 26 Blackhole replay guards pass standalone; Wormhole `offline_replay_test`
reproduces its data READs as before.

The load-bearing check is not any of those, and it was run explicitly rather
than inferred: **the device cycle count at DONE on all 26 Blackhole guards is
identical with parking on and off, under the model** (`four` 107,400,
`six` 85,500, `tilize` 73,200, `pipestall` 36,400, …) and, separately, with the
model off (unchanged from the landed numbers, guard for guard). The guards pump
in fixed chunks until the go message reads DONE, so an equal total is an equal
number of chunks — the kernel finished on exactly the same cycle. What none of
this establishes is anything about whether those cycle counts are *right*; that
is the gate's own standing caveat and it is unaffected here.

## The NIU register block: the number was in the table, under the wrong key

Landed 2026-08-06 — the ROADMAP item "cost the NIU register block", which had
stood as *blocked on provenance* and was queued to become a silicon probe.
**No number was sourced, derived or measured, because none had to be.** The
block's latency was already in `riscv.load_latency`, on a row this file's own
key name mis-described. What changed is a classifier, two key names and two
tests; every cycle count, bound and provenance in both YAML files is
byte-identical to before.

### The claim that was wrong, and what the page says

The recorded reason not to charge it, in `RV_UNNAMED_REGIONS`, in `cost.py`'s
docstring, in this file three times and in the ROADMAP:

> The ">= 7" row covers "TDMA / tile control / PIC / NoC **overlay**" —
> `0xFFB40000`, a different block. Charging the overlay's number to the NIUs
> would be a guess with a citation stapled to it.

The row does cover the overlay. Its cell also has five other entries, each on
its own line, and two of them are the NIUs. Verbatim from
`WormholeB0/TensixTile/BabyRISCV/README.md`:

| Load address range | Load latency (cycles) | Max loads in flight |
| --- | --- | --- |
| TDMA-RISC configuration and command<br/>Tile control / debug / status<br/>PIC configuration and status<br/>**NoC 0 configuration and command**<br/>**NoC 1 configuration and command**<br/>NoC overlay configuration and command | ≥ 7 (more in the case of access conflicts) | 4 |

The two bold lines link to `NoC/MemoryMap.md` — "Each NIU has an assortment of
command and configuration and status registers mapped into the address space of
the containing tile … `NIU_BASE` … `0xFFB2_0000` … `0xFFB3_0000`" — where the
overlay's line links to `NoC/Overlay/README.md`. Blackhole's table has the same
row with the same six entries and the same `≥ 7`, less TDMA, which its own `≥ 4`
row takes and which `_LOAD_LATENCY_KEYS` already split correctly.

So the gap was **one word in a key name**. `tdma_tilectrl_pic_noc_overlay`
reads as "…, PIC, NoC overlay" and was written down, propagated and defended
that way for as long as the row has been in the file. Both keys are now
`…_noc0_noc1_overlay`, the region constant is `RV_REGION_TILECTRL_PIC_NOC`,
and a test asserts the classification of `0xFFB20204` and `0xFFB30208` — the
two counters the barriers actually spin on — by name.

### A second, plainer bug in the same classifier

`NOC_OVERLAY_START_ADDR` is `0xFFB4_0000` **to `0xFFB7_FFFF`**: 64 stream
register spaces of 4 KiB, which is exactly what `NoCOverlay` models
(`NOC_NUM_STREAMS = 64`, `NOC_STREAM_REG_SPACE_SIZE = 0x1000`) and exactly
what `TensixTile` maps. `classify_address` ended the region at `0xFFB50000`
— one stream space — so overlay streams 16-63 fell to `RV_REGION_UNNAMED` and
were charged nothing. It is a slip, not a judgement, and it was found by the
same census: `wormhole/reduce` makes 399 loads there, all of stream 16's
registers 3 and 4, and `blackhole/six` makes 20,589.

### The exposure, measured before anything was charged

Every RV load in a guard run, classified by address block, `TT_SIM_COST_MODEL=1`
and the NIU block still uncharged (so these are the counts the gap actually
had):

| guard | RV loads | MMIO loads | NIU loads | of all | of MMIO | overlay tail |
| --- | --- | --- | --- | --- | --- | --- |
| `blackhole/six` | 67,462 | 65,055 | **30,834** | 45.7 % | 47.4 % | 20,589 |
| `blackhole/loopback` | 6,079 | 3,663 | 1,670 | 27.5 % | 45.6 % | 0 |
| `blackhole/three` | 6,893 | 2,500 | 1,674 | 24.3 % | 67.0 % | 0 |
| `blackhole/four` | 7,130 | 4,583 | 1,670 | 23.4 % | 36.4 % | 0 |
| `blackhole/nine` | 11,173 | 5,645 | 2,047 | 18.3 % | 36.3 % | 0 |
| `wormhole/reduce` | 4,684 | 2,980 | 1,254 | 26.8 % | 42.1 % | 399 |
| `wormhole/softplus` | 12,429 | 9,767 | 429 | 3.5 % | 4.4 % | 0 |

**The expected code path is confirmed rather than assumed.** Resolving each
NIU load to its register offset: `NIU_MST_WR_ACK_RECEIVED` (`NIU_BASE + 0x204`)
and `NIU_MST_RD_RESP_RECEIVED` (`+ 0x208`) are **97-99.5 %** of every guard's
NIU traffic — `six` 27,300 + 3,368 of 30,834, `nine` 1,986 of 2,047. Those are
the two counters `noc_async_read_barrier` and `noc_async_write_barrier` spin
on, and nothing else in the block is touched more than a few hundred times.
On `six` the two cores split the way the kernel does: NCRISC (the reader,
27,431 NIU loads) is polling read responses, BRISC (3,403) write acks.

`wormhole/softplus` is the useful low row: an SFPU-bound kernel whose TRISC2
does 7,770 loads and whose NIU traffic is 429. The claim is about *dataflow*
time, and a compute-bound guard shows the term almost absent, which is what
makes the high rows mean something.

### The sensitivity, and it is not linear, for a reason worth writing down

Modelled cycles at DONE, `PUMP_CHUNK` forced to 10 for one-cycle-order
resolution, model on. `before` is the tree as it stood; `charge N` reclassifies
the NIU blocks and charges the load-use interlock `N`, everything else held at
production. **7 is the shipped column** and the only one taken from a document.

| guard | before | 0 | 1 | **7** | 20 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| `blackhole/six` | 83,890 | 83,900 | 83,900 | **85,780** (+2.25 %) | 89,260 (+6.4 %) | 99,400 (+18.5 %) |
| `wormhole/reduce` | 9,500 | 9,580 | 9,580 | **9,690** (+2.00 %) | 9,840 (+3.6 %) | 10,310 (+8.5 %) |
| `blackhole/nine` | 12,820 | 12,820 | 12,820 | **12,940** (+0.94 %) | 13,260 (+3.4 %) | 13,780 (+7.5 %) |
| `blackhole/three` | 13,320 | 13,320 | 13,320 | **13,420** (+0.75 %) | 13,670 (+2.6 %) | 14,150 (+6.2 %) |
| `blackhole/loopback` | 11,170 | 11,170 | 11,170 | **11,250** (+0.72 %) | 11,460 (+2.6 %) | 11,770 (+5.4 %) |
| `blackhole/four` | 105,440 | 105,440 | 105,440 | **105,540** (+0.09 %) | 105,830 (+0.37 %) | 106,230 (+0.75 %) |
| `wormhole/softplus` | 19,910 | 19,910 | 19,910 | **19,960** (+0.25 %) | 20,020 (+0.55 %) | 20,200 (+1.46 %) |

Three things in that table are worth more than the numbers.

**Charging 1 is exactly charging 0.** Identical to the cycle in all seven rows,
because the interlock spends `latency - 1` and the docs' own floor is "the
minimum possible load latency is two cycles". A one-cycle latency is below the
mechanism's floor, not a small version of it.

**The term is self-limiting, which is why 50 is not 7× worse than 7.** A
barrier is a *waiting* loop: it iterates until the NoC answers, so making each
iteration cost more makes it iterate fewer times, and the number of NIU loads
falls roughly in inverse proportion to the charge (not exactly, because the
loop body has other instructions in it). On `six`, NIU loads go **30,834 →
7,988 → 3,252 → 1,578** at charges of 0, 7, 20 and 50 — a 32× charge for a 20×
drop in count. The consequence is that this term's *error* is cheap: being
wrong by a factor of seven (7 against 50) costs `six` 16 % and every other
guard under 7 %, and being wrong low costs less again. That bears directly on
"was this worth card time": the exposure justifies charging it, but a probe to
pin the number would have ranked **low** in the silicon bundle even had the row
genuinely been silent, because a plausible mis-charge moves a total by single
percent. It is off that list because the row is not silent.

**Charged is not delivered, again.** `six`'s charge is 50,220 extra RV stall
cycles for 1,890 delivered — 3.7 %. Same story as every previous RV instalment:
these runs are not issue-limited, they are waiting on each other.

Which is also why `four` moves least (0.09 %) despite having the same 1,670 NIU
loads as `three` and `loopback`. Its length is set by five cores' Tensix
handshaking, and its barrier polling sits off the critical path entirely.

### What this does not change: firmware-loop parking

The parking recogniser (`tt_sim/pe/rv/spin.py`) rejects any candidate loop
containing a load outside plain RAM, at RECORD, before a watch set is built —
"MMIO is rejected wholesale, which is what keeps a wall-clock timeout loop, a
mailbox pop, **an NIU-counter poll** or a PC-buffer wait from ever parking". So
an NIU poll loop never parked before this change and does not park after it: the
predicate is on the address class, not on any cost, and it must stay that way,
because the whole point is that the loaded value changes with no write for the
watch set to see.

What *does* change is that the un-parkable loop now runs ~4× fewer iterations
per unit of simulated time. No wall-clock claim is made here — that would need
the interleaved frozen-worktree A/B the working rules require, and it was not
run — but the direction is worth recording, because "NIU polling is the traffic
that keeps a core awake" was true and is now less true per cycle.

### Provenance: what was searched, including the dead ends

The row settles it at `isa_doc` and nothing below that rank can improve on it,
but the search was run to completion first, and two of its results are worth
keeping.

- **Both `BabyRISCV/README.md` pages** (WH and BH) — the answer, above. Also
  `MemoryOrdering.md` on both, which adds an *unclaimed* term rather than a
  number: "Each memory region can process at most one request per cycle" for
  every region but L1. That is a throughput bound on the NIU block, tt-sim
  models no per-region request queue, and it is now named in ROADMAP item 2.
- **`NoC/MemoryMap.md`, `NoC/Counters.md`, `NoC/README.md`** (WH and BH): no
  register-access latency anywhere. `NoC/README.md`'s hop table does give
  "~5 cycles" for NIU↔router, which is a **flight** time between an NIU and a
  router and is already consumed as `noc.hops`; it is not what a core pays to
  read a register and confusing the two would double-count.
- **ttsim** — clean negative, as expected and now concretely: `t_tile_mmio_rd32`
  (`src/tile.cpp`) decodes NOC0/NOC1/overlay/TDMA/debug identically and every
  one returns `{true, value}` in the same step; the only `{false, …}` returns in
  the file are functional (a Tensix drain, an empty mailbox). Its `data/*/
  tile_regs.json` carry names and offsets and no timing. Time is an externally
  driven step counter (`libttsim_clock`), and the README lists cycle counters
  among the things it does not reproduce.
- **tt-metal** — no number, and one **corroboration worth recording**. Three
  functions in `tt_metal/hw/inc/internal/tt-1xx/{blackhole,wormhole}/
  noc_nonblocking_api.h` (`noc_local_state_init`, `dynamic_noc_local_state_init`,
  `ncrisc_noc_counters_init`) issue five to ten `NOC_STATUS_READ_REG` of
  `NIU_MST_*` counters back to back before consuming any of them, under the
  comments `// Hide latency of NOC reg reads by reading first, writing second`
  and `// Pipeline all register reads first to hide latency`. That is the ISA
  docs' "N - 1 independent instructions need to follow the load" being obeyed
  by the firmware, at exactly the batch size a latency of 7 calls for, and it
  is independent evidence that the NIU read is a **slow-region** load rather
  than a fast one. It is qualitative — no cycle count anywhere in the tree — so
  it is `corroboration`, not provenance, and it changes no rank.
  The LLK trees (`tt_llk_{wormhole_b0,blackhole,quasar}`) never touch the block;
  `noc_estimator` measures end-to-end only and its YAML cannot separate the
  issuing core's path from flight; the one harness that *does* isolate issue
  cost (`tests/.../data_movement/noc_api_latency/`) times NoC cmd-buf **writes**
  and ships no results.

### What is still not charged, in this block specifically

- **The "more in the case of access conflicts" tail** on the `≥ 7` row. Same
  shape as every other `at_least` in these files: the low end is charged and
  the tail is unquantified.
- **Per-region request throughput** (one request per cycle, from
  `MemoryOrdering.md`). tt-sim has no queue in front of an MMIO region, so two
  cores hammering the same NIU cost what one does.
- **`max_loads_in_flight`: 4 in aggregate**, which is in the YAML and read by
  nothing. Re-reading the table for this change also corrected the note beside
  it: that cell is a four-row **rowspan**, so the budget of 4 is shared across
  every region *except* core-local data RAM — the mailbox group, the GPR/config
  group, the `≥ 7` group and L1 together — where the note had it covering the
  mailbox group alone. The number did not change and nothing consumes it. A
  barrier poll loop is precisely the shape that would hit it.
- **Stores** to the block, deliberately: "other memory regions can achieve a
  throughput of one store every cycle" is what the simulator already does. The
  ~6 NIU register writes per `noc_async_read` are therefore free and correctly
  so.

### The gate

`driver/tests/cost_model_gate.py` **PASS**, every stage. `pytest tt_sim/ driver/`
1,062 passed with the model off and again with `TT_SIM_COST_MODEL=1` (1,060
before; two new tests). All 26 Blackhole replay guards pass standalone.
`blackhole/six`'s PCC is **0.9982 at every charge in the sensitivity table**,
including 50 — the values are untouched by any of this, which is what the
sensitivity sweep was also checking.

Model off is byte-identical by construction: `classify_address` is only ever
called from `RiscvCostState.can_issue`, and `rv_cost` is `None` when the model
is off.

### What changed in the repository

- `tt_sim/pe/rv/cost.py` — `classify_address` gains `0xFFB20000-0xFFB3FFFF`
  and extends the overlay to its published `0xFFB7FFFF`; `_TILE_CTRL_END`
  extended over the PIC's 4 KiB for the same reason; the docstring's "not
  modelled" list loses the NIU entry and gains the account above.
- `tt_sim/perf/model.py` — `RV_REGION_TILECTRL_PIC_OVERLAY` →
  `RV_REGION_TILECTRL_PIC_NOC`, both `_LOAD_LATENCY_KEYS` entries renamed,
  `RV_UNNAMED_REGIONS` down to three entries with the retraction recorded.
- `tt_sim/perf/unit_costs.yaml` — the two row keys renamed and both notes
  extended. **No cycle count, bound or provenance changed.**
- `tt_sim/pe/rv/cost_test.py` — two new tests (the NIU rows on both arches by
  counter address; the overlay's published extent), the unnamed-region tests
  re-pointed at what is actually unnamed, and the PIC added to the memory-map
  test.
- `tt_sim/perf/noc_dataset_sweep.py`, `driver/tests/cost_model_gate.py` — prose
  that named the NIU block as uncosted.
- `ROADMAP.md` — item 2 replaced by its successor (the load/store unit's
  published queue limits and per-region throughput); the conditional NIU probe
  dropped from the silicon-session bundle.

## The load queue: two published statements, one mechanism, charged once

Landed 2026-08-06 — ROADMAP item 2, "the rest of the RV memory path", the
successor the NIU census produced. Three terms were named, all `isa_doc` and
all consumed by nothing. The outcome is one wired, one shown to be the *same*
term as the one wired (so charging it too would double-count), and one
measured and declined. No number was sourced, derived or invented; the only
value that moves in either YAML file is a note.

### The double-count question, settled by arithmetic rather than by taste

The two candidates sit a paragraph apart on the same page
(`WormholeB0/TensixTile/BabyRISCV/README.md`):

> **Maximum loads in flight** — 8 for core-local data RAM; "4 (in aggregate
> across all of these regions)" as a four-row rowspan over every other row.

> "Throughput of sustained loads is one per cycle if the load latency is less
> than five cycles. Otherwise, when the load latency is `N` cycles, the
> throughput of sustained loads is four such loads every `N - 1` cycles."

**They are one queue described twice.** Little's law reads the residency
straight off them: four loads in flight sustaining four loads per `N - 1`
cycles is a mean residency of exactly `N - 1` cycles per load. And the
converse is what settles which one to spend, because it is not a matter of
preference — *for every row either architecture publishes, the in-flight cap
is unreachable except in the regime the formula already limits*. A stream
issuing at the core's one load per cycle with a residency of `N - 1` has
`min(N - 1, 4)` loads in flight, so the cap of 4 binds only when `N >= 5` —
which is exactly the threshold `one_per_cycle_if_latency_under: 5` switches
the formula on at. Below it (core-local RAM at 2, the mailbox row at `>= 3`,
the GPR/config row at `>= 4`, Blackhole's L0 hit row at 2) the queue never
holds more than three, and the cap of 8 on core-local RAM is inert twice over.

So the formula is charged and the column is not. The formula is also the one
of the two that Blackhole's own silicon has been measured against
(`rv_load_indep`, below), and the one whose "four" is a number rather than a
table cell Blackhole does not print.

### What is charged

`RiscvCostModel.load_slots` = 4 slots; a load to a region whose latency is at
or above the five-cycle threshold takes the earliest-freeing slot and holds it
for `latency - 1` cycles; a load that finds every slot busy stalls until one
frees, under a new stall reason `load_rate`. Rows below the threshold take no
slot at all — "one per cycle" is what a single-issue core does anyway, and
charging them a smaller version of the same thing would be an invention.

The slot form is the sentence, not a generalisation of it: on a stream of one
latency it admits four loads every `N - 1` cycles and no more, which is
verbatim what is published; on a stream of *mixed* latencies — which the
sentence does not cover and every real kernel is — it is the only reading that
reduces to the sentence on every uniform stream. The unit tests pin the
uniform case at both ends (`4` back to back then a stall; the 37th load of a
sustained stream issuing on cycle 63, i.e. 1.75 cycles per load exactly).

Silicon: `rv_load_indep` — four rotating destination registers, so no load is
ever read early — reads **1.742** cycles per load on a Blackhole card where
four slots of seven cycles give **1.750**. That is a corroboration of the
charged mechanism, not its provenance, and it was already recorded on the
entry before this change consumed it.

One implementation note, because it was a real bug for a few minutes: on
Blackhole the *rate* depends on which of the two L1 rows a load pays, so the
L0 line lookup has to happen **before** the rate check, and `can_issue`'s
contract is that a refused cycle leaves the model exactly as it found it. A
lookup that evicted a line and was then re-offered on the next cycle would
evict a fresh line every cycle it stalled. `_l0_load` is therefore split into
`_l0_probe` (read-only) and `_l0_commit`, with a test that walks the tag array
across three stalled cycles.

### What it is worth: 4-9 cycles charged, 0-4 delivered

A/B on one binary — the same guards run twice, the second time with
`RiscvCostModel._load_throughput` stubbed off, so the two arms differ in this
term and nothing else — at `PUMP_CHUNK = 1`, i.e. one-cycle resolution:

| guard | cycles, term off | cycles, term on | delivered | charged |
| --- | --- | --- | --- | --- |
| `blackhole/three` | 13,415 | 13,415 | **0** | 4 |
| `blackhole/loopback` | 11,246 | 11,246 | **0** | 4 |
| `blackhole/nine` | 12,936 | 12,936 | **0** | 6 |
| `blackhole/six` | 85,778 | 85,780 | **+2** | 4 |
| `blackhole/four` | 105,534 | 105,534 | **0** | 4 |
| `wormhole/reduce` | 9,682 | 9,684 | **+2** | 9 |
| `wormhole/softplus` | 19,951 | 19,955 | **+4** | 9 |

At the `PUMP_CHUNK = 10` resolution the NIU instalment's table used, all seven
totals are unchanged to the cycle (85,780 / 105,540 / 12,940 / 13,420 /
11,250 / 9,690 / 19,960). `blackhole/six`'s PCC is **0.9982**, the same figure
as before the term existed.

**Why it is this small, and it is not a bug.** The load-use interlock gets
there first. Nothing in the tree issues sustained *independent* loads: a
barrier polls a counter and immediately compares it, a kernel loads an operand
and immediately uses it, so the core is already stalled on the dependent read
long before four loads are in flight. The four slots are reached only where
firmware deliberately batches independent loads — and there is exactly such a
path, named in the NIU instalment: tt-metal's `noc_local_state_init` and
friends issue five to ten `NOC_STATUS_READ_REG`s back to back "to hide latency
of NOC reg reads". That runs a handful of times per launch, and a handful of
stall cycles is what it costs.

Wormhole delivers most where Blackhole delivers least, for a documented reason: Blackhole
has an L0 data cache in front of L1, so most of its L1 loads pay the hit row's
2 and take no slot at all, while every Wormhole L1 load is `>= 8` and every one
of them is rate-eligible.

This is the fourth RV instalment in a row where charged and delivered differ
by an order of magnitude or more, and the ratio is now the *point* rather than
a footnote: these runs are not issue-limited, they are waiting on each other.

### Per-region request throughput: measured, and declined

> "If two clients are interacting through a memory region *other* than L1,
> then each client will be emitting its own stream of read/write requests
> against that region, and those streams will be combined into a single
> ordered stream as they reach the memory region. **Each memory region can
> process at most one request per cycle**" — `BabyRISCV/MemoryOrdering.md`,
> identically on both architectures.

This is the term the ROADMAP item described as "the one term that would make
two cores hammering the *same* NIU cost more than one core doing it". It is
not wired, and the first reason is that the workload it describes **does not
occur anywhere in the tree**.

Every MMIO request (loads *and* stores) in seven guard runs, keyed by tile,
by 64 KiB memory-map block — which separates NoC 0 at `0xFFB2` from NoC 1 at
`0xFFB3`, the mailboxes from the PCBufs, and so on — and by cycle. A
"collision" is two cores of one tile requesting the same region on the same
cycle, which is the only shape this term can ever charge:

| guard | MMIO requests | collisions | of requests | busiest region | NIU collisions |
| --- | --- | --- | --- | --- | --- |
| `wormhole/reduce` | 5,201 | 536 | 10.3 % | `0xFFE8` PCBufs/TTSync (341) | **0** |
| `blackhole/nine` | 12,662 | 689 | 5.4 % | `0xFFE8` (233) | **0** |
| `blackhole/loopback` | 6,864 | 233 | 3.4 % | `0xFFB1` tile control (112) | **0** |
| `blackhole/three` | 5,365 | 179 | 3.3 % | `0xFFB1` (112) | **0** |
| `blackhole/four` | 7,858 | 243 | 3.1 % | `0xFFB1` (112) | **0** |
| `wormhole/softplus` | 12,100 | 242 | 2.0 % | `0xFFB1` (112) | **0** |
| `blackhole/six` | 88,886 | 675 | 0.8 % | `0xFFE4` push buffers (375) | **0** |

**Zero collisions on either NIU register block, in any guard, over 138,936
MMIO requests.** The reason is structural rather than lucky: "NoC 0
configuration and command" and "NoC 1 configuration and command" are two
memory-map entries and therefore two regions, and tt-metal gives BRISC NoC 0
and NCRISC NoC 1. Resolved per core on `blackhole/three`: BRISC makes 299
requests to `0xFFB2` and 20 to `0xFFB3`; NCRISC makes 246 to `0xFFB3` and 1 to
`0xFFB2`. The two busiest MMIO pollers in every dataflow kernel in the tree
are polling *different regions*, so the term that would serialise them charges
nothing.

What is left is real but is neither what the item predicted nor large: tile
control / debug / status (`0xFFB1`), which every core touches during launch,
and the PCBufs/TTSync block (`0xFFE8`) on the two guards with the busiest
TRISC handshaking. **Charging one cycle per collision — the most this term
could conceivably deliver, which no RV term in this document has ever come
close to — is an upper bound of 0.2 % to 5.5 % of a guard's total**, and the
measured delivery of the term wired above (0-50 % of what it charged) says the
realised figure would be a small fraction of that.

Against that, the cost of building it, which is the second reason:

- **It is the first RV cost that is not per-core.** Every other quantity in
  `RiscvCostState` is one core's own; a region's request port is per tile and
  shared by five cores, so it needs a new shared object owned by `TensixTile`
  and threaded through `BabyRISCV.__init__` → `make_cost_state` on both
  arches, plus a decision about eth tiles and about cores built outside a
  device.
- **It puts cost state outside the parking proof.** `tt_sim/pe/rv/spin.py`
  parks a firmware loop by proving its state is a one-tick fixed point, over
  a signature that is by construction the *core's own* state
  (`RiscvCostState.spin_signature`). A charge that depends on what another
  core did this cycle is not in that signature, so a parked core's replayed
  charges could differ from what it would have been charged awake. Today the
  recogniser rejects any candidate loop containing an MMIO load outright,
  which happens to save the proof — but that would make the soundness of one
  module depend on an unrelated predicate in another staying exactly as it is,
  and that dependency deserves to be written down before it is relied on.
- **The single-core half is inert.** A single-issue core cannot issue two
  requests in one cycle, so there is no part of this term visible to one core
  in isolation: it is all-or-nothing on the cross-core plumbing.

So: deferred, with the census above as the reason rather than an intuition.
The measurement is cheap to repeat if a future workload puts two cores on one
region — a multi-core kernel sharing a semaphore is the obvious candidate, and
it would show up as collisions on `0xFFE8`.

### Arch scope, stated because the two pages differ here

Wormhole publishes both statements. **Blackhole's page publishes neither** —
it rewrote the Load/Store Unit section and dropped both the in-flight column
and the sustained-load sentence. What it keeps is the statement the formula is
about ("a latency of `N` cycles means that `N - 1` independent instructions
need to follow the load"), plus an eight-entry retire-order queue and the
advice to "use distinct destination registers for each of the seven
instructions following a load instruction": the same 8 and the same 7 as
Wormhole's "up to eight instructions ... the oldest non-retired load, plus (up
to) the next seven".

The term is charged on both, which is the file's ordinary deep-merge
convention (an override carries differences, not repetitions) and not a
Wormhole number pushed into a Blackhole gap — and the entry's Blackhole
corroboration is a direct measurement *of this formula on a Blackhole card*,
which is the strongest available statement that it belongs there. The
in-flight column, which Blackhole really does not print, is read on neither.
Both facts are now in the YAML notes rather than only here.

### What is still not charged, in this block

- **The Load/Store Unit's instruction queue** (8 deep on both arches, by two
  different descriptions). A third view of the same 8, and the load-use
  interlock's `N - 1` already spends it: at `N <= 8` the load leaves before
  the queue behind it can fill. It could only bite on Blackhole's `>= 12` L1
  atomic row, which no in-tree kernel reaches.
- **The "more in the case of access conflicts" tails** on the `>= 4` and
  `>= 7` rows, and "access port conflicts or bank conflicts" on L1. Still
  unquantified, and the per-region term above is the closest thing to a
  quantification of the first — one more reason it is worth revisiting if a
  workload ever makes it matter.
- **The three blocks in `RV_UNNAMED_REGIONS`** (MOP expander config, NCRISC
  IRAM, the Tensix instruction push buffers). None appears in any row of
  either table. Worth noting that the push buffers are the busiest colliding
  region on `six` — 375 of its 675 collisions — so if that block ever gains a
  published latency it arrives with a contention question attached.

### The gate

`driver/tests/cost_model_gate.py --jobs 4` **PASS**, all 44 guards it
discovers, and every poll-budget multiplier is the one recorded before it:
`blackhole/dramtop` 1×, `blackhole/two` 2×, `blackhole/offline` 4×. Nothing
moved onto a higher rung — the ladder is the gate's only measure of how much
slower a run got, and it did not register this change at all.

`pytest tt_sim/ driver/` 1,071 passed both with the model off and with
`TT_SIM_COST_MODEL=1` (1,063 before; eight new tests).

Run outside the gate, each of the 30 Blackhole guards as its own standalone
main under the model: 28 pass; `blackhole/two` and `blackhole/offline` fail at
the recorded 1× poll budget, which is what being budget-dependent *means*.
Both were re-run with this term stubbed off and fail **identically** without it
(`offline` the same 3 of 220 READ replies, `two` the same 100 of 100 elements),
so neither failure is this change's; the gate clears both at their usual
multiple.

Model off is byte-identical by construction: every line of this change is
inside `RiscvCostState`, which is `None` when the model is off.

### What changed in the repository

- `tt_sim/perf/model.py` — `RiscvCostModel._load_throughput`, giving
  `load_slots`, `load_slot_cycles` (per region) and `l1_miss_slot_cycles`;
  the class docstring's third bullet.
- `tt_sim/pe/rv/cost.py` — the slot check in `can_issue`, the `load_rate`
  stall reason, `_l0_load` split into `_l0_probe` / `_l0_commit`, the slots in
  the spin signature and its restore, and a docstring that now carries the
  double-count argument and the two declined terms.
- `tt_sim/perf/unit_costs.yaml` — `load_throughput` marked consumed with the
  measured worth, `load_latency`'s in-flight note rewritten as a decision, and
  a Blackhole note recording what that page does and does not print. **No
  cycle count, bound or provenance changed.**
- `tt_sim/pe/rv/cost_test.py`, `tt_sim/perf/model_test.py` — the new charge's
  arithmetic on both arches, the fast-row exemption, the stalled-load tag
  invariant, and the two published fours asserted equal.
- `ROADMAP.md` — item 2 rewritten to what is actually left.

## The issue loop: the residual on every rung-2 prediction, and it was the harness

Landed 2026-08-08. `ROADMAP.md` item 1 asked for an "issue-loop model" on the
grounds that "the residual on every rung-2 prediction is one constant
unmodelled issuing-core path (intercepts 77-94 cycles across four independent
series)". The premise is half right and the half that is wrong is the
important half. **The path was never unmodelled. The harness never ran it.**

Nothing was charged. There is no new table entry, no new provenance rank, no
edit to `unit_costs.yaml`, and not one simulated cycle moves on any in-tree
workload. What changed is that the rung-2 predictor now executes the same
program the measurement times, which closes 24-28 cycles of the intercept on
every series on both architectures and takes rung 2 from 6 retained entries to
14.

### The premise, reproduced

Both halves check out exactly as written, on the tree at `97635b2`:

| Wormhole series | intercept | slope |
| --- | --- | --- |
| L1 read diff-axis | **77** | +3.94 cycles/KiB |
| L1 read same-axis | **80** | +4.30 |
| L1 write diff-axis | **94** | +2.65 |
| L1 write same-axis | **89** | +2.83 |
| DRAM read | 79 | −0.65 |
| DRAM write | 129 | −3.21 |

The "four independent series" at 77-94 are the four L1 rows. Sole-cause
exclusion counts: **24** for `num_transactions per barrier != 1`, **2** for the
pattern (congestion) rule, 4 for `stateful`, 0 for the rest. Both numbers
reproduce on the nose.

### The axis nobody had looked down

The 24 sole-cause entries are the same six shapes at N = 4, 16, 64 and 256
transactions per barrier. Differencing along N is the one thing this dataset
does that is cleanly identifiable — unlike congestion, where three unknowns
move together, N moves alone with geometry, size, memory and direction all
held fixed. The marginal cost of one more transaction, at sizes below the
point where serialisation binds:

| | Wormhole | Blackhole |
| --- | --- | --- |
| L1 **write** | **22.0** | **23.0** |
| L1 read | 19 → 27 | 35.0 |
| DRAM read / write | 21 / 25 | 22 / 26 |

The write rows are the striking ones. 22.0 on Wormhole is flat to two decimal
places across N = 4 → 256 (a 64× range), across 64 B → 512 B, and across both
geometries — same-axis and different-axis agree to 0.01 cycles. A quantity
that ignores distance and size is not a network quantity. It is the issuing
core running instructions.

### The arithmetic that says which mechanism, done before anything was run

A constant intercept is consistent with several mechanisms — a fixed issue
latency, a per-request loop, a pipeline drain — and the roadmap item was right
that they need distinguishing. The N axis distinguishes them: a fixed latency
or a drain is paid **once per barrier** and cannot produce a slope in N; a
per-transaction loop is paid N times and can produce nothing else. The measured
number is a slope, so the mechanism is the loop.

Which loop is not a guess either. `ncrisc_noc_fast_write_any_len` in each
architecture's `noc_nonblocking_api.h` expands, at `DM_DEDICATED_NOC`, to a
`while (!noc_cmd_buf_ready(noc, cmd_buf));` poll, a fixed list of
command-buffer stores, and two counter increments. Count the instructions a
straightforward compilation gives:

| | stores | instructions | + interlock | dataset |
| --- | --- | --- | --- | --- |
| Wormhole write | 6 | 16 | 16 + 6 = **22** | **22.0** |
| Blackhole write | 7 | 17 | 17 + 6 = **23** | **23.0** |
| Wormhole read | 5 | 12 | 12 + 6 = **18** | 19 at low N |
| Blackhole read | 6 | 13 | 13 + 6 = **19** | 35 |

The six is the ">= 7" row of the RISC-V load-latency table — the row naming
"NoC 0 configuration and command", charged since [the NIU register
block](#the-niu-register-block-the-number-was-in-the-table-under-the-wrong-key)
landed on 2026-08-06 for entirely unrelated reasons. The instruction counts
come from gcc. **Neither number came from this dataset**, and their sum lands
on the measurement to 0.0 cycles on both architectures.

Better than the agreement is the *differential*. Blackhole costs exactly one
cycle more than Wormhole per write, and the reason is one line of Blackhole's
own header: it splits the destination across `NOC_RET_ADDR_MID` and
`NOC_RET_ADDR_COORDINATE` where Wormhole writes only the latter. One extra
store, one extra cycle, and the dataset says 23.0 against 22.0. A fitted
constant does not predict a cross-architecture difference of one cycle from a
header diff.

Checked rather than asserted: the reference build is
`riscv-tt-elf-gcc -O3 -march=rv32im` from tt-metal's own toolchain, on the C
above, run on a real simulated Wormhole and Blackhole tile. It produces
22 / 23 / 18 / 19, all four.

### What the data cannot distinguish, stated plainly

Within the per-transaction slope, the instruction stream and any **NIU
command-buffer occupancy** are not separable: both are per-transaction, both
serialise against the same poll, and the dataset holds one number. The
agreement above bounds the command buffer's contribution at *approximately
zero* on the write path — the instruction count alone already explains the
measurement — but "approximately zero" is an inference from a coincidence of
two independently sourced numbers, not a measurement of the command buffer.

The ISA docs are the reason it stays that way. `NoC/Counters.md` states the
protocol and no timing: *"After writing 1 to `NOC_CMD_CTRL` ... software must
not write to `NOC_CMD_CTRL` of any request initiator (at the same NIU) until
`NOC_CMD_CTRL` of the relevant request initiator reverts back to 0."* How long
that takes is not published anywhere in either tree. So the term is
`provenance: unknown` and may carry no number, which is also why the read gap
below is named and not charged.

### The one thing that is genuinely unmodelled, newly measured

Reads do not fit, and the misfit is the finding. tt-sim's read issue loop costs
18 (Wormhole) and 19 (Blackhole); the dataset measures 27 and 35 per
transaction at N ≥ 64. That excess — **9 cycles on Wormhole, 16 on Blackhole**
— is the initiator's outstanding-read-request credit limit. tt-sim's NIU
imposes no limit at all: `add_outstanding_noc_request` appends to a queue.

It is a different term from the one the roadmap named, it is now sized on both
architectures, and it is **not chargeable**: the number above is a measurement
on two parts, and no published statement or arithmetic on vendor numbers yields
it. `vendor_source_derived` is not available because there is no arithmetic
that separates the credit limit from the L1 read ports. It goes on the record
as a named, sized `unknown`, and the read rows at N > 1 stay excluded with that
reason written next to them.

### What was built

`tt_sim/perf/noc_issue_loop.py`: the per-transaction store list transcribed
from both architectures' `noc_nonblocking_api.h`, plus enough RV32I encoders to
assemble the loop and its barrier. `predict_timed_region` in the sweep loads it
onto the initiator tile's BRISC, releases the core and times the core's own
`START → DONE` markers — the same span `DeviceZoneScopedN` wraps.

`predict_cycles` is **kept**, unchanged, and the report runs both predictors
over the same points so the difference is printed rather than asserted. The
closed-form test that pins the network composition still points at it.

**This is a reconstruction of a compiled program, not a hardware constant**,
and that is why it is a harness and not a cost entry. The measured
per-transaction cost is the cost of *some* compilation of a vendor inline
function; a different compiler or a different tt-metal release would move it.
The chip has no opinion. What tt-sim does — execute whatever instructions the
kernel actually contains — is already the right answer, and adding a table
entry for the issue loop would double-charge every real workload.

### Before and after, on the same points

| Wormhole | n | min | median | max | mean |
| --- | --- | --- | --- | --- | --- |
| all retained, network only | 148 | 73 | 406 | 536,486 | 19,494 |
| all retained, + issue loop | 148 | **46** | **87** | **14,223** | **616** |
| N = 1 (the old retained set), network only | 60 | 73 | 90 | 349 | 118 |
| N = 1, + issue loop | 60 | **46** | **64** | **322** | **92** |
| N = 1, ≤ 512 B, network only | 24 | 73 | 79 | 201 | 87 |
| N = 1, ≤ 512 B, + issue loop | 24 | **46** | **54** | **178** | **62** |

| Blackhole | n | min | median | max | mean |
| --- | --- | --- | --- | --- | --- |
| all retained, network only | 148 | −28 | 356 | 268,799 | 10,041 |
| all retained, + issue loop | 148 | −52 | **68** | **7,654** | **340** |
| N = 1, network only | 60 | −28 | 92 | 152 | 82 |
| N = 1, + issue loop | 60 | −52 | **66** | **125** | **56** |

Per-series intercepts close by **24-28 cycles** on every N = 1 row on both
architectures, and by up to **2,685** on the N = 256 rows. Wormhole's four L1
series go 77 / 80 / 94 / 89 → **53 / 54 / 66 / 65**; Blackhole's go
93 / 99 / 80 / 83 → **67 / 72 / 55 / 58**.

**The spread question, answered honestly.** On *all* retained points the
absolute maximum residual is larger than the old sweep's 349 — but that is a
comparison between 60 points and 148, and the 88 new ones are burst rows that
multiply this file's already-recorded ~10 % L1 bandwidth optimism by up to 256.
Like for like, on the 60 points the sweep retained before, **every statistic
improves and none widens**: min 73 → 46, median 90 → 64, max 349 → 322, mean
118 → 92. The report prints all three scopes for exactly this reason.

**And one thing gets worse, on purpose.** Blackhole's DRAM-write row was
already the file's single `KNOWN_OVER_CHARGED` entry at −24; adding a real
issue cost on top of an already-too-large prediction takes it to −50. That is
arithmetic, not a new defect, and it does not change the row's status: the
over-charge is in `dram.access_latency` being derived from read figures, which
[the rung-2 instalment](#blackhole-and-the-first-place-the-model-over-charges-anything)
recorded and which splitting the entry by request action would fix.

### What is left in the residual, and why it may never be charged

46-58 cycles at ≤ 512 B, flat along all five axes including the new one. Two
things are in it and neither is hardware:

* **the profiler's own instrumentation** — `DeviceZoneScopedN`'s timestamp
  reads and their L1 stores sit inside the region it reports;
* **the barrier's discovery granularity** — the last acknowledgement lands at
  some cycle, and the core finds out on its next poll iteration, up to one poll
  period later. The predictor stops at the acknowledgement.

Charging either would mean putting tt-metal's measurement apparatus into a
model of a chip. The residual is the right size for them and it is flat, which
is the most that can be said without a second instrument.

### The gate

**PASS, exit 0**, and byte-identical: `dramtop` 1×, `two` 2×, `offline` 4× poll
budgets all unmoved, all 44 guards' values unchanged, all 30 Blackhole replay
guards pass, 1145 tests green with the model off and on. Every one of those is
a formality here and saying so is the point — **nothing outside
`tt_sim/perf/` was touched**, so no guard's cycle count could have moved, and
none did. This is the first rung-2 instalment for which "the gate cannot
fail" is a claim about the change's shape rather than a hope.

### What changed in the repository

- `tt_sim/perf/noc_issue_loop.py` — **new**. The store lists, the encoders, the
  program, and the arithmetic that predicts its cost.
- `tt_sim/perf/noc_issue_loop_test.py` — **new**. Encoders checked against
  words from the reference build; the store lists checked against the headers;
  the 22 / 23 / 18 / 19 claim run on both arches; flatness in N and in size;
  and a model-off control proving the six cycles are the model's.
- `tt_sim/perf/noc_dataset_sweep.py` — `predict_timed_region` alongside the
  unchanged `predict_cycles`; the N > 1 exclusion narrowed to everything
  outside the reconstructed L1 write path, with each of its three remaining
  reasons written out; `RESIDUAL_EXPECTATION` rewritten, including a retraction
  of its own "unmodelled issuing-core path" wording; a burst axis and a
  both-predictors readout in the report.
- `tt_sim/perf/noc_dataset_sweep_test.py` — one renamed rule.
- **No change to `unit_costs.yaml`, `costs.py`, `model.py` or anything under
  `tt_sim/network/`, `tt_sim/device/` or `tt_sim/pe/`.**

## The read floor: the number was wrong on one arch, and the name was wrong on both

*(Successor to the instalment above. Nothing is charged here; no executable
code changed at all. What changed is a headline number, a mechanism name, and
the addition of a card program that can settle it.)*

The previous instalment closed by naming one genuinely unmodelled term: an
excess of **9 cycles per transaction on Wormhole and 16 on Blackhole** on the
NoC read path, attributed to "the initiator's outstanding-read-request credit
limit", declared unpublished, and sent to the silicon session. Three things
were then done in the order the working rules ask for — reproduce the number,
sweep for provenance, and only then design anything. In that order:

1. **Blackhole's 16 reproduces exactly. Wormhole's 9 does not**, and the
   Wormhole reading has a different shape from the Blackhole one.
2. **No published bound exists**, in either architecture's ISA tree or in
   tt-metal's headers. Both sweeps are written out below so nobody repeats
   them.
3. **The name is wrong on both architectures.** The dataset already in the
   repository rules out a round-trip credit limit, using an axis nobody had
   differenced.

### The reproduction, and what it says instead

Differencing `noc_latencies.yaml` along the transactions-per-barrier axis, for
the unicast L1 read rows at 64-256 B (where nothing is link-bound), holding
geometry, size, memory and direction fixed:

| Wormhole, 64 B L1 read | N = 1→4 | 4→16 | 16→64 | 64→256 |
| --- | --- | --- | --- | --- |
| different-axis | 22.00 | **19.00** | 26.81 | **25.00** |
| same-axis | 19.33 | **19.00** | 26.81 | **25.01** |
| different-axis, `stateful` | 20.67 | 17.25 | 17.33 | **17.33** |

| Blackhole, 64 B L1 read | N = 1→4 | 4→16 | 16→64 | 64→256 |
| --- | --- | --- | --- | --- |
| different-axis | 29.33 | 34.92 | 34.96 | **35.01** |
| same-axis | 30.67 | 33.83 | 35.29 | **34.95** |
| different-axis, `stateful` | 23.33 | 34.08 | 33.98 | **34.00** |

Blackhole is one flat rate, 35.0, from N = 4 upwards; **35 − 19 = 16**, exactly
as recorded. Wormhole is not. It is two exact straight lines with a step
between them:

```
N ∈ {4, 16}    latency = 19 N + 283   (359, 587 — both exact)
N ∈ {64, 256}  latency = 25 N + 274   (1874, 6674 — both exact)
```

The 26.81 that became "27" is the secant across the step, not a rate. There is
no N at which Wormhole reads cost 27 cycles each. **The two Wormhole rates are
19 and 25**, and against a modelled issue loop of 18 the excesses are **1.0 and
7.0** — not 9. What is real on Wormhole is a **step of 6 cycles per transaction
that appears once the burst exceeds some depth between 16 and 64**, and that is
a smaller, differently-shaped claim than a flat 9.

The `stateful` rows are the ones that decide the mechanism, and they had not
been looked at because the exclusion ladder drops them:

- **Wormhole**: shortening the issue loop takes the sustained rate from 25.00
  to **17.33**, and removes the step entirely. A hardware floor of 25 cycles
  per read cannot exist on a part that sustains 17.33 with a shorter loop.
- **Blackhole**: the same shortening buys **1.0 cycle**, 35.0 → 34.0. The rate
  is pinned by something that is not the instruction stream, and that thing is
  worth about 15 cycles per read.

So the honest per-architecture statement is not "9 and 16". It is: **Blackhole
has a read-rate floor the issuing core cannot get under; Wormhole has a
burst-depth-triggered step worth 6.** Those may not even be the same mechanism.

### The dataset already refutes "credit limit", on both architectures

A credit limit of `K` outstanding requests makes the sustained rate `L / K`,
where `L` is the round trip. It is therefore obliged to move when `L` moves.
`L` moves by a large, known amount between the two geometries the dataset
measures — the N = 1 rows, which are one round trip each, differ by **88 cycles
on Wormhole** (205 same-axis against 293 different-axis) and **147 on
Blackhole** (226 against 373). The sustained rates do not move at all:

| | same-axis | different-axis | difference | `K` required to hide it |
| --- | --- | --- | --- | --- |
| Wormhole, N ≥ 64 | 25.01 | 25.00 | ≤ 0.01 | ≥ 8800 |
| Blackhole, N ≥ 64 | 34.95 | 35.01 | ≤ 0.06 | ≥ 2450 |

Both required values are impossible: `NIU_MST_REQS_OUTSTANDING_ID` is **8 bits**
in both architectures' `NoC/Counters.md`, and tt-metal's own self-throttle
stalls at 128. A credit limit small enough to bind at N = 64 would show a
per-geometry rate difference of several cycles. There is none, to two decimal
places, on either part.

**The limiting mechanism is distance-independent.** That is consistent with a
per-request service occupancy — at the initiator's NIU, at the responder's L1
read ports, or in a request FIFO — and inconsistent with a round-trip credit.
The roadmap's name for the term was a hypothesis and the data it was drawn from
already contradicted it.

### Provenance: two exhaustive sweeps, both negative

**The public ISA documentation.** All 319 markdown files of
`tenstorrent/tt-isa-documentation` were fetched and swept; the complete NoC
subtrees of both architectures were read in full (Wormhole `NoC/` 8 files plus
`NoC/Overlay/` 9; Blackhole — the directory is `BlackholeA0/`, not `Blackhole/`
— 6 files, with no `Ordering.md`, no `Alignment.md` and no `Overlay/` tree).
Terms confirmed **absent** from both NoC trees (they do occur elsewhere — see
the `DRAMTile` hit below, which is why "the NoC tree" is the right scope to
state): `credit`, `backpressure`,
`inflight`, `trid`, `out of order`, `concurrent`, `depth`, `slots`,
`command buffer`, `tokens`. The earlier "no credit or backpressure anywhere"
finding holds and now extends to the mechanism's other names.

What the sweep did find, none of which yields a per-architecture number:

- `NoC/Counters.md:23`, **identical in both trees** —
  `NIU_MST_REQS_OUTSTANDING_ID(i)`, for `0 ≤ i ≤ 15`, *"8 bits each (gets both
  incremented and decremented, so will only overflow or underflow if software
  has too many outstanding requests)"*. A counter, and the only documented
  consequence of many outstanding requests is counter overflow — no stall, no
  limit, and not per-arch.
- `NoC/MemoryMap.md`, both trees — *"Each NIU has four request initiators"*.
  Not per-arch.
- `WormholeB0/NoC/Ordering.md:20` — *"The recipient NIU can be acting on up to
  12 different request packets (and 4 different response packets)
  simultaneously"*. Recipient side, and Blackhole has no `Ordering.md` at all.
- `WormholeB0/TensixTile/BabyRISCV/README.md` has a **"Maximum loads in
  flight"** column giving **4** for the group that names *"NoC 0 configuration
  and command"*; Blackhole's page **deletes the column** and documents an
  *"eight-entry retire-order queue"* instead. A genuine per-arch 4 → 8, but it
  bounds RISC-V load instructions, not NoC transactions, and 4/8 is not 9/16.
- `NoC/README.md`, both trees — router-to-router latency **9 cycles**. A
  published 9 exists and is a hop latency, identical on both parts; matching it
  to Wormhole's excess would be numerology.

The single closest hit in the entire corpus is **not in the NoC tree at all**,
which is why the earlier search missed it. `WormholeB0/DRAMTile/README.md`
advises the exact behaviour that was hypothesised, and gives no number for it:

> Each router has a 2 KiB buffer per inbound port: each virtual channel has a
> guaranteed 32 bytes, and then the remaining 1½ KiB is dynamically shared, with
> each virtual channel able to claim up to 480 bytes from this shared pool. […]
> In the converse direction, when performing large reads, the headers of each
> read request consume 32 bytes of buffer space, so software is encouraged to
> **limit its number of outstanding read requests** to avoid buffers being
> filled by read request headers.

So an outstanding-read-request limit is a real, named, documented concern — as
*advice to software*, with the bound left to the reader. **Inference, labelled
as such and not charged**: 480 B claimable per VC ÷ 32 B per read-request header
is **15** headers per VC before that inbound port backpressures. That is 15, not
9; it is a *router inbound port*, not the initiator; and **there is no Blackhole
`DRAMTile/` tree at all**, so it has no per-arch counterpart and cannot produce
a pair. It is recorded here because it is the one place the mechanism is named,
not because it yields a number.

**tt-metal 0.74's headers.** `NOC_MAX_TRANSACTION_ID 0xF` and
`NOC_MAX_TRANSACTION_ID_COUNT 255` are **byte-identical** between
`internal/tt-1xx/wormhole/noc/noc_parameters.h:25-26` and
`internal/tt-1xx/blackhole/noc/noc_parameters.h:15-16`, as is
`NIU_MST_REQS_OUTSTANDING_ID(id) (0x10 + (id))` and the self-throttle
`while (NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(trid)) > ((NOC_MAX_TRANSACTION_ID_COUNT + 1) / 2));`.
`NUM_NOC_CMD_BUFS` is 4 on both. No `*_DEPTH`, `*_ENTRIES` or `*_SLOTS`
constant exists in either file. The read path differs by exactly one store
(Blackhole adds `NOC_TARG_ADDR_MID`), which is the 5 → 6 already transcribed in
`tt_sim/perf/noc_issue_loop.py`.

**Verdict: there is no published bound, and no arithmetic on published numbers
produces a per-architecture pair.** Every candidate is either identical across
the two parts (16 trids, 255 per trid, 4 initiators, 4 command buffers, 16 VCs,
the 9-cycle hop) or per-arch with the wrong values. This closes the search; it
should not be repeated.

### The one artefact worth going to a card for

Two independent sources describe a structure Blackhole has and Wormhole does
not, and **neither gives its size**:

- `BlackholeA0/NoC/MemoryMap.md:21` — `NIU_BASE + 0x0064`, *"NIU request FIFO
  status"*, read only, 8 bytes; and `:237`, `NIU_CFG_0` bit 16, *"Request FIFO
  enable"*. There is no section, no anchor and no depth anywhere in the
  repository. Wormhole's `NIU_BASE + 0x054` is *"NIU combined request initiator
  status"* instead and its `NIU_CFG_0` has no such bit.
- tt-metal `blackhole/noc/noc_parameters.h:56` —
  `#define CMD_BUF_AVAIL (NOC_REGS_START_ADDR + 0x64)` with the comment
  `[28:24], [20:16], [12:8], [4:0]`: four 5-bit per-command-buffer availability
  fields, so a depth of at most 31. Wormhole has no analogue, and **no code in
  tt-metal references it**.

A per-architecture NIU request FIFO is exactly the shape a per-architecture
rate floor needs, it is **readable**, and its depth is precisely what the
documentation omits. That is the thing a card session should read.

### `perfbench/nocreadbench` — the probe, and how it separates the two

The instalment above stated the blocker as *"there is no arithmetic that
separates the credit limit from the L1 read ports"*. There is not. A
measurement does, two ways — and the first needs no arithmetic at all, which is
why the separation problem does not arise for it.

**Directly.** `NIU_MST_REQS_OUTSTANDING_ID(0)` counts the initiator's own
in-flight requests and is readable from a kernel. Sample it inside a burst:

- if it **plateaus below the burst length**, the initiator does hold a bounded
  number in flight, and that bound is *read off a register* — no arithmetic, so
  nothing to separate;
- if it **climbs with the burst length**, there is no initiator-side limit, the
  cap is downstream, and the term is **retired rather than sized**.

The sample costs a `≥ 7`-cycle NIU load with a six-cycle interlock inside a loop
whose whole per-iteration cost is under 40 cycles, so the kernel runs the burst
**twice**: once timed and unsampled, once sampled and untimed. A rate is never
reported from a sampled loop.

**By shape.** Each experiment moves one axis, and the four candidates disagree
about every one of them. `H1` credit limit; `H2` responder L1 read port; `H3`
initiator NIU request occupancy; `H4` the initiator's L1 *write* port, which
every row of tt-metal's dataset loads maximally because every transaction in it
lands at the same address.

| experiment | varies | H1 | H2 | H3 | H4 |
| --- | --- | --- | --- | --- | --- |
| `dist` | hop distance to the one source | **rises** | flat | flat | flat |
| `srcfan` | number of distinct source tiles S | flat | **falls ∝ 1/S** | flat | flat |
| `dstspread` | stride between landing addresses | flat | flat | flat | **falls** |
| `srcspread` | stride between source addresses | flat | falls iff banks | flat | flat |
| `size` | bytes per transaction | rises | flat, then link-bound | flat, then link-bound | flat, then link-bound |
| `burst` | N — the control against `noc_latencies.yaml` | | | | |

`srcfan` is the axis the shipped dataset **structurally cannot provide**: every
`ONE_FROM_ONE` row has exactly one source tile, so responder-side and
initiator-side effects are perfectly confounded in all 740 of them. Fan the
same burst over 2, 4 and 8 equidistant sources and they come apart.

The program is `perfbench/nocreadbench/`, in the shape `nocbench` established:
one C++ host that builds against `TT::Metalium`, one data-movement kernel, one
CSV with **the physical coordinates recorded in every row**, and a `run_card.sh`
that prints a verdict and names the file to send back. It is checked in with
its Wormhole simulator CSV, whose reading is a **known null and not a result** —
tt-sim's NIU appends to an unbounded queue, so the counter reads 0 and every
axis is flat by construction. It is run only to prove the harness executes,
exactly the role `nocbench`'s `INVALID` verdict plays for congestion.

### What was deliberately not built

**No credit-limit mechanism was added to `tt_sim/network/tt_noc.py`**, not even
one defaulting to unlimited. The roadmap has precedent for shape-first work —
item 2's endpoint occupancy — but that precedent holds because the *shape* is
known to be right and only the number is missing. (That item landed on
2026-08-09, and the distinction held: the shape was right, and the number turned
out to be already in the table rather than missing. See ["Endpoint occupancy: the
queue was missing, the number was already
there"](#endpoint-occupancy-the-queue-was-missing-the-number-was-already-there).)
Here the shape is what the
evidence just contradicted: the mechanism that would have been built (bound the
outstanding-request queue, stall the initiator at `K`) predicts a rate of
`L / K` that rises with distance, and both parts say the rate does not move with
distance at all. Building it would install the one hypothesis the data ranks
last, and an unused mechanism is not neutral — it is a branch on the hot path of
every NoC request and a shape that the next person reads as settled.

The honest position is that `add_outstanding_noc_request` appending to a queue
is **not currently known to be wrong**, and stays until a measurement says which
of `H2`/`H3`/`H4` to build.

### What would make it chargeable

Nothing that exists today. Specifically:

- **`isa_doc`** — would need the Blackhole NIU request FIFO's depth published.
  The register is documented to exist; only its size is missing. This is a
  documentation request to Tenstorrent, it is one number, and it is by far the
  shortest route from here to a chargeable term.
- **`isa_doc_derived` / `vendor_source_derived`** — unavailable. Every
  outstanding-request quantity in both sources is identical across the two
  architectures, so no arithmetic over them can produce a per-arch pair.
- **`corroboration`** — where a `nocreadbench` result goes: a measurement on one
  part, attached to an existing entry, never provenance, never a number the
  model spends.

If the FIFO depth is never published, the honest end state is a **permanently
named `unknown`** — sized, per-arch, with the mechanism identified by the probe
and the reason it may carry no number written next to it. That is the same end
state `noc.congestion` has, and it is not a failure.

### Also worth an hour, and needing no card

Wormhole's non-stateful read loop measures 19.00 at N ≤ 16 against a modelled
18 — good — and 17.33 when shortened. The gap between the modelled loop and the
step-free regime is one cycle, so the reconstruction is sound. But
`ncrisc_noc_fast_read_any_len` wraps the transcribed store list in a
`while (len_bytes > NOC_MAX_BURST_SIZE)` test that the reconstruction does not
model, and the estimator kernel reaches it through `Noc::async_read`, which
passes `read_req_vc = NOC_UNICAST_WRITE_VC` — **virtual channel 1, the unicast
*write* VC** — where the stateful path leaves the sticky VC alone. Two read
paths on different virtual channels is a plausible, cheap explanation for why
only the non-stateful Wormhole rows show the burst-depth step, and settling it
needs a disassembly of the estimator kernel and a VC argument in the probe, not
silicon.

### The gate

**PASS, exit 0.** Byte-identical by construction: this instalment changes **no
executable code at all** — the new tree is C++ that only a card or an explicit
`perfbench/run.sh` invocation compiles, and nothing under `tt_sim/` or `driver/`
was touched. `dramtop` 1×, `two` 2× and `offline` 4× poll budgets unmoved, all
44 guards' values unchanged, model on and off identical.

### What changed in the repository

- `perfbench/nocreadbench/` — **new**: `README.md` (the hypothesis table, the
  degenerate-run checks, and what happens to the number afterwards),
  `run_card.sh`, `src/CMakeLists.txt`, `src/nocreadbench.cpp`,
  `src/kernels/nocreadbench_layout.h`,
  `src/kernels/dataflow/reader.cpp`, and `src/nocreadbench-wormhole-sim.csv`
  (the null).
- `perfbench/README.md` — the new tree in the tree diagram and the table.
- `docs/plans/cost-model.md` — this section.
- **No change to `unit_costs.yaml`, `costs.py`, `model.py`, or anything under
  `tt_sim/` or `driver/`.**

## The second rung-3 sample: sixteen probes corroborate, three contradict, and a retraction goes back in play

Every rung-3 conclusion in this document rested on **one** tensixbench run. The
2026-08-09 card session took a second, at the same parameters, plus five other
probes aimed at named open bullets. This section records what the six say. It
changes **no table**: a silicon measurement is `corroboration`, never
provenance, and nothing below is charged.

The session's own verdict lines are not repeated here uninspected. Several were
checked against the raw CSVs and one of them is wrong; that one is
"Phase R", below.

### What was run

`/mnt/ramdisk/tt_traces/card-session-blackhole/`, 13:41, a harvested Blackhole
(addressed worker columns `{1..7, 10..14}`, two chips present, auto-discovery
downgraded to 1×1). `tensixbench` was rebuilt from source at the head of the
run — the session log records `rebuilding tensixbench (source newer than the
binary: raw_probes.cpp)` — and invoked with `--dvalid-once`, the X1
configuration, which its summary confirms as `dvalid setup: once -- one
SETDVALID for the tile (X1, de-confounded)`.

### Sixteen probes corroborate, and so does phase B

Of the 19 series the sweep's exclusion ladder retains, **16 agree with the X1
dataset to 0.012 cycles/instruction or better**, which is a third of the
instrument's own declared resolution. `NOP` and `loop_overhead` are not merely
close but **bit-identical** across the two runs — 2137/4235/6347/8459 and
76/140/204/268 raw cycles — so the timer, the control subtraction and the clock
are the same instrument on both days. `ADDDMAREG`, `CMPDMAREG`, `MULDMAREG` and
`SHIFTDMAREG` all read 2.973 in both runs, and the whole SFPU family reads
0.998 in both.

Phase B corroborates too, and it is the half that matters to the tables:

| | X1 (run 1) | run 2 | delta |
|---|---|---|---|
| LoFi, cycles/`matmul_tiles` | 34.92 | 35.20 | +0.8 % |
| HiFi2 | 52.47 | 52.55 | +0.15 % |
| HiFi4 | 86.12 | 86.13 | +0.01 % |
| marginal MVMUL, `(HiFi4 − LoFi)/48` | **1.067** | **1.061** | −0.6 % |

The ~1.07 figure §6.1 rebuilt the MATH occupancy argument on **reproduces to
0.6 %**. That is the corroboration the roadmap asked for, and it is the one
that licenses the tables' `occupancy: 1`.

### Three probes contradict, and they are the three that matter

`MVMUL`, `ELWADD` and `ELWMUL` — the phase-A MATH probes — do not reproduce at
any thread count:

| probe | | t1 | t2 | t3 | aggregate at t3 |
|---|---|---|---|---|---|
| `MVMUL` | X1 (run 1) | 5.989 | 12.012 | 18.035 | 6.01 |
| `MVMUL` | run 2 | **0.998** | **6.080** | **12.100** | **4.03** |
| `ELWADD` | X1 | 5.976 | 11.970 | 17.970 | 5.99 |
| `ELWADD` | run 2 | 0.998 | 6.067 | 12.063 | 4.02 |
| `ELWMUL` | X1 | 5.974 | 11.970 | 17.969 | 5.99 |
| `ELWMUL` | run 2 | 0.998 | 6.066 | 12.063 | 4.02 |

Every column moves by very nearly exactly 6 cycles/instruction. This is not
drift and not scatter: it is the same offset at t1, t2 and t3, in a run whose
other sixteen series are bit-identical or within 0.012.

Two further facts pin it down.

**Run 2's MATH t1 series is bit-identical to its own `NOP` series.** Raw cycles
2137 / 4235 / 6347 / 8459 for `MVMUL`, against 2137 / 4235 / 6347 / 8459 for
`NOP` and 2137 / 4235 / 6347 / 8459 for `INCRWC`. So run 2's `t1` MATH reading
is not a measurement of the Matrix Unit at all — it is the front end's 1-IPC
floor, which the sweep already marks `testable: no`. It cannot confirm the
table and it cannot refute it.

**Run 2 reproduces the `--dvalid-per-thread` control, not the `--dvalid-once`
run it was configured as.** Compared row by row against the two tracked
datasets:

- against `tensixbench-blackhole-dvalid-per-thread.csv` (the *confounded*
  control): **479 of 480 common rows agree within 1 %**, worst case 1.0 %.
- against `tensixbench-blackhole.csv` (X1, the configuration run 2 actually
  requested): agreement fails by **up to 83 %**.

And it is not a one-off. `tensix-warm`, a **separate process** in the same
session, also `--dvalid-once`, reads 0.999 / 6.064–6.071 / 12.063–12.083 — run
2's numbers, not X1's. The effect reproduced twice in one session.

### What this does and does not license

It does **not** license calling run 2 a silicon finding. The most economical
reading is a benchmark-configuration difference: the X1 dataset was taken on an
older `raw_probes.cpp`, before `c752514` (the source-format axis) and `0daab58`
(the Src-bank release), which is why its `#` header lacks the `src_format=` /
`src_style=` / `variants=` tokens. `0daab58`'s own header comment records that
a completed `SETDVALID` run **leaves `SrcA[0]`/`SrcB[0]` owned by the Matrix
Unit** and that only `tt-smi -r 0` clears it, while deliberately giving the
`--dvalid-once` and `--dvalid-per-thread` paths no release. A phase-A MATH
probe's cost turns on exactly that state, so "which run started on a dirty
card" is a live confound that neither run controlled.

It does, however, **withdraw the support** under one published claim. §X1
concluded that "with one legal `SETDVALID` the matrix unit is a plain shared
port", on the strength of a t1 baseline of 5.989 that made aggregate throughput
flat at ~6.0 across one, two and three threads. That conclusion came from
comparing X1 against the per-thread control — a comparison in which **the
dvalid mode and the binary changed together**. The comparison now available
holds the binary fixed and varies only the mode, and it finds the two modes
give the *same* answer to within 1 %. So the 6× step the retraction attributed
to dvalid legality tracks the binary or the card state, not the mode. On run
2's numbers aggregate throughput is **not** flat — it degrades 0.998 → 3.03 →
4.03 from one thread to three — which is the shape the retraction retracted.

At t1 specifically the retraction's premise does not hold up on inspection
either: with one active thread (TRISC1, the only thread `t1` activates)
`--dvalid-once` and `--dvalid-per-thread` both issue exactly one
`TTI_SETDVALID(3)` from exactly one thread. They are the same program. They
cannot differ by 6 cycles/instruction *because of the dvalid mode*, and the
fact that the two banked datasets do is itself evidence that something else
moved.

**Nothing here is charged, and nothing here needs to be.** The MATH occupancy
of 1 rests on phase B's MOP-issued marginal cost, which reproduced to 0.6 %.
Both phase-A regimes remain `corroboration`. What must change is the
*confidence* attached to the phase-A figure: it is now one observation against
one, in a configuration that has been shown not to reproduce across sessions,
and the honest record is that **phase A cannot currently measure the Matrix
Unit reproducibly at all**.

The experiment that would settle it is cheap and specific: on one card, in one
session, run `--dvalid-once` and `--dvalid-per-thread` back to back, each
immediately after a `tt-smi -r 0`, and again each on a deliberately dirtied
card. Four runs, one binary. That separates mode from card state, which is the
pair no run so far has held apart.

### The warm-up bias, measured, and smaller than the term it sits next to

`tensix-warm` is the same phase A with `--blocks 64`, so the cold `n = 32`
burst is out of the fit. The difference between it and `tensix` **is** the
warm-up bias, and it is now measured rather than assumed:

- over all 114 common phase-A series: median **−0.0000**, mean −0.0008, range
  **[−0.0222, +0.0208]** cycles/instruction.
- over the **19 t1 series the exclusion ladder actually retains**: range
  **[−0.0022, +0.0013]**, median +0.0010.

Set that against the `resol` the sweep declares for the same series,
0.033–0.034. `resol` is the sum of two terms, and the control over-subtraction
alone is `slope(loop_overhead) / unroll = 2.00 / 64 = 0.031`. So on the
retained series **the warm-up contributes about 6 % of the resolution and the
control subtraction about 94 %**, and R² was already 0.99994–1.00000 before the
cold burst was dropped.

**Recommendation: do not make `--blocks 64` the default.** It doubles the work
per point to remove a bias fifteen times smaller than the resolution it sits
inside, and it changes no verdict — the residual table moves from −0.002/−0.027
to −0.001/−0.029, both still "inside the instrument". Keep it as the option it
is. The valuable output is the bound: anyone tempted to attack `resol` should
now attack the **control subtraction** (a larger `unroll`), because the cold
burst is not what is limiting it.

### The instruction-fetch step: a ramp, not a cliff

Phase G's three `--gset` runs bracket the boundary phase F left between 4 KiB
and 8 KiB. The `g_1024` control is **bit-identical in all three files**
(33169 / 65861 / 98778 / 131676 raw cycles), so the three runs are directly
comparable without a cross-build correction:

| loop body | cycles/instruction | step over 4096 B |
|---|---|---|
| 4096 B (`g_1024`) | 1.000 | — |
| 5120 B (`g_1280`, gset 0) | 1.153 | +0.153 |
| 6144 B (`g_1536`, gset 1) | 1.252 | +0.252 |
| 7168 B (`g_1792`, gset 2) | 1.252 | +0.251 |
| 8192 B (phase F) | 1.251 | +0.251 |

**What the three points pin.** The step is *graded*, not a cliff: 5120 B sits
at 61 % of the full step. And it **saturates at 6144 B** and is flat to ±0.001
across three further footprints (6144, 7168, 8192). A single-point capacity
cliff is therefore excluded *as a description of the whole boundary* — whatever
the mechanism, its cost is complete by 6144 B and constant after.

**What they cannot pin.** 4608 B and 5632 B were never built, so the onset is
bracketed only to `(4096, 5120]` and the completion only to `(5120, 6144]`.
These data cannot distinguish a smooth ramp from two smaller steps inside those
brackets, and — as phase G's own read-out insists — narrowing a boundary in
loop-body size still does not license the noun "cache". The plateau value,
1.252, is suggestively 5 cycles per 4 instructions, but one part and one
campaign cannot turn that into a mechanism.

Resolving the onset needs the two missing builds, 4608 B and 5632 B, and
nothing else.

### The Tensix instruction queue: resolved, and per-thread

The roadmap carried "~31–32-entry queue depth is a lower bound still growing".
**It is not still growing.** `rv-qdrain`'s phase Q, loop form, out to n = 1024:

```
backlog:   n=16 +0   n=32 +28   n=64 +65   n=128 +65   n=256 +65   n=512 +65   n=1024 +65
marginal:  16->32 1.25   32->64 1.84   64->128 3.00   128->256 3.00   256->512 3.00   512->1024 3.00
```

The backlog reaches 65 cycles at n = 64 and then does not move across **four
consecutive doublings**, while the marginal cost pins to 3.00 against a drained
rate of 3.000 — the core is back-pressured from n = 128 onward. That is
saturation, unambiguously, and n = 1024 was more than enough; n = 128 would
have sufficed. The cross-check run reads the same shape at a different absolute
(83 cycles, flat from n = 64). So the sweep is **not** a lower bound for want of
burst length, and no longer burst is needed.

The two forms give different absolutes and the difference is understood: phase
Q's ~65 cycles / 3.000 = **~22 instructions** drops the reference burst's own
occupancy, which phase S adds back and reads as **~31 entries** at one thread.
Across all nine measured slots the depth lands in **27–33 entries**, which is
the honest spread; "~31–32" is the centre of it, not a resolved constant.

**The queue is per-thread.** Phase S's discriminator, which `rv-qdrain` could
not run (it was invoked `--variants t1` and says so) but the main `rv` run
could:

| | run 2 `rv` | run 2 `rv-cross` | shared predicts | per-thread predicts |
|---|---|---|---|---|
| t2 vs t1 | 32 vs 31 = **1.03×** | 30 vs 31 = **0.97×** | 0.50× | 1.00× |
| t3 vs t1 | 33 vs 31 = **1.06×** | 33 vs 31 = **1.05×** | 0.33× | 1.00× |
| spin control | 31 (1.00×) | 31 (1.00×) | 1.00× | 1.00× |

Four independent readings, all within 6 % of the per-thread prediction and all
a factor of two-to-three from the shared one. This is a decisive answer, not a
marginal one: **each baby core has its own Tensix instruction queue.** The spin
controls sitting exactly at 1.00× is what makes it readable — "another core is
awake" costs nothing, only "another core is issuing" does.

### `rv-pairs`: the cleanest repeatability in the set

A focused single-thread re-run of the store-coalescing, multiply and divide
probes, against the banked first sample and both of run 2's full sweeps:

| probe | run 1 (tracked) | run 2 `rv` | run 2 `rv-cross` | run 2 `rv-pairs` |
|---|---|---|---|---|
| `rv_mul_indep` | 0.999 | 0.999 | 0.999 | 0.999 |
| `rv_mul_dep` | 1.985 | 1.985 | 1.985 | 1.985 |
| `rv_div` | 33.001 | 33.001 | 33.001 | 33.001 |
| `rv_store_coalesce` | 0.999 | 0.999 | 0.999 | 0.999 |
| `rv_store_spread` | 5.290 | 5.290 | 5.290 | 5.265 |

Four independent samples, agreeing to the third decimal on four of five probes
and to 0.5 % on the fifth, every R² = 1.0000. The Blackhole half of this
comparison is as settled as this instrument can make it; the cross-arch half
still waits on a Wormhole part, which the session did not have.

### Phase R: the gate that fired is not the one the name suggests

`TTRVBENCH_VALID_R: no (2 checks failed)` in both run-2 files. The natural
reading — and the one the session summary invites — is monotonicity. **It is
not.** riscvbench's phase counters are incremented from two places, and phase
R's failures are all the other one: `R² < 0.99`, printed as `<-- NONLINEAR`.
There is not a single `NOT MONOTONE: r/` line in either file; all 16 of those
are phase Q.

Every failure is the same probe, `rv_store_spread`, and only ever its
multi-thread variants:

| file | slot | R² |
|---|---|---|
| `rv.out` | t2 thread 0 | 0.9843 |
| `rv.out` | t3 thread 0 | 0.9863 |
| `rv-cross.out` | t2 thread 0 | 0.9702 |
| `rv-cross.out` | t2 thread 1 | 0.9893 |

Two runs at **identical parameters** fail a **different set of slots**, and
`t3 thread 0` fails in one while `t3 thread 1`/`t3 thread 2` fail in neither.
That is scatter about a threshold, not a defective point. Two further checks
confirm it: dropping the cold `n = 32` point does *not* reliably rescue the fit
(`rv` t3/thr0 goes 0.9863 → 0.9826, `cross` t2/thr1 goes 0.9893 → 0.9734), so
it is not a warm-up artefact either; and the raw series curve in *opposite*
directions between the two runs — `rv` t2/thr0 has increments 8765/11264/15353
(convex) where `cross` t2/thr0 has 16407/9135/8225 (concave).

**It is also new only in part.** Run 1's `rv.out` did read `VALID_R: yes`, but
run 1's `rv-cross.out` read `VALID_R: no (1 checks failed)` — alongside C, Q, F
and G, i.e. run 1's cross-check was the noisier of that pair overall. So phase
R has failed in three of the four files across the two sessions, always on the
same probe.

**The cost-relevant column is untouched.** `rv_store_spread` at t1 reads
R² = 1.0000 in all four samples, and `rv-pairs` — single-thread only — reports
`TTRVBENCH_VALID_R: yes` with every probe at R² = 1.0000. Nothing that feeds a
table is in question.

**Does the verdict logic need changing? Yes, but not the way phase Q's does.**
Phase Q's exemption is for `q_ctrl`, a probe with no burst to be monotone in;
phase S's is a *measured* tolerance derived from a repeat probe. Neither
transfers: phase R's failures are not concentrated at small n (they occur at
n = 128 as readily as n = 32), so an n-threshold exemption would not catch them.
The defensible narrow change is to **exempt the multi-thread variants of
`rv_store_spread` from the R² gate specifically**, because the phase's
linearity premise does not hold for it: two or three cores spreading stores
across the same L1 have no reason to cost a constant per store, and requiring
them to is asking the gate to enforce something the probe was not built to
show. The alternative, and it is the better one if the phase gains a repeat
probe, is phase S's device — run `rv_store_spread` twice and let the observed
disagreement set the R² floor rather than a constant 0.99. Either way the fix
should be recorded as a tolerance with a measured basis, not a loosened
constant.

Left as-is, the cost is small but real: `TTRVBENCH_VALID: no` on every
multi-thread run makes the whole-run marker useless as a gate, which is
precisely what the per-phase markers exist to avoid.

### The congestion re-read, and one number to correct

The congestion result was already analysed; regenerating both reports from the
run-2 CSVs (the two `.report.txt` files in the session directory are stale
11:25 leftovers containing only a `ModuleNotFoundError`, and were ignored)
reproduces it. `noc.blackhole.csv` gives `RESULT: CONGESTION MEASURED`, FLAT at
64 B (−0.01) and 512 B (+0.01), SATURATING at 2048 B (+2.49), 8192 B (+10.87)
and 16384 B (+22.49); `noc-epoch.blackhole.csv` gives +2.63, +10.97 and +22.47
for the same three and is correctly `INVALID` for congestion on its five
zero-overlap runs.

Two caveats belong on the record. **The hop coefficient is 8.85–9.03, not
8.50.** The two run-2 files fit `4364.0 + 9.03 * hops` and
`4361.7 + 8.85 * hops` respectively, r² 1.00 — a closer match to the ISA docs'
~9 than the 8.50 that has been quoted. But that line is fitted over **three
aggregated round-trip levels**, not 24 independent points: the per-family fits
(`row`, `col`, `diag`) against forward hops come out *negative* with r² of
0.02–0.44, so the hop effect is not resolvable within a family. **And the
saturating slopes have r² 0.34–0.39.** `SATURATING` is a shape verdict off a
poor linear fit; the magnitudes (+2.49, +10.87, +22.49) reproduce between files
to 1–6 % but are not well-determined coefficients, and the sweep's own
one-part-one-campaign caveat applies with full force.

### Nothing contradicts a charged cost

Explicitly, because it is the question the session existed to answer: **no
measurement in this set contradicts a value in `unit_costs.yaml`.** The one
result that could have — the phase-A MATH reading — does not, because run 2's
t1 sits at the front end's 1-IPC floor and is `testable: no` by construction,
and because the quantity the table describes is phase B's MOP-issued marginal
cost, which corroborated to 0.6 %. The four THCON `range` entries read 2.973
against a charged 3.00 in both runs, inside the instrument both times. The
19-series residual verdict is `yes` — the table is a floor — in run 1, run 2
and the warm run alike.

### What changed in the repository

- `docs/plans/cost-model.md` — this section.
- **No change to `unit_costs.yaml`, `costs.py`, `model.py`,
  `tensix_instruction_costs.yaml`, or anything under `tt_sim/`, `driver/` or
  `perfbench/`.** No provenance entry was added, no coefficient was fitted into
  a table, and no `estimated` entry exists anywhere as a result of this
  session.

## Endpoint occupancy: the queue was missing, the number was already there

ROADMAP item 1's first bullet — *"a second request is not queued behind the
first: latency without contention. Pure model shape; no new data needed"* — and
it is the rare case where both halves of that claim survive checking. The shape
really was missing, the duration really is already in the table at
`isa_doc_derived`, and **no number entered any file**. What changed is where
`dram.channel_serialisation.bytes_per_cycle: 24` is *spent*: it was charged as a
latency, once, per request; it is now also held as a **resource**, so a request
arriving while the channel is busy waits for it.

It is the same move ["The congestion step,
wired"](#the-congestion-step-wired-one-number-spent-a-third-time) made — a
figure already in a table, spent on one more axis rather than replaced by a
bigger one, there `noc.hops.router_to_router.throughput_flits_per_cycle` going
from the injecting NIU to every router link a packet crosses. It is deliberately
the *only* move available here, because every alternative shape needs a number
nobody publishes.

### The premise, checked before anything was built

Three 4 KiB reads handed to a Wormhole DRAM endpoint on the same cycle, model
on. Before:

```
delayed_arrivals -> {143: [req, req, req]}
```

All three serviced on cycle 143. `DRAMEndpointNUI.transmit` added
`service_cycles + channel_excess` to each arriving request independently; there
was no state at the endpoint that one request could leave behind for the next.
End to end, four concurrent 4 KiB reads from one Tensix tile landed 128 cycles
apart — `ceil(4096 / 32)`, the **NoC link's** rate, because the only queue on
the return path was the DRAM NIU's own injection port.

So the endpoint was not merely uncontended: **a tt-sim Wormhole DRAM channel
sustained 32 B/cycle, above the 24 GB/s the ISA docs publish for it**, and the
model was over-predicting DRAM throughput by a third while under-predicting
every latency around it. That is a sharper statement of the gap than "occupancy
is not modelled", and it is what makes the fix a floor rather than a guess: the
bytes have to cross a 24 B/cycle bus, so the channel cannot be busy for less
than `ceil(N / 24)`.

The same three reads now land at 143, 314, 485 — exactly `ceil(4096 / 24)`
apart — and the four concurrent reads come back at 23.95 B/cycle.

### The duration, and why only half of it is chargeable

Endpoint occupancy has a shape *and* a duration, and the honest answer splits
them:

- **The channel data bus: derivable, and already derived.**
  `dram.channel_serialisation` is `isa_doc_derived` from
  `dram.bandwidth.per_channel_gb_per_s: 24` (`isa_doc`, `wh_dram#performance`)
  at the `isa_doc` 1 GHz clock. A transfer of N bytes occupies the bus for
  `ceil(N / 24)` cycles. Nothing was fitted, nothing was measured, and the
  quantity was *already being spent* as `channel_excess_cycles`.
- **The device behind the bus: `unknown`, and therefore uncharged.**
  `dram.access_latency` is 99 on Wormhole and 126 on Blackhole, and it is a
  **latency**. Nothing in the ISA docs, tt-metal or ttsim publishes the
  re-issue interval or the pipelining depth behind it. Reading the latency as
  an occupancy would assert that a channel serves one request per 99 cycles —
  0.32 B/cycle for a 32-byte access, against 24 B/cycle published on the same
  page. It would have been the largest single over-charge in the file. It is
  not charged, and `DramCostModel.device_occupancy_modelled` names the gap so a
  report cannot imply otherwise.

The bound is charged at its low end in both directions: `ceil(N / 24)` is the
shortest the endpoint can possibly be busy, and the docs' own 92 % achievable
fraction — which would make the true occupancy ~9 % *longer* — is deliberately
not folded in, exactly as it is not folded into the latency term.

### Corroboration, from the same page, and not consumed as a number

`wh_dram#performance` prints a measured table nobody had had a use for: 1, 12
and 48 Tensix tiles each reading 1 MiB **from one DRAM channel simultaneously**
measure 22.2, 22.3 and 22.3 GB/s. The aggregate does not grow with the number
of readers. That *is* endpoint occupancy, stated as a vendor measurement, and
it is the shape this instalment installs — 48 readers of one channel share it
rather than each getting a link's worth. It stays a corroboration: what it
confirms is where an `isa_doc` figure belongs, not what it is.

### One tile is two channels, and modelling it as one would over-charge

The premise check turned up a second thing, and it is the reason this instalment
touches `ArchProfile`. A tt-sim `DRAMTile` is not a GDDR6 channel. `wh_dram`:
*"There are 18 DRAM tiles per Wormhole ASIC, collectively exposing 12x 1 GiB
channels of GDDR6 ... DRAM tiles occur in groups of three, with two channels of
GDDR6 present in each group."* tt-sim models a group as one tile covering the
group's whole 2 GiB, and the group's NoC address map names the halves: *"GDDR6
Channel 0 data"* from `0x0_0000_0000`, *"GDDR6 Channel 1 data"* from
`0x0_4000_0000`.

Two independent controllers. One queue for the pair would serialise traffic
that the hardware runs concurrently — an **over-charge**, the direction this
project's cost policy forbids, and one that a workload using tt-metal's second
1 GiB DRAM view would have hit. So `DramChannels` holds one watermark per
physical channel and picks by address, and `ArchProfile.dram_gddr_channel_size`
carries the split (`0x4000_0000` on Wormhole, `None` on Blackhole, which has no
DRAM tile page in the ISA docs at all).

It costs nothing today and that is worth stating plainly: every in-tree guard's
DRAM traffic lands in the low 1 GiB, so the split-channel and single-channel
models produce byte-identical claim, wait and cycle counts on every Wormhole
guard (and Blackhole queues nothing at all). It is in because the shape is
right, not because it moved a number.

### What was built, and where

- `DramChannels` (`tt_sim/device/tiles.py`) — one free-cycle watermark per
  physical GDDR6 channel, plus `claims` / `waits` / `cycles_waited`.
  Structurally the same object as `NocLinkRegistry` and `NUI._tx_free_cycle`,
  and owned by the **tile** rather than an NIU for `NocLinkRegistry`'s reason
  one level down: a channel is one piece of hardware behind two NoC interfaces,
  so a per-NIU queue would let a NoC 0 and a NoC 1 request cross the same bus
  at once.
- `DRAMEndpointNUI._channel_wait` — claims the channel at the request's
  *arrival* (`now + flight`), and adds the wait to the delay it was already
  computing. The claim start is the arrival rather than the moment the bytes
  really cross the bus, which is `service_cycles` later; that offset is the same
  constant for every request, so it can move a phase and not a spacing.
- `ArchProfile.dram_gddr_channel_size` / `dram_gddr_channels_per_tile`.
- `DramCostModel.occupancy_modelled` now answers **True on Wormhole and False
  on Blackhole**, because it is a per-arch consequence of whether the rate is
  published rather than a global switch; `device_occupancy_modelled` is the new
  flag for the half that is still a gap.

**Blackhole gets nothing, and for a reason already on the roadmap.** Its
`dram.bandwidth` is `provenance: unknown` (BlackholeA0 has no `DRAMTile`
directory), so `channel_bytes_per_cycle` is `None`, so no claim is ever made:
**every Blackhole guard is cycle-identical**, `six` included. This is the
second time that one missing figure has been the whole of a Blackhole gap — it
already costs ~24 % on `six` through `channel_serialisation` — and it makes the
roadmap's *"Blackhole `dram.bandwidth` stays unknown"* bullet strictly more
valuable than it was: measuring it would now buy the size term **and** the
occupancy term at once.

### No new consumer, and no new rank

`tt_sim/device/tiles.py` was already on `EXPECTED_CONSUMERS`; the new tests went
into `tt_sim/device/dram_cost_model_test.py`, also already on it. `PROVENANCE_
RANK` is untouched, `estimated` still has zero entries, and the only YAML edits
are notes that asserted the opposite of what the code now does.

### Which guards moved, and why each

Wormhole value guards at **one-cycle poll resolution**, cost model on, against
the same tree with `DramChannels.claim` returning 0 — a like-for-like A/B in
one checkout rather than a comparison across two.

| Wormhole guard | before | after | Δ | DRAM claims | waits | cycles waited |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matmulblock` | 15,087 | **15,177** | **+90** (+0.60 %) | 14 | 1 | 321 |
| `tilize` | 9,328 | **9,398** | **+70** (+0.75 %) | 7 | 2 | 185 |
| `matmulidx` | 9,864 | **9,904** | **+40** (+0.41 %) | 7 | 1 | 150 |
| `untilize` | 58,986 | **59,016** | **+30** (+0.05 %) | 5 | 1 | 66 |
| `reduce` | 9,684 | **9,704** | **+20** (+0.21 %) | 7 | 1 | 66 |
| `reduceneg` | 9,684 | **9,704** | **+20** (+0.21 %) | 7 | 1 | 66 |
| `pipestall` | 32,055 | 32,055 | 0 | 24 | 0 | 0 |
| `sfpumath` | 20,593 | 20,593 | 0 | 4 | 0 | 0 |
| `softplus` | 19,955 | 19,955 | 0 | 2 | 0 | 0 |
| `transpose` | 8,820 | 8,820 | 0 | 4 | 0 | 0 |
| `offline` | 8,178 | 8,178 | 0 | 3 | 0 | 0 |
| `noc_tile_transfer` | 7,711 | 7,711 | 0 | 2 | 0 | 0 |
| `examples` (†) | 80,501 | 80,501 | 0 | 233 | 13 | 1,219 |

(†) `examples` replays several programs in one process, so its figure is the
longest constituent run rather than a total — a change confined to a shorter
program would not show in it. Its 13 waits are counted across all of them.

Blackhole, all 28 guards the same harness can drive: **every one
cycle-identical, zero claims**, for the structural reason above. (`blackhole/two`
and `blackhole/offline` are the poll-budget guards, which this harness cannot
run at one-cycle resolution by construction; the gate covers them, and their
multipliers are unmoved.)

Every guard that moved has exactly the shape the mechanism predicts: a handful
of DRAM requests, one or two of which arrive while a previous one is still
streaming. `cycles_waited` is the charge; the delta is how much of it reached
the total, and the gap between them is the same critical-path story every
instalment in this file tells. **The nine guards with `waits = 0` did not move
at all**, which is the property that makes the term safe: it charges contention
and nothing else.

`wormhole/examples` is the interesting non-mover: 233 claims and 13 waits worth
1,219 cycles, and a total unmoved. It is the multi-program guard, and its waits
land inside `noc_async_read_barrier` polls that were already waiting on
something slower.

### Rung 2: byte-identical, and that is the result

`python3 -m tt_sim.perf.noc_dataset_sweep` produces **byte-identical output**
before and after — all 148 retained points, every residual, every percentile,
every fitted slope and every sustained-rate line. Not one statistic improved
and not one worsened.

That is not a disappointment, it is the control. Every DRAM row in the retained
set is `num_transactions = 1` per barrier, and an isolated request never finds
the channel busy. A change that moved a rung-2 residual here would have meant
the queue was charging something a *lone* transfer does, which is precisely the
double-billing this shape is constructed to avoid.

The corollary is that rung 2 **cannot validate this term**, and nothing in the
tree can. The dataset's multi-transaction rows are all L1, and its DRAM rows are
all N=1; the shipped `noc_latencies.yaml` has no "N flows into one DRAM channel"
configuration at all. What would validate it is the sustained-rate experiment
`wh_dram`'s own table describes — N Tensix tiles reading 1 MiB from one channel,
N swept — which is a card session, on the axis `perfbench/nocbench` already
knows how to sweep.

### The gate

**All 44 guards pass**, and no poll-budget multiplier moved — `dramtop` 1x,
`two` 2x, `offline` 4x, unchanged. That is the point of a term charging
contention and nothing else: it can move a total, and it may not move a result.
All 30 Blackhole replay guards also pass standalone.

The gate nevertheless reports `RESULT: FAIL`, exit 1, and it does so **at this
instalment's base commit too**, on two stage-1 unit tests that have nothing to
do with the cost model: `riscv_bench_sweep_test::test_the_tracked_datasets_
carry_their_own_provenance` and `tensix_bench_sweep_test::test_the_tracked_
dataset_carries_its_own_provenance`. The eleven 2026-08-09 campaign CSVs added
under `tt_sim/perf/datasets/` carry `device=blackhole-silicon` but none of the
`firmware_bundle=` / `kmd=` / `ONE RUN, ON ONE CARD` / `valid:` lines those
tests require. **Fixing it means writing down a card's firmware and KMD
versions, which is provenance and cannot be invented**, so it is left for
whoever ran the session. The gate output is otherwise byte-identical before and
after this change, modulo timings and the nine new tests (`2 failed, 1123
passed` -> `2 failed, 1132 passed`).

### What changed in the repository

- `tt_sim/device/tiles.py` — `DramChannels`; `DRAMEndpointNUI._channel_wait`
  and its `channels` argument; `DRAMTile` builds the set from the profile.
- `tt_sim/arch/profile.py`, `tt_sim/arch/wormhole.py` —
  `dram_gddr_channel_size` and `dram_gddr_channels_per_tile`.
- `tt_sim/perf/model.py` — `occupancy_modelled` becomes a per-arch consequence,
  `device_occupancy_modelled` is the remaining named gap, and two prose blocks
  that asserted the opposite were rewritten rather than deleted.
- `tt_sim/perf/unit_costs.yaml` — notes only, on `dram.channel_serialisation`
  and both arches' `dram.access_latency`. **No number changed, no entry was
  added, no provenance moved.**
- `tt_sim/device/dram_cost_model_test.py` — nine new tests: the premise as a
  regression, the inertness control, the per-channel split, and the two gap
  flags.
- `docs/plans/cost-model.md` — this section.

## The endpoint term, corroborated on silicon, and three reports re-read

The section above shipped `DramChannels` and then said, plainly, what it could
not do: "rung 2 **cannot validate this term**, and nothing in the tree can ...
What would validate it is the sustained-rate experiment `wh_dram`'s own table
describes — N Tensix tiles reading 1 MiB from one channel, N swept — which is a
card session."

That card session ran the same day, at 22:34, on the harvested Blackhole part.
This section records what it returned, together with three other probes from the
22:24 session whose reports were collected but never read — and two of those
three turn out to have been misread by the tools rather than by the card.

**Nothing here is charged.** Every number below is `corroboration`, on one part,
on one day. No `unit_costs.yaml` entry moved and none became chargeable.

### 47.1 B/cycle, twice, from two methods that share nothing

The cross-check is the part worth leading with, because it is the rarest kind of
evidence this file has collected.

`dramratebench` puts N Tensix tiles on one DRAM channel and sweeps N. Its
one-channel arm saturates at **47.14 B/cycle** (mean of three repeats at 120
readers; the run's own summary line reads 47.171 for its last repeat).

Rung 2 derives Blackhole's DRAM read rate from tt-metal's 8,140-point measured
NoC dataset, by fitting the large-transfer slope of the diff-axis series. Run
today, `python3 -m tt_sim.perf.noc_dataset_sweep --arch blackhole` prints:

```
  implied sustained bandwidth, from the large-transfer slope
    DRAM read diff-axis    measured  47.08 B/cycle   model  64.00 B/cycle
    DRAM write diff-axis   measured  59.36 B/cycle   model  64.00 B/cycle
```

**47.08 against 47.14 — 0.13 %.** The two have no input in common: one is a
vendor dataset of single-transaction latencies at many transfer sizes, fitted
for a slope; the other is a bandwidth sweep of many concurrent readers at one
transfer size, read off its plateau. They do not share a card, a kernel, a
harness, or a year. Agreement at this level is not a coincidence and it is not
circularity, because neither derivation can see the other.

It also retires a worry the previous section left open. That 47.1 was quoted as
"74 % of a 64 B/cycle link" and read as *suspicious* — Wormhole's two directions
agree to 0.6 % while Blackhole's differ by 26 %, "which is not" what one channel
rate looks like. The card says the read figure is simply correct. Whatever
explains Blackhole's read/write asymmetry, it is not an artefact of the rung-2
fit.

### One channel, 120 readers, and exactly 1/N

The shape is as clean as the rate.

| readers | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one channel, aggregate | 46.30 | 46.06 | 45.91 | 45.69 | 46.09 | 46.40 | 47.10 | 47.08 | 47.09 | 47.14 |
| one channel, per reader | 46.30 | 23.03 | 11.48 | 5.71 | 3.84 | 2.90 | 1.96 | 1.47 | 0.98 | 0.39 |
| fanned across 8 banks | 46.35 | 92.86 | 148.1 | 188.7 | 172.4 | 212.2 | 205.3 | 233.0 | 192.2 | 233.5 |

Per-reader throughput on one channel is `46.30 / N` to within 1.8 % at every
point in a 120× sweep. The aggregate moves by **×1.018** while the same readers,
the same issue loop and the same 4096 B transactions fanned across eight banks
move by **×5.04**.

The strongest way to say it: **one reader already saturates the channel.** A
single tile gets 46.3 B/cycle; a hundred and twenty tiles together get 47.1.
Adding 119 readers bought 1.8 %. That is what perfect serialisation at an
endpoint looks like, and it is the shape `DramChannels` asserts.

What it does *not* give is a depth. The term models a queue; this measures the
queue's output rate when it is permanently full, which is the one regime in
which depth does not matter. The card confirms the shape and says nothing about
the parameter.

### Two caveats that must travel with the number

**It is `corroboration`, never provenance.** BlackholeA0 publishes no DRAM tile
page at all, so `dram.bandwidth` stays `provenance: unknown` and unchargeable
there, exactly as the previous section left it. A measurement on one card is not
a published figure, and the guard in `DramCostModel` that refuses Wormhole's
deep-merged 24 must keep refusing. **This section adds evidence, not a source.**

**The `samecore` arm did not run**, and its absence is load-bearing.
`dramratebench` is built with three arms; `samecore` was to put N readers on two
banks that share one physical DRAM core — same inbound NoC link, different GDDR6
channel — and it is the arm that separates a *channel* limit from the limit of
the one router link every `onechan` flow converges on. On this part no physical
DRAM core fronts two banks (all eight are one bank each), so the benchmark
skipped it and said so.

So what is established is that **the endpoint costs the difference, not the
fabric** — the fan-out control travelled the same fabric and scaled ×5. What is
*not* established is which endpoint: the GDDR6 channel and its inbound router
link remain confounded. `DramChannels` charges the channel. That is still the
better-motivated of the two, and it is now unrefuted rather than corroborated
down to the component.

### `tensix-rdcfg`: a probe-design negative, recorded as one

`RDCFG`'s latency is documented `>= 2`, and the 22:24 session tried to measure it
as `(RDCFG + STALLWAIT) - (SETDMAREG + the same STALLWAIT)`.

The control moved: `SETDMAREG` alone reads 0.998 cycles/instruction and
`RDCFG + STALLWAIT` reads 2.968, so the stall instruction plainly costs
something. The paired difference is **+0.000 cycles per pair** (3.000 against
2.999), under the 0.5 cycle/pair floor a claim about a `>= 2` latency has to
clear.

This is the null, not a small value. Both slots pay for the stall *instruction*
while neither stall ever observes the config unit — `STALL_THREAD` on
`TRISC_CFG` is the condition that would have made the second operand wait, and
the construction never engaged it. The doc's `>= 2` stays as unchecked as it was,
and the next attempt needs a different construction rather than more repeats.

### The fetch ramp, resolved: graded, not a step

The previous section left phase G's instruction-fetch ramp bracketed: the onset
was somewhere in `(4096, 5120]` and everything between was interpolation. Four
`--gset` runs at 22:24 filled it in. With gset 0's 5120 B from the primary
dataset there are now **five measured intermediate footprints**, none inferred.

The `g_1024` baseline rows are bit-identical across all four gset files bar one
cell (gset 3 at n = 128, off by one cycle in 131,688), so the files compare
directly without renormalising — checked before comparing, because it is the
assumption the whole table rests on.

| footprint (B) | 4096 | 4608 | 5120 | 5632 | 6144 | 7168 |
| --- | --- | --- | --- | --- | --- | --- |
| cycles/instruction | 1.0022 | 1.0938 | 1.1598 | 1.2141 | 1.2565 | 1.2556 |
| normalised to 4096 | 1.0000 | 1.0915 | 1.1573 | 1.2115 | 1.2538 | 1.2529 |
| step from previous | — | +0.0915 | +0.0658 | +0.0542 | +0.0423 | −0.0009 |

**It is a graded ramp, and emphatically not one step.** Four resolved,
monotonically decreasing increments carry the cost from 1.000 to 1.254 across
4096 → 6144 B, and then it stops: 7168 B is 0.0009 *below* 6144 B, and the
session's own phase-F 8192 B point sits at 1.2522. Phase F also brackets the
ramp from below: 256, 512, 1024, 2048 and 4096 B all read 1.000–1.027, so
nothing happens until 4 KiB and everything happens in the 2 KiB above it. Any
single-step reading is excluded by the
data — a step would have put two adjacent footprints on 1.000 and 1.25 with
nothing between, and there are four points between.

The rising region has a specific shape, and it is the one a fixed-capacity
buffer predicts. If a buffer of capacity `C` covers the loop and everything
outside it costs `k` extra, the cost is `1 + k(1 − C/F)` — linear in `1/F`.
Fitted over the four rising points:

```
  cost = 1.7430 − 3000.2 / footprint      r2 = 0.99983
  implied k = 0.743,  implied capacity C = 4038 B
```

r2 0.99983 over four points, and an implied capacity of **4038 B against a
natural 4096**. That is a 1.4 % miss on a parameter the fit was not told about,
and it is the strongest single result in the phase-G series.

But the same fit **fails above 6144 B**, and the failure is the interesting
half. Extrapolated to 7168 B it predicts 1.3245; the card measured 1.2529. The
1/F curve should keep climbing towards 1.743 and instead the cost stops dead at
1.253 and stays there to 8192 B (1.2522).

So two mechanisms, not one. A **~4 KiB covered window** whose shrinking coverage
produces the graded 1/F rise, and a **ceiling at ~1.253 cycles/instruction** that
takes over once the window covers little enough. A pure capacity-miss model
cannot produce the plateau; a pure bandwidth model cannot produce the graded
rise. The ceiling is what a fetch path that can no longer be run ahead of looks
like — every instruction costs the same delivery cost, however much bigger the
loop gets.

One arithmetic observation, offered as a hypothesis and not a conclusion:
1.253 is within 0.3 % of **1.25 = four instructions per five cycles**, which is
what a 16 B fetch line delivered with one bubble per line would give. The
measurement cannot distinguish that from 1.253 exactly, and nothing in this file
should treat it as though it could.

`tt_sim` charges no instruction-fetch term on either architecture, so none of
this contradicts a charged cost. What it does is turn a "there is a step
somewhere above 4 KiB" note into a two-parameter shape with a capacity, a
ceiling, and five measured points between them.

### The congestion pair, and the epoch that was never absent

Both 22:24 congestion CSVs were collected and marked DEFERRED, because the
report generator needs `tt_sim/` and the card box does not have it. Run here they
say two things, one confirming and one correcting.

**The shape reproduces across sessions.** Four independent nocbench invocations
now agree:

| shared-link slope | 64 B | 512 B | 2048 B | 8192 B | 16384 B |
| --- | --- | --- | --- | --- | --- |
| 13:41 `noc` | −0.01 | +0.01 | +2.49 | +10.87 | +22.49 |
| 13:41 `noc-epoch` | −0.01 | +0.01 | +2.63 | +10.97 | +22.47 |
| 22:24 `noc` | −0.01 | +0.00 | +2.48 | +10.94 | +22.39 |
| 22:24 `noc-epoch` | −0.01 | +0.00 | +2.45 | +10.92 | +22.50 |

FLAT at 64 and 512 B, SATURATING at 2048 B and above, across two sessions nine
hours apart. **The caveat that has to travel with every one of those numbers is
unchanged: r2 is 0.03–0.47.** These are *shape* verdicts read off poor fits. The
shapes reproduce; the coefficients are not well determined and must not be
quoted as though they were.

**The correction is larger.** Both `noc-epoch` runs were graded `INVALID` — "5
multi-flow runs whose timed regions barely overlapped; those flows did not
contend" — and the 13:41 one was banked with a header saying the (11,2) clock
epoch "shows NO epoch here ... The epoch is per-RUN, not per-core."

All of that was a bug in the reader.

An epoch offset is only believed if it is *impossible as a delay*, which the
sweep tested by requiring it to exceed the session length. The session length
was computed as `max(t0) − min(t0)` over stamps that are the low 32 bits of a
free-running counter. Both `noc-epoch` runs happen to cross `2**32` part-way
through, so that span read **4,289,112,683** and **4,288,408,164** against true
elapsed times of 383,858,750 and 386,257,520. No real offset can clear a bar of
nearly `2**32`, so both files' offsets were discarded — and, not being
subtracted, the five runs core (11,2) appears in then reported ~0.00 overlap and
took the whole file down with them.

With the span computed wrap-aware, both files find their offset, all five of
those runs overlap **1.00**, and both read `CONGESTION MEASURED`. Nothing about
the card changed. `_elapsed_span` now walks the stamps in run order accumulating
signed-wrapped steps, and the 22:24 file is banked as the regression fixture.

That leaves the (11,2) question, and it can now be answered with the runs
pooled. The epoch is present in **5 of 5** nocbench invocations, never twice with
the same value:

| run | (11,2) offset (cycles) | spread |
| --- | --- | --- |
| 2026-08-05 | +1,143,914,610 | 1 over 2 flows |
| 08-09 13:41 `noc` | −495,379,666 | 4 over 5 |
| 08-09 13:41 `noc-epoch` | −782,897,612 | 4 over 5 |
| 08-09 22:24 `noc` | −1,460,817,587 | 4 over 5 |
| 08-09 22:24 `noc-epoch` | −1,760,493,889 | 6 over 5 |

So the detector's "reproduced over 5 runs" does count repeats *inside* one
invocation, exactly as suspected — but the conclusion drawn from that, that the
epoch is flaky or absent, was wrong. It is rock solid within a launch and
different in every launch.

It is **not a frequency difference**, and that can be shown from within a single
file without any modular ambiguity. Across the 22:24 `noc-epoch` run the offset
moved by 1 cycle while 212,681,996 cycles elapsed on the reference tile — a rate
difference below `5e-9`. Over the few minutes between that probe and the one
before it, that rate could produce at most a few thousand cycles of drift; the
observed change is 299,676,302. Refuted by five orders of magnitude. Whatever
re-bases that tile's counter happens *between* program launches, not during one.

The practical consequence is the one that closes the bullet: the offset must be
re-derived per file and never carried between files, which is what the reader
already does — and **same-core durations, which every coefficient in this file
is fitted to, never involve two tiles' stamps at all.** No cost model number is
exposed to this. It is a property of the harness's cross-core timestamps and of
nothing else.

### Phase Q at "n = 32": a mislabel, not a threshold

The 22:24 session ran `riscvbench` twice at identical parameters. The primary
was graded MEANINGFUL — phase Q's monotonicity complaints all at n ≤ 16, which
`riscvbench/README.md` documents as expected small-burst scatter. The repeat was
graded **SUSPECT**, "phase Q failed its gate at n = 32, above the small-burst
scatter the README accounts for."

Same binary, same flags, different verdict. The whole difference is one line:

```
NOT MONOTONE: q/t1/q_loop_adddmareg thread 1: n=16 -> 70 cycles, n=32 -> 48 cycles
```

Read the series against the primary run's and the diagnosis is immediate:

| n | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primary | 46 | 48 | 107 | 299 | 683 | 1451 | 2987 |
| repeat | **70** | 48 | 107 | 299 | 683 | 1451 | 2987 |

**Every point from n = 32 up is bit-identical between the two runs.** The n = 16
point moved by 24 cycles and nothing else moved at all. The violation is a
scatter at n = 16 — inside the documented band — and it was labelled "n = 32"
because `card_session_verdicts.sh` extracted the *second* n from the complaint
line. A pair is evidence about its smaller burst, since either point may be the
one that moved and only the smaller can be inside the scatter band.

It is scatter and not something systematic, on the phase-R precedent: of the
complaint slots the two runs raise, only 4 are common; the rest fall on
different `(variant, thread, probe, n)` slots, which is the same signature phase
R's R2 failures showed when they landed on different slots in two runs.

The honest tolerance question has a measured answer rather than a chosen one,
which is the precedent phase S set. Differencing the 390 phase-Q slots the two
runs have in common gives the run-to-run scatter, and comparing it to the
monotone step the gate is testing gives this:

| pair | median step (cycles) | max run-to-run scatter | testable? |
| --- | --- | --- | --- |
| 1 → 2 | 5.0 | 7 | no |
| 2 → 4 | 9.5 | 7 | marginal |
| 4 → 8 | 12.0 | 11 | marginal |
| 8 → 16 | 9.0 | 21 | **no** |
| 16 → 32 | 20.0 | 45 | **no** |
| 32 → 64 | 54.5 | 39 | yes |
| 64 → 128 | 192.0 | 42 | yes |
| 128 → 256 | 768.0 | 30 | yes |
| 256 → 512 | 1536.0 | 21 | yes |

The scatter is roughly **constant in absolute cycles** (21–45) while the step
the gate checks grows with the burst. So the gate is testing noise below the
crossover and hardware above it, and the crossover sits between the 16 → 32 pair
and the 32 → 64 pair. **The first pair phase Q can honestly be gated on is
32 → 64.**

That makes the README's "n ≤ 16" threshold defensible but one notch tight, and
it makes the labelling — not the threshold — the actual defect here. Grading a
pair by its smaller burst re-grades this run MEANINGFUL without moving the
threshold at all, and that is the change made. Anyone who wants to widen the
threshold to n ≤ 32 now has the measurement to justify it; it was not needed to
fix this run and so was not done.

### Nothing contradicts a charged cost

Worth stating explicitly, because it is the thing this file has to check every
time. Of everything above:

- The DRAM rate **agrees** with rung 2's derivation to 0.13 % and contradicts
  nothing; `dram.bandwidth` stays unchargeable on Blackhole.
- The congestion slopes are unchanged from the readings already recorded, and
  no congestion coefficient is charged anywhere.
- The fetch ramp measures a term `tt_sim` does not model on either
  architecture, so there is no charged value to contradict.
- `RDCFG` returned a null, which leaves its `>= 2` exactly as unchecked as it
  was.
- The (11,2) epoch is a cross-core timestamp artefact and touches no fitted
  coefficient.

Two *tools* were wrong and are fixed; no measured or charged number was.

### What changed in the repository

- `tt_sim/perf/noc_congestion_sweep.py` — `_elapsed_span`, and the session-span
  guard in `clock_skew_report` now uses it. **No coefficient, threshold or
  verdict rule changed**; a modular quantity is computed modularly.
- `tt_sim/perf/noc_congestion_sweep_test.py` — two tests: the wrap in the
  abstract, and the banked 22:24 file as a real fixture.
- `perfbench/card_session_verdicts.sh` — `_rv_q_worst_n` grades a phase-Q
  complaint by the smaller burst in its pair. The `n <= 16` threshold is
  untouched.
- `perfbench/card_session_verdicts_test.sh` — the 22:24 `rv-cross` line as a
  regression.
- `tt_sim/perf/datasets/` — four new files: the 22:24 congestion pair
  (`nocbench-blackhole-2026-08-09-2224.csv` and `-2224-epoch.csv`, the second
  being the wrap fixture) and the two new footprints
  (`riscvbench-blackhole-2026-08-09-gset3.csv`, `-gset4.csv`, 4608 and 5632 B).
- `tt_sim/perf/datasets/nocbench-blackhole-2026-08-09-repeat.csv` — header
  corrected. It asserted the file was INVALID and that (11,2) had no epoch;
  both were artefacts of the wrap bug and the file reads clean.
- `docs/plans/cost-model.md` — this section.

## The whole-thread issue interlock, and the Src hand-over it was blocked on

Landed 2026-08-11 — ROADMAP item 3's "documented per-op issue cadence". Two
units publish an interlock that is about the **thread**, not about the unit,
and neither could be expressed by the machinery that existed: `is_occupied`
refuses the unit an instruction was *offered to*, and the instruction a
held thread wants to issue next is usually offered somewhere else entirely.
**Both are now charged.** The second could not be until a *functional* bug was
fixed first — tt-sim handed a Src bank to the Matrix Unit in the tick the
`UNPACR` retired, where the documents put the hand-over at the end of the
transfer — and fixing that turned out to expose a second latent race in the
unpacker's counter update. Both fixes are below; neither changes a number in
either YAML, and neither changes anything at all with `TT_SIM_COST_MODEL`
unset.

> `TensixCoprocessor/ScalarUnit.md`: "No instructions of any kind from the
> issuing thread can pass through its Wait Gate. In other words, once a thread
> has started executing a Scalar Unit instruction, it cannot start executing
> its next instruction until the Scalar Unit instruction completes, **regardless
> of which unit that next instruction executes in**."

`TensixBackend.thread_issue_block` is one deadline per Tensix thread; the Wait
Gate refuses to pass anything from that thread until the cycle it names, and
publishes a `thread_issue_block` `StallEvent` naming the unit that imposed it
— `THCON` here, `UNPACK` for the interlock below — while it does.
**No cost-table entry was added and no number in either YAML changed.** The
deadline *is* the occupancy the table already charges, armed from the same
acceptance-anchored cycle, so it inherits the low-end-of-the-bound floor the
occupancy was charged at: `ADDDMAREG` is "3 or 4" charged at 3, so a thread
that issues one may not start anything anywhere for 3 cycles, and `ATCAS`'s
">= 15" holds it for 15.

### The unpacker's identical sentence

> `TensixCoprocessor/UNPACR_Regular.md`: "An `UNPACR` instruction spends at
> least two cycles calculating the initial input address ... For the duration
> of these cycles, **the issuing thread cannot start its next instruction**,
> nor can any other thread start an `UNPACR` instruction."

Same shape, same mechanism, and the deadline is the 2 the table already charges
as the address phase — armed from `backends/unpacker.py` at the acceptance
cycle, exactly where the occupancy is anchored. It is scoped to the address
phase alone ("for the duration of *these* cycles"): what follows "proceeds in
a pipelined fashion", and holds the unit but not the thread.

The **cross-unpacker half** of the same sentence ("nor can any other thread
start an `UNPACR`") is unchanged and unmodelled for the older reason: it needs
an arbitration rule between two unpackers that no source gives, exactly as the
joint 80 B/cycle ceiling does. Two halves of one sentence charged, one not.

### What blocked it: the Src hand-over happened a data phase too early

Wiring the interlock first turned three Wormhole value guards red —
**`softplus`, `tilize` and `untilize` all raising `SrcDvalidError`**
("`SETRWC` from thread 1 cleared dvalid on SrcB bank 0, but that bank's
`AllowedClient` is `Unpackers`"). Everything else — all 30 Blackhole guards,
the Wormhole matmuls, `six` — passed. The traced cause, at one-cycle
resolution, was not the charge:

```
  passing        failing (with the interlock)
  c11012 t0 UNPACR          c11012 t0 UNPACR       <- acquires SrcA
  c11013 t0 UNPACR_NOP      c11013     (held: address phase)
  c11014 t0 UNPACR_NOP      c11014 t0 UNPACR_NOP
  c11014    SrcB -> FPU     c11015 t0 UNPACR_NOP   <- SrcB acquire, executes c11016
  c11015 t1 SETRWC          c11015 t1 SETRWC       <- release, executes c11016, first
```

The LLK datacopy loop runs `UNPACR; UNPACR_NOP; UNPACR_NOP` on the unpack
thread against `MOVA2D; MOVA2D; SETRWC` on the math thread, with no semaphore
between them: the only ordering is that `MOVA2D` waits at the gate for SrcA,
after which the two threads race three instructions against two. That left
**exactly one cycle** between the `UNPACR_NOP` that hands SrcB to the FPU and
the `SETRWC` that hands it back — and `SETRWC` does *not* wait at the gate
(its page has no Wait Gate paragraph; it assigns `AllowedClient` outright).
A two-cycle interlock spends that one cycle, and the backend tick order
(`matrix_unit` first, `unpacker_units` last) then resolves the tie in the
release's favour.

**A one-cycle margin was never what the hardware has, and the documents say
where the margin comes from.** Three places, all agreeing:

> `UNPACR_Regular.md`, functional model: the whole "Main unpack loop" runs, and
> only *after* it — in the block headed "Update counters in preparation for
> next instruction" — does `(WhichUnpacker ? SrcB : SrcA)[CurrentUnpacker
> .SrcBank].AllowedClient = SrcClient::MatrixUnit`.

> `tensix_instructions.yaml`, the `SetDatValid` field of `UNPACR`: "Unpacker
> will set data valid bit for the registers it is unpacking into **once data
> has been written**."

> `UNPACR_Regular.md`, Performance: "Once these cycles are complete, execution
> proceeds in a pipelined fashion, with the primary bottleneck being the
> fetching of bytes from L1."

tt-sim moved every datum in the retire tick — which is unobservable, since
nothing may read the bank until it changes hands — but flipped `AllowedClient`
there too, letting the Matrix Unit start consuming a bank up to a whole data
phase before the transfer that fills it has finished. `UnPackerUnit` now owes
the hand-over (`_deferred_dvalid`) and settles it in `_hand_over_src_bank` at
`address phase + data phase` after acceptance — **the same deadline the unit's
own occupancy already sets**, so no new number and nothing new to source. The
bank pointer and row base go with it, because the functional model moves all
three together and because `STALLWAIT`'s C8/C9 read them to decide which bank
they are asking about.

With `TT_SIM_COST_MODEL` unset both phases are zero, the deadline is the
acceptance cycle, and the hand-over happens in the retire tick exactly as
before — which is why the flag-off numbers below do not move.

*Measured.* The minimum gap, over a whole run, between a SrcB bank being
acquired and the `SETRWC` that releases it — i.e. how much slack the loop has
before an acquire and a release collide in one cycle and the tick order
decides:

| guard | before | after |
| --- | --- | --- |
| `wormhole/softplus` | 1 | **9** |
| `wormhole/tilize` | 1 | **5** |
| `wormhole/untilize` | 1 | **2** |

It went from "the tick order decides" to "the data phase decides". And the
perturbation that used to break it now does not: injecting a one-cycle-in-five
stall into the math thread's Wait Gate takes `wormhole/reduce` from a wrong
value on the pristine tree to a pass on this one.

### And what that exposed: a stalled `UNPACR` undoing a later `SETADCZW`

Fixing the hand-over pushed the Matrix Unit later, which lengthened one
unpacker stall, which made `wormhole/reduce` and `wormhole/reduceneg` compute
a wrong value — `dst[2048] = 0xffff`, the GMPOOL minus-infinity sentinel. That
was **not** caused by the hand-over; it is a pre-existing race the timing
change happened to reach. The same one-cycle-in-five math-thread jitter
reproduces it on the untouched tree.

The mechanism, again at one-cycle resolution:

```
 pristine                            with the hand-over fixed
 c6272 LATCH  unp0 in=0x1a240 Z=3    c6272 LATCH  unp0 in=0x1a240 Z=3
 c6279 UPDATE_ADC       -> Z=4       c6280 SETADCXX t0
 c6280 SETADCXX t0                   c6285 SETADCZW t0      -> Z=0   (next op)
 c6285 SETADCZW t0      -> Z=0       c6286 UPDATE_ADC       -> Z=1   (too late)
 c6305 LATCH  unp0 in=0x19c40 Z=0    c6314 LATCH  unp0 in=0x19e40 Z=1
```

`_llk_unpack_reduce_` opens each call with a `SETADCZW` that zeroes the Z
counter. tt-sim applied an `UNPACR`'s `Ch0.Z += Ch0ZInc` when the *transfer*
completed, so an `UNPACR` still waiting on a Src bank when that reset landed
put the counter back — and the next reduction read every face one tile-row up
L1. `handle_regular` already latches the *configuration* at decode for the
mirror-image reason on the read side ("the issuing thread carries on while the
unpack is in flight ... reading the configuration afresh when a stalled unpack
finally runs would pick up the next matmul's context and base address"); the
counter update is the write side of the identical hazard, and it now happens
in the address phase with the input address generator that consumes it. The
doc's own heading is the argument — "in preparation for **next instruction**"
— and the next instruction cannot start before this unpacker's address phase
is over, so nothing inside the unit can tell the difference. The Src
hand-over stays where it is: it is the one member of that block the sources
time against the data, not against the next instruction.

### Measured, at one-cycle resolution

Ten Wormhole guards, replayed socket-free and pumped to `RUN_MSG_DONE` one
cycle at a time, each run on this tree and on the tree without the three
changes:

| guard | model off, before | model off, after | model on, before | model on, after |
| --- | --- | --- | --- | --- |
| `reduce` | 7,109 | **7,109** | 9,704 | 9,714 (+10) |
| `reduceneg` | 7,109 | **7,109** | 9,704 | 9,714 (+10) |
| `transpose` | 6,724 | **6,724** | 8,820 | 8,850 (+30) |
| `softplus` | 17,845 | **17,845** | 19,955 | 19,965 (+10) |
| `matmulblock` | 9,741 | **9,741** | 15,177 | 15,207 (+30) |
| `matmulidx` | 7,117 | **7,117** | 9,914 | 9,944 (+30) |
| `sfpumath` | 19,306 | **19,306** | 20,593 | 20,638 (+45) |
| `tilize` | 7,008 | **7,008** | 9,398 | 9,408 (+10) |
| `untilize` | 57,694 | **57,694** | 59,086 | 59,186 (+100) |
| `noc_tile_transfer` | 6,384 | **6,384** | 7,711 | 7,711 (+0) |

Eight Blackhole guards, same method:

| guard | model off, before | model off, after | model on, before | model on, after |
| --- | --- | --- | --- | --- |
| `reduce` | 8,876 | **8,876** | 11,608 | 11,608 (+0) |
| `transpose` | 8,670 | **8,670** | 10,558 | 10,567 (+9) |
| `tilize` | 69,961 | **69,961** | 72,146 | 72,146 (+0) |
| `untilize` | 11,306 | **11,306** | 12,437 | 12,509 (+72) |
| `softplus` | 17,359 | **17,359** | 18,454 | 18,454 (+0) |
| `optest` | 10,529 | **10,529** | 13,217 | 13,217 (+0) |
| `twolaunch` | 13,064 | **13,064** | 13,493 | 13,515 (+22) |
| `eight` | 8,677 | **8,677** | 10,231 | 10,231 (+0) |

Model off: byte-identical on all ten. Nothing arms an occupancy with
`TT_SIM_COST_MODEL` unset, so the interlock deadline is never set, the
hand-over deadline is the acceptance cycle, and the only unconditional change
— moving the counter update into the address phase — is unobservable there
because the address phase and the transfer are the same tick unless the unpack
blocks (and a blocked unpack's *own* next instruction cannot start either way).

Model on: **+0 to +100 cycles, every one of them slower**, which is the only
direction these changes can move a total. Two effects add: the address-phase
interlock stops a thread issuing for two cycles after each `UNPACR`, and the
Matrix Unit now waits for the transfer it consumes rather than for the tick
the instruction retired. `wormhole/noc_tile_transfer` moves by nothing at all
— it has no Tensix compute — and `wormhole/untilize`, which unpacks the most,
moves the most. Blackhole moves less because its unpacks run at the default
x4-"2x"/x8 rates, so the data phase it now waits for is a quarter to an eighth
the length: five of the eight totals do not move at all.

That totals barely move is the expected shape and not a disappointment: a
launch's completion is gated on the backend draining, and on these guards the
span is set by the RV load-use interlock and by semaphore waits, not by how
soon a thread may issue its next instruction. What the interlock buys is
*attribution* and a core-visible cadence, which is what ROADMAP item 3 was
for — and, structurally, the ability to stall a thread out of a unit it was
not offered to at all, which nothing else in the model can do.

### What was already there, and what is deliberately still missing

- **The srcA/srcB dvalid gating at the Wait Gate was already implemented**, on
  both sides — `WaitGate.checkIfFPUInstructionShouldStall` for the FPU's
  acquire and `UnPackerUnit.handle_regular`'s `blocked` for the unpacker's
  mid-execution wait, which is the asymmetry the WaitGate page's footnote
  describes. Re-checked against the per-instruction pages: every entry of
  `MATH_ALLOWED_CLIENT_INSTRUCTIONS` matches its document's `while
  (Src?[MatrixUnit.Src?Bank].AllowedClient != MatrixUnit)` block, with one
  exception recorded rather than fixed — `MOVDBGA2D`, whose page says to read
  `MOVA2D`'s model "ignoring the paragraph about the Wait Gate", is listed as
  waiting on SrcA. It is an over-charge, it is unreachable (the opcode is
  decode-only), and the list is pinned by a test whose purpose is to make
  edits to it deliberate.
- **PC-buffer write delays cannot be sourced at the point that matters.** The
  depth is published exactly — `BabyRISCV/PCBufs.md`, identical on both
  arches: "Each FIFO queue can hold up to 16 32-bit values" — but the next
  sentence is what a charge would need and does not give: "attempting to push
  more values than this will cause the writes to sit in **shared buffers within
  the RISCV B memory subsystem**. If those shared buffers become full, RISCV B
  will be stalled until space becomes available." The shared buffers' capacity
  is published nowhere, so the honest bound on when RISCV B stalls is ">= 16
  pending pushes", and for a *queue depth* the low end is the over-charging
  end — a bound of 16 invents back-pressure the hardware does not have, which
  is the one direction the floor policy forbids. (This is the same reasoning
  that put `CORE_PUSH_INFLIGHT_BOUND` safely *above* its measured floor rather
  than at it.) So the write delay stays uncosted and the FIFO stays unbounded.
  Two independent facts make that cheap: no in-tree workload touches a PCBuf
  at all (instrumented across `matmulblock` / `sfpumath` / `one`: zero reads,
  zero writes — tt-metal 0.74 launches TRISCs through mailboxes), and the RV
  side's PCBuf *load* latency is already charged, at the ">= 3" row
  `riscv.load_latency.mailboxes_pcbufs_ttsync_semaphores` gives it.
- **The MOP Expander's one-cycle transition penalty** ("After expanding a
  `MOP` instruction, if the next instruction is _not_ a `MOP` instruction,
  there is a one cycle transition penalty") is documented, exact, and
  unobservable here: tt-sim's expander emits a whole template in one tick and
  the rate downstream is set by the Replay Expander and the Wait Gate, which
  already move one instruction per cycle. Charging it would model a bubble in
  a stage that has no rate to bubble. Recorded, not wired.

### What changed in the repository

- `tt_sim/pe/tensix/backend.py` — `thread_issue_block` / `block_thread_issue`,
  with both sentences quoted at the state they justify.
- `tt_sim/pe/tensix/frontend.py` — the Wait Gate's check, first among the
  gate's refusal branches because the interlock outranks them all.
- `tt_sim/pe/tensix/backends/thcon.py` — the Scalar Unit's arming site.
- `tt_sim/pe/tensix/backends/unpacker.py` — the unpacker's arming site; the
  deferred Src hand-over (`_deferred_dvalid` / `_hand_over_src_bank`); and the
  counter update moved into `read_unpack_state`.
- `tt_sim/trace/events.py`, `docs/trace-schema.md` — the
  `thread_issue_block` stall reason.
- `tt_sim/pe/tensix/tensix_instruction_costs.yaml` — two `note:` fields
  recording what is now consumed. **No number changed.**
- Tests: three in `frontend_backpressure_test.py` (the interlock reaches a
  different unit, holds only the issuing thread, and is absent with the model
  off) and six in `unpacker_cost_model_test.py` — the address phase holds the
  unit *and* the thread, the thread block is not extended by the data phase,
  no thread is held with the model off, the hand-over lands at the end of the
  transfer (and in the retire tick with the model off), a stalled unpack does
  not undo a later ADC reset, and an unpack/math ping-pong whose hand-over
  cycles are unchanged by sliding the threads against each other or by
  reversing the backend unit tick order.

## `STALLWAIT` C1/C2: a transfer that was not reported as in flight

Landed 2026-08-11, immediately after the section above and for the same
window. **No number in either YAML changed, and nothing changes at all with
`TT_SIM_COST_MODEL` unset.**

The hand-over above fixed *when the Src bank changes hands*. It left the other
half of the same window wrong: **whether the thread that issued the `UNPACR`
is told its unpack is still running.**

> `STALLWAIT.md`, condition mask: **C1** "The current thread has an instruction
> in any stage of Unpacker 0's pipeline." **C2** the same sentence for
> "Unpacker 1's".

> `UNPACR_Regular.md`, Performance: "An `UNPACR` instruction spends at least
> two cycles calculating the initial input address ... Once these cycles are
> complete, execution proceeds **in a pipelined fashion**, with the primary
> bottleneck being the fetching of bytes from L1."

The L1 fetch is a stage of that pipeline, so C1/C2 are unmet for the whole of
the address phase *and* the data phase — which together are exactly the
occupancy `UnPackerUnit` already charges itself. `hasInflightInstructionsFromThread`
consulted only the issue queue and the `blocked` flag, both of which are done
with the instruction the moment it retires, so a `STALLWAIT` on C1 cleared
while the transfer was still moving datums. `wormhole/reduce` is the in-tree
program that leans hardest on this: its unpack/math pairing has no semaphore
and `STALLWAIT` is the only ordering it has.

The predicate now also answers "yes" while the unit is occupied *by that
thread* (`_occupied_thread`, written only where a hold is armed). **The scope
is this unit and no other, deliberately**: occupancy is throughput
back-pressure, and for a pipelined unit that is not residency — an instruction
can sit in a stage long after the unit will accept the next one. The unpacker
is the case where the two coincide, because the doc's bottleneck *is* the
transfer and tt-sim charges address+data as one hold. Reading another unit's
`busy_until` as residency would need that unit's own latency and is not done.

### And the same table's C2 asked the wrong unpacker

`check_for_semwait_condition_match` (the Wormhole branch) answered both C1 and
C2 from `unpacker_units[0]`. The Blackhole branch beside it has always had it
right, which is what makes it a transcription slip. It is live: probed under
the model, `wormhole/matmulblock` evaluates C2 106 times and the two unpackers
disagree at that moment 208 times across C1+C2. Fixing it can only make a
thread wait longer, never less.

### `MOVDBGA2D` was gated at the Wait Gate where its page says not to be

> `MOVDBGA2D.md`: "This instruction is identical to `MOVA2D`, except that it
> doesn't _automatically_ wait for `SrcA[MatrixUnit.SrcABank].AllowedClient ==
> MatrixUnit`" — and its functional model: "See `MOVA2D`'s functional model,
> **ignoring the paragraph about the Wait Gate**."

That paragraph is the `while (SrcA[...].AllowedClient != MatrixUnit) { wait; }`
loop `WaitGate.MATH_ALLOWED_CLIENT_INSTRUCTIONS[0]` implements, and
`MOVDBGA2D` was in the list. Not being gated is the instruction's entire
purpose. It is unreachable today — `matrix.py` has no handler, so an issued one
raises — so it moves no number; it is fixed anyway because it is an
**over-charge**, and over-charging is the one direction the bounds policy
forbids. `test_movdbga2d_is_not_gated` pins the removal with the citation, so
the list is now deliberate in both directions.

### Measured, at one-cycle resolution

Every replay guard in the tree, replayed socket-free and pumped to
`RUN_MSG_DONE` one cycle at a time, on this tree and on a frozen copy of the
tree without these changes. **Model off: every guard byte-identical**, on both
architectures — nothing arms an occupancy with `TT_SIM_COST_MODEL` unset, so
`_occupied_thread` is never written and the predicate never reaches
`is_occupied()`. Model on, the guards that moved (all others identical):

| guard | before | after | of which C2 |
| --- | --- | --- | --- |
| `blackhole/five` | 13,088 | 13,121 (+33) | — |
| `blackhole/five_fp` | 13,088 | 13,132 (+44) | — |
| `blackhole/four_fp` | 11,606 | 11,617 (+11) | — |
| `blackhole/loopback` | 11,246 | 11,290 (+44) | — |
| `blackhole/optest` | 13,217 | 13,228 (+11) | — |
| `blackhole/where` | 11,379 | 11,397 (+18) | — |
| `wormhole/examples:four-fp` | 9,197 | 9,241 (+44) | +22 |
| `wormhole/examples:five` | 10,699 | 10,806 (+107) | +0 |
| `wormhole/examples:five-fp` | 10,940 | 11,014 (+74) | +0 |
| `wormhole/examples:loopback` | 8,969 | 8,991 (+22) | +0 |

**+11 to +107 cycles, every one of them slower**, which is the only direction
a condition that used to clear early can move a total. The C2 column is a
third measurement, on a copy of this tree with that one line reverted:
Blackhole cannot be affected by it at all (it takes the other branch), and
across Wormhole it accounts for 22 cycles on one example and nothing anywhere
else. `wormhole/reduce` — the guard flagged as most at risk, because its
unpack/math pairing has no semaphore and leans on `STALLWAIT` alone — does not
move at all, and neither does any of the other twelve Wormhole guards.

Tests: five in `unpacker_cost_model_test.py` (the predicate during the
transfer; a `STALLWAIT` on C1 held for the whole of it; the same, unchanged,
under the two perturbations `_ping_pong` uses — sliding the thread by up to
five cycles and reversing the backend unit tick order; the retire-tick
behaviour with the model off; and C1/C2 naming unpacker 0 and 1 on both
architectures) and one in `waitgate_allowed_client_test.py`.

Also removed, with no behavioural change: eight keys in `read_unpack_state`'s
returned dict (`whichADC`, `ch0ZInc` and friends) that only the ADC counter
update read, and that update moved into the address phase in the section
above. A latched copy of state the address phase has already consumed is an
invitation to consume it twice.

## The Configuration Unit's residency: the Latency column, read at last

Landed 2026-08-12, the third fix in this family and the same shape as the two
above — *the model retires too early* — but on the only unit where the two
published columns genuinely disagree. **No number in either YAML changed, no
computed value moved, and no simulated cycle moved either, with the model on or
off.**

> `ConfigurationUnit.md` (BlackholeA0), instruction table:
> `SETC16` **latency 1 cycle, IPC 3**, group `ThreadConfig`; `STREAMWRCFG`
> **≥ 5 cycles, IPC 1**; `WRCFG` **2 cycles, IPC 1**; `CFGSHIFTMASK`
> **2 cycles, IPC ½**; `RMWCIB` **1 cycle, IPC 1**; `RDCFG` **≥ 2 cycles,
> IPC 1** — the last five all in group `Config`. Wormhole's page prints the
> same latencies against a prose Throughput column.

So `WRCFG` is **latency 2 at one instruction per cycle**: the unit accepts the
next instruction a cycle *before* the previous one has left. Until now tt-sim
read only the IPC column, and the Latency column had no consumer anywhere in
the tree. The consequence was at the Wait Gate:

> `STALLWAIT.md` (BlackholeA0), condition **C12**: "Any thread has an
> instruction in any stage of the Configuration Unit pipeline." Its note: "The
> block mask should include bit B7 ... This won't prevent other threads from
> issuing new Configuration Unit instructions though, and those new
> instructions will cause this thread to continue to wait."

`hasInflightInstructionsFromThread` answered from the issue queue, which this
unit drains in the tick after acceptance, so every instruction was reported as
gone after one cycle whatever its documented latency. `WaitGate.
_check_blackhole_condition`'s C12 branch was live and unreachable.

### It was worse than "always satisfied": it was tick-order

The issue queue is non-empty only between the *issuing* thread's Wait Gate
pushing into it and the unit's next tick draining it. Backend units tick before
the gates, so whether C12 saw anything at all came down to whether the issuing
thread's gate happened to run before the *waiting* thread's within the cycle —
an artefact of the order `TensixBackend.getClocks` returns, not a property of
the machine. Measured, on a two-thread program where thread 0 issues a burst of
`WRCFG` and thread 1 waits on C12 (`backend_cost_model_test`,
`_c12_drain_cycle`), the cycle the wait clears at:

| `WRCFG` burst | before, gate order as-is | before, gates reversed | after, all four orderings |
| --- | --- | --- | --- |
| 3 | 4 | 3 | 5 |
| 6 | 7 | 3 | 8 |
| 10 | 11 | 3 | 12 |

Reversed, **ten instructions in flight were as invisible as none**. After, the
answer is the burst plus the trailing pipeline cycle plus `STALLWAIT.md`'s own
"one cycle lag between the condition(s) being met and the block mask being
removed", under both tick-order perturbations (backend units reversed, Wait
Gates reversed) and at every burst length — because it is now a deadline the
Latency column sets.

### What is charged, and which end of each bound

`UnitCostModel.latency` resolves the Latency column through the same
`BOUND_POLICY` as `occupancy`, so `RDCFG`'s "≥ 2" arms **2** and
`STREAMWRCFG`'s "≥ 5" arms **5**. Every bound in this unit is an `at_least`,
and the low end of an `at_least` is the number the document prints — which for
a *residency* is the under-reporting end, and that is the safe one: a residency
held longer than the hardware's makes a `STALLWAIT` on C12 wait for a stall
that does not exist, which is inventing back-pressure, the one direction the
bounds policy forbids. (Had there been an `at_most` here the same reasoning
would charge its low end too, i.e. under-report; there is none.)

**A latency of 1 arms nothing at all**, by construction rather than by special
case: the deadline lands on the retire tick itself, and the instruction was
already visible through the issue queue for the cycle between acceptance and
that tick. `SETC16` and `RMWCIB` therefore keep exactly the behaviour they
always had, and only `WRCFG`, `RDCFG`, `CFGSHIFTMASK` and `STREAMWRCFG` extend
it. The hold runs from *acceptance*, one cycle before the retire tick, the same
anchor `TensixBackendUnit.clock_tick` uses for occupancy.

**The writes are untouched.** `_arm_residency` changes only how long an
instruction is *reported* as being in the unit; every handler still runs in the
tick it always ran in. Delaying the config write itself is the ordering bug
this unit already found once — an accepted `SETC16` overtaken by its own
thread's later `MVMUL`s, and `matmul_block` printing 608.0 for 1120.0 — and
this deliberately does not go near it.

### The 13-bit `wait_res`, and a source conflict resolved against the vendor

Residency alone would have left C12 unreachable from a real kernel, because
`TensixSyncUnit._read_wait_res` trimmed Blackhole's condition mask to 12 bits
and C12 is bit 12. That width came from ttsim's `data/bh/tensix_isa.json`
(`STALLWAIT/args/wait_res: "11:0"`), and the published page disagrees. **The
page wins, and the disagreement is recorded rather than papered over.** For 13
bits: the BlackholeA0 `STALLWAIT.md` syntax line reads
`TT_STALLWAIT(/* u9 */ BlockMask, /* u13 */ ConditionMask)`; its encoding
diagram (`Diagrams/Out/Bits32_STALLWAIT_BH.svg`) gives ConditionMask bits
**0–12**, BlockMask 15–23, opcode 24–31; its condition table has thirteen rows,
C0 through C12; and tt-metal's own Blackhole LLK header
(`tt_llk_blackhole/common/inc/ckernel_instr_params.h`) defines
`p_stall::CFGEXU = 0x1000`, which a 12-bit field could not hold. For 12: one
field in one vendor data file — whose own executor (`ttsim/src/tensix.cpp`,
`TT_ARCH_VERSION == 1`) still hands `0x1000` to
`TTSIM_VERIFY(!wait_res, UnimplementedFunctionality)`, i.e. is written as
though bit 12 could arrive while its decoder makes that impossible. Four
agreeing statements against one internally inconsistent one, and
`PROVENANCE_RANK` already ranks `isa_doc` above `vendor_source` for exactly
this case.

Bits 14:13 stay excluded, so the leak the trim was written to stop is still
stopped. And what the trim replaced was not a shorter wait but the *wrong* one:
a `STALLWAIT` whose only condition was C12 had its mask emptied and fell
through to `handle_stallwait`'s `0x7F` default, which on Blackhole selects
C0–C6 — a different set of conditions, not a subset. If a card ever shows bit
12 is inert, `TensixSyncUnit.WAIT_RES_BITS` is one entry; per ["why there is no
`measured` provenance"](#why-there-is-no-measured-provenance) a measurement
would not silently win against four citations.

### Measured, at one-cycle resolution — and nothing moved

Every replay guard in the tree (44), replayed socket-free and pumped to
`RUN_MSG_DONE` one cycle at a time, on this tree and on a frozen copy without
these changes. **Model off: all 44 byte-identical**, which also covers the
`wait_res` widening, since that is not gated on the model — no in-tree kernel
sets bit 12 (nothing in tt-metal 0.74 references `p_stall::CFGEXU`). **Model
on: all 44 byte-identical too.** The gate is `RESULT: PASS`, and each of the
three budget-dependent guards needs exactly the multiplier it needed on the
frozen tree — `dramtop` 1×, `two` 2×, `offline` 4×.

Zero movement is the honest result and **not** an absence of mechanism, so it
was probed rather than assumed. Instrumented under the model across the 42
guards a single un-laddered run can drive (`blackhole/offline` and
`blackhole/two` need the gate's poll-budget ladder), the residency is armed
**1,280 times** and live for **1,280 cycles**, in **33 of the 42** — most in
`wormhole/examples` (185), `blackhole/six` (165) and `blackhole/untilize`
(135). And **nothing looks**: of 4,739 calls to the predicate, 4 land while any
thread is resident and **0** concern the resident thread, because
`CoprocessorDoneCheck` is polled only a handful of times per guard; and C12 is
evaluated **0 times anywhere**, because no in-tree kernel issues a `STALLWAIT`
on it (nothing in tt-metal 0.74 emits `p_stall::CFGEXU`). So the residency is
live, correct and currently unobserved by the tree's workloads — which is
exactly why a card probe wanted it, and why the unit tests rather than the
guards are what pin it.

Tests: nine in `backend_cost_model_test.py` section 6 (the two columns read off
the table and shown to differ; a `WRCFG` reported in the pipeline for its two
cycles with the write still retiring where it did; `SETC16`'s latency 1 arming
nothing; the residency scoped to the issuing thread; the model-off predicate;
the C12 wait end to end; the same under all four tick orderings; and the
model-off measurement of the tick-order dependence, kept as the record of what
it was) and two in `blackhole_decode_test.py` (bit 12 in the field, bits 14:13
still out).

## Using it, when the time comes

```python
from tt_sim.perf.costs import load_costs

costs = load_costs("blackhole")
mvmul = costs.instruction("MATH", "MVMUL")   # or costs.find("MVMUL")
mvmul.occupancy.cycles                        # 1
mvmul.scales_with                             # "fidelity_phases"
costs.section("noc")["hops"]["router_to_router"]["latency"]   # 9
costs.unsourced()                             # the honest gaps, as CostEntry
```

Two things a consumer should do rather than assume:

- **Check `bound` before treating a number as exact.** `ATCAS` is `≥ 15`, not
  15. A model that reports `15` as though it were measured is making the
  estimator's central mistake.
- **Handle `None`.** An instruction with no published cost returns an entry
  whose `latency` and `occupancy` are both `None`, and `find` returns `None`
  for an opcode not in the table at all. Falling back to "1 cycle" silently is
  a choice, and it should be made explicitly and once, not per call site.

Both of those are now made once, in `tt_sim/perf/model.py`, and a unit should
reach the tables through it rather than call `load_costs` itself — otherwise it
is making the same judgement calls again, differently.
`test_the_cost_tables_have_exactly_the_consumers_we_expect` pins the list of
modules that name the tables at all, so adding one is a reviewed change rather
than a drive-by import.
