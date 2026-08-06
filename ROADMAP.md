# Roadmap

What to do next, in priority order. Performance — both the fidelity of
the modelled cycle counts and the simulator's own wall clock — is
weighted first throughout. History lives elsewhere: what has already
landed, and how, is in git history and the plan docs
(`docs/plans/blackhole-support.md`, `docs/plans/event-driven-pump.md`,
`docs/plans/cost-model.md`, `docs/plans/tensix-cost-benchmark.md`,
`docs/plans/riscv-front-end-benchmark.md`,
`driver/wormhole/docs/profiling.md`,
`docs/upstream-examples-status.md`). Older docs that cite this file's
previous §A–§L layout refer to the revision before this one — see
`git log -p ROADMAP.md`.

Context, in one paragraph: with Tenstorrent's
[ttsim](https://github.com/tenstorrent/ttsim) owning bit-exact
functional correctness (and tt-sim matching it across `optests/` on
both arches precisely *because* it is the oracle), tt-sim's lanes are
the **cycle-approximate performance estimator**, hackability,
observability tooling, differential testing, and education. The
priority list below is shaped by the first lane: the Tensix front-end
bound, the poll-until-DONE guard conversion, the silicon-backed RV cost
fixes, the last two Tensix unit costs and firmware-loop parking (with
and without the cost model) have all landed, so what remains is getting
that attribution out to the people who will act on it, and closing the
few uncosted terms that are left.
The first named external consumer is the **compiler team**: they will
drive kernels through tt-sim to trace where cycles go — instruction
mix, data movement, stalls — and generate more efficient code from it,
which is why cost-model fidelity and easily-consumed trace output
(item 1) carry the weight they do here.

---

## Priority list

### Tier 1 — load-bearing, start here

1. **Bottleneck-attribution workflow for the compiler team** — a
   documented, stable path from "run this kernel" to "ranked
   bottleneck report" over the Perfetto/Parquet outputs. Promoted:
   the fidelity work it depended on has landed (stall reasons are
   real, data movement is costed, grid-scale runs are affordable), so
   what remains is packaging the attribution for the people who will
   act on it.

### Tier 2 — high value, small-to-medium effort

2. **The rest of the RV memory path** — the load/store unit's
   published queue limits and per-region throughput, all `isa_doc`
   and all unconsumed. Successor to "cost the NIU register block",
   which landed 2026-08-06.
3. **Freeze the cheap guards**: `noc_tile_transfer`,
   `optests/dramtop`, `vecadd_sharding` on Blackhole, Wormhole
   `matmulidx`/`matmulblock` offline guards.
4. **Smarter `TT_SIM_CYCLES_PER_POLL` default** — the fixed 100-cycle
   pump after every wire message dominates at scale.

### Tier 3 — medium-term

5. **Issue-loop model** — worth more rung-2 dataset coverage than any
   remaining congestion work.
6. **DRAM residue** — endpoint occupancy first (pure model shape, no
   new data needed); the rest is measurement-blocked.
7. **Next silicon session on the Blackhole box** — a bundle of cheap
   probes once a card is in hand, including the first-ever Wormhole
   measurements.
8. **Tensix issue latency & wait-gates**, then mover / PC-buffer
    timing point fixes.
9. **Numba on the pump's event heap, and the `nogil` threading
    revival** — the ranked JIT target now that pump Phase 4 exists.
10. **Rung-4 calibration against silicon traces** — the eventual bar
    for claiming anything stronger than "estimator".

### Tier 4 — opportunistic / housekeeping

11. **Tracing & observability follow-ups** — the perf-budget decision,
    `chip_id`, and the long tail.
12. **Functional backlog** (pick up when a kernel demands it) — Tensix
    backend gaps, NoC registers/atomics, device/tile infrastructure,
    Blackhole ISA extensions.
13. **Architectural clarity & quick wins** — module boundaries,
    docstring audit, diagrams, ISA index; shellcheck,
    `MEM_BOOT_CODE_BASE`.
14. **Parked decisions & re-measurements** — Wormhole reset fan-out,
    the Blackhole watchdog A/B, re-timing the example sweep.

### Not to be started

- **Pump Phase 3** (per-component gating) — the plan doc forbids it as
  the next phase; revisit only if a multi-tile workload shows
  per-component cost dominating after Phase 4.
- **Congestion beyond the wired link step** — declared unmodellable
  with current evidence (needs a per-link flow census, not a
  coefficient); VC arbitration measured at ~1/50th of the link effect.
- **`SFPLOADMACRO`** — needs a deferred-issue notion tt-sim doesn't
  have; ttsim declines it too. Workaround:
  `TT_METAL_DISABLE_SFPLOADMACRO=1`.
- **Ethernet chip-to-chip** — no second chip to model against.
  Blackhole eth tiles deferred until an eth kernel needs them.
- **Native fast dispatch** — blocked at the UMD wire-protocol level
  upstream (see `CLAUDE.md`); slow dispatch suffices.
- **Quasar, L2CPU / Security tiles, harvesting** — out of scope by
  decision.
- Watch item, not work: Tenstorrent may ship their own perf model or
  public cost tables at any time — if so, the headline goal needs
  re-evaluation. Watch ttsim and tt-metal release notes.

---

## Working rules

These apply across every item below and are enforced by tests where
possible:

- **Cost model**: charge every bound at its low end (the model is a
  floor); no `estimated` provenance entries; a whole-tree timing
  change gets its own instalment in `docs/plans/cost-model.md` and its
  own `driver/tests/cost_model_gate.py` run; silicon measurements
  enter as `corroboration`, never as provenance.
- **Optimisation**: profile first; never use `TT_SIM_TRACE_COUNTERS`
  for simulator profiling (~2× distortion — it is for kernel
  profiling); every speedup claim is an interleaved A/B between two
  frozen git worktrees with a verified-zero control workload — the
  tree usually has concurrent work in it, so this is not optional.

---

## 1. Bottleneck attribution for the compiler team

The cost model's first named external consumer: the compiler team will
run generated kernels through tt-sim to find where cycles go —
instruction mix, data movement, stalls — and use that to emit more
efficient code. The pieces mostly exist; what's missing is the
consumable end of the pipe:

- **A documented, stable trace schema.** Perfetto (visual timelines)
  and the Parquet counter/NoC datasets (machine-readable) already
  carry the model's attribution — NoC flight and bandwidth, unit
  occupancy, RV stalls. Write the schema down (field meanings, units,
  stability expectations) so the team can build tooling against it
  without reading tt-sim internals; treat schema changes as breaking
  from then on.
- **Stall *reasons*, not just stall counts.** The front-end bound has
  made refused issues real, so the FPU/SFPU stalled-cycles and
  wait-reason fields are now expressible, and the unpacker wiring has
  made packer back-pressure and unpacker idle cycles real too. These
  are the fields a code generator acts on — surfacing them in the
  trace output is what is left.
- **Source-level attribution.** `DwarfIndex` function-name attribution
  (currently in item 11's long tail — promote it here) so a hotspot
  names a kernel function and line, not a bare PC; kernel-ELF
  auto-discovery so the workflow needs no hand-holding.
- **An entry-point workflow.** One documented invocation (env vars or
  a small driver) from "here is my kernel" to a ranked report of
  where modelled cycles went, with the Jupyter/NoC-heatmap examples
  as the worked demonstration.
- **Honest framing baked into the output.** Modelled numbers are
  floors built from published bounds, corroborated but not calibrated
  against end-to-end silicon (item 10) — the report should say so, so
  the team optimises against relative attribution, not absolute
  cycle promises.

A first version is assemblable today — `six` already attributes ~75 %
of its modelled cycles to a published number — and its fidelity
improves automatically as items 2 and 5–8 land. The tracing
overhead budget (item 11) matters doubly here, since the team will
run this at kernel scale.

## 2. The rest of the RV memory path

**"Cost the NIU register block" landed 2026-08-06 and was never
provenance-blocked.** The ">= 7" row's cell names six things and two of
them are "NoC 0 configuration and command" and "NoC 1 configuration and
command", on both architectures; only this file's key name for the row
(`tdma_tilectrl_pic_noc_overlay`) said otherwise. The blocks are
charged, `docs/plans/cost-model.md` has the instalment, and the item
7 probe it would have needed is off that list.

What the census behind it found is the successor, and it is the same
shape: **published, `isa_doc`, and consumed by nothing.**

- **`max_loads_in_flight`** — 8 for core-local RAM and **one shared
  budget of 4** across every other region (the column's cell is a
  four-row rowspan, which the YAML note had read as covering only the
  mailbox group; corrected). Already in `unit_costs.yaml`, read by no
  consumer, and a barrier poll loop is exactly the shape that hits an
  in-flight cap.
- **Per-region request throughput** — "Each memory region can process
  at most one request per cycle" for every region but L1
  (`BabyRISCV/MemoryOrdering.md`), which is the one term that would
  make two cores hammering the *same* NIU cost more than one core
  doing it. Not in the tables yet.
- **Sustained-load throughput** — `riscv.load_throughput` holds the
  docs' "four such loads every N - 1 cycles" formula, is corroborated
  by silicon to 0.008 of a cycle (`rv_load_indep`), and is likewise
  consumed by nothing.
- Still genuinely unsourced, and staying that way: the "more in the
  case of access conflicts" tail on the >= 7 row, and the three blocks
  left in `RV_UNNAMED_REGIONS` (MOP expander config, NCRISC IRAM, the
  Tensix instruction push buffers), none of which appears in any row
  of either architecture's table.

## 3. Freeze the cheap guards

From the upstream sweep's own value ranking
(`docs/upstream-examples-status.md`):

- `noc_tile_transfer` — the only multi-core example with a real
  self-check, both arches, 2 tiles, fast.
- `optests/dramtop` — a one-liner now that the DRAM top-down
  allocation fix is in.
- `vecadd_sharding` on Blackhole — the only in-tree coverage of
  L1-sharded buffers driven by a compute-only kernel (4 workers,
  14 s; recipe is in the sweep doc).
- `pad_multi_core` / `shard_data_rm` — 4-core sharded data movement;
  no self-check, so freeze expected DRAM contents.
- Wormhole offline guards for `optests/matmulidx` + `matmulblock`
  (Blackhole has them; Wormhole is live-diff only).
- `matmul_multi_core` @4x5 only if ~800 s of CI is acceptable.

## 4. `TT_SIM_CYCLES_PER_POLL` default

The bridge pumps `cycles_per_poll` (default 100) cycles after *every*
wire message, including the pure host DMA of input/output buffers — at
80 workers a readback ran for an hour that finishes in seconds at
`=10`. Fix the default rather than documenting around it: skip the
pump on host-DMA messages, or adapt it to grid size.

## 5. Issue-loop model

The residual on every rung-2 prediction is one constant unmodelled
issuing-core path (intercepts 77–94 cycles across four independent
series). An issue-loop term is worth more of the 8,140-point tt-metal
NoC dataset than congestion was — 24 sole-cause entries vs 2 — so it
is the ranked next step for widening rung-2 coverage.

## 6. DRAM residue

- **Endpoint occupancy** — a second request is not queued behind the
  first: latency without contention. Pure model shape; no new data
  needed. Do this one first.
- **Blackhole `dram.bandwidth`** stays `unknown` ⇒ no channel
  serialisation on BH — explicitly worth ~24 % on `six`, but BH
  read/write rates differ by 26 % so a scaled Wormhole number is
  wrong. Needs measurement.
- The BH DRAM-write over-charge (8 negative residuals, −12 to −28,
  pinned in `KNOWN_OVER_CHARGED`) — splitting `access_latency` by
  request action still rests on one arch's data.
- The BH `l1_local_cycles = 88` rung-1 anomaly (54 cycles
  unexplained) — rung 1 cannot fully pass on BH until explained.
- Bank conflicts / refresh windows — no DRAM bank model, nothing
  published; long-term.

## 7. Next silicon session (Blackhole box)

Cheap probes to bundle into one card session — plan and analyse here,
run there:

- A second `tensixbench` run (or a second part): every rung-3
  conclusion rests on one sample; `corroboration` fields are written
  to be extended.
- `ATCAS` / `ATINCGETPTR` probes against a real L1 semaphore — the
  `≥ 15` entries are the largest untested numbers left in the ThCon
  table.
- Latency-difference probes ((op + `STALLWAIT`) − (1-cycle op +
  `STALLWAIT`)) — `RDCFG`'s `≥ 2` latency is the only unchecked half
  of an entry left.
- The unroll sweep (`TTBENCH_UNROLL` ∈ {16, 32, 64, 128}) — closes the
  per-instruction-vs-per-block ambiguity under the ~6-cycle Wait-Gate
  MVMUL figure.
- Drop `tensixbench`'s n = 32 burst or add a warm-up — makes the fits
  exact and shrinks instrument resolution.
- `riscvbench` phase-G points at 4608/5632 B bodies — settles the
  instruction-fetch cliff's shape; seconds each.
- Congestion points between the 64 B and 16 KiB regimes
  (`--shared-sizes`); the `DIR_BIDIR` hang; the (11,2) tile clock
  epoch.
- **Wormhole, first measurements ever**: the store-coalescing pair
  (predicted identical on WH, measured 5.2× apart on BH) and the
  multiply pair are the cheapest cross-arch discriminators.
- A divide magnitude sweep (several dividend widths) — the only route
  to the 6–33 curve, since the docs publish no formula and one point
  cannot pin one (recorded as a negative result in the RV cost-fixes
  instalment).
- The longer `.ttinsn` burst sweep — silicon's ~31–32-entry queue
  depth is a lower bound still growing at the longest burst; needed
  before the front-end bound's constant can ever be called calibrated.

## 8. Tensix issue latency, wait-gates & PC-buffer timing

Wait-gate stalls on srcA/srcB availability and the documented per-op
issue cadence, now expressible through the landed front-end bound.
ttsim is **not** an oracle here (also not cycle-accurate); cycle
assertions must come from the ISA docs. PC-buffer write delays remain
a point fix. Two under-charges left behind by the unpacker wiring
belong here too, both pinned by name in the tests: the 80 B/cycle
joint ceiling shared between two streaming unpackers, and the
cross-unpacker half of the address-phase interlock ("nor can any
other thread start an `UNPACR`") — each needs a sourced arbitration
rule that the docs do not currently give.

## 9. Numba and the threading revival

Nothing has yet *needed* Numba; the ranked target is the event-driven
pump **after Phase 4's** heap-of-(cycle, unit, event) shape — a
polymorphic `clock_tick` over 240 objects is not a JIT target. Tensix
numeric inner loops must be re-evaluated against the post-optimisation
baseline (the cheap numpy wins already banked 1.6–1.8×). The
`@njit(nogil=True)` angle is what could make multi-Tensix threading
(structurally landed, strictly opt-in via `TT_SIM_THREADED=1`, a
1.56–2.4× measured regression under the GIL) a win without waiting for
free-threaded 3.13t — tempered by the fact that ~70 % of matmul time
is one unit on one tile. Numba-hostile, do not attempt: `MemoryMap`
dispatch, `EventBus.publish`, the current clock tick, `frontend.py`
YAML decode. Rejected alternatives, recorded so they aren't
re-litigated: Cython (build step breaks hackability), PyPy (numpy
interop), C extension (last resort).

- *Test:* re-run the 4-Tensix `four/`-derived benchmark with
  `TT_SIM_THREADED=1`; target wall clock **under** sequential.

## 10. Rung-4 calibration

The bar for claiming anything stronger than "first-order estimator":
match a captured silicon cycle trace within X %. Needs golden traces —
one per major unit (RV-only, Tensix-only, NoC-heavy) — checked in
under `driver/wormhole/server/traces/`. Until then, no in-tree cycle
count has ever been compared to silicon; say "performance estimator",
not "cycle-accurate".

## 11. Tracing & observability follow-ups

- **Perf-budget decision**: the ~30 % tracing overhead target is not
  met on RV-bound workloads (counters/Perfetto ~2×, JSONL ~4×) —
  re-target the budget or make `EventBus.publish` cheaper at the call
  site.
- Fields now expressible but not yet surfaced: packer back-pressure
  and unpacker idle cycles (the unpacker is wired), FPU/SFPU
  stalled-cycles (the front-end bound makes refused issues real —
  74 on `sfpumath` alone). Still genuinely gated: L1 bank conflicts
  (L1 is flat memory) and NoC `vc` + VC occupancy (no VCs modelled).
- **Parked spans in the event stream** (handed over by the
  firmware-loop parking work). A core
  the firmware-loop recogniser has parked is not traced, and the tile
  is not stepped while it is dormant, so a skipped cycle publishes no
  `InstrEvent` — which is why parking refuses whenever the bus or a
  core's snoop is on, and why kernel-scale tracing at grid scale still
  costs full price. Replaying the elided events at unpark is
  O(skipped cycles) (4.6 M on an 80-worker `nine` replay) and hands
  the entire win back, so the shape to design is one **summary event
  per parked span** — core, loop extent, iterations, cycles — plus
  whatever the Perfetto/Parquet consumers need to render it. New
  event kind, so it is a schema decision, not a patch.
- `chip_id` hard-coded to 0 (per-tile `core_y`/`core_x` already fan
  out) — multi-chip identity.
- Long tail: inline-print migration, `DwarfIndex` function-name
  attribution, kernel-ELF auto-discovery, invariant mining (tt-metal
  #28562), `hypothesis` property tests (needs a kernel generator),
  host-polling determinism (co-design with pump time-skipping),
  L1/DRAM snapshots in state dumps, `SpikeCommitlogWriter` mem-access
  decoration, Jupyter template, NoC heatmap example, Pyright cleanup.

## 12. Functional backlog

Pick up when a kernel or example actually demands it; everything here
fails loudly today. Grep for `NotImplementedError` in the named files.

**Tensix** (`tt_sim/pe/tensix/`):
- ThCon: `ATCAS` / `ATSWAP` / `ATINCGET` (`backends/thcon.py`; the
  NoC-dispatch half is shared with the NoC item below).
- Mover: region-crossing transfers (`backends/mover.py`); 16 KB /
  single-region limit; unmapped L0 writes silently dropped.
- Missing handlers: `MOVDBGA2D`, `SHIFTXA`, `SHIFTXB`, `SFPLZ`
  (decode-only); `SFPMUL24`'s `>>23` mode; `SFPMOV` `FROM_SPECIAL`;
  SFPU 16-bit lane-format match-defaults.
- Matrix: BF8/FP4-style input formats for the elementwise path
  (*test:* `four-bf8/` / `four-fp4/` ELWADD variants).
- Packer: ADC context, dest-addr offsets, per-datum edge mode;
  pack-side zero compression (raises when a driven packer requests
  it; ttsim models it nowhere).
- Unpacker: remaining NoOp modes; upsampling and `ColShift` raise
  (modelling `ColShift`'s drop needs silicon or a fixed ttsim — the
  ISA doc flags its own pseudocode as low-confidence).
- Numerics: rounding-mode overrides, per-thread `FP16A_FORCE` (parsed,
  not applied), accumulator persistence, the 32-bit ZEROACC bank fixup
  (deliberately left — tt-sim's `useDst32b` mapping diverges from
  ttsim's swizzle).
- **Acquire-half dvalid precondition unchecked** — the release half
  raises `SrcDvalidError`; acquiring a bank the Matrix Unit already
  owns is still silent (`backends/unpacker.py`, `backends/misc.py`).
- **Config-write ordering is a missing guarantee, currently
  unreachable** — nothing orders a config write against its readers;
  any future cost entry that delays a `SETC16`/`WRCFG` will produce a
  *wrong answer*, not a slow one. Flagged in `config.py` and
  `costs_test.py`.
- **Phase-B-shaped deadlock divergence** — `matmul_tiles` over
  resident operand tiles (`cb_wait_front` hoisted, no `cb_pop_front`)
  diverges from silicon on a shape no in-tree example uses
  (`docs/plans/tensix-cost-benchmark.md`).

**NoC** (`tt_sim/network/tt_noc.py`):
- Register coverage: many offsets beyond the basic counter set, the
  command buffers and `NOC_CFG` raise on access.
- Atomic ops beyond ATINC (semantics half of the ThCon item above).
  *Test:* a kernel using `noc_atomic_increment_with_response` waiting
  on `NIU_MST_ATOMIC_RESP_RECEIVED`.
- Router arbitration & flow control: no buffer back-pressure, no
  fairness, no virtual channels (which is why tt-sim is structurally
  blind to the `DIR_BIDIR` hang class).

**Device / tile** (`tt_sim/device/`, `tt_sim/misc/`):
- Soft-reset sequencing applied immediately (multi-step ISA sequence
  not modelled).
- Tile-control registers: `misc/tile_ctrl.py` is a silently-plausible
  generic store — a register with real side effects would go
  unnoticed.
- PC buffer write delays / synchronisation (`pe/pcbuf.py` TODO).
- Extra core types: `tt_device.py` raises beyond BRISC/NCRISC/TRISC0-2;
  widen if ERISC becomes bridge-visible.
- Watchdog residual: a loop longer than the confirmation window still
  reads as progress.
- ARC tile: nothing modelled; `arc_msg` returns 0.

**Blackhole RV extensions** (all loud guards, none reached by any
in-tree kernel): `mret`, F single-precision execution, the V vector
unit (TRISC2), Zfh's BF16 CSR mode, the L1 cache tag search
accelerator. Wormhole's RV32IM-only set is complete and correct.

## 13. Architectural clarity & quick wins

- Module boundaries per hardware block — `frontend.py` mixes decode
  and dispatch; `tt_noc.py` bundles NIU + both NoCs + directory.
  Acceptance: a table in `CLAUDE.md` mapping ISA-docs subsection → one
  file.
- Docstrings as architectural documentation — every hardware-block
  class carries block name, ISA-docs permalink, I/O, config registers
  owned, modelled-vs-stubbed list; audit script asserts the
  `ISA-docs:` link.
- Mermaid diagrams: top-level device block (root README), Tensix
  dataflow (`tt_sim/pe/tensix/README.md`), kernel-launch sequence
  (`driver/wormhole/README.md`).
- `tt_sim/ISA_INDEX.md` — grep-able ISA-docs cross-reference,
  including "not modelled".
- Shellcheck the `run.sh` family (`driver/{wormhole,blackhole}/run.sh`,
  `optests/diff.sh`, `driver/blackhole/tests/run_examples.sh`).
- `MEM_BOOT_CODE_BASE` boot-jump under the wire flow — verify against
  a captured trace whether tt-metal writes the `jal`; if not, the
  bridge should synthesise it.

## 14. Parked decisions & re-measurements

- **Wormhole NCRISC/TRISC reset fan-out**: the launch-message
  `enables` path is wired on Blackhole only; decide whether Wormhole
  should use it (needs the WH L1 offset for
  `kernel_config_msg_t.enables`). *Test:* a disabled-BRISC-bit variant
  asserting the core never leaves reset.
- **Live `vecadd_sharding` measurement of firmware-loop parking**: the
  landed 3.9× / ~1000× figures use the trace-replay proxy, so the
  14 s → 255 s figure that motivated the work has never been
  re-measured end to end.
- **Watchdog cost A/B on Blackhole**: the original ran before the
  watchdog was wired there, so both arms were watchdog-free.
- **Re-time the upstream example sweep** before quoting its numbers —
  the 2026-08-03 timings ran with concurrent edits in the tree.
- **`START` (cmd=4) wire handler** log-and-skips; revisit if a
  tt-metal release ever emits it.
- **Provenance process**: measured-only quantities (e.g. the ~25 %
  instruction-fetch cliff) currently land in `docs/bh_arch.md`; if
  that grows past a handful, decide deliberately whether provenance
  gains a `measured` rank or a companion schema — not by drift.
