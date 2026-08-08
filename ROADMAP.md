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
and without the cost model) have all landed, and as of 2026-08-07 so
has the attribution pipe that carries it to the people who will act on
it. What remains is closing the few uncosted terms that are left, and
the simulator's own wall clock.
The first named external consumer is the **compiler team**: they will
drive kernels through tt-sim to trace where cycles go — instruction
mix, data movement, stalls — and generate more efficient code from it.
Their contract is [`docs/trace-schema.md`](docs/trace-schema.md),
frozen at `SCHEMA_VERSION` 4 — schema changes are breaking from here.

---

## Priority list

### Tier 1 — load-bearing, start here

1. **The MVMUL gather/scatter** — *half done, 2026-08-08.* The format
   conversions and `_fpu_accumulate_batch` are now table-driven:
   **−7.1 % on `six`, −3.4 % on `matmulblock`** end-to-end (−10.5 % /
   −6.6 % with `TT_SIM_NUMBA=0`; −16 % of pump), bit-exact, all nine
   matmul-class optests still matching ttsim. Zero effect on the 23 of
   30 guards that issue no MVMUL, and **+0.9 % on `reduce`**, which
   cannot repay a 17 ms table build with 36 calls.
   What is left is **not numpy-addressable**: `_fpu_group_sums_batch`
   is 46 numpy calls at ~3.5 µs each *regardless of array size*, and
   three rewrites (operand-shaped hoisting, int32, table gather) all
   measured neutral or worse — the dead ends are recorded in the code.
   Its remaining 9× is the Numba kernel's or nobody's.
   Two follow-ups the profile exposed: the **672 ms numba on-disk
   cache load** (2.95 s cold) is now the largest single one-off in a
   matmul guard and makes the JIT only break-even below `six`; and the
   unpacker's `_unpack_block` converts *into* the same layouts and
   would take the same tables.

### Tier 2 — high value, small-to-medium effort

2. **Issue-loop model** — worth more rung-2 dataset coverage than any
   remaining congestion work.
3. **DRAM residue** — endpoint occupancy first (pure model shape, no
   new data needed); the rest is measurement-blocked.
4. **Next silicon session on the Blackhole box** — a bundle of cheap
   probes once a card is in hand, including the first-ever Wormhole
   measurements.
5. **Tensix issue latency & wait-gates**, then mover / PC-buffer
   timing point fixes.

### Tier 3 — larger or later

6. **Rung-4 calibration against silicon traces** — the eventual bar
    for claiming anything stronger than "estimator".

### Tier 4 — opportunistic / housekeeping

7. **Tracing & observability follow-ups** — the perf-budget decision,
    `chip_id`, the Jupyter / NoC-heatmap demonstration, and the long
    tail.
8. **Functional backlog** (pick up when a kernel demands it) — Tensix
    backend gaps, NoC registers/atomics, device/tile infrastructure,
    Blackhole ISA extensions.
9. **Architectural clarity & quick wins** — module boundaries,
    docstring audit, diagrams, ISA index; shellcheck,
    `MEM_BOOT_CODE_BASE`.
10. **Parked decisions & re-measurements** — Wormhole reset fan-out,
    the Blackhole watchdog A/B, re-timing the example sweep.

Closed 2026-08-07 and removed from this list: **bottleneck attribution
for the compiler team** (all five bullets — the frozen schema at
[`docs/trace-schema.md`](docs/trace-schema.md), `StallEvent`,
`TT_SIM_PROFILE`, source-level hotspots, honest framing) and **Numba /
the threading revival** (see "Not to be started" for what it settled).
History is in git; `driver/wormhole/docs/profiling.md` has the numbers.

### Not to be started

- **Pump Phase 3** (per-component gating) — the plan doc forbids it as
  the next phase; revisit only if a multi-tile workload shows
  per-component cost dominating after Phase 4.
- **Numba on the pump's event heap or the RV interpreter** — measured
  and declined 2026-08-07. The event heap is **0.9–1.6 % of a run**
  (the "6 %" once quoted was `TileClock.clock_tick`'s whole dispatch,
  a different thing); no dependency is worth 1 %. The RV interpreter
  is not a JIT target either, but *not* for polymorphism or call
  overhead — an `@njit` boundary is ~450 ns against ~11 µs per
  `BabyRISCV.clock_tick`. It is Numba-hostile **through** `MemoryMap`
  / `Register` / the Tensix queue: compiling it means replacing the
  state model, which is what this project exists not to do. That
  reason does not expire as the interpreter gets faster. Numba *is*
  used, optionally, for the one shape it suits — the exact FPU kernel
  (13.3×, −23.9 % on `blackhole/six`), lazily imported so the
  pure-Python tree is unchanged.
- **The threaded pump** (`TT_SIM_THREADED=1`) — measured at **5.56×
  slower** than sequential, worse than the 1.56–2.4× once recorded,
  and now understood: threading drives `stride_skipped_cycles` from
  5887 to **0**, so the threaded pump cannot stride at all. Every
  improvement to the sequential pump widens the gap. `nogil` scales
  1.68× at two threads then collapses, so **a free-threaded
  interpreter would not change this**; the ceiling is striding, plus
  ~70 % of matmul being one unit on one tile. Revisit only if a
  threaded pump learns to stride.
- **Cython, PyPy, C extension** — recorded so they aren't
  re-litigated: build step breaks hackability; numpy interop; last
  resort, respectively.
- **Vectorising the Tensix per-element datapath**
  (`handle_elwadd` / `handle_pacr`) — **declined 2026-08-08 on
  measurement.** The "~22 % of a tile tick" premise did not survive:
  across 27 replay workloads the two are a **median 5.8 %** of pump
  time and **14.8 % at most** (`four_fp`, `vecadd_sharding`), **0 % on
  six workloads**, and `handle_elwadd` is never called at all in 18 of
  27. A `perf_counter` shim and cProfile agree to 0.3 pp, so the
  earlier 13 %/9 % and 3 % figures differ by *workload*, not method.
  Only ~half of `handle_pacr` is per-element work — the rest is
  per-instruction config plumbing numpy cannot touch — leaving ~10 %
  addressable best case, ~2.5 % median, against a run-to-run noise
  floor of 4 % CV. Set against rewriting bf16 rounding, fidelity
  phases and Dst formats on a path with **no differential oracle
  coverage** (no optest issues a plain `add_tiles`), the risk/reward
  is negative. Revisit only if a workload puts `handle_elwadd` above
  ~15 % of pump.
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
- **Per-region RV request throughput** ("each memory region can
  process at most one request per cycle", every region but L1) —
  deferred on a census, not on provenance. The traffic it describes
  does not occur: **zero same-cycle collisions on either NIU block
  across seven guards and 138,936 MMIO requests**, because NoC 0 and
  NoC 1 are separate memory-map entries and tt-metal gives BRISC one
  and NCRISC the other. What does collide (tile control, PCBufs) is
  worth 0.2–5.5 % as an *upper* bound. Against that it would be the
  first RV cost state that is not per-core, and would sit outside the
  firmware-parking fixed-point proof. **Revisit when a workload puts
  two cores on one region** — a multi-core kernel sharing an L1
  semaphore, showing as collisions on `0xFFE8`. Census and method are
  in the `docs/plans/cost-model.md` instalment.
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

## 1. The MVMUL gather/scatter

*(formerly "a cheaper live Tensix tile". Its RV half closed 2026-08-07;
its per-element datapath half was declined 2026-08-08 on measurement —
see "Not to be started". What the measurement surfaced instead is
below.)*

**`perform_mvmul` is 47–51 % of pump time on `six` even with the Numba
FPU kernel engaged** — 745 µs per call, of which the fused `@njit`
kernel is only ~46 µs. The rest is the Dst/SrcA/SrcB rectangle gather
and the `FP32ToDstFormatBF16` / `BF16InDstToBF16` conversion helpers
around it: pure numpy work needing no new dependency, and — unlike the
declined ELWADD path — covered by a working differential oracle, since
`matmulblock` and `matmulidx` both pass against ttsim.

Two smaller findings from the same measurement, worth doing whenever
this area is next opened:

- `edge_masks_for_pacr` re-reads **20 config registers on every
  PACR** (371 µs/call, ~13 % of `handle_pacr`) for a result that
  depends only on `(stateID, packer)`. A fidelity-free memoisation,
  if invalidation on config write can be made safe.
- `optests/diff.sh` silently JITs kernels against whatever
  `TT_METAL_HOME` points at, which is what produced a phantom
  "`SrcOrder` has not been declared" break — the local
  `riscv/tt-metal` tree is v0.70.1 and does not declare it, while the
  venv's runtime root is v0.74.0 and does. Worth a one-line guard in
  `diff.sh` that fails loudly rather than JITting against the wrong
  tree.

**"Smarter `TT_SIM_CYCLES_PER_POLL` default" closed 2026-08-06, and the
default did not change.** Its evidence — "at 80 workers a readback ran
for an hour that finishes in seconds at `=10`" — predated firmware-loop
parking and no longer reproduces: `vecadd_multi_core` at the full
Wormhole 8x10 grid now PASSes in ~80 s at the default, and an
interleaved A/B against `=10` puts the two inside each other's spread
(the simulator does the same real work either way — 483 k vs 482 k live
tile ticks). The residual per-message-per-tile term the item named is
gone too: the pump now skips a window it has already proved nothing can
happen in, and inlines the dormant tick, which at 80 tiles takes a
quiescent `run(100)` from 58.7 µs to 6.1 µs and one-live-tile-among-80
from 4382 µs to 2398 µs — with every guard's simulated cycle total
byte-identical. Details in `docs/plans/event-driven-pump.md`
(Phase 4b) and `driver/wormhole/docs/profiling.md`. The advice to set
the knob by hand is retracted in both runbooks.

What that leaves is the other half of the same 2026-08-03 prediction,
and it is now the whole of the wide-grid cost: **a live Tensix tile is
expensive** (~150 µs per ticked tile, ~12 of them live at a time on the
8x10 `vecadd_multi_core` run, 71 s of its 81 s). Nothing in the pump
addresses that; it is the same target as item 2's issue-loop model,
approached from the wall-clock side. Rank it against that rather than
treating it as pump work.

**First instalment landed 2026-08-07: 146 µs → 131 µs.** The premise
was re-measured and held exactly (69.8 s of an 83.2 s run inside live
Tensix tile ticks, 477,571 of them at 146.2 µs). Profiling the real
80-worker server from the inside splits a tile tick **53 % baby RISC-V
/ 40 % Tensix coprocessor / 6 % tile-clock dispatch**, with no single
item above 13 %. What was taken out is the RV front end's *dispatch
overhead* rather than any simulated work: an instruction-fetch fast
path onto the plain-RAM leaf, the event bus held on `MemorySpace`
instead of `get_bus()` per access, `get_int` inlined and the immediate
decoders lifted out of their format-string dispatch — 1.6 M Python
calls removed from a 9.6 M-call guard, every one of the 44 guards
byte-identical in cycle accounting with the model on and off.
`driver/wormhole/docs/profiling.md` has the numbers.

**What is left is no longer one item, and the two halves rank
differently.**

- The RV interpreter is still ~45 % of a tile tick and is now close to
  what a pure-Python interpreter costs. **No further step is
  ranked**: a JIT was measured and declined (see "Not to be started"),
  and it is Numba-hostile through `MemoryMap` regardless of how fast
  the interpreter gets.
- The other ~45 % is the **Tensix datapath's per-element Python
  loops** — `handle_elwadd` walks 128 scalar FPU ops (13 % of a tick)
  and `handle_pacr` does 512 two-byte L1 writes (9 %). These are the
  largest single items in the tree and numpy could do each in one
  call, but they are a rewrite of the FPU and packer datapaths (bf16
  rounding, fidelity phases, Dst formats, edge masks) — a *fidelity*
  change, and the reason `six` (matmul, 497 µs/tick) gained least from
  this round. **This is now the whole of the item** — and it needs
  numpy, not Numba, so no new dependency.
- Measured and declined: eliding a parked core's tick (9 % at 80
  workers) needs the loop's whole watch re-read every cycle, because
  `TensixTile.next_wake_cycle`'s fast reject skips `wake_check` while
  any other core runs; ~2× on that path, against the cost model's
  scoreboard having to be time-translated per cycle. Reasoning in the
  profiling instalment.

## 2. Issue-loop model

The residual on every rung-2 prediction is one constant unmodelled
issuing-core path (intercepts 77–94 cycles across four independent
series). An issue-loop term is worth more of the 8,140-point tt-metal
NoC dataset than congestion was — 24 sole-cause entries vs 2 — so it
is the ranked next step for widening rung-2 coverage.

## 3. DRAM residue

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

## 4. Next silicon session (Blackhole box)

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

## 5. Tensix issue latency, wait-gates & PC-buffer timing

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

## 6. Rung-4 calibration

The bar for claiming anything stronger than "first-order estimator":
match a captured silicon cycle trace within X %. Needs golden traces —
one per major unit (RV-only, Tensix-only, NoC-heavy) — checked in
under `driver/wormhole/server/traces/`. Until then, no in-tree cycle
count has ever been compared to silicon; say "performance estimator",
not "cycle-accurate".

## 7. Tracing & observability follow-ups

- **Perf-budget decision**: the ~30 % tracing overhead target is not
  met on RV-bound workloads (counters/Perfetto ~2×, JSONL ~4×) —
  re-target the budget or make `EventBus.publish` cheaper at the call
  site.
- Surfaced 2026-08-07 in schema v4 as `StallEvent` (the "74 on
  `sfpumath`" turned out to be the unpacker, not FPU/SFPU). Packer and SFPU
  stall mechanisms exist but are unexercised by today's guards;
  unpacker *idle* cycles are declined with a reason. Still genuinely
  gated: L1 bank conflicts (L1 is flat memory) and NoC `vc` + VC
  occupancy (no VCs modelled).
- **The Jupyter / NoC-heatmap worked demonstration** — inherited from
  the closed attribution item's entry-point bullet, and the one piece
  of it still open.
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
- Long tail: inline-print migration, invariant mining (tt-metal
  #28562), `hypothesis` property tests (needs a kernel generator),
  host-polling determinism (co-design with pump time-skipping),
  L1/DRAM snapshots in state dumps, `SpikeCommitlogWriter` mem-access
  decoration, Pyright cleanup. (`DwarfIndex` function-name
  attribution and kernel-ELF auto-discovery closed 2026-08-07 with the
  attribution item; the Jupyter template and NoC heatmap example are
  promoted to their own bullet above.)

## 8. Functional backlog

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

## 9. Architectural clarity & quick wins

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

## 10. Parked decisions & re-measurements

- **Wormhole NCRISC/TRISC reset fan-out**: the launch-message
  `enables` path is wired on Blackhole only; decide whether Wormhole
  should use it (needs the WH L1 offset for
  `kernel_config_msg_t.enables`). *Test:* a disabled-BRISC-bit variant
  asserting the core never leaves reset.
- **Live `vecadd_sharding` measurement of firmware-loop parking**: the
  landed 3.9× / ~1000× figures use the trace-replay proxy, so the
  14 s → 255 s figure that motivated the work has never been
  re-measured end to end.
- **`matmul_multi_core` @4x5 as a guard** — declined when the cheap
  guards were frozen: ~800 s would roughly double the cost-model
  gate's wall clock, and what it uniquely adds is *scale* (20 workers
  of simultaneous DRAM I/O), not a new code path. Revisit if a
  routing or contention bug of that class recurs.
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
