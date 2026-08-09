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

1. **DRAM residue** — endpoint occupancy first: pure model shape, no
   new data needed, and now the only actionable fidelity item that
   does **not** need a card. The rest is measurement-blocked.

### Tier 2 — high value, small-to-medium effort

2. **Silicon follow-ups** — a full ten-probe Blackhole session ran
   2026-08-09 and settled seven bullets outright (see §2). What is
   left is one card experiment (the dvalid mode-vs-card-state run,
   which is what re-opens the Matrix Unit), two probes that need
   *code* rather than card time, and the Wormhole follow-on.
3. **Tensix issue latency & wait-gates**, then mover / PC-buffer
   timing point fixes.

### Tier 3 — larger or later

4. **Rung-4 calibration against silicon traces** — the eventual bar
    for claiming anything stronger than "estimator".

### Tier 4 — opportunistic / housekeeping

5. **Tracing & observability follow-ups** — the perf-budget decision,
    `chip_id`, the Jupyter / NoC-heatmap demonstration, and the long
    tail.
6. **Functional backlog** (pick up when a kernel demands it) — Tensix
    backend gaps, NoC registers/atomics, device/tile infrastructure,
    Blackhole ISA extensions.
7. **Architectural clarity & quick wins** — module boundaries,
    docstring audit, diagrams, ISA index; shellcheck,
    `MEM_BOOT_CODE_BASE`.
8. **Parked decisions & re-measurements** — Wormhole reset fan-out,
    the Blackhole watchdog A/B, re-timing the example sweep.

Closed and removed from this list: **bottleneck attribution for the
compiler team** (2026-08-07 — the frozen schema at
[`docs/trace-schema.md`](docs/trace-schema.md), `StallEvent`,
`TT_SIM_PROFILE`, source-level hotspots, honest framing), **Numba /
the threading revival** (2026-08-07), **a cheaper live Tensix tile**
and **the MVMUL gather/scatter** (2026-08-08), **the issue-loop
model** (2026-08-08) and **the outstanding-read-request credit limit**
(2026-08-09 — the number was wrong on one arch and the name wrong on
both; the probe that replaced it has since **retired the term on
silicon**, see §2). See "Not to be started" for what each settled. History is in git;
`driver/wormhole/docs/profiling.md` and `docs/plans/cost-model.md`
have the numbers.

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
- **Making the Numba warm-up cheaper** — **declined 2026-08-08 on
  measurement.** It is ~800 ms per process of which only **~30 ms is
  the on-disk cache**: ~450 ms is `import numba` and ~350 ms is
  numba's lazy target-context init, which a *one-line* kernel pays in
  full. Not reducible without an AOT build or a second dependency,
  both refused. Backgrounding it on a thread measures **+42 %**, not a
  win — the compile thread and the simulator's numpy calls convoy on
  the GIL. `TT_SIM_NUMBA_THRESHOLD` stays **512**: 128 costs
  `matmulidx` 789 ms, 1024 costs `matmulblock` 260 ms. Note the
  accelerator earns its keep only from `six` (4096 MVMULs, −16 %)
  upward and is a small net loss on `matmulblock`; the "~20 % on
  `matmulblock`" once claimed was conditional on removing a cost that
  cannot be removed.
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
- **Searching the docs for an outstanding-request bound** — **search
  closed 2026-08-09, do not repeat.** All 319 ISA-doc markdown files
  were swept, both NoC subtrees read in full (note Blackhole's is
  `BlackholeA0/`), plus both arches' tt-metal headers. `credit`,
  `backpressure`, `inflight`, `trid`, `depth`, `slots` and `command
  buffer` appear **nowhere** in either NoC tree. Every
  outstanding-request quantity that *is* published — 16 trids, 255
  each, 4 request initiators, 4 command buffers, the 128 self-throttle
  — is **byte-identical across the two parts**, so no arithmetic on
  published numbers can ever yield the per-arch pair the measurement
  needs. The one nearby hit is advice without a bound
  (`WormholeB0/DRAMTile/README.md`: software "is encouraged to limit
  its number of outstanding read requests"). The only route to a
  chargeable `isa_doc` entry is Tenstorrent publishing the Blackhole
  NIU request FIFO depth — **worth asking for; it is one number.**
- **A credit-limit mechanism in the NIU** — **not to be built until
  there is a number.** A bounded queue stalling at K predicts a rate
  of `L/K`, which must rise with distance; the dataset's own geometry
  pairs move `L` by 88 cycles (WH) and 147 (BH) while the sustained
  rate moves by ≤ 0.01 and ≤ 0.06. Hiding that needs K ≥ 8800 / 2450
  against an 8-bit counter. **The limiting mechanism is
  distance-independent and is not a round-trip credit**, so building
  the queue would install the hypothesis the evidence ranks last, at
  the cost of a branch on the hot path of every NoC request.
  `add_outstanding_noc_request` appending without a bound is not
  currently known to be wrong.
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

## 1. DRAM residue

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

## 2. Silicon follow-ups (Blackhole box)

**A full session ran 2026-08-09** via `perfbench/run_card_session.sh`,
ten probes in one block. Analysis is in `docs/plans/cost-model.md`,
"The second rung-3 sample" and "The read floor". What is below is what
that session settled, and what it left.

### Settled — do not re-run

- **The initiator credit limit is RETIRED.** Read off the register
  with a clean instrument (`inflight_rest = 0` in every row, two
  independent in-flight measures agreeing): in-flight is **2–3 at
  every burst length** (4/16/64/128), reaching 13 only in three rows
  at one transaction size, so it tracks size by Little's law rather
  than hitting a ceiling. Decisively, `cycles_per_tx` is **flat to
  1.5 % across hops 1–13**, and a credit limit `K` caps the rate at
  `round_trip/K`, which must rise with distance. This confirms on
  silicon what the vendor-dataset geometry argument concluded.
- **The Tensix instruction queue is resolved, and it is per-thread.**
  Backlog flat across four doublings to n = 1024; depth **27–33
  entries** (the old "~31–32" is the centre, not a constant). Phase S
  separates the hypotheses outright: t2 reads 1.03x/0.97x and t3
  1.06x/1.05x against a *shared* queue's 0.50x/0.33x, with spin
  controls at exactly 1.00x. **One queue per baby core.** The "longer
  `.ttinsn` burst sweep" bullet is closed — n = 128 would have done.
- **The `tensixbench` warm-up bias is measured, and is not worth
  acting on**: ≤ 0.0022 cycles/instruction on the retained t1 series,
  about **6 % of `resol`** against the control subtraction's 94 %.
  `--blocks 64` should **not** become the default — it doubles work
  per point to remove a bias 15x smaller than the resolution
  containing it. To shrink `resol`, attack `unroll`.
- **The instruction-fetch step is a ramp, not a cliff** — 4096 B
  1.000, 5120 B 1.153, 6144 B 1.252, and flat to ±0.001 across two
  further footprints. It **saturates at 6144 B**; the plateau is fully
  resolved.
- **Congestion is measured**, and reproduced across three independent
  runs to ~0.2 %: FLAT at 64 B and 512 B, **SATURATING** at 2 KiB
  (+2.49 cycles/shared link), 8 KiB (+10.87) and 16 KiB (+22.49), with
  the transition between 512 B and 2 KiB. Caveat that must travel with
  these: r2 is 0.34–0.39, so they are **shape verdicts off poor fits**
  — reproducible, not well-determined coefficients, and
  `vendor_source`-grade at best.
- **The hop line corroborates the ISA docs**: `4364.0 + 9.03*hops` and
  `4361.7 + 8.85*hops`, r2 1.00, against a published ~9 cycles
  router-to-router.
- **The (11,2) tile clock epoch is NOT a per-core constant.** It reads
  +323,438,586, then −495,379,666, then *absent* across three runs.
  The detector's "reproduced over 5 runs" counts repeats *inside* one
  invocation, so the "a constant that reproduces across independent
  runs cannot be a scheduling delay" argument never applied to it.
  Same-core durations, which every coefficient is fitted to, remain
  unaffected — this closes the bullet as a non-issue.
- **`CMD_BUF_AVAIL` is unreadable, and no re-run will change that.**
  It is an *occupancy* (reset default 0, paired with `CMD_BUF_OVFL`),
  and it reads `0x00000000` at rest **and** in every in-loop sample.
  Zero at rest is correct and is not a depth. The remaining route to
  the FIFO depth is **Tenstorrent publishing it — one number**, not
  measurement.

### Still open, and what each needs

- **Phase A cannot currently measure the Matrix Unit reproducibly.**
  16 of 19 probes and all of phase B corroborate the X1 dataset
  (≤ 0.012 cycles/instruction; the marginal MVMUL reproduces to
  0.6 %), but `MVMUL`/`ELWADD`/`ELWMUL` moved ~6 cycles and run 2's
  MATH t1 series is **bit-identical to its own NOP series** — the
  front end's 1-IPC floor, carrying no information about the unit.
  The X1 retraction ("with one legal SETDVALID the matrix unit is a
  plain shared port") rested on a comparison in which the dvalid mode
  and the binary changed *together*; its support is withdrawn.
  **Needs:** `--dvalid-once` vs `--dvalid-per-thread` on **one**
  binary, four runs, each after `tt-smi -r 0` and each on a
  deliberately dirtied card. Card state is the confound neither run
  controlled — SETDVALID runs leave the Src banks owned by the Matrix
  Unit with no release.
- **The fetch ramp's onset** is bracketed to (4096, 5120] and needs
  exactly the two builds that were never made: **4608 B and 5632 B**.
  New probe code, no card time.
- **What caps the read rate at ~50 cycles/tx**, now that the credit
  limit is retired and the rate is flat over 13 hops. A new question
  the session raised; the issue loop is the leading suspect, so it is
  better designed after the `riscvbench` work than before.
- **riscvbench phase R's verdict logic.** The failures are the R2
  gate, not monotonicity, and every one is multi-thread
  `rv_store_spread`; two runs at identical parameters fail *different*
  slots, so it is scatter about a threshold. Phase Q's n-threshold
  device will not help — this wants a measured tolerance, as phase S
  uses. No cost-relevant number is affected (t1 reads r2 1.0000 in all
  four samples). No card needed.

### Needs the planned Wormhole follow-on

- The store-coalescing and multiply pairs — predicted identical on
  Wormhole, measured 5.2x apart on Blackhole, and the cheapest
  cross-arch discriminators. Blackhole repeatability is already
  closed: four samples agreeing to three decimals.
- Any first-ever Wormhole rung-3 sample.
- The per-arch half of the read floor.

### Not built, by design

`ATCAS`/`ATINCGETPTR` against a real L1 semaphore; `RDCFG`'s latency
as `(op + STALLWAIT) - (1-cycle op + STALLWAIT)` (the cheapest, and
the obvious next one); the `TTBENCH_UNROLL` sweep (three blockers,
none a flag); a divide magnitude sweep (the dividend is hardcoded).
`perfbench/README.md`'s "Designed, not built" carries the recipes.
**Closed, do not build: the `DIR_BIDIR` hang** — `check_invariants`
refuses bidirectional flows under two tests, and tt-metal skips its
own `core_bidirectional` family with `// Timeout issue (#36428)`.

## 3. Tensix issue latency, wait-gates & PC-buffer timing

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

## 4. Rung-4 calibration

The bar for claiming anything stronger than "first-order estimator":
match a captured silicon cycle trace within X %. Needs golden traces —
one per major unit (RV-only, Tensix-only, NoC-heavy) — checked in
under `driver/wormhole/server/traces/`. Until then, no in-tree cycle
count has ever been compared to silicon; say "performance estimator",
not "cycle-accurate".

## 5. Tracing & observability follow-ups

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

## 6. Functional backlog

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

## 7. Architectural clarity & quick wins

- **Disassemble the estimator kernel's non-stateful read loop.** It
  reaches `ncrisc_noc_fast_read_any_len` via `Noc::async_read`, which
  passes `read_req_vc = NOC_UNICAST_WRITE_VC` — **VC 1, the unicast
  *write* VC** — where the stateful path leaves the sticky VC alone.
  Two read paths on different VCs is a cheap candidate for why only
  the non-stateful Wormhole rows show the burst-depth step. No card
  needed.
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

## 8. Parked decisions & re-measurements

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
