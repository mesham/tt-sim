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

1. **DRAM residue** — endpoint occupancy closed 2026-08-09 (see §1).
   What remains is blocked, but **not all of it on measurement**:
   Blackhole's `dram.bandwidth` is `unknown` because BlackholeA0 has
   **no DRAM tile directory in the ISA docs at all**, and a silicon
   measurement is `corroboration`, never provenance — so no card run
   can make it chargeable. That is a documentation gap, like
   `CMD_BUF_AVAIL`. What a card *could* do it now **has**: the
   endpoint-occupancy term is **corroborated on silicon** — one
   channel flat across 1 → 120 readers while a fan-out control scales
   x4.9, at a rate matching rung 2's independent derivation to
   ~0.05 %. See item 2.

### Tier 2 — high value, small-to-medium effort

2. **Silicon follow-ups** — three Blackhole sessions ran 2026-08-09
   (see §2). The endpoint-occupancy term is **corroborated on
   silicon**, the fetch onset is bracketed by five measured
   footprints, and the session is twelve probes. **All four offline
   analyses are done** (2026-08-10): the congestion pair reproduces,
   the (11,2) epoch is closed, the fetch ramp is a graded 1/F rise to
   a ceiling, and phase Q's failure was a grader mislabel. What is
   left needs **hardware**: the dvalid mode-vs-card-state run that
   re-opens the Matrix Unit, a Blackhole run of the redesigned `RDCFG`
   probe, and the **Wormhole follow-on** — which now also carries the
   DRAM sustained-rate sweep, whose prediction is committed. Both
   probe redesigns landed 2026-08-12; neither has been to a card.
3. **Tensix issue latency & wait-gates** — the ThCon whole-thread hold
   landed 2026-08-11 with no number changed, and srcA/srcB gating was
   already modelled (see §3). What is left is **three gaps with two
   causes**: two await an arbitration rule the docs do not publish,
   while the unpacker's issuing-thread interlock is *sourced* and
   blocked on timing the Src `AllowedClient` flip against the
   unpacker's occupancy. PC-buffer write delays are **unsourceable**,
   not deferred; the mover point fixes remain.

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

- **Endpoint occupancy — DONE 2026-08-09.** The premise held, and was
  worse than written: three same-cycle 4 KiB reads were *all* serviced
  on one cycle, and end to end a Wormhole DRAM channel sustained the
  NoC link's **32 B/cycle against the 24 B/cycle its own ISA page
  publishes**. `dram.channel_serialisation` existed but was only ever
  spent as a latency, never held as a resource; it is now both. **No
  number entered any table** — the charged quantity is the existing
  `isa_doc_derived` entry. The device's own re-issue interval stays
  `unknown` and uncharged: charging `access_latency` as occupancy
  would assert 0.32 B/cycle against 24 published on the same page.
  A second premise failed usefully — a tt-sim DRAM tile fronts **two**
  GDDR6 channels, so one queue per tile would over-charge, and each
  physical channel now has its own watermark. Six Wormhole guards move
  +0.05 % to +0.75 % and **every guard with zero waits moved by
  exactly zero**; all 30 Blackhole guards are cycle-identical.
- **Blackhole `dram.bandwidth`** stays `unknown` ⇒ **neither channel
  serialisation nor endpoint occupancy on BH** — one missing figure
  now gates *two* terms, which is why every BH guard is untouched by
  the work above. Worth ~24 % on `six` for the size term alone, but BH
  read/write rates differ by 26 % so a scaled Wormhole number is
  wrong. Needs measurement — and **the same experiment that measures
  it is the only thing that can validate endpoint occupancy on either
  arch**: rung 2 cannot, because every retained DRAM row is
  `num_transactions = 1` and a lone request never finds the channel
  busy. That experiment is the sustained-rate sweep `wh_dram`'s own
  table describes — N tiles reading 1 MiB from one channel, N swept.
  Two open items, one card measurement.
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
- **The (11,2) tile clock epoch is real in 5 of 5 runs, and its two
  "absences" were a reader bug.** `noc_congestion_sweep`'s session-span
  guard used `max − min` over **32-bit** stamps; both `noc-epoch` runs
  cross `2**32` mid-run, so the span read ~4.29e9 against a true
  ~3.8e8 and swallowed every real offset — which also forced their
  congestion verdicts to `INVALID`. Fixed (`_elapsed_span` accumulates
  signed-wrapped steps); both now read `CONGESTION MEASURED` with all
  five runs overlapping 1.00. The offset is constant *within* a launch
  and different in **every** launch (+1,143,914,610 / −495,379,666 /
  −782,897,612 / −1,460,817,587 / −1,760,493,889), and is **not**
  frequency drift: the within-file rate is < 5e-9, predicting a few
  thousand cycles between probes against 3e8 observed — refuted by
  five orders of magnitude. Same-core durations, which every
  coefficient is fitted to, never involve two tiles' stamps, so
  **closed as a non-issue** — for a sharper reason than first thought.
- **`CMD_BUF_AVAIL` is unreadable, and no re-run will change that.**
  It is an *occupancy* (reset default 0, paired with `CMD_BUF_OVFL`),
  and it reads `0x00000000` at rest **and** in every in-loop sample.
  Zero at rest is correct and is not a depth. The remaining route to
  the FIFO depth is **Tenstorrent publishing it — one number**, not
  measurement.

### Still open, and what each needs

- **The DRAM sustained-rate sweep — validation, not unblocking.** N
  Tensix tiles reading 1 MiB from **one** DRAM channel, N swept, as
  `wh_dram`'s own performance table describes. It is the **only
  validation available for the endpoint occupancy term on either
  arch**, because rung 2 cannot reach it: every retained DRAM row is
  `num_transactions = 1`, and a lone request never finds the channel
  busy. Predict against the published table before running — 1 / 12 /
  48 tiles on one channel measure 22.2 / 22.3 / 22.3 GB/s, an
  aggregate that does **not** grow with readers, which is endpoint
  occupancy as a vendor measurement.
  **What it does NOT do is make Blackhole chargeable.** Its
  `dram.bandwidth` is `unknown` for want of a published page, not for
  want of a number, and silicon enters as `corroboration` only —
  `costs_test.py::test_the_dram_channel_rate_is_exactly_its_own_derivation`
  exists precisely to stop Wormhole's 24 being laundered into that
  gap. Rung 2 already *sizes* Blackhole at 47.1 B/cycle reading and
  59.4 writing; the 26 % asymmetry is why a scaled Wormhole number is
  wrong, and it would remain wrong with a measured one.
  **RAN ON A BLACKHOLE CARD 2026-08-12, and the level disagrees with
  the model by 26–35 %.** The shape holds — fan-out control ×4.85
  against one-channel ×1.018, every gate passed — but the plateau is
  **47.147 B/cycle and sits at *neither* modelled ceiling**: not the
  64 B/cycle NoC link tt-sim flattens on, not the DRAM channel, which
  Blackhole does not publish. tt-sim predicted 62.2–64.0 across the
  sweep. The measured 47.1 agrees with **rung 2's independent sizing
  to ~0.05 %**, so two unrelated methods now size a bound the cost
  tables do not hold. This is currently the best-evidenced gap in the
  model, and it is **not closable by measurement** — no published page
  means no provenance. Artefacts:
  `perfbench/card-sessions/2026-08-12/`. Unexplained: 2 readers reads
  40.96 B/cycle, *below* 1 reader's 46.31, before recovering.
  Wormhole remains untested; that part is still the outstanding item.

  **The probe is built and the prediction is committed** (2026-08-12):
  `perfbench/dramratebench/prediction-sustained.csv`, pinned cell by
  cell, has Wormhole flat at 23.75 → 23.99 B/cycle — the right shape,
  **+7 % over the published 22.2 / 22.3 / 22.3**, which is the
  documented `achievable_fraction: 0.92` the cost table deliberately
  refuses to fold in (24 × 0.92 = 22.1). What remains is **a Wormhole
  part**, not code.
  One correction this forced: **shape alone cannot separate the
  endpoint from the DRAM tile's inbound link**, because a scaling
  ratio is invariant to the level and the two resources run at
  different rates. tt-sim reports `ENDPOINT BOUND` on Blackhole at the
  vendor's own parameters with **no endpoint modelled at all** — the
  flat arm is on the 64 B/cycle link. Read the plateau's height, not
  just its flatness; this entry measures *a bound at the shared
  endpoint*, and only the level says which one.

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
- **The fetch ramp is a GRADED ramp, not a step — analysis closed.**
  1.0000 / 1.0915 / 1.1573 / 1.2115 / 1.2538 at 4096–6144 B, then flat
  (1.2529 at 7168, 1.2522 at 8192). Four *decreasing* increments
  exclude a single step outright. The rising region is linear in
  **1/footprint, r2 0.99983**, implying a **~4038 B covered window
  against a natural 4096 — 1.4 % on a parameter the fit was never
  given**. But that fit predicts 1.3245 at 7168 B against a measured
  1.2529, so there are **two mechanisms**: the ~4 KiB window producing
  the 1/F rise, and a **~1.253 cycles/instruction ceiling** taking
  over. A capacity model cannot make the plateau; a bandwidth model
  cannot make the rise.
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

### Built and RUN on the card, 2026-08-09 — twelve probes

- **`dram`** (`perfbench/dramratebench/`) — **the endpoint-occupancy
  term is CORROBORATED on silicon.** One channel is flat at
  **46.33 → 47.12 B/cycle across 1 → 120 readers** (x1.02) while the
  same readers fanned across banks reach **227 B/cycle** (x4.90);
  per-reader throughput on one channel falls as exactly 1/N (46.3,
  23.1, 11.3, 5.68, 3.84, 2.89, 1.95, 1.47, 0.98, 0.39), which is what
  perfect serialisation at an endpoint looks like. Same reader count,
  same issue loop, same transaction size — only the endpoint differed.
  **And 47.1 B/cycle is what rung 2 already derives for Blackhole DRAM
  reads from the vendor's 8,140-point NoC dataset**: two wholly
  independent methods agreeing to ~0.05 %, which is the strongest
  cross-check this model has had.
  Still `corroboration`, never provenance — BlackholeA0 publishes no
  DRAM page, so `dram.bandwidth` stays unchargeable there. And the
  `samecore` arm fires on neither descriptor (tt-metal maps every bank
  to a distinct NoC coord), so this separates **the endpoint from the
  fabric**, not the GDDR6 channel from its inbound router link.
  Dataset: `tt_sim/perf/datasets/dramratebench-blackhole-2026-08-09.csv`.
  *First run was refused by its own tag check — the tag was written
  only at slice 0, so every reader past the first had nothing to match
  and the guard condemned data that was in fact clean. A check that
  cannot pass is as damaging as one that cannot fail.*
- **`tensix-rdcfg`** — **ran, and the construction does not reach the
  quantity.** The control moved (`SETDMAREG`+`STALLWAIT` 2.968 against
  0.998 bare, so the stall instruction costs something) but the paired
  difference is **0.0000 cycles/pair**, under the half-cycle floor.
  **Diagnosed and redesigned 2026-08-12.** `p_stall::TRISC_CFG` is
  condition C10/C13 — *the RISCV T core's* outstanding memory requests
  against GPRs, config or TDMA-RISC, not a Tensix instruction in the
  Configuration Unit's pipeline. The name is the trap: it was a
  correct measurement of the wrong quantity. The condition that does
  observe it is **C12 / `p_stall::CFGEXU`, Blackhole only** ("any
  thread has an instruction in any stage of the Configuration Unit
  pipeline"), which `RDCFG.md` explicitly recommends. Slots 22–25 take
  the difference against an in-unit `RMWCIB0` baseline; 20/21 stay as
  the falsification control. **Wormhole needs no probe** — there
  `RDCFG` blocks its issuing thread for the whole duration, so the
  `>= 2` is an occupancy and probe 14 already reaches it; only a
  dataset is missing.
  **The C12 construction ran on a card 2026-08-12 and did not reach
  the quantity either.** All three arms cost identically to four
  decimal places — 2.9682 whether the preceding instruction was
  `RDCFG`, an in-unit `RMWCIB0` or an off-unit `SETDMAREG` — with the
  intended condition confirmed used (`COND: C12 CFGEXU 0x1000`) and
  every bare occupancy at 0.998.
  **Settled from the docs 2026-08-12: no stall can reach it.**
  `ConfigurationUnit.md` tabulates the `>= 2` under a column headed
  **Latency** at **IPC 1**, so the 0.998 occupancy is the documented
  throughput, not a contradiction — it is a GPR-write latency, and a
  busy condition observes occupancy. Nor does the `riscvbench`
  dependent-operand method transplant: `RDCFG.md` says *"software must
  ensure that the instruction(s) immediately after `RDCFG` are not
  trying to consume the GPR written by"* it, and an obligation on
  software is the documented **absence of an interlock** — a close
  consumer reads a stale value rather than waiting.
  So it is measured as a **distance, not a duration**: sweep the
  producer-to-consumer separation and take the smallest at which the
  value is fresh every time (`--vis-reps`, `TTBENCH_VIS_DMIN`). That
  is a **lower** bound, the direction the charging policy takes bounds
  in; `d_min = 2` corroborates the `>= 2`, `d_min = 1` leaves it
  unreached rather than refuted. Slots 22–25 stay as the documented
  negative, and slots 28/29 are a **C12 liveness control** using the
  cross-thread path (C12 is "ANY thread"), which is the only
  doc-supported way to hold it longer than `STALLWAIT`'s own one-cycle
  lag. **Wormhole needs none of this** — there `RDCFG` blocks its
  issuing thread, so the `>= 2` is an occupancy and probe 14 reaches
  it.

  **Open simulator question this surfaced:** `_read_wait_res`
  (`tt_sim/pe/tensix/backends/sync.py`) decodes Blackhole `wait_res`
  as 12 bits where the ISA page gives `u13`, so `CFGEXU` was trimmed
  off and the wait degraded to the default mask — which on Blackhole
  selects C0–C6, i.e. **the wrong wait, not a shorter one**. There
  were **two independent gaps**: that decode width, and the
  Configuration Unit retiring inside the cycle that issued, which left
  `hasInflightInstructionsFromThread` empty whenever another thread's
  Wait Gate looked.
  **Both closed 2026-08-11.** The unit now holds the residency
  `ConfigurationUnit.md` tabulates in its **Latency** column (`WRCFG`
  2 at IPC 1 — the unit accepts its next instruction a cycle *before*
  the previous leaves, which is why residency needed its own column
  and could not be read off `busy_until`), and `wait_res` widened to
  13 bits. That last is a **source conflict decided on rank**: four
  sources say 13 — the page's `u13` syntax, its encoding diagram, its
  C0–C12 table, and tt-metal's own LLK header, where
  `p_stall::CFGEXU = 0x1000` cannot fit in 12 bits — against ttsim's
  `data/bh/tensix_isa.json` alone, whose own executor passes `0x1000`
  to a check its decoder can never deliver. `PROVENANCE_RANK` already
  puts `isa_doc` above `vendor_source`. Reversing it is one constant,
  `BLACKHOLE_WAIT_RES_BITS`.
  C12 is now observable and deterministic — `burst + 2` under every
  clock and Wait-Gate ordering, where before ten instructions in
  flight were as invisible as none. **No cycle moved**: the residency
  arms 1,280 times across 33 guards, and of 4,739 predicate calls none
  concerns the resident thread. Nothing in tt-metal 0.74 emits
  `CFGEXU`, so only the card probe reaches it.
- **phase G at 4608/5632 B** — ran; four gsets written, all rows
  differing. With gset 0 that is **five measured footprints — 4608,
  5120, 5632, 6144, 7168 B** — around a ramp whose onset was bracketed
  only to (4096, 5120]. **Answered 2026-08-10: a graded 1/F rise to a
  ~1.253 cycles/instruction ceiling, not a step** — see item 1.

Both magics bumped, which is safe: the sweep readers parse `magic=` as
metadata and never validate it (proven three ways). **Renumbering**
would not have been — `probe_id` is a CSV column *and* a bit of every
`--probes` mask — so all four slots are appended rather than inserted
in footprint order.

Also from that session: a **third** rung-3 `tensix` sample (86.144,
against 86.125 and X1's 86.12 — reproducing to 0.02 %); `nocread`
correctly re-graded `DEGENERATE` by the falsifying dist test, on
silicon; and `rv-cross`'s phase-Q failure, which was a **mislabel, not
a threshold breach** — the pair was `n=16 → 70, n=32 → 48`, everything
from n=32 up is **bit-identical** between the two runs, and the grader
read the pair's *larger* n. Fixed to grade by the smaller burst.
Measured run-to-run scatter is roughly constant at 21–45 cycles while
the monotone step grows with burst, so **the first pair phase Q can
honestly be gated on is 32 → 64**; the README's `n <= 16` is
defensible but one notch tight.

### Not built, by design

`ATCAS`/`ATINCGETPTR` against a real L1 semaphore (not side-effect-free
under a 64x-unrolled replayed block, which the plan doc does not
mention); the `TTBENCH_UNROLL` sweep (three blockers, none a flag); a
divide magnitude sweep (the dividend is hardcoded, and the interesting
end is the *narrow* 9-12 bit dividends real kernels use, not this
benchmark's 29 significant bits).
`perfbench/README.md`'s "Designed, not built" carries the recipes.
**Closed, do not build: the `DIR_BIDIR` hang** — `check_invariants`
refuses bidirectional flows under two tests, and tt-metal skips its
own `core_bidirectional` family with `// Timeout issue (#36428)`.

## 3. Tensix issue latency, wait-gates & PC-buffer timing

The ThCon half closed 2026-08-11. `ScalarUnit.md`'s "no instructions
of any kind from the issuing thread can pass through its Wait Gate …
regardless of which unit that next instruction executes in" is a
whole-thread hold that no per-unit issue refusal can express; it is
now a per-thread deadline at the gate (`thread_issue_block`), armed
from the ThCon occupancy already in the table, so **no number
changed**. srcA/srcB wait-gate stalls were already modelled on both
sides. ttsim is **not** an oracle here (also not cycle-accurate);
cycle assertions must come from the ISA docs.

**PC-buffer write delays are unsourceable, not deferred.**
`BabyRISCV/PCBufs.md` publishes the FIFO depth exactly — 16 32-bit
values — but the next sentence puts the overflow in "shared buffers
within the RISCV B memory subsystem", whose capacity is published
nowhere. For a queue depth the low end is the *over-charging* end, so
bounding at 16 invents back-pressure the hardware does not have, the
one direction the floor policy forbids. No in-tree workload touches a
PCBuf at all: tt-metal 0.74 launches TRISCs through mailboxes. The
**read** side is implemented per `PCBufs.md` (2026-08-11) and, having
no in-tree caller, is covered only by `tt_sim/pe/pcbuf_test.py`.

The unpacker's **issuing-thread** interlock closed 2026-08-11 once the
Src `AllowedClient` flip moved to the end of the transfer, where
`UNPACR_Regular.md` puts it — tt-sim had flipped it at retire, leaving
one-cycle margins that a correctly anchored charge spent. The same
window now answers `STALLWAIT`'s C1/C2 ("any stage of Unpacker N's
pipeline"), which used to clear at retire mid-transfer.

**Two gaps remain, one cause** — no published arbitration rule: the
80 B/cycle joint ceiling shared between two streaming unpackers, and
the cross-unpacker half of the address-phase interlock ("nor can any
other thread start an `UNPACR`"). Both are pinned by name in the
tests. **`STALLWAIT` C7 (Matrix) and C14 (SFPU) closed 2026-08-12**,
and the earlier "inert" reading was instructive: it was about the
*occupancy* column, and the question is about the other one.
`MatrixUnit.md` publishes **latency 5** for
`MVMUL`/`DOTPV`/`GAPOOL`/`GMPOOL`/`ELWMUL`/`ELWADD`/`ELWSUB` and **4**
for the `MOV*` forms, all at IPC 1; `VectorUnit.md` **2** for the
SFPU's arithmetic and LUT rows. Both units reported every instruction
out of the pipeline up to four cycles early, and the values were
already in `tensix_instruction_costs.yaml`, unread. The Configuration
Unit's residency moved up into `TensixBackendUnit` and both units now
read their own Latency column. **No computed value and no guard cycle
count moved**, model on or off — but unlike C12 this one is
*observed*: `blackhole/reduce` goes from 0 blocked looks to 14 of its
16 latched Matrix waits, `twolaunch` from 0 to 4. The guard totals
hold because the host polls every 100 cycles and the extra cycles are
absorbed inside a poll window — a statement about resolution, not a
claim of no effect. **C14 is inert for a stated reason**: the Wait
Gate costs three cycles before the instruction a `STALLWAIT` blocks
and the SFPU's deepest latency is 2, so nothing there reaches past it;
a row deeper than 3 would show without a rewrite.

Separately, `STALLWAIT`'s empty-condition-mask default is now
per-arch. `STALLWAIT.md` gives `0x0F` on Blackhole against Wormhole's
`0x7F`, and tt-sim used `0x7F` on both — which on Blackhole is **not a
superset but a different set**: bits 4–6 there are the Matrix Unit
pipeline (an invented wait) and `SrcA`/`SrcB` not yet back with the
*unpackers* (inverted). No number moves, because the LLK always passes
an explicit `p_stall::` mask. `SEMWAIT` with a zero mask is
`UndefinedBehavior()` on both arches, so tt-sim's fallback there is
its own choice; it now follows the same per-arch constant.

## 4. Rung-4 calibration

The bar for claiming anything stronger than "first-order estimator":
match a captured silicon cycle **trace** within X %, instruction for
instruction. Needs golden traces — one per major unit (RV-only,
Tensix-only, NoC-heavy) — checked in under
`driver/wormhole/server/traces/`.

**"No in-tree cycle count has ever been compared to silicon" is
retired as of 2026-08-12 — it is no longer true.** Three independent
comparisons now exist. Component slopes: six `perfbench` probes
against a Blackhole card, agreeing to ~1 % (the `divu` row's −82 % is
a deliberate charge-the-floor policy outcome, not a miss). An
application: `nekbone` (4 elements, 16³, single core) profiled on both
this simulator and a p150, **all fifteen per-core zones within
±10.2 %, mean absolute error 7.3 %, total −3.9 %**, host work
excluded — and, as the control, **2.7–5.3× under-prediction with the
cost model off**. A shape: `dramratebench`'s endpoint scaling, though
its *level* is 26–35 % out (§2).

What that does **not** license is "cycle-accurate". A per-launch total
matching to ±10 % is a much weaker claim than a matched trace: the
totals could agree while the interior is wrong in compensating
directions, and nothing yet checks the interior against hardware —
that is exactly what this item is for. Say **"performance
estimator, corroborated at the launch and slope level"**; do not say
cycle-accurate, and do not say uncompared.

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
- PC buffer waits emit no trace event, unlike `Mailbox` and `TTSync`;
  adding one needs a new `Unit` enum member and schema-doc updates,
  so it is a schema change rather than a point fix. (The read-side
  handshake itself landed 2026-08-11; write delays are unsourceable,
  see §3.)
- Extra core types: `tt_device.py` raises beyond BRISC/NCRISC/TRISC0-2;
  widen if ERISC becomes bridge-visible.
- Watchdog residual: a loop longer than the confirmation window still
  reads as progress. Note the **launch-on-a-missing-tile** hang is
  *not* this and was never a watchdog problem: the guard already named
  the absent tile, but its `os._exit` was itself the hang, because UMD
  calls `nng_recvmsg` with no deadline and a dead simulator strands the
  host for ever. Closed 2026-08-11 — the server now identifies the host
  as the owner of the wire's listening socket and stops it, on the
  launch guard and on any simulator exception, exiting only once it
  has. Fails closed on every ambiguity; `TT_SIM_NO_HOST_STOP=1`
  disables. Accepted residuals: a materialised core blocked on a ghost
  peer with no identifiable host still hangs, and PID reuse in the
  moment before `SIGTERM` is unguarded.
- Ctrl-C on the *host* orphans the server — `uv_spawn` uses
  `UV_PROCESS_DETACHED`, so the simulator never sees the terminal's
  SIGINT. The mirror image of the hang above; `driver/sim_procs.sh`
  exists to clean up after it.
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
