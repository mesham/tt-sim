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
the simulator's own wall clock — though note that as of 2026-08-12 the
wall clock does **not** degrade with grid size (1 to 80 workers is
1.8x for a problem that uses them); the cost is materialising workers a
program never launches on. See the v2.0 list, item 2.
The first named external consumer is the **compiler team**: they will
drive kernels through tt-sim to trace where cycles go — instruction
mix, data movement, stalls — and generate more efficient code from it.
Their contract is [`docs/trace-schema.md`](docs/trace-schema.md),
frozen at `SCHEMA_VERSION` 4 — schema changes are breaking from here.

---

## Priority list

Items 1–8 map to the detail sections §1–§8. **Do not renumber** —
other docs cite these numbers.

The live load-bearing work is the **"Toward a v2.0 release"** list
below, not this one: Tier 1 cleared on 2026-08-12 and Tier 2's
remaining half needs hardware, not code.

### Tier 1 — cleared 2026-08-12

1. **DRAM residue** (§1) — both load-bearing halves closed. What is
   left is residue: the BH `l1_local_cycles = 88` anomaly, the BH
   write over-charge, and bank/refresh terms nothing publishes.

### Tier 2 — high value, small-to-medium effort

2. **Silicon follow-ups** (§2) — **the Blackhole half is done**
   (2026-08-12): the dvalid matrix closed the Matrix Unit question and
   the redesigned `RDCFG` probe ran. What is left is **one hardware
   booking**: the Wormhole follow-on, carrying the committed DRAM
   sustained-rate prediction, the store-coalescing and multiply pairs,
   and the `nocreadbench` half. No code is outstanding — every probe
   already runs with `--arch wormhole`.
3. **Tensix issue latency & wait-gates** (§3) — two gaps remain, one
   cause: no published arbitration rule. PC-buffer write delays are
   **unsourceable**, not deferred. The mover point fixes remain.

### Tier 3 — larger or later

4. **Rung-4 calibration against silicon traces** (§4) — the bar for
   claiming anything stronger than "estimator".

### Tier 4 — opportunistic / housekeeping

5. **Tracing & observability follow-ups** (§5).
6. **Functional backlog** (§6) — pick up when a kernel demands it.
7. **Architectural clarity & quick wins** (§7).
8. **Parked decisions & re-measurements** (§8).

Closed and removed from this list, with the reasoning kept in
`docs/plans/cost-model.md` and git history: bottleneck attribution for
the compiler team (2026-08-07), Numba / the threading revival
(2026-08-07), a cheaper live Tensix tile and the MVMUL gather/scatter
(2026-08-08), the issue-loop model (2026-08-08), the
outstanding-read-request credit limit (2026-08-09 — **retired on
silicon**, not confirmed), the Tensix residency terms and the
PC-buffer read side (2026-08-11), and multi-tile with on-demand
materialisation (2026-08-12). See "Not to be started" for what must
not be re-attempted.


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

## Toward a v2.0 release

**The claim a 2.0 would be making.** As of 2026-08-12 the strongest
honest one is: *runs real tt-metal programs unmodified on both
architectures, with a documented-provenance timing model corroborated
at the slope and launch level*. Everything below either removes a
caveat from that sentence or extends it.

**Multi-tile, which was this list's first two items, is done**
(2026-08-12). A real grid is validated at every size on both arches,
and workers materialise on demand with no environment variable. What
is left is the credibility layer.

1. **NoC link contention** — **not a blocker; most of it is already
   done.** The cycle cost of sharing a link is sourced and wired
   (2026-08-05, `NocLinkRegistry`, `isa_doc_derived`, ~98 % of the
   measured effect). What is open:
   * a **citation edit** — the arbitration order *is* published, in
     tt-metal's `Saturating_DRAM_bandwidth` tech report ("first-come
     first-serve strategy within the same VC") and
     `memory_for_kernel_developers.rst`, both `vendor_source` and both
     **confirming** what tt-sim implements, while `unit_costs.yaml`
     still says "nothing publishes what the real order is". No number,
     no cycle, no gate run. Cheapest correctness win in the tree.
   * **validation**, which depends on nothing here: the term is inert
     on every in-tree workload (3,960 link claims, **zero waits** on
     `blackhole/six`), so it needs a workload that actually contends.
   * **buffer back-pressure stays `unknown`** — ~2 % of the effect,
     dynamics unpublished, and charging it would be `estimated` in the
     over-charging direction. Refused, as PC-buffer write delays are.
   `tenstorrent/tt-npe`, Tenstorrent's own NoC estimator, is cited
   nowhere here. Its *shape* corroborates tt-sim's; its numbers are
   **not importable** — every constant uncited, `getLinkBandwidth()`
   30/60.9 against the docs' 32/64, `CYCLES_PER_HOP` 10/11 against the
   docs' 9 and the 8.8/8.65 this project confirmed on a card, and a
   multicast sink derate that is dead code. Cross-check shapes only,
   normalising the hop constant out, and treat it as §3 treats ttsim:
   **not an oracle**.
2. **Rung-4, a matched cycle trace** (§4). Until totals are checked
   instruction for instruction, "estimator" is the ceiling: the
   interior can be wrong in compensating directions while the envelope
   agrees. The bottleneck report's 79.8 % NoC split on nekbone is
   exactly such an unverified interior.
3. **A Wormhole card session.** Not code — a hardware booking, so it
   is lead time rather than effort and should be requested early. It
   is the binding constraint on **three** items: the committed DRAM
   sustained-rate prediction (Wormhole-only), the store-coalescing and
   multiply pairs, and the `nocreadbench` half.
4. **Upstream example coverage as a release gate.**
   `docs/upstream-examples-status.md` exists; making the sweep
   pass/fail turns "runs real programs" into a number a release note
   can carry.
5. **Energy, at ranking level.** The activity counters already exist
   (`busy_cycles`, `instr_retired`, `noc_bytes_total`,
   `noc_flight_cycles`, `dispatch_total`). Absolute Joules are out of
   reach — board telemetry is ~1 Hz against microsecond kernels — but
   *which kernel costs more* is tractable and useful to the compiler
   team. Any coefficients would be **fitted**, so they must sit
   outside the provenance ladder and be labelled as such.

**A caution this list earned.** Three of its original premises were
wrong, all the same way — read from the tree's prose rather than its
code. Contention "unmodelled" (wired since 2026-08-05, the stale claim
surviving in seven files); C7/C14 "inert" (a sentence about the
*occupancy* column answering a question about *latency*); and
scale-is-the-problem (wall clock is ~flat in worker count; the cost was
materialising **unused** workers). Each was caught by an agent told to
verify the premise first-hand. **Prose here drifts from code; check the
code.**

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

Endpoint occupancy closed 2026-08-09 and Blackhole's channel read rate
2026-08-12; both are in `docs/plans/cost-model.md`. What survives here
is what would otherwise be re-litigated:

- **The device's own re-issue interval stays `unknown` and uncharged.**
  Charging `access_latency` as occupancy would assert 0.32 B/cycle
  against the 24 published on the same page.
- **A tt-sim DRAM tile fronts two GDDR6 channels**, so one queue per
  *tile* over-charges. Each physical channel has its own watermark.
- **Blackhole's DRAM *write* rate is deliberately uncharged.** Its
  secant lands within 2.5 % of the same arch's L1 rows, so the vendor
  dataset resolves no DRAM-specific write bound, and charging it would
  deepen `KNOWN_OVER_CHARGED`. `dram.bandwidth` — the GB/s spec block —
  is still `unknown`, still needs a document, and now gates nothing.
- The BH DRAM-write over-charge (8 negative residuals, −12 to −28,
  pinned in `KNOWN_OVER_CHARGED`) — splitting `access_latency` by
  request action still rests on one arch's data.
- The BH `l1_local_cycles = 88` rung-1 anomaly (54 cycles
  unexplained) — rung 1 cannot fully pass on BH until explained.
- Bank conflicts / refresh windows — no DRAM bank model, nothing
  published; long-term.
- **The 2-reader dip is not noise.** In the 2026-08-12 sweep two of
  three repeats land on *exactly* 100.0 cycles/tx (51,204 and 51,196)
  and the third at 88.9, where every point in the 2026-08-09 session
  sits. Bimodal per repeat, gone by N ≥ 16 — so the reported 40.96
  B/cycle at N=2 is a median over a bimodal sample. Report modes, or
  take more repeats.


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

- **The DRAM sustained-rate sweep — Wormhole half outstanding.** The
  probe is built and its prediction is committed at
  `perfbench/dramratebench/prediction-sustained.csv`: Wormhole flat at
  23.75 → 23.99 B/cycle, **+7 % over the published 22.2 / 22.3 / 22.3**,
  which is the documented `achievable_fraction: 0.92` the cost table
  deliberately refuses to fold in (24 × 0.92 = 22.1). **What remains is
  a Wormhole part, not code.**
  The Blackhole half ran 2026-08-12 and left a finding: the plateau is
  **47.147 B/cycle, at *neither* modelled ceiling** — not the 64 B/cycle
  NoC link tt-sim flattens on, not a channel rate Blackhole publishes.
  tt-sim predicted 62.2–64.0, so **the model is 26–35 % high on the
  level** even though the shape holds. The measured 47.1 agrees with
  rung 2's independent sizing to ~0.05 %. That gap is **not closable by
  measurement**; the read direction is now charged from the vendor
  dataset instead (§1).
  Two things that must survive: **shape alone cannot separate the
  endpoint from the DRAM tile's inbound link**, because a scaling ratio
  is invariant to the level — read the plateau's height, not just its
  flatness. And
  `costs_test.py::test_the_dram_channel_rate_is_exactly_its_own_derivation`
  now guards the **write** direction, which is the live laundering
  route since the read direction is charged from Blackhole's own row.

- **Phase A and the Matrix Unit — CLOSED 2026-08-12.** The dvalid
  matrix ran: one binary (`sha256 0afe8825…`, recorded), four runs,
  `--dvalid-once` and `--dvalid-per-thread` crossed with clean and
  deliberately dirtied card state. **Card state was the confound**, as
  suspected, and the pre-registered rule's first branch fired: runs 1
  and 4 (both clean/once) agree to a thousandth while run 3 (dirty)
  differs. The mode itself changes **nothing** in phase A — runs 1 and
  2 are identical at every thread count.
  The dirty run reproduced the exact failure that withdrew X1's
  support: `MVMUL t1 = 0.998`, **bit-identical to its own NOP series**,
  the front-end's 1-IPC floor carrying no information about the unit.
  Its mechanism is legible — dirty t2 (6.080) ≈ clean t1 (5.988) and
  dirty t3 (12.102) ≈ clean t2 (12.009), i.e. the card behaves as
  though it has **one fewer contending thread**, which is what Src
  banks left owned by the Matrix Unit with no release would do.
  **X1's claim is re-derived and stands**, on the reset-clean runs
  only. Per-thread occupancy normalised (`t1`, `t2/2`, `t3/3`) is
  constant across 1–3 contending threads to **0.38 %** for `MVMUL` and
  0.25 % for `ELWADD`, against 0.89 % for THCON's `ADDDMAREG` and
  1.42 % for the SFPU's `SFPADD`. The Matrix Unit is not merely *a*
  plain shared port; it is the **cleanest one in the probe set**. The
  qualifier "with one legal SETDVALID" turns out to be unnecessary for
  phase A.
  **But there is a double dissociation, and it must travel with this.**
  Phase A depends on card state and not on mode; **phase B depends on
  mode and not on card state**. `matmul_tiles` HiFi2 reads 52.0 in
  runs 1/3/4 and **35.3** in run 2, whose fidelity ladder gives
  LoFi→HiFi2 of **+0.23 against a predicted +16.00** where the others
  give +17.6 to +17.7. So `--dvalid-per-thread` breaks the fidelity
  measurement while leaving thread scaling untouched. "The dvalid mode
  does not matter" is true of phase A only.
  Marginal `MVMUL` — (HiFi4 − LoFi)/48 — reads **1.0606–1.0628** here,
  a **third** independent session inside the 1.061–1.067 band.
  Artefacts: `perfbench/card-sessions/2026-08-12b/`.
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

### Ran on the card — what each probe settled

Twelve probes on 2026-08-09, two more on 2026-08-12. Artefacts are
banked under `perfbench/card-sessions/`; the numbers and derivations
are in `docs/plans/cost-model.md`. Only what still bites is kept here.

**Corroborated** (all `corroboration`, never provenance): endpoint
occupancy — one channel flat at 46.3 → 47.1 B/cycle over 1 → 120
readers (x1.02) while a fan-out control reaches x4.9, per-reader
falling as exactly 1/N; the congestion step, whose height is the
packet's occupancy to 0.88–0.99; hop latency 8.8 (WH) / 8.65 (BH)
against the docs' 9; and rung 3's `tensix` sample reproducing to
0.02 % across three sessions.

**Negative results worth not re-deriving**: the `nocread` outstanding-
read-request term was **retired** on silicon, not confirmed. The
`tensix-rdcfg` C12 construction did not reach its quantity on a card
either — all three arms identical to four decimals — which is what
established that `RDCFG`'s `>= 2` is a GPR-write latency no
busy-condition can observe (§3).

**Method lessons, all of which cost a run**:
- **A guard that cannot pass is as damaging as one that cannot fail.**
  `dramratebench`'s first run was condemned by its own tag check,
  because the tag was written only at slice 0.
- **MEANINGFUL is only ever awarded for a control that MOVED.** The
  2026-08-09 session awarded two on the grounds that a value was
  present and well-formed; `card_session_verdicts.sh` now holds the
  checks and `card_session_verdicts_test.sh` runs them against that
  session's own CSVs.
- **Never renumber `probe_id`** — it is a CSV column *and* a bit in
  every `--probes` mask. Append. (Bumping `magic=` is safe.)
- Phase Q's failure was a **grader mislabel**, not a card fault: it
  graded the larger n of a pair that is bit-identical from n=32 up.
  The first pair it can honestly be gated on is 32 → 64.
- `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` is a **harvesting**
  workaround on real silicon, not a simulator one — removing it took
  rung 5 from PASS to a YAML conversion error. Its end coords are
  **inclusive**.


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

The ThCon whole-thread hold, the Src hand-over timing, the unpacker's
issuing-thread interlock and `STALLWAIT` C7/C14 all closed 2026-08-11
and 2026-08-12; the write-ups are in `docs/plans/cost-model.md`. What
is left, and what must not be re-attempted:

- **Two gaps remain, one cause — no published arbitration rule**: the
  80 B/cycle joint ceiling shared between two streaming unpackers, and
  the cross-unpacker half of the address-phase interlock ("nor can any
  other thread start an `UNPACR`"). Both pinned by name in the tests.
- **PC-buffer write delays are unsourceable, not deferred.**
  `PCBufs.md` publishes the FIFO depth (16) but puts the overflow in
  "shared buffers within the RISCV B memory subsystem", whose capacity
  is published nowhere. For a queue depth the low end is the
  **over-charging** end, so bounding at 16 invents back-pressure the
  hardware does not have. No in-tree workload touches a PCBuf at all —
  tt-metal 0.74 launches TRISCs through mailboxes. The **read** side
  is implemented, covered only by `tt_sim/pe/pcbuf_test.py`.
- **`RDCFG`'s `>= 2` stays UNREACHED — three card runs, two retired
  methods.** `ConfigurationUnit.md` tabulates it under **Latency** at
  IPC 1, so it is a GPR-write latency, and `RDCFG.md`'s "software must
  ensure" is the documented **absence of an interlock** — a close
  consumer reads a stale value rather than waiting.
  *Stall-based approaches are retired with evidence.* The C12 run on
  2026-08-12 moved its liveness control — stall floor 0.9718
  cycles/pair at t1 against **2.9186 at t3**, where two other threads
  hold the Configuration Unit — so **C12 is live on silicon**, and
  slots 22–25 reading 0.0000 means RDCFG's post-issue residency is no
  wider than the stall's own documented one-cycle lag: structurally
  invisible to any busy-condition, not absent.
  *The distance method also does not resolve it.* `VIS_DMIN = 1` in
  all 64 repetitions — the result is visible in the very next issue
  slot. **Unreached, not refuted**: a consumer that reads its operand
  a cycle into its own execution explains it completely, and the
  timing form (slots 26/27) reads 0.0001 as the documents predict.
  A sharper consumer would be `WRCFG`, which reads its GPR on entry —
  but it writes backend configuration and this benchmark does not
  mutate device state. **Do not attempt a fourth probe without solving
  that**, and note `d_min` is a *lower* bound either way.
- The mover / PC-buffer point fixes remain.
- **Open, and a vendor-vs-doc conflict**: tt-sim decodes Blackhole's
  `wait_res` at 13 bits per the ISA page, against ttsim's data file at
  12. Four sources say 13, including tt-metal's own LLK header where
  `p_stall::CFGEXU = 0x1000` cannot fit in 12. Reversing it is one
  constant, `BLACKHOLE_WAIT_RES_BITS`.

ttsim is **not** an oracle here (also not cycle-accurate); cycle
assertions must come from the ISA docs.


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
  blind to the `DIR_BIDIR` hang class). **Not to be misread as "no
  contention"** — the cycle cost of two flows sharing a link *is*
  modelled (`NocLinkRegistry`, wired 2026-08-05, `isa_doc_derived`,
  ~98 % of the measured effect). It is the arbitration *order* and the
  buffer dynamics that are absent. See the v2.0 list, item 3.
- **`NOC_CMD_BRCST_SRC_INCLUDE` (bit 17) is ignored**, so a multicast
  whose rectangle contains its own sender always loops back. Found
  2026-08-12 during the multi-tile survey. It does not bite the
  examples tested — `contributed/multicast` excludes the sender and
  matmul-mcast writes identical data — but it is a real gap and wants
  its own guard.

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
- **Watchdog verbosity at scale**: the deadlock watchdog fires benignly
  once per run at teardown, printing ~5 lines per materialised tile —
  **~400 lines on an 80-tile grid**, with every core in a *recognised*
  firmware wait loop during host readback. Summarise identical
  entries; do **not** suppress the case, because that would also
  silence the launch-on-a-missing-tile symptom.
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
- **`CLAUDE.md` says the SoC descriptor "matches" tt-metal's default
  compute-grid range. It *contains* it** — measured 2026-08-12, the
  default is 8x9 = 72 on Wormhole (not 80) and 13x10 = 130 on
  Blackhole (not 140). One word, and the runbook is already correct.
- **Retire the stale "tt-sim models no link congestion" claim**, which
  predates the 2026-08-05 wiring and survives in seven places:
  `perfbench/README.md:49,226,248`,
  `perfbench/card_session_verdicts.sh:499,743`,
  `perfbench/run_card_session.sh:670`,
  `docs/plans/evalplan.md:378,806`. The two in the verdict script are
  **operator-facing** — they tell someone at a card that a real
  disagreement is expected. While there, add the two `vendor_source`
  citations for the arbitration order (v2.0 item 3), correcting
  `unit_costs.yaml`'s "nothing publishes what the real order is".
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
