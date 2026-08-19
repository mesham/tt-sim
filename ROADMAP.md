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
As of 2026-08-19 they have used it in anger and reported back, which is
where the work of the last week came from: see **"What the first
external consumer found"** below, and note that three of the four
reports were tt-sim answering confidently and wrongly rather than
answering slowly or imprecisely.

---

## Priority list

Items 1–8 map to the detail sections §1–§8. **Do not renumber** —
other docs cite these numbers.

The live load-bearing work is no longer this list *or* the
**"Toward a v2.0 release"** list below: Tier 1 cleared on 2026-08-12,
Tier 2's remaining half needs hardware rather than code, and the v2.0
list closed on 2026-08-18 with every item delivered or proven
impossible. What has landed since is in **"What the first external
consumer found"**, and none of it was planned in either list.

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
- **A C++ core called from Python — SCOPED AND DECLINED 2026-08-18, on
  a profile.** The question was whether porting "key parts" to C++ with
  bindings would be faster. **It would, and not by enough to be worth
  what it costs — and "key parts" is not on offer.** Profiling
  `matmulblock` (cost model on, 6.0M calls) gives self time
  `pe/tensix` **29.9 %**, builtins 19.8 %, `pe/rv` **17.1 %**, stdlib
  and import 14.4 %, `device` 5.5 %, `memory` 4.1 %, `pe/register`
  3.4 %, `util` 2.6 %, everything else under 1 % each.
  **There is no hotspot**: the hottest single function is
  `rv32.clock_tick` at **6.2 %**, and the work is spread over ~5,600
  functions — the signature of interpreter overhead rather than of a
  hot kernel. So Amdahl decides it. Porting `pe/rv` alone caps at
  **1.21x**; `pe/tensix` alone at **1.43x**; both at 1.9x. A real win
  needs `pe/tensix` + `pe/rv` + `device` + `memory` + `register` — the
  whole simulator core — which at a realistic 20–30x on ported code
  lands near **5x**, maybe 10–12x on long runs as import cost
  amortises.
  **And that specific set is exactly what cannot be cut apart**, for
  the reason already recorded against Numba: those subsystems interlock
  through shared mutable state every cycle, so a boundary between them
  means either porting the state model too or paying a crossing per
  memory access. A per-`clock_tick` boundary is affordable (~1–2 %); a
  per-access one eats the whole gain. What *does* work is the shape
  already in the tree — a leaf kernel with array-shaped work and no
  state-model interaction, like the Numba FPU kernels — and the profile
  shows no further candidate of that shape carrying enough time.
  Weighed against ~5x: a build step, a near-total rewrite, and the
  property the README names as the reason the project exists.
- **Pump striding — CLOSED 2026-08-18, no headroom left.** Measured
  across six replay guards: strides land within a handful of cycles of
  the Tensix tile's *dormant* count every time (`matmulblock` 3,186
  against 3,188; `four` 95,479 against 95,518). **The pump strides
  exactly when the compute tile is idle and never misses one.**
  Strided share tracks how busy that tile is — `dramtop` **0 %** (tile
  awake 100 % of cycles), `loopback` 11.2 %, `six` 13.0 %,
  `matmulblock` 18.0 %, `sfpumath` 30.7 %, `four` **88.9 %** — and the
  payoff is real where it applies: `four` runs at **45,586 cycles/s**
  against ~3.6–6k for the compute-bound guards.
  So the remaining cost is **doing the work on a busy tile, not walking
  idle cycles**, and no scheduling change can reach it. The 12x spread
  between idle-dominated and compute-dominated guards is entirely
  inside the Tensix backends, which is the `pe/tensix` 29.9 % above and
  caps at 1.43x on its own. (A first pass at this measurement read
  **0 %** everywhere; it had sampled the wrong pump object. The numbers
  here are from the pump that actually ran the cycles.)
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

**The claim a 2.0 would be making.** As of **2026-08-18**, with every
item below closed, the strongest honest one is:

> *Runs real tt-metal programs unmodified on both architectures, on the
> full default grid, with a timing model of **documented provenance** —
> zero `estimated` entries, asserted by test — corroborated against
> silicon at the **slope** and **launch** level on both parts, and
> **by mechanism** for NoC-bound work on both parts.*

**The caveats belong inside the claim, not under it, because three of
them are permanent.** A reader who takes the sentence above without
these four is taking more than the evidence gives:

- **The model is a floor.** Every bound is charged at its low end, by
  rule. So it under-predicts, monotonically, wherever hardware sits
  above a documented minimum — and it is **not cycle-accurate** and
  does not claim to be.
- **Mechanism-level corroboration is one leg on both arches, not
  three.** The NoC leg is validated on Wormhole and Blackhole
  (0.2–9.3 % against a 25 % bar). The **RV-bound** leg is
  **Blackhole-only by construction** — Wormhole documents no CSRs, so
  there is no `minstret` and the leg refuses rather than degrading —
  and on its one silicon session it **FAILS**, `E_total` 15.91 %
  against a 10 % bar. The **Tensix mechanism** leg **cannot be built at
  all** from the counters this hardware exposes.
- **That RV-bound failure is a single, sourced, permanent cause.** All
  of it is the divide floor: 13.43 % of a 15.92 % error, against 2.49 %
  for everything else, with a compensation ratio of **1.00x** so
  nothing is concealed. It cannot be closed — a dividend-magnitude term
  is unlicensable at every rank of the provenance ladder, proven by
  exhaustive search of both doc trees and every vendor tree.
- **Energy is ranking-level only, and barely beats knowing nothing.**
  Two architectures pass all thirteen gates, but against a null model
  that takes energy as proportional to simulated cycles the fit is
  worth **one rank swap in nine on Wormhole and nothing at all on
  Blackhole**. The ratios are where it earns its keep, and only on
  Wormhole. Absolute joules are out of reach of the instrument.

**Two of the items below closed by being proven impossible rather than
by being delivered**, and that is a result rather than a gap: a
limitation that has been searched for and bounded is worth more than
one that is merely unmet. Multi-tile, this list's original first two
items, is done (2026-08-12) — a real grid at every size on both arches,
workers materialising on demand with no environment variable.

0. **NoC 1 coordinate correctness — five commits, 2026-08-13/16.**
   Not on the original list; it arrived as a bug report from the
   compiler team and turned out to be the largest correctness defect
   found this cycle. **NoC 1's destination directory was keyed in two
   coordinate conventions at once** — a kernel emits canonical worker
   coords and mirrored bank-table coords on the *same* NoC, both
   observed in one replayed trace — so mirror registrations displaced
   live workers. **56 of Wormhole's 80 workers and 102 of Blackhole's
   140 resolved to the wrong tile**, and because the impostor ACKs, a
   sender's `noc_async_write_barrier()` returned normally: silent
   misdelivery, not a fault. It had been live for months behind a test
   suite that *asserted the shadowing as correct behaviour* in three
   places, including one requiring all 140 Blackhole workers to be
   reachable at their mirror coord.
   What landed, in order: `90ec8f1` names every shadow at registration,
   at `add_tensix_tile` and at the directory-miss hook
   (`TT_SIM_NOC1_SHADOW`, warn by default because configurations that
   pass today carry an untouched shadow) and drops Blackhole's mirror
   aliases, which `virtual_noc0_coordinate`'s unconditional
   `|| arch == BLACKHOLE` early-out means were never addressed
   (census 102 -> 6). `93e68fb` fixes a second defect found chasing the
   first: tt-sim decoded **Wormhole's `NOC_ID_LOGICAL` index on both
   arches**, so every Blackhole core read 0 for its own coordinate and
   `my_x`/`my_y` were `(0,0)` — which had already cost `perfbench/
   nocbench` a documented workaround nobody had traced to a cause.
   `73ad018` fixes packet *timing*: a DRAM channel answers to several
   NoC cells and tt-sim billed every packet to its tile's primary NUI —
   wrong for **all 480** Wormhole worker/endpoint pairs on each NoC and
   for **every** Blackhole NoC 1 DRAM flight (65% of its destination
   resolutions), plus a grid guard so an out-of-grid coord raises
   instead of spinning forever in `noc_route_links`. `e1016bb` makes
   **NoC coordinate translation** work, which puts every tile in a
   disjoint key space and takes both censuses to **0**.
   **Reachable today with no upstream change**, and that was the
   session's biggest wrong turn corrected: UMD's `.so` extension test
   looked like a hard blocker, but `ClusterOptions::cluster_descriptor`
   bypasses it entirely and tt-metal already plumbs it for a Simulator
   target via `TT_METAL_MOCK_CLUSTER_DESC_PATH`. Descriptors are
   checked in per arch; see runbook §1.4.
   Translation on its own is *additive*, though, so the second
   convention was still carried and the endpoint-consistency residual
   survived it — measured, and contradicting the reason first pinned
   for it. Dropping the unmirrored NoC 1 keys under translation is what
   takes that residual to **0** on both arches, with the untranslated
   path byte-for-byte unchanged and every flight a kernel can actually
   address timed identically before and after. What survives beside the
   translated keys is geometry, not preference: Wormhole keeps its
   mirrors (its translated bands are off the physical grid, so a
   physical coord is still unambiguous — and its DRAM is never
   translated) and Blackhole drops them (its translated coords *are*
   physical coords, so one worker's mirror is another's translated
   key).
   **Silicon has now had its say, 2026-08-17.** The port was validated
   only against tt-sim's own invariants and a reading of tt-metal's
   source; a Wormhole card session closed that. Its arm-C destination
   is `(2, 2)`, matching the *translated* simulator, where the
   untranslated one says `(7, 9)` -- the grid mirror; and the card
   reports `self_noc=18,18` / `peer_noc=19,19` for itself, so
   tt-metal's plumbing on a real part agrees with the descriptor
   tt-sim was handed. Wormhole is the arch where this matters, because
   untranslated it still shadows 56 of 80 workers and its translated
   coord is *not* its physical one, so every mapping is exercised
   rather than being the identity.
   **The method lesson, which is the transferable part**: every one of
   these was found by an instrument, not by reading. The static
   directory probe predicted the runtime failure set exactly before any
   simulation ran; the endpoint-consistency invariant
   (`flight_cycles_to(directory[key])` vs
   `flight_cycles_to(NullEndpoint(key))`, ~2 s, no kernel) found the
   DRAM timing bug and a Blackhole case a design note had recorded as
   clean; and the shadow census is blind to *self*-coordinate
   reachability, which is why a regression in `90ec8f1` needed a
   different probe to see. Write the census before the fix.

1. **NoC link contention — VALIDATED 2026-08-12.** The term fires, for
   the first time, on both arches: **7,590 link claims / 45 waits** on
   Blackhole and 6,580 / 69 on Wormhole, both `MEANINGFUL —
   CONGESTION MEASURED`, against **zero waits** on every in-tree
   workload before. It reproduces the card's step to **2.1–4.4 %** at
   2/8/16 KiB with **nothing fitted** — the modelled step is
   `bytes / flit_bytes` from two `isa_doc` entries that predate the
   card campaign.
   **Why it had never fired is a harness bug, not a model limit**:
   `perfbench/run.sh` and `run_card_session.sh --sim` exported
   `TT_SIM_TENSIX_COORDS` unconditionally, and setting that variable
   *at all* pins the worker pool and switches off on-demand
   materialisation. Two more behind it: `HAVE_TT_SIM` probed with the
   system python3 before `--sim` put the venv on `PATH` (so both CSVs
   were collected and then reported `DEFERRED`), and `probe_noc_epoch`
   passed two CSVs to a `--measured` taking one. All three fixed.
   **Wormhole's row is prediction, not corroboration** — twice
   Blackhole's step from the 256- vs 512-bit flit, with no Wormhole
   silicon to check it against.
   The card's slopes read +2.55 / +10.98 / +22.47 at **r² 0.36** over
   all eight shared-link counts. They are lines drawn through a *step*
   and are **not coefficients**; the comparison that reproduces is
   step-to-step. Earlier figures of +2.49 / +10.87 / +22.49 are the
   same shape under a different aggregation — neither set is canonical.
   The arbitration order is now cited (`tm_dram_saturation`,
   `tm_memory_for_kernels`, both `vendor_source`, both **confirming**
   the first-come mechanism already implemented), and the stale
   "tt-sim models no link congestion" claim is retired from all eight
   places it survived — including one compiled into `dramratebench`'s
   operator-facing output.
   **What remains unmodelled is buffer back-pressure and virtual
   channels**, ~2 % of the effect, refused as `estimated` in the
   over-charging direction.
   Left open: the shipped `noc-plan-blackhole.csv` is a
   **harvested-card** plan whose tile check verifies existence but not
   *identity*, so it would bite a second harvested card; `--sim` now
   always plans from the simulator's own grid dump. Blackhole `noc1`
   shows 0 claims — nothing in tree exercises NoC 1's registry.

2. **Rung-4** (§4) — **all three buildable legs exist as of
   2026-08-18; the fourth is settled as impossible.** This is the
   v2.0 list's last substantial item, and what is left in it is one
   measured model gap (the divide floor, below) rather than any
   missing instrument. `perfbench/mechbench` plus
   `tt_sim.perf.stall_attribution` partition a core's span by stall
   mechanism on both sides and report `E_total`, `E_int` and the
   compensation ratio behind five refusing gates. The synthetic
   compensating case is the leg's argument in one file: **`E_total`
   2.92 % — inside the envelope limit — against `E_int` 39.33 %, ratio
   13.5x**, and all three of its *per-thread* comparisons pass. The
   compensation is visible only where the Src conditions are
   decomposed, which is exactly what an envelope or per-thread check
   misses.
   **Blocked on a measured question, not a missing instrument.** With
   the profiler readback fixed (§5) the intended cost-model-on regime
   is collectable, and in it **both arms report `srca_clear = 0` and
   `srcb_clear = 0`** (`elw` span 3275, `srca_valid` 2989; `mm` span
   3887, `srca_valid` 3572). The earlier explanation — "the regime
   that could show the reversal cannot be collected" — is spent. This
   is a question about tt-sim's Src-ownership modelling and **must not
   be tuned before card data exists**, or the comparison becomes
   circular. Written up in `perfbench/mechbench/README.md`.
   **Trap for the next person**: `perfbench/mechbench/testdata/sim-*.csv`
   was captured *before* the profiler fix and is silently truncated to
   **BRISC only** — NCRISC and all three TRISCs were dropped. Not
   recaptured, because that moves every pinned constant in
   `stall_attribution_test.py`.
   **AND THE MECHANISM LEG IS WORSE OFF THAN THE ABOVE READS,
   2026-08-17.** Its `partition_closes` gate assumes
   `WAITING_FOR_{NONZERO,NONFULL}_SEM_t` are disjoint sub-counts of
   `THREAD_STALLS_t`. They are not: the vendor's **own** metric 36 is
   a "Stall Cause Overlap Factor", documented identically for *both*
   architectures — a family with a documented overlap factor is not a
   partition. On a Wormhole card the buckets over-account by ~2,900 on
   a ~4,300-cycle span and the gate refuses, both arms, all repeats.
   **Blackhole's passes were never evidence**:
   `TensixPerfCounters.note_stall` increments the total and at most one
   reason bucket in the *same call*, so `sem_empty + sem_full <=
   thread_stalls` holds by construction, and every Blackhole result
   this leg has — the checked-in sim logs and the synthetic card files
   derived from them — inherits that identity. **tt-sim cannot fail
   this gate on either architecture**, so it has never tested hardware;
   Wormhole is simply the first silicon allowed to disagree. The
   refusal now *names* the counters and the excess instead of
   reporting a negative bucket. Widening a tolerance or clamping the
   bucket was rejected: a partition that cannot close is saying the
   model of the hardware is wrong.
   **SETTLED AS IMPOSSIBLE, 2026-08-17 — the mechanism leg cannot be
   built from the counters this hardware exposes, and this is now a
   fact rather than an open item.** The remaining move was to subtract
   the overlap the vendor documents rather than assume none. **Metric
   36 is not a counter**: the `INSTRN_THREAD` select maps
   (`hw_counters.h:116-175` WH, `:158-219` BH, 59 each) carry no
   overlap entry and no other bank does; it is computed on the host in
   Python (`tools/tracy/perf_counter_analysis.py:1332-1350`) and the
   LLK doc files it under `### Composite`
   (`performance_counters.md:785`). Its nine *inputs* are free — same
   bank, same window, both arches — but a scalar over nine reasons
   cannot say which pair of buckets overlaps, cannot separate "two
   reasons on one stalled cycle" from "a reason on an unstalled cycle"
   (opposite corrections), and readmits the seven declined unit-busy
   counters. **And the card falsifies the factor's documented
   meaning**: simultaneity cannot lift a *single* reason above the
   total, yet `WAITING_FOR_NONZERO_SEM_2` = 3,561 against
   `THREAD_STALLS_2` = 270 and `WAITING_FOR_{MATH,SFPU}_IDLE_1` = 38
   against 36. Using the factor as a divisor also self-destructs — its
   numerator contains the bucket, so the "corrected" value tends to
   `THREAD_STALLS_t` (251 for the measured 3,561; 260 for a
   hypothetical 7,122). `stall_reason_overlap` now reports
   `max_reason_excess` and `metric36_as_correction` computes the fit,
   so both are printed numbers, not remembered arguments.
   **And the gate can now fail in simulation.**
   `TensixPerfCounters.note_wait_condition` counts a latched wait
   condition without counting a stall — what `SEMWAIT.md` says the
   hardware does and what `note_stall` structurally could not express
   — and the suite drives a real counter bank through its own MMIO
   readback into an overlapping window, asserting the gate refuses it
   and passes a disjoint one built the same way. **The front end calls it as
   of 2026-08-18.** `WaitGate._tick_unheld_latched_wait` re-evaluates a
   latched wait on the cycles nothing is held by it, forgets it when met
   (per `SEMWAIT.md`) and counts an unmet one as a *reason* and not a
   stall, so the gate's refusal is now reachable from a simulated tile
   and not only a hand-built bank — a real `TensixTile` read back
   through its own MMIO gives `WAITING_FOR_NONZERO_SEM_1` 60 against
   `THREAD_STALLS_1` 19. Keeping a live latch awake costs **zero extra
   ticks**, measured on 9 replay guards run both ways: tiles do sleep
   there, never with a live latched wait.
   **What is still open is the MAGNITUDE, not the hook, and it is still
   not to be closed by scaling.** tt-sim's threads reach their blocked
   instruction almost immediately, so the un-held window barely exists
   and `mechbench` moves **+6 cycles** on both arms against the card's
   13x excess, with `THREAD_STALLS_2` and the total span unchanged and
   no pinned constant moved. That gap is a front-end issue-timing
   question.
   **A pre-existing correctness bug was found here and FIXED
   2026-08-18** (`waitgate_multi_semaphore_test.py`, 14 cases, 12 of
   which fail without the fix): `check_for_wait_condition_met`'s
   semaphore branch had
   a misindented `return True` **inside** its `for sem in sem_checks:`
   loop, so a `SEMWAIT` selecting several semaphores only ever tests the
   first and the thread is released early — silently, no fault, the NoC
   1 failure mode again. Reproduced: semaphores [0, 1] with C0, sem0 = 5
   and sem1 = 0 returns True where `SEMWAIT.md` requires all selected
   conditions to be met simultaneously. The neighbouring
   `_note_latched_wait` has the same loop with its `return` correctly at
   the `for` level, which is what makes this a typo rather than a
   design. Harmless for the single-semaphore masks every in-tree kernel
   uses; `semaphore_mask` is 8 bits, so the reachable case was real.
   The fix **delegates to `_latched_semaphore_reason`** rather than
   dedenting the `return`: that walk already existed and its docstring
   already said it was there "so they cannot drift", and the bug *was*
   that drift — a silently-diverged third reader. There is now one walk.
   `SEMWAIT.md`'s table settles the quantifier: C0/C1 say to keep
   waiting if **any** selected semaphore is zero / at max, the
   contrapositive of the summary's "until **all** of the selected
   conditions are simultaneously met", so release needs every selected
   (semaphore, condition) pair satisfied. No pinned number moved,
   which is the evidence the reachable case was untested rather than
   tested-and-passing.
   **The NoC-bound leg is built, 2026-08-16** — `perfbench/nocevbench`
   plus `tt_sim.perf.noc_events`, behind six refusing gates. Two of
   three legs now exist.
   **The RV-bound leg is built too, 2026-08-18 — all three now exist**
   (`perfbench/retirebench` + `tt_sim.perf.retire_attribution`, eight
   refusing gates). Its instrument is Zicsr and the Blackhole CSR file
   (§6), landed the same day: twelve zones on one baby RISC-V, each
   bracketed by an `mcycle` and a `minstret` read, both sides emitting
   the identical artefact so one parser reads each. The synthetic
   compensating case is the leg's argument in one file: `E_total`
   **0.91 %** either way — identical to two decimals, inside every
   envelope threshold in this repo — against `E_int` **1.15 %**
   agreeing and **45.48 %** compensating, ratio **49.91x**, and the
   compensating file passes *every gate* and fails only the criterion.
   **Its buckets are weaker than the other two legs' and it says so.**
   Every bucket's *magnitude* is hardware-measured (an `mcycle` delta on
   the core whose cycles it counts) and every *retired count* is too,
   but every bucket's *mechanism label* is **structural** — there is no
   per-mechanism counter for a baby RISC-V, so each zone is built to be
   dominated by one mechanism. What makes the label checkable is
   `minstret`: `retire_census_matches` demands exact per-zone equality
   and refuses otherwise. The CPI table is **not** an independent second
   check, unlike `nocevbench`'s latency table — once the census is
   equalised, CPI is the partition over a per-zone constant.
   **Blackhole-only by construction**, and it refuses in three places
   (card pre-flight, host program, `arch_supported`) rather than falling
   back to the elapsed-only envelope check rung 4 exists to distrust:
   the string `csr` appears zero times in the whole `WormholeB0/` doc
   tree, so a CSR instruction on a Wormhole core raises.
   **FIRST SILICON RESULT, 2026-08-18: it FAILS, and the failure is the
   useful kind.** A Blackhole p150 session, three repeats,
   `perfbench/card-sessions/2026-08-18-bh-retirebench/`. `E_total`
   **15.91 / 15.51 / 15.91 %** against a 10 % limit.
   **It is `E_total` that fails, not `E_int`** — the interior number
   *passes* at 15.91 % against its 25 % bar. That is not a detail: a
   strict floor model under-charges monotonically, so every bucket errs
   one way, `E_int` is identically equal to `E_total`, and **such a
   model can only ever fail rung 4 through the tighter envelope bar,
   never through the interior one the rung was designed around.**
   Rung 4 was built to catch compensation; a floor's failure mode is
   not compensation.
   **The compensation ratio is 1.00x** — `E_int` equals `E_total` to the
   decimal, so *nothing is hiding behind anything*: every zone errs in
   the same direction (tt-sim under-charges) and the triangle inequality
   holds with equality. For a model built deliberately as a floor that
   is the shape wanted, and it is the answer to the question the rung
   exists to ask.
   **The whole failure is the divide, and it is a deliberate modelling
   choice arriving where it was aimed.** The two divide zones are
   **13.43 %** of the 15.92 %; everything else totals **2.49 %**, which
   passes both thresholds with room to spare.
   `unit_costs.yaml`'s `divide_general: { cycles: 6, max: 33, bound:
   range }` (`isa_doc`, `wh_riscv#integer-unit`) is charged at its low
   end per the working rule that the model is a floor; the doc says
   "between six and 33 cycles ... dependent upon the magnitude of the
   dividend" and the card has now *measured* that dependence —
   **14.04 cycles per divide at a 12-bit dividend, 33.10 at a 29-bit
   one**, against 6 modelled.
   **THE GAP IS PERMANENT AND THAT IS NOW PROVEN, 2026-08-18.** A
   dividend-magnitude term is **unlicensable at every rank of the
   ladder**. Both ISA doc trees were searched exhaustively — 602 files,
   `quotient` appears **zero** times and `dividend` exactly four, being
   the same two sentences once per architecture. No radix, no iteration
   rule, no bits-per-cycle, no formula, no divider in any pipeline
   diagram; Blackhole's text is character-for-character Wormhole's.
   Every vendor tree yielded exactly one hit, `tt_metal/hw/inc/internal/
   mod_div_lib.h:129` — "This takes six to 33 cycles on WH/BH" — which
   is **strictly weaker** than the ISA doc: same band, no mechanism,
   and it drops the magnitude clause. So the term could only enter as
   `estimated`, which is forbidden. No sweep was built, because no
   quantity of card data can change a provenance question.
   **The obvious curve is now excluded rather than merely unpinned.**
   `cycles = bits + k` needs one `k` and gets 2.018 and 4.043; the
   affine law through both points is `0.589 + 1.119*bits`, which reads
   **1.71 at one bit** (under the documented floor of 6) and **36.40 at
   32 bits** (over the documented cap of 33). The simplest family is
   refuted by the band it would have to live inside. Both divide zones
   are **bit-identical across all three repeats** while every other zone
   moves, so the divider is an exactly reproducible function of its
   operands — what is missing is the function, not the precision.
   **Not to be "fixed" by charging 33**: that would over-charge every
   small divide and break the floor property everywhere else. This
   belongs beside the mechanism leg as a **permanent documented
   limitation of a floor model over a `bound: range` entry**, not as a
   deferred task.
   **Worth raising upstream**: three instruments put silicon at or
   fractionally *past* the published maximum — 33.10 here, `riscvbench`
   at 33.001/33.004, a Wormhole card at 33.03. The band's top is
   saturated, not bounding.
   **The prediction registered beforehand was half right, and the miss
   is recorded because that is the only thing that makes registering
   one worth doing.** It said `div_large` would be the single bad zone
   at ~7 % of the span: measured **7.85 %**, essentially exact. It did
   **not** predict `div_small`, which is also wrong, at 5.55 %. The
   error is in both divide zones, not one.
   **The instrument proved itself in the data**, which is what the
   retired census is for: across three repeats every per-zone `minstret`
   is *identical* — 6208, 6398, 3106, 6396, 567, 244, 795, 3621, 1259,
   6114, 6113, window 40915 — while cycle counts move by tenths of a per
   cent. `unattributed` is 120/121/123 cycles on a ~71,700-cycle window.
   Four zones corroborate `riscvbench`'s Blackhole silicon without being
   fitted to it (`mul_dep` 1.954 modelled against 1.960 measured,
   `load_dep` 7.728 against 7.925).
   **The Wormhole refusal fired on real Wormhole silicon**, same day, so
   it is a tested refusal and not an asserted one: the pre-flight asks
   the device which part it is and declines before building a kernel or
   launching anything.
   **VALIDATED ON BOTH ARCHES AGAINST SILICON, 2026-08-17.** Six
   comparisons on a Wormhole card all pass — arms A/B/C x 256/4096 B,
   errors **0.2-9.3 %** against a 25 % bar, every partition gate green
   — and three on Blackhole. Arm C also gave the NoC translation port
   its first hardware verdict at no extra card cost: card and
   *translated* simulator agree on destination `(2, 2)` where the
   untranslated simulator says `(7, 9)`, and the card reports
   `self_noc=18,18` / `peer_noc=19,19` for itself.
   **The pivotal question answered yes with no bridge work**:
   `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` runs against tt-sim on
   *both* arches today and writes `.logs/noc_trace_dev0_ID0.json`. NoC
   events go into the *same* per-RISC L1 profiler vector as the zone
   markers, so `Device.settle_profiler_flush` (§5) already covers them
   — every readback hazard was pre-paid.
   **What the artefact is forced a change to the leg's design, and it
   is not a compromise.** Every transaction event is stamped **at
   issue** (`kernel_profiler.hpp:178`; ~40 call sites in
   `dataflow_api.h`, macro always ahead of the issue); the *only*
   completion timestamp anywhere is a barrier's
   `READ/WRITE_BARRIER_END`. So **there is no hardware per-packet
   flight time** and this item's own phrasing — "agreement with
   `noc_flight_cycles` plus queueing" — presupposed one. Two
   corrections follow: the comparison is card-JSON against **the same
   artefact** emitted by tt-sim, one parser, no translation step; and
   `noc_flight_cycles` is reported *simulator-side only, as a
   diagnostic, never a gate*. "Plus queueing" would also
   **double-count** — `noc_flight_cycles` is `arrival - issue_cycle`
   and `NUI.transmit` stamps `issue_cycle` *after* `send_to` has
   charged flight + injection-port queueing + link wait +
   serialisation.
   The partition (`prologue`, `issue`, `read_wait`, `write_wait`,
   `other_wait`, `local`) telescopes to the zone span by construction,
   so `partition_closes` is really a monotonicity test on a 44-bit
   wall clock read as two register accesses. The synthetic
   compensating case is the leg's argument in one file: `E_total`
   **4.78 %**, `E_int` **48.33 %**, ratio **10.12x**, and **every
   per-class latency passes at 4.8 %** — an envelope check *and* a
   latency check both pass it; only the decomposition sees it.
   Stated limits, not deferred work: `read_wait`/`write_wait` and the
   per-class latencies are **two readings of the same intervals**, not
   independent detectors; addresses and VCs are unreachable in the
   file-dumping path; no distance arm (needs a remote L1 target).
   **The leg's silicon verification is done; nekbone's own number is
   not.** The card session above validated the *instrument* — its
   synthetic arms, on both parts. The bottleneck report's **79.8 % NoC
   split on nekbone** is a claim about a different workload, so what
   is left is running this decomposition *on nekbone* rather than
   building or trusting anything further. The blocker changed shape
   rather than clearing.
   **And its first real external capture broke it in two places,
   2026-08-19 — both in the reader, not the model.** The single-window
   gate was counting the profiler's own flush zone as a launch and
   refused all 32 streams of a legitimate 16-core Blackhole capture,
   and `prologue` was being read as a setup cost when it is
   `ZONE_START`-to-first-NoC-event and swallows any wait that emits no
   event. Both are written up in §4 point 4, including why the split
   the consumer asked for is refused rather than built. Neither moved a
   modelled cycle; both moved what the leg *says*.
3. **Wormhole access — FOUR OF FIVE GATED ITEMS RETIRED, 2026-08-17.**
   It was a booking; it became standing access, which changed the shape
   from "one irreplaceable session" to a programme: a de-risking pass
   first, then measurement. `docs/plans/wormhole-session.md` carries the
   runsheet, and five sessions are banked under
   `perfbench/card-sessions/2026-08-17-*`.
   **Retired.** *The DRAM sustained rate*: agrees with
   `wh_dram#performance` to **-0.2 / -1.5 / -0.9 %** at 1 / 12 / 48
   readers, with its own control moving x5.86 across channels while the
   same readers on one channel moved x0.995 — the endpoint shape
   tt-sim's `DramChannels` term asserts. *The store-coalescing and
   multiply pairs*: coalesce/spread **1.0010** on Wormhole against
   Blackhole's **5.2x** — a real architectural difference, now measured
   on both parts rather than inferred from one; divide lands at 33.03
   cycles/instr, exactly `divide_general`'s documented `max`.
   *The NoC-contention row*: **CONGESTION MEASURED**, saturating —
   +0.01 cycles/link at 64 B and 512 B, then +20.1 / +78.6 / +158.0 at
   2 / 8 / 16 KiB, with the first shared link costing ~500 cycles at
   16 KiB and the second and third ~18 and ~0.5. *The `nocreadbench`
   half*: see below, and it is the one that turned over.
   **The 44-vs-25 disagreement was ours, not the part's.** The first
   session measured a marginal **44.0 cycles/transaction** against
   tt-metal's shipped 25.00 — flat to 0.25 % across a 32x range in
   burst length and 1.0 % across hops — and it looked like the only
   genuine model-versus-silicon disagreement the programme had found.
   A second session with a **shorter issue loop** (the `set_state` /
   `with_state` pair the vendor's own stateful rows use) settled it:
   **45.03 -> 28.96, the loop bought 16.07**, landing inside the band
   the dataset occupies. Verdict **H-LOOP**, registered before the run:
   44 was this program's instruction stream, and Wormhole has no
   per-read floor. Every arm proved which loop it ran *from the
   returned payload* rather than from its flag, and the control
   reproduced the earlier session to 2.3 % on all three intervals.
   **AND BLACKHOLE SETTLED IT EXACTLY, same day.** With an
   instruction-level model now checked against a Wormhole card, the
   Blackhole prediction could be **absolute** rather than a drop, and it
   was printed by the runner before the measurement: **47.00 / 29.00,
   saving 18.00**. The card returned **47.00 / 29.00 / 18.00** — zero
   difference on all three, both rounds agreeing to 0.02 cycles across
   every burst interval, every arm confirmed from its payload. That is
   the strongest validation the cost model has: an instruction-level
   prediction landing on a real part it had never been checked against
   for this measurement.
   **It also refutes the Blackhole floor outright.** The shipped
   dataset says the shorter loop buys **1.0 cycle of 35.0** there —
   that one figure is the entire basis for "Blackhole reads have a
   floor the issue loop cannot get under", worth ~15 cycles a read, and
   for the roadmap's own "initiator outstanding-read-request credit
   limit" hypothesis. The card says it buys **18.00 of 47.00**. So
   *both* architectures return H-LOOP, neither part has a per-read
   floor, and the shipped rows describe a different kernel — 12 cycles
   adrift on the stateless arm and a factor of 18 on the saving. That
   is why comparing our absolutes against theirs could never have
   settled anything, which cost most of a day to learn.
   **A "missing constant" I reported here was my own error, and the
   correction is the lesson.** I first read tt-sim as under-predicting
   both arms by ~7 and ~6 cycles and called it a missing per-transaction
   term. It was not: the four checked-in `*-sim*.csv` references are
   **cost-model-off** runs, and a cost-model-off marginal is an
   *instruction count* by design and by test
   (`noc_issue_loop_test.py:180`). With `TT_SIM_COST_MODEL=1` the arms
   read **44.00 / 29.00** against the card's 45.03 / 28.96 — residuals
   **-1.03 (-2.3 %)** and **+0.04 (+0.1 %)**, opposite in sign, so not
   a shared constant at all. The 44 is fully accounted: 38 instructions
   at 1 cycle plus the **6-cycle load-use interlock** on the
   `noc_cmd_buf_ready` poll, both already `isa_doc` in
   `unit_costs.yaml`; verified at instruction level, 38 distinct PCs
   212 times each with one 6.00-cycle stall per iteration, and 23 PCs
   for stateful with the same stall. The arms differ in 15
   instructions and *share* the poll, which is exactly why the saving
   was right and the absolutes were not. **Reference CSVs must state
   their cost-model configuration**; comparing an instruction count to
   a card is a category error that looks like a finding.
   **Not retired, and the reason is worse than a missing session** —
   see item 2. Rung 4's two-arch claim **cannot be made, and that is now
   settled rather than pending**: the mechanism leg is a documented
   impossibility on both parts, and the RV-bound leg is Blackhole-only
   by construction because Wormhole documents no CSRs at all. Only the
   NoC leg is genuinely two-arch, and it is validated on both. No
   Wormhole session can change any of that, so this item is closed as
   *answered*, not left open.
   **Two things the programme learned that no simulator could have
   told us.** A **board reset belongs between sessions**: `mechbench`
   failed all six runs with `mismatches=4052` immediately after a
   congestion session and passed cleanly after `tt-smi -r 0` with a
   byte-identical configuration. It corrupts *data*, not launches —
   no abort, no diagnostic — so a session that skips it produces a
   plausible result. And **four card runners had never touched
   hardware**: all launch through `detail::LaunchProgram`, which needs
   slow dispatch, while a card defaults to fast, so every one aborted
   `rc=134` before doing any work. Both are invisible in simulation by
   construction — tt-sim supports no other dispatch flow and has no
   residual device state — which is exactly why the de-risking pass
   went first.

4. **Upstream example coverage — DONE 2026-08-13.**
   `driver/tests/upstream_sweep.py` runs 21 of tt-metal's own
   `programming_examples` on both arches with **no grid environment
   variable at all**, recording an all-green expectation that fails in
   **both** directions. **42/42 PASS, no regressions** against the
   2026-08-03 sweep — so a day of deep change (materialisation, the
   contention term firing, BH DRAM cycles, ConfigUnit/MatrixUnit
   residency, `STALLWAIT`, the host-stop path) broke nothing a real
   program reaches. New coverage too: the grid-sized programs now run
   at the **real** default grid (72 WH / 130 BH) rather than the old
   4x5 = 20-worker sub-block, and `matmul_single_core` moved
   TIMEOUT -> PASS. Two tiers: fast (~7.5 min) for every change, full
   (~70 min) pre-release. Coverage is stated honestly by `--list`: 15
   programs value-checked by themselves, 1 by the gate, **5
   completion-only** because upstream ships no self-check.
5. **Energy, at ranking level — TWO ARCHITECTURES 2026-08-17, and the
   headline is smaller than it looked.** All thirteen gates pass on a
   six-cycle Blackhole p150 session (**LOO Spearman 0.867**) and on a
   six-cycle Wormhole n300 one (**0.900**), with all four within-arm
   pairs ordering correctly on both and all four designed terms
   identified. Sessions at
   `perfbench/card-sessions/2026-08-13-energybench{,-2}/` and
   `2026-08-17-wh-energybench/`.
   **Both numbers now report a null model beside them, and it changes
   the reading.** Taking per-launch energy as proportional to the
   simulator's *cycle count* — no coefficients, no fit, nothing from
   the card — scores **0.867 on Blackhole and 0.800 on Wormhole**. So
   the four-term energy fit is worth **one rank swap in nine on
   Wormhole and nothing at all on Blackhole**. The reason is
   structural: the fit target is board power, but the reported ranking
   is `(P − P_floor)/rate`, and the power span is small against the
   floor (2.9 W over 30.1 W) while the launch rate varies 15×, so the
   measured energy ordering is largely an ordering by *how long the
   kernel took*. `target_triviality` cannot see this — it asks the
   question in power space, and Wormhole passes it at R² = 0.032 — so
   `spearman_null` is now computed, rendered next to the claim, written
   into the JSON and the coefficient record, and covered by five tests.
   It is **reported, not gated**: a refusal needs a threshold and there
   is none to justify, since a model that ties the null on ordering can
   still be well ahead on ratios, which is exactly what happened.
   **Where the fit does earn its keep is the ratios, and only on
   Wormhole**: median |log ratio error| ×1.22, worst ×1.65, against the
   null's ×2.15 / ×8.43 — a factor of five on the worst case. Blackhole
   manages ×1.98 / ×4.48 against ×2.36 / ×8.37, a much thinner margin.
   That inverts the Blackhole write-up, which had recorded the ratios
   as the part the data did *not* support.
   **A stronger result falls out of the same arithmetic and is not
   about energy at all**: simulated cycles order these nine workloads
   against the card's measured launch rate at **Spearman 1.0000 on both
   architectures**, with no fitting, no coefficients and no card-side
   calibration. The cycle model's wall-time ordering is on firmer
   ground than anything the energy fit claims.
   **The four fitted coefficients landed within 1.7× of each other on
   two different parts** — `noc_bytes_total` 1.03e-10 / 7.27e-11,
   `matrix_arith_cycles` 2.97e-09 / 3.90e-09, `sfpu_busy_cycles`
   9.07e-10 / 8.81e-10, `instr_retired` 4.15e-10 / 2.39e-10 J per unit
   (BH / WH) — over floors that differ as the parts do, 67.02 W against
   30.10 W. Two sessions three days apart, two independent simulator
   runs, two separate NNLS fits each free to clamp any term to zero (and
   both did clamp `c_launch`). This is **corroboration, not
   validation**: the fits share a design, a term set and an arms table,
   so a fault in any of those is common to both and invisible here.
   **The Wormhole session is the better instrument** — ~107 samples per
   slot against ~31, a 15× launch-rate span against 5.3×, and a design
   conditioning at 240 against 1.15e3 — and it is what the thermal
   gate's temperature fallback was built for, since that box publishes
   no `tt_therm_trip_count`. Its activity vectors cost 2 h 39 m of
   simulator time for 22.2M cycles at ~2,330 cycles/s.
   **What neither session supports is the within-arm 4× ratio in board
   power.** On Wormhole, the better of the two, only `mm` clears the
   noise floor (+1.076 W = 2.46 floors); `sfpu` +0.285, `rv` +0.227 and
   `noc` +0.158 W are all under it. The energy ordering does get all
   four pairs right, but that is the launch rate doing the work and it
   must not be quoted as the board resolving 4× the work as 4× the
   power.
   **On Blackhole the headline went DOWN as the model got right, and
   0.867 is the one that stands.** It read 0.900 against a *contaminated* matrix
   column, then 0.950 with that column *missing entirely* (a
   three-term model in which the `mm` arm had no characteristic term at
   all), and 0.867 once the corrected `matrix_arith_cycles` column
   existed. The first two were artefacts of a wrong model and an
   incomplete one; better numbers from a worse model is the trap this
   entry exists to record. The LOO/in-sample gap widened with the
   correct model too (0.867 against 0.950), so more of the apparent
   agreement is fitting than the earlier figures suggested.
   **What that session does not support**: its ratios. Median
   `|log ratio error|` 0.68 (x1.98), worst 1.50 (x4.48) — worse than
   the contaminated model's x1.77, and only a thin margin over its own
   null. Wormhole is where the ratios hold up. **Whether that is the
   architecture or the instrument is not established** — the sessions
   differ in both, and Blackhole's sampled a third as often over a
   launch-rate span a third as wide, which is the cheaper hypothesis
   and is testable by re-running Blackhole at the Wormhole session's
   settings. `c_launch` remains clamped at the NNLS boundary, the
   predicted collinearity with the ~14,600 firmware instructions every
   launch pays; `idle-0`'s measured energy is **negative** and dropping
   it moved the earlier fit 0.900 -> 0.857, so it stays the least
   trustworthy point in the set.
   **The SFPU term was freed by separating matrix arithmetic from dest
   bookkeeping**, and the diagnosis needed three wrong turns first. The
   `sfpu` arm dispatches 41 Matrix Unit ops per `add_int_tile`
   iteration — 32 `INCRWC` from sfpi's `dst_reg++`, 9 `SETRWC` from the
   LLK's per-face dest walk, and **zero arithmetic ops** — so
   `matrix_busy_cycles` absorbed the vector unit's energy and NNLS
   clamped `sfpu_busy_cycles` to 0. Killed along the way: the README's
   `copy_tile` explanation (it is called twice, *outside* the loop);
   an instr/dispatch split (made it worse); and a test of mine that
   removed the matrix column entirely and still saw 0 — which wrongly
   exonerated it. The term needed the *clean* column **present**, not
   absent. Counters now publish `bookkeeping_cycles` as a subset of
   `busy_cycles`, and `matrix_arith_cycles` is the difference — an
   **appended** term, so a stale CSV loudly drops it rather than
   quietly refitting the old column.
   **Getting there cost five analysis bugs, every one found by the
   data**, and the sequence is the lesson: a clean session is what
   exposes a modelling fault, and the 2026-08-13 `samples=0` disaster
   hid all of them. (i) The idle **baseline sat at 800 MHz against arms
   at 1350** — structural, not drift, since an idle board clocks down
   and `tt-smi` 6.2.0 offers no pinning; the floor is now *fitted* from
   the arm rows, because a busy-state floor is not measurable on a DVFS
   board. (ii) The old target `(P - P_baseline)/rate` was **99.95%
   reproducible from the launch rate alone** and no gate saw it; there
   is now a `target_triviality` gate (this session: 0.23 against a 0.95
   cap). (iii) The **noise floor included the baseline label**, whose
   2.4 W DVFS swing inflated it 0.441 -> 0.838 W and made `control` —
   the drift detector — twice as permissive; a constructed session with
   real control drift passed every gate and reported LOO Spearman
   1.0000 under the old arithmetic. (iv) The **term budget was
   `n_workloads - 3`**, reaching six deep into eleven correlated
   counters (51% of such subsets exceed the conditioning cap); it is
   now the arms table's one-term-per-arm set, fixed a priori. (v) The
   matrix/bookkeeping conflation above.
   **A negative result worth not repeating**: adding a third scale does
   *not* help identifiability. Interpolated sizes add rows, not
   directions — at 13 workloads all 66 ten-term subsets land at
   >= 2.9e16. Another *arm* would help; another scale never does.
   **A process lesson, paid for**: the regenerated activity CSV was
   written over the live fixture path, destroying the vectors behind
   the 0.900 figure permanently (untracked, no git copy). And a test
   pinned that large data artefact as a *stale* fixture, so it broke
   the moment the file was correctly regenerated; it is now synthetic.
   Fitted coefficients remain quarantined outside the provenance ladder
   by three independently asserted barriers, and nothing from this
   session may be quoted into a cost table.


**A caution this list earned.** Three of its original premises were
wrong, all the same way — read from the tree's prose rather than its
code. Contention "unmodelled" (wired since 2026-08-05, the stale claim
surviving in seven files); C7/C14 "inert" (a sentence about the
*occupancy* column answering a question about *latency*); and
scale-is-the-problem (wall clock is ~flat in worker count; the cost was
materialising **unused** workers). Each was caught by an agent told to
verify the premise first-hand. **Prose here drifts from code; check the
code.**

## What the first external consumer found

**The v2.0 list above is closed** — every item delivered or proven
impossible, 2026-08-18 — and what has landed since came from none of
the lists in this file. Seven things landed in the week after, and
**four arrived as reports from the compiler team using tt-sim in
anger** rather than from anything planned here: the multicast corner
order (below), the `prologue` misreading and the single-window gate
(§4 point 4), and the request for a version marker that became
`tt_sim/behaviour.py`. Each report is dated in the source it touched.

**Three of them were tt-sim returning a confident, plausible, wrong
answer** — the failure mode that consumer ranks above cycle accuracy,
in their own words as recorded in the v3 working note: *"the failures
that cost us most were the ones where a simulator returned a confident,
plausible, wrong answer. A tool that told us 'I do not model this'
would have been worth more than one that passed."* The three: a
multicast suite running green against a simulator that delivered the
packet to **nobody**; a `prologue` bucket reading **91.89 %** of a span
as setup when it was a `cb_wait_front` on the compute pipeline; and a
gate reporting **five program executions** where there was one kernel
launch and four profiler flushes. **Not one of the three was a
cycle-count error.** That is the through-line, and it is the strongest
evidence this file holds about what the next release should be about.
`docs/plans/v3.md` is a working note, not a commitment, and its two
strands — **observability** and **loudness** — were named before any of
this landed; what follows is what happened, not what was planned.

1. **Multicast corner order — a correctness defect, closed 2026-08-19**
   (`c541127`). The ISA docs require a broadcast rectangle's corners
   **swapped** on a NoC addressed in translated coordinates, because
   translation does not change the direction of data flow
   (`WormholeB0/NoC/Coordinates.md`, "Coordinate Translation";
   `BlackholeA0/NoC/Coordinates.md` says the same), and `MemoryMap.md`
   resolves a reversed rectangle as a **torus wrap** rather than as the
   empty set — so the packet lands on tiles the kernel never counted in
   `num_dests`, the ACK count cannot match, and
   `noc_async_write_barrier` spins for ever. That is the silicon
   symptom the compiler team reported. tt-metal does the swap in
   `Device::get_noc_multicast_encoding`
   (`tt_metal/impl/device/device.cpp:692`, the comment saying why at
   `:696` in the 0.74 tree); the kernel-side `get_noc_multicast_addr`
   passes its arguments straight through, which is why the swap is so
   easy to omit.
   tt-sim enumerated the corners literally — `range(start, end + 1)` —
   so there were **three silent cases**. A rectangle mis-ordered for
   its NoC reached *everyone* and produced exactly the right answer
   (green here, hang on the card). A reversed one reached *nobody*, so
   the barrier retired having sent nothing. And, the case that matters
   most, a **correctly-encoded translated NoC 1 multicast — which is
   required to descend — also reached nobody**: a live mismodel of
   correct code, not merely a failure to catch a bug.
   **Two changes, deliberately separate.** `rectangle_destinations`
   takes the closed interval whichever order the corners arrive in and
   **has no off switch**, because it is what makes correct code work;
   `check_corner_order` raises `MulticastOrderError` and does
   (`TT_SIM_DISABLE_MULTICAST_ORDER_CHECKS`, following
   `TT_SIM_DISABLE_ALIGNMENT_CHECKS`). A test asserts that disabling
   the guard does not undo the enumeration fix. The guard is
   deliberately **stricter than tt-metal's own watcher**, which skips
   the ordering check for Tensix-to-Tensix multicasts because the wrap
   is *legal* there: legal is not intended, and tt-sim cannot deliver a
   wrapped span at all, so such a multicast would be mismodelled
   silently today. If one is ever wanted, the answer is to model the
   wrap, not to lower this to a warning.
2. **`NoCAlignmentError` now names the transfer** (`72ab5e1`,
   `2f21f3b`). Two addresses and the rule they broke is enough to know
   a program has a bug and not enough to know where. The message now
   carries the transfer's size and transaction id, the address spans,
   the issuing core and its PC, the kernel function and source line
   (via `tt_sim.trace.elfdisc` + `DwarfIndex`), and the **circular
   buffer the L1 end lands in together with its page size** — the
   number that says whether a shard was split below tile granularity.
   **All of it is recovered at raise time, not recorded per transfer.**
   `NUI.write` calls `RequestInitiator.initiate` synchronously, so the
   issuer is a few frames up the Python stack when the check fires;
   alignment checking is on by default, so anything recorded per
   transfer would be paid for by every transfer in every run and read
   only by the ones that fail. `tt_sim/network/attribution.py` is
   imported *inside* the `except` clause, so a run that never faults
   never imports the describer, and the exception is enriched in place
   so type, identity and traceback are unchanged.
   **Three limits are stated rather than papered over.** Only
   byte-verified ELFs name a function — a confidently wrong name on a
   fatal error is worse than none. The page size comes from the
   firmware's `cb_interface` array in the issuing core's local memory
   (located by symbol, never by scanning, and decoded only when the
   decode is self-consistent), so a tt-metal layout change costs the
   line and not its truthfulness. And the buffer's **name** is not
   recoverable at all: tt-metal's allocator lives on the host and is
   never described to the simulator, so the address range is reported
   instead.
3. **Named behaviour markers an outside suite can assert on**
   (`b38b8ee`, `tt_sim/behaviour.py`). The compiler team asked for a
   version marker to pin. A version number says which build you have,
   never whether that build has the fix, and a consumer who pins one
   has encoded our release history into their test suite instead of
   what they actually depend on. What is published instead is
   **behaviour** — `require("noc1-multicast-corner-order")` raises
   `UnsupportedBehaviour`, `supports()` is the non-raising form, and
   `python3 -m tt_sim.behaviour` lists or checks from a shell. Against
   a tt-sim older than the module the import fails, which is the same
   outcome at the same moment. Three guarantees are published today —
   `noc1-multicast-corner-order`, `noc-transfer-alignment` and
   `riscv-ebreak-halts` — each naming what a run against this build
   will do and what it used to do instead.
   **The registry resists decay structurally**, which is the part worth
   keeping. Forward: every entry names the test that pins it and
   `behaviour_test.py` imports that module and looks the function up,
   so no guarantee outlives its check. Backward: the same test
   **AST-parses every non-test module under `tt_sim/`** for exception
   classes and requires each one to be either registered or listed in
   `_NOT_A_GUARANTEE` *with a reason* — so adding a loudness guard
   turns the suite red until somebody has decided, in writing, whether
   an outside consumer would want to assert on it. The declined list is
   itself a record: `UnitWedgedError` is declined because the deadlock
   watchdog's false-positive rate has not been argued, and
   `UnmodelledTileRegisterError` / `UnmodelledCSRError` because the set
   they cover moves with every release, so neither is a stable promise.
4. **A fixed cycle budget in a test is a latent cost-model failure**
   (`4283579`). `multicast_order_test.py` shipped in `c541127` with a
   64-cycle settle budget before reading the destinations. That is
   enough with the cost model **off** and not with it **on**: a charged
   NoC flight delivered **1 of 3** destinations by cycle 64 and all 3
   by 128. The cost-model gate is what caught it — it runs
   `pytest tt_sim -q` under `TT_SIM_COST_MODEL=1`
   (`driver/tests/cost_model_gate.py`) precisely so the model-on
   configuration cannot rot. The budget is now 1024, 8x the observed
   requirement and long enough that the negative tests' empty result
   means "never" rather than "not yet". The general form is worth
   remembering: **any test asserting on device state after N cycles is
   pinned to the cost model's current charges**, and it will fail on
   the day one of them changes rather than on the day it was written.

The other three landed elsewhere in this file: DRAM page-to-bank
distribution, and the host-DMA rate refused alongside it, are in §1;
the two `noc_events` corrections are in §4 point 4.

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

Endpoint occupancy closed 2026-08-09, Blackhole's channel read rate
2026-08-12 and its write *occupancy* 2026-08-17; the first two are in
`docs/plans/cost-model.md`. What survives here is what would otherwise
be re-litigated:

- **The device's own re-issue interval stays `unknown` and uncharged.**
  Charging `access_latency` as occupancy would assert 0.32 B/cycle
  against the 24 published on the same page.
- **A tt-sim DRAM tile fronts two GDDR6 channels**, so one queue per
  *tile* over-charges. Each physical channel has its own watermark.
- **Blackhole's DRAM *write* rate was uncharged on ONE axis too many —
  fixed 2026-08-17.** The channel figure is spent on two: a latency
  excess and the channel's occupancy. The refusal is an argument about
  the first only — the measured write row slopes at 59.36 B/cycle,
  within 2.5 % of the same file's four L1 rows, so a write's
  *completion* does not contain the channel drain. It says nothing
  about what those bytes cost the request behind them, and **cannot**:
  every DRAM row in `tm_noc_latencies` is one transaction per barrier,
  so occupancy is invisible to all of them in *both* directions. A new
  nested `write_occupancy: 47.0805` (`vendor_source_derived`) charges
  the second axis — the read rate's own `(8192−4096)/(665−578)`, to
  the digit, no second measurement — resting on one extra step stated
  separately so it can be argued with: a GDDR6 data bus moves bytes at
  the same rate each way, which is the **vendor's** claim, from the
  same DRAM page sentence that licenses charging Wormhole's 24 both
  ways. Only the *symmetry* crosses architectures; the level is
  Blackhole's own, and `costs_test` still asserts Wormhole's 24 cannot
  reach it. **The latency axis stays refused, with numbers**: charging
  it would take the rung-2 BH DRAM-write residuals from +52…+67 to
  +51…+21 (≈ −3.8 cycles/KiB), turning a mildly under-sloped row into
  a worse over-sloped one. The entry's *second* old reason — that it
  would deepen `KNOWN_OVER_CHARGED` — is void, since the earlier
  `access_latency.write` work emptied that set; it is recorded rather
  than deleted. Effect: a BH DRAM write stops being 36 % cheaper than
  a read. Modelled read/write at 4 tiles on one endpoint goes
  **1.360 → 1.014** against a card's 0.993 (2.1 % out, from 37 %);
  driven to saturation the two are equal by construction, verified
  directly — 16×4096 B on one channel finishes at cycle 696 in either
  direction on Blackhole and 1368 on Wormhole. Every one of the 42
  committed replay guards is **byte-identical** in total cycles on both
  arches, and rung 2 is unmoved to the cycle, because the term is inert
  for one-transaction rows by construction; the BH claims and waits do
  move underneath (`pad_multi_core` 1024→3072 claims, 48→240 waits),
  just never on a critical path at the pump's 100-cycle granularity.
  `dram.bandwidth` — the GB/s spec block — is still `unknown`, still
  needs a document, and gates nothing.
- **Wormhole's own write deficit is now recorded and declined.** Its
  modelled write sits 3.8 % above its card's, which the base note
  states as a discrepancy rather than acting on — there is no vendor
  row that resolves it.
- **The BH DRAM-write over-charge is gone.** It was 8 negative
  residuals (−12 to −28) pinned in `KNOWN_OVER_CHARGED`; splitting
  `access_latency` by request action closed them and **that set is now
  empty**, asserted rather than described — no row on either arch
  over-charges today, and one appearing is a change somebody has to
  make deliberately. The split still rests on one arch's data.
- The BH `l1_local_cycles = 88` rung-1 anomaly (54 cycles
  unexplained) — rung 1 cannot fully pass on BH until explained.
- **Page-to-bank *distribution* is modelled and was untested; bank
  *timing* is neither, and they are different questions.** The
  distribution question is the compiler team's, and answering it
  sharpened it: their symptom is not bank *conflicts* but page
  *distribution*, which is a tt-metal software decision rather than a
  hardware property. tt-metal
  spreads an interleaved buffer's pages round-robin over the DRAM banks
  (12 on Wormhole — 6 channels x 2 `dram_view`s — and 8 on Blackhole);
  the **host** computes the landing bank in
  `WriteToDeviceInterleavedContiguous` and the **kernel** recomputes it
  in `InterleavedAddrGen<true>::get_noc_addr` from the `NUM_DRAM_BANKS`
  JIT define and the tables the host wrote into L1, and *nothing
  cross-checks the two*. Since `bank(byte) = (offset / page_size) %
  num_banks`, a wrong page size moves every byte to a different bank
  while leaving every address legal — silently wrong data, not a fault.
  tt-sim reproduces that: it holds each bank as separate storage at its
  own NoC coordinate and **executes the kernel's real address
  arithmetic as RV32**, so a page computed into the wrong bank reads
  the wrong bytes here for the same reason it does on silicon. Give
  `examples/banks`' kernel a page size the host did not allocate with
  and Wormhole tt-sim returns `errors=3072 of 6144` — no crash, no
  `TT_FATAL`, clean device close — with the corruption starting at
  **page 12, the first bank wrap**, which is where the page size first
  enters the address at all.
  **This was a coverage gap, not a modelling gap, and it is now
  closed** (2026-08-19, `9b11f2f`). Every prior example allocated a
  *single-page* DRAM buffer and reached it with
  `get_noc_addr_from_bank_id<true>(0, ...)` — bank 0, hardcoded — so a
  simulator that flattened DRAM to one store would have replayed the
  whole suite green. `examples/banks` (24 pages x 1 KiB, walked with
  `InterleavedAddrGen`, both arches) and
  `driver/blackhole/server/banks_replay_test.py` close it; the guard
  reads the destination back **bank by bank**, at each bank's own
  coordinate and within-bank offset, because reading it as one
  contiguous range could not tell the two layouts apart.
  **The host-side half is refused, and stays refused.** The
  1.89-vs-5.81 GB/s figure quoted at us is a **PCIe/host-DMA rate**,
  which tt-sim cannot see by construction: there is no PCIe tile
  (`tt_sim/bridge/cores.py` stubs it to zeros), no host-DMA term in
  `tt_sim/perf/`, and a host `WRITE`/`READ` off the wire is applied to
  device memory immediately at zero cycles. A tt-sim run is not
  evidence about that ceiling in *either* direction. The device-side
  sibling — traffic collapsed onto one bank contending where spread
  traffic does not — needs nothing new, since each channel already
  carries an occupancy (`DramChannels` in `tt_sim/device/tiles.py`).
  Both halves are written up for consumers in
  `docs/cost-model-caveats-for-consumers.md`.
  **Bank-internal timing has no route to provenance**: bank conflicts,
  row hit/miss, precharge and refresh windows are absent from the
  public ISA docs for both parts (Blackhole has no DRAM tile page at
  all), so any such term would have to be invented or measured — and
  measurement is `corroboration` here, never provenance. Treat DRAM as
  a flat per-channel pipe at the published rate. Long-term, and not by
  measurement.
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

**Restated 2026-08-13: an *interior* match by mechanism and by zone.**
"Instruction for instruction" is **retired as unreachable**, not
deferred. It was written down as three independently sufficient
grounds; **one of them was false and is corrected below**, so the
retirement now rests on the remaining three — which is still more than
it needs, and the replacement ground is better sourced than the one it
replaces:

- **No instrument — CORRECTED 2026-08-18, this ground was wrong on
  Blackhole.** It read "the baby cores are RV32IM with no Zicsr — no
  `mcycle`, no `minstret`", with no architecture qualifier, and §6 has
  recorded the opposite the whole time: `BabyRISCV/CSRs.md` documents
  `0xb00 mcycle` and `0xb02 minstret` on Blackhole, and tt-sim has
  implemented them since 2026-08-18. **True on Wormhole only** (the
  string `csr` appears zero times in its doc tree). What survives is
  the rest: tt-metal 0.74, UMD and the public ISA docs contain no PC
  sampler and no instruction-trace buffer; the debug daisychain is
  documented "at least five cycles stale" and every consumer of it is
  commented out.
- **No fidelity from the CSRs either, and this is the ground that
  should have been written.** `cfg0`'s `DisCsrSync` (bit 10) is
  documented: *while clear*, once a `csrrw`/`csrrs`/`csrrc`/`csrrwi`/
  `csrrsi`/`csrrci` instruction leaves the frontend, the next
  instruction does not leave until the previous one has **retired**. It
  is clear at reset **and clear in tt-metal's own init** (the observed
  `cfg0 = 0x60008` sets only `DisLowCash`, `DisTriscCache` and
  `StMergeTimer`), so a CSR read is a retirement barrier on a real
  part. Bracketing single instructions with `mcycle` reads therefore
  destroys the very overlap it would be measuring — the same objection
  as the marker's 28 cycles, but sourced rather than measured, and it
  does not go away with a cheaper counter. **The counters remain right
  for a *window*** — two reads around a region of millions of
  instructions dilate nothing and serialise only at the boundaries,
  which is exactly what rung 4's RV-bound leg needs and is why the
  instrument was built.
- **No budget.** `PROFILER_L1_OPTIONAL_MARKER_COUNT = 250` → **125
  scopes per RISC per launch**, which tt-metal's own doc states and its
  own `test_full_buffer` asserts (125 recovered from a kernel asking
  150). `quick_push()` is compiled out on TRISC, so on the compute
  cores 125 is **unflushable**.
- **No fidelity.** A marker costs **≥ 28 device cycles** — measured
  from `perfbench/card-sessions/2026-08-10/paper-1` (28 on NCRISC, 36
  on TRISC), with tt-llk's CI asserting 30 on Blackhole. A baby core
  retires ~1 instruction/cycle, so per-instruction instrumentation
  dilates **≥ 28x** and destroys the overlap it was meant to measure.

**But the rung's purpose is reachable, by a better instrument.** Rung 4
exists because the interior can be wrong in compensating directions
while the envelope agrees. That is a question about **attribution**,
not ordering — and Tensix **hardware performance counters** answer it
at cycle resolution, per thread, per mechanism, with **no
instrumentation inside the measured window** (start/stop on TRISC1
wrap the kernel; readout on BRISC afterwards).
`TT_METAL_PROFILE_PERF_COUNTERS`'s Blackhole INSTRN bank is tt-sim's
own `STALL_REASONS` vocabulary in hardware: `THREAD_STALLS_{0,1,2}`,
`WAITING_FOR_{SRCA,SRCB}_{CLEAR,VALID}`,
`WAITING_FOR_{THCON,UNPACK,PACK,MATH,MOVE,SFPU}_IDLE_n`,
`WAITING_FOR_{NONZERO,NONFULL}_SEM_n`, `THREAD_INSTRUCTIONS_n`.

**What rung 4 now requires.** Three programs — RV-bound, Tensix-bound,
NoC-bound — run unmodified on silicon and tt-sim under the same
tt-metal build. **Every criterion is per core**: on the 2026-08-10 part
the whole physical column x=11 keeps a wall-clock epoch 1.5e13 cycles
from the rest, so no cross-core span is admissible.

1. **Zone decomposition**, ≤ 60 zones per RISC (half the hard 125, so a
   silent drop is impossible) and each ≥ 1000 cycles (so the ~56-cycle
   two-marker cost is ≤ 6 %).
2. **The interior criterion, stated so compensation cannot pass it.**
   With mechanisms `m` partitioning the span, including an explicit
   *unattributed* bucket on both sides so the denominators match:
   `E_total = |Σc_sim − Σc_hw| / Σc_hw` and
   **`E_int = Σ|c_m,sim − c_m,hw| / Σc_hw`**. Require **`E_int ≤ 25 %`**
   *and* `E_total ≤ 10 %`. The triangle inequality gives
   `E_total ≤ E_int` always, so **the ratio `E_int/E_total` is the
   compensation, measured**. This is what a passing total cannot fake.
3. **Mechanism attribution** from the perf counters above — the only
   check that has ever touched the five wired Tensix backends, which
   rungs 1 and 2 validate *not at all*.
4. **The NoC split, checked directly.** Built 2026-08-16 —
   `perfbench/nocevbench` + `tt_sim.perf.noc_events`; see the v2.0
   list's item 2 for what it established and what it corrected.
   `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1` needs **no kernel change**
   and brackets every `noc_async_read_barrier` with timestamped
   `READ_BARRIER_START/END`.
   **Correction, from reading tt-metal rather than assuming: there is
   no per-transaction completion timestamp**, so the "agreement with
   `noc_flight_cycles` plus queueing" once written here has no
   hardware counterpart and is *not* what the leg does. The comparison
   is tt-sim's own `noc_trace_*.json` against a card's, decomposed by
   mechanism at ± 25 %; `noc_flight_cycles` is a simulator-side
   diagnostic only, and already contains the queueing.
   **Validated against silicon on both arches 2026-08-17** — six
   Wormhole comparisons at 0.2–9.3 % against a 25 % bar, three on
   Blackhole. The bottleneck report's **79.8 % NoC split on nekbone**
   stays unverified all the same: that is this decomposition run on
   *nekbone*, which has not happened.
   **Two corrections landed 2026-08-19, both from a consumer's own
   capture, and neither moved a modelled cycle.**
   *(a) `gate_single_window` was reading the instrument, not the
   workload.* It counted every `ZONE_START` as a program execution,
   including `PROFILER-NOC-QUICK-PUSH` — the device profiler's own
   flush zone, nested inside the kernel zone and emitted every time the
   per-RISC L1 marker buffer fills, because `recordNocEvent` calls
   `flush_to_dram_if_full` *before* it records. A 16-core Blackhole
   `gemm_256_check` pair had all **32 streams refused** as
   `single_window` when all 32 in fact carry exactly one `*-KERNEL`
   pair, kernel `ZONE_START` first and `ZONE_END` last, with 0–4
   strictly nested pushes. **The zone is in the artefact by an upstream
   bug**: `convertNocTracePacketsToJson` filters the string
   `"PROFILER-NOC-QUICK-SEND"`
   (`tt_metal/impl/profiler/profiler.cpp:780`), which is the **only**
   occurrence of that string in the whole tt-metal tree — the zone was
   renamed to `PROFILER-NOC-QUICK-PUSH`
   (`tt_metal/tools/profiler/kernel_profiler.hpp:409`) and the filter
   never followed, so it leaks into every capture past roughly 150
   transactions on a RISC. (Both verified by grep against the 0.74
   checkout; worth raising upstream.) Windows are now identified **by
   name**, and a flush is **charged to its own `profiler` bucket**
   rather than absorbed into `issue`: it is instrumentation the capture
   added, and folding it in would charge the model for the difference
   between how tt-sim and a card execute the profiler's own DRAM write.
   It belongs in the partition because **both sides can fill it**,
   which is the test for whether a bucket belongs at all. The bucket is
   an *upper* bound — the `ZONE_END` marker is written before the flush
   is issued — and reads 161–166 cycles in all 112 pushes on the
   reference capture.
   *(b) `prologue` is caveated, not split.* It is defined mechanically
   as `ZONE_START` to the first NoC event, so **anything the core did
   in that interval lands in it, including a blocking wait**. A
   consumer's 4-core `gemm_128` writer kernel read `prologue` at
   **91.89 %** of a 6,932-cycle span with no semaphore anywhere in the
   build: it was `cb_wait_front(2, 1)` waiting on the compute pipeline.
   The asymmetry is structural — `cb_wait_front` / `cb_reserve_back`
   carry only a watcher `WAYPOINT` (four characters into a mailbox
   slot, no timestamp, no path into any artefact) and `NocEventType`
   has no CB member, whereas `noc_semaphore_wait`
   (`dataflow_api.h:1929`) does `RECORD_NOC_EVENT(SEMAPHORE_WAIT)`,
   which is exactly why `other_wait` exists and `prologue` cannot be
   split the same way. `TT_METAL_PROFILER_SUM=1` instruments the
   *compute* side only and is dropped by the JSON converter, so no
   tt-metal flag recovers the split for these streams (checked against
   0.74 rather than assumed).
   **tt-sim could fill the bucket from its own internals and refused.**
   It does know more here — `tt_sim.pe.rv.spin` recognises a baby core
   polling L1 — but a `pre_first_event_wait` bucket only the simulator
   can populate makes the card's share **zero by construction** and
   charges the model, through `E_int`, for an artefact of
   instrumentation. The consumer asked for precisely that split; the
   refusal is the same test the `profiler` bucket passes, applied the
   other way. Read `prologue` as *"time before the first NoC event,
   cause unknown"* — an upper bound on setup and nothing more — and
   never rank or sum it across cores: a reader at the head of a
   pipeline has genuine setup there, a writer at the tail has its
   producer's whole latency, and those are not the same quantity.

Thresholds: `E_total ≤ 10 %` is what nekbone already achieved, kept as
the control against envelope regression. `E_int ≤ 25 %` is 2.5x that —
a decomposition has strictly more ways to be wrong, the model is a
floor with a deliberate unattributed remainder, and it is still tighter
than the one known open level error (`dramratebench`, 26–35 %). Silicon
noise is not the limit: the two 2026-08-10 sessions reproduce device
cycles to 0.1 %.

**The tt-sim blocker closed 2026-08-13**: `RISCV_DEBUG_REG_PERF_CNT_*`
is modelled for all five banks, sourcing the `INSTRN_THREAD` counters
from the quantities tt-sim already tracks — per-thread stalls,
semaphore empty/full, Src ownership, dispatches. Verified end to end on
the wire with `TT_METAL_PROFILE_PERF_COUNTERS=32`. Three counter
families were **declined rather than forced**, most importantly
`WAITING_FOR_{UNIT}_IDLE_{n}`: the vendor tech report contradicts
itself across two sections and the later one is right — those count
cycles a *unit was busy*, not cycles a thread was stalled by it, and
produce ">100 %" values. Worth raising upstream.
`TensixTileControl` now raises on an unmodelled **status** register
read instead of answering a plausible zero; `TT_SIM_PERMISSIVE_TILE_CTRL=1`
restores the old behaviour. Note what depended on the old silent zero
and legitimately still does: `_blackhole_reset_pc` reads
`RESET_PC_OVERRIDE`, and **every Blackhole core boots because an
unwritten override reads zero** — which is why the fix is a three-way
register table, not a blanket raise.

**Blackhole-only bonus leg.** `minstret`/`mcycle` are ISA-documented on
Blackhole (`BabyRISCV/CSRs.md`) and would allow a per-instruction check
with no in-kernel marker. Wormhole has **no CSR path at all**, so this
is an extra leg Blackhole can clear, **not** the definition of the
rung — a two-tier rung 4 with no common bar would be worse than one
coarser bar that works everywhere. It also instruments the RV front end
rather than the coprocessor, which is where most of the subject is.
Prerequisite: `pe/rv/` implements no CSRs (§6).

Golden artefacts check in under `perfbench/card-sessions/` as
`corroboration` — the counter semantics come from a vendor tech report
and RTL, not the ISA docs, which is fine for corroboration and
**disqualifying for provenance**.

**What clearing it licenses**: *"a performance estimator whose cycle
attribution has been checked against silicon's own hardware stall
counters, mechanism by mechanism, at ± 25 % of span"*. Not
"cycle-accurate"; no ordering or per-instruction claim; nothing about
backends the three programs do not exercise.


## 5. Tracing & observability follow-ups

**The device-profiler readback starvation is fixed, 2026-08-13.** It had
cost the project twice. tt-metal's `brisc.cc:575` sets `RUN_MSG_DONE`
*inside* the `DeviceZoneScopedMainN("BRISC-FW")` block whose destructor
calls `finish_profiler()`, and the host's go-poll and its control-vector
read are **adjacent wire messages** — so BRISC got one `cycles_per_poll`
(100) of tail for a ~1400-cycle publish. The markers were always in L1
(`DEVICE_BUFFER_END_INDEX` non-zero); what had not run was the
`HOST_BUFFER_END_INDEX` stores and the NoC pushes, and
`readRiscProfilerResults` early-returns on those, giving a header-only
file with no zone markers either.
The bridge now fingerprints the profiler control vector on its *shape*
(the L1 offset is release-specific and arrives over the wire), arms on
`go=GO`, and at the readback runs cycles until `PROFILER_DONE` **and the
DRAM pushes have landed** — the landing phase was found by measurement,
since stopping at `PROFILER_DONE` alone dropped TRISC_1 and TRISC_2
(`profiler_noc_async_flush_posted_write()` waits on *sent*, not landed).
Capped at 200 000 cycles, then a warning and permanent give-up on that
worker. Cost 500–1400 cycles per profiled launch; **nothing is armed
with the profiler off** — `grep -c "size=128 "` over all 60+ traces is 0.
`TT_SIM_CYCLES_PER_POLL=5000` is no longer needed anywhere.
**What this retro-invalidates**: runs that appeared to work were also
truncated. A cost-model-off capture read `HOST_END_BR=368` with NCRISC
and all three TRISCs at **0** — one iteration of the publish loop, 63
BRISC rows and nothing else. Any profiler CSV captured before this fix
should be assumed BRISC-only.

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
  exists to clean up after it, but only for servers it *tagged*
  (`--run-tag`), deliberately, because a broad `pkill` kills concurrent
  agents' runs and turns a vanished server into a plausible wrong
  measurement rather than a loud failure. **Observed 2026-08-13**: five
  untagged orphans, one 20 hours old, holding 0.4 GB between them. An
  untagged server started by hand or by a tool that does not set the
  tag is currently unreapable by anything but a manual `kill`.
- ARC tile: nothing modelled; `arc_msg` returns 0.

**Blackhole RV extensions** (all loud guards, none reached by any
in-tree kernel). `BabyRISCV/README.md` states the set: *"the full
RV32IM instruction set … plus all of Zicsr / Zaamo / Zba / Zbb, plus
some (but not all) of Zicntr / F / Zfh. RISCV T2 additionally
implements some (but not all) of V."* Corrected 2026-08-13:

- **`mret` is RISCV B and NC only** — the TRISCs cannot execute it.
  `PIC.md` also records a hardware bug: interrupt handlers may *read*
  CSRs but cannot write them.
- **The BF16 mode is a *custom* CSR bit, not a Zfh one** — `cfg0`
  (`0x7c0`) bits 30 `EnBFloat` / 31 `EnBFloatRTNE`.
- **The L1 cache tag search accelerator uses no CSRs** — it is
  configured through Tensix backend `Config.L1_CACHE_TAG_SEARCH_ACCEL_*`
  and triggered implicitly by a RISCV-B load. (`0x7c4`/`0x7c5` look
  like its config and are documented as pure scratch — a red herring.)
- **The counter CSRs exist on Blackhole**: `BabyRISCV/CSRs.md` gives
  `0xb00 mcycle`, `0xb02 minstret`, their `h` halves, `mhpmcounter3/4`
  (only 3 and 4, with `mhpmevent` encodings unpublished), and the
  user-mode shadows `0xc00 cycle` / `0xc02 instret` — which are
  **writable**, and `mcountinhibit` cannot stop either counter. `time`
  is documented absent.

**Zicsr and the CSR file landed 2026-08-18** — `pe/rv/isa/zicsr_isa.py`,
the documented Blackhole registers behind the six Zicsr instructions.
The prerequisite for rung 4's Blackhole-only per-instruction leg (§4)
is met; the leg itself is not built.
**It closed a latent defect as well as adding a feature.** SYSTEM
funct3 1-7 was *claimed* by `RV_I_ISA.handle_i_misc`, which returned
True for every funct3 — so a CSR read executed nothing and left `rd`
holding its previous value. Silent, not UndefinedBehavior. Real
firmware reaches it: the pipestall replay shows **8 `cfg0`
read-modify-writes per core**, previously no-ops, now leaving
`cfg0 = 0x60008` (`DisLowCash` + `DisTriscCache`, set by tt-metal's
init, with the reset `StMergeTimer = 16` intact).
`mcycle` reads **the tile clock `RISCV_DEBUG_REG_WALL_CLOCK_*` already
samples** rather than a private counter that could drift, and a core
with no clock bound refuses instead of inventing one; `minstret`
counts on the retire path only, with `spin.py` adding back a parked
span's retires exactly so firmware-idle parking cannot deflate it.
**Refused, deliberately**: `tt_cfg_qstatus`/`bstatus` (the queue ->
bitmask mapping is unbuilt, and 0 would read as "coprocessor idle");
the TRISC `tt_cfg_sstatus*` (scratch only on B/NC, elsewhere an
unnamed overlay stream); `mhpmcounter3/4` once `mhpmevent3/4` is
non-zero (encodings unpublished); unknown addresses. `DisCsrSync`
serialisation is charged **nothing** — the doc gives no cycle number
and `estimated` is forbidden.

**Wormhole has no CSR path whatever** — the string `csr` appears zero
times in the whole `WormholeB0/` doc tree, so "RV32IM-only, complete
and correct" is confirmed. Its cycle source is MMIO
(`RISCV_DEBUG_REG_WALL_CLOCK_L/H`).

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
