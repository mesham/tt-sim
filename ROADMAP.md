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
priority list below is shaped by the first lane: the cost model's
tables and calibration are well ahead of the *mechanisms* that would
let them influence a simulated cycle count, and the simulator's wall
clock is what bounds the workloads the model can be exercised on.

---

## Priority list

### Tier 1 — load-bearing, start here

1. **Bound the Tensix instruction FIFOs and stall the `.ttinsn` write
   when full.** The single mechanism gating the entire Tensix half of
   the cost model, and a correctness fix at the same time (a wedged
   backend unit currently cannot stop a launch reporting done).
   Includes promoting `[UNIT WEDGED]` from warning to raise.
2. **Make a full grid affordable.** Wall clock is linear in
   *materialised* tiles regardless of what the program launches; the
   fix is firmware-loop recognition or a cheaper live Tensix tile.
   This is the lever for the only remaining upstream-example failures
   and for wide-grid perf-model workloads.
3. **Convert the Wormhole replay guards to poll-until-DONE.** Cheap,
   mechanical, and the standing blocker on ever defaulting
   `TT_SIM_COST_MODEL` on; de-risks every timing change in this list,
   so arguably do it first of the three.

### Tier 2 — high value, small-to-medium effort

4. **The three silicon-backed RV cost fixes**: dependent-chain loads,
   divide, Blackhole multiply latency scoreboard.
5. **Unpacker and mover occupancy** — the two Tensix units still
   unwired from the cost tables. Sequence after item 1, otherwise the
   new tables remain unobservable like the existing six.
6. **Cost the NIU register block** — the busiest MMIO load in the
   tree; where dataflow kernels' time actually lands.
7. **`MemoryMap` last-hit-range cache** — the one remaining named
   wall-clock optimisation, 3–7 % across workloads.
8. **Freeze the cheap guards**: `noc_tile_transfer`,
   `optests/dramtop`, `vecadd_sharding` on Blackhole, Wormhole
   `matmulidx`/`matmulblock` offline guards.
9. **Smarter `TT_SIM_CYCLES_PER_POLL` default** — the fixed 100-cycle
   pump after every wire message dominates at scale.

### Tier 3 — medium-term, mostly sequenced behind Tier 1

10. **Issue-loop model** — worth more rung-2 dataset coverage than any
    remaining congestion work.
11. **DRAM residue** — endpoint occupancy first (pure model shape, no
    new data needed); the rest is measurement-blocked.
12. **Next silicon session on the Blackhole box** — a bundle of cheap
    probes once a card is in hand, including the first-ever Wormhole
    measurements.
13. **Tensix issue latency & wait-gates**, then mover / PC-buffer
    timing point fixes.
14. **Numba on the pump's event heap, and the `nogil` threading
    revival** — the ranked JIT target now that pump Phase 4 exists.
15. **Rung-4 calibration against silicon traces** — the eventual bar
    for claiming anything stronger than "estimator".

### Tier 4 — opportunistic / housekeeping

16. **Tracing & observability follow-ups** — the perf-budget decision,
    `chip_id`, and the long tail.
17. **Functional backlog** (pick up when a kernel demands it) — Tensix
    backend gaps, NoC registers/atomics, device/tile infrastructure,
    Blackhole ISA extensions.
18. **Architectural clarity & quick wins** — module boundaries,
    docstring audit, diagrams, ISA index; shellcheck,
    `MEM_BOOT_CODE_BASE`.
19. **Parked decisions & re-measurements** — Wormhole reset fan-out,
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

## 1. Tensix front-end back-pressure

`TensixFrontend.push_mop_instruction` is an unbounded list append and
a `.ttinsn` store returns the same tick, so nothing ever back-pressures
the issuing baby core. Measured from both sides:

- **Correctness**: a permanently stalled backend unit cannot wedge the
  issuing core, so tt-sim runs to completion where Blackhole silicon
  deadlocks (the `UNPACR_NOP` acquire-without-release case — the setup
  acquires a Src bank and never releases it, wedging the next launch on
  the card until `tt-smi -r`). Mitigations already landed: the
  `[UNIT STALL]` hint and `[UNIT WEDGED]` terminal check in
  `tt_sim/device/deadlock.py`, both default-on.
- **Perf model**: six of six wired Tensix units moved **zero**
  simulated cycles on every workload — `tensixbench` reads a forced
  1.000 cyc/instr for every probe of every unit at every format
  against tt-sim. No cost-table entry above 1 cycle can matter until
  the FIFO is bounded.

Silicon gives the targets (`docs/plans/tensix-cost-benchmark.md`,
`docs/plans/riscv-front-end-benchmark.md`): ThCon at 3.0 cycles
back-pressures the core; shared 1-IPC units scale exactly 3.0× across
threads; the queue absorbs ~31–32 instructions at one issuing thread —
a **lower bound**, so do not hardcode a depth yet (run the longer burst
sweep first). What is licensed today: a 1-cycle occupancy on the
`.ttinsn` push itself, and back-pressure from a unit at its documented
occupancy.

Fold in: promote `[UNIT WEDGED]` to a raise — it cannot fire on a
correct kernel (surveyed across 41 workloads and every `pipestall`
configuration). A timing perturbation of this size needs its own
cost-model gate run, which is why item 3 should land first.

- *Test:* issue the dvalid setup twice with no intervening clear and
  assert the second blocks (`setdvalid_srcrow_test.py` has the
  single-issue half); the ThCon burst probes in `perfbench/` become
  self-checkable against tt-sim.

## 2. Full-grid affordability

A materialised worker is a live worker: tt-metal's grid-wide init
releases BRISC on every worker in the declared grid, which then spins
in the firmware go-message poll until teardown. Wall clock is therefore
linear in *materialised* tiles regardless of what the program launches
— measured: `vecadd_sharding` runs in 14 s with 4 workers materialised
and 255 s with 80, though it only launches on 4.

Explicitly **not** a pump problem: Phase 1 dormancy bought nothing here
(it only retires DRAM/eth tiles — dormant fraction of never-launched
workers was 2.9–14.5 %) and Phase 3 would not help either. The two
named fixes:

- **Firmware-loop recognition** — a BRISC spinning on a go-message
  poll is architecturally idle even though it retires instructions;
  recognise the loop and let the tile stride.
- **A cheaper live Tensix tile** — cut the per-cycle cost of a tile
  that is awake but doing nothing useful.

This is the lever for the two remaining throughput-bound upstream
examples (`matmul_single_core` at 8,000 tile-matmuls,
`matmul_multicore_reuse_mcast` at 32,768 — the only non-environment
failures left in the sweep) and for kernel-scale perf-model workloads
generally. Related knob while this is open: run with the smallest grid
the program accepts and materialise exactly those workers.

## 3. Wormhole replay guards → poll-until-DONE

The Wormhole example replay guards replay the host's *poll count*
verbatim, so any legitimate timing change — every cost-model instalment
— fails them spuriously (each failure is a READ that resolves within
~2,000 further cycles). The 21+ Blackhole value guards already have the
right shape: pump until the go-message reaches `RUN_MSG_DONE`, then
check values. Convert the Wormhole guards to match.

This is the standing blocker on ever shipping `TT_SIM_COST_MODEL` on by
default, and it removes friction from items 1, 4, 5, 6 and 13, each of
which perturbs timing. Do it first.

## 4. Silicon-backed RV cost fixes

Three under-charges found by `perfbench/riscvbench` on Blackhole
silicon, all in `tt_sim/pe/rv/cost.py` territory, each small and
well-scoped, each its own change + gate run:

- **Dependent-chain loads**: silicon measures 8.098 cycles
  (`rv_load_chase`); the model applies the 2-cycle d-cache hit.
  `rv_load_indep` (1.742) corroborates latency 8 via the docs' own
  throughput formula.
- **Divide**: silicon measures 33.001 on a 29-significant-bit dividend;
  the model charges the documented band's floor of 6. In-tree kernels
  divide 9–12-bit values 0–2 times per launch, so the *policy* (charge
  the floor) was reviewed and kept — what's open is modelling the
  magnitude-dependent curve.
- **Blackhole multiply latency**: occupancy 1, latency 2; the model
  charges occupancy with no scoreboard entry for the result, so
  dependent chains read 1.000 where silicon reads 1.985.

Together they pose the policy question of whether "charge every bound
at its low end" survives contact with silicon; answer it deliberately
in the cost-model doc when landing these.

## 5. Unpacker and mover occupancy

The two Tensix backend units deliberately unwired from the cost tables
(`UNWIRED_UNITS` in `tt_sim/perf/costs_test.py`; the third entry, TDMA,
is all-1-cycle and stays out to keep the allow-list honest):

- **Unpacker** — the genuinely non-constant entry: a ≥2-cycle address
  phase plus a data phase of transfer-bytes ÷ configured throttle
  (16/32/64 B/cyc from `THCON_SEC[n].Throttle_mode`) against an
  80 B/cyc joint ceiling. Wants more than the flat lookup the other
  units use.
- **Mover** — the table's `1` is issue cost only; the transfer duration
  is bandwidth-derived and lives in `tt_sim/perf/unit_costs.yaml`
  under `mover`, which nothing consumes yet.

Sequence after item 1 so the charges are observable. Unblocks the §H
observability fields gated on them (packer back-pressure, unpacker
idle cycles).

## 6. NIU register block cost

Loads from the NIU register block (`0xFFB20000`/`0xFFB30000`) are
uncosted (`RV_UNNAMED_REGIONS`), yet this is the busiest MMIO load in
the tree — every `noc_async_*_barrier` polls it — and where dataflow
kernels' time actually lands. Blocked on provenance: the ISA docs'
"≥ 7" covers the NoC *overlay* at `0xFFB40000`, a different block. If
no published number surfaces, this becomes a probe in the next silicon
session (item 12).

## 7. `MemoryMap` last-hit cache

`MemoryMap` interval lookup / `MemorySpace.read`: cache the last-hit
range — **not** a JIT (the polymorphic `mem_mapable` dispatch is
Numba-hostile). Worth 3–7 % of wall clock across workloads. Measurement
note carried from profiling: the "most-called function" premise was
wrong (`Register.read` and `RegisterFile.get` are called more), but the
cost-class argument stands.

## 8. Freeze the cheap guards

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

## 9. `TT_SIM_CYCLES_PER_POLL` default

The bridge pumps `cycles_per_poll` (default 100) cycles after *every*
wire message, including the pure host DMA of input/output buffers — at
80 workers a readback ran for an hour that finishes in seconds at
`=10`. Fix the default rather than documenting around it: skip the
pump on host-DMA messages, or adapt it to grid size.

## 10. Issue-loop model

The residual on every rung-2 prediction is one constant unmodelled
issuing-core path (intercepts 77–94 cycles across four independent
series). An issue-loop term is worth more of the 8,140-point tt-metal
NoC dataset than congestion was — 24 sole-cause entries vs 2 — so it
is the ranked next step for widening rung-2 coverage.

## 11. DRAM residue

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

## 12. Next silicon session (Blackhole box)

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
- If item 6 stays provenance-blocked: an NIU-register-load probe.

## 13. Tensix issue latency & wait-gates; mover / PC-buffer timing

Wait-gate stalls on srcA/srcB availability and the documented per-op
issue cadence, once the front end (item 1) can express them. ttsim is
**not** an oracle here (also not cycle-accurate); cycle assertions must
come from the ISA docs. Mover transfer timing and PC-buffer write
delays are point fixes once item 5's framework exists.

## 14. Numba and the threading revival

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

## 15. Rung-4 calibration

The bar for claiming anything stronger than "first-order estimator":
match a captured silicon cycle trace within X %. Needs golden traces —
one per major unit (RV-only, Tensix-only, NoC-heavy) — checked in
under `driver/wormhole/server/traces/`. Until then, no in-tree cycle
count has ever been compared to silicon; say "performance estimator",
not "cycle-accurate".

## 16. Tracing & observability follow-ups

- **Perf-budget decision**: the ~30 % tracing overhead target is not
  met on RV-bound workloads (counters/Perfetto ~2×, JSONL ~4×) —
  re-target the budget or make `EventBus.publish` cheaper at the call
  site.
- Fields gated on items 1/5: NoC `vc` + VC occupancy, packer
  back-pressure, unpacker idle cycles, L1 bank conflicts, FPU/SFPU
  stalled cycles (needs a refused issue — refusals are ~0 until item 1
  lands).
- `chip_id` hard-coded to 0 (per-tile `core_y`/`core_x` already fan
  out) — multi-chip identity.
- Long tail: inline-print migration, `DwarfIndex` function-name
  attribution, kernel-ELF auto-discovery, invariant mining (tt-metal
  #28562), `hypothesis` property tests (needs a kernel generator),
  host-polling determinism (co-design with pump time-skipping),
  L1/DRAM snapshots in state dumps, `SpikeCommitlogWriter` mem-access
  decoration, Jupyter template, NoC heatmap example, Pyright cleanup.

## 17. Functional backlog

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
  Naturally folds into item 1.
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

## 18. Architectural clarity & quick wins

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

## 19. Parked decisions & re-measurements

- **Wormhole NCRISC/TRISC reset fan-out**: the launch-message
  `enables` path is wired on Blackhole only; decide whether Wormhole
  should use it (needs the WH L1 offset for
  `kernel_config_msg_t.enables`). *Test:* a disabled-BRISC-bit variant
  asserting the core never leaves reset.
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
