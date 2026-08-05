# Roadmap

Inventory of known TODOs and gaps. Picked off opportunistically, not in
strict order. Items marked **★** are load-bearing for a near-term goal
(multi-Tensix tt-metal programs, broader kernel coverage, perf); the rest
are correctness/completeness backlog.

Each item carries a *Test:* sub-bullet naming an example that would
exercise it — either a new entry in the shared, arch-agnostic
`examples/` tree (the `one/`–`nine/`, `loopback/` series, which runs on
both Wormhole and Blackhole), a differential op test under `optests/`,
a script under `driver/simple/`, or an existing tt-metal
`programming_examples/` build that should be brought up once the gap
closes. Items marked *no example needed* are pure housekeeping,
documentation, or perf modelling.

Note on paths: the numbered examples used to live under
`driver/wormhole/<n>/` and now live at the repo top level in
`examples/<n>/`, arch-agnostic and selected by `TT_METAL_SIMULATOR`.
Older *Test:* bullets below that name a `driver/wormhole/<n>/` should be
read as `examples/<n>/`.

References:
- ISA docs: <https://github.com/tenstorrent/tt-isa-documentation>
  (`WormholeB0/`, `BlackholeA0/`).
- Project notes: `CLAUDE.md`, `driver/wormhole/server/README.md`,
  `driver/blackhole/README.md`.
- Landed plans: [`docs/plans/blackhole-support.md`](docs/plans/blackhole-support.md)
  (the multi-arch port + the ttsim differential work),
  [`docs/plans/event-driven-pump.md`](docs/plans/event-driven-pump.md) (§I's
  substrate), [`driver/wormhole/docs/profiling.md`](driver/wormhole/docs/profiling.md)
  (the measured perf baseline and what has since been optimised).
- External handoff:
  [`docs/handoff-matmul-block.md`](docs/handoff-matmul-block.md) — why the
  reported multi-tile GEMM failure is a kernel bug rather than a tt-sim
  one, and what `matmul_block`'s `kt_dim` actually means.

---

## Positioning (post-ttsim)

Tenstorrent released [ttsim](https://github.com/tenstorrent/ttsim) as
the open-source, bit-exact functional simulator for Wormhole /
Blackhole / Quasar — *"the official golden reference implementation of
the Tenstorrent ISA contract"*. It occupies the same
`TT_METAL_SIMULATOR` slot tt-sim's wire bridge does, in C++, faster,
vendor-maintained, aimed at safety-critical pre-silicon validation.

That changes the goal. tt-sim no longer competes for bit-exact
correctness, broad Tensix-instruction coverage, multi-arch coverage,
or the "run tt-metal binaries without silicon" role — ttsim wins each
on every axis. **This still holds**, and nothing below softens it: the
fact that tt-sim now matches ttsim bit-exactly across the `optests/`
suite on both arches is a *consequence of using ttsim as the oracle*
(lane 4), not a claim to have overtaken it. Coverage is what the op
tests and examples reach; ttsim's is the ISA. Defensible lanes:

1. **Cycle-approximate performance estimator** (§I) — ttsim is
   explicitly *not* cycle-accurate. tt-sim's per-unit cost model +
   observability stack is a tool for *understanding why kernels are
   slow*, complementary to ttsim's correctness role.
2. **Hackability** — pure Python, ISA-docs-shaped modules. Editing the
   simulator is editing `.py`, not forking C++.
3. **Observability tooling** (§H) — Perfetto with NoC arrows, LCOV via
   DWARF, Cachegrind memory hotspots, Spike commitlog diffing, Parquet
   counter datasets. Not present in ttsim's public surface; load-
   bearing for (1).
4. **Differential testing against ttsim** for correctness — use ttsim
   as oracle so tt-sim engineering focuses on the cycle side. **This
   one has landed** and is now one of the project's strongest assets:
   `optests/` + `optests/diff.sh` run the *same* compiled tt-metal
   program through tt-sim and ttsim and diff the DRAM dump, on either
   arch (`TT_SIM_ARCH`). See §H "Differential testing against ttsim".
5. **Education** — `driver/simple/ex1-5` ladder + readable backends.

Frame this honestly in user-facing docs: tt-sim is a *first-order
performance estimator* with hackable internals and rich tracing, not a
cycle-accurate validation tool. Cycle-accurate (matching silicon
within ~%) needs RTL or captured silicon traces; neither is publicly
available (see §I "Calibration against silicon traces"). Until that
data exists, call it "performance estimator", not "cycle-accurate".

### Re-prioritised headline goals

1. **§I cycle-approximate perf model** — promoted from opportunistic
   perf work to the main thing. **Foundation landed:** the
   event-driven cycle pump's Phase 1 (tile-level dormancy) is in, and
   the phased design through to multi-cycle occupancy — where the
   per-unit cost tables plug in — is
   [`docs/plans/event-driven-pump.md`](docs/plans/event-driven-pump.md).
   The model itself (cost tables, stalls, back-pressure) is still to
   come.
2. **§H follow-ups gated on §I** — Perfetto durations, NoC
   `vc`/`issue_cycle`/`arrival_cycle`, FPU/SFPU stall reasons, packer
   back-pressure, L1 bank conflicts. Schema exists; data becomes real
   when §I lands.
3. **§L profiling + optimisation** — **measurement done, four
   optimisations landed** (see §L). `six` (128³ bf16 matmul) went
   35.7 s → 7.9 s. Kernel-scale workloads are still far out of reach;
   the writeup is
   [`driver/wormhole/docs/profiling.md`](driver/wormhole/docs/profiling.md).
4. **Differential testing harness against ttsim** — **landed** (§H).
   Extending op coverage is now incremental work, not a workstream to
   stand up.

De-prioritised:

- Closing every §D `NotImplementedError` — cover what the headline
  cycle-approx examples exercise; let ttsim cover the rest. (The
  differential harness has since closed a large number of them on the
  paths real kernels take — see §D.)
- Multi-arch: **Blackhole is done** — see
  [`docs/plans/blackhole-support.md`](docs/plans/blackhole-support.md). The
  device layer is parameterised behind an arch profile
  (`tt_sim/arch/`), the full shared example suite (`one`–`nine`,
  `loopback`) runs live on both arches, and both arches match ttsim
  bit-exactly across the `optests/` differential suite. Every item in
  that plan is complete apart from those explicitly recorded as
  blocked (chip-to-chip ethernet: no second chip exists) or as
  waiting on a target workload (Blackhole eth tiles, the full worker
  grid, L2CPU/Security tiles, `SFPLOADMACRO`). Quasar remains out of
  scope. Continued Blackhole work is now ordinary maintenance, not a
  port.
- Native fast-dispatch. Slow-dispatch suffices for everything tt-sim
  is now aiming at.

---

## A. tt-metal wire bridge (`tt_sim/bridge/`)

The integration runs end-to-end for the canonical `programming_examples/`
under `TT_METAL_SLOW_DISPATCH_MODE=1` (see §E). The wire bridge
materialises the workers listed in `TT_SIM_TENSIX_COORDS` (default
`1-1` on Wormhole, `1-2` on Blackhole); other worker coords stay as
`NullCore` to keep the cycle pump cheap given tt-metal's grid-wide init
traffic.

The bridge itself is now **arch-agnostic** and lives in `tt_sim/bridge/`
(protocol, transport, fabric, cores, trace, `_flatbuf`, and the
cycle-pumping `Device` wrapper). `driver/wormhole/server/` and
`driver/blackhole/server/` are thin: each injects a device factory and a
coord map. Anything below that talks about `driver/wormhole/server/<x>.py`
for a shared module means `tt_sim/bridge/<x>.py`.

- **Multi-Tensix threading.** Structural refactor landed: `MultiTileClock`
  (`tt_sim/device/clock.py`) replaces the flat central `Clock`, holding
  one `Clock` per tile and spawning one daemon worker thread per tile
  with a per-cycle `threading.Barrier`. Auto-engages threading when ≥2
  tiles are marked `heavy=True` (Tensix only — DRAM/eth stay light);
  falls back to a sequential loop otherwise. `TT_SIM_THREADED=0` forces
  sequential. Cross-thread synchronisation: per-NUI `threading.Lock`
  guarding the `noc_new_requests_to_handle` inbox in `tt_noc.py:transmit`
  + the per-cycle swap in `clock_tick`; `EventBus.publish` snapshots
  the subscriber list under a lock with the disabled-path kept as a
  single attribute read. `TT_Device.shutdown()` joins worker threads;
  the wire-bridge `__main__.py` calls it on exit. Single-Tensix
  examples (`examples/one/`–`six/`, the default `Wormhole()`)
  keep the existing sequential path with no observable regression.
  Heavy-only worker pool landed: workers are now spawned only for
  Tensix tile clocks, and the coordinator thread ticks the 22 cheap
  (DRAM + eth) clocks inline in parallel with the workers, joining the
  per-cycle barrier as one extra party. Barrier participant count for a
  4-Tensix configuration is 5 (4 Tensix + 1 coordinator), not 26.
  Measured idle 4-Tensix `run(500)`: ~6× slower than sequential, down
  from ~28×. Real-workload validation completed across two kernel
  shapes (4 Tensix tiles in parallel each, each tile writing to a
  disjoint DRAM region):
    - **BRISC-only dataflow** (`one/` vector-add): threaded
      **2.4× slower** than `TT_SIM_THREADED=0` sequential (5.7s vs
      2.4s wall on 1200 cycles).
    - **Coprocessor-heavy matrix unit** (`four/` int8 ELWADD via the
      Tensix matrix unit, plus BRISC + NCRISC + 3 TRISCs): threaded
      **1.56× slower** than sequential (5.22s vs 3.34s wall, mean
      of 3 runs).
  Conclusion: at the current Python granularity neither dataflow- nor
  coprocessor-heavy kernels release the GIL enough per cycle to
  amortise the barrier. The matrix backend's numpy ops are short
  enough that GIL re-acquire dominates. Threading is structurally
  correct but a perf regression for every workload measured so far.
  Threading is now strictly opt-in: `MultiTileClock` defaults to the
  sequential tick loop and only engages workers when
  `TT_SIM_THREADED=1` (or `true`/`yes`/`on`) is set in the
  environment AND ≥2 clocks are flagged heavy. The structural
  threaded code path is preserved for future re-evaluation; default
  multi-Tensix runs no longer regress vs single-Tensix.
  Follow-up open:
    - Re-evaluate on free-threaded Python 3.13t (PEP 703) where the
      GIL doesn't serialise per-cycle Python work between workers.
      This is the most realistic path to a speedup without rewriting
      the cycle pump in C.
  - *Test:* no example needed for the structural refactor — covered by
    every existing single-tile example continuing to pass. Remaining
    perf follow-up wants a multi-Tensix, real-compute example (an
    `examples/seven/`) plus a slow-dispatch run of
    `matmul_multicore_reuse_mcast` once that lands. `examples/nine/` is
    now a genuine **two-tile** example (producer tile + consumer tile
    bridged over the NoC with a cross-tile semaphore) and passes on
    both arches, so the two-tile case is covered; the open item is a
    wider grid.
- **NCRISC / TRISC reset fan-out.** *Mechanism landed, wired on
  Blackhole only.* `tt_sim/bridge/device.py::Device.deassert_reset`
  releases the master BRISC plus each subordinate whose bit is set in
  the launch message's `enables` bitmask, read from L1 at the per-arch
  `launch_enables_offset` the driver injects. `driver/blackhole`
  injects `0xE8`; `driver/wormhole` passes nothing and keeps the legacy
  BRISC-only wire deassert. Landing this was what made multi-processor
  programs (`two` onwards) compute correctly on Blackhole — releasing
  every core unconditionally runs un-kerneled cores' idle firmware and
  perturbs BRISC's go-message completion, and releasing only BRISC
  never starts a writer.
  - *Test:* covered on Blackhole by every guard from
    `driver/blackhole/server/two_replay_test.py` onwards. Open: decide
    whether Wormhole should use the same path (it needs the WH L1
    offset for `kernel_config_msg_t.enables`), and a variant with the
    BRISC enable bit cleared to check a disabled core never leaves
    reset.
- **Two `LaunchProgram`s in one process stall the simulator.** *Fixed.*
  Found by `perfbench/tensixbench` phase B, which launches one program
  per math fidelity: either fidelity alone finished, LoFi → HiFi2 **in
  the same host process** ran **25 minutes** on one attempt and **18
  minutes** on a second without the second launch ever finishing, at
  **95 % CPU** throughout. It was nothing phase B does. The Sync Unit
  (`tt_sim/pe/tensix/backends/sync.py`) deleted granted waiters from
  `blocked_mutex` only `if len(to_remove) > 1`, so a *lone* grant — one
  mutex, one waiter, the common case — stayed queued for ever, and a
  queued entry is re-granted on every later `clock_tick`. The mutex was
  therefore pinned to that thread for the rest of the device's life,
  silently re-acquired the cycle after each `ATRELM`. Nothing failed
  while no other thread wanted it, which is why one launch was always
  fine and why phase A's three launches never tripped it; the first
  cross-thread `ATGETM` after the first contention blocked for ever. The
  stale entry also kept `is_clock_idle` false, so the tile never went
  dormant — hence the live spin rather than a quiet stall.
  `--fidelities LoFi,HiFi2` now completes end-to-end in **~9 s**, and the
  fidelity difference phase B exists to measure can be self-checked
  against tt-sim again.
  - *Test:* `tt_sim/pe/tensix/sync_mutex_queue_test.py` pins the queue
    behaviour at the unit (a granted waiter leaves the queue; a released
    mutex is really free; the unit goes idle again), and
    `driver/blackhole/server/twolaunch_replay_test.py` replays a captured
    two-launch phase-B conversation and requires **both** launches to
    reach `RUN_MSG_DONE`. Both fail before the fix.
- **`START` (cmd=4) handler.** `tt_sim/bridge/transport.py::_handle`
  log-and-skips it. Never observed in any captured trace; revisit if a
  future tt-metal release sends it.
  - *Test:* deferred — wait for a tt-metal release that actually emits
    cmd=4 in a captured trace before designing a regression.

---

## B. RISC-V baby cores (`tt_sim/pe/rv/`)

Per the [BabyRISCV ISA docs](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/BabyRISCV/README.md),
the five baby cores implement **RV32IM only**, plus Tenstorrent's
`.ttinsn` custom extension for pushing instructions to the Tensix
coprocessor. They do **not** implement RV32F, RV32D, RV32C, or Zicsr —
all floating-point math happens on the Tensix FPU/SFPU (see §D), not
on the baby cores. `fence` executes as `nop` and `ebreak`/`ecall`
trigger a debug pause. tt-sim's current behaviour (`NotImplementedError`
on F instructions, no Zicsr handlers) is therefore correct hardware
modelling, not a gap — kernels built for real Wormhole silicon will not
emit these instructions.

**Blackhole is different and is modelled.** Its baby cores add Zicsr,
Zaamo, Zba, Zbb and partial F/Zfh/V (V on TRISC2 only). tt-sim
implements Zba + Zbb (`isa/b_isa.py`), Zaamo local-L1 atomics
(`a_isa.py`) and Zfh half-precision (`zfh_isa.py`), all unit-tested,
with loud `NotImplementedError` guards (`guard_isa.py`) for the
`.s` single-precision and vector instructions that are not modelled
yet. The set is selected per arch via
`ArchProfile.baby_core_isa_extensions` / `trisc2_isa_extensions`, so
Wormhole stays RV32IM. Still future work there: `mret`, F
single-precision execution, the V vector unit, and Zfh's BF16 CSR mode
— none reached by any kernel in the tree.

The only outstanding RV item on Wormhole is functional-vs-perf:

- **No pipeline modelling.** Each instruction completes in one tick;
  no fetch/decode/issue latency, no memory-stall back-pressure. Fine
  for functional sim; a perf model would need this. Cross-ref §I
  "Cycle accuracy" — the per-unit cost framework is the home for it.
  - *Test:* no example needed — perf-model work.

---

## C. NoC (`tt_sim/network/tt_noc.py`)

Landed since this section was written: the per-arch coordinate-encoding
strategy (`network/noc_coords.py` — Wormhole packs the coord into the
MID register, Blackhole uses the dedicated `TARG/RET_ADDR_HI`), the
profile-driven grid dims and max burst size, all 8 Blackhole DRAM
channels, and **NoC alignment checking** (`network/alignment.py`):
hardware requires source and destination to be *congruent* in their low
bits, and a violation now raises `NoCAlignmentError` naming the path and
the required alignment (defaults on; `TT_SIM_DISABLE_ALIGNMENT_CHECKS`
disables). Also **response routing by endpoint identity**: a response now
goes back to the NUI that issued the request (`NoCDataRequest.reply_to` via
`NUI.send_response`) instead of re-resolving the requester's coordinate,
which on NoC 1 — whose directory also holds mirrored coords — could resolve
a *different* tile and kill multi-core runs (`noc_routing_test.py`).

- **NoC atomic ops beyond ATINC.** Atomic-add (`noc_semaphore_inc`)
  works via `RequestInitiator.handle_atomic_transfer`, posted and
  non-posted both. The richer atomic ops `ATCAS`, `ATSWAP`, `ATINCGET`
  issued by the Tensix ThCon backend are not implemented yet — they
  share the NoC-level dispatch but the actual op semantics live in
  §D "ThCon stalls".
  - *Test:* a non-posted-atomic kernel calling
    `noc_atomic_increment_with_response` and waiting on
    `NIU_MST_ATOMIC_RESP_RECEIVED` would cover the response-marked
    path that `examples/nine/` doesn't.
- **NoC register coverage.** `tt_noc.py` raises on unhandled register
  reads/writes (the `else` arms of `NUI.read` / `NUI.write`) — many
  offsets beyond the basic counter set, the command buffers and the
  `NOC_CFG` block are unimplemented.
  - *Test:* micro kernel under `driver/simple/` reading the specific
    unimplemented counter offsets (NIU_MST_RD_RESP_RECEIVED, etc.) and
    asserting each returns a plausible non-zero value after driving
    matching traffic.
- **Multi-hop routing latency.** Multicast write fan-out lands in
  `handle_multicast_write_transfer` (covers both
  `noc_async_write_multicast` and `noc_semaphore_set_multicast`).
  Multi-hop *latency* — per-hop cycle cost, multi-router pathfinding,
  head-of-line blocking — is not modelled; multicast fan-out is a
  single-dispatch operation today. Falls into §I cycle accuracy
  rather than NoC correctness.
  - *Test:* `matmul_multicore_reuse_mcast` via slow-dispatch once §A
    "Multi-Tensix threading" makes broader-grid runs practical.
- **Router arbitration & flow control.** All requests accepted
  immediately; no back-pressure or fairness.
  - *Test:* no example needed at functional level — extension of the
    multi-Tensix multicast example would surface perf differences but
    not correctness gaps.

---

## D. Tensix coprocessor (`tt_sim/pe/tensix/`)

**Read this section against `optests/` first.** The differential harness
(§H) has closed a large fraction of what this section used to list, on
both arches: the SFPU lane model, the FMA models, the FPU accumulate
datapath, the reduce/transpose/pool paths, the Blackhole shifted-field
decodes, and a run of general bugs that were arch-neutral. What remains
open below is what no op test or example has yet reached. Line numbers
have been dropped where they had already drifted; grep for
`NotImplementedError` in the named file instead.

### Frontend / decode pipeline
- ★ **A permanently stalled backend unit cannot wedge the issuing core, so
  tt-sim runs to completion where Blackhole silicon deadlocks.** Named because
  it was measured on a card, not inferred.

  `perfbench/tensixbench --dvalid-unpacr-nop` issues the Blackhole-sanctioned
  dvalid setup — `UNPACR_NOP` with `set_dvalid` and `Unpack_Pop = UNP_ZEROSRC`,
  copied verbatim from `tt_llk_blackhole/llk_lib/llk_unpack_common.h`'s
  `_llk_unpack_set_srcb_dummy_valid_` — and then deliberately never clears
  dvalid, because holding the valid bits for the whole burst is what the
  measurement needs. That form has two halves: it first **waits until the Src
  bank named by `MatrixUnit.Src{A,B}Bank` is no longer valid** (ttsim
  `src/tensix.cpp`, `TENSIX_EXECUTE_UNPACR_NOP`, Blackhole branch:
  `wait_bank = stall_clr_cntrl ? unpack_bank : matrix_bank; if (src_valid &
  (1 << wait_bank)) return false;`), and only then zeroes the bank and hands it
  over. So the setup **acquires without releasing**: the first run leaves
  `SrcA[0]`/`SrcB[0]` owned by the Matrix Unit, and the *next* execution of the
  same setup waits for ever — the next program launch in the same process, or
  the first launch of the next process on the same card. Only `tt-smi -r 0`
  clears it. Plain `SETDVALID` has no wait half at all, which is why
  `--dvalid-once` re-runs indefinitely on the same card; that is the entire
  difference between the two setups.

  tt-sim models the stall condition itself correctly and reaches the *identical*
  blocked state — instrumented, the t2 launch's `UNPACR_NOP` blocks on
  `wait_bank=0 matrix=0 client=MatrixUnit` and re-runs 17,329 times — and then
  reports the launch **done** anyway. Two reasons, and only the first is fixed:
  - `UnPackerUnit.hasInflightInstructionsFromThread` did not count a blocked
    unit's latched instruction, so `CoprocessorDoneCheck` (the PC-buffer drain
    and the deadlock watchdog's signature) saw the thread as idle. **Fixed**,
    with the wait reason now named in the watchdog's report.
  - The remaining half: **nothing back-pressures the baby RISC-V on Tensix
    instruction issue.** On silicon the thread's instruction FIFO fills behind
    the stalled unpacker instruction and the core stalls on its next `.ttinsn`
    store; tt-sim's core issues regardless, so a kernel with no drain point
    (`raw_probes.cpp` has none) finishes with a wedged unit behind it. This is
    the same missing mechanism as "the cost model is invisible to a device-side
    clock" in `docs/plans/tensix-cost-benchmark.md`.
  - ~~*Next step, and deliberately not a wedge:* a per-unit **blocked-cycle
    counter**, needing a false-positive survey first.~~ **Landed** as
    `TT_SIM_UNIT_STALL` / `TT_SIM_UNIT_STALL_THRESHOLD` in
    `tt_sim/device/deadlock.py`: any unit exposing `blocked_on()` is picked up
    at the watchdog's sampling cadence and then counted **per cycle**, and a
    `[UNIT STALL …]` block naming thread, opcode, Src register and bank fires
    where the global watchdog cannot, because the rest of the device is still
    making progress.
    - *The survey it was gated on.* Every replay guard, example and
      differential op test in the tree (41 workloads, Wormhole and Blackhole)
      was instrumented for the longest run of consecutive cycles each backend
      unit spent blocked while the device progressed, with and without
      `TT_SIM_COST_MODEL`. The unpacker is the only unit that meaningfully
      blocks at all: worst case **3,528 cycles** (WH `sfpumath`, unpacker 1's
      `UNPACR_NOP` waiting for the SFPU to hand SrcB back), then 3,007 (the BH
      run of the same workload), 944 (BH `matmulblock`), 925 (BH `sfpuchain`);
      nothing else exceeds 300. Sync's mutex queue peaks at 13 cycles; ThCon,
      matrix, SFPU, packer, misc and mover never block. No unit ended any
      workload still blocked. The threshold is 10,000 — 2.8x the worst measured
      legitimate wait, and inside the lifetime of every workload (the longest
      runs 27,499 cycles end to end). Zero of the 41 produce a report.
    - *Two findings that shaped it.* **(a)** A *sampled* counter is a false
      positive generator and was rejected on the data: a kernel is a loop, so a
      unit that legitimately blocks once per tile on the same instruction is
      blocked at consecutive sample points — `sfpumath`'s unpacker is blocked
      for 4,721 of ~20,000 cycles across 9 runs — and sampling cannot tell that
      apart from never having moved. Hence the per-cycle confirmation, which
      costs nothing while nothing is blocked. **(b)** It **warns rather than
      raises**, against the wanted behaviour above. The measured separation is
      clean, but the legitimate bound is *architecturally* the downstream
      pipeline's stall, not a constant: an unpacker waits for the math thread,
      which waits for Dst / output CB space, which waits on the packer, the
      writer core and — in a multi-core tt-metal pipeline — a remote consumer.
      No in-tree workload pipelines core-to-core, so that case is not in the
      numbers, and aborting a correct kernel on an extrapolation is the trade
      declined for the `SETDVALID` `NonContractualBehavior` guard. ~~Promote it
      to a raise once a core-to-core pipelined workload has been surveyed.~~
    - *That survey has now been done, and the answer is that the cycle
      threshold must never be promoted.* `examples/pipestall` is the missing
      workload: `nine`'s two-tile pipeline plus a reverse credit semaphore, so
      the producer's Tensix backs up behind the consumer **core** with the
      consumer's per-chunk cost as a runtime argument. The producer's unpacker
      blocks for `5 * delay + 56` cycles (r = 1.000, Wormhole; Blackhole
      matches), i.e. **linearly in downstream cost with no ceiling**: 1,056
      cycles at the frozen guard setting, **10,056 at a consumer costing
      ~10,000 cycles a tile**, 20,056 at twice that. Deeper buffering does not
      help — four credits or a four-page output CB each bought under 8 %,
      because a rate-limited producer stalls one consumer turnaround per tile
      however deep the pipe. And raising the threshold would defeat the other
      end: the minimal `UNPACR_NOP` reproduction runs ~18,300 cycles end to
      end, so anything above ~16,000 misses it. The interval between "high
      enough for a correct pipeline" and "low enough for a short wedge" is
      empty; 10,000 stays, as a **hint**, not a verdict.
    - *And it now reads and behaves like one.* It stays on by default — a
      broken third-party eight-core GEMM produced a report on all eight
      workers and none on the corrected form, which nobody would have seen
      behind an opt-in flag, and "suppress it while a downstream consumer is
      progressing" is unsound because the true positive also runs on a
      progressing device (that is the check's whole premise). What changed is
      the message: it prints both readings, declines to pick one, and is
      **de-duplicated to one report per unit per waited-on instruction** —
      the old once-per-window count was run-length dependent (576 reports on
      one run of that GEMM, 352 on a longer one) and so meant nothing. A unit
      that later unblocks prints `[UNIT STALL CLEARED]`, which is what lets a
      user settle the question mid-run instead of waiting for teardown.
    - *What replaces it:* `[UNIT WEDGED]`, a terminal check with no threshold
      at all. A Src bank is handed back by an instruction and no thread can
      issue one from soft reset, so a unit still blocked once **every baby core
      on its tile is in reset** can never be satisfied. It fires on a
      reproduction far too short for any cycle count, and cannot fire on a
      correct kernel (no unit ends a launch blocked in any of the 41 surveyed
      workloads or any `pipestall` configuration). Warning for now; this is the
      one to promote to a raise.
    - Also fixed here: the per-unit arming pass rode the *global* watchdog's
      cadence, so end-to-end latency was `unit_stall_threshold +
      deadlock_threshold / 8` and lowering `TT_SIM_UNIT_STALL_THRESHOLD` alone
      could not make a report arrive sooner. It has its own cadence now
      (`1.125x` its own threshold, whatever the other knob is set to).
    - *Guards:* `driver/{wormhole,blackhole}/server/pipestall_replay_test.py`
      replay the frozen trace, check the values **and** assert the longest
      blocked run is a real cross-core stall that stays under the shipped
      threshold.
    - *One value divergence came out of building it, and it was the workload's,
      not the simulator's.* `pipestall` sized its CB pages to the 256 B chunk
      its kernels move; `pack_tile` writes a whole 4,096 B tile, so at
      `OUT_DEPTH=2` page 1 sat inside page 0's pack footprint and the pack of
      chunk N shredded the unread chunk N-1 (chunk 1 came back all zeroes, both
      arches). Pages are `tt::tile_size(format)` now. `optests/packspill` pins
      the full-tile pack footprint against ttsim (agreement on all 1,024
      datums, 960 of them past the page), and `pipestall-2page` runs the shape
      live in both arch runners.
  - *Test:* `tt_sim/pe/tensix/setdvalid_srcrow_test.py` already pins the
    instruction word and the hand-over; the missing one issues the setup twice
    with no intervening clear and asserts the second blocks.
- **Wait-gate stalls** on srcA/srcB availability not fully enforced.
  - *Test:* no example needed — exercised indirectly by every Tensix
    compute kernel; targeted regression would be a perf-model task.
- **Instruction issue latency.** Operations complete the cycle they
  decode; no realistic issue/decode separation.
  - *Test:* no example needed — perf-model work.

### Backend execution units
- ~~★ **The sync unit's mutex branch tests for `"ATGEM"`, but the instruction
  table calls it `"ATGETM"`**~~ **Fixed** (`backends/sync.py`,
  `issueInstruction`). The branch never fired: `ATGETM` fell to the else-arm and
  was admitted as "one of any other instruction", so the *up to three mutex ops
  on distinct mutexes* rule and the same-mutex conflict check were both dead
  code — and, symmetrically, a queued `ATGETM` blocked every other sync op for a
  cycle. `SyncUnit.md`'s throughput table is explicit that the two rows are
  independent: mutex ops "issue up to three per cycle, provided they refer to
  different mutexes", semaphore ops "issue at most one of these per cycle", and
  "multiple instructions can be accepted per cycle, subject to the limitations
  in the Throughput column". ttsim agrees on the per-mutex semantics
  (`TENSIX_EXECUTE_ATGETM` keys the held-by state on `mutex_index` alone, and
  stalls only when *that* mutex is held); it models no issue width, so the doc
  is the oracle for the rule itself.
  **What moved, measured by running both decision functions side by side over
  the whole corpus:** with the cost model off (the default), *nothing* on
  Wormhole — all 11 example traces still replay byte-identical, and across them
  the corrected branch takes the same decision as the old one every time, because
  the queue drains each cycle and an `ATGETM` is only ever offered into an empty
  one. Blackhole: **two** decisions change, in `optest` and `sfpumath`, both an
  `ATGETM` now admitted in the same cycle as an already-queued `SEMPOST` /
  `SEMWAIT` instead of a cycle later. All 22 value guards still pass with
  unchanged numbers (`six` PCC 0.9982 unmoved), so this is scheduling only; **no
  trace was recaptured**. With `TT_SIM_COST_MODEL=1` one more decision changes
  (`loopback`: a `STALLWAIT` admitted alongside a queued `ATGETM`), and the
  cost-model replay mismatch set is byte-identical to what HEAD already produced.
  Side effect worth knowing: the mixed-queue `mutex_index` guard below is no
  longer cost-model-only — those two Blackhole guards reach it by default.
  - *Test:* `tt_sim/pe/tensix/sync_mutex_issue_test.py` — two `ATGETM`s on
    different mutexes share a cycle, two on the same mutex do not, a fourth
    distinct mutex is refused, and a semaphore op still shares a cycle with a
    queued `ATGETM`. Four of the five fail against the pre-fix code.
- ~~★ **Unpacker row stride / upsampling are accepted and silently ignored.**~~
  **Now raises**; still unmodelled. `check_modelled_settings` in
  `backends/unpacker.py` rejects, at decode and before a datum moves, any
  `Tileize_mode` `RowStride` other than the contiguous one and any
  `Upsample_rate != 0`, naming the configured value, the contiguous value and
  what the walk does instead. The contiguous stride is `DatumSizeBytes *
  UnpackRowWidth`, and `UnpackRowWidth` is 16 on Wormhole but **32 on Blackhole
  above 1 byte/datum** — so the test is "stride equals the contiguous value for
  *this* arch", not "stride is zero". Two decode bugs fell out: the
  `Shift_amount_cntx1` term was shifted by 4 instead of 8 (colliding with
  `cntx0`, so a decoded stride could not exceed 4 bits of the doc's 12), and the
  non-tileize stride was hardcoded to `× 16` on both arches.
  `Upsample_and_interleave` on its own is **not** rejected: at rate 0 the doc's
  upsample loop runs exactly once, so it is a genuine no-op. Deliberately *not*
  suppressible by an env var, unlike `TT_SIM_DISABLE_ALIGNMENT_CHECKS`: those
  guard hardware `UndefinedBehavior` a user may want to explore, this is
  "tt-sim does not implement it", where continuing just produces wrong numbers.
  A census over all 22 Blackhole guards and the Wormhole replays confirms
  nothing in the corpus trips it (12 combinations, every one contiguous,
  `Upsample_rate` and `ColShift` 0 throughout). `rowStride` / `upsampleZeroes` /
  `upsampleInterleave` — and, with the `ColShift` entry below, `colShift` — are
  no longer threaded into `perform_unpack` at all.
  **Update — the strided walk is now modelled** (upsampling still raises). The
  input walk is the doc's: `unpackRowWidth` datums contiguously, then a jump of
  `rowStride` bytes, i.e. input datum `i` at `InAddr_Datums + DatumSizeBytes *
  (i % UnpackRowWidth) + RowStride * (i / UnpackRowWidth)` — which degenerates
  to the old flat read whenever `RowStride` is the contiguous value, so nothing
  contiguous moved. Both walks carry it: the scalar loop rewinds and jumps every
  `unpackRowWidth` datums (the doc's incremental form), and the batched
  `_unpack_block` **was extended rather than declined** — the source offsets
  become one `np.arange` index array, so the 43× block path (and the destination
  rectangle, which the stride does not touch) still applies to every tilize
  unpack, which is exactly the hot path this matters for. It reads one L1 span
  ending at the walk's last datum, so it touches nothing the scalar loop would
  not; it declines a `RowStride` that is not a whole number of datums (it
  indexes in datums; `RowStride` is always a multiple of 16 bytes, so this
  cannot fire today) and the scalar loop, which walks in bytes, remains exact.
  `check_modelled_settings` keeps the upsample and `ColShift` rejections and
  gains one: `Tileize_mode` with a sub-byte datum (BFP2/BFP4), which the doc
  itself calls `UndefinedBehavior` and which the walk cannot address.
  - *Tests:* `tt_sim/pe/tensix/unpack_stride_test.py` (both arches, end to end
    through `read_unpack_state`): strided unpacks land the right datums at
    strides 96–65520; the walk uses *this* arch's row width (one L1 image, laid
    out for a 16-wide walk, must read as the 16-wide image on Wormhole and as
    the — different — 32-wide image on Blackhole); the batched gather equals the
    scalar loop datum for datum at strides shorter than a row, equal to it and
    longer; and the three `Shift_amount_cntx` fields are now pinned through the
    walk itself rather than through a rejection message. 14 of 15 strided cases
    fail against a walk that ignores `RowStride` (the 15th is Blackhole at
    stride 64, where contiguous *is* 64).
  - *Differentially:* `optests/tilize` — `copy_tile` control, `tilize_block`
    at block 1 (`RowStride` 64) and block 2 (`RowStride` 128), against a
    computed golden of 2048 distinct bfloat16-exact ramp values. **Wormhole:
    `TT_SIM_ARCH=wormhole ./optests/diff.sh tilize` PASSes** (2048 elements
    byte-identical to ttsim, all three ops 0 errors); with the walk forced back
    to contiguous, op1 is 960/1024 elements wrong. Frozen as
    `driver/wormhole/server/tilize_replay_test.py` (computed golden, all three
    ops checked).
  - *Blackhole now passes too*, once the packer stopped reading the PACR's
    bits 11:8 as Wormhole's `PackSel`. On Blackhole there is **one** packer and
    that field is `read_intf_sel`, a mask over its four *Dst read interfaces*
    (ttsim: `constexpr uint32_t n_packers = 1` under `TT_ARCH_VERSION == 1`,
    indexing only `THCON_SEC0_REG1`). The tilize pack MOP
    (`_llk_pack_mop_config_<tilize=true>`) issues `0b0101`/`0b1010`, so tt-sim
    was running "packers 2 and 3" off `THCON_SEC1_REG1` — a section tt-metal
    never writes on this arch — packing the tile a second time to a junk
    address and reading the unwritten, *inverted* `Disable_zero_compress` as
    "compressing", which refused the whole program with a bare
    `NotImplementedError`. `participating_packers` now scopes both refusals
    (zero compression and sub-16-bit output format) to a packer the PACR
    actually drives, and `dst_read_interfaces` gives interface `k` Dst row
    `base + k`, which is how `0b0101`/`0b1010` interleave two Dst halves into
    one tile. **`./optests/diff.sh tilize` PASSes on Blackhole** (2048 elements
    byte-identical, all three ops 0 errors), frozen as
    `driver/blackhole/server/tilize_replay_test.py`; without the row rebase the
    same trace is 768/1024 wrong on op1. Unit-pinned in
    `tt_sim/pe/tensix/pack_intf_sel_test.py`.
  - *Still unmodelled, now correctly scoped:* pack-side zero compression. A
    packer this PACR drives that genuinely requests it still raises; ttsim
    models it nowhere (`TENSIX_EXECUTE_PACR` never reads the field), and no
    in-tree workload asks for it.
- ~~★ **Unpack-side untilize (`untilize_block`) lands the whole tile on one
  SrcA row.**~~ **Fixed.** `llk_unpack_untilize` — the deprecated CB→CB
  untilize, and still the tt-xftn compiler team's codegen target — is an
  *output* address-generator problem, not an input-walk one: it never touches
  `Tileize_mode` or `RowStride`. It widens the tile descriptor's `YDim` to 16 so
  ADC channel 0's Z stride (`XDim * YDim`) is a whole 16×16 face, rewrites
  `UNP0_ADDR_CTRL_XY_REG_1_Ystride` to one face row, and issues UNPACRs with
  `AddrMode` = ch0 Z += 1 / ch1 Y += 1, so each 16-datum face row lands at its
  own SrcA row and the four faces interleave into output rows. Two defects, both
  arch-neutral, which is why the failure was byte-identical on Wormhole and
  Blackhole:
  - **The tile descriptor was read with a stride of four config words.**
    `TensixBackend.getConfigValue(..., words=4)` fetched `addr32 + 4*n` rather
    than `addr32 + n`, so descriptor words 1/2/3 came from `THCON_SEC0_REG1` /
    `REG2` / `REG3`. `UNPACR_Regular.md`'s field table puts `XDim` at bits
    16–31, `YDim` at 32–39, `ZDim` at 48–55 and `WDim` at 64–71 of one
    contiguous bit string — i.e. consecutive words — and ttsim agrees
    (`tile_desc1 = cfg53` on WH / `cfg65` on BH, one past the descriptor).
    It went unnoticed for as long as it did because `THCON_SEC0_REG1` happens to
    hold the right `ZDim` (4) and a `YDim` of 0, which the doc reads as 1 — the
    right answer for every ordinary tile. Untilize's `YDim = 16` is the first
    configuration where the two disagree, and there the Z stride collapsed from
    a face (256 datums) to a row (16), so both UNPACRs of a pair read face 0.
  - **The start row was dropped on the `SRCA_SET_SetOvrdWithAddr` path.** The
    doc derives both destination coordinates from one running counter (`Row =
    OutAddr / 16; Col = OutAddr & 15`, `++OutAddr` per datum), so the initial
    output address is part of every datum's row. tt-sim split the counter into
    `(i / 16, i % 16)` and, on the `SetOvrdWithAddr` branch — the branch *every*
    current LLK takes — never added `OutAddr / 16 - 4` back, so all sixteen
    UNPACRs of a pass wrote SrcA row 0. That alone is the reported 0.21875
    (`0x3e60`) in every output element.
  Fixing the row exposed that Wormhole's own LLK relies on the row **wrapping**
  at 64: the SrcA clear ahead of the int32/fp32 SFPU kernels (`five`, `five-fp`,
  `loopback`) arrives with ADC channel 1's Z accumulated to exactly 64 rows.
  Blackhole's doc states the wrap outright (`Row &= 63`, "allowed for BH fast
  tilize"); Wormhole's calls it `UndefinedBehavior`. **ttsim was the authority
  on the descriptor layout and the doc on the output loop; on the wrap the doc
  is split by arch and silicon decides** — SrcA is 64 rows with a six-bit index,
  and wrapping reproduces exactly what those three examples were already
  verified to produce, so both arches wrap.
  The batched `_unpack_block` **took the mode rather than declining it** —
  nothing here is a BFP width or an `FP32 → FP16` conversion, and the start row
  is one addend on the destination row array, so the 43× block path still covers
  every untilize unpack. `check_modelled_settings` **gains** a rejection and
  loses none: an `OutAddr` that is not a whole 16-datum row, which the doc marks
  `UnsupportedFunctionality` and ttsim refuses too (`!(dst_addr % ROW_SIZE)`),
  and which the row/column split would silently mis-place.
  - *Tests:* `tt_sim/pe/tensix/unpack_untilize_test.py` (both arches, end to end
    through `read_unpack_state` / `perform_unpack_state`): a full untilize pass
    scatters face rows down SrcA and is asserted against *both* wrong images the
    two defects produced; the descriptor is four consecutive config words; the
    `YDim` the descriptor carries is the one that scales the Z stride; the
    batched and scalar walks agree; the row wraps at 64; a misaligned `OutAddr`
    raises at decode with SrcA untouched. 12 of the 18 cases fail against the
    pre-fix code (10 to the two defects, 2 to the wrap).
  - *Differentially:* `optests/untilize` op1 (`untilize_block`), against the
    same computed ramp golden as op0/op2. **Wormhole
    `./optests/diff.sh untilize` now PASSes outright** (1536 elements
    byte-identical to ttsim, all three ops), taking every Wormhole op test
    ttsim-WH will run to green. On **Blackhole** op1 matches too; op2
    (`pack_untilize_dest` in `DST_ACCESS_STRIDED_MODE`) followed later — see
    "`DST_ACCESS_STRIDED_MODE`" below.
    Frozen by moving op1 into `CHECKED_OPS` in *both*
    `driver/{wormhole,blackhole}/server/untilize_replay_test.py` and recapturing
    both traces; each guard's only trace change is the one recorded reply that
    used to carry the wrong tile. Perturbing either fix fails both guards
    (1024/1024 and 768/1024 wrong on op1 respectively).
- ~~★ **Blackhole `pack_untilize_dest` (`DST_ACCESS_STRIDED_MODE`) reads the
  wrong Dst rows.**~~ **Fixed** — this was the last live differential miss on
  either arch (`optests/untilize` op2, 992/1024 elements wrong). Blackhole's
  pack-side untilize MOP runs one packer interface *pair* in
  `DST_ACCESS_STRIDED_MODE`: read interface `k` takes its 16 datums from Dst row
  `base + 16*k` instead of `base + k`, i.e. the same row of successive 16-row
  faces, so one PACR emits one 32-datum row of the row-major output and the MOP
  steps `base` down the 16 rows of a face. Two decode gaps, both in the shared
  Wormhole-layout PACR argument table:
  - **`dst_access_mode` (raw bit 17) had no field at all**, so every PACR read
    contiguous rows. Re-read from `instruction_info["raw_instruction"]` on
    Blackhole, the established idiom for a moved or added operand.
  - **`AddrMode` swallowed it.** It is the table's *last* argument, so the
    decoder gives it every bit up to the opcode — 23:15. Harmless on Wormhole
    (nothing above bit 16 is encoded) but Blackhole packs `dst_access_mode`,
    `row_pad_zero` (20:18) and `cfg_context` (22:21) into that span, so a
    strided `ADDR_MOD_1` decoded as section 5 — one no LLK programs — and the
    pack Y counter never advanced: all 16 row-closing PACRs re-read the same
    face row and one output row was replicated down the whole tile. Now read as
    the true 2-bit field on Blackhole.

  `DEST_ACCESS_CFG` writes also now drive `DstRegister`'s `Adj16`/`Adj32` gates,
  which had been implemented from the ISA docs in Phase 4c and left inert
  because nothing wrote the register; tt-metal's `_llk_math_reconfig_remap_` is
  the first caller. **Sources:** the ISA docs have no BlackholeA0 PACR or
  Packers chapter at all — `BlackholeA0/.../Dst.md` links to pages that do not
  exist — so all they settle is that `remap_addrs`/`swizzle_32b` "also affect
  how packers address `Dst`" beyond `Adj16`/`Adj32`. The stride of 16 comes from
  ttsim (`row = pack_row + 16*(i / ROW_SIZE)`, "remap and swizzle cause stride
  to be 16 rows and not 8 here") and from the LLK, which documents the two
  config bits as "needed for enabling stride of 16". It is applied to *logical*
  Dst rows, because tt-sim — like hardware and unlike ttsim, which marks
  `Adj16`/`Adj32` an unhandled TODO — puts every Dst access through them.
  - *Still unmodelled, now scoped:* strided mode without both `DEST_ACCESS_CFG`
    bits, and strided mode with a non-contiguous `read_intf_sel`. Both raise, as
    they do in ttsim.
  - *Tests:* `tt_sim/pe/tensix/pack_strided_test.py` (11 cases, both arches,
    driven through a real encoded PACR): strided interfaces read a face apart,
    normal mode still reads contiguously, Wormhole ignores bit 17, the 2-bit
    `AddrMode` selects the section that advances Y, a `DEST_ACCESS_CFG` write
    reaches the Dst gates, and the two refusals fire. Each of the three changes
    reverted individually fails 4, 1 and 6 of them.
  - *Differentially:* `./optests/diff.sh untilize` on Blackhole FAIL → **PASS**
    (1536 elements byte-identical to ttsim); Wormhole still PASSes. Frozen by
    moving op2 into `CHECKED_OPS` in
    `driver/blackhole/server/untilize_replay_test.py` (its Wormhole sibling
    already had it) and recapturing the trace.
- ~~★ **Four Blackhole PACR fields are silently ignored.**~~ **Now raise**;
  still unmodelled. `ctxt_ctrl` (3:2), `addr_cnt_context` (14:13),
  `row_pad_zero` (20:18) and `cfg_context` (22:21) exist only on Blackhole, so
  the shared Wormhole-layout argument table has no entry for them and nothing
  in `PackerUnit` reads them. Nothing was ever *mis*-decoded — each lands
  inside a decoded argument's span, but every consumer takes only the bits
  Wormhole defines: `ctxt_ctrl` inside `Flush` (3:1) and `addr_cnt_context`
  inside `ZeroWrite` (14:12), both consumed as `get_nth_bit(..., 0)`;
  `row_pad_zero` and `cfg_context` inside `AddrMode` (23:15), which on
  Blackhole is re-read from the raw word as the true 2-bit field precisely
  because that span is shared; `Last` is one bit, `Concat` is never read and
  `read_intf_sel` occupies `PackSel`'s 11:8 exactly. But a kernel *setting*
  one would have it quietly dropped, which is the failure shape everything
  else on this unit refuses. All four are now read from
  `instruction_info["raw_instruction"]` (not from a decoded argument — two
  separate bugs have come out of the 23:15 span) and raise on Blackhole
  naming the field, its value and its bit span. ttsim refuses all four the
  same way, as `MissingSpecification` at the top of `TENSIX_EXECUTE_PACR`
  under `TT_ARCH_VERSION == 1`. Wormhole is untouched — those bit positions
  mean something else there and its PACR is correct today.
  - *Test:* `tt_sim/pe/tensix/pack_strided_test.py`, three parametrised
    families over the four fields: each raises on Blackhole; each leaves every
    argument tt-sim actually reads bit-identical under the *real* decoder
    (`TensixInstructionDecoder.getInstructionInfo`, which the whole file now
    uses instead of a hand-rolled stand-in), which is what pins "ignored, not
    mis-decoded"; and Wormhole packs unchanged.
- ~~★ **Unpacker `ColShift` is partly modelled and wrong at the edge.**~~
  **Now raises**; still unmodelled. `THCON_SEC*_REG2_Shift_amount_cntx[ctx]`
  (when `Tileize_mode` is clear) shifts each datum left by `ColShift` columns,
  and the doc *drops* the datums with `Col < ColShift` (`UNPACR_Regular.md`:
  `if (Row < 4 || Col < ColShift) continue;`). Both the scalar walk and
  `_unpack_block` instead computed `outCol = col - colShift` unguarded, so those
  datums landed at a **negative** index — which numpy wraps to the far end of the
  row, clobbering columns that should have been left alone. Demonstrated before
  the fix, identically on both paths and both arches: at `ColShift = 2` the
  datums from columns 0 and 1 of every row appeared in columns **14 and 15**,
  the two columns the doc says to leave untouched.
  Of the two options recorded here — implement the drop, or raise — **raise**
  was taken, in `check_modelled_settings` beside the row-stride and upsample
  rejections, and `colShift` is no longer threaded into `perform_unpack` /
  `_unpack_block` at all (so no half-model is left to rot, and the batched
  path's index arrays stay a plain rectangle). The deciding evidence was that
  **there is no oracle**: the doc gates `ColShift` behind
  `UnsupportedFunctionality` in the *same* `if` as `UpsampleZeroes` and
  `UpsampleInterleave` ("no known usage, confidence in specification **below**
  is weak" — i.e. the drop pseudocode itself is what is flagged), and ttsim
  declines `col_shift` in the same `TTSIM_VERIFY` breath as `upsample_rate`
  (`tensix.cpp:2836`). Implementing the drop would therefore have replaced a
  silent wrong answer with a *confident-looking* one that nothing could check,
  while buying no coverage — the census finds `ColShift == 0` everywhere in the
  corpus. Raising closes the silent path outright and names exactly what to
  implement if a kernel ever needs it. Not env-var suppressible, for the same
  reason as the row-stride check.
  - *Test:* `tt_sim/pe/tensix/unpack_stride_test.py` (both arches): the
    rejection at `ColShift` 1/2/15; that it fires from `read_unpack_state`, at
    decode, with SrcA still untouched (the direct guard against the column
    14/15 clobber); and that `Shift_amount_cntx1`/`cntx2` do *not* leak into
    `ColShift` the way they do into `RowStride`, so the same three fields mean
    two different things either side of `Tileize_mode`.
  - *Still to do:* actually model the drop, if a kernel ever sets `ColShift` —
    at which point it will need silicon or a fixed ttsim to validate against,
    not the pseudocode alone.
- ~~★ **Blackhole SFPU computes an out-of-range Dst row on a bfloat16 Dst.**~~
  **Fixed.** Two independent Blackhole bugs, both on the default bf16-Dst SFPU
  path and both in `backends/vector.py`. The config table was *not* at fault —
  regenerating `tensix_backend_cfg_blackhole.yaml` from ttsim's
  `data/bh/tensix_regs.json` diffs clean on every field.
  1. `SFPLOAD`/`SFPSTORE` `dest_reg_addr` is the ISA's `imm10` — bits **9:0** on
     both arches — but `tensix_instructions.yaml` only stores each field's start
     bit and `TensixInstructionDecoder` infers the width from the *next* field's,
     so it decoded bits 13:0. On Blackhole bit 13 belongs to the 3-bit
     `sfpu_addr_mode` (15:13; Wormhole's is 15:14), and sfpi emits addr mode 3,
     so every SFPU Dst access leaked `0x2000` into the row address.
     `_read_dest_reg_addr` now masks to the documented 10 bits.
  2. Behind that crash, `SFPGT`/`SFPLE` ignored `instr_mod1`: mod1 1 sets the
     lane flags, mod1 8 writes an all-ones/all-zero **mask into VD**. Blackhole's
     `exp_tile` (`_sfpu_exp_21f_bf16_`) masks the integer part with that mask
     before `SFPSETEXP` instead of clamping with an `SFPSWAP`, so ignoring it
     left the exponent unmasked and `exp` returned ~2⁻¹²²-scale garbage for
     every result below 2.0. Comparisons now use the sign-magnitude total order
     (ttsim's `sign_mag32_total_order`) and any other modifier raises.
  All four upstream SFPU examples now pass on Blackhole (`sfpu_eltwise_chain`
  PCC 0.99985, inside the oracle's band).
  **Why the differential suite missed it:** the only Blackhole SFPU op test
  (`optests/sfpumath`) runs with `fp32_dest_acc_en = true` and takes the other
  branch, so the *default* bf16-Dst path had no coverage at all.
  - *Test:* `optests/sfpuchain` — five ops, one output tile each, so the failing
    tile names the op; PASSes bit-exactly on both arches and is frozen as
    `driver/blackhole/server/sfpuchain_replay_test.py` against the ttsim golden.
- **Matrix unit undefined-behaviour combinations** — `backends/matrix.py`
  raises for the `MOVB2D` / `MOVD2A` format+Dst-width combinations the
  ISA documents as undefined (and ttsim declines), for `GMPOOL` argmax /
  non-16×16 / INT8 math, and for one data-format match-default. These
  are deliberate refusals, not gaps: there is no reference to model them
  against. Genuinely missing: the BF8/FP4-style input formats for the
  elementwise path.
  - *Test:* variants of `four/` — `four-bf8/` and `four-fp4/` — running
    ELWADD with the missing input/output dataformats end-to-end.
- **Vector / SFPU lane-format edge cases** — `backends/vector.py` still
  has match-defaults that raise in the `SFPLOAD` / `SFPSTORE` 16-bit
  format handling, and `SFPMOV` with the `FROM_SPECIAL` mod is
  unsupported.
  - *Test:* variant of `five-fp/` exercising the lane formats that
    currently raise (extend the existing `fp16` / `tf32` sub-directories
    with the remaining unsupported widths).
- **Packer config variants** — nine `NotImplementedError` sites in
  `backends/packer.py` (ADC context, dest-addr offsets, the per-datum
  edge mode). The **reduce edge mask** (`PCK_EDGE_OFFSET_SEC*` +
  `TILE_ROW_SET_MAPPING`) has since been modelled — it was what zeroed
  the untouched rows/columns of a reduction's output tile — so
  `optests/reduce` covers that half.
  - *Test:* small standalone op test driving a PACK with a non-zero ADC
    context and non-zero dest-address offset (e.g. packing a sub-region
    of `dst` into an L1 buffer at an arbitrary alignment); check the
    packed bytes against ttsim via `optests/diff.sh`.
- **Unpacker NoOp modes & src-to-zero** — four `NotImplementedError`
  sites left in `backends/unpacker.py`. Blackhole's `UNPACR_NOP` is
  fully re-implemented (its field layout differs entirely from
  Wormhole's mode-select: `set_dvalid` hands the Src bank to the matrix
  unit, `stall_and_clear` waits and then zeroes it) — that is what
  unblocked all FP32 compute on Blackhole, and the haloize (transpose)
  path now fires, so `optests/transpose` covers it.
  - *Test:* unpacker micro op test that prefills srcA via a UNPACR NoOp
    mode not yet modelled, then runs a UNPACR src-to-zero clear, and
    confirms downstream FPU sees zeroed inputs.
- **ThCon stalls** — Tensix-coprocessor-side atomic ops (`ATCAS`,
  `ATSWAP`, `ATINCGET`) issue but stall resolution is incomplete: five
  `NotImplementedError` sites in `backends/thcon.py`. Note this is a
  separate code path from the NoC-level atomic-add that landed in §C —
  that one covers `noc_semaphore_inc` (cross-core L1 semaphore
  signalling over NoC), this one covers the Tensix coprocessor's own
  atomic unit operating on its private state. (A related latch
  deadlock in the wait gate — `mutex_stall` set before the `ATGETM` was
  accepted — was found and fixed by `optests/optest`.)
  - *Test:* two-thread sync kernel — TRISC0 spins on `ATCAS` for a flag
    that TRISC1 flips via `ATSWAP`, with `ATINCGET` used to hand out
    monotonic indices. Driver asserts no spurious early wakeups.
- **Mover region-crossing** — single move limited to 16 KB and one
  region; two `NotImplementedError` sites in `backends/mover.py`.
  Unmapped L0 writes silently dropped.
  - *Test:* MOVER kernel issuing a 32 KB copy and a second copy that
    crosses an L0 region boundary; verify destination matches source
    byte-for-byte and that an unmapped destination raises (rather than
    silently dropping).

### Tensix instructions known-missing or stubbed
Much of the original list has landed. **Implemented since:** `TRNSPSRCB`,
`MOVB2D`, `MOVB2A`, `MOVD2A`, `MOVD2B`, `GMPOOL`, `GAPOOL`, `DOTPV`,
plus the SFPU's `SFPCAST`, `SFPDIVP2`, `SFPLUT`, `SFPLUTFP32`,
`SFPSWAP`, `SFPSHFT2`, `SFPTRANSP`, `SFP_STOCH_RND`, `SFPARECIP`,
`SFPGT`, `SFPLE`, `SFPMUL24`, and the Blackhole `CFGSHIFTMASK`. The
Blackhole-only `MOVDBGB2D`, `RESOURCEDECL`, `STREAMWAIT` and
`STREAMWRCFG` decode but are **rejected loudly** — ttsim marks all four
unsupported, so there is nothing to port and a silent no-op would be the
worse failure.

Still missing:

- `MOVDBGA2D`, `SHIFTXA`, `SHIFTXB` — no handlers.
  - *Test:* a column-rotation op test calling `SHIFTXA`/`SHIFTXB`,
    diffed against ttsim.
- `SFPLZ` — decode-only, no handler in `backends/vector.py`.
  - *Test:* per-op micro kernel modelled on `optests/sfpumath`,
    asserting the lreg result against ttsim.
- `SFPLOADMACRO` — **deliberately unimplemented.** Its macro-scheduling
  semantics have no published functional model and ttsim declines it
  outright; tt-sim has no sub-unit or deferred-issue notion to port it
  onto. It raises naming `TT_METAL_DISABLE_SFPLOADMACRO=1`, which is the
  practical workaround (`optests/where` uses it and passes bit-exactly,
  so `where_tile`'s non-macro lowering *is* covered).
  - *Test:* blocked on §I/§H — a real model needs the deferred-issue
    machinery Phase 5 of the event-driven pump introduces.

### tt-metal `programming_examples/` via slow-dispatch
Run the canonical examples through tt-sim's wire bridge with:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_SIMULATOR=$(pwd)/driver/wormhole \
TT_METAL_HOME=/path/to/tt-metal \
[TT_SIM_TENSIX_COORDS=1-1,...] \
  $TT_METAL_HOME/build_Release/programming_examples/metal_example_<name>
```

Last recorded full sweep: 13 of 14 pass. That sweep predates the whole
§H differential round and the §L optimisations, so it needs re-running
before it is quoted again — several of the compute-path bugs it
surfaced have since been fixed, and the wall clock has moved by ~4.5× on
the matmul path.

Outstanding failure:

- **`metal_example_matmul_single_core`** — times out on its
  ``640 × 640 × 640`` bf16 matmul. Not a correctness gap; runs on a
  single Tensix tile, so the §A threading work does **not** unblock
  this one (the composite clock falls into the sequential path with
  one heavy tile). §L has since **quantified** it: 640³ is 512,000
  MVMULs, ~125× the work in `examples/six`, which at the measured
  baseline was ~1.5–2 hours; scaling `six`'s post-optimisation 7.9 s
  by the same factor still puts it at ~15–20 minutes. A one-minute run
  needs ~75× off the *baseline*, which is above
  even the idle single-tile pump ceiling — so the levers remain
  scaling the workload down (e.g. 64³) or a step change in per-cycle
  cost, not incremental tuning.

`metal_example_vecadd_multi_core` is also untested — it uses
`compute_with_storage_grid_size()` which is the full 8×8 worker grid
on Wormhole (64 cores). Materialisable; this one **is** the case §A
targets, gated on the §A perf follow-ups (heavy-only worker pool +
real-workload validation) landing.

### Numerical behaviour
This is the item the differential harness changed most. It used to read
"all compute is FP32 internally with conversion at the boundary". That
is **no longer true**, and the difference is the reason `optests/` match
ttsim bit-exactly rather than approximately:

- **The FPU accumulate is an exact fixed-point datapath**, ported from
  ttsim's C: products truncated to ~12 bits, aligned within groups of 8
  lanes, rounded into Dst *every instruction*. Gated on the 8-bit-exponent
  formats (BF16/TF32) and enabled on **both** arches, with Wormhole's
  documented `-1` renormalisation quirk modelled as
  `negOneRenormBug=True`. Pinned by `pe/tensix/fpu_accumulate_test.py`
  against vectors generated from ttsim's C model for both
  `TT_ARCH_VERSION` values.
- **The SFPU has a per-arch bit-exact FMA** (`fma_model_wh` /
  `fma_model_bh`, both verbatim ports fuzz-matched against ttsim's C over
  200 k random triples, ~20 k of them arch-differing) and a **uniform
  uint32 LReg model** — lanes hold bit patterns, never Python floats.
  The old mixed float/int model destroyed signalling-NaN payloads and
  was what made `tanh`/`sigmoid` return exactly −1.0 on Wormhole.
- **Blackhole implies the SrcA/SrcB operand format** from the format the
  unpacker last wrote the bank in, rather than reading
  `ALU_FORMAT_SPEC_REG0_SrcA`, matching ttsim's `TT_ARCH_VERSION >= 1`
  behaviour.
- **Dst row-validity bits** are modelled (a flag-cleared row reads back
  as all-ones for `GMPOOL` only, as ttsim does).

Still not modelled: rounding-mode overrides, per-thread `FP16A_FORCE`
enforcement (parsed, not applied), accumulator-persistence, and the
32-bit sibling of the ZEROACC bank fixup (deliberately left alone — the
`useDst32b` row mapping still diverges from ttsim's swizzle, so half the
fix would mislead).

- *Test:* `optests/` + `optests/diff.sh` is the standing test for all of
  the above, on both arches; the unit tests named above pin the ports.

---

## E. Device / tile (`tt_sim/device/`, `tt_sim/misc/`)

- **Command-queue (fast) dispatch flow** (dispatch via dedicated Tensix
  tiles) is unsupported; only the **direct** `LaunchProgram` flow works.
  Under fast dispatch any `EnqueueProgram`-driven example reaches the
  worker tile without the dispatch firmware having initialised L1
  (kernel-config buffer, runtime-args, CB descriptors, NoC bank tables);
  the kernel's `TensorAccessor` then reads garbage as a pointer and the
  load crashes `MemorySpace._locate_memory_space` with an unmapped
  high-MMIO address (e.g. `0xFFAE0062`, which lives in the
  0xFFA00000–0xFFB00000 gap and is not a real Wormhole register).
  **Workaround that exists today:** set
  `TT_METAL_SLOW_DISPATCH_MODE=1` in the host's environment — tt-metal's
  `EnqueueProgram` then internally calls `detail::LaunchProgram` (see
  `tt_metal/impl/dispatch/host_runtime_commands.cpp:258`), and the
  canonical `programming_examples/` flow runs unmodified through tt-sim's
  wire bridge. Validated against `metal_example_eltwise_sfpu` and
  `metal_example_eltwise_binary` — both launch, run kernels to
  completion, and return without crash; the result-correctness bugs they
  surfaced (SFPU exp returning ≈ 1.0, eltwise compute producing 0s) were
  separate compute-path issues, not dispatch-path blockers. Both of
  those symptoms have since been root-caused and fixed by the §H
  differential work (the SFPU value model and the FMA ports — the
  `exp` overflow guard in particular was `SFPIADD`'s unsign-extended
  12-bit immediate), so this pair wants a re-run to confirm.
  - *Test:* tt-metal `programming_examples/matmul/matmul_single_core`
    via the wire bridge with `TT_METAL_SLOW_DISPATCH_MODE=1` —
    structurally unblocked, gated only on §D matrix/compute fixes.
    Native fast-dispatch (the multi-Tensix dispatch-firmware path) is
    still out of scope.
- **Soft-reset sequencing.** Asserts/deasserts take effect immediately
  on the next clock tick; the ISA docs describe a multi-step sequence
  that's not modelled.
  - *Test:* driver-only script under `driver/simple/` that drives the
    documented multi-step soft-reset sequence and asserts the
    intermediate observable states (per-core PC, mailbox values) at
    each step.
- **Tile-control registers.** *Downgraded, not closed.*
  `misc/tile_ctrl.py` no longer raises: `SOFT_RESET`,
  `TRISC_PC_BUF_OVERRIDE`, the wall clock and `DBG_FEATURE_DISABLE` have
  real behaviour, and every other offset is a generic
  read-what-you-wrote store (needed by Blackhole's `NOC_CFG` /
  clock-gating blocks). So an unmodelled register is now silently
  plausible rather than loud — which is the right default for boot
  traffic, but means a register with real side effects would go
  unnoticed.
  - *Test:* driver-only script reading/writing each tile-ctrl offset
    that has documented side effects and asserting them, rather than
    the current round-trip.
- **PC buffer.** `pe/pcbuf.py` `TODO` — write delays / synchronisation
  not modelled.
  - *Test:* no example needed — perf/timing modelling.
  - *Fixed alongside:* the **blocking** words were decoded one word too
    high. `misc/ttsync.py` is mapped at `0xFFE80004`, so its offset `0x0`
    is `pc_buf_base[1]` — what `ckernel::tensix_sync()` reads — and that
    returned zero unconditionally, with the Tensix-FIFO-drain check
    sitting on `pc_buf_base[2]` (`mop_sync()`'s word) and the MOP check
    on the unused word 3. `tensix_sync()` was therefore a **no-op**
    against tt-sim and could not order anything: measured on the `six`
    replay, five reads, zero blocked cycles. Corrected against
    `tt_llk_*/common/inc/ckernel.h` and ttsim's `tensix_pc_buf_rd32`;
    the same five reads now block for 8,043 cycles. Nothing moved:
    776 tests green and `cost_model_gate.py` output byte-identical to
    the pre-change baseline.
- **Extra core types.** `device/tt_device.py` raises on anything beyond
  BRISC/NCRISC/TRISC0–2; needs widening if ERISC ever lands as a
  bridge-visible core type.
  - *Test:* covered by the §F ERISC example.
- **Per-architecture constructor drift.** *Closed structurally, worth
  remembering.* A device-level facility wired in one architecture's
  `__init__` and not the other's fails **silently**, and it happened
  twice: structured tracing (`enable_from_env`) was Wormhole-only, so
  Blackhole ignored every `TT_SIM_TRACE_*` var; and the deadlock
  watchdog was Wormhole-only, so a wedged Blackhole kernel hung with no
  `[DEADLOCK]` report at all — on the architecture under the most active
  development, and one measurement in
  `docs/upstream-examples-status.md` was drawn from an A/B where neither
  arm had a watchdog because of it. `TT_Device` now owns the whole
  construction sequence (`_begin_construction` → arch tiles →
  `__init__`'s registries + watchdog + tracing) and an architecture
  supplies only hardware facts (`tensix_tile_class` / `dram_tile_class`
  / `_tensix_physical_coord` / the `ArchProfile`). The same pattern was
  live one layer up in the server entry points, where the
  "worker not materialised" warning and the fatal go=GO guard were
  Wormhole-only; they are now `tt_sim.bridge.install_worker_guards`.
  - *Test:* `tt_sim/device/parity_test.py` (facility parity, including
    an attribute-set diff of the two constructors and a ban on an
    architecture importing a facility symbol), plus every behavioural
    watchdog test in `tt_sim/device/deadlock_test.py` parametrised over
    both architectures. The remaining deliberate asymmetry is
    Wormhole's ethernet tiles (§F).
- ~~**The deadlock watchdog cannot see a *spinning* wedge.**~~ **Fixed.**
  It was built to catch a device that had gone quiet, and every real
  wedge found so far is the opposite: cores retiring instructions in a
  poll loop at full speed, no forward progress, 95 % CPU. The progress
  signature used each core's *instantaneous* PC bucket, which any spin
  loop straddling a 64-byte boundary defeats — consecutive samples land
  either side of it, the signature "changes", and the stall window
  re-baselines for ever. That is what a wedged tt-metal firmware looks
  like (the go-message wait loop spans two buckets), and it is why the
  two-`LaunchProgram` hang (§A) ran for 25 minutes with the watchdog
  enabled and silent. The signature now carries each core's *code
  footprint* — the set of PC buckets in its recent-PC window — beside its
  **register file**, and the per-cycle confirmation pass asks whether any
  core reached a bucket it had not already been in rather than whether
  its PC moved (a spinning PC moves every cycle). The register component
  is what stops the footprint rule firing on a small loop that is
  genuinely computing: a working loop moves a register every iteration, a
  poll loop re-loads the same value for ever. Verified quiet across all
  37 replay guards and the cost-model gate. The remaining limit, stated
  in the module docstring: a loop longer than the confirmation window
  keeps reaching new buckets to the end of the burst and still reads as
  progress.
  - *Test:* `tt_sim/device/deadlock_test.py` —
    `test_fires_on_a_spinning_wedge_that_straddles_a_pc_bucket` against
    `test_quiet_on_a_tight_loop_that_is_making_progress`, the same loop
    shape with and without register motion.
- ~~**The deadlock watchdog cannot see a fully dormant device.**~~
  **Fixed.** Making the watchdog sampled rather than per-cycle (§L) left
  it dependent on the Phase 4 pump *visiting* a cycle, and a device every
  tile of which reports dormant is strided over in one jump per `run()`
  call — so a 1,000,000-cycle `run` took exactly one sample and a wedged
  but dormant device went unreported indefinitely (measured: 400,000
  cycles of a wedged BRISC, zero `[DEADLOCK]` lines). It was argued safe
  because dormancy implies every baby core is in soft reset
  (`TensixTile.next_wake_cycle` short-circuits to `cycle + 1` for any
  `soft_active` core), which is the one state the watchdog deliberately
  ignores — true today, but that is an invariant in *another* module
  holding up this one's latency bound, and `busy_until` / NoC-arrival
  deadlines are actively adding new ways to name a far-future wake.
  `DeadlockDetector.next_sample_cycle` is now handed to `MultiTileClock`
  as `on_tick_wake` and joins the stride computation exactly as a tile
  clock's `next_event_cycle` does, so the pump cannot jump past a
  scheduled sample. **Latency is unchanged** — still `threshold` to
  `threshold + threshold//8 + 64` cycles — and it is now a property of
  the detector alone. Two options were rejected: flooring the stride at
  the sample interval (same effect, but spends the budget on every
  consumer of the pump rather than the one that asked); and treating full
  dormancy as itself a deadlock signal, which is not provable — under the
  wire bridge the host writes between `run` calls, and "every core in
  soft reset" is the ordinary pre-launch and post-completion state.
  - *Cost:* one probe call per **stride decision**, and a stride decision
    is only reached when no Tensix tile wants the next cycle. A live
    workload therefore pays nothing at all (`one` offline replay: 1.27 s
    vs 1.30 s, min of 3, both arms 126/126 byte-identical). A *dormant*
    80-worker device goes from 0.008 to 0.041 µs/cycle — the 0.033 µs
    delta being the amortised signature scan (218.5 µs / 6,250) the
    watchdog was already documented as paying while awake, against a
    ~120 µs/cycle live floor. `TT_SIM_DEADLOCK=0` leaves `on_tick_wake`
    unwired along with `on_tick`, so disabling it still costs nothing.
  - *Test:* `tt_sim/device/deadlock_test.py` — fires on a fully dormant
    wedged device inside the same latency bound, is silent on the same
    device with the probe unwired (so the guard cannot go vacuous), and
    samples once per interval while striding; plus
    `tt_sim/device/clock_test.py`, which pins the pump's stop count at
    one per sample interval and none at all with the watchdog off.

---

## F. Missing tile classes (whole subsystems absent)

These are present in the ISA docs but have no (or only partial) code
under `tt_sim/`. Blackhole adds two more that are **deliberately out of
scope** per `docs/plans/blackhole-support.md`'s scope decisions: the
**L2CPU tiles** (4 × 4 SiFive x280 64-bit cores — would pull in a 64-bit
RISC-V model, unrelated to the Tensix compute path) and the **Security
tile**.

- **Ethernet tiles (ERISC).** Soc descriptor lists 16 eth coords. L1
  SRAM + ERisc baby core both landed: `EthTile` (256 KB L1, one
  RV32IM ERisc with 4 KB local mem + 64 KB IRAM, soft-reset bit 11
  in the tile's own `RISCV_DEBUG_REG_SOFT_RESET_0` per the WormholeB0
  EthernetTile docs) is instantiated for all 16 descriptor entries,
  registered in both NoC directories under its physical coord, and
  reachable through the wire bridge via `EthCore`. A driver script
  can load RV32IM code into eth L1, deassert ERisc reset, and run it.
  Single-chip kernels that hardcode an eth coord
  (e.g. `programming_examples/hello_world_datatypes_kernel` reads
  `(1, 0)`) get deterministic memory-backed state instead of the
  former `NullEndpoint` zero-fill. Still missing: chip-to-chip
  ethernet routing (**blocked** — no second chip exists, and modelling
  one is a larger scope decision than a tile class); and **Blackhole
  eth tiles**, whose 2-core / 512 KB variant is deliberately deferred
  until an eth kernel needs it, so BH eth coords still fall to
  `NullCore`.
  - *Test:* tt-metal `programming_examples/hello_world_datatypes_kernel`
    (single-chip eth read at `(1,0)`) — structurally unblocked, gated
    on a re-run through the wire bridge to confirm; plus a multi-chip
    eth-ping example as a new `examples/ten/` once chip-to-chip routing
    lands.
- **ARC tile / chip management.** Power state, telemetry, MSG channel —
  none modelled. tt-metal's `arc_msg` stub at the wire layer just
  returns 0.
  - *Test:* small driver script under `driver/wormhole/server/` that
    issues an `arc_msg` (e.g. telemetry/temperature query) and asserts
    a non-zero / documented response shape rather than the current
    blanket `0`.
- **PCIe tile.** Listed in soc descriptor (`pcie: [0-3]`); no host-side
  PCIe simulation. Not needed today because the simulator wire is the
  host channel.
  - *Test:* no example needed — superseded by the wire bridge.

---

## G. Observability

Current state — the gaps the §H plan is designed to close. (§H itself
has since shipped the event bus + eight writers, so the second and
fourth bullets below describe the *pre-§H* baseline that the inline
`print()` path still shares.)

- No waveform dump.
- No structured event log — diagnostics are inline `print()` to stderr,
  filtered by per-component snoop flags.
- Tensix backend diagnostics partial (issued instructions, FPU/SFPU
  calcs logged; full operation coverage incomplete).
- No cycle-accurate event export for external tooling.

→ Strategy and phasing for fixing all of the above lives in §H below.

---

## H. Tracing & instrumentation infrastructure

The event bus + seven writers are in production today. A single
opt-in env var turns each writer on; all eight can run simultaneously.
Validated on every wormhole example, and arch-agnostic — a Blackhole
device registers its tiles' trace unit-ids exactly as Wormhole does.
Bus publish overhead is **88 ns/call disabled** (target was <100 ns).

| Writer | Format | Env var |
|---|---|---|
| Event log | JSONL | `TT_SIM_TRACE` |
| Perfetto / Chrome Trace Event | JSON(.gz) with NoC flow arrows | `TT_SIM_TRACE_PERFETTO` |
| RISC-V commitlog | Spike `--log-commits` format + `diff_spike` helper | `TT_SIM_TRACE_COMMITLOG` |
| Performance counters | Hive-partitioned Parquet + canned DuckDB queries | `TT_SIM_TRACE_COUNTERS` |
| NoC transactions | Parquet, partitioned by chip | `TT_SIM_TRACE_NOC` |
| Memory hotspots | Cachegrind text for KCachegrind | `TT_SIM_TRACE_MEMORY` |
| Source coverage | LCOV via DWARF (`pyelftools`) | `TT_SIM_TRACE_LCOV` |
| Invariants + state dumps | JSONL + `diff_state` helper | `TT_SIM_TRACE_INVARIANTS`, `TT_SIM_TRACE_STATE_DUMP` |

### Differential testing against ttsim — **landed**

The workstream §Positioning lists as defensible lane 4 exists and is the
project's strongest correctness asset. `optests/diff.sh <name>` runs the
**same compiled tt-metal program** through tt-sim and through ttsim
(`libttsim_{bh,wh}.so`) and diffs the hex the program dumps
(`OPDIFF_RESULT:`). Because both sides run the same JIT'd kernel, the
check is immune to tt-metal C++→asm churn and ttsim is a true oracle.
`TT_SIM_ARCH` (default `blackhole`) switches both sides to Wormhole,
picking the oracle library, the driver and the default worker coord.

| | programs matching ttsim bit-exactly (swept 2026-08-03) |
|---|---|
| **Blackhole** | **all 13**: `optest` (Int32 tile path), `tilize`, `untilize`, `transpose`, `reduce`, `reduceneg` (bf16 pool), `sfpuchain`, `sfpumath` (FP32 recip/tanh/sigmoid), `softplus`, `where`, `matmulblock`, `matmulidx`, `dramtop` |
| **Wormhole** | **all 11 ttsim-WH will run**: `tilize`, `transpose`, `reduce`, `reduceneg`, `sfpuchain`, `sfpumath`, `softplus`, `matmulblock`, `matmulidx`, `dramtop`, `untilize`. `optest` and `where` are the Int32-tile pair ttsim-WH rejects up front (`unpack_to_dst=0 in_data_format=8`), so there is nothing to diff against |

**There is no live differential miss left on either architecture.** The last one
was `optests/untilize` **op2** (`pack_untilize_dest`), at 992/1024 elements on
Blackhole; it is now bit-exact and frozen in both arches' guards
(`CHECKED_OPS = [0, 1, 2]`). On Blackhole that op lowers to a MOP that runs one
packer interface *pair* in `DST_ACCESS_STRIDED_MODE`, where read interface `k`
takes its 16 datums from Dst row `base + 16*k` rather than `base + k` — the
same row of successive faces, so one PACR emits one 32-datum row of the
row-major output. Two decode gaps had to close, both in the shared
Wormhole-layout PACR argument table: `dst_access_mode` (raw bit 17) had no
field at all, and `AddrMode`, being the table's last argument, was given every
bit up to the opcode (23:15) and swallowed it — so a strided `ADDR_MOD_1`
resolved to a section no LLK programs and the pack Y counter never advanced,
replicating one output row down the whole tile. `DEST_ACCESS_CFG` writes now
also drive `DstRegister`'s long-dormant `Adj16`/`Adj32` gates. See
`docs/plans/blackhole-support.md`.

A passing differential run is then **frozen as an offline replay guard**
that runs with no oracle, no tt-metal and no socket — one per op test on each
arch (`driver/{blackhole,wormhole}/server/<name>_replay_test.py`). The Wormhole
goldens are ttsim-*Wormhole*'s own dumps, and are *not* the Blackhole ones,
since the two chips' FMAs differ. Where there is a closed-form
answer the guard checks that instead; where there isn't, the frozen
oracle dump is the golden.

Verified 2026-08-03 against `5bfb476` (plus concurrent unrelated work in
the tree): **all 24 Blackhole replay guards pass**, the Wormhole
`offline_replay_test` reproduces **126/126** READs bit-for-bit, `python3 -m
pytest tt_sim driver -q` is green (559 passed then, 620 after the observability
work below and concurrent §I work landed — the count moves as concurrent work
lands; treat "green", not the number, as the gate), `python3 -m pytest
driver/wormhole/server/ -q` is 20 passed, and `python3 -m
driver.tests.cost_model_gate` is PASS.

Open here:

- Nothing: every op test that runs on either arch matches ttsim bit-exactly.
- Extending op coverage is now incremental: add a program under
  `optests/<name>/src`, diff it, fix what tt-sim gets wrong, freeze.

### Follow-ups (grouped by what gates them)

**Unblocked by §I and landed (2026-08-03)** — the observability payoff
for the performance workstream. Every field below reads state §I
already produces; none of it is new modelling, and none of it moves a
simulated cycle:

- **Real Perfetto durations.** A `compute` slice is as wide as the
  occupancy `tensix_instruction_costs.yaml` charged
  (`ComputeEvent.duration`, from the `occupancy` the backend already
  computes at issue); a `noc` slice spans issue → arrival; a
  `stall:<reason>` slice is as wide as the cycles the RV load-use
  interlock held the instruction that follows it, abutting it exactly.
  `instr` and `dispatch` stay one cycle wide in both regimes — a core
  retires one instruction per cycle and an issue is a single-cycle act.
- **NoC per-transaction `issue_cycle` / `arrival_cycle` /
  `flight_cycles`**, in `NoCEvent` and as Parquet columns.
  `NUI.transmit` stamps the sending cycle on the packet; the event's
  own cycle is the arrival. Real in both regimes: with the model off a
  flight is genuinely the one cycle the two-list swap costs.
- **RV stall attribution**, as `stall_cycles` + `stall_<reason>`
  counters per baby core, from a per-instruction drain of
  `RiscvCostState`. Blackhole `four`: **80,695 stall cycles, 80,570 of
  them `load_use`** — the totals `stall_by_reason` already held, now
  attributable to the instruction and the cycle.
- **Per-unit `busy_cycles`**, from the same `ComputeEvent.duration`;
  against the run length that is a backend unit's utilisation.

**The regime is in the output, not implied.** With `TT_SIM_COST_MODEL`
unset every duration is 1 and every stall counter is absent — which is
the truthful rendering, because nothing stalls and every unit retires
in its issue tick, not a degraded one. What the writers refuse to do is
fabricate a plausible width. Perfetto states which regime a trace came
from three ways (`otherData.cost_model`, a per-tile `process_labels`,
and a `timing_model` argument on any slice whose width is not
modelled); the NoC Parquet dataset carries a `cost_model` column; the
counters encode it as absence rather than as zeroed rows.

NoC slices moved from `X` to **async** (`ph` `b`/`e`) in the process,
because a NIU can have several packets in flight and partially
overlapping `X` slices are not legally nestable. The request→response
arrows moved onto the slices as `bind_id` + `flow_out`/`flow_in`; a
standalone `s`/`f` flow event binds to a *thread* track's enclosing
slice, which an async slice is not.

**Still gated on §I** — not because a writer lacks a field but because
the simulator holds no such state:

- **NoC `vc` and VC occupancy.** No virtual channels are modelled.
  Deliberately no column rather than a column of zeroes.
- **Packer back-pressure and unpacker idle cycles.** The packer is
  charged its documented `PACR` *issue* cost only (the drain has no
  published figure) and the unpacker is not wired to the tables at all
  — see `UNWIRED_UNITS` in `tt_sim/perf/costs_test.py`. `busy_cycles`
  therefore reports these units at ~0, correctly.
- **L1 bank conflict count.** tt-sim models L1 as flat memory; there
  are no banks to conflict.
- **FPU/SFPU *stalled* cycles.** The active half landed as
  `busy_cycles`; the stalled half needs a refused issue to be
  observable, and §I's own measurement is that refusals are ~0 today
  (of `six`'s 416 occupied ThCon ticks, none had an instruction
  waiting) because the constraint is the un-modelled RISC-V front end.
  Worth a `DispatchEvent` field once a unit is ever actually contended.

**Multi-Tensix attribution (partly unblocked by §A)** — `chip_id` is
still hard-coded to 0, but per-tile `core_y/core_x` already fan out:
`TensixTile._register_trace_ids` uses the tile's own coord, so the
two-tile `examples/nine/` carries the right unit IDs for each tile.
Multi-chip identity is the remaining gap.

**Standalone (any time, not blocking):**

- **Inline-print migration.** Existing `if self.snoop: print(...)`
  sites still run alongside bus publish. Routing
  `DeviceTileDiagnostics` flags through bus subscribers is a
  separate refactor.
- **VS Code coverage extension** (Phase 6 stretch — rendering cycle
  hotspots as inline source decorations beyond what Coverage Gutters
  gives).
- **`DwarfIndex` function-name attribution** via DIE traversal.
  Today the index is `(file, line)` only.
- **Auto-discovery of kernel ELFs** from the tt-metal build tree.
  Today the user provides `TT_SIM_TRACE_LCOV_ELFS=` explicitly.
- ~~**`libttsim.so` differential testing harness.**~~ **Done** — see
  "Differential testing against ttsim" above. Phase 7 shipped the
  comparison primitive (`diff_state`); `optests/diff.sh` is the
  orchestration, and it compares the program's own DRAM dump rather
  than simulator state, which is what made it portable across the two
  simulators.
- **tt-metal issue 28562 invariant catalogue.** Phase 7 ships four
  seed invariants; mining the linked issue tracker for
  domain-specific architectural rules would expand the catalogue
  substantially.
- **`hypothesis` property tests.** Needs a kernel generator —
  separate research project.
- **Determinism: tighten the host polling loop.** State dumps at
  `kernel_done` vary by up to 100 cycles per run because the bridge pumps
  `cycles_per_poll` (default 100) cycles per wire message
  (`tt_sim/bridge/device.py`). Byte-exact regression comparison wants a
  tighter boundary. Note §I Phase 4 (time skipping) changes what a cycle
  boundary means here, so the two want designing together.
- **L1 / DRAM content snapshotting** in state dumps. Today only
  registers + NoC counters. Add when a consumer needs it.
- **CI examples.** A fixed workload runs in CI and produces all
  eight output formats — catches schema regressions early.
- **Trace replay.** Re-run any writer offline against a captured
  JSONL stream, no re-simulation needed. Decouples writers from sim
  runs.
- ~~**Aggregate performance-budget audit.**~~ **Measured** — see
  profiling.md "Tracing overhead". The documented ~30 % target is not
  met: counters and Perfetto cost **~2× on RV-bound workloads** (1.27×
  on the matmul, where the time is inside one Tensix op), and JSONL
  costs 3.7–4.0×. `TT_SIM_TRACE_COUNTERS_INTERVAL` is **not** a lever —
  the cost is `EventBus.enabled` turning publication on everywhere, not
  the flush cadence. Counters are the only writer plausibly usable at
  kernel scale (40 KB of Parquet for a run whose JSONL is 61 MB). Open:
  decide whether to re-target the budget or to make publication cheaper
  at the call site. **Re-measured 2026-08-03** when the cycle-attributing
  fields above landed, as a frozen-tree interleaved A/B (min of 5,
  Blackhole `four`): **0.95× with tracing off** — i.e. nothing
  measurable, the added work all sits inside `if trace_instr:` and the
  writers — **0.96× with Perfetto on**, and **1.07–1.13× with counters
  on**, riding along inside the 2–4× that was already being paid. The
  one thing that did cost was output size, and it was caught: a constant
  `"stall_cycles": 0` on every instruction slice added **+9 MB to a
  66 MB Perfetto trace** for a run where nothing can stall; emitting it
  only when non-zero brings the file back to +0.03 % and the wall clock
  from 1.09× to 0.96×.
- **Pyright type-cleanup** in the new modules. Pre-existing
  strict-mode warnings shared with the rest of the codebase.
- **Register-file accesses as `MemEvent`** — explicitly skipped
  today (volume too high vs. signal). Reintroduce if a consumer
  needs it.
- **`SpikeCommitlogWriter` enhancements:** memory-access decoration
  (`spike --log-commits --log-mem` equivalent), end-to-end "run both
  ELFs" mode.
- **Jupyter notebook template** for the Parquet counter dataset
  (Phase 4 nice-to-have).
- **NoC heatmap example** — a small `networkx` + matplotlib script
  consuming the NoC Parquet dataset (Phase 5 nice-to-have).

---

## I. Performance modelling (cycle-approximate)

**Headline goal** post-ttsim (see Positioning). The simulator is
functional, not cycle-accurate — every instruction retires in one tick,
NoC requests are accepted immediately, and there is no back-pressure
(per CLAUDE.md "The point is hackability"). Closing that gap is the
defining work for tt-sim's new lane: a first-order performance
estimator complementary to ttsim's bit-exact functional role.

Be explicit in user-facing docs that this is *cycle-approximate*, not
cycle-accurate. Cycle-accurate (matching silicon within ~%) needs RTL
or captured silicon traces to calibrate against — see the "Calibration
against silicon traces" bullet below. Until that data exists, the model
targets order-of-magnitude correctness on stalls, back-pressure, and
contention, not silicon-matching cycle counts.

The sub-items below appear elsewhere flagged as "perf-level"; collected
here so the strategy can be picked at one go rather than piecemeal.

The **event-driven cycle pump** is the substrate all of them plug into:
an instruction can only take more than one cycle once there is a queue
keyed by integer cycle counts to schedule its retirement on. The phased
design is
[`docs/plans/event-driven-pump.md`](docs/plans/event-driven-pump.md);
status there:

- **Phase 1 (tile-level dormancy) — landed.** A tile with nothing to do
  is no longer ticked. The idle floor improved **13.3× at one Tensix
  tile and 32.2× at eight**, and — the part that matters for §I — it
  stopped being linear in *component* count and is now flat at 0.19 µs
  per tile-clock. On real workloads that is a constant ~13 µs saved per
  simulated cycle (1.05× on the matmul, 1.16–1.45× on dataflow-bound
  runs). Tick *order* is unchanged, which is what let it be validated by
  byte-identical replay.
- **Phase 2 (lazy wall clock)** — next-but-one; removes the last
  per-cycle cost of a dormant tile.
- **Phase 3 (per-component gating inside a live tile) — deliberately
  deferred** (decided 2026-08-02). Phase 1's instrumentation killed its
  rationale: every DRAM tile sleeps ~100 % of a run but *no* Tensix tile
  ever sleeps (BRISC spins in the firmware loop from launch to
  teardown), so the remaining win is a few percent of single-tile wall
  clock — bought with the riskiest predicate in the document, whose
  failure mode is silently different numbers rather than a crash. Do not
  pick it up as the next phase.
- **Phase 4 (the integer-cycle event queue, where simulated time is
  allowed to skip)** is the live one, and Phases 4–5 do **not** depend on
  Phase 3. Judge them on whether a cost table can be expressed and a
  stall attributed, not on wall-clock factors.
- **Phase 5 — landed for six of the nine Tensix backend units**
  (2026-08-03; the config unit 2026-08-04). The matrix, vector (SFPU),
  scalar (ThCon), packer, sync and config units each charge an op the
  occupancy `tensix_instruction_costs.yaml` gives it, behind the opt-in
  `TT_SIM_COST_MODEL`; the unpacker, the mover and
  everything outside the coprocessor still retire in the tick they were
  issued. The matrix unit's fidelity-phase cost is computed from the
  table rather than looked up, and comes out at 1 cycle per `MVMUL` at
  every phase — the multiplier is carried by the instruction count, not
  by a longer instruction, so charging it per instruction would
  over-count a HiFi4 tile by 2.5×. The SFPU is the same shape of answer
  from the other direction: a published latency for all 42 opcodes,
  every one of them a *one-cycle occupancy*, because the sub-units are
  pipelined. ThCon is the first multi-cycle instruction in the tree
  (≥ 3 for the GPR and load/store ops, ≥ 15 for `ATCAS`).
  **No simulated cycle count moves**, on any of six guards, measured at
  one-cycle resolution: on `six` (128³ bf16 matmul, 27,500 cycles both
  ways) the five units are charged 5,812 cycles — 21.1 % of the run, up
  from the FPU's 15 % — and of the 416 ticks ThCon spent occupied,
  *none* had an instruction waiting. The units are never contended,
  because the constraint is the un-modelled RISC-V front end.
  The **config unit** was the exception and the interesting result:
  charging `RDCFG`'s documented ">= 2" delayed nine config writes by one
  cycle each and made `matmulblock` compute a wrong answer, so tt-sim
  had a missing ordering guarantee between a config write and its
  readers. Left uncosted rather than papered over — then fixed
  (2026-08-04, the batch-drain bullet below) and **wired**, since
  Blackhole's `CFGSHIFTMASK` is a 2-cycle occupancy the `untilize` guard
  executes 32 times, so it is not the no-op it was assumed to be. The
  cost-model gate passes with every poll-budget multiplier unchanged and
  not one simulated cycle moves. Occupancy is charged **per IPC group**
  rather than per unit (2026-08-04), transcribed from the `IPC group`
  column of Blackhole's Configuration Unit table — the only such column
  published for any Tensix unit — so a `CFGSHIFTMASK` holds `Config` and
  leaves `SETC16` free; the whole-unit over-charge it replaces turned out
  to refuse no issue anywhere in the tree, so `untilize` reads 12,189
  cycles either way. The
  unpacker and mover, NoC bandwidth/contention and DRAM latency are the
  rest of this section.
- **Phase 5, second half — the baby RISC-V load/store path** (2026-08-03),
  and the first change to move a simulated cycle count. The whole
  `riscv` block of `tt_sim/perf/unit_costs.yaml` is `isa_doc`; the
  consumer is `tt_sim/pe/rv/cost.py`, behind the same
  `TT_SIM_COST_MODEL` opt-in and behind one attribute read on the
  interpreter's hot path. The design point is that the **load-latency
  table is not an occupancy table** — the docs say "N − 1 independent
  instructions need to follow the load if the latency is to be entirely
  hidden", so it is spent as a **load-use interlock** (a per-GPR
  scoreboard, keyed by address region) rather than as cycles the core is
  held. That is this section's "memory-stall back-pressure on L1 / NoC
  reads" exactly. Charged as real occupancy: L1 stores at one every five
  cycles (with Blackhole's coalescing queue modelled by the docs' own
  ±4/16-byte predicate), multiply at 2 cycles on Wormhole and 0 on
  Blackhole where it pipelines, divide at the low end of 6–33 with its
  two-cycle special cases.
  **Cycle counts move**, at one-cycle resolution: `six` +7.8 % (27,168
  → 29,291, PCC unmoved at 0.9982), Wormhole `softplus` +6.9 %, the
  other Wormhole guards +1–2.9 %, the other Blackhole guards 0–1.2 %.
  The gap between charged and delivered is the headline finding: `four`
  charges its five cores **76,615 stall cycles** and gets **185 cycles
  longer**, and quadrupling the dominant load latency (Blackhole L1 at
  the ≥ 8 miss instead of the 2-cycle hit) adds a further 0.13 %. These
  runs are not issue-limited — their length is set by cross-core
  handshakes, and what those handshakes wait on is a NoC and a DRAM
  that answer in zero cycles. **So the NoC hop model is now the ranked
  next step**, not more per-unit tables.
  Two further findings. A timing perturbation crashed the Wormhole
  `loopback` replay (`KeyError: 'mutex_index'` — the sync unit walked a
  mixed issue queue); fixed, with a regression test, and a *second*
  sync-unit bug found and deliberately left (the branch tests `"ATGEM"`
  where the table says `ATGETM`, and fixing it changes behaviour with
  the model off) — since **fixed in its own change**, see §D: it moves
  two issue decisions on Blackhole and none on Wormhole, with every
  guard's value unchanged. And **byte-identical replay cannot be the acceptance
  gate for a timing model**: with the model on, 11 Wormhole example
  replays and the Blackhole `offline_replay_test` fail because they
  replay the host's *poll count* verbatim — every one is a single READ
  that resolves within 2,000 further cycles, zero still wrong, while
  the 21 Blackhole value guards (which pump until DONE) all pass.
  Uncharged and named as gaps: branch mispredicts (no predictor
  exists to count them), the NoC NIU register block (the ">= 7" row
  names the NoC *overlay*, a different block), sustained-load
  throughput, and i-cache misses.
- **Phase 5, third instalment — the NoC per-hop latency model**
  (2026-08-03), taken on the ranked evidence above, and the first term
  in the tables that is not a unit cost at all: the answer is a
  function of the distance between two endpoints. The consumer is
  `tt_sim/network/tt_noc.py`, behind the same `TT_SIM_COST_MODEL`
  opt-in; a packet is delayed `(~5 + ~5) + 9 × hops` before the
  destination NIU services it, instead of arriving on the next cycle.
  Hop count is a **forward** distance on each NoC's own directional
  torus — it wraps rather than turning round, it is **not symmetric**
  (a round trip between tiles differing on both axes is always
  `grid_x + grid_y` hops, 22 on Wormhole and 29 on Blackhole), and
  NoC 1 falls out of the same formula because both endpoints are
  mirrored. Both arches share both latency constants; the only per-arch
  difference that reaches a flight time is the torus size.
  **A total finally changes shape**: `six` goes 27,168 → **63,425**
  cycles (+126 % from the hop model alone), driven by just **144 NoC
  transactions** whose 38,520 cycles of flight time are almost all on
  the critical path — the dataflow kernel does a read-barrier per tile,
  so a round trip that cost 2 cycles now costs ~281. `matmulblock`
  +18 %, `matmulidx` / `reduce` +14 %, `three` +13 %, down to +1.3 % on
  `four`. `six`'s PCC is unmoved at 0.9982 and every value guard passes.
  Of `six`'s 63,425 cycles, **~73 % is now attributed to a published
  number** (60.7 % NoC flight, 9.2 % Tensix, ~3 % RV), against 21 %
  before.
  **Rung 1 of the calibration ladder passes, on Wormhole.** Subtracting
  the modelled NoC round trip from tt-metal's four measured end-to-end
  latencies leaves a residual that should be the issuing core's own
  path and therefore identical in all four: it is **36 (local L1), 38
  (remote write), 41 (remote read)** — the hop model explains the whole
  local-vs-remote difference to within 5 cycles, from two sources that
  do not cite each other. The DRAM row leaves ~100 cycles, the first
  constraint of any kind on the term the table calls `unknown`.
  Blackhole does not reproduce it (68 vs 122/123) and is recorded as
  unresolved.
  **Nothing broke that was a value.** Two unit tests with hardcoded
  cycle budgets (`noc_routing_test`'s `run(16)`, `bringup.py`'s
  `run(30)`) now wait on the condition they mean. The gate's report that
  `blackhole/two` was "still wrong after 200,000 cycles — NOT a timing
  artefact" was itself an artefact: `two`'s captured trace ends with
  `CMD_RESET_ASSERT` to every core, so pumping *after* a trace-pumped
  replay can never finish the kernel. Given the cycles a live host would
  have given (`cycles_per_poll` 200 rather than 100) it computes all 100
  elements correctly, and at `cycles_per_poll=400` **all eleven Wormhole
  example replays are byte-identical under the model — zero mismatches,
  not even a late one**, which is a stronger claim than the previous
  instalment could make. Still uncosted at the time: NoC bandwidth (a
  32-byte semaphore poke costs what an 8 KiB tile read costs),
  congestion (`provenance: unknown`), and DRAM — the first and third
  closed later the same day, in the two bullets below; congestion is
  still open and still unsourced.
- **Phase 5, fourth instalment — DRAM access latency** (2026-08-03),
  the term the instalment above ranked next, and the first entry in
  these tables that **no document publishes**. The consumer is
  `tt_sim/device/tiles.py`: a DRAM channel's NIU (`DRAMEndpointNUI`)
  holds an arriving *request* for the device's own service time before
  answering it — at the **endpoint**, deliberately not in the flight,
  reusing the hop model's parking lot so the tile sleeps through the
  wait rather than spinning.
  **The number is a derivation, and the provenance convention gained a
  rank to say so.** `dram = 358 − 259 = 99` cycles on Wormhole: the
  measured end-to-end DRAM read minus the measured end-to-end L1 remote
  read, same transaction shape from the same agent, so the NoC round
  trip and the issuing core's path cancel and no modelled quantity
  appears in it. Cross-checked against the hop-model residual (~102)
  from derivations sharing only their source data. That is neither
  `isa_doc_derived` nor `vendor_source` (nobody printed 99) nor
  `estimated` (nothing here is a judgement call), so the new
  **`vendor_source_derived`** sits below `vendor_source` and above
  `estimated`, requires a written-out `derivation` like
  `isa_doc_derived` does, and has its one entry pinned by a test.
  **Blackhole deliberately gets nothing** — the same subtraction gives
  126, but its end-to-end rows fail the consistency check Wormhole's
  passes (residuals 68 local vs 122/123 remote) and were measured
  against tt-metal's 1.2 GHz assumption against the ISA docs' 1.35, a
  fork between 126 and 142. So every Blackhole guard is unmoved, which
  is a statement about confidence and not about Blackhole's DRAM.
  **Measured** (Wormhole, one-cycle poll resolution, this term isolated
  from the rest of the model): `matmulblock` +9.8 %, `matmulidx` /
  `reduce` +6.6 %, `tilize` +5.7 %, `transpose` +4.9 %, `softplus`
  +1.0 %, `sfpumath` +0.9 %, `untilize` +0.4 %; every value guard still
  passes. The finding is *how much of the charge lands*: **47–100 % of
  the charged cycles reach the total**, against 0 % for the five Tensix
  units and 0.2 % for the RISC-V path, because a dataflow kernel's
  `noc_async_read` + `noc_async_read_barrier` puts a DRAM round trip on
  the critical path by construction. Still not modelled, and named
  rather than implied: bank conflicts, refresh windows, endpoint
  occupancy (a second request is not queued behind the first) and
  size-dependence — DRAM bandwidth stays unconsumed because it is the
  same serialisation the NoC's per-link bandwidth term describes.
- **Phase 5, fifth instalment — NoC bandwidth** (2026-08-03), closing
  the gap the hop-model instalment named first ("a 32-byte semaphore
  poke costs what an 8 KiB tile read costs"). Same consumer,
  `tt_sim/network/tt_noc.py`; one more sourced constant, and a shape
  that is neither a per-opcode occupancy nor a flight time but an
  **occupancy of a link**: `flits = ceil(bytes / flit_bytes)` (32 B
  Wormhole, 64 B Blackhole) at one flit per cycle, spent twice — the
  packet's tail arrives `flits − 1` after its head, **once and not per
  hop** because a *wormhole*-routed NoC lets the tail follow rather
  than re-assembling at each router, and the injecting NIU's outbound
  link is held for all `flits` cycles so the next packet waits. Per-NIU
  occupancy deliberately, not per-link: a packet only waits on a
  router-to-router link behind *another tile's* traffic, which needs an
  arbiter, which is `noc.congestion` (`provenance: unknown`). Two
  under-charges kept on purpose (no header flit is counted; a read
  request carries no payload, only its response does) and one
  over-charge avoided (a multicast claims the injection port **once**,
  because the hardware fans one packet out in the routers).
  **It differentiates, which was the whole point.** Measured at
  one-cycle poll resolution with the DRAM term isolated out: `six`
  (tile-streaming, 288 packets / 288 KiB) 63,425 → **67,550**,
  **+6.5 %**, charged 4,464 bandwidth cycles = **15.5 per packet**;
  `nine` (semaphore-heavy, 40 packets / 2.5 KiB) 12,060 → **12,064**,
  **+0.03 %**, charged **24 cycles = 0.6 per packet**. Under the hop
  model those were the same number per packet; they now differ **26×**.
  Also `optest` +3.8 %, `sfpumath` +0.6 %, `sfpuchain` +0.5 %, `four`
  +0.005 %, and `loopback` **−3 cycles** — a handshake landing the
  other side of a poll boundary, a reminder that these totals are set
  by when cores meet rather than by a sum of charges.
  **More plausible, not merely larger**: 288 KiB over 64-byte links is
  4,608 cycles of transfer by hand-arithmetic and the model charges
  4,464, so the term is the traffic's own DMA time. It is not the
  dominant one — a 2 KiB tile read is ~240 cycles of round trip against
  32 of serialisation, so these kernels are **latency-bound by ~7×**,
  and `six` is now ~75 % attributed to a published number.
  **The per-trid response-ordering hazard the hop model created is
  closed**, not merely still unobserved: every request carries its
  issuing NIU's monotonic `seq`, echoed by whatever answers it, and the
  outstanding store is `{trid: {seq: state}}` so a response finds its
  own request whatever order they return in. Removal rather than
  detection because the failure mode was silent (a read response handed
  the *other* request's L1 address); an unmatchable response raises
  `NoCResponseError`. A test *forces* the reorder (two reads, one trid,
  DRAM and local L1) and `NUI.out_of_order_responses` counts it in
  every run — still zero across ten guards. The injection-port
  occupancy is part of the answer too: without it a small packet would
  overtake a large one to the *same* destination, which no NoC does.
  Response routing is untouched — responses still carry `reply_to`, the
  endpoint object, and `seq` names *which request*, never *where*.
- **Rung 3 of the calibration ladder — the Tensix instruction costs,
  measured on Blackhole silicon** (2026-08-04). `perfbench/tensixbench`
  ran on a real card for the first time, and its output is now the only
  hardware-measured data in the tree. **Six datasets are tracked and
  they are not peers**:
  `tt_sim/perf/datasets/tensixbench-blackhole.csv` is the **primary**,
  504 raw points, `--blocks 32 --iters 64 --dvalid-once`, phases A *and*
  B, firmware bundle 19.6.0, KMD 2.9.0;
  `tensixbench-blackhole-dvalid-per-thread.csv` is a **control**, kept
  because it reproduces a known-bad probe setup and demonstrates the
  artefact that setup produces; and the four
  `tensixbench-blackhole-unpacr-nop-{bf16,fp32,tf32,fp16}.csv` are
  **experiment X2**, one measurement in four files — the source data
  format is a per-run configuration, so no one of them says anything
  alone. They report **no resolvable format effect** on the MATH probes
  (spread 0.001 cycles against a resolution of 0.036–0.052), with the
  `bf16`/`fp32` pair — same decoded `SrcAStyle`, predicted identical
  before the run — coming out at exactly 0.000. Caveats that travel with
  it: this is the Wait-Gate regime, not the MOP-issued cost the tables
  charge, and it is one run per format on one part.
  Provenance lives in each file's own `#`
  header so it cannot be separated from the numbers.
  `python3 -m tt_sim.perf.tensix_bench_sweep` reads the primary with no
  arguments — a *named* default now, not "the only file present" — and
  local run output stays gitignored. Full write-up in
  [`docs/plans/tensix-cost-benchmark.md`](docs/plans/tensix-cost-benchmark.md).
  **The ThCon "3 or 4" family is real**: `ADDDMAREG` / `MULDMAREG` /
  `SHIFTDMAREG` / `CMPDMAREG` at the low end of the documented range,
  and the only entries in the whole Tensix table a single thread could
  ever have tested. The other 15 retained series carry a 1-cycle
  occupancy and are **not testable from above by one thread at all** — a
  baby RISC-V issues at most 1 IPC, so 1.0 is the floor of the
  instrument, not a result. What confirms them is the *thread sweep*:
  0.998 / 1.968 / 2.970 at one, two and three TRISCs is 3.0× scaling on
  a shared 1-IPC unit, which is the ISA docs' prose constraint measured
  rather than read, and nothing had ever checked it.
  **Phase B ran, and the fidelity arithmetic is confirmed.** A real
  `matmul_tiles` inner loop at LoFi / HiFi2 / HiFi4 costs 34.92 / 52.47
  / 86.12 cycles per call, i.e. steps of **17.55 and 33.65 against 16
  and 32 predicted** by `mvmuls_per_tile = 16` × one cycle. That number
  was derived arithmetic, load-bearing on every simulated matmul, and
  wrong by 2.5× on all of them if it had been wrong at all; nothing had
  ever checked it. The feeder confound the design named in advance (all
  three slopes equal) did not occur. So **the rung is climbed for both
  phases**.
  **The matrix unit has two throughput regimes ~6× apart, and only one
  of them is in the tables.** Individually issued as `.ttinsn` words,
  `MVMUL` / `ELWADD` / `ELWMUL` cost ~6.0 cycles — the documented
  latency of 5 plus one — because each pays the Wait Gate's
  `SrcA[bank].AllowedClient` check; they scale a clean 3.0× across
  threads, so that is one shared unit at six cycles, not contention.
  Replayed back to back out of the MOP expander they cost **~1.07**,
  from phase B's marginal arithmetic. Phase A's probes are unrolled
  `TTI_*` macros by construction and can therefore *only* observe the
  slower regime. **Occupancy stays 1**, which is the MOP figure, because
  every non-experimental LLK path reaches these opcodes through a replay
  buffer (`llk_math_matmul.h`, `llk_math_reduce.h`,
  `llk_math_eltwise_binary.h`) — charging 6 would mis-cost every real
  matmul by ~6×. Both figures are recorded as `corroboration` with which
  one the table describes stated first.
  **A previous headline was retracted in the process.** The first
  hardware run showed the three MATH probes scaling ~12× across threads
  — aggregate throughput *falling* as issuers were added, which no ISA
  document supports. It was an artefact of issuing one `SETDVALID` per
  active thread, so that thread count was perfectly confounded with
  SrcA/SrcB bank state and the third one handed the Matrix Unit a bank
  it already owned. Experiment X1 (hoist it; one legal `SETDVALID`)
  retracted it, and moved the *single-thread* figure too — 0.998 to
  5.989 — so no column of the original run survived for those probes.
  The investigation is in
  [`docs/plans/matrix-unit-thread-contention.md`](docs/plans/matrix-unit-thread-contention.md);
  X2 and X6 remain interesting, X0/X3/X4/X5/X7 do not. The sweep's
  discriminator gained a fourth verdict band, `SUPERLINEAR (N.Nx) --
  INVESTIGATE`, above 1.15× perfect scaling: a shared 1-IPC unit is a
  ceiling and cannot degrade faster than the thread count, so printing
  `shared (12.1x)` was one word away from filing an artefact as
  documented behaviour.
  **One table entry was wrong, and it is a docs-versus-tables finding
  rather than a simulator bug.** `RDCFG` measured 0.998 against an
  `at_least 2.00` occupancy — but the ">= 2" is the ISA doc's *latency*,
  copied into the occupancy field because "the page gives no separate
  throughput figure", in a note whose own next clause names the
  throughput figure it does give ("at most one of these per cycle";
  Blackhole states the same as an IPC group). `WRCFG` two entries above
  it has the identical shape recorded correctly, so the error was
  visible before any hardware existed. Occupancy is now 1 (`isa_doc`),
  the `>= 2` latency is unchanged and still untested by anything — the
  slope method runs no dependent chain and structurally cannot see a
  latency. The config unit was unwired at the time, so **no simulated
  cycle moved** (it is wired now, and still none does);
  what does move is that the `matmulblock` divergence the old 2 provoked
  (a missing ordering guarantee between a config write and its readers)
  is no longer *reachable* from the tables, and `config.py`,
  `costs_test.py` and the plan doc now say so rather than letting a bug
  go quiet. (Not reachable is not fixed; it was fixed the next day — see
  the bullet below.)
  **No `measured` provenance rank was added, deliberately**, with the
  argument written into `tensix_instruction_costs.yaml`: a document and
  a measurement are usually not the same quantity (this case exactly), a
  document describes every part where a run samples one, and a top rank
  is a licence for one unreplicated run to overwrite a published figure.
  Instead there is a new optional `corroboration` field — a *different
  axis* from provenance, never affecting `PROVENANCE_RANK`, pinned to a
  list of entries by test, and required to say how many runs on how many
  parts. It says "ONE RUN, ON ONE CARD" (or "TWO RUNS, ON ONE CARD" for
  `RDCFG`, which reproduced across both tracked datasets). It now covers
  unit-level numbers too, not only instruction entries:
  `MATH.fidelity_phases.mvmuls_per_tile` carries one, and
  `CostTable.corroborated_extras()` exists so that the discipline tests
  can see it — a corroboration the pinning tests could not reach would
  make the field mean less everywhere it appeared.
  **Two sub-1 % systematics turned out to be the instrument.** Every
  1-cycle probe read 0.998 and every 3-cycle probe 2.973. The first is a
  single warm-up outlier: three of four raw points sit *exactly* on
  `66n + 11` and only n = 32 is off, by +15 cycles, which tilts a
  four-point least-squares line (R² is 1.0000 throughout — R² was never
  the right gate). The second is the control subtraction over-correcting:
  the clean increments are exactly 192.000 cycles/block = 64 × 3.000 with
  no room for the loop's 2 cycles, because the RISC-V loop counter and
  branch issue *underneath* a back-pressuring unit and cost nothing. The
  whole unit-limited half of the dataset reads as exact integers without
  the subtraction and the issue-limited half reads as exact integers with
  it. The sweep therefore now computes a **fit resolution** per series
  (`slope(control)/unroll` + 2 standard errors) and splits its "is the
  table a floor?" verdict three ways; it used to call all 19 series
  over-charged when 18 were fractions of a cycle inside its own
  resolution, and now names one. The admitted cost: it cannot detect an
  over-charge below ~0.03 cycles/instruction.
- **The RISC-V front end now has a measuring instrument**
  (2026-08-05). `perfbench/riscvbench` is the companion to `tensixbench`,
  and it exists because of that benchmark's headline: against tt-sim
  **every** probe of **every** Tensix unit at **every** data format reads
  exactly 1.000 cycles, because nothing back-pressures the core that
  issued it. Same conventions — one tt-metal binary for silicon and
  tt-sim, slope method so fixed costs cancel, per-phase verdict, CSV with
  its provenance in a `#` header — plus a companion sweep
  (`tt_sim/perf/riscv_bench_sweep.py`, 46 tests) that reads its
  predictions *out of* `unit_costs.yaml` rather than restating them,
  declares its exclusions before any residual, carries a per-series
  resolution term and reports residuals by axis. Five phases, scored
  independently: **R** straight-line RV32IM, **T** the `.ttinsn` issue
  path, **C** branch direction, **Q** the Tensix instruction queue's
  depth, **F** instruction footprint. **Nothing was added to any cost
  table.**
  The design's own hardest problem is that **a run where everything reads
  1.000 is simultaneously the expected answer and the signature of a
  benchmark that measured nothing** — a baby core issues at most 1 IPC,
  so 1.000 is the floor of the instrument. It is answered with four
  probes tt-sim *already* charges (`rv_mul_dep`, `rv_div`,
  `rv_load_chase`, `rv_store_spread`): with `TT_SIM_COST_MODEL=1` they
  land on 1.000 / 6.000 / 1.984 / 4.969 against table values of 2 / 6 /
  2 / 5 — the 2 being `l1_dcache_hit`, which is what the simulator
  charges every L1 load — and with the model off all four read 1.000 and
  both the benchmark and the sweep refuse the run in those words.
  **The strongest simulator result is not a null.** `q_adddmareg` and
  `q_adddmareg_sync` issue the same burst; the second drains inside its
  own timed region. At a 128-instruction burst the core sees **128
  cycles** and the work costs **505** — 377 cycles in flight when the
  last `.ttinsn` returned, growing linearly, so nothing was ever waited
  for. That is `cost-model.md`'s "the constraint is the un-modelled
  RISC-V front end" made quantitative in one number.
  It also found one real gap in tt-sim, recorded and **not** acted on:
  Blackhole's multiply is one cycle of EX1 plus one of EX2, an occupancy
  of 1 and a latency of 2, and `cost.py` charges the occupancy with no
  scoreboard entry for the result — so a dependent chain reads 1.000
  where the pipeline description gives 2. An under-charge, which is the
  direction the policy asks for; closing it is a separate change with its
  own gate run.
  The prediction placed on record before the hardware run: phase T's
  fusion delta, phase C's direction delta, phase Q's knee and phase F's
  footprint row are all **forced** to zero against tt-sim — it has no
  instruction cache, no branch predictor and an unbounded queue — so a
  difference on a card is the whole result. Runbook in
  `perfbench/riscvbench/README.md`, methodology in
  [`docs/plans/riscv-front-end-benchmark.md`](docs/plans/riscv-front-end-benchmark.md).
- **It has now run on Blackhole silicon** (2026-08-05), and three of the
  four pre-declared numbers came out **against** the null. Two datasets
  are tracked in `tt_sim/perf/datasets/`, so
  `python3 -m tt_sim.perf.riscv_bench_sweep` reproduces it all with no
  hardware. The instrument is live — `rv_div` 33.001, `rv_load_chase`
  8.098, `rv_store_spread` 5.217, `rv_mul_dep` 1.985, all four above the
  1.000 floor — so every 1.000 elsewhere in the run is a finding.
  **`.ttinsn` fusion does not happen.** `spread4 − fuse4` is **+0.077**
  cycles/group where `PushTensixInstruction.md` predicts **+3.000**, in
  all six thread slots and in both runs. The "two different quantities"
  reading that resolved the `RDCFG` conflict was looked for and closed:
  the kernel ELF was disassembled and the four `.ttinsn` words really are
  adjacent; the fetch-line-straddle hypothesis predicts +2 and is refuted
  outright by `tt_fuse2`, whose pairs are all co-resident in one 128-bit
  read and still cost two cycles. `riscv.ttinsn_fusion` keeps its numbers
  and its `isa_doc` rank and gains a **`contradiction`** field — a new
  key, for the case the provenance convention describes as "record both
  and say so" and had no vehicle for.
  **There is an instruction-fetch cliff nothing has published**: flat at
  ~0.998 cycles/instruction for 64…1024-instruction loop bodies and
  **1.251 at 2048**, so a boundary sits between 4 KiB and 8 KiB of text —
  identical at one, two and three threads, and identical in both runs.
  **Issuing a Tensix instruction costs the core one cycle** (0.996–0.998
  on three units) and scales 2×/3× with thread count for shared units
  while `TTI_NOP` on unit `NONE` stays flat, which is the control that
  makes the rest legible. `tt_adddmareg` at 2.972/5.987/8.995 reproduces
  `tensixbench`'s independent `ADDDMAREG` 3.0 — the only entry in either
  cost file now measured by two different benchmarks. **No branch penalty
  is resolvable** (`taken − not taken` = −0.047; taken is marginally
  *cheaper*), which bears on the mispredict **rate** and leaves the
  documented 4-cycle **bubble** untested.
  Phase Q was refused by the benchmark's own gate and its read-out has
  been **rebuilt**: the point-by-point `q_ctrl` subtraction rested on a
  false premise (the cascade evaluates all seven of its `if`s at every
  burst length, so its cost is constant in `n`, not growing) and was
  injecting ±17 cycles of structured error. Rebuilt on raw differences
  over a wide baseline it says the core outruns ThCon at one thread (2.22
  against 3.000 cycles/instruction, ~92 cycles still in flight at a
  128-burst) and is fully back-pressured at three (8.99 against 9.000) —
  but **a queue depth in entries is not resolvable** by this
  construction, and the sweep must reach n ≈ 1024 to settle it. The gate
  was not weakened. **Zero cost-table numbers changed**; seven
  `corroboration` fields were added, all under `arch_overrides.blackhole`.
- **Two of the silicon residuals were then adjudicated, and only one was
  a defect** (2026-08-05). `rv_div`'s 33.001 against a charged 6 is the
  floor policy working: 6–33 is a *data* dependence — "dependent upon the
  magnitude of the dividend" — so the ends are two operands, the function
  between them is published nowhere, and interpolating would be an
  invented curve wearing a citation. The exposure was measured rather
  than argued: across four in-tree Blackhole replay guards a kernel
  launch executes **0–2 divides in 40,000–80,000 instructions** with
  **9-to-12-bit** dividends, against the benchmark's 29 significant bits.
  **So 33 is the benchmark's operand, not the instruction as kernels use
  it** — a finding about the benchmark; the floor stands and the reasoning
  is recorded in `tt_sim/pe/rv/cost.py`.
  `rv_load_chase`'s 8.098 against a charged 2 **was** a defect, in the
  sweep and not in the table. The prediction was read off `model.py`'s
  `_LOAD_LATENCY_KEYS`, which names `l1_dcache_hit` because that is the
  low end of a two-ended cost and the right thing to *charge* a kernel
  whose hit rate nobody publishes — and the wrong authority for a probe
  whose access pattern is known. **The L0 data cache is 64 bytes, "4
  lines of 16 bytes each"**, and the chase walks a 1 KiB ring, so
  `l1_dcache_miss` is the row it reaches under any organisation. The
  sweep now picks the row from each probe's working set against that
  published capacity (new `riscv.l0_data_cache`, `isa_doc`, charged by
  nothing), and the residual is **+0.098** where it was +6.098.
  `rv_load_indep` sits *exactly* on the capacity, which the page does not
  settle either way, so its prediction was deliberately **not** moved to
  match its reading. **No simulated cycle changed**: this is a reporting
  fix, and the RV load path still charges the hit row.
- **Phases S and G ran on the card, and one of the two pre-declarations only
  half-held** (2026-08-05). Four more runs — `--blocks 32`, `--blocks 8` and
  two single-phase `--gset` runs — all four banked in `tt_sim/perf/datasets/`.
  **The Tensix instruction queue is ONE PER CORE**: `D` at k issuing threads
  over `D` at one reads 0.97×/0.95× at two and 1.06×/1.07× at three, where a
  shared queue predicts 0.50× and 0.33×, and the spinning control moves
  0.91×–1.00× as both hypotheses require. Four discriminating comparisons, two
  runs, none within a factor of two of the shared prediction.
  **Phase G tightened the footprint boundary from an octave to 4096–5120 bytes
  and found a plateau**: 1.000 at 4096, 1.153 at 5120, 1.252 at 6144 and 7168,
  1.251 at 8192. A graded rise that saturates is the signature of *partial*
  residency, and it is equally the signature of a fixed prefetch reach — so
  `docs/bh_arch.md` §1.1 records the shape and **keeps its noun**: a boundary
  in loop-body size, not a cache capacity, because nothing publishes one.
  **What did not hold is the SIZE of phase Q's correction.** It was declared
  that phase S should read above phase Q by the reference burst's own occupancy,
  ~8 entries. It reads 15.3 above — 31 against 16. Exactly 8.0 is the declared
  term, to a tenth of an entry; the remaining ~5–7 is a difference between the
  two burst *forms* that reproduces to the cycle in four runs and is **not
  explained**. A third estimator sharing no term with the backlog arithmetic
  puts the forms 7.0 entries apart too, so it is not the correction. The
  leading candidate — phase S drains the pipe before `t0` and phase Q does not,
  so a saturated backend carries the previous burst's residue into `plain`
  forever — is a one-parameter fit to three points and is untested. **So the
  depth is banked as a range, ~26–31 entries, and the old ~14–16 headline is
  retracted as two corrections short.** The sweep gained `_depth_reconcile`,
  which prints the whole comparison and says `RECONCILED` on a synthetic device
  whose queue really does hold one depth, so "the forms disagree" is a reading
  the instrument can distinguish rather than something it only ever prints.
  **Phase R regressed and it is not ours**: two R² failures, both
  `rv_store_spread` at t2/t3 — the only phase-R probe where three cores hammer
  L1 stores at once — and both single-point outliers rather than slope changes.
  Phase R's kernel carries no phase-S or phase-G text and every uncontended
  probe is unchanged to the cycle. Nothing was changed on the strength of it;
  the gate refusing a phase whose contended slots went non-linear is the gate
  working, and the single-thread column still passes 4 of 4. No cost table
  gained or lost a cycle count; one `corroboration` field was rewritten to
  carry phase G and to stop calling the step "an instruction-cache capacity".
  **— The untested candidate has since been tested and it was right; the
  ~26–31 range is superseded by ~31–32, in the entry below.**
- **The untimed drain: a pre-declared prediction, run and confirmed**
  (2026-08-05, fourth campaign). One line went into `QLOOPPROBE` — the untimed
  `ckernel::tensix_sync()` phase S always had — with its raw predicted cycles
  **and three refutation criteria written down before the edit**. The run
  (`--phase qs --variants t1 --blocks 32`, banked as
  `tt_sim/perf/datasets/riscvbench-qdrain.csv`) returned `q_loop_adddmareg` at
  **299 / 683 / 1451 / 2987** against a predicted **299 / 683 / 1451 / 2987** —
  a 21-cycle fall at every n ≥ 128 and zero below. **None of the three criteria
  fired**: `q_loop_adddmareg_sync` is identical to the cycle at all seven burst
  lengths and `q_loop_addi` at six of its seven, moving 3–4 cycles only at
  n = 16 against a 17-cycle control spread — reported rather than glossed,
  because n = 16 is the backlog's reference point and that blip is exactly why
  the levelled reconciliation reads 0.7 entries where 1.7 was predicted. The
  two burst forms now **reconcile**, and the estimator that shares no term with
  the others reads **32.3 for both**. `docs/bh_arch.md` §1.10 drops the ~26–31
  range for **~31–32 entries at one issuing thread**. **The per-core verdict is
  untouched**: that is a ratio between thread counts and this run has one, so it
  still rests entirely on the third campaign's two full runs. No cost table
  changed; no `estimated` entry exists.
- ~~**Two follow-ups to the silicon run are built and unrun**~~ **— run, see
  above** (2026-08-05). `riscvbench` gains **phase S** — is the Tensix
  instruction queue **shared between the three TRISCs or private to each?** —
  and
  **phase G**, which narrows the loop-body boundary phase F bracketed between
  4 KiB and 8 KiB by sweeping 1280/1536/1792. The interesting half is that
  **the construction §1.10 named as the answer does not work**: one thread
  issuing while another only *spins* separates nothing, because a spinning
  thread pushes no `.ttinsn` and therefore holds no queue entry under either
  hypothesis. Only a **saturated** second thread occupies entries (occupancy
  is arrival rate × residence time, so a slow one holds none), and the backend
  bandwidth it steals is measured in the same slot by `s_co_sync` and divided
  back out. The spinner is kept as the *control* — both hypotheses predict it
  changes nothing. The second half of the fix is the **reference burst**:
  phase Q subtracts its value at n = 16 as `tensix_sync()`'s own cost, which
  assumes the queue is empty there, and it is not — `n·(1 − p/S)` is ~10
  entries — so at two and three threads a shared queue's per-thread share can
  be *smaller than the reference*, which makes phase Q's multi-thread backlogs
  **structurally zero** rather than unluckily small. Phase S references n = 4
  and reports `D = backlog/S + 4·(1 − p/S)`; its verdict is a **ratio between
  thread counts** (1.00× per-thread, 1/k shared) and one variant produces no
  verdict at all. A consequence written down as a prediction rather than an
  edit: phase Q's depth is a **lower bound** by that same arithmetic.
  Phase G is a separate phase, and split into three `--gset` builds, for a
  measured reason — phase F's kernel is within a few hundred bytes of
  tt-metal's config buffer and widening it aborts the launch with `Program
  size (125040) too large for kernel config buffer (70656)` — which leaves
  phase F's bodies untouched and both tracked datasets reproducible.
  Verified against tt-sim, where **both answers are forced** (no instruction
  cache, `push_mop_instruction` is a list append): phase G reads a flat 1.001
  at every footprint and phase S refuses a depth in every slot and prints NO
  VERDICT. At the time of this entry neither had run on a card, no cost table
  changed, and `docs/bh_arch.md` §1.1 and §1.10 gained pointers and a
  retraction but no numbers.
- **`docs/bh_arch.md` is new** (2026-08-05) — a record of Blackhole
  microarchitectural facts established by measurement, categorised by
  whether each **agrees with**, **contradicts** or is **absent from** the
  published documentation. It exists because the cost tables' discipline
  has a gap the front-end run walked into: every table number must trace
  to a document and a measurement enters only as `corroboration`, so **a
  quantity no document mentions has nowhere to be recorded** — a
  corroboration needs something to corroborate. The i-cache cliff is
  exactly that, and is its first entry. The file loads into no code and
  competes with no table; where an entry does exist it points at that
  entry's `corroboration` rather than restating the number. **The tension
  is stated rather than filed as solved**: one number and a new document
  are proportionate, and if it becomes ten the question of whether the
  tables need a rank comes back and should be answered deliberately.
  Also holds `tensixbench`'s findings — the `ADDDMAREG` family at 3
  cycles, `RDCFG` at 1 rather than the documented ≥ 2, no resolvable
  source-data-format effect across four formats with a null control that
  makes it a finding, the fidelity arithmetic confirmed at +16/+32, and
  `MVMUL`'s **two regimes** (~1.07 cycles MOP-issued against ~6 through
  the Wait Gate), which is the one most likely to mislead a reader who
  quotes a single number. Everything in it is **Blackhole**, and it says
  which findings would need their own Wormhole run —
  `UnpackRowWidth` differing 16 versus 32 between the two being the
  standing reminder that they diverge in exactly this kind of detail.
- **The config-write ordering bug is fixed** (2026-08-04) — the one the
  `RDCFG` table entry above made unreachable without making untrue. The
  fault was in `TensixBackendUnit.clock_tick`, not in the config unit:
  a unit that accepts several instructions in one cycle (config: up to
  three `SETC16`, one per thread, plus one shared-IPC-group op; sync:
  three mutex ops; misc: one per thread) armed its occupancy *mid-batch*
  and returned, leaving the rest of that cycle's already-accepted
  instructions queued for later. The issuing threads had been told those
  instructions were accepted and had moved on, so with `RDCFG` charged 2
  the math thread's `SETC16` of `DEST_TARGET_REG_CFG_MATH_Offset` landed
  two cycles late — **behind two of that same thread's own `MVMUL`s**,
  which then accumulated into the wrong half of Dst and made
  `matmulblock` print 608.0 for C[0][0] against a golden of 1120.0. So
  it was an intra-thread reordering after all, of a kind `is_occupied`'s
  refusal cannot prevent because the instruction had already been
  accepted. The drain now retires the whole batch and holds the unit
  afterwards, for the longest cost in it: **occupancy is throughput
  back-pressure on the next instruction to enter a unit, never a delay
  of one already inside it**, which is what both arches' Configuration
  Unit pages describe (Blackhole names the pipeline stages, −4..+1) and
  what the vendor reference simulator enforces absolutely, by never
  starting a pipe's next instruction until this one's side effects are
  committed. Regression-tested with the config unit charged more than
  one cycle, at the unit and end-to-end on `matmulblock`, via a local
  stand-in cost model — **the tables are untouched**, because that
  `RDCFG` 1 is a silicon measurement. **No cycle count moves**: nothing
  arms an occupancy mid-batch today, so every guard is byte-identical.
  What did *not* survive is the second reason the config unit is
  unwired: "every entry is one cycle, so wiring it would charge
  nothing" is true on Wormhole and **false on Blackhole**, whose
  `CFGSHIFTMASK` is a 2-cycle occupancy that the `untilize` guard
  executes 32 times. Wiring the unit is therefore a real timing change
  owing its own cost-model-gate run, and it stays unwired until it gets
  one.

- **Per-unit cycle-cost tables.** Today every RV instruction, Tensix
  op, NoC request, and Mover transfer completes in the same tick it
  was issued. First step is a data-driven cost table per unit
  (cycles-per-op for FPU/SFPU/packer/unpacker/matrix; latency +
  bandwidth for NoC and DRAM; fetch/issue/retire stages for RV).
  Tables should live next to the YAML they parallel
  (`tensix_instructions.yaml` → `tensix_instruction_costs.yaml`).
  Gated on event-driven-pump Phase 5 for the *occupancy* half; the
  table itself can be written first.
  - *Test:* re-run `examples/four/` and `five/` and compare reported
    cycle counts against expected ranges derived from the ISA-doc
    latency tables; assertion lives in the replay guard.
- **RV pipeline modelling.** Cross-ref §B "No pipeline modelling".
  Minimum viable: fetch/decode/issue/retire stages with memory-stall
  back-pressure on L1 / NoC reads. **Landed 2026-08-03** as a load-use
  interlock plus L1-store and multiply/divide occupancy — see the Phase
  5 bullet above and
  [`docs/plans/cost-model.md`](docs/plans/cost-model.md#the-risc-v-cores-where-the-cycles-finally-moved).
  What is left of this bullet: branch mispredicts (blocked on there
  being a predictor to count them), i-cache miss cost (unpublished),
  sustained-load throughput and `max_loads_in_flight` (second-order
  next to the interlock), and a latency for the NoC NIU register block
  (unnamed by the docs' table).
  - *Test:* `tt_sim/pe/rv/cost_test.py` — a load from L1 holds issue for
    the documented latency − 1 cycles, independent instructions after it
    do not, and with the model off nothing changes.
- **Tensix issue latency & wait-gates.** Cross-ref §D "Wait-gate
  stalls", §D "Instruction issue latency". Wait-gates on srcA/srcB
  availability and the documented per-op issue cadence both need to
  apply.
  - *Test:* targeted Tensix op test under `optests/` that issues an
    ELWADD with srcA not yet ready and asserts the MATH thread stalls
    for the documented number of cycles. (Note ttsim is *not* an oracle
    for this — it is explicitly not cycle-accurate either, so cycle
    assertions have to come from the ISA docs.)
- **NoC timing model.** Cross-ref §C "Router arbitration", §C
  "Packet splitting". Per-hop latency, per-link bandwidth, head-of-
  line blocking on shared routers. **Per-hop latency landed
  2026-08-03**, **bandwidth the same day**; contention has not — see
  the Phase 5 bullets below and
  [`docs/plans/cost-model.md`](docs/plans/cost-model.md#the-noc-bandwidth-model-where-a-packets-size-starts-to-matter).
  What is left of this bullet: router arbitration and head-of-line
  blocking (`noc.congestion` is `provenance: unknown` — the ISA docs
  say congestion "can negatively impact latency" and quantify
  nothing), and with them the router-to-router links' own occupancy,
  which cannot be modelled without an arbiter between two senders.
  **And it cannot come from tt-metal's measured NoC dataset either**,
  tested 2026-08-05: that file changes the flow count only by resizing
  a grid, so concurrency and geometry are never separated, and its one
  scalar per key already holds the issuing core's loop. It *bounds*
  congestion (per-core bandwidth falls ~4× from a 2×2 grid to the full
  device) without supplying it. `python3 -m
  tt_sim.perf.noc_dataset_sweep` prints the argument and names the four
  card measurements that would settle it. **Those measurements were then
  taken, on a Blackhole card, 2026-08-05** (`perfbench/nocbench`): the
  uncongested line comes out at `4373.7 + 8.38 * round_trip_hops` (r2
  1.00), and congestion turns out to be a **step at the first shared
  link, not a per-link slope** — at 16 KiB two flows sharing one
  router-to-router link each pay 517.9 cycles/tx against 270.1 sharing
  none, and sharing two or three links costs no more. That is one
  link's bandwidth split between the saturating flows crossing it, so
  `noc.congestion` stays `provenance: unknown`: the model has no term
  of that shape, and it would need a per-link flow census rather than a
  coefficient. See
  [`docs/plans/cost-model.md`](docs/plans/cost-model.md), "Banked,
  2026-08-05".
  The per-trid response-ordering hazard is **closed** (responses match
  their request by issue number rather than by arrival order), not
  merely still absent.
  - *Test:* `tt_sim/network/noc_cost_model_test.py` — torus hop
    counting on both arches and both NoCs, a DRAM read landing on
    the cycle the model predicts, the flit-rate bandwidth terms, and a
    *forced* out-of-order response landing at its own L1 address. For
    contention, the two-tile
    `examples/nine/` (or a wider `seven/`) extended with overlapping
    NoC bursts that should serialise at a shared router.
- **DRAM access latency.** **Landed 2026-08-03 for Wormhole** — 99
  cycles of endpoint service time, derived rather than published, at
  the new `vendor_source_derived` provenance rank; see the Phase 5
  fourth-instalment bullet above and
  [`docs/plans/cost-model.md`](docs/plans/cost-model.md#dram-access-latency-where-the-number-had-to-be-derived-rather-than-read).
  What is left of this bullet:
  **bank-conflict and refresh-window costs**, which this bullet names
  and which remain unmodelled and unpublished (tt-sim has no DRAM bank
  model at all); **endpoint occupancy**, so a second request is not
  queued behind the first and the term adds latency without contention;
  **size-dependence**, since bandwidth is the NoC bandwidth term's
  serialisation and is charged in one place rather than two; and
  **Blackhole**, which stays `unknown` pending either an internally
  consistent end-to-end measurement set or a resolved 1.2-vs-1.35 GHz
  clock — worth ~24 % on `six`, so not a rounding error.
  - *Test:* `tt_sim/device/dram_cost_model_test.py` — off by default,
    the derivation and its rank, Blackhole charging nothing, and a DRAM
    read on a real device costing flight + service + flight. Still
    wanted: the DRAM-heavy `three/` kernel with a cycle-count
    assertion.
- **Mover & PC-buffer timing.** Cross-ref §D "Mover region-crossing",
  §E "PC buffer". Both are point fixes once the per-unit cost framework
  exists.
  - *Test:* covered by the §D mover and §E PC-buffer examples plus
    a cycle assertion.
- **Calibration against silicon traces.** Once any of the above lands,
  the bar is "match a captured cycle trace within X%". Need a small
  set of golden traces from the [tt-metal fork](https://github.com/mesham/tt-metal)
  to regress against.
  - *Test:* one captured-trace replay per major unit (RV-only,
    Tensix-only, NoC-heavy) checked in under `driver/wormhole/server/
    traces/`.

---

## J. Architectural clarity (module boundaries, docstrings, diagrams)

The simulator is intended to be hackable. As surface area has grown
(tt-metal wire bridge, 12-bank DRAM, growing Tensix backend) the
mental map has drifted from the underlying hardware. This section is
the housekeeping needed to keep "go read the code" a viable path for a
new contributor.

Partly addressed as a side effect of the Blackhole port: per-arch
constants are now one dataclass (`tt_sim/arch/profile.py`), the wire
bridge is its own package (`tt_sim/bridge/`), and the device layer split
into an arch-agnostic `device/tt_device.py` plus
`device/{wormhole,blackhole}.py`. The items below are what is left.

- **Module boundaries per hardware block.** Each hardware unit in the
  ISA docs should map 1:1 to a single Python module. Today
  `tt_sim/pe/tensix/backends/` mostly does this (matrix / vector /
  packer / unpacker / mover / thcon / config / sync / misc), but
  `frontend.py` mixes decode and dispatch responsibilities, and
  `tt_sim/network/tt_noc.py` bundles NIU, two NoCs, and the directory
  in one file. Split these so each file owns one hardware block.
  - *Test:* no example needed — acceptance check is a short table in
    `CLAUDE.md` mapping each ISA-docs subsection (`WormholeB0/
    TensixCoprocessor/FPU.md`, etc.) to exactly one `tt_sim/` file.
- **Docstrings as architectural documentation.** Every top-level class
  representing a hardware block should carry a docstring giving: the
  block it represents, a permalink to the matching ISA-docs section,
  inputs/outputs, configuration registers owned, and an explicit list
  of what is modelled vs stubbed. Today most classes have a one-line
  summary or none.
  - *Test:* no example needed — acceptance is `ruff` / mypy not
    relevant here; a one-shot audit script that walks `tt_sim/` and
    asserts every `class` whose name ends in `Tile`, `Core`, `Backend`,
    or `Unit` has a docstring containing an `ISA-docs:` link.
- **Diagrams.** Three families worth keeping in the repo (as Mermaid
  in markdown so they render on GitHub):
  - **Top-level device block diagram** — `Wormhole` → tiles → NoCs →
    DRAM controllers; lives in repo root README.
  - **Tensix dataflow** — unpacker → srcA/srcB → FPU/SFPU/MATH →
    dst → packer → L1, annotated with the L1 base addresses and the
    config registers that gate each hop; lives in `tt_sim/pe/tensix/
    README.md` (new).
  - **Kernel-launch sequence diagram** — host writes firmware, sets
    boot jump, deasserts BRISC, BRISC fetches launch_msg, launches
    NCRISC/TRISC, kernel runs, mailbox flips to `RUN_MSG_DONE`; lives
    in `driver/wormhole/README.md`.
  - *Test:* no example needed — acceptance is that each diagram
    renders on GitHub and stays in sync with the code (covered by
    the docstring audit above forcing reviewers past the matching
    file).
- **ISA-docs cross-reference index.** A single `tt_sim/ISA_INDEX.md`
  mapping every ISA-docs file under `WormholeB0/` (and the `BlackholeA0/`
  files, which exist only where behaviour differs — so that tree *is* a
  delta list) to the tt-sim file that implements it (or "not modelled").
  Closes the loop with the
  module-boundaries item — if a hardware block has no tt-sim home, it
  shows up here as a gap.
  - *Test:* no example needed — acceptance is the file existing and
    being grep-able for "not modelled" to enumerate gaps.

---

## K. Quick wins (small, well-isolated)

Loose ends that don't need design work:

- ~~**`stub_listener.py` and `driver/wormhole/six/` ruff errors.**~~
  **Done** — `ruff check .` and `ruff format .` are clean across the
  tree (verified 2026-08-02), and the numbered examples have moved to
  `examples/`.
- **`run.sh` quoting / shellcheck.** Defensible as-is, but `shellcheck`
  surfaces a few suggestions. Now two of them (`driver/wormhole/run.sh`
  and `driver/blackhole/run.sh`) plus the test scripts
  (`optests/diff.sh`, `driver/blackhole/tests/run_examples.sh`).
- **`MEM_BOOT_CODE_BASE` boot-jump under the wire flow.** tt-metal-host
  writes the firmware images; whether it also writes the `jal` at
  `MEM_BOOT_CODE_BASE` should be verified against the captured trace —
  if it doesn't, the wire bridge should synthesise it.
- *Test:* no example needed — housekeeping. Existing examples remain
  the acceptance bar (they must still pass after each fix).

---

## L. Profiling & optimisation

Now that §I cycle-approximate is the headline goal, the simulator must
be fast enough to run kernel-scale workloads. `matmul_single_core` 640³
bf16 already times out under the wire bridge — a leading indicator that
there's perf headroom to find *before* adding more per-cycle work
(cycle modelling, stall tracking, perf counters).

### Status: measured, then four optimisations landed

The "profile first" discipline below was followed, and it changed the
plan. Everything here is in
[`driver/wormhole/docs/profiling.md`](driver/wormhole/docs/profiling.md)
(appendix "where tt-sim's own wall clock goes", measured 2026-08-02 on a
12th-gen i7-12700H, CPython 3.12, sequential pump).

**The baseline.** tt-sim ran at **1–4 k simulated cycles/s**, dropping
to ~770 cycles/s with the matrix unit busy — roughly **50 µs and ~120
Python calls per simulated RV instruction**. Method: replay the
Blackhole offline guards (no tt-metal, no UMD, no IPC in the
measurement, 97–99 % of each run inside `MultiTileClock.run`), with
`cProfile` for call counts and a stack sampler for undistorted time
shares.

**What landed, in order:**

| # | change | effect |
|---|---|---|
| 2 | batch the FPU accumulate datapath with numpy (`_fpu_group_sums_batch` / `_fpu_accumulate_batch`, scalar pair kept as the fuzz-tested reference) | `six` **1.84×**, `matmulidx` 1.41× |
| 1+3 | RV32IM interpreter: GPR representation, decode, disassembly f-strings, tracing bookkeeping; plus the per-cycle soft-reset poll | **1.96–2.64×** on the five RV-bound workloads, 1.35× on the matmul |
| 6+4 | event-driven pump Phase 1 (tile dormancy) + `NUI.clock_tick` idle early-out | idle floor 13.3–32.2×; real workloads 1.05–1.45× |
| — | Tensix operand gather (`SrcRegister.readRows`, `getDst16bRows`/`setDst16bRows`, conversions applied to whole blocks) | `six` **1.60×**, `matmulblock` 1.31× |

Net: `examples/six` (128³ bf16 matmul) went **35.7 s → 7.9 s**. The
RV-bound workloads went from 2.7–5.8 k to 6.2–12.8 k cycles/s.

**Method note worth keeping:** every one of these was measured as an
**interleaved A/B between two frozen git worktrees**, minimum of N
rounds, with a verified-zero control workload. An earlier
non-interleaved sweep showed *every* workload "improving", including
ones that never enter the changed code; that pass was discarded. The
tree usually has concurrent work in it, so this is not optional.

**Still open:** target 5, `MemoryMap` interval lookup / `MemorySpace.read`
(3–7 % across workloads; the win is caching the last-hit range, not a
JIT). And kernel-scale workloads remain out of reach — see §D
`matmul_single_core`.

### Discipline

- **Profile first, optimise second.** No "use Numba" or "rewrite in
  Cython" decision without trace data showing where the cycles go.
  - *Test:* output is a writeup refreshed in
    `driver/wormhole/docs/profiling.md` per major optimisation round.
    (Note the profiling itself used `cProfile` + a hand-rolled stack
    sampler, not `TT_SIM_TRACE_COUNTERS` — counter tracing costs ~2× on
    RV-bound runs, so it distorts what it is meant to measure. Use it
    for *kernel* profiling, not simulator profiling.)
- **Land §I event-driven cycle pump *before* JIT'ing.** *Sequencing
  upheld, reasoning corrected.* The pump is **0.3–1.5 % of wall clock**
  today, so it is not the bottleneck and rewriting it for speed alone
  would be invisible. It is a **ceiling**: dead linear at ~0.94 µs per
  component-tick pre-Phase-1, capping one Tensix tile at 24.8 k
  cycles/s and eight at 4.4 k. The sequencing is right for a different
  reason than originally stated — the JIT targets in the current shape
  (per-component `clock_tick`) are not the JIT targets in the
  event-driven shape.
  - *Test:* covered by §I.
  - **Phase 1 done**, Phase 3 deferred, Phase 4 next — see §I for the
    per-phase status and
    [`docs/plans/event-driven-pump.md`](docs/plans/event-driven-pump.md)
    for the design.

### Numba

Natural first JIT to try because it preserves the hackability pitch —
decorators, no build step. **No Numba has been needed yet**: the two
targets below that have been attacked were both fixed in plain Python /
numpy, which is the cheaper move and keeps the dependency out. Numba is
still on the table for what is left.

- ~~**Tensix backend numeric inner loops**~~ — **done without Numba,
  and the stated rationale was wrong.** `_fpu_group_sums` /
  `_fpu_accumulate` were *not* "already numpy": they were scalar
  interpreted Python over lists and arbitrary-precision ints, so the
  headroom was far larger than the predicted 2–3×. A plain numpy/int64
  batch rewrite over a whole MVMUL got **1.84×** on the matmul (and the
  operand gather a further 1.60×), with the scalar pair retained as the
  fuzz-tested reference. Numba should now be evaluated against *that*
  baseline, not the original one.
- ~~**RV32IM execute** — only if rewritten table-driven over typed
  arrays; significant refactor~~ — **partly done, and it was
  under-prioritised here.** RV32IM was the single largest consumer of
  wall clock across the suite (65–84 % on four of five workloads),
  ahead of the Tensix backends outside matmul — it belonged first, not
  second. It also did not need the full table-driven refactor to pay:
  local edits to the GPR representation, the decode, the disassembly
  strings and the tracing bookkeeping gave **~2–2.6×**. The
  table-driven-over-typed-arrays rewrite remains available for more.
  - *Test:* `driver/simple/ex2`/`ex3` cycle-count regression +
    wall-clock measurement.
- **§I event-driven cycle pump** — best Numba target by construction,
  *after* Phase 4 gives it the heap-of-`(cycle, unit, event)` shape.
  Not before: a polymorphic `clock_tick` over 240 objects is not a JIT
  target.
  - *Test:* covered by §I.

Where Numba does **not** help — structurally incompatible with
`@njit`:

- `MemoryMap` interval lookup → polymorphic `mem_mapable` dispatch
  (most-called function in the sim).
- `EventBus.publish` subscriber walk.
- Current clock tick (`tick()` on every registered component).
- Tensix `frontend.py` YAML-driven decode → backend dispatch.

Sprinkling `@njit` on OOP-heavy methods rarely pays — per-call boundary
cost eats the JIT win.

Measurement note on the `MemoryMap` claim: it is **contradicted on call
count, supported on cost class**. On `four` the most-called tt-sim
functions are `Register.read` (2.41 M) and `RegisterFile.get` (2.83 M);
`memory_map.locate` is 1.27 M. It is still 3–7 % of wall clock and
still Numba-hostile, so the conclusion drawn from it stands even though
the premise was wrong.

### `nogil=True` — revive §A threading

§A threading is structurally correct but a perf regression because GIL
contention dominates per-cycle Python work. Numba can release the GIL
via `@njit(nogil=True)`. Wrapping Tensix backend hot inner ops this way
gives threading a real path to a speedup that doesn't depend on Python
3.13t — orthogonal to single-thread perf and arguably the most
interesting near-term Numba angle.

*Arithmetic that tempers this (no new evidence either way, but worth
having before the work is started):* on the matmul workload ~70 % of the
time is in **one** Tensix unit on **one** tile, so releasing the GIL
there only helps multi-tile runs — and the barrier those runs must cross
was the per-component-tick pump cost, which Phase 1 has cut but which
threading itself does not remove.

- *Test:* re-run §A's 4-Tensix `four/`-derived benchmark with Tensix
  inner ops Numba-`nogil` and `TT_SIM_THREADED=1`; target wall-clock
  under sequential, not over.

### Alternatives considered

- **Cython** — better for OOP than Numba, but adds a build step.
  Breaks hackability. Avoid unless profiling proves OOP dispatch
  dominates *and* no Numba-compatible rewrite is feasible.
- **PyPy** — drop-in for pure Python, but numpy interop has been
  historically weak; the Tensix backend is numpy-heavy. Not viable.
- **C extension** — fastest, worst for hackability. Last resort.
- **Free-threaded Python 3.13t (PEP 703)** — already flagged in §A.
  Orthogonal to Numba `nogil`; complementary.

### Watch: Tenstorrent shipping their own perf model

ttsim is explicit about *not* being cycle-accurate today, but
Tenstorrent has internal perf models (their compiler needs one). They
may not have released one because it exposes microarch detail — but
they could ship cycle accuracy or public cost tables at any time. If
they do, §I's headline status needs re-evaluation. Watch the ttsim and
tt-metal release notes.
