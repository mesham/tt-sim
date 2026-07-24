# Roadmap

Inventory of known TODOs and gaps. Picked off opportunistically, not in
strict order. Items marked **★** are load-bearing for a near-term goal
(multi-Tensix tt-metal programs, broader kernel coverage, perf); the rest
are correctness/completeness backlog.

Each item carries a *Test:* sub-bullet naming an example that would
exercise it — either a new entry under `driver/wormhole/` (numbered after
the existing `one/`–`six/` series) or `driver/simple/`, or an existing
tt-metal `programming_examples/` build that should be brought up once
the gap closes. Items marked *no example needed* are pure housekeeping,
documentation, or perf modelling.

References:
- ISA docs: <https://github.com/tenstorrent/tt-isa-documentation> (`WormholeB0/`).
- Project notes: `CLAUDE.md`, `driver/wormhole/server/README.md`.

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
on every axis. Defensible lanes:

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
   as oracle so tt-sim engineering focuses on the cycle side.
5. **Education** — `driver/simple/ex1-5` ladder + readable backends.

Frame this honestly in user-facing docs: tt-sim is a *first-order
performance estimator* with hackable internals and rich tracing, not a
cycle-accurate validation tool. Cycle-accurate (matching silicon
within ~%) needs RTL or captured silicon traces; neither is publicly
available (see §I "Calibration against silicon traces"). Until that
data exists, call it "performance estimator", not "cycle-accurate".

### Re-prioritised headline goals

1. **§I cycle-approximate perf model** — promoted from opportunistic
   perf work to the main thing.
2. **§H follow-ups gated on §I** — Perfetto durations, NoC
   `vc`/`issue_cycle`/`arrival_cycle`, FPU/SFPU stall reasons, packer
   back-pressure, L1 bank conflicts. Schema exists; data becomes real
   when §I lands.
3. **§L profiling + optimisation** (new) — measure before optimising;
   make the cycle pump fast enough to be usable at kernel scale.
4. **Differential testing harness against ttsim** — first-class
   workstream; ttsim is the new correctness oracle.

De-prioritised:

- Closing every §D `NotImplementedError` — cover what the headline
  cycle-approx examples exercise; let ttsim cover the rest.
- Multi-arch (Blackhole/Quasar). Out of scope.
- Native fast-dispatch. Slow-dispatch suffices for everything tt-sim
  is now aiming at.

---

## A. tt-metal wire bridge (`driver/wormhole/server/`)

The integration runs end-to-end for the canonical `programming_examples/`
under `TT_METAL_SLOW_DISPATCH_MODE=1` (see §E). The wire bridge
materialises the workers listed in `TT_SIM_TENSIX_COORDS` (default
`1-1`); other worker coords stay as `NullCore` to keep the cycle pump
cheap given tt-metal's grid-wide init traffic.

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
  examples (`driver/wormhole/one/`–`six/`, the default `Wormhole()`)
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
    perf follow-up wants a `driver/wormhole/seven/` (multi-Tensix, real
    compute) plus a slow-dispatch run of `matmul_multicore_reuse_mcast`
    once that lands.
- **NCRISC / TRISC reset fan-out.** `cores.py:TensixCore.deassert_reset`
  is BRISC-only. `launch_msg.enables` arrives in a WRITE to L1 offset
  `0x20`; snapshot it on the way through, apply on DEASSERT.
  - *Test:* variant of `two/` with the BRISC enable bit cleared (NCRISC-
    only kernel) and a second variant running a TRISC-only compute with
    BRISC firmware quiescent — both should advance to `RUN_MSG_DONE`
    without the disabled core ever leaving reset.
- **`START` (cmd=4) handler.** `transport.py:_handle` log-and-skips it.
  Never observed in any captured trace; revisit if a future tt-metal
  release sends it.
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

The only outstanding RV item is functional-vs-perf:

- **No pipeline modelling.** Each instruction completes in one tick;
  no fetch/decode/issue latency, no memory-stall back-pressure. Fine
  for functional sim; a perf model would need this. Cross-ref §I
  "Cycle accuracy" — the per-unit cost framework is the home for it.
  - *Test:* no example needed — perf-model work.

---

## C. NoC (`tt_sim/network/tt_noc.py`)

- **NoC atomic ops beyond ATINC.** Atomic-add (`noc_semaphore_inc`)
  works via `RequestInitiator.handle_atomic_transfer`, posted and
  non-posted both. The richer atomic ops `ATCAS`, `ATSWAP`, `ATINCGET`
  issued by the Tensix ThCon backend are not implemented yet — they
  share the NoC-level dispatch but the actual op semantics live in
  §D "ThCon stalls".
  - *Test:* a non-posted-atomic kernel calling
    `noc_atomic_increment_with_response` and waiting on
    `NIU_MST_ATOMIC_RESP_RECEIVED` would cover the response-marked
    path that `driver/wormhole/nine/` doesn't.
- **NoC register coverage.** `tt_noc.py:748 NotImplementedError` —
  many reads beyond the basic counter set are unimplemented.
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

### Frontend / decode pipeline
- **Wait-gate stalls** on srcA/srcB availability not fully enforced.
  - *Test:* no example needed — exercised indirectly by every Tensix
    compute kernel; targeted regression would be a perf-model task.
- **Instruction issue latency.** Operations complete the cycle they
  decode; no realistic issue/decode separation.
  - *Test:* no example needed — perf-model work.

### Backend execution units
- **Matrix unit format conversions** — `backends/matrix.py:614`
  `NotImplementedError` for some BF8/FP4-style formats.
  - *Test:* variants of `four/` — `four-bf8/` and `four-fp4/` — running
    ELWADD with the missing input/output dataformats end-to-end.
- **Vector / SFPU lane-format edge cases** — `backends/vector.py:853,
  940` `NotImplementedError` (match-default in data-format conversion).
  Separately, `SFPMOV` with `FROM_SPECIAL` mod is partially unsupported
  at `backends/vector.py:284`.
  - *Test:* variant of `five-fp/` exercising the lane formats that
    currently raise (extend the existing `fp16` / `tf32` sub-directories
    with the remaining unsupported widths).
- **Packer config variants** — `backends/packer.py:263, 299, 309,
  427, 429, 440, 450, 461` `NotImplementedError` (ADC context,
  dest-addr offsets).
  - *Test:* small standalone kernel in `driver/wormhole/` driving a
    PACK with a non-zero ADC context and non-zero dest-address offset
    (e.g. packing a sub-region of `dst` into an L1 buffer at an
    arbitrary alignment); checks the packed bytes against a CPU-computed
    reference.
- **Unpacker NoOp modes & src-to-zero** — `backends/unpacker.py:78,
  390, 807, 860` `NotImplementedError`.
  - *Test:* unpacker micro kernel under `driver/wormhole/` that prefills
    srcA via UNPACR NoOp mode, then runs a UNPACR src-to-zero clear, and
    confirms downstream FPU sees zeroed inputs.
- **ThCon stalls** — Tensix-coprocessor-side atomic ops (`ATCAS`,
  `ATSWAP`, `ATINCGET`) issue but stall resolution is incomplete:
  `backends/thcon.py:81, 140, 166, 192, 406`. Note this is a separate
  code path from the NoC-level atomic-add that landed in §C — that
  one covers `noc_semaphore_inc` (cross-core L1 semaphore signalling
  over NoC), this one covers the Tensix coprocessor's own atomic
  unit operating on its private state.
  - *Test:* two-thread sync kernel — TRISC0 spins on `ATCAS` for a flag
    that TRISC1 flips via `ATSWAP`, with `ATINCGET` used to hand out
    monotonic indices. Driver asserts no spurious early wakeups.
- **Mover region-crossing** — single move limited to 16 KB and one
  region; `backends/mover.py:84, 95` `NotImplementedError`. Unmapped
  L0 writes silently dropped (`mover.py:81`).
  - *Test:* MOVER kernel issuing a 32 KB copy and a second copy that
    crosses an L0 region boundary; verify destination matches source
    byte-for-byte and that an unmapped destination raises (rather than
    silently dropping).

### Tensix instructions known-missing or stubbed
- `MOVDBGA2D`, `SHIFTXA`, `SHIFTXB`, `TRANSPSRCB`, `MOVB2D`, `MOVB2A` —
  handlers missing or incomplete.
  - *Test:* tile-transpose kernel that calls `TRANSPSRCB` plus
    `SHIFTXA`/`SHIFTXB` for column rotation and `MOVB2A`/`MOVB2D` for
    register marshalling; output tile must equal the transpose of the
    input.
- **Other SFPU handlers absent in `backends/vector.py`** (decode-only
  today): `SFPCAST`, `SFPDIVP2`, `SFPLZ`, `SFPLUT`, `SFPLUTFP32`,
  `SFPSWAP`, `SFPSHFT2`, `SFPTRANSP`, `SFPLOADMACRO`,
  `SFP_STOCH_RND`. (`SFPMOV` is implemented as of
  `backends/vector.py:29`.) Add as kernels demand.
  - *Test:* per-op micro kernel modelled on `five/` — one minimal
    SFPU-only example per instruction, asserting the lreg result
    against a Python reference.

### tt-metal `programming_examples/` via slow-dispatch
Run the canonical examples through tt-sim's wire bridge with:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_SIMULATOR=$(pwd)/driver/wormhole \
TT_METAL_HOME=/path/to/tt-metal \
[TT_SIM_TENSIX_COORDS=1-1,...] \
  $TT_METAL_HOME/build_Release/programming_examples/metal_example_<name>
```

Last full sweep: 13 of 14 pass (see commit log for the matrix).
Outstanding failure:

- **`metal_example_matmul_single_core`** — times out on its
  ``640 × 640 × 640`` bf16 matmul. Not a correctness gap; runs on a
  single Tensix tile, so the §A threading work does **not** unblock
  this one (the composite clock falls into the sequential path with
  one heavy tile). Outstanding levers: scale the test workload down
  (e.g. 64³), or reduce per-cycle overhead in the Tensix backend
  itself.

`metal_example_vecadd_multi_core` is also untested — it uses
`compute_with_storage_grid_size()` which is the full 8×8 worker grid
on Wormhole (64 cores). Materialisable; this one **is** the case §A
targets, gated on the §A perf follow-ups (heavy-only worker pool +
real-workload validation) landing.

### Numerical behaviour
- All compute is **FP32 internally** with conversion at the boundary;
  no rounding-mode overrides, no per-thread `FP16A_FORCE` enforcement
  (parsed, not applied), no accumulator-persistence modelling.
  - *Test:* no example needed — documentation/design decision; would
    only matter once a perf or bit-exactness model is in scope.

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
  surface (SFPU exp returning ≈ 1.0, eltwise compute producing 0s) are
  separate compute-path issues tracked under §D, not dispatch-path
  blockers.
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
- **Tile-control registers.** `misc/tile_ctrl.py:34, 52`
  `NotImplementedError` — only `SOFT_RESET`, `TRISC_PC_BUF_OVERRIDE`,
  wall-clock, and `DBG_FEATURE_DISABLE` are covered.
  - *Test:* driver-only script reading/writing each unimplemented
    tile-ctrl offset and asserting documented round-trip behaviour.
- **PC buffer.** `pe/pcbuf.py:20, 23` `TODO` — write delays /
  synchronisation not modelled.
  - *Test:* no example needed — perf/timing modelling.
- **Extra core types.** `device/tt_device.py:94` raises on anything
  beyond BRISC/NCRISC/TRISC0–2; needs widening if ERISC ever lands.
  - *Test:* covered by the §F ERISC example.

---

## F. Missing tile classes (whole subsystems absent)

These are present in the WormholeB0 docs but have no code under
`tt_sim/`:

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
  ethernet routing (no second chip exists) — blocks any multi-chip
  tt-metal program.
  - *Test:* tt-metal `programming_examples/hello_world_datatypes_kernel`
    (single-chip eth read at `(1,0)`) — structurally unblocked, gated
    on a re-run through the wire bridge to confirm; plus a multi-chip
    eth-ping example as a new `driver/wormhole/ten/` once chip-to-chip
    routing lands.
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

Current state — the gaps the §H plan is designed to close:

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
Validated on every wormhole example. Bus publish overhead is **88 ns/call
disabled** (target was <100 ns).

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

### Follow-ups (grouped by what gates them)

**Gated on §I cycle accuracy** — fields exist in the writers but the
simulator doesn't populate them yet. Land automatically as §I
provides the underlying state:

- Real Perfetto durations (today `dur=1` synthetic per cycle).
- NoC per-transaction `vc` / `issue_cycle` / `arrival_cycle`.
- FPU/SFPU active-vs-stalled cycles, stall reasons.
- Packer back-pressure, unpacker idle cycles.
- L1 bank conflict count.
- NoC VC occupancy.

**Multi-Tensix attribution (partly unblocked by §A)** — `chip_id` is
still hard-coded to 0, but per-tile `core_y/core_x` already fan out:
`TensixTile._register_trace_ids` uses the tile's own coord, so traces
from `driver/wormhole/seven/` carry the right unit IDs for each tile.
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
- **`libttsim.so` differential testing harness.** Phase 7 ships the
  comparison primitive (`diff_state`); orchestration to drive both
  simulators on the same kernel is left to the user / a separate
  effort.
- **tt-metal issue 28562 invariant catalogue.** Phase 7 ships four
  seed invariants; mining the linked issue tracker for
  domain-specific architectural rules would expand the catalogue
  substantially.
- **`hypothesis` property tests.** Needs a kernel generator —
  separate research project.
- **Determinism: tighten the host polling loop.** State dumps at
  `kernel_done` vary by up to 100 cycles per run because
  `wormhole_driver.py` polls every 100 cycles. Byte-exact regression
  comparison wants a tighter boundary.
- **L1 / DRAM content snapshotting** in state dumps. Today only
  registers + NoC counters. Add when a consumer needs it.
- **CI examples.** A fixed workload runs in CI and produces all
  eight output formats — catches schema regressions early.
- **Trace replay.** Re-run any writer offline against a captured
  JSONL stream, no re-simulation needed. Decouples writers from sim
  runs.
- **Aggregate performance-budget audit.** Phase 1 bench covers the
  bus only; total overhead with all writers enabled vs no tracing
  hasn't been measured against the documented ~30% slowdown
  target.
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

- **Per-unit cycle-cost tables.** Today every RV instruction, Tensix
  op, NoC request, and Mover transfer completes in the same tick it
  was issued. First step is a data-driven cost table per unit
  (cycles-per-op for FPU/SFPU/packer/unpacker/matrix; latency +
  bandwidth for NoC and DRAM; fetch/issue/retire stages for RV).
  Tables should live next to the YAML they parallel
  (`tensix_instructions.yaml` → `tensix_instruction_costs.yaml`).
  - *Test:* re-run `four/` and `five/` and compare reported cycle
    counts against expected ranges derived from the ISA-doc latency
    tables; assertion lives in the driver.
- **RV pipeline modelling.** Cross-ref §B "No pipeline modelling".
  Minimum viable: fetch/decode/issue/retire stages with memory-stall
  back-pressure on L1 / NoC reads.
  - *Test:* extend `driver/simple/ex2` (RV32I binary) with cycle
    assertions; verify a load from a stalled L1 region holds issue.
- **Tensix issue latency & wait-gates.** Cross-ref §D "Wait-gate
  stalls", §D "Instruction issue latency". Wait-gates on srcA/srcB
  availability and the documented per-op issue cadence both need to
  apply.
  - *Test:* targeted Tensix kernel under `driver/wormhole/` that
    issues an ELWADD with srcA not yet ready and asserts the MATH
    thread stalls for the documented number of cycles.
- **NoC timing model.** Cross-ref §C "Router arbitration", §C
  "Packet splitting". Per-hop latency, per-link bandwidth, head-of-
  line blocking on shared routers.
  - *Test:* multi-Tensix kernel from §A `seven/` extended with
    overlapping NoC bursts that should serialise at a shared router;
    driver compares observed completion order against the model.
- **DRAM access latency.** Today `DramCore` responds immediately; real
  silicon has bank-conflict and refresh-window costs. Lower priority
  than RV/Tensix/NoC.
  - *Test:* DRAM-heavy kernel (`three/` chunked DRAM reads) with
    cycle-count assertion.
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
  mapping every ISA-docs file under `WormholeB0/` to the tt-sim file
  that implements it (or "not modelled"). Closes the loop with the
  module-boundaries item — if a hardware block has no tt-sim home, it
  shows up here as a gap.
  - *Test:* no example needed — acceptance is the file existing and
    being grep-able for "not modelled" to enumerate gaps.

---

## K. Quick wins (small, well-isolated)

Loose ends that don't need design work:

- **`stub_listener.py` and `driver/wormhole/six/`** pre-existing ruff
  errors (untracked at time of last sweep). Either format or `.ruff.toml`
  exclude.
- **`run.sh` quoting / shellcheck.** Defensible as-is, but `shellcheck`
  surfaces a few suggestions.
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

### Discipline

- **Profile first, optimise second.** No "use Numba" or "rewrite in
  Cython" decision without trace data showing where the cycles go.
  Use `TT_SIM_TRACE_COUNTERS` on a known-slow run (`four/` matmul, or
  slow-dispatch `matmul_single_core` once it stops timing out) and
  partition wall-clock spend.
  - *Test:* output is a Parquet dataset + writeup refreshed in
    `driver/wormhole/docs/profiling.md` per major optimisation round.
- **Land §I event-driven cycle pump *before* JIT'ing.** The current
  tick-every-component pump is being replaced anyway; the new loop
  (typed event queue, integer cycle counts, state machines) is a much
  better JIT target than the current OOP-heavy one.
  - *Test:* covered by §I.

### Numba

Natural first JIT to try because it preserves the hackability pitch —
decorators, no build step. Realistic targets:

- **Tensix backend numeric inner loops** (matrix, FPU/SFPU element-
  wise). Already numpy; Numba win on top usually 2–3×.
  - *Test:* benchmark `four/` and `five/` wall-clock pre/post; assert
    no behavioural diff.
- **RV32IM execute** — only if rewritten table-driven over typed
  arrays. Significant refactor of `tt_sim/pe/rv/isa/i_isa.py`.
  - *Test:* `driver/simple/ex2`/`ex3` cycle-count regression +
    wall-clock measurement.
- **§I event-driven cycle pump** — best Numba target by construction.
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

### `nogil=True` — revive §A threading

§A threading is structurally correct but a perf regression because GIL
contention dominates per-cycle Python work. Numba can release the GIL
via `@njit(nogil=True)`. Wrapping Tensix backend hot inner ops this way
gives threading a real path to a speedup that doesn't depend on Python
3.13t — orthogonal to single-thread perf and arguably the most
interesting near-term Numba angle.

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
