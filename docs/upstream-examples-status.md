# Upstream `programming_examples/` on tt-sim — status

A breadth sweep of tt-metal's **upstream** `programming_examples/` against tt-sim
on both architectures. These are programs nobody in this repo wrote, so they are
the evidence behind the claim *"runs real tt-metal programs unmodified"* in a way
the in-tree `examples/` ladder cannot be.

**The sweep is now a gate.** It lives in
[`driver/tests/upstream_sweep.py`](../driver/tests/upstream_sweep.py), carries its
own expected result, and prints `RESULT: PASS` / `RESULT: FAIL`:

```bash
source /path/to/venv/bin/activate
python3 -m driver.tests.upstream_sweep              # fast tier, both arches, ~8 min
python3 -m driver.tests.upstream_sweep --tier full  # + the four heavy programs
python3 -m driver.tests.upstream_sweep --list       # the table and its coverage
```

The runbook for driving these programs by hand is
[`running-tt-metal-on-the-simulator.md`](running-tt-metal-on-the-simulator.md); the
differential method used to triage a failure is `optests/diff.sh` (see
[`plans/blackhole-support.md`](plans/blackhole-support.md) Phase 8).

## What was tested against

| | |
| --- | --- |
| tt-sim | `d094097` (2026-08-13) plus this change, which adds the gate and these docs and touches no simulator file |
| tt-metal | `0.74` at `$TT_METAL_HOME`, prebuilt `build/programming_examples/` |
| oracle | a ttsim checkout (`$TTSIM_ROOT` below) — `oracle-wh/libttsim_wh.so`, `oracle-bh/libttsim_bh.so`. **No row needed triage** (nothing failed); it was used once, to settle `matmul_single_core`'s PCC. It is the **functional** oracle only, never a cycle oracle. |
| flow | `TT_METAL_SLOW_DISPATCH_MODE=1` (every upstream example calls `EnqueueProgram`); `TT_METAL_SIMULATOR` selects `driver/wormhole` or `driver/blackhole` |
| worker tiles | **nothing set.** No `TT_SIM_TENSIX_COORDS`, no `TT_SIM_TENSIX_CORES`, no `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` |

**Use the right tt-metal.** An older 0.70.1 checkout may also be on the box, and
fails these examples at *compile* time with `'SrcOrder' has not been declared`,
which looks like a simulator bug and is not one. That misreading has already cost
this project a wrong conclusion once.

## What changed since the previous sweep

The previous sweep (tt-sim `fe0d279` plus an uncommitted tree, 2026-08-03) is
superseded in two ways.

**Its baseline predates a day of deep change.** Since `fe0d279`: workers
materialise on demand (the NoC destination-resolution path was rewired), the NoC
contention term went from never firing to firing, Blackhole DRAM cycles moved on
13 guards, the Configuration Unit and Matrix Unit gained pipeline residency,
`STALLWAIT`'s decode and empty-mask default changed, and the host-stop path
changed how every run terminates. All of that is covered by the unit suite and
the cost-model gate; **none of it had been exercised against ~20 real upstream
programs on both architectures.** This sweep is that check.

**Its method is obsolete.** The old sweep pinned `TT_SIM_TENSIX_COORDS` per
example, and most of its grid-sized rows additionally pinned
`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4` to shrink the compute grid to a
4×5 = 20-worker sub-block. As of 2026-08-12 **setting `TT_SIM_TENSIX_COORDS` at
all pins the worker pool and switches off on-demand materialisation** — the exact
bug that kept the congestion probes from reaching tt-sim for months. So this
sweep sets **no grid variable at all**, which both tests the path a user with an
empty environment actually gets and deletes the per-example coordinate table the
old document carried. There is nothing left in it to keep.

The consequence is that the grid-sized programs now run at the **real** default
compute grid — 8×9 = 72 workers on Wormhole, 13×10 = 130 on Blackhole — rather
than the 20-worker sub-block the old sweep measured. That is more coverage, not
less.

## Results

Legend: **PASS** = exit 0, the program's own success line present, and no
forbidden line (`Some results did not match`, `PCC not high enough`, a mismatch
report, `can not handle instruction`, a simulator traceback). Several upstream
examples print per-element mismatches and still exit 0, so exit status alone is
not a correctness signal and the gate does not treat it as one.

The `check` column says how strong each row's verdict is — the gate refuses to
launder a completion into a correctness claim:

- **self** — the program validates its own result.
- **value** — no upstream self-check, but the gate checks the arithmetic it
  prints.
- **completion** — no value is checked at all. Running to its own final line is
  the whole signal.

### Fast tier — 17 programs × 2 arches, 34/34 PASS

| program | check | tt-sim WH | tt-sim BH |
| --- | --- | --- | --- |
| `add_2_integers_in_riscv` | self | **PASS** (9 s) | **PASS** (8 s) |
| `add_2_integers_in_compute` | self | **PASS** (8 s) | **PASS** (8 s) |
| `hello_world_datamovement_kernel` | completion | **PASS** (8 s) | **PASS** (8 s) |
| `hello_world_compute_kernel` | completion | **PASS** (8 s) | **PASS** (8 s) |
| `hello_world_datatypes_kernel` | completion | **PASS** (9 s) | **PASS** (8 s) |
| `loopback` | self | **PASS** (9 s) | **PASS** (9 s) |
| `eltwise_binary` (64 tiles, FPU add) | self | **PASS** (13 s) | **PASS** (11 s) |
| `eltwise_sfpu` (64 tiles, `exp_tile`) | self | **PASS** (32 s) | **PASS** (27 s) |
| `custom_sfpi_add` | self | **PASS** (20 s) | **PASS** (19 s) |
| `custom_smoothstep` | self | **PASS** (28 s) | **PASS** (31 s) |
| `sfpu_eltwise_chain` (softplus, PCC > 0.999) | self | **PASS** (9 s, PCC 0.9998722) | **PASS** (11 s, PCC 0.99986756) |
| `contributed/vecadd` | value | **PASS** (12 s) | **PASS** (14 s) |
| `noc_tile_transfer` (2 workers, semaphores) | self | **PASS** (8 s) | **PASS** (10 s) |
| `contributed/multicast` (4 workers, NoC multicast) | self | **PASS** (10 s) | **PASS** (13 s) |
| `vecadd_sharding` (4 workers, L1-sharded) | self | **PASS** (10 s) | **PASS** (13 s) |
| `shard_data_rm` (4 workers) | completion | **PASS** (9 s) | **PASS** (11 s) |
| `pad_multi_core` (4 workers) | completion | **PASS** (29 s) | **PASS** (35 s) |

### Full tier — 4 programs × 2 arches

Each fills the whole default compute grid except `matmul_single_core`, which is
deliberately one worker.

| program | workers used | tt-sim WH | tt-sim BH |
| --- | --- | --- | --- |
| `vecadd_multi_core` (640 tiles) | 72 / 130 | **PASS** (100 s) | **PASS** (173 s) |
| `matmul_single_core` (640³) | 1 | **PASS** (459 s, PCC 0.9810914) | **PASS** (493 s, PCC 0.9802104) |
| `matmul_multi_core` (640³) | 72 / 130 | **PASS** (568 s, PCC 0.99993193) | **PASS** (671 s, PCC 0.99984235) |
| `matmul_multicore_reuse` (640³) | 72 / 130 | **PASS** (579 s, PCC 0.99930096) | **PASS** (602 s, PCC 0.99930096) |

The three matmuls check themselves by Pearson correlation against a host-computed
golden and `TT_FATAL` below PCC 0.97, so a PASS is already a PCC claim; the gate
echoes the reported figure next to the verdict so a green run leaves a number
behind. Two things in that column are worth reading:

- **`matmul_multicore_reuse` gives 0.99930096 on both architectures** — identical
  to eight digits, identical to the figure the old sweep measured at a 4×5 grid,
  and identical to the oracle's. Different grid, different blocking, same answer.
- **`matmul_single_core` sits at 0.98**, an order of magnitude further from 1
  than the other two. **That is the program, not tt-sim.** Checked directly
  against the functional oracle on the same binary: ttsim-WH returns **0.9797904**
  and ttsim-BH **0.98187274**, so tt-sim's Wormhole figure is if anything the
  closer to 1 of the pair. Upstream's own threshold for this program is 0.97 for
  the same reason — 640-deep bfloat16 accumulation on one core.

Each of the three heavy programs was run twice on each arch (once before the PCC
capture was added, once after); the verdicts agreed and the wall clocks moved by
under 12 %.

**Whole sweep: 42/42 runs PASS, both architectures, no environment variable.**
Timings are single runs on a machine with other agents' work on it — read them as
shape, not as a benchmark, and note that this repository's rule for any *speed*
claim is an interleaved A/B on a frozen worktree, which none of these are.

### `distributed/`

`distributed_buffer_rw`, `distributed_eltwise_add`, `distributed_program_dispatch`,
`distributed_trace_and_events`: excluded, and **re-checked first-hand on
2026-08-13** rather than inherited from the old sweep. All four abort identically
on tt-metal 0.74:

```
TT_FATAL … MeshShape([2, 4]) requires 8 devices, but only 1 devices are
available in the system mesh MeshShape([1, 1]).   (assert.hpp:104)
```

These are 8-device (T3000) programs, and the abort is in the **host**, before it
asks the simulator for anything. Nothing to do with tt-sim; the old sweep
recorded the same failure against both oracles.

## Movement against the previous sweep

**Regressions: none.** No program that passed in the 2026-08-03 sweep fails now,
on either architecture. That is the load-bearing result: the day of deep change
listed above broke nothing that real tt-metal programs reach.

Improvements, in rough order of how much they buy:

1. **The per-example coordinate table is gone, and with it a class of silent
   wrong answers.** All 42 runs set no grid variable. The old sweep's table of
   `TT_SIM_TENSIX_COORDS` values per example is not merely unnecessary now — it
   was the mechanism by which a program addressing an unlisted worker got its
   traffic NullCore-swallowed and returned zeros. There is nothing left in that
   table to keep.
2. **The grid-sized programs now run at the real default compute grid**: 8×9 = 72
   workers on Wormhole, 13×10 = 130 on Blackhole. The old sweep ran the three
   grid-sized ones (`vecadd_multi_core`, `matmul_multi_core`,
   `matmul_multicore_reuse`) under `TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4`,
   i.e. a 4×5 = 20-worker sub-block, and its one full-grid attempt
   (`vecadd_multi_core` at 8×10) was a **CRASH** it never re-ran after the bug-4
   fix landed. Those rows are now the *real* shape of
   the program, which is strictly more coverage — a different work split, a
   different blocking, the same golden.
3. **`matmul_single_core` went TIMEOUT → PASS on Wormhole**, and gained a
   Blackhole row it never had ("not run — same size"). It now finishes inside the
   old sweep's own 900 s budget, at 467 s. That is a verdict change, not a timing
   measurement — these are single runs on a loaded machine and no A/B was done.
4. **`matmul_multicore_reuse` and `matmul_multi_core` need no poll-rate knob.**
   Every grid-sized row in the old sweep set `TT_SIM_CYCLES_PER_POLL=10`; every
   row here uses the default 100. That knob's retraction (recorded in
   [`running-tt-metal-on-the-simulator.md`](running-tt-metal-on-the-simulator.md)
   §4.1) is confirmed against the heaviest programs in the set.
5. **`contributed/vecadd` is no longer scored `PASS*`.** It ships with no
   self-check and seeds from `std::random_device`, so the old sweep could only
   score it on "did it crash". The gate runs it with `--seed 42` and checks the
   ten sums it prints, so it is now a value row. Its previous verdict — CRASH,
   bug 1 — is confirmed fixed on both arches.
6. **`sfpu_eltwise_chain` reports PCC 0.99986–0.99987 on both arches**
   (0.9998585 and 0.9998722 over two Wormhole runs, 0.99986756 on Blackhole) —
   inside the 0.99985–0.99988 band the old sweep measured for the *oracle* over
   8 runs. That is the figure bug 3 moved: tt-sim's pre-fix band was
   0.99860–0.99874, an order of magnitude further from 1 and not overlapping the
   oracle's. The example seeds from `std::random_device`, so this is a band, not
   a number.
7. **Two `contributed/` programs and four Blackhole SFPU programs stay fixed.**
   `contributed/vecadd`, `contributed/multicast` (bug 1) and `eltwise_sfpu`,
   `custom_sfpi_add`, `custom_smoothstep`, `sfpu_eltwise_chain` on Blackhole
   (bug 2) were all CRASH in the first pass of this sweep. All six are green, and
   are now held green by the gate rather than by a paragraph.

## What the full grid actually costs

Kept because other documents cite this section by name. **Its conclusion has been
overtaken**, and the numbers that supersede it are in
[`running-tt-metal-on-the-simulator.md`](running-tt-metal-on-the-simulator.md)
§1.3 and §1.3.2 — read those, not this.

The 2026-08-03 finding was that a materialised worker a program never launches on
does **not** stay dormant: tt-metal's grid-wide init handshake releases BRISC from
soft reset on *every* declared worker, which then spins in the firmware loop until
teardown, so wall clock was linear in materialised tiles. Measured then:
`vecadd_sharding`, which launches on exactly 4 workers, took 14 s with 4
materialised and 255 s with 80.

That finding produced a prediction — *"making the full grid affordable needs
firmware-loop recognition: a BRISC spinning on a go-message poll is
architecturally idle even though it retires instructions"* — and **the prediction
was right and the fix has landed** (`tt_sim/pe/rv/spin.py`). Together with
on-demand materialisation, the premise the section rests on no longer applies: a
program is charged for the workers it uses, not for the grid it was declared on,
and wall clock is now roughly *flat* in worker count for a fixed problem
(1 → 80 workers is ~1.8× — see §1.3.2). This sweep is consistent with that:
`vecadd_multi_core` runs on the full 72-worker Wormhole grid in 100 s.

The same study measured the *idle* floor with striding off, which is what a single
live tile forces: 124 k cycles/s at one worker down to **3.1 k** at 80 (324
µs/cycle). A dormant tile clock cost ~0.24 µs/cycle and the **deadlock watchdog** a
further ~3.0 µs per Tensix tile per cycle, because `DeadlockDetector.tick` walked
every tile and every core unconditionally — **92 % of the all-dormant floor at 80
workers**. That is the number
[`driver/wormhole/docs/profiling.md`](../driver/wormhole/docs/profiling.md#measure-first-where-the-watchdogs-time-actually-went)
cites; the watchdog is now sampled rather than per-cycle.

Two paragraphs of the original were retracted at the time and are not reproduced:
an awake-tile watchdog A/B that was run on an architecture with no watchdog wired,
and the advice to set `TT_SIM_CYCLES_PER_POLL=10` on wide grids, which stopped
reproducing once firmware-loop parking landed.

## The gate

`driver/tests/upstream_sweep.py`. Two tiers, because a gate that takes an hour is
not a gate:

| tier | what | wall clock | when to run it |
| --- | --- | --- | --- |
| `fast` (default) | 17 programs × 2 arches | **~8 min** | every change |
| `full` | + `vecadd_multi_core`, `matmul_single_core`, `matmul_multi_core`, `matmul_multicore_reuse` × 2 arches | **~70 min** (28 min WH + 32 min BH of heavy programs, on top of the fast tier) | deliberately, before a release |

```bash
python3 -m driver.tests.upstream_sweep                       # fast, both arches
python3 -m driver.tests.upstream_sweep --tier full           # everything
python3 -m driver.tests.upstream_sweep --arch wormhole sfpu  # filter by name
python3 -m driver.tests.upstream_sweep --list                # coverage statement
python3 -m driver.tests.upstream_sweep --record              # emit a new EXPECTED
```

The expected verdict per (arch, program) is recorded in `EXPECTED`, which is
**empty** — every program the gate runs is expected to pass on both
architectures, so the gate's recorded result is simply *all green*. A verdict
that differs from the record fails the gate **in both directions**: a
newly-passing program is as much a signal to act on (record it) as a
newly-failing one.

It sets `TT_METAL_SIMULATOR` itself, from the repo it lives in, and *removes*
`TT_SIM_TENSIX_COORDS` / `TT_SIM_TENSIX_CORES` /
`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE` from the environment it hands each
program — an inherited one would silently change what is being tested. It skips
cleanly (exit 0, `SKIP:` line) where tt-metal is not built, so it is safe to
invoke anywhere. It reaps only the simulator servers it started (a per-run
`TT_SIM_RUN_TAG`), so it is safe to run beside another live run.

**Checked to bite**, which is the only thing that makes a green gate mean
anything:

- Recording `add_2_integers_in_riscv` as an expected `FAIL` and re-running turns
  the row `BAD` and the gate `RESULT: FAIL`, exit 1.
- `driver/tests/upstream_sweep_test.py` (7 tests, no tt-metal needed) covers the
  one real value check the gate performs — a wrong sum, a truncated printout and
  bfloat16 rounding all come out the way they should — plus the table's shape and
  the `EXPECTED` record naming only programs the gate actually runs.

**What a green gate establishes**: every listed program reaches its own success
criterion on both simulated architectures with no environment variable set.
**What it does not**: anything about cycle counts — no timing is compared to
anything — and nothing about the values computed by the five `completion` rows,
which upstream ships with no self-check. Depth for four of those programs lives
in the offline replay guards
(`driver/{wormhole,blackhole}/server/*_replay_test.py`), which pin actual bytes.

## Excluded, and why

Every exclusion is reasoned. **A tt-sim bug belongs in `EXPECTED` as a recorded
failure, not here** — nothing is excluded to hide a failure.

- **`distributed/*` (4 programs)** — need an 8-device mesh (`MeshShape([2, 4])`).
  Re-verified first-hand on 0.74; the host aborts before it reaches the
  simulator. Not a simulator issue.
- **`matmul_multicore_reuse_mcast`** — 2048×1024×512 = 32768 tile-matmuls, four
  times `matmul_multi_core`. Killed at 46 minutes on Blackhole in the previous
  sweep with no `[DEADLOCK]` and still progressing: **throughput, not
  correctness**. It passes on both oracles. Excluded until tt-sim is fast enough
  to finish it; the moment it is, it belongs in the full tier.
- **Odd-width compute grids** (`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=2,2`,
  `=2,4`) — the **host** segfaults in ~1 s, before it talks to the simulator, and
  does so identically against the oracle. Not something the gate exercises at
  all now that it sets no override.
- **Fast dispatch** — blocked at the protocol level upstream, not by anything in
  this repo. See `CLAUDE.md`; the same wall applies to Tenstorrent's own
  simulator.

`matmul_single_core` is **not** excluded, despite being the slowest single-worker
program in the set. It is in the full tier with a 2400 s budget, and its long run
is arithmetic (8000 tile-matmuls on one worker), not a hang.

## Open, and not acted on here

This pass found no new tt-sim bug — every run passed — so nothing was fixed and
nothing needed to be. What it leaves behind:

1. **`matmul_multicore_reuse_mcast` is still out of reach.** 32768 tile-matmuls
   against a ~10-minute-per-640³-matmul budget. It is the only upstream program
   the gate cannot run, and the only thing standing between it and full coverage
   is tt-sim's own wall clock. It passes on both oracles.
2. **The NoC-1 directory is still ambiguous on both architectures** (bug 4's
   latent half). It is inert because the response direction no longer consults it,
   and `tt_sim/network/noc_routing_test.py` pins that; a new coordinate-keyed
   lookup on NoC 1 would revive it. Not a defect to fix today, but the kind of
   thing worth knowing before touching NoC routing.
3. **Five rows check no value.** The three `hello_world_*` programs and
   `pad_multi_core` / `shard_data_rm` have no upstream self-check. Two of the five
   are covered in depth by replay guards; the three `hello_world_*` are not, and
   could be, by diffing their DPRINT output against the oracle's. Cheap, and
   nobody has done it.

## Genuine tt-sim bugs found by this sweep

All four were found by the earlier passes of this sweep, all four are fixed, and
all four are frozen as guards. They are kept here because the reasoning is the
record of *why* those guards exist, and because two of them left latent
observations that are still true. Each was confirmed by running the *same binary*
against ttsim first — that discipline is why these findings are trustworthy, and
it is why the sweep's failure column can be believed.

### Bug 4 — NoC-1 responses were routed by the requester's NoC-0 coordinate (Wormhole) — FIXED

**Symptom.** Every Wormhole multi-core run whose compute grid included worker
column `x=4` killed the simulator server:

```
KeyError: 0
  tt_sim/network/tt_noc.py:1090  NUI.clock_tick
```

Hit by `vecadd_multi_core` (at 4×2, 4×5 **and** the full grid) and by
`matmul_multi_core` at 4×5; timing-independent (same crash at
`TT_SIM_CYCLES_PER_POLL` 10 and 100). Both programs passed on ttsim-WH in 2–32 s.

**Cause.** `NUI.id_pair` is the tile's SoC-physical / NoC-0 coordinate on *both*
NoCs, and every response was routed with `noc_directory[noc_request.source]` — but
`noc_1_directory` is keyed by **NoC-1 (mirrored)** coordinates. On Wormhole the
mirror is `(9 - x, 11 - y)`, so the DRAM column `x=5` mirrors onto worker column
`x=4` and shadows exactly three of the twenty workers in the 4×5 block —
`(4,2)`, `(4,3)`, `(4,4)`, the mirrors of DRAM channels 3, 4 and 5.

**Fix.** A NoC response now goes back to the *endpoint that issued the request*
(`NoCDataRequest.reply_to`, funnelled through the single `NUI.send_response`)
instead of being looked up by coordinate. That removes the only coordinate lookup
the response direction ever had, so the collision is impossible rather than
merely absent. `NoCDataRequest.source` was renamed `source_coord` so the field
that is *not* a routing key no longer reads like one.
`tt_sim/network/noc_routing_test.py` freezes both halves with no tt-metal, no
socket and no oracle.

**The latent half is still true.** NoC 1's directory really is ambiguous on
**both** arches — with all 140 Blackhole workers built, 102 worker coords in
`noc_1_directory` resolve to a different tile. Wormhole's DRAM placement merely
surfaced it earlier. It is inert only because the response direction no longer
consults that directory; a new coordinate-keyed lookup on NoC 1 would revive it.

### Bug 1 — top-down DRAM allocation fell outside the modelled DRAM (both arches) — FIXED

**Symptom.** `contributed/vecadd` and `contributed/multicast` killed the server on
**both** arches:

```
IndexError: Provided address '0x3fffd000' does not match any registered memory spaces
```

**Cause.** `DRAMTile.__init__` registered exactly two **10 MiB** banks, at `0x0`
and `0x4000_0000`, so everything from `0xA0_0000` to `0x4000_0000` was unmapped.
tt-metal's *default* allocation is bottom-up and stays inside the modelled 10 MiB
— which is why every other example worked. `DeviceLocalBufferConfig{.bottom_up =
false}`, which both `contributed/` examples pass, allocates from the **top** of
the bank and landed in the hole.

**Fix.** `DRAMTile` now registers each channel whole, sized from
`ArchProfile.dram_channel_size` (the SoC descriptors' `dram_bank_size`:
`0x8000_0000` Wormhole, `0xFF00_0000` Blackhole), over a chunk-on-demand
`SparseAddressableMemory` — 12 GiB / 32 GiB cannot be allocated eagerly, and the
vendor reference simulator faults its channels in lazily for the same reason. The
old two-bank layout was conceptually wrong as well as too small: tt-metal's
`dram_view_size` views are offsets *inside* one channel, not separate memories.
Frozen as `driver/blackhole/server/dramtop_replay_test.py`.

### Bug 2 — Blackhole SFPU computed an out-of-range Dst row with a bfloat16 Dst — FIXED

**Symptom.** On Blackhole, any SFPU op on a default (non-`fp32_dest_acc_en`)
bfloat16 Dst killed the server with `IndexError: index 8192 is out of bounds for
axis 0 with size 1024`. Every SFPU example on Blackhole hit it: `eltwise_sfpu`,
`custom_sfpi_add`, `custom_smoothstep`, `sfpu_eltwise_chain`.

It was **two** independent Blackhole bugs, and the config-space lead first
suspected was a red herring — regenerating `tensix_backend_cfg_blackhole.yaml`
from ttsim's `data/bh/tensix_regs.json` diffs clean on every field.

1. **The 8192 came from `imm10` itself, not a config read.** `SFPLOAD`/`SFPSTORE`
   `dest_reg_addr` is bits **9:0** on both arches, but `tensix_instructions.yaml`
   stores only each field's start bit and the decoder inferred the width from the
   *next* field's start — so with `sfpu_addr_mode` at start 14 it decoded bits
   13:0. On Blackhole `sfpu_addr_mode` is 15:**13** (Wormhole 15:14) and sfpi
   emits addr mode 3, so raw bit 13 was set on essentially every SFPU Dst access
   and `+0x2000` leaked into the row. `_read_dest_reg_addr` now masks to the
   documented 10 bits.
2. **`SFPGT`/`SFPLE` ignored `instr_mod1`.** mod1 1 sets the lane flags; mod1 8
   writes an all-ones / all-zero **mask into VD** and leaves the flags alone.
   Blackhole's `exp_tile` uses the mask form, so dropping it left the exponent
   unmasked and `exp(0)` came back as `0x0280` (≈1.9e-37) instead of `0x3f80`.
   Comparisons now use the sign-magnitude total order and any other modifier
   raises.

**Why the existing Blackhole guards missed it.** `optests/sfpumath` — the only BH
SFPU guard at the time — ran with `fp32_dest_acc_en = true`, taking the FP32
branch of `get_dst_address`. The default bfloat16-Dst path that every stock
`init_sfpu` + `exp_tile` kernel uses had no coverage. Now frozen as
`driver/blackhole/server/sfpuchain_replay_test.py`, bit-exact against the ttsim
golden on all 5120 elements.

### Bug 3 — RV32I `sh` wrote one byte, not two (both arches) — FIXED

**Symptom.** `sfpu_eltwise_chain` aborted on tt-sim WH with `PCC not high enough.
Result PCC: 0.9986145, Expected PCC: 0.999`. The example seeds from
`std::random_device`, so a single PCC proves nothing; repeating settled it —
**ttsim over 8 runs = 0.99985–0.99988** (spread 3e-5), **tt-sim over 6 runs =
0.99860–0.99874** (spread 1.4e-4). Two non-overlapping bands, an order of
magnitude apart.

**Cause.** `RV_I_ISA.handle_s_store` sliced the source register as `rs2_val[0:1]`
and handed it to `conv_to_bytes(..., 2)`, which passes a `bytes` value through
verbatim and ignores the width — so **every 16-bit store wrote a single byte** and
left the upper byte at its previous value. The kernel builds its ones tile in L1
with exactly that (`ptr[i] = fp32_to_bf16_truncate(1.0f)`), so the tile came out
`0x0080` (a bfloat16 denormal) instead of `0x3F80` and `add_binary_tile` added
nothing: the chain computed `log(exp(x))`. Since `log(exp(x)) = x` and
`log(1 + x) ≈ x` on `[0, 1)`, softplus is nearly linear there, which is why a
whole-datapath bug showed up as a marginal 0.9986 PCC — **the upstream self-check
is only just strong enough to catch it.**

The bug was in shared RV32I code, so Blackhole had it too; it was merely hidden
behind bug 2, which killed every bfloat16-Dst SFPU kernel before the result was
checked. Fixed in `tt_sim/pe/rv/isa/i_isa.py`; unit-guarded by
`test_stores_write_their_full_width` in `i_isa_test.py` and end-to-end by
`driver/{wormhole,blackhole}/server/softplus_replay_test.py`.

## Coverage this buys

Paths these upstream programs exercise that the in-tree `examples/` ladder and the
`optests/` differential programs do not:

- the **default bfloat16 Dst SFPU path** on Blackhole (bug 2) — every stock
  `init_sfpu` + `*_tile` kernel uses it;
- **top-down DRAM allocation** (bug 1) — the whole DRAM address space above
  10 MiB;
- **chained SFPU ops inside one `tile_regs_acquire`** and SFPU binary ops reading
  a second Dst tile (bug 3);
- **`TensorAccessor` / `noc_async_read_page` / `noc_async_write_page`** dataflow,
  and BRISC-generated tile contents (16-bit stores into an L1 CB) — where bug 3
  turned out to live;
- **multi-core data movement** at 2 and 4 workers with semaphores, NoC multicast
  and sharded row-major buffers;
- **L1-sharded buffers with no data-movement kernel at all** —
  `vecadd_sharding` writes its inputs straight into the workers' L1 and runs only
  a compute kernel over CBs bound to those L1 addresses;
- **the full default compute grid**, 72 workers on Wormhole and 130 on Blackhole,
  all doing DRAM I/O at once — the shape that surfaced bug 4, and which this
  sweep is the first to reach without a grid override.

## Frozen guards

The gate checks each program's *verdict*. Where actual **bytes** are pinned is the
offline replay guards, and this sweep is what put seven of them there — four
capturing an upstream program directly, three capturing an `optests/` reproducer
distilled from a bug it found. Upstream traces are recaptured with
`driver/tests/capture_upstream_traces.sh`.

Upstream programs, frozen:

| guard | what it pins |
| --- | --- |
| `{wormhole,blackhole}/noc_tile_transfer_replay_test.py` | all 1024 `uint16` elements of the destination tile |
| `blackhole/vecadd_sharding_replay_test.py` | 16384 elements, `c == a + b` bit for bit, golden read back out of L1 |
| `blackhole/pad_multi_core_replay_test.py` | 2048 interleaved destination pages against a computed golden |
| `blackhole/shard_data_rm_replay_test.py` | the four workers' CB `c_0` contents against a computed golden |

Reproducers distilled from the bugs above, frozen:

| guard | the bug it regression-tests |
| --- | --- |
| `blackhole/sfpuchain_replay_test.py` | bug 2 — 5120 bfloat16 elements across five SFPU ops, bit-exact vs ttsim |
| `blackhole/dramtop_replay_test.py` | bug 1 — a `.bottom_up = false` buffer at the top of a DRAM channel |
| `{wormhole,blackhole}/softplus_replay_test.py` | bug 3 — the upstream softplus chain, both arches |

Each was checked to actually bite: corrupting one operand byte in a copy of the
trace turns each of them red, and `vecadd_sharding` goes red both when its golden
is perturbed by a single bfloat16 ulp and when its output shard is read one tile
off.

**Deliberately not frozen: `matmul_multi_core` / `matmul_multicore_reuse`.** They
would roughly quintuple the cost-model gate's wall clock on their own, and the
paths they cover are covered qualitatively by `pad_multi_core` and by the in-tree
`six` / `matmulblock`. What they would uniquely add is *scale*; the full tier of
the gate now runs them live instead, which buys the same coverage without a
committed trace. Revisit if a routing or contention bug of bug 4's class recurs.

## Reproducing a single row by hand

```bash
source /path/to/venv/bin/activate
cd "$TT_METAL_RUNTIME_ROOT/build/programming_examples"

# Wormhole (the venv's default), no grid variable of any kind:
./metal_example_eltwise_sfpu

# Blackhole:
TT_METAL_SIMULATOR=$HOME/tt-sim/driver/blackhole ./metal_example_eltwise_sfpu

# differential against the oracle, per Tensix op:
TT_SIM_ARCH=wormhole ./optests/diff.sh softplus
```

**Running a whole upstream program against the oracle needs one staging step.**
UMD reads `<dir-of-the-.so>/soc_descriptor.yaml`, and ttsim's oracle directories
ship the descriptor under its architecture name, so pointing `TT_METAL_SIMULATOR`
straight at `oracle-wh/libttsim_wh.so` dies with
`YAML::BadFile: bad file: …/oracle-wh/soc_descriptor.yaml`. Symlink a staging
directory instead — which is exactly what `optests/diff.sh` does for you:

```bash
TTSIM_ROOT=/path/to/ttsim          # the checkout holding oracle-wh/ and oracle-bh/
mkdir -p /tmp/oracle-wh && cd /tmp/oracle-wh
ln -sf "$TTSIM_ROOT"/oracle-wh/libttsim_wh.so .
ln -sf "$TTSIM_ROOT"/oracle-wh/wormhole_b0_80_arch.yaml soc_descriptor.yaml
cd "$TT_METAL_RUNTIME_ROOT/build/programming_examples"
TT_METAL_SIMULATOR=/tmp/oracle-wh/libttsim_wh.so ./metal_example_matmul_single_core
# -> Metalium vs Golden -- PCC = 0.9797904 / Test Passed
```

UMD spawns the server as `python3 -u -m driver.<arch>.server`, and test scripts
used to clean up with a `pkill` on that pattern — so a concurrent run would kill a
long live run out from under you and the host would hang silently. Every script
here now stamps a per-run tag into the server's command line (`TT_SIM_RUN_TAG` →
`--run-tag`, see `driver/sim_procs.sh`) and kills only its own servers plus
servers left tagged by a run that has since died. A manual run like the ones above
carries no tag at all, so nothing reaps it. The one thing that still kills
everything is the explicit opt-in `TT_SIM_KILL_ALL_SERVERS=1`.
