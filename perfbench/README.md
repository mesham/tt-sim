# perfbench — cycle-cost measurement programs

A third program tree, alongside `examples/` (functional, arch-agnostic tt-metal
programs) and `optests/` (differential op tests against the vendor reference
simulator). These are **timing** programs: real tt-metal binaries whose output
is device cycle counts, built so that the *same binary* runs on silicon and
against tt-sim and the two can be diffed.

```
perfbench/
├── run_card_session.sh ONE card session: every silicon probe the roadmap wants
├── run.sh              simulator-side runner (arch, coords, venv, cost model)
├── tensixbench/src/    per-instruction Tensix cycle costs
├── riscvbench/src/     the baby RISC-V front end: issue rate, `.ttinsn` push
│                       cost, branch cost, instruction fetch, the Tensix
│                       instruction queue's depth and whether it is shared
├── nocbench/           NoC congestion: latency against hop count, and against
│                       the number of links two concurrent flows share
└── nocreadbench/       what caps the sustained NoC *read* rate: the initiator's
                        outstanding-request counter, read directly, plus the
                        source-fan-out axis tt-metal's dataset cannot express
```

The first two are complements, and the second exists because of the first's headline
result: `tensixbench` measures what a Tensix unit costs, and found that against
tt-sim **every** probe of **every** unit reads exactly 1.000 cycles because
nothing back-pressures the core that issued it. `riscvbench` measures that core.

| | tensixbench | riscvbench | nocbench | nocreadbench |
| --- | --- | --- | --- | --- |
| **Run it on hardware** | `tensixbench/run_card.sh` — or [its README](tensixbench/README.md) | `riscvbench/run_card.sh` — or [its README](riscvbench/README.md) | `nocbench/run_card.sh` — or [its README](nocbench/README.md) | `nocreadbench/run_card.sh` — or [its README](nocreadbench/README.md) |
| **Why it is shaped this way** | [`../docs/plans/tensix-cost-benchmark.md`](../docs/plans/tensix-cost-benchmark.md) | [`../docs/plans/riscv-front-end-benchmark.md`](../docs/plans/riscv-front-end-benchmark.md) | [`../docs/plans/cost-model.md`](../docs/plans/cost-model.md), "Rung 2" and its addendum | [`../docs/plans/cost-model.md`](../docs/plans/cost-model.md), "The read floor" |
| **Analyse the results** | `python3 -m tt_sim.perf.tensix_bench_sweep --measured <csv>` | `python3 -m tt_sim.perf.riscv_bench_sweep --measured <csv>` | `python3 -m tt_sim.perf.noc_congestion_sweep --measured <csv>` | by hand, against the README's prediction table |

`nocreadbench` is the newest and the only one whose *most important* reading
needs no arithmetic at all: `NIU_MST_REQS_OUTSTANDING_ID(0)` is a counter of the
initiator's own in-flight read requests, and whether it plateaus decides on its
own whether an outstanding-request limit exists to be modelled.

`nocbench` is the odd one out in two ways. It is the only one whose experiment
is *planned* by a separate, tested Python module
(`tt_sim.perf.noc_congestion_plan`) rather than being wired into the C++, because
the thing that makes or breaks a congestion measurement is which confounds are
held fixed, and an invariant that lives in tested code is checkable in a way that
one living in a comment is not. And it is the only one whose honest verdict
against tt-sim is `INVALID`: the simulator models no link congestion at all, so
the experiment is forced flat, and the harness refuses to report a flat reading
as a result when the control that proves flows contend does not move either.

## One card session

Each program has its own `run_card.sh`, but you rarely want one — you want the
whole block, once, while you have the card. `run_card_session.sh` is that:

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal
./perfbench/run_card_session.sh --list        # what would run, on which part
./perfbench/run_card_session.sh               # everything that applies
./perfbench/run_card_session.sh nocread rv    # or just these
./perfbench/run_card_session.sh --resume      # after one dies three probes in
```

It detects the part by running `nocbench --dump-grid`, whose output the
congestion probe needs anyway — pass `--arch blackhole` to assert it instead and
skip that build. It then runs the probes that apply to the part and **skips the
rest loudly**: several are per-part on purpose, and `cmdbuf` is Blackhole-only
because Wormhole has no such register. Everything lands in one directory as
`<probe>.<arch>.csv` plus a per-probe `.out`, `session.log`, and the analysis
reports.

Every probe ends in one of seven statuses. `MEANINGFUL` means the probe's own
control moved. `COLLECTED` means the CSV is written but nothing in-session can
grade it — the knee hunts are like this, and saying so is more honest than
claiming a verdict the run did not earn. `DEFERRED` means the CSV is written but
the *analysis* needs `tt_sim/`, which a card box usually does not have; the
session writes no report at all rather than a file called `.report.txt` holding
a traceback. `DEGENERATE` means the control did not move, which on a card is a
broken run rather than a result. `SKIPPED`, `FAILED`, `SUSPECT` and `UNCLEAR`
are what they sound like. The exit status is non-zero if anything needs a look —
with `DEGENERATE` counted on a card and not under `--sim`, where it is the
correct answer.

**The verdicts are a testable library, not inline greps.** They live in
`card_session_verdicts.sh` as pure functions of the files a probe left behind,
and `card_session_verdicts_test.sh` runs them against the CSVs a real Blackhole
card returned on 2026-08-09 — a session that awarded `MEANINGFUL` to three
probes whose controls had not moved:

| probe | what it printed | what the data was |
|---|---|---|
| `nocread` | "no initiator limit — this RETIRES the credit-limit term" | `outstanding_max` = 72 in all 129 rows, including bursts of **four** requests |
| `cmdbuf` | "the number the ISA docs withhold" | `0x00000000` at rest **and** mid-burst, in all 129 rows |
| `noc-epoch` | "no reproducing per-core epoch, which retires the (11,2) observation" | the detector never ran; its report held one line of `ModuleNotFoundError` |

All three are the same bug — a check on a value's *presence*, its *format*, or
the *absence* of a failure string, standing in for a check that the thing being
measured responded. The rule is now stated once, in one place, and tested:
**`MEANINGFUL` is only ever awarded for a control that moved.** Run the test
before a session:

```bash
perfbench/card_session_verdicts_test.sh     # 28 checks, no card needed
```

**About 90 minutes cold**, of which ~60 is building the four programs; about 30
once the build trees exist. `--list` breaks it down per probe.

### What to copy to the card box

**`perfbench/` alone is enough for all ten probes**, because the one step that
needs a planner ships its plan pre-built (below). That is the whole of it:

```bash
rsync -av --exclude 'build/' perfbench/ <card-box>:~/perfbench/
```

The card box needs a built tt-metal and nothing else — no venv, no `tt_sim/`, no
`driver/`, no repo checkout. `TT_METAL_HOME` is the only way anything is told
where tt-metal lives; no path is baked in.

**`--exclude 'build/'` is load-bearing.** A `CMakeCache.txt` records the
absolute path it was generated in, so shipping a build tree from a machine that
ever built here makes cmake refuse on the card — *"the current CMakeCache.txt
directory ... is different than the directory ... where CMakeCache.txt was
created"*. That failed eight of ten probes on 2026-08-09, and only
`nocreadbench` survived because this box had never built it. The session now
recognises a foreign cache and discards it (a build tree is derived and always
safe to throw away), so the exclude is a second line of defence rather than the
only one — but it also saves rsyncing several MB of object files each time.

Two steps are *analysis*, not collection, and they do need `tt_sim/`:

- the **`nocbench` planner**, which decides the congestion experiment; and
- the three **report generators** (`*_bench_sweep`, `noc_congestion_sweep`).

If `tt_sim/` is not importable the session says so, runs the four benches
anyway, collects every CSV, and skips only the planner-dependent `noc` and
`noc-epoch` probes. Nothing is silently lost: `nocbench-grid.csv` is still
dumped and sent back, and it is what lets the plan be built at home. Analysis of
a CSV is not time-critical; being at the card is.

**You do not need `tt_sim/` on the card box for the congestion probes either** —
plan at home instead. `perfbench/nocbench/noc-plan-blackhole.csv` is a
**pre-built plan** for the harvested Blackhole card, generated from that card's
own 2026-08-05 `--dump-grid` capture:

```bash
python3 -m tt_sim.perf.noc_congestion_plan \
  --grid tt_sim/perf/datasets/nocbench-grid-blackhole.csv \
  --out perfbench/nocbench/noc-plan-blackhole.csv \
  --shared-sizes 64,512,2048,8192,16384
```

It rsyncs with `perfbench/`, so **all ten probes run from `perfbench/` alone**.
The session picks it up automatically for a matching `--arch`; `--plan FILE`
overrides.

A plan is only valid for the grid it was built from — one naming a tile the part
does not have measures nothing, and on a harvested card that is easy to do by
accident. So before using any pre-built plan the session dumps the card's live
grid and checks **every** addressed tile against it, refusing with the offending
coordinates rather than measuring. Regenerate the plan if the card, its
harvesting, or the experiment arguments change.

Copy `tt_sim/` only if you want the **analysis** on the card box too — the three
report generators (`*_bench_sweep`, `noc_congestion_sweep`). It cannot be
trimmed to `tt_sim/perf/`: `noc_congestion_plan` imports `tt_sim.network.tt_noc`,
which reaches `tt_sim.perf.model` and `costs.py`, and `tensix_bench_sweep`
imports `costs_test`; transitively that needs **numpy** and **pyyaml**. Analysis
is not time-critical; being at the card is.

Do **not** copy `perfbench/run.sh` expecting it to help — it is the
simulator-side runner and points `TT_METAL_SIMULATOR` at tt-sim. The card path
never uses it, and `run_card_session.sh` refuses outright if that variable is
set.

Results default to `~/tt_traces/card-session-<arch>/`, which is where this card
box keeps them; `--out` overrides. (There is no `/mnt/ramdisk` on that box —
an earlier revision defaulted to one and fell back to `$PWD`, which scattered
CSVs wherever the session happened to be started from.)

### The card is harvested, and what that changes

The Blackhole part in use has physical worker columns `{1..7, 12..16}` — columns
8–11 are absent — and tt-metal warns about `tray_id` on that motherboard, which
is expected and not a failure. A sweep that picks coordinates *arithmetically*
would land in a missing column and produce a garbage row indistinguishable from
a measurement. None of these programs does:

- **`tensixbench` / `riscvbench`** use the single logical core `{0, 0}`.
- **`nocreadbench`** chooses every core in **logical** space
  (`compute_with_storage_grid_size`, `worker_core_from_logical_core`), so a
  harvested column simply is not addressable. Its hop-distance, fan-out and
  stride sweeps are all safe.
- **`nocbench`** dumps the grid by running a probe kernel that reads
  `NOC_NODE_ID` on each core, and `noc_congestion_plan` **refuses** a dump that
  is short of an unharvested part's worker count and carries no `phys_x/phys_y`
  column. This is not theoretical: that refusal exists *because* of this exact
  card, whose addressed columns `{1..7, 10..14}` are a legal-looking subset of
  the physical `{1..7, 10..16}` while actually being `{1..7, 12..16}`. Four
  shared-link counts in the first plan were wrong before it was added.

One thing is **not** safe, and it is a derived label rather than a measurement:
`nocreadbench`'s **`hops` column** computes a torus distance with a modulus of
(logical grid + 2), which is not the physical NoC width and is further off when
columns are missing. The coordinates it is derived from — `mst_node_x/y`,
`src0_x/y` — are true physical values and are all in the CSV, so recompute
`hops` at home rather than reading E1's distance sweep straight off the column.
The session runner prints this reminder after every `nocread` run.

Every probe ends in a one-line verdict, and the rule is the one `nocbench` set
with `INVALID` and `nocreadbench` with `DEGENERATE`: **a probe whose control did
not move says so instead of reporting a flat reading as a result.** The
2026-08-09 Blackhole session broke that rule three times, which is why the
verdicts now live in `card_session_verdicts.sh` with a test that replays that
session's own files. The summary block at the end is the handover checklist —
read it before packing up.

`--sim` runs the same block against tt-sim at smoke sizes, for checking the
harness. It stamps every artefact `NOT-A-MEASUREMENT`, and it is the only way
past the guard that otherwise refuses to run with `TT_METAL_SIMULATOR` set.
Against the simulator most probes read `DEGENERATE` **and that is correct** —
tt-sim models no link congestion, its NIU queue is unbounded, and nothing
back-pressures the core that issues a Tensix instruction. The table below is the
observed result of `--sim --arch blackhole` at smoke sizes with the cost model
**off**, not a prediction:

| probe | against tt-sim | on a card |
| --- | --- | --- |
| `nocread` | `DEGENERATE` — the NIU queue is unbounded, so E0 reads the full burst by construction | the whole question |
| `cmdbuf` | `DEGENERATE` — no command buffer is modelled; reads the "absent" sentinel | a peak occupancy that MOVED off its rest value, or the run says nothing |
| `tensix`, `tensix-warm` | `DEGENERATE` — every probe reads exactly 1.000, because nothing back-pressures the issuing core with the model off | real occupancy |
| `rv`, `rv-pairs` | `DEGENERATE` — riscvbench's own live-instrument check fires: `mul_dep`, `div` and `store_spread` all read ~1.0 | meaningful |
| `rv-qdrain` | `COLLECTED` — it is the knee hunt; nothing in-session grades it | `COLLECTED` |
| `rv-gset` | `SKIPPED` — minutes per gset against the simulator | `COLLECTED` |
| `noc`, `noc-epoch` | `SKIPPED` — see below | the experiment, or `DEFERRED` if the box has no `tt_sim/` |

**The congestion probes do not merely read `INVALID` against tt-sim; they hang**,
so `--sim` skips them unless you name them explicitly. `nocbench --dump-grid`
probes the first logical row and column of the whole compute grid, but the
simulator instantiates only the tiles listed in `TT_SIM_TENSIX_COORDS` — one, by
default — so the launch waits forever on cores that do not exist. Nothing is
lost by skipping: tt-sim models no link congestion, so the honest verdict there
is `INVALID` by construction. To force them, set a multi-tile
`TT_SIM_TENSIX_COORDS` and name the probe.

So **eight of the ten probes read `DEGENERATE` or `SKIPPED` against the
simulator and every one of those is correct.** The simulator is not the
instrument; it is how you check the instrument runs.

### Blackhole now, Wormhole as a follow-on

The lab has a Blackhole part and no Wormhole part; a Wormhole session is
**planned, not abandoned**. Every probe here already runs on both — `--arch
wormhole` executes the same block — so the follow-on is a hardware booking
rather than new work. That is checked, not assumed: `--sim --arch wormhole` has
been run end to end and all ten probes execute or skip with the right reason,
`cmdbuf` correctly reporting `needs a blackhole part; this is wormhole`. So the
follow-on should be a pure hardware run rather than a debugging session.

Two roadmap questions stay open until it happens, and the runner prints them
after every run so a clean session cannot be mistaken for a complete one:

- **The store-coalescing pair and the multiply pair on Wormhole.** Predicted
  identical there, measured 5.2× apart on Blackhole. The `rv-pairs` probe
  already produces them; it needs the part.
- **The per-arch half of the read floor.** `nocreadbench`'s claim *is* a
  per-architecture difference — Wormhole 25.00 cycles/tx and 17.33 with a
  shorter issue loop, Blackhole 35.0 and 34.0. A Blackhole-only session can
  size or retire **Blackhole's** initiator limit outright, which is a real
  result on its own; it **cannot settle the per-arch question**. Do not read a
  clean Blackhole `nocread` as closing that.

What Blackhole *uniquely* answers is the more valuable half anyway:

- **`CMD_BUF_AVAIL` at rest** has no Wormhole analogue at all. It is the single
  highest-value reading in the session — the number that moves the read-floor
  term from a permanent `unknown` to a chargeable `isa_doc` entry — and the one
  most easily got wrong, because `0xFFFFFFFF` means *the read failed*, not that
  the value is large. The runner banners it and decodes the four 5-bit fields.
- **The `NIU_MST_REQS_OUTSTANDING_ID(0)` plateau-vs-climb reading**, which
  either sizes the limit or retires the term, needs one part and Blackhole is
  the part where the floor reproduces exactly.

### Designed, not built

Six roadmap probes are deliberately absent, because each needs new device code
and a shipped block covering ten beats a half-built one covering sixteen. What
each would take, established by reading rather than guessed:

- **`ATCAS` / `ATINCGETPTR` against a real L1 semaphore.** No probe exists — the
  strings appear nowhere in `perfbench/`. `docs/plans/tensix-cost-benchmark.md`
  calls the probe "straightforward" and names it as one of two next steps; it
  needs two new `RUN(...)` slots in
  `tensixbench/src/kernels/compute/raw_probes.cpp` past the current 20, which
  crosses `TTBENCH_NUM_PROBES` and so the result-buffer size and
  `TTBENCH_MAGIC`. The wrinkle the plan does not spell out is that, unlike every
  existing phase-A probe, these are **not side-effect-free**: the block is
  unrolled 64× and replayed at four burst lengths, so the semaphore's value
  moves under the measurement. `ATINCGETPTR` tolerates that (it increments
  regardless); `ATCAS` does not, since a compare-and-swap that stops swapping
  stops costing the same. Pick the compare/set values so every iteration
  succeeds, or the slope measures two different instructions.
- **`RDCFG` latency, as `(op + STALLWAIT) − (1-cycle op + STALLWAIT)`.** Slot 14
  measures `RDCFG` *throughput* only. `TTI_STALLWAIT` already appears in the
  tree (`raw_probes.cpp`, in the `UNPACR_NOP` setup) so the instruction is
  reachable; the work is two more paired slots and the same magic bump. This is
  the cheapest of the six and the obvious next one to build.
- **The `TTBENCH_UNROLL` sweep over {16, 32, 64, 128}.** Three separate
  blockers, none of them a flag: `TTBENCH_UNROLL` is a `#define` in
  `bench_layout.h`; `tensixbench`'s `CMakeLists.txt` has no
  `target_compile_definitions`; and `CreateKernel` is called with a bare
  `ComputeConfig{}`, so **no JIT define reaches the device kernel at all**
  (contrast `riscvbench`, which does pass `RVBENCH_G_SET` that way). The unroll
  factor is also a literal `REP64` token in the `PROBE` macro rather than
  derived from the `#define`. Today the sweep means editing the header and
  rebuilding four times.
- **`riscvbench` phase-G at 4608 / 5632 B.** Those are 1152 and 1408
  instructions; the built points are 1024/1280/1536/1792 (4096/5120/6144/7168 B)
  and there is no `REP1152`/`REP1408`. Needs both macros, two probe slots,
  `RVBENCH_G_SETS` 3 → 5, and a magic bump. The kernel-size ceiling is *not* the
  obstacle: 1024+1408 is smaller than the 1024+1792 set that already builds.
- **A divide magnitude sweep.** `rv_probes.cpp` hardcodes
  `dividend = 0x12345678, k = 3`, and `riscvbench.cpp` hardcodes the same
  literals into the CSV header as text. Nothing can vary them; the file says so
  in a comment. Needs one slot per dividend width — and the widths worth picking
  are not evenly spaced: `docs/plans/riscv-front-end-benchmark.md` records that
  real kernels execute 0–2 divides in 40,000–80,000 instructions with **9-to-12
  bit** dividends, against this benchmark's 29 significant bits. The interesting
  half of the 6–33 curve is therefore the *narrow* end, which is exactly the end
  no measurement has touched.
- **A longer `.ttinsn` burst sweep — and mind which axis.** The roadmap bullet
  reads as though nothing has been done, but the "extend to n = 512 or 1024" ask
  was closed on 2026-08-05: phase Q's **loop** form already sweeps
  n = 16…1024 (`RVBENCH_Q_LOOP_MIN_N 16`, `RVBENCH_Q_LOOP_POINTS 7`). What is
  left is going *beyond* 1024, and the plan doc is emphatic that this must not
  be done by lengthening the **cascade**: a 1024-word straight-line burst is
  4 KiB of instruction text, the exact octave phase F found a fetch cliff in,
  and a phase-Q burst runs once, cold, with its own fetch inside the timed
  region — so a growing fetch cost would be folded into the one measurement
  whose question is whether cost per instruction grows with n. The **loop** form
  is safe, because its body stays 16 instructions (64 B) however long the burst.
  So the cheap version is one line — `RVBENCH_Q_LOOP_POINTS` 7 → 8, giving
  n = 2048, which still fits under the existing `RVBENCH_MAX_POINTS 8` cap and
  needs no layout or magic change. Anything past 2048 does need the cap raised,
  which widens `RVBENCH_RESULT_WORDS` and is a magic bump. Not taken here only
  because it changes phase Q's default shape for every existing run.
- **The `DIR_BIDIR` hang — do not build this one.** Its premise has already been
  settled against it. `noc_congestion_plan.check_invariants` **refuses**
  bidirectional flows outright, under two tests, because the first and only such
  point never returned on a Blackhole card while all 79 unidirectional flows in
  the same session completed — and tt-metal's own `core_bidirectional` suite
  skips its entire directed-ideal family with `// Timeout issue (#36428)`. tt-sim
  runs the identical plan to completion, so nothing can be learned here without
  the card, and reaching it means hand-writing a plan CSV that defeats a tested
  safety invariant and risks hanging the card mid-session. The virtual-channel
  question it was there to answer already survives in better, unidirectional
  form.

The **(11,2) clock epoch** needed no new probe: `noc_congestion_sweep`'s
`clock_skew_report` already detects a per-core wall-clock offset, and only
accepts one that reproduces across **two or more** independent runs. That is the
entire reason `noc-epoch` exists as a second, identical congestion run.

Against the simulator:

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh tensixbench -- --blocks 2 --iters 4
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh riscvbench  -- --blocks 2
```

`TT_SIM_ARCH=wormhole` picks the other simulator, `TT_SIM_COST_MODEL=1` turns
the cycle cost model on. Keep the burst sizes small — the simulator runs a few
tens of thousands of cycles per second, where hardware runs a billion. Run
phase B one fidelity at a time (`--phase b --iters 1 --fidelities HiFi2`); two
in the same process do not finish against the simulator.

Every number these programs report is a **slope** over several instruction
counts, so kernel launch, timer overhead and loop setup cancel exactly. No
single absolute measurement is ever reported as a cost, and nothing here writes
to the cost tables.

Both refuse to let a null pass as a confirmation, but they do it differently, and
`riscvbench`'s way is the one worth copying: a run in which every probe reads
exactly `1.000` is simultaneously the expected answer and the signature of a
benchmark that measured nothing, so `riscvbench` singles out four probes with a
documented cost above one cycle, checks them, and refuses the run in those words
if they read 1.000 too.
