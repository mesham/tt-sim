# energybench — ranking-level energy estimation

The apparatus and the protocol for asking **which workload costs more energy,
and roughly by how much**. It fits no coefficients today, because no data
exists yet; what is here is the harness, the card protocol, the simulator-side
activity vector, and the analysis that will consume the data when it comes back.

## What is achievable, stated up front

`tt-smi` reports **board-level** power at roughly **1 Hz**. Kernels in this
project run from 0.7 µs to 6.7 ms of device time. **A single launch is
invisible** against the idle floor — there is no arrangement of this instrument
that weighs one launch.

**`tt-smi` must be ≥ 4.0.0.** This is not a nicety and the harness refuses to
start below it. v4.0.0 changed the default backend from Luwen to tt-umd; the
Luwen path cannot open a chip that a tt-metal program is holding — BAR0's
mapping is exclusive and `pyluwen::PciChip::new` panics with `Failed to map
bar0_uc for 0 with error Invalid argument (os error 22)`. On 6.2.0 a busy read
returns normally (measured at 69.0 W with `--arm rv` holding the device) and
does not perturb the workload (404.3 launches/s while sampled, against 407.5 and
418.8 unsampled). See [what the card taught this
harness](#what-the-card-taught-this-harness-2026-08-13).

Board power also includes DRAM, PHYs, ARC, PCIe and the fans, so a delta between
two arms is attributable to Tensix activity only by *argument*, never by
construction.

So **absolute per-launch joules are out of reach**, and the target is
**ranking**: the ordering of workloads by cost, and the ratios between them,
validated on predicted-versus-measured *order* rather than on an R² against
absolute joules.

**Every coefficient here will be FITTED.** Tenstorrent publishes no per-event
energy figure — no pJ/op, no pJ/bit, nothing. That is why the coefficients live
in this directory and not in `tt_sim/perf/unit_costs.yaml`; see
[the quarantine](#the-coefficients-are-quarantined-on-purpose).

## The quantity being measured

The harness opens the device **once**, builds the program **once**, and then
calls `detail::LaunchProgram` back to back for tens of seconds while telemetry is
sampled **in slot, while the kernel is running**. What that produces is

> **steady-state repeated-kernel board power under sustained load**, in watts,
> against an idle baseline **measured in the same session** and against each
> slot's **own idle reading taken seconds before it**

and **not** the energy of one launch, and **not** a post-exit decaying edge.
That phrase appears verbatim in the runner's banner, the session log's handover
block, `aggregate_power.py`'s output, `tt_sim.perf.energy_rank`'s report and JSON
(as `QUANTITY`), and in every fitted-coefficient file, because it is the one
thing a later reader must not get wrong. Per-launch energy is only recovered by
dividing by the **measured launch rate**, which is why `energybench` reports
`launches` and `wall_s` on every run and why the analysis refuses a row without
them. The simulator has to predict that same quantity:

```
E_launch(w) = c_launch + Σ_j c_j · a_j(w)              [joules per launch]
P_board(w)  = P_baseline + rate(w) · E_launch(w)       [watts]
```

`a(w)` is the activity vector `tt_sim.perf.energy_activity` emits; `rate(w)` and
`P_baseline` are measured; `c` is fitted.

## The arms, and why these

A ranking needs something to rank, so the arms have to span genuinely different
activity mixes rather than differing in size:

| arm | what it does | what it moves in the activity vector |
| --- | --- | --- |
| `idle` | a kernel that returns immediately | nothing — the launch machinery alone |
| `rv` | a dependent integer chain (one MUL + three ALU ops per iteration) on BRISC | `instr_retired`, and essentially nothing else |
| `noc` | barriered DRAM→L1 reads on BRISC | `noc_bytes_total`, `noc_flight_cycles`, flat compute |
| `mm` | `matmul_tiles` on two **resident** tiles | `matrix_busy_cycles`, flat NoC |
| `sfpu` | Int32 tile add on the same two resident tiles | `sfpu_busy_cycles`, flat NoC |

Two design decisions in that table are load-bearing.

**The compute arms hold their tiles resident.** The reader pushes exactly two
tiles once; the inner loop never pops or refills. So a compute arm's NoC traffic
is *fixed* however large its inner loop grows, which is what makes the compute
column separable from the NoC column instead of the two moving together.

**`mm` and `sfpu` share a reader, a writer and a skeleton** and differ only in
which backend unit their inner loop occupies. Without that pair a single
"compute" coefficient would absorb both units and neither number would mean
anything.

**Every arm runs at two inner counts.** The fit spends one coefficient per
activity term plus one for the launch machinery, and leave-one-out needs a
degree of freedom spared, so it can identify `workloads − 2` terms. Five arms
would buy three terms. Two scales buys **nine workloads and seven terms** — and
the within-arm ratio (the same arm, 4× the work) is the sharpest ranking test in
the set, because its two points differ in exactly one thing.

## The card protocol

Copy-pasteable, for someone who is not the person who wrote it. `tt-smi` **≥
4.0.0** must be on `PATH`; nothing else from this repo is needed beyond
`perfbench/energybench/` and a built tt-metal.

```bash
# 0. Get the tree onto the card box (build trees excluded -- a CMakeCache.txt
#    records the absolute path it was generated in and cmake refuses a foreign
#    one).
rsync -av --exclude 'build/' perfbench/energybench/ <card-box>:~/energybench/

# 1. On the card box.
export TT_METAL_HOME=/path/to/your/built/tt-metal
cd ~/energybench

# 2. Check the tools BEFORE spending card time. Both of these are hard
#    prerequisites and run_card.sh exits 4 / 5 without them.
python3 telemetry_sample.py version      # needs tt-smi >= 4.0.0 (tt-umd backend)
python3 sysfs_sample.py --check          # needs tt_aiclk & friends in sysfs

# 3. Reset the board, then let it settle. A board that has just been reset or
#    has just finished someone else's job is on a fan and thermal ramp, and a
#    ramp under an interleaved schedule looks like a result.
tt-smi -r 0
sleep 120

# 4. Look at the schedule and the wall estimate before committing to it.
./run_card.sh --list

# 5. Run it. ~23 minutes at the defaults, plus the first build (~2 min).
#    Slots are 40 s, not 30, so that ~21 samples survive the settle trim and
#    clear the analysis's `samples` gate of 20 -- see --list, which prints the
#    expected count and warns if it is short.
./run_card.sh --out ~/tt_traces/energybench-session

# 6. Sanity-check on the spot. A non-zero exit means at least one slot did not
#    produce a clean measurement, and the handover block names which.
echo "exit=$?"
grep -E 'NO TELEMETRY|RUN FAILED|DID NOT PRODUCE' ~/tt_traces/energybench-session/session.log
column -s, -t ~/tt_traces/energybench-session/power.csv | less -S
#    Look for: every `status` is ok; `samples` is not 0 anywhere; `samples` and
#    `attempts` agree; `sysfs_aiclk_mean` is the same in every row; every
#    `therm_trip_delta` is 0; the control agrees with its twin; the arms
#    separate. If they do not, say so in the handover -- do not re-run and keep
#    the better session.

# 7. Send the WHOLE directory home. Analysis needs tt_sim/ and numpy and is not
#    time-critical; being at the card is.
rsync -av ~/tt_traces/energybench-session/ <home>:~/energybench-session/
```

`--bracket` additionally takes post-exit samples per slot and fits the board's
thermal decay into `decay.txt`. It is a **fallback and a diagnostic**, measuring
a decaying edge rather than sustained load, and nothing fits it; it costs about
another 8 minutes at the defaults.

### What the schedule is, and why it is interleaved

One **cycle** runs the baseline and then every workload once. The session runs
several cycles. Arms are **not** blocked (all of A, then all of B): this project
has been bitten by drift before, and a blocked schedule turns a slow thermal
ramp into a fake result. The established discipline here is interleaved A/B with
a verified-zero control, and that is what this is.

At the defaults, per cycle:

| slot | what | duration |
| --- | --- | --- |
| 0 | **baseline** — device open, nothing launching | 40 s |
| 1 | `idle-0` | 40 s |
| 2–5 | `rv-200000`, `noc-4096`, `mm-4096`, `sfpu-4096` | 40 s each |
| 6–9 | `rv-800000`, `noc-16384`, `mm-16384`, `sfpu-16384` | 40 s each |
| 10 | `noc-4096__control` | 40 s |

3 cycles × 462 s ≈ **23 minutes**, plus build and device open/close. `--cycles`
and `--seconds` change it; `--list` recomputes the estimate.

**Every slot is preceded by its own idle reading** (`--pre-samples`, default 2),
taken with the device free in the gap before the arm starts. That is the fix for
an unstable baseline: see [the baseline is
paired](#the-baseline-is-paired-not-assumed-stationary).

**The control is the point of the last slot.** `noc-4096__control` is the *same
workload* as `noc-4096`, run in a different slot of the same cycle, so its true
delta is **zero by construction**. If the two disagree by more than the noise
floor, the board drifted within the cycle and every other difference in the
session is unsafe — the analysis **refuses the whole session**, it does not warn.

**The first and last 5 s of every slot are discarded** (`--settle`). A ~1 Hz
sampler straddles slot edges, and the power rail and fan curve both have time
constants of seconds; trimming both ends is what makes the remainder a
measurement of a steady state rather than of a transition. The surviving sample
count is reported per row and the `samples` gate refuses a row with too few.

### What lands in the output directory

```
session.log     everything the run printed, including the tt-smi version and
                the handover block
slots.csv       one row per slot -- ALWAYS, including slots that failed --
                with its window, its launch count and its status
raw/*.pre.csv   the slot's own idle reference, taken just before it
raw/*.pow.csv   in-slot power samples: EVERY ATTEMPT, successes and failures,
                a failure carrying the tool's own error text
raw/*.clk.csv   sysfs clock and thermal samples, taken throughout the slot
raw/*.post.csv  --bracket only: post-exit samples on a decaying edge
launches.csv    energybench's own per-run summary
power.csv       the aggregated input to tt_sim.perf.energy_rank
decay.txt/json  --bracket only: the fitted thermal time constant and verdict
```

Send **all** of it. `raw/` is what makes a suspicious `power.csv` row
diagnosable later — and, since the rebuild, `raw/*.pow.csv` is what tells you
*why* a row is empty rather than leaving you to guess.

### Testing the harness without a card

```bash
perfbench/energybench/run_card_stub_test.sh    # no card, no tt-metal needed
```

`ENERGYBENCH_TELEMETRY_CMD`, `ENERGYBENCH_BIN` and `ENERGYBENCH_SYSFS_ROOT` stub
the three things that need hardware. The scheduling, the sampler lifecycle, the
per-slot windows, the launch-rate capture and the settle-trimmed aggregation are
exercised end to end, and the stub board carries a deliberate slow drift, which
is what the interleave is supposed to survive.

**The stub now holds an exclusive resource.** The old one answered every
telemetry request instantly and always succeeded, which is why it validated a
harness that then lost a whole card session: a stub that always answers cannot
tell you what the real tool does when the device is busy. The stub benchmark now
takes an exclusive `flock` for the whole of its run, and the stub telemetry tool
has two eras — `STUB_SMI_ERA=luwen` takes the lock non-blocking and reproduces
the `Failed to map bar0_uc` panic verbatim, `STUB_SMI_ERA=umd` reads through it.
The test asserts that the Luwen era **fails**, the tt-umd era **succeeds against
the same held lock**, and that a sampler which fails for any reason is loud:
non-zero exit, `NO TELEMETRY` on stderr, one row per failed attempt carrying the
panic text, an empty power cell rather than `0.0`, and a refusal from both the
aggregator and the analysis.

Anything collected that way is stamped **NOT-A-MEASUREMENT**.

## The simulator side

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/energybench/run_sim_activity.sh \
    --arch blackhole --out activity-sim-blackhole.csv
```

This runs each arm against tt-sim with `TT_SIM_TRACE_COUNTERS` and
`TT_SIM_COST_MODEL=1`, and reduces each run's counter dataset to a per-launch
activity vector (`tt_sim.perf.energy_activity`). The terms are a **fixed, ordered,
append-only** schema so that a term added later shifts no existing column.

**Keep the cost model on.** Every `*_busy_cycles` and `*_stall_cycles` counter is
*absent, not zero*, without it — the aggregator emits a counter only when
something incremented it — so a cost-model-off activity matrix has whole columns
of zeros that no fit can use. `--no-cost-model` exists to demonstrate that.

**The inner counts must match between the two sides.** A label is `<arm>-<inner>`
and the analysis joins on it; an activity vector collected at a different inner
count is silently a different workload, so it is dropped with a note rather than
matched. The simulator runs a few tens of thousands of cycles per second where
hardware runs a billion, so running the *card's* inner counts against tt-sim is
an overnight job, not a coffee break — plan for that rather than quietly
shrinking one side. `--inner arm=N` on both scripts is how a shared set is
pinned.

The checked-in `activity-sim-blackhole.csv` was collected at **smoke** inner
counts, which are the right size for showing the apparatus works and the wrong
size for fitting against a card.

### What it produced, 2026-08-13, Blackhole, cost model on

Per launch, one launch per run, `--scales "1 4"`:

```
    label    cycles     instr      disp      nocB       txn       MAT      SFPU      PACK       THC   rvstall   txstall    flight
   idle-0      7499     14610        27         0         0         1         3         0         0      9602         0         0
    rv-64      8099     16144        27         0         0         1         3         0         0     11074         0         0
   rv-256      9399     19624        27         0         0         1         3         0         0     14100         0         0
    noc-8     11499     22451        27     16384        16         1         3         0         0     21767         0      3600
   noc-32     23199     45356        27     65536        64         1         3         0         0     57368         0     14400
     mm-8     10499     23207      1262      6144         6       533         3        16        52     16017      3198      1338
    mm-32     12099     28620      3590      6144         6      2117         3        16       124     18591      6078      1338
   sfpu-8     11699     26328      4719     12288         6       342      1028        16        29     18888      4313      1456
  sfpu-32     16199     40865     17031     12288         6      1326      4100        16        29     26854      8801      1456
```

Read it as a design matrix and the separation is the thing to check, because it
is what decides whether any coefficient is identifiable:

* **`rv` moves `instr_retired` alone** — 16,144 → 19,624 with every other column
  pinned. Nothing else in the set does that.
* **`noc` moves bytes and flight alone** — 16,384 → 65,536 B and 3,600 → 14,400
  cycles, with `instr_retired` rising only because the issue loop is longer, and
  no Tensix dispatch at all beyond the 27 the firmware itself issues.
* **`mm` moves `MAT` 533 → 2,117 with `nocB` pinned at 6,144.** That flat NoC
  column is the resident-tile design working: 4× the matmul work, byte-identical
  traffic.
* **`sfpu` moves `SFPU` 1,028 → 4,100 with `nocB` pinned at 12,288**, likewise.
* **`idle` is not zero**, and should not be: 14,610 instructions and 9,602 stall
  cycles are the launch machinery and the firmware it runs. Every other arm pays
  that too, which is what the launch column is for.

One correlate is worth naming because it will show up in any fit: `mm` and
`sfpu` both raise `MAT` (the SFPU arm's `copy_tile` runs on the matrix path), so
`matrix_busy_cycles` is not a pure `mm` column. The two arms still separate —
`mm` at inner 32 has `MAT` 2,117 and `SFPU` 3, `sfpu` at inner 32 has `MAT`
1,326 and `SFPU` 4,100 — but a fit that reports a `matrix_busy_cycles`
coefficient is reporting one that both arms contributed to.

## The analysis

```bash
python3 -m tt_sim.perf.energy_rank \
    --activity perfbench/energybench/activity-sim-blackhole.csv \
    --measured ~/energybench-session/power.csv \
    --report report.txt --json report.json
```

### The gates

Eleven gates run before any ranking is reported, and a failure is a **refusal**,
not a warning. A fit is only meaningful if the measurement it is fitted to
actually said something.

| gate | passes when | refuses when |
| --- | --- | --- |
| `telemetry` | every row has a status of `ok`, ≥ 1 successful sample and a finite power | any row was never measured — **this is the 2026-08-13 gate** |
| `baseline` | the session measured its own idle floor | no `baseline` rows |
| `schedule` | every interleave cycle carries every label | a cycle has a hole in it |
| `thermal` | `tt_therm_trip_count` did not move | the part throttled, or there is no thermal record |
| `clock` | no slot's AI clock drifted past `--max-clock-drift-pct`, and the slots agree on it | the clock moved within a slot, slots ran at different clocks, or there is no clock record |
| `repeats` | every label has ≥ `--min-repeats` interleave cycles | any label has fewer |
| `samples` | every row kept ≥ `--min-samples` telemetry samples | any row was under-sampled |
| `control` | the control agrees with its twin inside `--sigma` × noise | it does not, or there is no control at all |
| `spread` | the workloads span more than `--sigma` × noise | they sit inside the noise floor |
| `identifiability` | coefficients ≤ workloads − 1 and condition number ≤ `--max-cond` | either bound is broken |
| `rankable` | ≥ 3 workloads have both a vector and a rate | fewer |

`--max-clock-drift-pct` defaults to **5 %**, which is 67 MHz at 1350 MHz: tight
enough to catch the 1350 → 800 MHz transition that broke the 2026-08-13
baselines, loose enough for ordinary DVFS ripple. It is a first guess, and the
first real session should set it from the observed `sysfs_aiclk_drift_pct`
column rather than leaving it at a number nobody has calibrated.

**A guard that cannot fail is as damaging as one that cannot pass**, and that is
not a slogan here. The term selector deliberately does **not** filter on the
identifiability gate's own threshold: it excludes only columns with no spread and
columns that are exact linear duplicates — both arithmetic, not judgement — so a
merely ill-conditioned set reaches the gate and is refused there. A selector that
avoided what the gate checks would make the gate incapable of failing.
`tt_sim/perf/energy_rank_test.py` builds a passing session and a failing one for
every gate in the table, and asserts both.

The noise floor is derived from **this session's own repeats** (the RMS of each
label's across-cycle standard deviation), never carried over from another
session — a floor from another session is exactly the drift the control exists
to catch.

## What the card taught this harness (2026-08-13)

A three-cycle session ran on a Blackhole p150 and **every arm slot recorded
`samples=0`, `power_w=0`**. Only the baselines — which launch nothing — got
telemetry. The launch counts were fine: 673/s idle down to 133/s for
`noc-16384`, reproducible across cycles to ~1 %. The session ran to completion,
wrote a full CSV, and was worthless.

Four things came out of it, and all four are now enforced.

**1. The tool version, not the architecture.** The box was on `tt-smi` 3.0.32,
whose Luwen backend cannot open a chip held by a tt-metal program. It has since
been upgraded to 6.2.0 — v4.0.0 moved the default backend to tt-umd — and a busy
read now works and does not perturb the workload. So in-slot sampling is the
design, and `run_card.sh` **refuses to start** below 4.0.0 rather than banking
zeros. The version it used is printed in the banner, written to `session.log`,
and carried in every CSV row as `tt_smi_version`.

**2. The silence, which was the real bug.** The old sampler piped `tt-smi` into
a parser wrapped in `except Exception: pass` and appended nothing when it threw.
A refusal was therefore indistinguishable from a quiet board, and `mean([]) →
0.0` finished the job. Now **every attempt is a row**, a failed one carries the
tool's error text, `attempts` sits beside `samples` so partial failure is visible
too, a dead slot is announced the moment it happens, an unmeasured cell is
written **empty rather than `0.0`**, the aggregator exits non-zero, and the
`telemetry` gate refuses. Five layers, because one was zero.

**3. The clock.** Cycle 0's baseline read 61.7 W at 1350 MHz; cycles 1 and 2
read ~39 W at 800 MHz, and a standalone idle read afterwards gave 70 W. That is
a 42 % swing in the quantity everything is differenced against, driven by a clock
nothing was recording. `sysfs_sample.py` now samples `tt_aiclk`, `tt_arcclk` and
`tt_therm_trip_count` throughout every slot — free, no device handle, unperturbed
by the workload — and the `clock` and `thermal` gates refuse a session that
moved.

**4. Cycle 2 was missing `idle-0` entirely** — the CSV jumped slot 0 to slot 2.
The cause is structural rather than mysterious: `run_card.sh` wrote a
`slots.csv` row **only for a slot that succeeded**, and `aggregate_power.py`
emits one row per manifest row, so a slot that crashed or printed no
`ENERGYBENCH` line left **no trace in either machine-readable output** — one line
in `session.log` and nothing else. A hole was indistinguishable from a schedule
that never had that slot. Now the manifest row is written **unconditionally**
with a `status` of `run_failed`, the runner says so at the time and again in the
handover block and exits non-zero, and the `schedule` gate independently checks
that every cycle carries every label. Both nets are asserted in
`run_card_stub_test.sh`.

## The baseline is paired, not assumed stationary

A reference that swings 42 % between cycles is not a reference. Three changes,
and the reasoning for each:

**Every slot gets its own idle reading, seconds before it** (`pre_idle_w`, and
`delta_w = power_w - pre_idle_w`). The device is free in the gap between slots,
so this costs two telemetry calls and no schedule change. It is the change that
actually helps: a drift that takes a cycle to develop cannot get between a slot
and a reading taken seconds earlier, whereas it sits squarely between a slot and
a per-cycle baseline.

**The baseline slot is measured identically** — same samplers, same settle trim,
same pre-idle probe, same duration — rather than being a bare `sleep` with a
different sampling path. If the baseline and the arms are not the same quantity
measured the same way, their difference is partly an instrument artefact.

**The clock is recorded on every row and gated**, so baselines taken at
different clocks are *refused* rather than averaged. This is the honest part: the
paired reference reduces the exposure, but nothing in this apparatus can rescue a
board that changed DVFS state mid-session, and pretending otherwise would be
worse than refusing.

`delta_w` is reported per row and is the right column to eyeball at the card.
The fit still uses the session baseline, unchanged — `energy_rank`'s arithmetic
was not touched, and the paired reading is a diagnostic and a cross-check rather
than a second, silently different, definition of the same number.

### The ranking metric

The headline number is **leave-one-out Spearman**: each workload is predicted by
a model refitted *without* it. In-sample Spearman is printed alongside, and the
gap between them is the honest measure of how much of the agreement is fitting
rather than predicting — with nine workloads and up to seven coefficients an
in-sample fit will look excellent whatever the truth is.

Ratio errors are reported as `|log(predicted ratio / measured ratio)|` over every
workload pair, median and max, because a ranking claim is really a claim about
ratios.

Coefficients are constrained non-negative (Lawson–Hanson NNLS). A negative energy
per instruction is not a finding, it is a fit artefact, and allowing one lets the
model buy accuracy with nonsense.

**One collinearity is known and will not go away**: the launch constant and the
`idle` arm's fixed instruction count (~14,600 instructions of firmware per
launch, whatever the kernel does) move together, because every arm pays both. On
synthetic data with a known truth the four activity coefficients come back to
within 0.2% while `c_launch` comes back at 57% of its true value, with the
difference absorbed into `instr_retired` — and the worst pairwise ratio error in
the whole set is the one involving `idle-0`. That is a real limit of this design,
not a bug: separating them needs an arm that launches without executing firmware,
which does not exist. Read `c_launch` as "launch machinery *plus* the firmware
instructions it always runs", and treat `idle-0`'s predicted energy as the least
trustworthy point in any fit.

## The coefficients are quarantined, on purpose

`tt_sim/perf/unit_costs.yaml` and `tt_sim/pe/tensix/tensix_instruction_costs.yaml`
run a provenance ladder — `isa_doc > isa_doc_derived > vendor_source >
vendor_source_derived > estimated > unknown` — whose whole purpose is to keep
un-sourced numbers out of the cycle model. `costs_test.py` records that there are
currently **zero** `estimated` entries in either file.

A fitted energy coefficient is **weaker than `estimated`**. It is a regression
coefficient from a ~1 Hz board-level power meter against a nine-point design, and
no document will ever exist that it could be traced to. `estimated` provenance is
forbidden in those tables and **this is exactly what that rule exists to keep
out**. Silicon is corroboration there, never provenance; energy coefficients are
not even that, because they live outside the system entirely.

So they live **here**, in `perfbench/energybench/fitted_energy_coefficients.yaml`,
written by `energy_rank --write-coefficients`. Three independent things stop the
obvious failure mode, which is not malice but tidying — a later reader moving the
file "where the other cost tables are":

1. **The file is stamped `provenance: fitted`**, a token that is *not in*
   `tt_sim.perf.costs.PROVENANCE_RANK`. The cost loader raises `KeyError` on any
   table carrying it, so pasting one of these entries into `unit_costs.yaml`
   breaks the loader rather than silently ranking the number.
2. **`energy_rank.check_destination` refuses to write anywhere under `tt_sim/`**,
   and refuses the two cost-table filenames by name. A file cannot become a cost
   table by accident if it cannot be written next to one.
3. **`tt_sim/perf/energy_quarantine_test.py`** asserts all of the above, plus
   that neither cost table contains any energy vocabulary (`energy`, `joule`,
   `watt`, `picojoule`, …) today — because a coefficient does not have to arrive
   labelled `fitted` to do damage.

Each of those is asserted in **both** directions: the refusals refuse, the
allowed paths are allowed, and the cost tables still load and still rank every
entry, so a clean run means "no energy got in" rather than "the tables were
unreadable".

`fitted_energy_coefficients.example.yaml` in this directory shows the shape with
no values in it.

## What this will and will not be able to claim

**Will**, once a card session exists and the gates pass:

* that tt-sim orders these workloads by energy the way the board does, with a
  stated leave-one-out rank correlation;
* how far off the *ratios* are, per pair, with a median and a worst case;
* a per-term fitted coefficient set with a written record of what it was fitted
  to, usable for comparing two candidate schedules of the same shape;
* an evidenced negative — that the ordering is *not* reproduced — which is a
  real result and the one this design is most likely to deliver first.

**Will not**, ever, on this instrument:

* absolute joules for a kernel launch, or anything convertible to one;
* attribution of a board-power delta to the Tensix array rather than to DRAM,
  PHYs, ARC, PCIe or the fans;
* per-instruction or per-bit energy that could enter the cost tables at any
  provenance;
* anything about a workload whose activity mix is outside the span of these five
  arms — the fit is an interpolation over the design, and nine points do not
  extrapolate;
* a claim from a session whose control moved, whose arms sat inside the noise
  floor, or whose repeats were too few. Those are refusals, and the refusal is
  the output.

One more, worth stating because it is the most tempting mistake: the coefficients
are fitted to **steady-state repeated-kernel board power under sustained load**,
sampled in slot. Using them to predict
a single cold launch is using them outside the quantity they were fitted to, and
the launch machinery term (`c_launch`) is the only part of the model that has
anything to say about it.
