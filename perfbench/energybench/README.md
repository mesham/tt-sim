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
sampled. What that produces is

> **steady-state repeated-kernel board power**, in watts, against an idle
> baseline **measured in the same session**

and **not** the energy of one launch. Per-launch energy is only recovered by
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

Copy-pasteable, for someone who is not the person who wrote it. `tt-smi` must be
on `PATH`; nothing else from this repo is needed beyond `perfbench/energybench/`
and a built tt-metal.

```bash
# 0. Get the tree onto the card box (build trees excluded -- a CMakeCache.txt
#    records the absolute path it was generated in and cmake refuses a foreign
#    one).
rsync -av --exclude 'build/' perfbench/energybench/ <card-box>:~/energybench/

# 1. On the card box.
export TT_METAL_HOME=/path/to/your/built/tt-metal
cd ~/energybench

# 2. Reset the board, then let it settle. A board that has just been reset or
#    has just finished someone else's job is on a fan and thermal ramp, and a
#    ramp under an interleaved schedule looks like a result.
tt-smi -r 0
sleep 120

# 3. Look at the schedule and the wall estimate before committing to it.
./run_card.sh --list

# 4. Run it. ~16 minutes at the defaults, plus the first build (~2 min).
./run_card.sh --out ~/tt_traces/energybench-session

# 5. Sanity-check on the spot: the control must agree with its twin, and the
#    arms must separate. If they do not, say so in the handover -- do not
#    re-run and keep the better session.
column -s, -t ~/tt_traces/energybench-session/power.csv | less -S

# 6. Send the WHOLE directory home. Analysis needs tt_sim/ and numpy and is not
#    time-critical; being at the card is.
rsync -av ~/tt_traces/energybench-session/ <home>:~/energybench-session/
```

### What the schedule is, and why it is interleaved

One **cycle** runs the baseline and then every workload once. The session runs
several cycles. Arms are **not** blocked (all of A, then all of B): this project
has been bitten by drift before, and a blocked schedule turns a slow thermal
ramp into a fake result. The established discipline here is interleaved A/B with
a verified-zero control, and that is what this is.

At the defaults, per cycle:

| slot | what | duration |
| --- | --- | --- |
| 0 | **baseline** — device open, nothing launching | 30 s |
| 1 | `idle-0` | 30 s |
| 2–5 | `rv-200000`, `noc-4096`, `mm-4096`, `sfpu-4096` | 30 s each |
| 6–9 | `rv-800000`, `noc-16384`, `mm-16384`, `sfpu-16384` | 30 s each |
| 10 | `noc-4096__control` | 30 s |

3 cycles × 330 s ≈ **16.5 minutes**, plus build and device open/close. `--cycles`
and `--seconds` change it; `--list` recomputes the estimate.

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
session.log     everything the run printed, including the handover block
slots.csv       one row per arm run: its window on the wall clock, its launches
raw/*.csv       the telemetry samples, per slot, timestamped by the shell
launches.csv    energybench's own per-run summary
power.csv       the aggregated input to tt_sim.perf.energy_rank
```

Send **all** of it. `raw/` is what makes a suspicious `power.csv` row
diagnosable later.

### Testing the harness without a card

```bash
perfbench/energybench/run_card_stub_test.sh    # no card, no tt-metal needed
```

`ENERGYBENCH_TELEMETRY_CMD` and `ENERGYBENCH_BIN` stub the two things that need
hardware, so the scheduling, the sampler lifecycle, the per-slot windows, the
launch-rate capture and the settle-trimmed aggregation are all exercised end to
end. The stub board carries a deliberate slow drift, which is what the
interleave is supposed to survive. Anything collected that way is stamped
**NOT-A-MEASUREMENT**.

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

Seven gates run before any ranking is reported, and a failure is a **refusal**,
not a warning. A fit is only meaningful if the measurement it is fitted to
actually said something.

| gate | passes when | refuses when |
| --- | --- | --- |
| `baseline` | the session measured its own idle floor | no `baseline` rows |
| `repeats` | every label has ≥ `--min-repeats` interleave cycles | any label has fewer |
| `samples` | every row kept ≥ `--min-samples` telemetry samples | any row was under-sampled |
| `control` | the control agrees with its twin inside `--sigma` × noise | it does not, or there is no control at all |
| `spread` | the workloads span more than `--sigma` × noise | they sit inside the noise floor |
| `identifiability` | coefficients ≤ workloads − 1 and condition number ≤ `--max-cond` | either bound is broken |
| `rankable` | ≥ 3 workloads have both a vector and a rate | fewer |

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
are fitted to **steady-state repeated-kernel board power**. Using them to predict
a single cold launch is using them outside the quantity they were fitted to, and
the launch machinery term (`c_launch`) is the only part of the model that has
anything to say about it.
