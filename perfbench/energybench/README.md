# energybench — ranking-level energy estimation

The apparatus and the protocol for asking **which workload costs more energy,
and roughly by how much**: the harness, the card protocol, the simulator-side
activity vector, and the analysis that consumes the data.

**Three card sessions have been collected and two of them fit**, on two
different architectures, both preserved under `perfbench/card-sessions/`:

| session | part | gates | LOO Spearman | null | ratio median | ratio max |
| --- | --- | --- | --- | --- | --- | --- |
| `2026-08-13-energybench` | Blackhole p150 | **refused** (`repeats`) | — | — | — | — |
| `2026-08-13-energybench-2` | Blackhole p150 | 13/13 | 0.867 | **0.867** | ×1.98 | ×4.48 |
| `2026-08-17-wh-energybench` | Wormhole n300 | 13/13 | **0.900** | 0.800 | **×1.22** | **×1.65** |

The **null** column is the number every other column has to be read against: a
model with no energy content at all, taking per-launch energy as proportional to
the simulator's cycle count. On Blackhole the fitted four-term model **does not
beat it on ordering** — see [the null model](#the-null-model-and-what-it-costs-the-headline).
Where the fit does earn its keep is the **ratios**, and only on Wormhole.

The first session is kept because it is a **sound measurement that the analysis
refuses**, for a reason it names and an operator can act on. What it taught is
written up in [what the first collected session
changed](#what-the-first-collected-session-changed-2026-08-13), and it changed
the arithmetic, not just the guards.

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
> with an idle baseline and a per-slot pre-idle reading **measured in the same
> session as diagnostics**

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
P_board(w)  = P_floor + rate(w) · E_launch(w)          [watts]
```

`a(w)` is the activity vector `tt_sim.perf.energy_activity` emits; `rate(w)` is
measured; **`P_floor` and every `c` are fitted**, from the arm rows, by
non-negative least squares on measured power directly. The design matrix is
`[1, rate(w), rate(w)·a_j(w)]`.

**`P_floor` is fitted rather than subtracted, and that is not a refinement.**
The board's busy-state floor is not measurable on a DVFS part: an idle slot
launches nothing by definition, so the board drops to its idle clock state —
measured, 800 MHz against the arms' 1350 — and an idle board is not the busy
board minus the kernel. There is no measurement of the busy floor to subtract,
so it has to be extrapolated from the arms. The measured baseline stays, as a
completeness check and as the recorded idle-state diagnostic; nothing is
differenced against it. The earlier parameterisation, which fitted
`(P − P_baseline)/rate`, is [why](#the-arithmetic-changed-and-why).

## The arms, and why these

A ranking needs something to rank, so the arms have to span genuinely different
activity mixes rather than differing in size:

| arm | what it does | what it moves in the activity vector |
| --- | --- | --- |
| `idle` | a kernel that returns immediately | nothing — the launch machinery alone |
| `rv` | a dependent integer chain (one MUL + three ALU ops per iteration) on BRISC | `instr_retired`, and essentially nothing else |
| `noc` | barriered DRAM→L1 reads on BRISC | `noc_bytes_total`, `noc_flight_cycles`, flat compute |
| `mm` | `matmul_tiles` on two **resident** tiles | `matrix_arith_cycles`, flat NoC |
| `sfpu` | Int32 tile add on the same two resident tiles | `sfpu_busy_cycles`, flat NoC |

`matrix_arith_cycles`, not `matrix_busy_cycles`. The Matrix Unit also executes
the dest-register bookkeeping every compute kernel pays, and the `sfpu` arm pays
**41 cycles of it per iteration against zero matrix arithmetic ops** — so
occupancy is a column both compute arms move and it is not a statement of what
one arm isolates. The measurement, and what it cost the first fit, is
[below](#the-sfpu-arm-was-not-running-on-the-matrix-unit-2026-08-13).

**That table IS the term set.** The right-hand column is not a description of
what was observed afterwards; it is what each arm was *built* to move, written
down before any board was plugged in, and it is transcribed verbatim into
`DESIGNED_ARM_TERMS` in `tt_sim/perf/energy_rank.py`. The fit's default terms are
exactly those — one per non-idle arm — which is why the design can be honoured
**without searching for a term set that works**. See [the term budget is set by
the arms, not the rows](#the-term-budget-is-set-by-the-arms-not-the-rows). If an
arm is added or changed here, that constant has to be edited by hand in the same
commit; the analysis refuses an arm it does not know rather than fitting a
smaller model than it reports.

Three design decisions in that table are load-bearing.

**The compute arms hold their tiles resident.** The reader pushes exactly two
tiles once; the inner loop never pops or refills. So a compute arm's NoC traffic
is *fixed* however large its inner loop grows, which is what makes the compute
column separable from the NoC column instead of the two moving together.

**`mm` and `sfpu` share a reader, a writer and a skeleton** and differ only in
which backend unit their inner loop occupies. Without that pair a single
"compute" coefficient would absorb both units and neither number would mean
anything.

**Every arm runs at two inner counts.** Not to buy more terms — see the section
below, they buy none — but because the within-arm ratio (the same arm, 4× the
work) is the sharpest ranking test in the *design*, its two points differing in
exactly one thing. Two scales is also what gives leave-one-out a degree of
freedom to spare: four terms plus launch plus floor is six coefficients, and
five workloads would not carry them.

**In the design. Not, on the evidence so far, in the instrument.** The first
collected session resolved the arms in the mean but moved none of the four
within-arm ratios by the 3 noise floors the set as a whole clears, and one of
the four sat inside the floor entirely —
the numbers are [below](#what-board-power-did-and-did-not-resolve). That is a
measured limit of ~1 Hz board telemetry over three interleave cycles, not a
reason to change the design, and more cycles is the only lever that moves it.

### The term budget is set by the arms, not the rows

The first collected session was refused by `identifiability` — 8 coefficients
against 9 workloads, condition number **2.34e+07** against a 1e6 cap. The
arithmetic was right and the cause was the **term budget**, not the measurement.

The budget used to be `n_workloads − 3` and nothing else. That is a
degrees-of-freedom bound and it is still correct as one, but it grows with the
number of *rows*, so more workloads let the spread-ranked selector reach further
into eleven mutually-correlated counters. Measured on this session's own design
matrix, over all 11 spread-carrying columns:

| budget | subsets over the 1e6 cap | worst condition number |
| --- | --- | --- |
| 3 | 0 / 165 (0 %) | 4.4e+05 |
| 4 | 2 / 330 (1 %) | 2.8e+16 |
| 5 | 19 / 462 (4 %) | 1.2e+17 |
| **6** — the session's budget | **237 / 462 (51 %)** | 4.0e+17 |

Fitted with the four terms the arms were *designed* to separate —
`instr_retired`, `noc_bytes_total`, `matrix_busy_cycles`, `sfpu_busy_cycles`
(the `mm` term has since become `matrix_arith_cycles`; these numbers are the
ones the session was refused against, and conditioning is a property of the
columns and the rates) — plus launch and floor:

```
as run,       2 scales,  9 workloads:  6 coefficients, cond = 6.16e+02   PASSES
with scale 2, 3 scales, 13 workloads:  6 coefficients, cond = 7.14e+02   PASSES
```

So the budget is now **`min(designed terms present, n_workloads − 3)`**. The
degrees-of-freedom rule still applies unchanged; it is simply no longer the only
bound. The number of independent activity *directions* is set by the number of
distinct arms — currently five — and no amount of extra rows changes it.

**Adding a third scale does not help, and costs card time for nothing.** This is
the tempting next move and it is a dead end, so it is recorded here rather than
left to be proposed again. Merging the intermediate-scale activity CSV in gives
13 workloads and, on the old rule, a budget of **10**; **all eleven** possible
ten-term subsets condition at **≥ 3.26e+16**. An extra scale is a row that
interpolates between rows already present. It buys repeat count and it buys
within-arm ratio resolution — which is a real reason to want it, see [what board
power did and did not
resolve](#what-board-power-did-and-did-not-resolve) — but it buys **no new
activity direction**, and it is not a route to identifying more terms. Another
arm would be; another scale never is.

**Why this is not laundering.** Restricting to these terms is legitimate for one
reason only: *they were fixed a priori*. The arms table above predates every
measurement, each arm was built to move exactly one term, and the code path is
built so that the restriction cannot come from anywhere else —
`designed_terms()` is handed **arm names and nothing else**: no design matrix, no
target, no condition number, no gate verdict. Choosing a term set by which one
conditions best, fits best, or passes `identifiability` would be exactly the
laundering this whole apparatus exists to prevent, and the difference has to be
structural rather than asserted, because a comment claiming good intentions is
worth nothing in a year. A term *outside* the designed set gets in by one route:
a human names it with `--terms`, before the run, and every report and coefficient
file of that fit is stamped `operator-specified` rather than `designed` so the
two models can never be confused. A designed term that cannot be fitted — no
spread across the workloads, an exact linear duplicate, or outside the degrees of
freedom — is **dropped and named in the report**, never dropped silently.

**And the gate can still fail.** 2 of the 330 four-term subsets on this session
still blow the cap, and a session refused by `identifiability` *with the designed
terms* is reporting something real: that the arms did not separate in that
measurement. `energy_rank_test.py` builds exactly that case — the `sfpu` column
made a near-multiple of the `mm` column — and asserts the refusal, alongside the
passing case.

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
#    `attempts` agree; every `therm_trip_delta` is 0; the control agrees with its
#    twin; the arms separate. On the clock, look at `sysfs_aiclk_drift_pct`
#    rather than at `sysfs_aiclk_mean`: the baselines WILL read ~800 MHz against
#    the arms' 1350 (an idle slot is in the idle DVFS state and no option pins
#    it), which is expected and handled, but any ARM row with a drift of a few
#    percent or more straddled a transition and will be excluded from the fit --
#    which costs that label a repeat. If they do not, say so in the handover --
#    do not re-run and keep the better session.

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
taken with the device free in the gap before the arm starts. It is a
**diagnostic**, and a fallible one — see [the baseline is a
diagnostic](#the-baseline-is-a-diagnostic-and-is-measured-like-one).

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

**Run it with an interpreter that has the trace dependencies.** The reduction
step imports `tt_sim.trace.report` → `tt_sim.trace.dwarf` → `pyelftools`, and
pyarrow behind it, and a bare system `python3` typically has neither. Pass a
virtualenv interpreter that does, as `TT_SIM_PYTHON=/path/to/venv/bin/python3`.

The script now **proves those imports before any arm runs** and exits 4 with the
missing module named, because the failure mode it had was the simulator-side
version of the card's silent slots: the import is deferred to reduction time, so
every arm booted the simulator, produced counters, threw `ModuleNotFoundError`,
printed the traceback, and the loop carried on to the next arm — finishing
"successfully", exit 0, with an activity CSV that was empty or short. A reduction
that fails now **fails that arm loudly**, the run exits non-zero, the CSV is
checked against the schedule afterwards, and any label missing from it is named
in a `THIS ACTIVITY MATRIX IS INCOMPLETE` handover block. A hole here is exactly
as dangerous as a hole in `slots.csv`, because `energy_rank` drops an unmatched
label *with a note* and then fits a smaller design than the operator thinks.
Both directions are asserted in `run_card_stub_test.sh`.

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
size for fitting against a card. `activity-sim-blackhole-card.csv` is the one at
the *card's* inner counts and is the file that joins with the collected session;
`activity-sim-blackhole-card-scale2.csv` is the intermediate scale (rv-400000,
noc-8192, mm-8192, sfpu-8192), collected to answer whether a third scale would
buy anything. [It does not.](#the-term-budget-is-set-by-the-arms-not-the-rows)

### What it produced, 2026-08-13, Blackhole, cost model on

Per launch, one launch per run, `--scales "1 4"`:

```
    label    cycles     instr      disp      nocB       txn       MAT   MATarith      SFPU      PACK       THC   rvstall   txstall    flight
   idle-0      7499     14610        27         0         0         1         1         3         0         0      9602         0         0
    rv-64      8099     16144        27         0         0         1         1         3         0         0     11074         0         0
   rv-256      9399     19624        27         0         0         1         1         3         0         0     14100         0         0
    noc-8     11499     22451        27     16384        16         1         1         3         0         0     21767         0      3600
   noc-32     23199     45356        27     65536        64         1         1         3         0         0     57368         0     14400
     mm-8     10499     23207      1262      6144         6       533       516         3        16        52     16017      3198      1338
    mm-32     12099     28620      3590      6144         6      2117      2052         3        16       124     18591      6078      1338
   sfpu-8     11699     26328      4719     12288         6       342        12      1028        16        29     18888      4313      1456
  sfpu-32     16199     40865     17031     12288         6      1326        12      4100        16        29     26854      8801      1456
```

(`MAT` is `matrix_busy_cycles`, the Matrix Unit's full occupancy; `MATarith` is
`matrix_arith_cycles`, the part of it that moved operand data.)

Read it as a design matrix and the separation is the thing to check, because it
is what decides whether any coefficient is identifiable:

* **`rv` moves `instr_retired` alone** — 16,144 → 19,624 with every other column
  pinned. Nothing else in the set does that.
* **`noc` moves bytes and flight alone** — 16,384 → 65,536 B and 3,600 → 14,400
  cycles, with `instr_retired` rising only because the issue loop is longer, and
  no Tensix dispatch at all beyond the 27 the firmware itself issues.
* **`mm` moves `MATarith` 516 → 2,052 with `nocB` pinned at 6,144.** That flat
  NoC column is the resident-tile design working: 4× the matmul work,
  byte-identical traffic.
* **`sfpu` moves `SFPU` 1,028 → 4,100 with `nocB` pinned at 12,288**, likewise —
  and with `MATarith` pinned at **12**, which is what makes the pair separable.
  Its `MAT` column moves 342 → 1,326 all the same, which is the whole of the
  next section.
* **`idle` is not zero**, and should not be: 14,610 instructions and 9,602 stall
  cycles are the launch machinery and the firmware it runs. Every other arm pays
  that too, which is what the launch column is for.

One correlate is worth naming because it showed up in the fit: `mm` and `sfpu`
**both** raise `MAT`, so `matrix_busy_cycles` is not a pure `mm` column — `mm`
at inner 32 has `MAT` 2,117 and `SFPU` 3, `sfpu` at inner 32 has `MAT` 1,326 and
`SFPU` 4,100.

> This paragraph used to attribute the `sfpu` arm's `MAT` column to `copy_tile`
> "running on the matrix path". **That was wrong.** `copy_tile` is called twice,
> outside the inner loop, and cannot produce a column that scales with `inner`
> at all — and on this arm it issues no Matrix Unit op whatever, because an
> Int32 tile is unpacked straight to Dest. The real cause, and the fix, is
> [below](#the-sfpu-arm-was-not-running-on-the-matrix-unit-2026-08-13). Since
> that fix the `mm` arm is fitted against `matrix_arith_cycles`, which the
> `sfpu` arm does not move.

### What it produced, 2026-08-17, Wormhole, cost model on

At the **card's own** inner counts, one launch per run, `--scales "1 4"` — the
same nine workloads the Blackhole session used, so the two are directly
comparable:

```
     label  cycles    instr    disp     nocB   txn     MAT MATarith    SFPU PACK   THC  rvstall txstall  flight
    idle-0    5599    11316      27        0     0       1        1       3    0     0     6449       0       0
 rv-200000 1605699  4787315      27        0     0       1        1       3    0     0  3230950       0       0
  noc-4096 1783399  4261977      27  8388608  8192       1        1       3    0     0  4644789       0 1646592
   mm-4096  282399  1009418  393671     6144     6  270341   262148       3    2 20530   392318  505823    1206
 sfpu-4096  786699  2670321 2138707    12288     6  167942        4  524292    2    51  1252923  781149    1461
 rv-800000 6405699 19114546      27        0     0       1        1       3    0     0 12903724       0       0
 noc-16384 7116397 17012895      27 33554432 32768       1        1       3    0     0 18558860       0 6586368
  mm-16384 1105699  3982308 1573319     6144     6 1081349  1048580       3    2 81970  1535937 2017247    1206
sfpu-16384 3121399 10623094 8553043    12288     6  671750        4 2097156    2    51  4973639 3115869    1461
```

The separation the design was built for survives at card scale, and one column
is worth checking first: **`MATarith` is 4 on both `sfpu` rows and 262,148 →
1,048,580 on the `mm` rows**, so the fix that freed `sfpu_busy_cycles`
[below](#the-sfpu-arm-was-not-running-on-the-matrix-unit-2026-08-13) holds on
Wormhole too and was not a Blackhole accident. `MAT` still moves on the `sfpu`
arm — 167,942 → 671,750 — which is the dest bookkeeping, and exactly why the
`mm` arm is fitted against the arithmetic column rather than the occupancy one.

**The nine runs took 2 h 39 m of simulator time** — 22.2M simulated cycles at
~2,330 cycles/s — which is what the [inner-count warning](#the-simulator-side)
above means in practice. `sfpu-16384` alone is 45 minutes of it.

### The `sfpu` arm was not running on the Matrix Unit (2026-08-13)

The first fitted session put **`sfpu_busy_cycles` at exactly 0** — clamped on the
NNLS non-negativity boundary, because the unconstrained fit wanted it negative —
and mispredicted `sfpu-16384` by 2.21×, predicting it almost entirely through
`matrix_busy_cycles`. That is not a collinearity artefact: across the nine
workloads `corr(rate·MAT, rate·SFPU)` is 0.147 and the design conditions at
1.18e3. The `sfpu` arm really was feeding the matrix column.

Running the arm under `TT_SIM_DIAG_CO_ISSUED=1` says exactly what with. Per
`add_int_tile<DataFormat::Int32>` iteration, on Blackhole:

| unit | opcode | per iteration | what it is |
| --- | --- | --- | --- |
| MATH | `INCRWC` | 32 | sfpi's `dst_reg++`, 8 per `_add_int_` call × 4 faces |
| MATH | `SETRWC` | 9 | 2 per face from the LLK's dest-address walk, + 1 `clear_dst_reg_addr` |
| SFPU | `SFPLOAD` | 64 | two operands × 8 unrolled iterations × 4 faces |
| SFPU | `SFPIADD` | 32 | the add |
| SFPU | `SFPSTORE` | 32 | the result |

**41 Matrix Unit ops per iteration and not one arithmetic op**: no `MVMUL`, no
`ELWADD`, no `MOV*`. The whole arm's matrix column is `41·inner + 14`, where the
14 is one-off `ZEROACC`/`ZEROSRC`/`SETRWC` at kernel start. The `mm` arm for
comparison is `66·inner + 5` — 64 `MVMUL` and 2 `SETRWC` per iteration.

So the arm was fine and the **attribution** was wrong. `SETRWC` / `INCRWC` /
`CLEARDVALID` / `GATESRCRST` are dispatched to the Matrix Unit (FPU) by the ISA
documentation and the cost tables charge them a cycle each, both correctly —
they really do take an issue slot. But they update the RWC counters, the dvalid
flags and the SrcB operand cache and move **no operand data**, and an energy
coefficient against that column is being asked for joules per cycle of the
matrix array. Every compute kernel tt-metal builds pays them; a vector kernel
pays 41 per iteration.

The fix is `tt_sim.trace.counters` publishing `bookkeeping_cycles` — a *subset*
of `busy_cycles`, cut by whether the opcode moved any operand data — and
`tt_sim.perf.energy_activity` deriving `matrix_arith_cycles` from the
difference. `matrix_busy_cycles` is unchanged and still the full occupancy,
because that is the right number for a performance reader; the term was
**added**, not redefined, so a fit cannot silently keep using the old column.
`tt_sim/trace/events.py` carries the opcode taxonomy, and a test asserts it
partitions `MatrixUnit.OPCODE_TO_HANDLER` exactly — a Matrix opcode added later
fails until somebody has said which kind it is.

The card activity CSVs were re-reduced afterwards and **both now carry the
column**, so both fits have an `mm` direction. The mechanism that made that safe
is worth keeping in view: the term was **appended** to the schema rather than
redefined, so a CSV collected before it reads back with the column at zero, the
designed fit drops it and *names it in a note* — a stale file loses the term
loudly instead of quietly refitting the old one.

Re-reducing costs a full re-run of the arms at the card's inner counts: about
22M simulator cycles, which measured out at **2 h 39 m** on the Wormhole side at
~2,330 cycles/s. Budget for that before changing a term, not after.

## The analysis

```bash
python3 -m tt_sim.perf.energy_rank \
    --activity perfbench/energybench/activity-sim-blackhole.csv \
    --measured ~/energybench-session/power.csv \
    --report report.txt --json report.json
# ...and on the session that has actually been collected, joined to the activity
# vectors at the SAME inner counts (the smoke CSV above joins with nothing here):
python3 -m tt_sim.perf.energy_rank \
    --activity perfbench/energybench/activity-sim-blackhole-card.csv \
    --measured perfbench/card-sessions/2026-08-13-energybench/power.csv
# -> identifiability PASSES (6 coefficients, cond 598); repeats REFUSES
#    ('rv-800000' has 2 usable cycles of 3); exit status 1.

# The two that fit. Each activity CSV joins only with its own architecture's
# session, because the labels are shared but the vectors are not.
python3 -m tt_sim.perf.energy_rank \
    --activity perfbench/energybench/activity-sim-blackhole-card.csv \
    --measured perfbench/card-sessions/2026-08-13-energybench-2/power.csv
# -> 13/13 gates; LOO Spearman 0.8667, null 0.8667; ratios x1.98 / x4.48.
python3 -m tt_sim.perf.energy_rank \
    --activity perfbench/energybench/activity-sim-wormhole-card.csv \
    --measured perfbench/card-sessions/2026-08-17-wh-energybench/power.csv
# -> 13/13 gates; LOO Spearman 0.9000, null 0.8000; ratios x1.22 / x1.65.
```

### The gates

Thirteen checks run before any ranking is reported. Twelve are **refusals**, not
warnings. A fit is only meaningful if the measurement it is fitted to actually
said something.

| gate | passes when | refuses when |
| --- | --- | --- |
| `telemetry` | every row has a status of `ok`, ≥ 1 successful sample and a finite power | any row was never measured |
| `baseline` | the session measured its own idle floor | no `baseline` rows. It is a **completeness check and a diagnostic**, not the fit reference |
| `schedule` | every interleave cycle carries every label | a cycle has a hole in it |
| `thermal` | `tt_therm_trip_count` did not move | the part throttled, or there is no thermal record |
| `clock` | the surviving arm slots agree on their clock to `--max-clock-drift-pct` | slots ran at different clocks, or there is no clock record. A slot whose clock moved **within itself** is *excluded and named*, not refused |
| `baseline_clock` | the baseline ran at the arms' clock | **never refuses.** A DVFS split is reported as a finding: *baseline subtraction is invalid for this session* |
| `repeats` | every label has ≥ `--min-repeats` **surviving** cycles | any label has fewer, after exclusions |
| `samples` | every row kept ≥ `--min-samples` telemetry samples | any row was under-sampled |
| `control` | the control agrees with its twin inside `--sigma` × noise | it does not, or there is no control at all |
| `spread` | the workloads span more than `--sigma` × noise | they sit inside the noise floor |
| `identifiability` | coefficients ≤ workloads − 1 and condition number ≤ `--max-cond` | either bound is broken. What it is *offered* is the **designed** term set — one per non-idle arm, budget `min(designed, workloads − 3)`, [fixed a priori](#the-term-budget-is-set-by-the-arms-not-the-rows) — never a set chosen because it passes this gate. It is also where "are `P_floor` and `c_launch` separable in *this* session?" is answered |
| `target_triviality` | the floor-and-launch-rate model alone explains less than `--max-triviality-r2` of the target | a model that knows **no activity at all** already reproduces the ranking |
| `rankable` | ≥ 3 workloads have both a vector and a rate | fewer |

`--max-clock-drift-pct` defaults to **5 %**, which is 67 MHz at 1350 MHz: tight
enough to catch a 1350 → 800 MHz transition, loose enough for ordinary DVFS
ripple. The first collected session bears that out — every row's within-slot
drift is 0.000 % or 0.519 % except one at 41.398 %, so 5 % separates them with
a factor of eight in hand either way.

**Two of these are new because of that session, and one of them is not a
refusal.** `baseline_clock` cannot refuse: the confound it detects is
structural, since a slot that launches nothing is in the idle DVFS state by
definition, and `tt-smi` 6.2.0 offers no way to pin the clock — so a refusal
there would be a gate that can never pass, which is as damaging as one that can
never fail. It is wired to the **arithmetic** instead: the finding is precisely
the reason `P_floor` is fitted rather than subtracted.

**One row can be excluded without discarding the session, and only one thing
does it.** A slot whose clock moved *during* it has a mean over two DVFS states
and is not a measurement of either; it is cut from the fit and printed under
`## Rows excluded from the fit (still part of the record)`. Whether what
survives is still a session is then decided independently by `repeats`, counting
survivors — so the exclusion can still cost the session, which is what keeps it
honest. On the first collected session it does exactly that.

**A guard that cannot fail is as damaging as one that cannot pass**, and that is
not a slogan here — `target_triviality` exists because the old arithmetic had a
hole no gate could see, and `baseline_clock` reports rather than refuses because
refusing would have been the other failure. The term selector likewise does
**not** filter on the identifiability gate's own threshold: within the candidate
set it is given it excludes only columns with no spread and columns that are
exact linear duplicates — both arithmetic, not judgement — so a merely
ill-conditioned set reaches the gate and is refused there. A selector that
avoided what the gate checks would make the gate incapable of failing. What that
candidate set *is* comes from the design and not from the data — the arms table's
own terms, [see above](#the-term-budget-is-set-by-the-arms-not-the-rows) — which
is a narrowing on a-priori grounds and is stamped as such in every report;
`energy_rank_test.py` asserts that a degenerate design is still refused with
those terms in place. It builds a passing session and a failing one for every
gate in the table, and asserts both.

The noise floor is derived from **this session's own repeats** (the RMS of each
label's across-cycle standard deviation), never carried over from another
session — a floor from another session is exactly the drift the control exists
to catch. It is derived from the **arm** repeats specifically, after the
within-slot drift exclusion, and the **baseline is not in it**: an idle-state
label's variance is not the noise of the busy-state measurements it would be
used to judge. On this session that is the difference between 0.441 W and
0.838 W, because the baselines sat at 800 MHz and swung 4.3 W between cycles
while no arm moved by more than 1.2 W. The control label *is* counted — it is a
genuine arm measurement, and including it lowers the floor here (0.441 W against
0.463 W without it), so it makes its own gate stricter rather than more
permissive.

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

## What the first collected session changed (2026-08-13)

Same day, same board, after the tool upgrade: a three-cycle session on a
Blackhole p150 with `tt-smi` 6.2.0 that **collected cleanly**. The whole
directory is preserved at `perfbench/card-sessions/2026-08-13-energybench/` —
`power.csv`, `slots.csv`, `launches.csv`, `session.log` and 99 files under
`raw/` — and the regression tests read it rather than a transcription of it.

**Collection is now sound.** 33/33 slots `status=ok`; 29–37 telemetry samples
per slot with `attempts == samples` and **zero** `sample_failures`;
`therm_trip_delta` 0 everywhere; launch rates reproducible across the three
interleave cycles to ~1 %. Every net added after the first session held, and
none of them fired. What the session produced, means over its three cycles:

```
label                 meanP(W)   sd     rate(/s)   aiclk(MHz)
baseline               37.736   2.36        0        800
idle-0                 66.003   1.149    701.9      1350
rv-200000              67.341   0.553    406.1      1350
noc-4096               67.872   0.319    339.7      1350
mm-4096                68.070   0.345    604.5      1350
sfpu-4096              68.086   0.177    491.8      1350
rv-800000              68.271   0.118    179.3     ~1329   <- see below
noc-16384              68.369   0.106    133.4      1350
sfpu-16384             68.442   0.094    263.1      1350
mm-16384               68.927   0.130    445.1      1350
noc-4096__control      68.386   0.106    340.4      1350

noise floor (RMS of per-label across-cycle SD, ARM rows)  0.441 W
arm spread                        2.924 W  =  6.63 x floor   -> `spread` PASSES
control vs its twin               0.513 W  =  1.16 x floor   -> `control` PASSES
```

**That floor is over the arm rows only, and the baseline's 2.40 W across-cycle
SD is not in it.** It is the same 800-versus-1350 MHz confound as everywhere
else on this page: the baseline is a diagnostic in the idle DVFS state, so its
swing is that state wandering rather than the repeatability of the arms.
Including it read 0.838 W — a factor of 1.9 on the yardstick that `control` and
`spread` are measured against, in a session whose own `baseline_clock` finding
says baseline subtraction is invalid. No verdict here flips either way (control
0.513 W against 3 × 0.441 = 1.32 W), but `control` passes when the disagreement
is *below* the floor, so the inflated version made the session's drift detector
roughly twice as permissive as intended.

The arms are in the right order and separate in the mean. Two things then went
wrong with the clock, and they are **different problems with different fixes**.

### The baseline is in a different DVFS state, and always will be

Every baseline slot ran at **800 MHz** and every arm at **1350 MHz**. Over all 33
rows that is a 42.33 % clock spread; over the arm rows alone, 1.59 %.

This is not drift. A baseline slot launches nothing *by definition*, so the board
drops to its idle DVFS state for it, and no scheduling change alters that.
**`tt-smi` 6.2.0 offers no clock-pinning option** — `--help` lists only
`-l/-v/-s/-ls/-f/-c/--offline/-r/--snapshot_no_tty/-glx_*/--no_reinit/
--use_luwen/--eth_train_skip` — so the confound cannot be removed at the source
either.

A gate that refuses on it therefore refuses *every session anyone can ever run*,
which is the "guard that cannot pass" failure. So:

* the `clock` gate's across-session check now runs over the **arm rows only**,
  at the same cap (1.59 % against 5 %, and 0.007 % over the rows it actually
  fits);
* a separate `baseline_clock` check reports the baseline-versus-arm gap as a
  **named finding**: *baseline subtraction is invalid for this session*;
* and that finding is wired to the arithmetic rather than to a refusal.

### The arithmetic changed, and why

The old target was `y(w) = (P(w) − P_baseline) / rate(w)`. On this session the
numerator is ~30 W of DVFS step plus **2.9 W** of actual workload difference, and
the denominator varies 5.3× — so the target is very nearly the launch period.
Measured: regressed on `1/rate` alone, **with no activity term whatsoever, R² =
0.9995**.

A fit on that would have reported an excellent leave-one-out Spearman for a
ranking that is 99.95 % *"whichever workload launches slowest costs most"*. It
passes `spread`. It passes `control`. It passes `identifiability`, which inspects
the design matrix and not the target. The hole was in the arithmetic, not in the
guards, and no guard could see it.

Two changes, and they are the same change:

1. **The floor is fitted, not subtracted.** `P(w) = P_floor + rate(w)·(c_launch +
   Σ c_j a_j(w))`, `P_floor ≥ 0` free and estimated from the arm rows. There is
   no measurement of the busy-state floor to subtract — an idle board is not the
   busy board minus the kernel — so it is extrapolated. The same session's target
   under the new parameterisation scores **R² = 0.36** against rate alone.
2. **A `target_triviality` gate**, defaulting to 0.95 and exposed as
   `--max-triviality-r2`, refuses when the floor-and-launch-rate model already
   explains the target. A model that reproduces the ranking without knowing any
   activity is not evidence about activity.

`P_floor` and `c_launch` are distinguishable here in principle because the launch
rates vary 5.3×, and empirically because that design's condition number is 4.7 —
**598** for the full six-coefficient design against this session's own activity
vectors — against a 1e6 cap, checked by `identifiability`, not assumed, and not
worked around.

### One slot genuinely was contaminated

Independently of all that, `sysfs_aiclk_drift_pct` shows **cycle 1's `rv-800000`
drifted 41.398 % within the slot** — `sysfs_aiclk_min` 800, `max` 1350: it
straddled a DVFS transition. The same row carries `power_sd_w` **5.14** against
0.1–0.9 everywhere else, `pre_idle_w` 75.0 and `delta_w` −6.81. Every other row
in the session drifted 0.000 % or 0.519 %.

That row's mean is a mean over two clock states and is not a measurement of
either. But one contaminated slot must not discard 32 sound ones, so it is
**excluded from the fit and named in the report**, and `repeats` then decides
independently whether what survives is a session. It is not: `rv-800000` drops to
two repeats against a `--min-repeats` of 3, and **the session is refused there**.

That is the right refusal reached through the right gate. It says what to do —
run more cycles — rather than merely that a clock moved.

### What board power did and did not resolve

The honest half, and it qualifies a claim this file used to make flatly. The arms
separate in the **mean**: 2.924 W, 6.63 noise floors, correct ordering with
`mm-16384` highest and `idle-0` lowest. The **within-arm 4× scalings**, billed
above as the sharpest ranking test in the set, do far less well:

| arm | lo → hi (W) | delta | vs noise floor |
| --- | --- | --- | --- |
| `rv` | 67.341 → 68.271 | 0.930 | 2.11× |
| `mm` | 68.070 → 68.927 | 0.858 | 1.94× |
| `noc` | 67.872 → 68.369 | 0.497 | 1.13× |
| `sfpu` | 68.086 → 68.442 | 0.356 | 0.81× |

(`rv`'s becomes 0.971 W, 2.20×, once the contaminated row above is excluded.)

All four have the right sign. Three clear the noise floor and `sfpu` is
**inside** it — but **not one of the four reaches the 3 floors that `spread`
holds the set as a whole to**, and 1.1× is not a margin worth claiming. **Board
power resolves these arms in the mean but does not resolve the within-arm ratio
to the bar the rest of this analysis is held to**, at three interleave cycles.
That is a statement about these four numbers and this session, and it should not
be generalised further.

> These ratios were previously quoted as 1.11× / 1.02× / 0.59× / 0.42× against a
> floor of 0.837 W. That floor included the baseline's DVFS swing and was the
> wrong yardstick; the deltas themselves are unchanged. Correcting it moves
> `noc` from inside the floor to just outside it, which is a correction to the
> scale and **not** new evidence that `noc`'s scaling was resolved.

The floor is the RMS of each **arm** label's across-cycle SD, so more cycles
shrink it: `--cycles 6` is about 46 minutes and is the only lever here that
moves it.

## The baseline is a diagnostic, and is measured like one

A reference that swings 42 % between cycles is not a reference — and the first
collected session showed that on a DVFS board it *cannot* be one, because the
idle slot and the arms are in different clock states by construction. The
baseline is therefore no longer differenced against anything; the fit
extrapolates the busy-state floor from the arms. What it is still for is
completeness (a session without one did not run this schedule) and diagnosis
(how far the fitted floor sits above the idle state).

Three things were done to it while it was still a reference, and all three are
worth keeping for that diagnostic role:

**Every slot gets its own idle reading, seconds before it** (`pre_idle_w`, and
`delta_w = power_w - pre_idle_w`). The device is free in the gap between slots,
so this costs two telemetry calls and no schedule change. A drift that takes a
cycle to develop cannot get between a slot and a reading taken seconds earlier,
whereas it sits squarely between a slot and a per-cycle baseline.

**But the probe has no way to make the board clock down or cool first**, and the
first collected session caught it twice: cycle 1's slot 6 read **75.0 W** — above
every arm in the session — for a `delta_w` of −6.81, and its slot 9 read
**62.0 W** against a 37.7 W idle baseline. Neither is an idle board; both caught
one still coming down from the slot before. That is why `delta_w` stayed a
**diagnostic and never became a fit input**, and why nothing gates on it —
refusing a session over a column nothing is computed from would be theatre. What
the analysis does instead is **flag the implausible cells by name** (above the
arm floor, negative `delta_w` on a slot that launched, or nearer the busy state
than the measured idle floor) so a reader scanning the column is not misled by
it.

**The baseline slot is measured identically** — same samplers, same settle trim,
same pre-idle probe, same duration — rather than being a bare `sleep` with a
different sampling path. If the baseline and the arms are not the same quantity
measured the same way, their difference is partly an instrument artefact.

**The clock is recorded on every row**, per slot, with a min, a max and a drift
percentage. That record is what makes all three of the clock findings above
possible, and it costs nothing: sysfs, no device handle, unperturbed by the
workload.

`delta_w` is reported per row and is still the right column to eyeball at the
card — with the flagging above in mind.

> **Correction.** An earlier version of this file said "the fit still uses the
> session baseline, unchanged — `energy_rank`'s arithmetic was not touched".
> That is **no longer true, and the sentence was wrong to keep**. The arithmetic
> was touched, deliberately and for a measured reason: the baseline is not
> subtracted at all any more. See [the arithmetic
> changed](#the-arithmetic-changed-and-why).

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

**One collinearity is known, will not go away, and the new parameterisation made
it worse rather than better.** The launch constant and the `idle` arm's fixed
instruction count (~14,600 instructions of firmware per launch, whatever the
kernel does) move together, because every arm pays both; fitting the floor adds
a third near-degenerate way to pay a per-launch constant. Measured on synthetic
data with a known truth (`energy_rank_test.py`): noise-free, everything comes
back exactly — `P_floor` to the watt, `c_launch` and all four activity
coefficients to 6 digits. Add 0.05 W of per-row noise and the activity
coefficients still come back within 0.7 %, but `c_launch` is driven to **zero**,
the non-negativity boundary, with the deficit absorbed into `instr_retired`.
Under the old parameterisation the same data gave 57 % of its true value.

So: `c_launch` is **not identifiable at this noise level** and must not be quoted
as a number. The ranking is unaffected — leave-one-out Spearman stays 1.0 on
that data, because the sum the model needs is recovered even when its split is
not. Read `c_launch` as a lower bound on "launch machinery *plus* the firmware
instructions it always runs", and treat `idle-0`'s predicted energy as the least
trustworthy point in any fit. Separating them needs an arm that launches without
executing firmware, which does not exist.

`P_floor` and `c_launch` are a different pair and they *are* separable in
principle, because the launch rate varies across the arms — 5.3× on Blackhole,
15× on Wormhole. Empirically the full six-coefficient design — the four designed
terms plus launch and floor, against each session's own activity vectors —
conditions at **1.15e3** on the Blackhole session and **240** on the Wormhole
one, against a `--max-cond` of 1e6. Those are facts about each session's rates,
checked by the `identifiability` gate rather than assumed, and a session whose
arms all launched at the same rate would be refused there. The wider rate span
is most of why Wormhole conditions almost 5× better.

### The null model, and what it costs the headline

Every ranking number is now reported beside a model with **no energy content at
all**: per-launch energy taken as proportional to the simulator's own cycle
count. No coefficients, no fit, nothing from the card. Both statistics are
scale-free, so the null needs no constant and there is no parameter in it to
tune.

It exists because the headline is easy to over-read. The fit target is board
power, but the *reported* ranking is per-launch energy, `(P − P_floor)/rate`,
and across these arms the power span is small against the floor — 2.9 W over a
fitted 30.1 W on Wormhole — while the launch rate varies 15×. So the measured
energy ordering is mostly an ordering by **how long the kernel took**, which the
cycle model already predicts and which needs no energy modelling whatever.

On both collected sessions:

| | Blackhole p150 | Wormhole n300 |
| --- | --- | --- |
| LOO Spearman, fitted | 0.8667 | 0.9000 |
| Spearman, null | **0.8667** | 0.8000 |
| ratio median, fitted (LOO) | ×1.98 | **×1.22** |
| ratio median, null | ×2.36 | ×2.15 |
| ratio max, fitted (LOO) | ×4.48 | **×1.65** |
| ratio max, null | ×8.37 | ×8.43 |

Read plainly: **on ordering the fit is worth one rank swap on Wormhole and
nothing at all on Blackhole.** Where it is clearly worth something is the
**ratios**, and there only on Wormhole — ×1.65 worst case against the null's
×8.43, a factor of five. That is the opposite of what the Blackhole write-up
expected, which said the ratios were the part the data did *not* support.

Two consequences worth stating:

* **`target_triviality` does not catch this and is not meant to.** That gate asks
  whether the *fit target* is reproducible from the floor and the launch rate, in
  power space; the null asks whether the *reported ranking* is reproducible from
  the cycle count, in energy space. They are independent, and the Wormhole
  session passes the first at R² = 0.03 while the null matches four fifths of
  the second.
* **It is reported, not gated.** A refusal needs a threshold and there is no
  principled one — a model that ties the null on ordering is still well ahead on
  ratios, which is exactly what Blackhole did. The honest treatment is to put
  both in front of the reader every time, so a Spearman is never quoted without
  the number it had to beat.

One genuinely positive result falls out of the same arithmetic and is easy to
miss because it is not about energy: **the cycle model orders these nine
workloads by wall time exactly right on both architectures** — Spearman 1.0000
between simulated cycles and the card's measured launch rate, on Wormhole and on
Blackhole. That needed no fitting, no coefficients and no card-side calibration,
and it is a stronger claim than anything the energy fit makes.

### The four coefficients landed within 1.7× of each other on two parts

Not designed for, not fitted jointly, and worth recording:

| term | Blackhole p150 | Wormhole n300 | apart |
| --- | --- | --- | --- |
| `noc_bytes_total` | 1.033e-10 | 7.275e-11 | 1.42× |
| `matrix_arith_cycles` | 2.968e-09 | 3.903e-09 | 1.32× |
| `sfpu_busy_cycles` | 9.068e-10 | 8.811e-10 | 1.03× |
| `instr_retired` | 4.146e-10 | 2.392e-10 | 1.73× |
| `P_floor` (W) | 67.02 | 30.10 | 2.23× |
| `c_launch` | 0 (clamped) | 0 (clamped) | — |

Two different parts, two sessions three days apart, two independent simulator
runs and two separate NNLS fits, each of which was free to clamp any term to
zero — and did clamp `c_launch` on both. Nothing in either fit knew about the
other.

**This is corroboration, not validation.** The two fits share a design, a term
set and an arms table, so an error in any of those is common to both and this
comparison cannot see it; and the parts differ in process and clock, so exact
agreement was never the expectation either. What it does say is that the
per-unit numbers are not fitting session noise — noise would not land four
coefficients this close twice — while the board floors, which *should* differ
between a p150 and an n300, differ by 2.2×.

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

**Will**, and now does — two sessions pass every gate, on two architectures:

* that tt-sim orders these workloads by energy the way the board does, with a
  stated leave-one-out rank correlation;
* how far off the *ratios* are, per pair, with a median and a worst case;
* a per-term fitted coefficient set with a written record of what it was fitted
  to, usable for comparing two candidate schedules of the same shape;
* an evidenced negative — that the ordering is *not* reproduced, or is
  reproduced no better than [a cycle count already
  did](#the-null-model-and-what-it-costs-the-headline). Both are real results,
  and the second is the one that actually arrived.

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
  the output;
* a value for `c_launch`. It is collinear with the firmware instruction count
  and with the fitted floor, and at 0.05 W of noise on synthetic data it goes to
  the non-negativity boundary. Read it as a lower bound, never as a number;
* the **within-arm 4× ratio**, on either session, *in board power*. Wormhole is
  the better of the two and still only clears the noise floor on one arm of
  four: `mm` +1.076 W = 2.46 floors, `sfpu` +0.285 W = 0.65, `rv` +0.227 W =
  0.52, `noc` +0.158 W = 0.36, against a 0.437 W floor. The reported *energy*
  ordering does get all four pairs right, but that is the launch rate doing the
  work — see [the null model](#the-null-model-and-what-it-costs-the-headline) —
  and it must not be quoted as the board resolving 4× more work as 4× more
  power. It does not;
* an ordering claim that beats a cycle count, on the evidence so far. Blackhole
  ties its null exactly and Wormhole beats it by one rank swap in nine.

One more, worth stating because it is the most tempting mistake: the coefficients
are fitted to **steady-state repeated-kernel board power under sustained load**,
sampled in slot. Using them to predict
a single cold launch is using them outside the quantity they were fitted to, and
the launch machinery term (`c_launch`) is the only part of the model that has
anything to say about it — and it is the least identifiable term in the set.
