# retirebench — rung 4's RV-bound leg

Three envelope comparisons against silicon already exist in this repo: the
component slopes (~1 %), nekbone's per-core zones (± 10.2 %) and
`dramratebench`'s shape. **All three compare totals.** A total can agree while
the decomposition behind it is wrong in compensating directions — an
over-charged ALU paying for an under-charged load path reads as a perfect span.
Rung 4 is the set of legs that look inside, decomposed *by mechanism* so that
compensating interior errors cannot pass. This is the third and last of them.

| leg | what it decomposes | what names its buckets |
| --- | --- | --- |
| [`mechbench`](../mechbench/README.md) | a Tensix-bound program | the hardware's own stall-counter selects |
| [`nocevbench`](../nocevbench/README.md) | a NoC-bound program | the events tt-metal's own NoC recorder emitted |
| **`retirebench`** | an **RV-bound** program | **the program's own structure** — see below |

---

## The criterion

With mechanisms `m` partitioning the span, and an explicit **unattributed**
bucket on both sides so the two denominators are the same quantity:

```
E_total = |Σ c_sim − Σ c_hw| / Σ c_hw      require ≤ 10 %
E_int   = Σ |c_m,sim − c_m,hw| / Σ c_hw    require ≤ 25 %
```

The triangle inequality gives `E_total ≤ E_int` always, so **`E_int / E_total`
is the compensation, measured** — the number a passing total cannot fake, and
the one the report prints most prominently.

Every criterion is **per core**. On the 2026-08-10 part the whole physical
column `x = 11` keeps a wall-clock epoch 1.5e13 cycles from the rest of the die,
so no cross-core span is admissible. Here one artefact is one core and one RISC
by construction, and `per_core` still refuses a sim core silently compared
against a differently-numbered card core — worker `(0,0)` is physical `(1,1)` on
Wormhole and `(1,2)` on Blackhole, which makes that an easy and invisible
mistake.

**Zone discipline**, from ROADMAP §4 and enforced by `zone_budget`: at most **60
zones** per RISC, every measured zone at least **1000 cycles**, and the
**measured** two-marker cost at most **6 %** of the smallest zone.

---

## Hardware-attributed versus structural — read this before quoting anything

This is the one thing about this leg that must not be got wrong, and it is the
respect in which it is honestly weaker than its two siblings.

* Each bucket's **magnitude is hardware-attributed**. It is an `mcycle` delta,
  read by the core whose cycles it counts, on whichever side produced the
  artefact. Nothing is inferred, modelled or apportioned.
* Each bucket's **mechanism label is structural**. "L1 load-use interlock" is a
  claim about how the kernel was *built* — a zone of `lw rd, 0(rd)` over a
  1 KiB ring — and **not** a hardware counter's opinion. The hardware has no
  per-mechanism counter for a baby RISC-V at all.
* What makes the structural label **checkable rather than merely asserted** is
  `minstret`. The same binary must retire the same instructions in the same
  zone on both sides. `retire_census_matches` requires **exact equality** and
  refuses otherwise — a zone whose two sides retired different instruction
  counts did not run the same zone, whatever it is called.

A zone label is therefore a *hypothesis about which mechanism dominates*, with a
hardware-measured magnitude and a hardware-measured instruction count attached.
That is what ROADMAP §4's "by mechanism **and by zone**" admits, and it is less
than what the other two legs have. Say so when quoting this leg.

**The cycles-per-instruction table is not an independent second check.** Once
`retire_census_matches` has made the two sides' retired counts equal, CPI is the
partition divided by a per-zone constant and carries exactly the same
information. `nocevbench`'s per-class latency table *is* independent of its
partition; this one is not. It is reported because cycles-per-instruction is the
unit a RISC-V mechanism is naturally quoted in, and because it is directly
comparable to a `riscvbench` slope — not because it can catch anything the
partition cannot.

---

## The instrument, and the constraint that shaped the program

`mcycle` (0xb00) and `minstret` (0xb02) are Blackhole baby-RISC-V CSRs
(`BlackholeA0/TensixTile/BabyRISCV/CSRs.md`), modelled in
`tt_sim/pe/rv/isa/zicsr_isa.py`. `mcycle` reads the same tile clock that
`RISCV_DEBUG_REG_WALL_CLOCK_*` samples, so a cycle here is the same cycle every
other perfbench program counts in.

`cfg0` bit 10 `DisCsrSync` is documented: **while it is clear**, once a `csrr*`
instruction leaves the front end the next instruction does not leave until the
previous has retired. It is clear at reset and clear in tt-metal's own init —
the observed `cfg0 = 0x60008` sets only `DisLowCash`, `DisTriscCache` and
`StMergeTimer`. So on a real part:

> **A CSR read is a retirement barrier.**

Therefore the kernel reads the counters **around windows and never per
instruction**. Two reads bracketing thousands of instructions serialise only at
the boundaries and dilate nothing in between. Per-instruction bracketing is
retired as unreachable in ROADMAP §4 for exactly this reason; nothing here has a
fidelity that depends on a CSR read being free.

The `marker_null` zone is two marker pairs with nothing between them, so **the
instrument's own cost is a measured number in every artefact** rather than an
assumption in a comment. The real simulator run reads **3 cycles**; a card's will
be larger, because tt-sim charges a CSR instruction nothing — `DisCsrSync`'s
serialisation has no published cycle count and `tt_sim/pe/rv/cost.py` declines to
invent one. That is a real and *expected* disagreement, it is reported as a note
with its span share, and the calibration zone is excluded from the CPI grading
because grading a zone against a number the model deliberately does not have
would fail every honest run.

---

## Why the decomposition stops here

Per zone, the hardware gives **elapsed cycles and retired instructions**. That is
close to all of it, and two obvious ways to get more were looked at and refused:

* **`mhpmcounter3` / `mhpmcounter4`** exist, but the encodings of their
  `mhpmevent3` / `mhpmevent4` selectors are **unpublished**. No event can be
  given a meaning, so no count can be honest. `zicsr_isa.py` already refuses
  those counters once software has selected an event. **No event mapping was
  invented to manufacture more buckets**, and none may be.
* **There is no PC sampler and no instruction-trace buffer** in tt-metal 0.74,
  in UMD, or in the public ISA docs. The debug daisychain is documented "at
  least five cycles stale" and every consumer of it is commented out.

A coarse decomposition that is honest beats a fine one that is invented.

---

## Blackhole only, and it refuses rather than degrades

The string `csr` appears **zero times** in the whole `WormholeB0` doc tree. A
Wormhole baby core has no CSRs to read: tt-sim raises `NoCSRsError` on a CSR
instruction there, and the part has no `minstret` either. Without retired counts
the structural labels stop being checkable and the leg collapses to an
elapsed-only envelope check — which is precisely what rung 4 exists to distrust.

So there are **three refusals, at three different times**, and none of them is a
warning:

1. `run_card.sh`'s pre-flight asks the device which part it is, before the build.
2. `retirebench.cpp` declines a non-Blackhole part before it builds its kernel
   or launches anything, and prints why plus what to run instead.
3. `retire_attribution.py`'s `arch_supported` gate refuses a non-Blackhole
   artefact before it computes anything.

**Do not add a Wormhole fallback.** On a Wormhole part, run
[`riscvbench`](../riscvbench/README.md) instead: it times the same instruction
mixes off the wall clock and needs no CSRs.

---

## How this differs from `riscvbench`

`perfbench/riscvbench` is a **rung-3 front-end characterisation**: it measures
what an instruction *costs* on the card. This is a **rung-4 attribution
comparison**: it measures whether tt-sim spends a span the same way hardware
does. They are different instruments on the same core, and they are complements:

| | `riscvbench` (rung 3) | `retirebench` (rung 4) |
| --- | --- | --- |
| Output | per-instruction **slopes** | one **partition** of one span, and `E_total` / `E_int` |
| Method | four burst lengths, least-squares slope, so launch and timer overhead cancel **exactly** | one absolute window, cut into zones that telescope to it |
| Instrument | `RISCV_DEBUG_REG_WALL_CLOCK_L` | `mcycle` + **`minstret`** CSRs |
| Architectures | Wormhole **and** Blackhole | **Blackhole only**, and refuses elsewhere |
| Card side needed? | yes, it *is* the measurement | yes, it is one of the two sides being compared |
| Can a number become a cost? | yes, that is the point | **no** — this leg fits nothing |

The slope method cannot do rung 4's job: it cancels the constant term on purpose,
and a partition needs the absolute cycles including it. The absolute method
cannot do rung 3's job: it carries the loop overhead and the marker cost inside
every bucket. Hence two benchmarks.

`retirebench`'s zone bodies **are `riscvbench`'s phase R and phase C probe
bodies, unchanged**, deliberately: that benchmark measured each of them on
Blackhole silicon on 2026-08-05, so this one's repetition counts are sourced from
a measurement rather than guessed, and its CPI column is directly comparable to
that one's slopes.

---

## The zones

Twelve, on BRISC, run back to back inside one outer window. No Tensix
instruction is issued and no NoC transaction is started, so nothing but the
scalar core is in the measurement.

| zone | body | mechanism it is built to be dominated by |
| --- | --- | --- |
| `marker_null` | nothing | **the instrument**: the two-marker cost alone |
| `alu_dep` | 64 × dependent `addi` | dependent integer chain (the forwarding path) |
| `alu_ind` | 16 × 4 independent `addi` | independent integer ops (issue width) |
| `mul_dep` | 64 × dependent `mul` | multiply **result latency** |
| `mul_ind` | 16 × 4 independent `mul` | multiply **unit occupancy** |
| `div_small` | 16 × `divu`, 12-bit dividend | divide at the operand real kernels use |
| `div_large` | 16 × `divu`, 29-bit dividend | divide at the top of the documented band |
| `load_dep` | 64 × `lw rd, 0(rd)` over a 1 KiB ring | L1 load-use interlock |
| `load_ind` | 8 × 8 `lw` over 8 distinct lines | sustained L1 load throughput |
| `store_spread` | 16 × 4 `sw`, non-coalescable | L1 store throughput |
| `branch_nt` | 64 × (`xori` + never-taken branch) | not-taken conditional branch |
| `branch_t` | 64 × (`xori` + always-taken branch) | taken conditional branch |

The two branch zones execute the **identical dynamic instruction sequence** — one
`xori` and one conditional branch to a target one instruction ahead, which is
where the not-taken branch falls through to — and differ in exactly one bit:
whether the branch was taken. tt-sim charges neither, and says so: the mispredict
bubble is sourced (4 cycles on Blackhole) but the predictor is undocumented, so
the number of mispredictions is unknowable and charging every taken branch would
be a fabrication. This pair is what will put a number on what that omission
costs.

### Three sizing decisions, each of which the first run forced

**Every zone is sized to ~6000 cycles on a card**, from `riscvbench`'s silicon
per-instruction costs. Equal-sized zones are a requirement, not tidiness: `E_int`
and `E_total` are both denominated in the whole span, so a zone allowed to
dominate the span would let *its* error dominate both numbers and the partition
would stop being a decomposition.

**Why 6000 and not 2000**, which a 1000-cycle floor and a ~30-cycle marker cost
would otherwise suggest. `div_large` is the constraint, and it is the one zone
whose two sides are **known to disagree before the program runs**: the divide's
cost is a documented data dependence ("between six and 33 cycles … dependent upon
the magnitude of the dividend"), tt-sim charges the documented floor of 6 for
every operand, and `riscvbench` read 33.001 on silicon at a 29-bit dividend. The
zone must be long enough that the *simulator* side — the short one — still clears
the 1000-cycle floor, which puts the card side near 6300; and every other zone
must then be sized alongside it, or that one known gap would be most of `E_total`
and this leg would be a test of a thing already measured. At these sizes it is
~7 % of the span: **visible in the interior, not in command of the total.** That
tension is itself a statement about how far apart the two sides are on that one
mechanism, and it is stated rather than resolved.

**`load_ind` rotates through eight 16-byte lines, not four**, and the first
simulator run is why. Blackhole publishes an L0 data cache of "a mere 64 bytes:
4 lines of 16 bytes each". A load zone rotating through exactly four lines has a
working set of exactly the published capacity, so whether they stay resident
depends on associativity, indexing and replacement — none of which the docs
publish. The zone's own name would then rest on an unpublished property: the
first run read **1.001** cycles/instruction against the **1.742** `riscvbench`
read on silicon at the same four addresses, because tt-sim's L0 model (fully
associative, by its own documented choice of the generous reading) kept all four.
Eight lines exceed the published capacity under *any* organisation, and the zone
now reads 1.761 against silicon's 1.742. The resident case is deliberately **not**
given a zone of its own: a zone whose meaning depends on unpublished cache
organisation is a zone whose label cannot be defended.

---

## Running it

### Simulator side

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal
./perfbench/retirebench/run_sim.sh --out /tmp/retirebench-sim
```

Sets `TT_SIM_COST_MODEL=1`, pins worker `1-2`, puts the repo venv on `PATH` (the
simulator server is spawned by UMD and needs tt_sim's dependencies), and writes
`retirebench-blackhole-sim.json`. `--no-cost-model` exists to demonstrate what
goes wrong and is refused by the analysis, not merely warned about.

### Card side

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal
./perfbench/retirebench/run_card.sh --preflight     # costs no card time
./perfbench/retirebench/run_card.sh --out ~/retirebench-session
```

No Tracy, no device profiler, no `tt-exalens`, no board reset, no root. Budget
~2 min for the build and ~20 s per run. It sets `TT_METAL_SLOW_DISPATCH_MODE=1`,
because `detail::LaunchProgram` is the only dispatch flow tt-sim supports and a
card defaults to fast dispatch — four earlier card runners aborted `rc=134`
before doing any work for exactly that reason.

**`--scale` must match on both sides.** A comparison across two scales is a
comparison of two different programs, and `zone_table_matches` refuses it.

### Analysis

```bash
python3 -m tt_sim.perf.retire_attribution --decompose-only --sim <artefact>

python3 -m tt_sim.perf.retire_attribution \
    --sim  /tmp/retirebench-sim/retirebench-blackhole-sim.json \
    --card ~/retirebench-session/runs/card-1/retirebench-blackhole-card-1.json \
    --report report.txt --json report.json
```

Exit status is 0 on PASS, 1 on FAIL or on any refusal.

---

## The eight gates

Each refuses rather than warns, and each has a passing *and* a refusing test.

| gate | refuses |
| --- | --- |
| `arch_supported` | a non-Blackhole artefact on either side |
| `zone_table_matches` | different zones, reps, `--scale` or artefact version — two different programs |
| `counters_advanced` | a window or zone whose counters stood still (reads back as a plausible zero-cycle partition) |
| `per_core` | a cross-core or cross-RISC comparison, unless `--map-core` states it |
| `zone_budget` | > 60 zones, a zone under 1000 cycles, a measured marker cost over 6 % of the smallest zone, or a missing calibration zone |
| `cost_model_engaged` | a run where the pointer chase costs what an `addi` costs — the `TT_SIM_COST_MODEL`-unset fingerprint, which is otherwise a *well-formed* artefact |
| `partition_closes` | a negative bucket, or buckets that do not sum to the window |
| `retire_census_matches` | any per-zone retired-instruction difference at all |

A refusal **stops the analysis**: no `E_total`, no `E_int`, no comparison in the
JSON. An `E_int` computed over a partition that did not close is a number, but it
is not a measurement of anything, and once it is exported somebody quotes it.

---

## What is checked in, and what it shows

See [`testdata/README.md`](testdata/README.md) for the full recipe. In short:

* `sim-blackhole-scale1.json` — **a real simulator run.** 60359 cycles, 40915
  instructions retired, two-marker cost 3, `unattributed` 107.
* `card-agreeing-…` — **synthetic.** `E_total = 0.91 %`, `E_int = 1.15 %`,
  compensation **1.26×**. **PASSES.**
* `card-compensating-…` — **synthetic.** The same total, `E_total = 0.91 %`, with
  13 500 cycles moved between six zones. `E_int = 45.48 %`, compensation
  **49.91×**. **FAILS**, having passed every gate.

**That last pair is the leg's argument.** The two spans agree to 0.91 % — inside
the 10 % criterion here, inside nekbone's ± 10.2 %, inside every envelope
threshold in this repo — and the decomposition behind the agreement is 45 %
wrong. A total cannot see it.

**There is no card session for this leg yet.** The card runner is written and
pre-flighted; nothing in this directory claims a silicon result.

---

## Provenance, and what may not happen to these numbers

`isa_doc` for the instrument, and **inert**. No number this leg reads, computes
or prints may become a cost. Nothing here writes to `unit_costs.yaml` or any
other cost table, and `costs_test.py` asserts the ladder holds
(`isa_doc > isa_doc_derived > vendor_source > vendor_source_derived`;
`estimated` is forbidden).

**This leg fits nothing.** A disagreement is the result, never a reason to tune
the simulator. If a card session comes back and `div_large` is 5.5× short, that
is a measurement of a documented one-sided floor, not a licence to pick a number
between 6 and 33 — the docs give the band and no function within it, and
`tt_sim/pe/rv/cost.py` has already recorded twice why an invented curve wearing a
citation is worse than a floor. Silicon is corroboration, never provenance.
