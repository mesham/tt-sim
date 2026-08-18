# retirebench testdata

Three artefacts. **One is a measurement and two are not**, and the filenames say
which is which because a filename is the only thing that travels with a file
somebody copies out of the tree.

| file | what it is |
| --- | --- |
| `sim-blackhole-scale1.json` | **A real run.** `run_sim.sh` against the tt-sim Blackhole simulator, `--scale 1`, `TT_SIM_COST_MODEL=1`, worker `1-2`, 2026-08-18. |
| `card-agreeing-SYNTHETIC-NOT-A-MEASUREMENT.json` | **Synthetic.** Hand-derived from the file above. No card has run this program. |
| `card-compensating-SYNTHETIC-NOT-A-MEASUREMENT.json` | **Synthetic.** Likewise. |

There is **no card session for this leg yet**. The two "card" files exist so that
`tt_sim/perf/retire_attribution_test.py` can exercise the criterion in both
directions — a case that must PASS and a case that must FAIL — and so that the
compensation ratio has a worked example attached to it. They carry
`"label": "...-SYNTHETIC"` inside as well as in the filename, and a test asserts
both, so a file that loses its name still says what it is.

## The real run

```
zone               cycles    retired  cyc/instr   mechanism (structural)
marker_null             3          1      3.000   the two-marker cost alone
alu_dep              6210       6208      1.000   dependent integer chain
alu_ind              6400       6398      1.000   independent integer ops
mul_dep              6069       3106      1.954   multiply result latency
mul_ind              6398       6396      1.000   multiply unit occupancy
div_small            3049        567      5.377   divide, 12-bit dividend
div_large            1286        244      5.270   divide, 29-bit dividend
load_dep             6144        795      7.728   L1 load-use interlock
load_ind             6377       3621      1.761   sustained L1 load throughput
store_spread         6085       1259      4.833   L1 store throughput
branch_nt            6116       6114      1.000   not-taken branch
branch_t             6115       6113      1.000   taken branch
unattributed          107         93          -
window              60359      40915
```

Four of those columns can be checked against `perfbench/riscvbench`'s
**Blackhole silicon** run of 2026-08-05 without leaving the repo, because the
zone bodies are that benchmark's probe bodies unchanged: `mul_dep` 1.954 against
its 1.985, `mul_ind` 1.000 against 0.999, `load_dep` 7.728 against 8.098,
`load_ind` 1.761 against 1.742. That is corroboration of the *instrument*, not a
result of this leg — the leg's result is `E_total`, `E_int` and their ratio
against a card running this same binary, and no card has.

The run is **deterministic**: it was captured twice against the simulator, an
hour and one unrelated Tensix `WaitGate` change apart, and the two artefacts are
byte-identical apart from the `label` field — same cycles, same retired counts,
same `unattributed`. That is worth recording, because it is the property
`run_card.sh` asks the operator to check on the card side: the program has no
data-dependent control flow, so the `retired` column must be **identical** across
repeats, not merely close, and a `retired` column that moves is a statement about
the instrument rather than about the part.

## How the two synthetic files were made

Both are the simulator run with **only the `cycles` column edited**. Every
`retired` count, the zone table, the reps, the scale, the core and the RISC are
untouched, so both pass `zone_table_matches` and `retire_census_matches` — they
are the *same program*, differing only in how long each zone took.

**`card-agreeing`** — every zone within a few per cent of the simulator, plus one
edit with a reason rather than a shape behind it: `marker_null` 3 → **30**.
tt-sim charges a CSR instruction nothing at all (`cfg0`'s `DisCsrSync`
serialisation has no published cycle count and `tt_sim/pe/rv/cost.py` declines to
invent one), so a card's marker pair *must* cost more than the simulator's 3
cycles. That edit is what exercises `marker_notes`.

**`card-compensating`** — the agreeing file with **4500 cycles moved out of each
of three zones and into three others**: `alu_ind → load_dep`,
`mul_ind → store_spread`, `load_ind → mul_dep`. The window total is untouched, so
`E_total` is the agreeing file's and only the interior moved.

```
card-agreeing:      E_total = 0.91 %, E_int =  1.15 %, ratio =  1.26x  -> PASS
card-compensating:  E_total = 0.91 %, E_int = 45.48 %, ratio = 49.91x  -> FAIL
```

**The second line is the leg's whole argument.** The two spans agree to 0.91 %,
which is inside every envelope threshold in this repo — the 10 % criterion here,
nekbone's ±10.2 %, the ~1 % component slopes — and the decomposition behind that
agreement is 45 % wrong, in compensating directions. A total cannot see it. The
ratio is what a passing total cannot fake.

The three donor zones were chosen because they are the largest, and 4500 is the
largest round move that leaves each of them above the 1000-cycle floor
`zone_budget` enforces: the compensating file has to pass **every gate** and fail
**only** the criterion, or it would demonstrate something about the gates instead.

## Rebuilding them

They are derived files, so if the simulator run is ever recaptured they must be
rebuilt from it rather than kept. The recipe above is complete: copy the new
simulator artefact, apply the listed `cycles` edits, and recompute
`window.cycles` as the sum of the zones plus the unchanged `unattributed`
remainder. The tests pin the numbers that matter, so a rebuild that changes the
argument will fail rather than pass quietly.
