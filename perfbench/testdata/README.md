# testdata — fixtures for `card_session_verdicts_test.sh`

**Some of these files are broken on purpose.** They are what a real Blackhole
card returned on 2026-08-09, kept unmodified so the verdict checks can prove
they would catch the same failures again. Nothing here is data to *use* — the
usable CSVs from that session are in
[`../../tt_sim/perf/datasets/`](../../tt_sim/perf/datasets/), named
`*-2026-08-09*.csv`, and the analysis is in `docs/plans/cost-model.md`,
"The second rung-3 sample".

Run the checks with:

```bash
perfbench/card_session_verdicts_test.sh     # 32 checks, no card needed
```

Delete any file below and checks stop testing anything — the suite drops to
23 passed / 9 failed.

## `card-session-blackhole-2026-08-09/` — the first run

The session that awarded **three** `MEANINGFUL` verdicts to probes whose
controls had not moved. Each file pins one of them.

| file | why it is kept |
| --- | --- |
| `nocread.blackhole.csv` | `outstanding_max` reads **72 in all 129 rows**, including bursts of **four** requests. A count of requests in flight cannot exceed the burst that produced it: the probe was reading a live register with no baseline subtracted. The session called it *"no initiator limit — this RETIRES the credit-limit term"*. |
| `nocread.out` | the benchmark's own verdict line for that CSV. |
| `noc.report.txt` | **intentionally a Python traceback.** One line of `ModuleNotFoundError` — the report generator ran on a card box with no `tt_sim/`. |
| `noc-epoch.report.txt` | **intentionally a Python traceback**, same cause. The old logic grepped it for the *absence* of a failure string, found none, and concluded *"no reproducing per-core epoch, which retires the (11,2) observation"* — a scientific claim drawn from an error message. |
| `rv.out` | phase Q failed 16 checks, **all at n ≤ 16** — the small-burst scatter `riscvbench/README.md` documents on silicon. Pins the exemption as a threshold, not a blanket suppression. |
| `rv-cross.out` | the `--blocks 8` cross-check, which failed five phases. Pins why a cross-check must run at the **same** parameters as the primary. |

## `card-session-blackhole-2026-08-09-run2/` — the re-run

The kernel fix worked here: `inflight_rest` is 0 in every row and the two
in-flight instruments agree. The benchmark then printed **`BOUNDED AT 13`** —
and it is still not a credit limit. 13 comes from three rows at one transaction
size (4096 B) while every other row reads 2–3, and `cycles_per_tx` is flat to
1.5 % across 13 hops. A credit limit `K` caps the rate at `round_trip/K`, which
*must* rise with distance.

| file | why it is kept |
| --- | --- |
| `nocread.blackhole.csv` | the `dist` sweep that refutes the bound, and the `outstanding_rest ≈ 68` baseline that shows why a raw-column check would reject every healthy run. |
| `nocread.out` | the `BOUNDED AT 13` verdict line, which the checks must now overrule. |

## The rule these fixtures exist to enforce

`MEANINGFUL` is only ever awarded for a control that **moved**. A value being
present, being in the right format, not being a known sentinel, or a failure
string being absent from a file are none of them evidence. Every false positive
above was one of those four standing in for a real check.
