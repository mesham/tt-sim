# mechbench testdata

Four device-profiler logs, guarded by `tt_sim/perf/stall_attribution_test.py`.
Two are real simulator output; **two are synthetic and are not measurements of
anything.**

| file | what it is |
| --- | --- |
| `sim-elw-blackhole.csv` | **Real.** tt-sim Blackhole, `mechbench elw 8`, `TT_METAL_PROFILE_PERF_COUNTERS=32`, 2026-08-13. The whole file as tt-metal wrote it, zone markers included, so the loader is exercised on rows it must skip. |
| `sim-mm-blackhole.csv` | **Real.** The same, `mechbench mm 8`. |
| `card-elw-agreeing-SYNTHETIC-NOT-A-MEASUREMENT.csv` | **Synthetic.** Hand-built to look like a plausible card ±5 % of the `elw` simulator run. Exists to prove the comparison can pass. |
| `card-elw-compensating-SYNTHETIC-NOT-A-MEASUREMENT.csv` | **Synthetic.** The same span to within 2.9 %, with the `srca_valid` mass moved into `unattributed_stall`. Exists to prove the comparison can fail on an interior that every envelope check in this repo would wave through. |

Both real logs were captured **before** the profiler-readback race was closed
(`Device.settle_profiler_flush`, 2026-08-13), in the only regime that then
produced a log at all. They therefore carry **BRISC's rows only** — the counter
samples this module reads are all BRISC's, so nothing the tests assert is
affected, but the NCRISC and TRISC zone markers a current run produces are
absent. A recapture would change every pinned constant in
`stall_attribution_test.py` and both synthetic card files derived from them, so
they are left as they are and their shape is stated here instead. **Do not
treat them as a picture of what a run emits today.**

## The synthetic files are not data

**No card session for this leg exists.** Nobody has run `mechbench` on silicon.
These two files were written by hand so that the passing and failing paths
through `tt_sim.perf.stall_attribution` are both exercised and both guarded, and
so that the compensation ratio has a worked example attached to it. They carry
`SYNTHETIC-NOT-A-MEASUREMENT` in their filenames precisely because a CSV full of
plausible integers is the easiest thing in this repo to mistake for a
measurement six months from now.

Nothing in them may be quoted as a property of any Tenstorrent part. When a real
session arrives it goes under `perfbench/card-sessions/<date>/` as
`corroboration` — the counter semantics come from a vendor tech report and RTL
rather than the ISA documentation, which is fine for corroboration and
**disqualifying as provenance**.

## What they produce

```
card-elw-agreeing:      E_total = 4.39 %, E_int =  6.31 %, ratio =  1.44x  -> PASS
card-elw-compensating:  E_total = 2.92 %, E_int = 39.33 %, ratio = 13.46x  -> FAIL
```

The second is the argument for this whole leg in one line: its total is well
inside the 10 % envelope limit and its interior is off by 39 %. Note also that
its three *per-thread* comparisons all pass at 2.92 % — the compensation is only
visible in the core-level partition, because that is the only place the four Src
conditions are decomposed. A per-thread-only criterion would have missed it
entirely.
