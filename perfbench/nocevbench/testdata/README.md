# nocevbench testdata

Four NoC event traces. **Two are measurements. Two are not.**

| file | what it is |
| --- | --- |
| `sim-blackhole-4096.json` | **Real.** `nocevbench 4096 8` against tt-sim's Blackhole, 2026-08-16, `TT_SIM_COST_MODEL=1`, `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1`, tt-metal 0.74. Unmodified. |
| `sim-gemm256-16core-brisc-EXTRACT.json` | **Real, and a reduced extract.** See below. |
| `card-agreeing-SYNTHETIC-NOT-A-MEASUREMENT.json` | **Synthetic.** No card session for this leg exists. |
| `card-compensating-SYNTHETIC-NOT-A-MEASUREMENT.json` | **Synthetic.** Likewise. |

## `sim-gemm256-16core-brisc-EXTRACT.json`

**A reduced extract, and the reduction is the only thing done to it.** The
source is the compiler team's 16-core Blackhole `gemm_256_check` capture of
2026-08-19 (the pre-slab half), 2.94 MB and 11,872 records across 32 streams.
Checking that in whole is not worth 2.94 MB for what it guards, so this file is
the **140 records of one stream** — BRISC on core `(1, 2)` — copied out
verbatim, in their original order, with no field altered and nothing rescaled.
Every number it yields is a real number from that run.

It is here because it is the input this leg used to **decline**. All 32 of that
capture's streams were refused as `single_window`, on the grounds of "2 to 5
`ZONE_START`, expected exactly one" — which was the device profiler's own
`PROFILER-NOC-QUICK-PUSH` flush zone nesting inside the kernel zone, not a
second launch. This stream carries exactly that: one `BRISC-KERNEL` pair
spanning 60,754 cycles, and one flush inside it costing 183 cycles.

The two synthetic files are stamped `NOT-A-MEASUREMENT` in their filenames, and
nothing in this repo may quote a number from them as a property of hardware.
They exist for one reason: a guard that cannot fail is as damaging as one that
cannot pass, so the criterion needs an input it must accept and an input it must
refuse.

Both are derived from `sim-blackhole-4096.json` by rescaling *inter-event
deltas* and re-accumulating. Every record, every field, every event ordering is
preserved, so both carry an **identical transaction census** to the simulator
run — which is what makes the FAIL below a finding about timing rather than
about two sides having run different programs.

* **`card-agreeing`** — every interval 5 % longer. `E_total` and `E_int` both
  ~4.8 %, every latency class ~4.8 % out. **PASSES.**
* **`card-compensating`** — the agreeing card, with 2 500 cycles moved out of
  BRISC's `SEMAPHORE_WAIT` interval and into its trailing `local` interval.
  Every event after the semaphore shifts earlier by the same amount, so **every
  inter-event distance among the NoC events is untouched**. The result:
  `E_total` **4.78 %** — comfortably inside the limit — against `E_int`
  **48.33 %**, compensation ratio **10.12x**, and *every per-class latency
  passes at 4.8 %*. **FAILS, on the partition alone.**

The second file is this leg's argument in one artefact. An envelope check passes
it. A per-transaction latency check passes it. Only the decomposition sees it.

Note what the construction also shows, and what is stated in the README rather
than hidden: the `read_wait` / `write_wait` buckets and the per-class latencies
are **not independent evidence** — they are two readings of the same clock
intervals. The independent part of the partition is `prologue`, `issue`,
`other_wait` and `local`, and that is exactly where the compensating file puts
its error.

Regenerating them is a deliberate act, not a build step: the transform is ~40
lines and is described above precisely enough to rewrite. If the simulator
artefact is ever recaptured, the two synthetic files must be regenerated from it
in the same commit, or the census gate will refuse the pair.
