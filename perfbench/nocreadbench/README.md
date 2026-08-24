# nocreadbench — what limits the sustained NoC *read* rate on a card

You have a Tenstorrent card. This asks it one question that neither the public
ISA documentation nor tt-metal's headers answer, and it asks it two ways: once
by reading a hardware counter, and once by the shape of a sweep.

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal   # built with ./build_metal.sh
cd perfbench/nocreadbench/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

../run_card.sh                    # the E0–E7 sweep, ~1–3 minutes
# -> nocreadbench-<arch>.csv   <- send this back, with the console output

../run_card.sh --preflight        # costs no card time
../run_card.sh --arms             # the STATEFUL comparison, ~3 minutes
# -> nocread-arms-session/     <- send the whole directory back

../run_card.sh --dram             # the DRAM BURST arm, ~3 minutes
# -> nocread-dram-session/     <- send the whole directory back
```

The `--arms` session has pre-registered predictions: see
[The stateful variant](#the-stateful-variant-and-the-two-predictions). The
`--dram` session is the newest question, has **absolute** pre-registered
predictions on both architectures, and is the one an outside team asked for:
see [The DRAM burst arm](#the-dram-burst-arm-what-does-one-more-batched-dram-read-cost).

**Both sessions refuse to run with `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` set**,
in the runner and again in the program. That switch puts a profiler write inside
the timed loop — +10–19 % of span on silicon — and it does not tax the arms
equally, which is the one failure mode these measurements cannot survive.

Run it on **both** a Wormhole part and a Blackhole part if you have both. The
whole reason this exists is a per-architecture difference, and one part cannot
show it.

Nothing here writes to a card's flash, changes any clock, or needs root. It
allocates L1 scratch, runs one data-movement kernel on one Tensix core, and
reads back cycle counts and two NIU registers.

## The question

`tt-metal` ships 740 rows of measured NoC latency
(`tt_metal/impl/experimental/noc_estimator/latencies/noc_latencies.yaml`).
Differenced along its transactions-per-barrier axis, a pipelined burst of
64 B L1 reads costs:

| | Wormhole | Blackhole |
| --- | --- | --- |
| N = 4 → 16 | **19.00** cycles/transaction | 34.9 |
| N = 64 → 256 | **25.00** | **35.0** |
| the same, `stateful` (a shorter issue loop) | **17.33** | **34.0** |
| tt-sim's reconstruction of the issue loop | 18 | 19 |
| **this program, on a real Wormhole part, 2026-08-17** | **44.0** | not yet run |

Two things in that table need explaining and the dataset cannot explain either.

1. **Blackhole reads have a floor the issue loop cannot get under.** Removing
   several instructions per iteration (the `stateful` row) buys 1.0 cycles. So
   something other than the instruction stream is setting the rate, and it is
   worth ~15 cycles per read.
2. **Wormhole's is not the same shape.** A shorter loop buys 7.67 cycles and
   lands *below* the 25 that the longer loop sustains — so there is no
   25-cycle-per-read hardware floor on Wormhole at all.

The roadmap named the Blackhole excess "the initiator's outstanding-read-request
credit limit". That is a hypothesis, and it is one of at least four. This
program is how you tell them apart.

## What a real Wormhole part actually said, and what it left ambiguous

On **2026-08-17** this program ran on a Wormhole part
(`perfbench/card-sessions/2026-08-17-wormhole-B/`). The `burst` control measured
a marginal **44.0 cycles per transaction**: 44.08 at N = 4 → 16, 44.00 at
16 → 64, 43.97 at 64 → 128 — flat to 0.25 % across a 32× range in burst length,
and flat to 1.0 % across hops 1..7, which is also what refuted the credit-limit
hypothesis on that part. Every experiment's *average* landed near 46.8.

That is **1.76× the 25.00 the shipped dataset carries for the same shape**. The
disagreement is not noise, is not an average-versus-marginal confusion, and is
not distance-dependent. It is also not, by itself, a statement about the part:
the vendor's dataset was taken on a different part with a different program, so
`noc_latencies.yaml`'s 25.00 is **not the published figure for the part under
test**, and the row above is not "the control failed".

Two candidates survive the flatness, and the dataset cannot separate them:

- **H-LOOP.** 44 is *our issue loop*. The rate is set by the instruction stream
  on the issuing baby RISC-V, ours is longer than the dataset's, and the number
  says nothing about the silicon.
- **H-FLOOR.** 44 is a per-read cost in the part that the dataset says is not
  there.

## The stateful variant, and the two predictions

The dataset's own `stateful` rows are the discriminator, because they are the
same experiment with a **shorter issue loop**: on Wormhole they buy 7.67 cycles
(25.00 → 17.33) and land *below* the long loop's figure, which is that dataset's
evidence that Wormhole has no per-read floor; on Blackhole they buy 1.0
(35.0 → 34.0), which is its evidence that Blackhole does. `--stateful` runs that
loop here.

### What it removes from the loop, and where that is written down

The stateless arm calls `noc_async_read`, which for these sizes reaches
`ncrisc_noc_fast_read`
(`tt_metal/hw/inc/internal/tt-1xx/wormhole/noc_nonblocking_api.h:415`). Per
transaction it writes five command-buffer registers — `NOC_RET_ADDR_LO`,
`NOC_TARG_ADDR_LO`, `NOC_TARG_ADDR_COORDINATE`, `NOC_AT_LEN_BE`, `NOC_CMD_CTRL`
— and extracts the coordinate from a 64-bit NoC address every time
(`(uint32_t)(src_addr >> NOC_ADDR_COORD_SHIFT)`, same function). Blackhole's
equivalent writes `NOC_TARG_ADDR_MID` as well: six.

The stateful arm calls `noc_async_read_set_state`
(`tt_metal/hw/inc/api/dataflow/dataflow_api.h:673`) once, which reaches
`ncrisc_noc_read_set_state` (`wormhole/noc_nonblocking_api.h:1117`,
`blackhole/noc_nonblocking_api.h:1321`) and writes **only** the coordinate — plus
`NOC_TARG_ADDR_MID` on Blackhole. Then per transaction it calls
`noc_async_read_with_state` (`dataflow_api.h:708`) →
`ncrisc_noc_read_with_state` (`wormhole:1163`, `blackhole:1369`), which writes
`NOC_RET_ADDR_LO`, `NOC_TARG_ADDR_LO`, `NOC_AT_LEN_BE`, `NOC_CMD_CTRL` and
nothing else.

So the removal is exactly: **one command-buffer store per transaction on
Wormhole, two on Blackhole**, plus the 64-bit address arithmetic that fed them —
the stateful loop deals only in 32-bit local addresses. Both arms keep the
`while (!noc_cmd_buf_ready(...))` poll, which is the loop's six-cycle load-use
interlock, and both keep the closing barrier.

This is the same pair tt-metal's own estimator kernel switches on for the
`stateful` rows:
`tests/tt_metal/tt_metal/data_movement/noc_estimator_tests/kernels/reader.cpp`,
`if constexpr (stateful)`, through `Noc::set_async_read_state` /
`Noc::async_read_with_state` (`tt_metal/hw/inc/api/dataflow/noc.h:252`), whose
default `max_page_size` puts it on those same any-length entry points.

### The predictions — written here before any card ran the variant

| | if **H-LOOP** | if **H-FLOOR** |
| --- | --- | --- |
| the stateful marginal | **falls** by roughly the removed instructions, to **≤ 30 cycles/tx** — inside the 15–30 band the shipped dataset's Wormhole rows occupy | **barely moves**: within 10 % of the stateless arm |
| what the 44 then is | a property of *this program's* instruction stream, and not evidence about the part | a per-read cost in the part, worth ~40 cycles that no instruction removes |

`check_mode.py` prints which one the numbers landed on, and can print
**NEITHER** — a drop too large to call unmoved that still lands outside the
dataset's band. That is a real outcome, not a failure, and it is written down so
that it cannot be quietly reclassified afterwards.

### The stateless arm is the control, and that is checked

`--stateful` changes the issue call and nothing else; the two arms are one kernel
body compiled twice (`if constexpr` on the mode). The default arm therefore has
to be the program the 2026-08-17 session ran, and `check_mode.py` compares its
marginals against that session's 44.08 / 44.00 / 43.97 with a 3 % tolerance and
**fails the run** if they moved. On a part with no recorded session it says so
instead of inventing a comparison, and on a simulator run (`sim=1` in the CSV
header) it applies no card control at all.

One honest caveat, and NRB4 repeated it: the shared kernel body moved the
stateless loop's **constant** term by a few cycles against NRB2 — measured on
tt-sim as **+4 cycles on Wormhole and +1 on Blackhole** — and NRB4's source
witness moved it again, by **−5 on Wormhole and +2 on Blackhole**. The
*per-transaction* cost is bit-identical across all three layouts: **38.00
(Wormhole) and 41.00 (Blackhole) stateless, 23.00 stateful on both**, at all
three burst intervals, cost model off. That is the control check the card cannot
do at home, and it is why the source witness is an L1 load outside the timed
region rather than a probe transaction inside it. That is register
allocation around a shared loop body, not a change in the loop. The marginal is
what the arms are compared on and what the control is checked on, precisely
because differencing the burst axis removes that constant; the whole-file
average, which does not remove it, is printed as informational, and at N = 64 the
shift moves it by 4/64 = 0.06 cycles.

The alternative was to duplicate the loop so the stateless arm could be
byte-identical. That trades away the property the experiment actually needs — the
two arms differing *only* in the issue call, with the same stride bookkeeping,
the same barrier and the same sampling — in exchange for agreeing with a
historical constant. The wrong trade, and it is recorded here rather than left to
be discovered.

### How a run proves its own variant

Not from the flag. `--stateful` on a stale binary, a JIT cache that kept the old
kernel, or a shell that dropped the argument all produce a well-formed CSV whose
rate is the stateless loop's — and that reading is **exactly what H-FLOOR
predicts**, so it would be read as a result.

So the mode is read out of returned payload. The host stamps a per-tile
signature into every participating core's source region. After all its bursts,
the kernel points the read state at a **witness** core — logical (1, 1), which is
never the initiator (0, 0) and never a source, since every source in the plan
sits on row 0 or column 0 — and issues **one transaction through the same API
call its timed loop used**, addressed at this point's real source:

- the **stateless** call rewrites `NOC_TARG_ADDR_COORDINATE`, so the **source**
  answers;
- the **stateful** call never writes it, so the **witness** answers.

The landed signature word is the `probe_word` column, beside `sig_src` and
`sig_witness`, so a row is `stateless` iff `probe_word == sig_src` and `stateful`
iff `probe_word == sig_witness`. The kernel also stamps the mode it actually
ran, including a `refused` value the host cannot ask for — a stateful run over
more than one source tile, which the stateful path cannot express because the
source tile *is* the state. The host refuses any point whose probe disagrees
with the mode requested, exits non-zero, and prints `MODE NOT CONFIRMED`;
`check_mode.py` re-derives the same thing from the CSV alone, on the card, with
nothing but the standard library.

## The DRAM burst arm: what does one more batched DRAM read cost?

Everything above reads **64 B out of another worker's L1**, where the issue loop
dominates and the endpoint is idle. That is where this program has been
validated — on Wormhole 44.00/29.00 predicted against 45.03/28.96 measured, on
Blackhole 47.00/29.00/18.00 predicted and measured exactly — and it is not the
cell anyone is now stuck on.

The empty cell is **tile-sized payloads out of DRAM, batched**: several
transactions issued back to back before one barrier. tt-sim charges the marginal
of that batch **pure bandwidth** — link occupancy plus the DRAM channel's excess,
and nothing else. Endpoint queueing, outstanding-transaction credits and response
reordering are charged **exactly zero, by construction**, which is pinned in the
simulator's own suite as
`tt_sim/network/noc_cost_model_test.py::test_an_extra_batched_read_costs_bandwidth_and_nothing_else`.

Nobody has checked that against silicon, and **the vendor campaign structurally
cannot**: every DRAM row in tt-metal's `tm_noc_latencies` is one transaction per
barrier, so the batching axis does not exist in the dataset at any size.

It is not an idle question. A consuming team's optimised variant batches its DRAM
reads and writes; tt-sim predicts it **wins** its first pass by 1.05–1.12×, and
silicon has it **losing** by 7–9 %. Every term tt-sim charges is a real published
floor, so no single total there is wrong — but a floor bounds a number, not a
comparison, and this is the axis on which the two dataflows are told apart.

### What `--dram` adds, and what it deliberately leaves alone

Three experiments, and the default plan is otherwise **byte-for-byte** the one the
2026-08-17 session ran, so its control still reproduces:

| experiment | source | payload | N |
| --- | --- | --- | --- |
| `dramburst` | a DRAM tile | 2048 B — one bfloat16 tile | 4, 16, 64, 128 |
| `dramsize` | a DRAM tile | 512, 1024, 4096 B | 4, 16, 64, 128 |
| `bigburst` | a worker's L1 | 512, 1024, 2048, 4096 B | 4, 16, 64, 128 |

`bigburst` is not padding. Without it, *"the source was DRAM"* and *"the payload
was 32× the 64 B every other arm uses"* are perfectly confounded — the same
mistake E2 exists to fix on the responder axis, applied to the endpoint one.

The issue loop, the barrier, the stride bookkeeping and the sampling are the ones
every other arm uses; only the address and the coordinate change. And the marginal
is read by **differencing the same burst axis**, which removes the loop's constant
term and leaves the cost of one more transaction in flight.

### The predictions — absolute, and written here before any card ran the arm

Generated by running **this program** against tt-sim with the cost model on
(`TT_SIM_COST_MODEL=1`) and differencing the same burst axis the card run is
differenced along, so the two sides are the same arithmetic on the same
experiment rather than a model quantity compared against a measurement. They live
in `check_mode.py`'s `SIM_DRAM_PREDICTION`, which is both what the runner prints
**before** the card and what grades the run afterwards — so the two cannot drift
apart. The checked-in artefacts are `src/nocreadbench-wormhole-sim-dram.csv` and
`src/nocreadbench-blackhole-sim-dram.csv`.

Marginal cycles per transaction:

| payload | WH DRAM | WH L1 | BH DRAM | BH L1 | what binds |
| --- | --- | --- | --- | --- | --- |
| 512 B | **44.00** | 44.00 | **47.00** | 47.00 | the issue loop, both arms |
| 1024 B | **44.00** | 44.00 | **47.00** | 47.00 | the issue loop, both arms |
| 2048 B | **86.00** | 64.00 | **47.00** | 47.00 | WH: the transfer. BH: still the loop |
| 4096 B | **171.00** | 128.00 | **87.00** | 64.00 | the transfer |

Read the structure, not just the numbers. Two terms compete and the larger wins:

- the **issue loop** — 44 cycles/transaction on Wormhole, 47 on Blackhole,
  independent of payload size. Both are already validated against silicon, so
  they are the part of this table least at risk.
- the **transfer** — payload over the sustained rate. An L1 read is charged the
  NoC link (32 B/cycle Wormhole, 64 Blackhole): 2048/32 = 64, 4096/64 = 64. A
  DRAM read is charged that plus the channel's excess: 86 − 64 = **22** cycles at
  2 KB on Wormhole, 171 − 128 = **43** at 4 KB, 87 − 64 = **23** at 4 KB on
  Blackhole. Across four sizes that is 23.3–24.0 B/cycle on Wormhole and
  47.1 on Blackhole.

So **the crossover sits at a different payload on each architecture** — 2048 B on
Wormhole, 4096 B on Blackhole — and that is itself a registered claim.

### The three outcomes, and one of them is a loss

`check_mode.py --dram` prints which one the numbers landed on. It cannot express
"somewhere in between" as a pass.

| | what it means |
| --- | --- |
| **PREDICTION 1 — bandwidth and nothing else** | every DRAM marginal within 10 % of its figure. Charging endpoint queueing zero is costing nothing at these sizes on this part. |
| **PREDICTION 2 — a DRAM-side cost the model does not charge** | the L1 arm reproduces and the DRAM arm sits above. The excess is then *sized*; its shape says which term is missing — roughly constant in payload is a per-transaction endpoint cost, growing with payload is a bandwidth misestimate. |
| **CONTROL MOVED** | the L1 arm missed its own figure too. The instruction-level model both arms rest on is wrong on this part, the pair is unreadable, and the honest report is two numbers with no mechanism attached. |

**The consuming team's own prediction, registered as theirs:** the card's
marginal batched 2 KB DRAM read lands *meaningfully above* 86.

### The sharpest falsifier needs none of the absolute numbers

**Below the crossover, tt-sim says the two arms are the same number.** At 512 B
and 1024 B on Wormhole — and at 2048 B as well on Blackhole — the issue loop is
slower than either transfer, so the endpoint costs nothing extra either way and
the model predicts DRAM − L1 = **0.00**.

A gap there cannot be absorbed by any bandwidth term, however wrong the bandwidth
term is. It would be a per-transaction cost at the DRAM endpoint: precisely the
term charged at zero. `check_mode.py` prints that difference per size and calls it
out separately from the totals, so it survives a run where both absolute figures
happen to land inside tolerance.

### Plain instrumentation, enforced twice

`TT_METAL_DEVICE_PROFILER_NOC_EVENTS` force-enables the device profiler and
injects `-DPROFILE_NOC_EVENTS=1` into every kernel compile, so the issuing core
writes an 8-byte record per transaction **inside the loop this program times**.
Measured on silicon that tax runs +10–19 % of span, and the size of it is not the
problem: it does not fall evenly. It inflated an unbatched variant more than a
batched one, turning a real 7–9 % deficit into 1.00–1.03 and flipping it to 0.969
at the largest size. The effect the DRAM arm measures is the one it masks.

So the program **refuses to run** with it set — no override flag — and
`run_card.sh` refuses before that, in every mode. Every CSV carries
`noc_events=0` in its header, and `check_mode.py` refuses a file that lacks the
field, so a file from a binary predating the gate cannot be quietly graded.

### How a DRAM run proves it read DRAM

Not from the flag, and for a sharper reason than the mode arm has: **a `--dram`
run that in fact read a worker's L1 reproduces the prediction perfectly**, because
below the crossover the prediction for the two arms is the same number. "The card
agrees with tt-sim" is exactly what the broken run prints.

So the arm is read out of returned payload. The host stamps every source region
with a signature naming its tile, and DRAM tiles get one whose **high half
differs** — `0xD4A5....` against a worker's `0x5A5A....`. The kernel then reads
back, with a plain L1 load and no transaction of its own, the first word its
**timed burst** landed:

- a row is a **DRAM read** iff `landed == sig_src` and `sig_src` is in the
  `0xD4A5` half;
- a row is an **L1 read** iff the same holds in the `0x5A5A` half.

That word can only have been put there by the burst that was measured, which is
stronger than a separate probe. `landed_ok` is **−1** where the landing address is
walked (`dstspread`, `srcspread`) or the sources cycled (`srcfan`) and the word
therefore names no single tile — that is *not looked at*, not *verified zero*, and
the checker declines to speak rather than inventing a gate that cannot fail. The
host refuses any attributable point that fails, exits non-zero, prints
`SOURCE NOT CONFIRMED`, and `check_mode.py` re-derives the whole thing from the
CSV alone.

### Congruence, and why the pad is on the L1 side

A DRAM→L1 read must satisfy `(src % n) == (dst % n)` with **n = 32 on Wormhole
and 64 on Blackhole**. It is a **congruence** rule, not an absolute alignment:
`WormholeB0/NoC/Alignment.md`'s table for this path carries only congruence codes
and no absolute source or destination alignment, so an absolute check would be
stricter than the hardware and would refuse correct kernels. Violations are
`UndefinedBehavior` — bytes skewed or dropped, **nothing faults** — which here
would present as a *rate* rather than as an error.

The DRAM buffer and the L1 arena come from two different allocators with two
different alignments, so congruence is not automatic. The host computes the
shortfall and shifts the **landing** side by up to n−1 bytes
(`NOCREADBENCH_A_DST_PAD`), which leaves every DRAM address exactly where its
allocator put it — keeping the host's own `WriteToDeviceDRAMChannel` aligned — and
costs a few dozen bytes of an arena with kibibytes of headroom. The kernel reports
the addresses it *actually* issued against (`src_base`, `dst_base`), and the host
re-derives the congruence from those rather than from its own arithmetic, so a pad
that never reached the kernel is caught.

On both parts as measured here the shortfall came out **0** and the pad was
unused. It exists so that a part where it is not zero produces a measurement
instead of silently skewed bytes.

## The hypotheses, and what each predicts — written before anything was run

Let `L` be the request/response round trip and `K` the number of read requests
the initiator may hold in flight. The four candidates are:

- **H1 credit limit.** The initiator stalls at `K` outstanding requests. The
  sustained rate is `L / K`.
- **H2 responder L1 read port.** The *source* tile serves one read request every
  `T` cycles regardless of who asked.
- **H3 initiator NIU request occupancy.** The initiator's NIU accepts one
  request every `R` cycles; `NOC_CMD_CTRL` does not clear until then.
- **H4 initiator L1 write port.** The *returning data* lands in the initiator's
  L1, and every transaction in tt-metal's dataset lands it at the same address.

| experiment | varies | H1 credit | H2 responder port | H3 initiator NIU | H4 landing port |
| --- | --- | --- | --- | --- | --- |
| **E0** `outstanding` | nothing — reads `NIU_MST_REQS_OUTSTANDING_ID(0)` during a burst | **plateaus at K** | climbs to N | climbs to N | climbs to N |
| **E1** `dist` | hop distance to the one source | **rises**, slope = (9 cycles/hop)/K | flat | flat | flat |
| **E2** `srcfan` | number of distinct source tiles S | flat | **falls ∝ 1/S** | flat | flat |
| **E3** `dstspread` | stride between landing addresses | flat | flat | flat | **falls** |
| **E4** `srcspread` | stride between source addresses | flat | falls *if banks, not port* | flat | flat |
| **E7** `size` | bytes per transaction | rises with size | flat then link-bound | flat then link-bound | flat then link-bound |
| **E6** `burst` | N, the control | reproduces the shipped dataset — if it does not, stop |

**E0 is the one that needs no arithmetic**, and the 2026-08-09 Blackhole card
session is why it now takes three readings instead of one. See
[E0 needs a baseline](#e0-needs-a-baseline-and-did-not-have-one) below before
reading anything off it. `NIU_MST_REQS_OUTSTANDING_ID(i)` is documented in both
architectures' `NoC/Counters.md` and defined identically in both
`noc_parameters.h` as `NOC_STATUS(0x10 + i)`. It counts the initiator's own
in-flight requests. Sample it inside a burst, **subtract that point's own rest
sample**, and the credit limit either is there, with its value printed, or is
not:

- **plateaus well below the burst length** → H1 is live and `K` is *measured*,
  not derived. Then check E1: H1 also requires the rate to rise with distance.
  If E0 plateaus but E1 is flat, the plateau is real and is still not what caps
  the rate.
- **climbs with the burst length** → the initiator holds no bounded number in
  flight, H1 is dead, and the cap is downstream. That **retires** the term
  rather than sizing it, which is a result.

**E2 is the one the shipped dataset structurally cannot provide**, and it is the
separation the cost model's instalment said no arithmetic could do: every
`ONE_FROM_ONE` row in `noc_latencies.yaml` has exactly one source tile, so
responder-side and initiator-side effects are perfectly confounded. Fan the same
burst over 2, 4 and 8 sources and they come apart: an initiator-side limit does
not care how many tiles answered, and a responder-side one is divided by S.

E3 and E4 remove two confounds the shipped dataset **bakes in and never varies**:
every transaction in it reads the same source address into the same destination
address, so every landing is a write-port conflict at the initiator and every
fetch is a bank conflict at the responder. Neither has ever been measured apart
from the rest.

## E0 needs a baseline, and did not have one

The 2026-08-09 Blackhole card session returned `outstanding_max` = **72 in all
129 rows** — at `num_tx` 4, 16, 64 and 128 alike — with
`outstanding_max == outstanding_end` everywhere. Four requests cannot put 72 in
a counter of requests in flight, so the column was not an occupancy. The
program's own verdict then compared 83 against a burst of 64, found
`worst + 4 >= num_tx`, and printed **NO INITIATOR LIMIT**, which the session
reported as *retiring* the credit-limit term. That conclusion is withdrawn: it
was reached from a control that never moved.

Three things were wrong, and all three are fixed in the NRB2 layout.

**1. The counter is live hardware state and does not start at zero.** The kernel
inherits whatever the NIU has been left holding. The card's readings were not
noise — 71 for one family of points, 72 for most, 83 for the single 4096-byte
point — but with no reference they could not be read. The kernel now samples
every counter once **at rest**, before a single request is issued, and the
occupancy is `outstanding_delta = outstanding_max - outstanding_rest`. On that
reading the card's own numbers become a *delta of 0–1 at 64 B and about 12 at
4096 B*, which is a sensible in-flight depth for an issue loop running at ~50
cycles per transaction — but it is a reconstruction, not a measurement, because
the baseline was never sampled. It has to be retaken.

**2. Plain `noc_async_read` does not set a transaction id, so sampling counter 0
was an assumption.** `ncrisc_noc_fast_read` in tt-metal's
`blackhole/noc_nonblocking_api.h` (and Wormhole's) writes `NOC_RET_ADDR_*`,
`NOC_TARG_ADDR_*`, `NOC_AT_LEN_BE` and `NOC_CMD_CTRL` — and never
`NOC_PACKET_TAG`. The id a read carries is therefore whatever that command
buffer's sticky tag last held; tt-metal has a separate
`noc_async_read_set_trid` / `ncrisc_noc_fast_read_with_transaction_id` pair for
when it wants one. tt-sim models the same sticky behaviour
(`extract_bits(self.packet_tag, 4, 10)` in `tt_sim/network/tt_noc.py`). The
kernel now calls `noc_async_read_set_trid(trid)` first, so the id is
*established*. For `trid` 0 that write is a literal zero — the same value
tt-metal's own `noc_clear_packet_tag` writes — so it disturbs no other tag field.

**3. One instrument cannot check itself.** The same quantity is now measured a
second, id-independent way: `NIU_MST_RD_REQ_SENT - NIU_MST_RD_RESP_RECEIVED`.
Both are cumulative one-per-read counters — tt-metal's own
`ncrisc_noc_reads_flushed` compares `RD_RESP_RECEIVED` against a software count
of `noc_async_read` calls — so their difference is the in-flight count whatever
id the requests carry, and it must read **zero at rest** because the kernel
starts after a barrier. That zero is the check that the status block is being
addressed at all. If `inflight_max` and `outstanding_delta` disagree, the
per-trid counter is not watching these reads and the id-free pair is the one to
believe; the program says so rather than picking one.

## The one place the documentation names the mechanism

`WormholeB0/DRAMTile/README.md` — not the NoC tree, which is why an earlier
grep for "credit" and "backpressure" across `NoC/` missed it:

> In the converse direction, when performing large reads, the headers of each
> read request consume 32 bytes of buffer space, so software is encouraged to
> **limit its number of outstanding read requests** to avoid buffers being
> filled by read request headers.

The bound is left to the reader. The same page says each router inbound port has
a 2 KiB buffer, 32 B guaranteed per virtual channel and "up to 480 bytes from
this shared pool" — from which 480 ÷ 32 = **15** read-request headers per VC is
an *inference*, not a published figure, and it applies to a router inbound port
rather than to the initiator. Blackhole has no `DRAMTile/` tree at all, so there
is no counterpart to compare against.

## Blackhole has a register Wormhole does not, and its depth is unpublished

Two independent sources agree, and neither gives a number:

- `BlackholeA0/NoC/MemoryMap.md:21` documents `NIU_BASE + 0x0064` as **"NIU
  request FIFO status", read only, 8 bytes**, with `NIU_CFG_0` bit 16 as
  "Request FIFO enable". There is no section, no anchor and no depth anywhere in
  the repository. Wormhole's `NIU_BASE + 0x054` is "NIU combined request
  initiator status" instead, and Wormhole's `NIU_CFG_0` has no such bit.
- tt-metal's `blackhole/noc/noc_parameters.h:56` defines
  `CMD_BUF_AVAIL (NOC_REGS_START_ADDR + 0x64)` with the comment
  `[28:24], [20:16], [12:8], [4:0]` — four 5-bit per-command-buffer availability
  fields, so a depth of at most 31. Wormhole has no analogue. **No code in
  tt-metal references it.**

A per-architecture NIU request FIFO that Blackhole has and Wormhole does not is
exactly the shape a per-architecture rate difference needs. On Wormhole these
columns read `0xFFFFFFFF`.

### It is an occupancy, so the at-rest read can never be the depth

The 2026-08-09 card session read `cmdbuf_avail_rest` **and**
`cmdbuf_avail_busy` as `0x00000000` in all 129 rows — identical at rest and
mid-burst — and the session called that MEANINGFUL because its only check was
"not `0xFFFFFFFF`". Two separate mistakes:

- **The register is a fill level, not a count of free slots.** No code in
  tt-metal reads it on Wormhole or Blackhole. The only register *descriptor* for
  it anywhere in the tree is Quasar's `noc/registers/noc_niu_reg.h`, where its
  reset value is `NOC_NIU_CMD_BUF_AVAIL_REG_DEFAULT (0x00000000)` and its
  immediate neighbour is `CMD_BUF_OVFL`. A field that resets to zero and is
  paired with an overflow register is an occupancy. **Zero at rest is the
  correct reading and says nothing at all about the depth**, so an earlier
  version of this page was wrong to say the at-rest read "should read the FIFO's
  depth". A maximum occupancy under load is a *lower bound* on the depth, and
  only `CMD_BUF_OVFL` moving proves the buffer was driven to its limit.
- **The "busy" sample was not taken while anything was busy.** It was read after
  the issue loop had ended, by which point every command buffer has long since
  handed its entry to the NIU. It is now sampled *inside* the loop, immediately
  after each `noc_async_read`, and the peak is reported as `cmdbuf_avail_max`.
  `CMD_BUF_OVFL` is read at rest and at the end as `cmdbuf_ovfl_rest` /
  `cmdbuf_ovfl_end`.

If `cmdbuf_avail_max` still equals `cmdbuf_avail_rest` on the next run, the
register is not backed on this part and the depth stays a named `unknown` — say
that, rather than quoting a flat reading as a number.

Both reads also go through the NoC instance offset now
(`CMD_BUF_AVAIL + (noc_index << NOC_INSTANCE_OFFSET_BIT)`); the previous kernel
dereferenced the bare macro and so read NoC 0's block whatever NoC it ran on.

## Running it

```bash
cd perfbench/nocreadbench/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/nocreadbench                       # writes nocreadbench-<arch>.csv
./build/nocreadbench --num-tx 64           # shorter bursts
./build/nocreadbench --repeats 5           # more repeats; default 3
./build/nocreadbench --no-sample           # skip E0's untimed second burst
./build/nocreadbench --stateful            # the SHORTER issue loop (see above)
./build/nocreadbench --dram                # ADD the DRAM burst arm (see above)
./build/nocreadbench --dram-bank 3         # ... out of a different DRAM bank
./build/nocreadbench --only burst          # just one axis, for the simulator
```

For the DRAM arm, use the session runner rather than the flag: it prints the
absolute prediction **before** any card time, smoke-tests the DRAM source on the
part, refuses instrumented runs, and grades the result against the same table it
printed.

```bash
perfbench/nocreadbench/run_card.sh --dram --preflight   # costs no card time
perfbench/nocreadbench/run_card.sh --dram               # ~3 minutes of card
```

`--dram` and `--arms` are two sessions, not one: the stateful arm varies the
issue loop at a fixed 64 B L1 source, the DRAM arm varies the source and the
payload at a fixed issue loop. Running them together is refused so that each
one's control is its own.

For the stateful comparison, do not run the two arms by hand — the protocol
interleaves them, checks each one's mode against its own payload, checks the
control against 2026-08-17 on the spot, and prints the paired verdict:

```bash
perfbench/nocreadbench/run_card.sh --preflight    # costs no card time
perfbench/nocreadbench/run_card.sh --arms         # ~3 minutes of card
```

`--stateful` drops the `srcfan` points with more than one source: the stateful
path holds the source tile in the command buffer, so cycling sources would pay
back the very store the variant removes. It says how many it dropped and why, and
the kernel refuses such a point independently — a drop that failed to happen
still cannot be measured.

`--only` changes nothing about a point that survives it, and that is checked
rather than argued: `--only burst` on the Wormhole simulator reproduces the full
plan's burst cycles exactly (192 / 648 / 2472 / 4904). The L1 arena is sized from
the largest transaction in the plan, and every path here already needs the 8 KiB
floor, so filtering does not move an address either.

`--num-tx` defaults to **128 and should not be raised much above it**:
`NIU_MST_REQS_OUTSTANDING_ID` is 8 bits, and both architectures'
`NoC/Counters.md` warns it "will only overflow or underflow if software has too
many outstanding requests".

The timed burst and the sampled bursts are **separate runs of the same loop**.
Sampling costs a `>= 7` cycle NIU load with a six-cycle load-use interlock, in a
loop whose whole per-iteration cost is under 40 cycles — sampling inside the
timed region would change the rate it is meant to explain. So `cycles` never
contains a sample, and no high-water mark ever comes from a timed loop. With
NRB2 there are three bursts per point: one timed and unsampled, one sampling the
per-trid counter and `CMD_BUF_AVAIL`, and one sampling the id-free
`RD_REQ_SENT - RD_RESP_RECEIVED` pair. Sampling slows the issue rate, and a
slower issue rate can only make an occupancy look *smaller* — every high-water
mark here is read as a lower bound.

## Telling a good run from a degenerate one

The program prints a verdict. These mean *do not read the rate columns*:

- `DEGENERATE — neither instrument moved off its rest value.` Either the
  sampling load was hoisted out of the loop, or every read completed before the
  next iteration. Send the CSV anyway and say which build flags you used.
- `DEGENERATE — the two instruments disagree.` One of the per-trid counter and
  the id-free pair saw requests in flight and the other did not, so at least one
  of them is not watching these reads.
- `DEGENERATE — requests already in flight before issuing anything.` The kernel
  starts after a barrier, so `inflight_rest` must be zero. If it is not, the
  status block is not being read the way this program assumes and nothing in the
  file means anything.
- `CMD_BUF_AVAIL: DEGENERATE` — rest, last in-loop sample and peak all agree, so
  the register reported nothing. **This is not a depth of zero.**
- `SOURCE NOT CONFIRMED` — one or more points could not prove which **kind** of
  tile their own timed burst read. On a `--dram` run this is the worst possible
  failure to ignore, because an L1 read wearing a DRAM label **reproduces the
  registered prediction exactly** below the crossover: it looks like agreement.
  `check_mode.py` re-derives it from the CSV alone.
- `nocreadbench: REFUSING TO RUN` — `TT_METAL_DEVICE_PROFILER_NOC_EVENTS` is set.
  Not a warning and not overridable: unset it and re-run. If you want NoC event
  traces, take them in a separate run and do not report them beside these rows.
- `MODE NOT CONFIRMED` — one or more points could not prove which issue loop they
  ran from the tile that answered their own probe. Every rate in the file is then
  unattributable, and a stateful number produced by the stateless loop is
  precisely the wrong answer to the question the variant is asked. `check_mode.py`
  re-derives the same thing from the CSV alone.
- On **Wormhole**, the `burst` marginals moved off the 2026-08-17 session's
  **44.08 / 44.00 / 43.97 cycles/transaction** by more than 3 %. That session is
  the control; `check_mode.py` checks it and fails the run. Note that the shipped
  dataset's 25.0 (Wormhole) / 35.0 (Blackhole) are **not** the control — they were
  taken on other parts, and this part already disagrees with the Wormhole figure
  by 1.76×. On **Blackhole** there is no recorded card session yet, so the
  stateless arm establishes a control rather than reproducing one. Note also that
  at `--num-tx 128` Wormhole is *above* the dataset's N = 16 → 64 regime change
  and Blackhole is not in a regime at all.
- Any row with `cycles == 0` or a missing result stamp — the kernel did not run.
  After a layout change the magic moves (`NRB3` → `NRB4`), and a host binary built
  before it reads the new kernel's stamp as garbage: **rebuild the host program
  whenever the kernel changes**, which `run_card.sh` does by default.

## What happens to the number afterwards

Whatever comes back is **a measurement on one part**. It enters
`tt_sim/perf/unit_costs.yaml` only as a `corroboration` field on an existing
entry, never as provenance, and it does not by itself become a charged cost
term. The rules are in `tt_sim/perf/costs.py`: every bound is charged at its low
end, there are no `estimated` entries, and a silicon measurement is corroboration
rather than a source.

What *would* make it chargeable is Tenstorrent publishing the depth of the
Blackhole NIU request FIFO — the register is documented to exist and its size is
not. That is a documentation request, not a measurement, and it is the shortest
route from here to a number the model may spend. Failing that, the term stays a
named, sized `unknown`, which is the honest end state and is written up in
[`../../docs/plans/cost-model.md`](../../docs/plans/cost-model.md).

## Against the simulator

```bash
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run_card_session.sh --sim --arch wormhole  nocread cmdbuf
TT_METAL_HOME=/path/to/tt-metal ./perfbench/run_card_session.sh --sim --arch blackhole nocread cmdbuf
```

`src/nocreadbench-wormhole-sim.csv` and `src/nocreadbench-blackhole-sim.csv` are
exactly those two runs, cost model **off**, at the session's smoke settings
(`--num-tx 8 --repeats 1`); `src/nocreadbench-wormhole-sim-stateful.csv` and
`src/nocreadbench-blackhole-sim-stateful.csv` are the same two with
`--stateful`. They are checked in as shape references for the columns, not as
measurements of anything, and they are what proves the NRB4 layout — both issue
loops, the mode witness and the source witness — builds and runs on both parts.

`src/nocreadbench-wormhole-sim-dram.csv` and `src/nocreadbench-blackhole-sim-dram.csv`
are the DRAM arm's, and they are a different kind of artefact: they are taken with
the **cost model ON**, and they *are* the registered prediction rather than a
shape reference. Reproduce them with

```bash
TT_SIM_COST_MODEL=1 TT_SIM_ARCH=wormhole perfbench/run.sh nocreadbench -- \
    --dram --only burst,bigburst,dramburst,dramsize --num-tx 4 --repeats 1 --no-sample
```

`check_mode_test.py` asserts they still agree with `SIM_DRAM_PREDICTION` to 0.01
cycles, so the table and the artefact cannot drift apart without a test failing.

### What tt-sim says about the two arms, and why it is not an answer

Differencing the `burst` axis of those four files:

| | stateless | stateful | the loop bought |
| --- | --- | --- | --- |
| tt-sim, Wormhole, **cost model OFF** | 38.00 cycles/tx | 23.00 | **15.00** |
| tt-sim, Blackhole, **cost model OFF** | 41.00 | 23.00 | **18.00** |
| tt-sim, Wormhole, **cost model ON** | **44.00** | **29.00** | **15.00** |
| tt-sim, Blackhole, **cost model ON** | **47.00** | **29.00** | **18.00** |
| Wormhole card, 2026-08-17 | 45.03 | 28.96 | 16.07 |
| the shipped dataset | 25.00 / 35.00 | 17.33 / 34.00 | 7.67 (WH) / 1.00 (BH) |

**Read the configuration before you read the numbers.** The four checked-in
`*-sim*.csv` files are **cost-model-off** runs, and a cost-model-off marginal is
an *instruction count* — by design, and asserted by
`tt_sim/perf/noc_issue_loop_test.py:180`. Comparing one to a card is a category
error, and on 2026-08-17 it produced a confident false finding: the off figures
sit ~7 and ~6 cycles below the card, which reads as a fixed missing
per-transaction term. It is not one. With the model on, tt-sim reads 44.00 and
29.00 against the card's 45.03 and 28.96 — residuals **-1.03** and **+0.04**,
*opposite in sign*, so there is no shared constant to find. The 44 is fully
accounted for: 38 instructions at one cycle plus the **6-cycle load-use
interlock** on the `noc_cmd_buf_ready` poll, both already `isa_doc` in
`unit_costs.yaml`.

Three things to read off it, and one not to.

- **tt-sim's stateless arm is unchanged by this work.** 38.00 and 41.00 are what
  the NRB2 program cost here, bit-for-bit, at all three burst intervals. That is
  the control check the card cannot do at home.
- **tt-sim's saving does not match either of the dataset's**, and on Blackhole it
  goes the wrong way: the dataset says the shorter loop buys 1.0 cycles there and
  7.67 on Wormhole, while tt-sim says 18 and 15. Nor is it the ~1–2 cycles that
  `tt_sim.perf.noc_issue_loop`'s reconstruction would predict from removing one
  store (Wormhole) or two (Blackhole) — the compiled loop differs by more than
  its stores, because the stateless arm also carries 64-bit address arithmetic
  the stateful one does not.
- **That mismatch is structural, not a bug to tune out.** tt-sim's NIU appends to
  an unbounded queue, so it has no per-read floor on either architecture *by
  construction*; every cycle the variant saves must therefore show up as
  instruction-stream saving. A dataset row where the shorter loop buys 1.0 of 35
  is the signature of something tt-sim does not model, and the simulator cannot
  be the witness for or against it. **Nothing in `tt_sim/perf/` was changed to
  narrow the gap.**

What the simulator side *does* establish is that both arms build, run and
attribute themselves correctly on both architectures, and that the control arm
did not move.

The Blackhole one could not exist until recently: the Blackhole build reads
`CMD_BUF_AVAIL` and `CMD_BUF_OVFL`, and tt-sim's NUI raised
`NotImplementedError` on `NOC_REGS_START_ADDR + 0x64` and `+ 0x68`, so the
Blackhole half of this program aborted against the simulator rather than
running. Both addresses now read as the all-ones "register absent" sentinel —
the same value this program's own `#else` branch produces on a part that does
not define them — so a simulator run reports the fields as *absent* instead of
inventing a plausible occupancy, and every `cmdbuf_*` column agrees on one value
for "absent". See `tt_sim/network/noc_registers_test.py`.

The simulator answer is a **known null and is not a result**: tt-sim's NIU
appends to an unbounded queue (`add_outstanding_noc_request`), so E0 reads the
full burst length by construction and E1–E4 are flat by construction. It is run
only to prove the harness executes and the columns are populated — exactly the
role `nocbench`'s `INVALID` verdict plays for congestion. Keep `--num-tx` small;
the simulator runs a few tens of thousands of cycles per second.
