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
```

The `--arms` session is the newer question and the one with pre-registered
predictions: see [The stateful variant](#the-stateful-variant-and-the-two-predictions).

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

One honest caveat: the shared kernel body moved the stateless loop's **constant**
term by a few cycles against NRB2 — measured on tt-sim as **+4 cycles on Wormhole
and +1 on Blackhole**, while the *per-transaction* cost is bit-identical (38.00
and 41.00 before and after, at all three burst intervals). That is register
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
./build/nocreadbench --only burst          # just one axis, for the simulator
```

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
  After a layout change the magic moves (`NRB2` → `NRB3`), and a host binary built
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
measurements of anything, and they are what proves the NRB3 layout — both arms,
and the mode witness — builds and runs on both parts.

### What tt-sim says about the two arms, and why it is not an answer

Differencing the `burst` axis of those four files:

| | stateless | stateful | the loop bought |
| --- | --- | --- | --- |
| tt-sim, Wormhole | 38.00 cycles/tx | 23.00 | **15.00** |
| tt-sim, Blackhole | 41.00 | 23.00 | **18.00** |
| the shipped dataset | 25.00 / 35.00 | 17.33 / 34.00 | 7.67 (WH) / 1.00 (BH) |

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
