# nocreadbench — what limits the sustained NoC *read* rate on a card

You have a Tenstorrent card. This asks it one question that neither the public
ISA documentation nor tt-metal's headers answer, and it asks it two ways: once
by reading a hardware counter, and once by the shape of a sweep.

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal   # built with ./build_metal.sh
cd perfbench/nocreadbench/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
../run_card.sh                                      # ~1–3 minutes
# -> nocreadbench-<arch>.csv   <- this is the file to send back, with the console output
```

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

**E0 is the one that needs no arithmetic.** `NIU_MST_REQS_OUTSTANDING_ID(i)` is
documented in both architectures' `NoC/Counters.md` and defined identically in
both `noc_parameters.h` as `NOC_STATUS(0x10 + i)`. It counts the initiator's own
in-flight requests. Sample it inside a burst and the credit limit either is
there, with its value printed, or is not:

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
exactly the shape a per-architecture rate difference needs. The kernel therefore
reads `CMD_BUF_AVAIL` twice — once at rest before any request is issued, which
should read the FIFO's *depth*, and once mid-burst, which reads its remaining
space — and reports both. On Wormhole those columns read `0xFFFFFFFF`.

If the at-rest read gives a clean small integer in all four fields, **that is the
number the documentation omits**, and it is a register read rather than a fit.

## Running it

```bash
cd perfbench/nocreadbench/src
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/nocreadbench                       # writes nocreadbench-<arch>.csv
./build/nocreadbench --num-tx 64           # shorter bursts
./build/nocreadbench --repeats 5           # more repeats; default 3
./build/nocreadbench --no-sample           # skip E0's untimed second burst
```

`--num-tx` defaults to **128 and should not be raised much above it**:
`NIU_MST_REQS_OUTSTANDING_ID` is 8 bits, and both architectures'
`NoC/Counters.md` warns it "will only overflow or underflow if software has too
many outstanding requests".

The timed burst and the sampled burst are **separate runs of the same loop**.
Sampling costs a `>= 7` cycle NIU load with a six-cycle load-use interlock, in a
loop whose whole per-iteration cost is under 40 cycles — sampling inside the
timed region would change the rate it is meant to explain. So `cycles` never
contains a sample, and `outstanding_max` never comes from a timed loop.

## Telling a good run from a degenerate one

The program prints a verdict. Three of them mean *do not read the rate columns*:

- `DEGENERATE — the counter never moved.` Either the sampling load was hoisted
  out of the loop, or every read completed before the next iteration. Send the
  CSV anyway and say which build flags you used.
- The `burst` rows disagree with `noc_latencies.yaml` (25.0 cycles/transaction
  on Wormhole, 35.0 on Blackhole, at 64 B and N ≥ 64). If this control does not
  reproduce, nothing downstream of it is worth reading. Note that at
  `--num-tx 128` Wormhole is *above* the N = 16 → 64 regime change and Blackhole
  is not in a regime at all.
- Any row with `cycles == 0` or a missing result stamp — the kernel did not run.

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
TT_METAL_HOME=/path/to/tt-metal TT_SIM_ARCH=wormhole \
  TT_SIM_TENSIX_COORDS=1-1,2-1,3-1,4-1,1-2,1-3,1-4,1-5 \
  TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4 \
  ./perfbench/run.sh nocreadbench -- --num-tx 8 --repeats 1
```

`src/nocreadbench-wormhole-sim.csv` is exactly that run, cost model **off**, at
`--num-tx 8 --repeats 1`. It is checked in as a shape reference for the columns,
not as a measurement of anything.

The simulator answer is a **known null and is not a result**: tt-sim's NIU
appends to an unbounded queue (`add_outstanding_noc_request`), so E0 reads the
full burst length by construction and E1–E4 are flat by construction. It is run
only to prove the harness executes and the columns are populated — exactly the
role `nocbench`'s `INVALID` verdict plays for congestion. Keep `--num-tx` small;
the simulator runs a few tens of thousands of cycles per second.
