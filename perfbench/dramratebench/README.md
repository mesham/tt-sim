# dramratebench — does a DRAM endpoint's rate grow with the number of participants?

`N` Tensix tiles each move 1 MiB to or from DRAM, simultaneously, with `N`
swept. Three arms differ in **which endpoint** those transfers land on and in
nothing else; a `--dir` axis differs in **which direction** they go and in
nothing else.

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal
perfbench/dramratebench/run_card.sh          # the READ campaign, on a card
perfbench/dramratebench/run_card_write.sh    # the WRITE campaign, on a card
./perfbench/run.sh dramratebench -- --bytes 8192 --tx-bytes 512 --readers 1,2
```

`--dir` defaults to `read`, so every invocation that predates the write
direction produces exactly the file it produced before. See
[the write direction](#the-write-direction) below.

**At the card, this is the whole of it** — copy and paste, no other setup, and
nothing but `perfbench/` needs to be on the box:

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal   # a BUILT tt-metal checkout
unset TT_METAL_SIMULATOR                            # the script refuses if it is set
./perfbench/dramratebench/run_card.sh -- --sustained
```

`--sustained` runs the vendor's own reader counts, 1 / 12 / 48, at the vendor's
own 1 MiB per reader and three repeats; the widest count the part can address is
always added on top. It prints a sustained-rate table per reader count, next to
the published 22.2 / 22.3 / 22.3 GB/s where the part is a Wormhole, and then its
verdict; the script names the CSV to send back. Drop `--sustained` (or run
`./perfbench/run_card_session.sh dram`) for the full 1,2,4,8,12,16,24,32,48 sweep
— about four times the card time, and the same reading with more points under it.

The prediction that run is to be read against was recorded **before** it, in
`perfbench/dramratebench/prediction-sustained.csv`, and travels to the card box
in the same rsync.

## What it is for

tt-sim charges DRAM **endpoint occupancy** since 2026-08-09: a request arriving
while the channel is streaming another one waits for it (`DramChannels` in
`tt_sim/device/tiles.py`). The number is not a new one — it is
`ceil(N / dram.channel_serialisation.bytes_per_cycle)`, which the model already
spends as a latency — so the term needs no measurement to be *sourced*.

What it does need is a check on the **shape** it asserts, and no existing probe
can reach that. Rung 2's retained DRAM rows are every one of them
`num_transactions = 1`, and a lone request never finds the channel busy.
Endpoint occupancy is, by construction, invisible to every single-transfer
measurement in this project.

`wh_dram#performance` states the shape as a vendor measurement, and
`unit_costs.yaml` already quotes it:

> 1, 12 and 48 Tensix tiles each reading 1 MiB **from one DRAM channel
> simultaneously** measure **22.2, 22.3 and 22.3 GB/s**.

The aggregate does not grow with the number of readers. That flatness *is*
endpoint occupancy stated as a measurement, and it is what this program
reproduces.

## Why a flat reading is not, on its own, a result

Flat is what a saturated channel looks like. It is also what a saturated NoC
link looks like, what a saturated initiator issue rate looks like, and what `N`
readers that never actually overlapped look like. A program that measured only
the one-channel arm would be unreadable for exactly the reason `nocreadbench`'s
first two attempts were, and reporting its flat curve would repeat the `nocread`
mistake of 2026-08-09, where a constant register read was written up as a
finding.

**So the flat curve is the expected answer, and it is therefore not the
control.** Three arms:

| arm | where the `N` readers read from | prediction |
| --- | --- | --- |
| `onechan` | all of them, ONE DRAM bank | **flat** — the measurement |
| `fanchan` | `N` DISTINCT banks, distinct physical cores first | **scales** — the control |
| `samecore` | the ≥ 2 banks that share ONE physical DRAM core | scales if the *channel* binds |

`fanchan` is what separates the endpoint from everything upstream of it. The
same `N` tiles, the same issue loop, the same transaction size, the same
barrier: **only the endpoint differs.** If `fanchan` scales and `onechan` does
not, the thing that saturated is the endpoint, because nothing else about the
two arms is different. If `fanchan` is also flat, something upstream caps both,
the one-channel flatness is uninterpretable, and the run reports `DEGENERATE`.

`samecore` splits the one ambiguity `fanchan` leaves. In `onechan` all `N` flows
converge on one DRAM tile's inbound router link, so a link limit and a channel
limit are still confounded — and on Wormhole the two rates differ (32 B/cycle on
the wire, 24 at the channel), so which one a plateau sits at is itself
diagnostic. `samecore` keeps that link and changes only the channel.

**It does not fire on either part in this tree, and that is measured rather than
assumed.** It wants two banks on one physical NoC coordinate, and tt-metal maps
every bank to a distinct coordinate on both descriptors: Blackhole has one view
per DRAM core, and Wormhole's twelve banks come back on twelve *different*
coordinates even though their `get_bank_offset` values alternate `0x0` /
`0x4000_0000` — exactly the two-channels-per-tile split `wh_dram` describes
(*"DRAM tiles occur in groups of three, with two channels of GDDR6 present in
each group"*). The pairs that share a tile are reached through aliased
sub-endpoints a host program cannot tell apart from independent ones. So the arm
is coded, is selected automatically wherever a mapping does front two banks on
one core, and is skipped by name with that reason today — which means **the
link-versus-channel ambiguity stays open** and a clean result must be reported
as "the endpoint" rather than "the channel".

### The verdict is a ratio, not a threshold

`onechan_scale / fanchan_scale`, graded at 0.75. An absolute rule — "`onechan`
must stay under ×1.5" — is really a question about how wide the sweep was, since
a channel cannot flatten a load that does not saturate it. On tt-sim's own
Wormhole numbers with the cost model on (four readers: `onechan` ×2.21 against
`fanchan` ×3.97) the absolute rule reports the endpoint-occupancy term REFUTED
by the run that demonstrates it. The ratio reads 0.56 and reports it bound.

## The sustained rate: the LEVEL, and the prediction it is read against

A ratio establishes that concentrating the readers *cost* something. It cannot
say **what they concentrated onto**, and it cannot say **where the curve
flattened** — and both of those are checkable against published numbers where a
ratio is not. A one-channel arm plateauing at 24 B/cycle and one plateauing at
40 give the identical ratio, and only one of them agrees with the model.

The model is specific. `dram.channel_serialisation.bytes_per_cycle` is **24** on
Wormhole, so a saturated one-channel aggregate cannot exceed 24 B/cycle there —
24.0 GB/s at the 1 GHz the same ISA doc publishes. The vendor's measured 22.2 is
**92.5%** of that, and the same page's own `achievable_fraction` is **0.92**, a
figure `unit_costs.yaml` deliberately refuses to fold into the charge ("an
efficiency fudge of exactly the kind these tables refuse"). So the model and the
published table already agree to within one documented, deliberately-unapplied
constant, and the sweep can say so per reader count rather than in the abstract.

### The prediction — recorded 2026-08-12, before any Wormhole card ran this

`perfbench/dramratebench/prediction-sustained.csv`, pinned cell by cell in
`tt_sim/perf/dram_rate_sweep_test.py` so that it cannot be quietly revised after
a measurement. tt-sim, `TT_SIM_COST_MODEL=1`, 1 MiB per reader, 4096 B
transactions, twelve materialised tiles:

| readers | predicted `onechan` B/cycle | → GB/s at 1 GHz | published |
| --- | --- | --- | --- |
| 1 | 23.749 | 23.75 | **22.2** |
| 2 | 23.860 | 23.86 | — |
| 4 | 23.918 | 23.92 | — |
| 8 | 23.946 | 23.95 | — |
| 12 | 23.958 | 23.96 | **22.3** |
| 48 | 23.990 *(plateau, not simulated)* | 23.99 | **22.3** |

**Stated plainly: the prediction is the right shape and about 7% high.** Flat
from one reader to forty-eight — the aggregate grows by ×1.010 across the whole
sweep, against the vendor's own ×1.0045 — and sitting **+7.0% to +7.6%** above
22.2/22.3 at every point. The gap is one-directional and its size is published in
advance: 24 × 0.92 = 22.1, which is the table.

Three ways it can be wrong, and each says something different:

* **the aggregate GROWS with readers** — endpoint occupancy is refuted, and the
  term shipped on 2026-08-09 is charging a serialisation the hardware does not
  have;
* **it plateaus near 32 B/cycle** — the flat arm found the NoC link, not the
  channel, and the term is charging the right shape at the wrong resource;
* **it plateaus far below 22** — something upstream of both binds first, and the
  run sizes the reader rather than the endpoint.

Beyond 12 readers tt-sim cannot be run at these parameters in reasonable time —
48 tiles × 48 MiB through one modelled channel is over two million simulated
cycles with 48 tiles stepping — so the 48-reader row is the model's own asymptote
and is labelled `plateau-extrapolated` in the file rather than passed off as a
simulated point.

### Blackhole's column is a RETRODICTION, and it is already contradicted

The lab has a Blackhole part and it ran this sweep on 2026-08-09, so nothing
about Blackhole here can be called a prediction. It is recorded anyway, labelled,
because of what it shows: tt-sim predicts a plateau at **64.0 B/cycle** and the
card measured **47.1**. The simulator is 36% high, and the plateau it produces
is the DRAM tile's NoC **link** — 64 B/cycle exactly — because Blackhole's
endpoint queue is switched off for want of a published per-channel rate. The
card's 47.1 sits at neither ceiling the model holds, and agrees to about 0.05%
with rung 2's wholly independent 47.1 B/cycle sizing of Blackhole DRAM reads.

**None of that made anything chargeable, and something became chargeable
anyway.** The card cannot supply provenance and never did. But re-reading the
sentence above showed the *block* had been misidentified: what tt-sim was
missing on Blackhole was `dram.channel_serialisation`, a bytes-per-cycle figure,
and a bytes-per-cycle figure does not have to arrive by unit conversion from a
published GB/s. Since 2026-08-12 it arrives instead from arithmetic on two of
tt-metal's own measured cycle counts — `vendor_source_derived`, 47.0805 B/cycle
for reads, the same two-point secant that recovers Wormhole's published 24 from
the same file. `dram.bandwidth` (the GB/s spec block) is still `unknown` and
still needs a document; it simply never gated the term.

So the standing of the numbers above has changed, and only in one direction:
the card's 47.147 is **corroboration** of a derivation that predates it and
never saw it, agreeing to 0.14 %. tt-sim's own plateau is no longer 64.0 — it
is the derived channel rate, and `plateau_sits_at` now answers `channel` for
this run where it answered `neither`.

### Reading a run

```bash
python3 -m tt_sim.perf.dram_rate_sweep --measured dram.wormhole.csv
```

It applies the gates below, prints the aggregate per reader count in
B/cycle and GB/s, says which ceiling the plateau landed on, and only then
compares to the committed prediction and to the published table. `--no-prediction`
reads the levels alone. The program itself prints the same table at the card, so
none of this needs `tt_sim/` on the card box.

## What it can never do

**Nothing here is provenance.** A number this program produces is a measurement
on one part and can enter `unit_costs.yaml` only as `corroboration`.

That is doubly true on Blackhole, which has **no published DRAM tile page at
all** — no per-channel bandwidth, no address map, no channel count. Nothing
measured on a Blackhole card can make `dram.bandwidth` chargeable there, and a
clean run must not be reported as though it could. What Blackhole *can* do is
validate the **shape**: an aggregate that refuses to grow with readers while a
fanned-out control grows freely is a bound at the shared endpoint, whatever the
absolute number turns out to be.

(What *did* make Blackhole's channel rate chargeable, on 2026-08-12, is not on
this page and not on a card: it is a two-point secant on tt-metal's shipped
`noc_latencies.yaml`, `vendor_source_derived`. The distinction is the whole
point of this section — vendor arithmetic is provenance, a card run is not,
and it stays that way however well the two agree.)

**And "at the shared endpoint" is as far as the shape alone goes** — it is not
the same claim as "at the GDDR6 channel". tt-sim produces exactly that shape on
Blackhole with no endpoint queue at all, out of the DRAM tile's inbound router
link. Only the level tells them apart, which is what the section above is for.

## Addressing, and why none of it is arithmetic

The 2026-08-09 Blackhole card is harvested — addressed worker columns
`{1..7, 10..14}` against a physical `{1..7, 10..16}` — and a sweep that picked
coordinates by arithmetic would land in a hole and return a row indistinguishable
from a measurement. Nothing here does:

* **Readers** are chosen in LOGICAL space from
  `compute_with_storage_grid_size()`, so a harvested column is not addressable
  at all. Same argument as `nocreadbench`'s.
* **DRAM banks** come from `logical_core_from_dram_channel()`,
  `virtual_core_from_logical_core(..., CoreType::DRAM)` and
  `get_bank_offset(BufferType::DRAM, id)` — what the allocator itself uses. The
  program refuses to run at all if `get_num_banks(DRAM) != num_dram_channels()`,
  because the whole plan assumes bank id *is* the channel index.
* **And then it checks.** The host writes a distinct tag word at the base of
  every bank's slice; every reader reads it back before the timed region and
  stamps whether it matched. `tags_ok` less than `num_readers` in any row
  condemns the file. A rate measured from the wrong endpoint is not a smaller
  result, it is a fictional one.

Two more things a row must prove before it is read:

* **`max_barrier_spins`** — did any reader wait for the others? The readers
  check in by writing one word into every participant's arrival array and then
  spin locally until all `N` slots are set. In a multi-reader point at least one
  reader must have spun, or the bursts ran one after another and the "N tiles
  simultaneously" never happened.
* **The clock.** Each reader differences its OWN wall clock and the aggregate is
  built from `max(t1 - t0)`, never `max(t1) - min(t0)`. Blackhole core (11, 2)
  carried a +323438586-cycle clock epoch reproduced over five runs in this
  project's own congestion data; differencing across cores would fold that in.
  `max` rather than `mean` charges the slowest reader against the aggregate,
  which can only make a scaling arm look *less* scaled.

## Against tt-sim

`--sim` is a harness check and not a measurement, and what it reads depends on
which simulator and how many tiles. All three of these are correct and none is a
fault:

| run | reads | why |
| --- | --- | --- |
| Blackhole, one tile | `DEGENERATE` | one reader count, so the fan-out control has no pair to compare and nothing separates the endpoint from anything |
| Blackhole, two tiles | `NO ENDPOINT BOUND` | recorded before 2026-08-12, when the endpoint queue was inert *by construction* on that part: `DramChannels.bytes_per_cycle` was `None` and every `claim()` a no-op, so both arms scaled ×2.00 exactly and there was nothing to contend for. That arch now carries a derived read rate, so this row is history rather than the current reading |
| Wormhole, four tiles, `TT_SIM_COST_MODEL=1` | `ENDPOINT BOUND`, ratio 0.56 | the term is live there, and this is the run that shows the probe reaching it |
| **Blackhole, twelve tiles, at the vendor's own parameters** | `ENDPOINT BOUND`, ratio 0.25 | recorded when the endpoint queue was **still switched off** on that part — the flat arm was the 64 B/cycle link, which is why the level and not just the shape has to be read. Both halves are now live |

The session's `--sim` path widens `TT_SIM_TENSIX_COORDS` to two tiles for this
probe's own run — scoped to it, so no later probe is slowed — because the reader
count is the axis the whole thing sweeps and at one reader the barrier is never
even exercised.

**The last row is not a fault either, and it is the most important thing on this
page.** At the smoke sizes the `--sim` path uses (8192 B per reader, 512 B
transactions) nothing saturates anything, both Blackhole arms scale ×2.00, and
the run reads `NO ENDPOINT BOUND`. At the parameters the vendor's table was
measured at — 1 MiB per reader, 4096 B transactions — the same simulator, on the
same part, gives

| readers | `onechan` B/cycle | `fanchan` B/cycle |
| --- | --- | --- |
| 1 | 62.189 | 62.189 |
| 2 | 63.112 | 123.384 |
| 4 | 63.630 | 246.767 |
| 8 | 63.892 | 252.936 |
| 12 | **64.011** | 254.689 |

`onechan` ×1.03 against `fanchan` ×4.10 — 25% of it — and the ratio verdict
says `ENDPOINT BOUND`, *"the endpoint is what cost the difference"*, on a part
where `DramChannels.bytes_per_cycle` is `None` and every `claim()` is a no-op.
Nothing about that verdict is wrong given what it can see. What flattened the
arm is the DRAM tile's inbound **NoC link**, whose 64 B/cycle the plateau sits
on to four significant figures, and the ratio cannot tell a link from a channel
because the two differ only in their **level**.

That is the whole argument for reading the level as well, and
`tt_sim.perf.dram_rate_sweep` does exactly that: it puts the plateau next to
both ceilings the cost tables hold — the link's `noc.flit_bits` × throughput and
the channel's `dram.channel_serialisation` — and says which one, or neither, it
landed on.

None of it is alarming at the card, and all of it is the documented answer.

## Telling a good run from a degenerate one

In order, and each can only ever fail the run:

1. `tags_ok == num_readers` in every row.
2. `max_barrier_spins > 0` in some multi-reader point.
3. the `fanchan` aggregate GREW between the narrowest and widest reader count.
4. only then is `onechan_scale / fanchan_scale` read, and only that ratio.
5. and only then the **level**: where the plateau sits, against the prediction
   and against the two ceilings — which is what says whether the thing it found
   was the channel or the link.

`perfbench/card_session_verdicts.sh`'s `dram_verdict` applies the first four, in
that order, and `card_session_verdicts_test.sh` runs them against synthetic files
that fail each one. `tt_sim.perf.dram_rate_sweep` applies the same first four —
as three gates, since 3 and 4 are one arm's reading — and then the fifth, and
`dram_rate_sweep_test.py` drives **every one of them in both directions**: rows
built to satisfy it and rows built to break it. That is not tidiness. This
probe's first card run was refused by its own tag check because the host tagged
slice 0 only, so every reader past the first had nothing to match; the data was
clean and was thrown away. A check that cannot pass is exactly as damaging as
one that cannot fail.

## The write direction

`unit_costs.yaml` charges one DRAM `access_latency` to a read, a write and an
atomic alike on Wormhole, and the entry records why it declined to split them:
Blackhole's `tm_noc_latencies` rows split cleanly (a write of 22 cycles against
a read of 126) and doing the identical subtraction on Wormhole's gives write
differences of `228 / 124 / 127 / 139 / 139 / 145 / 137 / 155` against read
differences `104 / 104 / 104 / 112 / 104 / 112 / 128 / 152` — *"no clean
asymmetry, a visible 64 B outlier, and a write that is if anything DEARER than a
read"*. It stays one figure **"until Wormhole's own data says otherwise"**.

A Wormhole card has since said something and it **agrees with the decline**: a
latency probe on 2026-08-17 put the DRAM write at `+21.2` cycles over the read
at 256 B and `+56.4` at 4096 B, dearer in both, opposite in sign to Blackhole's.
Copying Blackhole's split across would have been actively wrong, not merely
unsupported.

**Neither of those can settle it.** Both are *latency* instruments — one
transaction per row — and a latency instrument structurally cannot see an
endpoint's **occupancy**, which is how long it is unavailable to the *next*
request. That is a rate question, it is this program's shape, and `--dir write`
is what asks it.

### What it answers, and the three things it does not

It answers: *are `N` writers concentrated on one endpoint serialised by it more,
less or the same as `N` readers are?* It does **not** price a single write, it
cannot split service time from queueing, and — because the `samecore` arm does
not exist on either part — it cannot tell the endpoint from the one inbound
router link every concentrated flow converges on. Report **"the endpoint"**,
never "the channel", in either direction. Nothing it produces is provenance:
Wormhole's `dram.bandwidth` is `unknown` for want of a *document*, and no
measurement supplies one.

### How a write proves it wrote where it says

The bar is higher than the read arm's and it has to be. A read that misses its
endpoint returns the wrong bytes and the reader itself reports it; a write that
misses its endpoint completes, retires its barrier, and yields a perfectly
plausible rate with the bytes sitting somewhere the plan never named. Three
checks, and only the middle one has a read-arm analogue:

| check | who makes it | what it catches | what it cannot |
| --- | --- | --- | --- |
| the payload **is** the witness | the writer, before the barrier | nothing on its own — it is what makes the other two about the *measured* traffic rather than a separate untimed poke | — |
| `tags_ok` | the writer, after the barrier | a write that was dropped, mis-congruent or never issued: the block still holds POISON | **misdirection** — it re-reads through the coordinate and offset the write used, so a consistently wrong write reads back consistently wrong and matches |
| `witness_ok` / `stray_writes` | the **host**, over PCIe | misdirection, in both directions: every writer's target block must hold that writer's witness, and every block the point did *not* target must still hold POISON | bytes that landed in the right bank but on the wrong GDDR6 channel behind it — tt-metal fronts every bank on its own NoC coordinate, so a host program cannot tell two sub-endpoints of one tile apart |

The witness is keyed to the **target** `(bank, slice)` and not to the writer,
because a point with more writers than slices has two writers aimed at one block
and they must agree about what belongs in it. The host **re-lays POISON before
every write point**, not once at start-up: after the first repeat the region
already holds correct witnesses, so a later repeat whose writes were all
silently dropped would read back exactly the right words and certify itself.
This benchmark has been bitten by a check that could not *pass* (the 2026-08-09
card run tagged slice 0 only, `tags_ok` came back 1 of 24, and clean data was
thrown away); a check that cannot *fail* is the same defect with the sign
flipped, and it does not throw data away, it certifies it.

**The three checks were falsified, not assumed.** Two injections against the
Wormhole simulator, 2026-08-17:

* every writer aimed one slice past its target → `witness_ok` came back **0 of
  N on all four points**, `stray_writes` fired on three of them, and
  `tags_ok` **stayed at N throughout** — the device certified its own
  misdirected write, exactly as the table above says it must. The rates it
  produced (8.14 and 16.29 B/cycle) are entirely plausible and would have been
  published.
* every writer aimed at the *next bank's coordinate* → caught on one point in
  four, and the reason is worth recording: tt-sim fronts twelve Wormhole banks
  on **aliases of one modelled DRAM tile**, so a wrong coordinate at the same
  offset lands in the same memory. **The coordinate half of this check is weak
  against the simulator and strong against a card**, where the twelve banks are
  twelve independent tiles. The address half is aliasing-independent and is what
  the first injection exercises.

### What tt-sim predicts, on both parts

Four tiles, 4096 B transactions, `TT_SIM_COST_MODEL=1`:

| arch | `onechan` read | `onechan` write | ratio | why |
| --- | --- | --- | --- | --- |
| Wormhole | 23.406 | 23.327 | **0.997** | `dram.channel_serialisation` is 24 B/cycle for **both** directions there — one published per-channel figure its page states for reads and writes alike — so the model asserts a symmetric endpoint |
| Blackhole | 44.364 | 60.208 | **1.357** | a `vendor_source_derived` **read** rate of 47.08 B/cycle and **no write rate at all**, so `DramChannels.write_bytes_per_cycle` is `None`, every write claim is a no-op, and the unqueued write arm runs up against the 64 B/cycle NoC link instead |

**tt-sim predicts opposite signs on the two parts, and the Wormhole card's own
latency data leans a third way** (write dearer). Three positions; this campaign
can adjudicate only the occupancy one. Blackhole's predicted asymmetry is the
**shape of an absence** rather than of a measurement, and a card showing
Blackhole writes at the read's rate would be saying the model under-charges
there.

### A limitation the simulator has and the card does not

**tt-sim cannot currently produce a non-degenerate write fan-out control.** At
four tiles `fanchan-write` scales ×1.47 on Wormhole and ×1.30 on Blackhole,
under the ×1.5 gate, and the run correctly reports `DEGENERATE` rather than the
flat concentrated curve it was hoping for. The cause is topology: the four
workers `perfbench/run.sh` materialises sit in one router row, and a write
carries its payload *outbound* over the links between them, so the four flows
share links that four *reads* — whose data returns the other way round a
directional torus — do not. The read fan-out scales ×2.78 and ×4.00 in the same
runs. The card's 64 workers are the reason to expect the write control to move
there, and the read arm's own ×5.74 on that part is the evidence that it can.

### Running it

```bash
perfbench/dramratebench/run_card_write.sh --list        # schedule + wall estimate
perfbench/dramratebench/run_card_write.sh --preflight   # checks that cost no card time
perfbench/dramratebench/run_card_write.sh --out ~/tt_traces/dram-write-session
```

~21 min of card time, plus a cold build. It resets the board first — a heavy NoC
run on 2026-08-17 left state that silently corrupted the *data* of the next
program, with no abort and no diagnostic — and it writes its **pre-registered
predictions to the session directory before the card runs**, so what was
predicted travels with what was measured.

`check_run.py` grades a run and re-derives the read arm's control; it is
standard-library only, so it runs at the card:

```bash
perfbench/dramratebench/check_run.py --read-control          # both pinned controls
perfbench/dramratebench/check_run.py --read-control run1.csv # the one for its arch
perfbench/dramratebench/check_run.py --measured run1.csv     # the write gates
```

There is **one pinned control per architecture** and the file's own `arch=`
selects it — Wormhole from `card-sessions/2026-08-17-wormhole-B` and
`2026-08-17-wh-dramwrite`, Blackhole from `2026-08-17-bh-dramwrite`. A control
that does not apply **skips, with the reason printed**, and never fails: grading
a Blackhole run against Wormhole's pins is a category error, not a regression,
and the first time it happened it printed sixteen confident FAILs comparing 46
B/cycle against 22.

Every pinned figure is a **band**, not a point, and its width is the measured
spread across the sessions named beside it. Two Wormhole sessions read 20.344
and 22.225 B/cycle at two readers and 22.127 and 21.546 at four, so those two
bands are wide and the rest are not; the flat-band and scaling claims are pinned
separately, because they are statements about the sweep's shape and the shape
reproduces where individual points wobble. The Blackhole bands come from one
session and say so — they are point pins until a second Blackhole read arm
exists to set a spread.

`check_run_test.py` drives every one of those gates **in both directions** —
rows built to satisfy each and rows built to break it, including the doctored
files that prove a widened band still refuses a real move.

## Arguments

```
--out FILE         where to write the CSV
--bytes N          bytes each participant moves     (default 1048576, the vendor's)
--tx-bytes N       size of one noc_async_read/write (default 4096)
--repeats R        how many times to run the plan   (default 3)
--max-readers N    cap the sweep                    (default: the whole worker grid)
--readers a,b,c    participant counts to sweep      (default 1,2,4,8,12,16,24,32,48)
--arms LIST        substring filter over onechan,fanchan,samecore
--dir D            read (the default), write, or both
--sustained        the vendor's own reader counts, 1,12,48
```

`--dir` adds a `-write` copy of each selected arm; the read arm's rows keep
their unsuffixed names, so `tt_sim.perf.dram_rate_sweep` and
`card_session_verdicts.sh` — which both match `onechan` and `fanchan` exactly —
read a file with writes in it exactly as they read one without. The write
direction's three columns (`direction`, `witness_ok`, `stray_writes`) are
**appended**, and are `-1` in a read row: not "zero verified", but "the host did
not look, because for a read the device is the witness".

`--sustained` sets **only** the reader counts. 1 MiB per reader, 4096 B
transactions and three repeats are already the defaults, and the preset
deliberately does not restate them: two places to change one parameter is how
the two drift apart.
