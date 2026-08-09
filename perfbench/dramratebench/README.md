# dramratebench — does a DRAM channel's read rate grow with the number of readers?

`N` Tensix tiles each read 1 MiB from DRAM, simultaneously, with `N` swept. Three
arms differ in **which endpoint** those reads land on and in nothing else.

```bash
export TT_METAL_HOME=/path/to/your/built/tt-metal
perfbench/dramratebench/run_card.sh          # on a card
./perfbench/run.sh dramratebench -- --bytes 8192 --tx-bytes 512 --readers 1,2
```

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

## What it can never do

**Nothing here is provenance.** A number this program produces is a measurement
on one part and can enter `unit_costs.yaml` only as `corroboration`.

That is doubly true on Blackhole, which has **no published DRAM tile page at
all** — no per-channel bandwidth, no address map, no channel count. Nothing
measured on a Blackhole card can make `dram.bandwidth` chargeable there, and a
clean run must not be reported as though it could. What Blackhole *can* do is
validate the **shape**: an aggregate that refuses to grow with readers while a
fanned-out control grows freely is endpoint occupancy, whatever the absolute
number turns out to be.

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
| Blackhole, two tiles | `NO ENDPOINT BOUND` | the endpoint queue is inert *by construction* there: `ArchProfile.dram_gddr_channel_size` is `None`, Blackhole publishes no per-channel bandwidth, so `DramChannels.bytes_per_cycle` is `None` and every `claim()` is a no-op. Both arms scale ×2.00 exactly. There is nothing to contend for |
| Wormhole, four tiles, `TT_SIM_COST_MODEL=1` | `ENDPOINT BOUND`, ratio 0.56 | the term is live there, and this is the run that shows the probe reaching it |

The session's `--sim` path widens `TT_SIM_TENSIX_COORDS` to two tiles for this
probe's own run — scoped to it, so no later probe is slowed — because the reader
count is the axis the whole thing sweeps and at one reader the barrier is never
even exercised.

None of it is alarming at the card, and all of it is the documented answer.

## Telling a good run from a degenerate one

In order, and each can only ever fail the run:

1. `tags_ok == num_readers` in every row.
2. `max_barrier_spins > 0` in some multi-reader point.
3. the `fanchan` aggregate GREW between the narrowest and widest reader count.
4. only then is `onechan_scale / fanchan_scale` read, and only that ratio.

`perfbench/card_session_verdicts.sh`'s `dram_verdict` applies exactly these four,
in that order, and `card_session_verdicts_test.sh` runs them against synthetic
files that fail each one.

## Arguments

```
--out FILE         where to write the CSV
--bytes N          bytes each reader pulls          (default 1048576, the vendor's)
--tx-bytes N       size of one noc_async_read       (default 4096)
--repeats R        how many times to run the plan   (default 3)
--max-readers N    cap the sweep                    (default: the whole worker grid)
--readers a,b,c    reader counts to sweep           (default 1,2,4,8,12,16,24,32,48)
--arms LIST        substring filter over onechan,fanchan,samecore
```
