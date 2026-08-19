# Cost-model caveats for consumers

**Audience:** anyone weighing a codegen or scheduling decision against tt-sim's
cycle model — the compiler team first among them. It is the companion to
[`docs/trace-schema.md`](trace-schema.md): that document says what the outputs
*mean*, this one says where they are *known to be wrong*, and by how much.

Everything here is measured against real silicon and permanent unless the entry
says otherwise. A caveat that is merely suspected does not belong on this page.

## The model is a floor, by rule

Every bound in the cost tables is charged at its **low end**. Where the
documentation gives a range, tt-sim charges the minimum. So the model
**under-predicts**, monotonically, and never over-predicts — which makes a
tt-sim cycle count a *lower bound* on hardware, not an estimate of it.

This is deliberate: it keeps un-sourced numbers out of the tables, and it means
a disagreement with silicon always has a known sign. It also means the size of
the gap varies by instruction, and the one below is the largest known.

## Integer divide is under-charged by up to 5x

**tt-sim charges 6 cycles. Silicon delivers up to ~33.**

Measured on a Blackhole p150, three repeats, bit-identical across all three
(`perfbench/card-sessions/2026-08-18-bh-retirebench/`):

| dividend | significant bits | cycles per divide, measured |
| --- | --- | --- |
| `0xFFF` | 12 | **14.04** |
| `0x12345678` | 29 | **33.10** |

Two further instruments agree at the top end: `perfbench/riscvbench` reads
33.001 / 33.004 on Blackhole and 33.03 on a Wormhole card.

**What this means for codegen.** Any decision weighed against tt-sim's cost
model under-weights integer division, by a factor rising with the magnitude of
the dividend. If a transformation trades divides against almost anything else,
tt-sim will under-state the cost of keeping them.

**It only bites for runtime divisors.** `tt_metal/hw/inc/internal/mod_div_lib.h`
is a table of reciprocal multiply-shift helpers, and SFPI's GCC tuning
strength-reduces every compile-time-constant divisor, so a hardware divide
reaches a kernel only when the divisor is not known at compile time. A
constant-divisor division is not affected by this caveat at all.

**It cannot be fixed, and that is established rather than assumed.** The ISA
documentation gives a 6–33 cycle range and the words "dependent upon the
magnitude of the dividend", and nothing else: no radix, no per-bit rule, no
formula, no worked example. Both architecture doc trees were searched
exhaustively — 602 files, `quotient` appears zero times and `dividend` four,
being the same two sentences once per architecture. Every vendor tree yielded
one hit, `mod_div_lib.h:129`, which is strictly weaker still. A
magnitude-dependent term is therefore unlicensable at every rank of tt-sim's
provenance ladder, and the tables forbid un-sourced entries.

The obvious curves are excluded rather than merely unpinned: `cycles = bits + k`
needs a single `k` and gets 2.018 at 12 bits against 4.043 at 29, and the affine
law through both points reads 1.71 cycles at one bit (below the documented floor
of six) and 36.40 at 32 (above the documented cap of 33).

**Tracked upstream.** [`docs/upstream/divide-cycle-cap-report.md`](upstream/divide-cycle-cap-report.md)
is a drafted report asking Tenstorrent to state the rule. If it is answered,
this caveat lifts and the model gains a real term; until then, design around it.

## Mechanism-level attribution is validated for NoC work only

tt-sim's per-mechanism cycle attribution (rung 4) has three legs. Only one is
validated against silicon on both architectures:

| leg | status |
| --- | --- |
| NoC-bound | **validated on both**, 0.2–9.3 % against a 25 % bar |
| RV-bound | Blackhole only; **fails** its bar, entirely on the divide above |
| Tensix mechanism | **cannot be built** from the counters this hardware exposes |

So a NoC-bound attribution is corroborated; a per-mechanism split of Tensix
work is the model's own opinion, not a checked one. The RV-bound leg's failure
has the single cause above and is 2.49 % once the divide zones are removed.

## Energy figures are ranking-level and weak

`perfbench/energybench` fits energy coefficients that pass thirteen gates on two
architectures — but against a null model that takes energy as proportional to
simulated cycles, the fit is worth **one rank swap in nine on Wormhole and
nothing at all on Blackhole**. The ratios are better than the null, and only on
Wormhole.

Absolute joules are out of reach of the instrument (board-level power at ~1 Hz).
Do not use these coefficients to compare anything but workloads of the same
shape, and never quote them as energy costs.

## DRAM page-to-bank assignment: the correctness half is visible, the host-side bandwidth half is not

This one splits, and the two halves have opposite answers. Read both.

**A wrong-bank page lands wrong in tt-sim too, and you will see it.** tt-metal
spreads an interleaved buffer's pages round-robin over the device's DRAM banks
— 12 on Wormhole (6 channels x 2 `dram_view`s), 8 on Blackhole (8 channels x 1)
— and the host's scatter and the kernel's gather compute the landing site
*independently*. The host does it in `WriteToDeviceInterleavedContiguous`
(`bank = page % num_banks`, `addr = base + (page / num_banks) *
aligned_page_size`, sent to that bank's own worker core); the kernel does it in
`InterleavedAddrGen<true>::get_noc_addr`, from the JIT define `NUM_DRAM_BANKS`
and the `dram_bank_to_noc_xy` / `bank_to_dram_offset` tables the host wrote into
L1 at init. Nothing cross-checks the two. Since `bank(byte) = (byte_offset /
page_size) % num_banks`, changing the page size moves every byte to a different
bank — while leaving every address a legal address, so the failure is silently
wrong data rather than a fault.

tt-sim reproduces that exactly. It holds each bank as separate storage at its
own NoC coordinate, and it executes the kernel's real address arithmetic rather
than a model of it, so a page computed into the wrong bank reads the wrong
bytes here for the same reason it does on silicon. `examples/banks` is the
standing check: 24 pages walked with `InterleavedAddrGen`, correct on both
architectures. Give that kernel a page size the host did not allocate with and
Wormhole tt-sim returns `errors=3072 of 6144` — no crash, no `TT_FATAL`, clean
device close, the signature the bug wears on hardware. The corruption begins at
page 12, exactly the first bank wrap, which is where the page size first enters
the address at all.

The reason to say this loudly is that it is easy to have the opposite
impression, and the impression is well-founded elsewhere: a functional
simulator that flattens DRAM to one store passes every such test, and tt-metal's
own emulation runner carries a comment
(`tt_metal/impl/emulation/emulated_program_runner.cpp`) recording that without
the pow2/non-pow2 bank defines, "non-pow2 bank counts (12 on WH-N150) silently
fall through to a 0-bit shift and every page lands in bank 0". tt-sim is not on
that path — it drives the real JIT build — but nothing in the example suite
demonstrated it until `banks` existed, because every other example allocates a
single-page buffer and reaches it with `get_noc_addr_from_bank_id<true>(0, ...)`.

**Host-to-device transfer bandwidth is not modelled at all.** If your symptom is
a *rate* on the host link — "our buffers cap host-to-device at 1.89 GB/s against
a 5.81 measured ceiling" — tt-sim has nothing to say about it, and will not
grow anything to say. Host traffic arrives over UMD's simulation wire protocol
as `WRITE`/`READ` messages that are applied to device memory immediately; there
is no PCIe tile (`tt_sim/bridge/cores.py` stubs it to zeros) and no host-DMA
term anywhere in `tt_sim/perf/`. A tt-sim cycle count covers device-side work
only, and a host transfer costs zero of them. Do not read a tt-sim run as
evidence about that ceiling in either direction.

The device-side sibling of the same question *is* modelled, and needs nothing
new: DRAM traffic concentrated on one bank contends where traffic spread over
many does not, because each channel carries an occupancy
(`DramChannels` in `tt_sim/device/tiles.py`, at the rate
`dram.channel_serialisation.bytes_per_cycle`). So a kernel whose buffer collapsed
into one bank will show the serialisation in its *own* DRAM reads and writes.
That is the axis `perfbench/dramratebench` and `tt_sim/perf/dram_rate_sweep.py`
already exercise.

**Bank-internal timing is not modelled and has no route to being modelled.**
Bank conflicts, row hit/miss, precharge and refresh windows are absent from the
public ISA documentation for both architectures — the word "bank" never appears
in a DRAM context in either tree, and Blackhole has no DRAM tile page at all.
Any such term would have to be invented or measured, and measurement is
corroboration here, never provenance. Treat DRAM as a flat per-channel pipe at
the published rate.

## What is *not* on this page

Known-unreached functional edges — conditions the simulator does not model
because no kernel has reached them — are in `ROADMAP.md` §6. They are not
cost-model caveats: they are places the simulator would raise rather than
quietly mis-model, which is a different risk and one you would notice.
