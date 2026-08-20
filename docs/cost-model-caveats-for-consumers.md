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

## Compute-kernel init ordering: a partial oracle, not a blind one

**The claim this answers:** "both simulators are blind to init ordering". Measured,
that is too strong. tt-sim catches the init-ordering faults that change the
configuration its units read, and misses the ones that do not — three of each in
the six perturbations below. Which half you are in is predictable, and stated
below.

Measured live against tt-sim over the wire bridge (tt-metal 0.74, Wormhole,
`examples/four` = `binary_op_init_common` + `add_tiles_init` + `add_tiles` +
`pack_tile`; `examples/five` = `init_sfpu` + `add_int_tile_init` + an SFPU op).
Each row is one edit to the compute kernel, rebuilt through the normal JIT path:

| perturbation | tt-sim result |
| --- | --- |
| four: `copy_tile_init` inserted between `add_tiles_init` and `add_tiles` | **caught** — 254 of 256 elements wrong |
| four: `add_tiles_init` omitted | **caught** — `[UNIT STALL]` names the blocked unpacker at ~16k cycles, `[DEADLOCK]` every 50k thereafter (fixed 2026-08-19; it was silent before) |
| four: `add_tiles_init` called *before* `binary_op_init_common` | **missed** — passes, 256/256 correct |
| five: `init_sfpu` omitted | **caught** — fails |
| five: `add_int_tile_init` omitted | **missed** — passes |
| five: the two inits swapped | **missed** — passes |

**Why the split falls exactly there.** An `*_init()` is, at the hardware level, a
burst of writes to Tensix backend configuration, address modifiers and the MOP
(macro-op) expander configuration. tt-sim models all three. So a reordering that
leaves a *different* value in one of them at the moment an op reads it changes
tt-sim's answer, and a reordering that leaves the *same* values does not.

The first caught row is the sharp case. `mop_cfg[0..8]` is a single per-thread
register file at `TENSIX_MOP_CFG_BASE`; `ckernel_template::program()` writes all
nine unconditionally with no dirty check, and `run()` is `static` and
argument-free — it executes whatever was programmed last. `copy_tile_init`
programs the unpacker macro for A only, where `add_tiles_init` had programmed it
for A and B, so the following `add_tiles` unpacks no SrcB. That is architecturally
guaranteed last-writer-wins, tt-sim reproduces it, and it is the mechanism behind
this fault class's deadlocks: with no SrcB dvalid the matrix unit waits forever
and the consumer blocks on `cb_reserve_back`. The model is pinned by
`tt_sim/pe/tensix/mop_clobber_test.py`, in both directions — a clobber changes
the expansion, an unrelated write to the same bank does not — so the "caught"
rows above cannot rot into "missed" without a test going red.

The missed rows are the ones where the config state after the sequence is
*identical* either way. `add_tiles_init` before `binary_op_init_common` is an
ERROR under tt-metal's own contract — `compute_kernel_hw_startup.h` says the
startup call "should be called exactly once at the very beginning of the kernel,
before any operation-specific initialization functions", and the LLK sanitizer's
state machine encodes it as `"First transition must be INITIAL -> CONFIGURED"`.
But the violation is about *when the MMIO writes land relative to unit idleness*
("almost exclusively require the idle state of the execution units that should be
configured ... unsafe to call this function in the middle of a kernel execution"),
and tt-sim's configuration writes land instantly. It has no schedule in which the
write is late, so it always sees the lucky one. That is the same missing guarantee
recorded in `ROADMAP.md` §6 as "config-write ordering", seen from the other end.

**No guard is offered for the missed rows, deliberately.** The rule that is
violated is stated over the LLK C++ API — *which* `*_init` was called for *which*
op — and tt-sim observes only the Tensix instruction stream. In the missed rows
that stream contains the same instructions in a different order and leaves the
same state, so no instruction-level predicate separates them from a correct
program without inferring the API-level intent behind each write. A guard resting
on that inference would fire on correct kernels, and would be switched off.

**Use the oracle that sits at the right level.** tt-metal ships one:
`TT_METAL_LLK_SANITIZER=1` (with `TT_METAL_LLK_SANITIZER_ERROR`, on by default)
turns on `llk::san`, a per-thread state machine over
`INITIAL -> CONFIGURED -> INITIALIZED[Op] -> EXECUTED[Op]` that matches an init's
arguments against its op's and reports the source line of both. It is off by
default and its coverage is incomplete in a way that matters here: the tracked
`Operation` enum (`tt_metal/tt-llk/common/sanitizer/types.h`) currently lists only
`UnpackA`, `UnpackABMatmul`, `UnpackUntilize`, `EltwiseUnaryDatacopy`, `Matmul`,
`Pack` and `PackUntilize` — **no eltwise-binary entry and no SFPU entries at all**.
An FPU-binary-then-SFPU init sequence is therefore outside what it checks today.
Adding those entries is a far cheaper fix than a simulator guard, and it is the
only place the check can be made without guessing.

**It runs against tt-sim — but only from 2026-08-20.** The sanitizer reports
through DPRINT, and DPRINT could not start on tt-sim before that date: every
run with it enabled aborted in `TT_THROW: Timed out writing init magic`
(`dprint_server.cpp:147`) because tt-sim's stand-in cores zero-filled host
reads, so the host could never read back the magic word it had just written.
If you tried the sanitizer against tt-sim and it would not start, that was
this, and it is fixed; see
[`docs/running-tt-metal-on-the-simulator.md`](running-tt-metal-on-the-simulator.md)
§4.7, including what DPRINT costs in wall time.

## `pack_untilize_dest` on Blackhole: it works, but the init must precede the math

**The claim this answers:** "`pack_untilize_dest` cannot run on Blackhole".
Measured, that is wrong in general and right in one specific, diagnosable case.

`pack_untilize_dest` runs on Blackhole today and is bit-exact against ttsim:
`optests/diff.sh untilize` (whose op 2 *is* `copy_tile` → `pack_untilize_dest`)
passes on Blackhole, 1536 elements, and `optests/packuntilizeinit early` — the
same K=2 bf16 GEMM the compiler team reported, with the init in the position
tt-metal requires — returns `errors=0 of 1024` on both simulators with an
identical result hex. The `DST_ACCESS_STRIDED_MODE` Dst read that
`pack_untilize_dest` needs is modelled (`tt_sim/pe/tensix/backends/packer.py`,
pinned by `tt_sim/pe/tensix/pack_strided_test.py`).

**What does not run is the late-init form, and it does not run on hardware's
terms either.** On Blackhole — and only on Blackhole — `pack_untilize_dest_init`
expands to a MATH-thread arm that Wormhole does not have
(`tt_metal/hw/inc/api/compute/pack_untilize.h`, `#ifdef ARCH_BLACKHOLE`):
`llk_math_reconfig_remap(true)`, which sets `DEST_ACCESS_CFG_remap_addrs` and
`DEST_ACCESS_CFG_swizzle_32b`. Those two bits are what make the packer's Dst
read stride 16 rows; without them the pack has no defined addressing at all.
And `_llk_math_reconfig_remap_` opens with

```c
tensix_sync();
while (semaphore_read(semaphore::MATH_PACK) > 0) {};  // wait for previous packs
```

so if the init is issued after `tile_regs_commit()`, MATH blocks on the
semaphore that only the *following* pack will release, while PACK — already past
`tile_regs_wait()` — walks straight into the untilize PACR. The config write
never lands before the pack that needs it. Traced in tt-sim: in the early form
MATH issues two `RMWCIB`s to `DEST_ACCESS_CFG` (`0x2` then `0x3`) before the
PACR; in the late form the register is never touched and TRISC1 is still
spinning at `0xb594` (`lw a5, 0x24(a2)` — PC-buffer word 9, `MATH_PACK`) when
PACK issues the PACR.

**Both simulators refuse it, for the same reason, on the same instruction.**

| | on the late-init PACR |
| --- | --- |
| ttsim (vendor) | `UnimplementedFunctionality: tensix_pacr: dst_access_mode=1 swizzle_32b=0` |
| tt-sim | `NotImplementedError: PACR DST_ACCESS_STRIDED_MODE with DEST_ACCESS_CFG remap_addrs=0 swizzle_32b=0 …` |

ttsim's source states the rule outright: *"We currently require strided mode to
be tied to the swizzle_32b and remap_addrs features"* (`TENSIX_EXECUTE_PACR`).
The public ISA documentation cannot settle it either way — **BlackholeA0 has no
PACR page and no Packers chapter at all**; `BlackholeA0/…/Dst.md` says only that
the two bits "also affect how packers address `Dst`", and links to pages that do
not exist in the tree. So the strided address sequence *without* the remap is
neither specified nor referenced anywhere, and tt-sim refuses rather than invent
one — an invented address sequence would return plausible, silently wrong data,
which is the one outcome worse than stopping.

**What to do about it, measured.** Put `pack_untilize_dest_init` before the
math, which is where tt-metal's own API contract puts every `*_init`. If a
generator must emit it late, tt-metal ships the escape hatch: configure the
remap once up front, then spell the late call

```c
pack_untilize_dest_init<1, 1, false /*narrow_row*/, TILE_C_DIM,
                        false /*dense*/, false /*configure_remap*/>(cb_out);
```

— that last template parameter exists for exactly this ("Pass
`configure_remap = false` only when the caller has already configured BH DEST
remap"). Both are verified rather than suggested:
`optests/packuntilizeinit remapearly` keeps the reported late-init shape,
hoisting only the MATH arm, and returns `errors=0 of 1024` with the same result
hex as `early` on tt-sim *and* on ttsim, on Blackhole.

**This is not a cost-model caveat and not a cycle-count risk**; it is listed here
because it is the shape a consumer meets first: a GEMM that passes on Wormhole
and stops dead on Blackhole, with an error naming a Tensix mode rather than the
line of kernel source that has to move. The Wormhole side of the same op test is
a genuine tt-sim defect and was fixed (`optests/packuntilizeinit`, the Wait Gate
`STALLWAIT`/`SEMWAIT` fix); the Blackhole side is not the same bug wearing a
different hat.

## What is *not* on this page

Known-unreached functional edges — conditions the simulator does not model
because no kernel has reached them — are in `ROADMAP.md` §6. They are not
cost-model caveats: they are places the simulator would raise rather than
quietly mis-model, which is a different risk and one you would notice.
