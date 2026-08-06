# Blackhole microarchitecture, measured

Facts about Blackhole silicon established by running something on a card and
timing it, collected here because **several of them have nowhere else to go**.

The cost tables (`tt_sim/perf/unit_costs.yaml`,
`tt_sim/pe/tensix/tensix_instruction_costs.yaml`) are kept to a strict
discipline: every number in them comes from a document, and a measurement
enters as a `corroboration` field which never changes an entry's provenance
rank. The reasoning is written out at the top of `tensix_instruction_costs.yaml`
under "WHY THERE IS NO `measured` PROVENANCE" and it is right. But it has a
consequence nobody had had to face until 2026-08-05: **a quantity that no
document mentions at all cannot be recorded in the tables even when it has been
measured**, because there is no entry for a corroboration to hang off. The
instruction-cache cliff below is exactly that, and it is the first entry here
for that reason.

This file is a **complement to the tables, not a replacement for them and not a
competitor to them**. Nothing here is loaded by any code. Where a table entry
does exist, its `corroboration` field carries the same measurement and this file
points at it rather than restating it as if it were independent.

---

## Read this first: what these numbers are worth

- **One card.** Every figure below comes from a single Blackhole part, run by
  one operator. Nothing here says anything about part-to-part variation, and
  the clock state (Blackhole is DVFS-managed: 1.35 GHz busy, 0.8 GHz idle) was
  not controlled — though every measurement is in raw device cycles read off
  `RISCV_DEBUG_REG_WALL_CLOCK_L`, so the frequency cancels rather than
  entering.
- **Mostly one run per figure**, and where there are two they are two runs of
  the same binary on the same card in the same afternoon. That is a
  reproduction of the *instrument*, not of the *part*.
- **Three instruments.** `perfbench/nocbench` (§4) is unlike the other two: it
  times a whole pipelined region rather than fitting a slope, so it has no
  control-loop subtraction and no bias of the kind described next — its
  resolution is stated directly, as the spread of a family predicted constant
  (§4.1), and it refuses to report a coefficient at all unless its own positive
  controls moved.
- **Two of the three have known one-sided biases.**
  `perfbench/tensixbench` and `perfbench/riscvbench` both fit a slope over four
  block counts and subtract an empty-body control loop. The control subtraction
  over-corrects by up to `slope(control)/unroll` whenever the probe stalls
  (the loop's own instructions issue inside the stall window and cost nothing),
  which biases readings **downward** by ~0.03 cycles/instruction. Both sweeps
  compute that as a per-series `resol` term and refuse to call anything inside
  it a finding.
- **There is a floor of 1.000 cycles per instruction and it is structural.** A
  baby RISC-V core is single-issue, so nothing a kernel can time reads below
  one cycle per instruction. A measured 1.000 is consistent with a unit that
  could have gone ten times faster. This is why the "does it agree" column
  below distinguishes *confirmed* from *consistent with*.
- **It is evidence, not a datasheet.** Someone will eventually quote a number
  off this page. The claim each entry supports is written in the entry; please
  quote that rather than the figure alone.

## Read this second: all of it is Blackhole

Every measurement here was taken on Blackhole and **none of it transfers to
Wormhole**. That is not boilerplate caution. The two architectures differ in
precisely this kind of detail, repeatedly and in ways that would silently
invalidate a transferred number:

- multiply **blocks** the integer unit for two cycles on Wormhole and
  **pipelines** as EX1 + EX2 on Blackhole;
- the branch-mispredict bubble is 2 cycles on Wormhole and 4 on Blackhole;
- Blackhole has an L0 data cache and a coalescing store queue; Wormhole has
  neither;
- the `.ttinsn` fusion claim is on the Blackhole page only;
- and the standing reminder that this is not a theoretical concern:
  `UnpackRowWidth` is **16 on one architecture and 32 on the other**, a
  difference found the hard way.

Where a Wormhole run would settle something, the entry says so.

---

## How to read an entry

Each entry says which of three things it is, because they are genuinely
different claims and a reader needs to know which one is in front of them:

| Category | Meaning |
| --- | --- |
| **Agrees** | A document says a number and the measurement matches it. The document is still the authority; this is a check on it. |
| **Contradicts** | A document says a number, the measurement says a different one, *and* the "these are two different quantities" reading has been looked for and ruled out. |
| **Absent** | No document in either the ISA documentation or the vendor trees gives this quantity at all. The measurement is the only thing there is. |

---

# 1. Baby RISC-V front end

Source for all of §1: `perfbench/riscvbench`, Blackhole silicon, 2026-08-05,
`--blocks 32`, **all seven phases**, one/two/three issuing TRISCs. Banked as
`tt_sim/perf/datasets/riscvbench-blackhole.csv`; phases T, C, F, S and G passed
the benchmark's own validity gate, phase Q did not — on 9 monotonicity checks,
every one of them on a *cascade* point at n ≤ 16, which is why §1.9 and §1.10
are read off n ≥ 16 differences and nothing else — and **phase R did not
either**, on two R² checks that are both the contended store probe and are dealt
with in §3.4. A companion `--blocks 8` run
(`riscvbench-blackhole-blocks8.csv`) is banked with it, plus the two
single-phase `--gset` runs `riscvbench-blackhole-gset1.csv` and `-gset2.csv`
that carry the only measurements of a 6144- and a 7168-byte loop body. **Four
runs, one card, one operator, one day.** Reproduce the analysis with
`python3 -m tt_sim.perf.riscv_bench_sweep`.

A **fifth** run, `riscvbench-qdrain.csv`, was taken later the same day on the
same card after one line changed in phase Q's loop probe — an untimed
`ckernel::tensix_sync()`, so its bursts start drained as phase S's always did.
It is `--phase qs --variants t1`, two phases at one thread, and it exists to
score a prediction that was written down before the line was. **It bears on
§1.10's depth only.** It carries no phase R, so the live-instrument check cannot
run against it, and one thread count, so it says nothing about whether the queue
is shared — that verdict stays where it was, in the two full runs. It also means
`q_loop_addi` and `q_loop_adddmareg` in the four earlier datasets are a record
of a binary no longer in the tree, which their headers say; every other row in
them, and every other section of this document, is unaffected.

Phases **S** and **G** — the follow-ups §1.10 and §1.1 named — **have now run on
a card**, and this is the campaign they ran in. Phase S answers the question
§1.10 left open and moves its number; phase G narrows §1.1's bracket and finds a
shape inside it. Their binary bumps `RVBENCH_MAGIC` to `0x7B10CF03`; before it
was trusted, the continuity of everything *else* it measures was checked rather
than assumed, because linking more probe text has moved phase Q's absolutes
once before. Against the previous banked run: `rv_div` 33.001 against 33.001,
`rv_load_chase` 8.094 against 8.094, `rv_mul_dep` 1.985 against 1.985,
`tt_adddmareg` 2.972 against 2.972, the phase F cliff 1.251 against 1.254, and
every phase-Q loop-form point identical **to the cycle** at every n from 16 to
1024. One number moved beyond that — `rv_store_spread`, 5.217 to 5.290 — and
§3.4 is about why.

## 1.1 The instruction-footprint boundary is between 4096 and 5120 bytes, and the cost then plateaus — **absent**

**Measured.** The same `addi` in a loop body of K instructions, K from 64 to
2048 (256 bytes to 8 KiB of instruction text), cycles per instruction:

| loop body | bytes of text | cycles/instruction | phase |
| --- | --- | --- | --- |
| 64 | 256 | 0.998 | F |
| 128 | 512 | 0.998 | F |
| 256 | 1024 | 0.998 | F |
| 512 | 2048 | 0.998 | F |
| 1024 | 4096 | 1.000 | F |
| 1024 | 4096 | 1.000 | G, anchor |
| **1280** | **5120** | **1.153** | G, set 0 |
| **1536** | **6144** | **1.252** | G, set 1 |
| **1792** | **7168** | **1.252** | G, set 2 |
| **2048** | **8192** | **1.251** | F |

Flat to a thousandth of a cycle across a 16× range of footprint, then a **~25 %
step**. Identical at one, two and three issuing threads (phase F 1.251 / 1.253 /
1.253, phase G's `g_1280` 1.153 / 1.153 / 1.153), so whatever the resource is, it
is **per core and not shared** between the TRISCs. Reproduced in the `--blocks 8`
run at 1.251 and 1.150, whose flat rows read 0.991–0.995 — the flat level moves
with the baseline, the step does not move at all.

**The boundary is between 4096 and 5120 bytes, an eightfold tightening on the
octave phase F could give.** That is the bracket, and it is the whole of what
this locates.

**The shape inside it is the more interesting half, and it is not one cliff.**
The cost does not jump to its final value at 5120 bytes and stay there; it
rises to 1.153 and *then* rises again to 1.252, where it stops — 6144, 7168 and
8192 bytes agree to a thousandth of a cycle. A graded rise that plateaus is the
signature of **partial residency**: if some structure holds about 4 KiB, a
5120-byte body keeps ~80 % of it and pays the miss on the rest, and every body
past ~6 KiB is missing often enough that the amortised cost saturates. That
reading fits, and **two others fit as well** — a prefetcher whose reach is a
fixed number of bytes ahead of the loop back edge, or a fetch path whose cost
is a step function of some index bits — and nothing measured here chooses
between them.

**What the plateau does and does not license.** It is more than a bracket:
a single doubling could not distinguish a hard capacity from a gradual
degradation, and this shows the degradation is neither hard nor unbounded. But
it does not upgrade the noun. It is still a boundary in **loop-body size**, not
a cache capacity, for the reason it always was: no document in the ISA
documentation or in either vendor tree publishes an instruction-cache size or a
miss cost at all, so there is nothing for "4 KiB capacity" to be a reading *of*.
The ~0.25 cycles per instruction remains an **amortised** figure, and at 5120
bytes it is now demonstrably amortised over a body that is *partly* resident —
which is precisely why it cannot be converted into a miss cost.

**How.** `perfbench/riscvbench` phases F and G. Phase F's six probe bodies are
compiled only when phase F is selected, precisely so that the kernel measuring
instruction footprint does not also carry six other phases' text. Phase G is a
phase of its own for a *measured* reason: phase F's build already lands within a
few hundred bytes of tt-metal's kernel config buffer, and adding the three
intermediates aborts the launch with `Program size (125040) too large for kernel
config buffer (70656)`. Even the three alone do not fit one kernel, so phase G
takes three `--gset` builds, each pairing one intermediate with a `g_1024`
anchor **in the same kernel** — which is why a phase-G step is read without
crossing a build. Phase F's own bodies were left untouched, so the F rows above
are the same measurement they always were.

**Status: absent.** `riscv.instruction_fetch` gives the fetch *period* — "128-bit
reads against L1 ... each read yields four instructions, so instruction fetches
are expected to occur at most once every four cycles" — and its own note says
"instruction *cache* miss cost is not published". No cache **size** is published
either, in the ISA documentation or in ttsim or tt-metal. So there is no entry
in any table for this, which is why it is written here.

**What it does and does not say.** It locates a boundary **between 4096 and 5120
bytes of loop body**, and it says the cost past it rises once more and then
saturates. It does not say the boundary is a cache size rather than, say, a
TLB-like structure, a prefetch window or an L1 access pattern; it does not say
what a miss costs; and the plateau is consistent with partial residency of a
~4 KiB structure without being evidence *for* one, because a fixed prefetch
reach would produce the same column. **A narrower bracket is still a bracket in
loop-body size** — locating it more precisely does not weaken or strengthen that,
because what is missing is a *document*, not a finer measurement.

**What would sharpen it further.** The rise happens entirely between 4096 and
6144 bytes and is sampled at one point inside it. Bodies at 1152 and 1408
instructions (4608 and 5632 bytes) would say whether the rise is linear in
footprint — which is what partial residency of a fixed capacity predicts — or a
second step. Each is another `--gset` build, and they cost seconds on a card.

**Why it is not in the tables.** Under the discipline described at the top of
this file, a number with no documentary source has no home there — a
`corroboration` needs something to corroborate. The honest options were to
invent a table entry for it (which would make a measured number look sourced),
to leave it in a YAML comment, or to write it down somewhere that is explicitly
not a cost table. This is that place.
`arch_overrides.blackhole.riscv.instruction_fetch`'s `corroboration` field
records the *flat* part, which does correspond to a documented claim, and points
here for the step.

## 1.2 `.ttinsn` fusion does not happen — **contradicts**

**Measured.** Four probes, each 16 groups per block, cycles per group:

| probe | group contents | cycles/group | over `tt_pad` |
| --- | --- | --- | --- |
| `tt_pad` | 16 `addi` | 15.929 | — |
| `tt_fuse2` | 2 **adjacent** `.ttinsn` + 16 `addi` | 17.901 | +1.972 |
| `tt_fuse4` | 4 **adjacent** `.ttinsn` + 16 `addi` | 19.889 | +3.960 |
| `tt_spread4` | 4 `.ttinsn`, none adjacent, + 16 `addi` | 19.966 | +4.037 |

`tt_fuse4` and `tt_spread4` execute **exactly the same twenty instructions** and
differ only in adjacency, so `spread4 − fuse4` is the entire experiment. It is
**+0.077 cycles per group** where four-way fusion predicts **+3.000**. Every
`.ttinsn` word costs its own issue slot. Reproduced in all six thread slots of
that run (+0.077, −0.094, +0.002, +0.003, +0.002, +0.001) and in its
`--blocks 8` companion (+0.314 at one thread).

**The table above is the first campaign's, and the two campaigns since have
scattered about zero rather than converged on it.** The currently banked runs
read `spread4 − fuse4` at **−0.493** (`--blocks 32`) and **−0.376**
(`--blocks 8`), against a per-series resolution of 0.23–0.28 for these group
probes. The sign of a half-cycle residual on a twenty-instruction group is not a
finding either way; what is a finding is that four values spanning −0.5 to +0.3
are nowhere near **+3.000**, in four runs of two binaries. Nothing in this entry
turns on which side of zero it lands.

**Status: contradicts.**
`BlackholeA0/TensixTile/BabyRISCV/PushTensixInstruction.md`, recorded as
`riscv.ttinsn_fusion`, says "sequences of up to four adjacent .ttinsn
instructions can be fused together by the RISCV T0 / T1 / T2 instruction caches
and executed in a single cycle".

**The "two different quantities" reading was looked for and is not available.**
That reading is what resolved the only previous document/silicon conflict in
these tables (`CFG.RDCFG`, §2.2 below), so it was checked before this was
written down:

1. **Adjacency survives the compiler** — verified by disassembling the very
   kernel that ran, not assumed. The phase T ELF contains exactly 16 runs of
   four consecutive `08000000` (`ttnop`) words at four-byte stride, 16 runs of
   two, 64 singletons and one run of 64: `tt_fuse4`, `tt_fuse2`, `tt_spread4`
   and the sustained probe, with no instruction of any kind interposed.
2. **It is not a fetch-line straddle.** All 16 of `tt_fuse4`'s runs start at
   address 8 mod 16, so each does straddle a 128-bit fetch boundary 2/2 — but
   that hypothesis predicts two fused slots and `+2.000`, not `+0.077`. It is
   refuted outright by `tt_fuse2`, **all** of whose pairs are co-resident inside
   a single 128-bit read (they start at 0 or 8 mod 16, so both words are always
   in one line), and which still costs its full two cycles.
3. **The dequeue cap is not masking it.** That cap — one instruction per thread
   per cycle — is exactly why a sustained burst cannot see fusion, and it is why
   these probes are *groups*: sixteen `addi` follow each group, which is sixteen
   cycles for a queue draining at one per cycle to swallow four pushes.
4. **The measured quantity is the documented one.** The page distinguishes
   enqueue (four per cycle, via fusion) from dequeue (one per cycle). These
   probes time the **issuing core**, i.e. the enqueue side. The dequeue half of
   the same entry is confirmed — see §1.3.

**Still open, honestly.** The document attributes fusion to the instruction
*caches*; a core executing from local instruction RAM would bypass that path.
tt-metal built these kernels for execute-in-place out of L1 (the build cache
carries a `.xip.elf`), which is the cache path, but that has not been confirmed
at the hardware level.

**Consequence.** `riscv.ttinsn_fusion` keeps its numbers — nothing consumes
`max_fused`, so no simulated cycle depends on it — and now carries a
`contradiction` field. What depends on the outcome is whether a future front-end
model may assume a sub-one-cycle `.ttinsn` push. It may not.

## 1.3 Pushing a Tensix instruction costs the RISC-V core one cycle — **agrees**

**Measured**, sustained bursts of 64 back-to-back `.ttinsn` words, cycles per
instruction, at one / two / three issuing TRISCs:

| probe | backend unit | t1 | t2 | t3 |
| --- | --- | --- | --- | --- |
| `TTI_NOP` | NONE | 0.996 | 1.029 | 0.997 |
| `TTI_SFPNOP` | SFPU | 0.998 | 1.968 | 2.973 |
| `TTI_SETDMAREG` | ThCon | 0.996 | 1.969 | 2.972 |
| `TTI_ADDDMAREG` | ThCon | 2.972 | 5.987 | 8.995 |

**Status: agrees**, with `riscv.ttinsn_fusion.dequeue_per_thread_per_cycle: 1`.
One instruction per thread per cycle, read on three different backend units.

**The `TTI_NOP` row is the control that makes the rest legible**: its unit is
`NONE`, and it is the only one that does *not* scale with thread count. So what
grows when more TRISCs issue is the shared backend, not the per-thread push
path — the push really is one cycle per core regardless of who else is pushing.

**`ADDDMAREG` at 3.0 is the same number `tensixbench` measured independently the
day before** (§2.1), from a different program with a different kernel and a
different analysis harness.

## 1.4 No detectable branch penalty, and taken is slightly cheaper — **absent** (the rate) / **untested** (the size)

**Measured**, cycles per instruction, one issuing thread:

| pattern | bare | with a matched `xori` in every arm |
| --- | --- | --- |
| never taken | 0.999 | 1.995 |
| always taken | **0.963** | **1.954** |
| alternating | — | 1.975 |
| unconditional `j` | 0.962 | — |

`taken − not taken` = **−0.035**; `alternating − not taken` = **−0.020**. Both
negative, in all six thread slots of the run, and both far below one cycle. The
`--blocks 8` companion gives larger negatives (−0.135, −0.087) — a shorter
baseline, not a different device — and the previous campaign's `--blocks 32` run
gave −0.047 and −0.024. Four runs, four values between −0.14 and −0.02, none of
them a cycle.

**Status.** The two halves of this are different claims and conflating them is
the mistake worth avoiding:

- **The mispredict *size* is documented and remains untested.**
  `riscv.integer_unit.branch_mispredict_bubble: 4` (Blackhole; 2 on Wormhole),
  "which from an external perspective looks like mispredicted branches occupy
  EX1 for five cycles". Nothing here measures that. It may be exactly right.
- **The mispredict *rate* is undocumented, and it is what was measured.** No
  document describes the predictor at all. The answer is: not measurably. A
  four-cycle bubble occurring on any appreciable fraction of these branches
  could not hide inside a −0.047 delta. Whether that is because there is no
  penalty, or because a real predictor got every one of these very predictable
  patterns right (including the alternating one), this cannot separate.

**Why it matters.** `tt_sim/pe/rv/cost.py` declines to charge anything for
branches, on the stated grounds that "the number of mispredictions is unknowable
and charging every taken branch would be a fabrication". That refusal now has
evidence behind it rather than only an argument.

**Taken being *cheaper* than not-taken by ~0.05 cycles is not explained.** It is
small, it is consistent across thread slots and both runs, and it is the size of
the instrument's own known bias — so it is reported, not interpreted.

## 1.5 `divu` costs 33 cycles at this operand — **agrees**, at the far end

**Measured.** `divu 0x12345678, 3` reads **33.001** cycles per instruction
(33.004 in the companion run). `riscv.integer_unit.divide_general` documents
"between six and 33 cycles ... dependent upon the magnitude of the dividend".

**Status: agrees** — and lands on the `max` rather than the `cycles`.

**The practical consequence is a real under-charge.** `tt_sim/perf/model.py`
charges every bounded entry at its low end, so it charges **6** where this
operand costs **33**. One point is not the curve; the benchmark sweeps no other
dividends, and the sensible next run varies the dividend's magnitude to see
where on the 6–33 range real kernel operands land.

**The floor was reviewed on the strength of this and deliberately kept.** The
band is a *data* dependence, not an uncertainty band: 6 and 33 are two operands.
`_muldiv_occupancy` already holds the operands, so a magnitude-dependent charge
is mechanically easy — but **no document relates a dividend to a cycle count
between the endpoints**, so any interpolation would be an invented curve wearing
a citation. What decided it was the exposure, measured rather than assumed:
across the in-tree Blackhole replay guards (`three`, `matmulblock`, `reduce`,
`sfpumath`) one whole kernel launch executes **0–2 divides in 40,000–80,000
instructions**, and their dividends are **9–12 bits** — three orders of
magnitude below `0x12345678`, and therefore nowhere near the operand that costs
33. So the 5.5× looseness is a property of *this benchmark's dividend*, not of
the instruction as kernels use it, and it is worth under 0.15 % of a launch even
priced at the worst case. Recorded in `tt_sim/pe/rv/cost.py`'s docstring, which
is what charges the 6.

## 1.6 L1 load-to-use is 8 cycles, and the L0 d-cache does not help a pointer chase — **agrees** with the *miss* row

**Measured**, as pointer chases so that the loop body is one instruction and the
number is a load-use latency undivided:

| probe | cycles |
| --- | --- |
| L1, a 64-node ring 16 bytes apart (1 KiB working set) | **8.098** |
| the TRISC stack, at `0xFFB00F98` | **1.985** |
| independent L1 loads, four rotating destinations | **1.742** |

**Status: agrees**, but with `l1_dcache_miss: >= 8` and not with
`l1_dcache_hit: 2`. The stack row matches `core_local_data_ram: 2`; the address
is classified by `tt_sim/pe/rv/cost.classify_address`, the simulator's own
classifier, so the prediction and the simulation cannot disagree about which row
applies.

**Two independent constructions agree the latency in force is 8.** The
sustained-throughput probe reads 1.742, and the documented formula — "when the
load latency is N cycles, the throughput of sustained loads is four such loads
every N − 1 cycles" — gives (8 − 1)/4 = **1.750** at N = 8 and one-per-cycle at
N = 2. The two share no arithmetic.

**Why the L0 data cache does not hold a warm 1 KiB ring: because it is 64 bytes
long.** `BlackholeA0/TensixTile/BabyRISCV/README.md` — "the capacity of this L0
data cache is a mere 64 bytes: 4 lines of 16 bytes each" — settles the chase
outright. A 1 KiB ring is sixteen times the capacity and cannot be resident
under any organisation, so `l1_dcache_miss` is the row it reaches *by
construction*. This was not looked up until the measurement forced the question,
and the capacity is now recorded as `riscv.l0_data_cache` in
`tt_sim/perf/unit_costs.yaml` and read by `riscv_bench_sweep` to pick the row a
probe's access pattern actually reaches — and, since 2026-08-06, by
`tt_sim/pe/rv/cost.py`, whose per-core L0 line-tag model charges the miss row
to any L1 load whose line is not resident (so this chase pays 8 per load in
simulation too).

The *rate* half of that reading is charged too, since 2026-08-06: the docs'
formula is consumed as four in-flight load slots each held N − 1 cycles, so a
stream of miss-row loads runs at 1.750 cycles per load in simulation against
this 1.742 on the card (`docs/plans/cost-model.md`, "The load queue").

**What is still not established** is `rv_load_indep`. Its four addresses span
exactly four 16-byte lines — *exactly* the published capacity — and it still
reads the miss row's sustained rate. Associativity, indexing, replacement policy
and the exact rate of the documented "~0.8 % chance of flushing the entire
cache" are all unpublished, so a working set sitting on the capacity is not
predicted either way and its prediction has deliberately **not** been moved to
match the reading.

## 1.7 Store coalescing is worth 5× — **agrees**

**Measured**, cycles per store:

| probe | cycles |
| --- | --- |
| stores to four different 16-byte blocks in rotation | **5.290** |
| stores 4 bytes apart inside **one** 16-byte block | **0.999** |
| stores to the stack (core-local RAM) | **0.999** |

**Status: agrees** with `store_throughput.l1_period_cycles: 5`,
`l1_coalesced_period_cycles: 1` and `other_regions_period_cycles: 1`. The second
probe is written to hit the documented predicate exactly — "the same 16-byte
aligned region of L1, with start addresses within +/-4 of each other" — and the
first to miss it. **5.2× apart.**

**This is the cleanest architectural discriminator in the set.** Wormhole
publishes no coalescing store queue, so the same two probes are predicted
*identical* there. Running them on a Wormhole card is the cheapest
cross-architecture check available and nothing has done it.

**One caveat in the honest direction:** the spread probe reads *above* the
documented 5 by more than the instrument's resolution — the only RISC-V entry
where that happens. "At most one store every five cycles" is a floor and the
measurement respects it; the extra 0.2–0.3 cycles is unexplained and is not the
control over-subtraction, which biases the other way.

**It is also the only phase-R probe that moves between runs**, at 5.217, 5.217,
5.290 and 5.223 across four. That is the subject of §3.4, and it is the reason
the currently banked `--blocks 32` run's phase R does not pass its own gate.

## 1.8 One RV32IM instruction per cycle, with forwarding — **agrees**

`addi` chains read 0.999 whether every instruction depends on the previous one's
result or none of them do. Independent multiplies read 0.999 and a dependent
multiply chain reads **1.985** — an occupancy of one and a latency of two, which
is "exactly one cycle in EX1, and then exactly one cycle in EX2" measured
instead of read.

**Status: agrees**, with `riscv.issue.instructions_per_cycle: 1`,
`integer_unit.alu_forwarding: true`, `multiply: 1` and `multiply_ex2: 1`.

**This is the probe whose answer differs between the two chips for a documented
reason**, which makes it the best single check that the benchmark is measuring
the architecture it thinks it is: on Wormhole the multiply *blocks* the integer
unit, so both readings would be 2.0. Nothing has run it there.

## 1.9 The core outruns the Tensix backend at one thread and not at three — **absent**

**Measured**, `perfbench/riscvbench` phase Q: a burst of *n* `ADDDMAREG`s, timed
twice — once plain, and once with `ckernel::tensix_sync()` inside the timed
region so that it cannot return until the pipe has drained. Cycles per
instruction over the loop form's n = 16…1024:

| issuing threads | plain (the core's view) | with drain (the work's own rate) | knee |
| --- | --- | --- | --- |
| 1 | 2.953 | **3.000** | between n = 64 and 128 |
| 2 | 5.982 / 5.985 | **5.997 / 5.996** | between n = 32 and 64 |
| 3 | 9.008 / 9.020 / 9.022 | **9.006 / 9.002 / 9.003** | between n = 32 and 64 |

The issuing core runs ahead of the backend at short bursts and is
back-pressured beyond the knee, and **where the knee falls depends on the
thread count and not only on the burst length**: at one thread the marginal
cost of one more instruction steps 1.06 → 1.84 → 3.33 across n = 32, 64, 128
against a drained 3.000; at two and three threads it has stepped by n = 64.
Three issuing threads share one backend, so each thread's service rate is a
third of it and the queue fills three times sooner.

**The t1 plain column moved slightly after the drain fix** and the conclusion
does not. `riscvbench-qdrain.csv` (§1.10) reads 2.936 rather than 2.953 with the
same drained 3.000, and its marginal steps 1.25 → 1.84 → 3.00 rather than
1.06 → 1.84 → 3.33 — the knee is in the same place and it now arrives *at* the
drained rate instead of overshooting it, which is what removing the previous
burst's residue predicts. The two- and three-thread rows are from the full runs
and are untouched: the drain run has one thread.

**Status: absent.** Nothing in the ISA documentation or either vendor tree gives
a Tensix instruction queue depth.

**The drained column is the reliable half** and is worth trusting on its own.
`q_adddmareg_sync` reads 2.995–3.000 / 5.920–6.009 / 8.958–9.018 across six
thread slots and five runs, and phase S's independent `s_co_sync` reads
3.000 / 5.965–5.976 / 8.917–8.988 from a different probe body in the same
launch — a third and fourth independent route to ThCon's 3-cycle occupancy
(§2.1), sharing no arithmetic with either of the two burst measurements. The
plain column carries the full noise of a single-shot cold burst; see §3.2.

## 1.10 The Tensix instruction queue is **one per core**, and holds ~31–32 entries — **absent**

**Two claims, and they rest on different runs.** The first — that the queue is
private to each TRISC rather than shared between them — is a *ratio* between
thread counts, it is what phase S was built to measure, and it comes from the
two full three-thread runs of 2026-08-05. The second — the depth in entries — is
a *level* at **one** issuing thread, and it comes from those runs plus a third,
`riscvbench-qdrain.csv`, which is `--phase qs --variants t1` and therefore has
no bearing on the first claim whatsoever. The two constructions that measure the
level disagreed by ~5–7 entries until that third run; they now agree to 0.7, for
a pre-declared reason, and the settling is
[below](#the-magnitude-settles-and-a-pre-declared-prediction-is-what-settled-it).

### It is per-thread, and this was pre-declared

**Measured**, `perfbench/riscvbench` phase S at one, two and three *saturated*
issuing threads. The verdict is `D` at k threads over `D` at one:

| slot | `--blocks 32` | `--blocks 8` | per-thread predicts | shared predicts |
| --- | --- | --- | --- | --- |
| t2 vs t1 | 30 / 31 = **0.97×** | 29 / 31 = **0.95×** | 1.00× | 0.50× |
| t3 vs t1 | 33 / 31 = **1.06×** | 33 / 31 = **1.07×** | 1.00× | 0.33× |
| spinning control | 0.91× / 1.00× | 0.91× / 1.00× | 1.00× | 1.00× |

Two runs, four discriminating comparisons, all four within 7 % of 1.00 and none
within a factor of two of 1/k. **Each baby core has its own Tensix instruction
queue.** The construction is the one this entry was rewritten around after the
obvious one — a second thread that only spins — was retracted: only a
*saturated* second thread occupies queue entries (a spinning one holds none
under either hypothesis, and a slow one holds none by Little's law), and the
backend bandwidth a saturated one necessarily steals is measured in the same
slot by `s_co_sync` and divided back out.

**The spinning control behaves as both hypotheses require**, at 0.91× and 1.00×
— i.e. a second core merely being awake and fetching out of the same L1 does not
change what the issuing core can have in flight. That is what makes the co-issuing
ratio a statement about the *queue* and not about the tile.

**Status: absent.** Nothing in the ISA documentation or either vendor tree says
whether this queue exists as a queue, let alone how deep it is or whether it is
replicated.

**This verdict is untouched by everything below.** It is a ratio between one,
two and three issuing threads, measured in the two full runs; the run that
settled the *depth* is one thread and contributes nothing to it. If a later
reader finds only the drain run in front of them, the sharing question is not
answered by it.

### The magnitude settles, and a pre-declared prediction is what settled it

Phase S at one issuing thread reads **31 entries**. The pre-declaration said it
should land *above* phase Q's figure by roughly the reference burst's own
occupancy, ~8 entries. Phase Q on the same run read **~16**, so the direction was
confirmed — phase Q's figure was a lower bound, as declared — but the gap was
15.3 entries where 8.0 was predicted, and **5.3 of it survived the correction
arithmetic**. The diagnosis, the pre-declared test of it and the result are all
below; the short version is that phase S drained its pipe before every timed
burst and phase Q did not.

**The residual, as it stood.** The sweep prints the reconciliation (`Do the two
burst forms agree about the depth?`). Pre-drain, from
`riscvbench-blackhole.csv`:

| form | n_ref | p | S | backlog | `backlog/S` | levelled | run-ahead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase Q, 16-instruction block | 16 | 1.123 | 3.000 | 47 | **15.7** | 25.7 | 25.3 |
| phase S, 4-instruction block | 4 | 1.508 | 3.000 | 87 | 29.0 | **31.0** | 32.3 |

`backlog/S` is what phase Q's read-out prints; `levelled` adds `n_ref·(1 − p/S)`,
the reference burst's own occupancy, and is what phase S's read-out prints. 8.0
of the 15.3-entry gap was that term — 10.0 at phase Q's n = 16 against 2.0 at
phase S's n = 4 — precisely the correction the pre-declaration named, at
precisely the size it named. The other 5.3 was not the correction arithmetic: a
third estimator, run-ahead `(n·S + c − plain[n])/S`, which uses no `_sync`
probe, no reference burst and no `tensix_sync()` cost, agreed with `levelled`
*within* each form and put the two forms **7.0 entries apart**.

**It lived in one column, and entirely in `plain`.** The two forms' saturated
`sync − plain` was 79 cycles (phase Q) and 100 (phase S); `3n − plain[n]` was a
flat 64 for phase Q from n = 128 and a flat 85 for phase S — but phase Q's own
value was **85 at n = 64**, phase S's exactly, stepping down to 64 thereafter.
The `_sync` probes were identical to the cycle at all six shared burst lengths.
So it was a flat 21-cycle tax on phase Q's undrained `plain` probe once the
queue was full, and nowhere else.

**The explanation, and it was tested rather than argued.** Phase S runs an
untimed `tensix_sync()` immediately before every `t0`; phase Q did not. A
saturated backend never idles, so whatever the previous burst point left in the
queue is added to `plain` permanently rather than absorbed — a deficit that
grows with the *preceding* point and then plateaus once that point is itself
saturated. One line was added to `QLOOPPROBE`, and **the raw cycles it would
produce, plus three ways it would be refuted, were written down before the line
was** ([the plan doc](plans/riscv-front-end-benchmark.md), "The untimed drain,
pre-declared before it was written", and the section that scores it).

**What came back**, `riscvbench-qdrain.csv`, `--phase qs --variants t1
--blocks 32`, same card, `q_loop_adddmareg` at t1:

| n | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-drain | 31 | 48 | 107 | 320 | 704 | 1472 | 3008 |
| **predicted** | 31 | 48 | 107 | **299** | **683** | **1451** | **2987** |
| measured | 28 | 48 | 107 | **299** | **683** | **1451** | **2987** |

Four for four at the four burst lengths the prediction moved, to the cycle: a
21-cycle fall at every n ≥ 128 and zero below it. `q_loop_adddmareg_sync`, which
is not edited, is identical to the cycle to both banked runs. `q_loop_addi`, the
control that pushes no Tensix instruction and must therefore not move, is
identical at six of its seven points and **3–4 cycles low at n = 16** — reported
here rather than glossed, because n = 16 is the backlog's reference point; it is
3 cycles against a 17-cycle `q_ctrl` spread for one raw point, and the two banked
runs already disagree by a cycle there. It cost one entry off the levelled
figure, and the estimator that does not touch n = 16 landed dead on.

And the derived quantities, all pre-declared: `sync − plain` saturates at **100**
cycles (was 79), which is phase S's value; `3n − plain[n]` is a **flat 85** from
n = 64 to 1024 (it stepped 85 → 64), which is phase S's value; the backlog
flattens at a constant **65** cycles from n = 64 and the marginal cost reaches
exactly **3.00** against a drained 3.000. The reconciliation:

| form | n_ref | p | S | backlog | `backlog/S` | levelled | run-ahead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase Q, 16-instruction block | 16 | 1.127 | 3.000 | 65 | 21.7 | **31.7** | **32.3** |
| phase S, 4-instruction block | 4 | 1.502 | 3.000 | 87 | 29.0 | **31.0** | **32.3** |

`RECONCILED: 0.7 entries`, against ~7 before. **The run-ahead estimator shares no
term with the other two, uses neither `_sync` probe nor either reference burst,
and reads 32.3 for both forms.** That independence is what makes this a
confirmation and not a fit.

**So the number to quote is:**

> **~31–32 entries at one issuing thread**, per core. The estimators that use
> every measured term cluster at 31.0, 31.7, 32.3 and 32.3; the deficit-only
> reading that drops the timed region's measured 12-cycle fixed cost gives 28.3,
> which is the low end of the pre-declared ~28–32 bracket. Not ~26–31, which was
> a bracket built out of two forms that disagreed, and not ~14–16, which was two
> corrections short.

**What is retracted, in order.** "~14–16 instructions" was never wrong about
what it measured; it dropped the reference burst's occupancy, which the same run
had the data to compute, and it inherited the missing drain. The `~14` and `~16`
were not even two readings of the part: the asymptotic `sync − plain` is 79
cycles in all four pre-drain runs, and the 14-to-16 spread is entirely the
±6-cycle single-shot scatter of the *one* raw reference point at n = 16 — which
is why phase S moved its reference to n = 4. "~26–31" then bracketed two forms
that disagreed for a reason nobody had found yet; the reason has been found and
tested, so the bracket collapses rather than narrowing.

**What is still one card and one thread.** The depth is a level read at **one**
issuing thread on **one** part, now on three runs of two binaries. The two
constructions agreeing is a statement about the instrument's arithmetic, not a
second part. What would still sharpen it: (a) a phase-S variant with a
16-instruction block and an n = 4 reference, which separates "the block size"
from "the missing drain" — they were confounded before this run and the drain
run does not disentangle them either, it only shows the drain was worth 21
cycles; (b) a second card, which is what would make any of this a fact about
Blackhole rather than about this part.

### The multi-thread phase-Q slots, and why phase S exists

At two and three issuing threads phase Q's drained rate is 2× and 3× higher, so
the *same* queue holds a backlog that many times smaller in cycles — and it
disappears under the instrument's own noise. The multi-thread slots of the
banked runs give backlogs of +8, +28, −2, −8, +32, −17, −19, −50, −10 and −45
cycles against a `q_ctrl` spread of 13–19 cycles for a single raw point. **A
backlog of work in flight cannot be negative**; those are the subtraction
passing through zero. The read-out reports a depth only when the backlog clears
twice the control's measured spread *and* has stopped growing, which is true of
the one-thread slot and of nothing else.

**That is structural rather than unlucky, and it is the third thing phase S
fixed.** At n = 16 a shared queue's per-thread share can be *smaller* than
`16·(1 − p/S)`, so the reference point saturates and the subtraction removes the
entire signal. Phase S references n = 4, and its multi-thread slots resolve
depths of 30, 32, 32, 33 and 33 entries against a repeatability of 4–9 cycles
measured in the slot by a byte-identical repeat of the probe.

**What the loop form bought, and where it is verified.** The straight-line
cascade could not be extended to n = 1024 without becoming 4 KiB of instruction
text — the range §1.1 finds a fetch boundary in — and a phase-Q burst runs once,
cold, with its own fetch inside the timed region, so a fetch cost that grows
with *n* is indistinguishable from back-pressure. The loop form's footprint is
64 bytes (phase Q) or 16 bytes (phase S) at every burst length. That the change
of form did not change the *rate* is measured where the two overlap: at one
thread cascade and loop agree to **+0.062 cycles per instruction** over
n = 16…128. What appeared to change was the run-ahead, by the 21 cycles above —
and that turned out to be the missing drain rather than the form. The cascade is
not drained either and was not edited, and in the drain run it reads 101 and 319
cycles at n = 64 and 128, **identical to the cycle** to the pre-drain run, with
its `_sync` companion identical at all eight points; only the loop form moved.
The two forms consequently now differ by −0.161 cycles per instruction over
n = 16…128, which is the residue the cascade still carries and the loop no
longer does. So the loop form bought a fixed footprint and cost nothing, which
is what it was for.

---

# 2. Tensix backend

Source for all of §2: `perfbench/tensixbench`, Blackhole silicon, 2026-08-04,
`--blocks 32 --iters 64 --dvalid-once`, banked as
`tt_sim/perf/datasets/tensixbench-blackhole.csv` (+ four `--src-format` runs and
one deliberately-confounded control). Full write-up in
[`docs/plans/tensix-cost-benchmark.md`](plans/tensix-cost-benchmark.md).
**One run per figure, one card, one operator**, except where noted. Reproduce
with `python3 -m tt_sim.perf.tensix_bench_sweep`.

## 2.1 The `ADDDMAREG` family costs 3 cycles — **agrees**

`ADDDMAREG`, `CMPDMAREG`, `MULDMAREG` and `SHIFTDMAREG` all read **2.973**
cycles per instruction from one issuing thread, scaling a clean 3.0× to ~8.97 at
three. The documented occupancy is a range of "3 or 4"; this is the low end, and
nothing observed selects between the two.

**Status: agrees.** These are the only entries in the whole Tensix table that a
single-thread measurement could ever have tested — everything else has an
occupancy of 1, which is the instrument's own floor.

**Confirmed by a second instrument.** `perfbench/riscvbench` phase T read
`ADDDMAREG` at 2.972 / 5.987 / 8.995 the following day (§1.3), and its phase Q
reached 3.000 a third way from a drain-timed burst. Different program, different
kernel, different harness. **Three runs, one card, two days.** `SUBDMAREG` and
`BITWOPDMAREG` share the same table entry and have **never** been measured.

## 2.2 `RDCFG` issues one per cycle, not "≥ 2" — **contradicts** a misreading, **agrees** with the document

**Measured.** 0.998 cycles per instruction from one thread, 1.961 from two,
2.952 from three — 3.0× scaling, i.e. one shared unit accepting one instruction
per cycle across all three threads.

**Status.** This entry is here as the worked example of why "silicon contradicts
the docs" is usually the wrong conclusion. The table's occupancy field used to
read `>= 2`, copied from the ISA doc's **latency** column. The measurement
disagreed with that — and the resolution was not that the document is wrong. The
document also gives a *throughput* row ("issue at most one of these per cycle"),
which is a 1-cycle occupancy, and Blackhole states the same thing as a number.
**Two true facts about different quantities**, and the table entry was the thing
that was wrong.

The documented **latency** of ≥ 2 — the time until the destination GPR holds the
value — is **unchanged and still untested**: the benchmark runs no dependent
chain, so no latency is observable to it at all.

Reproduced across two tracked datasets from the same card, because the dvalid
setup that confounded the matrix probes touches nothing the config unit does.
**Two runs, one card.**

## 2.3 `MVMUL` has two regimes ~6× apart — **absent**, and the most misleadable number here

**This is the entry most likely to be quoted wrong**, so it is stated as a pair
and never as a single number:

| how the instruction is issued | cycles | what does this |
| --- | --- | --- |
| **back to back out of the MOP expander** | **~1.07** | `ttreplay` — the replay buffer feeds the FPU one per cycle |
| **individually, as a `.ttinsn` word** | **~6.0** | each op pays the Wait Gate's `SrcA[bank].AllowedClient` check |

**Measured, both ways, from the same dataset.** Phase A issues each op as an
individual `.ttinsn` word by construction and reads `MVMUL` 5.989, `ELWADD`
5.976, `ELWMUL` 5.974, scaling a clean 3.0× to 18.035 at three threads. Phase B
times a real `matmul_tiles` inner loop at three fidelities, where the only thing
that changes is the number of MVMULs: 34.92 / 52.47 / 86.12 cycles per call at
16 / 32 / 64 MVMULs, so the marginal cost is (86.12 − 34.92)/48 = **1.067**.

**Status: absent** — no document describes the split. `MatrixUnit.md`'s
throughput row gives the one-per-cycle figure and the Wait Gate is documented
separately; that the two combine into a 6× difference in observed cost is not
stated anywhere.

**Which number applies to real code: ~1.07.** Every non-experimental LLK path
that issues these opcodes goes through the MOP — checked in the vendor tree, not
assumed: `llk_math_matmul.h` and `llk_math_reduce.h` record their `TTI_MVMUL`s
into a replay buffer and replay them from a `ckernel_template`, and
`llk_math_eltwise_binary.h` drives `ELWADD`/`ELWMUL` entirely through
`ckernel_template`. The one exception is
`tt_llk_blackhole/llk_lib/experimental/llk_math_matmul_custom_no_mop.h`, which is
named for issuing them inline and is the single real kernel shape that would pay
the Wait-Gate figure. The cost tables charge 1, which is the MOP number.

**How this was nearly got wrong.** The first hardware run read phase A's MVMUL
at 0.998, and that was an artefact of a `SETDVALID` setup which violated the
instruction's own precondition — at three threads it handed the Matrix Unit a
bank it already owned. It took a second run to retract, and the retraction moved
the single-thread column too. The confounded run is kept deliberately, as
`tensixbench-blackhole-dvalid-per-thread.csv`, because it is the control that
demonstrates the artefact.

## 2.4 No resolvable source-data-format effect — **absent** (and a real null)

**Measured.** Four runs of the same binary differing only in the source data
format the Matrix Unit decodes:

| probe | bf16 | fp32 | fp16 | tf32 | spread |
| --- | --- | --- | --- | --- | --- |
| `MVMUL` | 5.988 | 5.988 | 5.988 | 5.989 | 0.001 |
| `ELWADD` | 5.974 | 5.974 | 5.974 | 5.975 | 0.001 |
| `ELWMUL` | 5.973 | 5.973 | 5.973 | 5.974 | 0.001 |

0.001 cycles of spread against a fitted resolution of 0.036–0.052: the
instrument could have seen an effect forty times smaller than the one it was
looking for, and saw none.

**Status: absent.** The cost tables have no data-format axis at all, so this
corroborates a *structural* choice rather than a number.

**What makes it a finding rather than a blind spot: the null control.** `bf16`
and `fp32` decode to the *same* `SrcAStyle` in `MVMUL.md`, `ELWADD.md` and
`ELWMUL.md`, so if the functional models are right they **cannot** differ. The
sweep predicts that in advance and labels the pair before any number is read.
They came out at exactly **0.000** on all three probes. An instrument that shows
no difference everywhere might simply be blind to the axis; this one was asked a
question whose answer was known beforehand and got it right.

**Two caveats travel with this and quoting the headline without them overstates
it.** (1) This is the **Wait-Gate regime** (§2.3), not the MOP-issued one the
tables charge, so a format effect confined to the MOP path would not appear here
at all — and there is currently no probe shape that would test it. (2) **One run
per format, on one part**: four runs total, four points of one axis rather than
four samples of anything.

## 2.5 The fidelity arithmetic is right — **agrees**

**Measured.** A real `matmul_tiles` inner loop at three fidelities, differenced
so that the circular-buffer waits, the two tile unpacks and the semaphore
handshake cancel:

| step | measured | predicted | residual |
| --- | --- | --- | --- |
| LoFi → HiFi2 | 17.55 | 16.00 | +1.55 |
| HiFi2 → HiFi4 | 33.65 | 32.00 | +1.65 |

**Status: agrees** with `MATH.fidelity_phases.mvmuls_per_tile = 16`, which was
the most load-bearing *derived* number in the Tensix table and which nothing had
ever checked. Had it been 8 the steps would have been 8 and 16; had it been 32,
32 and 64. Neither is within twice the residual.

The residual is +1.6 cycles on each step, flat rather than growing — the same
~1.07 per MVMUL the marginal arithmetic gives in §2.3, so the two readings of
the same phase agree with each other. What the extra 7 % is (the FPU, the
`ADDR_MOD` walk, something in the loop) is not separable from three points.

---

# 3. About the instruments themselves

Facts about the benchmarks rather than about the chip, kept here because they
bound everything above.

## 3.1 `--blocks 8` is below the instrument's minimum — for the *slope* phases

The `--blocks 8` companion run had **six of its seven phases refused** by
`riscvbench`'s own validity gate, and it is banked
(`riscvbench-blackhole-blocks8.csv`) partly for that reason.

Five of the seven R² failures are the **`loop_overhead` control itself**,
fitting to R² = 0.9894 and 0.9657 against a gate of 0.99. At `--blocks 8` the
four fitted points span n = 8…32; the empty loop costs ~2 cycles a block, so the
whole fitted range is ~48 cycles and the per-launch fixed cost plus a few cycles
of jitter is a visible fraction of it. The control is subtracted from every
probe in its phase, so a control that does not fit is a phase that cannot be
read whatever the probes did — and that is the *whole* of phases F's and G's
verdicts here, neither of which has a probe failure of its own. Most of the rest
are the **taken-branch** probes (`c_t`, `c_jal`), whose slope over the shortest
available baseline is barely above the control's — R² falls out first exactly
where the signal is smallest.

**And yet the numbers agree**, to a few thousandths of a cycle, wherever both
runs are readable: `rv_div` 33.004 against 33.001, `rv_load_chase` 8.097 against
8.094, `rv_store_spread` 5.223 against 5.290, the phase F step at 1.251 in both
and phase G's `g_1280` at 1.150 against 1.153. So the gate refused a run whose
answers were right — which is the correct behaviour for a gate that cannot know
that in advance, and a useful reminder that **agreement is not validity**.

**Minimum usable block count: above 8, at or below 32.** Nothing narrows it
further; the runbook asks for 32.

**None of this touches phases Q or S, and the distinction is load-bearing**
because §1.10's numbers come from a run whose header says six of seven phases
failed — and **phase S passed the gate in that run**, reproducing the primary's
per-thread verdict (0.95× and 1.07×) and its 31-entry one-thread depth.
`--blocks` sets the slope phases' four fitted points and *nothing else*: the
host labels a phase-Q point `n0 << k` (1, 2, 4 … 128 for the cascade, 16, 32 …
1024 for the loop form) where it labels a slope point `base_blocks * (k + 1)`,
and the kernel's `QPROBE`, `QLOOPPROBE`, `SPROBE_CO` and `SPROBE_SOLO` macros
never read `base_blocks` at all — only `PROBE`, the slope macro, does.
`riscvbench`'s validity gate agrees: it skips every phase-Q row before the R²
check, so phase Q's verdict is monotonicity of its own probes and carries
nothing from the slope phases. The measurement of that independence is that the
`--blocks 8` and `--blocks 32` runs reproduce every loop-form phase-Q point to
within seven cycles, and every phase-S depth to one entry, at a quarter the
block count.

## 3.2 Phase Q measures single shots and phase Q knows it

Every other phase of `riscvbench` fits a slope over four block counts, so the
fixed cost of the clock reads, the barrier and the surrounding call cancels
exactly. **Phase Q does not**: each burst length is one timed execution of code
that runs once, cold. Four consequences, all of which the read-out now states:

- Its own control, `q_ctrl`, spans **6 to 25 cycles** across the eight burst
  lengths — reproducibly (both full runs agree point for point) and
  **non-monotonically**. The design had assumed this control grew with the burst
  index and subtracted it point by point, which produced *negative* net costs at
  small *n*. It does not grow: all seven of the cascade's `if` tests are
  evaluated at every burst length. The subtraction has been retracted.
- Even probes that must structurally read ~1.0 cycles per instruction come out
  **bracketing** it — `q_nop` at 1.75 and `q_setdmareg` at 1.22 — because each
  burst's own instruction fetch is inside its timed region and nothing averages
  it out.
- Only differences between probes that **share a body** are stable, which is
  why §1.9's answer is read off the `q_adddmareg` / `q_adddmareg_sync` pair and
  not off either one's absolute rate.
- And **that spread is a threshold, not a caveat.** §1.10's backlog is
  `(sync[n] − plain[n]) − (sync[16] − plain[16])`: a difference of two
  differences, so four raw single-shot points. `q_ctrl` measures what one of
  them can be wrong by, so the read-out refuses to divide a backlog by a
  service rate unless it clears **two** of those spreads. Before 2026-08-05 it
  did not, and it printed "~1 INSTRUCTIONS in flight" from an 8-cycle backlog
  in one slot and reported three negative ones in others. The threshold is
  taken from each run's own measured spread rather than fixed in cycles,
  because it is the run's own statement about how noisy it was.
- **And the drain run put a number on how far that spread reaches.** It was
  pre-declared, from a deterministic tt-sim A/B of the two kernels, that adding
  the untimed `tensix_sync()` would move `q_ctrl` by ~1 cycle through register
  pressure across `kernel_main` and nothing else. On silicon `q_ctrl` moved at
  four of its eight points (13→6 at n = 1, 22→23, 13→17, 17→18) while the
  unedited cascade probe it is subtracted from reads identically to the cycle at
  n = 64 and 128. That is the same fact from the other side: at small *n* this
  phase's raw points carry more scatter than any effect it is looking for, which
  is why the depth is read off n ≥ 16 differences over a long span.

## 3.3 A 1.000 is only readable next to something that is not 1.000

Both benchmarks have the same failure mode: **a run in which every probe reads
exactly 1.000 is simultaneously the expected answer and the signature of an
instrument that measured nothing.** `riscvbench` answers it with four probes
that have a documented cost above one cycle — `rv_div` 33.001, `rv_load_chase`
8.094, `rv_store_spread` 5.290, `rv_mul_dep` 1.985, all four above 1.0 on the
currently banked run — and refuses the whole report if none of them clears it.
Every 1.000 elsewhere on this page is a finding on the strength of those four,
and on nothing else.

## 3.4 The contended store probe is the only unstable reading in phase R

**The currently banked `--blocks 32` run does not pass its own phase-R gate**,
which matters because §3.3's live-instrument check lives in phase R. Both
failures are the same probe: `rv_store_spread` at **t2 thread 0** (R² = 0.9883)
and **t3 thread 0** (R² = 0.9831), against a gate of 0.99. Its t3 readings also
moved, 10.407/10.414 in the previous campaign against 10.818/10.827 here.

**It is run-to-run variance in a contended probe, and it is not the new phases.**
Three things establish that, in order of how much they settle:

1. **The failures are single-point outliers, not slope changes.** t2 thread 0's
   four fitted points are 13757, 22518, 36849, 45041 where a line through the
   2nd and 4th predicts ~11.0k and ~33.8k: two points are 8–25 % high and the
   rest are on the line. t3 thread 0 has one high point of four. A slope that
   had genuinely moved would keep R² at 1.0000, which is exactly what the
   previous campaign's same slots did.
2. **Phase R's kernel does not carry phase S's or phase G's text.** Each phase's
   probe bodies are compiled only when that phase is selected and phase R runs
   in its own launches, so nothing was added to what phase R executes. The
   measurement of that: every phase-R probe that is *not* contended is unchanged
   to the cycle — `loop_overhead` slope 1.944 and 2.000, `rv_div` 2114.000,
   `rv_load_chase` 519.981 against 519.978.
3. **What the new binary did move is an address, and it is a benign one.** The
   result buffer gained ten probe slots, so the allocator placed the L1 scratch
   at `0x0017D440` instead of `0x0017D800` and the TRISC stack at `0xFFB00FA8`
   instead of `0xFFB00F98`. Both new addresses are 64-byte aligned, the
   per-thread scratch stride is unchanged at 2048 bytes, and the two probes
   whose answers depend on the documented 16-byte predicate — `rv_store_coalesce`
   and `rv_store_stack` — still read 0.999.

**Why this probe and no other.** `rv_store_spread` at t2/t3 is the only phase-R
probe in which three cores hammer L1 stores simultaneously; every other probe is
per-core work. And its t3 column was *already* strange before this run:
5.967 / 10.407 / 10.414, i.e. one thread wins the store port and two lose. Which
thread wins is not something the benchmark controls, so a run in which the split
lands differently part-way through the four fitted points produces exactly this
non-linearity.

**Nothing was changed on the strength of it**, and that is deliberate. The gate
refusing a phase whose contended slots are non-linear is the gate working; the
single-thread column, which is the only one the residual ladder and the
live-instrument check read, is intact and passes 4 of 4. What *would* be wrong
is quoting a t2 or t3 `rv_store_spread` number as a per-instruction cost, which
nothing does — the sweep drops every t2/t3 series from the ladder by
construction, precisely because a contended measurement is a shared resource's
throughput and not a cost.

---

# 4. The NoC

Source for all of §4: `perfbench/nocbench`, Blackhole silicon, 2026-08-05,
planned by `tt_sim/perf/noc_congestion_plan.py` against this card's own
`--dump-grid` capture and read back by `tt_sim/perf/noc_congestion_sweep.py`.
**Two runs, one card, one operator, one day**, and they are two runs of
*different plans* rather than a repetition — which claim rests on which is
stated in every entry below, and where the two overlap they are quoted as two
numbers and never averaged:

| dataset | what it is |
| --- | --- |
| `tt_sim/perf/datasets/nocbench-blackhole.csv` | **the main run.** 87 flows over 55 runs, all six experiments. The controls live here, so this is the only one of the two that can carry a verdict, and it carries `RESULT: CONGESTION MEASURED` |
| `tt_sim/perf/datasets/nocbench-blackhole-sizes.csv` | **the size sweep.** 96 flows over 48 runs, `shared` only, six transaction sizes × eight shared-link counts. No controls by construction, so its own verdict is `INVALID` and it is only ever read next to the main run |
| `tt_sim/perf/datasets/nocbench-grid-blackhole.csv` | the card's core map, from a probe kernel reading each core's own `NOC_NODE_ID` |

Reproduce with `python3 -m tt_sim.perf.noc_congestion_sweep` (the main run is
the default) or `--measured …-sizes.csv`.

**The card is harvested, and this is the campaign that found out.** Its
addressed worker columns are `{1..7, 10..14}` — a legal *subset* of an
unharvested Blackhole's physical `{1..7, 10..16}`, which is why the previous
campaign's planner concluded the dump was already physical. It is not: the
kernels' own `NOC_NODE_ID` puts the columns at `{1..6, 11..16}`, so physical 7
and 10 are the disabled pair and anything addressed at x ≥ 7 sits further right
than an unharvested map would put it. Every number in this section is on the
corrected geometry, with **zero coordinate mismatches and zero invariant
complaints** in both files — which the previous campaign, on the same
experiment, could not say.

## 4.1 A round trip costs 8.4–8.8 cycles per hop, and the NoC really is a directional torus — **agrees**

**Measured** (main run). One flow, one master, everything fixed but the
subordinate's coordinate, 64 × 4096 B per point. The round trip falls on
exactly three levels and the fit through them is

```
region cycles = 4363.4 + 8.80 * round_trip_hops     (r2 1.00)
```

| family | n | round-trip hops | mean region | spread |
| --- | --- | --- | --- | --- |
| col | 8 | 12 | 4470.2 | ±11.5 |
| row | 8 | 17 | 4511.0 | ±16.5 |
| diag | 8 | 29 | 4619.0 | ±16.5 |

**Status: agrees**, with the ISA docs' "the latency of each hop is at least 9
cycles". 8.80 is under 9 and the entry says *at least*, so this is the low end
of a bound rather than a contradiction. The previous campaign, on the wrong
geometry, fitted 8.38 through the same three levels; the two lines are not as
far apart as their slopes look, because they agree to ~5 cycles at every level
(4469/4474 at 12, 4513/4516 at 17, 4619/4617 at 29) and a 5-cycle disagreement
over a 17-hop lever is what moves a slope by 0.4. **Per transaction the two
campaigns differ by 0.08 cycles.** Quote the bracket, 8.4–8.8, not either
endpoint.

**The structural half is the more interesting one and it is confirmed.** Every
hop count in `tt_sim` assumes both NoCs are *directional* tori — a packet only
ever travels in the increasing direction of that NoC's coordinates and wraps
past the edge rather than turning round, so a request and its reply go the same
way round the ring and a round trip costs `grid_x`, `grid_y` or `grid_x +
grid_y` hops *whatever the coordinates*. That predicts three levels and
flatness within each. Both hold: within the row family the fit against
*forward* distance is slope −1.64 over an 8-point sweep, i.e. flat, and a
shortest-path NoC would have shown a rising line there instead.

**The noise floor everything else is read against.** The row family is
predicted constant, so its spread *is* this harness's resolution: **0.5
cycles/tx**.

## 4.2 Two flows sharing one router-to-router link each pay one extra transaction's link occupancy — **absent**

This is §4's result, and the shape of it matters more than any single number.

**Measured** (the size sweep; the main run's two columns are quoted beside it
below). Two flows, a searched placement in which *every* leg-pair link overlap
except the payload one is held at zero, flow A byte-identical at every point,
flow B's hop count fixed at 10 forward and 29 round trip. The only thing that
moves is how many router-to-router links the two payloads share.

| transaction | occupancy = bytes ÷ 64 | 0 shared links | 1 shared link | delta | delta ÷ occupancy | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 64 B | 1 | 39.9 | 39.8 | −0.1 | −0.09 | 1.00 |
| 512 B | 8 | 39.8 | 39.8 | 0.0 | 0.00 | 1.00 |
| 2048 B | 32 | 40.2 | 68.3 | **28.1** | 0.88 | 1.70 |
| 4096 B | 64 | 72.2 | 135.4 | **63.1** | 0.99 | 1.87 |
| 8192 B | 128 | 137.8 | 262.3 | **124.5** | 0.97 | 1.90 |
| 16384 B | 256 | 268.4 | 518.8 | **250.4** | 0.98 | 1.93 |

(cycles per transaction; 64 transactions per flow.)

**The claim is the fourth column: above a threshold, the cost of a second flow
on a shared link is one transaction's worth of that link's occupancy**, where
the occupancy is `bytes ÷ 64` because a Blackhole flit is 512 bits and a link
carries one per cycle. Four sizes, ratios 0.88 / 0.99 / 0.97 / 0.98.

**And below the threshold it is exactly nothing**, which is the negative
control the design predicted and is what makes the positive readings mean
something. A 64 B packet holds a link for 1 cycle and a 512 B packet for 8,
against a ~39.8-cycle issue loop: the link is 2 % busy, two flows almost never
want it at once, and nothing shows. **The regime boundary is between 512 B and
2048 B** — i.e. where occupancy approaches the issue interval — and the ratio
then climbs toward perfect halving as the fixed cost amortises: 1.70 → 1.87 →
1.90 → 1.93.

**Then it stops.** From 2 to 7 shared links the reading does not move at any
size. At 16 KiB the seven points from 1 to 7 span 15.8 cycles against a
250-cycle step — 6 % — so the whole effect is at the *first* shared link. A
naive regression through all eight points gives +22.5 cycles per shared link at
r² 0.37, and **that slope describes a machine that does not exist**: it would
charge two shared links twice what one costs, and the card charges the same.
The sweep names this shape `SATURATING` for exactly this reason, and prints the
per-share means so the step is visible.

**Status: absent.** No document in the ISA documentation or either vendor tree
gives a congestion figure at all; the NoC pages say congestion "can negatively
impact latency" and quantify nothing. What *is* documented is the thing this
turns out to be — `noc.hops.router_to_router.throughput_flits_per_cycle: 1`,
"one flit (256 bits) per cycle per axis" — which is why the measurement is
recorded as a `corroboration` on `arch_overrides.blackhole.noc` rather than as
a number of its own.

**And that is now what the simulator charges** (2026-08-05), which changes what
this entry can claim in one specific way and in no other. tt-sim holds one
free-cycle watermark per router-to-router link and charges each packet the
occupancy of every link it crosses, so the same `isa_doc` flit rate is spent
where the measurement says it is spent. Run through the same harness, the
simulator now reads +59.4 cycles/tx at 4096 B against an occupancy of 64,
0.0 at 512 B, and flat beyond the first shared link — the shape of the table
above, on a machine with no fitted parameter in it. **What that does *not* do is
turn the numbers above into sourced quantities.** They remain one part, one
operator, one day, and the reason they are safe to have influenced anything is
precisely that they did not: nothing here entered the tables, and the
`corroboration` field is where they stayed. See
[`docs/plans/cost-model.md`](plans/cost-model.md) for the build and its gate.

**Two independent confirmations in the same file, by different routes.**

* **The `readport` control**, which shares no geometry with the above. Two
  masters *reading* 64 × 8192 B from one subordinate: both response streams
  leave that subordinate's single NIU. Alone 138.6 cycles/tx; together 266.0 —
  **1.92×**, and 266.0 ≈ 2 × 128 + 10. Same arithmetic, different resource (an
  NIU injection port, not a router link), and it **PASSES on silicon** where
  the control it replaced read 1.00 and was blind to its own subject.
* **The endpoint ladder.** Six 4096 B flows into one subordinate come back at
  136.9 / 202.3 / 267.6 / 333.0 / 398.6 / 399.0 cycles per transaction — evenly
  spaced by ~65.5 where 4096 ÷ 64 is 64, in a stable rank order. N flows into
  one endpoint serialise at one occupancy each. This reproduces the previous
  campaign's ladder (137 / 203 / 268 / 333 / 399 / 399) to within a cycle,
  which is the only figure in §4 measured twice on two different plans.

**One residual, stated rather than smoothed.** The 4–7 shared-link half sits
above the 1–3 half by 8.8 cycles at 16 KiB (1.7 %), 2.5 at 8 KiB and −1.1 at
4 KiB. It is a step and not a trend, it falls exactly where flow B's route
starts wrapping past the torus edge through the non-worker columns, and no
recorded covariate accounts for it. It does not touch §4.2's claim, which is
measured between 0 and 1 shared links inside the non-wrapping half at every
size.

## 4.3 Different virtual channels do not avoid the split; sharing one costs a further 2 % — **agrees** on the mechanism, **absent** on the size

The ISA docs name exactly one congestion mechanism — "if the two packets have
the same virtual circuit number, then one packet will wait for the other" — and
every flow in §4.2 was on one channel, so §4.2 alone could not tell VC
arbitration from link occupancy. This separates them.

**Measured** (main run). Two writers, 64 × 16384 B, payloads sharing exactly
one router-to-router link. Flow A pinned at tt-metal's own
`NOC_UNICAST_WRITE_VC` (1); flow B swept 0–3, so `vc1` is "both writers on one
channel" and the other three are its control.

| flow B's VC | cycles/tx | |
| --- | --- | --- |
| 0 | 520.2 | different channels |
| 1 | 530.4 | **same channel as flow A** |
| 2 | 519.9 | different channels |
| 3 | 519.9 | different channels |

**The answer is no.** All four points are ~1.94× the 268.4 that the same
transaction costs with no shared link, so **putting the two writers on
different virtual channels does not avoid the halving**. The three
different-channel points agree to 0.3 cycles — inside the 0.5-cycle noise
floor — and the same-channel point stands 10.5 cycles above them, twenty times
the floor.

**Status.** The mechanism **agrees** with the document: sharing a channel does
cost extra, measurably and reproducibly across a three-point control. Its
**size is absent** from every source, and it is small: **2.0 %**. So the
documented mechanism is real and is worth one fiftieth of the effect; the other
98 % is the link's bandwidth being divided, which is not congestion in the
docs' sense at all but the published link rate arriving where the model was not
spending it.

This is also the experiment the previous campaign could not run: its
bidirectional form **hung the card** and no plan may emit one again. The
unidirectional redesign answers the question without it.

## 4.4 One tile's wall clock keeps its own epoch, by a constant, across resets — **absent**

**Measured.** Two of the main run's 25 multi-flow runs, and six of the size
sweep's 48, reported a timed-region overlap of 0.00 — "the flows barely
coincided", which is the harness's own refusal to call a flat reading a result.
All eight are the same experiment point and all eight have the **same core** as
flow 1: addressed (7, 2), SoC-physical **(11, 2)**, logical (6, 0).

It is not a placement, a rendezvous hazard or a harvested-grid artefact. It is
that this tile's `RISCV_DEBUG_REG_WALL_CLOCK` counts from a **different
epoch**:

| | |
| --- | --- |
| offset from every other tile | **+1,143,914,613 ± 4 cycles** (≈0.85 s at 1.35 GHz) |
| runs it reproduces in | 8 — 2 in the main run, 6 in the size sweep |
| spread of those 8 | **7 cycles** |
| implied relative rate error | < 3 × 10⁻⁸ over 2.3 × 10⁸ cycles of elapsed time |
| tiles used as masters | 19, across physical columns 1–6 and 11–15 |
| tiles showing it | **1** |

Four things establish it and one is deliberately left open.

1. **It is an epoch, not a delay.** The eight implied offsets agree to 7 cycles.
   A flow that genuinely started late cannot start late by the same number of
   cycles eight times. And the offset **exceeds the whole session** — the
   file's own stamps span 3.0 × 10⁸ cycles — so it is not a delay that could
   have happened inside the run at all.
2. **The two clocks run at the same rate.** Same-core durations are untouched
   and the offset does not drift across 2.3 × 10⁸ cycles of elapsed time. One
   counter, started earlier; not a second clock domain.
3. **It survives a device open.** The two files are separate program
   invocations whose stamp ranges overlap, so the counters were re-based
   between them — and the offset comes back identical to ±4 cycles. Reset-release
   jitter does not reproduce to four cycles across two resets.
4. **The stamps are 32 bits and one of the affected runs caught the wrap.** The
   size sweep's 16 KiB point stamps `t0 = 34,990,120` where its partner reads
   `3,186,042,806`; add the offset and 2³² lands on the partner exactly. That
   is the cleanest single confirmation in this entry, because it is the offset
   predicting a value rather than being fitted to one.
5. **The cause is not established, and cannot be from this data.** Whether the
   tile was released from reset earlier, or its counter is pre-loaded, or
   something in the ARC boot sequence touches that column, needs the card and a
   different experiment.

**Status: absent**, and it is the kind of absent this file exists for: no
document says the per-tile wall clocks share an epoch, and nothing in the vendor
tree aligns them. tt-metal's own profiler assumes they do — `syncDeviceHost`
runs the sync kernel on **one** logical core per device and `setShift` applies
that single answer device-wide (`tt_metal/impl/profiler/tt_metal_profiler.cpp`)
— so on this card a profiler timestamp from physical (11, 2) is out by 0.85 s
relative to every other core, silently. That is a claim about what the code
does, not a bug report: nothing here has run the profiler on this part.

**What it does *not* touch.** Every coefficient in §4 is fitted to `t1 − t0`
measured **on one core**, which no cross-tile epoch can reach; the affected
points' own cycle counts are indistinguishable from their neighbours (the
16 KiB share-4 point reads 527 cycles/tx where share-5, -6 and -7 read 527 and
the uncontended baseline is 268 — the flows plainly contended). Only the
cross-core overlap *check* was fooled, and `noc_congestion_sweep`'s
`clock_skew_report` now detects and subtracts it. That correction is
deliberately not per-run — it would then assume the overlap it is used to check
— but per **core**, and only when the offset reproduces across runs *and* is
too large to have been a delay. With it, the main run's median overlap is 1.00,
nothing is below 0.5, and the verdict is `CONGESTION MEASURED`.

---

# Where these came from

| | |
| --- | --- |
| `perfbench/riscvbench/` | The RISC-V front-end benchmark, and its operator runbook |
| `perfbench/tensixbench/` | The Tensix instruction-cost benchmark |
| `perfbench/nocbench/` | The NoC congestion harness, and its operator runbook |
| `tt_sim/perf/datasets/` | Every dataset quoted here, each with its provenance in its own `#` header |
| `tt_sim/perf/riscv_bench_sweep.py` | The §1 analysis; `python3 -m tt_sim.perf.riscv_bench_sweep` |
| `tt_sim/perf/tensix_bench_sweep.py` | The §2 analysis; `python3 -m tt_sim.perf.tensix_bench_sweep` |
| `tt_sim/perf/noc_congestion_sweep.py` | The §4 analysis; `python3 -m tt_sim.perf.noc_congestion_sweep` |
| [`docs/plans/riscv-front-end-benchmark.md`](plans/riscv-front-end-benchmark.md) | §1's design, method and full running record |
| [`docs/plans/tensix-cost-benchmark.md`](plans/tensix-cost-benchmark.md) | §2's, likewise |
| [`docs/plans/cost-model.md`](plans/cost-model.md) | What any of it is allowed to change in the simulator |
