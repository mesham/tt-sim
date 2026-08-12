// Shared between the host program (tensixbench.cpp) and both compute kernels.
// The kernels stamp the header words into the result buffer so the host can
// validate that the binary it built and the kernel that ran agree, rather than
// trusting a constant duplicated in two places.
#pragma once

// Bump on any layout change. Bumped 0x7B10CE02 -> 0x7B10CE03 on 2026-08-09 when
// slots 20 and 21 (the RDCFG latency difference) widened TTBENCH_NUM_PROBES and
// so the per-thread stride through the result buffer.
//
// THIS DOES NOT INVALIDATE THE TRACKED DATASETS, and that was established
// before the bump rather than after. The magic is a wire check between a host
// binary and the kernel it just built (`tensixbench.cpp` compares it against the
// stamp in the result buffer, twice, fatally); it is written into every CSV's
// `#` header as metadata, but NOTHING reads it back. `tt_sim/perf/
// tensix_bench_sweep.read_csv` parses that line into a `meta` dict and no code
// path anywhere consults `meta["magic"]` -- verified by grep and by re-reading
// every tracked dataset with the header rewritten to a bumped value, which
// yields byte-identical rows. The in-tree precedent is stronger still:
// `tt_sim/perf/riscv_bench_sweep_test.py` has always fed its fixture the STALE
// magic 0x7B10CF01 against a current 0x7B10CF03 and passes.
//
// What would have broken the datasets is renumbering, not the magic: `probe_id`
// is the CSV's own column. Slots 20 and 21 are APPENDED for that reason.
//
// Bumped 0x7B10CE03 -> 0x7B10CE04 on 2026-08-12 when slots 22-25 (the RDCFG
// latency difference taken against the condition bit that actually observes the
// Configuration Unit) widened TTBENCH_NUM_PROBES again. Same reasoning, and
// slots 0-21 keep their numbers: 20 and 21 are still the C10/TRISC_CFG pair,
// now kept as the falsification control rather than as the measurement.
//
// Bumped 0x7B10CE04 -> 0x7B10CE05 on 2026-08-12 when slots 26-29 and the
// visibility region below were appended. A Blackhole card ran slots 22-25 and
// measured all three arms at 2.9682 cycles/pair, identical to four decimals,
// which is the null; see "WHAT THE CARD SAID" below. Slots 0-25 keep their
// numbers and 22-25 keep their meaning as the documented negative.
#define TTBENCH_MAGIC 0x7B10CE05u

// Header word indices.
#define TTBENCH_HDR_MAGIC 0
#define TTBENCH_HDR_UNROLL 1
#define TTBENCH_HDR_NUM_PROBES 2
#define TTBENCH_HDR_NUM_POINTS 3
#define TTBENCH_HDR_BASE_BLOCKS 4
#define TTBENCH_HDR_ACTIVE_MASK 5
#define TTBENCH_HDR_PROBE_MASK 6
#define TTBENCH_HDR_WORDS 8

// Instructions per unrolled block. The slope is taken over `blocks`, so this is
// the divisor that turns a per-block slope into a per-instruction cost.
#define TTBENCH_UNROLL 64

// Number of (blocks, cycles) points per probe. blocks = base_blocks * (p + 1).
#define TTBENCH_NUM_POINTS 4

#define TTBENCH_NUM_PROBES 30
#define TTBENCH_MAX_THREADS 3

// The two latency-difference slots, named because two places have to agree
// about which they are: the kernel's `RUN` calls and the host's `--probes`
// recipe. `(slot 20) - (slot 21)` is the whole reading; neither is meaningful
// alone. See "THE LATENCY DIFFERENCE" in raw_probes.cpp.
//
// THESE TWO NO LONGER MEASURE THE LATENCY, and are kept because of it. They
// stall on `p_stall::TRISC_CFG`, which is condition C10 on Blackhole and C13 on
// Wormhole, and neither of those is about the Configuration Unit -- both are
// "the RISCV T core ... has a memory read-request or write-request ... not yet
// processed". A Blackhole card measured their difference at 0.0000 cycles/pair
// on 2026-08-09 for exactly that reason. Slots 22-24 below carry the corrected
// construction, and 20/21 stay as its FALSIFICATION CONTROL: same harness, same
// ops, same block mask, only the condition bit differs, so the two readings in
// one launch are what says the difference came from the condition and not from
// the harness.
#define TTBENCH_P_RDCFG_STALL 20
#define TTBENCH_P_SETDMA_STALL 21

// The corrected construction. `STALLWAIT` on Blackhole condition C12
// (`p_stall::CFGEXU`, 0x1000) -- "Any thread has an instruction in any stage of
// the Configuration Unit pipeline" (BlackholeA0 STALLWAIT.md) -- which is the
// only documented condition on either architecture that observes the unit
// `RDCFG` executes on. See "THE LATENCY DIFFERENCE, DONE PROPERLY" in
// raw_probes.cpp for the full argument and for why Wormhole cannot run this.
//
// Two baselines, because they fail differently:
//   RMWCIB0  a Configuration Unit instruction documented at 1 cycle that
//            occupies stage 0 and nothing else, so (22 - 23) is specifically
//            the extra pipeline residency RDCFG has over a 1-cycle config op.
//   SETDMAREG a ThCon instruction that never enters the config pipeline at all,
//            so (22 - 24) is RDCFG's whole post-issue residency. It is also the
//            movement control: (24 - 9) is what the STALLWAIT itself costs.
#define TTBENCH_P_RDCFG_CFGSTALL 22
#define TTBENCH_P_RMWCIB_CFGSTALL 23
#define TTBENCH_P_SETDMA_CFGSTALL 24
// RMWCIB0 bare, so the two paired config ops' OCCUPANCIES can be compared
// within the same run -- the same role slot 14 plays for RDCFG and slot 9 for
// SETDMAREG. Without it, (22 - 23) could be an occupancy difference.
#define TTBENCH_P_RMWCIB 25

// ---------------------------------------------------------------------------
// WHAT THE CARD SAID, and what slots 26-29 and the visibility region below are
// for. A Blackhole card ran slots 22-25 on 2026-08-12 with the intended
// condition (`TTBENCH_CFGLAT_COND: C12 CFGEXU 0x1000`) and measured
//
//     OCC   0.9978 0.9979 0.9979    RDCFG, RMWCIB0, SETDMAREG bare
//     PAIRS 2.9682 2.9682 2.9683    all three with the identical STALLWAIT
//     DIFF  0.0000 -0.0001
//
// -- every arm identical to four decimal places, whether what preceded the
// STALLWAIT was a Configuration Unit instruction or an op that never enters the
// unit. The stall costs a flat ~2 cycles regardless.
//
// The documents say why, and they say it in the instruction's own Performance
// section. `BlackholeA0/TensixTile/TensixCoprocessor/RDCFG.md`:
//
//     "This instruction requires at least two cycles to execute, and then
//      additional cycles if there is contention for GPR writes. Assuming no
//      contention, it is fully pipelined, so an `RDCFG` instruction can be
//      started every cycle. The issuing thread is not blocked, so it can
//      potentially start its next instructions (of any kind) during `RDCFG`'s
//      subsequent cycles."
//
// and `ConfigurationUnit.md` tabulates it under a column headed **Latency**:
// `RDCFG` ">= 2 cycles", IPC 1. So the ">= 2" is a LATENCY TO THE DESTINATION
// GPR and the throughput is one per cycle -- which is exactly the 0.998 the
// card measured bare. A busy-condition such as C12 can only ever see occupancy
// that outlasts the issue path, and `RDCFG` leaves at most one cycle of that
// (stage +1, the GPR write). The STALLWAIT's own floor is at least that wide:
// "There is a one cycle lag between the condition(s) being met and the block
// mask being removed" (STALLWAIT.md), measured at ~2 cycles per pair. The
// quantity completes inside the measuring apparatus's own overhead.
//
// So the reading is taken somewhere else. `RDCFG.md`'s "Instruction
// scheduling" section names the one observable the hardware does expose:
//
//     "Software must ensure that the instruction(s) immediately after `RDCFG`
//      are not trying to consume the GPR written by the `RDCFG` instruction. In
//      _most_ cases, this applies to the one instruction after `RDCFG`, but it
//      can apply to more than one instruction if there is contention for the
//      GPR write."
//
// That is an obligation on SOFTWARE, not a hardware interlock: a consumer too
// close to the producer does not stall, it reads the STALE value. The latency
// is therefore visible as a DISTANCE, not as a duration -- see
// TTBENCH_VIS_* below, which measures it -- and slots 26/27 are the timing
// arm that tests the interlock reading and is predicted to read zero.
// ---------------------------------------------------------------------------

// The dependent/independent pair. Two instructions in each arm, identical
// opcodes, differing in ONE operand field: which GPR the consumer reads.
//   26  RDCFG -> GPR 60 ; ADDDMAREG reading GPR 60   (dependent)
//   27  RDCFG -> GPR 60 ; ADDDMAREG reading GPR 59   (independent)
// This is the transplant of the house method riscvbench uses for "dependent
// multiply" and "dependent L1 load", and it is run because it is FALSIFIABLE
// rather than because it is expected to move: if Blackhole interlocks the GPR
// read-after-write, (26 - 27) IS the latency in cycles; if it does not -- which
// is what "Software must ensure" says -- (26 - 27) is zero and the quantity is
// not a duration at all. A null here is a result, and it is the result the
// documents predict.
#define TTBENCH_P_RDCFG_DEP 26
#define TTBENCH_P_RDCFG_INDEP 27

// The C12 liveness control, and the ONLY thing in this file that can tell
// "the latency is real but structurally invisible to a busy-condition" from
// "C12 does not behave as documented on this part". Both explain the 0.0000
// above and the visibility region cannot separate them, because it never
// consults C12.
//
// The bodies are THREAD-DEPENDENT, which nothing else in this benchmark is:
//   thread 1 (the graded thread)  STALLWAIT(STALL_THREAD, CFGEXU) ; NOP
//   threads 0 and 2              RMWCIB0(0,0,0) ; NOP
// C12 is "ANY thread has an instruction in any stage of the Configuration Unit
// pipeline", and STALLWAIT.md's own note says "This won't prevent other threads
// from issuing new Configuration Unit instructions though, and those new
// instructions will cause this thread to continue to wait." So at t3 threads 0
// and 2 keep the unit occupied from OUTSIDE thread 1's issue path -- which is
// the one way to make C12 hold for far longer than the stall's own one-cycle
// lag, and therefore the one way a busy-condition can be caught working.
//
// Slot 29 is the same choreography with thread 1 issuing NOP;NOP instead of the
// STALLWAIT, so the cross-thread ISSUE interference (which grows with the
// thread count for every probe in this benchmark) subtracts out. The reading is
//     d(v) = pair(28, v) - pair(29, v)   for v in {t1, t3}
// and the verdict is on d(t3) - d(t1): C12 live makes it large and positive,
// C12 dead leaves it at zero. Both outcomes are reachable, which is what makes
// it a control rather than a formality.
#define TTBENCH_P_C12_XT 28
#define TTBENCH_P_C12_XT_NULL 29

// The slots that need Blackhole's C12. The host clears these from `probe_mask`
// on any other architecture and says why: Wormhole's condition mask has no bit
// for the Configuration Unit (its C0-C14 are tabulated in
// WormholeB0/TensixTile/TensixCoprocessor/STALLWAIT.md and none of them mention
// it), so there is nothing to encode and a run that emitted these slots anyway
// would be timing a stall whose condition is unrelated to the unit under test.
#define TTBENCH_CFGLAT_PROBE_MASK                                              \
    ((1u << TTBENCH_P_RDCFG_CFGSTALL) | (1u << TTBENCH_P_RMWCIB_CFGSTALL) |    \
     (1u << TTBENCH_P_SETDMA_CFGSTALL) | (1u << TTBENCH_P_RMWCIB) |            \
     (1u << TTBENCH_P_C12_XT) | (1u << TTBENCH_P_C12_XT_NULL))
// The mask that runs the difference and its controls, and nothing else.
// Isolating it matters because phase A's validity gate is per-PHASE, not
// per-probe: one nonlinear new slot would flip TTBENCH_VALID_A for all
// twenty-two series at once, condemning the nineteen good ones with it.
//
// FIVE slots, not three, because the difference cannot be graded alone:
//   0   the empty-loop control every slope phase needs
//   9   SETDMAREG bare. Slot 21 is the same op PLUS the stall, so
//       (21 - 9) is what the STALLWAIT itself costs -- and a STALLWAIT that
//       costs nothing is a stall that never engaged, which makes (20 - 21)
//       uninterpretable. This is the probe's movement control.
//   14  RDCFG bare, so the OCCUPANCIES of the two paired ops can be compared
//       in the same run. If they differ, part of (20 - 21) is occupancy rather
//       than latency and the difference is confounded.
//   20  RDCFG    + STALLWAIT
//   21  SETDMAREG + the identical STALLWAIT
// NINE slots now, not five: the four appended above join the five that were
// already here, and every one of the four answers a named confound.
//   22  RDCFG     + STALLWAIT(STALL_THREAD, CFGEXU)   the measurement
//   23  RMWCIB0   + the identical STALLWAIT           the in-unit baseline
//   24  SETDMAREG + the identical STALLWAIT           the off-unit baseline,
//                                                     and (24 - 9) is the
//                                                     movement control
//   25  RMWCIB0 bare, so (25 vs 14) shows the two config ops cost the same to
//       issue and the (22 - 23) difference is therefore not an occupancy
// ELEVEN slots now. Slot 10 (ADDDMAREG bare) joins because slots 26/27 are
// (RDCFG + ADDDMAREG) pairs and their cost has to be shown to be the sum of the
// two bare costs -- otherwise "the difference is zero" could mean "neither
// instruction ran" rather than "there is no interlock". Slots 26-29 are the
// four appended above.
#define TTBENCH_LATENCY_PROBE_MASK                                          \
    ((1u << 0) | (1u << 9) | (1u << 10) | (1u << 14) |                      \
     (1u << TTBENCH_P_RDCFG_STALL) | (1u << TTBENCH_P_SETDMA_STALL) |       \
     (1u << TTBENCH_P_RDCFG_DEP) | (1u << TTBENCH_P_RDCFG_INDEP) |          \
     TTBENCH_CFGLAT_PROBE_MASK)  // = 0x3FF04601

// ---------------------------------------------------------------------------
// THE VISIBILITY REGION: where `RDCFG`'s ">= 2" is actually read.
//
// Not a probe slot, and deliberately not: every slot in this file carries a
// CYCLE COUNT that the host fits a slope through and grades for linearity and
// monotonicity, and what follows is a GPR VALUE. Putting values in a cycles
// series would have flipped TTBENCH_VALID_A for every other series in the run.
// It also keeps `probe_id` untouched, so no CSV column and no `--probes` bit
// moves.
//
// THE CONSTRUCTION. For a separation `d` (measured in ISSUE SLOTS between the
// producer and the consumer, so d = 1 means "the instruction immediately
// after"):
//
//     regfile[G] = seed              (a RISC-V write, read back to order it)
//     regfile[H] = TTBENCH_VIS_MARK
//     tensix_sync()                  the Tensix pipe is empty from here
//     TTI_RDCFG(G, TTBENCH_VIS_CFGIDX)
//     TTI_NOP x (d - 1)              one `.ttinsn` each, one issue slot each
//     TTI_ADDDMAREG(1, H, 0, G)      H = GPR[G] + 0 -- the consumer
//     tensix_sync()
//     observe regfile[H]
//
// and the observation is one of exactly four things:
//     == seed                the consumer ran and saw the STALE GPR
//     == the far-arm value   the consumer ran and saw RDCFG's result
//     == TTBENCH_VIS_MARK    the consumer never ran        (apparatus failure)
//     anything else          unexplained                   (apparatus failure)
//
// THE READING is the smallest `d` at which every repetition sees the fresh
// value. Because the consumer might read its operand some delta >= 0 cycles
// after its own issue, that d is a LOWER BOUND on the latency -- which is the
// right direction: the charging policy takes published bounds at their low end,
// and `d_min = 2` corroborates ">= 2" exactly while `d_min = 1` would leave the
// ">= 2" unreached rather than refuted.
//
// WHY THIS IS FREE OF THE C12 PROBLEM. It issues no STALLWAIT and consults no
// condition bit, so it is untouched by tt-sim reading Blackhole's condition
// mask as 12 bits where the ISA page gives 13 -- the defect that degrades slots
// 22-25 under the simulator. Under tt-sim it degrades for a different and much
// more informative reason; see raw_probes.cpp.
//
// THE CONTROLS FIRE IN BOTH DIRECTIONS, which is the whole reason there are
// two sentinels and a separate marker:
//   * two "no RDCFG at all" arms, one per sentinel, must read back their own
//     sentinel. A readout that returned a constant fails this, and so does one
//     where the seed never landed. This proves a STALE reading is representable.
//   * two "RDCFG at d = TTBENCH_VIS_FAR" arms, one per sentinel, must read back
//     the SAME value as each other and it must not be either sentinel. This
//     proves a FRESH reading is representable and that it is not an echo of the
//     seed.
//   * neither can pass vacuously: the first pair can only pass by returning two
//     DIFFERENT values, the second only by returning one value twice.
// ---------------------------------------------------------------------------
#define TTBENCH_VIS_WORDS 16

// Written into word 0 by the kernel when the region was actually produced, so
// "the probe did not run" is not a buffer of zeros that looks like data.
#define TTBENCH_VIS_STAMP 0x56495331u  // "VIS1"

// The two seeds and the never-ran marker. Chosen to be mutually distinct and
// to look nothing like a plausible configuration word; the host still checks
// that the config value read back differs from all three, and calls the run
// degenerate rather than reporting a threshold if it does not.
#define TTBENCH_VIS_S1 0xA5A5A5A5u
#define TTBENCH_VIS_S2 0x5A5A5A5Au
#define TTBENCH_VIS_MARK 0xDEADBEEFu

// Which configuration word `RDCFG` reads. Index 0, the same one probe 14 uses,
// so the two slots are reading the identical instruction. Its VALUE does not
// matter and is never predicted -- only that it differs from the seeds, which
// the host checks.
#define TTBENCH_VIS_CFGIDX 0
// Separations swept, 1 .. TTBENCH_VIS_MAXD, plus one far arm that is only there
// to learn what "fresh" looks like.
#define TTBENCH_VIS_MAXD 4
#define TTBENCH_VIS_FAR 8
// The GPRs. 60 is `p_gpr_math::TMP0` and is what probe 14 already writes; 63 is
// unused by `p_gpr_math`. The visibility block runs on the MATH thread only.
#define TTBENCH_VIS_GPR_SRC 60
#define TTBENCH_VIS_GPR_DST 63

// Word indices within the region.
#define TTBENCH_VIS_W_STAMP 0
#define TTBENCH_VIS_W_REPS 1
#define TTBENCH_VIS_W_FAR_S1 2   // RDCFG at d = FAR, seeded S1 -> the fresh value
#define TTBENCH_VIS_W_FAR_S2 3   // ditto seeded S2; must equal the above
#define TTBENCH_VIS_W_NORD_S1 4  // no RDCFG, seeded S1 -> must read back S1
#define TTBENCH_VIS_W_NORD_S2 5  // no RDCFG, seeded S2 -> must read back S2
#define TTBENCH_VIS_W_FRESH 6    // 6..9   fresh count per d = 1..MAXD
#define TTBENCH_VIS_W_STALE 10   // 10..13 stale count per d = 1..MAXD
#define TTBENCH_VIS_W_MARK 14    // repetitions the consumer never ran, all d
#define TTBENCH_VIS_W_OTHER 15   // repetitions that read none of the three, all d
// What every tracked dataset in tt_sim/perf/datasets/ was collected with, so a
// run meant to be compared against them can ask for exactly that experiment.
#define TTBENCH_LEGACY_PROBE_MASK 0xFFFFFu

// How the three MATH probes get their SrcA/SrcB data-valid bits. Runtime arg 5
// of kernels/compute/raw_probes.cpp; the host names them on the command line and
// records the choice in the CSV header as `dvalid_setup=`.
//
// The numeric values of PER_THREAD and ONCE are the 0/1 that arg 5 carried when
// it was a bare boolean, so the two pre-existing configurations are unchanged
// bit for bit. UNPACR_NOP is new; see experiment X2 of
// docs/plans/matrix-unit-thread-contention.md.
#define TTBENCH_DVALID_PER_THREAD 0  // one SETDVALID per ACTIVE thread (confounded)
#define TTBENCH_DVALID_ONCE 1        // one SETDVALID, thread 1, barriered (X1)
#define TTBENCH_DVALID_UNPACR_NOP 2  // one UNPACR_NOP+set_dvalid per unpacker (X2)

// The source data format programmed before an UNPACR_NOP setup, as the 4-bit
// code the hardware's data-format fields take (tt::DataFormat on the host side).
// Only these four are offered: they are the ones that reach a DISTINCT SrcAStyle
// in the Matrix Unit's documented decode, plus FP32, which deliberately does not
// -- see the header comment of raw_probes.cpp.
#define TTBENCH_FMT_FP32 0
#define TTBENCH_FMT_FP16 1
#define TTBENCH_FMT_TF32 4
#define TTBENCH_FMT_BF16 5

// The visibility region sits after every thread's probe array, so widening
// TTBENCH_NUM_PROBES again moves it without changing its internal layout.
#define TTBENCH_VIS_BASE \
    (TTBENCH_HDR_WORDS + TTBENCH_MAX_THREADS * TTBENCH_NUM_PROBES * TTBENCH_NUM_POINTS)

#define TTBENCH_RESULT_WORDS (TTBENCH_VIS_BASE + TTBENCH_VIS_WORDS)

// Phase B (matmul_tiles at a fixed math fidelity) reuses the same buffer with
// probe slot 0 only.
#define TTBENCH_MM_NUM_POINTS TTBENCH_NUM_POINTS

// Phase B blocking factor: how many tile pairs are waited for, multiplied and
// popped as one group. It exists to get the circular-buffer bookkeeping OUT of
// the per-matmul cost. One `cb_wait_front`/`cb_pop_front` pair per operand now
// covers TTBENCH_MM_BLOCK matmuls instead of one, so the RISC-V-side cost per
// `matmul_tiles` falls by roughly that factor and the Tensix side (16 x phases
// MVMULs, expanded by the MOP) gets a chance to become the limit. See the
// header comment of kernels/compute/matmul_fidelity.cpp.
//
// Both circular buffers must hold at least this many pages; the host allocates
// 2x so the feeder can run ahead by a whole block.
#define TTBENCH_MM_BLOCK 8

// Phase B times threads 0 (unpack) and 1 (math). Thread 2 (pack) has no
// per-iteration work in a matmul inner loop -- `cb_wait_front`, `cb_pop_front`
// and `matmul_tiles` are all UNPACK/MATH-only -- so its timed region is empty
// and reads ~1 cycle whatever the iteration count. It is not measured.
#define TTBENCH_MM_TIMED_THREADS 2
