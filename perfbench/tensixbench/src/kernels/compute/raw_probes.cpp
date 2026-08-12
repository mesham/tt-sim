// Phase A of the Tensix cycle-cost benchmark: per-instruction issue throughput,
// measured by the slope of an unrolled TTI_* burst.
//
// THE METHOD, in one paragraph. For each probe the kernel runs the SAME
// 64-instruction unrolled block `blocks` times, for four values of `blocks`,
// and timestamps each run off the device's own wall clock. The measured time is
//
//     cycles(blocks) = blocks * (64 * c_op + L) + K
//
// where `c_op` is the per-instruction cost we want, `L` is the RISC-V loop
// overhead (counter, compare, branch) and `K` is everything fixed: the two
// clock reads, the barrier, the surrounding call. A least-squares slope over
// the four points kills `K` exactly. Probe 0 runs the identical loop with an
// EMPTY body, so its slope is `L` alone, and
//
//     c_op = (slope(probe) - slope(probe 0)) / 64
//
// kills `L` exactly too. No absolute measurement is ever reported as a cost.
// This is the same "cancel the common term by subtraction" discipline that
// rungs 1 and 2 of the calibration ladder in docs/plans/cost-model.md used.
//
// WHY UNROLLED TTI_* AND NOT A COMPUTE-API LOOP. tt-sim does not model the
// RISC-V instruction fetch/issue path, so if the issuing baby core cannot feed
// the Tensix unit fast enough we measure the front end, not the unit. Each
// TTI_* macro is exactly one `.ttinsn` word in the RISC-V instruction stream --
// one issue slot -- so an unrolled block offers the Tensix one instruction per
// core cycle and nothing else. That is the *fastest a single thread can go*, so
// a single-thread measurement can only ever establish min(issue rate, unit
// throughput). Separating the two needs more than one issuer, which is what
// `active_mask` is for: run the identical probe from two or three TRISCs at
// once and see whether each thread's slope grows in proportion (unit is the
// limit, shared) or stays put (the front end was the limit).
//
// The probes are all state-free by construction. The SFPU ones touch LREGs
// only, the ThCon ones touch high GPR indices only, RDCFG only reads, and every
// RMWCIB0 here (slots 23, 25, and the hammering threads of 28/29) carries a zero
// mask, which `RMWCIB.md`'s own functional model makes the identity. The
// visibility region is the one place a GPR is written from the RISC-V side
// rather than by a Tensix instruction, and it writes only GPRs 60 and 63 of the
// MATH thread -- 60 is `p_gpr_math::TMP0`, a scratch temp that probe 14 already
// overwrites, and 63 is unused by `p_gpr_math` altogether. The three
// MATH probes at the end are the exception: MVMUL/ELWADD/ELWMUL wait on the
// SrcA/SrcB data-valid bits, so the kernel sets those with SETDVALID and issues
// every op with clear_dvalid = 0 so they stay set. They are last, and gated by
// `probe_mask`, so a hang bisects.
//
// HOW THE VALID BITS GET SET -- experiments X1 and X2. There are three setups,
// selected by runtime arg 5 (`dvalid_mode`, values in bench_layout.h), and the
// choice is recorded in the CSV header as `dvalid_setup=`:
//
//   PER_THREAD  one bare SETDVALID per ACTIVE thread     (original, confounded)
//   ONCE        one bare SETDVALID, thread 1, barriered  (X1, and the DEFAULT)
//   UNPACR_NOP  one UNPACR_NOP+set_dvalid per unpacker,
//               thread 1, barriered, at a chosen source
//               data format                              (X2)
//
// HOW MANY SETDVALIDs -- experiment X1. That setup used to sit unguarded in
// `kernel_main`, which every active thread runs, so the number of SETDVALIDs
// executed was exactly the thread count and `active_threads` was perfectly
// confounded with the SrcA/SrcB bank state the burst ran in. That matters: the
// first Blackhole run measured MVMUL/ELWADD/ELWMUL at 6.1x and 12.1x per-thread
// cost at two and three threads, against 2x/3x for every other unit, and the
// confound is a complete alternative explanation -- on Blackhole SETDVALID is
// `UnsupportedFunctionality` and leaves `ImpliedSrc{A,B}Fmt` `UnpredictableValue`,
// and the third thread hands the FPU a bank it already owns, which the vendor
// simulator asserts on as `NonContractualBehavior`.
//
// `TTBENCH_DVALID_ONCE` (runtime arg 5, the DEFAULT) issues exactly one
// SETDVALID, from thread 1, and barriers after it, so every variant runs the
// burst in the same Src state and only the thread count varies. This is sound
// because `AllowedClient` and both bank pointers are single globals with no
// thread dimension (`SrcASrcB.md`), so one SETDVALID satisfies every thread's
// Wait Gate. `TTBENCH_DVALID_PER_THREAD` (`--dvalid-per-thread` on the host)
// reproduces the original per-thread setup byte for byte, so the two can be run
// back to back and diffed.
// Full argument: docs/plans/matrix-unit-thread-contention.md, experiment X1.
//
// A SOURCE DATA FORMAT AXIS -- experiment X2. Even the single legal SETDVALID
// above leaves `ImpliedSrc{A,B}Fmt` an `UnpredictableValue()` on Blackhole
// (`SETDVALID.md`), and Blackhole's Matrix Unit reads exactly that field for the
// source format unless `DISABLE_IMPLIED_SRC{A,B}_FMT_Base` is set -- which no
// Blackhole LLK ever sets. So the format the FPU decodes during phase A's MATH
// probes is, today, undefined. `TTBENCH_DVALID_UNPACR_NOP` replaces SETDVALID
// with the Blackhole-sanctioned `UNPACR_NOP` carrying `set_dvalid`
// (`UNPACR_NOP_SETDVALID.md`), one per unpacker, issued from thread 1 with a
// barrier after. That instruction writes
// `ImpliedSrc{A,B}Fmt[bank] = THCON_SEC{0,1}_REG2_Out_data_format`, so the
// format becomes DEFINED -- and, being config, becomes something the run can
// choose. `src_format` (runtime arg 6) is programmed into those two registers
// and into the configured `ALU_FORMAT_SPEC_REG0_SrcA` / `_REG1_SrcB` (with the
// override bits cleared) immediately before, so the format is unambiguous
// whichever path the hardware reads and whichever architecture this is.
//
// WHAT THE ISA DOCS PREDICT ABOUT FORMAT, stated before any run. `MVMUL.md`,
// `ELWADD.md` and `ELWMUL.md` all reduce the source format to a three-way
// `SrcAStyle`:
//
//     FP32, BF16, BFP8, BFP4, BFP2, INT32, INT16  -> SrcAStyle = BF16
//     FP16, FP8, BFP8a, BFP4a, BFP2a, INT8        -> SrcAStyle = FP16
//     TF32                                        -> SrcAStyle = TF32
//
// so BF16 and FP32 land on the SAME decode and are predicted to be EXACTLY
// indistinguishable; TF32 and FP16 are the codes that actually change the
// datapath. `MatrixUnit.md`'s throughput table gives 1 IPC with no format
// qualification at all, so the docs predict no cost difference for any of them.
// The experiment is therefore EXPLORATORY, not confirmatory: it is looking for
// a per-format cost the documentation does not describe, and a null result is
// the expected one. All four codes are offered so that the bf16-vs-fp32 pair
// (predicted null by construction) is measured alongside a pair that at least
// moves the decode.
//
// LEAVING THE CARD CLEAN -- why the UNPACR_NOP setup, and only it, has a
// release at the end. `UNPACR_NOP` with `Unpack_Pop = UNP_ZEROSRC` is the
// ACQUIRE half of the unpacker's handshake: before it may zero its bank and
// hand it to the Matrix Unit it waits until the bank named by
// `MatrixUnit.Src{A,B}Bank` is no longer valid. Every MATH probe below issues
// `clear_dvalid = 0` -- which is exactly what makes the burst measurable -- so
// nothing in the timed region ever performs the release half, and a completed
// run used to leave `SrcA[0]`/`SrcB[0]` owned by the Matrix Unit. The NEXT
// execution of the setup then waited for that release for ever: the t2 launch
// in the same process, or the first launch of the next process on the same
// card. Only `tt-smi -r 0` cleared it, so a *successful* run poisoned the card
// for the following one -- confirmed on Blackhole silicon, twice, by
// experiments that could each have refuted it (see
// docs/plans/matrix-unit-thread-contention.md, "X2 on silicon").
//
// So the release is issued once, from thread 1, AFTER the last probe has
// written its last result word. It is outside every timed region by
// construction: `clear_dvalid = 0` on the probes is untouched, and phase A's
// data rows are byte-identical to before it existed.
//
// `SETDVALID` has no wait half -- it sets the bits and flips the pointer
// unconditionally -- so `--dvalid-once` and `--dvalid-per-thread` re-run on a
// dirty card indefinitely and get no release. That is deliberate: they cannot
// wedge, their datasets are already banked, and a release they do not need is
// a change to a measurement for nothing.
//
// Result buffer layout is in bench_layout.h, shared with the host.

#include <cstdint>

#include "api/compute/common.h"
#include "hostdevcommon/kernel_structs.h"

#include "cfg_defines.h"

#include "bench_layout.h"

// ---------------------------------------------------------------------------
// Unrolling. Variadic so a probe body may contain commas.
// ---------------------------------------------------------------------------
#define REP2(...) __VA_ARGS__ __VA_ARGS__
#define REP4(...) REP2(__VA_ARGS__) REP2(__VA_ARGS__)
#define REP8(...) REP4(__VA_ARGS__) REP4(__VA_ARGS__)
#define REP16(...) REP8(__VA_ARGS__) REP8(__VA_ARGS__)
#define REP32(...) REP16(__VA_ARGS__) REP16(__VA_ARGS__)
#define REP64(...) REP32(__VA_ARGS__) REP32(__VA_ARGS__)

namespace {

// The device's own free-running cycle counter -- the same register tt-metal's
// device profiler reads for DeviceZoneScopedN (kernel_profiler.hpp reads
// RISCV_DEBUG_REG_WALL_CLOCK_L / +4). Reading the low word latches the high
// word into +4; we only need the low word, since no probe here runs for 2^32
// cycles.
inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return p[0];
}

volatile tt_l1_ptr uint32_t* g_barrier = nullptr;
uint32_t g_thread = 0;
uint32_t g_active = 1;
uint32_t g_seq = 0;

// Line the participating TRISCs up so a contention probe really is contended.
// Each thread publishes a sequence number in its own word and spins until every
// active thread has published at least that number: no atomics needed, because
// each word has exactly one writer. The host zeroes the region before launch.
inline void bench_barrier() {
    g_seq++;
    g_barrier[g_thread] = g_seq;
    for (uint32_t t = 0; t < TTBENCH_MAX_THREADS; t++) {
        if (g_active & (1u << t)) {
            while (g_barrier[t] < g_seq) {
            }
        }
    }
}

// Program the source data format the Matrix Unit will decode during the MATH
// probes, in every place the ISA documentation says it could be read from.
//
//   THCON_SEC{0,1}_REG2_Out_data_format -- what the following UNPACR_NOP copies
//       into `ImpliedSrc{A,B}Fmt[bank]`. This is the live path on Blackhole,
//       because `DISABLE_IMPLIED_SRC{A,B}_FMT_Base` is never set by any
//       Blackhole LLK, and it is the whole point of the experiment.
//   ALU_FORMAT_SPEC_REG0_SrcA / _REG1_SrcB -- the CONFIGURED format. This is
//       what Wormhole always reads (it has no implied-format mechanism) and
//       what Blackhole would read if the disable bits were ever set. Written so
//       the two paths cannot disagree.
//   ALU_FORMAT_SPEC_REG_Src{A,B}_override -- cleared, because a set override
//       makes the Matrix Unit read `_val` instead of the register above and
//       would silently ignore everything this function does.
//
// These are Configuration Unit instructions and the UNPACR_NOPs that follow are
// Unpacker instructions, i.e. two different backend units. The ordering
// assumption -- that a config write issued earlier in one thread's instruction
// stream is visible to an unpack instruction issued later -- is the same one
// every Blackhole LLK makes (`llk_unpack_common.h` writes
// `THCON_SEC0_REG2_Out_data_format` and then unpacks with no barrier between).
// It is stated here rather than assumed silently.
inline void set_source_format(uint32_t fmt) {
    ckernel::cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(0);
    ckernel::cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcB_override_RMW>(0);
    ckernel::cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(fmt);
    ckernel::cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(fmt);
    ckernel::cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(fmt);
    ckernel::cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(fmt);
}

// Hand SrcA bank 0 and SrcB bank 0 to the Matrix Unit the way Blackhole's own
// LLKs do, taking a DEFINED implied format with them. One instruction per
// unpacker, because `UNPACR_NOP` addresses one unpacker at a time where
// `SETDVALID(3)` did both at once.
//
// The operand lists differ between the architectures because the instruction
// word does: Wormhole packs a 23-bit `NoOp` mode select (`UNP_SET_DVALID` =
// 0b111), Blackhole re-lays the word out into named fields and puts `set_dvalid`
// at bit 8. The Blackhole form is copied verbatim from
// `tt_llk_blackhole/llk_lib/llk_unpack_common.h`, `UNP_ZEROSRC` and all: that
// combination stalls until the Matrix Unit has released the bank, zeroes it, and
// only then hands it over, which is the sanctioned sequence rather than a
// minimal one. The preceding STALLWAIT (block B3, conditions C1|C2) is from the
// same source; here it is provably vacuous, since no probe before this one
// touches an unpacker, but running the documented sequence rather than a
// shortened one is the point of the experiment.
inline void give_both_srcs_to_fpu() {
    TTI_STALLWAIT(ckernel::p_stall::STALL_UNPACK, ckernel::p_stall::UNPACK);
#if defined(ARCH_BLACKHOLE)
    TTI_UNPACR_NOP(0, 0, 0, ckernel::p_unpacr_nop::SET_DVALID, 0, 0, 0, 0,
                   ckernel::p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(1, 0, 0, ckernel::p_unpacr_nop::SET_DVALID, 0, 0, 0, 0,
                   ckernel::p_unpacr_nop::UNP_ZEROSRC);
#else
    TTI_UNPACR_NOP(0, ckernel::p_unpacr_nop::UNP_SET_DVALID);
    TTI_UNPACR_NOP(1, ckernel::p_unpacr_nop::UNP_SET_DVALID);
#endif
}

// The other half of `give_both_srcs_to_fpu`: hand SrcA and SrcB back to the
// unpackers. See "LEAVING THE CARD CLEAN" in the header comment for why only
// the UNPACR_NOP setup needs this and why it is safe here and nowhere earlier.
//
// `CLEARDVALID` is a Matrix Unit instruction (`ckernel_ops.h`:
// `TT_OP_CLEARDVALID(cleardvalid, reset)`, opcode 0x36, `cleardvalid` at bit 22,
// bit 0 = SrcA and bit 1 = SrcB), and it is what the Blackhole LLKs use whenever
// a math op did not carry `clear_dvalid` itself -- `llk_math_reduce.h` issues
// `TTI_CLEARDVALID(clear_mode, 0)` after a `ckernel_template::run()` of MVMULs
// for exactly that reason, and `llk_math_eltwise_unary_datacopy.h` and
// `experimental/llk_math_mul_reduce_scalar.h` do the same. `reset` is 0: the
// vendor reference simulator rejects `reset & 1` outright ("unsafe and drops
// SrcA/B banks", tt-metal issue 22383), and `reset & 2` would suppress the bank
// flip, which is the half that makes the sequence repeatable -- clearing bank
// `MatrixUnit.Src{A,B}Bank` *and* advancing the pointer is what leaves the unit
// pointing at the bank the next `UNPACR_NOP` will fill.
inline void take_both_srcs_back_from_fpu() {
    TTI_CLEARDVALID(ckernel::p_setrwc::CLR_AB, 0);
}

// ---------------------------------------------------------------------------
// THE VISIBILITY CONSTRUCTION: RDCFG's ">= 2" read as a DISTANCE.
//
// WHY NOT AS A DURATION. `BlackholeA0/.../ConfigurationUnit.md` tabulates
// `RDCFG` under a column headed **Latency** at ">= 2 cycles" with **IPC** 1, and
// `RDCFG.md` says in as many words that "Assuming no contention, it is fully
// pipelined, so an `RDCFG` instruction can be started every cycle. The issuing
// thread is not blocked, so it can potentially start its next instructions (of
// any kind) during `RDCFG`'s subsequent cycles." A quantity that neither blocks
// the issuer nor lowers the issue rate cannot be timed from the issuing thread,
// and a card confirmed the throughput half at 0.998 cycles per instruction.
//
// WHAT THE HARDWARE DOES EXPOSE. `RDCFG.md`, "Instruction scheduling":
//
//     "Software must ensure that the instruction(s) immediately after `RDCFG`
//      are not trying to consume the GPR written by the `RDCFG` instruction."
//
// An obligation on software is not an interlock. A consumer placed too close
// does not wait -- it reads the OLD contents of the GPR. So the latency shows
// up as the smallest producer-to-consumer distance at which the new value is
// seen, and that distance is an integer, not a slope.
//
// WHICH INSTRUCTIONS CAN CONSUME IT. The Configuration Unit "`WRCFG` and
// `RDCFG` instructions access the same GPRs as the Scalar Unit (ThCon)"
// (ConfigurationUnit.md), so the consumers are ThCon's GPR-reading ops and
// `WRCFG`. `WRCFG` is the sharper instrument -- the stage table pins its GPR
// read to stage -1, the cycle it enters the pipeline -- but it WRITES backend
// configuration and this benchmark does not mutate device state, so the
// consumer here is `ADDDMAREG`, which is already probe 10. If ThCon reads its
// operand some delta >= 0 cycles into its own execution, the measured distance
// is short by delta, which makes it a LOWER BOUND on the latency. That is the
// direction the charging policy wants: bounds are taken at their low end.
//
// WHY THE SEQUENCE HAS TO BE LITERAL. Each `TTI_*` is one `.ttinsn` word in the
// RISC-V instruction stream -- `#define INSTRUCTION_WORD(x) __asm__ __volatile__
// (".ttinsn %0" : : "n"((x)))` -- so a run of them offers the Tensix front end
// one instruction per core cycle and the separation in issue slots is the
// separation in cycles. The runtime `TT_*` forms compute their instruction word
// first, which would insert RISC-V cycles between the pushes and silently widen
// the gap being measured. Hence the separation is a template parameter and the
// fillers are emitted by recursion: every one of them has to be an immediate.
// ---------------------------------------------------------------------------

template <uint32_t N>
inline __attribute__((always_inline)) void vis_fillers() {
    TTI_NOP;
    vis_fillers<N - 1>();
}
template <>
inline __attribute__((always_inline)) void vis_fillers<0>() {}

// One observation. Seeds the producer's destination GPR with `seed` and the
// consumer's destination with TTBENCH_VIS_MARK, drains the Tensix pipe, runs
// (optionally) RDCFG then `SEP - 1` fillers then the consumer, drains again and
// reads the consumer's destination back.
//
// The seeding is done by RISC-V writes to the GPR window at REGFILE_BASE rather
// than by `SETDMAREG`, because the seed value is a runtime argument here and a
// `TTI_SETDMAREG` immediate cannot be. `sync_regfile_write` reads the GPR back,
// which is the LLK's own way of ordering a RISC-V GPR write against what
// follows; `tensix_sync` then blocks until this thread's instruction pipe has
// drained, so nothing from the seeding phase is still in flight when the timed
// sequence starts.
template <uint32_t SEP, bool WITH_RDCFG>
inline uint32_t vis_observe(uint32_t seed) {
    ckernel::regfile[TTBENCH_VIS_GPR_SRC] = seed;
    ckernel::regfile[TTBENCH_VIS_GPR_DST] = TTBENCH_VIS_MARK;
    ckernel::sync_regfile_write(TTBENCH_VIS_GPR_SRC);
    ckernel::sync_regfile_write(TTBENCH_VIS_GPR_DST);
    ckernel::tensix_sync();

    if constexpr (WITH_RDCFG) {
        TTI_RDCFG(TTBENCH_VIS_GPR_SRC, TTBENCH_VIS_CFGIDX);
    }
    vis_fillers<SEP - 1>();
    // The consumer: GPR[DST] = GPR[SRC] + 0. `OpBisConst` is 1, so SRC is the
    // only GPR read and the arm differs from the no-RDCFG arm in nothing else.
    TTI_ADDDMAREG(1, TTBENCH_VIS_GPR_DST, 0, TTBENCH_VIS_GPR_SRC);

    ckernel::tensix_sync();
    return ckernel::regfile[TTBENCH_VIS_GPR_DST];
}

// The whole region, run by the MATH thread alone (see the guard at the call
// site: any other thread issuing Configuration Unit work would be a second
// writer of the unit's pipeline).
inline void run_visibility(volatile tt_l1_ptr uint32_t* vis, uint32_t reps) {
    // The four controls first, because the sweep is classified against them and
    // a sweep classified against a broken reference is worse than no sweep.
    const uint32_t far_s1 = vis_observe<TTBENCH_VIS_FAR, true>(TTBENCH_VIS_S1);
    const uint32_t far_s2 = vis_observe<TTBENCH_VIS_FAR, true>(TTBENCH_VIS_S2);
    const uint32_t nord_s1 = vis_observe<TTBENCH_VIS_MAXD, false>(TTBENCH_VIS_S1);
    const uint32_t nord_s2 = vis_observe<TTBENCH_VIS_MAXD, false>(TTBENCH_VIS_S2);

    uint32_t fresh[TTBENCH_VIS_MAXD] = {0, 0, 0, 0};
    uint32_t stale[TTBENCH_VIS_MAXD] = {0, 0, 0, 0};
    uint32_t marked = 0;
    uint32_t other = 0;

    // Classify one observation. `far_s1` is what a fresh read looks like; the
    // host refuses to report a threshold unless it differs from both sentinels
    // and from the marker, so the order of these tests cannot hide a
    // degeneracy -- it is reported instead.
    auto classify = [&](uint32_t d0, uint32_t v) {
        if (v == far_s1) {
            fresh[d0]++;
        } else if (v == TTBENCH_VIS_S1) {
            stale[d0]++;
        } else if (v == TTBENCH_VIS_MARK) {
            marked++;
        } else {
            other++;
        }
    };

    // Repeated, and every repetition counted, because a single observation
    // cannot tell a latency from one hiccup in RISC-V instruction delivery: if
    // the front end ever fails to offer the sequence at one instruction per
    // cycle the separation is wider than it looks, and that shows up here as a
    // MIXTURE rather than as a clean threshold.
    for (uint32_t r = 0; r < reps; r++) {
        classify(0, vis_observe<1, true>(TTBENCH_VIS_S1));
        classify(1, vis_observe<2, true>(TTBENCH_VIS_S1));
        classify(2, vis_observe<3, true>(TTBENCH_VIS_S1));
        classify(3, vis_observe<4, true>(TTBENCH_VIS_S1));
    }

    vis[TTBENCH_VIS_W_REPS] = reps;
    vis[TTBENCH_VIS_W_FAR_S1] = far_s1;
    vis[TTBENCH_VIS_W_FAR_S2] = far_s2;
    vis[TTBENCH_VIS_W_NORD_S1] = nord_s1;
    vis[TTBENCH_VIS_W_NORD_S2] = nord_s2;
    for (uint32_t i = 0; i < TTBENCH_VIS_MAXD; i++) {
        vis[TTBENCH_VIS_W_FRESH + i] = fresh[i];
        vis[TTBENCH_VIS_W_STALE + i] = stale[i];
    }
    vis[TTBENCH_VIS_W_MARK] = marked;
    vis[TTBENCH_VIS_W_OTHER] = other;
    // Last, so a host that sees the stamp knows every other word is written.
    vis[TTBENCH_VIS_W_STAMP] = TTBENCH_VIS_STAMP;
}

}  // namespace

// One probe: four (blocks, cycles) points into the caller's slot.
#define PROBE(SLOT, ...)                                             \
    do {                                                             \
        for (uint32_t p = 0; p < TTBENCH_NUM_POINTS; p++) {          \
            const uint32_t blocks = base_blocks * (p + 1);           \
            bench_barrier();                                         \
            const uint32_t t0 = wall_clock_lo();                     \
            for (uint32_t b = 0; b < blocks; b++) {                  \
                REP64(__VA_ARGS__)                                   \
            }                                                        \
            out[(SLOT) * TTBENCH_NUM_POINTS + p] = wall_clock_lo() - t0; \
        }                                                            \
    } while (0)

// A probe the host masked off. Still barriers, so the participating threads
// stay in step, and reports 0 so the analysis can tell "not run" from "0
// cycles".
#define PROBE_SKIP(SLOT)                                             \
    do {                                                             \
        for (uint32_t p = 0; p < TTBENCH_NUM_POINTS; p++) {          \
            bench_barrier();                                         \
            out[(SLOT) * TTBENCH_NUM_POINTS + p] = 0;                \
        }                                                            \
    } while (0)

#define RUN(SLOT, ...)                     \
    if (probe_mask & (1u << (SLOT))) {     \
        PROBE(SLOT, __VA_ARGS__);          \
    } else {                               \
        PROBE_SKIP(SLOT);                  \
    }

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(0);
    const uint32_t barrier_addr = get_arg_val<uint32_t>(1);
    const uint32_t base_blocks = get_arg_val<uint32_t>(2);
    const uint32_t probe_mask = get_arg_val<uint32_t>(3);
    const uint32_t active_mask = get_arg_val<uint32_t>(4);
    const uint32_t dvalid_mode = get_arg_val<uint32_t>(5);
    const uint32_t src_format = get_arg_val<uint32_t>(6);
    const uint32_t vis_reps = get_arg_val<uint32_t>(7);

#if defined(TRISC_UNPACK)
    g_thread = 0;
#elif defined(TRISC_MATH)
    g_thread = 1;
#else
    g_thread = 2;
#endif
    g_active = active_mask;
    g_barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);

    if ((active_mask & (1u << g_thread)) == 0) {
        return;
    }

    volatile tt_l1_ptr uint32_t* base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    if (g_thread == 1) {
        base[TTBENCH_HDR_MAGIC] = TTBENCH_MAGIC;
        base[TTBENCH_HDR_UNROLL] = TTBENCH_UNROLL;
        base[TTBENCH_HDR_NUM_PROBES] = TTBENCH_NUM_PROBES;
        base[TTBENCH_HDR_NUM_POINTS] = TTBENCH_NUM_POINTS;
        base[TTBENCH_HDR_BASE_BLOCKS] = base_blocks;
        base[TTBENCH_HDR_ACTIVE_MASK] = active_mask;
        base[TTBENCH_HDR_PROBE_MASK] = probe_mask;
    }
    volatile tt_l1_ptr uint32_t* out =
        base + TTBENCH_HDR_WORDS + g_thread * TTBENCH_NUM_PROBES * TTBENCH_NUM_POINTS;

    // The visibility region, before any probe. Gated on the MATH thread running
    // ALONE: the reading is about one thread's producer and its own consumer,
    // and a second thread issuing Configuration Unit instructions would be
    // contending for the GPR write that the documented ">= 2" is a latency to
    // ("additional cycles if there is contention for GPR writes"). The host
    // passes reps = 0 for every other variant, so this costs nothing there.
    if (vis_reps != 0 && g_thread == 1 && active_mask == (1u << 1)) {
        run_visibility(base + TTBENCH_VIS_BASE, vis_reps);
    }

    // --- the probe list. Slot numbers are the contract with the host table in
    // --- tensixbench.cpp; keep the two in the same order.

    // Loop overhead: the identical block loop with a body that emits no
    // instructions. `asm volatile` is never deleted, so the loop survives the
    // optimiser -- an empty body does not, and a probe 0 that reads a constant
    // 1 cycle is the signature of that having happened.
    RUN(0, asm volatile(""););

    RUN(1, TTI_NOP;);     // NONE
    RUN(2, TTI_DMANOP;);  // TDMA (miscellaneous unit)

    RUN(3, TTI_SFPNOP;);              // SFPU
    RUN(4, TTI_SFPMOV(0, 0, 1, 0););  // SFPU, doc latency 1
    RUN(5, TTI_SFPADD(0, 1, 2, 3, 0););  // SFPU, doc latency 2
    RUN(6, TTI_SFPMUL(0, 1, 2, 3, 0););  // SFPU, doc latency 2
    RUN(7, TTI_SFPMAD(0, 1, 2, 3, 0););  // SFPU, doc latency 2
    RUN(8, TTI_SFPABS(0, 0, 1, 0););     // SFPU, doc latency 1

    // ThCon. High GPR indices so nothing the LLK uses is clobbered. Reg index
    // is in 16-bit quanta for SETDMAREG and in 32-bit quanta for the rest.
    RUN(9, TTI_SETDMAREG(0, 7, 0, 120););       // doc occupancy 1
    RUN(10, TTI_ADDDMAREG(1, 60, 1, 60););      // doc occupancy "3 or 4"
    RUN(11, TTI_MULDMAREG(1, 61, 1, 60););      // doc occupancy "3 or 4"
    RUN(12, TTI_SHIFTDMAREG(1, 0, 61, 1, 60);); // doc occupancy "3 or 4"
    RUN(13, TTI_CMPDMAREG(1, 0, 62, 1, 60););   // doc occupancy "3 or 4"

    // Config unit. RDCFG is the op that made tt-sim's matmulblock guard compute
    // the wrong answer when charged its documented ">= 2" -- see "The unit that
    // would not go" in docs/plans/cost-model.md. Read-only, into a spare GPR.
    RUN(14, TTI_RDCFG(60, 0););

    // MATH, the two ops that need no SrcA/SrcB data.
    RUN(15, TTI_SETRWC(0, 0, 0, 0, 0, 0););
    RUN(16, TTI_INCRWC(0, 0, 0, 0););

    // MATH, the ops that do. Every op below leaves the valid bits alone
    // (clear_dvalid = 0) so they stay set for the whole burst, and
    // MatrixUnit.SrcABank/SrcBBank therefore never move: no shared state changes
    // inside any timed region, in either mode. See the X1 note in the header.
    if (probe_mask & ((1u << 17) | (1u << 18) | (1u << 19))) {
        if (dvalid_mode == TTBENCH_DVALID_UNPACR_NOP) {
            // X2: the Blackhole-sanctioned setup, at a chosen source format.
            // Same shape as ONCE -- thread 1 only, barrier after -- so the two
            // differ in the setup instruction and nothing else.
            if (g_thread == 1) {
                set_source_format(src_format);
                give_both_srcs_to_fpu();
            }
            bench_barrier();
        } else if (dvalid_mode == TTBENCH_DVALID_ONCE) {
            // One SETDVALID for the whole tile, whatever the thread count, and
            // a barrier so no thread starts probe 17 before it has landed.
            if (g_thread == 1) {
                TTI_SETDVALID(3);
            }
            bench_barrier();
        } else {
            // The original, confounded setup: one per active thread, unordered.
            TTI_SETDVALID(3);
        }
    }
    RUN(17, TTI_MVMUL(0, 0, 0, 0););
    RUN(18, TTI_ELWADD(0, 0, 0, 0, 0););
    RUN(19, TTI_ELWMUL(0, 0, 0, 0, 0););

    // -----------------------------------------------------------------------
    // THE LATENCY DIFFERENCE. Slots 20 and 21, and neither means anything on
    // its own -- the reading is `slot 20 - slot 21`.
    //
    // Every other probe in this phase measures OCCUPANCY: how long the issuing
    // thread is kept out of the unit. Latency -- how long the unit is busy
    // AFTER it has taken the instruction -- is structurally invisible to that,
    // because a pipelined unit releases the issuer immediately. `RDCFG` is the
    // case that made the difference matter: the ISA doc gives it ">= 2", slot
    // 14 measures 1.000 on silicon, and both are true statements about
    // different quantities. Charging the doc's 2 as an occupancy is what made
    // tt-sim's matmulblock guard compute the wrong answer.
    //
    // `docs/plans/tensix-cost-benchmark.md` names the measurable form:
    //
    //     (op + STALLWAIT) minus (a known 1-cycle op + STALLWAIT) cancels the
    //     sync overhead and leaves a latency difference.
    //
    // THE CANCELLATION IS ONLY EXACT IF THE STALLWAIT IS THE SAME INSTRUCTION,
    // and here it is, operand for operand:
    //
    //   stall_res = STALL_THREAD -- block EVERY one of this thread's issue
    //     paths, not just the config path. A narrower `STALL_CFG` would block
    //     slot 20's next instruction (another RDCFG) and NOT slot 21's (a ThCon
    //     op), so the two slots would be stalled by different amounts and
    //     nothing would cancel.
    //   wait_res = TRISC_CFG -- wait for the CONFIG unit to go idle. In slot 20
    //     it has just been given an RDCFG and must drain it; in slot 21 it was
    //     never touched, so the identical instruction clears at its floor.
    //
    // So the difference is the cycles the config unit stayed busy after RDCFG
    // was issued, minus the one cycle SETDMAREG's documented occupancy costs.
    // Read it as a LOWER BOUND on RDCFG's latency: a difference of zero says
    // the config unit released before the stall could observe it, not that the
    // latency is zero.
    //
    // `SETDMAREG` is the baseline because it is the ThCon op this benchmark
    // already documents at occupancy 1 (slot 9) and already measures at exactly
    // 1.000 on silicon, and because it does not touch the config unit -- so its
    // STALLWAIT is genuinely the floor rather than a second measurement.
    //
    // Both bodies are TWO instructions, and `PROBE` unrolls with a literal
    // `REP64`, so a block is 64 PAIRS. The host divides by TTBENCH_UNROLL as it
    // does for every other slot, which makes the reported `cyc/instr` cycles
    // per PAIR here. That is the right divisor for a difference of pairs and
    // the wrong one for a per-instruction occupancy; the host's probe table
    // labels both slots so the summary cannot be misread.
    //
    // AND IT DOES NOT REACH THE QUANTITY. `p_stall::TRISC_CFG` is 0x400 on
    // Blackhole and 0x2000 on Wormhole, i.e. condition C10 and C13, and BOTH
    // architectures define that bit as
    //
    //     "The RISCV T core associated with the current Tensix thread has a
    //      memory read-request or write-request against Tensix GPRs or Tensix
    //      configuration or TDMA-RISC that has been emitted from the RISCV core
    //      but not yet processed."
    //
    // -- which is about stores the baby core issued to `TENSIX_CFG_BASE`, not
    // about a `.ttinsn` sitting in the Configuration Unit's pipeline. The name
    // is the trap: `TRISC_CFG` reads as "the config unit" and means "this
    // TRISC's config *memory requests*". A Blackhole card ran these two slots on
    // 2026-08-09 and measured their difference at 0.0000 cycles/pair while the
    // movement control moved (SETDMAREG + STALLWAIT 2.968 against 0.998 bare),
    // which is precisely what a live stall whose condition is unrelated to the
    // unit under test looks like.
    //
    // THEY ARE KEPT ANYWAY, as the falsification control for slots 22-24. Same
    // harness, same ops, same STALL_THREAD block mask, one bit different in the
    // condition mask. A run in which 20-21 reads ~0 and 22-23 reads ~1 has shown
    // that the difference came from the condition and not from the construction;
    // a run in which BOTH move has shown the opposite and must not be believed.
    //
    // Against tt-sim this reads ~0 for a second, independent reason: its Wait
    // Gate answers an unmapped STALLWAIT condition with "satisfied" (`case _:
    // return True` in tt_sim/pe/tensix/frontend.py) and maps neither arch's
    // TRISC_CFG bit. That is a correct simulator reading, not a broken probe --
    // and it is also why this cannot hang the simulator.
    // -----------------------------------------------------------------------
    RUN(TTBENCH_P_RDCFG_STALL,
        TTI_RDCFG(60, 0);
        TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::TRISC_CFG););
    RUN(TTBENCH_P_SETDMA_STALL,
        TTI_SETDMAREG(0, 7, 0, 120);
        TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::TRISC_CFG););

    // -----------------------------------------------------------------------
    // THE LATENCY DIFFERENCE, DONE PROPERLY. Slots 22, 23, 24 and 25, and the
    // reading is `slot 22 - slot 23`, with `slot 22 - slot 24` beside it.
    //
    // WHICH CONDITION OBSERVES THE CONFIG UNIT. Exactly one, on exactly one
    // architecture. `BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md`
    // tabulates thirteen condition bits and C12 is
    //
    //     "Any thread has an instruction in any stage of the Configuration Unit
    //      pipeline."
    //
    // with the note "The block mask should include bit B7 to prevent new
    // instructions from this thread flowing into the Configuration Unit." B7 is
    // "Block thread from starting new Configuration Unit instructions", and the
    // same page's block table marks `RDCFG`, `RMWCIB`, `WRCFG`, `SETC16`,
    // `CFGSHIFTMASK` and `STREAMWRCFG` as the instructions B7 blocks -- so the
    // condition and the block bit name the same unit, which is what makes the
    // pairing a wait on that unit rather than on a thread.
    //
    // The LLK spells C12 `p_stall::CFGEXU` (0x1000, tt_llk_blackhole's
    // ckernel_instr_params.h). `RDCFG.md` asks for exactly this construction in
    // so many words: "After issuing one or more `RDCFG` instructions, software
    // is encouraged to use `STALLWAIT` to wait for the Configuration Unit to no
    // longer be busy."
    //
    // WHY THIS IS BLACKHOLE ONLY, and why that is not a gap in the probe.
    // Wormhole's condition mask is fifteen bits and NONE of them is about the
    // Configuration Unit: C0 is ThCon memory, C1-C2 the unpackers, C3-C6 the
    // four packers, C7 the FPU, C8-C11 the Src banks, C12 the mover, C13 the
    // RISCV memory request, C14 the SFPU (WormholeB0 STALLWAIT.md). Its
    // `p_stall` has no CFGEXU to emit. But Wormhole does not NEED this probe:
    // `WormholeB0/.../RDCFG.md` says "The issuing thread is blocked for the
    // entire duration", so there the documented ">= 2" is an OCCUPANCY and slot
    // 14 already measures it. Blackhole is the architecture where it went
    // invisible -- `BlackholeA0/.../RDCFG.md` says "The issuing thread is not
    // blocked, so it can potentially start its next instructions (of any kind)
    // during `RDCFG`'s subsequent cycles" -- and Blackhole is the architecture
    // that added C12. The one arch that hides the quantity is the one that
    // publishes the condition for seeing it.
    //
    // WHAT THE DIFFERENCE IS, cycle by cycle, from the documents alone.
    // `ConfigurationUnit.md`'s stage table puts `RDCFG` in stage 0 (config read)
    // and then stage +1 (GPR write), and `RMWCIB` in stage 0 only (config
    // read+write, "This instruction executes in a single cycle"). So with the
    // burst issuing one instruction per cycle:
    //
    //   arm 23   t: RMWCIB passes the Wait Gate, occupies stage 0.
    //          t+1: STALLWAIT passes and latches. The unit is already empty, so
    //               C12 is met at once -- but "there is a one cycle lag between
    //               the condition(s) being met and the block mask being removed
    //               ... the instruction immediately after STALLWAIT will always
    //               be subject to the block mask for at least one cycle".
    //          t+3: the next pair starts.               -> 3 cycles per pair
    //
    //   arm 22   t: RDCFG passes the Wait Gate, occupies stage 0.
    //          t+1: STALLWAIT latches; RDCFG is in stage +1, so C12 is UNMET.
    //          t+2: RDCFG has left; C12 met. Same one cycle lag.
    //          t+4: the next pair starts.               -> 4 cycles per pair
    //
    // The difference is therefore the number of cycles the Configuration Unit
    // stays busy with `RDCFG` AFTER the issue slot the occupancy probe already
    // charges: predicted >= 1 from the documented ">= 2", and larger if there is
    // GPR write contention ("except for `RDCFG`, which can potentially occupy
    // stage 1 for multiple cycles if there is GPR write contention"). Read the
    // measured value as a LOWER BOUND on that residency and nothing else.
    //
    // WHY THE PAIRING ISOLATES IT. Everything except the residency appears in
    // both arms and is subtracted:
    //   * the identical `TTI_STALLWAIT(STALL_THREAD, CFGEXU)` -- operand for
    //     operand, so its own issue cost and its one cycle of lag cancel;
    //   * the same STALL_THREAD (all nine block bits), so both arms' next
    //     instruction is blocked by the same rule, and B7 is present in both,
    //     which a narrower mask could not do symmetrically for slot 24;
    //   * the same RISC-V loop, so the loop overhead cancels without being
    //     measured;
    //   * the issue occupancy, IF the two ops cost the same to issue -- which is
    //     why slot 25 measures RMWCIB0 bare next to slot 14's RDCFG. If those
    //     two disagree, part of the difference is occupancy and the host says so
    //     instead of reporting a latency.
    // What does not cancel is that in arm 22 the unit is still busy when the
    // stall is evaluated and in arm 23 it is not. That is the quantity.
    //
    // THE OPS ARE STILL STATE-FREE. `RDCFG` only reads. `RMWCIB0(Mask=0,
    // NewValue=0, Index4=0)` is the identity by the functional model in
    // `RMWCIB.md`: `*CfgAddress = (NewValue & Mask) | (OldValue & ~Mask)` with
    // `Mask = 0` writes `OldValue` back. Index4 = 0 is in bounds for any
    // `CFG_STATE_SIZE`, so the `UndefinedBehavior()` arm is unreachable.
    //
    // SINGLE THREAD ONLY. C12 says "ANY thread", and the page's own note says
    // "This won't prevent other threads from issuing new Configuration Unit
    // instructions though, and those new instructions will cause this thread to
    // continue to wait." At t2/t3 every thread runs the same burst, so each
    // thread's stall would be observing the others' RDCFGs and the difference
    // would stop being a statement about one instruction. The host grades t1 and
    // says so for the rest; run this probe with `--variants t1`.
    // -----------------------------------------------------------------------
#if defined(ARCH_BLACKHOLE)
    RUN(TTBENCH_P_RDCFG_CFGSTALL,
        TTI_RDCFG(60, 0);
        TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::CFGEXU););
    RUN(TTBENCH_P_RMWCIB_CFGSTALL,
        TTI_RMWCIB0(0, 0, 0);
        TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::CFGEXU););
    RUN(TTBENCH_P_SETDMA_CFGSTALL,
        TTI_SETDMAREG(0, 7, 0, 120);
        TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::CFGEXU););
    RUN(TTBENCH_P_RMWCIB, TTI_RMWCIB0(0, 0, 0););
#else
    // Wormhole has no condition bit for the Configuration Unit, so there is
    // nothing to emit. The host clears these four from `probe_mask` before the
    // launch and prints the reason, so no row is written for them either; the
    // barriers still run so the active threads stay in step.
    PROBE_SKIP(TTBENCH_P_RDCFG_CFGSTALL);
    PROBE_SKIP(TTBENCH_P_RMWCIB_CFGSTALL);
    PROBE_SKIP(TTBENCH_P_SETDMA_CFGSTALL);
    PROBE_SKIP(TTBENCH_P_RMWCIB);
#endif

    // -----------------------------------------------------------------------
    // THE DEPENDENCE PAIR. Slots 26 and 27, and the reading is `26 - 27`.
    //
    // This is the house method -- `perfbench/riscvbench/src/riscvbench.cpp`
    // measures "dependent multiply" at 1.985 cycles and "dependent L1 load" at
    // 8.098 exactly this way -- transplanted to Tensix GPRs, and it is here
    // BECAUSE IT IS FALSIFIABLE, not because it is expected to move. Both arms
    // are the same two opcodes in the same order on the same two units. They
    // differ in one operand field:
    //
    //   26   RDCFG -> GPR 60 ;  ADDDMAREG reads GPR 60   the RDCFG destination
    //   27   RDCFG -> GPR 60 ;  ADDDMAREG reads GPR 59   never written by RDCFG
    //
    // so the issue cost of both instructions, the loop, the ThCon occupancy and
    // the Configuration Unit occupancy are all common and subtract exactly. What
    // does not subtract is the read-after-write.
    //
    // THE PREDICTION IS ZERO, and it comes from the ISA documentation rather
    // than from a hunch. `BlackholeA0/.../RDCFG.md` makes the separation a
    // SOFTWARE obligation -- "Software must ensure that the instruction(s)
    // immediately after `RDCFG` are not trying to consume the GPR written by the
    // `RDCFG` instruction" -- and an obligation on software is the documented
    // absence of an interlock. Wormhole's `ScalarUnit.md` says the same of ThCon
    // from the other side: "there is _usually_ enough latency ... for the GPR
    // write to complete before the instruction starts executing. However, to
    // guarantee race-free operation" software must insert a `STALLWAIT` or an
    // ordering of its own. So the hazard is resolved by value, not by time.
    //
    // WHY RUN IT AT ALL. Because if that reading is wrong -- if Blackhole does
    // interlock -- then `26 - 27` IS RDCFG's latency in cycles, directly, with
    // no apparatus overhead to subtract, and it would be the best possible form
    // of the measurement. Ruling that out costs two slots and rules it out for
    // good. A zero here is what sends the reading to the visibility region.
    //
    // ITS OWN CONTROL is that the pair must cost what the two instructions cost
    // separately: probe 14 (RDCFG bare) plus probe 10 (ADDDMAREG bare), both in
    // the same run. If the pair comes in under that sum, the two arms are not
    // both issuing what they claim to and a zero difference would mean nothing.
    // The host checks it and says so.
    // -----------------------------------------------------------------------
    RUN(TTBENCH_P_RDCFG_DEP,
        TTI_RDCFG(60, 0);
        TTI_ADDDMAREG(1, 63, 0, 60););
    RUN(TTBENCH_P_RDCFG_INDEP,
        TTI_RDCFG(60, 0);
        TTI_ADDDMAREG(1, 63, 0, 59););

    // -----------------------------------------------------------------------
    // THE C12 LIVENESS CONTROL. Slots 28 and 29, graded across VARIANTS rather
    // than within one: the reading is
    //
    //     d(v) = pair(28, v) - pair(29, v)      on thread 1
    //     the verdict is on   d(t3) - d(t1)
    //
    // WHAT IT SETTLES. Slots 22-25 measured 0.0000 on a card. Two explanations
    // survive that, and the visibility region above cannot separate them
    // because it never consults a condition bit:
    //
    //   (i)  C12 works, and RDCFG's post-issue residency (one cycle, stage +1,
    //        the GPR write) is no wider than the stall's own documented lag --
    //        "There is a one cycle lag between the condition(s) being met and
    //        the block mask being removed" -- so it completes inside the
    //        apparatus and no busy-condition could ever have seen it;
    //   (ii) C12 does not behave as documented on this part.
    //
    // Only a C12 signal much WIDER than one cycle tells those apart, and there
    // is exactly one way to make one that the documents support. C12 is "Any
    // thread has an instruction in any stage of the Configuration Unit
    // pipeline", and STALLWAIT.md's note on it says "This won't prevent other
    // threads from issuing new Configuration Unit instructions though, and those
    // new instructions will cause this thread to continue to wait." So the
    // busy-ness has to come from ANOTHER THREAD, where it is not coupled to the
    // waiting thread's own issue slots.
    //
    // Hence the only thread-dependent bodies in this benchmark. At t1 thread 1
    // runs alone and its C12 wait clears at the floor. At t2 and t3 threads 0
    // and 2 hold the Configuration Unit with a back-to-back `RMWCIB0` stream
    // (documented at one cycle, IPC 1, so the unit is saturated) and thread 1's
    // wait must extend if C12 is live.
    //
    // SLOT 29 IS THE SAME CHOREOGRAPHY WITHOUT THE STALL, so the cross-thread
    // issue interference -- which grows with the thread count for every probe
    // here, stall or no stall -- appears in both and subtracts. Without it,
    // "thread 1 got slower at t3" would be the expected reading whether or not
    // C12 exists.
    //
    // IT CANNOT HANG. The hammering threads issue a bounded burst and then wait
    // at `bench_barrier`; once they stop, the Configuration Unit drains, C12
    // clears and thread 1 finishes. The worst case is that thread 1's block
    // costs about as long as the hammer burst, which is linear in `blocks` like
    // everything else here.
    //
    // BOTH OUTCOMES ARE REACHABLE, which is what makes this a control and not a
    // formality: a large positive `d(t3) - d(t1)` says C12 is live and reading
    // (i) holds; a zero says C12 is inert on this part and reading (ii) holds.
    // Neither is the default and neither is the one the run is graded to find.
    // -----------------------------------------------------------------------
#if defined(ARCH_BLACKHOLE)
    if (probe_mask & (1u << TTBENCH_P_C12_XT)) {
        if (g_thread == 1) {
            PROBE(TTBENCH_P_C12_XT,
                  TTI_STALLWAIT(ckernel::p_stall::STALL_THREAD, ckernel::p_stall::CFGEXU);
                  TTI_NOP;);
        } else {
            PROBE(TTBENCH_P_C12_XT, TTI_RMWCIB0(0, 0, 0); TTI_NOP;);
        }
    } else {
        PROBE_SKIP(TTBENCH_P_C12_XT);
    }
    if (probe_mask & (1u << TTBENCH_P_C12_XT_NULL)) {
        if (g_thread == 1) {
            PROBE(TTBENCH_P_C12_XT_NULL, TTI_NOP; TTI_NOP;);
        } else {
            PROBE(TTBENCH_P_C12_XT_NULL, TTI_RMWCIB0(0, 0, 0); TTI_NOP;);
        }
    } else {
        PROBE_SKIP(TTBENCH_P_C12_XT_NULL);
    }
#else
    PROBE_SKIP(TTBENCH_P_C12_XT);
    PROBE_SKIP(TTBENCH_P_C12_XT_NULL);
#endif

    // Give the Src banks back. OUTSIDE every timed region -- the last PROBE has
    // written its last result word before this runs -- so the measurement is
    // untouched; see "LEAVING THE CARD CLEAN" in the header comment.
    //
    // Only the UNPACR_NOP setup: it is the only one with a wait half, so it is
    // the only one a dirty card can wedge. Adding a release to the SETDVALID
    // paths would perturb two datasets that are already banked and buy nothing.
    if (probe_mask & ((1u << 17) | (1u << 18) | (1u << 19))) {
        if (dvalid_mode == TTBENCH_DVALID_UNPACR_NOP) {
            // Ordering, in two steps, because the release is issued by ONE
            // thread and the burst was issued by all of them:
            //   tensix_sync()   -- blocks this RISC-V core until its own Tensix
            //                      instruction pipe has drained, so every MATH
            //                      probe this thread issued has passed the Wait
            //                      Gate. (An op past the gate cannot stall on
            //                      dvalid; only an undispatched one can.)
            //   bench_barrier() -- and now the same is true of every other
            //                      active thread.
            // Without both, thread 1's CLEARDVALID could overtake a still-queued
            // MVMUL on thread 0 or 2 and hang the run it was added to prevent.
            ckernel::tensix_sync();
            bench_barrier();
            if (g_thread == 1) {
                take_both_srcs_back_from_fpu();
                // ...and do not return until it has RETIRED, not merely been
                // issued. The launch ends with the TRISCs going back into soft
                // reset; a release still sitting in the instruction FIFO at
                // that point would be exactly as lost as never issuing it.
                ckernel::tensix_sync();
            }
        }
    }
}
