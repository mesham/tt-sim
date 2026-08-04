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
// only, the ThCon ones touch high GPR indices only, RDCFG only reads. The three
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
