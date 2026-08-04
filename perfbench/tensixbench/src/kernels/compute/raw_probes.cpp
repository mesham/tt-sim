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
// SrcA/SrcB data-valid bits, so the kernel sets those once with SETDVALID and
// issues every op with clear_dvalid = 0 so they stay set. They are last, and
// gated by `probe_mask`, so a hang bisects.
//
// Result buffer layout is in bench_layout.h, shared with the host.

#include <cstdint>

#include "api/compute/common.h"
#include "hostdevcommon/kernel_structs.h"

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

    // MATH, the ops that do. SETDVALID once; every op below leaves the valid
    // bits alone (clear_dvalid = 0) so they stay set for the whole burst.
    if (probe_mask & ((1u << 17) | (1u << 18) | (1u << 19))) {
        TTI_SETDVALID(3);
    }
    RUN(17, TTI_MVMUL(0, 0, 0, 0););
    RUN(18, TTI_ELWADD(0, 0, 0, 0, 0););
    RUN(19, TTI_ELWMUL(0, 0, 0, 0, 0););
}
