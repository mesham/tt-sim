// Phase B of the Tensix cycle-cost benchmark: the fidelity-phase claim.
//
// docs/plans/cost-model.md derives the single most load-bearing number in the
// MATH table this way: one MVMUL is eight rows of SrcB against a 16x16 SrcA, so
// a 32x32x32 tile matmul is 16 MVMULs *per fidelity phase*
// (`fidelity_phases.mvmuls_per_tile`), and the fidelity multiplier is carried by
// the instruction stream rather than by a longer instruction -- so an MVMUL
// costs 1 cycle at every phase and a tile matmul costs 16 / 32 / 48 / 64 cycles
// at LoFi / HiFi2 / HiFi3 / HiFi4. Nothing has ever checked that against
// hardware.
//
// What this kernel times is one iteration of a real matmul inner loop, which is
// NOT 16 MVMULs: it is 16 MVMULs plus a circular-buffer wait and pop on both
// operands, the LLK's unpack of both tiles, and the math/unpack semaphore
// handshake. That composite is a confounded number and is deliberately NOT
// reported as a per-instruction cost.
//
// The measurement that is clean is the DIFFERENCE between fidelities. Raising
// math fidelity changes the MVMUL count and nothing else -- same operands, same
// unpack, same circular buffers, same handshake -- so
//
//     slope(HiFi2) - slope(LoFi)  should be 16 cycles
//     slope(HiFi4) - slope(HiFi2) should be 32 cycles
//
// and those two numbers are the prediction. This is the same trick rung 1 used
// on tt-metal's end-to-end DRAM figures: choose two measurements so that
// everything you cannot model cancels.
//
// THE CONFOUND THAT WOULD INVALIDATE IT is the feeder. If the loop is limited by
// mm_feeder.cpp pushing pages rather than by the math thread, raising fidelity
// buys back slack instead of adding time and all three slopes come out equal.
// That is why the host refuses a phase B run whose fidelity slopes do not
// separate, rather than reporting a difference of zero as a result.
//
// The operand tiles are never written by anyone. Values are irrelevant to
// timing and leaving them uninitialised is what lets phase B run with no DRAM
// buffer and no golden.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

#include "bench_layout.h"

namespace {

inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return p[0];
}

}  // namespace

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(0);
    const uint32_t base_iters = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    uint32_t thread = 2;
#if defined(TRISC_UNPACK)
    thread = 0;
#elif defined(TRISC_MATH)
    thread = 1;
#endif

    volatile tt_l1_ptr uint32_t* base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    if (thread == 1) {
        base[TTBENCH_HDR_MAGIC] = TTBENCH_MAGIC;
        base[TTBENCH_HDR_UNROLL] = 1;  // one matmul_tiles iteration, not a block
        base[TTBENCH_HDR_NUM_PROBES] = 1;
        base[TTBENCH_HDR_NUM_POINTS] = TTBENCH_MM_NUM_POINTS;
        base[TTBENCH_HDR_BASE_BLOCKS] = base_iters;
        base[TTBENCH_HDR_ACTIVE_MASK] = 0x7;
        base[TTBENCH_HDR_PROBE_MASK] = 0x1;
    }
    volatile tt_l1_ptr uint32_t* out =
        base + TTBENCH_HDR_WORDS + thread * TTBENCH_NUM_PROBES * TTBENCH_NUM_POINTS;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t p = 0; p < TTBENCH_MM_NUM_POINTS; p++) {
        const uint32_t iters = base_iters * (p + 1);
        acquire_dst();
        const uint32_t t0 = wall_clock_lo();
        for (uint32_t i = 0; i < iters; i++) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
        out[p] = wall_clock_lo() - t0;
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        release_dst();
    }
}
