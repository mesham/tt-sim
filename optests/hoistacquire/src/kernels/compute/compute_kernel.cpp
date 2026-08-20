// Two matmul-and-pack loops that differ in exactly one thing: where
// `tile_regs_acquire()` sits.
//
//   PLAIN    -- acquire/commit/wait/pack/release once per output tile, the
//               canonical form every in-tree example uses.
//   HOISTED  -- the acquire is lifted *above* the loop and issued once; the
//               loop body still commits, waits, packs and releases every
//               iteration. This is the shape the hedgehope compiler team's
//               4-core bf16 GEMM codegen emits (`matmul_init` and
//               `tile_regs_acquire` hoisted out of the output-tile loop), and
//               the shape their 2-core codegen does *not*.
//
// Both loops compute the same NUM_OUT output tiles from the same resident
// operands, so the two halves of the DRAM dump must be identical: the only
// variable is the acquire.
//
// Why the shape is interesting. `tile_regs_acquire()` is
// `_llk_math_wait_for_dest_available_()`, i.e. `math_dest_wait()`, a SEMWAIT
// that stalls the math thread while the MATH_PACK semaphore is at its max
// (2 in SyncHalf). `tile_regs_commit()` posts that semaphore and flips DEST to
// the other half; `tile_regs_release()` gets it back on the pack side. Hoisting
// the acquire therefore leaves the semaphore accounting balanced (N posts, N
// gets) but removes the math thread's *back-pressure*: nothing stops math from
// flipping back onto a DEST half the packer has not drained yet.
//
// A[i][k] is a scaled identity and B is one ramp tile, so output tile i is
// exactly SUM_k scale[i][k] * B -- every scale is a power of two, so both the
// per-matmul products and their sum are exact in bfloat16, and each output
// tile has a distinct scale. A stale, duplicated or half-written tile cannot
// hide.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_a = tt::CBIndex::c_0;     // A, tiled, NUM_OUT * KT tiles resident
constexpr auto cb_b = tt::CBIndex::c_1;     // B, tiled, KT tiles resident
constexpr auto cb_out = tt::CBIndex::c_16;  // packed output, streamed to DRAM

constexpr uint32_t NUM_OUT = 6;  // MUST match NUM_OUT in hoistacquire.cpp
constexpr uint32_t KT = 2;       // MUST match KT in hoistacquire.cpp

#ifndef RUN_PLAIN
#define RUN_PLAIN 1
#endif
#ifndef RUN_HOISTED
#define RUN_HOISTED 1
#endif

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b);

    // Whole operand block resident, indexed directly -- the blocked-GEMM shape.
    cb_wait_front(cb_a, NUM_OUT * KT);
    cb_wait_front(cb_b, KT);

#if RUN_PLAIN
    // Reference form: the DST acquire/commit/wait/release cycle is per tile.
    for (uint32_t i = 0; i < NUM_OUT; i++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KT; k++) {
            matmul_tiles(cb_a, cb_b, i * KT + k, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
#endif

#if RUN_HOISTED
    // The reported form: one acquire for the whole loop.
    tile_regs_acquire();
    for (uint32_t i = 0; i < NUM_OUT; i++) {
        for (uint32_t k = 0; k < KT; k++) {
            matmul_tiles(cb_a, cb_b, i * KT + k, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
#endif

    cb_pop_front(cb_a, NUM_OUT * KT);
    cb_pop_front(cb_b, KT);
}
