// A loop of `matmul_tiles` -> `pack_tile` -> `untilize_block`, with and without
// the `matmul_init` that has to be re-issued inside the loop.
//
// This is the output-side twin of `optests/tilizematmul`. `untilize_init`
// reprograms the unpacker MOP (and the math thread's MOP) for the untilize
// datacopy, and `untilize_uninit` does not put the matmul programming back --
// it only restores the tile descriptor and the unpack config word. So iteration
// i's untilize leaves iteration i+1's `matmul_tiles` running the *untilize*
// MOP: the first output tile is right and the second wedges on silicon.
//
//   REISSUE_MATMUL_INIT=1 (default) -- `matmul_init` at the top of every
//       output-tile iteration, which is where the hedgehope compiler team's
//       codegen now puts it (immediately before `tile_regs_acquire`).
//   REISSUE_MATMUL_INIT=0 -- `matmul_init` hoisted above the loop, issued once.
//       Deadlocks on silicon from the *second* output tile (Watcher: CWFW, W,
//       UPMD, MWDD, K with the tiled operand CBs rcv != ack).
//
// A[i] = (i + 1) x identity and B is 1024 distinct exactly-representable
// bfloat16 values, so output tile i is exactly (i + 1) * B and the two
// iterations have different, exactly checkable answers -- an iteration that
// silently reran the previous tile, or copied an operand instead of
// multiplying, is unmistakable.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/untilize.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_a = tt::CBIndex::c_0;       // A, tiled
constexpr auto cb_b = tt::CBIndex::c_1;       // B, tiled
constexpr auto cb_interm = tt::CBIndex::c_24; // matmul result, tiled
constexpr auto cb_out = tt::CBIndex::c_16;    // untilized, row-major

constexpr uint32_t NUM_OUT = 2;  // MUST match NUM_OUT in matmuluntilize.cpp

#ifndef REISSUE_MATMUL_INIT
#define REISSUE_MATMUL_INIT 1
#endif

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);

    cb_wait_front(cb_a, NUM_OUT);
    cb_wait_front(cb_b, 1);

#if !REISSUE_MATMUL_INIT
    // The bug: issued once, above the loop. The untilize at the end of each
    // iteration undoes it, and nothing puts it back.
    matmul_init(cb_a, cb_b, 0);
#endif

    for (uint32_t i = 0; i < NUM_OUT; i++) {
#if REISSUE_MATMUL_INIT
        matmul_init(cb_a, cb_b, 0);
#endif
        tile_regs_acquire();
        matmul_tiles(cb_a, cb_b, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_interm, 1);
        pack_tile(0, cb_interm);
        cb_push_back(cb_interm, 1);
        tile_regs_release();

        cb_wait_front(cb_interm, 1);
        cb_reserve_back(cb_out, 1);
        untilize_init(cb_interm);
        untilize_block(cb_interm, 1, cb_out);
        untilize_uninit(cb_interm);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_interm, 1);
    }

    cb_pop_front(cb_a, NUM_OUT);
    cb_pop_front(cb_b, 1);
}
