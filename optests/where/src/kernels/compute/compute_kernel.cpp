// Differential op-coverage kernel for `where_tile` — the ternary
// `cond ? a : b` select. Both ops use the same three resident input tiles, with
// the value operands swapped, so each output tile covers both branches of the
// select and the two tiles are elementwise complements of one another.
//
// On Blackhole the where kernel (tt-llk ckernel_sfpu_where.h) issues its
// per-lane selects as SFPLOADMACRO sequences; both sims decline that op, so
// `optests/where/env` sets TT_METAL_DISABLE_SFPLOADMACRO=1 and this program
// covers the non-macro SFPU select sequence instead.
//
// To cover another op: add a block below and bump NUM_OPS in the host
// (where.cpp) to match.

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/tile_move_copy.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_cond = tt::CBIndex::c_0;
constexpr auto cb_a = tt::CBIndex::c_1;
constexpr auto cb_b = tt::CBIndex::c_2;
constexpr auto cb_out = tt::CBIndex::c_16;

// One op = load the condition into DST[0] and the two value tiles into DST[1]
// and DST[2], select into DST[2], and pack that out. Output-over-idst2 is the
// in-place form ttnn's own kernels use.
#define RUN_WHERE(CB_TRUE, CB_FALSE)                            \
    do {                                                        \
        tile_regs_acquire();                                    \
        copy_tile(cb_cond, 0, 0);                               \
        copy_tile(CB_TRUE, 0, 1);                               \
        copy_tile(CB_FALSE, 0, 2);                              \
        where_tile<DataFormat::Int32>(0, 1, 2, 2);              \
        tile_regs_commit();                                     \
        cb_reserve_back(cb_out, 1);                             \
        tile_regs_wait();                                       \
        pack_tile(2, cb_out);                                   \
        tile_regs_release();                                    \
        cb_push_back(cb_out, 1);                                \
    } while (0)

void kernel_main() {
    init_sfpu(cb_cond, cb_out);
    cb_wait_front(cb_cond, 1);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);

    where_tile_init();

    // --- op sequence (keep NUM_OPS in where.cpp in sync: NUM_OPS = 2) ---
    RUN_WHERE(cb_a, cb_b);  // 0: cond ? a : b
    RUN_WHERE(cb_b, cb_a);  // 1: cond ? b : a
    // --- end op sequence ---

    cb_pop_front(cb_cond, 1);
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
