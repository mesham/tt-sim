// Differential op-coverage kernel for `reduce_tile`. Reduces the *same*
// resident input tile once per (PoolType, ReduceDim) combination, packing one
// output tile per reduction, so a mismatch's element index maps straight back
// to which reduction disagreed.
//
// What each combination lowers to on Blackhole (tt-llk llk_math_reduce.h):
//   MAX  -> GMPOOL,  SUM -> GAPOOL (one per face)
//   REDUCE_COL    -> pool only
//   REDUCE_ROW    -> pool, then MOVD2B + TRNSPSRCB + ZEROSRC + ELWADD to turn
//                    the pooled row into a column in Dst
//   REDUCE_SCALAR -> pool every face into one, then MOVD2B + TRNSPSRCB +
//                    MOVB2A + ZEROACC + a final pool
//
// To cover another combination: add a RUN_REDUCE(...) below and bump NUM_OPS in
// the host (reduce.cpp) to match.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_scaler = tt::CBIndex::c_1;
constexpr auto cb_out = tt::CBIndex::c_16;

// One op = (re)configure the reduce pipeline, reduce the held input tile into
// DST[0], pack DST[0] to the next cb_out page, then drop the packer's reduce
// edge mask again so the next op starts from the default state.
#define RUN_REDUCE(POOL, DIM)                              \
    do {                                                   \
        reduce_init<POOL, DIM>(cb_in, cb_scaler, cb_out);  \
        acquire_dst();                                     \
        reduce_tile<POOL, DIM>(cb_in, cb_scaler, 0, 0, 0); \
        cb_reserve_back(cb_out, 1);                        \
        pack_tile(0, cb_out);                              \
        cb_push_back(cb_out, 1);                           \
        release_dst();                                     \
        reduce_uninit();                                   \
    } while (0)

void kernel_main() {
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);

    // Both operands stay resident (never popped), so every op reduces the same
    // input tile with the same scaler.
    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_scaler, 1);

    // --- op sequence (keep NUM_OPS in reduce.cpp in sync: NUM_OPS = 5) ---
    RUN_REDUCE(PoolType::MAX, ReduceDim::REDUCE_COL);     // 0: GMPOOL, pool only
    RUN_REDUCE(PoolType::MAX, ReduceDim::REDUCE_ROW);     // 1: GMPOOL + SrcB transpose
    RUN_REDUCE(PoolType::MAX, ReduceDim::REDUCE_SCALAR);  // 2: GMPOOL + MOVB2A + ZEROACC
    RUN_REDUCE(PoolType::SUM, ReduceDim::REDUCE_COL);     // 3: GAPOOL, pool only
    RUN_REDUCE(PoolType::SUM, ReduceDim::REDUCE_SCALAR);  // 4: GAPOOL + MOVB2A + ZEROACC
    // --- end op sequence ---

    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_scaler, 1);
}
