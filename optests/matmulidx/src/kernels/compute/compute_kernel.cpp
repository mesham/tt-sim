// Matmuls the four (in0_tile, in1_tile) index combinations over two resident
// operand tiles each, packing one output tile per combination, then a fifth
// tile accumulating two distinct-index matmuls into one DST. Distinct operand
// tile indices are the case this op test exists for; the two equal-index
// combinations are the control.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t NUM_IN_TILES = 2;
constexpr uint32_t NUM_SINGLE = 4;

// Must match IN0_INDICES / IN1_INDICES in the host (matmulidx.cpp).
constexpr uint32_t in0_indices[NUM_SINGLE] = {0, 1, 1, 0};
constexpr uint32_t in1_indices[NUM_SINGLE] = {0, 1, 0, 1};

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);
    matmul_init(cb_in0, cb_in1);

    // Hold the whole operand block resident rather than streaming a tile at a
    // time, so the matmuls below can index into it.
    cb_wait_front(cb_in0, NUM_IN_TILES);
    cb_wait_front(cb_in1, NUM_IN_TILES);

    for (uint32_t i = 0; i < NUM_SINGLE; i++) {
        acquire_dst();
        matmul_tiles(cb_in0, cb_in1, in0_indices[i], in1_indices[i], 0);
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        release_dst();
    }

    // One more output tile, accumulating two distinct-index matmuls into the
    // same DST — the shape a blocked GEMM's K loop has.
    acquire_dst();
    matmul_tiles(cb_in0, cb_in1, 0, 1, 0);
    matmul_tiles(cb_in0, cb_in1, 1, 0, 0);
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    release_dst();

    cb_pop_front(cb_in0, NUM_IN_TILES);
    cb_pop_front(cb_in1, NUM_IN_TILES);
}
