// `matmul_block` over a multi-tile block -- the block form of the operand
// addressing `optests/matmulidx` covers for `matmul_tiles`.
//
// Both operand CBs hold a whole 2x2 tile block, resident at once:
//     cb_in0 (A, row-major rt x kt): tiles 1.0, 2.0, 4.0, 8.0
//     cb_in1 (B, row-major kt x ct): tiles 3.0, 5.0, 16.0, 32.0
// Every output element of C = A @ B is then 32 * sum_k A[r][k] * B[k][c].
//
// One `matmul_block` call is a single K step: it walks `rt_dim` operand-A tiles
// (stride `kt_dim`) against `ct_dim` consecutive operand-B tiles, so the kernel
// loops K itself and accumulates into the same DST.
//
// Three shapes, covering both halves of the LLK's `reuse_a = ct_dim >= rt_dim`
// split as well as the tt-metal shape the compiler team reported:
//   1. ct=2, rt=1  -- reuse A, one output row per call (their `gemm` shape)
//   2. ct=2, rt=2  -- reuse A, the whole 2x2 block in one call
//   3. ct=1, rt=2  -- reuse B (the `!reuse_a` branch), one output column per call

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t RT = 2;  // operand A is RT x KT tiles
constexpr uint32_t CT = 2;  // operand B is KT x CT tiles
constexpr uint32_t KT = 2;
constexpr uint32_t NUM_IN_TILES = RT * KT;

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);

    // Hold both operand blocks resident, so the blocks below can index into
    // them rather than streaming a tile at a time.
    cb_wait_front(cb_in0, NUM_IN_TILES);
    cb_wait_front(cb_in1, NUM_IN_TILES);

    // 1. ct_dim=2, rt_dim=1: one output tile-row per call, K looped by hand.
    matmul_block_init(cb_in0, cb_in1, 0, CT, 1, KT);
    for (uint32_t r = 0; r < RT; r++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KT; k++) {
            matmul_block(cb_in0, cb_in1, r * KT + k, k * CT, 0, false, CT, 1, KT);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, CT);
        for (uint32_t c = 0; c < CT; c++) {
            pack_tile(c, cb_out);
        }
        cb_push_back(cb_out, CT);
        tile_regs_release();
    }

    // 2. ct_dim=2, rt_dim=2: the whole output block in one call per K step.
    matmul_block_init(cb_in0, cb_in1, 0, CT, RT, KT);
    tile_regs_acquire();
    for (uint32_t k = 0; k < KT; k++) {
        matmul_block(cb_in0, cb_in1, k, k * CT, 0, false, CT, RT, KT);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, RT * CT);
    for (uint32_t t = 0; t < RT * CT; t++) {
        pack_tile(t, cb_out);
    }
    cb_push_back(cb_out, RT * CT);
    tile_regs_release();

    // 3. ct_dim=1, rt_dim=2: the reuse-B branch, one output tile-column per call.
    matmul_block_init(cb_in0, cb_in1, 0, 1, RT, KT);
    for (uint32_t c = 0; c < CT; c++) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KT; k++) {
            matmul_block(cb_in0, cb_in1, k, k * CT + c, 0, false, 1, RT, KT);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, RT);
        for (uint32_t r = 0; r < RT; r++) {
            pack_tile(r, cb_out);
        }
        cb_push_back(cb_out, RT);
        tile_regs_release();
    }

    cb_pop_front(cb_in0, NUM_IN_TILES);
    cb_pop_front(cb_in1, NUM_IN_TILES);
}
