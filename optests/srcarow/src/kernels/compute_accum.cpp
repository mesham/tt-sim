#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

// w = Dt*ur + us*D + ut*D
// Three SFPU matmuls, accumulated.
// Reader sends broadcast tiles for each matmul sequentially via cb_left/cb_right.

namespace {

void sfpu_matmul(
    tt::CBIndex cb_left, tt::CBIndex cb_right,
    tt::CBIndex cb_accum, tt::CBIndex cb_tmp,
    tt::CBIndex cb_out
) {
    constexpr int N = 16;
    for (int k = 0; k < N; k++) {
        mul_tiles_init(cb_left, cb_right);
        tile_regs_acquire();
        mul_tiles(cb_left, cb_right, k, k, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (k == 0) {
            cb_reserve_back(cb_accum, 1);
            pack_tile(0, cb_accum);
            cb_push_back(cb_accum, 1);
        } else {
            cb_reserve_back(cb_tmp, 1);
            pack_tile(0, cb_tmp);
            cb_push_back(cb_tmp, 1);
            tile_regs_release();
            cb_wait_front(cb_accum, 1);
            cb_wait_front(cb_tmp, 1);
            add_tiles_init(cb_accum, cb_tmp);
            tile_regs_acquire();
            add_tiles(cb_accum, cb_tmp, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(cb_accum, 1);
            cb_pop_front(cb_tmp, 1);
            cb_reserve_back(cb_accum, 1);
            pack_tile(0, cb_accum);
            cb_push_back(cb_accum, 1);
        }
        tile_regs_release();
    }
    cb_wait_front(cb_accum, 1);
    copy_tile_init(cb_accum);
    tile_regs_acquire();
    copy_tile(cb_accum, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_accum, 1);
}

}  // namespace

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // Reader sends 3 rounds of 16+16 broadcast tiles per z-layer tile through these CBs
    constexpr tt::CBIndex cb_left    = tt::CBIndex::c_0;   // left col broadcasts (16 tiles per matmul)
    constexpr tt::CBIndex cb_right   = tt::CBIndex::c_1;   // right row broadcasts (16 tiles per matmul)
    constexpr tt::CBIndex cb_accum   = tt::CBIndex::c_4;   // inner matmul accumulator
    constexpr tt::CBIndex cb_tmp     = tt::CBIndex::c_5;   // inner matmul temporary
    constexpr tt::CBIndex cb_mm_out  = tt::CBIndex::c_6;   // single matmul result
    constexpr tt::CBIndex cb_sum     = tt::CBIndex::c_7;   // running sum of matmul results
    constexpr tt::CBIndex cb_w       = tt::CBIndex::c_16;  // final output

    // One-time HW configure (unpack/math/pack formats, fp32 dest-acc pack sync).
    // Required before any *_tiles_init/op call; without it the pack/unpack HW is
    // never programmed and every packed tile reads back as 1.0f on silicon.
    // All CBs here are Float32 with identical tile geometry, so this single call
    // covers the accum/tmp/sum/mm_out CBs too.
    binary_op_init_common(cb_left, cb_right, cb_w);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Matmul 1: Dt * ur
        cb_wait_front(cb_left, 16);    // Dt col broadcasts
        cb_wait_front(cb_right, 16);   // ur row broadcasts
        sfpu_matmul(cb_left, cb_right, cb_accum, cb_tmp, cb_sum);
        cb_pop_front(cb_left, 16);
        cb_pop_front(cb_right, 16);

        // Matmul 2: us * D
        cb_wait_front(cb_left, 16);    // us col broadcasts
        cb_wait_front(cb_right, 16);   // D row broadcasts
        sfpu_matmul(cb_left, cb_right, cb_accum, cb_tmp, cb_mm_out);
        cb_pop_front(cb_left, 16);
        cb_pop_front(cb_right, 16);

        // sum += matmul2
        cb_wait_front(cb_sum, 1);
        cb_wait_front(cb_mm_out, 1);
        add_tiles_init(cb_sum, cb_mm_out);
        tile_regs_acquire();
        add_tiles(cb_sum, cb_mm_out, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_sum, 1);
        cb_pop_front(cb_mm_out, 1);
        cb_reserve_back(cb_sum, 1);
        pack_tile(0, cb_sum);
        cb_push_back(cb_sum, 1);
        tile_regs_release();

        // Matmul 3: ut * D
        cb_wait_front(cb_left, 16);    // ut col broadcasts
        cb_wait_front(cb_right, 16);   // D row broadcasts
        sfpu_matmul(cb_left, cb_right, cb_accum, cb_tmp, cb_mm_out);
        cb_pop_front(cb_left, 16);
        cb_pop_front(cb_right, 16);

        // w = sum + matmul3
        cb_wait_front(cb_sum, 1);
        cb_wait_front(cb_mm_out, 1);
        add_tiles_init(cb_sum, cb_mm_out);
        tile_regs_acquire();
        add_tiles(cb_sum, cb_mm_out, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_pop_front(cb_sum, 1);
        cb_pop_front(cb_mm_out, 1);
        cb_reserve_back(cb_w, 1);
        pack_tile(0, cb_w);
        cb_push_back(cb_w, 1);
        tile_regs_release();
    }
}
