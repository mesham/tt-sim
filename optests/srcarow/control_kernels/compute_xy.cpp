#include <cstdint>
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

// SFPU matmul via outer products:
//   C = sum_{k=0}^{15} left_col_bcast[k] .* right_row_bcast[k]

namespace {

void sfpu_matmul(
    tt::CBIndex cb_left,     // 16 broadcast tiles (column broadcasts of left operand)
    tt::CBIndex cb_right,    // 16 broadcast tiles (row broadcasts of right operand)
    tt::CBIndex cb_accum,    // 1-tile scratch for accumulation
    tt::CBIndex cb_tmp,      // 1-tile scratch for intermediate product
    tt::CBIndex cb_out       // 1-tile output
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

    // Move final accumulator to output
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

    constexpr tt::CBIndex cb_D_col   = tt::CBIndex::c_0;   // D column broadcasts (16 tiles)
    constexpr tt::CBIndex cb_u_row   = tt::CBIndex::c_1;   // u row broadcasts (16 per tile)
    constexpr tt::CBIndex cb_u_col   = tt::CBIndex::c_2;   // u column broadcasts (16 per tile)
    constexpr tt::CBIndex cb_Dt_row  = tt::CBIndex::c_3;   // Dt row broadcasts (16 tiles)
    constexpr tt::CBIndex cb_accum   = tt::CBIndex::c_4;
    constexpr tt::CBIndex cb_tmp     = tt::CBIndex::c_5;
    constexpr tt::CBIndex cb_ur      = tt::CBIndex::c_16;
    constexpr tt::CBIndex cb_us      = tt::CBIndex::c_17;

    // One-time HW configure (unpack/math/pack formats, fp32 dest-acc pack sync).
    // Required before any *_tiles_init/op call; without it the pack/unpack HW is
    // never programmed and every packed tile reads back as 1.0f on silicon.
    // All CBs here are Float32 with identical tile geometry, so this single call
    // covers the accum/tmp/output CBs too.

    cb_wait_front(cb_D_col, 16);
    cb_wait_front(cb_Dt_row, 16);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        cb_wait_front(cb_u_row, 16);
        cb_wait_front(cb_u_col, 16);

        // ur = D * u
        sfpu_matmul(cb_D_col, cb_u_row, cb_accum, cb_tmp, cb_ur);

        // us = u * Dt
        sfpu_matmul(cb_u_col, cb_Dt_row, cb_accum, cb_tmp, cb_us);

        cb_pop_front(cb_u_row, 16);
        cb_pop_front(cb_u_col, 16);
    }
}
