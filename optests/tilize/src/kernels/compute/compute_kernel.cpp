// Differential op-coverage kernel for the row-major *tilize* unpack. Each op
// turns one resident input into output tiles in the tiled layout:
//
//   op 0  CONTROL  copy_tile of an already-tiled tile -> pack_tile
//   op 1           tilize_block(cb_rm1, 1, cb_out)
//   op 2           tilize_block(cb_rm2, 2, cb_out)
//
// Ops 1 and 2 go through `llk_unpack_tilize`, which sets `Tileize_mode` and a
// `RowStride` of `2 * 32 * block` bytes for bfloat16, so the unpacker walks L1
// in `UnpackRowWidth`-datum runs `RowStride` apart instead of contiguously.
// `UnpackRowWidth` is 16 on Wormhole and 32 on Blackhole, making op 1's stride
// (64 bytes) strided on Wormhole but exactly contiguous on Blackhole -- op 2
// (128 bytes) is strided on both, which is why both widths are here. Op 0
// shares none of that addressing and must pass either way.
//
// To cover another form: add a block below and bump NUM_OPS in the host
// (tilize.cpp) to match.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_tiled = tt::CBIndex::c_0;
constexpr auto cb_rm1 = tt::CBIndex::c_1;
constexpr auto cb_rm2 = tt::CBIndex::c_2;
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t WIDE_TILES = 2;  // MUST match WIDE_TILES in tilize.cpp

void kernel_main() {
    compute_kernel_hw_startup(cb_tiled, cb_out);

    // --- op 0: CONTROL. Already tiled, so this is a plain copy and repack. ---
    copy_tile_to_dst_init_short(cb_tiled);
    cb_wait_front(cb_tiled, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile(cb_tiled, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_tiled, 1);

    // --- op 1: one-tile-wide row-major block -> one tiled tile. ---
    tilize_init(cb_rm1, 1, cb_out);
    cb_wait_front(cb_rm1, 1);
    cb_reserve_back(cb_out, 1);
    tilize_block(cb_rm1, 1, cb_out);
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_rm1, 1);
    tilize_uninit(cb_rm1, cb_out);

    // --- op 2: two-tile-wide row-major block -> two tiled tiles. ---
    tilize_init(cb_rm2, WIDE_TILES, cb_out);
    cb_wait_front(cb_rm2, WIDE_TILES);
    cb_reserve_back(cb_out, WIDE_TILES);
    tilize_block(cb_rm2, WIDE_TILES, cb_out);
    cb_push_back(cb_out, WIDE_TILES);
    cb_pop_front(cb_rm2, WIDE_TILES);
    tilize_uninit(cb_rm2, cb_out);
    // --- end op sequence (keep NUM_OPS in tilize.cpp in sync: NUM_OPS = 3) ---
}
