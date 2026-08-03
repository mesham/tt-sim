// Differential op-coverage kernel for the row-major untilize pack. Every op
// shares the same phase 1 -- matmul the two resident operand tiles into DST[0]
// and pack that *tiled* into the intermediate CB -- and differs only in phase
// 2, how the intermediate tile reaches the output CB. The three phase 2 forms
// are transcribed from the tt-xftn compiler team's hand-validated variants in
// `docs/plans/gemm-ondevice-untilize-kernels.md` (their data points 4, 2 and 3):
//
//   op 0  CONTROL  copy_tile -> DST -> tiled pack_tile     (host untilizes)
//   op 1           untilize_block, CB -> CB
//   op 2           copy_tile -> DST -> pack_untilize_dest, DST -> CB
//
// Ops 1 and 2 are the row-major untilizes: the four 16x16 faces of the tile are
// interleaved into output rows, which needs the pack Y/Z address counters to
// accumulate across the PACRs of the tile. Op 0 packs each face at Y = 0 where
// set == accumulate, so it stays correct either way and isolates the fault to
// the untilize pack rather than the matmul, the intermediate CB or copy_tile.
//
// To cover another form: add a phase 2 below and bump NUM_OPS in the host
// (untilize.cpp) to match.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/untilize.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;
constexpr auto cb_interm = tt::CBIndex::c_24;
constexpr auto cb_out = tt::CBIndex::c_16;

// Phase 1, common to every op: C = in0 * in1 into DST[0], packed tiled into the
// intermediate CB. in1 is the identity tile, so C is in0 unchanged.
inline void matmul_to_interm() {
    matmul_init(cb_in0, cb_in1);
    tile_regs_acquire();
    matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_interm, 1);
    pack_tile(0, cb_interm);
    cb_push_back(cb_interm, 1);
    tile_regs_release();
}

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);

    // Both operands stay resident (never popped), so every op re-runs the same
    // matmul over the same tiles.
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    // --- op 0: CONTROL. Tiled repack, so the host untilizes this one. ---
    matmul_to_interm();
    copy_tile_to_dst_init_short(cb_interm);
    cb_wait_front(cb_interm, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile(cb_interm, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_interm, 1);

    // --- op 1: CB -> CB row-major untilize. ---
    matmul_to_interm();
    untilize_init(cb_interm);
    cb_wait_front(cb_interm, 1);
    cb_reserve_back(cb_out, 1);
    untilize_block(cb_interm, 1, cb_out);
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_interm, 1);
    untilize_uninit(cb_interm);

    // --- op 2: DST -> CB row-major untilize. ---
    matmul_to_interm();
    copy_tile_to_dst_init_short(cb_interm);
    pack_untilize_dest_init<1, 1>(cb_out);
    cb_wait_front(cb_interm, 1);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    copy_tile(cb_interm, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_untilize_dest<1, 1>(cb_out, 1, 0);
    tile_regs_release();
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_interm, 1);
    pack_untilize_uninit(cb_out);
    // --- end op sequence (keep NUM_OPS in untilize.cpp in sync: NUM_OPS = 3) ---

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}
