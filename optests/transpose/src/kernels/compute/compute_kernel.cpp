// Differential op-coverage kernel for the tile-transpose / Dst<->Src move path.
// Each op transforms the *same* resident Float32 input tile and packs its own
// output tile, so a mismatch's element index maps straight back to which op
// disagreed.
//
// What each op lowers to on Blackhole (tt-llk llk_math_transpose_dest.h,
// llk_math_eltwise_binary.h):
//   0: transpose_tile      -> unpacker face transpose + the in-Dst 4x 16x16
//                             face transpose (transpose_of_faces=false,
//                             is_32bit=true): MOVD2B / TRNSPSRCB / MOVB2A /
//                             MOVB2D / MOVA2D under SrcA format switching
//   1: transpose_dest      -> the full 32x32 in-Dst transpose
//                             (transpose_of_faces=true, is_32bit=true)
//   2: DEST_TO_SRCA binary -> MOVD2A (move a Dst face back into SrcA) + ELWADD
//
// To cover another op: add a block below and bump NUM_OPS in the host
// (transpose.cpp) to match.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_out = tt::CBIndex::c_16;

#define PACK_RESULT()               \
    do {                            \
        cb_reserve_back(cb_out, 1); \
        pack_tile(0, cb_out);       \
        cb_push_back(cb_out, 1);    \
    } while (0)

void kernel_main() {
    compute_kernel_hw_startup(cb_in, cb_out);

    // The input tile stays resident (never popped), so every op sees it.
    cb_wait_front(cb_in, 1);

    // --- op sequence (keep NUM_OPS in transpose.cpp in sync: NUM_OPS = 3) ---

    // 0: CB -> DST transpose. For a 32-bit dst format this unpacks straight to
    // Dst and then runs the within-face transpose in the matrix unit.
    transpose_init(cb_in);
    acquire_dst();
    transpose_tile(cb_in, 0, 0);
    PACK_RESULT();
    release_dst();

    // 1: in-place 32x32 transpose of a tile already in DST, faces included.
    copy_tile_init(cb_in);
    acquire_dst();
    copy_tile(cb_in, 0, 0);
    transpose_dest_init<true /* is_32bit */, true /* transpose_of_faces */>(cb_in);
    transpose_dest<true /* is_32bit */, true /* transpose_of_faces */>(0);
    PACK_RESULT();
    release_dst();

    // 2: binary add whose second operand is the tile already in DST, moved back
    // into SrcA — i.e. out = in + in.
    copy_tile_init(cb_in);
    acquire_dst();
    copy_tile(cb_in, 0, 0);
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in);
    binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_in, 0, 0);
    PACK_RESULT();
    release_dst();

    // --- end op sequence ---

    cb_pop_front(cb_in, 1);
}
