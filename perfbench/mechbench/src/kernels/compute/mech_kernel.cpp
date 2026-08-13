// mechbench compute kernel -- the Tensix-bound half of the mechanism
// attribution leg.
//
// One tile-at-a-time pipeline: wait for a pair of input tiles, do one op into
// dst, pack, push. Which op is a compile-time argument, and that single switch
// is the whole design:
//
//   MODE_ELW (0)  add_tiles      -- one ELWADD per face. The Matrix unit is
//                                  cheap relative to the two unpacks that feed
//                                  it, so the *math* thread is the one that
//                                  waits: SrcA/SrcB VALID.
//   MODE_MM  (1)  matmul_tiles   -- a full 32x32 tile matmul, an order of
//                                  magnitude more Matrix-unit occupancy for
//                                  byte-identical unpacker work. Now the
//                                  *unpacker* is the one that waits, for the
//                                  Matrix unit to release the Src bank:
//                                  SrcA/SrcB CLEAR.
//
// Same kernel, same circular buffers, same tile count, same NoC traffic --
// only the direction of the Src-ownership stall reverses. That is what makes
// the pair a test of attribution rather than of totals: a decomposition that
// is wrong in compensating directions cannot get both arms right, and the two
// arms' spans are not required to agree for that to bite.
//
// The tile_regs_* handshake between the math thread (TRISC1) and the packer
// thread (TRISC2) runs on Tensix semaphores, so both arms also carry
// WAITING_FOR_{NONZERO,NONFULL}_SEM traffic on those two threads.

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

#define MODE_ELW 0
#define MODE_MM 1

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    const uint32_t tiles = get_arg_val<uint32_t>(0);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    if constexpr (mode == MODE_MM) {
        mm_init(cb_in0, cb_in1, cb_out);
    } else {
        binary_op_init_common(cb_in0, cb_in1, cb_out);
        add_tiles_init(cb_in0, cb_in1);
    }

    for (uint32_t i = 0; i < tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();
        if constexpr (mode == MODE_MM) {
            matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
        } else {
            add_tiles(cb_in0, cb_in1, 0, 0, 0);
        }
        tile_regs_commit();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
    }
}
