// packspill compute: reserve exactly ONE page of the output CB and pack
// exactly ONE tile into it. The CB's page is a sixteenth of a tile, so if
// `pack_tile` really writes a whole tile (as pack.h documents and as the LLK's
// packer datum counter implies) this single call covers all sixteen pages.

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_out = tt::CBIndex::c_1;

void kernel_main() {
    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);

    cb_pop_front(cb_in, 1);
}
