#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

// Tile A compute (TRISC). Pops chunked inputs from CB0/CB1, runs eltwise
// add via the FPU, packs the int32 result into CB2 for the sender. Same
// shape as example four's compute_kernel.
namespace NAMESPACE {
void MAIN {
    uint32_t data_size = get_arg_val<uint32_t>(0);
    uint32_t chunk_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    uint32_t num_chunks = data_size / chunk_size;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    for (uint32_t i = 0; i < num_chunks; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);

        tile_regs_acquire();
        add_tiles(cb_in0, cb_in1, 0, 0, 0);
        tile_regs_commit();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        cb_reserve_back(cb_out0, 1);

        tile_regs_wait();
        pack_tile(0, cb_out0);
        tile_regs_release();

        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
