#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/add_int_sfpu.h"

namespace NAMESPACE {
void MAIN {
    uint32_t data_size = get_arg_val<uint32_t>(0);
    uint32_t chunk_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_1;

    uint32_t num_chunks = data_size / chunk_size;

    init_sfpu(cb_in0, cb_out0);
    add_int_tile_init();

    for (uint32_t i=0;i<num_chunks;i++) {
        // Wait for a block of tiles in each of input CBs
        cb_wait_front(cb_in0, 1); 

        // Aquire dst registers for compute core
        tile_regs_acquire();

        // Copy the tile from zero page in cb_in0 into segment
        // 2 of dst registers
        copy_tile(cb_in0, 0, 2); 

        // Commit the dst registers so they can be consumed
        tile_regs_commit();

        // Pop pages in the input CBs so these can be reused
        cb_pop_front(cb_in0, 1); 

        // Reserve a page in the output CB
        cb_reserve_back(cb_out0, 1); 

        // Wait for dst registers for packer RV core
        tile_regs_wait();
        // Pack from segement 2 in the dst register to the output CB
        pack_tile(2, cb_out0);
        // Release dst registers
        tile_regs_release();

        // Make output tile available to consumer
        cb_push_back(cb_out0, 1); 
    }   
}
}

