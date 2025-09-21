include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/add_int_sfpu.h"

namespace NAMESPACE {
void MAIN {
    uint32_t data_size = get_arg_val<uint32_t>(0);
    uint32_t chunk_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    uint32_t num_chunks = data_size / chunk_size;

    init_sfpu(cb_in0, cb_out0);
    add_int_tile_init();

    for (uint32_t i=0;i<num_chunks;i++) {
        // Wait for a block of tiles in each of input CBs
        cb_wait_front(cb_in0, 1); 
        cb_wait_front(cb_in1, 1); 

        // Aquire dst registers for compute core
        tile_regs_acquire();

        // Copy the tile from zero page in cb_in0 into segment
        // 0 of dst registers
        copy_tile(cb_in0, 0, 0); 
        // Copy the tile from zero page in cb_in1 into segment
        // 1 of dst registers
        copy_tile(cb_in1, 0, 1); 

        // Instruct the SFPU to add the tiles in segments
        // zero and one in the dst register, the first 
        // (0) dst register index is overwritten with results
        add_int32_tile(0, 1); 

        // Commit the dst registers so they can be consumed
        tile_regs_commit();

        // Pop pages in the input CBs so these can be reused
        cb_pop_front(cb_in0, 1); 
        cb_pop_front(cb_in1, 1); 

        // Reserve a page in the output CB
        cb_reserve_back(cb_out0, 1); 

        // Wait for dst registers for packer RV core
        tile_regs_wait();
        // Pack from segement 0 in the dst register to the output CB
        pack_tile(0, cb_out0);
        // Release dst registers
        tile_regs_release();

        // Make output tile available to consumer
        cb_push_back(cb_out0, 1); 
    }   
}
}
