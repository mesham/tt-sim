#include "dataflow_api.h"

// Output is int32
#define DATA_TYPE_BYTES 4

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t data_size = get_arg_val<uint32_t>(1);
    uint32_t chunk_size = get_arg_val<uint32_t>(2);

    // Get destination DRAM NoC address for writing
    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_addr);

    // Circular buffer index used to communicate data from compute core
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_2;

    // Calculate the number of chunks and the bytes per chunk
    uint32_t num_chunks = data_size / chunk_size;
    uint32_t bytes_per_chunk = DATA_TYPE_BYTES*chunk_size;

    // Proceed through each chunk, each of these is known as a tile
    for (uint32_t i=0;i<num_chunks;i++) {
        // Wait for a page to be made available in the CB
        cb_wait_front(cb_id_out0, 1); 
        // Now we have the page grab the read address of this
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        // Write the recieved data to DDR
        noc_async_write(l1_read_addr, dst_noc_addr+(i*bytes_per_chunk), bytes_per_chunk);
        noc_async_write_barrier();
        // Pop the page to make it available for writing again
        cb_pop_front(cb_id_out0, 1); 
    }   
}
