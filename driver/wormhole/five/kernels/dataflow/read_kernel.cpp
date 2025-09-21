#include "dataflow_api.h"

#define DATA_TYPE_BYTES 4

void kernel_main() {
    // Load in runtime arguments provided by the host
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t data_size = get_arg_val<uint32_t>(2);
    uint32_t chunk_size = get_arg_val<uint32_t>(3);

    // NoC coords (x,y) depending on DRAM location on-chip
    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src0_dram);
    uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src1_dram);

    // Index of circular buffer 0 (to communicate between this and the other data mover core)
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    // Calculate the number of chunks and the bytes per chunk
    uint32_t num_chunks = data_size / chunk_size;
    uint32_t bytes_per_chunk = DATA_TYPE_BYTES*chunk_size;

    // Proceed through each chunk, each of these is known as a tile
    for (uint32_t i=0;i<num_chunks;i++) {
        // Reserve a single page in the CBs
        cb_reserve_back(cb_id_in0, 1); 
        cb_reserve_back(cb_id_in1, 1); 

        // Now we have the pages grab the write pointers to them
        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // Read data from DRAM into L1 circular buffers 0 and 1
        noc_async_read(src0_dram_noc_addr+(i*bytes_per_chunk), l1_write_addr_in0, bytes_per_chunk);
        noc_async_read(src1_dram_noc_addr+(i*bytes_per_chunk), l1_write_addr_in1, bytes_per_chunk);
        noc_async_read_barrier();

        // Push the pages to make them available to the consumer of the CBs
        cb_push_back(cb_id_in0, 1); 
        cb_push_back(cb_id_in1, 1); 
    }   
}
