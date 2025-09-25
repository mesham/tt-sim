#include "dataflow_api.h"

#define DATA_TYPE_BYTES 4

void kernel_main() {
    // Load in runtime arguments provided by the host
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t buffer_1_addr = get_arg_val<uint32_t>(2);
    uint32_t buffer_2_addr = get_arg_val<uint32_t>(3);
    uint32_t data_size = get_arg_val<uint32_t>(4);

    // NoC coords (x,y) depending on DRAM location on-chip
    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src0_dram);
    uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src1_dram);

    // Index of circular buffer 0 (to communicate between this and the other data mover core)
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    // Internal L1 buffers to receive data into
    uint32_t * buffer_1 = (uint32_t*) buffer_1_addr;
    uint32_t * buffer_2 = (uint32_t*) buffer_2_addr;

    // Bytes that are read from DDR for the data
    uint32_t bytes_data_size=DATA_TYPE_BYTES * data_size;

    // Read data from DRAM into L1 buffers
    noc_async_read(src0_dram_noc_addr, buffer_1_addr, bytes_data_size);
    noc_async_read(src1_dram_noc_addr, buffer_2_addr, bytes_data_size);
    noc_async_read_barrier();

    // Reserve a single page in the CB
    cb_reserve_back(cb_id_in0, 1); 
    // Now we have the page grab the write pointer to this
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t* result_data = (uint32_t*)l1_write_addr_in0;

    for (uint32_t i=0;i<data_size;i++) {
        result_data[i] = buffer_1[i] + buffer_2[i];
    }   

    // Push the page to make it available to consumer of the CB
    cb_push_back(cb_id_in0, 1); 
}
