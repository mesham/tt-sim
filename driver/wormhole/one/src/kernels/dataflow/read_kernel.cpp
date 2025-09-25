#include "dataflow_api.h"

#define DATA_TYPE_BYTES 4

void kernel_main() {
    // Load in runtime arguments provided by the host
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t buffer_1_addr = get_arg_val<uint32_t>(3);
    uint32_t buffer_2_addr = get_arg_val<uint32_t>(4);
    uint32_t data_size = get_arg_val<uint32_t>(5);

    // NoC coords (x,y) depending on DRAM location on-chip
    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src0_dram);
    uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src1_dram);
    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_dram);

    uint32_t * buffer_1 = (uint32_t*) buffer_1_addr;
    uint32_t * buffer_2 = (uint32_t*) buffer_2_addr;

    uint32_t bytes_data_size = DATA_TYPE_BYTES*data_size;

    // Read data from DRAM into L1 buffers
    noc_async_read(src0_dram_noc_addr, buffer_1_addr, bytes_data_size);
    noc_async_read_barrier();
    noc_async_read(src1_dram_noc_addr, buffer_2_addr, bytes_data_size);
    noc_async_read_barrier();

    for (uint32_t i=0;i<data_size;i++) {
        buffer_1[i] = buffer_1[i] + buffer_2[i];
    }   

    // Write data from L1 circulr buffer CB 0 to DRAM
    noc_async_write(buffer_1_addr, dst_dram_noc_addr, bytes_data_size);
    noc_async_write_barrier();
}
