#include "dataflow_api.h"

#define DATA_TYPE_BYTES 4

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t data_size = get_arg_val<uint32_t>(1);

    // Get destination DRAM NoC address for writing
    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_addr);

    // Circular buffer index used to communicate data from the other data mover core
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_0;

    // Bytes that are written to DDR for the data
    uint32_t bytes_data_size=DATA_TYPE_BYTES * data_size;

    // Wait for a page to be made available in the CB
    cb_wait_front(cb_id_out0, 1); 
    // Now we have the page grab the read address of this
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
    // Write the recieved data to DDR
    noc_async_write(l1_read_addr, dst_noc_addr, bytes_data_size);
    noc_async_write_barrier();
    // Pop the page to make it available for writing again
    cb_pop_front(cb_id_out0, 1); 
}
