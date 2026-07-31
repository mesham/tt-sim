
#define DATA_TYPE_BYTES 4

// Reader kernel for example eight. Issues two concurrent reads against DRAM
// with distinct NoC transaction IDs (trid=1 and trid=2), then barriers on
// trid=2 first and trid=1 second — i.e. waits out-of-order vs. issue order.
// Each barrier polls its own ``NIU_MST_REQS_OUTSTANDING_ID_<n>`` counter, so
// the trid lifecycle in the sim must increment, route the response to the
// right return address, and decrement the right counter independently.
void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t buffer_1_addr = get_arg_val<uint32_t>(3);
    uint32_t buffer_2_addr = get_arg_val<uint32_t>(4);
    uint32_t data_size = get_arg_val<uint32_t>(5);

    uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src0_dram);
    uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, src1_dram);
    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_dram);

    uint32_t* buffer_1 = (uint32_t*)buffer_1_addr;
    uint32_t* buffer_2 = (uint32_t*)buffer_2_addr;

    uint32_t bytes_data_size = DATA_TYPE_BYTES * data_size;

    // Two concurrent reads with distinct trids and no intervening barrier.
    // The NOC_PACKET_TAG (trid) set here persists in the cmd buffer until the
    // next set; noc_async_read does not touch it.
    ncrisc_noc_set_transaction_id(noc_index, read_cmd_buf, 1);
    noc_async_read(src0_dram_noc_addr, buffer_1_addr, bytes_data_size);

    ncrisc_noc_set_transaction_id(noc_index, read_cmd_buf, 2);
    noc_async_read(src1_dram_noc_addr, buffer_2_addr, bytes_data_size);

    // Wait on trid=2 first, then trid=1 — opposite of issue order.
    noc_async_read_barrier_with_trid(2);
    noc_async_read_barrier_with_trid(1);

    for (uint32_t i = 0; i < data_size; i++) {
        buffer_1[i] = buffer_1[i] + buffer_2[i];
    }

    noc_async_write(buffer_1_addr, dst_dram_noc_addr, bytes_data_size);
    noc_async_write_barrier();
}
