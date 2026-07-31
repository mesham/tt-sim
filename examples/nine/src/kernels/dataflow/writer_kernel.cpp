
// Tile B writer (BRISC, NOC 0). Waits for chunks to land in the local
// receive buffer, signalled by the cross-tile semaphore the sender
// increments, then forwards each chunk to DRAM. Monotonically waiting on
// (i+1) means we never need to reset the semaphore between iterations.
#define DATA_TYPE_BYTES 4

void kernel_main() {
    uint32_t dst_dram = get_arg_val<uint32_t>(0);
    uint32_t data_size = get_arg_val<uint32_t>(1);
    uint32_t chunk_size = get_arg_val<uint32_t>(2);
    uint32_t recv_buffer_addr = get_arg_val<uint32_t>(3);
    uint32_t sem_id = get_arg_val<uint32_t>(4);

    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_dram);

    uint32_t num_chunks = data_size / chunk_size;
    uint32_t bytes_per_chunk = DATA_TYPE_BYTES * chunk_size;

    volatile tt_l1_ptr uint32_t* sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    for (uint32_t i = 0; i < num_chunks; i++) {
        noc_semaphore_wait_min(sem_ptr, i + 1);

        noc_async_write(
            recv_buffer_addr + i * bytes_per_chunk,
            dst_dram_noc_addr + i * bytes_per_chunk,
            bytes_per_chunk);
        noc_async_write_barrier();
    }
}
