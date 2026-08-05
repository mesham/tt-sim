
// pipestall Tile B writer (BRISC, NOC 0). The *consumer*, and the knob:
//
//   1. wait for the sender's data semaphore to reach i + 1,
//   2. burn `delay` iterations — the tunable part; this is what a slower
//      downstream stage (more work per tile, a deeper pipeline, a busier NoC)
//      looks like from the producer's side,
//   3. forward the chunk from the local receive buffer to DRAM,
//   4. release one credit back to Tile A so the producer may send the next.
//
// Step 4 is what turns `delay` into blocked cycles inside the producer's
// Tensix backend. The spin is on a volatile so the compiler cannot fold it.
#define DATA_TYPE_BYTES 4

void kernel_main() {
    uint32_t dst_dram = get_arg_val<uint32_t>(0);
    uint32_t data_size = get_arg_val<uint32_t>(1);
    uint32_t chunk_size = get_arg_val<uint32_t>(2);
    uint32_t recv_buffer_addr = get_arg_val<uint32_t>(3);
    uint32_t data_sem_id = get_arg_val<uint32_t>(4);
    uint32_t credit_sem_id = get_arg_val<uint32_t>(5);
    uint32_t delay = get_arg_val<uint32_t>(6);
    uint32_t tile_a_x = get_arg_val<uint32_t>(7);
    uint32_t tile_a_y = get_arg_val<uint32_t>(8);

    uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_dram);

    uint32_t num_chunks = data_size / chunk_size;
    uint32_t bytes_per_chunk = DATA_TYPE_BYTES * chunk_size;

    volatile tt_l1_ptr uint32_t* data_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_sem_id));
    uint64_t credit_sem_noc_addr = get_noc_addr(tile_a_x, tile_a_y, get_semaphore(credit_sem_id));

    volatile uint32_t sink = 0;
    for (uint32_t i = 0; i < num_chunks; i++) {
        noc_semaphore_wait_min(data_ptr, i + 1);

        for (uint32_t k = 0; k < delay; k++) {
            sink = sink + k;
        }

        noc_async_write(
            recv_buffer_addr + i * bytes_per_chunk,
            dst_dram_noc_addr + i * bytes_per_chunk,
            bytes_per_chunk);
        noc_async_write_barrier();

        noc_semaphore_inc(credit_sem_noc_addr, 1);
    }
}
