
// Tile A sender (NCRISC, NOC 1). Bridges Tile A's compute-output CB to a
// Tile B L1 receive buffer, paced by a semaphore. For each chunk:
//   1. cb_wait_front on the compute output (CB2),
//   2. noc_async_write the chunk to the matching offset in Tile B's
//      receive buffer,
//   3. noc_semaphore_inc Tile B's signal semaphore so the writer there
//      knows another chunk is in flight,
//   4. cb_pop_front to release the local slot.
//
// The chunk_size+i indexing means consumer and producer can both run
// pipelined: Tile A is free to compute chunk i+1 while Tile B is still
// draining chunk i to DRAM.
#define DATA_TYPE_BYTES 4

void kernel_main() {
    uint32_t data_size = get_arg_val<uint32_t>(0);
    uint32_t chunk_size = get_arg_val<uint32_t>(1);
    uint32_t tile_b_x = get_arg_val<uint32_t>(2);
    uint32_t tile_b_y = get_arg_val<uint32_t>(3);
    uint32_t recv_buffer_addr = get_arg_val<uint32_t>(4);
    uint32_t sem_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CBIndex::c_2;

    uint32_t num_chunks = data_size / chunk_size;
    uint32_t bytes_per_chunk = DATA_TYPE_BYTES * chunk_size;

    uint32_t sem_l1_addr = get_semaphore(sem_id);
    uint64_t sem_noc_addr = get_noc_addr(tile_b_x, tile_b_y, sem_l1_addr);

    for (uint32_t i = 0; i < num_chunks; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t local_addr = get_read_ptr(cb_out);

        uint64_t dst_noc_addr = get_noc_addr(
            tile_b_x, tile_b_y, recv_buffer_addr + i * bytes_per_chunk);
        noc_async_write(local_addr, dst_noc_addr, bytes_per_chunk);
        noc_async_write_barrier();

        noc_semaphore_inc(sem_noc_addr, 1);

        cb_pop_front(cb_out, 1);
    }
}
