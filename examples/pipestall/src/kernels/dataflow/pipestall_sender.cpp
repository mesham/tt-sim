
// pipestall Tile A sender (NCRISC, NOC 1). Forwards each compute-output chunk
// to Tile B's L1 receive buffer, but only once Tile B has released a credit:
// before sending chunk i it waits for credit_sem >= i + 1 - credits, so at most
// `credits` chunks are ever in flight.
//
// That reverse dependency is the whole point of the example. With credits = 1
// the sender cannot pop CB2 until the *remote* core has drained the previous
// chunk, so the producer's Tensix pipeline backs up behind a core it does not
// control.
#define DATA_TYPE_BYTES 4

void kernel_main() {
    uint32_t data_size = get_arg_val<uint32_t>(0);
    uint32_t chunk_size = get_arg_val<uint32_t>(1);
    uint32_t tile_b_x = get_arg_val<uint32_t>(2);
    uint32_t tile_b_y = get_arg_val<uint32_t>(3);
    uint32_t recv_buffer_addr = get_arg_val<uint32_t>(4);
    uint32_t data_sem_id = get_arg_val<uint32_t>(5);
    uint32_t credit_sem_id = get_arg_val<uint32_t>(6);
    uint32_t credits = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_out = tt::CBIndex::c_2;

    uint32_t num_chunks = data_size / chunk_size;
    uint32_t bytes_per_chunk = DATA_TYPE_BYTES * chunk_size;

    uint64_t data_sem_noc_addr = get_noc_addr(tile_b_x, tile_b_y, get_semaphore(data_sem_id));
    volatile tt_l1_ptr uint32_t* credit_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_sem_id));

    for (uint32_t i = 0; i < num_chunks; i++) {
        // Credit first: block before touching CB2 so the back-pressure reaches
        // the compute pipeline rather than being absorbed by a local buffer.
        if (i + 1 > credits) {
            noc_semaphore_wait_min(credit_ptr, i + 1 - credits);
        }

        cb_wait_front(cb_out, 1);
        uint32_t local_addr = get_read_ptr(cb_out);

        uint64_t dst_noc_addr =
            get_noc_addr(tile_b_x, tile_b_y, recv_buffer_addr + i * bytes_per_chunk);
        noc_async_write(local_addr, dst_noc_addr, bytes_per_chunk);
        noc_async_write_barrier();

        noc_semaphore_inc(data_sem_noc_addr, 1);

        cb_pop_front(cb_out, 1);
    }
}
