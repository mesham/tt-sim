// nocevbench arm C writer -- waits for the reader to finish, then pushes the
// same bytes into a PEER core's L1, one chunk per barrier.
//
// The mirror of kernels/dataflow/l1_reader.cpp, and the same one-substitution
// relationship to kernels/dataflow/writer.cpp: the destination is
// `get_noc_addr(peer_x, peer_y, addr)` rather than a DRAM bank.
//
// The two phases stay serialised by the semaphore, exactly as in arm A.
// Pipelining would make each RISC's span mostly a wait on the other one, and a
// stream whose interior is dominated by the peer measures the peer, not the
// NoC.

void kernel_main() {
    uint32_t peer_x = get_arg_val<uint32_t>(0);
    uint32_t peer_y = get_arg_val<uint32_t>(1);
    uint32_t remote_addr = get_arg_val<uint32_t>(2);
    uint32_t local_addr = get_arg_val<uint32_t>(3);
    uint32_t chunks = get_arg_val<uint32_t>(4);
    uint32_t bytes = get_arg_val<uint32_t>(5);
    uint32_t sem_id = get_arg_val<uint32_t>(6);

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, 1);

    for (uint32_t i = 0; i < chunks; i++) {
        uint64_t dst = get_noc_addr(peer_x, peer_y, remote_addr + i * bytes);
        noc_async_write(local_addr + i * bytes, dst, bytes);
        noc_async_write_barrier();
    }
}
