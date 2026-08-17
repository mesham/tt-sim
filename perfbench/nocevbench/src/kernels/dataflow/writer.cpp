// nocevbench writer -- waits for the reader to finish, then pushes the same
// bytes back out to DRAM, one chunk per barrier.
//
// The two phases are deliberately serialised by a semaphore rather than
// pipelined through a circular buffer. Pipelining would make each RISC's span
// mostly a wait on the other one, and this leg is about NoC timing: a stream
// whose interior is dominated by the peer's progress measures the peer, not
// the NoC. Serialised, the reader's span is reads and the writer's is writes.
//
// The wait itself is recorded (SEMAPHORE_WAIT) and lands in the `other_wait`
// bucket, so it is attributed rather than smeared across the rest.

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t chunks = get_arg_val<uint32_t>(1);
    uint32_t bytes = get_arg_val<uint32_t>(2);
    uint32_t sem_id = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, 1);

    constexpr uint32_t cb_scratch = tt::CBIndex::c_0;
    uint32_t scratch = get_write_ptr(cb_scratch);
    uint64_t base = get_noc_addr_from_bank_id<true>(0, dst_addr);

    for (uint32_t i = 0; i < chunks; i++) {
        noc_async_write(scratch + i * bytes, base + i * bytes, bytes);
        noc_async_write_barrier();
    }
}
