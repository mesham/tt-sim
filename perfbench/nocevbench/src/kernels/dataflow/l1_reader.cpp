// nocevbench arm C reader -- pulls `chunks` chunks of `bytes` out of a PEER
// core's L1 into this core's own L1, barriering after each one, then hands over
// to the writer.
//
// The only difference from kernels/dataflow/reader.cpp is where the bytes come
// from: `get_noc_addr(peer_x, peer_y, addr)` instead of
// `get_noc_addr_from_bank_id<true>(0, addr)`. That single substitution is the
// whole of arm C, and it is what takes the DRAM endpoint's service term off the
// critical path between the read and its barrier -- the term `unit_costs.yaml`
// already documents as over-charging a Blackhole DRAM write, and that
// `noc_dataset_sweep_test.py` pins as its one KNOWN_OVER_CHARGED row.
//
// Arm A's reader is left untouched rather than being generalised with a branch,
// because arm A is the control: it has to reproduce the 2026-08-17 session, and
// a kernel whose prologue grew a comparison is no longer that program.
//
// The peer runs no kernel at all. Its source region was written by the host
// before launch, at an address the L1 allocator guarantees is free on every
// bank, so nothing on the peer side can contend with or serialise against this
// core's transactions.

void kernel_main() {
    uint32_t peer_x = get_arg_val<uint32_t>(0);
    uint32_t peer_y = get_arg_val<uint32_t>(1);
    uint32_t remote_addr = get_arg_val<uint32_t>(2);
    uint32_t local_addr = get_arg_val<uint32_t>(3);
    uint32_t chunks = get_arg_val<uint32_t>(4);
    uint32_t bytes = get_arg_val<uint32_t>(5);
    uint32_t sem_id = get_arg_val<uint32_t>(6);

    for (uint32_t i = 0; i < chunks; i++) {
        uint64_t src = get_noc_addr(peer_x, peer_y, remote_addr + i * bytes);
        noc_async_read(src, local_addr + i * bytes, bytes);
        noc_async_read_barrier();
    }

    // Local store, not a NoC transaction: both RISCs are on this core and share
    // this L1. It is still recorded (as SEMAPHORE_SET) so the hand-over is
    // visible in the trace rather than hiding inside an unexplained gap.
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_set(sem, 1);
}
