// nocevbench reader -- pulls `chunks` chunks of `bytes` from DRAM into L1
// scratch, barriering after each one, then hands over to the writer.
//
// One barrier per chunk is the point. The NoC event trace records every
// transaction at *issue* and records completion only at a barrier's END
// (dataflow_api.h:1751), so a kernel that issues a hundred reads and barriers
// once yields exactly one completion timestamp and one enormous latency
// sample. Barriering per chunk gives one clean issue->completion pair per
// transaction, which is what makes the per-class latency in
// tt_sim.perf.noc_events a comparison of transactions rather than of batches.
//
// The circular buffer is used as flat L1 scratch -- no push, no pop. The
// writer reads the same base address. Nothing is computed here; this program
// exists to move bytes and to be timed doing it.

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t chunks = get_arg_val<uint32_t>(1);
    uint32_t bytes = get_arg_val<uint32_t>(2);
    uint32_t sem_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_scratch = tt::CBIndex::c_0;
    uint32_t scratch = get_write_ptr(cb_scratch);
    uint64_t base = get_noc_addr_from_bank_id<true>(0, src_addr);

    for (uint32_t i = 0; i < chunks; i++) {
        noc_async_read(base + i * bytes, scratch + i * bytes, bytes);
        noc_async_read_barrier();
    }

    // Local store, not a NoC transaction: both RISCs are on this core and share
    // this L1. It is still recorded (as SEMAPHORE_SET) so the hand-over is
    // visible in the trace rather than hiding inside an unexplained gap.
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_set(sem, 1);
}
