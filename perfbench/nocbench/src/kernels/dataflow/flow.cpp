// One NoC flow, timed on the issuing core's own wall clock.
//
// Every master core in a run gets one copy of this kernel. The kernel
//
//   1. rendezvouses with every other master (each master atomically increments
//      a semaphore that lives at the same L1 address on every participating
//      core, then waits for the count to reach num_peers), so that the flows
//      really are concurrent rather than merely both present in one program;
//   2. reads the free-running wall clock;
//   3. issues num_tx transactions of tx_bytes to/from its own subordinate,
//      with a single barrier at the end so the transactions pipeline;
//   4. reads the wall clock again and stamps the result -- together with its
//      OWN NoC coordinates, which is what makes the measurement identifiable.
//
// It deliberately does not use DeviceZoneScopedN: that needs a profiler build,
// writes its own CSV keyed by nothing this harness controls, and is not
// available against the simulator. The wall clock is the same register the
// profiler reads (kernel_profiler.hpp), so the units are identical.

#include "../nocbench_layout.h"

namespace {

inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(NOCBENCH_WALL_CLOCK_L);
    return p[0];
}

}  // namespace

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(NOCBENCH_A_RESULTS);
    const uint32_t data_addr = get_arg_val<uint32_t>(NOCBENCH_A_DATA);
    // The host passes a semaphore *id*; `get_semaphore` turns it into the L1
    // address the runtime placed it at, which is the same on every core in the
    // CoreRangeSet it was created over. Using the id as an address instead
    // silently pokes L1 offset 0 and the rendezvous never blocks -- which is
    // exactly what it did on the first simulator run, and the overlap check in
    // `noc_congestion_sweep` is what caught it.
    const uint32_t sem_addr = (uint32_t)get_semaphore(get_arg_val<uint32_t>(NOCBENCH_A_SEM));
    const uint32_t num_peers = get_arg_val<uint32_t>(NOCBENCH_A_NUM_PEERS);
    const uint32_t flow = get_arg_val<uint32_t>(NOCBENCH_A_FLOW);
    const uint32_t sub_x = get_arg_val<uint32_t>(NOCBENCH_A_SUB_X);
    const uint32_t sub_y = get_arg_val<uint32_t>(NOCBENCH_A_SUB_Y);
    const uint32_t num_tx = get_arg_val<uint32_t>(NOCBENCH_A_NUM_TX);
    const uint32_t tx_bytes = get_arg_val<uint32_t>(NOCBENCH_A_TX_BYTES);
    const uint32_t direction = get_arg_val<uint32_t>(NOCBENCH_A_DIRECTION);
    const uint32_t vc = get_arg_val<uint32_t>(NOCBENCH_A_VC);

    // The write source and the read destination are disjoint halves of the
    // data buffer, so a bidirectional flow is not racing itself.
    const uint32_t write_src = data_addr;
    const uint32_t read_dst = data_addr + tx_bytes;

    const uint64_t sub_write_addr = get_noc_addr(sub_x, sub_y, data_addr);
    const uint64_t sub_read_addr = get_noc_addr(sub_x, sub_y, data_addr + tx_bytes);

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    // --- 1. rendezvous -----------------------------------------------------
    // Each master pokes every other master's semaphore, then waits for all of
    // them. The pokes are single-flit atomics issued before the timed region;
    // what they cost is not measured and does not need to be.
    for (uint32_t i = 0; i < num_peers; i++) {
        const uint32_t px = get_arg_val<uint32_t>(NOCBENCH_A_PEERS + 2 * i);
        const uint32_t py = get_arg_val<uint32_t>(NOCBENCH_A_PEERS + 2 * i + 1);
        noc_semaphore_inc(get_noc_addr(px, py, sem_addr), 1);
    }
    const uint32_t rend_start = wall_clock_lo();
    if (num_peers > 0) {
        noc_semaphore_wait_min(sem, num_peers);
    }
    const uint32_t rend_end = wall_clock_lo();

    // --- 2..3. the timed region -------------------------------------------
    const uint32_t t0 = wall_clock_lo();
    for (uint32_t i = 0; i < num_tx; i++) {
        if (direction != NOCBENCH_DIR_READ) {
            noc_async_write(write_src, sub_write_addr, tx_bytes, noc_index, vc);
        }
        if (direction != NOCBENCH_DIR_WRITE) {
            noc_async_read(sub_read_addr, read_dst, tx_bytes);
        }
    }
    if (direction != NOCBENCH_DIR_READ) {
        noc_async_write_barrier();
    }
    if (direction != NOCBENCH_DIR_WRITE) {
        noc_async_read_barrier();
    }
    const uint32_t t1 = wall_clock_lo();

    // --- 4. the stamp ------------------------------------------------------
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    out[NOCBENCH_R_FLOW] = flow;
    out[NOCBENCH_R_T0] = t0;
    out[NOCBENCH_R_T1] = t1;
    out[NOCBENCH_R_CYCLES] = t1 - t0;  // unsigned wrap is the right answer
    out[NOCBENCH_R_NOC_X] = my_x[noc_index];
    out[NOCBENCH_R_NOC_Y] = my_y[noc_index];
    const uint32_t node = NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID);
    out[NOCBENCH_R_NODE_X] = node & NOC_NODE_ID_MASK;
    out[NOCBENCH_R_NODE_Y] = (node >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    out[NOCBENCH_R_NUM_TX] = num_tx;
    out[NOCBENCH_R_TX_BYTES] = tx_bytes;
    out[NOCBENCH_R_DIRECTION] = direction;
    out[NOCBENCH_R_VC] = vc;
    out[NOCBENCH_R_NOC] = noc_index;
    out[NOCBENCH_R_REND_CYCLES] = rend_end - rend_start;
    out[NOCBENCH_R_MAGIC] = NOCBENCH_MAGIC;  // last, so a partial write shows
}
