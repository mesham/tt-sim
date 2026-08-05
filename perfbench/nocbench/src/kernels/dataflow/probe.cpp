// Stamp this core's own physical NoC coordinate into L1, and nothing else.
//
// `--dump-grid` needs two coordinate spaces per core and the host API only
// offers one. `worker_core_from_logical_core` returns the coordinate a kernel
// ADDRESSES, which on a harvested part is a dense renumbering of the surviving
// workers; the link arithmetic in `tt_sim.perf.noc_congestion_plan` is only
// valid in SoC-physical NoC 0. The two are equal on an unharvested Blackhole
// and differ by the harvested columns otherwise -- and nothing in the dump
// says which columns went, so it cannot be inferred. This kernel reads the
// number off the core's own NIU, where it is not a matter of inference.
//
// It is deliberately the smallest kernel that can answer: no NoC traffic, no
// barrier, no semaphore, so `--dump-grid` stays a read-only operation.

#include "../nocbench_layout.h"

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    const uint32_t node = NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID);
    out[NOCBENCH_R_NODE_X] = node & NOC_NODE_ID_MASK;
    out[NOCBENCH_R_NODE_Y] = (node >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    out[NOCBENCH_R_MAGIC] = NOCBENCH_MAGIC;  // last, so a partial write shows
}
