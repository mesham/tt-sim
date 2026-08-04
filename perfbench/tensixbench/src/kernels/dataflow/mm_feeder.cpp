// Phase B feeder. Hands the compute kernel its operand pages and stops.
//
// It never writes them. `matmul_fidelity.cpp` measures cycles, not values, and
// leaving the tiles uninitialised is what lets the whole of phase B run with no
// DRAM buffer, no NoC traffic and no golden -- so the fidelity difference it
// reports cannot be contaminated by dataflow.
//
// Pages are pushed TTBENCH_MM_BLOCK at a time, matching the compute kernel's
// blocking. Pushing them one at a time costs two circular-buffer round trips per
// matmul on this core, which is the same order as the whole compute-side inner
// loop and would make the FEEDER the limit -- the one confound phase B is
// explicitly guarding against. Blocked, the feeder's share is a few cycles per
// matmul. The block is a convenience, not a contract: the circular buffer is a
// page FIFO, so the producer's grouping and the consumer's need not agree.
//
// The compute kernel consumes base_iters * (1 + 2 + 3 + 4) pages of each
// operand, which is the arithmetic below. If that ever disagrees, the run
// deadlocks rather than reporting a wrong number -- both over-pushing (the
// feeder parks in cb_reserve_back forever) and under-pushing (compute parks in
// cb_wait_front) hang, so the total must be exact.

#include <cstdint>

// Quoted include, so it resolves relative to this file: tt-metal's JIT build
// compiles the kernel in place and puts only the kernel's own directory on the
// include path.
#include "../compute/bench_layout.h"

void kernel_main() {
    const uint32_t base_iters = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in0 = 0;  // tt::CBIndex::c_0
    constexpr uint32_t cb_in1 = 1;  // tt::CBIndex::c_1

    const uint32_t total = base_iters * (1 + 2 + 3 + 4);
    for (uint32_t pushed = 0; pushed < total;) {
        uint32_t blk = total - pushed;
        if (blk > TTBENCH_MM_BLOCK) {
            blk = TTBENCH_MM_BLOCK;
        }
        cb_reserve_back(cb_in0, blk);
        cb_push_back(cb_in0, blk);
        cb_reserve_back(cb_in1, blk);
        cb_push_back(cb_in1, blk);
        pushed += blk;
    }
}
