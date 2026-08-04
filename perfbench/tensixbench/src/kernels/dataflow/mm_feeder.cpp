// Phase B feeder. Hands the compute kernel its operand pages and stops.
//
// It never writes them. `matmul_fidelity.cpp` measures cycles, not values, and
// leaving the tiles uninitialised is what lets the whole of phase B run with no
// DRAM buffer, no NoC traffic and no golden -- so the fidelity difference it
// reports cannot be contaminated by dataflow.
//
// The compute kernel consumes base_iters * (1 + 2 + 3 + 4) pages of each
// operand, which is the arithmetic below. If that ever disagrees, the run
// deadlocks rather than reporting a wrong number.

#include <cstdint>

void kernel_main() {
    const uint32_t base_iters = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in0 = 0;  // tt::CBIndex::c_0
    constexpr uint32_t cb_in1 = 1;  // tt::CBIndex::c_1

    const uint32_t total = base_iters * (1 + 2 + 3 + 4);
    for (uint32_t i = 0; i < total; i++) {
        cb_reserve_back(cb_in0, 1);
        cb_push_back(cb_in0, 1);
        cb_reserve_back(cb_in1, 1);
        cb_push_back(cb_in1, 1);
    }
}
