// Differential op-coverage kernel for the element-wise binary FPU ops. Runs
// mul_tiles, add_tiles and sub_tiles on the same two resident input tiles,
// packing one output tile per op (op i -> output tile i), so a mismatch's
// element index maps straight back to which op disagreed.
//
// The host launches this kernel once per (CB format, math fidelity) pair, so
// the multiply is covered at every fidelity level: ELWMUL forms its product
// from mantissa *slices* -- SrcA's 5 then 5 bits, SrcB's 7 then 4 -- one
// slice pair per fidelity phase, and accumulates each phase's partial product
// into Dst. Whether the phases are issued is the fidelity level; whether each
// one contributes what the hardware's does is what this diff checks.
//
// To cover another op: add a RUN_OP(...) block below and bump NUM_OPS in the
// host (elwmul.cpp) to match.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"

constexpr auto cb_a = tt::CBIndex::c_0;
constexpr auto cb_b = tt::CBIndex::c_1;
constexpr auto cb_out = tt::CBIndex::c_16;

// One op = compute (a op b) into DST[0], pack DST[0] to the next cb_out page.
#define RUN_OP(INIT, OP)             \
    do {                             \
        INIT;                        \
        tile_regs_acquire();         \
        OP;                          \
        tile_regs_commit();          \
        cb_reserve_back(cb_out, 1);  \
        tile_regs_wait();            \
        pack_tile(0, cb_out);        \
        tile_regs_release();         \
        cb_push_back(cb_out, 1);     \
    } while (0)

void kernel_main() {
    compute_kernel_hw_startup(cb_a, cb_b, cb_out);

    // Both operands stay resident (never popped) until the end, so every op
    // sees the same pair of input tiles.
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);

    // --- op sequence (keep NUM_OPS in elwmul.cpp in sync: NUM_OPS = 3) ---
    RUN_OP(mul_tiles_init(cb_a, cb_b), mul_tiles(cb_a, cb_b, 0, 0, 0));         // 0: ELWMUL
    RUN_OP(add_tiles_init(cb_a, cb_b, false), add_tiles(cb_a, cb_b, 0, 0, 0));  // 1: ELWADD
    RUN_OP(sub_tiles_init(cb_a, cb_b, false), sub_tiles(cb_a, cb_b, 0, 0, 0));  // 2: ELWSUB
    // --- end op sequence ---

    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
