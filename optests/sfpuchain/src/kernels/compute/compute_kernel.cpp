// Differential op-coverage kernel for the SFPU softplus chain that upstream's
// `sfpu_eltwise_chain` programming example runs (exp -> add_binary -> log).
//
// Each op re-reads the SAME resident bfloat16 input tile and packs its own
// output tile, so a mismatch's tile index says which link of the chain broke:
//
//   0: copy only          (baseline: unpack -> Dst -> pack, no SFPU math)
//   1: exp_tile
//   2: log_tile
//   3: add_binary_tile(x, ones)   (SFPU binary add, not the FPU ELWADD)
//   4: exp; add_binary(+1); log   (upstream's chain, verbatim)
//
// To cover another op: add a block below and bump NUM_OPS in sfpuchain.cpp.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_binary_sfpu.h"

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_ones = tt::CBIndex::c_1;
constexpr auto cb_out = tt::CBIndex::c_16;

// One op = acquire Dst, copy the input tile into Dst[0] (and the ones tile into
// Dst[1], which the binary ops need), run BODY, pack Dst[0] to the next cb_out
// page.
#define RUN_OP(BODY)                \
    do {                            \
        tile_regs_acquire();        \
        copy_tile(cb_in, 0, 0);     \
        copy_tile(cb_ones, 0, 1);   \
        BODY;                       \
        tile_regs_commit();         \
        tile_regs_wait();           \
        cb_reserve_back(cb_out, 1); \
        pack_tile(0, cb_out);       \
        cb_push_back(cb_out, 1);    \
        tile_regs_release();        \
    } while (0)

void kernel_main() {
    init_sfpu(cb_in, cb_out);

    // Both operands stay resident (never popped), so every op sees the same
    // input.
    cb_wait_front(cb_in, 1);
    cb_wait_front(cb_ones, 1);

    // --- op sequence (keep NUM_OPS in sfpuchain.cpp in sync: NUM_OPS = 5) ---
    RUN_OP();  // 0: copy only

    exp_tile_init();
    RUN_OP(exp_tile(0));  // 1: exp(x)

    log_tile_init();
    RUN_OP(log_tile(0));  // 2: log(x)

    add_binary_tile_init();
    RUN_OP(add_binary_tile(0, 1, 0));  // 3: x + 1

    RUN_OP({                       // 4: log(exp(x) + 1) -- upstream's chain
        exp_tile_init();           //
        exp_tile(0);               //
        add_binary_tile_init();    //
        add_binary_tile(0, 1, 0);  //
        log_tile_init();           //
        log_tile(0);               //
    });
    // --- end op sequence ---

    cb_pop_front(cb_in, 1);
    cb_pop_front(cb_ones, 1);
}
