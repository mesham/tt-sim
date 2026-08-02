// FP32 SFPU-math op-coverage kernel. Runs a sequence of SFPU math ops on one
// Float32 input tile, each writing its own output tile to cb_out. The host dumps
// every tile and optests/diff.sh compares against ttsim, so the SFPU
// approximations (e.g. recip's SFPARECIP LUT) must match the reference exactly.
//
// To cover another op: add a RUN_OP(...) block and bump NUM_OPS in sfpumath.cpp.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/tile_move_copy.h"

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_out = tt::CBIndex::c_1;

#define RUN_OP(BODY)                \
    do {                            \
        tile_regs_acquire();        \
        copy_tile(cb_in, 0, 0);     \
        BODY;                       \
        tile_regs_commit();         \
        cb_reserve_back(cb_out, 1); \
        tile_regs_wait();           \
        pack_tile(0, cb_out);       \
        tile_regs_release();        \
        cb_push_back(cb_out, 1);    \
    } while (0)

void kernel_main() {
    init_sfpu(cb_in, cb_out);
    cb_wait_front(cb_in, 1);

    // --- op sequence (keep NUM_OPS in sfpumath.cpp in sync) ---
    // Op 0 is frozen: driver/blackhole/server/sfpumath_replay_test.py checks the
    // first output tile against a recorded ttsim dump, so keep recip first and
    // append new ops after it.
    recip_tile_init();
    RUN_OP(recip_tile(0));  // 0: reciprocal -> exercises Blackhole SFPARECIP

    tanh_tile_init();
    RUN_OP(tanh_tile(0));  // 1: tanh -> piecewise LUT via SFPLUT

    sigmoid_tile_init();
    RUN_OP(sigmoid_tile(0));  // 2: sigmoid -> 6-entry FP32 LUT via SFPLUTFP32
    // --- end op sequence ---

    cb_pop_front(cb_in, 1);
}
