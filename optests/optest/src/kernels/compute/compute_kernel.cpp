// Differential op-coverage kernel. Runs a *sequence* of Tensix/SFPU ops on one
// input tile, each writing its own output tile to cb_out (op i -> output tile
// i). The host dumps every output tile and optests/diff.sh compares the whole
// dump against ttsim, so a mismatch's element index maps back to which op.
//
// To cover another op: add a RUN_OP(...) block below and bump NUM_OPS in the
// host (optest.cpp) to match. All ops here are Int32 so they share one input
// tile; float / other-format ops want their own input tile + section.

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/bitwise.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/tile_move_copy.h"

constexpr auto cb_in = tt::CBIndex::c_0;
constexpr auto cb_out = tt::CBIndex::c_1;

// One op = copy the (held) input tile into DST[0], apply BODY, pack DST[0] to
// the next cb_out page. cb_in is waited on once and never popped, so every op
// sees the same input tile.
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

    // --- op sequence (keep NUM_OPS in optest.cpp in sync: NUM_OPS = 6) ---
    RUN_OP(/* 0: identity / copy — plumbing sanity */);

    bitwise_and_tile_init();
    RUN_OP(bitwise_and_tile<DataFormat::Int32>(0, 0x0f0f0f0f));

    bitwise_or_tile_init();
    RUN_OP(bitwise_or_tile<DataFormat::Int32>(0, 0xf0f0f0f0));

    bitwise_xor_tile_init();
    RUN_OP(bitwise_xor_tile<DataFormat::Int32>(0, 0xffff0000));

    binop_with_scalar_tile_init();
    RUN_OP(add_unary_tile_int32(0, 12345));

    binop_with_scalar_tile_init();
    RUN_OP(sub_unary_tile_int32(0, 1000));
    // --- end op sequence ---

    cb_pop_front(cb_in, 1);
}
