// `pack_untilize_dest_init` issued *after* the math instead of before it.
//
// Reported by the hedgehope compiler team (2026-08-20). Moving that one call
// from before `matmul_init` to after `tile_regs_wait()` -- changing nothing
// else, and changing only the PACK thread's own instruction order, since the
// call compiles to nothing on UNPACK/MATH -- takes their single-core K=2 GEMM
// from `errors=0 of 1024` to `errors=320 of 1024` on tt-sim. Both forms are
// correct on an n300 card, and both are correct on ttsim.
//
// Fixed 2026-08-20 in the Wait Gate -- a `STALLWAIT` no longer overwrites an
// unsatisfied latched `SEMWAIT`; see the host program's header. Both forms
// pass now, and this kernel is the regression that keeps them passing.
//
// The co-factor is K >= 2: with a single `matmul_tiles` into DEST the late init
// is harmless; two or more accumulating into the same DEST tile are needed.
//
//   INIT_LATE=1 (default) -- `pack_untilize_dest_init` after `tile_regs_wait`,
//       which is the form that was the defect and what `optests/diff.sh`
//       runs.
//   INIT_LATE=0 -- the same program with the call before `matmul_init`. The
//       control: it must stay green, or the test is measuring something else.
//   KT -- the number of `matmul_tiles` accumulated into DEST tile 0. KT=1 is
//       the co-factor control.
//
// A[k] = SCALE[k] * identity and B[k] is one ramp tile, so the output is
// exactly (SCALE[0] + ... ) * B in row-major order, with 1024 distinct values
// -- an element that came from the wrong DEST row, the wrong half or the wrong
// read width cannot hide.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_a_rm = tt::CBIndex::c_0;   // A, row-major from DRAM
constexpr auto cb_b_rm = tt::CBIndex::c_1;   // B, row-major from DRAM
constexpr auto cb_a = tt::CBIndex::c_24;     // A, tiled on device
constexpr auto cb_b = tt::CBIndex::c_25;     // B, tiled on device
constexpr auto cb_out = tt::CBIndex::c_16;   // untilized output, row-major

#ifndef INIT_LATE
#define INIT_LATE 1
#endif
#ifndef KT
#define KT 2
#endif
#ifndef NUM_OUT
#define NUM_OUT 1
#endif
// Tilize the operands on device, as the reported kernel does, instead of taking
// them already tiled from DRAM. The tilize is a co-factor: without it the late
// init is harmless.
#ifndef ON_DEVICE_TILIZE
#define ON_DEVICE_TILIZE 1
#endif

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a_rm, cb_b_rm, cb_out);

#if ON_DEVICE_TILIZE
    // Mirrors the reported kernel: A a tile at a time, B as one block.
    for (uint32_t k = 0; k < KT; k++) {
        cb_wait_front(cb_a_rm, 1);
        cb_reserve_back(cb_a, 1);
        tilize_init(cb_a_rm, 1, cb_a);
        tilize_block(cb_a_rm, 1, cb_a);
        tilize_uninit(cb_a_rm, cb_a);
        cb_push_back(cb_a, 1);
        cb_pop_front(cb_a_rm, 1);
    }
    cb_wait_front(cb_b_rm, KT);
    cb_reserve_back(cb_b, KT);
    tilize_init(cb_b_rm, KT, cb_b);
    tilize_block(cb_b_rm, KT, cb_b);
    tilize_uninit(cb_b_rm, cb_b);
    cb_push_back(cb_b, KT);
    cb_pop_front(cb_b_rm, KT);
#endif

    cb_wait_front(cb_a, KT);
    cb_wait_front(cb_b, KT);

    for (uint32_t o = 0; o < NUM_OUT; o++) {
        cb_reserve_back(cb_out, 1);
#if !INIT_LATE
        pack_untilize_dest_init<1, 1>(cb_out);
#endif
        matmul_init(cb_a, cb_b, 0);
        tile_regs_acquire();
        for (uint32_t k = 0; k < KT; k++) {
            matmul_tiles(cb_a, cb_b, k, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
#if INIT_LATE
        pack_untilize_dest_init<1, 1>(cb_out);
#endif
        pack_untilize_dest<1, 1>(cb_out, 1, 0);
        tile_regs_release();
        pack_untilize_uninit(cb_out);
        cb_push_back(cb_out, 1);
    }

    cb_pop_front(cb_a, KT);
    cb_pop_front(cb_b, KT);
}
