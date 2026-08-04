// On-device `tilize_block` followed by `matmul_tiles`, with and without the
// `matmul_init` that has to be re-issued in between.
//
// The unpacker has one MOP program and one set of config registers, and every
// LLK op's `*_init` overwrites them. `tilize_init` programs the *tilize* MOP
// (SrcA UNPACR with Z-increment plus an UNPACR_NOP that zeroes SrcB and sets
// its dvalid) and `tilize_uninit` does not put the matmul MOP back -- it only
// restores the tile descriptor and clears `Tileize_mode`. So a `matmul_tiles`
// issued after a tilize, with no intervening `matmul_init`, runs the *tilize*
// MOP: SrcB is filled with zeroes by an UNPACR_NOP instead of being unpacked
// from operand A's L1 address.
//
//   REISSUE_MATMUL_INIT=1 (default) -- the correct sequence, as tt-metal's own
//       tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block.cpp
//       writes it: `matmul_init` after the tilize phase, before the matmul.
//   REISSUE_MATMUL_INIT=0 -- the bug the hedgehope compiler team's codegen hit:
//       `matmul_init` only at the top, before the tilize. Deadlocks on silicon
//       (Watcher: all compute cores at UPMD, tilize scratch CBs rcv != ack).
//
// Operands: A is the 32x32 identity, B is 1024 distinct exactly-representable
// bfloat16 values, both delivered row-major and tilized on device. C = A @ B is
// therefore exactly B, so a wrong tilize *or* a wrong matmul operand shows up
// as an unmistakable wrong value rather than a rounding question.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_rm_a = tt::CBIndex::c_0;   // A, row-major
constexpr auto cb_rm_b = tt::CBIndex::c_1;   // B, row-major
constexpr auto cb_tiled_a = tt::CBIndex::c_4;  // A, tilized on device
constexpr auto cb_tiled_b = tt::CBIndex::c_5;  // B, tilized on device
constexpr auto cb_out = tt::CBIndex::c_16;

#ifndef REISSUE_MATMUL_INIT
#define REISSUE_MATMUL_INIT 1
#endif

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_tiled_a, cb_tiled_b, cb_out);

    // The matmul config the buggy codegen relied on: issued once, at the top,
    // *before* any tilize. Every `tilize_init` below overwrites it.
    matmul_init(cb_tiled_a, cb_tiled_b, 0);

    // --- tilize A (cb_rm_a row-major -> cb_tiled_a) ---
    tilize_init(cb_rm_a, 1, cb_tiled_a);
    cb_wait_front(cb_rm_a, 1);
    cb_reserve_back(cb_tiled_a, 1);
    tilize_block(cb_rm_a, 1, cb_tiled_a);
    cb_push_back(cb_tiled_a, 1);
    cb_pop_front(cb_rm_a, 1);
    tilize_uninit(cb_rm_a, cb_tiled_a);

    // --- tilize B (cb_rm_b row-major -> cb_tiled_b) ---
    tilize_init(cb_rm_b, 1, cb_tiled_b);
    cb_wait_front(cb_rm_b, 1);
    cb_reserve_back(cb_tiled_b, 1);
    tilize_block(cb_rm_b, 1, cb_tiled_b);
    cb_push_back(cb_tiled_b, 1);
    cb_pop_front(cb_rm_b, 1);
    tilize_uninit(cb_rm_b, cb_tiled_b);

    // --- matmul over the tilized operands ---
    cb_wait_front(cb_tiled_a, 1);
    cb_wait_front(cb_tiled_b, 1);
#if REISSUE_MATMUL_INIT
    // The missing line. Without it the unpacker is still running the tilize MOP.
    matmul_init(cb_tiled_a, cb_tiled_b, 0);
#endif
    tile_regs_acquire();
    matmul_tiles(cb_tiled_a, cb_tiled_b, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_tiled_a, 1);
    cb_pop_front(cb_tiled_b, 1);
}
