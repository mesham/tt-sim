// Differential op-coverage kernel for `matmul_init`'s *transpose* operand.
//
// `matmul_init(in0, in1, transpose)` splits the flag over two engines:
//   * unpack -- `cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose)`,
//     the within-face 16x16 transpose of SrcA. Haloize_mode is a *one-bit*
//     field (BH/WH cfg_defines: SHAMT 8, MASK 0x100), so the RMWCIB write keeps
//     only bit 0 of the argument.
//   * math -- `_llk_math_matmul_init_(..., const bool transpose, ...)`, which
//     picks different address-mod increments and swaps faces 1 and 2 in the
//     MVMUL sequence. That one takes the argument as a *bool*, so any nonzero
//     value enables it.
//
// The two therefore disagree for any argument that is nonzero but even, which
// is what the hedgehope compiler team's codegen emitted (`matmul_init(5, 4, 4)`).
// Op 2 pins that down rather than leaving it to a reading of the headers.
//
//   op 0  CONTROL  matmul_init(in0, in1, 0)  -- C = A @ B
//   op 1           matmul_init(in0, in1, 1)  -- C = A @ B^T
//   op 2           matmul_init(in0, in1, 4)  -- 4 & 1 == 0 at the unpacker,
//                                               but true at the math engine
//
// A is the 32x32 identity, so C is exactly B (op 0) or exactly B^T (op 1) and
// every value is exact in bfloat16 -- a transposed-vs-not answer is unambiguous
// rather than a rounding question. Op 2 has no "right" answer to compute
// against; it is checked against the vendor reference simulator by
// optests/diff.sh, which is the whole point of running it.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

constexpr auto cb_in0 = tt::CBIndex::c_0;
constexpr auto cb_in1 = tt::CBIndex::c_1;
constexpr auto cb_out = tt::CBIndex::c_16;

constexpr uint32_t NUM_OPS = 3;  // MUST match NUM_OPS in matmultranspose.cpp
constexpr uint32_t TRANSPOSE[NUM_OPS] = {0, 1, 4};

void kernel_main() {
    // Matmul maps in0 -> SrcB and in1 -> SrcA, hence SrcOrder::Reverse.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_out);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    for (uint32_t op = 0; op < NUM_OPS; op++) {
        matmul_init(cb_in0, cb_in1, TRANSPOSE[op]);
        tile_regs_acquire();
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}
