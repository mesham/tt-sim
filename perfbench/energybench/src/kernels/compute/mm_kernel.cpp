// energybench arm `mm` -- the matrix-unit-heavy arm.
//
// Two bf16 tiles are already resident in cb_in0 / cb_in1 (the reader put them
// there once). The inner loop then issues ``matmul_tiles`` against those same
// two tiles ``inner`` times, accumulating into one destination register, and
// packs a single result at the end.
//
// Nothing is popped inside the loop, so no circular-buffer traffic, no NoC and
// no unpacker re-fill scale with ``inner``: the only thing that scales is the
// Matrix unit's occupancy. In the activity vector that shows up as
// ``busy_cycles`` on MATRIX (with the cost model on) and ``dispatch_to_MATH``
// (with it off).

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    const uint32_t inner = get_arg_val<uint32_t>(0);

    constexpr tt::CBIndex cb_in0 = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_in1 = tt::CBIndex::c_1;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    mm_init(cb_in0, cb_in1, cb_out);

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    acquire_dst();
    for (uint32_t i = 0; i < inner; i++) {
        matmul_tiles(cb_in0, cb_in1, 0, 0, 0);
    }

    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    release_dst();

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
}
