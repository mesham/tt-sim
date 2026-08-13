// energybench arm `sfpu` -- the vector-unit-heavy arm.
//
// Same shape as the `mm` arm, same reader and writer, same fixed two tiles of
// NoC traffic -- but the inner loop drives the SFPU (Int32 tile add) instead of
// the matrix unit. Two arms that differ in *which* unit they occupy while
// agreeing on everything else are what make a per-unit coefficient
// identifiable at all; without the pair, one "compute" column would absorb
// both.

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/add_int_sfpu.h"

void kernel_main() {
    const uint32_t inner = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_in0, cb_out);
    add_int_tile_init();

    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();
    copy_tile(cb_in0, 0, 0);
    copy_tile(cb_in1, 0, 1);
    for (uint32_t i = 0; i < inner; i++) {
        add_int_tile<DataFormat::Int32>(0, 1, 0);
    }
    tile_regs_commit();

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);

    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
