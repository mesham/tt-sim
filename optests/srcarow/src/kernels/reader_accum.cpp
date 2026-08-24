#include <cstdint>

// For w = Dt*ur + us*D + ut*D, the reader sends 3 rounds of broadcast tiles
// per z-layer through cb_left (c_0) and cb_right (c_1):
//   Round 1: Dt_col_bcast[16], ur_row_bcast[16]
//   Round 2: us_col_bcast[16], D_row_bcast[16]
//   Round 3: ut_col_bcast[16], D_row_bcast[16]

void kernel_main() {
    uint32_t Dt_col_bcast_addr  = get_arg_val<uint32_t>(0);
    uint32_t D_row_bcast_addr   = get_arg_val<uint32_t>(1);
    uint32_t ur_row_bcast_addr  = get_arg_val<uint32_t>(2);
    uint32_t us_col_bcast_addr  = get_arg_val<uint32_t>(3);
    uint32_t ut_col_bcast_addr  = get_arg_val<uint32_t>(4);
    uint32_t num_tiles          = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_left  = 0;    // left col broadcasts
    constexpr uint32_t cb_right = 1;    // right row broadcasts

    constexpr uint32_t tile_size_bytes = 32 * 32 * 4;   // 4096 bytes
    constexpr uint32_t BCAST_TILES = 16;

    constexpr auto Dt_col_args = TensorAccessorArgs<0>();
    const auto Dt_col_accessor = TensorAccessor(Dt_col_args, Dt_col_bcast_addr, tile_size_bytes);

    constexpr auto D_row_args = TensorAccessorArgs<Dt_col_args.next_compile_time_args_offset()>();
    const auto D_row_accessor = TensorAccessor(D_row_args, D_row_bcast_addr, tile_size_bytes);

    constexpr auto ur_row_args = TensorAccessorArgs<D_row_args.next_compile_time_args_offset()>();
    const auto ur_row_accessor = TensorAccessor(ur_row_args, ur_row_bcast_addr, tile_size_bytes);

    constexpr auto us_col_args = TensorAccessorArgs<ur_row_args.next_compile_time_args_offset()>();
    const auto us_col_accessor = TensorAccessor(us_col_args, us_col_bcast_addr, tile_size_bytes);

    constexpr auto ut_col_args = TensorAccessorArgs<us_col_args.next_compile_time_args_offset()>();
    const auto ut_col_accessor = TensorAccessor(ut_col_args, ut_col_bcast_addr, tile_size_bytes);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        uint32_t bcast_base = tile_idx * BCAST_TILES;

        // Round 1: Dt_col + ur_row → Dt * ur
        for (uint32_t k = 0; k < BCAST_TILES; k++) {
            cb_reserve_back(cb_left, 1);
            noc_async_read_tile(k, Dt_col_accessor, get_write_ptr(cb_left));
            noc_async_read_barrier();
            cb_push_back(cb_left, 1);
        }
        for (uint32_t k = 0; k < BCAST_TILES; k++) {
            cb_reserve_back(cb_right, 1);
            noc_async_read_tile(bcast_base + k, ur_row_accessor, get_write_ptr(cb_right));
            noc_async_read_barrier();
            cb_push_back(cb_right, 1);
        }

        // Round 2: us_col + D_row → us * D
        for (uint32_t k = 0; k < BCAST_TILES; k++) {
            cb_reserve_back(cb_left, 1);
            noc_async_read_tile(bcast_base + k, us_col_accessor, get_write_ptr(cb_left));
            noc_async_read_barrier();
            cb_push_back(cb_left, 1);
        }
        for (uint32_t k = 0; k < BCAST_TILES; k++) {
            cb_reserve_back(cb_right, 1);
            noc_async_read_tile(k, D_row_accessor, get_write_ptr(cb_right));
            noc_async_read_barrier();
            cb_push_back(cb_right, 1);
        }

        // Round 3: ut_col + D_row → ut * D
        for (uint32_t k = 0; k < BCAST_TILES; k++) {
            cb_reserve_back(cb_left, 1);
            noc_async_read_tile(bcast_base + k, ut_col_accessor, get_write_ptr(cb_left));
            noc_async_read_barrier();
            cb_push_back(cb_left, 1);
        }
        for (uint32_t k = 0; k < BCAST_TILES; k++) {
            cb_reserve_back(cb_right, 1);
            noc_async_read_tile(k, D_row_accessor, get_write_ptr(cb_right));
            noc_async_read_barrier();
            cb_push_back(cb_right, 1);
        }
    }
}
