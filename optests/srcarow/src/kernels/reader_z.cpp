#include <cstdint>

void kernel_main() {
    uint32_t D_col_bcast_addr     = get_arg_val<uint32_t>(0);
    uint32_t ut_row_bcast_addr    = get_arg_val<uint32_t>(1);
    uint32_t num_tiles            = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_D_col   = 0;   // D column broadcasts (16 tiles)
    constexpr uint32_t cb_ut_row  = 1;   // u_trans row broadcasts (16 per tile)

    constexpr uint32_t tile_size_bytes = 32 * 32 * 4;   // 4096 bytes
    constexpr uint32_t BCAST_TILES = 16;

    constexpr auto D_col_args = TensorAccessorArgs<0>();
    const auto D_col_accessor = TensorAccessor(D_col_args, D_col_bcast_addr, tile_size_bytes);

    constexpr auto ut_row_args = TensorAccessorArgs<D_col_args.next_compile_time_args_offset()>();
    const auto ut_row_accessor = TensorAccessor(ut_row_args, ut_row_bcast_addr, tile_size_bytes);

    // Load D column broadcasts once (16 tiles)
    cb_reserve_back(cb_D_col, BCAST_TILES);
    for (uint32_t k = 0; k < BCAST_TILES; k++) {
        noc_async_read_tile(k, D_col_accessor, get_write_ptr(cb_D_col));
        noc_async_read_barrier();
        cb_push_back(cb_D_col, 1);
        if (k + 1 < BCAST_TILES) cb_reserve_back(cb_D_col, 1);
    }

    // For each z-layer: load u_trans row broadcasts (16 tiles)
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        uint32_t bcast_base = tile_idx * BCAST_TILES;
        for (uint32_t k = 0; k < BCAST_TILES; k++) {
            cb_reserve_back(cb_ut_row, 1);
            noc_async_read_tile(bcast_base + k, ut_row_accessor, get_write_ptr(cb_ut_row));
            noc_async_read_barrier();
            cb_push_back(cb_ut_row, 1);
        }
    }
}
