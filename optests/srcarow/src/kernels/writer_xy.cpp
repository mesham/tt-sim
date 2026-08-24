#include <cstdint>

void kernel_main() {
    uint32_t ur_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t us_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_ur = 16;
    constexpr uint32_t cb_us = 17;
    constexpr uint32_t tile_size_bytes = 32 * 32 * 4;   // 4096 bytes

    constexpr auto ur_args = TensorAccessorArgs<0>();
    const auto ur_accessor = TensorAccessor(ur_args, ur_dram_addr, tile_size_bytes);

    constexpr auto us_args = TensorAccessorArgs<ur_args.next_compile_time_args_offset()>();
    const auto us_accessor = TensorAccessor(us_args, us_dram_addr, tile_size_bytes);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // Write ur
        cb_wait_front(cb_ur, 1);
        noc_async_write_tile(tile_idx, ur_accessor, get_read_ptr(cb_ur));
        noc_async_write_barrier();
        cb_pop_front(cb_ur, 1);

        // Write us
        cb_wait_front(cb_us, 1);
        noc_async_write_tile(tile_idx, us_accessor, get_read_ptr(cb_us));
        noc_async_write_barrier();
        cb_pop_front(cb_us, 1);
    }
}