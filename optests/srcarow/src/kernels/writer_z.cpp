#include <cstdint>

void kernel_main() {
    uint32_t ut_trans_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_ut = 16;
    constexpr uint32_t tile_size_bytes = 32 * 32 * 4;   // 4096 bytes

    constexpr auto ut_args = TensorAccessorArgs<0>();
    const auto ut_accessor = TensorAccessor(ut_args, ut_trans_dram_addr, tile_size_bytes);

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        cb_wait_front(cb_ut, 1);
        noc_async_write_tile(tile_idx, ut_accessor, get_read_ptr(cb_ut));
        noc_async_write_barrier();
        cb_pop_front(cb_ut, 1);
    }
}