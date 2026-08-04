// Fills the two operand CBs: NUM_OUT tiles of A and one tile of B, both already
// tiled. The compute kernel produces one output tile per A tile.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t a_dram = get_arg_val<uint32_t>(0);
    uint32_t b_dram = get_arg_val<uint32_t>(1);
    uint32_t num_a_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    cb_reserve_back(cb_a, num_a_tiles);
    cb_reserve_back(cb_b, 1);

    noc_async_read(get_noc_addr_from_bank_id<true>(0, a_dram), get_write_ptr(cb_a), num_a_tiles * TILE_BYTES);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, b_dram), get_write_ptr(cb_b), TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_a, num_a_tiles);
    cb_push_back(cb_b, 1);
}
