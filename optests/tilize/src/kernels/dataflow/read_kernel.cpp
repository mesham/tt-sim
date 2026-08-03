// Fills the three input CBs: one already-tiled tile (the control), one 32x32
// row-major tile, and one 32x(32*WIDE_TILES) row-major block. The row-major
// ones are just bytes as far as the reader is concerned -- it is the tilize
// unpack that gives them their layout.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t tiled_dram = get_arg_val<uint32_t>(0);
    uint32_t rm1_dram = get_arg_val<uint32_t>(1);
    uint32_t rm2_dram = get_arg_val<uint32_t>(2);
    uint32_t wide_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_tiled = tt::CBIndex::c_0;
    constexpr uint32_t cb_rm1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_rm2 = tt::CBIndex::c_2;

    cb_reserve_back(cb_tiled, 1);
    cb_reserve_back(cb_rm1, 1);
    cb_reserve_back(cb_rm2, wide_tiles);

    noc_async_read(get_noc_addr_from_bank_id<true>(0, tiled_dram), get_write_ptr(cb_tiled), TILE_BYTES);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, rm1_dram), get_write_ptr(cb_rm1), TILE_BYTES);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, rm2_dram), get_write_ptr(cb_rm2), wide_tiles * TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_tiled, 1);
    cb_push_back(cb_rm1, 1);
    cb_push_back(cb_rm2, wide_tiles);
}
