// Fills the data CB and the ones CB with one tile each. Both stay
// resident for the whole run, so the compute kernel can re-run each op on the same
// input tile.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t src_dram = get_arg_val<uint32_t>(0);
    uint32_t ones_dram = get_arg_val<uint32_t>(1);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(0, src_dram);
    uint64_t ones_noc_addr = get_noc_addr_from_bank_id<true>(0, ones_dram);

    constexpr uint32_t cb_id_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_ones = tt::CBIndex::c_1;

    cb_reserve_back(cb_id_in, 1);
    cb_reserve_back(cb_id_ones, 1);

    noc_async_read(src_noc_addr, get_write_ptr(cb_id_in), TILE_BYTES);
    noc_async_read(ones_noc_addr, get_write_ptr(cb_id_ones), TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_id_in, 1);
    cb_push_back(cb_id_ones, 1);
}
