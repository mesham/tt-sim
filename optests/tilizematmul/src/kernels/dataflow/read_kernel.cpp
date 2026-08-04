// Fills the two row-major operand CBs. Row-major is just bytes as far as the
// reader is concerned -- the compute kernel's `tilize_block` gives them their
// tiled layout.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t a_dram = get_arg_val<uint32_t>(0);
    uint32_t b_dram = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_rm_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_rm_b = tt::CBIndex::c_1;

    cb_reserve_back(cb_rm_a, 1);
    cb_reserve_back(cb_rm_b, 1);

    noc_async_read(get_noc_addr_from_bank_id<true>(0, a_dram), get_write_ptr(cb_rm_a), TILE_BYTES);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, b_dram), get_write_ptr(cb_rm_b), TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_rm_a, 1);
    cb_push_back(cb_rm_b, 1);
}
