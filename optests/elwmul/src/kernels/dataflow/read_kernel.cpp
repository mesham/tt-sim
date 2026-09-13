// Fills the two operand CBs with one tile each. Both stay resident for the
// whole run, so the compute kernel can apply every op to the same pair.

void kernel_main() {
    uint32_t a_dram = get_arg_val<uint32_t>(0);
    uint32_t b_dram = get_arg_val<uint32_t>(1);
    uint32_t tile_bytes = get_arg_val<uint32_t>(2);

    uint64_t a_noc_addr = get_noc_addr_from_bank_id<true>(0, a_dram);
    uint64_t b_noc_addr = get_noc_addr_from_bank_id<true>(0, b_dram);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);

    noc_async_read(a_noc_addr, get_write_ptr(cb_a), tile_bytes);
    noc_async_read(b_noc_addr, get_write_ptr(cb_b), tile_bytes);
    noc_async_read_barrier();

    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
}
