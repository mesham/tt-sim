// Fills the three operand CBs (condition, value-if-true, value-if-false) with
// one Int32 tile each. All three stay resident for the whole run.

#define TILE_BYTES 4096  // 32x32 Int32

void kernel_main() {
    uint32_t cond_dram = get_arg_val<uint32_t>(0);
    uint32_t a_dram = get_arg_val<uint32_t>(1);
    uint32_t b_dram = get_arg_val<uint32_t>(2);

    uint64_t cond_noc_addr = get_noc_addr_from_bank_id<true>(0, cond_dram);
    uint64_t a_noc_addr = get_noc_addr_from_bank_id<true>(0, a_dram);
    uint64_t b_noc_addr = get_noc_addr_from_bank_id<true>(0, b_dram);

    constexpr uint32_t cb_cond = tt::CBIndex::c_0;
    constexpr uint32_t cb_a = tt::CBIndex::c_1;
    constexpr uint32_t cb_b = tt::CBIndex::c_2;

    cb_reserve_back(cb_cond, 1);
    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);

    noc_async_read(cond_noc_addr, get_write_ptr(cb_cond), TILE_BYTES);
    noc_async_read(a_noc_addr, get_write_ptr(cb_a), TILE_BYTES);
    noc_async_read(b_noc_addr, get_write_ptr(cb_b), TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_cond, 1);
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
}
