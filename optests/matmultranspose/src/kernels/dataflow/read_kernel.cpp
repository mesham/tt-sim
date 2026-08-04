// Fills the two operand CBs with one already-tiled tile each.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t in0_dram = get_arg_val<uint32_t>(0);
    uint32_t in1_dram = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    cb_reserve_back(cb_in0, 1);
    cb_reserve_back(cb_in1, 1);

    noc_async_read(get_noc_addr_from_bank_id<true>(0, in0_dram), get_write_ptr(cb_in0), TILE_BYTES);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, in1_dram), get_write_ptr(cb_in1), TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_in0, 1);
    cb_push_back(cb_in1, 1);
}
