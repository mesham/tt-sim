// Fills both operand CBs with their whole tile block in one go, so the compute
// kernel can hold them resident and index tiles directly.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t a_dram = get_arg_val<uint32_t>(0);
    uint32_t a_tiles = get_arg_val<uint32_t>(1);
    uint32_t b_dram = get_arg_val<uint32_t>(2);
    uint32_t b_tiles = get_arg_val<uint32_t>(3);

    uint64_t a_noc_addr = get_noc_addr_from_bank_id<true>(0, a_dram);
    uint64_t b_noc_addr = get_noc_addr_from_bank_id<true>(0, b_dram);

    constexpr uint32_t cb_id_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_b = tt::CBIndex::c_1;

    // Reserve each whole block, so its pages are contiguous in L1 and a single
    // read fills them.
    cb_reserve_back(cb_id_a, a_tiles);
    cb_reserve_back(cb_id_b, b_tiles);

    noc_async_read(a_noc_addr, get_write_ptr(cb_id_a), a_tiles * TILE_BYTES);
    noc_async_read(b_noc_addr, get_write_ptr(cb_id_b), b_tiles * TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_id_a, a_tiles);
    cb_push_back(cb_id_b, b_tiles);
}
