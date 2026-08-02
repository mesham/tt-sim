// Fills both operand CBs with their whole tile block in one go, so the compute
// kernel can hold them resident and index tiles directly.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(0, src0_dram);
    uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(0, src1_dram);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    // Reserve the whole block, so the reserved pages are contiguous in L1 and a
    // single read per operand fills them.
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_reserve_back(cb_id_in1, num_tiles);

    noc_async_read(src0_noc_addr, get_write_ptr(cb_id_in0), num_tiles * TILE_BYTES);
    noc_async_read(src1_noc_addr, get_write_ptr(cb_id_in1), num_tiles * TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in1, num_tiles);
}
