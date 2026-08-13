// mechbench reader -- streams `tiles` tile pairs from DRAM into cb_in0/cb_in1.
//
// Deliberately *streaming* rather than resident (which is what energybench's
// compute arms do): the mechanism this leg is about is the hand-off between
// the unpackers and the Matrix unit, and that hand-off only happens when the
// unpacker has a next tile to fetch while the Matrix unit still owns the Src
// bank. A resident-tile inner loop never re-unpacks and so never produces a
// SrcA/SrcB CLEAR stall at all.

void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t tiles = get_arg_val<uint32_t>(2);
    uint32_t tile_bytes = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(0, src0_dram);
    uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(0, src1_dram);

    for (uint32_t i = 0; i < tiles; i++) {
        cb_reserve_back(cb_id_in0, 1);
        cb_reserve_back(cb_id_in1, 1);

        noc_async_read(src0_noc_addr + i * tile_bytes, get_write_ptr(cb_id_in0), tile_bytes);
        noc_async_read(src1_noc_addr + i * tile_bytes, get_write_ptr(cb_id_in1), tile_bytes);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, 1);
        cb_push_back(cb_id_in1, 1);
    }
}
