// energybench: the shared reader for the two compute arms (`mm`, `sfpu`).
//
// Reads exactly two tiles from DRAM into cb_in0 / cb_in1 and pushes them. The
// compute arm then works on those *resident* tiles for its whole inner loop, so
// the NoC cost of a compute arm is a fixed two tiles however large the inner
// loop is. That is what makes the compute column separable from the NoC column
// in the design matrix.

void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t tile_bytes = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(0, src0_dram);
    uint64_t src1_noc_addr = get_noc_addr_from_bank_id<true>(0, src1_dram);

    cb_reserve_back(cb_id_in0, 1);
    noc_async_read(src0_noc_addr, get_write_ptr(cb_id_in0), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, 1);

    cb_reserve_back(cb_id_in1, 1);
    noc_async_read(src1_noc_addr, get_write_ptr(cb_id_in1), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_id_in1, 1);
}
