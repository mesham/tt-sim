// Fills the two operand CBs with one tile each: in0 is the ramp, in1 is the
// identity tile. Both stay resident for the whole run, so the compute kernel
// can re-run the same phase-1 matmul once per op.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t in0_dram = get_arg_val<uint32_t>(0);
    uint32_t in1_dram = get_arg_val<uint32_t>(1);

    uint64_t in0_noc_addr = get_noc_addr_from_bank_id<true>(0, in0_dram);
    uint64_t in1_noc_addr = get_noc_addr_from_bank_id<true>(0, in1_dram);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;

    cb_reserve_back(cb_id_in0, 1);
    cb_reserve_back(cb_id_in1, 1);

    noc_async_read(in0_noc_addr, get_write_ptr(cb_id_in0), TILE_BYTES);
    noc_async_read(in1_noc_addr, get_write_ptr(cb_id_in1), TILE_BYTES);
    noc_async_read_barrier();

    cb_push_back(cb_id_in0, 1);
    cb_push_back(cb_id_in1, 1);
}
