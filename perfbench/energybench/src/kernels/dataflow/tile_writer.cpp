// energybench: the shared writer for the two compute arms (`mm`, `sfpu`).
//
// Drains the single output tile the compute kernel packs. It exists so the
// output circular buffer is genuinely consumed rather than left half-full
// across relaunches, and so the two compute arms carry an identical, fixed
// NoC tail.

void kernel_main() {
    uint32_t dst_dram = get_arg_val<uint32_t>(0);
    uint32_t tile_bytes = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_dram);

    cb_wait_front(cb_id_out0, 1);
    noc_async_write(get_read_ptr(cb_id_out0), dst_noc_addr, tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
