// Fills the input CB with the single Float32 tile the compute kernel keeps
// resident for the whole run.

#define TILE_BYTES 4096  // 32x32 Float32

void kernel_main() {
    uint32_t src_dram = get_arg_val<uint32_t>(0);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(0, src_dram);

    constexpr uint32_t cb_id_in = tt::CBIndex::c_0;

    cb_reserve_back(cb_id_in, 1);
    noc_async_read(src_noc_addr, get_write_ptr(cb_id_in), TILE_BYTES);
    noc_async_read_barrier();
    cb_push_back(cb_id_in, 1);
}
