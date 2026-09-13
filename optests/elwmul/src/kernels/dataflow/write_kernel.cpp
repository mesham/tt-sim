// Streams the compute kernel's output tiles out to DRAM, one per op.

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_bytes = get_arg_val<uint32_t>(2);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_addr);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), dst_noc_addr + (i * tile_bytes), tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
