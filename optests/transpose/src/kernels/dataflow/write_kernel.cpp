// Streams the compute kernel's output tiles out to DRAM, one per op.

#define TILE_BYTES 4096  // 32x32 Float32

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out0, 1);
        noc_async_write(get_read_ptr(cb_id_out0), dst_noc_addr + (i * TILE_BYTES), TILE_BYTES);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
