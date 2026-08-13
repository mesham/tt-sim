// mechbench writer -- drains the output circular buffer back to DRAM, one tile
// at a time, so the host can validate the arithmetic. Keeping the output CB
// genuinely consumed also stops the packer thread from blocking on a full CB
// for the whole run, which would swamp the measurement with one mechanism.

void kernel_main() {
    uint32_t dst_dram = get_arg_val<uint32_t>(0);
    uint32_t tiles = get_arg_val<uint32_t>(1);
    uint32_t tile_bytes = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_dram);

    for (uint32_t i = 0; i < tiles; i++) {
        cb_wait_front(cb_id_out0, 1);
        noc_async_write(get_read_ptr(cb_id_out0), dst_noc_addr + i * tile_bytes, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
