// Streams the single output tile out to DRAM.

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    cb_wait_front(cb_id_out0, 1);
    noc_async_write(get_read_ptr(cb_id_out0), dst_noc_addr, TILE_BYTES);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
