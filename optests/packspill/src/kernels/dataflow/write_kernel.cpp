
// packspill writer: wait for the ONE page the compute kernel reserved and
// pushed, then dump a whole tile's worth of L1 starting at that page — i.e.
// that page plus the fifteen behind it. Reading past the page is the probe:
// whatever lands there was put there by `pack_tile` overrunning its page, and
// the host compares the dump against the vendor reference simulator.

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t tile_bytes = get_arg_val<uint32_t>(1);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_addr);
    constexpr uint32_t cb_out = tt::CBIndex::c_1;

    cb_wait_front(cb_out, 1);
    noc_async_write(get_read_ptr(cb_out), dst_noc_addr, tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
