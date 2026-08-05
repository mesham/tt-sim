
// packspill reader: pull one whole Int32 tile from DRAM into cb_in's single
// full-tile page. Nothing subtle here — the point of the program is what the
// *packer* does on the way out.

void kernel_main() {
    uint32_t src_dram = get_arg_val<uint32_t>(0);
    uint32_t tile_bytes = get_arg_val<uint32_t>(1);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(0, src_dram);
    constexpr uint32_t cb_in = tt::CBIndex::c_0;

    cb_reserve_back(cb_in, 1);
    noc_async_read(src_noc_addr, get_write_ptr(cb_in), tile_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);
}
