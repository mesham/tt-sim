// Walk an interleaved DRAM buffer page by page, so every DRAM bank is touched.
//
// `InterleavedAddrGen` recomputes, device-side and per page, the bank index
// (`page_id % NUM_DRAM_BANKS`), the within-bank offset
// (`page_id / NUM_DRAM_BANKS * aligned_page_size`), the bank's base offset
// (`bank_to_dram_offset[bank]`) and the bank's NoC coordinate
// (`dram_bank_to_noc_xy[noc][bank]`). The latter two are tables the host wrote
// into this core's L1 at init, so the kernel's gather can only match the host's
// scatter if the simulator keeps each bank's bytes genuinely apart and routes
// to each bank's own coordinate.
void kernel_main() {
    uint32_t src0_dram = get_arg_val<uint32_t>(0);
    uint32_t src1_dram = get_arg_val<uint32_t>(1);
    uint32_t dst_dram = get_arg_val<uint32_t>(2);
    uint32_t buffer_1_addr = get_arg_val<uint32_t>(3);
    uint32_t buffer_2_addr = get_arg_val<uint32_t>(4);
    uint32_t page_size = get_arg_val<uint32_t>(5);
    uint32_t n_pages = get_arg_val<uint32_t>(6);

    const InterleavedAddrGen<true> src0 = {.bank_base_address = src0_dram, .page_size = page_size};
    const InterleavedAddrGen<true> src1 = {.bank_base_address = src1_dram, .page_size = page_size};
    const InterleavedAddrGen<true> dst = {.bank_base_address = dst_dram, .page_size = page_size};

    uint32_t* buffer_1 = (uint32_t*)buffer_1_addr;
    uint32_t* buffer_2 = (uint32_t*)buffer_2_addr;
    uint32_t elems = page_size / 4;

    for (uint32_t page = 0; page < n_pages; page++) {
        noc_async_read(src0.get_noc_addr(page), buffer_1_addr, page_size);
        noc_async_read(src1.get_noc_addr(page), buffer_2_addr, page_size);
        noc_async_read_barrier();

        for (uint32_t i = 0; i < elems; i++) {
            buffer_1[i] = buffer_1[i] + buffer_2[i];
        }

        noc_async_write(buffer_1_addr, dst.get_noc_addr(page), page_size);
        noc_async_write_barrier();
    }
}
