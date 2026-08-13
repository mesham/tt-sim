// energybench arm `noc` -- the NoC-heavy arm.
//
// Repeated DRAM -> L1 reads on one baby RISC-V core, each barriered, so the
// core spends its time waiting on the NoC rather than executing. No Tensix
// instruction is issued at all.
//
// What this arm contributes to the activity vector is ``noc_bytes_total`` and
// ``noc_flight_cycles``, with an ``instr_retired`` count that stays roughly
// fixed as the bytes grow -- which is the whole point of having it: it moves
// one column of the design matrix while holding the others still.

void kernel_main() {
    uint32_t inner = get_arg_val<uint32_t>(0);
    uint32_t src_dram = get_arg_val<uint32_t>(1);
    uint32_t nbytes = get_arg_val<uint32_t>(2);
    uint32_t l1_addr = get_write_ptr(tt::CBIndex::c_0);

    uint64_t src_noc_addr = get_noc_addr_from_bank_id<true>(0, src_dram);

    for (uint32_t i = 0; i < inner; i++) {
        noc_async_read(src_noc_addr, l1_addr, nbytes);
        noc_async_read_barrier();
    }
}
