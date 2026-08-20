// Streams the compute kernel's output tiles out to DRAM, one per
// matmul-and-pack iteration, in the order the compute kernel pushes them.
//
// `stall_iters` (runtime arg 2) spins before *every* tile is drained. With a
// 2-page output CB that back-pressures the pack thread, which is the only way
// to open a wide, deterministic window in which the math thread can run ahead
// of the packer -- the hazard a hoisted `tile_regs_acquire()` stops being
// protected against. Zero in the default run.
//
// It has to be per tile, not once up front: a single stall at the start is
// spent on the first loop the compute kernel runs and leaves the second one
// running at full speed (confirmed with a semaphore probe -- MATH_PACK never
// left 0/1 during the second loop).

#define TILE_BYTES 2048  // 32x32 bfloat16

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t stall_iters = get_arg_val<uint32_t>(2);

    uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(0, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;

    for (uint32_t i = 0; i < num_tiles; i++) {
        for (volatile uint32_t s = 0; s < stall_iters; s++) {
        }
        cb_wait_front(cb_id_out0, 1);
        noc_async_write(get_read_ptr(cb_id_out0), dst_noc_addr + (i * TILE_BYTES), TILE_BYTES);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}
