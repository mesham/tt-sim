// One reader's share of a simultaneous DRAM read, timed on its own wall clock.
//
// N copies of this kernel run at once, one per participating Tensix core. Each
// checks in with every other, waits until all N have checked in, and only then
// starts reading. The whole experiment is that overlap: an aggregate rate over
// N readers that never overlapped is N sequential measurements averaged, and it
// would be flat for a reason that has nothing to do with the endpoint.
//
// THREE THINGS ARE MEASURED, AND TWO OF THEM ARE CONTROLS
// -------------------------------------------------------
//   * `cycles` -- the timed burst. The measurement.
//   * `tag_ok` -- whether the bytes came from the DRAM bank the host meant.
//     Read once, before the barrier, outside the timed region. A rate measured
//     from the wrong endpoint is not a smaller result, it is a fictional one,
//     and on a harvested part it is the easiest mistake to make.
//   * `barrier_spins` -- whether this reader waited for the others. At least
//     one reader in a multi-reader point must read zero (it arrived last) and
//     at least one must read non-zero (it waited), or the readers did not
//     overlap and the point measures nothing.
//
// WHY THE CLOCK IS ONLY EVER DIFFERENCED WITHIN A CORE
// ---------------------------------------------------
// `t0` and `t1` are this core's own wall clock and only `t1 - t0` is reported
// as a duration. Blackhole cards in this project's own congestion runs carry a
// per-core clock epoch -- (11, 2) held a +323438586-cycle offset reproduced
// over five runs -- so `max(t1) - min(t0)` ACROSS cores is not a duration at
// all. The host builds the aggregate from `max(t1 - t0)`, which is epoch-free.
//
// SPDX-License-Identifier: Apache-2.0

#include "../dramratebench_layout.h"

namespace {

inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(DRAMRATEBENCH_WALL_CLOCK_L);
    return p[0];
}

}  // namespace

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(DRAMRATEBENCH_A_RESULTS);
    const uint32_t arrive_addr = get_arg_val<uint32_t>(DRAMRATEBENCH_A_ARRIVE);
    const uint32_t land_addr = get_arg_val<uint32_t>(DRAMRATEBENCH_A_LAND);
    const uint32_t land_bytes = get_arg_val<uint32_t>(DRAMRATEBENCH_A_LAND_BYTES);
    const uint32_t point = get_arg_val<uint32_t>(DRAMRATEBENCH_A_POINT);
    const uint32_t index = get_arg_val<uint32_t>(DRAMRATEBENCH_A_INDEX);
    const uint32_t num_readers = get_arg_val<uint32_t>(DRAMRATEBENCH_A_NUM_READERS);
    const uint32_t num_tx = get_arg_val<uint32_t>(DRAMRATEBENCH_A_NUM_TX);
    const uint32_t tx_bytes = get_arg_val<uint32_t>(DRAMRATEBENCH_A_TX_BYTES);
    const uint32_t dram_x = get_arg_val<uint32_t>(DRAMRATEBENCH_A_DRAM_X);
    const uint32_t dram_y = get_arg_val<uint32_t>(DRAMRATEBENCH_A_DRAM_Y);
    const uint32_t dram_addr = get_arg_val<uint32_t>(DRAMRATEBENCH_A_DRAM_ADDR);
    const uint32_t slice_bytes = get_arg_val<uint32_t>(DRAMRATEBENCH_A_SLICE_BYTES);
    const uint32_t tag = get_arg_val<uint32_t>(DRAMRATEBENCH_A_TAG);

    const uint64_t dram_noc = get_noc_addr(dram_x, dram_y, dram_addr);

    // --- the addressing proof, before anything else -------------------------
    // 32 bytes rather than 4: a DRAM-to-L1 read must be congruent in its low 5
    // bits on Wormhole and its low 6 on Blackhole, and both addresses here are
    // bank/arena bases, so a 32-byte read from offset 0 to offset 0 is
    // congruent by construction on either part.
    volatile tt_l1_ptr uint32_t* land = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land_addr);
    land[0] = ~tag;  // so a read that never happens cannot look like a match
    noc_async_read(dram_noc, land_addr, 32);
    noc_async_read_barrier();
    const uint32_t tag_seen = land[0];

    // --- check in with every reader, then wait for every reader -------------
    // The arrival array was zeroed by the host on every participating core
    // before the launch, so nothing here has to clear it -- and must not, since
    // a peer's write can arrive before this core has run its first instruction.
    volatile tt_l1_ptr uint32_t* arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr);
    const uint32_t my_slot = arrive_addr + DRAMRATEBENCH_SLOT_STRIDE * index;
    // A one-word source for the arrival writes, kept out of the landing arena
    // so an in-flight read cannot overwrite the value being sent.
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land_addr + land_bytes);
    flag[0] = DRAMRATEBENCH_ARRIVED;
    for (uint32_t p = 0; p < num_readers; p++) {
        if (p == index) {
            arrive[index * (DRAMRATEBENCH_SLOT_STRIDE / 4)] = DRAMRATEBENCH_ARRIVED;
            continue;
        }
        const uint32_t px = get_arg_val<uint32_t>(DRAMRATEBENCH_A_PEERS + 2 * p);
        const uint32_t py = get_arg_val<uint32_t>(DRAMRATEBENCH_A_PEERS + 2 * p + 1);
        noc_async_write(reinterpret_cast<uint32_t>(flag), get_noc_addr(px, py, my_slot), 4);
    }
    noc_async_write_barrier();

    uint32_t spins = 0;
    while (spins < DRAMRATEBENCH_BARRIER_LIMIT) {
        invalidate_l1_cache();
        uint32_t seen = 0;
        for (uint32_t p = 0; p < num_readers; p++) {
            if (arrive[p * (DRAMRATEBENCH_SLOT_STRIDE / 4)] != 0) {
                seen++;
            }
        }
        if (seen >= num_readers) {
            break;
        }
        spins++;
    }

    // --- the timed burst ----------------------------------------------------
    // Both offsets advance by the transaction size and wrap by compare-and-
    // reset rather than by `%`: a `remu` in the issue loop costs several cycles
    // of the very rate being measured, and one branch is the same branch in
    // every point so it cancels between them.
    const uint32_t land_span = (land_bytes > tx_bytes) ? (land_bytes - tx_bytes) : 0;
    const uint32_t slice_span = (slice_bytes > tx_bytes) ? (slice_bytes - tx_bytes) : 0;
    uint32_t src_off = 0, dst_off = 0;
    const uint32_t t0 = wall_clock_lo();
    for (uint32_t i = 0; i < num_tx; i++) {
        noc_async_read(dram_noc + src_off, land_addr + dst_off, tx_bytes);
        src_off += tx_bytes;
        if (src_off > slice_span) {
            src_off = 0;
        }
        dst_off += tx_bytes;
        if (dst_off > land_span) {
            dst_off = 0;
        }
    }
    noc_async_read_barrier();
    const uint32_t t1 = wall_clock_lo();

    // --- the stamp ----------------------------------------------------------
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    out[DRAMRATEBENCH_R_POINT] = point;
    out[DRAMRATEBENCH_R_INDEX] = index;
    out[DRAMRATEBENCH_R_T0] = t0;
    out[DRAMRATEBENCH_R_T1] = t1;
    out[DRAMRATEBENCH_R_CYCLES] = t1 - t0;  // unsigned wrap is the right answer
    const uint32_t node = NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID);
    out[DRAMRATEBENCH_R_NODE_X] = node & NOC_NODE_ID_MASK;
    out[DRAMRATEBENCH_R_NODE_Y] = (node >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    out[DRAMRATEBENCH_R_DRAM_X] = dram_x;
    out[DRAMRATEBENCH_R_DRAM_Y] = dram_y;
    out[DRAMRATEBENCH_R_NUM_TX] = num_tx;
    out[DRAMRATEBENCH_R_TX_BYTES] = tx_bytes;
    out[DRAMRATEBENCH_R_TAG_SEEN] = tag_seen;
    out[DRAMRATEBENCH_R_TAG_OK] = (tag_seen == tag) ? 1u : 0u;
    out[DRAMRATEBENCH_R_BARRIER_SPINS] = spins;
    out[DRAMRATEBENCH_R_NUM_READERS] = num_readers;
    out[DRAMRATEBENCH_R_MAGIC] = DRAMRATEBENCH_MAGIC;  // last, so a partial write shows
}
