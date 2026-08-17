// One writer's share of a simultaneous DRAM write, timed on its own wall clock.
//
// The mirror of dram_reader.cpp: same barrier, same clock discipline, same
// issue loop, same offset arithmetic, same stamp. Only `noc_async_read` becomes
// `noc_async_write` and only the direction of the addressing proof turns round.
//
// WHY THIS IS A SECOND FILE AND NOT A `if (direction)` IN THE FIRST
// -----------------------------------------------------------------
// The read arm of this benchmark has a published control to reproduce -- a
// Wormhole card measured it on 2026-08-17 and the numbers are pinned -- and the
// cheapest possible guarantee that a write arm did not perturb it is that the
// reader's translation unit is byte-for-byte the file that produced them. It
// is: dram_reader.cpp is unchanged by the write direction, including its stamp,
// because DRAMRATEBENCH_DIR_READ is 0 and the host zeroes the result block
// before every launch. The duplicated barrier below is the price, and it is the
// right way round: a shared kernel would have made the control cheaper to write
// and impossible to certify.
//
// WHAT A WRITER CAN AND CANNOT PROVE ABOUT ITSELF
// ----------------------------------------------
// A read that misses its endpoint returns the wrong bytes and the reader knows.
// A write that misses its endpoint returns nothing and completes normally, so
// there is no self-check that can catch it. This kernel does the half it can:
//
//   * it fills its whole landing arena with the witness word BEFORE the timed
//     region, so every byte of every timed write carries the proof -- the
//     addressing evidence is the measured traffic itself and not a separate
//     untimed poke afterwards that might have taken a different path;
//   * after the write barrier it reads its own target block back and stamps
//     what it found. That fires on a write that was dropped, mis-congruent or
//     never issued (the block still holds POISON) and it fires for free.
//
// It CANNOT fire on a write that went to the wrong bank, because the readback
// uses the same coordinate and the same offset the write used: a consistently
// misdirected write is read back consistently misdirected and matches. The
// host does that half, by reading every block of the write region back over
// PCIe and requiring that the untargeted ones still hold POISON.
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
    const uint32_t witness = get_arg_val<uint32_t>(DRAMRATEBENCH_A_WITNESS);

    const uint64_t dram_noc = get_noc_addr(dram_x, dram_y, dram_addr);

    // --- the payload IS the proof, laid down before anything is timed --------
    // Every word of the arena is the witness, so whatever slice of the arena a
    // timed write happens to pick up carries it. The alternative -- writing the
    // witness once, separately, after the burst -- would prove only that the
    // extra write was addressed correctly, which is not the traffic being
    // measured and could differ from it in exactly the way that matters.
    volatile tt_l1_ptr uint32_t* land = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(land_addr);
    for (uint32_t w = 0; w < land_bytes / 4; w++) {
        land[w] = witness;
    }

    // --- check in with every writer, then wait for every writer -------------
    // Identical to dram_reader.cpp's, deliberately: the barrier is not a
    // variable of this experiment and the two directions must not differ in it.
    // The arrival array was zeroed by the host on every participating core
    // before the launch, so nothing here has to clear it -- and must not, since
    // a peer's write can arrive before this core has run its first instruction.
    volatile tt_l1_ptr uint32_t* arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr);
    const uint32_t my_slot = arrive_addr + DRAMRATEBENCH_SLOT_STRIDE * index;
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
    // The reader's loop with the two addresses exchanged, and with the SAME
    // wrap-by-compare-and-reset rather than a `%`, for the same reason: a `remu`
    // in the issue loop costs several cycles of the rate being measured, and the
    // two directions must pay the identical instruction overhead or their
    // difference is partly this loop's.
    //
    // `dst_off` starts at 0, so the FIRST timed write covers the target block
    // the host reads back. A point whose writes were all dropped therefore
    // leaves POISON where the host looks, rather than leaving a stale correct
    // value from a previous repeat.
    const uint32_t land_span = (land_bytes > tx_bytes) ? (land_bytes - tx_bytes) : 0;
    const uint32_t slice_span = (slice_bytes > tx_bytes) ? (slice_bytes - tx_bytes) : 0;
    uint32_t src_off = 0, dst_off = 0;
    const uint32_t t0 = wall_clock_lo();
    for (uint32_t i = 0; i < num_tx; i++) {
        noc_async_write(land_addr + src_off, dram_noc + dst_off, tx_bytes);
        src_off += tx_bytes;
        if (src_off > land_span) {
            src_off = 0;
        }
        dst_off += tx_bytes;
        if (dst_off > slice_span) {
            dst_off = 0;
        }
    }
    noc_async_write_barrier();
    const uint32_t t1 = wall_clock_lo();

    // --- the half of the addressing proof this core can make ----------------
    // Untimed, after the barrier, and landing at the arena BASE. Not at the top
    // of the arena, which is where this first tried to put it: the base is the
    // only address in the arena guaranteed congruent with the slice base it is
    // read from, and a DRAM-to-L1 read must be congruent in its low 6 bits on
    // Blackhole. `land_addr + land_bytes - 32` happens to satisfy the low 5
    // Wormhole needs and fails Blackhole's low 6, so that version passed on one
    // part and raised NoCAlignmentError on the other -- which on a card would
    // have been UndefinedBehavior with no fault at all, and therefore a device
    // check that silently reported whatever the skew left behind.
    //
    // The arena's contents no longer matter here: every timed write has retired
    // at the barrier above, so overwriting the witness fill is free.
    land[0] = ~witness;  // so a read that never happens cannot look like a match
    noc_async_read(dram_noc, land_addr, DRAMRATEBENCH_TAG_BYTES);
    noc_async_read_barrier();
    const uint32_t seen = land[0];

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
    out[DRAMRATEBENCH_R_TAG_SEEN] = seen;
    out[DRAMRATEBENCH_R_TAG_OK] = (seen == witness) ? 1u : 0u;
    out[DRAMRATEBENCH_R_BARRIER_SPINS] = spins;
    out[DRAMRATEBENCH_R_NUM_READERS] = num_readers;
    out[DRAMRATEBENCH_R_WITNESS] = witness;
    out[DRAMRATEBENCH_R_DIR] = DRAMRATEBENCH_DIR_WRITE;
    out[DRAMRATEBENCH_R_MAGIC] = DRAMRATEBENCH_MAGIC;  // last, so a partial write shows
}
