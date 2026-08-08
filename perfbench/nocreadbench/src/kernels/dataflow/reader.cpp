// One NoC read burst, timed on the issuing core's own wall clock, plus an
// untimed repeat of the same burst during which the initiator's own
// outstanding-request counter is sampled.
//
// WHY THE TWO BURSTS ARE SEPARATE
// ------------------------------
// The sample is `NOC_STATUS_READ_REG(noc, NIU_MST_REQS_OUTSTANDING_ID(0))`, an
// MMIO load into the NoC register block. That block is the ">= 7" row of the
// baby RISC-V load-latency table, so each sample costs a six-cycle load-use
// interlock inside a loop whose whole per-iteration cost is under 40 cycles.
// Sampling inside the timed region would change the rate it is meant to
// explain. So: burst 1 is timed and unsampled, burst 2 is sampled and
// untimed, and only the counter's high-water mark comes out of burst 2.
//
// WHAT THE HIGH-WATER MARK DECIDES
// --------------------------------
// If the count plateaus at some K well below num_tx, the initiator IS holding
// a bounded number of read requests in flight, and K is measured directly --
// no arithmetic, and in particular no arithmetic that has to separate the
// limit from the responder's L1 read ports. If instead it climbs with num_tx
// (bounded only by the counter's own 8 bits), the initiator imposes no such
// limit and whatever caps the sustained rate is downstream of it.
//
// SPDX-License-Identifier: Apache-2.0

#include "../nocreadbench_layout.h"

namespace {

inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(NOCREADBENCH_WALL_CLOCK_L);
    return p[0];
}

inline uint32_t outstanding(uint32_t trid) { return NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(trid)); }

inline uint32_t cmd_buf_avail() {
#if NOCREADBENCH_HAVE_CMD_BUF_AVAIL
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(CMD_BUF_AVAIL);
    return p[0];
#else
    return 0xFFFFFFFFu;
#endif
}

}  // namespace

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(NOCREADBENCH_A_RESULTS);
    const uint32_t data_addr = get_arg_val<uint32_t>(NOCREADBENCH_A_DATA);
    const uint32_t data_bytes = get_arg_val<uint32_t>(NOCREADBENCH_A_DATA_BYTES);
    const uint32_t point = get_arg_val<uint32_t>(NOCREADBENCH_A_POINT);
    const uint32_t num_tx = get_arg_val<uint32_t>(NOCREADBENCH_A_NUM_TX);
    const uint32_t tx_bytes = get_arg_val<uint32_t>(NOCREADBENCH_A_TX_BYTES);
    const uint32_t dst_stride = get_arg_val<uint32_t>(NOCREADBENCH_A_DST_STRIDE);
    const uint32_t src_stride = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC_STRIDE);
    // Clamped, not trusted: `si` indexes `src_noc` below, so a plan that asks
    // for more sources than the array holds must fold rather than read past it.
    const uint32_t num_src_arg = get_arg_val<uint32_t>(NOCREADBENCH_A_NUM_SRC);
    const uint32_t num_src = (num_src_arg == 0) ? 1 : (num_src_arg > 8 ? 8 : num_src_arg);
    const uint32_t sample = get_arg_val<uint32_t>(NOCREADBENCH_A_SAMPLE);

    // The arena is split in half: reads land in the top half, sources sit in
    // the bottom half of the *remote* core's identically-addressed arena. The
    // halves never overlap, so a landing never clobbers a source even in the
    // loopback case.
    const uint32_t half = data_bytes / 2;
    const uint32_t dst_base = data_addr + half;
    const uint32_t src_base = data_addr;
    // The strides wrap by compare-and-reset, never by `%`: a `remu` in the
    // issue loop would cost several cycles of the very quantity being
    // measured. Compare-and-reset is one branch, and the same one in every
    // experiment, so it cancels between points.
    const uint32_t dst_span = (half > tx_bytes) ? (half - tx_bytes) : 0;
    const uint32_t src_span = (half > tx_bytes) ? (half - tx_bytes) : 0;

    const uint32_t rest_avail = cmd_buf_avail();

    // Source NoC addresses, precomputed so that cycling sources costs an array
    // load rather than a `get_noc_addr` per iteration.
    uint64_t src_noc[8];
    for (uint32_t s = 0; s < num_src; s++) {
        const uint32_t sx = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC + 2 * s);
        const uint32_t sy = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC + 2 * s + 1);
        src_noc[s] = get_noc_addr(sx, sy, src_base);
    }

    // --- burst 1: timed, unsampled -----------------------------------------
    uint32_t dst_off = 0, src_off = 0, si = 0;
    const uint32_t t0 = wall_clock_lo();
    for (uint32_t i = 0; i < num_tx; i++) {
        noc_async_read(src_noc[si] + src_off, dst_base + dst_off, tx_bytes);
        dst_off += dst_stride;
        if (dst_off > dst_span) {
            dst_off = 0;
        }
        src_off += src_stride;
        if (src_off > src_span) {
            src_off = 0;
        }
        si++;
        if (si >= num_src) {
            si = 0;
        }
    }
    noc_async_read_barrier();
    const uint32_t t1 = wall_clock_lo();

    // --- burst 2: sampled, untimed -----------------------------------------
    uint32_t out_max = 0, out_end = 0, samples = 0, busy_avail = 0xFFFFFFFFu;
    if (sample != 0) {
        dst_off = 0;
        src_off = 0;
        si = 0;
        for (uint32_t i = 0; i < num_tx; i++) {
            noc_async_read(src_noc[si] + src_off, dst_base + dst_off, tx_bytes);
            const uint32_t o = outstanding(0);
            if (o > out_max) {
                out_max = o;
            }
            out_end = o;
            samples++;
            dst_off += dst_stride;
            if (dst_off > dst_span) {
                dst_off = 0;
            }
            src_off += src_stride;
            if (src_off > src_span) {
                src_off = 0;
            }
            si++;
            if (si >= num_src) {
                si = 0;
            }
        }
        busy_avail = cmd_buf_avail();
        noc_async_read_barrier();
    }

    // --- the stamp ---------------------------------------------------------
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);
    out[NOCREADBENCH_R_POINT] = point;
    out[NOCREADBENCH_R_T0] = t0;
    out[NOCREADBENCH_R_T1] = t1;
    out[NOCREADBENCH_R_CYCLES] = t1 - t0;  // unsigned wrap is the right answer
    const uint32_t node = NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID);
    out[NOCREADBENCH_R_NODE_X] = node & NOC_NODE_ID_MASK;
    out[NOCREADBENCH_R_NODE_Y] = (node >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    out[NOCREADBENCH_R_NUM_TX] = num_tx;
    out[NOCREADBENCH_R_TX_BYTES] = tx_bytes;
    out[NOCREADBENCH_R_NOC] = noc_index;
    out[NOCREADBENCH_R_OUTSTANDING_MAX] = out_max;
    out[NOCREADBENCH_R_OUTSTANDING_END] = out_end;
    out[NOCREADBENCH_R_SAMPLES] = samples;
    out[NOCREADBENCH_R_CMDBUF_AVAIL_REST] = rest_avail;
    out[NOCREADBENCH_R_CMDBUF_AVAIL_BUSY] = busy_avail;
    out[NOCREADBENCH_R_MAGIC] = NOCREADBENCH_MAGIC;  // last, so a partial write shows
}
