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
// THE HIGH-WATER MARK IS A DIFFERENCE (this is what NRB1 got wrong)
// ----------------------------------------------------------------
// The 2026-08-09 Blackhole card session read `outstanding_max` = 72 in every
// one of its 129 rows, at num_tx = 4 as well as 128, and `outstanding_max ==
// outstanding_end` everywhere. Four requests cannot put 72 in a counter of
// in-flight requests, so the raw value was not the occupancy: the counter is
// live hardware state that the kernel inherits at some non-zero value and never
// referenced against. The structure was there all along -- 71 for one family of
// points, 72 for most, 83 for the one 4096-byte point -- but with no baseline
// sample it could not be read, and `worst + 4 >= num_tx` compared 83 against 64
// and printed NO INITIATOR LIMIT. So this version samples every counter once at
// rest, before a single request is issued, and the occupancy is the difference.
//
// It also stops assuming which transaction id the requests carry. Plain
// `noc_async_read` never writes NOC_PACKET_TAG (see `ncrisc_noc_fast_read` in
// tt-metal's blackhole/wormhole noc_nonblocking_api.h), so the id is whatever
// that command buffer's sticky tag last held. `noc_async_read_set_trid` pins
// it. And because a pinned id is still an assumption about which counter to
// watch, the same quantity is measured a second, id-independent way:
// NIU_MST_RD_REQ_SENT - NIU_MST_RD_RESP_RECEIVED, two cumulative one-per-read
// counters whose difference is the in-flight count whatever the id.
//
// THE TWO ISSUE LOOPS (NRB3)
// --------------------------
// `run_bursts` is one body compiled twice, `if constexpr` on the mode, so the
// STATELESS instantiation is the same program the 2026-08-17 Wormhole session
// timed and the STATEFUL one differs from it only where the difference is the
// experiment. What the stateful path removes, per transaction, comes straight
// out of the vendor headers:
//
//   stateless   `noc_async_read` -> `ncrisc_noc_fast_read`
//               (wormhole/noc_nonblocking_api.h:415) writes NOC_RET_ADDR_LO,
//               NOC_TARG_ADDR_LO, NOC_TARG_ADDR_COORDINATE, NOC_AT_LEN_BE and
//               NOC_CMD_CTRL -- five command-buffer stores, and the coordinate
//               is extracted from a 64-bit NoC address every iteration.
//               Blackhole's (:1051 region, `ncrisc_noc_fast_read`) writes
//               NOC_TARG_ADDR_MID as well: six.
//
//   stateful    `noc_async_read_set_state` (dataflow_api.h:673) ->
//               `ncrisc_noc_read_set_state` (wormhole:1117, blackhole:1321)
//               writes ONLY the coordinate (and, on Blackhole, TARG_ADDR_MID),
//               once, before the loop. The per-transaction call is
//               `noc_async_read_with_state` (dataflow_api.h:708) ->
//               `ncrisc_noc_read_with_state` (wormhole:1163, blackhole:1369),
//               which writes NOC_RET_ADDR_LO, NOC_TARG_ADDR_LO, NOC_AT_LEN_BE
//               and NOC_CMD_CTRL and nothing else. So the loop loses one store
//               on Wormhole and two on Blackhole, plus the 64-bit address
//               arithmetic that fed them: it deals only in 32-bit local
//               addresses.
//
// This is the same pair tt-metal's own estimator kernel switches on for the
// `stateful` rows of the shipped dataset --
// `tests/tt_metal/tt_metal/data_movement/noc_estimator_tests/kernels/reader.cpp`,
// `if constexpr (stateful)`, through `Noc::set_async_read_state` /
// `Noc::async_read_with_state` (`api/dataflow/noc.h:252`), whose default
// `max_page_size` puts it on exactly these any-length entry points.
//
// A stateful loop cannot cycle SOURCE TILES, because the source tile IS the
// state. A point with more than one source is refused rather than quietly
// measured against one of them.
//
// HOW A RUN PROVES ITS OWN MODE
// -----------------------------
// Not from the flag. After every burst is over, the kernel points the read
// state at a WITNESS tile -- a worker that is neither the initiator nor any
// source of this point, and whose source region the host has stamped with its
// own signature -- and then issues ONE transaction through the same API call
// the timed loop used, from this point's real source. The stateless call
// rewrites NOC_TARG_ADDR_COORDINATE and so answers from the SOURCE; the
// stateful call never writes it and so answers from the WITNESS. The signature
// word that lands is `NOCREADBENCH_R_PROBE`, and the host refuses any row whose
// probe disagrees with the mode it asked for.
//
// THE SOURCE ARM (NRB4)
// ---------------------
// The source of a burst is a runtime argument, not a compile-time shape, so the
// same loop reads a worker's L1 or a DRAM tile depending only on the address and
// coordinate it is handed. That is what makes the DRAM arm a *row in the same
// experiment* rather than a second program: the issue loop, the barrier, the
// stride bookkeeping and the sampling are bit-for-bit the ones the L1 arms use,
// so differencing the burst axis leaves the endpoint and nothing else.
//
// Two things that arm needs, and both are handled by the host:
//
//   * A DRAM->L1 read must satisfy `(src % n) == (dst % n)` with n = 32 on
//     Wormhole and 64 on Blackhole. It is a CONGRUENCE rule, not an absolute
//     alignment: `WormholeB0/NoC/Alignment.md` carries only congruence codes for
//     this path, violations are UndefinedBehavior (skewed or dropped bytes, no
//     fault raised), and an absolute rule would refuse correct kernels. The DRAM
//     buffer and the L1 arena come from different allocators, so the host
//     computes the shortfall and passes it as NOCREADBENCH_A_DST_PAD, shifting
//     the LANDING side and leaving DRAM addresses allocator-aligned.
//
//   * The arm has to be provable from payload. NOCREADBENCH_R_LANDED is the
//     first word the TIMED burst landed, read with a plain L1 load; the host
//     stamps DRAM tiles with NOCREADBENCH_SIG_DRAM and workers with
//     NOCREADBENCH_SIG, whose high halves differ, so a `--dram` run that in fact
//     read a worker's L1 is caught by the data rather than trusted from a flag.
//
// SPDX-License-Identifier: Apache-2.0

#include "../nocreadbench_layout.h"

namespace {

inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(NOCREADBENCH_WALL_CLOCK_L);
    return p[0];
}

inline uint32_t outstanding(uint32_t trid) { return NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(trid)); }

// Read requests in flight, independent of any transaction id. Both halves are
// cumulative and monotone, so the difference is exact even across a wrap.
inline uint32_t inflight() {
    const uint32_t sent = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_REQ_SENT);
    const uint32_t done = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
    return sent - done;
}

// Both of these must go through the NoC instance offset. NRB1 dereferenced the
// bare macro, so a kernel running on `noc_index` 1 read NoC 0's block.
inline uint32_t cmd_buf_avail() {
#if NOCREADBENCH_HAVE_CMD_BUF_AVAIL
    volatile tt_reg_ptr uint32_t* p =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(CMD_BUF_AVAIL + (noc_index << NOC_INSTANCE_OFFSET_BIT));
    return p[0];
#else
    return 0xFFFFFFFFu;
#endif
}

inline uint32_t cmd_buf_ovfl() {
#if NOCREADBENCH_HAVE_CMD_BUF_OVFL
    volatile tt_reg_ptr uint32_t* p =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(CMD_BUF_OVFL + (noc_index << NOC_INSTANCE_OFFSET_BIT));
    return p[0];
#else
    return 0xFFFFFFFFu;
#endif
}

// Everything one point needs, so the two instantiations of `run_bursts` differ
// only where the experiment says they differ.
struct Plan {
    uint64_t src_noc[8];
    uint64_t witness_noc;
    uint32_t src_base;
    // The initiator's own arena base. Identical to `src_base` for an L1 point
    // and NOT for a DRAM one, which is why the two exist separately: every
    // worker in this program has an arena here, and no worker has anything at
    // a DRAM tile's address.
    uint32_t local_base;
    uint32_t dst_base;
    uint32_t probe_dst;
    uint32_t point;
    uint32_t num_tx;
    uint32_t tx_bytes;
    uint32_t dst_stride;
    uint32_t src_stride;
    uint32_t dst_span;
    uint32_t src_span;
    uint32_t num_src;
    uint32_t trid;
    uint32_t sample;
    uint32_t src_kind;
};

template <bool STATEFUL>
void run_bursts(const Plan& p, volatile tt_l1_ptr uint32_t* out) {
    // Every loop-invariant out of the struct and into a local BEFORE the clock
    // is read, so the timed region holds the loop and the barrier and nothing
    // else.
    //
    // What this does and does not buy, measured rather than assumed. Against
    // NRB2 -- the program the 2026-08-17 Wormhole session ran -- the stateless
    // arm's PER-TRANSACTION cost is bit-identical on tt-sim: 38.00 cycles on
    // Wormhole and 41.00 on Blackhole, at all three burst intervals, before and
    // after. Its CONSTANT term moved by +4 cycles on Wormhole and +1 on
    // Blackhole, which is register allocation around a shared loop body and not
    // a change in the loop. The marginal is what the two arms are compared on
    // and what the control is checked against, precisely because differencing
    // the burst axis removes that constant; the whole-file average, which does
    // not remove it, is reported as informational for exactly this reason.
    //
    // The alternative was to duplicate the loop so the stateless arm could be
    // byte-identical. That trades the property the experiment needs -- the two
    // arms differing ONLY in the issue call, with the same stride bookkeeping,
    // the same barrier and the same sampling -- for agreement with a historical
    // constant. The wrong trade.
    const uint64_t* const src_noc = p.src_noc;
    const uint32_t src_base = p.src_base;
    const uint32_t dst_base = p.dst_base;
    const uint32_t num_tx = p.num_tx;
    const uint32_t tx_bytes = p.tx_bytes;
    const uint32_t dst_stride = p.dst_stride;
    const uint32_t src_stride = p.src_stride;
    const uint32_t dst_span = p.dst_span;
    const uint32_t src_span = p.src_span;
    const uint32_t num_src = p.num_src;
    const uint32_t trid = p.trid;
    const uint32_t sample = p.sample;

    // The state, set once. It carries the source TILE and nothing else on
    // Wormhole (`ncrisc_noc_read_set_state` writes NOC_TARG_ADDR_COORDINATE);
    // on Blackhole it carries NOC_TARG_ADDR_MID too. Deliberately OUTSIDE the
    // timed region: it is paid once for the whole burst, so leaving it in would
    // put a constant into a per-transaction rate. The marginal cost this
    // program is read for differences it away either way.
    if constexpr (STATEFUL) {
        noc_async_read_set_state(src_noc[0], noc_index);
    }

    // --- burst 1: timed, unsampled -----------------------------------------
    uint32_t dst_off = 0, src_off = 0, si = 0;
    const uint32_t t0 = wall_clock_lo();
    for (uint32_t i = 0; i < num_tx; i++) {
        if constexpr (STATEFUL) {
            noc_async_read_with_state(src_base + src_off, dst_base + dst_off, tx_bytes, noc_index);
        } else {
            noc_async_read(src_noc[si] + src_off, dst_base + dst_off, tx_bytes);
        }
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

    // --- the source witness -------------------------------------------------
    // A plain L1 load of the first word the TIMED burst landed. No transaction
    // of its own, nothing on the NoC, and it is taken here -- before the
    // sampled bursts and before the mode probe -- so that the only thing that
    // can have put a value at this address is burst 1.
    //
    // The host stamps every source region with a signature naming its tile, and
    // a DRAM tile's signature has a different high half from a worker's
    // (NOCREADBENCH_SIG_DRAM vs NOCREADBENCH_SIG), so this word says which KIND
    // of endpoint answered the burst as well as which tile. That is the source
    // arm proved from returned payload rather than from the `--dram` flag, and
    // it is the same discipline the mode probe applies to the issue loop.
    //
    // It is only DETERMINISTIC where one source is read at one offset into one
    // landing address -- src_stride == 0, dst_stride == 0, num_src == 1. The
    // host applies the check on exactly that condition and records the word
    // without judging it elsewhere; a gate that cannot fail is worse than none.
    const uint32_t landed = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_base);

    // --- the rest samples ---------------------------------------------------
    // Taken here, immediately after burst 1's barrier and immediately before
    // the sampled bursts, because that is the reference the sampled bursts are
    // differenced against. Nothing is in flight at this point by construction,
    // so `inflight_rest` MUST read zero -- and if it does not, the status block
    // is not being read the way this kernel assumes and nothing downstream of
    // it means anything. `out_rest` has no such guarantee: it is whatever live
    // hardware state the counter has been left holding, which is exactly what
    // NRB1 reported as an occupancy and read 72 for during a four-request
    // burst.
    const uint32_t rest_avail = cmd_buf_avail();
    const uint32_t rest_ovfl = cmd_buf_ovfl();
    const uint32_t out_rest = outstanding(trid);
    const uint32_t inflight_rest = inflight();

    // --- burst 2: sampled, untimed -----------------------------------------
    // Every sample is taken IMMEDIATELY after the read that could have moved
    // it, inside the loop. NRB1 read CMD_BUF_AVAIL only after the loop had
    // ended, by which point every command buffer has long since handed its
    // entry to the NIU -- which is why `cmdbuf_avail_rest` and
    // `cmdbuf_avail_busy` came back byte-identical zeros in all 129 rows of the
    // card session. Two 32-bit loads per iteration here rather than one; the
    // burst is untimed, so the only thing that costs is a slower issue rate,
    // and a slower issue rate can only ever make the occupancy look SMALLER.
    // A high-water mark measured this way is a lower bound and is read as one.
    uint32_t out_max = 0, out_end = 0, samples = 0;
    uint32_t busy_avail = 0xFFFFFFFFu, max_avail = 0;
    if (sample != 0) {
        dst_off = 0;
        src_off = 0;
        si = 0;
        for (uint32_t i = 0; i < num_tx; i++) {
            if constexpr (STATEFUL) {
                noc_async_read_with_state(src_base + src_off, dst_base + dst_off, tx_bytes, noc_index);
            } else {
                noc_async_read(src_noc[si] + src_off, dst_base + dst_off, tx_bytes);
            }
            const uint32_t o = outstanding(trid);
            const uint32_t a = cmd_buf_avail();
            if (o > out_max) {
                out_max = o;
            }
            if (a != 0xFFFFFFFFu && a > max_avail) {
                max_avail = a;
            }
            out_end = o;
            busy_avail = a;
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
        noc_async_read_barrier();
        // On a part with no CMD_BUF_AVAIL every sample is the sentinel, and a
        // peak of 0 beside a rest of 0xFFFFFFFF would read as movement. Carry
        // the sentinel through so "absent" stays one value in every column.
        if (busy_avail == 0xFFFFFFFFu) {
            max_avail = 0xFFFFFFFFu;
        }
    }

    // --- burst 3: the transaction-id-independent cross-check ----------------
    // Same burst again, sampling NIU_MST_RD_REQ_SENT - NIU_MST_RD_RESP_RECEIVED
    // instead. It answers the same question without depending on which counter
    // the requests land in, so a disagreement between `inflight_max` and
    // (`outstanding_max` - `outstanding_rest`) says the per-id counter is not
    // watching these reads and the id-free pair is the one to believe.
    uint32_t inflight_max = 0;
    if (sample != 0) {
        dst_off = 0;
        src_off = 0;
        si = 0;
        for (uint32_t i = 0; i < num_tx; i++) {
            if constexpr (STATEFUL) {
                noc_async_read_with_state(src_base + src_off, dst_base + dst_off, tx_bytes, noc_index);
            } else {
                noc_async_read(src_noc[si] + src_off, dst_base + dst_off, tx_bytes);
            }
            const uint32_t f = inflight();
            if (f > inflight_max) {
                inflight_max = f;
            }
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
    }
    const uint32_t end_ovfl = cmd_buf_ovfl();

    // --- the mode witness ---------------------------------------------------
    // Untimed, after everything, and it costs one transaction. The state is
    // pointed at the WITNESS tile; the transaction is then issued from this
    // point's REAL source through the same call the loop above used. Under
    // STATELESS that call rewrites NOC_TARG_ADDR_COORDINATE and the source
    // answers; under STATEFUL it does not and the witness answers. The landed
    // word is a tile signature the host stamped, so the answer names a tile.
    //
    // Pre-filled first, so "no transaction happened at all" reads as
    // NOCREADBENCH_PROBE_FILL rather than as whichever signature was there
    // before.
    volatile tt_l1_ptr uint32_t* probe = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p.probe_dst);
    probe[0] = NOCREADBENCH_PROBE_FILL;
    noc_async_read_set_state(p.witness_noc, noc_index);
    if constexpr (STATEFUL) {
        // `p.local_base`, not `src_base`: the state names the WITNESS, a worker
        // core, and a worker has nothing mapped at a DRAM tile's address. The
        // discriminator is which command-buffer register the call writes, not
        // which address it carries, so using the arena base here costs the
        // probe nothing and is the only address valid at the tile that is about
        // to answer.
        noc_async_read_with_state(p.local_base, p.probe_dst, tx_bytes, noc_index);
    } else {
        noc_async_read(src_noc[0], p.probe_dst, tx_bytes);
    }
    noc_async_read_barrier();
    const uint32_t probe_word = probe[0];

    // --- the stamp ---------------------------------------------------------
    out[NOCREADBENCH_R_POINT] = p.point;
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
    out[NOCREADBENCH_R_OUTSTANDING_REST] = out_rest;
    out[NOCREADBENCH_R_INFLIGHT_MAX] = inflight_max;
    out[NOCREADBENCH_R_INFLIGHT_REST] = inflight_rest;
    out[NOCREADBENCH_R_CMDBUF_AVAIL_MAX] = max_avail;
    out[NOCREADBENCH_R_CMDBUF_OVFL_REST] = rest_ovfl;
    out[NOCREADBENCH_R_CMDBUF_OVFL_END] = end_ovfl;
    out[NOCREADBENCH_R_TRID] = trid;
    out[NOCREADBENCH_R_MODE] = STATEFUL ? NOCREADBENCH_MODE_STATEFUL : NOCREADBENCH_MODE_STATELESS;
    out[NOCREADBENCH_R_PROBE] = probe_word;
    out[NOCREADBENCH_R_SRC_KIND] = p.src_kind;
    // The addresses the burst ACTUALLY issued against, so the host re-derives
    // the DRAM->L1 congruence rule from what ran rather than from what it
    // computed. A pad that failed to reach the kernel shows up here.
    out[NOCREADBENCH_R_SRC_BASE] = src_base;
    out[NOCREADBENCH_R_DST_BASE] = dst_base;
    out[NOCREADBENCH_R_LANDED] = landed;
    out[NOCREADBENCH_R_MAGIC] = NOCREADBENCH_MAGIC;  // last, so a partial write shows
}

}  // namespace

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(NOCREADBENCH_A_RESULTS);
    const uint32_t data_addr = get_arg_val<uint32_t>(NOCREADBENCH_A_DATA);
    const uint32_t data_bytes = get_arg_val<uint32_t>(NOCREADBENCH_A_DATA_BYTES);
    const uint32_t mode = get_arg_val<uint32_t>(NOCREADBENCH_A_MODE);
    // Clamped, not trusted: `si` indexes `src_noc` below, so a plan that asks
    // for more sources than the array holds must fold rather than read past it.
    const uint32_t num_src_arg = get_arg_val<uint32_t>(NOCREADBENCH_A_NUM_SRC);
    const uint32_t num_src = (num_src_arg == 0) ? 1 : (num_src_arg > 8 ? 8 : num_src_arg);

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);

    // The stateful issue path holds the source TILE in the command buffer, so
    // it cannot cycle sources without paying back the very store the variant
    // exists to remove. Refuse rather than measure a different experiment under
    // the same name. The host drops these points itself; this is the backstop.
    if (mode == NOCREADBENCH_MODE_STATEFUL && num_src > 1) {
        out[NOCREADBENCH_R_MODE] = NOCREADBENCH_MODE_REFUSED;
        out[NOCREADBENCH_R_MAGIC] = NOCREADBENCH_MAGIC;
        return;
    }

    Plan p;
    p.point = get_arg_val<uint32_t>(NOCREADBENCH_A_POINT);
    p.num_tx = get_arg_val<uint32_t>(NOCREADBENCH_A_NUM_TX);
    p.tx_bytes = get_arg_val<uint32_t>(NOCREADBENCH_A_TX_BYTES);
    p.dst_stride = get_arg_val<uint32_t>(NOCREADBENCH_A_DST_STRIDE);
    p.src_stride = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC_STRIDE);
    p.num_src = num_src;
    p.sample = get_arg_val<uint32_t>(NOCREADBENCH_A_SAMPLE);
    // Clamped for the same reason `num_src` is: it indexes a register file.
    p.trid = get_arg_val<uint32_t>(NOCREADBENCH_A_TRID) & 0xFu;
    p.src_kind = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC_KIND);

    // The arena is split in half: reads land in the top half, sources sit in
    // the bottom half of the *remote* core's identically-addressed arena. The
    // halves never overlap, so a landing never clobbers a source even in the
    // loopback case.
    //
    // A DRAM source is not in that arena at all: it is the DRAM tile's own
    // address, which the host passes explicitly. `dst_pad` shifts the LANDING
    // side by up to n-1 bytes so that `(src % n) == (dst % n)` holds with
    // n = 32 (Wormhole) / 64 (Blackhole); shifting the landing rather than the
    // source is what keeps every DRAM address allocator-aligned for the host's
    // own writes. Congruence, not absolute alignment -- the ISA docs' table
    // carries only congruence codes and an absolute rule would refuse correct
    // kernels.
    const uint32_t half = data_bytes / 2;
    const uint32_t src_base_arg = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC_BASE);
    const uint32_t src_span_arg = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC_SPAN);
    // Clamped, not trusted: the pad only ever needs to be under the largest
    // congruence modulus, and a pad that walked the landing base out of the
    // arena would corrupt whatever sits above it.
    const uint32_t dst_pad = get_arg_val<uint32_t>(NOCREADBENCH_A_DST_PAD) & 0x3Fu;
    p.dst_base = data_addr + half + dst_pad;
    p.src_base = (src_base_arg == 0) ? data_addr : src_base_arg;
    p.local_base = data_addr;
    // The strides wrap by compare-and-reset, never by `%`: a `remu` in the
    // issue loop would cost several cycles of the very quantity being
    // measured. Compare-and-reset is one branch, and the same one in every
    // experiment, so it cancels between points.
    const uint32_t land_room = (half > dst_pad) ? (half - dst_pad) : 0;
    p.dst_span = (land_room > p.tx_bytes) ? (land_room - p.tx_bytes) : 0;
    if (src_span_arg != 0) {
        p.src_span = (src_span_arg > p.tx_bytes) ? (src_span_arg - p.tx_bytes) : 0;
    } else {
        p.src_span = (half > p.tx_bytes) ? (half - p.tx_bytes) : 0;
    }
    // The witness payload lands in the initiator's own SOURCE half, which it
    // never reads into and never reads from -- this core is never a source. It
    // carries the same pad, so it stays congruent with whatever the burst read.
    p.probe_dst = data_addr + half - NOCREADBENCH_PROBE_BACKOFF + dst_pad;

    // Pin the transaction id before anything is issued, so both bursts carry
    // the same one and `outstanding(trid)` is known to be the right counter
    // rather than assumed to be. For trid 0 this writes the literal zero that
    // tt-metal's own `noc_clear_packet_tag` writes, so it disturbs no other
    // field of the tag.
    noc_async_read_set_trid(p.trid, noc_index);

    // Source NoC addresses, precomputed so that cycling sources costs an array
    // load rather than a `get_noc_addr` per iteration.
    for (uint32_t s = 0; s < num_src; s++) {
        const uint32_t sx = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC + 2 * s);
        const uint32_t sy = get_arg_val<uint32_t>(NOCREADBENCH_A_SRC + 2 * s + 1);
        p.src_noc[s] = get_noc_addr(sx, sy, p.src_base);
    }
    p.witness_noc = get_noc_addr(
        get_arg_val<uint32_t>(NOCREADBENCH_A_WITNESS_X),
        get_arg_val<uint32_t>(NOCREADBENCH_A_WITNESS_Y),
        p.local_base);

    if (mode == NOCREADBENCH_MODE_STATEFUL) {
        run_bursts<true>(p, out);
    } else {
        run_bursts<false>(p, out);
    }
}
