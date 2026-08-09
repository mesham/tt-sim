// Shared between the host program (nocreadbench.cpp) and the reader kernel.
//
// Every result carries the initiator's own physical NoC coordinate, the
// subordinate coordinate it was told to use, and the full experiment key, so a
// row is identifiable without reference to the host's loop counters. That is
// the property tt-metal's shipped `noc_latencies.yaml` lacks and the reason
// `tt_sim/perf/noc_dataset_sweep.py` can only difference it along one axis.
#pragma once

#define NOCREADBENCH_MAGIC 0x4E524232u  // "NRB2"; bump on any layout change

// ---------------------------------------------------------------------------
// Result word indices, per participating core.
// ---------------------------------------------------------------------------
#define NOCREADBENCH_R_MAGIC 0
#define NOCREADBENCH_R_POINT 1        // index of this point within the run
#define NOCREADBENCH_R_T0 2           // wall clock entering the timed region
#define NOCREADBENCH_R_T1 3           // wall clock leaving it
#define NOCREADBENCH_R_CYCLES 4       // t1 - t0, computed on device (wrap-safe)
#define NOCREADBENCH_R_NODE_X 5       // from NOC_NODE_ID, the physical coord
#define NOCREADBENCH_R_NODE_Y 6
#define NOCREADBENCH_R_NUM_TX 7
#define NOCREADBENCH_R_TX_BYTES 8
#define NOCREADBENCH_R_NOC 9
// The direct observable. Sampled OUTSIDE the timed region, in a second,
// untimed burst, because the sampling load is itself a >= 7-cycle NIU load and
// would perturb the rate it is trying to explain.
//
// EVERY ONE OF THESE IS A DIFFERENCE, NOT AN ABSOLUTE. The 2026-08-09 Blackhole
// card session recorded outstanding_max = 72 in all 129 rows -- at num_tx = 4 as
// well as 128 -- because it reported the counter's RAW value and the counter
// does not sit at zero when the kernel starts. A count of in-flight requests
// cannot read 72 during a four-request burst, so the raw number said nothing.
// `_REST` is the value before the kernel issues anything; the occupancy is
// (`_MAX` - `_REST`), and if that is zero the control did not move.
#define NOCREADBENCH_R_OUTSTANDING_MAX 10  // max NIU_MST_REQS_OUTSTANDING_ID(trid)
#define NOCREADBENCH_R_OUTSTANDING_END 11  // its value at the last sample
#define NOCREADBENCH_R_SAMPLES 12          // how many samples were taken
// Blackhole only: NOC_REGS_START_ADDR + 0x64, four 5-bit per-command-buffer
// fields, read through the NoC instance offset so `noc_index` 1 reads NoC 1's
// block and not NoC 0's. Wormhole has no such register and the kernel reports
// 0xFFFFFFFF there. `_BUSY` is now sampled INSIDE the issue loop -- the card
// session sampled it only after the loop had drained, which is why rest and
// busy were byte-identical zeros in all 129 rows.
#define NOCREADBENCH_R_CMDBUF_AVAIL_REST 13
#define NOCREADBENCH_R_CMDBUF_AVAIL_BUSY 14  // last in-loop sample
// --- added with NRB2, all of them controls the NRB1 layout could not express -
#define NOCREADBENCH_R_OUTSTANDING_REST 15  // the counter BEFORE anything is issued
// A second, transaction-id-independent measure of the same quantity:
// NIU_MST_RD_REQ_SENT - NIU_MST_RD_RESP_RECEIVED. Both are cumulative
// one-per-read counters (tt-metal's own `ncrisc_noc_reads_flushed` compares
// RD_RESP_RECEIVED against a software count of `noc_async_read` calls), so
// their difference is the number of read requests in flight whatever
// transaction id the requests carry. `_REST` must read 0 after a barrier; if it
// does not, neither instrument is trustworthy.
#define NOCREADBENCH_R_INFLIGHT_MAX 16
#define NOCREADBENCH_R_INFLIGHT_REST 17
#define NOCREADBENCH_R_CMDBUF_AVAIL_MAX 18  // max over the in-loop samples
// CMD_BUF_OVFL, the register immediately after CMD_BUF_AVAIL. If the command
// buffer was ever driven past its depth this moves, and a max occupancy that
// arrives WITH an overflow is the depth rather than a lower bound on it.
#define NOCREADBENCH_R_CMDBUF_OVFL_REST 19
#define NOCREADBENCH_R_CMDBUF_OVFL_END 20
#define NOCREADBENCH_R_TRID 21  // which transaction id the kernel pinned and sampled
#define NOCREADBENCH_R_WORDS 24

// ---------------------------------------------------------------------------
// Runtime argument indices for kernels/dataflow/reader.cpp.
// ---------------------------------------------------------------------------
#define NOCREADBENCH_A_RESULTS 0
#define NOCREADBENCH_A_DATA 1       // base of the local payload arena
#define NOCREADBENCH_A_DATA_BYTES 2 // how much of it the kernel may stride over
#define NOCREADBENCH_A_POINT 3
#define NOCREADBENCH_A_NUM_TX 4
#define NOCREADBENCH_A_TX_BYTES 5
#define NOCREADBENCH_A_DST_STRIDE 6 // bytes between successive read landings
#define NOCREADBENCH_A_SRC_STRIDE 7 // bytes between successive read sources
#define NOCREADBENCH_A_NUM_SRC 8    // how many distinct source tiles to cycle
// Which transaction id the kernel pins with `noc_async_read_set_trid` and then
// samples. Plain `noc_async_read` does NOT write NOC_PACKET_TAG on either
// Wormhole or Blackhole (see `ncrisc_noc_fast_read`), so without this the
// transaction id a read carries is whatever the read command buffer's sticky
// NOC_PACKET_TAG last held -- and sampling counter 0 was an assumption, not a
// fact. The host always passes 0, which makes the write a literal zero
// (NOC_PACKET_TAG_TRANSACTION_ID(0) == 0, the same value tt-metal's own
// `noc_clear_packet_tag` writes), so pinning it changes no other tag field.
#define NOCREADBENCH_A_TRID 9
#define NOCREADBENCH_A_SAMPLE 10  // 1 = also run the untimed sampling burst
#define NOCREADBENCH_A_SRC 11       // 2 * num_src words follow, (x, y) pairs

// ---------------------------------------------------------------------------
// Registers the kernel reads. Both spelled out rather than relying on a
// force-include, exactly as nocbench does for the wall clock.
// ---------------------------------------------------------------------------
#ifdef RISCV_DEBUG_REG_WALL_CLOCK_L
#define NOCREADBENCH_WALL_CLOCK_L RISCV_DEBUG_REG_WALL_CLOCK_L
#else
#define NOCREADBENCH_WALL_CLOCK_L 0xFFB121F0u
#endif

// `CMD_BUF_AVAIL` is defined only by Blackhole's noc_parameters.h. Its four
// 5-bit fields are at bits [4:0], [12:8], [20:16], [28:24], one per command
// buffer.
//
// NOTHING in tt-metal reads it on either Wormhole or Blackhole -- the only
// register descriptor for it in the tree is Quasar's, whose reset default is
// `NOC_NIU_CMD_BUF_AVAIL_REG_DEFAULT (0x00000000)` and whose neighbour is
// `CMD_BUF_OVFL`. A fill level paired with an overflow flag and resetting to
// zero is an OCCUPANCY, not a count of free slots, so reading zero at rest is
// the CORRECT answer and says nothing at all about the depth. Only the maximum
// occupancy seen while a command buffer actually holds an entry is informative,
// and even that is a LOWER BOUND on the depth unless CMD_BUF_OVFL also moves.
#ifdef CMD_BUF_AVAIL
#define NOCREADBENCH_HAVE_CMD_BUF_AVAIL 1
#else
#define NOCREADBENCH_HAVE_CMD_BUF_AVAIL 0
#endif

#ifdef CMD_BUF_OVFL
#define NOCREADBENCH_HAVE_CMD_BUF_OVFL 1
#else
#define NOCREADBENCH_HAVE_CMD_BUF_OVFL 0
#endif
