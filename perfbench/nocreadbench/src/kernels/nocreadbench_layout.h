// Shared between the host program (nocreadbench.cpp) and the reader kernel.
//
// Every result carries the initiator's own physical NoC coordinate, the
// subordinate coordinate it was told to use, and the full experiment key, so a
// row is identifiable without reference to the host's loop counters. That is
// the property tt-metal's shipped `noc_latencies.yaml` lacks and the reason
// `tt_sim/perf/noc_dataset_sweep.py` can only difference it along one axis.
#pragma once

#define NOCREADBENCH_MAGIC 0x4E524233u  // "NRB3"; bump on any layout change

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
// --- added with NRB3: the issue-loop variant, and the witness that proves it --
// `_MODE` is what the kernel ACTUALLY RAN, not what it was asked for; a kernel
// asked for the stateful loop under a plan it cannot express (more than one
// source tile, whose coordinate the stateful path holds in state) stamps
// `NOCREADBENCH_MODE_REFUSED` and measures nothing.
#define NOCREADBENCH_R_MODE 22
// The mode witness. It is a WORD OF PAYLOAD, read back over the NoC by the same
// API call the timed loop issued, and it is the reason a stateful run cannot be
// forged by passing a flag -- see the probe in reader.cpp.
#define NOCREADBENCH_R_PROBE 23
#define NOCREADBENCH_R_WORDS 28

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
// Which issue loop to time. STATELESS is the 2026-08-17 arm and must stay
// bit-for-bit the program that session ran; STATEFUL is `noc_async_read_set_state`
// once plus `noc_async_read_with_state` per transaction, which is what the
// `stateful` rows of tt-metal's shipped dataset time.
#define NOCREADBENCH_A_MODE 11
// A worker core that is NOT the initiator and NOT any source of this point. The
// kernel points the read state at it and then issues one transaction through the
// arm's own API call; which tile answers is the mode. See reader.cpp.
#define NOCREADBENCH_A_WITNESS_X 12
#define NOCREADBENCH_A_WITNESS_Y 13
#define NOCREADBENCH_A_SRC 14       // 2 * num_src words follow, (x, y) pairs

// ---------------------------------------------------------------------------
// Issue-loop variants.
// ---------------------------------------------------------------------------
#define NOCREADBENCH_MODE_STATELESS 0u
#define NOCREADBENCH_MODE_STATEFUL 1u
#define NOCREADBENCH_MODE_REFUSED 0xFFFFFFFFu

// ---------------------------------------------------------------------------
// The mode witness.
// ---------------------------------------------------------------------------
// The host stamps this signature into the first words of every participating
// core's source region, so a word of returned payload names the tile it came
// from. The stateless issue path writes NOC_TARG_ADDR_COORDINATE on every
// transaction and the stateful one never does (`ncrisc_noc_read_with_state`,
// wormhole/noc_nonblocking_api.h:1163; blackhole's at :1369), so a single
// transaction issued after the state has been pointed at a DIFFERENT tile comes
// back from the source under the stateless loop and from the witness under the
// stateful one. That is a property of the code that ran, not of the flag that
// asked for it, and it is the discipline `perfbench/nocevbench/check_arm.py`
// applies to the NoC ids in a trace.
#define NOCREADBENCH_SIG(x, y) (0x5A5A0000u | (((x) & 0xFFu) << 8) | ((y) & 0xFFu))
#define NOCREADBENCH_SIG_WORDS 4  // how many words of it the host stamps
#define NOCREADBENCH_PROBE_FILL 0xEEEEEEEEu  // pre-fill, so "nothing landed" shows
// Where the probe's payload lands: this many bytes below the top of the arena's
// SOURCE half, which the initiator never reads into and never reads from (it is
// only ever a source on the remote cores).
#define NOCREADBENCH_PROBE_BACKOFF 256u

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
