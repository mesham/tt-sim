// Shared between the host program (nocreadbench.cpp) and the reader kernel.
//
// Every result carries the initiator's own physical NoC coordinate, the
// subordinate coordinate it was told to use, and the full experiment key, so a
// row is identifiable without reference to the host's loop counters. That is
// the property tt-metal's shipped `noc_latencies.yaml` lacks and the reason
// `tt_sim/perf/noc_dataset_sweep.py` can only difference it along one axis.
#pragma once

#define NOCREADBENCH_MAGIC 0x4E524231u  // "NRB1"; bump on any layout change

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
#define NOCREADBENCH_R_OUTSTANDING_MAX 10  // max NIU_MST_REQS_OUTSTANDING_ID(0)
#define NOCREADBENCH_R_OUTSTANDING_END 11  // its value at the last sample
#define NOCREADBENCH_R_SAMPLES 12          // how many samples were taken
// Blackhole only: NOC_REGS_START_ADDR + 0x64, four 5-bit per-command-buffer
// availability fields. Read once at rest (before any request is issued) and
// once at the end of the untimed burst. Wormhole has no such register and the
// kernel reports 0xFFFFFFFF there. Neither the ISA docs nor tt-metal document
// its depth, which is exactly why it is worth reading rather than deriving.
#define NOCREADBENCH_R_CMDBUF_AVAIL_REST 13
#define NOCREADBENCH_R_CMDBUF_AVAIL_BUSY 14
#define NOCREADBENCH_R_WORDS 16

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
// Reserved for experiment E5 (the trid axis), which is NOT implemented here.
// It needs `noc_async_read_set_trid` plus the with-transaction-id read helper,
// whose signature has moved between tt-metal releases, and the E0 counter read
// answers the same question more directly. Kept in the layout so that adding
// it later does not renumber every argument. The host always passes 0.
#define NOCREADBENCH_A_NUM_TRID 9
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
#ifdef CMD_BUF_AVAIL
#define NOCREADBENCH_HAVE_CMD_BUF_AVAIL 1
#else
#define NOCREADBENCH_HAVE_CMD_BUF_AVAIL 0
#endif
