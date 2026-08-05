// Shared between the host program (nocbench.cpp) and the flow kernel.
//
// The kernel stamps the magic and its own NoC coordinates into the result
// buffer so the host can prove the kernel ran, that host and kernel agree on
// the layout, and -- the point of the whole exercise -- that the core the
// measurement came from is the core the plan asked for. A measurement whose
// coordinates are not recorded is exactly the thing that makes tt-metal's
// shipped NoC dataset unidentifiable; see
// tt_sim/perf/noc_dataset_sweep.py's CONGESTION_VERDICT.
#pragma once

#define NOCBENCH_MAGIC 0x4E4F4342u  // "NOCB"; bump on any layout change

// Result word indices, per participating core.
#define NOCBENCH_R_MAGIC 0
#define NOCBENCH_R_FLOW 1   // flow index within the run
#define NOCBENCH_R_T0 2     // wall clock at the start of the timed region
#define NOCBENCH_R_T1 3     // wall clock at the end of it
#define NOCBENCH_R_CYCLES 4 // t1 - t0, computed on device (wrap-safe)
#define NOCBENCH_R_NOC_X 5  // this core's own NoC x, read from my_x[noc_index]
#define NOCBENCH_R_NOC_Y 6
#define NOCBENCH_R_NUM_TX 7
#define NOCBENCH_R_TX_BYTES 8
#define NOCBENCH_R_DIRECTION 9
#define NOCBENCH_R_VC 10
#define NOCBENCH_R_NOC 11        // noc_index the kernel actually ran on
#define NOCBENCH_R_REND_CYCLES 12 // cycles spent in the rendezvous wait
// The core's own coordinate straight out of its NIU's NOC_NODE_ID register.
// `my_x[]`/`my_y[]` are NOT used for this: the firmware fills those from
// NOC_CFG(NOC_ID_LOGICAL), which is the *translated* coordinate on Blackhole
// and which tt-sim answers 0 for. NOC_NODE_ID is the physical NoC coordinate,
// which is the space every link and hop count in the plan is computed in --
// so this is the number worth checking the plan against.
#define NOCBENCH_R_NODE_X 13
#define NOCBENCH_R_NODE_Y 14
#define NOCBENCH_R_WORDS 16

// Direction codes (runtime arg 9).
#define NOCBENCH_DIR_WRITE 0
#define NOCBENCH_DIR_READ 1
#define NOCBENCH_DIR_BIDIR 2

// Runtime argument indices for kernels/dataflow/flow.cpp.
#define NOCBENCH_A_RESULTS 0
#define NOCBENCH_A_DATA 1
#define NOCBENCH_A_SEM 2
#define NOCBENCH_A_NUM_PEERS 3
#define NOCBENCH_A_FLOW 4
#define NOCBENCH_A_SUB_X 5
#define NOCBENCH_A_SUB_Y 6
#define NOCBENCH_A_NUM_TX 7
#define NOCBENCH_A_TX_BYTES 8
#define NOCBENCH_A_DIRECTION 9
#define NOCBENCH_A_VC 10
#define NOCBENCH_A_PEERS 11  // 2 * num_peers words follow, (x, y) pairs

// RISCV_DEBUG_REG_WALL_CLOCK_L. The macro is defined by the tt-1xx/tt-2xx
// tensix.h that the compute-kernel force-include chain pulls in; a data
// movement kernel does not necessarily see it, so fall back to the literal
// (RISCV_DEBUG_REGS_START_ADDR 0xFFB12000 | 0x1F0) that both architectures'
// headers spell out and that tt_sim/misc/tile_ctrl.py answers for.
#ifdef RISCV_DEBUG_REG_WALL_CLOCK_L
#define NOCBENCH_WALL_CLOCK_L RISCV_DEBUG_REG_WALL_CLOCK_L
#else
#define NOCBENCH_WALL_CLOCK_L 0xFFB121F0u
#endif
