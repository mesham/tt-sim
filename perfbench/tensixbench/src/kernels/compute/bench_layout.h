// Shared between the host program (tensixbench.cpp) and both compute kernels.
// The kernels stamp the header words into the result buffer so the host can
// validate that the binary it built and the kernel that ran agree, rather than
// trusting a constant duplicated in two places.
#pragma once

#define TTBENCH_MAGIC 0x7B10CE02u  // bump on any layout change

// Header word indices.
#define TTBENCH_HDR_MAGIC 0
#define TTBENCH_HDR_UNROLL 1
#define TTBENCH_HDR_NUM_PROBES 2
#define TTBENCH_HDR_NUM_POINTS 3
#define TTBENCH_HDR_BASE_BLOCKS 4
#define TTBENCH_HDR_ACTIVE_MASK 5
#define TTBENCH_HDR_PROBE_MASK 6
#define TTBENCH_HDR_WORDS 8

// Instructions per unrolled block. The slope is taken over `blocks`, so this is
// the divisor that turns a per-block slope into a per-instruction cost.
#define TTBENCH_UNROLL 64

// Number of (blocks, cycles) points per probe. blocks = base_blocks * (p + 1).
#define TTBENCH_NUM_POINTS 4

#define TTBENCH_NUM_PROBES 20
#define TTBENCH_MAX_THREADS 3

#define TTBENCH_RESULT_WORDS \
    (TTBENCH_HDR_WORDS + TTBENCH_MAX_THREADS * TTBENCH_NUM_PROBES * TTBENCH_NUM_POINTS)

// Phase B (matmul_tiles at a fixed math fidelity) reuses the same buffer with
// probe slot 0 only.
#define TTBENCH_MM_NUM_POINTS TTBENCH_NUM_POINTS
