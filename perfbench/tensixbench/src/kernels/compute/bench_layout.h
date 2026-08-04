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

// Phase B blocking factor: how many tile pairs are waited for, multiplied and
// popped as one group. It exists to get the circular-buffer bookkeeping OUT of
// the per-matmul cost. One `cb_wait_front`/`cb_pop_front` pair per operand now
// covers TTBENCH_MM_BLOCK matmuls instead of one, so the RISC-V-side cost per
// `matmul_tiles` falls by roughly that factor and the Tensix side (16 x phases
// MVMULs, expanded by the MOP) gets a chance to become the limit. See the
// header comment of kernels/compute/matmul_fidelity.cpp.
//
// Both circular buffers must hold at least this many pages; the host allocates
// 2x so the feeder can run ahead by a whole block.
#define TTBENCH_MM_BLOCK 8

// Phase B times threads 0 (unpack) and 1 (math). Thread 2 (pack) has no
// per-iteration work in a matmul inner loop -- `cb_wait_front`, `cb_pop_front`
// and `matmul_tiles` are all UNPACK/MATH-only -- so its timed region is empty
// and reads ~1 cycle whatever the iteration count. It is not measured.
#define TTBENCH_MM_TIMED_THREADS 2
