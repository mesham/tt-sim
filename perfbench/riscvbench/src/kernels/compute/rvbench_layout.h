// Shared between the host program (riscvbench.cpp) and the compute kernel
// (rv_probes.cpp). The kernel stamps the header words into the result buffer so
// the host can validate that the binary it built and the kernel that ran agree,
// rather than trusting a constant duplicated in two places. Same contract, and
// the same reasoning, as perfbench/tensixbench/src/kernels/compute/bench_layout.h.
#pragma once

#define RVBENCH_MAGIC 0x7B10CF01u  // bump on any layout change

// Header word indices.
#define RVBENCH_HDR_MAGIC 0
#define RVBENCH_HDR_NUM_PROBES 1
#define RVBENCH_HDR_MAX_POINTS 2
#define RVBENCH_HDR_BASE_BLOCKS 3
#define RVBENCH_HDR_ACTIVE_MASK 4
#define RVBENCH_HDR_PROBE_MASK 5
#define RVBENCH_HDR_PHASE 6
// The address of a stack local, reported so the analysis can classify which
// row of the ISA docs' load-latency table the `rv_load_stack` probe measured
// instead of assuming one. tt-metal is free to place a TRISC stack in local
// data RAM or in L1 and the two rows differ by 6 cycles on Wormhole.
#define RVBENCH_HDR_STACK_ADDR 7
#define RVBENCH_HDR_SCRATCH_ADDR 8
// The probe mask is 64 bits wide because there are more than 32 probes, and it
// travels as two runtime-arg words. Slot 32 onwards is phase F, whose bodies
// are the ones a `--probes` mask most wants to disable individually.
#define RVBENCH_HDR_PROBE_MASK_HI 9
#define RVBENCH_HDR_WORDS 12

// Instructions per unrolled block for the straight-line probes. The slope is
// taken over `blocks`, so this is the divisor that turns a per-block slope into
// a per-instruction cost. Each probe records its OWN divisor in the CSV's
// `unroll` column, because the group probes below do not use this one.
#define RVBENCH_UNROLL 64

// Phase T's group probes. A "group" is RVBENCH_PAD plain RV32I instructions
// with zero, two or four `.ttinsn` words placed among them, and a block is
// RVBENCH_GROUP_REPS groups. The reported quantity is cycles per GROUP, so the
// marginal cost of the `.ttinsn` words is a subtraction between probes rather
// than a division. See the header comment of rv_probes.cpp.
#define RVBENCH_GROUP_REPS 16
#define RVBENCH_PAD 16

// Points per probe. Phases R/T/C/F use the first RVBENCH_SLOPE_POINTS of them
// (n = base_blocks * (k + 1)); phase Q uses all eight, at n = 1, 2, 4, ... 128,
// because it is looking for a KNEE rather than fitting a slope.
#define RVBENCH_MAX_POINTS 8
#define RVBENCH_SLOPE_POINTS 4

#define RVBENCH_MAX_THREADS 3

// ---------------------------------------------------------------------------
// Probe slots. The order here is the contract with the host's table in
// riscvbench.cpp; keep the two in the same order. Slots are grouped by phase
// and a phase's bodies are only COMPILED when that phase is selected (the host
// passes -DRVBENCH_PHASE_<X> through ComputeConfig::defines), so the phase F
// footprints -- 16 KiB of instruction text on their own -- never inflate the
// kernel of a run that is not measuring instruction fetch.
// ---------------------------------------------------------------------------

// Phase R -- straight-line RV32IM, the baseline the other phases are read
// against.
#define RVBENCH_P_LOOP_OVERHEAD 0
#define RVBENCH_P_ADDI_INDEP 1
#define RVBENCH_P_ADDI_DEP 2
#define RVBENCH_P_MUL_INDEP 3
#define RVBENCH_P_MUL_DEP 4
#define RVBENCH_P_DIV 5
#define RVBENCH_P_LOAD_CHASE 6
#define RVBENCH_P_LOAD_INDEP 7
#define RVBENCH_P_STORE_SPREAD 8
#define RVBENCH_P_STORE_COALESCE 9
#define RVBENCH_P_LOAD_STACK 10
#define RVBENCH_P_STORE_STACK 11

// Phase T -- the `.ttinsn` issue path.
#define RVBENCH_P_TT_NOP 12
#define RVBENCH_P_TT_SFPNOP 13
#define RVBENCH_P_TT_SETDMAREG 14
#define RVBENCH_P_TT_ADDDMAREG 15
#define RVBENCH_P_TT_PAD 16
#define RVBENCH_P_TT_FUSE2 17
#define RVBENCH_P_TT_FUSE4 18
#define RVBENCH_P_TT_SPREAD4 19

// Phase C -- control flow.
#define RVBENCH_P_C_CTRL_XOR 20
#define RVBENCH_P_C_NT 21
#define RVBENCH_P_C_T 22
#define RVBENCH_P_C_XOR_NT 23
#define RVBENCH_P_C_XOR_T 24
#define RVBENCH_P_C_XOR_ALT 25
#define RVBENCH_P_C_JAL 26

// Phase Q -- how deep the Tensix instruction queue is before it back-pressures
// the issuing RISC-V core. Eight points, one per burst length.
#define RVBENCH_P_Q_CTRL 27
#define RVBENCH_P_Q_NOP 28
#define RVBENCH_P_Q_SETDMAREG 29
#define RVBENCH_P_Q_ADDDMAREG 30
#define RVBENCH_P_Q_ADDDMAREG_SYNC 31

// Phase F -- instruction footprint. The unroll of probe `f_K` is K.
#define RVBENCH_P_F_64 32
#define RVBENCH_P_F_128 33
#define RVBENCH_P_F_256 34
#define RVBENCH_P_F_512 35
#define RVBENCH_P_F_1024 36
#define RVBENCH_P_F_2048 37

#define RVBENCH_NUM_PROBES 38

#define RVBENCH_RESULT_WORDS \
    (RVBENCH_HDR_WORDS + RVBENCH_MAX_THREADS * RVBENCH_NUM_PROBES * RVBENCH_MAX_POINTS)

// Scratch the load/store probes address. Must be at least
// RVBENCH_UNROLL * 16 bytes so that `store_spread` can hit 64 distinct
// 16-byte-aligned blocks, which is what makes it NON-coalescable on Blackhole
// (the docs' predicate is "the same 16-byte aligned region of L1, with start
// addresses within +/-4 of each other").
#define RVBENCH_SCRATCH_BYTES 2048

// Phase codes, one bit each, mirrored into the CSV header as `phase=`.
#define RVBENCH_PHASE_R_BIT 1
#define RVBENCH_PHASE_T_BIT 2
#define RVBENCH_PHASE_C_BIT 4
#define RVBENCH_PHASE_Q_BIT 8
#define RVBENCH_PHASE_F_BIT 16
