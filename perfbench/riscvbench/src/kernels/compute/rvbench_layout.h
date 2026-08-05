// Shared between the host program (riscvbench.cpp) and the compute kernel
// (rv_probes.cpp). The kernel stamps the header words into the result buffer so
// the host can validate that the binary it built and the kernel that ran agree,
// rather than trusting a constant duplicated in two places. Same contract, and
// the same reasoning, as perfbench/tensixbench/src/kernels/compute/bench_layout.h.
#pragma once

#define RVBENCH_MAGIC 0x7B10CF03u  // bump on any layout change

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
// (n = base_blocks * (k + 1)); phase Q's CASCADE probes use all eight, at
// n = 1, 2, 4, ... 128, because they are looking for a KNEE rather than fitting
// a slope; phase Q's LOOP probes use the first RVBENCH_Q_LOOP_POINTS, at
// n = 16, 32, ... 1024; phase S's use all eight, at n = 4, 8, ... 512. Every
// probe's own point count and n-mapping travel in the host's probe table, so
// this is only the width of the result buffer.
#define RVBENCH_MAX_POINTS 8
#define RVBENCH_SLOPE_POINTS 4

// ---------------------------------------------------------------------------
// Phase Q's LOOP burst form, and why phase Q has two burst forms at all.
//
// The cascade emits 2^p copies of the body STRAIGHT-LINE, so its instruction
// stream grows with the burst: at n = 1024 it would be 4 KiB of `.ttinsn`
// words. `perfbench/riscvbench` phase F located an instruction-fetch cliff in
// exactly that octave -- flat at 0.998 cycles/instruction through a 4 KiB loop
// body and 1.251 at 8 KiB -- and every phase-Q burst runs exactly ONCE, cold,
// with its own instruction fetch inside the timed region. Extending the cascade
// to 1024 would therefore have folded a fetch cost that GROWS WITH n into the
// one measurement whose whole question is whether the cost per instruction
// grows with n. The answer would have looked like a queue knee and would have
// been instruction fetch. That is the confound phase F exists to have found.
//
// So the extension to n = 1024 is a SECOND FORM rather than a longer cascade:
// n / RVBENCH_Q_LOOP_BLOCK iterations of one unrolled block of
// RVBENCH_Q_LOOP_BLOCK instructions. The instruction footprint is then
// RVBENCH_Q_LOOP_BLOCK * 4 = 64 bytes at EVERY burst length, so no difference
// between two loop points can be a fetch effect -- the confound is removed by
// construction rather than corrected for by a subtraction. What the loop form
// costs instead is its own back-edge: the counter, compare and branch push the
// issue-limited rate above 1.0 cycles per instruction by about
// (loop overhead)/RVBENCH_Q_LOOP_BLOCK. That is not assumed either -- the
// `q_loop_addi` probe is the identical loop with `addi` bodies and measures it.
//
// The two forms overlap at n = 16, 32, 64, 128, where the cascade still runs.
// A loop reading that agrees with the cascade reading there is the evidence
// that the form change did not change the quantity; a disagreement is the
// evidence that it did, and either way it is measured rather than argued.
//
// RVBENCH_Q_LOOP_POINTS must be <= RVBENCH_MAX_POINTS -- the loop probes live
// in the same fixed-width result slots as everything else.
// ---------------------------------------------------------------------------
#define RVBENCH_Q_LOOP_BLOCK 16
#define RVBENCH_Q_LOOP_MIN_N 16
#define RVBENCH_Q_LOOP_POINTS 7  // n = 16, 32, 64, 128, 256, 512, 1024

// ---------------------------------------------------------------------------
// Phase S's burst form, and why phase Q's could not answer phase S's question.
//
// PHASE S ASKS ONE THING: is the ~14-16-entry Tensix instruction queue phase Q
// measured SHARED between the three TRISCs or PRIVATE to each? Phase Q cannot
// answer it, and neither could a longer or a repeated phase-Q run: a shared
// queue of depth D and a per-thread queue of depth D are the same device seen
// from one thread, and phase Q's multi-thread slots read zero.
//
// WHY THEY READ ZERO, which is what this form fixes. Phase Q's backlog is
// `(sync[n] - plain[n]) - (sync[16] - plain[16])`: the value at the smallest
// burst it runs is subtracted off as `tensix_sync()`'s own cost. That is only
// true if the queue is EMPTY at the end of an n = 16 burst. It is not. A core
// pushing at 1/p instructions per cycle against a backend draining one every S
// leaves `n * (1 - p/S)` instructions outstanding, which at n = 16 is ~10 at
// one issuing thread -- and at two or three, where the queue's per-thread share
// may be smaller than that, n = 16 is already SATURATED and the subtraction
// removes the entire signal. The multi-thread slots are structurally zero, not
// unluckily small. So phase S's reference burst has to be short enough that the
// queue is not full at it in ANY hypothesis, which means shorter than n = 16
// and therefore a smaller loop block than phase Q's.
//
//   RVBENCH_S_LOOP_BLOCK   4 instructions -- 16 bytes of text at every burst
//                          length, so nothing here can be instruction fetch,
//                          and n can start at 4.
//   RVBENCH_S_MIN_N        4, the reference burst. Its own occupancy is ~2
//                          entries at one thread, so what the reference
//                          subtracts off is small and MEASURED rather than
//                          assumed to be zero.
//   RVBENCH_S_POINTS       8: n = 4, 8, 16 ... 512.
//
// WHAT SEPARATES THE TWO HYPOTHESES, which is the design decision here and is
// argued at length in docs/plans/riscv-front-end-benchmark.md:
//
//   * A SPINNING second thread does not. It pushes nothing, so it holds no
//     queue entry under either hypothesis and the issuing thread sees a
//     full-depth queue either way. It is kept anyway, as the CONTROL that
//     separates "another core exists" from "another core issues".
//   * A second thread issuing at a deliberately LOW rate does not either.
//     Occupancy is arrival rate times residence time, so a thread that arrives
//     slower than it is served holds ~0 entries however long it runs.
//   * Only a SATURATED second thread holds entries, and the price of
//     saturation is that it also takes backend bandwidth. That is not a
//     confound here because the drained rate S is measured in the same slot,
//     by `s_co_sync`, and divided back out.
//
// The depth in entries is then
//
//     D = backlog_settled / S + RVBENCH_S_MIN_N * (1 - p/S)
//
// with p from `s_loop_addi` and S from `s_co_sync`/`s_solo_sync`. Equal D at
// one and at k issuing threads means the queue is PER-THREAD; D/k means it is
// SHARED. The two differ by a factor of k, which is why this is worth running.
//
// RVBENCH_S_ISSUER is thread 1 (TRISC_MATH), because that is the thread the
// `t1` variant activates, so the solo probes measure the same core in every
// variant and the only thing that changes is what the OTHER cores are doing.
// ---------------------------------------------------------------------------
#define RVBENCH_S_LOOP_BLOCK 4
#define RVBENCH_S_MIN_N 4
#define RVBENCH_S_POINTS 8  // n = 4, 8, 16, 32, 64, 128, 256, 512
#define RVBENCH_S_ISSUER 1
// How much longer than the issuer's burst a spinning thread keeps spinning.
// The issuer's ADDDMAREG burst costs ~3 cycles an instruction against the
// spinner's ~1.5, so 4x is comfortably past the end of it and the spinner is
// still awake when the issuer's clock is read.
#define RVBENCH_S_SPIN_MULT 4

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
// the issuing RISC-V core. Eight points, one per burst length, n = 1 ... 128.
// These are the CASCADE form; the loop form is at slots 38-40, deliberately not
// adjacent (see there).
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

// Phase Q, LOOP form -- the extension to n = 1024. Seven points, n = 16 ... 1024
// of a fixed 64-byte loop body, so the burst grows without the instruction
// stream growing. See the RVBENCH_Q_LOOP_* block above for why the cascade was
// not simply extended.
//
// APPENDED rather than inserted next to slots 27-31, because a probe's slot
// number is the `probe_id` column of every CSV already collected and the `bit i`
// of every `--probes` mask in the runbook. Keeping the existing 38 slots where
// they are costs one comment; renumbering them would silently re-label two
// tracked silicon datasets.
#define RVBENCH_P_Q_LOOP_ADDI 38
#define RVBENCH_P_Q_LOOP_ADDDMAREG 39
#define RVBENCH_P_Q_LOOP_ADDDMAREG_SYNC 40

// Phase S -- is that queue shared between the TRISCs or private to each? Six
// probes, all in the RVBENCH_S_* loop form above, n = 4 ... 512.
//
//   s_loop_addi    the issue-limited control: the identical loop with `addi`
//                  bodies, so the form's own back edge is measured rather than
//                  assumed. It is `p` in the depth formula.
//   s_co_plain     every ACTIVE thread issues the burst. At t1 that is one
//                  thread and at t2/t3 the queue is contended by construction.
//   s_co_repeat    a byte-identical second execution of `s_co_plain`. It is
//                  this phase's NOISE FLOOR: |co_plain - co_repeat| at each n
//                  is what one raw single-shot point can be wrong by, measured
//                  in the run rather than carried over from phase Q's cascade.
//   s_co_sync      `s_co_plain` with `ckernel::tensix_sync()` inside the timed
//                  region, so it cannot return until the pipe has drained. It
//                  supplies the drained rate S by measurement.
//   s_solo_plain   ONLY thread RVBENCH_S_ISSUER issues; the others spin on an
//                  `addi` loop. The control for "another core is awake" as
//                  distinct from "another core is issuing" -- a spinner holds
//                  no queue entry under either hypothesis, so this must read
//                  the same depth at t1, t2 and t3 whatever the answer is.
//   s_solo_sync    the drained pair for it.
//
// Only the issuer's rows are emitted for the two solo probes; the other
// threads have no measurement to report, and a row of zeroes would fail the
// monotonicity gate for saying nothing.
#define RVBENCH_P_S_LOOP_ADDI 41
#define RVBENCH_P_S_CO_PLAIN 42
#define RVBENCH_P_S_CO_REPEAT 43
#define RVBENCH_P_S_CO_SYNC 44
#define RVBENCH_P_S_SOLO_PLAIN 45
#define RVBENCH_P_S_SOLO_SYNC 46

// Phase G -- the intermediate footprints, which narrow the boundary
// docs/bh_arch.md 1.1 located between a 4 KiB and an 8 KiB loop body.
//
// WHY THIS IS A PHASE OF ITS OWN AND NOT THREE MORE PHASE F PROBES, which is a
// platform limit and not a design preference. tt-metal places a program's
// kernels in a fixed L1 config buffer -- 70656 bytes on this release -- and a
// compute kernel is built three times, once per TRISC, so a phase's probe
// bodies cost four bytes an instruction three times over. Phase F's six bodies
// are already 4032 instructions and its build lands within a few hundred bytes
// of that ceiling; adding 1280 + 1536 + 1792 to it produced
//
//     TT_FATAL: Program size (125040) too large for kernel config buffer (70656)
//
// measured, not predicted. So the intermediates get their own phase (whose
// build carries none of phase F's bodies) AND that phase is split into
// COMPILE-TIME SETS, because even 4608 instructions of intermediates alone do
// not fit. `--gset N` picks one:
//
//     set 0   g_1024 + g_1280
//     set 1   g_1024 + g_1536
//     set 2   g_1024 + g_1792
//
// Every set carries `g_1024`, so each intermediate is read against a
// 1024-instruction body compiled into the SAME kernel rather than against
// phase F's f_1024 from another build. Three runs of a two-probe phase are
// seconds each on a card.
//
// A SIDE EFFECT WORTH STATING: phase F's own text is untouched by all of this
// -- `--phase f` compiles exactly the six bodies it compiled before -- so the
// two tracked silicon datasets' phase F rows remain reproducible against this
// binary.
//
// And what a narrowed boundary is: a boundary in LOOP-BODY SIZE. Narrowing it
// does not turn it into a cache capacity. No document gives a cache size or a
// miss cost, and a step in cost against footprint is equally consistent with a
// prefetch window, a TLB-like structure or an L1 access pattern.
#define RVBENCH_P_G_1024 47
#define RVBENCH_P_G_1280 48
#define RVBENCH_P_G_1536 49
#define RVBENCH_P_G_1792 50

#define RVBENCH_NUM_PROBES 51

// How many compile-time footprint sets phase G has, and which intermediate
// each carries. The host validates `--gset` against this.
#define RVBENCH_G_SETS 3

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
#define RVBENCH_PHASE_S_BIT 32
#define RVBENCH_PHASE_G_BIT 64
