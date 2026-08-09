// riscvbench -- the baby RISC-V front end, measured from a kernel.
//
// WHY THIS EXISTS. `docs/plans/cost-model.md` now wires six of the nine Tensix
// backend units, and every single one of them moved ZERO simulated cycles. The
// reason was the same every time: nothing back-pressures the baby RISC-V, so
// unit occupancy is invisible to anything a kernel can time. Silicon said the
// same thing from the other direction -- `perfbench/tensixbench` phase A reads
// exactly 1.000 cycles per instruction against tt-sim for every probe of every
// unit at every data format, because `TensixFrontend.push_mop_instruction` is
// an unbounded list append and the `.ttinsn` write returns immediately. **The
// RISC-V front end is the term that makes every other term unobservable.**
// This kernel measures it.
//
// THE METHOD, in one paragraph, and it is `tensixbench`'s unchanged. For each
// probe the kernel runs the SAME unrolled block `blocks` times, for four values
// of `blocks`, and timestamps each run off the device's own wall clock. The
// measured time is
//
//     cycles(blocks) = blocks * (U * c_op + L) + K
//
// where `c_op` is the per-instruction cost wanted, `U` is the probe's unroll,
// `L` is the RISC-V loop overhead (counter, compare, branch) and `K` is
// everything fixed: the two clock reads, the barrier, the surrounding call. A
// least-squares slope over the four points kills `K` exactly. Probe 0 runs the
// identical loop with an EMPTY body, so its slope is `L` alone, and
//
//     c_op = (slope(probe) - slope(probe 0)) / U
//
// kills `L` exactly too. No absolute measurement is ever reported as a cost.
//
// TIMESTAMPS come from RISCV_DEBUG_REG_WALL_CLOCK_L, exactly the register
// tt-metal's device profiler reads for DeviceZoneScopedN. Reading it directly
// keeps the measurement identical on silicon and against tt-sim, needs no Tracy
// build, and sidesteps the profiler's dependence on a device AICLK a simulator
// does not have.
//
// ---------------------------------------------------------------------------
// THE SEVEN PHASES, and what each is for. Each is compiled only when selected
// (the host passes -DRVBENCH_PHASE_<X> through ComputeConfig::defines), so a
// run measuring instruction fetch does not carry phase R's text and vice versa.
// That matters for phase F in particular, whose bodies are 16 KiB of
// instructions on their own and would otherwise BE the thing they measure.
// ---------------------------------------------------------------------------
//
// PHASE R -- straight-line RV32IM. The baseline. Without it a phase T number is
// uninterpretable: "a `.ttinsn` costs 1 cycle" only means something next to
// "an `addi` costs 1 cycle". Also the one phase whose predictions the tables
// mostly make outright -- `riscv.integer_unit`, `riscv.load_latency` and
// `riscv.store_throughput` in tt_sim/perf/unit_costs.yaml are all `isa_doc`,
// and `tt_sim/pe/rv/cost.py` already CONSUMES three of them. So phase R is
// simultaneously the baseline and the instrument's own calibration: `mul`,
// `div`, the L1 load chase and the L1 store rate are the four probes tt-sim
// moves, which is what makes a null result elsewhere distinguishable from an
// un-instrumented one.
//
// PHASE T -- the `.ttinsn` issue path, and the reason this benchmark exists.
// Four straight bursts (one op per issue slot, at four different backend units)
// establish the sustained rate. The interesting half is the GROUP probes, which
// exist to test one specific documented claim that nothing has ever checked:
//
//     "sequences of up to four adjacent .ttinsn instructions can be fused
//      together by the RISCV T0 / T1 / T2 instruction caches and executed in a
//      single cycle (this allows four Tensix instructions to be queued up per
//      thread per cycle, but the maximum dequeue rate is only one instruction
//      per thread per cycle)"
//                     -- BlackholeA0/TensixTile/BabyRISCV/PushTensixInstruction.md,
//                        recorded as `riscv.ttinsn_fusion`, provenance isa_doc,
//                        consumed by nothing.
//
// A sustained burst CANNOT see fusion: the dequeue rate of one per thread per
// cycle throttles it, so 64 back-to-back `.ttinsn` words take 64 cycles whether
// the core pushed them in 64 cycles or in 16. Only a burst SHORT enough for the
// queue to drain underneath the following work can. So the group probes place
// two or four `.ttinsn` words among sixteen plain `addi`s and report cycles per
// GROUP:
//
//     tt_pad       16 addi                                  -> 16 cycles
//     tt_fuse2      2 adjacent .ttinsn + 16 addi            -> 17 fused, 18 not
//     tt_fuse4      4 adjacent .ttinsn + 16 addi            -> 17 fused, 20 not
//     tt_spread4    4 .ttinsn, none adjacent, + 16 addi     -> 20 either way
//
// and the discriminator is the pair `tt_spread4 - tt_fuse4`: 3 cycles if the
// instruction cache fuses, 0 if it does not. Both probes execute exactly the
// same twenty instructions, so nothing else can move it.
//
// PHASE C -- control flow. `riscv.integer_unit.branch_mispredict_bubble` is
// sourced (2 cycles on Wormhole, 4 on Blackhole) and `tt_sim/pe/rv/cost.py`
// declines to charge it, for a stated reason: "neither the docs nor tt-sim
// describe the predictor, so the number of mispredictions is unknowable and
// charging every taken branch would be a fabrication". This phase attacks that
// from the measurement side. Three probes execute an IDENTICAL instruction mix
// -- one `xori` and one conditional branch -- and differ only in the branch's
// direction pattern: never taken, always taken, alternating. Their DIFFERENCES
// are the mispredict cost with everything else cancelled, and they say what the
// predictor is:
//
//     all three equal                  -> no predictor, or no penalty at all
//     t > nt, alt ~ t                  -> predicts not-taken statically
//     alt > t ~ nt                     -> a real predictor, defeated by
//                                         alternation
//
// PHASE Q -- how deep the Tensix instruction queue is. Phase T measures the
// SUSTAINED rate, which is the queue's drain rate once it is full. This phase
// measures the burst length at which it fills, by timing bursts of n
// instructions and looking for the KNEE: below the queue depth the issuing core
// is not back-pressured and the marginal cost of one more instruction is one
// cycle; above it, the marginal cost is the backend unit's occupancy.
// `ADDDMAREG` is the burst op because tensixbench measured it at 3.0 cycles on
// Blackhole silicon, so the two regimes are far enough apart to see. Nothing in
// either the ISA documentation or the vendor trees gives a queue depth, so this
// phase is EXPLORATORY: it has no prediction to confirm.
//
// It sweeps n TWICE, in two burst forms, and the reason is the 2026-08-05
// silicon run: at ONE issuing thread the core had not stopped running ahead by
// n = 128, the largest burst the cascade emits, so the absorbed backlog was
// still growing where it needed to be an asymptote and a depth in ENTRIES was
// not resolvable. The fix is a longer burst -- but a longer STRAIGHT-LINE burst
// is a longer instruction stream, and 1024 `.ttinsn` words are 4 KiB of it, in
// the octave where phase F of this same benchmark found a fetch cliff. So:
//
//   cascade, n = 1 ... 128    2^p copies emitted straight-line. Unchanged, and
//                             still the form the tracked silicon datasets used.
//   loop,    n = 16 ... 1024  n/16 iterations of one 16-instruction block, so
//                             the text is 64 bytes at every n and nothing in
//                             this column can be instruction fetch.
//
// The two overlap at n = 16, 32, 64, 128. See the QLOOPBURST block below.
//
// PHASE F -- instruction footprint. A loop body of K instructions, for K from
// 64 to 2048, i.e. 256 bytes to 8 KiB of instruction text. The ISA docs give
// the fetch PERIOD ("128-bit reads against L1 ... each read yields four
// instructions, so instruction fetches are expected to occur at most once every
// four cycles") and say nothing about the cache size or the miss cost. So the
// prediction is only that small K costs 1.0 cycles per instruction; where it
// stops doing so is the measurement, and it is exploratory. Its bodies are
// UNCHANGED and deliberately so: two tracked silicon datasets carry its rows.
//
// PHASE G -- the same question at 1280, 1536 and 1792, which narrows the
// boundary the 2026-08-05 Blackhole run put between a 4 KiB and an 8 KiB loop
// body (0.998 flat from 64 through 1024, 1.251 at 2048). It is a separate
// phase because phase F's build is already within a few hundred bytes of
// tt-metal's kernel config buffer, and it is split into compile-time SETS
// because the three intermediates do not fit in one build either. It does not
// turn the boundary into a cache size: no document gives one, and a step in
// cost against loop-body size is equally consistent with a prefetch window or
// an L1 access pattern. What is measured is where the step is, in bytes of
// loop body.
//
// PHASE S -- is the queue phase Q measured SHARED between the three TRISCs, or
// PRIVATE to each? Phase Q resolved a depth at one issuing thread (~31-32
// entries on Blackhole silicon, once its read-out's dropped reference-burst
// term is added back and its bursts are drained) and could not resolve one at
// any other, so the two hypotheses are
// still identical from where it stands. The construction that separates them,
// and the two that look like they would and do not, are argued in
// rvbench_layout.h's RVBENCH_S_* block. In one line: only a SATURATED second
// thread occupies queue entries, so the discriminator has to be a second
// thread issuing flat out, with the backend bandwidth it steals measured in
// the same slot and divided back out -- and with a reference burst short
// enough (n = 4) that the queue is not already full at it, which is the thing
// phase Q's n = 16 reference got wrong.
//
// ---------------------------------------------------------------------------
// LEAVING THE CARD CLEAN. This is a design constraint rather than a cleanup
// step here, and it is worth being explicit about because tensixbench had to be
// fixed for getting it wrong: that benchmark acquired a SrcA/SrcB bank with
// `set_dvalid` and never released it, so a SUCCESSFUL run wedged the next one.
//
// riscvbench touches NO Tensix state that outlives an instruction. The five
// Tensix opcodes it issues are `NOP`, `SFPNOP`, `SETDMAREG`, `ADDDMAREG` and
// nothing else; the first two are architecturally inert, and the second two
// write Tensix GPR indices 60-62, which no LLK uses. It sets no data-valid bit,
// acquires no bank, starts no unpack or pack, writes no backend configuration
// and programs no data format. There is therefore nothing to hand back, and no
// ordering between threads to get right at the end. The L1 it writes is a
// buffer it allocated through the tt-metal allocator.
// ---------------------------------------------------------------------------

#include <cstdint>

#include "api/compute/common.h"
#include "hostdevcommon/kernel_structs.h"

#include "rvbench_layout.h"

// ---------------------------------------------------------------------------
// Unrolling. Variadic so a probe body may contain commas.
// ---------------------------------------------------------------------------
#define REP2(...) __VA_ARGS__ __VA_ARGS__
#define REP4(...) REP2(__VA_ARGS__) REP2(__VA_ARGS__)
#define REP8(...) REP4(__VA_ARGS__) REP4(__VA_ARGS__)
#define REP12(...) REP8(__VA_ARGS__) REP4(__VA_ARGS__)
#define REP16(...) REP8(__VA_ARGS__) REP8(__VA_ARGS__)
#define REP32(...) REP16(__VA_ARGS__) REP16(__VA_ARGS__)
#define REP64(...) REP32(__VA_ARGS__) REP32(__VA_ARGS__)
#define REP128(...) REP64(__VA_ARGS__) REP64(__VA_ARGS__)
#define REP256(...) REP128(__VA_ARGS__) REP128(__VA_ARGS__)
#define REP512(...) REP256(__VA_ARGS__) REP256(__VA_ARGS__)
#define REP1024(...) REP512(__VA_ARGS__) REP512(__VA_ARGS__)
#define REP1152(...) REP1024(__VA_ARGS__) REP128(__VA_ARGS__)
#define REP1280(...) REP1024(__VA_ARGS__) REP256(__VA_ARGS__)
#define REP1408(...) REP1024(__VA_ARGS__) REP256(__VA_ARGS__) REP128(__VA_ARGS__)
#define REP1536(...) REP1024(__VA_ARGS__) REP512(__VA_ARGS__)
#define REP1792(...) REP1024(__VA_ARGS__) REP512(__VA_ARGS__) REP256(__VA_ARGS__)
#define REP2048(...) REP1024(__VA_ARGS__) REP1024(__VA_ARGS__)

namespace {

// The device's own free-running cycle counter -- the same register tt-metal's
// device profiler reads for DeviceZoneScopedN (kernel_profiler.hpp reads
// RISCV_DEBUG_REG_WALL_CLOCK_L / +4). Reading the low word latches the high
// word into +4; only the low word is needed, since no probe here runs for 2^32
// cycles.
inline uint32_t wall_clock_lo() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return p[0];
}

volatile tt_l1_ptr uint32_t* g_barrier = nullptr;
uint32_t g_thread = 0;
uint32_t g_active = 1;
uint32_t g_seq = 0;

// Line the participating TRISCs up so a contention probe really is contended.
// Each thread publishes a sequence number in its own word and spins until every
// active thread has published at least that number: no atomics needed, because
// each word has exactly one writer. The host zeroes the region before launch.
inline void bench_barrier() {
    g_seq++;
    g_barrier[g_thread] = g_seq;
    for (uint32_t t = 0; t < RVBENCH_MAX_THREADS; t++) {
        if (g_active & (1u << t)) {
            while (g_barrier[t] < g_seq) {
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// The slope probes: four (blocks, cycles) points into the caller's slot.
// ---------------------------------------------------------------------------
#define PROBE(SLOT, ...)                                                 \
    do {                                                                 \
        for (uint32_t p = 0; p < RVBENCH_SLOPE_POINTS; p++) {            \
            const uint32_t blocks = base_blocks * (p + 1);               \
            bench_barrier();                                             \
            const uint32_t t0 = wall_clock_lo();                         \
            for (uint32_t b = 0; b < blocks; b++) {                      \
                __VA_ARGS__                                              \
            }                                                            \
            out[(SLOT) * RVBENCH_MAX_POINTS + p] = wall_clock_lo() - t0; \
        }                                                                \
    } while (0)

// A probe that was masked off, or whose phase was not compiled in. Still
// barriers, so the participating threads stay in step, and reports 0 so the
// analysis can tell "not run" from "0 cycles".
#define PROBE_SKIP(SLOT, POINTS)                          \
    do {                                                  \
        for (uint32_t p = 0; p < (POINTS); p++) {         \
            bench_barrier();                              \
            out[(SLOT) * RVBENCH_MAX_POINTS + p] = 0;     \
        }                                                 \
    } while (0)

#define PROBE_ON(SLOT) \
    (((((SLOT) < 32) ? probe_mask_lo : probe_mask_hi) >> ((SLOT) & 31)) & 1u)

#define RUN(SLOT, ...)                    \
    if (PROBE_ON(SLOT)) {                 \
        PROBE(SLOT, __VA_ARGS__);         \
    } else {                              \
        PROBE_SKIP(SLOT, RVBENCH_SLOPE_POINTS); \
    }

// ---------------------------------------------------------------------------
// The queue-depth probes: eight (burst length, cycles) points, burst length
// 2^p. The cascade below emits exactly 2^p copies of the body --
// 1 + 1 + 2 + 4 + ... + 2^(p-1) = 2^p -- and `q_ctrl` runs the identical
// cascade with an empty body so the up-to-eight compare-and-branch pairs it
// costs are subtracted rather than attributed to the Tensix queue.
// ---------------------------------------------------------------------------
#define QBURST(P, ...)                    \
    do {                                  \
        __VA_ARGS__                       \
        if ((P) >= 1) { __VA_ARGS__ }     \
        if ((P) >= 2) { REP2(__VA_ARGS__) }   \
        if ((P) >= 3) { REP4(__VA_ARGS__) }   \
        if ((P) >= 4) { REP8(__VA_ARGS__) }   \
        if ((P) >= 5) { REP16(__VA_ARGS__) }  \
        if ((P) >= 6) { REP32(__VA_ARGS__) }  \
        if ((P) >= 7) { REP64(__VA_ARGS__) }  \
    } while (0)

#define QPROBE(SLOT, ...)                                                \
    do {                                                                 \
        for (uint32_t p = 0; p < RVBENCH_MAX_POINTS; p++) {              \
            bench_barrier();                                             \
            const uint32_t t0 = wall_clock_lo();                         \
            QBURST(p, __VA_ARGS__);                                      \
            out[(SLOT) * RVBENCH_MAX_POINTS + p] = wall_clock_lo() - t0; \
        }                                                                \
    } while (0)

#define QRUN(SLOT, ...)                   \
    if (PROBE_ON(SLOT)) {                 \
        QPROBE(SLOT, __VA_ARGS__);        \
    } else {                              \
        PROBE_SKIP(SLOT, RVBENCH_MAX_POINTS); \
    }

// ---------------------------------------------------------------------------
// The queue-depth probes, LOOP form: n / RVBENCH_Q_LOOP_BLOCK iterations of one
// unrolled block, for n = 16, 32 ... 1024.
//
// WHY THE CASCADE WAS NOT SIMPLY EXTENDED TO 1024, which is the one design
// decision in this phase that could have produced a confident wrong answer.
// The cascade's instruction stream IS the burst: 1024 `.ttinsn` words are 4 KiB
// of straight-line text. Phase F of this same benchmark measured a fetch cliff
// in that octave (0.998 cycles/instruction flat through a 4 KiB loop body,
// 1.251 at 8 KiB), and -- unlike every slope probe here -- a phase-Q burst runs
// exactly ONCE, cold, with its own instruction fetch inside the timed region
// and no repetition to average it out. So the cascade's cost per instruction
// already carries a cold-fetch term, and lengthening the stream can only make
// that term grow with n. A cost per instruction that grows with n is EXACTLY
// the signature this phase reads as back-pressure. The two would have been
// indistinguishable, and the phase would have reported a queue depth that was
// really an instruction-fetch capacity.
//
// The loop form removes it rather than correcting for it: the body is
// RVBENCH_Q_LOOP_BLOCK instructions -- 64 bytes -- at every burst length, so
// the fetch cost is a per-burst CONSTANT and cancels exactly in any difference
// between two points. What it adds instead is the loop's own back edge, worth
// about (counter + compare + branch) / RVBENCH_Q_LOOP_BLOCK cycles per
// instruction, and `q_loop_addi` -- the identical loop with `addi` bodies --
// measures that instead of assuming it. The forms overlap at n = 16..128, where
// the cascade also runs, so "does the form change the answer" is a measurement
// too.
// ---------------------------------------------------------------------------
#define QLOOPBURST(N, ...)                                     \
    do {                                                       \
        const uint32_t iters = (N) / RVBENCH_Q_LOOP_BLOCK;     \
        for (uint32_t qi = 0; qi < iters; qi++) {              \
            REP16(__VA_ARGS__)                                 \
        }                                                      \
    } while (0)

// THE UNTIMED DRAIN, and why it is here. Every timed burst below starts from an
// EMPTY Tensix instruction queue, because `tensix_sync()` sits between
// `bench_barrier()` and `t0` -- outside the timed region, so it costs this probe
// no measured cycle. Without it the burst at point p starts with whatever point
// p-1 left in flight: a saturated backend never idles, so the residue is not
// absorbed, it back-pressures the core that many entries sooner and is added to
// `plain` permanently. The four banked 2026-08-05 runs show exactly that, at
// +21 cycles flat from n = 128 onward against phase S's identically-shaped
// probe, which has had this line since it was written. Phase S's block comment
// below, point (3), is the same argument.
//
// It is deliberately in the macro rather than in one probe, so `q_loop_addi` --
// the fetch- and structure-matched control -- gets it too. `q_loop_addi` pushes
// no Tensix instruction, so its drain returns on an empty queue and its numbers
// must not move at all: that is the control ON THIS EDIT as well as on the
// burst form. The `_sync` probe below needs no such line and does not have one:
// its timed region ENDS with `tensix_sync()`, so its next point already starts
// drained -- which is why the two forms' `_sync` probes already agree to the
// cycle where their `plain` probes do not.
#define QLOOPPROBE(SLOT, ...)                                            \
    do {                                                                 \
        for (uint32_t p = 0; p < RVBENCH_Q_LOOP_POINTS; p++) {           \
            bench_barrier();                                             \
            ckernel::tensix_sync();                                      \
            const uint32_t t0 = wall_clock_lo();                         \
            QLOOPBURST(RVBENCH_Q_LOOP_MIN_N << p, __VA_ARGS__);          \
            out[(SLOT) * RVBENCH_MAX_POINTS + p] = wall_clock_lo() - t0; \
        }                                                                \
    } while (0)

#define QLOOPRUN(SLOT, ...)                       \
    if (PROBE_ON(SLOT)) {                         \
        QLOOPPROBE(SLOT, __VA_ARGS__);            \
    } else {                                      \
        PROBE_SKIP(SLOT, RVBENCH_Q_LOOP_POINTS);  \
    }

// QLOOPBURST spells its unroll as REP16, so the two constants have to agree or
// every loop-form burst would be a different length than the host labels it.
static_assert(RVBENCH_Q_LOOP_BLOCK == 16, "QLOOPBURST unrolls with REP16");
static_assert(RVBENCH_Q_LOOP_POINTS <= RVBENCH_MAX_POINTS, "result slots are fixed width");

// ---------------------------------------------------------------------------
// PHASE S -- is the Tensix instruction queue shared between the TRISCs, or
// private to each? The design argument is in rvbench_layout.h's RVBENCH_S_*
// block and at length in docs/plans/riscv-front-end-benchmark.md; what is
// implemented here is the three things it concludes.
//
// (1) A SHORTER REFERENCE BURST. Every burst below starts at n = 4 rather than
//     phase Q's 16, because the reference point is subtracted off as
//     "tensix_sync()'s own cost" and that is only true where the queue is not
//     already full. At n = 16 it is -- structurally so at two and three
//     issuing threads, which is why phase Q's multi-thread slots read zero
//     rather than reading small. A four-instruction loop block is what lets n
//     start at 4; the body is 16 bytes at every burst length, so no difference
//     between two points here is instruction fetch either.
//
// (2) A DRAINED PAIR PER CONDITION, so the service rate is measured in the
//     slot it is used in rather than carried across from another thread count.
//     The whole difficulty of this question is that any second thread which
//     actually occupies queue entries must also be taking backend bandwidth --
//     an idle or a slow one holds nothing, by Little's law -- so the bandwidth
//     it takes has to be measured and divided back out, not avoided.
//
// (3) AN UNTIMED DRAIN BEFORE EVERY TIMED BURST. `tensix_sync()` immediately
//     before `t0` guarantees the queue is empty when the burst starts, in the
//     plain probe as well as the sync one. Without it the n = 4 point -- the
//     one the whole read-out is referenced to -- would carry whatever the
//     previous probe left behind.
// ---------------------------------------------------------------------------
#define SLOOPBURST(N, ...)                                     \
    do {                                                       \
        const uint32_t iters = (N) / RVBENCH_S_LOOP_BLOCK;     \
        for (uint32_t si = 0; si < iters; si++) {              \
            REP4(__VA_ARGS__)                                  \
        }                                                      \
    } while (0)

// What a non-issuing thread does during a solo probe: the same loop, with
// `addi` bodies, for RVBENCH_S_SPIN_MULT times as many instructions, so it is
// certainly still running when the issuer's second clock read happens. It
// pushes no Tensix instruction, which is the entire point -- it occupies no
// queue entry under either hypothesis.
#define SSPIN(N)                                                        \
    SLOOPBURST(RVBENCH_S_SPIN_MULT * (N), asm volatile("addi %0, %0, 1" : "+r"(a0));)

// The two values `DRAIN` takes below. `SDRAIN` puts `tensix_sync()` INSIDE the
// timed region, so the probe cannot return until the pipe has drained;
// `SNODRAIN` expands to nothing and the probe returns when the core's last
// push did. The difference between the pair at each n is the work still in
// flight, which is the whole measurement.
#define SNODRAIN
#define SDRAIN ckernel::tensix_sync();

// Every active thread issues.
#define SPROBE_CO(SLOT, DRAIN, ...)                                      \
    do {                                                                 \
        for (uint32_t p = 0; p < RVBENCH_S_POINTS; p++) {                \
            bench_barrier();                                             \
            ckernel::tensix_sync();                                      \
            const uint32_t t0 = wall_clock_lo();                         \
            SLOOPBURST(RVBENCH_S_MIN_N << p, __VA_ARGS__);               \
            DRAIN                                                        \
            out[(SLOT) * RVBENCH_MAX_POINTS + p] = wall_clock_lo() - t0; \
        }                                                                \
    } while (0)

// Only RVBENCH_S_ISSUER issues; every other active thread spins. The
// non-issuers write 0, and the host does not emit their rows at all.
#define SPROBE_SOLO(SLOT, DRAIN, ...)                                        \
    do {                                                                     \
        for (uint32_t p = 0; p < RVBENCH_S_POINTS; p++) {                    \
            bench_barrier();                                                 \
            ckernel::tensix_sync();                                          \
            if (g_thread == RVBENCH_S_ISSUER) {                              \
                const uint32_t t0 = wall_clock_lo();                         \
                SLOOPBURST(RVBENCH_S_MIN_N << p, __VA_ARGS__);               \
                DRAIN                                                        \
                out[(SLOT) * RVBENCH_MAX_POINTS + p] = wall_clock_lo() - t0; \
            } else {                                                         \
                SSPIN(RVBENCH_S_MIN_N << p);                                 \
                out[(SLOT) * RVBENCH_MAX_POINTS + p] = 0;                    \
            }                                                                \
        }                                                                    \
    } while (0)

#define SRUN_CO(SLOT, DRAIN, ...)                 \
    if (PROBE_ON(SLOT)) {                         \
        SPROBE_CO(SLOT, DRAIN, __VA_ARGS__);      \
    } else {                                      \
        PROBE_SKIP(SLOT, RVBENCH_S_POINTS);       \
    }

#define SRUN_SOLO(SLOT, DRAIN, ...)               \
    if (PROBE_ON(SLOT)) {                         \
        SPROBE_SOLO(SLOT, DRAIN, __VA_ARGS__);    \
    } else {                                      \
        PROBE_SKIP(SLOT, RVBENCH_S_POINTS);       \
    }

// SLOOPBURST spells its unroll as REP4, so the two constants have to agree.
static_assert(RVBENCH_S_LOOP_BLOCK == 4, "SLOOPBURST unrolls with REP4");
static_assert(RVBENCH_S_POINTS <= RVBENCH_MAX_POINTS, "result slots are fixed width");
static_assert(RVBENCH_S_ISSUER < RVBENCH_MAX_THREADS, "the issuer must be a real thread");

void kernel_main() {
    const uint32_t results_addr = get_arg_val<uint32_t>(0);
    const uint32_t barrier_addr = get_arg_val<uint32_t>(1);
    const uint32_t scratch_addr = get_arg_val<uint32_t>(2);
    const uint32_t base_blocks = get_arg_val<uint32_t>(3);
    // Two words, because there are more than 32 probes. PROBE_ON below indexes
    // whichever half a slot lives in; the `& 31` on both branches is what keeps
    // the dead branch from being a compile-time shift overflow.
    const uint32_t probe_mask_lo = get_arg_val<uint32_t>(4);
    const uint32_t active_mask = get_arg_val<uint32_t>(5);
    const uint32_t phase_bits = get_arg_val<uint32_t>(6);
    const uint32_t probe_mask_hi = get_arg_val<uint32_t>(7);

#if defined(TRISC_UNPACK)
    g_thread = 0;
#elif defined(TRISC_MATH)
    g_thread = 1;
#else
    g_thread = 2;
#endif
    g_active = active_mask;
    g_barrier = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_addr);

    if ((active_mask & (1u << g_thread)) == 0) {
        return;
    }

    // Each thread gets its own scratch region, so that a t3 run's three copies
    // of the store probes are not writing over each other's cache lines. That
    // would be a contention measurement, and a confounded one, rather than the
    // per-core cost this phase reports.
    const uint32_t my_scratch = scratch_addr + g_thread * RVBENCH_SCRATCH_BYTES;

    volatile tt_l1_ptr uint32_t* base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(results_addr);

    // Two stack locals pointing at each other: a two-node pointer chase, which
    // is how the stack load probe measures a load-use LATENCY rather than a
    // throughput. Their address is reported in the header so the analysis can
    // classify which row of the load-latency table they landed in -- tt-metal
    // is free to place a TRISC stack in core-local data RAM (latency 2) or in
    // L1 (>= 8 on Wormhole), and guessing would be worth six cycles.
    volatile uint32_t stack_node[2];
    stack_node[0] = reinterpret_cast<uint32_t>(&stack_node[1]);
    stack_node[1] = reinterpret_cast<uint32_t>(&stack_node[0]);

    if (g_thread == 1) {
        base[RVBENCH_HDR_MAGIC] = RVBENCH_MAGIC;
        base[RVBENCH_HDR_NUM_PROBES] = RVBENCH_NUM_PROBES;
        base[RVBENCH_HDR_MAX_POINTS] = RVBENCH_MAX_POINTS;
        base[RVBENCH_HDR_BASE_BLOCKS] = base_blocks;
        base[RVBENCH_HDR_ACTIVE_MASK] = active_mask;
        base[RVBENCH_HDR_PROBE_MASK] = probe_mask_lo;
        base[RVBENCH_HDR_PROBE_MASK_HI] = probe_mask_hi;
        base[RVBENCH_HDR_PHASE] = phase_bits;
        base[RVBENCH_HDR_STACK_ADDR] = reinterpret_cast<uint32_t>(&stack_node[0]);
        base[RVBENCH_HDR_SCRATCH_ADDR] = my_scratch;
    }
    volatile tt_l1_ptr uint32_t* out =
        base + RVBENCH_HDR_WORDS + g_thread * RVBENCH_NUM_PROBES * RVBENCH_MAX_POINTS;

    // The L1 pointer chase: 64 nodes, 16 bytes apart, each holding the address
    // of the next and the last wrapping to the first. `lw rd, 0(rd)` then costs
    // exactly one load latency per instruction with no second instruction in
    // the body to divide out.
    {
        volatile tt_l1_ptr uint32_t* chain = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(my_scratch);
        for (uint32_t i = 0; i < RVBENCH_UNROLL; i++) {
            const uint32_t next = my_scratch + ((i + 1) % RVBENCH_UNROLL) * 16;
            chain[i * 4] = next;
        }
    }

    // Operands the probe bodies work on. Held in locals so the compiler
    // allocates real registers for them; every body is `asm volatile`, so none
    // of it can be deleted, reordered across another body, or hoisted.
    uint32_t a0 = 1, a1 = 2, a2 = 3, a3 = 4;
    uint32_t k = 3;
    uint32_t dividend = 0x12345678u;
    uint32_t toggle = 0;
    uint32_t sink = 0;
    uint32_t chase = my_scratch;
    uint32_t stack_p = reinterpret_cast<uint32_t>(&stack_node[0]);
    const uint32_t scratch = my_scratch;
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)k;
    (void)dividend;
    (void)toggle;
    (void)sink;
    (void)chase;
    (void)stack_p;
    (void)scratch;
    (void)phase_bits;

    // -----------------------------------------------------------------------
    // Phase R -- straight-line RV32IM.
    // -----------------------------------------------------------------------
    // Slot 0 is the control for every slope probe in the run, whatever phase,
    // so it is compiled unconditionally. `asm volatile("")` is never deleted,
    // so the loop survives the optimiser -- an empty body does not, and a
    // control that reads a flat constant regardless of block count is the
    // signature of that having happened.
    RUN(RVBENCH_P_LOOP_OVERHEAD, asm volatile(""););

#ifdef RVBENCH_PHASE_R
    // Four independent chains, so the integer unit's forwarding path is not
    // what is being measured.
    RUN(RVBENCH_P_ADDI_INDEP,
        REP16(asm volatile("addi %0, %0, 1\n\t"
                           "addi %1, %1, 1\n\t"
                           "addi %2, %2, 1\n\t"
                           "addi %3, %3, 1"
                           : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3));));
    // One chain: every instruction reads the previous one's result. The docs
    // say the forwarding path makes this indistinguishable from the above
    // ("integer operations other than multiply and divide appear to execute
    // with latency 1 and throughput 1"), which is a claim worth checking.
    RUN(RVBENCH_P_ADDI_DEP, REP64(asm volatile("addi %0, %0, 1" : "+r"(a0));));

    // Multiply. The one probe whose prediction DIFFERS between the two
    // architectures: Wormhole blocks the integer unit for two cycles, Blackhole
    // splits it into one cycle of EX1 and one of EX2 and therefore pipelines.
    // So independent multiplies should read 2.0 on Wormhole and 1.0 on
    // Blackhole, and dependent ones 2.0 on both.
    RUN(RVBENCH_P_MUL_INDEP,
        REP16(asm volatile("mul %0, %0, %4\n\t"
                           "mul %1, %1, %4\n\t"
                           "mul %2, %2, %4\n\t"
                           "mul %3, %3, %4"
                           : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
                           : "r"(k));));
    RUN(RVBENCH_P_MUL_DEP, REP64(asm volatile("mul %0, %0, %1" : "+r"(a0) : "r"(k));));

    // Divide, at a fixed dividend and divisor because the docs make the cost
    // depend on both ("between six and 33 cycles ... dependent upon the
    // magnitude of the dividend"). 0x12345678 / 3 is one point on that curve
    // and the CSV header records it; a sweep of the curve is not attempted.
    RUN(RVBENCH_P_DIV, REP64(asm volatile("divu %0, %1, %2" : "=r"(sink) : "r"(dividend), "r"(k));));

    // L1 load-use latency, as a pointer chase so that the body is one
    // instruction and the measured number is the latency undivided.
    RUN(RVBENCH_P_LOAD_CHASE, REP64(asm volatile("lw %0, 0(%0)" : "+r"(chase));));
    // L1 loads with no consumer: sustained throughput, not latency. Four dest
    // registers rotating means no load is ever read before the next issues.
    RUN(RVBENCH_P_LOAD_INDEP,
        REP16(asm volatile("lw %0, 0(%4)\n\t"
                           "lw %1, 16(%4)\n\t"
                           "lw %2, 32(%4)\n\t"
                           "lw %3, 48(%4)"
                           : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                           : "r"(scratch));));
    // L1 stores to four DIFFERENT 16-byte-aligned blocks in rotation, so no two
    // consecutive stores can be coalesced under the documented predicate ("the
    // same 16-byte aligned region of L1, with start addresses within +/-4").
    // This is the "one store every five cycles" case on both architectures.
    RUN(RVBENCH_P_STORE_SPREAD,
        REP16(asm volatile("sw %0, 0(%1)\n\t"
                           "sw %0, 16(%1)\n\t"
                           "sw %0, 32(%1)\n\t"
                           "sw %0, 48(%1)"
                           :
                           : "r"(a0), "r"(scratch));));
    // The same store count into ONE 16-byte block at offsets four apart, which
    // is exactly what Blackhole's coalescing store queue is documented to
    // accept: "the constituent stores have a throughput of one store every
    // cycle". Wormhole has no such queue, so this probe and the one above
    // should be indistinguishable there and 5x apart on Blackhole.
    RUN(RVBENCH_P_STORE_COALESCE,
        REP16(asm volatile("sw %0, 0(%1)\n\t"
                           "sw %0, 4(%1)\n\t"
                           "sw %0, 0(%1)\n\t"
                           "sw %0, 4(%1)"
                           :
                           : "r"(a0), "r"(scratch));));
    // The stack, whichever region it turns out to be in. Same two-instruction-
    // free pointer chase as the L1 probe, over two mutually-pointing locals.
    RUN(RVBENCH_P_LOAD_STACK, REP64(asm volatile("lw %0, 0(%0)" : "+r"(stack_p));));
    RUN(RVBENCH_P_STORE_STACK, REP64(asm volatile("sw %0, 0(%1)" : : "r"(a0), "r"(stack_p));));
#else
    PROBE_SKIP(RVBENCH_P_ADDI_INDEP, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_ADDI_DEP, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_MUL_INDEP, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_MUL_DEP, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_DIV, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_LOAD_CHASE, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_LOAD_INDEP, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_STORE_SPREAD, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_STORE_COALESCE, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_LOAD_STACK, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_STORE_STACK, RVBENCH_SLOPE_POINTS);
#endif

    // -----------------------------------------------------------------------
    // Phase T -- the `.ttinsn` issue path.
    // -----------------------------------------------------------------------
#ifdef RVBENCH_PHASE_T
    // Sustained bursts: one `.ttinsn` per issue slot, at four backend units.
    // NOP and SFPNOP are architecturally inert; SETDMAREG has a documented
    // occupancy of 1 and ADDDMAREG of "3 or 4", so the pair says whether the
    // backend's occupancy reaches the issuing core at all.
    RUN(RVBENCH_P_TT_NOP, REP64(TTI_NOP;));
    RUN(RVBENCH_P_TT_SFPNOP, REP64(TTI_SFPNOP;));
    RUN(RVBENCH_P_TT_SETDMAREG, REP64(TTI_SETDMAREG(0, 7, 0, 120);));
    RUN(RVBENCH_P_TT_ADDDMAREG, REP64(TTI_ADDDMAREG(1, 60, 1, 60);));

    // The fusion group. Cycles per GROUP, not per instruction: the unroll
    // recorded for these four probes is RVBENCH_GROUP_REPS, so a reported
    // "cyc/instr" is really cycles per group and the analysis names it as such.
    // Every group carries the same sixteen `addi`s; only the number and the
    // adjacency of the `.ttinsn` words differ.
    RUN(RVBENCH_P_TT_PAD, REP16(REP16(asm volatile("addi %0, %0, 1" : "+r"(a0));)));
    RUN(RVBENCH_P_TT_FUSE2, REP16(TTI_NOP; TTI_NOP;
                                  REP16(asm volatile("addi %0, %0, 1" : "+r"(a0));)));
    RUN(RVBENCH_P_TT_FUSE4, REP16(TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP;
                                  REP16(asm volatile("addi %0, %0, 1" : "+r"(a0));)));
    // Same four `.ttinsn` words, same sixteen `addi`s, but no two `.ttinsn`
    // adjacent -- so the instruction cache has nothing to fuse. The difference
    // between this probe and the one above is the whole experiment.
    RUN(RVBENCH_P_TT_SPREAD4,
        REP16(REP4(TTI_NOP; asm volatile("addi %0, %0, 1" : "+r"(a0));)
              REP12(asm volatile("addi %0, %0, 1" : "+r"(a0));)));
#else
    PROBE_SKIP(RVBENCH_P_TT_NOP, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_TT_SFPNOP, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_TT_SETDMAREG, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_TT_ADDDMAREG, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_TT_PAD, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_TT_FUSE2, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_TT_FUSE4, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_TT_SPREAD4, RVBENCH_SLOPE_POINTS);
#endif

    // -----------------------------------------------------------------------
    // Phase C -- control flow.
    // -----------------------------------------------------------------------
#ifdef RVBENCH_PHASE_C
    // The matched control for the three `xori` + branch probes below.
    RUN(RVBENCH_P_C_CTRL_XOR, REP64(asm volatile("xori %0, %0, 1" : "+r"(toggle));));
    // A branch that is never taken and one that is ALWAYS taken, to a target
    // one instruction ahead, i.e. the same address the not-taken branch falls
    // through to. The two therefore execute the identical dynamic instruction
    // sequence and differ in exactly one bit: whether the branch was taken.
    RUN(RVBENCH_P_C_NT, REP64(asm volatile("bne %0, %0, 1f\n1:" : : "r"(a0));));
    RUN(RVBENCH_P_C_T, REP64(asm volatile("beq %0, %0, 1f\n1:" : : "r"(a0));));
    // The same three directions again, each preceded by an `xori`, so that the
    // alternating case -- which needs a value to alternate on -- has two
    // controls with its exact instruction mix.
    RUN(RVBENCH_P_C_XOR_NT,
        REP64(asm volatile("xori %0, %0, 1\n\t"
                           "bne %1, %1, 1f\n1:"
                           : "+r"(toggle)
                           : "r"(a0));));
    RUN(RVBENCH_P_C_XOR_T,
        REP64(asm volatile("xori %0, %0, 1\n\t"
                           "beq %1, %1, 1f\n1:"
                           : "+r"(toggle)
                           : "r"(a0));));
    RUN(RVBENCH_P_C_XOR_ALT,
        REP64(asm volatile("xori %0, %0, 1\n\t"
                           "beqz %0, 1f\n1:"
                           : "+r"(toggle));));
    // An unconditional jump: taken by definition, and nothing to predict.
    RUN(RVBENCH_P_C_JAL, REP64(asm volatile("j 1f\n1:");));
#else
    PROBE_SKIP(RVBENCH_P_C_CTRL_XOR, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_C_NT, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_C_T, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_C_XOR_NT, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_C_XOR_T, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_C_XOR_ALT, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_C_JAL, RVBENCH_SLOPE_POINTS);
#endif

    // -----------------------------------------------------------------------
    // Phase Q -- the Tensix instruction queue's depth.
    // -----------------------------------------------------------------------
#ifdef RVBENCH_PHASE_Q
    // The cascade's own compare-and-branch cost, which grows with p and so
    // cannot be treated as a constant. Subtracted point by point.
    QRUN(RVBENCH_P_Q_CTRL, asm volatile(""););
    QRUN(RVBENCH_P_Q_NOP, TTI_NOP;);
    QRUN(RVBENCH_P_Q_SETDMAREG, TTI_SETDMAREG(0, 7, 0, 120););
    QRUN(RVBENCH_P_Q_ADDDMAREG, TTI_ADDDMAREG(1, 60, 1, 60););
    // The same burst, but the timed region does not end until the Tensix
    // instruction pipe has drained. `tensix_sync()` writes PC buffer word 1 and
    // blocks until every instruction this thread has pushed has retired, so
    // this probe measures the burst's FULL cost where the one above measures
    // only the part the issuing core could not run ahead of. The difference
    // between them at each n is how much work was still in flight when the
    // burst's last `.ttinsn` word returned -- which is the queue's occupancy,
    // read directly rather than inferred from a knee.
    //
    // (Written out rather than going through QRUN because the drain has to sit
    // inside the timed region, between the burst and the second clock read.)
    if (PROBE_ON(RVBENCH_P_Q_ADDDMAREG_SYNC)) {
        // One untimed `tensix_sync()` first. The FIRST one of a launch is not
        // like its successors: at two and three issuing threads it measured 92
        // and 459 cycles against ~9 for every later call, which is a start-up
        // transient (the PC buffer's first round trip, and the other threads
        // still arriving) and not a property of a one-instruction burst. Left
        // un-warmed it made the n=1 point ten times the n=2 point and the
        // monotonicity gate -- correctly -- refused the phase. Warming it up
        // outside the timed region is the fix; weakening the gate would not be.
        ckernel::tensix_sync();
        for (uint32_t p = 0; p < RVBENCH_MAX_POINTS; p++) {
            bench_barrier();
            const uint32_t t0 = wall_clock_lo();
            QBURST(p, TTI_ADDDMAREG(1, 60, 1, 60););
            ckernel::tensix_sync();
            out[RVBENCH_P_Q_ADDDMAREG_SYNC * RVBENCH_MAX_POINTS + p] = wall_clock_lo() - t0;
        }
    } else {
        PROBE_SKIP(RVBENCH_P_Q_ADDDMAREG_SYNC, RVBENCH_MAX_POINTS);
    }

    // The loop form, out to n = 1024. `q_loop_addi` is not a spare probe: it is
    // the fetch- AND structure-matched control for the two below it. Same loop,
    // same trip count, same 4-byte instruction words, same 64-byte body, no
    // Tensix involvement -- so it measures what this burst form costs when
    // NOTHING can back-pressure it, which is the baseline the other two are
    // read against. Without it a loop-form rate above 1.0 would be ambiguous
    // between the queue and the loop's own back edge.
    QLOOPRUN(RVBENCH_P_Q_LOOP_ADDI, asm volatile("addi %0, %0, 1" : "+r"(a0)););
    QLOOPRUN(RVBENCH_P_Q_LOOP_ADDDMAREG, TTI_ADDDMAREG(1, 60, 1, 60););
    if (PROBE_ON(RVBENCH_P_Q_LOOP_ADDDMAREG_SYNC)) {
        // Warmed up outside the timed region for the same reason as the cascade
        // probe above: the first `tensix_sync()` of a launch is not like its
        // successors. Repeated here rather than relying on the cascade probe's
        // warm-up, because `--probes` can turn that one off.
        //
        // ONE warm-up is all this probe needs, where QLOOPPROBE drains before
        // every point: the loop below ENDS each timed region with a drain, so
        // every later point already starts on an empty queue. See QLOOPPROBE's
        // comment -- this asymmetry is the whole reason the two forms' `plain`
        // probes differed by 21 cycles while their `_sync` probes did not.
        ckernel::tensix_sync();
        for (uint32_t p = 0; p < RVBENCH_Q_LOOP_POINTS; p++) {
            bench_barrier();
            const uint32_t t0 = wall_clock_lo();
            QLOOPBURST(RVBENCH_Q_LOOP_MIN_N << p, TTI_ADDDMAREG(1, 60, 1, 60););
            ckernel::tensix_sync();
            out[RVBENCH_P_Q_LOOP_ADDDMAREG_SYNC * RVBENCH_MAX_POINTS + p] =
                wall_clock_lo() - t0;
        }
    } else {
        PROBE_SKIP(RVBENCH_P_Q_LOOP_ADDDMAREG_SYNC, RVBENCH_Q_LOOP_POINTS);
    }
#else
    PROBE_SKIP(RVBENCH_P_Q_CTRL, RVBENCH_MAX_POINTS);
    PROBE_SKIP(RVBENCH_P_Q_NOP, RVBENCH_MAX_POINTS);
    PROBE_SKIP(RVBENCH_P_Q_SETDMAREG, RVBENCH_MAX_POINTS);
    PROBE_SKIP(RVBENCH_P_Q_ADDDMAREG, RVBENCH_MAX_POINTS);
    PROBE_SKIP(RVBENCH_P_Q_ADDDMAREG_SYNC, RVBENCH_MAX_POINTS);
    PROBE_SKIP(RVBENCH_P_Q_LOOP_ADDI, RVBENCH_Q_LOOP_POINTS);
    PROBE_SKIP(RVBENCH_P_Q_LOOP_ADDDMAREG, RVBENCH_Q_LOOP_POINTS);
    PROBE_SKIP(RVBENCH_P_Q_LOOP_ADDDMAREG_SYNC, RVBENCH_Q_LOOP_POINTS);
#endif

    // -----------------------------------------------------------------------
    // Phase F -- instruction footprint. Six loop bodies, 64 to 2048
    // instructions, i.e. 256 bytes to 8 KiB of text. All six are the SAME
    // instruction, so the only thing that varies across the six probes is how
    // far apart in memory the loop's two ends are.
    // -----------------------------------------------------------------------
#ifdef RVBENCH_PHASE_F
    RUN(RVBENCH_P_F_64, REP64(asm volatile("addi %0, %0, 1" : "+r"(a0));));
    RUN(RVBENCH_P_F_128, REP128(asm volatile("addi %0, %0, 1" : "+r"(a0));));
    RUN(RVBENCH_P_F_256, REP256(asm volatile("addi %0, %0, 1" : "+r"(a0));));
    RUN(RVBENCH_P_F_512, REP512(asm volatile("addi %0, %0, 1" : "+r"(a0));));
    RUN(RVBENCH_P_F_1024, REP1024(asm volatile("addi %0, %0, 1" : "+r"(a0));));
    RUN(RVBENCH_P_F_2048, REP2048(asm volatile("addi %0, %0, 1" : "+r"(a0));));
#else
    PROBE_SKIP(RVBENCH_P_F_64, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_F_128, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_F_256, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_F_512, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_F_1024, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_F_2048, RVBENCH_SLOPE_POINTS);
#endif

    // -----------------------------------------------------------------------
    // Phase G -- the intermediate footprints, 1280 / 1536 / 1792 instructions,
    // one per compile-time set, each measured against a 1024-instruction body
    // in the SAME build. Phase F could not simply be widened: its six bodies
    // already sit within a few hundred bytes of tt-metal's kernel config
    // buffer, and adding these three to it aborts the launch outright. See
    // rvbench_layout.h's RVBENCH_P_G_* block for the measured numbers.
    //
    // The probes of a set that is not selected are not compiled AND not
    // skipped: every thread of a given build runs the identical sequence, so
    // the barriers stay in step without a PROBE_SKIP standing in for them.
    // -----------------------------------------------------------------------
#ifndef RVBENCH_G_SET
#define RVBENCH_G_SET 0
#endif
#ifdef RVBENCH_PHASE_G
    RUN(RVBENCH_P_G_1024, REP1024(asm volatile("addi %0, %0, 1" : "+r"(a0));));
    // Every arm is an explicit `== N`, and the last is an `#error` rather than
    // a bare `#else`. Before sets 3 and 4 existed the chain ended in `#else`,
    // which meant `--gset 3` would have compiled the 1792 body while the host
    // emitted only the g_1152 rows -- an empty phase-G CSV from a build that
    // silently measured something else. The host validates `--gset` against
    // RVBENCH_G_SETS, so this arm is unreachable through it; it is here so that
    // widening RVBENCH_G_SETS without widening this chain fails at the compiler
    // rather than at the card.
#if RVBENCH_G_SET == 0
    RUN(RVBENCH_P_G_1280, REP1280(asm volatile("addi %0, %0, 1" : "+r"(a0));));
#elif RVBENCH_G_SET == 1
    RUN(RVBENCH_P_G_1536, REP1536(asm volatile("addi %0, %0, 1" : "+r"(a0));));
#elif RVBENCH_G_SET == 2
    RUN(RVBENCH_P_G_1792, REP1792(asm volatile("addi %0, %0, 1" : "+r"(a0));));
#elif RVBENCH_G_SET == 3
    RUN(RVBENCH_P_G_1152, REP1152(asm volatile("addi %0, %0, 1" : "+r"(a0));));
#elif RVBENCH_G_SET == 4
    RUN(RVBENCH_P_G_1408, REP1408(asm volatile("addi %0, %0, 1" : "+r"(a0));));
#else
#error "RVBENCH_G_SET names no compiled footprint; add an arm above"
#endif
#else
    PROBE_SKIP(RVBENCH_P_G_1024, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_G_1280, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_G_1536, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_G_1792, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_G_1152, RVBENCH_SLOPE_POINTS);
    PROBE_SKIP(RVBENCH_P_G_1408, RVBENCH_SLOPE_POINTS);
#endif

    // -----------------------------------------------------------------------
    // Phase S -- is the Tensix instruction queue shared or per-thread? Six
    // probes in the RVBENCH_S_* loop form, n = 4 ... 512. The kernel TEXT is
    // identical in the t1, t2 and t3 launches -- only the runtime active mask
    // differs -- so the comparison across thread counts, which is where this
    // phase's answer lives, is not confounded by a different build.
    // -----------------------------------------------------------------------
#ifdef RVBENCH_PHASE_S
    // The issue-limited control: `p` in the depth formula, and the measurement
    // of what a four-instruction loop block costs in back edge.
    SRUN_CO(RVBENCH_P_S_LOOP_ADDI, SNODRAIN, asm volatile("addi %0, %0, 1" : "+r"(a0)););
    // Every active thread issuing. At t1 this is one thread and reproduces the
    // phase Q question with a smaller reference burst; at t2/t3 the queue is
    // contended, which is the condition that separates the two hypotheses.
    SRUN_CO(RVBENCH_P_S_CO_PLAIN, SNODRAIN, TTI_ADDDMAREG(1, 60, 1, 60););
    // Byte-identical to the probe above. The two differ only in having run at
    // a different moment, so |co_plain - co_repeat| is this phase's measured
    // single-shot repeatability -- the floor a backlog has to clear before it
    // is divided by anything.
    SRUN_CO(RVBENCH_P_S_CO_REPEAT, SNODRAIN, TTI_ADDDMAREG(1, 60, 1, 60););
    SRUN_CO(RVBENCH_P_S_CO_SYNC, SDRAIN, TTI_ADDDMAREG(1, 60, 1, 60););
    // One thread issues, the others only spin. THIS IS A CONTROL AND NOT THE
    // DISCRIMINATOR: a spinning thread pushes nothing, so it holds no queue
    // entry whether the queue is shared or private, and the issuer must see
    // the same depth here at t1, t2 and t3 under BOTH hypotheses. What it
    // separates is "another core is awake" -- competing for instruction fetch
    // out of the same L1 -- from "another core is issuing", which is what the
    // co probes above vary.
    SRUN_SOLO(RVBENCH_P_S_SOLO_PLAIN, SNODRAIN, TTI_ADDDMAREG(1, 60, 1, 60););
    SRUN_SOLO(RVBENCH_P_S_SOLO_SYNC, SDRAIN, TTI_ADDDMAREG(1, 60, 1, 60););
#else
    PROBE_SKIP(RVBENCH_P_S_LOOP_ADDI, RVBENCH_S_POINTS);
    PROBE_SKIP(RVBENCH_P_S_CO_PLAIN, RVBENCH_S_POINTS);
    PROBE_SKIP(RVBENCH_P_S_CO_REPEAT, RVBENCH_S_POINTS);
    PROBE_SKIP(RVBENCH_P_S_CO_SYNC, RVBENCH_S_POINTS);
    PROBE_SKIP(RVBENCH_P_S_SOLO_PLAIN, RVBENCH_S_POINTS);
    PROBE_SKIP(RVBENCH_P_S_SOLO_SYNC, RVBENCH_S_POINTS);
#endif

    // Nothing to release. See "LEAVING THE CARD CLEAN" in the header comment:
    // this kernel acquires no Tensix resource, so a completed run leaves the
    // coprocessor exactly as it found it and the next run needs no reset.
    // Phase S changes nothing about that -- it issues the same `ADDDMAREG` into
    // the same Tensix GPRs 60-62 and the same `tensix_sync()` the queue phase
    // already used, and a spinning thread executes `addi` and nothing else.
}
