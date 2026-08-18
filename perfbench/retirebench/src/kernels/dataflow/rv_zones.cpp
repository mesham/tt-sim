// retirebench -- rung 4's RV-bound leg, measured from a kernel.
//
// WHAT THIS IS. Twelve consecutive ZONES on one baby RISC-V core, each bracketed
// by a read of `mcycle` and a read of `minstret`, all nested inside one outer
// window. The zones telescope to the window by construction -- whatever is not
// inside a zone is the window minus the sum of the zones -- so the same binary
// run on silicon and against tt-sim yields two partitions of the same quantity
// and `tt_sim.perf.retire_attribution` can compare their interiors rather than
// only their totals.
//
// THE INSTRUMENT, and the one constraint that shapes every line below.
// `mcycle` (0xb00) and `minstret` (0xb02) are Blackhole baby-RISC-V CSRs
// (BlackholeA0/TensixTile/BabyRISCV/CSRs.md). `cfg0` bit 10 `DisCsrSync` is
// documented so: while it is CLEAR, once a csrr* instruction leaves the front
// end the next instruction does not leave until the previous has retired. It is
// clear at reset and clear in tt-metal's own init, so on a real part
//
//     A CSR READ IS A RETIREMENT BARRIER.
//
// Therefore the counters are read AROUND WINDOWS and never per instruction. Two
// reads bracketing thousands of instructions serialise only at the boundaries
// and dilate nothing in between; a per-instruction bracket would measure the
// barrier and call it the instruction. Zone 0 (`marker_null`) is two marker
// pairs with nothing between them, so the instrument's own cost is a MEASURED
// number in every artefact rather than an assumption in a comment.
//
// WHAT THE HARDWARE CAN AND CANNOT SAY. Per zone it gives elapsed cycles and
// retired instructions, and that is all of it. `mhpmcounter3/4` exist but the
// encodings of their `mhpmevent3/4` selectors are unpublished, so no event can
// be given a meaning -- tt-sim's own CSR file refuses those counters once an
// event is selected, and nothing here selects one. There is no PC sampler and
// no instruction-trace buffer in tt-metal 0.74, UMD or the public ISA docs. So
// the MECHANISM split has to come from the program's structure: each zone is
// built to be dominated by one RV mechanism, and `minstret` is what makes that
// structural claim checkable rather than merely asserted -- two sides that
// retired different instruction counts in a zone did not run the same zone, and
// the analysis refuses them.
//
// The zone bodies are `perfbench/riscvbench`'s phase R and phase C probe bodies
// unchanged, deliberately: that benchmark measured each of them on Blackhole
// silicon on 2026-08-05, so the repetition counts here can be SOURCED from a
// measurement instead of guessed, and a per-zone cycles-per-instruction figure
// in this leg's report is directly comparable to a slope in that one's.
//
// riscvbench and retirebench are different instruments on the same core: that
// one reports SLOPES over four burst lengths (kernel launch and timer overhead
// cancel exactly, and its output is a per-instruction cost); this one reports
// one ABSOLUTE partition of one span, which is the only form in which a total
// and its interior can be checked against each other at all.
//
// LEAVING THE CARD CLEAN. No Tensix instruction is issued, no NoC transaction
// is started, no semaphore is touched. The only device state written is two L1
// buffers this program's host side allocated through the tt-metal allocator.

#include <cstdint>

#include "retirebench_layout.h"

namespace {

// The two counters. `mcycle` reads the same tile clock that
// RISCV_DEBUG_REG_WALL_CLOCK_* samples, so a cycle here is the same cycle the
// device profiler and every other perfbench program count in.
//
// The "memory" clobber is load-bearing rather than defensive: it stops the
// compiler moving a result store across a marker, which would move real work
// from one bucket into another with nothing in the artefact to show for it.
inline uint32_t rd_mcycle() {
    uint32_t v;
    asm volatile("csrr %0, mcycle" : "=r"(v)::"memory");
    return v;
}

inline uint32_t rd_minstret() {
    uint32_t v;
    asm volatile("csrr %0, minstret" : "=r"(v)::"memory");
    return v;
}

}  // namespace

// Unrolling. Variadic so a body may contain commas.
#define REP2(...) __VA_ARGS__ __VA_ARGS__
#define REP4(...) REP2(__VA_ARGS__) REP2(__VA_ARGS__)
#define REP8(...) REP4(__VA_ARGS__) REP4(__VA_ARGS__)
#define REP16(...) REP8(__VA_ARGS__) REP8(__VA_ARGS__)
#define REP32(...) REP16(__VA_ARGS__) REP16(__VA_ARGS__)
#define REP64(...) REP32(__VA_ARGS__) REP32(__VA_ARGS__)

// One zone: two markers, `reps` iterations of the body, two markers, then the
// four values into L1. The stores are OUTSIDE the second marker pair on purpose
// -- they are the harness's own cost and belong in `unattributed`, which is the
// bucket that exists to hold exactly this.
#define ZONE(ID, REPS, ...)                                                     \
    do {                                                                        \
        const uint32_t zn = (REPS);                                             \
        const uint32_t zc0 = rd_mcycle();                                       \
        const uint32_t zi0 = rd_minstret();                                     \
        for (uint32_t zr = 0; zr < zn; zr++) {                                  \
            __VA_ARGS__                                                         \
        }                                                                       \
        const uint32_t zi1 = rd_minstret();                                     \
        const uint32_t zc1 = rd_mcycle();                                       \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_C0] = zc0; \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_C1] = zc1; \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_I0] = zi0; \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_I1] = zi1; \
    } while (0)

// The calibration zone: the same four marker reads with NOTHING between the
// inner pair. Its cycle count IS the two-marker cost, and the analysis divides
// the smallest real zone by it rather than trusting the "~30 cycles" that would
// otherwise sit in this comment.
#define ZONE_NULL(ID)                                                           \
    do {                                                                        \
        const uint32_t zc0 = rd_mcycle();                                       \
        const uint32_t zi0 = rd_minstret();                                     \
        const uint32_t zi1 = rd_minstret();                                     \
        const uint32_t zc1 = rd_mcycle();                                       \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_C0] = zc0; \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_C1] = zc1; \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_I0] = zi0; \
        out[RETIREBENCH_HDR_WORDS + (ID) * RETIREBENCH_ZONE_WORDS + RETIREBENCH_Z_I1] = zi1; \
    } while (0)

void kernel_main() {
    const uint32_t result_addr = get_arg_val<uint32_t>(0);
    const uint32_t scratch_addr = get_arg_val<uint32_t>(1);
    const uint32_t scale = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_addr);

    // The L1 pointer chase: 64 nodes 16 bytes apart, each holding the address of
    // the next and the last wrapping to the first, so `lw rd, 0(rd)` costs
    // exactly one load latency per instruction with no second instruction in the
    // body to divide out. 1 KiB of working set against a documented 64-byte L0
    // data cache, so every link misses whatever the L0's organisation turns out
    // to be. riscvbench read 8.098 cycles on Blackhole silicon for this shape.
    {
        volatile tt_l1_ptr uint32_t* chain = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr);
        for (uint32_t i = 0; i < RETIREBENCH_CHASE_NODES; i++) {
            chain[i * (RETIREBENCH_CHASE_STRIDE / 4)] =
                scratch_addr + ((i + 1) % RETIREBENCH_CHASE_NODES) * RETIREBENCH_CHASE_STRIDE;
        }
    }

    // Operands, held in locals so the compiler allocates real registers. Every
    // body is `asm volatile`, so none of it can be deleted, hoisted, or
    // reordered across another body.
    uint32_t a0 = 1, a1 = 2, a2 = 3, a3 = 4;
    const uint32_t k = 3;
    // Two dividends, because the documented divide cost is a data dependence
    // ("between six and 33 cycles are required, dependent upon the magnitude of
    // the dividend") and one operand cannot stand for the band. 0xFFF is a
    // 12-bit dividend, the top of the range the in-tree Blackhole replay guards
    // actually execute; 0x12345678 is 29 significant bits, the operand
    // riscvbench read 33.001 cycles at. tt-sim charges its documented floor of 6
    // for both. THE TWO ZONES ARE THEREFORE A PREDICTION: div_small should be
    // close and div_large should not, and if a card says otherwise that is the
    // measurement, not a defect in either zone.
    const uint32_t dividend_small = 0x00000FFFu;
    const uint32_t dividend_large = 0x12345678u;
    uint32_t sink = 0;
    uint32_t toggle = 0;
    uint32_t chase = scratch_addr;
    const uint32_t data = scratch_addr + RETIREBENCH_DATA_OFFSET;
    (void)sink;

    const uint32_t wc0 = rd_mcycle();
    const uint32_t wi0 = rd_minstret();

    ZONE_NULL(RETIREBENCH_Z_MARKER_NULL);

    // One chain: every instruction reads the previous one's result. The docs say
    // the forwarding path makes this indistinguishable from four chains, which
    // is what the next zone checks.
    ZONE(RETIREBENCH_Z_ALU_DEP, RETIREBENCH_REPS_ALU_DEP * scale,
         REP64(asm volatile("addi %0, %0, 1" : "+r"(a0));));

    // Four independent chains, so the integer unit's forwarding path is not what
    // is being measured.
    ZONE(RETIREBENCH_Z_ALU_IND, RETIREBENCH_REPS_ALU_IND * scale,
         REP16(asm volatile("addi %0, %0, 1\n\t"
                            "addi %1, %1, 1\n\t"
                            "addi %2, %2, 1\n\t"
                            "addi %3, %3, 1"
                            : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3));));

    // Multiply, dependent and independent. On Blackhole the multiply pipelines
    // -- "exactly one cycle in EX1, and then exactly one cycle in EX2" -- which
    // is an occupancy of 1 and a result LATENCY of 2, so the pair separates the
    // two. Silicon: 1.985 dependent, 0.999 independent.
    ZONE(RETIREBENCH_Z_MUL_DEP, RETIREBENCH_REPS_MUL_DEP * scale,
         REP64(asm volatile("mul %0, %0, %1" : "+r"(a0) : "r"(k));));
    ZONE(RETIREBENCH_Z_MUL_IND, RETIREBENCH_REPS_MUL_IND * scale,
         REP16(asm volatile("mul %0, %0, %4\n\t"
                            "mul %1, %1, %4\n\t"
                            "mul %2, %2, %4\n\t"
                            "mul %3, %3, %4"
                            : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
                            : "r"(k));));

    ZONE(RETIREBENCH_Z_DIV_SMALL, RETIREBENCH_REPS_DIV_SMALL * scale,
         REP16(asm volatile("divu %0, %1, %2" : "=r"(sink) : "r"(dividend_small), "r"(k));));
    ZONE(RETIREBENCH_Z_DIV_LARGE, RETIREBENCH_REPS_DIV_LARGE * scale,
         REP16(asm volatile("divu %0, %1, %2" : "=r"(sink) : "r"(dividend_large), "r"(k));));

    // L1 load-use interlock, as a pointer chase so the body is one instruction
    // and the measured number is the latency undivided.
    ZONE(RETIREBENCH_Z_LOAD_DEP, RETIREBENCH_REPS_LOAD_DEP * scale,
         REP64(asm volatile("lw %0, 0(%0)" : "+r"(chase));));
    // L1 loads with no consumer: sustained throughput, not latency. Four
    // destination registers rotating means no load is ever read before the next
    // issues, so nothing here is an interlock.
    // Eight distinct 16-byte lines, which is twice the published L0 data cache
    // (64 bytes, 4 lines of 16), so no organisation of it can hold the working
    // set and every load takes the L1 path. Four destination registers rotating
    // twice is still four rotating registers: no load is ever read before the
    // next issues, so nothing here is an interlock either.
    ZONE(RETIREBENCH_Z_LOAD_IND, RETIREBENCH_REPS_LOAD_IND * scale,
         REP8(asm volatile("lw %0, 0(%4)\n\t"
                           "lw %1, 16(%4)\n\t"
                           "lw %2, 32(%4)\n\t"
                           "lw %3, 48(%4)\n\t"
                           "lw %0, 64(%4)\n\t"
                           "lw %1, 80(%4)\n\t"
                           "lw %2, 96(%4)\n\t"
                           "lw %3, 112(%4)"
                           : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                           : "r"(data));));
    // L1 stores to four DIFFERENT 16-byte-aligned blocks in rotation, so no two
    // consecutive stores can be coalesced under the documented predicate ("the
    // same 16-byte aligned region of L1, with start addresses within +/-4").
    // This is the "one store every five cycles" case.
    ZONE(RETIREBENCH_Z_STORE_SPREAD, RETIREBENCH_REPS_STORE_SPREAD * scale,
         REP16(asm volatile("sw %0, 0(%1)\n\t"
                            "sw %0, 16(%1)\n\t"
                            "sw %0, 32(%1)\n\t"
                            "sw %0, 48(%1)"
                            :
                            : "r"(a0), "r"(data));));

    // Control flow. The two zones execute the IDENTICAL dynamic instruction
    // sequence -- one `xori` and one conditional branch to a target one
    // instruction ahead, which is where the not-taken branch falls through to --
    // and differ in exactly one bit: whether the branch was taken. tt-sim
    // charges neither, and says so: the mispredict bubble is sourced (4 cycles
    // on Blackhole) but the predictor is undocumented, so the number of
    // mispredictions is unknowable and charging every taken branch would be a
    // fabrication. This pair is what puts a number on what that costs.
    ZONE(RETIREBENCH_Z_BRANCH_NT, RETIREBENCH_REPS_BRANCH_NT * scale,
         REP64(asm volatile("xori %0, %0, 1\n\t"
                            "bne %1, %1, 1f\n1:"
                            : "+r"(toggle)
                            : "r"(a0));));
    ZONE(RETIREBENCH_Z_BRANCH_T, RETIREBENCH_REPS_BRANCH_T * scale,
         REP64(asm volatile("xori %0, %0, 1\n\t"
                            "beq %1, %1, 1f\n1:"
                            : "+r"(toggle)
                            : "r"(a0));));

    const uint32_t wi1 = rd_minstret();
    const uint32_t wc1 = rd_mcycle();

    out[RETIREBENCH_HDR_WINDOW_C0] = wc0;
    out[RETIREBENCH_HDR_WINDOW_C1] = wc1;
    out[RETIREBENCH_HDR_WINDOW_I0] = wi0;
    out[RETIREBENCH_HDR_WINDOW_I1] = wi1;
    out[RETIREBENCH_HDR_NUM_ZONES] = RETIREBENCH_NUM_ZONES;
    out[RETIREBENCH_HDR_SCALE] = scale;
    out[RETIREBENCH_HDR_VERSION] = RETIREBENCH_VERSION;
    // Written LAST, so a host that reads the buffer back and finds the magic
    // knows every other word was already written. A partially-written buffer
    // decodes as a plausible set of small numbers, which is the one failure this
    // artefact must not be able to present as a measurement.
    out[RETIREBENCH_HDR_MAGIC] = RETIREBENCH_MAGIC;

    // Keep every operand live to the end so nothing above can be dead-coded.
    // The zone bodies are all `asm volatile` and cannot be deleted regardless,
    // but the locals feeding them can be, and a body whose operand was folded to
    // a constant is a different instruction sequence than the one named.
    out[RETIREBENCH_SINK_WORD] = a0 + a1 + a2 + a3 + sink + toggle + chase;
}
