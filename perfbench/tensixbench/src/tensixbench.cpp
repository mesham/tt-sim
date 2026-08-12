// tensixbench -- a per-instruction cycle-cost measuring instrument for the
// Tenstorrent Tensix coprocessor.
//
// WHAT IT IS FOR. tt-sim's Tensix instruction cost table
// (tt_sim/pe/tensix/tensix_instruction_costs.yaml) is well sourced -- every
// entry traces to the public ISA documentation or to vendor source -- but
// provenance is not validation, and nothing has ever compared it to a
// measurement. Rungs 1 and 2 of the calibration ladder in
// docs/plans/cost-model.md validated the NoC and memory path only. This program
// is the instrument for the missing rung: run it on real silicon, run the SAME
// BINARY against tt-sim, and diff the cycles.
//
// The methodology, the confounds, and what each measurement can and cannot
// establish are in docs/plans/tensix-cost-benchmark.md. The short version:
//
//   * Every reported cost is a SLOPE over several instruction counts, never a
//     single absolute measurement, so kernel launch, zone entry, loop setup and
//     synchronisation all cancel exactly.
//   * Phase A times unrolled TTI_* bursts, one `.ttinsn` per issue slot, and
//     subtracts an empty-body control loop to cancel the RISC-V loop overhead.
//     It is repeated with one, two and three issuing TRISCs so an issue-limited
//     result can be told apart from a unit-limited one. `--dvalid-once`
//     (default) / `--dvalid-per-thread` / `--dvalid-unpacr-nop` selects how the
//     three MATH probes get their SrcA/SrcB valid bits; the first two are
//     experiment X1 of docs/plans/matrix-unit-thread-contention.md and the third
//     is X2, which additionally makes the source data format a runtime axis
//     (`--src-format`). That choice is the only thing that changes what phase A
//     measures.
//   * Phase B times `matmul_tiles` at three math fidelities. The absolute
//     number is a confounded composite; the DIFFERENCE between fidelities is
//     not, and is a direct check on `fidelity_phases.mvmuls_per_tile`. The
//     fidelity setting is known to reach the math thread -- see the disassembly
//     evidence in kernels/compute/matmul_fidelity.cpp -- so a null difference
//     means the loop was not math-bound, which is a result and not a fault.
//
//   * `--vis-reps N` adds the one reading in this program that is NOT a slope,
//     and it is not one because there is nothing to fit: RDCFG's documented
//     ">= 2" is a LATENCY TO A GPR with the occupancy at 1 (ConfigurationUnit.md
//     tabulates both), the issuing thread is not blocked, and the hardware does
//     not interlock the read-after-write -- "Software must ensure that the
//     instruction(s) immediately after `RDCFG` are not trying to consume the
//     GPR". So a consumer placed too close reads the STALE value rather than
//     waiting, and the quantity is the smallest producer-to-consumer SEPARATION
//     at which the fresh value appears. Two card runs (2026-08-09 and
//     2026-08-12) established by measurement that no stall can reach it; slots
//     20-25 are kept as those two documented negatives. Carries no timer, so
//     there is no overhead for an intercept to absorb, and no CSV row.
//
// THE VERDICT IS PER PHASE. Phase A and phase B fail independently, print
// independent `TTBENCH_VALID_A:`/`TTBENCH_VALID_B:` lines, and set independent
// bits in the exit status (1 = A, 2 = B). A phase B that measures nothing must
// not throw away a phase A that measured everything.
//
// TIMESTAMPS come from RISCV_DEBUG_REG_WALL_CLOCK_L (0xFFB121F0), which is
// exactly the register tt-metal's device profiler reads for DeviceZoneScopedN
// (tt_metal/tools/profiler/kernel_profiler.hpp). Reading it directly rather
// than going through the profiler keeps the measurement identical on silicon
// and on tt-sim, needs no Tracy build, and sidesteps the profiler's dependence
// on a device AICLK that a simulator does not have.
//
// OUTPUT is a CSV of raw (probe, threads, blocks, cycles) points, plus a
// human-readable summary. Nothing here fits or reports a cost model number; the
// comparison against the tables is tt_sim/perf/tensix_bench_sweep.py.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "kernels/compute/bench_layout.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// ---------------------------------------------------------------------------
// The probe table. Slot order is the contract with kernels/compute/raw_probes.cpp.
// `unit` is the `ex_resource` key the instruction carries in
// tt_sim/pe/tensix/tensix_instructions.yaml, so the analysis script can look
// the cost up without a second mapping.
// ---------------------------------------------------------------------------
struct Probe {
    const char* name;
    const char* unit;
};

const Probe PROBES[TTBENCH_NUM_PROBES] = {
    {"loop_overhead", "-"},   // 0  control: identical loop, empty body
    {"NOP", "NONE"},          // 1
    {"DMANOP", "TDMA"},       // 2
    {"SFPNOP", "SFPU"},       // 3
    {"SFPMOV", "SFPU"},       // 4
    {"SFPADD", "SFPU"},       // 5
    {"SFPMUL", "SFPU"},       // 6
    {"SFPMAD", "SFPU"},       // 7
    {"SFPABS", "SFPU"},       // 8
    {"SETDMAREG", "THCON"},   // 9
    {"ADDDMAREG", "THCON"},   // 10
    {"MULDMAREG", "THCON"},   // 11
    {"SHIFTDMAREG", "THCON"}, // 12
    {"CMPDMAREG", "THCON"},   // 13
    {"RDCFG", "CFG"},         // 14
    {"SETRWC", "MATH"},       // 15
    {"INCRWC", "MATH"},       // 16
    {"MVMUL", "MATH"},        // 17
    {"ELWADD", "MATH"},       // 18
    {"ELWMUL", "MATH"},       // 19
    // The latency-difference pair. APPENDED, never inserted: `probe_id` is a
    // column of every CSV already collected and bit `i` of every `--probes`
    // mask in the runbook, so slots 0-19 keep their numbers.
    //
    // Both bodies are an op PLUS an identical TTI_STALLWAIT, so `cyc/instr`
    // below is cycles per PAIR rather than per instruction -- the `unit` column
    // says `CFG-LAT` / `THCON-LAT` rather than `CFG` / `THCON` so the summary
    // cannot be read as an occupancy. The reading is the DIFFERENCE of the two
    // and nothing else; neither row means anything alone.
    //
    // The names deliberately do not match any entry in the shipped cost tables.
    // `attach_table` looks a probe up by name, and a second series called
    // `RDCFG` would have merged with slot 14's under the same series key while
    // measuring a different quantity.
    {"RDCFG_STALL", "CFG-LAT"},     // 20  RDCFG      + STALLWAIT(THREAD, TRISC_CFG)
    {"SETDMA_STALL", "THCON-LAT"},  // 21  SETDMAREG  + the identical STALLWAIT
    // The corrected construction, appended for the same reason. Slots 20 and 21
    // stall on TRISC_CFG, which is condition C10 on Blackhole and C13 on
    // Wormhole and is about the RISCV core's outstanding memory requests, not
    // about the Configuration Unit's pipeline -- a card measured their
    // difference at 0.0000 cycles/pair on 2026-08-09 while the stall itself
    // provably cost cycles. These four stall on Blackhole's C12
    // (`p_stall::CFGEXU`), the one documented condition on either architecture
    // that observes the unit RDCFG runs on. See "THE LATENCY DIFFERENCE, DONE
    // PROPERLY" in kernels/compute/raw_probes.cpp.
    {"RDCFG_CFGSTALL", "CFG-LAT"},     // 22  RDCFG     + STALLWAIT(THREAD, CFGEXU)
    {"RMWCIB_CFGSTALL", "CFG-LAT"},    // 23  RMWCIB0   + the identical STALLWAIT
    {"SETDMA_CFGSTALL", "THCON-LAT"},  // 24  SETDMAREG + the identical STALLWAIT
    // Bare, and named as the cost tables name it, so the ordinary summary row
    // and the rung-3 sweep pick it up as an occupancy series like any other.
    {"RMWCIB0", "CFG"},                // 25  RMWCIB0 bare
    // Appended after a Blackhole card measured slots 22-25 at 2.9682 cycles per
    // pair in all three arms -- identical to four decimal places -- on
    // 2026-08-12. Slots 22-25 keep their numbers and their meaning; they are the
    // evidence that a busy-condition cannot see this quantity, which is a result
    // and not dead weight.
    //
    // 26/27 are the same instruction pair differing only in which GPR the
    // consumer reads, so their difference is a read-after-write and nothing
    // else. `CFG-DEP` rather than `CFG`, for the same reason slots 20-24 are
    // labelled `-LAT`: the reported figure is cycles per PAIR and must never be
    // read off as an occupancy.
    {"RDCFG_DEP", "CFG-DEP"},          // 26  RDCFG -> GPR60 ; ADDDMAREG reads GPR60
    {"RDCFG_INDEP", "CFG-DEP"},        // 27  RDCFG -> GPR60 ; ADDDMAREG reads GPR59
    // 28/29 have THREAD-DEPENDENT bodies and are graded across variants, not
    // within one. See TTBENCH_P_C12_XT in bench_layout.h.
    {"C12_XTHREAD", "CFG-C12"},        // 28  thr1: STALLWAIT(C12)+NOP; others: RMWCIB0+NOP
    {"C12_XTHREAD_NULL", "CFG-C12"},   // 29  thr1: NOP+NOP;             others: RMWCIB0+NOP
};

// Probes that need the SrcA/SrcB data-valid bits set. Reported so the operator
// knows which mask bits to clear if the run hangs.
constexpr uint32_t DVALID_PROBE_MASK = (1u << 17) | (1u << 18) | (1u << 19);
constexpr uint32_t ALL_PROBES = (TTBENCH_NUM_PROBES >= 32) ? 0xFFFFFFFFu : ((1u << TTBENCH_NUM_PROBES) - 1u);

struct Row {
    std::string phase;
    std::string variant;
    int probe_id;
    std::string probe;
    std::string unit;
    int active_threads;
    int thread;
    uint32_t n;       // blocks (phase A) or matmul_tiles iterations (phase B)
    uint32_t unroll;  // instructions per block (phase A) or 1 (phase B)
    uint32_t cycles;
};

struct Fit {
    double slope;
    double intercept;
    double r2;
};

Fit least_squares(const std::vector<double>& xs, const std::vector<double>& ys) {
    const size_t n = xs.size();
    double mx = 0, my = 0;
    for (size_t i = 0; i < n; i++) {
        mx += xs[i];
        my += ys[i];
    }
    mx /= n;
    my /= n;
    double sxy = 0, sxx = 0, syy = 0;
    for (size_t i = 0; i < n; i++) {
        sxy += (xs[i] - mx) * (ys[i] - my);
        sxx += (xs[i] - mx) * (xs[i] - mx);
        syy += (ys[i] - my) * (ys[i] - my);
    }
    Fit f{};
    f.slope = sxx == 0 ? 0.0 : sxy / sxx;
    f.intercept = my - f.slope * mx;
    f.r2 = (sxx == 0 || syy == 0) ? 1.0 : (sxy * sxy) / (sxx * syy);
    return f;
}

const char* arch_name(IDevice* device) {
    switch (device->arch()) {
        case tt::ARCH::BLACKHOLE: return "blackhole";
        case tt::ARCH::WORMHOLE_B0: return "wormhole";
        default: return "unknown";
    }
}

int popcount(uint32_t v) {
    int n = 0;
    while (v) {
        n += v & 1;
        v >>= 1;
    }
    return n;
}

// ---------------------------------------------------------------------------
// The source data format axis (experiment X2). Only reachable with the
// UNPACR_NOP dvalid setup, because that is the only setup which gives the
// Matrix Unit a DEFINED source format to vary: a bare SETDVALID leaves
// `ImpliedSrc{A,B}Fmt` an `UnpredictableValue()` on Blackhole, and that field is
// what the FPU reads there.
//
// The `style` column is the three-way `SrcAStyle` the MVMUL/ELWADD/ELWMUL
// functional models reduce the format to. It is recorded because it is the
// documented prediction: two formats that share a style are predicted to be
// EXACTLY indistinguishable, which makes bf16-vs-fp32 a null control rather than
// a comparison.
// ---------------------------------------------------------------------------
struct SrcFormat {
    const char* name;
    uint32_t code;  // tt::DataFormat, and the 4-bit hardware field value
    const char* style;
};

const SrcFormat SRC_FORMATS[] = {
    {"bf16", 5, "BF16"},  // tt::DataFormat::Float16_b
    {"fp32", 0, "BF16"},  // tt::DataFormat::Float32 -- same style as bf16
    {"tf32", 4, "TF32"},  // tt::DataFormat::Tf32
    {"fp16", 1, "FP16"},  // tt::DataFormat::Float16
};

const SrcFormat* find_src_format(const std::string& name) {
    for (const auto& f : SRC_FORMATS) {
        if (name == f.name) {
            return &f;
        }
    }
    return nullptr;
}

std::string src_format_names() {
    std::string out;
    for (const auto& f : SRC_FORMATS) {
        out += (out.empty() ? "" : ",");
        out += f.name;
    }
    return out;
}

// Is `name` one of the comma-separated entries of `list`? Used for both the
// phase A variant filter and the phase B fidelity filter, so the two selectors
// accept exactly the same spelling.
bool selected(const std::string& list, const std::string& name) {
    return ("," + list + ",").find("," + name + ",") != std::string::npos;
}

// The three dvalid setups, in the order their numeric values take. The name is
// what goes into the CSV header's `dvalid_setup=` token and is how a dataset
// tells itself apart from the others.
const char* dvalid_setup_name(uint32_t mode) {
    switch (mode) {
        case TTBENCH_DVALID_PER_THREAD: return "per-thread";
        case TTBENCH_DVALID_ONCE: return "once";
        case TTBENCH_DVALID_UNPACR_NOP: return "unpacr-nop";
        default: return "unknown";
    }
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t base_blocks = 4;
    uint32_t base_iters = 8;
    uint32_t probe_mask = ALL_PROBES;
    std::string phases = "ab";
    std::string out_path;
    // Experiment X1 of docs/plans/matrix-unit-thread-contention.md. Default is
    // the de-confounded setup: exactly one SETDVALID regardless of thread count.
    uint32_t dvalid_mode = TTBENCH_DVALID_ONCE;
    // Experiment X2, ibid. Only meaningful with the UNPACR_NOP setup; left null
    // otherwise, because with a bare SETDVALID there is no defined format to
    // name and printing one would be a lie in the header.
    const SrcFormat* src_format = nullptr;
    // Which phase B fidelities to launch. On hardware, all three, always. On the
    // simulator a single fidelity is minutes and the three together have never
    // completed, so being able to ask for one is the difference between phase B
    // being checkable at all and not.
    std::string fidelity_filter = "LoFi,HiFi2,HiFi4";
    // Which phase A thread sets to launch, same shape as --fidelities and for
    // the same reason: a run that only needs one of them should not have to run
    // -- or survive -- the others. Experiment X2's measurement is a t1
    // comparison ACROSS FORMATS, so it needs t1 and nothing else, and on
    // Blackhole silicon the UNPACR_NOP setup hangs the t2 launch (see
    // "The UNPACR_NOP setup hangs at t2 on silicon" in the README). Selecting
    // the variants makes the measurement reachable without touching the hang.
    std::string variant_filter = "t1,t2,t3";
    // Repetitions of the RDCFG visibility sweep. Off by default: it is not a
    // slope, it writes no CSV row, and every existing recipe should keep
    // producing exactly the bytes it produced before this flag existed.
    uint32_t vis_reps = 0;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--blocks") {
            base_blocks = std::stoul(next());
        } else if (a == "--iters") {
            base_iters = std::stoul(next());
        } else if (a == "--probes") {
            probe_mask = std::stoul(next(), nullptr, 0) & ALL_PROBES;
        } else if (a == "--no-dvalid-probes") {
            probe_mask &= ~DVALID_PROBE_MASK;
        } else if (a == "--dvalid-per-thread") {
            dvalid_mode = TTBENCH_DVALID_PER_THREAD;
        } else if (a == "--dvalid-once") {
            dvalid_mode = TTBENCH_DVALID_ONCE;
        } else if (a == "--dvalid-unpacr-nop") {
            dvalid_mode = TTBENCH_DVALID_UNPACR_NOP;
        } else if (a == "--src-format") {
            const std::string name = next();
            src_format = find_src_format(name);
            if (src_format == nullptr) {
                fprintf(
                    stderr,
                    "unknown --src-format %s (want one of %s)\n",
                    name.c_str(),
                    src_format_names().c_str());
                return 2;
            }
        } else if (a == "--phase") {
            phases = next();
        } else if (a == "--fidelities") {
            fidelity_filter = next();
        } else if (a == "--variants") {
            variant_filter = next();
        } else if (a == "--vis-reps") {
            vis_reps = std::stoul(next());
        } else if (a == "--out") {
            out_path = next();
        } else if (a == "-h" || a == "--help") {
            printf(
                "usage: tensixbench [--blocks N] [--iters N] [--probes 0xMASK]\n"
                "                   [--no-dvalid-probes] [--phase a|b|ab] [--variants LIST]\n"
                "                   [--fidelities LIST] [--vis-reps N] [--out FILE]\n"
                "                   [--dvalid-once | --dvalid-per-thread |\n"
                "                    --dvalid-unpacr-nop [--src-format NAME]]\n"
                "\n"
                "  --blocks N            phase A sweeps blocks = N, 2N, 3N, 4N of %d\n"
                "                        instructions each (default 4)\n"
                "  --iters N             phase B sweeps N, 2N, 3N, 4N matmul_tiles calls\n"
                "                        (default 8)\n"
                "  --probes 0xMASK       bit i enables probe i (default all %d)\n"
                "  --no-dvalid-probes    drop MVMUL/ELWADD/ELWMUL, the only probes that\n"
                "                        depend on the SrcA/SrcB valid bits. Use this if\n"
                "                        phase A hangs, and say so in the report.\n"
                "  --dvalid-once         (default) exactly one SETDVALID before the three\n"
                "                        MATH probes, issued by thread 1, barriered. The\n"
                "                        de-confounded setup -- experiment X1 of\n"
                "                        docs/plans/matrix-unit-thread-contention.md.\n"
                "  --dvalid-per-thread   the original setup: one SETDVALID per ACTIVE\n"
                "                        thread, so the thread count and the SrcA/SrcB\n"
                "                        bank state move together. Run this to reproduce\n"
                "                        the 6.1x/12.1x MVMUL result and diff it against\n"
                "                        the default. Written to a distinct default CSV.\n"
                "  --dvalid-unpacr-nop   the Blackhole-SANCTIONED setup: one UNPACR_NOP\n"
                "                        carrying set_dvalid per unpacker, from thread 1,\n"
                "                        barriered. Unlike SETDVALID it leaves both banks\n"
                "                        a DEFINED ImpliedSrc{A,B}Fmt, taken from\n"
                "                        THCON_SEC*_REG2_Out_data_format -- which is what\n"
                "                        makes --src-format mean anything. Experiment X2\n"
                "                        of docs/plans/matrix-unit-thread-contention.md.\n"
                "  --src-format NAME     source data format for the MATH probes, one of\n"
                "                        %s. REQUIRES --dvalid-unpacr-nop:\n"
                "                        with a bare SETDVALID the format is\n"
                "                        UnpredictableValue on Blackhole and there is\n"
                "                        nothing well defined to vary. Each format gets\n"
                "                        its own default CSV name.\n"
                "  --phase a|b|ab        which phases to run (default ab)\n"
                "  --variants LIST       comma-separated subset of t1,t2,t3 for phase A\n"
                "                        (default all three), i.e. how many TRISCs issue\n"
                "                        the identical burst. Each is a separate program\n"
                "                        launch, so dropping one drops its launch entirely.\n"
                "                        `--variants t1` is what experiment X2 needs: a\n"
                "                        format comparison is a SINGLE-thread quantity, and\n"
                "                        the t2/t3 launches of --dvalid-unpacr-nop are known\n"
                "                        to hang on Blackhole silicon (see the README).\n"
                "  --fidelities LIST     comma-separated subset of LoFi,HiFi2,HiFi4 for\n"
                "                        phase B (default all three). For the simulator,\n"
                "                        where one fidelity is minutes; on hardware leave\n"
                "                        it alone -- the DIFFERENCE needs at least two.\n"
                "  --vis-reps N          repetitions of the RDCFG VISIBILITY sweep, the\n"
                "                        probe that reads the documented '>= 2' as a\n"
                "                        producer-to-consumer DISTANCE rather than as a\n"
                "                        duration (default 0 = off). Runs on the t1 launch\n"
                "                        only, issues no STALLWAIT, writes no cycle count\n"
                "                        and appears in no CSV row -- see TTBENCH_VIS_* in\n"
                "                        kernels/compute/bench_layout.h. 64 is plenty.\n"
                "  --out FILE            CSV path (default tensixbench-<arch>.csv)\n",
                TTBENCH_UNROLL,
                TTBENCH_NUM_PROBES,
                src_format_names().c_str());
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s (try --help)\n", a.c_str());
            return 2;
        }
    }
    if (base_blocks == 0 || base_iters == 0) {
        fprintf(stderr, "--blocks and --iters must be >= 1\n");
        return 2;
    }
    // A typo in --variants must not quietly run nothing, or -- worse -- write a
    // CSV whose header claims a selection the data does not contain.
    if (!selected(variant_filter, "t1") && !selected(variant_filter, "t2") && !selected(variant_filter, "t3")) {
        fprintf(stderr, "--variants %s selects none of t1,t2,t3\n", variant_filter.c_str());
        return 2;
    }
    // A format is only a *measurable* axis under the UNPACR_NOP setup. Refusing
    // the combination rather than quietly ignoring it is the point: a CSV
    // labelled `src_format=fp32` whose FPU actually decoded an
    // `UnpredictableValue()` would be worse than no CSV at all.
    if (src_format != nullptr && dvalid_mode != TTBENCH_DVALID_UNPACR_NOP) {
        fprintf(
            stderr,
            "--src-format needs --dvalid-unpacr-nop. A bare SETDVALID leaves\n"
            "ImpliedSrc{A,B}Fmt an UnpredictableValue() on Blackhole, and that is the\n"
            "field the Matrix Unit reads, so there would be no defined format to vary.\n"
            "See docs/plans/matrix-unit-thread-contention.md, experiment X2.\n");
        return 2;
    }
    // Under the UNPACR_NOP setup a format is always programmed -- there is no
    // "leave it alone" option, because the instruction copies whatever is in
    // THCON_SEC*_REG2_Out_data_format either way. bf16 is the default because it
    // is what phase B and every example in this repository use.
    if (dvalid_mode == TTBENCH_DVALID_UNPACR_NOP && src_format == nullptr) {
        src_format = find_src_format("bf16");
    }

    IDevice* device = CreateDevice(0);
    const std::string arch = arch_name(device);
    if (out_path.empty()) {
        // Every non-default configuration gets its own name so that runs which
        // measure different things cannot silently overwrite each other.
        std::string suffix;
        if (dvalid_mode == TTBENCH_DVALID_PER_THREAD) {
            suffix = "-dvalid-per-thread";
        } else if (dvalid_mode == TTBENCH_DVALID_UNPACR_NOP) {
            suffix = std::string("-unpacr-nop-") + src_format->name;
        }
        out_path = "tensixbench-" + arch + suffix + ".csv";
    }

    // The four config-latency slots need Blackhole's STALLWAIT condition C12,
    // `p_stall::CFGEXU`. Wormhole's condition mask has no bit for the
    // Configuration Unit -- its C0-C14 are the ThCon memory request, the two
    // unpackers, the four packers, the FPU, the four Src-bank clients, the
    // mover, the RISCV memory request and the SFPU (WormholeB0 STALLWAIT.md) --
    // and its `p_stall` has no CFGEXU to emit, so the kernel compiles those
    // slots to a skip there. Clearing them from the mask HERE is what keeps that
    // honest: a skipped slot writes four zero cycle counts, which is a flat
    // series that would fail the monotonicity check and condemn all of phase A.
    //
    // This is not a hole in the coverage. `WormholeB0/.../RDCFG.md` says "The
    // issuing thread is blocked for the entire duration", so on Wormhole the
    // documented ">= 2" is an OCCUPANCY and slot 14 already reaches it; it is
    // Blackhole, where "The issuing thread is not blocked", that needs a stall
    // to see it at all.
    const bool cfglat_supported = (arch == "blackhole");
    const bool cfglat_asked = (probe_mask & TTBENCH_CFGLAT_PROBE_MASK) != 0;
    if (!cfglat_supported && cfglat_asked) {
        printf(
            "note: dropping probes 22-25 and 28-29 (the C12 slots) on %s.\n"
            "  They stall on STALLWAIT condition C12, \"Any thread has an instruction in\n"
            "  any stage of the Configuration Unit pipeline\", which only BlackholeA0\n"
            "  defines. Wormhole's fifteen condition bits name no unit RDCFG runs on --\n"
            "  and do not need to: WormholeB0/RDCFG.md blocks the issuing thread for the\n"
            "  whole instruction, so there the documented \">= 2\" is an occupancy and\n"
            "  probe 14 already measures it.\n",
            arch.c_str());
        probe_mask &= ~TTBENCH_CFGLAT_PROBE_MASK;
    }

    constexpr CoreCoord core = {0, 0};

    // Reserve L1 through the allocator rather than picking an address, so this
    // cannot collide with the firmware or the circular buffers.
    constexpr uint32_t result_bytes = TTBENCH_RESULT_WORDS * 4;
    InterleavedBufferConfig l1_cfg{
        .device = device, .size = result_bytes, .page_size = result_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> result_scratch = CreateBuffer(l1_cfg);
    const uint32_t results_addr = result_scratch->address();

    constexpr uint32_t barrier_bytes = TTBENCH_MAX_THREADS * 4;
    InterleavedBufferConfig bar_cfg{
        .device = device, .size = 64, .page_size = 64, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> barrier_scratch = CreateBuffer(bar_cfg);
    const uint32_t barrier_addr = barrier_scratch->address();

    std::vector<Row> rows;
    // The RDCFG visibility region, harvested from whichever launch wrote it.
    // Empty means the sweep was not asked for or did not run.
    std::vector<uint32_t> vis;
    // The verdict is PER PHASE. A phase A run can be perfect and a phase B run
    // useless in the same process -- that is exactly what the first Blackhole
    // run was -- and a single global verdict threw away nineteen good series
    // because of one bad composite. Each phase now stands or falls alone, in the
    // printed summary and in the exit status.
    int fail_a = 0;
    int fail_b = 0;

    // Rewritten after every program launch, not once at the end. A launch is
    // seconds on silicon but can be tens of minutes against the simulator (a
    // HiFi4 matmul loop is 64 MVMULs per call through a Python FPU), and a run
    // that has to be killed part-way should still leave the phases that did
    // finish on disk rather than nothing at all.
    auto write_csv = [&]() {
        FILE* csv = fopen(out_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "cannot open %s for writing\n", out_path.c_str());
            return false;
        }
        fprintf(csv, "# tensixbench raw points -- see docs/plans/tensix-cost-benchmark.md\n");
        // Every token here is `key=value` because the analysis harness
        // (tt_sim/perf/tensix_bench_sweep.read_csv) harvests them into `meta`.
        // `src_format` is `undefined` rather than absent when no format was
        // programmed: under a bare SETDVALID the format the FPU decodes really
        // is undefined on Blackhole, and saying so is the honest header.
        fprintf(
            csv,
            "# arch=%s magic=0x%08X unroll=%u probe_mask=0x%X dvalid_setup=%s "
            "src_format=%s src_style=%s mm_block=%u variants=%s fidelities=%s\n",
            arch.c_str(),
            TTBENCH_MAGIC,
            TTBENCH_UNROLL,
            probe_mask,
            dvalid_setup_name(dvalid_mode),
            src_format ? src_format->name : "undefined",
            src_format ? src_format->style : "undefined",
            TTBENCH_MM_BLOCK,
            variant_filter.c_str(),
            fidelity_filter.c_str());
        fprintf(csv, "phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles\n");
        for (const auto& r : rows) {
            fprintf(
                csv,
                "%s,%s,%d,%s,%s,%d,%d,%u,%u,%u\n",
                r.phase.c_str(),
                r.variant.c_str(),
                r.probe_id,
                r.probe.c_str(),
                r.unit.c_str(),
                r.active_threads,
                r.thread,
                r.n,
                r.unroll,
                r.cycles);
        }
        fclose(csv);
        return true;
    };

    auto read_results = [&](std::vector<uint32_t>& words) {
        words.assign(TTBENCH_RESULT_WORDS, 0);
        detail::ReadFromDeviceL1(device, core, results_addr, result_bytes, words);
    };

    // -----------------------------------------------------------------------
    // Phase A: raw TTI_* issue throughput, at one, two and three issuing TRISCs.
    // -----------------------------------------------------------------------
    if (phases.find('a') != std::string::npos) {
        const struct {
            const char* variant;
            uint32_t mask;
        } thread_sets[] = {
            {"t1", 0x2},  // math thread alone
            {"t2", 0x3},  // unpack + math
            {"t3", 0x7},  // all three
        };

        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                printf("phase A [%s]: skipped (--variants %s)\n", ts.variant, variant_filter.c_str());
                fflush(stdout);
                continue;
            }
            Program program = CreateProgram();

            // A compute kernel needs at least one circular buffer for the
            // generated CB descriptor list to be well formed. Nothing reads it.
            CircularBufferConfig dummy_cb =
                CircularBufferConfig(1024, {{CBIndex::c_0, tt::DataFormat::UInt32}}).set_page_size(CBIndex::c_0, 1024);
            CreateCircularBuffer(program, core, dummy_cb);

            KernelHandle bench = CreateKernel(
                program, "kernels/compute/raw_probes.cpp", core, ComputeConfig{});
            SetRuntimeArgs(
                program,
                bench,
                core,
                {results_addr,
                 barrier_addr,
                 base_blocks,
                 probe_mask,
                 ts.mask,
                 dvalid_mode,
                 src_format ? src_format->code : 0u,
                 vis_reps});

            std::vector<uint32_t> zeros(16, 0);
            detail::WriteToDeviceL1(device, core, barrier_addr, zeros);
            std::vector<uint32_t> clear(TTBENCH_RESULT_WORDS, 0);
            detail::WriteToDeviceL1(device, core, results_addr, clear);

            detail::LaunchProgram(device, program, true, true);

            std::vector<uint32_t> words;
            read_results(words);
            if (words[TTBENCH_HDR_MAGIC] != TTBENCH_MAGIC) {
                fprintf(
                    stderr,
                    "FATAL: result header magic 0x%08X != 0x%08X -- the kernel did not run, "
                    "or the host and kernel disagree about the layout\n",
                    words[TTBENCH_HDR_MAGIC],
                    TTBENCH_MAGIC);
                CloseDevice(device);
                return 1;
            }
            // The visibility region, if this launch produced one. Kept rather
            // than re-read because only the t1 launch writes it and later
            // launches leave the region cleared.
            if (words[TTBENCH_VIS_BASE + TTBENCH_VIS_W_STAMP] == TTBENCH_VIS_STAMP) {
                vis.assign(
                    words.begin() + TTBENCH_VIS_BASE,
                    words.begin() + TTBENCH_VIS_BASE + TTBENCH_VIS_WORDS);
            }
            const int active = popcount(ts.mask);
            for (int t = 0; t < TTBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                const uint32_t* slot = words.data() + TTBENCH_HDR_WORDS + t * TTBENCH_NUM_PROBES * TTBENCH_NUM_POINTS;
                for (int p = 0; p < TTBENCH_NUM_PROBES; p++) {
                    if ((probe_mask & (1u << p)) == 0) {
                        continue;
                    }
                    for (int k = 0; k < TTBENCH_NUM_POINTS; k++) {
                        rows.push_back(Row{
                            "A",
                            ts.variant,
                            p,
                            PROBES[p].name,
                            PROBES[p].unit,
                            active,
                            t,
                            base_blocks * (uint32_t)(k + 1),
                            TTBENCH_UNROLL,
                            slot[p * TTBENCH_NUM_POINTS + k]});
                    }
                }
            }
            write_csv();
            printf(
                "phase A [%s]: done (%d issuing thread%s), %zu rows written to %s\n",
                ts.variant,
                active,
                active == 1 ? "" : "s",
                rows.size(),
                out_path.c_str());
            fflush(stdout);
        }
    }

    // -----------------------------------------------------------------------
    // Phase B: matmul_tiles at three math fidelities.
    // -----------------------------------------------------------------------
    if (phases.find('b') != std::string::npos) {
        const struct {
            const char* variant;
            tt::tt_metal::MathFidelity fidelity;
        } fidelities[] = {
            {"LoFi", tt::tt_metal::MathFidelity::LoFi},
            {"HiFi2", tt::tt_metal::MathFidelity::HiFi2},
            {"HiFi4", tt::tt_metal::MathFidelity::HiFi4},
        };
        constexpr uint32_t tile_bytes = 32 * 32 * 2;  // one bf16 tile

        for (const auto& f : fidelities) {
            if (!selected(fidelity_filter, f.variant)) {
                printf("phase B [%s]: skipped (--fidelities %s)\n", f.variant, fidelity_filter.c_str());
                fflush(stdout);
                continue;
            }
            Program program = CreateProgram();

            // Operand buffers hold two blocks so the feeder can run a whole
            // block ahead of the compute loop; the compute kernel waits on
            // TTBENCH_MM_BLOCK pages at a time, so anything less than one block
            // deadlocks. The output buffer holds one page per measurement point
            // so nothing has to drain it mid-run.
            constexpr uint32_t in_pages = 2 * TTBENCH_MM_BLOCK;
            CircularBufferConfig cb0 =
                CircularBufferConfig(in_pages * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
                    .set_page_size(CBIndex::c_0, tile_bytes);
            CreateCircularBuffer(program, core, cb0);
            CircularBufferConfig cb1 =
                CircularBufferConfig(in_pages * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
                    .set_page_size(CBIndex::c_1, tile_bytes);
            CreateCircularBuffer(program, core, cb1);
            CircularBufferConfig cb16 =
                CircularBufferConfig(TTBENCH_MM_NUM_POINTS * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
                    .set_page_size(CBIndex::c_16, tile_bytes);
            CreateCircularBuffer(program, core, cb16);

            KernelHandle feeder = CreateKernel(
                program,
                "kernels/dataflow/mm_feeder.cpp",
                core,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            SetRuntimeArgs(program, feeder, core, {base_iters});

            KernelHandle bench = CreateKernel(
                program,
                "kernels/compute/matmul_fidelity.cpp",
                core,
                ComputeConfig{.math_fidelity = f.fidelity});
            SetRuntimeArgs(program, bench, core, {results_addr, base_iters});

            std::vector<uint32_t> clear(TTBENCH_RESULT_WORDS, 0);
            detail::WriteToDeviceL1(device, core, results_addr, clear);

            detail::LaunchProgram(device, program, true, true);

            std::vector<uint32_t> words;
            read_results(words);
            if (words[TTBENCH_HDR_MAGIC] != TTBENCH_MAGIC) {
                fprintf(stderr, "FATAL: phase B header magic mismatch for %s\n", f.variant);
                CloseDevice(device);
                return 1;
            }
            // Threads 0 (unpack) and 1 (math) only. The pack thread's copy of
            // the inner loop is empty by construction -- `cb_wait_front`,
            // `cb_pop_front` and `matmul_tiles` all compile to nothing on
            // TRISC2 -- so it measured ~1 cycle at every iteration count,
            // failed linearity and monotonicity, and dragged an otherwise good
            // run's verdict down with it. The kernel no longer times it and
            // nothing is emitted for it.
            for (int t = 0; t < TTBENCH_MM_TIMED_THREADS; t++) {
                const uint32_t* slot = words.data() + TTBENCH_HDR_WORDS + t * TTBENCH_NUM_PROBES * TTBENCH_NUM_POINTS;
                for (int k = 0; k < TTBENCH_MM_NUM_POINTS; k++) {
                    rows.push_back(Row{
                        "B",
                        f.variant,
                        0,
                        "matmul_tiles",
                        "MATH",
                        3,
                        t,
                        base_iters * (uint32_t)(k + 1),
                        1,
                        slot[k]});
                }
            }
            write_csv();
            printf(
                "phase B [%s]: done, %zu rows written to %s\n",
                f.variant,
                rows.size(),
                out_path.c_str());
            fflush(stdout);
        }
    }

    CloseDevice(device);

    // The CSV was rewritten after every launch; this is the final one.
    if (!write_csv()) {
        return 1;
    }

    // -----------------------------------------------------------------------
    // Validity checks, so a degenerate run is caught here and not three days
    // later in the analysis. A run that trips any of these is not usable.
    // -----------------------------------------------------------------------
    printf("\n%s\n", std::string(78, '=').c_str());
    printf("tensixbench summary [%s] -- slopes only, no absolute measurement is a cost\n", arch.c_str());
    printf("%s\n", std::string(78, '=').c_str());
    if (phases.find('a') != std::string::npos) {
        const char* what = "one SETDVALID per ACTIVE thread (original, confounded)";
        if (dvalid_mode == TTBENCH_DVALID_ONCE) {
            what = "one SETDVALID for the tile (X1, de-confounded)";
        } else if (dvalid_mode == TTBENCH_DVALID_UNPACR_NOP) {
            what = "one UNPACR_NOP+set_dvalid per unpacker (X2, sanctioned)";
        }
        printf("dvalid setup: %s -- %s\n", dvalid_setup_name(dvalid_mode), what);
        if (src_format != nullptr) {
            printf(
                "source format: %s (code %u, SrcAStyle %s) -- programmed into\n"
                "  THCON_SEC{0,1}_REG2_Out_data_format and ALU_FORMAT_SPEC_REG{0_SrcA,1_SrcB}.\n"
                "  The ISA docs give the Matrix Unit 1 IPC with no format qualification, and\n"
                "  reduce every format to one of three SrcAStyles, so a cost difference\n"
                "  between two formats sharing a style would contradict the model outright\n"
                "  and a difference between styles is undocumented either way.\n",
                src_format->name,
                src_format->code,
                src_format->style);
        }
    }
    printf("\n");
    printf("%-14s %-7s %-9s %-4s %12s %10s %8s\n", "probe", "variant", "unit", "thr", "cyc/block", "cyc/instr", "R^2");

    struct Key {
        std::string phase, variant, probe;
        int thread;
    };
    // Slope of the empty-body control loop, per (variant, thread). Everything
    // in phase A is reported relative to it.
    std::vector<std::pair<std::string, double>> loop_slope;
    auto loop_key = [](const Row& r) { return r.variant + "/" + std::to_string(r.thread); };

    auto fit_for = [&](const std::string& phase, const std::string& variant, const std::string& probe, int thread) {
        std::vector<double> xs, ys;
        for (const auto& r : rows) {
            if (r.phase == phase && r.variant == variant && r.probe == probe && r.thread == thread) {
                xs.push_back(r.n);
                ys.push_back(r.cycles);
            }
        }
        return std::make_pair(xs.size() >= 2 ? least_squares(xs, ys) : Fit{0, 0, 0}, xs.size());
    };

    for (const auto& r : rows) {
        if (r.phase == "A" && r.probe_id == 0) {
            const std::string k = loop_key(r);
            bool seen = false;
            for (const auto& e : loop_slope) {
                if (e.first == k) {
                    seen = true;
                }
            }
            if (!seen) {
                loop_slope.emplace_back(k, fit_for("A", r.variant, "loop_overhead", r.thread).first.slope);
            }
        }
    }

    // Phase B slopes, kept so the fidelity deltas can be printed and read out
    // below without refitting.
    struct MMSlope {
        std::string variant;
        int thread;
        double slope;
    };
    std::vector<MMSlope> mm_slopes;

    std::vector<std::string> emitted;
    for (const auto& r : rows) {
        const std::string tag = r.phase + "/" + r.variant + "/" + r.probe + "/" + std::to_string(r.thread);
        bool seen = false;
        for (const auto& e : emitted) {
            if (e == tag) {
                seen = true;
            }
        }
        if (seen) {
            continue;
        }
        emitted.push_back(tag);

        auto [fit, n] = fit_for(r.phase, r.variant, r.probe, r.thread);
        double per_instr = 0;
        if (r.phase == "A") {
            double base = 0;
            const std::string k = loop_key(r);
            for (const auto& e : loop_slope) {
                if (e.first == k) {
                    base = e.second;
                }
            }
            per_instr = (fit.slope - base) / (double)TTBENCH_UNROLL;
        } else {
            per_instr = fit.slope;  // cycles per matmul_tiles call
            mm_slopes.push_back(MMSlope{r.variant, r.thread, fit.slope});
        }
        printf(
            "%-14s %-7s %-9s %-4d %12.2f %10.3f %8.4f%s\n",
            r.probe.c_str(),
            r.variant.c_str(),
            r.unit.c_str(),
            r.thread,
            fit.slope,
            per_instr,
            fit.r2,
            fit.r2 < 0.99 ? "  <-- NONLINEAR" : "");
        if (fit.r2 < 0.99) {
            (r.phase == "A" ? fail_a : fail_b)++;
        }
    }

    // Monotonicity: cycles must grow with n for every series.
    for (size_t i = 0; i + 1 < rows.size(); i++) {
        const Row& a = rows[i];
        const Row& b = rows[i + 1];
        if (a.phase == b.phase && a.variant == b.variant && a.probe == b.probe && a.thread == b.thread &&
            b.n > a.n && b.cycles <= a.cycles) {
            printf("  NOT MONOTONE: %s/%s/%s thread %d: n=%u -> %u cycles, n=%u -> %u cycles\n",
                   a.phase.c_str(), a.variant.c_str(), a.probe.c_str(), a.thread, a.n, a.cycles, b.n, b.cycles);
            (a.phase == "A" ? fail_a : fail_b)++;
        }
    }

    // -----------------------------------------------------------------------
    // The latency difference, read out. Every slot involved is (op + the
    // identical STALLWAIT), unrolled 64 times, so a slope is per BLOCK of 64
    // PAIRS and a per-pair figure is (slope - loop_slope) / TTBENCH_UNROLL.
    // Printing these as two more rows of the summary table above and leaving the
    // subtraction to the reader is how a per-pair figure gets quoted as a
    // per-instruction occupancy, so they are read out here instead.
    //
    // Everything the grader needs is also emitted as `TTBENCH_CFGLAT_*:` tag
    // lines. Those exist because the previous version of this section was graded
    // by scraping a five-field table out of the prose, and a table that gains a
    // column silently changes what the grader reads.
    // -----------------------------------------------------------------------
    auto loop_base = [&](const std::string& variant, int thread) {
        const std::string k = variant + "/" + std::to_string(thread);
        for (const auto& e : loop_slope) {
            if (e.first == k) {
                return e.second;
            }
        }
        return 0.0;
    };
    // Cycles per PAIR for a two-instruction slot, loop overhead removed, so the
    // printed absolutes can be read against the cycle-by-cycle prediction in
    // raw_probes.cpp and not just their difference. Returns false when the slot
    // was masked off or produced no fit.
    auto per_pair = [&](const std::string& variant, int thread, int slot, double* out) {
        auto [f, n] = fit_for("A", variant, PROBES[slot].name, thread);
        if (n == 0) {
            return false;
        }
        *out = (f.slope - loop_base(variant, thread)) / (double)TTBENCH_UNROLL;
        return true;
    };
    // The (variant, thread) keys this run actually produced for a given slot,
    // taken from the rows themselves rather than from a thread-set table that is
    // scoped to the launch loop -- so a `--variants` subset or a masked probe
    // yields fewer lines instead of empty ones.
    auto keys_for = [&](int slot) {
        std::vector<std::pair<std::string, int>> keys;
        for (const auto& r : rows) {
            if (r.phase != "A" || r.probe != PROBES[slot].name) {
                continue;
            }
            const auto key = std::make_pair(r.variant, r.thread);
            if (std::find(keys.begin(), keys.end(), key) == keys.end()) {
                keys.push_back(key);
            }
        }
        return keys;
    };

    // -----------------------------------------------------------------------
    // THE MEASUREMENT. Slots 22/23/24, stalling on Blackhole's condition C12 --
    // "Any thread has an instruction in any stage of the Configuration Unit
    // pipeline" -- which is the only documented condition on either
    // architecture that observes the unit `RDCFG` executes on.
    //
    // GRADED AT t1 ONLY, and this is not a convenience. C12 says ANY thread, and
    // STALLWAIT.md's own note for it says "This won't prevent other threads from
    // issuing new Configuration Unit instructions though, and those new
    // instructions will cause this thread to continue to wait." At t2/t3 every
    // active thread runs the identical burst, so each thread's stall observes
    // the others' RDCFGs and the difference stops being a statement about one
    // instruction. The other variants are printed and explicitly not graded.
    // -----------------------------------------------------------------------
    {
        const auto keys = keys_for(TTBENCH_P_RDCFG_CFGSTALL);
        if (!keys.empty()) {
            printf("\n%s\n", std::string(78, '-').c_str());
            printf("phase A: RDCFG latency by STALLWAIT on the Configuration Unit (C12/CFGEXU)\n");
            printf("%s\n", std::string(78, '-').c_str());
            printf(
                "  BlackholeA0 STALLWAIT.md, condition C12: \"Any thread has an instruction in\n"
                "  any stage of the Configuration Unit pipeline\", with block bit B7 (inside\n"
                "  STALL_THREAD) \"Block thread from starting new Configuration Unit\n"
                "  instructions\". BlackholeA0 RDCFG.md: \"After issuing one or more `RDCFG`\n"
                "  instructions, software is encouraged to use `STALLWAIT` to wait for the\n"
                "  Configuration Unit to no longer be busy.\"\n"
                "  Predicted from ConfigurationUnit.md's stage table: RDCFG holds stage 0 then\n"
                "  stage +1, RMWCIB holds stage 0 alone, so the difference is >= 1 cycle/pair.\n\n");
            printf(
                "  %-7s %-4s %11s %11s %11s %11s %11s\n",
                "variant", "thr", "rdcfg", "rmwcib", "setdma", "d(rmwcib)", "d(setdma)");
            bool have_t1 = false;
            double t1_rdcfg = 0, t1_rmw = 0, t1_setdma = 0;
            for (const auto& key : keys) {
                double a = 0, b = 0, c = 0;
                if (!per_pair(key.first, key.second, TTBENCH_P_RDCFG_CFGSTALL, &a) ||
                    !per_pair(key.first, key.second, TTBENCH_P_RMWCIB_CFGSTALL, &b) ||
                    !per_pair(key.first, key.second, TTBENCH_P_SETDMA_CFGSTALL, &c)) {
                    continue;
                }
                printf(
                    "  %-7s %-4d %11.3f %11.3f %11.3f %+11.3f %+11.3f%s\n",
                    key.first.c_str(), key.second, a, b, c, a - b, a - c,
                    key.first == "t1" ? "" : "   (not graded)");
                if (key.first == "t1" && !have_t1) {
                    have_t1 = true;
                    t1_rdcfg = a;
                    t1_rmw = b;
                    t1_setdma = c;
                }
            }
            if (!have_t1) {
                printf(
                    "\n  NO t1 SERIES. The difference is a single-thread quantity -- C12 is\n"
                    "  \"ANY thread\", so at t2/t3 each thread's stall observes the other\n"
                    "  threads' RDCFGs. Re-run with --variants t1.\n");
            } else {
                // The two bare occupancies the difference rests on. If RDCFG and
                // RMWCIB0 do not cost the same to ISSUE, part of (22 - 23) is
                // occupancy and none of it can be called latency.
                double occ_rdcfg = 0, occ_rmw = 0, occ_setdma = 0;
                const bool have_occ =
                    per_pair("t1", 1, 14, &occ_rdcfg) && per_pair("t1", 1, TTBENCH_P_RMWCIB, &occ_rmw) &&
                    per_pair("t1", 1, 9, &occ_setdma);
                printf("\n");
                if (have_occ) {
                    printf(
                        "  bare occupancies (must agree, or the difference is occupancy):\n"
                        "    RDCFG %.3f   RMWCIB0 %.3f   SETDMAREG %.3f\n\n",
                        occ_rdcfg, occ_rmw, occ_setdma);
                    printf("TTBENCH_CFGLAT_OCC: %.4f %.4f %.4f\n", occ_rdcfg, occ_rmw, occ_setdma);
                    // The movement control: adding the STALLWAIT to SETDMAREG
                    // must cost something, or the stall never engaged and the
                    // difference is two identical things subtracted.
                    printf("TTBENCH_CFGLAT_STALLCOST: %.4f\n", t1_setdma - occ_setdma);
                }
                printf("TTBENCH_CFGLAT_COND: C12 CFGEXU 0x1000\n");
                printf("TTBENCH_CFGLAT_PAIRS: %.4f %.4f %.4f\n", t1_rdcfg, t1_rmw, t1_setdma);
                printf("TTBENCH_CFGLAT_DIFF: %.4f %.4f\n", t1_rdcfg - t1_rmw, t1_rdcfg - t1_setdma);
                // HALF A CYCLE PER PAIR is the floor, and it is not a tuned
                // number: ConfigurationUnit.md predicts >= 1 cycle of difference
                // (RDCFG's ">= 2" against RMWCIB's 1), and half a cycle cannot
                // tell that from none. Below it the reading is the null,
                // whatever its sign.
                if (t1_rdcfg - t1_rmw > 0.5) {
                    printf(
                        "\n  Difference %.3f cycles per pair against the in-unit baseline, clear of\n"
                        "  the 0.5 floor. That is a LOWER BOUND on how long the Configuration Unit\n"
                        "  stays busy with RDCFG after the issue slot -- CORROBORATION for the ISA\n"
                        "  doc's `>= 2`, never provenance, and never an occupancy: probe 14\n"
                        "  measures that separately and the two describe different quantities.\n",
                        t1_rdcfg - t1_rmw);
                } else if (t1_rdcfg - t1_setdma > 0.5) {
                    printf(
                        "\n  RDCFG is INDISTINGUISHABLE FROM RMWCIB0 (%.3f cycles/pair apart) while\n"
                        "  both stand %.3f clear of the off-unit baseline. Read that as the config\n"
                        "  unit's post-issue busy window being the same for a 1-cycle op as for\n"
                        "  RDCFG: the difference measures the unit's handshake, not RDCFG's own\n"
                        "  residency, and the doc's `>= 2` stays unchecked.\n",
                        t1_rdcfg - t1_rmw, t1_rdcfg - t1_setdma);
                } else {
                    printf(
                        "\n  DIFFERENCE %.3f, BELOW THE 0.5 CYCLE/PAIR FLOOR against both baselines.\n"
                        "  ConfigurationUnit.md predicts >= 1, so this is the NULL and not a small\n"
                        "  value. EXPECTED against tt-sim, which reads Blackhole's STALLWAIT\n"
                        "  condition mask as 12 bits (raw 11:0, tt_sim/pe/tensix/backends/sync.py\n"
                        "  `_read_wait_res`) where the ISA doc gives 13, so C12 never survives the\n"
                        "  decode and the wait degrades to the 0x7F `all resources` fallback. On a\n"
                        "  card it means the construction did not reach the quantity.\n",
                        t1_rdcfg - t1_rmw);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // THE FALSIFICATION CONTROL. Slots 20 and 21 are the same two ops behind the
    // same STALL_THREAD block mask, waiting on `p_stall::TRISC_CFG` instead --
    // C10 on Blackhole, C13 on Wormhole, and on both it is "The RISCV T core ...
    // has a memory read-request or write-request against Tensix GPRs or Tensix
    // configuration or TDMA-RISC that has been emitted from the RISCV core but
    // not yet processed". No Tensix instruction is in that condition's scope, so
    // this difference is PREDICTED to be zero, and a Blackhole card measured it
    // at 0.0000 cycles/pair on 2026-08-09.
    //
    // It is here so the C12 reading above cannot be a harness artefact: if
    // BOTH differences move, the condition bit is not what separated them.
    // -----------------------------------------------------------------------
    {
        const auto keys = keys_for(TTBENCH_P_RDCFG_STALL);
        if (!keys.empty()) {
            printf("\n%s\n", std::string(78, '-').c_str());
            printf("phase A: the wrong-condition control (slots 20/21, TRISC_CFG)\n");
            printf("%s\n", std::string(78, '-').c_str());
            printf("  %-7s %-4s %12s %12s %12s\n", "variant", "thr", "rdcfg/pair", "base/pair", "difference");
            double worst = 0;
            int pairs = 0;
            for (const auto& key : keys) {
                double a = 0, b = 0;
                if (!per_pair(key.first, key.second, TTBENCH_P_RDCFG_STALL, &a) ||
                    !per_pair(key.first, key.second, TTBENCH_P_SETDMA_STALL, &b)) {
                    continue;
                }
                printf("  %-7s %-4d %12.3f %12.3f %+12.3f\n", key.first.c_str(), key.second, a, b, a - b);
                pairs++;
                if (a - b > worst) {
                    worst = a - b;
                }
            }
            if (pairs == 0) {
                printf("  neither slot produced a fit; nothing to difference\n");
            } else {
                printf("\nTTBENCH_CFGLAT_WRONGBIT: %.4f\n", worst);
                printf(
                    "  TRISC_CFG is about the RISCV core's outstanding memory requests, not\n"
                    "  about the Configuration Unit's pipeline, so ~0 here is the CORRECT\n"
                    "  reading and is what makes the C12 difference above attributable to the\n"
                    "  condition bit. A control that moved would mean it is not.\n");
            }
        }
    }

    // -----------------------------------------------------------------------
    // THE DEPENDENCE PAIR, read out. Slots 26 and 27, and the reading is
    // `26 - 27`: two arms of the same two opcodes differing only in which GPR
    // the consumer reads.
    //
    // ITS PREDICTION IS ZERO and the prediction comes from the ISA
    // documentation, so a zero is not a failed measurement -- it is the
    // measurement. `BlackholeA0/.../RDCFG.md` puts the producer-to-consumer
    // separation on SOFTWARE ("Software must ensure that the instruction(s)
    // immediately after `RDCFG` are not trying to consume the GPR written by the
    // `RDCFG` instruction"), which is the documented absence of an interlock,
    // and without an interlock a read-after-write costs no cycles at all: the
    // consumer simply reads the old value. That is what the visibility region
    // below measures instead.
    //
    // A NON-ZERO WOULD BE THE BETTER OUTCOME, which is why the arm exists. If
    // Blackhole does interlock, `26 - 27` is RDCFG's latency in cycles directly,
    // with nothing to subtract but itself.
    //
    // THE CONTROL, and it is the one that can fail in both directions: the pair
    // must cost what its two instructions cost apart, probe 14 + probe 10, both
    // measured in this same run. Under it means at least one arm is not issuing
    // what it claims and a null difference means nothing; matching means both
    // instructions ran and a null difference is a fact about the hardware.
    // -----------------------------------------------------------------------
    {
        const auto keys = keys_for(TTBENCH_P_RDCFG_DEP);
        if (!keys.empty()) {
            printf("\n%s\n", std::string(78, '-').c_str());
            printf("phase A: RDCFG -> GPR -> consumer, dependent against independent (26/27)\n");
            printf("%s\n", std::string(78, '-').c_str());
            printf(
                "  Both arms are TTI_RDCFG(60,0) followed by TTI_ADDDMAREG(1,63,0,X).\n"
                "  X = 60 in slot 26 (the RDCFG destination) and X = 59 in slot 27.\n"
                "  BlackholeA0 RDCFG.md makes the separation software's job, so the\n"
                "  documented prediction is that this difference is ZERO and the latency\n"
                "  is not a duration at all. See the visibility section below.\n\n");
            printf("  %-7s %-4s %12s %12s %12s\n", "variant", "thr", "dep/pair", "indep/pair", "difference");
            bool have_t1 = false;
            double d_t1 = 0, dep_t1 = 0;
            for (const auto& key : keys) {
                double a = 0, b = 0;
                if (!per_pair(key.first, key.second, TTBENCH_P_RDCFG_DEP, &a) ||
                    !per_pair(key.first, key.second, TTBENCH_P_RDCFG_INDEP, &b)) {
                    continue;
                }
                printf(
                    "  %-7s %-4d %12.3f %12.3f %+12.3f%s\n",
                    key.first.c_str(),
                    key.second,
                    a,
                    b,
                    a - b,
                    key.first == "t1" ? "" : "   (not graded)");
                if (key.first == "t1" && !have_t1) {
                    have_t1 = true;
                    d_t1 = a - b;
                    dep_t1 = a;
                }
            }
            if (!have_t1) {
                printf("\n  no t1 fit; the dependence difference is not reported\n");
            } else {
                double occ_rdcfg = 0, occ_add = 0;
                const bool have_parts = per_pair("t1", 1, 14, &occ_rdcfg) && per_pair("t1", 1, 10, &occ_add);
                printf("\nTTBENCH_DEP_DIFF: %.4f\n", d_t1);
                if (have_parts) {
                    printf("TTBENCH_DEP_PARTS: %.4f %.4f %.4f\n", dep_t1, occ_rdcfg, occ_add);
                    printf(
                        "  the dependent pair costs %.3f cycles against %.3f + %.3f = %.3f for\n"
                        "  the two instructions measured bare (probes 14 and 10). A pair that\n"
                        "  came in UNDER the sum would mean an arm is not issuing both\n"
                        "  instructions, and then a zero difference would say nothing.\n",
                        dep_t1,
                        occ_rdcfg,
                        occ_add,
                        occ_rdcfg + occ_add);
                } else {
                    printf(
                        "  probes 14 and 10 were not both in this run, so the pair cost cannot\n"
                        "  be checked against its parts. Add 0x4200 to --probes.\n");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // THE VISIBILITY REGION, read out. This is where RDCFG's ">= 2" is reached.
    //
    // It carries no cycle count, so nothing here is a slope and nothing here has
    // a launch or timer overhead for an intercept to absorb: the sweep is over
    // producer-to-consumer SEPARATION and the reading is the smallest separation
    // at which every repetition observes the value RDCFG wrote. See TTBENCH_VIS_*
    // in kernels/compute/bench_layout.h for the construction and
    // "THE VISIBILITY CONSTRUCTION" in kernels/compute/raw_probes.cpp for why
    // the sequence has to be literal `.ttinsn` immediates.
    // -----------------------------------------------------------------------
    if (!vis.empty()) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase A: RDCFG's latency as a producer-to-consumer DISTANCE\n");
        printf("%s\n", std::string(78, '-').c_str());
        const uint32_t reps = vis[TTBENCH_VIS_W_REPS];
        const uint32_t fresh_ref = vis[TTBENCH_VIS_W_FAR_S1];
        const uint32_t far_s2 = vis[TTBENCH_VIS_W_FAR_S2];
        const uint32_t nord_s1 = vis[TTBENCH_VIS_W_NORD_S1];
        const uint32_t nord_s2 = vis[TTBENCH_VIS_W_NORD_S2];
        printf(
            "  seeds 0x%08X / 0x%08X, never-ran marker 0x%08X, config index %d,\n"
            "  %u repetitions per separation.\n\n",
            TTBENCH_VIS_S1,
            TTBENCH_VIS_S2,
            TTBENCH_VIS_MARK,
            TTBENCH_VIS_CFGIDX,
            reps);

        // The controls, and BOTH DIRECTIONS have to be exercised for the sweep
        // to mean anything. The no-RDCFG arms can only pass by returning two
        // DIFFERENT values (their own seeds), so a readout stuck on a constant
        // fails them; the far arms can only pass by returning ONE value twice,
        // and one that is neither seed, so a readout that merely echoes the seed
        // fails them. Neither check has a way to pass vacuously.
        const bool stale_representable = (nord_s1 == TTBENCH_VIS_S1) && (nord_s2 == TTBENCH_VIS_S2);
        const bool fresh_representable = (fresh_ref == far_s2) && (fresh_ref != TTBENCH_VIS_S1) &&
                                         (fresh_ref != TTBENCH_VIS_S2) && (fresh_ref != TTBENCH_VIS_MARK);
        printf(
            "  control: no RDCFG at all  -> 0x%08X, 0x%08X (want 0x%08X, 0x%08X)  %s\n",
            nord_s1,
            nord_s2,
            TTBENCH_VIS_S1,
            TTBENCH_VIS_S2,
            stale_representable ? "ok" : "FAILED");
        printf(
            "  control: RDCFG at d=%-2d    -> 0x%08X, 0x%08X (want them equal, and\n"
            "                               neither seed nor marker)                %s\n",
            TTBENCH_VIS_FAR,
            fresh_ref,
            far_s2,
            fresh_representable ? "ok" : "FAILED");
        printf(
            "  consumer never ran: %u repetitions;  unexplained: %u\n\n",
            vis[TTBENCH_VIS_W_MARK],
            vis[TTBENCH_VIS_W_OTHER]);

        printf("  %-4s %10s %10s %10s\n", "d", "fresh", "stale", "other");
        int d_min = 0;
        for (int d = 1; d <= TTBENCH_VIS_MAXD; d++) {
            const uint32_t f = vis[TTBENCH_VIS_W_FRESH + d - 1];
            const uint32_t s = vis[TTBENCH_VIS_W_STALE + d - 1];
            printf("  %-4d %10u %10u %10u\n", d, f, s, reps - f - s);
            if (d_min == 0 && reps > 0 && f == reps) {
                d_min = d;
            }
        }
        const bool controls_ok = stale_representable && fresh_representable && reps > 0 &&
                                 vis[TTBENCH_VIS_W_MARK] == 0 && vis[TTBENCH_VIS_W_OTHER] == 0;
        printf("\nTTBENCH_VIS_CONTROLS: %s %s %u %u\n",
               stale_representable ? "stale-ok" : "stale-FAILED",
               fresh_representable ? "fresh-ok" : "fresh-FAILED",
               vis[TTBENCH_VIS_W_MARK],
               vis[TTBENCH_VIS_W_OTHER]);
        printf("TTBENCH_VIS_DMIN: %d %u\n", controls_ok ? d_min : 0, reps);
        printf(
            "TTBENCH_VIS_COUNTS: %u %u %u %u\n",
            vis[TTBENCH_VIS_W_FRESH + 0],
            vis[TTBENCH_VIS_W_FRESH + 1],
            vis[TTBENCH_VIS_W_FRESH + 2],
            vis[TTBENCH_VIS_W_FRESH + 3]);
        printf(
            "  d_min is the smallest separation at which EVERY repetition saw the value\n"
            "  RDCFG wrote. Because the consumer may read its operand some cycles into\n"
            "  its own execution, d_min is a LOWER BOUND on the latency -- which is the\n"
            "  direction the charging policy takes bounds in. d_min = 2 corroborates\n"
            "  ConfigurationUnit.md's '>= 2 cycles' exactly; d_min = 1 would leave the\n"
            "  '>= 2' unreached rather than refute it, since a consumer that reads late\n"
            "  is a complete alternative explanation. A MIXTURE at any d (fresh and\n"
            "  stale both non-zero) means the RISC-V front end did not deliver the\n"
            "  sequence at one instruction per cycle and the separation is not what it\n"
            "  says; that is why every repetition is counted rather than one taken.\n");
    }

    // -----------------------------------------------------------------------
    // THE C12 LIVENESS CONTROL, read out. Slots 28 and 29, and the reading is
    //     d(v) = pair(28, v) - pair(29, v)   on thread 1
    //     the verdict is on d(t3) - d(t1)  (or d(t2) - d(t1))
    //
    // This is the only thing in the benchmark that can separate "RDCFG's latency
    // is real and no busy-condition could ever have seen it" from "C12 does not
    // behave as documented on this part". Both explain slots 22-25 reading
    // 0.0000 on a card and the visibility region cannot choose between them,
    // because it never consults a condition bit.
    //
    // It needs at least two variants in the same run to say anything -- t1 for
    // the floor and t2 or t3 for the hammered case -- so `--variants t1,t3`.
    // With only t1 it prints the floor and declines to grade.
    // -----------------------------------------------------------------------
    {
        const auto keys = keys_for(TTBENCH_P_C12_XT);
        if (!keys.empty()) {
            printf("\n%s\n", std::string(78, '-').c_str());
            printf("phase A: is C12 live? the cross-thread control (slots 28/29)\n");
            printf("%s\n", std::string(78, '-').c_str());
            printf(
                "  thread 1 runs STALLWAIT(STALL_THREAD, CFGEXU)+NOP in slot 28 and NOP+NOP\n"
                "  in slot 29; threads 0 and 2 run RMWCIB0+NOP in BOTH, so at t2/t3 they\n"
                "  hold the Configuration Unit from outside thread 1's issue path. C12 is\n"
                "  \"Any thread has an instruction in any stage of the Configuration Unit\n"
                "  pipeline\", so a live C12 must make thread 1's stall grow with the\n"
                "  number of hammering threads. Slot 29 carries the cross-thread issue\n"
                "  interference that grows anyway, and subtracts it.\n\n");
            printf("  %-7s %-4s %12s %12s %12s\n", "variant", "thr", "stall/pair", "null/pair", "d = difference");
            bool have_t1 = false;
            double d_t1 = 0, d_multi = 0;
            std::string multi_variant;
            for (const auto& key : keys) {
                if (key.second != 1) {
                    continue;  // only the stalling thread is the measurement
                }
                double a = 0, b = 0;
                if (!per_pair(key.first, key.second, TTBENCH_P_C12_XT, &a) ||
                    !per_pair(key.first, key.second, TTBENCH_P_C12_XT_NULL, &b)) {
                    continue;
                }
                printf("  %-7s %-4d %12.3f %12.3f %+12.3f\n", key.first.c_str(), key.second, a, b, a - b);
                if (key.first == "t1") {
                    have_t1 = true;
                    d_t1 = a - b;
                } else if (a - b > d_multi || multi_variant.empty()) {
                    d_multi = a - b;
                    multi_variant = key.first;
                }
            }
            if (!have_t1 || multi_variant.empty()) {
                printf(
                    "\n  needs t1 AND one of t2/t3 in the same run to grade; got %s.\n"
                    "  Run with --variants t1,t3.\n",
                    !have_t1 ? "no t1" : "t1 only");
            } else {
                printf("\nTTBENCH_C12_LIVE: %.4f %.4f %s\n", d_t1, d_multi, multi_variant.c_str());
                printf(
                    "  d(t1) = %.3f is the stall's own floor with nobody else in the unit.\n"
                    "  d(%s) = %.3f is the same stall while %s hammering thread(s) keep the\n"
                    "  unit busy. A large positive difference says C12 IS live, and then\n"
                    "  slots 22-25 reading zero means RDCFG's post-issue residency is no\n"
                    "  wider than the stall's own documented one-cycle lag. A zero says C12\n"
                    "  is inert on this part, and then slots 22-25 say nothing about RDCFG\n"
                    "  at all. Both are reachable; neither is the default.\n",
                    d_t1,
                    multi_variant.c_str(),
                    d_multi,
                    multi_variant == "t3" ? "two" : "one");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase B, read out. The absolute slope is a confounded composite; the
    // deltas are the result, and the unpack thread's slope next to them is what
    // says whether the loop was math-bound at all.
    //
    // A null delta is NOT scored as a failure. The fidelity setting is known to
    // reach the math thread -- the three JIT-built TRISC1 ELFs differ in exactly
    // the MOP inner-loop count (1 / 2 / 4) and the ADDR_MOD_5 fidelity
    // increment, while TRISC0 and TRISC2 are byte-identical across fidelities --
    // so a zero delta means the MVMULs hid behind something slower, which is a
    // finding about the hardware and not a broken measurement.
    // -----------------------------------------------------------------------
    auto mm_slope = [&](const char* variant, int thread, double* out_slope) {
        for (const auto& s : mm_slopes) {
            if (s.variant == variant && s.thread == thread) {
                *out_slope = s.slope;
                return true;
            }
        }
        return false;
    };
    const struct {
        const char* from;
        const char* to;
        double predicted;
    } steps[] = {{"LoFi", "HiFi2", 16.0}, {"HiFi2", "HiFi4", 32.0}};
    if (!mm_slopes.empty()) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase B: fidelity deltas (math thread), and what bounded the loop\n");
        printf("%s\n", std::string(78, '-').c_str());
        double biggest_delta = 0;
        int deltas = 0;
        for (const auto& s : steps) {
            double a = 0, b = 0;
            if (!mm_slope(s.from, 1, &a) || !mm_slope(s.to, 1, &b)) {
                continue;
            }
            deltas++;
            const double d = b - a;
            if (d > biggest_delta) {
                biggest_delta = d;
            }
            printf(
                "  %-6s -> %-6s  measured %+8.2f   predicted %+7.2f   residual %+8.2f\n",
                s.from,
                s.to,
                d,
                s.predicted,
                d - s.predicted);
        }
        if (deltas == 0) {
            printf(
                "  fewer than two adjacent fidelities in this run -- nothing to\n"
                "  difference, and the absolute numbers below are a confounded composite\n"
                "  that is not a cost of anything.\n");
        }
        printf("  (predicted = 16 MVMULs per fidelity phase x 1 cycle each)\n\n");
        printf("  %-8s %14s %14s\n", "fidelity", "math (thr 1)", "unpack (thr 0)");
        for (const char* v : {"LoFi", "HiFi2", "HiFi4"}) {
            double m = 0, u = 0;
            if (mm_slope(v, 1, &m)) {
                const bool have_u = mm_slope(v, 0, &u);
                printf("  %-8s %14.2f", v, m);
                if (have_u) {
                    printf(" %14.2f", u);
                }
                printf("\n");
            }
        }
        if (deltas > 0 && biggest_delta < 4.0) {
            printf(
                "\n  The fidelity slopes do not separate. The MVMULs are emitted by the\n"
                "  Tensix MOP expander, so they only cost the math thread wall-clock time\n"
                "  when the coprocessor back-pressures the issuing core. Compare the two\n"
                "  columns above: if the unpack thread's slope is >= the math thread's and\n"
                "  is flat across fidelities, this loop is UNPACK-BOUND and the fidelity\n"
                "  arithmetic is untested rather than refuted. That is a result; it is not\n"
                "  scored as a failure and it does not affect phase A.\n");
        }
    }

    printf("\nwrote %zu rows to %s\n", rows.size(), out_path.c_str());

    // Per-phase verdicts. Report only the phases that actually ran.
    bool ran_a = false, ran_b = false;
    for (const auto& r : rows) {
        (r.phase == "A" ? ran_a : ran_b) = true;
    }
    if (ran_a) {
        if (fail_a == 0) {
            printf("TTBENCH_VALID_A: yes\n");
        } else {
            printf("TTBENCH_VALID_A: no (%d checks failed in phase A)\n", fail_a);
        }
    }
    if (ran_b) {
        if (fail_b == 0) {
            printf("TTBENCH_VALID_B: yes\n");
        } else {
            printf("TTBENCH_VALID_B: no (%d checks failed in phase B)\n", fail_b);
        }
    }
    const bool a_ok = !ran_a || fail_a == 0;
    const bool b_ok = !ran_b || fail_b == 0;
    if (a_ok && b_ok) {
        printf("TTBENCH_VALID: yes\n");
        printf("Completed successfully on the device\n");
    } else {
        printf(
            "TTBENCH_VALID: no -- but the verdict above is PER PHASE. Send the CSV\n"
            "  regardless: the rows of a phase that passed are unaffected by one that\n"
            "  did not, and %s.\n",
            a_ok    ? "phase A is usable here"
            : b_ok  ? "phase B is usable here"
                    : "neither phase is usable here");
    }
    // Exit status is a bit mask so a wrapper can tell the phases apart:
    // 1 = phase A failed, 2 = phase B failed, 3 = both, 0 = clean.
    return (a_ok ? 0 : 1) | (b_ok ? 0 : 2);
}
