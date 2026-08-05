// riscvbench -- a measuring instrument for the baby RISC-V front end.
//
// WHAT IT IS FOR. tt-sim's cost model now wires six of the nine Tensix backend
// units, the NoC hop and bandwidth terms, both arches' DRAM latency and the
// baby RISC-V load/store path. Six of those units moved ZERO simulated cycles,
// and `docs/plans/cost-model.md` records the same diagnosis each time: *"the
// constraint is the un-modelled RISC-V front end"*. Silicon agreed
// independently -- `perfbench/tensixbench` phase A reads exactly 1.000 cycles
// per instruction against tt-sim for every probe of every unit at every data
// format, because `TensixFrontend.push_mop_instruction` is an unbounded list
// append and the `.ttinsn` write returns immediately.
//
// So the front end is the term that currently makes every other term
// unobservable, and it is the one part of the machine the cost tables describe
// (`riscv:` in tt_sim/perf/unit_costs.yaml, almost entirely `isa_doc`) whose
// ISSUE side nothing has ever measured. This program measures it: fetch, issue
// rate, branch cost, and above all the cost of pushing a Tensix instruction out
// of a baby core.
//
// The methodology, the confounds, and what each measurement can and cannot
// establish are in docs/plans/riscv-front-end-benchmark.md. The short version:
//
//   * Every reported cost is a SLOPE over several block counts, never a single
//     absolute measurement, so kernel launch, the clock reads, the barrier and
//     the surrounding call all cancel exactly. An empty-body control loop then
//     cancels the RISC-V loop overhead too.
//   * Five phases, run and scored INDEPENDENTLY: R (straight-line RV32IM),
//     T (the `.ttinsn` issue path), C (control flow), Q (the Tensix instruction
//     queue's depth), F (instruction footprint). Each sets its own bit in the
//     exit status and prints its own TTRVBENCH_VALID_<X> line.
//   * Phase Q is deliberately NOT gated on linearity. It is looking for a knee;
//     a straight line there would be the null result, not the healthy one.
//
// HOW A NULL IS TOLD FROM AN UN-INSTRUMENTED RUN. This is the trap
// `tensixbench` fell into -- its phase A reads 1.000 everywhere against tt-sim
// *because of* the very gap it was trying to measure, so "everything is 1.000"
// is simultaneously the expected simulator output and the signature of a
// benchmark that measured nothing. riscvbench answers it with probes tt-sim
// ALREADY moves: `rv_mul_*` (2 cycles on Wormhole), `rv_div` (>= 6),
// `rv_load_chase` (>= 8 on Wormhole, 2 on Blackhole) and `rv_store_spread`
// (5 everywhere) are consumed by tt_sim/pe/rv/cost.py today. If those four read
// their table values and everything else reads 1.000, the instrument is live
// and the 1.000s are a finding. If those four ALSO read 1.000, the run is not
// measuring anything and the program says so.
//
// TIMESTAMPS come from RISCV_DEBUG_REG_WALL_CLOCK_L, exactly the register
// tt-metal's device profiler reads for DeviceZoneScopedN. Reading it directly
// keeps the measurement identical on silicon and against tt-sim, needs no Tracy
// build, and sidesteps the profiler's dependence on a device AICLK.
//
// OUTPUT is a CSV of raw (probe, threads, n, cycles) points, plus a
// human-readable summary. Nothing here fits or reports a cost-table number; the
// comparison against the tables is tt_sim/perf/riscv_bench_sweep.py.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "kernels/compute/rvbench_layout.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// ---------------------------------------------------------------------------
// The probe table. Slot order is the contract with
// kernels/compute/rv_probes.cpp.
//
// `unit` names the resource the probe stresses, not the instruction's opcode
// class, because that is what the analysis groups residuals by:
//
//   RV        the integer unit / issue path
//   RV_LSU    the load-store unit
//   RV_BR     the branch path
//   RV_FETCH  instruction fetch
//   NONE / SFPU / THCON  a Tensix backend unit, spelled exactly as the
//                        `ex_resource` key in tt_sim/pe/tensix/
//                        tensix_instructions.yaml, so a `.ttinsn` probe can be
//                        looked up in the Tensix cost table too
//   TTINSN    the `.ttinsn` issue path itself, measured per GROUP
//   TTQUEUE   the Tensix instruction queue, measured per burst length
//
// `unroll` is the divisor that turns a per-block slope into a per-instruction
// (or, for TTINSN, per-group) cost. It is per probe rather than global because
// phase F's footprints and phase T's groups do not share one.
// ---------------------------------------------------------------------------
struct Probe {
    const char* name;
    const char* unit;
    char phase;
    uint32_t unroll;
};

const Probe PROBES[RVBENCH_NUM_PROBES] = {
    // Phase R -- but the control is shared by every slope phase, so it carries
    // phase '*' and is emitted alongside whichever phase is running.
    {"loop_overhead", "-", '*', RVBENCH_UNROLL},
    {"rv_addi_indep", "RV", 'r', RVBENCH_UNROLL},
    {"rv_addi_dep", "RV", 'r', RVBENCH_UNROLL},
    {"rv_mul_indep", "RV", 'r', RVBENCH_UNROLL},
    {"rv_mul_dep", "RV", 'r', RVBENCH_UNROLL},
    {"rv_div", "RV", 'r', RVBENCH_UNROLL},
    {"rv_load_chase", "RV_LSU", 'r', RVBENCH_UNROLL},
    {"rv_load_indep", "RV_LSU", 'r', RVBENCH_UNROLL},
    {"rv_store_spread", "RV_LSU", 'r', RVBENCH_UNROLL},
    {"rv_store_coalesce", "RV_LSU", 'r', RVBENCH_UNROLL},
    {"rv_load_stack", "RV_LSU", 'r', RVBENCH_UNROLL},
    {"rv_store_stack", "RV_LSU", 'r', RVBENCH_UNROLL},
    // Phase T.
    {"tt_nop", "NONE", 't', RVBENCH_UNROLL},
    {"tt_sfpnop", "SFPU", 't', RVBENCH_UNROLL},
    {"tt_setdmareg", "THCON", 't', RVBENCH_UNROLL},
    {"tt_adddmareg", "THCON", 't', RVBENCH_UNROLL},
    {"tt_pad", "TTINSN", 't', RVBENCH_GROUP_REPS},
    {"tt_fuse2", "TTINSN", 't', RVBENCH_GROUP_REPS},
    {"tt_fuse4", "TTINSN", 't', RVBENCH_GROUP_REPS},
    {"tt_spread4", "TTINSN", 't', RVBENCH_GROUP_REPS},
    // Phase C.
    {"c_ctrl_xor", "RV_BR", 'c', RVBENCH_UNROLL},
    {"c_nt", "RV_BR", 'c', RVBENCH_UNROLL},
    {"c_t", "RV_BR", 'c', RVBENCH_UNROLL},
    {"c_xor_nt", "RV_BR", 'c', RVBENCH_UNROLL},
    {"c_xor_t", "RV_BR", 'c', RVBENCH_UNROLL},
    {"c_xor_alt", "RV_BR", 'c', RVBENCH_UNROLL},
    {"c_jal", "RV_BR", 'c', RVBENCH_UNROLL},
    // Phase Q -- `n` is a burst length, not a block count, and `unroll` is 1.
    {"q_ctrl", "TTQUEUE", 'q', 1},
    {"q_nop", "TTQUEUE", 'q', 1},
    {"q_setdmareg", "TTQUEUE", 'q', 1},
    {"q_adddmareg", "TTQUEUE", 'q', 1},
    {"q_adddmareg_sync", "TTQUEUE", 'q', 1},
    // Phase F -- the unroll IS the footprint.
    {"f_64", "RV_FETCH", 'f', 64},
    {"f_128", "RV_FETCH", 'f', 128},
    {"f_256", "RV_FETCH", 'f', 256},
    {"f_512", "RV_FETCH", 'f', 512},
    {"f_1024", "RV_FETCH", 'f', 1024},
    {"f_2048", "RV_FETCH", 'f', 2048},
};

// 64 bits wide, because there are more than 32 probes and phase F's six live
// above slot 31. It travels to the kernel as two runtime-arg words.
constexpr uint64_t ALL_PROBES = (1ull << RVBENCH_NUM_PROBES) - 1ull;

const char PHASE_LETTERS[] = "rtcqf";

struct PhaseInfo {
    char letter;
    const char* define;
    uint32_t bit;
    int fail_bit;
};

const PhaseInfo PHASES[] = {
    {'r', "RVBENCH_PHASE_R", RVBENCH_PHASE_R_BIT, 1},
    {'t', "RVBENCH_PHASE_T", RVBENCH_PHASE_T_BIT, 2},
    {'c', "RVBENCH_PHASE_C", RVBENCH_PHASE_C_BIT, 4},
    {'q', "RVBENCH_PHASE_Q", RVBENCH_PHASE_Q_BIT, 8},
    {'f', "RVBENCH_PHASE_F", RVBENCH_PHASE_F_BIT, 16},
};
constexpr int NUM_PHASES = 5;

struct Row {
    char phase;
    std::string variant;
    int probe_id;
    std::string probe;
    std::string unit;
    int active_threads;
    int thread;
    uint32_t n;
    uint32_t unroll;
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

bool selected(const std::string& list, const std::string& name) {
    return ("," + list + ",").find("," + name + ",") != std::string::npos;
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t base_blocks = 4;
    uint64_t probe_mask = ALL_PROBES;
    std::string phases = PHASE_LETTERS;
    std::string variant_filter = "t1,t2,t3";
    std::string out_path;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--blocks") {
            base_blocks = std::stoul(next());
        } else if (a == "--probes") {
            probe_mask = std::stoull(next(), nullptr, 0) & ALL_PROBES;
        } else if (a == "--phase") {
            phases = next();
        } else if (a == "--variants") {
            variant_filter = next();
        } else if (a == "--out") {
            out_path = next();
        } else if (a == "-h" || a == "--help") {
            printf(
                "usage: riscvbench [--blocks N] [--probes 0xMASK] [--phase LETTERS]\n"
                "                  [--variants LIST] [--out FILE]\n"
                "\n"
                "  --blocks N        the slope phases sweep blocks = N, 2N, 3N, 4N\n"
                "                    (default 4). Phase Q ignores it: its sweep is over\n"
                "                    BURST LENGTH, 1..128, which is the axis a queue\n"
                "                    depth lives on.\n"
                "  --probes 0xMASK   bit i enables probe i (default all %d). Probe 0 is\n"
                "                    the empty-loop control and every slope phase needs\n"
                "                    it; clearing bit 0 makes those phases unreadable.\n"
                "  --phase LETTERS   any subset of `%s` (default all five):\n"
                "                      r  straight-line RV32IM -- the baseline, and the\n"
                "                         four probes tt-sim already moves\n"
                "                      t  the `.ttinsn` issue path, including the\n"
                "                         instruction-cache fusion experiment\n"
                "                      c  branch direction and mispredict cost\n"
                "                      q  the Tensix instruction queue's depth\n"
                "                      f  instruction footprint, 256 B to 8 KiB\n"
                "                    Each phase is a SEPARATE kernel build: its probe\n"
                "                    bodies are only compiled when it is selected, so a\n"
                "                    phase F run does not carry phase R's text and vice\n"
                "                    versa. That matters -- phase F's bodies are 16 KiB\n"
                "                    of instructions and would otherwise BE the thing\n"
                "                    phase F measures.\n"
                "  --variants LIST   comma-separated subset of t1,t2,t3 (default all\n"
                "                    three), i.e. how many TRISCs run the identical\n"
                "                    probes at once. Each is a separate program launch.\n"
                "  --out FILE        CSV path (default riscvbench-<arch>.csv)\n",
                RVBENCH_NUM_PROBES,
                PHASE_LETTERS);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s (try --help)\n", a.c_str());
            return 2;
        }
    }
    if (base_blocks == 0) {
        fprintf(stderr, "--blocks must be >= 1\n");
        return 2;
    }
    if (!selected(variant_filter, "t1") && !selected(variant_filter, "t2") &&
        !selected(variant_filter, "t3")) {
        fprintf(stderr, "--variants %s selects none of t1,t2,t3\n", variant_filter.c_str());
        return 2;
    }
    {
        bool any = false;
        for (const char* p = PHASE_LETTERS; *p; p++) {
            if (phases.find(*p) != std::string::npos) {
                any = true;
            }
        }
        if (!any) {
            fprintf(stderr, "--phase %s selects none of %s\n", phases.c_str(), PHASE_LETTERS);
            return 2;
        }
    }

    IDevice* device = CreateDevice(0);
    const std::string arch = arch_name(device);
    if (out_path.empty()) {
        out_path = "riscvbench-" + arch + ".csv";
    }
    constexpr CoreCoord core = {0, 0};

    // Reserve L1 through the allocator rather than picking an address, so
    // nothing here can collide with the firmware or the circular buffers.
    constexpr uint32_t result_bytes = RVBENCH_RESULT_WORDS * 4;
    InterleavedBufferConfig l1_cfg{
        .device = device, .size = result_bytes, .page_size = result_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> result_scratch = CreateBuffer(l1_cfg);
    const uint32_t results_addr = result_scratch->address();

    InterleavedBufferConfig bar_cfg{
        .device = device, .size = 64, .page_size = 64, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> barrier_scratch = CreateBuffer(bar_cfg);
    const uint32_t barrier_addr = barrier_scratch->address();

    // One scratch region per thread: a t3 run's three copies of the store
    // probes writing over each other would be a contention measurement, and a
    // confounded one, rather than the per-core cost this phase reports.
    constexpr uint32_t scratch_bytes = RVBENCH_SCRATCH_BYTES * RVBENCH_MAX_THREADS;
    InterleavedBufferConfig scratch_cfg{
        .device = device, .size = scratch_bytes, .page_size = scratch_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> data_scratch = CreateBuffer(scratch_cfg);
    const uint32_t scratch_addr = data_scratch->address();

    std::vector<Row> rows;
    int fail[NUM_PHASES] = {0, 0, 0, 0, 0};
    bool ran[NUM_PHASES] = {false, false, false, false, false};
    uint32_t stack_addr = 0;

    auto write_csv = [&]() {
        FILE* csv = fopen(out_path.c_str(), "w");
        if (!csv) {
            fprintf(stderr, "cannot open %s for writing\n", out_path.c_str());
            return false;
        }
        fprintf(csv, "# riscvbench raw points -- see docs/plans/riscv-front-end-benchmark.md\n");
        // Every token is `key=value` because the analysis harness
        // (tt_sim/perf/riscv_bench_sweep.read_csv) harvests them into `meta`.
        // `stack_addr` is in there because the `rv_load_stack` probe's expected
        // cost depends on which memory region the stack landed in, and that is
        // a tt-metal placement decision this program does not get to make.
        fprintf(
            csv,
            "# arch=%s magic=0x%08X probe_mask=0x%llX phases=%s variants=%s "
            "base_blocks=%u group_reps=%u pad=%u stack_addr=0x%08X scratch_addr=0x%08X "
            "div_dividend=0x12345678 div_divisor=3\n",
            arch.c_str(),
            RVBENCH_MAGIC,
            (unsigned long long)probe_mask,
            phases.c_str(),
            variant_filter.c_str(),
            base_blocks,
            RVBENCH_GROUP_REPS,
            RVBENCH_PAD,
            stack_addr,
            scratch_addr);
        fprintf(csv, "phase,variant,probe_id,probe,unit,active_threads,thread,n,unroll,cycles\n");
        for (const auto& r : rows) {
            fprintf(
                csv,
                "%c,%s,%d,%s,%s,%d,%d,%u,%u,%u\n",
                r.phase,
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

    const struct {
        const char* variant;
        uint32_t mask;
    } thread_sets[] = {
        {"t1", 0x2},  // math thread alone
        {"t2", 0x3},  // unpack + math
        {"t3", 0x7},  // all three
    };

    for (int pi = 0; pi < NUM_PHASES; pi++) {
        const PhaseInfo& ph = PHASES[pi];
        if (phases.find(ph.letter) == std::string::npos) {
            continue;
        }
        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            Program program = CreateProgram();

            // A compute kernel needs at least one circular buffer for the
            // generated CB descriptor list to be well formed. Nothing reads it.
            CircularBufferConfig dummy_cb =
                CircularBufferConfig(1024, {{CBIndex::c_0, tt::DataFormat::UInt32}})
                    .set_page_size(CBIndex::c_0, 1024);
            CreateCircularBuffer(program, core, dummy_cb);

            KernelHandle bench = CreateKernel(
                program,
                "kernels/compute/rv_probes.cpp",
                core,
                ComputeConfig{.defines = {{ph.define, "1"}}});
            SetRuntimeArgs(
                program,
                bench,
                core,
                {results_addr,
                 barrier_addr,
                 scratch_addr,
                 base_blocks,
                 (uint32_t)(probe_mask & 0xFFFFFFFFull),
                 ts.mask,
                 ph.bit,
                 (uint32_t)(probe_mask >> 32)});

            std::vector<uint32_t> zeros(16, 0);
            detail::WriteToDeviceL1(device, core, barrier_addr, zeros);
            std::vector<uint32_t> clear(RVBENCH_RESULT_WORDS, 0);
            detail::WriteToDeviceL1(device, core, results_addr, clear);

            detail::LaunchProgram(device, program, true, true);

            std::vector<uint32_t> words(RVBENCH_RESULT_WORDS, 0);
            detail::ReadFromDeviceL1(device, core, results_addr, result_bytes, words);
            if (words[RVBENCH_HDR_MAGIC] != RVBENCH_MAGIC) {
                fprintf(
                    stderr,
                    "FATAL: result header magic 0x%08X != 0x%08X -- the kernel did not run, "
                    "or the host and kernel disagree about the layout\n",
                    words[RVBENCH_HDR_MAGIC],
                    RVBENCH_MAGIC);
                CloseDevice(device);
                return 1;
            }
            stack_addr = words[RVBENCH_HDR_STACK_ADDR];
            ran[pi] = true;

            const int active = popcount(ts.mask);
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                const uint32_t* slot =
                    words.data() + RVBENCH_HDR_WORDS + t * RVBENCH_NUM_PROBES * RVBENCH_MAX_POINTS;
                for (int p = 0; p < RVBENCH_NUM_PROBES; p++) {
                    if ((probe_mask & (1ull << p)) == 0) {
                        continue;
                    }
                    const Probe& probe = PROBES[p];
                    if (probe.phase != ph.letter && probe.phase != '*') {
                        continue;
                    }
                    const bool is_q = probe.phase == 'q';
                    const int points = is_q ? RVBENCH_MAX_POINTS : RVBENCH_SLOPE_POINTS;
                    for (int kx = 0; kx < points; kx++) {
                        const uint32_t n = is_q ? (1u << kx) : base_blocks * (uint32_t)(kx + 1);
                        rows.push_back(Row{
                            ph.letter,
                            ts.variant,
                            p,
                            probe.name,
                            probe.unit,
                            active,
                            t,
                            n,
                            probe.unroll,
                            slot[p * RVBENCH_MAX_POINTS + kx]});
                    }
                }
            }
            write_csv();
            printf(
                "phase %c [%s]: done (%d issuing thread%s), %zu rows written to %s\n",
                ph.letter,
                ts.variant,
                active,
                active == 1 ? "" : "s",
                rows.size(),
                out_path.c_str());
            fflush(stdout);
        }
    }

    CloseDevice(device);

    if (!write_csv()) {
        return 1;
    }

    // -----------------------------------------------------------------------
    // The summary. Slopes only; no absolute measurement is reported as a cost.
    // -----------------------------------------------------------------------
    printf("\n%s\n", std::string(78, '=').c_str());
    printf("riscvbench summary [%s] -- slopes only, no absolute measurement is a cost\n", arch.c_str());
    printf("%s\n", std::string(78, '=').c_str());
    printf("stack probe address: 0x%08X (which load-latency row that is, the analysis\n", stack_addr);
    printf("  classifies -- tt-metal places the stack, not this program)\n\n");

    auto fit_for = [&](char phase, const std::string& variant, const std::string& probe, int thread) {
        std::vector<double> xs, ys;
        for (const auto& r : rows) {
            if (r.phase == phase && r.variant == variant && r.probe == probe && r.thread == thread) {
                xs.push_back(r.n);
                ys.push_back(r.cycles);
            }
        }
        return xs.size() >= 2 ? least_squares(xs, ys) : Fit{0, 0, 0};
    };

    auto cycles_at = [&](char phase, const std::string& variant, const std::string& probe, int thread,
                         uint32_t n, uint32_t* out_cycles) {
        for (const auto& r : rows) {
            if (r.phase == phase && r.variant == variant && r.probe == probe && r.thread == thread &&
                r.n == n) {
                *out_cycles = r.cycles;
                return true;
            }
        }
        return false;
    };

    // Control slope per (phase, variant, thread): every slope probe is reported
    // relative to the empty-body loop measured in its own launch.
    struct ControlKey {
        char phase;
        std::string variant;
        int thread;
        double slope;
    };
    std::vector<ControlKey> controls;
    auto control_slope = [&](char phase, const std::string& variant, int thread) {
        for (const auto& c : controls) {
            if (c.phase == phase && c.variant == variant && c.thread == thread) {
                return c.slope;
            }
        }
        const double s = fit_for(phase, variant, "loop_overhead", thread).slope;
        controls.push_back(ControlKey{phase, variant, thread, s});
        return s;
    };

    printf("%-18s %-3s %-7s %-8s %-4s %12s %11s %8s\n",
           "probe", "ph", "variant", "unit", "thr", "cyc/n", "per-instr", "R^2");

    std::vector<std::string> emitted;
    for (const auto& r : rows) {
        if (r.phase == 'q') {
            continue;  // reported below, as a knee rather than a slope
        }
        const std::string tag =
            std::string(1, r.phase) + "/" + r.variant + "/" + r.probe + "/" + std::to_string(r.thread);
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

        const Fit fit = fit_for(r.phase, r.variant, r.probe, r.thread);
        const double base = control_slope(r.phase, r.variant, r.thread);
        const double per = (fit.slope - base) / (double)r.unroll;
        printf(
            "%-18s %-3c %-7s %-8s %-4d %12.2f %11.3f %8.4f%s\n",
            r.probe.c_str(),
            r.phase,
            r.variant.c_str(),
            r.unit.c_str(),
            r.thread,
            fit.slope,
            per,
            fit.r2,
            fit.r2 < 0.99 ? "  <-- NONLINEAR" : "");
        if (fit.r2 < 0.99) {
            for (int pi = 0; pi < NUM_PHASES; pi++) {
                if (PHASES[pi].letter == r.phase) {
                    fail[pi]++;
                }
            }
        }
    }

    // Monotonicity, for every phase including Q: more work must take more time.
    //
    // `q_ctrl` is exempt, and the exemption is not a convenience. It is the
    // subtrahend, not a datum -- exactly as `loop_overhead` is the subtrahend
    // for the slope phases -- and its cost is a STEP function of the burst
    // index rather than of the burst length: the cascade that emits 2^p copies
    // of a body executes p + 1 compare-and-branch pairs, so doubling n can
    // legitimately add nothing at all. Requiring it to grow with n would fail
    // every healthy phase Q run there is.
    for (size_t i = 0; i + 1 < rows.size(); i++) {
        const Row& a = rows[i];
        const Row& b = rows[i + 1];
        if (a.probe == "q_ctrl") {
            continue;
        }
        if (a.phase == b.phase && a.variant == b.variant && a.probe == b.probe &&
            a.thread == b.thread && b.n > a.n && b.cycles <= a.cycles) {
            printf("  NOT MONOTONE: %c/%s/%s thread %d: n=%u -> %u cycles, n=%u -> %u cycles\n",
                   a.phase, a.variant.c_str(), a.probe.c_str(), a.thread, a.n, a.cycles, b.n, b.cycles);
            for (int pi = 0; pi < NUM_PHASES; pi++) {
                if (PHASES[pi].letter == a.phase) {
                    fail[pi]++;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase T read-out: the fusion arithmetic. This is the whole point of the
    // phase and it is a set of DIFFERENCES, so it is printed rather than left
    // for the reader to do by eye off the table above.
    // -----------------------------------------------------------------------
    if (ran[1]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase T: does the instruction cache fuse adjacent .ttinsn words?\n");
        printf("%s\n", std::string(78, '-').c_str());
        printf(
            "  Every group below carries the same %d `addi`s. `fuse4` and `spread4`\n"
            "  additionally carry the same FOUR `.ttinsn` words and differ only in whether\n"
            "  any two of them are adjacent. BlackholeA0/.../PushTensixInstruction.md says\n"
            "  up to four adjacent ones fuse into a single cycle, so:\n"
            "    spread4 - fuse4  ==  3 cycles  ->  the cache fuses, as documented\n"
            "    spread4 - fuse4  ==  0 cycles  ->  it does not, and `riscv.ttinsn_fusion`\n"
            "                                       describes something a kernel cannot reach\n\n",
            RVBENCH_PAD);
        printf("  %-9s %-7s %-4s %12s %12s\n", "variant", "probe", "thr", "cyc/group", "vs tt_pad");
        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                const double base = control_slope('t', ts.variant, t);
                double pad = 0;
                bool have_pad = false;
                for (const char* name : {"tt_pad", "tt_fuse2", "tt_fuse4", "tt_spread4"}) {
                    const Fit fit = fit_for('t', ts.variant, name, t);
                    if (fit.slope == 0) {
                        continue;
                    }
                    const double per = (fit.slope - base) / (double)RVBENCH_GROUP_REPS;
                    if (std::string(name) == "tt_pad") {
                        pad = per;
                        have_pad = true;
                    }
                    printf("  %-9s %-7s %-4d %12.3f", ts.variant, name, t, per);
                    if (have_pad && std::string(name) != "tt_pad") {
                        printf(" %12.3f", per - pad);
                    }
                    printf("\n");
                }
                double f4 = 0, s4 = 0;
                const Fit ff = fit_for('t', ts.variant, "tt_fuse4", t);
                const Fit fs = fit_for('t', ts.variant, "tt_spread4", t);
                if (ff.slope != 0 && fs.slope != 0) {
                    f4 = (ff.slope - base) / (double)RVBENCH_GROUP_REPS;
                    s4 = (fs.slope - base) / (double)RVBENCH_GROUP_REPS;
                    printf(
                        "    -> spread4 - fuse4 = %+.3f cycles/group   (documented: +3.000)\n",
                        s4 - f4);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase C read-out: the branch-direction arithmetic.
    // -----------------------------------------------------------------------
    if (ran[2]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase C: is there a branch predictor, and what does a mispredict cost?\n");
        printf("%s\n", std::string(78, '-').c_str());
        printf(
            "  `c_nt` and `c_t` execute the IDENTICAL dynamic instruction sequence -- a\n"
            "  branch whose target is the address the not-taken path falls through to --\n"
            "  and differ in exactly one bit: whether it was taken. The `c_xor_*` trio adds\n"
            "  an `xori` so that the ALTERNATING pattern has a value to alternate on, and\n"
            "  carries it in all three so the mix is matched.\n\n");
        printf("  %-9s %-4s %10s %10s %10s %12s\n",
               "variant", "thr", "nt", "taken", "alt", "alt - nt");
        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                const double base = control_slope('c', ts.variant, t);
                auto per = [&](const char* name) {
                    return (fit_for('c', ts.variant, name, t).slope - base) / (double)RVBENCH_UNROLL;
                };
                printf("  %-9s %-4d %10.3f %10.3f %10s %12.3f    (bare)\n",
                       ts.variant, t, per("c_nt"), per("c_t"), "-", per("c_t") - per("c_nt"));
                printf("  %-9s %-4d %10.3f %10.3f %10.3f %12.3f    (with xori)\n",
                       ts.variant, t, per("c_xor_nt"), per("c_xor_t"), per("c_xor_alt"),
                       per("c_xor_alt") - per("c_xor_nt"));
            }
        }
        printf(
            "\n  All three equal -> no predictor, or no penalty. taken > nt -> a static\n"
            "  not-taken prediction. alt > taken ~ nt -> a real predictor, defeated by\n"
            "  alternation. tt_sim/perf/unit_costs.yaml records the bubble as 2 cycles on\n"
            "  Wormhole and 4 on Blackhole (`integer_unit.branch_mispredict_bubble`), which\n"
            "  is the SIZE of a mispredict; how OFTEN one happens is what this measures.\n");
    }

    // -----------------------------------------------------------------------
    // Phase Q read-out: the knee. Not a slope, and deliberately not gated on
    // linearity -- a straight line here would be the null result.
    // -----------------------------------------------------------------------
    if (ran[3]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase Q: how many .ttinsn words fit before the core is back-pressured?\n");
        printf("%s\n", std::string(78, '-').c_str());
        printf(
            "  Cycles for a burst of n, with the cascade's own compare-and-branch cost\n"
            "  (`q_ctrl`, which grows with n and is NOT a constant) subtracted point by\n"
            "  point. `marg` is the marginal cost of the instructions added since the\n"
            "  previous n. Below the queue depth it should be ~1.0 -- the core runs ahead\n"
            "  -- and above it, the backend unit's occupancy. The n at which it steps up is\n"
            "  the queue depth, and nothing in any document gives one, so this is\n"
            "  EXPLORATORY: there is no prediction here to confirm or refute.\n\n");
        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                for (const char* name : {"q_nop", "q_setdmareg", "q_adddmareg", "q_adddmareg_sync"}) {
                    printf("  %s [%s thread %d]\n", name, ts.variant, t);
                    printf("      %6s %10s %10s %8s\n", "n", "cycles", "net", "marg");
                    double prev_net = 0;
                    uint32_t prev_n = 0;
                    for (int kx = 0; kx < RVBENCH_MAX_POINTS; kx++) {
                        const uint32_t n = 1u << kx;
                        uint32_t c = 0, ctrl = 0;
                        if (!cycles_at('q', ts.variant, name, t, n, &c) ||
                            !cycles_at('q', ts.variant, "q_ctrl", t, n, &ctrl)) {
                            continue;
                        }
                        const double net = (double)c - (double)ctrl;
                        if (prev_n == 0) {
                            printf("      %6u %10u %10.1f %8s\n", n, c, net, "-");
                        } else {
                            printf("      %6u %10u %10.1f %8.3f\n", n, c, net,
                                   (net - prev_net) / (double)(n - prev_n));
                        }
                        prev_net = net;
                        prev_n = n;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase F read-out: cycles per instruction against footprint.
    // -----------------------------------------------------------------------
    if (ran[4]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase F: does a bigger loop body cost more per instruction?\n");
        printf("%s\n", std::string(78, '-').c_str());
        printf(
            "  Same instruction throughout; only the distance between the loop's two ends\n"
            "  changes. The ISA docs give the fetch PERIOD (one 128-bit L1 read per four\n"
            "  instructions) and no cache size and no miss cost, so a flat row is\n"
            "  consistent with the documentation and a cliff is a number nothing has\n"
            "  published. Both are results.\n\n");
        printf("  %-9s %-4s", "variant", "thr");
        for (const char* nm : {"f_64", "f_128", "f_256", "f_512", "f_1024", "f_2048"}) {
            printf(" %9s", nm);
        }
        printf("\n");
        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                const double base = control_slope('f', ts.variant, t);
                printf("  %-9s %-4d", ts.variant, t);
                for (int p = RVBENCH_P_F_64; p <= RVBENCH_P_F_2048; p++) {
                    const Fit fit = fit_for('f', ts.variant, PROBES[p].name, t);
                    // A probe the mask turned off contributed no rows, so its
                    // fit is the degenerate zero. Printing (0 - control)/unroll
                    // for it would put a small NEGATIVE cycles-per-instruction
                    // in the table, which is the one value this instrument says
                    // is impossible.
                    if (fit.slope == 0) {
                        printf(" %9s", "-");
                    } else {
                        printf(" %9.3f", (fit.slope - base) / (double)PROBES[p].unroll);
                    }
                }
                printf("\n");
            }
        }
    }

    // -----------------------------------------------------------------------
    // The instrumented-control check. See the header comment: without this a
    // run of 1.000s is ambiguous between "nothing back-pressures the core" and
    // "this benchmark measured nothing".
    // -----------------------------------------------------------------------
    if (ran[0]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("Is the instrument live? The probes with a non-trivial documented cost\n");
        printf("%s\n", std::string(78, '-').c_str());
        const double base = control_slope('r', "t1", 1);
        int above = 0, total = 0;
        for (const char* name : {"rv_mul_dep", "rv_div", "rv_load_chase", "rv_store_spread"}) {
            const Fit fit = fit_for('r', "t1", name, 1);
            if (fit.slope == 0) {
                continue;
            }
            const double per = (fit.slope - base) / (double)RVBENCH_UNROLL;
            total++;
            if (per > 1.5) {
                above++;
            }
            printf("  %-18s %8.3f cycles/instruction\n", name, per);
        }
        if (total == 0) {
            printf("  phase R was not run at t1; cannot check.\n");
        } else if (above == 0) {
            printf(
                "\n  ALL of them read ~1.0. Every one has a documented cost above one cycle\n"
                "  on at least one architecture (multiply, divide, an L1 load-use latency,\n"
                "  the five-cycle L1 store period), so this is the signature of a run that\n"
                "  measured nothing -- or of a device on which NOTHING back-pressures the\n"
                "  issuing core, which is exactly what tt-sim looks like with the cost\n"
                "  model off. On silicon, treat it as a broken run and say so.\n");
        } else {
            printf(
                "\n  %d of %d read above 1.0, so the timer resolves a real per-instruction\n"
                "  cost and a 1.000 elsewhere in this run is a finding rather than a floor.\n",
                above, total);
        }
    }

    printf("\nwrote %zu rows to %s\n", rows.size(), out_path.c_str());

    int status = 0;
    bool any_ok = false, any_bad = false;
    for (int pi = 0; pi < NUM_PHASES; pi++) {
        if (!ran[pi]) {
            continue;
        }
        const char L = (char)(PHASES[pi].letter - 32);  // upper-case, for the marker
        if (fail[pi] == 0) {
            printf("TTRVBENCH_VALID_%c: yes\n", L);
            any_ok = true;
        } else {
            printf("TTRVBENCH_VALID_%c: no (%d checks failed)\n", L, fail[pi]);
            status |= PHASES[pi].fail_bit;
            any_bad = true;
        }
    }
    if (!any_bad) {
        printf("TTRVBENCH_VALID: yes\n");
        printf("Completed successfully on the device\n");
    } else {
        printf(
            "TTRVBENCH_VALID: no -- but the verdict above is PER PHASE. Send the CSV\n"
            "  regardless: the rows of a phase that passed are unaffected by one that did\n"
            "  not, and %s.\n",
            any_ok ? "at least one phase here is usable" : "no phase here is usable");
    }
    return status;
}
