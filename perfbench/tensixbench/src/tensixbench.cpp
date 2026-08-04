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
//     result can be told apart from a unit-limited one.
//   * Phase B times `matmul_tiles` at three math fidelities. The absolute
//     number is a confounded composite; the DIFFERENCE between fidelities is
//     not, and is a direct check on `fidelity_phases.mvmuls_per_tile`.
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

}  // namespace

int main(int argc, char** argv) {
    uint32_t base_blocks = 4;
    uint32_t base_iters = 8;
    uint32_t probe_mask = ALL_PROBES;
    std::string phases = "ab";
    std::string out_path;

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
        } else if (a == "--phase") {
            phases = next();
        } else if (a == "--out") {
            out_path = next();
        } else if (a == "-h" || a == "--help") {
            printf(
                "usage: tensixbench [--blocks N] [--iters N] [--probes 0xMASK]\n"
                "                   [--no-dvalid-probes] [--phase a|b|ab] [--out FILE]\n"
                "\n"
                "  --blocks N            phase A sweeps blocks = N, 2N, 3N, 4N of %d\n"
                "                        instructions each (default 4)\n"
                "  --iters N             phase B sweeps N, 2N, 3N, 4N matmul_tiles calls\n"
                "                        (default 8)\n"
                "  --probes 0xMASK       bit i enables probe i (default all %d)\n"
                "  --no-dvalid-probes    drop MVMUL/ELWADD/ELWMUL, the only probes that\n"
                "                        depend on the SrcA/SrcB valid bits. Use this if\n"
                "                        phase A hangs, and say so in the report.\n"
                "  --phase a|b|ab        which phases to run (default ab)\n"
                "  --out FILE            CSV path (default tensixbench-<arch>.csv)\n",
                TTBENCH_UNROLL,
                TTBENCH_NUM_PROBES);
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

    IDevice* device = CreateDevice(0);
    const std::string arch = arch_name(device);
    if (out_path.empty()) {
        out_path = "tensixbench-" + arch + ".csv";
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
    int failures = 0;

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
        fprintf(
            csv,
            "# arch=%s magic=0x%08X unroll=%u probe_mask=0x%X\n",
            arch.c_str(),
            TTBENCH_MAGIC,
            TTBENCH_UNROLL,
            probe_mask);
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
            Program program = CreateProgram();

            // A compute kernel needs at least one circular buffer for the
            // generated CB descriptor list to be well formed. Nothing reads it.
            CircularBufferConfig dummy_cb =
                CircularBufferConfig(1024, {{CBIndex::c_0, tt::DataFormat::UInt32}}).set_page_size(CBIndex::c_0, 1024);
            CreateCircularBuffer(program, core, dummy_cb);

            KernelHandle bench = CreateKernel(
                program, "kernels/compute/raw_probes.cpp", core, ComputeConfig{});
            SetRuntimeArgs(
                program, bench, core, {results_addr, barrier_addr, base_blocks, probe_mask, ts.mask});

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
            Program program = CreateProgram();

            // Operand buffers are several pages deep so the feeder can run
            // ahead of the compute loop; the output buffer holds one page per
            // measurement point so nothing has to drain it mid-run.
            constexpr uint32_t in_pages = 4;
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
            for (int t = 0; t < TTBENCH_MAX_THREADS; t++) {
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
    printf("%s\n\n", std::string(78, '=').c_str());
    printf("%-14s %-7s %-6s %-4s %12s %10s %8s\n", "probe", "variant", "unit", "thr", "cyc/block", "cyc/instr", "R^2");

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
        }
        printf(
            "%-14s %-7s %-6s %-4d %12.2f %10.3f %8.4f%s\n",
            r.probe.c_str(),
            r.variant.c_str(),
            r.unit.c_str(),
            r.thread,
            fit.slope,
            per_instr,
            fit.r2,
            fit.r2 < 0.99 ? "  <-- NONLINEAR" : "");
        if (fit.r2 < 0.99) {
            failures++;
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
            failures++;
        }
    }

    printf("\nwrote %zu rows to %s\n", rows.size(), out_path.c_str());
    if (failures == 0) {
        printf("TTBENCH_VALID: yes\n");
        printf("Completed successfully on the device\n");
    } else {
        printf("TTBENCH_VALID: no (%d checks failed -- do not use this run)\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
