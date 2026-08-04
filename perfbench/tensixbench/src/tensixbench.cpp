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
//     (default) / `--dvalid-per-thread` selects how the three MATH probes get
//     their SrcA/SrcB valid bits; that is experiment X1 of
//     docs/plans/matrix-unit-thread-contention.md and it is the only thing that
//     changes what phase A measures.
//   * Phase B times `matmul_tiles` at three math fidelities. The absolute
//     number is a confounded composite; the DIFFERENCE between fidelities is
//     not, and is a direct check on `fidelity_phases.mvmuls_per_tile`. The
//     fidelity setting is known to reach the math thread -- see the disassembly
//     evidence in kernels/compute/matmul_fidelity.cpp -- so a null difference
//     means the loop was not math-bound, which is a result and not a fault.
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
    // Experiment X1 of docs/plans/matrix-unit-thread-contention.md. Default is
    // the de-confounded setup: exactly one SETDVALID regardless of thread count.
    uint32_t dvalid_once = 1;
    // Which phase B fidelities to launch. On hardware, all three, always. On the
    // simulator a single fidelity is minutes and the three together have never
    // completed, so being able to ask for one is the difference between phase B
    // being checkable at all and not.
    std::string fidelity_filter = "LoFi,HiFi2,HiFi4";

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
            dvalid_once = 0;
        } else if (a == "--dvalid-once") {
            dvalid_once = 1;
        } else if (a == "--phase") {
            phases = next();
        } else if (a == "--fidelities") {
            fidelity_filter = next();
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
                "  --dvalid-once         (default) exactly one SETDVALID before the three\n"
                "                        MATH probes, issued by thread 1, barriered. The\n"
                "                        de-confounded setup -- experiment X1 of\n"
                "                        docs/plans/matrix-unit-thread-contention.md.\n"
                "  --dvalid-per-thread   the original setup: one SETDVALID per ACTIVE\n"
                "                        thread, so the thread count and the SrcA/SrcB\n"
                "                        bank state move together. Run this to reproduce\n"
                "                        the 6.1x/12.1x MVMUL result and diff it against\n"
                "                        the default. Written to a distinct default CSV.\n"
                "  --phase a|b|ab        which phases to run (default ab)\n"
                "  --fidelities LIST     comma-separated subset of LoFi,HiFi2,HiFi4 for\n"
                "                        phase B (default all three). For the simulator,\n"
                "                        where one fidelity is minutes; on hardware leave\n"
                "                        it alone -- the DIFFERENCE needs at least two.\n"
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
        // The non-default dvalid setup gets its own name so the X1 pair cannot
        // silently overwrite each other.
        out_path = "tensixbench-" + arch + (dvalid_once ? "" : "-dvalid-per-thread") + ".csv";
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
        fprintf(
            csv,
            "# arch=%s magic=0x%08X unroll=%u probe_mask=0x%X dvalid_setup=%s mm_block=%u "
            "fidelities=%s\n",
            arch.c_str(),
            TTBENCH_MAGIC,
            TTBENCH_UNROLL,
            probe_mask,
            dvalid_once ? "once" : "per-thread",
            TTBENCH_MM_BLOCK,
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
                {results_addr, barrier_addr, base_blocks, probe_mask, ts.mask, dvalid_once});

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
            if (("," + fidelity_filter + ",").find("," + std::string(f.variant) + ",") == std::string::npos) {
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
        printf(
            "dvalid setup: %s -- %s\n",
            dvalid_once ? "once" : "per-thread",
            dvalid_once ? "one SETDVALID for the tile (X1, de-confounded)"
                        : "one SETDVALID per ACTIVE thread (original, confounded)");
    }
    printf("\n");
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
