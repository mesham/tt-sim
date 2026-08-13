// energybench -- the loop-to-steady-state workload driver for ranking-level
// energy estimation.
//
// WHAT THIS MEASURES, AND WHAT IT DOES NOT
// ----------------------------------------
// `tt-smi` reports BOARD power at roughly 1 Hz. The kernels in this project run
// for microseconds. A single launch is invisible against the idle floor, so
// there is no way to weigh one launch on this instrument.
//
// So this program does not try. It opens the device ONCE, builds the program
// ONCE, and then calls `detail::LaunchProgram` back to back for a wall-clock
// duration long enough that a ~1 Hz sampler collects tens of samples. The
// quantity the pair (this program + the sampler) produces is therefore
//
//     STEADY-STATE REPEATED-KERNEL BOARD POWER, in watts, UNDER SUSTAINED LOAD,
//
// which is emphatically NOT "the energy of one launch". It is the average power
// of a board that is executing this kernel over and over, including the launch
// machinery between every pair of executions, and including DRAM, PHYs, ARC,
// PCIe and fans.
//
// THE SAMPLER RUNS WHILE THIS PROGRAM HOLDS THE DEVICE
// ----------------------------------------------------
// That works, and is measured: `tt-smi` 6.2.0 reads a busy chip at 69.0 W with
// `--arm rv` running, and the workload is unperturbed (404.3 launches/s while
// sampled against 407.5 and 418.8 unsampled). It needs tt-smi >= 4.0.0, where
// the default backend changed from Luwen to tt-umd -- on 3.0.32 every in-slot
// read panics in pyluwen ("Failed to map bar0_uc"), because BAR0's mapping is
// exclusive on that path, and the harness banks a session of zeros. `run_card.sh`
// refuses to start below 4.0.0 for exactly that reason; see ../README.md.
//
// Per-launch energy can only be recovered from that by DIVIDING BY THE MEASURED
// LAUNCH RATE, which is why this program reports `launches` and `wall_s` in its
// CSV row and why the analysis refuses to run without them. The simulator must
// predict the same quantity: activity per launch, times launches per second,
// plus a static board term. See ../README.md.
//
// THE ARMS
// --------
//   idle  a kernel that returns immediately -- launch machinery only
//   rv    a dependent integer chain on BRISC -- instr_retired, nothing else
//   noc   barriered DRAM->L1 reads on BRISC -- noc bytes, flat instruction count
//   mm    matmul_tiles on two resident tiles -- Matrix unit occupancy
//   sfpu  Int32 tile add on two resident tiles -- Vector unit occupancy
//
// `mm` and `sfpu` share a reader and a writer and differ only in which backend
// unit their inner loop occupies. That pairing is what makes a per-unit
// coefficient identifiable rather than collapsing into one "compute" column.
//
// USAGE
// -----
//   ./build/energybench --arm mm --inner 4096 --seconds 30
//   ./build/energybench --arm idle --iters 1 --csv out.csv --label sim
//
// Exactly one of --seconds / --iters is honoured; --iters wins if both are
// given. The default (--iters 1) is the smoke shape used against tt-sim.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t TILE_HW = 32 * 32;
constexpr uint32_t BF16_TILE_BYTES = TILE_HW * 2;
constexpr uint32_t INT32_TILE_BYTES = TILE_HW * 4;

// The `noc` arm's per-read transfer size. 2 KiB is one bf16 tile, which is the
// unit everything else in this tree moves, and it is congruent-aligned by
// construction so the simulator's NoC alignment check has nothing to say.
constexpr uint32_t NOC_XFER_BYTES = 2048;

struct Options {
    std::string arm = "idle";
    uint32_t inner = 1024;
    uint32_t iters = 1;
    double seconds = 0.0;
    std::string csv;
    std::string label;
};

void usage() {
    std::printf(
        "usage: energybench [--arm idle|rv|noc|mm|sfpu] [--inner N] [--iters N]\n"
        "                   [--seconds S] [--csv PATH] [--label TEXT]\n");
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "energybench: %s needs a value\n", what);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--arm") {
            const char* v = next("--arm");
            if (!v) return false;
            o.arm = v;
        } else if (a == "--inner") {
            const char* v = next("--inner");
            if (!v) return false;
            o.inner = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        } else if (a == "--iters") {
            const char* v = next("--iters");
            if (!v) return false;
            o.iters = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
            o.seconds = 0.0;
        } else if (a == "--seconds") {
            const char* v = next("--seconds");
            if (!v) return false;
            o.seconds = std::strtod(v, nullptr);
            o.iters = 0;
        } else if (a == "--csv") {
            const char* v = next("--csv");
            if (!v) return false;
            o.csv = v;
        } else if (a == "--label") {
            const char* v = next("--label");
            if (!v) return false;
            o.label = v;
        } else if (a == "-h" || a == "--help") {
            usage();
            return false;
        } else {
            std::fprintf(stderr, "energybench: unknown argument '%s'\n", a.c_str());
            usage();
            return false;
        }
    }
    if (o.arm != "idle" && o.arm != "rv" && o.arm != "noc" && o.arm != "mm" && o.arm != "sfpu") {
        std::fprintf(stderr, "energybench: unknown arm '%s'\n", o.arm.c_str());
        return false;
    }
    if (o.iters == 0 && o.seconds <= 0.0) {
        std::fprintf(stderr, "energybench: need --iters N or --seconds S\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse(argc, argv, opt)) {
        return 2;
    }

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    // Buffers. Every arm allocates the same DRAM footprint so a DRAM refresh
    // difference between arms cannot be mistaken for a compute difference.
    const uint32_t dram_bytes = INT32_TILE_BYTES;
    InterleavedBufferConfig dram_config{
        .device = device, .size = dram_bytes, .page_size = dram_bytes, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> src0 = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> src1 = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> dst = CreateBuffer(dram_config);

    std::vector<uint32_t> seed(dram_bytes / 4);
    for (size_t i = 0; i < seed.size(); i++) {
        seed[i] = static_cast<uint32_t>(i * 2654435761u);
    }
    tt::tt_metal::detail::WriteToBuffer(src0, seed);
    tt::tt_metal::detail::WriteToBuffer(src1, seed);

    const bool compute_arm = (opt.arm == "mm" || opt.arm == "sfpu");
    const bool bf16 = (opt.arm == "mm");
    const uint32_t tile_bytes = bf16 ? BF16_TILE_BYTES : INT32_TILE_BYTES;
    const tt::DataFormat fmt = bf16 ? tt::DataFormat::Float16_b : tt::DataFormat::Int32;

    if (compute_arm) {
        for (auto idx : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_16}) {
            CircularBufferConfig cfg =
                CircularBufferConfig(tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes);
            tt_metal::CreateCircularBuffer(program, core, cfg);
        }

        KernelHandle reader = CreateKernel(
            program,
            "kernels/dataflow/tile_reader.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, reader, core, {src0->address(), src1->address(), tile_bytes});

        KernelHandle writer = CreateKernel(
            program,
            "kernels/dataflow/tile_writer.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, writer, core, {dst->address(), tile_bytes});

        KernelHandle compute = CreateKernel(
            program,
            bf16 ? "kernels/compute/mm_kernel.cpp" : "kernels/compute/sfpu_kernel.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = {},
            });
        SetRuntimeArgs(program, compute, core, {opt.inner});
    } else {
        // The scalar arms all get one scratch circular buffer so the kernel has
        // a legal L1 address to touch without the host having to hand it one.
        CircularBufferConfig cfg = CircularBufferConfig(NOC_XFER_BYTES, {{CBIndex::c_0, tt::DataFormat::Int32}})
                                       .set_page_size(CBIndex::c_0, NOC_XFER_BYTES);
        tt_metal::CreateCircularBuffer(program, core, cfg);

        const char* path = "kernels/dataflow/idle_kernel.cpp";
        if (opt.arm == "rv") {
            path = "kernels/dataflow/rv_kernel.cpp";
        } else if (opt.arm == "noc") {
            path = "kernels/dataflow/noc_kernel.cpp";
        }
        KernelHandle k = CreateKernel(
            program,
            path,
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        if (opt.arm == "rv") {
            SetRuntimeArgs(program, k, core, {opt.inner});
        } else if (opt.arm == "noc") {
            SetRuntimeArgs(program, k, core, {opt.inner, src0->address(), NOC_XFER_BYTES});
        } else {
            SetRuntimeArgs(program, k, core, {});
        }
    }

    // ---- the steady-state loop -------------------------------------------
    // Everything above happens once. The timed region is launches only: no
    // host allocation, no buffer write, no kernel build.
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t launches = 0;
    for (;;) {
        tt::tt_metal::detail::LaunchProgram(device, program, true, true);
        launches++;
        if (opt.iters != 0) {
            if (launches >= opt.iters) break;
        } else {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (elapsed >= opt.seconds) break;
        }
    }
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    CloseDevice(device);

    const double rate = wall_s > 0.0 ? static_cast<double>(launches) / wall_s : 0.0;
    std::printf(
        "ENERGYBENCH arm=%s inner=%u launches=%llu wall_s=%.6f launches_per_s=%.3f label=%s\n",
        opt.arm.c_str(),
        opt.inner,
        static_cast<unsigned long long>(launches),
        wall_s,
        rate,
        opt.label.c_str());

    if (!opt.csv.empty()) {
        const bool fresh = [&] {
            FILE* probe = std::fopen(opt.csv.c_str(), "r");
            if (!probe) return true;
            const int c = std::fgetc(probe);
            std::fclose(probe);
            return c == EOF;
        }();
        FILE* f = std::fopen(opt.csv.c_str(), "a");
        if (!f) {
            std::fprintf(stderr, "energybench: cannot append to %s\n", opt.csv.c_str());
            return 1;
        }
        if (fresh) {
            std::fprintf(f, "label,arm,inner,launches,wall_s,launches_per_s\n");
        }
        std::fprintf(
            f,
            "%s,%s,%u,%llu,%.6f,%.3f\n",
            opt.label.c_str(),
            opt.arm.c_str(),
            opt.inner,
            static_cast<unsigned long long>(launches),
            wall_s,
            rate);
        std::fclose(f);
    }

    std::printf("Completed successfully on the device\n");
    return 0;
}
