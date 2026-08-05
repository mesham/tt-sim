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
//   * Seven phases, run and scored INDEPENDENTLY: R (straight-line RV32IM),
//     T (the `.ttinsn` issue path), C (control flow), Q (the Tensix instruction
//     queue's depth), F (instruction footprint), S (is that queue shared
//     between the TRISCs or private to each?), G (the footprints between
//     phase F's 1024 and 2048, which narrow the boundary it found). Each sets
//     its own bit in the exit status and prints its own TTRVBENCH_VALID_<X>
//     line.
//   * Phase Q is deliberately NOT gated on linearity. It is looking for a knee;
//     a straight line there would be the null result, not the healthy one.
//   * Phase S's answer is a RATIO BETWEEN THREAD COUNTS and never a level. One
//     variant produces no verdict; the phase says so rather than reporting the
//     one number it does have.
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
#include <map>
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
//
// `points` and `n0` are the probe's own sweep axis, likewise per probe:
//
//   n0 == 0   a SLOPE probe. n = base_blocks * (k + 1), `points` of them.
//   n0 != 0   a BURST probe. n = n0 << k, i.e. geometric from n0. The phase-Q
//             cascade sweeps n = 1 ... 128 and the phase-Q loop form sweeps
//             n = 16 ... 1024, which is why this cannot be one global rule.
// ---------------------------------------------------------------------------
// `issuer_only` marks a probe only RVBENCH_S_ISSUER measures. The other active
// threads spin through it and record nothing, so emitting their rows would put
// a run of zeroes in the CSV that says "not measured" and reads, to the
// monotonicity gate, as "did not grow".
//
// `gset` is which compile-time footprint set of phase G carries the probe.
// ANY_GSET for everything outside phase G and for `g_1024`, which every set
// carries as its in-build flat anchor.
#define ANY_GSET (-1)

struct Probe {
    const char* name;
    const char* unit;
    char phase;
    uint32_t unroll;
    uint32_t points;
    uint32_t n0;
    bool issuer_only;
    int gset;
};

#define SLOPE_PROBE RVBENCH_SLOPE_POINTS, 0, false, ANY_GSET
#define CASCADE_PROBE RVBENCH_MAX_POINTS, 1, false, ANY_GSET
#define LOOP_PROBE RVBENCH_Q_LOOP_POINTS, RVBENCH_Q_LOOP_MIN_N, false, ANY_GSET
#define S_PROBE RVBENCH_S_POINTS, RVBENCH_S_MIN_N, false, ANY_GSET
#define S_SOLO_PROBE RVBENCH_S_POINTS, RVBENCH_S_MIN_N, true, ANY_GSET
#define G_PROBE(SET) RVBENCH_SLOPE_POINTS, 0, false, SET

const Probe PROBES[RVBENCH_NUM_PROBES] = {
    // Phase R -- but the control is shared by every slope phase, so it carries
    // phase '*' and is emitted alongside whichever phase is running.
    {"loop_overhead", "-", '*', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_addi_indep", "RV", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_addi_dep", "RV", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_mul_indep", "RV", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_mul_dep", "RV", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_div", "RV", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_load_chase", "RV_LSU", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_load_indep", "RV_LSU", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_store_spread", "RV_LSU", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_store_coalesce", "RV_LSU", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_load_stack", "RV_LSU", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    {"rv_store_stack", "RV_LSU", 'r', RVBENCH_UNROLL, SLOPE_PROBE},
    // Phase T.
    {"tt_nop", "NONE", 't', RVBENCH_UNROLL, SLOPE_PROBE},
    {"tt_sfpnop", "SFPU", 't', RVBENCH_UNROLL, SLOPE_PROBE},
    {"tt_setdmareg", "THCON", 't', RVBENCH_UNROLL, SLOPE_PROBE},
    {"tt_adddmareg", "THCON", 't', RVBENCH_UNROLL, SLOPE_PROBE},
    {"tt_pad", "TTINSN", 't', RVBENCH_GROUP_REPS, SLOPE_PROBE},
    {"tt_fuse2", "TTINSN", 't', RVBENCH_GROUP_REPS, SLOPE_PROBE},
    {"tt_fuse4", "TTINSN", 't', RVBENCH_GROUP_REPS, SLOPE_PROBE},
    {"tt_spread4", "TTINSN", 't', RVBENCH_GROUP_REPS, SLOPE_PROBE},
    // Phase C.
    {"c_ctrl_xor", "RV_BR", 'c', RVBENCH_UNROLL, SLOPE_PROBE},
    {"c_nt", "RV_BR", 'c', RVBENCH_UNROLL, SLOPE_PROBE},
    {"c_t", "RV_BR", 'c', RVBENCH_UNROLL, SLOPE_PROBE},
    {"c_xor_nt", "RV_BR", 'c', RVBENCH_UNROLL, SLOPE_PROBE},
    {"c_xor_t", "RV_BR", 'c', RVBENCH_UNROLL, SLOPE_PROBE},
    {"c_xor_alt", "RV_BR", 'c', RVBENCH_UNROLL, SLOPE_PROBE},
    {"c_jal", "RV_BR", 'c', RVBENCH_UNROLL, SLOPE_PROBE},
    // Phase Q, cascade form -- `n` is a burst length, not a block count, and
    // `unroll` is 1.
    {"q_ctrl", "TTQUEUE", 'q', 1, CASCADE_PROBE},
    {"q_nop", "TTQUEUE", 'q', 1, CASCADE_PROBE},
    {"q_setdmareg", "TTQUEUE", 'q', 1, CASCADE_PROBE},
    {"q_adddmareg", "TTQUEUE", 'q', 1, CASCADE_PROBE},
    {"q_adddmareg_sync", "TTQUEUE", 'q', 1, CASCADE_PROBE},
    // Phase F -- the unroll IS the footprint.
    {"f_64", "RV_FETCH", 'f', 64, SLOPE_PROBE},
    {"f_128", "RV_FETCH", 'f', 128, SLOPE_PROBE},
    {"f_256", "RV_FETCH", 'f', 256, SLOPE_PROBE},
    {"f_512", "RV_FETCH", 'f', 512, SLOPE_PROBE},
    {"f_1024", "RV_FETCH", 'f', 1024, SLOPE_PROBE},
    {"f_2048", "RV_FETCH", 'f', 2048, SLOPE_PROBE},
    // Phase Q, loop form -- the same question out to n = 1024, from a burst
    // whose instruction FOOTPRINT does not grow with n. Appended rather than
    // filed next to the cascade probes so that no existing slot number moves;
    // see rvbench_layout.h.
    {"q_loop_addi", "TTQUEUE", 'q', 1, LOOP_PROBE},
    {"q_loop_adddmareg", "TTQUEUE", 'q', 1, LOOP_PROBE},
    {"q_loop_adddmareg_sync", "TTQUEUE", 'q', 1, LOOP_PROBE},
    // Phase S -- shared queue or per-thread queue. Same burst form throughout,
    // n = 4 ... 512, `unroll` 1 because n is a burst length.
    {"s_loop_addi", "TTQUEUE", 's', 1, S_PROBE},
    {"s_co_plain", "TTQUEUE", 's', 1, S_PROBE},
    {"s_co_repeat", "TTQUEUE", 's', 1, S_PROBE},
    {"s_co_sync", "TTQUEUE", 's', 1, S_PROBE},
    {"s_solo_plain", "TTQUEUE", 's', 1, S_SOLO_PROBE},
    {"s_solo_sync", "TTQUEUE", 's', 1, S_SOLO_PROBE},
    // Phase G -- the intermediate footprints, one per compile-time set, each
    // against a 1024-instruction body compiled into the same kernel. `gset`
    // is the set that carries the probe, or ANY_GSET for one every set does.
    {"g_1024", "RV_FETCH", 'g', 1024, G_PROBE(ANY_GSET)},
    {"g_1280", "RV_FETCH", 'g', 1280, G_PROBE(0)},
    {"g_1536", "RV_FETCH", 'g', 1536, G_PROBE(1)},
    {"g_1792", "RV_FETCH", 'g', 1792, G_PROBE(2)},
};

// 64 bits wide, because there are more than 32 probes and phase F's six live
// above slot 31. It travels to the kernel as two runtime-arg words.
constexpr uint64_t ALL_PROBES = (1ull << RVBENCH_NUM_PROBES) - 1ull;

const char PHASE_LETTERS[] = "rtcqfsg";

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
    {'s', "RVBENCH_PHASE_S", RVBENCH_PHASE_S_BIT, 32},
    {'g', "RVBENCH_PHASE_G", RVBENCH_PHASE_G_BIT, 64},
};
constexpr int NUM_PHASES = 7;

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
    int gset = 0;

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
        } else if (a == "--gset") {
            gset = (int)std::stoul(next());
        } else if (a == "--out") {
            out_path = next();
        } else if (a == "-h" || a == "--help") {
            printf(
                "usage: riscvbench [--blocks N] [--probes 0xMASK] [--phase LETTERS]\n"
                "                  [--variants LIST] [--gset N] [--out FILE]\n"
                "\n"
                "  --blocks N        the slope phases sweep blocks = N, 2N, 3N, 4N\n"
                "                    (default 4). Phase Q ignores it: its sweep is over\n"
                "                    BURST LENGTH, which is the axis a queue depth lives\n"
                "                    on -- 1..128 straight-line and 16..1024 in a loop\n"
                "                    body of fixed size, so that the longer bursts do not\n"
                "                    also grow the instruction stream.\n"
                "  --probes 0xMASK   bit i enables probe i (default all %d). Probe 0 is\n"
                "                    the empty-loop control and every slope phase needs\n"
                "                    it; clearing bit 0 makes those phases unreadable.\n"
                "  --phase LETTERS   any subset of `%s` (default all six):\n"
                "                      r  straight-line RV32IM -- the baseline, and the\n"
                "                         four probes tt-sim already moves\n"
                "                      t  the `.ttinsn` issue path, including the\n"
                "                         instruction-cache fusion experiment\n"
                "                      c  branch direction and mispredict cost\n"
                "                      q  the Tensix instruction queue's depth\n"
                "                      f  instruction footprint, 256 B to 8 KiB\n"
                "                      s  is that queue SHARED between the TRISCs or\n"
                "                         private to each? Needs --variants t1,t2 at\n"
                "                         least; the answer is a comparison between\n"
                "                         thread counts and one alone says nothing\n"
                "                      g  the footprints BETWEEN phase F's 1024 and\n"
                "                         2048, one per --gset, each against a 1024\n"
                "                         body in the same build\n"
                "                    Each phase is a SEPARATE kernel build: its probe\n"
                "                    bodies are only compiled when it is selected, so a\n"
                "                    phase F run does not carry phase R's text and vice\n"
                "                    versa. That matters -- phase F's bodies are 16 KiB\n"
                "                    of instructions and would otherwise BE the thing\n"
                "                    phase F measures.\n"
                "  --variants LIST   comma-separated subset of t1,t2,t3 (default all\n"
                "                    three), i.e. how many TRISCs run the identical\n"
                "                    probes at once. Each is a separate program launch.\n"
                "  --gset N          phase G's compile-time footprint set, 0..%d\n"
                "                    (default 0): 0 is 1024+1280, 1 is 1024+1536, 2 is\n"
                "                    1024+1792. It is a SET rather than a flag because\n"
                "                    all three intermediates in one kernel exceed\n"
                "                    tt-metal's kernel config buffer -- measured, see\n"
                "                    rvbench_layout.h. Run phase G three times.\n"
                "  --out FILE        CSV path (default riscvbench-<arch>.csv)\n",
                RVBENCH_NUM_PROBES,
                PHASE_LETTERS,
                RVBENCH_G_SETS - 1);
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
    if (gset < 0 || gset >= RVBENCH_G_SETS) {
        fprintf(stderr, "--gset must be in 0..%d\n", RVBENCH_G_SETS - 1);
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
    int fail[NUM_PHASES] = {0, 0, 0, 0, 0, 0, 0};
    bool ran[NUM_PHASES] = {false, false, false, false, false, false, false};
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
            "base_blocks=%u gset=%d group_reps=%u pad=%u stack_addr=0x%08X "
            "scratch_addr=0x%08X div_dividend=0x12345678 div_divisor=3\n",
            arch.c_str(),
            RVBENCH_MAGIC,
            (unsigned long long)probe_mask,
            phases.c_str(),
            variant_filter.c_str(),
            base_blocks,
            gset,
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

            // Phase G additionally carries which compile-time footprint set it
            // is building, because all three intermediates in one kernel
            // exceed tt-metal's kernel config buffer.
            std::map<std::string, std::string> defines = {{ph.define, "1"}};
            if (ph.letter == 'g') {
                defines["RVBENCH_G_SET"] = std::to_string(gset);
            }
            KernelHandle bench = CreateKernel(
                program, "kernels/compute/rv_probes.cpp", core, ComputeConfig{.defines = defines});
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
                    // A phase-S solo probe is measured by one thread; the
                    // others spin through it and record 0. Emitting their rows
                    // would put "not measured" in the CSV as a number.
                    if (probe.issuer_only && t != RVBENCH_S_ISSUER) {
                        continue;
                    }
                    // A phase-G probe belonging to another --gset was not
                    // compiled into this kernel and did not run.
                    if (probe.gset != ANY_GSET && probe.gset != gset) {
                        continue;
                    }
                    // Each probe carries its own point count and n-mapping: the
                    // slope phases sweep block count, the phase-Q cascade
                    // sweeps burst length from 1 and the phase-Q loop form
                    // sweeps it from 16. A single global rule here would
                    // mislabel every loop-form point by a factor of 16.
                    for (uint32_t kx = 0; kx < probe.points; kx++) {
                        const uint32_t n =
                            probe.n0 ? (probe.n0 << kx) : base_blocks * (kx + 1);
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
    // `q_ctrl` is exempt, and the exemption is not a convenience. It is not a
    // datum: it runs the cascade with an EMPTY body, so it has no burst to be
    // monotone in. All seven of the cascade's `if (p >= k)` tests execute at
    // every burst length, so its instruction count is the same at n=1 and
    // n=128 and what varies is only which branches are taken and which blocks
    // are touched first -- which is non-monotonic on silicon, by 6 to 23
    // cycles, reproducibly. Requiring it to grow with n would fail every
    // healthy phase Q run there is. (The 2026-08-05 run also retired the idea
    // that this control could be SUBTRACTED point by point; see the phase Q
    // read-out.) No other probe is exempt, including the loop-form probes
    // added for the n = 1024 extension: those really do have to grow with n.
    //
    // PHASE S CARRIES A TOLERANCE, and it is measured rather than chosen. Its
    // reference burst is four instructions, so the step from n=4 to n=8 is
    // worth ~12 cycles at one issuing thread -- comparable to what a single
    // cold, once-only burst scatters by, which is why phase Q's cascade failed
    // 17 monotonicity checks at n <= 16 on silicon. Rather than exempt the
    // phase (which would stop the gate seeing a genuinely broken point) or
    // pick a constant, `s_co_repeat` runs `s_co_plain` a second time and the
    // largest disagreement between them across the whole run is what one raw
    // point can be wrong by. A phase-S pair is flagged only when it falls by
    // MORE than that. Every other phase keeps a tolerance of zero, unchanged.
    uint32_t s_noise = 0;
    for (const auto& a : rows) {
        if (a.phase != 's' || a.probe != "s_co_plain") {
            continue;
        }
        for (const auto& b : rows) {
            if (b.phase == 's' && b.probe == "s_co_repeat" && b.variant == a.variant &&
                b.thread == a.thread && b.n == a.n) {
                const uint32_t d = a.cycles > b.cycles ? a.cycles - b.cycles : b.cycles - a.cycles;
                if (d > s_noise) {
                    s_noise = d;
                }
            }
        }
    }
    for (size_t i = 0; i + 1 < rows.size(); i++) {
        const Row& a = rows[i];
        const Row& b = rows[i + 1];
        if (a.probe == "q_ctrl") {
            continue;
        }
        const uint32_t tolerance = a.phase == 's' ? s_noise : 0;
        if (a.phase == b.phase && a.variant == b.variant && a.probe == b.probe &&
            a.thread == b.thread && b.n > a.n && b.cycles + tolerance <= a.cycles) {
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
        // The cycles-per-instruction of one probe between the two furthest
        // apart burst lengths at or above `min_n`. A DIFFERENCE OF RAW POINTS,
        // deliberately: every phase-Q point carries the same fixed cost -- two
        // clock reads, a barrier, and (in the cascade form) all seven of the
        // cascade's `if (p >= k)` tests, which are evaluated at every burst
        // length -- so subtracting one point from another cancels it exactly.
        // Nothing here subtracts `q_ctrl` point by point; the 2026-08-05
        // silicon run showed that the premise licensing that subtraction was
        // false and that it was injecting up to +-17 cycles of structured
        // error. `q_ctrl` is this phase's declared NOISE FLOOR instead.
        struct Span {
            double rate;
            uint32_t lo, hi;
            uint32_t c_lo, c_hi;
            bool ok;
        };
        auto span_rate = [&](const std::string& variant, const char* probe, int thread,
                             uint32_t min_n) {
            Span s{0.0, 0, 0, 0, 0, false};
            for (const auto& r : rows) {
                if (r.phase != 'q' || r.variant != variant || r.probe != probe ||
                    r.thread != thread || r.n < min_n) {
                    continue;
                }
                if (s.lo == 0 || r.n < s.lo) {
                    s.lo = r.n;
                    s.c_lo = r.cycles;
                }
                if (r.n > s.hi) {
                    s.hi = r.n;
                    s.c_hi = r.cycles;
                }
            }
            if (s.lo == 0 || s.hi <= s.lo) {
                return s;
            }
            s.rate = ((double)s.c_hi - (double)s.c_lo) / (double)(s.hi - s.lo);
            s.ok = true;
            return s;
        };

        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase Q: how far does the core run ahead of the Tensix backend?\n");
        printf("%s\n", std::string(78, '-').c_str());
        printf(
            "  EXPLORATORY. Nothing in the ISA documentation or either vendor tree gives\n"
            "  a Tensix instruction queue depth, so there is no prediction here to\n"
            "  confirm or refute -- only a number nothing has measured.\n"
            "\n"
            "  CYCLES BELOW ARE RAW and every derived number is a DIFFERENCE of two of\n"
            "  them, over the widest span available. `q_ctrl` is NOT subtracted point by\n"
            "  point: the cascade evaluates all seven of its `if (p >= k)` tests at EVERY\n"
            "  burst length, so its cost is constant in n rather than growing with it, and\n"
            "  what does vary -- branch directions, first touch of the deeper blocks -- is\n"
            "  non-monotonic. It is the noise floor, printed as such.\n"
            "\n"
            "  TWO BURST FORMS, because the burst had to get longer and a longer\n"
            "  straight-line burst is a longer INSTRUCTION STREAM:\n"
            "    cascade  n = 1..128     2^p copies emitted straight-line. At n = 1024\n"
            "                            this would be 4 KiB of text, in the octave where\n"
            "                            phase F found a fetch cliff -- and a fetch cost\n"
            "                            that grows with n is indistinguishable from\n"
            "                            back-pressure, which is what this phase reads.\n"
            "                            So the cascade was NOT extended.\n"
            "    loop     n = 16..1024   n/%d iterations of one %d-instruction block: %d\n"
            "                            bytes of text at every n, so no difference\n"
            "                            between two loop points can be instruction\n"
            "                            fetch. `q_loop_addi` is the same loop with\n"
            "                            `addi` bodies and measures what the form itself\n"
            "                            costs, back edge included.\n"
            "  The forms OVERLAP at n = 16..128 and the overlap is checked below: it is\n"
            "  the evidence that changing the form did not change the quantity.\n\n",
            RVBENCH_Q_LOOP_BLOCK,
            RVBENCH_Q_LOOP_BLOCK,
            RVBENCH_Q_LOOP_BLOCK * 4);

        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                printf("  [%s thread %d]\n", ts.variant, t);
                for (int form = 0; form < 2; form++) {
                    const char* const cascade_probes[] = {
                        "q_ctrl", "q_nop", "q_setdmareg", "q_adddmareg", "q_adddmareg_sync"};
                    const char* const loop_probes[] = {
                        "q_loop_addi", "q_loop_adddmareg", "q_loop_adddmareg_sync"};
                    const char* const* names = form == 0 ? cascade_probes : loop_probes;
                    const int count = form == 0 ? 5 : 3;
                    const uint32_t n0 = form == 0 ? 1u : RVBENCH_Q_LOOP_MIN_N;
                    const uint32_t points = form == 0 ? RVBENCH_MAX_POINTS : RVBENCH_Q_LOOP_POINTS;
                    // A form whose probes were all masked off contributes
                    // nothing; do not print an empty table for it.
                    uint32_t probe_cycles = 0;
                    if (!cycles_at('q', ts.variant, names[0], t, n0, &probe_cycles)) {
                        continue;
                    }
                    printf("    %-22s", form == 0 ? "cascade, raw cycles" : "loop, raw cycles");
                    for (uint32_t kx = 0; kx < points; kx++) {
                        printf("%7u", n0 << kx);
                    }
                    printf("%10s\n", "rate");
                    for (int i = 0; i < count; i++) {
                        printf("    %-22s", names[i]);
                        for (uint32_t kx = 0; kx < points; kx++) {
                            uint32_t c = 0;
                            if (cycles_at('q', ts.variant, names[i], t, n0 << kx, &c)) {
                                printf("%7u", c);
                            } else {
                                printf("%7s", "-");
                            }
                        }
                        const Span s = span_rate(ts.variant, names[i], t, RVBENCH_Q_LOOP_MIN_N);
                        if (std::string(names[i]) == "q_ctrl") {
                            printf("%10s", "(noise)");
                        } else if (s.ok) {
                            printf("%10.3f", s.rate);
                        } else {
                            printf("%10s", "-");
                        }
                        printf("\n");
                    }
                }
                // The noise floor, from the cascade control's own spread.
                uint32_t ctrl_lo = 0, ctrl_hi = 0;
                bool have_ctrl = false;
                for (const auto& r : rows) {
                    if (r.phase != 'q' || r.variant != ts.variant || r.probe != "q_ctrl" ||
                        r.thread != t) {
                        continue;
                    }
                    if (!have_ctrl || r.cycles < ctrl_lo) {
                        ctrl_lo = r.cycles;
                    }
                    if (!have_ctrl || r.cycles > ctrl_hi) {
                        ctrl_hi = r.cycles;
                    }
                    have_ctrl = true;
                }
                if (have_ctrl) {
                    printf(
                        "    noise floor: `q_ctrl` spans %u-%u cycles (%u) across the burst "
                        "lengths;\n      that is what one raw point can be wrong by.\n",
                        ctrl_lo, ctrl_hi, ctrl_hi - ctrl_lo);
                }

                // -- does the form change the answer? ------------------------
                const Span cascade_over = span_rate(ts.variant, "q_adddmareg", t,
                                                    RVBENCH_Q_LOOP_MIN_N);
                bool overlap_printed = false;
                for (uint32_t n = RVBENCH_Q_LOOP_MIN_N; n <= 128; n *= 2) {
                    uint32_t a = 0, b = 0;
                    if (!cycles_at('q', ts.variant, "q_adddmareg", t, n, &a) ||
                        !cycles_at('q', ts.variant, "q_loop_adddmareg", t, n, &b)) {
                        continue;
                    }
                    if (!overlap_printed) {
                        printf(
                            "    form check, ADDDMAREG burst, cascade vs loop where both run:\n"
                            "      %8s %10s %10s %10s\n", "n", "cascade", "loop", "loop-casc");
                        overlap_printed = true;
                    }
                    printf("      %8u %10u %10u %+10.0f\n", n, a, b, (double)b - (double)a);
                }
                if (overlap_printed && cascade_over.ok) {
                    const Span loop_over_128 = [&]() {
                        // The loop form's rate over the SAME span the cascade
                        // covers, so the two are compared on equal terms rather
                        // than one of them over a span four times as wide.
                        Span s{0.0, 0, 0, 0, 0, false};
                        uint32_t c_lo = 0, c_hi = 0;
                        if (cycles_at('q', ts.variant, "q_loop_adddmareg", t, cascade_over.lo,
                                      &c_lo) &&
                            cycles_at('q', ts.variant, "q_loop_adddmareg", t, cascade_over.hi,
                                      &c_hi)) {
                            s.rate = ((double)c_hi - (double)c_lo) /
                                     (double)(cascade_over.hi - cascade_over.lo);
                            s.lo = cascade_over.lo;
                            s.hi = cascade_over.hi;
                            s.ok = true;
                        }
                        return s;
                    }();
                    if (loop_over_128.ok) {
                        printf(
                            "      -> over n=%u..%u: cascade %.3f, loop %.3f cycles per "
                            "instruction\n         (difference %+.3f -- the loop's own back "
                            "edge is in this, and\n         `q_loop_addi` above says how much "
                            "of it that is)\n",
                            cascade_over.lo, cascade_over.hi, cascade_over.rate,
                            loop_over_128.rate, loop_over_128.rate - cascade_over.rate);
                    }
                }

                // -- the extended read, from the loop form only --------------
                const Span plain = span_rate(ts.variant, "q_loop_adddmareg", t,
                                             RVBENCH_Q_LOOP_MIN_N);
                const Span synced = span_rate(ts.variant, "q_loop_adddmareg_sync", t,
                                              RVBENCH_Q_LOOP_MIN_N);
                const Span baseline = span_rate(ts.variant, "q_loop_addi", t,
                                                RVBENCH_Q_LOOP_MIN_N);
                if (!plain.ok || !synced.ok) {
                    printf("\n");
                    continue;
                }
                printf("    out to n=%u:  issuing core %.3f  |  drained %.3f  |  ",
                       plain.hi, plain.rate, synced.rate);
                if (baseline.ok) {
                    printf("issue-limited baseline %.3f (`q_loop_addi`)\n", baseline.rate);
                } else {
                    printf("issue-limited baseline not measured\n");
                }

                // The backlog: how much ThCon work the burst did not wait for.
                // sync - plain at each n, less its value at the smallest n,
                // which removes `tensix_sync()`'s own cost. The loop's back
                // edge, the clock reads and the instruction fetch are identical
                // in the two probes and cancel here exactly.
                uint32_t base_lo_plain = 0, base_lo_sync = 0;
                cycles_at('q', ts.variant, "q_loop_adddmareg", t, plain.lo, &base_lo_plain);
                cycles_at('q', ts.variant, "q_loop_adddmareg_sync", t, plain.lo, &base_lo_sync);
                const double base = (double)base_lo_sync - (double)base_lo_plain;
                printf(
                    "    backlog in flight when the last .ttinsn returned (cycles of ThCon\n"
                    "    work, less %.0f for tensix_sync()'s own cost at n=%u):\n      ",
                    base, plain.lo);
                double prev = 0.0, last_step = 0.0, prev_step = 0.0, last = 0.0;
                bool have_prev = false;
                for (uint32_t n = plain.lo; n <= plain.hi; n *= 2) {
                    uint32_t a = 0, b = 0;
                    if (!cycles_at('q', ts.variant, "q_loop_adddmareg", t, n, &a) ||
                        !cycles_at('q', ts.variant, "q_loop_adddmareg_sync", t, n, &b)) {
                        continue;
                    }
                    const double flight = (double)b - (double)a - base;
                    printf("n=%u %+.0f  ", n, flight);
                    if (have_prev) {
                        prev_step = last_step;
                        last_step = flight - prev;
                    }
                    prev = flight;
                    last = flight;
                    have_prev = true;
                }
                printf("\n");

                // The knee, from adjacent marginals of the loop form. Compared
                // against the DRAINED rate, which is the saturated rate by
                // measurement rather than by assuming the documented occupancy.
                uint32_t knee_lo = 0, knee_hi = 0;
                double knee_marg = 0.0;
                {
                    uint32_t prev_n = 0, prev_c = 0;
                    printf("    marginal cost between adjacent burst lengths:  ");
                    for (uint32_t n = plain.lo; n <= plain.hi; n *= 2) {
                        uint32_t c = 0;
                        if (!cycles_at('q', ts.variant, "q_loop_adddmareg", t, n, &c)) {
                            continue;
                        }
                        if (prev_n != 0) {
                            const double m =
                                ((double)c - (double)prev_c) / (double)(n - prev_n);
                            printf("%u->%u %.2f  ", prev_n, n, m);
                            if (knee_lo == 0 && m >= 0.9 * synced.rate) {
                                knee_lo = prev_n;
                                knee_hi = n;
                                knee_marg = m;
                            }
                        }
                        prev_n = n;
                        prev_c = c;
                    }
                    printf("\n");
                }
                if (knee_lo != 0) {
                    printf(
                        "    -> KNEE between n=%u and n=%u: the marginal reaches %.2f against a\n"
                        "       drained %.3f, so the core is back-pressured from there on.\n",
                        knee_lo, knee_hi, knee_marg, synced.rate);
                } else {
                    printf(
                        "    -> NO KNEE up to n=%u: the core never stops running ahead within\n"
                        "       this sweep.\n",
                        plain.hi);
                }
                // A depth in ENTRIES needs TWO things, and the second was added
                // on 2026-08-05 after a run in which three thread slots
                // reported a NEGATIVE backlog and two more divided a single-
                // digit one by a service rate to announce "~1 instruction in
                // flight". Work in flight cannot be negative; that is not a
                // noisy estimate, it is arithmetic run where there is no
                // signal.
                //
                // (1) THE BACKLOG MUST CLEAR THE NOISE FLOOR. It is
                //     (sync[n] - plain[n]) - (sync[lo] - plain[lo]): a
                //     difference of two differences, i.e. four raw single-shot
                //     points. `q_ctrl` measures what ONE such point can be
                //     wrong by -- 13 to 19 cycles on this part -- so two of its
                //     spreads is the least a backlog has to clear before the
                //     number is signal rather than the control's own scatter.
                //     The threshold is taken from the measured spread rather
                //     than from a constant, because it is the run's own
                //     statement about itself. This is what disqualifies the
                //     multi-thread slots: their drained rate is three times
                //     higher, so the SAME queue holds a backlog three times
                //     smaller in cycles, and it lands underneath the floor.
                //
                // (2) IT MUST HAVE STOPPED GROWING. While it is still growing
                //     the number it is growing towards has not been reached,
                //     and dividing the last value by the service rate reports a
                //     burst length rather than a capacity. The test is the last
                //     doubling's own contribution: an UNBOUNDED queue absorbs
                //     the whole of every doubling, so the backlog doubles too
                //     and the last step is ~50 % of the level; a queue that has
                //     saturated adds nothing more. A quarter of the level
                //     separates the two with room to spare and refuses to
                //     announce a depth from a series that is merely bending.
                if (!have_prev) {
                    printf("\n");
                    continue;
                }
                const double floor_cycles =
                    have_ctrl ? 2.0 * (double)(ctrl_hi - ctrl_lo) : 0.0;
                if (last <= 0.0) {
                    printf(
                        "    -> the backlog is NEGATIVE (%.0f cycles at n=%u). Work in flight\n"
                        "       cannot be negative, so this slot has NO SIGNAL: the true\n"
                        "       backlog is smaller than the %.0f-cycle spread `q_ctrl` measures\n"
                        "       for one raw point and the subtraction has gone through zero.\n"
                        "       NO DEPTH IN ENTRIES IS REPORTED, and none should be inferred.\n",
                        last, plain.hi, have_ctrl ? (double)(ctrl_hi - ctrl_lo) : 0.0);
                } else if (last <= floor_cycles) {
                    printf(
                        "    -> the backlog is %.0f cycles at n=%u, INSIDE THE NOISE FLOOR "
                        "(%.0f =\n       two `q_ctrl` spreads of %.0f, and the backlog is a "
                        "difference of two\n       differences of raw single-shot points). "
                        "Dividing it by the drained\n       rate would print a depth derived "
                        "from less signal than the control's\n       own scatter, so NO DEPTH "
                        "IN ENTRIES IS REPORTED for this slot.\n",
                        last, plain.hi, floor_cycles,
                        have_ctrl ? (double)(ctrl_hi - ctrl_lo) : 0.0);
                } else if (last_step > 0.25 * last) {
                    printf(
                        "    -> the backlog is STILL GROWING at the largest burst (last two\n"
                        "       doublings added %.0f then %.0f cycles), so it is not an "
                        "asymptote and\n       NO DEPTH IN ENTRIES IS RESOLVABLE from this "
                        "run. Either the queue is\n       deeper than n=%u, or nothing "
                        "back-pressures this core at all -- which is\n       what tt-sim is "
                        "by construction (`push_mop_instruction` is a list append).\n",
                        prev_step, last_step, plain.hi);
                } else {
                    printf(
                        "    -> the backlog FLATTENED at ~%.0f cycles (last two doublings added\n"
                        "       %.0f then %.0f), clearing a %.0f-cycle noise floor, and at the\n"
                        "       drained rate of %.3f cycles per instruction that is ~%.0f\n"
                        "       INSTRUCTIONS in flight -- a measurement of the Tensix\n"
                        "       instruction queue's depth in entries.\n",
                        last, prev_step, last_step, floor_cycles, synced.rate,
                        synced.rate > 0 ? last / synced.rate : 0.0);
                }
                printf("\n");
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
            "  published. Both are results.\n"
            "\n"
            "  THIS PHASE'S BODIES ARE FROZEN. Two tracked silicon datasets carry its\n"
            "  rows, and its build is within a few hundred bytes of tt-metal's kernel\n"
            "  config buffer besides. The footprints BETWEEN 1024 and 2048, which narrow\n"
            "  the boundary this phase located, are phase G.\n\n");
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
    // Phase S read-out: is the Tensix instruction queue shared between the
    // TRISCs or private to each?
    //
    // THE ANSWER IS A COMPARISON BETWEEN THREAD COUNTS and never a level, so
    // this section prints one depth per (variant, thread) slot and then only
    // ratios. A single variant produces no verdict at all and says so.
    // -----------------------------------------------------------------------
    if (ran[5]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase S: is the Tensix instruction queue shared, or one per thread?\n");
        printf("%s\n", std::string(78, '-').c_str());
        printf(
            "  EXPLORATORY, and it is the question phase Q left open: that phase resolved\n"
            "  a depth of ~14-16 entries at ONE issuing thread and could not resolve one\n"
            "  at any other, and a shared queue of depth D looks exactly like a per-thread\n"
            "  queue of depth D when only one thread is looking at it.\n"
            "\n"
            "  WHAT SEPARATES THE TWO, and what does not:\n"
            "    * a second thread that only SPINS does not. It pushes nothing, so it\n"
            "      holds no entry either way. It is run below anyway, as `s_solo_*`, and\n"
            "      it is the control for `another core is awake` as against `another core\n"
            "      is issuing` -- its depth must not move under EITHER hypothesis.\n"
            "    * a second thread issuing at a low, known rate does not either: queue\n"
            "      occupancy is arrival rate times residence time, so a thread served\n"
            "      faster than it arrives holds ~0 entries however long it runs.\n"
            "    * only a SATURATED second thread holds entries, and the backend\n"
            "      bandwidth it necessarily steals is measured here (`s_co_sync`) and\n"
            "      divided back out. That is `s_co_*`, and it is the discriminator.\n"
            "\n"
            "  THE REFERENCE BURST IS n=%d, not phase Q's 16, and that is the other half\n"
            "  of why this can answer what phase Q could not. The backlog is\n"
            "  (sync[n] - plain[n]) - (sync[ref] - plain[ref]), and the subtrahend is only\n"
            "  `tensix_sync()`'s own cost if the queue is EMPTY at the reference burst. At\n"
            "  n=16 it is not -- and at two or three issuing threads a shared queue's\n"
            "  per-thread share may be smaller than the ~10 entries an n=16 burst leaves\n"
            "  outstanding, which makes phase Q's multi-thread backlogs structurally zero\n"
            "  rather than merely small.\n"
            "\n"
            "  So the depth in ENTRIES reported below is\n"
            "      D = backlog / S + %d * (1 - p/S)\n"
            "  with S the drained rate from the `_sync` probe and p the issue-limited\n"
            "  rate from `s_loop_addi`. Both are measured in the same slot. The second\n"
            "  term is the reference burst's own occupancy, which phase Q's read-out\n"
            "  drops; it is ~2 entries here and ~10 there, so a phase-S depth at one\n"
            "  thread is expected to come out ABOVE phase Q's ~14-16 by roughly that\n"
            "  difference. If it does, phase Q's figure is a lower bound and this says by\n"
            "  how much. If it does not, this correction is wrong and should be said so.\n\n",
            RVBENCH_S_MIN_N,
            RVBENCH_S_MIN_N);

        const uint32_t s_hi_n = RVBENCH_S_MIN_N << (RVBENCH_S_POINTS - 1);
        auto s_at = [&](const std::string& variant, const char* probe, int thread, uint32_t n,
                        uint32_t* out_cycles) {
            for (const auto& r : rows) {
                if (r.phase == 's' && r.variant == variant && r.probe == probe &&
                    r.thread == thread && r.n == n) {
                    *out_cycles = r.cycles;
                    return true;
                }
            }
            return false;
        };
        // Cycles per instruction over the widest span the phase runs, as a
        // difference of two raw points: every phase-S point carries the same
        // fixed cost (barrier, untimed drain, two clock reads) and subtracting
        // one from another cancels it exactly.
        auto s_rate = [&](const std::string& variant, const char* probe, int thread) {
            uint32_t lo = 0, hi = 0;
            if (!s_at(variant, probe, thread, RVBENCH_S_MIN_N, &lo) ||
                !s_at(variant, probe, thread, s_hi_n, &hi)) {
                return 0.0;
            }
            return ((double)hi - (double)lo) / (double)(s_hi_n - RVBENCH_S_MIN_N);
        };
        // This slot's own single-shot repeatability: the largest disagreement
        // between two byte-identical executions of the same burst.
        auto s_noise_of = [&](const std::string& variant, int thread) {
            double worst = 0.0;
            for (uint32_t k = 0; k < RVBENCH_S_POINTS; k++) {
                uint32_t a = 0, b = 0;
                const uint32_t n = RVBENCH_S_MIN_N << k;
                if (!s_at(variant, "s_co_plain", thread, n, &a) ||
                    !s_at(variant, "s_co_repeat", thread, n, &b)) {
                    continue;
                }
                const double d = a > b ? (double)(a - b) : (double)(b - a);
                if (d > worst) {
                    worst = d;
                }
            }
            return worst;
        };

        struct Depth {
            bool ok;
            double entries;
            double backlog;
            double last_step;
            double rate;
            double issue;
            const char* refusal;
        };
        auto depth_of = [&](const std::string& variant, const char* plain, const char* sync,
                            int thread, double noise) {
            Depth d{false, 0.0, 0.0, 0.0, 0.0, 0.0, "not measured"};
            const double S = s_rate(variant, sync, thread);
            const double p = s_rate(variant, "s_loop_addi", thread);
            if (S <= 0.0) {
                return d;
            }
            d.rate = S;
            d.issue = p;
            uint32_t ref_plain = 0, ref_sync = 0;
            if (!s_at(variant, plain, thread, RVBENCH_S_MIN_N, &ref_plain) ||
                !s_at(variant, sync, thread, RVBENCH_S_MIN_N, &ref_sync)) {
                return d;
            }
            const double base = (double)ref_sync - (double)ref_plain;
            double values[RVBENCH_S_POINTS];
            int count = 0;
            for (uint32_t k = 0; k < RVBENCH_S_POINTS; k++) {
                uint32_t a = 0, b = 0;
                const uint32_t n = RVBENCH_S_MIN_N << k;
                if (!s_at(variant, plain, thread, n, &a) || !s_at(variant, sync, thread, n, &b)) {
                    continue;
                }
                values[count++] = (double)b - (double)a - base;
            }
            if (count < 3) {
                return d;
            }
            d.backlog = values[count - 1];
            d.last_step = values[count - 1] - values[count - 2];
            // The same two guards phase Q's loop read-out enforces, for the
            // same reasons: work in flight cannot be negative, a backlog under
            // twice one point's own scatter is the subtraction and not the
            // queue, and a backlog still growing at the longest burst has not
            // reached the number it is growing towards.
            if (d.backlog <= 0.0) {
                d.refusal = "not positive -- work in flight cannot be zero-or-negative";
                return d;
            }
            if (d.backlog <= 2.0 * noise) {
                d.refusal = "inside twice this slot's measured repeatability";
                return d;
            }
            if (d.last_step > 0.25 * d.backlog) {
                d.refusal = "still growing at the longest burst";
                return d;
            }
            d.ok = true;
            d.entries = d.backlog / S + (double)RVBENCH_S_MIN_N * (1.0 - (p > 0 ? p / S : 0.0));
            return d;
        };

        struct VariantResult {
            const char* variant;
            int issuers;
            bool have_co;
            double co_entries;
            bool have_solo;
            double solo_entries;
        };
        std::vector<VariantResult> results;

        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            VariantResult vr{ts.variant, popcount(ts.mask), false, 0.0, false, 0.0};
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                printf("  [%s thread %d]\n", ts.variant, t);
                printf("    %-16s", "raw cycles");
                for (uint32_t k = 0; k < RVBENCH_S_POINTS; k++) {
                    printf("%8u", RVBENCH_S_MIN_N << k);
                }
                printf("%10s\n", "rate");
                for (const char* nm : {"s_loop_addi", "s_co_plain", "s_co_repeat", "s_co_sync",
                                       "s_solo_plain", "s_solo_sync"}) {
                    uint32_t probe_cycles = 0;
                    if (!s_at(ts.variant, nm, t, RVBENCH_S_MIN_N, &probe_cycles)) {
                        continue;
                    }
                    printf("    %-16s", nm);
                    for (uint32_t k = 0; k < RVBENCH_S_POINTS; k++) {
                        uint32_t c = 0;
                        if (s_at(ts.variant, nm, t, RVBENCH_S_MIN_N << k, &c)) {
                            printf("%8u", c);
                        } else {
                            printf("%8s", "-");
                        }
                    }
                    printf("%10.3f\n", s_rate(ts.variant, nm, t));
                }
                const double noise = s_noise_of(ts.variant, t);
                printf(
                    "    repeatability: |s_co_plain - s_co_repeat| <= %.0f cycles; a backlog "
                    "must\n      clear twice that before it is divided by anything.\n",
                    noise);

                const Depth co = depth_of(ts.variant, "s_co_plain", "s_co_sync", t, noise);
                const Depth solo = depth_of(ts.variant, "s_solo_plain", "s_solo_sync", t, noise);
                for (int which = 0; which < 2; which++) {
                    const Depth& d = which == 0 ? co : solo;
                    const char* label = which == 0 ? "co-issuing" : "solo (others spin)";
                    if (d.rate <= 0.0) {
                        continue;
                    }
                    if (d.ok) {
                        printf(
                            "    %-19s backlog %.0f cycles (last doubling %+.0f), drained "
                            "%.3f,\n      issue-limited %.3f  ->  DEPTH ~%.0f ENTRIES\n",
                            label, d.backlog, d.last_step, d.rate, d.issue, d.entries);
                    } else {
                        printf(
                            "    %-19s backlog %.0f cycles: %s.\n      NO DEPTH IN ENTRIES IS "
                            "REPORTED for this slot.\n",
                            label, d.backlog, d.refusal);
                    }
                }
                if (t == RVBENCH_S_ISSUER) {
                    vr.have_co = co.ok;
                    vr.co_entries = co.entries;
                    vr.have_solo = solo.ok;
                    vr.solo_entries = solo.entries;
                }
                printf("\n");
            }
            results.push_back(vr);
        }

        // -- the verdict, which is a ratio and nothing else -------------------
        printf("  VERDICT (thread %d, the one every variant activates, so the only thing\n",
               RVBENCH_S_ISSUER);
        printf("  that changes between these rows is what the OTHER cores are doing):\n");
        const VariantResult* solo_ref = nullptr;
        for (const auto& r : results) {
            if (r.issuers == 1 && r.have_co) {
                solo_ref = &r;
            }
        }
        if (solo_ref == nullptr) {
            printf(
                "    the single-thread slot resolved no depth, so there is no baseline to\n"
                "    compare against and NO VERDICT. Against tt-sim this is the forced\n"
                "    outcome -- `push_mop_instruction` is an unbounded list append, so the\n"
                "    backlog is still growing at every burst length in every slot and no\n"
                "    depth is resolvable anywhere. That is a fact about the simulator.\n");
        } else {
            int compared = 0;
            for (const auto& r : results) {
                if (r.issuers <= 1) {
                    continue;
                }
                if (!r.have_co) {
                    printf(
                        "    %s: no depth resolved with %d threads issuing; nothing to "
                        "compare.\n",
                        r.variant, r.issuers);
                    continue;
                }
                compared++;
                const double ratio = r.co_entries / solo_ref->co_entries;
                const double shared_predicts = 1.0 / (double)r.issuers;
                const char* verdict;
                if (ratio >= 0.75) {
                    verdict = "PER-THREAD (each core has its own queue)";
                } else if (ratio <= 1.25 * shared_predicts) {
                    verdict = "SHARED (one queue, split between the issuers)";
                } else {
                    verdict = "AMBIGUOUS -- between the two predictions";
                }
                printf(
                    "    %s: %.0f entries against t1's %.0f = %.2fx. Per-thread predicts "
                    "1.00x,\n      shared predicts %.2fx.  ->  %s\n",
                    r.variant, r.co_entries, solo_ref->co_entries, ratio, shared_predicts,
                    verdict);
                if (r.have_solo) {
                    const double sratio = r.solo_entries / solo_ref->co_entries;
                    printf(
                        "      control: with the other %d core(s) only SPINNING, %.0f entries "
                        "(%.2fx).\n      Both hypotheses predict 1.00x here; a departure is "
                        "something other than\n      queue sharing -- instruction fetch out of "
                        "the same L1, most likely.\n",
                        r.issuers - 1, r.solo_entries, sratio);
                }
            }
            if (compared == 0) {
                printf(
                    "    only one thread count was run (--variants), and this phase's answer "
                    "is a\n    comparison between thread counts. Run at least "
                    "`--variants t1,t2`.\n");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase G read-out: where between a 4 KiB and an 8 KiB loop body does the
    // step phase F found actually fall?
    // -----------------------------------------------------------------------
    if (ran[6]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("phase G: narrowing the step phase F put between 4 KiB and 8 KiB\n");
        printf("%s\n", std::string(78, '-').c_str());
        printf(
            "  EXPLORATORY, and a NARROWING rather than a new question. Phase F reads a\n"
            "  flat 0.998 cycles/instruction from a 64-instruction loop body through 1024\n"
            "  and 1.251 at 2048, which locates a boundary in that octave and nothing\n"
            "  finer. This set adds ONE intermediate, compiled against a 1024-instruction\n"
            "  body in the SAME kernel so the comparison does not cross a build.\n"
            "\n"
            "  WHAT THIS CAN AND CANNOT SAY. It locates a boundary in LOOP-BODY SIZE. It\n"
            "  is not a cache capacity: no document in the ISA documentation or either\n"
            "  vendor tree gives an instruction cache size or a miss cost, and a step in\n"
            "  cost against footprint is equally consistent with a prefetch window, a\n"
            "  TLB-like structure or an L1 access pattern. Narrowing the boundary does\n"
            "  not license a different noun for it.\n"
            "\n"
            "  --gset %d of 0..%d is compiled in; run the other sets to fill the row.\n\n",
            gset,
            RVBENCH_G_SETS - 1);
        printf("  %-9s %-4s %10s  %-8s %6s %8s %11s\n", "variant", "thr", "g_1024",
               "probe", "bytes", "cyc/instr", "step");
        for (const auto& ts : thread_sets) {
            if (!selected(variant_filter, ts.variant)) {
                continue;
            }
            for (int t = 0; t < RVBENCH_MAX_THREADS; t++) {
                if ((ts.mask & (1u << t)) == 0) {
                    continue;
                }
                const double base = control_slope('g', ts.variant, t);
                const Fit anchor = fit_for('g', ts.variant, "g_1024", t);
                if (anchor.slope == 0) {
                    continue;
                }
                const double anchor_per = (anchor.slope - base) / 1024.0;
                for (int p = RVBENCH_P_G_1280; p <= RVBENCH_P_G_1792; p++) {
                    const Fit fit = fit_for('g', ts.variant, PROBES[p].name, t);
                    if (fit.slope == 0) {
                        continue;
                    }
                    const double per = (fit.slope - base) / (double)PROBES[p].unroll;
                    printf("  %-9s %-4d %10.3f  %-8s %4u B %8.3f %+11.3f\n", ts.variant, t,
                           anchor_per, PROBES[p].name, PROBES[p].unroll * 4, per,
                           per - anchor_per);
                }
            }
        }
        printf(
            "\n  A STEP at the intermediate puts the boundary at or below that many bytes\n"
            "  of loop body; a FLAT pair puts it above, with phase F's 8192 bounding it\n"
            "  from the other side. Read the three --gset runs together: they bracket it\n"
            "  between two footprints that were actually run, and that bracket is the\n"
            "  whole claim.\n");
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

    // Same question for the two additions, which have their own version of it:
    // a probe that reads the null and a probe that never ran look identical in
    // a summary table unless something says which. Nothing here is a verdict;
    // it is the presence check that makes the verdicts above readable.
    if (ran[5] || ran[6]) {
        printf("\n%s\n", std::string(78, '-').c_str());
        printf("Did the new probes run at all? (a null is not an absence)\n");
        printf("%s\n", std::string(78, '-').c_str());
        auto ran_probe = [&](char phase, const char* name) {
            for (const auto& r : rows) {
                if (r.phase == phase && r.probe == name && r.cycles != 0) {
                    return true;
                }
            }
            return false;
        };
        if (ran[6]) {
            printf("  phase G probes:        ");
            for (const char* nm : {"g_1024", "g_1280", "g_1536", "g_1792"}) {
                printf("%s %s  ", nm, ran_probe('g', nm) ? "ran" : "not in this --gset");
            }
            printf(
                "\n    Exactly one intermediate is compiled per --gset, so two of the three\n"
                "    read `not in this --gset` in every healthy run and that is not an\n"
                "    absence. Against tt-sim all of them must read the SAME as each other:\n"
                "    it models no instruction cache at all, so a flat row is forced there\n"
                "    and says nothing about any hardware. `ran` with a flat row is that\n"
                "    null; a probe that never executed would print nothing at all above.\n");
        }
        if (ran[5]) {
            printf("  phase S probes:        ");
            for (const char* nm : {"s_loop_addi", "s_co_plain", "s_co_repeat", "s_co_sync",
                                   "s_solo_plain", "s_solo_sync"}) {
                printf("%s %s  ", nm, ran_probe('s', nm) ? "ran" : "ABSENT");
            }
            printf(
                "\n    The structural check for this phase is that `s_co_sync` exceeds\n"
                "    `s_co_plain` at every burst length -- a drain cannot be free. If the\n"
                "    two are equal the phase measured nothing whatever the verdict said.\n"
                "    Against tt-sim the backlog grows without bound (its Tensix queue is a\n"
                "    list append), so every slot refuses a depth and there is no verdict:\n"
                "    forced, and a fact about the simulator rather than about any card.\n");
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
