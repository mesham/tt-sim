// retirebench -- the RV-bound program for rung 4's mechanism-attribution leg.
//
// A normal tt-metal program with no compute kernel and no NoC traffic: one
// core, one data-movement kernel on BRISC, two small L1 buffers, and a
// self-check on the result buffer the kernel wrote. It exists to be run twice --
// once on a Blackhole card and once against tt-sim -- so that the two
// decompositions of one RISC-V core's span can be compared by
// `tt_sim.perf.retire_attribution`.
//
// Both sides emit the IDENTICAL artefact, a `retirebench-*.json` written by
// this host program, so one parser reads both and there is no translation step
// in which a units mistake could hide. That is the same discipline
// `perfbench/nocevbench` follows, for the same reason.
//
// BLACKHOLE ONLY, AND IT REFUSES RATHER THAN DEGRADES. The instrument is the
// `mcycle` and `minstret` CSRs, which exist on Blackhole baby RISC-V cores
// (BlackholeA0/TensixTile/BabyRISCV/CSRs.md) and nowhere in the WormholeB0 doc
// tree -- the string "csr" does not appear in it at all. Without `minstret`
// there is no retired-instruction count, the structural zone labels become
// uncheckable, and the leg collapses to an elapsed-only envelope check, which
// is precisely what rung 4 exists to distrust. So this program refuses a
// non-Blackhole part before it builds its kernel or launches anything, and the
// analysis refuses a non-Blackhole artefact before it computes anything.
//
// Usage:  retirebench [--scale N] [--out DIR] [--label TEXT] [--describe]
//
// `--describe` opens the device, prints the part and the grid, and exits
// without launching anything. It is what the card pre-flight uses to establish
// that the part is a Blackhole, which is not knowable from the host.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "kernels/dataflow/retirebench_layout.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// The zone names, in layout order. The kernel knows the indices and this table
// knows the names; the `static_assert` below is what keeps them the same list.
struct ZoneSpec {
    const char* name;
    uint32_t base_reps;
    // What the zone is built to be dominated by. Reproduced verbatim into the
    // artefact so a reader of the JSON alone cannot mistake the label for a
    // hardware attribution -- it is a claim about the program's construction.
    const char* mechanism;
};

const ZoneSpec ZONES[] = {
    {"marker_null", RETIREBENCH_REPS_MARKER_NULL, "instrument calibration: the two-marker cost alone"},
    {"alu_dep", RETIREBENCH_REPS_ALU_DEP, "dependent integer chain (forwarding path)"},
    {"alu_ind", RETIREBENCH_REPS_ALU_IND, "independent integer ops (issue width)"},
    {"mul_dep", RETIREBENCH_REPS_MUL_DEP, "multiply result latency"},
    {"mul_ind", RETIREBENCH_REPS_MUL_IND, "multiply unit occupancy"},
    {"div_small", RETIREBENCH_REPS_DIV_SMALL, "divide, 12-bit dividend"},
    {"div_large", RETIREBENCH_REPS_DIV_LARGE, "divide, 29-bit dividend"},
    {"load_dep", RETIREBENCH_REPS_LOAD_DEP, "L1 load-use interlock (pointer chase)"},
    {"load_ind", RETIREBENCH_REPS_LOAD_IND, "sustained L1 load throughput"},
    {"store_spread", RETIREBENCH_REPS_STORE_SPREAD, "L1 store throughput, non-coalescable"},
    {"branch_nt", RETIREBENCH_REPS_BRANCH_NT, "not-taken conditional branch"},
    {"branch_t", RETIREBENCH_REPS_BRANCH_T, "taken conditional branch"},
};

static_assert(
    sizeof(ZONES) / sizeof(ZONES[0]) == RETIREBENCH_NUM_ZONES,
    "the zone name table and retirebench_layout.h's zone list have diverged");

const char* arch_name(IDevice* device) {
    switch (device->arch()) {
        case tt::ARCH::WORMHOLE_B0: return "wormhole";
        case tt::ARCH::BLACKHOLE: return "blackhole";
        default: return "unknown";
    }
}

// Unsigned difference of two low counter halves. A single wrap of the 32-bit
// half comes out right; a delta of 2^32 or more does not, and cannot happen
// over a ~25 kcycle kernel. A genuinely non-monotonic pair comes back as a
// value near 2^32, which drives the analysis's `unattributed` bucket negative
// and is refused there rather than being clamped here.
uint32_t delta(uint32_t before, uint32_t after) { return after - before; }

void write_json(
    const std::string& path,
    const char* arch,
    const std::string& label,
    const CoreCoord& logical,
    const CoreCoord& noc,
    uint32_t scale,
    const std::vector<uint32_t>& r) {
    FILE* fh = fopen(path.c_str(), "w");
    if (fh == nullptr) {
        fprintf(stderr, "retirebench: cannot write %s\n", path.c_str());
        return;
    }
    fprintf(fh, "{\n");
    fprintf(fh, "  \"retirebench\": %u,\n", RETIREBENCH_VERSION);
    fprintf(fh, "  \"arch\": \"%s\",\n", arch);
    fprintf(fh, "  \"label\": \"%s\",\n", label.c_str());
    fprintf(fh, "  \"risc\": \"BRISC\",\n");
    fprintf(fh, "  \"core\": [%u, %u],\n", (unsigned)noc.x, (unsigned)noc.y);
    fprintf(fh, "  \"logical_core\": [%u, %u],\n", (unsigned)logical.x, (unsigned)logical.y);
    fprintf(fh, "  \"scale\": %u,\n", scale);
    fprintf(
        fh,
        "  \"window\": {\"cycles\": %u, \"retired\": %u},\n",
        delta(r[RETIREBENCH_HDR_WINDOW_C0], r[RETIREBENCH_HDR_WINDOW_C1]),
        delta(r[RETIREBENCH_HDR_WINDOW_I0], r[RETIREBENCH_HDR_WINDOW_I1]));
    fprintf(fh, "  \"zones\": [\n");
    for (uint32_t z = 0; z < RETIREBENCH_NUM_ZONES; z++) {
        const uint32_t base = RETIREBENCH_HDR_WORDS + z * RETIREBENCH_ZONE_WORDS;
        fprintf(
            fh,
            "    {\"index\": %u, \"name\": \"%s\", \"mechanism\": \"%s\", \"reps\": %u, "
            "\"cycles\": %u, \"retired\": %u}%s\n",
            z,
            ZONES[z].name,
            ZONES[z].mechanism,
            ZONES[z].base_reps * scale,
            delta(r[base + RETIREBENCH_Z_C0], r[base + RETIREBENCH_Z_C1]),
            delta(r[base + RETIREBENCH_Z_I0], r[base + RETIREBENCH_Z_I1]),
            z + 1 == RETIREBENCH_NUM_ZONES ? "" : ",");
    }
    fprintf(fh, "  ]\n");
    fprintf(fh, "}\n");
    fclose(fh);
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t scale = 1;
    std::string out_dir = ".";
    std::string label = "run";
    bool describe = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--scale" && i + 1 < argc) {
            scale = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (a == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--label" && i + 1 < argc) {
            label = argv[++i];
        } else if (a == "--describe") {
            describe = true;
        } else {
            fprintf(stderr, "retirebench: unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (scale == 0) {
        fprintf(stderr, "retirebench: --scale must be >= 1\n");
        return 2;
    }

    IDevice* device = CreateDevice(0);
    const char* arch = arch_name(device);
    constexpr CoreCoord core = {0, 0};
    const CoreCoord noc_core = device->worker_core_from_logical_core(core);
    const CoreCoord grid = device->compute_with_storage_grid_size();

    printf("retirebench-config arch=%s scale=%u core_logical=%u,%u core_noc=%u,%u grid=%ux%u zones=%u\n",
           arch,
           scale,
           (unsigned)core.x,
           (unsigned)core.y,
           (unsigned)noc_core.x,
           (unsigned)noc_core.y,
           (unsigned)grid.x,
           (unsigned)grid.y,
           (unsigned)RETIREBENCH_NUM_ZONES);
    fflush(stdout);

    // THE REFUSAL. Before this program builds its kernel, launches anything or
    // allocates a buffer -- and before any card time is spent. (tt-metal has
    // already JIT-built its own firmware inside CreateDevice by the time this
    // runs; that is unavoidable, because the part is not knowable until the
    // device is open. Nothing of THIS benchmark has been built or run.)
    // A Wormhole baby core has no CSRs at all in its ISA documentation, so
    // `csrr %0, minstret` is not an instruction it can be asked to execute:
    // against tt-sim it raises NoCSRsError, and on silicon it would at best
    // return something this program has no licence to interpret. Running anyway
    // and reporting elapsed cycles alone would produce a well-formed artefact
    // that supports an envelope claim -- which is the claim rung 4 exists to
    // distrust -- so the program declines instead of degrading.
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        fprintf(stderr,
                "retirebench: this part is %s, and this benchmark is Blackhole only.\n"
                "  The instrument is the mcycle (0xb00) and minstret (0xb02) CSRs, which\n"
                "  BlackholeA0/TensixTile/BabyRISCV/CSRs.md documents and the WormholeB0\n"
                "  tree does not mention at all -- the string 'csr' appears nowhere in it.\n"
                "  Without minstret there is no retired-instruction count, the zone labels\n"
                "  become uncheckable, and what is left is an elapsed-only envelope check.\n"
                "  Three of those already exist in this repo. Rung 4 exists because they\n"
                "  cannot see a compensating interior, so producing a fourth under this\n"
                "  program's name would be worse than producing nothing.\n",
                arch);
        CloseDevice(device);
        return 3;
    }

    if (describe) {
        printf("retirebench-describe arch=%s grid=%ux%u ok=yes\n", arch, (unsigned)grid.x, (unsigned)grid.y);
        CloseDevice(device);
        return 0;
    }

    Program program = CreateProgram();

    // Two L1 buffers through the tt-metal allocator: the result table the kernel
    // writes, and the scratch the memory zones work on. Interleaved L1 with one
    // page, so each is a single bank offset on this core.
    const uint32_t result_bytes = RETIREBENCH_RESULT_WORDS * sizeof(uint32_t);
    InterleavedBufferConfig result_cfg{
        .device = device, .size = result_bytes, .page_size = result_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> result_buf = CreateBuffer(result_cfg);

    InterleavedBufferConfig scratch_cfg{
        .device = device,
        .size = RETIREBENCH_SCRATCH_BYTES,
        .page_size = RETIREBENCH_SCRATCH_BYTES,
        .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> scratch_buf = CreateBuffer(scratch_cfg);

    // Poison the result buffer, so a kernel that never ran cannot be read as a
    // run in which every counter happened to stand still.
    std::vector<uint32_t> poison(RETIREBENCH_RESULT_WORDS, 0xDEADBEEFu);
    tt::tt_metal::detail::WriteToDeviceL1(device, core, result_buf->address(), poison);

    KernelHandle zones = CreateKernel(
        program,
        "kernels/dataflow/rv_zones.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});
    SetRuntimeArgs(program, zones, core, {result_buf->address(), scratch_buf->address(), scale});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> r;
    tt::tt_metal::detail::ReadFromDeviceL1(device, core, result_buf->address(), result_bytes, r);
    CloseDevice(device);

    if (r.size() < RETIREBENCH_RESULT_WORDS) {
        printf("Failure on the device, result buffer came back %zu words, expected %u\n",
               r.size(),
               (unsigned)RETIREBENCH_RESULT_WORDS);
        return 1;
    }
    if (r[RETIREBENCH_HDR_MAGIC] != RETIREBENCH_MAGIC) {
        printf("Failure on the device, result magic is 0x%08x, expected 0x%08x -- the kernel did not "
               "finish writing its table\n",
               r[RETIREBENCH_HDR_MAGIC],
               RETIREBENCH_MAGIC);
        return 1;
    }
    if (r[RETIREBENCH_HDR_NUM_ZONES] != RETIREBENCH_NUM_ZONES) {
        printf("Failure on the device, kernel reports %u zones, host expects %u\n",
               r[RETIREBENCH_HDR_NUM_ZONES],
               (unsigned)RETIREBENCH_NUM_ZONES);
        return 1;
    }

    // The one self-check that matters, and it is the instrument checking itself:
    // if the counters never moved, every delta is zero and the artefact is a
    // perfectly plausible partition of a zero-cycle span. On tt-sim a core with
    // no clock bound refuses `mcycle` outright, so this catches the remaining
    // case -- a core that ran but whose counters did not.
    const uint32_t window_cycles = delta(r[RETIREBENCH_HDR_WINDOW_C0], r[RETIREBENCH_HDR_WINDOW_C1]);
    const uint32_t window_retired = delta(r[RETIREBENCH_HDR_WINDOW_I0], r[RETIREBENCH_HDR_WINDOW_I1]);
    if (window_cycles == 0 || window_retired == 0) {
        printf("Failure on the device, the outer window advanced %u cycles and retired %u "
               "instructions -- the counters did not run\n",
               window_cycles,
               window_retired);
        return 1;
    }

    const std::string path = out_dir + "/retirebench-" + arch + "-" + label + ".json";
    write_json(path, arch, label, core, noc_core, scale, r);

    uint64_t zone_sum = 0;
    printf("%-14s %10s %10s %10s\n", "zone", "cycles", "retired", "cyc/instr");
    for (uint32_t z = 0; z < RETIREBENCH_NUM_ZONES; z++) {
        const uint32_t base = RETIREBENCH_HDR_WORDS + z * RETIREBENCH_ZONE_WORDS;
        const uint32_t c = delta(r[base + RETIREBENCH_Z_C0], r[base + RETIREBENCH_Z_C1]);
        const uint32_t n = delta(r[base + RETIREBENCH_Z_I0], r[base + RETIREBENCH_Z_I1]);
        zone_sum += c;
        printf("%-14s %10u %10u %10.3f\n", ZONES[z].name, c, n, n ? (double)c / (double)n : 0.0);
    }
    printf("%-14s %10u %10u\n", "window", window_cycles, window_retired);
    printf("%-14s %10lld\n", "unattributed", (long long)((int64_t)window_cycles - (int64_t)zone_sum));
    printf("retirebench wrote %s\n", path.c_str());
    printf("Completed successfully on the device, %u zones over %u cycles\n",
           (unsigned)RETIREBENCH_NUM_ZONES,
           window_cycles);
    return 0;
}
