// nocevbench -- the NoC-bound program for rung 4's NoC-timing leg.
//
// A normal tt-metal program with no compute kernel at all: one core, two data
// movement kernels, a round trip, and an exact host-side check (the bytes that
// come back must be the bytes that went out, compared word for word).
//
// It exists to be run twice -- once on silicon and once against tt-sim, under
// TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 both times -- so that the NoC event
// trace each side produces can be compared by tt_sim.perf.noc_events. No
// profiler markers are placed by this program: the recorder is compiled into
// tt-metal's own dataflow API by -DPROFILE_NOC_EVENTS (jit_build/build.cpp:188)
// and the zone endpoints come from the firmware.
//
// THE ARMS
// --------
// The size arms are the pair `bytes` takes, and the pair is the point: a
// per-hop latency that is too large can be hidden by a per-byte rate that is
// too fast at one transfer size, but not at two sizes an order of magnitude
// apart. `--arm` is the second, orthogonal axis, and it exists because the
// first card session (2026-08-17) could not tell "writes are +95 cycles" from
// "NoC 0 is +95 cycles": the reader ran on NoC 1 and the writer on NoC 0, so
// direction and NoC were the same statement.
//
//   A  reader NCRISC/NOC_1 from DRAM, writer BRISC/NOC_0 to DRAM   (the CONTROL;
//      byte-for-byte the 2026-08-17 configuration, and it must reproduce it)
//   B  reader NCRISC/NOC_0 from DRAM, writer BRISC/NOC_1 to DRAM   (the NoCs
//      swapped, and nothing else -- two enum values, the same two kernels)
//   C  reader NCRISC/NOC_1 and writer BRISC/NOC_0, both against a PEER CORE's
//      L1 instead of DRAM, which removes the DRAM endpoint service term from
//      the path entirely
//
// Arm A's kernels (kernels/dataflow/reader.cpp, writer.cpp) are untouched by
// arms B and C precisely so that the control cannot drift: B changes only the
// `.noc` field of two DataMovementConfigs, and C uses its own kernel pair.
//
// Usage:  nocevbench [bytes] [chunks] [--arm A|B|C] [--peer X,Y] [--describe]
//         (defaults: 2048 8, arm A, peer logical 1,1)
//
// `--describe` opens the device, prints the configuration the arm resolves to
// -- including the peer's NoC coordinate and its distance class -- and exits
// without running anything. It is what the pre-flight uses to check that a
// peer core exists and sits where the arm claims, which is not knowable from
// the host without asking the device.

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

struct ArmSpec {
    char name;
    NOC reader_noc;
    NOC writer_noc;
    bool l1_target;
};

// The single source of the arm table on this side of the wire.
// `perfbench/nocevbench/check_arm.py` carries the same three rows and checks
// the emitted trace against them, because "the flag was passed" and "the NoC
// swap took" are different claims and only the second one matters.
const ArmSpec ARMS[] = {
    {'A', NOC::NOC_1, NOC::NOC_0, false},
    {'B', NOC::NOC_0, NOC::NOC_1, false},
    {'C', NOC::NOC_1, NOC::NOC_0, true},
};

const ArmSpec* find_arm(char name) {
    for (const ArmSpec& arm : ARMS) {
        if (arm.name == name) {
            return &arm;
        }
    }
    return nullptr;
}

const char* noc_name(NOC noc) { return noc == NOC::NOC_1 ? "NOC_1" : "NOC_0"; }

// The part, as the DEVICE reports it, for the `--describe` line only. A session
// directory whose env.txt cannot say which architecture produced it is not
// comparable to anything a year later, and the operator is the one person who
// cannot be asked afterwards. Reported on the describe path, which closes the
// device and returns before a single kernel is built, so no measured run's
// timing can depend on it.
const char* arch_name(IDevice* device) {
    switch (device->arch()) {
        case tt::ARCH::WORMHOLE_B0: return "wormhole";
        case tt::ARCH::BLACKHOLE: return "blackhole";
        default: return "unknown";
    }
}

// Which of the three geometries the peer sits in. Reported rather than assumed:
// picking the wrong peer does not fail, it silently measures a different
// geometry, and on a directional torus the geometry is the whole question.
const char* distance_class(const CoreCoord& self, const CoreCoord& peer) {
    if (self.x == peer.x && self.y == peer.y) {
        return "self";
    }
    if (self.x == peer.x) {
        return "same-column";
    }
    if (self.y == peer.y) {
        return "same-row";
    }
    return "both-axes";
}

void print_config(
    const ArmSpec& arm,
    uint32_t bytes,
    uint32_t chunks,
    const CoreCoord& self_noc,
    const CoreCoord& peer_logical,
    const CoreCoord& peer_noc,
    const CoreCoord& noc_grid,
    bool have_peer) {
    // One machine-readable line, parsed by check_arm.py and by run_card.sh.
    // Keep the key=value shape stable; both readers key on the names.
    printf("nocevbench-config arm=%c bytes=%u chunks=%u reader=NCRISC:%s writer=BRISC:%s target=%s",
           arm.name,
           bytes,
           chunks,
           noc_name(arm.reader_noc),
           noc_name(arm.writer_noc),
           arm.l1_target ? "l1" : "dram");
    printf(" self_noc=%u,%u", (unsigned)self_noc.x, (unsigned)self_noc.y);
    // The full NoC grid, so a reader can compute the NoC 1 grid mirror
    // `(GX-1-x, GY-1-y)` without knowing the part. check_arm.py uses it only to
    // NAME that case: a NoC 1 destination that is the mirror of the peer means
    // the run is in the untranslated convention, which is not the one a card is
    // in, and saying which of the two went wrong is the whole value of the check.
    printf(" noc_grid=%u,%u", (unsigned)noc_grid.x, (unsigned)noc_grid.y);
    if (have_peer) {
        printf(" peer_logical=%u,%u peer_noc=%u,%u distance=%s",
               (unsigned)peer_logical.x,
               (unsigned)peer_logical.y,
               (unsigned)peer_noc.x,
               (unsigned)peer_noc.y,
               distance_class(self_noc, peer_noc));
    } else {
        printf(" peer_logical=- peer_noc=- distance=-");
    }
    printf("\n");
    fflush(stdout);
}

bool parse_coord(const char* text, CoreCoord& out) {
    const char* comma = std::strchr(text, ',');
    if (comma == nullptr) {
        return false;
    }
    out = CoreCoord{
        static_cast<size_t>(std::strtoul(text, nullptr, 10)),
        static_cast<size_t>(std::strtoul(comma + 1, nullptr, 10))};
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    uint32_t bytes = 2048;
    uint32_t chunks = 8;
    char arm_name = 'A';
    CoreCoord peer_logical{1, 1};
    bool describe = false;
    int positional = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--arm" && i + 1 < argc) {
            arm_name = static_cast<char>(std::toupper(argv[++i][0]));
        } else if (a == "--peer" && i + 1 < argc) {
            if (!parse_coord(argv[++i], peer_logical)) {
                fprintf(stderr, "nocevbench: --peer expects X,Y (logical worker coords)\n");
                return 2;
            }
        } else if (a == "--describe") {
            describe = true;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "nocevbench: unknown option '%s'\n", argv[i]);
            return 2;
        } else if (positional == 0) {
            bytes = static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10));
            positional++;
        } else if (positional == 1) {
            chunks = static_cast<uint32_t>(std::strtoul(argv[i], nullptr, 10));
            positional++;
        } else {
            fprintf(stderr, "nocevbench: unexpected argument '%s'\n", argv[i]);
            return 2;
        }
    }

    const ArmSpec* arm = find_arm(arm_name);
    if (arm == nullptr) {
        fprintf(stderr, "nocevbench: --arm must be A, B or C (got '%c')\n", arm_name);
        return 2;
    }

    if (chunks == 0 || bytes == 0) {
        fprintf(stderr, "nocevbench: bytes and chunks must both be >= 1\n");
        return 2;
    }
    // 64 is the coarsest NoC source/destination congruence this repo models
    // (Blackhole DRAM reads, ArchProfile.noc_dram_read_congruence). Stepping
    // both the remote offset and the L1 offset by the same multiple of 64 keeps
    // every transaction congruent by construction, so the benchmark cannot
    // accidentally measure an undefined-behaviour transfer.
    if (bytes % 64 != 0) {
        fprintf(stderr, "nocevbench: bytes must be a multiple of 64 (got %u)\n", bytes);
        return 2;
    }
    // The profiler's event record stores the payload as 32-byte chunks in a
    // uint8 (event_metadata.hpp:84), so anything above 8160 bytes is recorded
    // saturated and the trace would carry a size the kernel did not use.
    if (bytes > 8160) {
        fprintf(stderr, "nocevbench: bytes must be <= 8160; the NoC event record saturates above that\n");
        return 2;
    }
    const uint64_t total = static_cast<uint64_t>(bytes) * chunks;
    if (total > 64 * 1024) {
        fprintf(stderr, "nocevbench: bytes*chunks must be <= 64 KiB of L1 scratch (got %llu)\n",
                static_cast<unsigned long long>(total));
        return 2;
    }

    const uint32_t words = static_cast<uint32_t>(total / sizeof(uint32_t));
    std::vector<uint32_t> pattern(words);
    for (uint32_t i = 0; i < words; i++) {
        // Every word distinct, so a transfer that lands at the wrong offset is
        // a mismatch rather than an invisible self-consistent shuffle.
        pattern[i] = 0xA5000000u ^ (i * 2654435761u);
    }

    IDevice* device = CreateDevice(0);
    constexpr CoreCoord core = {0, 0};
    const CoreCoord self_noc = device->worker_core_from_logical_core(core);

    const CoreCoord grid = device->compute_with_storage_grid_size();
    const bool peer_in_grid = peer_logical.x < grid.x && peer_logical.y < grid.y;
    CoreCoord peer_noc{0, 0};
    if (peer_in_grid) {
        peer_noc = device->worker_core_from_logical_core(peer_logical);
    }

    if (arm->l1_target && !peer_in_grid) {
        fprintf(stderr,
                "nocevbench: peer (%u,%u) is outside this device's compute grid (%ux%u)\n",
                (unsigned)peer_logical.x, (unsigned)peer_logical.y,
                (unsigned)grid.x, (unsigned)grid.y);
        CloseDevice(device);
        return 2;
    }
    if (arm->l1_target && peer_logical.x == core.x && peer_logical.y == core.y) {
        fprintf(stderr, "nocevbench: arm C needs a peer core distinct from the initiating core (0,0)\n");
        CloseDevice(device);
        return 2;
    }

    print_config(
        *arm, bytes, chunks, self_noc, peer_logical, peer_noc, device->grid_size(),
        arm->l1_target && peer_in_grid);

    if (describe) {
        printf("nocevbench-describe arch=%s grid=%ux%u peer_in_grid=%s\n",
               arch_name(device), (unsigned)grid.x, (unsigned)grid.y,
               peer_in_grid ? "yes" : "no");
        CloseDevice(device);
        return 0;
    }

    Program program = CreateProgram();
    const uint32_t sem_id = CreateSemaphore(program, core, 0);

    std::shared_ptr<Buffer> src;
    std::shared_ptr<Buffer> dst;
    std::shared_ptr<Buffer> l1_scratch;
    uint32_t peer_src_addr = 0;
    uint32_t peer_dst_addr = 0;
    uint32_t local_addr = 0;

    if (!arm->l1_target) {
        // One page per buffer, so the whole buffer lives in one DRAM bank and
        // get_noc_addr_from_bank_id(0, base) + i * bytes is the address the
        // kernel means. A multi-page interleaved buffer would spread pages
        // across banks on a card with more than one, and the kernel would fetch
        // someone else's data.
        InterleavedBufferConfig cfg{
            .device = device, .size = total, .page_size = total, .buffer_type = BufferType::DRAM};
        src = CreateBuffer(cfg);
        dst = CreateBuffer(cfg);

        // The circular buffer is flat L1 scratch: both kernels take its base
        // with get_write_ptr and index into it. Nothing is pushed or popped, so
        // no CB back-pressure enters the measurement.
        CircularBufferConfig scratch_cfg =
            CircularBufferConfig(total, {{CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(CBIndex::c_0, bytes);
        tt_metal::CreateCircularBuffer(program, core, scratch_cfg);

        tt::tt_metal::detail::WriteToBuffer(src, pattern);
    } else {
        // Arm C needs three L1 regions at an address that is free on EVERY
        // worker bank: the initiating core's scratch, and the peer's source and
        // destination. One interleaved L1 buffer gives exactly that -- the
        // allocator returns a single bank offset valid on every core, which is
        // the same trick perfbench/nocbench uses to reach a core the program
        // never launches. No circular buffer and no DRAM is involved, so the
        // DRAM endpoint service term is out of the measured path entirely.
        const uint64_t block = 3 * total;
        InterleavedBufferConfig l1_cfg{
            .device = device, .size = block, .page_size = block, .buffer_type = BufferType::L1};
        l1_scratch = CreateBuffer(l1_cfg);
        local_addr = l1_scratch->address();
        peer_src_addr = local_addr + static_cast<uint32_t>(total);
        peer_dst_addr = local_addr + static_cast<uint32_t>(2 * total);

        // Seed the peer's source and poison its destination. The poison is the
        // bitwise complement of the pattern, so a run in which nothing moved
        // cannot pass the check by accident -- which zero-fill could, for a
        // pattern word that happened to be zero.
        std::vector<uint32_t> poison(words);
        for (uint32_t i = 0; i < words; i++) {
            poison[i] = ~pattern[i];
        }
        tt::tt_metal::detail::WriteToDeviceL1(device, peer_logical, peer_src_addr, pattern);
        tt::tt_metal::detail::WriteToDeviceL1(device, peer_logical, peer_dst_addr, poison);
    }

    const char* reader_path =
        arm->l1_target ? "kernels/dataflow/l1_reader.cpp" : "kernels/dataflow/reader.cpp";
    const char* writer_path =
        arm->l1_target ? "kernels/dataflow/l1_writer.cpp" : "kernels/dataflow/writer.cpp";

    KernelHandle reader = CreateKernel(
        program,
        reader_path,
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = arm->reader_noc});
    KernelHandle writer = CreateKernel(
        program,
        writer_path,
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = arm->writer_noc});

    if (!arm->l1_target) {
        SetRuntimeArgs(program, reader, core, {src->address(), chunks, bytes, sem_id});
        SetRuntimeArgs(program, writer, core, {dst->address(), chunks, bytes, sem_id});
    } else {
        SetRuntimeArgs(
            program,
            reader,
            core,
            {static_cast<uint32_t>(peer_noc.x),
             static_cast<uint32_t>(peer_noc.y),
             peer_src_addr,
             local_addr,
             chunks,
             bytes,
             sem_id});
        SetRuntimeArgs(
            program,
            writer,
            core,
            {static_cast<uint32_t>(peer_noc.x),
             static_cast<uint32_t>(peer_noc.y),
             peer_dst_addr,
             local_addr,
             chunks,
             bytes,
             sem_id});
    }

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result(words, 0u);
    if (!arm->l1_target) {
        tt::tt_metal::detail::ReadFromBuffer(dst, result);
    } else {
        tt::tt_metal::detail::ReadFromDeviceL1(
            device, peer_logical, peer_dst_addr, static_cast<uint32_t>(total), result);
    }
    CloseDevice(device);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < words; i++) {
        if (result[i] != pattern[i]) {
            mismatches++;
        }
    }

    printf("nocevbench arm=%c bytes=%u chunks=%u total=%llu mismatches=%zu\n",
           arm->name, bytes, chunks, static_cast<unsigned long long>(total), mismatches);
    if (mismatches == 0) {
        printf("Completed successfully on the device, with %u words\n", words);
        return 0;
    }
    printf("Failure on the device, %zu mismatched words of %u\n", mismatches, words);
    return 1;
}
