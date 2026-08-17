// nocevbench -- the NoC-bound program for rung 4's NoC-timing leg.
//
// A normal tt-metal program with no compute kernel at all: one core, two data
// movement kernels, a DRAM round trip, and an exact host-side check (the bytes
// that come back must be the bytes that went out, compared word for word).
//
// It exists to be run twice -- once on silicon and once against tt-sim, under
// TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 both times -- so that the NoC event
// trace each side produces can be compared by tt_sim.perf.noc_events. No
// profiler markers are placed by this program: the recorder is compiled into
// tt-metal's own dataflow API by -DPROFILE_NOC_EVENTS (jit_build/build.cpp:188)
// and the zone endpoints come from the firmware.
//
// The arms are sizes on a fixed route, and the pair is the point. A per-hop
// latency that is too large can be hidden by a per-byte rate that is too fast
// at one transfer size; it cannot be hidden at two sizes an order of magnitude
// apart, because the two terms scale differently. NCRISC reads on NOC 1 and
// BRISC writes on NOC 0, so each run also measures two different routes.
//
// Usage:  nocevbench [bytes] [chunks]        (defaults: 2048 8)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    uint32_t bytes = argc > 1 ? static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 2048;
    uint32_t chunks = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 8;

    if (chunks == 0 || bytes == 0) {
        fprintf(stderr, "nocevbench: bytes and chunks must both be >= 1\n");
        return 2;
    }
    // 64 is the coarsest NoC source/destination congruence this repo models
    // (Blackhole DRAM reads, ArchProfile.noc_dram_read_congruence). Stepping
    // both the DRAM offset and the L1 offset by the same multiple of 64 keeps
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
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    // One page per buffer, so the whole buffer lives in one DRAM bank and
    // get_noc_addr_from_bank_id(0, base) + i * bytes is the address the kernel
    // means. A multi-page interleaved buffer would spread pages across banks on
    // a card with more than one, and the kernel would fetch someone else's data.
    InterleavedBufferConfig cfg{
        .device = device, .size = total, .page_size = total, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> src = CreateBuffer(cfg);
    std::shared_ptr<Buffer> dst = CreateBuffer(cfg);

    // The circular buffer is flat L1 scratch: both kernels take its base with
    // get_write_ptr and index into it. Nothing is pushed or popped, so no CB
    // back-pressure enters the measurement.
    CircularBufferConfig scratch_cfg =
        CircularBufferConfig(total, {{CBIndex::c_0, tt::DataFormat::Float16_b}}).set_page_size(CBIndex::c_0, bytes);
    tt_metal::CreateCircularBuffer(program, core, scratch_cfg);

    const uint32_t sem_id = CreateSemaphore(program, core, 0);

    tt::tt_metal::detail::WriteToBuffer(src, pattern);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(program, reader, core, {src->address(), chunks, bytes, sem_id});
    SetRuntimeArgs(program, writer, core, {dst->address(), chunks, bytes, sem_id});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result(words, 0u);
    tt::tt_metal::detail::ReadFromBuffer(dst, result);
    CloseDevice(device);

    size_t mismatches = 0;
    for (uint32_t i = 0; i < words; i++) {
        if (result[i] != pattern[i]) {
            mismatches++;
        }
    }

    printf("nocevbench bytes=%u chunks=%u total=%llu mismatches=%zu\n",
           bytes, chunks, static_cast<unsigned long long>(total), mismatches);
    if (mismatches == 0) {
        printf("Completed successfully on the device, with %u words\n", words);
        return 0;
    }
    printf("Failure on the device, %zu mismatched words of %u\n", mismatches, words);
    return 1;
}
