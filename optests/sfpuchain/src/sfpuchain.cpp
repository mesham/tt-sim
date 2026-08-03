// Minimal reproducer for the `sfpu_eltwise_chain` upstream programming example
// failing its PCC self-check on tt-sim (host side).
//
// Upstream `metal_example_sfpu_eltwise_chain` computes softplus as the SFPU
// chain exp -> add_binary(+1) -> log over a bfloat16 tile and asserts
// PCC > 0.999 against a CPU golden. On tt-sim it lands at ~0.9986 and aborts;
// on ttsim (the vendor reference) the same binary lands at ~0.9998 and passes.
// That example seeds its input from std::random_device, so it can't be diffed
// directly -- this program is the deterministic equivalent.
//
// It runs the SAME ops on a fixed bfloat16 input tile, one output tile per op,
// so the diff pinpoints *which* op disagrees rather than just "the chain":
//
//   0: copy only          -- baseline unpack/pack path, no SFPU math
//   1: exp                -- first link of the chain
//   2: log                -- last link of the chain
//   3: add_binary(x, 1)   -- middle link (SFPU binary add, not the FPU's ELWADD)
//   4: exp; add_binary(+1); log  -- the whole chain, exactly as upstream
//
// Dumps every output tile as `OPDIFF_RESULT:<hex>` (bfloat16 as 4 hex digits
// each); optests/diff.sh runs the same binary on tt-sim and on ttsim and
// compares. ttsim is the oracle -- there is no local golden.

#include <bit>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t NUM_OPS = 5;  // MUST match the compute kernel's op count

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMS;

    InterleavedBufferConfig tile_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = NUM_OPS * tile_bytes,
        .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto src_dram = CreateBuffer(tile_config);
    auto ones_dram = CreateBuffer(tile_config);
    auto dst_dram = CreateBuffer(out_config);

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in_config);
    CircularBufferConfig cb_ones_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_ones_config);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Input: k/256 for k in [0, 256), tiled 4x over the 1024 elements. Every
    // value is exact in bfloat16 (k has at most 8 significant bits), spans the
    // [0, 1) range upstream draws from, and sweeps 9 binades so any
    // exponent-range bug in exp/log shows up.
    std::vector<bfloat16> src_vec(TILE_ELEMS);
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        src_vec[i] = bfloat16(static_cast<float>(i % 256) / 256.0f);
    }
    tt::tt_metal::detail::WriteToBuffer(src_dram, src_vec);

    // The `ones` tile the chain adds (upstream fills a whole tile with 1.0).
    std::vector<bfloat16> ones_vec(TILE_ELEMS, bfloat16(1.0f));
    tt::tt_metal::detail::WriteToBuffer(ones_dram, ones_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {src_dram->address(), ones_dram->address()});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OPS});

    CreateKernel(program, "kernels/compute/compute_kernel.cpp", core, ComputeConfig{});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(NUM_OPS * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");
    printf("Completed successfully on the device, with %u op tiles\n", NUM_OPS);
    return 0;
}
