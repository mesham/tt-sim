// Op test for the tile-transpose path (host side). Feeds one Float32 input tile
// through `transpose_tile` / `transpose_dest` / a dest-reuse binary (see
// kernels/compute/compute_kernel.cpp), one output tile per op. The output tiles
// are read back and dumped as `OPDIFF_RESULT:<hex>`; optests/diff.sh runs this
// same binary on tt-sim and on ttsim and compares the dumps. ttsim is the
// oracle — there is no local golden.
//
// This is the program that reaches the matrix unit's Dst<->Src move ops:
// `transpose_dest` is a MOVD2B / TRNSPSRCB / MOVB2A / MOVB2D / MOVA2D sequence,
// and the DEST_TO_SRCA binary reuses Dst as an operand via MOVD2A.
//
// Float32 (with fp32_dest_acc_en) is deliberate: it selects the 32-bit
// transpose path, where each 32-bit datum is transposed as two 16-bit halves
// and the SrcA format is switched under the moves.
//
// Add an op: append a RUN_OP(...) in the compute kernel and bump NUM_OPS here.

#include <cstring>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t NUM_OPS = 3;  // MUST match the compute kernel's op count

static uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = 4 * TILE_ELEMS;  // Float32

    InterleavedBufferConfig in_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = NUM_OPS * tile_bytes,
        .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto src_dram = CreateBuffer(in_config);
    auto dst_dram = CreateBuffer(out_config);

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in_config);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Every datum distinct, and distinct in *both* 16-bit halves, so a
    // transpose that drops or swaps a half shows up immediately. i + 1 keeps
    // the value normal (no zero/denormal) and 1/256 keeps the exponent small.
    std::vector<uint32_t> in_data(TILE_ELEMS);
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        in_data[i] = f2u(1.0f + static_cast<float>(i + 1) / 256.0f);
    }
    tt::tt_metal::detail::WriteToBuffer(src_dram, in_data);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {src_dram->address()});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OPS});

    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> out_data(NUM_OPS * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_data);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_data.size(); i++) {
        printf("%08x", out_data[i]);
    }
    printf("\n");
    printf("Completed successfully on the device, with %u op tiles\n", NUM_OPS);
    return 0;
}
