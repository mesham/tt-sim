// FP32 SFPU-math op-coverage harness (host side). Mirrors optest but feeds a
// Float32 input tile through a sequence of SFPU math ops (see
// kernels/compute/compute_kernel.cpp), one output tile per op. The output tiles
// are read back and dumped as `OPDIFF_RESULT:<hex>`; optests/diff.sh runs the
// same binary on tt-sim and on ttsim and compares the dumps (ttsim is the
// oracle — the SFPU approximations must match bit-for-bit, not just numerically).
//
// Add an op: append a RUN_OP(...) in the compute kernel and bump NUM_OPS here.

#include <cstring>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

#define TILE_ELEMS 1024  // 32x32 tile
#define NUM_OPS 1        // MUST match the compute kernel's op count

using namespace tt;
using namespace tt::tt_metal;

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
        .device = device, .size = NUM_OPS * tile_bytes, .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto src_dram = CreateBuffer(in_config);
    auto dst_dram = CreateBuffer(out_config);

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in_config);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Varied FP32 input in [1.0, ~11.24): avoids 0/denormal/overflow so recip
    // and friends stay in their normal range.
    std::vector<uint32_t> in_data(TILE_ELEMS);
    for (int i = 0; i < TILE_ELEMS; i++) {
        in_data[i] = f2u(1.0f + static_cast<float>(i) * 0.01f);
    }
    tt::tt_metal::detail::WriteToBuffer(src_dram, in_data);

    KernelHandle reader = CreateKernel(
        program, "kernels/dataflow/read_kernel.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, reader, core, {src_dram->address(), TILE_ELEMS, TILE_ELEMS});

    KernelHandle writer = CreateKernel(
        program, "kernels/dataflow/write_kernel.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OPS * TILE_ELEMS, TILE_ELEMS});

    KernelHandle compute = CreateKernel(
        program, "kernels/compute/compute_kernel.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
    SetRuntimeArgs(program, compute, core, {});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> out_data(NUM_OPS * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_data);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_data.size(); i++) {
        printf("%08x", out_data[i]);
    }
    printf("\n");
    printf("Completed successfully on the device, with %d op tiles\n", NUM_OPS);
    return 0;
}
