// Differential op-coverage harness (host side). Feeds one Int32 input tile
// through the compute kernel, which applies a *sequence* of ops (see
// kernels/compute/compute_kernel.cpp) and emits one output tile per op. The
// output tiles are read back and dumped as `OPDIFF_RESULT:<hex>`; optests/diff.sh
// runs this same binary on tt-sim and on ttsim and compares the dumps. ttsim is
// the oracle — there is no local golden.
//
// Add an op: append a RUN_OP(...) in the compute kernel and bump NUM_OPS here.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

#define TILE_ELEMS 1024  // 32x32 tile
#define NUM_OPS 6        // MUST match the compute kernel's op count

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = 4 * TILE_ELEMS;  // Int32

    // One input tile in DRAM; NUM_OPS output tiles (one per op), contiguous.
    InterleavedBufferConfig in_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device, .size = NUM_OPS * tile_bytes, .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto src_dram = CreateBuffer(in_config);
    auto dst_dram = CreateBuffer(out_config);

    // Input CB (compute holds it for the whole run) and output CB (one tile per op).
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in_config);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Varied Int32 input (Knuth multiplicative hash — spreads bits so the
    // bitwise/arith ops produce distinct, non-trivial results).
    std::vector<uint32_t> in_data(TILE_ELEMS);
    for (int i = 0; i < TILE_ELEMS; i++) {
        in_data[i] = static_cast<uint32_t>(i) * 2654435761u;
    }
    tt::tt_metal::detail::WriteToBuffer(src_dram, in_data);

    KernelHandle reader = CreateKernel(
        program, "kernels/dataflow/read_kernel.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, reader, core, {src_dram->address(), TILE_ELEMS, TILE_ELEMS});  // 1 tile in

    KernelHandle writer = CreateKernel(
        program, "kernels/dataflow/write_kernel.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OPS * TILE_ELEMS, TILE_ELEMS});  // N tiles out

    KernelHandle compute = CreateKernel(
        program, "kernels/compute/compute_kernel.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});
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
