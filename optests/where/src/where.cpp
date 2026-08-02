// Op test for `where_tile` (host side). Feeds three Int32 input tiles — a
// condition and two value tiles — through the ternary select in
// kernels/compute/compute_kernel.cpp, one output tile per op. The output tiles
// are read back and dumped as `OPDIFF_RESULT:<hex>`; optests/diff.sh runs this
// same binary on tt-sim and on ttsim and compares the dumps. ttsim is the
// oracle — there is no local golden.
//
// On Blackhole the where kernel (tt-llk ckernel_sfpu_where.h) issues its
// per-lane selects as SFPLOADMACRO sequences unless DISABLE_SFPLOADMACRO is
// defined. Both simulators explicitly decline SFPLOADMACRO (ttsim: "explicitly
// out of scope"), so there is no oracle for the macro form — `optests/where/env`
// sets TT_METAL_DISABLE_SFPLOADMACRO=1 and this program diffs the non-macro
// select sequence both sims do model.
//
// Add an op: append a block in the compute kernel and bump NUM_OPS here.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t NUM_OPS = 2;  // MUST match the compute kernel's op count

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = 4 * TILE_ELEMS;  // Int32

    InterleavedBufferConfig tile_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = NUM_OPS * tile_bytes,
        .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto cond_dram = CreateBuffer(tile_config);
    auto a_dram = CreateBuffer(tile_config);
    auto b_dram = CreateBuffer(tile_config);
    auto dst_dram = CreateBuffer(out_config);

    const CBIndex in_cbs[3] = {CBIndex::c_0, CBIndex::c_1, CBIndex::c_2};
    for (auto cb : in_cbs) {
        CircularBufferConfig cfg =
            CircularBufferConfig(2 * tile_bytes, {{cb, tt::DataFormat::Int32}}).set_page_size(cb, tile_bytes);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    }
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Condition: a mix of zero and non-zero (including negatives), so both
    // branches of the select are taken within every face.
    std::vector<uint32_t> cond(TILE_ELEMS), a(TILE_ELEMS), b(TILE_ELEMS);
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        cond[i] = (i % 3 == 0) ? 0u : ((i % 3 == 1) ? (i + 1) : static_cast<uint32_t>(-static_cast<int32_t>(i + 1)));
        a[i] = 0x11110000u + i;
        b[i] = 0x22220000u + i;
    }
    tt::tt_metal::detail::WriteToBuffer(cond_dram, cond);
    tt::tt_metal::detail::WriteToBuffer(a_dram, a);
    tt::tt_metal::detail::WriteToBuffer(b_dram, b);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {cond_dram->address(), a_dram->address(), b_dram->address()});

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
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

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
