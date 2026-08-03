// Deterministic differential version of upstream's `sfpu_eltwise_chain`
// programming example (softplus = log(exp(x) + 1) through the SFPU).
//
// Upstream aborts on tt-sim with `PCC not high enough. Result PCC: 0.9986145,
// Expected PCC: 0.999` while ttsim (the vendor reference) passes at ~0.99986
// over 8 runs (spread 0.99985..0.99988) -- so the gap is real, not sampling
// noise. But upstream seeds its input from `std::random_device`, so the two
// sims never see the same numbers and the PCCs are not directly comparable.
//
// This program is upstream's, with the randomness removed and a hex dump added:
//   * kernels/{dataflow/reader.cpp, dataflow/writer.cpp, compute/compute.cpp}
//     are byte-for-byte copies of the upstream example's kernels, so the exact
//     same compiled code runs on both sims;
//   * the CB indices (c_0 input, c_1 ones, c_2 result), the on-device
//     generation of the ones tile by the reader, and the TensorAccessor-based
//     page read/write are all preserved -- these are what optests/sfpuchain
//     (which matches bit-exactly) does NOT cover;
//   * the input is fixed and the output tile is dumped as
//     `OPDIFF_RESULT:<hex>`, so optests/diff.sh can compare tt-sim against
//     ttsim bit-for-bit instead of comparing two PCCs of different data.
//
// No host tilize/untilize: the tile is dumped in device order, which is all a
// bit-for-bit diff needs.

#include <bit>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HW;

    // Fixed input in [0, 1) -- the range upstream draws from -- with every
    // value exact in bfloat16 (k/256 for k < 256, tiled 4x over the 1024
    // elements) so nothing here depends on host rounding.
    std::vector<bfloat16> src_vec(constants::TILE_HW);
    for (uint32_t i = 0; i < constants::TILE_HW; i++) {
        src_vec[i] = bfloat16(static_cast<float>(i % 256) / 256.0f);
    }

    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);
    tt::tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    constexpr uint32_t src_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src_config =
        CircularBufferConfig(single_tile_size, {{src_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src_config);

    constexpr uint32_t ones_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_ones_config =
        CircularBufferConfig(single_tile_size, {{ones_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ones_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_ones_config);

    constexpr uint32_t result_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_result_config =
        CircularBufferConfig(single_tile_size, {{result_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(result_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_result_config);

    std::vector<uint32_t> reader_compile_time_args = {src_cb_index, ones_cb_index};
    TensorAccessorArgs(*src_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program, "kernels/dataflow/reader.cpp", core, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {result_cb_index};
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program, "kernels/dataflow/writer.cpp", core, tt::tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    std::vector<uint32_t> compute_compile_time_args = {src_cb_index, ones_cb_index, result_cb_index};
    CreateKernel(
        program, "kernels/compute/compute.cpp", core, tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    SetRuntimeArgs(program, reader_kernel_id, core, {src_dram_buffer->address()});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer->address()});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(constants::TILE_HW);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");
    printf("Completed successfully on the device, with 1 op tile\n");
    return 0;
}
