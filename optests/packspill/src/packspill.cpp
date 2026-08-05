// Differential probe: **how much L1 does one `pack_tile` write?**
//
// `pack.h` documents `pack_tile` as copying "a single tile" and advancing the
// CB write pointer "by single tile size", and the LLK programs the packer's
// datum counter from face_r_dim/num_faces — never from the CB's page size. So a
// CB page smaller than a tile is written out of bounds by its own producer.
// That is a claim about *hardware*, and this program pins it against the vendor
// reference simulator rather than against tt-sim's own opinion.
//
// Shape: the output CB is one full Int32 tile of L1 (4096 B) carved into
// SPILL_PAGES = 16 pages of 256 B each. The compute kernel reserves ONE page
// and packs ONE tile into it; the writer then dumps all 16 pages. Every byte
// read is inside the CB, so nothing here depends on what the allocator put
// next in L1 — the dump is fully determined by the packer's footprint.
//
//   page 0 alone valid under the contract -> pages 1..15 are zero
//   pack_tile writes a whole tile         -> pages 1..15 hold datums 64..1023
//
// optests/diff.sh runs this on tt-sim and on ttsim and compares the dumps:
//
//   TT_SIM_ARCH=blackhole ./optests/diff.sh packspill
//
// Result: the two simulators agree on all 1024 datums, and 960 of the 960
// datums past page 0 were written by that one `pack_tile`. Diff on **Blackhole**
// — ttsim-Wormhole rejects the Int32 unpack-to-Dst path this (and `optests/optest`)
// needs with "UndefinedBehavior: tensix_unpacr: unpack_to_dst=0 in_data_format=8",
// an oracle limitation unrelated to what is being measured here.
//
// Why it was written: `examples/pipestall` sized its output CB page to the
// chunk of data its kernels move (256 B) rather than to a tile, and a two-page
// output CB then had page 1 sitting inside page 0's pack footprint — the pack
// of chunk N shredded the unread chunk N-1. This is the evidence that the
// overrun is architectural and the example was wrong, not the simulator.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

#define TILE_ELEMS 1024  // 32x32 tile
#define SPILL_PAGES 16   // output CB pages carved out of one tile's worth of L1

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = 4 * TILE_ELEMS;  // Int32
    constexpr uint32_t out_page = tile_bytes / SPILL_PAGES;

    InterleavedBufferConfig in_config{
        .device = device,
        .size = tile_bytes,
        .page_size = tile_bytes,
        .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = tile_bytes,
        .page_size = tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto src_dram = CreateBuffer(in_config);
    auto dst_dram = CreateBuffer(out_config);

    // Input CB: one page, one whole tile — legal, so the tile the packer sees
    // is fully determined by DRAM rather than by whatever follows the CB.
    CircularBufferConfig cb_in_config =
        CircularBufferConfig(tile_bytes, {{CBIndex::c_0, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in_config);

    // Output CB: the deliberately-undersized case under test. Total size is one
    // tile, so the pack's whole footprint stays inside the buffer.
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(SPILL_PAGES * out_page, {{CBIndex::c_1, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_1, out_page);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Varied Int32 input so a spilled datum is distinguishable from a zero fill
    // and from any other datum of the tile.
    std::vector<uint32_t> in_data(TILE_ELEMS);
    for (int i = 0; i < TILE_ELEMS; i++) {
        in_data[i] = static_cast<uint32_t>(i) * 2654435761u;
    }
    tt::tt_metal::detail::WriteToBuffer(src_dram, in_data);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, reader, core, {src_dram->address(), tile_bytes});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), tile_bytes});

    KernelHandle compute = CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false});
    SetRuntimeArgs(program, compute, core, {});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> out_data(TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_data);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_data.size(); i++) {
        printf("%08x", out_data[i]);
    }
    printf("\n");

    // Human-readable verdict alongside the machine-comparable dump.
    size_t nonzero_past_page0 = 0;
    for (size_t i = TILE_ELEMS / SPILL_PAGES; i < out_data.size(); i++) {
        if (out_data[i] != 0) nonzero_past_page0++;
    }
    printf(
        "packspill: %zu of %zu datums past page 0 were written by one pack_tile\n",
        nonzero_past_page0,
        out_data.size() - TILE_ELEMS / SPILL_PAGES);
    printf("Completed successfully on the device, with %d datums\n", TILE_ELEMS);
    return 0;
}
