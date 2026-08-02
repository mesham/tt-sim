// Op test for `reduce_tile` over an ENTIRELY NEGATIVE input tile. Identical to
// optests/reduce apart from the sign of the data, and that sign is the whole
// point: it is the case that needs Dst row-validity ("zero flags") to be
// modelled.
//
// `reduce_tile` ZEROACCs its destination and then accumulates into it with
// GMPOOL, which maxes each SrcA column against the Dst row it is accumulating
// into. ZEROACC does not zero that row's data, it clears the row's zero flag,
// and GMPOOL (alone among Dst consumers) reads a flag-cleared row as all-ones,
// i.e. minus infinity. A simulator that instead zeroes the data reads it back
// as +0, so every MAX reduction here saturates at 0 rather than returning the
// true (negative) maximum — a mismatch invisible to optests/reduce, whose
// inputs are all non-negative.
//
// Dumped as `OPDIFF_RESULT:<hex>`; optests/diff.sh runs this same binary on
// tt-sim and on ttsim and compares. ttsim is the oracle — no local golden.
//
// Add an op: append a RUN_REDUCE(...) in the compute kernel and bump NUM_OPS.

#include <bit>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t FACE_ELEMS = 16 * 16;
constexpr uint32_t NUM_FACES = 4;
constexpr uint32_t FACE_COLS = 16;
constexpr uint32_t NUM_OPS = 5;  // MUST match the compute kernel's op count

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMS;

    // One data tile, one scaler tile, NUM_OPS output tiles (one per reduction).
    InterleavedBufferConfig tile_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = NUM_OPS * tile_bytes,
        .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto src_dram = CreateBuffer(tile_config);
    auto scaler_dram = CreateBuffer(tile_config);
    auto dst_dram = CreateBuffer(out_config);

    CircularBufferConfig cb_in_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in_config);
    CircularBufferConfig cb_scaler_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_scaler_config);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Data tile: multiples of -0.5 in [-20.5, -0.5] — strictly negative, with
    // no zero anywhere, so a reduction that reads a cleared Dst row as +0
    // rather than as minus infinity saturates at 0 for every element. All are
    // exactly representable in bfloat16, and (i*37 % 41) is a permutation cycle
    // so every row and every column has a distinct maximum.
    std::vector<bfloat16> src_vec(TILE_ELEMS);
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        src_vec[i] = bfloat16(-0.5f * static_cast<float>((i * 37) % 41 + 1));
    }
    tt::tt_metal::detail::WriteToBuffer(src_dram, src_vec);

    // Scaler tile: 1.0 in the first row of each 16x16 face, zero elsewhere —
    // the layout reduce_tile documents (and what tt-metal's own
    // generate_reduce_scaler() writes).
    std::vector<bfloat16> scaler_vec(TILE_ELEMS, bfloat16(0.0f));
    for (uint32_t f = 0; f < NUM_FACES; f++) {
        for (uint32_t c = 0; c < FACE_COLS; c++) {
            scaler_vec[f * FACE_ELEMS + c] = bfloat16(1.0f);
        }
    }
    tt::tt_metal::detail::WriteToBuffer(scaler_dram, scaler_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {src_dram->address(), scaler_dram->address()});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OPS});

    CreateKernel(
        program, "kernels/compute/compute_kernel.cpp", core, ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

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
