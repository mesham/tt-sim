// Op test for `matmul_tiles` with *distinct* operand tile indices.
//
// The in-tree matmul example (`examples/six`) only ever calls
// `matmul_tiles(cb0, cb1, 0, 0, 0)` — it streams one tile at a time through a
// FIFO circular buffer. The other first-class tt-metal pattern keeps a whole
// operand block resident (`cb_wait_front(cb, num_tiles)`) and indexes the tiles
// directly, e.g. `matmul_tiles(cb0, cb1, i, j, 0)` with i != j.
//
// Both operand CBs hold two constant tiles:
//     cb_in0: tile 0 = 1.0, tile 1 = 2.0
//     cb_in1: tile 0 = 3.0, tile 1 = 5.0
// For 32x32 constant tiles a and b, matmul gives every output element 32*a*b,
// so each of the four index combinations has a distinct expected value:
//
//     (0,0) -> 1*3 -> 96      (1,1) -> 2*5 -> 320
//     (1,0) -> 2*3 -> 192     (0,1) -> 1*5 -> 160
//
// All four are exactly representable in bfloat16, so this checks bit-exactly.
// Also dumps `OPDIFF_RESULT:<hex>` so optests/diff.sh can diff it against ttsim.

#include <bit>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t NUM_IN_TILES = 2;
// Four single matmuls, then one accumulating two distinct-index matmuls.
constexpr uint32_t NUM_SINGLE = 4;
constexpr uint32_t NUM_OUT_TILES = NUM_SINGLE + 1;

// Must match the (in0_tile_index, in1_tile_index) sequence in the compute kernel.
constexpr float IN0_VALUES[NUM_IN_TILES] = {1.0f, 2.0f};
constexpr float IN1_VALUES[NUM_IN_TILES] = {3.0f, 5.0f};
constexpr uint32_t IN0_INDICES[NUM_SINGLE] = {0, 1, 1, 0};
constexpr uint32_t IN1_INDICES[NUM_SINGLE] = {0, 1, 0, 1};

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMS;

    // One DRAM buffer per operand (both tiles in a single page, so a single
    // bank) plus one for the four output tiles.
    InterleavedBufferConfig in_config{
        .device = device,
        .size = NUM_IN_TILES * tile_bytes,
        .page_size = NUM_IN_TILES * tile_bytes,
        .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = NUM_OUT_TILES * tile_bytes,
        .page_size = NUM_OUT_TILES * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto src0_dram = CreateBuffer(in_config);
    auto src1_dram = CreateBuffer(in_config);
    auto dst_dram = CreateBuffer(out_config);

    // Input CBs hold the whole operand block (both tiles resident at once) —
    // that residency is what lets the compute kernel index tiles directly.
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(NUM_IN_TILES * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in0_config);
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(NUM_IN_TILES * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in1_config);
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(2 * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Constant tiles — the tiled layout doesn't matter when every datum in a
    // tile is the same value.
    std::vector<bfloat16> src0_vec(NUM_IN_TILES * TILE_ELEMS);
    std::vector<bfloat16> src1_vec(NUM_IN_TILES * TILE_ELEMS);
    for (uint32_t t = 0; t < NUM_IN_TILES; t++) {
        for (uint32_t i = 0; i < TILE_ELEMS; i++) {
            src0_vec[t * TILE_ELEMS + i] = bfloat16(IN0_VALUES[t]);
            src1_vec[t * TILE_ELEMS + i] = bfloat16(IN1_VALUES[t]);
        }
    }
    tt::tt_metal::detail::WriteToBuffer(src0_dram, src0_vec);
    tt::tt_metal::detail::WriteToBuffer(src1_dram, src1_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {src0_dram->address(), src1_dram->address(), NUM_IN_TILES});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OUT_TILES});

    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(NUM_OUT_TILES * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");

    // Every element of output tile t should equal 32 * in0 * in1 (the last one
    // the sum of two such products).
    float expectations[NUM_OUT_TILES];
    for (uint32_t t = 0; t < NUM_SINGLE; t++) {
        expectations[t] = 32.0f * IN0_VALUES[IN0_INDICES[t]] * IN1_VALUES[IN1_INDICES[t]];
    }
    expectations[NUM_SINGLE] =
        32.0f * IN0_VALUES[0] * IN1_VALUES[1] + 32.0f * IN0_VALUES[1] * IN1_VALUES[0];

    bool pass = true;
    for (uint32_t t = 0; t < NUM_OUT_TILES; t++) {
        const float expected = expectations[t];
        float worst = expected;
        uint32_t bad = 0;
        for (uint32_t i = 0; i < TILE_ELEMS; i++) {
            const float got = static_cast<float>(out_vec[t * TILE_ELEMS + i]);
            if (got != expected) {
                bad++;
                worst = got;
            }
        }
        if (t < NUM_SINGLE) {
            const uint32_t i0 = IN0_INDICES[t];
            const uint32_t i1 = IN1_INDICES[t];
            printf(
                "matmul_tiles(in0=%u, in1=%u): expected %.1f, got %.1f%s%s\n",
                i0, i1, expected, worst,
                bad ? " -- MISMATCH" : "",
                (i0 == i1) ? "" : "   [distinct tile indices]");
        } else {
            printf(
                "matmul_tiles(0,1) + matmul_tiles(1,0) into one DST: expected %.1f, got %.1f%s\n",
                expected, worst, bad ? " -- MISMATCH" : "");
        }
        pass &= (bad == 0);
    }

    if (pass) {
        printf("Completed successfully on the device, with %u output tiles\n", NUM_OUT_TILES);
    } else {
        printf("Failure on the device, matmul_tiles produced wrong results\n");
    }
    return pass ? 0 : 1;
}
