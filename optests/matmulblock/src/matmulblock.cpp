// Op test for `matmul_block` over a multi-tile block (ct_dim / rt_dim / kt_dim > 1).
//
// `optests/matmulidx` covers the per-tile form (`matmul_tiles` with distinct
// operand tile indices). This is its block twin: `matmul_block` walks several
// operand tiles per call at computed L1 offsets, so it exercises the unpacker's
// MOP-driven base-address bump and the two config contexts far harder than a
// single-tile matmul does.
//
// Both operand CBs hold a whole 2x2 tile block, resident at once:
//     cb_in0 (A, row-major rt x kt): tiles 1.0, 2.0, 4.0, 8.0
//     cb_in1 (B, row-major kt x ct): tiles 3.0, 5.0, 16.0, 32.0
// For 32x32 constant tiles every output element of C = A @ B is
// 32 * sum_k A[r][k] * B[k][c], giving four distinct expected values, all exact
// in bfloat16 -- so a mis-addressed operand tile is unambiguous rather than a
// rounding question.
//
// The compute kernel emits the same C block three times, under the three
// interesting `matmul_block` shapes (ct=2/rt=1, ct=2/rt=2, ct=1/rt=2 -- the last
// taking the LLK's `!reuse_a` branch), so all twelve output tiles are checked
// against the same four values.
//
// Also dumps `OPDIFF_RESULT:<hex>` so optests/diff.sh can diff it against ttsim.

#include <bit>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t RT = 2;
constexpr uint32_t CT = 2;
constexpr uint32_t KT = 2;
constexpr uint32_t NUM_IN_TILES = RT * KT;
constexpr uint32_t NUM_SHAPES = 3;
constexpr uint32_t NUM_OUT_TILES = NUM_SHAPES * RT * CT;

// Constant tile values, row-major: A is RT x KT, B is KT x CT.
constexpr float A_VALUES[RT * KT] = {1.0f, 2.0f, 4.0f, 8.0f};
constexpr float B_VALUES[KT * CT] = {3.0f, 5.0f, 16.0f, 32.0f};

// Per shape, the (r, c) of each packed output tile, in pack order. Shapes 1 and
// 2 pack row-major; shape 3 (ct_dim=1) walks a tile-column at a time.
constexpr uint32_t OUT_R[NUM_SHAPES][RT * CT] = {{0, 0, 1, 1}, {0, 0, 1, 1}, {0, 1, 0, 1}};
constexpr uint32_t OUT_C[NUM_SHAPES][RT * CT] = {{0, 1, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}};
constexpr const char* SHAPE_NAME[NUM_SHAPES] = {
    "ct_dim=2 rt_dim=1 kt_dim=2 (reuse A, one output row per call)",
    "ct_dim=2 rt_dim=2 kt_dim=2 (reuse A, whole block per call)",
    "ct_dim=1 rt_dim=2 kt_dim=2 (reuse B, one output column per call)"};

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMS;

    // One DRAM buffer per operand block (all four tiles in a single page, so a
    // single bank) plus one for the twelve output tiles.
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

    // Input CBs hold the whole operand block (all four tiles resident at once) —
    // that residency is what lets `matmul_block` index into it.
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(NUM_IN_TILES * tile_bytes, {{CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in0_config);
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(NUM_IN_TILES * tile_bytes, {{CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_in1_config);
    // The compute kernel reserves a whole 2x2 block at once, so the output CB
    // needs room for four tiles.
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(RT * CT * tile_bytes, {{CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Constant tiles — the tiled layout doesn't matter when every datum in a
    // tile is the same value.
    std::vector<bfloat16> src0_vec(NUM_IN_TILES * TILE_ELEMS);
    std::vector<bfloat16> src1_vec(NUM_IN_TILES * TILE_ELEMS);
    for (uint32_t t = 0; t < NUM_IN_TILES; t++) {
        for (uint32_t i = 0; i < TILE_ELEMS; i++) {
            src0_vec[t * TILE_ELEMS + i] = bfloat16(A_VALUES[t]);
            src1_vec[t * TILE_ELEMS + i] = bfloat16(B_VALUES[t]);
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

    // C[r][c] element = 32 * sum_k A[r][k] * B[k][c].
    float golden[RT][CT];
    for (uint32_t r = 0; r < RT; r++) {
        for (uint32_t c = 0; c < CT; c++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < KT; k++) {
                acc += A_VALUES[r * KT + k] * B_VALUES[k * CT + c];
            }
            golden[r][c] = 32.0f * acc;
        }
    }

    bool pass = true;
    for (uint32_t s = 0; s < NUM_SHAPES; s++) {
        uint32_t shape_bad = 0;
        for (uint32_t t = 0; t < RT * CT; t++) {
            const uint32_t tile = s * RT * CT + t;
            const uint32_t r = OUT_R[s][t];
            const uint32_t c = OUT_C[s][t];
            const float expected = golden[r][c];
            float worst = expected;
            uint32_t bad = 0;
            for (uint32_t i = 0; i < TILE_ELEMS; i++) {
                const float got = static_cast<float>(out_vec[tile * TILE_ELEMS + i]);
                if (got != expected) {
                    bad++;
                    worst = got;
                }
            }
            if (bad) {
                printf(
                    "  C[%u][%u] (output tile %u): expected %.1f, got %.1f -- MISMATCH\n",
                    r, c, tile, expected, worst);
            }
            shape_bad += bad;
        }
        printf("matmul_block %s: %s\n", SHAPE_NAME[s], shape_bad ? "MISMATCH" : "ok");
        pass &= (shape_bad == 0);
    }

    if (pass) {
        printf("Completed successfully on the device, with %u output tiles\n", NUM_OUT_TILES);
    } else {
        printf("Failure on the device, matmul_block produced wrong results\n");
    }
    return pass ? 0 : 1;
}
