// Op test for the *transpose* operand of `matmul_init` (host side).
//
// Raised by the hedgehope compiler team: their codegen emitted
// `matmul_init(5, 4, 4)` -- the third operand is the transpose flag and 4 is
// truthy -- and tt-sim produced the un-transposed answer, which they read as
// tt-sim ignoring the flag. It does not; the flag lands in two places with
// different widths (see the compute kernel), and this program pins down all
// three cases against the vendor reference simulator:
//
//   op 0  CONTROL  matmul_init(in0, in1, 0)  -- C = A @ B      (computed golden)
//   op 1           matmul_init(in0, in1, 1)  -- C = A @ B^T    (computed golden)
//   op 2           matmul_init(in0, in1, 4)  -- differential only
//
// A is the 32x32 identity and B is 1024 distinct exactly-representable bfloat16
// values, so ops 0 and 1 have exact goldens and "transposed" versus "not" is an
// unmistakable difference rather than a rounding question. Op 2 has no golden --
// it is the argument the compiler team actually emitted, and what matters is
// that tt-sim and ttsim agree on it, which `optests/diff.sh matmultranspose`
// checks via the `OPDIFF_RESULT` dump.

#include <bit>
#include <cmath>
#include <cstdio>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t TILE_ELEMS = TILE_DIM * TILE_DIM;
constexpr uint32_t TILE_BYTES = sizeof(bfloat16) * TILE_ELEMS;
constexpr uint32_t NUM_OPS = 3;  // MUST match NUM_OPS in the compute kernel
constexpr const char* OP_NAME[NUM_OPS] = {"transpose=0 (control)", "transpose=1", "transpose=4"};

// Tilized index of row-major element (r, c) within one tile: four 16x16 faces
// in the order (top-left, top-right, bottom-left, bottom-right), each face
// row-major.
static uint32_t tilized_index(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE_DIM) * 2 + (c / FACE_DIM);
    return face * FACE_DIM * FACE_DIM + (r % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// 1024 distinct values, every one exactly representable in bfloat16.
static float ramp(uint32_t j) {
    return std::ldexp(1.0f + static_cast<float>(j % 128) / 128.0f, static_cast<int>(j / 128) - 4);
}

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    auto make_buffer = [&](uint32_t bytes) {
        InterleavedBufferConfig config{
            .device = device, .size = bytes, .page_size = bytes, .buffer_type = BufferType::DRAM};
        return CreateBuffer(config);
    };
    auto in0_dram = make_buffer(TILE_BYTES);
    auto in1_dram = make_buffer(TILE_BYTES);
    auto dst_dram = make_buffer(NUM_OPS * TILE_BYTES);

    auto make_cb = [&](CBIndex idx, uint32_t pages) {
        CircularBufferConfig cfg =
            CircularBufferConfig(pages * TILE_BYTES, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, TILE_BYTES);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 2);
    make_cb(CBIndex::c_1, 2);
    make_cb(CBIndex::c_16, 2 * NUM_OPS);

    // A = identity, B = ramp. Both written in the tiled layout the CBs expect.
    std::vector<bfloat16> in0_vec(TILE_ELEMS, bfloat16(0.0f));
    std::vector<bfloat16> in1_vec(TILE_ELEMS);
    std::vector<bfloat16> golden(NUM_OPS * TILE_ELEMS, bfloat16(0.0f));
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        in0_vec[tilized_index(r, r)] = bfloat16(1.0f);
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const bfloat16 value = bfloat16(ramp(r * TILE_DIM + c));
            in1_vec[tilized_index(r, c)] = value;
            golden[0 * TILE_ELEMS + tilized_index(r, c)] = value;  // op 0: C = B
            golden[1 * TILE_ELEMS + tilized_index(c, r)] = value;  // op 1: C = B^T
        }
    }
    tt::tt_metal::detail::WriteToBuffer(in0_dram, in0_vec);
    tt::tt_metal::detail::WriteToBuffer(in1_dram, in1_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {in0_dram->address(), in1_dram->address()});

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

    // Only ops 0 and 1 have a golden; op 2 is differential-only, so just report
    // whether it matched either of them (which is the interesting question).
    uint32_t total_errors = 0;
    for (uint32_t op = 0; op < 2; op++) {
        uint32_t errors = 0;
        for (uint32_t i = 0; i < TILE_ELEMS; i++) {
            const float got = static_cast<float>(out_vec[op * TILE_ELEMS + i]);
            const float want = static_cast<float>(golden[op * TILE_ELEMS + i]);
            if (got != want) {
                if (errors == 0) {
                    printf("op %u (%s): first mismatch at %u: got %g want %g\n", op, OP_NAME[op], i, got, want);
                }
                errors++;
            }
        }
        printf("op %u (%s): %u/%u errors\n", op, OP_NAME[op], errors, TILE_ELEMS);
        total_errors += errors;
    }
    uint32_t like0 = 0, like1 = 0;
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        like0 += (out_vec[2 * TILE_ELEMS + i] == out_vec[0 * TILE_ELEMS + i]);
        like1 += (out_vec[2 * TILE_ELEMS + i] == out_vec[1 * TILE_ELEMS + i]);
    }
    printf(
        "op 2 (%s): matches op 0 in %u/%u elements, op 1 in %u/%u (differential-only)\n",
        OP_NAME[2], like0, TILE_ELEMS, like1, TILE_ELEMS);

    if (total_errors != 0) {
        printf("Failed on the device, %u errors\n", total_errors);
        return 1;
    }
    printf("Completed successfully on the device, with %u op tiles\n", NUM_OPS);
    return 0;
}
