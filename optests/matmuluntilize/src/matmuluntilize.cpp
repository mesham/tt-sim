// Op test for a per-output-tile `matmul_tiles` -> `pack` -> `untilize_block`
// loop, and for the `matmul_init` that has to be re-issued *inside* it.
//
// Reported by the hedgehope compiler team in
// `tt-sim-untilize-between-tiles-matmul-init-handoff.md` -- the output-side
// twin of the input-side tilize case in
// `tt-sim-ondevice-tilize-matmul-init-handoff.md` (see optests/tilizematmul).
// Their 2-D grid GEMM emitted `matmul_init` once above the output-tile loop.
// The first output tile computes fine; the second wedges on Blackhole silicon
// (Watcher: CWFW, W, UPMD, MWDD, K, with the tiled operand CBs holding
// received-but-unacked pages and the NoC sanitizer clean).
//
// Two run modes, same binary:
//   (no argument)  `matmul_init` at the top of every iteration -- the correct
//                  form, and the false-positive guard: a kernel that
//                  interleaves matmul and untilize *correctly* must stay green.
//                  This is what `optests/diff.sh matmuluntilize` runs.
//   `buggy`        `matmul_init` hoisted above the loop -- the reported bug.
//
// A[i] = (i + 1) x identity and B is 1024 distinct exactly-representable
// bfloat16 values, so output tile i is exactly (i + 1) * B, delivered
// row-major by the untilize. The two iterations therefore have different exact
// answers: an iteration that reran the previous tile, or that copied an operand
// instead of multiplying, cannot hide.

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>

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
constexpr uint32_t NUM_OUT = 2;  // MUST match NUM_OUT in the compute kernel

static uint32_t tilized_index(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE_DIM) * 2 + (c / FACE_DIM);
    return face * FACE_DIM * FACE_DIM + (r % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// 1024 distinct values, every one exactly representable in bfloat16.
static float ramp(uint32_t j) {
    return std::ldexp(1.0f + static_cast<float>(j % 128) / 128.0f, static_cast<int>(j / 128) - 4);
}

int main(int argc, char** argv) {
    const bool buggy = (argc > 1 && std::strcmp(argv[1], "buggy") == 0);
    printf("matmul+untilize loop: matmul_init %s re-issued per output tile\n", buggy ? "NOT" : "IS");

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    auto make_buffer = [&](uint32_t bytes) {
        InterleavedBufferConfig config{
            .device = device, .size = bytes, .page_size = bytes, .buffer_type = BufferType::DRAM};
        return CreateBuffer(config);
    };
    auto a_dram = make_buffer(NUM_OUT * TILE_BYTES);
    auto b_dram = make_buffer(TILE_BYTES);
    auto dst_dram = make_buffer(NUM_OUT * TILE_BYTES);

    auto make_cb = [&](CBIndex idx, uint32_t pages) {
        CircularBufferConfig cfg =
            CircularBufferConfig(pages * TILE_BYTES, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, TILE_BYTES);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 2 * NUM_OUT);  // A, tiled
    make_cb(CBIndex::c_1, 2);            // B, tiled
    make_cb(CBIndex::c_24, 2);           // matmul result, tiled
    make_cb(CBIndex::c_16, 2 * NUM_OUT); // untilized output

    // A[i] = (i + 1) * identity, B = ramp; both written tiled.
    std::vector<bfloat16> a_vec(NUM_OUT * TILE_ELEMS, bfloat16(0.0f));
    std::vector<bfloat16> b_vec(TILE_ELEMS);
    std::vector<bfloat16> golden(NUM_OUT * TILE_ELEMS);  // row-major, as untilized out
    for (uint32_t i = 0; i < NUM_OUT; i++) {
        for (uint32_t r = 0; r < TILE_DIM; r++) {
            a_vec[i * TILE_ELEMS + tilized_index(r, r)] = bfloat16(static_cast<float>(i + 1));
        }
    }
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const float value = ramp(r * TILE_DIM + c);
            b_vec[tilized_index(r, c)] = bfloat16(value);
            for (uint32_t i = 0; i < NUM_OUT; i++) {
                golden[i * TILE_ELEMS + r * TILE_DIM + c] = bfloat16(value * static_cast<float>(i + 1));
            }
        }
    }
    tt::tt_metal::detail::WriteToBuffer(a_dram, a_vec);
    tt::tt_metal::detail::WriteToBuffer(b_dram, b_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {a_dram->address(), b_dram->address(), NUM_OUT});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OUT});

    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .defines = {{"REISSUE_MATMUL_INIT", buggy ? "0" : "1"}}});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(NUM_OUT * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");

    uint32_t total_errors = 0;
    for (uint32_t i = 0; i < NUM_OUT; i++) {
        uint32_t errors = 0;
        for (uint32_t j = 0; j < TILE_ELEMS; j++) {
            const float got = static_cast<float>(out_vec[i * TILE_ELEMS + j]);
            const float want = static_cast<float>(golden[i * TILE_ELEMS + j]);
            if (got != want) {
                if (errors == 0) {
                    printf(
                        "output tile %u: first mismatch at (%u, %u): got %g want %g\n",
                        i, j / TILE_DIM, j % TILE_DIM, got, want);
                }
                errors++;
            }
        }
        printf("output tile %u: %u/%u errors\n", i, errors, TILE_ELEMS);
        total_errors += errors;
    }

    if (total_errors != 0) {
        printf("Failed on the device, %u errors over %u output tiles\n", total_errors, NUM_OUT);
        return 1;
    }
    printf("Completed successfully on the device, with %u untilized output tiles\n", NUM_OUT);
    return 0;
}
