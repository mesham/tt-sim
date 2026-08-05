// Op test for on-device `tilize_block` -> `matmul_tiles` in one compute kernel,
// and for the `matmul_init` that has to be re-issued between the two.
//
// Reported by the hedgehope compiler team: a generated GEMM issued `matmul_init`
// once at the top of the kernel, before the tilize phase, and never again. That
// deadlocks on Blackhole and Wormhole silicon (Watcher: every compute core at
// UPMD, the tilize scratch CBs holding pushed-but-never-consumed pages, NoC
// sanitizer clean) because `tilize_init` reprograms the unpacker MOP and
// `tilize_uninit` does not put the matmul MOP back -- so the `matmul_tiles`
// runs the *tilize* MOP, which fills SrcB from an UNPACR_NOP zero rather than
// unpacking operand A, and leaves the Src dvalid handshake out of step with
// what the math thread's matmul expects.
//
// Two run modes, same binary:
//   (no argument)  the correct sequence -- `matmul_init` re-issued after the
//                  tilize phase, as tt-metal's own
//                  tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block.cpp
//                  does. This is what `optests/diff.sh tilizematmul` runs, and
//                  it is the false-positive guard: a correct tilize+matmul
//                  kernel must stay green. Exit 0, exact, on both arches.
//   `buggy`        the reported bug. tt-sim reaches the same wedge silicon
//                  does: **wrong numbers** on Blackhole (1024/1024 elements
//                  wrong, self-check fails, exit 1) and a **silent hang** on
//                  Wormhole (no self-check line, timeout). The vendor
//                  reference sim computes a (wrong but quiet) answer.
//
//                  There is deliberately *no* tt-sim check that names the
//                  cause: "the unpacker is in tilize mode" is not expressible
//                  against the hardware configuration registers, which are
//                  byte-identical at the matmul's UNPACRs between the two
//                  forms (measured; the difference is the leftover MOP/replay
//                  template, which is code, not configuration). The
//                  source-level invariant lives one level up, in the LLK
//                  contract sanitizer -- see the upstream issue on
//                  `llk_unpack_tilize.h` carrying no `llk::san`
//                  instrumentation.
//
// Operands: A is the 32x32 identity and B is 1024 distinct exactly-representable
// bfloat16 values, both delivered row-major and tilized on device. C = A @ B is
// exactly B, so a mis-tilized operand or a mis-unpacked matmul operand is an
// unmistakable wrong value rather than a rounding question.
//
// Runs on both architectures. (It once needed `TT_SIM_ARCH=wormhole`, because
// the Blackhole tilize pack MOP drives packers 2/3 off THCON_SEC1_REG1 and
// tt-sim's packer raised on it; that is fixed -- see optests/tilize.)

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

// Tilized index of row-major element (r, c) within one tile: four 16x16 faces
// in the order (top-left, top-right, bottom-left, bottom-right), each face
// row-major.
static uint32_t tilized_index(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE_DIM) * 2 + (c / FACE_DIM);
    return face * FACE_DIM * FACE_DIM + (r % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// 1024 distinct values, every one exactly representable in bfloat16 (7 explicit
// mantissa bits): 2^(k-4) * (1 + m/128) for k = j/128 in [0,8), m = j%128.
static float ramp(uint32_t j) {
    return std::ldexp(1.0f + static_cast<float>(j % 128) / 128.0f, static_cast<int>(j / 128) - 4);
}

int main(int argc, char** argv) {
    const bool buggy = (argc > 1 && std::strcmp(argv[1], "buggy") == 0);
    printf("tilize+matmul: matmul_init %s re-issued after the tilize phase\n", buggy ? "NOT" : "IS");

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    auto make_buffer = [&](uint32_t bytes) {
        InterleavedBufferConfig config{
            .device = device, .size = bytes, .page_size = bytes, .buffer_type = BufferType::DRAM};
        return CreateBuffer(config);
    };
    auto a_dram = make_buffer(TILE_BYTES);
    auto b_dram = make_buffer(TILE_BYTES);
    auto dst_dram = make_buffer(TILE_BYTES);

    // Everything is Float16_b, so the pipeline is lossless end to end.
    auto make_cb = [&](CBIndex idx, uint32_t pages) {
        CircularBufferConfig cfg =
            CircularBufferConfig(pages * TILE_BYTES, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, TILE_BYTES);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 2);   // A, row-major
    make_cb(CBIndex::c_1, 2);   // B, row-major
    make_cb(CBIndex::c_4, 2);   // A, tilized scratch
    make_cb(CBIndex::c_5, 2);   // B, tilized scratch
    make_cb(CBIndex::c_16, 2);  // output

    std::vector<bfloat16> a_vec(TILE_ELEMS, bfloat16(0.0f));
    std::vector<bfloat16> b_vec(TILE_ELEMS);
    std::vector<bfloat16> golden(TILE_ELEMS);  // tiled layout, as packed out
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        a_vec[r * TILE_DIM + r] = bfloat16(1.0f);  // A = identity, row-major
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const bfloat16 value = bfloat16(ramp(r * TILE_DIM + c));
            b_vec[r * TILE_DIM + c] = value;              // B, row-major
            golden[tilized_index(r, c)] = value;          // C = A @ B = B, tiled
        }
    }
    tt::tt_metal::detail::WriteToBuffer(a_dram, a_vec);
    tt::tt_metal::detail::WriteToBuffer(b_dram, b_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {a_dram->address(), b_dram->address()});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address()});

    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .defines = {{"REISSUE_MATMUL_INIT", buggy ? "0" : "1"}}});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");

    uint32_t errors = 0;
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const uint32_t j = tilized_index(r, c);
            const float got = static_cast<float>(out_vec[j]);
            const float want = static_cast<float>(golden[j]);
            if (got != want) {
                if (errors == 0) {
                    printf("first mismatch at (%u, %u): got %g want %g\n", r, c, got, want);
                }
                errors++;
            }
        }
    }
    if (errors != 0) {
        printf("Failed on the device, %u/%u elements wrong\n", errors, TILE_ELEMS);
        return 1;
    }
    printf("Completed successfully on the device, C = A @ B over on-device tilized operands\n");
    return 0;
}
