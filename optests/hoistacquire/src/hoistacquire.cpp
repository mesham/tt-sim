// Op test for a `tile_regs_acquire()` hoisted out of the output-tile loop.
//
// Reported by the hedgehope compiler team: their bf16 GEMM passes on a
// Wormhole n300 at every core count and on tt-sim at 1 and 2 cores, but fails
// on tt-sim at 4 cores. The only difference between their passing 2-core
// codegen and their failing 4-core codegen is the loop nest -- at 2 cores the
// DST acquire/commit/release cycle sits *inside* the output-tile loop, at 4
// cores `matmul_init` and `tile_regs_acquire` are hoisted *outside* it. This
// program isolates that one variable.
//
// The compute kernel runs the same matmul-and-pack loop twice over the same
// resident operands:
//
//   PLAIN    acquire / matmul K-loop / commit / wait / pack / release, per tile
//   HOISTED  one acquire above the loop; the body keeps commit / wait / pack /
//            release per tile
//
// Both loops must produce the same NUM_OUT output tiles. The dump is
// PLAIN's NUM_OUT tiles followed by HOISTED's, so "the two halves differ" is
// the whole test, on top of the golden check.
//
// A[i][k] = scale[i][k] * identity and B is a single ramp tile, so output tile
// i is exactly (sum_k scale[i][k]) * B. Every scale is a (signed) power of two
// and so is every row sum, so each per-matmul product and the accumulation of
// them are exact in bfloat16 even though DEST holds Float16_b here. The four
// row sums are 2, 4, 8, 16, 32, 64 -- distinct, so a stale or duplicated DST
// half is visible, and B's 1024 values are all distinct, so a partially written
// tile is visible too.
//
// NUM_OUT is 6 rather than 4 on purpose. DEST is double-banked and the pack
// thread blocks on a 2-page output CB, so even a fully stalled writer only lets
// the math thread run two output tiles ahead of the packer at NUM_OUT = 4 --
// exactly the two banks, i.e. still safe. Only from the fifth tile can math
// wrap onto a bank the packer has not drained, so NUM_OUT = 4 would have made
// the `stall` mode below prove nothing.
//
// Modes, same binary:
//   (no argument)  both loops -- what `optests/diff.sh hoistacquire` runs
//   `plain`        only the per-tile acquire loop
//   `hoisted`      only the hoisted-acquire loop
//   `stall <n>`    both loops, with the writer spinning <n> iterations before
//                  each tile it drains. That back-pressures the pack thread
//                  through the 2-page output CB and so opens a wide,
//                  deterministic window for the math thread to run ahead --
//                  the hazard the hoisted acquire removes the guard against.
//                  It answers "is the window real, or did the loop simply
//                  never get far enough ahead for the acquire to matter?".
// The single-shape modes exist so that a hang in one can be attributed; the
// default is the one that matters, because it holds both shapes constant
// against each other inside one run.
//
// What it found (2026-08-20, tt-metal 0.74):
//
//   * With the packer keeping up -- the default run -- the acquire placement
//     makes no difference at all, on either architecture, and tt-sim is
//     bit-identical to ttsim over all 12288 elements. The hoisted acquire is
//     modelled correctly.
//   * Under back-pressure -- `stall 50` and up on Blackhole -- the hoisted loop
//     goes wrong, and takes the *previous* loop's last tile with it: with no
//     acquire to stall on, math wraps onto a DEST bank the packer has not
//     drained. The threshold is the same in both simulators (both clean at
//     `stall 20`, both changed at `stall 50`); what differs is the reaction.
//     tt-sim carries on and returns the corruption. ttsim stops with
//     `NonContractualBehavior: tensix_sempost: sem=2 sem_max=2`, because the
//     math thread has posted MATH_PACK past the max its SEMINIT declared.
//
//   * Under `TT_SIM_COST_MODEL=1` the default run goes wrong on its own, with
//     no artificial stall: the model's timing is enough to let math wrap. That
//     is why this op test has no frozen replay guard -- freezing it would put a
//     timing-dependent value into `driver/tests/cost_model_gate.py`, whose
//     whole claim is that the cost model changes no computed value.
//
// So the shape is a *race*, not a miscompilation: `tile_regs_acquire()` is the
// math thread's only back-pressure (a SEMWAIT that stalls while MATH_PACK is at
// max), and hoisting it out is safe exactly while the packer keeps up. The ISA
// documentation's functional model for SEMPOST saturates at 15 and never
// stalls, so hardware has no back-pressure to fall back on either -- a card
// that passes this shape is passing it on timing.

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

constexpr uint32_t NUM_OUT = 6;  // MUST match NUM_OUT in the compute kernel
constexpr uint32_t KT = 2;       // MUST match KT in the compute kernel

// A[i][k] = SCALE[i][k] * identity. Signed powers of two whose row sums are
// also powers of two, so every intermediate is exact in bfloat16.
//
// The two K terms of a row are deliberately *unequal*. A row of {1, 1} --
// x + x accumulated into a Float16_b DEST -- comes back one ulp high for every
// B value with an odd bfloat16 mantissa, identically on tt-sim and on ttsim
// (bit-for-bit), so it is a property of the accumulate rather than a simulator
// defect. Using it here would have put an unrelated 512-element rounding
// difference into every golden check.
constexpr float SCALE[NUM_OUT][KT] = {
    {4.0f, -2.0f},
    {8.0f, -4.0f},
    {16.0f, -8.0f},
    {32.0f, -16.0f},
    {64.0f, -32.0f},
    {128.0f, -64.0f}};

static uint32_t tilized_index(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE_DIM) * 2 + (c / FACE_DIM);
    return face * FACE_DIM * FACE_DIM + (r % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// (face, row, col) of a tilized element index, for reporting.
static void tile_position(uint32_t j, uint32_t& face, uint32_t& row, uint32_t& col) {
    face = j / (FACE_DIM * FACE_DIM);
    const uint32_t within = j % (FACE_DIM * FACE_DIM);
    row = (face / 2) * FACE_DIM + within / FACE_DIM;
    col = (face % 2) * FACE_DIM + within % FACE_DIM;
}

// 1024 distinct values, every one exactly representable in bfloat16.
static float ramp(uint32_t j) {
    return std::ldexp(1.0f + static_cast<float>(j % 128) / 128.0f, static_cast<int>(j / 128) - 4);
}

int main(int argc, char** argv) {
    const bool only_plain = (argc > 1 && std::strcmp(argv[1], "plain") == 0);
    const bool only_hoisted = (argc > 1 && std::strcmp(argv[1], "hoisted") == 0);
    uint32_t stall_iters = 0;
    if (argc > 1 && std::strcmp(argv[1], "stall") == 0) {
        stall_iters = (argc > 2) ? static_cast<uint32_t>(std::atoi(argv[2])) : 20000;
        printf("writer stalls %u iterations before draining the output CB\n", stall_iters);
    }
    const bool run_plain = !only_hoisted;
    const bool run_hoisted = !only_plain;
    const uint32_t num_shapes = (run_plain ? 1 : 0) + (run_hoisted ? 1 : 0);
    const uint32_t num_out_tiles = num_shapes * NUM_OUT;
    printf(
        "acquire placement: running%s%s (%u output tiles)\n",
        run_plain ? " PLAIN(acquire per tile)" : "",
        run_hoisted ? " HOISTED(one acquire above the loop)" : "",
        num_out_tiles);

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    auto make_buffer = [&](uint32_t bytes) {
        InterleavedBufferConfig config{
            .device = device, .size = bytes, .page_size = bytes, .buffer_type = BufferType::DRAM};
        return CreateBuffer(config);
    };
    auto a_dram = make_buffer(NUM_OUT * KT * TILE_BYTES);
    auto b_dram = make_buffer(KT * TILE_BYTES);
    auto dst_dram = make_buffer(num_out_tiles * TILE_BYTES);

    auto make_cb = [&](CBIndex idx, uint32_t pages) {
        CircularBufferConfig cfg =
            CircularBufferConfig(pages * TILE_BYTES, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, TILE_BYTES);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, NUM_OUT * KT);  // A, whole block resident
    make_cb(CBIndex::c_1, KT);            // B, whole block resident
    make_cb(CBIndex::c_16, 2);            // output, double buffered

    // A[i][k] = SCALE[i][k] * identity; B[k] = the same ramp tile for every k
    // (a K step's operand *values* are not what this test varies).
    std::vector<bfloat16> a_vec(NUM_OUT * KT * TILE_ELEMS, bfloat16(0.0f));
    std::vector<bfloat16> b_vec(KT * TILE_ELEMS);
    for (uint32_t i = 0; i < NUM_OUT; i++) {
        for (uint32_t k = 0; k < KT; k++) {
            for (uint32_t r = 0; r < TILE_DIM; r++) {
                a_vec[(i * KT + k) * TILE_ELEMS + tilized_index(r, r)] = bfloat16(SCALE[i][k]);
            }
        }
    }
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const bfloat16 value = bfloat16(ramp(r * TILE_DIM + c));
            for (uint32_t k = 0; k < KT; k++) {
                b_vec[k * TILE_ELEMS + tilized_index(r, c)] = value;
            }
        }
    }
    tt::tt_metal::detail::WriteToBuffer(a_dram, a_vec);
    tt::tt_metal::detail::WriteToBuffer(b_dram, b_vec);

    // Output tile i is (sum_k SCALE[i][k]) * B, in B's tilized layout.
    std::vector<bfloat16> golden(NUM_OUT * TILE_ELEMS);
    for (uint32_t i = 0; i < NUM_OUT; i++) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < KT; k++) {
            sum += SCALE[i][k];
        }
        for (uint32_t j = 0; j < TILE_ELEMS; j++) {
            golden[i * TILE_ELEMS + j] = bfloat16(sum * static_cast<float>(b_vec[j]));
        }
    }

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {a_dram->address(), NUM_OUT * KT, b_dram->address(), KT});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), num_out_tiles, stall_iters});

    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .defines = {{"RUN_PLAIN", run_plain ? "1" : "0"}, {"RUN_HOISTED", run_hoisted ? "1" : "0"}}});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(num_out_tiles * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");

    uint32_t total_errors = 0;
    for (uint32_t t = 0; t < num_out_tiles; t++) {
        const char* shape = (run_plain && t < NUM_OUT) ? "PLAIN  " : "HOISTED";
        const uint32_t i = t % NUM_OUT;
        uint32_t errors = 0;
        uint32_t zeros = 0;
        uint32_t face_zeros[4] = {0, 0, 0, 0};
        uint32_t first_bad = 0;
        for (uint32_t j = 0; j < TILE_ELEMS; j++) {
            const float got = static_cast<float>(out_vec[t * TILE_ELEMS + j]);
            const float want = static_cast<float>(golden[i * TILE_ELEMS + j]);
            if (got != want) {
                if (errors == 0) {
                    first_bad = j;
                }
                errors++;
            }
            if (got == 0.0f) {
                uint32_t face, row, col;
                tile_position(j, face, row, col);
                zeros++;
                face_zeros[face]++;
            }
        }
        printf("%s tile %u: %u/%u errors, %u zeros (per face %u/%u/%u/%u)", shape, i, errors, TILE_ELEMS, zeros,
               face_zeros[0], face_zeros[1], face_zeros[2], face_zeros[3]);
        if (errors != 0) {
            uint32_t face, row, col;
            tile_position(first_bad, face, row, col);
            printf(
                "; first at face %u (%u, %u): got %g want %g",
                face, row, col,
                static_cast<float>(out_vec[t * TILE_ELEMS + first_bad]),
                static_cast<float>(golden[i * TILE_ELEMS + first_bad]));
        }
        printf("\n");
        total_errors += errors;
    }

    if (run_plain && run_hoisted) {
        uint32_t halves_differ = 0;
        for (uint32_t j = 0; j < NUM_OUT * TILE_ELEMS; j++) {
            if (static_cast<float>(out_vec[j]) != static_cast<float>(out_vec[NUM_OUT * TILE_ELEMS + j])) {
                halves_differ++;
            }
        }
        printf(
            "PLAIN vs HOISTED: %u/%u elements differ -- %s\n",
            halves_differ, NUM_OUT * TILE_ELEMS,
            halves_differ == 0 ? "the acquire placement made no difference" : "THE ACQUIRE PLACEMENT CHANGED THE RESULT");
    }

    if (total_errors != 0) {
        printf("Failed on the device, %u errors over %u output tiles\n", total_errors, num_out_tiles);
        return 1;
    }
    printf("Completed successfully on the device, with %u output tiles\n", num_out_tiles);
    return 0;
}
