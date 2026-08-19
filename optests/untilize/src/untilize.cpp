// Op test for the row-major *untilize* pack (host side). Runs the three
// variants the tt-xftn compiler team isolated in
// `docs/plans/gemm-ondevice-untilize-kernels.md`, all sharing one phase-1
// matmul -> DST -> tiled `pack_tile` into an intermediate CB, and differing
// only in how the intermediate tile reaches the output CB:
//
//   op 0  CONTROL  copy_tile -> DST -> *tiled* pack_tile   (their data point 4)
//   op 1           untilize_block CB->CB                   (their data point 2)
//   op 2           copy_tile -> DST -> pack_untilize_dest  (their data point 3)
//
// Ops 1 and 2 are row-major untilizes: they interleave the tile's four 16x16
// faces into output *rows*, so the pack Y (row) and Z (face) address counters
// must accumulate across the successive PACRs that pack one tile. A packer that
// overwrites those counters instead of accumulating lands only the first output
// row and zeros the other 31. Op 0 packs tiled -- each face at Y = 0, where set
// == accumulate -- so it is the built-in control: it must pass either way.
//
// The whole pipeline is exact, so unlike most op tests this one carries a
// *computed* golden rather than deferring to ttsim. The matmul's second operand
// is the identity tile, so C = A, and A is a ramp of 1024 distinct values each
// exactly representable in bfloat16; every stage (matmul, pack, copy_tile,
// untilize) is then a lossless permutation and the output must match
// bit-for-bit. The `OPDIFF_RESULT` dump still lets optests/diff.sh diff the raw
// bytes against ttsim.
//
// Add an op: append a phase in the compute kernel and bump NUM_OPS.
//
// Two run modes, same binary:
//   (no argument)  16-bit DEST, the default. All three ops pass on tt-sim and
//                  match ttsim; this is what `optests/diff.sh untilize` runs and
//                  what `driver/wormhole/server/untilize_replay_test.py` freezes.
//   `fp32`         `fp32_dest_acc_en` on, every CB still Float16_b -- a 32-bit
//                  DEST feeding a 16-bit output format. All three ops pass
//                  here too, and must keep passing: this arm is the regression
//                  test for the pack out of a 32-bit DEST. `UNTILIZE_FP32=1`
//                  in the environment selects the same arm, which is how
//                  `optests/diff.sh` -- it passes no arguments -- runs and
//                  records it; the recorded wire trace is frozen per
//                  architecture as `traces/untilize_fp32.trace` and replayed
//                  by `driver/<arch>/server/untilize_fp32_replay_test.py`.
//
//                  What it caught. tt-sim used to take the DEST read width
//                  from the pack source format, so with a 16-bit format it
//                  read DEST 16 bits at a time and most of the output was
//                  never written -- 896, 960, 896 elements of 1024 wrong on
//                  Wormhole and 896, 960, 960 on Blackhole, op 0 landing only
//                  the top-left 8 rows x 16 columns. That op 0 -- the *tiled
//                  control* -- failed too was the point: the fault was not in
//                  untilize but in the pack, which every op here shares. The
//                  width is `PCK_DEST_RD_CTRL_Read_32b_data`, which tt-metal
//                  sets to `is_32b_format || is_fp32_dest_acc_en` -- so a
//                  16-bit format does not imply a 16-bit DEST read.
//                  Reported by the tt-xftn compiler team as `gemm_bf16_check`
//                  giving `errors=4096 of 4096` on tt-sim Wormhole while both
//                  cards passed it, from a `ComputeConfig{.fp32_dest_acc_en
//                  = true}` over Float16_b CBs.
//
//                  Nothing in the tree reached this before: every other tt-sim
//                  program that sets `fp32_dest_acc_en` (optests/transpose,
//                  optests/sfpumath, examples/five-fp) makes its CBs Float32, so
//                  DEST and the output format are both 32-bit and the
//                  conversion this exercises never happens. The unit-level
//                  pin is tt_sim/pe/tensix/pack_dest_rd_ctrl_test.py.

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
constexpr uint32_t NUM_OPS = 3;  // MUST match the compute kernel's op count
// Op 0 packs tiled, ops 1 and 2 pack row-major (see the header comment).
constexpr bool OP_IS_TILED[NUM_OPS] = {true, false, false};
constexpr const char* OP_NAME[NUM_OPS] = {"pack_tile (tiled control)", "untilize_block", "pack_untilize_dest"};

// Tilized index of row-major element (r, c): four 16x16 faces in the order
// (top-left, top-right, bottom-left, bottom-right), each face row-major.
static uint32_t tilized_index(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE_DIM) * 2 + (c / FACE_DIM);
    return face * FACE_DIM * FACE_DIM + (r % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// 1024 distinct values, every one exactly representable in bfloat16 (7 explicit
// mantissa bits): 2^(k-4) * (1 + m/128) for k = j/128 in [0,8), m = j%128. Both
// the value and its magnitude are strictly monotone in j, so a wrong output row
// -- or a zeroed one -- is unmistakable in the hex dump.
static float ramp(uint32_t j) {
    return std::ldexp(1.0f + static_cast<float>(j % 128) / 128.0f, static_cast<int>(j / 128) - 4);
}

int main(int argc, char** argv) {
    // `fp32` selects a 32-bit DEST (`fp32_dest_acc_en`) while leaving every CB
    // Float16_b -- see the header comment. The golden is unchanged: bf16
    // operands accumulated in fp32 and packed back to bf16 is still exact.
    //
    // `UNTILIZE_FP32=1` in the environment selects the same arm, because
    // `optests/diff.sh` runs `./build/<name>` with no arguments -- an env var
    // is the only handle it, and the trace capture that piggybacks on it
    // (`TT_SIM_RECORD=...`), has on the arm.
    const char* fp32_env = std::getenv("UNTILIZE_FP32");
    const bool fp32_dest = (argc > 1 && std::strcmp(argv[1], "fp32") == 0) ||
                           (fp32_env != nullptr && fp32_env[0] != '\0' && std::strcmp(fp32_env, "0") != 0);
    printf("dest accumulate: %s\n", fp32_dest ? "fp32 (32-bit DEST, Float16_b CBs)" : "fp16 (default)");

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMS;

    InterleavedBufferConfig tile_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = NUM_OPS * tile_bytes,
        .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto in0_dram = CreateBuffer(tile_config);
    auto in1_dram = CreateBuffer(tile_config);
    auto dst_dram = CreateBuffer(out_config);

    // Two operand CBs, the intermediate CB the phase-1 matmul packs into, and
    // the output CB. Everything is Float16_b: the compiler team's intermediate
    // CB was Float32, but a uniform format keeps the pipeline lossless (so the
    // golden below can be exact) and avoids dragging pack/unpack data-format
    // reconfiguration into a test about address counters.
    auto make_cb = [&](CBIndex idx, uint32_t pages) {
        CircularBufferConfig cfg =
            CircularBufferConfig(pages * tile_bytes, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, tile_bytes);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 2);
    make_cb(CBIndex::c_1, 2);
    make_cb(CBIndex::c_24, 2);  // intermediate
    make_cb(CBIndex::c_16, 2);  // output

    // in0 = the ramp, in1 = the identity tile, so C = in0 * in1 = in0 exactly:
    // each output element is one product with 1.0 plus 31 products with 0.0.
    std::vector<bfloat16> golden(TILE_ELEMS);
    std::vector<bfloat16> in0_vec(TILE_ELEMS);
    std::vector<bfloat16> in1_vec(TILE_ELEMS, bfloat16(0.0f));
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const uint32_t j = r * TILE_DIM + c;
            golden[j] = bfloat16(ramp(j));
            in0_vec[tilized_index(r, c)] = golden[j];
        }
        in1_vec[tilized_index(r, r)] = bfloat16(1.0f);
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
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = fp32_dest});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(NUM_OPS * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");

    // Every op must reproduce the ramp bit-for-bit. The tiled control is
    // untilized on the host first (the compiler team's data point 4); the two
    // row-major ops are already in row-major order. Report the per-op error
    // count and the first output row that came back entirely zero -- the
    // signature of pack counters that do not advance across a tile's PACRs.
    uint32_t total_errors = 0;
    for (uint32_t op = 0; op < NUM_OPS; op++) {
        const bfloat16* got_tile = &out_vec[op * TILE_ELEMS];
        uint32_t errors = 0;
        int first_zero_row = -1;
        for (uint32_t r = 0; r < TILE_DIM; r++) {
            bool row_all_zero = true;
            for (uint32_t c = 0; c < TILE_DIM; c++) {
                const uint32_t j = r * TILE_DIM + c;
                const float got = static_cast<float>(got_tile[OP_IS_TILED[op] ? tilized_index(r, c) : j]);
                row_all_zero &= (got == 0.0f);
                if (got != static_cast<float>(golden[j])) {
                    if (errors == 0) {
                        printf(
                            "op %u (%s): first mismatch at (%u, %u): got %f want %f\n",
                            op,
                            OP_NAME[op],
                            r,
                            c,
                            got,
                            static_cast<float>(golden[j]));
                    }
                    errors++;
                }
            }
            if (row_all_zero && first_zero_row < 0) {
                first_zero_row = static_cast<int>(r);
            }
        }
        printf(
            "op %u (%s): %u/%u errors, first all-zero output row %d\n", op, OP_NAME[op], errors, TILE_ELEMS,
            first_zero_row);
        total_errors += errors;
    }

    if (total_errors != 0) {
        printf("Failed on the device, %u errors over %u op tiles\n", total_errors, NUM_OPS);
        return 1;
    }
    printf("Completed successfully on the device, with %u op tiles\n", NUM_OPS);
    return 0;
}
