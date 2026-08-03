// Op test for the row-major *tilize* unpack (host side). Tilize is the mirror
// of optests/untilize: it reads a **row-major** block out of L1 and lands it in
// SrcA as tiled faces, which the unpacker does by setting `Tileize_mode` and
// walking L1 discontiguously -- `UnpackRowWidth` datums, then a jump of
// `RowStride` bytes (UNPACR_Regular.md). The whole point of this program is to
// exercise that strided walk from a real `tilize_block`.
//
//   op 0  CONTROL  copy_tile of an already-tiled tile -> pack_tile
//   op 1           tilize_block(block = 1), 32x32 row-major -> 1 tile
//   op 2           tilize_block(block = 2), 32x64 row-major -> 2 tiles
//
// Why two block widths: the LLK sets `RowStride = 2 * 32 * block` bytes for
// bfloat16, and the *contiguous* stride is `DatumSizeBytes * UnpackRowWidth`,
// which is 32 bytes on Wormhole (row width 16) but 64 on Blackhole (row width
// 32 above one byte per datum). So op 1 is strided on Wormhole and exactly
// contiguous on Blackhole, and only op 2 (stride 128) is strided on both. A
// simulator that ignored RowStride would still pass op 1 on Blackhole.
//
// Op 0 shares neither the tilize unpack nor its addressing, so it isolates a
// fault to the tilize path rather than to copy_tile, the CBs or the pack.
//
// Status: PASSes on Wormhole against ttsim (`TT_SIM_ARCH=wormhole
// ./optests/diff.sh tilize`) and is frozen as
// driver/wormhole/server/tilize_replay_test.py. On Blackhole the strided unpack
// runs, but the *pack* that follows it does not: BH's tilize pack MOP issues
// PACRs with PACK_INTF_SEL 0b0101/0b1010, so packers 2/3 run off THCON_SEC1_REG1
// — which tt-metal never writes — and tt-sim's packer reads its zero-compress
// bit as "compressing" and raises. ttsim models no pack-side compression at all.
//
// The pipeline is exact, so like optests/untilize this carries a *computed*
// golden rather than deferring to ttsim: every input value is a distinct
// bfloat16 and every stage (unpack, datacopy, pack) is a lossless permutation,
// so a datum read from the wrong L1 address is an unmistakable wrong value
// rather than a rounding question. The `OPDIFF_RESULT` dump still lets
// optests/diff.sh diff the raw bytes against ttsim.

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
constexpr uint32_t WIDE_TILES = 2;         // op 2's block width, in tiles
constexpr uint32_t OUT_TILES = 1 + 1 + WIDE_TILES;
constexpr uint32_t NUM_OPS = 3;            // MUST match the compute kernel's op count
constexpr const char* OP_NAME[NUM_OPS] = {"copy_tile (tiled control)", "tilize_block x1", "tilize_block x2"};
// Where each op's output tiles start in the output buffer, and how many.
constexpr uint32_t OP_FIRST_TILE[NUM_OPS] = {0, 1, 2};
constexpr uint32_t OP_NUM_TILES[NUM_OPS] = {1, 1, WIDE_TILES};

// Tilized index of row-major element (r, c) within one tile: four 16x16 faces
// in the order (top-left, top-right, bottom-left, bottom-right), each face
// row-major.
static uint32_t tilized_index(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE_DIM) * 2 + (c / FACE_DIM);
    return face * FACE_DIM * FACE_DIM + (r % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// 2048 distinct values, every one exactly representable in bfloat16 (7 explicit
// mantissa bits): 2^(k-8) * (1 + m/128) for k = j/128 in [0,16), m = j%128.
// Both the value and its magnitude are strictly monotone in j, so a datum
// fetched from the wrong row of L1 is unmistakable in the hex dump.
static float ramp(uint32_t j) {
    return std::ldexp(1.0f + static_cast<float>(j % 128) / 128.0f, static_cast<int>(j / 128) - 8);
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
    auto tiled_dram = make_buffer(TILE_BYTES);
    auto rm1_dram = make_buffer(TILE_BYTES);
    auto rm2_dram = make_buffer(WIDE_TILES * TILE_BYTES);
    auto dst_dram = make_buffer(OUT_TILES * TILE_BYTES);

    // Everything is Float16_b, so the pipeline is lossless end to end and the
    // golden below can be exact.
    auto make_cb = [&](CBIndex idx, uint32_t pages) {
        CircularBufferConfig cfg =
            CircularBufferConfig(pages * TILE_BYTES, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, TILE_BYTES);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, 2);                // already-tiled input (op 0)
    make_cb(CBIndex::c_1, 2);                // 32x32 row-major input (op 1)
    make_cb(CBIndex::c_2, 2 * WIDE_TILES);   // 32x(32*WIDE_TILES) row-major input (op 2)
    make_cb(CBIndex::c_16, 2 * OUT_TILES);   // output

    // op 0 and op 1 carry the *same* logical 32x32 tile -- one already tiled,
    // one row-major -- so the control and the tilize must produce identical
    // output tiles. op 2 carries a 32x(32*WIDE_TILES) row-major block.
    std::vector<bfloat16> tiled_vec(TILE_ELEMS);
    std::vector<bfloat16> rm1_vec(TILE_ELEMS);
    std::vector<bfloat16> rm2_vec(WIDE_TILES * TILE_ELEMS);
    std::vector<bfloat16> golden(OUT_TILES * TILE_ELEMS);  // tiled layout, as packed out
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const bfloat16 value = bfloat16(ramp(r * TILE_DIM + c));
            rm1_vec[r * TILE_DIM + c] = value;
            tiled_vec[tilized_index(r, c)] = value;
            golden[0 * TILE_ELEMS + tilized_index(r, c)] = value;  // op 0
            golden[1 * TILE_ELEMS + tilized_index(r, c)] = value;  // op 1
        }
    }
    const uint32_t wide_cols = TILE_DIM * WIDE_TILES;
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < wide_cols; c++) {
            const bfloat16 value = bfloat16(ramp(r * wide_cols + c));
            rm2_vec[r * wide_cols + c] = value;  // row-major across the whole block
            golden[(2 + c / TILE_DIM) * TILE_ELEMS + tilized_index(r, c % TILE_DIM)] = value;
        }
    }
    tt::tt_metal::detail::WriteToBuffer(tiled_dram, tiled_vec);
    tt::tt_metal::detail::WriteToBuffer(rm1_dram, rm1_vec);
    tt::tt_metal::detail::WriteToBuffer(rm2_dram, rm2_vec);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {tiled_dram->address(), rm1_dram->address(), rm2_dram->address(), WIDE_TILES});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), OUT_TILES});

    CreateKernel(
        program, "kernels/compute/compute_kernel.cpp", core, ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(OUT_TILES * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");

    // Every op must reproduce its tiles bit-for-bit. Report the per-op error
    // count and the first mismatch, in row-major (r, c) terms -- which is where
    // a strided-walk bug shows up as a whole wrong input row.
    uint32_t total_errors = 0;
    for (uint32_t op = 0; op < NUM_OPS; op++) {
        uint32_t errors = 0;
        for (uint32_t t = 0; t < OP_NUM_TILES[op]; t++) {
            const uint32_t base = (OP_FIRST_TILE[op] + t) * TILE_ELEMS;
            for (uint32_t r = 0; r < TILE_DIM; r++) {
                for (uint32_t c = 0; c < TILE_DIM; c++) {
                    const uint32_t j = base + tilized_index(r, c);
                    const float got = static_cast<float>(out_vec[j]);
                    const float want = static_cast<float>(golden[j]);
                    if (got != want) {
                        if (errors == 0) {
                            printf(
                                "op %u (%s): first mismatch at tile %u (%u, %u): got %g want %g\n",
                                op,
                                OP_NAME[op],
                                t,
                                r,
                                c,
                                got,
                                want);
                        }
                        errors++;
                    }
                }
            }
        }
        printf("op %u (%s): %u/%u errors\n", op, OP_NAME[op], errors, OP_NUM_TILES[op] * TILE_ELEMS);
        total_errors += errors;
    }

    if (total_errors != 0) {
        printf("Failed on the device, %u errors over %u op tiles\n", total_errors, OUT_TILES);
        return 1;
    }
    printf("Completed successfully on the device, with %u op tiles\n", OUT_TILES);
    return 0;
}
