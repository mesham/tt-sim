// mechbench -- the Tensix-bound program for rung 4's mechanism-attribution leg.
//
// A normal tt-metal program: one core, three kernels, a DRAM round trip, and a
// host-side check of the arithmetic. It exists to be run twice -- once on
// silicon under TT_METAL_PROFILE_PERF_COUNTERS, once against tt-sim with the
// same environment -- so that the *decomposition* of the compute core's span
// by stall mechanism can be compared, not just its total.
//
// Two arms, chosen by argv, differing in exactly one instruction:
//
//   elw   add_tiles     -- cheap Matrix-unit work per unpack pair, so the math
//                          thread waits on the unpackers (SrcA/SrcB VALID).
//   mm    matmul_tiles  -- expensive Matrix-unit work for byte-identical
//                          unpacker work, so the unpackers wait on the Matrix
//                          unit (SrcA/SrcB CLEAR).
//
// Everything else -- tile count, tile format, circular-buffer depth, NoC
// traffic, both data-movement kernels -- is identical between the arms. The
// pair is the point: a cycle model that is wrong in compensating directions can
// match one arm's total and one arm's interior, but not both arms' interiors,
// because the two arms load the same mechanisms in opposite directions.
//
// No profiler markers are placed by this program. The counters are started and
// stopped by tt-metal's own firmware (TRISC1 wraps the compute kernel, BRISC
// reads back afterwards), so the measured window contains no instrumentation at
// all. That is the whole reason this leg is worth more than a zone match.
//
// Usage:  mechbench [elw|mm] [tiles]        (defaults: elw 32)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t TILE_ELEMS = TILE_DIM * TILE_DIM;

// Small integers only. bfloat16 carries 8 mantissa bits, so every value and
// every partial sum below stays exact and the check can be a strict equality
// rather than a correlation -- a loose check on a benchmark whose job is to be
// re-run on two machines is an invitation to compare two broken runs.
float a_value(uint32_t tile, uint32_t r, uint32_t c) { return static_cast<float>((tile + r + 2 * c) % 4); }
float b_value(uint32_t tile, uint32_t r, uint32_t c) { return static_cast<float>((tile + 3 * r + c) % 3); }

}  // namespace

int main(int argc, char** argv) {
    std::string arm = argc > 1 ? argv[1] : "elw";
    uint32_t tiles = argc > 2 ? static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 32;
    if (arm != "elw" && arm != "mm") {
        fprintf(stderr, "mechbench: unknown arm '%s' (expected elw or mm)\n", arm.c_str());
        return 2;
    }
    if (tiles == 0) {
        fprintf(stderr, "mechbench: tiles must be >= 1\n");
        return 2;
    }
    const uint32_t mode = (arm == "mm") ? 1u : 0u;

    const uint32_t tile_bytes = sizeof(bfloat16) * TILE_ELEMS;
    const uint32_t buffer_bytes = tile_bytes * tiles;

    // Build host-side inputs and the golden, per tile, in row-major.
    std::vector<bfloat16> a_rows(static_cast<size_t>(tiles) * TILE_ELEMS);
    std::vector<bfloat16> b_rows(static_cast<size_t>(tiles) * TILE_ELEMS);
    std::vector<bfloat16> golden_rows(static_cast<size_t>(tiles) * TILE_ELEMS);
    for (uint32_t t = 0; t < tiles; t++) {
        const size_t base = static_cast<size_t>(t) * TILE_ELEMS;
        for (uint32_t r = 0; r < TILE_DIM; r++) {
            for (uint32_t c = 0; c < TILE_DIM; c++) {
                a_rows[base + r * TILE_DIM + c] = bfloat16(a_value(t, r, c));
                b_rows[base + r * TILE_DIM + c] = bfloat16(b_value(t, r, c));
            }
        }
        for (uint32_t r = 0; r < TILE_DIM; r++) {
            for (uint32_t c = 0; c < TILE_DIM; c++) {
                float v = 0.f;
                if (mode == 1) {
                    for (uint32_t k = 0; k < TILE_DIM; k++) {
                        v += a_value(t, r, k) * b_value(t, k, c);
                    }
                } else {
                    v = a_value(t, r, c) + b_value(t, r, c);
                }
                golden_rows[base + r * TILE_DIM + c] = bfloat16(v);
            }
        }
    }

    // Tilize each 32x32 tile independently and concatenate; the kernels work
    // one tile at a time and never relate one tile to another.
    auto tilize_per_tile = [&](const std::vector<bfloat16>& rows) {
        std::vector<bfloat16> out;
        out.reserve(rows.size());
        for (uint32_t t = 0; t < tiles; t++) {
            std::vector<bfloat16> one(
                rows.begin() + static_cast<long>(t) * TILE_ELEMS, rows.begin() + static_cast<long>(t + 1) * TILE_ELEMS);
            std::vector<bfloat16> tiled = tilize_nfaces(one, TILE_DIM, TILE_DIM);
            out.insert(out.end(), tiled.begin(), tiled.end());
        }
        return out;
    };
    std::vector<bfloat16> a_tiled = tilize_per_tile(a_rows);
    std::vector<bfloat16> b_tiled = tilize_per_tile(b_rows);

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    // One page per buffer, deliberately: the reader addresses DRAM with
    // get_noc_addr_from_bank_id(0, base) + i * tile_bytes, which is only the
    // right address if every tile lives in the same bank. A multi-page
    // interleaved buffer would spread pages across banks and the reader would
    // silently fetch someone else's data on a card with more than one bank.
    InterleavedBufferConfig in_config{
        .device = device, .size = buffer_bytes, .page_size = buffer_bytes, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> src0 = CreateBuffer(in_config);
    std::shared_ptr<Buffer> src1 = CreateBuffer(in_config);
    std::shared_ptr<Buffer> dst = CreateBuffer(in_config);

    // Depth 2 on every circular buffer. Depth 1 would serialise the three
    // Tensix threads into lockstep and the only stall mechanism left would be
    // the circular buffers themselves; depth 2 lets the unpacker run ahead,
    // which is what puts the Src-ownership hand-off on the critical path.
    tt::DataFormat fmt = tt::DataFormat::Float16_b;
    constexpr uint32_t cb_depth = 2;
    for (auto idx : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_16}) {
        CircularBufferConfig cfg =
            CircularBufferConfig(cb_depth * tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    }

    tt::tt_metal::detail::WriteToBuffer(src0, a_tiled);
    tt::tt_metal::detail::WriteToBuffer(src1, b_tiled);

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle compute = CreateKernel(
        program,
        "kernels/compute/mech_kernel.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::LoFi, .compile_args = {mode}});

    SetRuntimeArgs(program, reader, core, {src0->address(), src1->address(), tiles, tile_bytes});
    SetRuntimeArgs(program, writer, core, {dst->address(), tiles, tile_bytes});
    SetRuntimeArgs(program, compute, core, {tiles});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> result(static_cast<size_t>(tiles) * TILE_ELEMS, bfloat16(0.f));
    tt::tt_metal::detail::ReadFromBuffer(dst, result);
    CloseDevice(device);

    // Untilize per tile and compare against the golden.
    size_t mismatches = 0;
    float worst = 0.f;
    for (uint32_t t = 0; t < tiles; t++) {
        std::vector<bfloat16> one(
            result.begin() + static_cast<long>(t) * TILE_ELEMS, result.begin() + static_cast<long>(t + 1) * TILE_ELEMS);
        std::vector<bfloat16> rows = untilize_nfaces(one, TILE_DIM, TILE_DIM);
        for (uint32_t i = 0; i < TILE_ELEMS; i++) {
            float got = static_cast<float>(rows[i]);
            float want = static_cast<float>(golden_rows[static_cast<size_t>(t) * TILE_ELEMS + i]);
            float err = std::fabs(got - want);
            worst = std::max(worst, err);
            if (err > 0.5f) {
                mismatches++;
            }
        }
    }

    printf("mechbench arm=%s tiles=%u tile_bytes=%u worst_abs_err=%g mismatches=%zu\n",
           arm.c_str(), tiles, tile_bytes, worst, mismatches);
    if (mismatches == 0) {
        printf("Completed successfully on the device, with %zu elements\n", result.size());
        return 0;
    }
    printf("Failure on the device, %zu mismatched elements of %zu\n", mismatches, result.size());
    return 1;
}
