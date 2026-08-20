// Op test for `pack_untilize_dest_init` issued after the math rather than
// before it -- the hedgehope compiler team's localised bf16 GEMM failure
// (2026-08-20).
//
// Their minimal case is one Tensix core, one output tile, K = 2, and the whole
// difference between pass and fail is where one call sits in the source:
//
//     -  pack_untilize_dest_init<1, 1>(cb_out);   // before matmul_init: PASSES
//        matmul_init(...);  tile_regs_acquire();
//        for (k = 0; k < 2; k++) matmul_tiles(..., 0);
//        tile_regs_commit();  tile_regs_wait();
//     +  pack_untilize_dest_init<1, 1>(cb_out);   // after tile_regs_wait: FAILS
//        pack_untilize_dest<1, 1>(cb_out, 1, 0);
//
// `pack_untilize_dest_init` compiles to nothing on the UNPACK and MATH builds
// of the kernel (see `tt_metal/hw/inc/api/compute/pack_untilize.h`: on Wormhole
// the body is three `PACK((...))` calls), so the move changes only the PACK
// thread's own instruction order relative to its own `tile_regs_wait`. Both
// orders are correct on an n300 card and both are correct on ttsim; only tt-sim
// separates them.
//
// Modes, same binary:
//   (no argument)  the late init -- the defect, and what
//                  `optests/diff.sh packuntilizeinit` runs
//   `early`        the same program with the init before `matmul_init`; the
//                  control, which must stay green
//   `k <n>`        set the number of accumulating `matmul_tiles` (default 2).
//                  `k 1` is the co-factor control: with one matmul into DEST
//                  the late init is reported harmless.
// `early` and `k` compose, e.g. `./build/packuntilizeinit early k 1`.
//
// A[k] = SCALE[k] * identity and B[k] is one ramp tile of 1024 distinct,
// exactly representable bfloat16 values, so the untilized 32x32 output is
// exactly (sum_k SCALE[k]) * B row-major. All the scales and their sum are
// powers of two, so nothing rounds; a wrong DEST row, half or read width shows
// up as an exact-value mismatch rather than as noise.
//
// The configuration matches theirs: Float16_b circular buffers with
// `fp32_dest_acc_en = true` (bf16 storage, 32-bit DEST), HiFi4.
//
// The on-device tilize is a *co-factor*, not decoration. With the operands
// taken already tiled from DRAM the late init is harmless at every K and every
// output-tile count tried; it is only with the operands tilized on device --
// which is what their generator emits -- that the two forms separate.
//
// Status when this was written, 2026-08-20, tt-metal 0.74:
//
//   arch       form     tt-sim            ttsim (oracle)
//   Wormhole   early    errors=0          errors=0
//   Wormhole   late     errors=232        errors=0        <-- the defect
//   Wormhole   late K=1 errors=0          errors=0
//   Blackhole  either   times out         UnimplementedFunctionality
//
// (Since the Wait Gate fix, Blackhole no longer times out: tt-sim reaches the
// `pack_untilize_dest` PACR and refuses it with a named `NotImplementedError`
// -- `DST_ACCESS_STRIDED_MODE` without the `DEST_ACCESS_CFG` remap -- where
// ttsim refuses the same instruction as `UnimplementedFunctionality`. Both
// simulators still decline the kernel, so Blackhole remains out of scope for
// this op test; it now declines loudly instead of hanging.)
//
// **Fixed the same day**: the Wait Gate let the `STALLWAIT` that opens
// `_llk_init_packer_dest_offset_registers_` walk past the still-unsatisfied
// `SEMWAIT` that `tile_regs_wait()` had latched, and *overwrite* it -- so the
// packer stopped waiting for MATH_PACK and packed a DEST one MVMUL short of
// its final value (first untilize PACR at cycle 6049, SEMPOST at 6108). `STALLWAIT.md`'s block-mask table ticks `STALLWAIT` in all
// nine columns; tt-sim caught it by its execution unit (Sync, bit B1) alone.
// Both forms now match ttsim at every K. This op test stays as the end-to-end
// regression; `tt_sim/pe/tensix/waitgate_stallwait_blocked_test.py` pins the
// mechanism without tt-metal or the oracle.
//
// ttsim passing both forms, on the same compiled kernels, is what made this a
// tt-sim defect rather than a kernel that is out of contract. (Blackhole is a
// separate matter: `pack_untilize_dest_init` takes the `llk_math_reconfig_remap`
// arm there, which ttsim declares unimplemented and which tt-sim does not
// finish -- neither simulator has an answer, so Blackhole is not evidence
// either way and this op test is a Wormhole one until that is untangled.)
//
// What the failure is *not*, measured rather than argued: the packer's latched
// state is bit-identical between the two forms -- `PCK_DEST_RD_CTRL_Read_32b_data`,
// the pack source format, the config `stateID` and every generated DEST address
// -- so the reported guess that a late init lands `PCK_DEST_RD_CTRL` after the
// packer sampled it is not what is happening. The math side's `useDst32b` is 1
// for all 160 MVMULs in both forms too. What differs is the *interleaving*: in
// the late form the packer's DEST reads begin about fifty DEST row-writes
// earlier in the combined sequence, and it reads datums whose mantissa field is
// doubled (`0x027c0000` where the correct accumulation gives `0x017c0000`) --
// i.e. one accumulation step's worth of the K loop is not in DEST yet, or not
// where the packer looked for it. That is consistent with the K >= 2 co-factor:
// with a single matmul there is nothing left to accumulate, so an early read
// still sees the final value.

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
constexpr uint32_t MAX_KT = 4;

// A[k] = SCALE[k] * identity. Signed powers of two, and every prefix sum is a
// power of two too, so the accumulation is exact whatever KT is.
constexpr float SCALE[MAX_KT] = {4.0f, -2.0f, -1.0f, -0.5f};

static uint32_t tilized_index(uint32_t r, uint32_t c) {
    const uint32_t face = (r / FACE_DIM) * 2 + (c / FACE_DIM);
    return face * FACE_DIM * FACE_DIM + (r % FACE_DIM) * FACE_DIM + (c % FACE_DIM);
}

// 1024 distinct values, every one exactly representable in bfloat16.
static float ramp(uint32_t j) {
    return std::ldexp(1.0f + static_cast<float>(j % 128) / 128.0f, static_cast<int>(j / 128) - 4);
}

int main(int argc, char** argv) {
    bool init_late = true;
    uint32_t kt = 2;
    uint32_t num_out = 1;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "early") == 0) {
            init_late = false;
        } else if (std::strcmp(argv[i], "k") == 0 && i + 1 < argc) {
            kt = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "n") == 0 && i + 1 < argc) {
            num_out = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
    }
    if (kt < 1 || kt > MAX_KT) {
        printf("k must be 1..%u\n", MAX_KT);
        return 2;
    }
    printf(
        "pack_untilize_dest_init %s the math, K = %u accumulating matmul_tiles, %u output tile(s)\n",
        init_late ? "AFTER (the reported defect)" : "before (control)", kt, num_out);

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    auto make_buffer = [&](uint32_t bytes) {
        InterleavedBufferConfig config{
            .device = device, .size = bytes, .page_size = bytes, .buffer_type = BufferType::DRAM};
        return CreateBuffer(config);
    };
    auto a_dram = make_buffer(kt * TILE_BYTES);
    auto b_dram = make_buffer(kt * TILE_BYTES);
    auto dst_dram = make_buffer(num_out * TILE_BYTES);

    auto make_cb = [&](CBIndex idx, uint32_t pages) {
        CircularBufferConfig cfg =
            CircularBufferConfig(pages * TILE_BYTES, {{idx, tt::DataFormat::Float16_b}}).set_page_size(idx, TILE_BYTES);
        tt_metal::CreateCircularBuffer(program, core, cfg);
    };
    make_cb(CBIndex::c_0, kt);   // A, row-major from DRAM
    make_cb(CBIndex::c_1, kt);   // B, row-major from DRAM
    make_cb(CBIndex::c_24, kt);  // A, tiled on device
    make_cb(CBIndex::c_25, kt);  // B, tiled on device
    make_cb(CBIndex::c_16, num_out);  // the untilized output tiles

    // A[k] = SCALE[k] * identity, B[k] = the same ramp tile for every k.
    std::vector<bfloat16> a_vec(kt * TILE_ELEMS, bfloat16(0.0f));
    std::vector<bfloat16> b_vec(kt * TILE_ELEMS);
    for (uint32_t k = 0; k < kt; k++) {
        for (uint32_t r = 0; r < TILE_DIM; r++) {
            a_vec[k * TILE_ELEMS + r * TILE_DIM + r] = bfloat16(SCALE[k]);
        }
    }
    // B is tilized in one `tilize_block(..., kt, ...)` call, which reads a
    // *block* of kt tiles: 32 rows of (32 * kt) datums, not kt separate tiles.
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const bfloat16 value = bfloat16(ramp(r * TILE_DIM + c));
            for (uint32_t k = 0; k < kt; k++) {
                b_vec[r * TILE_DIM * kt + k * TILE_DIM + c] = value;
            }
        }
    }
    tt::tt_metal::detail::WriteToBuffer(a_dram, a_vec);
    tt::tt_metal::detail::WriteToBuffer(b_dram, b_vec);

    float scale = 0.0f;
    for (uint32_t k = 0; k < kt; k++) {
        scale += SCALE[k];
    }
    // Golden: row-major 32x32, scale * B.
    std::vector<bfloat16> golden(TILE_ELEMS);
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            golden[r * TILE_DIM + c] = bfloat16(scale * ramp(r * TILE_DIM + c));
        }
    }

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {a_dram->address(), kt, b_dram->address(), kt});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), num_out});

    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .defines = {{"INIT_LATE", init_late ? "1" : "0"}, {"KT", std::to_string(kt)}, {"NUM_OUT", std::to_string(num_out)}}});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<bfloat16> out_vec(num_out * TILE_ELEMS);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram, out_vec);
    CloseDevice(device);

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%04x", static_cast<uint16_t>(std::bit_cast<uint32_t>(static_cast<float>(out_vec[i])) >> 16));
    }
    printf("\n");

    uint32_t errors = 0;
    uint32_t zeros = 0;
    uint32_t bad_rows[TILE_DIM] = {0};
    uint32_t bad_cols[TILE_DIM] = {0};
    for (uint32_t t = 0; t < num_out; t++) {
    uint32_t terr = 0;
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            const float got = static_cast<float>(out_vec[t * TILE_ELEMS + r * TILE_DIM + c]);
            const float want = static_cast<float>(golden[r * TILE_DIM + c]);
            if (got != want) {
                if (errors == 0) {
                    printf("first mismatch at tile %u row %u col %u: got %g want %g\n", t, r, c, got, want);
                }
                errors++;
                terr++;
                bad_rows[r]++;
                bad_cols[c]++;
            }
            if (got == 0.0f) {
                zeros++;
            }
        }
    }
    printf("output tile %u: errors=%u of %u\n", t, terr, TILE_ELEMS);
    }
    printf("errors=%u of %u zeros=%u\n", errors, num_out * TILE_ELEMS, zeros);
    if (errors != 0) {
        // Which rows and columns carry the damage: a wrong DEST row shows as a
        // row pattern, a wrong pack read width as a column pattern.
        printf("bad rows:");
        for (uint32_t r = 0; r < TILE_DIM; r++) {
            if (bad_rows[r]) {
                printf(" %u", r);
            }
        }
        printf("\nbad cols:");
        for (uint32_t c = 0; c < TILE_DIM; c++) {
            if (bad_cols[c]) {
                printf(" %u", c);
            }
        }
        printf("\nFailed on the device, pack_untilize_dest_init %s the math produced wrong results\n",
               init_late ? "after" : "before");
        return 1;
    }
    printf("Completed successfully on the device, with %u untilized output tile(s)\n", num_out);
    return 0;
}
