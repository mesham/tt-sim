// Op test for the element-wise binary FPU ops (host side). Feeds two tiles of
// pseudo-random data through mul_tiles / add_tiles / sub_tiles (see
// kernels/compute/compute_kernel.cpp), one output tile per op, once per
// (CB data format, math fidelity) pair. Every output tile is read back and
// dumped as one `OPDIFF_RESULT:<hex>`; optests/diff.sh runs this same binary
// on tt-sim and on ttsim and compares the dumps. ttsim is the oracle — there
// is no local golden.
//
// Why the sweep: ELWMUL's product is built from mantissa slices, one slice
// pair per fidelity phase, and the compiler team's Gauss-Seidel (fp32 CBs,
// fp32 Dst, HiFi4) showed tt-sim contributing only phase 0 -- every operand
// needing more than the phase-0 bits came out low. The fp32 arm here is that
// configuration; the bf16 arm is the 16-bit-Dst rounding of the same ops.
//
// Layout of the dump, in launch order: for each format in {Float32,
// Float16_b}, for each fidelity in {LoFi, HiFi2, HiFi3, HiFi4}, the three op
// tiles (mul, add, sub). Float32 elements are 8 hex digits, bf16 4.

#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t TILE_ELEMS = 32 * 32;
constexpr uint32_t NUM_OPS = 3;  // MUST match the compute kernel's op count

// Operand values: a fixed LCG over sign, an exponent in [2^-6, 2^6] and a full
// 23-bit mantissa, so the TF32 narrowing into Src, every fidelity slice and
// the round into Dst all see bits that matter. One element in sixteen is zero
// on each side, to cover the zero-operand path of the multiply.
static std::vector<uint32_t> make_operand(uint32_t seed) {
    std::vector<uint32_t> bits(TILE_ELEMS);
    uint32_t state = seed;
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        state = state * 1664525u + 1013904223u;
        uint32_t sign = (state >> 31) & 1;
        uint32_t exp = 127 - 6 + ((state >> 24) % 13);
        uint32_t man = state & 0x7FFFFF;
        bits[i] = ((state >> 20) & 0xF) == 0 ? 0 : (sign << 31) | (exp << 23) | man;
    }
    return bits;
}

struct Arm {
    tt::DataFormat format;
    bool fp32_dest_acc;
    uint32_t elem_bytes;
};

static void run_arm(IDevice* device, const Arm& arm, tt::tt_metal::MathFidelity fidelity, std::string& dump) {
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};
    const uint32_t tile_bytes = arm.elem_bytes * TILE_ELEMS;

    InterleavedBufferConfig tile_config{
        .device = device, .size = tile_bytes, .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_config{
        .device = device,
        .size = NUM_OPS * tile_bytes,
        .page_size = NUM_OPS * tile_bytes,
        .buffer_type = BufferType::DRAM};
    auto a_dram = CreateBuffer(tile_config);
    auto b_dram = CreateBuffer(tile_config);
    auto dst_dram = CreateBuffer(out_config);

    for (auto [index, cb] : {std::pair{CBIndex::c_0, a_dram}, {CBIndex::c_1, b_dram}, {CBIndex::c_16, dst_dram}}) {
        (void)cb;
        CircularBufferConfig cb_config =
            CircularBufferConfig(2 * tile_bytes, {{index, arm.format}}).set_page_size(index, tile_bytes);
        tt_metal::CreateCircularBuffer(program, core, cb_config);
    }

    // The same values feed both arms; the bf16 arm keeps their high halves.
    std::vector<uint32_t> a_bits = make_operand(0x5EED0001u);
    std::vector<uint32_t> b_bits = make_operand(0x5EED0002u);
    if (arm.elem_bytes == 4) {
        tt::tt_metal::detail::WriteToBuffer(a_dram, a_bits);
        tt::tt_metal::detail::WriteToBuffer(b_dram, b_bits);
    } else {
        std::vector<uint16_t> a16(TILE_ELEMS), b16(TILE_ELEMS);
        for (uint32_t i = 0; i < TILE_ELEMS; i++) {
            a16[i] = static_cast<uint16_t>(a_bits[i] >> 16);
            b16[i] = static_cast<uint16_t>(b_bits[i] >> 16);
        }
        tt::tt_metal::detail::WriteToBuffer(a_dram, a16);
        tt::tt_metal::detail::WriteToBuffer(b_dram, b16);
    }

    KernelHandle reader = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, reader, core, {a_dram->address(), b_dram->address(), tile_bytes});

    KernelHandle writer = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, writer, core, {dst_dram->address(), NUM_OPS, tile_bytes});

    CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{.math_fidelity = fidelity, .fp32_dest_acc_en = arm.fp32_dest_acc});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    char buf[16];
    if (arm.elem_bytes == 4) {
        std::vector<uint32_t> out(NUM_OPS * TILE_ELEMS);
        tt::tt_metal::detail::ReadFromBuffer(dst_dram, out);
        for (uint32_t v : out) {
            snprintf(buf, sizeof buf, "%08x", v);
            dump += buf;
        }
    } else {
        std::vector<uint16_t> out(NUM_OPS * TILE_ELEMS);
        tt::tt_metal::detail::ReadFromBuffer(dst_dram, out);
        for (uint16_t v : out) {
            snprintf(buf, sizeof buf, "%04x", v);
            dump += buf;
        }
    }
}

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);
    std::string dump;

    const Arm arms[] = {
        {tt::DataFormat::Float32, true, 4},
        {tt::DataFormat::Float16_b, false, 2},
    };
    const tt::tt_metal::MathFidelity fidelities[] = {
        tt::tt_metal::MathFidelity::LoFi,
        tt::tt_metal::MathFidelity::HiFi2,
        tt::tt_metal::MathFidelity::HiFi3,
        tt::tt_metal::MathFidelity::HiFi4};
    uint32_t launches = 0;
    for (const Arm& arm : arms) {
        for (tt::tt_metal::MathFidelity fidelity : fidelities) {
            run_arm(device, arm, fidelity, dump);
            launches++;
        }
    }
    CloseDevice(device);

    printf("OPDIFF_RESULT:%s\n", dump.c_str());
    printf("Completed successfully on the device, with %u launches of %u op tiles\n", launches, NUM_OPS);
    return 0;
}
