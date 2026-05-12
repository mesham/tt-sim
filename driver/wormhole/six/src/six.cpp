// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <random>
#include "host_api.hpp"
#include "device.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

// Pearson correlation coefficient between two bfloat16 vectors. Used to compare the
// device matmul output against the CPU golden — bfloat16 + HiFi4 won't be bit-exact,
// so we look for high correlation (close to 1.0) rather than exact equality.
float pearson_correlation(const std::vector<bfloat16>& a, const std::vector<bfloat16>& b) {
    const size_t n = a.size();
    double mean_a = 0;
    double mean_b = 0;
    for (size_t i = 0; i < n; i++) {
        mean_a += a[i].to_float();
        mean_b += b[i].to_float();
    }
    mean_a /= n;
    mean_b /= n;

    double cov = 0;
    double var_a = 0;
    double var_b = 0;
    for (size_t i = 0; i < n; i++) {
        double da = a[i].to_float() - mean_a;
        double db = b[i].to_float() - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    return static_cast<float>(cov / std::sqrt(var_a * var_b));
}

// Reference matmul on CPU so we can compare device output against a golden.
// A is MxK, B is KxN, output is MxN, all in row-major bfloat16.
void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            uint32_t idx_c = j + (i * N);
            uint32_t idx_a = i * K;
            uint32_t idx_b = j;
            float c_f = 0;
            for (uint32_t k_m = 0; k_m < K; k_m++) {
                c_f += a[idx_a].to_float() * b[idx_b].to_float();
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}

int main(int argc, char** argv) {
    // Create device handle
    IDevice* device = CreateDevice(0);

    // Setup program to execute along with its buffers and kernels to use
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    // Matrix dimensions (must each be divisible by the tile size)
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 128;
    constexpr uint32_t K = 128;

    static_assert(M % TILE_HEIGHT == 0, "M must be divisible by TILE_HEIGHT");
    static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");
    static_assert(K % TILE_WIDTH == 0, "K must be divisible by TILE_WIDTH");

    constexpr uint32_t Mt = M / TILE_HEIGHT;
    constexpr uint32_t Kt = K / TILE_WIDTH;
    constexpr uint32_t Nt = N / TILE_WIDTH;

    constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Create descriptors of DRAM allocations (one per input/output matrix)
    InterleavedBufferConfig dram_config_A{
        .device = device,
        .size = single_tile_size * Mt * Kt,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};

    InterleavedBufferConfig dram_config_B{
        .device = device,
        .size = single_tile_size * Kt * Nt,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};

    InterleavedBufferConfig dram_config_C{
        .device = device,
        .size = single_tile_size * Mt * Nt,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};

    // Use descriptor configuration to allocate buffers in DRAM on the device
    std::shared_ptr<Buffer> src0_dram_buffer = CreateBuffer(dram_config_A);
    std::shared_ptr<Buffer> src1_dram_buffer = CreateBuffer(dram_config_B);
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config_C);

    // Create L1 circular buffers shared between the data-movement and compute kernels.
    // Double-buffered (2 tiles each) so the reader can stage the next tile while the FPU
    // is working on the current one.
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    constexpr uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Allocate input data and fill it with random bfloat16 values in [0, 1)
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);
    std::vector<bfloat16> src0_vec(M * K);
    std::vector<bfloat16> src1_vec(K * N);

    for (bfloat16& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    for (bfloat16& v : src1_vec) {
        v = bfloat16(dist(rng));
    }

    // Run a CPU golden so we have something to compare the device output against
    std::vector<bfloat16> golden_vec(M * N, 0);
    golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

    // Tilize the input vectors to the 32x32 tiled layout the matrix engine expects
    src0_vec = tilize_nfaces(src0_vec, M, K);
    src1_vec = tilize_nfaces(src1_vec, K, N);

    // Write the src0 and src1 data to DRAM on the device
    tt::tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);
    tt::tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);

    // Reader kernel on the first RISC-V data-movement core
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/reader_single_core_mm.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Writer kernel on the second RISC-V data-movement core
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/writer_single_core_mm.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compute kernel - Mt/Kt/Nt are baked in at compile time, so it takes no runtime args
    std::vector<uint32_t> compute_compile_time_args = {Mt, Kt, Nt};
    CreateKernel(
        program,
        "kernels/compute/mm.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    // Configure runtime kernel arguments
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_dram_buffer->address(),
         src1_dram_buffer->address(),
         Mt,
         Kt,
         Nt});

    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {dst_dram_buffer->address(), Mt, Nt});

    // Launch kernels on device and wait for completion
    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    // Copy results back from device DRAM
    std::vector<bfloat16> result_vec(M * N, 0);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Untilize so the host sees row-major bfloat16
    result_vec = untilize_nfaces(result_vec, M, N);

    // Compare with golden using Pearson correlation; bfloat16 + HiFi4 won't be bit-exact
    float pearson = pearson_correlation(golden_vec, result_vec);
    printf("Device vs golden PCC = %f\n", pearson);

    CloseDevice(device);

    if (pearson > 0.97f) {
        printf("Completed successfully on the device, with %zu elements\n", result_vec.size());
    } else {
        printf("Failure on the device, PCC %f below 0.97 with %zu elements\n", pearson, result_vec.size());
    }
}
