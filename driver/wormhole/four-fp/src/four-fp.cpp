#include "host_api.hpp"
#include "device.hpp"
#include <tt-metalium/tt_metal.hpp>
#define DATA_SIZE 256
#define CHUNK_SIZE 64

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // Create device handle
    IDevice* device = CreateDevice(0);

    // Setup program to execute along with its buffers and kernels to use
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    // Create descriptor of DRAM allocation
    constexpr float in_ddr_tile_size = 4 * DATA_SIZE;
    InterleavedBufferConfig in_dram_config{
        .device = device, .size = in_ddr_tile_size, .page_size = in_ddr_tile_size, .buffer_type = BufferType::DRAM};
    // Outputs in DRAM are int32
    constexpr float out_ddr_tile_size = 4 * DATA_SIZE;
    InterleavedBufferConfig out_dram_config{
        .device = device, .size = out_ddr_tile_size, .page_size = out_ddr_tile_size, .buffer_type = BufferType::DRAM};

    // Use descriptor configuration to allocate buffers in DRAM on the device
    std::shared_ptr<Buffer> src0_dram_buffer = CreateBuffer(in_dram_config);
    std::shared_ptr<Buffer> src1_dram_buffer = CreateBuffer(in_dram_config);
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(out_dram_config);

    constexpr uint32_t l1_tile_size = 4 * CHUNK_SIZE;
    // Create L1 circular buffers to communicate between RV cores
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(l1_tile_size, {{CBIndex::c_0, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_0, l1_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(l1_tile_size, {{CBIndex::c_1, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_1, l1_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    CircularBufferConfig cb_src2_config =
        CircularBufferConfig(l1_tile_size, {{CBIndex::c_2, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_2, l1_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

    // Allocate input data and fill it with values (each will be added together)
    std::vector<float> src0_data(DATA_SIZE);
    std::vector<float> src1_data(DATA_SIZE);

    for (int i=0;i<DATA_SIZE;i++) {
        src0_data[i]=float(i);
        src1_data[i]=float((DATA_SIZE-i));
    }

    // Write the src0 and src1 data to DRAM on the device
    tt::tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_data);
    tt::tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_data);

    // Specify data movement kernel for launching on first RISC-V baby core
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Configure reader runtime kernel arguments
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_dram_buffer->address(),
         src1_dram_buffer->address(),
         DATA_SIZE,
         CHUNK_SIZE});

    // Specify data movement kernel for launching on last RISC-V baby core
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/write_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Configure writer runtime kernel arguments
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        {dst_dram_buffer->address(),
         DATA_SIZE,
         CHUNK_SIZE});

    // Compute kernel creation
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {},
        });

    // Configure compute runtime kernel arguments
    SetRuntimeArgs(
        program,
        compute_kernel_id,
        core,
        {DATA_SIZE,
         CHUNK_SIZE});

    // Launch kernel on device and wait for completion
    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    // Allocate result data on host for results and copy back to host
    std::vector<float> result_data(DATA_SIZE);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_data);

    int number_failures=0;
    for (int i=0;i<DATA_SIZE;i++) {
        if (result_data[i] != src0_data[i] + src1_data[i]) number_failures++;
    }

    CloseDevice(device);

    if (number_failures==0) {
        printf("Completed successfully on the device, with %d elements\n", DATA_SIZE);
    } else {
        printf("Failure on the device, %d fails with %d elements\n", number_failures, DATA_SIZE);
    }
}
