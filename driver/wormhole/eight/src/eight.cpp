#include <assert.h>
#include <tt-metalium/tt_metal.hpp>
#include "host_api.hpp"
#include "device.hpp"

#define DATA_SIZE 100

using namespace tt;
using namespace tt::tt_metal;

// Example eight — same elementwise-add as example one, but the BRISC reader
// issues the two source reads with *distinct* NoC transaction IDs and barriers
// on them out-of-order. Exercises the per-trid request lifecycle in tt-sim's
// NoC (tt_sim/network/tt_noc.py): per-trid FIFO of return addresses,
// per-trid `NIU_MST_REQS_OUTSTANDING_ID_<n>` counters, and the kernel-visible
// `noc_async_read_barrier_with_trid` API.
int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);

    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 4 * DATA_SIZE;
    InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    InterleavedBufferConfig l1_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> l1_buffer_1 = CreateBuffer(l1_config);
    std::shared_ptr<Buffer> l1_buffer_2 = CreateBuffer(l1_config);

    std::vector<uint32_t> src0_data(DATA_SIZE);
    std::vector<uint32_t> src1_data(DATA_SIZE);
    for (int i = 0; i < DATA_SIZE; i++) {
        src0_data[i] = i;
        src1_data[i] = DATA_SIZE - i;
    }

    tt::tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_data);
    tt::tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_data);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/read_kernel_trid.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_dram_buffer->address(),
         src1_dram_buffer->address(),
         dst_dram_buffer->address(),
         l1_buffer_1->address(),
         l1_buffer_2->address(),
         DATA_SIZE});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result_data(DATA_SIZE);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_data);

    int number_failures = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        if (result_data[i] != src0_data[i] + src1_data[i]) number_failures++;
    }

    CloseDevice(device);

    if (number_failures == 0) {
        printf("Completed successfully on the device, with %d elements\n", DATA_SIZE);
    } else {
        printf("Failure on the device, %d fails with %d elements\n", number_failures, DATA_SIZE);
    }
}
