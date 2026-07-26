#include <assert.h>
#include <cstdlib>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

#define DATA_SIZE 256
#define CHUNK_SIZE 64

using namespace tt;
using namespace tt::tt_metal;

// Example nine — eltwise-add split across two Tensix tiles with a circular
// buffer bridged over NoC. Tile A reads src0/src1 from DRAM, runs the
// TRISC compute path to produce the sum, and pushes each output tile across
// NoC into a Tile B L1 receive buffer (paced by a semaphore). Tile B writes
// each delivered tile back out to DRAM. Exercises ROADMAP §A multi-Tensix
// expansion end-to-end: per-tile L1 / coprocessor / mailbox independence,
// cross-tile NoC routing, semaphore signalling between cores.
int main(int argc, char** argv) {
    // tt-sim simulator hint: pre-construct both Tensix tiles we use. UMD's
    // tt_SimulationDevice spawns run.sh, which inherits the parent env,
    // which reaches the bridge's __main__.py. No-op on real Wormhole.
    setenv("TT_SIM_TENSIX_COORDS", "1-1,2-1", 1);

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();

    constexpr CoreCoord core_a = {0, 0};  // producer: reader + compute + sender
    constexpr CoreCoord core_b = {1, 0};  // consumer: writer

    // DRAM buffers — int8 inputs, int32 output (matches example four).
    constexpr uint32_t in_ddr_size = 1 * DATA_SIZE;
    constexpr uint32_t out_ddr_size = 4 * DATA_SIZE;
    InterleavedBufferConfig in_dram_config{
        .device = device,
        .size = in_ddr_size,
        .page_size = in_ddr_size,
        .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig out_dram_config{
        .device = device,
        .size = out_ddr_size,
        .page_size = out_ddr_size,
        .buffer_type = BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(in_dram_config);
    auto src1_dram_buffer = CreateBuffer(in_dram_config);
    auto dst_dram_buffer = CreateBuffer(out_dram_config);

    // L1 receive buffer on Tile B — sharded so it lives on core_b only,
    // not striped across worker banks. The full output (one int32 per
    // input element) fits in one allocation; the sender writes chunk by
    // chunk into successive offsets.
    constexpr uint32_t l1_recv_size = 4 * DATA_SIZE;
    ShardedBufferConfig l1_recv_config{
        .device = device,
        .size = l1_recv_size,
        .page_size = l1_recv_size,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>{CoreRange(core_b)}),
            {1, l1_recv_size},
            ShardOrientation::ROW_MAJOR,
            {1, l1_recv_size},
            {1, 1})};
    auto l1_recv_buffer = CreateBuffer(l1_recv_config);

    // Tile A's local pipeline — same shape as example four. CB0/CB1 take
    // int8 chunks from the reader; CB2 holds int32 chunks from compute.
    constexpr uint32_t l1_in_chunk = 1 * CHUNK_SIZE;
    constexpr uint32_t l1_out_chunk = 4 * CHUNK_SIZE;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(l1_in_chunk, {{CBIndex::c_0, tt::DataFormat::Int8}})
            .set_page_size(CBIndex::c_0, l1_in_chunk);
    tt_metal::CreateCircularBuffer(program, core_a, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(l1_in_chunk, {{CBIndex::c_1, tt::DataFormat::Int8}})
            .set_page_size(CBIndex::c_1, l1_in_chunk);
    tt_metal::CreateCircularBuffer(program, core_a, cb_src1_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(l1_out_chunk, {{CBIndex::c_2, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_2, l1_out_chunk);
    tt_metal::CreateCircularBuffer(program, core_a, cb_out_config);

    // Cross-tile pacing: a semaphore on Tile B that the sender increments
    // after every chunk is written. The writer waits until the count
    // reaches i+1 before forwarding chunk i to DRAM. Declaring on the
    // CoreRange covering both cores keeps get_semaphore() valid on either
    // side.
    auto signal_sem = tt_metal::CreateSemaphore(
        program, CoreRange(core_a, core_b), 0);

    // Inputs.
    std::vector<uint8_t> src0_data(DATA_SIZE);
    std::vector<uint8_t> src1_data(DATA_SIZE);
    for (int i = 0; i < DATA_SIZE; i++) {
        src0_data[i] = i % 128;
        src1_data[i] = (DATA_SIZE - i) % 128;
    }
    tt::tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_data);
    tt::tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_data);

    // Tile A: reader (BRISC, NOC 0).
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/read_kernel.cpp",
        core_a,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_a,
        {src0_dram_buffer->address(),
         src1_dram_buffer->address(),
         DATA_SIZE,
         CHUNK_SIZE});

    // Tile A: compute (TRISC).
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "kernels/compute/compute_kernel.cpp",
        core_a,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {}});
    SetRuntimeArgs(
        program,
        compute_kernel_id,
        core_a,
        {DATA_SIZE, CHUNK_SIZE});

    // Tile A: sender (NCRISC, NOC 1 default). tt-sim now keys NoC directories
    // by the canonical SoC-physical NoC 0 coord on both NoCs (ROADMAP §C
    // "Coord-system abstraction"), so kernels can use the same coord on
    // either NoC — matching real Wormhole behaviour under
    // ``translation_id_enabled``.
    KernelHandle sender_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/sender_kernel.cpp",
        core_a,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default});
    CoreCoord core_b_phys = device->worker_core_from_logical_core(core_b);
    SetRuntimeArgs(
        program,
        sender_kernel_id,
        core_a,
        {DATA_SIZE,
         CHUNK_SIZE,
         core_b_phys.x,
         core_b_phys.y,
         l1_recv_buffer->address(),
         signal_sem});

    // Tile B: writer (BRISC, NOC 0).
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/writer_kernel.cpp",
        core_b,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(
        program,
        writer_kernel_id,
        core_b,
        {dst_dram_buffer->address(),
         DATA_SIZE,
         CHUNK_SIZE,
         l1_recv_buffer->address(),
         signal_sem});

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result_data(DATA_SIZE);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_data);

    int number_failures = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        uint32_t expected = static_cast<uint32_t>(src0_data[i]) +
                            static_cast<uint32_t>(src1_data[i]);
        if (result_data[i] != expected) number_failures++;
    }

    CloseDevice(device);

    if (number_failures == 0) {
        printf("Completed successfully on the device, with %d elements\n", DATA_SIZE);
    } else {
        printf("Failure on the device, %d fails with %d elements\n",
               number_failures, DATA_SIZE);
    }

    return number_failures == 0 ? 0 : 1;
}
