#include <assert.h>
#include <cstdlib>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

#define DATA_SIZE 512
#define CHUNK_SIZE 64

using namespace tt;
using namespace tt::tt_metal;

// Example pipestall — a two-core producer/consumer pipeline whose producer
// Tensix is deliberately back-pressured by a *remote* core, with the length of
// the wait under the caller's control.
//
// Why it exists: the per-unit blocked-cycle detector
// (`TT_SIM_UNIT_STALL`, tt_sim/device/deadlock.py) was calibrated on a survey
// of every in-tree workload, none of which pipelines core-to-core. The
// legitimate bound on how long a Tensix unit may block is architecturally the
// whole downstream pipeline — unpacker -> math -> Dst / output CB -> packer ->
// sender -> a consumer on another core — so the survey could not see its own
// worst case. This example puts that case in the numbers.
//
// Shape (same eltwise-add pipeline as example nine, plus a reverse credit):
//
//   Tile A (producer)                       Tile B (consumer)
//   ------------------                      ------------------
//   BRISC   reader:  DRAM -> CB0/CB1
//   TRISC   compute: CB0+CB1 -> CB2
//   NCRISC  sender:  CB2 -> B.L1 recv  ---> BRISC writer:
//                    wait credit >= i+1-C      wait data >= i+1
//                                              spin `delay` iterations
//                                              L1 -> DRAM
//                    <--- credit sem ------    inc A's credit sem
//
// The output CB holds `out_depth` chunks and the sender may run at most
// `credits` chunks ahead of the consumer, so once both are full the producer's
// pack thread parks in cb_reserve_back, the math thread cannot free Dst, and
// the unpacker blocks on a Src bank the Matrix Unit has not handed back —
// for as long as the *remote* core takes. Three knobs, read from the
// environment so one binary covers the whole sweep:
//
//   PIPESTALL_DELAY      consumer spin iterations per chunk (default 200)
//   PIPESTALL_CREDITS    chunks the sender may run ahead     (default 1)
//   PIPESTALL_OUT_DEPTH  pages in the producer's output CB   (default 1)
//
// Known bad combination, reproducible and not yet explained: OUT_DEPTH=2 with
// CREDITS=1 and DELAY>0 leaves chunk 1 all zeroes in the result (64 of 512
// elements) on the Wormhole simulator, while DELAY=0, CREDITS>=2 and
// OUT_DEPTH=4 are all correct. Something about a two-page output CB with the
// producer running exactly one chunk ahead; the pack of chunk 1 does not reach
// the sender. Reproduce with
// `PIPESTALL_OUT_DEPTH=2 PIPESTALL_DELAY=1000 ./build/pipestall`.
//
// Result is checked on the host: dst[i] = src0[i] + src1[i].
static uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') return fallback;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(raw, &end, 10);
    if (end == raw) return fallback;
    return static_cast<uint32_t>(parsed);
}

int main(int argc, char** argv) {
    // tt-sim simulator hint: pre-construct both Tensix tiles we use. Wormhole
    // *fallback* only — overwrite=0 lets a caller that already exported
    // TT_SIM_TENSIX_COORDS (the Blackhole run script uses "1-2,2-2") win.
    setenv("TT_SIM_TENSIX_COORDS", "1-1,2-1", 0);

    const uint32_t consumer_delay = env_u32("PIPESTALL_DELAY", 200);
    const uint32_t credits = env_u32("PIPESTALL_CREDITS", 1);
    const uint32_t out_depth = env_u32("PIPESTALL_OUT_DEPTH", 1);
    assert(credits >= 1 && out_depth >= 1);

    IDevice* device = CreateDevice(0);
    Program program = CreateProgram();

    constexpr CoreCoord core_a = {0, 0};  // producer: reader + compute + sender
    constexpr CoreCoord core_b = {1, 0};  // consumer: writer

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

    // Tile B's receive buffer holds the whole output, so each chunk lands at
    // its own offset and the credit semaphore is pure pacing — no slot reuse,
    // hence no correctness dependence on `credits`.
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

    // Input CBs are deliberately deep: the reader must be able to run ahead of
    // the compute, otherwise the unpack thread parks in cb_wait_front (a RISC-V
    // spin) instead of the unpacker blocking on a Src bank, and the stall this
    // example exists to create never reaches the Tensix backend.
    constexpr uint32_t l1_in_chunk = 1 * CHUNK_SIZE;
    constexpr uint32_t in_depth = 4;
    constexpr uint32_t l1_out_chunk = 4 * CHUNK_SIZE;

    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(in_depth * l1_in_chunk, {{CBIndex::c_0, tt::DataFormat::Int8}})
            .set_page_size(CBIndex::c_0, l1_in_chunk);
    tt_metal::CreateCircularBuffer(program, core_a, cb_src0_config);

    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(in_depth * l1_in_chunk, {{CBIndex::c_1, tt::DataFormat::Int8}})
            .set_page_size(CBIndex::c_1, l1_in_chunk);
    tt_metal::CreateCircularBuffer(program, core_a, cb_src1_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(out_depth * l1_out_chunk, {{CBIndex::c_2, tt::DataFormat::Int32}})
            .set_page_size(CBIndex::c_2, l1_out_chunk);
    tt_metal::CreateCircularBuffer(program, core_a, cb_out_config);

    // Two semaphores, both declared over the range covering both cores so
    // get_semaphore() resolves on either side:
    //   data_sem   on Tile B, incremented by the sender once a chunk has landed
    //   credit_sem on Tile A, incremented by the writer once a chunk is drained
    auto data_sem = tt_metal::CreateSemaphore(program, CoreRange(core_a, core_b), 0);
    auto credit_sem = tt_metal::CreateSemaphore(program, CoreRange(core_a, core_b), 0);

    std::vector<uint8_t> src0_data(DATA_SIZE);
    std::vector<uint8_t> src1_data(DATA_SIZE);
    for (int i = 0; i < DATA_SIZE; i++) {
        src0_data[i] = i % 128;
        src1_data[i] = (DATA_SIZE - i) % 128;
    }
    tt::tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_data);
    tt::tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_data);

    CoreCoord core_a_phys = device->worker_core_from_logical_core(core_a);
    CoreCoord core_b_phys = device->worker_core_from_logical_core(core_b);

    // Tile A: reader (BRISC, NOC 0).
    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/pipestall_reader.cpp",
        core_a,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_a,
        {src0_dram_buffer->address(), src1_dram_buffer->address(), DATA_SIZE, CHUNK_SIZE});

    // Tile A: compute (TRISC).
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "kernels/compute/pipestall_compute.cpp",
        core_a,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {}});
    SetRuntimeArgs(program, compute_kernel_id, core_a, {DATA_SIZE, CHUNK_SIZE});

    // Tile A: sender (NCRISC, NOC 1).
    KernelHandle sender_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/pipestall_sender.cpp",
        core_a,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(
        program,
        sender_kernel_id,
        core_a,
        {DATA_SIZE,
         CHUNK_SIZE,
         core_b_phys.x,
         core_b_phys.y,
         l1_recv_buffer->address(),
         data_sem,
         credit_sem,
         credits});

    // Tile B: writer (BRISC, NOC 0).
    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/pipestall_writer.cpp",
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
         data_sem,
         credit_sem,
         consumer_delay,
         core_a_phys.x,
         core_a_phys.y});

    printf(
        "pipestall: %d elements, %d chunks, delay=%u credits=%u out_depth=%u\n",
        DATA_SIZE,
        DATA_SIZE / CHUNK_SIZE,
        consumer_delay,
        credits,
        out_depth);

    tt::tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result_data(DATA_SIZE);
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_data);

    int number_failures = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
        uint32_t expected =
            static_cast<uint32_t>(src0_data[i]) + static_cast<uint32_t>(src1_data[i]);
        if (result_data[i] != expected) number_failures++;
    }

    CloseDevice(device);

    if (number_failures == 0) {
        printf("Completed successfully on the device, with %d elements\n", DATA_SIZE);
    } else {
        printf(
            "Failure on the device, %d fails with %d elements\n", number_failures, DATA_SIZE);
    }

    return number_failures == 0 ? 0 : 1;
}
