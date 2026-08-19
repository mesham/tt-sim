#include <assert.h>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

// Example banks — the first example that spans more than one DRAM bank.
//
// Every other example in this tree allocates a single-page DRAM buffer and
// reaches it with `get_noc_addr_from_bank_id<true>(0, ...)`, i.e. bank 0,
// hardcoded. That leaves the whole of tt-metal's interleaved page->bank
// distribution untested: a simulator that modelled DRAM as one flat bank would
// pass the entire suite.
//
// Here the buffers are `page_size`-paged and `N_PAGES` long, so the allocator
// spreads their pages round-robin over all of the device's DRAM banks (12 on
// Wormhole: 6 controllers x 2 `dram_views`; 8 on Blackhole). The reader kernel
// walks the pages with `InterleavedAddrGen`, which recomputes
//
//     bank_index        = page_id % NUM_DRAM_BANKS
//     bank_offset_index = page_id / NUM_DRAM_BANKS
//     address           = bank_base + bank_offset_index * aligned_page_size
//                                   + bank_to_dram_offset[bank_index]
//     noc_xy            = dram_bank_to_noc_xy[noc][bank_index]
//
// device-side, from the tables the host wrote into L1 at init. So the host's
// scatter and the kernel's gather have to agree on the page size, on the bank
// count, and on every bank's NoC coordinate and base offset for the result to
// come back right. Disagreeing on any of them corrupts data silently: there is
// no fault, every address is a legal address, it is simply the wrong one.
#define N_PAGES 24
#define PAGE_SIZE 1024
#define PAGE_ELEMS (PAGE_SIZE / 4)
#define TOTAL_ELEMS (N_PAGES * PAGE_ELEMS)

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);

    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t buffer_size = N_PAGES * PAGE_SIZE;
    InterleavedBufferConfig dram_config{
        .device = device, .size = buffer_size, .page_size = PAGE_SIZE, .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    InterleavedBufferConfig l1_config{
        .device = device, .size = PAGE_SIZE, .page_size = PAGE_SIZE, .buffer_type = BufferType::L1};

    std::shared_ptr<Buffer> l1_buffer_1 = CreateBuffer(l1_config);
    std::shared_ptr<Buffer> l1_buffer_2 = CreateBuffer(l1_config);

    std::vector<uint32_t> src0_data(TOTAL_ELEMS);
    std::vector<uint32_t> src1_data(TOTAL_ELEMS);
    for (uint32_t i = 0; i < TOTAL_ELEMS; i++) {
        // Distinct per element AND per page, so a page landing in the wrong
        // bank (or at the wrong within-bank offset) cannot alias to the right
        // answer by accident.
        src0_data[i] = 0x1000 + i;
        src1_data[i] = 0x7000000 - 3 * i;
    }

    tt::tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_data);
    tt::tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_data);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "kernels/dataflow/bank_walk.cpp",
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
         (uint32_t)PAGE_SIZE,
         (uint32_t)N_PAGES});

    tt::tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result;
    tt::tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result);

    int errors = 0;
    for (uint32_t i = 0; i < TOTAL_ELEMS; i++) {
        uint32_t expected = src0_data[i] + src1_data[i];
        if (result[i] != expected) {
            if (errors < 8) {
                printf(
                    "Mismatch at element %u (page %u, bank %u): got 0x%08x expected 0x%08x\n",
                    i,
                    i / PAGE_ELEMS,
                    (i / PAGE_ELEMS) % 12,
                    result[i],
                    expected);
            }
            errors++;
        }
    }

    CloseDevice(device);

    if (errors != 0) {
        printf("FAILED: errors=%d of %d\n", errors, TOTAL_ELEMS);
        return 1;
    }
    printf("Completed successfully on the device, %d pages across the DRAM banks all correct\n", N_PAGES);
    return 0;
}
