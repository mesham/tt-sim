// Minimal reproducer for tt-sim's DRAM tile modelling only the bottom 10 MiB of
// each 1 GiB bank, so a *top-down* DRAM allocation lands outside the registered
// address ranges and kills the simulator server.
//
// Found via upstream `metal_example_contributed_vecadd` (built as
// `build/programming_examples/contributed/vecadd`), which allocates its three
// DRAM buffers with `DeviceLocalBufferConfig{.bottom_up = false}`. tt-metal
// then places them at the *top* of the DRAM bank, and the first host write hits
//
//   IndexError: Provided address '0x3fffd000' does not match any registered
//   memory spaces
//     tt_sim/memory/memory.py:90 _locate_memory_space
//
// `tt_sim/device/tiles.py DRAMTile.__init__` registers exactly two 10 MiB
// banks, at 0x0 and at 0x4000_0000. Real Wormhole/Blackhole DRAM banks are
// 1 GiB apiece (hence the 0x4000_0000 stride), so every address between
// 0xA0_0000 and 0x4000_0000 is unmapped. tt-metal's default *bottom-up* DRAM
// allocation starts low and stays inside the modelled 10 MiB, which is why
// every other example works; `bottom_up = false` walks down from the top of
// the bank and falls straight into the hole.
//
// The failure is loud on the simulator side (Python traceback, server exits)
// but silent to the host: tt-metal has no "simulator died" path, so the program
// hangs until its timeout. On ttsim (the vendor reference) the same binary runs
// to completion.
//
// This is that allocation and nothing else -- no kernel, no compute. One
// top-down DRAM buffer, a host write, a host read back, dumped as
// `OPDIFF_RESULT:<hex>` for optests/diff.sh. The values are distinct
// (0xA5A50000 + i) so a wrong answer is unambiguous.

#include <cstdio>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NUM_WORDS = 256;
constexpr uint32_t BUF_BYTES = NUM_WORDS * sizeof(uint32_t);

int main(int argc, char** argv) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();

    // `.bottom_up = false` is the whole point: it asks the allocator for the
    // TOP of the DRAM bank rather than the bottom. Everything else here is
    // deliberately boring.
    const distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = BUF_BYTES, .buffer_type = BufferType::DRAM, .bottom_up = false};
    const distributed::ReplicatedBufferConfig buffer_config{.size = BUF_BYTES};
    auto buf = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());

    printf("dram buffer address = 0x%x\n", buf->address());

    std::vector<uint32_t> src_vec(NUM_WORDS);
    for (uint32_t i = 0; i < NUM_WORDS; i++) {
        src_vec[i] = 0xA5A50000u + i;
    }

    // On tt-sim this write is already fatal to the simulator server.
    distributed::EnqueueWriteMeshBuffer(cq, buf, src_vec, false);

    std::vector<uint32_t> out_vec;
    distributed::EnqueueReadMeshBuffer(cq, out_vec, buf, true);
    mesh_device->close();

    printf("OPDIFF_RESULT:");
    for (size_t i = 0; i < out_vec.size(); i++) {
        printf("%08x", out_vec[i]);
    }
    printf("\n");
    printf("Completed successfully on the device, with %u words\n", NUM_WORDS);
    return 0;
}
