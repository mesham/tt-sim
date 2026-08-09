// dramratebench -- does a DRAM channel's aggregate read rate grow with the
// number of tiles reading it?
//
// WHAT THIS IS FOR
// ----------------
// On 2026-08-09 tt-sim started queueing a second DRAM request behind the first
// at the endpoint (`DramChannels` in tt_sim/device/tiles.py). Before that, a
// DRAM endpoint served every arrival independently, so three 4 KiB reads
// landing on the same cycle were all answered on the same cycle and a Wormhole
// channel sustained the *NoC link's* 32 B/cycle against the 24 B/cycle its own
// ISA doc page publishes.
//
// That term is charged from `dram.channel_serialisation.bytes_per_cycle`, which
// is `isa_doc_derived` and needs no measurement. What has never been checked is
// the SHAPE it asserts, and no existing probe can reach it: rung 2's retained
// DRAM rows are all `num_transactions = 1`, and a lone request never finds the
// channel busy. Endpoint occupancy is, by construction, invisible to every
// single-transfer measurement in this project.
//
// `wh_dram#performance` states the shape as a measurement, and
// `unit_costs.yaml` already quotes it: 1, 12 and 48 Tensix tiles each reading
// 1 MiB *from one DRAM channel simultaneously* measure 22.2, 22.3 and 22.3
// GB/s. The aggregate does not grow with the number of readers. That flatness
// IS endpoint occupancy stated as a vendor measurement, and it is what this
// program reproduces.
//
// WHY A FLAT READING IS NOT, ON ITS OWN, A RESULT
// ----------------------------------------------
// Flat is what a saturated channel looks like. It is also what a saturated NoC
// link looks like, what a saturated initiator issue rate looks like, and what N
// readers that never actually overlapped look like. A run that measured only
// the one-channel arm would be unreadable for exactly the reason nocreadbench's
// first two attempts were. So the flat curve is the EXPECTED answer and it is
// not the control. Three arms, and the second is the control:
//
//   onechan   N readers, all on ONE DRAM bank.   Expect FLAT.
//   fanchan   N readers on N DISTINCT banks.     Expect it to SCALE.  <-- control
//   samecore  N readers split across the >= 2 banks that share ONE physical
//             DRAM core. Same inbound NoC link, different GDDR6 channels.
//
// `fanchan` is what separates the endpoint from everything upstream of it. The
// same N tiles, the same issue loop, the same transaction size, the same
// barrier: only the endpoint differs. If `fanchan` scales and `onechan` does
// not, the thing that saturated is the endpoint, because nothing else about the
// two arms is different. If `fanchan` is ALSO flat, then something upstream --
// the readers' own issue rate, the fabric, or a barrier that did not hold --
// caps both arms, the one-channel flatness is uninterpretable, and this program
// says DEGENERATE rather than reporting the flat curve it was hoping for.
//
// `samecore` splits the remaining ambiguity. In `onechan` all N flows converge
// on one DRAM tile's inbound router link, so a link limit and a channel limit
// are still confounded. `samecore` keeps that link and changes the channel. If
// it beats `onechan`, the channel bound; if it matches, the link did.
//
// IT DOES NOT FIRE ON EITHER PART IN THIS TREE, and that is measured rather
// than assumed. It needs two banks on ONE physical NoC coordinate, and tt-metal
// maps every bank to a distinct coordinate on both descriptors -- Blackhole has
// one view per DRAM core, and Wormhole's twelve banks come back on twelve
// different coordinates even though their `get_bank_offset` values alternate
// 0x0 / 0x4000_0000, which is exactly the two-channels-per-tile split
// `wh_dram` describes. The pairs that share a tile are reachable through
// aliased sub-endpoints the host cannot tell apart from independent ones. So
// the arm is coded, is selected automatically wherever the mapping does front
// two banks on one core, and is SKIPPED with a reason on both parts today --
// which means the link-versus-channel ambiguity stays OPEN and must be said out
// loud when reporting, not quietly dropped.
//
// WHAT THIS CANNOT DO, ON EITHER PART
// -----------------------------------
// It cannot make anything chargeable. A number this program produces is a
// measurement on one part and can enter `unit_costs.yaml` only as
// `corroboration`, never as provenance. That is doubly true on Blackhole, which
// has NO published DRAM tile page at all -- no per-channel bandwidth, no
// address map, no channel count -- so `dram.bandwidth` cannot become chargeable
// there whatever this measures. Its value on Blackhole is validating the
// *shape*: an aggregate that refuses to grow with readers while a fanned-out
// control grows freely is endpoint occupancy, whatever the absolute number is.
//
//   dramratebench [--out FILE] [--bytes N] [--tx-bytes N] [--repeats R]
//                 [--max-readers N] [--readers 1,2,4,...] [--arms LIST]
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "kernels/dramratebench_layout.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

const char* arch_name(IDevice* device) {
    switch (device->arch()) {
        case tt::ARCH::WORMHOLE_B0: return "wormhole";
        case tt::ARCH::BLACKHOLE: return "blackhole";
        default: return "unknown";
    }
}

// One DRAM bank as the DEVICE reports it. Nothing here is computed from a grid
// dimension: `logical_core_from_dram_channel` and `get_bank_offset` are what
// the allocator itself uses, so a harvested or otherwise irregular part cannot
// be walked off the end of. The 2026-08-09 Blackhole card has worker columns
// {1..7, 10..14} against a physical {1..7, 10..16}; an arithmetic sweep over
// either DRAM or workers lands in a hole and returns a row that looks exactly
// like a measurement.
struct Bank {
    uint32_t id = 0;
    CoreCoord logical{0, 0};
    CoreCoord physical{0, 0};   // virtual/physical NoC coord the kernel addresses
    uint32_t offset = 0;        // bank-local base this bank's pages start at
    uint32_t tag = 0;           // the word the host wrote at `offset + base`
};

// One measured point: one arm at one reader count.
struct Point {
    std::string arm;
    uint32_t num_readers = 0;
    uint32_t num_tx = 0;
    uint32_t tx_bytes = 0;
    std::vector<CoreCoord> readers;       // logical
    std::vector<uint32_t> bank_of_reader; // index into the bank table
    // Filled in after the run.
    bool measured = false;
    uint32_t max_cycles = 0;
    uint32_t min_cycles = 0;
    uint32_t max_spins = 0;
    uint32_t tags_ok = 0;
    uint32_t distinct_banks = 0;
    uint32_t distinct_dram_cores = 0;
    double agg_bytes_per_cycle = 0.0;
};

std::vector<uint32_t> parse_list(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) {
            j = s.size();
        }
        const std::string tok = s.substr(i, j - i);
        if (!tok.empty()) {
            out.push_back((uint32_t)strtoul(tok.c_str(), nullptr, 0));
        }
        i = j + 1;
    }
    return out;
}

bool wants(const std::string& arms, const char* name) {
    return arms.empty() || arms.find(name) != std::string::npos;
}

// The aggregate is built from max(t1 - t0) over the readers, NEVER from
// max(t1) - min(t0). Each reader differences its OWN clock, so a per-core clock
// epoch cancels; differencing across cores does not. This project's own
// congestion runs found Blackhole core (11, 2) holding a +323438586-cycle
// offset that reproduced over five runs, so this is not a hypothetical.
// max() rather than mean() charges the slowest reader against the aggregate,
// which is the conservative direction: it can only make a scaling arm look
// LESS scaled, never more.
double aggregate_bytes_per_cycle(uint64_t total_bytes, uint32_t max_cycles) {
    if (max_cycles == 0) {
        return 0.0;
    }
    return (double)total_bytes / (double)max_cycles;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path, readers_arg, arms_arg;
    uint32_t bytes_per_reader = 1u << 20;  // 1 MiB, the vendor table's figure
    uint32_t tx_bytes = 4096;
    uint32_t repeats = 3;
    uint32_t max_readers = 0;  // 0 = whatever the grid gives, capped below
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--out" && i + 1 < argc) {
            out_path = next();
        } else if (a == "--bytes" && i + 1 < argc) {
            bytes_per_reader = (uint32_t)strtoul(next().c_str(), nullptr, 0);
        } else if (a == "--tx-bytes" && i + 1 < argc) {
            tx_bytes = (uint32_t)strtoul(next().c_str(), nullptr, 0);
        } else if (a == "--repeats" && i + 1 < argc) {
            repeats = (uint32_t)strtoul(next().c_str(), nullptr, 0);
        } else if (a == "--max-readers" && i + 1 < argc) {
            max_readers = (uint32_t)strtoul(next().c_str(), nullptr, 0);
        } else if (a == "--readers" && i + 1 < argc) {
            readers_arg = next();
        } else if (a == "--arms" && i + 1 < argc) {
            arms_arg = next();
        } else if (a == "-h" || a == "--help") {
            printf(
                "dramratebench [options]\n"
                "  --out FILE         where to write the CSV\n"
                "  --bytes N          bytes each reader pulls (default 1048576)\n"
                "  --tx-bytes N       size of one noc_async_read (default 4096)\n"
                "  --repeats R        how many times to run the whole plan (default 3)\n"
                "  --max-readers N    cap the sweep (default: the whole worker grid)\n"
                "  --readers a,b,c    the reader counts to sweep (default 1,2,4,8,12,16,24,32,48)\n"
                "  --arms LIST        substring filter over onechan,fanchan,samecore\n");
            return 0;
        } else {
            fprintf(stderr, "dramratebench: unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }
    if (tx_bytes == 0 || bytes_per_reader < tx_bytes) {
        fprintf(stderr, "dramratebench: --bytes must be at least --tx-bytes\n");
        return 2;
    }

    IDevice* device = CreateDevice(0);
    const std::string arch = arch_name(device);
    const CoreCoord grid = device->compute_with_storage_grid_size();
    const int clock_mhz = device->get_clock_rate_mhz();
    if (out_path.empty()) {
        out_path = "dramratebench-" + arch + ".csv";
    }

    // --- the DRAM banks, as the device reports them -------------------------
    const uint32_t nbanks = device->allocator()->get_num_banks(BufferType::DRAM);
    if ((int)nbanks != device->num_dram_channels()) {
        // The whole plan assumes bank id == DRAM channel index, which is what
        // lets `get_bank_offset` and `logical_core_from_dram_channel` be used
        // together. Refuse rather than measure the wrong endpoint: this program
        // exists to tell one channel from several.
        fprintf(stderr,
                "dramratebench: %u DRAM banks but %d DRAM channels; bank ids are not\n"
                "channel indices on this part and the plan cannot be built safely.\n",
                nbanks, device->num_dram_channels());
        CloseDevice(device);
        return 1;
    }
    std::vector<Bank> banks;
    for (uint32_t b = 0; b < nbanks; b++) {
        Bank bank;
        bank.id = b;
        bank.logical = device->logical_core_from_dram_channel(b);
        bank.physical = device->virtual_core_from_logical_core(bank.logical, CoreType::DRAM);
        bank.offset = (uint32_t)device->allocator()->get_bank_offset(BufferType::DRAM, b);
        bank.tag = 0xDB000000u | b;
        banks.push_back(bank);
    }
    // Which banks share a physical DRAM core? On Wormhole a DRAM tile is what
    // `wh_dram` calls a group -- "DRAM tiles occur in groups of three, with two
    // channels of GDDR6 present in each group" -- so two banks per core is the
    // expected shape and is what makes the `samecore` arm buildable. The
    // Blackhole descriptor has one view per core, so there it does not exist and
    // the arm is skipped by name rather than faked.
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> by_core;
    for (const Bank& b : banks) {
        by_core[{(uint32_t)b.physical.x, (uint32_t)b.physical.y}].push_back(b.id);
    }
    std::vector<uint32_t> shared_core_banks;
    for (const auto& kv : by_core) {
        if (kv.second.size() >= 2 && kv.second.size() > shared_core_banks.size()) {
            shared_core_banks = kv.second;
        }
    }
    // Banks on distinct physical cores, for the fan-out arm: preferring
    // distinct cores keeps `fanchan` from silently becoming `samecore` on a part
    // where two banks share one inbound link.
    std::vector<uint32_t> distinct_core_banks;
    {
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (const Bank& b : banks) {
            const auto key = std::make_pair((uint32_t)b.physical.x, (uint32_t)b.physical.y);
            if (seen.insert(key).second) {
                distinct_core_banks.push_back(b.id);
            }
        }
        for (const Bank& b : banks) {  // then the rest, so the arm can still widen
            if (std::find(distinct_core_banks.begin(), distinct_core_banks.end(), b.id) ==
                distinct_core_banks.end()) {
                distinct_core_banks.push_back(b.id);
            }
        }
    }

    // --- the reader cores, in LOGICAL space ---------------------------------
    // Every coordinate here comes from `compute_with_storage_grid_size`, so a
    // harvested column simply is not addressable and no reader count can walk
    // into one. Nothing is derived arithmetically from a physical coordinate.
    std::vector<CoreCoord> all_readers;
    for (uint32_t y = 0; y < (uint32_t)grid.y; y++) {
        for (uint32_t x = 0; x < (uint32_t)grid.x; x++) {
            all_readers.push_back(CoreCoord{x, y});
        }
    }
    const uint32_t grid_readers = (uint32_t)all_readers.size();
    if (max_readers == 0 || max_readers > grid_readers) {
        max_readers = grid_readers;
    }
    std::vector<uint32_t> sweep =
        readers_arg.empty() ? std::vector<uint32_t>{1, 2, 4, 8, 12, 16, 24, 32, 48}
                            : parse_list(readers_arg);
    {
        std::vector<uint32_t> keep;
        for (uint32_t n : sweep) {
            if (n >= 1 && n <= max_readers &&
                std::find(keep.begin(), keep.end(), n) == keep.end()) {
                keep.push_back(n);
            }
        }
        // Always carry the widest point the part can reach: the whole question
        // is whether the aggregate grows, and a sweep that stops early is a
        // weaker test of that than one that does not.
        if (!keep.empty() && keep.back() != max_readers) {
            keep.push_back(max_readers);
        }
        std::sort(keep.begin(), keep.end());
        sweep = keep;
    }
    if (sweep.empty()) {
        fprintf(stderr, "dramratebench: no usable reader counts\n");
        CloseDevice(device);
        return 1;
    }

    // --- the plan -----------------------------------------------------------
    std::vector<Point> plan;
    const uint32_t num_tx = bytes_per_reader / tx_bytes;
    for (uint32_t n : sweep) {
        for (const char* arm : {"onechan", "fanchan", "samecore"}) {
            if (!wants(arms_arg, arm)) {
                continue;
            }
            if (strcmp(arm, "samecore") == 0 && shared_core_banks.size() < 2) {
                continue;  // reported once, after the plan is printed
            }
            Point p;
            p.arm = arm;
            p.num_readers = n;
            p.num_tx = num_tx;
            p.tx_bytes = tx_bytes;
            for (uint32_t i = 0; i < n; i++) {
                p.readers.push_back(all_readers[i]);
                if (strcmp(arm, "onechan") == 0) {
                    p.bank_of_reader.push_back(banks[0].id);
                } else if (strcmp(arm, "fanchan") == 0) {
                    p.bank_of_reader.push_back(distinct_core_banks[i % distinct_core_banks.size()]);
                } else {
                    p.bank_of_reader.push_back(shared_core_banks[i % shared_core_banks.size()]);
                }
            }
            plan.push_back(p);
        }
    }
    if (plan.empty()) {
        fprintf(stderr, "dramratebench: --arms selected nothing\n");
        CloseDevice(device);
        return 1;
    }

    // --- DRAM payload -------------------------------------------------------
    // One page per bank, so every bank holds a slice at the SAME bank-local
    // address and a reader is aimed by coordinate rather than by address
    // arithmetic. Readers within a point start at different offsets inside the
    // slice, so N readers on one channel really do pull N slices through it
    // rather than re-reading one row.
    const uint32_t slices = std::min(max_readers, 16u);
    const uint32_t slice_bytes = bytes_per_reader;
    const uint32_t page_bytes = slices * slice_bytes;
    InterleavedBufferConfig dram_cfg{.device = device,
                                     .size = (uint64_t)page_bytes * nbanks,
                                     .page_size = page_bytes,
                                     .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> dram = CreateBuffer(dram_cfg);
    const uint32_t dram_base = (uint32_t)dram->address();

    // L1: the landing arena plus one word for the arrival flag source, and the
    // arrival array itself. Both are allocated through the allocator, so they
    // sit at the same address on every core -- which is what lets one runtime
    // argument name a peer's slot.
    const uint32_t land_bytes = std::max(tx_bytes * 2, 8192u);
    const uint32_t land_alloc = land_bytes + 64;
    InterleavedBufferConfig land_cfg{
        .device = device, .size = land_alloc, .page_size = land_alloc, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> land = CreateBuffer(land_cfg);
    const uint32_t land_addr = (uint32_t)land->address();

    // One 16-byte slot per reader: see DRAMRATEBENCH_SLOT_STRIDE. A 4-byte
    // stride makes every reader past 0 write to an address not congruent to
    // the source in its low 4 bits, which hardware drops silently.
    const uint32_t arrive_bytes = std::max(max_readers * DRAMRATEBENCH_SLOT_STRIDE, 64u);
    InterleavedBufferConfig arrive_cfg{
        .device = device, .size = arrive_bytes, .page_size = arrive_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> arrive = CreateBuffer(arrive_cfg);
    const uint32_t arrive_addr = (uint32_t)arrive->address();

    constexpr uint32_t result_bytes = DRAMRATEBENCH_R_WORDS * 4;
    InterleavedBufferConfig res_cfg{
        .device = device, .size = result_bytes, .page_size = result_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> results = CreateBuffer(res_cfg);
    const uint32_t results_addr = (uint32_t)results->address();

    // The addressing tags. Only 32 bytes per bank are written -- the rest of
    // the payload is never inspected, because a read rate does not depend on
    // what the bytes are, and writing hundreds of MiB over a simulator's wire
    // would take longer than every measurement in this program put together.
    // NOTE THE ASYMMETRY, and it is not a slip. `WriteToDeviceDRAMChannel`
    // takes an address "on DRAM channel" -- it resolves the channel to its
    // (core, offset) itself -- so the host passes the bank-relative
    // `dram_base`. The KERNEL addresses the tile directly with `get_noc_addr`,
    // which has no channel to resolve, so it must carry `dram_base + offset`
    // exactly as tt-metal's own `get_noc_addr_from_bank_id` adds
    // `bank_to_dram_offset[bank_id]`. Adding the offset in both places is
    // invisible on Blackhole, where every offset is 0, and walks a Wormhole
    // write off the end of the tile's 2 GiB address space -- which is how it
    // was found: `IndexError: 0x802d4c40 does not match any registered memory
    // spaces` from the simulator, on the first Wormhole run.
    for (const Bank& b : banks) {
        std::vector<uint32_t> tag(8, b.tag);
        detail::WriteToDeviceDRAMChannel(device, (int)b.id, dram_base, tag);
    }

    printf("dramratebench: arch=%s grid=%ux%u clock=%dMHz banks=%u readers<=%u points=%zu\n",
           arch.c_str(), (unsigned)grid.x, (unsigned)grid.y, clock_mhz, nbanks, max_readers,
           plan.size());
    printf("dramratebench: bytes/reader=%u tx=%uB num_tx=%u repeats=%u dram_base=0x%08X\n",
           bytes_per_reader, tx_bytes, num_tx, repeats, dram_base);
    for (const Bank& b : banks) {
        printf("  bank %2u  logical (%u,%u)  noc (%u,%u)  offset 0x%08X  tag 0x%08X\n", b.id,
               (unsigned)b.logical.x, (unsigned)b.logical.y, (unsigned)b.physical.x,
               (unsigned)b.physical.y, b.offset, b.tag);
    }
    if (shared_core_banks.size() < 2) {
        printf("dramratebench: no physical DRAM core fronts two banks on this part, so the\n"
               "  `samecore` arm (same inbound NoC link, different GDDR6 channel) is SKIPPED.\n"
               "  Without it, `onechan` cannot separate a channel limit from the limit of the\n"
               "  one router link every reader's flow converges on. Say so when reporting.\n");
    }
    fflush(stdout);

    FILE* out = fopen(out_path.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "dramratebench: cannot write '%s'\n", out_path.c_str());
        CloseDevice(device);
        return 1;
    }
    fprintf(out, "# dramratebench arch=%s magic=0x%08X grid=%ux%u clock_mhz=%d banks=%u\n",
            arch.c_str(), DRAMRATEBENCH_MAGIC, (unsigned)grid.x, (unsigned)grid.y, clock_mhz,
            nbanks);
    fprintf(out, "# bytes_per_reader=%u tx_bytes=%u num_tx=%u repeats=%u samecore=%s\n",
            bytes_per_reader, tx_bytes, num_tx, repeats,
            shared_core_banks.size() >= 2 ? "yes" : "absent-on-this-part");
    fprintf(out, "# agg_bytes_per_cycle = num_readers * bytes_per_reader / max(t1-t0); each\n");
    fprintf(out, "# reader differences its OWN clock, so a per-core clock epoch cancels\n");
    if (clock_mhz <= 0) {
        // tt-sim reports 0 MHz, which is honest -- it has no wall-clock rate.
        // Say so rather than emitting a GB/s column of zeros that reads as a
        // measured collapse. B/cycle is the column that means anything here,
        // and it is the one the verdict grades.
        fprintf(out, "# clock_mhz is 0: this device reports no frequency (tt-sim does not),\n");
        fprintf(out, "# so agg_gb_per_s is 0 THROUGHOUT and is not a reading. Use B/cycle.\n");
        printf("dramratebench: the device reports 0 MHz, so the agg_gb_per_s column is 0 in\n"
               "  every row and is NOT a measured collapse. Read agg_bytes_per_cycle.\n");
    }
    fprintf(out,
            "arm,repeat,point,num_readers,num_tx,tx_bytes,bytes_per_reader,total_bytes,"
            "max_cycles,min_cycles,agg_bytes_per_cycle,agg_gb_per_s,per_reader_bytes_per_cycle,"
            "distinct_banks,distinct_dram_cores,tags_ok,max_barrier_spins,measured_readers\n");

    size_t failures = 0;
    for (uint32_t rep = 0; rep < repeats; rep++) {
        for (size_t pi = 0; pi < plan.size(); pi++) {
            Point& p = plan[pi];

            std::vector<CoreCoord> reader_phys;
            for (const CoreCoord& c : p.readers) {
                reader_phys.push_back(device->worker_core_from_logical_core(c));
            }
            std::set<uint32_t> banks_used;
            std::set<std::pair<uint32_t, uint32_t>> cores_used;
            for (uint32_t b : p.bank_of_reader) {
                banks_used.insert(b);
                cores_used.insert({(uint32_t)banks[b].physical.x, (uint32_t)banks[b].physical.y});
            }
            p.distinct_banks = (uint32_t)banks_used.size();
            p.distinct_dram_cores = (uint32_t)cores_used.size();

            std::vector<CoreRange> ranges;
            for (const CoreCoord& c : p.readers) {
                ranges.push_back(CoreRange(c));
            }
            CoreRangeSet all_cores(std::set<CoreRange>(ranges.begin(), ranges.end()));

            Program program = CreateProgram();
            KernelHandle k = CreateKernel(
                program, "kernels/dataflow/dram_reader.cpp", all_cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

            for (uint32_t i = 0; i < p.num_readers; i++) {
                const Bank& b = banks[p.bank_of_reader[i]];
                std::vector<uint32_t> args(DRAMRATEBENCH_A_PEERS, 0);
                args[DRAMRATEBENCH_A_RESULTS] = results_addr;
                args[DRAMRATEBENCH_A_ARRIVE] = arrive_addr;
                args[DRAMRATEBENCH_A_LAND] = land_addr;
                args[DRAMRATEBENCH_A_LAND_BYTES] = land_bytes;
                args[DRAMRATEBENCH_A_POINT] = (uint32_t)pi;
                args[DRAMRATEBENCH_A_INDEX] = i;
                args[DRAMRATEBENCH_A_NUM_READERS] = p.num_readers;
                args[DRAMRATEBENCH_A_NUM_TX] = p.num_tx;
                args[DRAMRATEBENCH_A_TX_BYTES] = p.tx_bytes;
                args[DRAMRATEBENCH_A_DRAM_X] = (uint32_t)b.physical.x;
                args[DRAMRATEBENCH_A_DRAM_Y] = (uint32_t)b.physical.y;
                // Reader i starts one slice further into the bank's page, so N
                // readers on one channel pull N distinct slices rather than
                // re-reading one. `% slices` because the page holds a bounded
                // number of them; the tag lives at offset 0 of slice 0 and is
                // only ever read outside the timed region.
                args[DRAMRATEBENCH_A_DRAM_ADDR] =
                    dram_base + b.offset + (i % slices) * slice_bytes;
                args[DRAMRATEBENCH_A_SLICE_BYTES] = slice_bytes;
                // Only slice 0 carries the tag. Every other slice would need its
                // own write, and the point of the check is to prove the reader
                // reached the right BANK, which slice 0 establishes.
                args[DRAMRATEBENCH_A_TAG] = ((i % slices) == 0) ? b.tag : 0;
                for (const CoreCoord& c : reader_phys) {
                    args.push_back((uint32_t)c.x);
                    args.push_back((uint32_t)c.y);
                }
                SetRuntimeArgs(program, k, p.readers[i], args);
            }

            // Zero the arrival array on every participant BEFORE the launch. The
            // kernel must not clear it itself: a peer's write can land before
            // this core has run its first instruction, and a kernel that zeroed
            // its own array would drop that arrival and hang.
            std::vector<uint32_t> zeros(arrive_bytes / 4, 0);
            std::vector<uint32_t> res_zeros(DRAMRATEBENCH_R_WORDS, 0);
            for (const CoreCoord& c : p.readers) {
                detail::WriteToDeviceL1(device, c, arrive_addr, zeros);
                detail::WriteToDeviceL1(device, c, results_addr, res_zeros);
            }
            detail::LaunchProgram(device, program, true, true);

            uint32_t max_cycles = 0, min_cycles = 0xFFFFFFFFu, max_spins = 0, tags_ok = 0,
                     measured = 0;
            for (uint32_t i = 0; i < p.num_readers; i++) {
                std::vector<uint32_t> r;
                detail::ReadFromDeviceL1(device, p.readers[i], results_addr, result_bytes, r);
                if (r.size() <= DRAMRATEBENCH_R_MAGIC || r[DRAMRATEBENCH_R_MAGIC] != DRAMRATEBENCH_MAGIC) {
                    continue;
                }
                measured++;
                const uint32_t c = r[DRAMRATEBENCH_R_CYCLES];
                max_cycles = std::max(max_cycles, c);
                min_cycles = std::min(min_cycles, c);
                max_spins = std::max(max_spins, r[DRAMRATEBENCH_R_BARRIER_SPINS]);
                tags_ok += r[DRAMRATEBENCH_R_TAG_OK];
            }
            if (measured != p.num_readers) {
                failures++;
                fprintf(stderr, "dramratebench: point %zu (%s, n=%u): %u of %u readers stamped\n",
                        pi, p.arm.c_str(), p.num_readers, measured, p.num_readers);
                continue;
            }
            if (min_cycles == 0xFFFFFFFFu) {
                min_cycles = 0;
            }
            const uint64_t total_bytes = (uint64_t)p.num_readers * bytes_per_reader;
            const double agg = aggregate_bytes_per_cycle(total_bytes, max_cycles);
            p.measured = true;
            p.max_cycles = max_cycles;
            p.min_cycles = min_cycles;
            p.max_spins = max_spins;
            p.tags_ok = tags_ok;
            p.agg_bytes_per_cycle = agg;
            fprintf(out, "%s,%u,%zu,%u,%u,%u,%u,%llu,%u,%u,%.4f,%.3f,%.4f,%u,%u,%u,%u,%u\n",
                    p.arm.c_str(), rep, pi, p.num_readers, p.num_tx, p.tx_bytes, bytes_per_reader,
                    (unsigned long long)total_bytes, max_cycles, min_cycles, agg,
                    agg * (double)clock_mhz / 1000.0, agg / (double)p.num_readers,
                    p.distinct_banks, p.distinct_dram_cores, tags_ok, max_spins, measured);
            fflush(out);
        }
    }
    fclose(out);
    CloseDevice(device);

    // --- the verdict, in the terms the hypotheses are stated in -------------
    // The same three checks the shell verdict makes, in the same order, so the
    // program and the session can never disagree about what was measured. The
    // ORDER is the whole design: the addressing proof and the overlap proof
    // both come before the control, and the control comes before the result.
    printf("\ndramratebench: wrote %s (%zu failures)\n", out_path.c_str(), failures);
    auto best = [&](const char* arm, uint32_t n) -> const Point* {
        for (const Point& p : plan) {
            if (p.measured && p.arm == arm && p.num_readers == n) {
                return &p;
            }
        }
        return nullptr;
    };
    uint32_t bad_tags = 0, points = 0, overlapped = 0, multi = 0;
    for (const Point& p : plan) {
        if (!p.measured) {
            continue;
        }
        points++;
        if (p.tags_ok != p.num_readers) {
            bad_tags++;
        }
        if (p.num_readers > 1) {
            multi++;
            if (p.max_spins > 0) {
                overlapped++;
            }
        }
    }
    const uint32_t n_lo = sweep.front(), n_hi = sweep.back();
    const Point* fan_lo = best("fanchan", n_lo);
    const Point* fan_hi = best("fanchan", n_hi);
    const Point* one_lo = best("onechan", n_lo);
    const Point* one_hi = best("onechan", n_hi);
    printf("dramratebench: %u point(s); %u of %u multi-reader point(s) had a reader wait at the\n"
           "  barrier; %u point(s) had a reader that did not verify its DRAM tag\n",
           points, overlapped, multi, bad_tags);
    if (points == 0) {
        printf("  VERDICT: DEGENERATE -- nothing was measured.\n");
    } else if (bad_tags != 0) {
        printf("  VERDICT: DEGENERATE -- %u point(s) held a reader that read something other\n"
               "  than the tag its target bank was given. The readers were not all pointed at\n"
               "  the endpoint the plan names, so no rate in this file means what it says.\n",
               bad_tags);
    } else if (multi > 0 && overlapped == 0) {
        printf("  VERDICT: DEGENERATE -- no reader ever waited at the barrier in ANY\n"
               "  multi-reader point, so the bursts did not overlap. N readers running one\n"
               "  after another produce exactly the flat aggregate this benchmark looks for,\n"
               "  from an experiment that never happened.\n");
    } else if (fan_lo == nullptr || fan_hi == nullptr || n_hi == n_lo) {
        printf("  VERDICT: DEGENERATE -- the fan-out control has no low and high point to\n"
               "  compare, so nothing separates the endpoint from the fabric or the issue rate.\n"
               "  EXPECTED against tt-sim, which instantiates only the tiles named in\n"
               "  TT_SIM_TENSIX_COORDS and so cannot sweep the reader count.\n");
    } else {
        const double fan_scale = fan_lo->agg_bytes_per_cycle > 0
                                     ? fan_hi->agg_bytes_per_cycle / fan_lo->agg_bytes_per_cycle
                                     : 0.0;
        const double one_scale =
            (one_lo != nullptr && one_hi != nullptr && one_lo->agg_bytes_per_cycle > 0)
                ? one_hi->agg_bytes_per_cycle / one_lo->agg_bytes_per_cycle
                : 0.0;
        printf("  fanchan aggregate %.3f -> %.3f B/cycle over %u -> %u readers (x%.2f)\n",
               fan_lo->agg_bytes_per_cycle, fan_hi->agg_bytes_per_cycle, n_lo, n_hi, fan_scale);
        if (one_lo != nullptr && one_hi != nullptr) {
            printf("  onechan aggregate %.3f -> %.3f B/cycle over %u -> %u readers (x%.2f)\n",
                   one_lo->agg_bytes_per_cycle, one_hi->agg_bytes_per_cycle, n_lo, n_hi, one_scale);
        }
        if (fan_scale < 1.5) {
            printf("  VERDICT: DEGENERATE -- the fan-out CONTROL did not move (x%.2f over %u ->\n"
                   "  %u readers). Something upstream of the endpoint -- the readers' own issue\n"
                   "  rate, the fabric, or a barrier that did not hold -- caps both arms, so a\n"
                   "  flat one-channel curve says nothing about the endpoint. EXPECTED against\n"
                   "  tt-sim, which models no link congestion and, on Blackhole, publishes no\n"
                   "  per-channel DRAM rate for the endpoint queue to be built from.\n",
                   fan_scale, n_lo, n_hi);
        } else if (one_lo == nullptr || one_hi == nullptr) {
            printf("  VERDICT: DEGENERATE -- the control moved but the one-channel arm has no\n"
                   "  comparable pair, so there is nothing to read the flatness off.\n");
        }
        // THE DISCRIMINATOR IS THE RATIO OF THE TWO SCALINGS, not a threshold on
        // the one-channel arm alone. A fixed "onechan must stay under x1.5"
        // rule reads a genuine endpoint bound as a refutation the moment the
        // sweep is not wide enough to saturate: this simulator, on Wormhole
        // with the cost model on, gives onechan x2.21 against fanchan x3.97 at
        // four readers -- the concentrated arm achieving 56% of what the SAME
        // readers achieve fanned out, which is the endpoint costing something,
        // and an absolute x1.5 rule would have called it a refutation. What the
        // experiment actually asks is "did concentrating the readers on one
        // endpoint cost anything relative to spreading them", and that is a
        // ratio. 0.75: concentrating cost at least a quarter of the scaling
        // that spreading achieved.
        const double efficiency = fan_scale > 0 ? one_scale / fan_scale : 1.0;
        if (efficiency <= 0.75) {
            printf("  VERDICT: ENDPOINT BOUND -- the fan-out control scaled x%.2f while the\n"
                   "  same readers on ONE channel managed only x%.2f, which is %.0f%% of it.\n"
                   "  Same reader count, same issue loop, same transaction size: only the\n"
                   "  endpoint differed, so the endpoint is what cost the difference. This is\n"
                   "  the shape tt-sim's DramChannels term asserts. It is CORROBORATION and\n"
                   "  never provenance: it cannot make dram.bandwidth chargeable, least of all\n"
                   "  on a part with no published DRAM page.\n",
                   fan_scale, one_scale, 100.0 * efficiency);
        } else {
            printf("  VERDICT: NO ENDPOINT BOUND -- one channel kept up: x%.2f against the\n"
                   "  control's x%.2f, %.0f%% of it. N readers on one channel are not serialised\n"
                   "  by it, which is what tt-sim's endpoint-occupancy term asserts they are.\n"
                   "  BEFORE READING THAT AS A REFUTATION, check the demand: a channel cannot\n"
                   "  flatten a load it is not saturated by, so a sweep whose widest point is\n"
                   "  %u reader(s) at %.2f B/cycle aggregate may simply be under-powered rather\n"
                   "  than unserialised. Wormhole's published channel rate is 24 B/cycle; the\n"
                   "  vendor's own flat table used 12 and 48 tiles. Widen --readers before\n"
                   "  concluding anything, and send this CSV back either way.\n",
                   one_scale, fan_scale, 100.0 * efficiency, n_hi,
                   one_hi->agg_bytes_per_cycle);
        }
        const Point* same_hi = best("samecore", n_hi);
        if (same_hi == nullptr) {
            printf("  samecore: not run on this part, so a channel limit and the limit of the\n"
                   "  one router link every onechan flow converges on stay confounded.\n");
        } else if (one_hi != nullptr && one_hi->agg_bytes_per_cycle > 0) {
            const double ratio = same_hi->agg_bytes_per_cycle / one_hi->agg_bytes_per_cycle;
            printf("  samecore %.3f B/cycle at %u readers, x%.2f over onechan: %s\n",
                   same_hi->agg_bytes_per_cycle, n_hi, ratio,
                   ratio >= 1.5 ? "two channels behind ONE inbound link beat one, so the CHANNEL "
                                  "is the limit and not the link"
                                : "splitting the channel behind the same link changed nothing, so "
                                  "the LINK may be the limit -- do not read onechan as a channel "
                                  "rate");
        }
    }
    return failures == 0 ? 0 : 1;
}
