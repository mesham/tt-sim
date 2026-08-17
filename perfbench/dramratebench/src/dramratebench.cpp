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
// THE WRITE DIRECTION, ADDED 2026-08-17, AND WHY
// ----------------------------------------------
// `unit_costs.yaml` charges one `access_latency` to a DRAM read, a write and an
// atomic alike on Wormhole, and records the reason: Blackhole's rows split
// cleanly (a write of 22 cycles against a read of 126) while doing the same
// subtraction on Wormhole's gives write differences of 228 / 124 / 127 / 139 /
// 139 / 145 / 137 / 155 against read differences of 104 / 104 / 104 / 112 /
// 104 / 112 / 128 / 152 -- "no clean asymmetry, a visible 64 B outlier, and a
// write that is if anything DEARER than a read". The entry stays one figure
// "until Wormhole's own data says otherwise".
//
// A Wormhole card has since said something, and it AGREES with the decline: a
// latency probe on 2026-08-17 put the DRAM write at +21.2 cycles over the read
// at 256 B and +56.4 at 4096 B, dearer in both, which is the opposite sign to
// Blackhole's. So copying Blackhole's split across would have been actively
// wrong rather than merely unsupported.
//
// That probe cannot settle it, though, and neither can the vendor campaign it
// was read against: both are LATENCY instruments, one transaction at a time,
// and a latency instrument structurally cannot see an endpoint's OCCUPANCY --
// how long the endpoint is unavailable to the NEXT request. That is a rate
// question, it is the shape of this program, and `--dir write` is what asks it.
//
// A RATE CAMPAIGN AND A LATENCY CAMPAIGN ANSWER DIFFERENT QUESTIONS, and this
// one answers only its own. It can say whether N writers concentrated on one
// endpoint are serialised by it more, less, or the same as N readers are. It
// CANNOT say what one write costs, cannot decompose a difference into service
// time versus queueing, and cannot separate the endpoint from the one inbound
// router link every flow in the concentrated arm converges on -- the `samecore`
// arm, which is what would split those, does not exist on either part here.
// Report "the endpoint", never "the channel", in either direction.
//
// HOW A WRITE POINT PROVES IT WROTE WHERE IT SAYS
// ----------------------------------------------
// The bar is higher than the read arm's and it has to be. A read that misses
// its endpoint comes back with the wrong bytes and the reader itself reports
// it; a write that misses its endpoint completes, retires its barrier, and
// yields a perfectly plausible rate with the bytes sitting somewhere the plan
// never named. Three checks, and only the middle one has a read-arm analogue:
//
//   1. Every writer fills its whole L1 arena with a WITNESS word keyed to the
//      (bank, slice) it was aimed at, so every byte of the MEASURED traffic
//      carries the proof -- not a separate untimed poke that might have taken
//      a different path.
//   2. After the barrier each writer reads its own target block back and
//      stamps what it saw (`tags_ok`). Catches a write that was dropped or
//      never issued. Cannot catch misdirection: it re-reads through the same
//      coordinate the write used.
//   3. THE HOST reads every block of the write region back over PCIe and
//      requires BOTH that each writer's target block holds that writer's
//      witness (`witness_ok`) AND that every block the point did not target
//      still holds POISON (`stray_writes`). The negative half is the one that
//      catches misdirection, and the host re-poisons before EVERY point so it
//      cannot be satisfied by a correct value left behind by an earlier repeat.
//
// What that catches: a dropped write, a mis-congruent write, a bank offset
// applied twice or not at all, a wrong physical coordinate, a multicast where a
// unicast was meant, and a runtime argument that did not reach the kernel (the
// writer echoes the witness it was given and the host compares it to the one it
// sent). What it does NOT catch: bytes that landed correctly in the named bank
// but on the wrong GDDR6 channel behind it, because tt-metal fronts every bank
// on its own NoC coordinate on both parts and a host program cannot tell two
// sub-endpoints of one tile apart. That is the same ambiguity the read arm
// carries and it is not narrowed by measuring writes.
//
//   dramratebench [--out FILE] [--bytes N] [--tx-bytes N] [--repeats R]
//                 [--max-readers N] [--readers 1,2,4,...] [--arms LIST]
//                 [--dir read|write|both]
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

// One measured point: one arm at one reader count, in one direction.
struct Point {
    std::string arm;   // "onechan", or "onechan-write" in the write direction
    bool is_write = false;
    uint32_t num_readers = 0;
    uint32_t num_tx = 0;
    uint32_t tx_bytes = 0;
    std::vector<CoreCoord> readers;       // logical
    std::vector<uint32_t> bank_of_reader; // index into the bank table
    std::vector<uint32_t> slice_of_reader;  // which slice of that bank
    // Filled in after the run.
    bool measured = false;
    uint32_t max_cycles = 0;
    uint32_t min_cycles = 0;
    uint32_t max_spins = 0;
    uint32_t tags_ok = 0;
    // Write direction only; -1 in a read row, which is not "zero verified" but
    // "the host did not look, because for a read the DEVICE is the witness".
    int witness_ok = -1;
    int stray_writes = -1;
    uint32_t distinct_banks = 0;
    uint32_t distinct_dram_cores = 0;
    double agg_bytes_per_cycle = 0.0;
    // Every repeat's aggregate, kept for the sustained-rate table alone. The
    // fields above stay exactly what they were -- the LAST repeat's -- because
    // the verdict below is graded on them and a run recorded before this table
    // existed must keep reading the same way.
    std::vector<double> agg_samples;
};

// The MIDDLE repeat, not the mean and not the best. A repeat that hit an
// unrelated stall should move a sustained rate by nothing, and with the usual
// three repeats the median is one of the measurements rather than an average
// containing an outlier. (At an even count it is the mean of the middle two,
// which is the standard definition and is why `--repeats 3` is the default.)
double median_of(std::vector<double> v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    const size_t mid = v.size() / 2;
    return (v.size() % 2 == 1) ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

// `wh_dram#performance`, reads, static VC: 1, 12 and 48 Tensix tiles each
// reading 1 MiB from ONE DRAM channel simultaneously. WORMHOLE ONLY -- no
// Blackhole DRAM tile page is published at all, so there is no table to
// compare a Blackhole run against and inventing one from this would be
// laundering one part's published number into another part's gap.
const uint32_t VENDOR_READERS[3] = {1, 12, 48};
const double VENDOR_GB_PER_S[3] = {22.2, 22.3, 22.3};

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

// The CSV's arm field. The read direction's three names are UNCHANGED, so
// every parser that already grades this file -- `tt_sim.perf.dram_rate_sweep`,
// which matches `arm == "onechan"` exactly, and `card_session_verdicts.sh`,
// which matches the same string in awk -- keeps reading a read arm exactly as
// it did, and simply does not see the write rows. That is the intended
// behaviour and not an oversight: the read arm has a card control to reproduce
// and nothing about the write direction may perturb how it is graded.
std::string arm_name(const char* base, bool is_write) {
    return is_write ? std::string(base) + "-write" : std::string(base);
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
    std::string out_path, readers_arg, arms_arg, dir_arg = "read";
    uint32_t bytes_per_reader = 1u << 20;  // 1 MiB, the vendor table's figure
    uint32_t tx_bytes = 4096;
    uint32_t repeats = 3;
    uint32_t max_readers = 0;  // 0 = whatever the grid gives, capped below
    bool sustained_preset = false;
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
        } else if (a == "--dir" && i + 1 < argc) {
            // DEFAULTS TO `read`, and that is load-bearing rather than
            // conservative housekeeping. Every previously recorded invocation
            // of this program -- run_card.sh, run_card_session.sh, the sim
            // runners -- must keep producing exactly the file it produced
            // before, because the read arm carries a card control this project
            // reads other results against.
            dir_arg = next();
        } else if (a == "--sustained") {
            // The vendor's own experiment, as one flag, so that the run that
            // is meant to be comparable to a published table cannot be got
            // subtly wrong at the card. It sets only the reader counts: 1 MiB
            // per reader, 4096 B transactions and three repeats are ALREADY
            // the defaults, and this preset deliberately does not restate them
            // -- two places to change one parameter is how the two drift.
            sustained_preset = true;
        } else if (a == "-h" || a == "--help") {
            printf(
                "dramratebench [options]\n"
                "  --out FILE         where to write the CSV\n"
                "  --bytes N          bytes each reader pulls (default 1048576)\n"
                "  --tx-bytes N       size of one noc_async_read (default 4096)\n"
                "  --repeats R        how many times to run the whole plan (default 3)\n"
                "  --max-readers N    cap the sweep (default: the whole worker grid)\n"
                "  --readers a,b,c    the reader counts to sweep (default 1,2,4,8,12,16,24,32,48)\n"
                "  --arms LIST        substring filter over onechan,fanchan,samecore\n"
                "  --dir D            read (default), write, or both. `write` adds a -write\n"
                "                     copy of each arm, host-verified for misdirection\n"
                "  --sustained        the vendor's own reader counts, 1,12,48 (wh_dram#performance)\n");
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
    const bool do_read = (dir_arg == "read" || dir_arg == "both");
    const bool do_write = (dir_arg == "write" || dir_arg == "both");
    if (!do_read && !do_write) {
        fprintf(stderr, "dramratebench: --dir must be read, write or both (got '%s')\n",
                dir_arg.c_str());
        return 2;
    }
    // Slice bases must stay CONGRUENT with the L1 arena they are written from
    // and read into, and the arena is allocator-aligned. Congruence, not
    // absolute alignment: a NoC write out of L1 needs `(src % 16) == (dst % 16)`
    // and a DRAM-to-L1 read needs 32 on Wormhole, 64 on Blackhole. Both bases
    // are 0 mod 64 when the slice stride is, so requiring a multiple of 64 of
    // the slice stride makes every offset in this program congruent by
    // construction rather than by luck. A violation is UndefinedBehavior on
    // hardware -- silently skewed or dropped bytes, no fault -- which in the
    // write direction would present as a rate rather than as an error.
    //
    // Enforced for the WRITE direction only. The read arm has carried the same
    // exposure since it was written and has a card control recorded through it,
    // so turning a previously-accepted invocation into an error there would
    // change the arm this instrument exists to leave alone. It warns instead.
    if (bytes_per_reader % 64 != 0 || tx_bytes % 64 != 0) {
        if (do_write) {
            fprintf(stderr,
                    "dramratebench: --bytes (%u) and --tx-bytes (%u) must both be multiples\n"
                    "of 64 in the write direction, or the slice and transaction bases stop\n"
                    "being congruent with the L1 arena and hardware skews or drops the bytes\n"
                    "with no fault raised -- which would present as a rate, not an error.\n",
                    bytes_per_reader, tx_bytes);
            return 2;
        }
        fprintf(stderr,
                "dramratebench: WARNING --bytes (%u) / --tx-bytes (%u) are not multiples of\n"
                "64, so slice bases may not be congruent with the L1 arena on every part.\n",
                bytes_per_reader, tx_bytes);
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
    std::vector<uint32_t> default_sweep =
        sustained_preset ? std::vector<uint32_t>{1, 12, 48}
                         : std::vector<uint32_t>{1, 2, 4, 8, 12, 16, 24, 32, 48};
    std::vector<uint32_t> sweep = readers_arg.empty() ? default_sweep : parse_list(readers_arg);
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
    // The read direction is emitted FIRST and whole, then the write direction,
    // so that a run with `--dir both` holds the read arm's points at the same
    // plan indices and in the same order a `--dir read` run would. Interleaving
    // them would have been tidier and would have made the two directions'
    // points non-comparable with an earlier file.
    const uint32_t slices = std::min(max_readers, 16u);
    std::vector<Point> plan;
    const uint32_t num_tx = bytes_per_reader / tx_bytes;
    for (bool is_write : {false, true}) {
        if (is_write ? !do_write : !do_read) {
            continue;
        }
        for (uint32_t n : sweep) {
            for (const char* arm : {"onechan", "fanchan", "samecore"}) {
                if (!wants(arms_arg, arm)) {
                    continue;
                }
                if (strcmp(arm, "samecore") == 0 && shared_core_banks.size() < 2) {
                    continue;  // reported once, after the plan is printed
                }
                Point p;
                p.arm = arm_name(arm, is_write);
                p.is_write = is_write;
                p.num_readers = n;
                p.num_tx = num_tx;
                p.tx_bytes = tx_bytes;
                for (uint32_t i = 0; i < n; i++) {
                    p.readers.push_back(all_readers[i]);
                    if (strcmp(arm, "onechan") == 0) {
                        p.bank_of_reader.push_back(banks[0].id);
                    } else if (strcmp(arm, "fanchan") == 0) {
                        p.bank_of_reader.push_back(
                            distinct_core_banks[i % distinct_core_banks.size()]);
                    } else {
                        p.bank_of_reader.push_back(shared_core_banks[i % shared_core_banks.size()]);
                    }
                    p.slice_of_reader.push_back(i % slices);
                }
                plan.push_back(p);
            }
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
    const uint32_t slice_bytes = bytes_per_reader;
    const uint32_t region_bytes = slices * slice_bytes;
    // The write direction gets its OWN region rather than sharing the read
    // arm's. Writers overwrite whatever they land on, and the read arm's tags
    // sit at exactly the addresses a writer would target; sharing would have
    // made the read points that follow a write point fail their tag check, and
    // "the read arm broke once writes were enabled" is precisely the outcome
    // this program must not be able to produce.
    const uint32_t write_region_off = region_bytes;
    const uint32_t page_bytes = region_bytes * (do_write ? 2u : 1u);
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
    // One tag per SLICE, not one per bank. Reader `i` reads slice `i % slices`,
    // so tagging only slice 0 leaves every other reader with nothing to check
    // against -- which is exactly what the 2026-08-09 card run showed: `tags_ok`
    // came back 1 where `num_readers` was 24, the whole file was refused, and
    // the rates behind it were in fact clean (onechan flat at ~46 B/cycle from
    // 1 to 24 readers while fanchan scaled 46 -> 212). A verification that
    // cannot pass is worse than none: it condemns good data. `slices` is at
    // most 16 and each tag is 32 bytes, so this is half a kibibyte per bank.
    for (const Bank& b : banks) {
        std::vector<uint32_t> tag(8, b.tag);
        for (uint32_t s = 0; s < slices; ++s) {
            detail::WriteToDeviceDRAMChannel(device, (int)b.id, dram_base + s * slice_bytes, tag);
        }
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
    if (do_write) {
        fprintf(out, "# write region at bank-local +0x%08X; POISON 0x%08X re-laid before EVERY\n",
                write_region_off, DRAMRATEBENCH_POISON);
        fprintf(out, "# write point. witness_ok/stray_writes are -1 in a read row: not zero\n");
        fprintf(out, "# verified, but 'the host did not look, because for a read the DEVICE is\n");
        fprintf(out, "# the witness'. A write row is good only at witness_ok == num_readers AND\n");
        fprintf(out, "# stray_writes == 0 AND tags_ok == num_readers.\n");
    }
    // The eighteen DRB1 columns, in the DRB1 order, then the new ones appended.
    // Every consumer resolves columns by header NAME -- dram_rate_sweep.py via
    // csv.DictReader, card_session_verdicts.sh via `_csv_col` -- so appending
    // is invisible to both, and a banked DRB1 file stays readable by the same
    // code that reads a DRB2 one.
    fprintf(out,
            "arm,repeat,point,num_readers,num_tx,tx_bytes,bytes_per_reader,total_bytes,"
            "max_cycles,min_cycles,agg_bytes_per_cycle,agg_gb_per_s,per_reader_bytes_per_cycle,"
            "distinct_banks,distinct_dram_cores,tags_ok,max_barrier_spins,measured_readers,"
            "direction,witness_ok,stray_writes\n");

    // The write direction's host-side apparatus. `block_addr` is the ONE place
    // a write-region address is formed, so the read arm's addressing and the
    // write arm's cannot drift apart in a way that would look like a rate.
    std::vector<uint32_t> poison(DRAMRATEBENCH_TAG_BYTES / 4, DRAMRATEBENCH_POISON);
    auto block_addr = [&](uint32_t slice) {
        return dram_base + write_region_off + slice * slice_bytes;
    };
    // Bank-relative, as `WriteToDeviceDRAMChannel` and `ReadFromDeviceDRAMChannel`
    // want it: they resolve the channel to its (core, offset) themselves. The
    // KERNEL gets the same address plus `bank.offset`, exactly as the read arm
    // does and for the reason its comment above spells out.
    auto lay_poison = [&]() {
        for (const Bank& b : banks) {
            for (uint32_t s = 0; s < slices; ++s) {
                detail::WriteToDeviceDRAMChannel(device, (int)b.id, block_addr(s), poison);
            }
        }
    };

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
            // Two kernels, one per direction, and the reader's file is
            // byte-for-byte the one that produced the 2026-08-17 Wormhole card
            // control. See dram_writer.cpp's header for why that is worth the
            // duplicated barrier.
            KernelHandle k = CreateKernel(
                program,
                p.is_write ? "kernels/dataflow/dram_writer.cpp"
                           : "kernels/dataflow/dram_reader.cpp",
                all_cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

            std::vector<uint32_t> witness_of(p.num_readers, 0);
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
                const uint32_t slice = p.slice_of_reader[i];
                args[DRAMRATEBENCH_A_DRAM_ADDR] =
                    dram_base + b.offset + (p.is_write ? write_region_off : 0u) +
                    slice * slice_bytes;
                args[DRAMRATEBENCH_A_SLICE_BYTES] = slice_bytes;
                // Every slice carries this bank's tag, so every reader can prove
                // it reached the right BANK -- see the tag write above for why
                // tagging slice 0 alone made the check unpassable.
                args[DRAMRATEBENCH_A_TAG] = b.tag;
                // Keyed to the TARGET rather than to the writer, because a point
                // with more writers than slices has two writers aimed at one
                // block and they must agree about what belongs in it.
                witness_of[i] = DRAMRATEBENCH_WITNESS(b.id, slice);
                args[DRAMRATEBENCH_A_WITNESS] = p.is_write ? witness_of[i] : 0u;
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
            // Re-laid before EVERY write point, not once at start-up. After the
            // first repeat the region already holds correct witnesses, so a
            // second repeat whose writes were all silently dropped would read
            // back exactly the right words and certify itself. The whole value
            // of the negative check is that it can fail.
            if (p.is_write) {
                lay_poison();
            }
            detail::LaunchProgram(device, program, true, true);

            uint32_t max_cycles = 0, min_cycles = 0xFFFFFFFFu, max_spins = 0, tags_ok = 0,
                     measured = 0, arg_mismatch = 0;
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
                // Did the kernel receive the direction and the witness the host
                // believes it sent? A runtime argument that went astray is the
                // one failure that would otherwise make every other check agree
                // with itself: the device would verify its own wrong witness and
                // the host would look for it in the wrong place.
                const uint32_t want_dir =
                    p.is_write ? DRAMRATEBENCH_DIR_WRITE : DRAMRATEBENCH_DIR_READ;
                if (r[DRAMRATEBENCH_R_DIR] != want_dir ||
                    (p.is_write && r[DRAMRATEBENCH_R_WITNESS] != witness_of[i])) {
                    arg_mismatch++;
                }
            }
            if (arg_mismatch != 0) {
                failures++;
                fprintf(stderr,
                        "dramratebench: point %zu (%s, n=%u): %u participant(s) stamped a\n"
                        "  direction or witness the host did not send. The runtime arguments\n"
                        "  did not reach the kernel intact, so nothing in this point is graded.\n",
                        pi, p.arm.c_str(), p.num_readers, arg_mismatch);
                continue;
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

            // --- the write direction's host-side addressing proof ------------
            // Read the WHOLE write region back, not only the blocks this point
            // aimed at. The targeted ones must hold their witness, which says
            // the bytes arrived; the untargeted ones must still hold POISON,
            // which says no bytes arrived anywhere else. Only the second half
            // can catch a misdirected write, and a write that went to the wrong
            // bank is otherwise indistinguishable from one that went right --
            // it completes, retires, and yields a plausible rate.
            //
            // What this still cannot see: bytes that landed in the right bank
            // but on the wrong GDDR6 channel behind it. tt-metal fronts every
            // bank on its own NoC coordinate on both parts, so two
            // sub-endpoints of one tile are not distinguishable from a host
            // program at all. The `samecore` arm is the thing that would
            // distinguish them and it does not exist on either part here.
            int witness_ok = -1, stray_writes = -1;
            if (p.is_write) {
                witness_ok = 0;
                stray_writes = 0;
                std::set<std::pair<uint32_t, uint32_t>> targeted;
                for (uint32_t i = 0; i < p.num_readers; i++) {
                    targeted.insert({p.bank_of_reader[i], p.slice_of_reader[i]});
                }
                std::map<std::pair<uint32_t, uint32_t>, uint32_t> seen;
                for (const Bank& b : banks) {
                    for (uint32_t s = 0; s < slices; ++s) {
                        std::vector<uint32_t> got;
                        detail::ReadFromDeviceDRAMChannel(device, (int)b.id, block_addr(s),
                                                          DRAMRATEBENCH_TAG_BYTES, got);
                        const uint32_t word = got.empty() ? 0u : got[0];
                        seen[{b.id, s}] = word;
                        if (targeted.count({b.id, s}) == 0 && word != DRAMRATEBENCH_POISON) {
                            stray_writes++;
                        }
                    }
                }
                for (uint32_t i = 0; i < p.num_readers; i++) {
                    const auto key = std::make_pair(p.bank_of_reader[i], p.slice_of_reader[i]);
                    if (seen[key] == witness_of[i]) {
                        witness_ok++;
                    }
                }
                if (witness_ok != (int)p.num_readers || stray_writes != 0) {
                    fprintf(stderr,
                            "dramratebench: point %zu (%s, n=%u): witness_ok=%d of %u, "
                            "stray_writes=%d\n"
                            "  The writers were not all writing where the plan names, or bytes\n"
                            "  landed on a block the plan never named. The row is written anyway\n"
                            "  so the numbers can be seen, and it is CONDEMNED by the verdict.\n",
                            pi, p.arm.c_str(), p.num_readers, witness_ok, p.num_readers,
                            stray_writes);
                }
            }

            const uint64_t total_bytes = (uint64_t)p.num_readers * bytes_per_reader;
            const double agg = aggregate_bytes_per_cycle(total_bytes, max_cycles);
            p.measured = true;
            p.max_cycles = max_cycles;
            p.min_cycles = min_cycles;
            p.max_spins = max_spins;
            p.tags_ok = tags_ok;
            p.witness_ok = witness_ok;
            p.stray_writes = stray_writes;
            p.agg_bytes_per_cycle = agg;
            p.agg_samples.push_back(agg);
            fprintf(out, "%s,%u,%zu,%u,%u,%u,%u,%llu,%u,%u,%.4f,%.3f,%.4f,%u,%u,%u,%u,%u,%s,%d,%d\n",
                    p.arm.c_str(), rep, pi, p.num_readers, p.num_tx, p.tx_bytes, bytes_per_reader,
                    (unsigned long long)total_bytes, max_cycles, min_cycles, agg,
                    agg * (double)clock_mhz / 1000.0, agg / (double)p.num_readers,
                    p.distinct_banks, p.distinct_dram_cores, tags_ok, max_spins, measured,
                    p.is_write ? "write" : "read", witness_ok, stray_writes);
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
    uint32_t bad_tags = 0, points = 0, overlapped = 0, multi = 0, bad_witness = 0, strays = 0,
             write_points = 0;
    for (const Point& p : plan) {
        if (!p.measured) {
            continue;
        }
        points++;
        if (p.tags_ok != p.num_readers) {
            bad_tags++;
        }
        if (p.is_write) {
            write_points++;
            // Counted separately from `bad_tags` because they fail differently
            // and mean different things: `tags_ok` is the DEVICE's readback and
            // catches a write that never landed, `witness_ok` and
            // `stray_writes` are the HOST's and catch a write that landed
            // somewhere else. Folding them together would leave a reader unable
            // to tell "dropped" from "misdirected", which is the whole
            // diagnosis.
            if (p.witness_ok != (int)p.num_readers) {
                bad_witness++;
            }
            if (p.stray_writes > 0) {
                strays++;
            }
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

    // --- the sustained rate, per reader count -------------------------------
    // THE LEVEL, which the scaling verdict below cannot see. A run whose one
    // channel plateaus at 24 B/cycle and a run whose one channel plateaus at
    // 40 give the SAME ratio, and only one of them agrees with the model:
    // `dram.channel_serialisation.bytes_per_cycle` is 24 on Wormhole, so a
    // saturated one-channel aggregate cannot exceed it however many readers
    // push. Printed as data, before any verdict, and printed even when a gate
    // has failed -- a reader who can see the numbers can see what failed.
    printf("\ndramratebench: SUSTAINED RATE -- N readers, %u B each, ONE channel\n", bytes_per_reader);
    printf("  readers   aggregate B/cycle   per reader   aggregate GB/s   fanchan B/cycle\n");
    for (uint32_t n : sweep) {
        const Point* o = best("onechan", n);
        if (o == nullptr) {
            continue;
        }
        const Point* f = best("fanchan", n);
        const double agg = median_of(o->agg_samples);
        printf("  %7u   %17.3f   %10.3f   ", n, agg, agg / (double)n);
        if (clock_mhz > 0) {
            printf("%12.2f   ", agg * (double)clock_mhz / 1000.0);
        } else {
            printf("%12s   ", "-");
        }
        if (f != nullptr) {
            printf("%14.3f\n", median_of(f->agg_samples));
        } else {
            printf("%14s\n", "-");
        }
    }
    if (clock_mhz <= 0) {
        printf("  (this device reports no clock, so no GB/s column is shown: a GB/s converted\n"
               "   at a frequency the device did not report is not a reading. Use B/cycle.)\n");
    }
    if (arch == "wormhole" && clock_mhz > 0) {
        printf("  Against wh_dram#performance (reads, static VC, 1 MiB per tile, ONE channel):\n");
        printf("    readers   published GB/s   measured GB/s   deviation\n");
        for (size_t v = 0; v < 3; v++) {
            const Point* o = best("onechan", VENDOR_READERS[v]);
            if (o == nullptr) {
                printf("    %7u   %14.1f   %13s   %9s  (not swept)\n", VENDOR_READERS[v],
                       VENDOR_GB_PER_S[v], "-", "-");
                continue;
            }
            const double got = median_of(o->agg_samples) * (double)clock_mhz / 1000.0;
            printf("    %7u   %14.1f   %13.2f   %+8.1f%%\n", VENDOR_READERS[v], VENDOR_GB_PER_S[v],
                   got, 100.0 * (got - VENDOR_GB_PER_S[v]) / VENDOR_GB_PER_S[v]);
        }
    } else if (arch == "wormhole") {
        printf("  wh_dram#performance publishes 22.2 / 22.3 / 22.3 GB/s at 1 / 12 / 48 tiles, and\n"
               "  this run cannot be compared to it: the device reported no clock, so only\n"
               "  B/cycle is a reading here. 24 B/cycle is the channel rate the table converts\n"
               "  from at 1 GHz, and 22.2 is 92%% of it -- the page's own achievable fraction.\n");
    } else {
        printf("  No published table applies: wh_dram#performance is a WORMHOLE page, and this\n"
               "  part is %s. Blackhole publishes no DRAM tile page at all -- no per-channel\n"
               "  rate, no address map -- so there is nothing here to compare against and\n"
               "  nothing measured here can supply it.\n",
               arch.c_str());
    }
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
                   "  tt-sim, which models no NoC buffer back-pressure or virtual channels and,\n"
                   "  on Blackhole, publishes no per-channel DRAM rate for the endpoint queue to\n"
                   "  be built from. (It DOES model link congestion, so that is not the gap.)\n",
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

    // --- the write direction ------------------------------------------------
    // Printed after the read verdict and never mixed into it. The read arm has
    // a card control to reproduce and its verdict text is what other results in
    // this project are read against; a write arm that changed a single word of
    // it would make the two sessions incomparable for a reason that has nothing
    // to do with DRAM.
    if (write_points > 0) {
        printf("\ndramratebench: WRITE DIRECTION -- %u point(s)\n", write_points);
        printf("  addressing: %u point(s) failed the host's witness check (bytes did not\n"
               "  arrive where the plan named); %u point(s) had a STRAY write (bytes arrived\n"
               "  on a block the plan never named)\n",
               bad_witness, strays);
        const Point* w_one_lo = best("onechan-write", n_lo);
        const Point* w_one_hi = best("onechan-write", n_hi);
        const Point* w_fan_lo = best("fanchan-write", n_lo);
        const Point* w_fan_hi = best("fanchan-write", n_hi);
        printf("\n  writers   aggregate B/cycle   per writer   fanchan-write B/cycle   read "
               "onechan\n");
        for (uint32_t n : sweep) {
            const Point* o = best("onechan-write", n);
            if (o == nullptr) {
                continue;
            }
            const Point* f = best("fanchan-write", n);
            const Point* r = best("onechan", n);
            const double agg = median_of(o->agg_samples);
            printf("  %7u   %17.3f   %10.3f   ", n, agg, agg / (double)n);
            if (f != nullptr) {
                printf("%21.3f   ", median_of(f->agg_samples));
            } else {
                printf("%21s   ", "-");
            }
            if (r != nullptr) {
                printf("%12.3f\n", median_of(r->agg_samples));
            } else {
                printf("%12s\n", "-");
            }
        }
        if (bad_witness != 0 || strays != 0) {
            printf("\n  VERDICT (write): DEGENERATE -- %u point(s) could not prove the bytes went\n"
                   "  where the plan said and %u had bytes arrive somewhere it did not. A write\n"
                   "  that misses its endpoint completes, retires its barrier and yields a\n"
                   "  perfectly plausible rate, so no number in the write rows means what it\n"
                   "  says. Send the CSV home anyway; this is diagnosable.\n",
                   bad_witness, strays);
        } else if (w_one_lo == nullptr || w_one_hi == nullptr || w_fan_lo == nullptr ||
                   w_fan_hi == nullptr || n_hi == n_lo) {
            printf("\n  VERDICT (write): DEGENERATE -- the write direction has no low/high pair\n"
                   "  in both arms, so nothing separates the endpoint from anything upstream.\n"
                   "  EXPECTED against tt-sim, which cannot sweep the writer count far.\n");
        } else {
            const double wf = w_fan_lo->agg_bytes_per_cycle > 0
                                  ? w_fan_hi->agg_bytes_per_cycle / w_fan_lo->agg_bytes_per_cycle
                                  : 0.0;
            const double wo = w_one_lo->agg_bytes_per_cycle > 0
                                  ? w_one_hi->agg_bytes_per_cycle / w_one_lo->agg_bytes_per_cycle
                                  : 0.0;
            printf("\n  fanchan-write %.3f -> %.3f B/cycle over %u -> %u writers (x%.2f)\n",
                   w_fan_lo->agg_bytes_per_cycle, w_fan_hi->agg_bytes_per_cycle, n_lo, n_hi, wf);
            printf("  onechan-write %.3f -> %.3f B/cycle over %u -> %u writers (x%.2f)\n",
                   w_one_lo->agg_bytes_per_cycle, w_one_hi->agg_bytes_per_cycle, n_lo, n_hi, wo);
            if (wf < 1.5) {
                printf("  VERDICT (write): DEGENERATE -- the fan-out CONTROL did not move (x%.2f).\n"
                       "  Something upstream of the endpoint caps both write arms, so a flat\n"
                       "  one-endpoint write curve says nothing about the endpoint.\n",
                       wf);
            } else {
                const double weff = wf > 0 ? wo / wf : 1.0;
                printf("  VERDICT (write): %s -- concentrated writers reached %.0f%% of the\n"
                       "  scaling the same writers reached fanned out.\n",
                       weff <= 0.75 ? "ENDPOINT BOUND" : "NO ENDPOINT BOUND", 100.0 * weff);
            }
        }

        // THE COMPARISON THIS DIRECTION EXISTS FOR, and the only one it can
        // make. Not "what does a write cost" -- that is a latency question and
        // a rate campaign cannot answer it -- but "is a saturated endpoint's
        // sustained WRITE rate the same as its sustained READ rate". The two
        // arms differ in the direction of the transfer and in nothing else:
        // same cores, same barrier, same transaction size, same issue loop,
        // same bank, same slice stride.
        if (w_one_hi != nullptr && one_hi != nullptr && one_hi->agg_bytes_per_cycle > 0) {
            const double wr = median_of(w_one_hi->agg_samples);
            const double rd = median_of(one_hi->agg_samples);
            const double ratio = rd > 0 ? wr / rd : 0.0;
            printf("\n  SUSTAINED WRITE vs READ at the SAME endpoint, %u participants:\n", n_hi);
            printf("    read  %.3f B/cycle\n    write %.3f B/cycle   ratio %.3f\n", rd, wr, ratio);
            if (ratio > 1.05) {
                printf("    -> the endpoint sustains writes FASTER than reads. Read the level,\n"
                       "       not just the ratio: a write arm that plateaus at the NoC link's\n"
                       "       rate rather than at a channel rate has found the link.\n");
            } else if (ratio < 0.95) {
                printf("    -> the endpoint sustains writes SLOWER than reads, which is the\n"
                       "       direction Wormhole's own latency rows lean and the OPPOSITE of\n"
                       "       Blackhole's published-derived split.\n");
            } else {
                printf("    -> the endpoint sustains writes and reads at the same rate to within\n"
                       "       5%%. A rate campaign cannot then distinguish the two directions'\n"
                       "       OCCUPANCY, whatever their latencies differ by.\n");
            }
            printf("    This is an OCCUPANCY comparison and not a latency one. It cannot price a\n"
                   "    single write, cannot split service time from queueing, and -- with the\n"
                   "    samecore arm absent on this part -- cannot tell the endpoint from the one\n"
                   "    inbound router link every concentrated flow converges on. Report \"the\n"
                   "    endpoint\", never \"the channel\". CORROBORATION, never provenance.\n");
        }
    }
    return failures == 0 ? 0 : 1;
}
