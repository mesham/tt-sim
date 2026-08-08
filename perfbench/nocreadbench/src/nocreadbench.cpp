// nocreadbench -- what limits the sustained NoC *read* rate?
//
// WHAT THIS IS FOR
// ----------------
// tt-metal's shipped `noc_latencies.yaml` says a pipelined burst of L1 reads
// costs 25 cycles per transaction on Wormhole and 35 on Blackhole, at every
// transaction size from 64 B up to the point where the link binds, and in both
// geometries it measures. tt-sim's reconstruction of the issue loop costs 18
// and 19. The roadmap named the difference the initiator's
// outstanding-read-request credit limit and recorded it as unpublished.
//
// Neither the public ISA documentation nor tt-metal's headers bound the number
// of outstanding read requests per architecture -- both were swept, and both
// negatives are written up in `docs/plans/cost-model.md`. So the number cannot
// be sourced, and the instalment's own statement of why it cannot even be
// derived is the thing this program exists to fix:
//
//     `vendor_source_derived` is not available because there is no arithmetic
//     that separates the credit limit from the L1 read ports.
//
// No arithmetic does. A measurement does, two ways:
//
//   * DIRECTLY. `NIU_MST_REQS_OUTSTANDING_ID(0)` is a readable NIU counter of
//     the initiator's own in-flight requests. Sample it during a burst. If it
//     plateaus below the burst length, the initiator holds a bounded number in
//     flight and that bound is read off, not inferred. If it climbs with the
//     burst length, there is no initiator-side limit and the cap is downstream
//     -- which retires the term rather than sizing it.
//
//   * BY SHAPE. A round-trip credit limit K makes the sustained rate L/K, so
//     it must move when the round trip L moves: with hop distance (E1) and
//     with nothing else. A responder-side L1 read port must move when the
//     number of distinct responders moves (E2) and with nothing else. An
//     initiator-side per-request occupancy moves with neither. E3 and E4 strip
//     out the two bank-conflict confounds that tt-metal's own dataset bakes in
//     and never varies, because every transaction in it reads the same address
//     into the same address.
//
// The full prediction table -- what each experiment reads under each
// hypothesis, INCLUDING what "no effect" looks like -- is in
// perfbench/nocreadbench/README.md and was written before anything was run.
//
// Nothing here writes to a cost table. A number this program produces is a
// measurement on one part and can enter `unit_costs.yaml` only as
// `corroboration`, never as provenance.
//
//   nocreadbench [--out FILE] [--num-tx N] [--repeats R] [--no-sample]
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "kernels/nocreadbench_layout.h"

using namespace tt;
using namespace tt::tt_metal;

namespace {

const char* arch_name(IDevice* device) {
    switch (device->arch()) {
        case ARCH::WORMHOLE_B0: return "wormhole";
        case ARCH::BLACKHOLE: return "blackhole";
        default: return "unknown";
    }
}

// One measured point. `experiment` names the axis; everything else is the key.
struct Point {
    std::string experiment;
    CoreCoord master;                 // logical
    std::vector<CoreCoord> sources;   // logical
    uint32_t num_tx = 0;
    uint32_t tx_bytes = 0;
    uint32_t dst_stride = 0;
    uint32_t src_stride = 0;
    uint32_t hops = 0;  // planned Manhattan-ish distance, for E1's abscissa
    std::vector<uint32_t> result;
    bool measured = false;
};

// The directional-torus hop count both NoCs use: NoC 0 increases x then y,
// wrapping, so the distance from a to b along each axis is a one-way count.
uint32_t hop_count(CoreCoord a, CoreCoord b, uint32_t grid_x, uint32_t grid_y) {
    const uint32_t dx = (b.x + grid_x - a.x) % grid_x;
    const uint32_t dy = (b.y + grid_y - a.y) % grid_y;
    return dx + dy;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path;
    uint32_t num_tx = 128;  // <= 128: NIU_MST_REQS_OUTSTANDING_ID is 8 bits and
                            // the ISA docs warn it wraps if software has too
                            // many outstanding requests.
    uint32_t repeats = 3;
    uint32_t sample = 1;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--num-tx" && i + 1 < argc) {
            num_tx = (uint32_t)atoi(argv[++i]);
        } else if (a == "--repeats" && i + 1 < argc) {
            repeats = (uint32_t)atoi(argv[++i]);
        } else if (a == "--no-sample") {
            sample = 0;
        } else {
            fprintf(stderr, "nocreadbench: unknown argument '%s'\n", a.c_str());
            return 2;
        }
    }

    IDevice* device = CreateDevice(0);
    const std::string arch = arch_name(device);
    const CoreCoord grid = device->compute_with_storage_grid_size();
    if (grid.x < 4 || grid.y < 2) {
        fprintf(stderr, "nocreadbench: need at least a 4x2 worker grid, have %ux%u\n",
                (unsigned)grid.x, (unsigned)grid.y);
        CloseDevice(device);
        return 1;
    }
    if (out_path.empty()) {
        out_path = "nocreadbench-" + arch + ".csv";
    }

    const CoreCoord master{0, 0};
    // Physical coordinates, for the hop counts. Logical-to-physical is not
    // affine on a harvested part, so every distance is computed in the space
    // the routers actually use.
    const CoreCoord m_phys = device->worker_core_from_logical_core(master);

    // --- the plan ----------------------------------------------------------
    // Every experiment holds everything fixed but one axis. num_tx, the NoC,
    // the initiator core and the barrier structure never vary.
    std::vector<Point> plan;
    const uint32_t base_bytes = 64;

    // E7 `size`: the control. Establishes where the link bound starts, so the
    // others can be read as sub-link-bound. Expected: flat, then linear in
    // bytes above ~512 B (Wormhole, 32 B/cycle flit) / ~2048 B (Blackhole,
    // 64 B/cycle).
    for (uint32_t b : {64u, 128u, 256u, 512u, 1024u, 2048u, 4096u}) {
        Point p;
        p.experiment = "size";
        p.master = master;
        p.sources = {CoreCoord{(uint32_t)grid.x - 1, 0}};
        p.num_tx = num_tx;
        p.tx_bytes = b;
        plan.push_back(p);
    }

    // E1 `dist`: one source, swept over distance. THE credit-limit axis.
    for (uint32_t x = 1; x < grid.x; x++) {
        Point p;
        p.experiment = "dist";
        p.master = master;
        p.sources = {CoreCoord{x, 0}};
        p.num_tx = num_tx;
        p.tx_bytes = base_bytes;
        plan.push_back(p);
    }
    for (uint32_t y = 1; y < grid.y; y++) {
        Point p;
        p.experiment = "dist";
        p.master = master;
        p.sources = {CoreCoord{0, y}};
        p.num_tx = num_tx;
        p.tx_bytes = base_bytes;
        plan.push_back(p);
    }

    // E2 `srcfan`: the same burst spread over S distinct source tiles. THE
    // responder-side axis, and the one the shipped dataset cannot provide.
    for (uint32_t s : {1u, 2u, 4u, 8u}) {
        if (s > (uint32_t)grid.x - 1) {
            continue;
        }
        Point p;
        p.experiment = "srcfan";
        p.master = master;
        for (uint32_t i = 0; i < s; i++) {
            p.sources.push_back(CoreCoord{1 + i, 0});
        }
        p.num_tx = num_tx;
        p.tx_bytes = base_bytes;
        plan.push_back(p);
    }

    // E3 `dstspread`: where the returning data lands at the INITIATOR. The
    // shipped dataset lands every transaction at one address; this varies the
    // stride from 0 (all one bank) to the transaction size (consecutive banks).
    for (uint32_t stride : {0u, 64u, 128u, 512u}) {
        Point p;
        p.experiment = "dstspread";
        p.master = master;
        p.sources = {CoreCoord{(uint32_t)grid.x - 1, 0}};
        p.num_tx = num_tx;
        p.tx_bytes = base_bytes;
        p.dst_stride = stride;
        plan.push_back(p);
    }

    // E4 `srcspread`: where the data is READ FROM within one responder tile.
    for (uint32_t stride : {0u, 64u, 128u, 512u}) {
        Point p;
        p.experiment = "srcspread";
        p.master = master;
        p.sources = {CoreCoord{(uint32_t)grid.x - 1, 0}};
        p.num_tx = num_tx;
        p.tx_bytes = base_bytes;
        p.src_stride = stride;
        plan.push_back(p);
    }

    // E6 `burst`: the N axis, the one the shipped dataset does have. Kept so
    // that this run can be checked against `noc_latencies.yaml` before any of
    // the new axes are believed.
    for (uint32_t n : {4u, 16u, 64u, 128u}) {
        Point p;
        p.experiment = "burst";
        p.master = master;
        p.sources = {CoreCoord{(uint32_t)grid.x - 1, 0}};
        p.num_tx = n;
        p.tx_bytes = base_bytes;
        plan.push_back(p);
    }

    // --- L1 scratch, at one address common to every core --------------------
    uint32_t max_bytes = 0;
    for (const Point& p : plan) {
        max_bytes = std::max(max_bytes, p.tx_bytes);
    }
    // Room for the largest transaction plus the widest stride sweep, doubled
    // because the kernel splits the arena into a source half and a landing
    // half that must never overlap.
    const uint32_t arena = 2 * std::max(max_bytes * 2, 8192u);
    InterleavedBufferConfig data_cfg{
        .device = device, .size = arena, .page_size = arena, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> data_scratch = CreateBuffer(data_cfg);
    const uint32_t data_addr = data_scratch->address();

    constexpr uint32_t result_bytes = NOCREADBENCH_R_WORDS * 4;
    InterleavedBufferConfig res_cfg{
        .device = device, .size = result_bytes, .page_size = result_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> result_scratch = CreateBuffer(res_cfg);
    const uint32_t results_addr = result_scratch->address();

    printf("nocreadbench: arch=%s grid=%ux%u points=%zu repeats=%u num_tx=%u "
           "data_addr=0x%08X results_addr=0x%08X\n",
           arch.c_str(), (unsigned)grid.x, (unsigned)grid.y, plan.size(), repeats, num_tx,
           data_addr, results_addr);
    fflush(stdout);

    FILE* out = fopen(out_path.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "nocreadbench: cannot write '%s'\n", out_path.c_str());
        CloseDevice(device);
        return 1;
    }
    fprintf(out, "# nocreadbench arch=%s grid=%ux%u num_tx=%u repeats=%u\n",
            arch.c_str(), (unsigned)grid.x, (unsigned)grid.y, num_tx, repeats);
    fprintf(out, "# every row is one timed burst; cycles_per_tx = cycles / num_tx\n");
    fprintf(out,
            "experiment,repeat,point,mst_x,mst_y,mst_node_x,mst_node_y,num_src,src0_x,src0_y,"
            "hops,num_tx,tx_bytes,dst_stride,src_stride,cycles,cycles_per_tx,"
            "outstanding_max,outstanding_end,samples,cmdbuf_avail_rest,cmdbuf_avail_busy\n");

    size_t failures = 0;
    for (uint32_t rep = 0; rep < repeats; rep++) {
        for (size_t pi = 0; pi < plan.size(); pi++) {
            Point& p = plan[pi];

            std::vector<CoreCoord> src_phys;
            for (const CoreCoord& c : p.sources) {
                src_phys.push_back(device->worker_core_from_logical_core(c));
            }
            p.hops = hop_count(m_phys, src_phys[0], (uint32_t)grid.x + 2, (uint32_t)grid.y + 2);

            std::vector<CoreRange> ranges{CoreRange(p.master)};
            for (const CoreCoord& c : p.sources) {
                ranges.push_back(CoreRange(c));
            }
            CoreRangeSet all_cores(std::set<CoreRange>(ranges.begin(), ranges.end()));

            Program program = CreateProgram();
            KernelHandle k = CreateKernel(
                program,
                "kernels/dataflow/reader.cpp",
                p.master,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0});

            std::vector<uint32_t> args(NOCREADBENCH_A_SRC, 0);
            args[NOCREADBENCH_A_RESULTS] = results_addr;
            args[NOCREADBENCH_A_DATA] = data_addr;
            args[NOCREADBENCH_A_DATA_BYTES] = arena;
            args[NOCREADBENCH_A_POINT] = (uint32_t)pi;
            args[NOCREADBENCH_A_NUM_TX] = p.num_tx;
            args[NOCREADBENCH_A_TX_BYTES] = p.tx_bytes;
            args[NOCREADBENCH_A_DST_STRIDE] = p.dst_stride;
            args[NOCREADBENCH_A_SRC_STRIDE] = p.src_stride;
            args[NOCREADBENCH_A_NUM_SRC] = (uint32_t)p.sources.size();
            args[NOCREADBENCH_A_NUM_TRID] = 0;
            args[NOCREADBENCH_A_SAMPLE] = sample;
            for (const CoreCoord& c : src_phys) {
                args.push_back((uint32_t)c.x);
                args.push_back((uint32_t)c.y);
            }
            SetRuntimeArgs(program, k, p.master, args);

            std::vector<uint32_t> zeros(NOCREADBENCH_R_WORDS, 0);
            detail::WriteToDeviceL1(device, p.master, results_addr, zeros);
            detail::LaunchProgram(device, program, true, true);
            detail::ReadFromDeviceL1(device, p.master, results_addr, result_bytes, p.result);

            p.measured = p.result.size() > NOCREADBENCH_R_MAGIC &&
                         p.result[NOCREADBENCH_R_MAGIC] == NOCREADBENCH_MAGIC;
            if (!p.measured) {
                failures++;
                fprintf(stderr, "nocreadbench: point %zu (%s): no result stamp\n", pi, p.experiment.c_str());
                continue;
            }
            const uint32_t cycles = p.result[NOCREADBENCH_R_CYCLES];
            fprintf(out,
                    "%s,%u,%zu,%u,%u,%u,%u,%zu,%u,%u,%u,%u,%u,%u,%u,%u,%.3f,%u,%u,%u,0x%08X,0x%08X\n",
                    p.experiment.c_str(), rep, pi,
                    (unsigned)p.master.x, (unsigned)p.master.y,
                    p.result[NOCREADBENCH_R_NODE_X], p.result[NOCREADBENCH_R_NODE_Y],
                    p.sources.size(), (unsigned)src_phys[0].x, (unsigned)src_phys[0].y,
                    p.hops, p.num_tx, p.tx_bytes, p.dst_stride, p.src_stride,
                    cycles, (double)cycles / (double)p.num_tx,
                    p.result[NOCREADBENCH_R_OUTSTANDING_MAX],
                    p.result[NOCREADBENCH_R_OUTSTANDING_END],
                    p.result[NOCREADBENCH_R_SAMPLES],
                    p.result[NOCREADBENCH_R_CMDBUF_AVAIL_REST],
                    p.result[NOCREADBENCH_R_CMDBUF_AVAIL_BUSY]);
            fflush(out);
        }
    }
    fclose(out);
    CloseDevice(device);

    // --- the verdict, stated in the terms the hypotheses are stated in ------
    // Deliberately not a pass/fail: this program's job is to say which
    // mechanism the numbers are consistent with, and "none of them" has to be
    // expressible. The arithmetic is left to
    // `python3 -m tt_sim.perf.noc_dataset_sweep --arch <arch>` plus the
    // README's table; what is printed here is only the one reading that needs
    // no arithmetic at all.
    printf("nocreadbench: wrote %s (%zu failures)\n", out_path.c_str(), failures);
    if (sample != 0) {
        uint32_t worst = 0;
        for (const Point& p : plan) {
            if (p.measured) {
                worst = std::max(worst, p.result[NOCREADBENCH_R_OUTSTANDING_MAX]);
            }
        }
        printf("nocreadbench: highest NIU_MST_REQS_OUTSTANDING_ID(0) seen = %u (burst length %u)\n",
               worst, num_tx);
        if (worst == 0) {
            printf("  VERDICT: DEGENERATE -- the counter never moved. Either the sampling\n"
                   "  load is being hoisted, or reads are completing before the next sample.\n"
                   "  Do not read anything into the rate columns until this is non-zero.\n");
        } else if (worst + 4 >= num_tx) {
            printf("  VERDICT: NO INITIATOR LIMIT -- in-flight requests track the burst\n"
                   "  length, so the initiator is not holding a bounded number in flight and\n"
                   "  the sustained rate is capped downstream of it. The credit-limit term\n"
                   "  is retired, not sized.\n");
        } else {
            printf("  VERDICT: BOUNDED AT %u -- the initiator holds at most this many read\n"
                   "  requests in flight. Check it against the `dist` rows: a credit limit\n"
                   "  predicts cycles_per_tx = round_trip / %u, so cycles_per_tx MUST rise\n"
                   "  with hops. If the `dist` rows are flat, this bound is not what caps\n"
                   "  the rate either.\n", worst, worst);
        }
    }
    return failures == 0 ? 0 : 1;
}
