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
// THE SECOND QUESTION, AND WHY --stateful EXISTS
// ----------------------------------------------
// On a real Wormhole part on 2026-08-17 the `burst` control measured a marginal
// 44.0 cycles per transaction -- 44.08 at N = 4 -> 16, 44.00 at 16 -> 64, 43.97
// at 64 -> 128, flat to 0.25 % over a 32x range, and flat to 1.0 % over hops
// 1..7. The shipped dataset's own figure for the same shape is 25.00. The
// disagreement is real; its CAUSE is ambiguous, and only two candidates survive
// the flatness:
//
//   * a HARDWARE FLOOR this part has and the dataset says it does not, or
//   * OUR ISSUE LOOP IS SIMPLY LONGER than the dataset's, in which case 44 is a
//     property of this program and says nothing about the part.
//
// The dataset's own `stateful` rows are the discriminator, because they are the
// same experiment with a SHORTER ISSUE LOOP: on Wormhole they land 7.67 cycles
// below the long loop's figure, which is that dataset's evidence that Wormhole
// has no per-read floor; on Blackhole they buy 1.0, which is its evidence that
// Blackhole does. `--stateful` runs that loop here. The predictions, written
// down before any card ran it, are in README.md under "The stateful variant".
//
// The default is UNCHANGED and must stay so: it is the arm the 2026-08-17
// session ran, and a variant is only worth anything against a control that
// reproduces. The two arms share one kernel body, `if constexpr` on the mode.
//
// Nothing here writes to a cost table. A number this program produces is a
// measurement on one part and can enter `unit_costs.yaml` only as
// `corroboration`, never as provenance.
//
//   nocreadbench [--out FILE] [--num-tx N] [--repeats R] [--no-sample]
//                [--stateful] [--only EXPERIMENT[,EXPERIMENT...]]
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
    // The mode witness, resolved at write-out time so the CSV, the verdict and
    // the exit status can never disagree about what the kernel actually ran.
    uint32_t sig_src = 0;      // signature the host stamped at sources[0]
    uint32_t sig_witness = 0;  // and at the witness core
    bool mode_confirmed = false;
    // Derived at write-out time and reused by the verdict, so the verdict and
    // the CSV can never disagree about what was measured.
    uint32_t outstanding_delta = 0;  // max - rest; the occupancy, not the raw counter
    uint32_t inflight_max = 0;       // the transaction-id-independent cross-check
    bool cmdbuf_moved = false;       // did CMD_BUF_AVAIL ever leave its rest value
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
    uint32_t mode = NOCREADBENCH_MODE_STATELESS;
    std::string only;  // comma-separated experiment names; empty = all
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
        } else if (a == "--stateful") {
            mode = NOCREADBENCH_MODE_STATEFUL;
        } else if (a == "--only" && i + 1 < argc) {
            only = "," + std::string(argv[++i]) + ",";
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

    // Recorded in the artefact, not left to whoever reads it. A simulator CSV
    // read against the card's control band produces a "CONTROL MOVED" verdict
    // from two numbers that were never the same measurement, and `check_mode.py`
    // needs to be able to tell them apart with nothing but the file.
    const bool on_sim = getenv("TT_METAL_SIMULATOR") != nullptr;

    const CoreCoord master{0, 0};
    // Physical coordinates, for the hop counts. Logical-to-physical is not
    // affine on a harvested part, so every distance is computed in the space
    // the routers actually use.
    const CoreCoord m_phys = device->worker_core_from_logical_core(master);

    // The mode witness. Every source this program uses sits on row 0 or column
    // 0 (see the plan below), and the initiator is (0, 0), so logical (1, 1) is
    // never either -- checked per point rather than argued, because a witness
    // that coincided with the source would make the probe read the same
    // signature under both modes and quietly stop discriminating.
    const CoreCoord witness{1, 1};
    const CoreCoord w_phys = device->worker_core_from_logical_core(witness);
    const uint32_t sig_witness = NOCREADBENCH_SIG((uint32_t)w_phys.x, (uint32_t)w_phys.y);
    const char* mode_name = (mode == NOCREADBENCH_MODE_STATEFUL) ? "stateful" : "stateless";

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

    // --- the two filters ----------------------------------------------------
    // `--only` is for the simulator side, where 36 points at a few tens of
    // thousands of cycles a second is minutes and the `burst` axis is the one
    // the variant is read off. It changes nothing about a point that survives.
    const size_t planned = plan.size();
    if (!only.empty()) {
        std::vector<Point> kept;
        for (const Point& p : plan) {
            if (only.find("," + p.experiment + ",") != std::string::npos) {
                kept.push_back(p);
            }
        }
        plan = kept;
        if (plan.empty()) {
            fprintf(stderr, "nocreadbench: --only matched no experiment (have: size dist srcfan dstspread srcspread burst)\n");
            CloseDevice(device);
            return 2;
        }
    }
    // A stateful loop holds the source TILE in the read command buffer, so it
    // cannot cycle sources without paying back the store the variant exists to
    // remove. Those points are DROPPED rather than run against one source under
    // the srcfan name -- and the kernel refuses them independently, so a drop
    // that failed to happen still cannot be measured.
    size_t dropped_multisrc = 0;
    if (mode == NOCREADBENCH_MODE_STATEFUL) {
        std::vector<Point> kept;
        for (const Point& p : plan) {
            if (p.sources.size() > 1) {
                dropped_multisrc++;
                continue;
            }
            kept.push_back(p);
        }
        plan = kept;
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
    // One machine-readable line, in the shape `nocevbench-config` has, so a
    // checker reading only the logs can tell what the run was CONFIGURED as --
    // and then go and disagree with it from the data if the kernel did
    // something else.
    printf("nocreadbench-config arch=%s mode=%s sim=%u points=%zu of=%zu dropped_multisrc=%zu "
           "witness=%u,%u witness_sig=0x%08X num_tx=%u repeats=%u sample=%u only=%s\n",
           arch.c_str(), mode_name, on_sim ? 1u : 0u, plan.size(), planned, dropped_multisrc,
           (unsigned)witness.x, (unsigned)witness.y, sig_witness, num_tx, repeats, sample,
           only.empty() ? "-" : only.substr(1, only.size() - 2).c_str());
    if (dropped_multisrc != 0) {
        printf("nocreadbench: %zu srcfan point(s) dropped -- the stateful issue path holds the\n"
               "  source tile in the command buffer and cannot cycle sources without paying\n"
               "  back the store this variant exists to remove.\n",
               dropped_multisrc);
    }
    fflush(stdout);

    FILE* out = fopen(out_path.c_str(), "w");
    if (out == nullptr) {
        fprintf(stderr, "nocreadbench: cannot write '%s'\n", out_path.c_str());
        CloseDevice(device);
        return 1;
    }
    fprintf(out, "# nocreadbench arch=%s grid=%ux%u num_tx=%u repeats=%u mode=%s sim=%u\n",
            arch.c_str(), (unsigned)grid.x, (unsigned)grid.y, num_tx, repeats, mode_name,
            on_sim ? 1u : 0u);
    fprintf(out, "# every row is one timed burst; cycles_per_tx = cycles / num_tx\n");
    fprintf(out, "# mode is what the KERNEL stamped; probe_word says which tile answered a\n");
    fprintf(out, "# transaction issued with the state pointed elsewhere, so it is stateless\n");
    fprintf(out, "# iff probe_word == sig_src and stateful iff probe_word == sig_witness\n");
    fprintf(out,
            "experiment,repeat,point,mst_x,mst_y,mst_node_x,mst_node_y,num_src,src0_x,src0_y,"
            "hops,num_tx,tx_bytes,dst_stride,src_stride,cycles,cycles_per_tx,"
            "outstanding_max,outstanding_end,samples,cmdbuf_avail_rest,cmdbuf_avail_busy,"
            "outstanding_rest,outstanding_delta,inflight_max,inflight_rest,trid,"
            "cmdbuf_avail_max,cmdbuf_ovfl_rest,cmdbuf_ovfl_end,"
            "mode,probe_word,sig_src,sig_witness\n");

    size_t failures = 0;
    for (uint32_t rep = 0; rep < repeats; rep++) {
        for (size_t pi = 0; pi < plan.size(); pi++) {
            Point& p = plan[pi];

            std::vector<CoreCoord> src_phys;
            for (const CoreCoord& c : p.sources) {
                src_phys.push_back(device->worker_core_from_logical_core(c));
            }
            p.hops = hop_count(m_phys, src_phys[0], (uint32_t)grid.x + 2, (uint32_t)grid.y + 2);
            p.sig_src = NOCREADBENCH_SIG((uint32_t)src_phys[0].x, (uint32_t)src_phys[0].y);
            p.sig_witness = sig_witness;
            if (p.sig_src == p.sig_witness) {
                fprintf(stderr,
                        "nocreadbench: point %zu (%s): the witness core (%u,%u) IS the source, so the\n"
                        "  mode probe cannot discriminate. Refusing rather than reporting an\n"
                        "  unverifiable mode.\n",
                        pi, p.experiment.c_str(), (unsigned)w_phys.x, (unsigned)w_phys.y);
                failures++;
                continue;
            }
            // Stamp the signatures the probe reads back. Written EVERY launch,
            // not once, because the reads land in the same arena and a previous
            // point's landings may have crossed into the source half on a core
            // that was both source and landing target in some other geometry.
            std::vector<uint32_t> src_sig(NOCREADBENCH_SIG_WORDS, p.sig_src);
            std::vector<uint32_t> wit_sig(NOCREADBENCH_SIG_WORDS, p.sig_witness);
            detail::WriteToDeviceL1(device, p.sources[0], data_addr, src_sig);
            detail::WriteToDeviceL1(device, witness, data_addr, wit_sig);

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
            args[NOCREADBENCH_A_TRID] = 0;
            args[NOCREADBENCH_A_SAMPLE] = sample;
            args[NOCREADBENCH_A_MODE] = mode;
            args[NOCREADBENCH_A_WITNESS_X] = (uint32_t)w_phys.x;
            args[NOCREADBENCH_A_WITNESS_Y] = (uint32_t)w_phys.y;
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
            // THE MODE CHECK, and it is read out of returned PAYLOAD rather
            // than out of the flag that asked for the mode. The kernel issued
            // one transaction with the read state pointed at the witness core;
            // the stateless call rewrites the coordinate and the source
            // answers, the stateful call does not and the witness answers. A
            // stale binary, a dropped argument or a kernel that silently kept
            // the other loop all land on the wrong signature here.
            const uint32_t got_mode = p.result[NOCREADBENCH_R_MODE];
            const uint32_t probe = p.result[NOCREADBENCH_R_PROBE];
            const uint32_t want_probe = (mode == NOCREADBENCH_MODE_STATEFUL) ? p.sig_witness : p.sig_src;
            p.mode_confirmed = (got_mode == mode) && (probe == want_probe);
            if (!p.mode_confirmed) {
                failures++;
                fprintf(stderr,
                        "nocreadbench: point %zu (%s): asked for the %s loop; the kernel stamped "
                        "mode=0x%08X and its probe read 0x%08X (source 0x%08X, witness 0x%08X, "
                        "expected 0x%08X). THE ARM DID NOT TAKE.\n",
                        pi, p.experiment.c_str(), mode_name, got_mode, probe, p.sig_src,
                        p.sig_witness, want_probe);
            }
            // The occupancy is the DIFFERENCE. The counter is live hardware
            // state the kernel inherits, so its raw value carries a baseline
            // that has nothing to do with this burst -- 71 or 72 on the
            // 2026-08-09 Blackhole card, in rows whose bursts were 4 requests
            // long. Written out as a column rather than left to the reader.
            const uint32_t out_max = p.result[NOCREADBENCH_R_OUTSTANDING_MAX];
            const uint32_t out_rest = p.result[NOCREADBENCH_R_OUTSTANDING_REST];
            const uint32_t out_delta = out_max > out_rest ? out_max - out_rest : 0;
            p.outstanding_delta = out_delta;
            p.inflight_max = p.result[NOCREADBENCH_R_INFLIGHT_MAX];
            p.cmdbuf_moved =
                p.result[NOCREADBENCH_R_CMDBUF_AVAIL_REST] != 0xFFFFFFFFu &&
                p.result[NOCREADBENCH_R_CMDBUF_AVAIL_MAX] != p.result[NOCREADBENCH_R_CMDBUF_AVAIL_REST];
            fprintf(out,
                    "%s,%u,%zu,%u,%u,%u,%u,%zu,%u,%u,%u,%u,%u,%u,%u,%u,%.3f,%u,%u,%u,0x%08X,0x%08X,"
                    "%u,%u,%u,%u,%u,0x%08X,0x%08X,0x%08X,%s,0x%08X,0x%08X,0x%08X\n",
                    p.experiment.c_str(), rep, pi,
                    (unsigned)p.master.x, (unsigned)p.master.y,
                    p.result[NOCREADBENCH_R_NODE_X], p.result[NOCREADBENCH_R_NODE_Y],
                    p.sources.size(), (unsigned)src_phys[0].x, (unsigned)src_phys[0].y,
                    p.hops, p.num_tx, p.tx_bytes, p.dst_stride, p.src_stride,
                    cycles, (double)cycles / (double)p.num_tx,
                    out_max,
                    p.result[NOCREADBENCH_R_OUTSTANDING_END],
                    p.result[NOCREADBENCH_R_SAMPLES],
                    p.result[NOCREADBENCH_R_CMDBUF_AVAIL_REST],
                    p.result[NOCREADBENCH_R_CMDBUF_AVAIL_BUSY],
                    out_rest, out_delta,
                    p.result[NOCREADBENCH_R_INFLIGHT_MAX],
                    p.result[NOCREADBENCH_R_INFLIGHT_REST],
                    p.result[NOCREADBENCH_R_TRID],
                    p.result[NOCREADBENCH_R_CMDBUF_AVAIL_MAX],
                    p.result[NOCREADBENCH_R_CMDBUF_OVFL_REST],
                    p.result[NOCREADBENCH_R_CMDBUF_OVFL_END],
                    // The kernel's own word, not the host's flag: REFUSED is a
                    // value the kernel can stamp and the host cannot ask for.
                    got_mode == NOCREADBENCH_MODE_STATEFUL    ? "stateful"
                    : got_mode == NOCREADBENCH_MODE_STATELESS ? "stateless"
                                                              : "refused",
                    probe, p.sig_src, p.sig_witness);
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

    // --- the mode verdict, before anything else ----------------------------
    // Everything downstream is a comparison between two issue loops, so "which
    // loop ran" is prior to every other reading in the file. It is settled from
    // the returned payload, per point, and a run that cannot settle it is not a
    // result at whatever rate it printed.
    size_t confirmed = 0, measured_points = 0;
    for (const Point& p : plan) {
        if (!p.measured) {
            continue;
        }
        measured_points++;
        if (p.mode_confirmed) {
            confirmed++;
        }
    }
    if (measured_points != 0 && confirmed == measured_points) {
        printf("nocreadbench: MODE CONFIRMED -- all %zu measured point(s) ran the %s issue\n"
               "  loop, read back from which tile answered a transaction issued with the read\n"
               "  state pointed at the witness core. Not taken from the --stateful flag.\n",
               confirmed, mode_name);
    } else {
        printf("  VERDICT: MODE NOT CONFIRMED -- %zu of %zu measured point(s) proved the %s\n"
               "  loop from their own payload. Every rate in this file is unattributable:\n"
               "  a stateful number produced by the stateless loop is exactly the wrong\n"
               "  answer to the question this variant is asked.\n",
               confirmed, measured_points, mode_name);
    }

    if (sample != 0) {
        // Everything below is a DIFFERENCE against each point's own rest
        // sample. The previous revision took the maximum RAW counter value and
        // compared it against `num_tx`; on the 2026-08-09 Blackhole card that
        // was 83 against a burst of 64 and it printed NO INITIATOR LIMIT from a
        // control that had never moved. A raw counter cannot be compared to a
        // burst length, because it does not start at zero.
        uint32_t worst_delta = 0, worst_inflight = 0, bad_rest = 0, cmdbuf_moves = 0;
        uint32_t max_num_tx = 0;
        for (const Point& p : plan) {
            if (!p.measured) {
                continue;
            }
            worst_delta = std::max(worst_delta, p.outstanding_delta);
            worst_inflight = std::max(worst_inflight, p.inflight_max);
            max_num_tx = std::max(max_num_tx, p.num_tx);
            // The in-flight pair must read zero before anything is issued: the
            // kernel enters after a barrier. If it does not, both instruments
            // are being read through something that is not what we think.
            if (p.result[NOCREADBENCH_R_INFLIGHT_REST] != 0) {
                bad_rest++;
            }
            if (p.cmdbuf_moved) {
                cmdbuf_moves++;
            }
        }
        printf("nocreadbench: highest occupancy over rest, per-trid counter = %u; "
               "trid-independent RD_REQ_SENT - RD_RESP_RECEIVED = %u (longest burst %u)\n",
               worst_delta, worst_inflight, max_num_tx);
        printf("nocreadbench: CMD_BUF_AVAIL left its rest value in %u of %zu points\n",
               cmdbuf_moves, plan.size());
        if (bad_rest != 0) {
            printf("  VERDICT: DEGENERATE -- %u point(s) saw a non-zero in-flight count\n"
                   "  BEFORE issuing anything, and the kernel starts after a read barrier.\n"
                   "  The status block is not being read the way this program assumes; no\n"
                   "  occupancy in this file means anything until that is explained.\n",
                   bad_rest);
        } else if (worst_delta == 0 && worst_inflight == 0) {
            printf("  VERDICT: DEGENERATE -- neither instrument moved off its rest value in\n"
                   "  any point. Either the sampling load is being hoisted, or reads complete\n"
                   "  before the next sample. EXPECTED against tt-sim, whose responses resolve\n"
                   "  inside the pump that issued them. On a card this is a broken run: do not\n"
                   "  read anything into the rate columns.\n");
        } else if (worst_delta == 0 || worst_inflight == 0) {
            printf("  VERDICT: DEGENERATE -- the two instruments disagree about whether\n"
                   "  anything was in flight at all (per-trid delta %u, trid-independent %u).\n"
                   "  The one reading zero is not watching these reads. Believe neither until\n"
                   "  the disagreement is explained; the trid-independent pair is the one that\n"
                   "  does not depend on which counter the requests landed in.\n",
                   worst_delta, worst_inflight);
        } else if (worst_inflight + 4 >= max_num_tx) {
            printf("  VERDICT: NO INITIATOR LIMIT -- in-flight requests track the burst\n"
                   "  length, so the initiator is not holding a bounded number in flight and\n"
                   "  the sustained rate is capped downstream of it. The credit-limit term\n"
                   "  is retired, not sized.\n");
        } else {
            printf("  VERDICT: BOUNDED AT %u -- the initiator holds at most this many read\n"
                   "  requests in flight (per-trid occupancy peaked at %u). Check it against\n"
                   "  the `dist` rows: a credit limit predicts cycles_per_tx = round_trip / %u,\n"
                   "  so cycles_per_tx MUST rise with hops. If the `dist` rows are flat, this\n"
                   "  bound is not what caps the rate either.\n",
                   worst_inflight, worst_delta, worst_inflight);
        }
        if (cmdbuf_moves == 0) {
            printf("  CMD_BUF_AVAIL: DEGENERATE -- identical at rest and in every in-loop\n"
                   "  sample, so the register reports nothing about the command-buffer depth.\n"
                   "  Do not quote its value as a depth: it is an occupancy (reset default 0,\n"
                   "  paired with CMD_BUF_OVFL), and an occupancy that never moves is silence.\n");
        } else {
            printf("  CMD_BUF_AVAIL: moved. Its peak is a LOWER BOUND on the depth unless\n"
                   "  cmdbuf_ovfl_end also moved off cmdbuf_ovfl_rest, which is the only\n"
                   "  reading that proves the buffer was driven to its limit.\n");
        }
    }
    return failures == 0 ? 0 : 1;
}
