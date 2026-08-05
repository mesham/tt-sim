// nocbench -- NoC congestion microbenchmark, executor half.
//
// WHAT THIS IS FOR
// ----------------
// `tt_sim/perf/unit_costs.yaml` carries `noc.congestion` at
// `provenance: unknown`. tt-metal ships a 740-row measured NoC dataset, and
// `tt_sim/perf/noc_dataset_sweep.py` established that a congestion model cannot
// be DERIVED from it -- not because it is coarse but because it is
// unidentifiable: every multi-party row changes the flow count by resizing a
// grid, so flow count, path length and link sharing all move together; no core
// coordinates are recorded; and the one aggregate scalar per key already
// contains the issuing core's loop and the endpoint's L1 port arbitration.
//
// This program is the other half of the answer that module's CONGESTION_VERDICT
// names: measure it on a card, with the coordinates recorded and exactly one
// thing varying per experiment.
//
// WHY IT IS NOT tt-metal's OWN data_movement SUITE
// ------------------------------------------------
// `tests/tt_metal/tt_metal/data_movement/` is the right *shape* -- it exposes
// coordinates, unlike the shipped YAML -- but every coordinate, grid size and
// virtual channel in it is a compile-time literal inside a gtest body. There is
// no environment variable, no gtest flag and no argument that selects them;
// `TensixDataMovementOneToOneCustom`, the one test whose name promises it, is
// `GTEST_SKIP`ped and takes its coordinates from C++ default arguments. Placing
// two masters at a chosen link overlap is therefore not expressible by that
// suite without editing it. See perfbench/nocbench/README.md, "What the
// upstream suite cannot express".
//
// THE SPLIT
// ---------
// This program is a dumb executor. It takes a PLAN (a CSV of flows, one row per
// flow, grouped by run) and reports what each flow measured. Every decision
// about which coordinates hold which confound fixed is made -- and asserted --
// in `tt_sim/perf/noc_congestion_plan.py`, in Python, under test. That is
// deliberate: the invariants are the experiment, and an invariant that lives in
// tested code is checkable in a way that one living in a comment is not.
//
//   nocbench --dump-grid [--out FILE]     # logical -> NoC coordinate map
//   nocbench --plan PLAN.csv [--out FILE] # execute a plan
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "kernels/nocbench_layout.h"

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

// --- the world's smallest CSV reader ---------------------------------------
// Comment lines (leading '#') are preserved verbatim into the output so the
// plan's own provenance -- which planner built it, with which invariants --
// travels with the measurements rather than being re-attached afterwards.

struct Csv {
    std::vector<std::string> comments;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;

    int col(const std::string& name) const {
        for (size_t i = 0; i < header.size(); i++) {
            if (header[i] == name) {
                return (int)i;
            }
        }
        return -1;
    }
};

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

bool read_csv(const std::string& path, Csv& csv) {
    FILE* f = fopen(path.c_str(), "r");
    if (f == nullptr) {
        fprintf(stderr, "nocbench: cannot open plan '%s'\n", path.c_str());
        return false;
    }
    char buf[8192];
    while (fgets(buf, sizeof(buf), f) != nullptr) {
        std::string line(buf);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            csv.comments.push_back(line);
            continue;
        }
        if (csv.header.empty()) {
            csv.header = split(line);
        } else {
            csv.rows.push_back(split(line));
        }
    }
    fclose(f);
    return !csv.header.empty();
}

uint32_t cell_u32(const Csv& csv, const std::vector<std::string>& row, const char* name, bool& ok) {
    int c = csv.col(name);
    if (c < 0 || (size_t)c >= row.size()) {
        fprintf(stderr, "nocbench: plan has no column '%s'\n", name);
        ok = false;
        return 0;
    }
    return (uint32_t)strtoul(row[c].c_str(), nullptr, 10);
}

struct Flow {
    size_t row_index = 0;
    uint32_t flow = 0;
    CoreCoord master_logical{0, 0};
    CoreCoord sub_logical{0, 0};
    uint32_t plan_mst_nx = 0, plan_mst_ny = 0, plan_sub_nx = 0, plan_sub_ny = 0;
    uint32_t plan_mst_px = 0, plan_mst_py = 0;
    uint32_t num_tx = 0, tx_bytes = 0, direction = 0, noc = 0, vc = 0, proc = 0;
    // Filled in after the launch.
    bool measured = false;
    std::vector<uint32_t> result;
};

int dump_grid(IDevice* device, const std::string& out_path) {
    const CoreCoord grid = device->grid_size();
    const CoreCoord logical = device->compute_with_storage_grid_size();
    FILE* f = fopen(out_path.c_str(), "w");
    if (f == nullptr) {
        fprintf(stderr, "nocbench: cannot write '%s'\n", out_path.c_str());
        return 1;
    }
    fprintf(
        f,
        "# nocbench-grid arch=%s soc_grid_x=%u soc_grid_y=%u logical_grid_x=%u logical_grid_y=%u\n",
        arch_name(device),
        (unsigned)grid.x,
        (unsigned)grid.y,
        (unsigned)logical.x,
        (unsigned)logical.y);
    fprintf(f, "# noc_x/noc_y are worker_core_from_logical_core(), i.e. the coordinate a kernel addresses.\n");
    fprintf(f, "log_x,log_y,noc_x,noc_y\n");
    for (uint32_t y = 0; y < logical.y; y++) {
        for (uint32_t x = 0; x < logical.x; x++) {
            CoreCoord phys = device->worker_core_from_logical_core(CoreCoord(x, y));
            fprintf(f, "%u,%u,%u,%u\n", x, y, (unsigned)phys.x, (unsigned)phys.y);
        }
    }
    fclose(f);
    printf("nocbench: wrote %s (%u x %u logical workers on a %u x %u SoC grid)\n",
           out_path.c_str(),
           (unsigned)logical.x,
           (unsigned)logical.y,
           (unsigned)grid.x,
           (unsigned)grid.y);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string plan_path;
    std::string out_path;
    bool want_grid = false;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dump-grid") {
            want_grid = true;
        } else if (a == "--plan" && i + 1 < argc) {
            plan_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "-v" || a == "--verbose") {
            verbose = true;
        } else if (a == "-h" || a == "--help") {
            printf(
                "nocbench --dump-grid [--out FILE]\n"
                "nocbench --plan PLAN.csv [--out FILE] [-v]\n\n"
                "Plans are built by `python3 -m tt_sim.perf.noc_congestion_plan`.\n"
                "See perfbench/nocbench/README.md.\n");
            return 0;
        } else {
            fprintf(stderr, "nocbench: unknown argument '%s' (try --help)\n", argv[i]);
            return 2;
        }
    }
    if (!want_grid && plan_path.empty()) {
        fprintf(stderr, "nocbench: need --dump-grid or --plan (try --help)\n");
        return 2;
    }

    IDevice* device = CreateDevice(0);
    const std::string arch = arch_name(device);

    if (want_grid) {
        int rc = dump_grid(device, out_path.empty() ? ("nocbench-grid-" + arch + ".csv") : out_path);
        CloseDevice(device);
        return rc;
    }

    Csv plan;
    if (!read_csv(plan_path, plan)) {
        CloseDevice(device);
        return 2;
    }
    if (out_path.empty()) {
        out_path = "nocbench-" + arch + ".csv";
    }

    // --- parse the plan into runs ------------------------------------------
    bool ok = true;
    std::vector<uint32_t> run_order;
    std::map<uint32_t, std::vector<Flow>> runs;
    for (size_t i = 0; i < plan.rows.size(); i++) {
        const auto& row = plan.rows[i];
        uint32_t run = cell_u32(plan, row, "run", ok);
        Flow fl;
        fl.row_index = i;
        fl.flow = cell_u32(plan, row, "flow", ok);
        fl.master_logical = CoreCoord(cell_u32(plan, row, "mst_lx", ok), cell_u32(plan, row, "mst_ly", ok));
        fl.sub_logical = CoreCoord(cell_u32(plan, row, "sub_lx", ok), cell_u32(plan, row, "sub_ly", ok));
        fl.plan_mst_nx = cell_u32(plan, row, "mst_nx", ok);
        fl.plan_mst_ny = cell_u32(plan, row, "mst_ny", ok);
        fl.plan_sub_nx = cell_u32(plan, row, "sub_nx", ok);
        fl.plan_sub_ny = cell_u32(plan, row, "sub_ny", ok);
        fl.plan_mst_px = cell_u32(plan, row, "mst_px", ok);
        fl.plan_mst_py = cell_u32(plan, row, "mst_py", ok);
        fl.num_tx = cell_u32(plan, row, "num_tx", ok);
        fl.tx_bytes = cell_u32(plan, row, "tx_bytes", ok);
        fl.direction = cell_u32(plan, row, "direction", ok);
        fl.noc = cell_u32(plan, row, "noc", ok);
        fl.vc = cell_u32(plan, row, "vc", ok);
        fl.proc = cell_u32(plan, row, "proc", ok);
        if (fl.proc > 1) {
            fprintf(stderr, "nocbench: proc must be 0 (BRISC) or 1 (NCRISC), got %u\n", fl.proc);
            ok = false;
        }
        if (!ok) {
            CloseDevice(device);
            return 2;
        }
        if (runs.find(run) == runs.end()) {
            run_order.push_back(run);
        }
        runs[run].push_back(fl);
    }

    // --- the coordinate check ----------------------------------------------
    // The planner works out link overlaps in a coordinate space it took from a
    // --dump-grid of THIS device. If the plan was built against a different
    // machine, a different harvesting configuration or a stale dump, every
    // shared-link count in it is fiction. Refuse rather than measure something
    // adjacent: this is the one failure mode that would silently reproduce the
    // shipped dataset's problem.
    for (auto& [run, flows] : runs) {
        for (const Flow& fl : flows) {
            CoreCoord m = device->worker_core_from_logical_core(fl.master_logical);
            CoreCoord s = device->worker_core_from_logical_core(fl.sub_logical);
            if (m.x != fl.plan_mst_nx || m.y != fl.plan_mst_ny || s.x != fl.plan_sub_nx || s.y != fl.plan_sub_ny) {
                fprintf(
                    stderr,
                    "FATAL: plan run %u flow %u says logical (%u,%u)->(%u,%u) is NoC (%u,%u)->(%u,%u), "
                    "but this device says (%u,%u)->(%u,%u). Re-run --dump-grid and re-plan.\n",
                    run,
                    fl.flow,
                    (unsigned)fl.master_logical.x,
                    (unsigned)fl.master_logical.y,
                    (unsigned)fl.sub_logical.x,
                    (unsigned)fl.sub_logical.y,
                    fl.plan_mst_nx,
                    fl.plan_mst_ny,
                    fl.plan_sub_nx,
                    fl.plan_sub_ny,
                    (unsigned)m.x,
                    (unsigned)m.y,
                    (unsigned)s.x,
                    (unsigned)s.y);
                CloseDevice(device);
                return 1;
            }
        }
    }

    // --- L1 scratch, at one address common to every core --------------------
    // An L1 interleaved buffer is banked across the worker grid at a single
    // address, so `->address()` is valid on every participating core. The
    // magic word in the result header is what proves it, per run.
    uint32_t max_tx_bytes = 0;
    for (auto& [run, flows] : runs) {
        for (const Flow& fl : flows) {
            max_tx_bytes = std::max(max_tx_bytes, fl.tx_bytes);
        }
    }
    const uint32_t data_bytes = 2 * max_tx_bytes;  // write source | read destination
    InterleavedBufferConfig data_cfg{
        .device = device, .size = data_bytes, .page_size = data_bytes, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> data_scratch = CreateBuffer(data_cfg);
    const uint32_t data_addr = data_scratch->address();

    // Two result slots per core, one per data-movement RISC, so the `selfport`
    // control (two flows out of one core) does not have its two kernels
    // overwrite each other's stamp.
    constexpr uint32_t result_bytes = NOCBENCH_R_WORDS * 4;
    constexpr uint32_t result_block = 2 * result_bytes;
    InterleavedBufferConfig res_cfg{
        .device = device, .size = result_block, .page_size = result_block, .buffer_type = BufferType::L1};
    std::shared_ptr<Buffer> result_scratch = CreateBuffer(res_cfg);
    const uint32_t results_addr = result_scratch->address();

    printf("nocbench: arch=%s runs=%zu data_addr=0x%08X results_addr=0x%08X max_tx=%u B\n",
           arch.c_str(),
           run_order.size(),
           data_addr,
           results_addr,
           max_tx_bytes);
    fflush(stdout);

    // --- execute -----------------------------------------------------------
    size_t failures = 0;
    size_t node_id_reports = 0, node_id_zero = 0;
    for (uint32_t run : run_order) {
        std::vector<Flow>& flows = runs[run];

        std::set<CoreRange> core_ranges;
        for (const Flow& fl : flows) {
            core_ranges.insert(CoreRange(fl.master_logical));
            core_ranges.insert(CoreRange(fl.sub_logical));
        }
        CoreRangeSet all_cores(core_ranges);

        Program program = CreateProgram();
        const uint32_t sem_id = CreateSemaphore(program, all_cores, 0);

        for (const Flow& fl : flows) {
            KernelHandle k = CreateKernel(
                program,
                "kernels/dataflow/flow.cpp",
                fl.master_logical,
                DataMovementConfig{
                    .processor = fl.proc == 1 ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
                    .noc = fl.noc == 1 ? NOC::NOC_1 : NOC::NOC_0});
            std::vector<uint32_t> args(NOCBENCH_A_PEERS, 0);
            args[NOCBENCH_A_RESULTS] = results_addr + fl.proc * result_bytes;
            args[NOCBENCH_A_DATA] = data_addr;
            args[NOCBENCH_A_SEM] = sem_id;
            args[NOCBENCH_A_NUM_PEERS] = (uint32_t)flows.size() - 1;
            args[NOCBENCH_A_FLOW] = fl.flow;
            args[NOCBENCH_A_SUB_X] = fl.plan_sub_nx;
            args[NOCBENCH_A_SUB_Y] = fl.plan_sub_ny;
            args[NOCBENCH_A_NUM_TX] = fl.num_tx;
            args[NOCBENCH_A_TX_BYTES] = fl.tx_bytes;
            args[NOCBENCH_A_DIRECTION] = fl.direction;
            args[NOCBENCH_A_VC] = fl.vc;
            for (const Flow& peer : flows) {
                if (peer.flow == fl.flow) {
                    continue;
                }
                args.push_back(peer.plan_mst_nx);
                args.push_back(peer.plan_mst_ny);
            }
            SetRuntimeArgs(program, k, fl.master_logical, args);
        }

        std::vector<uint32_t> zeros(2 * NOCBENCH_R_WORDS, 0);
        for (const Flow& fl : flows) {
            detail::WriteToDeviceL1(device, fl.master_logical, results_addr, zeros);
        }

        detail::LaunchProgram(device, program, true, true);

        for (Flow& fl : flows) {
            detail::ReadFromDeviceL1(
                device, fl.master_logical, results_addr + fl.proc * result_bytes, result_bytes, fl.result);
            fl.measured = fl.result.size() > NOCBENCH_R_MAGIC && fl.result[NOCBENCH_R_MAGIC] == NOCBENCH_MAGIC;
            if (!fl.measured) {
                failures++;
                fprintf(stderr, "nocbench: run %u flow %u: no result stamp (kernel did not run?)\n", run, fl.flow);
                continue;
            }
            // The kernel's own view of where it ran, versus the plan's physical
            // coordinate. This is the second half of the coordinate check and
            // needs no host API to be trusted: the number came off the core's
            // own NIU. It is compared against the PHYSICAL column because
            // NOC_NODE_ID is a physical coordinate, and physical is the space
            // every link and hop count in the plan lives in.
            //
            // An all-zero self-report means the device does not answer that
            // register rather than that every kernel ran on core (0, 0) -- the
            // simulator is one such device -- so it is reported once, at the
            // end, and not counted as a per-flow failure. A MIXED result is a
            // real mismatch and is.
            const uint32_t kx = fl.result[NOCBENCH_R_NODE_X];
            const uint32_t ky = fl.result[NOCBENCH_R_NODE_Y];
            const bool noc1 = fl.result[NOCBENCH_R_NOC] == 1;
            node_id_reports++;
            if (kx == 0 && ky == 0) {
                node_id_zero++;
            } else if (!noc1 && (kx != fl.plan_mst_px || ky != fl.plan_mst_py)) {
                failures++;
                fprintf(
                    stderr,
                    "nocbench: run %u flow %u: NIU reports physical coord (%u,%u), plan says (%u,%u)\n",
                    run,
                    fl.flow,
                    kx,
                    ky,
                    fl.plan_mst_px,
                    fl.plan_mst_py);
            }
            if (verbose) {
                printf(
                    "  run %u flow %u  (%u,%u)->(%u,%u)  %u x %u B  %u cycles  rendezvous %u\n",
                    run,
                    fl.flow,
                    fl.plan_mst_nx,
                    fl.plan_mst_ny,
                    fl.plan_sub_nx,
                    fl.plan_sub_ny,
                    fl.num_tx,
                    fl.tx_bytes,
                    fl.result[NOCBENCH_R_CYCLES],
                    fl.result[NOCBENCH_R_REND_CYCLES]);
                fflush(stdout);
            }
        }
    }

    // --- report -------------------------------------------------------------
    // Plan columns verbatim plus the measurement, so one file carries the whole
    // experiment: what was asked for, what geometry it implied, and what came
    // back. `t0`/`t1` are raw wall clock, kept so the sweep can check that
    // concurrent flows really did overlap in time.
    FILE* csv = fopen(out_path.c_str(), "w");
    if (csv == nullptr) {
        fprintf(stderr, "nocbench: cannot write '%s'\n", out_path.c_str());
        CloseDevice(device);
        return 1;
    }
    for (const std::string& c : plan.comments) {
        fprintf(csv, "%s\n", c.c_str());
    }
    fprintf(csv, "# measured by nocbench on arch=%s magic=0x%08X\n", arch.c_str(), NOCBENCH_MAGIC);
    for (size_t i = 0; i < plan.header.size(); i++) {
        fprintf(csv, "%s%s", i ? "," : "", plan.header[i].c_str());
    }
    fprintf(csv, ",measured,cycles,t0,t1,rendezvous_cycles,kernel_node_x,kernel_node_y,kernel_noc\n");

    std::vector<const Flow*> by_row(plan.rows.size(), nullptr);
    for (auto& [run, flows] : runs) {
        for (const Flow& fl : flows) {
            by_row[fl.row_index] = &fl;
        }
    }
    for (size_t i = 0; i < plan.rows.size(); i++) {
        const auto& row = plan.rows[i];
        for (size_t c = 0; c < row.size(); c++) {
            fprintf(csv, "%s%s", c ? "," : "", row[c].c_str());
        }
        const Flow* fl = by_row[i];
        if (fl != nullptr && fl->measured) {
            fprintf(
                csv,
                ",1,%u,%u,%u,%u,%u,%u,%u\n",
                fl->result[NOCBENCH_R_CYCLES],
                fl->result[NOCBENCH_R_T0],
                fl->result[NOCBENCH_R_T1],
                fl->result[NOCBENCH_R_REND_CYCLES],
                fl->result[NOCBENCH_R_NODE_X],
                fl->result[NOCBENCH_R_NODE_Y],
                fl->result[NOCBENCH_R_NOC]);
        } else {
            fprintf(csv, ",0,0,0,0,0,0,0,0\n");
        }
    }
    fclose(csv);

    if (node_id_reports > 0 && node_id_zero == node_id_reports) {
        printf(
            "nocbench: NOTE -- every kernel's NOC_NODE_ID read back as 0, so this device does "
            "not answer that register (the simulator does not). The plan's coordinates were "
            "still checked against worker_core_from_logical_core on the host.\n");
    }
    printf("nocbench: wrote %s (%zu flows, %zu failures)\n", out_path.c_str(), plan.rows.size(), failures);
    printf("nocbench: analyse with `python3 -m tt_sim.perf.noc_congestion_sweep --measured %s`\n", out_path.c_str());
    CloseDevice(device);
    return failures == 0 ? 0 : 1;
}
