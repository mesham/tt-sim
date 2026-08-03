"""Rung 2 of the calibration ladder: tt-metal's measured NoC dataset, swept.

``docs/plans/cost-model.md`` lists four rungs of validation below "captured
silicon traces". Rung 1 (internal consistency against tt-metal's four
hardcoded end-to-end figures) is climbed. This module is rung 2:

    tt-metal's measured NoC dataset. Entries of empirically measured
    end-to-end latency keyed by transaction size, access pattern, memory type
    and subordinate count, shipped in tt-metal's ``noc_estimator``. Too coarse
    to *derive* a per-hop congestion model from -- it folds everything into one
    number -- but exactly the right thing to *validate* one against.

The distinction that makes this worth doing: **not one term of the assembled
model was fitted to this dataset.** The hop latencies and the flit width are
``isa_doc``; the DRAM access latency is ``vendor_source_derived`` from two
*different* vendor numbers (``tm_dm_common``'s hardcoded 358 and 259). So every
comparison below is out of sample.

What the dataset actually is
----------------------------

``tt_metal/impl/experimental/noc_estimator/latencies/noc_latencies.yaml``:
740 entries x 11 transaction sizes (64 B .. 64 KiB) = 8,140 measured points,
each a **device cycle count**, not a time. The measurement is the duration of
a ``DeviceZoneScopedN`` region in the data-movement suite's estimator kernels
(``noc_estimator_tests/kernels/{reader,writer,dram_reader,dram_writer}.cpp``)
that wraps *the whole issue loop plus the closing barrier*: N ``noc_async_*``
calls followed by one ``noc_async_*_barrier``. So it includes the issuing
RISC-V core's register writes and its barrier polling, which tt-sim does not
model -- see :data:`RESIDUAL_EXPECTATION`.

Three things about the keying are worth stating because they are easy to get
wrong and none of them is documented:

* ``num_transactions`` in a YAML key is **transactions per barrier**, not the
  total: ``noc_estimator.cpp`` builds its lookup key with
  ``.num_transactions = params.num_transactions_per_barrier`` and then
  multiplies the looked-up latency by ``ceil(total / per_barrier)``.
* The pattern enum in ``types.hpp`` (``ONE_FROM_ONE = 0``) is **not** the one
  in the test that produced the data (``ONE_TO_ONE = 0``). They are reconciled
  by the CSV carrying the pattern's *name*, which ``csv_reader.cpp`` maps to
  the ``types.hpp`` value -- so YAML ``pattern: 0`` is a read and ``pattern: 1``
  is a write.
* ``noc_index`` is 0 on all 740 entries and carries no information: the kernels
  log the column as ``"NoC Index"`` and ``csv_reader.cpp`` looks for
  ``"NOC index"``, so it never matches and every point takes the default. Rows
  measured on NoC 0 and NoC 1 therefore collide on one key. This does not
  affect anything here, because a *round trip* on a directional torus costs the
  same number of hops on either NoC.

Run it
------

::

    python3 -m tt_sim.perf.noc_dataset_sweep                  # auto-locate
    python3 -m tt_sim.perf.noc_dataset_sweep --dataset PATH
    python3 -m tt_sim.perf.noc_dataset_sweep --arch blackhole

The dataset lives outside this repo, so with no tt-metal checkout the script
prints where it looked and exits 0 -- the same "degrade gracefully" contract
``examples/examples_test.py`` uses for its tt-metal dependency.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# The dataset.
# ---------------------------------------------------------------------------

#: Path under a tt-metal checkout.
DATASET_SUFFIX = Path(
    "tt_metal/impl/experimental/noc_estimator/latencies/noc_latencies.yaml"
)

#: Every key field and the value the loader assumes when the entry omits it,
#: from ``types.hpp``'s ``DEFAULT_*`` constants. The YAML only writes
#: non-default fields, so an entry cannot be read without these.
KEY_DEFAULTS = {
    "mechanism": 0,  # UNICAST
    "pattern": 1,  # ONE_TO_ONE  (a write)
    "memory": 0,  # L1
    "arch": 2,  # WORMHOLE_B0
    "num_transactions": 1,  # ... per barrier
    "num_subordinates": 1,
    "same_axis": False,
    "stateful": False,
    "loopback": False,
    "noc_index": 0,
}

ARCH_IDS = {"wormhole": 2, "blackhole": 3}
PATTERN_READ, PATTERN_WRITE = 0, 1
MEMORY_L1, MEMORY_DRAM_INTERLEAVED, MEMORY_DRAM_SHARDED = 0, 1, 2

#: Largest transaction the DRAM sweep actually issues. ``dram_accessor_sweep``
#: caps at ``max_pages_per_txn = 256`` pages of ``obtain_page_size_bytes`` = 32
#: bytes. The offline ``data_extractor`` then pads every standard size above
#: the largest measured one by repeating the last value, so a DRAM row's
#: 16 KiB / 32 KiB / 64 KiB entries are fill, not measurements.
DRAM_SWEEP_MAX_BYTES = 8192


def default_dataset_path(env=None):
    """Where to look for the dataset, or ``None``.

    ``TT_SIM_NOC_LATENCIES`` wins if set; otherwise the same
    ``TT_METAL_RUNTIME_ROOT`` / ``TT_METAL_HOME`` pair the examples runner
    uses.
    """
    env = os.environ if env is None else env
    explicit = env.get("TT_SIM_NOC_LATENCIES")
    if explicit:
        return Path(explicit)
    root = env.get("TT_METAL_RUNTIME_ROOT") or env.get("TT_METAL_HOME")
    return Path(root) / DATASET_SUFFIX if root else None


def load_dataset(path):
    """``[(key_dict, [latency per standard size]), ...]`` plus the size list."""
    raw = yaml.safe_load(Path(path).read_text())
    sizes = list(raw["transaction_sizes"])
    entries = []
    for entry in raw["entries"]:
        key = dict(KEY_DEFAULTS)
        key.update(entry["key"])
        entries.append((key, [float(v) for v in entry["latencies"]]))
    return entries, sizes


# ---------------------------------------------------------------------------
# The exclusion criteria. DECLARED BEFORE ANY RESIDUAL WAS COMPUTED.
# ---------------------------------------------------------------------------
#
# This ordering matters more than the criteria do. Dropping entries because
# they *disagree* is fitting, not validating, and it would make the whole
# exercise worthless -- so each rule below names a term tt-sim does not model,
# and every one of them could have been written without the dataset in hand.
# The cost of each in entries is reported by the sweep so a reader can see how
# much of the dataset the model is declining to be tested against.

RESIDUAL_EXPECTATION = """\
The model predicts the NoC round trip and the endpoint's service time and
nothing else. The measurement additionally contains the issuing RISC-V core's
own path -- the DeviceZone entry, the ~6 NIU register stores per
``noc_async_read``, and the barrier's polling loop on
``NIU_MST_REQS_OUTSTANDING`` -- none of which tt-sim charges for (the NIU
register block is on ``RV_UNNAMED_REGIONS``, deliberately uncosted).

So the residual is NOT expected to be zero. It is expected to be a CONSTANT:
one issuing-core path, the same in every row, independent of transaction size,
of geometry, of memory type and of direction. Its value is not predicted; its
constancy is exactly what the assembled model claims. Any structure in the
residual along one of those four axes is model error, and names which term."""


def _exclusions(arch_id):
    """The retained-set predicate, as an ordered ladder of named rules."""
    return [
        (
            "arch",
            "swept one architecture at a time; see --arch",
            lambda k: k["arch"] == arch_id,
        ),
        (
            "mechanism != UNICAST",
            "tt-sim models a multicast as N unicast deliveries sharing one "
            "injection port. The hardware's router-level fan-out, and the "
            "arbitration between the copies, is not modelled at all.",
            lambda k: k["mechanism"] == 0,
        ),
        (
            "pattern not in {ONE_FROM_ONE, ONE_TO_ONE}",
            "every other pattern has >= 2 concurrent initiators or targets "
            "sharing links, which is congestion -- `noc.congestion` is "
            "`provenance: unknown` and nothing in tt-sim models it. The "
            "*_ALL patterns also sweep grids whose per-core distances the "
            "dataset does not record, so there is no hop count to predict.",
            lambda k: k["pattern"] in (PATTERN_READ, PATTERN_WRITE),
        ),
        (
            "num_transactions per barrier != 1",
            "N > 1 measures a pipelined burst, whose length is set by the "
            "initiator's outstanding-transaction credit limit and by the "
            "kernel loop's per-transaction issue cost. tt-sim models neither. "
            "N = 1 is the only shape where the measured zone is one round "
            "trip.",
            lambda k: k["num_transactions"] == 1,
        ),
        (
            "stateful",
            "stateful mode is an issue-side optimisation (`set_state` + "
            "`with_state` reuses the configured NoC command registers). "
            "tt-sim charges no per-transaction NoC register configuration "
            "cost, so it has no term that could distinguish the two and "
            "would predict them identical by construction.",
            lambda k: not k["stateful"],
        ),
        (
            "loopback",
            "a multicast-linked feature; nothing in tt-sim expresses it.",
            lambda k: not k["loopback"],
        ),
        (
            "memory == DRAM_INTERLEAVED",
            "interleaving spreads pages round-robin over 12 channels at "
            "different distances. tt-sim instantiates one DRAM tile and has "
            "no interleaving model.",
            lambda k: k["memory"] != MEMORY_DRAM_INTERLEAVED,
        ),
    ]


def retained(entries, arch_id):
    """``(kept, [(rule, removed, remaining), ...])`` -- the ladder, in order."""
    kept, ladder = entries, []
    for name, _reason, keep in _exclusions(arch_id):
        nxt = [e for e in kept if keep(e[0])]
        ladder.append((name, len(kept) - len(nxt), len(nxt)))
        kept = nxt
    return kept, ladder


def point_is_measured(key, size):
    """False for a size the offline extractor filled in rather than measured.

    The only case in the retained set is a DRAM row above
    :data:`DRAM_SWEEP_MAX_BYTES`: ``dram_accessor_sweep`` never issues one, and
    ``data_extractor.cpp`` pads the standard sizes above the largest measured
    one by repeating the last value. A structural fact about how the file was
    generated, visible in the raw data as a flat tail, and nothing to do with
    whether the model agrees.
    """
    return key["memory"] == MEMORY_L1 or size <= DRAM_SWEEP_MAX_BYTES


# ---------------------------------------------------------------------------
# Geometry: the one thing the dataset does not record and the model needs.
# ---------------------------------------------------------------------------
#
# ``run_single_core`` in ``test_noc_estimator.cpp`` fixes it:
#
#     CoreCoord master_coord = {0, 0};
#     CoreCoord sub_coord = cfg.same_axis ? CoreCoord{0, 1} : CoreCoord{1, 1};
#
# so ``same_axis`` means the two cores share their logical x, i.e. they are in
# the same column and differ only in y. The DRAM rows use logical core {0, 0}
# and DRAM bank 0.
#
# That is enough, and it is enough for a reason worth stating: on a directional
# torus a round trip between two tiles differing on both axes is exactly
# ``grid_x + grid_y`` hops *whatever* the distance between them, and one
# sharing an axis is exactly the other dimension. So the hop count is fixed by
# ``same_axis`` alone -- no core-placement assumption is doing any work, and
# the answer is the same on NoC 0 and NoC 1.

#: Logical -> tt-sim tile coord for the two cores the single-core patterns use.
#: Wormhole tiles are keyed by unified coord (18, 18) == SoC-physical (1, 1);
#: Blackhole tiles are keyed by physical NoC coord directly.
GEOMETRY = {
    "wormhole": {"master": (18, 18), "same_axis": (18, 19), "diff_axis": (19, 19)},
    "blackhole": {"master": (1, 2), "same_axis": (1, 3), "diff_axis": (2, 3)},
}

_L1_ADDR = 0x30000
_DRAM_ADDR = 0x1000


def _physical_of(arch, tile_coord):
    """SoC-physical NoC 0 coord of a tt-sim tile coord.

    Wormhole keys its tiles by *unified* coord (the 16-25 band) and Blackhole
    by the physical NoC coord directly, so this is the one place the two
    conventions have to be reconciled. The hop counts in this module are all
    computed in physical NoC space.
    """
    if arch == "wormhole":
        from tt_sim.device.wormhole import Wormhole

        return Wormhole.physical_noc0_coord_from_unified_worker(tile_coord)
    return tuple(tile_coord)


def _build_device(arch, tensix_coords):
    if arch == "wormhole":
        from tt_sim.device.wormhole import Wormhole

        return Wormhole(tensix_coords=list(tensix_coords))
    from tt_sim.device.blackhole import Blackhole

    return Blackhole(tensix_coords=list(tensix_coords))


def _set_target_coord(initiator, which, coord):
    from tt_sim.network.noc_coords import WormholeNocCoords

    x, y = coord
    if isinstance(initiator.nui.noc_coord_strategy, WormholeNocCoords):
        setattr(initiator, f"{which}_addr_mid", (x << 4) | (y << 10))
    else:
        setattr(initiator, f"{which}_addr_hi", (y << 6) | x)


def predict_cycles(arch, memory, is_read, same_axis, size, budget=500_000):
    """Cycles the **assembled model** takes for one transaction, measured.

    Not a closed form: this builds a real device with the cost model on, drives
    one NoC transaction through the initiator's registers exactly as tt-metal
    firmware does, and pumps until ``NIU_MST_REQS_OUTSTANDING`` returns to zero
    -- which is precisely what ``noc_async_read_barrier`` /
    ``noc_async_write_barrier`` wait for, and therefore precisely where the
    measured DeviceZone ends. Every term is exercised through its real
    consumer: the hop model and the injection port in ``tt_sim/network/``,
    burst splitting at ``noc_max_burst_size``, and the DRAM endpoint's service
    window in ``tt_sim/device/tiles.py``.
    """
    from tt_sim.network.tt_noc import NUI

    outstanding = NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
    geometry = GEOMETRY[arch]
    master = geometry["master"]
    sub = geometry["same_axis" if same_axis else "diff_axis"]

    if memory == MEMORY_L1:
        device = _build_device(arch, [master, sub])
        local, remote = device.tensix_tiles[0], device.tensix_tiles[1]
        remote_addr = _L1_ADDR
    else:
        device = _build_device(arch, [master])
        local, remote = device.tensix_tiles[0], device.dram_tiles[0]
        remote_addr = _DRAM_ADDR

    payload = bytes((i * 7) & 0xFF for i in range(size))
    source = remote if is_read else local
    device.write(source.get_coord_pair(), remote_addr if is_read else _L1_ADDR, payload)

    router = local.noc0_router
    initiator = router.request_initiators[0]
    initiator.packet_tag = 0
    initiator.at_len_be = size
    if is_read:
        _set_target_coord(initiator, "target", remote.noc0_router.id_pair)
        initiator.target_addr_low = remote_addr
        initiator.ret_addr_low = _L1_ADDR
        initiator.ctrl = 0
    else:
        initiator.target_addr_low = _L1_ADDR
        _set_target_coord(initiator, "ret", remote.noc0_router.id_pair)
        initiator.ret_addr_low = remote_addr
        # mode 2 = write, bit 4 = NOC_CMD_RESP_MARKED (non-posted: an ACK is
        # returned and OUTSTANDING tracks it, which is what a write barrier
        # waits on).
        initiator.ctrl = 2 | (1 << 4)
    initiator.cmd_ctrl = 1
    initiator.initiate()

    for cycle in range(1, budget + 1):
        device.run(1)
        if router.nui_counters[outstanding] == 0:
            return cycle
    raise RuntimeError(f"no completion within {budget} cycles for {arch} {size} B")


# ---------------------------------------------------------------------------
# The sweep.
# ---------------------------------------------------------------------------


class Residual:
    """One measured point and what the model said about it."""

    __slots__ = ("key", "size", "measured", "predicted")

    def __init__(self, key, size, measured, predicted):
        self.key = key
        self.size = size
        self.measured = measured
        self.predicted = predicted

    @property
    def residual(self):
        return self.measured - self.predicted

    @property
    def relative(self):
        return self.residual / self.measured

    @property
    def label(self):
        memory = {MEMORY_L1: "L1", MEMORY_DRAM_SHARDED: "DRAM"}[self.key["memory"]]
        direction = "read" if self.key["pattern"] == PATTERN_READ else "write"
        axis = "same-axis" if self.key["same_axis"] else "diff-axis"
        return f"{memory} {direction} {axis}"


def sweep(entries, sizes, arch, drop_unmeasured=True):
    """Predict every retained point. Returns ``[Residual, ...]``."""
    cache = {}
    rows = []
    for key, latencies in entries:
        for size, measured in zip(sizes, latencies):
            if drop_unmeasured and not point_is_measured(key, size):
                continue
            signature = (
                key["memory"],
                key["pattern"] == PATTERN_READ,
                key["same_axis"],
                size,
            )
            if signature not in cache:
                cache[signature] = predict_cycles(arch, *signature)
            rows.append(Residual(key, size, measured, cache[signature]))
    return rows


def _linear_fit(xs, ys):
    """Least-squares ``(intercept, slope)``; ``(mean, 0)`` for a degenerate x."""
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var
    return mean_y - slope * mean_x, slope


def _summary(values):
    values = sorted(values)
    quartile = len(values) // 4
    return {
        "n": len(values),
        "min": values[0],
        "p25": values[quartile],
        "median": statistics.median(values),
        "p75": values[-1 - quartile],
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def _grouped(rows, key_fn):
    groups = {}
    for row in rows:
        groups.setdefault(key_fn(row), []).append(row)
    return groups


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def report(entries, sizes, arch, out=sys.stdout):
    arch_id = ARCH_IDS[arch]
    kept, ladder = retained(entries, arch_id)

    def emit(line=""):
        print(line, file=out)

    emit("=" * 78)
    emit(f"Rung 2: tt-metal measured NoC dataset vs the assembled model [{arch}]")
    emit("=" * 78)
    emit()
    emit(
        f"dataset: {len(entries)} entries x {len(sizes)} sizes "
        f"= {len(entries) * len(sizes)} measured points, in device cycles"
    )
    emit()
    emit("Exclusion ladder (declared before any residual was computed):")
    total = len(entries)
    for name, removed, remaining in ladder:
        emit(f"  - {name:<44} removes {removed:>4}  leaves {remaining:>4}")
    dropped_points = sum(
        1 for key, _ in kept for size in sizes if not point_is_measured(key, size)
    )
    emit(
        f"  - {'sizes the DRAM sweep never issued':<44} removes "
        f"{dropped_points:>4} points (not entries)"
    )
    emit()
    emit(
        f"retained: {len(kept)} of {total} entries, "
        f"{len(kept) * len(sizes) - dropped_points} points"
    )
    emit()
    emit(RESIDUAL_EXPECTATION)
    emit()

    if not kept:
        emit("nothing retained; nothing to check.")
        return []

    rows = sweep(kept, sizes, arch)

    emit("-" * 78)
    emit("Per-point: measured (tt-metal) vs predicted (tt-sim), cycles")
    emit("-" * 78)
    emit(f"{'':<32}" + "".join(f"{s:>8}" for s in sizes))
    for group_label, group in sorted(_grouped(rows, lambda r: r.label).items()):
        emit()
        by_size = {r.size: r for r in group}
        for name, attr in (
            ("measured", "measured"),
            ("predicted", "predicted"),
            ("residual", "residual"),
        ):
            cells = "".join(
                f"{getattr(by_size[s], attr):>8.0f}" if s in by_size else f"{'-':>8}"
                for s in sizes
            )
            label = group_label if attr == "measured" else ""
            emit(f"{label:<22}{name:<10}{cells}")

    emit()
    emit("-" * 78)
    emit("Residual distribution")
    emit("-" * 78)
    absolute = _summary([r.residual for r in rows])
    relative = _summary([100 * r.relative for r in rows])
    emit(f"  n = {absolute['n']}")
    emit(
        f"  cycles : min {absolute['min']:>7.0f}  p25 {absolute['p25']:>7.0f}  "
        f"median {absolute['median']:>7.0f}  p75 {absolute['p75']:>7.0f}  "
        f"max {absolute['max']:>7.0f}"
    )
    emit(
        f"  percent: min {relative['min']:>7.1f}  p25 {relative['p25']:>7.1f}  "
        f"median {relative['median']:>7.1f}  p75 {relative['p75']:>7.1f}  "
        f"max {relative['max']:>7.1f}"
    )
    emit()
    over = sorted({r.label for r in rows if r.residual < 0})
    if not over:
        emit(
            "  The model is a FLOOR everywhere: every residual is positive, "
            "which is the\n  direction every bound in these tables is chosen "
            "to lean."
        )
    else:
        emit(
            "  The model is NOT a floor everywhere. It OVER-CHARGES these "
            "rows, i.e. it\n  invents back-pressure the hardware does not "
            "have:"
        )
        for label in over:
            group = [r for r in rows if r.label == label and r.residual < 0]
            worst = min(r.residual for r in group)
            emit(f"    {label:<22} {len(group)} points, worst {worst:.0f} cycles")

    emit()
    emit("-" * 78)
    emit("Residual by axis -- where the constant stops being constant")
    emit("-" * 78)
    for axis_name, key_fn in (
        (
            "geometry (tests the hop term)",
            lambda r: "same-axis" if r.key["same_axis"] else "diff-axis",
        ),
        (
            "memory (tests the DRAM access-latency term)",
            lambda r: {MEMORY_L1: "L1", MEMORY_DRAM_SHARDED: "DRAM"}[r.key["memory"]],
        ),
        (
            "direction",
            lambda r: "read" if r.key["pattern"] == PATTERN_READ else "write",
        ),
    ):
        emit()
        emit(f"  {axis_name}")
        for name, group in sorted(_grouped(rows, key_fn).items()):
            small = [r for r in group if r.size <= 512]
            stats = _summary([r.residual for r in group])
            emit(
                f"    {name:<12} n={stats['n']:<4} median {stats['median']:>7.0f}"
                f"   median at <= 512 B {statistics.median([r.residual for r in small]):>7.0f}"
            )

    emit()
    emit("  size (tests the bandwidth term)")
    xs = [r.size for r in rows]
    ys = [float(r.residual) for r in rows]
    intercept, slope = _linear_fit(xs, ys)
    emit(
        f"    least squares over all points: residual = {intercept:.0f}"
        f" + {slope * 1024:.2f} cycles/KiB"
    )
    for name, group in sorted(_grouped(rows, lambda r: r.label).items()):
        gx = [r.size for r in group]
        gy = [float(r.residual) for r in group]
        gi, gs = _linear_fit(gx, gy)
        emit(f"    {name:<22} residual = {gi:>6.0f} + {gs * 1024:>6.2f} cycles/KiB")

    emit()
    emit("  implied sustained bandwidth, from the large-transfer slope")
    sourced = _sourced_bandwidths(arch)
    for name, group in sorted(_grouped(rows, lambda r: r.label).items()):
        big = sorted([r for r in group if r.size >= 4096], key=lambda r: r.size)
        if len(big) < 2:
            continue
        d_bytes = big[-1].size - big[0].size
        measured_rate = d_bytes / (big[-1].measured - big[0].measured)
        model_rate = d_bytes / (big[-1].predicted - big[0].predicted)
        emit(
            f"    {name:<22} measured {measured_rate:>6.2f} B/cycle"
            f"   model {model_rate:>6.2f} B/cycle"
            f"   ({100 * measured_rate / model_rate:.0f} % of modelled peak)"
        )
    emit()
    emit("    for reference, the bandwidth figures the tables already hold:")
    for label, value in sourced.items():
        emit(f"      {label:<40} {value}")

    _bandwidth_ceiling_check(entries, sizes, arch_id, emit)
    return rows


def _sourced_bandwidths(arch):
    """The bandwidth constants already in the tables, for the report to quote.

    Read out of the YAML rather than restated here, so a table edit moves the
    reference number too.
    """
    from tt_sim.perf.costs import SOURCED_PROVENANCE, load_costs

    sections = load_costs(arch).sections
    noc = sections.get("noc") or {}
    dram = (sections.get("dram") or {}).get("bandwidth") or {}
    out = {}
    # Provenance is checked, not assumed. Blackhole's ``dram.bandwidth``
    # override is ``provenance: unknown``, and because the arch overrides are
    # deep-merged the Wormhole per-channel figure is still *present* under it.
    # Quoting it for Blackhole would be exactly the laundering the convention
    # exists to stop.
    if noc.get("flit_bits") and noc.get("provenance") in SOURCED_PROVENANCE:
        out["noc.flit_bits / 8 (B per cycle per link)"] = noc["flit_bits"] // 8
    if (
        dram.get("per_channel_gb_per_s")
        and dram.get("provenance") in SOURCED_PROVENANCE
    ):
        out["dram.bandwidth.per_channel_gb_per_s"] = dram["per_channel_gb_per_s"]
    return out or {"(none sourced for this architecture)": ""}


def _bandwidth_ceiling_check(entries, sizes, arch_id, emit):
    """Secondary, dataset-only: does any measurement beat the modelled link?

    The primary sweep excludes every entry with more than one transaction per
    barrier, because their *latency* folds in an issue loop tt-sim does not
    model. Their **bandwidth** is a different question and a one-sided one: the
    model says one NIU's injection link carries ``flit_bits / 8`` bytes per
    cycle and no more, so a measured single-initiator transfer that exceeded it
    would falsify the constant outright, whatever the issue path costs. That
    check needs no prediction and no simulation, so those 40 entries are worth
    something after all -- just not what the primary sweep wanted from them.
    """
    from tt_sim.perf.costs import load_costs

    arch = {v: k for k, v in ARCH_IDS.items()}[arch_id]
    flit_bits = (load_costs(arch).sections.get("noc") or {}).get("flit_bits")
    if not flit_bits:
        return
    ceiling = flit_bits / 8

    best = {}
    for key, latencies in entries:
        if key["arch"] != arch_id or key["mechanism"] != 0:
            continue
        if key["pattern"] not in (PATTERN_READ, PATTERN_WRITE):
            continue
        if key["memory"] != MEMORY_L1:
            continue
        direction = "read" if key["pattern"] == PATTERN_READ else "write"
        for size, latency in zip(sizes, latencies):
            rate = size * key["num_transactions"] / latency
            record = best.setdefault(direction, (0.0, None))
            if rate > record[0]:
                best[direction] = (rate, (key["num_transactions"], size))

    emit()
    emit("-" * 78)
    emit("Secondary check (dataset only, no simulation): the bandwidth ceiling")
    emit("-" * 78)
    emit(
        f"  the model's per-NIU injection link: {ceiling:.0f} B/cycle "
        f"(noc.flit_bits = {flit_bits}, isa_doc)"
    )
    emit("  fastest single-initiator L1 transfer anywhere in the dataset,")
    emit("  including the entries the primary sweep excluded:")
    for direction, (rate, where) in sorted(best.items()):
        transactions, size = where
        emit(
            f"    {direction:<6} {rate:>6.2f} B/cycle "
            f"({100 * rate / ceiling:>5.1f} % of the ceiling)"
            f"   at {transactions} x {size} B"
        )
    exceeded = [d for d, (rate, _) in best.items() if rate > ceiling]
    emit(
        "  -> nothing exceeds it, so the constant is not falsified."
        if not exceeded
        else f"  -> EXCEEDED by {exceeded}: the flit-width constant is wrong."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--arch", default="wormhole", choices=sorted(ARCH_IDS))
    args = parser.parse_args(argv)

    path = args.dataset or default_dataset_path()
    if path is None or not path.exists():
        print(
            "noc_latencies.yaml not found"
            + (f" at {path}" if path else "")
            + ".\nSet TT_SIM_NOC_LATENCIES, or TT_METAL_RUNTIME_ROOT to a "
            "tt-metal checkout,\nor pass --dataset. Skipping (the dataset "
            "lives outside this repo)."
        )
        return 0

    os.environ["TT_SIM_COST_MODEL"] = "1"
    entries, sizes = load_dataset(path)
    report(entries, sizes, args.arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
