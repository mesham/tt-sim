"""Read a ``dramratebench`` run as a SUSTAINED RATE, per reader count.

``perfbench/dramratebench`` runs the experiment ``wh_dram#performance``
describes in one line and one table: *N Tensix tiles each reading 1 MiB from
ONE DRAM channel, simultaneously, with N swept*. The vendor's own answer is

    1, 12 and 48 tiles -> 22.2, 22.3, 22.3 GB/s

-- an aggregate that does not grow with the number of readers. That flatness
is **endpoint occupancy stated as a vendor measurement**, and it is the only
validation available for tt-sim's ``DramChannels`` term on either
architecture: rung 2 structurally cannot reach it, because every retained DRAM
row is ``num_transactions = 1`` and a lone request never finds the channel
busy.

The probe already reports a *ratio* -- did concentrating the readers on one
endpoint cost anything relative to spreading them -- and that is the right
discriminator for whether an endpoint bound exists at all. This module reads
the other half, which nothing did: the **level**, per N, against a prediction
written down before the card ran.

Why the level needs its own reader
----------------------------------

A ratio says the endpoint costs something. It cannot say the plateau sits
where the model puts it, and the model is specific: Wormhole's channel is
``dram.channel_serialisation.bytes_per_cycle = 24``, so a saturated one-channel
aggregate cannot exceed 24 B/cycle however many readers push at it. At the
1 GHz the same ISA doc publishes that is 24.0 GB/s, and the vendor's measured
22.2-22.3 is 92-93% of it -- which is the same page's own
``achievable_fraction: 0.92``, arrived at from the other end. A run whose
one-channel arm plateaus at 24 and a run whose one-channel arm plateaus at 40
both give the same scaling ratio and only one of them agrees with the model.

Three things this deliberately does not do
------------------------------------------

* **It cannot make anything chargeable.** Every number here is a measurement
  on one part; it enters ``unit_costs.yaml`` as ``corroboration`` and never as
  provenance. That stayed true when Blackhole's channel rate became chargeable
  on 2026-08-12: what made it chargeable was arithmetic on tt-metal's *own*
  measured dataset, not this benchmark, and this benchmark's agreement with it
  to ~0.05 % is corroboration of a derivation that never saw it.
* **It never grades a flat curve on its own.** Flat is what a saturated
  channel looks like, and equally what a saturated link, a saturated issue
  rate, or N readers that never overlapped look like. The gates below run
  first and can only ever fail a run.
* **It is not a cycle oracle either way.** tt-sim is not cycle-accurate; the
  prediction it supplies is of a *shape* and a *level to within a band*, and
  :data:`LEVEL_BAND` is that band, stated once and applied to every point.

Run it
------

::

    python3 -m tt_sim.perf.dram_rate_sweep
    python3 -m tt_sim.perf.dram_rate_sweep --measured dram.wormhole.csv
    python3 -m tt_sim.perf.dram_rate_sweep --measured dram.csv --no-prediction

With no ``--measured`` it reads the tracked silicon dataset in
``tt_sim/perf/datasets/``. The prediction it compares against lives in
``perfbench/dramratebench/prediction-sustained.csv`` so that it travels to a
card box with ``perfbench/`` alone, where ``tt_sim/`` usually is not importable.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from tt_sim.perf.costs import load_costs
from tt_sim.perf.model import DramCostModel, NocCostModel

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = Path(__file__).resolve().parent / "datasets"

#: The tracked silicon dataset, read when ``--measured`` is omitted. Named
#: rather than globbed for the reason ``tensix_bench_sweep`` names its own: a
#: directory that grows a second file must not silently change what "the
#: measurement" means.
PRIMARY_DATASET = "dramratebench-blackhole-2026-08-09.csv"

#: The prediction, recorded before any card comparison. In ``perfbench/``
#: rather than in ``tt_sim/perf/datasets/``, twice over: it is not a
#: measurement on a part, so it must not sit in a directory whose contract says
#: every file in it is; and ``perfbench/`` is what gets rsynced to a card box,
#: so an operator can read what was predicted without a repo checkout.
PREDICTION_PATH = (
    _REPO_ROOT / "perfbench" / "dramratebench" / "prediction-sustained.csv"
)

#: ``wh_dram#performance``, quoted in ``unit_costs.yaml`` under
#: ``dram.bandwidth``: 1, 12 and 48 Tensix tiles each reading 1 MiB from one
#: DRAM channel simultaneously. Reads, static VC. WORMHOLE ONLY -- Blackhole
#: has no published DRAM tile page, and inventing one from this would be
#: exactly the laundering ``costs_test`` exists to stop.
VENDOR_READ_GB_S = {1: 22.2, 12: 22.3, 48: 22.3}
VENDOR_SOURCE = "wh_dram#performance"

#: A one-channel aggregate is called FLAT when its widest and narrowest swept
#: reader counts differ by no more than this. Not a tuned number: the vendor's
#: own three points span 1.0045, and the tracked Blackhole card spans 1.023
#: over 1 -> 120 readers. 1.15 clears both by a wide margin while still failing
#: outright on anything that scales -- the fan-out control on that same card
#: spans 4.90.
FLAT_BAND = 1.15

#: How far a measured plateau may sit from the predicted one and still be
#: called a hit. A quarter, and stated once: tt-sim is a functional oracle, not
#: a cycle oracle, so a tighter band would be a claim the simulator cannot
#: support, and a looser one would accept a plateau at the NoC link's 32
#: B/cycle as though it were the channel's 24.
LEVEL_BAND = 0.25

#: The fan-out control must grow by at least this much across the swept reader
#: counts, or nothing separates the endpoint from the fabric. Same figure the
#: program and ``card_session_verdicts.sh`` use, deliberately: three readers of
#: one experiment that disagreed about what it measured would be worse than
#: none.
CONTROL_MIN_SCALE = 1.5


def read_csv(path):
    """``(rows, meta)``. ``meta`` carries the ``#`` header's ``key=value``s.

    ``magic`` lands in ``meta`` like any other token and is deliberately never
    validated -- it is a wire check between a host binary and the kernel it
    just built, and a bumped magic must not make a banked dataset unreadable.
    """
    meta = {}
    lines = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            for token in line.lstrip("#").split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    meta[key] = value
            continue
        if line.strip():
            lines.append(line)
    rows = []
    for raw in csv.DictReader(lines):
        row = dict(raw)
        for key in (
            "repeat",
            "point",
            "num_readers",
            "num_tx",
            "tx_bytes",
            "bytes_per_reader",
            "total_bytes",
            "max_cycles",
            "min_cycles",
            "distinct_banks",
            "distinct_dram_cores",
            "tags_ok",
            "max_barrier_spins",
            "measured_readers",
        ):
            if key in row:
                row[key] = int(row[key])
        for key in (
            "agg_bytes_per_cycle",
            "agg_gb_per_s",
            "per_reader_bytes_per_cycle",
        ):
            if key in row:
                row[key] = float(row[key])
        rows.append(row)
    return rows, meta


def reference_datasets():
    """Every tracked ``dramratebench`` measurement, sorted by path."""
    if not DATASET_DIR.is_dir():
        return []
    return sorted(DATASET_DIR.glob("dramratebench-*.csv"))


def default_measured_path():
    """The tracked dataset to read when ``--measured`` is not given."""
    primary = DATASET_DIR / PRIMARY_DATASET
    if primary.exists():
        return primary
    found = reference_datasets()
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# The gates. Each is a pure function of the rows and can only ever FAIL a run.
# ---------------------------------------------------------------------------
#
# Every one of them has to be able to do both. `dramratebench`'s first card run
# was condemned by its own tag check because the host tagged slice 0 only, so
# every reader past the first had nothing to match and a clean file was thrown
# away -- a check that cannot pass is exactly as damaging as one that cannot
# fail. So each gate here is exercised in BOTH directions by
# `dram_rate_sweep_test.py`, on synthetic rows built to sit either side of it.


class Gate:
    """One validity check: a name, a verdict, and why."""

    __slots__ = ("name", "ok", "detail")

    def __init__(self, name, ok, detail):
        self.name = name
        self.ok = bool(ok)
        self.detail = detail

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Gate({self.name!r}, {self.ok!r}, {self.detail!r})"


def gate_tags(rows):
    """Did every reader read back the tag of the bank it was aimed at?

    A rate measured from the wrong endpoint is not a smaller result, it is a
    fictional one, and on a harvested part it is the easiest mistake to make.
    """
    bad = [r for r in rows if r["tags_ok"] != r["num_readers"]]
    if bad:
        where = ", ".join(f"{r['arm']}/n={r['num_readers']}" for r in bad[:4])
        return Gate(
            "tags",
            False,
            f"{len(bad)} of {len(rows)} row(s) held a reader that did not verify its bank's tag ({where})",
        )
    return Gate(
        "tags", True, f"all {len(rows)} row(s) verified every reader's bank tag"
    )


def gate_overlap(rows):
    """Did the readers in some multi-reader point actually overlap?

    N bursts run one after another produce exactly the flat aggregate this
    experiment is looking for, from an experiment that never happened.
    """
    multi = [r for r in rows if r["num_readers"] > 1]
    if not multi:
        return Gate(
            "overlap",
            False,
            "no multi-reader point at all, so no simultaneity was ever exercised",
        )
    waited = [r for r in multi if r["max_barrier_spins"] > 0]
    if not waited:
        return Gate(
            "overlap",
            False,
            f"no reader waited at the barrier in any of the {len(multi)} multi-reader row(s)",
        )
    return Gate(
        "overlap",
        True,
        f"{len(waited)} of {len(multi)} multi-reader row(s) had a reader wait at the barrier",
    )


def gate_control(rows):
    """Did the fan-out control MOVE?

    The one arm that separates the endpoint from everything upstream of it.
    Same readers, same issue loop, same transaction size, N distinct banks: if
    that does not grow either, something upstream caps both arms and the
    one-channel arm's flatness is uninterpretable.
    """
    table = sustained(rows, arm="fanchan")
    if len(table) < 2:
        return Gate(
            "control",
            False,
            "the fan-out control has no pair of reader counts to compare",
        )
    lo, hi = min(table), max(table)
    if table[lo].bytes_per_cycle <= 0:
        return Gate(
            "control",
            False,
            f"the fan-out control reads {table[lo].bytes_per_cycle} B/cycle at {lo} reader(s)",
        )
    scale = table[hi].bytes_per_cycle / table[lo].bytes_per_cycle
    detail = (
        f"fanchan {table[lo].bytes_per_cycle:.3f} -> {table[hi].bytes_per_cycle:.3f} B/cycle "
        f"over {lo} -> {hi} readers (x{scale:.2f})"
    )
    if scale < CONTROL_MIN_SCALE:
        return Gate(
            "control",
            False,
            detail + f", under the x{CONTROL_MIN_SCALE} a moving control has to clear",
        )
    return Gate("control", True, detail)


def gates(rows):
    """Every gate, in the order a reader must apply them."""
    return [gate_tags(rows), gate_overlap(rows), gate_control(rows)]


# ---------------------------------------------------------------------------
# The sustained rate itself.
# ---------------------------------------------------------------------------


class Reading:
    """One arm at one reader count, over however many repeats there were."""

    __slots__ = ("num_readers", "bytes_per_cycle", "gb_per_s", "per_reader", "repeats")

    def __init__(self, num_readers, bytes_per_cycle, gb_per_s, per_reader, repeats):
        self.num_readers = num_readers
        self.bytes_per_cycle = bytes_per_cycle
        self.gb_per_s = gb_per_s
        self.per_reader = per_reader
        self.repeats = repeats

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Reading(n={self.num_readers}, {self.bytes_per_cycle:.3f} B/cycle over {self.repeats} repeat(s))"


def sustained(rows, arm="onechan"):
    """``{num_readers: Reading}`` for one arm, MEDIAN over the repeats.

    Median rather than max or mean: a repeat that hit an unrelated stall should
    move the reported rate by nothing, and with the usual three repeats the
    median is the middle one rather than an average of an outlier.
    """
    by_n = {}
    for row in rows:
        if row.get("arm") != arm:
            continue
        by_n.setdefault(row["num_readers"], []).append(row)
    out = {}
    for n, group in sorted(by_n.items()):
        out[n] = Reading(
            n,
            statistics.median(r["agg_bytes_per_cycle"] for r in group),
            statistics.median(r.get("agg_gb_per_s", 0.0) for r in group),
            statistics.median(r.get("per_reader_bytes_per_cycle", 0.0) for r in group),
            len(group),
        )
    return out


def flatness(table):
    """``(ratio, lo_n, hi_n)`` -- how much the aggregate moved across the sweep.

    ``None`` when there is nothing to compare, which is a DEGENERATE run and
    not a flat one. The distinction is the whole point: one reader cannot
    demonstrate that a second one does not help.
    """
    if len(table) < 2:
        return None
    lo, hi = min(table), max(table)
    if table[lo].bytes_per_cycle <= 0:
        return None
    return table[hi].bytes_per_cycle / table[lo].bytes_per_cycle, lo, hi


# ---------------------------------------------------------------------------
# Which resource a plateau sits on.
# ---------------------------------------------------------------------------
#
# The scaling ratio establishes that CONCENTRATING the readers cost something.
# It cannot say what they concentrated onto, and there are two candidates that
# a host program cannot tell apart: the GDDR6 channel, and the one inbound NoC
# router link every flow in the `onechan` arm converges on. The `samecore` arm
# was built to separate them and fires on neither part's descriptor, so the
# ambiguity is open by construction -- but the two run at DIFFERENT RATES, and
# a level can therefore say which one bound where a ratio cannot.
#
# Both figures come from the cost tables rather than from literals here, so
# they carry the tables' own provenance discipline: `DramCostModel` refuses to
# read Wormhole's 24 through Blackhole's deep-merged override, so a plateau is
# never attributed to a rate the arch's own table does not carry. The channel
# figure taken is the READ one, because this is a read benchmark and Blackhole
# sources a rate for that direction only.
#
# One consequence of Blackhole gaining a read rate on 2026-08-12: this
# attribution stopped being a two-way choice on that part. Before it, `channel`
# was None there and a plateau could only read `link` or `neither` -- which is
# how the card's 47.147 came back `neither`, correctly, against a model that
# then held no such bound. It now reads `channel`, and the test that pinned
# `neither` says which of the two facts changed.

#: How close a plateau must sit to a ceiling to be attributed to it. Wormhole's
#: two ceilings are 24 and 32, a third apart, so a tenth cannot reach from one
#: to the other and an unattributable plateau stays unattributed.
CEILING_BAND = 0.10


def ceilings(arch):
    """``(link_bytes_per_cycle, channel_read_bytes_per_cycle)`` for one arch.

    Built from the sections directly rather than through ``noc_cost_model`` /
    ``dram_cost_model``, which return ``None`` unless ``TT_SIM_COST_MODEL`` is
    set in the environment. That gate is right for the execution path and wrong
    here: what a table publishes does not depend on whether this process
    happens to be simulating with it.

    The channel figure is the **read** rate because ``dramratebench`` reads;
    Blackhole's table carries no write rate at all, so asking for the wrong
    direction here would silently answer ``None``.
    """
    if arch not in ("wormhole", "blackhole"):
        return None, None
    sections = load_costs(arch).sections
    return (
        NocCostModel(sections, arch).bytes_per_cycle,
        DramCostModel(sections, arch).channel_bytes_per_cycle_read,
    )


def plateau_sits_at(plateau, arch, band=CEILING_BAND):
    """``"channel"``, ``"link"`` or ``"neither"`` for a measured plateau."""
    link, channel = ceilings(arch)
    if channel and abs(plateau - channel) / channel <= band:
        return "channel"
    if link and abs(plateau - link) / link <= band:
        return "link"
    return "neither"


# ---------------------------------------------------------------------------
# The prediction.
# ---------------------------------------------------------------------------


def load_prediction(path=None):
    """``(rows, meta)`` for the committed prediction, or ``([], {})``."""
    path = Path(path) if path is not None else PREDICTION_PATH
    if not path.exists():
        return [], {}
    meta = {}
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            for token in line.lstrip("#").split():
                if "=" in token:
                    key, value = token.split("=", 1)
                    meta[key] = value
            continue
        if line.strip():
            lines.append(line)
    rows = []
    for raw in csv.DictReader(lines):
        row = dict(raw)
        row["num_readers"] = int(row["num_readers"])
        row["bytes_per_reader"] = int(row["bytes_per_reader"])
        row["tx_bytes"] = int(row["tx_bytes"])
        row["agg_bytes_per_cycle"] = float(row["agg_bytes_per_cycle"])
        row["agg_gb_per_s"] = float(row["agg_gb_per_s"])
        rows.append(row)
    return rows, meta


def predicted_for(prediction_rows, arch, arm):
    """``{num_readers: row}`` from the prediction, for one arch and arm."""
    return {
        r["num_readers"]: r
        for r in prediction_rows
        if r["arch"] == arch and r["arm"] == arm
    }


def compare_to_prediction(table, predicted, band=LEVEL_BAND):
    """``[(n, measured, predicted_row, deviation, hit)]`` over shared points."""
    out = []
    for n in sorted(set(table) & set(predicted)):
        want = predicted[n]["agg_bytes_per_cycle"]
        got = table[n].bytes_per_cycle
        deviation = (got - want) / want if want else float("inf")
        out.append((n, got, predicted[n], deviation, abs(deviation) <= band))
    return out


def compare_to_vendor(table, clock_mhz, band=LEVEL_BAND):
    """``[(n, measured_gb_s, vendor_gb_s, deviation, hit)]``.

    Empty unless the run carries a real clock: tt-sim reports 0 MHz, honestly,
    and converting B/cycle at a clock the device did not report would invent
    the very number the comparison is about.
    """
    if not clock_mhz:
        return []
    out = []
    for n in sorted(set(table) & set(VENDOR_READ_GB_S)):
        want = VENDOR_READ_GB_S[n]
        got = table[n].bytes_per_cycle * clock_mhz / 1000.0
        deviation = (got - want) / want
        out.append((n, got, want, deviation, abs(deviation) <= band))
    return out


# ---------------------------------------------------------------------------
# The report.
# ---------------------------------------------------------------------------


def report(path, prediction_path=None, use_prediction=True, out=print):
    """Print the sustained-rate reading for one file; return a verdict dict."""
    rows, meta = read_csv(path)
    arch = meta.get("arch", "unknown")
    clock_mhz = int(meta.get("clock_mhz", 0) or 0)
    out(f"dramratebench sustained-rate sweep: {path}")
    out(f"  arch={arch} clock_mhz={clock_mhz} rows={len(rows)}")
    if not rows:
        out("  VERDICT: DEGENERATE -- the file has no data rows.")
        return {"verdict": "DEGENERATE", "gates": [], "onechan": {}}

    checks = gates(rows)
    out("")
    out("  Gates, in order, each of which can only fail the run:")
    for gate in checks:
        out(f"    [{'pass' if gate.ok else 'FAIL'}] {gate.name}: {gate.detail}")
    failed = [g for g in checks if not g.ok]

    one = sustained(rows, arm="onechan")
    fan = sustained(rows, arm="fanchan")
    out("")
    per_reader_bytes = rows[0].get("bytes_per_reader", 0)
    out(
        f"  ONE channel, N readers, {per_reader_bytes} B each -- the vendor's own experiment at 1 MiB:"
    )
    out(
        "    readers   aggregate B/cycle   per reader   aggregate GB/s   fanchan B/cycle"
    )
    for n, reading in sorted(one.items()):
        gb = (
            f"{reading.bytes_per_cycle * clock_mhz / 1000.0:12.2f}"
            if clock_mhz
            else "           -"
        )
        other = f"{fan[n].bytes_per_cycle:14.3f}" if n in fan else "             -"
        out(
            f"    {n:7d}   {reading.bytes_per_cycle:17.3f}   {reading.per_reader:10.3f}   {gb}   {other}"
        )
    if not clock_mhz:
        out(
            "    (the device reported 0 MHz -- tt-sim does -- so no GB/s column is shown;"
        )
        out(
            "     a GB/s converted at a clock the device did not report is not a reading)"
        )

    flat = flatness(one)
    plateau = None
    if flat is not None:
        ratio, lo, hi = flat
        verdict_flat = "FLAT" if ratio <= FLAT_BAND else "NOT FLAT"
        out("")
        out(
            f"  {verdict_flat}: the one-channel aggregate moved x{ratio:.3f} over {lo} -> {hi} readers"
        )
        out(
            f"    (the band is x{FLAT_BAND}; the vendor's own three points span x1.0045)"
        )
        plateau = one[hi].bytes_per_cycle
        link, channel = ceilings(arch)
        where = plateau_sits_at(plateau, arch)
        out("")
        out(
            f"  The plateau is {plateau:.3f} B/cycle at {hi} readers. The two ceilings a flow can hit:"
        )
        link_text = f"{link} B/cycle" if link else "unsourced on this arch"
        channel_text = f"{channel} B/cycle" if channel else "UNPUBLISHED on this arch"
        out(
            f"    NoC link      {link_text:26}  (noc.flit_bits x throughput_flits_per_cycle)"
        )
        out(f"    DRAM channel  {channel_text:26}  (dram.channel_serialisation)")
        if where == "channel":
            out(
                "    -> it sits at the CHANNEL. The endpoint is what bound, which is what the term asserts."
            )
        elif where == "link":
            out(
                "    -> it sits at the LINK, not the channel. A one-channel arm flattened by the"
            )
            out(
                "       DRAM tile's inbound router link is flat for a reason that is not endpoint"
            )
            out("       occupancy, and the ratio verdict cannot tell the two apart.")
        else:
            out(
                "    -> it sits at NEITHER. Whatever bound this aggregate is not a quantity this"
            )
            out(
                "       model holds, so the run sizes something the tables do not yet describe."
            )

    result = {
        "verdict": "DEGENERATE" if failed else "READ",
        "gates": checks,
        "onechan": one,
        "fanchan": fan,
        "flatness": flat,
        "plateau": plateau,
        "plateau_at": plateau_sits_at(plateau, arch) if plateau else None,
        "arch": arch,
        "clock_mhz": clock_mhz,
    }

    if use_prediction:
        pred_rows, pred_meta = load_prediction(prediction_path)
        predicted = predicted_for(pred_rows, arch, "onechan")
        out("")
        if not predicted:
            out(
                f"  No committed prediction for arch={arch}; nothing to compare the level against."
            )
        else:
            out(
                f"  Against the prediction committed on {pred_meta.get('recorded', '?')} (before any card run):"
            )
            out(
                "    readers   predicted B/cycle   measured B/cycle   deviation   basis"
            )
            comparison = compare_to_prediction(one, predicted)
            for n, got, want, deviation, hit in comparison:
                mark = "hit " if hit else "MISS"
                out(
                    f"    {n:7d}   {want['agg_bytes_per_cycle']:17.3f}   {got:16.3f}   "
                    f"{deviation:+8.1%} {mark}   {want['basis']}"
                )
            if not comparison:
                out("    (no reader count is in both the run and the prediction)")
            result["prediction"] = comparison

    vendor = compare_to_vendor(one, clock_mhz)
    out("")
    if arch != "wormhole":
        out(
            f"  No vendor table applies: {VENDOR_SOURCE} is a WORMHOLE page and this run is {arch}."
        )
        out(
            "    Blackhole publishes no DRAM tile page at all, so there is no published"
        )
        out(
            "    per-channel rate to compare against and nothing measured here can supply one."
        )
        out(
            "    Its chargeable channel rate is a DERIVATION off tt-metal's measured NoC"
        )
        out(
            "    dataset (dram.channel_serialisation, vendor_source_derived), not a page"
        )
        out("    and not this run -- see the plateau attribution above.")
    elif not vendor:
        out(
            f"  The vendor's {VENDOR_SOURCE} table (22.2 / 22.3 / 22.3 GB/s at 1 / 12 / 48 tiles)"
        )
        out(
            "    cannot be compared: this run reports no clock, and only B/cycle is a reading."
        )
    else:
        out(f"  Against the published table ({VENDOR_SOURCE}), reads, static VC:")
        out("    readers   published GB/s   measured GB/s   deviation")
        for n, got, want, deviation, hit in vendor:
            mark = "hit " if hit else "MISS"
            out(f"    {n:7d}   {want:14.1f}   {got:13.2f}   {deviation:+8.1%} {mark}")
        result["vendor"] = vendor

    out("")
    if failed:
        out("  VERDICT: DEGENERATE -- " + "; ".join(g.detail for g in failed))
        out(
            "    A flat one-channel curve is the EXPECTED answer here, so it is never the"
        )
        out(
            "    evidence. Until the control moves, flat and broken are the same picture."
        )
    else:
        out(
            "  VERDICT: READ -- every gate passed, so the levels above mean what they say."
        )
    out("    CORROBORATION, NEVER PROVENANCE. Nothing here can make a cost-table")
    out("    entry: Wormhole's channel rate is isa_doc_derived, Blackhole's is derived")
    out(
        "    from tt-metal's own measured dataset, and neither derivation has ever seen"
    )
    out("    this run. Agreement is a check on them, never their source.")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = default_measured_path()
    ap.add_argument(
        "--measured",
        default=str(default) if default else None,
        help="a dramratebench CSV (default: the tracked silicon run)",
    )
    ap.add_argument(
        "--prediction",
        default=None,
        help=f"the prediction CSV (default: {PREDICTION_PATH})",
    )
    ap.add_argument(
        "--no-prediction",
        action="store_true",
        help="read the run's own levels only, without comparing them to the committed prediction",
    )
    args = ap.parse_args(argv)
    if not args.measured:
        ap.error("no --measured file and no tracked dataset to fall back on")
    result = report(
        args.measured,
        prediction_path=args.prediction,
        use_prediction=not args.no_prediction,
    )
    return 0 if result["verdict"] == "READ" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
