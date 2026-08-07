"""Ranked bottleneck report — one page saying where a run's modelled
cycles went, and how much of the run the model could not name.

This is the consumer end of ``TT_SIM_PROFILE`` (see
:mod:`tt_sim.trace.auto`). It reads two artefacts a profiled run leaves
behind — the Parquet counter dataset and the per-PC hotspot table — and
writes ``report.md`` plus a machine-readable ``report.json``.
Regenerate at any time without re-running the simulator::

    python3 -m tt_sim.trace.report <profile-dir>

**Counters are classified by pattern, not by a list.** Anything named
``stall_<reason>`` is an RV stall with that reason; anything else ending
in ``_cycles`` is a cycle-bearing counter and is ranked alongside the
rest, marked *(discovered)* so a reader can see the report adopted a
counter it has no prose for. That is deliberate: counters are being
added to this tree faster than any hardcoded table would survive, and a
report that silently omitted a new stall reason would be wrong in the
most expensive direction — it would look complete.

**Shares are against the run's cycle span and units run concurrently**,
so per-source shares can and do sum past 100 %. The report says so in
its own body rather than in a docstring, because the number that matters
to a reader optimising a kernel is the *relative* ranking, and the way
to make that safe is to refuse to present a tidy pie.
"""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Counters that are cycle-bearing but not named ``*_cycles``.
_EXTRA_CYCLE_COUNTERS = {"instr_retired"}
#: ``stall_cycles`` is the sum of the per-reason rows, so counting it too
#: would double every RV stall.
_REDUNDANT = {"stall_cycles"}

#: Prose for the counters that exist today. A counter missing from here is
#: still reported — it is just labelled as discovered rather than described.
_DESCRIPTIONS = {
    "instr_retired": "RV instruction issue (1 cycle floor each)",
    "busy_cycles": "Tensix backend occupancy, from the cost tables",
    "noc_flight_cycles": "NoC packet flight, issue to service",
    "stall_load_use": "RV load-use interlock",
    "stall_load_rate": "RV sustained-load rate limit",
    "stall_store_rate": "RV sustained-store rate limit",
    "stall_integer_unit": "RV integer-unit latency",
}

STALL_PREFIX = "stall_"

FRAMING = """\
> **What these numbers are.** Every cycle below is *modelled*: charged from a
> bound published in Tenstorrent's ISA documentation, always at the low end of
> the published range. They are **floors**, corroborated against silicon
> measurements but **not calibrated** to them — no total here has ever been
> fitted to an end-to-end hardware run. Read the *relative* attribution and
> optimise against that. Do not read any absolute number as a cycle promise,
> and do not compare a total against hardware and conclude the model is off by
> the difference; the model has never claimed the difference was zero.
>
> The simulator is not cycle-accurate. A term that is uncosted contributes
> nothing rather than a guess, so an unattributed remainder is real missing
> coverage and is reported as such below rather than folded away.\
"""


@dataclass
class Contribution:
    """One cycle-bearing counter's total for one unit."""

    unit: str
    counter: str
    cycles: int
    described: str = ""
    discovered: bool = False

    @property
    def kind(self) -> str:
        if self.counter.startswith(STALL_PREFIX):
            return "stall"
        return "occupancy"


@dataclass
class Report:
    span: int = 0
    cost_model: bool | None = None
    contributions: list[Contribution] = field(default_factory=list)
    volumes: dict[str, int] = field(default_factory=dict)
    hotspots: dict = field(default_factory=dict)
    elfs: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    label: str = ""

    def per_unit(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for c in self.contributions:
            out[c.unit] += c.cycles
        return dict(out)

    def attributed(self) -> int:
        return sum(c.cycles for c in self.contributions)

    def shared_resource_cycles(self) -> dict[str, int]:
        """Cycles charged against a resource the whole device shares, by
        family.

        This is the roll-up ``docs/plans/cost-model.md`` uses when it asks
        what fraction of a run is "attributed to a published number", and
        it is the only sum in this report that is meaningful across units:
        two NIUs servicing packets at the same time really are two
        occupied links, and two Tensix backends really are two occupied
        pipes. Per-core stall cycles are not in here — see the note the
        report prints beside the table.
        """
        out: dict[str, int] = defaultdict(int)
        for c in self.contributions:
            if c.counter == "noc_flight_cycles":
                out["NoC flight and bandwidth"] += c.cycles
            elif c.counter == "busy_cycles":
                out["Tensix backend occupancy"] += c.cycles
        return dict(out)


# ---------------------------------------------------------------------------
# Reading the counter dataset
# ---------------------------------------------------------------------------


def load_counters(directory: Path | str) -> tuple[dict[tuple[str, str], int], int]:
    """``{(unit, counter): total}`` and the run's cycle span."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(Path(directory) / "**" / "*.parquet"), recursive=True))
    if not files:
        return {}, 0
    rows = pq.ParquetDataset(files).read().to_pylist()
    totals: dict[tuple[str, str], int] = defaultdict(int)
    span = 0
    for row in rows:
        unit = f"{row['core_y']},{row['core_x']} {row['unit']}"
        totals[(unit, row["counter_name"])] += row["value"]
        if row["cycle"] > span:
            span = row["cycle"]
    return dict(totals), span


def classify(
    totals: dict[tuple[str, str], int],
) -> tuple[list[Contribution], dict[str, int]]:
    """Split counters into cycle-bearing contributions and volume counters."""
    contributions: list[Contribution] = []
    volumes: dict[str, int] = defaultdict(int)
    for (unit, counter), value in totals.items():
        if counter in _REDUNDANT or value <= 0:
            continue
        cycle_bearing = (
            counter.endswith("_cycles")
            or counter in _EXTRA_CYCLE_COUNTERS
            or counter.startswith(STALL_PREFIX)
        )
        if not cycle_bearing:
            volumes[counter] += value
            continue
        described = _DESCRIPTIONS.get(counter, "")
        contributions.append(
            Contribution(
                unit=unit,
                counter=counter,
                cycles=value,
                described=described,
                discovered=not described,
            )
        )
    contributions.sort(key=lambda c: (-c.cycles, c.unit, c.counter))
    return contributions, dict(volumes)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build(
    counters_dir: Path | str | None,
    hotspots: dict | None = None,
    elfs: list[dict] | None = None,
    cost_model: bool | None = None,
    label: str = "",
    notes: list[str] | None = None,
) -> Report:
    report = Report(cost_model=cost_model, label=label, notes=list(notes or []))
    if counters_dir is not None and Path(counters_dir).is_dir():
        totals, span = load_counters(counters_dir)
        report.span = span
        report.contributions, report.volumes = classify(totals)
    report.hotspots = hotspots or {}
    report.elfs = elfs or []
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _pct(value: int, total: int) -> str:
    if not total:
        return "n/a"
    return f"{100.0 * value / total:.1f} %"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def render(report: Report, top: int = 25) -> str:
    span = report.span
    lines: list[str] = []
    title = (
        f"tt-sim bottleneck report — {report.label}"
        if report.label
        else "tt-sim bottleneck report"
    )
    lines.append(f"# {title}")
    lines.append("")
    lines.append(FRAMING)
    lines.append("")

    # -- run ---------------------------------------------------------------
    regime = {True: "on", False: "off", None: "unknown"}[report.cost_model]
    lines.append("## Run")
    lines.append("")
    lines += _table(
        ["", ""],
        [
            ["cycle span observed", f"{span:,}"],
            ["cost model (`TT_SIM_COST_MODEL`)", regime],
            ["units seen", str(len(report.per_unit()))],
        ],
    )
    if regime == "off":
        lines.append("")
        lines.append(
            "**The cost model is off, so nothing here can stall.** Occupancy and "
            "stall rows are absent rather than zero — an un-modelled run makes no "
            "claim about where its time went. Re-run with `TT_SIM_COST_MODEL=1`."
        )
    for note in report.notes:
        lines.append("")
        lines.append(f"_{note}_")
    lines.append("")

    # -- 1. ranked attribution --------------------------------------------
    lines.append("## 1. Where the modelled cycles went")
    lines.append("")

    shared = report.shared_resource_cycles()
    if shared:
        total = sum(shared.values())
        lines.append("**Headline — the shared resources the model can price.**")
        lines.append("")
        lines += _table(
            ["resource", "cycles", "of span"],
            [
                [k, f"{v:,}", _pct(v, span)]
                for k, v in sorted(shared.items(), key=lambda kv: -kv[1])
            ]
            + [
                [
                    "**total against a published number**",
                    f"**{total:,}**",
                    f"**{_pct(total, span)}**",
                ]
            ],
        )
        lines.append("")
        lines.append(
            "These are the two families that occupy a resource the whole device "
            "shares, so their sum against the span is meaningful in a way the "
            "per-core rows below are not. **RV stall cycles are deliberately "
            "excluded**: a core can be charged far more stall than the run is "
            "long, and charged is not delivered — the run's length is set by when "
            "cores meet each other, not by a sum of charges."
        )
        lines.append("")

    if not report.contributions:
        lines.append("_No cycle-bearing counters in this run._")
    else:
        rows = []
        for c in report.contributions[:top]:
            name = c.counter + (" *(discovered)*" if c.discovered else "")
            rows.append(
                [
                    c.unit,
                    name,
                    f"{c.cycles:,}",
                    _pct(c.cycles, span),
                    c.described or "—",
                ]
            )
        lines += _table(["unit", "counter", "cycles", "of span", "what it is"], rows)
        rest = report.contributions[top:]
        if rest:
            lines.append(
                f"| … | {len(rest)} more rows | {sum(c.cycles for c in rest):,} | "
                f"{_pct(sum(c.cycles for c in rest), span)} | — |"
            )
        lines.append("")
        lines.append(
            "Units run **concurrently**, so these shares are each against the whole "
            "span and do not sum to 100 %. A share above 100 % is not a bug: a core "
            "can be charged more stall cycles than the run is long when its stalls "
            "overlap other cores' work. Rank by the column, not by the sum."
        )
    lines.append("")

    # -- 2. per unit -------------------------------------------------------
    per_unit = report.per_unit()
    if per_unit:
        lines.append("### Per unit, and what the model could not name")
        lines.append("")
        rows = []
        for unit, cycles in sorted(per_unit.items(), key=lambda kv: -kv[1]):
            unnamed = max(0, span - cycles)
            rows.append(
                [
                    unit,
                    f"{cycles:,}",
                    _pct(cycles, span),
                    f"{unnamed:,}" if cycles <= span else "0 (over-subscribed)",
                    _pct(unnamed, span) if cycles <= span else "—",
                ]
            )
        lines += _table(
            ["unit", "named cycles", "of span", "unattributed", "of span"], rows
        )
        lines.append("")
        lines.append(
            "**`unattributed` is the honest column.** It is span minus everything "
            "the model can name for that unit: idle, waiting on a peer, or spent in "
            "a mechanism nothing charges yet. A large value is a gap in the model, "
            "not a quiet cycle."
        )
        lines.append("")

    # -- 3. source-level hotspots -----------------------------------------
    lines.append("## 2. Source-level hotspots")
    lines.append("")
    rows_in = report.hotspots.get("functions") or []
    if not rows_in:
        lines.append("_No per-PC data — the hotspot aggregator was not enabled._")
    else:
        rows = []
        for h in rows_in[:top]:
            reasons = h.get("by_reason") or {}
            top_reason = (
                max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else "—"
            )
            rows.append(
                [
                    h["unit"],
                    h["location"],
                    f"{h['cycles']:,}",
                    _pct(h["cycles"], span),
                    f"{h['stall_cycles']:,}",
                    top_reason,
                ]
            )
        lines += _table(
            [
                "unit",
                "function (file:line)",
                "cycles",
                "of span",
                "of which stalled",
                "top reason",
            ],
            rows,
        )
        lines.append("")
        lines.append(
            "A row's `cycles` is one per instruction retired there plus every cycle "
            "the core was held before issuing it. Rows are folded per function, so "
            "an inlined callee is named rather than the `kernel_main` it vanished "
            "into."
        )
    lines.append("")

    # -- 4. attribution coverage ------------------------------------------
    lines.append("## 3. Attribution coverage")
    lines.append("")
    hs = report.hotspots
    if hs:
        total = hs.get("total_cycles", 0)
        resolved = hs.get("resolved_cycles", 0)
        lines += _table(
            ["", "cycles", "share"],
            [
                ["PC cycles sampled", f"{total:,}", "100.0 %"],
                ["resolved to source", f"{resolved:,}", _pct(resolved, total)],
                [
                    "unresolved (no DWARF)",
                    f"{total - resolved:,}",
                    _pct(total - resolved, total),
                ],
            ],
        )
        lines.append("")
        missing = hs.get("unattributed_units") or []
        if missing:
            lines.append(
                f"No ELF was found for: **{', '.join(missing)}**. Their PCs appear "
                "as bare addresses above. Point `TT_SIM_PROFILE_ELFS="
                "<UNIT>:<path>,…` at the right ELFs to name them."
            )
            lines.append("")
    if report.elfs:
        lines.append("ELFs used for attribution:")
        lines.append("")
        lines += _table(
            ["unit", "role", "chosen how", "path"],
            [
                [e["unit"], e.get("role", "kernel"), e["how"], "`" + e["path"] + "`"]
                for e in report.elfs
            ],
        )
        lines.append("")
        lines.append(
            "`verified` means the ELF's loadable segments were compared byte for "
            "byte against what is resident in simulated memory — proof, not "
            "inference. `relocated +0x…` is the same proof for a kernel tt-metal "
            "placed somewhere other than its link address; the offset was "
            "recovered by finding the ELF's own text in L1. `recent` means it is "
            "the newest matching ELF in the tt-metal build cache and could "
            "**not** be confirmed against the device; treat those rows as a "
            "strong guess. `no match` means the device *was* readable and no "
            "candidate's code is what ran — that unit is deliberately left "
            "unattributed rather than labelled with some other kernel's function "
            "names. `explicit` came from `TT_SIM_PROFILE_ELFS`."
        )
    elif hs:
        lines.append(
            "_No ELFs were discovered, so no PC could be named._ Set "
            "`TT_METAL_CACHE` or `TT_SIM_PROFILE_ELFS`."
        )
    lines.append("")

    # -- 5. volumes --------------------------------------------------------
    if report.volumes:
        lines.append("## 4. Volume counters (not cycles)")
        lines.append("")
        top_vols = sorted(report.volumes.items(), key=lambda kv: -kv[1])[:15]
        lines += _table(["counter", "total"], [[k, f"{v:,}"] for k, v in top_vols])
        lines.append("")

    # -- 6. caveats --------------------------------------------------------
    lines.append("## 5. What this report is not")
    lines.append("")
    lines.append(
        "- **Not a prediction of silicon.** See the framing at the top. The cost "
        "model's own instalment log (`docs/plans/cost-model.md`) records, for every "
        "term, the published bound it came from and what remains uncharged.\n"
        "- **Not a partition of the run.** Charged is not delivered: a core can be "
        "charged tens of thousands of stall cycles that never reach the total, "
        "because the run's length is set by when cores meet each other, not by a "
        "sum of charges.\n"
        "- **Not stable across cost-model changes.** Each new charged term moves "
        "these totals by design. Compare two runs of the same tree, not a run "
        "against a number written down last month."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence + CLI
# ---------------------------------------------------------------------------


def hotspots_to_dict(table, top: int = 200) -> dict:
    """Serialise a :class:`~tt_sim.trace.hotspots.HotspotTable`."""
    return {
        "total_cycles": table.total_cycles(),
        "resolved_cycles": table.resolved_cycles(),
        "unattributed_units": table.unattributed_units,
        "functions": [
            {
                "unit": h.unit,
                "location": h.location(),
                "function": h.function,
                "file": h.file,
                "line": h.line,
                "cycles": h.cycles,
                "retired": h.retired,
                "stall_cycles": h.stall_cycles,
                "frontend_stalls": h.frontend_stalls,
                "by_reason": h.by_reason,
            }
            for h in table.by_function()[:top]
        ],
        "pcs": [
            {
                "unit": h.unit,
                "pc": h.pc,
                "location": h.location(),
                "cycles": h.cycles,
                "retired": h.retired,
                "stall_cycles": h.stall_cycles,
                "by_reason": h.by_reason,
            }
            for h in table.ranked(top)
        ],
    }


def write(report: Report, directory: Path | str, top: int = 25) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["attributed_cycles"] = report.attributed()
    payload["shared_resource_cycles"] = report.shared_resource_cycles()
    # report.json carries every row; report.md is truncated to ``top`` so it
    # stays readable. Nothing is lost, only paginated.
    (directory / "report.json").write_text(json.dumps(payload, indent=2))
    path = directory / "report.md"
    path.write_text(render(report, top=top))
    return path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m tt_sim.trace.report",
        description="Render a ranked bottleneck report from a TT_SIM_PROFILE run.",
    )
    parser.add_argument("directory", help="the TT_SIM_PROFILE output directory")
    parser.add_argument("--top", type=int, default=25, help="rows per ranked table")
    parser.add_argument(
        "--stdout", action="store_true", help="print instead of writing"
    )
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    meta_path = directory / "profile.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    hotspots_path = directory / "hotspots.json"
    hotspots = json.loads(hotspots_path.read_text()) if hotspots_path.exists() else {}

    report = build(
        directory / "counters",
        hotspots=hotspots,
        elfs=meta.get("elfs"),
        cost_model=meta.get("cost_model"),
        label=meta.get("label", directory.name),
        notes=meta.get("notes"),
    )
    if args.stdout:
        print(render(report, top=args.top))
        return 0
    out = write(report, directory, top=args.top)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
