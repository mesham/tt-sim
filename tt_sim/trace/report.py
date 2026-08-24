"""Ranked bottleneck report — one page saying where a run's modelled
cycles went, and how much of the run the model could not name.

This is the consumer end of ``TT_SIM_PROFILE`` (see
:mod:`tt_sim.trace.auto`). It reads two artefacts a profiled run leaves
behind — the Parquet counter dataset and the per-PC hotspot table — and
writes ``report.md`` plus a machine-readable ``report.json`` (versioned
by :data:`SCHEMA_VERSION`, see ``docs/trace-schema.md`` §9).
Regenerate at any time without re-running the simulator::

    python3 -m tt_sim.trace.report <profile-dir>

**Counters are classified by pattern, not by a list.** ``stall_<reason>``
is an RV stall, ``tensix_stall_<reason>`` a Tensix thread stall, and
anything else ending in ``_cycles`` a cycle-bearing counter; all are
ranked, marked *(discovered)* where the report has no prose for them. That
is deliberate: counters are being added to this tree faster than any
hardcoded table would survive, and a report that silently omitted a new
stall reason would be wrong in the most expensive direction — it would
look complete. :func:`is_cycle_bearing` and :func:`is_redundant` are the
two rules, exported because every consumer of the dataset needs them.

The redundancy rule is the subtle half. One partition of a thread's lost
cycles is written down three ways — the per-reason rows, their
``*_stall_cycles`` total, and (for Tensix) a ``tensix_stall_on_<unit>``
re-cut by which unit was to blame — and ranking more than one of them
multiplies the stall.

**Shares are against the run's cycle span and units run concurrently**,
so per-source shares can and do sum past 100 %. The report says so in
its own body rather than in a docstring, because the number that matters
to a reader optimising a kernel is the *relative* ranking, and the way
to make that safe is to refuse to present a tidy pie.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Version of the **profile artefact** schema — ``report.json``,
#: ``hotspots.json`` (including the copy embedded in ``report.json`` as
#: ``hotspots``) and ``profile.json``. Written into each of them as
#: ``schema_version`` and documented in ``docs/trace-schema.md`` §9.
#:
#: Deliberately **not** ``Event.SCHEMA_VERSION``: that one is scoped to
#: event shape, so reusing it would bump this contract whenever an event
#: changed and leave it still when a report field changed — precisely the
#: wrong signal in both directions. Same style as
#: :data:`tt_sim.trace.state_dump.SCHEMA_VERSION`, which versions its own
#: artefact the same way.
#:
#: Bump on **any** change to a documented field: additively for a new
#: field, breaking for a rename, a removal, or a change of meaning or
#: unit. Either way, say which in ``docs/trace-schema.md`` §9.
SCHEMA_VERSION = 3

#: Counters that are cycle-bearing but not named ``*_cycles``.
_EXTRA_CYCLE_COUNTERS = {"instr_retired"}

#: The per-transaction NoC latency split: ``{leg: counter}``, in the order
#: a packet travels. The three **partition** ``noc_flight_cycles`` — the
#: aggregator charges them off the same two cycles, via
#: :func:`tt_sim.trace.events.noc_flight_split`, so they telescope to it by
#: construction — which makes each of them redundant with the total in the
#: sense :func:`is_redundant` means. They are ranked nowhere; they are
#: reported as their own block (:meth:`Report.noc_latency_split`) because
#: their value is the *shape* of the flight rather than its size.
NOC_SPLIT_COUNTERS = {
    "issue_to_injection": "noc_issue_to_injection_cycles",
    "injection_to_arrival": "noc_injection_to_arrival_cycles",
    "arrival_to_service": "noc_arrival_to_service_cycles",
}
#: Totals that restate cycles a per-reason partition already carries.
#: ``stall_cycles`` is the sum of the ``stall_<reason>`` rows and
#: ``tensix_stall_cycles`` the sum of the ``tensix_stall_<reason>`` rows, so
#: ranking either alongside its parts doubles every stall it names.
#: ``bookkeeping_cycles`` is the same shape from the other end -- a *subset* of
#: one unit's ``busy_cycles``, cut by whether the opcode moved any operand
#: data -- so ranking it beside ``busy_cycles`` double-counts every cycle it
#: names. It is published for the energy activity vector, which asks about
#: work rather than occupancy; query it directly.
_REDUNDANT = {
    "stall_cycles",
    "tensix_stall_cycles",
    "bookkeeping_cycles",
    *NOC_SPLIT_COUNTERS.values(),
}
#: Counts, not cycles, despite sitting under a stall prefix.
_STALL_VOLUMES = {"tensix_stall_episodes"}

STALL_PREFIX = "stall_"
TENSIX_STALL_PREFIX = "tensix_stall_"
#: ``tensix_stall_on_<unit>`` re-cuts the same cycles as
#: ``tensix_stall_<reason>``, by the unit blamed rather than the mechanism.
#: It is a second view of one partition, not extra time, and it is *partial* —
#: a semaphore or mutex wait blames no unit — so it is neither ranked nor
#: summed. Query it directly when the question is "which unit held this
#: thread up".
_BLAME_PREFIX = "tensix_stall_on_"


def is_redundant(counter: str) -> bool:
    """True for a counter that restates cycles another row already carries.

    Summing a redundant counter alongside the partition it restates is the
    single easiest way to produce a wrong total from this dataset, so the
    rule lives here rather than in each consumer.
    """
    return counter in _REDUNDANT or counter.startswith(_BLAME_PREFIX)


def is_cycle_bearing(counter: str) -> bool:
    """True when a counter's ``value`` is a count of cycles.

    Pattern-matched, never enumerated: a stall reason added anywhere in the
    tree is ranked with no change here. The complement is a *volume* counter
    — bytes, or a number of events — which must never be added to a cycle
    total.
    """
    if counter in _STALL_VOLUMES:
        return False
    return (
        counter.endswith("_cycles")
        or counter in _EXTRA_CYCLE_COUNTERS
        or counter.startswith(STALL_PREFIX)
        or counter.startswith(TENSIX_STALL_PREFIX)
    )


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
        if self.counter.startswith(STALL_PREFIX) or self.counter.startswith(
            TENSIX_STALL_PREFIX
        ):
            return "stall"
        return "occupancy"


def empty_noc_latency_split() -> dict[str, int]:
    """The split's shape, all zero — what a run with no NoC traffic reports.

    Always the full set of keys, never a partial dict: a missing key reads as
    a null to a consumer, and "we did not measure this" and "we measured this
    and it was zero" are the two things this whole block exists to keep apart.
    ``transactions`` is what tells them apart.
    """
    return {leg: 0 for leg in NOC_SPLIT_COUNTERS} | {"total": 0, "transactions": 0}


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
    #: Per-transaction NoC latency, split into the three legs of a packet's
    #: journey and summed over every NIU. See :func:`noc_latency_split`.
    noc_latency_split: dict[str, int] = field(default_factory=empty_noc_latency_split)

    def per_unit(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for c in self.contributions:
            out[c.unit] += c.cycles
        return dict(out)

    def attributed(self) -> int:
        return sum(c.cycles for c in self.contributions)

    #: Which counter rolls up into which shared-resource family.
    _SHARED_FAMILIES = {
        "noc_flight_cycles": "NoC flight and bandwidth",
        "busy_cycles": "Tensix backend occupancy",
    }

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

        **This is an occupancy total, not a fraction of the run.** It sums
        one counter over every unit that carries it — on Wormhole that is
        14 NIUs (each worker's NOC0 and NOC1, plus two per DRAM tile), and
        a transfer is counted at both of its endpoints. Divided by the
        span it routinely exceeds 100 %, which is the tell: a fraction of
        runtime cannot. Use :meth:`shared_resource_units` to normalise, and
        see the note the report prints beside the table for what to use
        instead when the question really is "how much of the span went to
        the NoC".
        """
        out: dict[str, int] = defaultdict(int)
        for c in self.contributions:
            family = self._SHARED_FAMILIES.get(c.counter)
            if family is not None:
                out[family] += c.cycles
        return dict(out)

    def shared_resource_units(self) -> dict[str, int]:
        """How many distinct units contributed to each shared-resource family.

        The denominator that turns :meth:`shared_resource_cycles` into a
        real fraction: ``cycles / (span * units)`` is bounded by 1, because
        it asks what share of the available link-cycles were occupied
        rather than what share of the run they represent.
        """
        seen: dict[str, set[str]] = defaultdict(set)
        for c in self.contributions:
            family = self._SHARED_FAMILIES.get(c.counter)
            if family is not None:
                seen[family].add(c.unit)
        return {k: len(v) for k, v in seen.items()}


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
        if is_redundant(counter) or value <= 0:
            continue
        if not is_cycle_bearing(counter):
            volumes[counter] += value
            continue
        described = _DESCRIPTIONS.get(counter, "")
        discovered = not described
        if not described and counter.startswith(TENSIX_STALL_PREFIX):
            # Structural, not enumerated: the shape of the name is enough to
            # say what the row is, so a stall reason invented tomorrow is
            # described rather than left blank.
            described = f"Tensix thread stall: {counter[len(TENSIX_STALL_PREFIX) :]}"
        contributions.append(
            Contribution(
                unit=unit,
                counter=counter,
                cycles=value,
                described=described,
                discovered=discovered,
            )
        )
    contributions.sort(key=lambda c: (-c.cycles, c.unit, c.counter))
    return contributions, dict(volumes)


def noc_latency_split(totals: dict[tuple[str, str], int]) -> dict[str, int]:
    """The three legs of NoC flight, summed over every NIU in the dataset.

    ``issue -> injection`` is queueing for the sending NIU's injection port,
    ``injection -> arrival`` is transit (hops, the packet's tail, and waiting
    for a router link another tile is using), and ``arrival -> service`` is
    time at the destination once the packet is there. ``total`` is
    ``noc_flight_cycles`` over the same units, so a consumer can check the
    three telescope to it; ``transactions`` is ``noc_txns_timed``, the
    denominator for a per-transaction mean.

    **``arrival_to_service`` is zero except at a DRAM tile, and that zero is
    the point.** tt-sim charges endpoint time only for a DRAM channel; arrival
    buffering, outstanding-transaction credit limits and response reordering
    are not modelled anywhere, so a hardware residual in this leg is entirely
    unattributed. The bucket is reported rather than omitted so that fact is
    legible instead of being something a reader has to know.

    Like every other total in this file this is an **occupancy** sum across
    units, counting a transfer at both of its endpoints — see §4.3a. It is a
    decomposition of `noc_flight_cycles`, so it inherits every caveat that
    number carries, including that dividing it by the span means nothing.
    """
    out = empty_noc_latency_split()
    for (_unit, counter), value in totals.items():
        for leg, name in NOC_SPLIT_COUNTERS.items():
            if counter == name:
                out[leg] += value
        if counter == "noc_flight_cycles":
            out["total"] += value
        elif counter == "noc_txns_timed":
            out["transactions"] += value
    return out


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
        report.noc_latency_split = noc_latency_split(totals)
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


#: What each leg of the NoC flight split is, and how much of it is modelled.
#: The third row is the one worth reading twice.
_SPLIT_PROSE = {
    "issue_to_injection": "Queued for the sending NIU's injection port. Modelled.",
    "injection_to_arrival": (
        "Transit: hops, the packet's own tail, and waiting for a router link "
        "another tile is using. Modelled."
    ),
    "arrival_to_service": (
        "At the destination once it arrived. **Modelled only as a DRAM "
        "channel's time — zero at every other endpoint.**"
    ),
}


def _render_noc_latency_split(report: Report) -> list[str]:
    """The per-transaction flight split, or nothing if no packet was timed."""
    split = report.noc_latency_split or {}
    txns = split.get("transactions", 0)
    if not txns:
        return []
    total = split.get("total", 0)
    lines = ["**Per-transaction NoC latency, split by leg.**", ""]
    lines += _table(
        ["leg", "cycles", "per txn", "share", "what it is"],
        [
            [
                leg.replace("_", " ").replace(" to ", " → "),
                f"{split.get(leg, 0):,}",
                f"{split.get(leg, 0) / txns:.1f}",
                _pct(split.get(leg, 0), total),
                _SPLIT_PROSE[leg],
            ]
            for leg in NOC_SPLIT_COUNTERS
        ]
        + [
            [
                "**total** (`noc_flight_cycles`)",
                f"**{total:,}**",
                f"**{total / txns:.1f}**",
                "100.0 %",
                f"Over {txns:,} timed transactions.",
            ]
        ],
    )
    lines.append("")
    lines.append(
        "The three legs **telescope** to the total — they are charged off the "
        "same two cycles, so they partition it by construction rather than by "
        "three accumulations agreeing. This is the same occupancy sum as the "
        "table above, counted at both endpoints of every transfer, so read the "
        "*shape* rather than the size."
    )
    lines.append("")
    lines.append(
        "**A `0.0` in `arrival → service` is a finding, not a measurement.** "
        "tt-sim charges endpoint time only for a DRAM channel; arrival "
        "buffering, outstanding-transaction credit limits and response "
        "reordering are not modelled anywhere. So the simulator claims zero "
        "endpoint queueing, and any residual hardware shows in this leg is "
        "entirely unmodelled — which is exactly why the row is printed rather "
        "than omitted. Silicon cannot answer this at all: a card has no "
        "per-transaction completion timestamp, only a barrier's start/end pair."
    )
    lines.append("")
    lines.append(
        "This row is summed over **every** NIU, so a run that touches DRAM "
        "mixes the one endpoint that charges with the many that do not. For "
        "the split by endpoint, read `noc_arrival_to_service_cycles` per unit "
        "in the counter dataset: it is zero on every worker NIU."
    )
    lines.append("")
    return lines


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
    regime = {
        True: "on",
        False: "off",
        None: "unknown (no profile.json alongside the counters)",
    }[report.cost_model]
    if report.cost_model is not True:
        # A functional-only run produces spans that look entirely plausible and
        # are the wrong regime -- measured 51,699 against 75,499 cycles on the
        # same nekbone program. Nothing downstream can tell the difference, so
        # the report says it where a reader cannot skip past it.
        lines.append(
            "> [!WARNING]\n"
            f"> **Cost model {regime.split(' ')[0]} — these cycle counts are "
            "not the modelled ones.** Without `TT_SIM_COST_MODEL=1` the run "
            "charges no modelled occupancy, so every span here is a functional "
            "lower bound *below* the documented floor, not the floor itself. "
            "The numbers look reasonable and rank plausibly; they are simply a "
            "different quantity. Re-run with `TT_SIM_COST_MODEL=1` before "
            "comparing anything against hardware."
        )
        lines.append("")
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
        units = report.shared_resource_units()
        total = sum(shared.values())
        lines.append("**Headline — the shared resources the model can price.**")
        lines.append("")
        lines += _table(
            ["resource", "cycles", "units", "occupancy"],
            [
                [k, f"{v:,}", str(units.get(k, 0)), _pct(v, span * units.get(k, 0))]
                for k, v in sorted(shared.items(), key=lambda kv: -kv[1])
            ]
            + [["**total against a published number**", f"**{total:,}**", "", ""]],
        )
        lines.append("")
        lines.append(
            "**`occupancy` is not the fraction of the run spent on this "
            "resource.** Each row sums one counter over every unit that carries "
            "it — the `units` column says how many — so the raw `cycles` figure "
            "counts a NoC transfer at both of its endpoints and can exceed the "
            "span several times over. `occupancy` divides by `span × units` to "
            "ask the answerable question instead: **what share of the available "
            'link-cycles were busy**. Do not quote either column as "this run '
            'was N % NoC-bound".'
        )
        lines.append("")
        lines.append(
            "For that question — **how much of a kernel's span went to the NoC** "
            "— use the NoC event decomposition (the `noc_events` tool; runbook "
            "\u00a74.3a), which buckets a core's own span into `issue` / "
            "`read_wait` / `write_wait` and is validated against silicon. It is a "
            "per-core partition of elapsed time; this table is an aggregate "
            "occupancy across links."
        )
        lines.append("")
        lines.append(
            "**RV stall cycles are deliberately excluded** from both columns: a "
            "core can be charged far more stall than the run is long, and charged "
            "is not delivered — the run's length is set by when cores meet each "
            "other, not by a sum of charges."
        )
        lines.append("")

    lines += _render_noc_latency_split(report)

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
        # Same contract as report.json: this dict is written to
        # ``hotspots.json`` *and* embedded in ``report.json``, so it carries
        # the version rather than depending on which file it was read from.
        "schema_version": SCHEMA_VERSION,
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
    payload["schema_version"] = SCHEMA_VERSION
    payload["attributed_cycles"] = report.attributed()
    payload["shared_resource_cycles"] = report.shared_resource_cycles()
    payload["shared_resource_units"] = report.shared_resource_units()
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
    if not meta_path.exists():
        # Profile artefacts are written when the *server* shuts down, which can
        # be seconds after the host exits -- a script that renders at host-exit
        # can arrive before profile.json does and would otherwise silently get
        # a report with an "unknown" cost-model regime and no ELF provenance.
        print(
            f"warning: no profile.json in {directory} -- rendering without run "
            "metadata, so the cost-model regime, ELF provenance and label are "
            "unknown. If the run has only just finished, wait for the simulator "
            "server to exit and render again.",
            file=sys.stderr,
        )
    hotspots_path = directory / "hotspots.json"
    hotspots = json.loads(hotspots_path.read_text()) if hotspots_path.exists() else {}

    # The run records where its counters landed, because
    # ``TT_SIM_TRACE_COUNTERS`` can put them outside the profile directory.
    # Falling back to the default keeps older profile directories readable.
    counters = Path(meta.get("counters") or (directory / "counters"))
    if not counters.is_dir():
        counters = directory / "counters"
    report = build(
        counters,
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
