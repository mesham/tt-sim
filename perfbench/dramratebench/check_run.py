#!/usr/bin/env python3
"""Grade a dramratebench run, in the order its gates have to be applied.

Two jobs, and they are separate on purpose.

``--read-control`` re-derives the READ arm's readings from a recorded file and
compares them to figures pinned in this module. There is one pinned control PER
ARCHITECTURE and the file's own ``arch=`` selects which applies; a control that
does not apply SKIPS, with the reason printed, and never fails. Grading a
Blackhole run against Wormhole's pins is not a regression, it is a category
error, and the first time it happened it printed sixteen confident FAILs
comparing 46 B/cycle to 22. Each control's default target is its own committed
card CSV, which is immutable, so what this actually protects is the *analysis*:
if the write direction ever changed how a read row is named, columned or
aggregated, this stops being re-derivable and says so. The read arm is the
control every other DRAM reading in this project is placed against, and "the
write arm perturbed it" must be a loud failure rather than a slow drift.

``--measured`` grades a WRITE run. The gates are the read arm's, in the read
arm's order, plus the two the write direction needs and the read direction
cannot use -- see ``check_write`` for what each catches and, more importantly,
what none of them can.

Deliberately standalone: standard library only, so it runs on a card box with
nothing but ``perfbench/`` rsynced onto it and no tt-sim at all. It duplicates
some arithmetic the home-side ``dram_rate_sweep`` module also does, and that is
the point -- a check that shares an implementation with the thing it checks
agrees with it by construction.

It is NOT a consumer of the cost tables and must not become one. The two
ceilings below are transcribed constants with their source named in a comment;
nothing here imports them, which is what lets this file run on a box where the
simulator does not exist.

    ./check_run.py --read-control
    ./check_run.py --measured dramratebench-wormhole.csv
    ./check_run.py --measured new.csv --schema-against <a recorded read file>

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_SESSIONS = _REPO / "perfbench" / "card-sessions"


@dataclass(frozen=True)
class ReadControl:
    """One architecture's pinned DRAM read arm.

    Every figure is a BAND, ``(lo, hi)`` inclusive, not a point. A point pin
    only says "this is reproducible to the tolerance"; a band additionally says
    HOW reproducible, and the two are not the same claim when a reading is
    jittery. Where a band is wide, the width is the measured spread across the
    sessions named in ``sessions`` and the comment beside it says so -- a band
    widened to make a check pass, rather than because the card moved, is worse
    than no check.
    """

    arch: str
    csv: Path  # the committed session this control re-derives from
    sessions: tuple[str, ...]  # every session the bands were set from
    onechan: dict[int, tuple[float, float]]
    flat_band: tuple[float, float]
    flat_except: tuple[int, ...]
    fanchan_scale: tuple[tuple[float, float], ...]  # (lo point, hi point, ratio)
    onechan_scale: tuple[tuple[float, float], ...]
    # Wormhole only: a published vendor dataset to place the readings against.
    vendor_gb_s: dict[int, float] = field(default_factory=dict)
    deviation_pct: dict[int, float] = field(default_factory=dict)
    # Blackhole: no vendor DRAM table is transcribed in this project, so the
    # GB/s figures are pinned directly and are a re-derivation guard only.
    gb_s: dict[int, tuple[float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wormhole. Two sessions, both 2026-08-17 on the same p150:
#   A  card-sessions/2026-08-17-wormhole-B/dram.wormhole.csv   (read arm only)
#   B  card-sessions/2026-08-17-wh-dramwrite/run1.csv          (read + write)
#
# Every band below is `[min(A,B) - 2%, max(A,B) + 2%]`: the union of the two
# sessions' own re-derivation windows, since a single median-of-three is only
# good to the 2% `TOL` allows in the first place. Where A and B agree the band
# is therefore no looser than the point pin it replaces; where they disagree it
# is exactly as loose as the disagreement, and the comment gives both readings.
# ---------------------------------------------------------------------------
WORMHOLE = ReadControl(
    arch="wormhole",
    csv=_SESSIONS / "2026-08-17-wormhole-B" / "dram.wormhole.csv",
    sessions=("2026-08-17-wormhole-B", "2026-08-17-wh-dramwrite/run1"),
    # Median-over-repeats aggregate for `onechan`, per reader count.
    onechan={
        1: (21.703, 22.590),  # A 22.146  B 22.147   spread 0.0%
        # THE JITTERY POINT. A read 20.344 and it was written down as an
        # anomaly at the time -- 8% below both its neighbours, unexplained. B
        # read 22.225, i.e. right on the plateau with everything else. So the
        # anomaly does not reproduce, and the old point pin of 20.344 was
        # pinning a one-off: it failed B by +9.2% for being BETTER behaved.
        # The band spans both, and is 13.7% wide because that is what two
        # sessions of the same sweep at the same point actually did.
        2: (19.937, 22.670),  # A 20.344  B 22.225   spread 9.2%
        # Ordinary run-to-run variation, the largest anywhere else in the
        # sweep. B's 21.546 is 2.6% under A's 22.127 -- outside the 2% a
        # single median is trusted to, inside anything that would be a
        # regression, and the band is that spread and no more.
        4: (21.114, 22.570),  # A 22.127  B 21.546   spread 2.7%
        8: (21.463, 22.535),  # A 21.901  B 22.093   spread 0.9%
        12: (21.521, 22.418),  # A 21.961  B 21.977   spread 0.1%
        16: (21.588, 22.535),  # A 22.029  B 22.093   spread 0.3%
        24: (21.563, 22.503),  # A 22.004  B 22.061   spread 0.3%
        32: (21.636, 22.527),  # A 22.078  B 22.085   spread 0.0%
        48: (21.648, 22.556),  # A 22.091  B 22.113   spread 0.1%
        64: (21.640, 22.532),  # A 22.082  B 22.090   spread 0.0%
    },
    # The flat band, stated separately from the per-point figures and excluding
    # n=2. This is the claim "flat from 1 to 64 readers" as something that can
    # fail, so it is deliberately tighter than the per-point bands: it is a
    # statement about the SHAPE of the sweep, and the shape is reproducible even
    # where individual points wobble.
    #
    # 21.546 to 22.147 is the span the two sessions hold across every reader
    # count except two, rounded outward to the nearest 0.05. The old lower edge
    # of 21.85 was set from A alone and B's n=4 sits under it; 21.50 is set from
    # both. Still only 3.3% wide, so a sweep that actually scaled -- the thing
    # this exists to catch -- misses it by a factor, not by a percent.
    flat_band=(21.50, 22.20),
    # n=2 is excluded, and now for a better reason than "A was odd there": the
    # two sessions disagree by 9% at that one point, so it cannot support a
    # claim about the shape in either direction.
    flat_except=(2,),
    # The control arm, from the LAST repeat -- which is what the program's own
    # scaling verdict uses, and it is not the median. Both are kept because a
    # run that changed which one the verdict reads would otherwise pass
    # silently. B scaled BETTER than A (x5.96 against x5.74) on a fan-out whose
    # high point is 12 channels' worth of readers competing; the 3.4% spread at
    # n=64 is the whole of the difference and neither end is anomalous.
    fanchan_scale=((21.688, 22.663), (124.991, 134.492), (5.625, 6.078)),
    onechan_scale=((21.699, 22.659), (21.660, 22.560), (0.975, 1.019)),
    # Against `wh_dram#performance` (reads, static VC, 1 MiB per tile, ONE
    # channel) at the vendor's own 1 / 12 / 48 tiles, converted at the 1000 MHz
    # the device reported. Per cent deviation of measured from published; both
    # sessions land on the same figures to a tenth, so these stay points.
    vendor_gb_s={1: 22.2, 12: 22.3, 48: 22.3},
    deviation_pct={1: -0.2, 12: -1.5, 48: -0.9},
)

# ---------------------------------------------------------------------------
# Blackhole, from card-sessions/2026-08-17-bh-dramwrite/run1.csv (p150, 1350
# MHz, 8 banks). ONE session, so every band here is that session's figure at
# the same 2% a single median is trusted to -- these are point pins wearing the
# band's clothes, and they will only become real bands when a second Blackhole
# read arm exists to set the spread. Said out loud because a one-session band
# cannot distinguish "reproducible" from "measured once".
# ---------------------------------------------------------------------------
BLACKHOLE = ReadControl(
    arch="blackhole",
    csv=_SESSIONS / "2026-08-17-bh-dramwrite" / "run1.csv",
    sessions=("2026-08-17-bh-dramwrite/run1",),
    onechan={
        1: (45.352, 47.204),  # 46.278
        2: (45.219, 47.065),  # 46.142
        4: (45.479, 47.337),  # 46.408
        8: (44.757, 46.585),  # 45.671 -- the low point of the sweep
        12: (45.271, 47.120),  # 46.196
        16: (45.690, 47.556),  # 46.623
        24: (46.134, 48.018),  # 47.076
        32: (46.191, 48.077),  # 47.134
        48: (46.200, 48.087),  # 47.144
        120: (46.215, 48.102),  # 47.159
    },
    # 45.671 to 47.159 across 1 -> 120 participants, rounded outward to the
    # nearest 0.05. Wider in relative terms than Wormhole's (3.4% against 3.3%)
    # and for a reason worth writing down: this sweep is not flat like
    # Wormhole's, it DRIFTS UP 1.9% from one participant to 120 and dips at
    # eight. The band is a bound on a mild ramp, not a plateau, and it is the
    # ramp that a second session should be used to confirm or kill.
    flat_band=(45.65, 47.20),
    flat_except=(),
    # 46.368 -> 227.610 B/cycle over 1 -> 120 participants, x4.91.
    fanchan_scale=((45.441, 47.296), (223.058, 232.163), (4.810, 5.007)),
    onechan_scale=((45.368, 47.221), (46.196, 48.083), (0.997, 1.039)),
    # No vendor Blackhole DRAM read table is transcribed in this project, so
    # there is nothing to place these against and no deviation to pin. They are
    # the session's own GB/s at the vendor-comparable tile counts, kept so that
    # a change in the clock or in the B/cycle -> GB/s conversion fails here.
    gb_s={
        1: (61.226, 63.726),  # 62.476 GB/s
        12: (61.116, 63.612),  # 62.364
        48: (62.371, 64.917),  # 63.644
    },
)

CONTROLS = {c.arch: c for c in (WORMHOLE, BLACKHOLE)}

# Back-compat for anything that imported the flat names.
READ_CONTROL_CSV = WORMHOLE.csv

# ---------------------------------------------------------------------------
# The two ceilings a plateau can sit on, per arch. QUOTED from
# `tt_sim/perf/unit_costs.yaml`, not derived here and not modifiable here: this
# module reads a CSV and must run where tt-sim does not exist.
# ---------------------------------------------------------------------------
CEILINGS = {
    # channel read, channel write, NoC link -- B/cycle
    "wormhole": (24.0, 24.0, 32.0),
    # Blackhole's read rate is `vendor_source_derived`; its WRITE rate is None,
    # meaning no source gives one and the model charges no write occupancy at
    # all. That absence is a prediction in its own right and the reason this
    # table carries a `None` rather than repeating the read figure.
    "blackhole": (47.0805, None, 64.0),
}

TOL = 0.02  # 2% on a re-derived figure; these are medians of three repeats


def read_rows(path):
    """``(rows, meta)`` from a dramratebench CSV, comments parsed into meta."""
    meta, lines = {}, []
    with open(path) as handle:
        for raw in handle:
            if raw.startswith("#"):
                for token in raw.lstrip("# ").split():
                    if "=" in token:
                        key, _, value = token.partition("=")
                        meta.setdefault(key, value)
                continue
            lines.append(raw)
    return list(csv.DictReader(lines)), meta


def _num(row, key, default=0.0):
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def by_reader_count(rows, arm, how="median"):
    """``{num_readers: aggregate B/cycle}`` for one arm, over the repeats."""
    buckets = {}
    for row in rows:
        if row.get("arm") != arm:
            continue
        value = _num(row, "agg_bytes_per_cycle")
        if value <= 0:
            continue
        buckets.setdefault(int(_num(row, "num_readers")), []).append(value)
    if how == "last":
        return {n: v[-1] for n, v in buckets.items()}
    return {n: statistics.median(v) for n, v in buckets.items()}


def _close(got, want, tol=TOL):
    return abs(got - want) <= abs(want) * tol


def _band(got, band):
    lo, hi = band
    return lo <= got <= hi


def check_read_control(path=None):
    """Re-derive the pinned read control that applies to ``path``.

    Yields ``(ok, line)``; ``ok`` is ``None`` for a SKIP, which is neither a
    pass nor a failure and is the answer whenever a control does not apply to
    the file in hand. With no ``path``, every control is re-derived from its own
    committed session.

    Nothing here touches the write direction, and that is the whole point: this
    is the assertion that adding one did not move the read arm.
    """
    if path is None:
        for control in CONTROLS.values():
            yield from _grade_control(control, control.csv)
        return

    path = Path(path)
    if not path.exists():
        yield False, f"read control CSV not found: {path}"
        return
    rows, meta = read_rows(path)
    arch = meta.get("arch")

    # A file that does not say what it ran on cannot be placed against any
    # control, and guessing is how the Blackhole run got graded as a Wormhole.
    if not arch:
        yield False, f"{path.name} carries no `arch=` and cannot be graded"
        return

    for name, control in CONTROLS.items():
        if name != arch:
            yield (
                None,
                f"the {name} control does not apply: {path.name} is a {arch} "
                f"run. Its figures would be compared against a different part's "
                f"and every one of them would fail for the wrong reason.",
            )
    if arch not in CONTROLS:
        yield (
            None,
            f"no read control is pinned for arch={arch}. Record one from a "
            f"session on that part before this file can be graded.",
        )
        return
    yield from _grade_control(CONTROLS[arch], path, rows, meta)


def _grade_control(control, path, rows=None, meta=None):
    """Every gate of one architecture's read control, against one file."""
    path = Path(path)
    if rows is None:
        if not path.exists():
            yield False, f"{control.arch} read control CSV not found: {path}"
            return
        rows, meta = read_rows(path)
    yield (
        meta.get("arch") == control.arch,
        f"{path.name} is a {control.arch} run (arch={meta.get('arch')}), "
        f"graded against the control pinned from {', '.join(control.sessions)}",
    )

    one = by_reader_count(rows, "onechan")
    fan_last = by_reader_count(rows, "fanchan", how="last")
    one_last = by_reader_count(rows, "onechan", how="last")

    missing = sorted(set(control.onechan) - set(one))
    yield (
        not missing,
        f"every pinned reader count is present in the onechan arm (missing: {missing})",
    )
    for n, band in sorted(control.onechan.items()):
        if n not in one:
            continue
        lo, hi = band
        yield (
            _band(one[n], band),
            f"onechan n={n:<3} median {one[n]:.3f} B/cycle in pinned {lo:.3f}-{hi:.3f}",
        )

    lo, hi = control.flat_band
    outside = sorted(
        n for n, v in one.items() if n not in control.flat_except and not lo <= v <= hi
    )
    yield (
        not outside,
        f"onechan is FLAT in {lo}-{hi} B/cycle at every participant count except "
        f"{control.flat_except or 'none'} (outside: {outside})",
    )

    for arm, series, pinned in (
        ("fanchan", fan_last, control.fanchan_scale),
        ("onechan", one_last, control.onechan_scale),
    ):
        if not series:
            yield False, f"{arm} has no measured points"
            continue
        n_lo, n_hi = min(series), max(series)
        band_lo, band_hi, band_scale = pinned
        got_scale = series[n_hi] / series[n_lo] if series[n_lo] else 0.0
        yield (
            _band(series[n_lo], band_lo)
            and _band(series[n_hi], band_hi)
            and _band(got_scale, band_scale),
            f"{arm} {series[n_lo]:.3f} -> {series[n_hi]:.3f} B/cycle over "
            f"{n_lo} -> {n_hi} (x{got_scale:.2f}) in pinned "
            f"{band_lo[0]:.3f}-{band_lo[1]:.3f} -> {band_hi[0]:.3f}-{band_hi[1]:.3f} "
            f"(x{band_scale[0]:.2f}-{band_scale[1]:.2f})",
        )

    clock = float(meta.get("clock_mhz") or 0)
    yield clock > 0, f"the control file carries a clock ({clock:.0f} MHz)"
    for n, want_pct in sorted(control.deviation_pct.items()):
        if n not in one or clock <= 0:
            yield False, f"no onechan point at n={n} to compare to the vendor table"
            continue
        got_gb = one[n] * clock / 1000.0
        pct = 100.0 * (got_gb - control.vendor_gb_s[n]) / control.vendor_gb_s[n]
        # 0.15 percentage points: the published deviations are quoted to one
        # decimal, so anything tighter is checking the rounding.
        yield (
            abs(pct - want_pct) <= 0.15,
            f"n={n:<3} {got_gb:.2f} GB/s vs published {control.vendor_gb_s[n]}: "
            f"{pct:+.1f}% vs pinned {want_pct:+.1f}%",
        )
    for n, band in sorted(control.gb_s.items()):
        if n not in one or clock <= 0:
            yield False, f"no onechan point at n={n} to convert to GB/s"
            continue
        got_gb = one[n] * clock / 1000.0
        yield (
            _band(got_gb, band),
            f"n={n:<3} {got_gb:.2f} GB/s in pinned {band[0]:.2f}-{band[1]:.2f} "
            f"(no published figure to place it against)",
        )


def check_schema(path, against=None):
    """Does a new file still carry the read arm's schema, unchanged?

    Appending columns is safe -- every consumer resolves them by name -- but
    RENAMING or REORDERING the eighteen the control was recorded with is not,
    and neither is renaming an arm. Checked as a prefix so the write
    direction's own columns are free to exist.
    """
    against = Path(against) if against else READ_CONTROL_CSV
    if not against.exists():
        yield False, f"schema reference not found: {against}"
        return
    with open(against) as handle:
        ref = [line for line in handle if not line.startswith("#")]
    with open(path) as handle:
        got = [line for line in handle if not line.startswith("#")]
    ref_cols = next(csv.reader(ref), [])
    got_cols = next(csv.reader(got), [])
    yield (
        got_cols[: len(ref_cols)] == ref_cols,
        f"the first {len(ref_cols)} columns are the control's, in its order "
        f"(got {got_cols[: len(ref_cols)]})",
    )

    rows, _ = read_rows(path)
    read_rows_ = [r for r in rows if r.get("direction", "read") == "read"]
    yield (
        all(r.get("arm") in ("onechan", "fanchan", "samecore") for r in read_rows_),
        "every read row still uses an unsuffixed arm name",
    )
    # A read row whose write columns were filled in would mean the host graded a
    # read with a check that cannot fail. -1 is "the host did not look".
    if any("witness_ok" in r for r in read_rows_):
        yield (
            all(int(_num(r, "witness_ok", -1)) == -1 for r in read_rows_),
            "every read row leaves witness_ok at -1 ('the host did not look')",
        )
        yield (
            all(int(_num(r, "stray_writes", -1)) == -1 for r in read_rows_),
            "every read row leaves stray_writes at -1",
        )


def check_write(path):
    """Grade a write run. Each gate can only ever FAIL the run.

    In order, and the order is the design:

    1. every writer stamped a result at all;
    2. every writer's own read-back-after-write matched (``tags_ok``) -- the
       bytes were accepted and landed. Catches a dropped, mis-congruent or
       never-issued write. CANNOT catch misdirection: it re-reads through the
       coordinate the write used;
    3. every writer's target block holds that writer's witness, as read back by
       the HOST over PCIe (``witness_ok``) -- the positive half of misdirection;
    4. no block the point did not target moved off POISON (``stray_writes``) --
       the negative half, and the only gate here that can catch a write which
       went somewhere else entirely;
    5. somebody waited at the barrier, or the bursts never overlapped;
    6. the fan-out control MOVED, or something upstream caps both arms;
    7. only then the ratio, and only then the level.

    What none of them can do: separate the endpoint from the one inbound router
    link every concentrated flow converges on. The ``samecore`` arm is what
    would, and tt-metal fronts every bank on its own NoC coordinate on both
    parts, so it does not exist here. Report "the endpoint", never "the
    channel".
    """
    rows, meta = read_rows(path)
    writes = [r for r in rows if r.get("direction") == "write"]
    if not writes:
        yield False, "the file has no write rows (was --dir write passed?)"
        return

    def _bad(pred):
        return [
            f"{r['arm']}/n={r['num_readers']}/rep={r.get('repeat')}"
            for r in writes
            if pred(r)
        ]

    unstamped = _bad(
        lambda r: int(_num(r, "measured_readers")) != int(_num(r, "num_readers"))
    )
    yield not unstamped, f"every writer stamped a result ({unstamped[:4] or 'ok'})"

    dropped = _bad(lambda r: int(_num(r, "tags_ok")) != int(_num(r, "num_readers")))
    yield (
        not dropped,
        "every writer read its own bytes back after the barrier -- none was "
        f"dropped or mis-congruent ({dropped[:4] or 'ok'})",
    )

    misdirected = _bad(
        lambda r: int(_num(r, "witness_ok", -1)) != int(_num(r, "num_readers"))
    )
    yield (
        not misdirected,
        "the HOST found every writer's witness in the block the plan named "
        f"({misdirected[:4] or 'ok'})",
    )

    strays = _bad(lambda r: int(_num(r, "stray_writes", -1)) != 0)
    yield (
        not strays,
        "no block the plan did not name moved off POISON -- nothing landed "
        f"anywhere else ({strays[:4] or 'ok'})",
    )

    multi = [r for r in writes if int(_num(r, "num_readers")) > 1]
    yield (
        not multi or any(_num(r, "max_barrier_spins") > 0 for r in multi),
        "a writer waited at the barrier in some multi-writer point, so the "
        "bursts really did overlap",
    )

    fan = by_reader_count(rows, "fanchan-write")
    one = by_reader_count(rows, "onechan-write")
    if len(fan) < 2 or len(one) < 2:
        yield (
            False,
            "the write direction has no low/high pair in both arms, so nothing "
            "separates the endpoint from anything upstream of it (EXPECTED "
            "against tt-sim, which cannot sweep the writer count far)",
        )
        return
    n_lo, n_hi = min(fan), max(fan)
    fan_scale = fan[n_hi] / fan[n_lo] if fan[n_lo] else 0.0
    yield (
        fan_scale >= 1.5,
        f"the fan-out CONTROL moved: {fan[n_lo]:.3f} -> {fan[n_hi]:.3f} B/cycle "
        f"over {n_lo} -> {n_hi} writers (x{fan_scale:.2f})",
    )
    one_scale = one[n_hi] / one[n_lo] if one.get(n_lo) else 0.0
    eff = one_scale / fan_scale if fan_scale else 1.0
    yield (
        True,
        f"onechan-write x{one_scale:.2f} against the control's x{fan_scale:.2f}, "
        f"{100 * eff:.0f}% of it -> "
        f"{'ENDPOINT BOUND' if eff <= 0.75 else 'NO ENDPOINT BOUND'}",
    )

    # The level, and the comparison this direction exists for.
    read_one = by_reader_count(rows, "onechan")
    if n_hi in read_one and read_one[n_hi] > 0:
        ratio = one[n_hi] / read_one[n_hi]
        yield (
            True,
            f"at {n_hi} participants on the SAME endpoint: read "
            f"{read_one[n_hi]:.3f} B/cycle, write {one[n_hi]:.3f} B/cycle, "
            f"ratio {ratio:.3f}",
        )
    arch = meta.get("arch", "")
    if arch in CEILINGS:
        ch_r, ch_w, link = CEILINGS[arch]
        yield (
            True,
            f"{arch} ceilings: channel read {ch_r} B/cycle, channel write "
            f"{'unsourced -- no write occupancy is modelled' if ch_w is None else ch_w}, "
            f"NoC link {link}. A write plateau at the LINK is not a channel rate.",
        )
    yield (
        True,
        "OCCUPANCY only. This cannot price one write, cannot split service time "
        "from queueing, and cannot tell the endpoint from the one inbound "
        "router link. CORROBORATION, never provenance.",
    )


def _run(checks, label):
    """Print one section. Returns the failure count; SKIPs are not failures."""
    print(f"== {label}")
    failed = skipped = 0
    for ok, line in checks:
        if ok is True:
            print(f"   ok   {line}")
        elif ok is False:
            print(f"   FAIL {line}")
            failed += 1
        else:  # ok is None: does not apply, and says why
            print(f"   skip {line}")
            skipped += 1
    if skipped:
        print(f"   ({skipped} check(s) skipped -- see the reasons above)")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--read-control",
        nargs="?",
        const="",
        help="re-derive the read arm's pinned control for the file's own arch "
        "(default: every pinned control, each against its own committed CSV)",
    )
    ap.add_argument("--measured", help="a CSV with write rows, to grade")
    ap.add_argument(
        "--schema-against",
        nargs="?",
        const="",
        help="check --measured's read rows still carry the control's schema",
    )
    args = ap.parse_args(argv)
    if args.read_control is None and args.measured is None:
        args.read_control = ""  # the useful default on a bare invocation

    failed = 0
    if args.read_control is not None:
        failed += _run(
            check_read_control(args.read_control or None),
            "read arm: the pinned control for this file's arch, re-derived",
        )
    if args.measured and args.schema_against is not None:
        failed += _run(
            check_schema(args.measured, args.schema_against or None),
            "schema: the read arm's columns and arm names are unchanged",
        )
    if args.measured:
        failed += _run(check_write(args.measured), f"write arm: {args.measured}")
    print("----")
    print("PASS" if failed == 0 else f"FAIL ({failed} check(s))")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
