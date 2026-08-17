#!/usr/bin/env python3
"""Grade a dramratebench run, in the order its gates have to be applied.

Two jobs, and they are separate on purpose.

``--read-control`` re-derives the READ arm's readings from a recorded file and
compares them to figures pinned in this module. Its default target is the
Wormhole card CSV of 2026-08-17, which is committed and immutable, so what this
actually protects is the *analysis*: if the write direction ever changed how a
read row is named, columned or aggregated, this stops being re-derivable and
says so. The read arm is the control every other DRAM reading in this project
is placed against, and "the write arm perturbed it" must be a loud failure
rather than a slow drift.

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
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

# ---------------------------------------------------------------------------
# The read arm's control, recorded 2026-08-17 on a Wormhole part.
# ---------------------------------------------------------------------------
# Every figure below was READ OFF the committed CSV named here, and the CSV is
# the authority; these are transcribed so that a change in how they are derived
# fails rather than silently producing different numbers from the same bytes.
READ_CONTROL_CSV = (
    _REPO
    / "perfbench"
    / "card-sessions"
    / "2026-08-17-wormhole-B"
    / "dram.wormhole.csv"
)

# Median-over-repeats aggregate for `onechan`, per reader count. The sweep is
# FLAT -- 22.1 to 22.2 B/cycle from 1 reader to 64 -- with ONE exception, at two
# readers, and it is written down rather than smoothed away: 20.344 is 8% below
# its neighbours on both sides and nothing in the file explains it. A band that
# quietly contained it would be a band wide enough to contain a real regression.
READ_CONTROL_ONECHAN = {
    1: 22.146,
    2: 20.344,
    4: 22.127,
    8: 21.901,
    12: 21.961,
    16: 22.029,
    24: 22.004,
    32: 22.078,
    48: 22.091,
    64: 22.082,
}
# The flat band, stated separately from the per-point figures and excluding the
# n=2 outlier above. This is the claim "flat from 1 to 64 readers" as something
# that can fail.
#
# 21.90 to 22.15, which is what the file HOLDS. The session summary of this run
# quotes it as "22.1-22.2", and that is a rounding of the two ends rather than
# the span: the medians dip to 21.901 at eight readers and 21.961 at twelve. The
# band is written from the data and not from the summary, because a band copied
# off a summary is a band nobody checked.
READ_CONTROL_FLAT_BAND = (21.85, 22.20)
READ_CONTROL_FLAT_EXCEPT = (2,)

# The control arm, from the LAST repeat -- which is what the program's own
# scaling verdict uses, and it is not the median. Both are kept because a run
# that changed which one the verdict reads would otherwise pass silently.
READ_CONTROL_FANCHAN_SCALE = (22.218, 127.542, 5.74)
READ_CONTROL_ONECHAN_SCALE = (22.214, 22.102, 0.99)

# Against `wh_dram#performance` (reads, static VC, 1 MiB per tile, ONE channel)
# at the vendor's own 1 / 12 / 48 tiles, converted at the 1000 MHz the device
# reported. Per cent deviation of measured from published.
VENDOR_GB_S = {1: 22.2, 12: 22.3, 48: 22.3}
READ_CONTROL_DEVIATION_PCT = {1: -0.2, 12: -1.5, 48: -0.9}

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


def check_read_control(path=None):
    """Is the read arm still the arm that produced the 2026-08-17 control?

    Yields ``(ok, line)``. Nothing here touches the write direction, and that is
    the whole point: this is the assertion that adding one did not move it.
    """
    path = Path(path) if path else READ_CONTROL_CSV
    if not path.exists():
        yield False, f"read control CSV not found: {path}"
        return
    rows, meta = read_rows(path)
    yield (
        meta.get("arch") == "wormhole",
        f"the control file is a wormhole run (arch={meta.get('arch')})",
    )

    one = by_reader_count(rows, "onechan")
    fan_last = by_reader_count(rows, "fanchan", how="last")
    one_last = by_reader_count(rows, "onechan", how="last")

    missing = sorted(set(READ_CONTROL_ONECHAN) - set(one))
    yield (
        not missing,
        f"every pinned reader count is present in the onechan arm (missing: {missing})",
    )
    for n, want in sorted(READ_CONTROL_ONECHAN.items()):
        if n not in one:
            continue
        yield (
            _close(one[n], want),
            f"onechan n={n:<2} median {one[n]:.3f} B/cycle vs pinned {want:.3f}",
        )

    lo, hi = READ_CONTROL_FLAT_BAND
    outside = sorted(
        n
        for n, v in one.items()
        if n not in READ_CONTROL_FLAT_EXCEPT and not lo <= v <= hi
    )
    yield (
        not outside,
        f"onechan is FLAT in {lo}-{hi} B/cycle at every reader count except "
        f"{READ_CONTROL_FLAT_EXCEPT} (outside: {outside})",
    )

    for arm, series, pinned in (
        ("fanchan", fan_last, READ_CONTROL_FANCHAN_SCALE),
        ("onechan", one_last, READ_CONTROL_ONECHAN_SCALE),
    ):
        if not series:
            yield False, f"{arm} has no measured points"
            continue
        n_lo, n_hi = min(series), max(series)
        want_lo, want_hi, want_scale = pinned
        got_scale = series[n_hi] / series[n_lo] if series[n_lo] else 0.0
        yield (
            _close(series[n_lo], want_lo)
            and _close(series[n_hi], want_hi)
            and _close(got_scale, want_scale, 0.02),
            f"{arm} {series[n_lo]:.3f} -> {series[n_hi]:.3f} B/cycle over "
            f"{n_lo} -> {n_hi} (x{got_scale:.2f}) vs pinned "
            f"{want_lo:.3f} -> {want_hi:.3f} (x{want_scale:.2f})",
        )

    clock = float(meta.get("clock_mhz") or 0)
    yield clock > 0, f"the control file carries a clock ({clock:.0f} MHz)"
    for n, want_pct in sorted(READ_CONTROL_DEVIATION_PCT.items()):
        if n not in one or clock <= 0:
            yield False, f"no onechan point at n={n} to compare to the vendor table"
            continue
        got_gb = one[n] * clock / 1000.0
        pct = 100.0 * (got_gb - VENDOR_GB_S[n]) / VENDOR_GB_S[n]
        # 0.15 percentage points: the published deviations are quoted to one
        # decimal, so anything tighter is checking the rounding.
        yield (
            abs(pct - want_pct) <= 0.15,
            f"n={n:<2} {got_gb:.2f} GB/s vs published {VENDOR_GB_S[n]}: "
            f"{pct:+.1f}% vs pinned {want_pct:+.1f}%",
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
    print(f"== {label}")
    failed = 0
    for ok, line in checks:
        if ok is True:
            print(f"   ok   {line}")
        elif ok is False:
            print(f"   FAIL {line}")
            failed += 1
        else:  # informational rows yield ok=True; nothing else is expected
            print(f"        {line}")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--read-control",
        nargs="?",
        const="",
        help="re-derive the read arm's pinned control (default: the committed "
        "2026-08-17 Wormhole card CSV)",
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
            "read arm: the 2026-08-17 Wormhole control, re-derived",
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
