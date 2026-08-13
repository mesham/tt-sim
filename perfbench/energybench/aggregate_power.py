#!/usr/bin/env python3
"""Turn a card session's raw samples into the power CSV the analysis reads.

Standard library only -- this runs on the card box, which has no venv, no numpy
and no ``tt_sim/``.

WHAT THE OUTPUT IS
------------------

``power.csv``, whose ``power_w`` column is **steady-state repeated-kernel board
power under sustained load**, sampled *in slot* while the workload runs. That is
the quantity ``tt_sim.perf.energy_rank`` fits and the one the README describes.
It needs ``tt-smi`` >= 4.0.0 (the tt-umd backend); ``run_card.sh`` refuses to
start otherwise and the version used is recorded in ``session.log`` and in the
``tt_smi_version`` column.

If the session was run with ``--bracket``, the extra ``power_bracket_w`` /
``bracket_lag_s`` columns carry the **fallback** reading: a post-exit sample on a
decaying edge, a *different quantity*, never the analysis input. ``decay.txt``
says how much of the excursion survived to it.

INPUTS
------

The ``slots.csv`` manifest ``run_card.sh`` writes -- one row per slot, its window
on the wall clock, its launch count, its status, and where its sample streams
went:

``*.pre.csv``   telemetry taken just *before* the slot, device free: this slot's
                own local idle reference, seconds away from it rather than a
                whole cycle, which is what the 42% between-cycle baseline swing
                of 2026-08-13 demands.
``*.pow.csv``   telemetry sampled *throughout* the slot -- the measurement.
``*.clk.csv``   sysfs clock and thermal samples, also throughout: a second
                channel that opens no device and so cannot be refused.
``*.post.csv``  present only under ``--bracket``.

THE JUDGEMENTS HERE, AND ONLY THESE
-----------------------------------

1. **The settle trim.** The first and last ``--settle`` seconds of every slot
   are discarded from both in-slot streams. A ~1 Hz sampler straddles slot
   edges, and the power rail and fan curve have time constants of seconds;
   trimming both ends is what makes the remainder a measurement of a steady
   state rather than of a transition.

2. **A missing reading is written EMPTY, never zero.** The 2026-08-13 session
   recorded ``power_w=0, samples=0`` for every arm because a swallowed exception
   and a ``mean([]) -> 0.0`` turned "the tool refused to read the device" into
   "the board drew no power", and the session looked complete. Empty cells, an
   explicit ``status`` column, an ``attempts`` count beside ``samples``, and a
   non-zero exit make that unreadable as success.

3. **Attempts are counted, not just successes.** ``samples < attempts`` means
   the sampler was failing part of the time, which is a different and quieter
   illness than failing entirely.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_sample import read_samples  # noqa: E402

OUT_COLUMNS = (
    "session",
    "cycle",
    "slot",
    "label",
    "arm",
    "inner",
    "status",
    "launches",
    "wall_s",
    "launches_per_s",
    # -- in-slot board power under SUSTAINED LOAD: the measurement -----------
    "power_w",
    "power_sd_w",
    "samples",
    "attempts",
    "sample_failures",
    # -- the slot's own local idle reference, taken seconds before it --------
    "pre_idle_w",
    "pre_idle_samples",
    "delta_w",
    # -- confound control: sysfs, continuous, no device handle ---------------
    "aiclk_mhz",
    "temp_c",
    "sysfs_aiclk_mean",
    "sysfs_aiclk_min",
    "sysfs_aiclk_max",
    "sysfs_aiclk_drift_pct",
    "sysfs_arcclk_mean",
    "therm_trip_delta",
    "clock_samples",
    # -- the --bracket FALLBACK: a decaying edge, a different quantity -------
    "power_bracket_w",
    "bracket_lag_s",
    "bracket_samples",
    "tt_smi_version",
    "note",
)

STATUS_OK = "ok"


def mean(values):
    return sum(values) / len(values) if values else None


def stdev(values):
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def read_clock(path: Path, start: float, end: float) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            try:
                t = float(raw["t"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (start <= t <= end):
                continue
            row = {"t": t}
            for key in ("aiclk", "arcclk", "axiclk", "therm_trip"):
                try:
                    row[key] = float(raw[key])
                except (KeyError, TypeError, ValueError):
                    row[key] = None
            rows.append(row)
    rows.sort(key=lambda r: r["t"])
    return rows


def _fmt(value, digits=4):
    return "" if value is None else f"{value:.{digits}f}"


def _window(rows, start, end):
    return [r for r in rows if start <= r["t_mid"] <= end]


def aggregate(slots: list[dict], root: Path, settle: float) -> list[dict]:
    out = []
    for slot in slots:
        start = float(slot["start_s"]) + settle
        end = float(slot["end_s"]) - settle
        notes = []
        status = (slot.get("status") or STATUS_OK).strip() or STATUS_OK

        # -- in-slot power: the measurement --------------------------------
        pow_file = slot.get("pow_file") or ""
        every = (
            read_samples(root / pow_file, phase="run", only_ok=False)
            if pow_file
            else []
        )
        attempts = _window(every, start, end)
        window = [r for r in attempts if r["ok"] and r["power_w"] is not None]
        powers = [r["power_w"] for r in window]
        failures = len(attempts) - len(window)
        if not window:
            if status == STATUS_OK:
                status = "no_telemetry"
            errors = sorted({r["error"] for r in attempts if r.get("error")})
            notes.append(
                f"no in-slot power sample survived ({len(attempts)} attempts)"
                + (f": {errors[0]}" if errors else "")
            )
        elif failures:
            notes.append(f"{failures} of {len(attempts)} in-slot samples failed")

        # -- the slot's own local idle reference ---------------------------
        pre_file = slot.get("pre_file") or ""
        pre = read_samples(root / pre_file, phase="pre") if pre_file else []
        pre_w = mean([r["power_w"] for r in pre if r["power_w"] is not None])
        if pre_w is None and status == STATUS_OK:
            status = "no_pre_idle"
            notes.append("no pre-slot idle reference, so this row cannot be paired")

        # -- confound control ----------------------------------------------
        clk_file = slot.get("clk_file") or ""
        clock = read_clock(root / clk_file, start, end) if clk_file else []
        aiclks = [r["aiclk"] for r in clock if r["aiclk"] is not None]
        arcclks = [r["arcclk"] for r in clock if r["arcclk"] is not None]
        trips = [r["therm_trip"] for r in clock if r["therm_trip"] is not None]
        aiclk_mean = mean(aiclks)
        drift_pct = None
        if aiclks and aiclk_mean:
            drift_pct = 100.0 * (max(aiclks) - min(aiclks)) / aiclk_mean
        trip_delta = (trips[-1] - trips[0]) if trips else None
        if not clock:
            if status == STATUS_OK:
                status = "no_clock"
            notes.append(
                "no sysfs clock samples in the trimmed window; a slot with no clock "
                "record cannot show the clock held still"
            )

        # -- the --bracket fallback, if this session used it ----------------
        post_file = slot.get("post_file") or ""
        post = read_samples(root / post_file, phase="post") if post_file else []
        post = [r for r in post if r["power_w"] is not None]

        out.append(
            {
                "session": slot.get("session", ""),
                "cycle": slot["cycle"],
                "slot": slot["slot"],
                "label": slot["label"],
                "arm": slot.get("arm", ""),
                "inner": slot.get("inner", 0),
                "status": status,
                "launches": slot.get("launches", 0),
                "wall_s": slot.get("wall_s", 0),
                "launches_per_s": slot.get("launches_per_s", 0),
                "power_w": _fmt(mean(powers)),
                "power_sd_w": _fmt(stdev(powers)),
                "samples": len(window),
                "attempts": len(attempts),
                "sample_failures": failures,
                "pre_idle_w": _fmt(pre_w),
                "pre_idle_samples": len(pre),
                "delta_w": _fmt(
                    mean(powers) - pre_w if powers and pre_w is not None else None
                ),
                "aiclk_mhz": _fmt(
                    mean(
                        [r["aiclk_mhz"] for r in window if r["aiclk_mhz"] is not None]
                    ),
                    1,
                ),
                "temp_c": _fmt(
                    mean([r["temp_c"] for r in window if r["temp_c"] is not None]), 1
                ),
                "sysfs_aiclk_mean": _fmt(aiclk_mean, 1),
                "sysfs_aiclk_min": _fmt(min(aiclks) if aiclks else None, 1),
                "sysfs_aiclk_max": _fmt(max(aiclks) if aiclks else None, 1),
                "sysfs_aiclk_drift_pct": _fmt(drift_pct, 3),
                "sysfs_arcclk_mean": _fmt(mean(arcclks), 1),
                "therm_trip_delta": "" if trip_delta is None else f"{trip_delta:.0f}",
                "clock_samples": len(clock),
                "power_bracket_w": _fmt(post[0]["power_w"] if post else None),
                "bracket_lag_s": _fmt(post[0]["dt_s"] if post else None, 3),
                "bracket_samples": len(post),
                "tt_smi_version": slot.get("tt_smi_version", ""),
                "note": "; ".join(notes),
            }
        )
    return out


def missing_slots(rows: list[dict]) -> list[str]:
    """Every cycle must carry every label.

    Cycle 2 of the 2026-08-13 session was missing ``idle-0`` entirely: the CSV
    jumped slot 0 to slot 2. The cause was structural -- ``run_card.sh`` wrote a
    manifest row only on success, and this program emits one row per manifest
    row, so a slot that failed left **no trace at all** in either machine-readable
    output and only a line in ``session.log``. ``run_card.sh`` now records
    failures as rows; this is the second net, for a runner that died mid-cycle.
    """
    by_cycle: dict[str, set] = {}
    for row in rows:
        by_cycle.setdefault(str(row["cycle"]), set()).add(row["label"])
    every = set().union(*by_cycle.values()) if by_cycle else set()
    holes = []
    for cycle in sorted(by_cycle, key=lambda c: (len(c), c)):
        for label in sorted(every - by_cycle[cycle]):
            holes.append(f"cycle {cycle} is missing {label}")
    return holes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slots", required=True, help="slots.csv manifest")
    ap.add_argument("--out", required=True, help="power.csv to write")
    ap.add_argument(
        "--settle",
        type=float,
        default=5.0,
        help="seconds trimmed from each end of every slot's in-slot streams "
        "(default 5)",
    )
    args = ap.parse_args(argv)

    slots_path = Path(args.slots)
    with open(slots_path, newline="") as fh:
        slots = list(csv.DictReader(fh))
    if not slots:
        print(f"aggregate_power: {slots_path} has no rows", file=sys.stderr)
        return 2

    rows = aggregate(slots, slots_path.parent, args.settle)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(f"aggregate_power: {len(rows)} slots -> {args.out}")
    print(
        "aggregate_power: power_w is IN-SLOT board power under sustained load. "
        "power_bracket_w, if present, is the --bracket fallback: a post-exit "
        "sample on a decaying edge, a different quantity, never the fit input."
    )

    bad = [r for r in rows if r["status"] != STATUS_OK]
    holes = missing_slots(rows)
    if bad or holes:
        print("", file=sys.stderr)
        print(
            "aggregate_power: REFUSED -- this session is not a clean measurement",
            file=sys.stderr,
        )
        for row in bad:
            print(
                f"  cycle {row['cycle']} slot {row['slot']} {row['label']}: "
                f"status={row['status']}  {row['note']}",
                file=sys.stderr,
            )
        for hole in holes:
            print(f"  {hole}", file=sys.stderr)
        print(
            "  A row with no telemetry has an EMPTY power cell, never zero, and "
            "tt_sim.perf.energy_rank refuses the session rather than averaging it in.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
