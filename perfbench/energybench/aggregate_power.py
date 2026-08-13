#!/usr/bin/env python3
"""Turn a card session's raw telemetry samples into the power CSV the analysis
reads. Standard library only -- this runs on the card box, which has no venv,
no numpy and no ``tt_sim/``.

Input is the ``slots.csv`` manifest ``run_card.sh`` writes (one row per arm run,
recording the window it occupied and where its samples went) plus one raw sample
file per slot. Output is ``power.csv``, whose schema
``tt_sim.perf.energy_rank.load_measured`` reads.

The one piece of judgement here is the **settle trim**: the first and last
``--settle`` seconds of every slot are discarded. A ~1 Hz sampler straddles the
edges of a window -- the first sample may have been taken while the previous arm
was still winding down, and the board's power rail and fan curve both have time
constants of seconds. Trimming both ends is what makes the remaining samples a
measurement of a *steady state* rather than of a transition. The count that
survives is reported per row, and the analysis gate ``samples`` refuses a row
that has too few.
"""

import argparse
import csv
import sys
from pathlib import Path

SAMPLE_FIELDS = ("t", "aiclk", "voltage", "current", "power", "temp")

OUT_COLUMNS = (
    "session",
    "cycle",
    "slot",
    "label",
    "arm",
    "inner",
    "launches",
    "wall_s",
    "launches_per_s",
    "power_w",
    "power_sd_w",
    "samples",
    "aiclk_mhz",
    "temp_c",
)


def read_samples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            try:
                rows.append({k: float(raw[k]) for k in SAMPLE_FIELDS})
            except (KeyError, TypeError, ValueError):
                # A snapshot taken while tt-smi was mid-write, or a field the
                # part does not publish. Dropping the row is right; guessing a
                # value is not.
                continue
    return rows


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def aggregate(slots: list[dict], root: Path, settle: float) -> list[dict]:
    out = []
    for slot in slots:
        start = float(slot["start_s"]) + settle
        end = float(slot["end_s"]) - settle
        samples = read_samples(root / slot["samples_file"])
        window = [s for s in samples if start <= s["t"] <= end]
        powers = [s["power"] for s in window]
        row = {
            "session": slot.get("session", ""),
            "cycle": slot["cycle"],
            "slot": slot["slot"],
            "label": slot["label"],
            "arm": slot.get("arm", ""),
            "inner": slot.get("inner", 0),
            "launches": slot.get("launches", 0),
            "wall_s": slot.get("wall_s", 0),
            "launches_per_s": slot.get("launches_per_s", 0),
            "power_w": f"{mean(powers):.4f}",
            "power_sd_w": f"{stdev(powers):.4f}",
            "samples": len(window),
            "aiclk_mhz": f"{mean([s['aiclk'] for s in window]):.1f}",
            "temp_c": f"{mean([s['temp'] for s in window]):.1f}",
        }
        out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slots", required=True, help="slots.csv manifest")
    ap.add_argument("--out", required=True, help="power.csv to write")
    ap.add_argument(
        "--settle",
        type=float,
        default=5.0,
        help="seconds trimmed from each end of every slot (default 5)",
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

    empty = [r["label"] for r in rows if r["samples"] == 0]
    print(f"aggregate_power: {len(rows)} slots -> {args.out}")
    if empty:
        print(
            "aggregate_power: WARNING, no samples survived the settle trim for: "
            + ", ".join(sorted(set(empty))),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
