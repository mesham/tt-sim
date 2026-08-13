#!/usr/bin/env python3
"""Continuous clock and thermal sampling from sysfs -- free, and unperturbing.

WHY SYSFS AS WELL AS tt-smi
---------------------------

The tenstorrent kernel driver publishes a handful of attributes as plain files,
and reading a file needs **no device handle at all**. Checked on a Blackhole
p150, ``/sys/class/tenstorrent/tenstorrent!<N>/`` carries::

    tt_aiclk  tt_arcclk  tt_axiclk  tt_therm_trip_count

plus ids and versions. ``tt-smi`` 6.2.0 can read a busy device (v4.0.0 moved its
default backend from Luwen to tt-umd, which is what fixed it), so power sampling
runs in slot again and this is not a substitute for it. It is a **second,
independent channel** that costs nothing, opens nothing, and cannot be perturbed
by the workload -- so a disagreement between the two is itself informative, and
the clock record survives even if a power sample is refused.

WHAT IS NOT THERE
-----------------

There is **no power, no voltage, no current and no temperature**, and no
``hwmon`` entry. The ``power/`` subdirectory is Linux's generic runtime-PM
directory (``runtime_active_time``, ``control``) and has nothing to do with
watts. Do not add a watts column here; there is nothing to put in it.

WHY IT MATTERS ANYWAY
---------------------

The 2026-08-13 card session read a baseline of 61.7 W at 1350 MHz in one cycle
and ~39 W at 800 MHz in the next -- a 42% swing in the quantity everything is
differenced against, driven by a clock the harness was not recording. (A
standalone idle read afterwards gave 70 W, so the spread is not two states
either.) ``tt_aiclk`` sampled through every slot turns that from an invisible
confound into a column, and the analysis refuses a session whose clock moved
under it. ``tt_therm_trip_count`` moving at all means the part throttled, which
invalidates the session outright.

Standard library only: this runs on the card box.
"""

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

DEFAULT_ROOT = "/sys/class/tenstorrent"

#: sysfs attribute -> CSV column. Values are written **as sysfs publishes them**;
#: the analysis gates on *relative* drift, which is unit-agnostic, so a part that
#: published kHz would still be gated correctly.
ATTRS = {
    "tt_aiclk": "aiclk",
    "tt_arcclk": "arcclk",
    "tt_axiclk": "axiclk",
    "tt_therm_trip_count": "therm_trip",
}

FIELDS = ("t", *ATTRS.values())


def device_dir(root: str | Path, chip: int) -> Path:
    """The sysfs directory for chip ``chip``.

    The kernel encodes the ``/`` in the device name as ``!``, so device 0 is
    ``tenstorrent!0``. Both spellings are accepted because a future driver could
    reasonably use either.
    """
    root = Path(root)
    for name in (f"tenstorrent!{chip}", f"tenstorrent/{chip}", str(chip)):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / f"tenstorrent!{chip}"


def read_once(path: Path) -> dict:
    """One reading. Missing attributes come back absent, not zero."""
    out = {}
    for attr, column in ATTRS.items():
        try:
            out[column] = (path / attr).read_text().strip()
        except OSError:
            continue
    return out


def available(path: Path) -> bool:
    return bool(read_once(path))


def watch(out: Path, path: Path, interval: float, duration: float | None) -> int:
    """Sample until killed, or for ``duration`` seconds.

    Written line by line and flushed, because the caller kills this process at
    the end of a slot and a buffered tail would be lost.
    """
    stop = {"now": False}

    def _stop(_sig, _frame):
        stop["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n = 0
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        while not stop["now"]:
            now = time.time()
            values = read_once(path)
            if values:
                writer.writerow({"t": f"{now:.6f}", **values})
                fh.flush()
                n += 1
            if duration is not None and now - started >= duration:
                break
            time.sleep(interval)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="required unless --check")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--chip", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument(
        "--check",
        action="store_true",
        help="report whether the attributes are readable and exit",
    )
    args = ap.parse_args(argv)

    path = device_dir(args.root, args.chip)
    if args.check:
        if available(path):
            print(f"sysfs_sample: {path} publishes {sorted(read_once(path))}")
            return 0
        print(
            f"sysfs_sample: {path} publishes none of {sorted(ATTRS)} -- this session "
            "will have no clock record, and a session that cannot show its clock "
            "held still cannot be differenced",
            file=sys.stderr,
        )
        return 1

    if not args.out:
        ap.error("--out is required unless --check")
    n = watch(Path(args.out), path, args.interval, args.duration)
    if n == 0:
        print(f"sysfs_sample: no readings from {path}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
