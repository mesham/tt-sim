#!/usr/bin/env python3
"""Board-power telemetry sampling that CANNOT fail silently.

WHY THIS EXISTS
---------------

The 2026-08-13 card session produced ``samples=0, power_w=0`` for **every arm
slot** of a three-cycle run, while the launch counts were perfect and
reproducible to ~1%. Only the baselines -- which launch nothing -- got telemetry.
The session looked like a completed measurement and was a total loss.

Two things caused that, and only one of them has since been fixed upstream.

**The version.** The box was running ``tt-smi`` 3.0.32, whose backend is Luwen.
With a tt-metal program holding the device, ``pyluwen::PciChip::new`` panicked --

    panicked at crates/ttkmd-if/src/lib.rs:294:17:
    Failed to map bar0_uc for 0 with error Invalid argument (os error 22)

-- because BAR0's mapping is exclusive on that path. ``tt-smi`` **v4.0.0
switched its default backend from Luwen to tt-umd**, and on 6.2.0 a busy-device
read returns normally: measured at 69.0 W with ``--arm rv`` holding the device,
and the workload was unperturbed (404.3 launches/s while sampled against 407.5
and 418.8 unsampled). So **continuous in-slot sampling works**, and it is the
right measurement: sustained load, not a decaying edge. :func:`check_version`
refuses to start on anything older, because an old tool does not degrade, it
records a session of zeros.

**The swallowed exception.** The old sampler piped ``tt-smi`` into a parser
wrapped in ``except Exception: pass`` and appended nothing on failure. A chip
that refused to open was therefore indistinguishable from a chip reporting
nothing, and ``mean([]) -> 0.0`` downstream turned it into "the board drew no
power". That is the bug this module exists to make impossible: **every attempt is
written as a row**, and a failed attempt carries ``ok=0`` and the tool's own
error text. A slot with no readings is loud at every stage after it.

MODES
-----

``watch``    continuous sampling until killed -- the primary path, one row per
             attempt, successes and failures alike.
``probe``    a fixed number of snapshots. Used for the pre-slot idle reference
             (device free, so it is a clean local baseline) and for the optional
             post-exit bracket.
``version``  the guard. Reads ``host_sw_vers.tt_smi`` out of the telemetry JSON,
             falling back to ``tt-smi --version``, and refuses a Luwen-era tool.
``fit``      fits ``P(t) = P_inf + A*exp(-t/tau)`` to post-exit samples. This is
             only needed by the ``--bracket`` fallback, and it also measures the
             board's thermal time constant, which is what the settle trim is
             sized against.

Standard library only: this runs on the card box, which has no venv, no numpy
and no ``tt_sim/``.
"""

import argparse
import csv
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

#: One row per **attempt**. ``ok`` is the load-bearing column: a refused or
#: unparseable snapshot is kept, with its error, rather than dropped.
SAMPLE_FIELDS = (
    "phase",
    "i",
    "t_start",
    "t_end",
    "t_mid",
    "dt_s",
    "ok",
    "power_w",
    "voltage_v",
    "current_a",
    "aiclk_mhz",
    "temp_c",
    "error",
)

#: ``tt-smi`` v4.0.0 switched its default backend from Luwen to tt-umd. Anything
#: older cannot read a busy device: it panics in ``pyluwen``, and the harness
#: records a session of zeros rather than a measurement.
MIN_TT_SMI_VERSION = (4, 0, 0)

#: Below this a fitted post-exit excursion is indistinguishable from the board's
#: own wander -- used only by the ``--bracket`` fallback.
MIN_EXCURSION_W = 1.0

#: How much of the loaded excursion a bracketed sample must still hold for the
#: fallback to mean anything. ``retained = exp(-lag/tau)``.
RETAINED_OK = 0.70
RETAINED_MARGINAL = 0.30


# ---------------------------------------------------------------------------
# Running the tool
# ---------------------------------------------------------------------------


def run_tool(command: str, timeout: float) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            shlex.split(command), capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        return 127, "", f"telemetry command not found: {exc}"
    except subprocess.TimeoutExpired:
        return 124, "", f"telemetry command timed out after {timeout:g}s"
    return proc.returncode, proc.stdout, proc.stderr


def parse_telemetry(stdout: str, chip: int) -> dict:
    """Pull the telemetry block out of ``tt-smi -s`` JSON.

    Raises rather than returning a default. A parse failure is a **failed
    sample**, not a zero-watt board -- that distinction is the whole point of
    this module. Values are padded strings (``" 70.0"``) on every version seen,
    so the ``strip()`` is load-bearing.
    """
    doc = json.loads(stdout)
    t = doc["device_info"][chip]["telemetry"]
    return {
        "power_w": float(str(t["power"]).strip()),
        "voltage_v": float(str(t["voltage"]).strip()),
        "current_a": float(str(t["current"]).strip()),
        "aiclk_mhz": float(str(t["aiclk"]).strip()),
        "temp_c": float(str(t["asic_temperature"]).strip()),
    }


def take_sample(command: str, chip: int, timeout: float) -> tuple[dict, str]:
    """One snapshot. Returns ``(values, error)``; ``values`` is empty on failure.

    There is deliberately no default reading. Inventing one is how a whole
    session of zeros passed for a measurement.
    """
    rc, stdout, stderr = run_tool(command, timeout)
    if rc != 0:
        # Several lines, not one. A Rust panic puts "thread '<unnamed>' panicked
        # at <file>:<line>" first and the message that names the cause -- "Failed
        # to map bar0_uc" -- second, so keeping only the first line throws away
        # the diagnosis.
        lines = [
            ln.strip() for ln in (stderr or stdout or "").splitlines() if ln.strip()
        ]
        return {}, f"rc={rc}: {' | '.join(lines[:3]) if lines else 'no output'}"[:400]
    try:
        return parse_telemetry(stdout, chip), ""
    except Exception as exc:  # noqa: BLE001 - any parse failure is a failed sample
        return {}, f"unparseable telemetry: {type(exc).__name__}: {exc}"[:400]


# ---------------------------------------------------------------------------
# The version guard
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(text: str) -> tuple[int, int, int] | None:
    m = _VERSION_RE.search(text or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def detect_version(
    command: str, timeout: float
) -> tuple[tuple[int, int, int] | None, str, str]:
    """``(version, source, detail)``.

    The telemetry JSON is asked first: it is the same invocation the session
    will use, so it proves the tool works as well as reporting what it is. The
    ``--version`` flag is the fallback for a build that omits ``host_sw_vers``.
    """
    rc, stdout, stderr = run_tool(command, timeout)
    if rc == 0 and stdout:
        try:
            doc = json.loads(stdout)
            raw = str(doc.get("host_sw_vers", {}).get("tt_smi", "")).strip()
            version = parse_version(raw)
            if version:
                return version, "host_sw_vers.tt_smi", raw
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    base = shlex.split(command)[0]
    rc2, out2, err2 = run_tool(f"{base} --version", timeout)
    version = parse_version(out2) or parse_version(err2)
    if version:
        return version, f"{base} --version", (out2 or err2).strip().splitlines()[0]
    detail = (stderr or err2 or stdout or "").strip().splitlines()
    return (
        None,
        "unknown",
        (detail[0] if detail else f"rc={rc}/{rc2}, no version in output"),
    )


def check_version(command: str, timeout: float = 30.0) -> tuple[bool, str, str]:
    """``(ok, version_string, message)``.

    A tool older than :data:`MIN_TT_SMI_VERSION` does not fail loudly on its own
    -- it panics per sample and the harness banks a session of zeros. So this is
    a refusal to start, not a warning.
    """
    version, source, detail = detect_version(command, timeout)
    if version is None:
        return (
            False,
            "unknown",
            f"cannot determine the tt-smi version ({source}: {detail}). The harness "
            "will not start against an unidentifiable telemetry tool: versions "
            "before "
            + ".".join(map(str, MIN_TT_SMI_VERSION))
            + " use the Luwen backend, which panics on a busy device and records a "
            "whole session of zeros.",
        )
    text = ".".join(map(str, version))
    if version < MIN_TT_SMI_VERSION:
        return (
            False,
            text,
            f"tt-smi {text} (via {source}) predates "
            + ".".join(map(str, MIN_TT_SMI_VERSION))
            + ", which is where the default backend changed from Luwen to tt-umd. "
            "The Luwen path cannot open a device held by a tt-metal program -- it "
            "panics in pyluwen with 'Failed to map bar0_uc' -- so every in-slot "
            "sample would fail and the session would be zeros. Upgrade tt-smi.",
        )
    return (
        True,
        text,
        f"tt-smi {text} (via {source}): tt-umd backend, busy reads supported",
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _writer(fh):
    return csv.DictWriter(fh, fieldnames=list(SAMPLE_FIELDS))


def _row(
    phase: str,
    i: int,
    t_start: float,
    t_end: float,
    t0: float,
    values: dict,
    error: str,
) -> dict:
    t_mid = 0.5 * (t_start + t_end)
    row = {
        "phase": phase,
        "i": i,
        "t_start": f"{t_start:.6f}",
        "t_end": f"{t_end:.6f}",
        "t_mid": f"{t_mid:.6f}",
        "dt_s": f"{t_mid - t0:.6f}",
        "ok": 1 if values else 0,
        "error": error,
    }
    for key in ("power_w", "voltage_v", "current_a", "aiclk_mhz", "temp_c"):
        row[key] = f"{values[key]:.4f}" if values else ""
    return row


def probe(
    out: Path,
    command: str,
    count: int,
    chip: int,
    phase: str,
    t0: float,
    timeout: float,
    gap: float,
) -> tuple[int, int]:
    """``count`` snapshots back to back. Returns ``(ok, attempts)``.

    Nothing is retried and nothing is dropped: the caller wants to know how many
    attempts there were as well as how many worked.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not out.exists() or out.stat().st_size == 0
    ok = 0
    with open(out, "a", newline="") as fh:
        writer = _writer(fh)
        if fresh:
            writer.writeheader()
        for i in range(count):
            if i and gap > 0:
                time.sleep(gap)
            t_start = time.time()
            values, error = take_sample(command, chip, timeout)
            writer.writerow(_row(phase, i, t_start, time.time(), t0, values, error))
            fh.flush()
            ok += bool(values)
    return ok, count


def watch(
    out: Path,
    command: str,
    chip: int,
    phase: str,
    interval: float,
    timeout: float,
    duration: float | None,
) -> tuple[int, int]:
    """Sample continuously until killed. Returns ``(ok, attempts)``.

    This is the primary path. Every row is flushed as it is written, because the
    caller kills this process at the end of a slot and a buffered tail would be
    lost -- which would look exactly like a sampler that failed.
    """
    stop = {"now": False}

    def _stop(_sig, _frame):
        stop["now"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = attempts = 0
    with open(out, "w", newline="") as fh:
        writer = _writer(fh)
        writer.writeheader()
        fh.flush()
        while not stop["now"]:
            t_start = time.time()
            values, error = take_sample(command, chip, timeout)
            writer.writerow(
                _row(phase, attempts, t_start, time.time(), t0, values, error)
            )
            fh.flush()
            attempts += 1
            ok += bool(values)
            if duration is not None and time.time() - t0 >= duration:
                break
            if interval > 0 and not stop["now"]:
                time.sleep(interval)
    return ok, attempts


def read_samples(
    path: Path, phase: str | None = None, only_ok: bool = True
) -> list[dict]:
    """Read a sample CSV. ``only_ok=False`` keeps the failed attempts too."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            if phase is not None and raw.get("phase") != phase:
                continue
            ok = str(raw.get("ok", "0")).strip() in ("1", "true", "True")
            if only_ok and not ok:
                continue
            try:
                row = {
                    "phase": raw.get("phase", ""),
                    "ok": ok,
                    "dt_s": float(raw["dt_s"]),
                    "t_mid": float(raw["t_mid"]),
                    "error": raw.get("error", ""),
                }
                for key in ("power_w", "voltage_v", "current_a", "aiclk_mhz", "temp_c"):
                    row[key] = (
                        float(raw[key]) if raw.get(key) not in (None, "") else None
                    )
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["t_mid"])
    return rows


# ---------------------------------------------------------------------------
# The decay fit -- only the `--bracket` fallback needs this
# ---------------------------------------------------------------------------


def _lstsq2(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least squares ``y = a + b*x``. Returns ``(a, b, sse)``."""
    n = len(pairs)
    sx = sum(x for x, _ in pairs)
    sy = sum(y for _, y in pairs)
    sxx = sum(x * x for x, _ in pairs)
    sxy = sum(x * y for x, y in pairs)
    denom = n * sxx - sx * sx
    if denom == 0:
        a = sy / n
        return a, 0.0, sum((y - a) ** 2 for _, y in pairs)
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b, sum((y - (a + b * x)) ** 2 for x, y in pairs)


def fit_decay(points: list[tuple[float, float]]) -> dict:
    """Fit ``P(t) = P_inf + A*exp(-t/tau)`` to post-exit power.

    ``tau`` is searched on a log grid with ``(P_inf, A)`` solved linearly at each
    trial: robust, short, and scipy is not available on a card box. Two points
    give a slope but no time constant, and that is reported as such rather than
    fitted.
    """
    points = sorted(points)
    n = len(points)
    out = {
        "n": n,
        "t_first": points[0][0] if points else None,
        "t_last": points[-1][0] if points else None,
        "p_first": points[0][1] if points else None,
        "p_last": points[-1][1] if points else None,
        "tau_s": None,
        "p_inf_w": None,
        "amplitude_w": None,
        "initial_slope_w_per_s": None,
        "rms_residual_w": None,
    }
    if n < 2:
        return out
    span = points[-1][0] - points[0][0]
    if span <= 0:
        return out
    if n == 2:
        out["initial_slope_w_per_s"] = (points[0][1] - points[-1][1]) / span
        return out

    best = None
    lo, hi = 0.05, max(20.0 * span, 10.0)
    steps = 400
    for k in range(steps + 1):
        tau = lo * (hi / lo) ** (k / steps)
        p_inf, amp, sse = _lstsq2([(math.exp(-t / tau), p) for t, p in points])
        if best is None or sse < best[0]:
            best = (sse, tau, p_inf, amp)
    sse, tau, p_inf, amp = best
    out.update(
        tau_s=tau,
        p_inf_w=p_inf,
        amplitude_w=amp,
        initial_slope_w_per_s=amp / tau,
        rms_residual_w=math.sqrt(sse / n),
    )
    return out


def bracket_verdict(fit: dict, lag_s: float | None) -> dict:
    """Judge the ``--bracket`` fallback from a decay fit.

    ``retained = exp(-lag/tau)`` is the fraction of the loaded excursion that
    survives to the first post-exit sample. The fallback measures a different
    quantity from the primary path whatever this comes out at; a small value
    means it measures nothing useful at all.
    """
    out = {
        "lag_s": lag_s,
        "retained_fraction": None,
        "lost_w": None,
        "verdict": "UNKNOWN",
        "reason": "",
    }
    tau, amp = fit.get("tau_s"), fit.get("amplitude_w")
    if fit["n"] < 3:
        out["reason"] = (
            f"a decay probe of {fit['n']} points cannot resolve a time constant, so "
            "the bracketing error is unquantified"
        )
        return out
    if tau is None or amp is None:
        out["reason"] = "the decay probe did not fit"
        return out
    if amp <= MIN_EXCURSION_W:
        out.update(
            verdict="CONDEMNED",
            retained_fraction=0.0,
            lost_w=0.0,
            reason=(
                f"fitted excursion {amp:.3f} W is at or under {MIN_EXCURSION_W:g} W: "
                "the board shows no measurable step when the workload stops, so a "
                "post-exit sample carries no workload signal"
            ),
        )
        return out
    if lag_s is None:
        out["reason"] = "no post-exit sample lag was recorded"
        return out
    retained = math.exp(-max(lag_s, 0.0) / tau)
    out["retained_fraction"] = retained
    out["lost_w"] = amp * (1.0 - retained)
    common = (
        f"tau = {tau:.2f} s against a {lag_s:.2f} s sampling lag: "
        f"{100 * retained:.0f}% of the {amp:.2f} W excursion survives to the first "
        "post-exit sample"
    )
    if retained >= RETAINED_OK:
        out.update(verdict="OK", reason=common)
    elif retained >= RETAINED_MARGINAL:
        out.update(
            verdict="MARGINAL",
            reason=common
            + " -- bracketed deltas are compressed and the factor must be quoted",
        )
    else:
        out.update(
            verdict="CONDEMNED",
            reason=common
            + ". Board power falls faster than this instrument can be read, so "
            "bracketing cannot stand in for in-slot sampling on this part",
        )
    return out


def render_fit(fit: dict, verdict: dict) -> str:
    lines = ["=== decay probe: how fast board power falls after the workload stops ==="]
    lines.append(f"  samples             : {fit['n']}")
    if fit["n"]:
        lines.append(
            f"  first / last        : {fit['p_first']:.2f} W at t+{fit['t_first']:.2f} s"
            f"  ->  {fit['p_last']:.2f} W at t+{fit['t_last']:.2f} s"
        )
    if fit.get("tau_s") is not None:
        lines.append(f"  fitted tau          : {fit['tau_s']:.3f} s")
        lines.append(f"  fitted idle floor   : {fit['p_inf_w']:.2f} W")
        lines.append(f"  fitted excursion    : {fit['amplitude_w']:.2f} W at t=0")
        lines.append(f"  initial slope       : {fit['initial_slope_w_per_s']:.3f} W/s")
        lines.append(f"  RMS residual        : {fit['rms_residual_w']:.3f} W")
    elif fit.get("initial_slope_w_per_s") is not None:
        lines.append(
            f"  slope (2 points)    : {fit['initial_slope_w_per_s']:.3f} W/s "
            "-- two points cannot give a time constant"
        )
    if verdict.get("retained_fraction") is not None:
        lines.append(
            f"  retained at bracket : {100 * verdict['retained_fraction']:.1f}% "
            f"(lag {verdict['lag_s']:.2f} s, {verdict['lost_w']:.2f} W already gone)"
        )
    lines.append(f"  VERDICT             : BRACKETING {verdict['verdict']}")
    lines.append(f"  {verdict['reason']}")
    lines.append("")
    lines.append(
        "  This judges the --bracket FALLBACK only. The primary path samples power "
        "IN SLOT, under sustained load, which needs no such correction."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    default_cmd = os.environ.get("ENERGYBENCH_TELEMETRY_CMD", "tt-smi -s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    for name, helptext in (
        ("watch", "sample continuously until killed (the primary path)"),
        ("probe", "take a fixed number of snapshots"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--out", required=True)
        p.add_argument("--command", default=default_cmd)
        p.add_argument("--chip", type=int, default=0)
        p.add_argument("--timeout", type=float, default=30.0)
        p.add_argument("--phase", default="run")
        if name == "watch":
            p.add_argument("--interval", type=float, default=0.5)
            p.add_argument("--duration", type=float, default=None)
        else:
            p.add_argument("--count", type=int, default=3)
            p.add_argument("--gap", type=float, default=0.0)
            p.add_argument("--t0", type=float, default=None)

    v = sub.add_parser("version", help="the tt-smi version guard")
    v.add_argument("--command", default=default_cmd)
    v.add_argument("--timeout", type=float, default=30.0)

    f = sub.add_parser("fit", help="fit post-exit decay (the --bracket fallback)")
    f.add_argument("--samples", required=True)
    f.add_argument("--phase", default="post")
    f.add_argument("--lag", type=float, default=None)
    f.add_argument("--json")
    f.add_argument("--report")

    args = ap.parse_args(argv)

    if args.mode == "version":
        ok, text, message = check_version(args.command, args.timeout)
        print(f"tt_smi_version={text}")
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 4

    if args.mode in ("watch", "probe"):
        out = Path(args.out)
        if args.mode == "watch":
            ok, attempts = watch(
                out,
                args.command,
                args.chip,
                args.phase,
                args.interval,
                args.timeout,
                args.duration,
            )
        else:
            t0 = args.t0 if args.t0 is not None else time.time()
            ok, attempts = probe(
                out,
                args.command,
                args.count,
                args.chip,
                args.phase,
                t0,
                args.timeout,
                args.gap,
            )
        print(f"telemetry_sample: {ok}/{attempts} {args.phase} samples -> {out}")
        if ok == 0:
            print(
                f"telemetry_sample: NO TELEMETRY -- all {attempts} attempts failed. "
                "The reasons are in the `error` column. A device that refuses to be "
                "read is a FAILED SAMPLE, never a zero-watt board.",
                file=sys.stderr,
            )
            return 3
        return 0

    rows = read_samples(Path(args.samples), phase=args.phase)
    points = [(r["dt_s"], r["power_w"]) for r in rows if r["power_w"] is not None]
    fit = fit_decay(points)
    lag = args.lag if args.lag is not None else (points[0][0] if points else None)
    verdict = bracket_verdict(fit, lag)
    text = render_fit(fit, verdict)
    print(text, end="")
    if args.report:
        Path(args.report).write_text(text)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"fit": fit, "verdict": verdict}, indent=2)
        )
    return 0 if verdict["verdict"] in ("OK", "MARGINAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
