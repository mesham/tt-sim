#!/usr/bin/env python3
"""Did the issue loop the flag asked for actually run, and what did it cost?

**This is the check the stateful arm turns on.** The variant swaps
``noc_async_read`` for ``noc_async_read_set_state`` plus
``noc_async_read_with_state`` and changes nothing else, so a run that silently
kept the stateless loop produces a perfectly well-formed CSV, passes every
existing gate in ``nocreadbench``'s own verdict, and yields a confident wrong
conclusion: it would say the shorter loop bought nothing, which is precisely
what the *rival* hypothesis predicts. Passing ``--stateful`` proves nothing on
its own -- a stale binary, a JIT cache that kept the old kernel, a shell that
dropped the argument -- so the claim is checked against the **returned
payload**.

Every row carries ``probe_word``, ``sig_src`` and ``sig_witness``. The kernel
pointed the read state at a witness tile and then issued one transaction through
the same API call its timed loop used: the stateless call rewrites
``NOC_TARG_ADDR_COORDINATE`` and the source answers, the stateful call never
writes it and the witness answers. So

    stateless  iff  probe_word == sig_src
    stateful   iff  probe_word == sig_witness

and a row whose ``mode`` column disagrees with its own probe is refused.

Deliberately standalone: it imports nothing but the standard library, so it runs
on a card box that has only ``perfbench/nocreadbench/`` rsynced onto it and no
tt-sim at all.

    ./check_mode.py --expect stateless nocreadbench-wormhole.csv
    ./check_mode.py stateless.csv stateful.csv      # the paired verdict

Exit status is 0 when every row proves its own mode, non-zero otherwise -- so it
can gate a run in a shell script.
"""

from __future__ import annotations

import argparse
import re
import sys

#: The 2026-08-17 Wormhole card session, ``perfbench/card-sessions/
#: 2026-08-17-wormhole-B/nocread.wormhole.csv``, differenced along the burst
#: axis. THE CONTROL: the stateless arm must reproduce these, because a variant
#: is worth nothing against a control that moved. Marginals, not averages: an
#: average carries the loop's constant term divided by N and so moves whenever
#: anything outside the loop does.
WORMHOLE_CONTROL = {(4, 16): 44.08, (16, 64): 44.00, (64, 128): 43.97}

#: How far a control marginal may drift and still count as reproducing. The
#: session's own three marginals agree to 0.25 %; 3 % is loose enough that a
#: recompiled kernel with the same loop passes and tight enough that a loop one
#: instruction different (1 cycle, 2.3 %) is visible as a WARN.
CONTROL_TOLERANCE = 0.03

#: The band the shipped dataset's Wormhole rows occupy, stateful and stateless
#: (17.33 and 25.00 cycles/transaction). Used only to say which pre-registered
#: prediction a stateful marginal landed on. It is the vendor's dataset from a
#: DIFFERENT part -- not a published figure for the part under test, which is
#: the whole reason this measurement exists.
DATASET_BAND = (15.0, 30.0)

HEADER_RE = re.compile(r"^#\s*nocreadbench\s+(.*)$")
CONFIG_RE = re.compile(r"^nocreadbench-config\s+(.*)$", re.MULTILINE)


def parse_kv(text):
    """``a=1 b=2`` as a dict. Both the CSV header and the config line use it."""
    fields = {}
    for token in text.split():
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def read_csv(path):
    """``(header_fields, rows)`` for one nocreadbench CSV.

    Hand-rolled rather than :mod:`csv`, because the file carries ``#`` comment
    lines above the header row and the stdlib reader would take the first of
    them as the field names.
    """
    header, names, rows = {}, None, []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                match = HEADER_RE.match(line)
                if match is not None and not header:
                    header = parse_kv(match.group(1))
                continue
            fields = line.split(",")
            if names is None:
                names = fields
                continue
            rows.append(dict(zip(names, fields)))
    return header, rows


def check_rows(rows, expect=None):
    """``(problems, notes)`` -- does every row prove its own mode from payload?"""
    problems, notes = [], []
    if not rows:
        return ["the file has no data rows at all"], notes
    modes = {row.get("mode", "?") for row in rows}
    if "probe_word" not in rows[0]:
        return (
            [
                "no probe_word column -- this CSV predates the mode witness (NRB3) and "
                "cannot say which issue loop produced it"
            ],
            notes,
        )
    if len(modes) != 1:
        problems.append(
            f"the rows disagree about the mode: {sorted(modes)}. One CSV is one arm."
        )
    mode = sorted(modes)[0]
    if "refused" in modes:
        problems.append(
            "at least one row is 'refused' -- the kernel would not run the stateful "
            "loop for a point with more than one source tile, and the host should "
            "have dropped it before launching"
        )
    if expect is not None and mode != expect:
        problems.append(
            f"the rows say mode={mode!r} but this check was asked for {expect!r} -- "
            "the run and the analysis disagree about what ran"
        )

    bad, fill, checked = [], 0, 0
    for row in rows:
        probe = row["probe_word"]
        want = row["sig_witness"] if row["mode"] == "stateful" else row["sig_src"]
        other = row["sig_src"] if row["mode"] == "stateful" else row["sig_witness"]
        if row["sig_src"] == row["sig_witness"]:
            bad.append(
                f"{row['experiment']}#{row['point']}: the witness IS the source "
                f"({row['sig_src']}), so this row's probe cannot discriminate"
            )
            continue
        checked += 1
        if probe == want:
            continue
        if int(probe, 16) == 0xEEEEEEEE:
            fill += 1
        bad.append(
            f"{row['experiment']}#{row['point']}: mode={row['mode']} wants probe "
            f"{want}, got {probe} (the other arm's tile is {other})"
        )
    if bad:
        problems.append(
            f"{len(bad)} of {len(rows)} row(s) did not prove their own mode; "
            "THE ARM DID NOT TAKE. First few:"
        )
        problems.extend("    " + line for line in bad[:5])
        if fill:
            problems.append(
                f"    {fill} of them read the pre-fill 0xEEEEEEEE, so no transaction "
                "landed at all -- that is a broken probe, not a wrong arm"
            )
    else:
        notes.append(
            f"all {checked} row(s) proved mode={mode} from the tile that answered "
            "their probe, not from the flag"
        )
    return problems, notes


def marginals(rows):
    """``{(n_lo, n_hi): cycles_per_transaction}`` off the ``burst`` axis.

    The marginal, not the average: differencing consecutive burst lengths
    removes the loop's constant term (its prologue, the closing barrier, the
    launch) and leaves the per-transaction cost, which is the only quantity the
    two arms can be compared on.
    """
    per_n = {}
    for row in rows:
        if row.get("experiment") != "burst":
            continue
        per_n.setdefault(int(row["num_tx"]), []).append(int(row["cycles"]))
    means = {n: sum(v) / len(v) for n, v in per_n.items()}
    out, order = {}, sorted(means)
    for lo, hi in zip(order, order[1:]):
        out[(lo, hi)] = (means[hi] - means[lo]) / (hi - lo)
    return out, means


def average_per_tx(rows):
    """Mean ``cycles_per_tx`` over the whole file, for the record only."""
    values = [float(row["cycles_per_tx"]) for row in rows if row.get("cycles_per_tx")]
    return sum(values) / len(values) if values else 0.0


def report_one(path, expect, quiet):
    """Check one CSV. Returns ``(ok, header, marginals, average)``."""
    header, rows = read_csv(path)
    problems, notes = check_rows(rows, expect)
    marg, means = marginals(rows)
    average = average_per_tx(rows)
    arch = header.get("arch", "unknown")
    mode = rows[0].get("mode", header.get("mode", "unknown")) if rows else "unknown"

    if problems:
        print(f"check_mode: {path}: THE ARM DID NOT TAKE", file=sys.stderr)
        for problem in problems:
            print(f"  FAIL {problem}", file=sys.stderr)
        return False, header, marg, average

    if not quiet:
        print(f"check_mode: {path}: arch={arch} mode={mode} rows ok")
        for note in notes:
            print(f"  ok   {note}")
        if means:
            print(
                "  n    burst cycles: "
                + ", ".join(f"N={n} {means[n]:.0f}" for n in sorted(means))
            )
        for (lo, hi), value in sorted(marg.items()):
            print(f"  marg N={lo}->{hi}: {value:.2f} cycles/transaction")
        print(f"  avg  {average:.2f} cycles/transaction over every row (informational)")

    ok = True
    # The control has to reproduce, and on Wormhole there is a recorded session
    # to reproduce. On any other part there is not, and saying so is the honest
    # answer -- an arm that ESTABLISHES a control is not an arm that lost one.
    #
    # A SIMULATOR run is checked against nothing. tt-sim's issue loop is a
    # different program from the card's (its own RV32 model, its own compiler
    # output, no NIU backpressure at all), so comparing its marginal to the
    # card's control band manufactures a "CONTROL MOVED" from two numbers that
    # were never the same measurement. The header records which it was.
    if header.get("sim") == "1":
        print(
            "  note sim=1: this is a tt-sim run and is NOT a measurement. No card "
            "control is applied to it; its own two arms are still comparable to "
            "each other."
        )
    elif mode == "stateless":
        if arch == "wormhole":
            for key, want in sorted(WORMHOLE_CONTROL.items()):
                got = marg.get(key)
                if got is None:
                    print(
                        f"  WARN the control has no N={key[0]}->{key[1]} interval to "
                        "compare; was this run filtered with --only?"
                    )
                    continue
                drift = abs(got - want) / want
                if drift <= CONTROL_TOLERANCE:
                    print(
                        f"  ctrl N={key[0]}->{key[1]}: {got:.2f} vs 2026-08-17's "
                        f"{want:.2f} ({drift * 100:.1f} %) -- the control reproduces"
                    )
                else:
                    print(
                        f"  CTRL MOVED N={key[0]}->{key[1]}: {got:.2f} vs "
                        f"2026-08-17's {want:.2f} ({drift * 100:.1f} %). The stateless "
                        "arm is not the program that session ran, so the two arms are "
                        "not a like-for-like pair. SAY SO and send the session anyway: "
                        "a control that moved is the most important result in it.",
                        file=sys.stderr,
                    )
                    ok = False
        else:
            print(
                f"  note no recorded card control for {arch}; this arm ESTABLISHES one "
                "rather than reproducing one"
            )
    return ok, header, marg, average


def paired_verdict(runs):
    """The pre-registered read, per architecture, given both arms.

    The two predictions were written down before any card ran the variant -- see
    README.md, "The stateful variant" -- and this function does nothing but say
    which one the numbers landed on. It deliberately cannot express "somewhere
    in between" as a pass.
    """
    by_arch, sim = {}, {}
    for header, mode, marg in runs:
        arch = header.get("arch", "unknown")
        by_arch.setdefault(arch, {})[mode] = marg
        sim[arch] = sim.get(arch, False) or header.get("sim") == "1"
    for arch, arms in sorted(by_arch.items()):
        print("")
        tag = "  -- SIMULATOR, NOT A MEASUREMENT" if sim.get(arch) else ""
        print(f"== the paired verdict, {arch}{tag}")
        if "stateless" not in arms or "stateful" not in arms:
            print(f"   only {sorted(arms)} present; the verdict needs both arms")
            continue
        keys = sorted(set(arms["stateless"]) & set(arms["stateful"]))
        if not keys:
            print("   the two arms share no burst interval; not comparable")
            continue
        control = sum(arms["stateless"][k] for k in keys) / len(keys)
        variant = sum(arms["stateful"][k] for k in keys) / len(keys)
        delta = control - variant
        print(
            f"   stateless {control:.2f}  stateful {variant:.2f}  the loop bought {delta:.2f}"
        )
        in_band = DATASET_BAND[0] <= variant <= DATASET_BAND[1]
        barely = delta <= 0.1 * control
        if in_band and not barely:
            print(
                "   PREDICTION 1 -- THE FLOOR IS OUR ISSUE LOOP. The shorter loop fell "
                f"by {delta:.2f} cycles and landed at {variant:.2f}, inside the "
                f"{DATASET_BAND[0]:.0f}-{DATASET_BAND[1]:.0f} band the shipped dataset "
                "occupies. The stateless figure is then a property of THIS PROGRAM's "
                "instruction stream and is not evidence of a per-read cost in the part."
            )
        elif barely:
            print(
                "   PREDICTION 2 -- THERE IS A FLOOR. Removing the per-transaction "
                f"stores bought {delta:.2f} cycles of {control:.2f}, so the rate is not "
                "set by the instruction stream. Something downstream of the issue loop "
                "is, and it is worth about "
                f"{variant:.0f} cycles per read on this part."
            )
        else:
            print(
                f"   NEITHER PREDICTION. The loop bought {delta:.2f} cycles and landed "
                f"at {variant:.2f}, outside the dataset's band and too large to call "
                "unmoved. Report the two numbers and do not pick a mechanism: this is "
                "the outcome both pre-registered predictions were written to be able "
                "to lose to."
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Check that a nocreadbench CSV really ran the issue loop it claims, from "
            "the payload its probe returned rather than from the flag, and report the "
            "marginal cycles per transaction. Given both arms, print the paired "
            "verdict against the predictions registered in README.md."
        )
    )
    parser.add_argument(
        "csv", nargs="+", help="nocreadbench-<arch>.csv from the run(s)"
    )
    parser.add_argument(
        "--expect",
        choices=("stateless", "stateful"),
        help="the arm the run claims to be; refuses a CSV whose rows say otherwise",
    )
    parser.add_argument(
        "--stdout",
        help=(
            "the run's log; its nocreadbench-config line is cross-checked against the "
            "CSV, so a log and a CSV from different runs cannot be paired"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="print only on failure")
    args = parser.parse_args(argv)

    status = 0
    runs = []
    for path in args.csv:
        ok, header, marg, _ = report_one(path, args.expect, args.quiet)
        if not ok:
            status = 1
        _, rows = read_csv(path)
        mode = rows[0]["mode"] if rows and "mode" in rows[0] else "unknown"
        runs.append((header, mode, marg))

    if args.stdout:
        with open(args.stdout) as handle:
            match = CONFIG_RE.search(handle.read())
        if match is None:
            print(
                f"check_mode: {args.stdout} has no 'nocreadbench-config' line -- the "
                "binary predates the mode arms and cannot have honoured --stateful",
                file=sys.stderr,
            )
            status = 1
        else:
            config = parse_kv(match.group(1))
            for header, mode, _ in runs:
                if config.get("mode") != mode:
                    print(
                        f"check_mode: the log says mode={config.get('mode')!r} and the "
                        f"CSV rows say {mode!r} -- they are not from the same run",
                        file=sys.stderr,
                    )
                    status = 1
                if config.get("arch") != header.get("arch"):
                    print(
                        f"check_mode: the log says arch={config.get('arch')!r} and the "
                        f"CSV says {header.get('arch')!r}",
                        file=sys.stderr,
                    )
                    status = 1

    if len(runs) > 1:
        paired_verdict(runs)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
