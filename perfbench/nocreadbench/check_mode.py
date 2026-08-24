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

#: tt-sim's ABSOLUTE prediction for the ``--dram`` arm, registered here **before
#: any card ran it**, keyed ``(arch, source, tx_bytes) -> marginal cycles per
#: transaction``. Produced by running *this program* against tt-sim with the
#: cost model on (``TT_SIM_COST_MODEL=1``) and differencing the same burst axis
#: the card run is differenced along, so the two sides are the same arithmetic
#: on the same experiment and not a model quantity compared to a measurement.
#:
#: The structure is the prediction, not just the numbers. Two terms compete and
#: the larger wins:
#:
#: * the **issue loop** — 44 cycles/transaction on Wormhole, 47 on Blackhole,
#:   independent of payload size. Both are already validated against silicon
#:   (Wormhole 44.00 predicted vs 45.03 measured; Blackhole 47.00 predicted and
#:   measured exactly), so they are the part of this table least at risk.
#: * the **transfer** — payload over the sustained rate. tt-sim charges an L1
#:   read the NoC link (32 B/cycle Wormhole, 64 Blackhole) and a DRAM read the
#:   link plus the DRAM channel's excess, and **nothing else**: no endpoint
#:   queueing, no outstanding-transaction credit, no response reordering.
#:
#: So the model says DRAM and L1 are *identical* below the crossover and differ
#: by pure bandwidth above it. Wormhole crosses at 2048 B, Blackhole at 4096 B
#: — that difference is itself a registered claim.
SIM_DRAM_PREDICTION = {
    ("wormhole", "dram", 512): 44.00,
    ("wormhole", "dram", 1024): 44.00,
    ("wormhole", "dram", 2048): 86.00,
    ("wormhole", "dram", 4096): 171.00,
    ("wormhole", "l1", 512): 44.00,
    ("wormhole", "l1", 1024): 44.00,
    ("wormhole", "l1", 2048): 64.00,
    ("wormhole", "l1", 4096): 128.00,
    ("blackhole", "dram", 512): 47.00,
    ("blackhole", "dram", 1024): 47.00,
    ("blackhole", "dram", 2048): 47.00,
    ("blackhole", "dram", 4096): 87.00,
    ("blackhole", "l1", 512): 47.00,
    ("blackhole", "l1", 1024): 47.00,
    ("blackhole", "l1", 2048): 47.00,
    ("blackhole", "l1", 4096): 64.00,
}

#: The smallest payload at which the TRANSFER, rather than the issue loop, sets
#: each architecture's DRAM marginal, per the table above. STRICTLY BELOW this
#: size tt-sim predicts the DRAM and L1 arms are the SAME NUMBER, which is the
#: sharpest falsifier in the arm: a bandwidth term cannot explain a gap there.
DRAM_CROSSOVER = {"wormhole": 2048, "blackhole": 4096}

#: How far a DRAM marginal may sit from its registered figure and still count as
#: reproducing it. Looser than the control's 3 % because this is an absolute
#: cross-architecture prediction rather than a re-run of a recorded session, and
#: because the Wormhole issue loop's own residual against silicon was 2.3 %.
DRAM_TOLERANCE = 0.10

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
    problems.extend(check_source_arm(rows, notes))
    return problems, notes


def check_source_arm(rows, notes):
    """The SOURCE arm, re-derived from the CSV alone.

    The mode witness above says which issue loop ran. This says which kind of
    tile the timed burst actually read, and it is a separate question with the
    same failure mode: ``--dram`` on a stale binary, on a JIT cache that kept the
    old kernel, or in a shell that dropped the argument produces a well-formed
    file whose rates are an L1 read wearing a DRAM label -- and *that* reading is
    what "the card agrees with tt-sim" looks like, so it would be believed.

    ``landed`` is the first word the TIMED burst put at its landing address. The
    host stamps worker source regions ``0x5A5A....`` and DRAM tiles
    ``0xD4A5....``, so the word names the endpoint's kind as well as its
    coordinate, and the two halves of that space never overlap.
    """
    problems = []
    if not rows or "landed" not in rows[0]:
        if any(row.get("src_kind") == "dram" for row in rows):
            problems.append(
                "rows claim src_kind=dram but the file has no `landed` column, so no "
                "row can prove it read DRAM. This CSV predates the source witness"
            )
        return problems
    bad, checked, dram = [], 0, 0
    for row in rows:
        kind = row.get("src_kind", "?")
        sig = row.get("sig_src", "")
        # The signature half IS the arm. A `dram` row whose source is stamped
        # with a worker's signature was never aimed at a DRAM tile at all,
        # whatever landed.
        want_half = "0xd4a5" if kind == "dram" else "0x5a5a"
        if not sig.lower().startswith(want_half):
            bad.append(
                f"{row['experiment']}#{row['point']}: src_kind={kind} but its source is "
                f"stamped {sig}, which is the other arm's half of the signature space"
            )
            continue
        if kind == "dram":
            dram += 1
        # -1 means the landing address is walked or the sources cycled, so the
        # word names no single tile. Not looked at, which is not the same as
        # verified, and saying so is the point.
        if row.get("landed_ok", "-1") == "-1":
            continue
        checked += 1
        if row["landed"].lower() != sig.lower() or row["landed_ok"] != "1":
            bad.append(
                f"{row['experiment']}#{row['point']}: src_kind={kind} landed "
                f"{row['landed']}, but its source is stamped {sig}"
            )
    if bad:
        problems.append(
            f"{len(bad)} row(s) did not prove which KIND of tile their timed burst "
            "read; THE SOURCE ARM DID NOT TAKE. First few:"
        )
        problems.extend("    " + line for line in bad[:5])
    elif checked:
        notes.append(
            f"{checked} row(s) ({dram} of them DRAM) proved their source from the "
            "signature their own timed burst landed, not from the --dram flag"
        )
    return problems


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


def dram_marginals(rows):
    """``{(src_kind, tx_bytes): marginal cycles/transaction}`` for the DRAM arm.

    Exactly the arithmetic :func:`marginals` does on the ``burst`` axis, applied
    per (source kind, payload size) to the ``bigburst`` / ``dramburst`` /
    ``dramsize`` experiments. Differencing consecutive burst lengths removes the
    loop's constant term -- its prologue, the closing barrier, the launch -- so
    what is left is the cost of one more transaction in flight, which is the
    quantity tt-sim charges as pure bandwidth and the only one the registered
    prediction is about.

    The marginal reported per key is the mean over the intervals that exclude
    N = 4: the shortest burst carries a launch-warmth term that the longer ones
    have amortised, and it is the one interval where the simulator itself shows
    a fractional wobble.
    """
    per_key = {}
    for row in rows:
        if row.get("experiment") not in ("bigburst", "dramburst", "dramsize"):
            continue
        key = (row.get("src_kind", "?"), int(row["tx_bytes"]))
        per_key.setdefault(key, {}).setdefault(int(row["num_tx"]), []).append(
            int(row["cycles"])
        )
    out = {}
    for key, per_n in per_key.items():
        means = {n: sum(v) / len(v) for n, v in per_n.items()}
        order = sorted(means)
        steps = [
            (means[hi] - means[lo]) / (hi - lo) for lo, hi in zip(order, order[1:])
        ]
        long_steps = [
            (means[hi] - means[lo]) / (hi - lo)
            for lo, hi in zip(order, order[1:])
            if lo >= 16
        ]
        chosen = long_steps or steps
        if chosen:
            out[key] = sum(chosen) / len(chosen)
    return out


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

    # PLAIN INSTRUMENTATION, checked from the artefact rather than remembered.
    # tt-metal's NoC-event instrumentation injects a per-transaction profiler
    # write into the timed loop (+10-19 % of span on silicon) and does not tax
    # the arms equally -- it inflated an unbatched variant more than a batched
    # one and turned a real 7-9 % deficit into 1.00-1.03, sign-flipped at the
    # largest size. The program refuses to run with it set, so a file that
    # exists carries `noc_events=0`; a file without the field came from a binary
    # that predates the gate and cannot say either way.
    if header.get("noc_events") not in (None, "0"):
        problems.append(
            f"the header says noc_events={header['noc_events']} -- this run was "
            "instrumented, and the instrumentation masks the effect being measured"
        )

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


def dram_verdict(arch, marg, header):
    """The pre-registered read of the DRAM arm. Returns True when it holds.

    Every figure it compares against was written down in
    :data:`SIM_DRAM_PREDICTION` before any card ran the arm, and this function
    does nothing but say which of three outcomes the numbers landed on. As with
    the stateful arm, it deliberately cannot express "somewhere in between" as a
    pass, and one of the three is a loss for the simulator.
    """
    print("")
    tag = "  -- SIMULATOR, NOT A MEASUREMENT" if header.get("sim") == "1" else ""
    print(f"== the DRAM burst arm, {arch}{tag}")
    if header.get("noc_events") is None:
        print(
            "   REFUSED: this CSV has no noc_events field, so it cannot show it was "
            "taken without tt-metal's NoC instrumentation. Rebuild and retake."
        )
        return False
    if not marg:
        print("   no bigburst/dramburst/dramsize rows; was --dram passed?")
        return False

    rows, missed_dram, missed_l1 = [], [], []
    for (kind, size), got in sorted(marg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        want = SIM_DRAM_PREDICTION.get((arch, kind, size))
        if want is None:
            print(f"   {kind:>4} {size:>5} B: {got:7.2f}  (no registered prediction)")
            continue
        drift = (got - want) / want
        flag = "reproduces" if abs(drift) <= DRAM_TOLERANCE else "MISSED"
        print(
            f"   {kind:>4} {size:>5} B: measured {got:7.2f}  predicted {want:7.2f}  "
            f"{drift * 100:+6.1f} %  {flag}"
        )
        rows.append((kind, size, got, want, drift))
        if abs(drift) > DRAM_TOLERANCE:
            (missed_dram if kind == "dram" else missed_l1).append((size, got, want))

    # THE SHARPEST FALSIFIER, and it needs no prediction to be right about the
    # absolute numbers. Below the crossover tt-sim says the issue loop binds
    # both arms, so DRAM and L1 must be THE SAME NUMBER. A gap there cannot be
    # absorbed by any bandwidth term, however wrong the bandwidth term is.
    crossover = DRAM_CROSSOVER.get(arch)
    print("")
    if crossover is None:
        print(
            f"   no registered crossover for {arch}; the sub-crossover check is skipped"
        )
    else:
        gaps = []
        for size in sorted({size for _, size in marg}):
            if size >= crossover:
                continue
            dram, l1 = marg.get(("dram", size)), marg.get(("l1", size))
            if dram is None or l1 is None:
                continue
            gaps.append((size, dram - l1, dram, l1))
        if not gaps:
            print("   SUB-CROSSOVER: no size has both arms; the check cannot run")
        else:
            worst = max(gaps, key=lambda g: abs(g[1]))
            for size, gap, dram, l1 in gaps:
                print(
                    f"   SUB-CROSSOVER {size:>5} B: dram {dram:7.2f} - l1 {l1:7.2f} "
                    f"= {gap:+6.2f}  (tt-sim says 0.00: both arms are issue-loop bound)"
                )
            if abs(worst[1]) > 0.05 * worst[3]:
                print(
                    f"   -> A REAL GAP. At {worst[0]} B both arms issue the same "
                    "instructions at the same rate and tt-sim charges the transfer "
                    f"nothing extra either way, yet DRAM costs {worst[1]:+.2f} cycles "
                    "more per transaction. No bandwidth term can absorb that: it is a "
                    "per-transaction cost at the DRAM endpoint, which is the term "
                    "charged at exactly zero."
                )
            else:
                print(
                    "   -> No gap. Below the crossover the two endpoints cost the same, "
                    "as the model says, so any excess above the crossover is a "
                    "bandwidth question rather than a per-transaction one."
                )

    print("")
    if missed_l1:
        print(
            "   CONTROL MOVED -- the L1 arm missed its own registered figure at "
            f"{', '.join(str(s) for s, _, _ in missed_l1)} B. The instruction-level "
            "model of the issue loop is what both arms rest on, so the pair cannot be "
            "read: report the numbers and do NOT attribute the difference to DRAM."
        )
        return False
    if not missed_dram:
        print(
            "   PREDICTION 1 -- BANDWIDTH AND NOTHING ELSE. Every DRAM marginal landed "
            f"within {DRAM_TOLERANCE * 100:.0f} % of a figure derived from link "
            "occupancy plus the DRAM channel's excess, with endpoint queueing, "
            "outstanding-transaction credits and response reordering all charged zero. "
            "On this part, at these sizes, charging them zero is not costing anything."
        )
        return True
    excess = [(size, got - want) for size, got, want in missed_dram]
    over = [e for _, e in excess if e > 0]
    print(
        "   PREDICTION 2 -- THERE IS A DRAM-SIDE COST THE MODEL DOES NOT CHARGE. The "
        f"L1 arm reproduced its figures and the DRAM arm did not, at "
        f"{', '.join(f'{s} B ({e:+.2f} cycles)' for s, e in excess)}."
    )
    if len(over) > 1:
        flat = max(over) - min(over) <= 0.25 * max(over)
        print(
            "   The excess is roughly CONSTANT in payload size, so it is a "
            "per-transaction endpoint cost -- exactly the term charged at zero."
            if flat
            else "   The excess GROWS with payload size, so it is a bandwidth "
            "misestimate rather than a per-transaction cost. Different fix."
        )
    return False


def print_predictions(arch):
    """The registered table, for the runner to print BEFORE any card time.

    It lives here rather than in the shell script so that the figures the
    operator reads before the run and the figures the verdict is graded against
    are the same object. A prediction the runner prints and the checker does not
    apply is not a pre-registration.
    """
    sizes = sorted({size for a, _, size in SIM_DRAM_PREDICTION if a == arch})
    if not sizes:
        print(f"  no registered DRAM prediction for {arch!r}")
        return 1
    crossover = DRAM_CROSSOVER.get(arch)
    print(
        f"  tt-sim's ABSOLUTE prediction for {arch}, marginal cycles per transaction:"
    )
    print("")
    print("     payload    DRAM source    L1 source    what the model says binds")
    for size in sizes:
        dram = SIM_DRAM_PREDICTION.get((arch, "dram", size))
        l1 = SIM_DRAM_PREDICTION.get((arch, "l1", size))
        binds = (
            "the issue loop, both arms"
            if crossover is not None and size < crossover
            else "the transfer: link, plus the DRAM channel's excess"
        )
        print(f"     {size:>5} B    {dram:>11.2f}    {l1:>9.2f}    {binds}")
    print("")
    if crossover is not None:
        print(
            f"  Below {crossover} B the two arms are predicted to be THE SAME NUMBER: the"
        )
        print(
            "  issue loop is slower than either transfer, so the endpoint costs nothing"
        )
        print(
            "  extra. That is the sharpest thing to falsify here -- a gap there cannot be"
        )
        print("  explained by any bandwidth term, however wrong the bandwidth term is.")
    return 0


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
        "csv", nargs="*", help="nocreadbench-<arch>.csv from the run(s)"
    )
    parser.add_argument(
        "--print-predictions",
        metavar="ARCH",
        help=(
            "print the registered DRAM predictions for ARCH and exit, so the runner "
            "shows the operator the same figures the verdict will be graded against"
        ),
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
    parser.add_argument(
        "--dram",
        action="store_true",
        help=(
            "print the DRAM burst arm's verdict against the predictions registered "
            "in SIM_DRAM_PREDICTION, which were written down before any card ran it"
        ),
    )
    args = parser.parse_args(argv)
    if args.print_predictions:
        return print_predictions(args.print_predictions)
    if not args.csv:
        parser.error("at least one CSV is required")

    status = 0
    runs = []
    # Pooled across the CSVs of one session, because a `--dram` session runs the
    # same plan several times and the marginal wanted is the one over all of
    # them: rounds of an identical program are repeats, not separate results.
    dram_rows, dram_header = [], {}
    for path in args.csv:
        ok, header, marg, _ = report_one(path, args.expect, args.quiet)
        if not ok:
            status = 1
        _, rows = read_csv(path)
        mode = rows[0]["mode"] if rows and "mode" in rows[0] else "unknown"
        runs.append((header, mode, marg))
        if args.dram and mode == "stateless":
            dram_rows.extend(rows)
            dram_header = dram_header or header

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

    if args.dram:
        arch = dram_header.get("arch", "unknown")
        if not dram_verdict(arch, dram_marginals(dram_rows), dram_header):
            status = 1
    if len(runs) > 1:
        paired_verdict(runs)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
