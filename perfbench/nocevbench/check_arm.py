#!/usr/bin/env python3
"""Did the arm the flag asked for actually run?

**This is the check the arm-B session turns on.** Arm B swaps the two NoCs and
nothing else, so a run that silently kept arm A's pairing produces a perfectly
well-formed trace, passes every gate in ``tt_sim.perf.noc_events``, and yields a
confident wrong conclusion: it would say the +95-cycle error "stayed on the
write", which is precisely what the *rival* hypothesis predicts. Passing
``--arm B`` proves nothing on its own -- a stale binary, a build that did not
pick up the source change, a shell that dropped the argument -- so the claim is
checked against the **emitted trace**, which carries the NoC id each transaction
actually used (``noc`` in every record; ``profiler.cpp:840-887``).

Deliberately standalone: it imports nothing but the standard library, so it runs
on a card box that has only ``perfbench/nocevbench/`` rsynced onto it and no
tt-sim at all. The arm table below is the second copy of the one in
``src/nocevbench.cpp``; they are three rows, and the point of the duplication is
that this file checks the program rather than agreeing with it by construction.

    ./check_arm.py --arm B --trace .logs/noc_trace_dev0_ID0.json
    ./check_arm.py --arm C --trace .logs/noc_trace_dev0_ID0.json --stdout stdout.log

Exit status is 0 when everything the arm claims is visible in the trace, and
non-zero otherwise -- so it can gate a run in a shell script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

#: ``arm -> (reader RISC's NoC, writer RISC's NoC, target)``. The same three
#: rows as ``ARMS[]`` in ``src/nocevbench.cpp``.
ARMS = {
    "A": ("NOC_1", "NOC_0", "dram"),
    "B": ("NOC_0", "NOC_1", "dram"),
    "C": ("NOC_1", "NOC_0", "l1"),
}

#: Event types that put a packet on the wire, by direction. ``*_BARRIER_*``,
#: ``SEMAPHORE_*`` and the ``*_SET_STATE`` family are excluded: a barrier
#: records the NoC of the command buffer it polls, not of a transfer, and
#: including them would let a stream with zero transfers "pass" the pairing.
READ_ISSUES = {
    "READ",
    "READ_WITH_STATE",
    "READ_WITH_STATE_AND_TRID",
    "READ_DRAM_SHARDED_WITH_STATE",
}
WRITE_ISSUES = {
    "WRITE_",
    "WRITE_WITH_TRID",
    "WRITE_INLINE",
    "WRITE_MULTICAST",
    "WRITE_WITH_STATE",
    "WRITE_WITH_TRID_WITH_STATE",
}

CONFIG_RE = re.compile(r"^nocevbench-config\s+(.*)$", re.MULTILINE)


def parse_config_line(text):
    """The ``nocevbench-config`` line the program prints, as a dict.

    Returns ``None`` when the log carries no such line -- which itself means the
    binary predates the arms and cannot have honoured ``--arm``.
    """
    match = CONFIG_RE.search(text)
    if match is None:
        return None
    fields = {}
    for token in match.group(1).split():
        key, _, value = token.partition("=")
        fields[key] = value
    return fields


def issue_pairing(records):
    """``{(proc, direction): {noc ids seen}}`` over transaction-issuing events."""
    pairing = {}
    for rec in records:
        kind = rec.get("type")
        if kind in READ_ISSUES:
            direction = "read"
        elif kind in WRITE_ISSUES:
            direction = "write"
        else:
            continue
        key = (str(rec.get("proc", "")), direction)
        pairing.setdefault(key, set()).add(str(rec.get("noc", "")))
    return pairing


def destinations(records):
    """``{(proc, direction): {(dx, dy) seen}}`` over transaction-issuing events."""
    dests = {}
    for rec in records:
        kind = rec.get("type")
        if kind in READ_ISSUES:
            direction = "read"
        elif kind in WRITE_ISSUES:
            direction = "write"
        else:
            continue
        if "dx" not in rec or "dy" not in rec:
            continue
        key = (str(rec.get("proc", "")), direction)
        dests.setdefault(key, set()).add((int(rec["dx"]), int(rec["dy"])))
    return dests


def source_core(records):
    """The initiating core as the TRACE numbers it (``sx``/``sy``), or ``None``.

    Load-bearing, because the trace and the ``nocevbench-config`` line are not
    always in the same coordinate space. tt-metal's host writes every trace
    coordinate in **NOC 0** space (``profiler.cpp``,
    ``translateNocCoordinatesToNoc0``), while ``peer_noc`` on the config line is
    whatever ``worker_core_from_logical_core`` returned -- which is the
    *translated* coordinate on a part running with NoC coordinate translation.
    On Blackhole a worker's translated coord **is** its NOC 0 coord, so the two
    agree and this never mattered; on Wormhole they do not, so a naive equality
    test refuses a perfectly good run. Comparing ``self_noc`` against ``sx/sy``
    is how the two spaces are told apart with no knowledge of the part.
    """
    for rec in records:
        if "sx" in rec and "sy" in rec:
            return (int(rec["sx"]), int(rec["sy"]))
    return None


def latencies(records):
    """``{(noc, type, bytes): [samples]}`` -- issue to the barrier END covering it.

    The same quantity ``tt_sim.perf.noc_events`` reports, recomputed here in
    thirty lines so the operator can see the arm's numbers **on the card**,
    before the session is sent home. It is what makes "arm A is the control and
    must reproduce 2026-08-17" an on-the-spot check rather than a promise.

    It is **not** a per-packet flight time: the artefact carries no
    per-transaction completion timestamp, only a barrier's, so this is an upper
    bound that includes every other transaction in the same batch. Same bound on
    both sides, computed the same way, which is what makes it comparable.
    """
    per_proc = {}
    for rec in records:
        per_proc.setdefault(str(rec.get("proc", "")), []).append(rec)
    out = {}
    for stream in per_proc.values():
        stream.sort(key=lambda r: r["timestamp"])
        pending = {"read": [], "write": []}
        for rec in stream:
            kind = rec.get("type")
            if kind in READ_ISSUES:
                pending["read"].append(rec)
            elif kind in WRITE_ISSUES:
                pending["write"].append(rec)
            elif kind in ("READ_BARRIER_END", "WRITE_BARRIER_END"):
                side = "read" if kind == "READ_BARRIER_END" else "write"
                for issued in pending[side]:
                    key = (
                        str(issued.get("noc", "")),
                        str(issued.get("type", "")),
                        int(issued.get("num_bytes") or 0),
                    )
                    out.setdefault(key, []).append(
                        rec["timestamp"] - issued["timestamp"]
                    )
                pending[side] = []
    return out


def format_latencies(records):
    """One ``NOC_x TYPE bytes n mean min max`` line per class, as a string."""
    lines = []
    for key, samples in sorted(latencies(records).items()):
        mean = sum(samples) / len(samples)
        lines.append(
            f"{key[0]} {key[1]} {key[2]}B n={len(samples)} "
            f"mean={mean:.1f} min={min(samples)} max={max(samples)}"
        )
    return "; ".join(lines)


def _peer_mismatch(peer, seen_dests, records, config, notes):
    """Why arm C's destinations are not literally ``peer`` -- as problems, or none.

    Three outcomes, and telling them apart is the point:

    1. **A different coordinate space.** The config line reports the peer in the
       coordinate system ``worker_core_from_logical_core`` uses -- *translated*
       on a part running NoC coordinate translation, which every real Wormhole
       card does -- while the trace is written in NOC 0 space. Detected by the
       same disagreement showing on the initiating core (``self_noc`` vs the
       trace's ``sx``/``sy``), which no wrong-peer run can produce. The identity
       is then unverifiable from coordinates alone, so what is checked is the
       shape (one core, both directions, and not this core), and the run passes
       with both numbers named rather than being refused for being right.
    2. **The NoC 1 grid mirror.** The destination is ``(GX-1-x, GY-1-y)`` of the
       peer, which means the kernel put an *unmirrored* worker coordinate on
       NoC 1: tt-sim's untranslated convention, and not the one a card is in.
       Refused, naming the fix, because the two sides are then not comparable.
    3. Anything else -- the transfers went somewhere the arm did not name.
    """
    grid = None
    if config is not None and "," in config.get("noc_grid", ""):
        grid = tuple(int(v) for v in config["noc_grid"].split(","))
    if grid is not None:
        mirror = (grid[0] - 1 - peer[0], grid[1] - 1 - peer[1])
        # Either direction alone can carry the mirror: the writer runs NOC 0 and
        # addresses the peer, the reader runs NOC 1 and addresses its mirror, so
        # the untranslated shape is the mixed set as often as the pure one.
        if mirror in seen_dests and seen_dests <= {peer, mirror}:
            return [
                f"arm C addressed {mirror}, which is the NoC 1 grid mirror of the peer "
                f"{peer} on a {grid[0]}x{grid[1]} grid. The kernel put an UNMIRRORED "
                "worker coordinate on NoC 1, which is tt-sim's untranslated "
                "convention and not the one a card runs in, so the two sides are not "
                "comparable. Re-run with TT_METAL_MOCK_CLUSTER_DESC_PATH pointing at "
                "driver/<arch>/cluster_descriptor.yaml"
            ]

    self_noc = None
    if config is not None and "," in config.get("self_noc", ""):
        self_noc = tuple(int(v) for v in config["self_noc"].split(","))
    trace_self = source_core(records)
    if self_noc is not None and trace_self is not None and self_noc != trace_self:
        if len(seen_dests) != 1:
            return [
                f"arm C addresses one peer core in both directions, but this trace "
                f"addresses {sorted(seen_dests)}"
            ]
        if seen_dests == {trace_self}:
            return [
                f"arm C addressed {trace_self}, which is the INITIATING core -- the "
                "transfers never left it"
            ]
        notes.append(
            f"peer identity checked by shape, not by coordinate: the config reports "
            f"this core at {self_noc} while the trace numbers it {trace_self}, so the "
            f"two are in different coordinate spaces (translated vs NOC 0). Every "
            f"transfer addressed the single core {sorted(seen_dests)[0]}, which is "
            f"not this one; the config names the peer {peer}"
        )
        return []

    return [
        f"arm C addressed {sorted(seen_dests)} but the peer is {peer} -- the "
        "transfers did not go to the core the arm names"
    ]


def check(arm, records, config=None):
    """``(problems, notes)`` for one trace against one arm's claims."""
    reader_noc, writer_noc, target = ARMS[arm]
    problems, notes = [], []

    if config is not None:
        if config.get("arm") != arm:
            problems.append(
                f"the program reported arm={config.get('arm')!r} but this check was asked "
                f"for arm {arm!r} -- the run and the analysis disagree about what ran"
            )
        if config.get("reader") != f"NCRISC:{reader_noc}":
            problems.append(
                f"the program reported reader={config.get('reader')!r}, arm {arm} wants "
                f"NCRISC:{reader_noc} -- a stale binary that ignores --arm looks exactly like this"
            )
        if config.get("writer") != f"BRISC:{writer_noc}":
            problems.append(
                f"the program reported writer={config.get('writer')!r}, arm {arm} wants BRISC:{writer_noc}"
            )
        if config.get("target") != target:
            problems.append(
                f"the program reported target={config.get('target')!r}, arm {arm} wants {target}"
            )

    pairing = issue_pairing(records)
    expected = {("NCRISC", "read"): reader_noc, ("BRISC", "write"): writer_noc}
    for key, want in expected.items():
        seen = pairing.get(key)
        if not seen:
            problems.append(
                f"{key[0]} issued no {key[1]} transactions at all -- this trace is not "
                f"nocevbench arm {arm}"
            )
        elif seen != {want}:
            problems.append(
                f"{key[0]}'s {key[1]}s ran on {sorted(seen)}, arm {arm} asked for {want} "
                "-- THE NoC ASSIGNMENT DID NOT TAKE, and this run measured a different arm"
            )
        else:
            notes.append(
                f"{key[0]} {key[1]}s on {want}: {len(records)} records scanned"
            )
    for key in sorted(set(pairing) - set(expected)):
        problems.append(
            f"{key[0]} issued {key[1]} transactions, which no nocevbench arm does"
        )

    dests = destinations(records)
    seen_dests = set()
    for value in dests.values():
        seen_dests |= value
    if target == "l1":
        peer = None
        if config is not None and config.get("peer_noc", "-") != "-":
            peer = tuple(int(v) for v in config["peer_noc"].split(","))
        if peer is None:
            # Without the log we do not know *which* core, but we do know arm C
            # addresses ONE core in both directions while a DRAM arm addresses
            # two different cells of the channel -- one per NoC. So the count
            # still separates them, and the identity is what --stdout adds.
            if len(seen_dests) != 1:
                problems.append(
                    f"arm C addresses one peer core in both directions, but this "
                    f"trace addresses {sorted(seen_dests)} -- that is the shape of "
                    "a DRAM arm, not of arm C"
                )
            else:
                notes.append(
                    f"every transfer addressed {sorted(seen_dests)[0]}; pass "
                    "--stdout to check that it is the peer the arm named"
                )
        elif seen_dests == {peer}:
            notes.append(f"every transfer addressed the peer at {peer}")
        else:
            problems.extend(_peer_mismatch(peer, seen_dests, records, config, notes))
    else:
        notes.append(f"destinations {sorted(seen_dests)} (DRAM endpoints)")
        if config is not None and config.get("peer_noc", "-") != "-":
            notes.append("peer_noc reported on a DRAM arm; ignored")

    return problems, notes


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Check that a nocevbench NoC event trace really ran the arm it claims. "
            "Reads the trace, not the command line."
        )
    )
    parser.add_argument(
        "--arm",
        required=True,
        choices=sorted(ARMS),
        help="the arm the run claims to be",
    )
    parser.add_argument(
        "--trace", required=True, help="noc_trace_dev*_ID*.json from the run"
    )
    parser.add_argument(
        "--stdout",
        help=(
            "the run's stdout.log; its nocevbench-config line is cross-checked "
            "against the arm and against the trace"
        ),
    )
    parser.add_argument(
        "--latencies",
        action="store_true",
        help=(
            "also print each transaction class's observed issue-to-barrier-END "
            "latency, so the control arm can be checked against the previous "
            "session on the card rather than after the session is sent home"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="print only on failure")
    args = parser.parse_args(argv)

    with open(args.trace) as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        print(f"check_arm: {args.trace} is not a NoC trace array", file=sys.stderr)
        return 2

    config = None
    if args.stdout:
        with open(args.stdout) as handle:
            config = parse_config_line(handle.read())
        if config is None:
            print(
                f"check_arm: {args.stdout} has no 'nocevbench-config' line -- the binary "
                "predates --arm and cannot have honoured it",
                file=sys.stderr,
            )
            return 1

    problems, notes = check(args.arm, records, config)
    if problems:
        print(
            f"check_arm: ARM {args.arm} DID NOT RUN AS ASKED ({args.trace})",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  FAIL {problem}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"check_arm: arm {args.arm} confirmed from the trace ({args.trace})")
        for note in notes:
            print(f"  ok   {note}")
    if args.latencies:
        print(f"  lat  {format_latencies(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
