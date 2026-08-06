"""The cost-model gate — the acceptance run for ``TT_SIM_COST_MODEL=1``.

Why this exists
---------------

§I's cycle-cost model (``docs/plans/cost-model.md``) is a **timing**
perturbation: with ``TT_SIM_COST_MODEL=1`` the baby RISC-V cores stall on a
load-use interlock and five Tensix units hold themselves occupied, so simulated
cycle counts move. A guard that replays a captured wire conversation *verbatim*
— including the number of times the tt-metal host polled the go-message before
reading the result back — breaks by construction: a device that reaches the
same answer one poll later reads a partially written buffer and the replay
mismatches. **Byte-identical replay at the recorded poll count is a timing pin:
any change to timing makes it fail, whether or not a value went wrong.**

The Wormhole guards that used to have that shape
(``examples_replay_test`` / ``offline_replay_test``, and ``one_replay_test``
via ``replay.py``) have since been converted to poll-until-DONE: they still
assert every *data* READ reply bit-for-bit, but they pump the device to
``RUN_MSG_DONE`` first and exempt only the spin-polled go-message itself, so
their verdict no longer depends on the recorded poll budget. The one remaining
timing-pinned guard is ``blackhole/offline``.

The property that decides how a guard must be treated is therefore not what it
asserts but **whether its verdict depends on the recorded poll budget**:

``value/pump-to-done`` — budget-independent
    Pump until the go-message flips to ``RUN_MSG_DONE`` (mirroring tt-metal's
    own ``wait_until_cores_done``), *then* read the result buffer and check it.
    A slower device takes more pumps and reaches the same answer. **These are
    the gate**, run as-is under the model.

``timing-pinned`` — budget-dependent
    Assert that every READ reply reproduces the recorded one bit-for-bit. A
    correctness guard *and* a timing pin, inseparably.

``value/poll-budget`` — budget-dependent
    Check a computed value, but take the device's cycles from the trace's own
    polls rather than waiting for ``DONE``. What they *assert* is
    timing-agnostic; whether they get far enough to assert it is not.

**The proof, for both budget-dependent families, is to re-run at a larger
`cycles_per_poll`** — the one knob that says "the host waited longer", which is
what a live host does and what the recorded poll count cannot express. The
replay is re-run whole at 1×, 2×, 4×, 8× the recorded 100 cycles per poll, and
the guard must come out *completely clean* at some multiple: byte-identical for
the timing-pinned family, value-correct for the poll-budget family. The
multiplier it needed is reported.

**Do not "prove" a budget-dependent guard by pumping after the replay
finishes.** These traces end with ``RESET_ASSERT`` to every core followed by
``EXIT``: once the replay is exhausted every core is held in reset, so no amount
of further pumping can advance the kernel, and a perfectly benign timing shift
reads as a permanently wrong value. An earlier version of this module did
exactly that and reported a false "not a timing artefact" on
``blackhole/two`` — the most expensive kind of wrong answer a gate can give,
because it is the signal that is supposed to stop everyone and start an
investigation.

What a green gate establishes
-----------------------------

* The cost model changes **no computed value**: every budget-independent guard
  passes as-is, and every budget-dependent one is completely clean once given a
  realistic poll budget.
* The model does not crash, deadlock or reorder anything into a wrong answer on
  any in-tree workload.

What it does **not** establish
------------------------------

* **Nothing about whether the modelled cycle counts are right.** The gate never
  compares a cycle count to anything. ``cost-model.md`` is explicit that a total
  is not yet a prediction: there is no branch predictor to charge a mispredict
  against, no instruction-cache miss cost, and no router arbitration. A green
  gate means "the model is value-safe",
  never "the model is calibrated". The poll-budget multiplier is not a cycle
  measurement either — "2x" only brackets the slowdown at ladder resolution.
* Nothing about workloads with no in-tree guard.

Running it
----------

::

    python3 -m driver.tests.cost_model_gate            # the whole gate
    python3 -m driver.tests.cost_model_gate --list     # the classification only
    python3 -m driver.tests.cost_model_gate --stage value six two
    python3 -m driver.tests.cost_model_gate --model-off # the same set, model off

Exit status is 0 only if every stage passed. The guard *list* is discovered by
glob, not hardcoded, so a guard added by anyone else is picked up automatically;
:data:`BASELINE` records the classification that was validated by hand, and a
guard whose heuristic classification disagrees with it is a hard failure rather
than a silent reclassification (``driver/tests/guard_classification_test.py``
runs that check in milliseconds as part of the normal ``pytest driver``).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]

#: A guard that asserts recorded READ replies bit-for-bit. Correctness *and*
#: timing, inseparably — cannot validate a timing model as it stands.
TIMING_PINNED = "timing-pinned"
#: A guard that pumps until the go-message reads ``RUN_MSG_DONE`` and then
#: checks the result. Its verdict does not depend on the recorded poll budget,
#: so it is the gate proper.
VALUE_PUMPED = "value/pump-to-done"
#: A guard that checks a value it computed itself, but takes the device's cycles
#: from the trace's own polls rather than waiting for DONE. What it *asserts* is
#: timing-agnostic; whether it gets far enough to assert it is not.
VALUE_POLL_BUDGET = "value/poll-budget"

#: The cut that actually decides how the gate treats a guard: does its verdict
#: depend on how many cycles the *recorded* conversation happened to allow? Both
#: of these families need the poll-budget prover; neither can be run as-is under
#: a timing change and called a result.
POLL_BUDGET_DEPENDENT = (TIMING_PINNED, VALUE_POLL_BUDGET)
#: Everything else: run it under the model and believe the answer.
BUDGET_INDEPENDENT = (VALUE_PUMPED,)

#: The classification validated by hand at this tree state. Discovery drives the
#: run; this is the tripwire that stops a reclassification happening silently.
BASELINE_TREE = "24403ae (+ the Wormhole poll-until-DONE guard conversion)"
BASELINE = {
    "blackhole/dramtop": VALUE_POLL_BUDGET,
    "blackhole/eight": VALUE_PUMPED,
    "blackhole/five": VALUE_PUMPED,
    "blackhole/five_fp": VALUE_PUMPED,
    "blackhole/four": VALUE_PUMPED,
    "blackhole/four_fp": VALUE_PUMPED,
    "blackhole/loopback": VALUE_PUMPED,
    "blackhole/matmulblock": VALUE_PUMPED,
    "blackhole/matmulidx": VALUE_PUMPED,
    "blackhole/nine": VALUE_PUMPED,
    "blackhole/noc_tile_transfer": VALUE_PUMPED,
    "blackhole/pad_multi_core": VALUE_PUMPED,
    "blackhole/offline": TIMING_PINNED,
    "blackhole/pipestall": VALUE_PUMPED,
    "blackhole/shard_data_rm": VALUE_PUMPED,
    "blackhole/vecadd_sharding": VALUE_PUMPED,
    "blackhole/optest": VALUE_PUMPED,
    "blackhole/reduce": VALUE_PUMPED,
    "blackhole/reduceneg": VALUE_PUMPED,
    "blackhole/sfpuchain": VALUE_PUMPED,
    "blackhole/sfpumath": VALUE_PUMPED,
    "blackhole/six": VALUE_PUMPED,
    "blackhole/softplus": VALUE_PUMPED,
    "blackhole/three": VALUE_PUMPED,
    "blackhole/tilize": VALUE_PUMPED,
    "blackhole/transpose": VALUE_PUMPED,
    "blackhole/two": VALUE_POLL_BUDGET,
    "blackhole/twolaunch": VALUE_PUMPED,
    "blackhole/untilize": VALUE_PUMPED,
    "blackhole/where": VALUE_PUMPED,
    "wormhole/noc_tile_transfer": VALUE_PUMPED,
    "wormhole/examples": VALUE_PUMPED,
    "wormhole/matmulblock": VALUE_PUMPED,
    "wormhole/matmulidx": VALUE_PUMPED,
    "wormhole/offline": VALUE_PUMPED,
    "wormhole/one": VALUE_PUMPED,
    "wormhole/pipestall": VALUE_PUMPED,
    "wormhole/reduce": VALUE_PUMPED,
    "wormhole/reduceneg": VALUE_PUMPED,
    "wormhole/sfpumath": VALUE_PUMPED,
    "wormhole/softplus": VALUE_PUMPED,
    "wormhole/tilize": VALUE_PUMPED,
    "wormhole/transpose": VALUE_PUMPED,
    "wormhole/untilize": VALUE_PUMPED,
}

#: Budget-dependent guards the prover does not drive, and why. Kept as prose
#: rather than a bare set: an exclusion without a reason is how an exclusion
#: list rots into a way of not looking. Currently empty — the last entry
#: (``wormhole/one``) left when replay.py itself became poll-until-DONE, which
#: made the guard budget-independent and runnable under the model directly.
UNPROVABLE = {}

#: The bridge's recorded default: cycles the device advances per host message.
#: The captured traces were all taken at this value.
RECORDED_CYCLES_PER_POLL = 100
#: Multiples of the recorded budget the prover tries, in order. A budget-
#: dependent guard is proven benign by coming out *completely* clean at one of
#: them — not by mismatching less. 8x is where a genuinely wrong value stops
#: being plausibly "the host would have waited longer".
POLL_BUDGET_LADDER = (1, 2, 4, 8)


class Guard:
    """One discovered ``*_replay_test.py`` guard module."""

    def __init__(self, arch, name, path, kind):
        self.arch = arch
        self.name = name
        self.path = path
        self.kind = kind

    @property
    def key(self):
        return f"{self.arch}/{self.name}"

    @property
    def module(self):
        return f"driver.{self.arch}.server.{self.path.stem}"

    def __repr__(self):
        return f"<Guard {self.key} {self.kind}>"


def classify(source):
    """Classify a guard from its source. See the module docstring for the rules.

    A guard that pumps the device to ``RUN_MSG_DONE`` (the ``PUMP_CAP`` loop)
    is budget-independent even when it *also* compares recorded replies:
    everything it compares outside the spin-polled go-message is settled state
    once the kernel is DONE. Delegating to ``replay.py`` is the same shape —
    replay.py re-sends a spin-polled READ until the reply reaches its final
    recorded value, which is poll-until-DONE driven from the host side. A
    guard is timing-pinned when it compares replayed replies against the
    recording — indexes the parsed trace line's ``["reply"]`` field — with no
    such pump.
    """
    if "PUMP_CAP" in source or "replay.py" in source:
        return VALUE_PUMPED
    if re.search(r"""\[["']reply["']\]""", source):
        return TIMING_PINNED
    return VALUE_PUMPED if "RUN_MSG_DONE" in source else VALUE_POLL_BUDGET


def discover_guards(repo=REPO):
    """Every ``driver/<arch>/server/*_replay_test.py``, classified.

    Discovered rather than listed: two other workstreams are adding guards, and
    a gate that hardcoded the set would quietly stop covering them.
    """
    guards = []
    for path in sorted(repo.glob("driver/*/server/*_replay_test.py")):
        arch = path.parents[1].name
        name = path.stem[: -len("_replay_test")]
        guards.append(Guard(arch, name, path, classify(path.read_text())))
    return guards


def classification_problems(guards):
    """Disagreements between discovery and :data:`BASELINE`, as messages."""
    problems = []
    seen = set()
    for g in guards:
        seen.add(g.key)
        expected = BASELINE.get(g.key)
        if expected is None:
            problems.append(
                f"{g.key}: new guard, not in BASELINE — classify it as "
                f"{g.kind!r} (the heuristic's answer) after reading "
                f"{g.path.relative_to(REPO)}, and add it"
            )
        elif expected != g.kind:
            problems.append(
                f"{g.key}: classified {g.kind!r} but BASELINE says {expected!r} — "
                "the guard changed shape; re-read it before editing either"
            )
    for key in sorted(set(BASELINE) - seen):
        problems.append(f"{key}: in BASELINE but no such guard was discovered")
    for key in sorted(UNPROVABLE):
        if BASELINE.get(key) not in POLL_BUDGET_DEPENDENT:
            problems.append(
                f"{key}: listed UNPROVABLE but is not budget-dependent, so there "
                "is nothing for the prover to have been unable to do"
            )
    return problems


# --- the poll-budget prover ---------------------------------------------------
#
# A budget-dependent guard is proven benign by re-running it **whole** with a
# larger ``cycles_per_poll`` — the one knob that expresses "the host waited
# longer", which is what a live host does and what a recorded poll count cannot.
# The guard must come out completely clean at some multiple of the recorded
# budget: byte-identical for the timing-pinned family, value-correct for the
# poll-budget family. The multiplier it needed is the interesting number.
#
# NOT by pumping after the replay finishes. Every captured trace ends with
# RESET_ASSERT to all cores and then EXIT, so an exhausted replay leaves the
# device with every core held in reset and no amount of further pumping can
# advance the kernel. That mistake reports a benign timing shift as a
# permanently wrong value; see the module docstring.
#
# One class of READ can never reproduce under *any* timing change, in either
# direction: the go-message mailbox the host spin-polls. Its recorded value is
# literally "how far had the kernel got when the host looked", so a device that
# is slower misses a DONE and a device given a longer poll budget reports DONE
# where the recording still said RUNNING. The prover counts those separately
# rather than failing on them — and identifies them from the trace itself (an
# address read many times over from one core is a spin-poll) rather than from a
# hardcoded constant, so it works on either architecture and survives a
# tt-metal offset bump. Everything else, the result buffer included, must be
# byte-identical.


#: READs from the same core at the same address, this many times or more, are a
#: spin-poll. The separation is not marginal: in the captured traces the polled
#: address is read 30-101 times and every other address exactly once.
SPIN_POLL_READS = 8


def _spin_polled(trace):
    """The ``(core, address)`` pairs the host spin-polls in ``trace``."""
    from tt_sim.bridge import protocol as proto
    from tt_sim.bridge.trace import parse_trace_line

    counts = Counter()
    for line in trace.open():
        p = parse_trace_line(line)
        if p is not None and p["cmd"] == proto.CMD_READ:
            counts[(p["core"], p["address"])] += 1
    return {k for k, n in counts.items() if n >= SPIN_POLL_READS}


def _prove_trace(label, trace, build, tolerated):
    """Replay ``trace`` at each rung of the ladder until it is byte-identical."""
    from tt_sim.bridge import Transport
    from tt_sim.bridge import protocol as proto
    from tt_sim.bridge.trace import parse_trace_line

    polled = _spin_polled(trace)
    attempts = []
    for mult in POLL_BUDGET_LADDER:
        device, fabric = build(RECORDED_CYCLES_PER_POLL * mult)
        transport = Transport(addr=None)  # never connects; only _handle is used
        stats = Counter(
            {k: 0 for k in ("reads", "verified", "tolerated", "poll", "bad")}
        )
        failures = []
        with trace.open() as f:
            for lineno, line in enumerate(f, 1):
                p = parse_trace_line(line)
                if p is None:
                    continue
                reply = transport._handle(
                    fabric,
                    SimpleNamespace(
                        cmd=p["cmd"],
                        core=p["core"],
                        address=p["address"],
                        size=p["size"],
                        data=p["data"],
                    ),
                )
                if p["cmd"] != proto.CMD_READ or p["reply"] is None:
                    continue
                stats["reads"] += 1
                want = bytes(p["reply"])
                if bytes(reply) == want:
                    stats["verified"] += 1
                elif (p["core"], p["address"]) in polled:
                    stats["poll"] += 1
                elif tolerated(p):
                    stats["tolerated"] += 1
                else:
                    stats["bad"] += 1
                    if len(failures) < 5:
                        failures.append(
                            f"line {lineno} READ {p['core']}@0x{p['address']:x}: "
                            f"expected {want.hex()} got {bytes(reply).hex()}"
                        )
        device.tt_device.shutdown()
        attempts.append(mult)
        if not stats["bad"]:
            return {
                "label": label,
                "ok": True,
                "multiplier": mult,
                "tried": attempts,
                **stats,
            }
    return {
        "label": label,
        "ok": False,
        "multiplier": None,
        "tried": attempts,
        "failures": failures,
        **stats,
    }


def _blackhole_fabric(pool, cycles_per_poll):
    from driver.blackhole.server.bh_device import make_device
    from driver.blackhole.server.coords import DRAM_COORD_MAP, TENSIX_COORD_MAP
    from tt_sim.bridge import DramCore, Fabric, TensixCore

    device = make_device(cycles_per_poll=cycles_per_poll)
    fabric = Fabric()
    for translated, tile in DRAM_COORD_MAP.items():
        fabric.register(translated, DramCore(device, tile))
    for physical in pool:
        device.ensure_tensix_tile(physical)
        fabric.register(physical, TensixCore(device, TENSIX_COORD_MAP[physical]))
    return device, fabric


def _prove_blackhole_offline():
    from driver.blackhole.server import offline_replay_test as mod

    if not mod.TRACE.exists():
        return [{"label": "offline", "skipped": "trace absent"}]
    # The Blackhole trace tolerates nothing: every READ must reproduce.
    return [
        _prove_trace(
            "offline",
            mod.TRACE,
            lambda cpp: _blackhole_fabric(mod.TENSIX_POOL, cpp),
            lambda p: False,
        )
    ]


def _prove_value_guard(key):
    """Re-run a ``value/poll-budget`` guard's own check at a larger poll budget.

    Generic: the guard modules all pull ``make_device`` into their own namespace
    and call it with no arguments, so rebinding that name is enough to widen the
    budget without touching the guard, its assertions, or the simulator.
    """
    import contextlib
    import importlib
    import io

    arch, name = key.split("/")
    mod = importlib.import_module(f"driver.{arch}.server.{name}_replay_test")
    device_mod = importlib.import_module(
        f"driver.{arch}.server.{'bh' if arch == 'blackhole' else 'wh'}_device"
    )
    original = mod.make_device
    tried = []
    try:
        for mult in POLL_BUDGET_LADDER:
            cpp = RECORDED_CYCLES_PER_POLL * mult
            mod.make_device = lambda cpp=cpp: device_mod.make_device(
                cycles_per_poll=cpp
            )
            tried.append(mult)
            try:
                # The guard prints its own banner; the prover reports the rung.
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = mod.main()
            except AssertionError as exc:
                last = str(exc)
                continue
            if rc == 0:
                return [{"label": name, "ok": True, "multiplier": mult, "tried": tried}]
            last = f"main() returned {rc}"
    finally:
        mod.make_device = original
    return [
        {
            "label": name,
            "ok": False,
            "multiplier": None,
            "tried": tried,
            "failures": [last],
        }
    ]


#: Adapters for the guards whose replay the prover drives directly. Everything
#: else budget-dependent falls through to :func:`_prove_value_guard`, which
#: needs no per-guard knowledge at all.
PROVERS = {
    "blackhole/offline": _prove_blackhole_offline,
}


def run_prover(key):
    """Prove one budget-dependent guard benign. Returns (ok, lines)."""
    reports = PROVERS[key]() if key in PROVERS else _prove_value_guard(key)
    ok = True
    lines = []
    for r in reports:
        if r.get("skipped"):
            lines.append(f"    SKIP {r['label']}: {r['skipped']}")
            continue
        if not r["ok"]:
            ok = False
            lines.append(
                f"    FAIL {r['label']}: still not clean at "
                f"{POLL_BUDGET_LADDER[-1]}x the recorded poll budget "
                f"({RECORDED_CYCLES_PER_POLL * POLL_BUDGET_LADDER[-1]} cycles/poll) "
                "— NOT a poll-budget artefact"
            )
            lines.extend(f"         {f}" for f in r.get("failures", []))
        elif "reads" in r:
            lines.append(
                f"    ok   {r['label']}: all {r['verified']} data READs of "
                f"{r['reads']} bit-for-bit at {r['multiplier']}x poll budget "
                f"({RECORDED_CYCLES_PER_POLL * r['multiplier']} cycles/poll); "
                f"{r['poll']} go-message polls, {r['tolerated']} known-stale"
            )
        else:
            lines.append(
                f"    ok   {r['label']}: value correct at {r['multiplier']}x poll "
                f"budget ({RECORDED_CYCLES_PER_POLL * r['multiplier']} cycles/poll)"
            )
    return ok, lines


# --- stages -------------------------------------------------------------------


def _env(model_on):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}".rstrip(":")
    if model_on:
        env["TT_SIM_COST_MODEL"] = "1"
    else:
        env.pop("TT_SIM_COST_MODEL", None)
    return env


def _run(cmd, model_on, timeout=1800):
    t0 = time.monotonic()
    p = subprocess.run(
        cmd,
        cwd=REPO,
        env=_env(model_on),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p, time.monotonic() - t0


def stage_unit(model_on):
    """The simulator's own unit tests, under the model.

    ``pytest tt_sim`` rather than ``pytest tt_sim driver``: the driver tree
    contains the timing-pinned guards, which this gate handles in stage 3.
    """
    p, secs = _run([sys.executable, "-m", "pytest", "tt_sim", "-q"], model_on)
    tail = [ln for ln in p.stdout.strip().splitlines() if ln.strip()][-1:]
    print(
        f"  {'ok  ' if p.returncode == 0 else 'FAIL'} pytest tt_sim -q: "
        f"{tail[0] if tail else '(no output)'}  [{secs:.0f}s]"
    )
    if p.returncode:
        print("\n".join(f"       {ln}" for ln in p.stdout.splitlines()[-25:]))
    return p.returncode == 0


def stage_value(guards, model_on, jobs):
    """Every budget-independent guard, run as its own process under the model."""

    def one(g):
        p, secs = _run([sys.executable, "-m", g.module], model_on)
        return g, p, secs

    ok = True
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for g, p, secs in pool.map(one, guards):
            out = (p.stdout + p.stderr).strip().splitlines()
            last = out[-1] if out else "(no output)"
            if p.returncode:
                ok = False
                print(f"  FAIL {g.key:<24} [{secs:.0f}s]")
                print("\n".join(f"       {ln}" for ln in out[-12:]))
            else:
                print(f"  ok   {g.key:<24} [{secs:.0f}s] {last[:90]}")
    return ok


def stage_proof(guards, model_on, jobs):
    """Budget-dependent guards: re-run at a larger poll budget until clean."""

    def one(g):
        if g.key in UNPROVABLE:
            return g, None, 0.0
        p, secs = _run(
            [sys.executable, "-m", "driver.tests.cost_model_gate", "--prove", g.key],
            model_on,
        )
        return g, p, secs

    ok = True
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for g, p, secs in pool.map(one, guards):
            if p is None:
                print(f"  EXCL {g.key:<24} not provable in place:")
                print(f"       {UNPROVABLE[g.key]}")
                continue
            if p.returncode:
                ok = False
                print(f"  FAIL {g.key:<24} [{secs:.0f}s]")
            else:
                print(f"  ok   {g.key:<24} [{secs:.0f}s]")
            print(p.stdout.rstrip())
            if p.returncode and p.stderr.strip():
                print("\n".join(f"       {ln}" for ln in p.stderr.splitlines()[-15:]))
    return ok


def print_classification(guards):
    print(f"{'guard':<26} {'classification':<20} role in the gate")
    print("-" * 84)
    for g in guards:
        if g.key in UNPROVABLE:
            role = "excluded — " + UNPROVABLE[g.key].split(",")[0]
        elif g.kind in POLL_BUDGET_DEPENDENT:
            role = "poll-budget proof"
        else:
            role = "run under the model"
        print(f"{g.key:<26} {g.kind:<20} {role}")
    problems = classification_problems(guards)
    if problems:
        print("\nclassification problems:")
        for msg in problems:
            print(f"  ! {msg}")
    return not problems


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cost_model_gate",
        description="Run the timing-agnostic guards with TT_SIM_COST_MODEL=1.",
    )
    ap.add_argument("filters", nargs="*", help="only guards whose key contains this")
    ap.add_argument("--list", action="store_true", help="print the classification")
    ap.add_argument(
        "--stage",
        default="all",
        choices=("all", "unit", "value", "proof"),
        help="run only one stage",
    )
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument(
        "--model-off",
        action="store_true",
        help="run the same set with the model OFF (the A/B baseline)",
    )
    ap.add_argument("--prove", help=argparse.SUPPRESS)  # internal worker
    args = ap.parse_args(argv)
    # A stage takes minutes; stream its progress even when redirected to a file.
    sys.stdout.reconfigure(line_buffering=True)

    if args.prove:
        ok, lines = run_prover(args.prove)
        print("\n".join(lines))
        return 0 if ok else 1

    guards = discover_guards()
    if args.list:
        return 0 if print_classification(guards) else 1

    model_on = not args.model_off
    print(
        f"cost-model gate — TT_SIM_COST_MODEL={'1' if model_on else '(unset)'}\n"
        f"{len(guards)} guards discovered; classification baseline recorded at "
        f"tree {BASELINE_TREE}"
    )

    problems = classification_problems(guards)
    if problems:
        print("\nFAIL classification is out of date:")
        for msg in problems:
            print(f"  ! {msg}")
        return 1

    if args.filters:
        guards = [g for g in guards if any(f in g.key for f in args.filters)]

    value = [g for g in guards if g.kind in BUDGET_INDEPENDENT]
    pinned = [g for g in guards if g.kind in POLL_BUDGET_DEPENDENT]
    ok = True

    if args.stage in ("all", "unit"):
        print("\n[1] simulator unit tests under the model")
        ok &= stage_unit(model_on)

    if args.stage in ("all", "value"):
        print(f"\n[2] budget-independent value guards — {len(value)}")
        ok &= stage_value(value, model_on, args.jobs)

    if args.stage in ("all", "proof"):
        print(f"\n[3] budget-dependent guards — poll-budget proof — {len(pinned)}")
        ok &= stage_proof(pinned, model_on, args.jobs)

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    if model_on:
        print(
            "  Proves: the cost model changes no computed value on any in-tree "
            "workload.\n"
            "  Does not prove: that the modelled cycle counts are right. No cycle "
            "count is compared\n  to anything here, and the poll-budget multiplier "
            "only brackets a slowdown at\n  ladder resolution — a total is not yet "
            "a prediction (docs/plans/cost-model.md)."
        )
    else:
        print(
            "  This is the A/B baseline, not the gate: with the model off every "
            "trace should\n  be 100 % bit-for-bit with nothing late. A late READ "
            "here is a real regression."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
