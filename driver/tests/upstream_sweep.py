"""The upstream-example gate — tt-metal's own ``programming_examples/`` on tt-sim.

Why this exists
---------------

The claim a tt-sim release makes is *"runs real tt-metal programs unmodified on
both architectures"*. The in-tree ``examples/`` ladder does not establish that:
those programs were written here. **This gate runs programs nobody in this repo
wrote** — the ones that ship with tt-metal — against both simulated
architectures, and turns the answer into a number instead of an anecdote.

It is a **breadth** check, not a depth one. It asserts each program's own
verdict (exit status plus the success line the program prints), so what it
proves is "the whole stack got to the right answer", not which unit was
exercised. Depth lives elsewhere: the offline replay guards in
``driver/{wormhole,blackhole}/server/*_replay_test.py`` pin actual bytes for
four of these programs, and ``optests/diff.sh`` diffs Tensix ops against the
vendor reference simulator.

What a green run establishes, and what it does not
--------------------------------------------------

* **Establishes**: every program in the table below reaches its own success
  criterion on the architecture recorded for it, with **no environment variable
  set** — no grid override, no ``TT_SIM_TENSIX_COORDS``. Workers materialise on
  demand, so the gate covers the path a user with an empty environment gets.
* **Does not establish** anything about cycle counts. No timing is compared to
  anything here. It also does not establish correctness for the three
  ``hello_world_*`` programs or ``pad_multi_core`` / ``shard_data_rm``, which
  ship with no self-check — see the ``check`` column, which is honest about
  which is which.

Running it
----------

::

    source /path/to/venv/bin/activate   # or export the four vars
    python3 -m driver.tests.upstream_sweep               # fast tier, both arches
    python3 -m driver.tests.upstream_sweep --tier full   # + the heavy matmuls
    python3 -m driver.tests.upstream_sweep --list        # the table, run nothing
    python3 -m driver.tests.upstream_sweep --arch wormhole eltwise sfpu

Two tiers, because a gate that takes an hour is not a gate:

``fast``  (default)
    Every program that finishes in tens of seconds — 17 of the 21 runnable
    programs, both arches. This is what should run on every change.

``full``
    ``fast`` plus the four grid-sized matmul/vecadd programs, which are minutes
    to tens of minutes each. Invoke it deliberately, before a release.

The last line is ``RESULT: PASS`` or ``RESULT: FAIL``. A verdict that differs
from the one recorded in ``EXPECTED`` below fails the gate **in both
directions**: a newly-passing program is as much a signal to act on (update the
record) as a newly-failing one.

Environment
-----------

Needs a built tt-metal checkout whose ``build/programming_examples/`` binaries
exist (``TT_METAL_RUNTIME_ROOT`` / ``TT_METAL_HOME``), and
``TT_METAL_SLOW_DISPATCH_MODE=1`` — every upstream example calls
``EnqueueProgram``, which falls back to ``detail::LaunchProgram`` under slow
dispatch (the only launch path tt-sim models). ``TT_METAL_SIMULATOR`` is set by
the gate itself, per arch, from this repo — it does **not** use whatever the
caller's environment points at. It skips cleanly when tt-metal is absent.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Deliberate reuse of the sibling harness's internals rather than a second copy:
# these are the tt-metal-locating and simulator-process-cleanup helpers whose
# rationale (why UMD-spawned servers cannot be cleaned up by pid, and why a
# blanket pkill corrupts a concurrent run) is written out once, in
# examples/examples_test.py and driver/sim_procs.sh. ``_run_tag`` and
# ``_kill_own_servers`` take a runner label so this gate reaps only its own.
from examples.examples_test import (  # noqa: E402
    _build_dir,
    _kill_own_servers,
    _reap_orphans,
    _run_tag,
    _tt_metal_root,
)

REPO = Path(__file__).resolve().parents[2]
RUN_LABEL = "upstream_sweep"

# The tree the EXPECTED record below was taken at, and when.
BASELINE_TREE = "d094097"
BASELINE_DATE = "2026-08-13"
BASELINE_TT_METAL = "0.74"

ARCHES = ("wormhole", "blackhole")

# Lines that mean the run went wrong however the program exited. Several
# upstream examples print per-element mismatches and still exit 0, so exit
# status alone is not a correctness signal for them.
FORBIDDEN = re.compile(
    r"Some results did not match"
    r"|PCC not high enough"
    r"|does not match the golden"
    r"|Result mismatch at index"
    r"|Result does not match expected"
    r"|can not handle instruction"
    r"|Traceback \(most recent call last\)"
)

# Hints worth quoting when a run fails.
HINT = re.compile(
    r"can not handle instruction '[^']*'"
    r"|PCC not high enough[^\n]*"
    r"|Some results did not match[^\n]*"
    r"|Result mismatch at index[^\n]*"
    r"|\[DEADLOCK[^\]]*\]"
    r"|(?:\w+Error|TT_FATAL)[^\n]*"
)


class Program:
    """One upstream programming example.

    ``check`` says how strong this row's verdict is, and is printed by
    ``--list`` so the gate states its own coverage:

    ``self``       the program validates its own result and exits non-zero
                   (or prints a failure line) if it is wrong.
    ``value``      the program has no self-check, but prints enough for this
                   gate to check the arithmetic — see ``value_check``.
    ``completion`` no value is checked at all. The program running to its own
                   final line is the whole signal.

    ``report`` is an optional pattern whose match is echoed next to the verdict —
    the PCC a matmul reports, say. It is **not** a criterion: the program's own
    ``TT_FATAL`` already decides whether its PCC is good enough. It exists so a
    green run leaves a number behind rather than only a word.
    """

    def __init__(
        self,
        name,
        binary,
        success,
        *,
        check="self",
        tier="fast",
        timeout=240,
        args=(),
        value_check=None,
        report=None,
        note="",
    ):
        self.name = name
        self.binary = binary
        self.success = success
        self.check = check
        self.tier = tier
        self.timeout = timeout
        self.args = list(args)
        self.value_check = value_check
        self.report = re.compile(report) if report else None
        self.note = note


def _check_contributed_vecadd(out):
    """Verify the ten ``a + b = c`` lines contributed/vecadd prints.

    The example has no self-check and seeds from ``std::random_device`` unless
    given ``--seed``, so it is run with a fixed seed and its own printout is
    used as the value check: every printed sum must be the bfloat16 sum of its
    printed operands. bfloat16 has 8 mantissa bits, so a half-ulp of the result
    is the tolerance.
    """
    rows = re.findall(r"^\s+(\S+) \+ (\S+) = (\S+)\s*$", out, re.M)
    if len(rows) < 10:
        return f"expected 10 'a + b = c' lines, parsed {len(rows)}"
    for a, b, c in rows:
        try:
            fa, fb, fc = float(a), float(b), float(c)
        except ValueError:
            return f"unparseable result line: {a} + {b} = {c}"
        want = fa + fb
        if abs(fc - want) > max(abs(want) * 2**-8, 1e-3):
            return f"{fa} + {fb} = {fc}, want ~{want}"
    return None


# The programs, in the order the sweep runs them. `binary` is relative to
# $TT_METAL_RUNTIME_ROOT/build/programming_examples.
PROGRAMS = [
    # --- single core -------------------------------------------------------
    Program(
        "add_2_integers_in_riscv",
        "metal_example_add_2_integers_in_riscv",
        "Success: Result is 21",
    ),
    Program(
        "add_2_integers_in_compute",
        "metal_example_add_2_integers_in_compute",
        "Success: Result matches expected value!",
    ),
    Program(
        "hello_world_datamovement_kernel",
        "metal_example_hello_world_datamovement_kernel",
        "for the completed task.",
        check="completion",
        note="upstream ships no self-check; DPRINT only",
    ),
    Program(
        "hello_world_compute_kernel",
        "metal_example_hello_world_compute_kernel",
        "for the completed task.",
        check="completion",
        note="upstream ships no self-check; DPRINT only",
    ),
    Program(
        "hello_world_datatypes_kernel",
        "metal_example_hello_world_datatypes_kernel",
        "for handling the data.",
        check="completion",
        note="upstream ships no self-check; DPRINT only",
    ),
    Program("loopback", "metal_example_loopback", "Test Passed"),
    Program("eltwise_binary", "metal_example_eltwise_binary", "Test Passed"),
    Program("eltwise_sfpu", "metal_example_eltwise_sfpu", "Test Passed"),
    Program("custom_sfpi_add", "metal_example_custom_sfpi_add", "Test Passed"),
    Program("custom_smoothstep", "metal_example_custom_smoothstep", "Test Passed"),
    Program(
        "sfpu_eltwise_chain",
        "metal_example_sfpu_eltwise_chain",
        "Metalium vs Golden -- PCC =",
        report=r"PCC = [\d.]+",
        note="TT_FATALs below PCC 0.999; seeds from std::random_device",
    ),
    Program(
        "contributed/vecadd",
        "contributed/vecadd",
        "Partial results:",
        check="value",
        args=("--seed", "42"),
        value_check=_check_contributed_vecadd,
        note="no self-check upstream; this gate checks its printed sums",
    ),
    # --- small multi core --------------------------------------------------
    Program(
        "noc_tile_transfer",
        "metal_example_noc_tile_transfer",
        "Result = 14 : Expected = 14",
        note="2 workers, semaphores",
    ),
    Program(
        "contributed/multicast",
        "contributed/multicast",
        "receiver tiles match the golden tile.",
        note="4 workers, NoC multicast",
    ),
    Program(
        "vecadd_sharding",
        "metal_example_vecadd_sharding",
        "All results match expected values within tolerance.",
        note="4 workers, L1-sharded, no data-movement kernel",
    ),
    Program(
        "shard_data_rm",
        "metal_example_shard_data_rm",
        "Program finished successfully.",
        check="completion",
        note="no self-check; values pinned by shard_data_rm_replay_test.py",
    ),
    Program(
        "pad_multi_core",
        "metal_example_pad_multi_core",
        "Padded tensor with shape",
        check="completion",
        timeout=300,
        note="no self-check; values pinned by pad_multi_core_replay_test.py",
    ),
    # --- grid-sized, minutes each -----------------------------------------
    Program(
        "vecadd_multi_core",
        "metal_example_vecadd_multi_core",
        "All results match expected values within tolerance.",
        tier="full",
        timeout=900,
        note="fills the compute grid: 72 workers (WH) / 130 (BH)",
    ),
    Program(
        "matmul_single_core",
        "metal_example_matmul_single_core",
        "Test Passed",
        report=r"PCC = [\d.]+",
        tier="full",
        timeout=2400,
        note="640^3 = 8000 tile-matmuls on ONE worker; ~10 min, not a hang",
    ),
    Program(
        "matmul_multi_core",
        "metal_example_matmul_multi_core",
        "Test Passed",
        report=r"PCC = [\d.]+",
        tier="full",
        timeout=2400,
        note="640^3 across the whole grid; PCC-checked against a host golden",
    ),
    Program(
        "matmul_multicore_reuse",
        "metal_example_matmul_multicore_reuse",
        "Test Passed",
        report=r"PCC = [\d.]+",
        tier="full",
        timeout=3600,
        note="640^3, reuse-blocked; the heaviest program the gate runs",
    ),
]

# Programs deliberately not run, and why. Each is a *reasoned* exclusion, not a
# known failure being hidden — a tt-sim bug belongs in EXPECTED as a FAIL, not
# here.
EXCLUDED = {
    "distributed/distributed_buffer_rw": "needs an 8-device mesh (MeshShape([2,4]))",
    "distributed/distributed_eltwise_add": "needs an 8-device mesh (MeshShape([2,4]))",
    "distributed/distributed_program_dispatch": "needs an 8-device mesh (MeshShape([2,4]))",
    "distributed/distributed_trace_and_events": "needs an 8-device mesh (MeshShape([2,4]))",
    "matmul_multicore_reuse_mcast": (
        "2048x1024x512 = 32768 tile-matmuls, 4x matmul_multi_core. Throughput, "
        "not correctness: no [DEADLOCK] fires and the run is still progressing "
        "when killed. Excluded until tt-sim is fast enough to finish it."
    ),
}

# The recorded expected verdict, per (arch, program). Anything not listed is
# expected to PASS. Recorded at BASELINE_TREE on BASELINE_DATE; a difference in
# either direction fails the gate.
EXPECTED = {}


def expected(arch, prog):
    return EXPECTED.get((arch, prog.name), "PASS")


def skip_reason():
    home = _tt_metal_root()
    if not home:
        return "TT_METAL_RUNTIME_ROOT / TT_METAL_HOME not set"
    if _build_dir(home) is None:
        return f"no tt-metal build under {home}"
    if not (Path(home) / "build" / "programming_examples").is_dir():
        return (
            f"no build/programming_examples under {home} (upstream examples not built)"
        )
    return None


def _env(arch, home, translated=False):
    env = dict(os.environ)
    env["TT_METAL_RUNTIME_ROOT"] = str(home)
    env["TT_METAL_HOME"] = str(home)
    env["TT_METAL_SIMULATOR"] = str(REPO / "driver" / arch)
    env["LD_LIBRARY_PATH"] = f"{_build_dir(home)}/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    env["TT_SIM_RUN_TAG"] = _run_tag(RUN_LABEL)
    # The whole point of the 2026-08-12 rerun: set NO grid variable. Setting
    # either of these at all pins the worker pool and switches off on-demand
    # materialisation, so an inherited one would silently change what is tested.
    #
    # The cluster-descriptor path is dropped for the same reason and it matters
    # more: it decides which *coordinate convention* the host puts on the wire,
    # so an inherited one would change what is being tested without saying so.
    # ``--translated`` sets it deliberately, per arch, from this repo.
    for var in (
        "TT_SIM_TENSIX_COORDS",
        "TT_SIM_TENSIX_CORES",
        "TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE",
        "TT_METAL_MOCK_CLUSTER_DESC_PATH",
        "TT_SIM_NOC_TRANSLATION",
    ):
        env.pop(var, None)
    if translated:
        # The server inherits this through UMD's uv_spawn and derives its own
        # mode from it, so one variable configures both ends. See
        # ``tt_sim/network/noc_translation.py``.
        env["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(
            REPO / "driver" / arch / "cluster_descriptor.yaml"
        )
    env.setdefault("TT_LOGGER_LEVEL", "error")
    return env


def run_one(arch, prog, home, translated=False):
    """Run one program on one arch; return (verdict, seconds, detail)."""
    bin_dir = Path(home) / "build" / "programming_examples"
    exe = bin_dir / prog.binary
    if not exe.exists():
        return "MISSING", 0.0, f"{exe} not built"
    _kill_own_servers(RUN_LABEL)
    started = time.time()
    try:
        proc = subprocess.run(
            [str(exe), *prog.args],
            cwd=bin_dir,
            env=_env(arch, home, translated),
            timeout=prog.timeout,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - started, f"no exit within {prog.timeout}s"
    finally:
        _kill_own_servers(RUN_LABEL)
    secs = time.time() - started
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        m = HINT.search(out)
        return "FAIL", secs, m.group(0) if m else f"exit {proc.returncode}"
    if prog.success not in out:
        return "FAIL", secs, f"no success line {prog.success!r}"
    m = FORBIDDEN.search(out)
    if m:
        return "FAIL", secs, f"forbidden output: {m.group(0)}"
    if prog.value_check is not None:
        why = prog.value_check(out)
        if why:
            return "FAIL", secs, f"value check: {why}"
    noted = prog.report.search(out) if prog.report else None
    return "PASS", secs, noted.group(0) if noted else ""


def select(tier, filters):
    progs = [p for p in PROGRAMS if tier == "full" or p.tier == "fast"]
    if filters:
        progs = [p for p in progs if any(f in p.name for f in filters)]
    return progs


def print_table():
    by_check = {}
    for p in PROGRAMS:
        by_check.setdefault(p.check, []).append(p)
    print(f"{'program':<34} {'tier':<6} {'check':<11} note")
    print("-" * 100)
    for p in PROGRAMS:
        print(f"{p.name:<34} {p.tier:<6} {p.check:<11} {p.note}")
    print()
    print(f"{'EXCLUDED':<34} reason")
    print("-" * 100)
    for name, why in EXCLUDED.items():
        print(f"{name:<34} {why}")
    print()
    fast = sum(1 for p in PROGRAMS if p.tier == "fast")
    print(
        f"coverage: {len(PROGRAMS)} programs run ({fast} fast, "
        f"{len(PROGRAMS) - fast} full-tier only) x {len(ARCHES)} architectures; "
        f"{len(EXCLUDED)} excluded.\n"
        f"          value-checked: {len(by_check.get('self', []))} by the program "
        f"itself, {len(by_check.get('value', []))} by this gate; "
        f"{len(by_check.get('completion', []))} completion-only "
        f"(upstream ships no self-check)."
    )
    if EXPECTED:
        print("\nnon-PASS expectations recorded:")
        for (arch, name), verdict in sorted(EXPECTED.items()):
            print(f"  {arch:<10} {name:<34} {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="upstream_sweep",
        description="Run tt-metal's upstream programming_examples on tt-sim.",
    )
    ap.add_argument("filters", nargs="*", help="only programs whose name contains this")
    ap.add_argument("--arch", default="both", choices=(*ARCHES, "both"))
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--list", action="store_true", help="print the table, run nothing")
    ap.add_argument(
        "--record",
        action="store_true",
        help="print the EXPECTED table the run would need, instead of a verdict",
    )
    ap.add_argument(
        "--translated",
        action="store_true",
        help=(
            "run with NoC coordinate translation on, by pointing the host at "
            "driver/<arch>/cluster_descriptor.yaml (the simulator inherits it "
            "and configures itself to match)"
        ),
    )
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    if args.list:
        print_table()
        return 0

    reason = skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return 0

    home = _tt_metal_root()
    arches = ARCHES if args.arch == "both" else (args.arch,)
    progs = select(args.tier, args.filters)
    print(
        f"upstream-example gate — tt-metal {BASELINE_TT_METAL} at {home}\n"
        f"{len(progs)} programs x {len(arches)} arch = {len(progs) * len(arches)} runs, "
        f"tier={args.tier}, NO grid environment variable set, "
        f"noc_translation={'on' if args.translated else 'off'}\n"
        f"expectations recorded at tree {BASELINE_TREE} ({BASELINE_DATE})"
    )
    _reap_orphans()

    results, ok = {}, True
    for arch in arches:
        print(f"\n[{arch}]")
        for prog in progs:
            verdict, secs, detail = run_one(arch, prog, home, args.translated)
            results[(arch, prog.name)] = verdict
            want = expected(arch, prog)
            mark = "ok  " if verdict == want else "BAD "
            if verdict != want:
                ok = False
            print(f"  {mark} {verdict:<8} {prog.name:<34} [{secs:5.0f}s] {detail}")
            if verdict != want:
                print(f"       ^ expected {want} at {BASELINE_TREE}")

    if args.record:
        print("\nEXPECTED = {")
        for (arch, name), verdict in sorted(results.items()):
            if verdict != "PASS":
                print(f'    ("{arch}", "{name}"): "{verdict}",')
        print("}")

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    if ok:
        print(
            "  Proves: every program above reaches its own success criterion on both\n"
            "  simulated architectures with no environment variable set. Does not prove\n"
            "  anything about cycle counts, and the completion-only rows check no value\n"
            "  (see --list)."
        )
    else:
        print(
            "  A BAD row means the verdict differs from the one recorded at "
            f"{BASELINE_TREE}.\n"
            "  A newly-FAILING program is a regression: triage it against ttsim (the\n"
            "  FUNCTIONAL oracle, never a cycle oracle) before calling it a tt-sim bug —\n"
            "  see docs/upstream-examples-status.md. A newly-PASSING one means the\n"
            "  EXPECTED record is stale; re-record it with --record."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
