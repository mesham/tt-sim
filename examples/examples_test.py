"""Build every shared example against tt-metal and run it live on the simulator.

Each example under ``examples/<name>/src`` is a real, arch-agnostic
tt-metal host program (the same binary runs on either arch depending on
``TT_METAL_SIMULATOR``) that validates its own results and exits non-zero on
mismatch. This harness is the "run it like hardware, but the simulator route"
test: it builds each example with CMake (``find_package(TT-Metalium)``) and runs
the resulting binary against tt-sim via UMD's simulation backend, asserting the
process exits 0 and prints its success line.

It is the **Wormhole** live runner: the ``EXAMPLES`` table below carries the
Wormhole physical Tensix coords, and the caller points ``TT_METAL_SIMULATOR`` at
``driver/wormhole``. The Blackhole side runs the same shared sources through
``driver/blackhole/tests/run_examples.sh`` (with its own coords), plus the
offline-replay guards in ``driver/blackhole/server/``.

Run standalone (no pytest needed)::

    python3 -m examples.examples_test

Run under pytest (if installed)::

    pytest examples/examples_test.py -v

It skips cleanly when the tt-metal build environment is absent (no
``TT_METAL_RUNTIME_ROOT``/``TT_METAL_HOME``, no exported TT-Metalium package, no
``TT_METAL_SIMULATOR``, or missing ``cmake``/``clang``), so it is safe to collect
anywhere.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
SUCCESS_LINE = "Completed successfully on the device"
BUILD_TIMEOUT = 600
# Per-example run timeout. Heavy compute / multi-tile examples are slow in the
# simulator; raise this (e.g. TT_SIM_EXAMPLE_TIMEOUT=900) to give them longer.
RUN_TIMEOUT = int(os.environ.get("TT_SIM_EXAMPLE_TIMEOUT", "260"))

# (case label, example directory, physical Tensix coords, extra environment).
# Coords are PHYSICAL NoC coords (see docs/running-tt-metal-on-the-simulator.md);
# logical (col,row) -> physical (col+1,row+1). Every example runs on the
# default single tile "1-1" except ``nine`` and ``pipestall``, which bridge a CB
# across two tiles (logical (0,0)+(1,0) -> physical 1-1 and 2-1).
#
# The coords are a *pin*, not a requirement: tt-sim materialises the workers a
# program uses whether or not they are named. They are given here so the ladder
# keeps covering the pinned path (which the replay guards and the cost-model
# gate depend on), and ``None`` in that field means "pin nothing", which is what
# the ``nine-on-demand`` case exists to exercise.
#
# The label is normally the directory name; it differs only where one example is
# run in more than one configuration, which so far is just ``pipestall``.
EXAMPLES = [
    ("one", "one", "1-1", {}),
    ("two", "two", "1-1", {}),
    ("three", "three", "1-1", {}),
    ("four", "four", "1-1", {}),
    ("four-fp", "four-fp", "1-1", {}),
    ("five", "five", "1-1", {}),
    ("five-fp", "five-fp", "1-1", {}),
    ("six", "six", "1-1", {}),
    ("eight", "eight", "1-1", {}),
    ("nine", "nine", "1-1,2-1", {}),
    # The same two-tile example with *nothing* pinned: the second worker has to
    # be materialised on demand, which is what a user with no environment set
    # gets. Arch-agnostic precisely because it names no coordinate.
    ("nine-on-demand", "nine", None, {}),
    ("pipestall", "pipestall", "1-1,2-1", {}),
    # Regression: a *multi-page* output CB with the producer running exactly one
    # chunk ahead of the consumer. This configuration returned chunk 1 as 64
    # zeroes until the example's CB pages were sized to a tile — `pack_tile`
    # writes a whole tile, so page 1 lived inside page 0's pack footprint and the
    # pack of chunk N destroyed the unread chunk N-1. Nothing else in the tree
    # has an output CB deeper than one page, so without this entry the shape is
    # untested. See the note at the top of examples/pipestall/src/pipestall.cpp
    # and the oracle diff in optests/packspill.
    (
        "pipestall-2page",
        "pipestall",
        "1-1,2-1",
        {
            "PIPESTALL_OUT_DEPTH": "2",
            "PIPESTALL_CREDITS": "1",
            "PIPESTALL_DELAY": "1000",
        },
    ),
    ("loopback", "loopback", "1-1", {}),
]


def _tt_metal_root():
    # TT_METAL_RUNTIME_ROOT is the current tt-metal variable; TT_METAL_HOME is
    # the older name, kept as a fallback.
    return os.environ.get("TT_METAL_RUNTIME_ROOT") or os.environ.get("TT_METAL_HOME")


def _build_dir(home):
    """tt-metal build tree that exports the TT-Metalium CMake package."""
    for name in ("build", "build_Release"):
        cfg = (
            Path(home)
            / name
            / "lib"
            / "cmake"
            / "tt-metalium"
            / "tt-metalium-config.cmake"
        )
        if cfg.exists():
            return Path(home) / name
    return None


def skip_reason():
    """Why the suite can't run here, or None if it can."""
    home = _tt_metal_root()
    if not home:
        return "TT_METAL_RUNTIME_ROOT / TT_METAL_HOME not set"
    if _build_dir(home) is None:
        return f"no tt-metal build exporting the TT-Metalium package under {home}"
    if not os.environ.get("TT_METAL_SIMULATOR"):
        return "TT_METAL_SIMULATOR not set (no simulator target)"
    for tool in ("cmake", "clang++-17"):
        if shutil.which(tool) is None:
            return f"{tool} not found in PATH"
    return None


# --- simulator process cleanup ------------------------------------------------
# The Python twin of driver/sim_procs.sh, which carries the full rationale. In
# short: UMD spawns the server detached, so it can't be cleaned up by pid; this
# runner stamps a run tag into its command line (run.sh forwards
# TT_SIM_RUN_TAG as --run-tag) and kills only servers carrying that tag, plus
# servers tagged by a run whose owner process is gone (orphans from a crashed
# earlier run). It used to `pkill -9 -f 'driver\.(wormhole|blackhole)\.server'`,
# which killed every simulator on the machine — including ones a concurrent run
# in another terminal depended on, silently corrupting that run's results.
_TAG_PREFIX = "ttsim-run."
# Trailing space anchors the module name: unanchored, the marker also matches
# `-m driver.<arch>.server.<name>_replay_test`, so a live run would kill any
# offline replay guard running beside it.
_SERVER_MARKERS = ("-m driver.wormhole.server ", "-m driver.blackhole.server ")
_orphans_reaped = False


def _stat_field(pid, index):
    """Field ``index`` of /proc/<pid>/stat counting from the state field.

    Splitting after the ``") "`` that ends comm survives a command name with
    spaces or parens. Real field number minus 2 (state is field 3).
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat.rpartition(") ")[2].split()
    return fields[index - 1] if len(fields) >= index else None


def _starttime(pid):
    return _stat_field(pid, 20)


def _cmdline(pid):
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", "replace")
        )
    except OSError:
        return None


def _ancestry():
    """This process and every process above it — never a cleanup target."""
    seen, pid = set(), os.getpid()
    while pid and pid > 1 and pid not in seen and len(seen) < 64:
        seen.add(pid)
        parent = _stat_field(pid, 2)
        pid = int(parent) if parent and parent.isdigit() else None
    return seen


def _server_pids(match=None):
    """pids of real tt-sim server processes, optionally filtered by a substring."""
    skip = _ancestry()
    out = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in skip:
            continue
        cmd = _cmdline(entry.name)
        if not cmd or not any(m in cmd for m in _SERVER_MARKERS):
            continue
        # A server is a python running the server module, nothing else — a
        # *shell* whose argv merely quotes that module name is not a server.
        if not os.path.basename(cmd.split(" ", 1)[0]).startswith("python"):
            continue
        if match is not None and match not in cmd:
            continue
        out.append(int(entry.name))
    return out


def _kill(pid):
    try:
        os.kill(pid, 9)
    except OSError:
        pass


def _run_tag():
    """This runner's tag: ttsim-run.<label>.<owner pid>.<owner start time>.

    The start time makes "did the run that started this server finish?" exact
    instead of a pid-reuse guess.
    """
    return f"{_TAG_PREFIX}examples_test.{os.getpid()}.{_starttime(os.getpid())}"


def _reap_orphans():
    """Kill tagged servers whose owning run is gone; leave every other alone.

    ``TT_SIM_KILL_ALL_SERVERS=1`` opts back in to the old sledgehammer — every
    simulator on the machine, tagged or not — for clearing up after manual runs,
    which carry no tag and are otherwise never reaped.
    """
    if os.environ.get("TT_SIM_KILL_ALL_SERVERS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        for pid in _server_pids():
            _kill(pid)
        return
    for pid in _server_pids(f"--run-tag {_TAG_PREFIX}"):
        cmd = _cmdline(pid) or ""
        tag = cmd.partition("--run-tag ")[2].split(" ")[0]
        parts = tag[len(_TAG_PREFIX) :].split(".")
        if len(parts) != 3 or not parts[1].isdigit():
            continue  # unparseable: leave it alone rather than kill a stranger's
        if _starttime(parts[1]) == parts[2]:
            continue  # owner still running — a concurrent run's server
        _kill(pid)


def _kill_own_servers():
    for pid in _server_pids(f"--run-tag {_run_tag()}"):
        _kill(pid)


def _run_env(home, coords, extra=None):
    env = dict(os.environ)
    # Current tt-metal reads TT_METAL_RUNTIME_ROOT; also set TT_METAL_HOME for
    # older builds.
    env["TT_METAL_RUNTIME_ROOT"] = str(home)
    env["TT_METAL_HOME"] = str(home)
    env["LD_LIBRARY_PATH"] = f"{_build_dir(home)}/lib:" + env.get("LD_LIBRARY_PATH", "")
    # Pin exactly the physical tiles this example launches on. The two knobs are
    # mutually exclusive, so drop any inherited count. ``coords=None`` pins
    # nothing, leaving tt-sim to materialise workers on demand — the default a
    # user gets, and worth exercising live rather than only in unit tests.
    env.pop("TT_SIM_TENSIX_CORES", None)
    if coords is None:
        env.pop("TT_SIM_TENSIX_COORDS", None)
    else:
        env["TT_SIM_TENSIX_COORDS"] = coords
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    # Marker run.sh stamps into the server's command line so cleanup can find
    # exactly the servers this runner caused to start.
    env["TT_SIM_RUN_TAG"] = _run_tag()
    # Per-case knobs win over anything inherited, so a stray PIPESTALL_* in the
    # caller's environment cannot silently change which case is being run.
    env.update(extra or {})
    return env


def build_example(name):
    """CMake-configure and build one example; return the binary path."""
    src = EXAMPLES_DIR / name / "src"
    for cmd in (
        ["cmake", "-B", "build", "-S", ".", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", "build", "-j4"],
    ):
        subprocess.run(
            cmd,
            cwd=src,
            check=True,
            timeout=BUILD_TIMEOUT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    exe = src / "build" / name
    if not exe.exists():
        raise RuntimeError(f"{name}: build produced no binary at {exe}")
    return exe


def run_example(name, coords, extra_env=None):
    """Build and run one example against the simulator; return the CompletedProcess."""
    home = _tt_metal_root()
    src = EXAMPLES_DIR / name / "src"
    exe = build_example(name)
    global _orphans_reaped
    if not _orphans_reaped:
        # Once per process: clear servers left behind by an earlier run of this
        # runner that crashed or timed out before its own cleanup.
        _reap_orphans()
        _orphans_reaped = True
    _kill_own_servers()  # a leftover from an earlier example in this run
    try:
        return subprocess.run(
            [str(exe)],
            cwd=src,
            env=_run_env(home, coords, extra_env),
            timeout=RUN_TIMEOUT,
            capture_output=True,
            text=True,
        )
    finally:
        _kill_own_servers()


def _failure_hint(proc):
    blob = (proc.stdout or "") + (proc.stderr or "")
    m = re.search(
        r"can not handle instruction '[^']*'|std::bad_alloc|Segmentation|Failure on the device[^\n]*",
        blob,
    )
    return m.group(0) if m else f"exit {proc.returncode}"


# --- pytest entry point (only defined when pytest is installed) ---------------
try:
    import pytest

    @pytest.mark.parametrize(
        "label,name,coords,extra_env", EXAMPLES, ids=[e[0] for e in EXAMPLES]
    )
    def test_example(label, name, coords, extra_env):
        reason = skip_reason()
        if reason:
            pytest.skip(reason)
        proc = run_example(name, coords, extra_env)
        assert proc.returncode == 0, (
            f"{label} exited {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert SUCCESS_LINE in proc.stdout, (
            f"{label} missing success line\n{proc.stdout[-2000:]}"
        )

except ImportError:
    pass


def _main():
    reason = skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return 0
    failures = []
    for label, name, coords, extra_env in EXAMPLES:
        try:
            proc = run_example(name, coords, extra_env)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT  {label}")
            failures.append(label)
            continue
        except Exception as exc:  # noqa: BLE001 - report any build/run error per example
            print(f"FAIL     {label} :: {exc}")
            failures.append(label)
            continue
        if proc.returncode == 0 and SUCCESS_LINE in proc.stdout:
            print(f"PASS     {label}")
        else:
            print(f"FAIL     {label} :: {_failure_hint(proc)}")
            failures.append(label)
    print(f"\n{len(EXAMPLES) - len(failures)}/{len(EXAMPLES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
