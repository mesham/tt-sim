"""Build every shared example against tt-metal and run it live on the simulator.

Each example under ``driver/examples/<name>/src`` is a real, arch-agnostic
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

    python3 -m driver.examples.examples_test

Run under pytest (if installed)::

    pytest driver/examples/examples_test.py -v

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

# (example directory, physical Tensix coords the program launches on). Coords
# are PHYSICAL NoC coords (see docs/running-tt-metal-on-the-simulator.md);
# logical (col,row) -> physical (col+1,row+1). Every example runs on the
# default single tile "1-1" except ``nine``, which bridges a CB across two
# tiles (logical (0,0)+(1,0) -> physical 1-1 and 2-1). The exact coords must be
# materialised, so we pass TT_SIM_TENSIX_COORDS rather than a bare count.
EXAMPLES = [
    ("one", "1-1"),
    ("two", "1-1"),
    ("three", "1-1"),
    ("four", "1-1"),
    ("four-fp", "1-1"),
    ("five", "1-1"),
    ("five-fp", "1-1"),
    ("six", "1-1"),
    ("eight", "1-1"),
    ("nine", "1-1,2-1"),
    ("loopback", "1-1"),
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


def _reap_servers():
    # Bracket the pattern so pkill can't match this process's own command line.
    # Reap either arch's bridge server (the shared examples can target either).
    subprocess.run(
        ["pkill", "-9", "-f", "[d]river.(wormhole|blackhole).server"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _run_env(home, coords):
    env = dict(os.environ)
    # Current tt-metal reads TT_METAL_RUNTIME_ROOT; also set TT_METAL_HOME for
    # older builds.
    env["TT_METAL_RUNTIME_ROOT"] = str(home)
    env["TT_METAL_HOME"] = str(home)
    env["LD_LIBRARY_PATH"] = f"{_build_dir(home)}/lib:" + env.get("LD_LIBRARY_PATH", "")
    # Materialise exactly the physical tiles this example launches on. The two
    # knobs are mutually exclusive, so drop any inherited count.
    env.pop("TT_SIM_TENSIX_CORES", None)
    env["TT_SIM_TENSIX_COORDS"] = coords
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
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


def run_example(name, coords):
    """Build and run one example against the simulator; return the CompletedProcess."""
    home = _tt_metal_root()
    src = EXAMPLES_DIR / name / "src"
    exe = build_example(name)
    _reap_servers()
    try:
        return subprocess.run(
            [str(exe)],
            cwd=src,
            env=_run_env(home, coords),
            timeout=RUN_TIMEOUT,
            capture_output=True,
            text=True,
        )
    finally:
        _reap_servers()


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

    @pytest.mark.parametrize("name,coords", EXAMPLES, ids=[e[0] for e in EXAMPLES])
    def test_example(name, coords):
        reason = skip_reason()
        if reason:
            pytest.skip(reason)
        proc = run_example(name, coords)
        assert proc.returncode == 0, (
            f"{name} exited {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
        assert SUCCESS_LINE in proc.stdout, (
            f"{name} missing success line\n{proc.stdout[-2000:]}"
        )

except ImportError:
    pass


def _main():
    reason = skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return 0
    failures = []
    for name, coords in EXAMPLES:
        try:
            proc = run_example(name, coords)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT  {name}")
            failures.append(name)
            continue
        except Exception as exc:  # noqa: BLE001 - report any build/run error per example
            print(f"FAIL     {name} :: {exc}")
            failures.append(name)
            continue
        if proc.returncode == 0 and SUCCESS_LINE in proc.stdout:
            print(f"PASS     {name}")
        else:
            print(f"FAIL     {name} :: {_failure_hint(proc)}")
            failures.append(name)
    print(f"\n{len(EXAMPLES) - len(failures)}/{len(EXAMPLES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
