"""Compare two device state dumps and report the first divergence.

Usage:

    python3 -m tt_sim.trace.diff_state <a.json> <b.json>

Designed for catching nondeterminism / regressions: capture a state
dump on a known-good run, capture another on a candidate commit, diff.
A clean run prints ``state matches``; any divergence pinpoints the
first field that differs and exits non-zero.

The cross-simulator workflow (driving tt-sim and ``libttsim.so`` on
the same kernel) is documented in ROADMAP §H Phase 7 but the
``libttsim.so`` side isn't wired up here — generate two dumps via
your own orchestration and feed both to this tool.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _walk_diff(a: Any, b: Any, path: str) -> str | None:
    """Recursively compare two JSON-shaped values. Return the first
    divergence as a human-readable string, or None if they match."""
    if type(a) is not type(b):
        return f"{path}: type mismatch ({type(a).__name__} vs {type(b).__name__})"
    if isinstance(a, dict):
        keys_a = set(a)
        keys_b = set(b)
        for k in sorted(keys_a - keys_b):
            return f"{path}: key '{k}' present in A, missing in B"
        for k in sorted(keys_b - keys_a):
            return f"{path}: key '{k}' present in B, missing in A"
        for k in sorted(keys_a):
            sub = _walk_diff(a[k], b[k], f"{path}.{k}" if path else k)
            if sub is not None:
                return sub
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: list length differs ({len(a)} vs {len(b)})"
        for i, (va, vb) in enumerate(zip(a, b)):
            sub = _walk_diff(va, vb, f"{path}[{i}]")
            if sub is not None:
                return sub
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def diff(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    msg = _walk_diff(a, b, "")
    if msg is None:
        print("state matches")
        return 0
    print(f"first divergence:\n  {msg}", file=sys.stderr)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    return diff(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
