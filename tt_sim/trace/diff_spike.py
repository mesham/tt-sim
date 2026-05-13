"""Diff a tt-sim Spike-style commitlog against a Spike-produced one.

Usage:

    python3 -m tt_sim.trace.diff_spike <ttsim.commitlog> <spike.commitlog>

Walks both files line-by-line and reports the first divergence with
five lines of context on each side. The format is byte-exact compatible
so a clean run is just ``files match``; any divergence is either a
tt-sim correctness bug or a Spike-versus-spec discrepancy — both are
valuable to surface.

Caveats:

- Designed for a pure-RV32IM ELF that runs identically under both
  simulators. tt-metal firmware kernels touch Tensix / NoC MMIO that
  Spike has no concept of, so the diff would diverge on the first such
  access.
- The Spike side should be invoked with ``spike --log-commits <elf>``
  on a single-hart configuration (default). tt-sim's per-unit
  commitlog files all use ``core   0:`` so any tt-sim unit can be
  compared against a hart-0 Spike run.
"""

import argparse
import sys
from pathlib import Path

CONTEXT_LINES = 5


def diff(ttsim_path: Path, spike_path: Path) -> int:
    with ttsim_path.open() as f_t, spike_path.open() as f_s:
        ttsim_lines = f_t.readlines()
        spike_lines = f_s.readlines()

    n = min(len(ttsim_lines), len(spike_lines))
    for i in range(n):
        if ttsim_lines[i] != spike_lines[i]:
            _print_divergence(ttsim_lines, spike_lines, i)
            return 1

    if len(ttsim_lines) != len(spike_lines):
        print(
            f"length mismatch: tt-sim has {len(ttsim_lines)} lines, "
            f"spike has {len(spike_lines)} lines (matched first {n})",
            file=sys.stderr,
        )
        return 1

    print(f"files match ({n} lines)")
    return 0


def _print_divergence(ttsim_lines, spike_lines, idx):
    start = max(0, idx - CONTEXT_LINES)
    print(f"first divergence at line {idx + 1}:", file=sys.stderr)
    print("--- context (matching) ---", file=sys.stderr)
    for j in range(start, idx):
        sys.stderr.write(f"  {j + 1:6d}  {ttsim_lines[j]}")
    print("--- divergence ---", file=sys.stderr)
    sys.stderr.write(f"tt-sim {idx + 1:6d}: {ttsim_lines[idx]}")
    sys.stderr.write(f"spike  {idx + 1:6d}: {spike_lines[idx]}")
    print("--- following (tt-sim) ---", file=sys.stderr)
    end = min(len(ttsim_lines), idx + 1 + CONTEXT_LINES)
    for j in range(idx + 1, end):
        sys.stderr.write(f"  {j + 1:6d}  {ttsim_lines[j]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ttsim", type=Path, help="tt-sim commitlog file")
    parser.add_argument("spike", type=Path, help="spike --log-commits output")
    args = parser.parse_args(argv)
    return diff(args.ttsim, args.spike)


if __name__ == "__main__":
    sys.exit(main())
