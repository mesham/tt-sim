#!/usr/bin/env bash
# Run the tt-metal example programs (built under examples/<name>/src)
# against the Blackhole simulator and report pass/fail.
#
# The example *host* programs are architecture-agnostic — the same binary runs on
# Wormhole or Blackhole depending on which simulator TT_METAL_SIMULATOR points at
# (kernels JIT for the target arch). Only the worker coordinate convention
# differs: logical (0,0) -> physical (1,2) on Blackhole (vs (1,1) on Wormhole),
# so TT_SIM_TENSIX_COORDS is remapped below.
#
# Run in a NORMAL shell (not a sandbox that kills the spawned sim server):
#   TT_METAL_HOME=/path/to/tt-metal ./driver/blackhole/tests/run_examples.sh
#
# Options:
#   --record        record each example's wire trace to
#                   driver/blackhole/server/traces/<name>.trace (the committed
#                   set the *_replay_test.py guards replay), building any example
#                   that isn't built yet. Without it, unbuilt examples are
#                   skipped and no trace is written.
#   <name>...       run only the named case(s) instead of the whole suite.
#
# Most cases are named after their example directory. A case may instead be an
# example run in a second configuration (see DIRS/ENVS below) — currently just
# `pipestall-2page`, the multi-page output CB regression. Those are never
# trace-recorded, since the guards replay one trace per example.
#
# On failure the summary prints a per-case command to capture a trace under
# /tmp/bh_<name>.trace for offline debugging.
#
# Cleanup only touches the sim servers this run started (they carry its
# TT_SIM_RUN_TAG) plus orphans left by an earlier run of this script whose
# owner is gone, so a concurrent run in another terminal is never disturbed.
# Set TT_SIM_KILL_ALL_SERVERS=1 to instead kill every tt-sim server on the
# machine at startup — the way to clear up after manual runs, which carry no
# tag.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TESTS="$REPO/examples"
TRACES="$REPO/driver/blackhole/server/traces"
# shellcheck source=../../sim_procs.sh
. "$REPO/driver/sim_procs.sh"

RECORD=0
SELECT=()
for arg in "$@"; do
  case "$arg" in
    --record) RECORD=1 ;;
    -h|--help) sed -n '2,35p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) SELECT+=("$arg") ;;
  esac
done

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_SIMULATOR="$REPO/driver/blackhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TIMEOUT="${TT_SIM_EXAMPLE_TIMEOUT:-300}"
SUCCESS="Completed successfully on the device"

# case : Blackhole TT_SIM_TENSIX_COORDS (WH (x,1) -> BH (x,2); "nine" and
# "pipestall" are 2-tile)
declare -A COORDS=(
  [one]="1-2"   [two]="1-2"     [three]="1-2"  [four]="1-2"  [four-fp]="1-2"
  [five]="1-2"  [five-fp]="1-2" [six]="1-2"    [eight]="1-2" [loopback]="1-2"
  [banks]="1-2"
  [nine]="1-2,2-2" [pipestall]="1-2,2-2" [pipestall-2page]="1-2,2-2"
)
# A case whose name is not an example directory: which directory it runs, and
# the environment that configures it. Only `pipestall-2page` so far — a
# regression run of `pipestall` with a *multi-page* output CB and the producer
# exactly one chunk ahead, the shape that returned chunk 1 as 64 zeroes until
# the example's CB pages were sized to a tile (see examples/pipestall/src and
# optests/packspill). Extra cases are never trace-recorded: they have no replay
# guard of their own and would clobber the canonical example's trace.
declare -A DIRS=([pipestall-2page]="pipestall")
declare -A ENVS=([pipestall-2page]="PIPESTALL_OUT_DEPTH=2 PIPESTALL_CREDITS=1 PIPESTALL_DELAY=1000")
ORDER=(one two three four four-fp five five-fp six eight nine pipestall pipestall-2page loopback banks)
[ "${#SELECT[@]}" -gt 0 ] && ORDER=("${SELECT[@]}")
[ "$RECORD" -eq 1 ] && mkdir -p "$TRACES"

sim_procs_init run_examples
trap 'sim_kill_own_servers' EXIT INT TERM

pass=0; fail=0; failed=()
for name in "${ORDER[@]}"; do
  dir="${DIRS[$name]:-$name}"
  src="$TESTS/$dir/src"
  bin="$src/build/$dir"
  if [ ! -x "$bin" ]; then
    # --record is a recapture flow, so build a missing example; otherwise skip.
    if [ "$RECORD" -eq 1 ]; then
      echo "BUILD $name"
      ( cd "$src" \
        && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >"/tmp/bh_${name}_build.log" 2>&1 \
        && cmake --build build -j4 >>"/tmp/bh_${name}_build.log" 2>&1 ) \
        || { echo "FAIL  $name (build; see /tmp/bh_${name}_build.log)"; fail=$((fail+1)); failed+=("$name"); continue; }
    else
      echo "SKIP  $name (not built)"; continue
    fi
  fi
  # Clear a server this run leaked earlier (e.g. the previous case timed out).
  sim_kill_own_servers; sleep 0.5
  log="/tmp/bh_$name.out"
  # Run from the example's own src/ dir: the host programs pass kernel paths to
  # CreateKernel relative to the CWD ("kernels/dataflow/read_kernel.cpp"), so a
  # wrong CWD aborts host-side in KernelSource before the device ever runs.
  ( cd "$src"
    export TT_SIM_TENSIX_COORDS="${COORDS[$name]}"
    # shellcheck disable=SC2163
    for kv in ${ENVS[$name]:-}; do export "$kv"; done
    [ "$RECORD" -eq 1 ] && [ -z "${DIRS[$name]:-}" ] && export TT_SIM_RECORD="$TRACES/$name.trace"
    timeout "$TIMEOUT" "./build/$dir" ) >"$log" 2>&1
  if grep -q "$SUCCESS" "$log"; then
    if [ "$RECORD" -eq 1 ] && [ -z "${DIRS[$name]:-}" ]; then
      echo "PASS  $name  -> $TRACES/$name.trace"
    else
      echo "PASS  $name"
    fi
    pass=$((pass+1))
  else
    echo "FAIL  $name  (coords=${COORDS[$name]}, full log: $log)"
    fail=$((fail+1)); failed+=("$name")
    # Show the most informative lines: any tt-sim NotImplementedError / assert,
    # else the last few lines before the abort banner.
    grep -iE "NotImplementedError|AssertionError|Error:|not.*supported|does not match|mismatch|IndexError" "$log" | head -3 | sed 's/^/      > /'
    echo "      ...tail:"; grep -vE "End of error message|Backtrace|^\s*#[0-9]" "$log" | tail -6 | sed 's/^/      | /'
  fi
done
sim_kill_own_servers

echo "----"
echo "Blackhole examples: $pass passed, $fail failed"
if [ "$RECORD" -eq 1 ] && [ "$fail" -eq 0 ]; then
  echo "Traces recorded under $TRACES"
fi
if [ "$fail" -gt 0 ]; then
  echo "To capture a trace of a failing one for offline debugging:"
  for n in "${failed[@]}"; do
    d="${DIRS[$n]:-$n}"
    echo "  ( cd $TESTS/$d/src && TT_SIM_TENSIX_COORDS=${COORDS[$n]} ${ENVS[$n]:-} TT_SIM_RECORD=/tmp/bh_$n.trace ./build/$d )"
  done
fi
