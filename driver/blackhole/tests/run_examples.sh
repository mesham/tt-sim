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
#   <name>...       run only the named example(s) instead of the whole suite.
#
# On failure the summary prints a per-example command to capture a trace under
# /tmp/bh_<name>.trace for offline debugging.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TESTS="$REPO/examples"
TRACES="$REPO/driver/blackhole/server/traces"

RECORD=0
SELECT=()
for arg in "$@"; do
  case "$arg" in
    --record) RECORD=1 ;;
    -h|--help) sed -n '2,21p' "$0" | sed 's/^# \?//'; exit 0 ;;
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

# example : Blackhole TT_SIM_TENSIX_COORDS (WH (x,1) -> BH (x,2); "nine" and
# "pipestall" are 2-tile)
declare -A COORDS=(
  [one]="1-2"   [two]="1-2"     [three]="1-2"  [four]="1-2"  [four-fp]="1-2"
  [five]="1-2"  [five-fp]="1-2" [six]="1-2"    [eight]="1-2" [loopback]="1-2"
  [nine]="1-2,2-2" [pipestall]="1-2,2-2"
)
ORDER=(one two three four four-fp five five-fp six eight nine pipestall loopback)
[ "${#SELECT[@]}" -gt 0 ] && ORDER=("${SELECT[@]}")
[ "$RECORD" -eq 1 ] && mkdir -p "$TRACES"

pass=0; fail=0; failed=()
for name in "${ORDER[@]}"; do
  src="$TESTS/$name/src"
  bin="$src/build/$name"
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
  pkill -9 -f 'driver\.blackhole\.server( |$)' 2>/dev/null; sleep 0.5
  log="/tmp/bh_$name.out"
  # Run from the example's own src/ dir: the host programs pass kernel paths to
  # CreateKernel relative to the CWD ("kernels/dataflow/read_kernel.cpp"), so a
  # wrong CWD aborts host-side in KernelSource before the device ever runs.
  ( cd "$src"
    export TT_SIM_TENSIX_COORDS="${COORDS[$name]}"
    [ "$RECORD" -eq 1 ] && export TT_SIM_RECORD="$TRACES/$name.trace"
    timeout "$TIMEOUT" "./build/$name" ) >"$log" 2>&1
  if grep -q "$SUCCESS" "$log"; then
    if [ "$RECORD" -eq 1 ]; then echo "PASS  $name  -> $TRACES/$name.trace"; else echo "PASS  $name"; fi
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
pkill -9 -f 'driver\.blackhole\.server( |$)' 2>/dev/null

echo "----"
echo "Blackhole examples: $pass passed, $fail failed"
if [ "$RECORD" -eq 1 ] && [ "$fail" -eq 0 ]; then
  echo "Traces recorded under $TRACES"
fi
if [ "$fail" -gt 0 ]; then
  echo "To capture a trace of a failing one for offline debugging:"
  for n in "${failed[@]}"; do
    echo "  ( cd $TESTS/$n/src && TT_SIM_TENSIX_COORDS=${COORDS[$n]} TT_SIM_RECORD=/tmp/bh_$n.trace ./build/$n )"
  done
fi
