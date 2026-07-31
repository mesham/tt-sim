#!/usr/bin/env bash
# Run the tt-metal example programs (built under driver/examples/<name>/src)
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
# On failure, the run is retried with TT_SIM_RECORD to capture a wire trace under
# /tmp/bh_<name>.trace — send that over and it can be replayed + debugged offline.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TESTS="$REPO/driver/examples"

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_SIMULATOR="$REPO/driver/blackhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TIMEOUT="${TT_SIM_EXAMPLE_TIMEOUT:-300}"
SUCCESS="Completed successfully on the device"

# example : Blackhole TT_SIM_TENSIX_COORDS (WH (x,1) -> BH (x,2); "nine" is 2-tile)
declare -A COORDS=(
  [one]="1-2"   [two]="1-2"     [three]="1-2"  [four]="1-2"  [four-fp]="1-2"
  [five]="1-2"  [five-fp]="1-2" [six]="1-2"    [eight]="1-2" [loopback]="1-2"
  [nine]="1-2,2-2"
)
ORDER=(one two three four four-fp five five-fp six eight nine loopback)

pass=0; fail=0; failed=()
for name in "${ORDER[@]}"; do
  src="$TESTS/$name/src"
  bin="$src/build/$name"
  if [ ! -x "$bin" ]; then echo "SKIP  $name (not built)"; continue; fi
  pkill -9 -f 'driver.blackhole.server' 2>/dev/null; sleep 0.5
  log="/tmp/bh_$name.out"
  # Run from the example's own src/ dir: the host programs pass kernel paths to
  # CreateKernel relative to the CWD ("kernels/dataflow/read_kernel.cpp"), so a
  # wrong CWD aborts host-side in KernelSource before the device ever runs.
  ( cd "$src" && TT_SIM_TENSIX_COORDS="${COORDS[$name]}" timeout "$TIMEOUT" "./build/$name" ) >"$log" 2>&1
  if grep -q "$SUCCESS" "$log"; then
    echo "PASS  $name"; pass=$((pass+1))
  else
    echo "FAIL  $name  (coords=${COORDS[$name]}, full log: $log)"
    fail=$((fail+1)); failed+=("$name")
    # Show the most informative lines: any tt-sim NotImplementedError / assert,
    # else the last few lines before the abort banner.
    grep -iE "NotImplementedError|AssertionError|Error:|not.*supported|does not match|mismatch|IndexError" "$log" | head -3 | sed 's/^/      > /'
    echo "      ...tail:"; grep -vE "End of error message|Backtrace|^\s*#[0-9]" "$log" | tail -6 | sed 's/^/      | /'
  fi
done
pkill -9 -f 'driver.blackhole.server' 2>/dev/null

echo "----"
echo "Blackhole examples: $pass passed, $fail failed"
if [ "$fail" -gt 0 ]; then
  echo "To capture a trace of a failing one for offline debugging:"
  for n in "${failed[@]}"; do
    echo "  ( cd $TESTS/$n/src && TT_SIM_TENSIX_COORDS=${COORDS[$n]} TT_SIM_RECORD=/tmp/bh_$n.trace ./build/$n )"
  done
fi
