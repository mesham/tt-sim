#!/usr/bin/env bash
# Build + run every shared example (examples/<name>/src) against the
# WORMHOLE simulator and record each wire trace under
# driver/wormhole/server/traces/<name>.trace. Those traces back the per-example
# offline-replay guards (driver/wormhole/server/<name>_replay_test.py), the
# Wormhole counterpart of driver/blackhole/server/*_replay_test.py.
#
# The example host programs are architecture-agnostic; only the worker coord
# convention differs (logical (0,0) -> physical (1,1) on Wormhole vs (1,2) on
# Blackhole), so TT_SIM_TENSIX_COORDS is set per-example below.
#
# Run in a NORMAL shell (not a sandbox that kills the spawned sim server):
#   TT_METAL_HOME=/path/to/tt-metal ./driver/wormhole/tests/capture_traces.sh
#
# Optionally pass example names to capture a subset:
#   TT_METAL_HOME=... ./driver/wormhole/tests/capture_traces.sh six nine

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXAMPLES="$REPO/examples"
TRACES="$REPO/driver/wormhole/server/traces"

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_SIMULATOR="$REPO/driver/wormhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

TIMEOUT="${TT_SIM_EXAMPLE_TIMEOUT:-300}"
SUCCESS="Completed successfully on the device"

# example : Wormhole TT_SIM_TENSIX_COORDS (logical (x,y) -> physical (x+1,y+1);
# "nine" bridges two tiles).
declare -A COORDS=(
  [one]="1-1"   [two]="1-1"     [three]="1-1"  [four]="1-1"  [four-fp]="1-1"
  [five]="1-1"  [five-fp]="1-1" [six]="1-1"    [eight]="1-1" [loopback]="1-1"
  [nine]="1-1,2-1"
)
ORDER=(one two three four four-fp five five-fp six eight nine loopback)
[ "$#" -gt 0 ] && ORDER=("$@")

mkdir -p "$TRACES"
pass=0; fail=0; failed=()
for name in "${ORDER[@]}"; do
  src="$EXAMPLES/$name/src"
  if [ ! -d "$src" ]; then echo "SKIP  $name (no source at $src)"; continue; fi
  # Build if needed (stale build/ dirs were dropped in the examples move).
  if [ ! -x "$src/build/$name" ]; then
    echo "BUILD $name"
    ( cd "$src" \
      && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/tmp/wh_${name}_build.log 2>&1 \
      && cmake --build build -j4 >>/tmp/wh_${name}_build.log 2>&1 ) \
      || { echo "FAIL  $name (build; see /tmp/wh_${name}_build.log)"; fail=$((fail+1)); failed+=("$name"); continue; }
  fi
  pkill -9 -f 'driver.wormhole.server' 2>/dev/null; sleep 0.5
  log="/tmp/wh_$name.out"
  # Run from the example's own src/ dir: host programs pass kernel paths relative
  # to CWD, so a wrong CWD aborts host-side before the device runs.
  ( cd "$src" \
    && TT_SIM_TENSIX_COORDS="${COORDS[$name]}" \
       TT_SIM_RECORD="$TRACES/$name.trace" \
       timeout "$TIMEOUT" "./build/$name" ) >"$log" 2>&1
  if grep -q "$SUCCESS" "$log"; then
    echo "PASS  $name  -> $TRACES/$name.trace"; pass=$((pass+1))
  else
    echo "FAIL  $name  (coords=${COORDS[$name]}, full log: $log)"
    fail=$((fail+1)); failed+=("$name")
    grep -iE "NotImplementedError|AssertionError|Error:|not.*supported|mismatch|Failure on the device" "$log" | head -3 | sed 's/^/      > /'
  fi
done
pkill -9 -f 'driver.wormhole.server' 2>/dev/null

echo "----"
echo "Wormhole trace capture: $pass captured, $fail failed"
[ "$fail" -eq 0 ] && echo "All traces under $TRACES — ready to write the *_replay_test.py guards."
