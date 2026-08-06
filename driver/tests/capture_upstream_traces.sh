#!/usr/bin/env bash
# Record wire traces for the tt-metal **upstream** programming_examples that
# back offline replay guards, into driver/<arch>/server/traces/<name>.trace.
#
# The in-tree examples/ tree has its own capture scripts per arch
# (driver/wormhole/tests/capture_traces.sh, driver/blackhole/tests/run_examples.sh
# --record); those build from $REPO/examples. The programs here are different:
# they live in the tt-metal checkout and are already built there, so all this
# script does is run the right binary with the right worker coordinates and
# TT_SIM_RECORD pointed at the committed trace.
#
# Which guard replays which trace (see docs/upstream-examples-status.md for why
# these four and not others):
#
#   noc_tile_transfer  driver/{wormhole,blackhole}/server/noc_tile_transfer_replay_test.py
#   vecadd_sharding    driver/blackhole/server/vecadd_sharding_replay_test.py
#   pad_multi_core     driver/blackhole/server/pad_multi_core_replay_test.py
#   shard_data_rm      driver/blackhole/server/shard_data_rm_replay_test.py
#
# Run in a NORMAL shell (not a sandbox that kills the spawned sim server):
#   TT_METAL_HOME=/path/to/tt-metal ./driver/tests/capture_upstream_traces.sh [--arch blackhole] [name...]
#
# The guards hardcode the addresses their workload's buffers landed at, so a
# recapture that moves an allocation will fail the guard rather than silently
# check the wrong bytes — recapture only when the trace is genuinely stale (a
# tt-metal bump), never to make a test pass.
#
# Cleanup only touches the sim servers this run started (they carry its
# TT_SIM_RUN_TAG) plus orphans left by an earlier run of this script whose
# owner is gone, so a concurrent run in another terminal is never disturbed.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../sim_procs.sh
. "$REPO/driver/sim_procs.sh"

ARCH=blackhole
SELECT=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    -h|--help) sed -n '2,31p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) SELECT+=("$1"); shift ;;
  esac
done
case "$ARCH" in
  blackhole|bh) ARCH=blackhole ;;
  wormhole|wh)  ARCH=wormhole ;;
  *) echo "unknown --arch $ARCH (want blackhole|wormhole)" >&2; exit 2 ;;
esac

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_SIMULATOR="$REPO/driver/$ARCH"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

BIN_DIR="$TT_METAL_HOME/build/programming_examples"
TRACES="$REPO/driver/$ARCH/server/traces"
TIMEOUT="${TT_SIM_EXAMPLE_TIMEOUT:-900}"

# name : upstream binary
declare -A BIN=(
  [noc_tile_transfer]=metal_example_noc_tile_transfer
  [vecadd_sharding]=metal_example_vecadd_sharding
  [pad_multi_core]=metal_example_pad_multi_core
  [shard_data_rm]=metal_example_shard_data_rm
)
# Worker coords. Wormhole logical (x,y) -> physical (x+1,y+1); Blackhole
# (x+1,y+2). noc_tile_transfer uses logical (0,0)+(0,1); vecadd_sharding
# height-shards over (0..3,0); pad_multi_core and shard_data_rm over (0,0..3).
declare -A COORDS_wormhole=(
  [noc_tile_transfer]="1-1,1-2"
  [vecadd_sharding]="1-1,2-1,3-1,4-1"
  [pad_multi_core]="1-1,1-2,1-3,1-4"
  [shard_data_rm]="1-1,1-2,1-3,1-4"
)
declare -A COORDS_blackhole=(
  [noc_tile_transfer]="1-2,1-3"
  [vecadd_sharding]="1-2,2-2,3-2,4-2"
  [pad_multi_core]="1-2,1-3,1-4,1-5"
  [shard_data_rm]="1-2,1-3,1-4,1-5"
)
# The line that means the run was good. pad_multi_core has no self-check of any
# kind (its guard is what pins the values); the others do.
declare -A SUCCESS=(
  [noc_tile_transfer]="Result = 14 : Expected = 14"
  [vecadd_sharding]="All results match expected values within tolerance."
  [pad_multi_core]="Padded tensor with shape"
  [shard_data_rm]="Program finished successfully."
)
ORDER=(noc_tile_transfer vecadd_sharding pad_multi_core shard_data_rm)
[ "${#SELECT[@]}" -gt 0 ] && ORDER=("${SELECT[@]}")

mkdir -p "$TRACES"
sim_procs_init capture_upstream
trap 'sim_kill_own_servers' EXIT INT TERM

pass=0; fail=0
for name in "${ORDER[@]}"; do
  bin="${BIN[$name]:-}"
  [ -n "$bin" ] || { echo "SKIP  $name (not one of: ${!BIN[*]})"; continue; }
  if [ ! -x "$BIN_DIR/$bin" ]; then
    echo "SKIP  $name ($BIN_DIR/$bin not built)"; continue
  fi
  eval "coords=\${COORDS_$ARCH[$name]}"
  sim_kill_own_servers; sleep 0.5
  log="/tmp/${ARCH}_$name.out"
  ( cd "$BIN_DIR" \
    && TT_SIM_TENSIX_COORDS="$coords" \
       TT_SIM_RECORD="$TRACES/$name.trace" \
       timeout "$TIMEOUT" "./$bin" ) >"$log" 2>&1
  if grep -qF "${SUCCESS[$name]}" "$log"; then
    echo "PASS  $name  (coords=$coords) -> $TRACES/$name.trace"; pass=$((pass+1))
  else
    echo "FAIL  $name  (coords=$coords, full log: $log)"; fail=$((fail+1))
    grep -iE "NotImplementedError|AssertionError|Error:|Mismatch|not.*supported" "$log" | head -3 | sed 's/^/      > /'
  fi
done
sim_kill_own_servers

echo "----"
echo "Upstream trace capture ($ARCH): $pass captured, $fail failed"
