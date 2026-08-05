#!/usr/bin/env bash
# Build and run a perfbench program against the tt-sim simulator.
#
# This is the SIMULATOR side only. On real hardware you do not need this script
# and you must not use it: unset TT_METAL_SIMULATOR and run ./build/tensixbench
# directly, exactly as perfbench/tensixbench/README.md describes. The point of
# the exercise is that it is the same binary either way.
#
# Run in a NORMAL shell (not a sandbox that kills the spawned sim server):
#   TT_METAL_HOME=/path/to/tt-metal ./perfbench/run.sh [name] [-- prog args...]
#
# Environment:
#   TT_METAL_HOME / TT_METAL_RUNTIME_ROOT  built tt-metal checkout (required)
#   TT_SIM_ARCH        blackhole (default) | wormhole
#   TT_SIM_VENV        venv holding tt_sim's deps (default: <repo>/../venv)
#   TT_SIM_COST_MODEL  set to 1 to run the simulator with the cost model on

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="tensixbench"
ARGS=()
seen_sep=0
for arg in "$@"; do
  if [ "$seen_sep" = 1 ]; then ARGS+=("$arg"); continue; fi
  case "$arg" in
    --) seen_sep=1 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) NAME="$arg" ;;
  esac
done

SRC="$REPO/perfbench/$NAME/src"
[ -d "$SRC" ] || { echo "no perfbench program at $SRC" >&2; exit 2; }

ARCH="${TT_SIM_ARCH:-blackhole}"
case "$ARCH" in
  blackhole|bh) ARCH=blackhole; COORDS=1-2 ;;
  wormhole|wh)  ARCH=wormhole;  COORDS=1-1 ;;
  *) echo "unknown TT_SIM_ARCH=$ARCH (want blackhole|wormhole)" >&2; exit 2 ;;
esac

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_SIMULATOR="$REPO/driver/$ARCH"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-$COORDS}"
VENV="${TT_SIM_VENV:-$REPO/../venv}"
[ -x "$VENV/bin/python3" ] && export PATH="$VENV/bin:$PATH"

cd "$SRC" || exit 2
if [ ! -x build/"$NAME" ]; then
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
  cmake --build build -j >/dev/null || exit 1
fi

# Tag the server this run spawns, and clear tagged servers left by runs that
# have since died. This script cannot clean up after *itself* — it execs the
# benchmark, so no trap of ours survives — but the tag keeps a server it leaves
# behind recoverable by the next run of any of the test scripts, which is what
# the old machine-wide `pkill` in those scripts used to do implicitly. See
# driver/sim_procs.sh.
# shellcheck source=../driver/sim_procs.sh
. "$REPO/driver/sim_procs.sh"
sim_procs_init perfbench

exec ./build/"$NAME" "${ARGS[@]}"
