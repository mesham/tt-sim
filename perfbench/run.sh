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
# The sim runners share their build/ trees with the card runners, so a tree
# poisoned here shows up there. Same check, same library.
# shellcheck source=build_provenance.sh
. "$REPO/perfbench/build_provenance.sh"
[ -d "$SRC" ] || { echo "no perfbench program at $SRC" >&2; exit 2; }

ARCH="${TT_SIM_ARCH:-blackhole}"
case "$ARCH" in
  blackhole|bh) ARCH=blackhole ;;
  wormhole|wh)  ARCH=wormhole ;;
  *) echo "unknown TT_SIM_ARCH=$ARCH (want blackhole|wormhole)" >&2; exit 2 ;;
esac

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_SIMULATOR="$REPO/driver/$ARCH"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# TT_SIM_TENSIX_COORDS is deliberately NOT defaulted here. Setting it -- even to
# the same single worker the server would have built anyway -- is what tt-sim
# reads as "the user PINNED this pool", and a pinned pool switches off on-demand
# materialisation (driver/<arch>/server/__main__.py, `pinned`). This script used
# to export the arch's default coord, which silently made every perfbench run
# single-tile: a multi-core plan then died on the first kernel launch outside the
# pool. Left unset, the server builds the same default worker and materialises
# the rest as the program asks for them. Pass it through if the caller means it.
VENV="${TT_SIM_VENV:-$REPO/../venv}"
[ -x "$VENV/bin/python3" ] && export PATH="$VENV/bin:$PATH"

cd "$SRC" || exit 2
if [ ! -x build/"$NAME" ]; then
  bp_require_build "$SRC"
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
  cmake --build build -j >/dev/null || exit 1
  bp_record_build "$SRC"
else
  bp_require_build "$SRC" skip-build
fi

# Tag the server this run spawns, and clear tagged servers left by runs that
# have since died. This script cannot clean up after *itself* — it execs the
# benchmark, so no trap of ours survives — but the tag keeps a server it leaves
# behind recoverable by the next run of any of the test scripts, which is what
# the old machine-wide `pkill` in those scripts used to do implicitly.
#
# Under `driver/sim_procs.sh run <label> -- perfbench/run.sh ...` the inherited
# tag is kept rather than replaced by `perfbench`, which is what lets that
# wrapper reap the server this run leaves behind. See driver/sim_procs.sh.
# shellcheck source=../driver/sim_procs.sh
. "$REPO/driver/sim_procs.sh"
sim_procs_init perfbench

exec ./build/"$NAME" "${ARGS[@]}"
