#!/usr/bin/env bash
# retirebench, simulator side: run the RV zone program against tt-sim and
# collect the same artefact the card protocol produces on the other side.
#
# The two sides emit the IDENTICAL artefact -- a `retirebench-*.json` written by
# the host program -- so `tt_sim.perf.retire_attribution` parses both with one
# reader and there is no translation step in which a units mistake could hide.
#
#   TT_METAL_HOME=/path/to/tt-metal ./perfbench/retirebench/run_sim.sh \
#       --out /tmp/retirebench-sim
#
# Options:
#   --scale N       multiply every zone's repetition count (default 1)
#   --label TEXT    goes in the artefact's filename and its `label` field
#   --out DIR       output directory (default ./retirebench-sim)
#   --timeout SEC   per-run timeout (default 1800)
#   --no-cost-model run with TT_SIM_COST_MODEL unset. Exists to demonstrate what
#                   goes wrong, not to be used for a comparison: with it off,
#                   every RV load, store, multiply and divide costs one cycle and
#                   nine of the eleven measured zones collapse onto each other.
#
# BLACKHOLE ONLY, and there is no --arch. The instrument is the mcycle/minstret
# CSRs, which the WormholeB0 documentation does not describe -- the string "csr"
# does not appear in that tree at all -- so a Wormhole baby core in tt-sim has no
# CSR file and `csrr` raises NoCSRsError rather than returning a number. The host
# program refuses a non-Blackhole part before it builds its kernel or launches
# anything, and the analysis refuses a non-Blackhole artefact before it computes
# anything. Running this on Wormhole would leave an elapsed-only envelope check,
# which is exactly the kind of claim rung 4 exists to distrust.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../build_provenance.sh
. "$HERE/../build_provenance.sh"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"

SCALE=1
LABEL=sim
OUT="$PWD/retirebench-sim"
TIMEOUT=1800
COST_MODEL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --scale) SCALE="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --no-cost-model) COST_MODEL=0; shift ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_SIMULATOR="$REPO/driver/blackhole"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# The simulator server is spawned by UMD and runs whatever `python3` is first on
# PATH. The repo venv is where tt_sim's dependencies live; the system python
# lacks pyarrow and the server dies importing the trace writers.
VENV="${TT_SIM_VENV:-$REPO/../venv}"
[ -x "$VENV/bin/python3" ] && export PATH="$VENV/bin:$PATH"

# Worker (0,0) is physical (1,2) on Blackhole. Naming it up front rather than
# letting LazyTensixPool materialise it keeps the run to one tile.
export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-1-2}"

if [ "$COST_MODEL" -eq 1 ]; then
  export TT_SIM_COST_MODEL=1
else
  unset TT_SIM_COST_MODEL
  echo "WARNING: cost model off -- every RV load, store, multiply and divide" >&2
  echo "         costs one cycle, so nine of the eleven measured zones collapse" >&2
  echo "         onto each other. Do not use this for a result." >&2
fi

if [ ! -x "$SRC/build/retirebench" ]; then
  echo "== building retirebench"
  bp_require_build "$SRC"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || exit 1
  bp_record_build "$SRC"
else
  bp_require_build "$SRC" skip-build
fi

mkdir -p "$OUT"
. "$REPO/driver/sim_procs.sh"
sim_procs_init retirebench_sim
trap 'sim_kill_own_servers' EXIT INT TERM
sim_kill_own_servers; sleep 0.5

echo "== retirebench on blackhole, worker $TT_SIM_TENSIX_COORDS, scale $SCALE"
( cd "$SRC" && timeout "$TIMEOUT" ./build/retirebench \
    --scale "$SCALE" --label "$LABEL" --out "$OUT" ) >"$OUT/$LABEL.out" 2>&1
rc=$?
sim_kill_own_servers

if [ $rc -ne 0 ] || ! grep -q "Completed successfully on the device" "$OUT/$LABEL.out"; then
  echo "   FAILED (rc=$rc); see $OUT/$LABEL.out"
  tail -20 "$OUT/$LABEL.out"
  exit 1
fi

artefact="$OUT/retirebench-blackhole-$LABEL.json"
if [ ! -f "$artefact" ]; then
  echo "   the run reported success but wrote no $artefact"
  exit 1
fi

sed -n '/^zone /,$p' "$OUT/$LABEL.out"
echo "----"
echo "Simulator side complete. Decomposition:"
echo "  python3 -m tt_sim.perf.retire_attribution --decompose-only --sim $artefact"
