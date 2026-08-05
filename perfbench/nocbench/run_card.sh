#!/usr/bin/env bash
# Run the whole NoC congestion experiment on a real card, end to end.
#
# This is the HARDWARE side. Do not use perfbench/run.sh here: that one points
# TT_METAL_SIMULATOR at tt-sim. The point of the exercise is that it is the same
# binary either way, so on a card you run the executable directly, which is what
# this script does.
#
#   TT_METAL_HOME=/path/to/tt-metal ./perfbench/nocbench/run_card.sh [plan args...]
#
# Any extra arguments go to the planner, e.g.
#   ./run_card.sh --experiments all --num-tx 128
#
# Environment:
#   TT_METAL_HOME / TT_METAL_RUNTIME_ROOT   built tt-metal checkout (required)
#   NOCBENCH_OUT                            output directory (default: cwd)

set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"
OUT="${NOCBENCH_OUT:-$PWD}"

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Slow dispatch: tt-sim only supports the direct launch path, and using the same
# mode on hardware keeps the two runs comparable.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
unset TT_METAL_SIMULATOR || true

cd "$SRC"
if [ ! -x build/nocbench ]; then
  echo "== building nocbench"
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build build -j >/dev/null
fi

echo "== 1/3  dumping this card's core map"
./build/nocbench --dump-grid --out "$OUT/nocbench-grid.csv"
ARCH="$(sed -n 's/.*arch=\([a-z0-9_]*\).*/\1/p' "$OUT/nocbench-grid.csv" | head -1)"
echo "   arch=$ARCH"

echo "== 2/3  planning (this refuses rather than emit a confounded experiment)"
python3 -m tt_sim.perf.noc_congestion_plan \
  --grid "$OUT/nocbench-grid.csv" --out "$OUT/nocbench-plan.csv" "$@"

echo "== 3/3  running"
./build/nocbench --plan "$OUT/nocbench-plan.csv" --out "$OUT/nocbench-$ARCH.csv" -v

echo
python3 -m tt_sim.perf.noc_congestion_sweep --measured "$OUT/nocbench-$ARCH.csv" \
  | tee "$OUT/nocbench-$ARCH.report.txt"

echo
echo "send back: $OUT/nocbench-$ARCH.csv and $OUT/nocbench-$ARCH.report.txt"
