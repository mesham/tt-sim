#!/usr/bin/env bash
# Run dramratebench on a real card and leave one CSV to send back.
#
# This is the HARDWARE side. Do not use perfbench/run.sh here, and make sure
# TT_METAL_SIMULATOR is unset: the point of the exercise is that the same binary
# runs either way, and this script's job is only to build it, run it and say
# whether the run was worth keeping.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   perfbench/dramratebench/run_card.sh [-- prog args...]
#   perfbench/dramratebench/run_card.sh -- --sustained   # the vendor's 1/12/48
#
# The prediction this run is to be read against was recorded BEFORE it, in
# prediction-sustained.csv next to this script. Read it first.
#
# SPDX-License-Identifier: Apache-2.0

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"

if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is for a real card; unset it, or use perfbench/run.sh." >&2
  exit 2
fi

ARGS=()
seen_sep=0
for arg in "$@"; do
  if [ "$seen_sep" = 1 ]; then ARGS+=("$arg"); continue; fi
  case "$arg" in
    --) seen_sep=1 ;;
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) ARGS+=("$arg") ;;
  esac
done

cd "$SRC" || exit 2
if [ ! -x build/dramratebench ]; then
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
  cmake --build build -j >/dev/null || exit 1
fi

LOG="$SRC/dramratebench-run.log"
echo "dramratebench: running (log: $LOG)"
./build/dramratebench "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

CSV="$(ls -t "$SRC"/dramratebench-*.csv 2>/dev/null | head -1)"
echo
if [ -z "$CSV" ]; then
  echo "no CSV was produced; send $LOG" >&2
  exit "$rc"
fi
echo "SEND BACK: $CSV"
echo "       and: $LOG"
echo
echo "Sanity checks before you do (see README.md, 'Telling a good run from a"
echo "degenerate one'):"
echo "  * the printed VERDICT is not DEGENERATE. A FLAT onechan curve is the"
echo "    EXPECTED answer and is NOT on its own a result -- it is a result only"
echo "    when the fanchan control moved."
echo "  * tags_ok equals num_readers in every row, or the readers were not"
echo "    talking to the bank the plan names"
echo "  * max_barrier_spins is non-zero somewhere, or the bursts never overlapped"
echo
echo "The SUSTAINED RATE table above the verdict is the level, and the prediction"
echo "it is to be read against was recorded before this run:"
echo "  $HERE/prediction-sustained.csv"
echo "At home: python3 -m tt_sim.perf.dram_rate_sweep --measured <the CSV>"
exit "$rc"
