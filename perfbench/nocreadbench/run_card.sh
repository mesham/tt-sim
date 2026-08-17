#!/usr/bin/env bash
# Run nocreadbench on a real card and leave one CSV to send back.
#
# This is the HARDWARE side. Do not use perfbench/run.sh here, and make sure
# TT_METAL_SIMULATOR is unset: the point of the exercise is that the same binary
# runs either way, and this script's job is only to build it, run it and say
# whether the run was worth keeping.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   perfbench/nocreadbench/run_card.sh [-- prog args...]
#
# SPDX-License-Identifier: Apache-2.0

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"

# Its program launches through `detail::LaunchProgram`, the direct path, which
# requires slow dispatch; a card defaults to FAST dispatch, so without this every
# run aborts with rc=134 before doing any work. Measured on a Blackhole p150,
# 2026-08-17, when nocevbench hit exactly this. The simulator runners have always
# set it (tt-sim supports no other flow), which is why the gap survived in every
# card runner: the sim side cannot reproduce the failure.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
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
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) ARGS+=("$arg") ;;
  esac
done

cd "$SRC" || exit 2
if [ ! -x build/nocreadbench ]; then
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
  cmake --build build -j >/dev/null || exit 1
fi

LOG="$SRC/nocreadbench-run.log"
echo "nocreadbench: running (log: $LOG)"
./build/nocreadbench "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

CSV="$(ls -t "$SRC"/nocreadbench-*.csv 2>/dev/null | head -1)"
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
echo "  * the printed VERDICT is not DEGENERATE"
echo "  * the 'burst' rows at 64 B settle near 25 cycles/tx (Wormhole) or"
echo "    35 (Blackhole) -- that is the published-dataset control"
echo "  * on Blackhole, cmdbuf_avail_rest is NOT 0xFFFFFFFF"
exit "$rc"
