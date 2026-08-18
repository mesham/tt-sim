#!/usr/bin/env bash
# Run the documented tensixbench card session and leave CSVs to send back.
#
# This is the HARDWARE side. Do not use perfbench/run.sh here, and make sure
# TT_METAL_SIMULATOR is unset: the point of the exercise is that the same binary
# runs either way, and this script's job is only to build it, run it and say
# whether the run was worth keeping.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   perfbench/tensixbench/run_card.sh [stage ...] [-- extra prog args]
#
# Stages (default: main dvalid formats):
#   main      --blocks 32 --iters 64             the primary A+B sample
#   dvalid    --dvalid-per-thread --phase a      the dvalid-attribution control
#   formats   --src-format {bf16,fp32,tf32,fp16} the four unpack-format runs
#
# Environment:
#   TT_METAL_HOME / TT_METAL_RUNTIME_ROOT  built tt-metal checkout (required)
#   TENSIXBENCH_OUT                        output directory (default: src/)
#   TTBENCH_BLOCKS                         --blocks for every stage (default 32)
#   TTBENCH_ITERS                          --iters for main/dvalid (default 64)
#
# SPDX-License-Identifier: Apache-2.0

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"
# shellcheck source=../build_provenance.sh
. "$HERE/../build_provenance.sh"

BLOCKS="${TTBENCH_BLOCKS:-32}"
ITERS="${TTBENCH_ITERS:-64}"

# Arguments are parsed before the environment is required, so `--help` works on
# a box that does not have tt-metal set up yet.
STAGES=()
EXTRA=()
seen_sep=0
for arg in "$@"; do
  if [ "$seen_sep" = 1 ]; then EXTRA+=("$arg"); continue; fi
  case "$arg" in
    --) seen_sep=1 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
    main|dvalid|formats) STAGES+=("$arg") ;;
    *) echo "unknown stage '$arg' (want: main dvalid formats)" >&2; exit 2 ;;
  esac
done
[ ${#STAGES[@]} -eq 0 ] && STAGES=(main dvalid formats)

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Slow dispatch: tt-sim only supports the direct launch path, and using the same
# mode on hardware keeps the two runs comparable.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is for a real card; unset it, or use perfbench/run.sh." >&2
  exit 2
fi

cd "$SRC" || exit 2
OUT="${TENSIXBENCH_OUT:-$SRC}"
mkdir -p "$OUT" || exit 2

if [ ! -x build/tensixbench ]; then
  echo "== building tensixbench (this is most of the runtime, ~15 min)"
  bp_require_build "$SRC"
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
  cmake --build build -j >/dev/null || exit 1
  bp_record_build "$SRC"
else
  # An existing binary is never rebuilt here, so which tt-metal it was made
  # against has to be checked on every run and not only on the run that made it.
  bp_require_build "$SRC" skip-build
fi

LOG="$OUT/tensixbench-run.log"
: > "$LOG"
rc=0

run_stage() {
  local label="$1"; shift
  echo "== $label" | tee -a "$LOG"
  ./build/tensixbench "$@" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOG"
  local r=${PIPESTATUS[0]}
  # tensixbench's exit status is a bitmask: 1 = phase A checks failed,
  # 2 = phase B, 3 = both. Keep going -- a later stage is still worth having.
  [ "$r" -ne 0 ] && { echo "   ($label exited $r)" | tee -a "$LOG"; rc=$r; }
  return 0
}

want() { case " ${STAGES[*]} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if want main; then
  run_stage "main: --blocks $BLOCKS --iters $ITERS" \
    --blocks "$BLOCKS" --iters "$ITERS"
fi
if want dvalid; then
  run_stage "dvalid: --dvalid-per-thread --phase a" \
    --blocks "$BLOCKS" --iters "$ITERS" --dvalid-per-thread --phase a
fi
if want formats; then
  for f in bf16 fp32 tf32 fp16; do
    run_stage "format: $f" --blocks "$BLOCKS" --phase a --variants t1 \
      --dvalid-unpacr-nop --src-format "$f"
  done
fi

PRIMARY="$(ls -t "$OUT"/tensixbench-*.csv 2>/dev/null | grep -v -- '-fmt-\|-dvalid' | head -1)"
echo | tee -a "$LOG"
if [ -n "$PRIMARY" ]; then
  echo "== analysis" | tee -a "$LOG"
  python3 -m tt_sim.perf.tensix_bench_sweep --measured "$PRIMARY" 2>&1 \
    | tee "$OUT/tensixbench.report.txt" | tail -40
fi

echo
echo "SEND BACK: $OUT/tensixbench-*.csv"
echo "       and: $LOG"
echo
echo "Sanity checks before you do (see README.md):"
echo "  * TTBENCH_VALID_A and TTBENCH_VALID_B both read 'yes'"
echo "  * cyc/instr is >= 1.00 for every probe -- a run where EVERY probe"
echo "    reads exactly 1.000 measured nothing back-pressuring the issuer"
echo "  * the MATH probes (MVMUL/ELWADD/ELWMUL) are well above 1.0; if they"
echo "    are not, the dvalid setup did not take"
echo "  * phase B: the LoFi->HiFi2->HiFi4 deltas are near 16 and 32"
exit "$rc"
