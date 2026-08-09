#!/usr/bin/env bash
# Run the documented riscvbench card session and leave CSVs to send back.
#
# This is the HARDWARE side. Do not use perfbench/run.sh here, and make sure
# TT_METAL_SIMULATOR is unset: the point of the exercise is that the same binary
# runs either way, and this script's job is only to build it, run it and say
# whether the run was worth keeping.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   perfbench/riscvbench/run_card.sh [stage ...] [-- extra prog args]
#
# Stages (default: main cross gset qdrain pairs):
#   main    --blocks 32                      all seven phases, all thread sets
#   cross   --blocks 8                       the second, shorter cross-check
#   gset    --phase g --gset 1 and --gset 2  the instruction-footprint cliff;
#                                            --gset 0 already rides in `main`
#   qdrain  --phase qs --variants t1         Tensix queue depth + is it shared
#   pairs   --phase r --probes 0x339         store-coalescing, multiply, divide
#                                            in one fast focused run -- on
#                                            Wormhole these are first-ever
#                                            measurements, so they get their own
#                                            CSV rather than only riding in main
#
# `--blocks 32` is the floor for the primary run: at --blocks 8 six of seven
# phases were refused by the validity gate. `cross` is deliberately below it and
# is a cross-check, not a measurement.
#
# Environment:
#   TT_METAL_HOME / TT_METAL_RUNTIME_ROOT  built tt-metal checkout (required)
#   RISCVBENCH_OUT                         output directory (default: src/)
#   RVBENCH_BLOCKS                         --blocks for the primary (default 32)
#
# SPDX-License-Identifier: Apache-2.0

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"

BLOCKS="${RVBENCH_BLOCKS:-32}"

# Arguments are parsed before the environment is required, so `--help` works on
# a box that does not have tt-metal set up yet.
STAGES=()
EXTRA=()
seen_sep=0
for arg in "$@"; do
  if [ "$seen_sep" = 1 ]; then EXTRA+=("$arg"); continue; fi
  case "$arg" in
    --) seen_sep=1 ;;
    -h|--help) sed -n '2,33p' "$0" | sed 's/^# \?//'; exit 0 ;;
    main|cross|gset|qdrain|pairs) STAGES+=("$arg") ;;
    *) echo "unknown stage '$arg' (want: main cross gset qdrain pairs)" >&2; exit 2 ;;
  esac
done
[ ${#STAGES[@]} -eq 0 ] && STAGES=(main cross gset qdrain pairs)

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is for a real card; unset it, or use perfbench/run.sh." >&2
  exit 2
fi

cd "$SRC" || exit 2
OUT="${RISCVBENCH_OUT:-$SRC}"
mkdir -p "$OUT" || exit 2

if [ ! -x build/riscvbench ]; then
  echo "== building riscvbench (this is most of the runtime, ~15 min)"
  cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
  cmake --build build -j >/dev/null || exit 1
fi

LOG="$OUT/riscvbench-run.log"
: > "$LOG"
rc=0

run_stage() {
  local label="$1"; shift
  echo "== $label" | tee -a "$LOG"
  ./build/riscvbench "$@" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOG"
  local r=${PIPESTATUS[0]}
  # riscvbench's exit status is a per-phase bitmask (1=R 2=T 4=C 8=Q 16=F
  # 32=S 64=G). Record it and keep going -- a later stage is still worth
  # having, and a failed phase is a finding rather than a reason to stop.
  [ "$r" -ne 0 ] && { echo "   ($label exited $r)" | tee -a "$LOG"; rc=$r; }
  return 0
}

want() { case " ${STAGES[*]} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if want main; then
  run_stage "main: --blocks $BLOCKS (all phases, t1,t2,t3)" \
    --blocks "$BLOCKS" --out "$OUT/riscvbench-main.csv"
fi
if want cross; then
  run_stage "cross: --blocks 8" --blocks 8 --out "$OUT/riscvbench-blocks8.csv"
fi
if want gset; then
  for g in 1 2; do
    run_stage "gset $g: --phase g --variants t1" \
      --phase g --variants t1 --blocks "$BLOCKS" --gset "$g" \
      --out "$OUT/riscvbench-g$g.csv"
  done
fi
if want qdrain; then
  run_stage "qdrain: --phase qs --variants t1" \
    --phase qs --variants t1 --blocks "$BLOCKS" \
    --out "$OUT/riscvbench-qdrain.csv"
fi
if want pairs; then
  # 0x339 = loop_overhead(0x1) | mul_indep(0x8) | mul_dep(0x10) | div(0x20)
  #         | store_spread(0x100) | store_coalesce(0x200)
  run_stage "pairs: --phase r --probes 0x339 --variants t1" \
    --phase r --probes 0x339 --variants t1 --blocks "$BLOCKS" \
    --out "$OUT/riscvbench-pairs.csv"
fi

if [ -f "$OUT/riscvbench-main.csv" ]; then
  echo | tee -a "$LOG"
  echo "== analysis" | tee -a "$LOG"
  python3 -m tt_sim.perf.riscv_bench_sweep --measured "$OUT/riscvbench-main.csv" 2>&1 \
    | tee "$OUT/riscvbench.report.txt" | tail -40
fi

echo
echo "SEND BACK: $OUT/riscvbench-*.csv"
echo "       and: $LOG"
echo
echo "Sanity checks before you do (see README.md):"
echo "  * every phase prints TTRVBENCH_VALID_<L>: yes, then TTRVBENCH_VALID: yes"
echo "  * the 'Is the instrument live?' block does NOT say all four probes read"
echo "    ~1.0 -- that is the signature of a run that measured nothing, and on"
echo "    silicon it is a broken run even though the exit status stays 0"
echo "  * phase S gives a verdict (it needs at least two thread counts, so the"
echo "    main run must have kept --variants t1,t2,t3)"
echo "  * the store-coalescing pair differs on Blackhole (~5.2x); on Wormhole"
echo "    it is predicted IDENTICAL -- either way that is the result"
exit "$rc"
