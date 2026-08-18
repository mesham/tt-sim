#!/usr/bin/env bash
# mechbench on a real card: collect the Tensix hardware stall counters for
# rung 4's mechanism-attribution leg.
#
# YOU DO NOT NEED TO KNOW ANYTHING ABOUT tt-sim TO RUN THIS. It builds one
# normal tt-metal program, runs it a handful of times with the performance
# counters enabled, checks each run validated its own arithmetic, and leaves a
# directory to send home. No Tracy, no tt-exalens, no board reset, no root.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   ./run_card.sh --list                       # schedule and wall estimate
#   ./run_card.sh --out ~/tt_traces/mechbench-session
#
# Options:
#   --out DIR        where the session lands (default ./mechbench-session)
#   --tiles N        tiles per run (default 256). MUST match whatever the
#                    simulator side was run at -- see README, "The simulator
#                    side". A comparison across two tile counts is a comparison
#                    of two different programs.
#   --repeats N      runs per arm (default 3). Each lands in its own directory;
#                    they are never concatenated, because the analysis refuses a
#                    log carrying two windows.
#   --arms "elw mm"  which arms to run
#   --corroborate    also run pass B (mask 39: FPU|PACK|UNPACK|INSTRN). Card-only
#                    context -- tt-sim models none of those banks.
#   --l1             also run the five L1 bank passes, one run each. They share
#                    one hardware mux and CANNOT be combined; tt-metal throws if
#                    you try. tt-sim models none of them.
#   --list           print the schedule and the estimate, run nothing
#   --skip-build     assume build/mechbench is current
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
# shellcheck source=../build_provenance.sh
. "$HERE/../build_provenance.sh"

OUT="$PWD/mechbench-session"
TILES=256
REPEATS=3
ARMS="elw mm"
CORROBORATE=0
L1=0
LIST=0
SKIP_BUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --tiles) TILES="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --corroborate) CORROBORATE=1; shift ;;
    --l1) L1=1; shift ;;
    --list) LIST=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,35p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# The passes, as "<label>:<mask>". Bit 5 (32) is INSTRN_THREAD, the only bank
# whose counters this leg's partition consumes. Bits 0/1/2 (FPU/PACK/UNPACK)
# are independent instances and may share a run with it. Bits 3/4/6/7/8 are the
# five L1 banks behind ONE mux -- one per run, no exceptions.
PASSES="instrn:32"
[ "$CORROBORATE" -eq 1 ] && PASSES="$PASSES units:39"
[ "$L1" -eq 1 ] && PASSES="$PASSES l1_0:8 l1_1:16 l1_2:64 l1_3:128 l1_4:256"

n_arms=$(echo "$ARMS" | wc -w)
n_passes=$(echo "$PASSES" | wc -w)
n_runs=$((n_arms * REPEATS * n_passes))
# ~20 s per run: device open plus one launch. The kernel itself is microseconds.
est_s=$((n_runs * 20))

echo "mechbench card session"
echo "  arms      : $ARMS"
echo "  tiles     : $TILES   (must match the simulator side)"
echo "  repeats   : $REPEATS per arm per pass, each in its own directory"
echo "  passes    :"
for pass in $PASSES; do
  label="${pass%%:*}"; mask="${pass##*:}"
  case "$label" in
    instrn) why="REQUIRED. INSTRN_THREAD bank; every counter the criterion uses." ;;
    units)  why="corroboration only. FPU|PACK|UNPACK|INSTRN; tt-sim models none of the first three." ;;
    l1_*)   why="optional. One L1 bank; the five share a mux and cannot be combined." ;;
    *)      why="" ;;
  esac
  printf '      %-8s TT_METAL_PROFILE_PERF_COUNTERS=%-4s %s\n' "$label" "$mask" "$why"
done
echo "  runs      : $n_runs"
echo "  wall      : ~$((est_s / 60)) min $((est_s % 60)) s, plus the first build (~2 min)"
echo "  out       : $OUT"
[ "$LIST" -eq 1 ] && exit 0

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"

# A card box that also has a tt-sim checkout is exactly where this goes wrong:
# with TT_METAL_SIMULATOR left set, every run below completes, validates its own
# arithmetic, records counter samples and writes a session directory that is a
# SIMULATOR decomposition labelled as a card's -- which is the one artefact this
# leg must never produce, since comparing it against tt-sim is comparing tt-sim
# to itself. `dramratebench` and `energybench` have always refused it.
if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is the CARD side; unset it, or use ./run_sim.sh." >&2
  exit 2
fi

# Its program launches through `detail::LaunchProgram`, the direct path, which
# requires slow dispatch; a card defaults to FAST dispatch, so without this every
# run aborts with rc=134 before doing any work. Measured on a Blackhole p150,
# 2026-08-17, when nocevbench hit exactly this. The simulator runners have always
# set it (tt-sim supports no other flow), which is why the gap survived in every
# card runner: the sim side cannot reproduce the failure.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

# Which tt-metal the tree was made against, before cmake gets to reuse a cache
# that still points at a different one -- see build_provenance.sh.
if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$SRC/build/mechbench" ]; then
  bp_require_build "$SRC"
  echo "== building (tt-metal supplies every flag; nothing to configure)"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || { echo "BUILD FAILED"; exit 1; }
  bp_record_build "$SRC"
else
  bp_require_build "$SRC" skip-build
fi

mkdir -p "$OUT/runs"
SUMMARY="$OUT/summary.txt"
: >"$SUMMARY"

{
  echo "mechbench card session"
  echo "date        : $(date -Is)"
  echo "host        : $(hostname)"
  echo "tt-metal    : $TT_METAL_HOME"
  echo "tt-metal rev: $(git -C "$TT_METAL_HOME" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "tiles       : $TILES"
  echo "repeats     : $REPEATS"
  echo "passes      : $PASSES"
} >"$OUT/env.txt"

status=0
# Arms interleaved rather than blocked, per this project's established
# discipline: a blocked schedule turns any slow drift into a fake difference
# between the arms.
for pass in $PASSES; do
  label="${pass%%:*}"; mask="${pass##*:}"
  for rep in $(seq 1 "$REPEATS"); do
    for arm in $ARMS; do
      name="$label-$arm-$rep"
      dir="$OUT/runs/$name"
      mkdir -p "$dir"
      echo "== $name (mask $mask)"
      ( cd "$SRC" \
        && TT_METAL_PROFILE_PERF_COUNTERS="$mask" \
           TT_METAL_PROFILER_DIR="$dir" \
           ./build/mechbench "$arm" "$TILES" ) >"$dir/stdout.log" 2>&1
      rc=$?
      log="$dir/.logs/profile_log_device.csv"
      samples=0
      [ -f "$log" ] && samples=$(grep -c ',9090,' "$log" || true)
      if [ $rc -ne 0 ] || ! grep -q "Completed successfully on the device" "$dir/stdout.log"; then
        printf 'FAIL  %-18s rc=%s  the program did not validate its own output\n' \
          "$name" "$rc" | tee -a "$SUMMARY"
        status=1
        continue
      fi
      if [ "$samples" -eq 0 ]; then
        printf 'FAIL  %-18s no counter samples in %s -- was the mask honoured?\n' \
          "$name" "$log" | tee -a "$SUMMARY"
        status=1
        continue
      fi
      span=$(grep ',9090,' "$log" | head -1 | sed 's/.*"ref cnt"://; s/;.*//')
      stalls=$(grep 'THREAD_STALLS_1' "$log" | head -1 | sed 's/.*"value"://; s/}.*//')
      printf 'PASS  %-18s samples=%-4s span=%-8s THREAD_STALLS_1=%s\n' \
        "$name" "$samples" "$span" "$stalls" | tee -a "$SUMMARY"
    done
  done
done

{
  echo ""
  echo "Before sending this home, check by eye:"
  echo "  * every line above says PASS;"
  echo "  * the $REPEATS repeats of an arm agree on span and on THREAD_STALLS_1."
  echo "    They are separate runs of an identical program, so a big spread"
  echo "    means something else was on the card, and the session is suspect."
  echo "  * elw and mm DISAGREE. They differ by one instruction and are supposed"
  echo "    to load opposite mechanisms; identical numbers mean one arm did not"
  echo "    build the way it was meant to."
  echo ""
  echo "Then send the WHOLE directory, stdout logs included:"
  echo "  rsync -av $OUT/ <home>:~/mechbench-session/"
} | tee -a "$SUMMARY"

exit $status
