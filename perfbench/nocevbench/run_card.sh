#!/usr/bin/env bash
# nocevbench on a real card: collect tt-metal's NoC event traces for rung 4's
# NoC-timing leg.
#
# YOU DO NOT NEED TO KNOW ANYTHING ABOUT tt-sim TO RUN THIS. It builds one
# normal tt-metal program, runs it a handful of times with tt-metal's own NoC
# event profiler enabled, checks each run validated its own data, and leaves a
# directory to send home. No Tracy front end, no tt-exalens, no board reset,
# no root.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   ./run_card.sh --preflight                 # checks that cost no card time
#   ./run_card.sh --list                      # schedule and wall estimate
#   ./run_card.sh --out ~/tt_traces/nocevbench-session
#
# Options:
#   --out DIR         where the session lands (default ./nocevbench-session)
#   --bytes "256 4096"  the size arms (default "256 4096"). MUST match whatever
#                     the simulator side was run at. A comparison across two
#                     transfer sizes is a comparison of two different programs,
#                     and the analysis refuses it by census.
#   --chunks N        transfers per run (default 8). Same rule: must match.
#   --repeats N       runs per arm (default 3). Each lands in its own directory;
#                     they are never concatenated, because the analysis refuses
#                     a trace carrying two windows.
#   --preflight       run the checks that can fail before card time is spent,
#                     then stop
#   --list            print the schedule and the estimate, run nothing
#   --skip-build      assume build/nocevbench is current
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

OUT="$PWD/nocevbench-session"
BYTES="256 4096"
CHUNKS=8
REPEATS=3
LIST=0
PREFLIGHT=0
SKIP_BUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    --list) LIST=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

n_arms=$(echo "$BYTES" | wc -w)
n_runs=$((n_arms * REPEATS))
# ~20 s per run: device open plus one launch. The kernel itself is microseconds.
est_s=$((n_runs * 20))

echo "nocevbench card session"
echo "  arms      : $BYTES bytes per transfer  (must match the simulator side)"
echo "  chunks    : $CHUNKS transfers per run  (must match the simulator side)"
echo "  repeats   : $REPEATS per arm, each in its own directory"
echo "  instrument: TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
echo "              -> forces the device profiler on (rtoptions.cpp:854) and"
echo "                 injects -DPROFILE_NOC_EVENTS into the kernel JIT build"
echo "                 (jit_build/build.cpp:188). tt-metal warns that it costs"
echo "                 1-15 % of cycles; that overhead is paid on BOTH sides."
echo "  runs      : $n_runs"
echo "  wall      : ~$((est_s / 60)) min $((est_s % 60)) s, plus the first build (~2 min)"
echo "  out       : $OUT"
[ "$LIST" -eq 1 ] && exit 0

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"

# ---------------------------------------------------------------------------
# Pre-flight. Everything here fails for free; everything after it costs card
# time. The Tracy check is the one that matters: EVERY path that writes a NoC
# trace is inside `#if defined(TRACY_ENABLE)` (profiler.cpp:2531), and unlike
# TT_METAL_DEVICE_PROFILER=1 -- which TT_FATALs without it (rtoptions.cpp:815)
# -- the NoC-events flag has no such guard. On a non-Tracy build it runs to
# completion, reports PASS, and silently writes nothing.
# ---------------------------------------------------------------------------
preflight_fail=0
echo ""
echo "== pre-flight"

if [ -f "$TT_METAL_HOME/build/CMakeCache.txt" ] \
   && grep -qiE '^(ENABLE_TRACY|TT_METAL_ENABLE_TRACY):BOOL=(ON|TRUE|1)' \
        "$TT_METAL_HOME/build/CMakeCache.txt"; then
  echo "   ok   tt-metal was configured with Tracy enabled"
elif [ -f "$TT_METAL_HOME/build/CMakeCache.txt" ]; then
  echo "   WARN could not confirm Tracy in $TT_METAL_HOME/build/CMakeCache.txt."
  echo "        If the smoke run below produces no noc_trace_*.json, this is why:"
  echo "        rebuild with ./build_metal.sh --enable-profiler"
else
  echo "   WARN no build/CMakeCache.txt under TT_METAL_HOME; cannot check Tracy"
fi

for b in $BYTES; do
  if [ $((b % 64)) -ne 0 ]; then
    echo "   FAIL --bytes $b is not a multiple of 64; the NoC congruence rule"
    echo "        makes such a transfer undefined behaviour, not slow"
    preflight_fail=1
  elif [ "$b" -gt 8160 ]; then
    echo "   FAIL --bytes $b exceeds 8160; the profiler's event record stores the"
    echo "        payload as 32-byte chunks in a uint8 and would record it saturated"
    preflight_fail=1
  else
    echo "   ok   --bytes $b is congruent and inside the event record's range"
  fi
done

total=$(( $(echo "$BYTES" | tr ' ' '\n' | sort -n | tail -1) * CHUNKS ))
if [ "$total" -gt $((64 * 1024)) ]; then
  echo "   FAIL largest arm needs $total B of L1 scratch; the cap is 64 KiB"
  preflight_fail=1
else
  echo "   ok   largest arm needs $total B of L1 scratch"
fi

# 250 optional markers per RISC per launch (profiler_common.h), two words per
# NoC event. A run that overflows drops events SILENTLY, and the analysis's
# barriers_pair gate is the only thing that would notice.
events_per_run=$((CHUNKS * 3 + 2))
if [ "$events_per_run" -gt 100 ]; then
  echo "   FAIL --chunks $CHUNKS would record ~$events_per_run events per RISC;"
  echo "        keep it under 100, well inside the profiler's 125-marker budget,"
  echo "        because an overflow drops events with no error"
  preflight_fail=1
else
  echo "   ok   ~$events_per_run events per RISC per run, inside the marker budget"
fi

if [ "$preflight_fail" -ne 0 ]; then
  echo ""
  echo "PRE-FLIGHT FAILED -- no card time spent. Fix the above and re-run."
  exit 3
fi
echo "   pre-flight passed"
[ "$PREFLIGHT" -eq 1 ] && exit 0

if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$SRC/build/nocevbench" ]; then
  echo ""
  echo "== building (tt-metal supplies every flag; nothing to configure)"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || { echo "BUILD FAILED"; exit 1; }
fi

mkdir -p "$OUT/runs"
SUMMARY="$OUT/summary.txt"
: >"$SUMMARY"

{
  echo "nocevbench card session"
  echo "date        : $(date -Is)"
  echo "host        : $(hostname)"
  echo "tt-metal    : $TT_METAL_HOME"
  echo "tt-metal rev: $(git -C "$TT_METAL_HOME" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "arch        : ${ARCH_NAME:-unset}"
  echo "bytes       : $BYTES"
  echo "chunks      : $CHUNKS"
  echo "repeats     : $REPEATS"
  echo "instrument  : TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
} >"$OUT/env.txt"

status=0
# Arms interleaved rather than blocked, per this project's established
# discipline: a blocked schedule turns any slow drift into a fake difference
# between the arms. Cycle counts are far less exposed to drift than a power
# reading, but interleaving is free.
for rep in $(seq 1 "$REPEATS"); do
  for bytes in $BYTES; do
    name="$bytes-$rep"
    dir="$OUT/runs/$name"
    mkdir -p "$dir"
    echo "== $name"
    ( cd "$SRC" \
      && TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
         TT_METAL_PROFILER_DIR="$dir" \
         ./build/nocevbench "$bytes" "$CHUNKS" ) >"$dir/stdout.log" 2>&1
    rc=$?
    trace=$(ls "$dir"/.logs/noc_trace_dev*_ID*.json 2>/dev/null | head -1)
    if [ $rc -ne 0 ] || ! grep -q "Completed successfully on the device" "$dir/stdout.log"; then
      printf 'FAIL  %-12s rc=%s  the program did not validate its own data\n' \
        "$name" "$rc" | tee -a "$SUMMARY"
      status=1
      continue
    fi
    if [ -z "$trace" ]; then
      printf 'FAIL  %-12s no noc_trace_*.json in %s/.logs/ -- is this a Tracy build?\n' \
        "$name" "$dir" | tee -a "$SUMMARY"
      status=1
      continue
    fi
    n=$(python3 -c "import json,sys;print(sum(1 for r in json.load(open(sys.argv[1])) if 'type' in r))" "$trace" 2>/dev/null || echo 0)
    barriers=$(python3 -c "import json,sys;rs=json.load(open(sys.argv[1]));print(sum(1 for r in rs if str(r.get('type','')).endswith('BARRIER_END')))" "$trace" 2>/dev/null || echo 0)
    if [ "$n" -eq 0 ]; then
      printf 'FAIL  %-12s trace has zero NoC events -- the recorder was not compiled in\n' \
        "$name" | tee -a "$SUMMARY"
      status=1
      continue
    fi
    if [ "$barriers" -ne $((CHUNKS * 2)) ]; then
      printf 'FAIL  %-12s %s barrier ENDs, expected %s -- events were dropped\n' \
        "$name" "$barriers" "$((CHUNKS * 2))" | tee -a "$SUMMARY"
      status=1
      continue
    fi
    printf 'PASS  %-12s events=%-4s barrier_ends=%-4s %s\n' \
      "$name" "$n" "$barriers" "$(basename "$trace")" | tee -a "$SUMMARY"
  done
done

{
  echo ""
  echo "Before sending this home, check by eye:"
  echo "  * every line above says PASS;"
  echo "  * the $REPEATS repeats of an arm agree. They are separate runs of an"
  echo "    identical program, so a big spread means something else was on the"
  echo "    card and the session is suspect. Do NOT re-run and keep the better"
  echo "    session -- say so in the handover instead."
  echo "  * the arms DISAGREE with each other. They differ only in transfer size"
  echo "    and are supposed to separate the per-hop term from the per-byte one;"
  echo "    identical numbers mean one arm did not run the way it was meant to."
  echo ""
  echo "  A quick look at one trace, if you want one:"
  echo "    python3 -c \"import json;d=json.load(open('<trace>'));"
  echo "      print([(r.get('proc'),r.get('type'),r['timestamp']) for r in d][:12])\""
  echo ""
  echo "Then send the WHOLE directory, stdout logs included:"
  echo "  rsync -av $OUT/ <home>:~/nocevbench-session/"
} | tee -a "$SUMMARY"

exit $status
