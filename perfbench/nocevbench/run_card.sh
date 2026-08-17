#!/usr/bin/env bash
# nocevbench on a real card: collect tt-metal's NoC event traces for rung 4's
# NoC-timing leg.
#
# YOU DO NOT NEED TO KNOW ANYTHING ABOUT tt-sim TO RUN THIS. It builds one
# normal tt-metal program, runs it a handful of times with tt-metal's own NoC
# event profiler enabled, checks each run validated its own data AND that it
# really ran the arm it was asked for, and leaves a directory to send home. No
# Tracy front end, no tt-exalens, no board reset, no root.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   ./run_card.sh --preflight --arms "A B C"  # checks that cost no card time
#   ./run_card.sh --list --arms "A B C"       # schedule and wall estimate
#   ./run_card.sh --arms "A B C" --out ~/tt_traces/nocfixed-session
#
# Options:
#   --arm A|B|C       a single arm (default A). See "the arms" below.
#   --arms "A B C"    several arms, interleaved with the size arms
#   --peer X,Y        arm C's peer core, in LOGICAL worker coords (default 1,1)
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
#
# THE ARMS
#   A  reader NCRISC/NOC_1 from DRAM, writer BRISC/NOC_0 to DRAM.
#      THE CONTROL. It is the 2026-08-17 session's configuration, unchanged,
#      and it must reproduce it -- if it does not, nothing else in the session
#      is comparable to that one and the handover must say so.
#   B  the same program with the two NoCs SWAPPED and nothing else changed.
#      The discriminator: today "writes are +95 cycles" and "NoC 0 is +95
#      cycles" are the same statement, and only this arm can separate them.
#   C  reader and writer as arm A, but against a PEER CORE's L1 instead of
#      DRAM, which takes the DRAM endpoint's service term out of the path.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

OUT="$PWD/nocevbench-session"
ARMS="A"
PEER="1,1"
BYTES="256 4096"
CHUNKS=8
REPEATS=3
LIST=0
PREFLIGHT=0
SKIP_BUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARMS="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --arms) ARMS="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --peer) PEER="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    --list) LIST=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,43p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

n_sizes=$(echo "$BYTES" | wc -w)
n_arms=$(echo "$ARMS" | wc -w)
n_runs=$((n_sizes * n_arms * REPEATS))
# ~20 s per run: device open plus one launch. The kernel itself is microseconds.
est_s=$((n_runs * 20))

echo "nocevbench card session"
echo "  arms      : $ARMS  (A is the control and must reproduce 2026-08-17)"
echo "  sizes     : $BYTES bytes per transfer  (must match the simulator side)"
echo "  chunks    : $CHUNKS transfers per run  (must match the simulator side)"
echo "  peer      : $PEER (logical), used by arm C only"
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
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"

# A card box that also has a tt-sim checkout is exactly where this goes wrong:
# with TT_METAL_SIMULATOR left set, every run below completes, validates its own
# data, passes the arm check and writes a session directory that is a SIMULATOR
# trace labelled as a card's. `dramratebench` and `energybench` have always
# refused it; this one had not.
if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is the CARD side; unset it, or use ./run_sim.sh." >&2
  exit 2
fi

# `nocevbench.cpp` launches through `detail::LaunchProgram`, which is the direct
# path and requires slow dispatch. A card defaults to FAST dispatch, so without
# this every run aborts with rc=134 before it reaches a single transfer --
# measured on a Blackhole p150, 2026-08-17. `run_sim.sh` has always set it
# (tt-sim supports no other flow), which is exactly why the gap survived: the
# simulator side could never reproduce the failure. Overridable, because a
# future program on the command-queue flow would want it unset.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

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

for arm in $ARMS; do
  case "$arm" in
    A|B|C) echo "   ok   arm $arm is a known arm" ;;
    *) echo "   FAIL '$arm' is not an arm; expected A, B or C"; preflight_fail=1 ;;
  esac
done

case "$PEER" in
  [0-9]*,[0-9]*) echo "   ok   --peer $PEER parses as logical worker coords" ;;
  *) echo "   FAIL --peer $PEER is not X,Y"; preflight_fail=1 ;;
esac

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

biggest=$(echo "$BYTES" | tr ' ' '\n' | sort -n | tail -1)
total=$((biggest * CHUNKS))
# Arm C holds three regions of that size (this core's scratch, the peer's source
# and the peer's destination) instead of one circular buffer.
l1_need=$total
case " $ARMS " in *" C "*) l1_need=$((3 * total)) ;; esac
if [ "$l1_need" -gt $((192 * 1024)) ]; then
  echo "   FAIL largest arm needs $l1_need B of L1; keep it under 192 KiB"
  preflight_fail=1
elif [ "$total" -gt $((64 * 1024)) ]; then
  echo "   FAIL largest arm needs $total B of L1 scratch; the cap is 64 KiB"
  preflight_fail=1
else
  echo "   ok   largest arm needs $l1_need B of L1"
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

if [ ! -f "$HERE/check_arm.py" ]; then
  echo "   FAIL check_arm.py is missing; without it a run that silently kept a"
  echo "        different arm's NoC pairing would be indistinguishable from a"
  echo "        good one, and that is the failure this session cannot survive"
  preflight_fail=1
else
  echo "   ok   check_arm.py is present; every run will be checked against its arm"
fi

if [ "$preflight_fail" -ne 0 ]; then
  echo ""
  echo "PRE-FLIGHT FAILED -- no card time spent. Fix the above and re-run."
  exit 3
fi

if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$SRC/build/nocevbench" ]; then
  echo ""
  echo "== building (tt-metal supplies every flag; nothing to configure)"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || { echo "BUILD FAILED"; exit 1; }
fi

# ---------------------------------------------------------------------------
# The part of the pre-flight that needs the device. It opens it and closes it
# again without launching anything (~5 s), because two of the three things this
# experiment can get silently wrong are properties of the device and are not
# knowable from the host: whether the peer core exists in the compute grid, and
# which distance class it lands in. Getting the second one wrong does not fail
# -- it measures a different geometry, and on a directional torus the geometry
# is the whole question.
# ---------------------------------------------------------------------------
echo ""
echo "== pre-flight, on the device (opens and closes it; launches nothing)"
ARCH_NAME=""
for arm in $ARMS; do
  desc=$( cd "$SRC" && ./build/nocevbench "$biggest" "$CHUNKS" \
                         --describe --arm "$arm" --peer "$PEER" 2>&1 )
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "   FAIL arm $arm cannot be configured on this device:"
    echo "$desc" | sed 's/^/        /'
    preflight_fail=1
    continue
  fi
  cfg=$(echo "$desc" | grep '^nocevbench-config' | head -1)
  echo "   ok   arm $arm: $cfg"
  # The part, from the device rather than from the operator's memory. env.txt
  # used to record `arch : unset` in every session, because ARCH_NAME was
  # printed and never assigned.
  ARCH_NAME=$(echo "$desc" | sed -n 's/^nocevbench-describe arch=\([a-z0-9_]*\).*/\1/p' | head -1)
  if [ "$arm" = "C" ]; then
    case "$cfg" in
      *distance=both-axes*)
        echo "        peer differs on BOTH axes, which is the geometry arm C claims" ;;
      *)
        echo "   FAIL arm C's peer is not on a both-axes route; that is a different"
        echo "        experiment from the one the predictions were written for"
        preflight_fail=1 ;;
    esac
  fi
done

if [ "$preflight_fail" -ne 0 ]; then
  echo ""
  echo "PRE-FLIGHT FAILED -- no measurement taken. Fix the above and re-run."
  exit 3
fi
echo "   pre-flight passed"
echo "   part      : ${ARCH_NAME:-unknown} (read from the device, recorded in env.txt)"
if [ "${ARCH_NAME:-}" != "blackhole" ]; then
  echo ""
  echo "   NOTE: the arm-A control band printed at the end of this run"
  echo "   (WRITE 256 375-382, READ 256 480-510) is the 2026-08-17 BLACKHOLE"
  echo "   session's, and there is no prior card session on this part. On a"
  echo "   first ${ARCH_NAME:-non-blackhole} run those numbers do NOT apply:"
  echo "   arm A ESTABLISHES the control rather than reproducing one, and what"
  echo "   it is checked against is the simulator side at the same --arch,"
  echo "   --bytes and --chunks. Do not report 'the control moved' off them."
fi
echo ""
echo "   NOT checked here, and checked after every run instead: that the reader"
echo "   and writer really got the NoCs the arm asked for. It is a property of"
echo "   the emitted trace, not of the command line, so it cannot be asserted"
echo "   in advance -- check_arm.py reads it back out of every trace below."
[ "$PREFLIGHT" -eq 1 ] && exit 0

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
  echo "arms        : $ARMS"
  echo "peer        : $PEER (logical, arm C only)"
  echo "bytes       : $BYTES"
  echo "chunks      : $CHUNKS"
  echo "repeats     : $REPEATS"
  echo "instrument  : TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1"
} >"$OUT/env.txt"

status=0
# Arms interleaved rather than blocked, per this project's established
# discipline: a blocked schedule turns any slow drift into a fake difference
# between the arms. Cycle counts are far less exposed to drift than a power
# reading, but interleaving is free -- and it now matters more, because the
# experimental arms are compared to each other and not only to the simulator.
for rep in $(seq 1 "$REPEATS"); do
  for arm in $ARMS; do
    for bytes in $BYTES; do
      name="$arm-$bytes-$rep"
      dir="$OUT/runs/$name"
      mkdir -p "$dir"
      echo "== $name"
      ( cd "$SRC" \
        && TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
           TT_METAL_PROFILER_DIR="$dir" \
           ./build/nocevbench "$bytes" "$CHUNKS" --arm "$arm" --peer "$PEER" ) >"$dir/stdout.log" 2>&1
      rc=$?
      trace=$(ls "$dir"/.logs/noc_trace_dev*_ID*.json 2>/dev/null | head -1)
      if [ $rc -ne 0 ] || ! grep -q "Completed successfully on the device" "$dir/stdout.log"; then
        printf 'FAIL  %-14s rc=%s  the program did not validate its own data\n' \
          "$name" "$rc" | tee -a "$SUMMARY"
        status=1
        continue
      fi
      if [ -z "$trace" ]; then
        printf 'FAIL  %-14s no noc_trace_*.json in %s/.logs/ -- is this a Tracy build?\n' \
          "$name" "$dir" | tee -a "$SUMMARY"
        status=1
        continue
      fi
      n=$(python3 -c "import json,sys;print(sum(1 for r in json.load(open(sys.argv[1])) if 'type' in r))" "$trace" 2>/dev/null || echo 0)
      barriers=$(python3 -c "import json,sys;rs=json.load(open(sys.argv[1]));print(sum(1 for r in rs if str(r.get('type','')).endswith('BARRIER_END')))" "$trace" 2>/dev/null || echo 0)
      if [ "$n" -eq 0 ]; then
        printf 'FAIL  %-14s trace has zero NoC events -- the recorder was not compiled in\n' \
          "$name" | tee -a "$SUMMARY"
        status=1
        continue
      fi
      if [ "$barriers" -ne $((CHUNKS * 2)) ]; then
        printf 'FAIL  %-14s %s barrier ENDs, expected %s -- events were dropped\n' \
          "$name" "$barriers" "$((CHUNKS * 2))" | tee -a "$SUMMARY"
        status=1
        continue
      fi
      # THE ARM CHECK. Read out of the trace, not taken on trust from the flag.
      if ! armlog=$(python3 "$HERE/check_arm.py" --arm "$arm" --trace "$trace" \
                      --stdout "$dir/stdout.log" --latencies 2>&1); then
        printf 'FAIL  %-14s the arm did not take:\n%s\n' "$name" "$armlog" | tee -a "$SUMMARY"
        status=1
        continue
      fi
      printf 'PASS  %-14s events=%-4s barrier_ends=%-4s arm=%s confirmed  %s\n' \
        "$name" "$n" "$barriers" "$arm" "$(basename "$trace")" | tee -a "$SUMMARY"
      # The observed latencies, on the card, so the control can be checked here
      # rather than only at home. Arm A at 256 B is the one to look at.
      printf '      %s\n' "$(echo "$armlog" | sed -n 's/^  lat  //p')" | tee -a "$SUMMARY"
    done
  done
done

{
  echo ""
  echo "Before sending this home, check by eye:"
  echo "  * every line above says PASS. 'arm=X confirmed' means the NoC each"
  echo "    RISC actually used was read back out of the trace and matched the"
  echo "    arm -- a run without it measured something other than what it says."
  if [ "${ARCH_NAME:-}" = "blackhole" ]; then
    echo "  * ARM A IS THE CONTROL, and the indented line under each run carries"
    echo "    that run's observed per-class latencies. Arm A's WRITE 256 mean"
    echo "    should be 375-382 and its READ 256 mean 480-510: that is the"
    echo "    2026-08-17 session reproducing. If it is not, SAY SO and send the"
    echo "    session anyway -- a control that moved is the most important result"
    echo "    in the directory."
  else
    echo "  * ARM A IS THE CONTROL, and the indented line under each run carries"
    echo "    that run's observed per-class latencies. There is NO prior card"
    echo "    session on ${ARCH_NAME:-this part}, so arm A ESTABLISHES the control"
    echo "    rather than reproducing one, and the 375-382 / 480-510 band quoted"
    echo "    in the README is BLACKHOLE's -- do not read this part against it."
    echo "    What arm A is checked against is the simulator side run at the same"
    echo "    --arch, --bytes and --chunks, at home. What you CAN check here is"
    echo "    that the repeats agree and that the two size arms differ."
  fi
  echo "  * the $REPEATS repeats of an arm agree. They are separate runs of an"
  echo "    identical program, so a big spread means something else was on the"
  echo "    card and the session is suspect. Do NOT re-run and keep the better"
  echo "    session -- say so in the handover instead."
  echo "  * the size arms DISAGREE with each other. They differ only in transfer"
  echo "    size and are supposed to separate the per-hop term from the per-byte"
  echo "    one; identical numbers mean one arm did not run the way it was meant to."
  echo ""
  echo "  Arm B swapped the NoCs -- every run above was already refused if it"
  echo "  had not, but see it for yourself:"
  echo "    for t in $OUT/runs/B-*/.logs/noc_trace_dev0_ID0.json; do"
  echo "      $HERE/check_arm.py --arm B --trace \"\$t\"; done"
  echo "    Expect READ on NOC_0 and WRITE_ on NOC_1 -- the opposite of arm A."
  echo ""
  echo "Then send the WHOLE directory, stdout logs included -- they carry the"
  echo "nocevbench-config line that records what each run was configured as:"
  echo "  rsync -av $OUT/ <home>:~/$(basename "$OUT")/"
} | tee -a "$SUMMARY"

exit $status
