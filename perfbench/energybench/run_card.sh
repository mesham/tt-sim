#!/usr/bin/env bash
# energybench: ONE card session for ranking-level energy estimation.
#
# WHAT THIS MEASURES
# ------------------
# `tt-smi` reports BOARD power at roughly 1 Hz; the kernels here run for
# microseconds. So this does not, and cannot, weigh a single launch. It runs each
# arm back to back inside one process for tens of seconds, samples telemetry
# throughout, and reports the mean:
#
#     STEADY-STATE REPEATED-KERNEL BOARD POWER, in watts,
#
# against a MEASURED IDLE BASELINE taken in the same session. Board power
# includes DRAM, PHYs, ARC, PCIe and the fans, so a delta between two arms is
# attributable to Tensix activity only by argument, never by construction. Read
# ./README.md before quoting any number this produces.
#
# THE DISCIPLINE
# --------------
# Arms are INTERLEAVED, not blocked: one cycle runs every arm once, and the
# session runs several cycles. This project has been bitten by drift before, and
# running all of A then all of B turns a slow thermal ramp into a fake result.
# One arm is additionally run TWICE PER CYCLE, in two different slots, under the
# label `<arm>__control`. Its two readings must agree: that is the verified-zero
# control, and the analysis REFUSES the whole session if it does not.
#
# USAGE
# -----
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   ./run_card.sh --list                 # the schedule and how long it takes
#   ./run_card.sh                        # the whole session
#   ./run_card.sh --cycles 5 --seconds 45
#
# Results land in ~/tt_traces/energybench-session/ (--out overrides):
#   session.log     everything the run printed
#   slots.csv       one row per arm run: its window, its launch count
#   raw/*.csv       the telemetry samples, per slot, with timestamps
#   power.csv       the aggregated input to tt_sim.perf.energy_rank
#
# Analysis happens AT HOME, not here -- it needs tt_sim/ and numpy:
#   python3 -m tt_sim.perf.energy_rank --activity activity-sim.csv \
#       --measured power.csv
#
# HARNESS TESTING WITHOUT A CARD
# ------------------------------
# ENERGYBENCH_TELEMETRY_CMD and ENERGYBENCH_BIN override the two things that
# need hardware, so the scheduling, windowing and aggregation can be exercised
# end to end on any box. Anything collected that way is NOT A MEASUREMENT and
# the session log says so.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

ARMS="idle rv noc mm sfpu"
# Inner counts for a CARD. They are chosen so each arm runs for roughly tens of
# microseconds per launch -- long enough that the arm's own work dominates the
# launch machinery, short enough that thousands of launches fit in the window
# and the ~1 Hz sampler sees a genuine steady state.
declare -A INNER=([idle]=0 [rv]=200000 [noc]=4096 [mm]=4096 [sfpu]=4096)
# Every arm runs at more than one inner count. The fit spends one coefficient
# per activity term plus one for the launch machinery, and leave-one-out needs a
# degree of freedom spared, so it can identify (workloads - 2) terms. Five arms
# would buy three. Two scales buys nine workloads and seven terms -- and the
# within-arm ratio is the sharpest ranking test in the set, because the two
# points differ in one thing only.
SCALES="1 4"
CONTROL_ARM="noc"
CYCLES=3
SECONDS_PER_ARM=30
SETTLE=5
BASELINE_SECONDS=30
SAMPLE_INTERVAL=0.5
# Which entry of tt-smi's `device_info` to read. The benchmark opens device 0,
# so a box with more than one card must be told if that is not index 0 here --
# sampling the wrong chip gives a perfectly steady idle trace for every arm,
# which the `spread` gate refuses rather than reports.
CHIP="${ENERGYBENCH_CHIP:-0}"
OUT="${HOME}/tt_traces/energybench-session"
LIST=0
NOT_A_MEASUREMENT=0

TELEMETRY_CMD="${ENERGYBENCH_TELEMETRY_CMD:-tt-smi -s}"
[ -n "${ENERGYBENCH_TELEMETRY_CMD:-}" ] && NOT_A_MEASUREMENT=1
[ -n "${ENERGYBENCH_BIN:-}" ] && NOT_A_MEASUREMENT=1

while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2 ;;
    --scale|--scales) SCALES="$2"; shift 2 ;;
    --control) CONTROL_ARM="$2"; shift 2 ;;
    --cycles) CYCLES="$2"; shift 2 ;;
    --seconds) SECONDS_PER_ARM="$2"; shift 2 ;;
    --settle) SETTLE="$2"; shift 2 ;;
    --baseline-seconds) BASELINE_SECONDS="$2"; shift 2 ;;
    --interval) SAMPLE_INTERVAL="$2"; shift 2 ;;
    --chip) CHIP="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --inner) IFS='=' read -r _a _n <<< "$2"; INNER[$_a]="$_n"; shift 2 ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,45p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set. This is the CARD session; it must run against" >&2
  echo "hardware. Unset it, or use ./run_sim_activity.sh for the simulator side." >&2
  exit 2
fi

case " $ARMS " in
  *" $CONTROL_ARM "*) ;;
  *) echo "--control $CONTROL_ARM is not one of the arms ($ARMS)" >&2; exit 2 ;;
esac

# The schedule. The control sits in the LAST slot of each cycle while its twin
# sits in its natural position, so the pair straddles the cycle: if the board
# drifts within a cycle, the control is where it shows up.
SLOT_LABELS=()
SLOT_ARMS=()
SLOT_INNER=()
seen=" "
for scale in $SCALES; do
  for arm in $ARMS; do
    inner=$(( ${INNER[$arm]} * scale ))
    label="$arm-$inner"
    # `idle` has no inner loop, so it is the same workload at every scale.
    case "$seen" in *" $label "*) continue ;; esac
    seen="$seen$label "
    SLOT_LABELS+=("$label")
    SLOT_ARMS+=("$arm")
    SLOT_INNER+=("$inner")
  done
done
CONTROL_INNER=$(( ${INNER[$CONTROL_ARM]} * ${SCALES%% *} ))
SLOT_LABELS+=("$CONTROL_ARM-${CONTROL_INNER}__control")
SLOT_ARMS+=("$CONTROL_ARM")
SLOT_INNER+=("$CONTROL_INNER")

n_slots=${#SLOT_ARMS[@]}
per_cycle=$(( BASELINE_SECONDS + n_slots * SECONDS_PER_ARM ))
total=$(( CYCLES * per_cycle ))

echo "=== energybench card session ==="
[ "$NOT_A_MEASUREMENT" = 1 ] && echo "*** NOT-A-MEASUREMENT: telemetry and/or binary are stubbed ***"
echo "arms          : $ARMS"
echo "scales        : $SCALES"
echo "control       : $CONTROL_ARM at inner=$CONTROL_INNER (run twice per cycle)"
echo "cycles        : $CYCLES"
echo "per arm       : ${SECONDS_PER_ARM}s (${SETTLE}s trimmed from each end)"
echo "baseline      : ${BASELINE_SECONDS}s per cycle, device open, nothing launching"
echo "slots / cycle : $((n_slots + 1))  (baseline + $n_slots arm runs)"
echo "wall estimate : ~$((total / 60)) min ${CYCLES}x${per_cycle}s, plus build and device open/close"
echo "output        : $OUT"
echo ""
echo "schedule per cycle:"
printf '  %-3s %s\n' "0" "baseline (${BASELINE_SECONDS}s)"
for i in "${!SLOT_LABELS[@]}"; do
  printf '  %-3s %s (%ss, inner=%s)\n' "$((i + 1))" "${SLOT_LABELS[$i]}" "$SECONDS_PER_ARM" "${SLOT_INNER[$i]}"
done
[ "$LIST" = 1 ] && exit 0

command -v tt-smi >/dev/null 2>&1 || [ -n "${ENERGYBENCH_TELEMETRY_CMD:-}" ] || {
  echo "tt-smi not found on PATH -- this session cannot measure anything without it" >&2
  exit 2
}

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
# energybench drives detail::LaunchProgram directly. On a card, fast dispatch
# is live by default and collides with it -- every launch aborts (rc=134).
# Slow dispatch is what every other card runner here uses; see perfbench/run.sh.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"

BIN="${ENERGYBENCH_BIN:-$SRC/build/energybench}"
if [ -z "${ENERGYBENCH_BIN:-}" ] && [ ! -x "$BIN" ]; then
  echo "[build] $SRC"
  ( cd "$SRC" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j >/dev/null ) || {
    echo "build failed" >&2; exit 1; }
fi

mkdir -p "$OUT/raw" || exit 2
LOG="$OUT/session.log"
SLOTS="$OUT/slots.csv"
SESSION="$(date '+%Y%m%dT%H%M%S')"
: > "$LOG"
echo "session,cycle,slot,label,arm,inner,start_s,end_s,launches,wall_s,launches_per_s,samples_file" > "$SLOTS"

now() { date +%s.%N; }

# --- the sampler --------------------------------------------------------
# Snapshots telemetry into a timestamped CSV until it is killed. Timestamps are
# taken by the shell, not read out of tt-smi, so a slot's window and its samples
# are on the same clock.
SAMPLER_PID=""
start_sampler() {
  local dest="$1"
  echo "t,aiclk,voltage,current,power,temp" > "$dest"
  (
    while true; do
      ts="$(now)"
      $TELEMETRY_CMD 2>/dev/null | ENERGYBENCH_TS="$ts" ENERGYBENCH_CHIP="$CHIP" python3 -c '
import json, os, sys
ts = os.environ["ENERGYBENCH_TS"]
chip = int(os.environ.get("ENERGYBENCH_CHIP", "0"))
try:
    t = json.load(sys.stdin)["device_info"][chip]["telemetry"]
    print(",".join([ts, str(t["aiclk"]).strip(), str(t["voltage"]).strip(),
                    str(t["current"]).strip(), str(t["power"]).strip(),
                    str(t["asic_temperature"]).strip()]))
except Exception:
    pass
' >> "$dest"
      sleep "$SAMPLE_INTERVAL"
    done
  ) &
  SAMPLER_PID=$!
}
stop_sampler() {
  [ -n "$SAMPLER_PID" ] || return 0
  kill "$SAMPLER_PID" 2>/dev/null
  wait "$SAMPLER_PID" 2>/dev/null
  SAMPLER_PID=""
}
trap 'stop_sampler; exit 130' INT TERM

record() {
  local cycle="$1" slot="$2" label="$3" arm="$4" inner="$5"
  local start="$6" end="$7" launches="$8" wall="$9" rate="${10}" file="${11}"
  echo "$SESSION,$cycle,$slot,$label,$arm,$inner,$start,$end,$launches,$wall,$rate,$file" >> "$SLOTS"
}

status=0
for cycle in $(seq 0 $((CYCLES - 1))); do
  echo "" | tee -a "$LOG"
  echo "===== cycle $cycle / $((CYCLES - 1)) =====" | tee -a "$LOG"

  # --- slot 0: the baseline. Nothing launches; this is P_static. ---------
  raw="raw/c${cycle}-s0-baseline.csv"
  start_sampler "$OUT/$raw"
  t0="$(now)"
  sleep "$BASELINE_SECONDS"
  t1="$(now)"
  stop_sampler
  record "$cycle" 0 baseline baseline 0 "$t0" "$t1" 0 "$BASELINE_SECONDS" 0 "$raw"
  echo "  [slot 0] baseline ${BASELINE_SECONDS}s" | tee -a "$LOG"

  for i in "${!SLOT_ARMS[@]}"; do
    slot=$((i + 1))
    arm="${SLOT_ARMS[$i]}"
    label="${SLOT_LABELS[$i]}"
    inner="${SLOT_INNER[$i]}"
    raw="raw/c${cycle}-s${slot}-${label}.csv"
    csv="$OUT/launches.csv"

    start_sampler "$OUT/$raw"
    t0="$(now)"
    # CreateKernel resolves "kernels/..." relative to the WORKING DIRECTORY,
    # and the kernels live under $SRC. Run from there or every launch aborts.
    out="$(cd "$SRC" && "$BIN" --arm "$arm" --inner "$inner" --seconds "$SECONDS_PER_ARM" \
             --label "$label" --csv "$csv" 2>&1)"
    rc=$?
    t1="$(now)"
    stop_sampler
    printf '%s\n' "$out" >> "$LOG"

    line="$(printf '%s\n' "$out" | grep -m1 '^ENERGYBENCH ')"
    if [ "$rc" != 0 ] || [ -z "$line" ]; then
      echo "  [slot $slot] $label FAILED (rc=$rc)" | tee -a "$LOG"
      status=1
      continue
    fi
    launches="$(printf '%s\n' "$line" | sed -n 's/.*launches=\([0-9]*\).*/\1/p')"
    wall="$(printf '%s\n' "$line" | sed -n 's/.*wall_s=\([0-9.]*\).*/\1/p')"
    rate="$(printf '%s\n' "$line" | sed -n 's/.*launches_per_s=\([0-9.]*\).*/\1/p')"
    record "$cycle" "$slot" "$label" "$arm" "$inner" "$t0" "$t1" "$launches" "$wall" "$rate" "$raw"
    echo "  [slot $slot] $label  launches=$launches  rate=${rate}/s" | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
python3 "$HERE/aggregate_power.py" --slots "$SLOTS" --out "$OUT/power.csv" --settle "$SETTLE" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== handover ===" | tee -a "$LOG"
[ "$NOT_A_MEASUREMENT" = 1 ] && echo "*** NOT-A-MEASUREMENT: stubbed telemetry and/or binary ***" | tee -a "$LOG"
echo "Send back the WHOLE directory: $OUT" | tee -a "$LOG"
echo "What it measures: steady-state repeated-kernel BOARD power, against an" | tee -a "$LOG"
echo "in-session idle baseline. NOT the energy of one launch." | tee -a "$LOG"
echo "Analyse at home:" | tee -a "$LOG"
echo "  python3 -m tt_sim.perf.energy_rank --activity activity-sim.csv --measured $OUT/power.csv" | tee -a "$LOG"
exit $status
