#!/usr/bin/env bash
# energybench: ONE card session for ranking-level energy estimation.
#
# WHAT THIS MEASURES
# ------------------
# `tt-smi` reports BOARD power at roughly 1 Hz; the kernels here run for
# microseconds. So this does not, and cannot, weigh a single launch. It runs each
# arm back to back inside one process for tens of seconds, samples telemetry
# THROUGHOUT that run, and reports the mean:
#
#     STEADY-STATE REPEATED-KERNEL BOARD POWER, in watts, UNDER SUSTAINED LOAD,
#
# against a MEASURED IDLE BASELINE taken in the same session, and against each
# slot's OWN idle reading taken seconds before it. Board power includes DRAM,
# PHYs, ARC, PCIe and the fans, so a delta between two arms is attributable to
# Tensix activity only by argument, never by construction. Read ./README.md
# before quoting any number this produces.
#
# THE VERSION GUARD -- READ THIS ONE
# ----------------------------------
# In-slot sampling needs `tt-smi` >= 4.0.0, where the default backend changed
# from Luwen to tt-umd. On 3.0.32 every in-slot read PANICS in pyluwen
# ("Failed to map bar0_uc", BAR0's mapping being exclusive) and the harness banks
# a session of zeros that looks complete -- which is exactly what happened on
# 2026-08-13, for three full cycles. So this script REFUSES TO START on an older
# tool, and records the version it used in session.log and in every CSV row.
#
# THE OTHER HALF OF THAT BUG
# --------------------------
# The old sampler swallowed every exception and appended nothing, so a refusal
# was indistinguishable from silence and `mean([]) -> 0.0` made it "0 W". Now
# EVERY ATTEMPT IS A ROW, failures carry the tool's error text, a slot with no
# usable sample is announced at the moment it happens, and the session exits
# non-zero. A zero-sample slot can no longer be mistaken for a quiet success.
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
# Two things guard the CLOCK, which the 2026-08-13 session showed is a live
# confound (baselines of 61.7 W at 1350 MHz and ~39 W at 800 MHz in consecutive
# cycles): sysfs `tt_aiclk`/`tt_arcclk`/`tt_therm_trip_count` are sampled through
# every slot at no cost and with no device handle, and every slot is paired with
# its own idle reading taken seconds earlier rather than being differenced
# against a baseline a whole cycle away.
#
# USAGE
# -----
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   ./run_card.sh --list                 # the schedule and how long it takes
#   ./run_card.sh                        # the whole session
#   ./run_card.sh --cycles 5 --seconds 45
#   ./run_card.sh --bracket              # ALSO take post-exit samples (fallback)
#
# Results land in ~/tt_traces/energybench-session/ (--out overrides):
#   session.log     everything the run printed, including the tt-smi version
#   slots.csv       one row per slot: its window, its launch count, its STATUS
#   raw/*.pre.csv   the slot's own idle reference, taken just before it
#   raw/*.pow.csv   in-slot power samples -- every attempt, successes and failures
#   raw/*.clk.csv   sysfs clock and thermal samples, taken throughout
#   raw/*.post.csv  --bracket only: post-exit samples on a decaying edge
#   power.csv       the aggregated input to tt_sim.perf.energy_rank
#   decay.txt       --bracket only: the fitted thermal time constant
#
# Analysis happens AT HOME, not here -- it needs tt_sim/ and numpy:
#   python3 -m tt_sim.perf.energy_rank --activity activity-sim.csv \
#       --measured power.csv
#
# HARNESS TESTING WITHOUT A CARD
# ------------------------------
# ENERGYBENCH_TELEMETRY_CMD, ENERGYBENCH_BIN and ENERGYBENCH_SYSFS_ROOT override
# the three things that need hardware, so the scheduling, windowing, failure
# handling and aggregation can be exercised end to end on any box. Anything
# collected that way is NOT A MEASUREMENT and the session banner says so.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
PY="${TT_SIM_PYTHON:-python3}"

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
# 40 s, not 30. One `tt-smi -s` takes ~1.3 s, so a 30 s slot with 5 s trimmed
# from each end yields ~15 usable samples and the analysis's `samples` gate wants
# 20. Sizing the slot to clear the gate is cheaper than discovering at home that
# every row was refused.
SECONDS_PER_ARM=40
SETTLE=5
BASELINE_SECONDS=40
# Back to back. tt-smi's own ~1.3 s round trip is the real sampling period; an
# added sleep only buys fewer samples.
SAMPLE_INTERVAL=0
# The measured cost of one `tt-smi -s`, used only for the wall estimate and the
# expected-sample count.
SMI_COST=1.4
PRE_SAMPLES=2
POST_SAMPLES=6
BRACKET=0
# Which entry of tt-smi's `device_info` to read. The benchmark opens device 0,
# so a box with more than one card must be told if that is not index 0 here --
# sampling the wrong chip gives a perfectly steady idle trace for every arm,
# which the `spread` gate refuses rather than reports.
CHIP="${ENERGYBENCH_CHIP:-0}"
SYSFS_ROOT="${ENERGYBENCH_SYSFS_ROOT:-/sys/class/tenstorrent}"
OUT="${HOME}/tt_traces/energybench-session"
LIST=0
NOT_A_MEASUREMENT=0
ALLOW_OLD_SMI=0

TELEMETRY_CMD="${ENERGYBENCH_TELEMETRY_CMD:-tt-smi -s}"
[ -n "${ENERGYBENCH_TELEMETRY_CMD:-}" ] && NOT_A_MEASUREMENT=1
[ -n "${ENERGYBENCH_BIN:-}" ] && NOT_A_MEASUREMENT=1
[ -n "${ENERGYBENCH_SYSFS_ROOT:-}" ] && NOT_A_MEASUREMENT=1

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
    --pre-samples) PRE_SAMPLES="$2"; shift 2 ;;
    --post-samples) POST_SAMPLES="$2"; shift 2 ;;
    --bracket) BRACKET=1; shift ;;
    --chip) CHIP="$2"; shift 2 ;;
    --sysfs-root) SYSFS_ROOT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --inner) IFS='=' read -r _a _n <<< "$2"; INNER[$_a]="$_n"; shift 2 ;;
    --allow-old-tt-smi) ALLOW_OLD_SMI=1; shift ;;
    --list) LIST=1; shift ;;
    -h|--help) sed -n '2,85p' "$0" | sed 's/^# \?//'; exit 0 ;;
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
probe_cost="$($PY -c "print(int(($PRE_SAMPLES + $BRACKET * $POST_SAMPLES) * $SMI_COST))")"
per_cycle=$(( BASELINE_SECONDS + n_slots * SECONDS_PER_ARM + (n_slots + 1) * probe_cost ))
total=$(( CYCLES * per_cycle ))
expect_samples="$($PY -c "
window = $SECONDS_PER_ARM - 2 * $SETTLE
print(max(0, int(window / max($SMI_COST + $SAMPLE_INTERVAL, 1e-6))))")"

echo "=== energybench card session ==="
[ "$NOT_A_MEASUREMENT" = 1 ] && echo "*** NOT-A-MEASUREMENT: telemetry, sysfs and/or binary are stubbed ***"
echo "quantity      : in-slot board power under SUSTAINED LOAD (not a decaying edge)"
echo "arms          : $ARMS"
echo "scales        : $SCALES"
echo "control       : $CONTROL_ARM at inner=$CONTROL_INNER (run twice per cycle)"
echo "cycles        : $CYCLES"
echo "per arm       : ${SECONDS_PER_ARM}s (${SETTLE}s trimmed from each end) -> ~${expect_samples} samples"
echo "baseline      : ${BASELINE_SECONDS}s per cycle, device open, nothing launching"
echo "pre-idle      : ${PRE_SAMPLES} samples before every slot, device free (the paired reference)"
[ "$BRACKET" = 1 ] && echo "bracket       : ${POST_SAMPLES} post-exit samples per slot -- FALLBACK, a decaying edge"
echo "slots / cycle : $((n_slots + 1))  (baseline + $n_slots arm runs)"
echo "wall estimate : ~$((total / 60)) min (${CYCLES}x${per_cycle}s), plus build and device open/close"
echo "output        : $OUT"
echo ""
echo "schedule per cycle:"
printf '  %-3s %s\n' "0" "baseline (${BASELINE_SECONDS}s)"
for i in "${!SLOT_LABELS[@]}"; do
  printf '  %-3s %s (%ss, inner=%s)\n' "$((i + 1))" "${SLOT_LABELS[$i]}" "$SECONDS_PER_ARM" "${SLOT_INNER[$i]}"
done
if [ "$expect_samples" -lt 20 ]; then
  echo ""
  echo "WARNING: ~${expect_samples} samples per slot is under the analysis's default" >&2
  echo "--min-samples 20, so every row would be refused at home. Raise --seconds." >&2
fi
[ "$LIST" = 1 ] && exit 0

command -v tt-smi >/dev/null 2>&1 || [ -n "${ENERGYBENCH_TELEMETRY_CMD:-}" ] || {
  echo "tt-smi not found on PATH -- this session cannot measure anything without it" >&2
  exit 2
}

# --- the version guard --------------------------------------------------
# Before anything else, and before any card time is spent. An old tt-smi does
# not degrade: it records zeros for every slot that has the device open.
VERSION_OUT="$($PY "$HERE/telemetry_sample.py" version --command "$TELEMETRY_CMD" 2>&1)"
rc=$?
TT_SMI_VERSION="$(printf '%s\n' "$VERSION_OUT" | sed -n 's/^tt_smi_version=//p' | head -n1)"
printf '%s\n' "$VERSION_OUT"
if [ "$rc" != 0 ]; then
  if [ "$ALLOW_OLD_SMI" = 1 ]; then
    echo "*** --allow-old-tt-smi: proceeding anyway. Expect zeros. ***" >&2
  else
    echo "" >&2
    echo "REFUSING TO START. Upgrade tt-smi, or pass --allow-old-tt-smi if you" >&2
    echo "genuinely want a session that is expected to record nothing." >&2
    exit 4
  fi
fi

# --- the sysfs channel --------------------------------------------------
if ! $PY "$HERE/sysfs_sample.py" --check --root "$SYSFS_ROOT" --chip "$CHIP"; then
  echo "" >&2
  echo "The clock/thermal channel is unavailable, so this session could not show" >&2
  echo "that its clock held still -- the confound that made the 2026-08-13" >&2
  echo "baselines swing 42%. The analysis will refuse every row." >&2
  exit 5
fi

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
echo "tt_smi_version=$TT_SMI_VERSION" | tee -a "$LOG"
echo "session,cycle,slot,label,arm,inner,status,start_s,end_s,launches,wall_s,launches_per_s,pre_file,pow_file,clk_file,post_file,tt_smi_version" > "$SLOTS"

now() { $PY -c 'import time; print(f"{time.time():.6f}")'; }

# --- the samplers -------------------------------------------------------
# Two independent streams, started together and killed together. The power
# stream opens the device (needs tt-smi >= 4.0.0); the sysfs stream opens
# nothing and so cannot be refused. Both write one row per attempt.
POW_PID=""
CLK_PID=""
start_samplers() {
  local pow="$1" clk="$2"
  $PY "$HERE/telemetry_sample.py" watch --out "$pow" --command "$TELEMETRY_CMD" \
      --chip "$CHIP" --phase run --interval "$SAMPLE_INTERVAL" >/dev/null 2>&1 &
  POW_PID=$!
  $PY "$HERE/sysfs_sample.py" --out "$clk" --root "$SYSFS_ROOT" --chip "$CHIP" \
      --interval 0.25 >/dev/null 2>&1 &
  CLK_PID=$!
}
stop_samplers() {
  for pid in "$POW_PID" "$CLK_PID"; do
    [ -n "$pid" ] || continue
    kill -TERM "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
  done
  POW_PID=""; CLK_PID=""
}
trap 'stop_samplers; exit 130' INT TERM

# Successful samples in a stream, counted from the `ok` column. This is what
# makes a dead sampler loud AT THE MOMENT IT HAPPENS rather than at home.
count_ok() {
  [ -f "$1" ] || { echo 0; return; }
  awk -F, 'NR>1 && $7==1' "$1" | wc -l
}

record() {
  echo "$SESSION,$1,$2,$3,$4,$5,$6,$7,$8,$9,${10},${11},${12},${13},${14},${15},$TT_SMI_VERSION" >> "$SLOTS"
}

status=0
bad_slots=()

# One slot: idle reference, samplers, the work, samplers down, the verdict.
# $1 cycle  $2 slot  $3 label  $4 arm  $5 inner  $6 seconds  $7 "" for the
# baseline or the command to run.
run_slot() {
  local cycle="$1" slot="$2" label="$3" arm="$4" inner="$5" secs="$6" is_arm="$7"
  local stem="raw/c${cycle}-s${slot}-${label}"
  local pre="$stem.pre.csv" pow="$stem.pow.csv" clk="$stem.clk.csv" post=""
  local slot_status=ok launches=0 wall="$secs" rate=0

  # The device is free right now: this is the slot's own idle reference, taken
  # seconds before it rather than a whole cycle away.
  $PY "$HERE/telemetry_sample.py" probe --out "$OUT/$pre" --command "$TELEMETRY_CMD" \
      --chip "$CHIP" --phase pre --count "$PRE_SAMPLES" >> "$LOG" 2>&1

  start_samplers "$OUT/$pow" "$OUT/$clk"
  local t0 t1 out rc line
  t0="$(now)"
  if [ -n "$is_arm" ]; then
    # CreateKernel resolves "kernels/..." relative to the WORKING DIRECTORY,
    # and the kernels live under $SRC. Run from there or every launch aborts.
    out="$(cd "$SRC" && "$BIN" --arm "$arm" --inner "$inner" --seconds "$secs" \
             --label "$label" --csv "$OUT/launches.csv" 2>&1)"
    rc=$?
  else
    sleep "$secs"
    out=""; rc=0
  fi
  t1="$(now)"
  local t_exit="$t1"
  stop_samplers
  [ -n "$out" ] && printf '%s\n' "$out" >> "$LOG"

  if [ "$BRACKET" = 1 ]; then
    post="$stem.post.csv"
    $PY "$HERE/telemetry_sample.py" probe --out "$OUT/$post" --command "$TELEMETRY_CMD" \
        --chip "$CHIP" --phase post --count "$POST_SAMPLES" --t0 "$t_exit" >> "$LOG" 2>&1
  fi

  if [ -n "$is_arm" ]; then
    line="$(printf '%s\n' "$out" | grep -m1 '^ENERGYBENCH ')"
    if [ "$rc" != 0 ] || [ -z "$line" ]; then
      slot_status="run_failed"
      echo "  [slot $slot] $label RUN FAILED (rc=$rc)" | tee -a "$LOG"
    else
      launches="$(printf '%s\n' "$line" | sed -n 's/.*launches=\([0-9]*\).*/\1/p')"
      wall="$(printf '%s\n' "$line" | sed -n 's/.*wall_s=\([0-9.]*\).*/\1/p')"
      rate="$(printf '%s\n' "$line" | sed -n 's/.*launches_per_s=\([0-9.]*\).*/\1/p')"
    fi
  fi

  # The verdict, here and now. A slot with no telemetry is announced the moment
  # it happens: the 2026-08-13 session ran three full cycles of these without a
  # word, because nothing looked at the sample count until the analysis at home.
  local n_pow n_clk
  n_pow="$(count_ok "$OUT/$pow")"
  n_clk="$([ -f "$OUT/$clk" ] && { awk 'NR>1' "$OUT/$clk" | wc -l; } || echo 0)"
  if [ "$n_pow" = 0 ]; then
    [ "$slot_status" = ok ] && slot_status="no_telemetry"
    echo "  [slot $slot] $label *** NO TELEMETRY: 0 power samples ***" | tee -a "$LOG"
    awk -F, 'NR>1 && $7!=1 {print "        " $13; exit}' "$OUT/$pow" 2>/dev/null | tee -a "$LOG"
  fi
  if [ "$n_clk" = 0 ]; then
    [ "$slot_status" = ok ] && slot_status="no_clock"
    echo "  [slot $slot] $label *** NO CLOCK RECORD: 0 sysfs samples ***" | tee -a "$LOG"
  fi

  # ALWAYS recorded, whatever happened. A slot that failed and left no manifest
  # row is how cycle 2 of the 2026-08-13 session lost `idle-0` without a trace.
  record "$cycle" "$slot" "$label" "$arm" "$inner" "$slot_status" \
         "$t0" "$t1" "$launches" "$wall" "$rate" "$pre" "$pow" "$clk" "$post"

  if [ "$slot_status" != ok ]; then
    status=1
    bad_slots+=("cycle $cycle slot $slot $label: $slot_status")
  else
    if [ -n "$is_arm" ]; then
      echo "  [slot $slot] $label  launches=$launches  rate=${rate}/s  power_samples=$n_pow  clk=$n_clk" | tee -a "$LOG"
    else
      echo "  [slot $slot] baseline ${secs}s  power_samples=$n_pow  clk=$n_clk" | tee -a "$LOG"
    fi
  fi
}

for cycle in $(seq 0 $((CYCLES - 1))); do
  echo "" | tee -a "$LOG"
  echo "===== cycle $cycle / $((CYCLES - 1)) =====" | tee -a "$LOG"

  # slot 0: the baseline. Nothing launches; this is P_static. It is measured the
  # SAME WAY as every arm -- same samplers, same trim, same pre-idle probe -- so
  # the two are the same quantity and the difference between them is a
  # difference in the board, not in the instrument.
  run_slot "$cycle" 0 baseline baseline 0 "$BASELINE_SECONDS" ""

  for i in "${!SLOT_ARMS[@]}"; do
    run_slot "$cycle" "$((i + 1))" "${SLOT_LABELS[$i]}" "${SLOT_ARMS[$i]}" \
             "${SLOT_INNER[$i]}" "$SECONDS_PER_ARM" arm
  done
done

echo "" | tee -a "$LOG"
$PY "$HERE/aggregate_power.py" --slots "$SLOTS" --out "$OUT/power.csv" --settle "$SETTLE" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || status=1

if [ "$BRACKET" = 1 ]; then
  echo "" | tee -a "$LOG"
  # Pool the control arm's post-exit samples across cycles -- dt is measured from
  # each slot's own exit, so pooling is legitimate and buys the point count a
  # single slot cannot.
  cat "$OUT"/raw/*-"$CONTROL_ARM"-*.post.csv > "$OUT/decay.csv" 2>/dev/null
  $PY "$HERE/telemetry_sample.py" fit --samples "$OUT/decay.csv" \
      --report "$OUT/decay.txt" --json "$OUT/decay.json" 2>&1 | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "=== handover ===" | tee -a "$LOG"
[ "$NOT_A_MEASUREMENT" = 1 ] && echo "*** NOT-A-MEASUREMENT: stubbed telemetry, sysfs and/or binary ***" | tee -a "$LOG"
[ "$ALLOW_OLD_SMI" = 1 ] && echo "*** --allow-old-tt-smi WAS USED: tt-smi $TT_SMI_VERSION may not read a busy device, so every in-slot sample in this session is suspect ***" | tee -a "$LOG"
if [ "${#bad_slots[@]}" != 0 ]; then
  echo "*** ${#bad_slots[@]} SLOT(S) DID NOT PRODUCE A CLEAN MEASUREMENT ***" | tee -a "$LOG"
  for bad in "${bad_slots[@]}"; do echo "    $bad" | tee -a "$LOG"; done
  echo "    Do not analyse this session as if it were complete. Say so in the" | tee -a "$LOG"
  echo "    handover; the analysis will refuse it anyway." | tee -a "$LOG"
fi
echo "tt-smi version : $TT_SMI_VERSION" | tee -a "$LOG"
echo "Send back the WHOLE directory: $OUT" | tee -a "$LOG"
echo "What it measures: steady-state repeated-kernel BOARD power UNDER SUSTAINED" | tee -a "$LOG"
echo "LOAD, sampled in slot, against an in-session idle baseline and each slot's" | tee -a "$LOG"
echo "own pre-idle reading. NOT the energy of one launch." | tee -a "$LOG"
[ "$BRACKET" = 1 ] && echo "power_bracket_w is the FALLBACK: post-exit, a decaying edge, a different quantity." | tee -a "$LOG"
echo "Analyse at home:" | tee -a "$LOG"
echo "  python3 -m tt_sim.perf.energy_rank --activity activity-sim.csv --measured $OUT/power.csv" | tee -a "$LOG"
exit $status
