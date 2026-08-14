#!/usr/bin/env bash
# The SIMULATOR half of the energy harness: run each energybench arm against
# tt-sim under the counter writer and reduce each run to a per-launch activity
# vector.
#
# This produces the model's INPUT. It fits nothing and predicts nothing -- the
# coefficients do not exist yet and will be fitted to silicon by
# `tt_sim.perf.energy_rank` once `run_card.sh` has been run at a card.
#
#   TT_METAL_HOME=/path/to/tt-metal ./perfbench/energybench/run_sim_activity.sh
#   ... --arch wormhole --out /tmp/activity.csv
#   ... --arms "idle rv noc"          # a subset
#   ... --scale 4                     # multiply every arm's inner count
#
# The cost model is ON by default and should stay on: every `*_busy_cycles` and
# `*_stall_cycles` term is ABSENT, not zero, without it (tt_sim/trace/counters.py
# says so in its module docstring), so a cost-model-off activity matrix has whole
# columns of zeros that no fit can use. `--no-cost-model` is there to show that.
#
# Run in a NORMAL shell -- a sandbox that kills the spawned server will hang the
# host binary.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

ARCH="${TT_SIM_ARCH:-blackhole}"
OUT="$HERE/activity-sim.csv"
# Every arm is run at more than one inner count, and that is not a nicety. The
# fit spends one coefficient per activity term plus one for the launch
# machinery, and leave-one-out needs a spare, so the number of terms it can
# identify is (workloads - 2). Five arms would buy three terms for a five-term
# model. Two scales per arm buys nine workloads and seven terms, and the
# within-arm ratio is itself the sharpest ranking test in the set.
SCALES="1 4"
ITERS=1
COST_MODEL=1
KEEP=0
# The default inner counts are SMOKE sizes: the simulator runs a few tens of
# thousands of cycles per second where hardware runs a billion, so the numbers
# that make a card session meaningful would take hours here. They are chosen so
# each arm's characteristic term dominates its own vector, which is all the
# simulator side has to demonstrate -- the card side scales them up.
ARMS="idle rv noc mm sfpu"
declare -A INNER=([idle]=0 [rv]=64 [noc]=8 [mm]=8 [sfpu]=8)

while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --scale|--scales) SCALES="$2"; shift 2 ;;
    --inner) IFS='=' read -r _a _n <<< "$2"; INNER[$_a]="$_n"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --no-cost-model) COST_MODEL=0; shift ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"

# --- the reduction's imports, BEFORE any arm runs ------------------------
# `tt_sim.perf.energy_activity` pulls in `tt_sim.trace.report` -> `tt_sim.trace.dwarf`
# -> `elftools`, and an interpreter without pyelftools throws only when the FIRST
# ARM HAS ALREADY RUN -- a full simulator boot per arm, thrown away, with the
# traceback printed and the loop carrying on to the next arm. That is the same
# disease as the card side's silent slots: the run "finishes", exits 0 and leaves
# a short activity CSV that `energy_rank` then fits fewer workloads from than the
# operator thinks it did. So the imports are proved first, and cost nothing.
PY="${TT_SIM_PYTHON:-python3}"
if ! PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PY" - <<'EOF'
import importlib
import sys

missing = []
# `tt_sim.perf.energy_activity` imports at the top level and would pass on its
# own: the chain that actually breaks is deferred to reduction time, inside
# main(), which is precisely why the failure used to arrive one simulator boot
# too late. So the deferred import is named here explicitly.
for module in (
    "numpy",
    "elftools.elf.elffile",
    "tt_sim.perf.energy_activity",
    "tt_sim.trace.report",
):
    try:
        importlib.import_module(module)
    except ImportError as exc:
        missing.append(f"  {module}: {exc}")
if missing:
    print("the activity reduction cannot import what it needs:", file=sys.stderr)
    print("\n".join(missing), file=sys.stderr)
    sys.exit(1)
print(f"reduction prerequisites ok ({sys.executable})")
EOF
then
  echo "" >&2
  echo "REFUSING TO START: $PY cannot run the reduction step, so every arm would" >&2
  echo "boot the simulator, produce counters and then fail to reduce them." >&2
  echo "pyelftools is the usual one missing (tt_sim/trace/dwarf.py needs it)." >&2
  echo "Use an interpreter that has the trace dependencies, e.g." >&2
  echo "  TT_SIM_PYTHON=/path/to/venv/bin/python3 $0 ..." >&2
  exit 4
fi

WORK="$(mktemp -d)"
cleanup() { [ "$KEEP" = 1 ] || rm -rf "$WORK"; }
trap cleanup EXIT

rm -f "$OUT"
echo "# energybench activity vectors -- arch=$ARCH cost_model=$COST_MODEL scales='$SCALES'"
echo "# counters under $WORK"

status=0
seen=" "
expected=""
failed=""
for scale in $SCALES; do
for arm in $ARMS; do
  base="${INNER[$arm]:-0}"
  inner=$(( base * scale ))
  label="$arm-$inner"
  # `idle` has no inner loop, so every scale gives the same workload. Running it
  # again would put a duplicate row in the design matrix, which is a free rank
  # the fit has not earned.
  case "$seen" in *" $label "*) continue ;; esac
  seen="$seen$label "
  expected="$expected$label "
  cdir="$WORK/$label"
  log="$WORK/$label.log"
  echo ""
  echo "=== $label ==============================================="
  env TT_SIM_ARCH="$ARCH" \
      TT_SIM_COST_MODEL="$COST_MODEL" \
      TT_SIM_TRACE_COUNTERS="$cdir" \
      "$REPO/perfbench/run.sh" energybench -- \
        --arm "$arm" --inner "$inner" --iters "$ITERS" --label "$label" \
        > "$log" 2>&1
  rc=$?
  if [ "$rc" != 0 ] || ! grep -q "Completed successfully on the device" "$log"; then
    echo "*** $label FAILED TO RUN (rc=$rc); tail of $log:"
    tail -n 20 "$log"
    status=1
    failed="$failed$label "
    continue
  fi
  grep -h "^ENERGYBENCH " "$log"

  # A reduction that throws leaves this label out of the CSV entirely, and
  # `energy_rank` drops an unmatched label "with a note" -- so a hole here is a
  # design matrix quietly one workload shorter than the operator believes. Say so
  # at the time, name it again in the summary, and exit non-zero.
  if ! PYTHONPATH="$REPO:${PYTHONPATH:-}" "$PY" \
      -m tt_sim.perf.energy_activity \
        --counters "$cdir" --label "$label" --arm "$arm" --arch "$ARCH" \
        --inner "$inner" --launches "$ITERS" --cost-model "$COST_MODEL" \
        --out "$OUT" --append; then
    echo "*** $label REDUCTION FAILED: it ran, and produced NO ACTIVITY VECTOR."
    status=1
    failed="$failed$label "
  fi
done
done

# The activity CSV is checked against the schedule rather than trusted, for the
# same reason `run_card.sh` writes a manifest row for a slot that crashed: a
# missing label must not be indistinguishable from a schedule that never had it.
missing=""
for label in $expected; do
  if ! grep -q "^$label," "$OUT" 2>/dev/null; then
    missing="$missing$label "
  fi
done

echo ""
echo "activity vectors -> $OUT"
if [ -n "$failed" ] || [ -n "$missing" ]; then
  status=1
  echo ""
  echo "=== HANDOVER: THIS ACTIVITY MATRIX IS INCOMPLETE ==================="
  [ -n "$failed" ] && echo "arms that failed:   $failed"
  [ -n "$missing" ] && echo "labels not in CSV:  $missing"
  echo "Every arm is a column of the design matrix. Do not fit this against a"
  echo "card session and do not 'just use what came back' -- energy_rank will"
  echo "drop the unmatched labels with a note and fit a smaller design than the"
  echo "schedule claims. Fix the failure and re-run the missing arms."
  echo "==================================================================="
fi
exit $status
