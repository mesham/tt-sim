#!/usr/bin/env bash
# Exercise run_card.sh end to end WITHOUT A CARD.
#
# The two things that need hardware -- the telemetry command and the benchmark
# binary -- are stubbed, so what is under test is everything else: the interleave
# schedule, the sampler lifecycle, the per-slot windows, the launch-rate capture,
# and the settle-trimmed aggregation in aggregate_power.py.
#
# The stub board has a per-arm power so the aggregation has something to
# separate, plus a slow linear DRIFT across the whole session. The drift is the
# point: it is what an interleaved schedule is supposed to survive and a blocked
# one is not, and it is what the verified-zero control is there to bound.
#
#   perfbench/energybench/run_card_stub_test.sh
#
# Nothing this produces is a measurement, and the session banner says so.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fails=0
check() {
  local what="$1" got="$2" want="$3"
  if [ "$got" = "$want" ]; then
    echo "  ok   $what ($got)"
  else
    echo "  FAIL $what: got '$got', want '$want'"
    fails=$((fails + 1))
  fi
}

# --- the stub board -----------------------------------------------------
cat > "$WORK/telemetry.sh" <<'EOF'
#!/usr/bin/env bash
arm="$(cat "$STUB_STATE/arm" 2>/dev/null || echo baseline)"
case "$arm" in
  baseline) p=42 ;;
  idle)     p=44 ;;
  rv)       p=52 ;;
  noc)      p=61 ;;
  mm)       p=78 ;;
  sfpu)     p=69 ;;
  *)        p=42 ;;
esac
# A slow drift, so the session is not artificially stationary.
n=$(cat "$STUB_STATE/n" 2>/dev/null || echo 0)
echo $((n + 1)) > "$STUB_STATE/n"
p=$(python3 -c "print($p + 0.004 * $n)")
printf '{"device_info":[{"telemetry":{"aiclk":"800","voltage":"0.75","current":"%s","power":"%s","asic_temperature":"45.0"}}]}\n' "$p" "$p"
EOF

cat > "$WORK/energybench" <<'EOF'
#!/usr/bin/env bash
arm=idle; secs=1; inner=0; label=""
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) arm="$2"; shift 2 ;;
    --seconds) secs="$2"; shift 2 ;;
    --inner) inner="$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --csv) shift 2 ;;
    *) shift ;;
  esac
done
echo "$arm" > "$STUB_STATE/arm"
sleep "$secs"
echo baseline > "$STUB_STATE/arm"
# Cheap arms launch more often, which is what makes W -> J/launch non-trivial.
case "$arm" in
  idle) rate=20000 ;;
  rv)   rate=4000 ;;
  noc)  rate=2500 ;;
  mm)   rate=900 ;;
  sfpu) rate=1200 ;;
  *)    rate=1000 ;;
esac
launches=$(python3 -c "print(int($rate * $secs))")
echo "ENERGYBENCH arm=$arm inner=$inner launches=$launches wall_s=$secs launches_per_s=$rate label=$label"
echo "Completed successfully on the device"
EOF

chmod +x "$WORK/telemetry.sh" "$WORK/energybench"
mkdir -p "$WORK/state"
export STUB_STATE="$WORK/state"
echo baseline > "$STUB_STATE/arm"
echo 0 > "$STUB_STATE/n"

# --- run it -------------------------------------------------------------
echo "== the default schedule =="
TT_METAL_HOME="$WORK" "$HERE/run_card.sh" --list > "$WORK/list.log" 2>&1
check "default schedule has 9 workloads plus a control" \
  "$(grep -cE '^  [0-9]+ +[a-z]+-[0-9]+' "$WORK/list.log")" "10"
check "the control is scheduled last, not next to its twin" \
  "$(tail -n 1 "$WORK/list.log" | grep -c '__control')" "1"

# One scale for the stub run: what is under test here is the scheduling and the
# windowing, and a second scale only makes it slower.
echo ""
echo "== running a stubbed 3-cycle session (short slots, one scale) =="
ENERGYBENCH_TELEMETRY_CMD="$WORK/telemetry.sh" \
ENERGYBENCH_BIN="$WORK/energybench" \
TT_METAL_HOME="$WORK" \
  "$HERE/run_card.sh" --out "$WORK/session" --cycles 3 --seconds 3 --scales 1 \
    --baseline-seconds 3 --settle 0.4 --interval 0.15 > "$WORK/run.log" 2>&1
rc=$?
check "run_card.sh exit status" "$rc" "0"

if [ ! -f "$WORK/session/power.csv" ]; then
  echo "  FAIL no power.csv was written; tail of the run log:"
  tail -n 30 "$WORK/run.log"
  exit 1
fi

# --- assertions ---------------------------------------------------------
rows=$(($(wc -l < "$WORK/session/power.csv") - 1))
check "power.csv rows (3 cycles x 7 slots)" "$rows" "21"

check "the NOT-A-MEASUREMENT banner is present" \
  "$(grep -c 'NOT-A-MEASUREMENT' "$WORK/session/session.log")" "1"

python3 - "$WORK/session/power.csv" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
by = defaultdict(list)
for r in rows:
    by[r["label"]].append(float(r["power_w"]))

problems = []
if sorted(by) != sorted(["baseline", "idle-0", "rv-200000", "noc-4096", "mm-4096",
                         "sfpu-4096", "noc-4096__control"]):
    problems.append(f"labels: {sorted(by)}")
means = {k: sum(v) / len(v) for k, v in by.items()}
order = sorted(means, key=means.get)
want = ["baseline", "idle-0", "rv-200000", "noc-4096", "noc-4096__control",
        "sfpu-4096", "mm-4096"]
# the control and its twin are the same stub power, so their relative order is
# free; compare as a set at that position.
if set(order[:4]) != {"baseline", "idle-0", "rv-200000", "noc-4096"} or order[-1] != "mm-4096":
    problems.append(f"ordering: {order} (wanted roughly {want})")
ctrl = abs(means["noc-4096"] - means["noc-4096__control"])
if ctrl > 1.0:
    problems.append(f"control disagreement {ctrl:.3f} W is too large for a stub board")
if any(int(r["samples"]) < 5 for r in rows):
    problems.append("some slot kept fewer than 5 samples after the settle trim")
if problems:
    print("  FAIL " + "; ".join(problems))
    sys.exit(1)
print(f"  ok   per-arm powers separate and the control agrees to {ctrl:.3f} W")
PY
[ $? = 0 ] || fails=$((fails + 1))

# --- the settle trim, in both directions --------------------------------
# A trim wide enough to swallow every slot must leave nothing and SAY so. A trim
# that leaves nothing while reporting a mean is the failure mode this guards.
echo ""
echo "== the settle trim =="
python3 "$HERE/aggregate_power.py" --slots "$WORK/session/slots.csv" \
  --out "$WORK/overtrimmed.csv" --settle 60 > "$WORK/trim.log" 2>&1
check "an over-wide trim warns" \
  "$(grep -c 'no samples survived' "$WORK/trim.log")" "1"
check "an over-wide trim leaves no samples" \
  "$(python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$WORK/overtrimmed.csv')))
print(max(int(r['samples']) for r in rows))")" "0"
check "the session's own trim kept samples" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/session/power.csv')))
print(min(int(r['samples']) for r in rows) > 0)")" "True"

echo ""
if [ "$fails" = 0 ]; then
  echo "run_card_stub_test: PASS"
else
  echo "run_card_stub_test: $fails FAILED"
fi
exit $((fails > 0))
