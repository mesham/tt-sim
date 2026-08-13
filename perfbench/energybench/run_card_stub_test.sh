#!/usr/bin/env bash
# Exercise run_card.sh end to end WITHOUT A CARD.
#
# WHAT THE OLD VERSION OF THIS FILE COULD NOT CATCH
# -------------------------------------------------
# The previous stub board answered every telemetry request instantly and always
# succeeded. It proved the scheduling, the trimming, the gates and the
# aggregation -- and it could NEVER have caught the bug that lost a whole card
# session, because a stub that always answers cannot tell you what the real tool
# does when the device is busy. On 2026-08-13 `tt-smi` 3.0.32 panicked on every
# in-slot read (BAR0 is exclusive on its Luwen backend) and the harness banked
# three cycles of `samples=0, power_w=0` that looked like a finished measurement.
#
# So the stub now HOLDS AN EXCLUSIVE RESOURCE the way a real run does: the stub
# benchmark takes an exclusive `flock` on a lock file for its whole run, and the
# stub telemetry tool must take the same lock to answer. Two behaviours are
# switchable on that:
#
#   STUB_SMI_ERA=luwen   the tool takes the lock NON-BLOCKING and, when it
#                        cannot, reproduces the pyluwen panic verbatim and exits
#                        non-zero -- tt-smi < 4.0.0
#   STUB_SMI_ERA=umd     the tool reads through a held lock and answers -- the
#                        tt-umd backend of tt-smi >= 4.0.0, which is what the
#                        card box now runs
#
# What is under test is therefore: the version guard, the interleave schedule,
# the sampler lifecycle, the per-slot windows, the launch-rate capture, the
# settle-trimmed aggregation, AND -- the point of the rebuild -- that a sampler
# which fails FOR ANY REASON is loud rather than silent.
#
#   perfbench/energybench/run_card_stub_test.sh
#
# Nothing this produces is a measurement, and the session banner says so.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENERGYBENCH_HERE="$HERE"
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
# The telemetry tool. It reports a per-arm power, a slow drift, and a version --
# and, on the `luwen` era, it refuses while the device is held.
cat > "$WORK/telemetry.sh" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  --version) echo "tt-smi ${STUB_SMI_VERSION:-6.2.0}"; exit 0 ;;
esac

# The device lock. A real tt-smi opens the chip to read telemetry; whether that
# is possible while a tt-metal program holds BAR0 is exactly what changed
# between tt-smi 3.x (Luwen) and 4.0.0+ (tt-umd).
if [ "${STUB_SMI_ERA:-umd}" = "luwen" ]; then
  exec 9>>"$STUB_STATE/chip0.lock"
  if ! flock -n 9; then
    cat >&2 <<'PANIC'
thread '<unnamed>' panicked at crates/ttkmd-if/src/lib.rs:294:17:
Failed to map bar0_uc for 0 with error Invalid argument (os error 22)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
PANIC
    exit 101
  fi
fi

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
printf '{"host_sw_vers":{"tt_smi":"%s"},"device_info":[{"telemetry":{"aiclk":" 1350","voltage":" 0.75","current":" %s","power":" %s","asic_temperature":" 45.0"}}]}\n' \
  "${STUB_SMI_VERSION:-6.2.0}" "$p" "$p"
EOF

# The benchmark. It HOLDS THE DEVICE LOCK for the whole of its run, exclusively,
# which is what makes the telemetry tool's behaviour under contention testable.
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
exec 9>>"$STUB_STATE/chip0.lock"
flock 9 || exit 70
echo "$arm" > "$STUB_STATE/arm"
if [ -f "$STUB_STATE/fail_once_$label" ]; then
  rm -f "$STUB_STATE/fail_once_$label"
  echo baseline > "$STUB_STATE/arm"
  echo "energybench: device open failed" >&2
  exit 1
fi
sleep "$secs"
echo baseline > "$STUB_STATE/arm"
flock -u 9
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
: > "$STUB_STATE/chip0.lock"

# A stub sysfs tree: the four attributes the driver publishes, and nothing else.
SYSFS="$WORK/sysfs"
mkdir -p "$SYSFS/tenstorrent!0"
echo 1350 > "$SYSFS/tenstorrent!0/tt_aiclk"
echo 540  > "$SYSFS/tenstorrent!0/tt_arcclk"
echo 900  > "$SYSFS/tenstorrent!0/tt_axiclk"
echo 0    > "$SYSFS/tenstorrent!0/tt_therm_trip_count"

stub_env=(
  ENERGYBENCH_TELEMETRY_CMD="$WORK/telemetry.sh"
  ENERGYBENCH_BIN="$WORK/energybench"
  ENERGYBENCH_SYSFS_ROOT="$SYSFS"
  TT_METAL_HOME="$WORK"
)

# --- 1. the stub actually reproduces the original failure ---------------
# A stub that cannot reproduce the bug is not a test of the fix, so this is
# established FIRST and on its own terms, before anything else runs.
echo "== the stub reproduces the 2026-08-13 failure =="
(
  exec 9>>"$STUB_STATE/chip0.lock"
  flock 9
  sleep 3
) &
holder=$!
sleep 0.4

STUB_SMI_ERA=luwen STUB_SMI_VERSION=3.0.32 "$WORK/telemetry.sh" > "$WORK/busy.out" 2> "$WORK/busy.err"
check "tt-smi 3.0.32 (Luwen) exits non-zero while the device is held" "$?" "101"
check "...and it panics exactly as the card did" \
  "$(grep -c 'Failed to map bar0_uc' "$WORK/busy.err")" "1"

STUB_SMI_ERA=umd "$WORK/telemetry.sh" > "$WORK/busy2.out" 2>/dev/null
check "tt-smi 6.2.0 (tt-umd) reads through the same held lock" "$?" "0"
check "...and returns a power" \
  "$(python3 -c "
import json,sys
print(json.load(open('$WORK/busy2.out'))['device_info'][0]['telemetry']['power'].strip() != '')")" "True"
wait $holder

# The sampler over a busy device on the old tool: every attempt recorded, every
# one a failure, and the module exits non-zero rather than writing silence.
(
  exec 9>>"$STUB_STATE/chip0.lock"
  flock 9
  sleep 4
) &
holder=$!
sleep 0.4
STUB_SMI_ERA=luwen python3 "$HERE/telemetry_sample.py" watch --out "$WORK/busy.pow.csv" \
  --command "$WORK/telemetry.sh" --phase run --interval 0 --duration 1.5 \
  > "$WORK/busywatch.log" 2>&1
check "the sampler exits non-zero when every attempt fails" "$?" "3"
check "...it says NO TELEMETRY rather than nothing" \
  "$(grep -c 'NO TELEMETRY' "$WORK/busywatch.log")" "1"
check "...every failed attempt is a row, not a dropped exception" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/busy.pow.csv')))
print(len(rows) > 0 and all(r['ok']=='0' and r['power_w']=='' for r in rows))")" "True"
check "...and the panic text is carried into the row" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/busy.pow.csv')))
print(any('bar0_uc' in r['error'] for r in rows))")" "True"
wait $holder

# --- 2. the version guard -----------------------------------------------
echo ""
echo "== the version guard =="
env "${stub_env[@]}" STUB_SMI_ERA=luwen STUB_SMI_VERSION=3.0.32 \
  "$HERE/run_card.sh" --out "$WORK/old" --cycles 1 --seconds 1 --scales 1 \
    --baseline-seconds 1 > "$WORK/old.log" 2>&1
check "run_card.sh refuses to start on tt-smi 3.0.32" "$?" "4"
check "...and says why" "$(grep -c 'REFUSING TO START' "$WORK/old.log")" "1"
check "...and the refusal names the backend change" \
  "$(grep -c 'Luwen to tt-umd' "$WORK/old.log")" "1"
env "${stub_env[@]}" STUB_SMI_VERSION=4.0.0 \
  "$HERE/run_card.sh" --list > "$WORK/v4.log" 2>&1
check "a 4.0.0 tool is accepted" "$?" "0"

# --- 3. the schedule ----------------------------------------------------
echo ""
echo "== the default schedule =="
env "${stub_env[@]}" "$HERE/run_card.sh" --list > "$WORK/list.log" 2>&1
check "default schedule has 9 workloads plus a control" \
  "$(grep -cE '^  [0-9]+ +[a-z]+-[0-9]+' "$WORK/list.log")" "10"
check "the control is scheduled last, not next to its twin" \
  "$(grep '__control' "$WORK/list.log" | tail -n 1 | grep -c '__control')" "1"
check "the banner states the quantity" \
  "$(grep -c 'SUSTAINED LOAD' "$WORK/list.log")" "1"

# One scale for the stub run: what is under test here is the scheduling and the
# windowing, and a second scale only makes it slower.
echo ""
echo "== running a stubbed 3-cycle session (short slots, one scale) =="
env "${stub_env[@]}" \
  "$HERE/run_card.sh" --out "$WORK/session" --cycles 3 --seconds 3 --scales 1 \
    --baseline-seconds 3 --settle 0.4 --interval 0.05 --pre-samples 1 > "$WORK/run.log" 2>&1
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
check "the tt-smi version is banked with the session" \
  "$(grep -c 'tt_smi_version=6.2.0' "$WORK/session/session.log")" "1"

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
# the control and its twin are the same stub power, so their relative order is
# free; compare as a set at that position.
if set(order[:4]) != {"baseline", "idle-0", "rv-200000", "noc-4096"} or order[-1] != "mm-4096":
    problems.append(f"ordering: {order}")
ctrl = abs(means["noc-4096"] - means["noc-4096__control"])
if ctrl > 1.0:
    problems.append(f"control disagreement {ctrl:.3f} W is too large for a stub board")
if any(int(r["samples"]) < 5 for r in rows):
    problems.append("some slot kept fewer than 5 samples after the settle trim")
if any(r["status"] != "ok" for r in rows):
    problems.append("some slot did not come back ok")
# Every row must carry the confound control and the paired local reference.
if any(not r["sysfs_aiclk_mean"] or not r["therm_trip_delta"] == "0" for r in rows):
    problems.append("some slot has no sysfs clock/thermal record")
if any(not r["pre_idle_w"] or not r["delta_w"] for r in rows):
    problems.append("some slot has no pre-slot idle reference")
if any(r["tt_smi_version"] != "6.2.0" for r in rows):
    problems.append("the tt-smi version did not reach the CSV")
if problems:
    print("  FAIL " + "; ".join(problems))
    sys.exit(1)
print(f"  ok   per-arm powers separate and the control agrees to {ctrl:.3f} W")
print("  ok   every row carries a clock record, a thermal record and a pre-idle reference")
PY
[ $? = 0 ] || fails=$((fails + 1))

# --- 4. a zero-sample slot cannot look like success ---------------------
# Both directions: the clean session above passed, and a session with one dead
# slot must be refused by the aggregator, announced by the runner, and refused
# again by the analysis.
echo ""
echo "== a slot with no telemetry =="
python3 - "$WORK/session" <<'PY'
import csv, os, shutil, subprocess, sys
src = sys.argv[1]
dst = src + "-dead"
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)
# Kill one slot's telemetry the way the card did: every attempt present, every
# attempt refused. Nothing else about the session changes.
slots = list(csv.DictReader(open(os.path.join(dst, "slots.csv"))))
victim = next(s for s in slots if s["label"] == "mm-4096" and s["cycle"] == "1")
path = os.path.join(dst, victim["pow_file"])
rows = list(csv.DictReader(open(path)))
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        r.update(ok="0", power_w="", voltage_v="", current_a="", aiclk_mhz="",
                 temp_c="", error="rc=101: Failed to map bar0_uc for 0")
        w.writerow(r)
agg = os.path.join(os.environ["ENERGYBENCH_HERE"], "aggregate_power.py")
proc = subprocess.run(
    [sys.executable, agg, "--slots", os.path.join(dst, "slots.csv"),
     "--out", os.path.join(dst, "power.csv"), "--settle", "0.4"],
    capture_output=True, text=True)
problems = []
if proc.returncode == 0:
    problems.append("the aggregator accepted a session with a dead slot")
if "REFUSED" not in proc.stderr:
    problems.append("the aggregator did not say REFUSED")
if "no_telemetry" not in proc.stderr:
    problems.append("the aggregator did not name the failure")
out = {(r["cycle"], r["label"]): r for r in csv.DictReader(open(os.path.join(dst, "power.csv")))}
dead = out[("1", "mm-4096")]
if dead["power_w"] != "":
    problems.append(f"the dead slot's power is {dead['power_w']!r}, not empty -- "
                    "an empty cell is what stops 0 W being read as a measurement")
if dead["samples"] != "0" or dead["status"] != "no_telemetry":
    problems.append(f"the dead slot is not marked: {dead['samples']} {dead['status']}")
if len(out) != 21:
    problems.append(f"the dead slot vanished from the CSV: {len(out)} rows")
if problems:
    print("  FAIL " + "; ".join(problems))
    sys.exit(1)
print("  ok   a dead slot is written EMPTY, marked no_telemetry, and refused (rc=%d)"
      % proc.returncode)
PY
[ $? = 0 ] || fails=$((fails + 1))

# --- 5. a slot whose RUN fails is still recorded ------------------------
# This is cycle 2's missing `idle-0`: the old runner wrote a manifest row only on
# success, so a failed slot left no trace in either machine-readable output and
# the CSV silently skipped a slot number.
echo ""
echo "== a slot whose run fails =="
: > "$STUB_STATE/fail_once_idle-0"
env "${stub_env[@]}" \
  "$HERE/run_card.sh" --out "$WORK/crash" --cycles 2 --seconds 2 --scales 1 \
    --arms "idle noc" --control noc --baseline-seconds 2 --settle 0.3 \
    --interval 0.05 --pre-samples 1 > "$WORK/crash.log" 2>&1
check "run_card.sh exits non-zero when a slot's run fails" "$?" "1"
check "...and announces it at the moment it happens" \
  "$(grep -c 'RUN FAILED' "$WORK/crash.log")" "1"
check "...and again in the handover block" \
  "$(grep -c 'DID NOT PRODUCE A CLEAN MEASUREMENT' "$WORK/crash.log")" "1"
check "...and the failed slot is STILL a row (this is what cycle 2 lost)" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/crash/power.csv')))
print(sum(1 for r in rows if r['label']=='idle-0'))")" "2"
check "...marked run_failed rather than absent" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/crash/power.csv')))
print(sum(1 for r in rows if r['status']=='run_failed'))")" "1"

# --- 6. the settle trim, in both directions ------------------------------
# A trim wide enough to swallow every slot must leave nothing and SAY so. A trim
# that leaves nothing while reporting a mean is the failure mode this guards.
echo ""
echo "== the settle trim =="
python3 "$HERE/aggregate_power.py" --slots "$WORK/session/slots.csv" \
  --out "$WORK/overtrimmed.csv" --settle 60 > "$WORK/trim.log" 2>&1
check "an over-wide trim leaves no samples" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/overtrimmed.csv')))
print(max(int(r['samples']) for r in rows))")" "0"
check "an over-wide trim REFUSES rather than reporting a mean of nothing" \
  "$(grep -c 'REFUSED' "$WORK/trim.log")" "1"
check "...and writes no power at all for those rows" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/overtrimmed.csv')))
print(all(r['power_w']=='' for r in rows))")" "True"
check "the session's own trim kept samples" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/session/power.csv')))
print(min(int(r['samples']) for r in rows) > 0)")" "True"

# --- 7. the decay fit, in both directions --------------------------------
# The --bracket fallback is only meaningful if the decay is slow. Synthetic
# decays with KNOWN time constants: one the fallback survives, one it does not.
echo ""
echo "== the decay fit (the --bracket fallback) =="
python3 - "$WORK" <<'PY'
import csv, math, os, subprocess, sys
work = sys.argv[1]
here = os.environ["ENERGYBENCH_HERE"]

def probe(path, tau, p_inf=40.0, amp=30.0, lag=1.4):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["phase","i","t_start","t_end","t_mid","dt_s","ok","power_w",
                    "voltage_v","current_a","aiclk_mhz","temp_c","error"])
        for i in range(16):
            t = lag + 1.4 * i
            w.writerow(["post", i, 0, 0, t, t, 1, f"{p_inf + amp * math.exp(-t/tau):.4f}",
                        0.75, 1.0, 1350, 45.0, ""])

results = {}
for name, tau in (("slow", 40.0), ("fast", 0.4)):
    path = os.path.join(work, f"decay-{name}.csv")
    probe(path, tau)
    proc = subprocess.run(
        [sys.executable, os.path.join(here, "telemetry_sample.py"), "fit",
         "--samples", path, "--json", os.path.join(work, f"decay-{name}.json")],
        capture_output=True, text=True)
    results[name] = (proc.returncode, proc.stdout)

problems = []
rc, out = results["slow"]
if rc != 0 or "BRACKETING OK" not in out:
    problems.append(f"a tau=40 s decay should be usable: rc={rc}\n{out}")
if "tau" not in out or "40." not in out.split("fitted tau")[1][:16]:
    problems.append(f"tau=40 s was not recovered:\n{out}")
rc, out = results["fast"]
if rc == 0 or "BRACKETING CONDEMNED" not in out:
    problems.append(f"a tau=0.4 s decay must be condemned: rc={rc}\n{out}")
if problems:
    print("  FAIL " + "; ".join(problems))
    sys.exit(1)
print("  ok   tau=40 s recovered and bracketing passed; tau=0.4 s CONDEMNED")
PY
[ $? = 0 ] || fails=$((fails + 1))

# --- 8. the --bracket fallback is wired, and labelled as a fallback ------
echo ""
echo "== the --bracket fallback =="
env "${stub_env[@]}" \
  "$HERE/run_card.sh" --out "$WORK/brk" --cycles 1 --seconds 2 --scales 1 \
    --arms "idle noc" --control noc --baseline-seconds 2 --settle 0.3 \
    --interval 0.05 --pre-samples 1 --post-samples 4 --bracket \
    > "$WORK/brk.log" 2>&1
check "a --bracket session completes" "$?" "0"
check "...and says the fallback is a different quantity" \
  "$(python3 -c "print(open('$WORK/brk.log').read().count('decaying edge') >= 2)")" "True"
check "...and every slot has post-exit samples beside its in-slot ones" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/brk/power.csv')))
print(all(int(r['bracket_samples'])>0 and int(r['samples'])>0 for r in rows))")" "True"
check "...and the in-slot power is still the analysis column" \
  "$(python3 -c "
import csv
rows=list(csv.DictReader(open('$WORK/brk/power.csv')))
print(all(r['power_w'] and r['power_bracket_w'] for r in rows))")" "True"

echo ""
if [ "$fails" = 0 ]; then
  echo "run_card_stub_test: PASS"
else
  echo "run_card_stub_test: $fails FAILED"
fi
exit $((fails > 0))
