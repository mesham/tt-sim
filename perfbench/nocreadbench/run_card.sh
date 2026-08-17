#!/usr/bin/env bash
# Run nocreadbench on a real card and leave one directory to send back.
#
# This is the HARDWARE side. Do not use perfbench/run.sh here, and make sure
# TT_METAL_SIMULATOR is unset: the point of the exercise is that the same binary
# runs either way, and this script's job is only to build it, run it and say
# whether the run was worth keeping.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   perfbench/nocreadbench/run_card.sh --preflight        # costs no card time
#   perfbench/nocreadbench/run_card.sh                    # the E0-E7 sweep, as before
#   perfbench/nocreadbench/run_card.sh --arms             # THE STATEFUL SESSION (~3 min)
#
# Options:
#   --arms            run BOTH issue loops, interleaved, and print the paired
#                     verdict against the predictions registered below. This is
#                     the session `docs/plans/wormhole-session.md` section 4
#                     item 3 asks for; everything else here predates it.
#   --rounds N        rounds of (stateless, stateful) with --arms (default 2)
#   --repeats N       repeats inside each run (default 3)
#   --num-tx N        longest sampled burst (default 64; the `burst` axis always
#                     runs N = 4, 16, 64, 128 whatever this says)
#   --out DIR         where an --arms session lands (default ./nocread-arms-session)
#   --preflight       run every check that can fail for free, then stop
#   --skip-build      assume build/nocreadbench is current
#   -- prog args...   passed to the program in the single-run (non---arms) mode
#
# SPDX-License-Identifier: Apache-2.0

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"

ARMS=0
ROUNDS=2
REPEATS=3
NUM_TX=64
OUT="$PWD/nocread-arms-session"
PREFLIGHT=0
SKIP_BUILD=0
ARGS=()
seen_sep=0
for arg in "$@"; do
  if [ "$seen_sep" = 1 ]; then ARGS+=("$arg"); continue; fi
  case "$arg" in
    --) seen_sep=1 ;;
    --arms) ARMS=1 ;;
    --preflight) PREFLIGHT=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --rounds=*) ROUNDS="${arg#*=}" ;;
    --repeats=*) REPEATS="${arg#*=}" ;;
    --num-tx=*) NUM_TX="${arg#*=}" ;;
    --out=*) OUT="${arg#*=}" ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) ARGS+=("$arg") ;;
  esac
done
# The `--x y` spellings too, since every other runner in perfbench takes them.
set -- "${ARGS[@]+"${ARGS[@]}"}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds) ROUNDS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --num-tx) NUM_TX="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"

# Its program launches through `detail::LaunchProgram`, the direct path, which
# requires slow dispatch; a card defaults to FAST dispatch, so without this every
# run aborts with rc=134 before doing any work. Measured on a Blackhole p150,
# 2026-08-17, when nocevbench hit exactly this. The simulator runners have always
# set it (tt-sim supports no other flow), which is why the gap survived in every
# card runner: the sim side cannot reproduce the failure.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"

if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is for a real card; unset it, or use perfbench/run.sh." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# The single-run mode: what this script has always done. Unchanged.
# ---------------------------------------------------------------------------
if [ "$ARMS" -eq 0 ] && [ "$PREFLIGHT" -eq 0 ]; then
  cd "$SRC" || exit 2
  if [ ! -x build/nocreadbench ]; then
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
    cmake --build build -j >/dev/null || exit 1
  fi

  LOG="$SRC/nocreadbench-run.log"
  echo "nocreadbench: running (log: $LOG)"
  ./build/nocreadbench "${ARGS[@]+"${ARGS[@]}"}" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}

  CSV="$(ls -t "$SRC"/nocreadbench-*.csv 2>/dev/null | head -1)"
  echo
  if [ -z "$CSV" ]; then
    echo "no CSV was produced; send $LOG" >&2
    exit "$rc"
  fi
  echo "SEND BACK: $CSV"
  echo "       and: $LOG"
  echo
  echo "Sanity checks before you do (see README.md, 'Telling a good run from a"
  echo "degenerate one'):"
  echo "  * the printed VERDICT is not DEGENERATE"
  echo "  * MODE CONFIRMED is printed -- every point proved which issue loop it"
  echo "    ran from the tile that answered its own probe"
  echo "  * on Wormhole, the 'burst' rows at 64 B reproduce the 2026-08-17"
  echo "    session's marginal 44.0 cycles/tx. Run ./check_mode.py on the CSV and"
  echo "    it will say so for you."
  echo "  * on Blackhole, cmdbuf_avail_rest is NOT 0xFFFFFFFF"
  echo
  echo "For the stateful comparison -- the reason this program was extended --"
  echo "run it as:  $0 --arms"
  exit "$rc"
fi

# ===========================================================================
# THE STATEFUL SESSION
# ===========================================================================
n_runs=$((ROUNDS * 2))
# ~20 s per run: device open plus 3 x ~35 launches, each of which is microseconds
# of kernel. The 2026-08-17 Wormhole session's whole 110-row nocread probe took
# 0.2 s of device time inside a 25 s process.
est_s=$((n_runs * 25))

cat <<'PREDICTIONS'
nocreadbench stateful session
=============================
WHAT THIS SETTLES, AND THE TWO PREDICTIONS, WRITTEN DOWN BEFORE THE RUN
-----------------------------------------------------------------------
On 2026-08-17 this program measured, on a real Wormhole part, a marginal cost of
44.0 cycles per 64 B L1 read in a pipelined burst: 44.08 at N = 4 -> 16, 44.00 at
16 -> 64, 43.97 at 64 -> 128. Flat to 0.25 % across a 32x range in burst length
and to 1.0 % across hops 1..7. tt-metal's shipped `noc_latencies.yaml` -- the
vendor's dataset, taken on a DIFFERENT part -- says 25.00 for the same shape.

The disagreement is real. Its CAUSE is not settled, and there are two candidates:

  H-LOOP   44 is OUR ISSUE LOOP. The rate is set by the instruction stream on
           the issuing baby RISC-V, our loop is longer than the dataset's, and
           the number says nothing about the part.
  H-FLOOR  44 is a PER-READ COST IN THE PART that the dataset says is not there.

This session runs the same experiment with a SHORTER ISSUE LOOP -- tt-metal's own
`noc_async_read_set_state` once plus `noc_async_read_with_state` per transaction,
which is the pair the dataset's own `stateful` rows use. It removes, per
transaction, the NOC_TARG_ADDR_COORDINATE store (and NOC_TARG_ADDR_MID as well
on Blackhole) plus the 64-bit address arithmetic that fed them.

  IF H-LOOP: the stateful marginal FALLS by roughly the removed instructions and
             lands in the 15-30 cycles/transaction band the shipped dataset
             occupies. Written as: stateful marginal <= 30, and the drop is more
             than 10 % of the stateless one.
  IF H-FLOOR: the stateful marginal BARELY MOVES -- within 10 % of the stateless
             one -- because the instruction stream was never what capped it.

Both are written down here so the result cannot be read after the fact, and
`check_mode.py` prints which one the numbers landed on, including "NEITHER",
which is a real outcome and not a failure.

WHAT ELSE THE RUN HAS TO SATISFY
--------------------------------
  * The STATELESS arm is the CONTROL and must reproduce 2026-08-17's marginal
    44.0. A variant is worth nothing against a control that moved.
  * Each arm must PROVE WHICH LOOP IT RAN, from the returned payload rather than
    from the flag it was passed. Every row carries a probe word naming the tile
    that answered one transaction issued with the read state pointed elsewhere;
    the stateless call rewrites the coordinate and the source answers, the
    stateful call does not and a witness core answers. `--stateful` on a stale
    binary looks exactly like a real stateful run until that column is read.
PREDICTIONS

echo ""
echo "  rounds    : $ROUNDS  (each round runs stateless then stateful)"
echo "  repeats   : $REPEATS inside each run"
echo "  num_tx    : $NUM_TX  (the burst axis is always N = 4, 16, 64, 128)"
echo "  runs      : $n_runs, INTERLEAVED -- a blocked schedule turns drift into"
echo "              a fake difference between the arms"
echo "  wall      : ~$((est_s / 60)) min $((est_s % 60)) s, plus the first build (~2 min)"
echo "  out       : $OUT"

# ---------------------------------------------------------------------------
# Pre-flight. Everything here fails for free; everything after it costs card
# time.
# ---------------------------------------------------------------------------
preflight_fail=0
echo ""
echo "== pre-flight"

if [ "$ROUNDS" -ge 1 ] 2>/dev/null; then
  echo "   ok   --rounds $ROUNDS"
else
  echo "   FAIL --rounds $ROUNDS is not a positive integer"; preflight_fail=1
fi
if [ "$REPEATS" -ge 1 ] 2>/dev/null; then
  echo "   ok   --repeats $REPEATS"
else
  echo "   FAIL --repeats $REPEATS is not a positive integer"; preflight_fail=1
fi
# NIU_MST_REQS_OUTSTANDING_ID is 8 bits and both architectures' NoC/Counters.md
# warns it wraps if software has too many outstanding requests.
if [ "$NUM_TX" -le 128 ] 2>/dev/null && [ "$NUM_TX" -ge 4 ]; then
  echo "   ok   --num-tx $NUM_TX is inside the 8-bit occupancy counter's range"
else
  echo "   FAIL --num-tx $NUM_TX: keep it between 4 and 128. Above 128 the"
  echo "        occupancy counter wraps and E0 reads nonsense."
  preflight_fail=1
fi

if [ ! -f "$HERE/check_mode.py" ]; then
  echo "   FAIL check_mode.py is missing. Without it, a run that silently kept"
  echo "        the stateless loop while claiming the stateful one is"
  echo "        indistinguishable from a good one -- and that failure produces"
  echo "        exactly the reading H-FLOOR predicts. This session cannot"
  echo "        survive it."
  preflight_fail=1
elif ! python3 "$HERE/check_mode.py" --help >/dev/null 2>&1; then
  echo "   FAIL check_mode.py will not run under this python3"
  preflight_fail=1
else
  echo "   ok   check_mode.py runs; every CSV below will be checked against its arm"
fi

# The kernel's mode witness needs a worker at logical (1,1) that is neither the
# initiator (0,0) nor any source (every source is on row 0 or column 0). The host
# refuses per point if the witness collides with the source, but the grid size is
# knowable here for free.
if [ ! -x "$SRC/build/nocreadbench" ] && [ "$SKIP_BUILD" -eq 1 ]; then
  echo "   FAIL --skip-build, but $SRC/build/nocreadbench does not exist"
  preflight_fail=1
else
  echo "   ok   build present or will be built"
fi

if [ "$preflight_fail" -ne 0 ]; then
  echo ""
  echo "PRE-FLIGHT FAILED -- no card time spent. Fix the above and re-run."
  exit 3
fi

if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$SRC/build/nocreadbench" ]; then
  echo ""
  echo "== building (tt-metal supplies every flag; nothing to configure)"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) >/dev/null || { echo "BUILD FAILED"; exit 1; }
  echo "   built"
fi

# The part, and that the program can be configured on it at all, for ~20 s of
# device open. It also gets `nocreadbench-config` on the record before any
# measurement, which is what a stale binary fails at.
echo ""
echo "== pre-flight, on the device (one short run, --only burst --repeats 1)"
smoke="$SRC/nocread-smoke.log"
( cd "$SRC" && ./build/nocreadbench --only burst --num-tx 4 --repeats 1 \
                  --no-sample --out "$SRC/nocread-smoke.csv" ) >"$smoke" 2>&1
rc=$?
cfg=$(grep '^nocreadbench-config' "$smoke" | head -1)
ARCH_NAME=$(echo "$cfg" | sed -n 's/.*arch=\([a-z0-9_]*\).*/\1/p')
if [ $rc -ne 0 ] || [ -z "$cfg" ]; then
  echo "   FAIL the program did not run on this device (rc=$rc). Send $smoke."
  sed -n '$p' "$smoke" | sed 's/^/        /'
  exit 3
fi
echo "   ok   $cfg"
if ! grep -q 'MODE CONFIRMED' "$smoke"; then
  echo "   FAIL the smoke run did not print MODE CONFIRMED, so the mode witness"
  echo "        is not working on this part and neither arm could be attributed."
  echo "        Send $smoke."
  exit 3
fi
echo "   ok   the mode witness works on this part (MODE CONFIRMED in the smoke run)"
echo "   part : ${ARCH_NAME:-unknown}"
if [ "${ARCH_NAME:-}" != "wormhole" ]; then
  echo ""
  echo "   NOTE: the 44.0 control quoted above is WORMHOLE's, from 2026-08-17."
  echo "   On ${ARCH_NAME:-this part} the stateless arm ESTABLISHES a control"
  echo "   rather than reproducing one, and the two predictions are still the"
  echo "   two predictions -- read the DROP, not the absolute number, against"
  echo "   them. Blackhole's own dataset rows say the shorter loop buys 1.0"
  echo "   cycles there, which is that dataset's evidence for H-FLOOR."
fi
[ "$PREFLIGHT" -eq 1 ] && { echo ""; echo "pre-flight passed. Re-run without --preflight to measure."; exit 0; }

mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
: >"$SUMMARY"
{
  echo "nocreadbench stateful session"
  echo "date        : $(date -Is)"
  echo "host        : $(hostname)"
  echo "tt-metal    : $TT_METAL_HOME"
  echo "tt-metal rev: $(git -C "$TT_METAL_HOME" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "arch        : ${ARCH_NAME:-unset}"
  echo "rounds      : $ROUNDS (interleaved)"
  echo "repeats     : $REPEATS"
  echo "num_tx      : $NUM_TX"
} >"$OUT/env.txt"

status=0
CSVS=()
for r in $(seq 1 "$ROUNDS"); do
  for arm in stateless stateful; do
    name="$arm-$r"
    csv="$OUT/nocread-$name.csv"
    log="$OUT/nocread-$name.log"
    flag=""
    [ "$arm" = stateful ] && flag="--stateful"
    echo ""
    echo "== $name"
    # shellcheck disable=SC2086
    ( cd "$SRC" && ./build/nocreadbench $flag --num-tx "$NUM_TX" \
                     --repeats "$REPEATS" --out "$csv" ) >"$log" 2>&1
    rc=$?
    if [ ! -s "$csv" ]; then
      printf 'FAIL  %-14s rc=%s  no CSV; send %s\n' "$name" "$rc" "$(basename "$log")" \
        | tee -a "$SUMMARY"
      status=1
      continue
    fi
    # THE ARM CHECK. Read out of the returned payload, not taken on trust from
    # the flag, and it also prints this run's marginals so the control can be
    # judged HERE rather than only after the session is sent home.
    if ! chk=$(python3 "$HERE/check_mode.py" --expect "$arm" --stdout "$log" "$csv" 2>&1); then
      printf 'FAIL  %-14s the arm did not take:\n%s\n' "$name" "$chk" | tee -a "$SUMMARY"
      status=1
      continue
    fi
    printf 'PASS  %-14s arm=%s confirmed from payload\n' "$name" "$arm" | tee -a "$SUMMARY"
    echo "$chk" | sed -n 's/^  \(marg\|ctrl\|avg\|note\) /      \1 /p' | tee -a "$SUMMARY"
    CSVS+=("$csv")
  done
done

echo ""
echo "== the paired verdict, against the predictions printed before the run"
if [ ${#CSVS[@]} -ge 2 ]; then
  python3 "$HERE/check_mode.py" --quiet "${CSVS[@]}" | tee -a "$SUMMARY"
else
  echo "  not enough good runs to pair" | tee -a "$SUMMARY"
  status=1
fi

{
  echo ""
  echo "Before sending this home, check by eye:"
  echo "  * every line above says PASS. 'arm=X confirmed from payload' means the"
  echo "    issue loop was read back out of the data, not out of the flag."
  if [ "${ARCH_NAME:-}" = "wormhole" ]; then
    echo "  * the stateless rounds each printed 'the control reproduces' for all"
    echo "    three burst intervals. If any says CTRL MOVED, SAY SO and send the"
    echo "    session anyway -- a control that moved is the most important result"
    echo "    in the directory, and it means the two arms are not a pair."
  else
    echo "  * there is NO prior card control on ${ARCH_NAME:-this part}, so the"
    echo "    stateless rounds establish one. What you can check here is that the"
    echo "    rounds agree with each other."
  fi
  echo "  * the rounds of each arm agree ($ROUNDS of each). They are separate runs"
  echo "    of an identical program, so a big spread means something else was on"
  echo "    the card. Do NOT re-run and keep the better session -- say so instead."
  echo "  * the two arms DISAGREE, or agree, and either is a result. What is NOT"
  echo "    a result is a stateful round that failed its arm check: that number"
  echo "    is the stateless loop wearing the stateful label, and it reads as"
  echo "    H-FLOOR whatever the part does."
  echo ""
  echo "Then send the WHOLE directory, logs included -- they carry the"
  echo "nocreadbench-config line recording what each run was configured as:"
  echo "  rsync -av $OUT/ <home>:~/$(basename "$OUT")/"
} | tee -a "$SUMMARY"

exit $status
