#!/usr/bin/env bash
# retirebench on a real card: collect the per-zone mcycle/minstret table for
# rung 4's RV-bound leg.
#
# YOU DO NOT NEED TO KNOW ANYTHING ABOUT tt-sim TO RUN THIS. It builds one
# normal tt-metal program, runs it a handful of times, checks each run validated
# its own result buffer, and leaves a directory to send home. No Tracy front end,
# no device profiler, no tt-exalens, no board reset, no root.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   ./run_card.sh --preflight        # checks that cost no card time
#   ./run_card.sh --list             # schedule and wall estimate
#   ./run_card.sh --out ~/tt_traces/retirebench-session
#
# Options:
#   --scale N     multiply every zone's repetition count (default 1). MUST match
#                 whatever the simulator side was run at: a comparison across two
#                 scales is a comparison of two different programs, and the
#                 analysis refuses it by zone table and by retired census.
#   --repeats N   runs per session (default 3). Each lands in its own directory;
#                 they are never concatenated, because one artefact is one
#                 launch and averaging across two is averaging across two device
#                 states.
#   --out DIR     where the session lands (default ./retirebench-session)
#   --preflight   run the checks that can fail before card time is spent, then stop
#   --list        print the schedule and the estimate, run nothing
#   --skip-build  assume build/retirebench is current
#
# BLACKHOLE ONLY, AND IT REFUSES RATHER THAN DEGRADING.
#   The instrument is the mcycle (0xb00) and minstret (0xb02) CSRs, documented in
#   BlackholeA0/TensixTile/BabyRISCV/CSRs.md. The string "csr" appears ZERO times
#   in the whole WormholeB0 doc tree. Without minstret there is no retired
#   instruction count, this benchmark's zone labels stop being checkable against
#   anything, and what is left is an elapsed-only envelope check -- which is
#   exactly the kind of claim rung 4 exists to distrust. So the program declines
#   a non-Blackhole part before it builds its kernel or launches anything, and
#   the pre-flight below asks the device which part it is rather than taking
#   anyone's word for it.
#
# WHAT IT LEAVES BEHIND: nothing. No Tensix instruction is issued, no NoC
# transaction is started, no semaphore is touched. The only device state written
# is two small L1 buffers obtained from the tt-metal allocator.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
# shellcheck source=../build_provenance.sh
. "$HERE/../build_provenance.sh"

OUT="$PWD/retirebench-session"
SCALE=1
REPEATS=3
LIST=0
PREFLIGHT=0
SKIP_BUILD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --scale) SCALE="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    --list) LIST=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,42p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ~20 s per run: device open plus one launch. The kernel itself is microseconds.
est_s=$((REPEATS * 20))

echo "retirebench card session"
echo "  scale     : $SCALE  (must match the simulator side)"
echo "  repeats   : $REPEATS, each in its own directory"
echo "  instrument: the mcycle / minstret CSRs, read around 12 zones"
echo "              -> no profiler, no Tracy, no in-tree tt-metal patch"
echo "  runs      : $REPEATS"
echo "  wall      : ~$((est_s / 60)) min $((est_s % 60)) s, plus the first build (~2 min)"
echo "  out       : $OUT"
[ "$LIST" -eq 1 ] && exit 0

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"

# A card box that also has a tt-sim checkout is exactly where this goes wrong:
# with TT_METAL_SIMULATOR left set, every run below completes, validates its own
# result buffer and writes a session directory that is a SIMULATOR artefact
# labelled as a card's -- and this leg's whole output is the difference between
# the two.
if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is the CARD side; unset it, or use ./run_sim.sh." >&2
  exit 2
fi

# `retirebench.cpp` launches through `detail::LaunchProgram`, which is the direct
# path and requires slow dispatch. A card defaults to FAST dispatch, so without
# this every run aborts with rc=134 before it reaches a single zone -- measured
# on a Blackhole p150, 2026-08-17, and the reason four earlier card runners did
# no work at all. `run_sim.sh` has always set it (tt-sim supports no other flow),
# which is exactly why the gap survives review: the simulator side can never
# reproduce the failure. Overridable, because a future program on the
# command-queue flow would want it unset.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

# ---------------------------------------------------------------------------
# Pre-flight. Everything here fails for free; everything after it costs card
# time.
# ---------------------------------------------------------------------------
preflight_fail=0
echo ""
echo "== pre-flight"

if [ "$SCALE" -ge 1 ] 2>/dev/null; then
  echo "   ok   --scale $SCALE is a positive integer"
else
  echo "   FAIL --scale $SCALE must be an integer >= 1"
  preflight_fail=1
fi

if [ "$REPEATS" -ge 1 ] 2>/dev/null; then
  echo "   ok   --repeats $REPEATS is a positive integer"
else
  echo "   FAIL --repeats $REPEATS must be an integer >= 1"
  preflight_fail=1
fi

# The zone count, read out of the header rather than restated here, so this
# check cannot drift from the program. ROADMAP section 4 caps a RISC at 60
# zones; the analysis refuses more.
zones=$(sed -n 's/^#define RETIREBENCH_NUM_ZONES \([0-9]*\).*/\1/p' \
          "$SRC/kernels/dataflow/retirebench_layout.h")
if [ -n "$zones" ] && [ "$zones" -le 60 ]; then
  echo "   ok   $zones zones, inside the 60-zone ceiling"
else
  echo "   FAIL the kernel declares '${zones:-no}' zones; the ceiling is 60"
  preflight_fail=1
fi

# Which tt-metal the build tree was made against. cmake reuses a cache that
# still points at a previous checkout, so an existing tree is not evidence that
# it matches the library this run will load -- see build_provenance.sh.
bp_check_build "$SRC" "$([ "$SKIP_BUILD" -eq 1 ] && echo skip-build || echo build)" \
  || preflight_fail=1

if [ "$preflight_fail" -ne 0 ]; then
  echo ""
  echo "PRE-FLIGHT FAILED -- no card time spent. Fix the above and re-run."
  exit 3
fi

if [ "$SKIP_BUILD" -eq 0 ] || [ ! -x "$SRC/build/retirebench" ]; then
  echo ""
  echo "== building (tt-metal supplies every flag; nothing to configure)"
  bp_require_build "$SRC"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || { echo "BUILD FAILED"; exit 1; }
  bp_record_build "$SRC"
fi

# ---------------------------------------------------------------------------
# The part of the pre-flight that needs the device. It opens it and closes it
# again without launching anything (~5 s). The part is the one thing this
# benchmark cannot proceed without and cannot learn from the host.
# ---------------------------------------------------------------------------
echo ""
echo "== pre-flight, on the device (opens and closes it; launches nothing)"
desc=$( cd "$SRC" && ./build/retirebench --describe 2>&1 )
rc=$?
ARCH_NAME=$(echo "$desc" | sed -n 's/^retirebench-config arch=\([a-z0-9_]*\).*/\1/p' | head -1)
if [ $rc -ne 0 ]; then
  echo "   FAIL the device refused this benchmark:"
  echo "$desc" | sed 's/^/        /'
  echo ""
  if [ "${ARCH_NAME:-}" = "wormhole" ]; then
    echo "   This is a WORMHOLE part, and that refusal is correct rather than a"
    echo "   bug to work around. There is no mcycle and no minstret to read: the"
    echo "   string 'csr' does not appear anywhere in the WormholeB0 ISA"
    echo "   documentation. Running anyway would produce elapsed cycles with no"
    echo "   retired counts, the zone labels would stop being checkable, and the"
    echo "   result would be a fourth envelope check in a repo that already has"
    echo "   three and built rung 4 because they cannot see a compensating"
    echo "   interior. Do not patch this out. Run perfbench/riscvbench on this"
    echo "   part instead -- it times the same instruction mixes off the wall"
    echo "   clock and needs no CSRs."
  fi
  exit 3
fi
echo "   ok   $(echo "$desc" | grep '^retirebench-config' | head -1)"
echo "   part      : ${ARCH_NAME:-unknown} (read from the device, recorded in env.txt)"
[ "$PREFLIGHT" -eq 1 ] && exit 0

mkdir -p "$OUT/runs"
SUMMARY="$OUT/summary.txt"
: >"$SUMMARY"

{
  echo "retirebench card session"
  echo "date        : $(date -Is)"
  echo "host        : $(hostname)"
  echo "tt-metal    : $TT_METAL_HOME"
  echo "tt-metal rev: $(git -C "$TT_METAL_HOME" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "arch        : ${ARCH_NAME:-unset}"
  echo "scale       : $SCALE"
  echo "repeats     : $REPEATS"
  echo "zones       : $zones"
  echo "instrument  : mcycle (0xb00) / minstret (0xb02), read around each zone"
} >"$OUT/env.txt"

status=0
for rep in $(seq 1 "$REPEATS"); do
  name="card-$rep"
  dir="$OUT/runs/$name"
  mkdir -p "$dir"
  echo "== $name"
  ( cd "$SRC" && ./build/retirebench --scale "$SCALE" --label "$name" --out "$dir" ) \
      >"$dir/stdout.log" 2>&1
  rc=$?
  artefact=$(ls "$dir"/retirebench-*.json 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || ! grep -q "Completed successfully on the device" "$dir/stdout.log"; then
    printf 'FAIL  %-10s rc=%s  the program did not validate its own result buffer\n' \
      "$name" "$rc" | tee -a "$SUMMARY"
    status=1
    continue
  fi
  if [ -z "$artefact" ]; then
    printf 'FAIL  %-10s no retirebench-*.json in %s\n' "$name" "$dir" | tee -a "$SUMMARY"
    status=1
    continue
  fi
  window=$(sed -n 's/.*"window": {"cycles": \([0-9]*\).*/\1/p' "$artefact" | head -1)
  printf 'PASS  %-10s window=%s cycles  %s\n' \
    "$name" "$window" "$(basename "$artefact")" | tee -a "$SUMMARY"
  # The per-zone table, so the session can be sanity-checked HERE rather than
  # only at home. The two things worth an operator's eye are in it.
  sed -n '/^zone /,/^unattributed/p' "$dir/stdout.log" | sed 's/^/      /' | tee -a "$SUMMARY"
done

{
  echo ""
  echo "Before sending this home, check by eye:"
  echo "  * every line above says PASS."
  echo "  * the $REPEATS repeats agree. They are separate runs of an identical"
  echo "    deterministic program with no data-dependent control flow, so the"
  echo "    'retired' column must be IDENTICAL across repeats -- not close,"
  echo "    identical -- and the cycle columns should be within a per cent or so."
  echo "    A retired column that moves means something is wrong with the"
  echo "    instrument, and it is the most important thing in this directory."
  echo "    Do NOT re-run and keep the better session; say so in the handover."
  echo "  * 'unattributed' is small (a few hundred cycles). It is the harness's"
  echo "    own result stores between zones. A large or negative one means the"
  echo "    counter reads did not bracket what they were meant to."
  echo "  * the zones DISAGREE with each other in cyc/instr. They are twelve"
  echo "    different instruction mixes; if they all read ~1.0 the program did"
  echo "    not run the mixes it thinks it did."
  echo ""
  echo "Then send the WHOLE directory, stdout logs included -- they carry the"
  echo "retirebench-config line that records what each run was configured as:"
  echo "  rsync -av $OUT/ <home>:~/$(basename "$OUT")/"
  echo ""
  echo "At home, per repeat:"
  echo "  python3 -m tt_sim.perf.retire_attribution \\"
  echo "      --sim  <simulator artefact at the same --scale> \\"
  echo "      --card $OUT/runs/card-1/retirebench-blackhole-card-1.json"
} | tee -a "$SUMMARY"

exit $status
