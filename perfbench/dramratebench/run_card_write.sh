#!/usr/bin/env bash
# dramratebench, WRITE DIRECTION, on a real card: is a DRAM endpoint's sustained
# WRITE rate the same as its sustained READ rate?
#
# YOU DO NOT NEED TO KNOW ANYTHING ABOUT tt-sim TO RUN THIS. It builds one
# normal tt-metal program, resets the board, runs the program twice, checks each
# run proved where its bytes went, and leaves a directory to send home. No Tracy,
# no tt-exalens, no root.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   ./run_card_write.sh --preflight     # every check that costs no card time
#   ./run_card_write.sh --list          # the schedule and the wall estimate
#   ./run_card_write.sh --out ~/tt_traces/dram-write-session
#
# Options:
#   --out DIR        where the session lands (default ./dramratebench-write-session)
#   --preflight      run the free checks and the pre-registration, then stop
#   --list           print the schedule and the estimate, run nothing
#   --no-reset       skip the board reset. Read the RESET section before you do.
#   --sustained-only just the vendor's 1/12/48 counts. Faster, and the fan-out
#                    control is weaker for it -- see PREDICTIONS below.
#   --skip-build     assume build/dramratebench is current
#   --reset-cmd CMD  the board reset to run (default 'tt-smi -r 0')
#
# WHAT THIS ASKS, AND WHAT IT CANNOT ASK
#   `unit_costs.yaml` charges one DRAM `access_latency` to a read, a write and
#   an atomic alike on Wormhole. Blackhole's rows split cleanly (write 22
#   against read 126) and Wormhole's were looked at and DECLINED: its write
#   differences run 228 / 124 / 127 / 139 / 139 / 145 / 137 / 155 against read
#   differences 104 / 104 / 104 / 112 / 104 / 112 / 128 / 152 -- no clean
#   asymmetry, a 64 B outlier, and a write if anything DEARER than a read. A
#   Wormhole card agreed with the decline on 2026-08-17: +21.2 cycles at 256 B
#   and +56.4 at 4096 B, write over read, opposite in sign to Blackhole's.
#
#   Neither of those can settle it. Both are LATENCY instruments -- one
#   transaction at a time -- and a latency instrument cannot see an endpoint's
#   OCCUPANCY, which is how long it is unavailable to the NEXT request. This is
#   a RATE campaign and occupancy is the only thing it can see. It answers "are
#   N writers concentrated on one endpoint serialised by it more, less or the
#   same as N readers are". It does NOT answer "what does one write cost", it
#   cannot split service time from queueing, and -- because the `samecore` arm
#   does not exist on either part -- it cannot tell the endpoint from the one
#   inbound router link every concentrated flow converges on. Report "the
#   endpoint", never "the channel".
#
#   Nothing it produces is provenance. Wormhole's `dram.bandwidth` is `unknown`
#   for want of a DOCUMENT and no measurement supplies one. A clean result
#   enters `unit_costs.yaml` as `corroboration` or not at all.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"
# shellcheck source=../build_provenance.sh
. "$HERE/../build_provenance.sh"

OUT="$PWD/dramratebench-write-session"
PREFLIGHT=0
LIST=0
DO_RESET=1
SUSTAINED_ONLY=0
SKIP_BUILD=0
RESET_CMD="tt-smi -r 0"

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --preflight) PREFLIGHT=1; shift ;;
    --list) LIST=1; shift ;;
    --no-reset) DO_RESET=0; shift ;;
    --sustained-only) SUSTAINED_ONLY=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --reset-cmd) RESET_CMD="$2"; shift 2 ;;
    -h|--help) sed -n '2,55p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# The schedule and the estimate.
# ---------------------------------------------------------------------------
# Run 1 is the session: BOTH directions in one file, at the full sweep, so the
# read arm sits next to the write arm in the same device state and the same
# three repeats. The read arm inside it is the CONTROL and must reproduce the
# 2026-08-17 numbers; if it does not, the write arm next to it is not
# comparable to anything and the handover must say so rather than be re-run.
#
# Run 2 is an independent second sample of the write direction alone, at the
# vendor's own counts. It exists because two runs of one program on one card on
# one day is the standard this project already holds `nocbench` to, and because
# a write result nobody sampled twice is a write result with no spread on it.
if [ "$SUSTAINED_ONLY" -eq 1 ]; then
  RUN1_ARGS="--dir both --sustained"
  run1_min=6
else
  RUN1_ARGS="--dir both"
  run1_min=15
fi
RUN2_ARGS="--dir write --sustained"
run2_min=4
reset_min=2
total_min=$((run1_min + run2_min))
[ "$DO_RESET" -eq 1 ] && total_min=$((total_min + reset_min))

echo "dramratebench WRITE session"
echo "  run 1     : dramratebench $RUN1_ARGS      (~$run1_min min) -- the session."
echo "              Its READ rows are the control and must reproduce 2026-08-17."
echo "  run 2     : dramratebench $RUN2_ARGS      (~$run2_min min) -- a second,"
echo "              independent sample of the write direction alone."
if [ "$DO_RESET" -eq 1 ]; then
  echo "  reset     : $RESET_CMD, then 60 s        (~$reset_min min) -- BEFORE run 1"
else
  echo "  reset     : SKIPPED (--no-reset). Read the RESET section below."
fi
echo "  wall      : ~$total_min min of card time, plus ~2 min for a cold build"
echo "  out       : $OUT"
echo ""
echo "RESET, AND WHY IT IS NOT OPTIONAL HOUSEKEEPING"
echo "  A heavy NoC run on 2026-08-17 left board state that silently corrupted"
echo "  the DATA of the next program on the same card -- no abort, no"
echo "  diagnostic, just wrong bytes. This session's whole result is a rate"
echo "  measured next to an addressing proof, and corrupted state can fail the"
echo "  proof (which is survivable, the run is refused) or shift the rate"
echo "  (which is not, because it looks exactly like a measurement). Reset."

# ---------------------------------------------------------------------------
# THE PRE-REGISTRATION. Written down BEFORE the card runs, and copied into the
# session directory so that what was predicted travels with what was measured.
# ---------------------------------------------------------------------------
read -r -d '' PREDICTIONS <<'PRED'
PRE-REGISTERED 2026-08-17, before any Wormhole card ran the write direction.

WHAT THE SIMULATOR SAYS, ON BOTH PARTS, AT FOUR TILES AND 4096 B TRANSACTIONS
  Wormhole   onechan read 23.406 B/cycle, onechan write 23.327. Ratio 0.997.
             Both sit on the 24 B/cycle channel rate `dram.channel_serialisation`
             holds for BOTH directions on that arch, so the model predicts a
             SYMMETRIC endpoint and predicts it for a reason that is visible in
             the source: Wormhole converts one published per-channel figure,
             which its page states for reads and writes alike.
  Blackhole  onechan read 44.364 B/cycle, onechan write 60.208. Ratio 1.357 --
             the write is CHEAPER. Not because anything measured it: that arch
             has a `vendor_source_derived` READ rate of 47.08 B/cycle and NO
             write rate at all, so `DramChannels.write_bytes_per_cycle` is None
             and every write claim is a no-op. The write arm is therefore
             unqueued and runs up against the 64 B/cycle NoC link instead.
             The predicted asymmetry is the SHAPE OF AN ABSENCE, not of a
             measurement, and a card that shows Blackhole writes at the read's
             rate would be saying the model under-charges there.

  So tt-sim predicts OPPOSITE SIGNS on the two parts, and the Wormhole card's
  own latency data leans the third way (write dearer). Three positions, and
  this campaign can only adjudicate the occupancy one.

WORMHOLE, IF THE WRITE ASYMMETRY IS REAL AND WRITES ARE DEARER
  * onechan-write plateaus BELOW onechan read by more than the run-to-run
    spread. The read arm's spread across 3 repeats on 2026-08-17 was under 1%,
    so "below" means a write/read ratio under 0.95 at 48 and 64 participants,
    repeatably, in BOTH runs.
  * If the +21/+56-cycle latency gap were an occupancy gap it would be a large
    one: those are ~20% and ~45% of the read's own ~120-cycle endpoint
    difference, so a write plateau in the 15-20 B/cycle band -- not 21.9.
  * fanchan-write must still scale (x1.5 or better), or the flatness is
    unreadable and this prediction cannot be tested at all.

WORMHOLE, IF THE LATENCY RESULT WAS AN ARTEFACT OF THE BARRIER
  * write/read ratio inside 0.95-1.05 at EVERY participant count, with both
    arms plateauing together at 21.9-22.2 B/cycle.
  * THIS IS A POSITIVE RESULT AND NOT A NULL. It says the +21/+56 cycles a
    latency probe sees are issue- or completion-side -- how long
    noc_async_write_barrier takes to retire -- and not endpoint occupancy, so
    they belong nowhere near `dram.access_latency`. It also agrees with
    tt-sim's Wormhole prediction, and would leave `unit_costs.yaml`'s decision
    to keep one figure for all three request actions standing on a second,
    independent kind of evidence.

WHAT WOULD MAKE THE RESULT UNINTERPRETABLE (any one of these; report it, do
not re-run and keep the better session)
  * fanchan-write scales under x1.5. Something upstream of the endpoint caps
    both write arms and a flat concentrated arm says nothing. NOT hypothetical:
    tt-sim does exactly this at four tiles, where the four workers share a
    router row and the write control tops out near one link's rate. The card's
    64 workers are the reason to expect it to move here, and the read arm's own
    x5.74 on the same part is the evidence that it can.
  * any row with witness_ok < num_readers, or stray_writes > 0. Bytes did not
    go where the plan named, or went somewhere it did not.
  * any multi-participant point with max_barrier_spins == 0. The bursts never
    overlapped, and N writers running one after another produce exactly the
    flat aggregate this campaign is looking for.
  * the READ rows in the same file do not reproduce 2026-08-17: onechan flat in
    21.9-22.2 B/cycle from 1 to 64 (the n=2 point sat at 20.3 and may again),
    fanchan 22.2 -> 127.5 (x5.74), and -0.2 / -1.5 / -0.9 % against the
    published 22.2 / 22.3 / 22.3 at 1 / 12 / 48. If the control moved, the part
    or the session is not the one those figures came from.
  * a write plateau at or above 32 B/cycle. That is the NoC link on Wormhole,
    not an endpoint rate, and the run has sized the link.

WHAT THE ANSWER CAN BE USED FOR, WHATEVER IT IS
  Corroboration. Never provenance. `dram.bandwidth` is `unknown` on Wormhole
  for want of a published document and a measurement is not a document. And
  because the `samecore` arm does not exist on this part, a clean result is
  about "the endpoint", not about "the GDDR6 channel".
PRED

echo ""
echo "$PREDICTIONS"

# The block above is framed around WORMHOLE's question. Blackhole's is a
# different one, and reading the Wormhole framing at a Blackhole part gives the
# wrong verdict for the right numbers -- the same defect nocevbench had when it
# quoted Blackhole's control band at a Wormhole part.
#
# BOTH are printed, unconditionally, rather than detected. The predictions print
# before any device is opened (`--list` must cost nothing), so the part is not
# known here -- and pre-registering both is stronger than picking one at
# runtime anyway: neither can be chosen after seeing a number.
cat <<'BHPRED'

  IF THIS IS A BLACKHOLE PART, THE QUESTION IS A DIFFERENT ONE
  ------------------------------------------------------------------
  The block above is WORMHOLE's framing ("real and dearer" against "barrier
  artefact"), and it does not apply on Blackhole, which already has its split:
  a DRAM write's SERVICE LATENCY is 22 cycles against a read's 126.

  What Blackhole is being asked is sharper, and it is about an ABSENCE.
  tt-sim has NO Blackhole write rate at all: `write_bytes_per_cycle` is
  None, so every write claim is a no-op and the write arm runs up on the
  64 B/cycle link. That is why the simulator predicts

      onechan read 44.364 B/cycle,  onechan write 60.208.  Ratio 1.357

  i.e. it says a Blackhole write is 36 % FASTER than a read -- for no
  reason except that nobody has given it a figure. That number is the
  shape of a missing value, not of a model.

  SO THE PRE-REGISTERED QUESTION HERE IS: does the card agree?

    NOT FASTER  write/read at or below ~1.05. The 1.357 is then exposed as
                the artefact it is, and this session is the FIRST constraint
                on a value that does not currently exist. Enters as
                corroboration for a MISSING number, which is a cleaner
                position than correcting a wrong one.
    FASTER      write/read at or above ~1.25, repeatably. The absence was
                accidentally benign, and the session sizes what the figure
                should have been.
    BETWEEN     1.05 to 1.25. Report the level and the spread; it constrains
                the value without settling the shape.

  CAREFUL, and this is the part most likely to be got wrong: the 22-vs-126
  split is a LATENCY, and this is an OCCUPANCY measurement. They are
  different quantities. Agreement is not automatic and DISAGREEMENT WOULD
  NOT FALSIFY THE SPLIT. Do not report this session as a test of it.

  Everything else -- witness_ok == N, stray_writes == 0, the read rows
  reproducing their own control, the fan-out gate -- is arch-independent
  and applies on both parts unchanged. Those are what make the session valid
  at all.
BHPRED

[ "$LIST" -eq 1 ] && exit 0

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
# `detail::LaunchProgram` is the direct path and needs slow dispatch. A card
# defaults to FAST dispatch, so without this every run aborts with rc=134 before
# doing any work -- measured on a p150, 2026-08-17.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

# ---------------------------------------------------------------------------
# Pre-flight. Everything here fails for free; everything after it costs card
# time and, once the board is reset, costs somebody else's too.
# ---------------------------------------------------------------------------
fail=0
echo ""
echo "== pre-flight"

if [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "   FAIL TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR). This is the CARD"
  echo "        side; a session recorded through the simulator would pass every"
  echo "        gate below and be labelled a card's. Unset it."
  fail=1
else
  echo "   ok   TT_METAL_SIMULATOR is unset"
fi

if [ -d "$TT_METAL_HOME/build/lib" ]; then
  echo "   ok   tt-metal build tree at $TT_METAL_HOME/build"
else
  echo "   FAIL no $TT_METAL_HOME/build/lib; TT_METAL_HOME must be a BUILT checkout"
  fail=1
fi

if [ "$DO_RESET" -eq 1 ]; then
  if command -v "${RESET_CMD%% *}" >/dev/null 2>&1; then
    echo "   ok   reset command '${RESET_CMD%% *}' is on PATH"
  else
    echo "   FAIL '${RESET_CMD%% *}' not found. Install it, pass --reset-cmd, or"
    echo "        --no-reset and say so in the handover."
    fail=1
  fi
fi

for f in "$SRC/kernels/dataflow/dram_reader.cpp" "$SRC/kernels/dataflow/dram_writer.cpp"; do
  if [ -f "$f" ]; then
    echo "   ok   $(basename "$f") present"
  else
    echo "   FAIL missing $f"
    fail=1
  fi
done

# The write direction allocates a SECOND region of the same size as the read
# one, so the DRAM buffer doubles: 12 banks x 16 slices x 1 MiB x 2 = 384 MiB on
# a Wormhole. Cheap on any part that has DRAM at all, and named here because an
# allocation failure at run time costs the card time this check does not.
echo "   ok   write direction doubles the DRAM buffer (~384 MiB on a 12-bank part)"

# "A build exists" is not "the build matches the tt-metal that will be loaded".
# This session died at rc=127 on a symbol lookup once, after the board reset,
# because those two are different claims -- see build_provenance.sh.
bp_check_build "$SRC" "$([ "$SKIP_BUILD" -eq 1 ] && echo skip-build || echo build)" || fail=1

if [ "$fail" -eq 0 ] && [ "$SKIP_BUILD" -eq 0 ]; then
  echo "   .... building (first time ~2 min)"
  if ( cd "$SRC" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null \
       && cmake --build build -j"$(nproc)" >/dev/null ); then
    bp_record_build "$SRC"
    echo "   ok   build/dramratebench is current"
  else
    echo "   FAIL build failed; run cmake by hand in $SRC to see why"
    fail=1
  fi
elif [ "$SKIP_BUILD" -eq 1 ] && [ -x "$SRC/build/dramratebench" ]; then
  echo "   ok   build/dramratebench exists (--skip-build)"
elif [ "$SKIP_BUILD" -eq 1 ]; then
  echo "   FAIL --skip-build but there is no build/dramratebench"
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "pre-flight FAILED. No card time was spent and the board was not reset."
  exit 1
fi
echo "   pre-flight PASSED"

mkdir -p "$OUT"
printf '%s\n' "$PREDICTIONS" > "$OUT/PREDICTIONS.txt"
{
  echo "dramratebench write session"
  echo "date        : $(date -Iseconds)"
  echo "host        : $(hostname)"
  echo "tt-metal    : $TT_METAL_HOME"
  echo "tt-metal rev: $(cd "$TT_METAL_HOME" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "run 1       : dramratebench $RUN1_ARGS"
  echo "run 2       : dramratebench $RUN2_ARGS"
  echo "reset       : $([ "$DO_RESET" -eq 1 ] && echo "$RESET_CMD + 60 s" || echo SKIPPED)"
} > "$OUT/env.txt"
echo "   pre-registration written to $OUT/PREDICTIONS.txt BEFORE the card ran"

[ "$PREFLIGHT" -eq 1 ] && { echo ""; echo "--preflight: stopping before any card time."; exit 0; }

# ---------------------------------------------------------------------------
# The board reset.
# ---------------------------------------------------------------------------
if [ "$DO_RESET" -eq 1 ]; then
  echo ""
  echo "== board reset: $RESET_CMD"
  echo "   (a heavy NoC run on 2026-08-17 left state that silently corrupted the"
  echo "    DATA of the next program, with no abort and no diagnostic)"
  # shellcheck disable=SC2086
  $RESET_CMD || { echo "   reset FAILED; stopping rather than measuring through it"; exit 1; }
  echo "   sleeping 60 s for the board to come back"
  sleep 60
fi

# ---------------------------------------------------------------------------
# The runs.
# ---------------------------------------------------------------------------
status=0
run_one() { # label, args...
  local label="$1"; shift
  echo ""
  echo "== $label: dramratebench $*"
  # shellcheck disable=SC2086
  ( cd "$SRC" && ./build/dramratebench "$@" --out "$OUT/$label.csv" ) \
      > "$OUT/$label.out" 2>&1
  local rc=$?
  if [ ! -s "$OUT/$label.csv" ]; then
    echo "   FAILED (rc=$rc), no CSV; send $OUT/$label.out"
    status=1
    return
  fi
  echo "   rc=$rc -> $OUT/$label.csv"
  sed -n '/WRITE DIRECTION/,$p' "$OUT/$label.out" | head -40
}

# shellcheck disable=SC2086
run_one run1 $RUN1_ARGS
# shellcheck disable=SC2086
run_one run2 $RUN2_ARGS

# ---------------------------------------------------------------------------
# On-the-spot checks. These run at the card so a bad session is known to be bad
# before the operator walks away, not a week later at home.
# ---------------------------------------------------------------------------
echo ""
echo "== on-the-spot checks"
if command -v python3 >/dev/null 2>&1; then
  for label in run1 run2; do
    [ -s "$OUT/$label.csv" ] || continue
    python3 "$HERE/check_run.py" --measured "$OUT/$label.csv" \
        --schema-against 2>&1 | tee "$OUT/$label.check" | sed 's/^/   /'
    grep -q '^FAIL' "$OUT/$label.check" && status=1
  done
  echo ""
  echo "   the READ control, re-derived from run1's own read rows:"
  python3 "$HERE/check_run.py" --read-control "$OUT/run1.csv" 2>&1 | sed 's/^/   /'
  echo "   (a FAIL here is expected if run1 was --sustained-only: the pinned"
  echo "    control has ten reader counts and that preset sweeps four)"
else
  echo "   no python3 on this box; the checks run at home instead:"
  echo "     perfbench/dramratebench/check_run.py --measured run1.csv"
fi

cat <<EOF

== check by eye before you send it

  1. VERDICT (write) is not DEGENERATE in either run. A FLAT concentrated write
     curve is the EXPECTED answer and is NOT on its own a result -- it is a
     result only when fanchan-write MOVED.
  2. 'addressing:' reports 0 points failing the witness check and 0 with a
     stray write, in BOTH runs. A write that missed its endpoint completes,
     retires its barrier and yields a plausible rate; this line is the only
     thing standing between that and a published number.
  3. the READ rows of run1 reproduce 2026-08-17: onechan flat in 21.9-22.2
     B/cycle, fanchan 22.2 -> 127.5. If they do not, say so in the handover --
     do NOT reset and re-run until they do.
  4. run1 and run2 agree about the write direction. They are separate runs of
     an identical program; a large spread means something else was on the card.
  5. the SUSTAINED WRITE vs READ block: read it against $OUT/PREDICTIONS.txt,
     which was written before the card ran.

== send it home
  rsync -av $OUT/ <home>:~/dramratebench-write-session/

  At home:
    python3 perfbench/dramratebench/check_run.py --measured run1.csv
    python3 -m tt_sim.perf.dram_rate_sweep --measured run1.csv   # read arm only
EOF

exit "$status"
