#!/usr/bin/env bash
# One card session: every silicon probe ROADMAP.md item 2 asks for, in one run.
#
# This is the HARDWARE side. It builds the five perfbench programs, runs each
# probe the roadmap names, labels every output with the probe and the part, and
# prints a per-probe verdict so you can tell a real reading from a degenerate
# one before you pack the results up. Do NOT use perfbench/run.sh here -- that
# points TT_METAL_SIMULATOR at tt-sim.
#
#   export TT_METAL_HOME=/path/to/your/built/tt-metal
#   perfbench/run_card_session.sh                 # everything that applies
#   perfbench/run_card_session.sh --list          # what would run, and why
#   perfbench/run_card_session.sh nocread rv      # just those probes
#   perfbench/run_card_session.sh --resume        # skip probes already done
#
# Options:
#   --arch wormhole|blackhole  skip auto-detection and assert the part
#   --out DIR                  results directory
#                              (default: ~/tt_traces/card-session-<arch>)
#   --plan FILE                a congestion plan built at home, so the card box
#                              needs no tt_sim/. Defaults to
#                              nocbench/noc-plan-<arch>.csv if that exists.
#                              Verified against this card's live grid dump.
#   --list                     list probes, their parts and cost, then exit
#   --resume                   skip any probe whose CSV is already in --out
#   --dry-run                  print the commands without running them
#   --sim                      VALIDATION ONLY -- run against tt-sim at smoke
#                              sizes. Every output is stamped NOT-A-MEASUREMENT.
#   -h, --help                 this text
#
# Runtime: about 105 minutes cold, of which ~70 is building the five programs;
# about 40 minutes if the build trees already exist. `--list` prints the
# per-probe breakdown. Nothing here writes a card's flash, changes any clock,
# or needs root.
#
# SPDX-License-Identifier: Apache-2.0

set -u

PB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PB/.." && pwd)"

# The verdict logic is a separate, pure library so it can be run away from the
# card against files a card returned -- see card_session_verdicts_test.sh, which
# feeds it the 2026-08-09 Blackhole CSVs and asserts it calls them degenerate.
# shellcheck source=card_session_verdicts.sh
. "$PB/card_session_verdicts.sh"

ARCH=""
OUT=""
DO_LIST=0
DO_RESUME=0
DO_DRY=0
DO_SIM=0
PLAN_IN=""
WANT=()

# ---------------------------------------------------------------- probe table
# name | parts | est.min | roadmap bullet it discharges
PROBES=(
  "nocread|both|3|nocreadbench on both parts: the read-rate floor, E0-E7"
  "cmdbuf|blackhole|0|CMD_BUF_AVAIL at rest -- the depth the ISA docs withhold"
  "tensix|both|6|a second tensixbench sample; every rung-3 result rests on one"
  "tensix-warm|both|4|drop tensixbench's cold n=32 burst (--blocks 64)"
  "tensix-rdcfg|both|4|RDCFG LATENCY as a producer-to-consumer DISTANCE, plus the C12 liveness control"
  "dram|both|7|N tiles reading ONE DRAM channel: the endpoint-occupancy shape"
  "rv|both|8|riscvbench primary + a repeat cross-check at the SAME parameters"
  "rv-gset|both|3|phase-G footprint ramp, --gset 1..4 (4608/5632 B are new)"
  "rv-qdrain|both|2|Tensix queue depth and whether it is shared (.ttinsn burst)"
  "rv-pairs|both|1|store-coalescing pair, multiply pair, divide, in one focused run"
  "noc|both|5|congestion, with points filled in between the 64 B and 16 KiB regimes"
  "noc-epoch|both|5|a second congestion run: the clock-epoch detector needs two"
)

# Roadmap bullets this session cannot discharge because they need the other
# part. These are NOT "not applicable" and they are not stuck: Wormhole is a
# planned follow-on session. What matters is that a successful full run leaves
# them exactly as open as it found them, so they are printed after every run.
# Every probe below already runs on Wormhole -- pass `--arch wormhole` and the
# same block executes -- so the follow-on is a hardware booking, not new work.
DEFERRED_TO_WORMHOLE=(
  "the store-coalescing pair and the multiply pair on Wormhole -- predicted identical there, measured 5.2x apart on Blackhole. The cheapest cross-arch discriminators. The rv-pairs probe already produces them; it just needs the part."
  "the Wormhole half of nocreadbench. The read-floor claim IS a per-arch difference (Wormhole 25.00 cycles/tx, 17.33 with a shorter issue loop; Blackhole 35.0 and 34.0). This session can size or retire the BLACKHOLE limit on its own, but it cannot settle the per-arch question."
)

probe_field() { # name, field-index (1-based)
  local p
  for p in "${PROBES[@]}"; do
    case "$p" in "$1|"*) printf '%s' "$p" | cut -d'|' -f"$2"; return 0 ;; esac
  done
  return 1
}

probe_names() {
  local p
  for p in "${PROBES[@]}"; do printf '%s\n' "${p%%|*}"; done
}

# ------------------------------------------------------------------ arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="${2:-}"; shift 2 || exit 2 ;;
    --out) OUT="${2:-}"; shift 2 || exit 2 ;;
    --plan) PLAN_IN="${2:-}"; shift 2 || exit 2 ;;
    --list) DO_LIST=1; shift ;;
    --resume) DO_RESUME=1; shift ;;
    --dry-run) DO_DRY=1; shift ;;
    --sim) DO_SIM=1; shift ;;
    -h|--help) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *)
      if probe_field "$1" 1 >/dev/null; then WANT+=("$1"); else
        echo "unknown probe '$1'; try --list" >&2; exit 2
      fi
      shift ;;
  esac
done

case "${ARCH:-}" in
  ""|wormhole|blackhole) ;;
  wh) ARCH=wormhole ;;
  bh) ARCH=blackhole ;;
  *) echo "--arch must be wormhole or blackhole" >&2; exit 2 ;;
esac

if [ "$DO_LIST" = 1 ]; then
  printf '%-14s %-10s %6s  %s\n' PROBE PART MIN "DISCHARGES"
  total=0
  for p in "${PROBES[@]}"; do
    n="$(printf '%s' "$p" | cut -d'|' -f1)"
    a="$(printf '%s' "$p" | cut -d'|' -f2)"
    m="$(printf '%s' "$p" | cut -d'|' -f3)"
    d="$(printf '%s' "$p" | cut -d'|' -f4)"
    printf '%-14s %-10s %6s  %s\n' "$n" "$a" "$m" "$d"
    total=$((total + m))
  done
  echo
  echo "run time once built: ~${total} min. Cold, add ~70 min to build the five"
  echo "programs (tensixbench and riscvbench are ~15 min each)."
  echo
  echo "Awaiting the planned Wormhole follow-on (every probe below already runs"
  echo "there -- pass --arch wormhole -- so it is a hardware booking, not work):"
  for d in "${DEFERRED_TO_WORMHOLE[@]}"; do
    printf '  - %s\n' "$d" | fold -s -w 74 | sed '2,$s/^/    /'
  done
  echo
  echo "Not built, by design -- see perfbench/README.md 'Designed, not built':"
  echo "  ATCAS / ATINCGETPTR against a real L1 semaphore   (needs new probes)"
  echo "  the TTBENCH_UNROLL sweep over {16,32,64,128}       (needs build plumbing)"
  echo "  a divide magnitude sweep                           (dividend is hardcoded)"
  echo "  the DIR_BIDIR hang                                 (refused on purpose)"
  exit 0
fi

# ------------------------------------------------------------------ the world
: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Slow dispatch: tt-sim only supports the direct launch path, and using the same
# mode on hardware keeps the two runs comparable.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

if [ "$DO_SIM" = 1 ]; then
  # Validation path. This exists so the session block can be exercised end to
  # end without a card; it is NOT a measurement and every artefact says so.
  [ -n "$ARCH" ] || ARCH=blackhole
  export TT_METAL_SIMULATOR="$REPO/driver/$ARCH"
  # TT_SIM_TENSIX_COORDS is deliberately NOT defaulted. Exporting it -- even to
  # the one worker the server builds anyway -- is how tt-sim is told the pool is
  # PINNED, and a pinned pool switches off on-demand materialisation. This block
  # used to export the arch's default coord, which is why the congestion probes
  # could not run against the simulator: the multi-core plan died on the first
  # kernel launch outside the pool. Unset, the server builds the same default
  # worker and materialises the rest as the program reaches them.
  VENV="${TT_SIM_VENV:-$REPO/../venv}"
  [ -x "$VENV/bin/python3" ] && export PATH="$VENV/bin:$PATH"
  echo "!! --sim: running against tt-sim at smoke sizes. NOT A MEASUREMENT."
elif [ -n "${TT_METAL_SIMULATOR:-}" ]; then
  echo "TT_METAL_SIMULATOR is set ($TT_METAL_SIMULATOR)." >&2
  echo "This script is for a real card. Unset it, or pass --sim if you meant" >&2
  echo "to validate the session block against tt-sim." >&2
  exit 2
fi

# Is the analysis half available? The four C++ programs need nothing but a
# built tt-metal, so `rsync perfbench/` is enough to COLLECT every reading. The
# nocbench planner and the three report generators live in tt_sim/perf, which
# is normally NOT on a card box. Rather than fail, the session collects what it
# can and says which steps were deferred to the analysis box. Analysis of a CSV
# is not time-critical; being at the card is.
#
# Probed HERE, after the block above, and not earlier: `--sim` puts tt_sim's
# venv on PATH, and tt_sim/perf imports numpy and pyelftools. Probed before
# that, the answer was the SYSTEM python3's, which on this box cannot import it
# -- so a --sim session collected both congestion CSVs and then reported them
# DEFERRED "because tt_sim/ is not on this box", standing in the repo.
HAVE_TT_SIM=0
if [ -d "$REPO/tt_sim/perf" ] && \
   PYTHONPATH="$REPO" python3 -c "import tt_sim.perf.noc_congestion_plan" 2>/dev/null; then
  HAVE_TT_SIM=1
fi

# ------------------------------------------------------------ arch detection
detect_arch() {
  # nocbench --dump-grid runs a probe kernel and writes `arch=` into its header.
  # It is the cheapest thing in the tree that asks the device what it is, and
  # the `noc` probe needs the same file, so this is not wasted work.
  local grid="$1"
  ( cd "$PB/nocbench/src" || exit 2
    [ -x build/nocbench ] || {
      cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
      cmake --build build -j >/dev/null || exit 1; }
    ./build/nocbench --dump-grid --out "$grid" >/dev/null 2>&1 ) || return 1
  sed -n 's/.*arch=\([a-z0-9_]*\).*/\1/p' "$grid" | head -1
}

# Results go to ~/tt_traces, which is where this card box keeps them. (An
# earlier revision defaulted to /mnt/ramdisk; that path does not exist on the
# box and the fallback to $PWD scattered CSVs wherever the session was started
# from.) A session's CSVs are small and written repeatedly, so a ramdisk is
# still preferred if one happens to be mounted there.
# Detect FIRST, then name the directory. The previous order computed
# `card-session-${ARCH:-detect}` before detection ran, so the grid dump the
# detector needs landed in a directory called `card-session-detect` and a rename
# afterwards left it behind: the 2026-08-09 results came back with a stray
# `card-session-detect/nocbench-grid.csv` inside the real directory. One session
# writes one directory, and the detector's own dump is the first file in it.
if [ -z "$ARCH" ]; then
  if [ "$DO_DRY" = 1 ]; then
    ARCH=unknown
  else
    echo "== detecting the part"
    DETECT_TMP="$(mktemp -d "${TMPDIR:-/tmp}/card-session-detect.XXXXXX")" || exit 2
    ARCH="$(detect_arch "$DETECT_TMP/nocbench-grid.csv")"
    if [ -z "$ARCH" ]; then
      rm -rf "$DETECT_TMP"
      echo "could not detect the part; pass --arch" >&2
      exit 1
    fi
    echo "   arch=$ARCH"
  fi
fi

if [ -z "$OUT" ]; then
  OUT="$HOME/tt_traces/card-session-${ARCH}"
fi
mkdir -p "$OUT" || exit 2
GRID="$OUT/nocbench-grid.csv"
if [ -n "${DETECT_TMP:-}" ]; then
  # The `noc` probe needs the same dump, so detection is not wasted work.
  [ -s "$DETECT_TMP/nocbench-grid.csv" ] && cp "$DETECT_TMP/nocbench-grid.csv" "$GRID"
  rm -rf "$DETECT_TMP"
fi

LOG="$OUT/session.log"
[ "$DO_DRY" = 1 ] || : > "$LOG"

say() { echo "$@"; [ "$DO_DRY" = 1 ] || echo "$@" >> "$LOG"; }

run() { # everything else is logged through here
  if [ "$DO_DRY" = 1 ]; then echo "   + $*"; return 0; fi
  "$@" >> "$LOG" 2>&1
}

# A probe's verdict. Every probe ends in exactly one of these lines, and the
# handover checklist tells the user to read them before packing up. A probe
# whose control did not move must say so rather than let a flat reading pass as
# a result -- that is the rule nocbench's INVALID and nocreadbench's DEGENERATE
# already set, and it is why this block greps rather than trusting exit status.
VERDICTS=()
verdict() { VERDICTS+=("$1|$2|$3"); say "   VERDICT $1: $2 -- $3"; }

skip() { verdict "$1" SKIPPED "$2"; }

# Every probe below reaches its verdict through one of the pure functions in
# card_session_verdicts.sh, which echo `STATUS|text`. Going through this
# wrapper is what keeps the graded logic testable: nothing in a probe body
# decides a status.
graded() { # probe name, then the verdict function and its arguments
  local name="$1"; shift
  local line; line="$("$@")"
  verdict "$name" "${line%%|*}" "${line#*|}"
}

# ------------------------------------------------------------------- selection
# Naming a congestion probe on the command line is what opts a --sim run in to
# it; see probe_noc for why it cannot be part of the default sim set.
SIM_NOC_OPT_IN=0
case " ${WANT[*]-} " in *" noc "*|*" noc-epoch "*) SIM_NOC_OPT_IN=1 ;; esac

if [ ${#WANT[@]} -eq 0 ]; then
  while IFS= read -r n; do WANT+=("$n"); done < <(probe_names)
fi

applies() { # probe name -> 0 if it runs on this part
  local parts; parts="$(probe_field "$1" 2)"
  [ "$parts" = both ] && return 0
  [ "$parts" = "$ARCH" ] && return 0
  return 1
}

done_already() { # probe name -> 0 if --resume should skip it
  [ "$DO_RESUME" = 1 ] || return 1
  ls "$OUT"/"$1".*.csv >/dev/null 2>&1
}

# ---------------------------------------------------------------- the probes
SIM_NOTE=""
[ "$DO_SIM" = 1 ] && SIM_NOTE=" (SIM SMOKE -- NOT A MEASUREMENT)"

TB="$PB/tensixbench/src"
RB="$PB/riscvbench/src"
NRB="$PB/nocreadbench/src"
NB="$PB/nocbench/src"
DRB="$PB/dramratebench/src"

build_once() { # dir, binary
  [ "$DO_DRY" = 1 ] && { echo "   + build $2"; return 0; }
  if [ -x "$1/build/$2" ]; then
    # A card box keeps its build trees between sessions, so "the binary exists"
    # is not "the binary matches the source". A host program built before a
    # result-layout change reads the new kernel's stamp as garbage and every
    # point fails -- loudly, because the magic is checked, but only after the
    # card time has been spent. Rebuild if anything outside build/ is newer.
    local newer
    # Sources only: these trees also accumulate CSVs, and a bench run writing
    # its own output next to the binary must not trigger a rebuild next time.
    newer="$(find "$1" -path "$1/build" -prune -o -type f \
               \( -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) \
               -newer "$1/build/$2" -print 2>/dev/null | head -1)"
    [ -z "$newer" ] && return 0
    say "   rebuilding $2 (source newer than the binary: $(basename "$newer"))"
  else
    say "   building $2 (first time; this is the slow part)"
  fi
  # A CMakeCache.txt records the ABSOLUTE path it was generated in, so a build
  # tree that arrived by rsync from another machine makes cmake refuse outright
  # ("the current CMakeCache.txt directory ... is different than the directory
  # ... where CMakeCache.txt was created"). That is exactly what `rsync -av
  # perfbench/` does if the sending box ever built here, and it cost a card
  # session on 2026-08-09. A build tree is a derived artefact and always safe to
  # discard, so recognise a foreign cache and start clean rather than fail.
  if [ -f "$1/build/CMakeCache.txt" ] && \
     ! grep -qxF "CMAKE_CACHEFILE_DIR:INTERNAL=$1/build" "$1/build/CMakeCache.txt" 2>/dev/null; then
    say "   discarding a build tree generated elsewhere (rsync'd CMakeCache)"
    rm -rf "$1/build"
  fi
  ( cd "$1" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1 \
      && cmake --build build -j >/dev/null 2>&1 ) || {
    # A second chance from scratch: a cache can also be stale in ways the path
    # check cannot see (a moved tt-metal, a changed compiler).
    say "   build failed; retrying from a clean tree"
    rm -rf "$1/build"
    ( cd "$1" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release >/dev/null \
        && cmake --build build -j >/dev/null ) || return 1
  }
  # Stamp the binary even when the link was a no-op. Kernel sources are JIT'd at
  # run time and never enter the host binary, so cmake leaves its timestamp
  # alone and the check above would fire again every session.
  touch "$1/build/$2" 2>/dev/null
}

probe_nocread() {
  # Safe on a HARVESTED part: every core here is chosen in LOGICAL space
  # (compute_with_storage_grid_size / worker_core_from_logical_core), so no
  # coordinate is ever picked arithmetically and a missing column cannot be
  # addressed. See the README for the one derived column that is NOT safe.
  build_once "$NRB" nocreadbench || { verdict nocread FAILED "build failed"; return; }
  local csv="$OUT/nocread.$ARCH.csv" args
  if [ "$DO_SIM" = 1 ]; then args="--num-tx 8 --repeats 1"; else args="--num-tx 64 --repeats 3"; fi
  # shellcheck disable=SC2086
  ( cd "$NRB" && ./build/nocreadbench $args --out "$csv" ) >"$OUT/nocread.out" 2>&1
  local r=$?
  cat "$OUT/nocread.out" >> "$LOG"
  [ -s "$csv" ] || { verdict nocread FAILED "no CSV (rc=$r); send nocread.out"; return; }
  graded nocread nocread_verdict "$OUT/nocread.out" "$csv"
  # The `hops` COLUMN is derived with a torus modulus of (logical grid + 2),
  # which is not the physical NoC width and is further off on a harvested part.
  # The coordinates it is derived FROM (mst_node_x/y, src0_x/y) are true
  # physical NoC 0 values, so the column is recomputable at home -- but do not
  # read E1's distance sweep straight off it. Recorded rather than silently
  # trusted.
  say "   note: recompute the 'hops' column at home from mst_node_x/y and"
  say "         src0_x/y -- its torus modulus is approximate on a harvested part"
}

probe_cmdbuf() {
  # Zero-cost: it is a column nocread already wrote. Named separately because
  # it answers its own roadmap bullet and has its own failure mode.
  local csv="$OUT/nocread.$ARCH.csv"
  [ -s "$csv" ] || { skip cmdbuf "needs the nocread probe to have run first"; return; }
  local rest busy peak
  rest="$(awk -F, -v c="$(_csv_col "$csv" cmdbuf_avail_rest)" '/^#/{next} !h{h=1;next} c>0 {print $c; exit}' "$csv")"
  busy="$(awk -F, -v c="$(_csv_col "$csv" cmdbuf_avail_busy)" '/^#/{next} !h{h=1;next} c>0 {print $c; exit}' "$csv")"
  peak="$(awk -F, -v c="$(_csv_col "$csv" cmdbuf_avail_max)" '/^#/{next} !h{h=1;next} c>0 {print $c; exit}' "$csv")"
  say ""
  say "   *********************************************************************"
  say "   * CMD_BUF_AVAIL   at rest ${rest:-<none>}   in-loop last ${busy:-<none>}   in-loop peak ${peak:-<none>}"
  say "   * Blackhole-only -- Wormhole has no such register. Four 5-bit fields:"
  say "   *   [28:24] [20:16] [12:8] [4:0]"
  say "   * It is an OCCUPANCY, not a count of free slots: reset default 0,"
  say "   * neighbour CMD_BUF_OVFL, and nothing in tt-metal reads it. So zero"
  say "   * at rest is the CORRECT reading and is not a depth. Only the peak,"
  say "   * and only if it moved, is evidence -- and even then it is a LOWER"
  say "   * BOUND on the depth unless CMD_BUF_OVFL moved too."
  say "   *********************************************************************"
  say ""
  # Bash arithmetic, not awk: `strtonum` is a gawk extension and a card box
  # running mawk would silently decode nothing. $((0x...)) is builtin.
  local n f
  if [ -n "${peak:-}" ] && n=$((peak)) 2>/dev/null; then
    f="$(( (n >> 24) & 0x1F )) $(( (n >> 16) & 0x1F )) $(( (n >> 8) & 0x1F )) $(( n & 0x1F ))"
    say "   peak decoded per-command-buffer occupancy [28:24] [20:16] [12:8] [4:0]: $f"
  fi
  graded cmdbuf cmdbuf_verdict "$csv"
}

probe_tensix() {
  build_once "$TB" tensixbench || { verdict tensix FAILED "build failed"; return; }
  local csv="$OUT/tensix.$ARCH.csv" args
  # --probes 0xFFFFF pins this to the twenty slots every tracked dataset in
  # tt_sim/perf/datasets/ was collected with. tensixbench gained slots 20 and 21
  # (the RDCFG latency difference) on 2026-08-09 and they default ON, so without
  # this the second sample rung 3 needs would silently be a different experiment
  # from the first -- and phase A's validity gate is per-PHASE, so one nonlinear
  # new slot would condemn all nineteen good series with it. The new slots have
  # their own probe.
  if [ "$DO_SIM" = 1 ]; then args="--phase a --blocks 1 --variants t1 --probes 0xFFFFF"
  else args="--blocks 32 --iters 64 --probes 0xFFFFF"; fi
  # shellcheck disable=SC2086
  ( cd "$TB" && ./build/tensixbench $args --out "$csv" ) >"$OUT/tensix.out" 2>&1
  cat "$OUT/tensix.out" >> "$LOG"
  [ -s "$csv" ] || { verdict tensix FAILED "no CSV; send tensix.out"; return; }
  # Read the cyc/instr column rather than grepping for "1.000" anywhere in the
  # file: the report also prints R^2 values and per-block figures, and an
  # earlier version of this check called a run of twenty exact 1.000s
  # MEANINGFUL because something elsewhere on the page had three decimals.
  # Columns are: probe variant unit thr cyc/block cyc/instr R^2
  say "   highest cyc/instr over all probes: $(_tensix_peak "$OUT/tensix.out")"
  graded tensix tensix_verdict "$OUT/tensix.out" tensix
}

probe_tensix_warm() {
  # The roadmap asks to "drop tensixbench's n = 32 burst or add a warm-up".
  # Dropping it needs no new code: the four fit points are always N,2N,3N,4N,
  # so --blocks 64 starts the fit at 64 and the cold first burst is gone.
  build_once "$TB" tensixbench || { verdict tensix-warm FAILED "build failed"; return; }
  local csv="$OUT/tensix-warm.$ARCH.csv" args
  if [ "$DO_SIM" = 1 ]; then args="--phase a --blocks 2 --variants t1 --probes 0xFFFFF"
  else args="--blocks 64 --iters 64 --phase a --probes 0xFFFFF"; fi
  # shellcheck disable=SC2086
  ( cd "$TB" && ./build/tensixbench $args --out "$csv" ) >"$OUT/tensix-warm.out" 2>&1
  cat "$OUT/tensix-warm.out" >> "$LOG"
  [ -s "$csv" ] || { verdict tensix-warm FAILED "no CSV; send tensix-warm.out"; return; }
  say "   highest cyc/instr over all probes: $(_tensix_peak "$OUT/tensix-warm.out")"
  graded tensix-warm tensix_verdict "$OUT/tensix-warm.out" tensix-warm
}

probe_tensix_rdcfg() {
  # The one quantity phase A is structurally blind to. Every other tensixbench
  # probe measures OCCUPANCY -- how long the issuing thread is kept out of a
  # unit -- and a pipelined unit releases its issuer immediately, so LATENCY is
  # invisible to all twenty of them. RDCFG is the case that made that matter:
  # the ISA doc gives it ">= 2", slot 14 measures 1.000 on silicon, and charging
  # the doc's 2 as an occupancy is what made tt-sim's matmulblock guard compute
  # the wrong answer.
  #
  # 0x3FF04601 = slots 0, 9, 10, 14, 20-29: the empty-loop control, the paired
  # ops BARE (so a difference can be shown to be latency and not occupancy),
  # slots 20/21 and 22-25 kept as the two documented negatives, the dependence
  # pair 26/27 and the C12 liveness control 28/29. Run in isolation because
  # phase A's validity gate is per-PHASE: a nonlinear new slot inside the
  # `tensix` probe would flip TTBENCH_VALID_A for all nineteen good series at
  # once.
  #
  # `--vis-reps` turns on the visibility sweep -- RDCFG's documented ">= 2" read
  # as a producer-to-consumer DISTANCE, which is the one construction the ISA
  # documentation supports for it. It runs on the t1 launch only, writes no CSV
  # row and issues no STALLWAIT.
  build_once "$TB" tensixbench || { verdict tensix-rdcfg FAILED "build failed"; return; }
  local csv="$OUT/tensix-rdcfg.$ARCH.csv" args
  if [ "$DO_SIM" = 1 ]; then args="--phase a --blocks 1 --variants t1 --probes 0x3FF04601 --vis-reps 4"
  else args="--phase a --blocks 32 --iters 64 --variants t1 --probes 0x3FF04601 --vis-reps 64"; fi
  # shellcheck disable=SC2086
  ( cd "$TB" && ./build/tensixbench $args --out "$csv" ) >"$OUT/tensix-rdcfg.out" 2>&1
  cat "$OUT/tensix-rdcfg.out" >> "$LOG"
  [ -s "$csv" ] || { verdict tensix-rdcfg FAILED "no CSV; send tensix-rdcfg.out"; return; }

  # THE C12 LIVENESS CONTROL, in its OWN run, and that is not tidiness. It is
  # graded across variants -- t1 for the stall's floor, t3 for the same stall
  # while two other threads hold the Configuration Unit -- so it needs a t3
  # launch, and t3 series are contended by construction. Phase A's validity gate
  # is per-PHASE, so one nonlinear contended series in the run above would flip
  # TTBENCH_VALID_A for the visibility measurement and every other slot with it.
  # Slots 0, 28 and 29 only: the empty-loop control the slope needs, and the
  # pair.
  local c12csv="$OUT/tensix-rdcfg-c12.$ARCH.csv" c12args
  if [ "$DO_SIM" = 1 ]; then c12args="--phase a --blocks 1 --variants t1,t3 --probes 0x30000001"
  else c12args="--phase a --blocks 32 --iters 64 --variants t1,t3 --probes 0x30000001"; fi
  # shellcheck disable=SC2086
  ( cd "$TB" && ./build/tensixbench $c12args --out "$c12csv" ) >"$OUT/tensix-rdcfg-c12.out" 2>&1
  cat "$OUT/tensix-rdcfg-c12.out" >> "$LOG"

  graded tensix-rdcfg tensix_rdcfg_verdict "$OUT/tensix-rdcfg.out" "$OUT/tensix-rdcfg-c12.out"
}

probe_dram() {
  # Safe on a HARVESTED part in both of its coordinate spaces, and neither is
  # derived arithmetically. The readers come from `compute_with_storage_grid_size`
  # in LOGICAL space, so a missing worker column is simply not addressable. The
  # DRAM banks come from `logical_core_from_dram_channel` + `get_bank_offset`,
  # which is what the allocator itself uses -- and every reader then reads back a
  # per-bank tag the host wrote, so a row that reached the wrong endpoint says so
  # instead of looking like a measurement.
  build_once "$DRB" dramratebench || { verdict dram FAILED "build failed"; return; }
  # Scoped to this probe's own run below rather than exported, so widening the
  # simulator's tile set for the reader sweep cannot slow or alter any probe
  # that runs after it.
  local csv="$OUT/dram.$ARCH.csv" args sim_coords=""
  if [ "$DO_SIM" = 1 ]; then
    # TWO tiles, not the session's default one. The reader count is the axis
    # this whole probe sweeps, and at one reader it cannot sweep at all -- the
    # barrier is never exercised, no burst ever overlaps another, and the run
    # checks nothing but that the binary starts. A simulator materialises only
    # the tiles in TT_SIM_TENSIX_COORDS while `compute_with_storage_grid_size()`
    # still reports the whole grid, so a second reader on an unlisted tile is
    # NullCore-swallowed and never stamps a result -- which the program reports
    # as a failure, correctly, but pointlessly. Name the second tile instead.
    # Logical (0,0) and (1,0) are the first two entries of the descriptor's
    # `functional_workers`: 1-2, 2-2 on Blackhole and 1-1, 2-1 on Wormhole.
    # An operator who has already asked for more tiles keeps them.
    case "$ARCH" in
      blackhole) local pair=1-2,2-2 ;;
      *) local pair=1-1,2-1 ;;
    esac
    case "${TT_SIM_TENSIX_COORDS:-}" in
      *,*) pair="$TT_SIM_TENSIX_COORDS" ;;      # already multi-tile; leave it
    esac
    sim_coords="$pair"
    args="--bytes 8192 --tx-bytes 512 --repeats 1 --readers 1,2 --max-readers 2"
  else
    # 1 MiB per reader is the figure `wh_dram#performance` states its own
    # measurement at, and 1 / 12 / 48 readers are the three counts it reports.
    args="--bytes 1048576 --tx-bytes 4096 --repeats 3"
  fi
  # shellcheck disable=SC2086
  ( cd "$DRB" || exit 2
    [ -n "$sim_coords" ] && export TT_SIM_TENSIX_COORDS="$sim_coords"
    ./build/dramratebench $args --out "$csv" ) >"$OUT/dram.out" 2>&1
  local r=$?
  cat "$OUT/dram.out" >> "$LOG"
  [ -s "$csv" ] || { verdict dram FAILED "no CSV (rc=$r); send dram.out"; return; }
  say "   $(grep -m1 'fanchan aggregate' "$OUT/dram.out" 2>/dev/null || echo 'no fanchan control line')"
  say "   $(grep -m1 'onechan aggregate' "$OUT/dram.out" 2>/dev/null || echo 'no onechan line')"
  # The sustained-rate table is the LEVEL, and the scaling verdict cannot see
  # it: a one-channel arm plateauing at 24 B/cycle and one plateauing at 40
  # give the same ratio. It is also the only reading comparable to the vendor's
  # published 22.2 / 22.3 / 22.3 GB/s at 1 / 12 / 48 tiles, so it belongs in
  # the session log rather than only in dram.out. The prediction it is to be
  # read against was committed BEFORE this run:
  # perfbench/dramratebench/prediction-sustained.csv.
  while IFS= read -r line; do
    say "   $line"
  done < <(awk '/SUSTAINED RATE/ { p = 1 } /VERDICT:/ { p = 0 } p' "$OUT/dram.out" 2>/dev/null)
  graded dram dram_verdict "$OUT/dram.out" "$csv"
  say "   note: this is CORROBORATION of a SHAPE and never provenance. Blackhole"
  say "         has no published DRAM tile page at all -- no per-channel rate, no"
  say "         address map -- so nothing measured here can make dram.bandwidth"
  say "         chargeable on it. What it can do is confirm that an aggregate"
  say "         refuses to grow with readers while the fan-out control grows."
}

probe_rv() {
  # The cross-check runs the SAME parameters as the primary. It used to run
  # --blocks 8, which riscvbench/README.md puts below the minimum usable block
  # count ("above 8 and at or below 32"), so it was guaranteed to fail its gates
  # on silicon and did -- R, C, Q, F and G on 2026-08-09, against the primary's
  # phase Q alone. A cross-check establishes repeatability; repeating the
  # experiment is the only way to do that. Each run also writes its OWN .out:
  # the two used to be concatenated into one file that the verdict greped, so
  # either run's failure condemned both.
  build_once "$RB" riscvbench || { verdict rv FAILED "build failed"; return; }
  local a1 a2
  if [ "$DO_SIM" = 1 ]; then a1="--phase r --blocks 1 --variants t1"; a2=""
  else a1="--blocks 32"; a2="--blocks 32"; fi
  # shellcheck disable=SC2086
  ( cd "$RB" && ./build/riscvbench $a1 --out "$OUT/rv.$ARCH.csv" ) >"$OUT/rv.out" 2>&1
  cat "$OUT/rv.out" >> "$LOG"
  [ -s "$OUT/rv.$ARCH.csv" ] || { verdict rv FAILED "no CSV; send rv.out"; return; }
  graded rv rv_verdict "$OUT/rv.out" rv
  [ -n "$a2" ] || return
  # shellcheck disable=SC2086
  ( cd "$RB" && ./build/riscvbench $a2 --out "$OUT/rv-cross.$ARCH.csv" ) >"$OUT/rv-cross.out" 2>&1
  cat "$OUT/rv-cross.out" >> "$LOG"
  [ -s "$OUT/rv-cross.$ARCH.csv" ] || { verdict rv-cross FAILED "no CSV; send rv-cross.out"; return; }
  graded rv-cross rv_cross_verdict "$OUT/rv-cross.out" "$OUT/rv.$ARCH.csv" "$OUT/rv-cross.$ARCH.csv"
}

probe_rv_gset() {
  build_once "$RB" riscvbench || { verdict rv-gset FAILED "build failed"; return; }
  [ "$DO_SIM" = 1 ] && { skip rv-gset "phase G is minutes per gset against tt-sim; covered on the card"; return; }
  # Four sets, not two. Sets 3 and 4 build the 4608 B and 5632 B bodies, the
  # two footprints INSIDE the (4096, 5120] bracket the ramp's onset sits in and
  # the two nothing has ever measured. With gset 0's 5120 B from the `rv` probe
  # that is five measured footprints across the ramp instead of three.
  local g ok=1 csvs=()
  for g in 1 2 3 4; do
    ( cd "$RB" && ./build/riscvbench --phase g --variants t1 --blocks 32 --gset "$g" \
        --out "$OUT/rv-gset$g.$ARCH.csv" ) >>"$OUT/rv-gset.out" 2>&1
    if [ -s "$OUT/rv-gset$g.$ARCH.csv" ]; then csvs+=("$OUT/rv-gset$g.$ARCH.csv"); else ok=0; fi
  done
  cat "$OUT/rv-gset.out" >> "$LOG" 2>/dev/null
  if [ "$ok" != 1 ]; then verdict rv-gset FAILED "one or more gset runs produced no CSV"; return; fi
  graded rv-gset rv_gset_verdict "$OUT/rv-gset.out" "${csvs[@]}"
}

probe_rv_qdrain() {
  build_once "$RB" riscvbench || { verdict rv-qdrain FAILED "build failed"; return; }
  local args
  if [ "$DO_SIM" = 1 ]; then args="--phase q --variants t1 --blocks 1"
  else args="--phase qs --variants t1 --blocks 32"; fi
  # shellcheck disable=SC2086
  ( cd "$RB" && ./build/riscvbench $args --out "$OUT/rv-qdrain.$ARCH.csv" ) >"$OUT/rv-qdrain.out" 2>&1
  cat "$OUT/rv-qdrain.out" >> "$LOG"
  [ -s "$OUT/rv-qdrain.$ARCH.csv" ] || { verdict rv-qdrain FAILED "no CSV; send rv-qdrain.out"; return; }
  # Phase Q carries no phase R, so riscvbench's live-instrument check cannot
  # run against it and there is nothing here to grade honestly in-session. Say
  # COLLECTED rather than claim a verdict the run did not earn -- but "the file
  # exists" is still not evidence, so the sweep must at least have swept.
  graded rv-qdrain rv_qdrain_verdict "$OUT/rv-qdrain.$ARCH.csv"
}

probe_rv_pairs() {
  build_once "$RB" riscvbench || { verdict rv-pairs FAILED "build failed"; return; }
  local b=32; [ "$DO_SIM" = 1 ] && b=1
  # 0x339 = loop_overhead|mul_indep|mul_dep|div|store_spread|store_coalesce
  ( cd "$RB" && ./build/riscvbench --phase r --probes 0x339 --variants t1 \
      --blocks "$b" --out "$OUT/rv-pairs.$ARCH.csv" ) >"$OUT/rv-pairs.out" 2>&1
  cat "$OUT/rv-pairs.out" >> "$LOG"
  [ -s "$OUT/rv-pairs.$ARCH.csv" ] || { verdict rv-pairs FAILED "no CSV; send rv-pairs.out"; return; }
  # This run carries phase R, so riscvbench's own live-instrument check AND its
  # per-phase validity gates apply to it exactly as they do to the `rv` probe.
  # The old check only looked for the live-instrument string, so a phase that
  # failed its gate still read MEANINGFUL.
  graded rv-pairs rv_pairs_verdict "$OUT/rv-pairs.out" "$ARCH"
}

# Every (nx, ny) a plan addresses -- master and subordinate -- must exist in the
# card's own --dump-grid. Prints the offenders; empty output means the plan fits
# this part. Pure awk: the card box has no numpy and may have no tt_sim.
plan_tiles_missing_from_grid() {
  local plan="$1" grid="$2"
  awk -F, '
    FNR==NR {                                   # the grid dump
      if ($0 ~ /^#/) next
      if (FNR==1 || $1 == "log_x") { for (i=1;i<=NF;i++) g[$i]=i; next }
      have[$(g["noc_x"]) "-" $(g["noc_y"])] = 1
      next
    }
    /^#/ { next }                               # the plan
    !hdr { for (i=1;i<=NF;i++) p[$i]=i; hdr=1; next }
    {
      m = $(p["mst_nx"]) "-" $(p["mst_ny"]); s = $(p["sub_nx"]) "-" $(p["sub_ny"])
      if (!(m in have)) miss[m]=1
      if (!(s in have)) miss[s]=1
    }
    END { for (k in miss) printf "%s ", k }
  ' "$grid" "$plan"
}

probe_noc() {
  # Against tt-sim this needs an EXPLICIT opt-in, and the reason is COST, not
  # impossibility. It runs there and it reads CONGESTION MEASURED on both arches
  # (2026-08-12): tt-sim has modelled link congestion since 2026-08-05, and the
  # workers the plan addresses materialise on demand now that this script no
  # longer pins the pool. But it is minutes where the rest of the block is
  # seconds, so a plain `--sim` smoke run leaves it out. Set TT_SIM_COST_MODEL=1
  # when you name it -- with the model off the link term is never spent and the
  # sweep really is forced flat.
  if [ "$DO_SIM" = 1 ] && [ "$SIM_NOC_OPT_IN" != 1 ]; then
    skip noc "against tt-sim this is minutes where the rest of the block is seconds, so it is opt-in. It DOES run there and reads CONGESTION MEASURED -- name it explicitly, with TT_SIM_COST_MODEL=1"
    return
  fi
  build_once "$NB" nocbench || { verdict noc FAILED "build failed"; return; }
  [ -s "$GRID" ] || ( cd "$NB" && ./build/nocbench --dump-grid --out "$GRID" ) >>"$LOG" 2>&1
  local plan="$OUT/noc-plan.csv" csv="$OUT/noc.$ARCH.csv" extra=""
  # The roadmap asks for congestion points BETWEEN the 64 B and 16 KiB regimes.
  # --shared-sizes is exactly that knob; its default is the two endpoints only.
  local shared="64,512,2048,8192,16384"
  [ "$DO_SIM" = 1 ] && extra="--max-points 2 --num-tx 8"
  # A plan pre-built at home (shipped as nocbench/noc-plan-<arch>.csv, or given
  # with --plan) removes the one card-side step that needs tt_sim. It is only
  # safe if it was built for THIS card: a plan naming a tile the part does not
  # have measures nothing, and on a harvested card that is easy to do by
  # accident. So verify every addressed tile against the live dump first.
  if [ ! -s "$plan" ] && [ -n "$PLAN_IN" ]; then
    [ -s "$PLAN_IN" ] || { verdict noc FAILED "--plan $PLAN_IN not found"; return; }
    cp "$PLAN_IN" "$plan"
  fi
  # The shipped plan lives beside the bench, not in its src/ tree. Look in both,
  # because "it is right there and the session did not see it" is exactly the
  # failure that wastes a card session.
  # Not against tt-sim, though. The shipped plan was built for a HARVESTED
  # card, and the simulator has the whole grid: every tile it names exists, so
  # the check below passes, but the physical coordinates it was planned in are
  # a different part's. nocbench then reports "NIU reports physical coord X,
  # plan says Y" on most flows and the run measures the wrong geometry. The
  # simulator has no reason to avoid the planner -- it is the analysis box.
  local shipped
  if [ "$DO_SIM" != 1 ]; then
    for shipped in "$PB/nocbench/noc-plan-$ARCH.csv" "$NB/noc-plan-$ARCH.csv"; do
      if [ ! -s "$plan" ] && [ -s "$shipped" ]; then
        cp "$shipped" "$plan"
        say "   using the pre-built plan $shipped"
      fi
    done
  fi
  if [ -s "$plan" ] && [ -s "$GRID" ]; then
    local bad
    bad="$(plan_tiles_missing_from_grid "$plan" "$GRID")" || bad=""
    if [ -n "$bad" ]; then
      verdict noc FAILED "the plan addresses tiles this card does not have: $bad. It was built for a different grid (harvesting differs?). Delete $plan and re-plan from this card's own $GRID"
      return
    fi
  fi
  if [ ! -s "$plan" ]; then
    if [ "$HAVE_TT_SIM" != 1 ]; then
      skip noc "no plan for this part and the planner needs tt_sim/ (not on this box). $GRID has been kept: plan at home with \`python3 -m tt_sim.perf.noc_congestion_plan --grid <that dump> --out noc-plan-$ARCH.csv --shared-sizes $shared\`, drop it in perfbench/nocbench/, and re-run \`$0 noc noc-epoch\`"
      return
    fi
    # shellcheck disable=SC2086
    PYTHONPATH="$REPO" python3 -m tt_sim.perf.noc_congestion_plan --grid "$GRID" --out "$plan" \
      --shared-sizes "$shared" $extra >>"$LOG" 2>&1 || {
        verdict noc FAILED "the planner refused; it does that rather than emit a confounded experiment. On a HARVESTED part it refuses a grid dump with no phys_x/phys_y column -- read session.log"; return; }
  fi
  ( cd "$NB" && ./build/nocbench --plan "$plan" --out "$csv" -v ) >"$OUT/noc.out" 2>&1
  cat "$OUT/noc.out" >> "$LOG"
  [ -s "$csv" ] || { verdict noc FAILED "no CSV; send noc.out"; return; }
  # Only run the generator if it CAN run. On 2026-08-09 it could not, and its
  # traceback was written into a file called noc.report.txt -- which the verdict
  # then greped for the absence of "INVALID" and passed. A `.report.txt` holding
  # a traceback reads as a result to everything downstream of it.
  if [ "$HAVE_TT_SIM" = 1 ]; then
    PYTHONPATH="$REPO" python3 -m tt_sim.perf.noc_congestion_sweep --measured "$csv" \
      > "$OUT/noc.report.txt" 2>&1
    cat "$OUT/noc.report.txt" >> "$LOG"
  else
    rm -f "$OUT/noc.report.txt"
    say "   report deferred: tt_sim/ is not importable here, so no noc.report.txt is"
    say "   written. Generate it at the analysis box from noc.$ARCH.csv."
  fi
  graded noc noc_verdict "$OUT/noc.report.txt" "$HAVE_TT_SIM"
}

probe_noc_epoch() {
  # Not a different experiment -- the SAME one again. clock_skew_report only
  # accepts a per-core clock offset that reproduces across two or more
  # independent runs, so one run can never confirm the (11,2) epoch. This is
  # the second run, and it is why the bullet is a probe rather than a note.
  if [ "$DO_SIM" = 1 ] && [ "$SIM_NOC_OPT_IN" != 1 ]; then
    skip noc-epoch "see the noc probe: opt-in against tt-sim on cost. It runs there and reads COLLECTED -- a simulator has one clock, so the detector correctly names no per-tile epoch"
    return
  fi
  build_once "$NB" nocbench || { verdict noc-epoch FAILED "build failed"; return; }
  local plan="$OUT/noc-plan.csv" csv="$OUT/noc-epoch.$ARCH.csv"
  [ -s "$plan" ] || { skip noc-epoch "needs the noc probe's plan; run noc first"; return; }
  ( cd "$NB" && ./build/nocbench --plan "$plan" --out "$csv" -v ) >"$OUT/noc-epoch.out" 2>&1
  cat "$OUT/noc-epoch.out" >> "$LOG"
  [ -s "$csv" ] || { verdict noc-epoch FAILED "no CSV; send noc-epoch.out"; return; }
  if [ "$HAVE_TT_SIM" = 1 ]; then
    PYTHONPATH="$REPO" python3 -m tt_sim.perf.noc_congestion_sweep \
      --measured "$OUT/noc.$ARCH.csv" "$csv" > "$OUT/noc-epoch.report.txt" 2>&1
    cat "$OUT/noc-epoch.report.txt" >> "$LOG"
  else
    rm -f "$OUT/noc-epoch.report.txt"
    say "   report deferred: tt_sim/ is not importable here, so no noc-epoch.report.txt"
    say "   is written. Pool noc.$ARCH.csv and noc-epoch.$ARCH.csv at the analysis box."
  fi
  graded noc-epoch noc_epoch_verdict "$OUT/noc-epoch.report.txt" "$HAVE_TT_SIM"
}

# --------------------------------------------------------------------- driver
say "== card session: arch=$ARCH out=$OUT$SIM_NOTE"
say "   started $(date -Is)"
if [ "$HAVE_TT_SIM" != 1 ]; then
  say "   tt_sim/ is not importable here: the five benches still run and every"
  say "   CSV is still collected. The congestion probes use the pre-built plan"
  say "   nocbench/noc-plan-<arch>.csv (checked against this card's live grid"
  say "   before use), so they run too; only the report GENERATORS are deferred"
  say "   to the analysis box. This is the expected shape of a card box that"
  say "   was rsync'd perfbench/ alone."
fi
say ""

for name in "${WANT[@]}"; do
  if ! applies "$name"; then
    # Explicit skip-and-report, never a silent omission: several probes are
    # per-part on purpose and the roadmap's whole claim is a per-part difference.
    skip "$name" "needs a $(probe_field "$name" 2) part; this is $ARCH"
    continue
  fi
  if done_already "$name"; then skip "$name" "--resume: output already in $OUT"; continue; fi
  say "== $name  ($(probe_field "$name" 3) min)  $(probe_field "$name" 4)"
  if [ "$DO_DRY" = 1 ]; then echo "   + probe_${name//-/_}"; continue; fi
  "probe_${name//-/_}"
  say ""
done

[ "$DO_DRY" = 1 ] && exit 0

# ------------------------------------------------------------------- handover
# Counted out here rather than inside the pipeline below: a brace group in a
# pipeline runs in a subshell, so anything it computes -- including the exit
# status -- would be thrown away.
bad=0
for v in ${VERDICTS[@]+"${VERDICTS[@]}"}; do
  rest="${v#*|}"
  case "${rest%%|*}" in
    FAILED|SUSPECT|UNCLEAR) bad=$((bad + 1)) ;;
    # On a card a DEGENERATE probe is a broken run and must not pass silently.
    # Under --sim it is the correct, documented answer, so it is not counted.
    DEGENERATE) [ "$DO_SIM" = 1 ] || bad=$((bad + 1)) ;;
  esac
done

{
  echo
  echo "================ SESSION SUMMARY ($ARCH)$SIM_NOTE ================"
  for v in ${VERDICTS[@]+"${VERDICTS[@]}"}; do
    n="${v%%|*}"; rest="${v#*|}"; s="${rest%%|*}"; t="${rest#*|}"
    printf '%-14s %-12s %s\n' "$n" "$s" "$t"
  done
  echo
  echo "STATUSES: MEANINGFUL  the probe's own control moved; a real reading."
  echo "          COLLECTED   CSV written, but nothing in-session can grade it;"
  echo "                      the analysis box decides whether it says anything."
  echo "          DEFERRED    CSV written; the ANALYSIS needs tt_sim/, which is"
  echo "                      not on this box. Not a verdict, and not a failure."
  echo "          DEGENERATE  the control did NOT move. On a card that is a"
  echo "                      broken run, not a result. Only --sim expects it."
  echo "          SKIPPED     did not apply, or a dependency was missing."
  echo "          FAILED / SUSPECT / UNCLEAR  needs a look before you leave."
  echo
  echo "MEANINGFUL is only ever awarded for a control that MOVED. A value being"
  echo "present, being in the right format, not being a known sentinel, or a"
  echo "failure string being absent from a file are none of them evidence -- the"
  echo "2026-08-09 Blackhole session awarded two MEANINGFULs on exactly those"
  echo "grounds. perfbench/card_session_verdicts.sh holds the checks and"
  echo "card_session_verdicts_test.sh runs them against that session's own CSVs."
  echo
  echo "BEFORE YOU SEND ANYTHING BACK:"
  echo "  1. No probe should read DEGENERATE, FAILED, SUSPECT or UNCLEAR."
  echo "  2. SKIPPED probes that name the other part are correct and expected."
  echo "     If you have the other card too, run the session again there."
  echo "  3. nocread: check the 64 B burst rows settle near 25 cycles/tx"
  echo "     (Wormhole) or 35 (Blackhole) -- that is the published-dataset"
  echo "     control. If it does not, nothing else in that CSV is trustworthy."
  echo "  4. tensix / rv: a run where every probe reads exactly 1.000 measured"
  echo "     nothing. riscvbench prints this and still exits 0."
  if [ "$ARCH" = blackhole ]; then
    echo "  5. cmdbuf is the reading to check twice. 0xFFFFFFFF means the read"
    echo "     FAILED, not that the value is large. An identical value at rest"
    echo "     and in every in-loop sample means the register said NOTHING --"
    echo "     it is an occupancy, so zero at rest is correct and is not a"
    echo "     depth. Retake, or report it as unknown; do not quote a flat"
    echo "     reading as the depth."
  fi
  echo "  6. Send the WHOLE directory back, from the analysis box:"
  echo "       rsync -av <card-box>:$OUT/ ./card-session-$ARCH/"
  echo "     ($(ls "$OUT" 2>/dev/null | wc -l) files: the CSVs, the .out logs, session.log, the reports)"
  if [ "$HAVE_TT_SIM" != 1 ]; then
    echo "     nocbench-grid.csv matters even if the noc probes were skipped --"
    echo "     it is what lets the planner be run at home."
  fi
  echo
  echo "STILL OPEN AFTER A CLEAN RUN -- awaiting the planned Wormhole follow-on."
  echo "These are scheduled, not stuck: every probe below already runs on"
  echo "Wormhole (\`--arch wormhole\`), so the follow-on is a hardware booking."
  for d in "${DEFERRED_TO_WORMHOLE[@]}"; do
    printf '  - %s\n' "$d" | fold -s -w 74 | sed '2,$s/^/    /'
  done
  echo
  echo "  Not attempted, by design -- ATCAS/ATINCGETPTR, the TTBENCH_UNROLL"
  echo "  sweep, the divide magnitude sweep, and DIR_BIDIR. perfbench/README.md"
  echo "  says what each would need. Do not improvise them at the card."
  echo "  finished $(date -Is)"
  [ "$bad" -gt 0 ] && echo "  ($bad probe(s) need a look)"
  true
} | tee -a "$LOG"

# Non-zero when a probe needs attention. DEGENERATE counts on a card, where it
# means the instrument measured nothing, and does not under --sim, where it is
# the documented and correct answer.
[ "$bad" -eq 0 ]
