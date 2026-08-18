#!/usr/bin/env bash
# nocevbench, simulator side: run each size arm against tt-sim with tt-metal's
# NoC event profiler on, and collect the same artefact the card protocol
# produces on the other side.
#
# The two sides are deliberately the SAME artefact -- a tt-metal
# noc_trace_dev*_ID*.json -- so tt_sim.perf.noc_events parses both with one
# reader and there is no translation step in which a units mistake could hide.
#
#   TT_METAL_HOME=/path/to/tt-metal ./perfbench/nocevbench/run_sim.sh \
#       --arch blackhole --out /tmp/nocevbench-sim
#
# Options:
#   --arch blackhole|wormhole   which simulator to run against (default blackhole)
#   --arm A|B|C                 which experimental arm (default A, the control):
#                                 A  reader NCRISC/NOC_1 and writer BRISC/NOC_0,
#                                    both against DRAM -- the 2026-08-17 config
#                                 B  the two NoCs SWAPPED and nothing else
#                                 C  a peer core's L1 instead of DRAM
#   --peer X,Y                  arm C's peer, in LOGICAL worker coords (default 1,1)
#   --bytes "256 4096"          the size arms (default "256 4096")
#   --chunks N                  transfers per arm (default 8)
#   --out DIR                   output directory (default ./nocevbench-sim)
#   --timeout SECONDS           per-arm timeout (default 1800)
#   --no-internal               skip the TT_SIM_TRACE_NOC Parquet sidecar
#   --no-cost-model             run with TT_SIM_COST_MODEL unset. Exists to
#                               demonstrate what goes wrong, not to be used for
#                               a comparison.
#
# Each arm lands in <out>/<bytes>/.logs/noc_trace_dev0_ID0.json plus
# <out>/<bytes>.out, and (unless --no-internal) <out>/<bytes>-internal/ holding
# tt-sim's own per-transaction modelled cycles.
#
# EVERY RUN IS CHECKED AGAINST ITS ARM, from the trace rather than from the fact
# that the flag was passed -- see check_arm.py. A run that silently kept a
# different arm's NoC pairing is well-formed, passes every analysis gate, and
# supports the opposite conclusion, so it is the one failure this harness must
# never let through.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../build_provenance.sh
. "$HERE/../build_provenance.sh"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"

ARCH=blackhole
ARM=A
PEER="1,1"
BYTES="256 4096"
CHUNKS=8
OUT="$PWD/nocevbench-sim"
TIMEOUT=1800
COST_MODEL=1
INTERNAL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --arm) ARM="$(echo "$2" | tr '[:lower:]' '[:upper:]')"; shift 2 ;;
    --peer) PEER="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --no-internal) INTERNAL=0; shift ;;
    --no-cost-model) COST_MODEL=0; shift ;;
    -h|--help) sed -n '2,38p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$ARM" in
  A|B|C) ;;
  *) echo "unknown arm: $ARM (expected A, B or C)" >&2; exit 2 ;;
esac

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_SIMULATOR="$REPO/driver/$ARCH"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Worker (0,0) is physical (1,1) on Wormhole and (1,2) on Blackhole. Getting
# this wrong does not fail loudly -- it materialises a second worker and the
# trace comes back from a core that ran nothing.
# Translation is on when the host was handed a cluster descriptor that declares
# it -- the same TT_METAL_MOCK_CLUSTER_DESC_PATH the tt-metal host reads, which
# the server inherits. It changes which coordinate a kernel emits, so it changes
# which tile has to be pre-built.
TRANSLATED=0
[ -n "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-}" ] && TRANSLATED=1

case "$ARCH" in
  blackhole) SELF_COORD="1-2" ;;
  wormhole)  SELF_COORD="1-1" ;;
  *) echo "unknown arch: $ARCH" >&2; exit 2 ;;
esac

# Arm C addresses a second core's L1, so that core has to exist in the
# simulator. LazyTensixPool would materialise it on demand, but naming it up
# front is what lets the mismatch below be an error rather than a surprise: the
# device reports the peer's real NoC coordinate in its `nocevbench-config` line,
# and it is checked against this guess after the run.
PEER_COORD=""
if [ "$ARM" = "C" ]; then
  PEER_COORD=$(ARCH="$ARCH" PEER="$PEER" TRANSLATED="$TRANSLATED" python3 - <<'PY'
import os, sys
arch, peer = os.environ["ARCH"], os.environ["PEER"]
lx, ly = (int(v) for v in peer.split(","))
if arch == "blackhole":
    xs = list(range(1, 8)) + list(range(10, 17))
    ys = list(range(2, 12))
else:
    xs = [1, 2, 3, 4, 6, 7, 8, 9]
    ys = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
if lx >= len(xs) or ly >= len(ys):
    sys.exit(f"peer {peer} is outside the {arch} worker grid")
print(f"{xs[lx]}-{ys[ly]}")
PY
  ) || exit 2
  export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-$SELF_COORD,$PEER_COORD}"
else
  export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-$SELF_COORD}"
fi

# This is the whole instrument. It force-enables the device profiler
# (rtoptions.cpp:854) and injects -DPROFILE_NOC_EVENTS into every kernel compile
# (jit_build/build.cpp:188), so it must be set at BUILD time, not just run time
# -- which it is, because the JIT build happens inside this process.
export TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1

if [ "$COST_MODEL" -eq 1 ]; then
  export TT_SIM_COST_MODEL=1
else
  unset TT_SIM_COST_MODEL
  echo "WARNING: cost model off -- every NoC flight collapses to one cycle and" >&2
  echo "         the comparison measures nothing. Do not use this for a result." >&2
fi

# This tree is shared with nocevbench/run_card.sh; poisoning it here would be
# discovered there, on a card, after a board reset.
if [ ! -x "$SRC/build/nocevbench" ]; then
  echo "== building nocevbench"
  bp_require_build "$SRC"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || exit 1
  bp_record_build "$SRC"
else
  bp_require_build "$SRC" skip-build
fi

echo "== arm $ARM on $ARCH, workers $TT_SIM_TENSIX_COORDS"
ARM_ARGS="--arm $ARM"
[ "$ARM" = "C" ] && ARM_ARGS="$ARM_ARGS --peer $PEER"

mkdir -p "$OUT"
. "$REPO/driver/sim_procs.sh"
sim_procs_init nocevbench_sim
trap 'sim_kill_own_servers' EXIT INT TERM

status=0
for bytes in $BYTES; do
  sim_kill_own_servers; sleep 0.5
  armdir="$OUT/$bytes"
  rm -rf "$armdir"; mkdir -p "$armdir"
  echo "== ${bytes} B x ${CHUNKS} on $ARCH, arm $ARM"
  if [ "$INTERNAL" -eq 1 ]; then
    export TT_SIM_TRACE_NOC="$OUT/$bytes-internal"
    rm -rf "$TT_SIM_TRACE_NOC"
  else
    unset TT_SIM_TRACE_NOC
  fi
  # shellcheck disable=SC2086  -- ARM_ARGS is a deliberate word list
  ( cd "$SRC" && TT_METAL_PROFILER_DIR="$armdir" \
      timeout "$TIMEOUT" ./build/nocevbench "$bytes" "$CHUNKS" $ARM_ARGS ) >"$OUT/$bytes.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q "Completed successfully on the device" "$OUT/$bytes.out"; then
    echo "   FAILED (rc=$rc); see $OUT/$bytes.out"
    status=1
    continue
  fi
  trace=$(ls "$armdir"/.logs/noc_trace_dev*_ID*.json 2>/dev/null | head -1)
  if [ -z "$trace" ]; then
    echo "   NO NoC TRACE in $armdir/.logs/."
    echo "   The bridge waits for the profiler's publish since 2026-08-13"
    echo "   (Device.settle_profiler_flush), so this is not the old readback"
    echo "   race. Check the server's shutdown line for a 'profiler flush ...'"
    echo "   warning, and confirm tt-metal was built with Tracy enabled --"
    echo "   every dump path is inside #if defined(TRACY_ENABLE)."
    status=1
    continue
  fi
  n=$(python3 -c "import json,sys;print(sum(1 for r in json.load(open(sys.argv[1])) if 'type' in r))" "$trace")
  echo "   ok: $n NoC events -> $trace"
  # The arm check, from the trace. An arm that did not take is the one failure
  # that would otherwise look like a clean result.
  if ! python3 "$HERE/check_arm.py" --arm "$ARM" --trace "$trace" \
         --stdout "$OUT/$bytes.out" --latencies; then
    status=1
    continue
  fi
  if [ -n "$PEER_COORD" ]; then
    reported=$(sed -n 's/.*peer_noc=\([0-9]*\),\([0-9]*\).*/\1-\2/p' "$OUT/$bytes.out" | head -1)
    # TT_SIM_TENSIX_COORDS names TILES, which are keyed by PHYSICAL coord --
    # the SoC descriptor's own space, and the server rejects anything else.
    # `peer_noc` is the coordinate a KERNEL addresses, which translation moves:
    # on Wormhole a logical (lx,ly) becomes (lx+18, ly+18), confirmed by the
    # card reporting self_noc=18,18 / peer_noc=19,19 on 2026-08-17. So the two
    # are in different spaces whenever translation is on, and comparing them
    # raw is what this expectation exists to avoid.
    EXPECT_PEER="$PEER_COORD"
    if [ "$TRANSLATED" = 1 ] && [ "$ARCH" = wormhole ]; then
      EXPECT_PEER=$(echo "$PEER" | awk -F, '{printf "%d-%d", $1+18, $2+18}')
    fi
    if [ "$reported" != "$EXPECT_PEER" ]; then
      echo "   FAIL the device put the peer at $reported, but expected $EXPECT_PEER"
      echo "        pre-built $PEER_COORD. The logical->physical map in this script"
      echo "        is wrong for $ARCH, so the arm ran against a different core."
      status=1
    fi
  fi
done
sim_kill_own_servers

echo "----"
if [ $status -eq 0 ]; then
  echo "Simulator side complete (arm $ARM). Decomposition:"
  for bytes in $BYTES; do
    echo "  python3 -m tt_sim.perf.noc_events --decompose-only --expect-arm $ARM \\"
    echo "      --sim $OUT/$bytes/.logs/noc_trace_dev0_ID0.json \\"
    echo "      --sim-internal $OUT/$bytes-internal"
  done
fi
exit $status
