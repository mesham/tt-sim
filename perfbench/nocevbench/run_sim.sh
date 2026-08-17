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
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"

ARCH=blackhole
BYTES="256 4096"
CHUNKS=8
OUT="$PWD/nocevbench-sim"
TIMEOUT=1800
COST_MODEL=1
INTERNAL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --chunks) CHUNKS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --no-internal) INTERNAL=0; shift ;;
    --no-cost-model) COST_MODEL=0; shift ;;
    -h|--help) sed -n '2,29p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export TT_METAL_SIMULATOR="$REPO/driver/$ARCH"
export TT_METAL_SLOW_DISPATCH_MODE=1
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Worker (0,0) is physical (1,1) on Wormhole and (1,2) on Blackhole. Getting
# this wrong does not fail loudly -- it materialises a second worker and the
# trace comes back from a core that ran nothing.
case "$ARCH" in
  blackhole) export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-1-2}" ;;
  wormhole)  export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-1-1}" ;;
  *) echo "unknown arch: $ARCH" >&2; exit 2 ;;
esac

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

if [ ! -x "$SRC/build/nocevbench" ]; then
  echo "== building nocevbench"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || exit 1
fi

mkdir -p "$OUT"
. "$REPO/driver/sim_procs.sh"
sim_procs_init nocevbench_sim
trap 'sim_kill_own_servers' EXIT INT TERM

status=0
for bytes in $BYTES; do
  sim_kill_own_servers; sleep 0.5
  armdir="$OUT/$bytes"
  rm -rf "$armdir"; mkdir -p "$armdir"
  echo "== ${bytes} B x ${CHUNKS} on $ARCH"
  if [ "$INTERNAL" -eq 1 ]; then
    export TT_SIM_TRACE_NOC="$OUT/$bytes-internal"
    rm -rf "$TT_SIM_TRACE_NOC"
  else
    unset TT_SIM_TRACE_NOC
  fi
  ( cd "$SRC" && TT_METAL_PROFILER_DIR="$armdir" \
      timeout "$TIMEOUT" ./build/nocevbench "$bytes" "$CHUNKS" ) >"$OUT/$bytes.out" 2>&1
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
done
sim_kill_own_servers

echo "----"
if [ $status -eq 0 ]; then
  echo "Simulator side complete. Decomposition:"
  for bytes in $BYTES; do
    echo "  python3 -m tt_sim.perf.noc_events --decompose-only \\"
    echo "      --sim $OUT/$bytes/.logs/noc_trace_dev0_ID0.json \\"
    echo "      --sim-internal $OUT/$bytes-internal"
  done
fi
exit $status
