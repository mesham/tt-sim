#!/usr/bin/env bash
# mechbench, simulator side: run each arm against tt-sim with the Tensix
# performance counters armed, and collect the device-profiler log the card
# protocol produces on the other side.
#
# The two sides are deliberately the SAME artefact -- a tt-metal
# profile_log_device.csv full of PerfCounter markers -- so
# tt_sim.perf.stall_attribution parses both with one reader and there is no
# translation step in which a units mistake could hide.
#
#   TT_METAL_HOME=/path/to/tt-metal ./perfbench/mechbench/run_sim.sh \
#       --arch blackhole --out /tmp/mechbench-sim
#
# Options:
#   --arch blackhole|wormhole   which simulator to run against (default blackhole)
#   --tiles N                   tiles per arm (default 8; the simulator runs a
#                               few tens of thousands of cycles per second, so
#                               this is not the card's tile count)
#   --arms "elw mm"             which arms to run
#   --out DIR                   output directory (default ./mechbench-sim)
#   --timeout SECONDS           per-arm timeout (default 1800)
#   --no-cost-model             run with TT_SIM_COST_MODEL unset, which makes
#                               the Src-ownership columns ABSENT rather than
#                               zero. Exists to demonstrate that, not to be used
#                               for a comparison.
#
# Each arm lands in <out>/<arm>/.logs/profile_log_device.csv plus <out>/<arm>.out.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SRC="$HERE/src"

ARCH=blackhole
TILES=8
ARMS="elw mm"
OUT="$PWD/mechbench-sim"
TIMEOUT=1800
COST_MODEL=1

while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --tiles) TILES="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --no-cost-model) COST_MODEL=0; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \?//'; exit 0 ;;
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
# counters come back from a core that ran nothing.
case "$ARCH" in
  blackhole) export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-1-2}" ;;
  wormhole)  export TT_SIM_TENSIX_COORDS="${TT_SIM_TENSIX_COORDS:-1-1}" ;;
  *) echo "unknown arch: $ARCH" >&2; exit 2 ;;
esac

# Bit 5 is the INSTRN_THREAD bank -- the only one tt-sim sources, and the only
# one the partition needs. No L1 bank bits, so no mux pass structure here.
export TT_METAL_PROFILE_PERF_COUNTERS=32

# The cost model is ON by default and that is not a preference. With it unset
# no Tensix backend arms an occupancy, so the Matrix unit releases a Src bank
# the cycle it takes it and the SrcA/SrcB CLEAR mechanism is structurally
# ABSENT -- a column of zeros that reads exactly like "this never happens".
# Comparing that against a card is comparing against a measurement tt-sim did
# not make.
if [ "$COST_MODEL" -eq 1 ]; then
  export TT_SIM_COST_MODEL=1
else
  unset TT_SIM_COST_MODEL
  echo "WARNING: cost model off -- Src-ownership columns will be absent, not zero" >&2
fi

if [ ! -x "$SRC/build/mechbench" ]; then
  echo "== building mechbench"
  ( cd "$SRC" \
    && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" ) || exit 1
fi

mkdir -p "$OUT"
. "$REPO/driver/sim_procs.sh"
sim_procs_init mechbench_sim
trap 'sim_kill_own_servers' EXIT INT TERM

status=0
for arm in $ARMS; do
  sim_kill_own_servers; sleep 0.5
  armdir="$OUT/$arm"
  rm -rf "$armdir"; mkdir -p "$armdir"
  echo "== $arm ($TILES tiles) on $ARCH"
  ( cd "$SRC" && TT_METAL_PROFILER_DIR="$armdir" \
      timeout "$TIMEOUT" ./build/mechbench "$arm" "$TILES" ) >"$OUT/$arm.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q "Completed successfully on the device" "$OUT/$arm.out"; then
    echo "   FAILED (rc=$rc); see $OUT/$arm.out"
    status=1
    continue
  fi
  log="$armdir/.logs/profile_log_device.csv"
  if [ ! -f "$log" ]; then
    echo "   no profiler log at $log -- was TT_METAL_PROFILE_PERF_COUNTERS honoured?"
    status=1
    continue
  fi
  n=$(grep -c ',9090,' "$log" || true)
  if [ "$n" -eq 0 ]; then
    # This used to be the expected outcome with the cost model on: the host
    # read the profiler control vector one wire message after BRISC reported
    # RUN_MSG_DONE, before finish_profiler() had published the run, and got a
    # header-only CSV. The bridge now waits for that publish
    # (Device.settle_profiler_flush), so an empty log here is a NEW failure.
    echo "   EMPTY profiler log (0 samples, and no zone markers either)."
    echo "   The bridge waits for the profiler's publish since 2026-08-13, so"
    echo "   this is not the old readback race. Check the server's shutdown"
    echo "   line for a 'profiler flush ... TIMED OUT' warning, and see"
    echo "   tt_sim/bridge/profiler_readback_test.py."
    status=1
    continue
  fi
  echo "   ok: $n counter samples -> $log"
done
sim_kill_own_servers

echo "----"
if [ $status -eq 0 ]; then
  echo "Simulator side complete. Decomposition:"
  for arm in $ARMS; do
    echo "  python3 -m tt_sim.perf.stall_attribution --decompose-only \\"
    echo "      --sim $OUT/$arm/.logs/profile_log_device.csv"
  done
fi
exit $status
