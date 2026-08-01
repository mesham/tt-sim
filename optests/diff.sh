#!/usr/bin/env bash
# Differential Tensix-op test. Runs the *same* tt-metal program through tt-sim
# (ours) and through ttsim (the vendor reference sim), and diffs the hex the
# program dumps (`OPDIFF_RESULT:...`). A match means tt-sim reproduces ttsim's
# behaviour for whatever op the compute kernel exercises. Because both sims run
# the same compiled kernel, the check is immune to tt-metal C++->asm churn.
#
# Run in a NORMAL shell (not a sandbox that kills the spawned sim servers):
#   TT_METAL_HOME=/path/to/tt-metal \
#   TTSIM_ORACLE=/path/to/dir/with/libttsim_bh.so+soc_descriptor.yaml \
#   ./optests/diff.sh [op-program-name]      # default: optest
#
# Add op programs under optests/<name>/src (a tt-metal host that dumps its
# output tile as `OPDIFF_RESULT:<hex>`); the compute kernel picks the op.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:-optest}"
SRC="$REPO/optests/$NAME/src"
COORDS="${TT_SIM_TENSIX_COORDS:-1-2}"          # tt-sim needs the worker tile(s); ttsim doesn't
TIMEOUT="${TT_SIM_EXAMPLE_TIMEOUT:-300}"

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
: "${TTSIM_ORACLE:?set TTSIM_ORACLE to a dir holding libttsim_bh.so + soc_descriptor.yaml}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:${LD_LIBRARY_PATH:-}"
export TT_METAL_SLOW_DISPATCH_MODE=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# UMD spawns the tt-sim server via `python3 -m driver...` off PATH, so the
# venv (with tt_sim's deps) must be on PATH. Default to the sibling venv; set
# TT_SIM_VENV to override, or just have the venv active.
VENV="${TT_SIM_VENV:-$REPO/../venv}"
[ -x "$VENV/bin/python3" ] && export PATH="$VENV/bin:$PATH"

[ -d "$SRC" ] || { echo "no op program at $SRC" >&2; exit 2; }

if [ ! -x "$SRC/build/$NAME" ]; then
  echo "BUILD $NAME"
  ( cd "$SRC" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 ) \
    >"/tmp/optest_${NAME}_build.log" 2>&1 \
    || { echo "FAIL  $NAME (build; see /tmp/optest_${NAME}_build.log)"; exit 1; }
fi

# $1 = TT_METAL_SIMULATOR, $2 = TT_SIM_TENSIX_COORDS ("" to skip), $3 = logfile
_run() {
  pkill -9 -f 'driver.(wormhole|blackhole).server' 2>/dev/null; sleep 0.3
  ( cd "$SRC"
    export TT_METAL_SIMULATOR="$1"
    [ -n "$2" ] && export TT_SIM_TENSIX_COORDS="$2"
    timeout "$TIMEOUT" "./build/$NAME" ) >"$3" 2>&1
}
_result() { grep -o 'OPDIFF_RESULT:[0-9a-f]*' "$1" | head -1 | cut -d: -f2; }

ora_log="/tmp/optest_${NAME}_oracle.out"
our_log="/tmp/optest_${NAME}_ours.out"

echo "[oracle: ttsim]  $TTSIM_ORACLE/libttsim_bh.so"
_run "$TTSIM_ORACLE/libttsim_bh.so" "" "$ora_log"
echo "[ours:   tt-sim] $REPO/driver/blackhole  (coords=$COORDS)"
_run "$REPO/driver/blackhole" "$COORDS" "$our_log"
pkill -9 -f 'driver.(wormhole|blackhole).server' 2>/dev/null

ora="$(_result "$ora_log")"
our="$(_result "$our_log")"

if [ -z "$ora" ]; then echo "FAIL  $NAME: no result from ttsim (see $ora_log)"; exit 1; fi
if [ -z "$our" ]; then echo "FAIL  $NAME: no result from tt-sim (see $our_log)"; exit 1; fi

if [ "$ora" = "$our" ]; then
  echo "PASS  $NAME: tt-sim matches ttsim ($(( ${#ora} / 8 )) elements)"
  exit 0
fi
echo "FAIL  $NAME: tt-sim differs from ttsim"
echo "  oracle: ${ora:0:64}..."
echo "  ours:   ${our:0:64}..."
# first differing element
python3 - "$ora" "$our" <<'PY'
import sys
a, b = sys.argv[1], sys.argv[2]
for i in range(0, min(len(a), len(b)), 8):
    if a[i:i+8] != b[i:i+8]:
        print(f"  first diff at element {i//8}: oracle={a[i:i+8]} ours={b[i:i+8]}")
        break
PY
exit 1
