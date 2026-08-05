#!/usr/bin/env bash
# Differential Tensix-op test. Runs the *same* tt-metal program through tt-sim
# (ours) and through ttsim (the vendor reference sim), and diffs the hex the
# program dumps (`OPDIFF_RESULT:...`). A match means tt-sim reproduces ttsim's
# behaviour for whatever op the compute kernel exercises. Because both sims run
# the same compiled kernel, the check is immune to tt-metal C++->asm churn.
#
# Run in a NORMAL shell (not a sandbox that kills the spawned sim servers):
#   TT_METAL_HOME=/path/to/tt-metal \
#   TTSIM_ORACLE=/path/to/dir/with/libttsim_<bh|wh>.so + a SoC descriptor \
#   ./optests/diff.sh [op-program-name]      # default: optest
#
# The architecture defaults to blackhole; set TT_SIM_ARCH=wormhole to diff the
# same program against ttsim-Wormhole instead. That picks the oracle library
# name (libttsim_wh.so), our driver (driver/wormhole) and the default worker
# coordinate (logical (0,0) is physical 1-1 on WH, 1-2 on BH). TTSIM_ORACLE
# then has to point at the matching oracle dir (ttsim's oracle-wh/).
#
# The descriptor may be named `soc_descriptor.yaml` or, as ttsim's own oracle
# dirs ship it, `<arch>_arch.yaml` (`blackhole_140_arch.yaml` /
# `wormhole_b0_80_arch.yaml`) -- see _stage_oracle below.
#
# Add op programs under optests/<name>/src (a tt-metal host that dumps its
# output tile as `OPDIFF_RESULT:<hex>`); the compute kernel picks the op.
#
# Cleanup only touches the sim servers this run started (they carry its
# TT_SIM_RUN_TAG) plus orphans left by an earlier diff.sh whose owner is gone,
# so a concurrent run in another terminal is never disturbed. Set
# TT_SIM_KILL_ALL_SERVERS=1 to instead kill every tt-sim server on the machine
# at startup — the way to clear up after manual runs, which carry no tag.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Simulator cleanup that only touches the servers *this* run started; set
# TT_SIM_KILL_ALL_SERVERS=1 for the old kill-everything behaviour.
# shellcheck source=../driver/sim_procs.sh
. "$REPO/driver/sim_procs.sh"
NAME="${1:-optest}"
SRC="$REPO/optests/$NAME/src"
ARCH="${TT_SIM_ARCH:-blackhole}"
case "$ARCH" in
  blackhole|bh) ARCH=blackhole; ORACLE_SO=libttsim_bh.so; DEF_COORDS=1-2 ;;
  wormhole|wh)  ARCH=wormhole;  ORACLE_SO=libttsim_wh.so; DEF_COORDS=1-1 ;;
  *) echo "unknown TT_SIM_ARCH=$ARCH (want blackhole|wormhole)" >&2; exit 2 ;;
esac
COORDS="${TT_SIM_TENSIX_COORDS:-$DEF_COORDS}"  # tt-sim needs the worker tile(s); ttsim doesn't
TIMEOUT="${TT_SIM_EXAMPLE_TIMEOUT:-300}"

: "${TT_METAL_HOME:?set TT_METAL_HOME to your built tt-metal checkout}"
: "${TTSIM_ORACLE:?set TTSIM_ORACLE to a dir holding $ORACLE_SO + a SoC descriptor}"
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

# Per-program environment. A program needing a specific tt-metal knob (e.g.
# `optests/where` needs TT_METAL_DISABLE_SFPLOADMACRO=1, since *both* sims
# declare SFPLOADMACRO out of scope) drops an `env` file next to its src/ and
# it is sourced here, so `diff.sh <name>` just works instead of failing with
# "no result from ttsim". The file must only export variables.
if [ -f "$REPO/optests/$NAME/env" ]; then
  # shellcheck disable=SC1090
  . "$REPO/optests/$NAME/env"
  echo "[env] sourced optests/$NAME/env"
fi

# UMD hardcodes the descriptor's *filename*: for a .so simulator it reads
# `<dir-of-so>/soc_descriptor.yaml` (umd device/simulation/simulation_chip.cpp),
# so we cannot just hand it a differently-named file. ttsim's own oracle dirs
# ship the descriptor as `<arch>_arch.yaml`, which would otherwise make every
# run die in the YAML loader with a bare `bad file:` and report "no result
# from ttsim". When the expected name is missing, stage a scratch directory of
# symlinks with the name UMD wants, rather than writing into the ttsim checkout.
# Sets ORACLE_DIR (the directory to load the simulator from) and, when it had to
# build one, STAGED_ORACLE (removed on exit). Assigns rather than echoes: a
# command substitution runs in a subshell, so STAGED_ORACLE set there would be
# lost and the scratch directory would leak.
STAGED_ORACLE=""
ORACLE_DIR=""
_stage_oracle() {
  local abs_so abs_desc desc
  abs_so="$(cd "$TTSIM_ORACLE" 2>/dev/null && pwd)/$ORACLE_SO"
  [ -f "$abs_so" ] || { echo "no $ORACLE_SO in $TTSIM_ORACLE" >&2; return 1; }
  if [ -f "$TTSIM_ORACLE/soc_descriptor.yaml" ]; then
    ORACLE_DIR="$TTSIM_ORACLE"
    return 0
  fi
  for desc in "$TTSIM_ORACLE"/*_arch.yaml; do
    [ -f "$desc" ] || continue
    abs_desc="$(cd "$(dirname "$desc")" && pwd)/$(basename "$desc")"
    STAGED_ORACLE="$(mktemp -d)" || return 1
    ln -s "$abs_so" "$STAGED_ORACLE/$ORACLE_SO"
    ln -s "$abs_desc" "$STAGED_ORACLE/soc_descriptor.yaml"
    ORACLE_DIR="$STAGED_ORACLE"
    echo "[oracle] staged $(basename "$abs_desc") as soc_descriptor.yaml"
    return 0
  done
  echo "no soc_descriptor.yaml or *_arch.yaml in $TTSIM_ORACLE" >&2
  return 1
}

sim_procs_init diff
trap 'sim_kill_own_servers; [ -n "$STAGED_ORACLE" ] && rm -rf "$STAGED_ORACLE"' EXIT INT TERM
_stage_oracle || exit 2

if [ ! -x "$SRC/build/$NAME" ]; then
  echo "BUILD $NAME"
  ( cd "$SRC" && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 ) \
    >"/tmp/optest_${NAME}_build.log" 2>&1 \
    || { echo "FAIL  $NAME (build; see /tmp/optest_${NAME}_build.log)"; exit 1; }
fi

# $1 = TT_METAL_SIMULATOR, $2 = TT_SIM_TENSIX_COORDS ("" to skip), $3 = logfile
_run() {
  # Clear a server this run leaked earlier (e.g. the previous _run timed out).
  sim_kill_own_servers; sleep 0.3
  ( cd "$SRC"
    export TT_METAL_SIMULATOR="$1"
    [ -n "$2" ] && export TT_SIM_TENSIX_COORDS="$2"
    timeout "$TIMEOUT" "./build/$NAME" ) >"$3" 2>&1
}
_result() { grep -o 'OPDIFF_RESULT:[0-9a-f]*' "$1" | head -1 | cut -d: -f2; }

ora_log="/tmp/optest_${NAME}_${ARCH}_oracle.out"
our_log="/tmp/optest_${NAME}_${ARCH}_ours.out"

echo "[oracle: ttsim]  $ORACLE_DIR/$ORACLE_SO"
_run "$ORACLE_DIR/$ORACLE_SO" "" "$ora_log"
echo "[ours:   tt-sim] $REPO/driver/$ARCH  (coords=$COORDS)"
_run "$REPO/driver/$ARCH" "$COORDS" "$our_log"
sim_kill_own_servers

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
