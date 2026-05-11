#!/usr/bin/env bash
# Launched by UMD's tt_SimulationDevice. NNG_SOCKET_ADDR is already in env.
#
# Opt-in env knobs (UMD inherits the parent env, so set these before running
# the tt-metal program):
#   TT_SIM_RECORD=<path>      record every wire message to this file
#   TT_SIM_LOG_PROTOCOL=1     print every wire message to stderr
#   TT_SIM_CYCLES_PER_POLL=N  cycles to run after each message (default 100)
#   TT_SIM_MOCK_TENSIX=1      skip building Wormhole; every core is NullCore
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root must be on PYTHONPATH so `driver.wormhole.server` resolves.
export PYTHONPATH="$HERE/../..:${PYTHONPATH:-}"

extra=()
[ -n "${TT_SIM_RECORD:-}" ] && extra+=(--record "$TT_SIM_RECORD")
[ -n "${TT_SIM_LOG_PROTOCOL:-}" ] && extra+=(--log-protocol)
[ -n "${TT_SIM_CYCLES_PER_POLL:-}" ] && extra+=(--cycles-per-poll "$TT_SIM_CYCLES_PER_POLL")
[ -n "${TT_SIM_MOCK_TENSIX:-}" ] && extra+=(--mock-tensix)

echo "[run.sh] starting tt-sim server on $NNG_SOCKET_ADDR" >&2
exec python3 -u -m driver.wormhole.server "${extra[@]}" "$@"
