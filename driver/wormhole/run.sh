#!/usr/bin/env bash
# Launched by UMD's tt_SimulationDevice. NNG_SOCKET_ADDR is already in env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root must be on PYTHONPATH so `driver.wormhole.server` resolves.
export PYTHONPATH="$HERE/../..:${PYTHONPATH:-}"

echo "[run.sh] starting tt-sim server on $NNG_SOCKET_ADDR" >&2
exec python3 -u -m driver.wormhole.server --log-protocol "$@"
