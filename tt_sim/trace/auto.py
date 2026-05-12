"""Environment-driven auto-setup for tracing.

Called from `Wormhole.__init__` so any driver script that constructs a
device picks up trace env vars automatically — no per-example wiring.

Supported env vars (all optional, all default-off):

- ``TT_SIM_TRACE`` — JSONL writer; one event per line. Suitable for
  ``jq`` / ``duckdb`` / pandas analysis.
- ``TT_SIM_TRACE_PERFETTO`` — Chrome Trace Event Format writer; output
  loads directly into ``ui.perfetto.dev``. Use a ``.json.gz`` extension
  for gzip compression (Perfetto loads it natively).

Both writers can be enabled simultaneously; they subscribe to disjoint
event handling and write independent files.
"""

import atexit
import os

from tt_sim.trace.bus import get_bus
from tt_sim.trace.ids import get_registry
from tt_sim.trace.writers.jsonl import JSONLLogger
from tt_sim.trace.writers.perfetto import PerfettoWriter

_JSONL: JSONLLogger | None = None
_PERFETTO: PerfettoWriter | None = None


def enable_from_env() -> None:
    """Wire up any tracing writers configured via environment variables.

    Idempotent — calling more than once is safe; only the first call
    per process actually opens files.
    """
    global _JSONL, _PERFETTO
    jsonl_path = os.environ.get("TT_SIM_TRACE")
    perfetto_path = os.environ.get("TT_SIM_TRACE_PERFETTO")

    if not jsonl_path and not perfetto_path:
        return

    bus = get_bus()
    bus.enabled = True

    if jsonl_path and _JSONL is None:
        _JSONL = JSONLLogger(jsonl_path)

    if perfetto_path and _PERFETTO is None:
        _PERFETTO = PerfettoWriter(perfetto_path)

    if not getattr(enable_from_env, "_atexit_registered", False):

        def _on_exit():
            if _JSONL is not None:
                _JSONL.close()
                if jsonl_path:
                    get_registry().dump(jsonl_path + ".ids.json")
            if _PERFETTO is not None:
                _PERFETTO.close()

        atexit.register(_on_exit)
        enable_from_env._atexit_registered = True  # type: ignore[attr-defined]
