"""Environment-driven auto-setup for tracing.

Called from `Wormhole.__init__` so any driver script that constructs a
device picks up `TT_SIM_TRACE=path/to/file.jsonl` automatically. No
per-example wiring required.
"""

import atexit
import os

from tt_sim.trace.bus import get_bus
from tt_sim.trace.ids import get_registry
from tt_sim.trace.writers.jsonl import JSONLLogger

_LOGGER: JSONLLogger | None = None


def enable_from_env() -> JSONLLogger | None:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    path = os.environ.get("TT_SIM_TRACE")
    if not path:
        return None
    bus = get_bus()
    bus.enabled = True
    _LOGGER = JSONLLogger(path)

    def _on_exit():
        if _LOGGER is not None:
            _LOGGER.close()
            get_registry().dump(path + ".ids.json")

    atexit.register(_on_exit)
    return _LOGGER
