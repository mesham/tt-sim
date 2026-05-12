"""JSONL writer — one JSON object per event, streamed to a file.

Subscribes to every :class:`~tt_sim.trace.events.EventCategory`. Useful
as a low-effort sanity check that the bus is wiring up correctly;
downstream writers (Perfetto, Parquet, etc.) replace it in later phases.
"""

import json
from dataclasses import fields
from pathlib import Path
from typing import IO

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import Event, EventCategory


def _event_to_dict(event: Event) -> dict:
    d: dict = {
        "category": event.CATEGORY.value if event.CATEGORY is not None else None,
        "schema_version": event.SCHEMA_VERSION,
    }
    for f in fields(event):
        v = getattr(event, f.name)
        if isinstance(v, tuple):
            v = list(v)
        d[f.name] = v
    return d


class JSONLLogger:
    def __init__(self, path: Path | str, bus: EventBus | None = None):
        self._path = Path(path)
        self._file: IO = self._path.open("w")
        self._bus = bus if bus is not None else get_bus()
        for cat in EventCategory:
            self._bus.subscribe(cat, self._on_event)

    def _on_event(self, event: Event):
        line = json.dumps(_event_to_dict(event), separators=(",", ":"))
        self._file.write(line)
        self._file.write("\n")

    def close(self):
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
