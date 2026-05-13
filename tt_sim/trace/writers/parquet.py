"""Parquet writer for performance counters.

Subscribes to :class:`CounterSnapshot` events and writes them as a
partitioned Parquet dataset (``chip=<N>/kernel_id=<N>/<file>.parquet``)
suitable for direct ingestion into DuckDB / pandas / Polars.

Long format — one row per (cycle, unit, counter_name) sample. Buffers
in memory until ``buffer_size`` rows accumulate, then flushes to disk
in one ``pyarrow.parquet.write_to_dataset`` call. The aggregator's
own flush also forces a parquet write at lifecycle boundaries via
:meth:`close`.

Canned DuckDB queries against the resulting dataset live in
``tt_sim/trace/queries/README.md``.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import CounterSnapshot, EventCategory


class ParquetCounterWriter:
    def __init__(
        self,
        directory: Path | str,
        buffer_size: int = 1000,
        bus: EventBus | None = None,
    ):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []
        self._buffer_size = max(1, buffer_size)
        self._bus = bus if bus is not None else get_bus()
        self._bus.subscribe(EventCategory.COUNTER, self._on_event)

    def _on_event(self, event: CounterSnapshot):
        chip, core_y, core_x, unit = event.unit_id
        self._buffer.append(
            {
                "cycle": event.cycle,
                "chip": int(chip),
                "kernel_id": int(event.kernel_id),
                "core_y": int(core_y),
                "core_x": int(core_x),
                "unit": str(unit),
                "counter_name": event.counter_name,
                "value": int(event.value),
            }
        )
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer)
        pq.write_to_dataset(
            table,
            root_path=str(self._dir),
            partition_cols=["chip", "kernel_id"],
        )
        self._buffer = []

    def close(self):
        self._flush()
