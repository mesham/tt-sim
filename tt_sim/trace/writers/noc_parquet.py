"""Parquet writer for NoC transactions.

Subscribes to :class:`NoCEvent` and writes one row per emission — a
flat columnar form suitable for SQL analysis of data movement:

    cycle, chip, core_y, core_x, unit, phase, txn_type,
    src_y, src_x, dst_y, dst_x, size_bytes, txn_id

Partitioned by ``chip`` to keep multi-chip runs separable; not
partitioned by kernel_id because NoC transactions aren't naturally
kernel-bound (firmware setup also generates traffic).

The Phase 5 roadmap calls for additional fields (``vc``,
``issue_cycle``, ``arrival_cycle``) that depend on a cycle-accurate
queue-modelled NoC — those land alongside §I cycle-accuracy work
without requiring schema changes here.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import EventCategory, NoCEvent


class NoCParquetWriter:
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
        self._bus.subscribe(EventCategory.NOC, self._on_event)

    def _on_event(self, event: NoCEvent):
        chip, core_y, core_x, unit = event.unit_id
        src = event.src if len(event.src) >= 2 else (0, 0)
        dst = event.dst if len(event.dst) >= 2 else (0, 0)
        self._buffer.append(
            {
                "cycle": int(event.cycle),
                "chip": int(chip),
                "core_y": int(core_y),
                "core_x": int(core_x),
                "unit": str(unit),
                "phase": event.phase,
                "txn_type": event.txn_type,
                "src_x": int(src[0]),
                "src_y": int(src[1]),
                "dst_x": int(dst[0]),
                "dst_y": int(dst[1]),
                "size_bytes": int(event.size_bytes),
                "txn_id": int(event.txn_id),
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
            partition_cols=["chip"],
        )
        self._buffer = []

    def close(self):
        self._flush()
