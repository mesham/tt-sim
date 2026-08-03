"""Parquet writer for NoC transactions.

Subscribes to :class:`NoCEvent` and writes one row per emission — a
flat columnar form suitable for SQL analysis of data movement:

    cycle, chip, core_y, core_x, unit, phase, txn_type,
    src_y, src_x, dst_y, dst_x, size_bytes, txn_id,
    issue_cycle, arrival_cycle, flight_cycles, cost_model

Partitioned by ``chip`` to keep multi-chip runs separable; not
partitioned by kernel_id because NoC transactions aren't naturally
kernel-bound (firmware setup also generates traffic).

``issue_cycle`` is the cycle the *sending* NIU put the packet on the
wire and ``arrival_cycle`` (== ``cycle``, kept for readability at the
query site) the cycle the receiving NIU serviced it, so
``flight_cycles`` is their difference. All three are measurements in
either regime, but they mean different things and the ``cost_model``
column says which: with ``TT_SIM_COST_MODEL`` unset a packet is
delivered on the next cycle no matter how far it travelled, so every
flight is the one or two cycles the two-list swap in ``NUI.clock_tick``
really costs — a real observation about an un-modelled NoC, not an
estimate of a hop count. With it set, ``flight_cycles`` is the modelled
per-hop latency (plus any DRAM service time the destination charges).
``issue_cycle`` is ``-1``, and ``flight_cycles`` ``0``, for the one case
neither regime can time: a NIU with no owning tile clock.

Still not populated, and still gated on §I: ``vc``. Nothing in tt-sim
models virtual channels, so there is no column rather than a column of
zeroes.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tt_sim.perf.model import cost_model_enabled
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
        self._cost_model = cost_model_enabled()
        self._bus = bus if bus is not None else get_bus()
        self._bus.subscribe(EventCategory.NOC, self._on_event)

    def _on_event(self, event: NoCEvent):
        chip, core_y, core_x, unit = event.unit_id
        src = event.src if len(event.src) >= 2 else (0, 0)
        dst = event.dst if len(event.dst) >= 2 else (0, 0)
        issue = int(event.issue_cycle)
        flight = max(0, int(event.cycle) - issue) if issue >= 0 else 0
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
                "issue_cycle": issue,
                "arrival_cycle": int(event.cycle),
                "flight_cycles": flight,
                "cost_model": self._cost_model,
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
