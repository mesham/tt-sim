"""Environment-driven auto-setup for tracing.

Called from `Wormhole.__init__` so any driver script that constructs a
device picks up trace env vars automatically — no per-example wiring.

Supported env vars (all optional, all default-off):

- ``TT_SIM_TRACE`` — JSONL writer; one event per line. Suitable for
  ``jq`` / ``duckdb`` / pandas analysis.
- ``TT_SIM_TRACE_PERFETTO`` — Chrome Trace Event Format writer; output
  loads directly into ``ui.perfetto.dev``. Use a ``.json.gz`` extension
  for gzip compression (Perfetto loads it natively).
- ``TT_SIM_TRACE_COMMITLOG`` — RISC-V Spike-compatible commitlog
  writer; output is a directory containing one ``<unit>.commitlog``
  file per baby core, drop-in comparable to a ``spike --log-commits``
  run via ``diff`` for differential testing.
- ``TT_SIM_TRACE_COUNTERS`` — Parquet performance-counter writer;
  output is a partitioned dataset (``chip=N/kernel_id=N/*.parquet``)
  ingestable directly by DuckDB / pandas. Flush cadence is
  configurable via ``TT_SIM_TRACE_COUNTERS_INTERVAL`` (default: 100
  cycles).

All writers can be enabled simultaneously; they subscribe to disjoint
event handling and write independent outputs.
"""

import atexit
import os

from tt_sim.trace.bus import get_bus
from tt_sim.trace.counters import DEFAULT_FLUSH_INTERVAL_CYCLES, CounterAggregator
from tt_sim.trace.ids import get_registry
from tt_sim.trace.writers.commitlog import SpikeCommitlogWriter
from tt_sim.trace.writers.jsonl import JSONLLogger
from tt_sim.trace.writers.parquet import ParquetCounterWriter
from tt_sim.trace.writers.perfetto import PerfettoWriter

_JSONL: JSONLLogger | None = None
_PERFETTO: PerfettoWriter | None = None
_COMMITLOG: SpikeCommitlogWriter | None = None
_COUNTERS_AGG: CounterAggregator | None = None
_COUNTERS_WRITER: ParquetCounterWriter | None = None


def enable_from_env() -> None:
    """Wire up any tracing writers configured via environment variables.

    Idempotent — calling more than once is safe; only the first call
    per process actually opens files.
    """
    global _JSONL, _PERFETTO, _COMMITLOG, _COUNTERS_AGG, _COUNTERS_WRITER
    jsonl_path = os.environ.get("TT_SIM_TRACE")
    perfetto_path = os.environ.get("TT_SIM_TRACE_PERFETTO")
    commitlog_path = os.environ.get("TT_SIM_TRACE_COMMITLOG")
    counters_path = os.environ.get("TT_SIM_TRACE_COUNTERS")

    if not any((jsonl_path, perfetto_path, commitlog_path, counters_path)):
        return

    bus = get_bus()
    bus.enabled = True

    if jsonl_path and _JSONL is None:
        _JSONL = JSONLLogger(jsonl_path)

    if perfetto_path and _PERFETTO is None:
        _PERFETTO = PerfettoWriter(perfetto_path)

    if commitlog_path and _COMMITLOG is None:
        _COMMITLOG = SpikeCommitlogWriter(commitlog_path)

    if counters_path and _COUNTERS_WRITER is None:
        interval_env = os.environ.get("TT_SIM_TRACE_COUNTERS_INTERVAL")
        interval = (
            int(interval_env)
            if interval_env is not None
            else DEFAULT_FLUSH_INTERVAL_CYCLES
        )
        # Writer subscribes BEFORE the aggregator so the writer sees
        # everything the aggregator emits.
        _COUNTERS_WRITER = ParquetCounterWriter(counters_path)
        _COUNTERS_AGG = CounterAggregator(flush_interval_cycles=interval)

    if not getattr(enable_from_env, "_atexit_registered", False):

        def _on_exit():
            if _COUNTERS_AGG is not None:
                _COUNTERS_AGG.flush()
            if _JSONL is not None:
                _JSONL.close()
                if jsonl_path:
                    get_registry().dump(jsonl_path + ".ids.json")
            if _PERFETTO is not None:
                _PERFETTO.close()
            if _COMMITLOG is not None:
                _COMMITLOG.close()
            if _COUNTERS_WRITER is not None:
                _COUNTERS_WRITER.close()

        atexit.register(_on_exit)
        enable_from_env._atexit_registered = True  # type: ignore[attr-defined]
