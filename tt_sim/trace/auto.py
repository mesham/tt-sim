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
- ``TT_SIM_TRACE_NOC`` — Parquet writer for NoC transactions, one
  row per emission. Partitioned by ``chip``.
- ``TT_SIM_TRACE_MEMORY`` — Callgrind/Cachegrind text writer for
  memory accesses (L1 / MMIO). Output is a single file consumable by
  ``kcachegrind`` / ``qcachegrind`` / ``callgrind_annotate``.
- ``TT_SIM_TRACE_LCOV`` — LCOV coverage-format writer for source-level
  attribution. Requires ``TT_SIM_TRACE_LCOV_ELFS`` (comma-separated
  paths to ELFs with DWARF info) to map PCs back to source lines.

All writers can be enabled simultaneously; they subscribe to disjoint
event handling and write independent outputs.
"""

import atexit
import os

from tt_sim.trace.bus import get_bus
from tt_sim.trace.counters import DEFAULT_FLUSH_INTERVAL_CYCLES, CounterAggregator
from tt_sim.trace.dwarf import DwarfIndex
from tt_sim.trace.ids import get_registry
from tt_sim.trace.writers.cachegrind import MemoryTraceWriter
from tt_sim.trace.writers.commitlog import SpikeCommitlogWriter
from tt_sim.trace.writers.jsonl import JSONLLogger
from tt_sim.trace.writers.lcov import LCOVWriter
from tt_sim.trace.writers.noc_parquet import NoCParquetWriter
from tt_sim.trace.writers.parquet import ParquetCounterWriter
from tt_sim.trace.writers.perfetto import PerfettoWriter

_JSONL: JSONLLogger | None = None
_PERFETTO: PerfettoWriter | None = None
_COMMITLOG: SpikeCommitlogWriter | None = None
_COUNTERS_AGG: CounterAggregator | None = None
_COUNTERS_WRITER: ParquetCounterWriter | None = None
_NOC_WRITER: NoCParquetWriter | None = None
_MEMORY_WRITER: MemoryTraceWriter | None = None
_LCOV_WRITER: LCOVWriter | None = None


def enable_from_env() -> None:
    """Wire up any tracing writers configured via environment variables.

    Idempotent — calling more than once is safe; only the first call
    per process actually opens files.
    """
    global _JSONL, _PERFETTO, _COMMITLOG, _COUNTERS_AGG, _COUNTERS_WRITER
    global _NOC_WRITER, _MEMORY_WRITER, _LCOV_WRITER
    jsonl_path = os.environ.get("TT_SIM_TRACE")
    perfetto_path = os.environ.get("TT_SIM_TRACE_PERFETTO")
    commitlog_path = os.environ.get("TT_SIM_TRACE_COMMITLOG")
    counters_path = os.environ.get("TT_SIM_TRACE_COUNTERS")
    noc_path = os.environ.get("TT_SIM_TRACE_NOC")
    memory_path = os.environ.get("TT_SIM_TRACE_MEMORY")
    lcov_path = os.environ.get("TT_SIM_TRACE_LCOV")

    if not any(
        (
            jsonl_path,
            perfetto_path,
            commitlog_path,
            counters_path,
            noc_path,
            memory_path,
            lcov_path,
        )
    ):
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

    if noc_path and _NOC_WRITER is None:
        _NOC_WRITER = NoCParquetWriter(noc_path)

    if memory_path and _MEMORY_WRITER is None:
        _MEMORY_WRITER = MemoryTraceWriter(memory_path)

    if lcov_path and _LCOV_WRITER is None:
        elfs_env = os.environ.get("TT_SIM_TRACE_LCOV_ELFS", "")
        index = DwarfIndex()
        for elf_path in (p.strip() for p in elfs_env.split(",") if p.strip()):
            index.load(elf_path)
        _LCOV_WRITER = LCOVWriter(lcov_path, index)

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
            if _NOC_WRITER is not None:
                _NOC_WRITER.close()
            if _MEMORY_WRITER is not None:
                _MEMORY_WRITER.close()
            if _LCOV_WRITER is not None:
                _LCOV_WRITER.close()

        atexit.register(_on_exit)
        enable_from_env._atexit_registered = True  # type: ignore[attr-defined]
