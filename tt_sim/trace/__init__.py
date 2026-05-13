from tt_sim.trace.auto import enable_from_env
from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.counters import CounterAggregator
from tt_sim.trace.dwarf import DwarfIndex
from tt_sim.trace.events import (
    ComputeEvent,
    CounterSnapshot,
    DispatchEvent,
    Event,
    EventCategory,
    InstrEvent,
    LifecycleEvent,
    MemEvent,
    NoCEvent,
    SyncEvent,
    Unit,
)
from tt_sim.trace.ids import IDRegistry, UnitID, get_registry
from tt_sim.trace.invariants import (
    DEFAULT_INVARIANTS,
    Invariant,
    InvariantRunner,
    LifecycleOrderInvariant,
    MemAlignmentInvariant,
    NoCRequestResponseInvariant,
    PCAlignmentInvariant,
    Violation,
)
from tt_sim.trace.state_dump import StateDumpWriter, dump_device_state
from tt_sim.trace.writers.cachegrind import MemoryTraceWriter
from tt_sim.trace.writers.commitlog import SpikeCommitlogWriter
from tt_sim.trace.writers.jsonl import JSONLLogger
from tt_sim.trace.writers.lcov import LCOVWriter
from tt_sim.trace.writers.noc_parquet import NoCParquetWriter
from tt_sim.trace.writers.parquet import ParquetCounterWriter
from tt_sim.trace.writers.perfetto import PerfettoWriter

__all__ = [
    "DEFAULT_INVARIANTS",
    "ComputeEvent",
    "CounterAggregator",
    "CounterSnapshot",
    "DispatchEvent",
    "DwarfIndex",
    "Event",
    "EventBus",
    "EventCategory",
    "IDRegistry",
    "InstrEvent",
    "Invariant",
    "InvariantRunner",
    "JSONLLogger",
    "LCOVWriter",
    "LifecycleEvent",
    "LifecycleOrderInvariant",
    "MemAlignmentInvariant",
    "MemEvent",
    "MemoryTraceWriter",
    "NoCEvent",
    "NoCParquetWriter",
    "NoCRequestResponseInvariant",
    "PCAlignmentInvariant",
    "ParquetCounterWriter",
    "PerfettoWriter",
    "SpikeCommitlogWriter",
    "StateDumpWriter",
    "SyncEvent",
    "Unit",
    "UnitID",
    "Violation",
    "dump_device_state",
    "enable_from_env",
    "get_bus",
    "get_registry",
]
