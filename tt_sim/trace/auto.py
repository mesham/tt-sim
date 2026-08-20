"""Environment-driven auto-setup for tracing.

Called from `TT_Device.__init__` (and `_begin_construction`) so any driver
script that constructs a device picks up trace env vars automatically —
no per-example wiring, and no per-architecture wiring either. It used to be
called from `Wormhole.__init__` only, which is why Blackhole devices silently
ignored every var below; `tt_sim/device/parity_test.py` guards against that
returning.

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
- ``TT_SIM_TRACE_INVARIANTS`` — write violations of the seed
  architectural invariants (PC alignment, mem alignment, lifecycle
  ordering, NoC request/response pairing) to a JSONL file. Set
  ``TT_SIM_TRACE_INVARIANTS_STRICT=1`` to additionally raise on
  first violation.
- ``TT_SIM_TRACE_STATE_DUMP`` — capture a JSON state dump at each
  lifecycle boundary (kernel start/done) for cross-run / cross-sim
  diffing via ``python3 -m tt_sim.trace.diff_state``.

All writers can be enabled simultaneously; they subscribe to disjoint
event handling and write independent outputs.

**The one-shot entry point.** ``TT_SIM_PROFILE=<dir>`` is the documented
"here is my kernel, tell me where its cycles went" invocation: it turns
on the counter aggregator and the per-PC hotspot aggregator, discovers
the kernel ELFs tt-metal just built, and writes a ranked
``<dir>/report.md`` (plus ``report.json``, ``hotspots.json`` and the raw
``counters/`` dataset) when the process exits. Everything it enables is
also reachable individually via the vars above; the point of the single
var is that a reader who has never opened this file gets a usable answer
without choosing between them. Companion knobs:

- ``TT_SIM_PROFILE_ELFS`` — ``BRISC:/path.elf,TRISC2:/path.elf`` (or a
  bare comma-separated list, unit inferred from the filename) to skip
  auto-discovery entirely.
- ``TT_SIM_PROFILE_INTERVAL`` — counter flush cadence in cycles.
- ``TT_SIM_PROFILE_LABEL`` — a name for the run, printed in the report.
"""

import atexit
import json
import os
import sys
from pathlib import Path

from tt_sim.trace.bus import get_bus
from tt_sim.trace.counters import DEFAULT_FLUSH_INTERVAL_CYCLES, CounterAggregator
from tt_sim.trace.dwarf import DwarfIndex
from tt_sim.trace.elfdisc import REJECTED, discover, session_start
from tt_sim.trace.hotspots import HotspotAggregator
from tt_sim.trace.ids import get_registry
from tt_sim.trace.invariants import InvariantRunner
from tt_sim.trace.state_dump import StateDumpWriter
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
_INVARIANTS: InvariantRunner | None = None
_STATE_WRITER: StateDumpWriter | None = None
_HOTSPOTS: HotspotAggregator | None = None
_PROFILE: dict | None = None


def enable_from_env(device=None) -> None:
    """Wire up any tracing writers configured via environment variables.

    Idempotent — calling more than once is safe; only the first call
    per process actually opens files. The ``device`` argument is used
    by the state-dump writer (which needs to poll device state at
    lifecycle boundaries); other writers don't need it. Any arch's
    device works -- this is called from both Wormhole and Blackhole.
    """
    global _JSONL, _PERFETTO, _COMMITLOG, _COUNTERS_AGG, _COUNTERS_WRITER
    global _NOC_WRITER, _MEMORY_WRITER, _LCOV_WRITER, _INVARIANTS, _STATE_WRITER
    global _HOTSPOTS, _PROFILE
    jsonl_path = os.environ.get("TT_SIM_TRACE")
    perfetto_path = os.environ.get("TT_SIM_TRACE_PERFETTO")
    commitlog_path = os.environ.get("TT_SIM_TRACE_COMMITLOG")
    counters_path = os.environ.get("TT_SIM_TRACE_COUNTERS")
    noc_path = os.environ.get("TT_SIM_TRACE_NOC")
    memory_path = os.environ.get("TT_SIM_TRACE_MEMORY")
    lcov_path = os.environ.get("TT_SIM_TRACE_LCOV")
    invariants_path = os.environ.get("TT_SIM_TRACE_INVARIANTS")
    state_dump_path = os.environ.get("TT_SIM_TRACE_STATE_DUMP")
    profile_dir = os.environ.get("TT_SIM_PROFILE")

    # ``TT_SIM_PROFILE`` is a preset over the vars above: it owns the counter
    # dataset unless the user asked for one somewhere else, in which case that
    # wins and the profile just reads it.
    if profile_dir and not counters_path:
        counters_path = str(Path(profile_dir) / "counters")

    if not any(
        (
            jsonl_path,
            perfetto_path,
            commitlog_path,
            counters_path,
            noc_path,
            memory_path,
            lcov_path,
            invariants_path,
            state_dump_path,
            profile_dir,
        )
    ):
        return

    bus = get_bus()
    bus.enabled = True

    if profile_dir and _PROFILE is None:
        _PROFILE = {
            "dir": Path(profile_dir),
            "counters": counters_path,
            "since": session_start(),
            "label": os.environ.get("TT_SIM_PROFILE_LABEL", ""),
            "device": device,
        }
        if _HOTSPOTS is None:
            _HOTSPOTS = HotspotAggregator()
    elif _PROFILE is not None and _PROFILE.get("device") is None:
        # ``TT_Device`` calls this once before it has a device reference and
        # once after; keep the later, usable one for byte verification.
        _PROFILE["device"] = device

    if jsonl_path and _JSONL is None:
        _JSONL = JSONLLogger(jsonl_path)

    if perfetto_path and _PERFETTO is None:
        _PERFETTO = PerfettoWriter(perfetto_path)

    if commitlog_path and _COMMITLOG is None:
        _COMMITLOG = SpikeCommitlogWriter(commitlog_path)

    if counters_path and _COUNTERS_WRITER is None:
        interval_env = os.environ.get(
            "TT_SIM_TRACE_COUNTERS_INTERVAL"
        ) or os.environ.get("TT_SIM_PROFILE_INTERVAL")
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

    if invariants_path and _INVARIANTS is None:
        strict = os.environ.get("TT_SIM_TRACE_INVARIANTS_STRICT", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        _INVARIANTS = InvariantRunner(strict=strict)

    if state_dump_path and _STATE_WRITER is None and device is not None:
        _STATE_WRITER = StateDumpWriter(state_dump_path, device)

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
            if _INVARIANTS is not None and invariants_path is not None:
                n = _INVARIANTS.report(invariants_path)
                if n > 0:
                    print(
                        f"[tt-sim trace] {n} invariant violation(s) recorded "
                        f"to {invariants_path}",
                        file=__import__("sys").stderr,
                    )
            if _STATE_WRITER is not None:
                _STATE_WRITER.close()
            if _PROFILE is not None:
                write_profile_report(_PROFILE, _HOTSPOTS)

        atexit.register(_on_exit)
        enable_from_env._atexit_registered = True  # type: ignore[attr-defined]


#: How much of a tile's L1 to snapshot when looking for a relocated kernel.
#: Kernel text is placed in the low megabyte; snapshotting more costs time
#: at exit for no extra hits.
_L1_SEARCH_BYTES = 1 << 20


def _device_probes(device):
    """``(verifier, searcher)`` over the first live Tensix tile.

    ``verifier(unit, vaddr, size)`` proves a candidate ELF is resident where
    it was linked. It returns ``None`` — abstain, not "mismatch" — when the
    address is not readable through this tile, which is the common case for a
    core's private local-memory window.

    ``searcher(blob)`` finds a byte string anywhere in the tile's L1 and is
    what recovers the load bias of a kernel tt-metal placed somewhere other
    than its link address. The snapshot is taken once, lazily, and only if a
    verifier miss actually asks for it.
    """
    tiles = getattr(device, "tensix_tiles", None) or getattr(
        getattr(device, "tt_device", None), "tensix_tiles", None
    )
    if not tiles:
        return None, None
    tile = tiles[0]
    snapshot: list[bytes | None] = []

    def read(unit, vaddr, size):
        # Read through the *core's own* memory view, not the tile's. The five
        # baby cores each map a private local memory at 0xFFB00000 (and NCRISC
        # an IRAM at 0xFFC00000); a tile-level read of those addresses answers
        # for whichever core the tile map happens to resolve to, so it can
        # confidently contradict an ELF that is in fact resident.
        core = getattr(tile, str(unit).lower(), None)
        source = getattr(core, "visible_memory", None) or tile
        try:
            return bytes(source.read(vaddr, size))
        except Exception:
            return None

    def search(blob):
        if not snapshot:
            try:
                snapshot.append(bytes(tile.read(0, _L1_SEARCH_BYTES)))
            except Exception:
                snapshot.append(None)
        l1 = snapshot[0]
        if l1 is None:
            return None
        at = l1.find(blob)
        if at < 0:
            return None
        # Ambiguity is a wrong answer waiting to happen: two copies of the
        # same text mean the bias cannot be pinned down.
        if l1.find(blob, at + 1) >= 0:
            return None
        return at

    return read, search


def write_profile_report(profile: dict, hotspots) -> None:
    """Resolve the hotspot table against auto-discovered ELFs and render
    the ranked report. Never raises: a profiling run that has already
    produced its answer must not fail at exit because a cache moved."""
    from tt_sim.perf.model import cost_model_enabled
    from tt_sim.trace import report as report_mod

    directory = Path(profile["dir"])
    try:
        directory.mkdir(parents=True, exist_ok=True)
        verifier, searcher = _device_probes(profile.get("device"))
        found = discover(
            since=profile.get("since"), verifier=verifier, searcher=searcher
        )
        index = DwarfIndex()
        elf_meta = []
        for entry in found.elfs:
            row = {
                "unit": entry.unit,
                "role": entry.role,
                "path": str(entry.path),
                "how": entry.how,
                # The PCs in ``hotspots.json`` are the addresses the core
                # actually executed. ``bias`` is what tt-metal added to this
                # ELF's link addresses to get there, so ``pc - bias`` indexes
                # back into the ELF. It is recorded as a number, not only
                # inside the ``how`` prose, because a consumer converting
                # addresses must not have to parse English.
                "bias": int(entry.bias),
            }
            if entry.how == REJECTED:
                # Positively identified as the wrong ELF; loading it would put
                # some other kernel's function names on this run's hotspots.
                elf_meta.append(row)
                continue
            try:
                added = index.load(entry.path, unit=entry.unit, bias=entry.bias)
            except Exception as exc:  # a bad ELF costs its unit, not the run
                row["how"] = f"{entry.how} (unreadable: {exc})"
                elf_meta.append(row)
                continue
            if not added:
                row["how"] = f"{entry.how}, no DWARF"
            elf_meta.append(row)

        table = hotspots.resolve(index) if hotspots is not None else None
        payload = report_mod.hotspots_to_dict(table) if table is not None else {}
        if payload:
            # Provenance travels with the table, not only in profile.json. A
            # hotspot table is identical in shape whether every name on it was
            # byte-verified against the device or guessed by mtime from some
            # other kernel's build, and a consumer reading this file alone
            # would have no way to tell. ``how`` per unit is that way.
            payload["elfs"] = elf_meta
            payload["cost_model"] = cost_model_enabled()
            (directory / "hotspots.json").write_text(json.dumps(payload, indent=2))

        notes = [found.note] if found.note else []
        meta = {
            # One version across the profile artefact set (report.json,
            # hotspots.json, profile.json) — they are written by one run and
            # read together. See docs/trace-schema.md §9.
            "schema_version": report_mod.SCHEMA_VERSION,
            "label": profile.get("label") or directory.name,
            # Ask the model itself rather than re-parsing the variable, so
            # `TT_SIM_COST_MODEL=0` is reported as off, not as truthy.
            "cost_model": cost_model_enabled(),
            "elfs": elf_meta,
            "notes": notes,
            "elf_roots": found.roots,
            # Where the counter dataset actually landed. Usually
            # ``<dir>/counters``, but ``TT_SIM_TRACE_COUNTERS`` overrides it,
            # and without this the documented ``python3 -m tt_sim.trace.report
            # <dir>`` re-render silently produces a report with no counters.
            "counters": str(profile.get("counters") or ""),
        }
        (directory / "profile.json").write_text(json.dumps(meta, indent=2))

        built = report_mod.build(
            profile.get("counters"),
            hotspots=payload,
            elfs=elf_meta,
            cost_model=meta["cost_model"],
            label=meta["label"],
            notes=notes,
        )
        path = report_mod.write(built, directory)
        print(f"[tt-sim profile] ranked report written to {path}", file=sys.stderr)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[tt-sim profile] report generation failed: {exc}", file=sys.stderr)
