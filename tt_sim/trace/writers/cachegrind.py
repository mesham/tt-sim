"""Cachegrind-format memory access writer.

Subscribes to :class:`MemEvent` and emits a Callgrind/Cachegrind text
file consumable directly by ``kcachegrind`` / ``qcachegrind`` /
``callgrind_annotate``. Format reference:
<https://kcachegrind.github.io/html/CallgrindFormat.html>.

Each retiring memory access is bucketed by ``(pc, address)`` and
counted as a data read (``Dr``) or data write (``Dw``). Functions are
synthesised from the access region (``L1`` / ``MMIO`` / ...) so
KCachegrind's call-cost tree gives one collapsible entry per region;
addresses appear underneath, ranked by access count.

Accesses without a PC (NoC-driven or internal, ``pc == 0``) are
grouped under a synthetic ``no_pc`` function so they're visible but
don't pollute the per-instruction breakdown.

When §H Phase 6 (DWARF / LCOV) lands, the synthetic file/function
names can be replaced with real source coordinates with no change to
the writer's contract.
"""

from collections import defaultdict
from pathlib import Path

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import EventCategory, MemEvent


class MemoryTraceWriter:
    def __init__(self, path: Path | str, bus: EventBus | None = None):
        self._path = Path(path)
        # Bucket: (region, pc, address) -> (Dr_count, Dw_count)
        self._buckets: dict[tuple[str, int, int], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        self._bus = bus if bus is not None else get_bus()
        self._bus.subscribe(EventCategory.MEM, self._on_event)

    def _on_event(self, event: MemEvent):
        key = (event.region or "?", event.pc, event.address)
        bucket = self._buckets[key]
        if event.op == "read":
            bucket[0] += 1
        elif event.op == "write":
            bucket[1] += 1

    def close(self):
        # Group buckets by (region, pc) for the Callgrind "fn=" sections.
        by_fn: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
        for (region, pc, addr), (dr, dw) in self._buckets.items():
            by_fn[(region, pc)].append((addr, dr, dw))

        with self._path.open("w") as f:
            f.write("# tt-sim memory trace\n")
            f.write("# Callgrind text format for KCachegrind\n")
            f.write("events: Dr Dw\n")
            f.write("positions: instr\n")
            f.write("ob=tt-sim\n")
            f.write("fl=memory\n")
            for (region, pc), rows in by_fn.items():
                fn_name = f"{region}_pc_0x{pc:08x}" if pc else f"{region}_no_pc"
                f.write(f"fn={fn_name}\n")
                rows.sort()
                for addr, dr, dw in rows:
                    f.write(f"0x{addr:x} {dr} {dw}\n")
                f.write("\n")
        self._buckets.clear()
