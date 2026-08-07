"""LCOV writer — emit `genhtml`-compatible coverage info from PC
retirement counts.

Subscribes to :class:`InstrEvent` and increments a per-(file, line)
hit counter via :class:`~tt_sim.trace.dwarf.DwarfIndex` resolution.
PCs without DWARF coverage are silently skipped. Output is the LCOV
trace format readable by ``genhtml``, GitHub Codecov, the VS Code
Coverage Gutters extension, and most CI coverage reporters:

    TN:tt-sim
    SF:/abs/path/source.c
    DA:42,150
    DA:43,150
    DA:44,12
    LF:3
    LH:3
    end_of_record

"Coverage" here means "cycles spent at this line" rather than
"executed at least once" — the DA count is the number of times an
``InstrEvent`` fired at a PC mapping to that line. Tools render this
identically to standard coverage (heatmap on the source view), but
hot lines stand out instead of just covered lines.

Source-attribution-aware viewers (`genhtml`, Coverage Gutters) need
the source files to be present at the recorded paths. ELFs built with
``-g`` typically embed absolute paths from the build host; copy or
symlink the source tree appropriately if you're viewing on a
different machine.
"""

from collections import defaultdict
from pathlib import Path

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.dwarf import DwarfIndex
from tt_sim.trace.events import EventCategory, InstrEvent


class LCOVWriter:
    def __init__(
        self,
        path: Path | str,
        dwarf_index: DwarfIndex,
        test_name: str = "tt-sim",
        bus: EventBus | None = None,
    ):
        self._path = Path(path)
        self._dwarf = dwarf_index
        self._test_name = test_name
        # (file, line) -> hit count
        self._hits: dict[tuple[str, int], int] = defaultdict(int)
        self._bus = bus if bus is not None else get_bus()
        self._bus.subscribe(EventCategory.INSTR, self._on_event)

    def _on_event(self, event: InstrEvent):
        if event.stalled:
            return
        # ``nearest`` rather than ``lookup``: a DWARF line program records a
        # row only where the source position changes, so an exact-PC lookup
        # drops most of the run. See tt_sim/trace/dwarf.py.
        loc = self._dwarf.nearest(event.pc, unit=event.unit_id[-1])
        if loc is None:
            return
        self._hits[(loc.file, loc.line)] += 1

    def close(self):
        # Group by source file for LCOV record emission.
        by_file: dict[str, dict[int, int]] = defaultdict(dict)
        for (fname, line), count in self._hits.items():
            by_file[fname][line] = count

        with self._path.open("w") as f:
            for fname, lines in sorted(by_file.items()):
                f.write(f"TN:{self._test_name}\n")
                f.write(f"SF:{fname}\n")
                hit_count = 0
                for line, count in sorted(lines.items()):
                    f.write(f"DA:{line},{count}\n")
                    if count > 0:
                        hit_count += 1
                f.write(f"LF:{len(lines)}\n")
                f.write(f"LH:{hit_count}\n")
                f.write("end_of_record\n")
        self._hits.clear()
