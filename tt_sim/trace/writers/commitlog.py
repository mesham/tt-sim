"""Spike-compatible RISC-V commitlog writer.

Subscribes to ``InstrEvent`` filtered to the five baby cores and emits
one line per retirement in the format `spike --log-commits` produces.
Output is one file per RV unit so a single tt-sim run can be diffed
against a Spike run hart-by-hart.

Line format (RV32, all cores reported as machine mode = privilege 3,
hart 0 so the file is drop-in comparable to a single-hart Spike run):

    core   0: 3 0x{pc:08x} (0x{instr:08x})[ x{reg:2d} 0x{val:08x}]

The trailing register write is omitted when the instruction doesn't
write to an architectural register (stores, branches, jumps without
link, writes to x0).
"""

from pathlib import Path
from typing import IO

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import EventCategory, InstrEvent

RV_UNITS = frozenset({"BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"})

# Machine mode privilege level. tt-sim's baby cores don't model
# privilege transitions, so every retirement is reported at M-mode.
PRIVILEGE_LEVEL = 3


class SpikeCommitlogWriter:
    def __init__(self, directory: Path | str, bus: EventBus | None = None):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, IO] = {}
        self._bus = bus if bus is not None else get_bus()
        self._bus.subscribe(EventCategory.INSTR, self._on_event)

    def _file_for(self, unit: str) -> IO:
        f = self._files.get(unit)
        if f is None:
            f = (self._dir / f"{unit.lower()}.commitlog").open("w")
            self._files[unit] = f
        return f

    def _on_event(self, event: InstrEvent):
        unit = event.unit_id[3]
        if unit not in RV_UNITS:
            return
        # Stalled cycles don't retire an instruction; Spike doesn't emit
        # a line for them either.
        if event.stalled:
            return
        line = (
            f"core   0: {PRIVILEGE_LEVEL} 0x{event.pc:08x} (0x{event.instruction:08x})"
        )
        if event.reg_write_idx >= 0:
            line += (
                f" x{event.reg_write_idx:2d} 0x{event.reg_write_value & 0xFFFFFFFF:08x}"
            )
        line += "\n"
        self._file_for(unit).write(line)

    def close(self):
        for f in self._files.values():
            if not f.closed:
                f.flush()
                f.close()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
