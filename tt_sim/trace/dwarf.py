"""DWARF index — map PCs back to (source file, line) for trace
attribution.

Used by :class:`~tt_sim.trace.writers.lcov.LCOVWriter` (and any future
writer that needs source-level join keys). Built once at construction
time from one or more ELF files; lookup is a single dict access.

PC ranges from different ELFs may overlap (BRISC firmware + kernel
share an address space, for instance). The index uses last-load-wins
semantics: ELFs loaded later override entries from earlier ones at
the same PC. Pass kernel ELFs after firmware if you want kernel
attribution to take priority.

Missing DWARF info is not an error — ELFs without ``.debug_*``
sections (stripped builds, no ``-g``) load as no-ops and the writer
simply emits no coverage for instructions in those ranges.
"""

from pathlib import Path

from elftools.elf.elffile import ELFFile


class DwarfIndex:
    def __init__(self):
        # pc -> (file, line) — function name omitted for now, can be
        # filled in later via DIE traversal if a consumer needs it.
        self._pc_to_loc: dict[int, tuple[str, int]] = {}
        self._loaded_elfs: list[str] = []

    def load(self, elf_path: Path | str) -> int:
        """Load an ELF, populate the index. Returns the count of new PC
        entries added (may be 0 for stripped ELFs)."""
        path = Path(elf_path)
        with path.open("rb") as f:
            elf = ELFFile(f)
            if not elf.has_dwarf_info():
                self._loaded_elfs.append(str(path))
                return 0
            dwarfinfo = elf.get_dwarf_info()
            added = 0
            for cu in dwarfinfo.iter_CUs():
                lp = dwarfinfo.line_program_for_CU(cu)
                if lp is None:
                    continue
                # The line program's file_entry table is 1-indexed in
                # DWARF v4 and earlier; DWARF v5 includes index 0 as
                # the CU's own file. pyelftools normalises this.
                file_entries = lp["file_entry"]
                for entry in lp.get_entries():
                    state = entry.state
                    if state is None or state.end_sequence:
                        continue
                    file_idx = state.file
                    if 0 <= file_idx - 1 < len(file_entries):
                        fname = file_entries[file_idx - 1].name
                        if isinstance(fname, bytes):
                            fname = fname.decode("utf-8", "replace")
                    else:
                        fname = "<unknown>"
                    self._pc_to_loc[state.address] = (fname, state.line)
                    added += 1
        self._loaded_elfs.append(str(path))
        return added

    def lookup(self, pc: int) -> tuple[str, int] | None:
        return self._pc_to_loc.get(pc)

    def __len__(self) -> int:
        return len(self._pc_to_loc)

    def loaded_elfs(self) -> list[str]:
        return list(self._loaded_elfs)
