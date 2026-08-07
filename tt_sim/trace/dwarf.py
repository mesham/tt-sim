"""DWARF index — map PCs back to (function, source file, line) for trace
attribution.

Used by :class:`~tt_sim.trace.writers.lcov.LCOVWriter` and
:mod:`tt_sim.trace.hotspots` (the ranked-report path). Built once at
construction time from one or more ELF files; lookup is a dict access
plus, for :meth:`DwarfIndex.nearest`, one bisect.

**Per-unit scoping.** The five baby RISC-V cores in a Tensix tile each
run their own ELF and their address ranges overlap — BRISC kernel text
lives at the same L1 address as TRISC2's, and NCRISC's runs out of its
own IRAM window. Load each ELF against the unit that executes it
(``load(path, unit="TRISC2")``) and look up the same way; a lookup with
no unit, or for a unit that loaded nothing, falls back to the merged
index with last-load-wins semantics (which is what the LCOV writer,
which has no unit context per PC, has always used).

**Function names** come from a DIE walk over ``DW_TAG_subprogram`` and
``DW_TAG_inlined_subroutine``, resolving ``DW_AT_abstract_origin`` /
``DW_AT_specification`` chains for the (very common) case where the
concrete DIE carries no name of its own. The innermost range containing
a PC wins, so a PC inside an inlined ``cb_wait_front`` is attributed to
``cb_wait_front`` rather than to the ``kernel_main`` it was inlined
into. That distinction is the whole value of the feature on tt-metal
kernels, where LTO inlines essentially everything into one subprogram.
Names are DWARF source names, already human-readable — no demangling
step is needed.

**Relocations.** tt-metal's kernel ELFs are fully linked ``ET_EXEC``
files that nonetheless retain ``.rela.debug_*`` sections. pyelftools
defaults to applying those relocations and raises
``ELFRelocationError: Unsupported relocation type: 1`` on RISC-V, which
made every real tt-metal kernel ELF unloadable. The debug sections are
already relocated (verified against ``readelf --debug-dump=decodedline``,
which reports the same addresses), so the leftover ``.rela`` entries are
vestigial and we parse with ``relocate_dwarf_sections=False``.

Missing DWARF info is not an error — ELFs without ``.debug_*`` sections
(stripped builds, no ``-g``) load as no-ops and consumers simply get no
attribution for instructions in those ranges.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

from elftools.elf.elffile import ELFFile

#: Attributes consulted, in order, for a subprogram's name.
_NAME_ATTRS = ("DW_AT_name", "DW_AT_linkage_name")
#: Attributes followed when a DIE carries no name of its own.
_REF_ATTRS = ("DW_AT_abstract_origin", "DW_AT_specification")
#: Guard against a pathological (or cyclic) reference chain.
_MAX_REF_DEPTH = 8


@dataclass(frozen=True, slots=True)
class SourceLoc:
    """Where a PC came from. ``function`` is ``""`` when the ELF's DIE
    tree has no subprogram covering the PC — a real state (compiler-
    generated thunks, ``.init`` stubs), not a lookup failure."""

    file: str
    line: int
    function: str = ""

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}"
        return f"{self.function} ({where})" if self.function else where


def _die_name(die, depth: int = 0) -> str:
    """Resolve a DIE's name, following abstract_origin/specification."""
    if depth > _MAX_REF_DEPTH:
        return ""
    attrs = die.attributes
    for key in _NAME_ATTRS:
        if key in attrs:
            value = attrs[key].value
            return (
                value.decode("utf-8", "replace")
                if isinstance(value, bytes)
                else str(value)
            )
    for key in _REF_ATTRS:
        if key in attrs:
            try:
                ref = die.get_DIE_from_attribute(key)
            except Exception:
                return ""
            return _die_name(ref, depth + 1)
    return ""


def _die_pc_range(die) -> tuple[int, int] | None:
    """``(low, high)`` for a DIE with a contiguous range, else ``None``.

    DWARF4+ encodes ``DW_AT_high_pc`` either as an address or as an
    offset from ``low_pc``; the form discriminates.
    """
    low = die.attributes.get("DW_AT_low_pc")
    high = die.attributes.get("DW_AT_high_pc")
    if low is None or high is None:
        return None
    lo = low.value
    hi = high.value
    if high.form != "DW_FORM_addr":
        hi = lo + hi
    if hi <= lo:
        return None
    return lo, hi


class DwarfIndex:
    def __init__(self):
        # unit -> {pc: SourceLoc}. ``None`` is the merged/unscoped index.
        self._by_unit: dict[str | None, dict[int, SourceLoc]] = {}
        # unit -> sorted PC list, rebuilt lazily for nearest() lookups.
        self._sorted: dict[str | None, list[int]] = {}
        # unit -> [(low, high, name)] sorted by (low, -high) for innermost-wins.
        self._funcs: dict[str | None, list[tuple[int, int, str]]] = {}
        # unit -> [(low, high)] the line program actually covers. ``nearest``
        # refuses to answer outside these, so a PC from a *different* ELF is
        # reported unresolved instead of being attributed to whatever row
        # happens to sit below it. Without this bound a run whose ELFs could
        # not be found still reports ~100 % "resolved", pointing every hotspot
        # at some unrelated kernel's source.
        self._covered: dict[str | None, list[tuple[int, int]]] = {}
        self._loaded_elfs: list[str] = []

    # -- loading ---------------------------------------------------------

    def load(self, elf_path: Path | str, unit: str | None = None, bias: int = 0) -> int:
        """Load an ELF and populate the index for ``unit`` (and the merged
        index). Returns the count of PC entries added — 0 for a stripped
        ELF, which is not an error.

        ``bias`` is added to every address, for an ELF that runs somewhere
        other than where it was linked. tt-metal places a kernel at a
        runtime L1 base of its own choosing, so without a bias the kernel
        half of a run resolves to nothing.
        """
        path = Path(elf_path)
        self._loaded_elfs.append(str(path))
        with path.open("rb") as handle:
            elf = ELFFile(handle)
            if not elf.has_dwarf_info():
                return 0
            # See the module docstring: the .rela.debug_* sections tt-metal's
            # linker leaves behind are vestigial and pyelftools cannot apply
            # them on RISC-V.
            dwarfinfo = elf.get_dwarf_info(relocate_dwarf_sections=False)
            funcs = self._collect_functions(dwarfinfo, bias)
            added = self._collect_lines(dwarfinfo, funcs, unit, bias)
        return added

    def _collect_functions(
        self, dwarfinfo, bias: int = 0
    ) -> list[tuple[int, int, str]]:
        """Every named, contiguous subprogram/inlined range in the ELF."""
        found: list[tuple[int, int, str]] = []

        def walk(die):
            if die.tag in ("DW_TAG_subprogram", "DW_TAG_inlined_subroutine"):
                span = _die_pc_range(die)
                if span is not None:
                    name = _die_name(die)
                    if name:
                        found.append((span[0] + bias, span[1] + bias, name))
            for child in die.iter_children():
                walk(child)

        for cu in dwarfinfo.iter_CUs():
            try:
                walk(cu.get_top_DIE())
            except Exception:
                # A malformed CU should cost that CU's names, not the load.
                continue
        # Innermost wins: for equal low_pc prefer the *narrowest* range, so a
        # linear scan from the bisect point can stop at the first cover.
        found.sort(key=lambda f: (f[0], f[1] - f[0]))
        return found

    def _collect_lines(self, dwarfinfo, funcs, unit, bias: int = 0) -> int:
        added = 0
        covered: list[tuple[int, int]] = []
        for cu in dwarfinfo.iter_CUs():
            lineprog = dwarfinfo.line_program_for_CU(cu)
            if lineprog is None:
                continue
            # The line program's file_entry table is 1-indexed in DWARF v4 and
            # earlier; DWARF v5 includes index 0 as the CU's own file.
            # pyelftools normalises this.
            file_entries = lineprog["file_entry"]
            seq_start: int | None = None
            for entry in lineprog.get_entries():
                state = entry.state
                if state is None:
                    continue
                address = state.address + bias
                if state.end_sequence:
                    # The end_sequence row's address is one past the last byte
                    # of the sequence — exactly the covered range's upper bound.
                    if seq_start is not None and address > seq_start:
                        covered.append((seq_start, address))
                    seq_start = None
                    continue
                if seq_start is None:
                    seq_start = address
                idx = state.file
                if 0 <= idx - 1 < len(file_entries):
                    fname = file_entries[idx - 1].name
                    if isinstance(fname, bytes):
                        fname = fname.decode("utf-8", "replace")
                else:
                    fname = "<unknown>"
                # No function name here: it is resolved at lookup, against the
                # PC actually executed. Doing it per line-program row costs a
                # bisect for every row in the ELF — thousands per firmware
                # image, all at process exit — and is *less* accurate, because
                # a PC reached through ``nearest`` would inherit the function
                # of the row below it rather than its own innermost range.
                loc = SourceLoc(fname, state.line)
                for key in (unit, None):
                    table = self._by_unit.setdefault(key, {})
                    table[address] = loc
                    self._sorted.pop(key, None)
                added += 1
        for key in (unit, None):
            merged = self._funcs.setdefault(key, [])
            merged.extend(funcs)
            merged.sort(key=lambda f: (f[0], f[1] - f[0]))
            spans = self._covered.setdefault(key, [])
            spans.extend(covered)
            spans.sort()
        return added

    # -- lookup ----------------------------------------------------------

    def lookup(self, pc: int, unit: str | None = None) -> SourceLoc | None:
        """Exact-PC lookup. Tries the unit's own index first, then the
        merged one."""
        for key in (unit, None) if unit is not None else (None,):
            table = self._by_unit.get(key)
            if table is not None:
                loc = table.get(pc)
                if loc is not None:
                    return loc
        return None

    def nearest(self, pc: int, unit: str | None = None) -> SourceLoc | None:
        """Floor lookup — the location of the greatest indexed PC ``<= pc``.

        A DWARF line program records a row only where the source position
        *changes*, so most executed PCs have no exact entry. Attribution
        that only ever does an exact lookup silently drops the majority of
        a run; this is the lookup a profiler wants.

        **Bounded to what the line program covers.** A PC outside every
        recorded sequence returns ``None`` rather than the nearest row
        below it. Without that, an index built from the wrong ELF answers
        every query and a report shows ~100 % attribution to a kernel that
        never ran — a confidently wrong answer, which is worse than none.
        """
        for key in (unit, None) if unit is not None else (None,):
            table = self._by_unit.get(key)
            if not table or not _in_covered(self._covered.get(key, ()), pc):
                continue
            keys = self._sorted.get(key)
            if keys is None:
                keys = sorted(table)
                self._sorted[key] = keys
            idx = bisect.bisect_right(keys, pc) - 1
            if idx >= 0:
                return table[keys[idx]]
        return None

    def covers(self, pc: int, unit: str | None = None) -> bool:
        """Whether any loaded ELF's line program covers ``pc``."""
        for key in (unit, None) if unit is not None else (None,):
            if _in_covered(self._covered.get(key, ()), pc):
                return True
        return False

    def function_at(self, pc: int, unit: str | None = None) -> str:
        """Innermost named function range covering ``pc``, else ``""``."""
        for key in (unit, None) if unit is not None else (None,):
            name = _lookup_function(self._funcs.get(key, []), pc)
            if name:
                return name
        return ""

    def units(self) -> list[str]:
        return sorted(u for u in self._by_unit if u is not None)

    def __len__(self) -> int:
        return len(self._by_unit.get(None, {}))

    def loaded_elfs(self) -> list[str]:
        return list(self._loaded_elfs)


def _in_covered(spans, pc: int) -> bool:
    """Is ``pc`` inside any ``[low, high)`` of a sorted, possibly
    overlapping, span list?"""
    if not spans:
        return False
    idx = bisect.bisect_right(spans, (pc, float("inf"))) - 1
    # Spans from separate CUs can nest or overlap, so scan back a bounded
    # distance rather than trusting the single floor entry.
    scanned = 0
    while idx >= 0 and scanned < 64:
        low, high = spans[idx]
        if low <= pc < high:
            return True
        idx -= 1
        scanned += 1
    return False


def _lookup_function(funcs: list[tuple[int, int, str]], pc: int) -> str:
    """Innermost covering range in a list sorted by (low, width)."""
    if not funcs:
        return ""
    # Every candidate has low <= pc; scan back from the insertion point and
    # keep the narrowest cover. Ranges are small and heavily nested, so the
    # scan is bounded in practice by the nesting depth times a small constant.
    idx = bisect.bisect_right(funcs, (pc, float("inf"), "")) - 1
    best: tuple[int, str] | None = None
    scanned = 0
    while idx >= 0 and scanned < 512:
        low, high, name = funcs[idx]
        if low <= pc < high:
            width = high - low
            if best is None or width < best[0]:
                best = (width, name)
        idx -= 1
        scanned += 1
    return best[1] if best else ""
