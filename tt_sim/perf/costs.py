"""Loader for the per-unit cycle-cost tables.

Two YAML files, one schema, one loader:

* ``tt_sim/pe/tensix/tensix_instruction_costs.yaml`` -- per-instruction costs
  for the Tensix backend units, keyed by the same ``ex_resource`` names and
  instruction names that ``tensix_instructions.yaml`` uses, so the two read
  side by side (ROADMAP.md section I).
* ``tt_sim/perf/unit_costs.yaml`` -- the NoC, DRAM, baby RISC-V cores, Mover
  and L1.

**Consumed through :mod:`tt_sim.perf.model`, and only when asked.** The tables
landed ahead of any consumer so the schema and the provenance discipline could
be reviewed on their own; Phase 5 of ``docs/plans/event-driven-pump.md`` then
wired the first unit — the Tensix matrix unit — to them, behind
``TT_SIM_COST_MODEL``. The judgement calls a consumer has to make (an
untabulated opcode costs nothing; a bound is not an equals sign) are made once,
in :mod:`tt_sim.perf.model`, and a test in ``costs_test.py`` pins the exact set
of modules that reach the tables at all.

The one property that matters more than any number in these files is that a
reader can tell a sourced constant from a placeholder at a glance. tt-sim is a
cycle-*approximate* estimator (ROADMAP "Positioning") because calibrating
against silicon needs RTL or captured traces that are not public, so a table
that could not distinguish "the ISA doc says 5" from "5 seemed about right"
would be unimprovable. Every entry therefore carries a :data:`PROVENANCE_RANK`
kind, and :meth:`CostTable.unsourced` exists so that the gaps can be listed
rather than discovered.
"""

from __future__ import annotations

import importlib.resources as resources
from copy import deepcopy
from dataclasses import dataclass, field

from tt_sim.util.yaml_cache import load_yaml_cached

#: Provenance kinds, ranked strongest to weakest. An entry's provenance is the
#: weakest provenance of any number in it, so ranking them lets a consumer ask
#: "is everything in this table at least ``vendor_source``?" in one comparison.
#:
#: ``vendor_source_derived`` is the rank the DRAM access latency needed and is
#: exactly ``isa_doc_derived``'s relationship one tier down: **arithmetic on
#: vendor numbers**, requiring both a ``source`` and a ``derivation`` so a
#: reader can redo the subtraction. It sits *below* ``vendor_source`` because
#: no document prints the result, and *above* ``estimated`` because it is not a
#: judgement call — every input is a published figure and the arithmetic is
#: written down. It is emphatically not a licence to dress a guess up: the test
#: ``test_vendor_derived_entries_are_exactly_the_ones_we_expect`` pins the list,
#: so adding one means editing it.
PROVENANCE_RANK = {
    "isa_doc": 5,
    "isa_doc_derived": 4,
    "vendor_source": 3,
    "vendor_source_derived": 2,
    "estimated": 1,
    "unknown": 0,
}

#: Provenance kinds that stand for a real published number.
SOURCED_PROVENANCE = frozenset(
    {"isa_doc", "isa_doc_derived", "vendor_source", "vendor_source_derived"}
)

#: Kinds that must name a document in the file's ``documents`` /
#: ``vendor_documents`` map.
PROVENANCE_REQUIRING_SOURCE = frozenset(
    {"isa_doc", "isa_doc_derived", "vendor_source", "vendor_source_derived"}
)

#: Kinds that must also show their working, so a reader can redo the
#: arithmetic rather than take the result on trust.
PROVENANCE_REQUIRING_DERIVATION = frozenset(
    {"isa_doc_derived", "vendor_source_derived"}
)

#: Kinds that must explain themselves in prose instead.
PROVENANCE_REQUIRING_NOTE = frozenset({"estimated", "unknown"})

#: How a cycle count relates to the number the source actually printed. These
#: exist because the ISA docs genuinely write ">= 2 cycles", "~5 cycles" and
#: "<= 2 cycles", and flattening those to a bare integer would be inventing
#: precision that was never published.
BOUNDS = frozenset({"exact", "at_least", "at_most", "approximate", "range"})

_TENSIX_YAML = "tensix_instruction_costs.yaml"
_UNIT_YAML = "unit_costs.yaml"

#: Architectures the tables are defined for. Matches ``ArchProfile.name``.
ARCHITECTURES = ("wormhole", "blackhole")


@dataclass(frozen=True)
class CycleCost:
    """A cycle count as the source stated it.

    ``cycles`` is the number; ``bound`` says whether it was exact, a floor, a
    ceiling, an approximation, or the low end of a range (in which case
    ``max_cycles`` holds the high end).
    """

    cycles: float
    bound: str = "exact"
    max_cycles: float | None = None

    @classmethod
    def parse(cls, value):
        """Build from the YAML form: a bare number, or a mapping."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return cls(cycles=value)
        if not isinstance(value, dict):
            raise TypeError(f"cost must be a number or mapping, got {value!r}")
        bound = value.get("bound", "exact")
        if bound not in BOUNDS:
            raise ValueError(
                f"unknown bound {bound!r}, expected one of {sorted(BOUNDS)}"
            )
        if "cycles" not in value:
            raise ValueError(f"cost mapping needs a 'cycles' key: {value!r}")
        max_cycles = value.get("max")
        if bound == "range" and max_cycles is None:
            raise ValueError(f"bound 'range' needs a 'max': {value!r}")
        return cls(cycles=value["cycles"], bound=bound, max_cycles=max_cycles)

    def __str__(self):
        prefix = {
            "exact": "",
            "at_least": ">= ",
            "at_most": "<= ",
            "approximate": "~",
            "range": "",
        }[self.bound]
        if self.bound == "range":
            return f"{self.cycles}-{self.max_cycles} cycles"
        return f"{prefix}{self.cycles} cycles"


@dataclass(frozen=True)
class CostEntry:
    """One instruction's costs, plus where they came from."""

    name: str
    unit: str
    provenance: str
    latency: CycleCost | None = None
    occupancy: CycleCost | None = None
    throughput_ipc: float | None = None
    source: str | None = None
    derivation: str | None = None
    scales_with: str | None = None
    isa_doc_name: str | None = None
    note: str | None = None
    arch: str | None = None

    @property
    def is_sourced(self):
        """True when every number in this entry traces to a published one."""
        return self.provenance in SOURCED_PROVENANCE

    @property
    def has_costs(self):
        return (
            self.latency is not None
            or self.occupancy is not None
            or self.throughput_ipc is not None
        )

    @classmethod
    def parse(cls, name, unit, raw):
        return cls(
            name=name,
            unit=unit,
            provenance=raw.get("provenance"),
            latency=CycleCost.parse(raw.get("latency")),
            occupancy=CycleCost.parse(raw.get("occupancy")),
            throughput_ipc=raw.get("throughput_ipc"),
            source=raw.get("source"),
            derivation=raw.get("derivation"),
            scales_with=raw.get("scales_with"),
            isa_doc_name=raw.get("isa_doc_name"),
            note=raw.get("note"),
            arch=raw.get("arch"),
        )


@dataclass(frozen=True)
class UnitCosts:
    """One Tensix backend unit: its instructions and its unit-level extras."""

    name: str
    instructions: dict = field(default_factory=dict)
    isa_doc_name: str | None = None
    backend: str | None = None
    source: str | None = None
    note: str | None = None
    not_documented: tuple = ()
    extras: dict = field(default_factory=dict)

    def __getitem__(self, instruction):
        return self.instructions[instruction]

    def __contains__(self, instruction):
        return instruction in self.instructions


#: Keys an instruction entry may carry. Exported so the test can reject a
#: typo'd key: the loader would silently ignore one, and for a ``provenance``
#: or ``source`` field that is exactly the failure this schema exists to stop.
ENTRY_KEYS = frozenset(
    {
        "provenance",
        "latency",
        "occupancy",
        "throughput_ipc",
        "source",
        "derivation",
        "scales_with",
        "isa_doc_name",
        "note",
        "arch",
    }
)
_UNIT_META_KEYS = {
    "isa_doc_name",
    "backend",
    "source",
    "note",
    "instructions",
    "not_documented",
}


class CostTable:
    """The merged, arch-resolved view of both cost files.

    ``units`` holds the Tensix backend units keyed by ``ex_resource`` name;
    ``sections`` holds everything in ``unit_costs.yaml`` (``noc``, ``dram``,
    ``riscv``, ``mover``, ``l1``, ``clock``).
    """

    def __init__(self, arch, units, sections, documents):
        self.arch = arch
        self.units = units
        self.sections = sections
        self.documents = documents

    # -- Tensix -----------------------------------------------------------
    def unit(self, name):
        return self.units[name]

    def instruction(self, unit, name):
        """The :class:`CostEntry` for ``name``, or ``None`` if untabulated."""
        return self.units[unit].instructions.get(name) if unit in self.units else None

    def find(self, name):
        """Look an instruction up without knowing its unit."""
        for unit in self.units.values():
            if name in unit.instructions:
                return unit.instructions[name]
        return None

    # -- everything else --------------------------------------------------
    def section(self, name):
        return self.sections[name]

    # -- provenance -------------------------------------------------------
    def entries(self):
        for unit in self.units.values():
            yield from unit.instructions.values()

    def unsourced(self):
        """Every Tensix entry whose numbers are not traceable to a document.

        Returned rather than hidden: the honest gaps are part of the
        deliverable, and a caller that wants to refuse to model an opcode
        without a real cost needs to be able to enumerate them.
        """
        return tuple(e for e in self.entries() if not e.is_sourced)

    def weakest_provenance(self):
        return min(
            (PROVENANCE_RANK[e.provenance] for e in self.entries()),
            default=PROVENANCE_RANK["unknown"],
        )


def _deep_merge(base, override):
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_arch(raw, arch):
    """Resolve ``arch_overrides`` and drop entries for other architectures."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}, expected one of {ARCHITECTURES}")
    body = {k: v for k, v in raw.items() if k != "arch_overrides"}
    overrides = (raw.get("arch_overrides") or {}).get(arch) or {}
    return _deep_merge(body, overrides)


def _drop_foreign_arch(mapping, arch):
    """Remove entries marked as belonging to a different architecture."""
    return {
        name: raw
        for name, raw in mapping.items()
        if not isinstance(raw, dict) or raw.get("arch", arch) == arch
    }


def _load_tensix(arch):
    raw = load_yaml_cached(
        resources.files("tt_sim.pe.tensix").joinpath(_TENSIX_YAML),
        _TENSIX_YAML.removesuffix(".yaml"),
    )
    resolved = _apply_arch(raw, arch)
    units = {}
    for unit_name, unit_raw in (resolved.get("units") or {}).items():
        instructions = {
            name: CostEntry.parse(name, unit_name, entry)
            for name, entry in _drop_foreign_arch(
                unit_raw.get("instructions") or {}, arch
            ).items()
        }
        not_documented = (unit_raw.get("not_documented") or {}).get(
            "instructions"
        ) or ()
        units[unit_name] = UnitCosts(
            name=unit_name,
            instructions=instructions,
            isa_doc_name=unit_raw.get("isa_doc_name"),
            backend=unit_raw.get("backend"),
            source=unit_raw.get("source"),
            note=unit_raw.get("note"),
            not_documented=tuple(not_documented),
            extras={k: v for k, v in unit_raw.items() if k not in _UNIT_META_KEYS},
        )
    documents = {
        **(raw.get("documents") or {}),
        **(raw.get("vendor_documents") or {}),
    }
    return units, documents


def _load_units(arch):
    raw = load_yaml_cached(
        resources.files("tt_sim.perf").joinpath(_UNIT_YAML),
        _UNIT_YAML.removesuffix(".yaml"),
    )
    resolved = _apply_arch(raw, arch)
    reserved = {"schema_version", "documents", "vendor_documents"}
    sections = {k: v for k, v in resolved.items() if k not in reserved}
    documents = {
        **(raw.get("documents") or {}),
        **(raw.get("vendor_documents") or {}),
    }
    return sections, documents


_CACHE = {}


def load_costs(arch="wormhole"):
    """The cost table for ``arch``. Cached; the result is treated as read-only."""
    if arch not in _CACHE:
        units, tensix_docs = _load_tensix(arch)
        sections, unit_docs = _load_units(arch)
        _CACHE[arch] = CostTable(
            arch=arch,
            units=units,
            sections=sections,
            documents={**tensix_docs, **unit_docs},
        )
    return _CACHE[arch]


def raw_tables():
    """The unresolved YAML of both files, for tests and for tooling that needs
    to see ``arch_overrides`` rather than a merged view."""
    return (
        load_yaml_cached(
            resources.files("tt_sim.pe.tensix").joinpath(_TENSIX_YAML),
            _TENSIX_YAML.removesuffix(".yaml"),
        ),
        load_yaml_cached(
            resources.files("tt_sim.perf").joinpath(_UNIT_YAML),
            _UNIT_YAML.removesuffix(".yaml"),
        ),
    )
