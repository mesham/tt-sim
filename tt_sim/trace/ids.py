"""Stable ID registry for simulator components.

Each architectural unit gets a canonical ID at sim init: ``(chip_id,
core_y, core_x, unit)`` where ``unit`` is a :class:`Unit` enum value.
The same ID appears on every event the unit publishes, enabling
cross-tool correlation. Dump as JSON alongside any trace output via
:meth:`IDRegistry.dump`.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tt_sim.trace.events import Unit


@dataclass(frozen=True, slots=True)
class UnitID:
    chip_id: int
    core_y: int
    core_x: int
    unit: str

    def as_tuple(self) -> tuple:
        return (self.chip_id, self.core_y, self.core_x, self.unit)


class IDRegistry:
    def __init__(self):
        self._ids: list[UnitID] = []
        self._index: dict[tuple, UnitID] = {}

    def register(
        self, chip_id: int, core_y: int, core_x: int, unit: Unit | str
    ) -> UnitID:
        unit_name = unit.value if isinstance(unit, Unit) else unit
        uid = UnitID(chip_id, core_y, core_x, unit_name)
        key = uid.as_tuple()
        existing = self._index.get(key)
        if existing is not None:
            return existing
        self._ids.append(uid)
        self._index[key] = uid
        return uid

    def lookup(
        self, chip_id: int, core_y: int, core_x: int, unit: Unit | str
    ) -> UnitID:
        unit_name = unit.value if isinstance(unit, Unit) else unit
        return self._index[(chip_id, core_y, core_x, unit_name)]

    def all_ids(self) -> list[UnitID]:
        return list(self._ids)

    def dump(self, path: Path | str):
        path = Path(path)
        path.write_text(json.dumps([asdict(u) for u in self._ids], indent=2))

    def reset(self):
        self._ids = []
        self._index = {}


_REGISTRY: IDRegistry | None = None


def get_registry() -> IDRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = IDRegistry()
    return _REGISTRY
