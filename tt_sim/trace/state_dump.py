"""Device state-dump capture.

A point-in-time snapshot of the simulator: per-baby-core register
files, NoC counters, and any other state worth comparing across two
runs. Schema is intentionally narrow — adding fields is additive and
the JSON carries a ``schema_version`` so a future writer can refuse
unknown forms.

Used by:

- ``StateDumpWriter`` — captures dumps at lifecycle boundaries and
  writes them as JSON files for offline diffing.
- :mod:`tt_sim.trace.diff_state` — compares two dumps and reports
  the first divergence with context. Run as
  ``python3 -m tt_sim.trace.diff_state a.json b.json``.

The "diff this trace against another commit / branch" workflow falls
straight out: capture before, capture after, diff. Catches regressions
that move L1 contents or perturb counter trajectories without changing
visible kernel results.
"""

import json
from pathlib import Path
from typing import Any

from tt_sim.trace.bus import EventBus, get_bus
from tt_sim.trace.events import EventCategory, LifecycleEvent

SCHEMA_VERSION = 1


def dump_device_state(wormhole, kind: str = "snapshot") -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of relevant device state.

    Pokes around the Wormhole device's known structure. Today: per
    baby-core register files (32 GPRs + PC) and per-NUI counter sets.
    L1 / DRAM dumps are intentionally not included by default —
    serialising megabytes of RAM per checkpoint isn't useful without
    a query layer.
    """
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "cores": {},
        "noc_counters": {},
    }

    # Tensix-tile baby cores. The first Tensix tile is the only one
    # today (see ROADMAP §A multi-Tensix); iterate defensively in case
    # that changes.
    for tensix_tile in getattr(wormhole, "tensix_tiles", []) or []:
        coord = f"{tensix_tile.coord_x}_{tensix_tile.coord_y}"
        for attr in ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2"):
            core = getattr(tensix_tile, attr, None)
            if core is None:
                continue
            rf = core.register_file
            # 32 GPRs + PC (index 32)
            regs = []
            for idx in range(32):
                regs.append(int.from_bytes(rf.registers[idx].read(), "little"))
            pc = int.from_bytes(rf.registers[32].read(), "little")
            state["cores"][f"{coord}_{attr.upper()}"] = {
                "gpr": regs,
                "pc": pc,
            }

        # NoC counters live on each NUI; index by (coord, noc_number).
        for router_attr in ("noc0_router", "noc1_router"):
            router = getattr(tensix_tile, router_attr, None)
            if router is None:
                continue
            counters = getattr(router, "nui_counters", None)
            if counters is None or not hasattr(counters, "counters"):
                continue
            state["noc_counters"][f"{coord}_{router_attr}"] = list(counters.counters)

    return state


class StateDumpWriter:
    """Captures device state at every ``LifecycleEvent``.

    Writes one JSON file per dump: ``<dir>/<kind>_<seq>.json``. The
    directory is created if missing. Holding a reference to the
    Wormhole device is required so the writer can poll its state when
    the lifecycle fires (events alone don't carry the device).
    """

    def __init__(
        self,
        directory: Path | str,
        wormhole,
        bus: EventBus | None = None,
    ):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._wormhole = wormhole
        self._seq = 0
        self._bus = bus if bus is not None else get_bus()
        self._bus.subscribe(EventCategory.LIFECYCLE, self._on_lifecycle)

    def _on_lifecycle(self, event: LifecycleEvent):
        state = dump_device_state(self._wormhole, kind=event.kind)
        path = self._dir / f"{event.kind}_{self._seq:04d}.json"
        path.write_text(json.dumps(state, indent=2))
        self._seq += 1

    def close(self):
        # No buffering — every event flushes already. Method present
        # for API symmetry with other writers.
        pass
