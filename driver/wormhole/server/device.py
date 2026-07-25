"""Wrapper around tt-sim Wormhole — device + cycle pumping + reset tracking.

Cycle pumping rule (locked-in plan decision): after every read/write through
this Device, ``wormhole.run(cycles_per_poll)`` is invoked iff at least one
BRISC is out of reset. Mirrors the existing
``while not done: wormhole.run(100)`` pattern in wormhole_driver.py:51-57.
"""

import os

from tt_sim.device.tt_device import DeviceTileDiagnostics, Wormhole
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.pe.tensix.util import TensixCoprocessorDiagnostics

from .coords import TENSIX_COORD_MAP

# Map from individual env var → (group, field). Group "rv"/"noc" fields land
# on DeviceTileDiagnostics; group "co" fields land on TensixCoprocessorDiagnostics.
_DIAG_VARS = {
    "TT_SIM_DIAG_BRISC": ("rv", "brisc_diagnostics"),
    "TT_SIM_DIAG_NCRISC": ("rv", "ncrisc_diagnostics"),
    "TT_SIM_DIAG_TRISC0": ("rv", "trisc0_diagnostics"),
    "TT_SIM_DIAG_TRISC1": ("rv", "trisc1_diagnostics"),
    "TT_SIM_DIAG_TRISC2": ("rv", "trisc2_diagnostics"),
    "TT_SIM_DIAG_NOC0": ("noc", "noc0_diagnostics"),
    "TT_SIM_DIAG_NOC1": ("noc", "noc1_diagnostics"),
    "TT_SIM_DIAG_CO_ISSUED": ("co", "issued_instructions"),
    "TT_SIM_DIAG_CO_CONFIG": ("co", "configurations_set"),
    "TT_SIM_DIAG_CO_UNPACK": ("co", "unpacking"),
    "TT_SIM_DIAG_CO_PACK": ("co", "packing"),
    "TT_SIM_DIAG_CO_FPU": ("co", "fpu_calculations"),
    "TT_SIM_DIAG_CO_SFPU": ("co", "sfpu_calculations"),
    "TT_SIM_DIAG_CO_THCON": ("co", "thcon"),
}


def _truthy(val):
    return val is not None and val.strip().lower() in {"1", "true", "yes", "on"}


def diagnostics_from_env(env=None):
    """Build a DeviceTileDiagnostics from TT_SIM_DIAG_* env vars.

    Aggregates (TT_SIM_DIAG_ALL, _TRISC, _NOC, _CO) expand to their member
    flags first; an individual var present in env (even set to a falsy value)
    overrides whatever the aggregates produced. This lets users say
    ``TT_SIM_DIAG_ALL=1 TT_SIM_DIAG_NCRISC=0`` to mean "everything except NCRISC".
    """
    if env is None:
        env = os.environ

    flags = {name: False for name in _DIAG_VARS}

    if _truthy(env.get("TT_SIM_DIAG_ALL")):
        for name in flags:
            flags[name] = True
    if _truthy(env.get("TT_SIM_DIAG_TRISC")):
        for name in ("TT_SIM_DIAG_TRISC0", "TT_SIM_DIAG_TRISC1", "TT_SIM_DIAG_TRISC2"):
            flags[name] = True
    if _truthy(env.get("TT_SIM_DIAG_NOC")):
        for name in ("TT_SIM_DIAG_NOC0", "TT_SIM_DIAG_NOC1"):
            flags[name] = True
    if _truthy(env.get("TT_SIM_DIAG_CO")):
        for name, (group, _) in _DIAG_VARS.items():
            if group == "co":
                flags[name] = True

    for name in flags:
        if name in env:
            flags[name] = _truthy(env.get(name))

    rv_kwargs = {}
    co_kwargs = {}
    for name, (group, field) in _DIAG_VARS.items():
        if group == "co":
            co_kwargs[field] = flags[name]
        else:
            rv_kwargs[field] = flags[name]

    return DeviceTileDiagnostics(
        coprocessor_diagnostics=TensixCoprocessorDiagnostics(**co_kwargs),
        **rv_kwargs,
    )


def enabled_diagnostic_names(diagnostics):
    """Return a list of short names for diagnostics that are enabled (for logging)."""
    short = {
        "TT_SIM_DIAG_BRISC": "BRISC",
        "TT_SIM_DIAG_NCRISC": "NCRISC",
        "TT_SIM_DIAG_TRISC0": "TRISC0",
        "TT_SIM_DIAG_TRISC1": "TRISC1",
        "TT_SIM_DIAG_TRISC2": "TRISC2",
        "TT_SIM_DIAG_NOC0": "NOC0",
        "TT_SIM_DIAG_NOC1": "NOC1",
        "TT_SIM_DIAG_CO_ISSUED": "CO_ISSUED",
        "TT_SIM_DIAG_CO_CONFIG": "CO_CONFIG",
        "TT_SIM_DIAG_CO_UNPACK": "CO_UNPACK",
        "TT_SIM_DIAG_CO_PACK": "CO_PACK",
        "TT_SIM_DIAG_CO_FPU": "CO_FPU",
        "TT_SIM_DIAG_CO_SFPU": "CO_SFPU",
        "TT_SIM_DIAG_CO_THCON": "CO_THCON",
    }
    co = diagnostics.getTensixCoprocessorDiagnostics()
    on = []
    for name, (group, field) in _DIAG_VARS.items():
        target = co if group == "co" else diagnostics
        if getattr(target, field):
            on.append(short[name])
    return on


class Device:
    def __init__(self, *, cycles_per_poll: int = 100, diagnostics=None):
        self.wormhole = Wormhole(diagnostics or DeviceTileDiagnostics())
        self.cycles_per_poll = cycles_per_poll
        # unified_coord -> True if BRISC has been deasserted.
        self._brisc_running: dict[tuple[int, int], bool] = {}

    def ensure_tensix_tile(self, translated):
        """Lazily materialise the TensixTile addressed by a translated coord.

        Called on first access to a Tensix worker coord that isn't yet
        backed by a tt-sim tile. Builds the tile through
        ``Wormhole.add_tensix_tile`` (which registers it in both NoC
        directories under its canonical SoC-physical NoC 0 coord) and
        registers it for BRISC-reset tracking. Idempotent — returns the
        unified coord on repeat calls.
        """
        unified = TENSIX_COORD_MAP[translated]
        if unified not in self.wormhole.tile_directory:
            self.wormhole.add_tensix_tile(unified)
        self.register_tensix(unified)
        return unified

    def register_tensix(self, unified):
        self._brisc_running.setdefault(unified, False)

    def write(self, unified, addr, data):
        self.wormhole.write(unified, addr, data)
        self._maybe_pump()

    def read(self, unified, addr, size):
        result = bytes(self.wormhole.read(unified, addr, size))
        self._maybe_pump()
        return result

    def assert_reset(self, unified):
        self._brisc_running[unified] = False
        self.wormhole.assert_soft_reset(unified)

    def deassert_reset_brisc(self, unified):
        self.wormhole.deassert_soft_reset(unified, BabyRISCVCoreType.BRISC)
        # Clear stale CPU state before the deasserted core runs. Scope this to
        # the one tile: the global ``wormhole.reset()`` clobbers the PCs of
        # every other tile already running firmware, which corrupts the whole
        # grid once more than one worker is materialised.
        self.wormhole.reset_tile(unified)
        self._brisc_running[unified] = True
        self._maybe_pump()

    def _maybe_pump(self):
        if any(self._brisc_running.values()):
            self.wormhole.run(self.cycles_per_poll)
