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

from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP, noc1_mirror

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
        # Reverse-lookup table for lazy alias install on demand.
        self._unified_to_physical = {u: p for p, u in TENSIX_COORD_MAP.items()}
        # Eagerly install aliases for the DRAM tiles and any Tensix tiles
        # Wormhole already built — additional Tensix tiles install their
        # aliases through ``ensure_tensix_tile`` when materialised lazily.
        for translated, unified in DRAM_COORD_MAP.items():
            self._install_alias(translated, unified)
        for unified in self.wormhole.tile_directory:
            translated = self._unified_to_physical.get(unified)
            if translated is not None:
                self._install_alias(translated, unified)

    def _install_alias(self, translated, unified):
        """Make tt-sim's NoC directory accept a translated coord as an alias.

        tt-sim's ``Wormhole`` only registers unified coords (16-25). But
        tt-metal-over-the-wire writes the DRAM bank → NoC-XY mapping table
        to L1 with each NoC's own logical coord — NoC 0 uses (0, 11) for
        DRAM channel 0; NoC 1 uses the mirror (9, 0) because its origin is
        the bottom-right tile. The kernel emits NoC traffic targeting those
        coords, which would otherwise miss the directory and abort.

        Both NoC 0 and NoC 1 share their directory by reference across all
        NUIs on that NoC, so a single insert per NoC suffices.
        """
        target_tile = self.wormhole.tile_directory[unified]
        for noc_idx in (0, 1):
            key = translated if noc_idx == 0 else noc1_mirror(translated)
            target_nui = target_tile.get_noc_nui(noc_idx)
            target_nui.noc_directory[key] = target_nui

    def ensure_tensix_tile(self, translated):
        """Lazily materialise the TensixTile addressed by a translated coord.

        Called by the fabric on first access to a Tensix worker coord that
        isn't yet backed by a tt-sim tile. Builds the tile through
        ``Wormhole.add_tensix_tile``, installs its translated-coord NoC
        aliases on both NoCs, and registers it for BRISC-reset tracking.
        Idempotent — returns the existing tile on repeat calls.
        """
        unified = TENSIX_COORD_MAP[translated]
        if unified not in self.wormhole.tile_directory:
            self.wormhole.add_tensix_tile(unified)
            self._install_alias(translated, unified)
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
        # Match wormhole_driver.py:44 — propagate reset to clear stale CPU state
        # before the deasserted core starts executing.
        self.wormhole.reset()
        self._brisc_running[unified] = True
        self._maybe_pump()

    def _maybe_pump(self):
        if any(self._brisc_running.values()):
            self.wormhole.run(self.cycles_per_poll)
