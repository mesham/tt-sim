"""Wrapper around tt-sim Wormhole — device + cycle pumping + reset tracking.

Cycle pumping rule (locked-in plan decision): after every read/write through
this Device, ``wormhole.run(cycles_per_poll)`` is invoked iff at least one
BRISC is out of reset. Mirrors the existing
``while not done: wormhole.run(100)`` pattern in wormhole_driver.py:51-57.
"""

from tt_sim.device.tt_device import DeviceTileDiagnostics, Wormhole
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType

from .coords import DRAM_COORD_MAP, TENSIX_COORD_MAP, noc1_mirror


class Device:
    def __init__(self, *, cycles_per_poll: int = 100, diagnostics=None):
        self.wormhole = Wormhole(diagnostics or DeviceTileDiagnostics())
        self.cycles_per_poll = cycles_per_poll
        # unified_coord -> True if BRISC has been deasserted.
        self._brisc_running: dict[tuple[int, int], bool] = {}
        self._install_noc_coord_aliases()

    def _install_noc_coord_aliases(self):
        """Make tt-sim's NoC directory accept translated coords as aliases.

        tt-sim's ``Wormhole`` only registers unified coords (16-25). But
        tt-metal-over-the-wire writes the DRAM bank → NoC-XY mapping table to
        L1 with each NoC's own logical coord — NoC 0 uses (0, 11) for DRAM
        channel 0; NoC 1 uses the mirror (9, 0) because its origin is the
        bottom-right tile. The kernel emits NoC traffic targeting those
        coords, which would otherwise miss the directory and abort.

        Both NoC 0 and NoC 1 share their directory by reference across all
        NUIs on that NoC, so mutating once is enough.
        """
        any_tile = next(iter(self.wormhole.tile_directory.values()))
        coord_map = {**TENSIX_COORD_MAP, **DRAM_COORD_MAP}
        for noc_idx in (0, 1):
            directory = any_tile.get_noc_nui(noc_idx).noc_directory
            for translated, unified in coord_map.items():
                key = translated if noc_idx == 0 else noc1_mirror(translated)
                target_nui = self.wormhole.tile_directory[unified].get_noc_nui(noc_idx)
                directory[key] = target_nui

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
