"""Routes (core, address, ...) protocol operations to the right Core instance.

Unknown coordinates lazily allocate a NullCore — matches the observed behaviour
of the zero-stub against tt-metal init traffic.

TODO(future): multi-Tensix support. Today ``__main__.py`` registers exactly one
``TensixCore`` from ``TENSIX_COORD_MAP``. Adding more entries here is mechanical
but also requires extending ``Wormhole.__init__`` in
``tt_sim/device/tt_device.py`` to instantiate additional ``TensixTile``s —
currently it hard-codes one tile at unified (18, 18).
"""

from .cores import NullCore


class Fabric:
    def __init__(self):
        self.cores: dict[tuple[int, int], object] = {}

    def register(self, coord, core):
        self.cores[coord] = core

    def _core(self, coord):
        core = self.cores.get(coord)
        if core is None:
            core = NullCore(coord)
            self.cores[coord] = core
        return core

    def write(self, coord, addr, data):
        self._core(coord).write(addr, data)

    def read(self, coord, addr, size):
        return self._core(coord).read(addr, size)

    def assert_reset(self, coord):
        self._core(coord).assert_reset()

    def deassert_reset(self, coord):
        self._core(coord).deassert_reset()
