"""Routes (core, address, ...) protocol operations to the right Core instance.

Unknown coordinates lazily allocate a ``NullCore`` — matches the observed
behaviour of the zero-stub against tt-metal init traffic for eth / pcie /
arc / router-only endpoints and for worker coords the user didn't list in
``TT_SIM_TENSIX_COORDS`` (see ``__main__.py``).
"""

from collections.abc import Callable

from .cores import NullCore


class Fabric:
    def __init__(self):
        self.cores: dict[tuple[int, int], object] = {}
        # Invoked the first time a NullCore-backed coord receives a write
        # targeting L1 above NullCore.USER_DATA_ADDR_THRESHOLD — i.e. past
        # the kernel firmware / init scratch region, which is a strong
        # signal the host is treating this coord as a real worker. The
        # callback is wired by ``__main__.py`` to warn about coords the
        # user forgot to list in TT_SIM_TENSIX_COORDS. Without this the
        # silent NullCore zero-fill becomes a "shards 1-N return zeros"
        # debugging adventure.
        self.unmapped_callback: Callable[[tuple[int, int]], None] | None = None

    def register(self, coord, core):
        self.cores[coord] = core

    def _core(self, coord):
        core = self.cores.get(coord)
        if core is None:
            core = NullCore(coord, on_user_data_write=self.unmapped_callback)
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
