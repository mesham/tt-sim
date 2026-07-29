"""Blackhole-specific construction of the shared bridge ``Device``.

Injects the Blackhole device factory and the Blackhole ``TENSIX_COORD_MAP`` into
the architecture-agnostic :class:`tt_sim.bridge.Device`.
"""

from tt_sim.bridge import Device
from tt_sim.device.blackhole import Blackhole

from .coords import TENSIX_COORD_MAP


def blackhole_factory(diagnostics):
    # Blackhole does not model per-core diagnostics yet; accepted for
    # factory-signature parity with the Wormhole path.
    return Blackhole(diagnostics)


def make_device(*, cycles_per_poll=100, diagnostics=None):
    return Device(
        blackhole_factory,
        TENSIX_COORD_MAP,
        cycles_per_poll=cycles_per_poll,
        diagnostics=diagnostics,
    )
