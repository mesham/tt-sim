"""Wormhole-specific construction of the shared bridge ``Device``.

Injects the Wormhole device factory and the Wormhole ``TENSIX_COORD_MAP`` into
the architecture-agnostic :class:`tt_sim.bridge.Device`. ``driver/blackhole``
has its own equivalent.
"""

from tt_sim.bridge import Device
from tt_sim.device.tt_device import DeviceTileDiagnostics
from tt_sim.device.wormhole import Wormhole

from .coords import TENSIX_COORD_MAP


def wormhole_factory(diagnostics):
    return Wormhole(diagnostics or DeviceTileDiagnostics())


def make_device(*, cycles_per_poll=100, diagnostics=None):
    return Device(
        wormhole_factory,
        TENSIX_COORD_MAP,
        cycles_per_poll=cycles_per_poll,
        diagnostics=diagnostics,
    )
