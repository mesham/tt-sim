"""Wormhole-specific construction of the shared bridge ``Device``.

Injects the Wormhole device factory and the Wormhole ``TENSIX_COORD_MAP`` into
the architecture-agnostic :class:`tt_sim.bridge.Device`. ``driver/blackhole``
has its own equivalent.
"""

from tt_sim.bridge import Device
from tt_sim.device.tt_device import DeviceTileDiagnostics
from tt_sim.device.wormhole import Wormhole

from .coords import TENSIX_COORD_MAP


def wormhole_factory(diagnostics, noc_translation=False):
    return Wormhole(
        diagnostics or DeviceTileDiagnostics(), noc_translation=noc_translation
    )


def make_device(*, cycles_per_poll=100, diagnostics=None, noc_translation=False):
    # ``noc_translation`` is decided once by the server, from the cluster
    # descriptor the tt-metal host read, and handed to the device and to the
    # convention guard from that one place — see ``server/__main__.py``.
    return Device(
        lambda d: wormhole_factory(d, noc_translation),
        TENSIX_COORD_MAP,
        cycles_per_poll=cycles_per_poll,
        diagnostics=diagnostics,
    )
