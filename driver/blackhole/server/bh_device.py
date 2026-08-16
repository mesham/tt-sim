"""Blackhole-specific construction of the shared bridge ``Device``.

Injects the Blackhole device factory and the Blackhole ``TENSIX_COORD_MAP`` into
the architecture-agnostic :class:`tt_sim.bridge.Device`.
"""

from tt_sim.bridge import Device
from tt_sim.device.blackhole import Blackhole

from .coords import TENSIX_COORD_MAP

# L1 offset of the launch message's ``enables`` bitmask on Blackhole. The
# mailbox base is 0x60 (blackhole ``dev_mem_map.h``) and ``launch[0]`` sits at
# L1 0x70; ``kernel_config_msg_t.enables`` is +0x78 within it, so 0xE8. A wire
# DEASSERT reads this to decide which subordinate RISCs (NCRISC/TRISC) to run
# alongside BRISC. (Single-program slow-dispatch always uses launch slot 0.)
LAUNCH_ENABLES_OFFSET = 0xE8


def blackhole_factory(diagnostics, noc_translation=False):
    # ``None`` means "all off" — Blackhole fans the flags out to every tile
    # exactly as Wormhole does, so TT_SIM_DIAG_* works on both.
    return Blackhole(diagnostics, noc_translation=noc_translation)


def make_device(*, cycles_per_poll=100, diagnostics=None, noc_translation=False):
    # ``noc_translation`` is decided once by the server, from the cluster
    # descriptor the tt-metal host read, and handed to the device and to the
    # convention guard from that one place — see ``server/__main__.py``.
    return Device(
        lambda d: blackhole_factory(d, noc_translation),
        TENSIX_COORD_MAP,
        cycles_per_poll=cycles_per_poll,
        diagnostics=diagnostics,
        launch_enables_offset=LAUNCH_ENABLES_OFFSET,
    )
