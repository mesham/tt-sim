"""Architecture profiles for tt-sim.

An :class:`~tt_sim.arch.profile.ArchProfile` carries every hardware constant
that differs between Tenstorrent architectures (Wormhole, Blackhole, ...). The
device/tile classes read these values from a profile rather than hardcoding
them, so a new architecture is (largely) a new profile plus per-arch strategies
rather than a fork. See ``docs/plans/blackhole-support.md``.
"""

from tt_sim.arch.profile import ArchProfile
from tt_sim.arch.wormhole import WORMHOLE_PROFILE

__all__ = ["ArchProfile", "WORMHOLE_PROFILE"]
