"""The Blackhole half of the full-grid addressability invariant.

Same check as ``driver/wormhole/server/multi_tensix_test.py``'s third test, on
the architecture where the NoC 1 directory is *more* ambiguous: with all 140
workers built, mirror coords collide with canonical ones far more often than on
Wormhole's 80. Mirrors are registered authoritatively and canonicals only fill
what is left, so the invariant that must hold is the one real tt-metal kernels
depend on — a worker is reachable at its canonical coord on NoC 0 and at its
mirror on NoC 1 (``NOC_X``/``NOC_Y`` mirror on NoC 1).

Socket-free and about a second, so it lives in the ordinary suite.
"""

from tt_sim.bridge import Fabric, TensixCore

from .bh_device import make_device
from .coords import TENSIX_COORD_MAP


def test_every_worker_of_the_full_grid_resolves_to_itself_on_both_nocs():
    device = make_device()
    fabric = Fabric()
    for coord in sorted(TENSIX_COORD_MAP):
        device.ensure_tensix_tile(coord)
        fabric.register(coord, TensixCore(device, TENSIX_COORD_MAP[coord]))

    tt_device = device.tt_device
    grid_x, grid_y = tt_device.profile.noc_grid_x, tt_device.profile.noc_grid_y
    noc0, noc1 = tt_device.noc_0_directory, tt_device.noc_1_directory

    wrong_noc0 = [
        c for c in TENSIX_COORD_MAP if noc0.get(c) is None or noc0[c].id_pair != c
    ]
    assert wrong_noc0 == [], (
        f"NoC 0 misroutes {len(wrong_noc0)} workers: {wrong_noc0[:5]}"
    )

    wrong_noc1 = []
    for coord in TENSIX_COORD_MAP:
        mirror = (grid_x - 1 - coord[0], grid_y - 1 - coord[1])
        endpoint = noc1.get(mirror)
        if endpoint is None or endpoint.id_pair != coord:
            wrong_noc1.append((coord, mirror))
    assert wrong_noc1 == [], (
        f"NoC 1 misroutes {len(wrong_noc1)} workers by their mirror coord: "
        f"{wrong_noc1[:5]}"
    )
