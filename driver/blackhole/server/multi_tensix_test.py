"""The Blackhole half of the full-grid addressability invariant.

Same question as ``driver/wormhole/server/multi_tensix_test.py``'s third test —
"with every worker built, can each one still be addressed?" — but the **answer
is a different coordinate**, and that difference is the whole point of this
file.

On Wormhole a worker is reachable at its canonical coord on NoC 0 and at its
``(GRID-1-x, GRID-1-y)`` mirror on NoC 1, because tt-metal's
``l1_bank_to_noc_xy`` really does mirror its NoC 1 half there. On **Blackhole it
does not**: ``RiscFirmwareInitializer::virtual_noc0_coordinate`` early-outs on
``|| cluster_.arch() == ARCH::BLACKHOLE``, so the table's two halves are
byte-identical and every worker coord on the Blackhole wire is canonical, on
both NoCs. tt-sim registers no Tensix mirror alias here at all
(``ArchProfile.noc1_tensix_mirror_aliases``), so the invariant to check is
canonical-on-both-NoCs.

**This test used to assert the mirror form on Blackhole too**, and passed —
because the aliases it was asserting existed, and in existing displaced 96 of
the 140 workers from their own canonical NoC 1 cell. Asserting the mirror here
was asserting the bug. Both directions are pinned below so that reintroducing
the aliases fails, and so does deleting Wormhole's.

Socket-free and about a second, so it lives in the ordinary suite.
"""

from tt_sim.bridge import Fabric, TensixCore

from .bh_device import make_device
from .coords import TENSIX_COORD_MAP

#: Workers whose canonical NoC 1 cell is taken by a DRAM tile's mirror. These
#: are the six shadows that legitimately survive: ``dram_bank_to_noc_xy`` *is*
#: mirrored on NoC 1 on Blackhole, so DRAM's aliases are load-bearing and there
#: is no lookup order that serves both. Fixed only by NoC coordinate
#: translation — see ``docs/plans/noc1-translation-feasibility.md``.
DRAM_SHADOWED = {(7, 4), (7, 7), (7, 10), (16, 4), (16, 7), (16, 10)}


def _full_grid_device():
    device = make_device()
    fabric = Fabric()
    for coord in sorted(TENSIX_COORD_MAP):
        device.ensure_tensix_tile(coord)
        fabric.register(coord, TensixCore(device, TENSIX_COORD_MAP[coord]))
    return device.tt_device


def test_every_worker_of_the_full_grid_resolves_to_itself_on_both_nocs():
    tt_device = _full_grid_device()
    noc0, noc1 = tt_device.noc_0_directory, tt_device.noc_1_directory

    wrong_noc0 = [
        c for c in TENSIX_COORD_MAP if noc0.get(c) is None or noc0[c].id_pair != c
    ]
    assert wrong_noc0 == [], (
        f"NoC 0 misroutes {len(wrong_noc0)} workers: {wrong_noc0[:5]}"
    )

    # NoC 1 is addressed by the *canonical* coord on Blackhole, not the mirror.
    wrong_noc1 = []
    for coord in TENSIX_COORD_MAP:
        endpoint = noc1.get(coord)
        if endpoint is None or endpoint.id_pair != coord:
            wrong_noc1.append(coord)
    assert set(wrong_noc1) == DRAM_SHADOWED, (
        f"NoC 1 misroutes {len(wrong_noc1)} workers by their canonical coord, "
        f"expected only the {len(DRAM_SHADOWED)} DRAM-shadowed ones: "
        f"{sorted(set(wrong_noc1) ^ DRAM_SHADOWED)[:5]}"
    )


def test_no_worker_claims_a_mirror_cell_on_noc1():
    """The mechanism behind the test above, pinned separately.

    A Tensix mirror alias on Blackhole can only evict the worker that owns the
    cell canonically — nothing addresses a Blackhole worker that way. Whoever
    holds a mirror cell here, it is never the worker it mirrors.
    """
    tt_device = _full_grid_device()
    grid_x, grid_y = tt_device.profile.noc_grid_x, tt_device.profile.noc_grid_y

    for coord in TENSIX_COORD_MAP:
        mirror = (grid_x - 1 - coord[0], grid_y - 1 - coord[1])
        endpoint = tt_device.noc_1_directory.get(mirror)
        assert endpoint is None or endpoint.id_pair != coord, (
            f"worker {coord} is reachable at its NoC 1 mirror {mirror}; "
            f"Blackhole registers no Tensix mirror aliases"
        )
