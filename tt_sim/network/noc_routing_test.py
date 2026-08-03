"""NoC response routing: a response must reach the endpoint that asked for it.

The two NoC directories are both ``{(x, y): NUI}`` dicts, but they are keyed in
*different coordinate spaces*: NoC 0 by the canonical SoC-physical coord, NoC 1
additionally by the mirrored ``(GRID_X-1-x, GRID_Y-1-y)`` coord that tt-metal's
bank-to-noc table emits (see ``TT_Device._register_tile_internals``). The two
spaces overlap, so on NoC 1 the same tuple can name two different tiles — and
whichever registration lands last wins.

Routing a *response* by the requester's coordinate therefore delivered it to
some other tile's NUI, which popped an outstanding-request FIFO it never
filled (``KeyError``) and killed the simulator server. These tests pin the
hazard (the directory really is ambiguous) and the property that makes it
harmless: responses route by endpoint identity, never by coordinate.

No tt-metal, no socket, no oracle — the cheapest guard available for this.
"""

import pytest

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.noc_coords import WormholeNocCoords
from tt_sim.network.tt_noc import NUI

_OUTSTANDING_ID_0 = NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0

# Physical NoC 0 coords of the 4x5 worker sub-block tt-metal selects with
# ``TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4`` — the smallest grid the live
# multi-core examples actually ran on, and the one that crashed.
_WH_4X5 = [(x, y) for x in (1, 2, 3, 4) for y in (1, 2, 3, 4, 5)]

# Two Blackhole workers that mirror onto each other: (16-1, 11-2) == (15, 9).
_BH_SWAPPED_PAIR = [(1, 2), (15, 9)]

_L1_SRC = 0x20000
_L1_DST = 0x21000


def _wormhole_with_4x5_grid():
    device = Wormhole()
    unified_of_physical = {
        Wormhole.physical_noc0_coord_from_unified_worker((ux, uy)): (ux, uy)
        for ux in range(18, 26)
        for uy in range(16, 26)
    }
    workers = {}
    for physical in _WH_4X5:
        unified = unified_of_physical[physical]
        if unified not in device.tile_directory:
            device.add_tensix_tile(unified)
        workers[physical] = device.tile_directory[unified]
    return device, workers


def _blackhole_with_swapped_pair():
    device = Blackhole()
    workers = {}
    for physical in _BH_SWAPPED_PAIR:
        if physical not in device.tile_directory:
            device.add_tensix_tile(physical)
        workers[physical] = device.tile_directory[physical]
    return device, workers


def _noc1_key(device, tile):
    """The NoC 1 directory key a kernel uses to address ``tile``."""
    source = getattr(tile, "noc1_endpoint_coord", None) or tile.noc0_router.id_pair
    return (
        device.profile.noc_grid_x - 1 - source[0],
        device.profile.noc_grid_y - 1 - source[1],
    )


def _set_coord(initiator, which, coord):
    """Write ``coord`` into the initiator's ``target``/``ret`` coord register.

    Wormhole packs it into the MID address register (X@4, Y@10); Blackhole has
    a dedicated HI register holding ``(Y << 6) | X``. Mirrors
    ``tt_sim.network.noc_coords``, which only reads them.
    """
    x, y = coord
    if isinstance(initiator.nui.noc_coord_strategy, WormholeNocCoords):
        setattr(initiator, f"{which}_addr_mid", (x << 4) | (y << 10))
    else:
        setattr(initiator, f"{which}_addr_hi", (y << 6) | x)


def _noc1_dram_roundtrip(device, tile, dram, dram_address, payload):
    """Write ``payload`` to DRAM over NoC 1 and read it back, both marked.

    Returns what came back into the worker's L1. Raises whatever the NoC
    raises: with responses routed by coordinate this blows up inside the
    *wrong* tile's ``clock_tick`` with ``KeyError``.
    """
    unified = tile.get_coord_pair()
    dram_coord = _noc1_key(device, dram)
    device.write(unified, _L1_SRC, payload)

    initiator = tile.noc1_router.request_initiators[0]
    _set_coord(initiator, "ret", dram_coord)
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = dram_address
    initiator.at_len_be = len(payload)
    initiator.ctrl = 2 | (1 << 4)  # mode 2 = write, resp marked
    initiator.cmd_ctrl = 1
    initiator.initiate()
    device.run(16)

    _set_coord(initiator, "target", dram_coord)
    initiator.target_addr_low = dram_address
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(payload)
    initiator.ctrl = 0  # mode 0 = read
    initiator.cmd_ctrl = 1
    initiator.initiate()
    device.run(16)

    return bytes(device.read(unified, _L1_DST, len(payload)))


def _assert_settled(nui):
    """Every request the NUI issued has been answered, exactly once."""
    for trid, fifo in nui.outstanding_noc_requests.items():
        assert fifo == [], f"NUI {nui.id_pair} still awaiting trid {trid}: {fifo}"
        assert nui.nui_counters[_OUTSTANDING_ID_0 + trid] == 0, (
            f"NUI {nui.id_pair} OUTSTANDING[{trid}] = "
            f"{nui.nui_counters[_OUTSTANDING_ID_0 + trid]}"
        )


def _assert_never_responded_to(nui):
    """A tile that issued nothing must never have been handed a response."""
    assert nui.outstanding_noc_requests == {}, (
        f"NUI {nui.id_pair} was sent a response it never asked for: "
        f"{nui.outstanding_noc_requests}"
    )
    assert nui.nui_counters[NUI.NUICounters.CounterNames.NIU_MST_WR_ACK_RECEIVED] == 0
    assert nui.nui_counters[NUI.NUICounters.CounterNames.NIU_MST_RD_RESP_RECEIVED] == 0


# ---------------------------------------------------------------------------
# The hazard: NoC 1's directory really does resolve some coords to a foreign
# tile. These two tests are characterisation — if the directories are ever
# unified into a single coordinate space they should be updated (or dropped)
# together, since the routing tests below would then be guarding a hazard that
# no longer exists.
# ---------------------------------------------------------------------------


def test_wormhole_noc1_directory_shadows_worker_coords():
    device, workers = _wormhole_with_4x5_grid()

    shadowed = {
        physical
        for physical, tile in workers.items()
        if device.noc_1_directory[physical] is not tile.noc1_router
    }
    # DRAM's worker-visible endpoints (5, 9) / (5, 8) / (5, 7) mirror onto the
    # worker column x = 4, and the mirror registration is authoritative.
    assert shadowed == {(4, 2), (4, 3), (4, 4)}
    assert device.noc_1_directory[(4, 2)].tile_kind == "D"

    # NoC 0 is keyed in one space only, so it is unambiguous.
    assert all(
        device.noc_0_directory[physical] is tile.noc0_router
        for physical, tile in workers.items()
    )


def test_blackhole_noc1_directory_swaps_a_mirrored_worker_pair():
    device, workers = _blackhole_with_swapped_pair()

    # Each worker's mirror is the *other* worker's canonical coord, so the two
    # NoC 1 entries end up transposed. Latent on Blackhole only because its
    # DRAM does not sit on a worker mirror; with the full 140-worker grid
    # materialised, 102 worker coords resolve to a foreign tile.
    assert device.noc_1_directory[(1, 2)] is workers[(15, 9)].noc1_router
    assert device.noc_1_directory[(15, 9)] is workers[(1, 2)].noc1_router
    assert all(
        device.noc_0_directory[physical] is tile.noc0_router
        for physical, tile in workers.items()
    )


# ---------------------------------------------------------------------------
# The property: responses are unaffected by any of that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, channel",
    [(_wormhole_with_4x5_grid, 3), (_blackhole_with_swapped_pair, 0)],
    ids=["wormhole-4x5", "blackhole-mirrored-pair"],
)
def test_noc1_responses_return_to_the_issuing_worker(build, channel):
    device, workers = build()
    dram = device.dram_tiles[channel]

    for index, (physical, tile) in enumerate(sorted(workers.items())):
        address = 0x1000 + index * 0x100
        payload = bytes([0xA0 + index, 0xB0, 0xC0, index]) * 4
        got = _noc1_dram_roundtrip(device, tile, dram, address, payload)
        assert got == payload, f"worker {physical} read back {got.hex()}"
        assert (
            bytes(device.read(dram.get_coord_pair(), address, len(payload))) == payload
        )
        _assert_settled(tile.noc1_router)

    # Nothing else on the fabric was handed one of those responses: the DRAM
    # tile that served them issued no requests of its own, and neither did any
    # worker's NoC 0 side.
    _assert_never_responded_to(dram.noc1_router)
    for tile in workers.values():
        _assert_never_responded_to(tile.noc0_router)


def test_a_request_needing_a_response_must_name_its_requester():
    """``reply_to`` is what makes the response direction unroutable by coord."""
    device, workers = _wormhole_with_4x5_grid()
    tile = workers[(4, 2)]
    request = NUI.NoCDataRequest(
        0x40,
        NUI.NoCDataRequest.DataRequestAction.WRITE,
        4,
        tile.noc1_router.id_pair,
        0,
        b"\xde\xad\xbe\xef",
    )
    tile.noc1_router.transmit(request)
    with pytest.raises(AssertionError, match="reply_to"):
        device.run(4)


#: The (device builder, DRAM channel) pairs the routing test is run over.
_ARCH_CASES = [(_wormhole_with_4x5_grid, 3), (_blackhole_with_swapped_pair, 0)]


def main():
    test_wormhole_noc1_directory_shadows_worker_coords()
    test_blackhole_noc1_directory_swaps_a_mirrored_worker_pair()
    for build, channel in _ARCH_CASES:
        test_noc1_responses_return_to_the_issuing_worker(build, channel)
    test_a_request_needing_a_response_must_name_its_requester()
    print(
        "noc_routing_test OK: NoC 1's directory is ambiguous on both arches "
        "(WH DRAM shadows the x=4 worker column, BH workers mirror onto each "
        "other), and every NoC 1 response still lands on its requester"
    )


if __name__ == "__main__":
    main()
