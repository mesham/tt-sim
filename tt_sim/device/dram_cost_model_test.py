"""DRAM access latency: the endpoint's own service time.

The fourth consumer of the cost tables, and the one whose *number* took more
argument than its mechanism. ``docs/plans/cost-model.md`` ranks DRAM latency as
the largest remaining uncosted term and records that no document publishes one;
what exists is tt-metal's measured end-to-end figures, and the entry this file
exercises is the difference of two of them — 358 (DRAM) − 259 (L1 remote read),
both the same transaction shape from the same agent, so the NoC round trip and
the issuing core's path cancel and 99 cycles of DRAM-versus-L1 device time is
what is left.

So the properties worth pinning divide in three:

1. **The opt-in**, as everywhere: with ``TT_SIM_COST_MODEL`` unset a DRAM tile
   answers on the next cycle exactly as it always did.
2. **The provenance**, because these are the file's only
   ``vendor_source_derived`` entries and the whole convention turns on a reader
   being able to tell them from published ones. Blackhole got *nothing* until
   2026-08-03; it now gets 529 − 403 = 126 from the same table by the same
   subtraction, and that the two arches share one arithmetic is pinned here
   rather than left to drift apart.
3. **Where the cost is charged** — at the endpoint, on the request, and not in
   the flight. Charging it in the flight would produce the same total on a
   round trip while being wrong about everything else: it would double-count
   against the derivation, and it would charge a response.

Runs standalone (``python3 -m tt_sim.device.dram_cost_model_test``) or under
pytest.
"""

import os
from contextlib import contextmanager

import pytest

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.device.blackhole import Blackhole
from tt_sim.device.tiles import DRAMEndpointNUI
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.noc_coords import WormholeNocCoords
from tt_sim.network.tt_noc import NUI, noc_hop_count
from tt_sim.perf.model import dram_cost_model

_L1_DST = 0x21000
_PAYLOAD = bytes(range(32))
_ENDPOINT_CYCLES = 10
_PER_HOP_CYCLES = 9


@contextmanager
def _env(value):
    previous = os.environ.get("TT_SIM_COST_MODEL")
    if value is None:
        os.environ.pop("TT_SIM_COST_MODEL", None)
    else:
        os.environ["TT_SIM_COST_MODEL"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TT_SIM_COST_MODEL", None)
        else:
            os.environ["TT_SIM_COST_MODEL"] = previous


# ---------------------------------------------------------------------------
# 1. Opt-in.
# ---------------------------------------------------------------------------


def test_dram_is_untimed_unless_the_env_var_says_otherwise():
    with _env(None):
        assert dram_cost_model("wormhole") is None
        for tile in Wormhole().dram_tiles:
            assert tile.noc0_router.service_cycles is None
            assert tile.noc1_router.service_cycles is None


@pytest.mark.parametrize("raw", ["0", "false", "off", ""])
def test_a_falsy_value_leaves_dram_untimed(raw):
    with _env(raw):
        assert dram_cost_model("wormhole") is None
        assert Wormhole().dram_tiles[0].noc0_router.service_cycles is None


def test_a_dram_tile_built_without_the_model_is_still_a_dram_endpoint():
    """The subclass is unconditional; only its number is conditional. That is
    what keeps the model-off path one attribute read rather than a second code
    path nobody exercises."""
    with _env(None):
        assert isinstance(Wormhole().dram_tiles[0].noc0_router, DRAMEndpointNUI)


# ---------------------------------------------------------------------------
# 2. The number, and what it is worth.
# ---------------------------------------------------------------------------


def test_the_wormhole_service_time_is_the_difference_of_two_measurements():
    """358 − 259: tt-metal's measured initial latency for a DRAM read minus the
    same measurement for an L1 remote read. Same transaction shape, same
    issuing agent, so the NoC round trip and the core's own path cancel.

    That the two figures still say 358 and 259 is checked where the raw table
    is legitimately readable — ``costs_test.test_the_dram_latency_is_exactly_
    its_own_derivation`` — so this side asserts only the charged number."""
    with _env("1"):
        assert dram_cost_model("wormhole").service_cycles == 358 - 259


def test_the_derived_latency_does_not_claim_to_be_exact_or_published():
    """Two separate honesty claims, both load-bearing. The provenance is
    ``vendor_source_derived``: arithmetic on vendor numbers, ranked below a
    published figure and above a guess. The bound is ``at_least``, because 99
    is (DRAM service − L1 service) and tt-sim charges an L1 endpoint nothing,
    so it is a floor under the absolute device latency."""
    with _env("1"):
        model = dram_cost_model("wormhole")
    assert model.provenance == "vendor_source_derived"
    assert model.bound == "at_least"
    assert model.is_exact is False


def test_blackhole_is_the_same_subtraction_on_the_same_table():
    """529 − 403, the same arithmetic on the same four constants as Wormhole's
    358 − 259. Pinned as arithmetic rather than as 126 for the same reason the
    Wormhole side is, and pinned as the *same* arithmetic because a Blackhole
    figure derived some other way would not be commensurable with Wormhole's.

    This was ``None`` until 2026-08-03, on two grounds that both dissolved: a
    1.2-vs-1.35 GHz clock conflict that turns out not to touch a measured cycle
    count, and an internal-consistency failure that turns out to be confined to
    the ``l1_local_cycles`` row — which is not one of the two rows subtracted
    here. See the entry's ``derivation`` in ``unit_costs.yaml``."""
    with _env("1"):
        model = dram_cost_model("blackhole")
        assert model.service_cycles == 529 - 403
        assert model.provenance == "vendor_source_derived"
        assert model.bound == "at_least"
        for tile in Blackhole().dram_tiles:
            assert tile.noc0_router.service_cycles == 529 - 403


def test_the_gaps_are_named_rather_than_implied():
    """ROADMAP §I asks for bank-conflict and refresh-window costs by name. No
    source quantifies either, and there is no DRAM bank model in tt-sim at all,
    so neither is charged — and the model says so rather than leaving a reader
    to infer that a "DRAM latency" covers them. Occupancy likewise: a second
    request is not queued behind the first."""
    with _env("1"):
        model = dram_cost_model("wormhole")
    assert model.bank_conflicts_modelled is False
    assert model.refresh_modelled is False
    assert model.occupancy_modelled is False


# ---------------------------------------------------------------------------
# 3. End to end: the cost lands at the endpoint, on the request.
# ---------------------------------------------------------------------------


def _set_coord(initiator, which, coord):
    x, y = coord
    if isinstance(initiator.nui.noc_coord_strategy, WormholeNocCoords):
        setattr(initiator, f"{which}_addr_mid", (x << 4) | (y << 10))
    else:
        setattr(initiator, f"{which}_addr_hi", (y << 6) | x)


def _read_from_dram(tile, dram_coord, dram_address=0x1000):
    initiator = tile.noc0_router.request_initiators[0]
    _set_coord(initiator, "target", dram_coord)
    initiator.target_addr_low = dram_address
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(_PAYLOAD)
    initiator.ctrl = 0
    return initiator


def _cycles_until_landed(device, tile, initiator, expected, budget):
    unified = tile.get_coord_pair()
    initiator.cmd_ctrl = 1
    initiator.initiate()
    for cycle in range(1, budget + 1):
        device.run(1)
        if bytes(device.read(unified, _L1_DST, len(expected))) == expected:
            return cycle
    return None


def _flight(src, dst):
    grid = (WORMHOLE_PROFILE.noc_grid_x, WORMHOLE_PROFILE.noc_grid_y)
    return _ENDPOINT_CYCLES + _PER_HOP_CYCLES * noc_hop_count(src, dst, *grid)


def test_a_dram_read_costs_the_flight_plus_the_devices_own_service_time():
    """The whole thing on a real device: out, serviced, back. The service term
    is added once and only on the outbound leg, which is what distinguishes
    this model from one that simply made the DRAM further away."""
    with _env("1"):
        device = Wormhole()
        tile = device.tensix_tiles[0]
        dram = device.dram_tiles[0]
        service = dram_cost_model("wormhole").service_cycles
    src = tile.noc0_router.id_pair
    dst = dram.noc0_router.id_pair
    device.write(dram.get_coord_pair(), 0x1000, _PAYLOAD)
    initiator = _read_from_dram(tile, dst)
    landed = _cycles_until_landed(device, tile, initiator, _PAYLOAD, budget=4000)
    # The trailing +1 is the counting loop's own off-by-one (it pumps before it
    # looks), the same one ``noc_cost_model_test`` carries.
    assert landed == _flight(src, dst) + service + _flight(dst, src) + 1


def test_the_untimed_dram_still_answers_in_two_cycles():
    """The number every guard in the tree was recorded against, unchanged."""
    with _env(None):
        device = Wormhole()
        tile = device.tensix_tiles[0]
        dram = device.dram_tiles[0]
    device.write(dram.get_coord_pair(), 0x1000, _PAYLOAD)
    initiator = _read_from_dram(tile, dram.noc0_router.id_pair)
    assert _cycles_until_landed(device, tile, initiator, _PAYLOAD, budget=64) == 3


def test_only_requests_are_serviced_not_responses():
    """A response arriving at a DRAM NIU would be charged nothing, because the
    device has nothing to do with it. Nothing in tt-sim sends one (a DRAM tile
    has no request initiator reachable from any core, so it is never a
    requester), which is exactly why the rule is asserted rather than assumed."""
    with _env("1"):
        dram = Wormhole().dram_tiles[0]
    nui = dram.noc0_router
    actions = NUI.NoCDataRequest.DataRequestAction
    for action in (actions.READ, actions.WRITE, actions.ATOMIC):
        assert action in DRAMEndpointNUI._SERVICED_ACTIONS
    for action in (actions.RESPONSE_READ, actions.ACK, actions.RESPONSE_ATOMIC):
        assert action not in DRAMEndpointNUI._SERVICED_ACTIONS
        response = NUI.NoCDataRequest(None, action, 4, nui.id_pair, 0, b"\0\0\0\0")
        nui.transmit(response, delay=None)
        assert nui.delayed_arrivals == {}
        assert nui.noc_new_requests_to_handle.pop() is response


def test_the_tile_sleeps_through_the_service_window():
    """The service time reuses the flight model's parking lot, so the DRAM tile
    names the cycle it will answer on and the pump strides over the wait rather
    than ticking the tile 99 times for nothing."""
    with _env("1"):
        device = Wormhole()
        tile = device.tensix_tiles[0]
        dram = device.dram_tiles[0]
        service = dram_cost_model("wormhole").service_cycles
    src = tile.noc0_router.id_pair
    dst = dram.noc0_router.id_pair
    device.write(dram.get_coord_pair(), 0x1000, _PAYLOAD)
    initiator = _read_from_dram(tile, dst)
    initiator.cmd_ctrl = 1
    initiator.initiate()
    device.run(1)
    arrival = dram.noc0_router.next_arrival
    assert arrival == _flight(src, dst) + service
    assert dram.next_wake_cycle(1) == arrival


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            marks = getattr(fn, "pytestmark", [])
            params = [m for m in marks if m.name == "parametrize"]
            if not params:
                fn()
                continue
            for mark in params:
                for value in mark.args[1]:
                    fn(value)
    print("dram cost model: all tests passed")


if __name__ == "__main__":
    main()
