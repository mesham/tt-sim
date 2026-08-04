"""The NoC per-hop latency model: hop counting, flight time, and opt-in.

``docs/plans/cost-model.md`` ranks this as the step that finally moves a
simulated total, because the in-tree workloads are not issue-limited — their
length is set by cross-core handshakes, and a handshake waits on the NoC. So
the properties worth pinning are the ones that decide *how long a packet is in
flight*, and they divide in two:

1. **Hop counting is topology**, and the easy thing to get wrong. Each NoC is a
   torus routed in one direction only, so distance is a forward modular
   distance and is **not symmetric** — a request and its response cover
   different numbers of hops. NoC 1 is the same formula in the mirrored
   coordinate space, which reverses it. Both are pinned below, on both
   architectures, because Wormhole and Blackhole differ in grid width and a
   model that hardcoded one would still pass every guard on the other.
2. **Flight time is the cost table**, reached only through
   ``tt_sim/perf/model.py`` so the file's three policies apply here as they do
   everywhere else.

And, above all of it, the opt-in: with ``TT_SIM_COST_MODEL`` unset a packet is
delivered on the next cycle exactly as it always was.

Runs standalone (``python3 -m tt_sim.network.noc_cost_model_test``) or under
pytest.
"""

import os
from contextlib import contextmanager

import pytest

from tt_sim.arch import BLACKHOLE_PROFILE, WORMHOLE_PROFILE
from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.noc_coords import WormholeNocCoords
from tt_sim.network.tt_noc import noc_hop_count
from tt_sim.perf.model import dram_cost_model, noc_cost_model

_L1_SRC = 0x20000
_L1_DST = 0x21000
_PAYLOAD = bytes(range(32))


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
# 1. Opt-in. Nothing below matters if this does not hold.
# ---------------------------------------------------------------------------


def test_the_noc_is_untimed_unless_the_env_var_says_otherwise():
    with _env(None):
        device = Wormhole()
        for tile in device.tile_directory.values():
            assert tile.noc0_router.noc_latency is None
            assert tile.noc1_router.noc_latency is None
            assert tile.noc0_router.delayed_arrivals == {}
            assert tile.noc0_router.next_arrival is None


@pytest.mark.parametrize("raw", ["0", "false", "off", ""])
def test_a_falsy_value_leaves_the_noc_untimed(raw):
    with _env(raw):
        assert noc_cost_model("wormhole") is None
        assert Wormhole().dram_tiles[0].noc0_router.noc_latency is None


def test_an_nui_built_without_an_architecture_never_opts_in():
    """``driver/simple`` and the NoC unit tests construct NUIs directly. They
    pass no arch, so they cannot be slowed down by an environment variable set
    for something else in the same process."""
    from tt_sim.network.tt_noc import NUI

    with _env("1"):
        assert NUI(0, 1, 1, None).noc_latency is None


# ---------------------------------------------------------------------------
# 2. Hop counting: topology, and the part that is easy to get backwards.
# ---------------------------------------------------------------------------


def test_hops_are_the_forward_distance_along_each_axis():
    assert noc_hop_count((1, 1), (4, 5), 10, 12) == 3 + 4
    assert noc_hop_count((1, 1), (1, 1), 10, 12) == 0
    # Same tile on one axis contributes nothing, not a wrap.
    assert noc_hop_count((1, 1), (1, 5), 10, 12) == 4


def test_hops_wrap_rather_than_turning_round():
    """A packet travels in the increasing direction only, so getting from the
    far edge back to the origin is one hop across the torus seam, not nine."""
    assert noc_hop_count((9, 0), (0, 0), 10, 12) == 1
    assert noc_hop_count((0, 0), (9, 0), 10, 12) == 9


def test_hops_are_not_symmetric_and_a_round_trip_costs_a_whole_ring():
    """The property a naive Manhattan distance would get wrong, and the one
    that makes the model checkable against tt-metal's measured end-to-end
    numbers: on a directional torus the outbound and return hop counts sum to
    the grid dimension on every axis where the endpoints differ, whatever the
    distance between them."""
    for a, b in [((1, 1), (4, 5)), ((2, 9), (8, 2)), ((0, 0), (9, 11))]:
        assert noc_hop_count(a, b, 10, 12) + noc_hop_count(b, a, 10, 12) == 10 + 12
    # Asymmetric in general (a pair equidistant both ways is the exception, not
    # the rule -- ``(2, 9) -> (8, 2)`` above is 11 hops in either direction).
    assert noc_hop_count((1, 1), (4, 5), 10, 12) == 7
    assert noc_hop_count((4, 5), (1, 1), 10, 12) == 15
    # Endpoints sharing an axis pay nothing for it, in either direction.
    assert (
        noc_hop_count((3, 1), (3, 7), 10, 12) + noc_hop_count((3, 7), (3, 1), 10, 12)
        == 12
    )


@pytest.mark.parametrize("grid_x,grid_y", [(10, 12), (17, 12)])
def test_noc1_is_noc0_reversed_because_both_ends_are_mirrored(grid_x, grid_y):
    """NoC 1's origin is the opposite corner, which tt-sim models by giving an
    ``NUI`` on NoC 1 the mirrored coord. Mirroring *both* endpoints negates dx
    and dy, which is exactly the reversal of routing direction — so the single
    formula gives NoC 1's opposite-direction hop count with no special case."""

    def mirror(c):
        return (grid_x - 1 - c[0], grid_y - 1 - c[1])

    for a, b in [((1, 2), (6, 9)), ((4, 0), (0, 11)), ((2, 2), (2, 7))]:
        noc0 = noc_hop_count(a, b, grid_x, grid_y)
        noc1 = noc_hop_count(mirror(a), mirror(b), grid_x, grid_y)
        assert noc1 == noc_hop_count(b, a, grid_x, grid_y)
        assert noc0 + noc1 == (grid_x if a[0] != b[0] else 0) + (
            grid_y if a[1] != b[1] else 0
        )


# ---------------------------------------------------------------------------
# 3. Flight time: the two constants, read through the model.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_flight_time_is_the_published_three_term_hop_model(arch):
    """~5 NIU->router, 9 per router->router hop, ~5 router->NIU. The Blackhole
    NoC page changes the flit width and not the latencies, so the two arches
    agree here — the arch difference that reaches a *flight time* is the torus
    size, and that comes from the profile."""
    with _env("1"):
        model = noc_cost_model(arch)
    assert model.endpoint_cycles == 10
    assert model.per_hop_cycles == 9
    assert model.flight_cycles(0) == 10
    assert model.flight_cycles(1) == 19
    assert model.flight_cycles(22) == 10 + 9 * 22


def test_a_modelled_flight_time_does_not_claim_to_be_exact():
    """Both endpoint terms are published as "~5 cycles". The table keeps the
    tilde as ``bound: approximate`` and the model refuses to report the result
    as counted — the estimator's central mistake this project's provenance
    convention exists to prevent."""
    with _env("1"):
        assert noc_cost_model("wormhole").is_exact is False


def test_congestion_is_not_modelled_and_says_so():
    """The ISA docs say congestion "can negatively impact latency" and give no
    number, so ``noc.congestion`` is ``provenance: unknown`` and a packet costs
    the same on an empty link and a saturated one. An honest gap, kept
    reachable so a report can name it rather than imply otherwise."""
    with _env("1"):
        assert noc_cost_model("blackhole").congestion_modelled is False


# ---------------------------------------------------------------------------
# 4. End to end: a real transfer over a real device takes the modelled time.
# ---------------------------------------------------------------------------


def _set_coord(initiator, which, coord):
    x, y = coord
    if isinstance(initiator.nui.noc_coord_strategy, WormholeNocCoords):
        setattr(initiator, f"{which}_addr_mid", (x << 4) | (y << 10))
    else:
        setattr(initiator, f"{which}_addr_hi", (y << 6) | x)


def _cycles_until_landed(device, tile, initiator, expected, budget):
    """Pump one cycle at a time until the read data appears in the tile's L1."""
    unified = tile.get_coord_pair()
    initiator.cmd_ctrl = 1
    initiator.initiate()
    for cycle in range(1, budget + 1):
        device.run(1)
        if bytes(device.read(unified, _L1_DST, len(expected))) == expected:
            return cycle
    return None


def _read_from_dram(device, tile, dram_coord, noc, dram_address=0x1000):
    """Issue one marked NoC read of DRAM into ``tile``'s L1, over ``noc``."""
    router = tile.noc0_router if noc == 0 else tile.noc1_router
    initiator = router.request_initiators[0]
    _set_coord(initiator, "target", dram_coord)
    initiator.target_addr_low = dram_address
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(_PAYLOAD)
    initiator.ctrl = 0
    return initiator


def _wormhole_worker_and_dram():
    device = Wormhole()
    tile = device.tensix_tiles[0]
    dram = device.dram_tiles[0]
    return device, tile, dram


def test_a_dram_read_lands_on_the_cycle_the_hop_model_predicts():
    """The whole point, end to end: request flight + response flight, each
    ``10 + 9 * hops`` in its *own* direction. The trailing ``+ 1`` is the
    counting loop's own off-by-one (it pumps before it looks), not a cost."""
    with _env("1"):
        device, tile, dram = _wormhole_worker_and_dram()
    src = tile.noc0_router
    dst = dram.noc0_router
    grid = (WORMHOLE_PROFILE.noc_grid_x, WORMHOLE_PROFILE.noc_grid_y)
    there = noc_hop_count(src.id_pair, dst.id_pair, *grid)
    back = noc_hop_count(dst.id_pair, src.id_pair, *grid)
    assert there + back == sum(grid)  # the worker and the DRAM share no axis

    device.write(dram.get_coord_pair(), 0x1000, _PAYLOAD)
    initiator = _read_from_dram(device, tile, dst.id_pair, noc=0)
    landed = _cycles_until_landed(device, tile, initiator, _PAYLOAD, budget=2000)
    # The DRAM endpoint's own service time is deliberately *not* part of the
    # flight (see ``tt_sim/device/tiles.py``), so it is added here rather than
    # folded into either leg — a round trip is what the two models sum to. The
    # channel term rides with it: one cycle at this 32-byte payload, which the
    # 24 B/cycle channel needs two cycles for and the 32 B/cycle link one.
    with _env("1"):
        model = dram_cost_model("wormhole")
    service = model.service_cycles
    channel = model.channel_excess_cycles(len(_PAYLOAD), len(_PAYLOAD) // 32)
    assert channel == 1
    assert landed == (10 + 9 * there) + service + channel + (10 + 9 * back) + 1


def test_the_untimed_noc_still_answers_in_two_cycles():
    """The same transfer with the model off: one cycle to service the request,
    one to service the response. This is the number every guard in the tree was
    recorded against."""
    with _env(None):
        device, tile, dram = _wormhole_worker_and_dram()
    device.write(dram.get_coord_pair(), 0x1000, _PAYLOAD)
    initiator = _read_from_dram(device, tile, dram.noc0_router.id_pair, noc=0)
    landed = _cycles_until_landed(device, tile, initiator, _PAYLOAD, budget=64)
    assert landed == 1 + 1 + 1


def test_the_two_nocs_charge_the_same_pair_different_amounts():
    """The asymmetry, observed on a device rather than argued from a formula:
    the same worker reading the same DRAM channel takes a different number of
    cycles on NoC 0 and NoC 1, because the two route in opposite directions."""
    with _env("1"):
        device = Blackhole()
        tile = device.tensix_tiles[0]
    dram = device.dram_tiles[0]
    grid = (BLACKHOLE_PROFILE.noc_grid_x, BLACKHOLE_PROFILE.noc_grid_y)
    src = tile.noc0_router
    dst = dram.noc0_router
    on_noc0 = noc_hop_count(src.id_pair, dst.id_pair, *grid)
    on_noc1 = noc_hop_count(dst.id_pair, src.id_pair, *grid)
    assert on_noc0 != on_noc1
    assert tile.noc0_router.flight_cycles_to(dst) == 10 + 9 * on_noc0
    # NoC 1's directory is keyed by the mirror coord; the NUI holds it too.
    assert tile.noc1_router.flight_cycles_to(dram.noc1_router) == 10 + 9 * on_noc1


def test_an_unmodelled_destination_is_delayed_like_a_real_one():
    """A coord with no tile behind it gets a ``NullEndpoint`` so the requester
    still completes. It has no latency model of its own, so the *requester*
    times the return flight from the coord it addressed — otherwise a kernel
    polling an unmodelled tile would look infinitely fast."""
    with _env("1"):
        device, tile, _ = _wormhole_worker_and_dram()
    unknown = (7, 7)
    # A null-routed read comes back zero-filled, so poison the landing zone
    # first -- otherwise "the zeros arrived" is indistinguishable from "L1 was
    # already zero" and the test passes on cycle one.
    device.write(tile.get_coord_pair(), _L1_DST, b"\xff" * 32)
    initiator = _read_from_dram(device, tile, unknown, noc=0)
    landed = _cycles_until_landed(device, tile, initiator, bytes(32), budget=2000)
    grid = (WORMHOLE_PROFILE.noc_grid_x, WORMHOLE_PROFILE.noc_grid_y)
    src = tile.noc0_router.id_pair
    assert (
        landed
        == (10 + 9 * noc_hop_count(src, unknown, *grid))
        + (10 + 9 * noc_hop_count(unknown, src, *grid))
        + 1
    )


# ---------------------------------------------------------------------------
# 5. Bandwidth: the term that makes a packet's *size* cost something.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch,flit_bytes", [("wormhole", 32), ("blackhole", 64)])
def test_bandwidth_is_the_flit_rate_and_the_two_recorded_figures_agree(
    arch, flit_bytes
):
    """One flit per cycle, 256 bits on Wormhole and 512 on Blackhole.

    The table records the same fact twice and independently — ``flit_bits``
    with a ``throughput_flits_per_cycle`` of 1, and a
    ``link_bandwidth_gb_per_s`` of 32 — and the two agree exactly: at the
    ``clock`` section's 1 GHz, 32 bytes per cycle *is* 32 GB/s. That is a
    property of the data rather than of this consumer, so the cross-check
    lives in ``tt_sim/perf/costs_test.py``, where reading the raw tables is
    what the file is for. (No Blackhole GB/s figure is published; its override
    marks the field ``unknown`` rather than scaling Wormhole's.)
    """
    with _env("1"):
        model = noc_cost_model(arch)
    assert model.flit_bytes == flit_bytes
    assert model.flits_per_cycle == 1
    assert model.bytes_per_cycle == flit_bytes
    assert model.bandwidth_modelled is True


def test_a_packet_pays_for_its_own_size_once_and_not_once_per_hop():
    """Wormhole routing: the head flit propagates and the tail follows it one
    cycle behind, rather than each router receiving the whole packet before
    forwarding it. So a packet's bytes cost the same whether it crosses one
    router or twenty, and the hop term is untouched by size."""
    with _env("1"):
        model = noc_cost_model("wormhole")
    assert model.tail_cycles(32) == 0  # one flit: nothing follows the head
    assert model.tail_cycles(64) == 1
    assert model.tail_cycles(8192) == 255
    assert model.serialisation_cycles(8192) == 256
    # A packet with no payload at all -- a read request, a write ACK -- is
    # still one flit, and still holds the injection port for its cycle.
    assert model.serialisation_cycles(0) == 1
    assert model.tail_cycles(0) == 0
    # The distance term does not know about any of this.
    assert model.flight_cycles(3) == 10 + 27


def test_a_semaphore_poke_no_longer_costs_what_a_tile_read_costs():
    """The headline, end to end and on a real device: two DRAM reads over the
    same path, 32 bytes and 2 KiB (a bf16 32x32 tile). Under the hop model
    alone they landed on the same cycle. They now differ by the tile's
    serialisation — ``2048 / 32 - 1`` cycles on Wormhole's 256-bit flit — plus
    the DRAM channel's *excess* over that link rate, since 2026-08-04. The two
    are added here rather than pinned as one number because they are two
    different queues and the second one is Wormhole-only; on a path with no
    DRAM in it the first term is the whole story."""
    small = _PAYLOAD
    large = bytes((i * 7) & 0xFF for i in range(2048))
    with _env("1"):
        device, tile, dram = _wormhole_worker_and_dram()
        device.write(dram.get_coord_pair(), 0x1000, small)
        initiator = _read_from_dram(device, tile, dram.noc0_router.id_pair, noc=0)
        small_cycles = _cycles_until_landed(device, tile, initiator, small, budget=4000)

        device, tile, dram = _wormhole_worker_and_dram()
        device.write(dram.get_coord_pair(), 0x2000, large)
        initiator = _read_from_dram(
            device, tile, dram.noc0_router.id_pair, noc=0, dram_address=0x2000
        )
        initiator.at_len_be = len(large)
        large_cycles = _cycles_until_landed(device, tile, initiator, large, budget=4000)

    with _env("1"):
        dram_model = dram_cost_model("wormhole")
    link = len(large) // 32 - 1
    channel = dram_model.channel_excess_cycles(
        len(large), len(large) // 32
    ) - dram_model.channel_excess_cycles(len(small), len(small) // 32)
    assert link == 63
    assert channel == 22 - 1
    assert large_cycles - small_cycles == link + channel


def test_the_injection_port_is_held_for_the_whole_packet():
    """The serialisation half, which is the half that makes bandwidth a
    *bandwidth* model rather than a size-dependent latency: while a big packet
    is being pushed onto the wire, everything behind it waits. It is also what
    stops a small packet overtaking a large one to the same destination, which
    no NoC does and which nothing downstream should have to cope with."""
    with _env("1"):
        device, tile, dram = _wormhole_worker_and_dram()
    src = tile.noc0_router
    flight = src.flight_cycles_to(dram.noc0_router)
    payload = bytes(4096)
    packet = _write_packet(src, payload)

    first = src._bandwidth_delay(packet, flight)
    second = src._bandwidth_delay(_write_packet(src, payload), flight)
    flits = len(payload) // 32
    # The first packet's tail lands ``flits - 1`` after its head; the second
    # cannot even start until the first has finished being injected.
    assert first == flight + flits - 1
    assert second == first + flits


def _write_packet(nui, payload):
    from tt_sim.network.tt_noc import NUI

    return NUI.NoCDataRequest(
        0x1000,
        NUI.NoCDataRequest.DataRequestAction.WRITE,
        len(payload),
        nui.id_pair,
        0,
        payload,
        reply_to=nui,
    )


def test_a_multicast_is_injected_once_however_wide_the_rectangle():
    """The place this model could easily over-charge, and the policy says not
    to. A multicast write leaves the NIU as **one** packet that the routers
    split across the destination rectangle; tt-sim models the fan-out as N
    unicasts, so charging each of them the full injection time would invent
    serialisation the hardware does not have. The port is claimed once."""
    with _env("1"):
        device, tile, dram = _wormhole_worker_and_dram()
    src = tile.noc0_router
    payload = bytes(1024)
    device.write(tile.get_coord_pair(), _L1_SRC, payload)

    initiator = src.request_initiators[0]
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(payload)
    x_start, y_start, x_end, y_end = 1, 1, 4, 3  # 4 x 3 = 12 destinations
    initiator.ret_addr_mid = (
        (x_end << 4) | (y_end << 10) | (x_start << 16) | (y_start << 22)
    )
    initiator.ctrl = 2 | (1 << 5)  # write, broadcast
    before = src._tx_free_cycle
    initiator.cmd_ctrl = 1
    initiator.initiate()
    # One packet's worth of injection time, not twelve.
    assert src._tx_free_cycle - before == len(payload) // 32


def test_size_costs_nothing_with_the_model_off():
    """The opt-in, for the bandwidth half specifically: an 8 KiB write and a
    4-byte one are still delivered on the same cycle with the switch unset."""
    with _env(None):
        device, tile, dram = _wormhole_worker_and_dram()
    big = bytes(8192)
    device.write(tile.get_coord_pair(), _L1_SRC, big)
    initiator = tile.noc0_router.request_initiators[0]
    _set_coord(initiator, "ret", dram.noc0_router.id_pair)
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = 0x3000
    initiator.at_len_be = len(big)
    initiator.ctrl = 2 | (1 << 4)
    initiator.cmd_ctrl = 1
    initiator.initiate()
    device.run(2)
    assert bytes(device.read(dram.get_coord_pair(), 0x3000, 16)) == big[:16]


# ---------------------------------------------------------------------------
# 6. Response ordering: the hazard variable latency creates, and bandwidth
#    makes likelier.
# ---------------------------------------------------------------------------


def test_a_response_that_overtakes_an_older_one_still_lands_where_it_belongs():
    """Two reads, one transaction ID, two destinations at different distances.

    This is the hazard the per-hop latency model created and the bandwidth
    model widens: the outstanding-request store used to be a per-trid FIFO, so
    the *first* response to arrive was handed the *first* request's L1
    destination address whether or not it was answering it. Nothing about a
    NoC guarantees the order — here a local L1 read (0 hops each way) is
    answered while a DRAM read issued before it is still in flight — and the
    failure mode was a silently wrong address, not a crash.

    Both payloads must land at their own address, and the NIU must *know* it
    happened: ``out_of_order_responses`` is the instrumentation that turns
    "this cannot happen today" into a number.
    """
    with _env("1"):
        device, tile, dram = _wormhole_worker_and_dram()
    unified = tile.get_coord_pair()
    far_payload = _PAYLOAD
    near_payload = bytes(range(64, 96))
    device.write(dram.get_coord_pair(), 0x1000, far_payload)
    device.write(unified, 0x22000, near_payload)
    device.write(unified, _L1_DST, b"\xff" * 32)
    device.write(unified, 0x23000, b"\xff" * 32)

    initiator = tile.noc0_router.request_initiators[0]
    # 1. The far read: worker -> DRAM and back, a couple of hundred cycles.
    _set_coord(initiator, "target", dram.noc0_router.id_pair)
    initiator.target_addr_low = 0x1000
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = 32
    initiator.ctrl = 0
    initiator.cmd_ctrl = 1
    initiator.initiate()
    # 2. The near read, same trid: this tile's own L1, 0 hops, ~20 cycles.
    _set_coord(initiator, "target", tile.noc0_router.id_pair)
    initiator.target_addr_low = 0x22000
    initiator.ret_addr_low = 0x23000
    initiator.at_len_be = 32
    initiator.ctrl = 0
    initiator.cmd_ctrl = 1
    initiator.initiate()

    device.run(1000)
    assert bytes(device.read(unified, 0x23000, 32)) == near_payload
    assert bytes(device.read(unified, _L1_DST, 32)) == far_payload
    assert tile.noc0_router.out_of_order_responses == 1


def test_a_response_no_request_accounts_for_is_a_loud_failure():
    """The other half of dropping the FIFO. Matching on the issue number means
    a response can fail to match — a duplicate, or one for a trid this NIU
    never used — and there is nothing sensible to return for it. tt-sim's
    established answer for a case it cannot model is to say so."""
    from tt_sim.network.tt_noc import NUI, NoCResponseError

    with _env("1"):
        device, tile, dram = _wormhole_worker_and_dram()
    nui = tile.noc0_router
    stray = NUI.NoCDataRequest(
        None,
        NUI.NoCDataRequest.DataRequestAction.RESPONSE_READ,
        32,
        dram.noc0_router.id_pair,
        3,
        bytes(32),
        seq=99,
    )
    nui.transmit(stray)
    with pytest.raises(NoCResponseError, match="matches no outstanding request"):
        device.run(4)


def test_a_tile_sleeps_through_a_packets_flight_rather_than_spinning():
    """A NIU with a packet in flight names the cycle it lands on, so the pump
    can stride over the flight instead of ticking the tile ~130 times for
    nothing. Without this the latency model would cost wall clock to buy
    simulated cycles."""
    with _env("1"):
        device, tile, dram = _wormhole_worker_and_dram()
    device.write(dram.get_coord_pair(), 0x1000, _PAYLOAD)
    initiator = _read_from_dram(device, tile, dram.noc0_router.id_pair, noc=0)
    initiator.cmd_ctrl = 1
    initiator.initiate()
    device.run(1)
    arrival = dram.noc0_router.next_arrival
    assert arrival is not None
    assert arrival > 1
    assert dram.next_wake_cycle(1) == arrival


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            marks = getattr(fn, "pytestmark", [])
            params = [m for m in marks if m.name == "parametrize"]
            if not params:
                fn()
                continue
            for case in params[0].args[1]:
                fn(*(case if isinstance(case, tuple) else (case,)))
    print("noc_cost_model_test OK")


if __name__ == "__main__":
    main()
