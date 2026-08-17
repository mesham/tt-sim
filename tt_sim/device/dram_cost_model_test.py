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

Since 2026-08-04 the endpoint charges a second, unrelated thing: the DRAM
channel's bandwidth, which is a *rate* rather than a latency and which
``dram.bandwidth`` sourced at ``isa_doc`` long before anything consumed it. The
properties worth pinning there are the derivation (24 GB/s at 1 GHz is 24 B/cycle
and nothing else), the shape (the **excess** over what the NoC link already
charged, so two queues in series cost the slower and not the sum), and the
refusal — the deep-merged overrides mean declining another arch's number takes
an explicit check rather than an absent key.

Since 2026-08-12 that rate is **per direction**, because Blackhole joined it by
a different route: no page and so no GB/s to convert, but one division on two
of ``tm_noc_latencies``' measured cycle counts gives 47.0805 B/cycle for reads,
which is `vendor_source_derived` — the rank asks for vendor arithmetic, not for
a document. Its *write* direction is deliberately unsourced, so what this file
pins there is a refusal that has to survive a deep merge: a Blackhole write
must read ``None``, never Wormhole's 24.

Since 2026-08-17 the **latency** is per direction too, on Blackhole only, and
that is the change this file's newest two tests pin. 126 and 99 are both *read*
figures by construction — each is one measured remote read minus another — so
charging them to a write asserted that a write is answered when the array has
been written. The same vendor campaign says a Blackhole write is answered ~104
cycles sooner, and its own DRAM-minus-L1 subtraction (22 cycles, flat over
64 B..8 KiB) is what that direction is charged now. Wormhole's rows resolve no
such asymmetry, so it declines and both directions keep the one 99 — which
makes the two tests a pair rather than one test and one restatement.

Since 2026-08-09 that same rate is spent a *second* way and the endpoint has a
queue: the channel carries one transfer at a time, so a request arriving while
it is busy waits ``ceil(N / 24)`` for it. What is pinned here is that it needed
no new number, that it is inert for a lone request (hence no rung-2 row and no
single-transaction prediction can move), that it bites on a stream — four
concurrent 4 KiB reads used to come back at the NoC link's 32 B/cycle and now
come back at the channel's 24 — and that the *device* behind the bus is still
not a queue, because nothing publishes its re-issue interval.

Runs standalone (``python3 -m tt_sim.device.dram_cost_model_test``) or under
pytest.
"""

import math
import os
from contextlib import contextmanager

import pytest

from tt_sim.arch import BLACKHOLE_PROFILE, WORMHOLE_PROFILE
from tt_sim.device.blackhole import Blackhole
from tt_sim.device.tiles import DramChannels, DRAMEndpointNUI
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


def test_the_blackhole_service_time_is_per_action_and_wormholes_is_not():
    """The 2026-08-17 split, pinned as the arithmetic it is on both arches.

    126 is a *read* figure — ``dram_cycles - l1_remote_read_cycles``, two
    remote reads — so charging it to a write asserts that a write is answered
    when the array has been written. tt-metal's ``tm_noc_latencies`` says it is
    not: the same DRAM-minus-L1 subtraction in the ONE_TO_ONE direction is
    22 / 29 / 28 / 22 / 21 / 19 / 22 / 13 over 64 B..8 KiB on Blackhole, mean
    22.0, so a Blackhole DRAM write is charged 22 and a read is still charged
    126.

    Wormhole is the control and its answer is a refusal: the identical
    subtraction on its rows returns no clean asymmetry and if anything a
    *dearer* write, so its entry declines the split and both directions read
    the same 99. That the refusal survives the deep merge is the same property
    the channel-rate test below pins — a Blackhole figure must not become a
    Wormhole one by being present in the tree."""
    with _env("1"):
        blackhole = dram_cost_model("blackhole")
        wormhole = dram_cost_model("wormhole")
    assert blackhole.service_cycles == 529 - 403
    assert blackhole.service_cycles_write == 22
    assert blackhole.service_cycles_for(is_write=True) == 22
    assert blackhole.service_cycles_for() == 126
    assert blackhole.service_split_by_action is True
    # Its own provenance and its own bound, because it is its own subtraction
    # on its own campaign rather than a qualifier on the read figure.
    assert blackhole.write_provenance == "vendor_source_derived"
    assert blackhole.write_bound == "at_least"

    assert wormhole.service_cycles == 358 - 259
    assert wormhole.service_cycles_write == 99
    assert wormhole.service_cycles_for(is_write=True) == 99
    assert wormhole.service_split_by_action is False
    assert wormhole.write_provenance is None


def test_a_blackhole_dram_write_waits_the_write_service_and_a_read_the_read_one():
    """The consumer end of the same split, on a real device: the endpoint holds
    a write for 22 cycles and a read for 126, and the difference is exactly the
    104 cycles the two table entries differ by. Wormhole's endpoint holds both
    for the same 99, so the arch that declines the split is charged as it was
    before it existed."""
    expected = {"blackhole": (126, 22), "wormhole": (99, 99)}
    actions = NUI.NoCDataRequest.DataRequestAction
    for arch, device_class in (("blackhole", Blackhole), ("wormhole", Wormhole)):
        read, write = expected[arch]
        with _env("1"):
            for tile in device_class().dram_tiles:
                for nui in (tile.noc0_router, tile.noc1_router):
                    assert nui.service_cycles == read
                    assert nui.service_cycles_write == write
        for action, service in (
            (actions.READ, read),
            (actions.WRITE, write),
            # An ATOMIC reads the array before it modifies it, so it takes the
            # read figure — the same call ``_is_write`` makes for the rates.
            (actions.ATOMIC, read),
        ):
            # A device of its own per action: the channel is a queue since
            # 2026-08-09, so three requests through one endpoint would measure
            # the second and third waiting for the first rather than their own
            # service times.
            with _env("1"):
                nui = device_class().dram_tiles[0].noc0_router
            request = NUI.NoCDataRequest(None, action, 4, nui.id_pair, 0, b"\0\0\0\0")
            nui.transmit(request, delay=None)
            assert sorted(nui.delayed_arrivals) == [service], (arch, action)


def test_the_channel_rate_is_the_published_bandwidth_at_the_published_clock():
    """24 GB/s per channel (isa_doc) at 1 GHz (isa_doc) is 24 bytes per cycle,
    and that is the whole derivation — a unit conversion on two published
    numbers, neither of which was chosen by looking at a measurement. That the
    two inputs still say 24 and 1000 is checked where the raw table is
    legitimately readable (``costs_test.test_the_dram_channel_rate_is_exactly_
    its_own_derivation``), so this side asserts only the charged number."""
    with _env("1"):
        model = dram_cost_model("wormhole")
    # One published figure, stated by the page for reads and writes alike, so
    # both directions read it.
    assert model.channel_bytes_per_cycle() == 24
    assert model.channel_bytes_per_cycle(is_write=True) == 24


def test_the_blackhole_channel_rate_is_derived_for_reads_and_refused_for_writes():
    """BlackholeA0 has no DRAMTile directory in the ISA docs, so there is no
    GB/s to convert — but ``vendor_source_derived`` asks for arithmetic on
    vendor numbers, not for a page, so the read rate comes instead from one
    division on two of ``tm_noc_latencies``' measured cycle counts (4096 bytes
    of extra transfer costing 87 extra cycles). The write direction is refused:
    its own secant lands within 2.5 % of the same arch's L1 rows, so it
    resolves no endpoint limit, and charging it would deepen the row already
    pinned in ``KNOWN_OVER_CHARGED``.

    The check that matters is the same one it always was — that the *deep
    merge* does not launder Wormhole's 24 in. The number is still physically
    present under Blackhole's override (``costs_test`` asserts exactly that),
    and the rule that keeps it out is that a directional key replaces the flat
    one rather than falling back to it. So a Blackhole write must read
    ``None``, not 24."""
    with _env("1"):
        model = dram_cost_model("blackhole")
    rate = model.channel_bytes_per_cycle()
    assert 4096 / 87 <= rate < 4096 / 87 + 1e-3  # rounded up: the low end
    # And the charge reproduces the two vendor points it came from: 4096 more
    # bytes cost 87 more cycles, which is the derivation run backwards.
    assert model.channel_serialisation_cycles(8192) == 174
    assert (
        model.channel_serialisation_cycles(8192)
        - model.channel_serialisation_cycles(4096)
        == 665 - 578
    )
    # The excess over a 64 B/cycle link, which is what actually gets charged.
    assert model.channel_excess_cycles(8192, 128) == 174 - 128
    # Writes: nothing, and emphatically not Wormhole's 24.
    assert model.channel_bytes_per_cycle(is_write=True) is None
    assert model.channel_serialisation_cycles(8192, is_write=True) is None
    assert model.channel_excess_cycles(8192, 128, is_write=True) == 0


def test_the_channel_is_charged_as_an_excess_over_the_link_not_as_a_second_bill():
    """The whole shape of the term. Two queues in series at two rates means the
    transfer runs at the slower one, so the endpoint charges
    ``ceil(N/24) - ceil(N/32)`` and the round trip's size cost comes to
    ``ceil(N/24)`` once — not ``ceil(N/24) + ceil(N/32)``, which is the
    double-billing that (wrongly) kept this section unconsumed until rung 2."""
    with _env("1"):
        model = dram_cost_model("wormhole")
    for size in (32, 64, 256, 1024, 4096, 8192):
        link = math.ceil(size / 32)
        assert model.channel_serialisation_cycles(size) == math.ceil(size / 24)
        assert model.channel_excess_cycles(size, link) == math.ceil(size / 24) - link
    # Never negative: an arch whose link were the slower gets nothing from
    # here rather than a refund.
    assert model.channel_excess_cycles(8192, 1000) == 0
    # And nothing at all when the NoC bandwidth term is not modelled, because
    # then there is nothing for this to be the excess *of*.
    assert model.channel_excess_cycles(8192, None) == 0


def test_the_gaps_are_named_rather_than_implied():
    """ROADMAP §I asks for bank-conflict and refresh-window costs by name. No
    source quantifies either, and there is no DRAM bank model in tt-sim at all,
    so neither is charged — and the model says so rather than leaving a reader
    to infer that a "DRAM latency" covers them.

    Occupancy is the one that changed on 2026-08-09, and it changed by *half*:
    the channel is now a queue, the device behind it is still not one. Both
    halves are flagged, because ``occupancy_modelled`` alone would read as a
    claim about the endpoint rather than about its bus."""
    with _env("1"):
        model = dram_cost_model("wormhole")
    assert model.bank_conflicts_modelled is False
    assert model.refresh_modelled is False
    assert model.occupancy_modelled is True
    assert model.device_occupancy_modelled is False


def test_occupancy_is_modelled_exactly_where_the_channel_rate_is_sourced():
    """The flag is not a global switch, it is a per-arch and now per-direction
    consequence: both arches source a read rate, so both hold a read off, and
    Blackhole sources no write rate, so a Blackhole write queues behind
    nothing. ``device_occupancy_modelled`` stays False everywhere — the array
    behind the bus is still not a queue on either arch."""
    with _env("1"):
        for arch in ("wormhole", "blackhole"):
            model = dram_cost_model(arch)
            assert model.occupancy_modelled is True
            assert model.device_occupancy_modelled is False
        assert dram_cost_model("blackhole").channel_bytes_per_cycle(True) is None


def test_the_channel_occupancy_is_the_serialisation_and_not_a_second_number():
    """The whole provenance of the term. A held channel costs
    ``ceil(N / 24)`` — the figure ``channel_serialisation`` already carries at
    ``isa_doc_derived`` and the latency term already spends as an excess. One
    number on two axes, so nothing new had to be sourced to make a second
    request wait."""
    with _env("1"):
        model = dram_cost_model("wormhole")
        channels = Wormhole().dram_tiles[0].channels
    assert channels.bytes_per_cycle == model.channel_bytes_per_cycle()
    assert channels.write_bytes_per_cycle == model.channel_bytes_per_cycle(True)
    for size in (32, 64, 256, 1024, 4096, 8192):
        assert channels.occupancy_cycles(size) == model.channel_serialisation_cycles(
            size
        )


def test_an_idle_channel_charges_nothing_so_a_lone_request_cannot_move():
    """The property that keeps this from being a second bill: ``claim`` answers
    0 whenever the channel is free, so a workload issuing one transfer at a
    time is bit-identical to the model without it — and rung 2, whose DRAM rows
    are all one transaction per barrier, cannot move either."""
    channels = DramChannels(24)
    assert channels.claim(1000, 4096) == 0
    assert channels.free_cycle() == 1000 + math.ceil(4096 / 24)
    assert channels.waits == 0
    # A request arriving after the first has finished still waits for nothing.
    assert channels.claim(1000 + math.ceil(4096 / 24), 4096) == 0
    assert channels.waits == 0
    assert channels.claims == 2


def test_a_channel_with_no_sourced_rate_never_makes_anything_wait():
    """Asserted on the mechanism rather than on an arch, so it stays true for
    the next architecture — or the next direction — whose rate is unsourced.
    Blackhole is now the *directional* case: its reads queue and its writes do
    not, because its table derives one rate and refuses the other."""
    channels = DramChannels(None)
    assert channels.occupancy_cycles(8192) is None
    assert channels.claim(0, 8192) == 0
    assert channels.claim(0, 8192) == 0
    assert channels.claims == 0
    with _env("1"):
        bh = Blackhole().dram_tiles[0].channels
    assert bh.bytes_per_cycle is not None
    assert bh.write_bytes_per_cycle is None
    assert bh.occupancy_cycles(8192, is_write=True) is None
    assert bh.claim(0, 8192, is_write=True) == 0
    assert bh.claims == 0


def test_the_two_nocs_share_one_set_of_channels_because_the_hardware_does():
    """Two GDDR6 channels behind two NoC interfaces. A per-NIU queue would let
    a NoC 0 and a NoC 1 request stream across the same bus at once, which is
    the contention the term exists to remove."""
    with _env("1"):
        tile = Wormhole().dram_tiles[0]
    assert tile.noc0_router.channels is tile.noc1_router.channels is tile.channels


def test_a_wormhole_dram_tile_fronts_two_independent_gddr6_channels():
    """``wh_dram``: "DRAM tiles occur in groups of three, with two channels of
    GDDR6 present in each group", and the group's NoC address map puts "GDDR6
    Channel 0 data" at 0 and "GDDR6 Channel 1 data" at 0x4000_0000. Modelling
    the pair as one queue would serialise two independent controllers against
    each other — an over-charge, and the direction this project's policy
    forbids. Blackhole has no DRAM tile page at all, so it declines to split
    and (publishing no rate either) queues nothing regardless."""
    assert WORMHOLE_PROFILE.dram_gddr_channel_size == 0x4000_0000
    assert WORMHOLE_PROFILE.dram_gddr_channels_per_tile == 2
    assert BLACKHOLE_PROFILE.dram_gddr_channel_size is None
    assert BLACKHOLE_PROFILE.dram_gddr_channels_per_tile == 1
    with _env("1"):
        channels = Wormhole().dram_tiles[0].channels
    assert channels.count == 2
    assert channels.index_for(0) == 0
    assert channels.index_for(0x3FFF_FFFF) == 0
    assert channels.index_for(0x4000_0000) == 1
    assert channels.index_for(0x7FFF_FFFF) == 1
    # Past the last window (the register aperture) clamps rather than raising.
    assert channels.index_for(0xFFB2_0000) == 1


def test_traffic_to_the_two_halves_of_a_tile_does_not_contend():
    """The consequence of the split, on the mechanism: two transfers landing on
    the same cycle in opposite 1 GiB halves are two controllers' work and
    neither waits. Charged as one queue, the second would have paid the first's
    whole occupancy."""
    channels = DramChannels(24, count=2, channel_size=0x4000_0000)
    assert channels.claim(100, 8192, 0x0000_1000) == 0
    assert channels.claim(100, 8192, 0x4000_1000) == 0
    assert channels.waits == 0
    # Same half, and now it does wait.
    assert channels.claim(100, 8192, 0x0000_2000) == math.ceil(8192 / 24)
    assert channels.waits == 1


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
    this model from one that simply made the DRAM further away.

    ``excess`` is the channel term — one cycle at this 32-byte payload, since
    the channel needs two cycles for it and the link one."""
    with _env("1"):
        device = Wormhole()
        tile = device.tensix_tiles[0]
        dram = device.dram_tiles[0]
        model = dram_cost_model("wormhole")
    service = model.service_cycles
    excess = model.channel_excess_cycles(len(_PAYLOAD), math.ceil(len(_PAYLOAD) / 32))
    assert excess == 1
    src = tile.noc0_router.id_pair
    dst = dram.noc0_router.id_pair
    device.write(dram.get_coord_pair(), 0x1000, _PAYLOAD)
    initiator = _read_from_dram(tile, dst)
    landed = _cycles_until_landed(device, tile, initiator, _PAYLOAD, budget=4000)
    # The trailing +1 is the counting loop's own off-by-one (it pumps before it
    # looks), the same one ``noc_cost_model_test`` carries.
    assert landed == _flight(src, dst) + service + excess + _flight(dst, src) + 1


def test_a_large_dram_read_ends_up_limited_by_the_channel_and_not_the_link():
    """The point of the whole term, measured on a real device rather than
    asserted from the model: grow the transfer and the round trip should grow
    at the *channel's* 24 B/cycle, not the NoC link's 32. Two sizes an exact
    KiB apart, so the difference is the rate."""
    big, small = 4096, 2048
    landed = {}
    for size in (small, big):
        with _env("1"):
            device = Wormhole()
            tile = device.tensix_tiles[0]
            dram = device.dram_tiles[0]
        payload = bytes((i * 7) & 0xFF for i in range(size))
        device.write(dram.get_coord_pair(), 0x1000, payload)
        initiator = _read_from_dram(tile, dram.noc0_router.id_pair)
        initiator.at_len_be = size
        landed[size] = _cycles_until_landed(
            device, tile, initiator, payload, budget=8000
        )
    grew = landed[big] - landed[small]
    assert grew == math.ceil(big / 24) - math.ceil(small / 24)
    assert (big - small) / grew == pytest.approx(24, abs=0.1)


def _concurrent_reads(count, size):
    """``count`` reads issued back to back from one tile, all to one channel.

    Returns the cycle each landed on. Distinct initiators and distinct landing
    addresses, so nothing but the endpoint can serialise them.
    """
    with _env("1"):
        device = Wormhole()
        tile = device.tensix_tiles[0]
        dram = device.dram_tiles[0]
    payload = bytes((i * 7) & 0xFF for i in range(size))
    for k in range(count):
        device.write(dram.get_coord_pair(), 0x1000 + k * 0x10000, payload)
    unified = tile.get_coord_pair()
    for k in range(count):
        initiator = tile.noc0_router.request_initiators[k]
        _set_coord(initiator, "target", dram.noc0_router.id_pair)
        initiator.target_addr_low = 0x1000 + k * 0x10000
        initiator.ret_addr_low = _L1_DST + k * 0x10000
        initiator.at_len_be = size
        initiator.ctrl = 0
        initiator.cmd_ctrl = 1
        initiator.initiate()
    landed = {}
    for cycle in range(1, 20000):
        device.run(1)
        for k in range(count):
            if k not in landed and (
                bytes(device.read(unified, _L1_DST + k * 0x10000, size)) == payload
            ):
                landed[k] = cycle
        if len(landed) == count:
            break
    return [landed[k] for k in sorted(landed)], dram.channels


def test_a_second_request_waits_for_the_channel_the_first_is_streaming_across():
    """The item, end to end, and the measurement that motivated it.

    Before 2026-08-09 four concurrent 4 KiB reads came back 128 cycles apart —
    ``ceil(4096 / 32)``, the NoC *link's* rate — so a tt-sim Wormhole DRAM
    channel sustained 32 B/cycle against the 24 GB/s the ISA docs publish for
    it. They now come back ``ceil(4096 / 24)`` apart, and the sustained rate is
    the channel's."""
    size = 4096
    landed, channel = _concurrent_reads(4, size)
    occupancy = math.ceil(size / 24)
    gaps = [b - a for a, b in zip(landed, landed[1:])]
    assert gaps == [occupancy] * 3
    assert size / occupancy == pytest.approx(24, abs=0.1)
    # Three of the four found the channel busy, and the first did not.
    assert channel.claims == 4
    assert channel.waits == 3


def test_one_request_at_a_time_is_unmoved_by_the_queue():
    """The control for the test above, and the reason no single-transaction
    prediction in the tree moves: a lone read never finds the channel busy, so
    not a cycle is charged and its landing is the one the two tests above
    already pin from the flight, the service time and the excess."""
    landed, channel = _concurrent_reads(1, 4096)
    assert channel.claims == 1
    assert channel.waits == 0
    assert channel.cycles_waited == 0
    assert len(landed) == 1


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
        model = dram_cost_model("wormhole")
    service = model.service_cycles + model.channel_excess_cycles(
        len(_PAYLOAD), math.ceil(len(_PAYLOAD) / 32)
    )
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
