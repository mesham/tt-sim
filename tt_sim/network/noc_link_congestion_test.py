"""Router-to-router link occupancy: the shared half of the bandwidth model.

The injecting NIU's own port has been occupied since the bandwidth term landed
(``noc_cost_model_test.py`` section 5). This is the same occupancy — one flit
per cycle per axis, ``noc.hops.router_to_router.throughput_flits_per_cycle``,
already in ``tt_sim/perf/unit_costs.yaml`` at ``isa_doc`` — charged on each
router-to-router link a packet crosses instead of only where it is injected.
No new number enters anything; what changes is where an existing one is spent.

Three things had to be built for it, and each of them is a way this could be
wrong rather than a detail:

1. **The state is on the device**, one watermark per link per NoC, because a
   link is crossed by every tile whose route passes through it. On an NIU it
   would be per-tile state and two tiles could not contend on it, which is the
   entire effect.
2. **The network layer now commits to a hop *order***, not just a count, since
   a link needs an identity. It is the experiment planner's order, literally
   the same function — see ``noc_congestion_plan_test``, which pins that.
3. **A multicast claims each link once.** tt-sim models a multicast write as N
   unicasts; claiming per destination would serialise the launch-message path
   every tt-metal program uses against itself, which is the over-charge
   ``claim_injection_port`` was given its odd signature to avoid.

And the property that keeps the whole thing from being a second charge for the
same bytes: **it is inert for a single flow**. A flow's packets leave its
injection port exactly one occupancy apart, so they reach every link on their
route one occupancy apart and never queue behind themselves. Only another
tile's traffic costs anything.

The measurement this reproduces is ``docs/bh_arch.md`` §4.2: on a Blackhole
card two flows sharing one router-to-router link each pay one extra
transaction's link occupancy above the issue loop, and nothing at all below it.

Runs standalone (``python3 -m tt_sim.network.noc_link_congestion_test``) or
under pytest.
"""

import os
from contextlib import contextmanager

from tt_sim.device.blackhole import Blackhole
from tt_sim.network.tt_noc import NUI, NocLinkRegistry, noc_route_links

_GRID_X, _GRID_Y = 17, 12
_FLIT_BYTES = 64  # Blackhole: 512-bit flits, one per cycle per axis
_L1_SRC = 0x40000
_L1_DST = 0x60000


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


def _device(coords, model="1"):
    with _env(model):
        return Blackhole(tensix_coords=list(coords))


def _tiles(device):
    return {tile.get_coord_pair(): tile for tile in device.tensix_tiles}


def _arm_write(device, master, sub, payload, *, broadcast_to=None):
    """Load ``payload`` into ``master``'s L1 and arm a write of it to ``sub``."""
    device.write(master, _L1_SRC, payload)
    initiator = _tiles(device)[master].noc0_router.request_initiators[0]
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(payload)
    if broadcast_to is None:
        # Blackhole holds the destination coord in the dedicated HI register.
        initiator.ret_addr_hi = (sub[1] << 6) | sub[0]
        initiator.ctrl = 2 | (1 << 4)  # write, response marked
    else:
        # Blackhole packs both corners of the rectangle into the HI register.
        (x_start, y_start), (x_end, y_end) = sub, broadcast_to
        initiator.ret_addr_hi = x_end | (y_end << 6) | (x_start << 12) | (y_start << 18)
        initiator.ctrl = 2 | (1 << 4) | (1 << 5)  # write, marked, broadcast
    return initiator


def _issue(initiator):
    initiator.cmd_ctrl = 1
    initiator.initiate()


# ---------------------------------------------------------------------------
# Blocker 1: the state is on the device, because links are shared between tiles.
# ---------------------------------------------------------------------------


def test_the_link_watermarks_live_on_the_device_one_per_noc():
    """On an NIU this state would be per-tile, and two tiles could not contend.

    Both NoCs need their own registry for the opposite reason: NoC 0 and NoC 1
    are separate networks whose links happen to be named by the same tuples.
    """
    device = _device([(1, 2), (5, 2)])
    nuis = [tile.noc0_router for tile in device.tensix_tiles]
    assert len({id(nui.noc_link_registry) for nui in nuis}) == 1
    assert nuis[0].noc_link_registry is device.noc_link_registries[0]
    for tile in device.tensix_tiles:
        assert tile.noc1_router.noc_link_registry is device.noc_link_registries[1]
    assert device.noc_link_registries[0] is not device.noc_link_registries[1]
    # DRAM tiles are on the same links as the workers and share the registry.
    assert (
        device.dram_tiles[0].noc0_router.noc_link_registry is nuis[0].noc_link_registry
    )
    device.shutdown()


def test_a_tile_materialised_later_joins_the_same_registry():
    """``add_tensix_tile`` is how the wire bridge grows the device, so a lazily
    materialised worker that got its own registry would silently stop
    contending with everything already there."""
    device = _device([(1, 2)])
    with _env("1"):
        late = device.add_tensix_tile((5, 2))
    assert late.noc0_router.noc_link_registry is device.noc_link_registries[0]
    device.shutdown()


def test_two_tiles_crossing_one_link_pay_one_transactions_occupancy():
    """The headline, at one packet. Two masters on row 2 whose payload routes
    share exactly the link out of ``(4, 2)``: the second packet to reach it
    waits for the first to finish crossing, and the wait is one occupancy."""
    payload = bytes(4096)
    occupancy = len(payload) // _FLIT_BYTES
    device = _device([(1, 2), (4, 2), (5, 2), (6, 2)])
    a = _arm_write(device, (1, 2), (5, 2), payload)
    b = _arm_write(device, (4, 2), (6, 2), payload)
    shared = set(noc_route_links((1, 2), (5, 2), _GRID_X, _GRID_Y)) & set(
        noc_route_links((4, 2), (6, 2), _GRID_X, _GRID_Y)
    )
    assert shared == {("X", 2, 4)}

    registry = device.noc_link_registries[0]
    device.run(1)
    _issue(a)
    _issue(b)
    assert registry.waits == 1
    assert registry.cycles_waited == occupancy
    # ...and the link is booked out for both packets, back to back.
    assert registry.free_cycle(("X", 2, 4)) - device.clocks[0].current_cycle == (
        2 * occupancy
    )
    device.shutdown()


def test_a_flow_never_queues_behind_itself():
    """The property that stops this being a second charge for the same bytes.

    Eight packets out of one NIU over a ten-hop route. The injection port
    already spaces them one occupancy apart, so each one reaches each link on
    its route exactly as the previous one leaves it: eighty link claims, no
    wait. Anything else would mean a single-flow workload got slower for
    crossing more routers, which is the per-hop *latency* term's job and is
    already charged.
    """
    payload = bytes(2048)
    device = _device([(1, 2), (5, 4)])
    initiator = _arm_write(device, (1, 2), (5, 4), payload)
    registry = device.noc_link_registries[0]
    hops = len(noc_route_links((1, 2), (5, 4), _GRID_X, _GRID_Y))
    for _ in range(8):
        _issue(initiator)
        device.run(1)
    assert registry.claims == 8 * hops
    assert registry.waits == 0
    assert registry.cycles_waited == 0
    device.shutdown()


# ---------------------------------------------------------------------------
# Blocker 2: a link identity needs a hop *order*, and there is only one.
# ---------------------------------------------------------------------------


def test_the_route_a_packet_takes_is_as_long_as_the_hops_it_is_charged_for():
    """The route and the flight time have to be the same journey. They are
    computed by two different functions off the same pair of coords, so this is
    the pin that stops one of them being changed alone."""
    device = _device([(1, 2), (5, 4)])
    src = _tiles(device)[(1, 2)].noc0_router
    dst = _tiles(device)[(5, 4)].noc0_router
    links = src.route_links_to(dst)
    assert links == noc_route_links((1, 2), (5, 4), _GRID_X, _GRID_Y)
    assert len(links) == (5 - 1) + (4 - 2)
    assert src.flight_cycles_to(dst) == 5 + 5 + 9 * len(links)
    device.shutdown()


def test_noc1_routes_in_its_own_mirrored_space():
    """A link is a tuple of small integers, so a NoC 1 packet routed with NoC 0
    coords would silently claim a NoC 0 link's *name* on the other network. The
    coords come off the endpoints, mirrored, exactly as the hop count's do."""
    device = _device([(1, 2), (5, 4)])
    src = _tiles(device)[(1, 2)].noc1_router
    dst = _tiles(device)[(5, 4)].noc1_router
    mirror = lambda c: (_GRID_X - 1 - c[0], _GRID_Y - 1 - c[1])  # noqa: E731
    assert src.route_links_to(dst) == noc_route_links(
        mirror((1, 2)), mirror((5, 4)), _GRID_X, _GRID_Y
    )
    # NoC 1 runs the other way, so the same pair is the far side of the torus.
    assert len(src.route_links_to(dst)) == (_GRID_X - 4) + (_GRID_Y - 2)
    device.shutdown()


# ---------------------------------------------------------------------------
# Blocker 3: a multicast is one packet, and the launch path is made of them.
# ---------------------------------------------------------------------------


def test_a_multicast_claims_each_link_once_not_once_per_destination():
    """The over-charge this had to solve before it could be turned on.

    A multicast crosses a *tree* — the union of the dimension-ordered routes to
    each destination — and the routers fan it out, so every link on that tree
    carries the packet exactly once. Charging per modelled unicast would book a
    link out for twelve occupancies here and invent serialisation on the packet
    every tt-metal program's launch message is.
    """
    payload = bytes(1024)
    occupancy = len(payload) // _FLIT_BYTES
    device = _device([(1, 2)])
    rectangle = [(x, y) for x in range(2, 6) for y in range(2, 5)]
    initiator = _arm_write(device, (1, 2), (2, 2), payload, broadcast_to=(5, 4))
    registry = device.noc_link_registries[0]
    device.run(1)
    now = device.clocks[0].current_cycle
    _issue(initiator)

    routes = [noc_route_links((1, 2), d, _GRID_X, _GRID_Y) for d in rectangle]
    tree = set().union(*(set(r) for r in routes))
    assert len(rectangle) == 12
    # The whole reason the tree is a tree: twelve unicasts would be 42 claims.
    assert sum(len(r) for r in routes) == 42
    assert len(tree) == 12
    for link in tree:
        assert registry.free_cycle(link) == now + occupancy, link
    assert registry.claims == len(tree)
    assert registry.waits == 0
    device.shutdown()


def test_the_multicast_tree_is_exactly_the_union_of_its_destinations_routes():
    """Once, but not fewer than once: every link a destination's packet needs
    is on the tree, or a copy would cross an unclaimed link."""
    payload = bytes(512)
    device = _device([(1, 2)])
    initiator = _arm_write(device, (1, 2), (3, 3), payload, broadcast_to=(6, 6))
    registry = device.noc_link_registries[0]
    device.run(1)
    _issue(initiator)
    needed = set()
    for x in range(3, 7):
        for y in range(3, 7):
            needed |= set(noc_route_links((1, 2), (x, y), _GRID_X, _GRID_Y))
    assert registry.claims == len(needed)
    assert all(registry.free_cycle(link) for link in needed)


# ---------------------------------------------------------------------------
# The measurement: a step at the first shared link, sized like the occupancy.
# ---------------------------------------------------------------------------


def _drain_cycles(flows, size, num_tx=16, interval=40):
    """Cycles for every flow in ``flows`` to land ``num_tx`` transactions.

    ``nocbench``'s ``shared`` experiment with the tt-metal kernel taken out:
    each master arms one write and re-issues it every ``interval`` cycles,
    which is what that kernel's issue loop does at ~39.8 cycles on silicon.
    """
    device = _device(sorted({c for flow in flows for c in flow}))
    payload = bytes((i * 7) & 0xFF for i in range(size))
    initiators = [_arm_write(device, master, sub, payload) for master, sub in flows]
    cycles = 0
    for _ in range(num_tx):
        for initiator in initiators:
            _issue(initiator)
        device.run(interval)
        cycles += interval
    outstanding = NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0
    while any(i.nui.nui_counters[outstanding] for i in initiators):
        device.run(interval)
        cycles += interval
    device.shutdown()
    return cycles / num_tx


#: Flow A, and the flow Bs that share 0, 1 and 2 of its payload links. A is
#: ``(1, 2) -> (7, 2)``, i.e. the row-2 links out of x = 1..6; a B starting at
#: x = 6 shares one of them, at x = 5 shares two, and one row down shares none.
_FLOW_A = ((1, 2), (7, 2))
_FLOW_B = {0: ((1, 4), (7, 4)), 1: ((6, 2), (8, 2)), 2: ((5, 2), (8, 2))}


def test_two_flows_sharing_a_link_each_pay_one_transactions_occupancy():
    """``docs/bh_arch.md`` §4.2, reproduced: above the issue loop the cost of a
    second flow on a shared link is one transaction's worth of that link's
    occupancy. On the card, at 4096 B, the delta was 63.1 cycles against an
    occupancy of 64 -- a ratio of 0.99, with no fitted parameter anywhere.
    """
    size = 4096
    occupancy = size // _FLIT_BYTES
    alone = _drain_cycles([_FLOW_A, _FLOW_B[0]], size)
    shared = _drain_cycles([_FLOW_A, _FLOW_B[1]], size)
    assert 0.85 <= (shared - alone) / occupancy <= 1.15


def test_below_the_issue_loop_a_shared_link_costs_exactly_nothing():
    """The negative control, and the reason no threshold had to be chosen. A
    512 B packet holds a link for 8 cycles against a 40-cycle issue loop, so
    the link is 20 % busy and two flows almost never want it at once. The card
    reads 39.8 both with and without the shared link; this reads equality."""
    size = 512
    assert _drain_cycles([_FLOW_A, _FLOW_B[0]], size) == _drain_cycles(
        [_FLOW_A, _FLOW_B[1]], size
    )


def test_the_second_shared_link_costs_almost_nothing_more():
    """A step, not a slope -- the property that makes the fitted per-shared-link
    coefficient the card's own analysis prints a description of a machine that
    does not exist. Two flows already queued on one link arrive at the next one
    in step, so it is never busy when they get there. On silicon the 1-to-7
    span was 6 % of the step at 16 KiB."""
    size = 4096
    one = _drain_cycles([_FLOW_A, _FLOW_B[1]], size)
    two = _drain_cycles([_FLOW_A, _FLOW_B[2]], size)
    assert abs(two - one) <= 0.1 * (size // _FLIT_BYTES)


# ---------------------------------------------------------------------------
# The opt-in, and the mechanism on its own.
# ---------------------------------------------------------------------------


def test_no_link_is_claimed_with_the_model_off():
    """The gate's first question. With ``TT_SIM_COST_MODEL`` unset the NUIs
    have no cost model, so nothing reaches the registry and the device is the
    one every guard in the tree was recorded against."""
    device = _device([(1, 2), (5, 2)], model=None)
    initiator = _arm_write(device, (1, 2), (5, 2), bytes(4096))
    for _ in range(4):
        _issue(initiator)
        device.run(1)
    registry = device.noc_link_registries[0]
    assert (registry.claims, registry.waits, registry.cycles_waited) == (0, 0, 0)
    assert registry.free_cycle(("X", 2, 1)) == 0
    device.shutdown()


def test_an_nui_with_no_device_behind_it_charges_no_link():
    """``driver/simple`` and the unit tests build NUIs directly. They have no
    registry, the same way they have no cost model, and must not acquire a
    timing model from an environment variable set for something else."""
    with _env("1"):
        nui = NUI(0, 1, 2, None)
    assert nui.noc_link_registry is None
    assert nui.route_links_to(nui) == ()
    assert nui.claim_route_links((("X", 2, 1),), 4096) == 0


def test_the_watermark_walks_the_route_and_the_wait_accumulates():
    """The mechanism on its own, away from any device.

    A packet held up at one link reaches the next one later, so the wait is
    cumulative along the route — which is what makes a contended route cost the
    *arrival* and not merely the link. Two busy links, two waits, one sum.
    """
    registry = NocLinkRegistry()
    first, second = ("X", 2, 1), ("X", 2, 2)
    assert registry.claim((first, second), 0, 10) == 0
    assert registry.free_cycle(first) == registry.free_cycle(second) == 10
    # A second packet arriving at cycle 4 waits 6 for the first link, and by
    # then the second link is free too -- the wait it already paid covers it.
    assert registry.claim((first, second), 4, 10) == 6
    assert registry.free_cycle(first) == registry.free_cycle(second) == 20
    assert (registry.claims, registry.waits, registry.cycles_waited) == (4, 1, 6)


def test_the_shutdown_summary_distinguishes_inert_from_dead():
    """``waits == 0`` is the whole point of printing this, so it must be said.

    The term was inert on every in-tree workload for a week and the only way to
    see that was a one-off instrumentation patch. A summary that reported claims
    alone would not have distinguished "nothing contended" from "the term is
    dead code", which is the one distinction it exists to make.
    """
    from tt_sim.bridge.device import link_contention_summary

    class _Dev:
        def __init__(self, registries):
            self.noc_link_registries = registries

    # Nothing to say: no registries, and registries nothing ever claimed.
    assert link_contention_summary(None) == ""
    assert link_contention_summary(_Dev((NocLinkRegistry(), NocLinkRegistry()))) == ""

    noc0, noc1 = NocLinkRegistry(), NocLinkRegistry()
    link = ("X", 2, 1)
    noc0.claim((link,), 0, 10)
    noc0.claim((link,), 4, 10)  # waits 6
    summary = link_contention_summary(_Dev((noc0, noc1)))
    assert "2 claims, 1 waits, 6 cycles waited" in summary
    assert "noc1: 0 claims, 0 waits" in summary

    # An uncontended run says so rather than falling silent, which is what makes
    # "the term fired and found nothing" a reportable reading.
    only_claims = NocLinkRegistry()
    only_claims.claim((link,), 0, 10)
    assert "1 claims, 0 waits" in link_contention_summary(_Dev((only_claims,)))


def test_an_empty_route_and_an_unsourced_occupancy_charge_nothing():
    """Two endpoints on one tile cross no router-to-router link at all, and a
    cost table with no flit width sources no occupancy. Neither is an error and
    neither may fabricate a wait."""
    registry = NocLinkRegistry()
    assert registry.claim((), 0, 10) == 0
    assert registry.claim((("X", 2, 1),), 0, None) == 0
    assert registry.claims == 0


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
