"""Where an endpoint *stands* on the NoC, and what the hop model does about it.

Two properties, both about the coordinate a flight time is measured to rather
than about the flight time itself. They are cheap — no trace, no kernel, no
simulation, about two seconds per architecture — and between them they cover
three separate defects, of which only the first had been noticed.

**1. The grid guard.** :func:`~tt_sim.network.tt_noc.noc_hop_count` and
:func:`~tt_sim.network.tt_noc.noc_route_links` walk a torus. Handed a
coordinate outside it, the first returned a modular fiction and the second
**did not terminate** — its walk steps ``x = (x + 1) % grid_x``, which never
reaches an out-of-grid ``dst[0]``, so it spun in pure Python with the device
clock stopped and nothing able to report it. Both now refuse. The live case is
a translated Wormhole worker coordinate, ``(18..25, 18..27)`` against a 10x12
grid: outside it on both axes.

**2. The endpoint-consistency invariant.** For every key in a NoC's
destination directory::

    flight_cycles_to(directory[key]) == flight_cycles_to(NullEndpoint(key))

A ``NullEndpoint`` is *this cell, without a model*, so wherever both are
defined they must agree; a disagreement means at least one side is timing a
packet somewhere other than where it is going. This one assertion caught the
Wormhole DRAM sub-endpoint mis-timing that nobody had reported — six of the
108 Wormhole NoC 0 keys, and the dominant traffic class in every workload in
the tree.

**Blackhole is not the clean control it looks like.** It has one worker-visible
endpoint per channel per NoC, which reads as unambiguous; what that misses is
that a channel's NoC 1 endpoint is a *different subchannel* from its NoC 0 one
(``(0, 11)`` and ``(0, 1)``, two rows of the same three-endpoint DRAM), while
the tile's NoC 1 NUI was built from the NoC 0 coord. All eight channels were
mis-timed on NoC 1 — 65 % of every destination the committed Blackhole replay
guards resolve.

The residual, which is expected and is pinned rather than waived
------------------------------------------------------------------

NoC 1's directory is keyed in **two conventions at once** (see
``TT_Device._register_tile_internals``): a tile's canonical (SoC-physical
NoC 0) coord, *and* the ``(GRID-1-x, GRID-1-y)`` mirror that tt-metal's
bank-to-noc table emits. Only the second is a NoC 1 coordinate. A
``NullEndpoint`` built from a canonically keyed cell therefore holds a coord in
the *other* space, and disagrees with the NUI — whose coord is the mirror, and
is the correct one. That is a real defect and it is **not** what these tests
fix: it is the two-convention keying itself, and it is cleared by removing that
keying rather than by patching a coordinate at the point of use.

**Enabling NoC coordinate translation does not, by itself, clear it** — a
correction to what this note first claimed, made after measuring rather than
reasoning. Translation is *additive*: it gives every tile an unambiguous key in
a disjoint space, which is what finally makes the canonical/mirror pair safe to
delete, but it adds those keys and leaves the canonical NoC 1 entries exactly
where they were. Measured on the full grid with translation on, these counts
are **unchanged on Wormhole** (28 keys / 24 mis-costed) and **six higher on
Blackhole** (148 / 142) — the six DRAM channels whose NoC 1 mirror cell a
translated worker key takes over. Removing the dual keying is a separate
change, and it is the one that moves these numbers to zero.

So the invariant is asserted in full on NoC 0, and on NoC 1 for every key that
is in NoC 1's own space; every remaining key must miss by *exactly* the mirror
and by nothing else. The counts are pinned so that a change which moves them
has to say so. They are measured on an untranslated device, which is the
default and the configuration every committed trace replays in.
"""

import os
from contextlib import contextmanager

import pytest

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.tt_noc import (
    NoCCoordinateError,
    NullEndpoint,
    _endpoint_noc_coord,
    noc_hop_count,
    noc_route_links,
)

#: Every Wormhole functional worker, in tt-sim's unified coord band — the
#: product of the descriptor's 8 columns and 10 rows (``driver/wormhole/
#: soc_descriptor.yaml``, paired with the band by ``server/coords.py``).
#: Spelled out here because ``tt_sim`` never imports from ``driver``.
_WH_WORKERS = [(x, y) for x in range(18, 26) for y in range(16, 26)]
#: Every Blackhole functional worker; Blackhole keys tiles by physical coord,
#: and ``x = 8, 9`` is the ARC / DRAM gap.
_BH_WORKERS = [
    (x, y) for x in list(range(1, 8)) + list(range(10, 17)) for y in range(2, 12)
]

#: Initiators the audit measures from. Any tile does — the invariant is about
#: the *destination* — so these are just the first worker of each grid.
_WH_INITIATOR = (18, 18)
_BH_INITIATOR = (1, 2)


@contextmanager
def _model_on():
    """The cost model on and NoC 1 shadow reporting quiet, restored after.

    Every worker is materialised here, which on Wormhole genuinely shadows 56
    of them on NoC 1 — a real defect with its own tests
    (``tt_sim/network/noc_routing_test.py``), and not something this module
    should print 56 lines about.
    """
    previous = {
        k: os.environ.get(k) for k in ("TT_SIM_COST_MODEL", "TT_SIM_NOC1_SHADOW")
    }
    os.environ["TT_SIM_COST_MODEL"] = "1"
    os.environ["TT_SIM_NOC1_SHADOW"] = "off"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_DEVICES = {}


def _device(arch):
    """A fully populated device of ``arch``, built once per session."""
    if arch not in _DEVICES:
        with _model_on():
            _DEVICES[arch] = (
                Wormhole(tensix_coords=_WH_WORKERS)
                if arch == "wormhole"
                else Blackhole(tensix_coords=_BH_WORKERS)
            )
    return _DEVICES[arch]


def _initiator(device, coord, noc):
    return device.tile_directory[coord].get_noc_nui(noc)


# ---------------------------------------------------------------------------
# 1. The grid guard: a coordinate off the grid is refused, not walked.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coord", [(18, 20), (10, 0), (0, 12), (-1, 0), (0, -1)])
def test_an_out_of_grid_destination_is_refused_by_both_distance_functions(coord):
    """The Wormhole translated worker band ``(18..25, 18..27)`` is the live case.

    ``noc_route_links`` used to spin forever on any of these; ``noc_hop_count``
    used to answer with a modulus of a coordinate that names no router.
    """
    for fn in (noc_hop_count, noc_route_links):
        with pytest.raises(NoCCoordinateError) as excinfo:
            fn((1, 1), coord, 10, 12)
        message = str(excinfo.value)
        assert str(coord) in message, message
        assert "10x12" in message, message
        assert "destination" in message, message


def test_an_out_of_grid_source_is_refused_too_and_says_so():
    for fn in (noc_hop_count, noc_route_links):
        with pytest.raises(NoCCoordinateError) as excinfo:
            fn((18, 20), (1, 1), 10, 12)
        assert "source" in str(excinfo.value)


def test_the_congestion_planner_shares_the_guard():
    """``tt_sim.perf.noc_congestion_plan.route_links`` **is** ``noc_route_links``.

    Pinned because the planner and the simulator have to name links
    identically, so they must also refuse identically.
    """
    from tt_sim.perf import noc_congestion_plan as plan

    assert plan.route_links is noc_route_links
    with pytest.raises(NoCCoordinateError):
        plan.route_links((1, 1), (18, 20), 10, 12)


def test_a_null_routed_out_of_grid_coord_names_itself_instead_of_hanging():
    """The grid guard end to end, through a real device's NUI.

    A ``NullEndpoint`` is the one endpoint whose coordinate is not checked
    against the grid on the way in — it is whatever the initiator's command
    registers held — so it is where an out-of-grid coordinate actually
    arrives.
    """
    device = _device("wormhole")
    nui = _initiator(device, _WH_INITIATOR, 0)
    for call in (nui.flight_cycles_to, nui.route_links_to):
        with pytest.raises(NoCCoordinateError) as excinfo:
            call(NullEndpoint((18, 20)))
        assert "NullEndpoint" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2. The endpoint-consistency invariant.
# ---------------------------------------------------------------------------


def _mirror(device, coord):
    return (
        device.profile.noc_grid_x - 1 - coord[0],
        device.profile.noc_grid_y - 1 - coord[1],
    )


def _audit(device, initiator, noc):
    """Classify every directory key by where its endpoint actually stands.

    Returns ``(same_space, mirror_keyed, misplaced)`` — lists of keys whose
    endpoint sits, respectively, on the cell the key names, on that cell's
    grid mirror (a canonically keyed NoC 1 entry, the documented residual), or
    somewhere else entirely (which is a bug with no explanation).
    """
    directory = device.noc_0_directory if noc == 0 else device.noc_1_directory
    src = _initiator(device, initiator, noc)
    same, mirrored, misplaced = [], [], []
    for key in sorted(directory):
        endpoint = directory[key]
        # The same reader the hop model uses, so this cannot drift from it.
        where = _endpoint_noc_coord(endpoint)
        if where == key:
            same.append(key)
        elif where == _mirror(device, key):
            mirrored.append(key)
        else:
            misplaced.append((key, where))
    return src, directory, same, mirrored, misplaced


@pytest.mark.parametrize(
    "arch, initiator", [("wormhole", _WH_INITIATOR), ("blackhole", _BH_INITIATOR)]
)
@pytest.mark.parametrize("noc", [0, 1])
def test_every_endpoint_stands_on_the_cell_its_key_names_or_that_cells_mirror(
    arch, initiator, noc
):
    """No endpoint may sit anywhere a coordinate convention cannot explain.

    This is the assertion that fails on the DRAM sub-endpoint aliases: key
    ``(0, 1)`` resolved to the NUI at ``(0, 11)``, the far end of the same
    column, which is neither the cell nor its mirror. Nothing about it is
    NoC 1 specific and nothing about it needs the two-convention keying — it
    was wrong on Wormhole NoC 0, where there is only one convention.
    """
    device = _device(arch)
    _, _, _, _, misplaced = _audit(device, initiator, noc)
    assert misplaced == [], (
        f"{arch} NoC{noc}: {len(misplaced)} directory keys resolve to an "
        f"endpoint standing on neither the cell they name nor its mirror, so "
        f"packets to them are timed to the wrong physical NIU: {misplaced[:6]}"
    )


@pytest.mark.parametrize(
    "arch, initiator", [("wormhole", _WH_INITIATOR), ("blackhole", _BH_INITIATOR)]
)
@pytest.mark.parametrize("noc", [0, 1])
def test_a_modelled_endpoint_and_a_null_one_cost_the_same_flight(arch, initiator, noc):
    """The invariant proper, for every key in this NoC's own coordinate space.

    The keys excluded are exactly the canonically keyed NoC 1 entries, whose
    ``NullEndpoint`` coordinate is in the *other* convention — the module
    docstring says why that is out of scope, and the test below pins how many
    there are so the exclusion cannot quietly grow.
    """
    device = _device(arch)
    src, directory, same, _, _ = _audit(device, initiator, noc)
    disagreements = [
        (
            key,
            src.flight_cycles_to(directory[key]),
            src.flight_cycles_to(NullEndpoint(key)),
        )
        for key in same
    ]
    disagreements = [d for d in disagreements if d[1] != d[2]]
    assert disagreements == [], (
        f"{arch} NoC{noc}: {len(disagreements)} keys are timed differently "
        f"depending on whether the cell happens to be modelled: {disagreements[:6]}"
    )


#: Directory keys written in the *canonical* convention on NoC 1, per
#: architecture. Every one of them costs a ``NullEndpoint`` flight in the wrong
#: coordinate space — the two-convention residual the module docstring
#: describes, which only removing the dual keying clears. NoC 0 has one
#: convention and so must always be zero.
#:
#: Wormhole 28: its 80 workers are all registered canonically *and* by mirror,
#: and a mirror registration is authoritative, so only the canonical keys no
#: mirror claimed survive. Blackhole 142: it registers no Tensix mirrors at all
#: (``noc1_tensix_mirror_aliases=False`` — its bank-to-noc table never emits
#: one), so nearly every worker key is canonical.
_CANONICALLY_KEYED_ON_NOC1 = {"wormhole": 28, "blackhole": 142}

#: How many of those actually cost a different flight from the initiator this
#: module measures from. Fewer than the keys above, because a cell and its
#: mirror are sometimes the same distance away by coincidence. Initiator-
#: dependent, hence the fixed initiators.
#:
#: These two numbers are the *whole* residual. Before the DRAM sub-endpoint fix
#: the same audit read 6 / 31 / 0 / 144 across (WH NoC0, WH NoC1, BH NoC0,
#: BH NoC1); both NoC 0 columns are now zero for good, and what is left on
#: NoC 1 is the coordinate-convention gap alone.
_NOC1_FLIGHT_DISAGREEMENTS = {"wormhole": 24, "blackhole": 136}


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_the_two_convention_residual_is_confined_to_noc1_and_does_not_grow(arch):
    device = _device(arch)
    initiator = _initiator_coord(arch)
    _, _, _, noc0_mirrored, _ = _audit(device, initiator, 0)
    assert noc0_mirrored == [], (
        f"{arch} NoC 0 is keyed in one convention only; a mirror-keyed entry "
        f"there is a bug, not a residual: {noc0_mirrored[:6]}"
    )
    src, directory, _, noc1_mirrored, _ = _audit(device, initiator, 1)
    assert len(noc1_mirrored) == _CANONICALLY_KEYED_ON_NOC1[arch], (
        f"{arch} NoC 1 has {len(noc1_mirrored)} canonically keyed entries, "
        f"expected {_CANONICALLY_KEYED_ON_NOC1[arch]} — the two-convention "
        "residual moved; see this module's docstring before re-pinning it"
    )
    disagreeing = [
        key
        for key in noc1_mirrored
        if src.flight_cycles_to(directory[key])
        != src.flight_cycles_to(NullEndpoint(key))
    ]
    assert len(disagreeing) == _NOC1_FLIGHT_DISAGREEMENTS[arch], (
        f"{arch} NoC 1: {len(disagreeing)} of the canonically keyed entries "
        f"are mis-costed, expected {_NOC1_FLIGHT_DISAGREEMENTS[arch]}"
    )


def _initiator_coord(arch):
    return _WH_INITIATOR if arch == "wormhole" else _BH_INITIATOR


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
