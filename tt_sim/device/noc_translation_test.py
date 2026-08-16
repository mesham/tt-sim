"""Devices built with NoC coordinate translation enabled.

The four things that have to hold, on both architectures:

1. The translated coordinates tt-metal puts on the wire resolve, on both NoCs,
   to the tile that owns them. These are checked against coordinates *measured*
   on the wire from translated runs, not against a re-derivation of the same
   formula.
2. ``NOC_ID_LOGICAL`` — the register firmware fills ``my_x``/``my_y`` from, and
   compares against virtual bank coordinates — reports the translated coord and
   resolves back to the endpoint that reported it. ``NOC_NODE_ID`` is a
   different register with a different job (the physical node ID) and is
   checked separately, against the physical directory keys translation leaves
   in place.
3. A translated directory key carries the *physical* cell into the hop model,
   so a run with ``TT_SIM_COST_MODEL`` set is timed correctly rather than
   merely not fatal.
4. Nothing about the default (untranslated) path changes.
"""

import pytest

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.tt_noc import (
    NoCCoordinateError,
    _endpoint_noc_coord,
    resolved_nui,
)
from tt_sim.util.bits import extract_bits

# Every functional worker on each architecture, so the census is the full grid
# rather than the handful a program happens to launch on.
WH_WORKERS = [(ux, uy) for ux in range(18, 26) for uy in range(16, 26)]
BH_WORKERS = [
    (x, y) for x in list(range(1, 8)) + list(range(10, 17)) for y in range(2, 12)
]


@pytest.fixture(scope="module")
def wormhole_translated():
    return Wormhole(tensix_coords=WH_WORKERS, noc_translation=True)


@pytest.fixture(scope="module")
def blackhole_translated():
    return Blackhole(tensix_coords=BH_WORKERS, noc_translation=True)


def _decode(reg):
    return (extract_bits(reg, 6, 0), extract_bits(reg, 6, 6))


def _disagreements(device, register):
    """Endpoints whose ``register`` coord does not resolve back to them.

    ``resolved_nui`` because a directory entry may be an ``AliasedEndpoint``
    view of the NUI rather than the NUI itself — the packet still lands on the
    same interface, which is what "resolves back to them" means here.
    """
    bad = []
    for tile in device.tile_directory.values():
        for noc, directory in (
            (0, device.noc_0_directory),
            (1, device.noc_1_directory),
        ):
            nui = tile.get_noc_nui(noc)
            coord = _decode(getattr(nui, register))
            entry = directory.get(coord)
            if entry is None or resolved_nui(entry) is not nui:
                bad.append((noc, coord, tile.get_coord_pair()))
    return bad


def test_translation_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TT_SIM_NOC_TRANSLATION", raising=False)
    monkeypatch.delenv("TT_METAL_MOCK_CLUSTER_DESC_PATH", raising=False)
    device = Wormhole()
    assert device.noc_translation is False
    assert device.translated_displacements == {}
    for tile in device.tile_directory.values():
        assert tile.translated_coords == ()
        assert tile.get_noc_nui(0).translated_coord is None
        assert tile.get_noc_nui(1).translated_coord is None


def test_wormhole_workers_and_eth_move_dram_does_not(wormhole_translated):
    device = wormhole_translated
    workers = sorted(device.tile_directory[c].translated_coords[0] for c in WH_WORKERS)
    # Measured on the wire from a translated ``examples/one`` run.
    assert workers == sorted((x, y) for x in range(18, 26) for y in range(18, 28))
    eth = sorted(
        t.translated_coords[0]
        for t in device.tile_directory.values()
        if t.tile_role == "eth"
    )
    assert eth == sorted((x, y) for x in range(18, 26) for y in (16, 17))
    # "DRAM cores are not translated in Wormhole" — so a DRAM tile keeps its
    # physical coords and its mirrored NoC 1 alias in every mode.
    assert all(
        t.translated_coords == ()
        for t in device.tile_directory.values()
        if t.tile_role == "dram"
    )


def test_blackhole_dram_moves_workers_do_not(blackhole_translated):
    device = blackhole_translated
    workers = {device.tile_directory[c].translated_coords[0] for c in BH_WORKERS}
    assert workers == set(BH_WORKERS)
    dram = sorted(
        c
        for t in device.tile_directory.values()
        if t.tile_role == "dram"
        for c in t.translated_coords
    )
    # The NoC 0 and NoC 1 halves of ``dram_bank_to_noc_xy``, decoded off the
    # wire from a translated run — different sub-endpoints of the same bank,
    # not a mirror pair.
    assert dram == sorted(
        [(17, 14), (17, 15), (17, 18), (17, 21), (18, 14), (18, 17), (18, 20), (18, 23)]
        + [
            (17, 13),
            (17, 16),
            (17, 19),
            (17, 22),
            (18, 13),
            (18, 16),
            (18, 19),
            (18, 22),
        ]
    )


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_translated_coords_resolve_on_both_nocs(
    arch, wormhole_translated, blackhole_translated
):
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    for tile in device.tile_directory.values():
        for coord in tile.translated_coords:
            for noc, directory in (
                (0, device.noc_0_directory),
                (1, device.noc_1_directory),
            ):
                entry = directory.get(coord)
                assert entry is not None, f"{arch} NoC{noc} {coord}"
                assert resolved_nui(entry) is tile.get_noc_nui(noc), (
                    f"{arch} NoC{noc} {coord}"
                )


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_no_translated_worker_is_shadowed_on_noc1(
    arch, wormhole_translated, blackhole_translated
):
    """The whole point: 56 shadowed workers on Wormhole and 102 on Blackhole
    become zero once the coordinate spaces are disjoint."""
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    coords = WH_WORKERS if arch == "wormhole" else BH_WORKERS
    shadowed = [
        c
        for c in coords
        if resolved_nui(
            device.noc_1_directory[device.tile_directory[c].translated_coords[0]]
        )
        is not device.tile_directory[c].get_noc_nui(1)
    ]
    assert shadowed == []


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_noc_id_logical_agrees_with_the_directory(
    arch, wormhole_translated, blackhole_translated
):
    """Hazard 2, in the register that actually carries it under translation.

    ``risc_init`` fills ``my_x``/``my_y`` from ``NOC_ID_LOGICAL`` and tt-metal
    both compares them against virtual bank coordinates and emits them as a
    destination in single-argument ``get_noc_addr``. So the coordinate must
    resolve back to the endpoint that reported it.
    """
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    assert _disagreements(device, "noc_id_logical") == []


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_noc_id_logical_is_the_translated_coord_on_both_nocs(
    arch, wormhole_translated, blackhole_translated
):
    """A translated coordinate is NoC-independent — that is what makes it work
    where a physical one does not."""
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    for tile in device.tile_directory.values():
        if not tile.translated_coords:
            continue
        for noc, expected in (
            (0, tile.translated_coords[0]),
            (1, tile.translated_coords[-1]),
        ):
            nui = tile.get_noc_nui(noc)
            assert _decode(nui.noc_id_logical) == expected
            # NIU_CFG_0 bit 14 is "coordinate translation enable".
            assert extract_bits(nui.niu_cfg_0, 1, 14) == 1


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_noc_node_id_stays_the_physical_node_id(
    arch, wormhole_translated, blackhole_translated
):
    """Translation does not move where an interface physically sits.

    ``NOC_NODE_ID`` reports that, and keeps reporting it: on silicon the
    firmware self-address derived from it still routes, because the NIU goes on
    accepting physical coordinates alongside translated ones. tt-sim models the
    same thing by *adding* translated keys rather than substituting them, so
    the physical coord stays a directory key. Pinned so that a future change
    which "unifies" the two registers has to argue with this test.
    """
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    for tile in device.tile_directory.values():
        for noc in (0, 1):
            nui = tile.get_noc_nui(noc)
            assert _decode(nui.noc_node_id) == (nui.x_coord, nui.y_coord)


def test_wormhole_translation_displaces_nothing(wormhole_translated):
    """Wormhole's translated bands sit entirely off the 10x12 physical grid, so
    no legacy NoC 1 mirror can be in the way."""
    assert wormhole_translated.translated_displacements == {}


def test_blackhole_translation_displaces_only_noc1_mirrors(blackhole_translated):
    """Blackhole's translated Tensix coord *is* its physical coord, so the
    translated key does land on cells legacy NoC 1 mirrors hold. Every one of
    them is a mirror, on NoC 1, and taking it is correct: under translation
    nothing addresses a Blackhole tile by its mirror."""
    device = blackhole_translated
    assert device.translated_displacements
    assert all(noc == 1 for noc, _ in device.translated_displacements)
    for (_, coord), (displaced, claimer) in device.translated_displacements.items():
        # The displaced entry never owned this cell canonically — it only ever
        # held it as a mirror of one of its own coords.
        mirrors = {device.noc1_mirror(displaced.get_id_pair())}
        owner = device.tile_directory.get(displaced.get_id_pair())
        if owner is not None:
            mirrors |= {device.noc1_mirror(a) for a in owner.noc_aliases}
            noc1_endpoint = getattr(owner, "noc1_endpoint_coord", None)
            if noc1_endpoint is not None:
                mirrors.add(device.noc1_mirror(noc1_endpoint))
        assert coord in mirrors, (coord, displaced.get_id_pair())
        assert claimer.translated_coord == coord


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_physical_keys_survive_translation(arch):
    """Translation is *additive*: the physical coordinate space is still there,
    exactly as on silicon, which is what lets the mode be opt-in."""
    plain = (
        Wormhole(tensix_coords=WH_WORKERS[:4], noc_translation=False)
        if arch == "wormhole"
        else Blackhole(tensix_coords=BH_WORKERS[:4], noc_translation=False)
    )
    translated = (
        Wormhole(tensix_coords=WH_WORKERS[:4], noc_translation=True)
        if arch == "wormhole"
        else Blackhole(tensix_coords=BH_WORKERS[:4], noc_translation=True)
    )
    for coord in plain.noc_0_directory:
        assert coord in translated.noc_0_directory, coord


# ---------------------------------------------------------------------------
# The hop model. A translated coordinate is an identity, not a grid position.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_every_translated_key_presents_an_on_grid_coord_to_the_hop_model(
    arch, wormhole_translated, blackhole_translated
):
    """The invariant that keeps ``TT_SIM_COST_MODEL`` working under translation.

    ``noc_hop_count`` / ``noc_route_links`` walk a torus of the architecture's
    grid, so what they are handed has to be a cell of it. Translated bands sit
    outside the grid by construction — Wormhole's workers at
    ``(18..25, 18..27)`` against 10x12 — so a directory entry that reported its
    *translated* coord would take the walk off the grid. That is not a
    hypothetical: before the entries carried physical cells it spun forever,
    and the guard that now catches it turns the spin into a raise rather than
    into a correct hop count. This asserts the coordinate is right, which is
    the thing the guard cannot check.
    """
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    grid_x, grid_y = device.profile.noc_grid_x, device.profile.noc_grid_y
    checked = 0
    for tile in device.tile_directory.values():
        for coord in tile.translated_coords:
            for directory in (device.noc_0_directory, device.noc_1_directory):
                x, y = _endpoint_noc_coord(directory[coord])
                where = f"{arch} translated key {coord} presents off-grid {(x, y)}"
                assert 0 <= x < grid_x, where
                assert 0 <= y < grid_y, where
                checked += 1
    assert checked, "no translated keys were checked"


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_a_translated_flight_is_timed_like_the_physical_one(arch, monkeypatch):
    """Same journey, same cycles — translation renames endpoints, not distance.

    Both devices are built with ``TT_SIM_COST_MODEL`` set, because the model is
    read once per NIU at construction — and because that is the configuration
    the off-grid walk lived in and the one no translated run had exercised.
    """
    monkeypatch.setenv("TT_SIM_COST_MODEL", "1")
    workers = WH_WORKERS[:6] if arch == "wormhole" else BH_WORKERS[:6]
    plain = (
        Wormhole(tensix_coords=workers)
        if arch == "wormhole"
        else Blackhole(tensix_coords=workers)
    )
    translated = (
        Wormhole(tensix_coords=workers, noc_translation=True)
        if arch == "wormhole"
        else Blackhole(tensix_coords=workers, noc_translation=True)
    )
    source = plain.tile_directory[workers[0]]
    source_t = translated.tile_directory[workers[0]]
    compared = 0
    for noc in (0, 1):
        src = source.get_noc_nui(noc)
        src_t = source_t.get_noc_nui(noc)
        assert src.noc_latency is not None, "cost model did not engage"
        for dest_coord in workers[1:]:
            physical_key = plain.tile_directory[dest_coord].get_noc_nui(noc).id_pair
            key = physical_key if noc == 0 else plain.noc1_mirror(physical_key)
            directory = plain.noc_0_directory if noc == 0 else plain.noc_1_directory
            if key not in directory:
                continue
            translated_key = translated.tile_directory[dest_coord].translated_coords[0]
            t_dir = (
                translated.noc_0_directory if noc == 0 else translated.noc_1_directory
            )
            assert src.flight_cycles_to(directory[key]) == src_t.flight_cycles_to(
                t_dir[translated_key]
            ), f"{arch} NoC{noc} {dest_coord}"
            compared += 1
    assert compared, "no flights were compared"


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_an_unmodelled_translated_destination_still_raises(arch, monkeypatch):
    """The residual, stated rather than smoothed over.

    A translated coordinate naming a tile kind this architecture does not model
    at all (Blackhole ethernet) misses the directory and becomes a
    ``NullEndpoint`` holding the translated key, which is off-grid. With the
    cost model on that is a ``NoCCoordinateError``. Nothing tt-sim models sits
    there, so there is no physical cell to substitute and no honest distance to
    charge; the named error is the right outcome and is pinned here so it stays
    a decision rather than a surprise.
    """
    from tt_sim.network.tt_noc import NullEndpoint

    monkeypatch.setenv("TT_SIM_COST_MODEL", "1")
    device = (
        Wormhole(tensix_coords=WH_WORKERS[:2], noc_translation=True)
        if arch == "wormhole"
        else Blackhole(tensix_coords=BH_WORKERS[:2], noc_translation=True)
    )
    coords = WH_WORKERS[:2] if arch == "wormhole" else BH_WORKERS[:2]
    nui = device.tile_directory[coords[0]].get_noc_nui(1)
    off_grid = (device.profile.noc_grid_x + 5, device.profile.noc_grid_y + 5)
    with pytest.raises(NoCCoordinateError):
        nui.flight_cycles_to(NullEndpoint(off_grid))
