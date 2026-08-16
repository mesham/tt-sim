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


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_translation_displaces_nothing_on_either_architecture(
    arch, wormhole_translated, blackhole_translated
):
    """Nothing a translated key lands on was still registered.

    Wormhole's translated bands sit entirely off the 10x12 physical grid, so no
    key of any convention can be in the way. Blackhole's *are* physical coords,
    and its NoC 1 mirrors used to be overwritten here — six of them, one per
    DRAM sub-endpoint whose mirror parks on a live worker. Those mirrors are no
    longer registered at all under translation, which is the stronger form of
    the same answer: a key that is never written cannot displace anything, and
    cannot be resurrected by a registration order this test does not control.
    """
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    assert device.translated_displacements == {}


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_noc0_physical_keys_survive_translation(arch):
    """On NoC 0 translation is purely *additive*, exactly as on silicon.

    NoC 0 has one convention and its keys are the cells they name, so nothing
    there has to give way. NoC 1 is the opposite case and has its own test
    below.
    """
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
# NoC 1 carries one convention under translation, and which one is geometry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_the_off_grid_claim_holds_of_the_device_it_is_declared_for(
    arch, wormhole_translated, blackhole_translated
):
    """``ArchProfile.translated_coords_off_physical_grid`` against the tiles.

    The flag decides whether NoC 1 keeps its mirror keys under translation, so
    it is checked rather than trusted. Where it claims disjointness, *every*
    translated coordinate must be off the grid — one on it would be enough to
    make some mirror ambiguous, which is exactly the case the other branch
    covers.
    """
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    grid_x, grid_y = device.profile.noc_grid_x, device.profile.noc_grid_y
    coords = [c for t in device.tile_directory.values() for c in t.translated_coords]
    assert coords
    on_grid = [c for c in coords if 0 <= c[0] < grid_x and 0 <= c[1] < grid_y]
    if device.profile.translated_coords_off_physical_grid:
        assert on_grid == []
    else:
        # One on-grid translated coordinate is enough to make every mirror key
        # ambiguous, and Blackhole has 140 of them — its DRAM is off-grid at
        # ``{17,18} x {12..23}``, its workers are the physical grid itself.
        assert on_grid


@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_noc1_carries_no_unmirrored_key_under_translation(
    arch, wormhole_translated, blackhole_translated
):
    """The change this module's siblings measure: the second convention is gone.

    A tile's canonical (SoC-physical NoC 0) coord is not a NoC 1 coordinate —
    on NoC 1 the same tile stands at the grid mirror. Untranslated, NoC 1 holds
    both anyway, because ``get_noc_addr`` emits the canonical form while the
    bank-to-noc table emits the mirror; under translation the kernel emits a
    translated coord instead and the canonical key is dead weight that shadows
    live workers. So every key that resolves to a tile must be one of that
    tile's translated coords or one of its mirrors — never a canonical coord,
    unless (Blackhole) the two are the same tuple.
    """
    device = wormhole_translated if arch == "wormhole" else blackhole_translated
    stray = []
    for tile in device.tile_directory.values():
        nui1 = tile.get_noc_nui(1)
        allowed = set(tile.translated_coords) | {
            device.noc1_mirror(coord)
            for coord in (
                getattr(tile, "noc1_endpoint_coord", None)
                or tile.get_noc_nui(0).id_pair,
                *tile.noc_aliases,
            )
        }
        stray += [
            (key, tile.get_coord_pair())
            for key, entry in device.noc_1_directory.items()
            if resolved_nui(entry) is nui1 and key not in allowed
        ]
    assert stray == []


def test_wormhole_keeps_its_physical_noc1_keys(wormhole_translated):
    """Where they are unambiguous they stay, and the self-address needs them.

    Wormhole's translated bands are off the grid, so a physical NoC 1
    coordinate still names exactly one tile — which is also what its ID
    translation table does on silicon, being the identity everywhere the
    physical grid lives. Concretely: Wormhole DRAM is never translated and goes
    on being addressed by its mirror, and every tile's ``NOC_NODE_ID`` — the
    per-NoC *physical* node ID, which is that mirror on NoC 1 — goes on
    resolving to itself.
    """
    device = wormhole_translated
    for tile in device.tile_directory.values():
        if not getattr(tile, "register_noc1_mirror", True):
            continue  # eth, which has never had a mirror key
        nui1 = tile.get_noc_nui(1)
        mirror = device.noc1_mirror(
            getattr(tile, "noc1_endpoint_coord", None) or tile.get_noc_nui(0).id_pair
        )
        assert resolved_nui(device.noc_1_directory[mirror]) is nui1, mirror


def test_blackhole_drops_its_physical_noc1_keys(blackhole_translated):
    """Where they are ambiguous they go, and nothing is left holding one.

    ``(16-x, 11-y)`` of one Blackhole worker is the translated coordinate of
    another, so a mirror key there is not dead weight but a live misroute. The
    only mirrors Blackhole ever registered were DRAM's; under translation its
    bank table emits translated DRAM coords instead, so they are gone.
    """
    device = blackhole_translated
    mirrors = {
        device.noc1_mirror(coord)
        for tile in device.tile_directory.values()
        for coord in (
            (getattr(tile, "noc1_endpoint_coord", None) or tile.get_noc_nui(0).id_pair),
            *tile.noc_aliases,
        )
    }
    # A mirror cell may still be a key — as some *other* tile's translated
    # coord. What must not happen is a key resolving to the tile it mirrors.
    for cell in mirrors:
        entry = device.noc_1_directory.get(cell)
        if entry is None:
            continue
        owner = device.tile_directory.get(cell)
        assert owner is not None, cell
        assert resolved_nui(entry) is owner.get_noc_nui(1), cell


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
