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

Requests are **not** harmless, and the second half of this file measures how
far that goes. A canonical worker coord taken by somebody else's mirror is a
live Tensix nothing on NoC 1 can reach: the impostor accepts the write and
ACKs it, so ``noc_async_write_barrier`` returns and neither end learns
anything. With every functional worker built that is 56 of Wormhole's 80
coords, and on-demand materialisation manufactures collisions that a fixed grid
would not have. None of that is fixed here — on Wormhole the two coordinate
conventions genuinely collide — but every instance is named at the moment it is
created (``tt_sim/network/noc_shadow.py``).

**Blackhole is not Wormhole, and the difference is the trap this file guards.**
Blackhole registers no Tensix mirror alias at all, because tt-metal's
``virtual_noc0_coordinate`` early-outs on that architecture and so has never
emitted a mirrored worker coord there; its census is 6 of 140, all of them a
DRAM mirror landing on a worker column. It is tempting to conclude the Tensix
mirrors are dead weight *everywhere* and delete them on both arches — that
would break every Wormhole L1-sharded-buffer program, whose ``l1_bank_to_noc_xy``
really is mirrored on NoC 1. ``test_the_tensix_mirror_alias_policy_is_per_arch``
and its two neighbours exist so that change fails loudly rather than silently.

No tt-metal, no socket, no oracle — the cheapest guard available for this.
"""

import pytest

from tt_sim.arch import BLACKHOLE_PROFILE, WORMHOLE_PROFILE
from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.noc_coords import WormholeNocCoords
from tt_sim.network.noc_shadow import POLICY_ENV, NoC1ShadowError
from tt_sim.network.tt_noc import NUI

_OUTSTANDING_ID_0 = NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0

# Physical NoC 0 coords of the 4x5 worker sub-block tt-metal selects with
# ``TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,4`` — the smallest grid the live
# multi-core examples actually ran on, and the one that crashed.
_WH_4X5 = [(x, y) for x in (1, 2, 3, 4) for y in (1, 2, 3, 4, 5)]

# Two Blackhole workers that mirror onto each other: (16-1, 11-2) == (15, 9).
# They used to be *transposed* on NoC 1; Blackhole no longer registers Tensix
# mirrors, so each now keeps its own cell. Kept as the pair most likely to
# regress if that ever changes.
_BH_MIRRORED_PAIR = [(1, 2), (15, 9)]

#: Blackhole workers a DRAM tile's NoC 1 mirror lands on — the 6 shadows that
#: survive, and must, because ``dram_bank_to_noc_xy`` addresses DRAM by mirror
#: on both arches.
_BH_DRAM_SHADOWED = {(7, 4), (7, 7), (7, 10), (16, 4), (16, 7), (16, 10)}

#: Wormhole's unified worker band, and so its 80 functional workers.
_WH_UNIFIED_TO_PHYSICAL = {
    (ux, uy): Wormhole.physical_noc0_coord_from_unified_worker((ux, uy))
    for ux in range(18, 26)
    for uy in range(16, 26)
}
_WH_PHYSICAL_TO_UNIFIED = {v: k for k, v in _WH_UNIFIED_TO_PHYSICAL.items()}

#: Blackhole's 140 functional workers, per ``driver/blackhole/soc_descriptor.yaml``
#: — columns 1-7 and 10-16 (8 and 9 are ARC / router-only), rows 2-11.
_BH_WORKERS = [
    (x, y) for x in list(range(1, 8)) + list(range(10, 17)) for y in range(2, 12)
]

#: The 8x8 compute grid ``TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=7,7`` selects,
#: as physical rows. Wormhole has no worker row 6.
_WH_8X8_ROWS = (1, 2, 3, 4, 5, 7, 8, 9)

_L1_SRC = 0x20000
_L1_DST = 0x21000


def _wormhole_with_workers(physical_coords, **kwargs):
    """A Wormhole whose only workers are ``physical_coords``.

    Built through the constructor rather than ``add_tensix_tile`` so the
    registration order is the one a device with a fixed grid gets.
    """
    return Wormhole(
        tensix_coords=[_WH_PHYSICAL_TO_UNIFIED[c] for c in physical_coords], **kwargs
    )


def _wormhole_with_4x5_grid():
    device = Wormhole()
    workers = {}
    for physical in _WH_4X5:
        unified = _WH_PHYSICAL_TO_UNIFIED[physical]
        if unified not in device.tile_directory:
            device.add_tensix_tile(unified)
        workers[physical] = device.tile_directory[unified]
    return device, workers


def _blackhole_with_mirrored_pair():
    device = Blackhole()
    workers = {}
    for physical in _BH_MIRRORED_PAIR:
        if physical not in device.tile_directory:
            device.add_tensix_tile(physical)
        workers[physical] = device.tile_directory[physical]
    return device, workers


def _kind(tile):
    """The tile-kind letter its NUI carries (``D`` for DRAM, ``T`` for Tensix)."""
    return tile.get_noc_nui(0).tile_kind


def _shadow_census(device, physical_to_tile):
    """``{worker physical coord: owning tile}`` for every shadowed worker."""
    by_nui = {}
    for tile in device.tile_directory.values():
        for noc in (0, 1):
            by_nui[id(tile.get_noc_nui(noc))] = tile
    census = {}
    for physical, tile in physical_to_tile.items():
        entry = device.noc_1_directory.get(physical)
        if entry is not tile.get_noc_nui(1):
            census[physical] = by_nui[id(entry)]
    return census


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


def _run_until_settled(device, nui, budget=4000):
    """Pump until every request ``nui`` issued has been answered.

    Deliberately not a fixed cycle count. A NoC round trip costs two cycles
    with the per-hop latency model off and a few hundred with it on
    (``TT_SIM_COST_MODEL``, see ``tt_sim/network/noc_cost_model_test.py``), so
    a hardcoded budget makes this test a *timing pin* on a routing property
    that has nothing to do with timing. Waiting on the outstanding-request
    FIFOs instead is what the test actually means, and it fails the same way
    (a response delivered to the wrong tile never drains this one) rather than
    silently reading L1 too early.
    """
    for _ in range(budget):
        if all(not fifo for fifo in nui.outstanding_noc_requests.values()):
            return
        device.run(1)
    raise AssertionError(
        f"NUI {nui.id_pair} still awaiting {nui.outstanding_noc_requests} "
        f"after {budget} cycles"
    )


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
    _run_until_settled(device, tile.noc1_router)

    _set_coord(initiator, "target", dram_coord)
    initiator.target_addr_low = dram_address
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(payload)
    initiator.ctrl = 0  # mode 0 = read
    initiator.cmd_ctrl = 1
    initiator.initiate()
    _run_until_settled(device, tile.noc1_router)

    return bytes(device.read(unified, _L1_DST, len(payload)))


def _assert_settled(nui):
    """Every request the NUI issued has been answered, exactly once."""
    for trid, pending in nui.outstanding_noc_requests.items():
        assert not pending, f"NUI {nui.id_pair} still awaiting trid {trid}: {pending}"
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


def test_blackhole_noc1_directory_keeps_a_mirrored_worker_pair_apart():
    """Two Blackhole workers that mirror onto each other each keep their cell.

    **This assertion used to say the opposite** — that ``(1, 2)`` resolved to
    ``(15, 9)``'s NUI and vice versa — and called it characterisation. It was
    characterising a bug: Blackhole never needed those mirror aliases, so the
    transposition was pure loss, and pinning it as expected behaviour is how it
    survived. Now the pair is simply routed correctly.
    """
    device, workers = _blackhole_with_mirrored_pair()

    for physical, tile in workers.items():
        assert device.noc_1_directory[physical] is tile.noc1_router
        assert device.noc_0_directory[physical] is tile.noc0_router

    # A lone worker claims no mirror cell at all, which is the mechanism: the
    # pair above is only routed correctly because neither alias exists.
    lone = Blackhole(tensix_coords=[(3, 5)])
    assert lone.noc1_mirror((3, 5)) not in lone.noc_1_directory


def test_wormhole_8x8_grid_row_sweep_pins_the_affected_workers():
    """The static reproducer from the bug report, row by row.

    One row of four workers at a time (``x in 1..4``, the columns a program
    on the 8x8 compute grid fills first), swept over the grid's eight physical
    rows. Column 4 mirrors onto Wormhole's DRAM column 5, so rows whose DRAM
    sub-endpoint exists lose their ``(4, y)`` worker; the rest are clean. This
    predicts the live failure set exactly and costs a second, so it is the
    cheapest possible pin on the affected set.
    """
    shadowed = set()
    clean = set()
    for row in _WH_8X8_ROWS:
        physical = [(x, row) for x in (1, 2, 3, 4)]
        device = _wormhole_with_workers(physical)
        census = _shadow_census(
            device,
            {c: device.tile_directory[_WH_PHYSICAL_TO_UNIFIED[c]] for c in physical},
        )
        assert all(_kind(t) == "D" for t in census.values()), census
        shadowed |= set(census)
        if not census:
            clean.add(row)

    assert shadowed == {(4, 2), (4, 3), (4, 4), (4, 8), (4, 9)}
    assert clean == {1, 5, 7}


def test_wormhole_full_worker_grid_shadow_census():
    """All 80 workers: 56 unreachable on NoC 1, 48 of them behind each other."""
    physical = sorted(_WH_UNIFIED_TO_PHYSICAL.values())
    device = _wormhole_with_workers(physical)
    census = _shadow_census(
        device,
        {c: device.tile_directory[_WH_PHYSICAL_TO_UNIFIED[c]] for c in physical},
    )

    by_dram = {c for c, tile in census.items() if _kind(tile) == "D"}
    by_worker = {c for c, tile in census.items() if tile.is_tensix}
    assert len(census) == 56
    assert by_dram | by_worker == set(census)
    assert by_dram == {
        (4, 2),
        (4, 3),
        (4, 4),
        (4, 8),
        (4, 9),
        (4, 10),
        (9, 4),
        (9, 10),
    }
    assert len(by_worker) == 48
    # Worker-on-worker shadowing is always a transposed pair: each worker's
    # mirror is the other's canonical coord, so they swap and neither is
    # reachable. That is why the count is even.
    for coord in by_worker:
        assert device.noc1_mirror(coord) in by_worker

    # Every shadowed coord was reported, and nothing else was.
    assert set(device.shadow_reporter.shadowed) == set(census)


def test_blackhole_full_worker_grid_shadow_census():
    """All 140 workers: 6 unreachable on NoC 1, every one of them behind DRAM.

    **This test used to assert 102, with 96 worker-behind-worker.** Those 96
    were pure loss — Blackhole's ``l1_bank_to_noc_xy`` is byte-identical on its
    two NoC halves, so nothing there ever addresses a worker by its mirror, and
    registering the alias only evicted the worker that owned the cell.
    Dropping the Tensix aliases (``ArchProfile.noc1_tensix_mirror_aliases``)
    clears all 96 and leaves the 6 that are not a bug.
    """
    device = Blackhole(tensix_coords=_BH_WORKERS)
    census = _shadow_census(device, {c: device.tile_directory[c] for c in _BH_WORKERS})

    by_dram = {c for c, tile in census.items() if _kind(tile) == "D"}
    by_worker = {c for c, tile in census.items() if tile.is_tensix}
    assert len(census) == 6
    # Blackhole's DRAM columns are 0 and 9; a DRAM tile's NoC 1 endpoint
    # mirrors onto worker columns 16 and 7 respectively. These must NOT be
    # cleared along with the Tensix ones: ``dram_bank_to_noc_xy`` genuinely is
    # mirrored on NoC 1 on both arches, so the DRAM aliases are load-bearing
    # and these six workers stay shadowed until translation lands.
    assert by_dram == _BH_DRAM_SHADOWED
    assert by_worker == set()
    assert set(device.shadow_reporter.shadowed) == set(census)

    # NoC 0 was never ambiguous and still isn't.
    assert all(
        device.noc_0_directory[c] is device.tile_directory[c].noc0_router
        for c in _BH_WORKERS
    )


# ---------------------------------------------------------------------------
# The per-arch mirror-alias policy, and the trap around it.
#
# The obvious-looking generalisation — "Blackhole proved the Tensix mirror
# aliases are dead weight, so drop them everywhere" — is wrong, and wrong in a
# way no Blackhole test can catch. On Wormhole ``l1_bank_to_noc_xy``'s NoC 1
# half really is ``mirror(NoC 0)``, so a kernel using an L1-sharded or
# interleaved-L1 buffer addresses its peers by mirror and nothing else; delete
# the aliases there and those programs write into empty grid cells. The three
# tests below pin both halves of the asymmetry by value, so a change that
# unifies them fails here rather than in a multi-core program's answer.
# ---------------------------------------------------------------------------


def test_the_tensix_mirror_alias_policy_is_per_arch():
    """The two arches must disagree. Making them agree is the trap."""
    assert WORMHOLE_PROFILE.noc1_tensix_mirror_aliases is True
    assert BLACKHOLE_PROFILE.noc1_tensix_mirror_aliases is False


def test_wormhole_workers_stay_reachable_by_their_mirror_key():
    """Wormhole's Tensix mirror aliases are load-bearing, so all 80 exist.

    ``l1_bank_to_noc_xy`` on Wormhole emits ``mirror(canonical)`` for every
    worker on NoC 1 (measured off the wire), so this *is* the key an
    L1-sharded buffer's ``TensorAccessor`` puts in the command register. If
    this test starts failing, Wormhole's sharded programs are writing into
    empty cells whatever else passes.
    """
    physical = sorted(_WH_UNIFIED_TO_PHYSICAL.values())
    device = _wormhole_with_workers(physical)

    for coord in physical:
        tile = device.tile_directory[_WH_PHYSICAL_TO_UNIFIED[coord]]
        assert device.noc_1_directory[device.noc1_mirror(coord)] is tile.noc1_router


def test_blackhole_workers_are_never_reachable_by_a_mirror_key():
    """The converse on Blackhole: no worker answers at its mirror coord.

    A Blackhole mirror cell is either empty or owned by whichever *other*
    worker holds it canonically — never by the worker it mirrors. Nothing on
    Blackhole emits such a key, so an entry here is an alias that can only
    evict a live worker.
    """
    device = Blackhole(tensix_coords=_BH_WORKERS)

    for coord in _BH_WORKERS:
        tile = device.tile_directory[coord]
        mirror = device.noc1_mirror(coord)
        assert device.noc_1_directory.get(mirror) is not tile.noc1_router


# ---------------------------------------------------------------------------
# Failing loudly: the shadow is reported at the moment it is created.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [lambda: Wormhole(), lambda: Blackhole()],
    ids=["wormhole", "blackhole"],
)
def test_a_configuration_with_no_shadow_reports_nothing(build, monkeypatch):
    """The default single-worker device is clean, under the strictest policy.

    The guard is only worth anything if it stays quiet on the configurations
    that work — every single-tile example, and every replay guard.
    """
    monkeypatch.setenv(POLICY_ENV, "error")
    device = build()
    assert device.shadow_reporter.shadowed == {}


def test_a_shadowed_worker_is_named_at_registration(monkeypatch):
    monkeypatch.setenv(POLICY_ENV, "error")
    with pytest.raises(NoC1ShadowError, match=r"\(4, 2\)"):
        _wormhole_with_workers([(1, 2), (2, 2), (3, 2), (4, 2)])


def test_the_shadow_report_names_both_tiles():
    device = _wormhole_with_workers([(4, 2)])
    (message,) = device.shadow_reporter.shadowed.values()
    assert "TensixTile(4, 2)" in message
    assert "DRAMTile(5, 2)" in message


def test_blackhole_worker_on_worker_shadowing_is_no_longer_reported(
    monkeypatch, capsys
):
    """The reporter must be silent for the case Blackhole no longer has.

    Under the strictest policy, the full 140-worker grid — which used to raise
    on the first of 96 worker-behind-worker collisions — now reports only the
    six DRAM ones, and prints not a word about any worker pair.
    """
    monkeypatch.setenv(POLICY_ENV, "warn")
    device = Blackhole(tensix_coords=_BH_WORKERS)

    assert set(device.shadow_reporter.shadowed) == _BH_DRAM_SHADOWED
    err = capsys.readouterr().err
    for coord in _BH_DRAM_SHADOWED:
        assert f"{coord} is the canonical coord" in err
    # Every line names a DRAM impostor; not one names a Tensix one.
    assert "BlackholeTensixTile" in err  # ...as the *victim*
    assert err.count("resolves it to BlackholeDRAMTile") == len(_BH_DRAM_SHADOWED)
    assert "resolves it to BlackholeTensixTile" not in err


def test_blackhole_still_reports_a_dram_shadowed_worker(monkeypatch):
    """...and the other direction: the six that remain still fail loudly.

    ``dram_bank_to_noc_xy`` is mirrored on NoC 1 on Blackhole too, so DRAM's
    mirror aliases stay and these six workers stay unreachable. A change that
    generalised the Tensix rule to DRAM would silence this.
    """
    monkeypatch.setenv(POLICY_ENV, "error")
    with pytest.raises(NoC1ShadowError, match=r"\(7, 4\).*DRAMTile"):
        Blackhole(tensix_coords=[(7, 4)])


def test_blackhole_a_mirrored_worker_pair_is_clean_under_the_error_policy(monkeypatch):
    monkeypatch.setenv(POLICY_ENV, "error")
    device = Blackhole(tensix_coords=_BH_MIRRORED_PAIR)
    assert device.shadow_reporter.shadowed == {}


def test_a_cached_null_route_shadows_a_worker_built_afterwards():
    """A miss caches a ``NullEndpoint`` in the directory, and it outlives the
    miss: a worker built at that coord later can never claim its own cell, and
    every NoC 1 request to it is zero-filled or blackholed. Reported like any
    other shadow, naming the null route rather than a tile."""
    device = _wormhole_with_workers([(1, 1)])
    nui = device.tile_directory[_WH_PHYSICAL_TO_UNIFIED[(1, 1)]].get_noc_nui(1)
    nui.resolve_destination((2, 2))  # no miss hook installed: null-routed

    device.add_tensix_tile(_WH_PHYSICAL_TO_UNIFIED[(2, 2)])
    assert "NullEndpoint" in device.shadow_reporter.shadowed[(2, 2)]


def test_a_worker_added_after_construction_is_checked_both_ways(monkeypatch):
    """``add_tensix_tile`` must fire from either side of the collision.

    Which tile is the victim and which the impostor depends only on build
    order, and the wire bridge's order is the program's, not ours. Registering
    a worker into a cell a mirror already holds and registering a mirror over a
    cell a live worker holds are separate code paths, so both are pinned.
    """
    monkeypatch.setenv(POLICY_ENV, "error")

    # Victim second: DRAM's mirror is already at (4, 2) when the worker lands.
    device = Wormhole()
    with pytest.raises(NoC1ShadowError, match=r"\(4, 2\)"):
        device.add_tensix_tile(_WH_PHYSICAL_TO_UNIFIED[(4, 2)])

    # Impostor second: worker (1, 1) is live, and worker (8, 10)'s mirror is
    # exactly (1, 1). Both cells of the transposed pair are reported.
    device = _wormhole_with_workers([(1, 1)])
    with pytest.raises(NoC1ShadowError, match=r"\(8, 10\)|\(1, 1\)"):
        device.add_tensix_tile(_WH_PHYSICAL_TO_UNIFIED[(8, 10)])


def test_on_demand_materialisation_creates_a_shadow_construction_would_not():
    """The distinct runtime failure mode, and the one that is strictly worse.

    A device built with workers ``(1, 1)`` and ``(2, 2)`` resolves both
    correctly: neither is the other's mirror, and no DRAM mirror lands on
    either. But if ``(2, 2)`` is addressed over NoC 1 *before* it is
    materialised, the directory-miss hook reads the miss as a mirror, builds
    worker ``(7, 9)`` — which the program never asked for — and that worker
    then owns ``(2, 2)`` for the rest of the run. So on-demand materialisation
    does not merely inherit the construction-time hazard, it manufactures one.
    """
    clean = _wormhole_with_workers([(1, 1), (2, 2)])
    assert clean.shadow_reporter.shadowed == {}

    device = _wormhole_with_workers([(1, 1)])
    coord_map = dict(_WH_PHYSICAL_TO_UNIFIED)
    built = []

    def miss(noc_number, coord):
        # ``LazyTensixPool.on_directory_miss``, without the wire bridge.
        if noc_number == 1:
            mirrored = device.noc1_mirror(coord)
            if mirrored in coord_map:
                if coord in coord_map:
                    device.shadow_reporter.report_materialised_mirror(coord, mirrored)
                if coord_map[mirrored] not in device.tile_directory:
                    device.add_tensix_tile(coord_map[mirrored])
                    built.append(mirrored)
                if coord in device.noc_1_directory:
                    return
        if coord in coord_map and coord_map[coord] not in device.tile_directory:
            device.add_tensix_tile(coord_map[coord])
            built.append(coord)

    device.set_directory_miss_hook(miss)
    nui = device.tile_directory[_WH_PHYSICAL_TO_UNIFIED[(1, 1)]].get_noc_nui(1)
    resolved = nui.resolve_destination((2, 2))

    assert built == [(7, 9)]
    assert resolved is device.tile_directory[
        _WH_PHYSICAL_TO_UNIFIED[(7, 9)]
    ].get_noc_nui(1)
    assert (2, 2) in device.shadow_reporter.shadowed
    assert "(7, 9)" in device.shadow_reporter.shadowed[(2, 2)]


# ---------------------------------------------------------------------------
# Multicast rectangles that overrun the worker columns.
# ---------------------------------------------------------------------------


def _multicast(tile, rectangle, payload):
    """Issue one multicast write from ``tile`` over NoC 1 into ``rectangle``."""
    x_start, y_start, x_end, y_end = rectangle
    initiator = tile.noc1_router.request_initiators[0]
    initiator.ret_addr_hi = x_end | (y_end << 6) | (x_start << 12) | (y_start << 18)
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(payload)
    initiator.ctrl = 2 | (1 << 4) | (1 << 5)  # write, resp marked, broadcast
    initiator.cmd_ctrl = 1
    initiator.initiate()


def test_a_multicast_over_a_non_worker_column_names_the_gap_cells(capsys):
    """Blackhole's columns 8 and 9 are not workers; a rectangle spanning them
    is ACKed by cells the caller never counted, and ``noc_async_write_barrier``
    then waits on an equality that can never hold. tt-sim cannot see the
    kernel's ``num_dests`` (it reaches no command register), but it can see the
    cells nothing answers for, which is the same bug from the other end.
    """
    device = Blackhole(tensix_coords=[(2, 2), (10, 2)])
    tile = device.tile_directory[(2, 2)]
    _multicast(tile, (2, 2, 10, 2), b"\xa5" * 16)

    err = capsys.readouterr().err
    assert "multicast rectangle (2, 2)..(10, 2)" in err
    assert "9 cells" in err
    # Columns 3-7 have no tile in this device either, so the message names them
    # too; what matters is that the true gap column is called out.
    assert "(8, 2)" in err

    # One line per rectangle, however many packets take that route.
    _multicast(tile, (2, 2, 10, 2), b"\xa5" * 16)
    assert capsys.readouterr().err == ""


def test_a_multicast_that_lands_only_on_real_tiles_is_silent(capsys):
    device = Blackhole(tensix_coords=[(1, 2), (2, 2), (3, 2)])
    tile = device.tile_directory[(1, 2)]
    _multicast(tile, (1, 2, 3, 2), b"\xa5" * 16)
    assert "multicast rectangle" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The property: responses are unaffected by any of that.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, channel",
    [(_wormhole_with_4x5_grid, 3), (_blackhole_with_mirrored_pair, 0)],
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
        device.run(4)  # arrives next cycle: an inbound transmit is never delayed


#: The (device builder, DRAM channel) pairs the routing test is run over.
_ARCH_CASES = [(_wormhole_with_4x5_grid, 3), (_blackhole_with_mirrored_pair, 0)]


def main():
    test_wormhole_noc1_directory_shadows_worker_coords()
    test_blackhole_noc1_directory_keeps_a_mirrored_worker_pair_apart()
    test_wormhole_8x8_grid_row_sweep_pins_the_affected_workers()
    test_wormhole_full_worker_grid_shadow_census()
    test_blackhole_full_worker_grid_shadow_census()
    test_the_tensix_mirror_alias_policy_is_per_arch()
    test_wormhole_workers_stay_reachable_by_their_mirror_key()
    test_blackhole_workers_are_never_reachable_by_a_mirror_key()
    test_the_shadow_report_names_both_tiles()
    test_on_demand_materialisation_creates_a_shadow_construction_would_not()
    for build, channel in _ARCH_CASES:
        test_noc1_responses_return_to_the_issuing_worker(build, channel)
    test_a_request_needing_a_response_must_name_its_requester()
    print(
        "noc_routing_test OK: NoC 1's directory is still ambiguous on Wormhole "
        "(56 of 80 workers shadowed, DRAM over the x=4 column and workers over "
        "each other) and every instance is named at registration; Blackhole "
        "registers no Tensix mirror at all, so only its 6 DRAM shadows remain; "
        "and every NoC 1 response still lands on its requester"
    )


if __name__ == "__main__":
    main()
