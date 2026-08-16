"""The coordinate-convention guard, in both directions and on both arches.

A forgotten ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` is the one way translated
operation can go wrong quietly: the host emits untranslated coordinates, every
one of them misses tt-sim's directory, and the traffic lands on a ``NullCore``
that zero-fills reads and swallows writes. The run then finishes with wrong
numbers or hangs. So the guard is tested for what it catches *and* for what it
must not cry wolf about — a false positive here would break every existing
untranslated run.
"""

import pytest

from driver.blackhole.server import coords as bh_coords
from driver.wormhole.server import coords as wh_coords
from tt_sim.bridge import Fabric, install_convention_guard

ARCHES = {"wormhole": wh_coords, "blackhole": bh_coords}


def _make(coords_module, *, translated):
    alias, translated_only, untranslated_only = coords_module.wire_conventions()
    fabric = Fabric()
    caught = []
    install_convention_guard(
        fabric,
        translated=translated,
        wire_alias=alias if translated else {},
        foreign_coords=untranslated_only if translated else translated_only,
        reason="test",
        descriptor_hint=coords_module.CLUSTER_DESCRIPTOR_PATH,
        on_error=caught.append,
    )
    return fabric, caught, alias, translated_only, untranslated_only


@pytest.mark.parametrize("arch", ARCHES)
def test_untranslated_host_against_translated_sim_is_caught(arch):
    """The forgotten-env-var direction, which is the one that costs time."""
    fabric, caught, _, _, untranslated_only = _make(ARCHES[arch], translated=True)
    coord = sorted(untranslated_only)[0]
    fabric.read(coord, 0, 4)
    assert caught == [coord]


@pytest.mark.parametrize("arch", ARCHES)
def test_translated_host_against_untranslated_sim_is_caught(arch):
    """The other direction: descriptor exported, server started with
    ``TT_SIM_NOC_TRANSLATION=0``."""
    fabric, caught, _, translated_only, _ = _make(ARCHES[arch], translated=False)
    coord = sorted(translated_only)[0]
    fabric.write(coord, 0, b"\0\0\0\0")
    assert caught == [coord]


@pytest.mark.parametrize("arch", ARCHES)
def test_it_fires_once_not_once_per_message(arch):
    fabric, caught, _, _, untranslated_only = _make(ARCHES[arch], translated=True)
    for coord in sorted(untranslated_only)[:5]:
        fabric.read(coord, 0, 4)
    assert len(caught) == 1


@pytest.mark.parametrize("arch", ARCHES)
def test_the_right_convention_never_trips_it(arch):
    """Every coordinate the matching convention uses must pass through."""
    coords_module = ARCHES[arch]
    fabric, caught, alias, _, _ = _make(coords_module, translated=True)
    for coord in alias:
        fabric.read(coord, 0, 4)
    for coord in coords_module.DRAM_COORD_MAP:
        # Wormhole DRAM is untranslated and so is addressed identically in
        # both conventions; it must not be mistaken for the wrong one.
        if arch == "wormhole":
            fabric.read(coord, 0, 4)
    assert caught == []

    fabric, caught, _, _, _ = _make(coords_module, translated=False)
    for coord in coords_module.TENSIX_COORD_MAP:
        fabric.read(coord, 0, 4)
    for coord in coords_module.DRAM_COORD_MAP:
        fabric.read(coord, 0, 4)
    assert caught == []


@pytest.mark.parametrize("arch", ARCHES)
def test_the_two_conventions_are_disjoint(arch):
    """If they overlapped, the guard could not tell them apart at all."""
    _, translated_only, untranslated_only = ARCHES[arch].wire_conventions()
    assert translated_only & untranslated_only == frozenset()
    assert translated_only
    assert untranslated_only


@pytest.mark.parametrize("arch", ARCHES)
def test_wire_alias_maps_translated_coords_onto_the_tiles(arch):
    """Aliasing at the fabric edge is why nothing downstream had to change:
    a translated wire coord reaches the core registered under the physical one."""
    coords_module = ARCHES[arch]
    fabric, caught, alias, _, _ = _make(coords_module, translated=True)
    sentinel = object()
    for translated_coord, physical in alias.items():
        fabric.cores.clear()
        fabric.register(physical, sentinel)
        assert fabric._core(translated_coord) is sentinel
    assert caught == []


def test_translated_wormhole_workers_land_where_the_wire_says():
    """Spot-check against coordinates measured on the wire, not re-derived:
    logical (0,0) is physical 1-1 and translated (18, 18)."""
    alias, _, _ = wh_coords.wire_conventions()
    assert alias[(18, 18)] == (1, 1)
    assert alias[(25, 27)] == (9, 11)
    # Eth band, directly below the workers.
    assert alias[(18, 16)] == (1, 0)
    assert alias[(25, 17)] == (9, 6)


def test_translated_blackhole_dram_lands_where_the_wire_says():
    alias, _, _ = bh_coords.wire_conventions()
    # ch0's NoC 0 sub-endpoint, the coord the translated trace actually carries.
    assert alias[(17, 14)] == (0, 11)
    # ...and its NoC 1 sub-endpoint, the same tile.
    assert alias[(17, 13)] == (0, 11)
    assert alias[(18, 23)] == (9, 6)


@pytest.mark.parametrize("arch", ARCHES)
def test_the_server_factory_builds_the_device_in_the_mode_it_was_told(arch):
    """The mode is decided once, in ``server/__main__.py``, and handed to both
    the device and the guard — so a device keyed one way behind a guard set the
    other way is not a state this can reach."""
    if arch == "wormhole":
        from driver.wormhole.server.wh_device import make_device
    else:
        from driver.blackhole.server.bh_device import make_device

    assert make_device().tt_device.noc_translation is False
    assert make_device(noc_translation=True).tt_device.noc_translation is True


def test_blackhole_workers_are_the_same_coord_in_both_conventions():
    """Which is why Blackhole's discriminator has to be DRAM and eth."""
    _, translated_only, untranslated_only = bh_coords.wire_conventions()
    for coord in bh_coords.TENSIX_COORD_MAP:
        assert coord not in translated_only
        assert coord not in untranslated_only
