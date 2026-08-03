"""Constructor parity between the architectures.

Twice now a **device-level facility** has been wired in one architecture's
``__init__`` and not the other's, and both times the symptom was silence rather
than a failure:

* ``enable_from_env`` (structured tracing) was called only from ``wormhole.py``,
  so a Blackhole device ignored every ``TT_SIM_TRACE_*`` env var;
* the progress watchdog was Wormhole-only, so a wedged Blackhole kernel hung
  with no ``[DEADLOCK]`` diagnostic at all — on the architecture under the most
  active development.

The structural answer is that ``TT_Device`` now owns the whole construction
sequence (see its docstring) and an architecture supplies only per-arch
hardware facts. These tests are the guard on that: they fail if an
architecture starts doing device-level work itself, if the two constructors
produce differently-shaped devices, or if a facility stops being wired for
everyone. The rule they encode:

    a difference between the architectures must be a *hardware* difference.

The one standing exception is Wormhole's ethernet tiles: Blackhole's are 2-core
and not modelled yet (Phase 6 of ``docs/plans/blackhole-support.md``).
"""

import ast
import inspect

import pytest

from tt_sim.device import blackhole as blackhole_module
from tt_sim.device import wormhole as wormhole_module
from tt_sim.device.blackhole import Blackhole
from tt_sim.device.deadlock import DeadlockDetector
from tt_sim.device.tt_device import DeviceTileDiagnostics, TT_Device
from tt_sim.device.wormhole import Wormhole

ARCH_MODULES = {"wormhole": wormhole_module, "blackhole": blackhole_module}
DEVICES = pytest.mark.parametrize(
    "device_class", [Wormhole, Blackhole], ids=lambda c: c.__name__
)

# A second worker coord to lazily materialise, per architecture. Wormhole's is
# a unified coord (physical (2, 1)); Blackhole's is a physical NoC 0 coord.
SECOND_TENSIX = {Wormhole: (19, 18), Blackhole: (2, 2)}

# Methods that implement device-level facilities rather than hardware facts.
# An architecture that overrides one of these has forked the sequence and can
# drift from the other — which is exactly how both past bugs happened.
SHARED_FACILITIES = (
    "_begin_construction",
    "_build_dram_tiles",
    "_build_tensix_tile",
    "add_tensix_tile",
    "register_tensix_tile",
    "_register_tile_internals",
    "_noc1_mirror",
    "reset",
    "reset_tile",
    "shutdown",
    "read",
    "write",
)

# Symbols whose presence in an architecture module means that architecture is
# wiring a device-level facility itself instead of inheriting it.
FACILITY_SYMBOLS = (
    "enable_from_env",
    "DeadlockDetector",
    "deadlock_config_from_env",
)


def _imported_names(module):
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )
    return names


@pytest.mark.parametrize("arch", sorted(ARCH_MODULES))
def test_no_architecture_wires_a_device_facility_itself(arch):
    leaked = _imported_names(ARCH_MODULES[arch]) & set(FACILITY_SYMBOLS)
    assert not leaked, (
        f"{arch}.py imports {sorted(leaked)} — device-level facilities belong "
        f"in TT_Device so every architecture gets them. Wiring one here is how "
        f"tracing and the deadlock watchdog each went missing on Blackhole."
    )


@DEVICES
def test_no_architecture_overrides_a_shared_facility(device_class):
    overridden = [
        name
        for name in SHARED_FACILITIES
        if getattr(device_class, name) is not getattr(TT_Device, name)
    ]
    assert not overridden, (
        f"{device_class.__name__} overrides {overridden}; these are "
        f"arch-agnostic. Per-arch hardware facts go in ``tensix_tile_class`` / "
        f"``dram_tile_class`` / ``_tensix_physical_coord`` / the ArchProfile."
    )


def test_both_constructors_produce_the_same_shaped_device():
    """The blunt guard: an attribute one architecture sets and the other does
    not is a facility one of them is missing."""
    assert set(vars(Wormhole())) == set(vars(Blackhole()))


@DEVICES
def test_every_architecture_wires_the_progress_watchdog(device_class):
    device = device_class()
    detector = device.deadlock_detector
    assert isinstance(detector, DeadlockDetector)
    assert device.clocks[0].on_tick == detector.tick
    # Every tile the device built is watched, or a stall on it is invisible.
    watched = {coord for coord, _tile, _cores in detector.tile_cores}
    assert watched == {tile.get_coord_pair() for tile in device.tensix_tiles}
    assert detector.dram_tiles == device.dram_tiles


@DEVICES
def test_watchdog_honours_the_env_switch_on_every_architecture(
    device_class, monkeypatch
):
    monkeypatch.setenv("TT_SIM_DEADLOCK", "0")
    device = device_class()
    assert not device.deadlock_detector.enabled
    # Unwired rather than wired-and-returning-immediately, so disabling it
    # costs nothing per cycle.
    assert device.clocks[0].on_tick is None


@DEVICES
def test_every_architecture_enables_tracing_from_the_environment(
    device_class, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        "tt_sim.device.tt_device.enable_from_env",
        lambda device=None: calls.append(device),
    )
    device = device_class()
    # Once before tiles exist (so no construction event is missed), once with
    # the assembled device (the state-dump writer polls it).
    assert calls == [None, device]


@DEVICES
def test_lazy_materialisation_wires_a_tile_into_everything(device_class):
    """The wire bridge's ``add_tensix_tile`` path must reach every registry
    ``__init__`` reaches — the watchdog included."""
    device = device_class()
    coord = SECOND_TENSIX[device_class]
    before_clocks = len(device.clocks[0]._tile_clocks)
    before_resets = len(device.resets[0].reset_items)

    tile = device.add_tensix_tile(coord)

    assert device.tile_directory[coord] is tile
    assert tile in device.tensix_tiles
    assert device.noc_0_directory[
        tile.get_noc_nui(0).get_id_pair()
    ] is tile.get_noc_nui(0)
    assert tile.get_noc_nui(1).get_id_pair() in device.noc_1_directory
    assert len(device.clocks[0]._tile_clocks) == before_clocks + 1
    assert len(device.resets[0].reset_items) > before_resets
    assert tile.clock is not None
    watched = {c for c, _tile, _cores in device.deadlock_detector.tile_cores}
    assert coord in watched, "lazily-materialised tile is invisible to the watchdog"


@DEVICES
def test_diagnostic_flags_reach_every_core_on_every_architecture(device_class):
    flags = DeviceTileDiagnostics(
        brisc_diagnostics=True,
        ncrisc_diagnostics=True,
        trisc0_diagnostics=True,
        trisc1_diagnostics=True,
        trisc2_diagnostics=True,
        noc0_diagnostics=True,
        noc1_diagnostics=True,
    )
    device = device_class(flags)
    # Both the tiles built by __init__ and those materialised later.
    tiles = list(device.tensix_tiles) + [
        device.add_tensix_tile(SECOND_TENSIX[device_class])
    ]
    for tile in tiles:
        for core in (tile.brisc, tile.ncrisc, tile.trisc0, tile.trisc1, tile.trisc2):
            assert core.snoop
        assert tile.get_noc_nui(0).snoop
        assert tile.get_noc_nui(1).snoop


@DEVICES
def test_dram_tiles_follow_the_profile_on_every_architecture(device_class):
    """The shared DRAM builder reads every profile field, so a field only one
    architecture populates today (the NoC 1 worker endpoint) is honoured for
    whoever populates it next."""
    device = device_class()
    profile = device.profile
    assert [tile.get_coord_pair() for tile in device.dram_tiles] == list(
        profile.dram_channel_unified_coords
    )
    expected = profile.dram_channel_physical_noc1_coords or (
        (None,) * len(device.dram_tiles)
    )
    assert [tile.noc1_endpoint_coord for tile in device.dram_tiles] == list(expected)
