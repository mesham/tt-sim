"""Tests for the event-driven pump's tile-level dormancy.

See ``docs/plans/event-driven-pump.md``. The property under test is narrow:
a tile whose every component reports :meth:`Clockable.is_clock_idle` stops
being ticked, and each of the three external stimuli brings it back.
"""

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.clock import Clockable, TileClock
from tt_sim.device.tt_device import DeviceTileDiagnostics
from tt_sim.network.tt_noc import NUI

TENSIX_COORD = (1, 2)
SOFT_RESET_ADDR = 0xFFB121B0


class _Counter(Clockable):
    """Clockable whose idleness the test drives directly."""

    def __init__(self, idle=True):
        self.ticks = 0
        self.idle = idle

    def clock_tick(self, cycle_num):
        self.ticks += 1

    def is_clock_idle(self):
        return self.idle


def test_tile_clock_stops_ticking_gated_items_when_quiescent():
    gated = _Counter()
    always = _Counter()
    seen = []
    clock = TileClock(
        [gated],
        always=[always],
        quiescent=gated.is_clock_idle,
        on_tick=seen.append,
    )

    clock.run(10)

    # First cycle ticks (the tile starts awake), then the probe puts it under.
    assert gated.ticks == 1
    assert clock.dormant_cycles == 9
    # The always-list and the on_tick hook see every cycle regardless.
    assert always.ticks == 10
    assert seen == list(range(10))


def test_tile_clock_wakes_on_demand():
    gated = _Counter()
    clock = TileClock([gated], quiescent=gated.is_clock_idle)

    clock.run(5)
    assert gated.ticks == 1

    clock.wake()
    clock.run(5)
    assert gated.ticks == 2


def test_tile_clock_never_sleeps_while_a_component_is_busy():
    busy = _Counter(idle=False)
    clock = TileClock([busy], quiescent=busy.is_clock_idle)

    clock.run(7)

    assert busy.ticks == 7
    assert clock.dormant_cycles == 0


def test_clockable_default_is_never_idle():
    """The conservative default: opting out keeps a tile awake forever."""

    class NotOptedIn(Clockable):
        def clock_tick(self, cycle_num):
            pass

    assert NotOptedIn().is_clock_idle() is False


def _blackhole_with_tensix():
    device = Blackhole(DeviceTileDiagnostics())
    if TENSIX_COORD not in device.tile_directory:
        device.add_tensix_tile(TENSIX_COORD)
    return device


def test_device_goes_fully_dormant_with_every_core_in_reset():
    device = _blackhole_with_tensix()

    device.run(200)

    for tile in device.tile_directory.values():
        assert tile.clock.dormant_cycles > 0, (
            f"tile {tile.get_coord_pair()} never slept"
        )
        assert not tile.clock.awake


def test_host_write_wakes_a_dormant_tile():
    device = _blackhole_with_tensix()
    device.run(200)
    tile = device.tile_directory[TENSIX_COORD]
    assert not tile.clock.awake

    device.write(TENSIX_COORD, 0x100, b"\x01\x02\x03\x04")

    assert tile.clock.awake


def test_deasserting_soft_reset_wakes_and_keeps_the_tile_awake():
    device = _blackhole_with_tensix()
    device.run(200)
    tile = device.tile_directory[TENSIX_COORD]
    before = tile.clock.dormant_cycles

    device.deassert_soft_reset(TENSIX_COORD)
    device.run(20)

    # BRISC is now running, so the tile must not have slept again.
    assert tile.clock.dormant_cycles == before
    assert tile.brisc.soft_active


def test_noc_transmit_wakes_a_dormant_tile():
    device = _blackhole_with_tensix()
    device.run(200)
    tile = device.tile_directory[TENSIX_COORD]
    assert not tile.clock.awake

    request = NUI.NoCDataRequest(
        0x40,
        NUI.NoCDataRequest.DataRequestAction.WRITE,
        4,
        tile.noc0_router.get_id_pair(),
        0,
        b"\xde\xad\xbe\xef",
    )
    tile.noc0_router.transmit(request)

    assert tile.clock.awake
    device.run(2)
    assert bytes(device.read(TENSIX_COORD, 0x40, 4)) == b"\xde\xad\xbe\xef"
