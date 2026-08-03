"""Tests for the event-driven pump.

See ``docs/plans/event-driven-pump.md``. Two properties, one per phase:

* **Phase 1 (dormancy)** — a tile whose every component reports
  :meth:`Clockable.is_clock_idle` stops being ticked, and each of the three
  external stimuli brings it back.
* **Phase 4 (the event queue)** — a component can name the cycle at which it
  next needs attention, and the pump jumps straight there without visiting
  anything in between. The tests pin down that striding is *observationally*
  identical to the cycle-by-cycle loop: the same components are ticked on the
  same cycles, and ``run(N)`` still advances simulated time by exactly ``N``.
"""

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.clock import Clockable, MultiTileClock, TileClock
from tt_sim.device.tt_device import DeviceTileDiagnostics
from tt_sim.network.tt_noc import NUI
from tt_sim.pe.tensix.backends.backend_base import TensixBackendUnit

TENSIX_COORD = (1, 2)
SOFT_RESET_ADDR = 0xFFB121B0
WALL_CLOCK_L_ADDR = 0xFFB121F0


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


# --------------------------------------------------------------------------
# Phase 4 — components that name a deadline, and a pump that jumps to it.
# --------------------------------------------------------------------------


class _ArmedClockable(Clockable):
    """Clockable that needs attention only on an explicit schedule."""

    def __init__(self, schedule):
        self.schedule = sorted(schedule)
        self.ticked = []

    def clock_tick(self, cycle_num):
        self.ticked.append(cycle_num)

    def next_wake_cycle(self, cycle_num):
        for when in self.schedule:
            if when > cycle_num:
                return when
        return None


def _armed_pump(stride):
    a = _ArmedClockable([10, 40])
    b = _ArmedClockable([25])
    pump = MultiTileClock(stride=stride)
    pump.add_tile_clock(TileClock([a], next_wake=a.next_wake_cycle))
    pump.add_tile_clock(TileClock([b], next_wake=b.next_wake_cycle))
    return pump, a, b


def test_pump_strides_to_the_earliest_armed_deadline():
    pump, a, b = _armed_pump(stride=True)

    pump.run(50)

    # Cycle 0 always runs (clocks start awake); after that the pump visits
    # only the cycles some component asked for, and each tile only on its own.
    assert a.ticked == [0, 10, 40]
    assert b.ticked == [0, 25]
    assert pump.clock_tick_num == 50
    assert pump.stride_skipped_cycles == 50 - 4


def test_striding_is_observationally_identical_to_ticking_every_cycle():
    """The Phase 4 acceptance test: same ticks, same cycle count, any batching.

    Batching matters because the wire bridge drives the pump in
    ``TT_SIM_CYCLES_PER_POLL``-sized calls — a stride must never run past the
    end of the window it was given, or the host's next poll lands late.
    """

    def drive(stride, batch):
        pump, a, b = _armed_pump(stride)
        for _ in range(50 // batch):
            pump.run(batch)
        return pump.clock_tick_num, a.ticked, b.ticked

    baseline = drive(stride=False, batch=50)
    assert baseline == (50, [0, 10, 40], [0, 25])
    for stride in (False, True):
        for batch in (1, 2, 5, 10, 25, 50):
            assert drive(stride, batch) == baseline, (stride, batch)


def test_stride_accounting_adds_up_to_every_cycle():
    pump, _a, _b = _armed_pump(stride=True)

    pump.run(50)

    for tile_clock in pump._tile_clocks:
        ticked = len(tile_clock.clock_items[0].ticked)
        assert ticked + tile_clock.dormant_cycles == 50


def test_a_tile_clock_reports_when_it_next_needs_attention():
    armed = _ArmedClockable([10])
    clock = TileClock([armed], next_wake=armed.next_wake_cycle)

    clock.clock_tick(0)

    assert not clock.awake
    assert clock.wake_at == 10
    assert clock.next_event_cycle(0) == 10
    # An external wake beats an armed deadline.
    clock.wake()
    assert clock.next_event_cycle(0) == 1


def test_backend_unit_occupancy_defers_retire_and_arms_the_pump():
    """A unit says "I am busy for N cycles" and both halves follow."""
    unit = TensixBackendUnit(None, {}, "matrix")
    unit.next_instruction.append(("instruction", 0))

    unit.occupy_for(100, 8)
    assert unit.busy_until == 108

    for cycle in range(100, 108):
        unit.clock_tick(cycle)
        # Occupied: nothing retires, and the pump is told to skip to 108.
        assert len(unit.next_instruction) == 1
        assert unit.next_wake_cycle(cycle) == 108

    unit.next_instruction.clear()
    unit.clock_tick(108)
    assert unit.busy_until is None
    assert unit.next_wake_cycle(108) is None


def test_backend_unit_occupancy_is_dormant_until_a_cost_table_sets_it():
    unit = TensixBackendUnit(None, {}, "matrix")

    # A one-cycle cost is the status quo, so nothing is armed.
    unit.occupy_for(100, 1)
    assert unit.busy_until is None
    assert unit.next_wake_cycle(100) is None

    unit.next_instruction.append(("instruction", 0))
    assert unit.next_wake_cycle(100) == 101


def _wall_clock(cycles, monkeypatch, stride):
    monkeypatch.setenv("TT_SIM_PUMP_STRIDE", "1" if stride else "0")
    device = _blackhole_with_tensix()
    device.run(cycles)
    raw = bytes(device.read(TENSIX_COORD, WALL_CLOCK_L_ADDR, 4))
    return int.from_bytes(raw, "little")


def test_wall_clock_is_sampled_lazily_and_unaffected_by_striding(monkeypatch):
    """Phase 2: the wall clock no longer needs a tick to stay correct.

    It used to be latched by ``TensixTileControl.clock_tick`` every cycle of
    every tile. Reading it from the tile's clock on demand gives the identical
    value — the last cycle executed — whether or not the pump ticked, which is
    what makes Phase 4's skipping possible at all.
    """
    strided = _wall_clock(5000, monkeypatch, stride=True)
    every_cycle = _wall_clock(5000, monkeypatch, stride=False)

    assert strided == every_cycle == 4999


def test_a_fully_dormant_device_costs_nothing_per_cycle(monkeypatch):
    monkeypatch.setenv("TT_SIM_PUMP_STRIDE", "1")
    device = _blackhole_with_tensix()

    device.run(1_000_000)

    pump = device.clocks[0]
    assert pump.clock_tick_num == 1_000_000
    # One tick per tile settles it; the remaining ~10^6 cycles are jumped.
    assert pump.stride_skipped_cycles == 1_000_000 - 1
    for tile in device.tile_directory.values():
        assert tile.clock.dormant_cycles == 1_000_000 - 1
