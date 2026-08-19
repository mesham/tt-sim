"""A multicast rectangle whose corners are in the wrong order for its NoC.

The rule and its sources are in :mod:`tt_sim.network.multicast_order`; what is
asserted here is that both shapes of the mistake are now loud, that the shape
which is *not* a mistake now works, and that nothing about a correctly-ordered
rectangle changed.

The two shapes, and why each used to be silent:

* **Corners reversed for an ascending NoC** (NoC 0, or an untranslated NoC 1).
  ``range(start, end + 1)`` with ``start > end`` is empty, so tt-sim sent the
  packet to nobody, ``num_dests`` was 0, and ``noc_async_write_barrier`` retired
  with no ACKs to wait for. Green run, nothing written.
* **Corners ascending on a NoC 1 addressed in translated coordinates**, which is
  the defect that prompted this: right for NoC 0, wrong for NoC 1. The range is
  non-empty, so tt-sim delivered to every intended tile and produced exactly the
  right numbers, while on silicon the span wraps the torus and the barrier never
  retires.

And the case that was not a mistake but behaved like one: a *correctly* encoded
NoC 1 translated multicast descends, so ``range(start, end + 1)`` was empty and
tt-sim silently delivered nothing. :func:`rectangle_destinations` fixes that
independently of the check, so turning the check off does not put it back.
"""

import pytest

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.multicast_order import (
    DISABLE_ENV_VAR,
    MulticastOrderError,
    checking_enabled,
    rectangle_destinations,
    refresh_from_env,
    set_checking_enabled,
)

_L1_SRC = 0x40000
_L1_DST = 0x60000
_PAYLOAD = b"\xa5" * 16

# Three Wormhole workers in a row, in the unified coordinates the device is
# keyed by; under translation these are also the NoC directory keys.
_WH_ROW = [(18, 18), (19, 18), (20, 18)]


def _wormhole(translated):
    return Wormhole(tensix_coords=list(_WH_ROW), noc_translation=translated)


def _multicast(device, origin, rectangle, noc, payload=_PAYLOAD):
    """Issue one multicast write from ``origin`` over ``noc`` into ``rectangle``.

    ``rectangle`` is ``(x_start, y_start, x_end, y_end)`` exactly as software
    would write it into the command registers -- the point of these tests is
    which order that is, so nothing is normalised on the way in.
    """
    x_start, y_start, x_end, y_end = rectangle
    device.write(origin, _L1_SRC, payload)
    tile = device.tile_directory[origin]
    initiator = tile.get_noc_nui(noc).request_initiators[0]
    # Wormhole packs the rectangle into the MID register: end at bits 4/10,
    # start at bits 16/22 (``NOC_MULTICAST_ADDR``).
    initiator.ret_addr_mid = (
        (x_end << 4) | (y_end << 10) | (x_start << 16) | (y_start << 22)
    )
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(payload)
    initiator.ctrl = 2 | (1 << 4) | (1 << 5)  # write, resp marked, broadcast
    initiator.cmd_ctrl = 1
    initiator.initiate()
    return initiator


def _row_keys(device, noc):
    """The coordinates ``noc``'s directory knows the three workers by.

    Under translation that is the translated coordinate tt-metal puts on the
    wire; without it, the NoC's own physical coordinate -- which on NoC 1 is the
    grid mirror. Spelling it out here keeps the tests about corner *order*
    rather than about which coordinate space a device happens to be keyed in.
    """
    keys = []
    for coord in _WH_ROW:
        nui = device.tile_directory[coord].get_noc_nui(noc)
        keys.append(nui.translated_coord or (nui.x_coord, nui.y_coord))
    return keys


def _rectangle(device, noc, *, descending):
    """The three-worker row as a rectangle, corners in the requested order."""
    keys = _row_keys(device, noc)
    low, high = keys[0], keys[-1]
    if descending:
        low, high = high, low
    return (low[0], low[1], high[0], high[1])


def _received(device, coords):
    """Which of ``coords`` have the payload sitting at the destination address."""
    device.clocks[0].run(64)
    return [
        c for c in coords if bytes(device.read(c, _L1_DST, len(_PAYLOAD))) == _PAYLOAD
    ]


@pytest.fixture(autouse=True)
def _checking_on():
    """Every test states its own expectation; none inherits an env var."""
    set_checking_enabled(True)
    yield
    refresh_from_env()


# ---------------------------------------------------------------------------
# The direction each NoC requires.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("translated", [False, True])
def test_noc0_takes_the_low_corner_first(translated):
    """NoC 0's coordinates increment with its own data flow in both spaces, so
    the low corner goes in the Start field either way."""
    device = _wormhole(translated)
    _multicast(device, (18, 18), _rectangle(device, 0, descending=False), noc=0)
    assert _received(device, _WH_ROW) == _WH_ROW


def test_an_untranslated_noc1_also_takes_the_low_corner_first():
    """Without translation the coordinates on NoC 1 are NoC 1's own, and those
    increment with NoC 1's data flow -- so the corners ascend, as on NoC 0.
    This is the convention ``driver/``'s hand-written kernels use."""
    device = Blackhole(tensix_coords=[(1, 2), (2, 2), (3, 2)])
    tile = device.tile_directory[(1, 2)]
    initiator = tile.get_noc_nui(1).request_initiators[0]
    assert initiator.nui.broadcast_corners_descend is False
    initiator.ret_addr_hi = 3 | (2 << 6) | (1 << 12) | (2 << 18)
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(_PAYLOAD)
    initiator.ctrl = 2 | (1 << 4) | (1 << 5)
    initiator.cmd_ctrl = 1
    initiator.initiate()  # does not raise


def test_a_translated_noc1_takes_the_high_corner_first_and_delivers():
    """The swap ``WormholeB0/NoC/Coordinates.md`` requires, and the reason the
    fix is not only a check: written correctly the corners *descend*, which
    ``range(start, end + 1)`` reads as the empty set. Before this change a
    correctly encoded NoC 1 multicast delivered to nobody and its barrier
    retired having sent nothing."""
    device = _wormhole(translated=True)
    initiator = _multicast(device, (18, 18), (20, 18, 18, 18), noc=1)
    assert initiator.nui.broadcast_corners_descend is True
    assert _received(device, _WH_ROW) == _WH_ROW


@pytest.mark.parametrize("translated", [False, True])
@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_only_a_translated_noc1_descends(arch, translated):
    """The flag the check reads, asserted against every NUI of a built device
    rather than against the one line that sets it -- on both architectures and
    in both coordinate modes, because getting it backwards on one of the four
    would turn the check into an alarm that fires on correct programs."""
    device = (
        Wormhole(tensix_coords=list(_WH_ROW), noc_translation=translated)
        if arch == "wormhole"
        else Blackhole(tensix_coords=[(1, 2), (2, 2)], noc_translation=translated)
    )
    for tile in device.tile_directory.values():
        assert tile.get_noc_nui(0).broadcast_corners_descend is False
        assert tile.get_noc_nui(1).broadcast_corners_descend is translated


# ---------------------------------------------------------------------------
# ...and what happens when it is the other way round.
# ---------------------------------------------------------------------------


def test_a_translated_noc1_multicast_written_low_first_raises():
    """The reported defect: corners emitted low-first, which is right for NoC 0
    and wrong for NoC 1. tt-sim used to deliver to every intended tile and
    produce the right answer while the same program hung on silicon."""
    device = _wormhole(translated=True)
    with pytest.raises(MulticastOrderError) as excinfo:
        _multicast(device, (18, 18), (18, 18, 20, 18), noc=1)
    message = str(excinfo.value)
    assert "NoC1" in message
    assert "Start (18, 18)" in message
    assert "End (20, 18)" in message
    assert "Start >= End" in message
    assert "ascending on X" in message
    assert "WormholeB0/NoC/Coordinates.md" in message
    assert DISABLE_ENV_VAR in message


@pytest.mark.parametrize("translated", [False, True])
def test_a_noc0_multicast_written_high_first_raises(translated):
    """The other shape, on the NoC where ascending is required: reversed corners
    are exactly the rectangle ``range(start, end + 1)`` reads as empty, so this
    is the ``num_dests == 0`` case as well as an ordering violation."""
    device = _wormhole(translated)
    with pytest.raises(MulticastOrderError) as excinfo:
        _multicast(device, (18, 18), _rectangle(device, 0, descending=True), noc=0)
    message = str(excinfo.value)
    assert "NoC0" in message
    assert "Start <= End" in message
    assert "descending on X" in message
    # The empty-set consequence is named, because it is what made this silent.
    assert "empty set" in message
    assert "retired immediately" in message


def test_the_offending_axis_is_named():
    """A rectangle can be right on one axis and wrong on the other, and the
    message says which -- a 4x4 grid multicast that only got Y wrong is not
    findable from 'the corners are in the wrong order'."""
    device = _wormhole(translated=True)
    with pytest.raises(MulticastOrderError) as excinfo:
        # X descends (correct for a translated NoC 1), Y ascends (not).
        _multicast(device, (20, 18), (20, 16, 18, 18), noc=1)
    message = str(excinfo.value)
    assert "ascending on Y" in message
    assert "on X/Y" not in message


@pytest.mark.parametrize("noc,translated", [(0, False), (0, True), (1, True)])
def test_a_degenerate_rectangle_never_fires(noc, translated):
    """A single tile, and a single row or column, satisfy either direction --
    the commonest multicast shape there is, and the one a direction check must
    not fire on."""
    device = _wormhole(translated)
    nui = device.tile_directory[(18, 18)].get_noc_nui(noc)
    x, y = nui.translated_coord or (nui.x_coord, nui.y_coord)
    _multicast(device, (18, 18), (x, y, x, y), noc=noc)
    assert _received(device, [(18, 18)]) == [(18, 18)]


# ---------------------------------------------------------------------------
# The rectangle -> destinations mapping, on its own.
# ---------------------------------------------------------------------------


def test_both_corner_orders_name_the_same_tiles():
    """The set of destinations is a property of the rectangle, not of the order
    its corners were written in -- which is why the enumeration takes min/max
    and the *check* is what carries the direction rule."""
    ascending = rectangle_destinations(18, 16, 20, 18)
    descending = rectangle_destinations(20, 18, 18, 16)
    assert sorted(ascending) == sorted(descending)
    assert len(ascending) == 9


def test_a_rectangle_always_covers_at_least_its_own_corners():
    """There is no encoding of a rectangle that reaches zero tiles: the closed
    interval between two coordinates always contains both of them. The
    ``num_dests == 0`` hole is closed by construction, not by a check that could
    be switched off."""
    for rectangle in ((5, 5, 5, 5), (9, 2, 1, 8), (1, 8, 9, 2)):
        assert len(rectangle_destinations(*rectangle)) >= 1


# ---------------------------------------------------------------------------
# The off switch.
# ---------------------------------------------------------------------------


def test_the_env_var_turns_the_check_off(monkeypatch):
    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv(DISABLE_ENV_VAR, truthy)
        assert refresh_from_env() is False, truthy
        assert checking_enabled() is False
    for falsy in ("0", "false", "", "maybe"):
        monkeypatch.setenv(DISABLE_ENV_VAR, falsy)
        assert refresh_from_env() is True, falsy
    monkeypatch.delenv(DISABLE_ENV_VAR, raising=False)
    assert refresh_from_env() is True


def test_disabling_the_check_does_not_undo_the_enumeration_fix():
    """The check is a diagnostic; ``rectangle_destinations`` is a model fix. A
    user who silences the first still gets the second, so switching the alarm
    off cannot resurrect the silent-drop bug it was written alongside."""
    set_checking_enabled(False)
    try:
        device = _wormhole(translated=True)
        _multicast(device, (18, 18), (20, 18, 18, 18), noc=1)
        assert _received(device, _WH_ROW) == _WH_ROW
    finally:
        set_checking_enabled(True)
