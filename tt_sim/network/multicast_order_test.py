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

from tt_sim.behaviour import require
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
from tt_sim.network.noc_coords import BlackholeNocCoords

_L1_SRC = 0x40000
_L1_DST = 0x60000
_PAYLOAD = b"\xa5" * 16

# Three Wormhole workers in a row, in the unified coordinates the device is
# keyed by; under translation these are also the NoC directory keys.
_WH_ROW = [(18, 18), (19, 18), (20, 18)]


def _wormhole(translated):
    return Wormhole(tensix_coords=list(_WH_ROW), noc_translation=translated)


def _encode_rectangle(initiator, rectangle):
    """Write ``rectangle`` into this initiator's command registers.

    Wormhole packs both corners into the MID address register (end at bits
    4/10, start at 16/22 -- ``NOC_MULTICAST_ADDR``); Blackhole uses the
    dedicated HI register (end at 0/6, start at 12/18). The round-trip through
    the architecture's own decoder is asserted rather than assumed, so a test
    cannot quietly encode a rectangle other than the one it names.
    """
    x_start, y_start, x_end, y_end = rectangle
    if isinstance(initiator.nui.noc_coord_strategy, BlackholeNocCoords):
        initiator.ret_addr_hi = x_end | (y_end << 6) | (x_start << 12) | (y_start << 18)
    else:
        initiator.ret_addr_mid = (
            (x_end << 4) | (y_end << 10) | (x_start << 16) | (y_start << 22)
        )
    assert initiator.nui.noc_coord_strategy.broadcast_coords(initiator) == rectangle


def _multicast(device, origin, rectangle, noc, payload=_PAYLOAD):
    """Issue one multicast write from ``origin`` over ``noc`` into ``rectangle``.

    ``rectangle`` is ``(x_start, y_start, x_end, y_end)`` exactly as software
    would write it into the command registers -- the point of these tests is
    which order that is, so nothing is normalised on the way in.
    """
    device.write(origin, _L1_SRC, payload)
    tile = device.tile_directory[origin]
    initiator = tile.get_noc_nui(noc).request_initiators[0]
    _encode_rectangle(initiator, rectangle)
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


#: Cycles to let a multicast land before reading destinations. 64 is enough
#: with the cost model OFF, and NOT enough with it ON -- a charged NoC flight
#: delivered 1 of 3 destinations in 64 cycles and all 3 by 128. Several tests
#: assert an EMPTY result (the tiles outside the rectangle, and the tiles a
#: rejected multicast must not have reached), so this must also be long enough
#: that "nothing arrived" means it never will.
#:
#: Measured worst case, re-checked for the two-dimensional cases below because
#: a 2x2 rectangle is more destinations than the row was: every destination of
#: every case in this file has the payload within 254 cycles with the cost
#: model on (Blackhole NoC 0; Wormhole NoC 0 needs 191, either arch's NoC 1
#: needs 47) and within 2 cycles with it off. 1024 is 4x the worst case.
_SETTLE_CYCLES = 1024


def _received(device, coords):
    """Which of ``coords`` have the payload sitting at the destination address."""
    device.clocks[0].run(_SETTLE_CYCLES)
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
    This is the convention ``driver/``'s hand-written kernels use.

    Delivery is asserted, not just the absence of a raise: a guard that keeps
    quiet while the enumeration reaches nobody is the bug this module was
    written for, wearing the other hat."""
    row = [(1, 2), (2, 2), (3, 2)]
    device = Blackhole(tensix_coords=list(row))
    initiator = device.tile_directory[(1, 2)].get_noc_nui(1).request_initiators[0]
    assert initiator.nui.broadcast_corners_descend is False
    _multicast(device, (1, 2), (1, 2, 3, 2), noc=1)
    assert _received(device, row) == row


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
# Two dimensions at once, on every NoC and in both coordinate spaces.
# ---------------------------------------------------------------------------
#
# Everything above this line is one-dimensional: a row of three workers, whose
# Y axis is degenerate and therefore satisfies either direction. That leaves
# two things correct only by argument, and this repo does not count arguments
# as evidence:
#
# * **The two-dimensional case.** ``check_corner_order`` tests X and Y
#   independently, so a rectangle right on one axis and wrong on the other
#   *should* fire -- but nothing measured it, here or in the corpus a consumer
#   re-ran against this guard, whose every rectangle sat in a single column.
#   That combination is the one independent per-axis checking could plausibly
#   get wrong, so it is the one worth pinning.
# * **NoC 0.** Every multicast a real consumer emits sits in a reader on NoC 1,
#   so the ``descending=False`` branch is dead in their corpus and was tested
#   here on Wormhole only. Blackhole's NoC 0 had no test at all.
#
# The cases below cross both: 2x2 rectangles on both architectures, both NoCs
# and both coordinate conventions, with the corners right, wrong on X only,
# wrong on Y only, and wrong on both.

#: A 3x3 block of workers per architecture, the 2x2 sub-block a multicast
#: targets, and an origin *outside* that sub-block. The five workers left over
#: are the evidence for "and nobody else": a guard that stays quiet while the
#: enumeration delivers to the wrong set is the original bug in a new costume,
#: and a 2x2 taken out of a 2x2 could not tell the difference.
_WH_BLOCK = [(x, y) for x in (18, 19, 20) for y in (18, 19, 20)]
_BH_BLOCK = [(x, y) for x in (1, 2, 3) for y in (2, 3, 4)]

#: ``arch -> (block, target rectangle, multicast origin)``.
_BLOCKS = {
    "wormhole": (_WH_BLOCK, [(18, 18), (18, 19), (19, 18), (19, 19)], (20, 20)),
    "blackhole": (_BH_BLOCK, [(1, 2), (1, 3), (2, 2), (2, 3)], (3, 4)),
}

#: Every combination the guard has a branch for: two architectures, both
#: coordinate conventions, both NoCs. ``descending`` is required on exactly one
#: of the eight, and each test asserts which one rather than trusting the flag
#: it reads.
_CONFIGURATIONS = [
    (arch, translated, noc)
    for arch in ("wormhole", "blackhole")
    for translated in (False, True)
    for noc in (0, 1)
]


def _block_device(arch, translated):
    """A device with this architecture's 3x3 worker block built."""
    block = _BLOCKS[arch][0]
    if arch == "wormhole":
        return Wormhole(tensix_coords=list(block), noc_translation=translated)
    return Blackhole(tensix_coords=list(block), noc_translation=translated)


def _noc_key(device, noc, coord):
    """The coordinate ``noc``'s directory knows the tile at ``coord`` by.

    Under translation, the translated coord tt-metal puts on the wire. Without
    it, the NoC's own physical coord -- the grid mirror, on NoC 1 -- where that
    is registered (Wormhole), and the canonical NoC 0 coord where it is not
    (Blackhole, whose bank-to-noc table never emits a NoC 1 mirror, so
    ``_register_tile_internals`` registers none). Asked of the directory rather
    than recomputed here, so these tests stay about corner *order* and not about
    which coordinate space a device happens to be keyed in.
    """
    nui = device.tile_directory[coord].get_noc_nui(noc)
    directory = device.noc_0_directory if noc == 0 else device.noc_1_directory
    candidates = (
        [nui.translated_coord]
        if nui.translated_coord is not None
        else [(nui.x_coord, nui.y_coord), nui.get_id_pair()]
    )
    for candidate in candidates:
        if directory.get(candidate) is nui:
            return candidate
    raise AssertionError(f"{coord} has no NoC{noc} directory key: {candidates}")


def _corners(device, noc, targets):
    """The low and high corners of ``targets`` in ``noc``'s coordinate space.

    Taken from the tiles rather than written down, because NoC 1's mirror flips
    both axes: the tile at the *low* corner in one space is at the *high* corner
    in the other, and a hand-written rectangle would silently name a different
    shape on half of these configurations.
    """
    keys = [_noc_key(device, noc, coord) for coord in targets]
    low = (min(k[0] for k in keys), min(k[1] for k in keys))
    high = (max(k[0] for k in keys), max(k[1] for k in keys))
    # ...and the shape those corners name has to be exactly the target tiles,
    # with no gap cell in the middle -- otherwise "delivered to its rectangle"
    # would be a claim about a rectangle with holes in it.
    assert sorted(rectangle_destinations(*low, *high)) == sorted(keys)
    return low, high


def _corner_order(low, high, *, descending, wrong_x=False, wrong_y=False):
    """``(x_start, y_start, x_end, y_end)`` written the way this NoC requires.

    ``wrong_x`` / ``wrong_y`` reverse a single axis, leaving the other one
    correct -- the case the whole of this section exists for, since a check that
    collapsed the two axes into one comparison would pass it.
    """
    start, end = (high, low) if descending else (low, high)
    x_start, x_end = start[0], end[0]
    y_start, y_end = start[1], end[1]
    if wrong_x:
        x_start, x_end = x_end, x_start
    if wrong_y:
        y_start, y_end = y_end, y_start
    return (x_start, y_start, x_end, y_end)


def _case(arch, translated, noc, targets=None):
    """A built device, the corners of ``targets`` in ``noc``'s space, and the
    direction that NoC requires them in."""
    device = _block_device(arch, translated)
    targets = _BLOCKS[arch][1] if targets is None else targets
    low, high = _corners(device, noc, targets)
    nui = device.tile_directory[_BLOCKS[arch][2]].get_noc_nui(noc)
    # Which way round is a property of the NoC and the coordinate space, and is
    # asserted here rather than merely read, so a flag that got set backwards
    # could not quietly redefine what these tests mean by "correct".
    assert nui.broadcast_corners_descend is (translated and noc == 1)
    return device, low, high, nui.broadcast_corners_descend


@pytest.mark.parametrize("arch,translated,noc", _CONFIGURATIONS)
def test_a_two_dimensional_multicast_delivers_to_exactly_its_rectangle(
    arch, translated, noc
):
    """A rectangle spanning X *and* Y, written correctly, reaches all four of
    its tiles and none of the five around them.

    The delivery half of the two-dimensional case: the guard staying silent is
    only half an answer, since the enumeration it guards is what actually picks
    the destinations. Asserting the whole 3x3 block -- not just the target --
    is what makes "and nobody else" a measurement.
    """
    device, low, high, descending = _case(arch, translated, noc)
    block, targets, origin = _BLOCKS[arch]
    # Genuinely two-dimensional: neither axis is degenerate, so reversing
    # either one is a real change rather than a no-op.
    assert low[0] != high[0]
    assert low[1] != high[1]
    rectangle = _corner_order(low, high, descending=descending)
    assert len(rectangle_destinations(*rectangle)) == len(targets)
    _multicast(device, origin, rectangle, noc)
    assert _received(device, block) == sorted(targets)


@pytest.mark.parametrize("axis", ["X", "Y"])
@pytest.mark.parametrize("arch,translated,noc", _CONFIGURATIONS)
def test_a_two_dimensional_multicast_wrong_on_one_axis_raises(
    arch, translated, noc, axis
):
    """Right on one axis, wrong on the other -- and it still fires, naming the
    axis at fault and not the other one.

    This is the case no corpus reaches and the only one where testing the axes
    independently could have been the wrong design: a check that compared the
    corners as points, or that stopped at the first axis it liked, would let
    exactly this rectangle through. On hardware the good axis is delivered as
    asked and the bad one wraps the torus, so the packet lands on a superset of
    the tiles the kernel counted -- a hang, not a wrong number.
    """
    device, low, high, descending = _case(arch, translated, noc)
    block, _, origin = _BLOCKS[arch]
    rectangle = _corner_order(
        low, high, descending=descending, wrong_x=axis == "X", wrong_y=axis == "Y"
    )
    with pytest.raises(MulticastOrderError) as excinfo:
        _multicast(device, origin, rectangle, noc)
    message = str(excinfo.value)
    wrote = "ascending" if descending else "descending"
    assert f"{wrote} on {axis}" in message
    assert f"on {'Y' if axis == 'X' else 'X'}" not in message
    assert "on X/Y" not in message
    # The guard fires *instead of* the transfer, not alongside it: a rectangle
    # tt-sim cannot model must not be half-delivered on the way to the raise.
    assert _received(device, block) == []


@pytest.mark.parametrize("arch,translated,noc", _CONFIGURATIONS)
def test_a_two_dimensional_multicast_wrong_on_both_axes_names_both(
    arch, translated, noc
):
    """Both axes reversed names both axes -- the one shape where ``X/Y`` is the
    right answer, so the single-axis tests above are asserting a real
    distinction rather than a message that always says the same thing."""
    device, low, high, descending = _case(arch, translated, noc)
    block, _, origin = _BLOCKS[arch]
    rectangle = _corner_order(
        low, high, descending=descending, wrong_x=True, wrong_y=True
    )
    with pytest.raises(MulticastOrderError) as excinfo:
        _multicast(device, origin, rectangle, noc)
    wrote = "ascending" if descending else "descending"
    assert f"{wrote} on X/Y" in str(excinfo.value)
    assert _received(device, block) == []


@pytest.mark.parametrize("orientation", ["row", "column"])
@pytest.mark.parametrize("arch,translated,noc", _CONFIGURATIONS)
def test_a_degenerate_axis_inside_a_grid_still_never_fires(
    arch, translated, noc, orientation
):
    """A single row or column, in a device that *has* both dimensions.

    The commonest multicast shape there is -- and one where the degenerate axis
    is equal at both corners, so it satisfies ascending and descending at once.
    Pinned separately from the 1x1 case because the obvious way to "fix" a
    per-axis check into a stricter one is to demand a strict inequality, which
    would reject every single-row multicast tt-metal emits.
    """
    _, targets, origin = _BLOCKS[arch]
    fixed = (
        min(t[1] for t in targets)
        if orientation == "row"
        else min(t[0] for t in targets)
    )
    line = [t for t in targets if (t[1] if orientation == "row" else t[0]) == fixed]
    assert len(line) == 2
    device, low, high, descending = _case(arch, translated, noc, targets=line)
    # Degenerate on exactly one axis, which is the whole point.
    assert (low[0] == high[0]) is (orientation == "column")
    assert (low[1] == high[1]) is (orientation == "row")
    _multicast(device, origin, _corner_order(low, high, descending=descending), noc)
    assert _received(device, line) == sorted(line)


@pytest.mark.parametrize("translated", [False, True])
@pytest.mark.parametrize("arch", ["wormhole", "blackhole"])
def test_noc0_delivers_low_first_and_raises_high_first_on_both_arches(arch, translated):
    """NoC 0, on purpose, on both architectures and in both coordinate spaces.

    The branch a real consumer's corner-selection code never takes, because
    every multicast it emits is issued from a reader on NoC 1 -- and which had
    no Blackhole test here at all. NoC 0's coordinates increment with its own
    data flow in *both* spaces, so translation changes nothing about it: the low
    corner goes in the Start field either way, and the reverse is the shape
    ``range(start, end + 1)`` used to read as the empty set.
    """
    _, targets, origin = _BLOCKS[arch]
    row = [t for t in targets if t[1] == min(t[1] for t in targets)]
    device, low, high, descending = _case(arch, translated, 0, targets=row)
    assert descending is False
    _multicast(device, origin, _corner_order(low, high, descending=False), noc=0)
    assert _received(device, row) == sorted(row)

    device, low, high, _ = _case(arch, translated, 0, targets=row)
    with pytest.raises(MulticastOrderError) as excinfo:
        _multicast(device, origin, _corner_order(low, high, descending=True), noc=0)
    assert "NoC0" in str(excinfo.value)
    assert "Start <= End" in str(excinfo.value)
    assert _received(device, row) == []


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


def test_the_behaviour_marker_for_this_guard_is_published():
    """The guard and the name external suites assert on live and die together.

    ``tt_sim.behaviour`` publishes ``noc1-multicast-corner-order`` so a consumer's suite can refuse
    to run against a tt-sim that lacks this check rather than collect another
    set of green results that exercised nothing. Deleting the registry entry
    therefore has to turn *this* suite red — a marker quietly withdrawn is
    exactly the failure the marker exists to prevent.
    """
    require("noc1-multicast-corner-order")
