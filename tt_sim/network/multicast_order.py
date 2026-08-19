"""Multicast rectangle corner-ordering checking.

A NoC broadcast names its destinations as a rectangle: a ``Start`` corner and an
``End`` corner, packed into the NIU's command registers. Which corner goes in
which field is **not** free — the corners have to be written in the order the
packet travels, and that order is a property of the NoC the request is issued
on, so the same rectangle is encoded differently on NoC 0 and NoC 1.

Getting it wrong is silent on both hardware and (until this module) tt-sim, in
opposite and equally useless ways. It is the defect this module exists for: a
compiler back end emitted every multicast rectangle low-corner-first, which is
right for NoC 0 and wrong for NoC 1, and every one of its multicast programs
deadlocked on silicon while running green on two simulators.

The rule
--------
Each NoC has its **own** coordinate system, whose axes increment in that NoC's
direction of data flow: NoC #0 has ``(0, 0)`` at the top left and flows
rightwards and downwards, NoC #1 has ``(0, 0)`` at the bottom right and flows
leftwards and upwards (``WormholeB0/NoC/Coordinates.md``,
``WormholeB0/NoC/RoutingPaths.md``). In a NoC's own coordinates the rectangle is
therefore always written ``Start ≤ End``.

Coordinate *translation* — which is what tt-metal uses, and what tt-sim's
``noc_translation`` mode models — overlays a single coordinate range on both
NoCs. That range increments with NoC 0's data flow, and, in the ISA docs' own
words:

    Uniform coordinate system for both NoC #0 and NoC #1 (*)

    (*) Except when performing broadcasts, where StartX ↔ EndX need to be
    swapped by software, and likewise StartY ↔ EndY need to be swapped. This is
    because coordinate translation does not change the direction of data flow.

    -- ``WormholeB0/NoC/Coordinates.md``, "Coordinate Translation"
       (``BlackholeA0/NoC/Coordinates.md`` says the same)

So, in the coordinate space the registers actually hold:

===================  ==============  ==============
Coordinate space     NoC 0           NoC 1
===================  ==============  ==============
translated           ``Start ≤ End``  ``Start ≥ End``
raw NoC coordinates  ``Start ≤ End``  ``Start ≤ End``
===================  ==============  ==============

tt-metal implements exactly the top row. ``Device::get_noc_multicast_encoding``
(``tt_metal/impl/device/device.cpp``) reads:

    // NOC 1 mcasts from bottom left to top right, so we need to reverse the coords
    if (noc_index == 0) {
        return hal.noc_multicast_encoding(start.x, start.y, end.x, end.y);
    }
    return hal.noc_multicast_encoding(end.x, end.y, start.x, start.y);

and its own watcher enforces the same table, keyed on whether the coordinates
are virtual (translated) or physical
(``tt_metal/hw/inc/internal/debug/sanitize.h``,
``DebugSanitizeNocMulticastInvalidRange``)::

    if (is_virtual_coord && is_virtual_coord_end) {
        if (noc_id == 0) { if (x > x_end || y > y_end) ... }
        else             { if (x_end > x || y_end > y) ... }
    } else {
        if (x > x_end || y > y_end) ...
    }

Note that the kernel-side ``get_noc_multicast_addr`` does **not** swap anything:
it passes its four arguments straight through to ``NOC_MULTICAST_ADDR``. The
swap is the caller's job, which is why it is so easy to omit.

What hardware does with a wrongly-ordered rectangle
---------------------------------------------------
Not nothing, and not what was asked for. Per ``WormholeB0/NoC/MemoryMap.md``
(and ``BlackholeA0/NoC/MemoryMap.md``), the span is resolved in NoC coordinates
*after* translation, and the reversed case wraps around the torus rather than
being empty:

    When ``StartX ≤ EndX``, the X span of the broadcast is all ``x`` such that
    ``StartX ≤ x ≤ EndX``. When ``StartX > EndX``, the X span of the broadcast
    is all ``x`` such that ``x ≤ EndX || StartX ≤ x``.

So the packet lands on a different — generally much larger — set of tiles than
the kernel counted in its ``num_dests`` argument, the ACK count the NIU sees
never equals the count the kernel is waiting for, and ``noc_async_write_barrier``
spins forever. That is the reported silicon symptom.

tt-sim does not model the torus wrap (it enumerates the closed interval between
the two corners, whichever order they arrive in), so without this check it
completes either way: a reversed rectangle used to enumerate an *empty* range,
sending nothing and letting the barrier retire with no ACKs to wait for, while a
rectangle written the NoC 0 way on NoC 1 enumerates every intended tile and
produces exactly the right answer. Both are green runs of a program that hangs
on hardware, which is the worst failure a simulator has.

Raising rather than warning
---------------------------
This check fires on precisely the rectangles tt-sim cannot model: the ones where
its interval enumeration and the hardware's span disagree. There is no case in
which it fires and tt-sim was about to produce the answer hardware produces, so
it cannot cry wolf in the way a heuristic would — the alternative to raising is
not "carry on correctly", it is "carry on wrongly and silently".

It is deliberately *stricter* than tt-metal's watcher, which skips the ordering
check entirely for Tensix-to-Tensix multicasts on Wormhole and Blackhole because
the hardware's wrap-around is legal there. Legal is not the same as intended,
and tt-sim cannot deliver a wrapped span anyway: a wrap-around Tensix multicast
would be mismodelled silently today. If one ever turns out to be wanted, the
answer is to model the wrap, not to lower this to a warning.

Checking is on by default and is disabled by setting
``TT_SIM_DISABLE_MULTICAST_ORDER_CHECKS`` to a truthy value
(``1``/``true``/``yes``/``on``, case-insensitive), following the same
convention as ``TT_SIM_DISABLE_ALIGNMENT_CHECKS``.
"""

from __future__ import annotations

import os

#: Environment variable that turns multicast corner-order checking off.
DISABLE_ENV_VAR = "TT_SIM_DISABLE_MULTICAST_ORDER_CHECKS"

_TRUTHY = {"1", "true", "yes", "on"}


class MulticastOrderError(RuntimeError):
    """Raised when a multicast rectangle's corners are ordered against the
    direction of data flow of the NoC the request was issued on."""


def _env_disabled() -> bool:
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() in _TRUTHY


#: Cached at import so the common path does not hit ``os.environ`` per request.
#: Call :func:`refresh_from_env` after mutating the environment.
_disabled = _env_disabled()


def refresh_from_env() -> bool:
    """Re-read :data:`DISABLE_ENV_VAR`. Returns whether checking is now enabled."""
    global _disabled
    _disabled = _env_disabled()
    return not _disabled


def checking_enabled() -> bool:
    """Whether corner-order checking is currently active."""
    return not _disabled


def set_checking_enabled(enabled: bool) -> None:
    """Force checking on or off, ignoring the environment (for tests)."""
    global _disabled
    _disabled = not enabled


def rectangle_destinations(x_start, y_start, x_end, y_end):
    """Every ``(x, y)`` the broadcast rectangle covers.

    The closed interval between the two corners on each axis, **whichever order
    they arrive in** — because the order is a function of the NoC (see the
    module docstring) and both orders name the same set of tiles. Taking the
    corners literally as ``range(start, end + 1)`` instead silently drops every
    correctly-encoded NoC 1 multicast under coordinate translation, where the
    corners are required to descend.

    tt-sim does not model the torus wrap that hardware performs when the
    corners are ordered against the direction of data flow;
    :func:`check_corner_order` is what stops such a rectangle from reaching here
    unremarked.
    """
    return [
        (x, y)
        for x in range(min(x_start, x_end), max(x_start, x_end) + 1)
        for y in range(min(y_start, y_end), max(y_start, y_end) + 1)
    ]


def check_corner_order(
    x_start,
    y_start,
    x_end,
    y_end,
    *,
    descending: bool,
    noc_number: int,
    initiator: object = None,
) -> None:
    """Raise :class:`MulticastOrderError` unless the corners are ordered along
    the direction of data flow of this NoC.

    ``descending`` says which way that is in the coordinate space the registers
    hold: ``True`` on a NoC 1 addressed in translated coordinates, ``False``
    otherwise (see the table in the module docstring). A degenerate axis — one
    where both corners are the same coordinate, which is every single-row or
    single-column multicast — satisfies either direction and never fires.
    """
    if _disabled:
        return
    if descending:
        bad_axes = [
            name
            for name, start, end in (("X", x_start, x_end), ("Y", y_start, y_end))
            if start < end
        ]
    else:
        bad_axes = [
            name
            for name, start, end in (("X", x_start, x_end), ("Y", y_start, y_end))
            if start > end
        ]
    if not bad_axes:
        return

    where = f"NoC{noc_number}"
    if initiator is not None:
        where = f"{where} {initiator}"
    required = "Start >= End" if descending else "Start <= End"
    wrote = "ascending" if descending else "descending"
    flow = (
        "NoC 1 data flows leftwards and upwards, so in the translated "
        "coordinate space -- which increments with NoC 0's data flow -- a "
        "broadcast rectangle is written high corner first"
        if descending
        else "this NoC's coordinates increment in its own direction of data "
        "flow, so a broadcast rectangle is written low corner first"
    )
    # Reversed corners are the case tt-sim used to enumerate as the empty set:
    # ``range(start, end + 1)`` with ``start > end`` sends to nobody at all, and
    # the barrier then has no ACKs to wait for. Saying so makes the two shapes
    # of this bug distinguishable in the message.
    consequence = (
        "Hardware resolves this as a torus wrap-around, not as an empty "
        "rectangle: per WormholeB0/NoC/MemoryMap.md the span on such an axis is "
        "every coordinate at or below End together with every coordinate at or "
        "above Start -- so the packet lands on tiles the kernel never counted in num_dests "
        "and noc_async_write_barrier waits forever for an ACK count that cannot "
        "match. tt-sim does not model the wrap"
    )
    if wrote == "descending":
        consequence += (
            ", and before this check it enumerated the rectangle as the empty "
            "set: nothing was sent and the barrier retired immediately"
        )
    consequence += "."

    raise MulticastOrderError(
        f"{where} multicast: rectangle corners ordered against the direction "
        f"of data flow. Start ({x_start}, {y_start}) and End ({x_end}, {y_end}) "
        f"are {wrote} on {'/'.join(bad_axes)}, but this request requires "
        f"{required} on every axis: {flow}. "
        f"{consequence} "
        f"Software must swap the corners for the NoC it is issuing on -- see "
        f"WormholeB0/NoC/Coordinates.md 'Coordinate Translation' and "
        f"Device::get_noc_multicast_encoding in tt-metal, which does exactly "
        f"that swap for noc_index 1. "
        f"Set {DISABLE_ENV_VAR}=1 to disable multicast order checking."
    )
