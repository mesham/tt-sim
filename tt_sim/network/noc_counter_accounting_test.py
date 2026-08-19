"""NIU counter accounting: nothing may be decremented that was never counted.

The NIU counters are the only view a kernel has of its own outstanding NoC
traffic, and firmware spins on them: ``noc_async_write_barrier_with_trid``
waits for ``NIU_MST_REQS_OUTSTANDING_ID(trid)`` to read 0, and
``ncrisc_noc_nonposted_write_with_transaction_id_sent`` waits for
``NIU_MST_WRITE_REQS_OUTGOING_ID(trid)`` to read 0 (both in tt-metal's
``noc_nonblocking_api.h``). So a counter that drifts is not bookkeeping — it is
a barrier that retires at the wrong moment.

What prompted this file is that the progress watchdog printed

    NoC tile=(18, 18) nui=1: 0 read(s), -4 write(s) outstanding

and a hardware counter cannot be negative. It got there because the ACK arm of
:meth:`NUI.clock_tick` decremented ``NIU_MST_WRITE_REQS_OUTGOING_ID(trid)`` on
every write acknowledgement received. Per `WormholeB0/NoC/Counters.md`, that
counter has nothing to do with acknowledgements: it goes **up** as software
writes ``NOC_CMD_CTRL`` and back **down** "after the data reads from L1 or
register space are complete" — both at the initiating NIU, both before the
packet is anywhere near its destination. The ACK-side decrement had no matching
increment anywhere, so every ACK pushed the counter one further below zero, and
inline and multicast writes (which never incremented it at all) drifted fastest.

Two consequences, and the difference between them matters:

* **A barrier cannot be released early by this.** The drift only ever pushes a
  counter *below* zero, and the counter firmware polls for a write barrier —
  ``REQS_OUTSTANDING`` — was never touched by the ACK-side decrement in the
  first place. ``test_the_write_barrier_counter_reaches_zero_only_after_...``
  pins that as a property rather than as an accident: the counter a barrier
  spins on may not read 0 while any destination is still missing the payload.
* **A kernel that reads the drifted counter dies.** The NoC register read path
  converts through ``conv_to_bytes(..., signed=False)``, so
  ``NOC_STATUS_READ_REG(noc, NIU_MST_WRITE_REQS_OUTGOING_ID(trid))`` — which is
  exactly what ``noc_async_writes_flushed`` on a trid compiles to — raised
  ``OverflowError`` out of the middle of the simulator once any write had been
  acknowledged. That is the reachable failure, and it is why clamping to
  ``max(0, ...)`` would have been the wrong fix: it would have handed that poll
  a 0 it had not earned.

Hardware does not saturate these counters either — the ISA docs call them "8
bits each (gets both incremented and decremented, so will only overflow or
underflow if software has too many outstanding requests)" — so a floor guard
would not even be faithful. The accounting is what had to change.

No tt-metal, no socket: every case here drives the command registers directly.
"""

import pytest

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.wormhole import Wormhole
from tt_sim.network.tt_noc import NUI
from tt_sim.util.conversion import conv_to_uint32

_OUTSTANDING_ID_0 = int(NUI.NUICounters.CounterNames.NIU_MST_REQS_OUTSTANDING_ID_0)
_OUTGOING_ID_0 = int(NUI.NUICounters.CounterNames.NIU_MST_WRITE_REQS_OUTGOING_ID_0)

#: ``NOC_STATUS(cnt)`` — the counter array as software addresses it.
_NOC_STATUS_BASE = 0x200

#: Three Wormhole workers in a row, in the unified coordinates the device is
#: keyed by and (under translation) the NoC directories are keyed by too.
_ROW = [(18, 18), (19, 18), (20, 18)]
_ORIGIN, _TARGET = _ROW[0], _ROW[1]

#: The Blackhole equivalent, in the physical coordinates that arch is keyed by.
_BH_ROW = [(1, 2), (2, 2), (3, 2)]

_L1_SRC = 0x40000
_L1_DST = 0x60000
_PAYLOAD = b"\xa5" * 32

#: Long enough for a charged NoC flight to a three-tile rectangle to complete;
#: the polls below stop as soon as their counter clears, so this is only a
#: failure bound. (~250 cycles is the measured worst case with the cost model
#: on, 2 with it off — see ``multicast_order_test._SETTLE_CYCLES``.)
_BUDGET = 4096


def _device():
    return Wormhole(tensix_coords=list(_ROW), noc_translation=True)


def _nui(device, coord=_ORIGIN, noc=0):
    return device.tile_directory[coord].get_noc_nui(noc)


def _noc_status(nui, index):
    """Read counter ``index`` the way a kernel does — ``NOC_STATUS_READ_REG``.

    Going through :meth:`NUI.read` rather than indexing ``nui_counters``
    directly is the whole point in one of the tests below: the register path is
    where a negative counter becomes an ``OverflowError``.
    """
    return conv_to_uint32(nui.read(_NOC_STATUS_BASE + 4 * index, 4))


def _key(device, coord, noc=0):
    """The directory coordinate ``noc`` addresses ``coord`` by."""
    nui = device.tile_directory[coord].get_noc_nui(noc)
    return nui.translated_coord or (nui.x_coord, nui.y_coord)


def _issue(
    device,
    *,
    trid=0,
    marked=True,
    inline=False,
    broadcast=False,
    payload=_PAYLOAD,
    noc=0,
):
    """Issue one write from ``_ORIGIN``; return the initiating NUI.

    ``broadcast`` covers the whole ``_ROW`` rectangle, so it is three
    destinations and three acknowledgements for one command.
    """
    device.write(_ORIGIN, _L1_SRC, payload)
    nui = _nui(device, noc=noc)
    initiator = nui.request_initiators[0]
    initiator.packet_tag = trid << 10
    if broadcast:
        (x_start, y_start), (x_end, y_end) = (
            _key(device, _ROW[0], noc),
            _key(device, _ROW[-1], noc),
        )
        initiator.ret_addr_mid = (
            (x_end << 4) | (y_end << 10) | (x_start << 16) | (y_start << 22)
        )
    else:
        x, y = _key(device, _TARGET, noc)
        initiator.ret_addr_mid = (x << 4) | (y << 10)
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(payload)
    initiator.ctrl = 2 | (int(bool(marked)) << 4) | (int(bool(inline)) << 3)
    initiator.ctrl |= int(bool(broadcast)) << 5
    initiator.cmd_ctrl = 1
    initiator.initiate()
    return nui


def _read(device, trid=0, *, noc=0):
    """Issue one read of the payload back out of ``_TARGET``'s L1."""
    nui = _nui(device, noc=noc)
    initiator = nui.request_initiators[0]
    initiator.packet_tag = trid << 10
    x, y = _key(device, _TARGET, noc)
    initiator.target_addr_mid = (x << 4) | (y << 10)
    initiator.target_addr_low = _L1_DST
    initiator.ret_addr_low = _L1_SRC
    initiator.at_len_be = len(_PAYLOAD)
    initiator.ctrl = 0
    initiator.cmd_ctrl = 1
    initiator.initiate()
    return nui


def _settle(device, nui, budget=_BUDGET):
    """Pump until every request ``nui`` issued has been answered."""
    for _ in range(budget):
        if all(not pending for pending in nui.outstanding_noc_requests.values()):
            return
        device.run(1)
    raise AssertionError(f"NUI {nui.id_pair} still awaiting after {budget} cycles")


def _negative(nui):
    """``{counter name: value}`` for every counter that has gone below zero."""
    return {
        name.name: nui.nui_counters[name]
        for name in NUI.NUICounters.CounterNames
        if nui.nui_counters[name] < 0
    }


# ---------------------------------------------------------------------------
# The defect: a decrement with no matching increment.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, kwargs",
    [
        ("unicast marked", {}),
        ("unicast posted", {"marked": False}),
        ("unicast on trid 5", {"trid": 5}),
        ("inline", {"inline": True, "payload": b"\xa5\xa5\xa5\xa5"}),
        ("multicast", {"broadcast": True}),
        ("multicast posted", {"broadcast": True, "marked": False}),
    ],
)
def test_a_completed_write_leaves_no_counter_below_zero(kind, kwargs):
    """Every shape of write, and the invariant stated over the whole counter
    array rather than the one slot each case happens to touch. Hardware cannot
    represent a negative here at all, so any of these is a modelling error —
    and which slot drifts depends on the transaction ID, which is the caller's
    choice, not ours."""
    device = _device()
    nui = _issue(device, **kwargs)
    _settle(device, nui)
    assert _negative(nui) == {}, f"{kind} write drove counters negative"


def test_a_read_and_a_write_sharing_a_trid_leave_the_slots_at_rest():
    """The realistic sequence — write, barrier, read back — because the two
    directions decrement different counters and a trid is reused constantly."""
    device = _device()
    nui = _issue(device)
    _settle(device, nui)
    _read(device)
    _settle(device, nui)
    assert _negative(nui) == {}
    for i in range(16):
        assert nui.nui_counters[_OUTSTANDING_ID_0 + i] == 0
        assert nui.nui_counters[_OUTGOING_ID_0 + i] == 0


def test_repeated_writes_do_not_accumulate_drift():
    """One ACK per write was one decrement per write, so the counter walked
    steadily away from zero over a program rather than settling at some fixed
    wrong value. Eight writes, because a drift that only shows up on the
    hundredth is still the same bug."""
    device = _device()
    for _ in range(8):
        nui = _issue(device, broadcast=True)
        _settle(device, nui)
    assert _negative(nui) == {}


# ---------------------------------------------------------------------------
# What it costs: the counter is register-visible, and firmware reads it.
# ---------------------------------------------------------------------------


def test_a_kernel_can_read_the_outgoing_counter_after_a_write():
    """``noc_async_writes_flushed`` on a transaction ID compiles to
    ``NOC_STATUS_READ_REG(noc, NIU_MST_WRITE_REQS_OUTGOING_ID(trid))``. With
    the counter negative, that read raised ``OverflowError`` from
    ``conv_to_bytes`` — the simulator died inside a register read that on
    silicon cannot fail. This is the reachable consequence of the drift."""
    device = _device()
    nui = _issue(device, broadcast=True)
    _settle(device, nui)
    for trid in range(16):
        assert _noc_status(nui, _OUTGOING_ID_0 + trid) == 0


@pytest.mark.parametrize("broadcast", [False, True])
def test_the_transaction_ids_own_slot_is_the_one_that_moves(broadcast):
    """There are sixteen of these counters because software chooses which one a
    request lands in. The increment used to be hardcoded to slot 0 while the
    (spurious) decrement used the real transaction ID, so the pair did not even
    describe the same counter.

    The docs put the increment "as software writes to ``NOC_CMD_CTRL``" and the
    decrement "after the data reads from L1 ... are complete", so the moment the
    payload is pulled out of L1 is exactly the window in which the counter is
    up — which is where this test looks. A broadcast reads its payload out of
    L1 once however wide the rectangle is, so it bumps the slot by one too: the
    fan-out is the routers' work, and it is the *acknowledgements* that are per
    destination."""
    device = _device()
    device.write(_ORIGIN, _L1_SRC, _PAYLOAD)
    nui = _nui(device)
    memory = nui.attached_memory
    original = memory.read
    seen = []

    def spy(addr, *args, **kwargs):
        if addr == _L1_SRC:
            seen.append([nui.nui_counters[_OUTGOING_ID_0 + i] for i in range(16)])
        return original(addr, *args, **kwargs)

    memory.read = spy
    try:
        _issue(device, trid=5, broadcast=broadcast)
    finally:
        memory.read = original
    _settle(device, nui)

    assert seen, "the write never read its payload out of L1"
    assert seen[0][5] == 1, f"trid 5's slot did not go up: {seen[0]}"
    assert sum(seen[0]) == 1, f"some other trid's slot moved too: {seen[0]}"
    assert all(v == 0 for v in seen[0][:5] + seen[0][6:])


# ---------------------------------------------------------------------------
# The question the drift raises: can a barrier retire early?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("broadcast", [False, True])
def test_the_write_barrier_counter_reaches_zero_only_after_the_data_has_landed(
    broadcast,
):
    """``noc_async_write_barrier_with_trid`` spins on
    ``NIU_MST_REQS_OUTSTANDING_ID(trid) == 0``. Poll it exactly as that loop
    does, one cycle at a time, and the first zero it sees must come no earlier
    than the payload arriving at *every* destination.

    This is the answer to "can the drift release a barrier early": no — the
    drift pushes counters below zero, which cannot make a ``== 0`` poll pass,
    and it never reached this counter in the first place. Asserted rather than
    argued, and for the multicast case too, where one command produces N
    acknowledgements and so had the most room to be wrong."""
    device = _device()
    nui = _issue(device, broadcast=broadcast)
    destinations = _ROW if broadcast else [_TARGET]

    assert _noc_status(nui, _OUTSTANDING_ID_0) > 0, (
        "the barrier counter was already zero at issue, so the barrier below "
        "proves nothing"
    )
    for _ in range(_BUDGET):
        if _noc_status(nui, _OUTSTANDING_ID_0) == 0:
            break
        device.run(1)
    else:
        raise AssertionError("the write barrier never retired")

    for coord in destinations:
        assert bytes(device.read(coord, _L1_DST, len(_PAYLOAD))) == _PAYLOAD, (
            f"barrier retired with {coord} still missing the payload"
        )


def test_blackhole_accounts_the_same_way():
    """The counter block is shared code and both architectures' ``Counters.md``
    state the same rules (only the packet-split threshold differs: 8192 bytes
    on Wormhole, 16384 on Blackhole, which is what ``noc_max_burst_size``
    already carries). Asserted rather than assumed, because the *addressing* of
    a multicast rectangle is arch-specific and it is the multicast path that
    drifted fastest."""
    device = Blackhole(tensix_coords=list(_BH_ROW))
    device.write(_BH_ROW[0], _L1_SRC, _PAYLOAD)
    nui = device.tile_directory[_BH_ROW[0]].get_noc_nui(0)
    initiator = nui.request_initiators[0]
    (x_start, y_start), (x_end, y_end) = _BH_ROW[0], _BH_ROW[-1]
    initiator.ret_addr_hi = x_end | (y_end << 6) | (x_start << 12) | (y_start << 18)
    initiator.target_addr_low = _L1_SRC
    initiator.ret_addr_low = _L1_DST
    initiator.at_len_be = len(_PAYLOAD)
    initiator.ctrl = 2 | (1 << 4) | (1 << 5)
    initiator.cmd_ctrl = 1
    initiator.initiate()
    _settle(device, nui)
    assert _negative(nui) == {}


def test_an_unmarked_write_never_bumps_the_barrier_counter():
    """The mirror of the above: a posted write is not acknowledged to the
    counter at all, so a barrier counter it never incremented must not be
    decremented on its behalf either. That asymmetry is the shape the
    ACK-side decrement had."""
    device = _device()
    nui = _issue(device, marked=False)
    assert _noc_status(nui, _OUTSTANDING_ID_0) == 0
    _settle(device, nui)
    assert _noc_status(nui, _OUTSTANDING_ID_0) == 0
