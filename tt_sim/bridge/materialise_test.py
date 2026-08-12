"""On-demand worker materialisation, and the early-packet race it must close.

The race is the whole reason this file exists. Under
``detail::LaunchProgram`` the host writes ``go=GO`` to each of a program's
cores in turn, pumping simulator cycles between wire messages, so a worker
launched early is executing its kernel long before the last worker is
mentioned. If workers are materialised only when the host launches on them,
that early worker can write into a peer's L1 — or bump its semaphore — while
the peer is still a stand-in, and ``NUI.resolve_destination`` null-routes the
packet: acknowledged, dropped, no crash, **wrong answer**. That is strictly
worse than the hang on-demand materialisation replaces, so it is pinned first
and by value.

``test_a_peers_packet_is_lost_without_the_directory_miss_hook`` is the
counterfactual: the identical sequence with only the host triggers left in
place, asserting the payload is *gone*. It is what makes the NoC trigger
load-bearing rather than belt-and-braces — delete the hook and the first test
fails on the value (zeros where the payload should be) while the second passes.

No sockets and no tt-metal: the fabric, the bridge ``Device`` and a real
Wormhole, driven directly.
"""

import pytest

from tt_sim.bridge.cores import DeferredTensixCore
from tt_sim.bridge.device import Device
from tt_sim.bridge.fabric import Fabric
from tt_sim.bridge.materialise import LazyTensixPool, _CatchUpTensixCore
from tt_sim.device.wormhole import Wormhole

#: Physical NoC 0 worker coord -> tt-sim unified tile coord, as a server's
#: ``TENSIX_COORD_MAP`` is (built from the same inverse the wire bridge uses,
#: so this test needs nothing out of ``driver/``).
TENSIX_COORD_MAP = {
    Wormhole.physical_noc0_coord_from_unified_worker((ux, uy)): (ux, uy)
    for ux in range(18, 26)
    for uy in range(16, 26)
}

#: The worker every Wormhole example runs on, and the one the device builds by
#: default — so it stands in for the server's eager pool.
PEER = (1, 1)
#: A worker the host has said things to but not yet launched on.
LATE = (2, 1)

SRC_ADDR = 0x20000
DST_ADDR = 0x30000
FIRMWARE_ADDR = 0x1000
GO_ADDR = 0x4A0
GO_GO = b"\x00\x00\x00\x80"
GO_INIT = b"\x00\x00\x00\x40"

PAYLOAD = bytes(range(16)) * 2
FIRMWARE = bytes(range(32, 64))


def _build(hook=True):
    """A fabric + Wormhole with ``PEER`` eager and everything else on demand."""
    device = Device(Wormhole, TENSIX_COORD_MAP, cycles_per_poll=100)
    fabric = Fabric()
    pool = LazyTensixPool(fabric, device, TENSIX_COORD_MAP, eager=[PEER])
    if not hook:
        # Leave the host triggers in place and take the NoC one away — the
        # design the ROADMAP sketch would have shipped without the survey's
        # warning.
        device.tt_device.set_directory_miss_hook(None)
    return device, fabric, pool


def _peer_writes_to(device, pool, dest, payload):
    """``PEER`` issues a NoC 0 unicast write into ``dest``'s L1 at DST_ADDR.

    Driven through the initiator's command registers, which is what a kernel's
    ``noc_async_write`` ends up doing, so the request takes the same
    ``resolve_destination`` path a real one does.
    """
    tile = device.tt_device.tile_directory[pool.coord_map[PEER]]
    device.tt_device.write(tile.get_coord_pair(), SRC_ADDR, payload)

    initiator = tile.noc0_router.request_initiators[0]
    # Wormhole packs the destination coord into the MID address register.
    initiator.ret_addr_mid = (dest[0] << 4) | (dest[1] << 10)
    initiator.target_addr_low = SRC_ADDR
    initiator.ret_addr_low = DST_ADDR
    initiator.at_len_be = len(payload)
    initiator.ctrl = 2 | (1 << 4)  # mode 2 = write, response marked
    initiator.cmd_ctrl = 1
    initiator.initiate()

    for _ in range(4000):
        if all(not f for f in tile.noc0_router.outstanding_noc_requests.values()):
            return
        device.tt_device.run(1)
    raise AssertionError("peer's write was never acknowledged")


def _l1(fabric, coord, addr, size):
    """Read a worker's L1 the way the host does — through the fabric.

    Deliberately not ``tt_device.read``: a worker that was never materialised
    has no tile to read, and that would turn the race below into a crash-shaped
    failure. Through the fabric an unmaterialised worker answers with zeros,
    which is exactly the silently-wrong answer being guarded against.
    """
    return bytes(fabric.read(coord, addr, size))


# ---------------------------------------------------------------------------
# The race.
# ---------------------------------------------------------------------------


def test_a_peers_packet_materialises_the_worker_it_is_addressed_to():
    """The deliverable: data sent to a not-yet-launched worker still arrives."""
    device, fabric, pool = _build()

    # The host has written this worker's firmware but has not launched on it,
    # so nothing has told the simulator the worker is used.
    fabric.write(LATE, FIRMWARE_ADDR, FIRMWARE)
    fabric.write(LATE, GO_ADDR, GO_INIT)  # grid-wide init: not a launch
    assert isinstance(fabric.cores[LATE], DeferredTensixCore)
    assert LATE not in pool.materialised

    _peer_writes_to(device, pool, LATE, PAYLOAD)

    # The value first, because the value is the point: the failure this guards
    # against is not a crash but zeros where the peer's data should be.
    assert _l1(fabric, LATE, DST_ADDR, len(PAYLOAD)) == PAYLOAD
    # ...and the journal was replayed *before* the packet landed, so the host's
    # earlier writes are there too and neither has clobbered the other.
    assert _l1(fabric, LATE, FIRMWARE_ADDR, len(FIRMWARE)) == FIRMWARE
    assert LATE in pool.materialised
    assert pool.on_demand == [(LATE, "noc")]


def test_a_peers_packet_is_lost_without_the_directory_miss_hook():
    """The counterfactual that makes trigger 2 load-bearing.

    With only ``go=GO`` materialising workers the packet is null-routed: the
    peer's write is acknowledged, the run completes, and the answer is silently
    wrong. Pinned so that removing the hook fails the test above rather than
    quietly restoring this.
    """
    device, fabric, pool = _build(hook=False)

    fabric.write(LATE, FIRMWARE_ADDR, FIRMWARE)
    _peer_writes_to(device, pool, LATE, PAYLOAD)

    assert LATE not in pool.materialised

    # The host launches on it later; by then the peer's data is gone for good.
    fabric.write(LATE, GO_ADDR, GO_GO)
    assert LATE in pool.materialised
    assert _l1(fabric, LATE, FIRMWARE_ADDR, len(FIRMWARE)) == FIRMWARE
    assert _l1(fabric, LATE, DST_ADDR, len(PAYLOAD)) == bytes(len(PAYLOAD))


def test_a_noc1_miss_resolves_through_the_mirror():
    """NoC 1 addresses a tile by ``(GRID-1-x, GRID-1-y)``; the hook unmirrors."""
    device, fabric, pool = _build()
    mirror = device.tt_device.noc1_mirror(LATE)

    pool.on_directory_miss(1, mirror)

    assert LATE in pool.materialised
    assert device.tt_device.noc_1_directory[mirror] is (
        device.tt_device.tile_directory[TENSIX_COORD_MAP[LATE]].noc1_router
    )


def test_a_noc1_miss_falls_back_to_the_canonical_reading():
    """Wormhole's ``x=9`` column mirrors onto ``x=0``, which holds no worker,
    so those NoC 1 keys are claimed canonically and the hook must say so."""
    device, fabric, pool = _build()
    canonical = (9, 1)
    assert device.tt_device.noc1_mirror(canonical) not in TENSIX_COORD_MAP

    pool.on_directory_miss(1, canonical)

    assert canonical in pool.materialised
    assert device.tt_device.noc_1_directory[canonical] is (
        device.tt_device.tile_directory[TENSIX_COORD_MAP[canonical]].noc1_router
    )


def test_a_miss_on_something_that_is_not_a_worker_materialises_nothing():
    device, fabric, pool = _build()

    pool.on_directory_miss(0, (0, 11))  # DRAM
    pool.on_directory_miss(0, (13, 13))  # off-grid

    assert pool.materialised == {PEER}


# ---------------------------------------------------------------------------
# The host trigger, and the journal.
# ---------------------------------------------------------------------------


def test_go_go_materialises_and_replays_in_arrival_order():
    device, fabric, pool = _build()

    fabric.write(LATE, FIRMWARE_ADDR, FIRMWARE)
    fabric.write(LATE, FIRMWARE_ADDR + 4, b"\xaa\xbb\xcc\xdd")  # overwrites part
    fabric.write(LATE, GO_ADDR, GO_GO)

    assert LATE in pool.materialised
    assert pool.on_demand == [(LATE, "host")]
    expected = FIRMWARE[:4] + b"\xaa\xbb\xcc\xdd" + FIRMWARE[8:]
    assert _l1(fabric, LATE, FIRMWARE_ADDR, len(FIRMWARE)) == expected
    assert _l1(fabric, LATE, GO_ADDR, 4) == GO_GO


def test_a_write_after_the_deassert_materialises_and_settles_init(monkeypatch):
    """The ordinary trigger, and the deadline the journal has to respect.

    Once the host has released a core it believes that core is executing: it
    polled the go-message to DONE (out of the zero-fill) before writing a
    single kernel binary. So the first post-DEASSERT write both identifies the
    worker as used and marks the last moment the journal is still a faithful
    picture — the firmware has to run the init state *before* the binaries
    land, not after.
    """
    device, fabric, pool = _build()
    monkeypatch.setattr(device.tt_device, "run", lambda n: None)
    settled = []
    monkeypatch.setattr(
        device, "settle_go_message", lambda u, a: settled.append((u, a))
    )

    fabric.write(LATE, FIRMWARE_ADDR, FIRMWARE)
    fabric.write(LATE, GO_ADDR, GO_INIT)
    fabric.deassert_reset(LATE)
    assert LATE not in pool.materialised  # still just a journal

    fabric.write(LATE, 0x8000, b"kernel binary")

    assert pool.on_demand == [(LATE, "host")]
    assert settled == [(TENSIX_COORD_MAP[LATE], GO_ADDR)]
    assert _l1(fabric, LATE, 0x8000, 13) == b"kernel binary"


def test_a_noc_materialised_worker_settles_init_at_its_first_host_op(monkeypatch):
    """The NoC trigger fires inside the clock, so its catch-up is deferred.

    Pumping from inside ``tt_device.run`` would re-enter the clock with tiles
    mid-cycle; the settle waits for the first host operation instead, which is
    the first moment the simulator is provably not inside a run.
    """
    device, fabric, pool = _build()
    fabric.write(LATE, GO_ADDR, GO_INIT)
    fabric.deassert_reset(LATE)

    pool.on_directory_miss(0, LATE)

    core = fabric.cores[LATE]
    assert isinstance(core, _CatchUpTensixCore)
    monkeypatch.setattr(device.tt_device, "run", lambda n: None)
    settled = []
    monkeypatch.setattr(
        device, "settle_go_message", lambda u, a: settled.append((u, a))
    )

    core.read(0x0, 4)
    core.read(0x0, 4)

    assert settled == [(TENSIX_COORD_MAP[LATE], GO_ADDR)]  # once, not twice


def test_a_deferred_worker_reads_back_zeros_before_it_exists():
    """``go=INIT`` must read back as ``RUN_MSG_DONE`` or device init hangs.

    The journal is deliberately write-only: serving reads out of it would echo
    the host's own ``INIT`` back at it, and ``wait_until_cores_done`` would spin
    for ever on a worker no kernel ever runs on.
    """
    _device, fabric, _pool = _build()

    fabric.write(LATE, GO_ADDR, GO_INIT)

    assert fabric.read(LATE, GO_ADDR, 4) == b"\x00\x00\x00\x00"


def test_reset_transitions_are_journalled_with_the_writes():
    """The launch message's ``enables`` must be in L1 before the DEASSERT that
    reads it, so replay preserves order across kinds, not just within writes."""
    _device, fabric, pool = _build()

    fabric.assert_reset(LATE)
    fabric.write(LATE, FIRMWARE_ADDR, FIRMWARE)
    fabric.deassert_reset(LATE)

    assert [kind for kind, _, _ in fabric.cores[LATE].journal] == ["a", "w", "d"]


def test_only_functional_workers_are_deferred():
    """eth / pcie / arc / router-only coords still fall through to NullCore."""
    _device, fabric, pool = _build()

    assert pool.deferred_core((1, 0)) is None  # eth
    assert pool.deferred_core((0, 11)) is None  # DRAM
    assert pool.deferred_core(LATE) is not None


def test_materialising_twice_is_a_no_op():
    device, fabric, pool = _build()

    fabric.write(LATE, FIRMWARE_ADDR, FIRMWARE)
    first = pool.materialise(LATE)
    core = fabric.cores[LATE]
    second = pool.materialise(LATE)

    assert first == second
    assert fabric.cores[LATE] is core
    assert _l1(fabric, LATE, FIRMWARE_ADDR, len(FIRMWARE)) == FIRMWARE


def test_a_pinned_pool_installs_neither_trigger():
    """``TT_SIM_TENSIX_COORDS`` still means exactly these workers and no more."""
    device = Device(Wormhole, TENSIX_COORD_MAP, cycles_per_poll=100)
    fabric = Fabric()
    device.ensure_tensix_tile(PEER)

    assert fabric.core_factory is None
    nui = device.tt_device.tile_directory[TENSIX_COORD_MAP[PEER]].noc0_router
    assert nui.directory_miss_hook is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
