"""The device profiler's readback: the host must not read the answer too early.

The bug this pins, twice observed and twice diagnosed from scratch:
``perfbench/mechbench`` gets a ``profile_log_device.csv`` back with nothing but
its header — no counter samples *and no zone markers* — whenever a run is long
enough or the cost model is on. Profiling nekbone needed
``TT_SIM_CYCLES_PER_POLL=5000`` for the same reason.

Nothing is lost in transit. tt-metal's BRISC writes ``RUN_MSG_DONE`` to the go
mailbox **before** ``finish_profiler()`` publishes the run (in 0.74,
``brisc.cc:575`` sits inside the ``DeviceZoneScopedMainN("BRISC-FW")`` block
whose destructor calls it). The host stops driving the clock at DONE, and its
very next wire message is the control-vector read — so the firmware gets
``cycles_per_poll`` cycles to do a job that measurably takes thousands, and
``DeviceProfiler::readRiscProfilerResults`` early-returns on
``HOST_BUFFER_END_INDEX_BR_ER == 0 == ..._NC`` for both its DRAM and its L1
source.

The program here is that firmware tail, reduced to the four stores that matter,
and the driver is the host's own sequence. No tt-metal and no sockets: a real
Wormhole under the wire bridge's :class:`~tt_sim.bridge.device.Device`.

``test_the_control_vector_read_sees_the_published_run`` is the reproduction —
it fails on a tree without :meth:`Device.settle_profiler_flush`, reading
``HOST_BUFFER_END_INDEX_BR_ER = 0`` and ``PROFILER_DONE = 0``. Everything else
in this file exists to stop the cure being worse than the disease: the
fingerprint must not match ordinary traffic, and a firmware that never
publishes must not pump for ever.
"""

import pytest

from tt_sim.bridge.device import Device
from tt_sim.device.wormhole import Wormhole
from tt_sim.perf.noc_issue_loop import addi, bne, jal, lui, sw
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

TENSIX_COORD_MAP = {
    Wormhole.physical_noc0_coord_from_unified_worker((ux, uy)): (ux, uy)
    for ux in range(18, 26)
    for uy in range(16, 26)
}
WORKER = (1, 1)

PROGRAM_ADDR = 0x0
GO_ADDR = 0x4A0
CTRL_ADDR = 0xB50

#: ``kernel_profiler::ControlBuffer`` indices this test writes or reads.
HOST_END_BR, HOST_END_NC = 0, 1
DEVICE_END_BR = 5
PROFILER_DONE = 19

#: Stand-ins for a real run's end indices — the mechbench ``elw`` arm reports
#: exactly these (368 words of BRISC markers, 12 of NCRISC's).
BRISC_MARKERS = 368
NCRISC_MARKERS = 12

#: Loop trips in the stand-in for ``risc_finished_profiling()`` plus the
#: publish loop. Two instructions a trip, so ~4 000 cycles — comfortably more
#: than the 100-cycle poll budget and comfortably less than the settle cap.
TAIL_TRIPS = 2000

CYCLES_PER_POLL = 100
POLL_LIMIT = 200


def _li(rd, value):
    """``li rd, value`` as ``lui`` + ``addi``, the way gcc expands it."""
    upper = (value + 0x800) >> 12
    lower = value - (upper << 12)
    return [lui(rd, upper), addi(rd, rd, lower)]


def _firmware_tail(publishes=True):
    """The four stores of ``finish_profiler()``, in the firmware's order.

    ``RUN_MSG_DONE`` first — that is the whole problem — then the device-side
    end index that ``risc_finished_profiling()`` stamps, then a long stretch of
    work, then the host-side end indices and ``PROFILER_DONE`` that the host is
    actually waiting for. ``publishes=False`` stops after the work, standing in
    for firmware that never publishes at all.
    """
    words = []
    words += _li(1, GO_ADDR)
    words += _li(2, CTRL_ADDR)
    words += [sw(0, 1, 0)]  # go_msg.signal = RUN_MSG_DONE
    words += _li(4, BRISC_MARKERS)
    words += [sw(4, 2, DEVICE_END_BR * 4)]
    words += _li(3, TAIL_TRIPS)
    words += [addi(3, 3, -1), bne(3, 0, -4)]
    if publishes:
        words += [sw(4, 2, HOST_END_BR * 4)]
        words += _li(5, NCRISC_MARKERS)
        words += [sw(5, 2, HOST_END_NC * 4)]
        words += _li(6, 1)
        words += [sw(6, 2, PROFILER_DONE * 4)]
    words += [jal(0, 0)]
    return words


def _control_vector():
    """The 32-word vector tt-metal writes before a launch.

    Word values captured from a Blackhole ``mechbench`` run: the DRAM profiler
    address, the flat core id and the cores-per-DRAM count, everything else
    zero.
    """
    words = [0] * 32
    words[12] = 64  # DRAM_PROFILER_ADDRESS_DEFAULT
    words[16] = 1  # FLAT_ID
    words[17] = 19  # CORE_COUNT_PER_DRAM
    return b"".join(conv_to_bytes(w) for w in words)


def _launch(control_vector=None, publishes=True):
    """Device + worker driven exactly as the tt-metal host drives one.

    Returns ``(device, unified)`` with BRISC released and the go message not
    yet polled.
    """
    device = Device(Wormhole, TENSIX_COORD_MAP, cycles_per_poll=CYCLES_PER_POLL)
    unified = device.ensure_tensix_tile(WORKER)
    if control_vector is not None:
        device.write(unified, CTRL_ADDR, control_vector)
    for index, word in enumerate(_firmware_tail(publishes)):
        device.tt_device.write(unified, PROGRAM_ADDR + 4 * index, conv_to_bytes(word))
    device.write(unified, GO_ADDR, b"\x00\x00\x00\x80")  # go=GO
    device.deassert_reset(unified)
    return device, unified


def _poll_until_done(device, unified):
    """The host's ``wait_until_cores_done``, pumping a poll budget per read."""
    for _ in range(POLL_LIMIT):
        if device.read(unified, GO_ADDR, 4)[3] == 0x00:
            return True
    return False


def _words(control):
    return [
        conv_to_uint32(bytes(control[i * 4 : i * 4 + 4]))
        for i in range(len(control) // 4)
    ]


# -- the reproduction -------------------------------------------------------


def test_the_control_vector_read_sees_the_published_run():
    """The one that fails without the settle.

    Without it the read lands ``cycles_per_poll`` cycles after DONE, with the
    firmware still inside its tail: ``HOST_BUFFER_END_INDEX_BR_ER`` and
    ``PROFILER_DONE`` both zero, which is exactly the header-only CSV.
    """
    device, unified = _launch(control_vector=_control_vector())
    assert _poll_until_done(device, unified), "firmware never reported DONE"

    control = _words(device.read(unified, CTRL_ADDR, 128))
    device.tt_device.shutdown()

    assert control[DEVICE_END_BR] == BRISC_MARKERS, (
        "the markers were never in L1 — this test's stand-in firmware is broken, "
        "not the bridge"
    )
    assert control[HOST_END_BR] == BRISC_MARKERS
    assert control[HOST_END_NC] == NCRISC_MARKERS
    assert control[PROFILER_DONE] == 1


def test_the_settle_is_reported_and_costs_only_the_tail():
    device, unified = _launch(control_vector=_control_vector())
    _poll_until_done(device, unified)
    device.read(unified, CTRL_ADDR, 128)
    from tt_sim.bridge.device import profiler_flush_summary

    summary = profiler_flush_summary(device)
    device.tt_device.shutdown()

    assert device.profiler_flush_settles == 1
    assert device.profiler_flush_timeouts == 0
    # Two instructions a trip, so the loop is ~2 * TAIL_TRIPS cycles; the last
    # go-message poll already paid one 100-cycle chunk of it, and the settle
    # overshoots by at most one chunk. What matters is that the settle pays the
    # tail and not the cap.
    assert (
        2 * TAIL_TRIPS - 2 * CYCLES_PER_POLL
        <= device.profiler_flush_cycles
        <= 2 * TAIL_TRIPS + 4 * CYCLES_PER_POLL
    )
    assert "profiler flush: 1 settles" in summary


def test_a_second_read_does_not_settle_again():
    """One settle per launch: the arm is a ``go=GO``, not a control-vector read."""
    device, unified = _launch(control_vector=_control_vector())
    _poll_until_done(device, unified)
    device.read(unified, CTRL_ADDR, 128)
    first = device.profiler_flush_cycles
    device.read(unified, CTRL_ADDR, 128)
    device.tt_device.shutdown()

    assert device.profiler_flush_settles == 1
    assert device.profiler_flush_cycles == first


# -- the mechanism must not fire on anything else ---------------------------


def test_without_a_control_vector_write_nothing_is_armed():
    """A run with the profiler off never writes one, so nothing is perturbed.

    The counterfactual for the test above: the identical program and the
    identical host sequence, minus the control-vector write. The read comes
    back mid-tail — the pre-fix behaviour — and not one extra cycle was run.
    """
    device, unified = _launch(control_vector=None)
    assert _poll_until_done(device, unified)
    control = _words(device.read(unified, CTRL_ADDR, 128))
    from tt_sim.bridge.device import profiler_flush_summary

    summary = profiler_flush_summary(device)
    device.tt_device.shutdown()

    assert device.profiler_flush_settles == 0
    assert device.profiler_flush_cycles == 0
    assert summary == ""
    assert control[HOST_END_BR] == 0
    assert control[PROFILER_DONE] == 0


def test_a_128_byte_payload_that_is_not_a_control_vector_does_not_arm():
    """The fingerprint is on the vector's shape, and it has to be tight."""
    payload = bytes(((i * 37) & 0xFF) for i in range(128))
    device, unified = _launch(control_vector=payload)
    assert _poll_until_done(device, unified)
    device.read(unified, CTRL_ADDR, 128)
    device.tt_device.shutdown()

    assert device.profiler_flush_settles == 0
    assert device.profiler_flush_cycles == 0


@pytest.mark.parametrize(
    "index,value,why",
    [
        (HOST_END_BR, 1, "a host end index is already set"),
        (DEVICE_END_BR, 1, "a device end index is already set"),
        (10, 1, "FW_RESET_H is set"),
        (PROFILER_DONE, 1, "PROFILER_DONE is set"),
        (20, 1, "TRACE_REPLAY_STATUS is set"),
        (26, 1, "a word past the last ControlBuffer member is set"),
        (17, 0, "CORE_COUNT_PER_DRAM is zero, which the firmware divides by"),
    ],
)
def test_the_fingerprint_rejects(index, value, why):
    words = _words(_control_vector())
    words[index] = value
    mutated = b"".join(conv_to_bytes(w) for w in words)
    assert not Device._looks_like_profiler_control_vector(mutated), why


def test_the_fingerprint_accepts_the_captured_vectors():
    """All three control-vector writes a Blackhole mechbench run makes.

    Two at device init (before and after the DRAM profiler address is handed
    over) and one between launches, which additionally carries the per-RISC
    DRAM addresses in words 21-25.
    """
    captured = []
    words = [0] * 32
    words[16], words[17] = 1, 19
    captured.append(list(words))
    words[12] = 64
    captured.append(list(words))
    for i in range(21, 26):
        words[i] = 64
    captured.append(list(words))
    for vector in captured:
        payload = b"".join(conv_to_bytes(w) for w in vector)
        assert Device._looks_like_profiler_control_vector(payload)


def test_a_payload_of_the_wrong_length_is_not_a_control_vector():
    assert not Device._looks_like_profiler_control_vector(b"\x00" * 64)
    assert not Device._looks_like_profiler_control_vector(b"\x00" * 132)


# -- the cure must be bounded ----------------------------------------------


def test_firmware_that_never_publishes_gives_up_rather_than_hanging(monkeypatch):
    """The cap, and that it is paid once per worker rather than once a launch."""
    monkeypatch.setattr(Device, "_PROFILER_FLUSH_CAP", 5_000)
    device, unified = _launch(control_vector=_control_vector(), publishes=False)
    assert _poll_until_done(device, unified)

    control = _words(device.read(unified, CTRL_ADDR, 128))
    assert device.profiler_flush_timeouts == 1
    assert device.profiler_flush_cycles == 5_000
    assert control[PROFILER_DONE] == 0

    # Re-arm as a second launch would, and confirm the worker is not retried.
    device.write(unified, GO_ADDR, b"\x00\x00\x00\x80")
    device.read(unified, CTRL_ADDR, 128)
    from tt_sim.bridge.device import profiler_flush_summary

    summary = profiler_flush_summary(device)
    device.tt_device.shutdown()

    assert device.profiler_flush_timeouts == 1
    assert device.profiler_flush_cycles == 5_000
    assert "TIMED OUT" in summary


def test_publishing_is_not_the_end_of_the_wait(monkeypatch):
    """The second phase: the pushes have to land, not merely be sent.

    ``PROFILER_DONE`` is set one instruction after a flush that waits on the
    NIU's *sent* counter, so the last RISCs' payloads can still be in the NoC.
    Measured on ``mechbench elw``: stopping at ``PROFILER_DONE`` recovered
    BRISC, NCRISC and TRISC 0 and dropped TRISC 1 and TRISC 2 — the last two
    pushes the publish loop issues. Here the landing check is held off for a
    known number of chunks and the settle must keep running for them.
    """
    held = 3
    calls = []
    real = Device._profiler_writes_landed

    def landed(self, unified):
        calls.append(unified)
        if len(calls) <= held:
            return False
        return real(self, unified)

    monkeypatch.setattr(Device, "_profiler_writes_landed", landed)
    device, unified = _launch(control_vector=_control_vector())
    _poll_until_done(device, unified)
    device.read(unified, CTRL_ADDR, 128)
    without_hold = 2 * TAIL_TRIPS
    device.tt_device.shutdown()

    assert device.profiler_flush_settles == 1
    # The publish point plus the held chunks; the pre-check consumes one call.
    assert device.profiler_flush_cycles >= without_hold - 2 * CYCLES_PER_POLL
    assert len(calls) > held


def test_the_landing_check_is_scoped_to_the_worker_and_dram():
    """Exactly the tiles a profiler push travels between, and no others.

    Widening it to the whole grid would let a peer still running a kernel hold
    the wait open to the cap, so the scope is a decision and not an accident.
    """
    device, unified = _launch(control_vector=_control_vector())
    seen = []
    for coord, tile in device.tt_device.tile_directory.items():
        for router in (tile.noc0_router, tile.noc1_router):
            original = router.is_clock_idle

            def probe(_original=original, _coord=coord):
                seen.append(_coord)
                return _original()

            router.is_clock_idle = probe

    assert device._profiler_writes_landed(unified)
    expected = {t.get_coord_pair() for t in device.tt_device.dram_tiles} | {unified}
    device.tt_device.shutdown()

    assert set(seen) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
