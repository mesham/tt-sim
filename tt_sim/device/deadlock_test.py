"""Tests for the progress watchdog (:mod:`tt_sim.device.deadlock`).

The watchdog is sampled rather than polled per cycle (see the module
docstring), so there are two things to pin down and they pull in opposite
directions:

* it still **fires** on a genuinely wedged device, within a stated latency
  bound; and
* it does **not** fire on a device that is progressing — including the one
  workload sampling alone would get wrong, a loop whose period is exactly the
  sample interval.

Plus a cost guard: the number of full signature scans over a run is the run
length divided by the sample interval, not the run length.

Every behavioural test runs against **both architectures**: the watchdog was
Wormhole-only for a long time, so a wedged Blackhole kernel hung with no
``[DEADLOCK]`` report at all. It is wired in ``TT_Device`` now — these
parametrised cases are what keeps it wired.
"""

import re

import pytest

from tt_sim.device.blackhole import Blackhole
from tt_sim.device.deadlock import (
    _CONFIRM_TICKS,
    DeadlockDetector,
    deadlock_config_from_env,
)
from tt_sim.device.wormhole import Wormhole
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType

# BRISC boots at L1 offset 0 on both architectures.
BRISC_START = 0x0
NOP = b"\x13\x00\x00\x00"  # addi x0, x0, 0
INC = b"\x93\x80\x10\x00"  # addi x1, x1, 1 -- moves architectural state
WEDGE = b"\x6f\x00\x00\x00"  # jal x0, 0  -> `j .`

THRESHOLD = 800  # -> sample interval 100

# Every architecture whose device wires the watchdog, i.e. all of them.
DEVICES = pytest.mark.parametrize(
    "device_class", [Wormhole, Blackhole], ids=lambda c: c.__name__
)


def _device_with_watchdog(device_class=Wormhole, threshold=THRESHOLD):
    """A one-Tensix device whose watchdog has a test-sized window."""
    device = device_class()
    device.reset()
    coord = next(pair for pair, tile in device.tile_directory.items() if tile.is_tensix)
    # The device already built a detector (with the production 50,000-cycle
    # window); swap in one whose window a unit test can afford to wait out.
    detector = DeadlockDetector(
        threshold, True, [device.tile_directory[coord]], device.dram_tiles
    )
    device.deadlock_detector = detector
    device.clocks[0].on_tick = detector.tick
    device.clocks[0].on_tick_wake = detector.next_sample_cycle
    return device, coord, detector


def _send_every_tile_dormant(device):
    """Make the pump believe every tile has nothing left to do.

    The Phase 4 pump strides over a cycle no tile clock asks for, and a tile
    asks by way of its ``next_wake_cycle`` probe. Today no in-tree tile
    answers "never" while one of its baby cores is out of soft reset
    (``TensixTile.next_wake_cycle`` short-circuits to ``cycle + 1`` for any
    ``soft_active`` core), so the only way to reach full dormancy *and* a
    wedged core is to say so directly — which is precisely the hazard: the
    watchdog's latency bound must not rest on an invariant that lives in
    another module and that a new component's ``busy_until`` / arrival
    deadline could quietly break.
    """
    for tile in device.tile_directory.values():
        tile.clock.wake_probe = lambda cycle: None


def _launch_brisc(device, coord, program):
    device.write(coord, BRISC_START, program)
    device.deassert_soft_reset(coord, core_type=BabyRISCVCoreType.BRISC)


def _loop(instructions, body_instruction=NOP):
    """``instructions - 1`` copies of ``body_instruction``, then a jump back.

    The default body is a NOP, which is a loop making no observable progress —
    the watchdog is entitled to report it. Pass ``INC`` for a loop that is
    progressing (it moves a register every cycle) and so must stay quiet.
    """
    body = body_instruction * (instructions - 1)
    back = -(4 * (instructions - 1))
    # jal x0, offset -- J-type immediate, offset is a multiple of 2.
    imm = back & 0x1FFFFF
    encoded = (
        ((imm >> 20) & 0x1) << 31
        | ((imm >> 1) & 0x3FF) << 21
        | ((imm >> 11) & 0x1) << 20
        | ((imm >> 12) & 0xFF) << 12
        | 0x6F
    )
    return body + encoded.to_bytes(4, "little")


def _spin_across_a_bucket_boundary():
    """A four-instruction loop placed so it straddles a 64-byte PC bucket.

    The shape that defeated the old instantaneous-PC signature: the loop body
    lives at 0x38-0x44, so consecutive samples land in bucket 0 or bucket 1
    depending on nothing in particular, and a signature built from "which
    bucket is this core in right now" changed on its own every few samples and
    re-baselined the stall window for ever. Every real spin looks like this —
    tt-metal's go-message wait loop spans two buckets — so the watchdog stayed
    silent on a device burning cycles and making no progress.
    """
    body_at = 0x38
    prog = bytearray(NOP * (body_at // 4))
    prog += NOP * 3  # 0x38, 0x3c, 0x40
    back = -12  # 0x44 -> 0x38
    imm = back & 0x1FFFFF
    encoded = (
        ((imm >> 20) & 0x1) << 31
        | ((imm >> 1) & 0x3FF) << 21
        | ((imm >> 11) & 0x1) << 20
        | ((imm >> 12) & 0xFF) << 12
        | 0x6F
    )
    prog += encoded.to_bytes(4, "little")
    return bytes(prog)


@DEVICES
def test_fires_on_a_spinning_wedge_that_straddles_a_pc_bucket(capsys, device_class):
    """The live-spin case: a core looping, not frozen, and getting nowhere."""
    device, coord, _detector = _device_with_watchdog(device_class)
    _launch_brisc(device, coord, _spin_across_a_bucket_boundary())

    device.run(6 * THRESHOLD)

    err = capsys.readouterr().err
    assert "[DEADLOCK" in err, err
    assert "BRISC: oscillating in [0x38, 0x44]" in err, err


@DEVICES
def test_quiet_on_a_tight_loop_that_is_making_progress(capsys, device_class):
    """The false positive the register component exists to prevent.

    The A/B against the test above: same loop length, same handful of PCs, same
    revisited code footprint — the only difference is that this one moves a
    register every iteration. A watchdog that judged spinning on code footprint
    alone would report a working kernel as a deadlock, which is worse than
    missing one.
    """
    device, coord, _detector = _device_with_watchdog(device_class)
    _launch_brisc(device, coord, _loop(4, body_instruction=INC))

    device.run(6 * THRESHOLD)

    assert "[DEADLOCK" not in capsys.readouterr().err


@DEVICES
def test_fires_on_a_wedged_device(capsys, device_class):
    device, coord, _detector = _device_with_watchdog(device_class)
    _launch_brisc(device, coord, WEDGE)

    device.run(THRESHOLD + 4 * _CONFIRM_TICKS)

    err = capsys.readouterr().err
    assert "[DEADLOCK" in err, err
    assert "BRISC: frozen at 0x0" in err, err
    assert "TT_SIM_DEADLOCK=0" in err


@DEVICES
def test_wedged_device_reports_through_the_device_s_own_detector(capsys, device_class):
    """The watchdog every architecture's ``__init__`` wires, not a hand-built
    one: TT_SIM_DEADLOCK_THRESHOLD in, ``[DEADLOCK]`` on stderr out."""
    device = device_class()
    device.reset()
    assert device.deadlock_detector.enabled
    assert device.clocks[0].on_tick == device.deadlock_detector.tick
    # Retune in place rather than rebuilding — same object the device wired.
    device.deadlock_detector.threshold = THRESHOLD
    device.deadlock_detector.sample_interval = THRESHOLD // 8
    coord = next(pair for pair, tile in device.tile_directory.items() if tile.is_tensix)
    _launch_brisc(device, coord, WEDGE)

    device.run(2 * THRESHOLD)

    assert "[DEADLOCK" in capsys.readouterr().err


@DEVICES
def test_detection_latency_is_bounded(capsys, device_class):
    """Late by at most one sample interval plus the confirmation window."""
    device, coord, detector = _device_with_watchdog(device_class)
    _launch_brisc(device, coord, WEDGE)

    device.run(2 * THRESHOLD)

    reports = re.findall(r"\[DEADLOCK cycle=(\d+)\]", capsys.readouterr().err)
    assert reports, "watchdog never fired on a wedged device"
    first = int(reports[0])
    assert THRESHOLD <= first <= THRESHOLD + detector.sample_interval + _CONFIRM_TICKS


@DEVICES
def test_fires_on_a_fully_dormant_wedged_device(capsys, device_class):
    """The blind spot the detector's own wake cycle closes.

    A device every tile of which reports dormant is strided over in one jump
    per ``run()`` call, so ``on_tick`` fires once and a wedged core is never
    sampled. With ``on_tick_wake`` wired the pump has to stop at each sample,
    and the report — including its cycle — is identical to the awake case.
    """
    device, coord, detector = _device_with_watchdog(device_class)
    _launch_brisc(device, coord, WEDGE)
    device.run(4)
    _send_every_tile_dormant(device)

    device.run(2 * THRESHOLD)

    err = capsys.readouterr().err
    assert "[DEADLOCK" in err, err
    assert "BRISC: frozen at 0x0" in err, err
    first = int(re.findall(r"\[DEADLOCK cycle=(\d+)\]", err)[0])
    assert THRESHOLD <= first <= THRESHOLD + detector.sample_interval + _CONFIRM_TICKS


@DEVICES
def test_unwired_wake_probe_is_the_blind_spot(capsys, device_class):
    """Guards the guard: without ``on_tick_wake`` the same device is silent.

    If this ever starts reporting, the fix above has become dead code and the
    test that pins it is no longer testing anything.
    """
    device, coord, _detector = _device_with_watchdog(device_class)
    device.clocks[0].on_tick_wake = None
    _launch_brisc(device, coord, WEDGE)
    device.run(4)
    _send_every_tile_dormant(device)

    device.run(50 * THRESHOLD)

    assert capsys.readouterr().err == ""


@DEVICES
def test_dormant_device_is_still_sampled_once_per_interval(device_class):
    """Striding does not cost samples — one per interval, dormant or not."""
    device, _coord, detector = _device_with_watchdog(device_class)
    sampled = []
    inner = detector._sample
    detector._sample = lambda cycle: (sampled.append(cycle), inner(cycle))[1]

    # Every core is in soft reset, so every tile clock goes dormant on the
    # first tick and the pump would otherwise jump the whole window.
    cycles = 20 * detector.sample_interval
    device.run(cycles)

    assert device.clocks[0].stride_skipped_cycles > 0, "the pump never strode"
    assert len(sampled) >= cycles // detector.sample_interval


@DEVICES
def test_quiet_while_every_core_is_in_reset(capsys, device_class):
    device, _coord, _detector = _device_with_watchdog(device_class)

    device.run(4 * THRESHOLD)

    assert capsys.readouterr().err == ""


@DEVICES
def test_quiet_on_a_loop_whose_period_is_the_sample_interval(capsys, device_class):
    """The case sampling alone would call a deadlock, and the confirmation
    pass rejects: a core looping with exactly the sampling period, so every
    sample point sees the identical PC."""
    device, coord, detector = _device_with_watchdog(device_class)
    _launch_brisc(device, coord, _loop(detector.sample_interval))

    device.run(6 * THRESHOLD)

    assert "[DEADLOCK" not in capsys.readouterr().err


def test_sampling_scans_once_per_interval():
    """The cost claim, on a device that is *working*.

    The workload has to be a progressing one: a core in an infinite NOP loop is
    a wedge by any reading, and the watchdog now says so, which costs a
    confirmation burst of per-cycle scans per window. Those bursts are bounded
    and only happen on a device that has already failed — they must not be
    what this measures.
    """
    device, coord, detector = _device_with_watchdog()
    scans = 0
    inner = detector._sample

    def counting(cycle):
        nonlocal scans
        scans += 1
        return inner(cycle)

    detector._sample = counting
    _launch_brisc(device, coord, _loop(64, body_instruction=INC))

    cycles = 20 * detector.sample_interval
    device.run(cycles)

    # One scan per interval, and nothing anywhere near one per cycle.
    assert scans <= cycles // detector.sample_interval + 2
    assert scans * 10 < cycles


def test_disabled_detector_never_samples():
    detector = DeadlockDetector(THRESHOLD, False, [], [])
    scanned = []
    detector._sample = scanned.append
    for cycle in range(1000):
        detector.tick(cycle)
    assert scanned == []


def test_sample_interval_follows_the_threshold():
    enabled, threshold = deadlock_config_from_env({"TT_SIM_DEADLOCK_THRESHOLD": "80"})
    assert enabled
    assert threshold == 80
    assert DeadlockDetector(threshold, True, [], []).sample_interval == 10
    # A threshold small enough to want per-cycle resolution still gets it.
    assert DeadlockDetector(4, True, [], []).sample_interval == 1
