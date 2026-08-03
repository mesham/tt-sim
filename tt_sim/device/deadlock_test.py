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
    return device, coord, detector


def _launch_brisc(device, coord, program):
    device.write(coord, BRISC_START, program)
    device.deassert_soft_reset(coord, core_type=BabyRISCVCoreType.BRISC)


def _loop(instructions):
    """``instructions`` NOPs followed by a jump back to the first of them."""
    body = NOP * (instructions - 1)
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
    device, coord, detector = _device_with_watchdog()
    scans = 0
    inner = detector._sample

    def counting(cycle):
        nonlocal scans
        scans += 1
        return inner(cycle)

    detector._sample = counting
    _launch_brisc(device, coord, _loop(64))

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
