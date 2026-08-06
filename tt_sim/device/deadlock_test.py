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

from tt_sim.arch import WORMHOLE_PROFILE
from tt_sim.device.blackhole import Blackhole
from tt_sim.device.deadlock import (
    _CONFIRM_TICKS,
    _WEDGE_CONFIRM_CYCLES,
    DEFAULT_UNIT_STALL_THRESHOLD,
    DeadlockDetector,
    UnitWedgedError,
    deadlock_config_from_env,
    unit_stall_config_from_env,
)
from tt_sim.device.wormhole import Wormhole
from tt_sim.pe.rv.babyriscv import BabyRISCVCoreType
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixInstructionDecoder

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


class _FakeBlockedUnit:
    """Stands in for an unpacker: ``blocked_on()`` and a switch to unblock it.

    The real thing needs a compute kernel to reach the blocking site, and what
    is under test here is the detector's counting rule, not the unpacker's.
    ``tt_sim/pe/tensix/setdvalid_srcrow_test`` pins the unpacker end.
    """

    def __init__(self, waiting=(0, "UNPACR_NOP", "SrcB", 0)):
        self.waiting = waiting
        self.blocked = True

    def blocked_on(self):
        return self.waiting if self.blocked else None


def _watch(detector, unit, name="Unpacker 1", coord=(1, 1)):
    detector.stallable_units.append((coord, name, unit))


UNIT_STALL_THRESHOLD = 500


def _detector_with_unit_stall_check(threshold=UNIT_STALL_THRESHOLD):
    """A detector with no tiles: the global watchdog has nothing to say."""
    return DeadlockDetector(THRESHOLD, True, [], [], unit_stall_threshold=threshold)


def _pump(detector, cycles):
    for cycle in range(cycles):
        detector.tick(cycle)


def _pump_from(detector, start, cycles):
    """``_pump`` for a test that has to change the unit's state mid-run.

    ``_pump`` always restarts at cycle 0, which a detector already past its
    threshold would read as the clock rewinding; this one carries the cursor.
    """
    for cycle in range(start, start + cycles):
        detector.tick(cycle)
    return start + cycles


def test_reports_a_unit_blocked_past_the_threshold(capsys):
    detector = _detector_with_unit_stall_check()
    _watch(detector, _FakeBlockedUnit())

    _pump(detector, 2 * UNIT_STALL_THRESHOLD)

    err = capsys.readouterr().err
    assert "[UNIT STALL" in err, err
    assert "Unpacker 1" in err
    assert "UNPACR_NOP from thread 0" in err
    assert "waiting for SrcB bank 0" in err
    assert "TT_SIM_UNIT_STALL=0" in err


def test_unit_stall_report_is_late_by_at_most_one_sample_interval(capsys):
    """The unit is picked up at the sampling cadence, then counted per cycle."""
    detector = _detector_with_unit_stall_check()
    _watch(detector, _FakeBlockedUnit())

    _pump(detector, 3 * UNIT_STALL_THRESHOLD)

    first = int(re.findall(r"\[UNIT STALL cycle=(\d+)\]", capsys.readouterr().err)[0])
    assert (
        UNIT_STALL_THRESHOLD
        <= first
        <= UNIT_STALL_THRESHOLD + detector.sample_interval + 1
    )


def test_quiet_on_a_unit_that_unblocks_between_samples(capsys):
    """The false positive that made a sampled counter unusable.

    A kernel is a loop, so an unpacker that legitimately waits for the math
    thread blocks on *the same instruction* once per tile — and is therefore
    blocked at consecutive sample points. Counting samples cannot tell that
    apart from never having moved; counting cycles can, and this is the case
    that says so. The unit here is blocked at every sample point and never for
    more than a fraction of the threshold at a time.
    """
    detector = _detector_with_unit_stall_check()
    unit = _FakeBlockedUnit()
    _watch(detector, unit)
    period = detector.sample_interval

    for cycle in range(20 * UNIT_STALL_THRESHOLD):
        # Blocked across every sample point, clear in between.
        unit.blocked = (cycle % period) < period // 2
        detector.tick(cycle)

    assert "[UNIT STALL" not in capsys.readouterr().err


def test_quiet_on_a_unit_that_moves_to_a_different_instruction(capsys):
    detector = _detector_with_unit_stall_check()
    unit = _FakeBlockedUnit()
    _watch(detector, unit)

    for cycle in range(20 * UNIT_STALL_THRESHOLD):
        unit.waiting = (0, "UNPACR", "SrcB", cycle // (UNIT_STALL_THRESHOLD // 4) % 2)
        detector.tick(cycle)

    assert "[UNIT STALL" not in capsys.readouterr().err


def test_unit_stall_survives_every_core_going_back_into_reset(capsys):
    """The state the global watchdog rebaselines on, and this must not.

    A launch ends with every baby core back in soft reset. That is the exact
    moment a wedged unit becomes invisible to the progress signature — and the
    moment the real Blackhole ``UNPACR_NOP`` deadlock was left standing in.
    """
    device, _coord, detector = _device_with_watchdog()
    detector.unit_stall_threshold = UNIT_STALL_THRESHOLD
    _watch(detector, _FakeBlockedUnit())

    device.run(3 * UNIT_STALL_THRESHOLD)

    assert "[UNIT STALL" in capsys.readouterr().err


def test_unit_stall_is_reported_once_however_long_the_run(capsys):
    """The de-duplication rule, and why the old count was meaningless.

    Repeats carried no information — same unit, same latched instruction, same
    bank — but made the number of reports a function of *run length*: the same
    broken eight-core GEMM gave 576 on one run and 352 on a longer one. One
    report per unit per waited-on instruction makes the count "how many units
    are affected", which is a quantity worth reading.
    """
    detector = _detector_with_unit_stall_check()
    _watch(detector, _FakeBlockedUnit())

    _pump(detector, 40 * UNIT_STALL_THRESHOLD)

    err = capsys.readouterr().err
    assert len(re.findall(r"\[UNIT STALL cycle=", err)) == 1, err
    assert "will not repeat" in err


def test_a_recovered_unit_retracts_its_own_stall_report(capsys):
    """The half of the hint that settles it, without waiting for teardown.

    A deep pipeline recovers and says so; a wedge never does. That difference
    is what a user reads, not the cycle count.
    """
    detector = _detector_with_unit_stall_check()
    unit = _FakeBlockedUnit()
    _watch(detector, unit)

    at = _pump_from(detector, 0, 2 * UNIT_STALL_THRESHOLD)
    assert "[UNIT STALL cycle=" in capsys.readouterr().err
    unit.blocked = False
    _pump_from(detector, at, 4)

    err = capsys.readouterr().err
    assert "[UNIT STALL CLEARED cycle=" in err, err
    assert "Unpacker 1" in err
    assert "got SrcB bank 0 back" in err
    assert "deep pipeline, not a wedge" in err


def test_a_second_block_on_the_same_instruction_is_not_reported_again(capsys):
    """``pipestall`` blocks once per tile on the same instruction, for ever."""
    detector = _detector_with_unit_stall_check()
    unit = _FakeBlockedUnit()
    _watch(detector, unit)

    at = 0
    for _round in range(4):
        unit.blocked = True
        at = _pump_from(detector, at, 2 * UNIT_STALL_THRESHOLD)
        unit.blocked = False
        at = _pump_from(detector, at, 4)

    err = capsys.readouterr().err
    assert len(re.findall(r"\[UNIT STALL cycle=", err)) == 1, err
    assert len(re.findall(r"\[UNIT STALL CLEARED cycle=", err)) == 1, err


def test_a_wedged_unit_does_not_retract_its_report(capsys):
    """A block that "clears" because the tile was torn down is not a recovery.

    ``[UNIT WEDGED]`` is the authoritative signal and nothing may walk it back:
    printing ``[UNIT STALL CLEARED]`` after it would tell the user the proof was
    a false alarm. (The raise terminates a real run; a harness that catches it
    and keeps pumping — as this test does — must still not see a retraction.)
    """
    detector, state = _detector_with_a_fake_tile(in_reset=False)
    unit = _FakeBlockedUnit()
    _watch(detector, unit)

    at = _pump_from(detector, 0, 2 * UNIT_STALL_THRESHOLD)
    state.set(True)
    with pytest.raises(UnitWedgedError):
        _pump_from(detector, at, 2 * _WEDGE_CONFIRM_CYCLES)
    at += 2 * _WEDGE_CONFIRM_CYCLES
    unit.blocked = False
    _pump_from(detector, at, 4)

    err = capsys.readouterr().err
    assert "[UNIT WEDGED" in err, err
    assert "[UNIT STALL CLEARED cycle=" not in err


def test_unit_stall_check_can_be_disabled(capsys):
    detector = DeadlockDetector(
        THRESHOLD,
        True,
        [],
        [],
        unit_stall_enabled=False,
        unit_stall_threshold=UNIT_STALL_THRESHOLD,
    )
    _watch(detector, _FakeBlockedUnit())

    _pump(detector, 20 * UNIT_STALL_THRESHOLD)

    assert "[UNIT STALL" not in capsys.readouterr().err


@DEVICES
def test_every_unpacker_is_registered_for_the_stall_check(device_class):
    """Guards the wiring: a device's own detector watches its own unpackers."""
    device = device_class()
    device.reset()
    detector = device.deadlock_detector
    assert detector.unit_stall_enabled
    assert detector.unit_stall_threshold == DEFAULT_UNIT_STALL_THRESHOLD
    names = {name for _coord, name, _unit in detector.stallable_units}
    assert names == {"Unpacker 0", "Unpacker 1"}
    assert len(detector.stallable_units) == 2 * len(detector.tile_cores)


def _wedged_unpacker():
    """A really-blocked unpacker, reached the way the Blackhole bug reaches it.

    ``UNPACR_NOP`` in its give-Src-to-the-Matrix-Unit form hands SrcB over; the
    zero-Src form issued straight after waits for that same bank to come back,
    and nothing here ever clears dvalid. That is the acquire-without-release
    shape of ``perfbench/tensixbench --dvalid-unpacr-nop`` (ROADMAP.md,
    "Unpacker dvalid deadlock"), with no device and no kernel around it.
    """
    backend = TensixCoProcessor(
        None,
        WORMHOLE_PROFILE.tensix_cfg_state_size,
        WORMHOLE_PROFILE.tensix_thd_state_size,
    ).getBackend()
    unpacker = backend.unpacker_units[1]
    unpacr_nop = 0x43 << 24 | 1 << 23  # opcode, unpacker_select = SrcB
    for noop_mode in (0x7, 0x1):  # give SrcB to the FPU, then wait for it back
        info = TensixInstructionDecoder.getInstructionInfo(unpacr_nop | noop_mode)
        unpacker.handle_unpacr_nop(info, 0, info["instr_args"])
    assert unpacker.blocked_on() == (0, "UNPACR_NOP", "SrcB", 0)
    return unpacker


def test_reports_a_really_wedged_unpacker(capsys):
    """End to end on the real unit, not the stand-in above."""
    detector = _detector_with_unit_stall_check()
    unpacker = _wedged_unpacker()
    _watch(detector, unpacker)

    for cycle in range(2 * UNIT_STALL_THRESHOLD):
        unpacker.clock_tick(cycle)  # re-runs the latched instruction, as the tile does
        detector.tick(cycle)

    err = capsys.readouterr().err
    assert "[UNIT STALL" in err, err
    assert "UNPACR_NOP from thread 0" in err
    assert "waiting for SrcB bank 0" in err


def test_unit_stall_latency_does_not_depend_on_the_deadlock_window(capsys):
    """The two thresholds are independent knobs, and must behave like it.

    The arming pass used to ride the *global* watchdog's cadence, so end-to-end
    latency was ``unit_stall_threshold + deadlock_threshold / 8`` and a user who
    lowered ``TT_SIM_UNIT_STALL_THRESHOLD`` to catch a short wedge could not
    make the report arrive any sooner — the fix that costs nothing while
    nothing is blocked, because arming only walks units.
    """
    detector = DeadlockDetector(
        1_000_000,  # a global window far larger than the unit-stall one
        True,
        [],
        [],
        unit_stall_threshold=UNIT_STALL_THRESHOLD,
    )
    _watch(detector, _FakeBlockedUnit())

    _pump(detector, 3 * UNIT_STALL_THRESHOLD)

    reported = re.findall(r"\[UNIT STALL cycle=(\d+)\]", capsys.readouterr().err)
    assert reported, "no report at all — the arming pass is still coupled"
    first = int(reported[0])
    assert first <= UNIT_STALL_THRESHOLD + detector.unit_stall_arm_interval + 1


class _FakeResetTile:
    """A tile whose baby cores are all in soft reset, or all out of it."""

    def __init__(self, in_reset):
        self.in_reset = in_reset

    def set(self, in_reset):
        self.in_reset = in_reset


def _detector_with_a_fake_tile(in_reset=False, threshold=UNIT_STALL_THRESHOLD):
    detector = _detector_with_unit_stall_check(threshold)
    state = _FakeResetTile(in_reset)
    detector._tiles_by_coord[(1, 1)] = (state, [])
    detector._tile_fully_in_reset = lambda coord: state.in_reset
    return detector, state


def test_reports_a_unit_left_blocked_when_the_whole_tile_goes_into_reset(capsys):
    """The terminal condition, which needs no threshold to be trustworthy.

    Handing a Src bank back takes an instruction and no thread can issue one
    from reset, so a unit still blocked here can never be satisfied. That is a
    proof rather than a heuristic, it fires immediately — which is what makes
    a *short* wedge reproduction (one that ends long before any cycle
    threshold could elapse) visible at all — and it **raises**, so a run with a
    wedged unit behind it cannot report success (ROADMAP item 1).
    """
    detector, state = _detector_with_a_fake_tile(in_reset=False)
    _watch(detector, _FakeBlockedUnit())

    half = UNIT_STALL_THRESHOLD // 4  # far short of the cycle threshold
    for cycle in range(half):
        detector.tick(cycle)
    assert "[UNIT WEDGED" not in capsys.readouterr().err
    state.set(True)
    with pytest.raises(UnitWedgedError, match="Unpacker 1.*wedged"):
        for cycle in range(half, 2 * half):
            detector.tick(cycle)

    err = capsys.readouterr().err
    assert "[UNIT WEDGED" in err, err
    assert "every baby core on the tile in soft reset" in err
    assert "UNPACR_NOP from thread 0" in err
    assert "waiting for SrcB bank 0" in err


def test_quiet_when_a_unit_clears_before_the_tile_settles_into_reset(capsys):
    """The one way a *correct* workload could reach the terminal check.

    A backend instruction already in flight when the reset lands can still hand
    the bank back. The grace period covers that; a unit that clears inside it
    must not be called wedged.
    """
    detector, state = _detector_with_a_fake_tile(in_reset=False)
    unit = _FakeBlockedUnit()
    _watch(detector, unit)

    quarter = UNIT_STALL_THRESHOLD // 4
    for cycle in range(quarter):
        detector.tick(cycle)
    state.set(True)
    for cycle in range(quarter, quarter + 8):  # well inside the grace period
        detector.tick(cycle)
    unit.blocked = False
    for cycle in range(quarter + 8, 2 * quarter):
        detector.tick(cycle)

    assert "[UNIT WEDGED" not in capsys.readouterr().err


@DEVICES
def test_a_really_wedged_unpacker_on_a_reset_tile_is_reported_at_once(
    capsys, device_class
):
    """The whole path, on a real device: real unpacker, real soft-reset register.

    A launch ends with every baby core back in soft reset, and the unpacker here
    is in the acquire-without-release state the Blackhole ``UNPACR_NOP`` bug
    leaves behind. Nothing can hand that Src bank back any more, so this is
    reported inside the grace period — **far** short of the cycle threshold,
    which is the point: a minimal reproduction of the bug finishes long before
    any threshold could elapse, and that is exactly when someone is looking.
    """
    device, coord, detector = _device_with_watchdog(device_class)
    detector.unit_stall_enabled = True
    detector.unit_stall_threshold = UNIT_STALL_THRESHOLD
    detector.unit_stall_arm_interval = UNIT_STALL_THRESHOLD // 8
    _watch(detector, _wedged_unpacker(), coord=coord)

    with pytest.raises(UnitWedgedError, match="wedged"):
        device.run(4 * _WEDGE_CONFIRM_CYCLES)  # << UNIT_STALL_THRESHOLD

    err = capsys.readouterr().err
    assert "[UNIT WEDGED" in err, err
    assert "every baby core on the tile in soft reset" in err
    assert "waiting for SrcB bank 0" in err
    assert "[UNIT STALL" not in err  # the cycle threshold is nowhere near


def test_wedge_report_is_disabled_with_the_unit_stall_check(capsys):
    detector = DeadlockDetector(
        THRESHOLD,
        True,
        [],
        [],
        unit_stall_enabled=False,
        unit_stall_threshold=UNIT_STALL_THRESHOLD,
    )
    detector._tile_fully_in_reset = lambda coord: True
    _watch(detector, _FakeBlockedUnit())

    _pump(detector, 20 * UNIT_STALL_THRESHOLD)

    assert "[UNIT WEDGED" not in capsys.readouterr().err


def test_unit_stall_config_from_env():
    enabled, threshold = unit_stall_config_from_env({})
    assert enabled
    assert threshold == DEFAULT_UNIT_STALL_THRESHOLD
    enabled, threshold = unit_stall_config_from_env(
        {"TT_SIM_UNIT_STALL": "0", "TT_SIM_UNIT_STALL_THRESHOLD": "1234"}
    )
    assert not enabled
    assert threshold == 1234
    # Independent of the no-progress window: shrinking that one to surface
    # stalls sooner must not drag this one through the measured legitimate
    # maximum (3,528 cycles) and start warning about correct kernels.
    _enabled, threshold = unit_stall_config_from_env(
        {"TT_SIM_DEADLOCK_THRESHOLD": "100"}
    )
    assert threshold == DEFAULT_UNIT_STALL_THRESHOLD


def test_sample_interval_follows_the_threshold():
    enabled, threshold = deadlock_config_from_env({"TT_SIM_DEADLOCK_THRESHOLD": "80"})
    assert enabled
    assert threshold == 80
    assert DeadlockDetector(threshold, True, [], []).sample_interval == 10
    # A threshold small enough to want per-cycle resolution still gets it.
    assert DeadlockDetector(4, True, [], []).sample_interval == 1
