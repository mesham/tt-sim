"""Firmware-loop recognition (``tt_sim/pe/rv/spin.py``).

Runs standalone (``python3 -m tt_sim.pe.rv.spin_test``) or under pytest.

The claims, in the order the module's risk analysis makes them:

1. A pure poll loop (loads from plain RAM only, no stores, register fixed
   point) is detected and parks the core; ``next_wake_cycle`` answers ``None``.
2. Skipping cycles while parked and restoring at wake is **cycle-exact**: a
   run with arbitrary skip spans is bit-identical, register for register and
   cycle for cycle, to a run ticked every cycle — including the cycle at which
   the loop exits once the polled flag is written.
3. A write to any watched byte flips ``next_wake_cycle`` to "next cycle" and
   unparks, so a stride can never hide the go signal.
4. Loops that are not provably pure never park: a loop containing a store, and
   a loop polling MMIO (whose value can change without any write).
5. The kill switch (``TT_SIM_FIRMWARE_IDLE=0``) disables everything.
6. Under ``TT_SIM_COST_MODEL`` the same claims hold, and the fixed point now
   covers the cost scoreboard: a periodic poll loop still parks on both arches,
   a *stall* is never mistaken for one, a skipped span leaves the model's own
   charges exactly where an unparked run would, and the signature/restore pair
   round-trips. Each of these fails if either half of the mechanism (the
   fresh-arrival gate, the scoreboard fixed point) is removed.
7. End to end through a real device, model off *and* on: a Blackhole worker
   whose BRISC spins in a poll loop goes dormant (the full-grid affordability
   property), wakes on the host write, and the run is bit-identical to one with
   the feature off; and a core alone on a tile in a mostly-stalled loop does
   exactly the same work either way.
"""

import os
from contextlib import contextmanager

from tt_sim.memory.memory import DRAM, VisibleMemory
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.misc.tile_ctrl import TensixTileControl
from tt_sim.pe.rv.babyriscv import BabyRISCV, BabyRISCVCoreType
from tt_sim.pe.rv.spin import SPIN_PARKED
from tt_sim.util.conversion import conv_to_bytes, conv_to_uint32

# -- instruction encoders ---------------------------------------------------


def _lw(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (0x2 << 12) | (rd << 7) | 0x03


def _sw(rs2, rs1, imm):
    imm &= 0xFFF
    return (
        ((imm >> 5) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (0x2 << 12)
        | ((imm & 0x1F) << 7)
        | 0x23
    )


def _addi(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (0x0 << 12) | (rd << 7) | 0x13


def _branch(funct3, rs1, rs2, offset):
    imm = offset & 0x1FFF
    return (
        (((imm >> 12) & 0x1) << 31)
        | (((imm >> 5) & 0x3F) << 25)
        | (rs2 << 20)
        | (rs1 << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xF) << 8)
        | (((imm >> 11) & 0x1) << 7)
        | 0x63
    )


def _beq(rs1, rs2, offset):
    return _branch(0x0, rs1, rs2, offset)


def _jal(rd, offset):
    imm = offset & 0x1FFFFF
    return (
        (((imm >> 20) & 0x1) << 31)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 11) & 0x1) << 20)
        | (((imm >> 12) & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
    )


A4, A5 = 14, 15
FLAG = 0x100
MARKER = 0x104
SCRATCH = 0x108

#: The go-poll shape observed on real firmware: load a flag from L1, loop
#: while it is zero, then do post-loop work.
POLL_PROGRAM = [
    _lw(A5, 0, FLAG),  # 0x00: a5 = mem[FLAG]
    _beq(A5, 0, -4),  # 0x04: while a5 == 0 goto 0x00
    _sw(A5, 0, MARKER),  # 0x08: mem[MARKER] = a5
    _addi(A4, A4, 1),  # 0x0c: post-loop activity
    _jal(0, 0),  # 0x10: j .
]

#: A loop that stores every iteration — must never park.
STORE_LOOP_PROGRAM = [
    _lw(A5, 0, FLAG),  # 0x00
    _sw(A5, 0, SCRATCH),  # 0x04: a store, so the loop is not pure
    _beq(A5, 0, -8),  # 0x08: while a5 == 0 goto 0x00
    _jal(0, 0),  # 0x0c
]

# TensixTileControl is mapped at its usual tile base; loading from it is an
# MMIO load (the wall clock lives there) and must never park.
TILE_CTRL_BASE = 0xFFB12000
MMIO_LOOP_PROGRAM = [
    _lw(A5, A4, 0x1F0),  # 0x00: a5 = wall clock (a4 holds the base)
    _beq(0, 0, -4),  # 0x04: unconditional loop
]


@contextmanager
def _env(**overrides):
    saved = {}
    for key, value in overrides.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, prev in saved.items():
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


def _make_core(program, arch=None, isa_extensions=()):
    ram = DRAM(0x1000)
    tile_ctrl = TensixTileControl()  # soft-reset register reads 0: running
    memory_map = MemoryMap()
    memory_map[AddressRange(0, ram.getSize())] = ram
    memory_map[AddressRange(TILE_CTRL_BASE, tile_ctrl.getSize())] = tile_ctrl
    visible = VisibleMemory(memory_map)
    for i, word in enumerate(program):
        ram.write(i * 4, conv_to_bytes(word))
    core = BabyRISCV(
        BabyRISCVCoreType.BRISC, [visible], arch=arch, isa_extensions=isa_extensions
    )
    core.reset()
    return core, ram


def _tick_span(core, start, count):
    for cycle in range(start, start + count):
        core.clock_tick(cycle)
    return start + count


def test_pure_poll_loop_parks_and_reports_dormant():
    core, _ram = _make_core(POLL_PROGRAM)
    cycle = _tick_span(core, 0, 400)
    assert core.spin_parked, "pure poll loop was not parked within 400 cycles"
    assert core._spin.state == SPIN_PARKED
    assert core._spin.parks == 1
    # Dormant while the watched memory holds still.
    assert core.next_wake_cycle(cycle - 1) is None


def test_parked_skips_are_cycle_exact_including_the_loop_exit():
    flag_at = 1500
    end = 1520

    def run(core, ram, skip_while_parked):
        cycle = 0
        while cycle < end:
            if cycle == flag_at:
                ram.write(FLAG, conv_to_bytes(7))
            if (
                skip_while_parked
                and core.spin_parked
                and cycle < flag_at - 1
                and core.next_wake_cycle(cycle - 1) is None
            ):
                # Skip a span, exactly as the pump would after the tile's
                # probe answered None: memory cannot change during the span
                # (nothing ticks; a write would have woken the tile first).
                cycle = min(cycle + 37, flag_at - 1, end - 1)
                continue
            core.clock_tick(cycle)
            cycle += 1

    with _env(TT_SIM_FIRMWARE_IDLE="0"):
        control, control_ram = _make_core(POLL_PROGRAM)
    assert control._spin is None
    run(control, control_ram, skip_while_parked=False)

    subject, subject_ram = _make_core(POLL_PROGRAM)
    run(subject, subject_ram, skip_while_parked=True)

    assert subject._spin.parks >= 1
    assert subject._spin.skipped_cycles > 0, "no cycles were actually skipped"
    # Bit-identical outcome: every register (GPRs, PC, next-PC) and the memory
    # the program wrote.
    control_regs = [r.value for r in control.register_file.registers]
    subject_regs = [r.value for r in subject.register_file.registers]
    assert subject_regs == control_regs
    assert subject_ram.read(MARKER, 4) == control_ram.read(MARKER, 4)
    assert subject_ram.read(MARKER, 4) == conv_to_bytes(7)


def test_minstret_survives_a_parked_span():
    """A parked span must not deflate the retire count either.

    ``mcycle`` keeps advancing across a skipped span (it is a function of the
    pump's cycle number), so an unrepaired ``minstret`` would make the skipped
    iterations look like cycles in which the core retired nothing — exactly the
    silent wrong the rest of this module exists to avoid. The skipped ticks are
    added back from the loop's proved periodicity, so the two runs agree.
    """
    flag_at = 1500
    end = 1520

    def run(core, ram, skip_while_parked):
        cycle = 0
        while cycle < end:
            if cycle == flag_at:
                ram.write(FLAG, conv_to_bytes(7))
            if (
                skip_while_parked
                and core.spin_parked
                and cycle < flag_at - 1
                and core.next_wake_cycle(cycle - 1) is None
            ):
                cycle = min(cycle + 37, flag_at - 1, end - 1)
                continue
            core.clock_tick(cycle)
            cycle += 1

    with _env(TT_SIM_FIRMWARE_IDLE="0"):
        control, control_ram = _make_core(POLL_PROGRAM, isa_extensions=("zicsr",))
    assert control._spin is None
    run(control, control_ram, skip_while_parked=False)

    subject, subject_ram = _make_core(POLL_PROGRAM, isa_extensions=("zicsr",))
    run(subject, subject_ram, skip_while_parked=True)

    assert subject._spin.skipped_cycles > 0, "no cycles were actually skipped"
    assert subject.csrs.retired == control.csrs.retired
    # And the count is a real one, not two matching zeroes.
    assert control.csrs.retired > 1000


def test_watched_write_unparks_and_refuses_dormancy():
    core, ram = _make_core(POLL_PROGRAM)
    cycle = _tick_span(core, 0, 400)
    assert core.spin_parked
    ram.write(FLAG, conv_to_bytes(1))
    # The probe the pump consults must now refuse dormancy and unpark.
    assert core.next_wake_cycle(cycle - 1) == cycle
    assert not core.spin_parked
    # The core then genuinely exits the loop.
    cycle = _tick_span(core, cycle, 10)
    assert ram.read(MARKER, 4) == conv_to_bytes(1)


def test_store_loop_never_parks():
    core, _ram = _make_core(STORE_LOOP_PROGRAM)
    _tick_span(core, 0, 2000)
    assert not core.spin_parked
    assert core._spin.parks == 0


def test_mmio_poll_loop_never_parks():
    core, _ram = _make_core(MMIO_LOOP_PROGRAM)
    core.register_file[A4].write(conv_to_bytes(TILE_CTRL_BASE))
    _tick_span(core, 0, 2000)
    assert not core.spin_parked
    assert core._spin.parks == 0


def test_kill_switch_disables_recognition():
    with _env(TT_SIM_FIRMWARE_IDLE="0"):
        core, _ram = _make_core(POLL_PROGRAM)
    assert core._spin is None
    cycle = _tick_span(core, 0, 500)
    assert not core.spin_parked
    assert core.next_wake_cycle(cycle - 1) == cycle


# -- the cost model ---------------------------------------------------------
#
# With TT_SIM_COST_MODEL on, "the registers came back to their anchor values"
# stops being sufficient: a stalled tick retires nothing, so it satisfies that
# test trivially. The fixed point therefore covers the cost scoreboard's
# cycle-relative normal form as well, and an unpark re-bases it onto the wake
# cycle. These are the cases that mechanism rests on.


def _stalling_program(blocks):
    """A loop of stores to distinct 16-byte L1 blocks. Blackhole sustains one
    store every five cycles unless they coalesce, so most ticks are stalls at a
    PC that never moves — the shape a register-only fixed point misreads."""
    program = [_sw(A5, 0, 0x200 + i * 0x40) for i in range(blocks)]
    program.append(_addi(A4, A4, 1))
    program.append(_jal(0, -4 * (len(program))))
    return program


def test_cost_model_poll_loop_parks_on_both_arches():
    # Wormhole's L1 load latency is >= 8 and Blackhole's L0 hit is 2, so the
    # same two-instruction poll is a 9-tick and a 3-tick loop respectively —
    # both periodic, both parkable.
    for arch, expected_len in (("wormhole", 9), ("blackhole", 3)):
        with _env(TT_SIM_COST_MODEL="1"):
            core, _ram = _make_core(POLL_PROGRAM, arch=arch)
        assert core.rv_cost is not None
        _tick_span(core, 0, 600)
        assert core.spin_parked, f"{arch}: poll loop did not park under the model"
        assert core._spin.loop_len == expected_len, arch


def test_cost_model_stall_is_never_mistaken_for_a_fixed_point():
    # Four ticks in five of this loop are store-rate stalls, and it stores, so
    # nothing here may ever park. Before the scoreboard joined the fixed point,
    # the recogniser parked *inside* the stalls, reporting "pure poll loop of
    # 1 ticks".
    with _env(TT_SIM_COST_MODEL="1"):
        core, _ram = _make_core(_stalling_program(8), arch="blackhole")
    assert core.rv_cost is not None
    _tick_span(core, 0, 4000)
    assert not core.spin_parked
    assert core._spin.parks == 0


def test_cost_model_parked_skips_are_cycle_exact_and_charge_the_same():
    """The load-bearing property, with the model on: a run that skips arbitrary
    spans while parked ends bit-identical to one ticked every cycle — registers,
    memory, *and* everything the cost model charged."""
    flag_at = 1500
    end = 1520

    def run(core, ram, skip_while_parked):
        cycle = 0
        while cycle < end:
            if cycle == flag_at:
                ram.write(FLAG, conv_to_bytes(7))
            if (
                skip_while_parked
                and core.spin_parked
                and cycle < flag_at - 1
                and core.next_wake_cycle(cycle - 1) is None
            ):
                cycle = min(cycle + 37, flag_at - 1, end - 1)
                continue
            core.clock_tick(cycle)
            cycle += 1

    with _env(TT_SIM_COST_MODEL="1", TT_SIM_FIRMWARE_IDLE="0"):
        control, control_ram = _make_core(POLL_PROGRAM, arch="blackhole")
    assert control._spin is None
    assert control.rv_cost is not None
    run(control, control_ram, skip_while_parked=False)

    with _env(TT_SIM_COST_MODEL="1"):
        subject, subject_ram = _make_core(POLL_PROGRAM, arch="blackhole")
    run(subject, subject_ram, skip_while_parked=True)

    assert subject._spin.parks >= 1
    assert subject._spin.skipped_cycles > 0, "no cycles were actually skipped"
    control_regs = [r.value for r in control.register_file.registers]
    subject_regs = [r.value for r in subject.register_file.registers]
    assert subject_regs == control_regs
    assert subject_ram.read(MARKER, 4) == conv_to_bytes(7)
    # The cost model's own accounting, not just the architectural state: the
    # skipped iterations are charged rather than lost.
    assert subject.rv_cost.summary() == control.rv_cost.summary()


def test_cost_model_scoreboard_signature_round_trips():
    """``spin_signature`` / ``spin_restore`` are the time translation. Re-basing
    a signature onto a later cycle must give a scoreboard that answers every
    future question the way the original would have."""
    with _env(TT_SIM_COST_MODEL="1"):
        core, _ram = _make_core(POLL_PROGRAM, arch="blackhole")
    cost = core.rv_cost
    _tick_span(core, 0, 40)
    at = 40
    signature = cost.spin_signature(at)
    later = at + 913
    cost.spin_restore(signature, later)
    assert cost.spin_signature(later) == signature


def test_soft_reset_unparks():
    core, _ram = _make_core(POLL_PROGRAM)
    _tick_span(core, 0, 400)
    assert core.spin_parked
    # Assert soft reset: the very next tick must drop the parked state.
    tile_ctrl_val = 1 << 11
    core.visible_memory.write(TILE_CTRL_BASE + 0x1B0, conv_to_bytes(tile_ctrl_val))
    core.clock_tick(400)
    assert not core.spin_parked
    assert not core.soft_active


# -- end to end through a real device ---------------------------------------


def _spin_device_run(firmware_idle, cost_model=False, program=None, cycles=1000):
    """Boot a Blackhole worker whose BRISC runs ``program``, park (or not),
    write the flag, and return (dormant_cycles, marker_bytes, brisc_regs, tile)."""
    from tt_sim.device.blackhole import Blackhole
    from tt_sim.device.tt_device import DeviceTileDiagnostics

    coord = (1, 2)
    program = POLL_PROGRAM if program is None else program
    # The gate runs this suite with TT_SIM_COST_MODEL set, so pin it either way
    # rather than inheriting whatever the caller happens to have exported.
    env = {
        "TT_SIM_FIRMWARE_IDLE": None if firmware_idle else "0",
        "TT_SIM_COST_MODEL": "1" if cost_model else None,
    }
    with _env(**env):
        device = Blackhole(DeviceTileDiagnostics())
        if coord not in device.tile_directory:
            device.add_tensix_tile(coord)
    tile = device.tile_directory[coord]
    for i, word in enumerate(program):
        device.write(coord, i * 4, conv_to_bytes(word))
    device.reset_tile(coord)
    device.deassert_soft_reset(coord, BabyRISCVCoreType.BRISC)
    device.run(cycles)
    dormant_while_spinning = tile.clock.dormant_cycles
    device.write(coord, FLAG, conv_to_bytes(9))
    device.run(50)
    marker = bytes(device.read(coord, MARKER, 4))
    regs = [r.value for r in tile.brisc.register_file.registers]
    device.shutdown()
    return dormant_while_spinning, marker, regs, tile


def test_device_worker_spinning_in_firmware_goes_dormant_and_wakes():
    dormant_on, marker_on, regs_on, tile_on = _spin_device_run(firmware_idle=True)
    assert tile_on.brisc.spin_parked is False  # unparked once the flag landed
    assert tile_on.brisc._spin.parks >= 1
    assert dormant_on > 0, "a parked spinning worker never let its tile sleep"
    assert marker_on == conv_to_bytes(9)

    dormant_off, marker_off, regs_off, _tile_off = _spin_device_run(firmware_idle=False)
    assert dormant_off == 0, "control run unexpectedly slept while spinning"
    assert marker_off == conv_to_bytes(9)
    # Bit-identical architectural outcome, feature on vs off.
    assert regs_on == regs_off


def test_device_worker_goes_dormant_under_the_cost_model():
    """The whole point of the item: the ~1000x reaches TT_SIM_COST_MODEL runs,
    and reaches them without moving anything."""
    dormant_on, marker_on, regs_on, tile_on = _spin_device_run(
        firmware_idle=True, cost_model=True
    )
    assert tile_on.brisc._spin.parks >= 1
    assert dormant_on > 0, "a parked spinning worker never let its tile sleep"
    assert marker_on == conv_to_bytes(9)

    dormant_off, marker_off, regs_off, tile_off = _spin_device_run(
        firmware_idle=False, cost_model=True
    )
    assert dormant_off == 0
    assert regs_on == regs_off
    assert marker_on == marker_off
    assert tile_on.brisc.rv_cost.summary() == tile_off.brisc.rv_cost.summary()


def test_device_stall_loop_does_the_same_work_parked_or_not():
    """A core alone on a tile, in a loop that is mostly L1-store-rate stalls:
    nothing else keeps the tile awake, so a core parked inside a stall would
    sleep through the rest of it. Measured before the scoreboard joined the
    fixed point: 41 iterations against 399."""
    program = _stalling_program(3)
    counts = []
    for firmware_idle in (False, True):
        _dormant, _marker, regs, tile = _spin_device_run(
            firmware_idle=firmware_idle,
            cost_model=True,
            program=program,
            cycles=4000,
        )
        counts.append(conv_to_uint32(regs[A4]))
        assert not tile.brisc.spin_parked
    assert counts[0] == counts[1], f"parking changed the work done: {counts}"
    assert counts[0] > 0


def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"{name} OK")
    print(f"{len(tests)} spin tests passed")


if __name__ == "__main__":
    main()
