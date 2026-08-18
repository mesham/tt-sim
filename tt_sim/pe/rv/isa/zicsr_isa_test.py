"""Unit tests for Zicsr and the Blackhole baby-RISC-V CSR file.

Every expected value here comes from
``BlackholeA0/TensixTile/BabyRISCV/CSRs.md``; where that doc and the RISC-V
spec disagree (the writable ``cycle`` / ``instret`` shadows, ``mcountinhibit``
being unable to inhibit anything that counts, ``vstart`` tied to zero) the tests
assert **the doc**, because the doc describes the hardware tt-sim is modelling.

Both directions throughout: that a correct access reads/writes what it should,
and that an access tt-sim cannot answer truthfully is refused rather than
answered with a plausible zero.
"""

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.arch.wormhole import WORMHOLE_PROFILE
from tt_sim.memory.memory import DRAM, VisibleMemory
from tt_sim.memory.memory_map import AddressRange, MemoryMap
from tt_sim.misc.tile_ctrl import TensixTileControl
from tt_sim.pe.register.register import Register, RegisterAccessMode
from tt_sim.pe.register.register_file import RegisterFile
from tt_sim.pe.rv.babyriscv import BabyRISCV, BabyRISCVCoreType
from tt_sim.pe.rv.isa.i_isa import RV_I_ISA
from tt_sim.pe.rv.isa.zicsr_isa import (
    RV_ZICSR_ISA,
    CSRFile,
    NoCSRsError,
    UnknownCSRError,
    UnmodelledCSRError,
)
from tt_sim.pe.rv.rv32 import FCSR_INDEX, FP_REGISTER_BASE, REGISTER_NAME_MAPPING
from tt_sim.util.conversion import conv_to_bytes

M = 0xFFFFFFFF

# Addresses, spelled out here rather than imported so a typo in the module's
# own table cannot make these tests agree with it.
FCSR = 0x003
VSTART = 0x008
MSTATUS = 0x300
MISA = 0x301
MCOUNTINHIBIT = 0x320
MHPMEVENT3 = 0x323
CFG0 = 0x7C0
CFG1 = 0x7C3
MCYCLE = 0xB00
MINSTRET = 0xB02
MHPMCOUNTER3 = 0xB03
MCYCLEH = 0xB80
MINSTRETH = 0xB82
QSTATUS = 0xBC0
BSTATUS = 0xBC1
SSTATUS0 = 0xBC2
INTP_RESTORE_PC = 0xBCA
CYCLE = 0xC00
INSTRET = 0xC02
VLENB = 0xC22
CYCLEH = 0xC80
INSTRETH = 0xC82
MHARTID = 0xF14
MEPC = 0x341  # exists in the RISC-V spec, absent from the Blackhole table

CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI = 1, 2, 3, 5, 6, 7


def _csr_instr(funct3, rd, rs1, addr):
    return (addr << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | 0x73


class _Clock:
    """Stand-in for the tile's ``TileClock`` — the one thing the CSR file needs
    from the device is a cycle number it did not make up itself."""

    def __init__(self, cycle=0):
        self.current_cycle = cycle


def _register_file(fp=True):
    registers = [Register(4, conv_to_bytes(0), RegisterAccessMode.R, False)]
    registers += [Register(4) for _ in range(33)]
    if fp:
        registers += [Register(4) for _ in range(33)]  # 32 f-regs + fcsr
    return RegisterFile(registers, REGISTER_NAME_MAPPING)


def _csrs(cycle=0, scratch_sstatus=True, fp=True, clock=True):
    rf = _register_file(fp=fp)
    rf.csrs = CSRFile(
        rf,
        core_label="BRISC",
        scratch_sstatus=scratch_sstatus,
        clock_owner=_Clock(cycle) if clock else None,
    )
    return rf, rf.csrs


def _exec(rf, instr, snoop=False):
    """Run one instruction the way ``RV32I.clock_tick`` would: base I first."""
    if RV_I_ISA.run(rf, None, snoop, instr):
        return "i_isa"
    if RV_ZICSR_ISA.run(rf, None, snoop, instr):
        return "zicsr"
    return None


# -- decode: the silent no-op this replaces ---------------------------------


def test_a_csr_read_is_no_longer_a_silent_no_op():
    """The defect: ``RV_I_ISA.handle_i_misc`` claimed the whole SYSTEM opcode,
    so ``csrr t0, mcycle`` executed nothing and left t0 holding its old value."""
    rf, _ = _csrs(cycle=1234)
    rf[5].write(conv_to_bytes(0xDEADBEEF))  # t0, deliberately not the answer
    # csrr t0, mcycle == csrrs t0, mcycle, x0
    assert _exec(rf, _csr_instr(CSRRS, 5, 0, MCYCLE)) == "zicsr"
    assert rf[5].read_uint() == 1234


@pytest.mark.parametrize("funct3", [CSRRW, CSRRS, CSRRC, CSRRWI, CSRRSI, CSRRCI])
def test_base_i_declines_every_csr_funct3(funct3):
    rf, _ = _csrs()
    assert RV_I_ISA.run(rf, None, False, _csr_instr(funct3, 5, 1, CFG0)) is False


def test_ecall_still_belongs_to_base_i():
    rf, _ = _csrs()
    instr = 0x73  # ecall (funct3 0; ebreak is the same funct3 and traps)
    assert RV_I_ISA.run(rf, None, False, instr) is True
    assert RV_ZICSR_ISA.run(rf, None, False, instr) is False


def test_funct3_four_is_not_a_zicsr_encoding():
    """SYSTEM funct3 = 4 is not a CSR instruction; it must reach tt-sim's
    unknown-instruction path (the doc's UndefinedBehavior), not be executed."""
    rf, _ = _csrs()
    instr = _csr_instr(0x4, 5, 1, CFG0)
    assert RV_I_ISA.run(rf, None, False, instr) is False
    assert RV_ZICSR_ISA.run(rf, None, False, instr) is False


# -- attachment: Blackhole only, and per core -------------------------------


def _baby_core(core_type, isa_extensions):
    ram = DRAM(0x1000)
    memory_map = MemoryMap()
    memory_map[AddressRange(0, ram.getSize())] = ram
    memory_map[AddressRange(0xFFB12000, 0xFFF)] = TensixTileControl()
    core = BabyRISCV(
        core_type, [VisibleMemory(memory_map)], isa_extensions=isa_extensions
    )
    return core, ram


def test_wormhole_baby_cores_have_no_csrs_and_refuse_csr_instructions():
    assert "zicsr" not in WORMHOLE_PROFILE.baby_core_isa_extensions
    core, _ = _baby_core(
        BabyRISCVCoreType.BRISC, WORMHOLE_PROFILE.baby_core_isa_extensions
    )
    assert core.csrs is None
    assert core.register_file.csrs is None
    with pytest.raises(NoCSRsError):
        RV_I_ISA.run(core.register_file, None, False, _csr_instr(CSRRS, 5, 0, MCYCLE))


def test_blackhole_baby_cores_get_a_csr_file():
    assert "zicsr" in BLACKHOLE_PROFILE.baby_core_isa_extensions
    core, _ = _baby_core(
        BabyRISCVCoreType.BRISC, BLACKHOLE_PROFILE.baby_core_isa_extensions
    )
    assert core.csrs is not None
    # One object, two references: the ISA reaches it through the register file.
    assert core.register_file.csrs is core.csrs


def test_sstatus_is_scratch_on_b_and_nc_only():
    for core_type in (BabyRISCVCoreType.BRISC, BabyRISCVCoreType.NCRISC):
        core, _ = _baby_core(core_type, BLACKHOLE_PROFILE.baby_core_isa_extensions)
        core.csrs.write(SSTATUS0, 0x1234)
        assert core.csrs.read(SSTATUS0) == 0x1234
    for core_type in (
        BabyRISCVCoreType.TRISC0,
        BabyRISCVCoreType.TRISC2,
        BabyRISCVCoreType.ERISC,
    ):
        core, _ = _baby_core(core_type, BLACKHOLE_PROFILE.baby_core_isa_extensions)
        with pytest.raises(UnmodelledCSRError):
            core.csrs.read(SSTATUS0)
        with pytest.raises(UnmodelledCSRError):
            core.csrs.write(SSTATUS0, 1)


def test_a_baby_core_in_a_device_has_its_csr_clock_bound():
    """``mcycle`` is only real because the device hands each core the tile clock
    that ``RISCV_DEBUG_REG_WALL_CLOCK_*`` reads — a private counter here could
    disagree with every other cycle number in the simulator."""
    from tt_sim.device.blackhole import Blackhole

    device = Blackhole()
    tile = device.tensix_tiles[0]
    for core in tile.get_baby_cores():
        assert core.csrs is not None
        assert core.csrs.clock_owner is tile.clock
    device.run(20)
    # Same clock the tile-control wall clock samples, so the two agree to the
    # cycle (tile_ctrl deliberately reports the previous cycle; see its docs).
    assert tile.brisc.csrs.read(MCYCLE) == tile.clock.current_cycle
    assert tile.brisc.csrs.read(MCYCLE) == tile.tile_ctrl.cycle_num + 1


# -- the counters -----------------------------------------------------------


def test_mcycle_follows_the_clock_it_was_given():
    rf, csrs = _csrs(cycle=10)
    assert csrs.read(MCYCLE) == 10
    csrs.clock_owner.current_cycle = 4321
    assert csrs.read(MCYCLE) == 4321
    assert csrs.read(MCYCLEH) == 0


def test_mcycle_without_a_clock_refuses_rather_than_inventing_one():
    _, csrs = _csrs(clock=False)
    with pytest.raises(UnmodelledCSRError, match="no clock bound"):
        csrs.read(MCYCLE)


def test_writing_mcycle_rebases_it_and_it_keeps_counting():
    _, csrs = _csrs(cycle=1000)
    csrs.write(MCYCLE, 7)
    assert csrs.read(MCYCLE) == 7
    csrs.clock_owner.current_cycle = 1005
    assert csrs.read(MCYCLE) == 12  # still counting, from the written value


def test_the_high_half_carries_and_can_be_written_alone():
    _, csrs = _csrs(cycle=0)
    csrs.write(MCYCLEH, 3)
    csrs.clock_owner.current_cycle = 9
    assert csrs.read(MCYCLEH) == 3
    assert csrs.read(MCYCLE) == 9
    # A full 64-bit rebase: low half just below the wrap, then one more cycle.
    csrs.write(MCYCLE, M)
    csrs.write(MCYCLEH, 0)
    csrs.clock_owner.current_cycle = 10
    assert csrs.read(MCYCLE) == 0
    assert csrs.read(MCYCLEH) == 1


@pytest.mark.parametrize(
    "shadow,machine",
    [(CYCLE, MCYCLE), (CYCLEH, MCYCLEH), (INSTRET, MINSTRET), (INSTRETH, MINSTRETH)],
)
def test_zicntr_shadows_are_read_write_aliases(shadow, machine):
    """Non-conformant on purpose: the spec makes these read-only aliases, the
    doc says they are writable. One piece of state, two names."""
    _, csrs = _csrs(cycle=0)
    csrs.write(shadow, 0x40)
    assert csrs.read(machine) == 0x40
    csrs.write(machine, 0x99)
    assert csrs.read(shadow) == 0x99


def test_minstret_counts_retired_instructions():
    core, ram = _baby_core(
        BabyRISCVCoreType.BRISC, BLACKHOLE_PROFILE.baby_core_isa_extensions
    )
    core.bind_clock(_Clock(0))
    program = [
        0x00100093,  # addi x1, x0, 1
        0x00200113,  # addi x2, x0, 2
        0x002081B3,  # add  x3, x1, x2
        0x0000007F,  # a reserved opcode: claimed by nothing, retires nothing
        0x0000006F,  # jal x0, 0 (spin in place, one retire per tick)
    ]
    for i, word in enumerate(program):
        ram.write(i * 4, conv_to_bytes(word))
    core.reset()
    for cycle in range(4):
        core.clock_tick(cycle)
    assert core.unknown_instructions == 1
    assert core.csrs.read(MINSTRET) == 3, "an unknown instruction retired nothing"
    core.clock_tick(4)
    assert core.csrs.read(MINSTRET) == 4


def test_minstret_is_writable_and_keeps_counting():
    core, ram = _baby_core(
        BabyRISCVCoreType.BRISC, BLACKHOLE_PROFILE.baby_core_isa_extensions
    )
    core.bind_clock(_Clock(0))
    ram.write(0, conv_to_bytes(0x0000006F))  # jal x0, 0
    core.reset()
    core.clock_tick(0)  # first tick takes the core out of soft reset
    core.csrs.write(MINSTRET, 500)
    for cycle in range(1, 4):
        core.clock_tick(cycle)
    assert core.csrs.read(MINSTRET) == 503


def test_mcountinhibit_cannot_inhibit_mcycle_or_minstret():
    """A documented non-conformance: writable, but powerless over the two
    counters anyone would want to stop."""
    core, ram = _baby_core(
        BabyRISCVCoreType.BRISC, BLACKHOLE_PROFILE.baby_core_isa_extensions
    )
    clock = _Clock(0)
    core.bind_clock(clock)
    ram.write(0, conv_to_bytes(0x0000006F))  # jal x0, 0
    core.reset()
    core.clock_tick(0)  # first tick takes the core out of soft reset
    core.csrs.write(MCOUNTINHIBIT, M)
    assert core.csrs.read(MCOUNTINHIBIT) == M  # the write did land
    for cycle in range(1, 4):
        clock.current_cycle = cycle
        core.clock_tick(cycle)
    assert core.csrs.read(MINSTRET) == 4
    assert core.csrs.read(MCYCLE) == 3


# -- side-effect suppression ------------------------------------------------


def test_csrrw_with_rd_x0_does_not_read_the_csr():
    """The spec's rule, and here it is load-bearing: ``intp_restore_pc`` refuses
    reads until something has written it, so a spurious read would raise."""
    rf, csrs = _csrs()
    rf[6].write(conv_to_bytes(0x2000))
    _exec(rf, _csr_instr(CSRRW, 0, 6, INTP_RESTORE_PC))  # csrw, no read
    assert csrs.read(INTP_RESTORE_PC) == 0x2000

    rf2, _ = _csrs()
    rf2[6].write(conv_to_bytes(0x2000))
    with pytest.raises(UnmodelledCSRError):
        # Same instruction with a real rd: now it must read first, and cannot.
        _exec(rf2, _csr_instr(CSRRW, 7, 6, INTP_RESTORE_PC))


class _CountingCSRFile(CSRFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writes = 0

    def write(self, addr, value, pc=None):
        self.writes += 1
        return super().write(addr, value, pc)


def _counting(scratch_sstatus=True):
    rf = _register_file()
    rf.csrs = _CountingCSRFile(rf, "BRISC", scratch_sstatus, _Clock(0))
    return rf, rf.csrs


def test_csrrs_with_rs1_x0_performs_no_write():
    rf, csrs = _counting()
    _exec(rf, _csr_instr(CSRRS, 5, 0, CFG0))  # csrr t0, cfg0
    assert csrs.writes == 0
    assert rf[5].read_uint() == 16 << 13


def test_csrrsi_with_a_zero_immediate_performs_no_write():
    rf, csrs = _counting()
    _exec(rf, _csr_instr(CSRRSI, 5, 0, CFG0))
    assert csrs.writes == 0


def test_a_register_holding_zero_is_not_the_same_as_x0():
    """The suppression rule keys off the *register specifier*, not its value —
    ``csrrs rd, csr, t1`` with t1 == 0 still writes."""
    rf, csrs = _counting()
    rf[6].write(conv_to_bytes(0))
    _exec(rf, _csr_instr(CSRRS, 5, 6, CFG0))
    assert csrs.writes == 1


def test_csrrs_and_csrrc_set_and_clear_bits():
    rf, csrs = _csrs()
    rf[6].write(conv_to_bytes(0b1010))
    _exec(rf, _csr_instr(CSRRS, 5, 6, CFG1))
    assert csrs.read(CFG1) == 0b1010
    assert rf[5].read_uint() == 0, "rd gets the *old* value"
    rf[6].write(conv_to_bytes(0b0010))
    _exec(rf, _csr_instr(CSRRC, 5, 6, CFG1))
    assert csrs.read(CFG1) == 0b1000
    assert rf[5].read_uint() == 0b1010


def test_immediate_forms_use_the_rs1_field_as_a_five_bit_value():
    rf, csrs = _csrs()
    _exec(rf, _csr_instr(CSRRWI, 0, 0x1F, CFG1))
    assert csrs.read(CFG1) == 0x1F
    _exec(rf, _csr_instr(CSRRCI, 0, 0x0F, CFG1))
    assert csrs.read(CFG1) == 0x10


def test_writes_to_x0_are_dropped():
    rf, _ = _csrs(cycle=77)
    _exec(rf, _csr_instr(CSRRS, 0, 0, MCYCLE))
    assert rf[0].read_uint() == 0


# -- reset values and read-only registers -----------------------------------


def test_cfg0_resets_to_st_merge_timer_sixteen_and_nothing_else():
    _, csrs = _csrs()
    value = csrs.read(CFG0)
    assert (value >> 13) & 0x1F == 16
    assert value & ~(0x1F << 13) == 0


def test_vlenb_is_initially_sixteen():
    _, csrs = _csrs()
    assert csrs.read(VLENB) == 16


@pytest.mark.parametrize(
    "addr,value",
    [(MSTATUS, 0x80006600), (MISA, 0x40201123), (MHARTID, 0), (VSTART, 0)],
)
def test_read_only_csrs_read_their_documented_constant_and_ignore_writes(addr, value):
    rf, csrs = _csrs()
    assert csrs.read(addr) == value
    rf[6].write(conv_to_bytes(0xFFFFFFFF))
    _exec(rf, _csr_instr(CSRRW, 5, 6, addr))
    assert rf[5].read_uint() == value  # rd still got the old (only) value
    assert csrs.read(addr) == value  # and the write went nowhere


def test_a_core_reset_restores_the_csr_reset_values():
    core, ram = _baby_core(
        BabyRISCVCoreType.BRISC, BLACKHOLE_PROFILE.baby_core_isa_extensions
    )
    core.bind_clock(_Clock(0))
    ram.write(0, conv_to_bytes(0x0000006F))
    core.reset()
    core.clock_tick(0)  # first tick takes the core out of soft reset
    core.csrs.write(CFG0, 0)
    assert core.csrs.read(CFG0) == 0
    assert core.csrs.read(MINSTRET) == 1
    core.reset()
    assert core.csrs.read(CFG0) == 16 << 13
    assert core.csrs.read(MINSTRET) == 0


# -- fcsr aliasing ----------------------------------------------------------


def test_fcsr_aliases_the_fp_register_file_entry():
    rf, csrs = _csrs()
    csrs.write(FCSR, 0x55)
    assert rf[FCSR_INDEX].read_uint() == 0x55
    assert rf["fcsr"].read_uint() == 0x55
    rf[FP_REGISTER_BASE + 32].write(conv_to_bytes(0xAA))
    assert csrs.read(FCSR) == 0xAA


def test_fcsr_on_a_core_with_no_fp_register_file_refuses():
    _, csrs = _csrs(fp=False)
    with pytest.raises(UnmodelledCSRError, match="floating-point register file"):
        csrs.read(FCSR)


# -- refusals ---------------------------------------------------------------


@pytest.mark.parametrize("addr", [QSTATUS, BSTATUS])
def test_tensix_status_csrs_are_refused_rather_than_answered_with_zero(addr):
    rf, csrs = _csrs()
    with pytest.raises(UnmodelledCSRError):
        csrs.read(addr)
    with pytest.raises(UnmodelledCSRError):
        csrs.write(addr, 0)
    with pytest.raises(UnmodelledCSRError):
        _exec(rf, _csr_instr(CSRRS, 5, 0, addr))


def test_an_address_the_hardware_does_not_recognise_raises():
    rf, csrs = _csrs()
    with pytest.raises(UnknownCSRError):
        csrs.read(MEPC)
    with pytest.raises(UnknownCSRError):
        csrs.write(MEPC, 1)
    with pytest.raises(UnknownCSRError):
        _exec(rf, _csr_instr(CSRRW, 0, 6, MEPC))


def test_hpm_counters_read_zero_until_an_event_is_selected():
    _, csrs = _csrs()
    assert csrs.read(MHPMCOUNTER3) == 0, "no event selected: 0 is the true count"
    csrs.write(MHPMEVENT3, 1)
    with pytest.raises(UnmodelledCSRError, match="unpublished"):
        csrs.read(MHPMCOUNTER3)


def test_intp_restore_pc_refuses_until_software_writes_it():
    _, csrs = _csrs()
    with pytest.raises(UnmodelledCSRError, match="no interrupts"):
        csrs.read(INTP_RESTORE_PC)
    csrs.write(INTP_RESTORE_PC, 0x1234)
    assert csrs.read(INTP_RESTORE_PC) == 0x1234


# -- disassembly ------------------------------------------------------------


def test_snoop_names_the_csr(capsys):
    rf, _ = _csrs(cycle=5)
    _exec(rf, _csr_instr(CSRRS, 5, 0, MCYCLE), snoop=True)
    out = capsys.readouterr().out
    assert "csrrs t0, mcycle, zero" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
