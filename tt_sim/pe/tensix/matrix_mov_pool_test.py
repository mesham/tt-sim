"""Tests for the matrix-unit ops MOVB2D / MOVD2A / GMPOOL.

All three are emitted by real tt-metal Blackhole kernels (``copy_tile`` and the
transpose LLKs for MOVB2D, ``cmath_common``'s dest-to-SrcA move for MOVD2A, and
``reduce_tile``'s max path for GMPOOL) and all three carry a field Blackhole
moved, so the decode is arch-specific: MOVB2D's ``movb2d_instr_mod`` is raw
13:11 (Wormhole 14:12), MOVB2D/MOVD2A's ``addr_mode`` is 16:14 (Wormhole 16:15)
and GMPOOL's ``pool_addr_mode`` is 17:15 (Wormhole 16:15).

Every case is driven through a real backend (issue + clock_tick) so the decoder
and the handler are both exercised, and each Blackhole case is paired with the
same instruction word on a Wormhole unit so the gating is checked in both
directions. Semantics follow the tt-isa-documentation functional models, with
modes the vendor reference simulator (ttsim) declines rejected rather than
guessed.

Runs standalone (``python3 -m tt_sim.pe.tensix.matrix_mov_pool_test``) or under
pytest.
"""

from contextlib import contextmanager

import pytest

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.backends.backend_base import DataFormat
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixConfigurationConstants
from tt_sim.util.conversion import conv_to_uint32

MOVD2A = 0x08
MOVB2D = 0x13
GMPOOL = 0x33

# MOVB2D instruction modifiers (tt-llk ``p_movb2d``)
MOV_1_ROW = 0x0
MOV_1_ROW_D0_BRCST = 0x1
MOV_8_ROW_BRCST = 0x2
MOV_4_ROWS = 0x4

# MOVD2A instruction modifiers (tt-llk ``p_movd2a``)
MOVD2A_1_ROW = 0x0
MOVD2A_4_ROWS = 0x2

# GMPOOL ``instr_mod19`` (tt-llk ``p_gpool``)
DIM_16X16 = 0x1


@contextmanager
def _backend(blackhole):
    """A Tensix backend on the given architecture.

    The config-register layout is a process-global selection, so restore the
    Wormhole layout on the way out — other tests in the same session (and the
    Wormhole replay guards) expect it.
    """
    try:
        yield TensixCoProcessor(
            None,
            BLACKHOLE_PROFILE.tensix_cfg_state_size if blackhole else None,
            BLACKHOLE_PROFILE.tensix_thd_state_size if blackhole else None,
            blackhole=blackhole,
        ).getBackend()
    finally:
        TensixConfigurationConstants.use_blackhole(False)


def _issue(backend, instruction, thread=0):
    """Issue through the backend and run the matrix unit for one cycle."""
    assert backend.issueInstruction(instruction, thread)
    backend.matrix_unit.clock_tick(0)


def _set_config(backend, key, value, state_id=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    old = backend.config_unit.get_config_entry(state_id, addr32)
    backend.config_unit.setConfig(
        state_id, addr32, (old & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)
    )


def _set_thread_config(backend, key, value, thread=0):
    addr32 = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    cfg = backend.config_unit.threadConfig[thread]
    cfg[addr32] = (cfg[addr32] & ~mask & 0xFFFFFFFF) | ((value << shamt) & mask)


# --- datum helpers -----------------------------------------------------------
#
# Src holds a datum as Sign,Man(10b),Exp(8b); the 16-bit Dst BF16 form is
# Sign,Man(7b),Exp(8b). Both are built here from an FP32 bit pattern truncated
# to BF16, which is what every op below moves around.


def _bf16_fields(value):
    bits = conv_to_uint32(value)
    return (bits >> 31) & 1, (bits >> 23) & 0xFF, (bits >> 16) & 0x7F


def _src_bf16(value):
    sign, exp, man = _bf16_fields(value)
    return (sign << 18) | (man << 11) | exp


def _dst_bf16(value):
    sign, exp, man = _bf16_fields(value)
    return (sign << 15) | (man << 8) | exp


# --- instruction builders ----------------------------------------------------


def _movb2d(dst=0, instr_mod=MOV_1_ROW, addr_mode=0, src=0, dest_32b_lo=0, bh=True):
    mod_shift, addr_shift = (11, 14) if bh else (12, 15)
    return (
        (MOVB2D << 24)
        | (dest_32b_lo << 23)
        | (src << 17)
        | (addr_mode << addr_shift)
        | (instr_mod << mod_shift)
        | dst
    )


def _movd2a(dst=0, instr_mod=MOVD2A_1_ROW, addr_mode=0, src=0, dest_32b_lo=0, bh=True):
    return (
        (MOVD2A << 24)
        | (dest_32b_lo << 23)
        | (src << 17)
        | (addr_mode << (14 if bh else 15))
        | (instr_mod << 12)
        | dst
    )


def _gmpool(dst=0, clear_dvalid=0, instr_mod19=DIM_16X16, addr_mode=0, argmax=0):
    # The shifts are identical on both architectures; only the width of the
    # addressing-mode field differs (Blackhole 17:15, Wormhole 16:15).
    return (
        (GMPOOL << 24)
        | (clear_dvalid << 22)
        | (instr_mod19 << 19)
        | (addr_mode << 15)
        | (argmax << 14)
        | dst
    )


# --- MOVB2D ------------------------------------------------------------------


def test_movb2d_moves_four_rows_of_srcb_into_dst():
    values = [1.0, 2.0, 3.5, -4.0]
    with _backend(True) as backend:
        srcB = backend.getSrcB(backend.matrix_unit.srcBBank)
        for row, value in enumerate(values):
            for col in range(16):
                srcB[row, col] = _src_bf16(value)
        _issue(backend, _movb2d(instr_mod=MOV_4_ROWS))
        for row, value in enumerate(values):
            for col in range(16):
                assert backend.getDst().getDst16b(row, col) == _dst_bf16(value)
        # The block really is four rows wide
        assert backend.getDst().getDst16b(4, 0) == 0


def test_movb2d_broadcasts_one_row_to_eight():
    with _backend(True) as backend:
        srcB = backend.getSrcB(backend.matrix_unit.srcBBank)
        for col in range(16):
            srcB[1, col] = _src_bf16(2.0)
            srcB[2, col] = _src_bf16(8.0)
        _issue(backend, _movb2d(instr_mod=MOV_8_ROW_BRCST, src=1))
        for row in range(8):
            assert backend.getDst().getDst16b(row, 0) == _dst_bf16(2.0)
        assert backend.getDst().getDst16b(8, 0) == 0


def test_movb2d_broadcasts_column_zero():
    with _backend(True) as backend:
        srcB = backend.getSrcB(backend.matrix_unit.srcBBank)
        srcB[0, 0] = _src_bf16(3.5)
        for col in range(1, 16):
            srcB[0, col] = _src_bf16(1.0)
        _issue(backend, _movb2d(instr_mod=MOV_1_ROW_D0_BRCST))
        for col in range(16):
            assert backend.getDst().getDst16b(0, col) == _dst_bf16(3.5)


def test_movb2d_flushes_denormals_from_srcb():
    # A datum whose Src exponent field is zero is flushed to zero unless
    # ALU_ACC_CTRL_Zero_Flag_disabled_src says otherwise.
    with _backend(True) as backend:
        backend.getSrcB(backend.matrix_unit.srcBBank)[0, 0] = 0x7F << 11
        _issue(backend, _movb2d())
        assert backend.getDst().getDst16b(0, 0) == 0

    with _backend(True) as backend:
        _set_config(backend, "ALU_ACC_CTRL_Zero_Flag_disabled_src", 1)
        backend.getSrcB(backend.matrix_unit.srcBBank)[0, 0] = 0x7F << 11
        _issue(backend, _movb2d())
        assert backend.getDst().getDst16b(0, 0) == 0x7F00


def test_movb2d_dest_32b_lo_only_replaces_the_low_half():
    with _backend(True) as backend:
        backend.getDst().setDst32b(0, 0, 0xAAAA5555)
        backend.getSrcB(backend.matrix_unit.srcBBank)[0, 0] = _src_bf16(3.5)
        _issue(backend, _movb2d(dest_32b_lo=1))
        assert backend.getDst().getDst32b(0, 0) == 0xAAAA0000 | _dst_bf16(3.5)


def test_movb2d_implied_srcb_format_selects_the_tf32_write_on_blackhole():
    # Blackhole implies *both* of MOVB2D's format selects from the format SrcB
    # was last written in; TF32 makes the instruction write a full 32-bit Dst
    # datum instead of a 16-bit one. Wormhole has no such register and keeps
    # reading the configured FP32, so it writes 16 bits.
    # Src bits 10:8 are the extra mantissa only TF32 carries; a BF16/FP32 SrcB
    # format drops them, TF32 keeps them and widens the write to 32 bits.
    src_value = _src_bf16(3.5) | (5 << 8)
    with _backend(True) as backend:
        backend.getSrcB(backend.matrix_unit.srcBBank).setDataFormat(DataFormat.TF32)
        backend.getSrcB(backend.matrix_unit.srcBBank)[0, 0] = src_value
        _issue(backend, _movb2d())
        assert backend.getDst().getDst32b(0, 0) == (_dst_bf16(3.5) << 16) | (5 << 13)

    with _backend(False) as backend:
        backend.getSrcB(backend.matrix_unit.srcBBank).setDataFormat(DataFormat.TF32)
        backend.getSrcB(backend.matrix_unit.srcBBank)[0, 0] = src_value
        _issue(backend, _movb2d(bh=False))
        assert backend.getDst().getDst16b(0, 0) == _dst_bf16(3.5)


def test_movb2d_tf32_with_dest_32b_lo_raises():
    # Undefined behaviour (Blackhole HW erratum TEN-4245); ttsim declines it too.
    with _backend(True) as backend:
        backend.getSrcB(backend.matrix_unit.srcBBank).setDataFormat(DataFormat.TF32)
        with pytest.raises(NotImplementedError):
            _issue(backend, _movb2d(dest_32b_lo=1))


def _movb2d_rows_written(backend, instruction):
    for row in range(16):
        for col in range(16):
            backend.getDst().setDst16b(row, col, 0)
    for row in range(16):
        backend.getSrcB(backend.matrix_unit.srcBBank)[row, 0] = _src_bf16(1.0)
    _issue(backend, instruction)
    return sum(1 for row in range(16) if backend.getDst().getDst16b(row, 0) != 0)


def test_movb2d_instr_mod_is_bits_13_11_on_blackhole():
    # The same word means four rows on Blackhole (13:11 == 0b100) but an
    # eight-row broadcast on Wormhole (14:12 == 0b010).
    instruction = _movb2d(instr_mod=MOV_4_ROWS)
    with _backend(True) as backend:
        assert _movb2d_rows_written(backend, instruction) == 4
    with _backend(False) as backend:
        assert _movb2d_rows_written(backend, instruction) == 8


def test_movb2d_wormhole_instr_mod_stays_at_bits_14_12():
    # And in the other direction: a Wormhole-encoded four-row move still moves
    # four rows on Wormhole, while on Blackhole 13:11 reads as a single row.
    instruction = _movb2d(instr_mod=MOV_4_ROWS, bh=False)
    with _backend(False) as backend:
        assert _movb2d_rows_written(backend, instruction) == 4
    with _backend(True) as backend:
        assert _movb2d_rows_written(backend, instruction) == 1


# --- MOVD2A ------------------------------------------------------------------


def test_movd2a_moves_four_rows_of_dst_into_srca():
    values = [1.0, 2.0, 3.5, -4.0]
    with _backend(True) as backend:
        for row, value in enumerate(values):
            for col in range(16):
                backend.getDst().setDst16b(row, col, _dst_bf16(value))
        _issue(backend, _movd2a(instr_mod=MOVD2A_4_ROWS))
        srcA = backend.getSrcA(backend.matrix_unit.srcABank)
        for row, value in enumerate(values):
            for col in range(16):
                assert srcA[row, col] == _src_bf16(value)


def test_movd2a_moves_a_single_row():
    with _backend(True) as backend:
        for col in range(16):
            backend.getDst().setDst16b(5, col, _dst_bf16(3.5))
        srcA = backend.getSrcA(backend.matrix_unit.srcABank)
        for col in range(16):
            srcA[6, col] = 0
        _issue(backend, _movd2a(dst=5, src=5, instr_mod=MOVD2A_1_ROW))
        assert srcA[5, 0] == _src_bf16(3.5)
        assert srcA[6, 0] == 0


def test_movd2a_reads_dst32b_when_fp32_is_enabled():
    with _backend(True) as backend:
        _set_config(backend, "ALU_ACC_CTRL_Fp32_enabled", 1)
        backend.getDst().setDst32b(0, 0, (_dst_bf16(3.5) << 16) | 0x1234)
        _issue(backend, _movd2a())
        assert backend.getSrcA(backend.matrix_unit.srcABank)[0, 0] == _src_bf16(3.5)


def test_movd2a_undefined_16_bit_modes_raise():
    # Both combinations are documented undefined behaviour and ttsim declines
    # them, so they must fail loudly rather than write a guessed value.
    with _backend(True) as backend:
        with pytest.raises(NotImplementedError):
            _issue(backend, _movd2a(dest_32b_lo=1))

    with _backend(True) as backend:
        backend.getSrcA(backend.matrix_unit.srcABank).setDataFormat(DataFormat.TF32)
        with pytest.raises(NotImplementedError):
            _issue(backend, _movd2a())


def _dst_rwc_after(backend, instruction):
    _set_thread_config(backend, "ADDR_MOD_DST_SEC2_DestIncr", 7)
    _set_thread_config(backend, "ADDR_MOD_DST_SEC5_DestIncr", 3)
    _issue(backend, instruction)
    return backend.getRWC(0).Dst


def test_movd2a_addr_mode_is_bits_16_14_on_blackhole():
    # ADDR_MOD_5 encoded Blackhole-style selects section 5 there, but Wormhole
    # reads bits 16:15 of the same word and lands on section 2.
    instruction = _movd2a(addr_mode=5)
    with _backend(True) as backend:
        assert _dst_rwc_after(backend, instruction) == 3
    with _backend(False) as backend:
        assert _dst_rwc_after(backend, instruction) == 7


# --- GMPOOL ------------------------------------------------------------------


def _load_gmpool_operands(backend, column_values, scale=1.0, dst_seed=None):
    srcA = backend.getSrcA(backend.matrix_unit.srcABank)
    srcB = backend.getSrcB(backend.matrix_unit.srcBBank)
    for row in range(16):
        for col in range(16):
            srcA[row, col] = _src_bf16(column_values[row])
        srcB[0, row] = _src_bf16(scale)
    if dst_seed is not None:
        for col in range(16):
            backend.getDst().setDst16b(0, col, _dst_bf16(dst_seed))


def test_gmpool_reduces_each_srca_column_by_max():
    values = [1.0, 2.0, 3.5, 0.5] * 4
    with _backend(True) as backend:
        _load_gmpool_operands(backend, values)
        _issue(backend, _gmpool())
        for col in range(16):
            assert backend.getDst().getDst16b(0, col) == _dst_bf16(3.5)


def test_gmpool_zeroes_the_rest_of_the_four_row_block():
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [2.0] * 16)
        for row in range(1, 4):
            for col in range(16):
                backend.getDst().setDst16b(row, col, 0x1234)
        _issue(backend, _gmpool())
        for row in range(1, 4):
            for col in range(16):
                assert backend.getDst().getDst16b(row, col) == 0


def test_gmpool_maxes_against_the_existing_dst_row():
    # The seed in Dst takes part in the reduction, which is why software issues
    # ZEROACC first when it wants a plain max over SrcA.
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [1.0] * 16, dst_seed=8.0)
        _issue(backend, _gmpool())
        assert backend.getDst().getDst16b(0, 0) == _dst_bf16(8.0)

    with _backend(True) as backend:
        _load_gmpool_operands(backend, [16.0] * 16, dst_seed=8.0)
        _issue(backend, _gmpool())
        assert backend.getDst().getDst16b(0, 0) == _dst_bf16(16.0)


def test_gmpool_scales_srca_by_the_srcb_exponent():
    # SrcB supplies a per-column scale factor taken from its exponent alone, so
    # a row of 2.0 doubles every SrcA datum.
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [1.0] * 16, scale=2.0)
        _issue(backend, _gmpool())
        assert backend.getDst().getDst16b(0, 0) == _dst_bf16(2.0)


def test_gmpool_zero_srcb_exponent_is_minus_infinity():
    # An exponent of zero scales by zero, which the hardware models as minus
    # infinity, so the Dst seed always wins.
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [16.0] * 16, scale=0.0, dst_seed=1.0)
        _issue(backend, _gmpool())
        assert backend.getDst().getDst16b(0, 0) == _dst_bf16(1.0)


def test_gmpool_flushes_srca_denormals():
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [1.0] * 16)
        srcA = backend.getSrcA(backend.matrix_unit.srcABank)
        for row in range(16):
            srcA[row, 0] = 0x7F << 11  # mantissa bits only, exponent field zero
        _issue(backend, _gmpool())
        assert backend.getDst().getDst16b(0, 0) == 0


def test_gmpool_reads_a_flag_cleared_dst_row_as_minus_infinity():
    # ZEROACC clears a row's zero flag rather than its data, and GMPOOL is the
    # one Dst consumer that reads a cleared row as all-ones, i.e. minus
    # infinity. That is what makes `reduce_tile(MAX)` over negative data work:
    # read the row as its (zeroed) data instead and every result saturates at 0.
    for blackhole in (True, False):
        with _backend(blackhole) as backend:
            _load_gmpool_operands(backend, [-1.0, -2.0, -3.5, -0.5] * 4)
            backend.getDst().setUndefinedRow(0)
            _issue(backend, _gmpool())
            for col in range(16):
                assert backend.getDst().getDst16b(0, col) == _dst_bf16(-0.5)


def test_gmpool_reads_a_valid_zero_dst_row_as_zero():
    # The flag, not the data, is what makes the row lose: an explicitly written
    # zero is a real 0.0 seed and clamps a max over negative SrcA at zero.
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [-1.0] * 16, dst_seed=0.0)
        _issue(backend, _gmpool())
        assert backend.getDst().getDst16b(0, 0) == 0


def test_gmpool_holds_the_flag_until_the_last_column():
    # The accumulating row's flag is re-asserted only once column 15 lands, so
    # every column of one pass reduces against minus infinity rather than
    # against the max column 0 just wrote. Column 0 gets the largest datum, so
    # a flag re-asserted early would leak -0.5 into columns 1-15.
    with _backend(True) as backend:
        srcA = backend.getSrcA(backend.matrix_unit.srcABank)
        srcB = backend.getSrcB(backend.matrix_unit.srcBBank)
        for row in range(16):
            for col in range(16):
                srcA[row, col] = _src_bf16(-0.5 if col == 0 else -4.0)
            srcB[0, row] = _src_bf16(1.0)
        backend.getDst().setUndefinedRow(0)
        _issue(backend, _gmpool())
        assert backend.getDst().getDst16b(0, 0) == _dst_bf16(-0.5)
        for col in range(1, 16):
            assert backend.getDst().getDst16b(0, col) == _dst_bf16(-4.0)


def test_gmpool_clears_dvalid_and_flips_banks():
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [1.0] * 16)
        _issue(backend, _gmpool(clear_dvalid=0x3))
        assert backend.matrix_unit.srcABank == 1
        assert backend.matrix_unit.srcBBank == 1


def test_gmpool_result_is_identical_on_wormhole():
    # Nothing in the reduction itself is architecture-specific: the same word
    # over the same operands must produce the same Dst on both.
    values = [1.0, 2.0, 3.5, 0.5] * 4
    results = []
    for blackhole in (True, False):
        with _backend(blackhole) as backend:
            _load_gmpool_operands(backend, values)
            _issue(backend, _gmpool())
            results.append(backend.getDst().getDst16b(0, 0))
    assert results[0] == results[1] == _dst_bf16(3.5)


def test_gmpool_addr_mode_is_bits_17_15_on_blackhole():
    # Blackhole reads three bits (so this word selects ADDR_MOD_5) while the
    # shared Wormhole-layout table reads four and lands on a section that does
    # not exist, leaving the RWC alone.
    instruction = _gmpool(addr_mode=0xD)
    with _backend(True) as backend:
        _load_gmpool_operands(backend, [1.0] * 16)
        assert _dst_rwc_after(backend, instruction) == 3
    with _backend(False) as backend:
        _load_gmpool_operands(backend, [1.0] * 16)
        assert _dst_rwc_after(backend, instruction) == 0


def test_gmpool_rejects_unmodelled_modes():
    # ttsim declines both, and neither is emitted by tt-metal's reduce LLK.
    for instruction in (_gmpool(argmax=1), _gmpool(instr_mod19=0)):
        with _backend(True) as backend:
            with pytest.raises(NotImplementedError):
                _issue(backend, instruction)

    with _backend(True) as backend:
        _set_config(backend, "ALU_ACC_CTRL_INT8_math_enabled", 1)
        with pytest.raises(NotImplementedError):
            _issue(backend, _gmpool())


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print(
        f"matrix_mov_pool_test OK: all {len(tests)} MOVB2D/MOVD2A/GMPOOL tests passed"
    )


if __name__ == "__main__":
    main()
