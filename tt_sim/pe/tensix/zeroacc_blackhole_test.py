"""Tests for Blackhole's ZEROACC field re-layout and 16-row math bank offset.

Runs standalone (``python3 -m tt_sim.pe.tensix.zeroacc_blackhole_test``) or under
pytest. The shared ``tensix_instructions.yaml`` encodes the Wormhole bit
positions, and ZEROACC is the instruction Blackhole moves most: ``addr_mode``
widens down into bit 14, ``use_32_bit_mode`` becomes its own field at bit 18
instead of riding along as ``clear_mode`` bit 2, and ``clear_mode`` narrows to
20:19. Bit ranges are from ttsim's ``data/{bh,wh}/tensix_isa.json``; the
fold-back into a Wormhole-shaped ``clear_mode`` and the 16-row bank offset are
ttsim's ``TENSIX_EXECUTE_ZEROACC`` under ``TT_ARCH_VERSION == 1``.

Every Blackhole case is paired with the same instruction word on a Wormhole unit
so the gating is checked in both directions.
"""

from contextlib import contextmanager

from tt_sim.arch.blackhole import BLACKHOLE_PROFILE
from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import (
    TensixConfigurationConstants,
    TensixInstructionDecoder,
)

ZEROACC = 0x10

ZEROACC_MODE_ONE_ROW = 0
ZEROACC_MODE_16_ROWS = 1
ZEROACC_MODE_HALF_OF_DST = 2
ZEROACC_MODE_ALL_OF_DST = 3


@contextmanager
def _blackhole_matrix():
    """A Blackhole-configured matrix unit.

    The config-register layout is a process-global selection, so restore the
    Wormhole layout on the way out — other tests in the same session (and the
    Wormhole replay guards) expect it.
    """
    try:
        yield (
            TensixCoProcessor(
                None,
                BLACKHOLE_PROFILE.tensix_cfg_state_size,
                BLACKHOLE_PROFILE.tensix_thd_state_size,
                blackhole=True,
            )
            .getBackend()
            .matrix_unit
        )
    finally:
        TensixConfigurationConstants.use_blackhole(False)


@contextmanager
def _wormhole_matrix():
    yield TensixCoProcessor(None).getBackend().matrix_unit


def _decode(instruction):
    info = TensixInstructionDecoder.getInstructionInfo(instruction)
    return info, info["instr_args"]


def _zeroacc_bh(where=0, addr_mode=0, clear_zero_flags=0, use_32_bit=0, clear_mode=0):
    """Assemble a ZEROACC using Blackhole's field layout."""
    return (
        (ZEROACC << 24)
        | (clear_mode << 19)
        | (use_32_bit << 18)
        | (clear_zero_flags << 17)
        | (addr_mode << 14)
        | where
    )


def _zeroacc_wh(dst=0, addr_mode=0, clear_mode=0):
    """Assemble a ZEROACC using Wormhole's field layout."""
    return (ZEROACC << 24) | (clear_mode << 19) | (addr_mode << 15) | dst


def _read(matrix, instruction):
    info, args = _decode(instruction)
    return matrix._read_zeroacc_fields(info, args)


def _set_thread_config(backend, thread, key, value):
    idx = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    unit = backend.getConfigUnit()
    current = unit.threadConfig[thread][idx] & ~mask
    unit.setThreadConfig(thread, idx, current | ((value << shamt) & mask))


def _set_config(backend, state_id, key, value):
    idx = TensixConfigurationConstants.get_addr32(key)
    shamt = TensixConfigurationConstants.get_shamt(key)
    mask = TensixConfigurationConstants.get_mask(key)
    unit = backend.getConfigUnit()
    current = unit.get_config_entry(state_id, idx) & ~mask
    unit.setConfig(state_id, idx, current | ((value << shamt) & mask))


def test_blackhole_addr_mode_is_bits_16_14():
    # Blackhole has 8 ADDR_MOD sections, so addr_mode is 3 bits (16:14); the
    # shared table starts the field at 15 and would drop the low bit.
    with _blackhole_matrix() as matrix:
        for addr_mode in range(8):
            _, _, decoded = _read(matrix, _zeroacc_bh(addr_mode=addr_mode))
            assert decoded == addr_mode


def test_blackhole_use_32_bit_mode_folds_in_as_clear_mode_bit_2():
    # ttsim: `clear_mode |= use_32_bit_mode << 2` — remapped so the shared
    # Wormhole-shaped mode/useDst32b split below works unchanged.
    with _blackhole_matrix() as matrix:
        for clear_mode in range(4):
            for use_32_bit in range(2):
                _, decoded, _ = _read(
                    matrix,
                    _zeroacc_bh(clear_mode=clear_mode, use_32_bit=use_32_bit),
                )
                assert decoded & 0x7 == clear_mode | (use_32_bit << 2)


def test_blackhole_ignores_wormholes_use_32_bit_position():
    # Raw bit 21 carried useDst32b on Wormhole but is unassigned on Blackhole,
    # so reading it there invents a 32-bit clear out of nothing.
    with _blackhole_matrix() as matrix:
        _, decoded, _ = _read(matrix, _zeroacc_bh(clear_mode=1) | (1 << 21))
        assert decoded & 0x7 == 1


def test_blackhole_where_is_10_bits():
    # Blackhole's `where` is 9:0, but the shared table runs `dst` up to bit 14
    # (the next field starts at 15) so addr_mode's new low bit leaks in.
    with _blackhole_matrix() as matrix:
        decoded_dst, _, _ = _read(matrix, _zeroacc_bh(where=0x2A, addr_mode=7))
        assert decoded_dst == 0x2A


def test_wormhole_zeroacc_fields_unchanged():
    with _wormhole_matrix() as matrix:
        instruction = _zeroacc_wh(dst=0x2A, addr_mode=3, clear_mode=5)
        info, args = _decode(instruction)
        decoded_dst, clear_mode, addr_mode = matrix._read_zeroacc_fields(info, args)
        assert (decoded_dst, addr_mode) == (args["dst"], 3)
        assert clear_mode == args["clear_mode"]
        assert clear_mode & 0x3 == 1
        assert (clear_mode >> 2) & 1 == 1


def test_copy_tile_workaround_zeroacc_decodes_as_ttsim_expects():
    # 0x100EC000 is the `clear_zero_flags` ZEROACC that Blackhole's `copy_tile`
    # emits after unpack-to-dest (seen in the five/optest/loopback traces).
    # ttsim asserts clear_mode == 5 whenever clear_zero_flags is set, which only
    # holds once use_32_bit_mode is folded back in; the Wormhole decode gives
    # mode 1 / addr_mode 1 instead of the ADDR_MOD_3 the LLK asks for.
    with _blackhole_matrix() as matrix:
        decoded_dst, clear_mode, addr_mode = _read(matrix, 0x100EC000)
        assert clear_mode == 5
        assert addr_mode == 3
        assert decoded_dst == 0


def test_clear_zero_flags_applies_the_decoded_addr_mod():
    # The early return still advances the RWC, so the wrong ADDR_MOD is applied
    # on a path that really runs. ADDR_MOD_1 and ADDR_MOD_3 are given different
    # Dest increments so the applied section is observable.
    with _blackhole_matrix() as matrix:
        backend = matrix.backend
        _set_thread_config(backend, 0, "ADDR_MOD_DST_SEC1_DestIncr", 1)
        _set_thread_config(backend, 0, "ADDR_MOD_DST_SEC3_DestIncr", 9)
        info, args = _decode(0x100EC000)
        matrix.handle_zeroacc(info, 0, args)
        assert backend.getRWC(0).Dst == 9


def test_16_row_clear_follows_the_math_bank_on_blackhole():
    # ttsim case 1: a math offset into DEST's high half moves the cleared block
    # up by 32 (512 rows) unless zeroacc_absolute_tile_mode is set.
    with _blackhole_matrix() as matrix:
        backend = matrix.backend
        _set_thread_config(backend, 0, "DEST_TARGET_REG_CFG_MATH_Offset", 512)
        dst = backend.getDst()
        dst.dstBits[:, :] = 0xFFFF
        info, args = _decode(_zeroacc_bh(where=1, clear_mode=ZEROACC_MODE_16_ROWS))
        matrix.handle_zeroacc(info, 0, args)
        assert (dst.dstBits[16:32, :] == 0xFFFF).all()
        assert (dst.dstBits[528:544, :] == 0).all()


def test_16_row_bank_offset_suppressed_by_absolute_tile_mode():
    with _blackhole_matrix() as matrix:
        backend = matrix.backend
        _set_thread_config(backend, 0, "DEST_TARGET_REG_CFG_MATH_Offset", 512)
        _set_config(backend, 0, "DEST_ACCESS_CFG_zeroacc_absolute_tile_mode", 1)
        dst = backend.getDst()
        dst.dstBits[:, :] = 0xFFFF
        info, args = _decode(_zeroacc_bh(where=1, clear_mode=ZEROACC_MODE_16_ROWS))
        matrix.handle_zeroacc(info, 0, args)
        assert (dst.dstBits[16:32, :] == 0).all()
        assert (dst.dstBits[528:544, :] == 0xFFFF).all()


def test_16_row_bank_offset_needs_bit_9_of_the_offset():
    # Only bit 9 of (DEST_TARGET_REG_CFG_MATH_Offset + RWC.Dst) selects the
    # bank, so a low offset must not shift the clear.
    with _blackhole_matrix() as matrix:
        backend = matrix.backend
        _set_thread_config(backend, 0, "DEST_TARGET_REG_CFG_MATH_Offset", 256)
        dst = backend.getDst()
        dst.dstBits[:, :] = 0xFFFF
        info, args = _decode(_zeroacc_bh(where=1, clear_mode=ZEROACC_MODE_16_ROWS))
        matrix.handle_zeroacc(info, 0, args)
        assert (dst.dstBits[16:32, :] == 0).all()
        assert (dst.dstBits[528:544, :] == 0xFFFF).all()


def test_16_row_bank_offset_not_applied_on_wormhole():
    # Wormhole has no zeroacc_absolute_tile_mode and never rebases the clear.
    with _wormhole_matrix() as matrix:
        backend = matrix.backend
        _set_thread_config(backend, 0, "DEST_TARGET_REG_CFG_MATH_Offset", 512)
        dst = backend.getDst()
        dst.dstBits[:, :] = 0xFFFF
        info, args = _decode(_zeroacc_wh(dst=1, clear_mode=ZEROACC_MODE_16_ROWS))
        matrix.handle_zeroacc(info, 0, args)
        assert (dst.dstBits[16:32, :] == 0).all()
        assert (dst.dstBits[528:544, :] == 0xFFFF).all()


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "zeroacc_blackhole_test OK: Blackhole ZEROACC addr_mode/use_32_bit_mode/"
        "clear_mode/where decodes and the 16-row math bank offset verified"
    )


if __name__ == "__main__":
    main()
