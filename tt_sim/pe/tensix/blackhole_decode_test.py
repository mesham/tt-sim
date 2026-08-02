"""Tests for the Blackhole shifted-field instruction decodes.

Runs standalone (``python3 -m tt_sim.pe.tensix.blackhole_decode_test``) or under
pytest. The shared ``tensix_instructions.yaml`` encodes the Wormhole bit
positions, so any field Blackhole moved has to be re-read from the raw 32-bit
word by the handler. These tests pin the raw-bit readers for STALLWAIT's
``wait_res`` and GAPOOL/GMPOOL's ``pool_addr_mode``, check the generic math
``addr_mode`` reader still behaves, and assert Wormhole decodes are untouched.
Bit ranges are from ttsim's ``data/{bh,wh}/tensix_isa.json``.
"""

from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixInstructionDecoder

STALLWAIT = 0xA2
GAPOOL = 0x34
MVMUL = 0x26
MOP = 0x1


def _backend(blackhole):
    return TensixCoProcessor(blackhole=blackhole).getBackend()


def _decode(instruction):
    info = TensixInstructionDecoder.getInstructionInfo(instruction)
    return info, info["instr_args"]


def _stallwait(wait_res, stall_res=1):
    return (STALLWAIT << 24) | (stall_res << 15) | wait_res


def _gapool(pool_addr_mode, max_pool_index_en=0, dst=0):
    # Blackhole layout: dst 9:0, max_pool_index_en 14:14, pool_addr_mode 17:15
    return (GAPOOL << 24) | (pool_addr_mode << 15) | (max_pool_index_en << 14) | dst


def _stallwait_cond_mask(backend, instruction):
    info, args = _decode(instruction)
    backend.sync_unit.handle_stallwait(info, 0, args)
    return backend.getFrontendThread(0).wait_gate.latchedWaitInstruction.condition_mask


def test_stallwait_wait_res_is_12_bits_on_blackhole():
    # Blackhole's wait_res is raw 11:0 (Wormhole 14:0), so anything the shared
    # (Wormhole-width) table pulls in from bits 14:12 must not reach the
    # condition mask. 0x400 is p_stall::TRISC_CFG as the Blackhole LLK emits it.
    instruction = _stallwait(0x400 | (1 << 13))
    assert _stallwait_cond_mask(_backend(True), instruction) == 0x400


def test_stallwait_wait_res_unchanged_on_wormhole():
    instruction = _stallwait(0x400 | (1 << 13))
    _, args = _decode(instruction)
    assert args["wait_res"] == 0x2400
    assert _stallwait_cond_mask(_backend(False), instruction) == 0x2400


def test_stallwait_empty_mask_still_waits_on_everything():
    # An all-zero condition mask means "wait for all resources" (0x7F) on both
    # architectures; on Blackhole that now includes a mask left empty by the
    # 12-bit trim.
    assert _stallwait_cond_mask(_backend(True), _stallwait(1 << 12)) == 0x7F
    assert _stallwait_cond_mask(_backend(False), _stallwait(0)) == 0x7F


def test_gapool_pool_addr_mode_is_bits_17_15_on_blackhole():
    # Blackhole renames GAPOOL/GMPOOL's addr_mode to pool_addr_mode and widens
    # it to 17:15 (Wormhole 16:15) - unlike every other math instruction, which
    # moves to 16:14, so bit 14 here is max_pool_index_en and must not leak in.
    matrix = _backend(True).matrix_unit
    for addr_mod in range(8):
        info, args = _decode(_gapool(addr_mod, max_pool_index_en=1))
        assert matrix._read_pool_addr_mode(info, args) == addr_mod


def test_gapool_generic_addr_mode_reader_would_be_wrong_on_blackhole():
    # Guards the reason _read_pool_addr_mode exists: the generic reader takes
    # bits 16:14, which for GAPOOL is a shifted-in max_pool_index_en.
    matrix = _backend(True).matrix_unit
    info, args = _decode(_gapool(5, max_pool_index_en=1))
    assert matrix._read_addr_mode(info, args) == ((5 << 1) | 1) & 0x7


def test_gapool_pool_addr_mode_unchanged_on_wormhole():
    matrix = _backend(False).matrix_unit
    info, args = _decode(_gapool(3))
    assert matrix._read_pool_addr_mode(info, args) == args["addr_mode"]


def test_math_addr_mode_reader_unaffected():
    # MVMUL keeps the common math layout: raw 16:14 on Blackhole, the decoded
    # (16:15) argument on Wormhole.
    instruction = (MVMUL << 24) | (5 << 14)
    info, args = _decode(instruction)
    assert _backend(True).matrix_unit._read_addr_mode(info, args) == 5
    assert _backend(False).matrix_unit._read_addr_mode(info, args) == args["addr_mode"]


def test_mop_fields_are_architecture_independent():
    # MOP's zmask/loop_count fields sit at the same bits on both architectures
    # (Blackhole only renames zmask_lo16), so the MOP expander needs no raw-bit
    # read - this pins that.
    instruction = (MOP << 24) | (1 << 23) | (0x2A << 16) | 0xBEEF
    _, args = _decode(instruction)
    assert args["zmask_lo16"] == 0xBEEF
    assert args["loop_count"] == 0x2A
    assert args["mop_type"] == 1


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("blackhole_decode_test OK: STALLWAIT/GAPOOL/MVMUL/MOP decodes verified")


if __name__ == "__main__":
    main()
