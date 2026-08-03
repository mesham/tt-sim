"""Tests for the Blackhole SFPU superset ops (SFPGT / SFPLE / SFPMUL24 / SFPARECIP).

Runs standalone (``python3 -m tt_sim.pe.tensix.sfpu_blackhole_test``) or under
pytest. Drives a real ``VectorUnit`` (via a coprocessor) so the handlers are
exercised exactly as a decoded instruction would reach them; also checks the
decoder recognises the new opcodes. SFPARECIP's LUT-based reciprocal is a
verbatim port of ttsim's data/bh reference, checked against exact bit patterns.
"""

from tt_sim.pe.tensix.tensix import TensixCoProcessor
from tt_sim.pe.tensix.util import TensixInstructionDecoder


def _vector_unit():
    return TensixCoProcessor(None).getBackend().vector_unit


def _args(**kw):
    base = {"instr_mod1": 0, "lreg_dest": 0, "lreg_c": 1, "imm12_math": 0}
    base.update(kw)
    return base


def test_decoder_recognises_blackhole_sfpu_ops():
    for opcode, name in [
        (0x96, "SFPLE"),
        (0x97, "SFPGT"),
        (0x98, "SFPMUL24"),
        (0x99, "SFPARECIP"),
    ]:
        instruction = opcode << 24
        assert TensixInstructionDecoder.isInstructionRecognised(instruction)
        assert TensixInstructionDecoder.getInstructionInfo(instruction)["name"] == name


def test_sfpgt_sets_lane_flag_when_vd_gt_vc():
    vu = _vector_unit()
    vu.lregs[0][0], vu.lregs[1][0] = 3.0, 2.0  # 3 > 2 -> True
    vu.lregs[0][1], vu.lregs[1][1] = 1.0, 5.0  # 1 > 5 -> False
    vu.handle_sfpgt(None, 0, _args(instr_mod1=1, lreg_dest=0, lreg_c=1))
    assert vu.laneFlags[0] is True
    assert vu.laneFlags[1] is False


def test_sfple_sets_lane_flag_when_vd_le_vc():
    vu = _vector_unit()
    vu.lregs[0][0], vu.lregs[1][0] = 2.0, 2.0  # 2 <= 2 -> True
    vu.lregs[0][1], vu.lregs[1][1] = 3.0, 2.0  # 3 <= 2 -> False
    vu.handle_sfple(None, 0, _args(instr_mod1=1, lreg_dest=0, lreg_c=1))
    assert vu.laneFlags[0] is True
    assert vu.laneFlags[1] is False


def test_sfpgt_mod1_8_writes_mask_into_vd():
    # instr_mod1 == 8 writes an all-ones / all-zero mask into VD and leaves the
    # lane flags alone. This is the form every stock Blackhole exp_tile emits
    # (masking the integer part before SFPSETEXP).
    vu = _vector_unit()
    vu.lregs[0][0], vu.lregs[1][0] = 127.0, 0.0  # 127 > 0 -> all ones
    vu.lregs[0][1], vu.lregs[1][1] = -1.0, 0.0  # -1 > 0 -> zero
    vu.laneFlags[0] = vu.laneFlags[1] = False
    vu.handle_sfpgt(None, 0, _args(instr_mod1=8, lreg_dest=0, lreg_c=1))
    assert vu.lregs[0][0] == 0xFFFFFFFF
    assert vu.lregs[0][1] == 0
    assert vu.laneFlags[0] is False
    assert vu.laneFlags[1] is False


def test_sfple_mod1_8_writes_mask_into_vd():
    vu = _vector_unit()
    vu.lregs[0][0], vu.lregs[1][0] = 2.0, 2.0  # 2 <= 2 -> all ones
    vu.lregs[0][1], vu.lregs[1][1] = 3.0, 2.0  # 3 <= 2 -> zero
    vu.handle_sfple(None, 0, _args(instr_mod1=8, lreg_dest=0, lreg_c=1))
    assert vu.lregs[0][0] == 0xFFFFFFFF
    assert vu.lregs[0][1] == 0


def test_sfpgt_orders_lanes_as_sign_magnitude():
    # Sign-magnitude total order, not IEEE compare: -0.0 sorts below +0.0 and
    # a more negative magnitude sorts lower (ttsim's sign_mag32_total_order).
    vu = _vector_unit()
    vu.lregs[0][0], vu.lregs[1][0] = 0.0, -0.0  # +0 > -0 -> True
    vu.lregs[0][1], vu.lregs[1][1] = -2.0, -5.0  # -2 > -5 -> True
    vu.lregs[0][2], vu.lregs[1][2] = -5.0, -2.0  # -5 > -2 -> False
    vu.handle_sfpgt(None, 0, _args(instr_mod1=8, lreg_dest=0, lreg_c=1))
    assert vu.lregs[0][0] == 0xFFFFFFFF
    assert vu.lregs[0][1] == 0xFFFFFFFF
    assert vu.lregs[0][2] == 0


def test_sfpgt_rejects_unmodelled_modifier():
    vu = _vector_unit()
    try:
        vu.handle_sfpgt(None, 0, _args(instr_mod1=0, lreg_dest=0, lreg_c=1))
    except NotImplementedError:
        return
    raise AssertionError("SFPGT with instr_mod1=0 should raise")


def test_sfpmul24_low_23_bit_product():
    vu = _vector_unit()
    vu.lregs[2][0], vu.lregs[3][0] = 100, 3  # 300, fits in 23 bits
    vu.lregs[2][1], vu.lregs[3][1] = 0x7FFFFF, 2  # overflows 23 bits -> masked
    vu.handle_sfpmul24(None, 0, _args(lreg_dest=0, lreg_src_a=2, lreg_src_b=3))
    assert vu.lregs[0][0] == 300
    assert vu.lregs[0][1] == ((0x7FFFFF * 2) & 0x7FFFFF)


def test_sfparecip_approx_reciprocal():
    # Exact bit patterns from the ttsim data/bh reference (approx_recip): sign
    # preserved, magnitude replaced by the LUT 1/x approximation.
    vu = _vector_unit()
    vu.lregs[1][0] = 1.0  # 1/1 ~= 0.996  -> 0x3F7F0000
    vu.lregs[1][1] = 2.0  # 1/2 ~= 0.499  -> 0x3EFF0000
    vu.lregs[1][2] = -1.0  # sign preserved -> 0xBF7F0000
    vu.lregs[1][3] = 0.0  # 1/0 -> +inf    -> 0x7F800000
    vu.handle_sfparecip(None, 0, _args(lreg_dest=0, lreg_c=1))
    assert vu.lregs[0][0] == 0x3F7F0000
    assert vu.lregs[0][1] == 0x3EFF0000
    assert vu.lregs[0][2] == 0xBF7F0000
    assert vu.lregs[0][3] == 0x7F800000


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("sfpu_blackhole_test OK: SFPGT/SFPLE/SFPMUL24/SFPARECIP verified")


if __name__ == "__main__":
    main()
