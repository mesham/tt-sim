"""Tests for the Blackhole SFPU superset ops (SFPGT / SFPLE / SFPMUL24).

Runs standalone (``python3 -m tt_sim.pe.tensix.sfpu_blackhole_test``) or under
pytest. Drives a real ``VectorUnit`` (via a coprocessor) so the handlers are
exercised exactly as a decoded instruction would reach them; also checks the
decoder recognises the new opcodes. SFPARECIP is intentionally not implemented
(hardware approximation) and must fail loudly.
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
    vu.handle_sfpgt(None, 0, _args(lreg_dest=0, lreg_c=1))
    assert vu.laneFlags[0] is True
    assert vu.laneFlags[1] is False


def test_sfple_sets_lane_flag_when_vd_le_vc():
    vu = _vector_unit()
    vu.lregs[0][0], vu.lregs[1][0] = 2.0, 2.0  # 2 <= 2 -> True
    vu.lregs[0][1], vu.lregs[1][1] = 3.0, 2.0  # 3 <= 2 -> False
    vu.handle_sfple(None, 0, _args(lreg_dest=0, lreg_c=1))
    assert vu.laneFlags[0] is True
    assert vu.laneFlags[1] is False


def test_sfpmul24_low_23_bit_product():
    vu = _vector_unit()
    vu.lregs[2][0], vu.lregs[3][0] = 100, 3  # 300, fits in 23 bits
    vu.lregs[2][1], vu.lregs[3][1] = 0x7FFFFF, 2  # overflows 23 bits -> masked
    vu.handle_sfpmul24(None, 0, _args(lreg_dest=0, lreg_src_a=2, lreg_src_b=3))
    assert vu.lregs[0][0] == 300
    assert vu.lregs[0][1] == ((0x7FFFFF * 2) & 0x7FFFFF)


def test_sfparecip_fails_loudly():
    vu = _vector_unit()
    try:
        vu.handle_sfparecip(None, 0, _args())
    except NotImplementedError:
        return
    raise AssertionError("SFPARECIP should raise NotImplementedError")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(
        "sfpu_blackhole_test OK: SFPGT/SFPLE/SFPMUL24 verified, SFPARECIP fails loudly"
    )


if __name__ == "__main__":
    main()
